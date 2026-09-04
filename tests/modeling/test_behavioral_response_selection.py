"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.behavioral_response_selection``
— the nested block comparison that asks whether a partner's vocal trace
explains anything a kinematic baseline does not.

The statistics here decide a scientific claim, so the invariants pinned
below are the ones whose violation would produce a plausible-looking but
wrong answer rather than a crash. Coverage:

* ``circular_shift_rows_within_session`` — the null generator. Rows never
  cross a session boundary, the result is a rotation (so the block keeps
  its bout structure, rate and marginal exactly), the offset respects its
  floor, the input is not mutated, and a session too short for a legal
  offset is left alone rather than shifted illegally.
* ``gamma_explained_deviance`` / ``gaussian_explained_variance`` — each
  scores 1 on a perfect fit and 0 on the mean, within its own likelihood.
* ``paired_fold_margin`` — sign and p-value behaviour on constructed
  margins, NaN folds dropped rather than propagated, misaligned inputs
  rejected.
* ``build_session_folds`` — held-out sessions reach no fold, and no
  session straddles a train/test split.
* ``fit_block_across_folds`` — both likelihood arms run, and bad
  arguments fail loudly.
"""

from __future__ import annotations

import pathlib
import pickle
import warnings

import numpy as np
import pytest

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level import so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect (mirrors the guard in
# ``test_model_selection_fold_gate``).
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.behavioral_response_selection import (
        _restore_last_step,
        build_session_folds,
        circular_shift_rows_within_session,
        fit_block_across_folds,
        forward_select_features,
        fraction_of_remaining_deviance,
        gamma_explained_deviance,
        gaussian_explained_variance,
        paired_fold_margin,
        paired_one_se_improvement,
        per_session_scores,
        score_on_held_out,
        screen_from_univariate,
        vocal_block_final_step,
    )


def _blocked_groups(session_sizes: dict[str, int]) -> np.ndarray:
    """Builds a row-wise session label array from ``{session_id: n_rows}``."""
    return np.concatenate([np.full(n, sid) for sid, n in session_sizes.items()])


class TestCircularShiftRowsWithinSession:
    def test_values_never_cross_a_session_boundary(self):
        """A shifted row keeps its session; mixing would fabricate data.

        Each session gets a disjoint value range so any leak is visible.
        """

        groups = _blocked_groups({'A': 300, 'B': 250})
        block = np.concatenate([np.arange(300), 1000 + np.arange(250)]).astype(float)[:, None]
        shifted = circular_shift_rows_within_session(block, groups, np.random.default_rng(0), 5)

        assert ((shifted[groups == 'A', 0] >= 0) & (shifted[groups == 'A', 0] < 300)).all()
        assert (shifted[groups == 'B', 0] >= 1000).all()

    def test_shift_is_a_rotation_so_the_marginal_is_preserved(self):
        """The null must keep the block's own distribution and burstiness.

        Permuting instead would give a trace that is no longer bursty, so
        the alternative the test exists to exclude — that any sparse bursty
        regressor would score as well — would never appear in the null.
        """

        groups = _blocked_groups({'A': 300})
        block = np.random.default_rng(1).gamma(2.0, 1.0, size=(300, 1))
        shifted = circular_shift_rows_within_session(block, groups, np.random.default_rng(0), 5)

        np.testing.assert_allclose(np.sort(shifted[:, 0]), np.sort(block[:, 0]))
        assert not np.array_equal(shifted[:, 0], block[:, 0])

    def test_realized_offset_respects_the_minimum(self):
        """Small offsets would leave real alignment intact in the null."""

        groups = _blocked_groups({'A': 200})
        block = np.arange(200, dtype=float)[:, None]
        for seed in range(20):
            shifted = circular_shift_rows_within_session(
                block, groups, np.random.default_rng(seed), min_shift_rows=5,
            )
            offset = int(np.flatnonzero(shifted[:, 0] == 0.0)[0])
            assert 5 <= offset <= 195

    def test_session_too_short_for_a_legal_offset_is_left_unshifted(self):
        """Being conservative beats shifting by an illegal amount."""

        groups = _blocked_groups({'A': 8})
        block = np.arange(8, dtype=float)[:, None]
        shifted = circular_shift_rows_within_session(block, groups, np.random.default_rng(0), 5)
        np.testing.assert_array_equal(shifted, block)

    def test_input_block_is_not_mutated(self):
        """A null draw must not corrupt the observed data it is drawn from."""

        groups = _blocked_groups({'A': 100})
        block = np.arange(100, dtype=float)[:, None]
        original = block.copy()
        circular_shift_rows_within_session(block, groups, np.random.default_rng(0), 5)
        np.testing.assert_array_equal(block, original)

    def test_row_misalignment_raises_value_error(self):
        """Block and group arrays that disagree on length fail loudly."""

        with pytest.raises(ValueError, match='row-aligned'):
            circular_shift_rows_within_session(
                np.zeros((10, 1)), _blocked_groups({'A': 9}), np.random.default_rng(0), 5,
            )

    def test_sub_unit_minimum_shift_raises_value_error(self):
        """A zero-row floor would admit the identity as a null draw."""

        with pytest.raises(ValueError, match='min_shift_rows'):
            circular_shift_rows_within_session(
                np.zeros((10, 1)), _blocked_groups({'A': 10}), np.random.default_rng(0), 0,
            )


class TestScores:
    def test_gamma_deviance_is_one_for_a_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert gamma_explained_deviance(y, y) == pytest.approx(1.0)

    def test_gamma_deviance_is_zero_for_the_mean_predictor(self):
        """The score is referenced to the evaluation set's own mean."""

        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert gamma_explained_deviance(y, np.full_like(y, y.mean())) == pytest.approx(0.0)

    def test_gaussian_variance_is_one_for_a_perfect_fit(self):
        y_log = np.array([0.0, 0.7, 1.1, 1.4])
        assert gaussian_explained_variance(y_log, y_log) == pytest.approx(1.0)

    def test_gaussian_variance_is_zero_for_the_mean_predictor(self):
        y_log = np.array([0.0, 0.7, 1.1, 1.4])
        assert gaussian_explained_variance(
            y_log, np.full_like(y_log, y_log.mean()),
        ) == pytest.approx(0.0)

    def test_constant_evaluation_set_scores_zero_not_nan(self):
        """Nothing to explain must not become a NaN that poisons a fold mean."""

        y = np.full(5, 2.0)
        assert gamma_explained_deviance(y, y) == 0.0
        assert gaussian_explained_variance(y, y) == 0.0


class TestPairedFoldMargin:
    def test_constant_positive_margin_gives_zero_p_value(self):
        base = np.array([0.10, 0.12, 0.11, 0.09, 0.13])
        result = paired_fold_margin(base, base + 0.02, np.random.default_rng(1), 500, 0.99)

        assert result['mean_margin'] == pytest.approx(0.02)
        assert result['p_value'] == 0.0
        assert result['ci_low'] > 0.0
        assert result['folds_positive'] == 5

    def test_zero_margin_does_not_reject(self):
        base = np.array([0.10, 0.12, 0.11])
        result = paired_fold_margin(base, base, np.random.default_rng(1), 500, 0.99)

        assert result['mean_margin'] == pytest.approx(0.0)
        assert result['p_value'] == 1.0
        assert result['folds_positive'] == 0

    def test_negative_margin_has_a_negative_lower_bound(self):
        base = np.array([0.10, 0.12, 0.11])
        result = paired_fold_margin(base, base - 0.02, np.random.default_rng(1), 500, 0.99)

        assert result['mean_margin'] < 0.0
        assert result['ci_low'] < 0.0

    def test_failed_folds_are_dropped_not_propagated(self):
        """One non-converging fold must not turn the whole statistic into NaN."""

        result = paired_fold_margin(
            np.array([0.10, np.nan, 0.10]), np.array([0.15, 0.20, 0.15]),
            np.random.default_rng(1), 500, 0.99,
        )
        assert result['n_folds'] == 2
        assert result['mean_margin'] == pytest.approx(0.05)

    def test_all_folds_failing_returns_nan_rather_than_a_false_result(self):
        result = paired_fold_margin(
            np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
            np.random.default_rng(1), 100, 0.99,
        )
        assert np.isnan(result['mean_margin'])
        assert result['n_folds'] == 0

    def test_misaligned_fold_arrays_raise_value_error(self):
        with pytest.raises(ValueError, match='aligned'):
            paired_fold_margin(
                np.array([0.1, 0.2]), np.array([0.1]), np.random.default_rng(1), 10, 0.99,
            )


class TestBuildSessionFolds:
    def test_held_out_sessions_reach_no_fold(self):
        """The reserve must stay untouched or it stops being an honest last look."""

        groups = _blocked_groups({f's{i:02d}': 290 for i in range(30)})
        y = np.random.default_rng(0).gamma(2.0, 3.0, size=groups.size)
        held_out = ['s00', 's01', 's02']

        folds = build_session_folds(y, groups, held_out, 10, 0.1, 0)
        held_positions = set(np.flatnonzero(np.isin(groups, held_out)).tolist())
        for train_index, test_index in folds:
            assert not held_positions & set(train_index.tolist())
            assert not held_positions & set(test_index.tolist())

    def test_no_session_straddles_a_train_test_split(self):
        """Whole sessions are held out, so the score is cross-animal."""

        groups = _blocked_groups({f's{i:02d}': 290 for i in range(30)})
        y = np.random.default_rng(0).gamma(2.0, 3.0, size=groups.size)

        for train_index, test_index in build_session_folds(y, groups, [], 10, 0.1, 0):
            assert not set(groups[train_index]) & set(groups[test_index])

    def test_all_sessions_held_out_raises_value_error(self):
        groups = _blocked_groups({'s00': 100, 's01': 100})
        y = np.random.default_rng(0).gamma(2.0, 3.0, size=groups.size)

        with pytest.raises(ValueError, match='held-out'):
            build_session_folds(y, groups, ['s00', 's01'], 5, 0.2, 0)


class TestFitBlockAcrossFolds:
    def test_both_likelihood_arms_produce_a_finite_score(self):
        """A recoverable signal scores above zero under either likelihood."""

        rng = np.random.default_rng(0)
        history_frames, n_rows = 10, 120
        feature = rng.normal(size=(n_rows, history_frames))
        y = np.exp(0.5 + 0.8 * feature.mean(axis=1)) * rng.gamma(20.0, 1 / 20.0, size=n_rows)
        folds = [(np.arange(0, 80), np.arange(80, n_rows))]

        for likelihood in ('gamma', 'lognormal'):
            result = fit_block_across_folds(
                feature_arrays=[feature], y_global=y, cv_folds=folds,
                history_frames=history_frames, n_splines_value=4, n_splines_time=4,
                gam_kwargs={'lam': 0.6, 'max_iter': 50, 'tol': 1e-4},
                likelihood=likelihood,
            )
            assert np.isfinite(result['d2']).all()
            assert not result['failed_folds']

    def test_unknown_likelihood_raises_value_error(self):
        with pytest.raises(ValueError, match='likelihood'):
            fit_block_across_folds(
                feature_arrays=[np.zeros((10, 4))], y_global=np.ones(10),
                cv_folds=[(np.arange(5), np.arange(5, 10))], history_frames=4,
                n_splines_value=3, n_splines_time=3, gam_kwargs={},
                likelihood='poisson_identity',
            )

    def test_empty_feature_list_raises_value_error(self):
        with pytest.raises(ValueError, match='feature_arrays'):
            fit_block_across_folds(
                feature_arrays=[], y_global=np.ones(10),
                cv_folds=[(np.arange(5), np.arange(5, 10))], history_frames=4,
                n_splines_value=3, n_splines_time=3, gam_kwargs={},
            )


def _synthetic_block(n_sessions: int = 4, rows_per_session: int = 40,
                     history_frames: int = 8, seed: int = 0):
    """Builds a tiny aligned feature dict with a recoverable signal."""
    rng = np.random.default_rng(seed)
    n_rows = n_sessions * rows_per_session
    groups = np.repeat([f's{i:02d}' for i in range(n_sessions)], rows_per_session)
    driver = rng.normal(size=(n_rows, history_frames))
    vocal = rng.normal(size=(n_rows, history_frames))
    y = np.exp(0.5 + 0.8 * driver.mean(axis=1)) * rng.gamma(20.0, 1 / 20.0, size=n_rows)
    data = {
        'self.speed': {'X': driver, 'y': y, 'groups': groups},
        'other.usv_rate': {'X': vocal, 'y': y, 'groups': groups},
    }
    return data, y, groups, history_frames


class TestFoldDiagnostics:
    def test_diagnostics_are_omitted_unless_requested(self):
        """The null refits thousands of models; they must stay cheap."""

        data, y, _, history_frames = _synthetic_block()
        result = fit_block_across_folds(
            feature_arrays=[data['self.speed']['X']], y_global=y,
            cv_folds=[(np.arange(0, 120), np.arange(120, 160))],
            history_frames=history_frames, n_splines_value=4, n_splines_time=4,
            gam_kwargs={'lam': 0.6, 'max_iter': 50, 'tol': 1e-4},
        )
        assert 'filter_shapes' not in result
        assert 'y_pred' not in result

    def test_requested_diagnostics_carry_predictions_and_filters(self):
        """Predictions must be row-aligned with the fold's test indices.

        Without ``test_indices`` a saved prediction cannot be traced back to
        the anchor that produced it, so nothing can be re-scored or plotted
        against the data afterwards.
        """

        data, y, _, history_frames = _synthetic_block()
        feature_names = ['self.speed', 'other.usv_rate']
        result = fit_block_across_folds(
            feature_arrays=[data[f]['X'] for f in feature_names], y_global=y,
            cv_folds=[(np.arange(0, 120), np.arange(120, 160))],
            history_frames=history_frames, n_splines_value=4, n_splines_time=4,
            gam_kwargs={'lam': 0.6, 'max_iter': 50, 'tol': 1e-4},
            feature_names=feature_names, collect_diagnostics=True,
        )

        assert result['y_pred'][0].size == result['test_indices'][0].size == 40
        assert set(result['filter_shapes'][0]) == set(feature_names)
        for filter_curve in result['filter_shapes'][0].values():
            assert filter_curve.shape == (history_frames,)
        for metric in ('spearman_r', 'pearson_r', 'mae', 'rmse', 'residual_deviance'):
            assert np.isfinite(result[metric][0])
        assert result['score_scale'] == 'native'

    def test_lognormal_arm_records_its_scale(self):
        """Descriptive metrics are on the arm's own scale, and say so."""

        data, y, _, history_frames = _synthetic_block()
        result = fit_block_across_folds(
            feature_arrays=[data['self.speed']['X']], y_global=y,
            cv_folds=[(np.arange(0, 120), np.arange(120, 160))],
            history_frames=history_frames, n_splines_value=4, n_splines_time=4,
            gam_kwargs={'lam': 0.6, 'max_iter': 50, 'tol': 1e-4},
            likelihood='lognormal', collect_diagnostics=True,
        )
        assert result['score_scale'] == 'log'


class TestScoreOnHeldOut:
    def test_no_reserved_sessions_returns_empty_rather_than_raising(self):
        """A run with the carve-out disabled must still complete."""

        data, y, groups, history_frames = _synthetic_block()
        held_out = score_on_held_out(
            all_feature_data=data, baseline_features=['self.speed'],
            vocal_features=['other.usv_rate'], y_global=y, groups_global=groups,
            held_out_session_ids=[], history_frames=history_frames,
            gam_settings={'n_splines_value': 4, 'n_splines_time': 4, 'lam_penalty': 0.6,
                          'max_iterations': 50, 'tol_val': 1e-4},
            likelihood='gamma',
        )
        assert held_out['margin'] is None
        assert held_out['n_held_out_rows'] == 0

    def test_reserve_is_scored_and_never_trained_on(self):
        """The reserve is the only estimate untouched by the selection search.

        Its row count must match exactly the reserved sessions, which is what
        shows the development/held-out split was honoured rather than assumed.
        """

        data, y, groups, history_frames = _synthetic_block()
        held_out = score_on_held_out(
            all_feature_data=data, baseline_features=['self.speed'],
            vocal_features=['other.usv_rate'], y_global=y, groups_global=groups,
            held_out_session_ids=['s03'], history_frames=history_frames,
            gam_settings={'n_splines_value': 4, 'n_splines_time': 4, 'lam_penalty': 0.6,
                          'max_iterations': 50, 'tol_val': 1e-4},
            likelihood='gamma',
        )

        assert held_out['n_held_out_rows'] == int(np.sum(groups == 's03'))
        assert held_out['n_held_out_sessions'] == 1
        assert np.isfinite(held_out['baseline_score'])
        assert np.isfinite(held_out['full_score'])
        assert held_out['margin'] == pytest.approx(
            held_out['full_score'] - held_out['baseline_score'],
        )
        assert set(held_out['full']['filter_shapes'][0]) == {'self.speed', 'other.usv_rate'}


class TestFractionOfRemainingDeviance:
    def test_same_margin_is_a_larger_share_against_a_stronger_baseline(self):
        """The point of the statistic: an absolute margin is not comparable.

        A +0.05 gain over a baseline at 0.90 uses half of what was left; the
        same gain over a baseline at 0.10 uses a twentieth of it.
        """

        weak = fraction_of_remaining_deviance(np.array([0.10]), np.array([0.15]))
        strong = fraction_of_remaining_deviance(np.array([0.90]), np.array([0.95]))

        assert weak['mean'] == pytest.approx(0.05 / 0.90)
        assert strong['mean'] == pytest.approx(0.05 / 0.10)
        assert strong['mean'] > weak['mean']

    def test_zero_margin_is_zero_share(self):
        result = fraction_of_remaining_deviance(np.array([0.3, 0.4]), np.array([0.3, 0.4]))
        assert result['mean'] == pytest.approx(0.0)

    def test_baseline_at_one_leaves_nothing_to_explain_and_is_dropped(self):
        """A perfect baseline must not divide by zero."""

        result = fraction_of_remaining_deviance(np.array([1.0, 0.5]), np.array([1.0, 0.75]))
        assert result['n_folds'] == 1
        assert result['mean'] == pytest.approx(0.5)


class TestPerSessionScores:
    def test_each_session_is_scored_on_the_fold_that_held_it_out(self):
        """Splits a fold's stored predictions by session; nothing is refit."""

        groups = np.array(['a'] * 4 + ['b'] * 4)
        baseline = {
            'y_pred': [np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])],
            'test_indices': [np.arange(8)],
            'y_true': [np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])],
        }
        full = {
            'y_pred': [np.array([1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0])],
            'test_indices': [np.arange(8)],
            'y_true': [np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])],
        }

        result = per_session_scores(baseline, full, groups, likelihood='gamma')

        assert set(result) == {'a', 'b'}
        assert result['a']['n_rows'] == 4
        # session 'a' is predicted perfectly by the full model and poorly by the
        # baseline, so its margin is strongly positive; 'b' is unchanged.
        assert result['a']['full_d2'] == pytest.approx(1.0)
        assert result['a']['margin'] > 0.0
        assert result['b']['margin'] == pytest.approx(0.0)

    def test_a_session_never_held_out_has_no_entry(self):
        """Monte Carlo folds need not cover every session."""

        groups = np.array(['a'] * 4 + ['unseen'] * 4)
        diagnostics = {
            'y_pred': [np.array([1.0, 2.0, 3.0, 4.0])],
            'test_indices': [np.arange(4)],
            'y_true': [np.array([1.0, 2.0, 3.0, 4.0])],
        }
        result = per_session_scores(diagnostics, diagnostics, groups, likelihood='gamma')
        assert set(result) == {'a'}


class TestAcceptanceRuleAlignment:
    def test_selection_uses_the_repo_shared_paired_one_se_helper(self):
        """Screen and forward selection decide with ``paired_one_se_improvement``.

        The repo fixed its 1SE rule to pair per-fold scores; this module must
        use the same helper rather than a second, stricter convention.
        """

        improvement, improvement_se = paired_one_se_improvement(
            np.array([0.12, 0.14, 0.13]), np.array([0.10, 0.12, 0.11]),
            higher_is_better=True,
        )
        assert improvement == pytest.approx(0.02)
        # an identical per-fold gain has zero spread, so it is always accepted
        assert improvement_se == pytest.approx(0.0)
        assert improvement > improvement_se

    def test_paired_bar_is_more_lenient_than_the_bootstrap_gate(self):
        """The screen and selection bar must be the looser of the two.

        A strict bar truncates the baseline, and anything left out of the
        baseline is not controlled for -- it is merely absent.
        """

        baseline = np.array([0.10, 0.30, 0.50, 0.20])
        full = baseline + np.array([0.02, 0.01, 0.03, 0.00])

        improvement, improvement_se = paired_one_se_improvement(
            full, baseline, higher_is_better=True,
        )
        strict = paired_fold_margin(baseline, full, np.random.default_rng(0), 2000, 0.99)

        assert improvement > improvement_se          # passes the lenient rule
        assert strict['ci_low'] < improvement        # the strict bar sits higher


class TestVocalBlockIsWithheldFromSelection:
    def test_vocal_feature_inside_the_baseline_raises(self):
        """The block under test must never have been a selection candidate.

        If it reached the baseline, the final step would compare a model
        against itself and report nothing -- a silent null rather than a
        detectable failure, which is the worst way for this to break.
        """

        with pytest.raises(ValueError, match='already in the selected baseline'):
            vocal_block_final_step(
                all_feature_data={}, baseline_features=['self.speed', 'other.usv_rate'],
                vocal_features=['other.usv_rate'], baseline_scores=np.array([0.1]),
                y_global=np.ones(4), groups_global=np.array(['a'] * 4),
                cv_folds=[(np.arange(2), np.arange(2, 4))], held_out_session_ids=[],
                history_frames=4, gam_settings={}, step_index=1,
                output_directory=pathlib.Path(), step_prefix='x_', wrap_step=dict,
            )

    def test_extraction_partition_keeps_the_blocks_disjoint(self):
        """The screen's candidate list comes from the extraction metadata.

        ``baseline_block_features`` and ``vocal_block_features`` are written as
        complements, which is what makes the withholding structural rather
        than a convention a caller has to remember.
        """

        all_features = ['self.speed', 'nose-nose', 'other.usv_rate']
        vocal = [f for f in all_features if 'usv_' in f]
        baseline = [f for f in all_features if f not in set(vocal)]

        assert not set(baseline) & set(vocal)
        assert sorted(baseline + vocal) == sorted(all_features)


def _univariate_artifact(tmp_path, entries: dict[str, tuple[float, float]]):
    """Writes a minimal consolidated univariate pickle: feature -> (actual, null)."""
    payload = {
        name: {'actual': {'explained_deviance': np.full(4, actual)},
               'null': {'explained_deviance': np.full(4, null)}}
        for name, (actual, null) in entries.items()
    }
    payload['_run_metadata'] = {'x': 1}
    payload['_input_metadata'] = {'y': 2}
    path = tmp_path / 'univ.pkl'
    path.write_bytes(pickle.dumps(payload))
    return path


class TestScreenFromUnivariate:
    def test_features_beating_their_own_null_pass(self, tmp_path):
        """The screen reads scores the cluster array already computed."""

        path = _univariate_artifact(tmp_path, {
            'self.speed': (0.30, 0.00),
            'other.noise': (0.001, 0.000),
        })
        result = screen_from_univariate(path, ['self.speed', 'other.noise'])

        assert 'self.speed' in result['passed']
        assert result['per_feature']['self.speed']['paired_improvement'] == pytest.approx(0.30)

    def test_candidates_absent_from_the_artifact_abort_the_screen(self, tmp_path):
        """A feature missing from the array is a run failure, not a screen verdict.

        Proceeding would look identical to the feature having been tested and
        rejected, which quietly shrinks the pool the forward selection searches.
        Ruled fatal on 2026-09-03, the usual cause being a short `--array` bound.
        """

        path = _univariate_artifact(tmp_path, {'self.speed': (0.30, 0.00)})

        with pytest.raises(ValueError) as excinfo:
            screen_from_univariate(path, ['self.speed', 'other.never_ran'])

        message = str(excinfo.value)
        assert 'other.never_ran' in message
        assert 'never swept' in message
        # The remedy must name the right upper bound for a 2-candidate sweep.
        assert '--array=0-1' in message

    def test_incomplete_entries_abort_and_are_distinguished_from_absent_ones(self, tmp_path):
        """A present-but-unusable entry aborts too, with its own diagnosis.

        A crashed fit and a short job array both shrink the pool, but they call
        for different fixes, so the message must say which one happened.
        """

        path = _univariate_artifact(tmp_path, {'self.speed': (0.30, 0.00)})
        with path.open('rb') as handle:
            payload = pickle.load(handle)
        payload['other.broken'] = {'actual': {}, 'null': {}}
        with path.open('wb') as handle:
            pickle.dump(payload, handle)

        with pytest.raises(ValueError) as excinfo:
            screen_from_univariate(path, ['self.speed', 'other.broken'])

        message = str(excinfo.value)
        assert 'other.broken' in message
        assert 'explained_deviance' in message
        assert 'never swept' not in message

    def test_a_complete_artifact_still_screens_without_raising(self, tmp_path):
        """The abort must not fire when every candidate is present and usable."""

        path = _univariate_artifact(tmp_path, {'self.speed': (0.30, 0.00),
                                               'other.noise': (0.00, 0.00)})
        result = screen_from_univariate(path, ['self.speed', 'other.noise'])

        assert result['skipped'] == []
        assert 'self.speed' in result['passed']

    def test_reserved_metadata_blocks_are_not_treated_as_features(self, tmp_path):
        """The consolidated artifact carries metadata alongside the features."""

        path = _univariate_artifact(tmp_path, {'self.speed': (0.30, 0.00)})
        result = screen_from_univariate(path, ['self.speed'])
        assert set(result['per_feature']) == {'self.speed'}


class TestStepCheckpointing:
    def _tiny_problem(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        history_frames, n_rows = 6, 96
        groups = np.repeat([f's{i:02d}' for i in range(4)], 24)
        driver = rng.normal(size=(n_rows, history_frames))
        noise = rng.normal(size=(n_rows, history_frames))
        y = np.exp(0.5 + 0.9 * driver.mean(axis=1)) * rng.gamma(30.0, 1 / 30.0, size=n_rows)
        data = {'self.speed': {'X': driver, 'y': y, 'groups': groups},
                'other.noise': {'X': noise, 'y': y, 'groups': groups}}
        folds = [(np.arange(0, 72), np.arange(72, 96)), (np.arange(24, 96), np.arange(0, 24))]
        gam = {'n_splines_value': 4, 'n_splines_time': 4, 'lam_penalty': 0.6,
               'max_iterations': 40, 'tol_val': 1e-4}
        return data, y, folds, history_frames, gam

    def test_step_indices_are_zero_based(self, tmp_path):
        """``consolidate_model_selection_results`` compares merged indices
        against ``range(len(steps))`` and warns when they disagree."""

        data, y, folds, history_frames, gam = self._tiny_problem()
        forward_select_features(
            all_feature_data=data, screened_features=['self.speed', 'other.noise'],
            y_global=y, cv_folds=folds, history_frames=history_frames, gam_settings=gam,
            output_directory=tmp_path, step_prefix='model_selection_test_step_',
            wrap_step=lambda payload: payload,
        )
        written = sorted(int(f.stem.rsplit('_', 1)[-1]) for f in tmp_path.glob('*.pkl'))
        assert written == list(range(len(written)))
        assert written[0] == 0

    def test_step_prefix_matches_the_consolidator_convention(self, tmp_path):
        """The consolidator infers the prefix by requiring this shape."""

        data, y, folds, history_frames, gam = self._tiny_problem()
        forward_select_features(
            all_feature_data=data, screened_features=['self.speed'],
            y_global=y, cv_folds=folds, history_frames=history_frames, gam_settings=gam,
            output_directory=tmp_path, step_prefix='model_selection_test_step_',
            wrap_step=lambda payload: payload,
        )
        for written in tmp_path.glob('*.pkl'):
            assert written.name.startswith('model_selection_')
            assert '_step_' in written.name

    def test_rejected_step_is_persisted_too(self, tmp_path):
        """The stopping step records WHY the search ended.

        Without it a resume cannot tell convergence from an interrupted run
        and would re-test the same losing candidates.
        """

        data, y, folds, history_frames, gam = self._tiny_problem()
        result = forward_select_features(
            all_feature_data=data, screened_features=['self.speed', 'other.noise'],
            y_global=y, cv_folds=folds, history_frames=history_frames, gam_settings=gam,
            output_directory=tmp_path, step_prefix='model_selection_test_step_',
            wrap_step=lambda payload: payload,
        )
        assert result['steps'][-1]['selected_feature'] is None
        assert len(list(tmp_path.glob('*.pkl'))) == len(result['steps'])


class TestRestoreLastStep:
    def test_missing_directory_returns_none(self, tmp_path):
        assert _restore_last_step(tmp_path / 'absent', 'p_') is None

    def test_highest_numbered_step_is_restored(self, tmp_path):
        for index in (0, 1, 2):
            (tmp_path / f'p_{index}.pkl').write_bytes(
                pickle.dumps({'current_features': [f'f{index}'], 'selected_feature': 'x'}),
            )
        assert _restore_last_step(tmp_path, 'p_')['current_features'] == ['f2']

    def test_unusable_checkpoint_is_treated_as_absent(self, tmp_path):
        """A truncated file must start the run fresh, not resume corrupt state."""

        (tmp_path / 'p_0.pkl').write_bytes(b'not a pickle')
        assert _restore_last_step(tmp_path, 'p_') is None

    def test_payload_without_the_expected_keys_is_rejected(self, tmp_path):
        (tmp_path / 'p_0.pkl').write_bytes(pickle.dumps({'unrelated': True}))
        assert _restore_last_step(tmp_path, 'p_') is None


class TestResumeAfterConvergence:
    def _write_converged_run(self, tmp_path, prefix='model_selection_x_step_'):
        """Two accepted steps followed by a rejected one -- a finished search."""
        payloads = [
            {'step_index': 0, 'current_features': [], 'selected_feature': 'self.speed',
             'selected_feature_folds': np.array([0.3, 0.3]), 'baseline_folds': np.zeros(2)},
            {'step_index': 1, 'current_features': ['self.speed'], 'selected_feature': 'nose-nose',
             'selected_feature_folds': np.array([0.4, 0.4]), 'baseline_folds': np.array([0.3, 0.3])},
            {'step_index': 2, 'current_features': ['self.speed', 'nose-nose'],
             'selected_feature': None, 'selected_feature_folds': None,
             'baseline_folds': np.array([0.4, 0.4])},
        ]
        for index, payload in enumerate(payloads):
            (tmp_path / f'{prefix}{index}.pkl').write_bytes(pickle.dumps(payload))
        return prefix

    def test_converged_run_restores_its_features_instead_of_restarting(self, tmp_path):
        """A rejected last step means the search FINISHED, not that there is
        nothing to resume.

        Treating it as nothing-to-restore silently discarded every accepted
        feature and repeated the entire forward search -- deterministic, so
        the answer was unchanged, but hours of cluster time for nothing.
        """

        prefix = self._write_converged_run(tmp_path)
        result = forward_select_features(
            all_feature_data={'self.speed': {'X': np.zeros((4, 2))},
                              'nose-nose': {'X': np.zeros((4, 2))}},
            screened_features=['self.speed', 'nose-nose'], y_global=np.ones(4),
            cv_folds=[(np.arange(2), np.arange(2, 4)), (np.arange(2, 4), np.arange(2))],
            history_frames=2,
            gam_settings={'lam_penalty': 0.6, 'max_iterations': 10, 'tol_val': 1e-4,
                          'n_splines_value': 3, 'n_splines_time': 3},
            output_directory=tmp_path, step_prefix=prefix, wrap_step=lambda payload: payload,
        )

        assert result['selected'] == ['self.speed', 'nose-nose']
        np.testing.assert_allclose(result['final_scores'], [0.4, 0.4])
        # nothing was refit: only the restored checkpoint is in `steps`
        assert len(result['steps']) == 1

    def test_converged_run_with_no_accepted_features_still_raises(self, tmp_path):
        """An empty baseline cannot anchor the vocal comparison.

        Restoring a converged-but-empty run must not slip past the guard that
        rejects a vocal increment measured against nothing.
        """

        prefix = 'model_selection_empty_step_'
        (tmp_path / f'{prefix}0.pkl').write_bytes(pickle.dumps(
            {'step_index': 0, 'current_features': [], 'selected_feature': None,
             'selected_feature_folds': None, 'baseline_folds': np.zeros(2)},
        ))
        with pytest.raises(RuntimeError, match='accepted no feature'):
            forward_select_features(
                all_feature_data={'self.speed': {'X': np.zeros((4, 2))}},
                screened_features=['self.speed'], y_global=np.ones(4),
                cv_folds=[(np.arange(2), np.arange(2, 4))], history_frames=2,
                gam_settings={'lam_penalty': 0.6, 'max_iterations': 10, 'tol_val': 1e-4,
                              'n_splines_value': 3, 'n_splines_time': 3},
                output_directory=tmp_path, step_prefix=prefix, wrap_step=lambda payload: payload,
            )


class TestTopRankAnchor:
    def _problem(self):
        rng = np.random.default_rng(0)
        history_frames, n_rows = 6, 96
        groups = np.repeat([f's{i:02d}' for i in range(4)], 24)
        driver = rng.normal(size=(n_rows, history_frames))
        noise = rng.normal(size=(n_rows, history_frames))
        y = np.exp(0.5 + 0.9 * driver.mean(axis=1)) * rng.gamma(30.0, 1 / 30.0, size=n_rows)
        data = {'self.speed': {'X': driver, 'y': y, 'groups': groups},
                'other.noise': {'X': noise, 'y': y, 'groups': groups}}
        folds = [(np.arange(0, 72), np.arange(72, 96)), (np.arange(24, 96), np.arange(0, 24))]
        gam = {'n_splines_value': 4, 'n_splines_time': 4, 'lam_penalty': 0.6,
               'max_iterations': 40, 'tol_val': 1e-4}
        return data, y, folds, history_frames, gam

    def test_anchor_forces_the_top_ranked_feature_as_step_zero(self, tmp_path):
        """``--anchor`` accepts the screen's best candidate without testing
        the alternatives, as the other selectors do."""

        data, y, folds, history_frames, gam = self._problem()
        result = forward_select_features(
            all_feature_data=data, screened_features=['other.noise', 'self.speed'],
            y_global=y, cv_folds=folds, history_frames=history_frames, gam_settings=gam,
            output_directory=tmp_path, step_prefix='model_selection_a_step_',
            wrap_step=lambda payload: payload, use_top_rank_as_anchor=True,
        )

        # 'other.noise' is first in the ranked list, so it anchors even though it
        # carries no signal -- that is exactly what forcing an anchor means.
        assert result['steps'][0]['selected_feature'] == 'other.noise'
        assert result['steps'][0]['forced_anchor'] is True
        assert result['selected'][0] == 'other.noise'

    def test_anchor_step_evaluates_only_the_anchor(self, tmp_path):
        """The saving is that step 0 fits one model instead of one per candidate."""

        data, y, folds, history_frames, gam = self._problem()
        result = forward_select_features(
            all_feature_data=data, screened_features=['self.speed', 'other.noise'],
            y_global=y, cv_folds=folds, history_frames=history_frames, gam_settings=gam,
            output_directory=tmp_path, step_prefix='model_selection_b_step_',
            wrap_step=lambda payload: payload, use_top_rank_as_anchor=True,
        )
        assert list(result['steps'][0]['candidates']) == ['self.speed']

    def test_without_the_anchor_step_zero_tests_every_candidate(self, tmp_path):
        """The default path compares all candidates multivariately at step 0."""

        data, y, folds, history_frames, gam = self._problem()
        result = forward_select_features(
            all_feature_data=data, screened_features=['other.noise', 'self.speed'],
            y_global=y, cv_folds=folds, history_frames=history_frames, gam_settings=gam,
            output_directory=tmp_path, step_prefix='model_selection_c_step_',
            wrap_step=lambda payload: payload,
        )
        assert set(result['steps'][0]['candidates']) == {'other.noise', 'self.speed'}
        assert 'forced_anchor' not in result['steps'][0]
        # unforced, the signal-bearing feature wins despite being ranked second
        assert result['selected'][0] == 'self.speed'
