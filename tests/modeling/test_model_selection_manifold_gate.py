"""
@author: bartulem
Regression tests for the manifold (behaviour -> acoustic-position) selection gates.

Coverage: two rules are pinned here.

* The fold-margin p-value is an exact one-sided sign-flip permutation test over the
  folds (``_sign_flip_p_value``): calibrated under a symmetric null, never 0, with the
  floor ``1 / 2**n_folds`` (``(1 + k) / (1 + B)`` over the ``B = 2**n - 1`` non-identity
  sign patterns) and, above ``_SIGN_FLIP_EXACT_MAX_FOLDS`` folds, ``1 / (1 + B)`` over a
  seeded Monte-Carlo set of ``B`` patterns. The previous fold-bootstrap probability
  ``mean(boot_means <= 0)`` rejected a true null 8% of the time at 0.05 with 10 folds and
  could return exactly 0. The shipped ``model_validation.manifold_n_cv_folds`` is 20, so the
  exact floor (9.5e-7) sits below the BH rank-1 threshold ``q / m``; every other pipeline
  keeps ``model_validation.n_cv_folds`` at 10.
* On the torus the anchor / first forward feature is gated against a training-fold
  INTERCEPT-ONLY predictor (the circular mean of each fold's training targets), not the
  empirical-density random draw; the draw stays in the Step-0 file as the reported chance
  floor. A model carrying no behavioural information is therefore rejected (it used to be
  accepted because any concentrated prediction beats a random draw from the repertoire).
"""

from __future__ import annotations

import itertools
import json
import pickle
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest

# Headless backend before the modeling modules (which import pyplot) load.
mpl.use('Agg')

with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import usv_playpen.modeling.model_selection as model_selection_module
    from tests.modeling.test_pipeline_manifold import (
        HISTORY_FRAMES,
        _build_manifold_settings,
        _build_signal_continuous_pickle,
    )
    from usv_playpen.modeling.manifold_metric import (
        circular_mean,
        inverse_region_frequency_weights,
    )
    from usv_playpen.modeling.model_selection import (
        _fold_paired_margin_bootstrap,
        _frozen_selection_kappa,
        _sign_flip_p_value,
        continuous_vocal_manifold_model_selection,
    )
    from usv_playpen.modeling.modeling_usv_manifold_position import (
        ContinuousModelingPipeline,
        ContinuousModelRunner,
    )

SHIPPED_SETTINGS = Path(model_selection_module.__file__).resolve().parent.parent / '_parameter_settings' / 'modeling_settings.json'
with SHIPPED_SETTINGS.open() as _shipped_settings_file:
    _SHIPPED_MODEL_PARAMS = json.load(_shipped_settings_file)['model_params']
# The shipped sign-flip enumeration cap and Monte-Carlo draw count the selection gate reads.
_SIGN_FLIP_EXACT_MAX_FOLDS = _SHIPPED_MODEL_PARAMS['selection_sign_flip_exact_max_folds']
_SIGN_FLIP_MONTE_CARLO_DRAWS = _SHIPPED_MODEL_PARAMS['selection_sign_flip_monte_carlo_draws']
_SIGN_FLIP_KWARGS = {'exact_max_folds': _SIGN_FLIP_EXACT_MAX_FOLDS,
                     'monte_carlo_draws': _SIGN_FLIP_MONTE_CARLO_DRAWS}


def _fold_dicts_from_scores(actual_scores: np.ndarray, null_scores: np.ndarray, scores: dict) -> tuple[dict, dict]:
    """
    Description
    -----------
    Builds an ``actual`` and a ``null`` fold bundle whose per-fold score is preset: every
    fold gets its own zero prediction array, and ``scores`` maps that array's ``id`` to the
    score a patched euclidean scorer returns for it.

    Parameters
    ----------
    actual_scores (np.ndarray)
        ``(n_folds,)`` scores the ``actual`` folds should receive.
    null_scores (np.ndarray)
        ``(n_folds,)`` scores the ``null`` folds should receive.
    scores (dict)
        Filled in place: ``id(y_pred_xy) -> score``.

    Returns
    -------
    actual (dict)
        The ``actual`` fold bundle (``test_indices``, ``y_pred_xy``, ``y_true``).
    null (dict)
        The ``null`` fold bundle with the same keys.
    """
    actual = {'test_indices': [], 'y_pred_xy': [], 'y_true': []}
    null = {'test_indices': [], 'y_pred_xy': [], 'y_true': []}
    for a_score, n_score in zip(actual_scores, null_scores, strict=True):
        y_a, y_n = np.zeros((40, 2)), np.zeros((40, 2))
        scores[id(y_a)], scores[id(y_n)] = float(a_score), float(n_score)
        for bundle, y_pred in ((actual, y_a), (null, y_n)):
            bundle['test_indices'].append(np.arange(40))
            bundle['y_pred_xy'].append(y_pred)
            bundle['y_true'].append(np.zeros((40, 2)))
    return actual, null


class TestSignFlipPValue:
    """The exact sign-flip permutation p-value of the per-fold paired margin."""

    def test_matches_brute_force_enumeration(self):
        """The doubling construction counts the same patterns as enumerating every sign vector."""
        margins = np.random.default_rng(3).normal(0.2, 1.0, 9)
        n_at_least = sum(
            float(np.dot(signs, margins)) >= float(margins.sum()) - 1e-12
            for signs in itertools.product((1.0, -1.0), repeat=margins.size))
        assert _sign_flip_p_value(margins, random_state=0, **_SIGN_FLIP_KWARGS) == pytest.approx(n_at_least / 2 ** margins.size)

    def test_floor_at_twenty_folds(self):
        """Twenty positive margins reach the exact floor 1 / 2**20 (never 0)."""
        p_value = _sign_flip_p_value(np.linspace(0.1, 2.0, 20), random_state=0, **_SIGN_FLIP_KWARGS)
        assert p_value == 1.0 / 2 ** 20
        assert p_value < 0.01 / 19

    def test_ties_with_zero_margins_count(self):
        """A zero margin flips to itself, so its two patterns tie the observed sum."""
        assert _sign_flip_p_value(np.array([0.0, 1.0, 2.0]), random_state=0, **_SIGN_FLIP_KWARGS) == pytest.approx(2.0 / 8.0)
        assert _sign_flip_p_value(np.zeros(5), random_state=0, **_SIGN_FLIP_KWARGS) == 1.0

    def test_monte_carlo_branch_above_the_exact_cap(self):
        """Above the enumeration cap the seeded Monte-Carlo set gives (1 + k) / (1 + B), floor 1 / (1 + B)."""
        margins = np.ones(_SIGN_FLIP_EXACT_MAX_FOLDS + 2)
        p_value = _sign_flip_p_value(margins, random_state=0, **_SIGN_FLIP_KWARGS)
        assert 1.0 / (1.0 + _SIGN_FLIP_MONTE_CARLO_DRAWS) <= p_value <= 3.0 / (1.0 + _SIGN_FLIP_MONTE_CARLO_DRAWS)
        assert p_value == _sign_flip_p_value(margins, random_state=0, **_SIGN_FLIP_KWARGS)

    def test_fold_margin_p_is_calibrated_under_the_null(self, monkeypatch):
        """
        Under a symmetric null (per-fold margins centred on zero) the fold-margin p rejects
        at about the nominal rate and is never 0. The old fold-bootstrap probability rejected
        ~8% at alpha = 0.05 with 10 folds and returned exact zeros.
        """
        scores = {}
        monkeypatch.setattr(model_selection_module, 'dcor_prediction_truth',
                            lambda y_pred, _y_true, **_kwargs: scores[id(y_pred)])
        rng = np.random.default_rng(0)
        n_folds, n_rep = 10, 3000
        p_values = np.empty(n_rep)
        for rep in range(n_rep):
            actual, null = _fold_dicts_from_scores(rng.normal(0.0, 1.0, n_folds), np.zeros(n_folds), scores)
            result = _fold_paired_margin_bootstrap(actual, null, metric='euclidean', period=1.0,
                                                   n_bootstrap=50, ci_level=0.99, random_state=rep, min_region_events=1, min_fold_events=30, sign_flip_exact_max_folds=22, sign_flip_monte_carlo_draws=2**20, dcor_n_sub=2500, dcor_n_rep=1)
            p_values[rep] = result['p_value']
            scores.clear()
        assert 0.035 <= float(np.mean(p_values <= 0.05)) <= 0.065
        assert float(p_values.min()) >= 1.0 / 2 ** n_folds

    def test_shipped_manifold_fold_count_is_twenty(self):
        """The shipped manifold outer fold count is 20, so the exact sign-flip floor clears q / m at rank 1."""
        shipped = json.loads(SHIPPED_SETTINGS.read_text())
        assert shipped['model_validation']['manifold_n_cv_folds'] == 20
        # 19 features enter the shipped cohort screen; BH's rank-1 threshold is q / 19.
        assert shipped['model_params']['selection_p_val'] / 19 > 1.0 / 2 ** 20

    def test_shipped_fold_count_of_the_other_pipelines_is_ten(self):
        """Every pipeline other than the manifold position keeps 10 outer folds."""
        shipped = json.loads(SHIPPED_SETTINGS.read_text())
        assert shipped['model_validation']['n_cv_folds'] == 10


class TestStepZeroGateBaseline:
    """The torus anchor / first-feature gate compares against a training-fold intercept-only predictor."""

    def test_intercept_only_candidate_ties_its_own_baseline(self):
        """
        Repro basis: a zero-information intercept-only candidate gated against the
        intercept-only baseline has margin 0 and a CI that is not above 0 (REJECT), while
        against the old random-draw baseline it passed (margin ~0.8, CI above 0).
        """
        period = 2.0 * np.pi
        rng = np.random.default_rng(1)
        n_events, n_folds = 4000, 10
        Y = np.mod(rng.vonmises(0.0, 2.0, size=(n_events, 2)), period)
        regions = rng.integers(0, 4, n_events).astype(float)
        kappa = _frozen_selection_kappa(Y, np.ones(n_events), metric='torus', period=period)
        folds = np.array_split(rng.permutation(n_events), n_folds)
        intercept = {'test_indices': [], 'y_pred_xy': [], 'y_true': []}
        draw = {'test_indices': [], 'y_pred_xy': [], 'y_true': []}
        for k, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(n_events), test_idx)
            mu = circular_mean(Y[train_idx], metric='torus', period=period,
                               weights=np.full(train_idx.size, 1.0 / train_idx.size))
            draw_idx = np.random.default_rng(k + 2).choice(train_idx.size, size=test_idx.size, replace=True)
            for bundle, y_pred in ((intercept, np.tile(mu, (test_idx.size, 1))), (draw, Y[train_idx][draw_idx])):
                bundle['test_indices'].append(test_idx)
                bundle['y_true'].append(Y[test_idx])
                bundle['y_pred_xy'].append(y_pred)
        kwargs = {'metric': 'torus', 'period': period, 'n_bootstrap': 1000, 'ci_level': 0.99, 'random_state': 0,
                      'event_to_region': regions, 'min_region_events': 1, 'kappa': kappa,
                      'min_fold_events': 30, 'sign_flip_exact_max_folds': 22,
                      'sign_flip_monte_carlo_draws': 2**20, 'dcor_n_sub': 2500, 'dcor_n_rep': 1}
        versus_intercept = _fold_paired_margin_bootstrap(intercept, intercept, **kwargs)
        versus_draw = _fold_paired_margin_bootstrap(intercept, draw, **kwargs)
        assert versus_intercept['mean_margin'] == 0.0
        assert not versus_intercept['ci_low'] > 0.0
        assert versus_draw['ci_low'] > 0.0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_behaviour_free_target_is_rejected_by_the_anchor_gate(self, tmp_path):
        """
        End to end on the torus: the univariate ranking comes from a signal-bearing input,
        so the screen passes the first feature, but the selection itself runs on a target
        with NO dependence on behaviour (a concentrated von Mises cloud). The anchor and every
        forward step must be rejected against the intercept-only baseline, the Step-0 file
        must carry that baseline (first, named in ``baseline_strategy``) beside the
        separately stored density-draw chance floor, and the anchor gate must have compared
        against constant per-fold predictions at the training circular mean. Under the old
        random-draw baseline this anchor was accepted.
        """
        settings, _save_dir = _build_manifold_settings(
            tmp_path, split_strategy='session', split_num=10, test_proportion=0.3,
        )
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        settings['hyperparameters']['linear_models']['manifold_regression']['bin_resizing_factor'] = 1
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(25)]
        input_md = {
            'analysis_type': 'continuous',
            'analysis_tag': 'manifold_qlvm_category',
            'session_ids': session_ids,
            'n_events_per_session': dict.fromkeys(session_ids, 60),
            'analysis_specific': {
                'usv_category_column_name': 'qlvm_category',
                'manifold_metric': 'torus',
                'manifold_period': 1.0,
            },
        }
        signal_pkl = _build_signal_continuous_pickle(
            save_path=tmp_path / 'manifold_signal_input.pkl', feature_names=feature_names,
            session_ids=session_ids, history_frames=HISTORY_FRAMES, input_metadata=input_md,
            target_kind='wound_torus',
        )
        runner = ContinuousModelRunner(ContinuousModelingPipeline(modeling_settings_dict=settings))
        combined = {feature: runner.run_univariate_training(str(signal_pkl), feature) for feature in feature_names}
        # The univariate stage copies the PICKLE's metadata block verbatim (the builder
        # copied `input_md` before recording the trace sidecar in it) and the selection
        # prefers that copy, so take it from the pickle as the dispatcher does.
        with signal_pkl.open('rb') as fh:
            combined['_input_metadata'] = pickle.load(fh)['_input_metadata']
        combined_path = tmp_path / 'univariate_combined.pkl'
        with combined_path.open('wb') as fh:
            pickle.dump(combined, fh)

        # The same events with a behaviour-free target: concentrated on the torus, independent of X.
        with signal_pkl.open('rb') as fh:
            artifact = pickle.load(fh)
        target_rng = np.random.default_rng(7)
        for sess_id in session_ids:
            angles = target_rng.vonmises(0.0, 4.0, size=(60, 2))
            y_free = np.mod(angles / (2.0 * np.pi), 1.0).astype(np.float32)
            for feature in feature_names:
                artifact[feature][sess_id]['Y'] = y_free
        free_pkl = tmp_path / 'manifold_free_input.pkl'
        with free_pkl.open('wb') as fh:
            pickle.dump(artifact, fh)

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()

        # A behaviour-free anchor is accepted only when none of the B = floor(m / alpha)
        # shifted refits reaches its in-sample gain, i.e. with probability 1 / (B + 1) under
        # exchangeability: alpha = 0.02 over the m screened candidates (the screen decides
        # how many of the three) makes that 1 / (floor(m / 0.02) + 1); the seeds fix the outcome.
        continuous_vocal_manifold_model_selection(
            univariate_results_path=str(combined_path), input_data_path=str(free_pkl),
            output_directory=str(ms_dir), settings_path=str(settings_json),
            use_top_rank_as_anchor=True, p_val=0.02,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_continuous_manifold_*_step_*.pkl'),
                           key=lambda path: int(path.stem.rsplit('_', 1)[1]))
        steps = []
        for path in step_pkls:
            with path.open('rb') as fh:
                steps.append(pickle.load(fh))
        # The selection accepted nothing: the behaviour-free anchor was rejected.
        assert steps[-1]['final_model_features'] == []
        assert all(step['selected_feature'] in ('null_model_free', None) for step in steps)

        step0 = steps[0]
        assert step0['selected_feature'] == 'null_model_free'
        assert step0['baseline_strategy'] == 'null_model_intercept'
        assert list(step0['candidates_summary']) == ['null_model_intercept', 'null_model_free']
        assert step0['baseline_score'] == step0['candidates_summary']['null_model_intercept']['mean_r2']
        assert step0['chance_floor_score'] == step0['candidates_summary']['null_model_free']['mean_r2']
        assert step0['_run_metadata']['extra_knobs']['step1_gate_baseline'] == 'null_model_intercept'
        assert step0['_run_metadata']['extra_knobs']['manifold_n_cv_folds'] == settings['model_validation']['manifold_n_cv_folds']

        # The stored Step-0 intercept baseline is the training-fold circular mean: one
        # constant prediction per fold.
        intercept_folds = step0['candidates_summary']['null_model_intercept']['folds']
        for stored_pred in intercept_folds['y_pred_xy']:
            fold_pred = np.asarray(stored_pred)
            assert np.all(fold_pred == fold_pred[0])
        # The forced anchor was tested by the per-step shift test against the development
        # intercept baseline, with Bonferroni over every screened feature, and rejected.
        anchor_test = steps[1]['step_test']
        assert anchor_test['accepted'] is False
        screened = steps[1]['_run_metadata']['extra_knobs']['step_test']['screened_features']
        assert anchor_test['m_candidates'] == len(screened)
        assert 1 <= len(screened) <= len(feature_names)
        assert anchor_test['n_refits'] == int(np.floor(len(screened) / 0.02))
        assert anchor_test['winner'] == next(iter(steps[1]['candidates_summary']))
        assert np.isfinite(anchor_test['score_incumbent'])
        assert steps[1]['_run_metadata']['extra_knobs']['step_test']['rule'] == 'shift_refit_bonferroni'

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_pure_noise_anchor_ties_the_region_reweighted_intercept(self, tmp_path):
        """
        Repro basis: with imbalanced acoustic regions (70% of the USVs in one region, 30% in
        another, at well-separated torus positions) and a target carrying NO behavioural
        information, the candidates are fitted with the KDE weights times the equal-region
        reweighting, so as their coefficients shrink they predict the REGION-REWEIGHTED circular
        mean. The Step-0 intercept baseline used to be the KDE-weighted mean alone, which gave
        the pure-noise anchor a large positive margin and accepted it. Built with the same
        weighting, the anchor ties the baseline (margin ~0) and is rejected, and every stored
        Step-0 intercept prediction sits at the region-reweighted training-fold mean.
        """
        settings, _save_dir = _build_manifold_settings(
            tmp_path, split_strategy='session', split_num=10, test_proportion=0.3,
        )
        settings['vocal_features']['usv_manifold_metric'] = 'torus'
        settings['vocal_features']['usv_manifold_min_region_events'] = 5
        settings['hyperparameters']['linear_models']['manifold_regression']['bin_resizing_factor'] = 1
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(25)]
        n_per_session = 120
        input_md = {
            'analysis_type': 'continuous',
            'analysis_tag': 'manifold_qlvm_category',
            'session_ids': session_ids,
            'n_events_per_session': dict.fromkeys(session_ids, n_per_session),
            'analysis_specific': {
                'usv_category_column_name': 'qlvm_category',
                'manifold_metric': 'torus',
                'manifold_period': 1.0,
            },
        }
        signal_pkl = _build_signal_continuous_pickle(
            save_path=tmp_path / 'manifold_signal_input.pkl', feature_names=feature_names,
            session_ids=session_ids, history_frames=HISTORY_FRAMES, input_metadata=input_md,
            n_per_session=n_per_session, target_kind='wound_torus',
        )
        runner = ContinuousModelRunner(ContinuousModelingPipeline(modeling_settings_dict=settings))
        combined = {feature: runner.run_univariate_training(str(signal_pkl), feature) for feature in feature_names}
        # The univariate stage copies the PICKLE's metadata block verbatim (the builder
        # copied `input_md` before recording the trace sidecar in it) and the selection
        # prefers that copy, so take it from the pickle as the dispatcher does.
        with signal_pkl.open('rb') as fh:
            combined['_input_metadata'] = pickle.load(fh)['_input_metadata']
        combined_path = tmp_path / 'univariate_combined.pkl'
        with combined_path.open('wb') as fh:
            pickle.dump(combined, fh)

        # A behaviour-free target in two imbalanced, well-separated regions, with region labels.
        with signal_pkl.open('rb') as fh:
            artifact = pickle.load(fh)
        target_rng = np.random.default_rng(11)
        for sess_id in session_ids:
            region = (target_rng.random(n_per_session) < 0.3).astype(np.float32)
            centre = np.where(region[:, None] == 1.0, 0.6, 0.2)
            y_free = np.mod(centre + 0.05 * target_rng.standard_normal((n_per_session, 2)), 1.0).astype(np.float32)
            for feature in feature_names:
                artifact[feature][sess_id]['Y'] = y_free
                artifact[feature][sess_id]['category'] = region
        free_pkl = tmp_path / 'manifold_free_input.pkl'
        with free_pkl.open('wb') as fh:
            pickle.dump(artifact, fh)

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))
        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()

        # See the sibling test: alpha = 0.02 keeps the chance acceptance of a noise anchor at
        # 1 / (B + 1) = 1 / 151.
        continuous_vocal_manifold_model_selection(
            univariate_results_path=str(combined_path), input_data_path=str(free_pkl),
            output_directory=str(ms_dir), settings_path=str(settings_json),
            use_top_rank_as_anchor=True, p_val=0.02,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_continuous_manifold_*_step_*.pkl'),
                           key=lambda path: int(path.stem.rsplit('_', 1)[1]))
        steps = []
        for path in step_pkls:
            with path.open('rb') as fh:
                steps.append(pickle.load(fh))
        # The anchor's in-sample gain over the region-reweighted intercept is matched by the
        # shifted refits: the shift test rejects it and the selection accepts nothing.
        anchor_test = steps[1]['step_test']
        assert anchor_test['accepted'] is False
        assert anchor_test['k_exceed'] >= 1
        assert steps[-1]['final_model_features'] == []

        # Every stored Step-0 intercept prediction is the region-reweighted training-fold mean.
        y_all = np.vstack([artifact[feature_names[0]][sess]['Y'] for sess in sorted(session_ids)]).astype(np.float64)
        regions_all = np.concatenate([artifact[feature_names[0]][sess]['category'] for sess in sorted(session_ids)])
        intercept_folds = steps[0]['candidates_summary']['null_model_intercept']['folds']
        all_test = np.concatenate([np.asarray(test_idx) for test_idx in intercept_folds['test_indices']])
        for test_idx, stored_pred in zip(intercept_folds['test_indices'], intercept_folds['y_pred_xy'], strict=True):
            train_idx = np.setdiff1d(all_test, np.asarray(test_idx))
            weights = inverse_region_frequency_weights(regions_all[train_idx])
            expected = circular_mean(y_all[train_idx], metric='torus', period=1.0, weights=weights / weights.sum())
            np.testing.assert_allclose(np.asarray(stored_pred, dtype=np.float64)[0], expected, atol=1e-4)
