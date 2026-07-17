"""
@author: bartulem
Per-fold fit-FAILURE coverage tests for the multinomial and manifold forward-
stepwise selectors in ``usv_playpen.modeling.model_selection``.

These tests deliberately do NOT re-walk the happy paths already covered by the
``test_pipeline_multinomial`` / ``test_pipeline_manifold`` smoke suites. Instead
they drive the *uncovered* per-fold fit-failure NaN-padding branches of the two
JAX-driven fold-fitters — the analogue of the existing
``compute_filter_shapes_per_fold_vocal_onset`` failure test in
``test_model_selection_filter_shapes`` / ``test_model_selection_tail``, but for
the multinomial and manifold orchestrators:

* ``TestMultinomialFoldFailure`` runs the real
  ``multinomial_vocal_category_model_selection`` on a strong-signal three-class
  synthetic input pickle with a matching hand-built univariate ranking, but
  monkeypatches the module-level ``SmoothMultinomialLogisticRegression`` so its
  ``.fit`` always raises. The Step-0 model-free marginal-prior baseline (which
  uses no learned model) still establishes successfully, and every *learned*
  fold (the auto-anchor block and every forward-selection trial) takes the
  per-fold ``except`` branch: scalar-NaN metrics, ``None`` weights / intercepts,
  empty-but-well-shaped ``y_true`` / ``y_pred`` / ``y_probs`` / ``test_indices``,
  a ``(K, K)`` NaN confusion matrix, and the NaN hyperparameter-audit
  placeholders. The anchor's "every fold errored out" arm and the forward
  loop's "no finite folds" skip both fire, the REJECT branch persists the
  (empty-candidate) step pickle, and the final-promotion ``try`` runs with the
  ``null_model_free`` winner (so no weight reshaping is attempted). Drives
  L3326-3355 (anchor padding), L3486-3554 (forward padding), and the surrounding
  accounting / finalize blocks.

* ``TestManifoldFoldFailure`` runs the real
  ``continuous_vocal_manifold_model_selection`` on a strong-signal synthetic
  manifold input pickle plus a freshly-computed univariate ranking, but
  monkeypatches the module-level ``resolve_manifold_regressor_cls`` factory so
  every fit raises. The Step-0 empirical-density baseline still establishes, and
  every learned fold takes the ``_append_failed_fold`` placeholder branch
  (L4314-4328), the anchor's failure arm (L4443-4445) and the forward loop's
  "no finite folds" skip (L4569) all fire, the REJECT branch persists the step
  pickle, and the finalize block runs.

To keep the data tiny and the run fast, the univariate "ranking" pickle each
selector screens is synthesized directly (the multinomial one via the shared
``build_univariate_ranking_pickle`` and the manifold one via the real runner on
the same signal pickle), and the JAX knobs are shrunk to the bone via the
existing test helpers. Everything lives under ``tmp_path`` so the source-tree
integrity guard never trips.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling import
chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so the top-level
modeling imports are wrapped in a ``warnings.catch_warnings`` block that ignores
``DeprecationWarning`` during import. The forced-failure folds, the all-NaN
metric aggregations, and the JAX / sklearn / SciPy stack emit assorted benign
``RuntimeWarning`` / ``UserWarning`` / ``DeprecationWarning`` instances; these
are demoted with narrow per-test ``@pytest.mark.filterwarnings`` markers.
``matplotlib`` is forced onto the headless ``Agg`` backend because the modeling
import chain pulls ``pyplot``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
import pytest

matplotlib.use('Agg')

from tests.modeling.test_model_selection_tail import ms
from tests.modeling.test_pipeline_manifold import (
    HISTORY_FRAMES as MF_HISTORY_FRAMES,
)
from tests.modeling.test_pipeline_manifold import (
    ContinuousModelingPipeline,
    ContinuousModelRunner,
    _build_manifold_settings,
    _build_signal_continuous_pickle,
    continuous_vocal_manifold_model_selection,
)
from tests.modeling.test_pipeline_multinomial import (
    HISTORY_FRAMES as MN_HISTORY_FRAMES,
)
from tests.modeling.test_pipeline_multinomial import (
    N_CATEGORIES,
    _build_extraction_settings,
    build_multinomial_input_pickle,
    build_univariate_ranking_pickle,
    multinomial_vocal_category_model_selection,
)
from tests.modeling.test_pipeline_multinomial import (
    N_SESSIONS as MN_N_SESSIONS,
)


def _settings_json(tmp_path: Path, settings: dict) -> str:
    """
    Description
    -----------
    Serializes a synthetic ``modeling_settings`` dict to ``tmp_path`` and returns
    its path, so the selectors load the shrunk settings rather than the package
    JSON. Mirrors the helper of the same name in ``test_model_selection_tail``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory.
    settings (dict)
        The settings dictionary to serialize.

    Returns
    -------
    path (str)
        Absolute path to the written JSON file.
    """

    p = tmp_path / 'settings.json'
    p.write_text(json.dumps(settings))
    return str(p)


class _BoomMultinomial:
    """
    Description
    -----------
    Stand-in for ``SmoothMultinomialLogisticRegression`` whose constructor is a
    no-op and whose ``.fit`` always raises, forcing every learned multinomial
    fold (auto-anchor + forward-selection trials) into its per-fold ``except``
    NaN-padding branch. No other estimator method is reached because ``.fit`` is
    the first call inside every fold's ``try`` block.
    """

    def __init__(self, *args, **kwargs):
        """Accept and discard every constructor argument."""

        pass

    def fit(self, *args, **kwargs):
        """Raise unconditionally so the per-fold failure branch executes."""

        raise RuntimeError("forced multinomial fit failure")


class _BoomManifold:
    """
    Description
    -----------
    Stand-in regressor class whose constructor is a no-op and whose ``.fit``
    always raises, forcing every learned manifold fold (auto-anchor + forward-
    selection trials) into the ``_append_failed_fold`` placeholder branch. Used
    as the return value of a monkeypatched ``resolve_manifold_regressor_cls`` so
    the geometry-resolved estimator is replaced wholesale.
    """

    def __init__(self, *args, **kwargs):
        """Accept and discard every constructor argument."""

        pass

    def fit(self, *args, **kwargs):
        """Raise unconditionally so the per-fold failure branch executes."""

        raise RuntimeError("forced manifold fit failure")


class TestMultinomialFoldFailure:
    """Per-fold fit-failure NaN-padding branches of the multinomial selector."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_every_fold_fails_pads_and_finalizes(self, tmp_path, monkeypatch):
        """
        Forcing ``SmoothMultinomialLogisticRegression.fit`` to raise makes every
        learned multinomial fold fail. The Step-0 model-free marginal-prior
        baseline (which fits no model) still establishes and is persisted, so the
        anchored auto-anchor block runs and every fold takes the per-fold
        ``except`` arm: the metric lists gain scalar NaNs, ``weights`` /
        ``intercepts`` gain ``None``, the data arrays become empty-but-shaped, the
        confusion matrix becomes a ``(K, K)`` NaN block, and the audit
        placeholders are appended. The anchor's "every fold errored out" arm then
        fires (no finite AUC), the forward loop's "no finite folds" skip fires for
        every candidate, the REJECT branch persists the empty-candidate step
        pickle, and the final-promotion block runs with the ``null_model_free``
        winner (so no weight reshaping is attempted). The run completes without
        the forced exception propagating and writes at least the Step-0 baseline
        plus a forward-selection step pickle.
        """

        monkeypatch.setattr(ms, 'SmoothMultinomialLogisticRegression', _BoomMultinomial)

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
            test_proportion=0.4,
        )

        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(MN_N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=MN_HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=18,
        ))
        univ_pkl = str(build_univariate_ranking_pickle(
            save_path=tmp_path / 'univariate_combined.pkl',
            feature_names=feature_names,
            n_splits=settings['model_params']['split_num'],
        ))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        multinomial_vocal_category_model_selection(
            univariate_results_path=univ_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_multinomial_*_step_*.pkl'))
        # Step-0 baseline (no model -> succeeds) plus a forward-selection step
        # whose every candidate failed every fold (REJECT branch still persists).
        assert len(step_pkls) >= 2

        # The Step-0 baseline carries the model-free marginal prior; the forward
        # step that follows it must have an empty candidates_summary (every
        # candidate hit "no finite folds") and select nothing.
        with step_pkls[0].open('rb') as fh:
            step0 = pickle.load(fh)
        assert step0['step_idx'] == 0
        assert step0['selected_feature'] == 'null_model_free'

        with step_pkls[-1].open('rb') as fh:
            last_step = pickle.load(fh)
        # No learned candidate produced a finite fold, so nothing was accepted.
        assert last_step['selected_feature'] is None
        assert last_step['candidates_summary'] == {}

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_no_anchor_forward_failure_pads_and_finalizes(self, tmp_path, monkeypatch):
        """
        The same forced-failure regime but with ``use_top_rank_as_anchor=False``,
        so the auto-anchor block is skipped and the forward-selection loop is
        entered straight from the Step-0 baseline. Every forward-selection
        candidate fold still fails (the per-fold ``except`` padding in the forward
        loop fires for each feature), the "no finite folds" skip fires for every
        candidate, the REJECT branch persists the empty-candidate step pickle, and
        the run finalizes on the ``null_model_free`` winner. Exercises the forward
        loop's failure padding independently of the anchor block.
        """

        monkeypatch.setattr(ms, 'SmoothMultinomialLogisticRegression', _BoomMultinomial)

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
            test_proportion=0.4,
        )

        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(MN_N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=MN_HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=16,
        ))
        univ_pkl = str(build_univariate_ranking_pickle(
            save_path=tmp_path / 'univariate_combined.pkl',
            feature_names=feature_names,
            n_splits=settings['model_params']['split_num'],
        ))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        multinomial_vocal_category_model_selection(
            univariate_results_path=univ_pkl,
            input_data_path=input_pkl,
            settings_path=_settings_json(tmp_path, settings),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=False,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_multinomial_*_step_*.pkl'))
        assert len(step_pkls) >= 2
        with step_pkls[-1].open('rb') as fh:
            last_step = pickle.load(fh)
        assert last_step['selected_feature'] is None
        assert last_step['candidates_summary'] == {}


class TestManifoldFoldFailure:
    """Per-fold fit-failure ``_append_failed_fold`` branches of the manifold selector."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_every_fold_fails_pads_and_finalizes(self, tmp_path, monkeypatch):
        """
        Monkeypatching ``resolve_manifold_regressor_cls`` to return a class whose
        ``.fit`` always raises makes every learned manifold fold fail. The
        univariate ranking is still computed by the real runner (whose own
        estimator resolution happens before the patch takes the selector's path),
        so the ranking is schema-correct and at least the signal feature clears
        the screening gate. Inside the selector the Step-0 empirical-density
        baseline (no learned model) still establishes, the auto-anchor block runs
        and every fold takes the ``_append_failed_fold`` placeholder arm (scalar-
        NaN metrics, ``None`` weights / intercepts, empty-but-shaped ``y_true`` /
        ``y_pred_xy`` / ``w_test`` / ``test_indices``, NaN audit), the anchor's
        "every fold errored out" arm fires, the forward loop's "no finite folds"
        skip fires for every candidate, the REJECT branch persists the empty-
        candidate step pickle, and the finalize block runs. The run completes and
        writes at least the Step-0 baseline plus a forward step pickle.
        """

        # Session-holdout with a large session panel: the session-grain screen
        # bootstraps SESSIONS, so the signal feature only clears the gate (and the
        # anchor/forward failure paths this test exercises are only reached) with
        # enough sessions for a tight bootstrap CI.
        gate_n_sessions = 25
        settings, _ = _build_manifold_settings(
            tmp_path, split_strategy='session', split_num=10, test_proportion=0.3,
        )
        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(gate_n_sessions)]

        input_md = {
            'analysis_type': 'continuous',
            'analysis_tag': 'manifold_vae_supercategory',
            # `session_ids` + `n_events_per_session` are required by the manifold
            # session-grain screen (each session owns a contiguous 60-event block,
            # matching `_build_signal_continuous_pickle`'s default `n_per_session`).
            'session_ids': session_ids,
            'n_events_per_session': {sess_id: 60 for sess_id in session_ids},
            'analysis_specific': {
                'usv_category_column_name': 'vae_supercategory',
                'manifold_metric': 'euclidean',
                'manifold_period': 1.0,
            },
        }
        input_pkl = str(_build_signal_continuous_pickle(
            save_path=tmp_path / 'manifold_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=MF_HISTORY_FRAMES,
            input_metadata=input_md,
        ))

        # Build the univariate ranking with the REAL runner (before the selector
        # patch) so the ranking is schema-correct and the signal feature clears
        # the screening gate; the selector itself then resolves the boom class.
        runner = ContinuousModelRunner(
            ContinuousModelingPipeline(modeling_settings_dict=settings)
        )
        combined = {}
        for feature in feature_names:
            combined[feature] = runner.run_univariate_training(input_pkl, feature)
        combined['_input_metadata'] = input_md
        combined_path = tmp_path / 'univariate_combined.pkl'
        with combined_path.open('wb') as fh:
            pickle.dump(combined, fh)

        # Patch only the selector's estimator-class resolution so every learned
        # fold's fit raises; the runner's ranking above is already computed.
        monkeypatch.setattr(
            ms, 'resolve_manifold_regressor_cls', lambda *a, **k: _BoomManifold
        )

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        continuous_vocal_manifold_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=input_pkl,
            output_directory=str(ms_dir),
            settings_path=_settings_json(tmp_path, settings),
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_continuous_manifold_*_step_*.pkl'))
        # Step-0 empirical-density baseline (succeeds) plus a forward step whose
        # every candidate failed every fold (REJECT branch still persists).
        assert len(step_pkls) >= 2

        with step_pkls[0].open('rb') as fh:
            step0 = pickle.load(fh)
        assert step0['selected_feature'] == 'null_model_free'

        with step_pkls[-1].open('rb') as fh:
            last_step = pickle.load(fh)
        # No learned candidate produced a finite fold, so nothing was accepted.
        assert last_step['selected_feature'] is None
        assert last_step['candidates_summary'] == {}
