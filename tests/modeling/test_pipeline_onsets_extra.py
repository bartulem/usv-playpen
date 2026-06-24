"""
@author: bartulem
Branch-coverage tests for the USV vocal-onset modeling pipeline that the
end-to-end smoke suite (``test_pipeline_smoke.py``) deliberately leaves
uncovered.

The smoke suite drives the *happy path* of
``VocalOnsetModelingPipeline.extract_and_save_modeling_input_data`` plus the
fast ``sklearn`` univariate / model-selection dispatchers. This file instead
targets the *alternate* branches of ``modeling_vocal_onsets.py`` directly:

* ``TestInitBranches`` — the constructor's optional / error paths:
  ``modeling_settings_dict=None`` (load shipped JSON), the
  ``feature_boundaries`` capture, the ``history_frames`` ``KeyError`` guard,
  and the ``**kwargs`` attribute injection.

* ``TestDataSplitStrategies`` — every branch of the two-way
  ``create_data_splits`` generator that the smoke suite never reaches:
  ``strategy_override``, the ``mixed`` empty-class and no-balanced-train
  early-outs, the ``session`` too-high-test-proportion and no-balanced-train
  skips, and the unknown-strategy ``ValueError``.

* ``TestPygamEngine`` — the entire ``_run_model_for_feature_pygam`` method
  (never touched by the sklearn-only smoke suite), under both the ``mixed``
  and ``session`` split strategies; the null is a per-split label-shuffle of
  the training labels in both cases.

* ``TestExtractionGuards`` — the data-prep guard / error paths in
  ``extract_and_save_modeling_input_data``: the empty-target-bout session
  removal, the dyad-renamed bare-suffix generic-key branch, the
  "no features selected" ``ValueError``, and the save-failure ``except``.

All artifacts live strictly under the pytest ``tmp_path`` scratch directory;
nothing is ever written into the package ``src/`` tree (the session-end
integrity check would otherwise fail the suite). In-memory ``feature_data``
dictionaries (built locally, NOT via ``_synth``) feed the split / pygam paths
so those branches run without any on-disk extraction.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``; the
top-level modeling imports are wrapped in a ``warnings.catch_warnings`` block
that ignores ``DeprecationWarning`` during import. ``pygam`` (Python 3.13)
emits ``DeprecationWarning: Bitwise inversion '~' on bool`` from inside its GAM
fit, demoted with a narrow per-test marker; ``astropy``'s Gaussian smoothing
emits an ``AstropyUserWarning`` on the extraction path, demoted the same way.
``matplotlib`` is forced onto the headless ``Agg`` backend in case any imported
plotting code is pulled in transitively.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from tests.modeling._synth import (
    build_behavioral_features_csv,
    build_modeling_settings,
    build_session_tree,
    build_track_h5,
    build_usv_summary_csv,
    write_session_list_file,
)

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level imports so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.modeling_vocal_onsets import VocalOnsetModelingPipeline


# Tiny-data geometry shared across the direct-call (split / pygam) tests. These
# constants are intentionally smaller than the smoke suite's so the GAM fit
# stays fast; the history window width (HISTORY_FRAMES) is the column count of
# every per-event array and must equal floor(CAMERA_FPS * FILTER_HISTORY).
CAMERA_FPS = 30.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))


def _make_feature_data(
        session_ids: list[str],
        history_frames: int = HISTORY_FRAMES,
        n_usv: int = 24,
        n_no_usv: int = 60,
        pos_mean: float = 0.6,
        neg_mean: float = -0.6,
        seed: int = 0,
) -> dict:
    """
    Description
    -----------
    Builds a single-feature ``{session_id: {usv_feature_arr, no_usv_feature_arr}}``
    dictionary in memory (no disk I/O), matching the contract that
    ``create_data_splits`` and the per-feature model runners consume.

    Every session receives ``n_usv`` positive and ``n_no_usv`` negative event
    windows, each ``history_frames`` wide. The two classes carry separable means
    (``pos_mean`` vs ``neg_mean``) plus Gaussian noise so a fitted classifier can
    find above-chance structure on the ``usv`` vs ``no_usv`` contrast.

    Parameters
    ----------
    session_ids (list of str)
        Session identifiers to populate.
    history_frames (int)
        Number of temporal lags (columns) per event window.
    n_usv (int)
        Positive (USV) events per session.
    n_no_usv (int)
        Negative (No-USV) events per session.
    pos_mean (float)
        Additive mean of the positive-class windows.
    neg_mean (float)
        Additive mean of the negative-class windows.
    seed (int)
        Base seed for the per-session RNG (each session offsets it).

    Returns
    -------
    feature_data (dict)
        ``{session_id: {'usv_feature_arr': ndarray, 'no_usv_feature_arr': ndarray}}``.
    """

    feature_data: dict = {}
    for s_idx, sess in enumerate(session_ids):
        rng = np.random.default_rng(seed + s_idx)
        usv_arr = (pos_mean + rng.standard_normal((n_usv, history_frames))).astype(float)
        no_usv_arr = (neg_mean + rng.standard_normal((n_no_usv, history_frames))).astype(float)
        feature_data[sess] = {
            'usv_feature_arr': usv_arr,
            'no_usv_feature_arr': no_usv_arr,
        }
    return feature_data


def _pipeline(tmp_path: Path, **overrides) -> VocalOnsetModelingPipeline:
    """
    Description
    -----------
    Constructs a ``VocalOnsetModelingPipeline`` from a trimmed synthetic
    settings dict (no session tree required) so the split / pygam runner methods
    can be exercised directly on in-memory ``feature_data``.

    A throwaway session-list file under ``tmp_path`` is written purely to
    satisfy ``build_modeling_settings``' ``io.session_list_file`` key; it is
    never read because these tests never invoke the on-disk extraction.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        The per-test scratch directory; all artifacts live below it.
    overrides (dict)
        Keyword arguments forwarded to ``build_modeling_settings`` (e.g.
        ``model_engine``, ``split_strategy``, ``split_num``).

    Returns
    -------
    pipeline (VocalOnsetModelingPipeline)
        A constructed pipeline whose ``history_frames`` equals ``HISTORY_FRAMES``.
    """

    list_file = write_session_list_file([tmp_path / 'unused_session'], tmp_path / 'session_list.txt')
    save_dir = tmp_path / 'out'
    save_dir.mkdir(parents=True, exist_ok=True)
    settings = build_modeling_settings(
        session_list_file=list_file,
        save_directory=save_dir,
        camera_sampling_rate=CAMERA_FPS,
        filter_history=FILTER_HISTORY,
        **overrides,
    )
    settings['model_params']['usv_bout_time'] = FILTER_HISTORY
    return VocalOnsetModelingPipeline(modeling_settings_dict=settings)


def _drain(generator) -> list:
    """
    Description
    -----------
    Materializes a ``create_data_splits`` generator into a list of
    ``(X_train, y_train, X_test, y_test)`` tuples, forcing every branch inside
    the generator body to execute (generators are lazy, so simply constructing
    one runs no code).

    Parameters
    ----------
    generator (collections.abc.Generator)
        The lazy split generator returned by ``create_data_splits``.

    Returns
    -------
    splits (list of tuple)
        All yielded ``(X_train, y_train, X_test, y_test)`` tuples.
    """

    return list(generator)


class TestInitBranches:
    """Constructor optional / error paths not hit by the happy-path smoke suite."""

    def test_init_from_none_loads_shipped_settings(self):
        """
        Passing ``modeling_settings_dict=None`` resolves and loads the shipped
        ``modeling_settings.json`` relative to the package, then derives
        ``history_frames`` from its ``io`` / ``model_params`` blocks — the
        default-config branch the smoke suite (which always injects a dict)
        never touches. Read-only: nothing is written.
        """

        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=None)
        assert isinstance(pipeline.history_frames, int)
        assert pipeline.history_frames >= 0
        assert 'model_params' in pipeline.modeling_settings

    def test_init_captures_feature_boundaries_and_kwargs(self, tmp_path):
        """
        When the settings carry the optional top-level ``feature_boundaries``
        key it is captured onto the instance, and any extra ``**kwargs`` are
        injected as attributes — both side branches of ``__init__``.
        """

        pipeline = _pipeline(tmp_path)
        pipeline.modeling_settings['feature_boundaries'] = {'speed': [0.0, 5.0]}
        # Re-construct so the constructor re-reads the now-present key.
        rebuilt = VocalOnsetModelingPipeline(
            modeling_settings_dict=pipeline.modeling_settings,
            sentinel_attr='set_via_kwargs',
        )
        assert rebuilt.feature_boundaries == {'speed': [0.0, 5.0]}
        assert rebuilt.sentinel_attr == 'set_via_kwargs'

    def test_init_missing_history_setting_raises_keyerror(self, tmp_path):
        """
        Dropping ``model_params.filter_history`` makes the ``history_frames``
        computation raise; the constructor re-raises it as a wrapped
        ``KeyError`` (the guard branch around the floor computation).
        """

        pipeline = _pipeline(tmp_path)
        broken = pipeline.modeling_settings
        del broken['model_params']['filter_history']
        with pytest.raises(KeyError):
            VocalOnsetModelingPipeline(modeling_settings_dict=broken)


class TestDataSplitStrategies:
    """Every branch of the two-way ``create_data_splits`` generator."""

    def test_mixed_strategy_yields_balanced_train_natural_test(self, tmp_path):
        """
        The default ``mixed`` strategy pools all sessions, stratified-splits at
        the natural rate, and balances only the training fold 50/50. Asserts the
        training fold is class-balanced while the test fold preserves the
        natural (imbalanced) prior.
        """

        pipeline = _pipeline(tmp_path, split_strategy='mixed', split_num=2)
        feature_data = _make_feature_data([f'session_{i}' for i in range(3)])
        splits = _drain(pipeline.create_data_splits(feature_data))
        assert len(splits) == 2
        X_train, y_train, _X_test, y_test = splits[0]
        assert X_train.shape[1] == HISTORY_FRAMES
        # Balanced training fold: equal positives and negatives.
        assert int(np.sum(y_train == 1)) == int(np.sum(y_train == 0))
        # Natural-rate test fold: more negatives than positives (the source
        # ratio was 24:60 per session).
        assert int(np.sum(y_test == 0)) > int(np.sum(y_test == 1))

    def test_strategy_override_argument_is_honored(self, tmp_path):
        """
        Passing ``strategy_override`` supersedes the settings strategy: settings
        say ``mixed`` but the override forces ``session``, so the session-level
        ShuffleSplit path runs instead (covers the override assignment branch).
        """

        pipeline = _pipeline(
            tmp_path, split_strategy='mixed', split_num=2, test_proportion=0.5,
        )
        feature_data = _make_feature_data([f'session_{i}' for i in range(4)])
        splits = _drain(
            pipeline.create_data_splits(feature_data, strategy_override='session')
        )
        assert len(splits) >= 1
        for X_train, y_train, X_test, y_test in splits:
            assert X_train.shape[1] == HISTORY_FRAMES
            assert int(np.sum(y_train == 1)) == int(np.sum(y_train == 0))

    def test_mixed_strategy_empty_class_skips(self, tmp_path):
        """
        With a positive class entirely absent, ``mixed`` hits the
        ``n_pos_total == 0`` early-return and yields nothing.
        """

        pipeline = _pipeline(tmp_path, split_strategy='mixed', split_num=2)
        feature_data = _make_feature_data(['session_0'], n_usv=0, n_no_usv=40)
        # Zero-positive arrays still need the right column count for pooling.
        feature_data['session_0']['usv_feature_arr'] = np.empty((0, HISTORY_FRAMES))
        splits = _drain(pipeline.create_data_splits(feature_data))
        assert splits == []

    def test_session_strategy_too_high_proportion_skips(self, tmp_path):
        """
        A single session cannot be split into train + test, so the ``session``
        strategy's ``n_sessions * (1 - test_proportion) < 1`` guard returns
        without yielding.
        """

        pipeline = _pipeline(
            tmp_path, split_strategy='session', split_num=2, test_proportion=0.9,
        )
        feature_data = _make_feature_data(['session_0'])
        splits = _drain(pipeline.create_data_splits(feature_data))
        assert splits == []

    def test_session_strategy_skips_split_with_unbalanced_train(self, tmp_path):
        """
        When a training session contributes zero positives, the per-split
        balance step produces an empty balanced training set and the ``session``
        strategy skips that split (``X_pos_train_bal.shape[0] == 0``). Built so
        at least one session is positive-free; asserts the run completes (some
        splits may be skipped, the rest yield balanced training folds).
        """

        pipeline = _pipeline(
            tmp_path, split_strategy='session', split_num=3, test_proportion=0.34,
            random_seed=1,
        )
        # session_0 carries the only positives; any split that routes it to the
        # test side leaves the training pool positive-free -> skipped.
        feature_data = _make_feature_data([f'session_{i}' for i in range(3)])
        for sess in ('session_1', 'session_2'):
            feature_data[sess]['usv_feature_arr'] = np.empty((0, HISTORY_FRAMES))
        splits = _drain(pipeline.create_data_splits(feature_data))
        for X_train, y_train, X_test, y_test in splits:
            assert int(np.sum(y_train == 1)) == int(np.sum(y_train == 0))

    def test_unknown_strategy_raises_value_error(self, tmp_path):
        """
        An unrecognized ``split_strategy`` falls through to the terminal
        ``ValueError`` (the strategy-dispatch ``else`` branch).
        """

        pipeline = _pipeline(tmp_path, split_strategy='mixed', split_num=2)
        feature_data = _make_feature_data(['session_0', 'session_1'])
        with pytest.raises(ValueError, match='Unknown split_strategy'):
            _drain(pipeline.create_data_splits(feature_data, strategy_override='nonsense'))


class TestSklearnRunner:
    """The univariate sklearn runner called directly across split strategies."""

    def test_sklearn_runner_mixed_produces_branch_metrics(self, tmp_path):
        """
        ``_run_model_for_feature_sklearn`` (called directly, bypassing the
        dispatcher) projects each split onto an identity basis, fits a
        ``LogisticRegressionCV`` plus a label-shuffled null per split, and fills
        both branches' scalar-metric / filter-shape arrays. Asserts the full key
        set is present, sized to ``split_num`` and ``HISTORY_FRAMES``, with at
        least one finite fitted fold (strong-signal data).
        """

        pipeline = _pipeline(tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2)
        feature_data = _make_feature_data(
            [f'session_{i}' for i in range(3)], n_usv=30, n_no_usv=60,
        )
        basis = np.eye(HISTORY_FRAMES, dtype=float)
        feat_name, results = pipeline._run_model_for_feature_sklearn(
            feature_name='self.speed', feature_data=feature_data, basis_matrix=basis,
        )
        assert feat_name == 'self.speed'
        for branch in ('actual', 'null'):
            for metric in ('ll', 'auc', 'score', 'brier', 'ece', 'mcc', 'recall', 'f1',
                           'optimal_C', 'n_iter', 'converged', 'fit_time'):
                assert results[branch][metric].shape == (2,)
            assert results[branch]['filter_shapes'].shape == (2, HISTORY_FRAMES)
            assert results[branch]['confusion_matrix'].shape == (2, 2, 2)
        assert np.isfinite(results['actual']['auc']).any()

    def test_sklearn_runner_no_valid_splits_reports_empty(self, tmp_path):
        """
        A single session with too-high test proportion starves the session
        splitter so no fold has data: the sklearn runner takes the
        ``no valid splits`` reporting branch and returns all-NaN AUC.
        """

        pipeline = _pipeline(
            tmp_path, model_engine='sklearn', split_strategy='session', split_num=2,
            test_proportion=0.95,
        )
        feature_data = _make_feature_data(['session_0'], n_usv=20, n_no_usv=40)
        basis = np.eye(HISTORY_FRAMES, dtype=float)
        feat_name, results = pipeline._run_model_for_feature_sklearn(
            feature_name='self.speed', feature_data=feature_data, basis_matrix=basis,
        )
        assert feat_name == 'self.speed'
        assert np.isnan(results['actual']['auc']).all()


class TestPygamEngine:
    """The full ``_run_model_for_feature_pygam`` path (never run by the sklearn smoke suite)."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_pygam_mixed_strategy_produces_branch_metrics(self, tmp_path):
        """
        ``_run_model_for_feature_pygam`` under the ``mixed`` split strategy fits
        a tensor-product-spline ``LogisticGAM`` per split, with the null fit on
        a per-split label-shuffle of the training labels (the same
        permutation-test null the sklearn engine uses).
        Asserts the returned ``actual`` / ``null`` branches carry the expected
        scalar-metric arrays sized to ``split_num`` and the per-split filter
        shapes sized to ``HISTORY_FRAMES``, with at least one finite fitted fold.
        """

        pipeline = _pipeline(tmp_path, model_engine='pygam', split_strategy='mixed', split_num=2)
        feature_data = _make_feature_data(
            [f'session_{i}' for i in range(3)], n_usv=24, n_no_usv=60,
        )
        feat_name, results = pipeline._run_model_for_feature_pygam(
            feature_name='self.speed', feature_data=feature_data, basis_matrix=None,
        )
        assert feat_name == 'self.speed'
        for branch in ('actual', 'null'):
            for metric in ('ll', 'auc', 'score', 'brier', 'ece', 'mcc'):
                assert results[branch][metric].shape == (2,)
            assert results[branch]['filter_shapes'].shape == (2, HISTORY_FRAMES)
        assert np.isfinite(results['actual']['auc']).any()

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_pygam_session_strategy_produces_branch_metrics(self, tmp_path):
        """
        Under the ``session`` split strategy the pygam runner fits the actual
        GAM on session-held-out folds and the null on a per-split label-shuffle
        of the training labels. Asserts the run completes and the actual branch
        has at least one finite fitted fold.
        """

        pipeline = _pipeline(
            tmp_path, model_engine='pygam', split_strategy='session', split_num=2,
            test_proportion=0.5,
        )
        feature_data = _make_feature_data(
            [f'session_{i}' for i in range(4)], n_usv=20, n_no_usv=80,
        )
        feat_name, results = pipeline._run_model_for_feature_pygam(
            feature_name='other.speed', feature_data=feature_data, basis_matrix=None,
        )
        assert feat_name == 'other.speed'
        assert results['actual']['filter_shapes'].shape == (2, HISTORY_FRAMES)
        assert np.isfinite(results['actual']['auc']).any()

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    def test_pygam_gcv_when_lam_none(self, tmp_path):
        """
        Setting ``lam_penalty`` to ``None`` selects the GCV branch
        (``"Using GCV to find optimal smoothness"``) for the actual GAM instead
        of a fixed smoothness penalty. Uses tiny data + few iterations so GCV
        stays fast; asserts the run completes. On degenerate tiny data the GCV
        fit may leave every fold NaN, so the terminal ``np.nanmean`` "Mean of
        empty slice" ``RuntimeWarning`` is demoted for this test.
        """

        pipeline = _pipeline(tmp_path, model_engine='pygam', split_strategy='mixed', split_num=1)
        pipeline.modeling_settings['hyperparameters']['classical']['pygam']['lam_penalty'] = None
        feature_data = _make_feature_data(
            [f'session_{i}' for i in range(2)], n_usv=16, n_no_usv=32,
        )
        feat_name, results = pipeline._run_model_for_feature_pygam(
            feature_name='self.speed', feature_data=feature_data, basis_matrix=None,
        )
        assert feat_name == 'self.speed'
        assert results['actual']['filter_shapes'].shape == (1, HISTORY_FRAMES)

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    def test_pygam_missing_params_uses_hardcoded_fallback(self, tmp_path):
        """
        Removing the ``pygam`` hyperparameter block makes the param read raise
        ``KeyError``, so the runner falls back to its hard-coded defaults
        (8/5 splines, lam=0.6, 100 iters). Tiny data keeps the fallback fit
        bounded; asserts the run completes.
        """

        pipeline = _pipeline(tmp_path, model_engine='pygam', split_strategy='mixed', split_num=1)
        del pipeline.modeling_settings['hyperparameters']['classical']['pygam']
        feature_data = _make_feature_data(['session_0', 'session_1'], n_usv=14, n_no_usv=28)
        feat_name, results = pipeline._run_model_for_feature_pygam(
            feature_name='self.speed', feature_data=feature_data, basis_matrix=None,
        )
        assert feat_name == 'self.speed'
        assert results['actual']['filter_shapes'].shape == (1, HISTORY_FRAMES)

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    def test_pygam_no_valid_splits_reports_empty(self, tmp_path):
        """
        Feeding a single session with too-high test proportion starves both the
        actual and null session splitters, so no fold has data: the pygam runner
        takes the ``no valid splits`` reporting branch and returns all-NaN
        metric arrays.
        """

        pipeline = _pipeline(
            tmp_path, model_engine='pygam', split_strategy='session', split_num=2,
            test_proportion=0.95,
        )
        feature_data = _make_feature_data(['session_0'], n_usv=20, n_no_usv=40)
        feat_name, results = pipeline._run_model_for_feature_pygam(
            feature_name='self.speed', feature_data=feature_data, basis_matrix=None,
        )
        assert feat_name == 'self.speed'
        assert np.isnan(results['actual']['auc']).all()


class TestExtractionGuards:
    """Guard / error paths inside ``extract_and_save_modeling_input_data``."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_drops_empty_target_bout_session(self, tmp_path):
        """
        A session whose target mouse emits *no* USVs (only the partner does)
        yields zero positive events and is dropped by the
        ``identify_empty_event_sessions`` removal branch, while the valid
        sessions still produce a non-empty input pickle. A partner vocal
        predictor (``categories_rate``) is enabled so the ``if new_voc_cols``
        column-attachment branch and the vocal-column metadata path also run.
        """

        base = tmp_path / 'sessions'
        # Two healthy sessions (target emits bouts) plus one silent-target
        # session built by hand so its target mouse never vocalizes.
        session_roots = build_session_tree(
            base_dir=base,
            n_sessions=2,
            n_frames=3600,
            camera_fps=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
            n_bouts=8,
            usv_per_bout=3,
        )

        silent_root = base / 'session_silent'
        silent_root.mkdir(parents=True, exist_ok=True)
        silent_mice = ['ssil_m_male', 'ssil_m_female']
        build_behavioral_features_csv(
            session_root=silent_root,
            mouse_names=silent_mice,
            n_frames=3600,
            egocentric_features=['speed'],
            seed=99,
        )
        build_track_h5(
            session_root=silent_root,
            mouse_names=silent_mice,
            camera_fps=CAMERA_FPS,
        )
        # Emitter is the PARTNER (index 1) for both, leaving the target (index 0)
        # without a single vocalization -> zero positive events.
        build_usv_summary_csv(
            session_root=silent_root,
            target_mouse=silent_mice[1],
            partner_mouse=silent_mice[1],
            camera_fps=CAMERA_FPS,
            n_frames=3600,
            filter_history=FILTER_HISTORY,
            n_bouts=6,
            usv_per_bout=3,
            seed=99,
        )
        session_roots.append(silent_root)

        list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
        save_dir = tmp_path / 'out'
        save_dir.mkdir(parents=True, exist_ok=True)
        settings = build_modeling_settings(
            session_list_file=list_file,
            save_directory=save_dir,
            camera_sampling_rate=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
            usv_predictor_type='categories_rate',
        )
        settings['model_params']['usv_bout_time'] = FILTER_HISTORY

        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_modeling_input_data()

        pkls = list(save_dir.glob('modeling_*.pkl'))
        assert len(pkls) == 1

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_save_failure_is_caught(self, tmp_path, monkeypatch):
        """
        When the terminal ``pickle.dump`` raises, the extraction's save ``try``
        swallows it and prints an error rather than propagating — the
        save-failure ``except`` branch. The module-level ``pickle.dump`` is
        monkeypatched to raise; everything upstream still runs to completion.
        """

        session_roots = build_session_tree(
            base_dir=tmp_path / 'sessions',
            n_sessions=2,
            n_frames=3600,
            camera_fps=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
            n_bouts=8,
            usv_per_bout=3,
        )
        list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
        save_dir = tmp_path / 'out'
        save_dir.mkdir(parents=True, exist_ok=True)
        settings = build_modeling_settings(
            session_list_file=list_file,
            save_directory=save_dir,
            camera_sampling_rate=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
        )
        settings['model_params']['usv_bout_time'] = FILTER_HISTORY

        import usv_playpen.modeling.modeling_vocal_onsets as onsets_mod

        def _boom(*_args, **_kwargs):
            raise OSError("simulated pickle write failure")

        monkeypatch.setattr(onsets_mod.pickle, 'dump', _boom)

        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
        # Must NOT raise: the save failure is caught and logged. The
        # ``save_path.open('wb')`` may leave a zero-byte file behind (the failure
        # happens during ``pickle.dump``), so any artifact present is empty.
        pipeline.extract_and_save_modeling_input_data()
        for pkl in save_dir.glob('modeling_*.pkl'):
            assert pkl.stat().st_size == 0

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_no_features_raises_value_error(self, tmp_path):
        """
        Configuring every kinematic bucket empty (and no vocal predictor) leaves
        zero behavioral predictors after harmonization, tripping the
        ``"No features selected."`` ``ValueError`` guard.
        """

        session_roots = build_session_tree(
            base_dir=tmp_path / 'sessions',
            n_sessions=2,
            n_frames=3600,
            camera_fps=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
            n_bouts=8,
            usv_per_bout=3,
        )
        list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
        save_dir = tmp_path / 'out'
        save_dir.mkdir(parents=True, exist_ok=True)
        settings = build_modeling_settings(
            session_list_file=list_file,
            save_directory=save_dir,
            camera_sampling_rate=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=[],
            dyadic_features=[],
            engagement_features=[],
        )
        settings['model_params']['usv_bout_time'] = FILTER_HISTORY

        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
        with pytest.raises(ValueError, match='No features selected'):
            pipeline.extract_and_save_modeling_input_data()

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_category_tag_and_metadata(self, tmp_path):
        """
        With ``model_target_vocal_type='individual'`` and an
        ``onset_target_category`` set, the saved input pickle's filename and
        ``_input_metadata`` carry a category-aware ``analysis_tag`` that embeds
        BOTH the category column name and the index (so VAE-vs-QLVM and
        category-vs-supercategory are unambiguous downstream), and
        ``analysis_specific`` records the category provenance. The synthetic
        summaries write ``vae_supercategory == 1`` for every USV, so targeting
        category 1 keeps every onset and both sessions survive.
        """

        session_roots = build_session_tree(
            base_dir=tmp_path / 'sessions',
            n_sessions=2,
            n_frames=3600,
            camera_fps=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
            n_bouts=8,
            usv_per_bout=3,
        )
        list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
        save_dir = tmp_path / 'out'
        save_dir.mkdir(parents=True, exist_ok=True)
        settings = build_modeling_settings(
            session_list_file=list_file,
            save_directory=save_dir,
            camera_sampling_rate=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
            egocentric_features=['speed'],
        )
        settings['model_params']['usv_bout_time'] = FILTER_HISTORY
        settings['model_params']['model_target_vocal_type'] = 'individual'
        settings['model_params']['onset_target_category'] = 1

        pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_modeling_input_data()

        pkls = list(save_dir.glob('modeling_*.pkl'))
        assert len(pkls) == 1
        expected_tag = 'individual_cat_vae_supercategory_1'
        assert expected_tag in pkls[0].name

        with pkls[0].open('rb') as fh:
            artifact = pickle.load(fh)
        md = artifact['_input_metadata']
        assert md['analysis_tag'] == expected_tag
        assert md['analysis_specific']['onset_target_category'] == 1
        assert md['analysis_specific']['usv_category_column_name'] == 'vae_supercategory'
