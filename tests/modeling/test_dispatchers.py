"""
@author: bartulem
Unit tests for the two HPC entry-point dispatchers —
``usv_playpen.modeling.main_univariate_dispatcher`` (per-feature SLURM
array driver) and ``usv_playpen.modeling.main_model_selection_dispatcher``
(forward-stepwise selection driver).

Both dispatchers are thin routers: they resolve the shipped
``modeling_settings.json`` relative to the package, validate inputs, map
an ``analysis_type`` onto a concrete pipeline / selector, and persist (or
delegate the persistence of) the result. The heavy numerical work lives
in the pipeline / selector callables, so these tests stub those callables
out (via ``mocker.patch.object`` on the dispatcher module) and assert the
*routing and validation contract* only.

Coverage:

  * ``get_basis_matrix_standardized`` — the basis branch matrix and the
    once-per-run atomic plotting lock.
  * ``dispatch_univariate_job`` — settings load, feature-index mapping
    (in/out of bounds), the five ``analysis_type`` routes, the
    legacy-input warning, the per-feature output filename + embedded
    metadata blocks, and the failure modes (bad settings, bad index,
    pipeline exception).
  * ``validate_paths`` — the missing-mount-point guard.
  * ``dispatch_model_selection`` — the five selector routes (asserting the
    exact kwargs each receives), the unknown-type branch, and the
    catch-all exception handler.
  * both ``__main__`` CLI blocks, driven in-process with ``runpy``.

Every test is self-contained under ``tmp_path``; nothing is written into
``src/``. Synthetic ``modeling_settings`` dictionaries come from the
shared ``_synth.build_modeling_settings`` builder (which deep-copies the
real shipped JSON, so every key the metadata builders read is present)
and are injected by patching the dispatcher module's ``json.load`` —
matching the dispatchers' hardcoded package-relative JSON path.

The dispatcher modules transitively import the JAX/optax stack, which
emits a ``DeprecationWarning`` at import time; under the project's
``filterwarnings = ["error"]`` policy that would abort collection, so the
two dispatcher modules are imported inside a ``warnings.catch_warnings``
block here.
"""

from __future__ import annotations

import argparse
import json
import pickle
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from . import _synth

# The dispatcher modules pull in the JAX/optax stack at import time, which
# raises a DeprecationWarning that the global `filterwarnings = ["error"]`
# policy would otherwise turn into a hard collection failure. Import them
# under a suppressing `catch_warnings` block (the warning is third-party
# and orthogonal to anything these tests exercise).
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from usv_playpen.modeling import main_model_selection_dispatcher as sel
    from usv_playpen.modeling import main_univariate_dispatcher as uni
    from usv_playpen.modeling import model_selection as msrc
    from usv_playpen.os_utils import resolve_modeling_setting

REPO_SETTINGS = Path(__file__).resolve().parents[2] / 'src' / 'usv_playpen' / '_parameter_settings' / 'modeling_settings.json'


# Helpers


def _settings(tmp_path, model_engine='sklearn'):
    """
    Description
    -----------
    Build a synthetic ``modeling_settings`` dict (deep copy of the
    shipped JSON) repointed at scratch paths under ``tmp_path``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory; the session list and save directory go here.
    model_engine (str)
        Value for the settings' model engine ('sklearn', 'pygam', ...).

    Returns
    -------
    settings (dict)
        The modeling-settings dictionary from ``_synth.build_modeling_settings``.
    """

    session_list = _synth.write_session_list_file(
        [tmp_path / 'session_0'],
        tmp_path / 'sessions_list_intact_partners_male.txt',
    )
    return _synth.build_modeling_settings(
        session_list_file=session_list,
        save_directory=tmp_path / 'save',
        model_engine=model_engine,
    )


def _input_pickle(tmp_path, features, with_metadata=True):
    """
    Description
    -----------
    Write a tiny modeling-input pickle (feature keys plus an optional
    ``_input_metadata`` block) and return its path.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory; the pickle is written as ``input.pkl``.
    features (list of str)
        Feature keys, each holding one session of zero USV / no-USV arrays.
    with_metadata (bool)
        Whether to add the ``_input_metadata`` block (False exercises the
        legacy-input path).

    Returns
    -------
    path (pathlib.Path)
        Path of the written pickle.
    """

    artifact = {feat: {'session_0': {'usv_feature_arr': np.zeros((2, 3)),
                                     'no_usv_feature_arr': np.zeros((2, 3))}}
                for feat in features}
    if with_metadata:
        artifact['_input_metadata'] = {'analysis_tag': 'bout',
                                       'experimental_condition': 'intact_partners_male'}
    path = tmp_path / 'input.pkl'
    with path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return path


def _uni_args(analysis_type, feature_idx, input_path, output_dir):
    """
    Description
    -----------
    Build the argparse Namespace the univariate dispatcher consumes.

    Parameters
    ----------
    analysis_type (str)
        Route name ('onset', 'category', 'params', 'multinomial', 'continuous', ...).
    feature_idx (int)
        SLURM array index selecting the feature.
    input_path (pathlib.Path)
        Modeling-input pickle path.
    output_dir (pathlib.Path)
        Output directory for the per-feature result.

    Returns
    -------
    args (argparse.Namespace)
        Namespace with ``analysis_type``, ``feature_idx``, ``input_data`` and ``output_dir``.
    """

    return argparse.Namespace(
        analysis_type=analysis_type,
        feature_idx=feature_idx,
        input_data=str(input_path),
        output_dir=str(output_dir),
    )


def _sel_args(analysis_type, tmp_path, **overrides):
    """
    Description
    -----------
    Build the argparse Namespace the selection dispatcher consumes,
    with all required path slots pointing at real files under
    ``tmp_path`` so ``validate_paths`` passes.

    Parameters
    ----------
    analysis_type (str)
        Route name passed through to the dispatcher.
    tmp_path (pathlib.Path)
        Per-test scratch directory; placeholder univariate / input files are written here.
    **overrides
        Attribute overrides applied to the Namespace after construction.

    Returns
    -------
    args (argparse.Namespace)
        Namespace with the path slots, ``output_dir`` and ``anchor=True``.
    """

    univ = tmp_path / 'univariate.pkl'
    inp = tmp_path / 'input.pkl'
    univ.write_bytes(b'x')
    inp.write_bytes(b'x')
    ns = argparse.Namespace(
        analysis_type=analysis_type,
        univariate_path=str(univ),
        input_path=str(inp),
        output_dir=str(tmp_path / 'out'),
        anchor=True,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# get_basis_matrix_standardized


class TestBasisMatrix:
    """``get_basis_matrix_standardized``: basis branches and the once-per-array plotting lock."""

    def test_pygam_engine_returns_none(self, tmp_path):
        """When the engine is ``pygam`` the function short-circuits to
        ``None`` (pygam uses its own internal splines)."""

        settings = _settings(tmp_path, model_engine='pygam')
        out = uni.get_basis_matrix_standardized(settings, history_frames=8,
                                                output_dir=str(tmp_path))
        assert out is None

    def test_identity_basis_builds_matrix_and_plots_once(self, tmp_path):
        """The ``identity`` basis yields a square matrix and the atomic
        lock-plus-plot side effect fires exactly once: the first call
        writes both the lock file and the verification PNG."""

        settings = _settings(tmp_path, model_engine='sklearn')
        settings['model_params']['model_basis_function'] = 'identity'
        out_dir = tmp_path / 'basis_out'
        out_dir.mkdir()

        mat = uni.get_basis_matrix_standardized(settings, history_frames=6,
                                                output_dir=str(out_dir))
        assert isinstance(mat, np.ndarray)
        assert mat.shape[0] == 6
        assert (out_dir / '.basis_plotted').exists()
        assert (out_dir / 'basis_verification.png').exists()

    @pytest.mark.parametrize('basis_name', ['raised_cosine', 'bspline', 'laplacian_pyramid'])
    def test_parametric_basis_branches_build_matrix(self, tmp_path, basis_name):
        """The raised-cosine, B-spline, and laplacian-pyramid branches read
        their hyperparameter blocks and return a matrix with ``history_frames``
        rows. A wide history (90 frames) is used so the shipped B-spline
        config (32 splines, degree 3) keeps every knot distinct — at a
        short width the integer-cast knots collapse and ``_normalizecols``
        divides a zero column (RuntimeWarning). The dispatcher passes
        ``step=p['step']`` to ``laplacian_pyramid``, so that branch builds
        normally (it was previously thought dead due to a missing ``step``)."""

        settings = _settings(tmp_path, model_engine='sklearn')
        settings['model_params']['model_basis_function'] = basis_name
        out_dir = tmp_path / f'basis_{basis_name}'
        out_dir.mkdir()
        mat = uni.get_basis_matrix_standardized(settings, history_frames=90,
                                                output_dir=str(out_dir))
        assert isinstance(mat, np.ndarray)
        assert mat.shape[0] == 90

    def test_existing_lock_skips_plot(self, tmp_path):
        """A pre-existing ``.basis_plotted`` lock suppresses the plot so
        only the first job in a SLURM array renders it."""

        settings = _settings(tmp_path, model_engine='sklearn')
        settings['model_params']['model_basis_function'] = 'identity'
        out_dir = tmp_path / 'basis_out'
        out_dir.mkdir()
        (out_dir / '.basis_plotted').touch()

        mat = uni.get_basis_matrix_standardized(settings, history_frames=5,
                                                output_dir=str(out_dir))
        assert isinstance(mat, np.ndarray)
        assert not (out_dir / 'basis_verification.png').exists()

    def test_plot_bool_toggles_parametric_basis_plot(self, tmp_path):
        """A configurable basis honours its ``plot_bool``: ``raised_cosine``
        with ``plot_bool=False`` builds the matrix but writes NO verification
        PNG, while ``plot_bool=True`` writes it. (The ``identity`` basis has no
        ``plot_bool`` and always plots — covered above.)"""

        out_off = tmp_path / 'off'
        out_off.mkdir()
        s_off = _settings(tmp_path, model_engine='sklearn')
        s_off['model_params']['model_basis_function'] = 'raised_cosine'
        s_off['hyperparameters']['basis_functions']['raised_cosine']['plot_bool'] = False
        mat = uni.get_basis_matrix_standardized(s_off, history_frames=90,
                                                output_dir=str(out_off))
        assert isinstance(mat, np.ndarray)
        assert not (out_off / 'basis_verification.png').exists()

        out_on = tmp_path / 'on'
        out_on.mkdir()
        s_on = _settings(tmp_path, model_engine='sklearn')
        s_on['model_params']['model_basis_function'] = 'raised_cosine'
        s_on['hyperparameters']['basis_functions']['raised_cosine']['plot_bool'] = True
        uni.get_basis_matrix_standardized(s_on, history_frames=90,
                                          output_dir=str(out_on))
        assert (out_on / 'basis_verification.png').exists()


# dispatch_univariate_job


class TestDispatchUnivariate:
    """``dispatch_univariate_job``: settings, feature mapping, routing, output and failure modes."""

    def test_settings_load_failure_returns_early(self, tmp_path, mocker, capsys):
        """A settings-JSON load failure is caught, printed as ``FATAL``,
        and the function returns without touching the input pickle."""

        mocker.patch.object(uni.json, 'load', side_effect=ValueError('boom'))
        args = _uni_args('onset', 0, tmp_path / 'unused.pkl', tmp_path / 'out')
        uni.dispatch_univariate_job(args)
        assert 'FATAL: Settings load failed' in capsys.readouterr().out

    def test_feature_index_out_of_bounds_returns_early(self, tmp_path, mocker, capsys):
        """A feature index past the end of the sorted feature list is
        reported and aborts before any pipeline is constructed."""

        settings = _settings(tmp_path)
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])
        args = _uni_args('onset', 5, inp, tmp_path / 'out')
        uni.dispatch_univariate_job(args)
        assert 'out of bounds' in capsys.readouterr().out

    def test_legacy_input_without_metadata_warns(self, tmp_path, mocker, capsys):
        """An input pickle lacking ``_input_metadata`` is still processed
        but emits the legacy-provenance warning and falls back to the
        analysis_type as the filename tag."""

        settings = _settings(tmp_path)
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'], with_metadata=False)
        out_dir = tmp_path / 'out'

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_sklearn.return_value = (
            'self.speed', {'auc': 0.6})
        mocker.patch.object(uni, 'VocalOnsetModelingPipeline',
                            return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=np.eye(4))

        uni.dispatch_univariate_job(_uni_args('onset', 0, inp, out_dir))
        out = capsys.readouterr().out
        assert 'no `_input_metadata` block' in out
        written = list(out_dir.glob('univariate_onset_*.pkl'))
        assert len(written) == 1

    def test_onset_sklearn_route_writes_artifact(self, tmp_path, mocker):
        """The ``onset`` + sklearn route fits via
        ``_run_model_for_feature_sklearn`` and writes a per-feature
        pickle whose filename embeds the upstream ``analysis_tag`` and
        which carries both the ``_run_metadata`` and the hoisted
        ``_input_metadata`` blocks."""

        settings = _settings(tmp_path, model_engine='sklearn')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed', 'other.speed'])
        out_dir = tmp_path / 'out'

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_sklearn.return_value = (
            'self.speed', {'auc': 0.7})
        mocker.patch.object(uni, 'VocalOnsetModelingPipeline',
                            return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=np.eye(4))

        uni.dispatch_univariate_job(_uni_args('onset', 0, inp, out_dir))

        fake_pipeline._run_model_for_feature_sklearn.assert_called_once()
        written = list(out_dir.glob('univariate_bout_0000_*.pkl'))
        assert len(written) == 1
        with written[0].open('rb') as fh:
            payload = pickle.load(fh)
        assert '_run_metadata' in payload
        assert '_input_metadata' in payload

    def test_onset_pygam_route(self, tmp_path, mocker):
        """The ``onset`` + pygam route fits via the pygam method and
        passes ``None`` for the basis (pygam owns its splines)."""

        settings = _settings(tmp_path, model_engine='pygam')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_pygam.return_value = (
            'self.speed', {'auc': 0.55})
        mocker.patch.object(uni, 'VocalOnsetModelingPipeline',
                            return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=None)

        uni.dispatch_univariate_job(_uni_args('onset', 0, inp, tmp_path / 'out'))
        fake_pipeline._run_model_for_feature_pygam.assert_called_once()

    def test_category_route(self, tmp_path, mocker):
        """The ``category`` route fits via ``_run_modeling_category``."""

        settings = _settings(tmp_path, model_engine='sklearn')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_modeling_category.return_value = ('self.speed', {'x': 1})
        mocker.patch.object(uni, 'VocalCategoryModelingPipeline',
                            return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=np.eye(4))

        uni.dispatch_univariate_job(_uni_args('category', 0, inp, tmp_path / 'out'))
        fake_pipeline._run_modeling_category.assert_called_once()

    def test_params_route(self, tmp_path, mocker):
        """The ``params`` route fits the bout-parameter pipeline; with
        sklearn it consumes the basis matrix."""

        settings = _settings(tmp_path, model_engine='sklearn')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_sklearn.return_value = (
            'self.speed', {'r2': 0.3})
        mocker.patch.object(uni, 'BoutParameterPipeline', return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=np.eye(4))

        uni.dispatch_univariate_job(_uni_args('params', 0, inp, tmp_path / 'out'))
        fake_pipeline._run_model_for_feature_sklearn.assert_called_once()

    def test_params_pygam_route(self, tmp_path, mocker):
        """The ``params`` route on the pygam engine fits via the pygam
        method and passes ``None`` for the basis."""

        settings = _settings(tmp_path, model_engine='pygam')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_pygam.return_value = (
            'self.speed', {'r2': 0.2})
        mocker.patch.object(uni, 'BoutParameterPipeline', return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=None)

        uni.dispatch_univariate_job(_uni_args('params', 0, inp, tmp_path / 'out'))
        fake_pipeline._run_model_for_feature_pygam.assert_called_once()

    def test_feature_mapping_failure_returns_early(self, tmp_path, mocker, capsys):
        """An unreadable input pickle trips the feature-mapping
        try/except and aborts with ``FATAL: Feature mapping failed``."""

        settings = _settings(tmp_path)
        mocker.patch.object(uni.json, 'load', return_value=settings)
        bad_input = tmp_path / 'not_a_pickle.pkl'
        bad_input.write_bytes(b'this is not a pickle')
        uni.dispatch_univariate_job(_uni_args('onset', 0, bad_input, tmp_path / 'out'))
        assert 'FATAL: Feature mapping failed' in capsys.readouterr().out

    def test_save_failure_is_caught(self, tmp_path, mocker, capsys):
        """A failure during the final ``pickle.dump`` write is caught and
        reported as ``FATAL: Saving results failed`` without raising."""

        settings = _settings(tmp_path, model_engine='sklearn')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])
        out_dir = tmp_path / 'out'

        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_sklearn.return_value = (
            'self.speed', {'auc': 0.7})
        mocker.patch.object(uni, 'VocalOnsetModelingPipeline',
                            return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized',
                            return_value=np.eye(4))
        mocker.patch.object(uni.pickle, 'dump',
                            side_effect=OSError('disk full'))

        uni.dispatch_univariate_job(_uni_args('onset', 0, inp, out_dir))
        assert 'FATAL: Saving results failed' in capsys.readouterr().out

    def test_multinomial_route(self, tmp_path, mocker):
        """The ``multinomial`` route builds the JAX pipeline + runner and
        calls ``run_univariate_training`` with the pkl path and feature."""

        settings = _settings(tmp_path)
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])

        fake_runner = mocker.MagicMock()
        fake_runner.run_univariate_training.return_value = (None, {'acc': 0.4})
        mocker.patch.object(uni, 'MultinomialModelingPipeline',
                            return_value=mocker.MagicMock())
        mocker.patch.object(uni, 'MultinomialModelRunner', return_value=fake_runner)

        uni.dispatch_univariate_job(_uni_args('multinomial', 0, inp, tmp_path / 'out'))
        fake_runner.run_univariate_training.assert_called_once()

    def test_continuous_route_unwraps_tuple(self, tmp_path, mocker):
        """The ``continuous`` route unwraps a 2-tuple training return into
        just the result dict before serialization."""

        settings = _settings(tmp_path)
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])
        out_dir = tmp_path / 'out'

        fake_runner = mocker.MagicMock()
        fake_runner.run_univariate_training.return_value = ('ignored', {'r2': 0.9})
        mocker.patch.object(uni, 'ContinuousModelingPipeline',
                            return_value=mocker.MagicMock())
        mocker.patch.object(uni, 'ContinuousModelRunner', return_value=fake_runner)

        uni.dispatch_univariate_job(_uni_args('continuous', 0, inp, out_dir))
        written = list(out_dir.glob('univariate_bout_0000_*.pkl'))
        assert len(written) == 1
        with written[0].open('rb') as fh:
            payload = pickle.load(fh)
        assert payload['self.speed'] == {'r2': 0.9}

    def test_pipeline_exception_is_caught(self, tmp_path, mocker, capsys):
        """An exception raised inside the modeling block is caught,
        reported as ``FATAL ERROR``, and aborts before any artifact is
        written."""

        settings = _settings(tmp_path, model_engine='sklearn')
        mocker.patch.object(uni.json, 'load', return_value=settings)
        inp = _input_pickle(tmp_path, ['self.speed'])
        out_dir = tmp_path / 'out'

        # Raise from inside the modeling block (the dispatcher now reads the feature
        # data from the single input-pickle load, so the failure is injected via the
        # pipeline's fit method rather than a separate load_pickle_modeling_data call).
        fake_pipeline = mocker.MagicMock()
        fake_pipeline.history_frames = 4
        fake_pipeline._run_model_for_feature_sklearn.side_effect = RuntimeError('fit blew up')
        mocker.patch.object(uni, 'VocalOnsetModelingPipeline', return_value=fake_pipeline)
        mocker.patch.object(uni, 'get_basis_matrix_standardized', return_value=np.eye(4))
        uni.dispatch_univariate_job(_uni_args('onset', 0, inp, out_dir))
        assert 'FATAL ERROR' in capsys.readouterr().out
        assert not list(out_dir.glob('univariate_*.pkl'))


# validate_paths


class TestValidatePaths:
    """``validate_paths``: the missing-mount-point guard of the selection dispatcher."""

    def test_all_present_passes(self, tmp_path):
        """When every path exists the guard returns ``None`` quietly."""

        a, b, c = (tmp_path / n for n in ('a', 'b', 'c'))
        for p in (a, b, c):
            p.write_bytes(b'x')
        assert sel.validate_paths(str(a), str(b), str(c)) is None

    def test_missing_path_raises(self, tmp_path):
        """A missing path raises ``FileNotFoundError`` naming the mount
        point that could not be resolved."""

        present = tmp_path / 'present'
        present.write_bytes(b'x')
        with pytest.raises(FileNotFoundError):
            sel.validate_paths(str(present), str(tmp_path / 'absent'), str(present))


# dispatch_model_selection


class TestDispatchModelSelection:
    """``dispatch_model_selection``: selector routing, settings-sourced kwargs and failure handling."""

    def test_onset_route(self, tmp_path, mocker):
        """The ``onset`` route calls ``vocal_onset_model_selection`` with
        the anchor flag and the settings' ``selection_p_val`` threaded through."""

        spy = mocker.patch.object(sel, 'vocal_onset_model_selection')
        args = _sel_args('onset', tmp_path)
        sel.dispatch_model_selection(args)
        spy.assert_called_once()
        kwargs = spy.call_args.kwargs
        assert kwargs['use_top_rank_as_anchor'] is True
        assert kwargs['p_val'] == resolve_modeling_setting('model_params', 'selection_p_val')
        assert kwargs['univariate_results_path'] == args.univariate_path

    @pytest.mark.parametrize(('analysis_type', 'selector'), [
        ('onset', 'vocal_onset_model_selection'),
        ('category', 'vocal_category_model_selection'),
        ('params', 'bout_parameter_model_selection'),
        ('multinomial', 'multinomial_vocal_category_model_selection'),
        ('continuous', 'continuous_vocal_manifold_model_selection'),
    ])
    def test_significance_and_target_come_from_the_settings_file(self, tmp_path, mocker, analysis_type, selector):
        """``selection_p_val`` and ``model_target_variable`` are read from the
        settings file the dispatcher resolves, never from hard-coded CLI defaults:
        a settings file carrying 0.05 and 'mean_mask_complexity' reaches the selector."""

        settings = json.loads(REPO_SETTINGS.read_text())
        settings['model_params']['selection_p_val'] = 0.05
        settings['model_params']['model_target_variable'] = 'mean_mask_complexity'
        settings_file = tmp_path / 'modeling_settings.json'
        settings_file.write_text(json.dumps(settings))
        mocker.patch.object(sel, '_SETTINGS_PATH', settings_file)
        spy = mocker.patch.object(sel, selector)
        assert sel.dispatch_model_selection(_sel_args(analysis_type, tmp_path)) == 0
        kwargs = spy.call_args.kwargs
        assert kwargs['p_val'] == 0.05
        assert kwargs['settings_path'] == str(settings_file)
        if analysis_type == 'params':
            assert kwargs['target_variable'] == 'mean_mask_complexity'

    def test_category_route(self, tmp_path, mocker):
        """The ``category`` route calls ``vocal_category_model_selection``."""

        spy = mocker.patch.object(sel, 'vocal_category_model_selection')
        sel.dispatch_model_selection(_sel_args('category', tmp_path))
        spy.assert_called_once()

    def test_params_route_threads_target_variable(self, tmp_path, mocker, capsys):
        """The ``params`` route forwards the settings' ``model_target_variable`` to
        ``bout_parameter_model_selection`` and echoes it."""

        spy = mocker.patch.object(sel, 'bout_parameter_model_selection')
        args = _sel_args('params', tmp_path)
        sel.dispatch_model_selection(args)
        spy.assert_called_once()
        target_variable = resolve_modeling_setting('model_params', 'model_target_variable')
        assert spy.call_args.kwargs['target_variable'] == target_variable
        assert target_variable in capsys.readouterr().out

    def test_multinomial_route(self, tmp_path, mocker):
        """The ``multinomial`` route calls the multinomial selector."""

        spy = mocker.patch.object(sel, 'multinomial_vocal_category_model_selection')
        sel.dispatch_model_selection(_sel_args('multinomial', tmp_path))
        spy.assert_called_once()

    def test_continuous_route(self, tmp_path, mocker):
        """The ``continuous`` route calls the manifold selector."""

        spy = mocker.patch.object(sel, 'continuous_vocal_manifold_model_selection')
        sel.dispatch_model_selection(_sel_args('continuous', tmp_path))
        spy.assert_called_once()

    def test_unknown_analysis_type_reports_and_returns(self, tmp_path, capsys):
        """An ``analysis_type`` outside the known set is reported as
        ``FATAL`` and returns without invoking any selector."""

        args = _sel_args('bogus', tmp_path)
        sel.dispatch_model_selection(args)
        assert 'Unknown analysis type' in capsys.readouterr().out

    def test_selector_exception_is_caught(self, tmp_path, mocker, capsys):
        """An exception inside the selected algorithm is caught by the
        dispatcher's try/except and surfaced as a ``CRITICAL FAILURE``
        rather than propagating out."""

        mocker.patch.object(sel, 'vocal_onset_model_selection',
                            side_effect=RuntimeError('selection blew up'))
        sel.dispatch_model_selection(_sel_args('onset', tmp_path))
        assert 'CRITICAL FAILURE' in capsys.readouterr().out

    def test_missing_input_path_raises_through_validate(self, tmp_path):
        """A non-existent univariate path trips ``validate_paths`` and
        the ``FileNotFoundError`` propagates (it is raised before the
        routing try/except)."""

        args = _sel_args('onset', tmp_path, univariate_path=str(tmp_path / 'gone.pkl'))
        with pytest.raises(FileNotFoundError):
            sel.dispatch_model_selection(args)


# Module-level CLI (`python -m ...`) entry points


@pytest.mark.filterwarnings(
    "ignore:.*found in sys.modules after import of package.*:RuntimeWarning"
)
class TestDispatcherCLI:
    """Drive both dispatcher ``__main__`` argparse blocks in-process via
    ``runpy.run_module(..., run_name='__main__')`` so the coverage tracer
    sees the guard body.

    ``run_module`` re-executes the module under a fresh namespace, so a
    stub set on the already-imported module object would NOT be picked up
    by the re-run. Instead each test neutralises the *real* worker's heavy
    effects at the dependency boundary the fresh module re-imports — for
    the univariate CLI by feeding an out-of-bounds ``--feature_idx`` (the
    real ``dispatch_univariate_job`` returns early after settings load +
    feature mapping, never constructing a pipeline); for the selection CLI
    by patching the selector at its source module
    (``usv_playpen.modeling.model_selection``) so the freshly-imported
    name resolves to the spy."""

    def test_univariate_main_parses_and_dispatches(self, tmp_path, monkeypatch,
                                                   capsys):
        """The univariate ``__main__`` block parses the four required
        flags and hands the Namespace to ``dispatch_univariate_job``,
        which (with an out-of-bounds feature index) aborts after feature
        mapping and exits with status 1, so SLURM records the failed task."""

        settings = _settings(tmp_path)
        monkeypatch.setattr(uni.json, 'load', lambda *_a, **_k: settings)
        inp = _input_pickle(tmp_path, ['self.speed'])
        monkeypatch.setattr(sys, 'argv', [
            'main_univariate_dispatcher',
            '--analysis_type', 'onset',
            '--feature_idx', '99',  # out of bounds -> early clean return
            '--input_data', str(inp),
            '--output_dir', str(tmp_path / 'out'),
        ])
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module('usv_playpen.modeling.main_univariate_dispatcher',
                             run_name='__main__')
        assert exit_info.value.code == 1
        assert 'out of bounds' in capsys.readouterr().out

    def test_selection_main_parses_and_dispatches(self, tmp_path, monkeypatch):
        """The selection ``__main__`` block parses its flags (including
        the ``--anchor`` store-true) and routes to the selector with the
        settings' ``model_target_variable`` and ``selection_p_val``, patched
        here at its source module so the fresh import resolves to the spy."""

        calls = {}

        def _stub(**kwargs):
            """
            Description
            -----------
            Stand-in selector that records the keyword arguments it receives.

            Parameters
            ----------
            **kwargs
                The selector keyword arguments the dispatcher passes.

            Returns
            -------
            None
            """
            calls.update(kwargs)

        monkeypatch.setattr(msrc, 'bout_parameter_model_selection', _stub)
        univ = tmp_path / 'univariate.pkl'
        inp = tmp_path / 'input.pkl'
        univ.write_bytes(b'x')
        inp.write_bytes(b'x')
        monkeypatch.setattr(sys, 'argv', [
            'main_model_selection_dispatcher',
            '--analysis_type', 'params',
            '--univariate_path', str(univ),
            '--input_path', str(inp),
            '--output_dir', str(tmp_path / 'out'),
            '--anchor',
        ])
        # __main__ now propagates the dispatcher's exit code via sys.exit, so a
        # successful run raises SystemExit(0) (this is what lets SLURM/`set -e`
        # detect a crashed run, which would exit non-zero).
        with pytest.raises(SystemExit) as exc:
            runpy.run_module('usv_playpen.modeling.main_model_selection_dispatcher',
                             run_name='__main__')
        assert exc.value.code == 0
        assert calls['target_variable'] == resolve_modeling_setting('model_params', 'model_target_variable')
        assert calls['p_val'] == resolve_modeling_setting('model_params', 'selection_p_val')
        assert calls['use_top_rank_as_anchor'] is True
