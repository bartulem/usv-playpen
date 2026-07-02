"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.modeling_metadata`` — the
provenance-metadata builders, the reserved-key guards, the cohort-label
derivation, and the consolidated ``selection_*.pkl`` loader.

These functions are the project's audit trail: they hash settings, stamp
schema versions, and split reserved metadata blocks from feature data.
The tests pin the contracts downstream consumers rely on — a stable
settings hash, reserved-key collision detection, the documented
target-centric cohort labels, and a pickle round-trip for
``load_selection_results`` (exercised with ``tmp_path`` so no fixture
files are committed).
"""

from __future__ import annotations

import os
import pickle
import re

import pytest

from usv_playpen.modeling.modeling_metadata import (
    RESERVED_METADATA_KEYS,
    SCHEMA_VERSIONS,
    _utcnow_iso,
    assert_no_reserved_keys,
    build_consolidation_metadata,
    build_input_metadata,
    build_run_metadata,
    build_selection_metadata,
    compute_settings_sha256,
    derive_camera_fps_field,
    derive_experimental_condition,
    derive_feature_zoo_full,
    extract_metadata_blocks,
    get_git_commit_info,
    get_package_version,
    inject_metadata,
    load_selection_results,
    metadata_blocks_equal,
)


# compute_settings_sha256


class TestComputeSettingsSha256:

    def test_dict_hash_is_stable_and_order_independent(self):
        """Two dicts with identical content but different key insertion
        order hash to the same canonical digest (``sort_keys=True``)."""

        a = {'b': 2, 'a': 1, 'nested': {'y': 1, 'x': 0}}
        b = {'a': 1, 'nested': {'x': 0, 'y': 1}, 'b': 2}
        h_a = compute_settings_sha256(a)
        assert re.fullmatch(r'[0-9a-f]{64}', h_a)
        assert h_a == compute_settings_sha256(b)

    def test_file_hash_matches_bytes(self, tmp_path):
        """A path argument hashes the file bytes verbatim."""

        import hashlib
        p = tmp_path / 'settings.json'
        p.write_bytes(b'{"k": 1}\n')
        expected = hashlib.sha256(b'{"k": 1}\n').hexdigest()
        assert compute_settings_sha256(str(p)) == expected

    def test_missing_file_returns_unknown(self):
        """An unreadable source never raises — it returns ``'unknown'``."""

        assert compute_settings_sha256('/no/such/settings.json') == 'unknown'


# assert_no_reserved_keys


class TestAssertNoReservedKeys:

    def test_clean_payload_passes(self):
        """A payload with only feature keys passes silently."""

        assert assert_no_reserved_keys({'feat_a': 1, 'feat_b': 2}) is None

    def test_collision_raises(self):
        """A payload already carrying a reserved key is rejected."""

        with pytest.raises(ValueError):
            assert_no_reserved_keys({'_input_metadata': {}})

    def test_custom_reserved_tuple(self):
        """A narrower reserved tuple only flags those keys."""

        # '_run_metadata' is reserved globally but not in the custom set.
        assert assert_no_reserved_keys({'_run_metadata': {}},
                                       reserved=('_input_metadata',)) is None


# extract_metadata_blocks / inject_metadata


class TestExtractInjectMetadata:

    def test_split_separates_reserved_from_data(self):
        """Reserved blocks land in the metadata dict; everything else
        stays in the clean data dict."""

        full = {'feat_a': [1, 2], '_input_metadata': {'x': 1},
                '_run_metadata': {'y': 2}}
        clean, meta = extract_metadata_blocks(full)
        assert clean == {'feat_a': [1, 2]}
        assert meta == {'_input_metadata': {'x': 1}, '_run_metadata': {'y': 2}}

    def test_non_reserved_underscore_key_kept_as_data(self, capsys):
        """A non-reserved underscore key is treated as data (with a
        warning print), not silently dropped."""

        clean, meta = extract_metadata_blocks({'_weird': 9, 'feat': 1})
        assert clean == {'_weird': 9, 'feat': 1}
        assert meta == {}
        assert 'WARNING' in capsys.readouterr().out

    def test_inject_round_trip(self):
        """Injecting then extracting recovers the original payload and
        metadata; the source payload is not mutated."""

        payload = {'feat_a': 1}
        md = {'x': 1}
        out = inject_metadata(payload, _input_metadata=md)
        assert payload == {'feat_a': 1}  # unchanged
        clean, meta = extract_metadata_blocks(out)
        assert clean == {'feat_a': 1}
        assert meta == {'_input_metadata': md}

    def test_inject_non_reserved_name_raises(self):
        """Injecting under a non-reserved key name is rejected."""

        with pytest.raises(ValueError):
            inject_metadata({'feat': 1}, _bogus_metadata={})

    def test_inject_collision_raises(self):
        """Injecting a block the payload already carries is rejected."""

        with pytest.raises(ValueError):
            inject_metadata({'_input_metadata': {}}, _input_metadata={'x': 1})


# metadata_blocks_equal


class TestMetadataBlocksEqual:

    def test_identical_blocks_equal(self):
        """Structurally identical blocks compare equal, including nested
        dicts."""

        a = {'k': 1, 'nested': {'x': [1, 2]}}
        b = {'k': 1, 'nested': {'x': [1, 2]}}
        assert metadata_blocks_equal(a, b) is True

    def test_ignore_keys_skips_timestamp(self):
        """Keys named in ``ignore_keys`` (e.g. per-file timestamps) are
        excluded from the comparison."""

        a = {'cfg': 1, 'created_utc': 'T1'}
        b = {'cfg': 1, 'created_utc': 'T2'}
        assert metadata_blocks_equal(a, b, ignore_keys=('created_utc',)) is True
        assert metadata_blocks_equal(a, b) is False

    def test_differing_key_set_unequal(self):
        """Blocks with different key sets are unequal."""

        assert metadata_blocks_equal({'a': 1}, {'a': 1, 'b': 2}) is False

    def test_nested_dict_value_mismatch_unequal(self):
        """A divergent value inside a nested dict makes the blocks unequal
        (the recursive-mismatch branch)."""

        a = {'cfg': {'x': 1, 'y': 2}}
        b = {'cfg': {'x': 1, 'y': 99}}
        assert metadata_blocks_equal(a, b) is False


# derive_experimental_condition


class TestDeriveExperimentalCondition:

    def _settings(self, fname, pred_idx=0):
        return {'io': {'session_list_file': f'/data/{fname}'},
                'model_params': {'model_predictor_mouse_index': pred_idx}}

    def test_mute_female_maps_to_male_mute_partner(self):
        """A ``mute_female`` partner cohort (intact predictor is male)
        returns the legacy ``male_mute_partner`` label."""

        s = self._settings('behavioral_courtship_mute_female_sessions_list.txt')
        assert derive_experimental_condition(s) == 'male_mute_partner'

    def test_mute_male_maps_to_female_mute_partner(self):
        """A ``mute_male`` partner cohort returns ``female_mute_partner``."""

        s = self._settings('mute_male_sessions_list.txt')
        assert derive_experimental_condition(s) == 'female_mute_partner'

    def test_intact_partners_target_is_opposite_of_predictor(self):
        """Intact cohort: predictor slot 0 -> target female; slot 1 ->
        target male."""

        s0 = self._settings('intact_partners_list.txt', pred_idx=0)
        s1 = self._settings('intact_partners_list.txt', pred_idx=1)
        assert derive_experimental_condition(s0) == 'intact_partners_female'
        assert derive_experimental_condition(s1) == 'intact_partners_male'

    def test_unrecognised_returns_unspecified(self):
        """An unmatched filename yields ``'unspecified'`` rather than
        raising."""

        assert derive_experimental_condition(self._settings('random.txt')) == 'unspecified'


# derive_feature_zoo_full


class TestDeriveFeatureZooFull:

    def _settings(self):
        return {'kinematic_features': {
            'egocentric': ['speed', 'neck_elevation'],
            'dyadic_pose': ['allo_yaw-tti'],
            'dyadic_engagement': ['orofacial-sei'],
        }}

    def test_per_mouse_expansion_sorted(self):
        """Each egocentric suffix expands into self/other columns; dyadic
        names pass through; the result is sorted."""

        out = derive_feature_zoo_full(self._settings())
        assert out == sorted([
            'self.speed', 'other.speed',
            'self.neck_elevation', 'other.neck_elevation',
            'allo_yaw-tti', 'orofacial-sei',
        ])

    def test_bare_suffixes_when_flag_false(self):
        """With ``include_egocentric_per_mouse=False`` the bare egocentric
        suffixes are returned (no self/other prefix)."""

        out = derive_feature_zoo_full(self._settings(), include_egocentric_per_mouse=False)
        assert 'speed' in out
        assert 'self.speed' not in out


# derive_camera_fps_field


class TestDeriveCameraFpsField:

    def test_homogeneous_returns_single_float(self):
        """When every session shares an fps the field collapses to one
        float."""

        out = derive_camera_fps_field({'s1': 150.0, 's2': 150.0})
        assert out == 150.0
        assert isinstance(out, float)

    def test_heterogeneous_returns_dict(self):
        """Mixed fps values are preserved as a per-session dict."""

        out = derive_camera_fps_field({'s1': 150.0, 's2': 120.0})
        assert out == {'s1': 150.0, 's2': 120.0}

    def test_empty_returns_empty_dict(self):
        """An empty input maps to an empty dict."""

        assert derive_camera_fps_field({}) == {}


# build_consolidation_metadata


class TestBuildConsolidationMetadata:

    def test_structure_and_schema_version(self):
        """The block carries the consolidation schema version, coerced
        counts, and the verbatim file lists."""

        md = build_consolidation_metadata(
            n_files_merged=2,
            individual_file_paths=['/a.pkl', '/b.pkl'],
            individual_file_timestamps=['T1', 'T2'],
            consolidator_name='consolidate_univariate_results',
            consolidator_version=3,
        )
        assert md['_schema_version'] == SCHEMA_VERSIONS['consolidation']
        assert md['n_files_merged'] == 2
        assert md['consolidator_version'] == 3
        assert md['individual_file_paths'] == ['/a.pkl', '/b.pkl']
        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
                            md['consolidated_at_utc'])


# load_selection_results


def _write_consolidated(path, steps, extra_meta=None):
    """Pickle a minimal consolidated artifact (a ``steps`` list plus any
    reserved metadata blocks) to ``path``."""

    cons = {'steps': steps}
    if extra_meta:
        cons.update(extra_meta)
    with path.open('wb') as fh:
        pickle.dump(cons, fh)


class TestLoadSelectionResults:

    def test_file_path_round_trip(self, tmp_path):
        """Loading an explicit consolidated file returns its steps, its
        basename, and the harvested reserved metadata blocks."""

        steps = [{'step_idx': 0}, {'step_idx': 1}]
        pkl = tmp_path / 'selection_male_run.pkl'
        _write_consolidated(pkl, steps,
                            extra_meta={'_input_metadata': {'cohort': 'x'}})
        out_steps, name, meta = load_selection_results(str(pkl))
        assert out_steps == steps
        assert name == 'selection_male_run.pkl'
        assert meta == {'_input_metadata': {'cohort': 'x'}}

    def test_directory_picks_latest(self, tmp_path):
        """A directory load returns the most-recently-modified
        ``selection_*.pkl``."""

        old = tmp_path / 'selection_old.pkl'
        new = tmp_path / 'selection_new.pkl'
        _write_consolidated(old, [{'step_idx': 0}])
        _write_consolidated(new, [{'step_idx': 99}])
        # Force `old` to be older than `new`.
        os.utime(old, (1_000, 1_000))
        os.utime(new, (2_000, 2_000))
        out_steps, name, _ = load_selection_results(str(tmp_path))
        assert name == 'selection_new.pkl'
        assert out_steps == [{'step_idx': 99}]

    def test_non_consolidated_pickle_raises_value_error(self, tmp_path):
        """A pickle without a ``steps`` list is not a consolidated
        artifact -> ``ValueError``."""

        pkl = tmp_path / 'selection_bad.pkl'
        with pkl.open('wb') as fh:
            pickle.dump({'not_steps': 1}, fh)
        with pytest.raises(ValueError):
            load_selection_results(str(pkl))

    def test_missing_path_raises_file_not_found(self, tmp_path):
        """A non-existent path raises ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            load_selection_results(str(tmp_path / 'nope'))

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        """A directory with no ``selection_*.pkl`` raises
        ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            load_selection_results(str(tmp_path))


# misc never-raise helpers


class TestMiscHelpers:

    def test_get_package_version_returns_str(self):
        """The version helper always returns a string and never raises."""

        assert isinstance(get_package_version(), str)

    def test_get_package_version_failure_returns_unknown(self, mocker):
        """When ``usv_playpen.__version__`` is missing the helper hits an
        ``AttributeError`` on the ``from .. import __version__`` line,
        swallows it, and returns the ``'unknown'`` placeholder."""

        import usv_playpen
        # The relative import resolves the already-cached parent package
        # and reads its __version__ attribute; removing the attribute for
        # the duration of the call forces the AttributeError failure path.
        # The value is captured and restored even if the assertion fails.
        sentinel = object()
        saved = getattr(usv_playpen, '__version__', sentinel)
        if saved is not sentinel:
            delattr(usv_playpen, '__version__')
        try:
            assert get_package_version() == 'unknown'
        finally:
            if saved is not sentinel:
                usv_playpen.__version__ = saved

    def test_utcnow_iso_format(self):
        """The UTC timestamp matches the documented ``...Z`` ISO form."""

        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', _utcnow_iso())

    def test_reserved_keys_constant_shape(self):
        """The reserved-key tuple is the canonical 4-block set the
        splitters key off."""

        assert set(RESERVED_METADATA_KEYS) == {
            '_input_metadata', '_run_metadata',
            '_univariate_metadata', '_consolidation_metadata',
        }


# synthetic settings fixture for the metadata builders


def _full_modeling_settings(model_engine='sklearn',
                            model_basis_function='raised_cosine',
                            session_list_file='/data/intact_partners_list.txt',
                            predictor_idx=0,
                            tune_regularization=True):
    """
    Construct a fully-populated synthetic ``modeling_settings`` dict that
    satisfies every nested key the metadata builders read.

    The builders never use ``.get`` with defaults — each one indexes the
    settings dict directly — so every key the three builders touch
    (``io``, ``model_params``, ``vocal_features``, ``kinematic_features``,
    and the ``hyperparameters`` sub-tree with its ``jax_linear``,
    ``classical``, and ``basis_functions`` blocks) must be present here.

    Parameters
    ----------
    model_engine : str, default 'sklearn'
        Value written into ``model_params['model_engine']``. Drives the
        CPU-path branch in :func:`build_run_metadata` (``'sklearn'`` vs
        ``'pygam'``).
    model_basis_function : str, default 'raised_cosine'
        Basis-function name stored in ``model_params`` and used as a key
        into ``hyperparameters['basis_functions']`` on the sklearn path.
    session_list_file : str, default '/data/intact_partners_list.txt'
        Path written into ``io['session_list_file']``; controls the
        cohort label produced by :func:`derive_experimental_condition`.
    predictor_idx : int, default 0
        Predictor-mouse index stored in
        ``model_params['model_predictor_mouse_index']``.
    tune_regularization : bool, default True
        Value of the JAX ``tune_regularization_bool`` flag, which decides
        whether the inner-CV grid block is emitted on the JAX path.

    Returns
    -------
    dict
        A deep-enough synthetic settings dict suitable for every builder.
    """

    return {
        'io': {'session_list_file': session_list_file},
        'model_params': {
            'model_engine': model_engine,
            'model_basis_function': model_basis_function,
            'model_predictor_mouse_index': predictor_idx,
            'mixture_model_component_index': 3,
            'mixture_model_z_score': 1.5,
            'random_seed': 42,
            'spatial_cluster_num': 8,
            'test_proportion': 0.25,
            'session_split_max_attempts': 100,
            'session_split_widen_step': 0.05,
            'session_split_widen_every': 10,
        },
        'vocal_features': {
            'usv_predictor_type': 'rate',
            'usv_predictor_partner_only': True,
            'usv_predictor_smoothing_sd': 0.1,
            'usv_noise_categories': [0],
        },
        'kinematic_features': {
            'egocentric': ['speed', 'neck_elevation'],
            'dyadic_pose': ['allo_yaw-tti'],
            'dyadic_engagement': ['orofacial-sei', 'anogenital-sei'],
            'dyadic_pose_symmetric': True,
        },
        'hyperparameters': {
            'basis_functions': {
                'raised_cosine': {'n_bases': 5, 'spread': 2.0},
            },
            'classical': {
                'pygam': {
                    'n_splines_time': 10,
                    'n_splines_value': 8,
                    'lam_penalty': 0.6,
                    'max_iterations': 250,
                    'tol_val': 1e-4,
                    'distribution': 'binomial',
                    'link': 'logit',
                },
                'logistic_regression': {'C': 1.0, 'penalty': 'l2'},
                'ridge_regression': {'alpha': 1.0},
            },
            'jax_linear': {
                'multinomial_logistic': {
                    'bin_resizing_factor': 2,
                    'lambda_smooth_fixed': 0.01,
                    'l2_reg_fixed': 0.001,
                    'smoothness_derivative_order': 2,
                    'learning_rate': 0.05,
                    'max_iter': 500,
                    'tol': 1e-6,
                    'random_state': 7,
                    'use_lax_loop': True,
                    'tune_regularization_bool': tune_regularization,
                    'focal_loss_gamma': 2.0,
                    'balance_predictions_bool': True,
                    'balance_train_bool': False,
                    'tune_regularization_params': {
                        'lambda_smooth_decades_each_side': 3,
                        'l2_reg_decades_each_side': 2,
                        'inner_cv_folds': 5,
                        'inner_cv_scoring_metric': 'log_loss',
                        'inner_cv_use_one_se_rule': True,
                        'inner_max_iter': 200,
                    },
                },
                'bivariate': {
                    'bin_resizing_factor': 4,
                    'lambda_smooth_fixed': 0.02,
                    'l2_reg_fixed': 0.002,
                    'smoothness_derivative_order': 1,
                    'learning_rate': 0.1,
                    'max_iter': 1000,
                    'tol': 1e-5,
                    'random_state': 11,
                    'use_lax_loop': False,
                    'tune_regularization_bool': tune_regularization,
                    'tune_regularization_params': {
                        'lambda_smooth_decades_each_side': 4,
                        'l2_reg_decades_each_side': 3,
                        'inner_cv_folds': 4,
                        'inner_cv_scoring_metric': 'r2',
                        'inner_cv_use_one_se_rule': False,
                        'inner_max_iter': 150,
                    },
                },
            },
        },
    }


def _input_builder_kwargs(modeling_settings, analysis_specific=None,
                          settings_path=None):
    """
    Assemble the full scalar / list / dict argument set for
    :func:`build_input_metadata`.

    Centralizes the ~20 positional-ish arguments the input builder takes
    so each test can override just the one or two it cares about while the
    rest stay valid.

    Parameters
    ----------
    modeling_settings : dict
        The synthetic settings dict the builder reads nested keys from.
    analysis_specific : dict, optional
        Per-analysis knob dict; defaults to a small onset-style payload.
    settings_path : str, optional
        Forwarded as the builder's ``settings_path`` argument so the
        SHA-256 can be computed on a file rather than the in-memory dict.

    Returns
    -------
    dict
        Keyword arguments ready to splat into ``build_input_metadata``.
    """

    if analysis_specific is None:
        analysis_specific = {'model_target_vocal_type': 'usv',
                             'usv_count_threshold': 5}
    return {
        'modeling_settings': modeling_settings,
        'analysis_type': 'onset',
        'analysis_tag': 'onsets_bout',
        'pipeline_class': 'VocalOnsetModelingPipeline',
        'target_idx': 1,
        'predictor_idx': 0,
        'n_sessions_used': 3,
        'session_ids': ['s1', 's2', 's3'],
        'n_events_per_session': {'s1': 10, 's2': 20, 's3': 30},
        'feature_zoo_full': ['self.speed', 'other.speed'],
        'feature_zoo_kept': ['self.speed'],
        'dyadic_engagement_features_used': ['orofacial-sei'],
        'dyadic_pose_symmetric_features_used': True,
        'noise_vocal_categories_excluded': [0],
        'vocal_signal_columns_added': ['usv_rate_partner'],
        'filter_history_seconds': 2.0,
        'filter_history_frames': 300,
        'camera_sampling_rate_hz': 150.0,
        'ibi_thresholds': {'male': 0.4, 'female': 0.5},
        'analysis_specific': analysis_specific,
        'settings_path': settings_path,
    }


# get_git_commit_info


class TestGetGitCommitInfo:

    def test_clean_repo_reports_commit_and_clean(self, mocker):
        """A repo whose ``rev-parse`` succeeds with empty
        ``status --porcelain`` reports the commit SHA and ``dirty`` False."""

        def fake_run(cmd, *args, **kwargs):
            result = mocker.Mock()
            result.returncode = 0
            if 'rev-parse' in cmd:
                result.stdout = 'abc1234deadbeef\n'
            else:
                result.stdout = ''
            return result

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.subprocess.run',
            side_effect=fake_run,
        )
        info = get_git_commit_info()
        assert info == {'commit': 'abc1234deadbeef', 'dirty': False}

    def test_dirty_repo_flags_dirty(self, mocker):
        """Non-empty ``status --porcelain`` output flips ``dirty`` True."""

        def fake_run(cmd, *args, **kwargs):
            result = mocker.Mock()
            result.returncode = 0
            if 'rev-parse' in cmd:
                result.stdout = 'cafef00d\n'
            else:
                result.stdout = ' M some_file.py\n'
            return result

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.subprocess.run',
            side_effect=fake_run,
        )
        info = get_git_commit_info(repo_root='/some/repo')
        assert info == {'commit': 'cafef00d', 'dirty': True}

    def test_failure_returns_placeholder(self, mocker):
        """Any subprocess exception yields the ``unknown`` / False
        placeholder rather than raising."""

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.subprocess.run',
            side_effect=OSError('git not found'),
        )
        assert get_git_commit_info() == {'commit': 'unknown', 'dirty': False}

    def test_nonzero_returncode_keeps_placeholder(self, mocker):
        """A non-zero ``rev-parse`` return code (non-git dir) leaves the
        placeholder commit/dirty in place without raising."""

        def fake_run(cmd, *args, **kwargs):
            result = mocker.Mock()
            result.returncode = 128
            result.stdout = ''
            return result

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.subprocess.run',
            side_effect=fake_run,
        )
        assert get_git_commit_info() == {'commit': 'unknown', 'dirty': False}


# build_input_metadata


class TestBuildInputMetadata:

    def test_full_block_structure_and_values(self, mocker):
        """The Level-1 block carries the schema version, target-centric
        cohort label, coerced scalar fields, and every documented field
        group."""

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.get_git_commit_info',
            return_value={'commit': 'deadbeef', 'dirty': True},
        )
        settings = _full_modeling_settings(
            session_list_file='/data/intact_partners_list.txt',
            predictor_idx=0)
        md = build_input_metadata(**_input_builder_kwargs(settings))

        assert md['_schema_version'] == SCHEMA_VERSIONS['input']
        # Cohort / experimental scope.
        assert md['experimental_condition'] == 'intact_partners_female'
        assert md['target_idx'] == 1
        assert md['target_mouse_sex'] == 'female'
        assert md['predictor_idx'] == 0
        assert md['predictor_mouse_sex'] == 'male'
        assert md['n_sessions_used'] == 3
        assert md['session_ids'] == ['s1', 's2', 's3']
        assert md['n_events_per_session'] == {'s1': 10, 's2': 20, 's3': 30}
        # Behavioral feature provenance.
        assert md['feature_zoo_full'] == ['self.speed', 'other.speed']
        assert md['feature_zoo_kept'] == ['self.speed']
        assert md['dyadic_engagement_features_used'] == ['orofacial-sei']
        assert md['dyadic_pose_symmetric_features_used'] is True
        assert md['noise_vocal_categories_excluded'] == [0]
        # Vocal-input shape (read from settings).
        assert md['usv_predictor_type'] == 'rate'
        assert md['usv_predictor_partner_only'] is True
        assert md['usv_predictor_smoothing_sd'] == 0.1
        assert md['vocal_signal_columns_added'] == ['usv_rate_partner']
        # Temporal frame.
        assert md['filter_history_seconds'] == 2.0
        assert md['filter_history_frames'] == 300
        assert md['camera_sampling_rate_hz'] == 150.0
        assert md['mixture_model_component_index'] == 3
        assert md['mixture_model_z_score'] == 1.5
        assert md['ibi_thresholds'] == {'male': 0.4, 'female': 0.5}
        # Analysis-specific passthrough.
        assert md['analysis_specific'] == {'model_target_vocal_type': 'usv',
                                           'usv_count_threshold': 5}
        # Provenance.
        assert md['analysis_type'] == 'onset'
        assert md['analysis_tag'] == 'onsets_bout'
        assert md['pipeline_class'] == 'VocalOnsetModelingPipeline'
        assert md['session_list_file'] == '/data/intact_partners_list.txt'
        assert md['git_commit'] == 'deadbeef'
        assert md['git_dirty'] is True
        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
                            md['created_utc'])
        assert isinstance(md['package_version'], str)
        # In-memory settings -> a real 64-hex digest, not 'unknown'.
        assert re.fullmatch(r'[0-9a-f]{64}', md['settings_sha256'])

    def test_target_idx_zero_is_male(self, mocker):
        """A target slot of 0 maps to ``'male'`` and predictor 1 to
        ``'female'`` (the sex coercion branch)."""

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.get_git_commit_info',
            return_value={'commit': 'x', 'dirty': False},
        )
        settings = _full_modeling_settings()
        kwargs = _input_builder_kwargs(settings)
        kwargs['target_idx'] = 0
        kwargs['predictor_idx'] = 1
        md = build_input_metadata(**kwargs)
        assert md['target_mouse_sex'] == 'male'
        assert md['predictor_mouse_sex'] == 'female'

    def test_settings_path_hashes_file_bytes(self, tmp_path, mocker):
        """When ``settings_path`` is supplied the SHA-256 is computed on
        the file bytes rather than the in-memory dict."""

        import hashlib
        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.get_git_commit_info',
            return_value={'commit': 'x', 'dirty': False},
        )
        p = tmp_path / 'modeling_settings.json'
        p.write_bytes(b'{"k": 1}\n')
        settings = _full_modeling_settings()
        kwargs = _input_builder_kwargs(settings, settings_path=str(p))
        md = build_input_metadata(**kwargs)
        assert md['settings_sha256'] == hashlib.sha256(b'{"k": 1}\n').hexdigest()

    def test_camera_fps_dict_passed_through(self, mocker):
        """A heterogeneous per-session fps dict is stored verbatim."""

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.get_git_commit_info',
            return_value={'commit': 'x', 'dirty': False},
        )
        settings = _full_modeling_settings()
        kwargs = _input_builder_kwargs(settings)
        kwargs['camera_sampling_rate_hz'] = {'s1': 150.0, 's2': 120.0}
        md = build_input_metadata(**kwargs)
        assert md['camera_sampling_rate_hz'] == {'s1': 150.0, 's2': 120.0}


# build_run_metadata


class TestBuildRunMetadata:

    def _common_assertions(self, md, analysis_type, engine, basis):
        """Assert the engine / outer-loop / provenance fields shared by
        every analysis_type branch of :func:`build_run_metadata`."""

        assert md['_schema_version'] == SCHEMA_VERSIONS['run']
        assert md['analysis_type'] == analysis_type
        assert md['model_engine'] == engine
        assert md['basis_function'] == basis
        assert md['null_strategy'] == 'x_history_shuffle'
        assert md['n_outer_folds'] == 5
        assert md['split_strategy'] == 'mixed'
        assert md['random_seed_outer'] == 42
        assert md['spatial_cluster_num'] == 8
        assert md['test_proportion'] == 0.25
        assert md['session_split_max_attempts'] == 100
        assert md['session_split_widen_step'] == 0.05
        assert md['session_split_widen_every'] == 10
        assert md['git_commit'] == 'gitsha'
        assert md['git_dirty'] is False
        assert isinstance(md['package_version'], str)
        assert re.fullmatch(r'[0-9a-f]{64}', md['settings_sha256'])

    def _build(self, settings, analysis_type, mocker):
        """Patch git info and call ``build_run_metadata`` with stable
        outer-loop arguments."""

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.get_git_commit_info',
            return_value={'commit': 'gitsha', 'dirty': False},
        )
        return build_run_metadata(
            modeling_settings=settings,
            analysis_type=analysis_type,
            null_strategy='x_history_shuffle',
            n_outer_folds=5,
            split_strategy='mixed',
        )

    def test_onset_sklearn_emits_logistic_block(self, mocker):
        """``onset`` + sklearn engine emits the sklearn block with the
        basis-function params and the logistic-regression sub-block; no
        JAX / pygam blocks."""

        settings = _full_modeling_settings(model_engine='sklearn')
        md = self._build(settings, 'onset', mocker)
        self._common_assertions(md, 'onset', 'sklearn', 'raised_cosine')
        assert 'jax_hyperparameters' not in md
        assert 'pygam_hyperparameters' not in md
        skl = md['sklearn_hyperparameters']
        assert skl['basis_function'] == 'raised_cosine'
        assert skl['basis_function_params'] == {'n_bases': 5, 'spread': 2.0}
        assert skl['logistic_regression'] == {'C': 1.0, 'penalty': 'l2'}
        assert 'ridge_regression' not in skl

    def test_category_sklearn_emits_logistic_block(self, mocker):
        """``category`` + sklearn also routes through the logistic
        sub-block (shared with onset)."""

        settings = _full_modeling_settings(model_engine='sklearn')
        md = self._build(settings, 'category', mocker)
        self._common_assertions(md, 'category', 'sklearn', 'raised_cosine')
        assert 'logistic_regression' in md['sklearn_hyperparameters']

    def test_params_sklearn_emits_ridge_block(self, mocker):
        """``params`` + sklearn routes through the ridge-regression
        sub-block instead of logistic."""

        settings = _full_modeling_settings(model_engine='sklearn')
        md = self._build(settings, 'params', mocker)
        self._common_assertions(md, 'params', 'sklearn', 'raised_cosine')
        skl = md['sklearn_hyperparameters']
        assert skl['ridge_regression'] == {'alpha': 1.0}
        assert 'logistic_regression' not in skl

    def test_onset_pygam_emits_pygam_block(self, mocker):
        """``onset`` + pygam engine emits the pygam hyperparameter block
        and no sklearn block."""

        settings = _full_modeling_settings(model_engine='pygam')
        md = self._build(settings, 'onset', mocker)
        self._common_assertions(md, 'onset', 'pygam', 'raised_cosine')
        assert 'sklearn_hyperparameters' not in md
        assert 'jax_hyperparameters' not in md
        pgm = md['pygam_hyperparameters']
        assert pgm['n_splines_time'] == 10
        assert pgm['n_splines_value'] == 8
        assert pgm['lam_penalty'] == 0.6
        assert pgm['max_iterations'] == 250
        assert pgm['tol_val'] == 1e-4
        assert pgm['distribution'] == 'binomial'
        assert pgm['link'] == 'logit'

    def test_multinomial_jax_block_with_tuning(self, mocker):
        """``multinomial`` emits the JAX block from the
        ``multinomial_logistic`` sub-tree, including the focal-loss /
        balance flags and the inner-CV tuning grid (tuning on)."""

        settings = _full_modeling_settings(tune_regularization=True)
        md = self._build(settings, 'multinomial', mocker)
        self._common_assertions(md, 'multinomial', 'sklearn', 'raised_cosine')
        assert 'sklearn_hyperparameters' not in md
        assert 'pygam_hyperparameters' not in md
        jax = md['jax_hyperparameters']
        assert jax['jax_block_kind'] == 'multinomial_logistic'
        assert jax['bin_resizing_factor'] == 2
        assert jax['lambda_smooth_fixed'] == 0.01
        assert jax['l2_reg_fixed'] == 0.001
        assert jax['smoothness_derivative_order'] == 2
        assert jax['learning_rate'] == 0.05
        assert jax['max_iter'] == 500
        assert jax['tol'] == 1e-6
        assert jax['random_state'] == 7
        assert jax['use_lax_loop'] is True
        assert jax['tune_regularization_bool'] is True
        assert jax['focal_loss_gamma'] == 2.0
        assert jax['balance_predictions_bool'] is True
        assert jax['balance_train_bool'] is False
        tp = jax['tune_regularization_params']
        assert tp['lambda_smooth_decades_each_side'] == 3
        assert tp['l2_reg_decades_each_side'] == 2
        assert tp['inner_cv_folds'] == 5
        assert tp['inner_cv_scoring_metric'] == 'log_loss'
        assert tp['inner_cv_use_one_se_rule'] is True
        assert tp['inner_max_iter'] == 200

    def test_continuous_jax_block_without_tuning(self, mocker):
        """``continuous`` emits the JAX block from the ``bivariate``
        sub-tree and, with tuning off, omits the inner-CV grid and the
        multinomial-only focal-loss / balance fields."""

        settings = _full_modeling_settings(tune_regularization=False)
        md = self._build(settings, 'continuous', mocker)
        self._common_assertions(md, 'continuous', 'sklearn', 'raised_cosine')
        jax = md['jax_hyperparameters']
        assert jax['jax_block_kind'] == 'bivariate'
        assert jax['bin_resizing_factor'] == 4
        assert jax['use_lax_loop'] is False
        assert jax['tune_regularization_bool'] is False
        assert 'tune_regularization_params' not in jax
        assert 'focal_loss_gamma' not in jax
        assert 'balance_predictions_bool' not in jax

    def test_continuous_jax_block_with_tuning(self, mocker):
        """``continuous`` with tuning on emits the bivariate inner-CV
        grid (the tuning branch for the non-multinomial JAX path)."""

        settings = _full_modeling_settings(tune_regularization=True)
        md = self._build(settings, 'continuous', mocker)
        tp = md['jax_hyperparameters']['tune_regularization_params']
        assert tp['lambda_smooth_decades_each_side'] == 4
        assert tp['inner_cv_scoring_metric'] == 'r2'
        assert tp['inner_cv_use_one_se_rule'] is False

    def test_multinomial_jax_block_without_tuning(self, mocker):
        """``multinomial`` with tuning off still emits the focal-loss /
        balance fields but drops the inner-CV grid."""

        settings = _full_modeling_settings(tune_regularization=False)
        md = self._build(settings, 'multinomial', mocker)
        jax = md['jax_hyperparameters']
        assert jax['focal_loss_gamma'] == 2.0
        assert 'tune_regularization_params' not in jax


# build_selection_metadata


class TestBuildSelectionMetadata:

    def _build(self, mocker, extra_knobs=None, settings_path=None):
        """Patch git info and call ``build_selection_metadata`` with a
        stable Level-3 argument set."""

        mocker.patch(
            'usv_playpen.modeling.modeling_metadata.get_git_commit_info',
            return_value={'commit': 'selsha', 'dirty': True},
        )
        return build_selection_metadata(
            modeling_settings=_full_modeling_settings(),
            selection_function='vocal_onset_model_selection',
            selection_metric='AUC',
            n_splits_selection=5,
            test_proportion=0.2,
            split_strategy='session',
            random_seed=123,
            one_se_rule_used=True,
            aic_termination_used=False,
            n_anchor_features=4,
            anchor_feature='self.speed',
            gam_kwargs={'n_splines': 10, 'lam': 0.6},
            extra_knobs=extra_knobs,
            settings_path=settings_path,
        )

    def test_full_block_structure_and_values(self, mocker):
        """The Level-3 selection block carries the schema version, all
        selection / outer-CV / termination / anchor fields, and an empty
        ``extra_knobs`` dict when none is supplied."""

        md = self._build(mocker)
        assert md['_schema_version'] == SCHEMA_VERSIONS['selection']
        assert md['selection_function'] == 'vocal_onset_model_selection'
        assert md['selection_metric'] == 'AUC'
        assert md['n_splits_selection'] == 5
        assert md['test_proportion'] == 0.2
        assert md['split_strategy'] == 'session'
        assert md['random_seed'] == 123
        assert md['one_se_rule_used'] is True
        assert md['aic_termination_used'] is False
        assert md['n_anchor_features'] == 4
        assert md['anchor_feature'] == 'self.speed'
        assert md['gam_kwargs'] == {'n_splines': 10, 'lam': 0.6}
        assert md['extra_knobs'] == {}
        assert md['git_commit'] == 'selsha'
        assert md['git_dirty'] is True
        assert isinstance(md['package_version'], str)
        assert re.fullmatch(r'[0-9a-f]{64}', md['settings_sha256'])

    def test_extra_knobs_passed_through(self, mocker):
        """A supplied ``extra_knobs`` dict is stored verbatim."""

        md = self._build(mocker, extra_knobs={'max_steps': 12})
        assert md['extra_knobs'] == {'max_steps': 12}

    def test_settings_path_hashes_file(self, tmp_path, mocker):
        """A ``settings_path`` argument hashes the file bytes."""

        import hashlib
        p = tmp_path / 'modeling_settings.json'
        p.write_bytes(b'{"a": 2}\n')
        md = self._build(mocker, settings_path=str(p))
        assert md['settings_sha256'] == hashlib.sha256(b'{"a": 2}\n').hexdigest()
