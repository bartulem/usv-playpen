"""
@author: bartulem
Unit tests for the two Level-2/Level-3 consolidators —
``usv_playpen.modeling.consolidate_univariate_results`` (per-feature
pickles) and ``usv_playpen.modeling.consolidate_model_selection_results``
(per-step pickles). Both are covered here because they share helper
shapes (filename parsing, ``_diff_metadata``, metadata hoisting).

The consolidators merge a directory of SLURM-array pickles into one
artifact, asserting every part agrees on the run-level metadata before
merging. The tests pin: the filename parsers / prefix inference, the
leaf-level metadata diff, and a full ``consolidate()`` round-trip built
from synthetic pickles under ``tmp_path`` (never committed) — including
the metadata-mismatch and malformed-input failure modes.
"""

from __future__ import annotations

import pickle
import runpy
import sys
from pathlib import Path

import pytest

from usv_playpen.modeling import consolidate_model_selection_results as cms
from usv_playpen.modeling import consolidate_univariate_results as cuv


# Helpers


def _write_pkl(path, obj):
    """Pickle ``obj`` to ``path``."""

    with path.open('wb') as fh:
        pickle.dump(obj, fh)


def _univariate_feature_pkl(feat_name, md_in, md_run, data=None):
    """Build a per-feature univariate payload: one feature key plus the
    two reserved metadata blocks the consolidator requires."""

    return {feat_name: (data if data is not None else {'r2': 0.5}),
            '_input_metadata': md_in, '_run_metadata': md_run}


def _step_pkl(step_data, md_in, md_run):
    """Build a per-step selection payload: step data plus the two
    reserved metadata blocks."""

    return {**step_data, '_input_metadata': md_in, '_run_metadata': md_run}


# consolidate_univariate_results — pure helpers


class TestUnivariateHelpers:

    def test_parse_feature_idx_extracts_first_numeric_token(self):
        """The first integer token after the leading ``univariate`` tag is
        the feature index."""

        assert cuv._parse_feature_idx('univariate_mytag_0007_self.speed_ts.pkl') == 7

    def test_parse_feature_idx_returns_minus_one_on_no_match(self):
        """A filename with no numeric token parses to ``-1``."""

        assert cuv._parse_feature_idx('noindex.pkl') == -1

    def test_build_default_output_filename(self):
        """The default filename embeds the analysis tag and cohort."""

        md = {'analysis_tag': 'onsets_bout', 'experimental_condition': 'male_mute_partner'}
        out = cuv._build_default_output_filename(md)
        assert out.startswith('univariate_onsets_bout_male_mute_partner_')
        assert out.endswith('.pkl')

    def test_diff_metadata_reports_leaf_and_missing(self):
        """``_diff_metadata`` enumerates leaf disagreements and
        missing-key asymmetries, recursing into nested dicts."""

        a = {'k': 1, 'nested': {'x': 1}, 'only_a': 9}
        b = {'k': 2, 'nested': {'x': 2}, 'only_b': 8}
        diffs = cuv._diff_metadata(a, b)
        joined = '\n'.join(diffs)
        assert 'k: 1 != 2' in joined
        assert 'nested.x: 1 != 2' in joined
        assert 'only_a: missing in B' in joined
        assert 'only_b: missing in A' in joined


class TestUnivariateConsolidate:

    def _md(self):
        return ({'analysis_tag': 'tag', 'experimental_condition': 'male_mute_partner'},
                {'split_strategy': 'mixed'})

    def test_round_trip_merges_features(self, tmp_path):
        """Two matching per-feature pickles consolidate into one artifact
        carrying both features plus the hoisted metadata blocks."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl',
                   _univariate_feature_pkl('feat_a', md_in, md_run))
        _write_pkl(tmp_path / 'univariate_tag_0001_b.pkl',
                   _univariate_feature_pkl('feat_b', md_in, md_run))

        out_path = cuv.consolidate(str(tmp_path))
        with open(out_path, 'rb') as fh:
            cons = pickle.load(fh)
        assert 'feat_a' in cons and 'feat_b' in cons
        assert cons['_input_metadata'] == md_in
        assert cons['_run_metadata'] == md_run
        assert cons['_consolidation_metadata']['n_files_merged'] == 2

    def test_metadata_mismatch_raises(self, tmp_path):
        """A per-feature file disagreeing on a run-level metadata key
        aborts the merge with ``ValueError``."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl',
                   _univariate_feature_pkl('feat_a', md_in, md_run))
        _write_pkl(tmp_path / 'univariate_tag_0001_b.pkl',
                   _univariate_feature_pkl('feat_b', md_in, {'split_strategy': 'session'}))
        with pytest.raises(ValueError):
            cuv.consolidate(str(tmp_path))

    def test_input_metadata_mismatch_raises(self, tmp_path):
        """A per-feature file disagreeing on an ``_input_metadata`` key
        aborts the merge with a precise diff (exercises the input-block
        equality branch, distinct from the run-block branch)."""

        md_in, md_run = self._md()
        other_in = dict(md_in)
        other_in['analysis_tag'] = 'different'
        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl',
                   _univariate_feature_pkl('feat_a', md_in, md_run))
        _write_pkl(tmp_path / 'univariate_tag_0001_b.pkl',
                   _univariate_feature_pkl('feat_b', other_in, md_run))
        with pytest.raises(ValueError, match='_input_metadata'):
            cuv.consolidate(str(tmp_path))

    def test_multiple_feature_keys_raises(self, tmp_path):
        """A file carrying more than one feature key is malformed."""

        md_in, md_run = self._md()
        bad = {'feat_a': {}, 'feat_b': {}, '_input_metadata': md_in, '_run_metadata': md_run}
        _write_pkl(tmp_path / 'univariate_tag_0000_bad.pkl', bad)
        with pytest.raises(ValueError):
            cuv.consolidate(str(tmp_path))

    def test_legacy_without_flag_raises(self, tmp_path):
        """A pickle lacking metadata blocks is rejected unless
        ``allow_legacy`` is set."""

        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl', {'feat_a': {}})
        with pytest.raises(ValueError):
            cuv.consolidate(str(tmp_path))

    def test_empty_directory_raises(self, tmp_path):
        """A directory with no pickles raises ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            cuv.consolidate(str(tmp_path))

    def test_nonexistent_directory_raises(self, tmp_path):
        """A path that is not a directory raises ``FileNotFoundError``
        before any globbing is attempted."""

        with pytest.raises(FileNotFoundError):
            cuv.consolidate(str(tmp_path / 'does_not_exist'))

    def test_duplicate_feature_key_raises(self, tmp_path):
        """Two per-feature files carrying the *same* feature key abort
        the merge — the consolidator never silently overwrites an
        already-merged feature."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl',
                   _univariate_feature_pkl('feat_dup', md_in, md_run))
        _write_pkl(tmp_path / 'univariate_tag_0001_b.pkl',
                   _univariate_feature_pkl('feat_dup', md_in, md_run))
        with pytest.raises(ValueError, match='Duplicate feature'):
            cuv.consolidate(str(tmp_path))

    def test_allow_legacy_writes_legacy_filename(self, tmp_path):
        """With ``allow_legacy`` a metadata-less per-feature pickle is
        merged, the equality assert is skipped, and the consolidated
        artifact lands under a ``legacy_univariate_<ts>.pkl`` name
        carrying no hoisted ``_input_metadata`` / ``_run_metadata``."""

        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl', {'feat_a': {'r2': 0.1}})
        _write_pkl(tmp_path / 'univariate_tag_0001_b.pkl', {'feat_b': {'r2': 0.2}})

        out_path = cuv.consolidate(str(tmp_path), allow_legacy=True)
        assert Path(out_path).name.startswith('legacy_univariate_')
        with open(out_path, 'rb') as fh:
            cons = pickle.load(fh)
        assert 'feat_a' in cons and 'feat_b' in cons
        assert '_input_metadata' not in cons
        assert '_run_metadata' not in cons
        assert cons['_consolidation_metadata']['n_files_merged'] == 2

    def test_delete_individuals_after_removes_merged(self, tmp_path):
        """``delete_individuals_after`` unlinks every per-feature pickle
        that was merged, leaving only the consolidated artifact behind."""

        md_in, md_run = self._md()
        f0 = tmp_path / 'univariate_tag_0000_a.pkl'
        f1 = tmp_path / 'univariate_tag_0001_b.pkl'
        _write_pkl(f0, _univariate_feature_pkl('feat_a', md_in, md_run))
        _write_pkl(f1, _univariate_feature_pkl('feat_b', md_in, md_run))

        out_path = cuv.consolidate(str(tmp_path), delete_individuals_after=True)
        assert not f0.exists()
        assert not f1.exists()
        assert Path(out_path).exists()

    def test_explicit_output_dir_and_filename(self, tmp_path):
        """An explicit ``output_dir`` (created on demand) plus an
        explicit ``output_filename`` are honoured verbatim."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl',
                   _univariate_feature_pkl('feat_a', md_in, md_run))
        out_dir = tmp_path / 'consolidated_out'
        out_path = cuv.consolidate(str(tmp_path), output_dir=str(out_dir),
                                   output_filename='custom.pkl')
        assert Path(out_path) == out_dir / 'custom.pkl'
        assert Path(out_path).exists()


# consolidate_model_selection_results — pure helpers


class TestSelectionHelpers:

    def test_parse_step_idx_matches_prefix(self):
        """A filename matching ``<prefix><k>.pkl`` yields ``k``; a
        non-matching name yields ``-1``."""

        prefix = 'model_selection_test_step_'
        assert cms._parse_step_idx('model_selection_test_step_3.pkl', prefix) == 3
        assert cms._parse_step_idx('other_step_3.pkl', prefix) == -1

    def test_infer_prefix_homogeneous(self, tmp_path):
        """A directory of files sharing one descriptor yields that prefix."""

        files = [tmp_path / f'model_selection_test_step_{k}.pkl' for k in range(3)]
        for f in files:
            f.touch()
        assert cms._infer_prefix(files) == 'model_selection_test_step_'

    def test_infer_prefix_ambiguous_raises(self, tmp_path):
        """Two distinct descriptors cannot be disambiguated -> raise."""

        files = [tmp_path / 'model_selection_aaa_step_0.pkl',
                 tmp_path / 'model_selection_bbb_step_0.pkl']
        for f in files:
            f.touch()
        with pytest.raises(ValueError):
            cms._infer_prefix(files)

    def test_infer_prefix_none_match_raises(self, tmp_path):
        """No file matching the step schema -> raise."""

        files = [tmp_path / 'unrelated.pkl']
        files[0].touch()
        with pytest.raises(ValueError):
            cms._infer_prefix(files)

    def test_extract_run_timestamp(self):
        """The ``YYYYMMDD_HHMMSSZ`` token is pulled from the prefix; a
        token-less prefix yields the empty string."""

        prefix = 'model_selection_male_20260511_203829Z_bout_step_'
        assert cms._extract_run_timestamp(prefix) == '20260511_203829Z'
        assert cms._extract_run_timestamp('model_selection_test_step_') == ''

    def test_build_default_output_filename_strips_sex_suffix(self):
        """The cohort label has its trailing ``_<sex>`` stripped and the
        run timestamp is appended when present in the prefix."""

        md_in = {'target_mouse_sex': 'male',
                 'experimental_condition': 'intact_partners_male',
                 'analysis_tag': 'bout'}
        md_run = {'split_strategy': 'mixed'}
        prefix = 'model_selection_male_20260511_203829Z_bout_step_'
        out = cms._build_default_output_filename(md_in, md_run, step_prefix=prefix)
        assert out == ('model_selection_final_male_intact_partners_bout_'
                       'mixed_20260511_203829Z.pkl')

    def test_build_default_output_filename_unknown_fallback(self):
        """Missing metadata blocks fall back to ``'unknown'`` tokens."""

        out = cms._build_default_output_filename(None, None, step_prefix='')
        assert out == 'model_selection_final_unknown_unknown_unknown_unknown.pkl'

    def test_build_default_output_filename_grafts_category_index_tag(self):
        """A bare ``category_<idx>`` analysis_tag is rewritten to
        ``category_<col>_<idx>`` by grafting the USV category column out
        of the ``analysis_specific`` block."""

        md_in = {'target_mouse_sex': 'female',
                 'experimental_condition': 'intact_partners_female',
                 'analysis_tag': 'category_3',
                 'analysis_specific': {'usv_category_column_name': 'vae_supercategory'}}
        md_run = {'split_strategy': 'session'}
        out = cms._build_default_output_filename(md_in, md_run, step_prefix='')
        assert out == ('model_selection_final_female_intact_partners_'
                       'category_vae_supercategory_3_session.pkl')

    def test_build_default_output_filename_grafts_bare_tag(self):
        """A non-``category_`` bare tag (e.g. ``multinomial``) gets the
        USV category column appended as ``<tag>_<col>``."""

        md_in = {'target_mouse_sex': 'male',
                 'experimental_condition': 'male_mute_partner',
                 'analysis_tag': 'multinomial',
                 'analysis_specific': {'usv_category_column_name': 'vae_supercategory'}}
        md_run = {'split_strategy': 'mixed'}
        out = cms._build_default_output_filename(md_in, md_run, step_prefix='')
        assert out == ('model_selection_final_male_male_mute_partner_'
                       'multinomial_vae_supercategory_mixed.pkl')

    def test_diff_metadata_reports_leaf_and_missing(self):
        """The selection consolidator's ``_diff_metadata`` (duplicated
        from the univariate one) enumerates leaf disagreements,
        recurses into nested dicts, and flags missing-key asymmetries."""

        a = {'k': 1, 'nested': {'x': 1}, 'only_a': 9}
        b = {'k': 2, 'nested': {'x': 2}, 'only_b': 8}
        diffs = cms._diff_metadata(a, b)
        joined = '\n'.join(diffs)
        assert 'k: 1 != 2' in joined
        assert 'nested.x: 1 != 2' in joined
        assert 'only_a: missing in B' in joined
        assert 'only_b: missing in A' in joined


class TestSelectionConsolidate:

    def _md(self):
        return ({'target_mouse_sex': 'male',
                 'experimental_condition': 'intact_partners_male',
                 'analysis_tag': 'bout'},
                {'split_strategy': 'mixed'})

    def test_round_trip_merges_steps(self, tmp_path):
        """Two matching per-step pickles consolidate into a ``steps`` list
        (in step order) plus the hoisted metadata blocks."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        _write_pkl(tmp_path / 'model_selection_test_step_1.pkl',
                   _step_pkl({'selected_feature': 'b'}, md_in, md_run))

        out_path = cms.consolidate(str(tmp_path), prefix='model_selection_test_step_')
        with open(out_path, 'rb') as fh:
            cons = pickle.load(fh)
        assert isinstance(cons['steps'], list)
        assert len(cons['steps']) == 2
        assert cons['steps'][0]['selected_feature'] == 'a'
        assert cons['steps'][1]['selected_feature'] == 'b'
        assert cons['_input_metadata'] == md_in
        assert cons['_consolidation_metadata']['n_files_merged'] == 2

    def test_metadata_mismatch_raises(self, tmp_path):
        """A step file disagreeing on run-level metadata aborts the
        merge."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        _write_pkl(tmp_path / 'model_selection_test_step_1.pkl',
                   _step_pkl({'selected_feature': 'b'}, md_in, {'split_strategy': 'session'}))
        with pytest.raises(ValueError):
            cms.consolidate(str(tmp_path), prefix='model_selection_test_step_')

    def test_empty_directory_raises(self, tmp_path):
        """An empty directory raises ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            cms.consolidate(str(tmp_path), prefix='model_selection_test_step_')

    def test_nonexistent_directory_raises(self, tmp_path):
        """A path that is not a directory raises ``FileNotFoundError``."""

        with pytest.raises(FileNotFoundError):
            cms.consolidate(str(tmp_path / 'nope'),
                            prefix='model_selection_test_step_')

    def test_prefix_inferred_when_omitted(self, tmp_path):
        """With ``prefix=None`` the consolidator infers the shared
        ``model_selection_..._step_`` prefix from the directory, hoists
        the ``_univariate_metadata`` block when present, and merges."""

        md_in, md_run = self._md()
        md_univ = {'model_engine': 'sklearn'}
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   {'selected_feature': 'a', '_input_metadata': md_in,
                    '_run_metadata': md_run, '_univariate_metadata': md_univ})
        _write_pkl(tmp_path / 'model_selection_test_step_1.pkl',
                   {'selected_feature': 'b', '_input_metadata': md_in,
                    '_run_metadata': md_run, '_univariate_metadata': md_univ})

        out_path = cms.consolidate(str(tmp_path))  # prefix inferred
        with open(out_path, 'rb') as fh:
            cons = pickle.load(fh)
        assert len(cons['steps']) == 2
        assert cons['_univariate_metadata'] == md_univ

    def test_univariate_metadata_mismatch_raises(self, tmp_path):
        """A step file disagreeing on ``_univariate_metadata`` aborts the
        merge (both files must carry the block for the check to fire)."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   {'selected_feature': 'a', '_input_metadata': md_in,
                    '_run_metadata': md_run,
                    '_univariate_metadata': {'model_engine': 'sklearn'}})
        _write_pkl(tmp_path / 'model_selection_test_step_1.pkl',
                   {'selected_feature': 'b', '_input_metadata': md_in,
                    '_run_metadata': md_run,
                    '_univariate_metadata': {'model_engine': 'pygam'}})
        with pytest.raises(ValueError, match='_univariate_metadata'):
            cms.consolidate(str(tmp_path), prefix='model_selection_test_step_')

    def test_input_metadata_mismatch_raises(self, tmp_path):
        """A step file disagreeing on ``_input_metadata`` aborts."""

        md_in, md_run = self._md()
        other_in = dict(md_in)
        other_in['analysis_tag'] = 'different'
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        _write_pkl(tmp_path / 'model_selection_test_step_1.pkl',
                   _step_pkl({'selected_feature': 'b'}, other_in, md_run))
        with pytest.raises(ValueError, match='_input_metadata'):
            cms.consolidate(str(tmp_path), prefix='model_selection_test_step_')

    def test_no_files_match_prefix_raises(self, tmp_path):
        """Pickles present but none matching the explicit ``prefix``
        raise ``FileNotFoundError``."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        with pytest.raises(FileNotFoundError, match='match prefix'):
            cms.consolidate(str(tmp_path), prefix='model_selection_other_step_')

    def test_allow_legacy_writes_legacy_filename(self, tmp_path):
        """With ``allow_legacy`` metadata-less step pickles merge and the
        artifact lands under a ``legacy_selection_<ts>.pkl`` name with no
        hoisted upstream metadata blocks."""

        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   {'selected_feature': 'a'})
        _write_pkl(tmp_path / 'model_selection_test_step_1.pkl',
                   {'selected_feature': 'b'})

        out_path = cms.consolidate(str(tmp_path),
                                   prefix='model_selection_test_step_',
                                   allow_legacy=True)
        assert Path(out_path).name.startswith('legacy_selection_')
        with open(out_path, 'rb') as fh:
            cons = pickle.load(fh)
        assert len(cons['steps']) == 2
        assert '_input_metadata' not in cons
        assert '_run_metadata' not in cons

    def test_legacy_without_flag_raises(self, tmp_path):
        """A metadata-less step pickle is rejected unless
        ``allow_legacy`` is set."""

        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   {'selected_feature': 'a'})
        with pytest.raises(ValueError, match='allow_legacy'):
            cms.consolidate(str(tmp_path), prefix='model_selection_test_step_')

    def test_noncontiguous_step_indices_warn_but_merge(self, tmp_path, capsys):
        """A gap in the step indices (step_0 + step_2, no step_1) emits a
        non-contiguous warning but still consolidates what is on disk."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        _write_pkl(tmp_path / 'model_selection_test_step_2.pkl',
                   _step_pkl({'selected_feature': 'c'}, md_in, md_run))

        out_path = cms.consolidate(str(tmp_path),
                                   prefix='model_selection_test_step_')
        captured = capsys.readouterr()
        assert 'not contiguous' in captured.out
        with open(out_path, 'rb') as fh:
            cons = pickle.load(fh)
        assert len(cons['steps']) == 2

    def test_move_to_steps_subdir_relocates(self, tmp_path):
        """``move_to_steps_subdir`` relocates every consumed step pickle
        into ``<input_dir>/steps/`` after consolidation succeeds."""

        md_in, md_run = self._md()
        f0 = tmp_path / 'model_selection_test_step_0.pkl'
        f1 = tmp_path / 'model_selection_test_step_1.pkl'
        _write_pkl(f0, _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        _write_pkl(f1, _step_pkl({'selected_feature': 'b'}, md_in, md_run))

        out_path = cms.consolidate(str(tmp_path),
                                   prefix='model_selection_test_step_',
                                   move_to_steps_subdir=True)
        steps_dir = tmp_path / 'steps'
        assert (steps_dir / f0.name).exists()
        assert (steps_dir / f1.name).exists()
        assert not f0.exists()
        assert not f1.exists()
        assert Path(out_path).exists()

    def test_explicit_output_dir_and_filename(self, tmp_path):
        """An explicit ``output_dir`` / ``output_filename`` is honoured."""

        md_in, md_run = self._md()
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        out_dir = tmp_path / 'final'
        out_path = cms.consolidate(str(tmp_path),
                                   prefix='model_selection_test_step_',
                                   output_dir=str(out_dir),
                                   output_filename='final.pkl')
        assert Path(out_path) == out_dir / 'final.pkl'


# Module-level CLI (`python -m ...`) entry points


@pytest.mark.filterwarnings(
    "ignore:.*found in sys.modules after import of package.*:RuntimeWarning"
)
class TestConsolidatorCLI:
    """Drive both consolidators through their ``__main__`` argparse
    blocks in-process via ``runpy.run_module(..., run_name='__main__')``
    so the coverage tracer sees the guard body. Each test patches
    ``sys.argv`` and asserts both the happy path (``OK:`` printed) and
    the error path (``FATAL`` + ``SystemExit(1)`` on a missing input
    directory)."""

    @staticmethod
    def _run_main(module_name, argv, monkeypatch):
        """Execute ``module_name`` as ``__main__`` with the supplied
        ``argv`` (the leading program-name slot is filled in here).
        Returns the captured ``SystemExit`` (or ``None`` if the module
        ran to completion without raising)."""

        monkeypatch.setattr(sys, 'argv', [module_name, *argv])
        try:
            runpy.run_module(module_name, run_name='__main__')
            return None
        except SystemExit as exc:
            return exc

    def test_univariate_cli_round_trip(self, tmp_path, monkeypatch, capsys):
        """The univariate ``__main__`` block consolidates a synthetic
        per-feature directory and prints an ``OK:`` line."""

        md_in = {'analysis_tag': 'tag', 'experimental_condition': 'male_mute_partner'}
        md_run = {'split_strategy': 'mixed'}
        _write_pkl(tmp_path / 'univariate_tag_0000_a.pkl',
                   _univariate_feature_pkl('feat_a', md_in, md_run))
        exc = self._run_main(
            'usv_playpen.modeling.consolidate_univariate_results',
            ['--input_dir', str(tmp_path)], monkeypatch,
        )
        assert exc is None
        assert 'OK:' in capsys.readouterr().out

    def test_univariate_cli_missing_dir_exits_1(self, tmp_path, monkeypatch, capsys):
        """A missing input directory drives the univariate CLI to a
        ``FATAL`` message and ``SystemExit(1)``."""

        exc = self._run_main(
            'usv_playpen.modeling.consolidate_univariate_results',
            ['--input_dir', str(tmp_path / 'absent')], monkeypatch,
        )
        assert isinstance(exc, SystemExit) and exc.code == 1
        assert 'FATAL' in capsys.readouterr().err

    def test_selection_cli_round_trip(self, tmp_path, monkeypatch, capsys):
        """The selection ``__main__`` block consolidates a synthetic
        per-step directory and prints an ``OK:`` line."""

        md_in = {'target_mouse_sex': 'male',
                 'experimental_condition': 'intact_partners_male',
                 'analysis_tag': 'bout'}
        md_run = {'split_strategy': 'mixed'}
        _write_pkl(tmp_path / 'model_selection_test_step_0.pkl',
                   _step_pkl({'selected_feature': 'a'}, md_in, md_run))
        exc = self._run_main(
            'usv_playpen.modeling.consolidate_model_selection_results',
            ['--input_dir', str(tmp_path),
             '--prefix', 'model_selection_test_step_'], monkeypatch,
        )
        assert exc is None
        assert 'OK:' in capsys.readouterr().out

    def test_selection_cli_missing_dir_exits_1(self, tmp_path, monkeypatch, capsys):
        """A missing input directory drives the selection CLI to a
        ``FATAL`` message and ``SystemExit(1)``."""

        exc = self._run_main(
            'usv_playpen.modeling.consolidate_model_selection_results',
            ['--input_dir', str(tmp_path / 'absent'),
             '--prefix', 'model_selection_test_step_'], monkeypatch,
        )
        assert isinstance(exc, SystemExit) and exc.code == 1
        assert 'FATAL' in capsys.readouterr().err
