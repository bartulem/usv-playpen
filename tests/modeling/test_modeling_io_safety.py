"""
@author: bartulem
Crash-safety tests for the modeling stage's persisted artifacts.

Every pickle the modeling stage writes (input artifacts, univariate /
selection step pickles, consolidated artifacts, GLM-HMM / CNN / audit
results) is read back later -- by the next pipeline stage, by a resumed
selection run, or by the figure code. A write straight to the final path
truncates the previous file before the new bytes land, so a crash, kill or
full disk mid-write destroys it. The modeling writers therefore publish
through ``os_utils.atomic_output_path`` (temp sibling + ``os.replace``).

Coverage:
- a static guard: every ``pickle.dump`` call in ``src/usv_playpen/modeling``
  sits inside a ``with`` statement that enters ``atomic_output_path``;
- behaviour: a failed write in either consolidator propagates and leaves the
  previously published consolidated artifact byte-identical, with no
  temporary sibling left behind.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import pytest

from usv_playpen.modeling import consolidate_model_selection_results as cms
from usv_playpen.modeling import consolidate_univariate_results as cuv

MODELING_DIR = Path(cuv.__file__).resolve().parent


def _non_atomic_pickle_dumps(source: str) -> list[int]:
    """
    Description
    -----------
    Lists the line numbers of ``pickle.dump`` calls that are not lexically
    inside a ``with`` statement entering ``atomic_output_path(...)``.

    Parameters
    ----------
    source (str)
        Python source of one module.

    Returns
    -------
    lines (list[int])
        Line numbers of the offending ``pickle.dump`` calls (empty when every
        dump is atomic).
    """

    tree = ast.parse(source)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def _is_atomic_with(node) -> bool:
        """
        Description
        -----------
        Whether ``node`` is a ``with`` statement one of whose items calls
        ``atomic_output_path``.

        Parameters
        ----------
        node (ast.AST)
            Candidate ancestor node.

        Returns
        -------
        is_atomic (bool)
            True for a ``with atomic_output_path(...)`` statement.
        """

        if not isinstance(node, ast.With):
            return False
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call):
                func = call.func
                name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
                if name == 'atomic_output_path':
                    return True
        return False

    offending = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'dump' and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'pickle'):
            # The module root has no parent entry: it ends the ancestor walk as None.
            ancestor = parents[node] if node in parents else None
            while ancestor is not None and not _is_atomic_with(ancestor):
                ancestor = parents[ancestor] if ancestor in parents else None
            if ancestor is None:
                offending.append(node.lineno)
    return offending


@pytest.mark.parametrize('module_path', sorted(MODELING_DIR.glob('*.py')), ids=lambda p: p.name)
def test_every_modeling_pickle_write_is_atomic(module_path):
    """
    Every ``pickle.dump`` in a modeling module must write through
    ``atomic_output_path`` so a failed write never truncates the artifact a
    later stage (or a resumed run) reads back.
    """

    offending = _non_atomic_pickle_dumps(module_path.read_text())
    assert offending == [], f"{module_path.name}: non-atomic pickle.dump at lines {offending}"


def _partial_then_boom(_obj, handle, *_args, **_kwargs):
    """
    Description
    -----------
    Stand-in for ``pickle.dump`` that writes a few bytes and then fails, as a
    crash or a full disk would mid-write.

    Parameters
    ----------
    _obj (object)
        The object that would have been pickled (ignored).
    handle (io.BufferedWriter)
        The open binary handle the dump writes to.
    _args (tuple)
        Further positional arguments of ``pickle.dump`` (ignored).
    _kwargs (dict)
        Further keyword arguments of ``pickle.dump`` (ignored).

    Returns
    -------
    None
        Never returns; always raises ``OSError``.
    """

    handle.write(b'partial')
    message = "simulated write failure"
    raise OSError(message)


def test_univariate_consolidation_failure_keeps_published_artifact(tmp_path, monkeypatch):
    """
    A univariate consolidation whose write fails must raise and leave the
    previously published consolidated artifact intact (it used to open the
    final path with 'wb', truncating it before the dump failed).
    """

    in_dir = tmp_path / 'in'
    out_dir = tmp_path / 'out'
    in_dir.mkdir()
    md_in = {'analysis_tag': 'tag', 'experimental_condition': 'male_mute_partner'}
    md_run = {'split_strategy': 'mixed'}
    with (in_dir / 'univariate_tag_0000_a.pkl').open('wb') as fh:
        pickle.dump({'feat_a': {'r2': 0.5}, '_input_metadata': md_in, '_run_metadata': md_run}, fh)

    published = Path(cuv.consolidate(str(in_dir), output_dir=str(out_dir), output_filename='cons.pkl'))
    original = published.read_bytes()

    monkeypatch.setattr(cuv.pickle, 'dump', _partial_then_boom)
    with pytest.raises(OSError, match="simulated write failure"):
        cuv.consolidate(str(in_dir), output_dir=str(out_dir), output_filename='cons.pkl')
    assert published.read_bytes() == original
    assert sorted(p.name for p in out_dir.iterdir()) == ['cons.pkl']


def test_selection_consolidation_failure_keeps_published_artifact(tmp_path, monkeypatch):
    """
    The model-selection consolidator obeys the same rule: a failed write
    raises and the previously published consolidated artifact survives
    byte-identical, with no temporary sibling left behind.
    """

    in_dir = tmp_path / 'in'
    out_dir = tmp_path / 'out'
    in_dir.mkdir()
    md_in = {'target_mouse_sex': 'male', 'experimental_condition': 'intact_partners_male',
             'analysis_tag': 'bout', 'analysis_specific': {}}
    md_run = {'split_strategy': 'mixed'}
    with (in_dir / 'model_selection_test_step_0.pkl').open('wb') as fh:
        pickle.dump({'selected_feature': 'a', '_input_metadata': md_in, '_run_metadata': md_run}, fh)

    published = Path(cms.consolidate(str(in_dir), prefix='model_selection_test_step_',
                                     output_dir=str(out_dir), output_filename='cons.pkl'))
    original = published.read_bytes()

    monkeypatch.setattr(cms.pickle, 'dump', _partial_then_boom)
    with pytest.raises(OSError, match="simulated write failure"):
        cms.consolidate(str(in_dir), prefix='model_selection_test_step_',
                        output_dir=str(out_dir), output_filename='cons.pkl')
    assert published.read_bytes() == original
    assert sorted(p.name for p in out_dir.iterdir()) == ['cons.pkl']
