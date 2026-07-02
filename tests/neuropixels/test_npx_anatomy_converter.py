"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.anatomy_converter``.

The module regenerates the sites-to-anatomy converter JSON so its
per-probe region ranges live in Kilosort-row space. These tests cover
the pure run-length compressor with hand-computed ranges, and drive the
file-joining helpers / top-level regenerator against a fully synthetic
ephys + histology directory tree built under ``tmp_path`` (no real data,
no network).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from usv_playpen.neuropixels.anatomy_converter import (
    _build_ks_keyed_block,
    _cli,
    _load_ibl_position_to_region,
    _runs_to_ranges,
    _write_converter,
    add_session_to_anatomy_converter,
    regenerate_anatomy_converter,
)


def test_runs_to_ranges_empty_is_empty_dict():
    """
    Description
    -----------
    An empty per-row sequence compresses to an empty mapping (no
    region runs to record).
    """

    assert _runs_to_ranges([]) == {}


def test_runs_to_ranges_single_channel():
    """
    Description
    -----------
    A single channel yields a single half-open run ``[0, 1)`` for its
    region.
    """

    assert _runs_to_ranges(["PAG"]) == {"PAG": [[0, 1]]}


def test_runs_to_ranges_maximal_contiguous_runs():
    """
    Description
    -----------
    Adjacent equal labels collapse into one maximal ``[lo, hi)`` run;
    distinct regions each get their own run.
    """

    out = _runs_to_ranges(["VISp", "VISp", "VISa", "VISa", "VISa", "PAG"])
    assert out == {"VISp": [[0, 2]], "VISa": [[2, 5]], "PAG": [[5, 6]]}


def test_runs_to_ranges_interleaved_regions_get_multiple_runs():
    """
    Description
    -----------
    A region that reappears after another region gets a second
    (disjoint) run rather than being merged across the gap.
    """

    out = _runs_to_ranges(["A", "B", "A"])
    assert out == {"A": [[0, 1], [2, 3]], "B": [[1, 2]]}


def _write_ibl_json(histology_root, mouse_id, rec_date, hemi_dir, entries):
    """
    Description
    -----------
    Write a synthetic IBL ``channel_locations.json`` under the expected
    ``<mouse>/<rec_date>/<hemi_dir>/`` layout.

    Parameters
    ----------
    histology_root : pathlib.Path
        Root histology directory.
    mouse_id : str
        Mouse identifier.
    rec_date : int
        Recording date (``YYYYMMDD``).
    hemi_dir : str
        Hemisphere subdir (``ibl_RH`` / ``ibl_LH``).
    entries : dict
        Raw JSON content to dump.

    Returns
    -------
    pathlib.Path
        The written JSON path.
    """

    d = histology_root / mouse_id / str(rec_date) / hemi_dir
    d.mkdir(parents=True, exist_ok=True)
    path = d / "channel_locations.json"
    path.write_text(json.dumps(entries))
    return path


def test_load_ibl_position_to_region_keys_by_position_and_skips_non_channel(tmp_path):
    """
    Description
    -----------
    The loader maps integer ``(lateral, axial)`` to brain region for
    every ``channel_*`` entry and ignores non-channel keys (e.g.
    ``origin``). Hemisphere ``'R'`` selects the ``ibl_RH`` subdir.
    """

    _write_ibl_json(
        tmp_path, "mouse_A", 20240101, "ibl_RH",
        {
            "channel_0": {"lateral": 0, "axial": 0, "brain_region": "PAG"},
            "channel_1": {"lateral": 0, "axial": 20, "brain_region": "VISp"},
            "origin": {"x": 1, "y": 2},
        },
    )
    pos = _load_ibl_position_to_region(tmp_path, "mouse_A", 20240101, "R")
    assert pos == {(0, 0): "PAG", (0, 20): "VISp"}


def test_build_ks_keyed_block_joins_positions_and_marks_unknown(tmp_path):
    """
    Description
    -----------
    ``_build_ks_keyed_block`` reads ``channel_positions.npy``, looks up
    each KS row's region by physical position, labels positions absent
    from the IBL map ``"unknown"``, then compresses contiguous runs.
    """

    ks_dir = tmp_path / "kilosort4"
    ks_dir.mkdir()
    np.save(
        ks_dir / "channel_positions.npy",
        np.array([[0, 0], [0, 20], [0, 40]], dtype=np.int64),
    )
    pos_to_region = {(0, 0): "PAG", (0, 20): "PAG"}
    block = _build_ks_keyed_block(ks_dir, pos_to_region)
    assert block == {"PAG": [[0, 2]], "unknown": [[2, 3]]}


def _make_full_dataset(tmp_path):
    """
    Description
    -----------
    Build a minimal but complete ephys + histology + converter tree for
    one ``(mouse_A, 20240101_120000, imec0)`` probe-day with three KS
    rows spanning two regions.

    Returns
    -------
    tuple
        ``(converter_path, ephys_root, histology_root)``.
    """

    ephys_root = tmp_path / "ephys"
    histology_root = tmp_path / "histology"
    ks_dir = ephys_root / "20240101_imec0" / "kilosort4"
    ks_dir.mkdir(parents=True)
    np.save(
        ks_dir / "channel_positions.npy",
        np.array([[0, 0], [0, 20], [0, 40]], dtype=np.int64),
    )
    _write_ibl_json(
        histology_root, "mouse_A", 20240101, "ibl_RH",
        {
            "channel_0": {"lateral": 0, "axial": 0, "brain_region": "PAG"},
            "channel_1": {"lateral": 0, "axial": 20, "brain_region": "PAG"},
            "channel_2": {"lateral": 0, "axial": 40, "brain_region": "VISp"},
        },
    )
    converter_path = tmp_path / "converter.json"
    converter_path.write_text(
        json.dumps({"mouse_A": {"20240101_120000": {"imec0": {}}}})
    )
    return converter_path, ephys_root, histology_root


def test_regenerate_dry_run_returns_ks_keyed_block(tmp_path):
    """
    Description
    -----------
    A dry run regenerates the per-probe block in KS-row space and
    returns it in-memory without touching the converter file on disk.
    """

    converter_path, ephys_root, histology_root = _make_full_dataset(tmp_path)
    before = converter_path.read_text()

    summary = regenerate_anatomy_converter(
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        dry_run=True,
    )
    assert summary["n_triples_total"] == 1
    assert summary["n_triples_regenerated"] == 1
    assert summary["n_triples_skipped"] == 0
    assert summary["output"]["mouse_A"]["20240101_120000"]["imec0"] == {
        "PAG": [[0, 2]],
        "VISp": [[2, 3]],
    }
    assert converter_path.read_text() == before  # untouched on dry run


def test_regenerate_writes_file_when_not_dry_run(tmp_path):
    """
    Description
    -----------
    A non-dry run rewrites the converter JSON on disk; the written file
    re-parses to the KS-row-keyed structure and ``output`` is the path.
    """

    converter_path, ephys_root, histology_root = _make_full_dataset(tmp_path)
    summary = regenerate_anatomy_converter(
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        dry_run=False,
    )
    assert summary["output"] == str(converter_path)
    written = json.loads(converter_path.read_text())
    assert written["mouse_A"]["20240101_120000"]["imec0"] == {
        "PAG": [[0, 2]],
        "VISp": [[2, 3]],
    }


def test_regenerate_records_skip_reasons(tmp_path):
    """
    Description
    -----------
    Each unusable triple is skipped with a descriptive reason: a
    non-date session id, a missing ``channel_positions.npy``, and a
    probe with no hemisphere mapping all land in ``skipped_reasons``
    without aborting the run.
    """

    ephys_root = tmp_path / "ephys"
    histology_root = tmp_path / "histology"
    ephys_root.mkdir()
    histology_root.mkdir()
    # imec9 has valid positions so it gets past the channel-positions
    # guard and reaches the (missing) hemisphere mapping.
    ks_dir_imec9 = ephys_root / "20240101_imec9" / "kilosort4"
    ks_dir_imec9.mkdir(parents=True)
    np.save(ks_dir_imec9 / "channel_positions.npy", np.array([[0, 0]], dtype=np.int64))
    converter_path = tmp_path / "converter.json"
    converter_path.write_text(json.dumps({
        "mouse_A": {
            "not_a_date": {"imec0": {}},               # bad session id
            "20240101_120000": {
                "imec0": {},                            # no channel_positions.npy
                "imec9": {},                            # no hemisphere mapping
            },
        }
    }))

    summary = regenerate_anatomy_converter(
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        dry_run=True,
    )
    assert summary["n_triples_regenerated"] == 0
    assert summary["n_triples_total"] == 3
    reasons = " | ".join(summary["skipped_reasons"])
    assert "bad session_id" in reasons
    assert "no channel_positions.npy" in reasons
    assert "no hemisphere mapping" in reasons


def test_regenerate_skips_when_ibl_json_missing(tmp_path):
    """
    Description
    -----------
    When the Kilosort positions exist and the probe maps to a
    hemisphere but the IBL ``channel_locations.json`` is absent, the
    triple is skipped with the IBL-missing reason.
    """

    ephys_root = tmp_path / "ephys"
    histology_root = tmp_path / "histology"
    ks_dir = ephys_root / "20240101_imec0" / "kilosort4"
    ks_dir.mkdir(parents=True)
    np.save(ks_dir / "channel_positions.npy", np.array([[0, 0]], dtype=np.int64))
    histology_root.mkdir()
    converter_path = tmp_path / "converter.json"
    converter_path.write_text(
        json.dumps({"mouse_A": {"20240101_120000": {"imec0": {}}}})
    )

    summary = regenerate_anatomy_converter(
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        dry_run=True,
    )
    assert summary["n_triples_regenerated"] == 0
    assert "no IBL channel_locations.json" in summary["skipped_reasons"][0]


def test_cli_dry_run_prints_summary(tmp_path, monkeypatch, capsys):
    """
    Description
    -----------
    The argparse entry point wires its flags through to
    :func:`regenerate_anatomy_converter` and prints a JSON summary; a
    ``--regenerate-all --dry-run`` invocation over a synthetic dataset reports
    one regenerated triple without writing the converter.
    """

    converter_path, ephys_root, histology_root = _make_full_dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--converter-path", str(converter_path),
        "--ephys-root", str(ephys_root),
        "--histology-root", str(histology_root),
        "--regenerate-all",
        "--dry-run",
    ])
    _cli()
    out = json.loads(capsys.readouterr().out)
    assert out["n_triples_regenerated"] == 1
    assert out["output"] == "(dict returned in-memory)"


def test_add_session_adds_new_triple_and_preserves_others(tmp_path):
    """
    Description
    -----------
    Adding a ``(mouse, session, probe)`` absent from the converter builds
    its KS-row-keyed block and merges it in, leaving every other entry
    byte-for-byte intact.
    """

    _, ephys_root, histology_root = _make_full_dataset(tmp_path)
    converter_path = tmp_path / "conv_add.json"
    converter_path.write_text(
        json.dumps({"other_mouse": {"20230101_000000": {"imec1": {"AAA": [[0, 5]]}}}})
    )
    summary = add_session_to_anatomy_converter(
        "mouse_A", "20240101_120000", "imec0",
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
    )
    assert summary["status"] == "added"
    conv = json.loads(converter_path.read_text())
    assert conv["mouse_A"]["20240101_120000"]["imec0"] == {"PAG": [[0, 2]], "VISp": [[2, 3]]}
    assert conv["other_mouse"] == {"20230101_000000": {"imec1": {"AAA": [[0, 5]]}}}


def test_add_session_already_present_is_noop(tmp_path):
    """
    Description
    -----------
    Add-if-missing: when the triple is already in the converter and
    ``force`` is False, it is left untouched and the file is not rewritten.
    """

    _, ephys_root, histology_root = _make_full_dataset(tmp_path)
    converter_path = tmp_path / "conv_noop.json"
    converter_path.write_text(
        json.dumps({"mouse_A": {"20240101_120000": {"imec0": {"OLD": [[0, 3]]}}}})
    )
    before = converter_path.read_text()
    summary = add_session_to_anatomy_converter(
        "mouse_A", "20240101_120000", "imec0",
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
    )
    assert summary["status"] == "already_present"
    assert converter_path.read_text() == before


def test_add_session_force_regenerates_existing(tmp_path):
    """
    Description
    -----------
    ``force=True`` rebuilds and overwrites an already-present triple's
    block (status ``'updated'``).
    """

    _, ephys_root, histology_root = _make_full_dataset(tmp_path)
    converter_path = tmp_path / "conv_force.json"
    converter_path.write_text(
        json.dumps({"mouse_A": {"20240101_120000": {"imec0": {"OLD": [[0, 3]]}}}})
    )
    summary = add_session_to_anatomy_converter(
        "mouse_A", "20240101_120000", "imec0", force=True,
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
    )
    assert summary["status"] == "updated"
    conv = json.loads(converter_path.read_text())
    assert conv["mouse_A"]["20240101_120000"]["imec0"] == {"PAG": [[0, 2]], "VISp": [[2, 3]]}


def test_add_session_creates_converter_if_absent(tmp_path):
    """
    Description
    -----------
    When the converter file does not exist yet, it is created holding just
    the newly added triple.
    """

    _, ephys_root, histology_root = _make_full_dataset(tmp_path)
    converter_path = tmp_path / "does_not_exist.json"
    summary = add_session_to_anatomy_converter(
        "mouse_A", "20240101_120000", "imec0",
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
    )
    assert summary["status"] == "added"
    assert converter_path.exists()
    conv = json.loads(converter_path.read_text())
    assert conv == {"mouse_A": {"20240101_120000": {"imec0": {"PAG": [[0, 2]], "VISp": [[2, 3]]}}}}


def test_add_session_skips_when_data_missing_without_writing(tmp_path):
    """
    Description
    -----------
    A triple whose Kilosort output is missing is skipped (status
    ``'skipped'`` with a reason) and nothing is written to the converter.
    """

    converter_path = tmp_path / "conv_skip.json"
    converter_path.write_text(json.dumps({}))
    summary = add_session_to_anatomy_converter(
        "mouse_A", "20240101_120000", "imec0",
        converter_path=converter_path,
        ephys_root=tmp_path / "no_ephys",
        histology_root=tmp_path / "no_hist",
    )
    assert summary["status"] == "skipped"
    assert "no channel_positions.npy" in summary["reason"]
    assert json.loads(converter_path.read_text()) == {}


def test_regenerate_preserves_existing_block_for_skipped_triple(tmp_path):
    """
    Description
    -----------
    The batch regenerator seeds from the existing converter, so a triple it
    cannot regenerate (here ``imec7`` — no Kilosort output) keeps its
    previous on-disk block instead of being dropped when the file is
    rewritten, while a regenerable sibling (``imec0``) is updated.
    """

    _, ephys_root, histology_root = _make_full_dataset(tmp_path)
    converter_path = tmp_path / "conv_preserve.json"
    converter_path.write_text(json.dumps({
        "mouse_A": {"20240101_120000": {
            "imec0": {"OLD": [[0, 3]]},
            "imec7": {"KEEP_ME": [[0, 9]]},
        }}
    }))
    summary = regenerate_anatomy_converter(
        converter_path=converter_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        dry_run=False,
    )
    conv = json.loads(converter_path.read_text())
    assert conv["mouse_A"]["20240101_120000"]["imec0"] == {"PAG": [[0, 2]], "VISp": [[2, 3]]}
    assert conv["mouse_A"]["20240101_120000"]["imec7"] == {"KEEP_ME": [[0, 9]]}
    assert summary["n_triples_regenerated"] == 1
    assert summary["n_triples_skipped"] == 1


def test_write_converter_keeps_region_rows_on_one_line(tmp_path):
    """
    Description
    -----------
    The shared guarded writer collapses each region's ``[[lo, hi], ...]``
    list onto a single line and round-trips to an identical dict.
    """

    converter_path = tmp_path / "fmt.json"
    payload = {"m": {"s": {"imec0": {"PAG": [[0, 40], [72, 136]], "MRN": [[40, 72]]}}}}
    _write_converter(converter_path, payload)
    txt = converter_path.read_text()
    assert '"PAG": [[0, 40], [72, 136]]' in txt
    assert '"MRN": [[40, 72]]' in txt
    assert json.loads(txt) == payload


def test_cli_single_triple_mode_adds_one_triple(tmp_path, monkeypatch, capsys):
    """
    Description
    -----------
    Passing ``--mouse/--session/--probe`` routes the CLI to the
    incremental adder, which merges just that triple into the converter.
    """

    _, ephys_root, histology_root = _make_full_dataset(tmp_path)
    converter_path = tmp_path / "conv_cli_single.json"
    converter_path.write_text(json.dumps({}))
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--converter-path", str(converter_path),
        "--ephys-root", str(ephys_root),
        "--histology-root", str(histology_root),
        "--mouse", "mouse_A",
        "--session", "20240101_120000",
        "--probe", "imec0",
    ])
    _cli()
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "added"
    conv = json.loads(converter_path.read_text())
    assert conv["mouse_A"]["20240101_120000"]["imec0"] == {"PAG": [[0, 2]], "VISp": [[2, 3]]}


def test_cli_bare_invocation_prints_help_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """
    Description
    -----------
    With no action flag (neither ``--regenerate-all`` nor
    ``--mouse``/``--session``/``--probe``) the CLI prints usage and leaves the
    converter untouched, so a stray bare invocation can never overwrite it.
    """

    converter_path, ephys_root, histology_root = _make_full_dataset(tmp_path)
    before = converter_path.read_text()
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--converter-path", str(converter_path),
        "--ephys-root", str(ephys_root),
        "--histology-root", str(histology_root),
    ])
    _cli()
    out = capsys.readouterr().out
    assert "usage" in out.lower()
    assert converter_path.read_text() == before


def test_cli_regenerate_all_and_single_triple_are_mutually_exclusive(tmp_path, monkeypatch):
    """
    Description
    -----------
    Combining ``--regenerate-all`` with the single-triple flags is rejected
    (argparse ``error`` raises ``SystemExit``), so the two modes cannot be
    requested at once.
    """

    converter_path, ephys_root, histology_root = _make_full_dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--converter-path", str(converter_path),
        "--ephys-root", str(ephys_root),
        "--histology-root", str(histology_root),
        "--regenerate-all",
        "--mouse", "mouse_A", "--session", "20240101_120000", "--probe", "imec0",
    ])
    with pytest.raises(SystemExit):
        _cli()
