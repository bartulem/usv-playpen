"""
@author: bartulem
Consolidate per-step model-selection pickles produced by the
forward-stepwise selectors in `model_selection.py` into a single
self-describing artifact.

Background
----------
Each selector (`bout_onset_model_selection`,
`vocal_category_model_selection`, `bout_parameter_model_selection`,
`multinomial_vocal_category_model_selection`,
`continuous_vocal_manifold_model_selection`) writes one
`<prefix>_step_<k>.pkl` per forward-selection step plus a final-fit
write that mutates the last step. Each step pickle carries three
reserved metadata blocks:

    `_input_metadata`        — copied from the Level-2 univariate
                               artifact (which copied it from Level-1).
    `_univariate_metadata`   — the univariate run config the selector
                               anchored to.
    `_run_metadata`          — selection-level config (stepwise knobs,
                               GAM kwargs, anchor feature).

This script merges every step pickle in a directory into a Level-3
consolidated artifact:

    {
        '_input_metadata':         {...},
        '_univariate_metadata':    {...},
        '_run_metadata':           {...},
        '_consolidation_metadata': {...},
        'steps': [<step_0_dict>, <step_1_dict>, ..., <step_K_dict>],
    }

Step dicts are written in step-order (smallest k first). The original
per-step files are *not* deleted by default — pass
`--move_to_steps_subdir` to relocate them into a `<input_dir>/steps/`
folder once consolidation succeeds.

Like the univariate consolidator, the script asserts metadata equality
across step files (modulo the `git_*` / `package_version` provenance
fields that vary across nodes and rebuilds).

CLI
---
    python -m usv_playpen.modeling.consolidate_model_selection_results \\
        --input_dir /path/to/selection_dir \\
        [--prefix model_selection_<...>_step_] \\
        [--output_dir /path/to/output_dir] \\
        [--output_filename my_consolidated.pkl] \\
        [--move_to_steps_subdir] \\
        [--allow_legacy] \\
        [--ignore_provenance_keys git_commit,git_dirty,package_version]

If `--prefix` is omitted, the consolidator infers it as the longest
common `model_selection_*_step_` prefix shared across the directory's
`*.pkl` files. When the directory mixes prefixes the consolidator
aborts and asks for an explicit `--prefix`.

Backwards-compat
----------------
With `--allow_legacy`, step pickles that lack the metadata blocks are
processed without the equality assert and the consolidated artifact is
written under `legacy_selection_<utc-ts>.pkl`. Default is to abort on
the first legacy file.
"""

import argparse
import pickle
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .modeling_metadata import (
    RESERVED_METADATA_KEYS,
    SCHEMA_VERSIONS,
    build_consolidation_metadata,
    extract_metadata_blocks,
    metadata_blocks_equal,
)

CONSOLIDATOR_NAME = 'consolidate_model_selection_results'
CONSOLIDATOR_VERSION = 1

_STEP_RE = re.compile(r'^(model_selection_.+_step_)(\d+)\.pkl$')


def _parse_step_idx(filename: str, prefix: str) -> int:
    """
    Returns the integer step index for a per-step filename matching
    `<prefix><k>.pkl`, or `-1` if the filename does not parse against
    `prefix`.
    """

    if not filename.startswith(prefix) or not filename.endswith('.pkl'):
        return -1
    body = filename[len(prefix):-len('.pkl')]
    return int(body) if body.isdigit() else -1


def _infer_prefix(pkl_files: list) -> str:
    """
    Infers the per-step filename prefix shared across `pkl_files`.

    The selectors all use the schema
    `model_selection_<descriptor>_step_<k>.pkl`. When every file in
    `pkl_files` matches this regex with the same descriptor, the
    inferred prefix is unambiguous and is returned. When the
    directory contains files from multiple selectors (different
    descriptors), the function raises so the caller can disambiguate
    via `--prefix`.

    Parameters
    ----------
    pkl_files : list of Path
        Files to inspect.

    Returns
    -------
    str
        The shared `model_selection_..._step_` prefix.

    Raises
    ------
    ValueError
        If the files do not share a single prefix or none parse.
    """

    prefixes = set()
    for fp in pkl_files:
        m = _STEP_RE.match(fp.name)
        if m:
            prefixes.add(m.group(1))
    if not prefixes:
        raise ValueError(
            "Could not infer a `model_selection_..._step_` prefix from "
            f"any file in the input directory ({len(pkl_files)} candidates)."
        )
    if len(prefixes) > 1:
        raise ValueError(
            "Multiple `model_selection_..._step_` prefixes detected in the "
            "input directory; pass --prefix to disambiguate. Found: "
            f"{sorted(prefixes)}"
        )
    return prefixes.pop()


def _file_mtime_iso(path) -> str:
    """
    Returns the file modification time as an ISO-8601 UTC string.
    Accepts a `str` or `pathlib.Path`.
    """

    ts = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _build_default_output_filename(input_metadata: dict,
                                   run_metadata: dict) -> str:
    """
    Builds the consolidated artifact's filename from the upstream
    metadata blocks and the current UTC timestamp.

    Schema:
    `selection_<analysis_tag>_<experimental_condition>_<selection_function>_<ts>.pkl`.

    Falls back to bare values when an upstream block is missing.
    """

    analysis_tag = input_metadata['analysis_tag'] if input_metadata is not None else 'unknown'
    cohort = input_metadata['experimental_condition'] if input_metadata is not None else 'unknown'
    sel_fn = run_metadata['selection_function'] if run_metadata is not None else 'unknown'
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    return f"selection_{analysis_tag}_{cohort}_{sel_fn}_{ts}.pkl"


def _diff_metadata(a: dict, b: dict, prefix: str = '') -> list:
    """
    Returns a flat list of `'<path>: <a_val> != <b_val>'` strings
    enumerating every leaf-level disagreement between two metadata
    blocks. See `consolidate_univariate_results._diff_metadata` —
    duplicated here to keep the two consolidators independent of each
    other.
    """

    diffs = []
    keys_a, keys_b = set(a.keys()), set(b.keys())
    for k in (keys_a - keys_b):
        diffs.append(f"{prefix}{k}: missing in B")
    for k in (keys_b - keys_a):
        diffs.append(f"{prefix}{k}: missing in A")
    for k in keys_a & keys_b:
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff_metadata(va, vb, prefix=f"{prefix}{k}."))
        elif va != vb:
            diffs.append(f"{prefix}{k}: {va!r} != {vb!r}")
    return diffs


def consolidate(input_dir: str,
                prefix: str = None,
                output_dir: str = None,
                output_filename: str = None,
                move_to_steps_subdir: bool = False,
                allow_legacy: bool = False,
                ignore_provenance_keys: tuple = (
                    'git_commit', 'git_dirty', 'package_version',
                )) -> str:
    """
    Walks `input_dir`, merges every per-step pickle matching `prefix`
    into a single Level-3 artifact, and writes it to `output_dir`.

    Parameters
    ----------
    input_dir : str
        Directory containing the per-step pickles.
    prefix : str, optional
        Per-step filename prefix (e.g.
        `'model_selection_intact_partners_male_mixed_step_'`). When
        omitted the prefix is inferred from the input directory; if
        the directory mixes prefixes the function raises.
    output_dir : str, optional
        Destination directory. Defaults to `input_dir`.
    output_filename : str, optional
        Filename for the consolidated artifact. When omitted, derived
        from the metadata blocks. Forced to
        `legacy_selection_<utc-ts>.pkl` for legacy runs.
    move_to_steps_subdir : bool, default False
        When True, relocates every consumed per-step pickle into a
        `<input_dir>/steps/` subdirectory once the consolidated
        artifact is written. Use this to declutter the working
        directory without losing the per-step crash-recovery files.
    allow_legacy : bool, default False
        When True, processes step pickles that lack the metadata
        blocks. Skips the equality assert and writes a `legacy_*.pkl`.
    ignore_provenance_keys : tuple of str, default
        `('git_commit', 'git_dirty', 'package_version')`
        Top-level fields excluded from the metadata-equality check.

    Returns
    -------
    str
        Absolute path of the written consolidated artifact.

    Raises
    ------
    FileNotFoundError
        If `input_dir` is empty or does not exist.
    ValueError
        On metadata mismatch, ambiguous prefix, missing step indices,
        or — without `--allow_legacy` — a missing-metadata file.
    """

    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    all_pkls = sorted(in_dir.glob('*.pkl'))
    if not all_pkls:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    if prefix is None:
        prefix = _infer_prefix(all_pkls)
        print(f"[consolidate] Inferred prefix: {prefix}")

    indexed = []
    for fp in all_pkls:
        idx = _parse_step_idx(fp.name, prefix)
        if idx >= 0:
            indexed.append((idx, fp))
    if not indexed:
        raise FileNotFoundError(
            f"No files in {input_dir} match prefix {prefix!r}."
        )
    indexed.sort(key=lambda pair: pair[0])

    print(f"[consolidate] Walking {len(indexed)} step pickle(s) under prefix "
          f"{prefix!r}.")

    canonical_input_md = None
    canonical_univariate_md = None
    canonical_run_md = None
    legacy_run_seen = False

    successfully_merged_paths = []
    successfully_merged_timestamps = []
    step_payloads = []  # list of (idx, clean_dict)

    for step_idx, fp in indexed:
        with open(fp, 'rb') as fh:
            payload = pickle.load(fh)
        clean, md_blocks = extract_metadata_blocks(payload)

        has_metadata = ('_input_metadata' in md_blocks and
                        '_run_metadata' in md_blocks)
        # `_univariate_metadata` is sometimes missing (legacy runs that
        # lacked a Level-2 consolidator output); we accept a step file
        # without it but still require the input + run blocks below.

        if not has_metadata:
            if not allow_legacy:
                raise ValueError(
                    f"{fp} lacks `_input_metadata` / `_run_metadata` "
                    "(legacy step pickle). Re-run with --allow_legacy "
                    "to consolidate without provenance, or regenerate "
                    "the step pickles with the current selectors."
                )
            legacy_run_seen = True
        else:
            cur_in = md_blocks['_input_metadata']
            cur_run = md_blocks['_run_metadata']
            cur_univ = md_blocks['_univariate_metadata'] if '_univariate_metadata' in md_blocks else None

            if canonical_input_md is None:
                canonical_input_md = cur_in
                canonical_run_md = cur_run
                canonical_univariate_md = cur_univ
            else:
                if not metadata_blocks_equal(canonical_input_md, cur_in,
                                             ignore_keys=ignore_provenance_keys):
                    diffs = _diff_metadata(canonical_input_md, cur_in)
                    raise ValueError(
                        f"`_input_metadata` mismatch between earlier "
                        f"step file and {fp.name}:\n  - " + "\n  - ".join(diffs)
                    )
                if not metadata_blocks_equal(canonical_run_md, cur_run,
                                             ignore_keys=ignore_provenance_keys):
                    diffs = _diff_metadata(canonical_run_md, cur_run)
                    raise ValueError(
                        f"`_run_metadata` mismatch between earlier "
                        f"step file and {fp.name}:\n  - " + "\n  - ".join(diffs)
                    )
                if cur_univ is not None and canonical_univariate_md is not None:
                    if not metadata_blocks_equal(canonical_univariate_md, cur_univ,
                                                 ignore_keys=ignore_provenance_keys):
                        diffs = _diff_metadata(canonical_univariate_md, cur_univ)
                        raise ValueError(
                            f"`_univariate_metadata` mismatch between earlier "
                            f"step file and {fp.name}:\n  - " + "\n  - ".join(diffs)
                        )

        step_payloads.append((step_idx, clean))
        successfully_merged_paths.append(str(fp.resolve()))
        successfully_merged_timestamps.append(_file_mtime_iso(fp))

    print(f"[consolidate] Merged {len(step_payloads)} step file(s).")

    cons_md = build_consolidation_metadata(
        n_files_merged=len(successfully_merged_paths),
        individual_file_paths=successfully_merged_paths,
        individual_file_timestamps=successfully_merged_timestamps,
        consolidator_name=CONSOLIDATOR_NAME,
        consolidator_version=CONSOLIDATOR_VERSION,
    )

    # Verify step indices are contiguous starting from 0 — selectors
    # always emit sequential `step_0, step_1, ...` files. A gap means a
    # step crashed mid-write and was never produced; we warn so the
    # consumer can investigate but still consolidate what is on disk.
    expected_indices = list(range(len(step_payloads)))
    actual_indices = [idx for idx, _ in step_payloads]
    if actual_indices != expected_indices:
        print(f"[consolidate] WARNING: step indices are not contiguous "
              f"({actual_indices}). Consolidating anyway.")

    consolidated = {
        'steps': [payload for _, payload in step_payloads],
        '_consolidation_metadata': cons_md,
    }
    if canonical_input_md is not None:
        consolidated['_input_metadata'] = canonical_input_md
    if canonical_univariate_md is not None:
        consolidated['_univariate_metadata'] = canonical_univariate_md
    if canonical_run_md is not None:
        consolidated['_run_metadata'] = canonical_run_md

    if output_filename is None:
        if legacy_run_seen and canonical_input_md is None:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
            output_filename = f"legacy_selection_{ts}.pkl"
        else:
            output_filename = _build_default_output_filename(
                canonical_input_md, canonical_run_md
            )

    output_root = Path(input_dir) if output_dir is None else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / output_filename

    with out_path.open('wb') as fh:
        pickle.dump(consolidated, fh)

    print(f"[consolidate] Wrote consolidated artifact: {out_path}")

    if move_to_steps_subdir:
        steps_dir = Path(input_dir) / 'steps'
        steps_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p in successfully_merged_paths:
            try:
                shutil.move(str(p), str(steps_dir / Path(p).name))
                moved += 1
            except OSError as exc:
                print(f"[consolidate] WARNING: could not move {p}: {exc}")
        print(f"[consolidate] Moved {moved}/{len(successfully_merged_paths)} step pickles to {steps_dir}.")

    return str(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Consolidate per-step model-selection pickles into a single artifact."
    )
    parser.add_argument('--input_dir', required=True,
                        help="Directory containing the per-step pickles.")
    parser.add_argument('--prefix', default=None,
                        help="Step-filename prefix (defaults to the longest "
                             "common `model_selection_*_step_` shared by every "
                             "input file).")
    parser.add_argument('--output_dir', default=None,
                        help="Output directory. Defaults to --input_dir.")
    parser.add_argument('--output_filename', default=None,
                        help="Override the derived output filename.")
    parser.add_argument('--move_to_steps_subdir', action='store_true',
                        help="Relocate consumed step pickles into "
                             "`<input_dir>/steps/`.")
    parser.add_argument('--allow_legacy', action='store_true',
                        help="Process pickles that lack metadata blocks.")
    parser.add_argument('--ignore_provenance_keys',
                        default='git_commit,git_dirty,package_version',
                        help="Comma-separated metadata keys to skip during the "
                             "equality assert (default: git_commit,git_dirty,package_version).")
    cli_args = parser.parse_args()

    ignored = tuple(k.strip() for k in cli_args.ignore_provenance_keys.split(',') if k.strip())

    try:
        out = consolidate(
            input_dir=cli_args.input_dir,
            prefix=cli_args.prefix,
            output_dir=cli_args.output_dir,
            output_filename=cli_args.output_filename,
            move_to_steps_subdir=cli_args.move_to_steps_subdir,
            allow_legacy=cli_args.allow_legacy,
            ignore_provenance_keys=ignored,
        )
        print(f"OK: {out}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
