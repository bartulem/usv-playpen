"""
@author: bartulem
Consolidate per-feature univariate-modeling pickles produced by the
`main_univariate_dispatcher.py` SLURM job array into a single
self-describing artifact.

Background
----------
The dispatcher writes one `univariate_<tag>_<idx>_<feat>_<ts>.pkl` per
feature, each containing the per-feature result dict plus two reserved
metadata blocks:

    `_input_metadata` — copied verbatim from the Level-1 input pickle
    (cohort, kept feature zoo, temporal frame, …).

    `_run_metadata`  — built by the dispatcher (engine, regularization
    knobs, inner-CV grid, outer-loop layout, …).

This script merges those per-feature pickles into a Level-2 consolidated
artifact whose layout is:

    {
        '<feature_name_1>': {...result dict...},
        ...
        '<feature_name_N>': {...result dict...},
        '_input_metadata': {...},
        '_run_metadata':   {...},
        '_consolidation_metadata': {...},
    }

Two safety properties are enforced:

1.  **Metadata equality**: every per-feature pickle's `_input_metadata`
    and `_run_metadata` blocks must be structurally equal (modulo
    explicitly ignored fields like per-file `git_dirty` / package
    version). Stray files from a different run cause the consolidator
    to abort with a precise diff.
2.  **Reserved-key isolation**: the per-feature dict must contain
    exactly one feature key (the dispatcher writes it that way). A
    file that contains zero or more than one feature key triggers an
    abort.

CLI
---
    python -m usv_playpen.modeling.consolidate_univariate_results \\
        --input_dir /path/to/per_feature_dir \\
        [--output_dir /path/to/output_dir] \\
        [--output_filename my_consolidated.pkl] \\
        [--delete_individuals_after] \\
        [--allow_legacy] \\
        [--ignore_provenance_keys git_commit,git_dirty,package_version]

Defaults:
    output_dir       = same as input_dir
    output_filename  = derived from `_input_metadata` →
                       `univariate_<analysis_tag>_<condition>_<ts>.pkl`
                       (the `<ts>` is the consolidation moment, UTC).
    ignored keys     = `('git_commit', 'git_dirty', 'package_version')`
                       — these vary across the SLURM array (different
                       nodes can have slightly different working trees
                       or installed packages) but the resolved settings
                       hash and the substantive knob values still must
                       agree.

Backwards-compat
----------------
Per-feature pickles produced before the metadata schema rolled out have
no `_input_metadata` / `_run_metadata` blocks. With `--allow_legacy`,
the consolidator processes them too: it warns once, skips the equality
asserts, and writes the consolidated artifact under a `legacy_<ts>.pkl`
filename without the cohort tag. Without `--allow_legacy`, the
consolidator aborts on the first legacy file.
"""

import argparse
import pickle
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

CONSOLIDATOR_NAME = 'consolidate_univariate_results'
CONSOLIDATOR_VERSION = 1


def _parse_feature_idx(filename: str) -> int:
    """
    Extracts the feature index from a per-feature filename.

    The dispatcher's filename schema is
    `univariate_<analysis_tag>_<idx:04d>_<safe_feat>_<ts>.pkl`. The
    legacy schema was `univariate_<idx>_<safe_feat>_<ts>.pkl`. Both
    use the same split-on-underscore strategy: the first integer
    token after the tag (or after `'univariate'`) is the index.

    Parameters
    ----------
    filename : str
        Bare filename without directory components.

    Returns
    -------
    int
        Feature index, or -1 if the filename does not parse.
    """

    base = Path(filename).stem
    parts = base.split('_')
    for tok in parts[1:]:
        if tok.isdigit():
            return int(tok)
    return -1


def _file_mtime_iso(path) -> str:
    """
    Returns the file modification time as an ISO-8601 UTC string.
    Accepts a `str` or `pathlib.Path`.
    """

    ts = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _build_default_output_filename(input_metadata: dict) -> str:
    """
    Builds the consolidated artifact's filename from the upstream
    `_input_metadata` block and the current UTC timestamp.

    Schema: `univariate_<analysis_tag>_<experimental_condition>_<ts>.pkl`.
    """

    tag = input_metadata['analysis_tag']
    cohort = input_metadata['experimental_condition']
    ts = (datetime.now(timezone.utc)
          .strftime('%Y%m%d_%H%M%SZ'))
    return f"univariate_{tag}_{cohort}_{ts}.pkl"


def _diff_metadata(a: dict, b: dict, prefix: str = '') -> list:
    """
    Returns a flat list of `'<path>: <a_val> != <b_val>'` strings
    enumerating every leaf-level disagreement between two metadata
    blocks. Used by the equality check to give an actionable error
    message rather than a generic "blocks differ" abort.
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
                output_dir: str = None,
                output_filename: str = None,
                delete_individuals_after: bool = False,
                allow_legacy: bool = False,
                ignore_provenance_keys: tuple = (
                    'git_commit', 'git_dirty', 'package_version',
                )) -> str:
    """
    Walks `input_dir`, merges every `*.pkl` it contains into a single
    Level-2 artifact, and writes it to `output_dir`. Returns the
    consolidated artifact's absolute path.

    The function is the library entry point; the module's `__main__`
    block exposes the same logic via argparse.

    Parameters
    ----------
    input_dir : str
        Directory containing the per-feature pickles. Sub-directories
        are not walked.
    output_dir : str, optional
        Destination directory. Defaults to `input_dir`.
    output_filename : str, optional
        Filename for the consolidated artifact. When omitted, the
        filename is built from `_input_metadata['analysis_tag']` and
        `_input_metadata['experimental_condition']`. For legacy runs
        (no metadata) the filename is forced to
        `legacy_<utc-ts>.pkl`.
    delete_individuals_after : bool, default False
        When True, removes every per-feature pickle that was
        successfully merged. The consolidator never deletes a file it
        did not merge.
    allow_legacy : bool, default False
        When True, processes per-feature pickles that lack the
        `_input_metadata` / `_run_metadata` blocks. Skips the equality
        assert and writes the consolidated artifact under
        `legacy_<utc-ts>.pkl`. Default is to abort on the first such
        file.
    ignore_provenance_keys : tuple of str, default
        `('git_commit', 'git_dirty', 'package_version')`
        Top-level fields excluded from the metadata-equality check.
        Useful when the SLURM array runs across nodes with slightly
        divergent environments — those keys vary, but the substantive
        run configuration (settings_sha256 + numerical knobs) still
        agrees.

    Returns
    -------
    str
        Absolute path of the written consolidated artifact.

    Raises
    ------
    FileNotFoundError
        If `input_dir` is empty or does not exist.
    ValueError
        If any per-feature file disagrees on a non-ignored metadata
        key, contains zero or more than one feature, or — when
        `allow_legacy` is False — lacks the metadata blocks.
    """

    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pkl_files = sorted(in_dir.glob('*.pkl'),
                       key=lambda p: (_parse_feature_idx(p.name), p.name))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    print(f"[consolidate] Walking {len(pkl_files)} per-feature pickle(s) in {input_dir}")

    consolidated = {}
    canonical_input_md = None
    canonical_run_md = None
    legacy_run_seen = False

    successfully_merged_paths = []
    successfully_merged_timestamps = []

    for idx, fp in enumerate(pkl_files):
        with open(fp, 'rb') as fh:
            payload = pickle.load(fh)
        feat_dict, md_blocks = extract_metadata_blocks(payload)

        # The dispatcher writes exactly one feature key per file. A file
        # with zero or many feature keys is malformed (or was hand-edited);
        # we abort loudly rather than silently merge.
        feat_keys = list(feat_dict.keys())
        if len(feat_keys) != 1:
            raise ValueError(
                f"Expected exactly one feature key in {fp}, got "
                f"{len(feat_keys)}: {feat_keys}"
            )
        feat_name = feat_keys[0]

        has_metadata = ('_input_metadata' in md_blocks and
                        '_run_metadata' in md_blocks)

        if not has_metadata:
            if not allow_legacy:
                raise ValueError(
                    f"{fp} lacks `_input_metadata` / `_run_metadata` "
                    "(legacy artifact). Re-run with --allow_legacy to "
                    "consolidate without provenance, or regenerate the "
                    "per-feature pickles with the current dispatcher."
                )
            legacy_run_seen = True
        else:
            cur_in = md_blocks['_input_metadata']
            cur_run = md_blocks['_run_metadata']

            if canonical_input_md is None:
                canonical_input_md = cur_in
                canonical_run_md = cur_run
            else:
                # Equality is structural. Top-level "provenance" keys
                # listed in `ignore_provenance_keys` are excluded
                # because they vary across SLURM nodes / package
                # rebuilds without changing the substantive run
                # configuration.
                if not metadata_blocks_equal(canonical_input_md, cur_in,
                                             ignore_keys=ignore_provenance_keys):
                    diffs = _diff_metadata(canonical_input_md, cur_in)
                    raise ValueError(
                        f"`_input_metadata` mismatch between {pkl_files[0].name} "
                        f"and {fp.name}:\n  - " + "\n  - ".join(diffs)
                    )
                if not metadata_blocks_equal(canonical_run_md, cur_run,
                                             ignore_keys=ignore_provenance_keys):
                    diffs = _diff_metadata(canonical_run_md, cur_run)
                    raise ValueError(
                        f"`_run_metadata` mismatch between {pkl_files[0].name} "
                        f"and {fp.name}:\n  - " + "\n  - ".join(diffs)
                    )

        if feat_name in consolidated:
            raise ValueError(
                f"Duplicate feature {feat_name!r} encountered in {fp.name}; "
                f"already merged from a prior file."
            )

        consolidated[feat_name] = feat_dict[feat_name]
        successfully_merged_paths.append(str(fp.resolve()))
        successfully_merged_timestamps.append(_file_mtime_iso(fp))

    print(f"[consolidate] Merged {len(consolidated)} feature(s).")

    cons_md = build_consolidation_metadata(
        n_files_merged=len(successfully_merged_paths),
        individual_file_paths=successfully_merged_paths,
        individual_file_timestamps=successfully_merged_timestamps,
        consolidator_name=CONSOLIDATOR_NAME,
        consolidator_version=CONSOLIDATOR_VERSION,
    )

    consolidated['_consolidation_metadata'] = cons_md
    if canonical_input_md is not None:
        consolidated['_input_metadata'] = canonical_input_md
    if canonical_run_md is not None:
        consolidated['_run_metadata'] = canonical_run_md

    # Choose output filename
    if output_filename is None:
        if legacy_run_seen and canonical_input_md is None:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
            output_filename = f"legacy_univariate_{ts}.pkl"
        else:
            output_filename = _build_default_output_filename(canonical_input_md)

    output_root = Path(input_dir) if output_dir is None else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / output_filename

    with out_path.open('wb') as fh:
        pickle.dump(consolidated, fh)

    print(f"[consolidate] Wrote consolidated artifact: {out_path}")

    if delete_individuals_after:
        deleted = 0
        for p in successfully_merged_paths:
            try:
                Path(p).unlink()
                deleted += 1
            except OSError as exc:
                print(f"[consolidate] WARNING: could not delete {p}: {exc}")
        print(f"[consolidate] Removed {deleted}/{len(successfully_merged_paths)} per-feature pickles.")

    return str(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Consolidate per-feature univariate pickles into a single artifact."
    )
    parser.add_argument('--input_dir', required=True,
                        help="Directory containing the per-feature pickles.")
    parser.add_argument('--output_dir', default=None,
                        help="Output directory. Defaults to --input_dir.")
    parser.add_argument('--output_filename', default=None,
                        help="Override the derived output filename.")
    parser.add_argument('--delete_individuals_after', action='store_true',
                        help="Remove successfully-merged per-feature pickles.")
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
            output_dir=cli_args.output_dir,
            output_filename=cli_args.output_filename,
            delete_individuals_after=cli_args.delete_individuals_after,
            allow_legacy=cli_args.allow_legacy,
            ignore_provenance_keys=ignored,
        )
        print(f"OK: {out}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
