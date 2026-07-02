"""
@author: bartulem
Regenerate ``neuropixels_sites_to_anatomy_converter.json`` so its per-
``(mouse_id, session_id, probe)`` channel-index ranges live in
**Kilosort-row** space rather than the raw-meta channel ids the
historical generator stored.

Why
---
The codebase consistently identifies channels by **Kilosort row index**
— unit file names look like ``cl0017_ch042_good.npy`` where the ``042``
is a KS row, the catalog stores ``closest_ch`` as a KS row, and
``channel_positions.npy`` / ``channel_shanks.npy`` are KS-row arrays.
The on-disk anatomy converter, however, was generated with ranges
keyed by the raw-meta channel ids (= ``imro_rows[k+1][0]`` =
``channel_locations.json``'s ``channel_{i}`` numeric suffix), so
``converter[region]`` membership checks against a KS row returned the
anatomy of a different physical electrode for ~99% of channels. See the
channel-numbering conventions section of ``docs/Neuropixels.rst`` for the
full landscape.

This module regenerates the converter so that, for every probe-day,
``region: [[lo, hi], ...]`` ranges enumerate **contiguous KS-row id
runs** that fall in the same brain region. Internally the function
joins the IBL anatomy to KS rows by physical ``(lateral, axial)``
position (``channel_positions.npy[i]`` ↔ ``channel_locations.json``
``lateral``/``axial`` fields agree byte-for-byte for every electrode
on every audited probe-day), so no ``channel_map.npy`` lookup is
required.

The fix is purely on-disk: after regeneration, every downstream
consumer that already treated converter ranges as KS-row index space
(``make_behavioral_videos.find_region_by_channel`` and friends) just
starts returning the right region. No consumer-side code changes
needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import warnings
from collections import defaultdict
from typing import Any

import numpy as np

from ..os_utils import resolve_data_root


# Data-location defaults are configured in `analyses_settings.json` under
# `data_roots` and resolved to the host OS via `configure_path` (see
# `resolve_data_root`), so they are user-editable and OS-portable rather than
# hard-coded; the CLI flags below still override them.
_DEFAULT_CONVERTER_PATH = resolve_data_root("converter_path")
_DEFAULT_EPHYS_ROOT = resolve_data_root("ephys_root")
_DEFAULT_HISTOLOGY_ROOT = resolve_data_root("histology_root")

# Constant across all sessions in this dataset (user-confirmed). Update
# this if a future probe is wired differently and the dict needs to be
# per-session instead.
_PROBE_TO_HEMISPHERE: dict[str, str] = {"imec0": "R", "imec1": "L"}


def _load_ibl_position_to_region(
        histology_root: pathlib.Path,
        mouse_id: str,
        rec_date: int,
        hemisphere: str,
) -> dict[tuple[int, int], str]:
    """
    Description
    -----------
    Build a ``(lateral, axial) -> brain_region`` lookup from the IBL
    alignment JSON for one ``(mouse, recording_date, hemisphere)``.

    The IBL JSON publishes its own ``lateral``/``axial`` fields per
    entry, and those values agree byte-for-byte with the absolute
    ``(lateral, axial)`` SpikeInterface writes to
    ``channel_positions.npy``. So joining by physical position is
    canonical: it sidesteps both IBL's raw-meta channel keys and the
    ``channel_map.npy`` permutation Kilosort applied.

    Parameters
    ----------
    histology_root (pathlib.Path)
        Root directory containing ``<mouse_id>/<rec_date>/ibl_{LH,RH}/
        channel_locations.json``.
    mouse_id (str)
        Catalog ``mouse_id`` string.
    rec_date (int)
        Recording date as ``YYYYMMDD`` integer.
    hemisphere (str)
        ``'L'`` or ``'R'``.

    Returns
    -------
    pos_to_region (dict[tuple[int, int], str])
        Mapping from integer ``(lateral_um, axial_um)`` to the IBL
        brain region label at that electrode.
    """

    hemi_dir = "ibl_RH" if hemisphere.upper().startswith("R") else "ibl_LH"
    path = histology_root / mouse_id / str(rec_date) / hemi_dir / "channel_locations.json"
    with path.open() as fh:
        ibl = json.load(fh)
    pos_to_region: dict[tuple[int, int], str] = {}
    for key, entry in ibl.items():
        if not key.startswith("channel_"):
            continue
        # Round (not truncate) before the integer (lateral, axial) join:
        # int() floors toward zero, so a coordinate stored as 31.9999 on one
        # side and 32.0 on the other would key to 31 vs 32 and never join.
        # Both sides of the join must round identically.
        pos_to_region[(int(round(entry["lateral"])), int(round(entry["axial"])))] = (
            entry["brain_region"]
        )
    return pos_to_region


def _runs_to_ranges(
        per_row_region: list[str],
) -> dict[str, list[list[int]]]:
    """
    Description
    -----------
    Compress a per-channel sequence of region labels into the
    ``{region: [[lo, hi], ...]}`` half-open-range layout the existing
    converter uses. Runs are maximal — ``[lo, hi)`` means KS rows
    ``lo`` inclusive through ``hi`` exclusive all share the same
    region.

    Parameters
    ----------
    per_row_region (list[str])
        Length-``n_channels`` list of brain region labels indexed by
        Kilosort row.

    Returns
    -------
    ranges (dict[str, list[list[int]]])
        ``{region_name: [[lo, hi], ...]}``.
    """

    ranges: dict[str, list[list[int]]] = defaultdict(list)
    if not per_row_region:
        return dict(ranges)
    run_start = 0
    for i in range(1, len(per_row_region)):
        if per_row_region[i] != per_row_region[run_start]:
            ranges[per_row_region[run_start]].append([run_start, i])
            run_start = i
    ranges[per_row_region[run_start]].append([run_start, len(per_row_region)])
    return dict(ranges)


def _build_ks_keyed_block(
        ks_dir: pathlib.Path,
        pos_to_region: dict[tuple[int, int], str],
) -> dict[str, list[list[int]]]:
    """
    Description
    -----------
    Build the per-probe ``{region: [[lo, hi], ...]}`` block for one
    Kilosort directory. Joins the IBL position map to KS rows by
    physical ``(lateral, axial)``, then compresses contiguous runs.

    Parameters
    ----------
    ks_dir (pathlib.Path)
        Path to the per-probe ``kilosort4/`` directory containing
        ``channel_positions.npy``.
    pos_to_region (dict[tuple[int, int], str])
        IBL position-to-region map for this (mouse, date, hemisphere).

    Returns
    -------
    ranges (dict[str, list[list[int]]])
        Per-region KS-row ranges for this probe.
    """

    channel_positions = np.load(ks_dir / "channel_positions.npy")
    per_row_region: list[str] = []
    for ks_row in range(channel_positions.shape[0]):
        # Round (not truncate) to match the IBL side of the join in
        # `_load_ibl_position_to_region`; int() floors toward zero and would
        # desync the two integer keys at sub-unit coordinate differences.
        pos = (
            int(round(channel_positions[ks_row, 0])),
            int(round(channel_positions[ks_row, 1])),
        )
        # Positions absent from the IBL map (e.g. reference/disconnected
        # sites) become a literal "unknown" region; `_runs_to_ranges`
        # emits it as its own first-class key, so downstream
        # `find_region_by_channel` can return "unknown" as a brain region.
        region = pos_to_region[pos] if pos in pos_to_region else "unknown"
        per_row_region.append(region)
    return _runs_to_ranges(per_row_region)


def _write_converter(
        converter_path: pathlib.Path,
        converter: dict[str, Any],
) -> None:
    """
    Description
    -----------
    Write ``converter`` to ``converter_path`` as JSON, keeping each region's
    ``[[lo, hi], ...]`` range list on a single line.

    Python's default ``json.dump(indent=4)`` would split every ``[lo, hi]``
    AND the outer list across many lines — a wall of one-int rows that is
    painful to read and produces enormous diffs. This renders with stock
    ``indent=4`` first (to get the nested structure) then collapses in two
    passes: (1) every ``[\\n  lo,\\n  hi\\n]`` -> ``[lo, hi]``; (2) every outer
    list whose items are now all ``[int, int]`` -> ``[[lo, hi], ...]`` on one
    line. Every write path routes through here, so the on-disk row formatting
    is never accidentally reflowed.

    Parameters
    ----------
    converter_path (pathlib.Path)
        Destination path for the converter JSON.
    converter (dict[str, Any])
        The nested ``{mouse: {session: {probe: {region: [[lo, hi], ...]}}}}``
        converter dict.

    Returns
    -------
    None
    """

    rendered = json.dumps(converter, indent=4)
    rendered = re.sub(
        r"\[\s+(-?\d+),\s+(-?\d+)\s+\]",
        r"[\1, \2]",
        rendered,
    )
    rendered = re.sub(
        r"\[\s*((?:\[-?\d+, -?\d+\](?:,\s*)?)+)\s*\]",
        lambda m: "[" + ", ".join(
            re.findall(r"\[-?\d+, -?\d+\]", m.group(0))
        ) + "]",
        rendered,
    )
    # Create the parent directory (e.g. a fresh experimenter's EPHYS/) if it does
    # not exist yet, so a first-time write never fails.
    converter_path.parent.mkdir(parents=True, exist_ok=True)
    converter_path.write_text(rendered)


def _regenerate_probe_block(
        mouse_id: str,
        session_id: str,
        rec_date: int,
        probe: str,
        ephys_root: pathlib.Path,
        histology_root: pathlib.Path,
        probe_to_hemisphere: dict[str, str],
) -> tuple[dict[str, list[list[int]]] | None, str | None]:
    """
    Description
    -----------
    Build the Kilosort-row-keyed ``{region: [[lo, hi], ...]}`` block for one
    ``(mouse_id, session_id, probe)``, or return a human-readable skip reason.
    Shared by the batch :func:`regenerate_anatomy_converter` and the
    incremental :func:`add_session_to_anatomy_converter` so both apply the same
    checks and the same all-``"unknown"`` guard.

    A probe is skipped (block ``None``, reason set) when its
    ``channel_positions.npy`` is missing, its ``probe`` has no hemisphere
    mapping, its IBL ``channel_locations.json`` is absent, or every Kilosort row
    joins to ``"unknown"`` (a failed position join — wrong-hemisphere file,
    units/coordinate mismatch, or a stale ``channel_positions.npy``); the last
    also emits a ``warnings.warn`` so a silent all-unknown overwrite cannot pass
    unnoticed.

    Parameters
    ----------
    mouse_id (str)
        Catalog mouse id (used for the skip-reason string).
    session_id (str)
        Session id (used for the skip-reason string).
    rec_date (int)
        Recording date as ``YYYYMMDD`` integer; locates the Kilosort and IBL
        outputs.
    probe (str)
        Probe id (``'imec0'`` / ``'imec1'``).
    ephys_root (pathlib.Path)
        Root containing ``<rec_date>_<probe>/kilosort4/``.
    histology_root (pathlib.Path)
        Root containing ``<mouse_id>/<rec_date>/ibl_{LH,RH}/``.
    probe_to_hemisphere (dict[str, str])
        Probe-to-hemisphere mapping.

    Returns
    -------
    (block, reason) (tuple[dict | None, str | None])
        ``(block, None)`` on success, or ``(None, reason)`` when skipped.
    """

    ks_dir = ephys_root / f"{rec_date}_{probe}" / "kilosort4"
    if not (ks_dir / "channel_positions.npy").exists():
        return None, f"{mouse_id}/{session_id}/{probe}: no channel_positions.npy"
    if probe not in probe_to_hemisphere:
        return None, f"{mouse_id}/{session_id}/{probe}: no hemisphere mapping"
    hemisphere = probe_to_hemisphere[probe]
    try:
        pos_to_region = _load_ibl_position_to_region(
            histology_root, str(mouse_id), int(rec_date), hemisphere,
        )
    except FileNotFoundError:
        return None, f"{mouse_id}/{session_id}/{probe}: no IBL channel_locations.json"
    ks_block = _build_ks_keyed_block(ks_dir, pos_to_region)
    # A block whose ONLY region is "unknown" means every Kilosort channel failed
    # the IBL (lateral, axial) position join (e.g. a wrong-hemisphere file, a
    # units/coordinate mismatch, or a stale channel_positions.npy). Silently
    # overwriting a probe with an all-"unknown" block is the worst outcome
    # (find_region_by_channel then returns "unknown"/None for every cluster), so
    # warn loudly and skip, leaving the probe's existing converter entry intact.
    # A handful of legitimately-unknown sites (reference/disconnected channels)
    # still produce real regions and so are unaffected.
    if set(ks_block) == {"unknown"}:
        msg = (
            f"{mouse_id}/{session_id}/{probe}: every channel mapped to "
            f"'unknown' (IBL position join failed -- wrong-hemisphere file, "
            f"units/coordinate mismatch, or stale channel_positions.npy); "
            f"probe skipped, converter NOT overwritten for it."
        )
        warnings.warn(msg, stacklevel=2)
        return None, msg
    return ks_block, None


def regenerate_anatomy_converter(
        converter_path: str | pathlib.Path = _DEFAULT_CONVERTER_PATH,
        ephys_root: str | pathlib.Path = _DEFAULT_EPHYS_ROOT,
        histology_root: str | pathlib.Path = _DEFAULT_HISTOLOGY_ROOT,
        probe_to_hemisphere: dict[str, str] | None = None,
        dry_run: bool = False,
) -> dict[str, Any]:
    """
    Description
    -----------
    Rewrite ``converter_path`` in place so every per-probe block has
    Kilosort-row-indexed ``[[lo, hi], ...]`` ranges (instead of the
    raw-meta channel ids the historical generator stored). Iterates
    over every ``(mouse_id, session_id, probe)`` triple already in the
    converter; per-day same-mouse same-probe sessions share the same
    Kilosort directory and IBL histology output, so they yield
    identical per-probe blocks (they're regenerated independently here
    for simplicity but the result is the same). Triples that cannot be
    regenerated (missing Kilosort/IBL output or an all-``"unknown"`` join)
    keep their existing on-disk block and are reported in ``skipped_reasons``.

    Parameters
    ----------
    converter_path (str | pathlib.Path)
        Path to the converter JSON to rewrite.
    ephys_root (str | pathlib.Path)
        Root containing ``<YYYYMMDD>_imec<i>/kilosort4/`` per probe.
    histology_root (str | pathlib.Path)
        Root containing ``<mouse_id>/<rec_date>/ibl_{LH,RH}/
        channel_locations.json``.
    probe_to_hemisphere (dict[str, str] | None)
        Mapping from probe id (``'imec0'`` / ``'imec1'``) to hemisphere
        (``'L'`` / ``'R'``). Defaults to the module-level constant.
    dry_run (bool)
        If True, return the regenerated converter dict without writing
        it back to disk.

    Returns
    -------
    summary (dict)
        Keys ``n_triples_total``, ``n_triples_regenerated``,
        ``n_triples_skipped``, ``skipped_reasons``, ``output``.
        ``output`` is the regenerated dict when ``dry_run=True``,
        otherwise the path written to.
    """

    converter_path = pathlib.Path(converter_path)
    ephys_root = pathlib.Path(ephys_root)
    histology_root = pathlib.Path(histology_root)
    if probe_to_hemisphere is None:
        # Defensive copy of the shared module constant so any future
        # in-function mutation cannot corrupt the global for later calls.
        probe_to_hemisphere = dict(_PROBE_TO_HEMISPHERE)

    with converter_path.open() as fh:
        existing = json.load(fh)

    # Seed from a deep copy of the existing converter so that any triple we skip
    # below (missing Kilosort output, no IBL JSON, an all-"unknown" join) keeps
    # its previous on-disk block instead of being dropped when the file is
    # rewritten; only successfully-regenerated triples are overwritten.
    regenerated: dict[str, Any] = copy.deepcopy(existing)
    n_total = 0
    n_done = 0
    skipped: list[str] = []
    for mouse_id, mouse_entry in existing.items():
        for session_id, probes_entry in mouse_entry.items():
            try:
                rec_date = int(session_id[:8])
            except ValueError:
                n_total += len(probes_entry)
                for p in probes_entry:
                    skipped.append(f"{mouse_id}/{session_id}/{p}: bad session_id")
                continue
            for probe in probes_entry:
                n_total += 1
                ks_block, reason = _regenerate_probe_block(
                    mouse_id, session_id, rec_date, probe,
                    ephys_root, histology_root, probe_to_hemisphere,
                )
                if reason is not None:
                    skipped.append(reason)
                    continue
                regenerated[mouse_id][session_id][probe] = ks_block
                n_done += 1

    if not dry_run:
        _write_converter(converter_path, regenerated)

    return {
        "n_triples_total": n_total,
        "n_triples_regenerated": n_done,
        "n_triples_skipped": len(skipped),
        "skipped_reasons": skipped,
        "output": str(converter_path) if not dry_run else regenerated,
    }


def add_session_to_anatomy_converter(
        mouse_id: str,
        session_id: str,
        probe: str,
        *,
        force: bool = False,
        converter_path: str | pathlib.Path = _DEFAULT_CONVERTER_PATH,
        ephys_root: str | pathlib.Path = _DEFAULT_EPHYS_ROOT,
        histology_root: str | pathlib.Path = _DEFAULT_HISTOLOGY_ROOT,
        probe_to_hemisphere: dict[str, str] | None = None,
        dry_run: bool = False,
) -> dict[str, Any]:
    """
    Description
    -----------
    Add one ``(mouse_id, session_id, probe)``'s Kilosort-row-keyed region block
    to the converter JSON, MERGING into whatever is already on disk — every
    other mouse / session / probe is preserved byte-for-byte. Idempotent: if the
    triple is already present and ``force`` is False, it is left untouched and
    nothing is written. This is the per-session notebook entry point (histology
    pipeline step 6); the batch :func:`regenerate_anatomy_converter` instead
    rewrites every triple already in the file. If ``converter_path`` does not
    exist yet it is created with just this triple.

    Parameters
    ----------
    mouse_id (str)
        Catalog mouse id (the histology directory name).
    session_id (str)
        Session id; its first 8 characters must be the ``YYYYMMDD`` recording
        date.
    probe (str)
        Probe id (``'imec0'`` / ``'imec1'``).
    force (bool)
        If True, rebuild and overwrite the triple's block even when it is
        already present (e.g. after a re-alignment). Defaults to add-if-missing.
    converter_path (str | pathlib.Path)
        Path to the converter JSON to update.
    ephys_root (str | pathlib.Path)
        Root containing ``<rec_date>_<probe>/kilosort4/``.
    histology_root (str | pathlib.Path)
        Root containing ``<mouse_id>/<rec_date>/ibl_{LH,RH}/``.
    probe_to_hemisphere (dict[str, str] | None)
        Probe-to-hemisphere mapping. Defaults to the module-level constant.
    dry_run (bool)
        If True, compute the merged converter without writing it back; the dict
        is returned under ``output``.

    Returns
    -------
    summary (dict)
        Keys ``status`` (``'added'`` / ``'updated'`` / ``'already_present'`` /
        ``'skipped'``), ``reason`` (skip reason or ``None``), and ``output``
        (the path written, or the merged dict when ``dry_run=True``).
    """

    converter_path = pathlib.Path(converter_path)
    ephys_root = pathlib.Path(ephys_root)
    histology_root = pathlib.Path(histology_root)
    if probe_to_hemisphere is None:
        # Defensive copy of the shared module constant so any future in-function
        # mutation cannot corrupt the global for later calls.
        probe_to_hemisphere = dict(_PROBE_TO_HEMISPHERE)

    # Load the current converter (start empty if the file does not exist yet) so
    # the merge preserves every entry we are not touching.
    if converter_path.exists():
        with converter_path.open() as fh:
            converter = json.load(fh)
    else:
        converter = {}

    already_present = (
        mouse_id in converter
        and session_id in converter[mouse_id]
        and probe in converter[mouse_id][session_id]
    )
    if already_present and not force:
        return {"status": "already_present", "reason": None, "output": str(converter_path)}

    try:
        rec_date = int(str(session_id)[:8])
    except ValueError:
        return {
            "status": "skipped",
            "reason": f"{mouse_id}/{session_id}/{probe}: bad session_id",
            "output": str(converter_path),
        }

    ks_block, reason = _regenerate_probe_block(
        mouse_id, session_id, rec_date, probe,
        ephys_root, histology_root, probe_to_hemisphere,
    )
    if reason is not None:
        return {"status": "skipped", "reason": reason, "output": str(converter_path)}

    converter.setdefault(mouse_id, {}).setdefault(session_id, {})[probe] = ks_block
    if not dry_run:
        _write_converter(converter_path, converter)
    return {
        "status": "updated" if already_present else "added",
        "reason": None,
        "output": converter if dry_run else str(converter_path),
    }


def _cli() -> None:
    """
    Description
    -----------
    Argparse entry point. Runs nothing by default (prints usage), so an
    accidental bare invocation never rewrites the shared converter. Select an
    action explicitly:

    * ``--regenerate-all`` -- bulk-regenerate every triple already in the
      converter (mirrors :func:`regenerate_anatomy_converter`); for one-time
      maintenance after a change to the region-joining logic.
    * ``--mouse`` / ``--session`` / ``--probe`` -- add just that one triple
      (mirrors :func:`add_session_to_anatomy_converter`, add-if-missing, with
      ``--force`` to refresh an already-present block).

    Either action writes to disk unless ``--dry-run`` is passed, which prints
    the summary without writing.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(
        description=(
            "Update neuropixels_sites_to_anatomy_converter.json with KS-row-keyed "
            "per-region channel ranges. Pass --regenerate-all to rewrite every "
            "triple already in the file, or --mouse/--session/--probe to add just "
            "one; with no action it prints this help and writes nothing."
        )
    )
    parser.add_argument(
        "--converter-path", default=str(_DEFAULT_CONVERTER_PATH),
        help="Path to the converter JSON to update.",
    )
    parser.add_argument(
        "--ephys-root", default=str(_DEFAULT_EPHYS_ROOT),
        help="Root directory containing per-probe Kilosort outputs.",
    )
    parser.add_argument(
        "--histology-root", default=str(_DEFAULT_HISTOLOGY_ROOT),
        help="Root directory containing per-mouse IBL histology output.",
    )
    parser.add_argument(
        "--regenerate-all", action="store_true",
        help="Bulk-regenerate EVERY triple already in the converter (mutually exclusive with --mouse/--session/--probe).",
    )
    parser.add_argument(
        "--mouse", default=None,
        help="Mouse id for single-triple mode (requires --session and --probe).",
    )
    parser.add_argument(
        "--session", default=None,
        help="Session id (YYYYMMDD...) for single-triple mode.",
    )
    parser.add_argument(
        "--probe", default=None,
        help="Probe id ('imec0'/'imec1') for single-triple mode.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="In single-triple mode, re-regenerate the triple even if it is already present.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print the summary without writing the converter back to disk.",
    )
    args = parser.parse_args()

    single = args.mouse is not None or args.session is not None or args.probe is not None
    if single and args.regenerate_all:
        parser.error("--regenerate-all cannot be combined with --mouse/--session/--probe")
    if single:
        if not (args.mouse and args.session and args.probe):
            parser.error("single-triple mode requires all of --mouse, --session and --probe")
        summary = add_session_to_anatomy_converter(
            args.mouse, args.session, args.probe,
            force=args.force,
            converter_path=args.converter_path,
            ephys_root=args.ephys_root,
            histology_root=args.histology_root,
            dry_run=args.dry_run,
        )
    elif args.regenerate_all:
        summary = regenerate_anatomy_converter(
            converter_path=args.converter_path,
            ephys_root=args.ephys_root,
            histology_root=args.histology_root,
            dry_run=args.dry_run,
        )
    else:
        # No action selected: print usage and exit without touching the converter.
        parser.print_help()
        return

    # Drop the verbose `output` blob from the printed summary unless it's the
    # small path string (dry runs return the in-memory dict).
    summary_print = dict(summary)
    if args.dry_run:
        summary_print["output"] = "(dict returned in-memory)"
    print(json.dumps(summary_print, indent=2))


if __name__ == "__main__":
    _cli()
