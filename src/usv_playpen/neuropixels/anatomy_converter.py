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
anatomy of a different physical electrode for ~99% of channels. See
``docs/channel_indexing.rst`` for the full landscape.

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
        pos_to_region[(int(entry["lateral"]), int(entry["axial"]))] = (
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
        pos = (
            int(channel_positions[ks_row, 0]),
            int(channel_positions[ks_row, 1]),
        )
        # Positions absent from the IBL map (e.g. reference/disconnected
        # sites) become a literal "unknown" region; `_runs_to_ranges`
        # emits it as its own first-class key, so downstream
        # `find_region_by_channel` can return "unknown" as a brain region.
        region = pos_to_region[pos] if pos in pos_to_region else "unknown"
        per_row_region.append(region)
    return _runs_to_ranges(per_row_region)


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
    for simplicity but the result is the same).

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

    regenerated: dict[str, Any] = {}
    n_total = 0
    n_done = 0
    skipped: list[str] = []
    for mouse_id, mouse_entry in existing.items():
        new_mouse_entry: dict[str, dict] = {}
        for session_id, probes_entry in mouse_entry.items():
            try:
                rec_date = int(session_id[:8])
            except ValueError:
                n_total += len(probes_entry)
                for p in probes_entry:
                    skipped.append(f"{mouse_id}/{session_id}/{p}: bad session_id")
                continue
            new_probes_entry: dict[str, dict] = {}
            for probe, _old_block in probes_entry.items():
                n_total += 1
                ks_dir = ephys_root / f"{rec_date}_{probe}" / "kilosort4"
                if not (ks_dir / "channel_positions.npy").exists():
                    skipped.append(
                        f"{mouse_id}/{session_id}/{probe}: no channel_positions.npy"
                    )
                    continue
                if probe not in probe_to_hemisphere:
                    skipped.append(
                        f"{mouse_id}/{session_id}/{probe}: no hemisphere mapping"
                    )
                    continue
                hemisphere = probe_to_hemisphere[probe]
                try:
                    pos_to_region = _load_ibl_position_to_region(
                        histology_root, str(mouse_id), int(rec_date), hemisphere,
                    )
                except FileNotFoundError:
                    skipped.append(
                        f"{mouse_id}/{session_id}/{probe}: no IBL channel_locations.json"
                    )
                    continue
                ks_block = _build_ks_keyed_block(ks_dir, pos_to_region)
                # A block whose ONLY region is "unknown" means every Kilosort
                # channel failed the IBL (lateral, axial) position join (e.g. a
                # wrong-hemisphere file, a units/coordinate mismatch, or a stale
                # channel_positions.npy). Since this module exists to FIX the
                # converter, silently overwriting a probe with an all-"unknown"
                # block is the worst outcome (find_region_by_channel then returns
                # "unknown"/None for every cluster), so warn loudly, record it in
                # skipped_reasons, and leave that probe's converter entry untouched
                # rather than counting it as a successful regeneration. A handful
                # of legitimately-unknown sites (reference/disconnected channels)
                # still produce real regions and so are unaffected.
                if set(ks_block) == {"unknown"}:
                    msg = (
                        f"{mouse_id}/{session_id}/{probe}: every channel mapped to "
                        f"'unknown' (IBL position join failed -- wrong-hemisphere file, "
                        f"units/coordinate mismatch, or stale channel_positions.npy); "
                        f"probe skipped, converter NOT overwritten for it."
                    )
                    warnings.warn(msg, stacklevel=2)
                    skipped.append(msg)
                    continue
                new_probes_entry[probe] = ks_block
                n_done += 1
            if new_probes_entry:
                new_mouse_entry[session_id] = new_probes_entry
        if new_mouse_entry:
            regenerated[mouse_id] = new_mouse_entry

    if not dry_run:
        # Pretty-print the converter so that the entire per-region
        # `[[lo, hi], [lo, hi], ...]` list lands on a single line:
        #
        #   "PAG": [[0, 40], [72, 136], [174, 192], ...]
        #
        # Python's default `json.dump(indent=4)` would split every
        # `[lo, hi]` AND the outer list across many lines — wall of
        # one-int rows that's painful to read. Render with stock
        # `indent=4` first to get the nested structure, then collapse
        # in two passes:
        #   1. Every `[\n  lo,\n  hi\n]` -> `[lo, hi]`.
        #   2. Every outer list whose items are now all `[int, int]`
        #      -> `[[lo, hi], [lo, hi], ...]` on one line.
        rendered = json.dumps(regenerated, indent=4)
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
        converter_path.write_text(rendered)

    return {
        "n_triples_total": n_total,
        "n_triples_regenerated": n_done,
        "n_triples_skipped": len(skipped),
        "skipped_reasons": skipped,
        "output": str(converter_path) if not dry_run else regenerated,
    }


def _cli() -> None:
    """
    Description
    -----------
    Argparse entry point for the regenerator. Mirrors the kwargs of
    :func:`regenerate_anatomy_converter`.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate neuropixels_sites_to_anatomy_converter.json with "
            "KS-row-keyed per-region channel ranges."
        )
    )
    parser.add_argument(
        "--converter-path", default=str(_DEFAULT_CONVERTER_PATH),
        help="Path to the converter JSON to overwrite.",
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
        "--dry-run", action="store_true",
        help="Compute and print the regeneration counts summary without writing the converter back to disk.",
    )
    args = parser.parse_args()

    summary = regenerate_anatomy_converter(
        converter_path=args.converter_path,
        ephys_root=args.ephys_root,
        histology_root=args.histology_root,
        dry_run=args.dry_run,
    )
    # Drop the verbose `output` blob from the printed summary unless
    # it's the small dry-run path string.
    summary_print = dict(summary)
    if args.dry_run:
        summary_print["output"] = "(dict returned in-memory)"
    print(json.dumps(summary_print, indent=2))


if __name__ == "__main__":
    _cli()
