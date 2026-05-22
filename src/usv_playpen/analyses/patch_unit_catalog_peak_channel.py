"""
@author: bartulem
Patch ``unit_catalog.csv`` in place by re-running monopolar-source
triangulation per unit with the candidate channel set RESTRICTED to
the unit's template-peak shank, then overwriting ``closest_ch``,
``brain_area`` and the ``loc_ap`` / ``loc_ml`` / ``loc_dv`` columns
with the resulting position.

Background
----------
The catalog was originally written by
``SpikeQualityMetricsExtractor.compute_unit_locations`` on the
`neural` branch, which fits a monopolar source over the unit's
sparse channel set spanning **all four shanks** of the Neuropixels
2.0 probe. With the IBL-aligned channel coordinates used here every
channel on a probe shares the same ML (``x``) value and the
inter-shank separation is folded entirely into AP (``y``). Templates
are dense (non-zero on every channel), so even very small "ghost"
amplitudes on far shanks can pull the unconstrained 3D centroid
AP-ward and the catalog's ``closest_ch = argmin_ch ||chan - src||``
ends up on a different shank than the one where the template peaks.

This patch fixes the shank-jump in two steps per unit:

1. Find the channel where the Kilosort template's per-channel
   peak-to-peak amplitude is largest (``template_peak_ch``) and read
   its shank from ``channel_shanks.npy``.
2. Re-fit the monopolar triangulation **using only the 96 channels
   on that shank**. The fitted source is therefore guaranteed to live
   on the correct shank; ``closest_ch`` is then ``argmin`` distance
   within that shank, ``brain_area = converter[closest_ch]``, and
   ``loc_ap / loc_ml / loc_dv`` are the fitted source's ``(y, x, z)``
   in the same folded coordinate space the original pipeline used.

Re-running ``SpikeQualityMetricsExtractor`` over the whole dataset is
not an option for this dataset, so this script writes the corrected
columns directly to ``unit_catalog.csv`` (with a timestamped backup).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import datetime

import numpy as np
import pandas as pd

from usv_playpen.analyses.npx_monopolar_triangulation import (
    solve_monopolar_triangulation_3d,
)


_DEFAULT_CATALOG_PATH = pathlib.Path("/mnt/falkner/Bartul/EPHYS/unit_catalog.csv")
_DEFAULT_EPHYS_ROOT = pathlib.Path("/mnt/falkner/Bartul/EPHYS")
_DEFAULT_CONVERTER_PATH = (
    pathlib.Path("/mnt/falkner/Bartul/EPHYS")
    / "neuropixels_sites_to_anatomy_converter.json"
)
_DEFAULT_HISTOLOGY_ROOT = pathlib.Path("/mnt/falkner/Bartul/histology")

# Matches the constants in the `npx_spike_quality_metrics` block of
# `_parameter_settings/analyses_settings.json` on the `neural` branch.
_SHANK_WIDTH_UM = 70.0
_SHANK_SPACING_UM = 250.0
_MAX_DISTANCE_UM = 1000
_OPTIMIZER = "minimize_with_log_penality"

# The Neuropixels 2.0 mapping is the same across every session and
# every mouse in this dataset (user confirmed). If that ever changes
# this dict needs to be swapped for a per-session lookup.
_PROBE_TO_HEMISPHERE = {"imec0": "R", "imec1": "L"}

_PROBE_RE = re.compile(r"(imec\d)")
_CLUSTER_NUM_RE = re.compile(r"cl(\d{4})")


def _build_channel_region_lookup(
        converter: dict,
        mouse_id: str,
        rec_date: int,
        probe: str,
) -> dict[int, str]:
    """
    Description
    -----------
    Build a per-probe ``{channel_index: region_name}`` lookup from the
    ``neuropixels_sites_to_anatomy_converter.json`` payload.

    Same-day sessions share the per-probe channel-to-region map
    (Kilosort is run on the within-day concatenation), so the first
    session key matching the given ``rec_date`` is used.

    Parameters
    ----------
    converter (dict)
        Parsed JSON payload of the anatomy converter.
    mouse_id (str)
        Catalog ``mouse_id`` string.
    rec_date (int)
        Recording date as YYYYMMDD integer.
    probe (str)
        ``'imec0'`` or ``'imec1'``.

    Returns
    -------
    ch_to_region (dict[int, str])
        Mapping from probe channel index to the region name that
        contains it. Channels outside any converter range are absent
        from the dict (callers should default to ``'unknown'``).
    """

    mouse_sessions = converter.get(mouse_id, {})
    matching = [k for k in mouse_sessions if k.startswith(str(rec_date))]
    if not matching:
        return {}
    probe_anatomy = mouse_sessions[matching[0]].get(probe, {})
    out: dict[int, str] = {}
    for region_name, ranges in probe_anatomy.items():
        for lo, hi in ranges:
            for ch in range(int(lo), int(hi)):
                out[ch] = region_name
    return out


def _load_ibl_channel_locations(
        histology_root: pathlib.Path,
        mouse_id: str,
        rec_date: int,
        hemisphere: str,
) -> dict:
    """
    Description
    -----------
    Load the IBL-aligned per-channel coordinates JSON for one
    (mouse, recording-date, hemisphere). Returns the raw mapping
    ``{'channel_<i>': {'x': ml, 'y': ap, 'z': dv, 'lateral': lat,
    'axial': axi, 'brain_region': str, ...}}`` so the caller can apply
    the original ``compute_unit_locations`` lateral-into-AP fold per
    channel.

    Parameters
    ----------
    histology_root (pathlib.Path)
        Root directory of the histology tree
        (``<mouse_id>/<rec_date>/ibl_{L,R}H/channel_locations.json``).
    mouse_id (str)
        Catalog ``mouse_id`` string.
    rec_date (int)
        Recording date as YYYYMMDD integer.
    hemisphere (str)
        ``'L'`` or ``'R'``.

    Returns
    -------
    channel_locations (dict)
        Per-channel dict keyed by ``"channel_<int>"``.
    """

    hemi_dir = "ibl_RH" if hemisphere.upper().startswith("R") else "ibl_LH"
    path = histology_root / mouse_id / str(rec_date) / hemi_dir / "channel_locations.json"
    with path.open() as fh:
        return json.load(fh)


def _build_folded_channel_locs(
        channel_locations: dict,
        hemisphere: str,
        n_channels: int,
) -> np.ndarray:
    """
    Description
    -----------
    Apply the lateral-into-AP fold used by the original
    ``compute_unit_locations`` to every channel and return a dense
    ``(n_channels, 3)`` array of ``(x, y_folded, z)``.

    The fold mirrors the original implementation byte-for-byte:

        within_shank_lateral = lateral % shank_spacing
        if hemisphere == 'R':
            y_folded = y + shank_width - within_shank_lateral
        else:
            y_folded = y - shank_width + within_shank_lateral

    Parameters
    ----------
    channel_locations (dict)
        Output of ``_load_ibl_channel_locations`` for this
        (mouse, date, hemisphere).
    hemisphere (str)
        ``'L'`` or ``'R'``.
    n_channels (int)
        Total number of channels on the probe (typically 384).

    Returns
    -------
    chan_locs (np.ndarray)
        ``(n_channels, 3)`` array of ``[x, y_folded, z]`` per channel.
    """

    is_right = hemisphere.upper().startswith("R")
    chan_locs = np.zeros((n_channels, 3), dtype=float)
    for ch in range(n_channels):
        entry = channel_locations[f"channel_{ch}"]
        within = float(entry["lateral"]) % _SHANK_SPACING_UM
        if is_right:
            y_folded = float(entry["y"]) + _SHANK_WIDTH_UM - within
        else:
            y_folded = float(entry["y"]) - _SHANK_WIDTH_UM + within
        chan_locs[ch] = (float(entry["x"]), y_folded, float(entry["z"]))
    return chan_locs


def _triangulate_within_shank_per_cluster(
        ks_dir: pathlib.Path,
        chan_locs_folded: np.ndarray,
        cluster_nums: list[int],
) -> dict[int, dict]:
    """
    Description
    -----------
    For each cluster in ``cluster_nums``:

    1. Resolve its primary Kilosort template (the template ID held by
       the most spikes — robust to manual merges).
    2. Compute per-channel peak-to-peak amplitude.
    3. Pick the channel with the largest amplitude as
       ``template_peak_ch``; read its shank from ``channel_shanks.npy``.
    4. Restrict the contact set to that shank (96 channels) and run
       ``solve_monopolar_triangulation_3d`` over those channels with
       the ``minimize_with_log_penality`` optimiser.
    5. Find ``closest_ch`` as the channel on that shank with smallest
       Euclidean distance to the fitted source.

    Parameters
    ----------
    ks_dir (pathlib.Path)
        Path to the per-probe ``kilosort4/`` directory.
    chan_locs_folded (np.ndarray)
        ``(n_channels, 3)`` array of ``(x, y_folded, z)`` per channel,
        in the same coordinate space the original triangulation used.
    cluster_nums (list[int])
        Cluster IDs to update.

    Returns
    -------
    cluster_to_result (dict[int, dict])
        Mapping ``cluster_num -> {'closest_ch': int, 'loc_ml': float,
        'loc_ap': float, 'loc_dv': float, 'template_peak_ch': int}``.
        Clusters with no spikes in ``spike_clusters.npy`` are absent.
    """

    spike_clusters = (
        np.load(ks_dir / "spike_clusters.npy").flatten().astype(np.int64)
    )
    spike_templates = (
        np.load(ks_dir / "spike_templates.npy").flatten().astype(np.int64)
    )
    templates = np.load(ks_dir / "templates.npy", mmap_mode="r")
    channel_shanks = (
        np.load(ks_dir / "channel_shanks.npy").astype(int)
    )

    out: dict[int, dict] = {}
    for cluster_num in cluster_nums:
        mask = spike_clusters == cluster_num
        if not mask.any():
            continue
        template_ids, counts = np.unique(
            spike_templates[mask], return_counts=True
        )
        primary = int(template_ids[counts.argmax()])
        template = np.asarray(templates[primary])
        ptp_per_ch = np.ptp(template, axis=0)
        template_peak_ch = int(ptp_per_ch.argmax())
        peak_shank = int(channel_shanks[template_peak_ch])

        same_shank_idx = np.where(channel_shanks == peak_shank)[0]
        wf_data = ptp_per_ch[same_shank_idx].astype(float)
        chan_locs_local = chan_locs_folded[same_shank_idx]

        loc = solve_monopolar_triangulation_3d(
            wf_data, chan_locs_local, _MAX_DISTANCE_UM, _OPTIMIZER,
        )
        loc_array = np.asarray(loc, dtype=float)
        if loc_array.size >= 3:
            src_xyz = loc_array[:3]
        else:
            # Optimiser failed; fall back to the template peak channel.
            src_xyz = chan_locs_local[
                int(np.argmax(wf_data))
            ]

        distances = np.linalg.norm(chan_locs_local - src_xyz, axis=1)
        closest_local = int(np.argmin(distances))
        closest_channel = int(same_shank_idx[closest_local])

        out[cluster_num] = {
            "closest_ch": closest_channel,
            "loc_ml": float(src_xyz[0]),
            "loc_ap": float(src_xyz[1]),
            "loc_dv": float(src_xyz[2]),
            "template_peak_ch": template_peak_ch,
        }
    return out


def patch_unit_catalog_peak_channel(
        catalog_path: str | pathlib.Path = _DEFAULT_CATALOG_PATH,
        ephys_root: str | pathlib.Path = _DEFAULT_EPHYS_ROOT,
        converter_path: str | pathlib.Path = _DEFAULT_CONVERTER_PATH,
        histology_root: str | pathlib.Path = _DEFAULT_HISTOLOGY_ROOT,
        backup: bool = True,
        dry_run: bool = False,
) -> dict:
    """
    Description
    -----------
    Rewrite the catalog's ``closest_ch``, ``brain_area``,
    ``loc_ap``, ``loc_ml`` and ``loc_dv`` columns by re-running
    monopolar 3D triangulation with the candidate channel set
    restricted to each unit's template-peak shank.

    The patch is in-place: ``catalog_path`` is overwritten. When
    ``backup=True`` (default) a timestamped backup file is dropped
    alongside it before any writes. When ``dry_run=True`` no files
    are touched; the diff summary is still returned.

    Parameters
    ----------
    catalog_path (str | pathlib.Path)
        Path to ``unit_catalog.csv``.
    ephys_root (str | pathlib.Path)
        Root containing ``<YYYYMMDD>_imec<i>/kilosort4/`` per probe.
    converter_path (str | pathlib.Path)
        Path to ``neuropixels_sites_to_anatomy_converter.json``.
    histology_root (str | pathlib.Path)
        Root containing ``<mouse_id>/<rec_date>/ibl_{LH,RH}/
        channel_locations.json``.
    backup (bool)
        If True (default) write a timestamped ``.bak`` next to the
        catalog before overwriting it.
    dry_run (bool)
        If True, compute the diff but do not write anything.

    Returns
    -------
    summary (dict)
        Keys ``n_total``, ``n_closest_ch_changed``,
        ``n_brain_area_changed``, ``n_loc_changed``,
        ``brain_area_transitions`` (``{'from->to': count}``), and
        ``backup_path`` (or ``None``).
    """

    catalog_path = pathlib.Path(catalog_path)
    ephys_root = pathlib.Path(ephys_root)
    converter_path = pathlib.Path(converter_path)
    histology_root = pathlib.Path(histology_root)

    if backup and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = catalog_path.with_name(
            f"{catalog_path.stem}.bak_{timestamp}{catalog_path.suffix}"
        )
        backup_path.write_bytes(catalog_path.read_bytes())
    else:
        backup_path = None

    with converter_path.open() as fh:
        converter = json.load(fh)

    df = pd.read_csv(catalog_path)
    df["_probe"] = df["unit_id"].str.extract(_PROBE_RE, expand=False)
    df["_cluster_num"] = (
        df["unit_id"].str.extract(_CLUSTER_NUM_RE, expand=False).astype("Int64")
    )

    original_closest_ch = df["closest_ch"].copy()
    original_brain_area = df["brain_area"].copy()
    original_loc_ap = df["loc_ap"].copy()
    original_loc_ml = df["loc_ml"].copy()
    original_loc_dv = df["loc_dv"].copy()

    for (mouse_id, rec_date, probe), group in df.groupby(
        ["mouse_id", "rec_date", "_probe"]
    ):
        ks_dir = ephys_root / f"{rec_date}_{probe}" / "kilosort4"
        if not ks_dir.exists():
            continue

        hemisphere = _PROBE_TO_HEMISPHERE[str(probe)]
        try:
            channel_locations = _load_ibl_channel_locations(
                histology_root, str(mouse_id), int(rec_date), hemisphere,
            )
        except FileNotFoundError:
            # Histology not present for this session; leave rows alone.
            continue

        ch_to_region = _build_channel_region_lookup(
            converter, str(mouse_id), int(rec_date), str(probe),
        )

        # Build the folded (x, y, z) lookup once per probe-day. The
        # IBL JSON also carries non-channel metadata keys ("origin"),
        # so count only the "channel_<int>" entries.
        n_channels = sum(
            1 for k in channel_locations if k.startswith("channel_")
        )
        chan_locs_folded = _build_folded_channel_locs(
            channel_locations, hemisphere, n_channels,
        )

        cluster_nums = sorted({
            int(c) for c in group["_cluster_num"].dropna()
        })
        cluster_to_result = _triangulate_within_shank_per_cluster(
            ks_dir, chan_locs_folded, cluster_nums,
        )

        for idx in group.index:
            cnum = df.at[idx, "_cluster_num"]
            if pd.isna(cnum):
                continue
            res = cluster_to_result.get(int(cnum))
            if res is None:
                continue
            df.at[idx, "closest_ch"] = res["closest_ch"]
            df.at[idx, "brain_area"] = ch_to_region.get(
                res["closest_ch"], "unknown"
            )
            df.at[idx, "loc_ml"] = res["loc_ml"]
            df.at[idx, "loc_ap"] = res["loc_ap"]
            df.at[idx, "loc_dv"] = res["loc_dv"]

    df = df.drop(columns=["_probe", "_cluster_num"])

    n_closest_ch_changed = int((df["closest_ch"] != original_closest_ch).sum())
    n_brain_area_changed = int((df["brain_area"] != original_brain_area).sum())
    n_loc_changed = int(
        (df["loc_ap"] != original_loc_ap).any()
        or (df["loc_ml"] != original_loc_ml).any()
        or (df["loc_dv"] != original_loc_dv).any()
    )
    n_loc_rows_changed = int(
        (
            (df["loc_ap"] != original_loc_ap)
            | (df["loc_ml"] != original_loc_ml)
            | (df["loc_dv"] != original_loc_dv)
        ).sum()
    )
    transitions = (
        pd.DataFrame({"from": original_brain_area, "to": df["brain_area"]})
        .loc[lambda d: d["from"] != d["to"]]
        .groupby(["from", "to"])
        .size()
        .to_dict()
    )

    if not dry_run:
        df.to_csv(catalog_path, index=False)

    return {
        "n_total": int(len(df)),
        "n_closest_ch_changed": n_closest_ch_changed,
        "n_brain_area_changed": n_brain_area_changed,
        "n_loc_rows_changed": n_loc_rows_changed,
        "brain_area_transitions": {
            f"{frm}->{to}": int(n) for (frm, to), n in transitions.items()
        },
        "backup_path": str(backup_path) if backup_path else None,
    }


def _cli() -> None:
    """
    Description
    -----------
    Argparse entry point so the patch can be triggered from the
    command line. Mirrors the kwargs of
    :func:`patch_unit_catalog_peak_channel`.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(
        description=(
            "Patch unit_catalog.csv closest_ch / brain_area / loc_* by "
            "re-running monopolar triangulation restricted to the "
            "template peak's shank."
        )
    )
    parser.add_argument(
        "--catalog-path", default=str(_DEFAULT_CATALOG_PATH),
        help="Path to unit_catalog.csv (will be overwritten).",
    )
    parser.add_argument(
        "--ephys-root", default=str(_DEFAULT_EPHYS_ROOT),
        help="Root directory containing per-probe Kilosort outputs.",
    )
    parser.add_argument(
        "--converter-path", default=str(_DEFAULT_CONVERTER_PATH),
        help="Path to neuropixels_sites_to_anatomy_converter.json.",
    )
    parser.add_argument(
        "--histology-root", default=str(_DEFAULT_HISTOLOGY_ROOT),
        help="Root directory containing per-mouse/per-session IBL JSONs.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip writing a timestamped backup of the catalog.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the diff summary without writing anything to disk.",
    )
    args = parser.parse_args()

    summary = patch_unit_catalog_peak_channel(
        catalog_path=args.catalog_path,
        ephys_root=args.ephys_root,
        converter_path=args.converter_path,
        histology_root=args.histology_root,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
