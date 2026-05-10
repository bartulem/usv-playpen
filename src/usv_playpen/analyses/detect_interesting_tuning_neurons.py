"""
@author: bartulem
Per-session triage: scan the per-cluster `*_tuning_curves_data.pkl`
files written by `generate-rm`, apply thresholds to the pre-computed
`triage_stats` block, and emit one JSON summary listing flagged
clusters and the modality / direction that fired the flag.

The compute step writes ALL triage statistics (peak_z, divergence runs,
selectivity, monotonicity, info / sparsity / coherence, VMI + Wilcoxon
p-values) so this module is a pure pkl-to-JSON pass — no spike or USV
data are reloaded. Thresholds live in `analyses_settings.json` under
`detect_interesting_tuning_neurons` and can be adjusted without re-
running compute.

Output:
  ephys/tuning_curves/interesting_neurons_<YYYYMMDD>_<HHMMSS>.json
"""

from __future__ import annotations

import json
import math
import pathlib
import pickle
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np


# JSON encoder helpers


def _to_jsonable(o: Any) -> Any:
    """
    Description
    -----------
    Coerce numpy / non-finite scalar types into JSON-friendly Python
    natives so `json.dump` survives without per-call cleanup. NaN and
    +/- inf become `None` (JSON has no native NaN representation).

    Parameters
    ----------
    o (Any)
        Value to coerce. Handled types: `np.integer`, `np.floating`,
        `np.bool_`, `np.ndarray`, `pathlib.Path`. Everything else is
        passed through and will hit `json.dump`'s default error.

    Returns
    -------
    coerced (Any)
        Python-native equivalent of `o`.
    """

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.PurePath):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _safe_float(x: Any) -> float | None:
    """
    Description
    -----------
    Coerce `x` to `float`, returning `None` for non-finite or
    unparseable values. Used to keep the JSON output tidy (no NaN
    sprinkled through the per-cluster details dicts).

    Parameters
    ----------
    x (Any)
        Value expected to be a float-like scalar; may be NaN, None, or
        a numpy scalar.

    Returns
    -------
    out (float | None)
        Finite float, or `None` if not finite / not coercible.
    """

    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# Per-modality flag helpers


def _flag_vmi(
    vmi_payload: dict, alpha: float, min_bouts: int
) -> tuple[str | None, dict | None]:
    """
    Description
    -----------
    Decide whether a cluster's VMI block crosses the significance
    gate. A cluster is flagged when:
      * `n_bouts >= min_bouts`
      * `wilcoxon_pvalue < alpha`
      * `vmi` is finite

    The sign of `vmi` selects the flag direction (`excit` for vmi > 0,
    `suppress` for vmi < 0).

    Parameters
    ----------
    vmi_payload (dict)
        `triage_stats["vmi"][emitter]` block.
    alpha (float)
        Wilcoxon p-value threshold (e.g. 0.01).
    min_bouts (int)
        Minimum number of bouts for VMI to be considered meaningful.

    Returns
    -------
    direction (str | None)
        `"excit"`, `"suppress"`, or `None` (not flagged).
    details (dict | None)
        Compact summary of the VMI evidence, or `None` if not flagged.
    """

    n_bouts = int(vmi_payload.get("n_bouts", 0) or 0)
    if n_bouts < min_bouts:
        return None, None
    vmi = _safe_float(vmi_payload.get("vmi"))
    p = _safe_float(vmi_payload.get("wilcoxon_pvalue"))
    if vmi is None or p is None:
        return None, None
    if p >= alpha:
        return None, None
    direction = "excit" if vmi > 0 else "suppress"
    details = {
        "vmi": vmi,
        "p": p,
        "n_bouts": n_bouts,
        "fr_baseline": _safe_float(vmi_payload.get("fr_baseline")),
        "fr_usv": _safe_float(vmi_payload.get("fr_usv")),
    }
    return direction, details


def _flag_runs(
    direction_block: dict, z_threshold: float, min_run: int
) -> dict | None:
    """
    Description
    -----------
    Decide whether one direction (excit / suppress) of a run-analysis
    block crosses the joint (consecutive-bins, |peak Z|) gate. A
    direction is flagged when:
      * `max_run >= min_run`
      * `|peak_z| >= z_threshold`

    Parameters
    ----------
    direction_block (dict)
        The `excit` or `suppress` sub-dict from a `_run_analysis`
        result. Must contain `max_run`, `peak_z`; may contain run /
        peak axis-value annotations (`run_t_start`, `run_t_end`,
        `peak_t`, `range_low`, `range_high`, `peak_bin_value`,
        `n_bins`).
    z_threshold (float)
        Magnitude threshold on `peak_z`.
    min_run (int)
        Required consecutive-bin run length.

    Returns
    -------
    summary (dict | None)
        Compact dict of evidence (n_bins, max_run, peak_z, plus any
        axis annotations present in the input), or `None` if the
        direction is not flagged.
    """

    if int(direction_block.get("max_run", 0) or 0) < min_run:
        return None
    peak_z = _safe_float(direction_block.get("peak_z"))
    if peak_z is None or abs(peak_z) < z_threshold:
        return None
    keep_keys = (
        "n_bins", "max_run", "peak_z",
        "run_t_start", "run_t_end", "peak_t",
        "range_low", "range_high", "peak_bin_value",
        "run_start_idx", "run_end_idx", "peak_idx",
    )
    out = {}
    for k in keep_keys:
        if k in direction_block:
            v = direction_block[k]
            if isinstance(v, (int, np.integer)):
                out[k] = int(v)
            else:
                out[k] = _safe_float(v)
    return out


def _flag_categorical(payload: dict, z_threshold: float) -> dict | None:
    """
    Description
    -----------
    Categorical-axis flag (no run-analysis applies — categories are
    unordered). Flags when `peak_abs_z >= z_threshold`.

    Parameters
    ----------
    payload (dict)
        `triage_stats["usv_category_tuning"][emitter][cat_feat]` block.
    z_threshold (float)
        Threshold on `peak_abs_z`.

    Returns
    -------
    summary (dict | None)
        Compact dict of evidence, or `None` if not flagged.
    """

    pz = _safe_float(payload.get("peak_abs_z"))
    if pz is None or pz < z_threshold:
        return None
    return {
        "peak_abs_z": pz,
        "peak_signed_z": _safe_float(payload.get("peak_signed_z")),
        "best_cat": int(payload.get("best_cat", -1) or -1),
        "n_sig_categories": int(payload.get("n_sig_categories", 0) or 0),
        "selectivity": _safe_float(payload.get("selectivity")),
    }


def _flag_spatial(payload: dict, info_threshold: float) -> dict | None:
    """
    Description
    -----------
    Spatial flag based on Skaggs information rate. Flags when
    `info_rate_bps >= info_threshold`. Sparsity and coherence are
    reported in the evidence dict but not gated.

    Parameters
    ----------
    payload (dict)
        `triage_stats["spatial"][offset][feature_key]` block.
    info_threshold (float)
        Information-rate threshold (bits/spike).

    Returns
    -------
    summary (dict | None)
        Compact dict of evidence, or `None` if not flagged.
    """

    info = _safe_float(payload.get("info_rate_bps"))
    if info is None or info < info_threshold:
        return None
    return {
        "info_rate_bps": info,
        "sparsity": _safe_float(payload.get("sparsity")),
        "coherence": _safe_float(payload.get("coherence")),
        "peak_rate_sps": _safe_float(payload.get("peak_rate_sps")),
        "peak_row": int(payload.get("peak_row", -1) or -1),
        "peak_col": int(payload.get("peak_col", -1) or -1),
    }


# Emitter -> role lookup (read from any vocal payload that carries it)


def _emitter_role_map(cluster_data: dict) -> dict:
    """
    Description
    -----------
    Build an `{emitter_id: role}` mapping from any vocal payload in
    the pkl that carries `role` per emitter (`usv_peth`, the per-prop
    payloads of `usv_property_tuning`, `usv_category_tuning`,
    `usv_category_peth`). The compute writes `role="self"` /
    `"partner"` consistently across all four blocks, so any one
    suffices; we walk them all defensively in case some are missing.

    Parameters
    ----------
    cluster_data (dict)
        Loaded per-cluster pkl payload.

    Returns
    -------
    mapping (dict)
        `{emitter_id: role_string}`. May be empty if the cluster has
        no vocal payload at all.
    """

    out: dict[str, str] = {}

    def _take(emitter, payload):
        """
        Description
        -----------
        Inner closure: record `emitter -> role` in the enclosing
        `out` dict if `payload` carries a `role` key and the emitter
        has not yet been seen.

        Parameters
        ----------
        emitter (str)
            Emitter ID candidate.
        payload (dict | Any)
            Possibly-`dict` payload to inspect for a `role` field.

        Returns
        -------
        None
        """
        if isinstance(payload, dict) and "role" in payload and emitter not in out:
            out[emitter] = str(payload["role"])

    for block_name in ("usv_peth", "usv_category_tuning", "usv_category_peth"):
        block = cluster_data.get(block_name)
        if not isinstance(block, dict):
            continue
        for emitter, payload in block.items():
            if block_name == "usv_peth":
                _take(emitter, payload)
            else:
                # cat_feat layer in between — payload is a dict-of-cat-feats
                if isinstance(payload, dict):
                    for sub in payload.values():
                        _take(emitter, sub)
                        if emitter in out:
                            break
    # usv_property_tuning has an extra layer (per property)
    block = cluster_data.get("usv_property_tuning")
    if isinstance(block, dict):
        for emitter, props in block.items():
            if emitter in out:
                continue
            if isinstance(props, dict):
                for sub in props.values():
                    _take(emitter, sub)
                    if emitter in out:
                        break
    return out


# Top-level entry point


def detect_interesting_clusters(
    root_directory: str | pathlib.Path,
    *,
    z_threshold: float = 3.0,
    min_consecutive_bins: int = 3,
    vmi_alpha: float = 0.01,
    vmi_min_bouts: int = 10,
    spatial_info_bps_threshold: float = 0.5,
    message_output: Callable = print,
) -> pathlib.Path | None:
    """
    Description
    -----------
    Scan every `*_tuning_curves_data.pkl` under
    `<root_directory>/ephys/tuning_curves/`, apply the configured
    thresholds to each cluster's `triage_stats` block, and write a
    timestamped JSON summary
    (`interesting_neurons_<YYYYMMDD>_<HHMMSS>.json`) listing flagged
    clusters by modality and by cluster.

    Pkls without a `triage_stats` block are skipped silently (likely
    older pkls produced before triage was wired into compute). Returns
    `None` and logs a graceful skip if the tuning_curves directory
    is empty or absent.

    Parameters
    ----------
    root_directory (str | pathlib.Path)
        Session root. The pkls live at
        `<root_directory>/ephys/tuning_curves/`.
    z_threshold (float)
        Magnitude threshold on per-direction `peak_z` in
        usv_peth / usv_property_tuning / usv_category_tuning /
        usv_category_peth / behavioral. Combined with the
        `min_consecutive_bins` rule for the run-analysis modalities.
    min_consecutive_bins (int)
        Required consecutive-bin run length for a direction (excit or
        suppress) to be flagged. Does not apply to categorical (no
        ordered axis) or spatial (uses Skaggs info instead).
    vmi_alpha (float)
        Wilcoxon two-sided p-value threshold for VMI significance.
    vmi_min_bouts (int)
        Minimum bout count required to consider VMI meaningful.
    spatial_info_bps_threshold (float)
        Skaggs information-rate threshold (bits/spike) for spatial
        flag.
    message_output (Callable)
        Logger; defaults to `print`.

    Returns
    -------
    out_path (pathlib.Path | None)
        Path of the written JSON summary, or `None` if nothing was
        written (no pkls / no tuning_curves dir).
    """

    root = pathlib.Path(root_directory)
    pkl_dir = root / "ephys" / "tuning_curves"
    if not pkl_dir.exists():
        message_output(
            f"  detect-interesting: {pkl_dir} not found; skipping."
        )
        return None
    pkls = sorted(pkl_dir.glob("*_tuning_curves_data.pkl"))
    if not pkls:
        message_output(
            f"  detect-interesting: no tuning pkls in {pkl_dir}; skipping."
        )
        return None

    by_modality: dict[str, list[str]] = {}
    by_cluster: dict[str, dict] = {}
    n_skipped_no_triage = 0

    for pkl in pkls:
        cluster_id = pkl.stem.replace("_tuning_curves_data", "")
        try:
            with pkl.open("rb") as fh:
                cluster_data = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            message_output(f"  failed to load {pkl.name}: {exc}")
            continue

        ts = cluster_data.get("triage_stats")
        if not isinstance(ts, dict):
            n_skipped_no_triage += 1
            continue

        emitter_roles = _emitter_role_map(cluster_data)

        flagged: list[str] = []
        details: dict[str, Any] = {}

        # VMI per emitter
        for emitter, payload in (ts.get("vmi") or {}).items():
            role = (
                payload.get("role")
                or emitter_roles.get(emitter)
                or emitter
            )
            direction, info = _flag_vmi(payload, vmi_alpha, vmi_min_bouts)
            if direction:
                key = f"vmi_{role}_{direction}"
                flagged.append(key)
                details[key] = info
                by_modality.setdefault(key, []).append(cluster_id)

        # usv_peth per emitter (1D PETH)
        for emitter, payload in (ts.get("usv_peth") or {}).items():
            role = emitter_roles.get(emitter, emitter)
            for direction in ("excit", "suppress"):
                block = payload.get(direction)
                if not isinstance(block, dict):
                    continue
                info = _flag_runs(block, z_threshold, min_consecutive_bins)
                if info:
                    key = f"usv_peth_{role}_{direction}"
                    flagged.append(key)
                    info["ramp_index"] = _safe_float(payload.get("ramp_index"))
                    details[key] = info
                    by_modality.setdefault(key, []).append(cluster_id)

        # usv_property_tuning per (emitter, property)
        for emitter, props in (ts.get("usv_property_tuning") or {}).items():
            role = emitter_roles.get(emitter, emitter)
            for prop, payload in props.items():
                for direction in ("excit", "suppress"):
                    block = payload.get(direction)
                    if not isinstance(block, dict):
                        continue
                    info = _flag_runs(
                        block, z_threshold, min_consecutive_bins
                    )
                    if info:
                        key = f"usv_property_{role}_{prop}_{direction}"
                        flagged.append(key)
                        info["selectivity"] = _safe_float(
                            payload.get("selectivity")
                        )
                        info["monotonicity"] = _safe_float(
                            payload.get("monotonicity")
                        )
                        details[key] = info
                        by_modality.setdefault(key, []).append(cluster_id)

        # usv_category_tuning per (emitter, cat_feat) — categorical, no run rule
        for emitter, cats in (ts.get("usv_category_tuning") or {}).items():
            role = emitter_roles.get(emitter, emitter)
            for cat_feat, payload in cats.items():
                info = _flag_categorical(payload, z_threshold)
                if info:
                    key = f"usv_category_{role}_{cat_feat}"
                    flagged.append(key)
                    details[key] = info
                    by_modality.setdefault(key, []).append(cluster_id)

        # usv_category_peth per (emitter, cat_feat) — best per cat_feat
        for emitter, cats in (ts.get("usv_category_peth") or {}).items():
            role = emitter_roles.get(emitter, emitter)
            for cat_feat, payload in cats.items():
                for direction_name, key_tag in (
                    ("best_excit", "excit"),
                    ("best_suppress", "suppress"),
                ):
                    block = payload.get(direction_name)
                    if not isinstance(block, dict):
                        continue
                    info = _flag_runs(
                        block, z_threshold, min_consecutive_bins
                    )
                    if info:
                        key = f"usv_category_peth_{role}_{cat_feat}_{key_tag}"
                        flagged.append(key)
                        info["best_cat"] = int(
                            payload.get("best_cat", -1) or -1
                        )
                        info["best_t"] = _safe_float(payload.get("best_t"))
                        details[key] = info
                        by_modality.setdefault(key, []).append(cluster_id)

        # behavioral 1D per (offset, feature)
        for offset_key, feats in (ts.get("behavioral") or {}).items():
            for feat_key, payload in feats.items():
                for direction in ("excit", "suppress"):
                    block = payload.get(direction)
                    if not isinstance(block, dict):
                        continue
                    info = _flag_runs(
                        block, z_threshold, min_consecutive_bins
                    )
                    if info:
                        key = (
                            f"behavioral_{offset_key}_{feat_key}_{direction}"
                        )
                        flagged.append(key)
                        info["selectivity"] = _safe_float(
                            payload.get("selectivity")
                        )
                        info["monotonicity"] = _safe_float(
                            payload.get("monotonicity")
                        )
                        info["is_circular"] = bool(
                            payload.get("is_circular", False)
                        )
                        details[key] = info
                        by_modality.setdefault(key, []).append(cluster_id)

        # spatial per (offset, feature)
        for offset_key, feats in (ts.get("spatial") or {}).items():
            for feat_key, payload in feats.items():
                info = _flag_spatial(payload, spatial_info_bps_threshold)
                if info:
                    key = f"spatial_{offset_key}_{feat_key}"
                    flagged.append(key)
                    details[key] = info
                    by_modality.setdefault(key, []).append(cluster_id)

        if flagged:
            by_cluster[cluster_id] = {
                "modalities_flagged": flagged,
                "details": details,
            }

    out = {
        "session_root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds_used": {
            "z_threshold": z_threshold,
            "min_consecutive_bins": min_consecutive_bins,
            "vmi_alpha": vmi_alpha,
            "vmi_min_bouts": vmi_min_bouts,
            "spatial_info_bps_threshold": spatial_info_bps_threshold,
        },
        "n_clusters_total": len(pkls),
        "n_clusters_skipped_no_triage": n_skipped_no_triage,
        "n_clusters_flagged": len(by_cluster),
        "by_modality": {k: sorted(v) for k, v in sorted(by_modality.items())},
        "by_cluster": by_cluster,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = pkl_dir / f"interesting_neurons_{timestamp}.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, default=_to_jsonable)
    message_output(
        f"  detect-interesting: {len(by_cluster)} / {len(pkls)} cluster(s) "
        f"flagged (skipped {n_skipped_no_triage} pkl(s) lacking triage_stats). "
        f"Wrote {out_path.name}"
    )
    return out_path
