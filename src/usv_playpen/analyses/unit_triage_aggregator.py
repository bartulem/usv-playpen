"""
@author: bartulem
Cross-session / cross-condition unit-triage aggregator.

`aggregate_units_across_conditions` applies per-cluster significance
rules (delegated to `flag_one_cluster`) across a labelled set of
sessions — one `.txt` list per experimental condition — joins each
cluster with `unit_catalog.csv`, and pickles a unit-keyed roll-up so
the same physical unit recorded across replicate sessions in a day
is represented once with per-session evidence stacked underneath
each modality.

The compute step writes ALL triage statistics (peak_z, divergence
runs, selectivity, monotonicity, info / sparsity / coherence, VMI +
Wilcoxon p-values) so this module is a pure pkl-to-pickle pass — no
spike or USV data are reloaded. Thresholds live in
`analyses_settings.json` under `detect_interesting_tuning_neurons`
and can be adjusted without re-running compute.

Output:
  <out_dir>/unit_triage_<YYYYMMDD>_<HHMMSS>.pkl
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
import pandas as pd

from ..os_utils import atomic_output_path, resolve_data_root


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
        `np.bool_`, `np.ndarray`, `pathlib.PurePath`. Everything else is
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


# Per-modality metric extractors (no threshold gates)


def _extract_vmi_metrics(payload: dict) -> dict:
    """
    Description
    -----------
    Pull the metric fields from a VMI block (`triage_stats["vmi"][emitter]`)
    without applying any significance gate. The same field set is used by
    `_flag_vmi` (which returns the dict only when significant) and by
    `flag_one_cluster` (which records the dict every time the block was
    evaluated, with the significance verdict on the side).

    Parameters
    ----------
    payload (dict)
        `triage_stats["vmi"][emitter]` block.

    Returns
    -------
    metrics (dict)
        `{vmi, p, n_bouts, n_valid_pairs, fr_baseline, fr_usv}` with floats
        coerced via `_safe_float` (NaN / non-finite → None) and integers
        coerced to Python `int`. `n_valid_pairs` (the number of bouts that
        actually entered the Wilcoxon test) is `None` for legacy pkls written
        before the field existed; gating falls back to `n_bouts` in that case.
    """

    _n_valid = payload.get("n_valid_pairs")
    return {
        "vmi": _safe_float(payload["vmi"]),
        "p": _safe_float(payload["wilcoxon_pvalue"]),
        "n_bouts": int(payload["n_bouts"]),
        "n_valid_pairs": int(_n_valid) if _n_valid is not None else None,
        "fr_baseline": _safe_float(payload["fr_baseline"]),
        "fr_usv": _safe_float(payload["fr_usv"]),
    }


def _extract_run_metrics(direction_block: dict) -> dict:
    """
    Description
    -----------
    Pull the metric fields from a run-analysis direction block (the
    `excit` or `suppress` sub-dict of a `usv_peth` / `usv_property_tuning`
    / `usv_category_peth` / `behavioral` payload). Shared by `_flag_runs`
    (after the threshold gate passes) and `flag_one_cluster` (called on
    every eligible block).

    Parameters
    ----------
    direction_block (dict)
        The `excit` or `suppress` sub-dict from a run-analysis result.

    Returns
    -------
    metrics (dict)
        Subset of `(n_bins, max_run, peak_z, run_t_start, run_t_end,
        peak_t, range_low, range_high, peak_bin_value, run_start_idx,
        run_end_idx, peak_idx)` that were present in the block, with
        integers coerced to `int` and floats coerced via `_safe_float`.
    """

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


def _extract_categorical_metrics(payload: dict) -> dict:
    """
    Description
    -----------
    Pull the metric fields from a categorical-axis tuning block
    (`triage_stats["usv_category_tuning"][emitter][cat_feat]`). Shared by
    `_flag_categorical` and `flag_one_cluster`.

    Parameters
    ----------
    payload (dict)
        `triage_stats["usv_category_tuning"][emitter][cat_feat]` block.

    Returns
    -------
    metrics (dict)
        `{peak_abs_z, peak_signed_z, best_cat, n_sig_categories,
        selectivity}` with the usual numeric coercions.
    """

    _best_cat = payload["best_cat"]
    return {
        "peak_abs_z": _safe_float(payload["peak_abs_z"]),
        "peak_signed_z": _safe_float(payload["peak_signed_z"]),
        "best_cat": int(_best_cat) if _best_cat is not None else -1,
        "n_sig_categories": int(payload["n_sig_categories"]),
        "selectivity": _safe_float(payload["selectivity"]),
    }


def _extract_spatial_metrics(payload: dict) -> dict:
    """
    Description
    -----------
    Pull the metric fields from a spatial tuning block
    (`triage_stats["spatial"][offset][feature_key]`). Shared by
    `_flag_spatial` and `flag_one_cluster`.

    Parameters
    ----------
    payload (dict)
        `triage_stats["spatial"][offset][feature_key]` block.

    Returns
    -------
    metrics (dict)
        `{info_rate_bps, sparsity, coherence, peak_rate_sps, peak_row,
        peak_col}` with the usual numeric coercions.
    """

    _peak_row = payload["peak_row"]
    _peak_col = payload["peak_col"]
    return {
        "info_rate_bps": _safe_float(payload["info_rate_bps"]),
        "sparsity": _safe_float(payload["sparsity"]),
        "coherence": _safe_float(payload["coherence"]),
        "peak_rate_sps": _safe_float(payload["peak_rate_sps"]),
        "peak_row": int(_peak_row) if _peak_row is not None else -1,
        "peak_col": int(_peak_col) if _peak_col is not None else -1,
    }


# Per-modality flag helpers


def _flag_vmi(
    vmi_payload: dict, alpha: float, min_bouts: int
) -> tuple[str | None, dict | None]:
    """
    Description
    -----------
    Decide whether a cluster's VMI block crosses the significance
    gate. A cluster is flagged when:
      * `n_valid_pairs >= min_bouts`
      * `wilcoxon_pvalue < alpha`
      * `vmi` is finite

    The bout-count gate uses `n_valid_pairs` (the bouts that actually
    entered the Wilcoxon test) when the payload carries it, and falls back
    to the total `n_bouts` for legacy pkls written before that field
    existed.

    The sign of `vmi` selects the flag direction (`excit` for vmi > 0,
    `suppress` for vmi < 0).

    Parameters
    ----------
    vmi_payload (dict)
        `triage_stats["vmi"][emitter]` block.
    alpha (float)
        Wilcoxon p-value threshold (e.g. 0.01).
    min_bouts (int)
        Minimum number of paired bouts for VMI to be considered meaningful.

    Returns
    -------
    direction (str | None)
        `"excit"`, `"suppress"`, or `None` (not flagged).
    details (dict | None)
        Compact summary of the VMI evidence, or `None` if not flagged.
    """

    metrics = _extract_vmi_metrics(vmi_payload)
    n_for_gate = (
        metrics["n_valid_pairs"]
        if metrics["n_valid_pairs"] is not None
        else metrics["n_bouts"]
    )
    if n_for_gate < min_bouts:
        return None, None
    vmi = metrics["vmi"]
    p = metrics["p"]
    if vmi is None or p is None:
        return None, None
    if p >= alpha:
        return None, None
    direction = "excit" if vmi > 0 else "suppress"
    return direction, metrics


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

    if int(direction_block["max_run"] or 0) < min_run:
        return None
    peak_z = _safe_float(direction_block["peak_z"])
    if peak_z is None or abs(peak_z) < z_threshold:
        return None
    return _extract_run_metrics(direction_block)


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

    metrics = _extract_categorical_metrics(payload)
    pz = metrics["peak_abs_z"]
    if pz is None or pz < z_threshold:
        return None
    return metrics


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

    metrics = _extract_spatial_metrics(payload)
    info = metrics["info_rate_bps"]
    if info is None or info < info_threshold:
        return None
    return metrics


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


# Administrative keys that `flag_one_cluster` attaches to each record
# alongside the metric scalars. Consumers that only want the metric
# values filter these out via `_FLAG_ADMIN_KEYS`.
_FLAG_ADMIN_KEYS = frozenset(
    {"tested", "significant", "role", "property",
     "cat_feat", "feature", "offset"}
)


# Per-cluster significance builder


def flag_one_cluster(
    cluster_data: dict,
    *,
    z_threshold: float,
    min_consecutive_bins: int,
    vmi_alpha: float,
    vmi_min_bouts: int,
    spatial_info_bps_threshold: float,
) -> dict[str, dict]:
    """
    Description
    -----------
    Walk one cluster's `triage_stats` block and emit a per-modality-key
    record for every (emitter, direction, property, feature, offset)
    tuple that was *eligible* for evaluation. Each record carries a
    `significant` bool plus the metric scalars present in the underlying
    block, so the caller can ask both "was this modality tested?" (key
    present) and "did the unit fire on it?" (`significant`).

    This is the single source of truth for the threshold rules; the
    cross-session aggregator (`aggregate_units_across_conditions`)
    consumes it to accumulate per-session evidence per unit.

    Modality keys follow the convention:

      vmi_<role>_<direction>
      usv_peth_<role>_<direction>
      usv_property_<role>_<prop>_<direction>
      usv_category_<role>_<cat_feat>
      usv_category_peth_<role>_<cat_feat>_<direction>
      behavioral_<offset>_<feature>_<direction>
      spatial_<offset>_<feature>

    For VMI, `direction` is inferred from the sign of `vmi`: positive ->
    `excit`, negative -> `suppress`. If `vmi` is non-finite the block is
    omitted (no direction can be assigned).

    Parameters
    ----------
    cluster_data (dict)
        Loaded per-cluster pkl payload. The top-level `triage_stats`
        sub-dict is the only field consumed; absence of `triage_stats`
        yields an empty result.
    z_threshold (float)
        Magnitude threshold on `peak_z` for run-analysis and categorical
        modalities. See `_flag_runs` / `_flag_categorical`.
    min_consecutive_bins (int)
        Minimum run length for run-analysis modalities. See `_flag_runs`.
    vmi_alpha (float)
        Wilcoxon p-value cutoff for VMI significance. See `_flag_vmi`.
    vmi_min_bouts (int)
        Minimum bout count required to consider VMI meaningful. See
        `_flag_vmi`.
    spatial_info_bps_threshold (float)
        Skaggs information-rate cutoff (bits/spike) for spatial
        modalities. See `_flag_spatial`.

    Returns
    -------
    records (dict[str, dict])
        Mapping `modality_key -> {"significant": bool, "tested": True,
        ...metric scalars}`. A key is present iff the corresponding
        block existed in `triage_stats` (i.e., the modality was eligible
        for evaluation). When `cluster_data["triage_stats"]` is missing
        or not a dict the function returns an empty dict.
    """

    ts = cluster_data.get("triage_stats")
    if not isinstance(ts, dict):
        return {}

    emitter_roles = _emitter_role_map(cluster_data)
    records: dict[str, dict] = {}

    # VMI: triage_stats["vmi"][emitter] -> payload (one per emitter).
    for emitter, payload in (ts.get("vmi") or {}).items():
        if not isinstance(payload, dict):
            continue
        role = (
            payload.get("role")
            or emitter_roles.get(emitter)
            or emitter
        )
        metrics = _extract_vmi_metrics(payload)
        vmi = metrics["vmi"]
        if vmi is None or not math.isfinite(vmi):
            continue
        inferred_dir = "excit" if vmi > 0 else "suppress"
        flagged_dir, _ = _flag_vmi(payload, alpha=vmi_alpha, min_bouts=vmi_min_bouts)
        significant = flagged_dir is not None
        key = f"vmi_{role}_{inferred_dir}"
        records[key] = {
            "tested": True,
            "significant": significant,
            "role": role,
            **metrics,
        }

    # usv_peth: per emitter, per direction (`excit` / `suppress`).
    for emitter, payload in (ts.get("usv_peth") or {}).items():
        if not isinstance(payload, dict):
            continue
        role = emitter_roles.get(emitter, emitter)
        ramp_index = _safe_float(payload.get("ramp_index"))
        for direction in ("excit", "suppress"):
            block = payload.get(direction)
            if not isinstance(block, dict):
                continue
            metrics = _extract_run_metrics(block)
            metrics["ramp_index"] = ramp_index
            significant = _flag_runs(
                block, z_threshold=z_threshold, min_run=min_consecutive_bins
            ) is not None
            key = f"usv_peth_{role}_{direction}"
            records[key] = {
                "tested": True,
                "significant": significant,
                "role": role,
                **metrics,
            }

    # usv_property_tuning: per emitter, per property, per direction.
    for emitter, props in (ts.get("usv_property_tuning") or {}).items():
        if not isinstance(props, dict):
            continue
        role = emitter_roles.get(emitter, emitter)
        for prop, payload in props.items():
            if not isinstance(payload, dict):
                continue
            selectivity = _safe_float(payload.get("selectivity"))
            monotonicity = _safe_float(payload.get("monotonicity"))
            for direction in ("excit", "suppress"):
                block = payload.get(direction)
                if not isinstance(block, dict):
                    continue
                metrics = _extract_run_metrics(block)
                metrics["selectivity"] = selectivity
                metrics["monotonicity"] = monotonicity
                significant = _flag_runs(
                    block, z_threshold=z_threshold, min_run=min_consecutive_bins
                ) is not None
                key = f"usv_property_{role}_{prop}_{direction}"
                records[key] = {
                    "tested": True,
                    "significant": significant,
                    "role": role,
                    "property": prop,
                    **metrics,
                }

    # usv_category_tuning: per emitter, per cat_feat (categorical axis).
    for emitter, cats in (ts.get("usv_category_tuning") or {}).items():
        if not isinstance(cats, dict):
            continue
        role = emitter_roles.get(emitter, emitter)
        for cat_feat, payload in cats.items():
            if not isinstance(payload, dict):
                continue
            metrics = _extract_categorical_metrics(payload)
            significant = _flag_categorical(payload, z_threshold=z_threshold) is not None
            key = f"usv_category_{role}_{cat_feat}"
            records[key] = {
                "tested": True,
                "significant": significant,
                "role": role,
                "cat_feat": cat_feat,
                **metrics,
            }

    # usv_category_peth: per emitter, per cat_feat, best_{excit,suppress}.
    for emitter, cats in (ts.get("usv_category_peth") or {}).items():
        if not isinstance(cats, dict):
            continue
        role = emitter_roles.get(emitter, emitter)
        for cat_feat, payload in cats.items():
            if not isinstance(payload, dict):
                continue
            _best_cat = payload.get("best_cat")
            best_cat = int(_best_cat) if _best_cat is not None else -1
            best_t = _safe_float(payload.get("best_t"))
            for direction_name, key_tag in (
                ("best_excit", "excit"),
                ("best_suppress", "suppress"),
            ):
                block = payload.get(direction_name)
                if not isinstance(block, dict):
                    continue
                metrics = _extract_run_metrics(block)
                metrics["best_cat"] = best_cat
                metrics["best_t"] = best_t
                significant = _flag_runs(
                    block, z_threshold=z_threshold, min_run=min_consecutive_bins
                ) is not None
                key = f"usv_category_peth_{role}_{cat_feat}_{key_tag}"
                records[key] = {
                    "tested": True,
                    "significant": significant,
                    "role": role,
                    "cat_feat": cat_feat,
                    **metrics,
                }

    # behavioral: per offset, per feature, per direction.
    for offset_key, feats in (ts.get("behavioral") or {}).items():
        if not isinstance(feats, dict):
            continue
        for feat_key, payload in feats.items():
            if not isinstance(payload, dict):
                continue
            selectivity = _safe_float(payload.get("selectivity"))
            monotonicity = _safe_float(payload.get("monotonicity"))
            is_circular = bool(payload.get("is_circular", False))
            for direction in ("excit", "suppress"):
                block = payload.get(direction)
                if not isinstance(block, dict):
                    continue
                metrics = _extract_run_metrics(block)
                metrics["selectivity"] = selectivity
                metrics["monotonicity"] = monotonicity
                metrics["is_circular"] = is_circular
                significant = _flag_runs(
                    block, z_threshold=z_threshold, min_run=min_consecutive_bins
                ) is not None
                key = f"behavioral_{offset_key}_{feat_key}_{direction}"
                records[key] = {
                    "tested": True,
                    "significant": significant,
                    "feature": feat_key,
                    "offset": offset_key,
                    **metrics,
                }

    # spatial: per offset, per feature.
    for offset_key, feats in (ts.get("spatial") or {}).items():
        if not isinstance(feats, dict):
            continue
        for feat_key, payload in feats.items():
            if not isinstance(payload, dict):
                continue
            metrics = _extract_spatial_metrics(payload)
            significant = _flag_spatial(
                payload, info_threshold=spatial_info_bps_threshold
            ) is not None
            key = f"spatial_{offset_key}_{feat_key}"
            records[key] = {
                "tested": True,
                "significant": significant,
                "feature": feat_key,
                "offset": offset_key,
                **metrics,
            }

    return records


# Data-location defaults come from `analyses_settings.json` under `data_roots`,
# resolved to the host OS via `configure_path` (see `resolve_data_root`), so
# they are user-editable + OS-portable rather than hard-coded.
_DEFAULT_CATALOG_PATH = str(resolve_data_root("catalog_path"))
_DEFAULT_AGGREGATOR_OUT_DIR = str(resolve_data_root("aggregator_out_dir"))
_DEFAULT_DATA_ROOT = str(resolve_data_root("data_root"))


def _load_default_thresholds() -> dict:
    """
    Description
    -----------
    Read the `detect_interesting_tuning_neurons` block from the package
    `analyses_settings.json` so the aggregator and per-session detector
    share defaults. The caller may override any field via kwargs.

    Returns
    -------
    cfg (dict)
        Dict with keys `z_threshold`, `min_consecutive_bins`,
        `vmi_alpha`, `vmi_min_bouts`, `spatial_info_bps_threshold`.
    """

    settings_path = (
        pathlib.Path(__file__).parent.parent
        / "_parameter_settings"
        / "analyses_settings.json"
    )
    with settings_path.open() as fh:
        cfg = json.load(fh)
    return cfg["detect_interesting_tuning_neurons"]


def _parse_unit_id(unit_id: str) -> tuple[int, int, int, str]:
    """
    Description
    -----------
    Parse a `unit_id` of the form `imec<i>_cl<NNNN>_ch<NNN>_<label>`
    (e.g. `imec0_cl0007_ch207_good`) into its four components. The same
    layout is used by Kilosort pkl filenames and by the
    `unit_catalog.csv` `unit_id` column.

    Parameters
    ----------
    unit_id (str)
        Cluster identifier string with the four `_`-separated tokens.

    Returns
    -------
    imec (int)
        Probe index parsed from `imec<i>`.
    cluster_num (int)
        Kilosort cluster number parsed from `cl<NNNN>`.
    peak_channel (int)
        Peak channel parsed from `ch<NNN>`.
    kslabel (str)
        Curation label (`good` / `mua`).

    Raises
    ------
    ValueError
        If `unit_id` does not have exactly four underscore-separated
        tokens, or if the numeric prefixes are malformed.
    """

    parts = unit_id.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"unit_id {unit_id!r} does not have 4 underscore-separated tokens"
        )
    imec_tok, cl_tok, ch_tok, kslabel = parts
    imec = int(imec_tok.removeprefix("imec"))
    cluster_num = int(cl_tok.removeprefix("cl"))
    peak_channel = int(ch_tok.removeprefix("ch"))
    return imec, cluster_num, peak_channel, kslabel


def _aggregate_modality_stats(mod_key: str, per_session: list[dict]) -> dict:
    """
    Description
    -----------
    Compute cross-session summary statistics for one modality's per-
    session evidence list. Picks the right headline scalars depending
    on the modality kind, inferred from the `mod_key` prefix:

      * `vmi_*`              -> max_abs_vmi, min_pvalue
      * `spatial_*`          -> max_info_rate_bps, median_info_rate_bps
      * `usv_category_*`     -> max_peak_abs_z, median_peak_abs_z
        (only when NOT a `usv_category_peth_*` key)
      * everything else      -> max_abs_peak_z, median_peak_z

    Missing / non-finite per-session values are skipped; if no usable
    values remain, the relevant aggregate field is `None`.

    Parameters
    ----------
    mod_key (str)
        The modality key emitted by `flag_one_cluster`. Its prefix
        chooses the aggregation rule.
    per_session (list[dict])
        Per-session evidence list — each entry carries the modality's
        metric scalars plus a `session` and `significant` field.

    Returns
    -------
    agg (dict)
        Modality-appropriate summary scalars. Always returns a dict
        (possibly with `None` values), never raises.
    """

    def _vals(field: str) -> list[float]:
        """Pull finite floats for one metric across per_session."""
        out = []
        for e in per_session:
            v = e.get(field)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out.append(f)
        return out

    if mod_key.startswith("vmi_"):
        vmis = _vals("vmi")
        ps = _vals("p")
        return {
            "max_abs_vmi": max((abs(v) for v in vmis), default=None),
            "min_pvalue": min(ps, default=None),
        }
    if mod_key.startswith("spatial_"):
        infos = _vals("info_rate_bps")
        return {
            "max_info_rate_bps": max(infos, default=None),
            "median_info_rate_bps": (
                float(np.median(infos)) if infos else None
            ),
        }
    if (
        mod_key.startswith("usv_category_")
        and not mod_key.startswith("usv_category_peth_")
    ):
        pzs = _vals("peak_abs_z")
        return {
            "max_peak_abs_z": max(pzs, default=None),
            "median_peak_abs_z": (
                float(np.median(pzs)) if pzs else None
            ),
        }
    # run-based: usv_peth_, usv_property_, usv_category_peth_, behavioral_
    pzs = _vals("peak_z")
    return {
        "max_abs_peak_z": (
            max((abs(p) for p in pzs), default=None)
        ),
        "median_peak_z": (
            float(np.median(pzs)) if pzs else None
        ),
    }


def aggregate_units_across_conditions(
    condition_to_session_list: dict[str, str | pathlib.Path],
    catalog_path: str | pathlib.Path = _DEFAULT_CATALOG_PATH,
    out_dir: str | pathlib.Path = _DEFAULT_AGGREGATOR_OUT_DIR,
    data_root: str | pathlib.Path = _DEFAULT_DATA_ROOT,
    *,
    z_threshold: float | None = None,
    min_consecutive_bins: int | None = None,
    vmi_alpha: float | None = None,
    vmi_min_bouts: int | None = None,
    spatial_info_bps_threshold: float | None = None,
    message_output: Callable = print,
) -> pathlib.Path:
    """
    Description
    -----------
    Cross-session / cross-condition unit triage. For each `(condition,
    session)` pair drawn from the supplied condition session lists, the
    per-cluster `*_tuning_curves_data.pkl` files under
    `<data_root>/<session>/ephys/tuning_curves/` are loaded, run through
    `flag_one_cluster`, and joined with `unit_catalog.csv` to enrich
    each cluster with `mouse_id`, `rec_date`, and `brain_area`.

    Because per-day Kilosort sorts are concatenated, the same physical
    unit appears under the same `unit_id` in every same-day session;
    those rows are folded into a single unit record keyed by
    `unit_uid = f"{mouse_id}_{rec_date}_{unit_id}"`, with per-session
    metrics stacked underneath each modality (see the per-modality
    `per_session` list in the output).

    Thresholds default to the values in `analyses_settings.json` under
    `detect_interesting_tuning_neurons`; kwargs override on a per-call
    basis. The output is a pickle written to
    `<out_dir>/unit_triage_<YYYYMMDD_HHMMSS>.pkl`.

    Output schema (the pickled object is a single dict):

      {
        "generated_at": ISO-8601 string,
        "thresholds_used": {z_threshold, min_consecutive_bins,
                            vmi_alpha, vmi_min_bouts,
                            spatial_info_bps_threshold},
        "catalog_path": str,
        "data_root": str,
        "conditions_included": {<cond>: [<session>, ...], ...},
        "sessions_skipped":    {<cond>: [<session>, ...], ...},
        "n_units_total": int,
        "n_units_per_condition": {<cond>: int, ...},
        "units": {
          "<unit_uid>": {
            "unit_uid": str, "mouse_id": str, "rec_date": int,
            "imec": int, "cluster_num": int, "peak_channel": int,
            "kslabel": str, "unit_id": str, "anatomy_region": str,
            "conditions": {
              "<cond>": {
                "sessions_tested": [<session>, ...],
                "modalities": {
                  "<modality_key>": {
                    "n_significant": int, "n_tested": int,
                    "consistency": float,
                    "aggregate": { ...modality-specific scalars... },
                    "per_session": [
                      {"session": str, "significant": bool,
                       ...modality-specific metric scalars... },
                      ...
                    ],
                  },
                  ...
                },
              },
              ...
            },
          },
          ...
        },
      }

    Parameters
    ----------
    condition_to_session_list (dict[str, str | pathlib.Path])
        Mapping from condition label to a `.txt` file with one session
        path (or session timestamp) per line. The session basename is
        used to locate `<data_root>/<basename>/ephys/tuning_curves/`.
    catalog_path (str | pathlib.Path)
        Path to `unit_catalog.csv`. Used to derive `mouse_id` from
        `rec_date` and to read `brain_area` per cluster.
    out_dir (str | pathlib.Path)
        Directory where the timestamped pickle is written. Created if
        absent.
    data_root (str | pathlib.Path)
        Root directory under which per-session `<session>/ephys/...`
        layouts live.
    z_threshold (float | None)
        Override for the `peak_z` magnitude threshold; `None` uses the
        value from `analyses_settings.json`.
    min_consecutive_bins (int | None)
        Override for the consecutive-bin run threshold.
    vmi_alpha (float | None)
        Override for the VMI Wilcoxon p-value cutoff.
    vmi_min_bouts (int | None)
        Override for the VMI minimum-bout requirement.
    spatial_info_bps_threshold (float | None)
        Override for the Skaggs info-rate (bits/spike) cutoff.
    message_output (Callable)
        Logger; defaults to `print`. Receives one line per skipped
        session and a final summary line on success.

    Returns
    -------
    out_path (pathlib.Path)
        Path of the written `unit_triage_<TS>.pkl`.

    Raises
    ------
    FileNotFoundError
        If `catalog_path` or `analyses_settings.json` is missing, or if
        any condition `.txt` does not exist.
    ValueError
        If `unit_catalog.csv` maps any `rec_date` to more than one
        `mouse_id` (the one-date-one-mouse invariant the per-session
        `mouse_id` lookup relies on).
    KeyError
        If any pkl on disk has no catalog row for its
        `(mouse_id, rec_date, unit_id)` tuple. The catalog is the
        authoritative scope, so an orphan pkl (stale catalog or stray
        file) is a hard error; all orphans are collected and reported
        together at the end rather than aborting on the first.
    """

    # 1. Resolve thresholds (kwargs > analyses_settings.json defaults).
    defaults = _load_default_thresholds()
    thresholds = {
        "z_threshold": (
            z_threshold
            if z_threshold is not None
            else defaults["z_threshold"]
        ),
        "min_consecutive_bins": (
            min_consecutive_bins
            if min_consecutive_bins is not None
            else defaults["min_consecutive_bins"]
        ),
        "vmi_alpha": (
            vmi_alpha
            if vmi_alpha is not None
            else defaults["vmi_alpha"]
        ),
        "vmi_min_bouts": (
            vmi_min_bouts
            if vmi_min_bouts is not None
            else defaults["vmi_min_bouts"]
        ),
        "spatial_info_bps_threshold": (
            spatial_info_bps_threshold
            if spatial_info_bps_threshold is not None
            else defaults["spatial_info_bps_threshold"]
        ),
    }

    # 2. Load the catalog and build lookup structures.
    catalog_path = pathlib.Path(catalog_path)
    catalog = pd.read_csv(
        catalog_path,
        usecols=["mouse_id", "rec_date", "unit_id", "brain_area"],
    )
    catalog["mouse_id"] = catalog["mouse_id"].astype(str)
    catalog["rec_date"] = catalog["rec_date"].astype(int)
    catalog["unit_id"] = catalog["unit_id"].astype(str)
    catalog["brain_area"] = catalog["brain_area"].astype(str)
    # (mouse_id, rec_date, unit_id) -> brain_area
    catalog_lookup: dict[tuple[str, int, str], str] = {
        (row.mouse_id, row.rec_date, row.unit_id): row.brain_area
        for row in catalog.itertuples(index=False)
    }
    # rec_date -> mouse_id (each date maps to exactly one mouse). Enforce the
    # invariant rather than silently taking the first: a date mapping to more
    # than one mouse would mis-attribute every unit recorded that day.
    date_mouse_counts = catalog.groupby("rec_date")["mouse_id"].nunique()
    ambiguous_dates = date_mouse_counts[date_mouse_counts > 1]
    if not ambiguous_dates.empty:
        details = "; ".join(
            f"{int(d)} -> "
            f"{sorted(catalog.loc[catalog['rec_date'] == d, 'mouse_id'].unique())}"
            for d in ambiguous_dates.index
        )
        raise ValueError(
            "unit_catalog.csv maps a rec_date to multiple mouse_ids "
            f"(expected one-date-one-mouse): {details}"
        )
    date_to_mouse: dict[int, str] = (
        catalog.groupby("rec_date")["mouse_id"].first().to_dict()
    )

    # 3. Read condition session lists.
    condition_sessions: dict[str, list[str]] = {}
    for cond, lst_path in condition_to_session_list.items():
        lst_path = pathlib.Path(lst_path)
        if not lst_path.exists():
            raise FileNotFoundError(
                f"session list for condition {cond!r} not found: {lst_path}"
            )
        sessions = []
        for line in lst_path.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                sessions.append(pathlib.Path(stripped).name)
        condition_sessions[cond] = sessions

    # 4. Iterate condition -> session -> pkl; build unit records.
    data_root = pathlib.Path(data_root)
    units: dict[str, dict] = {}
    sessions_skipped: dict[str, list[str]] = {c: [] for c in condition_sessions}
    orphan_pkls: list[tuple[str, int, str, pathlib.Path]] = []
    n_pkls_processed = 0

    for cond, sessions in condition_sessions.items():
        for sess in sessions:
            try:
                rec_date = int(sess[:8])
            except ValueError:
                message_output(
                    f"  aggregator: skipping {sess} — cannot parse rec_date prefix"
                )
                sessions_skipped[cond].append(sess)
                continue
            mouse_id = date_to_mouse.get(rec_date)
            if mouse_id is None:
                message_output(
                    f"  aggregator: skipping {sess} — rec_date {rec_date} "
                    "not present in catalog"
                )
                sessions_skipped[cond].append(sess)
                continue
            tuning_dir = data_root / sess / "ephys" / "tuning_curves"
            if not tuning_dir.exists():
                message_output(
                    f"  aggregator: skipping {sess} — no tuning_curves dir"
                )
                sessions_skipped[cond].append(sess)
                continue
            pkls = sorted(tuning_dir.glob("*_tuning_curves_data.pkl"))
            if not pkls:
                message_output(
                    f"  aggregator: skipping {sess} — no tuning pkls"
                )
                sessions_skipped[cond].append(sess)
                continue

            for pkl in pkls:
                unit_id = pkl.stem.replace("_tuning_curves_data", "")
                key = (mouse_id, rec_date, unit_id)
                if key not in catalog_lookup:
                    # Collect every orphan (pkl with no catalog row) and fail
                    # at the end with the full list, rather than aborting the
                    # whole multi-condition run on the first one.
                    orphan_pkls.append((mouse_id, rec_date, unit_id, pkl))
                    continue
                anatomy_region = catalog_lookup[key]

                with pkl.open("rb") as fh:
                    cluster_data = pickle.load(fh)
                if not isinstance(cluster_data.get("triage_stats"), dict):
                    # No triage to draw on; treat as not-tested for any modality.
                    # The unit slot is still created so its presence is recorded.
                    records = {}
                else:
                    records = flag_one_cluster(cluster_data, **thresholds)

                unit_uid = f"{mouse_id}_{rec_date}_{unit_id}"
                if unit_uid not in units:
                    imec, cluster_num, peak_channel, kslabel = _parse_unit_id(
                        unit_id
                    )
                    units[unit_uid] = {
                        "unit_uid": unit_uid,
                        "mouse_id": mouse_id,
                        "rec_date": rec_date,
                        "imec": imec,
                        "cluster_num": cluster_num,
                        "peak_channel": peak_channel,
                        "kslabel": kslabel,
                        "unit_id": unit_id,
                        "anatomy_region": anatomy_region,
                        "conditions": {},
                    }
                cond_block = units[unit_uid]["conditions"].setdefault(
                    cond, {"sessions_tested": [], "modalities": {}}
                )
                if sess not in cond_block["sessions_tested"]:
                    cond_block["sessions_tested"].append(sess)

                for mod_key, rec in records.items():
                    mod_block = cond_block["modalities"].setdefault(
                        mod_key, {"per_session": []}
                    )
                    entry: dict = {"session": sess}
                    for k, v in rec.items():
                        if k == "tested":
                            continue
                        entry[k] = v
                    mod_block["per_session"].append(entry)

                n_pkls_processed += 1

    if orphan_pkls:
        details = "\n".join(
            f"  - {pkl} (mouse_id={mid!r}, rec_date={rd}, unit_id={uid!r})"
            for mid, rd, uid, pkl in orphan_pkls
        )
        raise KeyError(
            f"{len(orphan_pkls)} pkl(s) have no catalog row (refusing to "
            "silently drop units unknown to the catalog); the catalog is the "
            "authoritative scope, so a stale catalog or stray pkl must be "
            f"resolved:\n{details}"
        )

    # 5. Compute per-modality aggregates across the per_session lists.
    for unit in units.values():
        for cond_block in unit["conditions"].values():
            for mod_block in cond_block["modalities"].values():
                ps = mod_block["per_session"]
                n_tested = len(ps)
                n_significant = sum(1 for e in ps if e["significant"])
                mod_block["n_tested"] = n_tested
                mod_block["n_significant"] = n_significant
                mod_block["consistency"] = (
                    n_significant / n_tested if n_tested else 0.0
                )
        # Sort sessions_tested for stable output ordering.
        for cond_block in unit["conditions"].values():
            cond_block["sessions_tested"] = sorted(
                cond_block["sessions_tested"]
            )

    # _aggregate_modality_stats is keyed by mod_key, so do this after the
    # per_session list is finalised (sorted, complete) for determinism.
    for unit in units.values():
        for cond_block in unit["conditions"].values():
            for mod_key, mod_block in cond_block["modalities"].items():
                mod_block["per_session"] = sorted(
                    mod_block["per_session"], key=lambda e: e["session"]
                )
                mod_block["aggregate"] = _aggregate_modality_stats(
                    mod_key, mod_block["per_session"]
                )

    # 6. Assemble output dict.
    now = datetime.now()
    n_units_per_condition = {
        cond: sum(1 for u in units.values() if cond in u["conditions"])
        for cond in condition_sessions
    }
    out: dict = {
        "generated_at": now.isoformat(timespec="seconds"),
        "thresholds_used": thresholds,
        "catalog_path": str(catalog_path),
        "data_root": str(data_root),
        "conditions_included": dict(condition_sessions),
        "sessions_skipped": sessions_skipped,
        "n_units_total": len(units),
        "n_units_per_condition": n_units_per_condition,
        "units": dict(sorted(units.items())),
    }

    # 7. Write pickle.
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"unit_triage_{timestamp}.pkl"
    with atomic_output_path(out_path) as tmp_path, tmp_path.open("wb") as fh:
        pickle.dump(out, fh)

    message_output(
        f"  aggregator: {len(units)} unique unit(s) across "
        f"{n_pkls_processed} pkl(s) "
        f"({sum(len(s) for s in condition_sessions.values())} session(s); "
        f"{sum(len(s) for s in sessions_skipped.values())} skipped). "
        f"Wrote {out_path}"
    )
    return out_path
