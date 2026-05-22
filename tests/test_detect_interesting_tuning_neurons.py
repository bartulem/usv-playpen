"""
@author: bartulem
Tests for `usv_playpen.analyses.detect_interesting_tuning_neurons`.

Two surfaces are covered:

  1. `flag_one_cluster` — the single source of truth for per-modality
     significance gating. Synthetic `triage_stats` payloads exercise
     every modality path (VMI, USV PETH run, USV property tuning, USV
     category tuning, USV category PETH, behavioral, spatial) for both
     "above threshold" and "below threshold" cases, plus the
     administrative bookkeeping (`tested`, `significant`, role/feature
     discriminators) the cross-session aggregator depends on.

  2. `aggregate_units_across_conditions` — the cross-session /
     cross-condition roll-up. Tests build a tmp_path session tree
     containing per-cluster pkls and a minimal `unit_catalog.csv`, then
     drive the aggregator and inspect the pickled output. Coverage:
     same-day duplicate units collapsed, same unit appearing under
     multiple conditions, modalities absent from a session's pkl not
     counted in `n_tested`, sessions with missing `tuning_curves`
     directories surfaced under `sessions_skipped`, `consistency =
     n_significant / n_tested`, and the orphan-pkl `KeyError` guard
     that protects catalog authority.
"""

from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any

import pytest

from usv_playpen.analyses.detect_interesting_tuning_neurons import (
    _aggregate_modality_stats,
    _parse_unit_id,
    aggregate_units_across_conditions,
    flag_one_cluster,
)


# Shared synthetic-payload helpers


def _vmi_payload(
    *,
    vmi: float,
    p: float,
    n_bouts: int,
    role: str = "self",
) -> dict:
    """
    Description
    -----------
    Build a minimal `triage_stats["vmi"][emitter]` payload with the
    fields `flag_one_cluster` reads.

    Parameters
    ----------
    vmi (float)
        Vocal modulation index value. Sign chooses inferred direction
        (`excit` for vmi > 0, `suppress` for vmi < 0).
    p (float)
        Wilcoxon two-sided p-value associated with `vmi`.
    n_bouts (int)
        Bout count used for the `vmi_min_bouts` gate.
    role (str)
        Emitter role string; placed both on the payload and used as the
        emitter key so the role propagates into the modality key.

    Returns
    -------
    payload (dict)
        The VMI block.
    """

    return {
        "vmi": vmi,
        "wilcoxon_pvalue": p,
        "n_bouts": n_bouts,
        "fr_baseline": 1.0,
        "fr_usv": 5.0,
        "role": role,
    }


def _run_direction_block(*, max_run: int, peak_z: float, n_bins: int = 20) -> dict:
    """
    Description
    -----------
    Build a minimal `excit` / `suppress` direction sub-dict for a
    run-analysis block (USV PETH, USV property, USV category PETH,
    behavioral). Carries just enough fields that
    `_extract_run_metrics` can record metrics and the gate can decide.

    Parameters
    ----------
    max_run (int)
        Longest consecutive-bin run length.
    peak_z (float)
        Peak Z-score within the run.
    n_bins (int)
        Total number of bins in the underlying curve.

    Returns
    -------
    block (dict)
        A direction sub-dict.
    """

    return {
        "max_run": max_run,
        "peak_z": peak_z,
        "n_bins": n_bins,
        "run_t_start": 0.0,
        "run_t_end": 0.2,
        "peak_t": 0.1,
        "peak_idx": 5,
        "run_start_idx": 3,
        "run_end_idx": 8,
        "peak_bin_value": 7.5,
    }


def _spatial_payload(*, info_rate_bps: float) -> dict:
    """
    Description
    -----------
    Build a minimal `triage_stats["spatial"][offset][feat]` payload.

    Parameters
    ----------
    info_rate_bps (float)
        Skaggs information rate in bits/spike — the only gated field.

    Returns
    -------
    payload (dict)
        The spatial block.
    """

    return {
        "info_rate_bps": info_rate_bps,
        "sparsity": 0.5,
        "coherence": 0.7,
        "peak_rate_sps": 8.0,
        "peak_row": 10,
        "peak_col": 15,
    }


def _categorical_payload(*, peak_abs_z: float) -> dict:
    """
    Description
    -----------
    Build a minimal `triage_stats["usv_category_tuning"][emitter][cat_feat]`
    payload.

    Parameters
    ----------
    peak_abs_z (float)
        Magnitude of the peak Z across categories — the only gated
        field.

    Returns
    -------
    payload (dict)
        The categorical block.
    """

    return {
        "peak_abs_z": peak_abs_z,
        "peak_signed_z": peak_abs_z,
        "best_cat": 2,
        "n_sig_categories": 1,
        "selectivity": 0.6,
    }


def _full_triage_stats(
    *,
    vmi_significant: bool = True,
    peth_significant: bool = True,
    property_significant: bool = True,
    cat_significant: bool = True,
    cat_peth_significant: bool = True,
    behavioral_significant: bool = True,
    spatial_significant: bool = True,
) -> dict:
    """
    Description
    -----------
    Assemble a complete `triage_stats` block covering every modality
    `flag_one_cluster` walks. Each modality can be toggled between an
    "above threshold" and a "below threshold" payload so individual
    tests can dial in the verdict without rebuilding the whole block.

    Default thresholds used by callers (and matching the package
    `analyses_settings.json`):
      z_threshold = 3.0
      min_consecutive_bins = 3
      vmi_alpha = 0.01
      vmi_min_bouts = 10
      spatial_info_bps_threshold = 0.5

    Parameters
    ----------
    vmi_significant (bool)
    peth_significant (bool)
    property_significant (bool)
    cat_significant (bool)
    cat_peth_significant (bool)
    behavioral_significant (bool)
    spatial_significant (bool)
        Per-modality verdict switches. `True` builds a payload that
        clears the default thresholds; `False` builds one that fails.

    Returns
    -------
    triage_stats (dict)
        Full mock `triage_stats` block.
    """

    sig_run = _run_direction_block(max_run=5, peak_z=4.5)
    nonsig_run = _run_direction_block(max_run=1, peak_z=0.5)
    triage_stats: dict[str, Any] = {
        "vmi": {
            "self": _vmi_payload(
                vmi=0.7 if vmi_significant else 0.0001,
                p=1e-4 if vmi_significant else 0.5,
                n_bouts=50,
            ),
        },
        "usv_peth": {
            "self": {
                "role": "self",
                "ramp_index": 0.4,
                "excit": sig_run if peth_significant else nonsig_run,
                "suppress": nonsig_run,
            },
        },
        "usv_property_tuning": {
            "self": {
                "duration": {
                    "role": "self",
                    "selectivity": 0.3,
                    "monotonicity": 0.2,
                    "excit": sig_run if property_significant else nonsig_run,
                    "suppress": nonsig_run,
                },
            },
        },
        "usv_category_tuning": {
            "self": {
                "call_category": {
                    "role": "self",
                    **_categorical_payload(
                        peak_abs_z=4.0 if cat_significant else 0.5
                    ),
                },
            },
        },
        "usv_category_peth": {
            "self": {
                "call_category": {
                    "role": "self",
                    "best_cat": 2,
                    "best_t": 0.1,
                    "best_excit": sig_run if cat_peth_significant else nonsig_run,
                    "best_suppress": nonsig_run,
                },
            },
        },
        "behavioral": {
            "0ms": {
                "speed": {
                    "selectivity": 0.4,
                    "monotonicity": 0.5,
                    "is_circular": False,
                    "excit": sig_run if behavioral_significant else nonsig_run,
                    "suppress": nonsig_run,
                },
            },
        },
        "spatial": {
            "0ms": {
                "self_xy": _spatial_payload(
                    info_rate_bps=1.2 if spatial_significant else 0.1
                ),
            },
        },
    }
    return triage_stats


DEFAULT_THRESHOLDS = {
    "z_threshold": 3.0,
    "min_consecutive_bins": 3,
    "vmi_alpha": 0.01,
    "vmi_min_bouts": 10,
    "spatial_info_bps_threshold": 0.5,
}


# flag_one_cluster — per-modality gating


def test_flag_one_cluster_returns_empty_when_no_triage_stats():
    """
    Description
    -----------
    A pkl that predates triage compute (no `triage_stats` block, or a
    non-dict value there) must yield an empty record dict so the
    per-session detector and the aggregator can both treat the cluster
    as "no modalities tested" without crashing.
    """

    assert flag_one_cluster({}, **DEFAULT_THRESHOLDS) == {}
    assert flag_one_cluster({"triage_stats": None}, **DEFAULT_THRESHOLDS) == {}
    assert flag_one_cluster(
        {"triage_stats": "not a dict"}, **DEFAULT_THRESHOLDS
    ) == {}


def test_flag_one_cluster_emits_every_modality_key_when_all_significant():
    """
    Description
    -----------
    With a fully-populated `triage_stats` block (every modality clears
    the default thresholds), the record dict must contain exactly the
    seven expected modality keys and mark each one significant. The
    keys check the role/feature discriminator wiring as well as the
    modality prefix convention.
    """

    cluster_data = {"triage_stats": _full_triage_stats()}
    records = flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS)

    expected_keys = {
        "vmi_self_excit",
        "usv_peth_self_excit",
        "usv_property_self_duration_excit",
        "usv_category_self_call_category",
        "usv_category_peth_self_call_category_excit",
        "behavioral_0ms_speed_excit",
        "spatial_0ms_self_xy",
    }
    assert expected_keys <= set(records.keys())
    for key in expected_keys:
        assert records[key]["tested"] is True
        assert records[key]["significant"] is True


def test_flag_one_cluster_marks_each_modality_not_significant_when_below_threshold():
    """
    Description
    -----------
    Toggling each modality to its "below threshold" payload must flip
    `significant` to `False` while leaving `tested=True` (the block was
    still present and evaluated). All other modalities remain
    significant — verifies the gating is independent across paths.
    """

    cluster_data = {
        "triage_stats": _full_triage_stats(
            vmi_significant=False,
            peth_significant=False,
            property_significant=False,
            cat_significant=False,
            cat_peth_significant=False,
            behavioral_significant=False,
            spatial_significant=False,
        )
    }
    records = flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS)
    for key, rec in records.items():
        assert rec["tested"] is True
        assert rec["significant"] is False, f"{key} should not be significant"


def test_flag_one_cluster_vmi_direction_follows_sign():
    """
    Description
    -----------
    The VMI direction is inferred from the sign of `vmi` (the payload
    has no `direction` field of its own). Positive maps to `excit`,
    negative maps to `suppress`. Both must be marked significant when
    the p-value and bout count clear their gates.
    """

    cluster_data = {
        "triage_stats": {
            "vmi": {
                "self": _vmi_payload(vmi=-0.4, p=1e-3, n_bouts=50),
            },
        }
    }
    records = flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS)
    assert "vmi_self_suppress" in records
    assert records["vmi_self_suppress"]["significant"] is True


def test_flag_one_cluster_vmi_excluded_when_n_bouts_below_threshold():
    """
    Description
    -----------
    VMI must be `tested=True, significant=False` when `n_bouts` is
    below `vmi_min_bouts`, regardless of the p-value. The key must
    still be present so consumers know the block was eligible.
    """

    cluster_data = {
        "triage_stats": {
            "vmi": {
                "self": _vmi_payload(vmi=0.7, p=1e-4, n_bouts=5),
            },
        }
    }
    records = flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS)
    assert records["vmi_self_excit"]["significant"] is False
    assert records["vmi_self_excit"]["tested"] is True


def test_flag_one_cluster_absent_modality_yields_no_record():
    """
    Description
    -----------
    A modality absent from `triage_stats` (e.g. spatial omitted)
    produces no record at all — not even a `tested=False` stub. This
    is the contract the aggregator relies on to compute `n_tested`
    correctly across sessions where some pkls lack some modality
    blocks.
    """

    cluster_data = {
        "triage_stats": {
            "vmi": {
                "self": _vmi_payload(vmi=0.7, p=1e-4, n_bouts=50),
            },
            # no usv_peth, no usv_property_tuning, no behavioral, no spatial...
        }
    }
    records = flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS)
    assert set(records.keys()) == {"vmi_self_excit"}


def test_flag_one_cluster_records_carry_admin_and_metric_fields():
    """
    Description
    -----------
    Each emitted record must carry the administrative bookkeeping
    (`tested`, `significant`, role / property / cat_feat / feature /
    offset discriminator) plus the metric scalars the aggregator
    feeds into `_aggregate_modality_stats`.
    """

    cluster_data = {"triage_stats": _full_triage_stats()}
    records = flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS)

    peth = records["usv_peth_self_excit"]
    assert peth["role"] == "self"
    assert peth["peak_z"] == pytest.approx(4.5)
    assert peth["max_run"] == 5

    prop = records["usv_property_self_duration_excit"]
    assert prop["property"] == "duration"
    assert prop["role"] == "self"

    cat = records["usv_category_self_call_category"]
    assert cat["cat_feat"] == "call_category"
    assert cat["peak_abs_z"] == pytest.approx(4.0)

    behav = records["behavioral_0ms_speed_excit"]
    assert behav["feature"] == "speed"
    assert behav["offset"] == "0ms"

    spatial = records["spatial_0ms_self_xy"]
    assert spatial["feature"] == "self_xy"
    assert spatial["info_rate_bps"] == pytest.approx(1.2)


# Pure-helper unit tests


def test_parse_unit_id_components():
    """
    Description
    -----------
    `_parse_unit_id` returns `(imec, cluster_num, peak_channel, kslabel)`
    parsed from the canonical four-token cluster identifier used by
    both Kilosort pkls and the catalog `unit_id` column.
    """

    imec, cluster_num, peak_channel, kslabel = _parse_unit_id(
        "imec0_cl0007_ch207_good"
    )
    assert imec == 0
    assert cluster_num == 7
    assert peak_channel == 207
    assert kslabel == "good"


def test_parse_unit_id_raises_on_malformed_input():
    """
    Description
    -----------
    Inputs that do not split into exactly four underscore-separated
    tokens must raise `ValueError` so the aggregator fails loud
    rather than producing a unit record with garbage fields.
    """

    with pytest.raises(ValueError):
        _parse_unit_id("imec0_cl0007_good")  # only 3 tokens
    with pytest.raises(ValueError):
        _parse_unit_id("imec0_cl0007_ch207_extra_good")  # 5 tokens


def test_aggregate_modality_stats_dispatch_by_prefix():
    """
    Description
    -----------
    `_aggregate_modality_stats` chooses its scalar set by modality-key
    prefix. This test pins the four code paths:
      * `vmi_*`            -> max_abs_vmi, min_pvalue
      * `spatial_*`        -> max_info_rate_bps, median_info_rate_bps
      * `usv_category_*`   -> max_peak_abs_z, median_peak_abs_z
        (excluding `usv_category_peth_*`)
      * default run-based  -> max_abs_peak_z, median_peak_z
    """

    vmi_agg = _aggregate_modality_stats(
        "vmi_self_excit",
        [{"vmi": 0.5, "p": 1e-3}, {"vmi": -0.7, "p": 5e-4}],
    )
    assert vmi_agg["max_abs_vmi"] == pytest.approx(0.7)
    assert vmi_agg["min_pvalue"] == pytest.approx(5e-4)

    spatial_agg = _aggregate_modality_stats(
        "spatial_0ms_self_xy",
        [{"info_rate_bps": 1.2}, {"info_rate_bps": 0.8}, {"info_rate_bps": 2.0}],
    )
    assert spatial_agg["max_info_rate_bps"] == pytest.approx(2.0)
    assert spatial_agg["median_info_rate_bps"] == pytest.approx(1.2)

    cat_agg = _aggregate_modality_stats(
        "usv_category_self_call_category",
        [{"peak_abs_z": 3.5}, {"peak_abs_z": 5.0}],
    )
    assert cat_agg["max_peak_abs_z"] == pytest.approx(5.0)
    assert cat_agg["median_peak_abs_z"] == pytest.approx(4.25)

    cat_peth_agg = _aggregate_modality_stats(
        "usv_category_peth_self_call_category_excit",
        [{"peak_z": 4.0}, {"peak_z": -5.0}],
    )
    # Falls through to the run-based default because of the `_peth_` check.
    assert cat_peth_agg["max_abs_peak_z"] == pytest.approx(5.0)
    assert cat_peth_agg["median_peak_z"] == pytest.approx(-0.5)

    run_agg = _aggregate_modality_stats(
        "usv_peth_self_excit",
        [{"peak_z": 4.0}, {"peak_z": 6.0}],
    )
    assert run_agg["max_abs_peak_z"] == pytest.approx(6.0)
    assert run_agg["median_peak_z"] == pytest.approx(5.0)


# Aggregator end-to-end tests (synthetic on-disk session tree + catalog)


def _write_cluster_pkl(
    tuning_dir: pathlib.Path,
    unit_id: str,
    triage_stats: dict | None,
) -> pathlib.Path:
    """
    Description
    -----------
    Write one `*_tuning_curves_data.pkl` carrying the supplied
    `triage_stats` block under `tuning_dir`. Mirrors the on-disk
    layout the aggregator scans (one pkl per cluster). A `None`
    triage block writes a pkl without the `triage_stats` key — the
    aggregator must treat that as "tested zero modalities" rather
    than fail.

    Parameters
    ----------
    tuning_dir (pathlib.Path)
        Destination directory, typically `<sess>/ephys/tuning_curves`.
    unit_id (str)
        Cluster identifier in the canonical
        `imec<i>_cl<NNNN>_ch<NNN>_<label>` form.
    triage_stats (dict | None)
        The `triage_stats` block to embed; `None` writes a pkl with no
        triage block at all.

    Returns
    -------
    pkl_path (pathlib.Path)
        Path to the written pickle.
    """

    pkl_path = tuning_dir / f"{unit_id}_tuning_curves_data.pkl"
    payload: dict[str, Any] = {}
    if triage_stats is not None:
        payload["triage_stats"] = triage_stats
    with pkl_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return pkl_path


def _write_catalog(
    catalog_path: pathlib.Path,
    rows: list[tuple[str, int, str, str]],
) -> None:
    """
    Description
    -----------
    Write a minimal `unit_catalog.csv` containing only the four
    columns the aggregator reads via `usecols`: `mouse_id`, `rec_date`,
    `unit_id`, `brain_area`. The catalog is the authoritative scope —
    pkls without a matching row trip the aggregator's hard error.

    Parameters
    ----------
    catalog_path (pathlib.Path)
        Where to write the CSV.
    rows (list[tuple[str, int, str, str]])
        Iterable of `(mouse_id, rec_date, unit_id, brain_area)`.

    Returns
    -------
    None
    """

    lines = ["mouse_id,rec_date,unit_id,brain_area"]
    for mouse_id, rec_date, unit_id, brain_area in rows:
        lines.append(f"{mouse_id},{rec_date},{unit_id},{brain_area}")
    catalog_path.write_text("\n".join(lines) + "\n")


def _build_aggregator_tree(tmp_path: pathlib.Path) -> dict[str, Any]:
    """
    Description
    -----------
    Lay down an end-to-end aggregator fixture on disk:

      <tmp_path>/data/20240101_100000/ephys/tuning_curves/<pkl>     intact_female
      <tmp_path>/data/20240101_120000/ephys/tuning_curves/<pkl>     intact_female
                                                                   (same day -> same unit)
      <tmp_path>/data/20240102_100000/ephys/tuning_curves/<pkl>     mute_female
      <tmp_path>/data/20240103_100000/                              (no tuning_curves dir; intact)
      <tmp_path>/catalog.csv
      <tmp_path>/lists/intact.txt
      <tmp_path>/lists/mute.txt
      <tmp_path>/out/                                                aggregator output dir

    The unit `imec0_cl0001_ch100_good` (mouse `M01`) appears in BOTH
    20240101 sessions — that's the same-day duplicate the aggregator
    must collapse into one unit record while preserving per-session
    evidence. On 20240102 (mouse `M02`) and 20240103 (mouse `M03`)
    the unit is `imec0_cl0002_ch200_good`. The unit on 20240102 also
    appears under both conditions because we list it in both .txt
    files — driving the "same unit, both conditions" path.

    For the 20240101_100000 pkl, the USV PETH modality is set
    significant; in 20240101_120000 it is set NOT significant — so
    `consistency = 1 / 2 = 0.5`. The 20240101_120000 pkl also omits
    the spatial block entirely so we can assert "modality absent ->
    not counted in n_tested" (spatial appears only once -> n_tested=1).

    The 20240103 session is in the intact `.txt` list but has no
    `tuning_curves` directory on disk, so the aggregator must record
    it under `sessions_skipped` and not raise.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory.

    Returns
    -------
    fixture (dict[str, Any])
        Keys: `intact_list`, `mute_list`, `catalog`, `data_root`,
        `out_dir`, plus `unit_uid_day1`, `unit_uid_day2`.
    """

    data_root = tmp_path / "data"
    sess_a = data_root / "20240101_100000" / "ephys" / "tuning_curves"
    sess_b = data_root / "20240101_120000" / "ephys" / "tuning_curves"
    sess_c = data_root / "20240102_100000" / "ephys" / "tuning_curves"
    sess_d_root = data_root / "20240103_100000"  # intentionally empty
    for d in (sess_a, sess_b, sess_c):
        d.mkdir(parents=True)
    sess_d_root.mkdir(parents=True)

    unit_day1 = "imec0_cl0001_ch100_good"
    unit_day2 = "imec0_cl0002_ch200_good"

    # 20240101 session A: USV PETH significant, spatial significant.
    _write_cluster_pkl(
        sess_a,
        unit_day1,
        _full_triage_stats(
            vmi_significant=True,
            peth_significant=True,
            spatial_significant=True,
        ),
    )

    # 20240101 session B: same unit, USV PETH NOT significant, spatial OMITTED.
    triage_b = _full_triage_stats(
        vmi_significant=True,
        peth_significant=False,
        spatial_significant=True,
    )
    triage_b.pop("spatial")
    _write_cluster_pkl(sess_b, unit_day1, triage_b)

    # 20240102 session C: different unit / different mouse, all significant.
    _write_cluster_pkl(sess_c, unit_day2, _full_triage_stats())

    catalog_path = tmp_path / "catalog.csv"
    _write_catalog(
        catalog_path,
        [
            ("M01", 20240101, unit_day1, "PAG"),
            ("M02", 20240102, unit_day2, "MPOA"),
            ("M03", 20240103, unit_day2, "VMH"),
        ],
    )

    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    intact_list = lists_dir / "intact.txt"
    mute_list = lists_dir / "mute.txt"
    intact_list.write_text(
        "20240101_100000\n20240101_120000\n20240102_100000\n20240103_100000\n"
    )
    mute_list.write_text("20240102_100000\n")

    out_dir = tmp_path / "out"

    return {
        "intact_list": intact_list,
        "mute_list": mute_list,
        "catalog": catalog_path,
        "data_root": data_root,
        "out_dir": out_dir,
        "unit_uid_day1": f"M01_20240101_{unit_day1}",
        "unit_uid_day2": f"M02_20240102_{unit_day2}",
    }


def test_aggregator_writes_pickle_with_expected_top_level_schema(tmp_path):
    """
    Description
    -----------
    The aggregator's output must be a pickle whose top-level dict
    carries every documented key, in the right scalar / container
    shape. Loadability and key set are checked here; deeper structural
    invariants live in the dedicated tests below.
    """

    fx = _build_aggregator_tree(tmp_path)
    out_path = aggregate_units_across_conditions(
        condition_to_session_list={
            "intact_female": fx["intact_list"],
            "mute_female": fx["mute_list"],
        },
        catalog_path=fx["catalog"],
        out_dir=fx["out_dir"],
        data_root=fx["data_root"],
        **DEFAULT_THRESHOLDS,
    )
    assert out_path.exists()
    assert out_path.suffix == ".pkl"

    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    expected_top_keys = {
        "generated_at",
        "thresholds_used",
        "catalog_path",
        "data_root",
        "conditions_included",
        "sessions_skipped",
        "n_units_total",
        "n_units_per_condition",
        "units",
    }
    assert expected_top_keys == set(out.keys())
    assert out["thresholds_used"] == DEFAULT_THRESHOLDS
    assert out["n_units_total"] == 2
    assert set(out["units"].keys()) == {fx["unit_uid_day1"], fx["unit_uid_day2"]}


def test_aggregator_collapses_same_day_unit_across_sessions(tmp_path):
    """
    Description
    -----------
    A unit recorded in two same-day sessions of the same condition
    appears as ONE `units[uid]` entry, with both sessions stacked
    under `conditions["intact_female"]["sessions_tested"]` and both
    rows present under each shared modality's `per_session` list.

    For the toggled USV PETH modality (significant in session A,
    NOT significant in session B), the aggregate must report
    `n_significant=1`, `n_tested=2`, `consistency=0.5`.
    """

    fx = _build_aggregator_tree(tmp_path)
    out_path = aggregate_units_across_conditions(
        condition_to_session_list={
            "intact_female": fx["intact_list"],
            "mute_female": fx["mute_list"],
        },
        catalog_path=fx["catalog"],
        out_dir=fx["out_dir"],
        data_root=fx["data_root"],
        **DEFAULT_THRESHOLDS,
    )
    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    unit = out["units"][fx["unit_uid_day1"]]
    assert unit["mouse_id"] == "M01"
    assert unit["rec_date"] == 20240101
    assert unit["anatomy_region"] == "PAG"
    assert unit["imec"] == 0
    assert unit["cluster_num"] == 1
    assert unit["peak_channel"] == 100
    assert unit["kslabel"] == "good"

    intact = unit["conditions"]["intact_female"]
    assert intact["sessions_tested"] == ["20240101_100000", "20240101_120000"]

    peth = intact["modalities"]["usv_peth_self_excit"]
    assert peth["n_tested"] == 2
    assert peth["n_significant"] == 1
    assert peth["consistency"] == pytest.approx(0.5)
    assert {row["session"] for row in peth["per_session"]} == {
        "20240101_100000",
        "20240101_120000",
    }


def test_aggregator_modality_absent_in_session_not_counted_in_n_tested(tmp_path):
    """
    Description
    -----------
    Session B's pkl omits the `spatial` block entirely. The spatial
    modality therefore appears in session A only, and the unit's
    aggregate must report `n_tested=1` (not 2). This is the
    invariant that lets the aggregator stay honest when older /
    partial pkls coexist with full ones.
    """

    fx = _build_aggregator_tree(tmp_path)
    out_path = aggregate_units_across_conditions(
        condition_to_session_list={
            "intact_female": fx["intact_list"],
            "mute_female": fx["mute_list"],
        },
        catalog_path=fx["catalog"],
        out_dir=fx["out_dir"],
        data_root=fx["data_root"],
        **DEFAULT_THRESHOLDS,
    )
    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    intact = out["units"][fx["unit_uid_day1"]]["conditions"]["intact_female"]
    spatial = intact["modalities"]["spatial_0ms_self_xy"]
    assert spatial["n_tested"] == 1
    assert spatial["n_significant"] == 1
    assert spatial["consistency"] == pytest.approx(1.0)
    assert [row["session"] for row in spatial["per_session"]] == [
        "20240101_100000"
    ]


def test_aggregator_same_unit_appears_under_both_conditions(tmp_path):
    """
    Description
    -----------
    The 20240102 session is listed in BOTH the intact_female and
    mute_female `.txt` files. The aggregator must therefore record
    one unit with TWO entries under `conditions`, each carrying its
    own `sessions_tested` and `modalities` block. `n_units_total`
    must still be 2 (the unit is not double-counted across
    conditions).
    """

    fx = _build_aggregator_tree(tmp_path)
    out_path = aggregate_units_across_conditions(
        condition_to_session_list={
            "intact_female": fx["intact_list"],
            "mute_female": fx["mute_list"],
        },
        catalog_path=fx["catalog"],
        out_dir=fx["out_dir"],
        data_root=fx["data_root"],
        **DEFAULT_THRESHOLDS,
    )
    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    unit = out["units"][fx["unit_uid_day2"]]
    assert set(unit["conditions"].keys()) == {"intact_female", "mute_female"}
    assert unit["conditions"]["intact_female"]["sessions_tested"] == [
        "20240102_100000"
    ]
    assert unit["conditions"]["mute_female"]["sessions_tested"] == [
        "20240102_100000"
    ]
    assert out["n_units_total"] == 2
    assert out["n_units_per_condition"]["intact_female"] == 2
    assert out["n_units_per_condition"]["mute_female"] == 1


def test_aggregator_skips_sessions_without_tuning_curves_dir(tmp_path):
    """
    Description
    -----------
    A session listed in a condition `.txt` but missing its
    `<sess>/ephys/tuning_curves/` directory on disk must be recorded
    under `sessions_skipped[<cond>]` and not contribute units. The
    aggregator must NOT raise on this case — it is the expected
    state for sessions that have not yet been processed by
    `generate-rm`.
    """

    fx = _build_aggregator_tree(tmp_path)
    out_path = aggregate_units_across_conditions(
        condition_to_session_list={
            "intact_female": fx["intact_list"],
            "mute_female": fx["mute_list"],
        },
        catalog_path=fx["catalog"],
        out_dir=fx["out_dir"],
        data_root=fx["data_root"],
        **DEFAULT_THRESHOLDS,
    )
    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    assert "20240103_100000" in out["sessions_skipped"]["intact_female"]
    assert out["sessions_skipped"]["mute_female"] == []
    # Sanity: the surviving conditions still produced their units.
    assert out["n_units_total"] == 2


def test_aggregator_raises_on_orphan_pkl(tmp_path):
    """
    Description
    -----------
    The catalog is the authoritative scope: a pkl whose
    `(mouse_id, rec_date, unit_id)` triple has no catalog row is an
    integrity error, not something to silently drop. The aggregator
    must raise `KeyError`. We simulate this by writing an extra pkl
    with a `unit_id` the catalog does not know about.
    """

    fx = _build_aggregator_tree(tmp_path)
    # Write an extra cluster pkl on day 1 that the catalog does not list.
    sess_a = fx["data_root"] / "20240101_100000" / "ephys" / "tuning_curves"
    _write_cluster_pkl(
        sess_a,
        "imec0_cl9999_ch300_good",
        _full_triage_stats(),
    )
    with pytest.raises(KeyError):
        aggregate_units_across_conditions(
            condition_to_session_list={
                "intact_female": fx["intact_list"],
                "mute_female": fx["mute_list"],
            },
            catalog_path=fx["catalog"],
            out_dir=fx["out_dir"],
            data_root=fx["data_root"],
            **DEFAULT_THRESHOLDS,
        )


def test_aggregator_raises_on_missing_session_list(tmp_path):
    """
    Description
    -----------
    A condition pointing at a `.txt` file that does not exist must
    raise `FileNotFoundError` — the aggregator refuses to silently
    treat the condition as empty.
    """

    fx = _build_aggregator_tree(tmp_path)
    with pytest.raises(FileNotFoundError):
        aggregate_units_across_conditions(
            condition_to_session_list={
                "intact_female": fx["intact_list"],
                "mute_female": tmp_path / "does_not_exist.txt",
            },
            catalog_path=fx["catalog"],
            out_dir=fx["out_dir"],
            data_root=fx["data_root"],
            **DEFAULT_THRESHOLDS,
        )


def test_aggregator_defaults_to_settings_json_thresholds(tmp_path):
    """
    Description
    -----------
    When no threshold kwargs are supplied, the aggregator must load
    defaults from `analyses_settings.json[detect_interesting_tuning_neurons]`
    and record them under `thresholds_used`. This locks in the
    "shared defaults with the per-session detector" contract.
    """

    fx = _build_aggregator_tree(tmp_path)
    out_path = aggregate_units_across_conditions(
        condition_to_session_list={
            "intact_female": fx["intact_list"],
            "mute_female": fx["mute_list"],
        },
        catalog_path=fx["catalog"],
        out_dir=fx["out_dir"],
        data_root=fx["data_root"],
    )
    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    settings_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src"
        / "usv_playpen"
        / "_parameter_settings"
        / "analyses_settings.json"
    )
    with settings_path.open() as fh:
        cfg = json.load(fh)
    expected = cfg["detect_interesting_tuning_neurons"]
    assert out["thresholds_used"] == expected
