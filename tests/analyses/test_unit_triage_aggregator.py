"""
@author: bartulem
Tests for `usv_playpen.analyses.unit_triage_aggregator`.

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

from usv_playpen.analyses.unit_triage_aggregator import (
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
                "vocal_category": {
                    "role": "self",
                    **_categorical_payload(
                        peak_abs_z=4.0 if cat_significant else 0.5
                    ),
                },
            },
        },
        "usv_category_peth": {
            "self": {
                "vocal_category": {
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
    "ibi_bout_alpha": 0.01,
    "ibi_bout_min_bouts": 10,
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
        "usv_category_self_vocal_category",
        "usv_category_peth_self_vocal_category_excit",
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

    cat = records["usv_category_self_vocal_category"]
    assert cat["cat_feat"] == "vocal_category"
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
        "usv_category_self_vocal_category",
        [{"peak_abs_z": 3.5}, {"peak_abs_z": 5.0}],
    )
    assert cat_agg["max_peak_abs_z"] == pytest.approx(5.0)
    assert cat_agg["median_peak_abs_z"] == pytest.approx(4.25)

    cat_peth_agg = _aggregate_modality_stats(
        "usv_category_peth_self_vocal_category_excit",
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
    must raise `KeyError` — and it collects ALL orphans and reports them
    together rather than aborting on the first, so writing two unknown
    pkls must yield one error naming both.
    """

    fx = _build_aggregator_tree(tmp_path)
    # Write TWO extra cluster pkls the catalog does not list, so the run
    # must reach both before failing (proves fail-at-end, not abort-on-first).
    sess_a = fx["data_root"] / "20240101_100000" / "ephys" / "tuning_curves"
    _write_cluster_pkl(sess_a, "imec0_cl9999_ch300_good", _full_triage_stats())
    _write_cluster_pkl(sess_a, "imec0_cl8888_ch400_good", _full_triage_stats())
    with pytest.raises(KeyError) as exc_info:
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
    msg = str(exc_info.value)
    assert "2 pkl(s)" in msg
    assert "cl9999" in msg and "cl8888" in msg


def test_aggregator_raises_on_date_mapped_to_multiple_mice(tmp_path):
    """
    Description
    -----------
    The per-session `mouse_id` lookup assumes each `rec_date` maps to
    exactly one mouse. A catalog that maps a date to two mice would
    mis-attribute every unit recorded that day, so the aggregator must
    raise `ValueError` rather than silently taking the first mouse.
    """

    fx = _build_aggregator_tree(tmp_path)
    # Append a row giving 20240101 a SECOND mouse, violating the invariant.
    with open(fx["catalog"], "a") as fh:
        fh.write("M99,20240101,imec0_cl0003_ch300_good,PAG\n")
    with pytest.raises(ValueError, match="one-date-one-mouse"):
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
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "src"
        / "usv_playpen"
        / "_parameter_settings"
        / "analyses_settings.json"
    )
    with settings_path.open() as fh:
        cfg = json.load(fh)
    expected = cfg["detect_interesting_tuning_neurons"]
    assert out["thresholds_used"] == expected


# Aggregator edge branches ---------------------------------------------------


def test_aggregate_modality_stats_empty_per_session_yields_none():
    """
    Description
    -----------
    Every modality-stats branch must return its scalar keys with `None`
    values (never raise) when the per-session evidence list is empty.
    """

    for key in (
        "vmi_self_excit",
        "spatial_0ms_self_xy",
        "usv_category_self_cat",
        "usv_peth_self_excit",
    ):
        agg = _aggregate_modality_stats(key, [])
        assert agg and all(v is None for v in agg.values())


def test_flag_one_cluster_skips_non_dict_blocks():
    """
    Description
    -----------
    `flag_one_cluster` must skip (not crash) when a modality block or
    its inner payload is not a dict — both the emitter-level and the
    property/feature/direction-level guards.
    """

    outer = {"triage_stats": {
        "vmi": {"e1": "x"},
        "usv_peth": {"e1": "x"},
        "usv_property_tuning": {"e1": "x"},
        "usv_category_tuning": {"e1": "x"},
        "usv_category_peth": {"e1": "x"},
        "behavioral": {"off0": "x"},
        "spatial": {"off0": "x"},
    }}
    assert flag_one_cluster(outer, **DEFAULT_THRESHOLDS) == {}

    inner = {"triage_stats": {
        "usv_peth": {"e1": {"excit": "x"}},          # direction block not a dict
        "usv_property_tuning": {"e1": {"dur": "x"}},  # prop payload not a dict
        "usv_category_tuning": {"e1": {"cat": "x"}},  # cat payload not a dict
        "usv_category_peth": {"e1": {"cat": "x"}},
        "behavioral": {"off0": {"feat": "x"}},
        "spatial": {"off0": {"feat": "x"}},
    }}
    assert flag_one_cluster(inner, **DEFAULT_THRESHOLDS) == {}


def test_aggregator_logs_and_skips_bad_sessions(tmp_path):
    """
    Description
    -----------
    The three "skip with a logged reason" branches must all fire and be
    recorded under `sessions_skipped`: an unparseable date prefix, a
    rec_date absent from the catalog, and a `tuning_curves` dir with no
    pkls. No unit is produced.
    """

    data_root = tmp_path / "data"
    (data_root / "badname" / "ephys" / "tuning_curves").mkdir(parents=True)
    (data_root / "20240505_100000" / "ephys" / "tuning_curves").mkdir(parents=True)
    (data_root / "20240101_100000" / "ephys" / "tuning_curves").mkdir(parents=True)

    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, [("M01", 20240101, "imec0_cl0001_ch100_good", "PAG")])
    lst = tmp_path / "intact.txt"
    lst.write_text("badname\n20240505_100000\n20240101_100000\n")

    out_path = aggregate_units_across_conditions(
        condition_to_session_list={"intact_female": lst},
        catalog_path=catalog,
        out_dir=tmp_path / "out",
        data_root=data_root,
        **DEFAULT_THRESHOLDS,
    )
    with out_path.open("rb") as fh:
        out = pickle.load(fh)
    skipped = set(out["sessions_skipped"]["intact_female"])
    assert {"badname", "20240505_100000", "20240101_100000"} <= skipped
    assert out["n_units_total"] == 0


# usv_ibi_bout_onset labels


def _ibi_bout_payload(
    *,
    onset_gain: float | None,
    first_over_later: float | None,
    p: float,
    n_bouts: int,
    role: str | None = "self",
) -> dict:
    """
    Description
    -----------
    Build a minimal `triage_stats["usv_ibi_bout_onset"][emitter]` block with the fields
    `flag_one_cluster` reads: the bout count, the IBI threshold and, for each of the five
    descriptors, its observed value and two-sided shift-null p. The three descriptors that carry
    no label (`peak_latency`, `pre_onset_gain`, `fraction_first_above`) get fixed values.

    Parameters
    ----------
    onset_gain (float | None)
        Observed onset gain; above 1 reads as excitation, below 1 as suppression.
    first_over_later (float | None)
        Observed first-over-later USV response ratio; above 1 reads as `first`, below as `later`.
    p (float)
        Two-sided p shared by the two labelled descriptors.
    n_bouts (int)
        Number of IBI bouts, gated by `ibi_bout_min_bouts`.
    role (str | None)
        Emitter role placed on the payload; None leaves the `role` key out.

    Returns
    -------
    payload (dict)
        The bout-onset block.
    """

    payload = {
        "n_bouts": n_bouts,
        "n_shuffles": 100,
        "ibi_threshold_seconds": 0.1155,
        "onset_gain": {"observed": onset_gain, "p_two_sided": p},
        "peak_latency": {"observed": 0.08, "p_two_sided": 0.5},
        "pre_onset_gain": {"observed": 1.0, "p_two_sided": 0.9},
        "first_over_later": {"observed": first_over_later, "p_two_sided": p},
        "fraction_first_above": {"observed": 0.6, "p_two_sided": 0.3},
    }
    if role is not None:
        payload["role"] = role
    return payload


def _ibi_records(records: dict) -> dict:
    """
    Description
    -----------
    Keep only the bout-onset records of a `flag_one_cluster` result.

    Parameters
    ----------
    records (dict)
        Output of `flag_one_cluster`.

    Returns
    -------
    ibi (dict)
        The records whose key starts with `ibi_bout_`.
    """

    return {k: v for k, v in records.items() if k.startswith("ibi_bout_")}


def test_flag_one_cluster_ibi_bout_onset_significant_excitation_and_first():
    """
    Description
    -----------
    An onset gain above 1 and a first-over-later ratio above 1, both with p under
    `ibi_bout_alpha` and enough bouts, yield `ibi_bout_onset_<role>_excit` and
    `ibi_bout_first_usv_<role>_first`, each tested and significant and carrying every extracted
    metric (observed values, p-values, bout count and threshold).
    """

    cluster_data = {"triage_stats": {"usv_ibi_bout_onset": {
        "m1": _ibi_bout_payload(onset_gain=2.0, first_over_later=1.5, p=0.001, n_bouts=20)}}}
    records = _ibi_records(flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS))

    assert set(records) == {"ibi_bout_onset_self_excit", "ibi_bout_first_usv_self_first"}
    for record in records.values():
        assert record["tested"] is True
        assert record["significant"] is True
        assert record["role"] == "self"
        assert record["n_bouts"] == 20.0
        assert record["ibi_threshold_seconds"] == pytest.approx(0.1155)
        assert record["onset_gain"] == pytest.approx(2.0)
        assert record["first_over_later_p"] == pytest.approx(0.001)
        assert record["peak_latency"] == pytest.approx(0.08)
        assert record["fraction_first_above_p"] == pytest.approx(0.3)


def test_flag_one_cluster_ibi_bout_onset_significant_suppression_and_later():
    """
    Description
    -----------
    Values below 1 flip the direction: a significant onset gain under 1 is `suppress` and a
    significant first-over-later ratio under 1 is `later`.
    """

    cluster_data = {"triage_stats": {"usv_ibi_bout_onset": {
        "m1": _ibi_bout_payload(onset_gain=0.4, first_over_later=0.7, p=0.001, n_bouts=20)}}}
    records = _ibi_records(flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS))

    assert set(records) == {"ibi_bout_onset_self_suppress", "ibi_bout_first_usv_self_later"}
    assert all(record["significant"] for record in records.values())


@pytest.mark.parametrize(
    ("p", "n_bouts"),
    [(0.001, 5), (0.2, 20)],
    ids=["below_min_bouts", "p_above_alpha"],
)
def test_flag_one_cluster_ibi_bout_onset_tested_but_not_significant(p, n_bouts):
    """
    Description
    -----------
    A unit with fewer bouts than `ibi_bout_min_bouts` (even at a tiny p), or with p above
    `ibi_bout_alpha`, still gets both labels recorded as tested, with the direction inferred
    from the observed value, but neither is significant.
    """

    cluster_data = {"triage_stats": {"usv_ibi_bout_onset": {
        "m1": _ibi_bout_payload(onset_gain=2.0, first_over_later=0.8, p=p, n_bouts=n_bouts)}}}
    records = _ibi_records(flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS))

    assert set(records) == {"ibi_bout_onset_self_excit", "ibi_bout_first_usv_self_later"}
    for record in records.values():
        assert record["tested"] is True
        assert record["significant"] is False


def test_flag_one_cluster_ibi_bout_onset_skips_non_finite_observed():
    """
    Description
    -----------
    A descriptor with no defined observed value (NaN, e.g. a bout with no later USVs, or None)
    carries no direction, so its label is left out while the other label of the same block is
    still recorded; a non-dict payload is skipped outright.
    """

    cluster_data = {"triage_stats": {"usv_ibi_bout_onset": {
        "m1": _ibi_bout_payload(onset_gain=float("nan"), first_over_later=1.4, p=0.001,
                                n_bouts=20),
        "f1": _ibi_bout_payload(onset_gain=1.8, first_over_later=None, p=0.001, n_bouts=20,
                                role="partner"),
        "x1": "not a dict",
    }}}
    records = _ibi_records(flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS))

    assert set(records) == {"ibi_bout_first_usv_self_first", "ibi_bout_onset_partner_excit"}
    assert records["ibi_bout_first_usv_self_first"]["onset_gain"] is None


@pytest.mark.parametrize(
    "triage_stats",
    [{}, {"usv_ibi_bout_onset": {}}, {"usv_ibi_bout_onset": None}],
    ids=["block_absent", "block_empty", "block_none"],
)
def test_flag_one_cluster_ibi_bout_onset_absent_block_yields_no_records(triage_stats):
    """
    Description
    -----------
    Pickles written before the bout-onset statistics existed carry no `usv_ibi_bout_onset`
    block (or an empty one); they must produce no bout-onset record rather than raise.
    """

    assert _ibi_records(flag_one_cluster({"triage_stats": triage_stats},
                                         **DEFAULT_THRESHOLDS)) == {}


def test_flag_one_cluster_ibi_bout_onset_role_resolution():
    """
    Description
    -----------
    The role in the modality key comes from the payload's own `role` when present; otherwise
    from the emitter-to-role map built from the other vocal payloads (here `usv_peth`); and
    failing both, from the emitter name itself.
    """

    cluster_data = {
        "usv_peth": {"f1": {"role": "partner"}},
        "triage_stats": {"usv_ibi_bout_onset": {
            "m1": _ibi_bout_payload(onset_gain=2.0, first_over_later=1.5, p=0.001,
                                    n_bouts=20, role="self"),
            "f1": _ibi_bout_payload(onset_gain=2.0, first_over_later=1.5, p=0.001,
                                    n_bouts=20, role=None),
            "u1": _ibi_bout_payload(onset_gain=2.0, first_over_later=1.5, p=0.001,
                                    n_bouts=20, role=""),
        }},
    }
    records = _ibi_records(flag_one_cluster(cluster_data, **DEFAULT_THRESHOLDS))

    assert {"ibi_bout_onset_self_excit", "ibi_bout_onset_partner_excit",
            "ibi_bout_onset_u1_excit"} <= set(records)
    assert records["ibi_bout_onset_partner_excit"]["role"] == "partner"
    assert records["ibi_bout_onset_u1_excit"]["role"] == "u1"


def test_aggregator_ibi_bout_thresholds_fall_back_to_settings_and_gate_the_labels(tmp_path):
    """
    Description
    -----------
    End to end through the cross-session aggregator: with `ibi_bout_alpha` and
    `ibi_bout_min_bouts` left as None (other thresholds given), `thresholds_used` records the
    shipped settings values for those two, and a unit with p = 0.1 over 5 bouts is tested but not
    significant. Passing a looser alpha (0.2) and a lower bout floor (3) records those values and
    flips the same unit to significant, so the two keywords really reach the flagging.
    """

    data_root = tmp_path / "data"
    tuning_dir = data_root / "20240101_100000" / "ephys" / "tuning_curves"
    tuning_dir.mkdir(parents=True)
    unit_id = "imec0_cl0001_ch100_good"
    _write_cluster_pkl(tuning_dir, unit_id, {"usv_ibi_bout_onset": {
        "m1": _ibi_bout_payload(onset_gain=2.0, first_over_later=1.5, p=0.1, n_bouts=5)}})
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, [("M01", 20240101, unit_id, "PAG")])
    session_list = tmp_path / "intact.txt"
    session_list.write_text("20240101_100000\n")

    settings_path = (pathlib.Path(__file__).resolve().parent.parent.parent / "src"
                     / "usv_playpen" / "_parameter_settings" / "analyses_settings.json")
    with settings_path.open() as fh:
        shipped = json.load(fh)["detect_interesting_tuning_neurons"]

    explicit = {k: v for k, v in DEFAULT_THRESHOLDS.items() if not k.startswith("ibi_bout_")}
    explicit["vmi_alpha"] = 0.05
    outcomes = {}
    for name, ibi_kwargs in (("defaults", {"ibi_bout_alpha": None, "ibi_bout_min_bouts": None}),
                             ("loose", {"ibi_bout_alpha": 0.2, "ibi_bout_min_bouts": 3})):
        out_path = aggregate_units_across_conditions(
            condition_to_session_list={"intact_female": session_list},
            catalog_path=catalog,
            out_dir=tmp_path / name,
            data_root=data_root,
            **explicit,
            **ibi_kwargs,
        )
        with out_path.open("rb") as fh:
            outcomes[name] = pickle.load(fh)

    used = outcomes["defaults"]["thresholds_used"]
    assert used["ibi_bout_alpha"] == shipped["ibi_bout_alpha"]
    assert used["ibi_bout_min_bouts"] == shipped["ibi_bout_min_bouts"]
    assert used["vmi_alpha"] == 0.05
    assert outcomes["loose"]["thresholds_used"]["ibi_bout_alpha"] == 0.2
    assert outcomes["loose"]["thresholds_used"]["ibi_bout_min_bouts"] == 3

    uid = f"M01_20240101_{unit_id}"
    for name, expected in (("defaults", 0), ("loose", 1)):
        modalities = outcomes[name]["units"][uid]["conditions"]["intact_female"]["modalities"]
        for key in ("ibi_bout_onset_self_excit", "ibi_bout_first_usv_self_first"):
            assert modalities[key]["n_tested"] == 1
            assert modalities[key]["n_significant"] == expected
