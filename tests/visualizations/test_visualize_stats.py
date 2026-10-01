"""
@author: bartulem
Mock-based tests for visualizations/usv_summary_statistics.py and
visualizations/usv_interval_summary_statistics.py — both files are pure
compute / data-loading helpers when wired to synthetic sessions on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pls
import pytest

# Force a non-interactive matplotlib backend before any plotting helpers
# import matplotlib so figure-rendering tests don't try to open a window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_playpen.visualizations.usv_summary_statistics import (
    extract_session_metadata,
    load_and_filter_usv_data,
    extract_category_embedding_data,
    get_session_behavioral_features,
    merge_usv_and_behavioral_features,
    build_master_usv_dataframe,
    plot_assignment_stacked_bars,
    plot_assignment_summary_panel,
    plot_animal_participation_stats,
    plot_polar_kde_distance_angle,
    plot_behavior_duration_regressions,
    plot_distance_by_assignment_kde_anova,
    plot_duration_histograms_by_sex,
    plot_estrous_ratio_scatter,
    plot_estrous_usv_rates,
    plot_estrous_stage_pie_chart,
    plot_category_global_fatigue_heatmap,
    plot_category_estrous_rates_grid,
    plot_category_estrous_ratio_grid,
    plot_unassigned_proportion_vs_distance_jointplot,
    plot_session_squeak_time_heatmap,
    plot_hourly_regressions,
    plot_local_fatigue_binned_trends,
    plot_category_local_fatigue_heatmap,
    plot_category_prevalence_and_embedding,
    plot_category_polar_kde_grid,
    plot_estrous_category_kde_grid,
)
from usv_playpen.visualizations.usv_interval_summary_statistics import (
    find_latest_archive,
    load_intervals_from_h5,
    load_mixture_model_fits_from_h5,
    load_best_fit_from_h5,
    load_lrt_sweep_from_h5,
    selected_K_from_h5,
    plot_log_usv_interval_histograms,
    plot_ic_curves,
    plot_qq,
    plot_best_fit_with_annotations,
    plot_bootstrap_lrt_panel,
    serial_dependence_pairs,
    load_serial_dependence_from_h5,
    plot_serial_dependence,
    load_tied_model_from_h5,
    load_peak_lrt_sweep_from_h5,
    load_tied_ic_table_from_h5,
)
from usv_playpen.visualizations import usv_interval_summary_statistics as uiss
from usv_playpen.analyses.compute_inter_usv_interval_distributions import fit_mixture_model_sweep
from usv_playpen.analyses.usv_interval_archive import write_ivi_h5


def test_component_palette_colors_derive_from_settings():
    """The per-mixture-component palette pairs code-defined line styles with
    colours read from the ``component_colors`` block of
    ``visualizations_settings.json`` rather than hard-coded hex, so the palette's
    colour column must equal the shipped list (line styles stay in code)."""

    settings_path = (Path(uiss.__file__).parent.parent
                     / "_parameter_settings" / "visualizations_settings.json")
    with settings_path.open() as fh:
        component_colors = json.load(fh)["component_colors"]
    palette_colors = [color for color, _linestyle in uiss._COMPONENT_PALETTE]
    assert palette_colors == list(component_colors[:len(palette_colors)])


# ---------------------------------------------------------------------------
# Synthetic session builder — produces the disk layout extract_session_metadata
# / load_and_filter_usv_data / build_master_usv_dataframe expect.
# ---------------------------------------------------------------------------


def _make_synthetic_session(
    session_root: Path,
    *,
    male_id: str = "Mr_X",
    female_id: str = "Ms_Y",
    frame_rate: float = 150.0,
    experiment_code: str = "exp1",
    n_male_calls: int = 4,
    n_female_calls: int = 3,
    n_unassigned: int = 1,
    include_behavioral: bool = True,
    include_embedding: bool = True,
    write_noise_column: bool = True,
    cat_col: str = "usv_supercategory",
    n_noise: int = 2,
    n_squeak: int = 0,
    n_both: int = 0,
    write_call_class_column: bool = True,
):
    """Build a session_root containing:

    - A 3D-tracking H5 file at <root>/video/<session_id>_points3d_translated_rotated_metric.h5
      with `track_names`, `recording_frame_rate`, `experimental_code`.
    - A USV summary CSV at <root>/audio/<session_id>_usv_summary.csv with
      `start`, `duration`, `emitter`, the noise column, the category column and
      the `call_class` column (`usv` on the calls, `n_squeak` male `squeak` rows and
      `n_both` male `both` rows appended before the noise rows, null on noise rows).
    - Optionally a behavioral features CSV at
      <root>/<session_id>_behavioral_features.csv with the standard
      `nose-nose`, `<X>-allo_yaw-nose`, `<X>-nose-allo_yaw` suffix columns.

    The session_root.name is taken as <YYYYMMDD>_<HHMMSS> (e.g. "20260101_120000")
    to satisfy the build_master_usv_dataframe split('_') / int parsing.
    """
    session_root.mkdir(parents=True, exist_ok=True)
    session_id = session_root.name  # e.g. "20260101_120000"

    # ---- tracking H5 ------------------------------------------------------
    video_dir = session_root / "video"
    video_dir.mkdir(exist_ok=True)
    h5_path = video_dir / f"{session_id}_points3d_translated_rotated_metric.h5"
    with h5py.File(h5_path, "w") as f:
        # track_names is read as a list of bytes that gets `.decode('utf-8')`d
        f.create_dataset("track_names", data=np.array([male_id.encode(), female_id.encode()]))
        f.create_dataset("recording_frame_rate", data=np.float64(frame_rate))
        f.create_dataset("experimental_code", data=np.bytes_(experiment_code))

    # ---- USV summary CSV --------------------------------------------------
    audio_dir = session_root / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows_start = []
    rows_dur = []
    rows_emitter = []
    rows_noise = []
    rows_class = []
    rows_cat = []
    rows_umap_x = []
    rows_umap_y = []

    t = 0.05
    for _ in range(n_male_calls):
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append(male_id)
        rows_noise.append(False)  # a real vocalization
        rows_class.append("usv")
        rows_cat.append(1)
        rows_umap_x.append(np.random.RandomState(0).rand())
        rows_umap_y.append(np.random.RandomState(1).rand())
        t += 0.5
    for _ in range(n_female_calls):
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append(female_id)
        rows_noise.append(False)
        rows_class.append("usv")
        rows_cat.append(2)
        rows_umap_x.append(0.5)
        rows_umap_y.append(0.5)
        t += 0.5
    for _ in range(n_unassigned):
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append("UNKNOWN")
        rows_noise.append(False)
        rows_class.append("usv")
        rows_cat.append(3)
        rows_umap_x.append(0.7)
        rows_umap_y.append(0.7)
        t += 0.5
    for squeak_class in ["squeak"] * n_squeak + ["both"] * n_both:
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append(male_id)
        rows_noise.append(False)
        rows_class.append(squeak_class)
        rows_cat.append(4)
        rows_umap_x.append(0.9)
        rows_umap_y.append(0.9)
        t += 0.5
    for _ in range(n_noise):
        rows_start.append(t)
        rows_dur.append(0.01)
        rows_emitter.append(male_id)
        rows_noise.append(True)  # flagged by the noise classifier
        rows_class.append(None)
        rows_cat.append(99)
        rows_umap_x.append(0.0)
        rows_umap_y.append(0.0)
        t += 0.5

    cols = {
        "start": rows_start,
        "duration": rows_dur,
        "emitter": rows_emitter,
    }
    if write_noise_column:
        cols["noise"] = rows_noise
    if write_call_class_column:
        cols["call_class"] = pls.Series(rows_class, dtype=pls.Utf8)
    if include_embedding:
        cols[cat_col] = rows_cat
        cols["umap_x"] = rows_umap_x
        cols["umap_y"] = rows_umap_y
    else:
        cols[cat_col] = rows_cat

    usv_csv = audio_dir / f"{session_id}_usv_summary.csv"
    pls.DataFrame(cols).write_csv(usv_csv)

    # ---- Behavioral features CSV -----------------------------------------
    if include_behavioral:
        n_frames = int(frame_rate * (t + 1.0))
        beh_cols = {
            # Suffix-based columns matching the build_master_usv_dataframe
            # default suffixes (nose-nose, allo_yaw-nose, nose-allo_yaw):
            "X-nose-nose": np.linspace(0.05, 0.5, n_frames).tolist(),
            "X-allo_yaw-nose": np.linspace(-1.5, 1.5, n_frames).tolist(),
            "X-nose-allo_yaw": np.linspace(1.5, -1.5, n_frames).tolist(),
        }
        beh_csv = session_root / f"{session_id}_behavioral_features.csv"
        pls.DataFrame(beh_cols).write_csv(beh_csv)

    return session_root, h5_path, usv_csv


# ---------------------------------------------------------------------------
# extract_session_metadata
# ---------------------------------------------------------------------------


def test_extract_session_metadata_happy_path(tmp_path):
    """Reads track_names, frame_rate, experimental_code from a synthetic H5."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, male_id="Mx", female_id="Fy",
                            frame_rate=200.0, experiment_code="my_exp")
    md = extract_session_metadata(str(sess))
    assert md["male_id"] == "Mx"
    assert md["female_id"] == "Fy"
    assert md["frame_rate"] == 200.0
    assert md["experiment_code"] == "my_exp"
    assert md["tracking_file"].name.endswith("_points3d_translated_rotated_metric.h5")


def test_extract_session_metadata_missing_h5_raises(tmp_path):
    """No tracking file → FileNotFoundError."""
    sess = tmp_path / "20260101_120000"
    sess.mkdir()
    with pytest.raises(FileNotFoundError):
        extract_session_metadata(str(sess))


def test_extract_session_metadata_single_track_raises(tmp_path):
    """A tracking H5 with only one track → IndexError (need at least 2 mice)."""
    sess = tmp_path / "20260101_120000"
    video = sess / "video"
    video.mkdir(parents=True)
    h5_path = video / "20260101_120000_points3d_translated_rotated_metric.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("track_names", data=np.array([b"only_one"]))
        f.create_dataset("recording_frame_rate", data=np.float64(150.0))
        f.create_dataset("experimental_code", data=np.bytes_("e"))
    with pytest.raises(IndexError):
        extract_session_metadata(str(sess))


# ---------------------------------------------------------------------------
# load_and_filter_usv_data
# ---------------------------------------------------------------------------


def test_load_and_filter_usv_data_drops_noise_and_adds_frame_index(tmp_path):
    """Noise rows must be removed; frame_index = floor(start * frame_rate)."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, n_male_calls=2, n_female_calls=1,
                            n_unassigned=0, n_noise=3)
    md = extract_session_metadata(str(sess))
    df = load_and_filter_usv_data(
        session_root=str(sess), frame_rate=md["frame_rate"],
        exclude_noise_usvs=True,
    )
    # Only non-noise rows survive: 2 male + 1 female = 3
    assert df.height == 3
    assert "frame_index" in df.columns
    # Spot-check: frame_index of the first call (start = 0.05) at 150 Hz = 7
    assert df["frame_index"][0] == int(0.05 * 150.0)


def test_load_and_filter_usv_data_raises_when_the_noise_column_is_absent(tmp_path):
    """A summary written by das-summarize carries no ``noise`` column until
    detect-usv-noise has run. Silently keeping every row there is how the previous
    convention quietly stopped filtering, so the loader now refuses and the error
    names both ways out."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, n_male_calls=2, n_female_calls=1,
                            n_unassigned=0, n_noise=3, write_noise_column=False)
    md = extract_session_metadata(str(sess))

    with pytest.raises(KeyError, match="detect-usv-noise"):
        load_and_filter_usv_data(
            session_root=str(sess), frame_rate=md["frame_rate"],
            exclude_noise_usvs=True,
        )

    # With the filter off the same summary loads untouched.
    kept = load_and_filter_usv_data(
        session_root=str(sess), frame_rate=md["frame_rate"],
        exclude_noise_usvs=False,
    )
    assert kept.height == 6
    assert "frame_index" in kept.columns


def test_load_and_filter_usv_data_missing_csv_raises(tmp_path):
    """No USV CSV → FileNotFoundError."""
    sess = tmp_path / "20260101_120000"
    sess.mkdir()
    with pytest.raises(FileNotFoundError):
        load_and_filter_usv_data(str(sess), frame_rate=150.0,
                                 exclude_noise_usvs=True)


# ---------------------------------------------------------------------------
# extract_category_embedding_data
# ---------------------------------------------------------------------------


def test_extract_category_embedding_data_concats_sessions(tmp_path):
    """Two sessions → one merged DataFrame with sex + dim1/dim2 columns."""
    sess1 = tmp_path / "20260101_120000"
    sess2 = tmp_path / "20260102_120000"
    _make_synthetic_session(sess1)
    _make_synthetic_session(sess2)
    df = extract_category_embedding_data(
        session_roots=[str(sess1), str(sess2)],
        exclude_noise_usvs=True,
        usv_category_col="usv_supercategory",
        usv_continuous_cols=("umap_x", "umap_y"),
    )
    assert set(df.columns) == {"sex", "category", "dim1", "dim2"}
    assert df.height > 0
    # Both 'male' and 'female' rows should appear
    sexes = set(df["sex"].to_list())
    assert "male" in sexes and "female" in sexes


def test_extract_category_embedding_data_skips_missing_columns(tmp_path):
    """A session whose CSV lacks the embedding cols is silently skipped."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, include_embedding=False)
    df = extract_category_embedding_data(
        session_roots=[str(sess)],
        exclude_noise_usvs=True,
        usv_category_col="usv_supercategory",
        usv_continuous_cols=("umap_x", "umap_y"),
    )
    # No matching columns → the per-session block continues; final result empty.
    assert df.height == 0


def test_extract_category_embedding_data_skips_bad_session(tmp_path):
    """A non-existent session is skipped silently (FileNotFoundError caught)."""
    df = extract_category_embedding_data(
        session_roots=["/no/such/session"],
        exclude_noise_usvs=True,
        usv_category_col="usv_supercategory",
        usv_continuous_cols=("umap_x", "umap_y"),
    )
    assert df.height == 0


# ---------------------------------------------------------------------------
# get_session_behavioral_features / merge_usv_and_behavioral_features
# ---------------------------------------------------------------------------


def test_get_session_behavioral_features_loads_csv_with_frame_index(tmp_path):
    """Loads behavioral CSV and prepends a frame_index row counter."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, include_behavioral=True)
    df = get_session_behavioral_features(str(sess))
    assert "frame_index" in df.columns
    # The frame_index is a row counter starting at 0
    assert df["frame_index"][0] == 0


def test_get_session_behavioral_features_missing_raises(tmp_path):
    """No behavioral CSV → FileNotFoundError."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, include_behavioral=False)
    with pytest.raises(FileNotFoundError):
        get_session_behavioral_features(str(sess))


def test_merge_usv_and_behavioral_features_join_shape(tmp_path):
    """Inner-joins USV onto behavioral by frame_index; output cols match docstring."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, include_behavioral=True)
    md = extract_session_metadata(str(sess))
    usv = load_and_filter_usv_data(str(sess), md["frame_rate"], True)
    beh = get_session_behavioral_features(str(sess))
    out = merge_usv_and_behavioral_features(
        usv_info=usv, behavioral_features=beh,
        nose_distance_col="X-nose-nose",
        mf_angle_col="X-allo_yaw-nose",
        fm_angle_col="X-nose-allo_yaw",
        usv_category_col="usv_supercategory",
    )
    assert set(out.columns) >= {
        "frame_index", "emitter", "category", "usv_duration",
        "distance", "mf_angle", "fm_angle",
    }


# ---------------------------------------------------------------------------
# build_master_usv_dataframe — top-level pipeline entry point
# ---------------------------------------------------------------------------


def test_build_master_usv_dataframe_returns_two_frames_and_count(tmp_path):
    """Two sessions → a usv_df, a background_df, and a noise-filtered count."""
    sess1 = tmp_path / "20260101_120000"
    sess2 = tmp_path / "20260102_120000"
    _make_synthetic_session(sess1, n_noise=2)
    _make_synthetic_session(sess2, n_noise=3)
    usv_df, bg_df, n_noise_total = build_master_usv_dataframe(
        session_roots=[str(sess1), str(sess2)],
        exclude_noise_usvs=True,
        usv_category_col="usv_supercategory",
        distance_suffix="nose-nose",
        mf_angle_suffix="allo_yaw-nose",
        fm_angle_suffix="nose-allo_yaw",
    )
    assert usv_df.height > 0
    assert bg_df.height > 0
    # 2 + 3 noise rows total
    assert n_noise_total == 5
    # Standard columns from the docstring
    for col in ("session_id", "date", "hour", "male_id", "female_id",
                "experiment_code", "emitter", "sex", "category",
                "start", "duration", "frame_index",
                "distance", "mf_angle", "fm_angle"):
        assert col in usv_df.columns


def test_build_master_usv_dataframe_handles_heterogeneous_session_dtypes(tmp_path):
    """Per-session CSVs are read with per-file dtype inference, so the same
    column can come back Int64 in one session and Float64 in another. The master
    concat must upcast (vertical_relaxed) rather than raising a SchemaError."""
    sess1 = tmp_path / "20260101_120000"
    sess2 = tmp_path / "20260102_120000"
    _make_synthetic_session(sess1)
    _make_synthetic_session(sess2)
    # Rewrite session 2's USV CSV so `start` is integer-typed (re-read as Int64),
    # diverging from session 1's Float64 `start`.
    csv2 = sess2 / "audio" / f"{sess2.name}_usv_summary.csv"
    pls.read_csv(csv2).with_columns(pls.col("start").cast(pls.Int64)).write_csv(csv2)
    usv_df, _bg, _n = build_master_usv_dataframe(
        session_roots=[str(sess1), str(sess2)],
        exclude_noise_usvs=True,
        usv_category_col="usv_supercategory",
        distance_suffix="nose-nose",
        mf_angle_suffix="allo_yaw-nose",
        fm_angle_suffix="nose-allo_yaw",
    )
    assert usv_df.height > 0


def test_build_master_usv_dataframe_raises_when_all_skipped(tmp_path):
    """Every session missing → RuntimeError with diagnostic message."""
    with pytest.raises(RuntimeError, match="loaded 0 sessions"):
        build_master_usv_dataframe(
            session_roots=[str(tmp_path / "absent")],
            exclude_noise_usvs=True,
            usv_category_col="usv_supercategory",
            distance_suffix="nose-nose",
            mf_angle_suffix="allo_yaw-nose",
            fm_angle_suffix="nose-allo_yaw",
        )


def test_build_master_usv_dataframe_skips_session_without_category_col(tmp_path):
    """A session whose USV CSV is missing the requested category column is
    dropped quietly. With every session skipped, the function raises
    RuntimeError (no usable data)."""
    sess1 = tmp_path / "20260101_120000"
    sess2 = tmp_path / "20260102_120000"
    _make_synthetic_session(sess1)
    _make_synthetic_session(sess2)
    with pytest.raises(RuntimeError):
        build_master_usv_dataframe(
            session_roots=[str(sess1), str(sess2)],
            exclude_noise_usvs=True,
            usv_category_col="some_missing_col",  # neither session has this
            distance_suffix="nose-nose",
            mf_angle_suffix="allo_yaw-nose",
            fm_angle_suffix="nose-allo_yaw",
        )


def test_build_master_usv_dataframe_counts_pure_usvs_only(tmp_path):
    """With ``usv_only`` (the default) the master frame counts call_class 'usv' rows only:
    squeak and both rows are left out (so every USV count built on it is a USV count), and
    ``usv_only=False`` keeps every non-noise vocalization."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, n_male_calls=4, n_female_calls=3, n_unassigned=1,
                            n_squeak=2, n_both=1, n_noise=2)
    common = dict(session_roots=[str(sess)], exclude_noise_usvs=True, usv_category_col="usv_supercategory",
                  distance_suffix="nose-nose", mf_angle_suffix="allo_yaw-nose", fm_angle_suffix="nose-allo_yaw")
    usv_only_df, _bg, n_noise = build_master_usv_dataframe(**common)
    assert usv_only_df.height == 8 and n_noise == 2
    assert 4 not in usv_only_df["category"].to_list()
    every_call_df, _bg, _n = build_master_usv_dataframe(**common, usv_only=False)
    assert every_call_df.height == 11


def test_build_master_usv_dataframe_usv_only_needs_call_class(tmp_path):
    """A summary without ``call_class`` cannot be split into USVs and squeaks, so
    ``usv_only`` raises a KeyError naming the column rather than counting squeaks as USVs."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, write_call_class_column=False)
    with pytest.raises(KeyError, match="call_class"):
        build_master_usv_dataframe(
            session_roots=[str(sess)], exclude_noise_usvs=True, usv_category_col="usv_supercategory",
            distance_suffix="nose-nose", mf_angle_suffix="allo_yaw-nose", fm_angle_suffix="nose-allo_yaw",
        )


def test_extract_category_embedding_data_keeps_pure_usvs_only(tmp_path):
    """The category embedding keeps call_class 'usv' rows only by default; squeak and both
    rows (category 4 in the synthetic session) come back with ``usv_only=False``."""
    sess = tmp_path / "20260101_120000"
    _make_synthetic_session(sess, n_squeak=1, n_both=1)
    kwargs = dict(session_roots=[str(sess)], exclude_noise_usvs=True, usv_category_col="usv_supercategory",
                  usv_continuous_cols=("umap_x", "umap_y"))
    assert 4 not in extract_category_embedding_data(**kwargs)["category"].to_list()
    assert extract_category_embedding_data(**kwargs, usv_only=False)["category"].to_list().count(4) == 2


# ===========================================================================
# usv_interval_summary_statistics — loaders driven off an HDF5 archive
# ===========================================================================


def _build_archive(tmp_path: Path, *, with_mixture_model: bool = True, with_lrt: bool = True,
                   corrected_lrt: bool = False) -> Path:
    """Constructs a usv_interval_analysis_<ts>.h5 file with both modes populated.
    ``corrected_lrt`` adds the session-corrected columns a current sweep writes."""
    intervals_df = pls.DataFrame({
        "session_id": ["s1", "s1", "s1", "s2"],
        "source_list": ["g", "g", "g", "g"],
        "interval_type": ["s2s"] * 4,
        "sex": ["male", "male", "female", "male"],
        "interval_s": [0.5, 0.7, 0.3, 0.9],
        "log_interval": np.log([0.5, 0.7, 0.3, 0.9]).tolist(),
        "male_id": ["M"] * 4,
        "female_id": ["F"] * 4,
    })
    drop_counts = pls.DataFrame({
        "session_id": ["s1", "s2"],
        "n_dropped_male": [0, 1],
        "n_dropped_female": [0, 0],
    })

    payload = {
        "s2s": {
            "attrs": {
                "alpha_effective": 0.05,
                "K_selected_male": 2,
                "K_selected_female": 3,
            },
            "intervals": intervals_df,
            "drop_counts": drop_counts,
        },
    }

    if with_mixture_model:
        mixture_model_rows = []
        for sex in ("male", "female"):
            for K in (1, 2):
                row = {
                    "sex": sex, "n_comp": K, "rep": 0,
                    "bic": 10.0, "aic": 10.0, "icl": 10.0,
                    "cv_neg_loglik": 1.0,
                    "model_class": "gauss",
                }
                # NaN-pad up to K=2
                for k in range(2):
                    row[f"weight_{k+1}"] = 0.5 if k < K else float("nan")
                    row[f"logmean_{k+1}"] = float(k - 0.5) if k < K else float("nan")
                    row[f"logsd_{k+1}"] = 0.5 if k < K else float("nan")
                    row[f"nu_{k+1}"] = float("nan")
                mixture_model_rows.append(row)
        payload["s2s"]["mixture_model_fits"] = pls.DataFrame(mixture_model_rows)

    if with_lrt:
        # Schema must match what load_lrt_sweep_from_h5 expects to read.
        payload["s2s"]["bootstrap_lrt"] = pls.DataFrame({
            "sex": ["male", "female"],
            "K_null": [1, 1],
            "K_alt": [2, 2],
            "B": [5, 5],
            "n_subsample": [100, 100],
            "model_class": ["gauss", "gauss"],
            "lr_obs": [4.5, 6.5],
            "p_value": [0.02, 0.01],
            "null_mean": [0.5, 0.4],
            "null_p95": [2.0, 1.8],
            "null_max": [2.5, 2.1],
            "K_selected_step_up": [2, 3],
        })
        if corrected_lrt:
            payload["s2s"]["bootstrap_lrt"] = payload["s2s"]["bootstrap_lrt"].with_columns(
                pls.Series("design_effect", [1.5, 2.0]),
                pls.Series("design_effect_raw", [1.5, 2.0]),
                pls.Series("lr_corrected", [3.0, 3.25]),
                pls.Series("p_value_corrected", [0.2, 0.4]),
            )
        payload["s2s"]["bootstrap_lrt_null"] = pls.DataFrame({
            "sex": ["male"] * 5,
            "K_null": [1] * 5,
            "K_alt": [2] * 5,
            "b": [0, 1, 2, 3, 4],
            "lr_b": [0.1, 0.5, 0.7, 1.1, 2.0],
        })

    out = tmp_path / "usv_interval_analysis_20260101_120000.h5"
    write_ivi_h5(out,
        analysis_attrs={"created_at_iso": "2026-01-01T12:00:00",
                        "git_sha": "abc123",
                        "n_sessions_loaded": 2,
                        "tau": 0.5},
        per_mode=payload)
    return out


# ---- find_latest_archive --------------------------------------------------


def test_find_latest_archive_picks_newest(tmp_path):
    """sorted(glob) → last element is the newest by lexicographic timestamp."""
    older = tmp_path / "usv_interval_analysis_20240101_000000.h5"
    newer = tmp_path / "usv_interval_analysis_20260101_120000.h5"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    assert find_latest_archive(str(tmp_path)) == newer


def test_find_latest_archive_raises_when_none(tmp_path):
    """Empty directory → FileNotFoundError with an actionable message."""
    with pytest.raises(FileNotFoundError, match="no usv_interval_analysis"):
        find_latest_archive(str(tmp_path))


# ---- load_intervals_from_h5 / load_mixture_model_fits_from_h5 / load_best_fit -------


def test_load_intervals_from_h5_returns_tidy_frame(tmp_path):
    """Loaded interval frame matches the schema written by write_ivi_h5."""
    arc = _build_archive(tmp_path)
    df = load_intervals_from_h5(str(arc), interval_type="s2s")
    assert df.height == 4
    assert set(df.columns) >= {"session_id", "sex", "interval_s", "log_interval"}


def test_load_intervals_from_h5_unknown_mode_raises(tmp_path):
    """Mode label not present in the archive → ValueError."""
    arc = _build_archive(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        load_intervals_from_h5(str(arc), interval_type="e2s")


def test_load_mixture_model_fits_from_h5_round_trip(tmp_path):
    """mixture_model_fits comes back with all per-component columns intact."""
    arc = _build_archive(tmp_path, with_mixture_model=True)
    df = load_mixture_model_fits_from_h5(str(arc), interval_type="s2s")
    for col in ("weight_1", "logmean_1", "logsd_1", "model_class"):
        assert col in df.columns


def test_load_mixture_model_fits_from_h5_missing_raises(tmp_path):
    """Archive without mixture-model sweep (fit_mixture_model=false at compute time) → ValueError."""
    arc = _build_archive(tmp_path, with_mixture_model=False)
    with pytest.raises(ValueError, match="contains no\\s+mixture model sweep"):
        load_mixture_model_fits_from_h5(str(arc), interval_type="s2s")


def test_load_best_fit_from_h5_returns_a_model(tmp_path):
    """End-to-end: load fits then reconstruct the (sex, K) model."""
    arc = _build_archive(tmp_path, with_mixture_model=True)
    model, order = load_best_fit_from_h5(str(arc), interval_type="s2s",
                                         sex="male", K=2)
    assert model is not None
    np.testing.assert_array_equal(order, np.arange(2))


# ---- load_lrt_sweep_from_h5 -----------------------------------------------


def test_load_lrt_sweep_from_h5_returns_dict(tmp_path):
    """Re-hydrates the per-(sex, K_null, K_alt) sweep dict."""
    arc = _build_archive(tmp_path, with_mixture_model=True, with_lrt=True)
    sweep = load_lrt_sweep_from_h5(str(arc), interval_type="s2s")
    assert "male" in sweep
    # Each key is a (K_null, K_alt) tuple
    male_keys = list(sweep["male"].keys())
    assert (1, 2) in male_keys
    male_entry = sweep["male"][(1, 2)]
    assert "lr_obs" in male_entry
    assert "p_value" in male_entry
    assert "lr_null" in male_entry
    # lr_null restored to numpy array
    assert isinstance(male_entry["lr_null"], np.ndarray)


def test_load_lrt_sweep_from_h5_reads_the_corrected_statistic(tmp_path):
    """A session-corrected sweep is read like the tied test: lr_obs / p_value are the
    corrected ones the step-up decided on, and the raw ones sit next to the design effect."""
    arc = _build_archive(tmp_path, with_mixture_model=True, with_lrt=True, corrected_lrt=True)
    entry = load_lrt_sweep_from_h5(str(arc), interval_type="s2s")["male"][(1, 2)]
    assert entry["lr_obs"] == 3.0 and entry["p_value"] == 0.2
    assert entry["lr_raw"] == 4.5 and entry["p_value_raw"] == 0.02
    assert entry["design_effect"] == 1.5


def test_serial_dependence_pairs_never_chains_two_animals_of_one_sex():
    """Two females in one session: each animal's intervals pair only among themselves, so
    the last interval of one female is never paired with the first of the other."""
    df = pls.DataFrame({
        "session_id": ["s1"] * 5,
        "sex": ["female"] * 5,
        "interval_s": [0.1, 0.2, 0.3, 5.0, 6.0],
        "emitter_id": ["A", "A", "A", "B", "B"],
    })
    current, following = serial_dependence_pairs(df, "female")
    np.testing.assert_allclose(current, [0.1, 0.2, 5.0])
    np.testing.assert_allclose(following, [0.2, 0.3, 6.0])


def test_load_lrt_sweep_from_h5_missing_tables_raises(tmp_path):
    """Archive without bootstrap_lrt → ValueError."""
    arc = _build_archive(tmp_path, with_mixture_model=False, with_lrt=False)
    with pytest.raises(ValueError, match="missing the\\s+bootstrap-LRT"):
        load_lrt_sweep_from_h5(str(arc), interval_type="s2s")


# ---- selected_K_from_h5 ---------------------------------------------------


def test_selected_K_from_h5_reads_attrs(tmp_path):
    """K_selected_{male,female} attrs come back as ints."""
    arc = _build_archive(tmp_path, with_mixture_model=True, with_lrt=True)
    sel = selected_K_from_h5(str(arc), interval_type="s2s")
    assert sel == {"male": 2, "female": 3}


def test_selected_K_from_h5_unknown_mode_raises(tmp_path):
    """Asking for a mode not in the archive → ValueError."""
    arc = _build_archive(tmp_path, with_mixture_model=True, with_lrt=True)
    with pytest.raises(ValueError, match="not found"):
        selected_K_from_h5(str(arc), interval_type="e2s")




# Smoke tests for the figure-rendering functions of usv_summary_statistics.
# These were previously uncovered (the module sat at ~11 %). Each builds a
# minimal, directly-constructed input (the functions take plain frames / dicts
# / arrays, not the full on-disk pipeline) and asserts a Figure is produced
# without error. Every returned figure is closed so the suite never trips the
# matplotlib ">20 open figures" warning (which filterwarnings=error promotes).

_HEX_MALE = "#202020"
_HEX_FEMALE = "#A83232"
_HEX_UNASSIGNED = "#7A7A7A"
_HEX_LINE = "#1A1A1A"


def _assignment_frame() -> pls.DataFrame:
    """
    Description
    -----------
    Build a small per-session assignment frame with the four columns the
    assignment plots read: `session`, `male`, `female`, `unassigned`
    (raw USV counts per category per session).

    Parameters
    ----------

    Returns
    -------
    df (pls.DataFrame)
        Five-session synthetic assignment frame.
    """

    return pls.DataFrame({
        "session": [f"s{i}" for i in range(5)],
        "male": [40, 55, 30, 62, 48],
        "female": [22, 18, 35, 27, 30],
        "unassigned": [8, 12, 5, 15, 10],
    })


def test_plot_assignment_stacked_bars_counts_and_proportions():
    """
    Description
    -----------
    `plot_assignment_stacked_bars` must render the horizontal stacked-bar
    chart in both the raw-count and the proportion modes, returning a
    Figure and a stats dict summarising the session totals.

    Parameters
    ----------

    Returns
    -------
    None
    """

    df = _assignment_frame()
    for plot_proportions in (False, True):
        fig, ax, stats = plot_assignment_stacked_bars(
            df, plot_proportions, _HEX_MALE, _HEX_FEMALE, _HEX_UNASSIGNED,
        )
        assert fig is not None
        assert stats["total_sessions"] == 5
        plt.close(fig)


def test_plot_assignment_summary_panel_three_panels():
    """
    Description
    -----------
    `plot_assignment_summary_panel` must render its three panels (scatter,
    violin, aggregate bar) and return per-category global medians / totals
    / proportions in the stats dict.

    Parameters
    ----------

    Returns
    -------
    None
    """

    fig, axes, stats = plot_assignment_summary_panel(
        _assignment_frame(), _HEX_MALE, _HEX_FEMALE, _HEX_UNASSIGNED, jitter_strength=0.05,
    )
    assert len(axes) == 3
    assert {"male_median", "female_median", "grand_total"}.issubset(stats)
    plt.close(fig)


def test_plot_animal_participation_stats():
    """
    Description
    -----------
    `plot_animal_participation_stats` must turn the nested per-animal
    `{session_count, total_usvs}` dict into the two-panel session-count /
    vocal-rate bar figure and report the animal-count summary.

    Parameters
    ----------

    Returns
    -------
    None
    """

    animal_stats = {
        "m1": {"session_count": 4, "total_usvs": 120},
        "m2": {"session_count": 2, "total_usvs": 30},
        "m3": {"session_count": 6, "total_usvs": 240},
    }
    fig, axes, stats = plot_animal_participation_stats(
        animal_stats, sex_label="Male", bar_color=_HEX_MALE, text_color=_HEX_LINE,
    )
    assert len(axes) == 2
    assert stats["total_animals"] == 3
    plt.close(fig)


def test_plot_polar_kde_distance_angle():
    """
    Description
    -----------
    `plot_polar_kde_distance_angle` must compute the raw / occupancy-
    normalised polar KDEs from the USV-moment and all-frame
    distance/angle arrays and return the two polar axes plus the point
    counts.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(0)
    usv_dist = rng.uniform(0, 25, 300)
    usv_ang = rng.uniform(-180, 180, 300)
    all_dist = rng.uniform(0, 25, 2000)
    all_ang = rng.uniform(-180, 180, 2000)
    fig, axes, stats = plot_polar_kde_distance_angle(
        usv_dist, usv_ang, all_dist, all_ang,
        max_distance=30.0, colormap="inferno", ylabel="distance (cm)",
        occupancy_threshold=1e-6,
    )
    assert len(axes) == 2
    assert stats["n_usv_points"] == 300
    plt.close(fig)


def test_plot_behavior_duration_regressions():
    """
    Description
    -----------
    `plot_behavior_duration_regressions` must render the 2x2 grid of
    distance/angle-vs-duration regressions for both sexes and return the
    Pearson statistics dict.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(1)

    def _df(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "distance": rng.uniform(0, 25, n),
            "angle": rng.uniform(-180, 180, n),
            "usv_duration": rng.uniform(0.02, 0.3, n),
        })

    fig, axes, stats = plot_behavior_duration_regressions(
        _df(60), _df(50), _HEX_MALE, _HEX_FEMALE, _HEX_LINE,
    )
    assert fig is not None
    assert isinstance(stats, dict)
    plt.close(fig)


_ESTROUS_ORDER = ["p", "e", "m", "d"]
_ESTROUS_LABELS = ["Proestrus", "Estrus", "Metestrus", "Diestrus"]
_ESTROUS_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231"]


def test_plot_distance_by_assignment_kde_anova():
    """
    Description
    -----------
    `plot_distance_by_assignment_kde_anova` must render the overlaid
    per-category distance KDEs and, with enough samples per group, run the
    one-way ANOVA + Tukey post-hoc, returning the ANOVA stats.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(2)
    rows = []
    for cat, loc in (("male", 8.0), ("female", 12.0), ("unassigned", 15.0)):
        for d in rng.normal(loc, 3.0, 40):
            rows.append({"distance": float(d), "category": cat})
    df_plot = pd.DataFrame(rows)
    fig, ax, stats = plot_distance_by_assignment_kde_anova(
        df_plot, min_samples_anova=5,
        male_color=_HEX_MALE, female_color=_HEX_FEMALE, unassigned_color=_HEX_UNASSIGNED,
    )
    assert fig is not None
    assert "anova" in stats
    plt.close(fig)


def test_plot_duration_histograms_by_sex():
    """
    Description
    -----------
    `plot_duration_histograms_by_sex` must render the stacked male / female
    duration histograms and return the per-sex mean / median durations.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(3)
    rows = []
    for sex, loc in (("male", 60.0), ("female", 90.0)):
        for d in rng.uniform(loc - 30, loc + 30, 80):
            rows.append({"sex": sex, "duration_ms": float(d)})
    plot_data = pd.DataFrame(rows)
    fig, axes, stats = plot_duration_histograms_by_sex(
        plot_data, bin_width_ms=10.0, max_duration_ms=200.0,
        male_color=_HEX_MALE, female_color=_HEX_FEMALE,
    )
    assert len(axes) == 2
    assert "male_mean" in stats and "female_mean" in stats
    plt.close(fig)


def test_plot_estrous_ratio_scatter():
    """
    Description
    -----------
    `plot_estrous_ratio_scatter` must render the jittered per-stage
    male/female ratio scatter and return per-stage descriptive stats
    (n, mean, sem, ...).

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(4)
    ratio_dict = {stage: rng.uniform(0.5, 3.0, 6).tolist() for stage in _ESTROUS_ORDER}
    fig, ax, stats = plot_estrous_ratio_scatter(
        ratio_dict, _ESTROUS_ORDER, _ESTROUS_LABELS, _ESTROUS_COLORS,
        line_color=_HEX_LINE, text_color=_HEX_LINE,
    )
    assert fig is not None
    assert all(stage in stats for stage in _ESTROUS_ORDER)
    plt.close(fig)


def test_plot_estrous_usv_rates():
    """
    Description
    -----------
    `plot_estrous_usv_rates` must render the side-by-side per-stage male /
    female USV-rate bars and return the per-stage rate stats.

    Parameters
    ----------

    Returns
    -------
    None
    """

    session_counts = {"p": 5, "e": 4, "m": 6, "d": 3}
    male_usv_counts = {"p": 120, "e": 90, "m": 200, "d": 60}
    female_usv_counts = {"p": 60, "e": 45, "m": 80, "d": 30}
    fig, axes, stats = plot_estrous_usv_rates(
        session_counts, male_usv_counts, female_usv_counts,
        _ESTROUS_ORDER, _ESTROUS_LABELS, _HEX_MALE, _HEX_FEMALE, _HEX_LINE,
    )
    assert len(axes) == 2
    assert stats["p"]["male_rate"] == 24.0
    plt.close(fig)


def test_plot_estrous_stage_pie_chart():
    """
    Description
    -----------
    `plot_estrous_stage_pie_chart` must render the estrous-stage session
    pie chart from the per-stage session counts and return per-stage
    proportions.

    Parameters
    ----------

    Returns
    -------
    None
    """

    session_counts = {"p": 5, "e": 4, "m": 6, "d": 3}
    label_map = dict(zip(_ESTROUS_ORDER, _ESTROUS_LABELS))
    fig, ax, stats = plot_estrous_stage_pie_chart(
        session_counts, label_map, _ESTROUS_COLORS,
    )
    assert fig is not None
    assert isinstance(stats, dict)
    plt.close(fig)


def test_plot_log_usv_interval_histograms():
    """
    Description
    -----------
    `plot_log_usv_interval_histograms` must overlay the per-sex log-interval
    histograms (normalised per sex) and return the per-sex counts and
    median intervals (in seconds).

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(6)
    df = pls.DataFrame({
        "sex": (["male"] * 100) + (["female"] * 80),
        "log_interval": np.concatenate([
            rng.normal(0.0, 1.0, 100), rng.normal(0.5, 1.0, 80),
        ]).tolist(),
    })
    fig, ax, stats = plot_log_usv_interval_histograms(
        df, bins=30, male_color=_HEX_MALE, female_color=_HEX_FEMALE,
    )
    assert stats["n_M"] == 100 and stats["n_F"] == 80
    plt.close(fig)


def test_plot_ic_curves():
    """
    Description
    -----------
    `plot_ic_curves` must plot the per-sex minimum information-criterion
    curve vs. n_comp on twin y-axes and return the per-sex min-IC-per-K
    summary, optionally highlighting the bootstrap-LRT-selected K.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(7)
    rows = []
    for sex in ("male", "female"):
        for n_comp in range(1, 6):
            for _rep in range(3):
                rows.append({
                    "sex": sex,
                    "n_comp": n_comp,
                    "bic": float(1000.0 - 30.0 * n_comp + rng.normal(0, 5)),
                })
    df_results = pls.DataFrame(rows)
    fig, axes, stats = plot_ic_curves(
        df_results, _HEX_MALE, _HEX_FEMALE,
        selected_n_components={"male": 3, "female": 2},
    )
    assert len(axes) == 2
    assert "male" in stats and "female" in stats
    plt.close(fig)


def test_plot_qq():
    """
    Description
    -----------
    `plot_qq` must draw the empirical-vs-model quantile plot by inverting
    the fitted log-space mixture model CDF and return the log-space Pearson r
    goodness-of-fit.

    Parameters
    ----------

    Returns
    -------
    None
    """

    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(8)
    intervals_sec = np.exp(rng.normal(0.0, 1.0, 300))
    mixture_model = GaussianMixture(n_components=2, random_state=0).fit(
        np.log(intervals_sec).reshape(-1, 1)
    )
    fig, ax, stats = plot_qq(intervals_sec, mixture_model, dot_color=_HEX_MALE)
    assert "pearson_r" in stats
    plt.close(fig)


def _category_estrous_data(rng) -> dict:
    """
    Description
    -----------
    Build the nested per-category estrous_data structure consumed by both
    plot_category_estrous_rates_grid (session_counts / male_usv_counts /
    female_usv_counts) and plot_category_estrous_ratio_grid
    (male_female_ratios), so a single fixture drives both.

    Parameters
    ----------
    rng (np.random.Generator)
        Source of the synthetic per-stage ratio lists.

    Returns
    -------
    estrous_data (dict)
        Mapping category-id -> per-stage count / ratio sub-dicts.
    """

    data = {}
    for cat in (1, 2, 3, 4):
        data[cat] = {
            "session_counts": {s: 4 + i for i, s in enumerate(_ESTROUS_ORDER)},
            "male_usv_counts": {s: 100 + 10 * i for i, s in enumerate(_ESTROUS_ORDER)},
            "female_usv_counts": {s: 50 + 5 * i for i, s in enumerate(_ESTROUS_ORDER)},
            "male_female_ratios": {s: rng.uniform(0.5, 3.0, 6).tolist() for s in _ESTROUS_ORDER},
        }
    return data


def test_plot_category_global_fatigue_heatmap():
    """
    Description
    -----------
    `plot_category_global_fatigue_heatmap` must bin the day into 2-hour
    blocks, build per-sex category-by-time heatmaps (smoothed + row-
    normalised), and return the processed pivot tables.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(9)
    rows = []
    for sess in range(3):
        for sex in ("male", "female"):
            for cat in (1, 2, 3):
                for hour in (8, 10, 12, 14, 16, 18):
                    for _ in range(int(rng.integers(1, 5))):
                        rows.append({
                            "session_id": f"s{sess}", "sex": sex,
                            "category": cat, "hour": hour,
                        })
    global_usv_df = pls.DataFrame(rows)
    fig, axes, stats = plot_category_global_fatigue_heatmap(
        global_usv_df, smoothing_sigma=0.75, colormap="inferno",
    )
    assert len(axes) == 2
    assert isinstance(stats, dict)
    plt.close(fig)


def test_plot_category_estrous_rates_grid():
    """
    Description
    -----------
    `plot_category_estrous_rates_grid` must render the per-category facet
    grid of dual-axis male/female USV-rate bars and return per-category
    mean rates.

    Parameters
    ----------

    Returns
    -------
    None
    """

    estrous_data = _category_estrous_data(np.random.default_rng(10))
    fig, axes, stats = plot_category_estrous_rates_grid(
        estrous_data, _ESTROUS_ORDER, _HEX_MALE, _HEX_FEMALE,
    )
    assert set(stats) == set(estrous_data)
    plt.close(fig)


def test_plot_category_estrous_ratio_grid():
    """
    Description
    -----------
    `plot_category_estrous_ratio_grid` must render its two figures (the
    per-category facet grid and the per-stage view) from the per-stage
    male/female ratio lists and return the per-category-stage stats.

    Parameters
    ----------

    Returns
    -------
    None
    """

    estrous_data = _category_estrous_data(np.random.default_rng(11))
    (figs, axes, stats) = plot_category_estrous_ratio_grid(
        estrous_data, _ESTROUS_ORDER, _ESTROUS_COLORS,
    )
    assert len(figs) == 2
    for f in figs:
        plt.close(f)


def test_plot_unassigned_proportion_vs_distance_jointplot():
    """
    Description
    -----------
    `plot_unassigned_proportion_vs_distance_jointplot` must render the
    Seaborn JointGrid of per-session median distance vs. unassigned-call
    proportion and return the Pearson r / p.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(12)
    dist = rng.uniform(2.0, 8.0, 25)
    df_combined = pd.DataFrame({
        "median_distance": dist,
        "unassigned_prop": np.clip(0.05 + 0.02 * dist + rng.normal(0, 0.02, 25), 0, 1),
    })
    g, stats = plot_unassigned_proportion_vs_distance_jointplot(
        df_combined, _HEX_MALE, _HEX_LINE, _HEX_FEMALE,
    )
    assert "pearson_r" in stats
    plt.close(g.figure)


def test_plot_hourly_regressions():
    """
    Description
    -----------
    `plot_hourly_regressions` must render the per-sex hour-vs-metric
    regression panels and return the per-sex Pearson r / p.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(13)
    rows = []
    for sex in ("male", "female"):
        for _ in range(60):
            hour = int(rng.integers(12, 23))
            rows.append({"hour": hour, "sex": sex, "usv_count": float(hour * 2 + rng.normal(0, 5))})
    df_raw = pd.DataFrame(rows)
    fig, axes, stats = plot_hourly_regressions(
        df_raw, y_col="usv_count", y_label="USVs/session",
        male_color=_HEX_MALE, female_color=_HEX_FEMALE, line_color=_HEX_LINE,
    )
    assert len(axes) == 2
    assert "male_r" in stats and "female_r" in stats
    plt.close(fig)


def test_plot_local_fatigue_binned_trends():
    """
    Description
    -----------
    `plot_local_fatigue_binned_trends` must render the per-sex binned
    mean +/- SEM fatigue trend lines and return the global min/max.

    Parameters
    ----------

    Returns
    -------
    None
    """

    n_bins = 6
    rng = np.random.default_rng(14)
    rows = []
    for sex in ("male", "female"):
        for b in range(n_bins):
            rows.append({
                "sex": sex,
                "time_bin": b,
                "mean_rate": float(10.0 - b + rng.normal(0, 0.5)),
                "sem_rate": float(abs(rng.normal(0.5, 0.1))),
            })
    binned_df = pd.DataFrame(rows)
    fig, axes, stats = plot_local_fatigue_binned_trends(
        binned_df, y_mean_col="mean_rate", y_sem_col="sem_rate", y_label="rate",
        bin_width_seconds=300, n_bins=n_bins,
        male_color=_HEX_MALE, female_color=_HEX_FEMALE, use_log_scale=False,
    )
    assert len(axes) == 2
    assert "global_max" in stats
    plt.close(fig)


def test_plot_best_fit_with_annotations():
    """
    Description
    -----------
    `plot_best_fit_with_annotations` must render the best-fit mixture model density
    over the log-interval histogram, annotate each component mean, draw the
    Q-Q inset, and return the mixture model summary (incl. the inset's log-log
    Pearson r). Driven by a directly-fitted sklearn GaussianMixture.

    Parameters
    ----------

    Returns
    -------
    None
    """

    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(15)
    intervals_sec = np.exp(np.concatenate([
        rng.normal(-1.0, 0.5, 200), rng.normal(1.0, 0.5, 200),
    ]))
    mixture_model = GaussianMixture(n_components=2, random_state=0).fit(
        np.log(intervals_sec).reshape(-1, 1)
    )
    mixture_model_order = np.argsort(mixture_model.means_.ravel())
    fig, ax, summary = plot_best_fit_with_annotations(
        intervals_sec, mixture_model, mixture_model_order, color=_HEX_MALE,
    )
    assert "qq_pearson_r" in summary
    plt.close(fig)


def test_fit_mixture_model_sweep_feeds_plot_ic_curves():
    """
    Description
    -----------
    The CLI's `fit_mixture_model_sweep` must fit the mixture-model sweep across
    n_components for each sex and return a tidy results table; that table must then
    drive `plot_ic_curves` end-to-end (real compute -> real figure) with no mocks.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(16)
    df_results = fit_mixture_model_sweep(
        intervals_by_key={
            "male": np.exp(rng.normal(-0.5, 0.6, 200)),
            "female": np.exp(rng.normal(0.6, 0.6, 200)),
        },
        n_components_min=1, n_components_max=3, n_repeats=2,
        max_modes_reported=3, random_seed_base=0, model_class="gauss",
    )
    assert df_results.height > 0
    assert {"sex", "n_comp", "bic"}.issubset(df_results.columns)

    fig, axes, stats = plot_ic_curves(df_results, _HEX_MALE, _HEX_FEMALE)
    assert len(axes) == 2
    plt.close(fig)


# ===========================================================================
# usv_summary_statistics — category-level grid / embedding figures
#
# These four figure functions were previously uncovered. Each consumes a
# synthetic frame / nested-dict fixture spread densely enough to keep the
# internal gaussian_kde / griddata calls non-singular, and asserts the
# returned figure + stats structure. Error-path branches (bad sex_key,
# missing columns, empty categories, insufficient background) are exercised
# separately so the guardrails stay covered.
# ===========================================================================


def _embedding_frame() -> pls.DataFrame:
    """
    Description
    -----------
    Build a synthetic `extract_category_embedding_data`-style frame for
    `plot_category_prevalence_and_embedding`: `dim1` / `dim2` (spread 2D
    embedding coords), `category` (>= 2 ids so the territorial-boundary
    contour has levels), and `sex` (each of male / female / unassigned
    given > 3 points so every per-sex KDE branch runs).

    Parameters
    ----------

    Returns
    -------
    df (pls.DataFrame)
        Embedding frame with `dim1`, `dim2`, `category`, `sex`.
    """

    rng = np.random.default_rng(101)
    per_sex = 25
    sexes = (["male"] * per_sex) + (["female"] * per_sex) + (["unassigned"] * per_sex)
    n = len(sexes)
    return pls.DataFrame({
        "dim1":     rng.normal(0.0, 1.0, n),
        "dim2":     rng.normal(0.0, 1.0, n),
        "category": rng.integers(1, 4, n),
        "sex":      sexes,
    })


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_category_local_fatigue_heatmap_both_sexes():
    """
    Description
    -----------
    `plot_category_local_fatigue_heatmap` must build per-sex smoothed,
    row-normalised category-by-time-bin heatmaps and return the per-sex
    peak raw rates. Drives the main (smoothed) path with both sexes
    populated.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(102)
    n_bins = 6
    rows = []
    for sess in range(3):
        for sex in ("male", "female"):
            for cat in (1, 2, 3):
                for tb in range(n_bins):
                    rows.append({
                        "session_id": f"s{sess}", "sex": sex,
                        "category": cat, "time_bin": tb,
                        "usv_count": float(rng.integers(0, 12)),
                    })
    binned_df = pd.DataFrame(rows)
    fig, axes, stats = plot_category_local_fatigue_heatmap(
        binned_df, bin_width_seconds=120, n_bins=n_bins, smoothing_sigma=0.75,
    )
    assert len(axes) == 2
    assert "male_peaks" in stats and "female_peaks" in stats
    assert stats["male_peaks"], "expected per-category male peak rates"
    plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_category_local_fatigue_heatmap_empty_sex_and_no_smoothing():
    """
    Description
    -----------
    With one sex entirely absent, the empty-subset branch must render its
    "No data available" placeholder rather than crash; `smoothing_sigma=0`
    additionally takes the un-smoothed normalisation path. Asserts a
    figure is still returned with both axes.

    Parameters
    ----------

    Returns
    -------
    None
    """

    n_bins = 5
    rows = []
    for cat in (1, 2):
        for tb in range(n_bins):
            rows.append({
                "session_id": "s0", "sex": "male",
                "category": cat, "time_bin": tb, "usv_count": float(tb + 1),
            })
    binned_df = pd.DataFrame(rows)  # no female rows
    fig, axes, stats = plot_category_local_fatigue_heatmap(
        binned_df, bin_width_seconds=120, n_bins=n_bins, smoothing_sigma=0.0,
    )
    assert len(axes) == 2
    assert stats["female_peaks"] == {}, "female branch should stay empty"
    plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("plot_type", ["density", "scatter"])
def test_plot_category_prevalence_and_embedding(plot_type):
    """
    Description
    -----------
    `plot_category_prevalence_and_embedding` must render the 4x2 grid
    (per-assignment prevalence bars + embedding maps, plus the global
    summary row) for both the density and scatter rendering modes, with
    global territorial boundaries overlaid. Asserts the 4x2 axes grid.

    Parameters
    ----------
    plot_type (str)
        Embedding rendering mode under test.

    Returns
    -------
    None
    """

    df_embedding = _embedding_frame()
    fig, axes = plot_category_prevalence_and_embedding(
        df_embedding,
        male_color=_HEX_MALE, female_color=_HEX_FEMALE,
        unassigned_color=_HEX_UNASSIGNED,
        plot_type=plot_type, log_scale_bars=True, grid_res=25,
    )
    assert axes.shape == (4, 2)
    plt.close(fig)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_category_prevalence_and_embedding_non_contiguous_category_ids():
    """Territorial boundaries must render for NON-CONTIGUOUS category IDs (e.g.
    {0, 5, 11}). The boundary contour interpolates ordinal CODES so the integer
    `arange(N+1)-0.5` levels still land between adjacent categories; the old
    raw-value interpolation tied the levels to the category count, so boundaries
    between non-contiguous IDs were mis-placed or omitted. Exercises that path."""
    rng = np.random.default_rng(7)
    per_cat = 25
    # Three well-separated clusters with non-contiguous category IDs.
    centers = {0: (-4.0, -4.0), 5: (0.0, 4.0), 11: (4.0, -4.0)}
    dim1, dim2, cats, sexes = [], [], [], []
    for cat, (cx, cy) in centers.items():
        dim1.extend(rng.normal(cx, 0.4, per_cat))
        dim2.extend(rng.normal(cy, 0.4, per_cat))
        cats.extend([cat] * per_cat)
        sexes.extend((["male"] * 9) + (["female"] * 9) + (["unassigned"] * 7))
    df_embedding = pls.DataFrame(
        {"dim1": dim1, "dim2": dim2, "category": cats, "sex": sexes}
    )
    fig, axes = plot_category_prevalence_and_embedding(
        df_embedding,
        male_color=_HEX_MALE, female_color=_HEX_FEMALE,
        unassigned_color=_HEX_UNASSIGNED,
        plot_type="density", log_scale_bars=True, grid_res=25,
    )
    assert axes.shape == (4, 2)
    plt.close(fig)


def test_plot_category_prevalence_and_embedding_bad_plot_type_raises():
    """
    Description
    -----------
    An unsupported `plot_type` must raise ValueError before any rendering.

    Parameters
    ----------

    Returns
    -------
    None
    """

    with pytest.raises(ValueError, match="plot_type must be"):
        plot_category_prevalence_and_embedding(
            _embedding_frame(), _HEX_MALE, _HEX_FEMALE, _HEX_UNASSIGNED,
            plot_type="banana",
        )


def _polar_metrics(rng, *, sparse_last: bool = True) -> dict:
    """
    Description
    -----------
    Build the nested `global_behavior_metrics` dict consumed by
    `plot_category_polar_kde_grid`: an `'all_frames'` background block of
    distance / mf_angle / fm_angle arrays, plus four category blocks each
    carrying male / female distance+angle arrays. The last category is
    given only 2 points (below any sensible `threshold`) so the
    insufficient-data placeholder branch renders.

    Parameters
    ----------
    rng (np.random.Generator)
        Source of synthetic coordinates.
    sparse_last (bool)
        When True, the 4th category is sparse (triggers the "Insufficient
        Data" branch); when False all categories are dense.

    Returns
    -------
    metrics (dict)
        Nested behavior-metrics dict.
    """

    def _block(k):
        """Build one distance / angle metrics block of ``k`` synthetic points
        (distance plus the male- and female-referenced angles)."""
        return {
            "distance":  rng.uniform(1.0, 18.0, k),
            "mf_angle":  rng.uniform(-180.0, 180.0, k),
            "fm_angle":  rng.uniform(-180.0, 180.0, k),
        }

    metrics = {"all_frames": _block(300)}
    for idx, cat in enumerate(("1", "2", "3", "4")):
        k = 2 if (sparse_last and cat == "4") else 60
        metrics[cat] = {"male": _block(k), "female": _block(k)}
    return metrics


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_category_polar_kde_grid_dense_and_sparse():
    """
    Description
    -----------
    `plot_category_polar_kde_grid` must occupancy-normalise per-category
    USV density against the background, render the half-circle polar
    small-multiples, blank out the trailing unused axes, and (for the
    sparse 4th category) draw the "Insufficient Data" placeholder.
    Asserts the returned `global_vmax` scaling scalar.

    Parameters
    ----------

    Returns
    -------
    None
    """

    metrics = _polar_metrics(np.random.default_rng(103))
    fig, axes, stats = plot_category_polar_kde_grid(
        metrics, sex_key="male", max_distance=20.0, threshold=10,
        colormap="inferno",
    )
    assert "global_vmax" in stats and stats["sex_plotted"] == "male"
    plt.close(fig)


def test_plot_category_polar_kde_grid_error_branches():
    """
    Description
    -----------
    The three guardrails of `plot_category_polar_kde_grid` must each
    raise ValueError: an invalid `sex_key`, a metrics dict with no
    categories (only `all_frames`), and insufficient background tracking
    data.

    Parameters
    ----------

    Returns
    -------
    None
    """

    rng = np.random.default_rng(104)
    metrics = _polar_metrics(rng)

    with pytest.raises(ValueError, match="sex_key must be"):
        plot_category_polar_kde_grid(
            metrics, sex_key="other", max_distance=20.0, threshold=10,
            colormap="inferno",
        )

    with pytest.raises(ValueError, match="No USV categories"):
        plot_category_polar_kde_grid(
            {"all_frames": metrics["all_frames"]}, sex_key="male",
            max_distance=20.0, threshold=10, colormap="inferno",
        )

    thin_bg = {
        "all_frames": {
            "distance": np.array([1.0, 2.0]),
            "mf_angle": np.array([10.0, 20.0]),
            "fm_angle": np.array([10.0, 20.0]),
        },
        "1": metrics["1"],
    }
    with pytest.raises(ValueError, match="Insufficient background"):
        plot_category_polar_kde_grid(
            thin_bg, sex_key="male", max_distance=20.0, threshold=10,
            colormap="inferno",
        )


def _estrous_kde_frames(rng):
    """
    Description
    -----------
    Build the `(usv_pls, bg_pls)` polars pair consumed by
    `plot_estrous_category_kde_grid`: a per-(category, stage) USV frame
    (`category`, `estrous_stage`, `sex`, `distance`, `mf_angle`,
    `fm_angle`) with one deliberately-sparse cell, and a background
    occupancy frame (`distance`, `mf_angle`, `fm_angle`). Two categories
    x two stages keeps the axes grid 2D without hitting the single-row /
    single-column reshape branches.

    Parameters
    ----------
    rng (np.random.Generator)
        Source of synthetic coordinates.

    Returns
    -------
    usv_pls, bg_pls (tuple[pls.DataFrame, pls.DataFrame])
        The USV and background frames.
    """

    cats = (1, 2)
    stages = ("p", "e")
    rows = []
    for cat in cats:
        for stage in stages:
            n_pts = 5 if (cat, stage) == (2, "e") else 45
            for _ in range(n_pts):
                rows.append({
                    "category":      cat,
                    "estrous_stage": stage,
                    "sex":           "male",
                    "distance":      float(rng.uniform(1.0, 18.0)),
                    "mf_angle":      float(rng.uniform(-180.0, 180.0)),
                    "fm_angle":      float(rng.uniform(-180.0, 180.0)),
                })
    usv_pls = pls.DataFrame(rows)
    bg_pls = pls.DataFrame({
        "distance": rng.uniform(1.0, 18.0, 200),
        "mf_angle": rng.uniform(-180.0, 180.0, 200),
        "fm_angle": rng.uniform(-180.0, 180.0, 200),
    })
    return usv_pls, bg_pls


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_estrous_category_kde_grid_dense_and_sparse():
    """
    Description
    -----------
    `plot_estrous_category_kde_grid` must build the category x stage polar
    KDE grid against a shared background occupancy KDE, render dense cells
    as filled contours and the sparse `(2, 'e')` cell as "N/A", subsample
    the background when it exceeds `max_kde_points`, and return the
    per-cell point counts. Asserts the `n_points` map and grid shape.

    Parameters
    ----------

    Returns
    -------
    None
    """

    usv_pls, bg_pls = _estrous_kde_frames(np.random.default_rng(105))
    fig, axes, stats = plot_estrous_category_kde_grid(
        usv_pls, bg_pls, sex_key="male",
        valid_stages=["p", "e"],
        stage_label_map={"p": "Proestrus", "e": "Estrus"},
        max_distance=20.0, occupancy_threshold=0.01, colormap="inferno",
        threshold=30, max_kde_points=50,
    )
    assert axes.shape == (2, 2)
    assert stats["n_points"][(2, "e")] == 5, "sparse cell point count wrong"
    assert stats["sex_plotted"] == "male"
    plt.close(fig)


def test_plot_estrous_category_kde_grid_error_branches():
    """
    Description
    -----------
    The guardrails of `plot_estrous_category_kde_grid` must each raise
    ValueError: an invalid `sex_key`, and a `usv_pls` missing the required
    `estrous_stage` column.

    Parameters
    ----------

    Returns
    -------
    None
    """

    usv_pls, bg_pls = _estrous_kde_frames(np.random.default_rng(106))

    with pytest.raises(ValueError, match="sex_key must be"):
        plot_estrous_category_kde_grid(
            usv_pls, bg_pls, sex_key="nope", valid_stages=["p", "e"],
            stage_label_map={"p": "Proestrus", "e": "Estrus"},
            max_distance=20.0, occupancy_threshold=0.01, colormap="inferno",
        )

    with pytest.raises(ValueError, match="estrous_stage"):
        plot_estrous_category_kde_grid(
            usv_pls.drop("estrous_stage"), bg_pls, sex_key="male",
            valid_stages=["p", "e"],
            stage_label_map={"p": "Proestrus", "e": "Estrus"},
            max_distance=20.0, occupancy_threshold=0.01, colormap="inferno",
        )

    # No categories survive (all filtered out) -> guardrail before bg KDE.
    with pytest.raises(ValueError, match="No USV categories"):
        plot_estrous_category_kde_grid(
            usv_pls.filter(pls.col("category") == 999), bg_pls, sex_key="male",
            valid_stages=["p", "e"],
            stage_label_map={"p": "Proestrus", "e": "Estrus"},
            max_distance=20.0, occupancy_threshold=0.01, colormap="inferno",
        )

    # Too few valid background frames to fit the shared occupancy KDE.
    thin_bg = pls.DataFrame({
        "distance": [1.0, 2.0, 3.0],
        "mf_angle": [10.0, 20.0, 30.0],
        "fm_angle": [10.0, 20.0, 30.0],
    })
    with pytest.raises(ValueError, match="Insufficient background"):
        plot_estrous_category_kde_grid(
            usv_pls, thin_bg, sex_key="male", valid_stages=["p", "e"],
            stage_label_map={"p": "Proestrus", "e": "Estrus"},
            max_distance=20.0, occupancy_threshold=0.01, colormap="inferno",
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "cats, stages, want_shape",
    [
        ((1,), ("p", "e"), (1, 2)),   # single category -> row reshape branch
        ((1, 2), ("p",),  (2, 1)),    # single stage    -> column reshape branch
    ],
)
def test_plot_estrous_category_kde_grid_degenerate_grid_shapes(cats, stages, want_shape):
    """
    Description
    -----------
    The axes-normalisation branches must coerce a 1xN (single category) or
    Nx1 (single stage) subplot result back to a 2D array so the
    `axes[r, c]` indexing in the render loop stays valid. Asserts the
    returned axes grid has the expected 2D shape for each degenerate case.

    Parameters
    ----------
    cats (tuple[int, ...])
        Category ids to populate.
    stages (tuple[str, ...])
        Estrous stages to populate.
    want_shape (tuple[int, int])
        Expected `axes.shape`.

    Returns
    -------
    None
    """

    rng = np.random.default_rng(107)
    rows = []
    for cat in cats:
        for stage in stages:
            for _ in range(45):
                rows.append({
                    "category":      cat,
                    "estrous_stage": stage,
                    "sex":           "male",
                    "distance":      float(rng.uniform(1.0, 18.0)),
                    "mf_angle":      float(rng.uniform(-180.0, 180.0)),
                    "fm_angle":      float(rng.uniform(-180.0, 180.0)),
                })
    usv_pls = pls.DataFrame(rows)
    bg_pls = pls.DataFrame({
        "distance": rng.uniform(1.0, 18.0, 200),
        "mf_angle": rng.uniform(-180.0, 180.0, 200),
        "fm_angle": rng.uniform(-180.0, 180.0, 200),
    })
    fig, axes, _stats = plot_estrous_category_kde_grid(
        usv_pls, bg_pls, sex_key="male", valid_stages=list(stages),
        stage_label_map={s: s.upper() for s in stages},
        max_distance=20.0, occupancy_threshold=0.01, colormap="inferno",
        threshold=30,
    )
    assert axes.shape == want_shape
    plt.close(fig)


# ===========================================================================
# usv_interval_summary_statistics — bootstrap-LRT sweep, panel, and the
# notebook -> HDF5 archive writer.
#
# The panel / cell-pair / step-up / archive paths consume hand-built sweep
# dicts (shaped exactly like load_lrt_sweep_from_h5 / bootstrap_lrt output)
# so the broken-axis, normal, and empty-fill rendering branches are all
# driven without paying the bootstrap cost; one small real sweep covers the
# compute loop and its size<2 skip.
# ===========================================================================

def _lrt_res(K_n, K_a, lr_obs, lr_null, p_value):
    """
    Description
    -----------
    Build one `(K_null, K_alt)` bootstrap-LRT result dict shaped exactly
    like `mixture_model_utils.bootstrap_lrt` returns it, so the same dict
    drives `plot_bootstrap_lrt_panel`.

    Parameters
    ----------
    K_n (int)
        Null-model component count.
    K_a (int)
        Alternative-model component count.
    lr_obs (float)
        Observed likelihood-ratio statistic.
    lr_null (array-like)
        Bootstrap null LR draws.
    p_value (float)
        Bootstrap p-value.

    Returns
    -------
    res (dict)
        Result dict with the full key set the consumers read.
    """

    lr_null = np.asarray(lr_null, dtype=float)
    return {
        "K_null":      int(K_n),
        "K_alt":       int(K_a),
        "lr_obs":      float(lr_obs),
        "lr_null":     lr_null,
        "null_mean":   float(lr_null.mean()),
        "null_p95":    float(np.quantile(lr_null, 0.95)),
        "null_max":    float(lr_null.max()),
        "p_value":     float(p_value),
        "B":           int(lr_null.size),
        "n_subsample": 100,
        "model_class": "gauss",
    }


def _lrt_sweep() -> dict:
    """
    Description
    -----------
    Hand-build a two-sex bootstrap-LRT sweep that exercises every panel
    rendering branch: `male` has a far-out `LR_obs` (triggers the broken
    axis + `_draw_lrt_cell_pair`) and a non-significant in-bulk pair;
    `female` has a single significant pair (one fewer column, so the
    panel's empty-column fill branch runs).

    Parameters
    ----------

    Returns
    -------
    sweep (dict)
        `key -> {(K_null, K_alt) -> result_dict}`.
    """

    rng = np.random.default_rng(200)
    null_bulk = rng.uniform(0.0, 2.0, 40)
    return {
        "male": {
            (1, 2): _lrt_res(1, 2, 25.0, null_bulk, 0.004),  # far out -> break
            (2, 3): _lrt_res(2, 3, 1.4, null_bulk, 0.42),    # in-bulk  -> normal
        },
        "female": {
            (1, 2): _lrt_res(1, 2, 3.2, null_bulk, 0.03),    # near bulk -> normal
        },
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_bootstrap_lrt_panel_broken_normal_and_empty_fill():
    """
    Description
    -----------
    `plot_bootstrap_lrt_panel` must render one row per key and one column
    per K-pair: the far-out male `(1,2)` cell becomes a broken-axis
    `(ax_left, ax_right)` tuple (via `_draw_lrt_cell_pair`), the in-bulk
    cells stay single Axes, and female's missing second column is filled
    with a turned-off placeholder axis. Asserts the `(2, 2)` axes grid and
    that exactly the broken cell is a 2-tuple.

    Parameters
    ----------

    Returns
    -------
    None
    """

    fig, axes = plot_bootstrap_lrt_panel(_lrt_sweep())
    assert axes.shape == (2, 2)
    assert isinstance(axes[0, 0], tuple) and len(axes[0, 0]) == 2, (
        "far-out LR_obs cell should be rendered as a broken-axis pair"
    )
    assert not isinstance(axes[0, 1], tuple), "in-bulk cell should be a single Axes"
    plt.close(fig)


def test_plot_bootstrap_lrt_panel_empty_sweep_returns_single_axis():
    """
    Description
    -----------
    An empty sweep dict must short-circuit to a single 1x1 placeholder
    figure rather than crash on `max()` over no pairs.

    Parameters
    ----------

    Returns
    -------
    None
    """

    fig, axes = plot_bootstrap_lrt_panel({})
    assert axes.shape == (1, 1)
    plt.close(fig)


def _fit_two_comp_gmm(seed: int):
    """
    Description
    -----------
    Fit a 2-component sklearn `GaussianMixture` to a bimodal synthetic
    log-interval sample and return `(intervals_sec, mixture_model, mixture_model_order)` ready
    for `plot_best_fit_with_annotations`.

    Parameters
    ----------
    seed (int)
        RNG seed for the synthetic sample.

    Returns
    -------
    intervals_sec, mixture_model, mixture_model_order (tuple)
        Positive intervals, the fitted mixture, and the ascending-log-mean
        component order.
    """

    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    intervals_sec = np.exp(np.concatenate([
        rng.normal(-1.0, 0.5, 200), rng.normal(1.0, 0.5, 200),
    ]))
    mixture_model = GaussianMixture(n_components=2, random_state=0).fit(
        np.log(intervals_sec).reshape(-1, 1)
    )
    mixture_model_order = np.argsort(mixture_model.means_.ravel())
    return intervals_sec, mixture_model, mixture_model_order


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("corner", ["upper left", "lower right", "lower left"])
def test_plot_best_fit_with_annotations_corners_auto_inset_and_components(corner):
    """
    Description
    -----------
    Exercise the optional rendering paths of
    `plot_best_fit_with_annotations` the default-args test does not reach:
    each non-default legend corner (driving the corner branches of
    `_draw_text_legend`), the `auto_inset_below_legend=True` path (which
    measures the rendered legend bbox and places the Q-Q inset beneath it),
    and `show_components=True` (overlaying the per-component shapes via
    `_draw_mixture_components`).

    Parameters
    ----------
    corner (str)
        Legend corner under test.

    Returns
    -------
    None
    """

    intervals_sec, mixture_model, mixture_model_order = _fit_two_comp_gmm(seed=21)
    fig, ax, summary = plot_best_fit_with_annotations(
        intervals_sec, mixture_model, mixture_model_order, color=_HEX_MALE,
        legend_corner=corner,
        auto_inset_below_legend=True,
        show_components=True,
    )
    assert "qq_pearson_r" in summary
    plt.close(fig)


def test_plot_best_fit_with_annotations_invalid_corner_raises():
    """
    Description
    -----------
    An unknown `legend_corner` must propagate the `_draw_text_legend`
    ValueError rather than silently mis-placing the legend.

    Parameters
    ----------

    Returns
    -------
    None
    """

    intervals_sec, mixture_model, mixture_model_order = _fit_two_comp_gmm(seed=22)
    with pytest.raises(ValueError, match="unknown corner"):
        plot_best_fit_with_annotations(
            intervals_sec, mixture_model, mixture_model_order, color=_HEX_MALE,
            legend_corner="middle",
        )


def _serial_dependence_tables():
    """Archived-shape serial-dependence tables for one male USV pool: a curve rising to a flat
    level near 90 ms, bands around it, a bend at 110 ms, and five replicate bends."""
    grid = np.geomspace(30.0, 20000.0, 40)
    line = np.minimum(grid, 110.0) * 0.8
    identity = {"sex": ["male"] * grid.size, "call_type": ["usv"] * grid.size,
                "adjacency": ["filtered"] * grid.size}
    curves = pls.DataFrame({**identity, "current_ms": grid, "spline_ms": line,
                            "spline_low_ms": line * 0.95, "spline_high_ms": line * 1.05,
                            "bent_ms": line, "bent_low_ms": line * 0.97, "bent_high_ms": line * 1.03})
    fit = pls.DataFrame([{"sex": "male", "call_type": "usv", "adjacency": "filtered", "n_pairs": 500,
                          "n_sessions": 10, "n_knots": 6, "corner_width": 0.05, "bend_ms": 110.0,
                          "bend_low_ms": 100.0, "bend_high_ms": 125.0, "slope": 0.55,
                          "flat_level_ms": 88.0, "level": 99.0, "n_bootstrap": 5,
                          "spline_not_converged": 0, "bent_not_converged": 0, "loss_w_0p05": 1.0}])
    bends = pls.DataFrame({"sex": ["male"] * 5, "call_type": ["usv"] * 5, "adjacency": ["filtered"] * 5,
                           "b": np.arange(5), "bend_ms": [105.0, 108.0, 110.0, 112.0, 120.0]})
    return curves, fit, bends


def test_load_serial_dependence_from_h5_round_trips_one_pool(tmp_path):
    """The three serial-dependence tables survive the archive; the loader returns the pool's curves
    ascending in the current interval and its fit row, and refuses a pool with no fit."""
    curves, fit, bends = _serial_dependence_tables()
    out = tmp_path / "usv_interval_analysis_20260101_120000.h5"
    write_ivi_h5(out, analysis_attrs={"git_sha": "abc"},
                 per_mode={"e2s": {"attrs": {}, "serial_dependence_curves": curves.reverse(),
                                   "serial_dependence_fit": fit, "serial_dependence_bends": bends}})
    loaded_curves, loaded_fit = load_serial_dependence_from_h5(out, "e2s", "male", "usv")
    np.testing.assert_allclose(loaded_curves["current_ms"].to_numpy(), curves["current_ms"].to_numpy())
    assert loaded_fit["bend_ms"] == 110.0 and loaded_fit["level"] == 99.0
    with pytest.raises(KeyError, match="no serial-dependence fit"):
        load_serial_dependence_from_h5(out, "e2s", "female", "usv")


def test_plot_serial_dependence_draws_spline_and_bent_panels():
    """Two square panels share the pairs; the right one carries the bend with its interval and the
    single colorbar, and the stats echo the fit."""
    curves, fit, _ = _serial_dependence_tables()
    rng = np.random.default_rng(0)
    current = np.exp(rng.uniform(np.log(0.03), np.log(5.0), 500))
    following = np.exp(rng.uniform(np.log(0.03), np.log(5.0), 500))
    f, axes, stats = plot_serial_dependence((current, following), "#4C9BB5", curves,
                                            fit.row(0, named=True), boundary_ms=115.5)
    assert len(axes) == 2 and stats["n_pairs"] == 500
    assert stats["bend_ms"] == 110.0 and stats["bend_high_ms"] == 125.0
    assert "bend 110 ms (100-125, 99%)" in [t.get_text() for t in axes[1].get_legend().get_texts()]
    assert axes[0].get_title() == "A: median spline" and axes[1].get_title() == "B: bent-line median fit"
    # the one colorbar is an inset of the right panel, none on the left
    assert len(axes[1].child_axes) == 1 and len(axes[0].child_axes) == 0
    plt.close(f)


def _tied_archive(tmp_path):
    """An archive holding a male USV tied ladder at 1 and 2 peaks (plus 2 background components
    each), the session-corrected peak test for both rungs with their null draws, and the
    selected peak count (2) in the mode attributes."""
    rows = []
    for n_peak, loglik, bic in ((1, -100.0, 230.0), (2, -90.0, 215.0)):
        roles = ["peak"] * n_peak + ["background"] * 2
        means = [-2.8, -1.7][:n_peak] + [-1.2, 2.4]
        scales = [0.24] * n_peak + [1.2, 1.1]
        for component, (role, mean, scale) in enumerate(zip(roles, means, scales)):
            rows.append({"sex": "male", "call_type": "usv", "adjacency": "filtered", "n_peak": n_peak,
                         "n_background": 2, "component": component, "role": role, "logmean": mean,
                         "median_sec": float(np.exp(mean)), "logscale": scale, "nu": 20.0,
                         "weight": 1.0 / len(roles), "log_likelihood": loglik, "n_parameters": 10 + n_peak,
                         "bic": bic, "shared_peak_scale": 0.24})
    peak_lrt = pls.DataFrame({
        "sex": ["male"] * 2, "call_type": ["usv"] * 2, "adjacency": ["filtered"] * 2,
        "n_peak_null": [1, 2], "n_peak_alt": [2, 3], "n_background": [2, 2],
        "lr_obs": [68.0, 11.3], "design_effect": [1.56, 1.35], "design_effect_raw": [1.56, 1.35],
        "effective_n": [6400.0, 7400.0], "lr_corrected": [43.5, 8.4], "null_p95": [6.0, 7.0],
        "threshold": [10.0, 12.4], "p_value": [0.0, 0.009], "p_value_corrected": [0.0, 0.028],
        "negative_fraction": [0.0, 0.01], "B": [3, 3], "n_subsample": [10000, 10000],
        "alpha_used": [0.0033, 0.0033], "rejected": [True, False]})
    peak_lrt_null = pls.DataFrame({
        "sex": ["male"] * 6, "call_type": ["usv"] * 6, "adjacency": ["filtered"] * 6,
        "n_peak_null": [1, 1, 1, 2, 2, 2], "b": [2, 0, 1, 0, 1, 2],
        "lr_b": [3.0, 1.0, 2.0, 4.0, 5.0, 6.0]})
    out = tmp_path / "usv_interval_analysis_20260101_120000.h5"
    write_ivi_h5(out, analysis_attrs={"git_sha": "abc"},
                 per_mode={"e2s": {"attrs": {"selected_n_peak_male_usv": 2, "alpha_effective_male_usv": 0.0033},
                                   "tied_fits": pls.DataFrame(rows), "peak_lrt": peak_lrt,
                                   "peak_lrt_null": peak_lrt_null}})
    return out


def test_load_tied_model_from_h5_rebuilds_the_selected_and_requested_fits(tmp_path):
    """Without n_peak the selected count (2) is rebuilt from its components; an explicit count is
    honoured; a pool with no fit raises."""
    arc = _tied_archive(tmp_path)
    model, order, n_peak, rows = load_tied_model_from_h5(arc, "e2s", "male", "usv")
    assert n_peak == 2 and rows.height == 4
    np.testing.assert_allclose(np.asarray(model.means_).ravel(), [-2.8, -1.7, -1.2, 2.4])
    np.testing.assert_allclose(np.asarray(model.covariances_).ravel(), np.array([0.24, 0.24, 1.2, 1.1]) ** 2)
    np.testing.assert_array_equal(order, [0, 1, 2, 3])
    assert rows["role"].to_list() == ["peak", "peak", "background", "background"]
    _, _, n_one, rows_one = load_tied_model_from_h5(arc, "e2s", "male", "usv", n_peak=1)
    assert n_one == 1 and rows_one.height == 3
    with pytest.raises(KeyError):
        load_tied_model_from_h5(arc, "e2s", "male", "usv", n_peak=4)


def test_load_peak_lrt_sweep_from_h5_reads_the_corrected_statistic(tmp_path):
    """Each rung is keyed (n_null, n_alt) and carries the CORRECTED statistic and p-value the test
    decided on, the null draws in replicate order, and the per-rung level the test used."""
    sweep, alpha = load_peak_lrt_sweep_from_h5(_tied_archive(tmp_path), "e2s", "male", "usv")
    assert alpha == 0.0033
    assert list(sweep["male"]) == [(1, 2), (2, 3)]
    first = sweep["male"][(1, 2)]
    assert first["lr_obs"] == 43.5 and first["p_value"] == 0.0
    np.testing.assert_allclose(first["lr_null"], [1.0, 2.0, 3.0])
    assert sweep["male"][(2, 3)]["p_value"] == 0.028 and sweep["male"][(2, 3)]["null_max"] == 6.0


def test_load_tied_ic_table_from_h5_gives_one_row_per_peak_count(tmp_path):
    """One row per peak count in the plot_ic_curves layout, n_comp holding the PEAK count."""
    table = load_tied_ic_table_from_h5(_tied_archive(tmp_path), "e2s", "male", "usv")
    assert table["n_comp"].to_list() == [1, 2]
    assert table["bic"].to_list() == [230.0, 215.0]
    assert set(table["sex"].to_list()) == {"male"} and set(table["rep"].to_list()) == {0}


_SQUEAK_STYLES = {
    "courtship": {"color": "#023047", "label": "courtship"},
    "female_female": {"color": "#C1121F", "label": "female-female"},
}


def _write_squeak_session(root: Path, starts: list[float], call_class: list[str | None], noise: list[bool]) -> None:
    """Writes ``<root>/audio/<name>_usv_summary.csv`` with the three columns the heatmap reads
    (``call_class`` null on noise rows, as ``detect-usv-squeaks`` writes it)."""

    (root / "audio").mkdir(parents=True)
    pls.DataFrame({"start": starts, "call_class": call_class, "noise": noise},
                  schema_overrides={"call_class": pls.Utf8}).write_csv(
        str(root / "audio" / f"{root.name}_usv_summary.csv"))


def _squeak_cohort(tmp_path: Path) -> dict[str, list[str]]:
    """
    Three sessions over two conditions, written to disk with one session-list file per
    condition:

    * ``s_hi`` (courtship): a cluster at 10-13 s of which 2 of 4 segments hold a squeak (a pure
      squeak at 10 s, a squeak-plus-USV ``both`` segment at 13 s), one USV at 90 s, and one
      noise segment at 95 s (no class) that the noise filter must remove -- rate 2/5 under
      ``squeak+both``;
    * ``s_lo`` (courtship): 5 segments at 10-14 s, 1 pure squeak -- rate 1/5;
    * ``s_none`` (female_female): 5 USVs, no squeak -- drawn as a 0 % row.
    """

    sessions = {
        "s_hi": ([10.0, 11.0, 12.0, 13.0, 90.0, 95.0], ["squeak", "usv", "usv", "both", "usv", None],
                 [False, False, False, False, False, True]),
        "s_lo": ([10.0, 11.0, 12.0, 13.0, 14.0], ["squeak", "usv", "usv", "usv", "usv"], [False] * 5),
        "s_none": ([10.0, 11.0, 12.0, 13.0, 14.0], ["usv"] * 5, [False] * 5),
    }
    for name, (starts, squeak, noise) in sessions.items():
        _write_squeak_session(tmp_path / name, starts, squeak, noise)
    courtship_list = tmp_path / "courtship.txt"
    courtship_list.write_text(f"{tmp_path / 's_lo'}\n{tmp_path / 's_hi'}\n")
    female_list = tmp_path / "female_female.txt"
    female_list.write_text(f"{tmp_path / 's_none'}\n")
    return {"courtship": [str(courtship_list)], "female_female": [str(female_list)]}


def _squeak_heatmap(lists, exclude_noise_usvs, min_session_segments, squeak_class="squeak+both"):
    """Runs the heatmap on a 180 s axis with a 1 s grid and a 2 s kernel."""

    return plot_session_squeak_time_heatmap(
        condition_session_lists=lists, condition_styles=_SQUEAK_STYLES,
        exclude_noise_usvs=exclude_noise_usvs, kernel_sigma_s=2.0, grid_step_s=1.0,
        min_vocal_density=0.5, session_length_s=180.0, min_session_segments=min_session_segments,
        vmax_percent=100.0, zero_tint=0.12, nodata_color="#FFFFFF", squeak_class=squeak_class)


def test_plot_session_squeak_time_heatmap_rates_order_and_silence(tmp_path):
    """
    Checks the numbers behind the figure: noise segments are dropped before counting, a
    session without any squeak is still drawn (as a 0 % row), drawn rows are sorted by squeak rate
    within their condition, the kernel share at the centre of the 10-13 s cluster equals the
    cluster's squeak fraction (the two squeaks sit at 10 s and 13 s, symmetric about 11.5 s,
    so the share there is exactly the weight of the outer pair, strictly between 0 and 1/2),
    a lone non-squeak far from any squeak reads 0 %, and a stretch with no vocalization within
    the kernel's reach is NaN (blank) rather than 0.
    """

    fig, axes, stats = _squeak_heatmap(_squeak_cohort(tmp_path), exclude_noise_usvs=True,
                                       min_session_segments=2)
    try:
        sessions = stats["sessions"].sort("session_id")
        assert sessions["session_id"].to_list() == ["s_hi", "s_lo", "s_none"]
        assert sessions["n_segments"].to_list() == [5, 5, 5]
        assert sessions["n_noise_dropped"].to_list() == [1, 0, 0]
        assert sessions["drawn"].to_list() == [True, True, True]
        assert stats["n_drawn"] == 3 and stats["n_no_squeak"] == 1 and stats["n_too_few"] == 0
        matrix, grid = stats["rate_matrix"], stats["grid_s"]
        assert matrix.shape == (3, 180) and grid[0] == pytest.approx(0.5)
        # Row 2 is s_none (female_female block, after both courtship rows): 0 % where it vocalizes.
        assert matrix[2, int(np.argmin(np.abs(grid - 12.5)))] == pytest.approx(0.0, abs=1e-9)
        at = int(np.argmin(np.abs(grid - 11.5)))
        outer = 2.0 * np.exp(-0.5 * (1.5 / 2.0) ** 2)
        inner = 2.0 * np.exp(-0.5 * (0.5 / 2.0) ** 2)
        # Row 0 is s_hi (rate 2/5 beats s_lo's 1/5).
        assert matrix[0, at] == pytest.approx(outer / (outer + inner), abs=1e-3)
        assert matrix[0, int(np.argmin(np.abs(grid - 90.5)))] == pytest.approx(0.0, abs=1e-6)
        assert np.isnan(matrix[0, int(np.argmin(np.abs(grid - 50.5)))])
        assert np.isnan(matrix[1, int(np.argmin(np.abs(grid - 150.5)))])
        assert np.nanmax(matrix) <= 1.0 and np.nanmin(matrix) >= 0.0
        assert len(axes) == 4
    finally:
        plt.close(fig)


def test_plot_session_squeak_time_heatmap_keeps_noise_when_asked(tmp_path):
    """With ``exclude_noise_usvs=False`` the noise segment stays in the denominator but, having
    no call class, is never a squeak; sessions below the segment minimum are reported as too few
    rather than drawn."""

    fig, _axes, stats = _squeak_heatmap(_squeak_cohort(tmp_path), exclude_noise_usvs=False,
                                        min_session_segments=6)
    try:
        sessions = stats["sessions"].sort("session_id")
        assert sessions["n_segments"].to_list() == [6, 5, 5]
        assert sessions["n_squeaks"].to_list() == [2, 1, 0]
        assert stats["n_drawn"] == 1 and stats["n_too_few"] == 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("squeak_class", "expected_squeaks"),
    [("squeak+both", [2, 1, 0]), ("squeak", [1, 1, 0]), ("both", [1, 0, 0])],
)
def test_plot_session_squeak_time_heatmap_class_selector(tmp_path, squeak_class, expected_squeaks):
    """``squeak_class`` picks which call classes count as squeaks -- pure squeaks, the mixed
    ``both`` segments, or either -- while every non-noise segment stays in the denominator; the
    selection is echoed in ``stats_dict``."""

    fig, _axes, stats = _squeak_heatmap(_squeak_cohort(tmp_path), exclude_noise_usvs=True,
                                        min_session_segments=2, squeak_class=squeak_class)
    try:
        sessions = stats["sessions"].sort("session_id")
        assert sessions["n_squeaks"].to_list() == expected_squeaks
        assert sessions["n_segments"].to_list() == [5, 5, 5]
        assert stats["squeak_class"] == squeak_class
    finally:
        plt.close(fig)


def test_plot_session_squeak_time_heatmap_rejects_unknown_class(tmp_path):
    """An unknown squeak-class selection is a ValueError naming the valid ones."""

    with pytest.raises(ValueError, match="Unknown squeak class selection"):
        _squeak_heatmap(_squeak_cohort(tmp_path), exclude_noise_usvs=True, min_session_segments=2,
                        squeak_class="usv")


def test_plot_session_squeak_time_heatmap_needs_call_class(tmp_path):
    """A summary without ``call_class`` (scored only by the retired binary squeak detector) is a
    KeyError, never a silent all-USV session."""

    root = tmp_path / "s_old"
    (root / "audio").mkdir(parents=True)
    pls.DataFrame({"start": [1.0, 2.0], "squeak": [True, False], "noise": [False, False]}).write_csv(
        str(root / "audio" / f"{root.name}_usv_summary.csv"))
    session_list = tmp_path / "list.txt"
    session_list.write_text(f"{root}\n")
    with pytest.raises(KeyError, match="call_class"):
        _squeak_heatmap({"courtship": [str(session_list)]}, exclude_noise_usvs=True, min_session_segments=1)


def test_plot_session_squeak_time_heatmap_rejects_session_in_two_conditions(tmp_path):
    """A session listed under two conditions is a ValueError, since its row would be ambiguous."""

    lists = _squeak_cohort(tmp_path)
    lists["female_female"].append(lists["courtship"][0])
    with pytest.raises(ValueError, match="listed under both"):
        _squeak_heatmap(lists, exclude_noise_usvs=True, min_session_segments=2)


def test_plot_session_squeak_time_heatmap_missing_summary_raises(tmp_path):
    """A listed session without a USV summary is a FileNotFoundError naming its audio folder."""

    (tmp_path / "s_empty").mkdir()
    session_list = tmp_path / "list.txt"
    session_list.write_text(f"{tmp_path / 's_empty'}\n")
    with pytest.raises(FileNotFoundError, match="usv_summary"):
        _squeak_heatmap({"courtship": [str(session_list)]}, exclude_noise_usvs=True, min_session_segments=2)
