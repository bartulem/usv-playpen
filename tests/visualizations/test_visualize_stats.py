"""
@author: bartulem
Mock-based tests for visualizations/usv_summary_statistics.py and
visualizations/usv_interval_summary_statistics.py — both files are pure
compute / data-loading helpers when wired to synthetic sessions on disk.
"""

from __future__ import annotations

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
)
from usv_playpen.visualizations.usv_interval_summary_statistics import (
    build_master_usv_interval_dataframe,
    find_latest_archive,
    load_intervals_from_h5,
    load_gmm_fits_from_h5,
    load_best_fit_from_h5,
    load_lrt_sweep_from_h5,
    selected_K_from_h5,
)
from usv_playpen.analyses.usv_interval_archive import write_ivi_h5


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
    noise_col: str = "cluster",
    cat_col: str = "usv_supercategory",
    n_noise: int = 2,
):
    """Build a session_root containing:

    - A 3D-tracking H5 file at <root>/video/<session_id>_points3d_translated_rotated_metric.h5
      with `track_names`, `recording_frame_rate`, `experimental_code`.
    - A USV summary CSV at <root>/audio/<session_id>_usv_summary.csv with
      `start`, `duration`, `emitter`, the noise column and the category column.
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
    rows_cat = []
    rows_umap_x = []
    rows_umap_y = []

    t = 0.05
    for _ in range(n_male_calls):
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append(male_id)
        rows_noise.append(0)  # not noise
        rows_cat.append(1)
        rows_umap_x.append(np.random.RandomState(0).rand())
        rows_umap_y.append(np.random.RandomState(1).rand())
        t += 0.5
    for _ in range(n_female_calls):
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append(female_id)
        rows_noise.append(0)
        rows_cat.append(2)
        rows_umap_x.append(0.5)
        rows_umap_y.append(0.5)
        t += 0.5
    for _ in range(n_unassigned):
        rows_start.append(t)
        rows_dur.append(0.05)
        rows_emitter.append("UNKNOWN")
        rows_noise.append(0)
        rows_cat.append(3)
        rows_umap_x.append(0.7)
        rows_umap_y.append(0.7)
        t += 0.5
    for _ in range(n_noise):
        rows_start.append(t)
        rows_dur.append(0.01)
        rows_emitter.append(male_id)
        rows_noise.append(99)  # noise category
        rows_cat.append(99)
        rows_umap_x.append(0.0)
        rows_umap_y.append(0.0)
        t += 0.5

    cols = {
        "start": rows_start,
        "duration": rows_dur,
        "emitter": rows_emitter,
        noise_col: rows_noise,
    }
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
        noise_col_id="cluster", noise_categories=[99],
    )
    # Only non-noise rows survive: 2 male + 1 female = 3
    assert df.height == 3
    assert "frame_index" in df.columns
    # Spot-check: frame_index of the first call (start = 0.05) at 150 Hz = 7
    assert df["frame_index"][0] == int(0.05 * 150.0)


def test_load_and_filter_usv_data_missing_csv_raises(tmp_path):
    """No USV CSV → FileNotFoundError."""
    sess = tmp_path / "20260101_120000"
    sess.mkdir()
    with pytest.raises(FileNotFoundError):
        load_and_filter_usv_data(str(sess), frame_rate=150.0,
                                 noise_col_id="cluster", noise_categories=[])


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
        noise_col_id="cluster", noise_categories=[99],
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
        noise_col_id="cluster", noise_categories=[99],
        usv_category_col="usv_supercategory",
        usv_continuous_cols=("umap_x", "umap_y"),
    )
    # No matching columns → the per-session block continues; final result empty.
    assert df.height == 0


def test_extract_category_embedding_data_skips_bad_session(tmp_path):
    """A non-existent session is skipped silently (FileNotFoundError caught)."""
    df = extract_category_embedding_data(
        session_roots=["/no/such/session"],
        noise_col_id="cluster", noise_categories=[99],
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
    usv = load_and_filter_usv_data(str(sess), md["frame_rate"], "cluster", [99])
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
        noise_col_id="cluster", noise_categories=[99],
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


def test_build_master_usv_dataframe_raises_when_all_skipped(tmp_path):
    """Every session missing → RuntimeError with diagnostic message."""
    with pytest.raises(RuntimeError, match="loaded 0 sessions"):
        build_master_usv_dataframe(
            session_roots=[str(tmp_path / "absent")],
            noise_col_id="cluster", noise_categories=[],
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
            noise_col_id="cluster", noise_categories=[99],
            usv_category_col="some_missing_col",  # neither session has this
            distance_suffix="nose-nose",
            mf_angle_suffix="allo_yaw-nose",
            fm_angle_suffix="nose-allo_yaw",
        )


# ===========================================================================
# usv_interval_summary_statistics — loaders driven off an HDF5 archive
# ===========================================================================


def _build_archive(tmp_path: Path, *, with_gmm: bool = True, with_lrt: bool = True) -> Path:
    """Constructs a usv_interval_analysis_<ts>.h5 file with both modes populated."""
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

    if with_gmm:
        gmm_rows = []
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
                gmm_rows.append(row)
        payload["s2s"]["gmm_fits"] = pls.DataFrame(gmm_rows)

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


# ---- load_intervals_from_h5 / load_gmm_fits_from_h5 / load_best_fit -------


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


def test_load_gmm_fits_from_h5_round_trip(tmp_path):
    """gmm_fits comes back with all per-component columns intact."""
    arc = _build_archive(tmp_path, with_gmm=True)
    df = load_gmm_fits_from_h5(str(arc), interval_type="s2s")
    for col in ("weight_1", "logmean_1", "logsd_1", "model_class"):
        assert col in df.columns


def test_load_gmm_fits_from_h5_missing_raises(tmp_path):
    """Archive without GMM sweep (fit_gmm=false at compute time) → ValueError."""
    arc = _build_archive(tmp_path, with_gmm=False)
    with pytest.raises(ValueError, match="contains no\\s+GMM sweep"):
        load_gmm_fits_from_h5(str(arc), interval_type="s2s")


def test_load_best_fit_from_h5_returns_a_model(tmp_path):
    """End-to-end: load fits then reconstruct the (sex, K) model."""
    arc = _build_archive(tmp_path, with_gmm=True)
    model, order = load_best_fit_from_h5(str(arc), interval_type="s2s",
                                         sex="male", K=2)
    assert model is not None
    np.testing.assert_array_equal(order, np.arange(2))


# ---- load_lrt_sweep_from_h5 -----------------------------------------------


def test_load_lrt_sweep_from_h5_returns_dict(tmp_path):
    """Re-hydrates the per-(sex, K_null, K_alt) sweep dict."""
    arc = _build_archive(tmp_path, with_gmm=True, with_lrt=True)
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


def test_load_lrt_sweep_from_h5_missing_tables_raises(tmp_path):
    """Archive without bootstrap_lrt → ValueError."""
    arc = _build_archive(tmp_path, with_gmm=False, with_lrt=False)
    with pytest.raises(ValueError, match="missing the\\s+bootstrap-LRT"):
        load_lrt_sweep_from_h5(str(arc), interval_type="s2s")


# ---- selected_K_from_h5 ---------------------------------------------------


def test_selected_K_from_h5_reads_attrs(tmp_path):
    """K_selected_{male,female} attrs come back as ints."""
    arc = _build_archive(tmp_path, with_gmm=True, with_lrt=True)
    sel = selected_K_from_h5(str(arc), interval_type="s2s")
    assert sel == {"male": 2, "female": 3}


def test_selected_K_from_h5_unknown_mode_raises(tmp_path):
    """Asking for a mode not in the archive → ValueError."""
    arc = _build_archive(tmp_path, with_gmm=True, with_lrt=True)
    with pytest.raises(ValueError, match="not found"):
        selected_K_from_h5(str(arc), interval_type="e2s")


# ---- build_master_usv_interval_dataframe ---------------------------------


def test_build_master_usv_interval_dataframe_empty_when_no_sessions(tmp_path):
    """Session list resolves to no readable sessions → empty frame, not error."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/nonexistent/session\n")
    df, summary = build_master_usv_interval_dataframe(
        session_lists=[str(list_file)],
        noise_col_id="cluster", noise_categories=[99],
        message_output=lambda *_a, **_kw: None,
    )
    assert df.height == 0
    assert summary["n_sessions_loaded"] == 0
    # Schema is still set up so downstream filter() calls don't crash
    assert "interval_type" in df.columns


def test_build_master_usv_interval_dataframe_aggregates_two_sessions(tmp_path):
    """Two synthetic sessions → tidy interval frame with both modes present."""
    sess1 = tmp_path / "20260101_120000"
    sess2 = tmp_path / "20260102_120000"
    _make_synthetic_session(sess1, n_male_calls=4, n_female_calls=3, n_unassigned=0)
    _make_synthetic_session(sess2, n_male_calls=3, n_female_calls=3, n_unassigned=0)
    list_file = tmp_path / "sessions.txt"
    list_file.write_text(f"{sess1}\n{sess2}\n")
    df, summary = build_master_usv_interval_dataframe(
        session_lists=[str(list_file)],
        noise_col_id="cluster", noise_categories=[99],
        message_output=lambda *_a, **_kw: None,
    )
    assert df.height > 0
    assert set(df["interval_type"].unique().to_list()) == {"s2s", "e2s"}
    assert summary["n_sessions_loaded"] == 2


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
