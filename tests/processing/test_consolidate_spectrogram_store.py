"""
@author: bartulem
Tests for processing/consolidate_spectrogram_store.SpectrogramStoreConsolidator.

Coverage: two synthetic sessions and a synthetic QLVM model package in the v3
layout (the five production cells, corpus/SESSION_H5_BASELINE.tsv,
MANIFEST.sha256) consolidate into one ``spectrograms_qlvmv3_*`` store with the
shared frequency_bins, byte-equal spectrogram/mask groups, the regular model's
qlvm_dim, per-model coordinates (NaN where not embedded), int16 labels (0 where
not embedded), per-call status codes, per-model provenance attrs and package
tables, the sessions table and the root provenance attrs. Refusals (nothing
written): a spectrogram H5 whose SHA-256 differs from the baseline, a session
outside the package corpus, a summary lacking QLVM columns, an embedding that
disagrees with the status codes, a divergent frequency axis, a row-count
mismatch, a missing session H5. The baseline resolves to session roots, the CLI
takes --root-directories and / or --session-lists, or --package-corpus, and
os_utils.resolve_consolidated_h5_path picks the newest store of either name.

Production mode: two synthetic sessions (embedded, too-long, maskless, both,
pure-squeak rows) with every production map column, synthetic production cells
under a patched os_utils.QLVM_MODEL_PACKAGE_ROOT / QLVM_SQUEAK_PACKAGE_ROOT and
the synthetic category bundle consolidate into one ``spectrograms_production_*``
store: byte-equal spectrogram/mask groups, per-map coordinates (NaN where not
embedded, squeak map NaN off the squeak rows), status bit fields, qlvm_category
with the bundle's per-call agreement and uncertain flag, per-map provenance
(model id, cell directory, checkpoint SHA-256), the bundle's grids and identity,
the sessions table and the root attrs; the store is the one
resolve_consolidated_h5_path picks and the torus-traversal video accepts its
provenance. Refusals (nothing written, every problem in one error): missing
columns, a category from another bundle, coordinates the status or the squeak
flag excludes, a row-count mismatch, a regular cell without require_mask. The
CLI's --store-mode reaches the settings block, and an unknown mode is refused.
"""

from __future__ import annotations

import json
import os
import pathlib

import h5py
import numpy as np
import polars as pls
import pytest
from click.testing import CliRunner

from usv_playpen import os_utils
from usv_playpen.analyses.usv_interval_archive import _h5_to_polars
from usv_playpen.os_utils import (
    QLVM_MAPS,
    QLVM_SQUEAK_MAP,
    configure_path,
    qlvm_cell_model_id,
    qlvm_map_cell_directory,
    resolve_consolidated_h5_path,
)
from usv_playpen.processing import consolidate_spectrogram_store as css
from usv_playpen.processing.build_qlvm_training_set import file_sha256
from usv_playpen.visualizations.qlvm_torus_traversal_video import (
    check_store_map_provenance,
    pool_latents_from_h5,
)

SESSIONS = ("20260101_120000", "20260102_120000")
QLVM_PRODUCTION_MODEL_CELLS = css.V3_MODEL_CELLS
PREFIXES = tuple(QLVM_PRODUCTION_MODEL_CELLS)
# Five calls per session: durations and SAM-mask counts give statuses 0, 1, 2, 3, 0.
DURATIONS = np.array([10, 200, 10, 200, 10], dtype=np.int64)
MASK_INDEX = np.array([0, 1, 4, 4], dtype=np.int64)
EXPECTED_STATUS = np.array([0, 1, 2, 3, 0], dtype=np.int8)
EMBEDDED_ROWS = (0, 4)
SPEC_SETTINGS = {"num_freq_bins": 16, "num_time_bins": 8, "min_freq": 30000, "max_freq": 120000}


def _build_session(base_dir, session_id, freq0=30000.0, seed=0):
    """Create one session dir with a per-session spectrograms H5 (five calls, mask
    group) and a usv_summary CSV carrying every production QLVM column: coordinates
    and labels on the embedded rows 0 and 4, nulls elsewhere. Returns the root."""
    root = base_dir / session_id
    spec_dir = root / "audio" / "spectrograms"
    spec_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    n_usv = DURATIONS.shape[0]
    with h5py.File(spec_dir / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(freq0, 120000.0, 16))
        grp = f.create_group(f"spectrogram/{session_id}")
        grp.create_dataset("spectrograms", data=rng.random((n_usv, 16, 8)).astype(np.float32), compression="gzip")
        grp.create_dataset("durations", data=DURATIONS)
        mgrp = f.create_group(f"mask/{session_id}")
        mgrp.create_dataset("segmentations", data=rng.random((MASK_INDEX.shape[0], 16, 8)) > 0.5)
        mgrp.create_dataset("spectrogram_index", data=MASK_INDEX)
    rows = {
        "usv_id": [f"{i:04d}" for i in range(n_usv)],
        "start": [0.1 + 0.2 * i for i in range(n_usv)],
        "stop": [0.15 + 0.2 * i for i in range(n_usv)],
    }
    for index, prefix in enumerate(PREFIXES):
        for axis in (1, 2):
            rows[f"{prefix}{axis}"] = [
                float(rng.uniform(0.0, 1.0)) if row in EMBEDDED_ROWS else None for row in range(n_usv)
            ]
        rows[f"{prefix}_category"] = [index + row + 1 if row in EMBEDDED_ROWS else None for row in range(n_usv)]
        rows[f"{prefix}_supercategory"] = [row + 1 if row in EMBEDDED_ROWS else None for row in range(n_usv)]
    pls.DataFrame(rows).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root


def _write_fake_package(tmp_path, roots):
    """A QLVM model package in the v3 layout at ``<tmp_path>/qlvm_models_latest/v3``:
    MANIFEST.sha256, corpus/SESSION_H5_BASELINE.tsv listing ``roots`` with their H5
    SHA-256, and every production cell with config/training_contract.json,
    config/run_config.json and inference/clusters_fine/ (plus clusters_coarse/ with
    fine_to_coarse.csv); the regular cell's
    inference/recon_mse_breakdown.npz types each session. Returns the package root."""
    package = tmp_path / "qlvm_models_latest" / "v3"
    (package / "corpus").mkdir(parents=True)
    (package / "MANIFEST.sha256").write_text("0" * 64 + "  README.md\n")
    lines = ["session\th5_rows\tcorpus_rows\tbytes\tsha256\tpath"]
    for root in roots:
        session_id = root.name
        h5_path = root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5"
        lines.append(
            f"{session_id}\t5\t2\t{h5_path.stat().st_size}\t{file_sha256(h5_path)}\t"
            f"Bartul/Data/{session_id}/audio/spectrograms/{session_id}_spectrograms.h5"
        )
    (package / "corpus" / "SESSION_H5_BASELINE.tsv").write_text("\n".join(lines) + "\n")
    conditions = {"qlvm": None, "qlvm_dur": "duration", "qlvm_mf": "mean_freq", "qlvm_bw": "bandwidth", "qlvm_loud": "loudness"}
    for index, (prefix, relative_cell) in enumerate(QLVM_PRODUCTION_MODEL_CELLS.items()):
        cell = package / relative_cell
        (cell / "config").mkdir(parents=True)
        contract = {"length_threshold": 128.0, "conditional": conditions[prefix], "floor": 0.2, "c_dim": int(prefix != "qlvm")}
        (cell / "config" / "training_contract.json").write_text(json.dumps(contract))
        (cell / "config" / "run_config.json").write_text(json.dumps({"phase": relative_cell.split("/")[0], "mask_tag": "unmasked_floor"}))
        for level in ("fine", "coarse"):
            clusters = cell / "inference" / f"clusters_{level}"
            clusters.mkdir(parents=True)
            pls.DataFrame({
                "label": [1, 2], "n_calls": [30 + index, 20], "peak_x": [0.1, 0.6], "stable": [True, False],
                "fine_clusters": ["1;3", "2"],
            }).write_csv(clusters / "clusters.csv")
            pls.DataFrame({"basin_a": [1], "basin_b": [2], "saddle": [0.5], "supported": [True]}).write_csv(
                clusters / "boundaries.csv"
            )
            np.save(clusters / "label_grid.npy", (np.arange(40000).reshape(200, 200) % (3 + index) + 1).astype(np.int16))
            if level == "coarse":
                pls.DataFrame({"fine_label": [1, 2, 3], "coarse_label": [1, 2, 1]}).write_csv(clusters / "fine_to_coarse.csv")
    regular = package / QLVM_PRODUCTION_MODEL_CELLS["qlvm"]
    np.savez(
        regular / "inference" / "recon_mse_breakdown.npz",
        spec_id=np.array([f"{root.name}_{row}" for root in roots for row in EMBEDDED_ROWS]),
        session_types=np.array([session_type for session_type in ("MF_intact", "MF_mute") for _row in EMBEDDED_ROWS]),
    )
    return package


def _consolidate(tmp_path, roots, package, mocker):
    """Run the consolidator into ``<tmp_path>/store`` with silent messages; returns
    (the stores in the output dir, the returned path)."""
    mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.smart_wait")
    out_dir = tmp_path / "store"
    out_dir.mkdir(exist_ok=True)
    returned = css.SpectrogramStoreConsolidator(
        root_directories=[str(r) for r in roots],
        input_parameter_dict={
            "spectrograms_root": str(out_dir),
            "generate_spectrograms": SPEC_SETTINGS,
            "consolidate_spectrogram_store": {"store_mode": "v3"},
        },
        message_output=lambda *_a, **_kw: None,
        package_root=str(package),
    ).consolidate_spectrogram_store()
    return sorted(out_dir.iterdir()), returned


@pytest.fixture
def corpus(tmp_path):
    """Two sessions and the fake package whose baseline lists them."""
    roots = [_build_session(tmp_path / "data", session_id, seed=i) for i, session_id in enumerate(SESSIONS)]
    return roots, _write_fake_package(tmp_path, roots)


def test_store_layout_values_and_provenance(tmp_path, mocker, corpus):
    """The v3 store: file name, shared axis, byte-equal spectrogram and mask groups,
    qlvm_dim = the regular model's coordinates, per-model coordinates with NaN and
    int16 labels with 0 off the embedded rows, status codes, model attrs and tables
    (coarse level and fine_to_coarse for the regular model only), sessions table and
    root attrs."""
    roots, package = corpus
    stores, returned = _consolidate(tmp_path, roots, package, mocker)
    assert stores == [returned]
    assert returned.name.startswith("spectrograms_qlvmv3_2sessions_10vocalizations_")
    with h5py.File(returned, "r") as f:
        assert f.attrs["n_sessions"] == 2
        assert f.attrs["n_vocalizations"] == 10
        assert f.attrs["created_by"] == "consolidate-spectrogram-store"
        assert f.attrs["package_root"] == str(package)
        assert f.attrs["package_manifest_sha256"] == file_sha256(package / "MANIFEST.sha256")
        assert isinstance(f.attrs["git_commit"], str)
        assert f.attrs["git_commit"]
        assert json.loads(f.attrs["generate_spectrograms_settings"]) == SPEC_SETTINGS
        assert "not recorded" in f.attrs["generate_spectrograms_settings_note"]
        assert f["frequency_bins"].shape == (16,)
        assert json.loads(f["qlvm"].attrs["status_codes"])["3"].startswith("too long and no SAM mask")
        for root in roots:
            session_id = root.name
            summary = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
            with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "r") as source:
                for key in ("spectrogram/{s}/spectrograms", "spectrogram/{s}/durations", "mask/{s}/segmentations",
                            "mask/{s}/spectrogram_index"):
                    np.testing.assert_array_equal(f[key.format(s=session_id)][()], source[key.format(s=session_id)][()])
            np.testing.assert_array_equal(f[f"qlvm/{session_id}/status"][()], EXPECTED_STATUS)
            assert f[f"qlvm/{session_id}/status"].dtype == np.int8
            for prefix in PREFIXES:
                expected = summary.select(f"{prefix}1", f"{prefix}2").fill_null(np.nan).to_numpy().astype(np.float64)
                stored = f[f"qlvm/{session_id}/{prefix}"][()]
                assert stored.dtype == np.float64
                assert stored.shape == (5, 2)
                np.testing.assert_array_equal(stored, expected)
                assert np.isnan(stored[[1, 2, 3]]).all()
            np.testing.assert_array_equal(f[f"spectrogram/{session_id}/qlvm_dim"][()], f[f"qlvm/{session_id}/qlvm"][()])
            for column in ("qlvm_category", "qlvm_supercategory", "qlvm_dur_category", "qlvm_dur_supercategory",
                           "qlvm_mf_category", "qlvm_mf_supercategory", "qlvm_bw_category",
                           "qlvm_bw_supercategory", "qlvm_loud_category", "qlvm_loud_supercategory"):
                labels = f[f"qlvm/{session_id}/{column}"][()]
                assert labels.dtype == np.int16
                np.testing.assert_array_equal(labels, summary[column].fill_null(0).to_numpy())
                assert (labels[[1, 2, 3]] == 0).all()
                assert (labels[list(EMBEDDED_ROWS)] > 0).all()

        for prefix, relative_cell in QLVM_PRODUCTION_MODEL_CELLS.items():
            model = f[f"qlvm_models/{prefix}"]
            cell = package / relative_cell
            assert model.attrs["package_version"] == "v3"
            assert model.attrs["package_name"] == "qlvm_models_latest"
            assert model.attrs["cell"] == relative_cell
            assert model.attrs["phase"] == relative_cell.split("/")[0]
            assert model.attrs["design"] == "natural_5strata_N29000"
            assert model.attrs["condition"] == ("none" if prefix == "qlvm" else json.loads(
                (cell / "config" / "training_contract.json").read_text())["conditional"])
            assert json.loads(model.attrs["training_contract"]) == json.loads(
                (cell / "config" / "training_contract.json").read_text())
            levels = ("fine", "coarse")
            assert sorted(json.loads(model.attrs["label_columns"])) == sorted(levels)
            for level in levels:
                directory = cell / "inference" / f"clusters_{level}"
                assert _h5_to_polars(model[f"clusters_{level}"]).equals(pls.read_csv(directory / "clusters.csv"))
                assert _h5_to_polars(model[f"boundaries_{level}"]).equals(pls.read_csv(directory / "boundaries.csv"))
                assert model[f"label_grid_{level}"].dtype == np.int16
                np.testing.assert_array_equal(model[f"label_grid_{level}"][()], np.load(directory / "label_grid.npy"))
            assert "fine_to_coarse" in model
            assert "clusters_coarse" in model

        sessions = _h5_to_polars(f["sessions"])
        assert sessions["session_id"].to_list() == list(SESSIONS)
        assert sessions["session_type"].to_list() == ["MF_intact", "MF_mute"]
        assert sessions["n_rows"].to_list() == [5, 5]
        assert sessions["n_embedded"].to_list() == [2, 2]
        assert sessions["spectrogram_h5_sha256"].to_list() == [
            file_sha256(root / "audio" / "spectrograms" / f"{root.name}_spectrograms.h5") for root in roots
        ]
        assert sessions["usv_summary_sha256"].to_list() == [
            file_sha256(root / "audio" / f"{root.name}_usv_summary.csv") for root in roots
        ]


def test_sha_mismatch_is_refused_before_writing(tmp_path, mocker, corpus):
    """A session H5 that changed since the package (SHA-256 differs from the
    baseline) refuses the whole build, naming the session; nothing is written."""
    roots, package = corpus
    baseline = package / "corpus" / "SESSION_H5_BASELINE.tsv"
    true_digest = file_sha256(roots[1] / "audio" / "spectrograms" / f"{SESSIONS[1]}_spectrograms.h5")
    baseline.write_text(baseline.read_text().replace(true_digest, "f" * 64))
    with pytest.raises(ValueError, match=rf"{SESSIONS[1]}: spectrogram H5 SHA-256 .* differs from the package baseline"):
        _consolidate(tmp_path, roots, package, mocker)
    assert list((tmp_path / "store").iterdir()) == []


def test_session_outside_the_package_corpus_is_refused(tmp_path, mocker, corpus):
    """A session the baseline does not list cannot be verified and is refused."""
    roots, package = corpus
    extra = _build_session(tmp_path / "data", "20260103_120000", seed=5)
    with pytest.raises(ValueError, match="20260103_120000: not in the package corpus"):
        _consolidate(tmp_path, [*roots, extra], package, mocker)
    assert list((tmp_path / "store").iterdir()) == []


def test_missing_qlvm_columns_are_refused(tmp_path, mocker, corpus):
    """A summary without a v3 coordinate column refuses the build, listing it (label
    columns are not required: infer-qlvm-latents writes none by default)."""
    roots, package = corpus
    summary_path = roots[0] / "audio" / f"{SESSIONS[0]}_usv_summary.csv"
    pls.read_csv(summary_path).drop("qlvm_bw1", "qlvm_loud_category", "qlvm_loud_supercategory").write_csv(summary_path)
    with pytest.raises(ValueError, match=r"lacks the QLVM column\(s\) \['qlvm_bw1'\]"):
        _consolidate(tmp_path, roots, package, mocker)
    assert list((tmp_path / "store").iterdir()) == []


def test_embedding_that_contradicts_the_status_is_refused(tmp_path, mocker, corpus):
    """Regular coordinates on a call without a SAM mask (status 2) mean the summary
    does not follow the package's rules: refused."""
    roots, package = corpus
    summary_path = roots[0] / "audio" / f"{SESSIONS[0]}_usv_summary.csv"
    summary = pls.read_csv(summary_path)
    summary.with_columns(
        qlvm1=pls.Series([0.1, None, 0.2, None, 0.3]), qlvm2=pls.Series([0.1, None, 0.2, None, 0.3]),
    ).write_csv(summary_path)
    with pytest.raises(ValueError, match="1 rows with a nonzero status have them"):
        _consolidate(tmp_path, roots, package, mocker)


def test_labels_off_the_embedded_rows_are_refused(tmp_path, mocker, corpus):
    """A label where the model's coordinates are null is refused."""
    roots, package = corpus
    summary_path = roots[0] / "audio" / f"{SESSIONS[0]}_usv_summary.csv"
    pls.read_csv(summary_path).with_columns(qlvm_mf_category=pls.Series([1, 1, None, None, 1])).write_csv(summary_path)
    with pytest.raises(ValueError, match="qlvm_mf_category is not present exactly where qlvm_mf1/qlvm_mf2 are"):
        _consolidate(tmp_path, roots, package, mocker)


def test_frequency_axis_rowcount_and_missing_h5_are_refused_together(tmp_path, mocker, corpus):
    """A divergent frequency axis, a summary/H5 row-count mismatch and a missing H5
    are all reported in one error."""
    roots, package = corpus
    other = _build_session(tmp_path / "other", SESSIONS[1], freq0=25000.0)
    summary_path = roots[0] / "audio" / f"{SESSIONS[0]}_usv_summary.csv"
    pls.read_csv(summary_path).head(4).write_csv(summary_path)
    bare = tmp_path / "data" / "20260108_120000"
    (bare / "audio").mkdir(parents=True)
    with pytest.raises(ValueError, match="refused") as error:
        _consolidate(tmp_path, [roots[0], other, bare], package, mocker)
    message = str(error.value)
    assert "frequency_bins" in message
    assert "inconsistent, 5 spectrogram rows but 4 USV summary rows" in message
    assert "20260108_120000: no per-session spectrogram H5" in message
    assert list((tmp_path / "store").iterdir()) == []


def test_package_corpus_root_directories_follow_the_baseline(corpus):
    """Each baseline path resolves to its session root (the H5's third parent under /mnt/falkner)."""
    _roots, package = corpus
    assert css.package_corpus_root_directories(package) == [
        configure_path(f"/mnt/falkner/Bartul/Data/{session_id}") for session_id in SESSIONS
    ]


def test_cli_takes_exactly_one_session_source(mocker, corpus, tmp_path_factory):
    """--package-corpus resolves the package's sessions; --root-directories splits the
    list and --session-lists appends the listed roots (each once); giving an explicit
    list together with --package-corpus, or nothing, is a usage error."""
    _roots, package = corpus
    consolidator = mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.SpectrogramStoreConsolidator")
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--package-corpus", "--package-root", str(package)])
    assert result.exit_code == 0, result.output
    assert consolidator.call_args.kwargs["root_directories"] == css.package_corpus_root_directories(package)
    assert consolidator.call_args.kwargs["package_root"] == str(package)
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--root-directories", "/a/b, /c/d"])
    assert result.exit_code == 0, result.output
    assert consolidator.call_args.kwargs["root_directories"] == ["/a/b", "/c/d"]
    session_list = tmp_path_factory.mktemp("lists") / "sessions.txt"
    session_list.write_text("# cohort\n/c/d\n\n/e/f\n")
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--root-directories", "/a/b, /c/d", "--session-lists", str(session_list)])
    assert result.exit_code == 0, result.output
    assert consolidator.call_args.kwargs["root_directories"] == ["/a/b", "/c/d", "/e/f"]
    for arguments in (["--root-directories", "/a/b", "--package-corpus"], ["--session-lists", str(session_list), "--package-corpus"], []):
        result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, arguments)
        assert result.exit_code == 2
        assert "--package-corpus (not both)" in result.output


def test_resolver_picks_the_newest_store_of_either_name(tmp_path, mocker, corpus):
    """resolve_consolidated_h5_path finds the new spectrograms_qlvmv3_* store and still
    picks the newest file when an older-named spectrograms_sam2masks_* store is newer."""
    roots, package = corpus
    out_dir = tmp_path / "store"
    out_dir.mkdir()
    old_store = out_dir / "spectrograms_sam2masks_1sessions_5vocalizations_20250101_000000Z.h5"
    old_store.write_bytes(b"")
    os.utime(old_store, (1_000_000_000, 1_000_000_000))
    _stores, returned = _consolidate(tmp_path, roots, package, mocker)
    assert resolve_consolidated_h5_path(str(out_dir)) == str(returned)
    os.utime(old_store, None)
    os.utime(returned, (1_000_000_000, 1_000_000_000))
    assert resolve_consolidated_h5_path(str(out_dir)) == str(old_store)


def test_readers_of_the_store_work_on_the_new_file(tmp_path, mocker, corpus):
    """The torus-traversal video pools a map's qlvm/<session>/<map> coordinates of the
    new store (embedded rows only)."""
    roots, package = corpus
    _stores, returned = _consolidate(tmp_path, roots, package, mocker)
    with h5py.File(returned, "r") as f:
        coords, index = pool_latents_from_h5(f, "qlvm")
    assert coords.shape == (4, 2)
    assert index == [(session_id, row) for session_id in SESSIONS for row in EMBEDDED_ROWS]


# Production mode: six calls per session. Rows 0-3 are pure USVs (status 0, 1 too
# long, 2 no SAM mask, 3 both), row 4 holds a USV and a squeak (status 0), row 5 is a
# pure squeak with a mask (status 4). The USV maps place rows 0 and 4, the squeak map
# rows 4 and 5.
PRODUCTION_DURATIONS = np.array([10, 200, 10, 200, 10, 10], dtype=np.int64)
PRODUCTION_MASK_INDEX = np.array([0, 1, 4, 4, 5], dtype=np.int64)
PRODUCTION_USV = [True, True, True, True, True, False]
PRODUCTION_SQUEAK = [False, False, False, False, True, True]
PRODUCTION_STATUS = np.array([0, 1, 2, 3, 0, 4], dtype=np.int8)
PRODUCTION_EMBEDDED_ROWS = (0, 4)
PRODUCTION_SQUEAK_ROWS = (4, 5)
# Regular-map positions of the embedded rows on the synthetic bundle's quadrants:
# row 0 at (0.1, 0.1) is R-1 in a low-agreement pixel (x < 0.25), row 4 at (0.8, 0.3)
# is R-2 in a full-agreement pixel.
PRODUCTION_REGULAR_COORDS = {0: (0.1, 0.1), 4: (0.8, 0.3)}
PRODUCTION_CATEGORIES = {0: 1, 4: 2}
PRODUCTION_STORE_SETTINGS = {"store_mode": "production"}


def _build_production_session(base_dir, session_id, seed=0):
    """
    Description
    -----------
    Creates one session directory with a per-session spectrograms H5 (six
    calls, mask group) and a usv_summary CSV in the canonical production layout:
    the usv / squeak flags, coordinates of every USV map on rows 0 and 4,
    squeak-map coordinates on rows 4 and 5, and qlvm_category of the regular
    map's positions on the synthetic bundle; nulls elsewhere.

    Parameters
    ----------
    base_dir (pathlib.Path)
        Directory the session folder is created in.
    session_id (str)
        The session id (folder name).
    seed (int)
        Seed of the random spectrograms, masks and conditional-map coordinates.

    Returns
    -------
    root (pathlib.Path)
        The session root directory.
    """

    root = base_dir / session_id
    spec_dir = root / "audio" / "spectrograms"
    spec_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    n_rows = PRODUCTION_DURATIONS.shape[0]
    with h5py.File(spec_dir / f"{session_id}_spectrograms.h5", "w") as f:
        f.create_dataset("frequency_bins", data=np.linspace(30000.0, 120000.0, 16))
        grp = f.create_group(f"spectrogram/{session_id}")
        grp.create_dataset("spectrograms", data=rng.random((n_rows, 16, 8)).astype(np.float32), compression="gzip")
        grp.create_dataset("durations", data=PRODUCTION_DURATIONS)
        mgrp = f.create_group(f"mask/{session_id}")
        mgrp.create_dataset("segmentations", data=rng.random((PRODUCTION_MASK_INDEX.shape[0], 16, 8)) > 0.5)
        mgrp.create_dataset("spectrogram_index", data=PRODUCTION_MASK_INDEX)
    rows = {
        "usv_id": [f"{i:04d}" for i in range(n_rows)],
        "start": [0.1 + 0.2 * i for i in range(n_rows)],
        "stop": [0.15 + 0.2 * i for i in range(n_rows)],
        "usv": PRODUCTION_USV,
        "squeak": PRODUCTION_SQUEAK,
    }
    for qlvm_map in QLVM_MAPS:
        for axis in (1, 2):
            if qlvm_map == "qlvm":
                values = [PRODUCTION_REGULAR_COORDS[row][axis - 1] if row in PRODUCTION_EMBEDDED_ROWS else None for row in range(n_rows)]
            else:
                values = [float(rng.uniform(0.0, 1.0)) if row in PRODUCTION_EMBEDDED_ROWS else None for row in range(n_rows)]
            rows[f"{qlvm_map}{axis}"] = values
        if qlvm_map == "qlvm":
            rows["qlvm_category"] = [PRODUCTION_CATEGORIES[row] if row in PRODUCTION_EMBEDDED_ROWS else None for row in range(n_rows)]
    for axis in (1, 2):
        rows[f"{QLVM_SQUEAK_MAP}{axis}"] = [
            float(rng.uniform(0.0, 1.0)) if row in PRODUCTION_SQUEAK_ROWS else None for row in range(n_rows)
        ]
    pls.DataFrame(rows).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root


def _write_production_cells(tmp_path, monkeypatch, require_mask=True):
    """
    Description
    -----------
    Writes synthetic production cells (``checkpoint.tar`` bytes and
    ``config/training_contract.json``) for every USV map and the squeak map,
    under two package roots in ``tmp_path``, and points
    ``os_utils.QLVM_MODEL_PACKAGE_ROOT`` / ``QLVM_SQUEAK_PACKAGE_ROOT`` at them,
    so ``os_utils.qlvm_map_cell_directory`` resolves to them.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest's per-test directory.
    monkeypatch (pytest.MonkeyPatch)
        Pytest's monkeypatch fixture.
    require_mask (bool)
        The regular cell contract's ``require_mask``.

    Returns
    -------
    None
    """

    usv_root = tmp_path / "packages" / "masked_clean"
    squeak_root = tmp_path / "packages" / "squeaks"
    monkeypatch.setattr(os_utils, "QLVM_MODEL_PACKAGE_ROOT", str(usv_root))
    monkeypatch.setattr(os_utils, "QLVM_SQUEAK_PACKAGE_ROOT", str(squeak_root))
    for index, qlvm_map in enumerate((*QLVM_MAPS, QLVM_SQUEAK_MAP)):
        cell = pathlib.Path(qlvm_map_cell_directory(qlvm_map))
        (cell / "config").mkdir(parents=True)
        (cell / "checkpoint.tar").write_bytes(f"weights of {qlvm_map}".encode() * (index + 1))
        contract = {
            "conditional": None if qlvm_map in ("qlvm", QLVM_SQUEAK_MAP) else qlvm_map.removeprefix("qlvm_"),
            "length_threshold": 128.0,
            "require_mask": require_mask if qlvm_map == "qlvm" else qlvm_map != QLVM_SQUEAK_MAP,
            "masking_type": "none" if qlvm_map == QLVM_SQUEAK_MAP else "sam",
        }
        (cell / "config" / "training_contract.json").write_text(json.dumps(contract))


@pytest.fixture
def production_corpus(tmp_path, monkeypatch, qlvm_category_bundle):
    """
    Description
    -----------
    Two production sessions, the synthetic production cells and the synthetic
    category bundle with its agreement lowered to 0.5 on the pixels with
    x < 0.25 (below its uncertain_agreement of 0.6).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest's per-test directory.
    monkeypatch (pytest.MonkeyPatch)
        Pytest's monkeypatch fixture.
    qlvm_category_bundle (pathlib.Path)
        The synthetic bundle directory (conftest fixture).

    Returns
    -------
    roots (list[pathlib.Path])
        The two session roots.
    """

    grids_path = qlvm_category_bundle / "category_grids.npz"
    with np.load(grids_path) as grids:
        arrays = {key: grids[key] for key in grids.files}
    resolution = arrays["agreement"].shape[0]
    arrays["agreement"] = np.ones((resolution, resolution))
    arrays["agreement"][:, : resolution // 4] = 0.5
    np.savez_compressed(grids_path, **arrays)
    _write_production_cells(tmp_path, monkeypatch)
    return [_build_production_session(tmp_path / "data", session_id, seed=i) for i, session_id in enumerate(SESSIONS)]


def _consolidate_production(tmp_path, roots, mocker):
    """
    Description
    -----------
    Runs the consolidator in production mode into ``<tmp_path>/store`` with
    silent messages.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest's per-test directory.
    roots (list[pathlib.Path])
        Session roots, in store order.
    mocker (pytest_mock.MockerFixture)
        Pytest-mock fixture (silences smart_wait).

    Returns
    -------
    stores, returned (tuple)
        The files in the output directory and the returned store path.
    """

    mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.smart_wait")
    out_dir = tmp_path / "store"
    out_dir.mkdir(exist_ok=True)
    returned = css.SpectrogramStoreConsolidator(
        root_directories=[str(r) for r in roots],
        input_parameter_dict={
            "spectrograms_root": str(out_dir),
            "generate_spectrograms": SPEC_SETTINGS,
            "consolidate_spectrogram_store": PRODUCTION_STORE_SETTINGS,
        },
        message_output=lambda *_a, **_kw: None,
    ).consolidate_spectrogram_store()
    return sorted(out_dir.iterdir()), returned


def test_production_store_layout_values_and_provenance(tmp_path, mocker, production_corpus, qlvm_category_bundle):
    """The production store: file name, root attrs, byte-equal spectrogram and mask
    groups, coordinates of every map (NaN off the embedded rows; squeak map on the
    squeak rows only), status bit fields, qlvm_category with the bundle's agreement
    and uncertain flag, per-map provenance, the bundle's grids, the sessions table;
    the resolver picks it and the torus-traversal video accepts its provenance."""
    roots = production_corpus
    stores, returned = _consolidate_production(tmp_path, roots, mocker)
    assert stores == [returned]
    assert returned.name.startswith("spectrograms_production_2sessions_12vocalizations_")
    assert resolve_consolidated_h5_path(str(tmp_path / "store")) == str(returned)
    with h5py.File(returned, "r") as f:
        assert f.attrs["store_layout"] == "spectrograms_production"
        assert f.attrs["n_sessions"] == 2
        assert f.attrs["n_vocalizations"] == 12
        assert f.attrs["created_by"] == "consolidate-spectrogram-store"
        assert "category_bundle" in f.attrs["category_bundle_identity"]
        assert json.loads(f.attrs["generate_spectrograms_settings"]) == SPEC_SETTINGS
        assert json.loads(f["qlvm"].attrs["coordinate_datasets"]) == [*QLVM_MAPS, QLVM_SQUEAK_MAP]
        assert json.loads(f["qlvm"].attrs["status_codes"])["4"].startswith("bit 2: pure squeak")
        for root in roots:
            session_id = root.name
            summary = pls.read_csv(root / "audio" / f"{session_id}_usv_summary.csv")
            with h5py.File(root / "audio" / "spectrograms" / f"{session_id}_spectrograms.h5", "r") as source:
                for key in ("spectrogram/{s}/spectrograms", "spectrogram/{s}/durations", "mask/{s}/segmentations",
                            "mask/{s}/spectrogram_index"):
                    np.testing.assert_array_equal(f[key.format(s=session_id)][()], source[key.format(s=session_id)][()])
            assert "qlvm_dim" not in f[f"spectrogram/{session_id}"]
            np.testing.assert_array_equal(f[f"qlvm/{session_id}/status"][()], PRODUCTION_STATUS)
            assert f[f"qlvm/{session_id}/status"].dtype == np.int8
            for qlvm_map in (*QLVM_MAPS, QLVM_SQUEAK_MAP):
                expected = summary.select(f"{qlvm_map}1", f"{qlvm_map}2").fill_null(np.nan).to_numpy().astype(np.float64)
                stored = f[f"qlvm/{session_id}/{qlvm_map}"][()]
                assert stored.dtype == np.float64
                np.testing.assert_array_equal(stored, expected)
                placed_rows = PRODUCTION_SQUEAK_ROWS if qlvm_map == QLVM_SQUEAK_MAP else PRODUCTION_EMBEDDED_ROWS
                assert np.flatnonzero(~np.isnan(stored[:, 0])).tolist() == list(placed_rows)
            category = f[f"qlvm/{session_id}/qlvm_category"][()]
            assert category.dtype == np.int16
            np.testing.assert_array_equal(category, [1, 0, 0, 0, 2, 0])
            agreement = f[f"qlvm/{session_id}/qlvm_category_agreement"][()]
            np.testing.assert_array_equal(agreement[list(PRODUCTION_EMBEDDED_ROWS)], [0.5, 1.0])
            assert np.isnan(agreement[[1, 2, 3, 5]]).all()
            np.testing.assert_array_equal(
                f[f"qlvm/{session_id}/qlvm_category_uncertain"][()], [True, False, False, False, False, False]
            )

        for qlvm_map in (*QLVM_MAPS, QLVM_SQUEAK_MAP):
            attrs = f[f"qlvm_models/{qlvm_map}"].attrs
            cell_directory = qlvm_map_cell_directory(qlvm_map)
            assert attrs["cell_directory"] == cell_directory
            assert attrs["model_id"] == qlvm_cell_model_id(cell_directory)
            assert attrs["checkpoint_sha256"] == file_sha256(pathlib.Path(cell_directory) / "checkpoint.tar")
            contract = json.loads((pathlib.Path(cell_directory) / "config" / "training_contract.json").read_text())
            assert json.loads(attrs["training_contract"]) == contract
            assert attrs["condition"] == (contract["conditional"] if contract["conditional"] is not None else "none")
        assert check_store_map_provenance(f, "qlvm") == qlvm_cell_model_id(qlvm_map_cell_directory("qlvm"))

        categories = f["qlvm_categories"]
        assert categories.attrs["column"] == "qlvm_category"
        assert categories.attrs["uncertain_agreement"] == 0.6
        assert json.loads(categories.attrs["names"]) == ["R-1", "R-2", "R-3", "R-4"]
        assert categories.attrs["directory"] == str(qlvm_category_bundle)
        with np.load(qlvm_category_bundle / "category_grids.npz") as grids:
            np.testing.assert_array_equal(categories["label_grid"][()], grids["label_grid"])
            np.testing.assert_array_equal(categories["agreement"][()], grids["agreement"])

        sessions = _h5_to_polars(f["sessions"])
        assert sessions["session_id"].to_list() == list(SESSIONS)
        assert sessions["n_rows"].to_list() == [6, 6]
        assert sessions["n_embedded"].to_list() == [2, 2]
        assert sessions["n_squeak_embedded"].to_list() == [2, 2]
        assert sessions["spectrogram_h5_sha256"].to_list() == [
            file_sha256(root / "audio" / "spectrograms" / f"{root.name}_spectrograms.h5") for root in roots
        ]
        assert sessions["usv_summary_sha256"].to_list() == [
            file_sha256(root / "audio" / f"{root.name}_usv_summary.csv") for root in roots
        ]
        coords, index = pool_latents_from_h5(f, "qlvm")
    assert coords.shape == (4, 2)
    assert index == [(session_id, row) for session_id in SESSIONS for row in PRODUCTION_EMBEDDED_ROWS]


def test_production_store_refuses_inconsistent_sessions_together(tmp_path, mocker, production_corpus):
    """A missing column, a category from another bundle, regular and conditional
    coordinates on excluded rows, squeak coordinates without a squeak and a
    row-count mismatch are all reported in one error; nothing is written."""
    roots = production_corpus
    first, second = (root / "audio" / f"{root.name}_usv_summary.csv" for root in roots)
    pls.read_csv(first).drop("qlvm_squeak2").write_csv(first)
    pls.read_csv(second).with_columns(
        qlvm_category=pls.Series([3, None, None, None, 2, None]),
        qlvm1=pls.Series([0.1, 0.5, None, None, 0.8, None]),
        qlvm2=pls.Series([0.1, 0.5, None, None, 0.3, None]),
        qlvm_entropy1=pls.Series([0.2, None, 0.2, None, 0.2, None]),
        qlvm_entropy2=pls.Series([0.2, None, 0.2, None, 0.2, None]),
        qlvm_squeak1=pls.Series([0.4, None, None, None, 0.4, 0.4]),
        qlvm_squeak2=pls.Series([0.4, None, None, None, 0.4, 0.4]),
    ).write_csv(second)
    short = _build_production_session(tmp_path / "other", "20260103_120000", seed=7)
    short_summary = short / "audio" / "20260103_120000_usv_summary.csv"
    pls.read_csv(short_summary).head(5).write_csv(short_summary)
    with pytest.raises(ValueError, match="refused") as error:
        _consolidate_production(tmp_path, [*roots, short], mocker)
    message = str(error.value)
    assert f"{SESSIONS[0]}: {SESSIONS[0]}_usv_summary.csv lacks the column(s) ['qlvm_squeak2']" in message
    assert f"{SESSIONS[1]}: 0 rows with status 0 have no qlvm1/qlvm2 and 1 rows with a nonzero status have them" in message
    assert f"{SESSIONS[1]}: 1 rows with a nonzero status have qlvm_entropy1/qlvm_entropy2" in message
    assert f"{SESSIONS[1]}: 1 rows without a squeak have qlvm_squeak1/qlvm_squeak2" in message
    assert f"{SESSIONS[1]}: qlvm_category is not present exactly where qlvm1/qlvm2 are" in message
    assert "20260103_120000: inconsistent, 6 spectrogram rows but 5 USV summary rows" in message
    assert list((tmp_path / "store").iterdir()) == []


def test_production_store_refuses_a_category_from_another_bundle(tmp_path, mocker, production_corpus):
    """A qlvm_category that differs from the bundle's category of the call's pixel
    (the summary was labelled with another bundle) is refused."""
    roots = production_corpus
    summary_path = roots[0] / "audio" / f"{SESSIONS[0]}_usv_summary.csv"
    pls.read_csv(summary_path).with_columns(qlvm_category=pls.Series([4, None, None, None, 2, None])).write_csv(summary_path)
    with pytest.raises(ValueError, match=rf"{SESSIONS[0]}: 1 calls' qlvm_category differs from the category bundle's"):
        _consolidate_production(tmp_path, roots, mocker)
    assert list((tmp_path / "store").iterdir()) == []


@pytest.mark.usefixtures("qlvm_category_bundle")
def test_production_store_refuses_a_regular_cell_without_require_mask(tmp_path, mocker, monkeypatch):
    """The status rule assumes the regular cell drops maskless calls; a contract
    without require_mask is refused before any session is read."""
    _write_production_cells(tmp_path, monkeypatch, require_mask=False)
    roots = [_build_production_session(tmp_path / "data", SESSIONS[0])]
    with pytest.raises(ValueError, match="does not set require_mask"):
        _consolidate_production(tmp_path, roots, mocker)


def test_unknown_store_mode_is_refused(tmp_path, mocker):
    """A store_mode outside STORE_MODES raises before anything is read."""
    mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.smart_wait")
    consolidator = css.SpectrogramStoreConsolidator(
        root_directories=["/a/b"],
        input_parameter_dict={"spectrograms_root": str(tmp_path), "consolidate_spectrogram_store": {"store_mode": "v4"}},
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(ValueError, match=r"store_mode must be one of \['production', 'v3'\]"):
        consolidator.consolidate_spectrogram_store()


def test_cli_store_mode_reaches_the_settings_block(tmp_path, mocker):
    """--store-mode overrides consolidate_spectrogram_store.store_mode (shipped as
    production), --spectrograms-root overrides spectrograms_root, and an unknown
    mode is a usage error."""
    consolidator = mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.SpectrogramStoreConsolidator")
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--root-directories", "/a/b"])
    assert result.exit_code == 0, result.output
    assert consolidator.call_args.kwargs["input_parameter_dict"]["consolidate_spectrogram_store"]["store_mode"] == "production"
    result = CliRunner().invoke(
        css.consolidate_spectrogram_store_cli,
        ["--root-directories", "/a/b", "--store-mode", "v3", "--spectrograms-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    settings = consolidator.call_args.kwargs["input_parameter_dict"]
    assert settings["consolidate_spectrogram_store"]["store_mode"] == "v3"
    assert settings["spectrograms_root"] == str(tmp_path)
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--root-directories", "/a/b", "--store-mode", "v4"])
    assert result.exit_code == 2
