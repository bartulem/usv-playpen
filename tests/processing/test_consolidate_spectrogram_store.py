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
takes exactly one of --root-directories / --package-corpus, and
os_utils.resolve_consolidated_h5_path picks the newest store of either name.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import polars as pls
import pytest
from click.testing import CliRunner

from usv_playpen.analyses.usv_interval_archive import _h5_to_polars
from usv_playpen.os_utils import (
    QLVM_PRODUCTION_MODEL_CELLS,
    configure_path,
    resolve_consolidated_h5_path,
)
from usv_playpen.processing import consolidate_spectrogram_store as css
from usv_playpen.processing.build_qlvm_training_set import file_sha256
from usv_playpen.visualizations.qlvm_torus_traversal_video import pool_latents_from_h5

SESSIONS = ("20260101_120000", "20260102_120000")
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
    rows["qlvm_supercategory"] = [row + 1 if row in EMBEDDED_ROWS else None for row in range(n_usv)]
    pls.DataFrame(rows).write_csv(root / "audio" / f"{session_id}_usv_summary.csv")
    return root


def _write_fake_package(tmp_path, roots):
    """A QLVM model package in the v3 layout at ``<tmp_path>/qlvm_models_latest/v3``:
    MANIFEST.sha256, corpus/SESSION_H5_BASELINE.tsv listing ``roots`` with their H5
    SHA-256, and every production cell with config/training_contract.json,
    config/run_config.json and inference/clusters_fine/ (plus clusters_coarse/ with
    fine_to_coarse.csv for the regular cell, and a coarse level on a conditional cell
    too, which the store must not write); the regular cell's
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
        input_parameter_dict={"spectrograms_root": str(out_dir), "generate_spectrograms": SPEC_SETTINGS},
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
            for column in ("qlvm_category", "qlvm_supercategory", "qlvm_dur_category", "qlvm_mf_category",
                           "qlvm_bw_category", "qlvm_loud_category"):
                labels = f[f"qlvm/{session_id}/{column}"][()]
                assert labels.dtype == np.int16
                np.testing.assert_array_equal(labels, summary[column].fill_null(0).to_numpy())
                assert (labels[[1, 2, 3]] == 0).all()
                assert (labels[list(EMBEDDED_ROWS)] > 0).all()
            assert "qlvm_dur_supercategory" not in f[f"qlvm/{session_id}"]

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
            levels = ("fine", "coarse") if prefix == "qlvm" else ("fine",)
            assert sorted(json.loads(model.attrs["label_columns"])) == sorted(levels)
            for level in levels:
                directory = cell / "inference" / f"clusters_{level}"
                assert _h5_to_polars(model[f"clusters_{level}"]).equals(pls.read_csv(directory / "clusters.csv"))
                assert _h5_to_polars(model[f"boundaries_{level}"]).equals(pls.read_csv(directory / "boundaries.csv"))
                assert model[f"label_grid_{level}"].dtype == np.int16
                np.testing.assert_array_equal(model[f"label_grid_{level}"][()], np.load(directory / "label_grid.npy"))
            assert ("fine_to_coarse" in model) == (prefix == "qlvm")
            assert ("clusters_coarse" in model) == (prefix == "qlvm")

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


def test_non_default_label_level_is_stored_when_every_session_has_it(tmp_path, mocker, corpus):
    """qlvm_dur_supercategory (not a default level) is stored once every session carries it."""
    roots, package = corpus
    for root in roots:
        summary_path = root / "audio" / f"{root.name}_usv_summary.csv"
        summary = pls.read_csv(summary_path)
        summary.with_columns(qlvm_dur_supercategory=summary["qlvm_supercategory"]).write_csv(summary_path)
    _stores, returned = _consolidate(tmp_path, roots, package, mocker)
    with h5py.File(returned, "r") as f:
        assert f[f"qlvm/{SESSIONS[0]}/qlvm_dur_supercategory"].dtype == np.int16
        assert "clusters_coarse" in f["qlvm_models/qlvm_dur"]
        assert "fine_to_coarse" in f["qlvm_models/qlvm_dur"]


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
    """A summary without the production QLVM columns refuses the build, listing them."""
    roots, package = corpus
    summary_path = roots[0] / "audio" / f"{SESSIONS[0]}_usv_summary.csv"
    pls.read_csv(summary_path).drop("qlvm_bw1", "qlvm_loud_category").write_csv(summary_path)
    with pytest.raises(ValueError, match=r"lacks the QLVM column\(s\) \['qlvm_bw1', 'qlvm_loud_category'\]"):
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


def test_cli_takes_exactly_one_session_source(mocker, corpus):
    """--package-corpus resolves the package's sessions; --root-directories splits the
    list; giving both or neither is a usage error."""
    _roots, package = corpus
    consolidator = mocker.patch("usv_playpen.processing.consolidate_spectrogram_store.SpectrogramStoreConsolidator")
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--package-corpus", "--package-root", str(package)])
    assert result.exit_code == 0, result.output
    assert consolidator.call_args.kwargs["root_directories"] == css.package_corpus_root_directories(package)
    assert consolidator.call_args.kwargs["package_root"] == str(package)
    result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, ["--root-directories", "/a/b, /c/d"])
    assert result.exit_code == 0, result.output
    assert consolidator.call_args.kwargs["root_directories"] == ["/a/b", "/c/d"]
    for arguments in (["--root-directories", "/a/b", "--package-corpus"], []):
        result = CliRunner().invoke(css.consolidate_spectrogram_store_cli, arguments)
        assert result.exit_code == 2
        assert "exactly one of" in result.output


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
    """The torus-traversal video pools qlvm_dim of the new store (embedded rows only)."""
    roots, package = corpus
    _stores, returned = _consolidate(tmp_path, roots, package, mocker)
    with h5py.File(returned, "r") as f:
        coords, index = pool_latents_from_h5(f)
    assert coords.shape == (4, 2)
    assert index == [(session_id, row) for session_id in SESSIONS for row in EMBEDDED_ROWS]
