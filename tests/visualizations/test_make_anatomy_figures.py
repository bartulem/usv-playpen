"""
@author: bartulem
Smoke / unit tests for visualizations/make_anatomy_figures.py.

The module renders the multi-panel anatomy overview of the Neuropixels
dataset from a `unit_catalog.csv` plus the `brain_area_colors` palette.
Most rendering paths were previously untested (0 % coverage) because the
heavy 3D methods download Allen CCFv3 meshes over the network and the
waveform panel needs on-disk ephys / histology artefacts.

These tests exercise the tractable surface without any network or ephys
dependency:
  * the pure coordinate / parsing helpers (`_ccf_to_bregma`,
    `_load_obj_mesh`, `_classify_cell_type`);
  * catalog ingestion (`_load_catalog`, incl. the excluded-mouse filter
    and the derived `bucket` / `cell_type` columns);
  * the recording-yield two-panel figure (`make_recording_yield_figure`),
    rendered from a synthetic catalog and written to a tmp directory;
  * the 3D unit-positions figure (`make_unit_positions_figure`), with the
    Allen-mesh download stubbed out by a tiny in-repo cube OBJ so the
    builder, the per-bucket bounding-box computation, and the mesh-to-
    axes path all run against deterministic geometry.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

# Force a non-interactive matplotlib backend before the plotting module
# (which imports matplotlib at module scope) is pulled in, so the figure
# paths never try to open a window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_playpen.visualizations import make_anatomy_figures as maf
from usv_playpen.visualizations.make_anatomy_figures import (
    AnatomyFigureMaker,
    _ccf_to_bregma,
    _load_obj_mesh,
)


# A minimal axis-aligned cube in CCF µm (8 vertices, 12 triangle faces).
# OBJ indices are one-based; `_load_obj_mesh` is expected to drop them to
# zero-based. Used to stand in for every Allen structure mesh so the 3D
# builder runs against deterministic, network-free geometry.
_CUBE_OBJ = """\
v 0 0 0
v 1000 0 0
v 1000 1000 0
v 0 1000 0
v 0 0 1000
v 1000 0 1000
v 1000 1000 1000
v 0 1000 1000
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


# Seven-bucket hex palette matching `_BRAIN_AREA_BUCKETS`. Every bucket
# must be present or `_render_per_celltype_panel` / the 3D mesh loop trip
# a KeyError.
_BRAIN_AREA_COLORS = {
    "PAG": "#E6194B",
    "MRN": "#3CB44B",
    "VTA": "#4363D8",
    "MB": "#F58231",
    "CENT": "#911EB4",
    "SC": "#42D4F4",
    "other": "#9E9E9E",
}


@pytest.fixture
def viz_settings() -> dict:
    """
    Description
    -----------
    Minimal visualizations-settings dict carrying the two blocks the
    anatomy renderer reads: the seven-bucket `brain_area_colors` palette
    and a `figures` block. Save paths are always overridden per-call in
    these tests, so `figures.save_directory` is intentionally absent.

    Parameters
    ----------

    Returns
    -------
    settings (dict)
        Settings dict with `brain_area_colors` and `figures` keys.
    """

    return {
        "brain_area_colors": dict(_BRAIN_AREA_COLORS),
        "cell_type_colors": ["#1A1A1A", "#7A7A7A", "#CFCFCF"],
        "figures": {"fig_format": "png", "dpi": 100, "timestamp_in_name": False},
    }


def _write_catalog(path: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    Write a small synthetic `unit_catalog.csv` covering every cell-type
    and several brain-area buckets across two retained mice plus one
    excluded mouse (`147366`), so the loader's filter and the stacked-bar
    panels both have non-trivial input.

    Parameters
    ----------
    path (pathlib.Path)
        Destination CSV path.

    Returns
    -------
    path (pathlib.Path)
        The written CSV path (returned for call-site convenience).
    """

    rows = [
        # mouse 111111
        ("111111", 20240101, 0, "good", True, "PAG", 100, 5000.0, -5000.0, 0.0),
        ("111111", 20240101, 1, "good", False, "MRN", 101, 5010.0, -5010.0, 10.0),
        ("111111", 20240101, 2, "mua", False, "VTA", 102, 5020.0, -5020.0, 20.0),
        ("111111", 20240101, 3, "good", True, "SCdw", 103, 5030.0, -5030.0, 30.0),
        # mouse 222222
        ("222222", 20240102, 0, "good", True, "CENT2", 200, 4800.0, -4800.0, 5.0),
        ("222222", 20240102, 1, "mua", True, "ZI", 201, 4810.0, -4810.0, 15.0),
        ("222222", 20240102, 2, "good", True, "MB", 202, 4820.0, -4820.0, 25.0),
        # excluded mouse -> must be dropped by `_load_catalog`
        ("147366", 20240103, 0, "good", True, "PAG", 300, 4900.0, -4900.0, 0.0),
    ]
    columns = [
        "mouse_id", "rec_date", "unit_id", "cluster_group", "somatic",
        "brain_area", "closest_ch", "loc_ap", "loc_ml", "loc_dv",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


@pytest.fixture
def catalog_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    Materialise the synthetic catalog CSV in the test's tmp directory.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory.

    Returns
    -------
    path (pathlib.Path)
        Path to the written `unit_catalog.csv`.
    """

    return _write_catalog(tmp_path / "unit_catalog.csv")


def test_ccf_to_bregma_transform_matches_formula() -> None:
    """
    Description
    -----------
    `_ccf_to_bregma` must apply the documented per-axis transform:
    AP_bregma = 5400 - AP_ccf, DV_bregma = 332 - DV_ccf,
    ML_bregma = ML_ccf - 5739.

    Parameters
    ----------

    Returns
    -------
    None
    """

    verts_ccf = np.array([[0.0, 0.0, 0.0], [1000.0, 500.0, 6000.0]])
    out = _ccf_to_bregma(verts_ccf)
    expected = np.array([
        [5400.0, 332.0, -5739.0],
        [4400.0, -168.0, 261.0],
    ])
    np.testing.assert_allclose(out, expected)


def test_load_obj_mesh_parses_vertices_and_zero_based_faces(tmp_path: pathlib.Path) -> None:
    """
    Description
    -----------
    `_load_obj_mesh` must parse the `v` / `f` records of a Wavefront OBJ
    into an `(N, 3)` float vertex array and an `(M, 3)` int face array,
    converting OBJ's one-based vertex indices to zero-based.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory.

    Returns
    -------
    None
    """

    obj_path = tmp_path / "cube.obj"
    obj_path.write_text(_CUBE_OBJ)
    verts, faces = _load_obj_mesh(obj_path)
    assert verts.shape == (8, 3)
    assert faces.shape == (12, 3)
    # First OBJ face "f 1 2 3" -> zero-based [0, 1, 2].
    np.testing.assert_array_equal(faces[0], np.array([0, 1, 2]))
    assert faces.min() == 0 and faces.max() == 7


@pytest.mark.parametrize(
    ("cluster_group", "somatic", "expected"),
    [
        ("mua", True, "Multi-unit"),
        ("mua", False, "Multi-unit"),
        ("good", True, "Somatic"),
        ("good", False, "Non-somatic"),
    ],
)
def test_classify_cell_type_three_way(cluster_group: str, somatic: bool, expected: str) -> None:
    """
    Description
    -----------
    `_classify_cell_type` collapses every `mua` row to `Multi-unit`
    regardless of the somatic flag, and splits the remaining
    well-isolated single units into `Somatic` / `Non-somatic` on the
    somatic flag.

    Parameters
    ----------
    cluster_group (str)
        Kilosort curation label (`good` / `mua`).
    somatic (bool)
        Post-curation somatic annotation.
    expected (str)
        Expected three-way label.

    Returns
    -------
    None
    """

    row = pd.Series({"cluster_group": cluster_group, "somatic": somatic})
    assert AnatomyFigureMaker._classify_cell_type(row) == expected


def test_load_catalog_excludes_mouse_and_derives_columns(catalog_path, viz_settings) -> None:
    """
    Description
    -----------
    Construction must read the catalog, drop the excluded mouse
    (`147366`), and append the derived `bucket` and `cell_type` columns.
    Every derived bucket must be one of the seven canonical buckets.

    Parameters
    ----------
    catalog_path (pathlib.Path)
        Synthetic catalog CSV.
    viz_settings (dict)
        Visualizations-settings fixture.

    Returns
    -------
    None
    """

    maker = AnatomyFigureMaker(catalog_path, viz_settings, message_output=lambda *_: None)
    df = maker.catalog
    assert "147366" not in set(df["mouse_id"])
    assert len(df) == 7
    assert {"bucket", "cell_type"}.issubset(df.columns)
    assert set(df["bucket"]).issubset(set(_BRAIN_AREA_COLORS))
    assert set(df["cell_type"]).issubset({"Somatic", "Non-somatic", "Multi-unit"})


def test_make_recording_yield_figure_writes_file(catalog_path, viz_settings, tmp_path) -> None:
    """
    Description
    -----------
    `make_recording_yield_figure` must render both stacked-bar panels and
    write a single image file to the overridden output directory, logging
    the destination through `message_output`.

    Parameters
    ----------
    catalog_path (pathlib.Path)
        Synthetic catalog CSV.
    viz_settings (dict)
        Visualizations-settings fixture.
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory (used as the output dir).

    Returns
    -------
    None
    """

    messages: list[str] = []
    maker = AnatomyFigureMaker(catalog_path, viz_settings, message_output=messages.append)
    out_dir = tmp_path / "figs"
    # Other test modules may leak open figures onto pyplot's global stack;
    # assert this call leaves the stack as it found it (i.e. closed its own
    # figure) rather than that the stack is globally empty.
    fig_nums_before = plt.get_fignums()
    out_path = maker.make_recording_yield_figure(out_dir=out_dir, fig_format="png")
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    assert out_path.parent == out_dir
    assert any("recording-yield" in m for m in messages)
    assert plt.get_fignums() == fig_nums_before


def test_make_unit_positions_figure_writes_file(catalog_path, viz_settings, tmp_path, monkeypatch) -> None:
    """
    Description
    -----------
    `make_unit_positions_figure` must build the 3D scene (whole-brain
    shell scatter, six translucent bucket meshes, per-bucket unit dots)
    and write a static image, with the Allen-mesh download stubbed by a
    tiny in-repo cube OBJ. `filter_outliers=False` keeps the synthetic
    dots so the per-bucket scatter branch is exercised; mesh I/O,
    `_compute_bucket_bboxes`, and `_add_mesh_to_axes` all run against the
    cube geometry.

    Parameters
    ----------
    catalog_path (pathlib.Path)
        Synthetic catalog CSV.
    viz_settings (dict)
        Visualizations-settings fixture.
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory.
    monkeypatch (pytest.MonkeyPatch)
        Used to redirect `_download_allen_mesh` to the cube OBJ.

    Returns
    -------
    None
    """

    cube_path = tmp_path / "cube.obj"
    cube_path.write_text(_CUBE_OBJ)
    monkeypatch.setattr(maf, "_download_allen_mesh", lambda structure_id: cube_path)

    maker = AnatomyFigureMaker(catalog_path, viz_settings, message_output=lambda *_: None)
    out_dir = tmp_path / "figs3d"
    fig_nums_before = plt.get_fignums()
    out_path = maker.make_unit_positions_figure(
        out_dir=out_dir,
        fig_format="png",
        shell_vertex_stride=1,
        rasterize_dense=False,
        filter_outliers=False,
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    assert plt.get_fignums() == fig_nums_before


def test_build_unit_positions_figure_filters_outliers(catalog_path, viz_settings, tmp_path, monkeypatch) -> None:
    """
    Description
    -----------
    With `filter_outliers=True` (the default), `build_unit_positions_figure`
    drops dots whose stereotaxic coordinates fall outside their bucket's
    Allen-mesh bounding box. The synthetic catalog's `loc_*` values lie
    far outside the cube stand-in mesh, so every dot is filtered and the
    pooled-scatter branch is skipped -- yet the builder must still return
    a valid 3D figure.

    Parameters
    ----------
    catalog_path (pathlib.Path)
        Synthetic catalog CSV.
    viz_settings (dict)
        Visualizations-settings fixture.
    tmp_path (pathlib.Path)
        Pytest-provided temporary directory.
    monkeypatch (pytest.MonkeyPatch)
        Used to redirect `_download_allen_mesh` to the cube OBJ.

    Returns
    -------
    None
    """

    cube_path = tmp_path / "cube.obj"
    cube_path.write_text(_CUBE_OBJ)
    monkeypatch.setattr(maf, "_download_allen_mesh", lambda structure_id: cube_path)

    maker = AnatomyFigureMaker(catalog_path, viz_settings, message_output=lambda *_: None)
    fig = maker.build_unit_positions_figure(
        shell_vertex_stride=1,
        rasterize_dense=False,
        filter_outliers=True,
    )
    assert fig.axes
    assert fig.axes[0].name == "3d"
    plt.close(fig)


# ===========================================================================
# Single-unit waveform helpers: catalog slicing, per-probe Kilosort context
# loading, IBL brain-coordinate parsing, and the two LineCollection-based
# waveform renderers. Driven by a synthetic Kilosort probe directory plus a
# catalog whose unit_id strings carry the `imec<d>` / `cl<dddd>` tokens the
# slicer parses.
# ===========================================================================


# 8-channel single-shank probe: two lateral columns (0 / 32 µm) over four
# axial rows (0 / 20 / 40 / 60 µm).
_WF_POSITIONS = np.array(
    [[0, 0], [32, 0], [0, 20], [32, 20], [0, 40], [32, 40], [0, 60], [32, 60]],
    dtype=float,
)
_WF_N_CHANNELS = 8
_WF_N_SAMPLES = 60


def _build_ks_probe(ephys_root: pathlib.Path, rec_date: int) -> pathlib.Path:
    """
    Description
    -----------
    Materialise a synthetic `<ephys_root>/<rec_date>_imec0/kilosort4/`
    directory carrying the Kilosort assets `_gather_probe_context_for_unit`
    and the waveform renderers read: `spike_clusters.npy`,
    `spike_templates.npy`, `templates.npy`, `channel_positions.npy`, and
    `channel_shanks.npy`. Cluster 5 maps to template 0 (trough on ch 3),
    cluster 7 to template 1 (trough on ch 5).

    Parameters
    ----------
    ephys_root (pathlib.Path)
        Root holding the per-probe directories.
    rec_date (int)
        Recording date as YYYYMMDD integer.

    Returns
    -------
    ks_dir (pathlib.Path)
        The written `kilosort4` directory.
    """

    ks_dir = ephys_root / f"{rec_date}_imec0" / "kilosort4"
    ks_dir.mkdir(parents=True)

    spike_clusters = np.array([5] * 40 + [7] * 40, dtype=np.int64)
    spike_templates = np.array([0] * 40 + [1] * 40, dtype=np.int64)
    templates = np.zeros((3, _WF_N_SAMPLES, _WF_N_CHANNELS), dtype=np.float32)
    # Negative trough mid-window on the respective peak channels.
    trough = -np.exp(-((np.arange(_WF_N_SAMPLES) - 30) ** 2) / (2 * 5.0 ** 2))
    templates[0, :, 3] = trough
    templates[1, :, 5] = trough

    np.save(ks_dir / "spike_clusters.npy", spike_clusters)
    np.save(ks_dir / "spike_templates.npy", spike_templates)
    np.save(ks_dir / "templates.npy", templates)
    np.save(ks_dir / "channel_positions.npy", _WF_POSITIONS)
    np.save(ks_dir / "channel_shanks.npy", np.zeros(_WF_N_CHANNELS, dtype=np.int64))
    return ks_dir


def _write_waveform_catalog(path: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    Write a `unit_catalog.csv` whose `unit_id` strings carry the
    `imec0` / `cl0005` tokens `_collect_session_clusters` parses, with two
    good + somatic clusters (PAG peak-ch 3, MRN peak-ch 5) and one MUA row
    that the slicer must drop.

    Parameters
    ----------
    path (pathlib.Path)
        Destination CSV path.

    Returns
    -------
    path (pathlib.Path)
        The written CSV path.
    """

    rows = [
        ("111111", 20240101, "imec0_cl0005_ch103_good", "good", True, "PAG", 3, 5000.0, -5000.0, 0.0),
        ("111111", 20240101, "imec0_cl0007_ch105_good", "good", True, "MRN", 5, 5010.0, -5010.0, 10.0),
        ("111111", 20240101, "imec0_cl0009_ch107_mua",  "mua", True, "VTA", 7, 5020.0, -5020.0, 20.0),
        # good + somatic but absent from spike_clusters -> exercises the
        # "no spikes for this cluster -> skip template" branch in
        # _gather_probe_context_for_unit.
        ("111111", 20240101, "imec0_cl0011_ch101_good", "good", True, "PAG", 1, 5030.0, -5030.0, 30.0),
    ]
    columns = [
        "mouse_id", "rec_date", "unit_id", "cluster_group", "somatic",
        "brain_area", "closest_ch", "loc_ap", "loc_ml", "loc_dv",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def _waveform_maker(tmp_path, viz_settings):
    """
    Description
    -----------
    Construct an `AnatomyFigureMaker` backed by the waveform catalog.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    viz_settings (dict)
        Visualizations-settings fixture.

    Returns
    -------
    maker (AnatomyFigureMaker)
        Maker wired to the waveform catalog.
    """

    cat = _write_waveform_catalog(tmp_path / "wf_catalog.csv")
    return maf.AnatomyFigureMaker(cat, viz_settings, message_output=lambda *_: None)


def test_collect_session_clusters_adds_probe_and_cluster_num(tmp_path, viz_settings):
    """
    Description
    -----------
    `_collect_session_clusters` must slice the catalog to one mouse/day's
    good + somatic clusters and decorate each row with the parsed `probe`
    (`imec0`) and integer `cluster_num`. The MUA row is excluded.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    viz_settings (dict)
        Visualizations-settings fixture.

    Returns
    -------
    None
    """

    maker = _waveform_maker(tmp_path, viz_settings)
    sub = maker._collect_session_clusters("111111", 20240101)
    assert sorted(sub["cluster_num"].tolist()) == [5, 7, 11]
    assert set(sub["probe"]) == {"imec0"}


def test_gather_probe_context_builds_templates_and_mappings(tmp_path, viz_settings):
    """
    Description
    -----------
    `_gather_probe_context_for_unit` must load the probe's channel
    positions and, per cluster, resolve the modal Kilosort template into a
    `(n_samples, n_channels)` array, alongside the cluster->bucket and
    cluster->peak-channel maps.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    viz_settings (dict)
        Visualizations-settings fixture.

    Returns
    -------
    None
    """

    maker = _waveform_maker(tmp_path, viz_settings)
    ephys_root = tmp_path / "ephys"
    _build_ks_probe(ephys_root, 20240101)
    clusters = maker._collect_session_clusters("111111", 20240101)

    ctx = maker._gather_probe_context_for_unit(
        ephys_root=ephys_root, rec_date=20240101, probe="imec0", clusters=clusters,
    )
    assert ctx["cluster_to_bucket"] == {5: "PAG", 7: "MRN", 11: "PAG"}
    assert ctx["cluster_to_peakch"] == {5: 3, 7: 5, 11: 1}
    assert ctx["cluster_templates"][5].shape == (_WF_N_SAMPLES, _WF_N_CHANNELS)
    # Cluster 11 has no spikes in spike_clusters -> no resolved template.
    assert 11 not in ctx["cluster_templates"]
    assert ctx["channel_positions"].shape == (_WF_N_CHANNELS, 2)


@pytest.mark.parametrize("hemisphere,sub", [("L", "ibl_LH"), ("R", "ibl_RH")])
def test_load_ibl_brain_coords_both_hemispheres(tmp_path, hemisphere, sub):
    """
    Description
    -----------
    `_load_ibl_brain_coords` must read the per-channel `(x, y, z)` from a
    session's `channel_locations.json` into `ml` / `ap` / `dv` arrays,
    selecting `ibl_LH` vs `ibl_RH` by hemisphere.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    hemisphere (str)
        `'L'` or `'R'`.
    sub (str)
        Expected hemisphere subdirectory name.

    Returns
    -------
    None
    """

    hist_root = tmp_path / "histology"
    ch_dir = hist_root / "111111" / "20240101" / sub
    ch_dir.mkdir(parents=True)
    payload = {
        f"channel_{ch}": {"x": float(ch), "y": float(ch + 100), "z": float(ch + 200)}
        for ch in range(_WF_N_CHANNELS)
    }
    payload["origin"] = {"x": 0, "y": 0, "z": 0}  # non-channel key, must be ignored
    (ch_dir / "channel_locations.json").write_text(json.dumps(payload))

    coords = maf.AnatomyFigureMaker._load_ibl_brain_coords(
        histology_root=hist_root, mouse_id="111111", rec_date=20240101,
        hemisphere=hemisphere,
    )
    assert coords["ml"].shape == (_WF_N_CHANNELS,)
    assert coords["ap"][0] == 100.0 and coords["dv"][0] == 200.0


def test_draw_single_unit_waveforms_adds_line_collections(tmp_path, viz_settings):
    """
    Description
    -----------
    `_draw_single_unit_waveforms` must render the peak + neighbour-channel
    waveforms as opacity-binned `LineCollection`s onto the axes and zoom to
    a box around the peak channel. Asserts at least one collection is added.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    viz_settings (dict)
        Visualizations-settings fixture.

    Returns
    -------
    None
    """

    maker = _waveform_maker(tmp_path, viz_settings)
    ephys_root = tmp_path / "ephys"
    _build_ks_probe(ephys_root, 20240101)
    clusters = maker._collect_session_clusters("111111", 20240101)
    ctx = maker._gather_probe_context_for_unit(
        ephys_root=ephys_root, rec_date=20240101, probe="imec0", clusters=clusters,
    )

    fig, ax = plt.subplots()
    try:
        maker._draw_single_unit_waveforms(
            ax, ctx=ctx, cluster_num=5, template=ctx["cluster_templates"][5],
            peakch=3, waveform_width_um=20.0, waveform_voltage_uv_scale=10.0,
            opacity_sigma_um=40.0, n_neighbors_each_side=2,
            zoom_axial_um=50.0, zoom_lateral_um=50.0,
        )
        assert len(ax.collections) >= 1, "no waveform LineCollections drawn"
    finally:
        plt.close(fig)


def test_draw_single_unit_waveforms_in_brain_space_returns_positions(tmp_path, viz_settings):
    """
    Description
    -----------
    `_draw_single_unit_waveforms_in_brain_space` must render the same
    neighbour set and return the lateral/axial centres of every drawn
    channel (peak + sibling + 2 above + 2 below = 6). Asserts the returned
    arrays' length and that a collection was added.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temp directory.
    viz_settings (dict)
        Visualizations-settings fixture.

    Returns
    -------
    None
    """

    maker = _waveform_maker(tmp_path, viz_settings)
    ephys_root = tmp_path / "ephys"
    _build_ks_probe(ephys_root, 20240101)
    clusters = maker._collect_session_clusters("111111", 20240101)
    ctx = maker._gather_probe_context_for_unit(
        ephys_root=ephys_root, rec_date=20240101, probe="imec0", clusters=clusters,
    )

    fig, ax = plt.subplots()
    try:
        ap_drawn, dv_drawn = maker._draw_single_unit_waveforms_in_brain_space(
            ax, ctx=ctx, cluster_num=5, template=ctx["cluster_templates"][5],
            peakch=3, waveform_width_um=20.0, waveform_voltage_uv_scale=10.0,
            opacity_sigma_um=40.0, n_neighbors_each_side=2, lateral_offset_um=5.0,
        )
        assert ap_drawn.shape == dv_drawn.shape == (6,)
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)


def test_download_allen_mesh_uses_cache_without_network(tmp_path, monkeypatch) -> None:
    """
    Description
    -----------
    `_download_allen_mesh` must return an already-cached OBJ without
    touching the network: with the structure's `.obj` present under the
    (redirected) cache directory, `urlopen` must never be called.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory (stand-in mesh cache).
    monkeypatch (pytest.MonkeyPatch)
        Redirects the cache dir and traps `urlopen`.

    Returns
    -------
    None
    """

    monkeypatch.setattr(maf, "_ALLEN_MESH_CACHE", tmp_path)
    cached = tmp_path / "997.obj"
    cached.write_text("v 0 0 0\n")

    def _boom(*_a, **_k):
        raise AssertionError("urlopen must not be called when the mesh is cached")

    monkeypatch.setattr(maf.urllib.request, "urlopen", _boom)
    assert maf._download_allen_mesh(997) == cached


def test_anatomy_maker_raises_on_missing_catalog(tmp_path, viz_settings) -> None:
    """
    Description
    -----------
    The catalog is the authoritative unit scope, so constructing an
    `AnatomyFigureMaker` against a non-existent `unit_catalog.csv` must
    fail loud with a clear `FileNotFoundError` rather than a deep pandas
    traceback.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temp directory.
    viz_settings (dict)
        Visualizations-parameter dict fixture.

    Returns
    -------
    None
    """

    with pytest.raises(FileNotFoundError, match="unit_catalog.csv not found"):
        AnatomyFigureMaker(
            catalog_path=tmp_path / "does_not_exist.csv",
            visualizations_parameter_dict=viz_settings,
            message_output=lambda *a, **k: None,
        )
