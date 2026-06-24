"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.patch_unit_catalog_peak_channel``.

Covers the position-joined coordinate fold (hand-computed), the
per-cluster within-shank triangulation against synthetic Kilosort
``.npy`` arrays, and the top-level catalog patcher end-to-end on a
fully synthetic ephys + histology + CSV tree under ``tmp_path``. The
template is constructed so its peak amplitude sits unambiguously on
shank 1, which forces ``closest_ch`` onto that shank and flips the
catalog ``brain_area`` from the shank-0 region to the shank-1 region.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from usv_playpen.neuropixels.patch_unit_catalog_peak_channel import (
    _build_folded_channel_locs,
    _cli,
    _load_ibl_channel_locations,
    _triangulate_within_shank_per_cluster,
    patch_unit_catalog_peak_channel,
)


def _ibl_json(brain_regions=("PAG", "PAG", "VISp", "VISp")):
    """
    Description
    -----------
    Build a four-channel IBL ``channel_locations`` dict: two channels
    on lateral 0 (shank 0) and two on lateral 250 (shank 1), each with
    explicit ``(x, y, z)`` and a brain region.

    Returns
    -------
    dict
        IBL-style ``channel_<i>`` mapping (plus an ``origin`` entry to
        confirm non-channel keys are ignored).
    """

    laterals = [0, 0, 250, 250]
    axials = [0, 20, 0, 20]
    xs = [0.0, 0.0, 10.0, 10.0]
    out = {"origin": {"x": 0, "y": 0, "z": 0}}
    for i, (lat, axi, x, region) in enumerate(zip(laterals, axials, xs, brain_regions)):
        out[f"channel_{i}"] = {
            "lateral": lat, "axial": axi,
            "x": x, "y": float(axi), "z": 0.0,
            "brain_region": region,
        }
    return out


_CHANNEL_POSITIONS = np.array([[0, 0], [0, 20], [250, 0], [250, 20]], dtype=np.int64)
_CHANNEL_SHANKS = np.array([0, 0, 1, 1], dtype=np.int64)


def test_build_folded_channel_locs_right_hemisphere_fold():
    """
    Description
    -----------
    On the right hemisphere the fold is
    ``y_folded = y + shank_width - (lateral % shank_spacing)``. With
    shank width 70 and spacing 250: a lateral-0 channel at y=0 folds to
    70; a lateral-250 channel (``250 % 250 == 0``) at y=20 folds to 90.
    """

    chan_locs, pos_to_region = _build_folded_channel_locs(
        _ibl_json(), "R", _CHANNEL_POSITIONS,
    )
    assert chan_locs.shape == (4, 3)
    # ch0: x=0, y_folded = 0 + 70 - (0 % 250) = 70
    np.testing.assert_allclose(chan_locs[0], [0.0, 70.0, 0.0])
    # ch3: x=10, y_folded = 20 + 70 - (250 % 250) = 90
    np.testing.assert_allclose(chan_locs[3], [10.0, 90.0, 0.0])
    assert pos_to_region[(250, 20)] == "VISp"


def test_build_folded_channel_locs_left_hemisphere_sign_flips():
    """
    Description
    -----------
    On the left hemisphere the fold sign flips:
    ``y_folded = y - shank_width + (lateral % shank_spacing)``; a
    lateral-0 channel at y=0 folds to -70.
    """

    chan_locs, _ = _build_folded_channel_locs(_ibl_json(), "L", _CHANNEL_POSITIONS)
    np.testing.assert_allclose(chan_locs[0], [0.0, -70.0, 0.0])


def test_load_ibl_channel_locations_reads_raw_json(tmp_path):
    """
    Description
    -----------
    The loader returns the raw IBL JSON dict from the
    ``<mouse>/<date>/ibl_RH/channel_locations.json`` path for the right
    hemisphere.
    """

    d = tmp_path / "M1" / "20240101" / "ibl_RH"
    d.mkdir(parents=True)
    (d / "channel_locations.json").write_text(json.dumps(_ibl_json()))
    loaded = _load_ibl_channel_locations(tmp_path, "M1", 20240101, "R")
    assert loaded["channel_2"]["brain_region"] == "VISp"


def _make_ks_dir(tmp_path):
    """
    Description
    -----------
    Write a minimal Kilosort directory: one template whose peak-to-peak
    amplitude is largest on channel 2 (shank 1), three spikes all in
    cluster 7 from template 0, plus the channel-shank map.

    Returns
    -------
    pathlib.Path
        The ``kilosort4`` directory.
    """

    ks_dir = tmp_path / "20240101_imec0" / "kilosort4"
    ks_dir.mkdir(parents=True)
    np.save(ks_dir / "spike_clusters.npy", np.array([7, 7, 7], dtype=np.int64))
    np.save(ks_dir / "spike_templates.npy", np.array([0, 0, 0], dtype=np.int64))
    np.save(ks_dir / "channel_shanks.npy", _CHANNEL_SHANKS)
    np.save(ks_dir / "channel_positions.npy", _CHANNEL_POSITIONS)
    template = np.zeros((1, 5, 4), dtype=np.float32)
    template[0, 0, 0], template[0, 4, 0] = -0.5, 0.5    # ptp 1.0
    template[0, 0, 1], template[0, 4, 1] = -0.5, 0.5    # ptp 1.0
    template[0, 0, 2], template[0, 4, 2] = -5.0, 5.0    # ptp 10.0 (peak)
    template[0, 0, 3], template[0, 4, 3] = -1.0, 1.0    # ptp 2.0
    np.save(ks_dir / "templates.npy", template)
    return ks_dir


def test_triangulate_within_shank_per_cluster_locks_to_peak_shank(tmp_path):
    """
    Description
    -----------
    The per-cluster triangulation resolves the primary template, reads
    its peak channel (channel 2 on shank 1), restricts the fit to that
    shank, and returns a ``closest_ch`` that stays on the peak shank
    (channels 2 or 3). A cluster with no spikes is absent from the
    result.
    """

    ks_dir = _make_ks_dir(tmp_path)
    chan_locs, _ = _build_folded_channel_locs(_ibl_json(), "R", _CHANNEL_POSITIONS)

    out = _triangulate_within_shank_per_cluster(ks_dir, chan_locs, [7, 99])
    assert set(out) == {7}                       # cluster 99 has no spikes
    res = out[7]
    assert res["template_peak_ch"] == 2
    assert res["closest_ch"] in (2, 3)           # stays on the peak shank
    assert all(isinstance(res[k], float) for k in ("loc_ml", "loc_ap", "loc_dv"))


def _make_catalog(tmp_path):
    """
    Description
    -----------
    Build the full synthetic tree: ``unit_catalog.csv`` with one PAG
    unit on ``imec0`` cluster 7, the matching Kilosort dir, and the IBL
    histology JSON.

    Returns
    -------
    tuple
        ``(catalog_path, ephys_root, histology_root)``.
    """

    ephys_root = tmp_path / "ephys"
    histology_root = tmp_path / "histology"
    _make_ks_dir(ephys_root)

    hist_dir = histology_root / "M1" / "20240101" / "ibl_RH"
    hist_dir.mkdir(parents=True)
    (hist_dir / "channel_locations.json").write_text(json.dumps(_ibl_json()))

    catalog_path = tmp_path / "unit_catalog.csv"
    pd.DataFrame([{
        "unit_id": "M1_20240101_imec0_cl0007",
        "mouse_id": "M1",
        "rec_date": 20240101,
        "closest_ch": 0,
        "brain_area": "PAG",
        "loc_ap": 0.0,
        "loc_ml": 0.0,
        "loc_dv": 0.0,
    }]).to_csv(catalog_path, index=False)
    return catalog_path, ephys_root, histology_root


def test_patch_catalog_dry_run_leaves_files_untouched(tmp_path):
    """
    Description
    -----------
    A dry run computes the diff (here the unit moves PAG → VISp) but
    writes nothing: no backup is created and the CSV bytes are
    unchanged.
    """

    catalog_path, ephys_root, histology_root = _make_catalog(tmp_path)
    before = catalog_path.read_bytes()

    summary = patch_unit_catalog_peak_channel(
        catalog_path=catalog_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        backup=True,
        dry_run=True,
    )
    assert summary["n_total"] == 1
    assert summary["backup_path"] is None
    assert summary["n_brain_area_changed"] == 1
    assert summary["brain_area_transitions"] == {"PAG->VISp": 1}
    assert catalog_path.read_bytes() == before


def test_patch_catalog_writes_backup_and_patches_columns(tmp_path):
    """
    Description
    -----------
    A live run drops a timestamped backup alongside the catalog and
    rewrites the unit's ``closest_ch`` onto the peak shank with
    ``brain_area`` updated to the shank-1 region (VISp).
    """

    catalog_path, ephys_root, histology_root = _make_catalog(tmp_path)
    summary = patch_unit_catalog_peak_channel(
        catalog_path=catalog_path,
        ephys_root=ephys_root,
        histology_root=histology_root,
        backup=True,
        dry_run=False,
    )
    assert summary["backup_path"] is not None
    backups = list(tmp_path.glob("unit_catalog.bak_*.csv"))
    assert len(backups) == 1

    patched = pd.read_csv(catalog_path)
    assert patched.loc[0, "brain_area"] == "VISp"
    assert int(patched.loc[0, "closest_ch"]) in (2, 3)


def test_patch_catalog_skips_when_kilosort_dir_missing(tmp_path):
    """
    Description
    -----------
    When the Kilosort directory for a probe-day is absent the rows are
    left untouched: nothing changes and no backup mismatch occurs.
    """

    catalog_path, _ephys_root, histology_root = _make_catalog(tmp_path)
    empty_ephys = tmp_path / "no_ephys"
    empty_ephys.mkdir()
    summary = patch_unit_catalog_peak_channel(
        catalog_path=catalog_path,
        ephys_root=empty_ephys,
        histology_root=histology_root,
        backup=False,
        dry_run=True,
    )
    assert summary["n_closest_ch_changed"] == 0
    assert summary["n_brain_area_changed"] == 0


def test_patch_catalog_skips_when_histology_json_missing(tmp_path):
    """
    Description
    -----------
    When the Kilosort directory exists but the IBL histology JSON for the
    session is absent, the probe-day is skipped (``FileNotFoundError``
    swallowed) and no rows change.
    """

    catalog_path, ephys_root, _histology_root = _make_catalog(tmp_path)
    empty_histology = tmp_path / "no_histology"
    empty_histology.mkdir()
    summary = patch_unit_catalog_peak_channel(
        catalog_path=catalog_path,
        ephys_root=ephys_root,
        histology_root=empty_histology,
        backup=False,
        dry_run=True,
    )
    assert summary["n_brain_area_changed"] == 0


def test_cli_dry_run_prints_summary(tmp_path, monkeypatch, capsys):
    """
    Description
    -----------
    The argparse entry point forwards its flags to
    :func:`patch_unit_catalog_peak_channel` and prints the JSON summary;
    a ``--dry-run`` ``--no-backup`` invocation over the synthetic dataset
    reports the PAG→VISp move without writing anything.
    """

    catalog_path, ephys_root, histology_root = _make_catalog(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--catalog-path", str(catalog_path),
        "--ephys-root", str(ephys_root),
        "--histology-root", str(histology_root),
        "--no-backup",
        "--dry-run",
    ])
    _cli()
    out = json.loads(capsys.readouterr().out)
    assert out["n_brain_area_changed"] == 1
    assert out["backup_path"] is None
