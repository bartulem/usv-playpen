"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.histology_ibl_alignment_export``.

Two tiers of checks:

1. Self-contained tests that exercise the pure-numpy helpers
   (:func:`ccf_apdvml_to_xyz_mlapdv_um`, :func:`parse_imro_table`,
   :func:`read_ap_meta`, :func:`sample_to_volts_ap`) without touching the
   network or any external atlas. These run anywhere the package can be
   imported.

2. Cross-checks against ``iblatlas`` that guard against silent drift of
   the Allen CCF bregma landmark or the apdvml→mlapdv affine. The
   runtime module deliberately does **not** depend on ``iblatlas`` — see
   the rationale in
   ``src/usv_playpen/neuropixels/histology_ibl_alignment_export.py``
   for why the constants are pinned inline — but if the upstream atlas
   ever rebases on a future CCFv4 or recalibrates the bregma offset,
   these tests are how we find out about it. The tests are skipped when
   ``iblatlas`` is not installed; install it via the ``test`` dependency
   group (``pip install -e .[test]`` or ``uv sync --group test``) to
   enable them.

The cross-checks use ``AllenAtlas(mock=True)`` so the test suite never
downloads the ~300 MB Allen NRRD volumes; ``mock=True`` still installs
the ``BrainCoordinates`` affine that ``ccf2xyz`` relies on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from usv_playpen.neuropixels.histology_ibl_alignment_export import (
    ALLEN_BREGMA_MLAPDV_UM,
    IBLAlignmentExporter,
    NP2_PROBE_TYPES,
    N_CLOSEST_CHANNELS,
    _REQUIRED_COPIES,
    ccf_apdvml_to_xyz_mlapdv_um,
    parse_imro_table,
    read_ap_meta,
    sample_to_volts_ap,
)


def test_ccf_bregma_maps_to_origin():
    """
    Description
    -----------
    The Allen CCF bregma landmark, expressed in CCF apdvml voxel-origin
    micrometres, must map to the origin of IBL mlapdv-space (bregma is
    by construction at ``[0, 0, 0]`` mlapdv). Verifies the column
    permutation, translation and axis flips all line up.
    """
    apdvml = ALLEN_BREGMA_MLAPDV_UM[[1, 2, 0]].reshape(1, 3)
    out = ccf_apdvml_to_xyz_mlapdv_um(apdvml)
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.0]], atol=0)


def test_ccf_axis_flips_are_correct():
    """
    Description
    -----------
    Moving 100 µm posterior in CCF AP should produce an mlapdv AP of
    -100 µm (AP is negated under the apdvml→mlapdv flip). Moving 100 µm
    deeper in CCF DV should produce an mlapdv DV of -100 µm. Moving
    100 µm lateral-right in CCF ML should produce an mlapdv ML of
    +100 µm (no flip on the ML axis).
    """
    posterior = np.array([ALLEN_BREGMA_MLAPDV_UM[1] + 100,
                          ALLEN_BREGMA_MLAPDV_UM[2],
                          ALLEN_BREGMA_MLAPDV_UM[0]]).reshape(1, 3)
    deeper = np.array([ALLEN_BREGMA_MLAPDV_UM[1],
                       ALLEN_BREGMA_MLAPDV_UM[2] + 100,
                       ALLEN_BREGMA_MLAPDV_UM[0]]).reshape(1, 3)
    lateral_right = np.array([ALLEN_BREGMA_MLAPDV_UM[1],
                              ALLEN_BREGMA_MLAPDV_UM[2],
                              ALLEN_BREGMA_MLAPDV_UM[0] + 100]).reshape(1, 3)
    np.testing.assert_allclose(ccf_apdvml_to_xyz_mlapdv_um(posterior),
                               [[0.0, -100.0, 0.0]], atol=0)
    np.testing.assert_allclose(ccf_apdvml_to_xyz_mlapdv_um(deeper),
                               [[0.0, 0.0, -100.0]], atol=0)
    np.testing.assert_allclose(ccf_apdvml_to_xyz_mlapdv_um(lateral_right),
                               [[100.0, 0.0, 0.0]], atol=0)


def test_ccf_broadcasts_over_leading_dims():
    """
    Description
    -----------
    The transform should broadcast over any leading shape. A
    ``(2, 3, 3)`` input must yield a ``(2, 3, 3)`` output with the same
    per-point conversion applied independently.
    """
    pts = np.random.default_rng(0).uniform(0, 10000, size=(2, 3, 3))
    out = ccf_apdvml_to_xyz_mlapdv_um(pts)
    assert out.shape == pts.shape
    flat_in = pts.reshape(-1, 3)
    flat_out = ccf_apdvml_to_xyz_mlapdv_um(flat_in)
    np.testing.assert_allclose(out.reshape(-1, 3), flat_out)


def test_parse_imro_table_np2_multishank_rows():
    """
    Description
    -----------
    A canonical NP2.0 four-shank IMRO header
    (``(probe_type, n_channels)(chan shank bank refid elecid)...``)
    should parse to a list whose first row is the
    ``[probe_type, n_channels]`` header and subsequent rows are
    5-integer channel descriptors with the shank index in column 1.
    """
    raw = "(2013,384)(0 3 1 0 576)(1 3 1 0 577)(2 2 1 0 578)"
    parsed = parse_imro_table(raw)
    assert parsed[0] == [2013, 384]
    assert parsed[1] == [0, 3, 1, 0, 576]
    assert parsed[2] == [1, 3, 1, 0, 577]
    assert parsed[3] == [2, 2, 1, 0, 578]


def test_parse_imro_table_snsgeommap_colon_format():
    """
    Description
    -----------
    The SpikeGLX ``snsGeomMap`` field uses colon-delimited 4-tuples
    inside each parenthesised group; the parser should pick ``:`` as
    the delimiter and return the four integer fields per row.
    """
    raw = "(0,0)(0:27:0:1)(0:59:0:1)"
    parsed = parse_imro_table(raw)
    assert parsed == [[0, 0], [0, 27, 0, 1], [0, 59, 0, 1]]


def test_parse_imro_table_empty_string_returns_empty_list():
    """
    Description
    -----------
    A falsy ``data_string`` short-circuits before the regex pass and
    returns an empty list — the documented behaviour of the parser.
    """
    assert parse_imro_table("") == []


def test_read_ap_meta_strips_tildes_and_parses_canonical_keys(tmp_path):
    """
    Description
    -----------
    SpikeGLX prefixes multi-line / structured metadata keys with ``~``
    (e.g. ``~imroTbl``, ``~snsGeomMap``). :func:`read_ap_meta` must
    drop those tildes so callers can use the canonical IBL/spikeglx
    key names (``meta['imroTbl']`` rather than ``meta['~imroTbl']``).
    Surrounding whitespace must be stripped and lines without ``=``
    must be skipped silently.
    """
    meta_path = tmp_path / "fake.ap.meta"
    meta_path.write_text(
        "imSampRate=30000.0\n"
        "imAiRangeMax=0.62\n"
        "imMaxInt=2048\n"
        "imDatPrb_type=2013\n"
        "~imroTbl=(2013,384)(0 3 1 0 576)\n"
        "garbage line without equals\n"
        "  ~snsGeomMap  =  (0,0)(0:27:0:1)  \n",
        encoding="utf-8",
    )
    meta = read_ap_meta(meta_path)
    assert meta["imSampRate"] == "30000.0"
    assert meta["imAiRangeMax"] == "0.62"
    assert meta["imMaxInt"] == "2048"
    assert meta["imDatPrb_type"] == "2013"
    assert meta["imroTbl"].startswith("(2013,384)")
    assert meta["snsGeomMap"].startswith("(0,0)")
    assert "~imroTbl" not in meta
    assert "~snsGeomMap" not in meta


def test_sample_to_volts_ap_np2_known_constants():
    """
    Description
    -----------
    For a Neuropixels 2.0 probe (``imDatPrb_type=2013``) with
    ``imAiRangeMax=0.62`` and ``imMaxInt=2048`` the int16→volt scaling
    must reduce to ``0.62 / 2048 / 80`` exactly. This is the value used
    by the IBL pipeline for the project's reference session.
    """
    meta = {
        "imDatPrb_type": "2013",
        "imAiRangeMax": "0.62",
        "imMaxInt": "2048",
    }
    np.testing.assert_allclose(sample_to_volts_ap(meta), 0.62 / 2048 / 80)


@pytest.mark.parametrize("bad_probe_type", [0, 1020, 1100, 1300, 9999])
def test_sample_to_volts_ap_raises_for_non_np2(bad_probe_type):
    """
    Description
    -----------
    The module supports only Neuropixels 2.0 probes (see
    :data:`NP2_PROBE_TYPES`). Calling
    :func:`sample_to_volts_ap` with any other probe type — including
    Neuropixels 1.0 (``imDatPrb_type=0``) — must raise
    ``NotImplementedError`` so a downstream caller cannot silently
    apply the wrong gain formula.
    """
    meta = {
        "imDatPrb_type": str(bad_probe_type),
        "imAiRangeMax": "0.62",
        "imMaxInt": "2048",
    }
    assert bad_probe_type not in NP2_PROBE_TYPES
    with pytest.raises(NotImplementedError):
        sample_to_volts_ap(meta)


def _bare_exporter(**attrs):
    """
    Description
    -----------
    Build an :class:`IBLAlignmentExporter` without running ``__init__``
    (which requires on-disk meta files), then set only the attributes a
    given pure-numpy helper consumes. Lets the array helpers be tested
    in isolation from the file-discovery constructor.

    Parameters
    ----------
    **attrs
        Attribute name → value pairs to set on the bare instance.

    Returns
    -------
    IBLAlignmentExporter
        An uninitialised instance carrying only ``attrs``.
    """

    exporter = object.__new__(IBLAlignmentExporter)
    for key, value in attrs.items():
        setattr(exporter, key, value)
    return exporter


def test_constructor_rejects_invalid_hemisphere(tmp_path):
    """
    Description
    -----------
    The hemisphere must be ``'L'`` or ``'R'``; any other value raises
    ``ValueError`` before any filesystem access.
    """

    with pytest.raises(ValueError, match="hemisphere must be 'L' or 'R'"):
        IBLAlignmentExporter(tmp_path, "M1", "20240101", "imec0", "X")


def test_constructor_raises_when_no_meta_found(tmp_path):
    """
    Description
    -----------
    With a valid hemisphere but no ``concatenated_*.ap.meta`` under the
    EPHYS directory, construction raises ``FileNotFoundError``.
    """

    with pytest.raises(FileNotFoundError, match="No concatenated_.*ap.meta"):
        IBLAlignmentExporter(tmp_path, "M1", "20240101", "imec0", "R")


def test_constructor_raises_on_multiple_meta(tmp_path):
    """
    Description
    -----------
    Two ``concatenated_*.ap.meta`` files are ambiguous and raise
    ``RuntimeError``.
    """

    ephys = tmp_path / "EPHYS" / "20240101_imec0"
    ephys.mkdir(parents=True)
    (ephys / "concatenated_a.ap.meta").write_text("imDatPrb_type=2013\n")
    (ephys / "concatenated_b.ap.meta").write_text("imDatPrb_type=2013\n")
    with pytest.raises(RuntimeError, match="Multiple concatenated"):
        IBLAlignmentExporter(tmp_path, "M1", "20240101", "imec0", "R")


def test_constructor_rejects_non_np2_probe(tmp_path):
    """
    Description
    -----------
    A non-Neuropixels-2.0 probe type in the meta raises
    ``NotImplementedError``.
    """

    ephys = tmp_path / "EPHYS" / "20240101_imec0"
    ephys.mkdir(parents=True)
    (ephys / "concatenated_x.ap.meta").write_text("imDatPrb_type=0\n")
    with pytest.raises(NotImplementedError, match="not a supported Neuropixels 2.0"):
        IBLAlignmentExporter(tmp_path, "M1", "20240101", "imec0", "R")


def test_constructor_resolves_paths_and_parses_meta(tmp_path):
    """
    Description
    -----------
    A successful construction (valid hemisphere + a single NP2.0
    ``concatenated_*.ap.meta``) resolves the EPHYS / Kilosort / output
    paths, parses the meta, records the NP2.0 probe type and multishank
    flag, and computes the int16→volt scale and IMRO / geom tables.
    """

    ephys = tmp_path / "EPHYS" / "20240101_imec0"
    ephys.mkdir(parents=True)
    (ephys / "concatenated_run.ap.meta").write_text(
        "imSampRate=30000.0\n"
        "imAiRangeMax=0.62\n"
        "imMaxInt=2048\n"
        "imDatPrb_type=2013\n"
        "~imroTbl=(2013,384)(0 3 1 0 576)\n"
        "~snsGeomMap=(0,0)(0:27:0:1)\n"
    )
    exporter = IBLAlignmentExporter(tmp_path, "M1", "20240101", "imec0", "R")
    assert exporter.probe_type == 2013
    assert exporter.is_multishank is True
    assert exporter.sample_rate == 30000.0
    assert exporter.sample2v == pytest.approx(0.62 / 2048 / 80)
    assert exporter.imro_rows[0] == [2013, 384]
    assert exporter.ephys_out_path.is_dir()


def test_build_cluster_template_map_counts_pairs_and_flags_empty():
    """
    Description
    -----------
    The vectorised cluster→template map counts spikes per
    (cluster, template) pair and reports clusters that received no
    spikes. With clusters ``[0,0,0,0,1]`` and templates ``[0,0,0,1,1]``
    over 3 clusters: cluster 0 → {0:3, 1:1}, cluster 1 → {1:1}, and
    cluster 2 is empty.
    """

    exporter = _bare_exporter()
    pair_counts, nan_clusters = exporter._build_cluster_template_map(
        np.array([0, 0, 0, 0, 1]), np.array([0, 0, 0, 1, 1]), n_clusters=3,
    )
    assert pair_counts == {0: {0: 3, 1: 1}, 1: {1: 1}, 2: {}}
    np.testing.assert_array_equal(nan_clusters, [2])


def test_waveform_durations_peak_to_trough_in_ms():
    """
    Description
    -----------
    The duration is ``(argmax - argmin) / sample_rate`` on the peak
    channel, in milliseconds. A trough at sample 10 and peak at sample
    30 at 30 kHz gives ``20 / 30000 * 1000 = 0.667`` ms.
    """

    exporter = _bare_exporter(sample_rate=30000.0)
    w = np.zeros((1, 100, 2))
    w[0, 10, 0] = -5.0     # trough
    w[0, 30, 0] = 5.0      # peak
    durations = exporter._waveform_durations(w, np.array([0]))
    assert durations[0] == pytest.approx(20.0 / 30000.0 * 1e3)


def test_restrict_to_nearest_channels_keeps_n_closest():
    """
    Description
    -----------
    With more channels than :data:`N_CLOSEST_CHANNELS` the helper keeps
    exactly that many nearest the peak (by Manhattan distance) and drops
    the rest. On a 33-channel line with the peak at channel 0, the 32
    nearest are channels 0..31 and the farthest (32) is dropped.
    """

    n_channels = N_CLOSEST_CHANNELS + 1
    waveforms = np.arange(n_channels, dtype=float).reshape(1, 1, n_channels)
    positions = np.column_stack([np.zeros(n_channels), np.arange(n_channels)])
    inds, sparse = _bare_exporter()._restrict_to_nearest_channels(
        waveforms, np.array([0]), positions,
    )
    assert inds.shape == (1, N_CLOSEST_CHANNELS)
    assert sparse.shape == (1, 1, N_CLOSEST_CHANNELS)
    assert set(inds[0].tolist()) == set(range(N_CLOSEST_CHANNELS))
    assert n_channels - 1 not in inds[0]


def test_compute_spike_depths_pc_weighted_channel_y():
    """
    Description
    -----------
    Per-spike depth is the (positive, squared) first-PC-weighted mean
    of the local channel y-positions. For one spike with first-PC
    features ``[3, 4]`` (weights ``[9, 16]``) on channels at y ``[10,
    20]``: depth = ``(10·9 + 20·16) / 25 = 16.4``.
    """

    exporter = _bare_exporter()
    ks = {
        "pc_features": np.array([[[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]]),
        "pc_feature_ind": np.array([[0, 1]]),
        "spike_templates": np.array([0]),
        "channel_positions": np.array([[0.0, 10.0], [0.0, 20.0]]),
    }
    depths = exporter._compute_spike_depths(ks)
    assert depths[0] == pytest.approx(16.4)


def test_write_xyz_picks_converts_tracks_and_skips_existing(tmp_path):
    """
    Description
    -----------
    Each ``{hemisphere}*.npy`` brainreg track is converted to bregma
    mlapdv µm and written as ``xyz_picks_shank{n}.json`` with the
    ``{"xyz_picks": [...]}`` schema. A second call skips the already
    written files and returns an empty list.
    """

    brainreg = tmp_path / "brainreg"
    out = tmp_path / "out"
    brainreg.mkdir()
    out.mkdir()
    np.save(brainreg / "L1.npy", np.array([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]]))
    np.save(brainreg / "L2.npy", np.array([[120.0, 220.0, 320.0]]))
    exporter = _bare_exporter(brainreg_path=brainreg, ephys_out_path=out, hemisphere="L")

    written = exporter.write_xyz_picks()
    assert {p.name for p in written} == {"xyz_picks_shank1.json", "xyz_picks_shank2.json"}
    payload = json.loads((out / "xyz_picks_shank1.json").read_text())
    assert set(payload) == {"xyz_picks"}
    assert len(payload["xyz_picks"]) == 2

    assert exporter.write_xyz_picks() == []   # all already present


def test_remap_channel_ids_to_raw_relabels_per_shank(tmp_path):
    """
    Description
    -----------
    Multi-shank per-shank JSONs keyed by GUI per-shank indices
    (``channel_0..m-1``) are relabelled to raw channel ids drawn from
    the IMRO table, walking channels per shank. ``origin`` is carried
    through.
    """

    out = tmp_path / "out"
    out.mkdir()
    # IMRO header + rows (channel, shank, ...); shanks interleave 0/1.
    imro_rows = [[2013, 384], [0, 0, 1, 0, 100], [1, 1, 1, 0, 101],
                 [2, 0, 1, 0, 102], [3, 1, 1, 0, 103]]
    (out / "channel_locations_shank1.json").write_text(json.dumps(
        {"channel_0": {"v": "a"}, "channel_1": {"v": "b"}, "origin": {"o": 1}}
    ))
    (out / "channel_locations_shank2.json").write_text(json.dumps(
        {"channel_0": {"v": "c"}, "channel_1": {"v": "d"}}
    ))
    exporter = _bare_exporter(
        is_multishank=True, imro_rows=imro_rows, ephys_out_path=out,
    )
    rewritten = exporter.remap_channel_ids_to_raw()
    assert len(rewritten) == 2
    shank1 = json.loads((out / "channel_locations_shank1.json").read_text())
    # shank 0 holds raw channels 0 and 2.
    assert set(shank1) == {"channel_0", "channel_2", "origin"}


def test_remap_channel_ids_to_raw_single_shank_is_noop():
    """
    Description
    -----------
    Single-shank probes have no shank column to remap, so the method is
    a no-op returning an empty list.
    """

    assert _bare_exporter(is_multishank=False).remap_channel_ids_to_raw() == []


def test_write_unified_channel_locations_merges_and_sorts(tmp_path):
    """
    Description
    -----------
    Per-shank JSONs merge into one ``channel_locations.json`` sorted by
    integer channel index, with non-channel keys (``origin``) pushed to
    the end.
    """

    out = tmp_path / "out"
    out.mkdir()
    (out / "channel_locations_shank1.json").write_text(json.dumps(
        {"channel_2": {"v": 2}, "channel_0": {"v": 0}}
    ))
    (out / "channel_locations_shank2.json").write_text(json.dumps(
        {"channel_1": {"v": 1}, "origin": {"o": 9}}
    ))
    exporter = _bare_exporter(ephys_out_path=out)
    unified = exporter.write_unified_channel_locations()
    keys = list(json.loads(unified.read_text()))
    assert keys == ["channel_0", "channel_1", "channel_2", "origin"]


def test_write_unified_channel_locations_raises_when_no_shank_jsons(tmp_path):
    """
    Description
    -----------
    With no per-shank JSONs to merge the method raises
    ``FileNotFoundError``.
    """

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="No channel_locations_shank"):
        _bare_exporter(ephys_out_path=out).write_unified_channel_locations()


def test_load_kilosort_arrays_reads_all_keys(tmp_path):
    """
    Description
    -----------
    The loader reads every Kilosort array into one dict with the
    documented squeeze/cast applied: spike vectors become 1-D, the
    template stack keeps its ``(n_templates, n_samples, n_channels)``
    shape, and positions stay ``(n_channels, 2)``.
    """

    ks = tmp_path / "kilosort4"
    ks.mkdir()
    np.save(ks / "spike_times.npy", np.array([[0], [10], [20]], dtype=np.int64))
    np.save(ks / "spike_clusters.npy", np.array([0, 0, 1], dtype=np.int64))
    np.save(ks / "spike_templates.npy", np.array([0, 0, 0], dtype=np.int64))
    np.save(ks / "amplitudes.npy", np.array([1.0, 2.0, 3.0]))
    np.save(ks / "templates.npy", np.zeros((1, 4, 2)))
    np.save(ks / "whitening_mat_inv.npy", np.eye(2))
    np.save(ks / "channel_map.npy", np.array([0, 1], dtype=np.int64))
    np.save(ks / "channel_positions.npy", np.array([[0.0, 0.0], [0.0, 20.0]]))
    np.save(ks / "pc_features.npy", np.zeros((3, 3, 2)))
    np.save(ks / "pc_feature_ind.npy", np.array([[0, 1]], dtype=np.int64))

    out = _bare_exporter(ks_path=ks)._load_kilosort_arrays()
    assert out["spike_times"].shape == (3,)
    assert out["templates"].shape == (1, 4, 2)
    assert out["channel_positions"].shape == (2, 2)


def test_copy_direct_files_copies_required_and_skips_optional(tmp_path):
    """
    Description
    -----------
    Required Kilosort files are copied to the output directory under
    their ALF destination names; the optional ``_phy_spikes_subset``
    triplet is silently skipped when absent. A missing required source
    raises ``FileNotFoundError``.
    """

    ks = tmp_path / "kilosort4"
    out = tmp_path / "out"
    ks.mkdir()
    out.mkdir()
    for src_name, _dst in _REQUIRED_COPIES:
        (ks / src_name).write_text("x")
    exporter = _bare_exporter(ks_path=ks, ephys_out_path=out)
    exporter._copy_direct_files()
    for _src, dst_name in _REQUIRED_COPIES:
        assert (out / dst_name).is_file()

    (ks / _REQUIRED_COPIES[0][0]).unlink()
    with pytest.raises(FileNotFoundError, match="is missing; cannot complete ALF export"):
        exporter._copy_direct_files()


def test_remap_channel_ids_to_raw_missing_shank_json_raises(tmp_path):
    """
    Description
    -----------
    A multi-shank probe whose IMRO table references a per-shank JSON that
    is absent from the output directory raises ``FileNotFoundError``.
    """

    out = tmp_path / "out"
    out.mkdir()
    imro_rows = [[2013, 384], [0, 0, 1, 0, 100], [1, 1, 1, 0, 101]]
    exporter = _bare_exporter(is_multishank=True, imro_rows=imro_rows, ephys_out_path=out)
    with pytest.raises(FileNotFoundError, match="Missing channel_locations_shank"):
        exporter.remap_channel_ids_to_raw()


def test_write_unified_channel_locations_sorts_malformed_channel_keys_last(tmp_path):
    """
    Description
    -----------
    A ``channel_<non-int>`` key cannot be ordered numerically, so it
    sorts to the end alongside other non-channel keys rather than
    raising.
    """

    out = tmp_path / "out"
    out.mkdir()
    (out / "channel_locations_shank1.json").write_text(json.dumps(
        {"channel_1": {"v": 1}, "channel_bad": {"v": "x"}, "channel_0": {"v": 0}}
    ))
    keys = list(json.loads(_bare_exporter(ephys_out_path=out)
                           .write_unified_channel_locations().read_text()))
    assert keys[:2] == ["channel_0", "channel_1"]
    assert "channel_bad" in keys[2:]


def _write_synthetic_kilosort_dir(ks):
    """
    Description
    -----------
    Populate a Kilosort directory with the 10 arrays
    :meth:`_load_kilosort_arrays` reads plus the 5 required-copy source
    files, sized for 2 templates / 2 clusters / 20 spikes / 4 channels —
    small enough to hand-trace the ALF write pass.

    Returns
    -------
    None
    """

    ks.mkdir(parents=True)
    n_spikes, n_templates, n_samples, n_channels, n_pcs = 20, 2, 60, 4, 3
    rng = np.random.default_rng(0)
    np.save(ks / "spike_times.npy", np.sort(rng.integers(0, 3_000_000, n_spikes)).astype(np.int64))
    # Phy-curated layout (spike_clusters != spike_templates): cluster 0
    # draws from a single template (single-template path), cluster 2 from
    # both (merge → weighted-average path), and cluster 1 receives no
    # spikes (empty-cluster NaN path).
    np.save(ks / "spike_clusters.npy", np.tile([0, 0, 2, 2], n_spikes // 4).astype(np.int64))
    np.save(ks / "spike_templates.npy", np.tile([0, 0, 0, 1], n_spikes // 4).astype(np.int64))
    np.save(ks / "amplitudes.npy", (1.0 + 0.1 * rng.standard_normal(n_spikes)))
    templates = np.zeros((n_templates, n_samples, n_channels))
    for t in range(n_templates):
        templates[t, :, t] = _bump_template()      # each template peaks on its own channel
    np.save(ks / "templates.npy", templates)
    np.save(ks / "whitening_mat_inv.npy", np.eye(n_channels))
    np.save(ks / "channel_map.npy", np.arange(n_channels, dtype=np.int64))
    np.save(ks / "channel_positions.npy",
            np.column_stack([np.zeros(n_channels), np.arange(n_channels) * 20.0]))
    np.save(ks / "pc_features.npy", np.abs(rng.standard_normal((n_spikes, n_pcs, n_channels))))
    np.save(ks / "pc_feature_ind.npy", np.tile(np.arange(n_channels), (n_templates, 1)).astype(np.int64))
    # required-copy sources
    (ks / "params.py").write_text("dat_path = 'x'\n")
    (ks / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n")
    np.save(ks / "whitening_mat.npy", np.eye(n_channels))


def _bump_template(n_samples=60, trough=20, peak=35):
    """Single-channel biphasic template (negative trough, positive rebound)."""
    t = np.arange(n_samples)
    return -1.0 * np.exp(-0.5 * ((t - trough) / 2.5) ** 2) + 0.4 * np.exp(-0.5 * ((t - peak) / 3.0) ** 2)


def test_write_alf_outputs_end_to_end(tmp_path):
    """
    Description
    -----------
    :meth:`write_alf_outputs` loads the Kilosort arrays, copies the
    required files, and writes the full spike / template / cluster ALF
    layout. Driven end-to-end against a synthetic Kilosort directory, it
    must produce every canonical output array with the right leading
    shape and copy the renamed required files.
    """

    ks = tmp_path / "kilosort4"
    _write_synthetic_kilosort_dir(ks)
    out = tmp_path / "ibl_RH"
    out.mkdir()
    exporter = _bare_exporter(ks_path=ks, ephys_out_path=out,
                              sample_rate=30000.0, sample2v=2.34e-6)

    exporter.write_alf_outputs()

    # Spike-level outputs (20 spikes).
    assert np.load(out / "spikes.times.npy").shape == (20,)
    assert np.load(out / "spikes.amps.npy").shape == (20,)
    assert np.load(out / "spikes.depths.npy").shape == (20,)
    # Template-level outputs (2 templates).
    assert np.load(out / "templates.amps.npy").shape == (2,)
    assert np.load(out / "templates.waveforms.npy").shape[0] == 2
    # Cluster-level outputs (clusters 0..2; cluster 1 is empty → NaN).
    assert np.load(out / "clusters.channels.npy").shape == (3,)
    peak_to_trough = np.load(out / "clusters.peakToTrough.npy")
    assert peak_to_trough.shape == (3,)
    assert np.isnan(peak_to_trough[1])
    # Channel-level + renamed required copy.
    assert np.load(out / "channels.rawInd.npy").shape == (4,)
    assert (out / "channels.localCoordinates.npy").is_file()
    assert (out / "params.py").is_file()


# Cross-checks against iblatlas — skipped when the package is missing.


@pytest.fixture(scope="module")
def iblatlas_module():
    """
    Description
    -----------
    Lazy import of ``iblatlas.atlas`` shared across the iblatlas
    cross-check tests in this module. When the package is not
    installed, the entire fixture (and therefore every dependent test)
    is skipped with a clear message; the rest of the suite still runs.
    """
    return pytest.importorskip(
        "iblatlas.atlas",
        reason="iblatlas not installed; install via the 'test' dependency "
               "group to enable Allen CCF cross-checks.",
    )


@pytest.fixture(scope="module")
def mock_allen_atlas(iblatlas_module):
    """
    Description
    -----------
    A mocked :class:`iblatlas.atlas.AllenAtlas` instantiated with
    ``mock=True`` so no NRRD volumes are downloaded. The
    ``BrainCoordinates`` affine — which is the only part of
    ``AllenAtlas`` that :meth:`ccf2xyz` consults — is fully initialised
    in mock mode, so the cross-checks below exercise the same affine
    that runs in production usage of ``iblatlas``.
    """
    return iblatlas_module.AllenAtlas(res_um=25, mock=True)


def test_bregma_constant_matches_iblatlas(iblatlas_module):
    """
    Description
    -----------
    The pinned :data:`ALLEN_BREGMA_MLAPDV_UM` constant in the runtime
    module must exactly equal the bregma landmark in
    ``iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM``. If iblatlas ever
    rebases on a future CCFv4 or recalibrates the landmark, this test
    fails loudly and prompts a manual update of the pinned constant.
    """
    np.testing.assert_array_equal(
        ALLEN_BREGMA_MLAPDV_UM,
        iblatlas_module.ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"].astype(np.float64),
    )


def test_ccf_apdvml_to_xyz_mlapdv_um_matches_iblatlas_on_random_batch(mock_allen_atlas):
    """
    Description
    -----------
    On a 1024-point pseudorandom batch drawn from the interior of the
    Allen CCF 25 µm volume, the inline affine
    :func:`ccf_apdvml_to_xyz_mlapdv_um` must agree with
    ``AllenAtlas(25).ccf2xyz(..., ccf_order='apdvml') * 1e6`` to
    floating-point precision (1e-6 µm absolute, 1e-9 relative). If
    iblatlas changes the affine, the implementation, or the resolution
    convention, this assertion fails before any downstream IBL ALF
    export silently disagrees with the reference pipeline.
    """
    rng = np.random.default_rng(20250912)
    points_apdvml_um = rng.uniform(
        low=[0.0, 0.0, 0.0],
        high=[13200.0, 8000.0, 11400.0],
        size=(1024, 3),
    )

    ours = ccf_apdvml_to_xyz_mlapdv_um(points_apdvml_um)
    theirs = mock_allen_atlas.ccf2xyz(points_apdvml_um, ccf_order="apdvml") * 1e6

    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-9)
