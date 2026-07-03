"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.sglx_meta_to_coords``.

The module is a clean-room SpikeGLX ``.meta`` → probe-geometry
converter. These tests fabricate minimal but valid SpikeGLX metadata
strings (``snsGeomMap`` / ``snsShankMap`` / ``imroTbl`` / ``snsApLfSy``)
entirely in-code, exercise the pure parsers against hand-computed
electrode coordinates, and round-trip every writer through ``tmp_path``.
No real ``.meta`` files, hardware, or network are required.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io as sio

import usv_playpen.neuropixels.sglx_meta_to_coords as sglx
from usv_playpen.neuropixels.sglx_meta_to_coords import (
    ChannelCounts,
    OutputFormat,
    ProbeGeometry,
    PROBE_GEOMETRY,
    MUX_TABLE,
    ShankCoords,
    _imro_derived_meta_fields,
    _lookup_geometry,
    _probe_part_number,
    _sns_geom_map_string,
    apply_bad_channels,
    augment_legacy_meta,
    channel_counts,
    coords_from_meta,
    parse_geom_map,
    parse_shank_map,
    parse_spikeglx_meta,
    plot_coords,
    write_coords_file,
    write_jrclust_coords,
    write_kilosort_chanmap,
    write_npy_coords,
    write_text_coords,
)


def _two_channel_coords():
    """
    Description
    -----------
    Build a synthetic two-channel :class:`ShankCoords` spanning two
    shanks with a 250 µm shank pitch — small enough to hand-verify the
    ``x_global_um`` offset arithmetic that every writer relies on.

    Returns
    -------
    ShankCoords
        Channel 0 on shank 0 at local ``(10, 0)``; channel 1 on shank
        1 at local ``(20, 20)``.
    """

    return ShankCoords(
        n_shanks=2,
        shank_width_um=70.0,
        shank_pitch_um=250.0,
        shank_index=np.array([0, 1], dtype=np.int64),
        x_um=np.array([10.0, 20.0], dtype=np.float64),
        y_um=np.array([0.0, 20.0], dtype=np.float64),
        connected=np.array([True, True], dtype=bool),
    )


def test_probe_geometry_columns_per_shank_is_derived():
    """
    Description
    -----------
    ``columns_per_shank`` is ``electrodes_per_shank // rows_per_shank``;
    for the staggered NP1.0 family (960 electrodes, 480 rows) that is
    exactly 2 columns.
    """

    geom = PROBE_GEOMETRY["NP1010"]
    assert geom.columns_per_shank == 2
    assert ProbeGeometry(1, 70.0, 0.0, 0.0, 0.0, 6.0, 6.0, 48, 384).columns_per_shank == 8


def test_geometry_and_mux_tables_share_part_number_key_set():
    """
    Description
    -----------
    Every part number we accept geometrically must also have a MUX
    table available for legacy-meta augmentation, and vice versa — the
    two module-level dicts must have identical key sets.
    """

    assert set(PROBE_GEOMETRY) == set(MUX_TABLE)
    assert PROBE_GEOMETRY["3A"] is PROBE_GEOMETRY["NP1010"]


def test_output_format_values():
    """
    Description
    -----------
    The ``OutputFormat`` enum values double as the Qt chooser labels;
    pin them so a rename can't silently break the GUI dispatch.
    """

    assert OutputFormat.TEXT.value == "text"
    assert OutputFormat.KILOSORT_MAT.value == "kilosort_mat"
    assert OutputFormat.JRCLUST_STRINGS.value == "jrclust_strings"
    assert OutputFormat.NPY.value == "npy"
    assert OutputFormat.LEGACY_META_AUGMENT.value == "legacy_meta_augment"


def test_shank_coords_n_channels_and_x_global():
    """
    Description
    -----------
    ``n_channels`` reflects the array length and ``x_global_um`` applies
    the per-shank offset ``shank_index * shank_pitch_um + x_um``: the
    two-channel fixture yields global x of ``[10, 270]``.
    """

    coords = _two_channel_coords()
    assert coords.n_channels == 2
    np.testing.assert_array_equal(coords.x_global_um(), [10.0, 270.0])


def test_parse_spikeglx_meta_strips_tilde_and_skips_keyless_lines(tmp_path):
    """
    Description
    -----------
    The parser strips the leading ``~`` on mutable keys, keeps values
    as raw strings, and silently drops any line without an ``=``.
    """

    meta_file = tmp_path / "run.imec0.ap.meta"
    meta_file.write_text(
        "snsApLfSy=384,384,1\n"
        "imDatPrb_pn=NP1010\n"
        "~snsShankMap=(1,2,480)(0:0:0:1)\n"
        "a_line_without_equals\n"
    )
    meta = parse_spikeglx_meta(meta_file)
    assert meta["snsApLfSy"] == "384,384,1"
    assert meta["imDatPrb_pn"] == "NP1010"
    assert meta["snsShankMap"] == "(1,2,480)(0:0:0:1)"
    assert "a_line_without_equals" not in meta


def test_parse_spikeglx_meta_missing_file_raises(tmp_path):
    """
    Description
    -----------
    A non-existent meta path raises ``FileNotFoundError`` rather than
    returning an empty dict.
    """

    with pytest.raises(FileNotFoundError):
        parse_spikeglx_meta(tmp_path / "absent.meta")


def test_channel_counts_parses_ap_lf_sy_triple():
    """
    Description
    -----------
    ``channel_counts`` splits the ``snsApLfSy`` comma-triple into a
    structured ``ChannelCounts(ap, lf, sy)``.
    """

    counts = channel_counts({"snsApLfSy": "384,384,1"})
    assert counts == ChannelCounts(ap=384, lf=384, sy=1)


def test_probe_part_number_defaults_to_3a():
    """
    Description
    -----------
    A meta dict without ``imDatPrb_pn`` identifies a pre-part-number
    3A-phase probe, for which the resolver returns the literal ``'3A'``.
    """

    assert _probe_part_number({"imDatPrb_pn": "NP2010"}) == "NP2010"
    assert _probe_part_number({}) == "3A"


def test_lookup_geometry_resolves_and_rejects_unknown_part_number():
    """
    Description
    -----------
    A known part number resolves to its ``ProbeGeometry``; an unknown
    one raises ``ValueError`` naming the offending part number.
    """

    assert _lookup_geometry({"imDatPrb_pn": "NP2010"}) is PROBE_GEOMETRY["NP2010"]
    with pytest.raises(ValueError, match="unsupported Imec probe part number"):
        _lookup_geometry({"imDatPrb_pn": "NOT_A_PROBE"})


def test_parse_geom_map_decodes_channels_and_connected_flag():
    """
    Description
    -----------
    ``snsGeomMap`` carries ``(x, y, connected)`` per channel directly;
    the parser reads the header (n_shanks / pitch / width) and one
    group per channel, mapping ``connected == 1`` to ``True``.
    """

    meta = {
        "snsGeomMap": "(NP1010,1,0.0,70.0)(0:27.0:0.0:1)(0:11.0:20.0:1)(0:27.0:40.0:0)"
    }
    coords = parse_geom_map(meta)
    assert coords.n_shanks == 1
    assert coords.shank_pitch_um == 0.0
    assert coords.shank_width_um == 70.0
    assert coords.n_channels == 3
    np.testing.assert_array_equal(coords.shank_index, [0, 0, 0])
    np.testing.assert_array_equal(coords.x_um, [27.0, 11.0, 27.0])
    np.testing.assert_array_equal(coords.y_um, [0.0, 20.0, 40.0])
    np.testing.assert_array_equal(coords.connected, [True, True, False])


def test_parse_geom_map_multi_shank_pitch():
    """
    Description
    -----------
    For a four-shank NP2.0 ``snsGeomMap`` the header shank pitch is
    parsed and ``x_global_um`` lifts a shank-3 channel by
    ``3 * 250 = 750`` µm.
    """

    meta = {
        "snsGeomMap": "(NP2010,4,250.0,70.0)(0:0.0:0.0:1)(3:32.0:15.0:1)"
    }
    coords = parse_geom_map(meta)
    assert coords.n_shanks == 4
    assert coords.shank_pitch_um == 250.0
    np.testing.assert_array_equal(coords.x_global_um(), [0.0, 750.0 + 32.0])


def test_parse_shank_map_converts_col_row_to_microns():
    """
    Description
    -----------
    Legacy ``snsShankMap`` carries ``(col, row)`` indices that the
    parser converts to micrometres with the staggered NP1.0 geometry
    (even-row offset 27, odd-row offset 11, h-pitch 32, v-pitch 20):

    * ch0 col0 row0(even) → x = 0*32 + 27 = 27, y = 0
    * ch1 col1 row0(even) → x = 1*32 + 27 = 59, y = 0
    * ch2 col0 row1(odd)  → x = 0*32 + 11 = 11, y = 20

    Only the first ``ap`` entries (here 3) are consumed.
    """

    meta = {
        "snsApLfSy": "3,0,1",
        "imDatPrb_pn": "NP1010",
        "snsShankMap": "(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)",
    }
    coords = parse_shank_map(meta)
    assert coords.n_channels == 3
    np.testing.assert_array_equal(coords.x_um, [27.0, 59.0, 11.0])
    np.testing.assert_array_equal(coords.y_um, [0.0, 0.0, 20.0])
    np.testing.assert_array_equal(coords.connected, [True, True, True])


def test_coords_from_meta_prefers_geom_map_then_shank_map_then_raises():
    """
    Description
    -----------
    The dispatcher prefers the new-style ``snsGeomMap`` over
    ``snsShankMap`` when both are present, falls back to the legacy key
    otherwise, and raises ``KeyError`` when neither exists.
    """

    geom_meta = {
        "snsGeomMap": "(NP1010,1,0.0,70.0)(0:27.0:0.0:1)",
        "snsApLfSy": "1,0,1",
        "imDatPrb_pn": "NP1010",
        "snsShankMap": "(1,2,480)(0:1:0:1)",
    }
    # Geom map wins: x should be 27 (geom), not 59 (col-1 shank map).
    np.testing.assert_array_equal(coords_from_meta(geom_meta).x_um, [27.0])

    legacy_meta = {
        "snsApLfSy": "1,0,1",
        "imDatPrb_pn": "NP1010",
        "snsShankMap": "(1,2,480)(0:1:0:1)",
    }
    np.testing.assert_array_equal(coords_from_meta(legacy_meta).x_um, [59.0])

    with pytest.raises(KeyError):
        coords_from_meta({"snsApLfSy": "1,0,1"})


def test_apply_bad_channels_clips_out_of_range_and_does_not_mutate():
    """
    Description
    -----------
    Marking bad channels clears only the in-range indices' ``connected``
    flag, silently clips out-of-bounds indices, and returns a new
    instance without mutating the source.
    """

    coords = ShankCoords(
        n_shanks=1,
        shank_width_um=70.0,
        shank_pitch_um=0.0,
        shank_index=np.zeros(4, dtype=np.int64),
        x_um=np.arange(4, dtype=np.float64),
        y_um=np.arange(4, dtype=np.float64),
        connected=np.ones(4, dtype=bool),
    )
    out = apply_bad_channels(coords, np.array([1, 99]))
    np.testing.assert_array_equal(out.connected, [True, False, True, True])
    # Source untouched.
    np.testing.assert_array_equal(coords.connected, [True, True, True, True])


def test_write_text_coords_round_trip(tmp_path):
    """
    Description
    -----------
    The text writer emits one tab-delimited
    ``index x_global y shank`` row per channel (no header), using the
    shank-offset-applied global x.
    """

    coords = _two_channel_coords()
    dst = tmp_path / "coords.txt"
    write_text_coords(coords, dst)
    lines = dst.read_text().splitlines()
    assert lines == ["0\t10\t0\t0", "1\t270\t20\t1"]


def test_write_npy_coords_round_trip(tmp_path):
    """
    Description
    -----------
    The NPY writer stores an ``(n_channels, 2)`` array of
    ``(x_global, y)`` readable back with ``np.load``.
    """

    coords = _two_channel_coords()
    dst = tmp_path / "coords.npy"
    write_npy_coords(coords, dst)
    arr = np.load(dst)
    assert arr.shape == (2, 2)
    np.testing.assert_array_equal(arr, [[10.0, 0.0], [270.0, 20.0]])


def test_write_jrclust_coords_emits_one_based_matlab_strings(tmp_path):
    """
    Description
    -----------
    JRClust is MATLAB-native, so ``shankMap`` and ``siteMap`` are
    1-based while ``siteLoc`` carries the absolute ``(x, y)`` pairs.
    """

    coords = _two_channel_coords()
    dst = tmp_path / "jrc.txt"
    write_jrclust_coords(coords, dst)
    lines = dst.read_text().splitlines()
    assert lines[0] == "shankMap = [1,2];"
    assert lines[1] == "siteLoc = [10,0;270,20];"
    assert lines[2] == "siteMap = [1,2];"


def test_write_kilosort_chanmap_round_trip(tmp_path):
    """
    Description
    -----------
    The Kilosort ``.mat`` carries 1-based and 0-based channel maps,
    absolute x/y, 1-based shank indices (``kcoords``), the connected
    mask, and the ``name`` scalar — all readable back with
    ``scipy.io.loadmat``.
    """

    coords = _two_channel_coords()
    coords.connected[0] = False
    dst = tmp_path / "chanMap.mat"
    write_kilosort_chanmap(coords, dst, name="run.imec0.ap")
    mat = sio.loadmat(str(dst))
    np.testing.assert_array_equal(mat["chanMap"].ravel(), [1.0, 2.0])
    np.testing.assert_array_equal(mat["chanMap0ind"].ravel(), [0.0, 1.0])
    np.testing.assert_array_equal(mat["xcoords"].ravel(), [10.0, 270.0])
    np.testing.assert_array_equal(mat["ycoords"].ravel(), [0.0, 20.0])
    np.testing.assert_array_equal(mat["kcoords"].ravel(), [1.0, 2.0])
    np.testing.assert_array_equal(mat["connected"].ravel().astype(bool), [False, True])
    assert mat["name"][0] == "run.imec0.ap"


def test_imro_derived_fields_np2_fixed_gain():
    """
    Description
    -----------
    NP2.0 / NP2.0-4S (probe types 21 / 24) have a fixed gain of 80 and
    expose only AP-band data, so ``imAnyChanFullBand`` is always true.
    """

    ap, lf, full = _imro_derived_meta_fields({"imroTbl": "(21,384)"})
    assert ap == "imChan0apGain=80\n"
    assert lf == "imChan0lfGain=80\n"
    assert full == "imAnyChanFullBand=true\n"


def test_imro_derived_fields_np1110_reads_header_gains():
    """
    Description
    -----------
    NP1110 (UHD-2, type 1110) carries gain/filter info in the imro
    header: fields 3/4 are AP/LF gain, field 5 == "0" means full band.
    """

    ap, lf, full = _imro_derived_meta_fields({"imroTbl": "(1110,0,1,500,250,0)"})
    assert ap == "imChan0apGain=500\n"
    assert lf == "imChan0lfGain=250\n"
    assert full == "imAnyChanFullBand=true\n"


def test_imro_derived_fields_np1_per_channel_gain_and_fullband_scan():
    """
    Description
    -----------
    NP1.0 (type 0) reads gain from the first per-channel entry and
    scans every entry for a filter bypass (field 5 == "0"): all-"1"
    entries give ``imAnyChanFullBand=false``, while any "0" flips it
    true.
    """

    no_full = _imro_derived_meta_fields(
        {"imroTbl": "(0,384)(0 0 0 250 450 1)(1 0 0 250 450 1)"}
    )
    assert no_full[0] == "imChan0apGain=250\n"
    assert no_full[1] == "imChan0lfGain=450\n"
    assert no_full[2] == "imAnyChanFullBand=false\n"

    with_full = _imro_derived_meta_fields(
        {"imroTbl": "(0,384)(0 0 0 250 450 1)(1 0 0 250 450 0)"}
    )
    assert with_full[2] == "imAnyChanFullBand=true\n"


def test_imro_derived_fields_3a_phase_treated_as_type_zero():
    """
    Description
    -----------
    3A-phase probes encode a >50000 probe type in the imro header but
    behave like NP1.0 type 0 for gain/filter purposes.
    """

    ap, _lf, _full = _imro_derived_meta_fields(
        {"imroTbl": "(1030101,384)(0 0 0 250 450 1)"}
    )
    assert ap == "imChan0apGain=250\n"


def test_sns_geom_map_string_round_trips_through_parser():
    """
    Description
    -----------
    The synthesised ``snsGeomMap`` line carries the part number in its
    header and one group per channel; stripping the leading ``~`` and
    feeding it back through :func:`parse_geom_map` recovers the original
    coordinates.
    """

    coords = parse_geom_map(
        {"snsGeomMap": "(NP1010,1,0.0,70.0)(0:27.0:0.0:1)(0:11.0:20.0:0)"}
    )
    line = _sns_geom_map_string({"imDatPrb_pn": "NP1010"}, coords)
    assert line.startswith("~snsGeomMap=(NP1010,1,0,70)")
    reparsed = parse_geom_map({"snsGeomMap": line.split("=", 1)[1].strip()})
    np.testing.assert_array_equal(reparsed.x_um, coords.x_um)
    np.testing.assert_array_equal(reparsed.connected, coords.connected)


def test_augment_legacy_meta_appends_fields_and_backs_up(tmp_path):
    """
    Description
    -----------
    Augmenting a pre-032623 meta appends the gain/filter, mux, and
    geom fields, preserves the original as ``<stem>_orig.meta``, and is
    idempotent (a second call with the gain field already present is a
    no-op that adds no further lines).
    """

    meta_file = tmp_path / "run.imec0.ap.meta"
    meta_file.write_text(
        "imDatPrb_pn=NP1010\n"
        "snsApLfSy=2,0,1\n"
        "imroTbl=(0,384)(0 0 0 250 450 1)(1 0 0 250 450 1)\n"
    )
    coords = _two_channel_coords()

    returned = augment_legacy_meta(meta_file, coords)
    assert returned == meta_file
    augmented = parse_spikeglx_meta(meta_file)
    assert augmented["imChan0apGain"] == "250"
    assert augmented["imAnyChanFullBand"] == "false"
    assert "snsGeomMap" in augmented
    assert "muxTbl" in augmented
    assert (tmp_path / "run.imec0.ap_orig.meta").exists()

    line_count = len(meta_file.read_text().splitlines())
    augment_legacy_meta(meta_file, coords)
    assert len(meta_file.read_text().splitlines()) == line_count


def test_augment_legacy_meta_unknown_part_number_raises(tmp_path):
    """
    Description
    -----------
    A part number absent from ``MUX_TABLE`` cannot have its ``muxTbl``
    synthesised, so augmentation raises ``ValueError``.
    """

    meta_file = tmp_path / "weird.ap.meta"
    meta_file.write_text("imDatPrb_pn=NOT_A_PROBE\nsnsApLfSy=1,0,1\n")
    with pytest.raises(ValueError, match="unsupported probe part number"):
        augment_legacy_meta(meta_file, _two_channel_coords())


@pytest.mark.parametrize(
    "output_format,suffix",
    [
        (OutputFormat.TEXT, "_siteCoords.txt"),
        (OutputFormat.NPY, "_siteCoords.npy"),
        (OutputFormat.JRCLUST_STRINGS, "_forJRCprm.txt"),
        (OutputFormat.KILOSORT_MAT, "_kilosortChanMap.mat"),
    ],
)
def test_write_coords_file_dispatches_by_format(tmp_path, output_format, suffix):
    """
    Description
    -----------
    ``write_coords_file`` builds the destination from ``save_dir`` +
    ``base_name`` + the per-format suffix and returns the path it
    actually wrote.
    """

    coords = _two_channel_coords()
    written = write_coords_file(
        meta={"imDatPrb_pn": "NP1010"},
        coords=coords,
        output_format=output_format,
        save_dir=tmp_path,
        base_name="run.imec0.ap",
    )
    assert written == tmp_path / f"run.imec0.ap{suffix}"
    assert written.exists()


def test_write_coords_file_legacy_meta_augment_in_place(tmp_path):
    """
    Description
    -----------
    For ``LEGACY_META_AUGMENT`` the destination is the meta file itself
    (``save_dir / base_name.meta``); ``write_coords_file`` augments it
    in place and returns that same path.
    """

    base_name = "run.imec0.ap"
    meta_file = tmp_path / f"{base_name}.meta"
    meta_file.write_text(
        "imDatPrb_pn=NP1010\n"
        "snsApLfSy=2,0,1\n"
        "imroTbl=(0,384)(0 0 0 250 450 1)(1 0 0 250 450 1)\n"
    )
    written = write_coords_file(
        meta={"imDatPrb_pn": "NP1010"},
        coords=_two_channel_coords(),
        output_format=OutputFormat.LEGACY_META_AUGMENT,
        save_dir=tmp_path,
        base_name=base_name,
    )
    assert written == meta_file
    assert "snsGeomMap" in parse_spikeglx_meta(meta_file)


def test_plot_coords_returns_figure_with_requested_size():
    """
    Description
    -----------
    ``plot_coords`` returns a matplotlib ``Figure`` of the requested
    size carrying the background-electrode and saved-channel scatter
    collections (one pair per shank).
    """

    coords = parse_geom_map(
        {"snsGeomMap": "(NP1010,1,0.0,70.0)(0:27.0:0.0:1)(0:11.0:20.0:1)"}
    )
    fig = plot_coords(coords, {"imDatPrb_pn": "NP1010"}, figsize=(3.0, 9.0))
    try:
        assert tuple(fig.get_size_inches()) == (3.0, 9.0)
        # One background + one saved-channel collection for the single shank.
        assert len(fig.axes[0].collections) == 2
    finally:
        plt.close(fig)


def _write_geom_meta(tmp_path, name="run.imec0.ap.meta"):
    """Write a minimal SpikeGLX meta with a snsGeomMap + part number so
    the GUI's parse → coords → write chain succeeds end to end."""
    meta_file = tmp_path / name
    meta_file.write_text(
        "imDatPrb_pn=NP1010\n"
        "snsGeomMap=(NP1010,1,0.0,70.0)(0:27.0:0.0:1)(0:11.0:20.0:1)\n"
    )
    return meta_file


class _FakeFileDialog:
    """Qt ``QFileDialog`` stand-in returning a preset path."""
    _result = ("", "")

    @classmethod
    def getOpenFileName(cls, *args, **kwargs):
        return cls._result


class _FakeInputDialog:
    """Qt ``QInputDialog`` stand-in returning a preset (label, ok)."""
    _result = ("text", True)

    @classmethod
    def getItem(cls, *args, **kwargs):
        return cls._result


class _FakeStandardButton:
    Yes = 1
    No = 2


class _FakeMessageBox:
    """Qt ``QMessageBox`` stand-in; ``question`` returns a preset button."""
    StandardButton = _FakeStandardButton
    _answer = _FakeStandardButton.No

    @classmethod
    def question(cls, *args, **kwargs):
        return cls._answer

    @classmethod
    def critical(cls, *args, **kwargs):
        return None


def _patch_qt(monkeypatch, file_result, item_result=("text", True), answer=_FakeStandardButton.No):
    """Patch the module's Qt dialog classes and the app bootstrap so
    ``main`` runs headless with scripted dialog responses."""
    _FakeFileDialog._result = file_result
    _FakeInputDialog._result = item_result
    _FakeMessageBox._answer = answer
    monkeypatch.setattr(sglx, "_ensure_qt_app", lambda: None)
    monkeypatch.setattr(sglx, "QFileDialog", _FakeFileDialog)
    monkeypatch.setattr(sglx, "QInputDialog", _FakeInputDialog)
    monkeypatch.setattr(sglx, "QMessageBox", _FakeMessageBox)


def test_main_writes_selected_format_and_returns_zero(tmp_path, monkeypatch):
    """
    Description
    -----------
    Driving the GUI with a chosen meta file and the ``text`` format must
    write the coordinate file beside the meta and return exit code 0.
    """

    meta_file = _write_geom_meta(tmp_path)
    _patch_qt(monkeypatch, file_result=(str(meta_file), ""), item_result=("text", True))
    assert sglx.main([]) == 0
    assert (tmp_path / "run.imec0.ap_siteCoords.txt").is_file()


def test_main_returns_zero_when_file_dialog_cancelled(tmp_path, monkeypatch):
    """
    Description
    -----------
    Cancelling the file picker (empty path) is a clean exit: ``main``
    returns 0 and writes nothing.
    """

    _patch_qt(monkeypatch, file_result=("", ""))
    assert sglx.main([]) == 0


def test_main_returns_zero_when_format_dialog_cancelled(tmp_path, monkeypatch):
    """
    Description
    -----------
    Cancelling the format chooser (``ok=False``) is a clean exit
    returning 0.
    """

    meta_file = _write_geom_meta(tmp_path)
    _patch_qt(monkeypatch, file_result=(str(meta_file), ""), item_result=("text", False))
    assert sglx.main([]) == 0


def test_main_returns_one_on_conversion_error(tmp_path, monkeypatch):
    """
    Description
    -----------
    A meta with neither ``snsGeomMap`` nor ``snsShankMap`` makes
    ``coords_from_meta`` raise ``KeyError``; ``main`` surfaces it via a
    critical dialog and returns exit code 1.
    """

    bad_meta = tmp_path / "bad.imec0.ap.meta"
    bad_meta.write_text("imDatPrb_pn=NP1010\n")
    _patch_qt(monkeypatch, file_result=(str(bad_meta), ""), item_result=("text", True))
    assert sglx.main([]) == 1


def test_main_shows_plot_when_requested(tmp_path, monkeypatch):
    """
    Description
    -----------
    Answering "yes" to the post-save prompt renders the probe layout;
    with ``plt.show`` stubbed out the plot branch runs and ``main`` still
    returns 0.
    """

    meta_file = _write_geom_meta(tmp_path)
    _patch_qt(
        monkeypatch, file_result=(str(meta_file), ""),
        item_result=("text", True), answer=_FakeStandardButton.Yes,
    )
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    assert sglx.main([]) == 0


def test_main_legacy_meta_augment_in_place(tmp_path, monkeypatch):
    """
    Description
    -----------
    Selecting the ``legacy_meta_augment`` format drives ``main`` down the
    in-place augmentation branch: the chosen legacy meta gains a
    ``snsGeomMap`` and ``main`` returns 0.
    """

    meta_file = tmp_path / "legacy.imec0.ap.meta"
    meta_file.write_text(
        "imDatPrb_pn=NP1010\n"
        "snsApLfSy=2,0,1\n"
        "imroTbl=(0,384)(0 0 0 250 450 1)(1 0 0 250 450 1)\n"
        "snsShankMap=(1,2,480)(0:0:0:1)(0:1:0:1)\n"
    )
    _patch_qt(
        monkeypatch, file_result=(str(meta_file), ""),
        item_result=("legacy_meta_augment", True),
    )
    assert sglx.main([]) == 0
    assert "snsGeomMap" in parse_spikeglx_meta(meta_file)
def test_main_headless_writes_selected_format_and_returns_zero(tmp_path):
    """
    Description
    -----------
    In headless mode (``--meta-file`` given) the conversion runs without
    Qt: passing the meta path and ``--output-format text`` writes the
    coordinate file beside the meta and returns exit code 0.
    """

    meta_file = _write_geom_meta(tmp_path)
    exit_code = sglx.main(
        ["--meta-file", str(meta_file), "--output-format", "text"]
    )
    assert exit_code == 0
    assert (tmp_path / "run.imec0.ap_siteCoords.txt").is_file()
