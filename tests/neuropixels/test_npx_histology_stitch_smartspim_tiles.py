"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.histology_stitch_smartspim_tiles``.

Covers the pure stage-coordinate layout and feather-weight helpers
against hand-computed values, the metadata / tile-directory parsing, and
an end-to-end two-tile stitch of tiny synthetic uint16 planes in
``tmp_path``. With ``feather_pixels=1`` every weight saturates to 1.0,
so overlap pixels become the plain mean of the contributing tiles and
single-coverage pixels equal their source exactly — making the blend
hand-verifiable.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from usv_playpen.neuropixels.histology_stitch_smartspim_tiles import (
    _compute_tile_layout,
    _enumerate_tile_dirs,
    _list_plane_files,
    _make_bevel_weights,
    _parse_smartspim_metadata,
    _resolve_channel_dir,
    stitch_smartspim_tiles,
)


def test_compute_tile_layout_offsets_and_canvas():
    """
    Description
    -----------
    Stage coordinates (0.1 µm units) convert to integer pixel offsets
    relative to the min-X/min-Y tile, and the canvas is the bounding
    box. With pixel size 1 µm a 100-unit stage step is 10 px:
    (100,200)→(10,0), (200,100)→(0,10), origin (100,100)→(0,0); canvas
    20x20.
    """

    layout, canvas = _compute_tile_layout(
        [(100, 100), (100, 200), (200, 100)],
        pixel_size_um=1.0,
        tile_shape_yx=(10, 10),
    )
    assert layout[(100, 100)] == (0, 0)
    assert layout[(100, 200)] == (10, 0)
    assert layout[(200, 100)] == (0, 10)
    assert canvas == (20, 20)


def test_compute_tile_layout_single_tile_is_tile_sized():
    """
    Description
    -----------
    A single tile places at the origin and the canvas equals the tile
    shape.
    """

    layout, canvas = _compute_tile_layout([(50, 50)], 1.0, (8, 8))
    assert layout == {(50, 50): (0, 0)}
    assert canvas == (8, 8)


def test_compute_tile_layout_empty_raises():
    """
    Description
    -----------
    An empty tile list cannot define a canvas and raises ``ValueError``.
    """

    with pytest.raises(ValueError, match="tile_positions is empty"):
        _compute_tile_layout([], 1.0, (4, 4))


def test_make_bevel_weights_plateau_and_floor():
    """
    Description
    -----------
    The bevel weight map is float32 of tile shape, ramps from the edge
    to a central plateau of 1.0, and a very large ``feather_pixels``
    drives the edge pixels down to the ``1e-3`` floor.
    """

    w = _make_bevel_weights((10, 10), feather_pixels=3)
    assert w.dtype == np.float32
    assert w.shape == (10, 10)
    assert w[5, 5] == pytest.approx(1.0)
    assert w[0, 0] == pytest.approx(1.0 / 3.0)
    assert w[5, 5] >= w[0, 0]

    floored = _make_bevel_weights((4, 4), feather_pixels=100_000)
    assert floored[0, 0] == pytest.approx(1e-3)


def test_make_bevel_weights_rejects_feather_below_one():
    """
    Description
    -----------
    A feather width below 1 pixel is degenerate and raises
    ``ValueError``.
    """

    with pytest.raises(ValueError, match="feather_pixels must be >= 1"):
        _make_bevel_weights((4, 4), feather_pixels=0)


def test_parse_smartspim_metadata_reads_pixel_and_z_step(tmp_path):
    """
    Description
    -----------
    The pixel size and Z step are read from columns 5 and 6 of the data
    row following the ``Obj`` header line.
    """

    meta_file = tmp_path / "metadata.txt"
    meta_file.write_text(
        "Obj\tcol1\tcol2\tcol3\tcol4\tcol5\n"
        "val0\tval1\tval2\tval3\t0.76\t1.5\n"
    )
    info = _parse_smartspim_metadata(meta_file)
    assert info == {"pixel_size_um": 0.76, "z_step_um": 1.5}


def test_parse_smartspim_metadata_without_obj_header_raises(tmp_path):
    """
    Description
    -----------
    A metadata file with no ``Obj`` header row cannot be parsed and
    raises ``ValueError``.
    """

    meta_file = tmp_path / "metadata.txt"
    meta_file.write_text("Header\tnope\n1\t2\n")
    with pytest.raises(ValueError, match="Could not parse pixel size"):
        _parse_smartspim_metadata(meta_file)


def test_enumerate_tile_dirs_maps_stage_coords_and_skips_malformed(tmp_path):
    """
    Description
    -----------
    Tile directories ``{X}/{X}_{Y}/`` map to ``(x, y)`` stage coords;
    a non-integer outer dir and an ``{X}_{Y}`` whose X disagrees with
    the parent are both skipped.
    """

    chan = tmp_path / "Ex_488_Ch0"
    (chan / "100" / "100_200").mkdir(parents=True)
    (chan / "100" / "100_300").mkdir(parents=True)
    (chan / "100" / "999_400").mkdir(parents=True)   # X mismatch -> skip
    (chan / "100" / "100").mkdir(parents=True)        # single token -> skip
    (chan / "100" / "100_zz").mkdir(parents=True)     # non-int y -> skip
    (chan / "abc" / "abc_1").mkdir(parents=True)      # non-int outer -> skip

    tiles = _enumerate_tile_dirs(chan)
    assert set(tiles) == {(100, 200), (100, 300)}


def test_list_plane_files_sorted(tmp_path):
    """
    Description
    -----------
    Plane files are returned in ascending lexicographic (zero-padded)
    order.
    """

    tile = tmp_path / "tile"
    tile.mkdir()
    for name in ("000200.tiff", "000000.tiff", "000100.tiff"):
        tifffile.imwrite(tile / name, np.zeros((2, 2), dtype=np.uint16))
    assert [p.name for p in _list_plane_files(tile)] == [
        "000000.tiff", "000100.tiff", "000200.tiff",
    ]


def test_resolve_channel_dir_maps_wavelength_and_validates(tmp_path):
    """
    Description
    -----------
    488 / 561 map to ``Ex_488_Ch0`` / ``Ex_561_Ch1``; an unknown
    wavelength raises ``ValueError`` and a missing directory raises
    ``FileNotFoundError``.
    """

    (tmp_path / "Ex_488_Ch0").mkdir()
    assert _resolve_channel_dir(tmp_path, 488) == tmp_path / "Ex_488_Ch0"

    with pytest.raises(ValueError, match="wavelength_nm must be one of"):
        _resolve_channel_dir(tmp_path, 405)
    with pytest.raises(FileNotFoundError, match="channel directory not found"):
        _resolve_channel_dir(tmp_path, 561)


def _make_smartspim_tree(tmp_path):
    """
    Description
    -----------
    Build a minimal SmartSPIM acquisition: ``metadata.txt`` (1 µm
    pixel) and a single 488 channel with two horizontally adjacent
    4x4 tiles two planes deep. Tile A (stage (0,0)) is filled with one
    set of values, tile B (stage (20,0) → 2 px offset) with another, so
    the tiles overlap in 2 columns.

    Returns
    -------
    pathlib.Path
        The acquisition root.
    """

    root = tmp_path / "acq"
    (root).mkdir()
    (root / "metadata.txt").write_text(
        "Obj\ta\tb\tc\td\te\n"
        "v0\tv1\tv2\tv3\t1.0\t2.0\n"
    )
    tile_a = root / "Ex_488_Ch0" / "0" / "0_0"
    tile_b = root / "Ex_488_Ch0" / "20" / "20_0"
    tile_a.mkdir(parents=True)
    tile_b.mkdir(parents=True)
    # Plane 0 then plane 1; tile A vs tile B fill values.
    for name, val in (("000000.tiff", 100), ("000100.tiff", 10)):
        tifffile.imwrite(tile_a / name, np.full((4, 4), val, dtype=np.uint16))
    for name, val in (("000000.tiff", 200), ("000100.tiff", 20)):
        tifffile.imwrite(tile_b / name, np.full((4, 4), val, dtype=np.uint16))
    return root


def test_stitch_smartspim_tiles_blends_overlap_mean(tmp_path):
    """
    Description
    -----------
    Two 4x4 tiles offset by 2 px stitch into a 4x6 canvas per plane.
    With ``feather_pixels=1`` all weights are 1.0, so single-coverage
    columns equal their source tile and the 2 overlap columns are the
    plain mean of the two tiles: plane 0 → [100,100,150,150,200,200],
    plane 1 → [10,10,15,15,20,20].
    """

    root = _make_smartspim_tree(tmp_path)
    out = tmp_path / "stitched.tif"
    written = stitch_smartspim_tiles(root, out, wavelength_nm=488, feather_pixels=1)
    assert written == [out]

    volume = tifffile.imread(out)
    assert volume.shape == (2, 4, 6)
    np.testing.assert_array_equal(volume[0, 0], [100, 100, 150, 150, 200, 200])
    np.testing.assert_array_equal(volume[1, 0], [10, 10, 15, 15, 20, 20])


def test_stitch_smartspim_tiles_z_flip_reverses_planes(tmp_path):
    """
    Description
    -----------
    ``z_flip=True`` reverses the written plane order, so the volume's
    plane 0 carries what plane 1 held without the flip.
    """

    root = _make_smartspim_tree(tmp_path)
    out = tmp_path / "stitched_flip.tif"
    stitch_smartspim_tiles(root, out, wavelength_nm=488, feather_pixels=1, z_flip=True)
    volume = tifffile.imread(out)
    np.testing.assert_array_equal(volume[0, 0], [10, 10, 15, 15, 20, 20])


def test_stitch_smartspim_tiles_missing_metadata_raises(tmp_path):
    """
    Description
    -----------
    A SmartSPIM root without ``metadata.txt`` raises
    ``FileNotFoundError``.
    """

    root = tmp_path / "acq"
    (root / "Ex_488_Ch0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="metadata.txt not found"):
        stitch_smartspim_tiles(root, tmp_path / "o.tif", wavelength_nm=488)


def test_stitch_smartspim_tiles_validates_wavelength_arguments(tmp_path):
    """
    Description
    -----------
    Wavelength arguments are validated before any I/O: an empty iterable,
    an unknown wavelength, and a multi-wavelength request without a
    ``{wavelength_nm}`` placeholder each raise ``ValueError``.
    """

    with pytest.raises(ValueError, match="non-empty"):
        stitch_smartspim_tiles(tmp_path, tmp_path / "o.tif", wavelength_nm=())
    with pytest.raises(ValueError, match="wavelength_nm values must each be one of"):
        stitch_smartspim_tiles(tmp_path, tmp_path / "o.tif", wavelength_nm=405)
    with pytest.raises(ValueError, match="placeholder"):
        stitch_smartspim_tiles(tmp_path, tmp_path / "o.tif", wavelength_nm=(488, 561))


def test_stitch_smartspim_tiles_inconsistent_plane_counts_raises(tmp_path):
    """
    Description
    -----------
    Two tiles with different Z-plane counts cannot be stitched into one
    aligned volume and raise ``ValueError``.
    """

    root = _make_smartspim_tree(tmp_path)
    # Drop one plane from tile B so the per-tile plane counts disagree.
    (root / "Ex_488_Ch0" / "20" / "20_0" / "000100.tiff").unlink()
    with pytest.raises(ValueError, match="Inconsistent number of Z planes"):
        stitch_smartspim_tiles(root, tmp_path / "o.tif", wavelength_nm=488, feather_pixels=1)


def test_stitch_smartspim_tiles_rejects_non_unsigned_tiles(tmp_path):
    """
    Description
    -----------
    SmartSPIM tiles must be unsigned-integer images; a float tile raises
    ``ValueError``.
    """

    root = _make_smartspim_tree(tmp_path)
    # Overwrite every plane with float data.
    for tile in ("0/0_0", "20/20_0"):
        for name in ("000000.tiff", "000100.tiff"):
            tifffile.imwrite(
                root / "Ex_488_Ch0" / tile.split("/")[0] / tile.split("/")[1] / name,
                np.zeros((4, 4), dtype=np.float32),
            )
    with pytest.raises(ValueError, match="unsigned integer tile dtype"):
        stitch_smartspim_tiles(root, tmp_path / "o.tif", wavelength_nm=488, feather_pixels=1)
