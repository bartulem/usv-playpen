"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.histology_stack_lightsheet_volume``.

Covers the pure in-plane flip helper against hand-computed arrays, the
LaVision filename discovery / sidecar exclusion, and an end-to-end stack
of tiny synthetic 4x4 uint16 OME-TIFF planes written into ``tmp_path``
(no real microscope data).
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from usv_playpen.neuropixels.histology_stack_lightsheet_volume import (
    _apply_xy_flip,
    _find_lavision_files,
    stack_lightsheet_volume,
)


def test_apply_xy_flip_all_modes():
    """
    Description
    -----------
    Each flip mode maps to the corresponding NumPy slice: ``none`` is
    identity, ``vertical`` reverses rows, ``horizontal`` reverses
    columns, and ``both`` reverses both axes.
    """

    img = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(_apply_xy_flip(img, "none"), img)
    np.testing.assert_array_equal(_apply_xy_flip(img, "vertical"), img[::-1, :])
    np.testing.assert_array_equal(_apply_xy_flip(img, "horizontal"), img[:, ::-1])
    np.testing.assert_array_equal(_apply_xy_flip(img, "both"), img[::-1, ::-1])


def test_apply_xy_flip_rejects_unknown_mode():
    """
    Description
    -----------
    An unrecognised flip mode raises ``ValueError`` rather than
    silently passing the image through.
    """

    with pytest.raises(ValueError, match="xy_flip must be one of"):
        _apply_xy_flip(np.zeros((2, 2)), "diagonal")


def _make_plane(tmp_path, name, value):
    """
    Description
    -----------
    Write a 4x4 uint16 TIFF filled with ``value`` at ``tmp_path/name``.

    Returns
    -------
    pathlib.Path
        Path to the written TIFF.
    """

    path = tmp_path / name
    tifffile.imwrite(path, np.full((4, 4), value, dtype=np.uint16))
    return path


def test_find_lavision_files_sorts_and_excludes_filter_sidecars(tmp_path):
    """
    Description
    -----------
    Discovery returns the channel's planes in ascending zero-padded Z
    order and excludes the ``Filter`` illumination sidecars and the
    other channel's files.
    """

    _make_plane(tmp_path, "t_UltraII_C00_xyz-Table Z0001.ome.tif", 1)
    _make_plane(tmp_path, "t_UltraII_C00_xyz-Table Z0000.ome.tif", 0)
    _make_plane(tmp_path, "t_UltraII_C00_Filter0000.ome.tif", 9)
    _make_plane(tmp_path, "t_UltraII_C01_xyz-Table Z0000.ome.tif", 5)

    found = _find_lavision_files(tmp_path, "C00")
    names = [p.name for p in found]
    assert names == [
        "t_UltraII_C00_xyz-Table Z0000.ome.tif",
        "t_UltraII_C00_xyz-Table Z0001.ome.tif",
    ]


def test_stack_lightsheet_volume_skips_first_and_stacks_in_order(tmp_path):
    """
    Description
    -----------
    A single-channel stack drops the metadata-bearing ``Z0000`` plane
    (``skip_first`` default) and writes the remaining planes in Z order
    to a BigTIFF readable back as a ``(n_planes, H, W)`` volume.
    """

    raw = tmp_path / "raw"
    raw.mkdir()
    _make_plane(raw, "t_UltraII_C00_xyz-Table Z0000.ome.tif", 0)
    _make_plane(raw, "t_UltraII_C00_xyz-Table Z0001.ome.tif", 11)
    _make_plane(raw, "t_UltraII_C00_xyz-Table Z0002.ome.tif", 22)

    out = tmp_path / "stack_488.tif"
    written = stack_lightsheet_volume(raw, out, wavelength_nm=488)
    assert written == [out]

    volume = tifffile.imread(out)
    assert volume.shape == (2, 4, 4)
    assert volume[0].max() == 11 and volume[0].min() == 11
    assert volume[1].max() == 22 and volume[1].min() == 22


def test_stack_lightsheet_volume_applies_xy_flip(tmp_path):
    """
    Description
    -----------
    With ``skip_first=False`` and a vertical flip, the single written
    plane equals the row-reversed source plane.
    """

    raw = tmp_path / "raw"
    raw.mkdir()
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(
        raw / "t_UltraII_C00_xyz-Table Z0000.ome.tif", arr
    )

    out = tmp_path / "flipped.tif"
    stack_lightsheet_volume(raw, out, wavelength_nm=488, xy_flip="vertical", skip_first=False)
    volume = tifffile.imread(out)
    np.testing.assert_array_equal(volume, arr[::-1, :])


def test_stack_lightsheet_volume_multi_wavelength_requires_placeholder(tmp_path):
    """
    Description
    -----------
    Requesting more than one wavelength without a ``{wavelength_nm}``
    placeholder in ``output_path`` raises ``ValueError`` before any I/O.
    """

    with pytest.raises(ValueError, match="placeholder"):
        stack_lightsheet_volume(tmp_path, tmp_path / "out.tif", wavelength_nm=(488, 561))


def test_stack_lightsheet_volume_rejects_unknown_wavelength(tmp_path):
    """
    Description
    -----------
    A wavelength outside the LaVision channel map (488 / 561) raises
    ``ValueError``.
    """

    with pytest.raises(ValueError, match="wavelength_nm values must each be one of"):
        stack_lightsheet_volume(tmp_path, tmp_path / "out.tif", wavelength_nm=405)


def test_stack_lightsheet_volume_missing_channel_files_raises(tmp_path):
    """
    Description
    -----------
    An acquisition directory with no files for the requested channel
    raises ``FileNotFoundError``.
    """

    raw = tmp_path / "empty"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="No LaVision OME-TIFF files"):
        stack_lightsheet_volume(raw, tmp_path / "out.tif", wavelength_nm=488)


def test_stack_lightsheet_volume_empty_wavelength_iterable_raises(tmp_path):
    """
    Description
    -----------
    An empty ``wavelength_nm`` iterable has no channel to stack and
    raises ``ValueError``.
    """

    with pytest.raises(ValueError, match="non-empty"):
        stack_lightsheet_volume(tmp_path, tmp_path / "out.tif", wavelength_nm=())


def test_stack_lightsheet_volume_z_flip_reverses_plane_order(tmp_path):
    """
    Description
    -----------
    With ``z_flip=True`` and ``skip_first=False`` the written volume's
    plane order is reversed relative to the lexicographic file order.
    """

    raw = tmp_path / "raw"
    raw.mkdir()
    _make_plane(raw, "t_UltraII_C00_xyz-Table Z0000.ome.tif", 1)
    _make_plane(raw, "t_UltraII_C00_xyz-Table Z0001.ome.tif", 2)
    out = tmp_path / "zflip.tif"
    stack_lightsheet_volume(raw, out, wavelength_nm=488, z_flip=True, skip_first=False)
    volume = tifffile.imread(out)
    assert volume[0].max() == 2 and volume[1].max() == 1
