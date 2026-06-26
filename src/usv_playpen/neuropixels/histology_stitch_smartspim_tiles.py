"""
@author: bartulem
Stitches a tiled LifeCanvas SmartSPIM acquisition into a single BigTIFF
volume per fluorescence channel using stage-coordinate tile placement
and linear feather blending in tile overlaps. Designed for downstream
histology / atlas-registration workflows.

SmartSPIM acquisition layout assumed
------------------------------------
``{root}/``

    ``Ex_488_Ch0/{X}/{X}_{Y}/{NNNNNN}.tiff``
    ``Ex_561_Ch1/{X}/{X}_{Y}/{NNNNNN}.tiff``
    ``metadata.txt``
    ...

Each ``{X}/{X}_{Y}/`` directory contains the Z-stack for one tile, with
plane files sorted lexicographically by their zero-padded numeric name.
The ``X`` and ``Y`` tokens in the directory names are integer stage
coordinates in units of 0.1 micrometres (SmartSPIM convention).

``metadata.txt`` provides the lateral pixel size in micrometres and the
axial Z step in micrometres, parsed from the data row immediately
following the ``Obj`` header line.

Z-direction control
-------------------
The output volume's Z order is controlled by the ``z_flip`` argument.
When ``z_flip=True`` the Z iteration is reversed; otherwise the planes
are written in the lexicographic on-disk order. The correct value
depends on how the brain was mounted in the SmartSPIM holder and on
the orientation downstream tools expect (``napari`` / ``brainglobe``
render coronal and sagittal views with plane 0 at the bottom, so
ventral-first input yields dorsal-at-top).

Stitching strategy
------------------
Tile placement: stage coordinates are converted to integer pixel
offsets in a global canvas whose origin coincides with the upper-left
corner of the tile with the smallest stage X and Y. Tile overlaps are
blended with a bevel-shaped linear feather weight map (distance to the
nearest tile edge clipped to ``feather_pixels``), which yields smooth
seams without sub-pixel registration.

Streaming: the volume is written plane-by-plane, so peak memory stays
at roughly two canvas-sized float32 planes plus the per-tile uint16
plane reads (typically well under 1 GB), regardless of total stack
size.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


_SMARTSPIM_CHANNEL_FROM_WAVELENGTH = {488: 'Ex_488_Ch0', 561: 'Ex_561_Ch1'}
_STAGE_UNIT_UM = 0.1


def _parse_smartspim_metadata(metadata_path: Path) -> dict:
    """
    Description
    -----------
    Parses a SmartSPIM ``metadata.txt`` file and returns the lateral
    pixel size and the axial Z step in micrometres. Both values are
    read from the tab-separated data row immediately following the
    header line that begins with ``Obj``. The ``pixel_size_um`` value
    is the 5th column and ``z_step_um`` is the 6th column of that data
    row.

    The metadata file uses a ``micro`` character that may be written
    in a non-UTF-8 single-byte encoding, so the file is read as
    ``latin-1`` to tolerate this.

    Parameters
    ----------
    metadata_path (Path)
        Path to a SmartSPIM acquisition ``metadata.txt`` file.

    Returns
    -------
    info (dict)
        Dictionary with keys ``'pixel_size_um'`` (float) and
        ``'z_step_um'`` (float).
    """

    text = metadata_path.read_text(encoding='latin-1')
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        fields = [f.strip() for f in line.split('\t')]
        if fields and fields[0] == 'Obj' and idx + 1 < len(lines):
            data = [f.strip() for f in lines[idx + 1].split('\t')]
            # Only treat this as the data row when it has the expected
            # column count; a short/blank row (e.g. a trailing empty line
            # that splits to ['']) would otherwise raise a bare IndexError
            # on data[4]/data[5], bypassing the descriptive ValueError
            # below. Falling through keeps scanning and ultimately raises
            # the informative error that carries the metadata path.
            if len(data) > 5:
                return {
                    'pixel_size_um': float(data[4]),
                    'z_step_um': float(data[5]),
                }
    raise ValueError(
        f"Could not parse pixel size / Z step from {metadata_path}."
    )


def _enumerate_tile_dirs(channel_dir: Path) -> dict[tuple[int, int], Path]:
    """
    Description
    -----------
    Returns a mapping from ``(x_stage, y_stage)`` (in native 0.1 um
    units, as written in the SmartSPIM tile directory names) to the
    directory containing that tile's Z-plane TIFFs. The SmartSPIM
    channel directory layout is ``{channel}/{X}/{X}_{Y}/``, where both
    ``X`` and ``Y`` are integer-valued in the directory names.

    Sub-directories whose names do not match the expected integer
    pattern, or whose ``{X}_{Y}`` parent disagrees with the outer
    ``{X}`` directory, are skipped silently.

    Parameters
    ----------
    channel_dir (Path)
        Channel-level directory (for example
        ``.../Ex_488_Ch0``).

    Returns
    -------
    tile_dirs (dict)
        Mapping ``(x, y) -> Path`` where ``x`` and ``y`` are integers
        in 0.1 um units and ``Path`` points to the directory holding
        the tile's Z-plane files.
    """

    tile_dirs: dict[tuple[int, int], Path] = {}
    for x_dir in sorted(p for p in channel_dir.iterdir() if p.is_dir()):
        try:
            x_val = int(x_dir.name)
        except ValueError:
            continue
        for xy_dir in sorted(p for p in x_dir.iterdir() if p.is_dir()):
            parts = xy_dir.name.split('_')
            if len(parts) != 2:
                continue
            try:
                x_check, y_val = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if x_check != x_val:
                continue
            tile_dirs[(x_val, y_val)] = xy_dir
    return tile_dirs


def _list_plane_files(tile_dir: Path) -> list[Path]:
    """
    Description
    -----------
    Returns the sorted list of TIFF Z-plane files inside one SmartSPIM
    tile directory. Sorting is lexicographic, which matches the
    zero-padded numeric SmartSPIM filename convention (for example
    ``000000.tiff``, ``000100.tiff``, ...) and therefore yields the
    Z-planes in ascending acquisition order.

    Parameters
    ----------
    tile_dir (Path)
        Directory containing a tile's Z-plane TIFFs.

    Returns
    -------
    files (list[Path])
        Sorted plane file paths.
    """

    return sorted(tile_dir.glob('*.tiff'))


def _compute_tile_layout(
    tile_positions: list[tuple[int, int]],
    pixel_size_um: float,
    tile_shape_yx: tuple[int, int],
) -> tuple[dict[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    """
    Description
    -----------
    Converts SmartSPIM stage coordinates (in 0.1 um units) to integer
    pixel offsets in a global canvas. The canvas origin is the
    upper-left corner of the tile with the smallest stage ``X`` and
    smallest stage ``Y``. Stage ``X`` is mapped to the canvas column
    axis; stage ``Y`` is mapped to the canvas row axis. The canvas
    shape is taken as the bounding box of all tile placements.

    Parameters
    ----------
    tile_positions (list[tuple[int, int]])
        Tile positions as ``(x_stage, y_stage)`` integer pairs in
        0.1 um units.
    pixel_size_um (float)
        Lateral pixel size in micrometres.
    tile_shape_yx (tuple[int, int])
        Tile image shape ``(height, width)`` in pixels.

    Returns
    -------
    layout (dict)
        Mapping ``(x_stage, y_stage) -> (y_offset_px, x_offset_px)``.
    canvas_shape (tuple[int, int])
        ``(height, width)`` of the global canvas in pixels.
    """

    if not tile_positions:
        raise ValueError("tile_positions is empty.")

    min_x = min(x for x, _ in tile_positions)
    min_y = min(y for _, y in tile_positions)
    height, width = tile_shape_yx

    layout: dict[tuple[int, int], tuple[int, int]] = {}
    max_y_end = 0
    max_x_end = 0
    for x_stage, y_stage in tile_positions:
        x_off_um = (x_stage - min_x) * _STAGE_UNIT_UM
        y_off_um = (y_stage - min_y) * _STAGE_UNIT_UM
        x_off_px = int(round(x_off_um / pixel_size_um))
        y_off_px = int(round(y_off_um / pixel_size_um))
        layout[(x_stage, y_stage)] = (y_off_px, x_off_px)
        max_y_end = max(max_y_end, y_off_px + height)
        max_x_end = max(max_x_end, x_off_px + width)

    return layout, (max_y_end, max_x_end)


def _make_bevel_weights(
    tile_shape_yx: tuple[int, int],
    feather_pixels: int,
) -> np.ndarray:
    """
    Description
    -----------
    Builds a 2D linear feather weight map for one tile. The weight at
    pixel ``(i, j)`` is the minimum of the row-distance to the nearest
    top/bottom edge and the column-distance to the nearest left/right
    edge, divided by ``feather_pixels`` and clipped to a small
    positive floor (``1e-3``) and ``1``. The result is a bevel-shaped
    weight map that ramps from a small positive value at the tile
    boundary up to ``1`` inside the central plateau.

    The boundary weight is strictly positive because the edge distance
    is 1-based (the outermost row/column has distance ``1``, never
    ``0``), so the ramp value at the boundary is ``1 / feather_pixels``
    (~0.0156 at the default ``feather_pixels=64``). The ``1e-3`` floor
    is a defensive lower bound that does not engage for normal feather
    widths -- it only takes effect when ``feather_pixels`` is so large
    that ``1 / feather_pixels < 1e-3`` (i.e. ``feather_pixels > 1000``).
    Either way the per-pixel normalization stays well-defined for canvas
    pixels covered by a single tile.

    Parameters
    ----------
    tile_shape_yx (tuple[int, int])
        Tile image shape ``(height, width)``.
    feather_pixels (int)
        Width of the linear ramp in pixels. Must be ``>= 1``.

    Returns
    -------
    weights (np.ndarray)
        2D ``float32`` array of shape ``(height, width)``.
    """

    if feather_pixels < 1:
        raise ValueError(
            f"feather_pixels must be >= 1, got {feather_pixels}."
        )
    height, width = tile_shape_yx
    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)
    # 1-based distance to the nearest edge: the ``+ 1.0`` on only the
    # top/left side makes the outermost row/column get distance 1 (not 0),
    # so the boundary ramp value is 1/feather_pixels and is never fully
    # zero-weighted. This is intentional, not an off-by-one.
    row_edge_dist = np.minimum(rows + 1.0, height - rows)
    col_edge_dist = np.minimum(cols + 1.0, width - cols)
    row_weight = np.clip(row_edge_dist / feather_pixels, 1e-3, 1.0)
    col_weight = np.clip(col_edge_dist / feather_pixels, 1e-3, 1.0)
    return np.minimum(row_weight[:, None], col_weight[None, :])


def _resolve_channel_dir(raw_dir: Path, wavelength_nm: int) -> Path:
    """
    Description
    -----------
    Resolves the SmartSPIM channel directory for the requested
    wavelength under the acquisition root.

    Parameters
    ----------
    raw_dir (Path)
        SmartSPIM acquisition root directory.
    wavelength_nm (int)
        Fluorescence channel wavelength in nanometres.

    Returns
    -------
    channel_dir (Path)
        The matching ``Ex_{nnn}_Ch{k}`` directory under ``raw_dir``.
    """

    if wavelength_nm not in _SMARTSPIM_CHANNEL_FROM_WAVELENGTH:
        raise ValueError(
            f"wavelength_nm must be one of "
            f"{sorted(_SMARTSPIM_CHANNEL_FROM_WAVELENGTH)}, got "
            f"{wavelength_nm!r}."
        )
    channel_dir = raw_dir / _SMARTSPIM_CHANNEL_FROM_WAVELENGTH[wavelength_nm]
    if not channel_dir.is_dir():
        raise FileNotFoundError(
            f"SmartSPIM channel directory not found: {channel_dir}."
        )
    return channel_dir


def stitch_smartspim_tiles(
    raw_dir: str | Path,
    output_path: str | Path,
    wavelength_nm: int | Iterable[int] = (488, 561),
    *,
    z_flip: bool = False,
    feather_pixels: int = 64,
) -> list[Path]:
    """
    Description
    -----------
    Stitches a SmartSPIM acquisition into one BigTIFF volume per
    requested fluorescence channel using stage-coordinate tile
    placement and linear feather blending. By default, the function
    processes both channels (``Ex_488_Ch0`` and ``Ex_561_Ch1``) and
    writes one BigTIFF per channel.

    Per-channel destination paths are derived from ``output_path`` via
    a ``{wavelength_nm}`` substring placeholder. The placeholder is
    required when more than one wavelength is requested. For a single
    wavelength the placeholder is optional; when present it is
    formatted with the wavelength as an integer, and when absent the
    literal ``output_path`` is used.

    For each channel, stitching is performed plane-by-plane: for
    every Z index in ``[0, n_planes)`` the function

    1. allocates a float32 accumulator and a float32 weight buffer of
       canvas shape,
    2. for each tile reads its Z-th plane, multiplies the plane by the
       tile's bevel weight map, adds the result into the accumulator,
       and adds the weight map into the weight buffer,
    3. divides the accumulator by the weight buffer with safe handling
       of fully-uncovered pixels, casts back to the input integer
       dtype, and appends the stitched plane to the output BigTIFF.

    Tile positions are taken from the on-disk directory names rather
    than the ``metadata.txt`` tile table, because the on-disk layout
    is the authoritative record of which tiles were actually written
    for the requested channel. The ``metadata.txt`` file is used only
    to obtain the lateral pixel size, which is required to convert
    stage coordinates (0.1 um units) into integer pixel offsets, and
    is parsed once per call regardless of the number of channels.

    Z-direction handling: the Z iteration order is reversed iff
    ``z_flip=True``. Per-plane in-plane rotation is intentionally not
    applied here, because tile placement is expressed in the SmartSPIM
    stage frame; any whole-volume reorientation can be applied as a
    separate downstream step.

    Parameters
    ----------
    raw_dir (str | Path)
        SmartSPIM acquisition root (the parent directory containing
        ``Ex_488_Ch0``, ``Ex_561_Ch1``, ``metadata.txt`` and the other
        SmartSPIM sidecar files).
    output_path (str | Path)
        Destination BigTIFF path. May contain a ``{wavelength_nm}``
        placeholder that is formatted per channel. The placeholder is
        required when ``wavelength_nm`` is an iterable of more than
        one value. Parent directories are created if they do not
        already exist.
    wavelength_nm (int | Iterable[int])
        Fluorescence channel wavelength(s) in nanometres. Each value
        must be one of ``488`` (``Ex_488_Ch0``) or ``561``
        (``Ex_561_Ch1``). Defaults to ``(488, 561)``, which processes
        both channels.
    z_flip (bool)
        If ``True``, the Z iteration order is reversed before writing
        (the on-disk plane order is reversed). Defaults to ``False``.
        Pick by trial: if the output renders upside-down in coronal /
        sagittal views of ``napari`` / ``brainglobe``, toggle this
        flag.
    feather_pixels (int)
        Width of the linear feather ramp at each tile edge, in pixels.
        Defaults to ``64``. Should be at most half the smaller tile
        dimension; for typical SmartSPIM tiles (~1600-2000 px) the
        default is comfortably inside that bound.

    Returns
    -------
    output_paths (list[Path])
        Absolute paths to the written stitched BigTIFF stacks, one
        entry per wavelength in the order given.
    """

    if isinstance(wavelength_nm, int):
        wavelengths: tuple[int, ...] = (wavelength_nm,)
    else:
        wavelengths = tuple(wavelength_nm)
    if not wavelengths:
        raise ValueError(
            "wavelength_nm must be a non-empty int or iterable of ints."
        )
    for wl in wavelengths:
        if wl not in _SMARTSPIM_CHANNEL_FROM_WAVELENGTH:
            raise ValueError(
                f"wavelength_nm values must each be one of "
                f"{sorted(_SMARTSPIM_CHANNEL_FROM_WAVELENGTH)}, got {wl!r}."
            )

    output_path_template = str(output_path)
    has_placeholder = '{wavelength_nm}' in output_path_template
    if len(wavelengths) > 1 and not has_placeholder:
        raise ValueError(
            "output_path must contain a '{wavelength_nm}' placeholder when "
            "more than one wavelength is requested."
        )

    raw_path = Path(raw_dir)
    metadata_path = raw_path / 'metadata.txt'
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"SmartSPIM metadata.txt not found: {metadata_path}."
        )
    meta = _parse_smartspim_metadata(metadata_path)

    written: list[Path] = []
    for wl in wavelengths:
        channel_dir = _resolve_channel_dir(raw_path, wl)

        tile_dirs = _enumerate_tile_dirs(channel_dir)
        if not tile_dirs:
            raise FileNotFoundError(
                f"No SmartSPIM tile directories found under {channel_dir}."
            )

        plane_files_per_tile: dict[tuple[int, int], list[Path]] = {}
        for tile_xy, tile_dir in tile_dirs.items():
            files = _list_plane_files(tile_dir)
            if not files:
                raise FileNotFoundError(
                    f"No TIFF planes found in tile directory {tile_dir}."
                )
            plane_files_per_tile[tile_xy] = files

        plane_counts = {len(v) for v in plane_files_per_tile.values()}
        if len(plane_counts) != 1:
            raise ValueError(
                f"Inconsistent number of Z planes across tiles in "
                f"{channel_dir}: {sorted(plane_counts)}."
            )
        n_planes = plane_counts.pop()

        first_tile_files = next(iter(plane_files_per_tile.values()))
        first_plane = tifffile.imread(first_tile_files[0])
        if first_plane.ndim != 2:
            raise ValueError(
                f"Expected 2D tile planes, got shape {first_plane.shape} from "
                f"{first_tile_files[0]}."
            )
        if not np.issubdtype(first_plane.dtype, np.unsignedinteger):
            raise ValueError(
                f"Expected unsigned integer tile dtype, got "
                f"{first_plane.dtype}."
            )
        tile_shape_yx = (int(first_plane.shape[0]), int(first_plane.shape[1]))
        tile_dtype = first_plane.dtype
        dtype_max = int(np.iinfo(tile_dtype).max)

        layout, canvas_shape = _compute_tile_layout(
            tile_positions=list(plane_files_per_tile.keys()),
            pixel_size_um=meta['pixel_size_um'],
            tile_shape_yx=tile_shape_yx,
        )
        weights_2d = _make_bevel_weights(tile_shape_yx, feather_pixels)

        z_indices = list(range(n_planes))
        if z_flip:
            z_indices.reverse()

        tile_height, tile_width = tile_shape_yx
        out_str = (
            output_path_template.format(wavelength_nm=wl)
            if has_placeholder else output_path_template
        )
        out_path = Path(out_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tifffile.TiffWriter(out_path, bigtiff=True) as writer:
            for z in tqdm(z_indices, desc=f'Stitching {channel_dir.name}'):
                accumulator = np.zeros(canvas_shape, dtype=np.float32)
                weight_sum = np.zeros(canvas_shape, dtype=np.float32)
                for tile_xy, files in plane_files_per_tile.items():
                    tile_img = tifffile.imread(files[z]).astype(np.float32)
                    if tile_img.shape != tile_shape_yx:
                        raise ValueError(
                            f"Plane shape {tile_img.shape} at {files[z]} does "
                            f"not match reference {tile_shape_yx}."
                        )
                    y_off, x_off = layout[tile_xy]
                    y_slice = slice(y_off, y_off + tile_height)
                    x_slice = slice(x_off, x_off + tile_width)
                    accumulator[y_slice, x_slice] += tile_img * weights_2d
                    weight_sum[y_slice, x_slice] += weights_2d
                safe_weights = np.where(weight_sum > 0.0, weight_sum, 1.0)
                stitched = np.where(
                    weight_sum > 0.0,
                    accumulator / safe_weights,
                    0.0,
                )
                stitched = np.clip(stitched, 0, dtype_max).astype(tile_dtype)
                # Append this 2D plane as the next Z slice; contiguous=True
                # together with the axes='ZYX' tag makes tifffile build a
                # single ZYX BigTIFF across loop iterations (do not drop
                # contiguous=True or change the axes tag, or stacking breaks).
                writer.write(stitched, metadata={'axes': 'ZYX'}, contiguous=True)

        written.append(out_path)

    return written
