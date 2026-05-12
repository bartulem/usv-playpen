"""
@author: bartulem
Stacks individual LaVision UltraMicroscope OME-TIFF Z-planes from a single
acquisition into one BigTIFF volume per fluorescence channel. Designed for
downstream histology / atlas-registration workflows.

LaVision file-naming conventions assumed
----------------------------------------
Two filename layouts written by the LaVision Inspector software are
recognised:

* New convention:
  ``{HH-MM-SS}_{user}_{cageID}_lv_{dv|vd}_{n}_UltraII_C0{0|1}_xyz-Table Z{####}.ome.tif``
* Older convention:
  ``{HH-MM-SS}_{user}-{cageID}-{dv|vd}_UltraII_C0{0|1}_xyz-Table Z{####}.ome.tif``

In both layouts:

* ``C00`` is the autofluorescence channel (488 nm).
* ``C01`` is the excitation channel (561 nm).
* ``UltraII Filter000{0,1}.ome.tif`` sidecars hold illumination-side
  metadata (left/right LaVision lightsheet) and are excluded from the
  stack.

Z-direction control
-------------------
The output volume's Z order is controlled by the ``z_flip`` argument.
When ``z_flip=True`` the Z iteration is reversed; otherwise the planes
are written in lexicographic filename order (which matches the
acquisition Z order written by ImSpector). The correct value depends
on how the brain was mounted in the LaVision holder and on the
orientation downstream tools expect (``napari`` / ``brainglobe``
render coronal and sagittal views with plane 0 at the bottom, so
ventral-first input yields dorsal-at-top).
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


_LAVISION_CHANNEL_FROM_WAVELENGTH = {488: 'C00', 561: 'C01'}


def _find_lavision_files(tiff_dir: Path, channel_token: str) -> list[Path]:
    """
    Description
    -----------
    Returns the sorted list of LaVision OME-TIFF Z-slice files for the
    requested channel, excluding the ``Filter000{0,1}`` illumination
    sidecars. Sorting is lexicographic, which matches the zero-padded
    ``Z####`` index in the LaVision filename convention and therefore
    yields the slices in ascending acquisition Z order.

    Parameters
    ----------
    tiff_dir (Path)
        Acquisition directory containing the flat list of LaVision
        OME-TIFF files.
    channel_token (str)
        LaVision channel token (``'C00'`` or ``'C01'``).

    Returns
    -------
    files (list[Path])
        Sorted, filter-excluded list of OME-TIFF paths for the
        requested channel.
    """

    pattern = str(tiff_dir / f'*{channel_token}*.ome.tif')
    return sorted(
        Path(p)
        for p in glob.glob(pattern)
        if 'Filter' not in os.path.basename(p)
    )


_XY_FLIP_OPTIONS = ('none', 'vertical', 'horizontal', 'both')


def _apply_xy_flip(image: np.ndarray, flip: str) -> np.ndarray:
    """
    Description
    -----------
    Applies an in-plane axis flip to a 2D image.

    Parameters
    ----------
    image (np.ndarray)
        2D image array.
    flip (str)
        Flip mode. One of:

        * ``'none'`` — return the image unchanged.
        * ``'vertical'`` — reverse rows (top <-> bottom).
        * ``'horizontal'`` — reverse columns (left <-> right).
        * ``'both'`` — reverse both axes (equivalent to a 180-degree
          in-plane rotation).

    Returns
    -------
    flipped (np.ndarray)
        The flipped image. The original ``image`` is returned
        unmodified when ``flip == 'none'``.
    """

    if flip == 'none':
        return image
    if flip == 'vertical':
        return image[::-1, :]
    if flip == 'horizontal':
        return image[:, ::-1]
    if flip == 'both':
        return image[::-1, ::-1]
    raise ValueError(
        f"xy_flip must be one of {_XY_FLIP_OPTIONS}, got {flip!r}."
    )


def stack_lightsheet_volume(
    raw_dir: str | Path,
    output_path: str | Path,
    wavelength_nm: int | Iterable[int] = (488, 561),
    *,
    xy_flip: str = 'none',
    z_flip: bool = False,
    skip_first: bool = True,
) -> list[Path]:
    """
    Description
    -----------
    Stacks LaVision UltraMicroscope OME-TIFF Z-planes from a single
    acquisition directory into one BigTIFF volume per requested
    fluorescence channel. By default, the function processes both
    channels (488 nm autofluorescence and 561 nm excitation) and
    writes one BigTIFF per channel.

    Per-channel destination paths are derived from ``output_path`` via
    a ``{wavelength_nm}`` substring placeholder. The placeholder is
    required when more than one wavelength is requested. For a single
    wavelength the placeholder is optional; when present it is
    formatted with the wavelength as an integer, and when absent the
    literal ``output_path`` is used.

    The function handles:

    * Channel resolution from a semantic wavelength argument
      (488 nm -> ``C00`` autofluorescence; 561 nm -> ``C01`` excitation).
    * Exclusion of LaVision ``Filter000{0,1}`` illumination sidecars.
    * Optional in-plane axis flip (`'none'`, `'vertical'`,
      `'horizontal'`, `'both'`) applied to every plane prior to
      writing. ``'both'`` is equivalent to a 180-degree rotation.
    * Optional Z-axis reversal via ``z_flip``. The correct value
      depends on how the brain was mounted in the LaVision holder
      and on the orientation downstream tools expect.
    * Skipping of the first sorted file. The LaVision ImSpector
      software writes the entire OME-XML acquisition description
      (typically ~300-400 KB: ``<OME>``, ``<Experimenter>``,
      ``<Pixels>``, the per-plane ``<TiffData>`` table, and a long
      ``<Properties>`` block of laser/exposure/stage parameters) into
      the ``ImageDescription`` TIFF tag of the ``Z0000`` plane
      only — every subsequent ``Z####.ome.tif`` is written without
      that tag. ``tifffile.TiffWriter`` in ``contiguous=True`` mode
      assumes a uniform tag layout across pages, so feeding ``Z0000``
      as the first page causes the writer to either embed ``Z0000``'s
      OME header into the multi-page output and then stall on the
      next page (whose tag set is smaller), or to fail with a tag
      mismatch. Dropping the first sorted file sidesteps the issue
      and loses no image data: ``Z0000`` is itself a valid 2D plane,
      and the OME-XML's ``<TiffData>`` list shows that the plane
      content is acquisition slice 0, immediately followed by the
      identically-sized ``Z0001``. The default is therefore
      ``True``; set to ``False`` only for the rare acquisition that
      is known to be uniform across all sorted planes.

    Parameters
    ----------
    raw_dir (str | Path)
        LaVision acquisition directory containing the flat list of
        ``*_UltraII_C0{0,1}_xyz-Table Z{####}.ome.tif`` files.
    output_path (str | Path)
        Destination path for the BigTIFF stacks. May contain a
        ``{wavelength_nm}`` placeholder that is formatted per channel.
        The placeholder is required when ``wavelength_nm`` is an
        iterable of more than one value. Parent directories are
        created if they do not already exist.
    wavelength_nm (int | Iterable[int])
        Fluorescence channel wavelength(s) in nanometres. Each value
        must be one of ``488`` (autofluorescence channel ``C00``) or
        ``561`` (excitation channel ``C01``). Defaults to
        ``(488, 561)``, which processes both channels.
    xy_flip (str)
        In-plane axis flip applied to every 2D plane before writing.
        One of ``{'none', 'vertical', 'horizontal', 'both'}``.
        ``'vertical'`` reverses rows (top <-> bottom), ``'horizontal'``
        reverses columns (left <-> right), and ``'both'`` reverses
        both axes (equivalent to a 180-degree in-plane rotation).
        Defaults to ``'none'``. The caller is responsible for
        selecting the value that matches the LaVision sample-mount
        convention used for the acquisition.
    z_flip (bool)
        If ``True``, the Z iteration order is reversed before writing
        (the lexicographic file order is reversed). Defaults to
        ``False``. The correct value depends on the sample mount and
        on whether the downstream tool (e.g. ``napari`` /
        ``brainglobe``) expects ventral-first or dorsal-first Z.
        Pick by trial: if the output renders upside-down in coronal /
        sagittal views, toggle this flag.
    skip_first (bool)
        If ``True`` (default), the first sorted file is dropped. This
        is necessary on LaVision acquisitions because the LaVision
        ImSpector software writes the full OME-XML acquisition
        description (~300-400 KB) into the ``ImageDescription`` TIFF
        tag of the ``Z0000`` plane only; later planes carry no such
        tag. ``tifffile.TiffWriter`` in ``contiguous=True`` mode
        assumes uniform tags across pages and stalls when the second
        page's tag set diverges from the first. Set to ``False`` only
        for the rare acquisition known to be uniform across all
        sorted planes.

    Returns
    -------
    output_paths (list[Path])
        Absolute paths to the written BigTIFF stacks, one entry per
        wavelength in the order given.
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
        if wl not in _LAVISION_CHANNEL_FROM_WAVELENGTH:
            raise ValueError(
                f"wavelength_nm values must each be one of "
                f"{sorted(_LAVISION_CHANNEL_FROM_WAVELENGTH)}, got {wl!r}."
            )

    output_path_template = str(output_path)
    has_placeholder = '{wavelength_nm}' in output_path_template
    if len(wavelengths) > 1 and not has_placeholder:
        raise ValueError(
            "output_path must contain a '{wavelength_nm}' placeholder when "
            "more than one wavelength is requested."
        )

    raw_path = Path(raw_dir)
    written: list[Path] = []
    for wl in wavelengths:
        channel_token = _LAVISION_CHANNEL_FROM_WAVELENGTH[wl]
        files = _find_lavision_files(raw_path, channel_token)
        if not files:
            raise FileNotFoundError(
                f"No LaVision OME-TIFF files for channel {channel_token} "
                f"found in {raw_path}."
            )

        if skip_first:
            files = files[1:]

        if z_flip:
            files = list(reversed(files))

        out_str = (
            output_path_template.format(wavelength_nm=wl)
            if has_placeholder else output_path_template
        )
        out = Path(out_str)
        out.parent.mkdir(parents=True, exist_ok=True)

        with tifffile.TiffWriter(out, bigtiff=True) as writer:
            for path in tqdm(files, desc=f'Stacking {channel_token}'):
                plane = tifffile.imread(path)
                plane = _apply_xy_flip(plane, xy_flip)
                writer.write(plane, metadata={'axes': 'ZYX'}, contiguous=True)

        written.append(out)

    return written
