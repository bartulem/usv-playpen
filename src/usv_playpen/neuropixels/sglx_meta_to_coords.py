"""
@author: bartulem (clean-room re-implementation; see module docstring)

Convert a SpikeGLX ``.meta`` file into a probe-geometry artefact
consumable by downstream spike sorters and analysis tools.

Background
----------
Imec / SpikeGLX records its probe geometry next to every binary by
writing a sidecar ``<run>_t0.imec0.ap.meta`` text file. Two relevant
keys live there:

- ``snsShankMap`` -- present in older metadata files
  (pre-SpikeGLX 032623); encodes (shank, column, row, connected) per
  saved channel and requires an external probe-type → geometry table
  to translate to (x, y) micrometres.
- ``snsGeomMap`` -- present in newer files (SpikeGLX 032623 onward);
  encodes (shank, x_um, y_um, connected) directly and supersedes
  ``snsShankMap``.

Downstream sorters / packages each consume a different per-channel
geometry artefact:

- Kilosort 2/3/4: a MATLAB ``chanMap.mat`` file with one column each
  of ``chanMap``, ``chanMap0ind``, ``xcoords``, ``ycoords``,
  ``kcoords``, ``connected``, plus a ``name`` scalar.
- JRClust:       three string assignments (``shankMap``, ``siteLoc``,
                 ``siteMap``) to paste into a ``.prm`` file.
- YASS / others: a ``(n_channels, 2)`` ``.npy`` array of ``(x, y)``.
- Generic:       a tab-delimited text file
                 ``index<TAB>x_um<TAB>y_um<TAB>shank_index``.
- Legacy meta:   the original file augmented in-place with the new
                 fields (``snsGeomMap``, ``muxTbl``, etc.) that
                 SpikeGLX 032623+ otherwise writes; a backup of the
                 original is preserved as ``<base>_orig.meta``.

Provenance
----------
This module is a clean-room re-implementation -- it does NOT
incorporate any source from Jennifer Colonell's ``SGLXMetaToCoords``
repository (https://github.com/jenniferColonell/SGLXMetaToCoords),
which carries no LICENSE file. The probe geometry numbers, MUX table
patterns, and metadata key/value layouts encoded below are factual
hardware specifications and file-format definitions taken from the
public SpikeGLX documentation
(https://billkarsh.github.io/SpikeGLX/Sgl_help/Metadata_30.html) and
Imec's Neuropixels documentation.

Usage
-----
Programmatic::

    from pathlib import Path
    from usv_playpen.neuropixels.sglx_meta_to_coords import (
        OutputFormat,
        parse_spikeglx_meta,
        coords_from_meta,
        write_coords_file,
    )
    meta = parse_spikeglx_meta(Path('/path/to/run.imec0.ap.meta'))
    coords = coords_from_meta(meta)
    dst = write_coords_file(
        meta=meta,
        coords=coords,
        output_format=OutputFormat.KILOSORT_MAT,
        save_dir=Path('/some/dir'),
        base_name='run.imec0.ap',
    )

Interactive (Qt-based GUI; same family as the project's main GUI)::

    npx-meta-to-coords           # console-script entry point
    # or
    python -m usv_playpen.neuropixels.sglx_meta_to_coords
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

logger = logging.getLogger(__name__)


# Imec probe geometry -- one dataclass instance per probe family.
# All numbers come from Imec's published Neuropixels documentation
# (electrode pitch, shank pitch / width, electrode-array sizes).
# Fields are intentionally explicit so the resulting code reads as
# physics-of-the-probe rather than as an opaque seven-tuple.
@dataclass(frozen=True)
class ProbeGeometry:
    """
    Description
    -----------
    Physical layout parameters for one Imec Neuropixels probe family.

    The ``x_offset_even_um`` / ``x_offset_odd_um`` split exists because
    several probe families (NP1.0, NP-NHP staggered, NP1300) use a
    staggered checkerboard layout in which even-row electrodes sit at
    one x-offset and odd-row electrodes at another. For non-staggered
    (linear) probes the two offsets are equal.

    Parameters
    ----------
    n_shanks (int)
        Number of shanks on this probe.
    shank_width_um (float)
        Width of one shank in micrometres.
    shank_pitch_um (float)
        Centre-to-centre distance between adjacent shanks in
        micrometres (``0`` for single-shank probes).
    x_offset_even_um (float)
        x position of an electrode in column 0 on an even row.
    x_offset_odd_um (float)
        x position of an electrode in column 0 on an odd row.
    horizontal_pitch_um (float)
        Centre-to-centre spacing between adjacent columns of
        electrodes within one shank.
    vertical_pitch_um (float)
        Centre-to-centre spacing between adjacent rows of electrodes
        within one shank.
    rows_per_shank (int)
        Number of electrode rows on one shank (``electrodes_per_shank
        / columns_per_shank``).
    electrodes_per_shank (int)
        Total electrode count on one shank.
    """

    n_shanks: int
    shank_width_um: float
    shank_pitch_um: float
    x_offset_even_um: float
    x_offset_odd_um: float
    horizontal_pitch_um: float
    vertical_pitch_um: float
    rows_per_shank: int
    electrodes_per_shank: int

    @property
    def columns_per_shank(self) -> int:
        """Number of electrode columns on one shank (derived)."""

        return self.electrodes_per_shank // self.rows_per_shank


# Geometry families used by multiple part numbers; each is referenced
# by name from PROBE_GEOMETRY below to keep the part-number → geometry
# mapping compact and easy to audit against Imec's docs.
_NP1_STAG_70UM = ProbeGeometry(1, 70.0, 0.0, 27.0, 11.0, 32.0, 20.0, 480, 960)
_NHP_LIN_70UM = ProbeGeometry(1, 70.0, 0.0, 27.0, 27.0, 32.0, 20.0, 480, 960)
_NHP_STAG_125UM_MED = ProbeGeometry(1, 125.0, 0.0, 27.0, 11.0, 87.0, 20.0, 1368, 2496)
_NHP_STAG_125UM_LONG = ProbeGeometry(1, 125.0, 0.0, 27.0, 11.0, 87.0, 20.0, 2208, 4416)
_NHP_LIN_125UM_MED = ProbeGeometry(1, 125.0, 0.0, 11.0, 11.0, 103.0, 20.0, 1368, 2496)
_NHP_LIN_125UM_LONG = ProbeGeometry(1, 125.0, 0.0, 11.0, 11.0, 103.0, 20.0, 2208, 4416)
_UHD_8COL_1BANK = ProbeGeometry(1, 70.0, 0.0, 14.0, 14.0, 6.0, 6.0, 48, 384)
_UHD_8COL_16BANK = ProbeGeometry(1, 70.0, 0.0, 14.0, 14.0, 6.0, 6.0, 768, 6144)
_NP2_SINGLE_SHANK = ProbeGeometry(1, 70.0, 0.0, 27.0, 27.0, 32.0, 15.0, 640, 1280)
_NP2_FOUR_SHANK = ProbeGeometry(4, 70.0, 250.0, 27.0, 27.0, 32.0, 15.0, 640, 1280)
_NP1120 = ProbeGeometry(1, 70.0, 0.0, 6.75, 6.75, 4.5, 4.5, 192, 384)
_NP1121 = ProbeGeometry(1, 70.0, 0.0, 6.25, 6.25, 3.0, 3.0, 384, 384)
_NP1122 = ProbeGeometry(1, 70.0, 0.0, 6.75, 6.75, 4.5, 4.5, 24, 384)
_NP1123 = ProbeGeometry(1, 70.0, 0.0, 10.25, 10.25, 3.0, 3.0, 32, 384)
_NP1300 = ProbeGeometry(1, 70.0, 0.0, 11.0, 11.0, 48.0, 20.0, 480, 960)
_NP1200_128CH = ProbeGeometry(1, 70.0, 0.0, 27.0, 11.0, 32.0, 20.0, 64, 128)
_NXT3000_128CH = ProbeGeometry(1, 70.0, 0.0, 53.0, 53.0, 0.0, 15.0, 128, 128)


# Probe part number (the value of ``imDatPrb_pn`` in the SpikeGLX
# meta file) → ``ProbeGeometry``. The literal '3A' sentinel maps the
# pre-PartNumber 3A phase probes onto the NP1.0 staggered geometry,
# since those metadata files lack ``imDatPrb_pn`` entirely.
PROBE_GEOMETRY: dict[str, ProbeGeometry] = {
    "3A": _NP1_STAG_70UM,
    "PRB_1_4_0480_1": _NP1_STAG_70UM,
    "PRB_1_4_0480_1_C": _NP1_STAG_70UM,
    "NP1010": _NP1_STAG_70UM,
    "NP1011": _NP1_STAG_70UM,
    "NP1012": _NP1_STAG_70UM,
    "NP1013": _NP1_STAG_70UM,
    "NP1015": _NHP_LIN_70UM,
    "NP1016": _NHP_LIN_70UM,
    "NP1017": _NHP_LIN_70UM,
    "NP1020": _NHP_STAG_125UM_MED,
    "NP1021": _NHP_STAG_125UM_MED,
    "NP1030": _NHP_STAG_125UM_LONG,
    "NP1031": _NHP_STAG_125UM_LONG,
    "NP1022": _NHP_LIN_125UM_MED,
    "NP1032": _NHP_LIN_125UM_LONG,
    "NP1100": _UHD_8COL_1BANK,
    "NP1110": _UHD_8COL_16BANK,
    "PRB2_1_4_0480_1": _NP2_SINGLE_SHANK,
    "PRB2_1_2_0640_0": _NP2_SINGLE_SHANK,
    "NP2000": _NP2_SINGLE_SHANK,
    "NP2003": _NP2_SINGLE_SHANK,
    "NP2004": _NP2_SINGLE_SHANK,
    "PRB2_4_2_0640_0": _NP2_FOUR_SHANK,
    "PRB2_4_4_0480_1": _NP2_FOUR_SHANK,
    "NP2010": _NP2_FOUR_SHANK,
    "NP2013": _NP2_FOUR_SHANK,
    "NP2014": _NP2_FOUR_SHANK,
    "NP1120": _NP1120,
    "NP1121": _NP1121,
    "NP1122": _NP1122,
    "NP1123": _NP1123,
    "NP1300": _NP1300,
    "NP1200": _NP1200_128CH,
    "NXT3000": _NXT3000_128CH,
}


# Mux family identifiers. Each value is the literal string SpikeGLX
# 032623+ writes into the ``muxTbl`` field of the augmented meta file.
# These are factual ADC multiplexing patterns from Imec's hardware
# documentation, expressed in the exact format the SpikeGLX metadata
# spec requires (``(n_adcs,channels_per_adc)`` header followed by one
# parenthesised group of channel indices per ADC).
_MUX_NP1 = (
    "~muxTbl=(32,12)"
    "(0 1 24 25 48 49 72 73 96 97 120 121 144 145 168 169 192 193 216 217 240 241 264 265 288 289 312 313 336 337 360 361)"
    "(2 3 26 27 50 51 74 75 98 99 122 123 146 147 170 171 194 195 218 219 242 243 266 267 290 291 314 315 338 339 362 363)"
    "(4 5 28 29 52 53 76 77 100 101 124 125 148 149 172 173 196 197 220 221 244 245 268 269 292 293 316 317 340 341 364 365)"
    "(6 7 30 31 54 55 78 79 102 103 126 127 150 151 174 175 198 199 222 223 246 247 270 271 294 295 318 319 342 343 366 367)"
    "(8 9 32 33 56 57 80 81 104 105 128 129 152 153 176 177 200 201 224 225 248 249 272 273 296 297 320 321 344 345 368 369)"
    "(10 11 34 35 58 59 82 83 106 107 130 131 154 155 178 179 202 203 226 227 250 251 274 275 298 299 322 323 346 347 370 371)"
    "(12 13 36 37 60 61 84 85 108 109 132 133 156 157 180 181 204 205 228 229 252 253 276 277 300 301 324 325 348 349 372 373)"
    "(14 15 38 39 62 63 86 87 110 111 134 135 158 159 182 183 206 207 230 231 254 255 278 279 302 303 326 327 350 351 374 375)"
    "(16 17 40 41 64 65 88 89 112 113 136 137 160 161 184 185 208 209 232 233 256 257 280 281 304 305 328 329 352 353 376 377)"
    "(18 19 42 43 66 67 90 91 114 115 138 139 162 163 186 187 210 211 234 235 258 259 282 283 306 307 330 331 354 355 378 379)"
    "(20 21 44 45 68 69 92 93 116 117 140 141 164 165 188 189 212 213 236 237 260 261 284 285 308 309 332 333 356 357 380 381)"
    "(22 23 46 47 70 71 94 95 118 119 142 143 166 167 190 191 214 215 238 239 262 263 286 287 310 311 334 335 358 359 382 383)"
)

_MUX_NP2 = (
    "~muxTbl=(24,16)"
    "(0 1 32 33 64 65 96 97 128 129 160 161 192 193 224 225 256 257 288 289 320 321 352 353)"
    "(2 3 34 35 66 67 98 99 130 131 162 163 194 195 226 227 258 259 290 291 322 323 354 355)"
    "(4 5 36 37 68 69 100 101 132 133 164 165 196 197 228 229 260 261 292 293 324 325 356 357)"
    "(6 7 38 39 70 71 102 103 134 135 166 167 198 199 230 231 262 263 294 295 326 327 358 359)"
    "(8 9 40 41 72 73 104 105 136 137 168 169 200 201 232 233 264 265 296 297 328 329 360 361)"
    "(10 11 42 43 74 75 106 107 138 139 170 171 202 203 234 235 266 267 298 299 330 331 362 363)"
    "(12 13 44 45 76 77 108 109 140 141 172 173 204 205 236 237 268 269 300 301 332 333 364 365)"
    "(14 15 46 47 78 79 110 111 142 143 174 175 206 207 238 239 270 271 302 303 334 335 366 367)"
    "(16 17 48 49 80 81 112 113 144 145 176 177 208 209 240 241 272 273 304 305 336 337 368 369)"
    "(18 19 50 51 82 83 114 115 146 147 178 179 210 211 242 243 274 275 306 307 338 339 370 371)"
    "(20 21 52 53 84 85 116 117 148 149 180 181 212 213 244 245 276 277 308 309 340 341 372 373)"
    "(22 23 54 55 86 87 118 119 150 151 182 183 214 215 246 247 278 279 310 311 342 343 374 375)"
    "(24 25 56 57 88 89 120 121 152 153 184 185 216 217 248 249 280 281 312 313 344 345 376 377)"
    "(26 27 58 59 90 91 122 123 154 155 186 187 218 219 250 251 282 283 314 315 346 347 378 379)"
    "(28 29 60 61 92 93 124 125 156 157 188 189 220 221 252 253 284 285 316 317 348 349 380 381)"
    "(30 31 62 63 94 95 126 127 158 159 190 191 222 223 254 255 286 287 318 319 350 351 382 383)"
)

# NP1110 / UHD-2 reuses the NP1 mux layout exactly; alias for clarity
# at call sites that look up by part number.
_MUX_NP1110 = _MUX_NP1

# 128-channel single-shank probes (NP1200 / NXT3000) have their own
# 12-ADC × 12-channels layout, with channel index 128 acting as a
# sentinel "unused slot" filler.
_MUX_128CH = (
    "~muxTbl=(12,12)"
    "(84 11 85 5 74 10 56 112 46 121 39 127)"
    "(100 26 110 33 69 24 63 109 45 93 25 99)"
    "(87 0 82 6 71 15 53 117 43 122 42 116)"
    "(102 28 81 34 70 18 60 103 17 94 27 101)"
    "(73 1 86 7 68 16 50 106 40 123 128 128)"
    "(105 29 75 35 67 12 54 89 20 95 128 128)"
    "(76 2 83 8 65 13 47 118 49 124 128 128)"
    "(108 30 78 36 64 14 51 90 23 96 128 128)"
    "(79 3 80 9 62 114 44 119 52 125 128 128)"
    "(104 31 72 37 61 113 57 91 19 97 128 128)"
    "(88 4 77 21 59 111 41 120 55 126 128 128)"
    "(107 32 66 38 58 115 48 92 22 98 128 128)"
)


# Probe part number → MUX table string. Same key set as
# ``PROBE_GEOMETRY`` so that any part number we accept geometrically
# also has a MUX table available for legacy-meta augmentation.
MUX_TABLE: dict[str, str] = {
    "3A": _MUX_NP1,
    "PRB_1_4_0480_1": _MUX_NP1,
    "PRB_1_4_0480_1_C": _MUX_NP1,
    "NP1010": _MUX_NP1,
    "NP1011": _MUX_NP1,
    "NP1012": _MUX_NP1,
    "NP1013": _MUX_NP1,
    "NP1015": _MUX_NP1,
    "NP1016": _MUX_NP1,
    "NP1017": _MUX_NP1,
    "NP1020": _MUX_NP1,
    "NP1021": _MUX_NP1,
    "NP1030": _MUX_NP1,
    "NP1031": _MUX_NP1,
    "NP1022": _MUX_NP1,
    "NP1032": _MUX_NP1,
    "NP1100": _MUX_NP1,
    "NP1110": _MUX_NP1110,
    "PRB2_1_4_0480_1": _MUX_NP2,
    "PRB2_1_2_0640_0": _MUX_NP2,
    "NP2000": _MUX_NP2,
    "NP2003": _MUX_NP2,
    "NP2004": _MUX_NP2,
    "PRB2_4_2_0640_0": _MUX_NP2,
    "PRB2_4_4_0480_1": _MUX_NP2,
    "NP2010": _MUX_NP2,
    "NP2013": _MUX_NP2,
    "NP2014": _MUX_NP2,
    "NP1120": _MUX_NP1,
    "NP1121": _MUX_NP1,
    "NP1122": _MUX_NP1,
    "NP1123": _MUX_NP1,
    "NP1300": _MUX_NP1,
    "NP1200": _MUX_128CH,
    "NXT3000": _MUX_128CH,
}


@dataclass(frozen=True)
class ChannelCounts:
    """
    Description
    -----------
    Number of AP (action-potential, ~30 kHz), LF (local-field, ~2.5
    kHz), and SY (sync) timepoints per sample stored in the SpikeGLX
    binary, as advertised by the ``snsApLfSy`` meta key.

    Parameters
    ----------
    ap (int)
        Count of AP channels.
    lf (int)
        Count of LF channels (NP1.0 only; NP2 and later are AP-only).
    sy (int)
        Count of SYNC channels (typically 1).
    """

    ap: int
    lf: int
    sy: int


@dataclass
class ShankCoords:
    """
    Description
    -----------
    Per-saved-channel geometry, indexed by file-order.

    The arrays are aligned: ``shank_index[i]``, ``x_um[i]``,
    ``y_um[i]``, ``connected[i]`` describe the i-th channel in the
    SpikeGLX binary's channel order (not the original electrode index
    on the probe). This is exactly what every downstream sorter
    expects.

    Parameters
    ----------
    n_shanks (int)
        Total shanks on the probe (typically read from the meta file
        header for new-style ``snsGeomMap`` files, or from
        ``PROBE_GEOMETRY`` for legacy ``snsShankMap`` files).
    shank_width_um (float)
        Width of one shank in micrometres.
    shank_pitch_um (float)
        Centre-to-centre distance between adjacent shanks in
        micrometres.
    shank_index (np.ndarray)
        ``(n_channels,)`` int array; 0-based shank index for each
        saved channel.
    x_um (np.ndarray)
        ``(n_channels,)`` float array; x position of each saved
        channel within its shank, in micrometres.
    y_um (np.ndarray)
        ``(n_channels,)`` float array; y position of each saved
        channel within its shank, in micrometres.
    connected (np.ndarray)
        ``(n_channels,)`` bool array; True iff the electrode is
        wired and not a reference/dead channel.
    """

    n_shanks: int
    shank_width_um: float
    shank_pitch_um: float
    shank_index: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    connected: np.ndarray

    @property
    def n_channels(self) -> int:
        """Number of saved channels."""

        return self.shank_index.size

    def x_global_um(self) -> np.ndarray:
        """
        Return x coordinates with the per-shank offset applied:
        ``shank_index * shank_pitch_um + x_um``. Useful for
        downstream tools that expect a single absolute x per channel
        (KS chanMap, NPY, plain text) rather than separate "x within
        shank" + "shank index" fields.
        """

        return self.shank_index * self.shank_pitch_um + self.x_um


class OutputFormat(Enum):
    """
    Description
    -----------
    Supported output formats for ``write_coords_file``. The enum
    values double as human-readable labels for the Qt format chooser.

    - ``TEXT``: tab-delimited ``index<TAB>x_um<TAB>y_um<TAB>shank``.
    - ``KILOSORT_MAT``: ``.mat`` file consumed by Kilosort 2/3/4.
    - ``JRCLUST_STRINGS``: three string assignments for a JRClust
      ``.prm`` file (``shankMap``, ``siteLoc``, ``siteMap``).
    - ``NPY``: ``(n_channels, 2)`` ``.npy`` of ``(x, y)`` for YASS and
      other Python-native sorters.
    - ``LEGACY_META_AUGMENT``: in-place rewrite of a pre-032623
      SpikeGLX ``.meta`` file with the new fields (``snsGeomMap``,
      ``muxTbl``, ``imChan0apGain``, ``imChan0lfGain``,
      ``imAnyChanFullBand``); the original is backed up alongside as
      ``<base>_orig.meta``.
    """

    TEXT = "text"
    KILOSORT_MAT = "kilosort_mat"
    JRCLUST_STRINGS = "jrclust_strings"
    NPY = "npy"
    LEGACY_META_AUGMENT = "legacy_meta_augment"


def parse_spikeglx_meta(meta_path: Path) -> dict[str, str]:
    """
    Description
    -----------
    Parse a SpikeGLX ``.meta`` text file into a dict of {key: value}
    strings, stripping any leading ``~`` on mutable-keys (e.g.
    ``~snsShankMap=...`` becomes ``snsShankMap``) so downstream
    lookups don't need to know which keys are mutable.

    Parameters
    ----------
    meta_path (Path)
        Absolute path to a ``*.meta`` file written by SpikeGLX.

    Returns
    -------
    meta (dict[str, str])
        Key → value mapping; values are kept as raw strings. Numeric
        fields are parsed at the point of use, not here, so that
        format-specific (e.g. ``snsGeomMap``) sub-parsing happens in
        one place per format.

    Raises
    ------
    FileNotFoundError
        If ``meta_path`` does not exist.
    """

    if not meta_path.exists():
        msg = f"meta file does not exist: {meta_path}"
        raise FileNotFoundError(msg)

    meta: dict[str, str] = {}
    with meta_path.open() as fh:
        for raw_line in fh.read().splitlines():
            if "=" not in raw_line:
                continue
            key, _, value = raw_line.partition("=")
            key = key.lstrip("~")
            meta[key] = value
    return meta


def channel_counts(meta: dict[str, str]) -> ChannelCounts:
    """
    Description
    -----------
    Read the ``snsApLfSy`` meta key (a comma-separated triple of
    channel counts: ``ap,lf,sy``) and return it as a structured
    ``ChannelCounts``.

    Parameters
    ----------
    meta (dict[str, str])
        Parsed SpikeGLX meta dict from ``parse_spikeglx_meta``.

    Returns
    -------
    counts (ChannelCounts)
        Triple of (ap, lf, sy) channel counts.
    """

    ap_str, lf_str, sy_str = meta["snsApLfSy"].split(",")
    return ChannelCounts(ap=int(ap_str), lf=int(lf_str), sy=int(sy_str))


def _probe_part_number(meta: dict[str, str]) -> str:
    """
    Return the probe part number from ``imDatPrb_pn``, or the literal
    string ``'3A'`` if the key is absent (which identifies a pre-part-
    number 3A-phase probe).
    """

    if "imDatPrb_pn" in meta:
        return meta["imDatPrb_pn"]
    return "3A"


def _lookup_geometry(meta: dict[str, str]) -> ProbeGeometry:
    """
    Resolve ``meta`` to a ``ProbeGeometry`` from ``PROBE_GEOMETRY``,
    raising ``ValueError`` with a clear message if the probe part
    number isn't in our table.
    """

    pn = _probe_part_number(meta)
    if pn not in PROBE_GEOMETRY:
        msg = (
            f"unsupported Imec probe part number {pn!r}; known part "
            f"numbers: {sorted(PROBE_GEOMETRY)}"
        )
        raise ValueError(msg)
    return PROBE_GEOMETRY[pn]


def parse_geom_map(meta: dict[str, str]) -> ShankCoords:
    """
    Description
    -----------
    Decode the ``snsGeomMap`` key (present in SpikeGLX 032623+ meta
    files) into a ``ShankCoords``.

    Format of ``snsGeomMap``::

        snsGeomMap=(<pn>,<n_shanks>,<shank_pitch_um>,<shank_width_um>)
                   (<shank>:<x_um>:<y_um>:<connected>)
                   (<shank>:<x_um>:<y_um>:<connected>)
                   ...

    One trailing group per saved channel, in file order.

    Parameters
    ----------
    meta (dict[str, str])
        Parsed SpikeGLX meta dict containing a ``snsGeomMap`` entry.

    Returns
    -------
    coords (ShankCoords)
        Per-channel geometry.
    """

    parts = meta["snsGeomMap"].split(")")
    header = parts[0].lstrip("(")
    header_fields = header.split(",")
    n_shanks = int(header_fields[1])
    shank_pitch_um = float(header_fields[2])
    shank_width_um = float(header_fields[3])

    # parts[0] is the header; everything after it is one per-channel
    # group. A well-formed snsGeomMap value ends with ')', so the final
    # split fragment is the empty string -- but slicing off parts[-1]
    # unconditionally would silently drop a real channel if the value is
    # truncated and does NOT end with ')'. Instead, drop the header and
    # filter out only genuinely empty fragments (the trailing '' and any
    # stray whitespace between groups), keeping every channel that carries
    # content. Assert the value was '(...)'-terminated so a malformed
    # (truncated) map fails loudly here rather than producing a geometry
    # that is silently one channel short.
    if not meta["snsGeomMap"].rstrip().endswith(")"):
        msg = (
            "snsGeomMap value is not ')'-terminated; the channel list is "
            "likely truncated and would otherwise be parsed one channel short"
        )
        raise ValueError(msg)
    channel_entries = [frag for frag in parts[1:] if frag.strip() != ""]
    n_channels = len(channel_entries)

    shank_index = np.zeros(n_channels, dtype=np.int64)
    x_um = np.zeros(n_channels, dtype=np.float64)
    y_um = np.zeros(n_channels, dtype=np.float64)
    connected = np.zeros(n_channels, dtype=bool)
    for i, entry in enumerate(channel_entries):
        shank_str, x_str, y_str, conn_str = entry.lstrip("(").split(":")
        shank_index[i] = int(shank_str)
        x_um[i] = float(x_str)
        y_um[i] = float(y_str)
        connected[i] = int(conn_str) == 1

    return ShankCoords(
        n_shanks=n_shanks,
        shank_width_um=shank_width_um,
        shank_pitch_um=shank_pitch_um,
        shank_index=shank_index,
        x_um=x_um,
        y_um=y_um,
        connected=connected,
    )


def parse_shank_map(meta: dict[str, str]) -> ShankCoords:
    """
    Description
    -----------
    Decode the legacy ``snsShankMap`` key (pre-SpikeGLX 032623) into a
    ``ShankCoords``, using ``PROBE_GEOMETRY`` to translate (column,
    row) electrode indices into (x, y) micrometres.

    Format of ``snsShankMap``::

        snsShankMap=(<n_shanks>,<columns>,<rows>)
                    (<shank>:<col>:<row>:<connected>)
                    (<shank>:<col>:<row>:<connected>)
                    ...

    Only the per-channel entries are consumed here; the (col, row)
    indices are converted to micrometres using::

        x = col * horizontal_pitch + offset_for_row
        y = row * vertical_pitch

    where ``offset_for_row`` alternates between
    ``x_offset_even_um`` and ``x_offset_odd_um`` depending on the row
    parity (staggered NP1.0 layout) or is constant on linear probes.

    Parameters
    ----------
    meta (dict[str, str])
        Parsed SpikeGLX meta dict containing a ``snsShankMap`` entry.

    Returns
    -------
    coords (ShankCoords)
        Per-channel geometry; the count of channels equals the AP
        count from ``snsApLfSy`` (``snsShankMap`` can include a
        trailing SYNC entry, which we exclude by anchoring on the AP
        count).
    """

    counts = channel_counts(meta)
    geometry = _lookup_geometry(meta)

    parts = meta["snsShankMap"].split(")")
    channel_entries = parts[1 : 1 + counts.ap]

    shank_index = np.zeros(counts.ap, dtype=np.int64)
    col_index = np.zeros(counts.ap, dtype=np.int64)
    row_index = np.zeros(counts.ap, dtype=np.int64)
    connected = np.zeros(counts.ap, dtype=bool)
    for i, entry in enumerate(channel_entries):
        shank_str, col_str, row_str, conn_str = entry.lstrip("(").split(":")
        shank_index[i] = int(shank_str)
        col_index[i] = int(col_str)
        row_index[i] = int(row_str)
        connected[i] = int(conn_str) == 1

    even_rows = (row_index % 2) == 0
    x_um = col_index * geometry.horizontal_pitch_um
    x_um = np.where(
        even_rows,
        x_um + geometry.x_offset_even_um,
        x_um + geometry.x_offset_odd_um,
    )
    y_um = row_index * geometry.vertical_pitch_um

    return ShankCoords(
        n_shanks=geometry.n_shanks,
        shank_width_um=geometry.shank_width_um,
        shank_pitch_um=geometry.shank_pitch_um,
        shank_index=shank_index,
        x_um=x_um.astype(np.float64),
        y_um=y_um.astype(np.float64),
        connected=connected,
    )


def coords_from_meta(meta: dict[str, str]) -> ShankCoords:
    """
    Description
    -----------
    Top-level coordinate resolver: dispatch to ``parse_geom_map`` if
    the new-style ``snsGeomMap`` key is present, otherwise fall back
    to ``parse_shank_map``. Callers should prefer this over the
    per-format parsers so they don't have to know which SpikeGLX
    vintage produced the metadata file.

    Parameters
    ----------
    meta (dict[str, str])
        Parsed SpikeGLX meta dict.

    Returns
    -------
    coords (ShankCoords)
        Per-channel geometry.

    Raises
    ------
    KeyError
        If neither ``snsGeomMap`` nor ``snsShankMap`` is present in
        the meta dict (i.e. the file is too corrupt or too old to
        place channels on the probe).
    """

    if "snsGeomMap" in meta:
        return parse_geom_map(meta)
    if "snsShankMap" in meta:
        return parse_shank_map(meta)
    msg = (
        "meta file has neither 'snsGeomMap' nor 'snsShankMap'; cannot "
        "derive probe geometry"
    )
    raise KeyError(msg)


def apply_bad_channels(coords: ShankCoords, bad_channels: np.ndarray) -> ShankCoords:
    """
    Description
    -----------
    Mark a set of saved channels as not-connected by clearing their
    ``connected`` flag. Channel indices outside ``[0, n_channels)``
    are silently clipped (e.g. a Kilosort-helper bad-channel list
    that contains the SYNC channel slot).

    Parameters
    ----------
    coords (ShankCoords)
        Source per-channel geometry; not mutated.
    bad_channels (np.ndarray)
        1-D int array of file-order channel indices to clear.

    Returns
    -------
    coords (ShankCoords)
        New ``ShankCoords`` instance with the supplied channels
        marked disconnected.
    """

    bad = bad_channels[(bad_channels >= 0) & (bad_channels < coords.n_channels)]
    connected = coords.connected.copy()
    connected[bad] = False
    return ShankCoords(
        n_shanks=coords.n_shanks,
        shank_width_um=coords.shank_width_um,
        shank_pitch_um=coords.shank_pitch_um,
        shank_index=coords.shank_index.copy(),
        x_um=coords.x_um.copy(),
        y_um=coords.y_um.copy(),
        connected=connected,
    )


def write_text_coords(coords: ShankCoords, dst: Path) -> None:
    """
    Description
    -----------
    Write a plain tab-delimited coordinate file at ``dst``. One row
    per saved channel, columns:
    ``channel_index<TAB>x_um<TAB>y_um<TAB>shank_index`` (no header).

    The ``channel_index`` written is the 0-based position in the
    SpikeGLX binary's channel order, not the original electrode index
    on the probe.

    Parameters
    ----------
    coords (ShankCoords)
        Per-channel geometry.
    dst (Path)
        Output file path.
    """

    x_global = coords.x_global_um()
    with dst.open("w") as fh:
        for i in range(coords.n_channels):
            fh.write(
                f"{i}\t{x_global[i]:g}\t{coords.y_um[i]:g}\t{coords.shank_index[i]:g}\n"
            )


def write_npy_coords(coords: ShankCoords, dst: Path) -> None:
    """
    Description
    -----------
    Write a ``(n_channels, 2)`` ``.npy`` array of ``(x_um, y_um)``
    coordinates at ``dst``. The x coordinate already has the per-shank
    offset applied so that downstream tools see a single absolute x.

    Parameters
    ----------
    coords (ShankCoords)
        Per-channel geometry.
    dst (Path)
        Output ``.npy`` path.
    """

    geometry_xy = np.column_stack([coords.x_global_um(), coords.y_um])
    np.save(dst, geometry_xy)


def write_jrclust_coords(coords: ShankCoords, dst: Path) -> None:
    """
    Description
    -----------
    Write three string assignments to ``dst`` in the order JRClust's
    ``.prm`` file expects:

    1. ``shankMap = [s1,s2,...,sN];``        -- 1-based shank index
       per channel.
    2. ``siteLoc = [x1,y1;x2,y2;...;xN,yN];`` -- (x, y) per channel,
       absolute (with shank offset applied), in micrometres.
    3. ``siteMap = [1,2,...,N];``            -- file-order channel
       index (1-based for MATLAB).

    JRClust uses 1-based indexing throughout (it's MATLAB-native), so
    both ``shankMap`` and ``siteMap`` are emitted 1-based.

    Parameters
    ----------
    coords (ShankCoords)
        Per-channel geometry.
    dst (Path)
        Output text-file path.
    """

    n = coords.n_channels
    shank_one_based = coords.shank_index + 1
    x_global = coords.x_global_um()

    shank_map = ",".join(f"{s:g}" for s in shank_one_based)
    site_loc = ";".join(f"{x_global[i]:g},{coords.y_um[i]:g}" for i in range(n))
    site_map = ",".join(str(i + 1) for i in range(n))

    with dst.open("w") as fh:
        fh.write(f"shankMap = [{shank_map}];\n")
        fh.write(f"siteLoc = [{site_loc}];\n")
        fh.write(f"siteMap = [{site_map}];\n")


def write_kilosort_chanmap(
    coords: ShankCoords, dst: Path, *, name: str
) -> None:
    """
    Description
    -----------
    Write a Kilosort-compatible ``chanMap.mat`` at ``dst``. The file
    is consumed verbatim by Kilosort 2.0 / 2.5 / 3.0 / 4.0 and by
    Phy's data-loader.

    The MATLAB-side fields (and their KS-specific conventions):

    - ``chanMap``     : (n, 1) double; 1-based file-order channel
                        indices.
    - ``chanMap0ind`` : (n, 1) double; same, 0-based.
    - ``connected``   : (n, 1) logical; True if the channel is
                        electrode-backed (not reference / not in the
                        bad-channel list).
    - ``name``        : MATLAB string scalar; used by KS for log
                        messages; we set it to the meta file's
                        basename.
    - ``xcoords``     : (n, 1) double; absolute x in micrometres.
    - ``ycoords``     : (n, 1) double; absolute y in micrometres.
    - ``kcoords``     : (n, 1) double; 1-based shank index; KS uses
                        ``kcoords`` to define spike-sorting "groups"
                        on multishank probes.

    Parameters
    ----------
    coords (ShankCoords)
        Per-channel geometry.
    dst (Path)
        Output ``.mat`` path.
    name (str)
        Value stored under the MATLAB ``name`` field (typically the
        meta file basename).
    """

    n = coords.n_channels
    chan_map_zero = np.arange(n, dtype=np.float64).reshape(n, 1)
    chan_map_one = chan_map_zero + 1.0
    x_global = coords.x_global_um().astype(np.float64).reshape(n, 1)
    y = coords.y_um.astype(np.float64).reshape(n, 1)
    kcoords = (coords.shank_index + 1).astype(np.float64).reshape(n, 1)
    connected_col = coords.connected.astype(bool).reshape(n, 1)

    sio.savemat(
        str(dst),
        {
            "chanMap": chan_map_one,
            "chanMap0ind": chan_map_zero,
            "connected": connected_col,
            "name": name,
            "xcoords": x_global,
            "ycoords": y,
            "kcoords": kcoords,
        },
    )


def _imro_derived_meta_fields(meta: dict[str, str]) -> tuple[str, str, str]:
    """
    Parse the ``imroTbl`` field and synthesize the three fields that
    SpikeGLX 032623+ writes about gain / filter state:
    ``imChan0apGain``, ``imChan0lfGain``, ``imAnyChanFullBand``.

    NP2.0 / NP2.0-4S (probe types ``21`` / ``24``) have a fixed gain
    of 80 and no LF channel; ``imAnyChanFullBand`` is always ``true``
    on those since they expose only the AP-band data.

    NP1.0 (probe type ``0``, or 3A which behaves as type ``0``) has
    per-channel gain and an optional AP-filter bypass; we walk the
    table to detect any channel with the filter disabled.

    NP1110 (probe type ``1110``) puts the gain and filter info in the
    imro table header rather than per channel.
    """

    table = meta["imroTbl"].split(")")
    header_fields = table[0].lstrip("(").split(",")
    probe_type = int(header_fields[0])
    if probe_type > 50_000:
        # 3A-phase probes encode their part number in the imro header
        # but behave like NP1.0 type 0 for gain / filter purposes.
        probe_type = 0

    if probe_type in (21, 24):
        return (
            "imChan0apGain=80\n",
            "imChan0lfGain=80\n",
            "imAnyChanFullBand=true\n",
        )

    if probe_type == 1110:
        ap_gain = header_fields[3]
        lf_gain = header_fields[4]
        any_full = "true" if header_fields[5] == "0" else "false"
        return (
            f"imChan0apGain={ap_gain}\n",
            f"imChan0lfGain={lf_gain}\n",
            f"imAnyChanFullBand={any_full}\n",
        )

    first_entry = table[1].lstrip("(").split(" ")
    ap_gain = first_entry[3]
    lf_gain = first_entry[4]

    any_full = "false"
    if len(first_entry) == 6:
        for raw_entry in table[1:-1]:
            entry_fields = raw_entry.lstrip("(").split(" ")
            if entry_fields[5] == "0":
                any_full = "true"
                break

    return (
        f"imChan0apGain={ap_gain}\n",
        f"imChan0lfGain={lf_gain}\n",
        f"imAnyChanFullBand={any_full}\n",
    )


def _sns_geom_map_string(meta: dict[str, str], coords: ShankCoords) -> str:
    """
    Build a SpikeGLX 032623+ ``snsGeomMap=(...)``-style line from the
    parsed per-channel geometry, using the probe part number from
    ``meta`` for the header. Trailing newline included so the line
    is ready to append to a meta file.
    """

    pn = _probe_part_number(meta)
    header = (
        f"~snsGeomMap=({pn},{coords.n_shanks:d},"
        f"{coords.shank_pitch_um:g},{coords.shank_width_um:g})"
    )
    entries = "".join(
        f"({coords.shank_index[i]:g}:{coords.x_um[i]:g}:"
        f"{coords.y_um[i]:g}:{int(coords.connected[i]):g})"
        for i in range(coords.n_channels)
    )
    return f"{header}{entries}\n"


def augment_legacy_meta(
    meta_path: Path, coords: ShankCoords, *, backup_suffix: str = "_orig"
) -> Path:
    """
    Description
    -----------
    Rewrite an old (pre-SpikeGLX 032623) meta file in place, appending
    the fields newer SpikeGLX versions write automatically:
    ``imChan0apGain``, ``imChan0lfGain``, ``imAnyChanFullBand``,
    ``muxTbl``, ``snsGeomMap``. The original is preserved alongside
    as ``<base><backup_suffix>.meta``.

    No-op if the meta already has the ``imChan0apGain`` field (i.e.
    it was already written by a 032623+ build).

    Parameters
    ----------
    meta_path (Path)
        Path to the meta file to augment. Must exist; must be writable.
    coords (ShankCoords)
        Per-channel geometry to encode into the new ``snsGeomMap``.
    backup_suffix (str)
        Suffix appended to the original file basename for the backup.

    Returns
    -------
    augmented_meta_path (Path)
        Path to the augmented meta file (same as ``meta_path``).

    Raises
    ------
    FileNotFoundError
        If ``meta_path`` does not exist.
    ValueError
        If the probe part number isn't in our ``MUX_TABLE``.
    """

    meta = parse_spikeglx_meta(meta_path)
    if "imChan0apGain" in meta:
        logger.info(
            "%s already has imChan0apGain; nothing to augment", meta_path.name
        )
        return meta_path

    pn = _probe_part_number(meta)
    if pn not in MUX_TABLE:
        msg = (
            f"unsupported probe part number {pn!r} for legacy-meta "
            f"augmentation; cannot synthesise muxTbl"
        )
        raise ValueError(msg)

    backup_path = meta_path.with_name(f"{meta_path.stem}{backup_suffix}.meta")
    shutil.copy2(meta_path, backup_path)

    ap_gain_line, lf_gain_line, any_full_line = _imro_derived_meta_fields(meta)
    mux_line = MUX_TABLE[pn] + "\n"
    geom_line = _sns_geom_map_string(meta, coords)

    with meta_path.open("a") as fh:
        fh.write(ap_gain_line)
        fh.write(lf_gain_line)
        fh.write(any_full_line)
        fh.write(mux_line)
        fh.write(geom_line)
    return meta_path


# Dispatch table: ``OutputFormat`` → writer callable. Each callable
# takes ``(meta, coords, dst, base_name)`` and returns the actual
# destination path written. ``meta`` and ``base_name`` are only
# consulted by formats that need them (KS and legacy-meta).
def _write_text(
    meta: dict[str, str], coords: ShankCoords, dst: Path, base_name: str
) -> Path:
    write_text_coords(coords, dst)
    return dst


def _write_npy(
    meta: dict[str, str], coords: ShankCoords, dst: Path, base_name: str
) -> Path:
    write_npy_coords(coords, dst)
    return dst


def _write_jrclust(
    meta: dict[str, str], coords: ShankCoords, dst: Path, base_name: str
) -> Path:
    write_jrclust_coords(coords, dst)
    return dst


def _write_kilosort(
    meta: dict[str, str], coords: ShankCoords, dst: Path, base_name: str
) -> Path:
    write_kilosort_chanmap(coords, dst, name=base_name)
    return dst


def _write_legacy_meta(
    meta: dict[str, str], coords: ShankCoords, dst: Path, base_name: str
) -> Path:
    # ``dst`` for the legacy-meta path is the meta file itself; the
    # writer mutates it in place and creates a backup beside it.
    return augment_legacy_meta(dst, coords)


_DEFAULT_SUFFIXES: dict[OutputFormat, str] = {
    OutputFormat.TEXT: "_siteCoords.txt",
    OutputFormat.KILOSORT_MAT: "_kilosortChanMap.mat",
    OutputFormat.JRCLUST_STRINGS: "_forJRCprm.txt",
    OutputFormat.NPY: "_siteCoords.npy",
    OutputFormat.LEGACY_META_AUGMENT: ".meta",
}

_WRITERS: dict[
    OutputFormat,
    Callable[[dict[str, str], ShankCoords, Path, str], Path],
] = {
    OutputFormat.TEXT: _write_text,
    OutputFormat.KILOSORT_MAT: _write_kilosort,
    OutputFormat.JRCLUST_STRINGS: _write_jrclust,
    OutputFormat.NPY: _write_npy,
    OutputFormat.LEGACY_META_AUGMENT: _write_legacy_meta,
}


def write_coords_file(
    *,
    meta: dict[str, str],
    coords: ShankCoords,
    output_format: OutputFormat,
    save_dir: Path,
    base_name: str,
) -> Path:
    """
    Description
    -----------
    Dispatch to the correct writer for ``output_format`` and return
    the path actually written.

    The destination path is derived as
    ``save_dir / (base_name + _DEFAULT_SUFFIXES[output_format])``,
    except for ``LEGACY_META_AUGMENT`` whose suffix is just
    ``.meta`` (the writer expects to mutate the original file in
    place and create a sibling backup).

    Parameters
    ----------
    meta (dict[str, str])
        Parsed SpikeGLX meta dict.
    coords (ShankCoords)
        Per-channel geometry.
    output_format (OutputFormat)
        Which downstream artefact to produce.
    save_dir (Path)
        Directory to write into; must exist.
    base_name (str)
        Stem for the produced filename (typically
        ``meta_path.stem``).

    Returns
    -------
    written_path (Path)
        Absolute path of the file we just wrote.
    """

    suffix = _DEFAULT_SUFFIXES[output_format]
    dst = save_dir / f"{base_name}{suffix}"
    writer = _WRITERS[output_format]
    return writer(meta, coords, dst, base_name)


def plot_coords(
    coords: ShankCoords, meta: dict[str, str], *, figsize: tuple[float, float] = (2.0, 12.0)
) -> plt.Figure:
    """
    Description
    -----------
    Render a simple visualization of the probe with all electrodes
    drawn as small empty squares and the saved channels overlaid as
    filled blue squares. For multi-shank probes, each shank is drawn
    at its physical x-offset (``shank_index * shank_pitch_um``).

    Useful as a sanity check that the meta file's saved channels are
    where we think they are on the probe.

    Parameters
    ----------
    coords (ShankCoords)
        Per-channel geometry.
    meta (dict[str, str])
        Parsed SpikeGLX meta dict (used to look up the full electrode
        layout for the background "all electrodes" markers).
    figsize (tuple[float, float])
        Matplotlib figure size in inches.

    Returns
    -------
    fig (matplotlib.figure.Figure)
        The created figure; caller can save / display as they like.
    """

    geometry = _lookup_geometry(meta)

    # Build the full per-shank electrode layout. Each shank's
    # electrode count is ``electrodes_per_shank``; row index runs
    # ``[0, rows_per_shank)``; column index runs
    # ``[0, columns_per_shank)``.
    electrode_idx = np.arange(geometry.electrodes_per_shank)
    row_idx = electrode_idx // geometry.columns_per_shank
    col_idx = electrode_idx % geometry.columns_per_shank
    even_rows = (row_idx % 2) == 0
    background_x = col_idx * geometry.horizontal_pitch_um
    background_x = np.where(
        even_rows,
        background_x + geometry.x_offset_even_um,
        background_x + geometry.x_offset_odd_um,
    )
    background_y = row_idx * geometry.vertical_pitch_um

    fig = plt.figure(figsize=figsize)
    for shank in range(geometry.n_shanks):
        x_offset = shank * geometry.shank_pitch_um
        plt.scatter(
            background_x + x_offset,
            background_y,
            c="#FFFFFF",
            edgecolor="#000000",
            marker="s",
            s=5,
            linewidths=0.4,
            zorder=1,
        )
        on_this_shank = coords.shank_index == shank
        plt.scatter(
            coords.x_um[on_this_shank] + x_offset,
            coords.y_um[on_this_shank],
            c="#1F77B4",
            edgecolor="#0B3D6E",
            marker="s",
            s=15,
            linewidths=0.4,
            zorder=2,
        )
    plt.xlabel("x ($\\mu$m)")
    plt.ylabel("y ($\\mu$m)")
    return fig


def _ensure_qt_app() -> QApplication:
    """
    Return the running ``QApplication`` instance, creating one if
    none exists. The standalone GUI entry point uses this so that the
    script also works when invoked from a parent process that already
    started a Qt event loop (e.g. the main usv_playpen GUI).
    """

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _convert_meta(
    meta_path: Path, output_format: OutputFormat
) -> tuple[Path, ShankCoords, dict[str, str]]:
    """
    Description
    -----------
    Parse one SpikeGLX meta file, derive its per-channel geometry, and
    write the requested artefact next to the meta file. Shared by the
    interactive GUI and the headless command-line paths so both produce
    byte-identical output.

    Parameters
    ----------
    meta_path (Path)
        Path to the SpikeGLX ``*.ap.meta`` file to convert.
    output_format (OutputFormat)
        Which artefact to emit (Kilosort chanMap, text / NumPy site
        coordinates, JRClust strings, or a legacy-meta in-place upgrade).

    Returns
    -------
    written (Path)
        The path of the artefact that was written.
    coords (ShankCoords)
        The derived per-channel coordinates (returned for plotting).
    meta (dict[str, str])
        The parsed meta key/value mapping (returned for plotting).
    """

    meta = parse_spikeglx_meta(meta_path)
    coords = coords_from_meta(meta)
    if output_format is OutputFormat.LEGACY_META_AUGMENT:
        written = augment_legacy_meta(meta_path, coords)
    else:
        written = write_coords_file(
            meta=meta,
            coords=coords,
            output_format=output_format,
            save_dir=meta_path.parent,
            base_name=meta_path.stem,
        )
    return written, coords, meta


def _run_gui() -> int:
    """
    Description
    -----------
    Interactive Qt front-end for :func:`main`. Three modal dialogs run in
    sequence:

    1. ``QFileDialog`` to pick the meta file.
    2. ``QInputDialog`` to pick the output format (defaults to
       Kilosort, the most common case).
    3. ``QMessageBox`` to confirm the destination path and
       optionally show the probe layout plot.

    Cancelling any dialog returns 0 (clean exit). Errors during the
    conversion are surfaced as a ``QMessageBox.critical`` rather than
    propagating an uncaught exception.

    Returns
    -------
    exit_code (int)
        ``0`` on success or clean cancellation; ``1`` on error
        (caught and surfaced to the user as a dialog).
    """

    app = _ensure_qt_app()

    meta_path_str, _ = QFileDialog.getOpenFileName(
        None,
        "Select SpikeGLX metadata file",
        "",
        "SpikeGLX metadata (*.meta);;All files (*)",
    )
    if not meta_path_str:
        return 0
    meta_path = Path(meta_path_str)

    format_labels = [fmt.value for fmt in OutputFormat]
    default_index = format_labels.index(OutputFormat.KILOSORT_MAT.value)
    selected_label, ok = QInputDialog.getItem(
        None,
        "Output format",
        f"Choose the output format for {meta_path.name}:",
        format_labels,
        default_index,
        False,
    )
    if not ok:
        return 0
    output_format = OutputFormat(selected_label)

    try:
        written, coords, meta = _convert_meta(meta_path, output_format)
        logger.info("wrote %s", written)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        QMessageBox.critical(None, "Conversion failed", str(exc))
        return 1

    plot_button = QMessageBox.StandardButton.Yes
    skip_button = QMessageBox.StandardButton.No
    answer = QMessageBox.question(
        None,
        "Saved",
        f"Wrote:\n{written}\n\nShow probe layout for sanity check?",
        plot_button | skip_button,
        skip_button,
    )
    if answer == plot_button:
        fig = plot_coords(coords, meta)
        plt.show()
        plt.close(fig)

    _ = app
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Description
    -----------
    Convert one SpikeGLX meta file into a per-channel geometry artefact,
    either headlessly from command-line flags or, when none are given,
    through the interactive Qt GUI (:func:`_run_gui`).

    Headless mode is triggered by passing ``--meta-file``: the whole
    conversion then runs without Qt, so it can be scripted or run on a
    cluster next to the spike-sorting step. ``--output-format`` selects
    the artefact (defaults to the Kilosort chanMap); the probe layout can
    be shown with ``--plot`` or written to a file with ``--save-plot`` for
    a headless sanity check. With no ``--meta-file`` the Qt dialogs run.

    Parameters
    ----------
    argv (list[str] | None)
        Command-line arguments (defaults to ``sys.argv[1:]`` when None).

    Returns
    -------
    exit_code (int)
        ``0`` on success or clean cancellation; ``1`` on a conversion
        error (surfaced to stderr in headless mode, or as a
        ``QMessageBox.critical`` dialog in GUI mode).
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    format_labels = [fmt.value for fmt in OutputFormat]
    parser = argparse.ArgumentParser(
        prog="npx-meta-to-coords",
        description=(
            "Convert a SpikeGLX *.ap.meta file into a per-channel geometry "
            "artefact (Kilosort chanMap, site coordinates, JRClust strings, or "
            "a legacy-meta upgrade). Pass --meta-file to run headlessly; with no "
            "arguments an interactive Qt GUI is launched instead."
        ),
    )
    parser.add_argument(
        "--meta-file", default=None,
        help="Path to the SpikeGLX *.ap.meta file. When given, runs headlessly (no GUI).",
    )
    parser.add_argument(
        "--output-format", default=OutputFormat.KILOSORT_MAT.value, choices=format_labels,
        help="Output artefact format for headless mode (default: kilosort_mat).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="After a headless conversion, show the probe-layout plot interactively.",
    )
    parser.add_argument(
        "--save-plot", default=None,
        help="After a headless conversion, write the probe-layout plot to this path (no display needed).",
    )
    args = parser.parse_args(argv)

    # No --meta-file -> fall back to the interactive Qt GUI.
    if args.meta_file is None:
        return _run_gui()

    # Headless path: convert straight from the flags, no Qt.
    meta_path = Path(args.meta_file)
    output_format = OutputFormat(args.output_format)
    try:
        written, coords, meta = _convert_meta(meta_path, output_format)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Conversion failed: %s", exc)
        return 1
    logger.info("wrote %s", written)

    if args.save_plot is not None:
        fig = plot_coords(coords, meta)
        fig.savefig(args.save_plot, bbox_inches="tight")
        plt.close(fig)
        logger.info("saved probe-layout plot to %s", args.save_plot)
    elif args.plot:
        fig = plot_coords(coords, meta)
        plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
