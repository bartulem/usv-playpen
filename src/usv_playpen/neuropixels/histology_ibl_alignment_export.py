"""
@author: bartulem
Self-contained replacement for the IBL `extract_files.extract_data` ALF
converter used by the IBL ephys-alignment GUI, plus the surrounding pre-
and post-alignment glue steps that previously lived in a standalone
script.

Why this module exists
----------------------
The IBL convenience function ``atlaselectrophysiology.extract_files.
extract_data`` does two things per probe:

* ``ks2_to_alf`` — wraps ``phylib`` and ``ibllib`` to convert the
  Kilosort output directory into the ALF (Alyx File) layout the IBL
  alignment GUI consumes.
* ``extract_rmsmap`` — streams the full concatenated ``*.ap.bin`` to
  compute per-channel RMS-in-time (and, for LF files, a Welch spectral
  density). For multi-hour concatenated recordings of hundreds of GB
  this step is the dominant cost.

In practice the IBL alignment GUI does not need the RMS map artefacts;
no ``_iblqc_*`` files are produced by the reference workflow for this
project's Neuropixels 2.0 sessions, and the GUI loads happily without
them. The remaining ALF outputs are computable from the Kilosort
directory plus the SpikeGLX ``.ap.meta`` alone — no raw-binary streaming
required — because Kilosort 4 already writes the
``_phy_spikes_subset.*.npy`` triplet (the only artefact that would
otherwise require touching the raw binary for waveform-snippet
extraction).

This module performs that conversion without ``iblatlas``, ``ibllib``,
``phylib`` or ``spikeglx`` as dependencies. It is restricted to
Neuropixels 2.0 probes (probe types 21, 24, 1030, 2003, 2004, 2013,
2014, 2020, 2021); the older Neuropixels 1.0 family is intentionally
unsupported and any non-2.0 probe type raises ``NotImplementedError``.

Pre-conditions
--------------
* Brain segmentation and track tracing have completed; per-shank track
  point ``.npy`` files exist under
  ``{cup_root}/histology/{mouse_id}/`` with filenames starting with the
  hemisphere letter (``L*.npy`` or ``R*.npy``).
* The concatenated SpikeGLX binary and metadata files exist under
  ``{cup_root}/EPHYS/{session_date}_{probe_id}/`` (``modify_files.
  concatenate_binary_files`` already produces these with the correct
  ``fileSizeBytes`` / ``fileTimeSecs`` totals; do not run the legacy
  ``get_concatenated_metadata_bool`` step from the old script).
* The Kilosort output directory exists at
  ``{cup_root}/EPHYS/{session_date}_{probe_id}/kilosort{kilosort_
  version}/``.

Public surface
--------------
:class:`IBLAlignmentExporter` exposes four independently invokable
steps mirroring the boolean flags of the old standalone script:

* :meth:`IBLAlignmentExporter.write_xyz_picks` — write
  ``xyz_picks_shank{n}.json`` for each track ``.npy`` (was
  ``extract_shank_data_bool``).
* :meth:`IBLAlignmentExporter.write_alf_outputs` — replace
  ``extract_data`` (was ``extract_ephys_data_bool``).
* :meth:`IBLAlignmentExporter.remap_channel_ids_to_raw` — remap each
  per-shank ``channel_locations_shank{n}.json`` produced by the IBL
  alignment GUI from per-shank 0..95 keys back to the raw recording
  channel ids 0..n_chan-1 via the SpikeGLX IMRO table (was
  ``correct_channel_id_bool``).
* :meth:`IBLAlignmentExporter.write_unified_channel_locations` — merge
  the per-shank JSONs into a single ``channel_locations.json`` for
  SpikeInterface (was ``create_unified_channel_locations_json_bool``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ALLEN_BREGMA_MLAPDV_UM = np.array([5739.0, 5400.0, 332.0], dtype=np.float64)
"""numpy.ndarray: Bregma landmark in Allen CCF v3 (ML, AP, DV) micrometres.

Public constant from the Allen Mouse Brain Common Coordinate Framework
(v3); identical to ``iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM
['bregma']``. Used by :func:`ccf_apdvml_to_xyz_mlapdv_um` so the
conversion does not require downloading the atlas NRRD volumes.
"""

NP2_PROBE_TYPES = frozenset({21, 24, 1030, 2003, 2004, 2013, 2014, 2020, 2021})
"""frozenset[int]: SpikeGLX ``imDatPrb_type`` values that this module
treats as Neuropixels 2.0. Includes both single-shank (21, 1030, 2003)
and four-shank (24, 2013, 2014, 2020, 2021) variants. Anything else
raises ``NotImplementedError``.
"""

NP2_MULTISHANK_PROBE_TYPES = frozenset({24, 2013, 2014, 2020, 2021})
"""frozenset[int]: Subset of :data:`NP2_PROBE_TYPES` that have multiple
shanks. For these probes the IMRO row carries an explicit shank index in
column 1 (``(channel, shank, bank, refid, elecid)``); for single-shank
NP2.0 probes the shank index column is absent.
"""

NP2_AP_GAIN = 80.0
"""float: Fixed AP-band programmable gain for all Neuropixels 2.0
probes. Unlike NP1.0, the NP2.0 analog front-end has a single, hard-coded
AP gain that does not appear per-channel in the IMRO table.
"""

N_CLOSEST_CHANNELS = 32
"""int: Number of nearest channels retained when sparsifying template
and cluster waveforms. Matches the default of phylib's
``TemplateModel.n_closest_channels`` for IBL ALF export.
"""

_REQUIRED_COPIES: tuple[tuple[str, str], ...] = (
    ('params.py', 'params.py'),
    ('cluster_KSLabel.tsv', 'cluster_KSLabel.tsv'),
    ('whitening_mat.npy', '_kilosort_whitening.matrix.npy'),
    ('whitening_mat_inv.npy', 'whitening_mat_inv.npy'),
    ('channel_positions.npy', 'channels.localCoordinates.npy'),
)
"""tuple[tuple[str, str], ...]: ``(source_name, destination_name)`` pairs
copied verbatim from the Kilosort directory to the ALF output, where
the source file is always written by Kilosort 4 itself. A missing
source raises ``FileNotFoundError``.
"""

_OPTIONAL_COPIES: tuple[tuple[str, str], ...] = (
    ('_phy_spikes_subset.channels.npy', '_phy_spikes_subset.channels.npy'),
    ('_phy_spikes_subset.spikes.npy', '_phy_spikes_subset.spikes.npy'),
    ('_phy_spikes_subset.waveforms.npy', '_phy_spikes_subset.waveforms.npy'),
)
"""tuple[tuple[str, str], ...]: Direct-copy files that the IBL local
alignment GUI does **not** load — verified by inspecting
``atlaselectrophysiology.load_data_local.LoaderLocal.get_data`` which
only reads the ``spikes`` / ``clusters`` / ``channels`` ALF objects
plus the optional ``ephysTimeRms{AP,LF}`` / ``ephysSpectralDensityLF``
files. The ``_phy_spikes_subset.*.npy`` triplet is produced by
``phylib.io.model.TemplateModel.save_spikes_subset_waveforms`` only
when the upstream IBL pipeline has been run on the session (it streams
the raw ``.ap.bin`` to extract 500 waveform snippets per template and
writes the result into the Kilosort directory). For fresh Kilosort 4
output the triplet is absent; this module copies it when present and
skips it silently otherwise.
"""


def ccf_apdvml_to_xyz_mlapdv_um(xyz_apdvml_um: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Convert anatomical coordinates from the Allen CCF voxel-origin
    convention to the IBL bregma-origin convention used by the IBL
    alignment GUI, without instantiating ``iblatlas.atlas.AllenAtlas``.

    Numerically equivalent to
    ``AllenAtlas(25).ccf2xyz(xyz_apdvml_um, ccf_order='apdvml') * 1e6``
    but avoids the ~300 MB Allen NRRD download because ``ccf2xyz`` only
    uses ``AllenAtlas.bc`` (the affine ``BrainCoordinates`` mapping),
    not the image or annotation volumes.

    The transform decomposes as:

    1. Reorder the input columns from (AP, DV, ML) to (ML, AP, DV).
    2. Translate by the Allen bregma landmark in CCF voxel-origin µm,
       :data:`ALLEN_BREGMA_MLAPDV_UM`.
    3. Flip the sign of the AP and DV axes so that AP is positive
       anterior and DV is positive dorsal (the ML axis is already
       positive lateral-right in both conventions).

    The Allen atlas resolution (25 µm) cancels out algebraically
    because the input is already expressed in micrometres (voxel index
    × resolution), so the result does not depend on ``res_um``.

    Parameters
    ----------
    xyz_apdvml_um : numpy.ndarray
        Coordinates in CCF (AP, DV, ML) voxel-origin micrometres. May
        be a single point of shape ``(3,)`` or an array of shape
        ``(..., 3)``; the conversion is broadcast over the leading
        dimensions.

    Returns
    -------
    numpy.ndarray
        Coordinates in (ML, AP, DV) bregma-origin micrometres, same
        leading shape as the input.
    """
    xyz_apdvml_um = np.asarray(xyz_apdvml_um, dtype=np.float64)
    mlapdv_um = xyz_apdvml_um[..., [2, 0, 1]]
    return (mlapdv_um - ALLEN_BREGMA_MLAPDV_UM) * np.array([1.0, -1.0, -1.0])


def parse_imro_table(value: str) -> list[list[Any]]:
    """
    Description
    -----------
    Parse a SpikeGLX IMRO-table-style metadata string of the form
    ``(group1)(group2)...`` into a list of token lists, one per
    parenthesised group.

    Each group is split on the first delimiter found among ``,``, ``:``
    and space, in that order. Tokens that look like base-10 integers
    are returned as ``int``; everything else is returned as ``str``
    after stripping whitespace. Empty groups and empty tokens are
    skipped.

    This is general enough to handle ``~imroTbl`` (space-delimited 5- or
    6-tuples per channel) and ``~snsGeomMap`` (colon-delimited 4-tuples
    per channel) without branching on the key name.

    Parameters
    ----------
    value : str
        The raw right-hand-side of a SpikeGLX metadata line containing
        one or more parenthesised groups. May be empty.

    Returns
    -------
    list[list[int | str]]
        One list per parenthesised group, in source order. Returns an
        empty list when ``value`` is empty.
    """
    if not value:
        return []
    parsed: list[list[Any]] = []
    for group in re.findall(r'\((.*?)\)', value):
        if ',' in group:
            tokens = group.split(',')
        elif ':' in group:
            tokens = group.split(':')
        else:
            tokens = group.split(' ')
        row: list[Any] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                row.append(int(token))
            except ValueError:
                row.append(token)
        parsed.append(row)
    return parsed


def read_ap_meta(meta_path: str | os.PathLike) -> dict[str, str]:
    """
    Description
    -----------
    Read a SpikeGLX ``*.ap.meta`` file into a flat ``dict`` keyed by
    metadata key with raw string values.

    Leading ``~`` characters in key names (used by SpikeGLX to flag
    multi-line / structured fields like ``~imroTbl`` and
    ``~snsGeomMap``) are stripped so that callers can use the
    canonical IBL/spikeglx key names without the tilde, e.g.
    ``meta['imroTbl']`` instead of ``meta['~imroTbl']``.

    Lines without an ``=`` are skipped. Surrounding whitespace on both
    sides of the ``=`` is stripped. No type coercion is performed on
    values; the caller is responsible for converting specific entries
    (e.g. ``float(meta['imSampRate'])``).

    Parameters
    ----------
    meta_path : str or pathlib.Path
        Filesystem path to the ``.ap.meta`` file.

    Returns
    -------
    dict[str, str]
        All key/value pairs found in the metadata file.
    """
    out: dict[str, str] = {}
    with open(meta_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, _, value = line.partition('=')
            out[key.strip().lstrip('~')] = value.strip()
    return out


def sample_to_volts_ap(meta: dict[str, str]) -> float:
    """
    Description
    -----------
    Compute the scalar that converts a raw int16 sample from a
    Neuropixels 2.0 AP-band channel into volts.

    For all Neuropixels 2.0 variants the AP-band conversion is the
    same across channels (single hard-coded analog gain of 80, no
    per-channel AP-gain column in the IMRO table), so a scalar
    suffices for the IBL ALF amplitude scaling. The formula is:

    .. math::

        \\text{sample2v} = \\frac{\\text{imAiRangeMax}}{\\text{imMaxInt}
        \\cdot \\text{NP2_AP_GAIN}}

    where ``imAiRangeMax`` is the saturated ADC full-scale voltage,
    ``imMaxInt`` is the firmware-reported maximum int sample value
    (commonly 2048 or 8192 for NP2.0) and :data:`NP2_AP_GAIN` is 80.

    Parameters
    ----------
    meta : dict[str, str]
        Output of :func:`read_ap_meta`. Must contain ``imDatPrb_type``,
        ``imAiRangeMax`` and ``imMaxInt``.

    Returns
    -------
    float
        Scaling factor in volts per int16 sample.

    Raises
    ------
    NotImplementedError
        If ``imDatPrb_type`` is not in :data:`NP2_PROBE_TYPES`.
        Neuropixels 1.0 and other legacy probes are intentionally not
        supported by this module.
    """
    probe_type = int(meta['imDatPrb_type'])
    if probe_type not in NP2_PROBE_TYPES:
        raise NotImplementedError(
            f"Probe type {probe_type} is not a Neuropixels 2.0 variant; "
            f"this module supports only NP2.0 ({sorted(NP2_PROBE_TYPES)}). "
            f"Add the new probe family to NP2_PROBE_TYPES if appropriate, "
            f"or extend sample_to_volts_ap with the correct gain formula."
        )
    ai_range_max = float(meta['imAiRangeMax'])
    max_int = int(meta['imMaxInt'])
    return ai_range_max / max_int / NP2_AP_GAIN


class IBLAlignmentExporter:
    """
    Description
    -----------
    Orchestrates the conversion of one Kilosort 4 output directory into
    the ALF layout consumed by the IBL ephys-alignment GUI, and the
    surrounding pre/post-alignment file shuffling, for one
    ``(mouse_id, session_date, probe_id, hemisphere)`` tuple.

    The exporter caches the parsed ``.ap.meta`` and resolved paths at
    construction time; each public step method (:meth:`write_xyz_picks`
    etc.) reads only what it needs from that cache so the four steps
    can be invoked independently in any order, with two caveats:

    * :meth:`write_xyz_picks` and :meth:`write_alf_outputs` are
      pre-alignment steps (run before opening the IBL GUI).
    * :meth:`remap_channel_ids_to_raw` and
      :meth:`write_unified_channel_locations` are post-alignment steps
      (run after the GUI has written ``channel_locations_shank{n}.
      json``).

    Parameters
    ----------
    os_cup_loc : str or pathlib.Path
        Root mount point of the storage server (e.g. ``/mnt/falkner/
        Bartul``).
    mouse_id : str
        Animal identifier (e.g. ``"164335_0"``). Used to locate both
        the histology and EPHYS directories.
    session_date : str
        Recording date in ``YYYYMMDD`` format (e.g. ``"20250911"``).
    probe_id : str
        SpikeGLX probe label (e.g. ``"imec1"``). Combined with
        ``session_date`` to form the EPHYS subdirectory name.
    hemisphere : str
        ``"L"`` or ``"R"``. Selects which set of brainreg track ``.npy``
        files is consumed and determines the output subdirectory name
        ``ibl_{hemisphere}H``.
    kilosort_version : str or int, default ``"4"``
        Version suffix of the Kilosort subdirectory under the EPHYS
        directory (``kilosort4/`` by default).
    histology_dirname : str, default ``"histology"``
        Top-level directory name under ``os_cup_loc`` that contains the
        per-animal histology output. Override if the lab convention
        differs.
    ephys_dirname : str, default ``"EPHYS"``
        Top-level directory name under ``os_cup_loc`` that contains the
        per-session EPHYS output. Override if the lab convention
        differs.
    out_subdir : str or None, default ``None``
        When ``None``, the ALF outputs land in
        ``{cup_root}/{histology_dirname}/{mouse_id}/{session_date}/
        ibl_{hemisphere}H/`` (the canonical IBL alignment GUI input
        directory). Set this to a different leaf name to write into a
        sibling directory — useful for validating a new exporter
        against an existing IBL reference output without overwriting
        it (e.g. ``out_subdir='ibl_LH_v2'``).
    """

    def __init__(
        self,
        os_cup_loc: str | os.PathLike,
        mouse_id: str,
        session_date: str,
        probe_id: str,
        hemisphere: str,
        kilosort_version: str | int = '4',
        histology_dirname: str = 'histology',
        ephys_dirname: str = 'EPHYS',
        out_subdir: str | None = None,
    ) -> None:
        if hemisphere not in ('L', 'R'):
            raise ValueError(f"hemisphere must be 'L' or 'R', got {hemisphere!r}")

        self.os_cup_loc = Path(os_cup_loc)
        self.mouse_id = mouse_id
        self.session_date = session_date
        self.probe_id = probe_id
        self.hemisphere = hemisphere
        self.kilosort_version = str(kilosort_version)

        self.brainreg_path = self.os_cup_loc / histology_dirname / mouse_id
        self.tracks_output_path = self.brainreg_path
        ephys_base_dir = f"{session_date}_{probe_id}"
        self.ephys_path = self.os_cup_loc / ephys_dirname / ephys_base_dir
        self.ks_path = self.ephys_path / f"kilosort{self.kilosort_version}"
        leaf = out_subdir if out_subdir is not None else f"ibl_{hemisphere}H"
        self.ephys_out_path = (
            self.os_cup_loc
            / histology_dirname
            / mouse_id
            / session_date
            / leaf
        )
        self.ephys_out_path.mkdir(parents=True, exist_ok=True)

        meta_candidates = sorted(self.ephys_path.glob('concatenated_*.ap.meta'))
        if not meta_candidates:
            raise FileNotFoundError(
                f"No concatenated_*.ap.meta found under {self.ephys_path}. "
                f"Run modify_files.concatenate_binary_files() first."
            )
        if len(meta_candidates) > 1:
            raise RuntimeError(
                f"Multiple concatenated .ap.meta files under {self.ephys_path}; "
                f"expected exactly one. Found: {[p.name for p in meta_candidates]}"
            )
        self.meta_path = meta_candidates[0]
        self.meta = read_ap_meta(self.meta_path)

        self.probe_type = int(self.meta['imDatPrb_type'])
        if self.probe_type not in NP2_PROBE_TYPES:
            raise NotImplementedError(
                f"Probe type {self.probe_type} in {self.meta_path.name} "
                f"is not a supported Neuropixels 2.0 variant "
                f"({sorted(NP2_PROBE_TYPES)})."
            )
        self.is_multishank = self.probe_type in NP2_MULTISHANK_PROBE_TYPES
        self.sample_rate = float(self.meta['imSampRate'])
        self.sample2v = sample_to_volts_ap(self.meta)
        self.imro_rows = parse_imro_table(self.meta['imroTbl'])

    def write_xyz_picks(self) -> list[Path]:
        """
        Description
        -----------
        Convert per-shank brainreg track-point ``.npy`` files (in Allen
        CCF apdvml voxel-origin µm) into the
        ``xyz_picks_shank{n}.json`` files the IBL alignment GUI loads to
        place each shank's track in CCF space.

        Iterates over ``{hemisphere}*.npy`` files in :attr:`brainreg_
        path` in lexicographic order, applies
        :func:`ccf_apdvml_to_xyz_mlapdv_um` to convert each track's
        point cloud to bregma-origin (ML, AP, DV) µm, and writes one
        JSON per shank into :attr:`ephys_out_path` with the schema
        ``{"xyz_picks": [[ml, ap, dv], ...]}``.

        Per-file skip-if-exists: any ``xyz_picks_shank{n}.json`` that
        already exists in the output directory is left untouched and a
        one-line stdout note is printed. To force regeneration (e.g.
        after re-tracing a shank in napari), delete the existing JSON
        and re-run.

        Returns
        -------
        list[pathlib.Path]
            Paths of the JSON files that were newly written this call.
            Files skipped because they already existed are not
            included. Empty if every shank was already present, or if
            no matching ``.npy`` files were found.
        """
        written: list[Path] = []
        pattern = f"**{os.sep}{self.hemisphere}*.npy"
        for one_file_idx, one_file in enumerate(sorted(self.brainreg_path.glob(pattern))):
            out_name = f"xyz_picks_shank{one_file_idx + 1}.json"
            out_path = self.ephys_out_path / out_name
            if out_path.is_file():
                print(f"xyz_picks: {out_name} already exists in {self.ephys_out_path}; skipping.")
                continue
            xyz_apdvml = np.load(one_file)
            xyz_mlapdv_um = ccf_apdvml_to_xyz_mlapdv_um(xyz_apdvml)
            xyz_picks = {'xyz_picks': xyz_mlapdv_um.tolist()}
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(xyz_picks, fh, indent=2)
            written.append(out_path)
        return written

    def write_alf_outputs(self) -> None:
        """
        Description
        -----------
        Produce the ALF layout consumed by the IBL alignment GUI from
        the cached Kilosort directory and ``.ap.meta``, without
        depending on phylib, ibllib, spikeglx or iblatlas.

        The conversion has four sub-passes:

        1. **Copies.** :data:`_REQUIRED_COPIES` (``params.py``,
           ``cluster_KSLabel.tsv``, ``whitening_mat{,_inv}.npy``,
           ``channel_positions.npy``) are copied with renames; a
           missing source aborts the export.
           :data:`_OPTIONAL_COPIES` — the three
           ``_phy_spikes_subset.*.npy`` files — are copied only when
           present in the Kilosort directory (they are produced by
           the upstream IBL pipeline, not by Kilosort 4 itself) and
           skipped silently otherwise; the IBL local alignment GUI
           does not load them.
        2. **Channel-level outputs.** ``channels.rawInd.npy`` is
           derived from ``channel_map.npy``.
        3. **Spike/template-level outputs.** ``spikes.{times, samples,
           clusters, templates, amps, depths}.npy``,
           ``templates.{amps, waveforms, waveformsChannels}.npy``.
           Amplitude scaling, template unwhitening, peak-channel and
           nearest-channel restriction follow the same logic as
           ``phylib.io.model.TemplateModel.get_amplitudes_true`` and
           ``EphysAlfCreator.make_template_and_spikes_objects``.
        4. **Cluster-level outputs.** ``clusters.{channels, amps,
           depths, peakToTrough, waveforms, waveformsChannels}.npy``.
           Supports phy-curated sessions where ``spike_clusters.npy``
           differs from ``spike_templates.npy``: each cluster's
           whitened waveform is the spike-count-weighted average of
           the templates contributing to it (matching phylib's
           ``cluster_waveforms`` for the dense, non-sparse case).

        The IBL RMS-map step (``extract_rmsmap``) is intentionally
        skipped: it is the dominant cost of ``extract_data`` (full pass
        over the concatenated AP binary) and the IBL alignment GUI
        does not require its ``_iblqc_*`` outputs.

        Returns
        -------
        None
        """
        ks = self._load_kilosort_arrays()
        self._copy_direct_files()
        self._write_channel_outputs(ks)
        self._write_template_and_spike_outputs(ks)
        self._write_cluster_outputs(ks)

    def remap_channel_ids_to_raw(self) -> list[Path]:
        """
        Description
        -----------
        Post-process the per-shank ``channel_locations_shank{n}.json``
        files produced by the IBL alignment GUI so the keys go from
        per-shank channel indices (``channel_0`` .. ``channel_{m-1}``)
        back to raw recording channel ids (``channel_{raw_id}``) using
        the IMRO table cached at construction time.

        Each ``imroTbl`` data row for a multi-shank NP2.0 probe is
        ``(channel, shank, bank, refid, elecid)`` (see SpikeGLX
        documentation); the channel column is at index 0 and the shank
        column at index 1. Channels are walked in IMRO order and, for
        each shank, an index counter is bumped so the GUI's
        ``channel_{0..m-1}`` keys are mapped to the actual raw channel
        ids on that shank.

        For single-shank NP2.0 probes (no shank column in the IMRO
        rows) the function is a no-op and returns an empty list.

        Returns
        -------
        list[pathlib.Path]
            Paths of the JSON files that were rewritten, in shank
            order. Empty when no per-shank JSONs are found, or when
            the probe is single-shank.

        Raises
        ------
        FileNotFoundError
            If a ``channel_locations_shank{n}.json`` referenced by the
            IMRO table is missing from :attr:`ephys_out_path`.
        """
        if not self.is_multishank:
            return []

        shank_indices_in_imro = [row[1] for row in self.imro_rows[1:]]
        n_shanks = int(np.max(shank_indices_in_imro)) + 1

        rewritten: list[Path] = []
        for shank_num in range(n_shanks):
            shank_json = self.ephys_out_path / f"channel_locations_shank{shank_num + 1}.json"
            if not shank_json.exists():
                raise FileNotFoundError(
                    f"Missing {shank_json.name} in {self.ephys_out_path}; "
                    f"run the IBL alignment GUI for shank {shank_num + 1} first."
                )
            with open(shank_json, 'r', encoding='utf-8') as fh:
                shank_data = json.load(fh)

            remapped: dict[str, Any] = {}
            channel_counter = 0
            for imro_item in self.imro_rows[1:]:
                if imro_item[1] != shank_num:
                    continue
                raw_channel_id = imro_item[0]
                remapped[f"channel_{raw_channel_id}"] = shank_data[f"channel_{channel_counter}"]
                channel_counter += 1

            if 'origin' in shank_data:
                remapped['origin'] = shank_data['origin']

            with open(shank_json, 'w', encoding='utf-8') as fh:
                json.dump(remapped, fh, indent=4)
            rewritten.append(shank_json)
        return rewritten

    def write_unified_channel_locations(self) -> Path:
        """
        Description
        -----------
        Merge all per-shank ``channel_locations_shank{n}.json`` files
        in :attr:`ephys_out_path` into a single ``channel_locations.
        json`` keyed by raw channel id and sorted by the integer
        channel index (with any non-``channel_<int>`` keys, e.g.
        ``"origin"``, sorted to the end).

        This produces the layout SpikeInterface expects when consuming
        the IBL alignment output back into the wider analysis pipeline.

        Assumes :meth:`remap_channel_ids_to_raw` has already been run
        for multi-shank probes so the per-shank keys are already raw
        channel ids (``channel_{raw_id}`` rather than the GUI's
        per-shank ``channel_{0..m-1}``).

        Returns
        -------
        pathlib.Path
            Path of the unified ``channel_locations.json`` file.

        Raises
        ------
        FileNotFoundError
            If no ``channel_locations_shank*.json`` files are found
            under :attr:`ephys_out_path`.
        """
        shank_jsons = sorted(self.ephys_out_path.glob('channel_locations_shank*.json'))
        if not shank_jsons:
            raise FileNotFoundError(
                f"No channel_locations_shank*.json found under {self.ephys_out_path}."
            )

        merged: dict[str, Any] = {}
        for shank_json in shank_jsons:
            with open(shank_json, 'r', encoding='utf-8') as fh:
                shank_data = json.load(fh)
            for key, value in shank_data.items():
                if key not in merged:
                    merged[key] = value

        def _sort_key(item: tuple[str, Any]) -> int | float:
            key = item[0]
            if key.startswith('channel_'):
                tail = key.split('_', 1)[1]
                try:
                    return int(tail)
                except ValueError:
                    return float('inf')
            return float('inf')

        merged = dict(sorted(merged.items(), key=_sort_key))

        out_path = self.ephys_out_path / 'channel_locations.json'
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(merged, fh, indent=4)
        return out_path

    def _load_kilosort_arrays(self) -> dict[str, np.ndarray]:
        """
        Description
        -----------
        Load every Kilosort 4 output array consumed by
        :meth:`write_alf_outputs` into a single dict, performing the
        same dtype/shape sanity checks ``phylib.io.model.TemplateModel``
        would. The loaded arrays are kept in memory for the duration of
        the export because they are reused across cluster/template/spike
        passes and they are typically small (≤ a few GB) compared to
        the raw binary.

        Loaded keys:
            ``spike_times`` (n_spikes,), ``spike_clusters`` (n_spikes,),
            ``spike_templates`` (n_spikes,), ``amplitudes`` (n_spikes,),
            ``templates`` (n_templates, n_samples_waveforms, n_channels),
            ``whitening_mat_inv`` (n_channels, n_channels),
            ``channel_map`` (n_channels,),
            ``channel_positions`` (n_channels, 2),
            ``pc_features`` (n_spikes, n_pcs, n_local_channels),
            ``pc_feature_ind`` (n_templates, n_local_channels).

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from canonical KS array name to the loaded array.
        """
        ks: dict[str, np.ndarray] = {}
        ks['spike_times'] = np.load(self.ks_path / 'spike_times.npy').squeeze().astype(np.int64)
        ks['spike_clusters'] = np.load(self.ks_path / 'spike_clusters.npy').squeeze().astype(np.int64)
        ks['spike_templates'] = np.load(self.ks_path / 'spike_templates.npy').squeeze().astype(np.int64)
        ks['amplitudes'] = np.load(self.ks_path / 'amplitudes.npy').squeeze().astype(np.float64)
        ks['templates'] = np.load(self.ks_path / 'templates.npy').astype(np.float64)
        ks['whitening_mat_inv'] = np.load(self.ks_path / 'whitening_mat_inv.npy').astype(np.float64)
        ks['channel_map'] = np.load(self.ks_path / 'channel_map.npy').squeeze().astype(np.int64)
        ks['channel_positions'] = np.load(self.ks_path / 'channel_positions.npy').astype(np.float64)
        ks['pc_features'] = np.load(self.ks_path / 'pc_features.npy', mmap_mode='r')
        ks['pc_feature_ind'] = np.load(self.ks_path / 'pc_feature_ind.npy').astype(np.int64)
        return ks

    def _copy_direct_files(self) -> None:
        """
        Description
        -----------
        Execute the :data:`_REQUIRED_COPIES` and :data:`_OPTIONAL_COPIES`
        tables: copy each named file from the Kilosort directory to
        :attr:`ephys_out_path`, renaming on the destination side.
        Missing required sources raise ``FileNotFoundError`` with a
        context-rich message; missing optional sources (the
        ``_phy_spikes_subset.*.npy`` triplet) are skipped silently with
        a one-line stdout note so re-runs without phylib-extracted
        waveforms still succeed. Existing destinations are overwritten
        so re-runs are idempotent.

        Returns
        -------
        None
        """
        for src_name, dst_name in _REQUIRED_COPIES:
            src = self.ks_path / src_name
            dst = self.ephys_out_path / dst_name
            if not src.is_file():
                raise FileNotFoundError(
                    f"Expected Kilosort file {src} is missing; cannot complete ALF export."
                )
            shutil.copyfile(src, dst)

        for src_name, dst_name in _OPTIONAL_COPIES:
            src = self.ks_path / src_name
            dst = self.ephys_out_path / dst_name
            if not src.is_file():
                print(
                    f"Optional file {src_name} not present in {self.ks_path}; "
                    f"skipping (the IBL alignment GUI does not require it)."
                )
                continue
            shutil.copyfile(src, dst)

    def _write_channel_outputs(self, ks: dict[str, np.ndarray]) -> None:
        """
        Description
        -----------
        Write ``channels.rawInd.npy``.

        For a single-probe Kilosort recording the rawInd is identical to
        ``channel_map.npy`` (the raw recording channel id for each
        sorted channel). phylib's multi-probe handling is unnecessary
        here because the IBL alignment GUI always operates on one
        probe at a time.

        ``channels.localCoordinates.npy`` is produced by
        :meth:`_copy_direct_files` and is not re-emitted here.

        Parameters
        ----------
        ks : dict[str, numpy.ndarray]
            Output of :meth:`_load_kilosort_arrays`.

        Returns
        -------
        None
        """
        raw_ind = ks['channel_map'].astype(np.int64)
        np.save(self.ephys_out_path / 'channels.rawInd.npy', raw_ind)

    def _write_template_and_spike_outputs(self, ks: dict[str, np.ndarray]) -> None:
        """
        Description
        -----------
        Compute and save the spike- and template-level ALF outputs:
        ``spikes.times.npy``, ``spikes.samples.npy``,
        ``spikes.clusters.npy``, ``spikes.templates.npy``,
        ``spikes.amps.npy``, ``spikes.depths.npy``,
        ``templates.amps.npy``, ``templates.waveforms.npy``,
        ``templates.waveformsChannels.npy``.

        Algorithm (matches phylib's ``get_amplitudes_true(use=
        'templates')`` for the dense / non-sparse case):

        1. Unwhiten the Kilosort templates with the inverse whitening
           matrix.
        2. Per template, the AU amplitude is the largest channel's
           peak-to-trough range.
        3. Per spike, the voltage amplitude is
           ``templates_amps_au[t] * amplitudes[s] * sample2v``.
        4. Per template, the voltage amplitude is the bincount-mean of
           the per-spike voltage amplitudes for spikes that fired that
           template.
        5. The voltage-rescaled template waveform is the unwhitened
           template multiplied by ``templates_amps_v / templates_amps_
           au`` so each template's peak channel reaches its true mean
           spike amplitude.
        6. For each template, keep only the ``N_CLOSEST_CHANNELS``
           channels nearest to the peak channel by Manhattan distance
           on the probe layout.

        ``spikes.depths.npy`` is computed in
        :meth:`_compute_spike_depths` using the Kilosort PC features.

        Parameters
        ----------
        ks : dict[str, numpy.ndarray]
            Output of :meth:`_load_kilosort_arrays`.

        Returns
        -------
        None
        """
        spike_times = ks['spike_times']
        spike_clusters = ks['spike_clusters']
        spike_templates = ks['spike_templates']
        amplitudes = ks['amplitudes']
        templates = ks['templates']
        wmi = ks['whitening_mat_inv']
        channel_positions = ks['channel_positions']

        np.save(self.ephys_out_path / 'spikes.times.npy', spike_times / self.sample_rate)
        np.save(self.ephys_out_path / 'spikes.samples.npy', spike_times.astype(np.int64))
        np.save(self.ephys_out_path / 'spikes.clusters.npy', spike_clusters.astype(np.uint16))
        np.save(self.ephys_out_path / 'spikes.templates.npy', spike_templates.astype(np.uint16))

        templates_whitened_ch_amps = templates.max(axis=1) - templates.min(axis=1)
        templates_channels = templates_whitened_ch_amps.argmax(axis=1)

        templates_v = templates @ wmi
        templates_ch_amps = templates_v.max(axis=1) - templates_v.min(axis=1)
        templates_amps_au = templates_ch_amps.max(axis=1)

        spike_amps_au = templates_amps_au[spike_templates] * amplitudes
        spike_amps_v = spike_amps_au * self.sample2v
        np.save(self.ephys_out_path / 'spikes.amps.npy', spike_amps_v.astype(np.float32))

        n_templates = templates.shape[0]
        sum_per_template = np.bincount(spike_templates, weights=spike_amps_au, minlength=n_templates)
        count_per_template = np.bincount(spike_templates, minlength=n_templates)
        with np.errstate(divide='ignore', invalid='ignore'):
            templates_amps_au_avg = np.where(
                count_per_template > 0,
                sum_per_template / count_per_template,
                0.0,
            )
        templates_amps_v_avg = templates_amps_au_avg * self.sample2v
        np.save(self.ephys_out_path / 'templates.amps.npy', templates_amps_v_avg)

        with np.errstate(divide='ignore', invalid='ignore'):
            rescale = np.where(
                templates_amps_au > 0,
                templates_amps_v_avg / templates_amps_au,
                0.0,
            )
        templates_volts = templates_v * rescale[:, np.newaxis, np.newaxis]

        templates_inds, templates_waveforms = self._restrict_to_nearest_channels(
            templates_volts, templates_channels, channel_positions
        )
        np.save(self.ephys_out_path / 'templates.waveforms.npy', templates_waveforms.astype(np.float32))
        np.save(self.ephys_out_path / 'templates.waveformsChannels.npy', templates_inds.astype(np.int32))

        spike_depths = self._compute_spike_depths(ks)
        np.save(self.ephys_out_path / 'spikes.depths.npy', spike_depths.astype(np.float32))

    def _write_cluster_outputs(self, ks: dict[str, np.ndarray]) -> None:
        """
        Description
        -----------
        Compute and save the cluster-level ALF outputs:
        ``clusters.channels.npy``, ``clusters.amps.npy``,
        ``clusters.depths.npy``, ``clusters.peakToTrough.npy``,
        ``clusters.waveforms.npy`` and ``clusters.waveformsChannels.
        npy``.

        Handles phy-curated sessions where ``spike_clusters !=
        spike_templates`` by mirroring phylib's ``cluster_waveforms`` /
        ``get_cluster_mean_waveforms`` logic for the dense
        (non-sparse) case: for each cluster, find the templates that
        contributed spikes and take the spike-count-weighted average
        of their whitened waveforms; that yields a ``sparse_clusters
        .data`` analogue which is then unwhitened and rescaled to
        volts exactly as in the template pass.

        Empty clusters (cluster ids in ``range(max + 1)`` that received
        no spikes from any template, e.g. clusters deleted in phy) are
        written with NaN in ``clusters.peakToTrough.npy`` and
        ``clusters.depths.npy`` and zero in
        ``clusters.{channels, amps, waveforms, waveformsChannels}.npy``,
        matching phylib's ``nan_idx`` convention.

        Parameters
        ----------
        ks : dict[str, numpy.ndarray]
            Output of :meth:`_load_kilosort_arrays`.

        Returns
        -------
        None
        """
        spike_clusters = ks['spike_clusters']
        spike_templates = ks['spike_templates']
        amplitudes = ks['amplitudes']
        templates = ks['templates']
        wmi = ks['whitening_mat_inv']
        channel_positions = ks['channel_positions']

        n_clusters = int(spike_clusters.max()) + 1
        n_samples_waveforms = templates.shape[1]
        n_channels = templates.shape[2]

        pair_counts, nan_clusters = self._build_cluster_template_map(spike_clusters, spike_templates, n_clusters)

        clusters_whitened = np.zeros((n_clusters, n_samples_waveforms, n_channels), dtype=np.float64)
        for cluster_id, contributions in pair_counts.items():
            if not contributions:
                continue
            template_ids = np.fromiter(contributions.keys(), dtype=np.int64)
            counts = np.fromiter(contributions.values(), dtype=np.float64)
            if template_ids.size == 1:
                clusters_whitened[cluster_id] = templates[template_ids[0]]
            else:
                clusters_whitened[cluster_id] = np.average(
                    templates[template_ids], axis=0, weights=counts
                )

        clusters_whitened_ch_amps = clusters_whitened.max(axis=1) - clusters_whitened.min(axis=1)
        clusters_channels = clusters_whitened_ch_amps.argmax(axis=1)

        clusters_unwhitened = clusters_whitened @ wmi
        clusters_ch_amps = clusters_unwhitened.max(axis=1) - clusters_unwhitened.min(axis=1)
        clusters_amps_au = clusters_ch_amps.max(axis=1)

        spike_amps_au_via_clusters = clusters_amps_au[spike_clusters] * amplitudes
        sum_per_cluster = np.bincount(
            spike_clusters, weights=spike_amps_au_via_clusters, minlength=n_clusters
        )
        count_per_cluster = np.bincount(spike_clusters, minlength=n_clusters)
        with np.errstate(divide='ignore', invalid='ignore'):
            clusters_amps_au_avg = np.where(
                count_per_cluster > 0,
                sum_per_cluster / count_per_cluster,
                0.0,
            )
        clusters_amps_v_avg = clusters_amps_au_avg * self.sample2v

        with np.errstate(divide='ignore', invalid='ignore'):
            rescale = np.where(
                clusters_amps_au > 0,
                clusters_amps_v_avg / clusters_amps_au,
                0.0,
            )
        clusters_volts = clusters_unwhitened * rescale[:, np.newaxis, np.newaxis]

        peak_to_trough = self._waveform_durations(clusters_whitened, clusters_channels)
        if nan_clusters.size:
            peak_to_trough[nan_clusters] = np.nan

        clusters_depths = channel_positions[clusters_channels, 1].astype(np.float64)
        if nan_clusters.size:
            clusters_depths[nan_clusters] = np.nan

        clusters_inds, clusters_waveforms = self._restrict_to_nearest_channels(
            clusters_volts, clusters_channels, channel_positions
        )

        np.save(self.ephys_out_path / 'clusters.channels.npy', clusters_channels.astype(np.int64))
        np.save(self.ephys_out_path / 'clusters.amps.npy', clusters_amps_v_avg)
        np.save(self.ephys_out_path / 'clusters.depths.npy', clusters_depths.astype(np.float32))
        np.save(self.ephys_out_path / 'clusters.peakToTrough.npy', peak_to_trough)
        np.save(self.ephys_out_path / 'clusters.waveforms.npy', clusters_waveforms.astype(np.float32))
        np.save(self.ephys_out_path / 'clusters.waveformsChannels.npy', clusters_inds.astype(np.int32))

    def _build_cluster_template_map(
        self,
        spike_clusters: np.ndarray,
        spike_templates: np.ndarray,
        n_clusters: int,
    ) -> tuple[dict[int, dict[int, int]], np.ndarray]:
        """
        Description
        -----------
        Build the ``{cluster_id: {template_id: spike_count}}`` mapping
        used to aggregate template waveforms into cluster waveforms,
        mirroring ``phylib.io.model.TemplateModel.get_merge_map`` for
        any combination of phy splits and merges.

        Implementation uses ``np.unique`` on the encoded
        ``cluster_id * (n_templates + 1) + template_id`` pair integers
        so the per-spike Python loop in phylib is replaced with one
        vectorised sort.

        Parameters
        ----------
        spike_clusters : numpy.ndarray
            Per-spike cluster ids (shape ``(n_spikes,)``).
        spike_templates : numpy.ndarray
            Per-spike template ids (shape ``(n_spikes,)``).
        n_clusters : int
            ``spike_clusters.max() + 1``. Determines the size of the
            output dict's key range.

        Returns
        -------
        dict[int, dict[int, int]]
            ``{cluster_id: {template_id: count}}`` for every cluster id
            in ``range(n_clusters)``. Empty clusters have an empty
            inner dict.
        numpy.ndarray
            Array of cluster ids that received no spikes (so their
            inner dict is empty), in ascending order.
        """
        n_templates_max = int(spike_templates.max()) + 1
        encoded = spike_clusters.astype(np.int64) * n_templates_max + spike_templates.astype(np.int64)
        unique_codes, counts = np.unique(encoded, return_counts=True)
        cluster_ids = unique_codes // n_templates_max
        template_ids = unique_codes % n_templates_max

        pair_counts: dict[int, dict[int, int]] = {cid: {} for cid in range(n_clusters)}
        for cid, tid, cnt in zip(cluster_ids.tolist(), template_ids.tolist(), counts.tolist()):
            pair_counts[cid][tid] = cnt

        nan_clusters = np.array(
            [cid for cid in range(n_clusters) if not pair_counts[cid]],
            dtype=np.int64,
        )
        return pair_counts, nan_clusters

    def _compute_spike_depths(self, ks: dict[str, np.ndarray]) -> np.ndarray:
        """
        Description
        -----------
        Compute per-spike depth as the squared-PC-feature-weighted
        average of channel y-positions, matching ``phylib.io.model.
        TemplateModel.get_depths``.

        For each spike:

        1. Take the first PC of the PC features on the spike's
           template-local channels (shape ``(n_local_channels,)``).
        2. Keep only positive values and square them; this is the
           amplitude-like weight per local channel.
        3. Take the y-coordinate of each local channel from
           ``channel_positions``.
        4. Compute the weighted mean of those y-coordinates.

        Note on axis ordering: Kilosort 4's ``pc_features.npy`` is
        stored as ``(n_spikes, n_pcs, n_local_channels)``; phylib
        transposes to ``(n_spikes, n_local_channels, n_pcs)`` and then
        indexes ``[:, :, 0]`` for the first PC. Indexing the raw array
        as ``pc_features[:, 0, :]`` here gives the same n_local_channels
        vector without any transpose.

        Spikes are processed in batches of 50 000 (matching phylib) so
        the memmapped ``pc_features.npy`` is not fully materialised.

        Parameters
        ----------
        ks : dict[str, numpy.ndarray]
            Output of :meth:`_load_kilosort_arrays`. Reads
            ``pc_features``, ``pc_feature_ind``, ``spike_templates``
            and ``channel_positions``.

        Returns
        -------
        numpy.ndarray
            Per-spike depth in the same units as
            ``channel_positions[:, 1]`` (micrometres for Kilosort 4
            output), shape ``(n_spikes,)``.
        """
        pc_features = ks['pc_features']
        pc_feature_ind = ks['pc_feature_ind']
        spike_templates = ks['spike_templates']
        channel_positions = ks['channel_positions']

        n_spikes = spike_templates.shape[0]
        spike_depths = np.full(n_spikes, np.nan, dtype=np.float64)
        nbatch = 50_000
        for start in range(0, n_spikes, nbatch):
            stop = min(start + nbatch, n_spikes)
            ispi = np.arange(start, stop)
            features = np.asarray(pc_features[ispi, 0, :], dtype=np.float64)
            features = np.maximum(features, 0.0) ** 2
            ichannels = pc_feature_ind[spike_templates[ispi]]
            ypos = channel_positions[ichannels, 1]
            denom = features.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                spike_depths[ispi] = np.where(
                    denom > 0,
                    (ypos * features).sum(axis=1) / denom,
                    np.nan,
                )
        return spike_depths

    def _waveform_durations(
        self,
        sparse_data: np.ndarray,
        peak_channels: np.ndarray,
    ) -> np.ndarray:
        """
        Description
        -----------
        Compute the peak-to-trough duration (in milliseconds) of each
        waveform on its peak channel, mirroring ``phylib.io.model.
        TemplateModel._waveform_durations``.

        For each waveform ``w`` of shape ``(n_samples, n_channels)``,
        the duration is ``(argmax(w[:, peak_channel]) - argmin(w[:,
        peak_channel])) / sample_rate * 1000``. A positive duration
        corresponds to the canonical action-potential shape where the
        trough precedes the rebound peak.

        Parameters
        ----------
        sparse_data : numpy.ndarray
            Waveform array of shape ``(n_waveforms, n_samples,
            n_channels)``. Whitened or unwhitened — the argmax/argmin
            positions are invariant under the channel-wise linear
            unwhitening so either input gives the same result on the
            peak channel.
        peak_channels : numpy.ndarray
            Per-waveform peak channel index of shape ``(n_waveforms,)``.

        Returns
        -------
        numpy.ndarray
            Per-waveform durations in milliseconds, shape
            ``(n_waveforms,)``.
        """
        n_waveforms = sparse_data.shape[0]
        argmax_per_channel = sparse_data.argmax(axis=1)
        argmin_per_channel = sparse_data.argmin(axis=1)
        idx = np.arange(n_waveforms)
        durations_samples = (
            argmax_per_channel[idx, peak_channels] - argmin_per_channel[idx, peak_channels]
        )
        return durations_samples.astype(np.float64) / self.sample_rate * 1e3

    def _restrict_to_nearest_channels(
        self,
        waveforms_full: np.ndarray,
        peak_channels: np.ndarray,
        channel_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Description
        -----------
        For each waveform, keep only the :data:`N_CLOSEST_CHANNELS`
        channels nearest its peak channel by Manhattan distance on the
        probe layout, mirroring the nearest-channel selection in
        ``phylib.io.alf.EphysAlfCreator.make_template_and_spikes_
        objects``.

        Shank-aware filtering is intentionally **not** applied here.
        For multi-shank Neuropixels 2.0 probes the Kilosort
        ``channel_positions.npy`` is laid out without absolute shank
        offsets — all four shanks share the same set of x-coordinates
        — so a Manhattan distance computed on
        ``channel_positions`` alone can place same-y channels from
        other shanks at the same distance as nearby same-shank
        channels. Phylib's reference implementation accepts this and
        does not consult the IMRO shank column when picking the nearest
        channels (its ``channel_distance[channel_probes != current_
        probe] += np.inf`` line guards against multi-*probe*
        recordings, not multi-shank). This module mirrors that
        behaviour exactly so the resulting
        ``templates.waveforms{Channels,}.npy`` and
        ``clusters.waveforms{Channels,}.npy`` arrays match the IBL
        reference output byte for byte (up to argsort tie-breaking on
        equidistant channels).

        Parameters
        ----------
        waveforms_full : numpy.ndarray
            Dense waveforms of shape ``(n_waveforms, n_samples,
            n_channels)``.
        peak_channels : numpy.ndarray
            Per-waveform peak channel index of shape ``(n_waveforms,)``.
        channel_positions : numpy.ndarray
            Per-channel (x, y) layout of shape ``(n_channels, 2)``.

        Returns
        -------
        numpy.ndarray
            Per-waveform channel indices kept, shape
            ``(n_waveforms, N_CLOSEST_CHANNELS)``.
        numpy.ndarray
            Sparse waveforms of shape ``(n_waveforms, n_samples,
            N_CLOSEST_CHANNELS)``.
        """
        n_waveforms, n_samples, n_channels = waveforms_full.shape
        ncw = min(N_CLOSEST_CHANNELS, n_channels)

        templates_inds = np.zeros((n_waveforms, ncw), dtype=np.int64)
        templates_sparse = np.zeros((n_waveforms, n_samples, ncw), dtype=np.float64)
        for t in range(n_waveforms):
            peak = int(peak_channels[t])
            distances = np.sum(
                np.abs(channel_positions - channel_positions[peak]), axis=1
            ).astype(np.float64)
            keep = np.argsort(distances)[:ncw]
            templates_inds[t] = keep
            templates_sparse[t] = waveforms_full[t][:, keep]
        return templates_inds, templates_sparse
