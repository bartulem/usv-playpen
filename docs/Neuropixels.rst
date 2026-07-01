.. _Neuropixels:

Neuropixels
===========
This page explains how to use the **Neuropixels histology and
unit-quality pipelines** in ``usv_playpen.neuropixels``. Where the
:ref:`Process` section ends with spike-sorted, session-split clusters,
this subsystem brackets the manual brainreg + napari steps you run
outside the GUI: it assembles raw light-sheet microscopy into registrable
volumes, bridges Kilosort output and brainreg track tracing to the IBL
ephys-alignment GUI, and finally distils every unit into a single
quality-metrics catalog anchored to the Allen CCF.

The pipeline has three phases that run in order, with two manual
(external) steps in between:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Phase
     - Produces
     - Driver
   * - Light-sheet volume assembly
     - one BigTIFF volume per channel (brainreg / napari input)
     - ``stack_lightsheet_volume`` / ``stitch_smartspim_tiles``
   * - *(external)* brainreg + napari
     - registered volume + per-shank track ``.npy`` files
     - brainreg / napari (outside this notebook)
   * - IBL ephys-alignment export (pre)
     - ``xyz_picks_shank{n}.json`` + the ALF layout
     - ``IBLAlignmentExporter``
   * - *(external)* IBL alignment GUI
     - one ``channel_locations_shank{n}.json`` per shank
     - IBL ephys-alignment GUI (outside this notebook)
   * - IBL ephys-alignment export (post)
     - a unified ``channel_locations.json``
     - ``IBLAlignmentExporter``
   * - Spike quality metrics
     - the global ``unit_catalog.csv`` + per-shank channel order
     - ``SpikeQualityMetricsExtractor``

Only **Neuropixels 2.0** probes are supported; anything else raises
``NotImplementedError``. The same 384 electrodes are referred to by four
different indexing conventions across the files this pipeline produces and
consumes — see the :ref:`channel-indexing` section below for the full landscape,
which is essential when joining IBL anatomy to Kilosort spike data.

.. note::

   These pipelines are **not** exposed as a GUI tab. Run them from the
   :ref:`histology-notebook` (interactive, single node, linked at the
   bottom of this page) or by importing the classes directly. The
   :ref:`SpikeGLX meta → geometry converter <histology-meta-to-coords>`
   below is the one piece with a console-script GUI launcher
   (``npx-meta-to-coords``).

Settings and inputs
--------------------
The stable per-step tunables live in
``_parameter_settings/analyses_settings.json`` and are read once at the
top of the notebook; each section's acquisition paths and session
identifiers are passed in explicitly. The relevant blocks are:

- ``npx_histology_stack_lightsheet_volume`` — ``xy_flip``
  (``'none'`` / ``'vertical'`` / ``'horizontal'`` / ``'both'``),
  ``z_flip``, and ``skip_first`` (drops the LaVision ``Z0000`` plane,
  which alone carries the OME-XML header).
- ``npx_histology_stitch_smartspim_tiles`` — ``z_flip`` and
  ``feather_pixels`` (width of the linear feather ramp blended over tile
  seams).
- ``npx_histology_ibl_alignment_export`` — ``probe_to_hemisphere``, the
  per-lab convention mapping each ``imecN`` probe to the hemisphere it was
  inserted into (defaults ``imec0`` → right, ``imec1`` → left).
- ``npx_spike_quality_metrics`` — ``kilosort_version``, ``phy_curated``,
  ``num_channels_sparsity`` (channels per unit in the phy-peak-centred
  sparsity), ``shank_width_microns`` / ``shank_spacing_microns`` (used to
  fold the within-shank lateral offset into the anatomical AP axis),
  ``somatic_classifier`` (the peak/trough thresholds for the somatic /
  non-somatic decision), ``probe_to_hemisphere``, and ``job_kwargs``
  (``n_jobs`` is the main throughput knob for the single recording read).

All filesystem paths in the notebook are written ``/mnt/...`` and wrapped
in ``configure_path`` so they resolve on macOS (``/Volumes/...``) too.

.. _histology-lightsheet:

1. Light-sheet volume assembly
-------------------------------
Two acquisition modalities are supported. **LaVision UltraMicroscope**
acquisitions are a flat directory of OME-TIFF Z-planes per channel
(single tile); ``stack_lightsheet_volume`` glues the planes into one
BigTIFF per channel, optionally flipping each plane in-plane and reversing
the Z order. **LifeCanvas SmartSPIM** acquisitions are a tiled XY grid of
Z-stacks per channel under ``Ex_{wavelength}_Ch{n}/{X}/{X}_{Y}/``;
``stitch_smartspim_tiles`` reads ``metadata.txt``, converts the stage
coordinates (0.1 µm units) to pixel offsets, and streams a plane-by-plane
stitch with a bevel-shaped linear feather over the tile seams.

Both functions default to ``wavelength_nm=(488, 561)`` and process both
channels in one call. The ``output_path`` must contain a
``{wavelength_nm}`` placeholder whenever more than one wavelength is
requested; it is formatted per channel.

.. code-block:: python

    from usv_playpen.neuropixels.histology_stack_lightsheet_volume import (
        stack_lightsheet_volume,
    )
    from usv_playpen.neuropixels.histology_stitch_smartspim_tiles import (
        stitch_smartspim_tiles,
    )

    # LaVision: stack a flat directory of OME-TIFF Z-planes
    stack_lightsheet_volume(
        raw_dir="/mnt/lightsheet/.../251015_bmimica_178621-dv-lv-1_09-44-40",
        output_path="/mnt/falkner/.../178621_{wavelength_nm}nm_fullsize.tif",
        xy_flip="none",
        z_flip=False,
        skip_first=True,
    )

    # SmartSPIM: stitch a tiled acquisition
    stitch_smartspim_tiles(
        raw_dir="/mnt/lightsheet/.../20251118_..._181321_1x_vd_ss_1",
        output_path="/mnt/falkner/.../181321_{wavelength_nm}nm_stitched.tif",
        z_flip=True,
        feather_pixels=64,
    )

Two independent orientation knobs are set by trial-and-error per dataset:
``xy_flip`` flips each plane in-plane (it affects the **axial** view in
napari), while ``z_flip`` reverses the Z iteration order (it affects the
**coronal** and **sagittal** views). If the brain renders flipped in
napari: wrong in the axial view → tweak ``xy_flip``; right in axial but
upside-down in coronal and sagittal → toggle ``z_flip``.

.. _histology-ibl-export:

2. IBL ephys-alignment export (pre-alignment)
----------------------------------------------
Once the volumes are registered with brainreg and per-shank tracks are
traced in napari, the IBL ephys-alignment GUI anchors every recording
channel to an Allen CCF region. It needs two inputs per session / probe /
hemisphere: the per-shank **track points** in IBL mlapdv space, and an
**ALF dataset** of ``spikes.*`` / ``clusters.*`` / ``templates.*`` /
``channels.*`` arrays.

``IBLAlignmentExporter`` replicates these from the Kilosort directory plus
the SpikeGLX ``.ap.meta`` alone — with no raw-binary streaming and no
``iblatlas`` / ``ibllib`` / ``phylib`` / ``spikeglx`` dependency — in
seconds rather than the hours the upstream
``atlaselectrophysiology.extract_files.extract_data`` takes (it streams
the hundreds-of-GB concatenated AP binary to compute per-channel RMS maps
the GUI does not need). The exporter caches the parsed ``.ap.meta`` and
resolved paths at construction time, so its step methods can be called
independently.

.. code-block:: python

    from usv_playpen.neuropixels.histology_ibl_alignment_export import (
        IBLAlignmentExporter,
    )

    probe_to_hemisphere = {"imec0": "R", "imec1": "L"}  # from settings
    for probe_id, hemisphere in probe_to_hemisphere.items():
        exporter = IBLAlignmentExporter(
            os_cup_loc="/mnt/falkner/Bartul",
            mouse_id="164335_0",
            session_date="20250912",
            probe_id=probe_id,
            hemisphere=hemisphere,
            kilosort_version="4",
        )
        exporter.write_xyz_picks()  # xyz_picks_shank{n}.json per shank
        exporter.write_alf_outputs()  # full ALF layout in ibl_{H}H/

``write_xyz_picks`` converts each shank's brainreg track point cloud
(Allen CCF apdvml voxel-origin µm) into the IBL mlapdv (bregma-origin)
space the GUI loads — a pure affine that needs no NRRD volumes, hence no
``iblatlas`` download. ``write_alf_outputs`` copies the required Kilosort
files, then computes the spike-, template-, and cluster-level ALF arrays
(amplitude scaling, template unwhitening, peak-channel and
nearest-channel restriction), faithfully mirroring phylib's logic for the
dense case — including phy-curated sessions where ``spike_clusters`` differ
from ``spike_templates``.

.. note::

   **Run the IBL ephys-alignment GUI here**, outside this notebook. With
   each ``ibl_{hemisphere}H/`` directory populated, walk every shank's
   track through the GUI; it writes one ``channel_locations_shank{n}.json``
   per shank back into the same directory. Continue with the
   post-alignment step once those JSONs exist for every probe.

.. _histology-ibl-postalign:

3. IBL ephys-alignment export (post-alignment)
-----------------------------------------------
Two pure-JSON steps consume the GUI's per-shank output:

.. code-block:: python

    for probe_id, hemisphere in probe_to_hemisphere.items():
        exporter = IBLAlignmentExporter(
            os_cup_loc="/mnt/falkner/Bartul",
            mouse_id="181322_2",
            session_date="20251012",
            probe_id=probe_id,
            hemisphere=hemisphere,
        )
        exporter.remap_channel_ids_to_raw()  # per-shank 0..m-1 -> raw ids
        exporter.write_unified_channel_locations()  # one channel_locations.json

``remap_channel_ids_to_raw`` re-keys each per-shank JSON from the GUI's
per-shank channel indices (``channel_0`` .. ``channel_{m-1}``) back to raw
recording channel ids, using the IMRO table cached at construction (a
no-op on single-shank probes). ``write_unified_channel_locations`` then
merges the per-shank JSONs into a single ``channel_locations.json`` keyed
by raw channel id and sorted by integer index — the layout SpikeInterface
expects downstream.

.. _histology-spike-quality:

4. Spike quality metrics
-------------------------
With the unified ``channel_locations.json`` in place for every probe,
``SpikeQualityMetricsExtractor`` computes the per-unit quality-metrics
catalog on pinned stock ``spikeinterface==0.104.3``. It reads the
hundreds-of-GB recording **once**, in two passes: a recording-free *core
pass* for the spike-train metrics, and a single sequential *recording-
dependent pass* that extracts windowed waveforms for a uniform per-unit
random subsample and derives the template, amplitude, PCA and ``sd_ratio``
metrics from them (avoiding the second whole-recording stream the original
workflow used for ``spike_amplitudes``).

.. code-block:: python

    from usv_playpen.neuropixels.spike_quality_metrics import (
        SpikeQualityMetricsExtractor,
    )

    for probe_id, hemisphere in probe_to_hemisphere.items():
        extractor = SpikeQualityMetricsExtractor(
            os_cup_loc="/mnt/falkner/Bartul",
            mouse_id="158112_0",
            session_date="20241107",
            probe_id=probe_id,
            hemisphere=hemisphere,
            num_channels_sparsity=7,
            shank_width_microns=70,
            shank_spacing_microns=250,
            job_kwargs={"n_jobs": 16, "chunk_duration": "1s", "progress_bar": True},
        )
        catalog = extractor.run()

``run`` orchestrates the whole per-session pipeline (load → core pass →
recording-dependent pass → unit locations → per-shank channel order →
catalog) and is **idempotent**: rows already present for this
``mouse_id`` + ``rec_date`` + probe are dropped before the fresh rows are
appended, so re-processing a session updates it in place rather than
duplicating rows. Unit locations are estimated by 3D monopolar source
triangulation (``monopolar_triangulation``) restricted to the unit's
template-peak shank, with IBL anatomy looked up by physical electrode
position rather than by raw channel index. Pass ``overwrite=True`` to
recompute a session already in the catalog.

The phy-peak-centred channel sparsity (a SpikeInterface fork patch the
original workflow relied on) is reimplemented in ``spikeinterface_helpers``,
alongside the somatic / non-somatic single-channel classifier — a waveform
peak/trough shape rule (after Deligkaris et al. 2016) that flags a unit
non-somatic when a large, narrow positive peak precedes the main trough, and
records the peak/trough sizes, widths and ratios it rests on as catalog
columns. ``run`` writes the global ``EPHYS/unit_catalog.csv`` and a per-probe
``channel_order_per_shank.json``.

.. _channel-indexing:

Channel indexing
----------------

The same 384 Neuropixels electrodes are referred to by **four different
indexing conventions** across the files this pipeline produces and
consumes. Understanding which space each file lives in is essential
when joining IBL anatomy to Kilosort spike data, building the
channel-to-region converter, or interpreting unit names.

The four spaces
~~~~~~~~~~~~~~~

**Raw-meta channel id** (``imro_rows[k+1][0]``, also the numeric
suffix in ``channel_locations.json``'s ``channel_{i}`` keys).
The hardware channel id SpikeGLX assigned via the IMRO table.
For NP 2.0 4-shank (probe type 2013) the raw-meta-to-physical-
shank mapping is not monotonic — consecutive raw-meta channels
can sit on different shanks.

**Kilosort row index** (row of ``channel_positions.npy`` and
``channel_shanks.npy``, peak channel inside ``templates.npy``,
``spike_clusters.npy``).
The row in the recording after SpikeInterface applied the IMRO
permutation. Kilosort orders rows so that rows ``0..95`` sit on
physical shank 1, ``96..191`` on shank 2, ``192..287`` on shank 3,
``288..383`` on shank 4 (shank-major blocks). Within each shank
the axial order is *not* strictly monotonic — on shanks 1 and 2
of the NP 2.0 4-shank probes in this dataset, the KS rows step
through axial mid → top, then wrap to the bottom band — so when
the per-row anatomy is compressed into ``[lo, hi]`` KS-row ranges,
a single anatomical band (e.g. shank-1 ventral MRN) can land on
two non-contiguous KS-row intervals. Phy reads these sidecars
directly, so phy channel labels and unit file names like
``cl0017_ch042_good.npy`` are in this space.

**Phy channel** (peak channel of a unit as phy displays it).
For this codebase, this is the same as the Kilosort row index —
phy is reading the Kilosort outputs.

**Physical position** (``(lateral, axial)`` in microns, as it
appears in ``channel_positions.npy[i]`` and in every IBL JSON
entry's ``lateral``/``axial`` fields).
The actual electrode site, independent of indexing. Within each
shank, lateral is one of two values 27 µm apart; across shanks
the absolute lateral is offset by ``shank * 250 µm`` (4-shank
center-to-center spacing).

The bridges
~~~~~~~~~~~

The permutation between raw-meta and Kilosort row is stored
explicitly in ``channel_map.npy`` under each Kilosort directory::

    raw_meta_channel = channel_map.npy[KS_row]

Physical position is the universal bridge — for every electrode,
SpikeInterface's ``channel_positions.npy[KS_row]`` agrees byte-for-
byte with the ``lateral`` and ``axial`` fields inside the
corresponding IBL JSON entry. So::

    cp[i]              ==  (IBL[f"channel_{cm[i]}"]["lateral"],
                            IBL[f"channel_{cm[i]}"]["axial"])

This means **any join between a Kilosort artifact and an IBL artifact
can be done by position**, with no need to load ``channel_map.npy``.

Which file lives in which space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw-meta channel id space:

- SpikeGLX ``~imroTbl`` rows (``concatenated_<date>_imec<i>.ap.meta``).
- SpikeGLX ``~snsGeomMap`` rows (same meta file).
- IBL ``channel_locations.json`` and the per-shank
  ``channel_locations_shank{1..4}.json`` files.

Kilosort row space:

- ``channel_positions.npy``
- ``channel_shanks.npy``
- ``templates.npy`` (the last axis)
- ``spike_clusters.npy`` peak channels
- ``channel_order_per_shank.json`` (after the snsGeomMap-based fix)
- ``unit_catalog.csv`` ``closest_ch`` column
- Unit file names like ``cl0017_ch042_good.npy``
- The regenerated ``neuropixels_sites_to_anatomy_converter.json`` (see
  below)

Physical position space:

- The brain-coord columns of ``unit_catalog.csv`` (``loc_ap``,
  ``loc_ml``, ``loc_dv``) and the Allen Bregma µm of every histology
  output — these are not channel-indexed at all.

Why Kilosort row was chosen as the canonical space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most artifacts that any downstream consumer touches — the catalog,
unit files, the per-shank channel order JSON, templates — already
live in Kilosort row space. The only artifact that historically
broke that convention was the channel-to-region anatomy converter,
which the original generator wrote with raw-meta channel ranges.

Since downstream consumers (e.g.
``make_behavioral_videos.find_region_by_channel``) read unit names
or catalog values and pass Kilosort row numbers into converter
membership checks, the easiest path to correctness is to regenerate
the converter into Kilosort row space. After that, every consumer
that already treated the converter as KS-keyed begins returning the
right region without any code change.

Translating between spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you ever need to go from a KS row to a raw-meta channel (e.g. to
look up an unprocessed IBL JSON entry directly), use::

    cm = np.load(ks_dir / "channel_map.npy").flatten()
    raw_meta_ch = int(cm[ks_row])
    ibl_entry = ibl_json[f"channel_{raw_meta_ch}"]

The inverse (raw-meta to KS row) is::

    ks_row = int(np.argwhere(cm == raw_meta_ch)[0, 0])

For any case where you just want to know "what is at this physical
position", join on ``cp[i]`` against the IBL entry's
``lateral``/``axial`` fields and skip the index translation.

.. _histology-utilities:

Anatomy-converter utility
---------------------------
The sites-to-anatomy converter (:mod:`usv_playpen.neuropixels.anatomy_converter`)
maps Kilosort-row index ranges to brain regions. The original generator wrote it
with raw-meta channel ranges; it is regenerated into Kilosort-row space (see
:ref:`channel-indexing`) so that downstream consumers passing Kilosort row numbers
into membership checks return the right region without any code change.

For every ``(mouse, session, probe)`` triple already present in the converter:

1. Load the IBL ``channel_locations.json`` for the appropriate hemisphere
   (``imec0`` → right, ``imec1`` → left in this dataset), and build a
   ``(lateral, axial) -> brain_region`` lookup.
2. Load ``channel_positions.npy`` from the Kilosort directory.
3. For every Kilosort row ``i``, the row's region is the IBL region at the
   physical position ``cp[i]``.
4. Compress contiguous runs of identical regions into ``[lo, hi]`` half-open
   ranges per region.

The generated entries are written back to
``neuropixels_sites_to_anatomy_converter.json`` in the same nested
``{mouse: {session: {probe: {region: [[lo, hi], ...]}}}}`` layout. Because
Kilosort row ordering is shank-major, each probe's regenerated entry has every
range bounded inside one shank's KS-row block (rows 0..95, 96..191, etc.); the
within-shank axial ordering is not always monotonic, so a single anatomical band
on a shank may appear as two non-contiguous ``[lo, hi]`` KS-row intervals — set
membership still resolves the right region regardless.

It is exposed as a ``python -m`` module CLI that defaults to a dry run via
``--dry-run`` and prints a JSON summary. Its path flags default to the paths in
``analyses_settings.json`` under ``data_roots`` (translated to the host OS via
``configure_path``), so pass them only to override:

.. code-block:: bash

    # Regenerate the sites-to-anatomy converter so its per-region ranges
    # are keyed by Kilosort-row index (see the Channel indexing section).
    $ python -m usv_playpen.neuropixels.anatomy_converter --dry-run \
        --converter-path /mnt/falkner/Bartul/EPHYS/neuropixels_sites_to_anatomy_converter.json \
        --ephys-root /mnt/falkner/Bartul/EPHYS \
        --histology-root /mnt/falkner/Bartul/histology

.. _histology-meta-to-coords:

SpikeGLX meta → probe-geometry converter
-----------------------------------------
``sglx_meta_to_coords`` converts a SpikeGLX ``.ap.meta`` file into a
per-channel geometry artifact for a downstream sorter — a Kilosort
``chanMap.mat``, a JRClust ``.prm`` string set, a ``(n_channels, 2)``
``.npy``, a plain tab-delimited file, or an in-place upgrade of a legacy
(pre-SpikeGLX 032623) meta. It is a clean-room reimplementation built from
the public SpikeGLX / Imec documentation. Run it as a GUI via the
``npx-meta-to-coords`` console script (see :ref:`CLI`), or programmatically:

.. code-block:: python

    from pathlib import Path
    from usv_playpen.neuropixels.sglx_meta_to_coords import (
        OutputFormat,
        parse_spikeglx_meta,
        coords_from_meta,
        write_coords_file,
    )

    meta = parse_spikeglx_meta(Path("/path/to/run.imec0.ap.meta"))
    coords = coords_from_meta(meta)  # auto-picks snsGeomMap vs snsShankMap
    write_coords_file(
        meta=meta,
        coords=coords,
        output_format=OutputFormat.KILOSORT_MAT,
        save_dir=Path("/some/dir"),
        base_name="run.imec0.ap",
    )

.. _histology-notebook:

Interactive notebook
---------------------
The ``npx_histology_unit_quality_processing.ipynb`` notebook is the
recommended entry point — it runs the whole workflow above in order from a
single **Parameters** cell. Its detailed walkthrough, knobs, and rendered
source live in :doc:`Notebooks`.
