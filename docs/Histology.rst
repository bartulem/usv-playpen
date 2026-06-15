.. _Histology:

Histology
==================
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
consumes — see :doc:`channel_indexing` for the full landscape, which is
essential when joining IBL anatomy to Kilosort spike data.

.. note::

   These pipelines are **not** exposed as a GUI tab. Run them from the
   :ref:`histology-notebook` (interactive, single node, linked at the
   bottom of this page) or by importing the classes directly. The
   :ref:`SpikeGLX meta → geometry converter <histology-meta-to-coords>`
   below is the one piece with a console-script GUI launcher
   (``npx-meta-to-coords``).

Settings and inputs
^^^^^^^^^^^^^^^^^^^^
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
  ``probe_to_hemisphere``, and ``job_kwargs`` (``n_jobs`` is the main
  throughput knob for the single recording read).

All filesystem paths in the notebook are written ``/mnt/...`` and wrapped
in ``configure_path`` so they resolve on macOS (``/Volumes/...``) too.

.. _histology-lightsheet:

1. Light-sheet volume assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^
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

The two SpikeInterface fork patches the original workflow relied on — a
phy-peak-centred channel sparsity and the somatic / non-somatic
single-channel classifier — are reimplemented in
``spikeinterface_helpers``; ``run`` writes the global
``EPHYS/unit_catalog.csv`` and a per-probe
``channel_order_per_shank.json``.

.. _histology-utilities:

Converter and catalog-patch utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Two maintenance utilities are exposed as ``python -m`` module CLIs. Both
default to a dry run via ``--dry-run`` and print a JSON summary:

.. code-block:: bash

    # Regenerate the sites-to-anatomy converter so its per-region ranges
    # are keyed by Kilosort-row index (see :doc:`channel_indexing`).
    $ python -m usv_playpen.neuropixels.anatomy_converter --dry-run \
        --converter-path /mnt/falkner/Bartul/EPHYS/neuropixels_sites_to_anatomy_converter.json \
        --ephys-root /mnt/falkner/Bartul/EPHYS \
        --histology-root /mnt/falkner/Bartul/histology

    # Re-triangulate each catalog unit restricted to its template-peak
    # shank, patching closest_ch / brain_area / loc_* in unit_catalog.csv.
    $ python -m usv_playpen.neuropixels.patch_unit_catalog_peak_channel --dry-run \
        --catalog-path /mnt/falkner/Bartul/EPHYS/unit_catalog.csv \
        --ephys-root /mnt/falkner/Bartul/EPHYS \
        --histology-root /mnt/falkner/Bartul/histology

.. _histology-meta-to-coords:

SpikeGLX meta → probe-geometry converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^
The ``npx_histology_unit_quality_processing.ipynb`` notebook is the
recommended entry point. It runs the whole workflow above in order — every
acquisition path and session identifier lives in a single **Parameters**
cell near the top (grouped by section), so a run is configured in one
place. The full notebook lives in the repository at
`npx_histology_unit_quality_processing.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/analyses_notebooks/npx_histology_unit_quality_processing.ipynb>`_.
