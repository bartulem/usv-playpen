.. _Neuropixels:

Neuropixels
===========
This page explains how to use the **Neuropixels histology and
unit-quality pipelines** in ``usv_playpen.neuropixels``. Where the
:ref:`Process` section ends with spike-sorted, session-split clusters,
this subsystem brackets the two manual steps you run outside the GUI —
probe-track registration in *napari* (with the *brainreg* and
*brainglobe-segmentation* plugins) and channel alignment in the International Brain Laboratory (IBL)
ephys-alignment GUI. It assembles raw light-sheet microscopy into registrable
volumes, bridges *Kilosort* output and the traced probe tracks into the format
that GUI expects, and finally distils every unit into a single quality-metrics
catalog anchored to the Allen Common Coordinate Framework (CCF, the Allen atlas).

Step 0 is done at acquisition time — it builds the probe geometry, spike-sorts
the recording with *kilosort4*, and manually curates the result in *Phy2* (each in
its own conda environment). The histology and unit-quality workflow itself then
runs in seven steps (1–7) — five automated (run from the notebook) and two manual
(run outside it, steps 2 and 4), interleaved in order:

.. list-table::
   :header-rows: 1
   :widths: 5 32 34 29

   * - #
     - Step
     - Produces
     - Driver
   * - 0
     - Probe geometry + spike sorting *(at acquisition)*
     - curated ``kilosort4/`` clusters (via the chanMap)
     - ``sglx_meta_to_coords`` → *kilosort4* → *Phy2*
   * - 1
     - Light-sheet volume assembly
     - one BigTIFF volume per channel
     - ``stack_lightsheet_volume`` / ``stitch_smartspim_tiles``
   * - 2
     - Register probe tracks *(manual)*
     - registered volume + per-shank track ``.npy`` files
     - *napari*: ``brainreg`` (registration), ``brainglobe-segmentation`` (tracing)
   * - 3
     - IBL ephys pre-alignment export
     - ``xyz_picks_shank{n}.json`` + ALF layout
     - ``IBLAlignmentExporter``
   * - 4
     - IBL ephys channel alignment *(manual)*
     - one ``channel_locations_shank{n}.json`` per shank
     - IBL ephys-alignment GUI (``iblapps``)
   * - 5
     - IBL ephys post-alignment export
     - unified ``channel_locations.json``
     - ``IBLAlignmentExporter``
   * - 6
     - Channel-brain area converter
     - KS-row-keyed regions merged into ``…anatomy_converter.json``
     - ``anatomy_converter`` (notebook + CLI)
   * - 7
     - Spike quality metrics
     - ``unit_catalog.csv`` + per-shank channel order
     - ``SpikeQualityMetricsExtractor``

Only **Neuropixels 2.0** probes are supported; anything else raises
``NotImplementedError``. Both one- and two-probe setups work — ``imec0``
alone, or ``imec0`` and ``imec1`` — as every step runs once per probe listed in
``probe_to_hemisphere``. The same 384 electrodes are referred to by two
different indexing conventions across the files this pipeline produces and
consumes — see the :ref:`channel-numbering conventions <channel-indexing>` section below for the full landscape,
which is essential when joining IBL anatomy to *Kilosort* spike data.

.. note::

   These pipelines are **not** exposed as a GUI tab. Run them from the
   :ref:`npx_histology_unit_quality_processing.ipynb <histology-notebook>`
   (interactive, single node, linked at the bottom of this page) or by
   importing the classes directly. The
   :ref:`SpikeGLX meta to geometry converter <histology-meta-to-coords>`
   (Step 0) is the one piece with a console-script GUI launcher
   (``npx-meta-to-coords``).

Settings and inputs
--------------------
The stable per-step tunables live in
``_parameter_settings/analyses_settings.json`` and are read once at the
top of the notebook; each section's acquisition paths and session
identifiers are passed in explicitly. Each step's block — its keys, an example
JSON, and where the step writes its outputs — is documented in that step's own
section below: ``npx_histology_stack_lightsheet_volume`` /
``npx_histology_stitch_smartspim_tiles`` (step 1),
``npx_histology_ibl_alignment_export`` (steps 3 and 5), and
``npx_spike_quality_metrics`` (step 7).

All filesystem paths in the notebook are written in Linux form (``/mnt/...``)
and wrapped in ``configure_path``, which rewrites the leading mount root for the
host OS — so the same path also resolves on macOS (``/Volumes/...``) and Windows
(a mapped drive, e.g. ``F:\...``).

.. _histology-meta-to-coords:

0. Probe geometry and spike sorting
-----------------------------------
Run this **prior to spike sorting**, whenever you are at that stage: the geometry
file it produces (a *Kilosort* ``chanMap.mat``) is what *Kilosort* needs to
spike-sort the recording — the step you typically run while the animal is still
alive, long before the brain is extracted for light-sheet imaging.

``sglx_meta_to_coords`` converts a SpikeGLX ``.ap.meta`` file into a
per-channel geometry artifact for a downstream sorter — a *Kilosort*
``chanMap.mat``, a JRClust ``.prm`` string set, a ``(n_channels, 2)``
``.npy``, a plain tab-delimited file, or an in-place upgrade of a legacy
(pre-SpikeGLX 032623) meta. Run it headlessly
(``npx-meta-to-coords --meta-file <meta>``) or as an interactive GUI (with no
arguments) via the ``npx-meta-to-coords`` console script (see :ref:`CLI`), or
programmatically:

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

With the chanMap in hand, spike-sort the recording with `kilosort4
<https://github.com/MouseLand/kilosort>`_ and manually curate the output in `Phy2
<https://github.com/cortex-lab/phy>`_ — both at acquisition time, each in its
**own dedicated conda environment** (kilosort4 needs a CUDA PyTorch build, Phy2 its
own PyQt stack; separate from *usv-playpen* and from each other). *usv-playpen* runs
neither — it only consumes the resulting curated ``kilosort4/`` clusters. A
ready-to-edit runner for the sorting step is at
``src/usv_playpen/other/kilosort/run_kilosort.py``.

.. _histology-lightsheet:

1. Light-sheet volume assembly
-------------------------------
Two acquisition modalities are supported. **LaVision UltraMicroscope**
acquisitions are a flat directory of Open Microscopy Environment TIFF (OME-TIFF) Z-planes per channel
(single tile); ``stack_lightsheet_volume`` glues the planes into one
BigTIFF (the BigTIFF large-image format) per channel, optionally mirroring each plane and reversing the Z
order. **LifeCanvas SmartSPIM** light-sheet microscope acquisitions are a tiled XY grid of Z-stacks
per channel under ``Ex_{wavelength}_Ch{n}/{X}/{X}_{Y}/``, where the ``X`` / ``Y``
tokens are each tile's **stage position** (in 0.1 µm units). To assemble the tiles
into one image, ``stitch_smartspim_tiles`` needs to know where each lands in
pixels: it reads the pixel size (µm per pixel) from the acquisition's own
``metadata.txt``, converts every tile's stage position to a **pixel offset** on a
common canvas (stage µm ÷ pixel size), and streams a plane-by-plane stitch with a
bevel-shaped linear feather over the tile seams. That ``metadata.txt`` ships with
the SmartSPIM acquisition and is required — the run fails if the pixel size cannot
be read from it.

Both functions default to ``wavelength_nm=(488, 561)`` — the two acquired
channels: ``488`` nm (autofluorescence, ``C00``, used to register the brain to
the atlas in Step 2) and ``561`` nm (excitation, ``C01``, the CM-DiI dye channel
that carries the probe tracks) — and process both in one call. The ``output_path`` must contain a
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

Two independent orientation knobs are set by trial-and-error per dataset.
``xy_flip`` mirrors each Z-plane image — ``'vertical'``, ``'horizontal'`` or
``'both'`` (``'none'`` leaves it as acquired) — and affects the **axial** view in
*napari*; ``z_flip`` reverses the Z iteration order and affects the **coronal**
and **sagittal** views. If the brain renders flipped in *napari*: wrong in the
axial view → try the other ``xy_flip`` orientations (``'vertical'`` /
``'horizontal'`` / ``'both'``); right in axial but upside-down in coronal and
sagittal → toggle ``z_flip``.

Each function writes one BigTIFF per wavelength, expanding the ``{wavelength_nm}``
placeholder in ``output_path`` (defaults 488 + 561 nm) — so one call produces two
files at the chosen ``output_path``:

.. parsed-literal::

    /mnt/falkner/.../
    ├── **178621_488nm_fullsize.tif**    (LaVision, via ``stack_lightsheet_volume``)
    └── **178621_561nm_fullsize.tif**

    /mnt/falkner/.../
    ├── **181321_488nm_stitched.tif**    (SmartSPIM, via ``stitch_smartspim_tiles``)
    └── **181321_561nm_stitched.tif**

The tunables live in ``analyses_settings.json`` (read at the top of the notebook and
passed to the calls above).

``npx_histology_stack_lightsheet_volume``:

* **xy_flip** : mirror each Z-plane image — ``'none'`` / ``'vertical'`` / ``'horizontal'`` / ``'both'`` (affects the axial view in *napari*).
* **z_flip** : reverse the Z iteration order (affects the coronal / sagittal views).
* **skip_first** : drop the LaVision ``Z0000`` plane, which alone carries the OME-XML header.

``npx_histology_stitch_smartspim_tiles``:

* **z_flip** : reverse the Z iteration order (as above, for the SmartSPIM path).
* **feather_pixels** : width (px) of the linear feather ramp blended over the tile seams.

.. code-block:: json

    "npx_histology_stack_lightsheet_volume": {
        "xy_flip": "none",
        "z_flip": false,
        "skip_first": true
    },
    "npx_histology_stitch_smartspim_tiles": {
        "z_flip": true,
        "feather_pixels": 64
    }

.. _histology-registration:

2. Register probe tracks to anatomy
------------------------------------
This **manual, GUI-driven** step runs outside *usv-playpen*, between the automated
volume assembly (Step 1) and the IBL export (Step 3). It takes the light-sheet BigTIFFs
from Step 1 and produces, per shank, a probe-track point cloud (``.npy``) in Allen CCF
space — the input the exporter converts to ``xyz_picks`` in Step 3. Neuropixels 2.0
geometry is assumed: one track per shank (eight for a dual-probe implant).

**Install napari and the BrainGlobe plugins.** `napari
<https://github.com/napari/napari>`_ (`napari.org <https://napari.org>`_) is the image
viewer; registration and track tracing are BrainGlobe plugins that run inside it:

.. code-block:: bash

    conda create -y -n napari-env -c conda-forge python=3.11
    conda activate napari-env
    conda install -c conda-forge napari pyqt pytables
    # conda install -c conda-forge hdf5     # Apple-silicon macOS only
    conda update napari
    napari

In napari, open **Plugins → Install/Uninstall Plugins** and install `brainreg
<https://github.com/brainglobe/brainreg-napari>`_ (registration), `brainglobe-segmentation
<https://github.com/brainglobe/brainglobe-segmentation>`_ (track tracing), and
``brainglobe-napari-io`` (I/O); restart napari afterwards. `brainrender
<https://github.com/brainglobe/brainrender>`_ is optional (only the "To brainrender"
preview uses it), and the shared `brainglobe-utils
<https://github.com/brainglobe/brainglobe-utils>`_ library is pulled in automatically as a
dependency — no separate install.

**Load the volumes.** Drag the Step-1 ``C00`` (488 nm autofluorescence) and ``C01``
(561 nm excitation) BigTIFFs into napari, choosing the **default napari loader** (not the
BrainGlobe one). The lower-left axis widget switches between CORONAL / SAGITTAL / AXIAL
views ("change order of the visible axes"; "transpose the last two" tilts the image and is
less useful).

**Register with brainreg.** Open **Atlas Registration (brainreg)** (`plugin user guide
<https://brainglobe.info/documentation/brainreg/user-guide/brainreg-napari.html>`_) and
first run **Check
orientation** to find the data orientation (``sar`` for our LaVision acquisitions).
Register the ``C00`` layer with only ``C01`` selected on the right, so the transform is
applied to ``C01`` as well, and set an **output directory**. As an example, the parameters
we use ("damaged brain" variants in parentheses):

* data orientation ``sar``; brain geometry: full brain
* voxel size z / x / y — ``10.00`` / ``5.91`` / ``5.91`` µm
* affine downsampling steps, calculate / use — ``6`` / ``5``  (``4`` / ``3``)
* freeform downsampling steps, calculate / use — ``6`` / ``4``  (``4`` / ``2``)
* bending-energy weight — ``0.95``  (``0.9``)
* grid spacing — ``-15``  (``-10``)

Everything else is left at default (the atlas is also selectable).

**Trace shank tracks with brainglobe-segmentation.** Open **brainglobe-segmentation → Load
project (ATLAS space)** (`track-tracing tutorial
<https://brainglobe.info/tutorials/segmenting-1d-tracks.html>`_), then **Track tracing**: drop the registered-image opacity to
~0.75, enable continuous auto-contrast on the CM-DiI channel, and reduce the spline points
to under 100 (this sets how many times along each track the region is sampled). Add one
track per shank (**Add track** — eight for a dual 2.0 implant), select a track, and place
points from top to bottom along that shank (a ruler helps if the tissue is distorted).
**Trace tracks**, then **To brainrender** when satisfied; screenshot each tracked probe.

.. note::

   **Which hemisphere is which** — the brain can render mirrored, and if the implant
   strategy is identical across hemispheres it is hard to tell by eye which hemisphere is
   left and which one is right. The best strategy to avoid confusion is to leave a small
   mark or cut on one of the hemispheres post-perfusion and note it down for reference.
   Because the 2.0 probes are inserted in a way where contacts face each other, the anterior → posterior shank order is
   ``L_track_3, L_track_2, L_track_1, L_track_0`` for the left hemisphere and
   ``R_track_0, R_track_1, R_track_2, R_track_3`` for the right (per the Neuropixels 2.0
   user manual).

The traced per-shank ``.npy`` point clouds (Allen CCF apdvml, the anterior-posterior/dorsal-ventral/medial-lateral axis order, voxel-origin µm) are exactly
what Step 3's ``write_xyz_picks`` converts into the ``xyz_picks_shank{n}.json`` files the
IBL alignment GUI loads.

.. _histology-ibl-export:

3. IBL ephys pre-alignment export
---------------------------------
Once the volumes are registered with *brainreg* and per-shank tracks traced
with *brainglobe-segmentation* (Step 2), the IBL ephys-alignment GUI anchors every recording
channel to an Allen CCF region. It needs two inputs per session / probe /
hemisphere: the per-shank **track points** in IBL mlapdv (the medial-lateral/anterior-posterior/dorsal-ventral axis order) space, and an
**ALF dataset** (the IBL's standardized ALF data layout) of ``spikes.*`` / ``clusters.*`` / ``templates.*`` /
``channels.*`` arrays.

``IBLAlignmentExporter`` replicates these from the *Kilosort* directory plus
the SpikeGLX ``.ap.meta`` alone — with no raw-binary streaming and no
``iblatlas`` / ``ibllib`` / ``phylib`` / ``spikeglx`` dependency. The
exporter caches the parsed ``.ap.meta`` and
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

``write_xyz_picks`` converts each shank's *brainglobe-segmentation* track point cloud
(Allen CCF apdvml voxel-origin µm) into the IBL mlapdv (bregma-origin)
space the GUI loads — a pure affine that needs no NRRD volumes, hence no
``iblatlas`` download. ``write_alf_outputs`` copies the required *Kilosort*
files, then computes the spike-, template-, and cluster-level ALF arrays
(amplitude scaling, template unwhitening, peak-channel and
nearest-channel restriction), faithfully mirroring phylib's logic for the
dense case — including *phy*-curated sessions where ``spike_clusters`` differ
from ``spike_templates``.

The pre-alignment steps populate one ``ibl_{hemisphere}H`` directory per probe with
the GUI's two inputs — the per-shank track points and the ALF dataset:

.. parsed-literal::

    <histology>/164335_0/20250912/
    └── **ibl_RH**/                          (one per probe + hemisphere)
        ├── **xyz_picks_shank1.json**, …     (one per shank — ``write_xyz_picks``)
        ├── **the ALF dataset**              (spikes / clusters / templates / channels arrays — ``write_alf_outputs``)
        └── channel_locations_shank1.json    (written *later* by the IBL GUI, one per shank)

The probe → hemisphere map is read from the ``npx_histology_ibl_alignment_export``
block of ``analyses_settings.json`` (and reused by the post-alignment and
quality-metrics steps):

* **probe_to_hemisphere** : per-lab convention mapping each ``imecN`` probe to the hemisphere it was inserted into (defaults ``imec0`` → ``R``, ``imec1`` → ``L``).

.. code-block:: json

    "npx_histology_ibl_alignment_export": {
        "probe_to_hemisphere": {
            "imec0": "R",
            "imec1": "L"
        }
    }

.. _histology-ibl-gui:

4. IBL ephys channel alignment
------------------------------
This is the second **manual** step, run outside *usv-playpen*. With each
``ibl_{hemisphere}H/`` directory populated by Step 3, you walk every shank's track
through the IBL ephys-alignment GUI; it writes one ``channel_locations_shank{n}.json``
per shank back into that directory. Continue to Step 5 once those JSONs exist for every
probe.

**Install iblapps** in its own environment (its dependencies conflict with the main
project):

.. code-block:: bash

    conda update -n base -c defaults conda
    conda create --name iblenv python=3.10 --yes
    conda activate iblenv
    git clone https://github.com/int-brain-lab/iblapps
    pip install --editable iblapps

The `iblapps <https://github.com/int-brain-lab/iblapps>`_ environment can be brittle (a
colormap call may need a small in-place edit). For **Neuropixels 2.0** the GUI auto-detects
the shank count from ``channels.localCoordinates`` and offers a shank dropdown; the only
extra requirement over NP1 is one ``xyz_picks_shank{n}.json`` per shank — which Step 3
already wrote.

**Launch the GUI** and load the data (see the `usage instructions
<https://github.com/int-brain-lab/iblapps/wiki/2.-Usage-instructions>`_):

.. code-block:: bash

    conda activate iblenv
    cd <path-to>/iblapps/atlaselectrophysiology
    python ephys_atlas_gui.py -o True

In the GUI, click the **"..."** button (top-right, in place of the subject dropdown),
navigate to the probe's ``ibl_{hemisphere}H/`` folder, choose the shank (``1/4`` etc.), and
click **Get Data**. Align each shank's electrophysiology features to the anatomy — you can
**scale** regions where there is a tissue void, otherwise leave as-is — then click
**Upload** to write ``channel_locations_shank{n}.json``.

.. note::

   **Atlas-id sign convention** — the GUI writes regions in the left hemisphere with
   *negative* atlas ids and those in the right hemisphere with *positive* atlas ids; Step 5
   preserves this when it merges the per-shank JSONs.

.. _histology-ibl-postalign:

5. IBL ephys post-alignment export
----------------------------------
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
recording channel ids, using the probe's IMEC readout (IMRO) table cached at construction (a
no-op on single-shank probes). ``write_unified_channel_locations`` then
merges the per-shank JSONs into a single ``channel_locations.json`` keyed
by raw channel id and sorted by integer index — the layout *SpikeInterface*
expects downstream.

The post-alignment steps write one unified ``channel_locations.json`` per probe
alongside the GUI's per-shank inputs; they reuse the same ``probe_to_hemisphere``
map as the pre-alignment step (no separate settings):

.. parsed-literal::

    <histology>/181322_2/20251012/ibl_LH/
    ├── channel_locations_shank1.json, …    (input — from the IBL GUI, one per shank)
    └── **channel_locations.json**          (output — unified, raw-channel-id keyed for *SpikeInterface*)

.. _histology-utilities:

6. Channel-brain area converter
-------------------------------
This step folds the session's per-probe brain-region map into the global
channel-brain area converter (:mod:`usv_playpen.neuropixels.anatomy_converter`),
which maps *Kilosort*-row index ranges to brain regions. It is keyed by the
*Kilosort*-row index (see :ref:`channel-numbering conventions <channel-indexing>`)
so that downstream consumers passing *Kilosort* row numbers into membership checks
return the right region.

Run per session from the notebook via ``add_session_to_anatomy_converter`` — it is
**add-if-missing**: a ``(mouse, session, probe)`` already in the converter is left
untouched, and setting ``ANATOMY_FORCE = True`` re-regenerates an existing block.
The same work is also available as a batch ``python -m`` CLI (below) that rewrites
every triple already in the file.

For each ``(mouse, session, probe)``:

1. Load the IBL ``channel_locations.json`` for the appropriate hemisphere
   (``imec0`` → right, ``imec1`` → left in this dataset), and build a
   ``(lateral, axial) -> brain_region`` lookup.
2. Load ``channel_positions.npy`` from the *Kilosort* directory.
3. For every *Kilosort* row ``i``, the row's region is the IBL region at the
   physical position ``cp[i]``.
4. Compress contiguous runs of identical regions into ``[lo, hi]`` half-open
   ranges per region.

The block is merged back into ``neuropixels_sites_to_anatomy_converter.json`` in
the same nested ``{mouse: {session: {probe: {region: [[lo, hi], ...]}}}}`` layout
— **every other mouse / session / probe is preserved**, and the compact
one-line-per-region formatting is kept. Because *Kilosort* row ordering is
shank-major, each probe's entry has every range bounded inside one shank's Kilosort (KS)-row
block (rows 0..95, 96..191, etc.); the within-shank axial ordering is not always
monotonic, so a single anatomical band on a shank may appear as two non-contiguous
``[lo, hi]`` KS-row intervals — set membership still resolves the right region
regardless. One probe's entry looks like:

.. code-block:: json

    {
        "158112_0": {
            "20241107_114630": {
                "imec0": {
                    "PAG": [[0, 40], [72, 136], [174, 192], [246, 288], [346, 384]],
                    "MRN": [[40, 72], [136, 174], [192, 246], [288, 346]]
                }
            }
        }
    }

Here periaqueductal gray (PAG) and midbrain reticular nucleus (MRN) interleave — each is several non-contiguous KS-row ranges (the
shank-major wrap above) — yet a membership test on any KS row still returns the
right region.

For bulk maintenance — rebuilding **every** existing entry at once after a change
to the region-joining logic, which the per-session notebook step cannot do — the
same module is also a batch CLI. Run with **no action** it just prints usage and
writes nothing; the destructive whole-file rebuild is gated behind an explicit
``--regenerate-all`` (add ``--dry-run`` to preview the counts and which triples
would skip). Its path flags default to the ``data_roots`` block of
``analyses_settings.json`` (host-OS translated via ``configure_path``), so you
normally pass none:

.. code-block:: bash

    # Preview a full rebuild — reports counts + skips, writes nothing:
    $ python -m usv_playpen.neuropixels.anatomy_converter --regenerate-all --dry-run

    # Actually rewrite every triple, keyed by Kilosort-row index:
    $ python -m usv_playpen.neuropixels.anatomy_converter --regenerate-all

    # Add or refresh a single triple instead (--force to overwrite an existing one):
    $ python -m usv_playpen.neuropixels.anatomy_converter \
        --mouse 158112_0 --session 20241107_114630 --probe imec0

.. _histology-spike-quality:

7. Spike quality metrics
-------------------------
With the unified ``channel_locations.json`` in place for every probe,
``SpikeQualityMetricsExtractor`` computes the per-unit quality-metrics
catalog on pinned stock ``spikeinterface==0.104.3``. It reads the
hundreds-of-GB recording **once**, in two passes: a recording-free *core
pass* for the spike-train metrics, and a single sequential *recording-
dependent pass* that extracts windowed waveforms for a uniform per-unit
random subsample and derives the template, amplitude, principal component analysis (PCA) and ``sd_ratio``
metrics from them; the amplitude metrics come from those windowed waveforms, so
``spike_amplitudes`` (which would re-stream the whole recording per spike) is not
computed.

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

The *phy*-peak-centred channel sparsity is implemented in
``spikeinterface_helpers``,
alongside the somatic / non-somatic single-channel classifier — a waveform
peak/trough shape rule (after Deligkaris et al. 2016) that flags a unit
non-somatic when a large, narrow positive peak precedes the main trough, and
records the peak/trough sizes, widths and ratios it rests on as catalog
columns. ``run`` writes the global ``EPHYS/unit_catalog.csv`` and a per-probe
``channel_order_per_shank.json``.

.. parsed-literal::

    <EPHYS>/
    ├── **unit_catalog.csv**                 (global — every session's units; idempotent)
    └── 20241107_imec0/
        └── **channel_order_per_shank.json** (per probe)

The metrics tunables live in the ``npx_spike_quality_metrics`` block of
``analyses_settings.json``:

* **kilosort_version** : which ``kilosort{N}`` subdirectory to read.
* **phy_curated** : whether the session was curated in *phy* (``spike_clusters`` differ from ``spike_templates``).
* **num_channels_sparsity** : channels per unit in the *phy*-peak-centred sparsity.
* **shank_width_microns** / **shank_spacing_microns** : geometry used to fold the within-shank lateral offset into the anatomical AP axis.
* **probe_to_hemisphere** : the same probe → hemisphere map as the alignment step.
* **job_kwargs** : *SpikeInterface* job settings — ``n_jobs`` (the main throughput knob for the single recording read), ``chunk_duration``, ``progress_bar``.

The ``somatic_classifier`` sub-block holds the waveform peak/trough shape thresholds
for the somatic / non-somatic decision (after Deligkaris et al. 2016 — a unit reads as
non-somatic when a large, narrow positive peak precedes the main trough):

* **min_trough_to_peak2_ratio** (``5.0``) — non-somatic requires the main-trough-to-pre-trough-peak amplitude ratio *below* this.
* **min_width_first_peak** (``4``) / **min_width_main_trough** (``5``) — non-somatic requires the pre-trough peak and main-trough half-prominence widths (in samples) *below* these.
* **max_peak1_to_peak2_ratio** (``3.0``) — non-somatic requires the pre-trough-to-post-trough peak amplitude ratio *above* this.
* **max_main_peak_to_trough_ratio** (``0.8``) — a unit is non-somatic outright when its largest positive peak exceeds this fraction of the trough amplitude.
* **standard_spike_width** (``61``) — reference template length (samples) the two width thresholds scale against.

.. code-block:: json

    "npx_spike_quality_metrics": {
        "kilosort_version": 4,
        "phy_curated": true,
        "num_channels_sparsity": 7,
        "shank_width_microns": 70,
        "shank_spacing_microns": 250,
        "somatic_classifier": {
            "min_trough_to_peak2_ratio": 5.0,
            "min_width_first_peak": 4,
            "min_width_main_trough": 5,
            "max_peak1_to_peak2_ratio": 3.0,
            "max_main_peak_to_trough_ratio": 0.8,
            "standard_spike_width": 61
        },
        "probe_to_hemisphere": {
            "imec0": "R",
            "imec1": "L"
        },
        "job_kwargs": {
            "n_jobs": 16,
            "chunk_duration": "1s",
            "progress_bar": true
        }
    }

Every unit becomes one row of ``unit_catalog.csv``. Its columns, by group:

*Identity & provenance*

* **rec_date** — recording date (``YYYYMMDD``).
* **mouse_id** — animal id.
* **rec_sessions** — the session(s) concatenated for joint sorting.
* **probe_sn** / **hs_sn** — Neuropixels probe and headstage serial numbers.
* **kilosort_version** — the ``kilosort{N}`` version the unit came from.
* **phy_curated** — whether the session was manually curated in *Phy2*.
* **unit_id** — the unit label (``imecN_cl{cluster}_ch{ch}_{group}``).
* **cluster_group** — the sorter / *Phy* label (``good`` / ``mua`` / ``noise``).

*Classification & anatomy*

* **somatic** — the somatic / non-somatic call (``True`` = somatic; the rule described above).
* **spiking_profile** — reserved placeholder (``NaN`` here; filled by a downstream step).
* **loc_ap** / **loc_ml** / **loc_dv** — the unit's 3-D location (Allen CCF AP / ML / DV — anterior-posterior / medial-lateral / dorsal-ventral, µm) from monopolar triangulation.
* **closest_ch** — the recording channel nearest that triangulated location.
* **brain_area** — the Allen CCF region acronym at the unit's location.

*Rate & noise*

* **firing_rate** — the unit's median firing rate across the concatenated sessions (Hz).
* **firing_rate_si** — *SpikeInterface*'s single-pass firing rate (Hz).
* **firing_range** — 5–95th-percentile spread of the per-bin firing rate (Hz).
* **num_spikes** — total spike count.
* **noise_level** — root-mean-square (RMS) noise on the unit's peak channel.

*Waveform (template) shape*

* **waveform_duration** — peak-to-trough (peak-to-valley) duration.
* **peak_trough_ratio** — post-trough-peak amplitude / trough amplitude.
* **fwhm** — full width at half-maximum of the trough.
* **repolarization_slope** — rising slope out of the trough.
* **recovery_slope** — slope after the post-trough peak.
* **exp_decay** — spatial exponential-decay constant of the template amplitude across channels (µm).
* **spread** — spatial extent of the template over channels (µm).

*Somatic-classifier features* (the peak/trough measurements the ``somatic`` rule rests on)

* **main_trough_size** / **main_peak_before_size** / **main_peak_after_size** — absolute amplitudes of the main trough and the largest positive peaks before / after it.
* **main_peak_before_width** / **main_trough_width** — half-prominence widths (samples) of the pre-trough peak and the trough.
* **peak1_to_peak2_ratio** / **trough_to_peak2_ratio** / **main_peak_to_trough_ratio** — the amplitude ratios the non-somatic rule thresholds.

*Amplitude*

* **amplitude_median** — median spike amplitude.
* **amplitude_cutoff** — estimated fraction of spikes missed below the detection threshold.
* **amplitude_cv_median** / **amplitude_cv_range** — median and 5–95th-percentile range of the spike-amplitude coefficient of variation.

*Refractory, presence & contamination*

* **isi_violations_ratio** / **isi_violations_count** — inter-spike-interval refractory violations (contamination proxy).
* **rp_contamination** — refractory-period contamination estimate.
* **presence_ratio** — fraction of the recording (60-s bins) in which the unit fires.
* **snr** — signal-to-noise ratio (template peak amplitude vs noise).
* **sd_ratio** — spike-amplitude standard deviation (SD) relative to noise SD.
* **sync_spike_2** / **sync_spike_4** / **sync_spike_8** — fraction of spikes coincident with ≥ 2 / 4 / 8 other units (synchrony / cross-unit contamination).

*Isolation (PC-space)*

* **isolation_distance** — Mahalanobis isolation distance of the unit's cluster.
* **l_ratio** — L-ratio (cluster contamination).
* **d_prime** — linear-discriminant separation from neighbouring units.
* **silhouette** — silhouette score (cluster separation).
* **nn_hit_rate** / **nn_miss_rate** — nearest-neighbour hit / miss rates.

The rate/refractory/presence/contamination, amplitude and isolation metrics come from
*SpikeInterface* (``compute_quality_metrics`` and the template metrics); the ``somatic``
call and its peak/trough features are the owned addition in ``spikeinterface_helpers``.

.. _channel-indexing:

Reconciling channel numbering conventions
-----------------------------------------

Understanding which of the **two indexing conventions** each file uses is
essential when joining IBL anatomy to *Kilosort* spike data, building the
channel-to-region converter, or interpreting unit names.

Mapping conventions
~~~~~~~~~~~~~~~~~~~

The same 384 electrodes are indexed two ways across the files this pipeline
produces and consumes:

**Raw-meta channel id** — the hardware channel id SpikeGLX assigned via the
IMRO table. For the *k*-th electrode it is the first field of that electrode's
IMRO row (``imro_rows[k+1][0]``; ``imro_rows[0]`` is the table header, so
electrode *k* is row ``k+1``), and it is the numeric suffix ``id`` in the unified
``channel_locations.json``'s ``channel_{id}`` keys. For NP 2.0 4-shank (probe
type 2013) the raw-meta-to-physical-shank mapping is not monotonic — consecutive
raw-meta channels can sit on different shanks.

***Kilosort* row index** (row of ``channel_positions.npy`` and
``channel_shanks.npy``, peak channel inside ``templates.npy``,
``spike_clusters.npy``).
The row in the recording after *SpikeInterface* applied the IMRO
permutation. *Kilosort* orders rows to follow whatever electrodes the IMRO
selected — grouped shank-major (all of one shank's channels, then the next),
and within a shank by increasing raw-meta channel id. **The block layout
therefore depends on the IMRO**: a one-shank recording puts all 384 rows on
that shank, a two-shank recording splits them roughly in half, and different
bank selections rearrange them again. And within a shank the raw-meta id
order is *not* guaranteed to be axially monotonic, so a contiguous anatomical
band can map to non-contiguous KS rows — any consumer that compresses per-row
anatomy into ``[lo, hi]`` KS-row ranges must not assume it is.

For example, in a 4-shank / 96-per-shank config, rows ``0..95``
sit on shank 1, ``96..191`` on shank 2, ``192..287`` on shank 3 and
``288..383`` on shank 4. Shanks 1 and 2 were each assigned two disjoint
raw-meta id ranges in different axial bands (shank 1: ids ~48–87 at axial
2160–2445 µm, then ids ~280–383 at 1740–2145 µm), so their KS rows step
through axial mid → top then wrap to the bottom band, while shanks 3 and 4
got a single contiguous range and stay monotonically ascending — which is why
a single band there (e.g. shank-1 ventral MRN) lands on two non-contiguous
KS-row intervals.

*Phy* reads these sidecars directly and keeps the same numbering, so *phy*
channel labels and unit file names like ``imec0_cl0017_ch042_good.npy`` follow the
same convention.

Both indices point at the same electrode, whose actual ``(lateral, axial)``
**physical position** in microns is the common anchor used to bridge them
(below) — not an indexing convention in its own right. Within each shank the
lateral is one of two values 32 µm apart; across shanks the absolute lateral is
offset by ``shank * 250 µm`` (4-shank center-to-center spacing).

Conversion principles
~~~~~~~~~~~~~~~~~~~~~

The permutation between raw-meta and *Kilosort* row is stored explicitly in
``channel_map.npy`` under each *Kilosort* directory, so converting either way is a
one-liner (``cm`` is the loaded map)::

    cm = np.load(ks_dir / "channel_map.npy").flatten()
    raw_meta_ch = int(cm[ks_row])                            # KS row -> raw-meta channel
    ks_row      = int(np.argwhere(cm == raw_meta_ch)[0, 0])  # raw-meta channel -> KS row

The first direction is what you need to read an IBL JSON entry directly —
``ibl_json[f"channel_{cm[ks_row]}"]``.

Physical position is the universal bridge — for every electrode,
*SpikeInterface*'s ``channel_positions.npy[KS_row]`` agrees byte-for-byte with the
``lateral`` / ``axial`` fields of the matching entry in the unified IBL
``channel_locations.json`` (Step 5) — the ``channel_{raw_id}`` entry, where
``raw_id = channel_map.npy[KS_row]``. So (``IBL`` = the loaded
``channel_locations.json``, ``cp`` = ``channel_positions.npy``, ``cm`` as above)::

    cp[i]  ==  (IBL[f"channel_{cm[i]}"]["lateral"],
                IBL[f"channel_{cm[i]}"]["axial"])

So **any join between a *Kilosort* artifact and an IBL artifact can be done by
position** — match on ``cp[i]`` against the IBL entry's ``lateral`` / ``axial``
fields and skip the index translation entirely, with no need to load
``channel_map.npy``.

Convention used by file
~~~~~~~~~~~~~~~~~~~~~~~

Raw-meta channel id:

- SpikeGLX ``~imroTbl`` rows (``concatenated_<date>_imec<i>.ap.meta``).
- SpikeGLX ``~snsGeomMap`` rows (same meta file).
- IBL ``channel_locations.json`` and the per-shank
  ``channel_locations_shank{1..4}.json`` files.

*Kilosort* row index:

- ``channel_positions.npy``
- ``channel_shanks.npy``
- ``templates.npy`` (the last axis)
- ``spike_clusters.npy`` peak channels
- ``channel_order_per_shank.json`` (after the snsGeomMap-based fix)
- ``unit_catalog.csv`` ``closest_ch`` column
- Unit file names like ``imec0_cl0017_ch042_good.npy``
- The regenerated ``neuropixels_sites_to_anatomy_converter.json`` (see
  below)

Physical coordinates (not a channel-indexing convention):

- The brain-coord columns of ``unit_catalog.csv`` (``loc_ap``,
  ``loc_ml``, ``loc_dv``) and the Allen Bregma µm of every histology
  output — these are not channel-indexed at all.

Chosen canonical convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every artifact a downstream consumer touches — the catalog, unit files, the
per-shank channel order JSON, templates — uses the *Kilosort* row index, and
those consumers (e.g. ``make_behavioral_videos.find_region_by_channel``) read
unit names or catalog values and pass *Kilosort* row numbers into the
channel-to-region converter's membership checks. The converter is therefore
keyed by the *Kilosort* row index to match, so those lookups return the right
region.

.. _histology-notebook:

Notebook
--------
The ``npx_histology_unit_quality_processing.ipynb`` notebook is the
recommended entry point — it runs the whole workflow above in order from a
single **Parameters** cell. Its detailed walkthrough, knobs, and rendered
source live in :doc:`Notebooks`.
