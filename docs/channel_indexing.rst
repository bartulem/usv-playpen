Channel indexing
================

The same 384 Neuropixels electrodes are referred to by **four different
indexing conventions** across the files this pipeline produces and
consumes. Understanding which space each file lives in is essential
when joining IBL anatomy to Kilosort spike data, building the
channel-to-region converter, or interpreting unit names.

The four spaces
---------------

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
-----------

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
-------------------------------

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
--------------------------------------------------

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

How the regenerator works
-------------------------

For every ``(mouse, session, probe)`` triple already present in the
converter:

1. Load the IBL ``channel_locations.json`` for the appropriate
   hemisphere (``imec0`` → right, ``imec1`` → left in this dataset),
   and build a ``(lateral, axial) -> brain_region`` lookup.
2. Load ``channel_positions.npy`` from the Kilosort directory.
3. For every Kilosort row ``i``, the row's region is the IBL region
   at the physical position ``cp[i]``.
4. Compress contiguous runs of identical regions into
   ``[lo, hi]`` half-open ranges per region.

The generated entries are written back to
``neuropixels_sites_to_anatomy_converter.json`` in the same nested
``{mouse: {session: {probe: {region: [[lo, hi], ...]}}}}`` layout.

Because Kilosort row ordering is shank-major, each probe's
regenerated entry has every range bounded inside one shank's KS-row
block (rows 0..95, 96..191, etc.). The within-shank axial ordering
is not always monotonic, so a single anatomical band on a shank may
appear as two non-contiguous ``[lo, hi]`` KS-row intervals in the
JSON. Set membership against the converter still resolves the right
region regardless.

Translating between spaces
--------------------------

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

Module reference
----------------

The regenerator lives in
:mod:`usv_playpen.analyses.npx_anatomy_converter`. It can be run from
the command line::

    uv run python -m usv_playpen.analyses.npx_anatomy_converter

with ``--dry-run`` to inspect the summary without writing the file.
