Notebooks
=========

The repository ships a set of Jupyter notebooks (plus one `marimo <https://marimo.io>`_
app, the marimo reactive-notebook) under ``src/usv_playpen/notebooks/`` that drive the
advanced analysis and figure-generation workflows. This page is their single, detailed
home -- one section per notebook, walking through its cells as code blocks with the
parameters explained -- and the topical sections (:doc:`Analyze`, :doc:`Modeling`,
:doc:`Visualize`, :doc:`Neuropixels`) link here rather than duplicating the explanation.

.. note::

   The code blocks below mirror each notebook's cells; cell outputs are not shown (they
   are stripped from the committed notebooks via ``nbstripout``). Run a notebook locally
   to see its figures, or open its source on GitHub (linked at the end of each section).

Notebooks configure a run either from a single **Parameters** cell near the top or from
**per-section parameters** (each cell defining its own knobs inline at its top) -- the layout
is noted per notebook below. Paths are written ``/mnt/falkner/...`` and ``configure_path()``-
normalised to the host OS. Each plotting cell is independent -- re-run any single one once the
imports / parameters / setup cells above it have run.

Neuropixels processing
----------------------
**npx_histology_unit_quality_processing.ipynb** is the end-to-end histology /
Neuropixels-alignment workflow for one session: it assembles light-sheet
histology volumes, bridges the Kilosort spike-sorter output and brainreg (the
atlas-registration tool) track tracing into the International Brain
Laboratory (IBL) ephys-alignment GUI, and distils every unit
into an Allen-CCF-anchored ``unit_catalog.csv``. It is not a GUI tab — run it
cell by cell for one session. See :doc:`Neuropixels` for the conceptual workflow
and the underlying helpers.

The notebook runs steps **1, 3, 5, 6 and 7**; the acquisition-time and manual
steps happen outside it — **step 0** (probe geometry + spike sorting) at
acquisition, **step 2** (register probe tracks to anatomy, in brainreg + napari)
between steps 1 and 3, and **step 4** (the IBL ephys-alignment GUI) between steps
3 and 5.

The cells are
organised as *(1) imports* then *(2) per-section execution*, and **each section
cell defines its own parameters inline at the top** (edit those values plus
``EXPERIMENTER`` in the imports cell). ``analyses_settings.json`` is loaded once
in the imports cell, and each section pulls its own stable-tunable block from it.
Paths are written ``/mnt/...`` and wrapped in ``configure_path()`` so they resolve
on macOS (``/Volumes/...``) too.

**Imports.** Set the experimenter, import the histology helpers, apply the plot style, and
load ``analyses_settings.json`` once (each section cell below pulls its own
stable-tunable block from it). The cell also enables ``autoreload`` so source
edits are picked up without a kernel restart.

.. code-block:: python

    from __future__ import annotations

    import json
    import os
    from pathlib import Path

    # Set this BEFORE the usv_playpen imports below. The data paths in this
    # notebook are written under the shipped experimenter (Bartul) and re-keyed
    # to the experimenter in use by resolve_experimenter_path().
    EXPERIMENTER = None
    if EXPERIMENTER:
        os.environ["EXPERIMENTER_ID"] = EXPERIMENTER

    from usv_playpen.os_utils import configure_path, resolve_experimenter_path
    from usv_playpen.visualizations.plot_style import apply_plot_style
    from usv_playpen.neuropixels.histology_stack_lightsheet_volume import (
        stack_lightsheet_volume,
    )
    from usv_playpen.neuropixels.histology_stitch_smartspim_tiles import (
        stitch_smartspim_tiles,
    )
    from usv_playpen.neuropixels.histology_ibl_alignment_export import IBLAlignmentExporter
    from usv_playpen.neuropixels.anatomy_converter import add_session_to_anatomy_converter
    from usv_playpen.neuropixels.spike_quality_metrics import SpikeQualityMetricsExtractor

    apply_plot_style()

    # Load the analyses settings once; each section cell pulls its stable-tunable block.
    with open(
        Path.cwd().parent / "_parameter_settings" / "analyses_settings.json", "r"
    ) as analyses_settings_file:
        analyses_settings = json.load(analyses_settings_file)

* **EXPERIMENTER** — must be set *before* the ``usv_playpen`` imports, because the data paths are re-keyed at import time. The paths are written under the shipped experimenter (``Bartul``) and re-keyed to the experimenter in use by ``resolve_experimenter_path``. ``None`` uses this machine's configured experimenter (from ``_config/behavioral_experiments_settings.toml``); an id like ``"Annegret"`` (which sets ``EXPERIMENTER_ID``) resolves every path under that experimenter's tree instead (restart the kernel to change it after import).
* **analyses_settings** — the parsed ``_parameter_settings/analyses_settings.json``, loaded once here; each section below indexes into it for its own settings block.

**1. Light-sheet assembly.** Combine a raw light-sheet microscopy acquisition into one BigTIFF volume (the large-image TIFF format) per channel (wavelength) for brainreg / napari (the image viewer) registration. Two acquisition modalities are supported:

* **LaVision UltraMicroscope** — 1.1× objective, 5.91 µm/pix lateral, 10 µm axial. The raw acquisition is a flat directory of OME-TIFF Z-planes per channel, single tile. ``stack_lightsheet_volume`` glues the planes into one BigTIFF per channel, optionally flipping each plane in-plane and reversing Z order so the output is dorsal-first.
* **LifeCanvas SmartSPIM** — 1.625× objective, 4.02 µm/pix lateral, 10 µm axial. The raw acquisition is a tiled XY grid of Z-stacks per channel under ``Ex_{wavelength}_Ch{n}/{X}/{X}_{Y}/``. ``stitch_smartspim_tiles`` reads ``metadata.txt``, converts stage coordinates (0.1 µm units) to pixel offsets, and streams a plane-by-plane stitch with a bevel-shaped linear feather (``feather_pixels``) over tile seams.

Both functions default to ``wavelength_nm=(488, 561)`` and process both channels
in one call, writing one BigTIFF per channel; pass an ``int`` (e.g.
``wavelength_nm=561``) to restrict to a single channel.

.. code-block:: python

    # LaVision UltraMicroscope light-sheet stacking
    STACK_CFG = analyses_settings["npx_histology_stack_lightsheet_volume"]
    STACK_RAW_DIR = configure_path(
        "/mnt/lightsheet/_rawData/LaVision/bmimica/251015_bmimica_178621-dv-lv-1_09-44-40"
    )
    STACK_OUTPUT_PATH = resolve_experimenter_path(
        "/mnt/falkner/Bartul/histology/178621_2/registration/{wavelength_nm}nm/178621_{wavelength_nm}nm_fullsize.tif"
    )

    stack_lightsheet_volume(
        raw_dir=STACK_RAW_DIR,
        output_path=STACK_OUTPUT_PATH,
        **STACK_CFG,
    )

.. code-block:: python

    # LifeCanvas SmartSPIM light-sheet stitching
    STITCH_CFG = analyses_settings["npx_histology_stitch_smartspim_tiles"]
    STITCH_RAW_DIR = configure_path(
        "/mnt/lightsheet/_rawData/SmartSPIM/jj9483/20251118_14_09_42_jj9483_181321_1x_vd_ss_1"
    )
    STITCH_OUTPUT_PATH = resolve_experimenter_path(
        "/mnt/falkner/Bartul/histology/181321_1/registration/{wavelength_nm}nm/181321_{wavelength_nm}nm_stitched.tif"
    )

    stitch_smartspim_tiles(
        raw_dir=STITCH_RAW_DIR,
        output_path=STITCH_OUTPUT_PATH,
        **STITCH_CFG,
    )

* **STACK_RAW_DIR** / **STITCH_RAW_DIR** — the raw LaVision / SmartSPIM acquisition directory.
* **STACK_OUTPUT_PATH** / **STITCH_OUTPUT_PATH** — output path template. When multiple wavelengths are requested it **must** contain ``{wavelength_nm}`` (filled per channel); the shipped ``Bartul`` component is re-keyed to the experimenter in use.
* **STACK_CFG** / **STITCH_CFG** — the ``npx_histology_stack_lightsheet_volume`` / ``npx_histology_stitch_smartspim_tiles`` settings block, holding the stable tunables (``xy_flip`` / ``z_flip`` / ``skip_first`` and ``z_flip`` / ``feather_pixels`` respectively). Override any by passing an explicit kwarg to the call.

**Orientation controls.** Two independent knobs, both set by trial-and-error per
dataset. ``xy_flip`` flips each 2D plane *in-plane* (one of ``'none'``,
``'vertical'``, ``'horizontal'``, ``'both'`` = 180° rotation) and affects the
**axial** view in napari; ``z_flip`` reverses the Z (depth) iteration order and
affects the **coronal** and **sagittal** views (both share the Z axis). The
earlier auto-detection of acquisition direction (``dv`` / ``vd``) was dropped
because the labels written by ImSpector / the SmartSPIM app proved unreliable —
picking ``z_flip`` per dataset is simpler and explicit.

**Troubleshooting in napari.** If the brain renders flipped: wrong in **axial**
view → tweak ``xy_flip``; right in axial but **upside-down in coronal and
sagittal** → toggle ``z_flip``.

**3. Pre-alignment export.** Once the BigTIFF volumes are registered with brainreg and per-shank tracks are
traced in napari, ``IBLAlignmentExporter`` writes the two kinds of input the IBL
ephys-alignment GUI needs per session/probe/hemisphere: **track points** (one
``xyz_picks_shank{n}.json`` per shank in IBL mlapdv space, produced by an affine
on each shank's brainreg ``.npy``) and **an ALF dataset** (the IBL canonical
``spikes.* / clusters.* / templates.* / channels.*`` layout). It replaces the
upstream ``atlaselectrophysiology.extract_files.extract_data`` call, which streams
the raw concatenated ``.ap.bin`` (hundreds of GB, hours of wall-clock) to compute
``_iblqc_*`` RMS maps the GUI does not need — every output is derivable from the
Kilosort directory + the ``.ap.meta`` alone. Only Neuropixels 2.0 probes are
supported. The cell loops over the ``probe_to_hemisphere`` mapping and runs
``write_xyz_picks`` then ``write_alf_outputs`` for each probe.

.. code-block:: python

    PREALIGN_PROBE_TO_HEMISPHERE = analyses_settings["npx_histology_ibl_alignment_export"][
        "probe_to_hemisphere"
    ]
    PREALIGN_OS_CUP_LOC = resolve_experimenter_path("/mnt/falkner/Bartul")
    PREALIGN_MOUSE_ID = "164335_0"
    PREALIGN_SESSION_DATE = "20250912"
    PREALIGN_KILOSORT_VERSION = "4"

    for probe_id, hemisphere in PREALIGN_PROBE_TO_HEMISPHERE.items():
        print(f"--- {probe_id} ({hemisphere}H) ---")
        exporter = IBLAlignmentExporter(
            os_cup_loc=PREALIGN_OS_CUP_LOC,
            mouse_id=PREALIGN_MOUSE_ID,
            session_date=PREALIGN_SESSION_DATE,
            probe_id=probe_id,
            hemisphere=hemisphere,
            kilosort_version=PREALIGN_KILOSORT_VERSION,
            out_subdir=None,
        )
        for p in exporter.write_xyz_picks():
            print(p)
        exporter.write_alf_outputs()
        print(f"ALF outputs written to: {exporter.ephys_out_path}")

* **PREALIGN_PROBE_TO_HEMISPHERE** — the per-lab ``{"imec0": "R", "imec1": "L"}`` convention read from ``analyses_settings.json``; the notebook does not infer hemisphere from the data, and the loop over this mapping processes a two-probe session in one cell.
* **PREALIGN_OS_CUP_LOC** — the experimenter's file-server root (the shipped ``/mnt/falkner/Bartul`` re-keyed to the experimenter in use).
* **PREALIGN_MOUSE_ID** / **PREALIGN_SESSION_DATE** / **PREALIGN_KILOSORT_VERSION** — which session and Kilosort output to export.
* **out_subdir** — leave ``None`` for production; set it to write the ALF outputs to a sibling of ``ibl_{hemisphere}H`` when validating a new run against a reference.

**4. Channel alignment.** This step runs **outside the notebook**. With the contents of each
``ibl_{hemisphere}H/`` in place, launch the IBL ephys-alignment GUI separately and
walk each shank's track through the alignment workflow. The GUI writes one
``channel_locations_shank{n}.json`` per shank back into the same
``ibl_{hemisphere}H/`` directory, keyed by per-shank channel index (``channel_0``
.. ``channel_{m-1}``, 96 for NP2.0 four-shank). Once all per-shank JSONs exist for
every probe, continue with the post-alignment cell below.

**5. Post-alignment export.** Post-process the GUI's per-shank JSONs — two steps, both pure JSON manipulation.
``remap_channel_ids_to_raw`` re-keys each per-shank JSON from local shank indices
(``channel_0`` .. ``channel_{m-1}``) to raw recording channel ids using the IMRO
table cached at construction (a no-op for single-shank NP2.0), and
``write_unified_channel_locations`` concatenates all ``channel_locations_shank*.json``
into a single SpikeInterface-ready (the SpikeInterface framework)
``channel_locations.json`` per probe, sorted by integer channel id.

.. code-block:: python

    POSTALIGN_PROBE_TO_HEMISPHERE = analyses_settings["npx_histology_ibl_alignment_export"][
        "probe_to_hemisphere"
    ]
    POSTALIGN_OS_CUP_LOC = resolve_experimenter_path("/mnt/falkner/Bartul")
    POSTALIGN_MOUSE_ID = "181322_2"
    POSTALIGN_SESSION_DATE = "20251012"
    POSTALIGN_KILOSORT_VERSION = "4"

    for probe_id, hemisphere in POSTALIGN_PROBE_TO_HEMISPHERE.items():
        print(f"--- {probe_id} ({hemisphere}H) ---")
        exporter = IBLAlignmentExporter(
            os_cup_loc=POSTALIGN_OS_CUP_LOC,
            mouse_id=POSTALIGN_MOUSE_ID,
            session_date=POSTALIGN_SESSION_DATE,
            probe_id=probe_id,
            hemisphere=hemisphere,
            kilosort_version=POSTALIGN_KILOSORT_VERSION,
            out_subdir=None,
        )
        for p in exporter.remap_channel_ids_to_raw():
            print(p)
        out_path = exporter.write_unified_channel_locations()
        print(f"Unified channel_locations.json written to: {out_path}")

* Same ``IBLAlignmentExporter`` fields as pre-alignment, now pointing at the session whose GUI alignment is complete (``probe_to_hemisphere`` / ``os_cup_loc`` read as before; ``mouse_id`` / ``session_date`` / ``kilosort_version`` name the aligned session).

**6. Channel-brain converter.** Fold this session's per-probe brain-region map — keyed by **Kilosort row index** —
into the global channel-brain area converter
``EPHYS/neuropixels_sites_to_anatomy_converter.json``. For each
``(mouse, session, probe)`` it joins the IBL ``channel_locations.json`` regions to
Kilosort rows by physical ``(lateral, axial)`` position, compresses contiguous
same-region runs into half-open KS-row ranges, and **merges** that block into the
converter so every other mouse / session / probe is preserved byte-for-byte.
``add_session_to_anatomy_converter`` is **add-if-missing**: a triple already in the
converter is left untouched.

.. code-block:: python

    ANATOMY_PROBE_TO_HEMISPHERE = analyses_settings["npx_histology_ibl_alignment_export"][
        "probe_to_hemisphere"
    ]
    ANATOMY_MOUSE_ID = "158112_0"
    ANATOMY_SESSION_ID = "20241107_114630"
    ANATOMY_FORCE = False

    for probe_id, hemisphere in ANATOMY_PROBE_TO_HEMISPHERE.items():
        print(f"--- {probe_id} ({hemisphere}H) ---")
        summary = add_session_to_anatomy_converter(
            ANATOMY_MOUSE_ID,
            ANATOMY_SESSION_ID,
            probe_id,
            force=ANATOMY_FORCE,
            probe_to_hemisphere=ANATOMY_PROBE_TO_HEMISPHERE,
        )
        detail = f": {summary['reason']}" if summary["reason"] else ""
        print(f"{summary['status']}{detail} -> {summary['output']}")

* **ANATOMY_MOUSE_ID** — the tail-tagged animal id (the histology directory name).
* **ANATOMY_SESSION_ID** — the full session id used as the converter key (e.g. ``20241107_114630``); its first eight characters are the recording date that locates the Kilosort and IBL outputs.
* **ANATOMY_FORCE** — ``False`` (add-if-missing) by default; ``True`` re-regenerates a single existing block (e.g. after a re-alignment).
* Converter / ephys / histology paths default to the ``data_roots`` block of ``analyses_settings.json`` (resolved via ``configure_path``), so they need not be set here.

**Rebuilding the whole converter from scratch.** To rewrite *every* session at
once — e.g. after changing the region-join logic — don't loop this per-session
cell; use the batch entry point ``regenerate_anatomy_converter``, exposed as
``python -m usv_playpen.neuropixels.anatomy_converter --regenerate-all`` (add
``--dry-run`` to preview, or ``--mouse/--session/--probe`` to target one triple).

**7. Spike-quality metrics.** With the unified ``channel_locations.json`` in place for every probe, compute the
per-unit spike quality-metrics catalog. ``SpikeQualityMetricsExtractor`` ports the
per-session half of the ``si_quality_metrics_Neuropixels2.0`` workflow onto pinned
stock ``spikeinterface==0.104.3`` and reads the (hundreds-of-GB, multi-hour)
recording **once** in two passes: a recording-free core pass for the spike-train
metrics, and a single sequential recording read for the ``waveforms`` extension
from which the template / somatic / location / SNR / PCA / amplitude metrics all
derive. It writes the global 55-column ``EPHYS/unit_catalog.csv`` (rows merged in
idempotently per ``mouse_id`` + ``rec_date`` + probe) plus each probe's
``channel_order_per_shank.json``.

.. code-block:: python

    SQM_SETTINGS = analyses_settings["npx_spike_quality_metrics"]
    SQM_PROBE_TO_HEMISPHERE = SQM_SETTINGS["probe_to_hemisphere"]
    SQM_EXTRACTOR_KWARGS = {
        key: value for key, value in SQM_SETTINGS.items() if key != "probe_to_hemisphere"
    }
    SQM_OS_CUP_LOC = resolve_experimenter_path("/mnt/falkner/Bartul")
    SQM_MOUSE_ID = "158112_0"
    SQM_SESSION_DATE = "20241107"

    for probe_id, hemisphere in SQM_PROBE_TO_HEMISPHERE.items():
        print(f"--- {probe_id} ({hemisphere}H) ---")
        extractor = SpikeQualityMetricsExtractor(
            os_cup_loc=SQM_OS_CUP_LOC,
            mouse_id=SQM_MOUSE_ID,
            session_date=SQM_SESSION_DATE,
            probe_id=probe_id,
            hemisphere=hemisphere,
            **SQM_EXTRACTOR_KWARGS,
        )
        catalog = extractor.run()
        print(f"catalog ({catalog.shape[0]} units) written to: {extractor.catalog_path}")

* **SQM_SETTINGS** — the ``npx_spike_quality_metrics`` block (``kilosort_version``, ``num_channels_sparsity``, ``shank_width_microns``, ``shank_spacing_microns``, ``phy_curated``, ``somatic_classifier``, ``job_kwargs``, plus the same ``probe_to_hemisphere`` convention as the IBL export). ``job_kwargs['n_jobs']`` is the main throughput knob for the single recording read.
* **SQM_EXTRACTOR_KWARGS** — that block minus ``probe_to_hemisphere``, splatted into the extractor.
* **SQM_MOUSE_ID** — the tail-tagged animal id; it names the histology directory (locating ``channel_locations.json``) **and** is written into the catalog's ``mouse_id`` column.
* **SQM_OS_CUP_LOC** / **SQM_SESSION_DATE** — the session to catalog.

Source: `npx_histology_unit_quality_processing.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/npx_histology_unit_quality_processing.ipynb>`_.

.. _modeling-notebook:

Modeling analyses
-----------------

**modeling_analyses.ipynb** is the end-to-end, interactive entry point for the
**vocal-modeling pipelines** in ``usv_playpen.modeling``. Run its sections in
order: extract per-session loader output into per-pipeline modeling-input
pickles, audit the predictors, consolidate the univariate / model-selection
outputs that the cluster produced, then render every diagnostic figure and,
finally, fit the CNN — each stage consumes artifacts written by an earlier one.
Univariate ranking and forward-stepwise model selection are **not** run here:
they run on the cluster (``main_univariate_dispatcher`` /
``main_model_selection_dispatcher``) and this notebook only consumes their
outputs. It is not a GUI tab; run it cell by cell for one analysis condition.

**Most sections
define their own parameters** in a small ``# Parameters -- ...`` cell at its top
(paths are written ``/mnt/falkner/...`` and ``configure_path()``-normalised to
the host OS), and nothing downstream redefines them. Every pipeline reads
``_parameter_settings/modeling_settings.json`` via ``modeling_settings_dict=None``
(the project default); pass an explicit dict for a one-off override.

**Imports.** Import the modeling pipelines, consolidators, and plotters, and apply the shared
matplotlib style. There is no experimenter knob here — configuration flows
entirely through ``modeling_settings.json`` and the per-section Parameters cells.

.. code-block:: python

    import pathlib

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.plot_style import apply_plot_style

    from usv_playpen.modeling.consolidate_univariate_results import (
        consolidate as consolidate_univariate,
    )
    from usv_playpen.modeling.consolidate_model_selection_results import (
        consolidate as consolidate_model_selection,
    )
    from usv_playpen.modeling.modeling_usv_manifold_position import (
        ContinuousModelingPipeline,
    )
    from usv_playpen.modeling.modeling_vocal_bout_parameters import BoutParameterPipeline
    from usv_playpen.modeling.modeling_vocal_categories_multinomial import (
        MultinomialModelingPipeline,
    )
    from usv_playpen.modeling.modeling_vocal_onsets import VocalOnsetModelingPipeline
    from usv_playpen.modeling.jax_neural_network_cnn import NeuralContinuousCNNRunner

    from usv_playpen.visualizations.modeling_plots import (
        DeepResultsVisualizer,
        plot_collinearity_audit,
        plot_feature_ranking,
        plot_manifold_multivariate_filters,
        plot_manifold_selection_trajectory,
        plot_model_selection_results,
        plot_multinomial_multivariate_filters,
        plot_multinomial_selection_diagnosis,
        plot_multinomial_selection_trajectory,
        plot_raw_feature_difference,
        plot_significant_filters,
        plot_significant_filters_grid,
        plot_timescale_audit,
        plot_timescale_audit_per_feature,
        plot_univariate_multinomial_performance,
    )

    apply_plot_style()

**1. Extract input data.** Convert session-level loader output into the modeling-input pickle that the
downstream univariate / model-selection runners consume. The four calls differ
only in **what gets predicted** (and therefore which event timestamps populate
the timescale audit's ``Y(t)`` impulse trace). Each call writes the modeling-input
pickle to ``io.save_directory`` (filename embeds the cohort label, e.g.
``male_mute_partner``, plus a timestamp) and the paired predictor-diagnostics
audits ``*_collinearity.pkl`` and ``*_timescales.pkl``; the cohort label is
derived from ``io.session_list_file``.

.. code-block:: python

    VocalOnsetModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()
    BoutParameterPipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()
    MultinomialModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_multinomial_input_data()
    ContinuousModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_continuous_data()

* **VocalOnsetModelingPipeline** — "does a frame start a vocal event (bout / USV)?"; ``Y(t)`` impulses at bout / USV onsets.
* **BoutParameterPipeline** — per-bout duration / complexity / intensity; ``Y(t)`` impulses at bout starts.
* **MultinomialModelingPipeline** — per-USV vocal category; ``Y(t)`` impulses at per-USV starts.
* **ContinuousModelingPipeline** — per-USV UMAP manifold position; ``Y(t)`` impulses at per-USV starts.

Passing ``modeling_settings_dict=None`` loads the project default settings; no
Parameters-cell variables are consumed here.

**2. Predictor diagnostics.** Run after the extract step has produced the paired ``_collinearity.pkl`` and
``_timescales.pkl`` audits. The three plots share feature ordering and per-group
colour so one feature can be cross-referenced across all three figures — run the
per-feature plot first (the ground-truth view), then the cohort summary, then the
collinearity heatmap.

.. code-block:: python

    diag_timescale_pkl = configure_path(
        "/mnt/falkner/Bartul/modeling/..._male_mute_partner_..._timescales.pkl"
    )
    diag_collinearity_pkl = diag_timescale_pkl.replace(
        "_timescales.pkl", "_collinearity.pkl"
    )
    diag_save_fig_bool = False
    diag_save_fig_format = "svg"
    diag_save_fig_directory = pathlib.Path(configure_path("/mnt/falkner/Bartul/figures"))

    diag_save_fig_directory.mkdir(exist_ok=True, parents=True)

    plot_timescale_audit_per_feature(
        diag_timescale_pkl,
        save_plot_bool=diag_save_fig_bool,
        save_dir=diag_save_fig_directory,
        plot_format=diag_save_fig_format,
    )

    plot_timescale_audit(
        diag_timescale_pkl,
        save_plot_bool=diag_save_fig_bool,
        save_dir=diag_save_fig_directory,
        plot_format=diag_save_fig_format,
    )

    plot_collinearity_audit(
        diag_collinearity_pkl,
        save_plot_bool=diag_save_fig_bool,
        save_dir=diag_save_fig_directory,
        plot_format=diag_save_fig_format,
    )

* **diag_timescale_pkl** — the ``_timescales.pkl`` audit written next to the modeling input pickle; feeds both timescale plots.
* **diag_collinearity_pkl** — derived by string-swapping the suffix, so it always pairs with the same run.
* **diag_save_fig_bool** / **diag_save_fig_format** / **diag_save_fig_directory** — whether to write figures, the output format, and the destination directory (created if missing).

The plots read further knobs from inside the audits: ``diagnostics.timescale_*``
(max lag, n_shuffles, shuffle range, signal floor, min run length),
``model_params.mixture_model_component_index``,
``model_params.mixture_model_z_score``, and the ``mixture_model_params`` IBI
threshold reference line.

**3. Consolidate pickles.** Run after the SLURM job array (``main_univariate_dispatcher.py`` /
``main_model_selection_dispatcher.py``) finishes. The consolidators assert
metadata equality across every per-feature / per-step pickle, hoist the agreed
``_input_metadata`` / ``_run_metadata`` / ``_univariate_metadata`` blocks to the
top of one consolidated artifact, and emit a self-describing filename.

.. code-block:: python

    cons_univariate_input_dir = configure_path(
        ".../cluster/univariate_results_multi_file/male/male_multinomial_vae_supercategory"
    )
    cons_univariate_delete_individuals_after = False
    cons_selection_input_dir = configure_path(
        ".../model_selection_results/male/vocal_onset"
    )
    cons_selection_move_to_steps_subdir = False
    cons_selection_ignore_provenance_keys = (
        "git_commit",
        "git_dirty",
        "package_version",
        "settings_sha256",
    )

    # 1. Univariate per-feature -> consolidated univariate_<tag>_<condition>_<ts>.pkl
    consolidate_univariate(
        input_dir=cons_univariate_input_dir,
        delete_individuals_after=cons_univariate_delete_individuals_after,
    )

    # 2. Model-selection per-step -> consolidated
    #    model_selection_final_<sex>_<condition>_<analysis_tag>_<split_strategy>[_<ts>].pkl
    consolidate_model_selection(
        input_dir=cons_selection_input_dir,
        move_to_steps_subdir=cons_selection_move_to_steps_subdir,
        ignore_provenance_keys=cons_selection_ignore_provenance_keys,
    )

* **cons_univariate_input_dir** / **cons_selection_input_dir** — directories of per-feature and per-step pickles to merge.
* **cons_univariate_delete_individuals_after** — set ``True`` to delete the individual pickles once the consolidated artifact is verified.
* **cons_selection_move_to_steps_subdir** — when ``True``, relocate consumed step pickles into ``<input_dir>/steps/``.
* **cons_selection_ignore_provenance_keys** — extends the default provenance-key exclusions with ``settings_sha256`` so an unrelated mid-run edit to the settings file does not abort the ``_run_metadata`` equality check (drop it to restore the safety net).

**4. Univariate visualisations.** Plots over the consolidated **univariate** pickle (one fit per behavioural
feature, no forward stepping). Use these to triage which features look promising
before committing to model selection, or to compare cohorts. The section covers
**two target families**, separated by the markdown sub-header below.

**Scalar single-target analyses** — the plotters below serve any analysis that
predicts a single scalar target per feature: **vocal onsets**
(``VocalOnsetModelingPipeline``, binary — does a frame start a bout / individual
USV; metric ``ll`` or ``auc``), **bout parameters** (``BoutParameterPipeline``,
regression on per-bout duration / complexity / intensity; metric
``explained_deviance`` or ``spearman_r``), and **binomial USV category**
(``VocalCategoryModelingPipeline``, one target category vs a pooled "other";
metric ``ll`` or ``auc``). Point the ``uni_*`` paths at the chosen analysis's
consolidated univariate pickle and set the metric accordingly.

.. code-block:: python

    figures_dir = configure_path("/mnt/falkner/Bartul/modeling/figures")

    uni_ranking_results = configure_path(
        ".../univariate_results/univariate_multinomial_vae_supercategory_..._male_...Z.pkl"
    )
    uni_ranking_p_val = 0.01
    uni_filters_results = configure_path(
        ".../gam_results_male_mute_partner_category_18_...pkl"
    )
    uni_filters_grid_results = configure_path(".../gam_results_male_category_10_...pkl")
    uni_raw_diff_pkl = configure_path(
        ".../modeling_category_3_male_presence_all_...hist4s.pkl"
    )
    uni_raw_diff_feature_key = "self.neck_elevation"
    uni_raw_diff_feature_color = "#9AC0CD"

    plot_feature_ranking(
        results_file_loc=uni_ranking_results,
        p_val=uni_ranking_p_val,
        evaluation_metric="ll",
        evaluation_metric_name="Negative Log-Likelihood (held-out data)",
        secondary_metric="score",
        secondary_metric_name="Accuracy (held-out data)",
        ignore_features=None,
        save_plot=False,
        output_dir=figures_dir,
    )

    plot_significant_filters(
        results_file_loc=uni_filters_results,
        metric="ll",
        ignore_features=None,
        p_val=uni_ranking_p_val,
        save_plot=False,
        output_dir=figures_dir,
    )

    plot_significant_filters_grid(
        results_file_loc=uni_filters_grid_results,
        ignore_features=None,
        metric="ll",
        p_val_threshold=uni_ranking_p_val,
        save_plot=False,
        output_dir=figures_dir,
    )

    plot_raw_feature_difference(
        pickle_file_path=uni_raw_diff_pkl,
        feature_key=uni_raw_diff_feature_key,
        feature_color=uni_raw_diff_feature_color,
        subset_fraction=0.05,
        n_bootstraps=100,
        save_plots=False,
        output_dir=figures_dir,
    )

* **figures_dir** — shared figure output directory used by every visualisation section.
* **uni_ranking_results** / **uni_ranking_p_val** — the consolidated univariate pickle to rank and the significance threshold applied to it.
* **uni_filters_results** / **uni_filters_grid_results** — GAM-result pickles for the single-filter and gridded-filter plots.
* **uni_raw_diff_pkl** / **uni_raw_diff_feature_key** / **uni_raw_diff_feature_color** — the input pickle, the feature to contrast, and its hex plot colour.

**Multinomial target** — predicting a USV's vocal category across *all* categories
jointly (a one-vs-rest multinomial classifier) is **not** a scalar target:
performance is per-feature **and per-class**, so it gets its own univariate
plotter reading the consolidated univariate-multinomial pickle.

.. code-block:: python

    mn_univariate_results = configure_path(
        ".../univariate_results/multinomial_categories/univariate_multinomial_categories_male_...pkl"
    )

    plot_univariate_multinomial_performance(
        results_file_loc=mn_univariate_results,
        evaluation_metric="auc",
        evaluation_metric_name="Area Under the ROC Curve",
        secondary_metric="score",
        secondary_metric_name="Balanced Accuracy",
        p_val_threshold=0.05,
        diff_cmap="bwr",
        save_plot=False,
        output_dir=figures_dir,
    )

* **mn_univariate_results** — consolidated univariate-multinomial pickle; the ``diff_cmap`` panel shows each per-feature × per-class cell relative to the cohort mean.

**5. Model-selection visualisations.** Plots of the forward-stepwise selection trajectory — how held-out performance
evolves as features are stacked (selection itself runs on the cluster). As in
section 4, **two target families**, separated by the markdown sub-header below.

**Scalar single-target analyses** — the plotter below serves the forward-stepwise
selection of the same three scalar targets: **vocal onsets**
(``VocalOnsetModelingPipeline`` → ``vocal_onset_model_selection``), **bout
parameters** (``BoutParameterPipeline`` → ``bout_parameter_model_selection``), and
**binomial USV category** (``VocalCategoryModelingPipeline`` →
``vocal_category_model_selection``). Point ``msv_results_path`` at the chosen
analysis's consolidated ``model_selection_final_*.pkl``.

.. code-block:: python

    msv_results_path = configure_path(
        ".../model_selection_results/male/vocal_onset/model_selection_final_male_..._bout_mixed_...Z.pkl"
    )

    plot_model_selection_results(
        selection_results_path=msv_results_path,
        metric_secondary="score",
        save_plots=True,
        output_dir=figures_dir,
    )

* **msv_results_path** — the consolidated model-selection pickle whose per-step trajectory (plus the final accepted model's temporal filters) is plotted.

**Multinomial target** — multinomial (one-vs-rest) selection is per-class, so it
has its own trajectory / filter / diagnosis plotters reading the consolidated
multinomial selection artifacts.

.. code-block:: python

    mn_trajectory_results = configure_path(
        ".../model_selection_results/male/multinomial_qlvm_supercategory/model_selection_final_..._mixed_...Z.pkl"
    )
    mn_filters_results = configure_path(
        ".../model_selection_results/multinomial_category/male"
    )
    mn_diagnosis_results = configure_path(
        ".../model_selection_results/multinomial_category/male_mute_partner"
    )

    plot_multinomial_selection_trajectory(
        selection_results_path=mn_trajectory_results,
        metric_primary="auc",
        primary_metric_name="Area Under the ROC Curve",
        metric_secondary="score",
        secondary_metric_name="Balanced Accuracy",
        save_plot=False,
        output_dir=figures_dir,
        secondary_ylim_max=0.26,
    )

    plot_multinomial_multivariate_filters(
        selection_results_path=mn_filters_results,
        history_window_sec=4.0,
        cmap="bwr",
        save_plot=True,
        output_dir=figures_dir,
    )

    plot_multinomial_selection_diagnosis(
        selection_results_path=mn_diagnosis_results,
        cmap_diff="bwr",
        save_plot=True,
        output_dir=figures_dir,
    )

* **mn_trajectory_results** — primary + secondary metric per forward-stepwise iteration, broken down by class.
* **mn_filters_results** — final selected filters (one panel per class × selected feature), shared diverging colormap.
* **mn_diagnosis_results** — how far the multivariate selection departs from picking the top univariate feature per class (base + difference heatmaps).

**6. Manifold visualisations.** These two plotters consume the consolidated artifact written by
``continuous_vocal_manifold_model_selection`` (forward-stepwise selection for the
2-D acoustic-manifold regression) — same ``selection_*.pkl`` schema as the
multinomial plotters but with continuous regression metrics (``r2_spatial``,
``mahalanobis_mae``, ``euclidean_mae*``, ``pearson_x/y``, ``spearman_x/y``) and a
2-D output dim.

.. code-block:: python

    man_trajectory_results = configure_path(
        ".../model_selection_results/male/usv_manifold_vae_supercategory/model_selection_final_..._mixed_...Z.pkl"
    )
    man_filters_results = man_trajectory_results

    plot_manifold_selection_trajectory(
        selection_results_path=man_trajectory_results,
        metric_primary="r2_spatial",
        primary_metric_name="R² (spatial, KDE-weighted)",
        metric_secondary="pearson_y",
        secondary_metric_name="Pearson r (manifold y)",
        save_plot=False,
        output_dir=figures_dir,
    )

    plot_manifold_multivariate_filters(
        selection_results_path=man_filters_results,
        history_window_sec=4.0,
        cmap="RdBu_r",
        save_plot=False,
        output_dir=figures_dir,
    )

* **man_trajectory_results** — cumulative primary metric across forward-stepwise iterations plus secondary-metric bars for best-univariate vs the final stacked model.
* **man_filters_results** — final-model per-feature temporal filter atlas (rows = manifold-x / manifold-y, columns = time bins), reusing the same path.

**7. CNN pipeline.** A non-linear baseline (1-D ResNet) for the continuous manifold-position
regression. Load the multivariate feature blocks into the ``(N, F, T)`` tensor the
network expects, train, then render one of five diagnostic visualisation modes
over the saved predictions. The input pickle must carry per-USV ``supercategory``
labels (the pre-flight check inside ``run_cnn_training`` fails fast if missing).

.. code-block:: python

    cnn_input_pkl = configure_path(
        ".../modeling_manifold_vae_supercategory_intact_partners_male_...pkl"
    )
    cnn_results_pkl = configure_path(
        ".../cnn_manifold_integrated_predictions_male_QLVM_...pkl"
    )
    cnn_choose_analysis = "regional_saliency"

    runner = NeuralContinuousCNNRunner(modeling_settings=None)
    data_blocks = runner.load_multivariate_data_blocks(pkl_path=cnn_input_pkl)
    runner.run_cnn_training(data_blocks=data_blocks)

    deep_visualizer = DeepResultsVisualizer(
        results_pkl_path=cnn_results_pkl,
        modeling_settings=None,
        visualization_settings=None,
    )

    if cnn_choose_analysis == "permutation_test":
        deep_visualizer.plot_permutation_test(
            save_plot=False, output_dir=figures_dir, file_format="svg"
        )
    elif cnn_choose_analysis == "feature_importance":
        deep_visualizer.plot_feature_importance(
            snr_threshold=3.0,
            error_bar_color="#000000",
            save_plot=False,
            output_dir=figures_dir,
            file_format="svg",
        )
    elif cnn_choose_analysis == "spatial_precision_grid":
        deep_visualizer.plot_spatial_precision_grid(
            grid_shape=(4, 4),
            patch_size=0.20,
            min_samples=25,
            plot_type="density",
            bg_pt_color="#E0E0E0",
            peak_pt_color="#00FFFF",
            square_edge_color="#000000",
            panel_fontsize=9,
            figsize_unit=2.0,
            save_plot=False,
            output_dir=figures_dir,
            file_format="svg",
        )
    elif cnn_choose_analysis == "error_landscape":
        deep_visualizer.plot_error_landscape(
            gridsize=30,
            vmax_percentile=95.0,
            save_plot=False,
            output_dir=figures_dir,
            file_format="svg",
        )
    elif cnn_choose_analysis == "regional_saliency":
        deep_visualizer.plot_regional_saliency_inset(
            region_key="supercategory_7",
            category_name="QLVM supercategory 7",
            prediction_plot_type="density",
            radius=0.15,
            smoothing_sigma=10.0,
            save_plot=False,
            output_dir=figures_dir,
            file_format="svg",
        )
    else:
        print(f"Option {cnn_choose_analysis} not recognized.")

* **cnn_input_pkl** — the multivariate feature-block pickle from ``ContinuousModelingPipeline.extract_and_save_continuous_data()``, fed to training.
* **cnn_results_pkl** — the predictions pickle written at the end of ``run_cnn_training``, read by every visualisation mode.
* **cnn_choose_analysis** — selects the diagnostic projection: ``permutation_test``, ``feature_importance``, ``spatial_precision_grid``, ``error_landscape``, or ``regional_saliency``.

Source: `modeling_analyses.ipynb <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/modeling_analyses.ipynb>`_.

.. _unit-triage-aggregator:

Neuronal tuning summary
-----------------------
**neuronal_tuning_summary.ipynb** turns the per-cluster ``triage_stats`` blocks
that ``generate-rm`` writes into every ``*_tuning_curves_data.pkl`` into two
independent products. Its first half is a cross-session / cross-condition
**unit-triage aggregator**: it re-applies the significance rules over the
pre-computed stats, joins each cluster to ``unit_catalog.csv`` for anatomy, and
pickles a unit-keyed roll-up (``unit_triage_*.pkl``) so the same physical unit
recorded across replicate sessions in a day is represented once with per-session
evidence stacked beneath each modality. Its second half renders the
**anatomy / dataset-overview and population-summary figures** — recording-yield
bars, per-probe waveforms, a rotating-brain video and a static still, and the
population Vocalization Modulation Index (VMI) / USV / behavioural figures that
honour the ``UNIT_KSLABELS`` / ``UNIT_SOMATIC_FILTER`` knobs. It is a pure
pkl-to-pickle / catalog-to-figure pass and never touches spike or USV data; run
it cell by cell.

Every figure / section cell defines its
own knobs at the top, so you edit a cell's parameters in place to sweep values
across the same set of pkls without re-computing tuning. Shared values are
reassigned wherever they are used (``CATALOG_PATH`` is set again in the anatomy
cell, ``figure_condition`` is set again in each behavioural cell), while the two
figure-output routing knobs (``SAVE_FIGURES`` / ``SCRATCH_FIG_DIR``) live in the
Setup cell and thread ``FIG_OUT_DIR`` into every figure call. Significance
thresholds default to the ``detect_interesting_tuning_neurons`` block of
``_parameter_settings/analyses_settings.json``.

**Imports.** Import the aggregator and the two figure builders (plus the shared plot-style
helper).

.. code-block:: python

    import json
    from pathlib import Path

    import matplotlib as mpl

    from usv_playpen.analyses.unit_triage_aggregator import (
        aggregate_units_across_conditions,
    )
    from usv_playpen.visualizations.make_anatomy_figures import AnatomyFigureMaker
    from usv_playpen.visualizations.make_neuronal_tuning_figures import (
        NeuronalTuningFigureMaker,
    )
    from usv_playpen.visualizations.plot_style import apply_plot_style

**Setup.** Font/style registration, ``visualizations_settings.json`` load, and figure-output
routing. The two routing knobs (``SAVE_FIGURES`` / ``SCRATCH_FIG_DIR``) sit at the
top of the cell; everything below them is derived. ``apply_plot_style()``
registers the five bundled Helvetica weights with matplotlib and activates the
project mplstyle; the weight is then nudged to ``light`` so SVG export resolves to
Helvetica-Light.

.. code-block:: python

    # Figure-output routing knobs (top, editable)
    SAVE_FIGURES = True
    SCRATCH_FIG_DIR = "/tmp/usv_figures_scratch"

    apply_plot_style()
    mpl.rcParams["font.weight"] = "light"
    mpl.rcParams["axes.labelweight"] = "light"
    mpl.rcParams["svg.fonttype"] = "none"

    with open(
        Path.cwd().parent / "_parameter_settings" / "visualizations_settings.json"
    ) as f:
        vis_settings = json.load(f)

    # SAVE_FIGURES=True -> canonical figures.save_directory; False -> SCRATCH_FIG_DIR
    FIG_OUT_DIR = None if SAVE_FIGURES else SCRATCH_FIG_DIR
    if FIG_OUT_DIR is not None:
        Path(FIG_OUT_DIR).mkdir(parents=True, exist_ok=True)

* **SAVE_FIGURES** — ``True`` writes every figure to the canonical ``figures.save_directory`` from ``visualizations_settings.json``; ``False`` sets ``FIG_OUT_DIR`` to ``SCRATCH_FIG_DIR`` so the canonical directory stays clean while iterating.
* **SCRATCH_FIG_DIR** — the scratch directory used when ``SAVE_FIGURES`` is ``False``; ``FIG_OUT_DIR`` (derived) is threaded into every figure call as ``out_dir``.

**Aggregator.** Build or load the aggregator pickle keyed by ``(animal_id, YYYYMMDD, imec,
cluster_id)``. Each unit carries its identity, the catalog ``anatomy_region``, and
a ``conditions`` block — one entry per condition listing, per modality,
``n_significant`` / ``n_tested`` / ``consistency`` / an aggregate scalar / per-session
evidence rows. Each value of ``CONDITION_TO_SESSION_LIST`` is a ``.txt`` of session
roots (one per line); sessions missing a ``tuning_curves`` directory are recorded
under ``sessions_skipped``, and orphan pkls with no catalog row raise. The output
is written to ``<out_dir>/unit_triage_<YYYYMMDD>_<HHMMSS>.pkl``. All parameters are
defined at the top of this cell.

.. code-block:: python

    # Significance thresholds (default to detect_interesting_tuning_neurons)
    THRESHOLDS = {
        "z_threshold": 3.0,
        "min_consecutive_bins": 3,
        "vmi_alpha": 0.01,
        "vmi_min_bouts": 10,
        "spatial_info_bps_threshold": 0.5,
    }

    # One .txt of session roots per condition; authoritative catalog; output + data roots
    CONDITION_TO_SESSION_LIST = {
        "intact_female": "/mnt/falkner/.../ephys_courtship_intact_partners_sessions_list.txt",
        "mute_female": "/mnt/falkner/.../ephys_courtship_mute_female_sessions_list.txt",
    }
    CATALOG_PATH = "/mnt/falkner/Bartul/EPHYS/unit_catalog.csv"
    AGGREGATOR_OUT_DIR = "/mnt/falkner/Bartul/neuronal_tuning"
    DATA_ROOT = "/mnt/falkner/Bartul/Data"

    # None -> rebuild + save; "auto" -> reuse newest existing (rebuild if none);
    # "<abs path>" -> load that file verbatim
    AGGREGATOR_PKL = "auto"

    if AGGREGATOR_PKL is None:
        aggregator_pkl_path = aggregate_units_across_conditions(
            condition_to_session_list=CONDITION_TO_SESSION_LIST,
            catalog_path=CATALOG_PATH,
            out_dir=AGGREGATOR_OUT_DIR,
            data_root=DATA_ROOT,
            z_threshold=THRESHOLDS["z_threshold"],
            min_consecutive_bins=THRESHOLDS["min_consecutive_bins"],
            vmi_alpha=THRESHOLDS["vmi_alpha"],
            vmi_min_bouts=THRESHOLDS["vmi_min_bouts"],
            spatial_info_bps_threshold=THRESHOLDS["spatial_info_bps_threshold"],
            message_output=print,
        )
    elif AGGREGATOR_PKL == "auto":
        existing = sorted(Path(AGGREGATOR_OUT_DIR).glob("unit_triage_*.pkl"))
        aggregator_pkl_path = (
            existing[-1] if existing else None
        )  # newest existing; None case rebuilds
    else:
        aggregator_pkl_path = Path(AGGREGATOR_PKL)

* **THRESHOLDS** — significance knobs applied by the aggregator; default to the ``detect_interesting_tuning_neurons`` block, override here to sweep without re-computing tuning.
* **CONDITION_TO_SESSION_LIST** — one ``.txt`` of session roots (one per line) per condition; each condition becomes a block in every unit's pickle entry.
* **CATALOG_PATH** — the authoritative ``unit_catalog.csv`` supplying each cluster's anatomy (reassigned again in the anatomy cell below).
* **AGGREGATOR_OUT_DIR** / **DATA_ROOT** — where ``unit_triage_<YYYYMMDD>_<HHMMSS>.pkl`` is written, and the root the aggregator walks for ``*_tuning_curves_data.pkl``.
* **AGGREGATOR_PKL** — pickle source: ``None`` rebuilds and saves (~minutes, ~1.9 GB); ``"auto"`` reuses the newest existing pickle (rebuild only if none); an absolute path loads that file verbatim.

**Anatomy figures.** Build an ``AnatomyFigureMaker`` and render the corpus-level panels straight from
``unit_catalog.csv`` and the per-session Kilosort outputs (they do not need the
aggregator pickle): a two-panel recording-yield bar, per-probe unit waveforms on a
four-shank schematic, a 360° rotating-brain video, and a static unit-positions
still. Each method writes one timestamped file; output directory, format and dpi
default to the ``figures`` block of ``visualizations_settings.json``. ``CATALOG_PATH``
is reassigned at the top of the first cell, and each figure cell sets its own knobs
at the top.

.. code-block:: python

    # authoritative unit catalog (also read by the aggregator cell above)
    CATALOG_PATH = "/mnt/falkner/Bartul/EPHYS/unit_catalog.csv"

    anatomy_maker = AnatomyFigureMaker(
        catalog_path=CATALOG_PATH,
        visualizations_parameter_dict=vis_settings,
        message_output=print,
    )

    yield_path = anatomy_maker.make_recording_yield_figure(out_dir=FIG_OUT_DIR)

    # Per-probe waveform target session (top of the waveform cell)
    ANATOMY_WAVEFORM_MOUSE = "158114_2"
    ANATOMY_WAVEFORM_SESSION = "20241115_162223"
    for probe, schematic_side in (("imec0", "right"), ("imec1", "left")):
        wf_path = anatomy_maker.make_unit_waveform_figure(
            mouse_id=ANATOMY_WAVEFORM_MOUSE,
            session_id=ANATOMY_WAVEFORM_SESSION,
            probe_filter=probe,
            schematic_side=schematic_side,
            out_dir=FIG_OUT_DIR,
        )

    # 360 degree rotating brain video (top of the video cell)
    video_n_frames = 180
    video_fps = 30
    video_format = "mp4"
    video_path = anatomy_maker.make_unit_positions_video(
        n_frames=video_n_frames,
        fps=video_fps,
        video_format=video_format,
        out_dir=FIG_OUT_DIR,
    )

    # Unit-positions still (top of the still cell)
    still_fig_format = "svg"
    still_view_elev = 35.0
    still_view_azim = -45.0
    still_path = anatomy_maker.make_unit_positions_figure(
        fig_format=still_fig_format,
        view_elev=still_view_elev,
        view_azim=still_view_azim,
        out_dir=FIG_OUT_DIR,
    )

* **CATALOG_PATH** — the same authoritative ``unit_catalog.csv`` from the aggregator cell, reassigned at the top of this cell so anatomy figures can be run independently.
* **ANATOMY_WAVEFORM_MOUSE** / **ANATOMY_WAVEFORM_SESSION** — the one ``(mouse_id, session_id)`` pair whose top-amplitude single-unit (SU)-somatic templates are drawn, one figure per probe (schematic side flips so rostral sits on the left for both).
* **video_n_frames** / **video_fps** / **video_format** — frame count, frame rate, and container of the 360° rotating-brain video; ``n_frames`` / ``fps`` set both duration and rotation smoothness.
* **still_fig_format** / **still_view_elev** / **still_view_azim** — format and camera elevation / azimuth of the static unit-positions still (tilted from above, AP axis running left-right so both hemispheres are visible).

**VMI summary figures.** Build a ``NeuronalTuningFigureMaker`` filtered by the unit-selection knobs, then
render the population figures from the aggregator pickle. The unit-selection knobs
live at the top of the builder cell (applied at the figure layer, so changing them
needs no rebuild); every surviving unit is then grouped into the periaqueductal gray (PAG) / midbrain reticular nucleus
(MRN) / ventral tegmental area (VTA) / MB / CENT / SC / Other anatomy buckets. The remaining knobs are
dispersed into the cells that use them: ``PETH_DIRECTION`` at the top of the peri-event time histogram (PETH)
cell, ``PROPERTY_DIRECTION`` at the top of the property-tuning cell, and
``figure_condition`` reassigned at the top of *each* behavioural cell. The VMI
figures (FR-confound diagnostic, cross-condition stability,
magnitude-vs-consistency, sign-flip summary, the PAG
anatomical gradient, per-region distribution histograms) come first, then the USV
PETH / property / category suites, then the two
condition-scoped behavioural summaries. Each call returns the written path.

.. code-block:: python

    # Unit-selection filter (top of the builder cell)
    UNIT_KSLABELS = ("good",)  # e.g. ("good",) or ("good", "mua")
    UNIT_SOMATIC_FILTER = "somatic"  # "somatic" | "non_somatic" | "both"

    tuning_maker = NeuronalTuningFigureMaker(
        visualizations_parameter_dict=vis_settings,
        message_output=print,
        kslabels=UNIT_KSLABELS,
        somatic_filter=UNIT_SOMATIC_FILTER,
    )

    # VMI population figures
    vmi_fr_path = tuning_maker.make_vmi_fr_confound_figure(
        triage_pkl_path=aggregator_pkl_path, out_dir=FIG_OUT_DIR
    )
    vmi_stab_path = tuning_maker.make_vmi_cross_condition_stability_figure(
        triage_pkl_path=aggregator_pkl_path, out_dir=FIG_OUT_DIR
    )
    vmi_mag_path = tuning_maker.make_vmi_magnitude_consistency_figure(
        triage_pkl_path=aggregator_pkl_path, out_dir=FIG_OUT_DIR
    )
    vmi_sf_path = tuning_maker.make_vmi_sign_flip_summary_figure(
        triage_pkl_path=aggregator_pkl_path, out_dir=FIG_OUT_DIR
    )
    vmi_pag_anat_path = tuning_maker.make_pag_anatomical_gradient_figure(
        triage_pkl_path=aggregator_pkl_path, out_dir=FIG_OUT_DIR
    )
    vmi_dist_path = tuning_maker.make_vmi_distribution_figure(
        triage_pkl_path=aggregator_pkl_path, out_dir=FIG_OUT_DIR
    )

    # USV PETH / property / category suites
    PETH_DIRECTION = "excit"  # "excit" or "suppress" (embedded in filename)
    peth_timing_path = tuning_maker.make_peth_timing_distribution_figure(
        triage_pkl_path=aggregator_pkl_path,
        direction=PETH_DIRECTION,
        out_dir=FIG_OUT_DIR,
    )

    PROPERTY_DIRECTION = "excit"  # "excit" or "suppress"
    property_paths = tuning_maker.make_all_property_tuning_distribution_figures(
        triage_pkl_path=aggregator_pkl_path,
        direction=PROPERTY_DIRECTION,
        out_dir=FIG_OUT_DIR,
    )

    category_paths = tuning_maker.make_all_category_figures(
        triage_pkl_path=aggregator_pkl_path,
        out_dir=FIG_OUT_DIR,
    )

    # Behavioural summaries (figure_condition reassigned at the top of EACH cell)
    figure_condition = "intact_female"
    beh_summary_path = tuning_maker.make_behavioral_tuning_summary_figure(
        triage_pkl_path=aggregator_pkl_path,
        condition=figure_condition,
        out_dir=FIG_OUT_DIR,
    )

    figure_condition = "intact_female"
    overlap_venn_path = tuning_maker.make_per_region_overlap_venn_figure(
        triage_pkl_path=aggregator_pkl_path,
        condition=figure_condition,
        out_dir=FIG_OUT_DIR,
    )

* **UNIT_KSLABELS** — Kilosort labels to include, e.g. ``("good",)`` or ``("good", "mua")``; applied at the figure layer, so changing it needs no pickle rebuild.
* **UNIT_SOMATIC_FILTER** — ``"somatic"`` | ``"non_somatic"`` | ``"both"``; with the default ``("good",)`` + ``"somatic"`` reproducing the historical good-and-somatic scope.
* **PETH_DIRECTION** — ``"excit"`` or ``"suppress"``; switches the PETH timing overlay and is embedded in the filename (``peth_excit_*`` / ``peth_suppress_*``). Set at the top of the PETH cell.
* **PROPERTY_DIRECTION** — ``"excit"`` or ``"suppress"``; selects excit- vs suppress-tuned units for the eight per-property distribution figures. Set at the top of the property cell.
* **figure_condition** — one key of ``CONDITION_TO_SESSION_LIST``; selects which condition's per-session evidence feeds the behavioural heatmap and the overlap Venns. Reassigned at the top of each of the two behavioural cells.

Source: `neuronal_tuning_summary.ipynb <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/neuronal_tuning_summary.ipynb>`_.

USV neuronal coactivity analyses
--------------------------------
**usv_neuronal_coactivity_analyses.ipynb** asks how coordinated a brain-region
population (PAG by default) is during one class of USV versus another. For a
30 ms window locked to each call onset it computes three coactivity metrics —
pairwise spike-count correlation (``r_sc``), population-vector cosine similarity
(``similarity``), and population-vector Pearson correlation (``pop_corr``) — and
tests them with a pooled trial-count bootstrap, a chained circular-shuffle null,
and a direct label-permutation test. It reads nothing from
``analyses_settings.json``: every knob lives in the **Parameters** cell.

The statistics and figures are factored out of the notebook: the compute lives
in ``usv_playpen.analyses.neuronal_coactivity_engine`` (each section calls
``run_group_comparison``, ``compare_groups``, ``pool_group_count_matrices`` or
``per_session_group_metrics``) and the plots plus printed tables in
``usv_playpen.visualizations.make_coactivity_figures`` (the ``plot_*`` and
``summarize_*`` helpers), so every section below is a short call into those two
modules. The cells are organised as *(1) imports*, *(2) parameters — every
tweakable knob*, *(3) setup & data load* (which builds the ``sessions_data``
consumed by every later section), then *(4) per-section compute + plot*. Run
**Imports**, **Parameters**, and **Setup & load data** first; each later cell is
an independent compute-plus-plot step you re-run on top of them.

**Imports.** Import the plotting stack, the path normaliser, the coactivity engine (as a
module, so every routine is called as ``engine.<fn>``), and the ``plot_*`` /
``summarize_*`` figure helpers, then apply the repository plot style.

.. code-block:: python

    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as st

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.plot_style import apply_plot_style
    import usv_playpen.analyses.neuronal_coactivity_engine as engine
    from usv_playpen.visualizations.make_coactivity_figures import (
        plot_acoustic_confound,
        plot_amplitude_stratified,
        plot_cross_animal_slope,
        plot_null_distributions,
        plot_per_session_pop_corr,
        summarize_acoustic_confound,
        summarize_amplitude_stratified,
        summarize_group_comparison,
    )

    apply_plot_style()

**Parameters.** Every user-tweakable knob for a run lives in this one cell: the segmentation
column and the two category-id groups to contrast, the three-criteria unit
filter, the animal→sessions map and chosen animal, the coactivity
hyperparameters, and the per-group plot colours. Nothing downstream redefines
these. Paths are written ``/mnt/falkner/...`` and wrapped in ``configure_path()``
so they resolve on macOS (``/Volumes/falkner``) too.

.. code-block:: python

    # Segmentation configuration
    CATEGORY_COLUMN = "qlvm_supercategory"
    GROUP_A_IDS = [1]
    GROUP_A_LABEL = "complex"
    GROUP_B_IDS = [7]
    GROUP_B_LABEL = "simple"

    # Unit-filter configuration (cluster_group + somatic + brain area)
    CATALOG_PATH = pathlib.Path(
        configure_path("/mnt/falkner/Bartul/EPHYS/unit_catalog.csv")
    )
    UNIT_BRAIN_AREAS = {"PAG"}
    UNIT_REQUIRE_SOMATIC = True
    UNIT_CLUSTER_GROUP = "good"

    # Animal -> sessions mapping. Sessions can span several recording days;
    # Kilosort runs per day (cluster IDs aren't stable across days), so the
    # loader picks the single-day block with the most units passing the catalog
    # filters. Sessions here and further animals truncated for brevity.
    DATA_ROOT = pathlib.Path(configure_path("/mnt/falkner/Bartul/Data"))
    ANIMALS_TO_SESSIONS: dict[str, list[str]] = {
        "178621_2": [
            "20250927_142335",
            "20250927_145144",
            "20250927_151825",
            "20250928_172408",
            "20250928_175135",
            "20250928_182348",
        ],
        # ...
    }
    CHOSEN_ANIMAL = "178621_2"

    # Coactivity hyperparameters
    SEED = 0  # base RNG seed; each routine draws a reproducible stream at SEED + offset
    USV_BOOTSTRAP_NUM = 300
    N_BOOT_ITERATIONS = 1000
    N_SHUFFLES = 1000
    N_PERMUTATIONS = 1000
    WINDOW_S = 0.030
    PER_SESSION_N_SHUFFLES = 500

    # Group plotting colours (hex)
    GROUP_A_COLOR = "#DC143C"
    GROUP_B_COLOR = "#1E90FF"
    NULL_COLOR = "#808080"
    THRESHOLD_COLOR = "#000000"

* **CATEGORY_COLUMN** / **GROUP_A_IDS** / **GROUP_B_IDS** — the ``usv_summary`` column that labels each call and the two sets of category ids contrasted (default ``complex`` [1] vs ``simple`` [7]); ``*_LABEL`` names them in tables and plots.
* **CATALOG_PATH** / **UNIT_BRAIN_AREAS** / **UNIT_REQUIRE_SOMATIC** / **UNIT_CLUSTER_GROUP** — the unit-catalog file and the three-criteria filter (region, somatic waveform, Kilosort ``cluster_group``) applied to select the population.
* **ANIMALS_TO_SESSIONS** / **CHOSEN_ANIMAL** / **DATA_ROOT** — the per-animal session lists (Kilosort is per-day, so the loader keeps the single best-populated day), the focal animal for single-animal cells, and the data root.
* **SEED** — base RNG seed; each stochastic routine offsets it (``SEED + k``) for an independent, reproducible stream.
* **USV_BOOTSTRAP_NUM** / **N_BOOT_ITERATIONS** / **N_SHUFFLES** / **N_PERMUTATIONS** / **PER_SESSION_N_SHUFFLES** — matched trial count and iteration counts for the pooled bootstrap, chained/per-session shuffle nulls, and label permutation.
* **WINDOW_S** — the post-onset window (30 ms) over which spikes are counted per call.
* **GROUP_A_COLOR** / **GROUP_B_COLOR** / **NULL_COLOR** / **THRESHOLD_COLOR** — hex colours for the two groups, the null histograms, and the 99th-percentile / threshold lines.

**Setup & load data.** Read the unit catalog once with ``engine.load_unit_catalog``, then load the
chosen animal's data through ``engine.load_animal_sessions``: the three-criteria
unit filter (``cluster_group`` + ``somatic`` + ``brain_area``), the
single-best-day population selection (Kilosort is per-day, so units aren't
comparable across days), and the per-session ``group_a``/``group_b`` category
split all happen inside the engine. This builds the ``sessions_data`` that every
later section consumes. Edit inputs in **Parameters** — this cell should not need
changing.

.. code-block:: python

    catalog = engine.load_unit_catalog(CATALOG_PATH)

    print(f"Trial split:  `{CATEGORY_COLUMN}`")
    print(f"  group A ({GROUP_A_LABEL}) = IDs {GROUP_A_IDS}")
    print(f"  group B ({GROUP_B_LABEL}) = IDs {GROUP_B_IDS}")
    print(
        f"Unit filter:  cluster_group='{UNIT_CLUSTER_GROUP}'  "
        f"somatic={UNIT_REQUIRE_SOMATIC}  brain_area in {sorted(UNIT_BRAIN_AREAS) or 'ANY'}"
    )
    print(f"Chosen animal: {CHOSEN_ANIMAL}")

    sessions_data = engine.load_animal_sessions(
        CHOSEN_ANIMAL,
        ANIMALS_TO_SESSIONS[CHOSEN_ANIMAL],
        data_root=DATA_ROOT,
        catalog=catalog,
        category_column=CATEGORY_COLUMN,
        group_a_ids=GROUP_A_IDS,
        group_b_ids=GROUP_B_IDS,
        cluster_group=UNIT_CLUSTER_GROUP,
        require_somatic=UNIT_REQUIRE_SOMATIC,
        brain_areas=UNIT_BRAIN_AREAS,
    )
    n_common = len(next(iter(sessions_data))["neural_data"]) if sessions_data else 0
    print(
        f"Loaded {len(sessions_data)} sessions for {CHOSEN_ANIMAL}; common filtered units = {n_common}"
    )

**Acoustic confound.** The fixed 30 ms window equalises call duration, but the two categories could
still differ acoustically (loudness, pitch) in ways that drive population
activity independently of call identity. For every call
``engine.compute_group_acoustics`` reads the loudest-channel waveform snippet
``[onset, onset + WINDOW_S)`` and computes four features — root-mean-square (RMS)
amplitude plus energy-weighted mean frequency and bandwidth, and the peak (loudest-bin) frequency.
``summarize_acoustic_confound`` then prints a per-feature Mann–Whitney U +
Cohen's *d* table over the pooled ``complex``-vs-``simple`` distributions and
``plot_acoustic_confound`` overlays density histograms. This is diagnostic only:
it tells you *which* features differ so a control can be chosen;
``ACOUSTIC_FEATURES`` / ``ACOUSTIC_LABELS`` are this section's only local knobs.

.. code-block:: python

    # Features checked + their axis labels (this section's only knobs).
    ACOUSTIC_FEATURES = ["rms", "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz"]
    ACOUSTIC_LABELS = {
        "rms": "RMS amplitude (a.u.)",
        "mean_freq_hz": "mean frequency (Hz)",
        "peak_freq_hz": "peak frequency (Hz)",
        "freq_bandwidth_hz": "frequency bandwidth (Hz)",
    }

    # Pool per-call features across the animal's sessions, per group.
    group_a_acoustics = {feature: [] for feature in ACOUSTIC_FEATURES}
    group_b_acoustics = {feature: [] for feature in ACOUSTIC_FEATURES}
    for sess in sessions_data:
        a_feats = engine.compute_group_acoustics(sess, "group_a_df", WINDOW_S)
        b_feats = engine.compute_group_acoustics(sess, "group_b_df", WINDOW_S)
        for feature in ACOUSTIC_FEATURES:
            group_a_acoustics[feature].append(a_feats[feature])
            group_b_acoustics[feature].append(b_feats[feature])
    group_a_acoustics = {
        f: np.concatenate(v) if v else np.array([]) for f, v in group_a_acoustics.items()
    }
    group_b_acoustics = {
        f: np.concatenate(v) if v else np.array([]) for f, v in group_b_acoustics.items()
    }

    # Per-feature Mann-Whitney U + Cohen's d table, then overlaid density histograms.
    print(
        summarize_acoustic_confound(
            group_a_acoustics,
            group_b_acoustics,
            features=ACOUSTIC_FEATURES,
            chosen_animal=CHOSEN_ANIMAL,
            label_a=GROUP_A_LABEL,
            label_b=GROUP_B_LABEL,
        )
    )
    plot_acoustic_confound(
        group_a_acoustics,
        group_b_acoustics,
        features=ACOUSTIC_FEATURES,
        feature_labels=ACOUSTIC_LABELS,
        chosen_animal=CHOSEN_ANIMAL,
        label_a=GROUP_A_LABEL,
        label_b=GROUP_B_LABEL,
        group_a_color=GROUP_A_COLOR,
        group_b_color=GROUP_B_COLOR,
    )

**Compute.** The core statistical cell, now a single call. ``engine.run_group_comparison``
runs the full single-animal pipeline: pool per-session count matrices → matched-N
pooled bootstrap of each group plus a direct label-permutation test → a chained
circular-shuffle null per group → the per-session observed-metric breakdown, with
every stochastic step seeded from ``SEED``. ``summarize_group_comparison`` then
prints the per-session deltas, each group vs its chained null, and the direct
group-A-vs-group-B permutation test.

.. code-block:: python

    # Full single-animal pipeline in one call: pool per-session count matrices -> matched-N
    # pooled bootstrap of each group + a direct label-permutation test -> chained
    # circular-shuffle null per group -> per-session observed-metric breakdown. Every
    # stochastic step derives an independent, reproducible stream from `SEED`.
    results = engine.run_group_comparison(
        sessions_data,
        window_s=WINDOW_S,
        bootstrap_n=USV_BOOTSTRAP_NUM,
        n_boot=N_BOOT_ITERATIONS,
        n_shuffles=N_SHUFFLES,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
    )

    # Per-session deltas, each group vs its chained null, and the direct A-vs-B permutation test.
    print(summarize_group_comparison(results, label_a=GROUP_A_LABEL, label_b=GROUP_B_LABEL))

* **WINDOW_S** — the per-call spike-count window used to build every matrix.
* **USV_BOOTSTRAP_NUM** — the matched trial count both groups are bootstrapped (and onset-sampled) to.
* **N_BOOT_ITERATIONS** / **N_SHUFFLES** / **N_PERMUTATIONS** — iteration counts for the pooled bootstrap, the chained null, and the label permutation.
* **SEED** — the base seed the engine offsets per internal routine for independent, reproducible streams.

**Null distributions.** ``plot_null_distributions`` draws a 3-metric × 2-group grid from the ``results``
object: each group's chained-null histogram overlaid with its observed
pooled-bootstrap mean and the null's 99th percentile.

.. code-block:: python

    plot_null_distributions(
        results,
        category_column=CATEGORY_COLUMN,
        group_a_ids=GROUP_A_IDS,
        group_b_ids=GROUP_B_IDS,
        label_a=GROUP_A_LABEL,
        label_b=GROUP_B_LABEL,
        group_a_color=GROUP_A_COLOR,
        group_b_color=GROUP_B_COLOR,
        null_color=NULL_COLOR,
        threshold_color=THRESHOLD_COLOR,
    )

**Per-session pop_corr.** ``engine.per_session_group_metrics`` returns, per session, each group's observed
``pop_corr`` plus a within-session circular-shuffle null (both groups' onsets
pooled, so the null reflects neural-timing shifts) using
``PER_SESSION_N_SHUFFLES`` shuffles; ``plot_per_session_pop_corr`` draws one panel
per session. Sessions with fewer than two trials in either group are dropped.

.. code-block:: python

    # Per-session observed pop_corr for both groups against a within-session circular-shuffle
    # null (both groups' onsets pooled, so the null reflects neural-timing shifts). Sessions
    # with < 2 trials in either group are dropped.
    per_session_rows = engine.per_session_group_metrics(
        sessions_data,
        WINDOW_S,
        n_shuffles=PER_SESSION_N_SHUFFLES,
        seed=SEED + 100,
    )
    plot_per_session_pop_corr(
        per_session_rows,
        chosen_animal=CHOSEN_ANIMAL,
        category_column=CATEGORY_COLUMN,
        label_a=GROUP_A_LABEL,
        label_b=GROUP_B_LABEL,
        group_a_color=GROUP_A_COLOR,
        group_b_color=GROUP_B_COLOR,
        null_color=NULL_COLOR,
        threshold_color=THRESHOLD_COLOR,
    )

* **PER_SESSION_N_SHUFFLES** — shuffle count for each session's null (smaller than the chained null since it runs per session).
* **SEED** — offset (``SEED + 100``) for the per-session nulls' stream.

**Cross-animal summary.** Loops over every focal animal in ``ANIMALS_TO_SESSIONS``, loading its single best
day and pooling + comparing via ``engine.pool_group_count_matrices`` +
``engine.compare_groups`` (the matched-N bootstrap plus label-permutation test).
It collects each animal's ``pop_corr`` means and two-tailed permutation *p*, then
``plot_cross_animal_slope`` draws the per-animal slope from ``pop_corr(group A)``
to ``pop_corr(group B)``, coloured by the permutation significance.

.. code-block:: python

    # Per animal: load its best-day sessions, pool the count matrices, and run the matched-N
    # bootstrap + label-permutation comparison (engine.compare_groups). Collect the per-animal
    # pop_corr means + two-tailed permutation p for the slope plot.
    cross_animal_results: dict[str, dict] = {}
    for animal_idx, (animal_id, session_names) in enumerate(ANIMALS_TO_SESSIONS.items()):
        print(f"Animal {animal_id} ({len(session_names)} sessions) ...", flush=True)
        animal_sessions = engine.load_animal_sessions(
            animal_id,
            session_names,
            data_root=DATA_ROOT,
            catalog=catalog,
            category_column=CATEGORY_COLUMN,
            group_a_ids=GROUP_A_IDS,
            group_b_ids=GROUP_B_IDS,
            cluster_group=UNIT_CLUSTER_GROUP,
            require_somatic=UNIT_REQUIRE_SOMATIC,
            brain_areas=UNIT_BRAIN_AREAS,
        )
        if not animal_sessions:
            print("  no sessions loaded, skipping")
            continue
        pooled_a, pooled_b = engine.pool_group_count_matrices(animal_sessions, WINDOW_S)
        if pooled_a.shape[1] < 1 or pooled_b.shape[1] < 1:
            print("  insufficient trials, skipping")
            continue
        bootstrap_target = min(USV_BOOTSTRAP_NUM, pooled_a.shape[1], pooled_b.shape[1])
        comparison = engine.compare_groups(
            pooled_a,
            pooled_b,
            bootstrap_n=bootstrap_target,
            n_boot=N_BOOT_ITERATIONS,
            n_permutations=N_PERMUTATIONS,
            seed=SEED + 200 + 3 * animal_idx,
        )
        pop_a_obs = float(np.mean(comparison["boot_a"]["pop_corr"]))
        pop_b_obs = float(np.mean(comparison["boot_b"]["pop_corr"]))
        perm = comparison["perm"]["pop_corr"]
        cross_animal_results[animal_id] = {
            "n_sessions": len(animal_sessions),
            "n_a": pooled_a.shape[1],
            "n_b": pooled_b.shape[1],
            "n_units": pooled_a.shape[0],
            "pop_a": pop_a_obs,
            "pop_b": pop_b_obs,
            "p_two": perm["p_two_tailed"],
            "p_a_gt_b": perm["p_a_gt_b"],
            "z": perm["z_score"],
        }
        print(
            f"  units={pooled_a.shape[0]:>3}  n_a={pooled_a.shape[1]:>5}  n_b={pooled_b.shape[1]:>5}"
            f"  pop_a={pop_a_obs:+.4f}  pop_b={pop_b_obs:+.4f}"
            f"  Δ={pop_a_obs - pop_b_obs:+.4f}  p_two={perm['p_two_tailed']:.3f}  Z={perm['z_score']:+.2f}"
        )

    plot_cross_animal_slope(
        cross_animal_results,
        category_column=CATEGORY_COLUMN,
        group_a_ids=GROUP_A_IDS,
        group_b_ids=GROUP_B_IDS,
        label_a=GROUP_A_LABEL,
        label_b=GROUP_B_LABEL,
        group_a_color=GROUP_A_COLOR,
        group_b_color=GROUP_B_COLOR,
        null_color=NULL_COLOR,
        threshold_color=THRESHOLD_COLOR,
    )

* **ANIMALS_TO_SESSIONS** — the full set of focal mice looped over (not just ``CHOSEN_ANIMAL``).
* **USV_BOOTSTRAP_NUM** / **N_BOOT_ITERATIONS** — matched-N target (capped at each animal's trial count) and bootstrap iterations.
* **N_PERMUTATIONS** — permutation count for the per-animal A-vs-B test.
* **SEED** — offset per animal (``SEED + 200 + 3 * animal_idx``) for independent streams.

**Amplitude-stratified.** The confound check shows the groups differ acoustically (complex calls are
louder); this cell asks whether loudness actually *drives* the ``pop_corr`` gap.
It bins all calls by their 30 ms RMS into quantile bins and, for bins holding at
least ``MIN_BIN_TRIALS`` of both groups (the overlap region), bootstraps each
group to a matched N and re-runs the same ``complex``-vs-``simple``
label-permutation test (``engine.compare_groups``) *within* the loudness-matched
bin. ``summarize_amplitude_stratified`` prints the per-bin table and
``plot_amplitude_stratified`` draws the per-bin ``pop_corr`` with the
unstratified matched-N bootstrap means as references. If the gap persists within
bins, loudness is not the explanation; if it collapses, amplitude is a confound.
``N_AMPLITUDE_BINS`` and ``MIN_BIN_TRIALS`` are local knobs; the per-call RMS is
reused from the acoustic confound-check cell, so re-run that first.

.. code-block:: python

    # Stratification knobs.
    N_AMPLITUDE_BINS = 5
    MIN_BIN_TRIALS = 15  # required per group, per bin, for a bin to be compared

    # Reuse the per-call RMS from the acoustic confound check above (aligned to the same
    # session + dataframe order as the pooled spike-count matrices built here).
    a_rms = group_a_acoustics["rms"]
    b_rms = group_b_acoustics["rms"]
    a_counts, b_counts = engine.pool_group_count_matrices(sessions_data, WINDOW_S)
    assert (
        a_counts.shape[1] == a_rms.shape[0] and b_counts.shape[1] == b_rms.shape[0]
    ), "RMS / count-matrix misalignment — re-run the acoustic confound check cell first."

    # Quantile bin edges over the pooled finite-positive RMS of both groups.
    pooled_rms = np.concatenate([a_rms, b_rms])
    pooled_rms = pooled_rms[np.isfinite(pooled_rms) & (pooled_rms > 0)]
    bin_edges = np.quantile(pooled_rms, np.linspace(0.0, 1.0, N_AMPLITUDE_BINS + 1))
    bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)  # make the top edge inclusive

    stratified_rows = []
    for bin_idx in range(N_AMPLITUDE_BINS):
        lo, hi = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        a_sel = np.isfinite(a_rms) & (a_rms >= lo) & (a_rms < hi)
        b_sel = np.isfinite(b_rms) & (b_rms >= lo) & (b_rms < hi)
        n_a, n_b = int(a_sel.sum()), int(b_sel.sum())
        row = {
            "lo": lo,
            "hi": hi,
            "n_a": n_a,
            "n_b": n_b,
            "pop_a": np.nan,
            "pop_b": np.nan,
            "p_two": np.nan,
        }
        if n_a >= MIN_BIN_TRIALS and n_b >= MIN_BIN_TRIALS:
            n_match = min(n_a, n_b)  # matched N within the bin -> fair pop_corr comparison
            comparison = engine.compare_groups(
                a_counts[:, a_sel],
                b_counts[:, b_sel],
                bootstrap_n=n_match,
                n_boot=N_BOOT_ITERATIONS,
                n_permutations=N_PERMUTATIONS,
                seed=SEED + 300 + 3 * bin_idx,
            )
            row["pop_a"] = float(np.mean(comparison["boot_a"]["pop_corr"]))
            row["pop_b"] = float(np.mean(comparison["boot_b"]["pop_corr"]))
            row["p_two"] = comparison["perm"]["pop_corr"]["p_two_tailed"]
        stratified_rows.append(row)

    # Unstratified reference (all trials, matched-N bootstrap) for context.
    n_overall = min(a_counts.shape[1], b_counts.shape[1], USV_BOOTSTRAP_NUM)
    pop_a_overall = float(
        np.mean(
            engine.bootstrap_coactivity_distribution(
                a_counts, n_overall, N_BOOT_ITERATIONS, seed=SEED + 600
            )["pop_corr"]
        )
    )
    pop_b_overall = float(
        np.mean(
            engine.bootstrap_coactivity_distribution(
                b_counts, n_overall, N_BOOT_ITERATIONS, seed=SEED + 601
            )["pop_corr"]
        )
    )

    print(
        summarize_amplitude_stratified(
            stratified_rows,
            pop_a_overall,
            pop_b_overall,
            chosen_animal=CHOSEN_ANIMAL,
            n_bins=N_AMPLITUDE_BINS,
            label_a=GROUP_A_LABEL,
            label_b=GROUP_B_LABEL,
        )
    )
    plot_amplitude_stratified(
        stratified_rows,
        pop_a_overall,
        pop_b_overall,
        chosen_animal=CHOSEN_ANIMAL,
        label_a=GROUP_A_LABEL,
        label_b=GROUP_B_LABEL,
        group_a_color=GROUP_A_COLOR,
        group_b_color=GROUP_B_COLOR,
        threshold_color=THRESHOLD_COLOR,
    )

* **N_AMPLITUDE_BINS** / **MIN_BIN_TRIALS** — the quantile-bin count and the per-group, per-bin trial floor for a bin to be compared.
* **WINDOW_S** — window for the count matrices (aligned to the reused per-call RMS).
* **USV_BOOTSTRAP_NUM** — cap for the unstratified reference bootstrap (per-bin N is matched to the smaller group instead).
* **N_BOOT_ITERATIONS** / **N_PERMUTATIONS** — bootstrap and permutation iteration counts per bin.
* **SEED** — offset per bin and stream (``SEED + 300 + 3 * bin_idx``, ``SEED + 600/601``).
* **GROUP_A_COLOR** / **GROUP_B_COLOR** / **THRESHOLD_COLOR** — per-group bin lines, dashed unstratified reference lines, and the ``p < 0.05`` markers.

Source: `usv_neuronal_coactivity_analyses.ipynb <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_neuronal_coactivity_analyses.ipynb>`_.

Inter-USV interval analyses
---------------------------
**inter_usv_interval_analyses.ipynb** fits and visualises mixture models on
the distribution of inter-USV intervals (in seconds, log-transformed) across one
or more sessions. Compute is split from plotting: the compute cells build a master
interval DataFrame, run an information criterion (IC) sweep and a bootstrap likelihood-ratio test (LRT) over candidate component
counts for the selected mixture family (Gaussian or Student-t, per ``model_class``), and persist everything to a single
self-describing HDF5 archive. The plot cells then read that archive back, so figures
can be re-rendered without refitting — even across kernel restarts.

Two cells configure everything — **Imports** (styling, palette, and the
settings JSONs) and **Configuration** (session lists, interval modes, and plot
knobs) — after which the compute cells run once and the plot cells re-read the
newest archive.

**Imports.** Import the interval-summary helpers, apply the shared plot style, and load the
``visualizations_settings.json`` / ``analyses_settings.json`` blocks that drive the
run (the cell also enables ``autoreload`` so source edits are picked up without a
kernel restart).

.. code-block:: python

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.plot_style import apply_plot_style
    from usv_playpen.visualizations.figure_io import save_figure
    import usv_playpen.visualizations.usv_interval_summary_statistics as ivs

    apply_plot_style()

    base_path = Path.cwd().parent
    with open(
        base_path / "_parameter_settings" / "visualizations_settings.json"
    ) as vis_settings_file:
        vis_settings = json.load(vis_settings_file)
    with open(
        base_path / "_parameter_settings" / "analyses_settings.json"
    ) as ana_settings_file:
        ana_settings = json.load(ana_settings_file)

    male_color = vis_settings["male_colors"][0]
    female_color = vis_settings["female_colors"][0]
    usv_interval_cfg = ana_settings["compute_inter_usv_interval_distributions"]

* **usv_interval_cfg** — the ``compute_inter_usv_interval_distributions`` block; every numeric compute / display knob lives here, so the notebook itself only assigns convenience aliases.
* **male_color** / **female_color** — per-sex palette entries pulled from the visualizations settings.

**Configuration.** Resolve the session lists to include, name the two interval modes, and read the
plot-only knobs straight from the settings block. This is the single place to
change what gets analysed.

.. code-block:: python

    output_directory = usv_interval_cfg["output_directory"]
    session_lists = [
        str(Path(configure_path(p))) for p in usv_interval_cfg["session_lists"]
    ]

    interval_types = ("s2s", "e2s")
    mode_label = {
        "s2s": "start-to-start USV intervals",
        "e2s": "end-to-start USV intervals",
    }

    plot_log_xlims = tuple(usv_interval_cfg["plot_log_xlims"])
    bins_per_sex = usv_interval_cfg["bins_per_sex"]
    tau = usv_interval_cfg["tau"]
    model_class = usv_interval_cfg["model_class"]

* **output_directory** — where the HDF5 archive is written and where the plot cells look for the newest run.
* **session_lists** — the configured session-list files, each ``configure_path``-resolved to the host OS.
* **interval_types** / **mode_label** — the two interval definitions (``s2s`` start-to-start, ``e2s`` end-to-start) and their human-readable titles; every compute and plot cell loops over these.
* **model_class** — ``"gauss"`` or ``"t"``, selecting the Gaussian or Student-t mixture family for the whole run.
* **plot_log_xlims** / **bins_per_sex** — plot-only knobs read straight from JSON (not archived in the HDF5); **tau** is likewise read from JSON but *is* archived in the HDF5.

**Compute the fits.** Run once. First, walk every session in the list, read its ``*_usv_summary.csv``,
compute consecutive inter-USV intervals for both modes, and append them with
sex metadata to one Polars DataFrame:

.. code-block:: python

    usv_interval_df, usv_interval_summary = ivs.build_master_usv_interval_dataframe(
        session_lists=session_lists,
        noise_col_id=usv_interval_cfg["noise_col_id"],
        noise_categories=usv_interval_cfg["noise_categories"],
    )

Then fit each mixture family for ``K = n_components_min … n_components_max`` and
record every IC per ``K`` (the minimum-IC point is the preliminary model order):

.. code-block:: python

    mixture_model_fits_by_mode = {}
    for it in interval_types:
        sub = usv_interval_df.filter(pls.col("interval_type") == it)
        mixture_model_fits_by_mode[it] = ivs.run_bic_sweep(
            usv_interval_df=sub,
            n_components_min=usv_interval_cfg["n_components_min"],
            n_components_max=usv_interval_cfg["n_components_max"],
            n_repeats=usv_interval_cfg["n_repeats"],
            max_modes_reported=usv_interval_cfg["max_modes_reported"],
            random_seed_base=usv_interval_cfg["random_seed_base"],
            model_class=model_class,
        )

Next, the slow step: for each candidate ``K`` resample the data, refit, and build
the empirical null distribution of the log-likelihood-ratio of ``K`` vs. ``K-1``
components (the step-up selection rule is applied later, at save time):

.. code-block:: python

    lrt_sweep_by_mode = {}
    for it in interval_types:
        sub = usv_interval_df.filter(pls.col("interval_type") == it)
        male_arr = sub.filter(pls.col("sex") == "male")["interval_s"].to_numpy()
        female_arr = sub.filter(pls.col("sex") == "female")["interval_s"].to_numpy()
        lrt_sweep_by_mode[it] = ivs.run_bootstrap_lrt_sweep(
            intervals_by_key={"male": male_arr, "female": female_arr},
            n_components_min=usv_interval_cfg["n_components_min"],
            n_components_max=usv_interval_cfg["n_components_max"],
            B=usv_interval_cfg["bootstrap_lrt_B"],
            n_subsample=usv_interval_cfg["bootstrap_lrt_n_subsample"],
            model_class=model_class,
            n_init_obs=usv_interval_cfg["mixture_model_n_init"],
            n_init_boot=max(1, usv_interval_cfg["mixture_model_n_init"] - 7),
            reg_covar=usv_interval_cfg["mixture_model_reg_covar"],
            seed=usv_interval_cfg["random_seed_base"],
        )

Finally, bundle the master DataFrame, the IC sweep, the LRT results, and the
best-fit models into one ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5`` archive
(the step-up rule is applied here, so the per-mode selected-``K`` attrs match the
alpha / bonferroni settings):

.. code-block:: python

    h5_path = ivs.save_notebook_archive_to_h5(
        output_directory=output_directory,
        usv_interval_df=usv_interval_df,
        usv_interval_summary=usv_interval_summary,
        usv_interval_cfg=usv_interval_cfg,
        mixture_model_fits_by_mode=mixture_model_fits_by_mode,
        lrt_sweep_by_mode=lrt_sweep_by_mode,
    )

**Select the archive.** Pick the archive every plot cell reads from and decide whether figures are written
to disk. ``find_latest_archive`` picks the newest run so the plot cells survive
kernel restarts; assign ``h5_path`` manually to re-render an older run.

.. code-block:: python

    save_fig_bool = False
    h5_path = ivs.find_latest_archive(output_directory)

* **save_fig_bool** — when ``True``, each plot cell also calls ``save_figure`` into the configured figure directory.
* **h5_path** — the archive to plot from; override with an absolute path to compare cohorts across runs.

**Diagnostics.** Three read-only figures, each looping over both interval modes. First, the
sanity-check histogram of ``log(interval)`` per sex with its empirical density
(confirming the short intra-bout / long inter-bout bimodality before any model is
fit):

.. code-block:: python

    for it in interval_types:
        sub = ivs.load_intervals_from_h5(str(h5_path), it)
        fig, ax, hist_stats = ivs.plot_log_usv_interval_histograms(
            usv_interval_df=sub,
            bins=max(bins_per_sex.values()),
            male_color=male_color,
            female_color=female_color,
            xlims=plot_log_xlims,
        )
        ax.set_title(f"log_interval distribution -- {mode_label[it]}")
        if save_fig_bool:
            save_figure(fig, f"ivi_log_histogram_{it}", vis_settings)
        plt.show()

Second, the bootstrap LRT null distribution against the observed statistic for each
``K``-vs-``(K-1)`` comparison (the p-value is the right-tail mass):

.. code-block:: python

    for it in interval_types:
        sweep = ivs.load_lrt_sweep_from_h5(str(h5_path), it)
        fig, _ = ivs.plot_bootstrap_lrt_panel(sweep)
        fig.suptitle(f"Bootstrap LRT null distributions -- {mode_label[it]}", y=1.02)
        fig.tight_layout()
        if save_fig_bool:
            save_figure(fig, f"ivi_bootstrap_lrt_{it}", vis_settings)
        plt.show()
        selected = ivs.selected_K_from_h5(str(h5_path), it)

Third, the Bayesian information criterion (BIC) and Akaike information criterion (AIC) curves vs. ``K`` on twin axes (male left, female right),
with the LRT-selected ``K`` marked:

.. code-block:: python

    for it in interval_types:
        df_ic = ivs.load_mixture_model_fits_from_h5(str(h5_path), it)
        selected = ivs.selected_K_from_h5(str(h5_path), it)
        for ic_to_plot in ("bic", "aic"):
            fig, (ax_left, ax_right), _ = ivs.plot_ic_curves(
                df_results=df_ic,
                male_color=male_color,
                female_color=female_color,
                ic_col=ic_to_plot,
                selected_n_components=selected,
            )
            ax_left.set_title(f"{ic_to_plot.upper()} vs n_components -- {mode_label[it]}")
            if save_fig_bool:
                save_figure(fig, f"ivi_{ic_to_plot}_curve_{it}", vis_settings)
            plt.show()

**Best-fit overlay.** For each mode and sex, reconstruct the best-rep fitted mixture at the LRT-selected
``K`` directly from the archive (no refit) and overlay its density on the
``log(interval)`` histogram, with a quantile-quantile (Q-Q) inset comparing empirical to model
quantiles. Component log-means are marked with triangles; tail deviations are where
the Gaussian assumption breaks down, which is what the t-mixture variant exists for.

.. code-block:: python

    color_for = {"male": male_color, "female": female_color}
    model_class_label = {"t": "t-distribution", "gauss": "Gaussian"}.get(
        model_class, model_class
    )
    show_mixture_components = False

    for it in interval_types:
        sub = ivs.load_intervals_from_h5(str(h5_path), it)
        selected = ivs.selected_K_from_h5(str(h5_path), it)
        intervals_by_sex = {
            "male": sub.filter(pls.col("sex") == "male")["interval_s"].to_numpy(),
            "female": sub.filter(pls.col("sex") == "female")["interval_s"].to_numpy(),
        }
        for sex, intervals_sec in intervals_by_sex.items():
            if intervals_sec.size < 2 or sex not in selected:
                continue
            n_comp = int(selected[sex])
            mixture_model, mixture_model_order = ivs.load_best_fit_from_h5(
                h5_path=str(h5_path),
                interval_type=it,
                sex=sex,
                K=n_comp,
            )
            kw = (
                dict(auto_inset_below_legend=True, auto_inset_size=(0.45, 0.45))
                if sex == "male"
                else dict(auto_inset_below_legend=True, auto_inset_size=(0.25, 0.25))
            )
            fig_fit, ax_fit, fit_summary = ivs.plot_best_fit_with_annotations(
                intervals_sec=intervals_sec,
                mixture_model=mixture_model,
                mixture_model_order=mixture_model_order,
                color=color_for[sex],
                figsize=(5, 5),
                bins=bins_per_sex[sex],
                xlims=plot_log_xlims,
                tau=tau,
                legend_corner="upper right",
                show_components=show_mixture_components,
                **kw,
            )
            ax_fit.set_title(
                f"{mode_label[it].capitalize()} {model_class_label} "
                f"mixture model LRT-selected K={n_comp}"
            )
            if save_fig_bool:
                save_figure(fig_fit, f"ivi_best_fit_{model_class}_{sex}_{it}", vis_settings)
            plt.show()

**Fitted parameters.** Final summary: pull the LRT-selected best-rep components straight from the archive
and print the per-sex log-mean and log-sd for each component, ready to cite as
numerical results (the notebook formats them into a pasteable ``mixture_model_params`` JSON block).

.. code-block:: python

    for it in interval_types:
        selected = ivs.selected_K_from_h5(str(h5_path), it)
        for sex in ("male", "female"):
            if sex not in selected:
                continue
            K = int(selected[sex])
            mixture_model, _ = ivs.load_best_fit_from_h5(
                h5_path=str(h5_path),
                interval_type=it,
                sex=sex,
                K=K,
            )
            means = [round(float(m), 5) for m in np.asarray(mixture_model.means_).flatten()]
            sds = [
                round(float(s), 5)
                for s in np.sqrt(np.asarray(mixture_model.covariances_).reshape(-1))
            ]
            print(it, sex, means, sds)

Source: `inter_usv_interval_analyses.ipynb <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/inter_usv_interval_analyses.ipynb>`_.

USV general analyses
--------------------

**usv_general_analyses.ipynb** is the single home for the descriptive / cross-session USV analyses — the non-neural, non-modeling views of a set of recording sessions' ultrasonic vocalizations (USVs). It runs *after* the processing pipeline has produced, per session, a concatenated multi-channel ``*_int16.mmap`` audio file, a ``*_usv_summary.csv`` (per-USV start / stop / duration / category / emitter), and the 3D translated / rotated / metric tracks; the spectrogram views also read the consolidated SAM2 + spectrogram HDF5 store.

It has two parts: **Individual USV views** — render the spectrogram(s) of one session (single-channel / all-channel / stitched), driven from ``make_usv_spectrograms``; and **Cross-session summaries** — pooled acoustic-property histograms, per-session-type USV counts, and per-session timelines (also from ``make_usv_spectrograms``), followed by the full behavioral summary-statistics suite from ``usv_summary_statistics`` (assignment, participation, rate / fatigue, proximity, estrous-stage, and spatial-distribution figures).

Parameters follow a **hybrid layout**: each spectrogram figure defines its own knobs at the top of its cell (the cells are independent of one another), while the statistics suite is driven by one shared **Statistics parameters** cell plus a **Statistics setup** cell that builds the master per-USV dataframe every statistics section consumes. Paths are written ``/mnt/...`` and wrapped in ``configure_path()`` so they resolve on macOS too.

**Imports.** A single **Imports** cell loads everything the notebook needs — the spectrogram plotter and its module-level helpers from ``make_usv_spectrograms``, the ``usv_summary_statistics`` module (aliased ``uss``), the figure / colormap helpers, and the shared plot style — then calls ``apply_plot_style()``.

.. code-block:: python

    from __future__ import annotations

    import json
    from pathlib import Path
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pls

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.plot_style import apply_plot_style
    from usv_playpen.visualizations.figure_io import save_figure
    from usv_playpen.visualizations.auxiliary_plot_functions import create_colormap
    import usv_playpen.visualizations.usv_summary_statistics as uss
    from usv_playpen.visualizations.make_usv_spectrograms import (
        USVSpectrogramPlotter,
        plot_session_type_usv_counts,
        plot_session_usv_timeline,
        plot_usv_property_histograms,
    )

    apply_plot_style()

**Individual USV views.** The first part renders the spectrogram(s) of a single session. Its one figure cell carries its own parameters at the top (the per-figure half of the hybrid layout) and builds the per-sex colormaps it needs inline.

**1. Spectrogram plotter.** ``USVSpectrogramPlotter`` renders one session's USV spectrograms in the mode set by ``vis_settings['make_usv_spectrograms']['mode']``: ``"single"`` (one channel, optional stacked raw waveform, dB scale), ``"all"`` (every channel vertically stacked, dB scale), or ``"stitched"`` (a session-timeline spectrogram assembled from the pre-computed ``[0, 1]``-normalized per-USV spectrograms in the consolidated HDF5 store, placed at their on-session start times with linear normalized amplitude and a fixed ``[0, 1]`` colorbar). The cell defines its own two knobs, loads ``visualizations_settings.json`` into ``vis_settings``, builds one sequential per-sex (base-color → white) colormap, resolves the ``cmap_override`` from the string choice, and runs the plotter.

.. code-block:: python

    spectrogram_session_root = configure_path("/mnt/falkner/Bartul/Data/20230124_192908")
    spectrogram_cmap_choice = "female"

    with open(
        Path.cwd().parent / "_parameter_settings" / "visualizations_settings.json", "r"
    ) as vis_settings_file:
        vis_settings = json.load(vis_settings_file)

    male_color = vis_settings["male_colors"][0]
    female_color = vis_settings["female_colors"][0]

    # One sequential base-color -> white colormap per sex.
    sex_cmaps = {}
    for cmap_name, base_hex in (("female_cm", female_color), ("male_cm", male_color)):
        sex_cmaps[cmap_name] = create_colormap(
            input_parameter_dict={
                "cm_length": 255,
                "cm_name": cmap_name,
                "cm_type": "sequential",
                "cm_start": (
                    int(base_hex[1:3], 16),
                    int(base_hex[3:5], 16),
                    int(base_hex[5:7], 16),
                ),
                "cm_end": (255, 255, 255),
                # ...
            }
        )
    female_cmap = sex_cmaps["female_cm"]
    male_cmap = sex_cmaps["male_cm"]

    spectrogram_cmap_override = {"female": female_cmap, "male": male_cmap, None: None}[
        spectrogram_cmap_choice
    ]

    plotter = USVSpectrogramPlotter(
        root_directory=spectrogram_session_root,
        visualizations_parameter_dict=vis_settings,
        message_output=print,
        cmap_override=spectrogram_cmap_override,
    )

    fig_spectrogram = plotter.make_usv_spectrograms()
    plt.show()

* **spectrogram_session_root** — a session directory holding a ``*_int16.mmap*`` audio file (and, for stitched mode, a ``*_usv_summary.csv`` 1:1 with the consolidated h5 entries). All other spectrogram knobs (mode, channel, ``time_window``, ``freq_limits``, ``nfft``, colorbar, save) live in the ``make_usv_spectrograms`` block of ``visualizations_settings.json``; override them on ``vis_settings``.
* **spectrogram_cmap_choice** — ``'female'`` / ``'male'`` selects the matching per-sex colormap; ``None`` falls back to ``vis_settings['make_usv_spectrograms']['spectrogram_cmap']``.

**Cross-session summaries.** The second part pools many sessions. Its three ``make_usv_spectrograms`` helper figures each carry their own parameters (still the per-figure half of the hybrid layout); all three share the same noise filter, ``noise_col_id = 'vae_supercategory'`` with ``noise_categories = (0,)``, which drops noise rows before plotting.

**2. Property histograms.** ``plot_usv_property_histograms`` is a module-level helper (not a method on ``USVSpectrogramPlotter``) that pools per-USV properties across many sessions into a single five-panel figure: ``duration`` (ms), ``mean_amplitude`` (a.u.), ``mean_freq_hz`` (kHz), ``freq_bandwidth_hz`` (kHz), ``spectral_entropy`` (nats). Each panel uses 36 linearly-spaced bins over the FeatureZoo theoretical range; the title reports the number of sessions loaded and the total pooled vocalizations.

.. code-block:: python

    noise_col_id = "vae_supercategory"
    noise_categories = (0,)

    histograms_sessions_txt_path = configure_path(
        "/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt"
    )
    histograms_output_path = None
    histograms_fig_format = "svg"

    fig_histograms = plot_usv_property_histograms(
        sessions_txt_path=histograms_sessions_txt_path,
        output_path=histograms_output_path,
        fig_format=histograms_fig_format,
        noise_col_id=noise_col_id,
        noise_categories=noise_categories,
    )
    plt.show()

* **histograms_sessions_txt_path** — a txt file listing the session roots to pool.
* **histograms_output_path** — where to save the figure; ``None`` shows it without saving.
* **histograms_fig_format** — saved-figure format (e.g. ``'svg'``).

**3. Session-type counts.** ``plot_session_type_usv_counts`` takes three session-list txt files (male-female, female-female, lone-male) and renders a horizontal bar chart comparing the mean number of non-noise USVs per session, with standard error of the mean (SEM) error bars (``std(counts, ddof=1) / sqrt(n_sessions)``). For each list it discovers ``*_usv_summary.csv``, drops noise rows, and counts the rest; sessions whose CSV is missing or unreadable are logged and excluded from that group's mean.

.. code-block:: python

    noise_col_id = "vae_supercategory"
    noise_categories = (0,)

    male_female_txt_path = configure_path(
        "/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt"
    )
    female_female_txt_path = configure_path(
        "/mnt/falkner/Bartul/modeling/input_files/behavioral_female_female_sessions_list.txt"
    )
    lone_male_txt_path = configure_path(
        "/mnt/falkner/Bartul/modeling/input_files/ephys_lone_male_sessions_list.txt"
    )
    session_counts_output_path = configure_path(
        "/Users/bmimica/Downloads/session_type_usv_counts"
    )
    session_counts_fig_format = "svg"

    fig_session_counts = plot_session_type_usv_counts(
        male_female_txt_path=male_female_txt_path,
        female_female_txt_path=female_female_txt_path,
        lone_male_txt_path=lone_male_txt_path,
        output_path=session_counts_output_path,
        fig_format=session_counts_fig_format,
        noise_col_id=noise_col_id,
        noise_categories=noise_categories,
    )
    plt.show()

* **male_female_txt_path** / **female_female_txt_path** / **lone_male_txt_path** — the three per-session-type session lists (each a bar in the chart).
* **session_counts_output_path** / **session_counts_fig_format** — save location and format for the figure.

**4. Session timeline.** ``plot_session_usv_timeline`` draws every non-noise USV in one session as a colored rectangle spanning its ``[start, stop]`` interval on a single horizontal strip. The session's male / female track ids are read from ``<session>/video/*_points3d_translated_rotated_metric.h5`` (``track_names[0]`` = male, ``track_names[1]`` = female); each USV's CSV ``emitter`` field is matched against them — male → ``#9AC0CD``, female → ``#FF6347``, anything else → ``#C0C0C0`` (unassigned). The title reports the total non-noise count.

.. code-block:: python

    noise_col_id = "vae_supercategory"
    noise_categories = (0,)

    timeline_session_root = configure_path("/mnt/falkner/Bartul/Data/20230119_172410")
    timeline_window = (457, 462)  # (start_s, end_s); None shows the whole session
    timeline_output_path = configure_path(
        "/Users/bmimica/Downloads/session_usv_timeline_20230119_172410_457s-462s"
    )
    timeline_fig_format = "svg"

    fig_timeline = plot_session_usv_timeline(
        session_root=timeline_session_root,
        time_window=timeline_window,
        output_path=timeline_output_path,
        fig_format=timeline_fig_format,
        noise_col_id=noise_col_id,
        noise_categories=noise_categories,
    )
    plt.show()

* **timeline_session_root** — the single session to draw.
* **timeline_window** — ``(start_s, end_s)`` clip in seconds; ``None`` shows the whole session.
* **timeline_output_path** / **timeline_fig_format** — save location and format for the figure.

**Statistics parameters.** The statistics half of the notebook is driven by one shared **Statistics parameters** cell — every knob a user might tweak lives here (data source, segmentation model, noise / category columns, feature suffixes, output toggle, and all per-figure styling / thresholds). Nothing downstream redefines these.

.. code-block:: python

    # Data source
    sessions_list_path = "/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt"

    # Segmentation model: 'vae' or 'qlvm' (drives the per-USV category basis and the
    # embedding coordinates, both derived in Setup). Noise is a VAE-only label, so it is
    # ALWAYS filtered on vae_supercategory == 0 regardless of this choice.
    embedding_model = "vae"
    noise_col_id = "vae_supercategory"
    noise_categories = [0]

    # Behavioral-feature column suffixes
    distance_suffix = "nose-nose"
    mf_angle_suffix = "allo_yaw-nose"
    fm_angle_suffix = "nose-allo_yaw"

    # Output
    save_fig_bool = False

    # Shared styling
    line_color = "#202020"
    colormap = "magma"  # category fatigue heatmaps
    smoothing_sigma = 1.0  # category fatigue heatmaps

    # Category-embedding panel / duration histograms / local-fatigue bins / ANOVA
    embedding_boundary_color = "#000000"
    embedding_log_scale_bars = False
    embedding_plot_type = "density"
    embedding_grid_res = 400
    duration_bin_width_ms = 20.0
    duration_max_ms = 300.0
    bin_width_seconds = 120
    max_time_seconds = 1200
    fatigue_facet_figsize = (12, 10)
    min_samples_anova = 30

    # Estrous
    estrous_code_index = -1
    valid_stages = ["p", "e", "m", "d"]
    label_map = {"p": "Proestrus", "e": "Estrus", "m": "Metestrus", "d": "Diestrus"}
    category_order = ["p", "e", "m", "d"]
    estrous_colors = ["#810000", "#ff1714", "#ff5555", "#ffaaaa"]
    estrous_confidence_level = 0.99
    estrous_use_log_scale = True

    # Spatial / polar-KDE
    max_plot_distance = 20.0
    occupancy_thresh = 0.001
    kde_max_points = 50000
    polar_grid_threshold_male = 100
    polar_grid_threshold_female = 50
    estrous_kde_min_points = 30

    # Jointplot colors
    jointplot_scatter_color = "#808080"
    jointplot_line_color = "#FF0000"
    jointplot_hist_color = "#A0A0A0"

* **sessions_list_path** — the ``.txt`` file listing one session root per line; its name also derives ``session_type``, the prefix on every saved figure.
* **embedding_model** — ``"vae"`` or ``"qlvm"``; picks the per-USV category basis (``usv_category_col``) and the embedding coordinates (``usv_continuous_cols``), both resolved in the Setup cell.
* **noise_col_id** / **noise_categories** — the column and value that flag the noise bucket dropped during extraction (noise is a variational autoencoder (VAE)-only label).
* **distance_suffix** / **mf_angle_suffix** / **fm_angle_suffix** — which behavioral-feature columns become ``distance`` / ``mf_angle`` / ``fm_angle``.
* **save_fig_bool** — when ``True``, every cell writes its figures to disk via ``save_figure``; when ``False`` figures are only shown inline.
* The remaining knobs are per-figure styling / binning / thresholds consumed by the individual statistics sections below.

**Statistics setup.** The **Statistics setup** cell builds objects that follow from the parameters plus ``visualizations_settings.json`` — the ``session_type`` label, the fatigue bin count ``n_bins``, the category basis / embedding coordinates for the chosen ``embedding_model``, the per-sex colors, and one sequential per-sex colormap. You should not need to edit it.

.. code-block:: python

    session_type = re.sub(
        r"courtship_behavioral_|_list.txt", "", Path(sessions_list_path).name
    )
    n_bins = max_time_seconds // bin_width_seconds

    usv_category_col = f"{embedding_model}_supercategory"
    usv_continuous_cols = (f"{embedding_model}_umap1", f"{embedding_model}_umap2")

    with open(
        Path.cwd().parent / "_parameter_settings" / "visualizations_settings.json", "r"
    ) as vis_settings_file:
        vis_settings = json.load(vis_settings_file)

    male_color = vis_settings["male_colors"][0]
    female_color = vis_settings["female_colors"][0]
    unassigned_color = vis_settings["unassigned_colors"][0]

    # One sequential base-color -> white colormap per sex.
    sex_cmaps = {}
    for cmap_name, base_hex in (("female_cm", female_color), ("male_cm", male_color)):
        sex_cmaps[cmap_name] = create_colormap(
            input_parameter_dict={
                "cm_length": 255,
                "cm_name": cmap_name,
                "cm_type": "sequential",
                "cm_start": (
                    int(base_hex[1:3], 16),
                    int(base_hex[3:5], 16),
                    int(base_hex[5:7], 16),
                ),
                "cm_end": (255, 255, 255),
                # ...
            }
        )
    female_cmap = sex_cmaps["female_cm"]
    male_cmap = sex_cmaps["male_cm"]

**Load the session.** The **Extract data** cell fans the sessions-list file out into per-session roots and folds them into one master per-USV frame (``usv_pls``) plus a background-frames frame (``bg_pls``), filtering the noise bucket in the process. It also derives the pandas view ``usv_df`` with a ``duration_ms`` column. Every statistics cell downstream consumes these in-memory objects.

.. code-block:: python

    txt_sessions_file = Path(configure_path(sessions_list_path))
    with txt_sessions_file.open("r") as sessions_txt_file:
        session_roots = [
            configure_path(line.strip()) for line in sessions_txt_file if line.strip()
        ]

    usv_pls, bg_pls, noise_filtered_count = uss.build_master_usv_dataframe(
        session_roots=session_roots,
        noise_col_id=noise_col_id,
        noise_categories=noise_categories,
        usv_category_col=usv_category_col,
        distance_suffix=distance_suffix,
        mf_angle_suffix=mf_angle_suffix,
        fm_angle_suffix=fm_angle_suffix,
    )

    usv_df = usv_pls.to_pandas()
    usv_df["duration_ms"] = usv_df["duration"] * 1000

**Assignment & participation (§5–7).** Three cells summarise *who* vocalized. **§5 Per-session assignment summary** draws per-session stacked bars (raw counts and proportions) plus a global assignment summary panel of USVs attributed to male / female / unassigned. **§6 Assignment status by USV category** overlays category prevalence on the embedding, split by assignment. **§7 Per-mouse participation** reports per-animal participation (sessions and total USVs) for males and females separately. All three share the per-sex colors resolved in the Setup cell.

.. code-block:: python

    assignment_df = (
        usv_pls.group_by(["session_id", "sex"])
        .agg(pls.len().alias("count"))
        .pivot(values="count", index="session_id", on="sex")
        .fill_null(0)
    )
    for col in ["male", "female", "unassigned"]:
        if col not in assignment_df.columns:
            assignment_df = assignment_df.with_columns(pls.lit(0).alias(col))
    assignment_df = assignment_df.rename({"session_id": "session"})

    fig_bars_raw, ax_bars_raw, stats_bars_raw = uss.plot_assignment_stacked_bars(
        assignment_df=assignment_df,
        plot_proportions=False,
        male_color=male_color,
        female_color=female_color,
        unassigned_color=unassigned_color,
    )
    if save_fig_bool:
        save_figure(fig_bars_raw, f"{session_type}_usv_assignment_raw_counts", vis_settings)
    plt.show()
    # ... proportions bars, global summary panel (plot_assignment_summary_panel)

The §6 category-embedding panel reads its own knobs from the **Statistics parameters** cell (``embedding_boundary_color``, ``embedding_log_scale_bars``, ``embedding_plot_type``, ``embedding_grid_res``) and first extracts the embedding via ``uss.extract_category_embedding_data`` before calling ``uss.plot_category_prevalence_and_embedding``. §7 builds per-animal ``session_count`` / ``total_usvs`` dicts for each sex and passes them to ``uss.plot_animal_participation_stats``.

* **embedding_boundary_color** / **embedding_log_scale_bars** — outline color and log/linear scaling of the prevalence bars.
* **embedding_plot_type** — ``"density"`` vs. scatter rendering of the embedding.
* **embedding_grid_res** — resolution of the density grid.

**Rate & fatigue (§8–11).** Four cells characterise how vocalization rate and duration evolve over a session. **§8 Global vocalization rate over time** draws duration histograms by sex and hourly regressions of USV duration and count; **§9 Global rate by category** is a category-resolved global-fatigue heatmap; **§10 Local fatigue around emission events** bins USV counts into a fixed window around session onset and plots the mean trend; **§11 Local fatigue by category** breaks the same signal down by call category. All four read their binning / smoothing / colormap knobs from the **Statistics parameters** cell.

.. code-block:: python

    fig_hist, axes_hist, stats_hist = uss.plot_duration_histograms_by_sex(
        plot_data=usv_df,
        bin_width_ms=duration_bin_width_ms,
        max_duration_ms=duration_max_ms,
        male_color=male_color,
        female_color=female_color,
    )
    if save_fig_bool:
        save_figure(fig_hist, f"{session_type}_global_duration_histograms", vis_settings)
    plt.show()
    # ... hourly regressions (uss.plot_hourly_regressions), then the global and
    # local category fatigue heatmaps (uss.plot_category_global_fatigue_heatmap,
    # uss.plot_local_fatigue_binned_trends, uss.plot_category_local_fatigue_heatmap)

* **duration_bin_width_ms** / **duration_max_ms** — bin width and right edge of the duration histograms.
* **bin_width_seconds** / **max_time_seconds** — time-bin width and horizon for the local-fatigue trends (``n_bins`` is derived in Setup).
* **fatigue_facet_figsize** — figure size for the category-faceted local-fatigue heatmap.
* **colormap** / **smoothing_sigma** — colormap and Gaussian smoothing applied to both category fatigue heatmaps.

**Proximity & duration (§12–14).** Three cells relate USV assignment and duration to spatial behavior. **§12 Unassigned-rate vs. proximity (session-level)** aggregates per session (median distance vs. unassigned proportion, as a jointplot); **§13 Unassigned-rate vs. proximity (per-USV level)** asks the same per call (distance-by-assignment KDEs with an analysis of variance, ANOVA); **§14 USV duration vs. spatial behavior** is a regression grid of USV duration on each spatial / postural feature, computed separately for males and females.

.. code-block:: python

    df_anova = (
        usv_pls.select(["sex", "distance"])
        .drop_nulls()
        .rename({"sex": "category"})
        .to_pandas()
    )

    fig_anova, ax_anova, stats_anova = uss.plot_distance_by_assignment_kde_anova(
        df_plot=df_anova,
        min_samples_anova=min_samples_anova,
        male_color=male_color,
        female_color=female_color,
        unassigned_color=unassigned_color,
    )
    if save_fig_bool:
        save_figure(
            fig_anova, f"{session_type}_distance_by_assignment_kde_anova", vis_settings
        )
    plt.show()
    # ... session-level jointplot (uss.plot_unassigned_proportion_vs_distance_jointplot)
    # and the duration regression grid (uss.plot_behavior_duration_regressions)

* **min_samples_anova** — minimum per-group sample size before a group enters the distance ANOVA.
* **jointplot_scatter_color** / **jointplot_line_color** / **jointplot_hist_color** — scatter, fit-line, and marginal-histogram colors of the session-level proximity jointplot.

**Estrous stage (§15–16).** Two cells aggregate USV metrics by the female's estrous stage (decoded from ``experiment_code``). **§15 Estrous-stage USV metrics** attaches the stage, then draws a stage-distribution pie, a per-stage USV-rates bar, and a male-to-female ratio scatter; **§16 Estrous metrics by category** is the category-resolved version — facet grids of per-category rates and of M:F ratios broken down both by category and by stage. Both read the estrous knobs from the **Statistics parameters** cell.

.. code-block:: python

    usv_pls = usv_pls.with_columns(
        pls.col("experiment_code").str.slice(estrous_code_index).alias("estrous_stage")
    )
    estrous_subset = usv_pls.filter(pls.col("estrous_stage").is_in(valid_stages))
    # ... build session_counts / male_usv_counts / female_usv_counts / estrous_data

    fig_rates, axes_rates, stats_rates = uss.plot_estrous_usv_rates(
        session_counts=session_counts,
        male_usv_counts=male_usv_counts,
        female_usv_counts=female_usv_counts,
        category_order=category_order,
        category_labels=category_labels,
        male_color=male_color,
        female_color=female_color,
        text_color=line_color,
    )
    if save_fig_bool:
        save_figure(fig_rates, f"{session_type}_estrous_usv_rates_bar", vis_settings)
    plt.show()
    # ... pie chart (uss.plot_estrous_stage_pie_chart), ratio scatter
    # (uss.plot_estrous_ratio_scatter) and the category grids
    # (uss.plot_category_estrous_rates_grid, uss.plot_category_estrous_ratio_grid)

* **estrous_code_index** — slice offset into ``experiment_code`` that yields the one-letter stage.
* **valid_stages** / **label_map** / **category_order** — the recognised stage codes, their display names, and their plotting order.
* **estrous_colors** — the four per-stage slice / scatter colors.
* **estrous_confidence_level** / **estrous_use_log_scale** — confidence interval and log scaling for the M:F ratio scatter.

**Spatial distributions (§17–19).** Three cells map *where* USVs occur relative to each animal's body axis, normalised by background occupancy. **§17 Spatial vocalization distributions** draws one polar KDE of distance vs. angle per sex; **§18 Spatial likelihood grid by category** is a per-category likelihood grid for males and females; **§19 Spatial likelihood, category × estrous-stage** crosses USV category with estrous stage. All three read the spatial knobs from the **Statistics parameters** cell and use the per-sex colormaps built in Setup.

.. code-block:: python

    male_usv = (
        usv_pls.filter(pls.col("sex") == "male")
        .select(["distance", "mf_angle"])
        .drop_nulls()
    )
    female_usv = (
        usv_pls.filter(pls.col("sex") == "female")
        .select(["distance", "fm_angle"])
        .drop_nulls()
    )
    all_dist = bg_pls["distance"].drop_nulls().to_numpy()
    all_mf_angle = bg_pls["mf_angle"].drop_nulls().to_numpy()

    fig_polar_m, axes_polar_m, stats_polar_m = uss.plot_polar_kde_distance_angle(
        usv_distances=male_usv["distance"].to_numpy(),
        usv_angles_deg=male_usv["mf_angle"].to_numpy(),
        all_distances=all_dist,
        all_angles_deg=all_mf_angle,
        max_distance=max_plot_distance,
        colormap=female_cmap,
        ylabel="nose-nose distance (cm)",
        occupancy_threshold=occupancy_thresh,
        max_kde_points=kde_max_points,
    )
    if save_fig_bool:
        save_figure(
            fig_polar_m, f"{session_type}_male_polar_kde_spatial_distribution", vis_settings
        )
    plt.show()
    # ... female polar KDE, the per-category likelihood grids
    # (uss.plot_category_polar_kde_grid) and the category X estrous-stage grids
    # (uss.plot_estrous_category_kde_grid)

* **max_plot_distance** — outer radial limit (cm) of every polar plot.
* **occupancy_thresh** — minimum background occupancy below which a cell is masked out.
* **kde_max_points** — cap on points fed to the KDE (subsampled above it).
* **polar_grid_threshold_male** / **polar_grid_threshold_female** — minimum per-category USV count to render a panel in the sex likelihood grids.
* **estrous_kde_min_points** — minimum count to render a category × estrous-stage panel.

Source: `usv_general_analyses.ipynb <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_general_analyses.ipynb>`_.

USV embedding explorer
----------------------

**usv_embedding_explorer.py** is an interactive `marimo <https://marimo.io>`_ app — not a
Jupyter notebook, so it is not rendered below. It pools every selected session's
``*_usv_summary.csv`` into a single scatter of the chosen embedding map, lets you brush a
region, and renders a grid of example spectrograms sampled from inside it. Hovering a point
reveals that USV's identity and acoustics.

**Embedding maps.**

* **VAE UMAP** — a 2-D UMAP of the variational-autoencoder acoustic latents.
* **QLVM torus** — the toroidal (doughnut-shaped) surface of the in-house QLVM
  (quasi-Monte Carlo latent variable model).

**Controls** (stacked above the plot):

* **Session lists** — every ``*.txt`` list in the configured input-files directory (playback
  lists excluded). The union of the picked lists is pooled — and parquet-cached under
  ``~/.usv_playpen_cache`` — only when **Load** is clicked.
* **Sessions** — narrows the loaded pool to individual sessions. Empty (the default) shows
  every session; pick one or more to isolate them.
* **Map** — VAE UMAP or QLVM torus.
* **Color by** — a categorical label (fine / coarse category, session type, session id, or
  emitter sex) or a continuous metric (point density, or a per-USV acoustic feature), the
  latter rendered through the project colormap.
* **Boundaries** — optional k-NN cluster outlines for the chosen categorical label.
* **Examples (spectrograms) plotted** — 5–50, sampled along an Archimedean spiral (centre →
  edge) and laid out as a square grid, each call's width preserving its true duration.
* **Max points** — caps how many points the scatter draws, keeping the chart under marimo's
  ``output_max_bytes``.
* **Apply mask** — multiplies each sampled spectrogram by its SAM2 segmentation mask, so only
  the segmented call shows.

**Hover tooltip.** Hovering a point shows its ``session id``, ``emitter`` (the animal id, or
``unassigned``), ``mean amplitude``, ``mean frequency`` (kHz), and ``spectral entropy``.

**Paths.** The session-list directory and the consolidated spectrogram / SAM2 store come from
the ``shared_resources`` block of ``visualizations_settings.json`` (``input_files_directory`` /
``spectrograms_dir``); the shipped ``Bartul`` paths are re-keyed to the experimenter in use and
resolved per host by ``resolve_experimenter_path``.

Launch it from the repo root in either of two modes:

.. code-block:: bash

    # editable, reactive code view (for tweaking the notebook)
    uv run marimo edit src/usv_playpen/notebooks/usv_embedding_explorer.py

    # clean app view (just the controls + plot, no code)
    uv run marimo run src/usv_playpen/notebooks/usv_embedding_explorer.py

    # run as a specific experimenter (else the host config's experimenter is used) --
    # sets both the spectrogram .h5 path and the session-list picker; fixed for the
    # session, so restart marimo to change it
    EXPERIMENTER_ID=Bartul uv run marimo run src/usv_playpen/notebooks/usv_embedding_explorer.py

Both open in the browser at ``http://localhost:2718``. Source:
`usv_embedding_explorer.py
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_embedding_explorer.py>`_.
