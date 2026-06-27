Notebooks
=========

The repository ships a set of Jupyter notebooks (plus one `marimo <https://marimo.io>`_
app) under ``src/usv_playpen/notebooks/`` that drive the advanced analysis and
figure-generation workflows. This page is the single, detailed home for them: each
notebook is described here and rendered in full below (code only — outputs are stripped
on commit), and the topical sections (:doc:`Analyze`, :doc:`Modeling`, :doc:`Visualize`,
:doc:`Histology`) link here rather than duplicating the explanation.

.. note::

   The rendered notebooks show **code only** — cell outputs are stripped from the
   committed notebooks (via ``nbstripout``) and nbsphinx does not execute them
   (``nbsphinx_execute = "never"``). Run a notebook locally to see its figures.

Every notebook collects all paths, toggles, and thresholds in a single **Parameters**
cell near the top (nothing downstream redefines them), so a run is configured in one
place; paths are written ``/mnt/falkner/...`` and ``configure_path()``-normalised to the
host OS. Each plotting cell is independent — re-run any single one once the imports /
parameters / setup cells have run.

.. _modeling-notebook:

Modeling
--------
**modeling_analyses.ipynb** — the end-to-end interactive entry point for the
vocal-modeling pipelines in ``usv_playpen.modeling``. Run the sections in order:
earlier sections produce the artifacts that later ones consume.

* **Extract modeling input data** — convert per-session loader output into the
  per-pipeline modeling-input pickle and run the predictor-diagnostics audits
  (``_collinearity.pkl`` + ``_timescales.pkl``).
* **Predictor diagnostics plots** — visualise the audit artifacts before committing to
  fitting.
* **Model selection** — univariate GAM/linear ranking, then forward-stepwise feature
  selection, run locally or dispatched to SLURM at cohort scale.
* **Univariate / multinomial / continuous-manifold visualisations** — render the fitted
  predictors for the categorical and continuous targets.
* **CNN manifold-position pipeline** — the 1-D ResNet baseline for the continuous
  UMAP-manifold target, plus its ``DeepResultsVisualizer`` diagnostics (permutation test,
  feature importance, spatial-precision grid, error landscape, regional saliency).

Every stochastic step is seeded from ``model_params.random_seed`` for reproducibility, and
every pipeline reads ``_parameter_settings/modeling_settings.json`` (override by passing an
explicit ``modeling_settings_dict``). See :doc:`Modeling` for the conceptual workflow.
Source: `modeling_analyses.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/modeling_analyses.ipynb>`_.

Neural
------
**neuronal_tuning_summary.ipynb** — has two independent halves.

* **Cross-session unit-triage aggregator** — drives ``aggregate_units_across_conditions``:
  for each condition (one ``.txt`` session list per condition) it loads every session's
  ``*_tuning_curves_data.pkl``, re-applies the significance rules to the pre-computed
  ``triage_stats`` block, joins each cluster with ``unit_catalog.csv`` for ``mouse_id`` /
  ``rec_date`` / ``brain_area``, collapses same-day duplicate units into one record with
  per-session evidence stacked underneath each modality, and pickles a unit-keyed roll-up
  ``<out_dir>/unit_triage_<YYYYMMDD>_<HHMMSS>.pkl``. It never re-loads spike or USV data —
  a pure pkl-to-pickle pass — so thresholds can be swept without re-running compute. Edit
  ``CONDITION_TO_SESSION_LIST`` to point at the lists and optionally ``THRESHOLDS``
  (mirrors the ``detect_interesting_tuning_neurons`` block of ``analyses_settings.json``:
  ``z_threshold``, ``min_consecutive_bins``, ``vmi_alpha``, ``vmi_min_bouts``,
  ``spatial_info_bps_threshold``).
* **Anatomy / dataset-overview figures** — renders per-session SVG/PNG anatomy panels
  (recording yield by mouse and cell type, per-probe unit waveforms with the four-shank
  schematic, a 360° rotating brain video of every SU-somatic unit's 3D position coloured by
  brain-area bucket), read straight from ``unit_catalog.csv`` and the per-session pkls. The
  figure half honours two **Parameters** knobs — ``UNIT_KSLABELS`` (Kilosort labels to
  include, e.g. ``("good",)`` or ``("good", "mua")``) and ``UNIT_SOMATIC_FILTER``
  (``"somatic"`` / ``"non_somatic"`` / ``"both"``), both defaulting to the historical
  good + somatic scope. Because the aggregator pickle holds *every* unit, changing these
  re-filters figures with no pickle rebuild, and every caption reflects the active filter.

See :doc:`Analyze` for the compute step that produces the per-cluster pkls. Source:
`neuronal_tuning_summary.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/neuronal_tuning_summary.ipynb>`_.

**neuronal_coactivity_analyses.ipynb** — quantifies how coordinated a PAG (or other
region) population is during one USV class versus another, using the pairwise spike-count
correlation, population-vector cosine similarity, and population-vector Pearson
correlation from ``usv_playpen.analyses.neuronal_coactivity_engine``. The workflow is a
pooled trial-count bootstrap to a matched N, a chained circular-shuffle null per group,
and a direct group-A-vs-group-B label-permutation test — reported as summary tables,
per-metric null-distribution plots, a per-session breakdown, and a cross-animal slope
plot. It is **not** in the GUI and reads **nothing** from ``analyses_settings.json``:
the **Parameters** cell holds the segmentation column and the two category-id groups, the
three-criteria unit filter (``cluster_group`` + ``somatic`` + ``brain_area``, looked up
per unit in ``unit_catalog.csv``), the animal-to-sessions map, the coactivity
hyperparameters (window, bootstrap N, shuffle / permutation counts), and per-group plot
colors. The loader picks, per animal, the single recording day with the largest
filtered-unit pool so the analysed population is fixed across the day's sessions. Every
stochastic routine accepts an optional ``seed`` for reproducible nulls. See :doc:`Analyze`
for context. Source: `neuronal_coactivity_analyses.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/neuronal_coactivity_analyses.ipynb>`_.

**npx_histology_unit_quality_processing.ipynb** — end-to-end histology /
Neuropixels-alignment workflow for one session, covering two phases that bracket the
manual brainreg + napari steps run outside the notebook:

* **Light-sheet volume assembly** — combine raw light-sheet microscopy acquisitions into
  single BigTIFF volumes that brainreg / napari ingest (LaVision UltraMicroscope and
  LifeCanvas SmartSPIM modalities supported).
* **IBL ephys-alignment export** — bridge Kilosort output + brainreg track tracing into
  the IBL ephys-alignment GUI, then post-process the GUI's per-shank channel-location
  JSONs into a single SpikeInterface-ready file (replacing the slow upstream
  ``atlaselectrophysiology.extract_files.extract_data`` call), plus Neuropixels
  spike-quality metrics.

Every acquisition path and session identifier lives in the **Parameters** cell. See
:doc:`Histology` for the conceptual workflow and the underlying helpers. Source:
`npx_histology_unit_quality_processing.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/npx_histology_unit_quality_processing.ipynb>`_.

USV
---
**usv_spectrogram_analyses.ipynb** — renders USV spectrograms and the embedding / summary
figures derived from them, all driven from
``usv_playpen.visualizations.make_usv_spectrograms``. Runs **after** processing has
produced, per session, a concatenated multi-channel ``*_int16.mmap`` audio file, a
``*_usv_summary.csv``, the 3D translated / rotated / metric tracks, and — for the stitched
figures — the consolidated SAM2 + spectrogram HDF5 store. The per-session / pooled helpers
it imports are:

* ``USVSpectrogramPlotter`` — single-channel, all-channel and stitched session-timeline
  spectrograms read from a session's concatenated ``*_int16.mmap`` audio. Single / all
  modes show a dB amplitude scale over a user-defined ``time_window``; the stitched mode
  places the pre-computed ``[0, 1]``-normalized per-USV spectrograms from the consolidated
  HDF5 store at their true on-session times.
* ``plot_usv_property_histograms`` — five pooled per-USV property histograms (duration,
  mean amplitude, mean frequency, frequency bandwidth, spectral entropy) across every
  session in a text file.
* ``plot_session_type_usv_counts`` — mean USVs per session across the male-female,
  female-female and lone-male session types, with SEM error bars.
* ``plot_session_usv_timeline`` — every non-noise USV in one session drawn as a colored
  interval keyed to the male / female / unassigned emitter.

The spectrogram-plotter rendering knobs live in the ``make_usv_spectrograms`` block of
``visualizations_settings.json`` (mode, channel, ``time_window``, ``freq_limits``,
``nfft``, colorbar limits, save options); the helper inputs are surfaced in the
**Parameters** cell. The cohort-level ``plot_embedding_with_category_thumbnails`` figure
is GUI-driven instead — see :doc:`Visualize`. Source: `usv_spectrogram_analyses.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_spectrogram_analyses.ipynb>`_.

**usv_interval_mixture_models_plots.ipynb** — fits and visualises mixture models on the
distribution of inter-USV intervals (in seconds, log-transformed) for one or more
sessions. It reads the consolidated ``usv_interval_analysis_<...>.h5`` archive via the
``ivs.load_*_from_h5`` helpers (compute is split from plotting, so figures re-render
without refitting) and produces a master interval DataFrame, a BIC/AIC information-criterion
sweep over candidate component counts (Gaussian and Student-t mixtures), a bootstrap
likelihood-ratio test selecting the best ``K`` per family, and the fit plots: the bootstrap
LRT null-distribution panel (with broken-axis support when ``LR_obs`` falls far above the
null), the BIC and AIC sweeps with the LRT-selected K highlighted, the best-fit mixture with
per-component triangles labelled ``(a)``, ``(b)``, ... and a left-aligned legend mapping each
letter to its component median in seconds, an optional per-component pdf overlay, and a
log-log Q-Q diagnostic inset. The GUI does not expose this analysis; its parameters
(``session_lists``, ``output_directory``, ``fit_gmm``, the ``model_class`` t/gauss choice,
the component-count sweep and EM/CV knobs, ``bins_per_sex``, ``plot_log_xlims``, …) live in
``analyses_settings.json``. See :doc:`Analyze` for the archive schema and the compute step.
Source: `usv_interval_mixture_models_plots.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_interval_mixture_models_plots.ipynb>`_.

**usv_summary_statistics_plots.ipynb** — produces every figure summarising a recording
session's USVs. Runs **after** processing has produced a ``*_usv_summary.csv`` (per-USV
start / stop / duration / category / emitter) and the 3D translated / rotated / metric
tracks. The **Parameters** cell selects the segmentation model (``embedding_model``,
``'vae'`` or ``'qlvm'`` — noise is a VAE-only label, so noise is always filtered on
``vae_supercategory == 0`` regardless), the behavioural-feature column suffixes
(``nose-nose`` distance, ``allo_yaw-nose`` / ``nose-allo_yaw`` angles), the ``save_fig_bool``
output toggle, and per-figure styling/thresholds. Sex colors and per-sex colormaps are
derived from ``visualizations_settings.json``. The figures, each in an independent cell, are:

* Per-session assignment summary and assignment status by USV category — the first
  sanity check (≫50% unassigned usually flags a sound-localization problem upstream).
* Per-mouse participation; global vocalization rate over time and broken down by category.
* Local fatigue around emission events and by category (time-binned, ``bin_width_seconds``
  / ``max_time_seconds``).
* Unassigned-rate vs. inter-animal proximity at session and per-USV level; USV duration
  vs. spatial behavior (``min_samples_anova`` ANOVA gate).
* Category-embedding density panel (``embedding_plot_type`` / ``embedding_grid_res``) and
  duration histograms (``duration_bin_width_ms`` / ``duration_max_ms``).
* Estrous-stage USV metrics and the same broken down by category
  (``valid_stages`` p/e/m/d, ``estrous_confidence_level``).
* Spatial vocalization distributions (polar KDEs), spatial likelihood grid by category,
  and spatial likelihood crossed with estrous stage (``max_plot_distance``,
  ``occupancy_thresh``, ``kde_max_points``, per-sex polar grid thresholds).

Source: `usv_summary_statistics_plots.ipynb
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_summary_statistics_plots.ipynb>`_.

Interactive (marimo)
--------------------
**usv_embedding_explorer.py** — an interactive `marimo <https://marimo.io>`_ app (not a
Jupyter notebook, so it is not rendered below): it pools every selected session's
``*_usv_summary.csv`` into one scatter of the chosen embedding map (VAE UMAP or QLVM torus),
lets you brush a region, and shows a grid of example spectrograms sampled from inside it.
The controls (above the plot) cover **Session lists** (every ``*.txt`` list in the configured
input-files directory, playback lists excluded; pooled and parquet-cached under
``~/.usv_playpen_cache`` only when **Load** is clicked), **Map** (VAE UMAP / QLVM torus),
**Color by** (a categorical label or a continuous metric rendered through the project
colormap), **Boundaries** (optional kNN cluster outlines), and **Examples (spectrograms)
plotted** (5–50, sampled along an Archimedean spiral and laid out as a square grid, each
call's width preserving its true duration). The session-list directory and the consolidated
spectrogram/SAM2 store are read from the ``shared_resources`` block of
``visualizations_settings.json`` (``input_files_directory`` / ``spectrograms_dir``), resolved
per-host via ``configure_path``. See :doc:`Visualize` for launch instructions. Source:
`usv_embedding_explorer.py
<https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/notebooks/usv_embedding_explorer.py>`_.

Rendered notebooks
------------------

.. toctree::
   :maxdepth: 1

   notebooks/modeling_analyses
   notebooks/neuronal_tuning_summary
   notebooks/neuronal_coactivity_analyses
   notebooks/npx_histology_unit_quality_processing
   notebooks/usv_spectrogram_analyses
   notebooks/usv_interval_mixture_models_plots
   notebooks/usv_summary_statistics_plots
