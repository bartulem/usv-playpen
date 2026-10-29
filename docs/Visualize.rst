.. _Visualize:

Visualize
==================
This page explains how to use the data visualization functionalities in the *usv-playpen* GUI (graphical user interface):

In order to run any of the functions detailed below, select an experimenter name from the dropdown menu and click the *Visualize* button on the GUI main display:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_0a.png
   :align: center
   :alt: Visualize Step 0a

.. raw:: html

   <br>

Clicking the *Visualize* button will open a new window with all the offered functionalities (see below):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_0b.png
   :align: center
   :alt: Visualize Step 0b

.. raw:: html

   <br>

All the main functions are outlined in orange, and black fields are function-specific options tunable by the user in the GUI. It is important to note that these are not necessarily *all* the options the user can set, and the full list of options can be found under each function in the */usv-playpen/_parameter_settings/visualizations_settings.json* file. Each time the user clicks the *Next* button in the window above, *visualizations_settings.json* is modified to the newest input configuration.

The *Root directories* field enables you to list the directories containing the data you want to visualize. Each root directory should be in its **own row**; for example, three sessions should be listed as follows:

.. parsed-literal::

    /mnt/falkner/Bartul/Data/20250430_145017
    /mnt/falkner/Bartul/Data/20250430_165730
    /mnt/falkner/Bartul/Data/20250430_182145

Besides the per-function blocks documented below, *visualizations_settings.json* carries the bookkeeping keys the GUI and ``visualize_data`` read:

.. code-block:: json

    "visualize_booleans": {
        "make_neuronal_tuning_figures_bool": false,
        "make_behavioral_videos_bool": false,
        "make_usv_spectrograms_bool": false,
        "make_qlvm_torus_traversal_video_bool": false,
        "make_embedding_thumbnails_bool": false
    },
    "visualize_data": {
        "root_directories": []
    },
    "credentials_directory": "",
    "send_email": {
        "visualizations_pc_list": ["A84E Windows", "A84I Linux", "A84I Windows", "165B Audio", "165B Neural"],
        "visualizations_pc_choice": "A84I Linux",
        "experimenter": "Bartul",
        "send_message": {
            "receivers": []
        }
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`visualize_booleans <vis-visualize-booleans>`
     - all ``false``
     - five ``true`` / ``false`` switches
   * - :ref:`visualize_data.root_directories <vis-visualize-data-root-directories>`
     - ``[]``
     - list of session root directories
   * - :ref:`credentials_directory <vis-credentials-directory>`
     - ``""``
     - path to a directory
   * - :ref:`send_email <vis-send-email>`
     - see JSON above
     - PC name, experimenter, list of receivers

.. _vis-visualize-booleans:

``visualize_booleans``
   The on / off switch of every GUI-exposed visualization.

   :Options:
      * ``make_neuronal_tuning_figures_bool`` — ``true`` / ``false``
      * ``make_behavioral_videos_bool`` — ``true`` / ``false``
      * ``make_usv_spectrograms_bool`` — ``true`` / ``false``
      * ``make_qlvm_torus_traversal_video_bool`` — ``true`` / ``false``
      * ``make_embedding_thumbnails_bool`` — ``true`` / ``false``
   :Default: ``false`` for all five
   :Notes: the GUI's Yes / No toggles write them; ``Visualizer.visualize_data`` runs the first three per listed root directory and the two cohort-level ones (the torus-traversal video, the embedding thumbnails) once, after the per-session loop.

.. _vis-visualize-data-root-directories:

``visualize_data.root_directories``
   The session root directories of the *Root directories* field above (one per row).

   :Options: a list of session root directories
   :Default: ``[]``
   :Notes: the default list ``Visualizer`` runs on when no list is passed in.

.. _vis-credentials-directory:

``credentials_directory``
   The directory holding ``email_config.ini``, the credentials of the start / completion e-mails.

   :Options: path (string) to a directory
   :Default: ``""``

.. _vis-send-email:

``send_email``
   The e-mail notification block.

   :Options:
      * ``visualizations_pc_list`` / ``visualizations_pc_choice`` — the lab PC named in the "PC is busy" / "PC is available again" messages
      * ``experimenter`` — named in the messages and passed to the behavioral videos as the experimenter id
      * ``send_message.receivers`` — the recipients; with the list empty no e-mail is sent
   :Default: see the JSON above (``visualizations_pc_choice`` ``"A84I Linux"``, ``experimenter`` ``"Bartul"``, ``send_message.receivers`` ``[]``)
   :Notes: the completion e-mail lists every visualization that failed.

Plot neuronal tuning figures
----------------------------
Once the *Compute neuronal tuning curves* function from the *Analyze* section has completed, you have the ability to plot its results. Output is one combined multi-page document per cluster: a behavioral page per temporal offset and per plot-feature group (``individual.<mouse>`` and ``social``) followed by one vocal page per emitter: the bout raster + pooled peri-onset (ultrasonic vocalization) ``usv_peth`` on top, the ``usv_property_tuning`` continuous-property grid in the middle, and at the bottom the single ``usv_category_tuning`` row of the QLVM category ``qlvm_category`` (rate and occupancy maps over the category bundle's R-1 … R-k regions, ``os_utils.load_qlvm_category_bundle``, and the tuning-vs-shuffle strip), drawn on the property grid's columns so no page carries a lone figure row. The per-category PETH (peri-event time histogram, ``usv_category_peth``) stays in the pickle but is not drawn. 1D ratemaps are drawn as a line spanning the plot, colored by the per-mouse palette (or the social color for social features). The 99% CI (confidence interval) of the shuffled distribution is shown as a shaded band around the line.

To obtain this visualization, list the root directories of interest, select *Plot neuronal tuning figures* in the GUI and click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_1.png
   :align: center
   :alt: Visualize Step 1

.. raw:: html

   <br>

Running this function results in the population of the *tuning_curves* subdirectory with one combined output per cluster (PDF by default; configurable in the GUI / settings). A cluster whose pickle cannot be loaded or whose pages fail to render is reported and the remaining clusters are still drawn; the session then ends with an error naming every failed cluster, so a partial run is never reported as finished. The figure code lives in ``usv_playpen.visualizations.make_neuronal_tuning_figures``:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ├── tuning_curves
    │   │   │   ├── **imec0_cl0000_ch361_good_neuronal_tuning.pdf**
    │   │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

For non-PDF formats, each page is written to a separate file with a ``_p{N}_{label}`` suffix (e.g. ``..._p1_behavioral_beh_offset=0s_individual.<mouse>.png``, ``..._p3_vocal_male.png``). PDF emits a single multi-page file. Each emitter gets one vocal page: the bout raster and pooled peri-USV PETH on top, the 5 x 4 USV-property tuning grid in the middle, and the QLVM category tuning row (rate and occupancy watersheds over the category bundle's R-1 … R-k regions, ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY``, and the tuning-vs-shuffle strip) at the bottom, on the property grid's columns, so no page carries a lone figure row; an unreadable category bundle stops the rendering rather than drawing placeholder panels.

The rendering-side knobs live in the project-wide ``figures`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (compute-side knobs such as ``smoothing_sd`` and ``behavioral_min_occupancy_seconds`` live in *analyses_settings.json* under ``calculate_neuronal_tuning_curves`` — see the *Analyze* page):

.. code-block:: json

    "figures": {
        "save_directory": "/mnt/falkner/Bartul/figures",
        "fig_format": "png",
        "dpi": 300,
        "timestamp_in_name": true,
        "diverging_cmap": "BrBG_r",
        "sequential_cmap": "inferno",
        "seed": 0
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`save_directory <vis-figures-save-directory>`
     - ``/mnt/falkner/Bartul/figures``
     - path to a directory
   * - :ref:`fig_format <vis-figures-fig-format>`
     - ``png``
     - ``pdf``, ``png``, ``jpg``, ``svg``
   * - :ref:`dpi <vis-figures-dpi>`
     - ``300``
     - dots per inch
   * - :ref:`timestamp_in_name <vis-figures-timestamp-in-name>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`diverging_cmap <vis-figures-diverging-cmap>`
     - ``BrBG_r``
     - any Matplotlib diverging map
   * - :ref:`sequential_cmap <vis-figures-sequential-cmap>`
     - ``inferno``
     - ``viridis``, ``cividis``, ``plasma``, ``inferno``, ``magma``
   * - :ref:`seed <vis-figures-seed>`
     - ``0``
     - integer

.. _vis-figures-save-directory:

``save_directory``
   Default output directory for figures that aren't written next to their source data (e.g. cross-session anatomy plots).

   :Options: path (string) to a directory
   :Default: ``/mnt/falkner/Bartul/figures``
   :Notes: per-figure code may override this with an explicit ``out_dir`` argument; per-cluster ratemaps and other session-bound figures always stay next to the data.

.. _vis-figures-fig-format:

``fig_format``
   Default output file format.

   :Options:
      * ``pdf`` — for the per-cluster ratemap PDFs, produces a single multi-page document
      * ``png`` / ``jpg`` / ``svg`` — write one file per page
   :Default: ``png``
   :Notes: read directly by the neuronal tuning figures: a settings file missing it raises ``KeyError`` instead of drawing at a substituted value.

.. _vis-figures-dpi:

``dpi``
   Default raster resolution applied to every ``fig.savefig`` callsite that goes through ``visualizations.figure_io.save_figure``.

   :Options: a positive integer, in dots per inch
   :Default: ``300``
   :Notes: ``fig_format``, ``dpi``, ``seed`` and ``sequential_cmap`` are read directly by the neuronal tuning figures: a settings file missing any of them raises ``KeyError`` instead of drawing at a substituted value.

.. _vis-figures-timestamp-in-name:

``timestamp_in_name``
   When ``true``, ``_YYYYMMDD_HHMMSS`` is appended to figure stems by default.

   :Options:
      * ``true`` — append the timestamp
      * ``false`` — no timestamp
   :Default: ``true``
   :Notes: session-bound figures opt out of this with ``timestamp_in_name=false`` since their filenames already embed a session id or unit id.

.. _vis-figures-diverging-cmap:

``diverging_cmap``
   The white-centred diverging colormap for **signed** modeling-filter figures — the selection-trajectory difference maps and the ``e(theta).W`` torus affinity fields in the manifold filter atlas (``plot_manifold_filter_atlas``).

   :Options: any Matplotlib diverging map (``RdBu_r``, ``coolwarm``, ``bwr``, …)
   :Default: ``BrBG_r`` (teal for negative, white at zero, yellow-brown for positive; chosen so neither side can be read as the male blue or the female red)

.. _vis-figures-sequential-cmap:

``sequential_cmap``
   The sequential (non-negative) colormap used by every heatmap / ratemap / spectrogram / density callsite.

   :Options:
      * ``viridis``
      * ``cividis``
      * ``plasma``
      * ``inferno``
      * ``magma``
   :Default: ``inferno``
   :Notes:
      * Its users: the per-cluster ratemaps and neuronal-tuning densities, the ``make_behavioral_videos`` spectrogram subplot, the ``qlvm_torus_traversal_video`` spectrogram tiles, the USV-summary heatmaps, the vocal-space atlas of real USVs in ``plot_manifold_filter_atlas``, the QLVM figures of ``qlvm_figures`` (maps, fields and spectrograms) and the property colour scales of the ``qlvm_property_sweep_video`` (used from its colormap floor upward).
      * This is the colormap the GUI's colormap dropdown edits.
      * Read directly by the neuronal tuning figures: a settings file missing it raises ``KeyError`` instead of drawing at a substituted value.

.. _vis-figures-seed:

``seed``
   Random seed for figure-side stochastic operations (the embedding-thumbnail sampling, the QLVM figures' draws, folds and spirals), for reproducibility.

   :Options: an integer
   :Default: ``0``
   :Notes: read directly by the neuronal tuning figures: a settings file missing it raises ``KeyError`` instead of drawing at a substituted value.

Colour palettes
~~~~~~~~~~~~~~~
Top-level ``*_colors`` blocks in *visualizations_settings.json* hold the semantic
(data-meaning) colours shared across figures; every entry is a hex string, so a
whole figure family can be recoloured in one place. Sex / emitter colours are
lists (index ``0`` primary, ``1`` secondary) while the grouped palettes are named
maps.

.. code-block:: json

    "male_colors": ["#9AC0CD", "#8CA252"],
    "female_colors": ["#FF6347", "#B851B4"],
    "social_colors": ["#5A6470"],
    "unassigned_colors": ["#C0C0C0"],
    "session_condition_styles": {
        "courtship_intact_partners": {"color": "#CD928A", "label": "courtship intact partners"},
        "courtship_mute_female": {"color": "#CD928A80", "label": "courtship mute female"},
        "female_female": {"color": "#FF6347", "label": "female-female"},
        "courtship_male_male": {"color": "#9AC0CD", "label": "male-male"},
        "lone_male": {"color": "#2A9D8F", "label": "isolated male"}
    },
    "brain_area_colors": {"PAG": "#677470", "MRN": "#939884", "VTA": "#F5D27A",
                          "SC": "#9FB7D8", "CENT": "#D88080", "MB": "#9BBE85", "other": "#B8B8B8"},
    "cell_type_colors": ["#1A1A1A", "#7A7A7A", "#CFCFCF"],
    "coactivity_colors": {"group_a": "#DC143C", "group_b": "#1E90FF", "null": "#808080", "threshold": "#000000"},
    "component_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    "corner_colors": {"North": "#FF0000", "West": "#FFFF00", "South": "#008000", "East": "#0000FF"}

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`male_colors <vis-emitter-colors>`
     - ``["#9AC0CD", "#8CA252"]``
     - list of hex colours (primary, secondary)
   * - :ref:`female_colors <vis-emitter-colors>`
     - ``["#FF6347", "#B851B4"]``
     - list of hex colours (primary, secondary)
   * - :ref:`unassigned_colors <vis-emitter-colors>`
     - ``["#C0C0C0"]``
     - list of hex colours
   * - :ref:`session_condition_styles <vis-session-condition-styles>`
     - see JSON above
     - ``color`` + ``label`` per session condition
   * - :ref:`social_colors <vis-social-colors>`
     - ``["#5A6470"]``
     - one hex colour
   * - :ref:`brain_area_colors <vis-brain-area-colors>`
     - see JSON above
     - hex colour per brain-region bucket
   * - :ref:`cell_type_colors <vis-cell-type-colors>`
     - ``["#1A1A1A", "#7A7A7A", "#CFCFCF"]``
     - three hex colours
   * - :ref:`coactivity_colors <vis-coactivity-colors>`
     - see JSON above
     - hex colour per coactivity role
   * - :ref:`component_colors <vis-component-colors>`
     - see JSON above
     - list of hex colours (a cycle)
   * - :ref:`corner_colors <vis-corner-colors>`
     - see JSON above
     - hex colour per arena corner

.. _vis-emitter-colors:

``male_colors`` / ``female_colors`` / ``unassigned_colors``
   Per-emitter USV colours (male / female / no-emitter-assigned), used by the USV-timeline, spectrogram-overlay, embedding-scatter and modeling figures.

   :Options: a list of hex strings, index ``0`` primary, ``1`` secondary
   :Default: ``["#9AC0CD", "#8CA252"]`` (male), ``["#FF6347", "#B851B4"]`` (female) and ``["#C0C0C0"]`` (unassigned)
   :Notes: an emitter's sex is always read from the ``Subjects`` block of the session's ``*_metadata.yaml`` (the ``subject_id`` matching the emitter / track name, ``yaml_utils.extract_animal_sexes``), never from the slot its track occupies, so same-sex sessions are coloured correctly; a tracked animal the metadata cannot resolve raises.

.. _vis-session-condition-styles:

``session_condition_styles``
   Per-condition ``color`` and ``label`` of the session types (courtship intact partners, courtship mute female, female-female, male-male, isolated male), used by the per-session squeak-timing heatmap (``plot_session_squeak_time_heatmap``).

   :Options: per condition, a ``color`` (hex string) and a ``label`` (string); colours may carry an alpha channel (8-digit hex)
   :Default: see the JSON above:

      * female-female takes ``female_colors[0]``
      * male-male takes ``male_colors[0]``
      * courtship intact partners takes the channel-wise mean of the two (``#CD928A``)
      * courtship mute female takes the same at 50 % opacity
      * isolated male is ``#2A9D8F``

.. _vis-social-colors:

``social_colors``
   The single colour for social / dyadic features (e.g. the social-feature ratemaps and the timescale-audit "social" trace).

   :Options: a one-entry list holding a hex string
   :Default: ``["#5A6470"]``

.. _vis-brain-area-colors:

``brain_area_colors``
   Per-brain-region palette shared by the behavioral-video overlays and the anatomy / tuning figures.

   :Options: a seven-bucket map: ``PAG`` / ``MRN`` / ``VTA`` / ``SC`` / ``CENT`` / ``MB`` / ``other``, each a hex string
   :Default: see the JSON above

.. _vis-cell-type-colors:

``cell_type_colors``
   The mono-grey triad for the per-mouse cell-type stacked bars in the anatomy figures.

   :Options: a list of three hex strings
   :Default: ``["#1A1A1A", "#7A7A7A", "#CFCFCF"]``

.. _vis-coactivity-colors:

``coactivity_colors``
   The ``group_a`` / ``group_b`` / ``null`` / ``threshold`` colours for the neuronal-coactivity figures.

   :Options: a map of ``group_a`` / ``group_b`` / ``null`` / ``threshold``, each a hex string
   :Default: see the JSON above
   :Notes: also the defaults the coactivity notebook may override inline.

.. _vis-component-colors:

``component_colors``
   The per-mixture-component colour cycle for the inter-USV-interval mixture-model figures.

   :Options: a list of hex strings
   :Default: see the JSON above (eight colours)
   :Notes: paired in code with distinct line styles.

.. _vis-corner-colors:

``corner_colors``
   The arena-corner marker colours for the behavioral-video arena overlay.

   :Options: a map of ``North`` / ``West`` / ``South`` / ``East``, each a hex string
   :Default: see the JSON above

These blocks hold the analysis choices, sizes, colours and statistical levels of particular figures (structural grid geometry and line-style cycles stay in the code):

.. code-block:: json

    "neuronal_tuning_figures": {
      "bout_session_bin_seconds": 1.0,
      "bout_zoom_seconds": 20.0,
      "bout_zoom_bin_seconds": 0.05,
      "bout_smooth_bins": 2.0,
      "bout_trigger_window_seconds": [-1.0, 2.0],
      "bout_trigger_bin_seconds": 0.02,
      "bout_trigger_smooth_bins": 2.5,
      "bout_density_grid": 160,
      "bout_density_levels": [0.5, 0.75, 0.9],
      "ink_color": "#1F1F1F",
      "band_color": "#D3D3D3",
      "dash_color": "#888888",
      "light_color": "#CCCCCC",
      "hatch_color": "#BBBBBB",
      "vocal_page_figsize_inches": [16.0, 27.0],
      "behavioral_page_width_inches": 16.0,
      "behavioral_row_height_inches": 2.6,
      "region_grid_figsize_inches": [14.0, 6.4],
      "sign_flip_figsize_inches": [10.0, 5.4],
      "pag_gradient_figsize_inches": [18.0, 4.8],
      "peth_timing_figsize_inches": [18.0, 6.6],
      "behavioral_tier_figsize_inches": [7.5, 3.5],
      "overlap_venn_figsize_inches": [6.5, 5.5],
      "per_region_venn_figsize_inches": [18.0, 9.0],
      "bout_legend_fontsize": 9,
      "bout_legend_line_step": 0.075,
      "peth_tick_seconds": 0.5,
      "qlvm_category_label_fontsize": 14,
      "qlvm_label_inset": 0.06,
      "strip_log_ratio_threshold": 10.0,
      "strip_symlog_linthresh": 0.5,
      "stability_marker_size_range": [4.0, 70.0],
      "stability_fr_log10_range": [-1.0, 1.5],
      "consistency_marker_size_range": [6.0, 60.0],
      "consistency_n_tested_range": [2, 6],
      "breadth_marker_size_range": [8.0, 60.0],
      "breadth_z_range": [3.0, 20.0],
      "stability_min_units": 5,
      "vmi_bootstrap_n": 2000,
      "vmi_bootstrap_quantiles": [0.025, 0.975],
      "vmi_n_tested_min": 2,
      "pag_kde_grid_resolution": 220,
      "vmi_histogram_bins": 20,
      "peth_consistency_tolerance_seconds": 0.1,
      "peth_timing_bin_seconds": 0.05,
      "consistency_k_min": 2,
      "consistency_require_majority": true
    },
    "neuronal_decoding_figures": {
      "trace_color": "#1F1F1F",
      "page_color": "#FFFFFF",
      "opacity_range": [0.45, 1.0],
      "main_panel_inches": [5.6, 4.0],
      "side_gap_inches": 0.44,
      "panel_gap_inches": 0.2,
      "page_margin_inches": {
        "left": 0.78,
        "right": 0.14,
        "bottom": 0.6,
        "top": 0.74
      },
      "ci_quantile": 0.995,
      "null_percentile": 99.5
    },
    "anatomy_figures": {
      "excluded_mouse_ids": ["147366"],
      "recording_yield": {
        "fig_size_inches": [7.0, 2.6]
      },
      "unit_positions": {
        "fig_size_inches": [6.0, 5.0],
        "view_elev": 18.0,
        "view_azim": -60.0,
        "shell_vertex_stride": 6,
        "shell_point_size": 0.7,
        "shell_point_alpha": 0.5,
        "region_alpha": 0.1,
        "mb_alpha": 0.04,
        "dot_size": 9.0,
        "dot_alpha": 1.0,
        "legend_marker_scale": 3.0
      },
      "unit_positions_video": {
        "n_frames": 240,
        "fps": 30,
        "dpi": 600,
        "bitrate": 4000
      },
      "unit_waveforms": {
        "lateral_jitter_um": 12.0,
        "waveform_width_um": 12.0,
        "peak_amplitude_target_um": 25.0,
        "inter_shank_wspace": 0.05,
        "opacity_sigma_um": 20.0,
        "n_neighbors_each_side": 5,
        "ap_padding_um": 20.0,
        "dv_padding_um": 30.0,
        "fig_size_inches": [4.5, 7.0]
      }
    },
    "behavioral_feature_distributions": {
      "figure_dpi": 600,
      "panels_per_row": 6,
      "figure_width_inches": 6.4,
      "row_height_inches": 1.0
    },
    "usv_summary_figures": {
      "density_clip_percentile": 98,
      "colorbar_nbins": 5,
      "distance_kde_bw_adjust": 1.5,
      "anova_alpha": 0.05,
      "polar_kde": {
        "grid_points": 200,
        "contour_levels": 100
      },
      "category_polar_kde_grid": {
        "grid_points": 100,
        "contour_levels": 50,
        "panel_inches": [4, 4]
      },
      "estrous_category_kde_grid": {
        "grid_points": 100,
        "contour_levels": 50,
        "panel_inches": [4, 4]
      },
      "panel_inches": {
        "category_estrous_rates_grid": [6, 4.5],
        "category_estrous_ratio_grid": [5, 5]
      },
      "figsizes": {
        "assignment_stacked_bars": [6, 10],
        "assignment_summary_panel": [6, 9],
        "animal_participation_stats": [12, 5],
        "polar_kde_distance_angle": [10, 4],
        "behavior_duration_regressions": [10, 10],
        "distance_by_assignment_kde_anova": [10, 6],
        "duration_histograms_by_sex": [8, 8],
        "hourly_regressions": [4, 5],
        "local_fatigue_binned_trends": [5, 5],
        "estrous_ratio_scatter": [4, 3],
        "estrous_usv_rates": [12, 5],
        "estrous_stage_pie_chart": [7, 7],
        "category_prevalence_and_embedding": [12, 18],
        "category_global_fatigue_heatmap": [14, 10],
        "category_estrous_ratio_grid_stages": [12, 12],
        "session_squeak_time_heatmap": [9.2, 8.2]
      }
    },
    "usv_interval_figures": {
      "edge_color": "#202020",
      "log_histogram": {
        "figsize": [5, 5],
        "alpha": 0.5
      },
      "ic_curves": {
        "figsize": [6, 4]
      },
      "best_fit": {
        "figsize": [5, 5],
        "qq_quantiles": 200,
        "density_grid_points": 500,
        "bin_mean_grid_points": 4000,
        "component_grid_points": 500,
        "component_linewidth": 1.5
      },
      "qq_plot": {
        "figsize": [5, 5],
        "n_quantiles": 200,
        "line_color": "#202020"
      },
      "bootstrap_lrt": {
        "figsize_per_panel": [4, 3],
        "break_gap_factor": 1.5,
        "bins": 50,
        "null_color": "#a0a0a0",
        "observed_color": "#cc0000"
      },
      "serial_dependence": {
        "figsize": [6.2, 2.6],
        "grid_bins": 70,
        "lims_ms": [15.0, 20000.0],
        "density_dark_color": "#1F3A45",
        "diagonal_color": "#8A8F98",
        "boundary_color": "#B4404A",
        "band_color": "#1F1F1F",
        "line_color": "#000000"
      }
    },
    "coactivity_figures": {
      "significance_alpha": 0.05,
      "acoustic_confound": {
        "bins": 40,
        "alpha": 0.5
      },
      "null_distributions": {
        "percentile": 99,
        "bins": 40,
        "null_alpha": 0.5,
        "obs_linewidth": 3.0,
        "threshold_linewidth": 1.0,
        "legend_fontsize": 8
      },
      "per_session_pop_corr": {
        "percentile": 99,
        "bins": 30,
        "null_alpha": 0.55,
        "obs_linewidth": 2.2,
        "threshold_linewidth": 0.8,
        "legend_fontsize": 7
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`neuronal_tuning_figures (bout panels) <vis-neuronal-tuning-figures-bout-panels>`
     - see JSON above
     - bins, windows, smoothing and density grid of the vocal page's bout panels
   * - :ref:`neuronal_tuning_figures (styling and population figures) <vis-neuronal-tuning-figures-styling>`
     - see JSON above
     - inks, figure sizes, fonts, marker-size maps and population-figure statistics
   * - :ref:`neuronal_decoding_figures <vis-neuronal-decoding-figures>`
     - see JSON above
     - inks, opacity ramp, geometry and statistics of the decoding figure
   * - :ref:`anatomy_figures <vis-anatomy-figures>`
     - see JSON above
     - excluded mice and the ``AnatomyFigureMaker`` rendering knobs
   * - :ref:`behavioral_feature_distributions <vis-behavioral-feature-distributions>`
     - see JSON above
     - resolution and layout of the feature-histogram PDF
   * - :ref:`usv_summary_figures <vis-usv-summary-figures>`
     - see JSON above
     - KDE, colorbar, ANOVA and figure-size knobs
   * - :ref:`usv_interval_figures <vis-usv-interval-figures>`
     - see JSON above
     - per-figure sizes, grids and inks of the inter-USV interval figures
   * - :ref:`coactivity_figures <vis-coactivity-figures>`
     - see JSON above
     - significance level and per-figure histogram / line knobs

.. _vis-neuronal-tuning-figures-bout-panels:

``neuronal_tuning_figures`` (bout panels)
   The bout panels of the per-unit vocal page.

   :Options:
      * ``bout_session_bin_seconds`` — the whole-session rate trace's bins; 1 s
      * ``bout_zoom_seconds`` / ``bout_zoom_bin_seconds`` — the densest-window zoom's width and bins; 20 s and 50 ms
      * ``bout_smooth_bins`` — the Gaussian smoothing of both traces, in bins; 2
      * ``bout_trigger_window_seconds`` / ``bout_trigger_bin_seconds`` / ``bout_trigger_smooth_bins`` — the bout-onset-triggered raster's window, bins and smoothing; -1 to 2 s, 20 ms, 2.5 bins
      * ``bout_density_grid`` — the first-versus-later scatter's density grid; 160 per side
      * ``bout_density_levels`` — the bout shares its contours enclose, at most three; 0.5, 0.75, 0.9
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)

.. _vis-neuronal-tuning-figures-styling:

``neuronal_tuning_figures`` (styling and population figures)
   The neutral inks shared by every tuning page and population figure, the figure sizes, the vocal page's fonts and spacing, the marker-size maps and the population-figure statistics.

   :Options:
      **Neutral inks:**

      * ``ink_color`` — near-black text / traces / identity lines; ``#1F1F1F``
      * ``band_color`` — the shuffle-CI band; ``#D3D3D3``
      * ``dash_color`` — dashed references and "no data" notes; ``#888888``
      * ``light_color`` — unassigned-USV bars and their legend; ``#CCCCCC``
      * ``hatch_color`` — hatch edges; ``#BBBBBB``

      **Figure sizes in inches:**

      * ``vocal_page_figsize_inches`` — the one-page vocal layout; 16 x 27
      * ``behavioral_page_width_inches`` / ``behavioral_row_height_inches`` — 16 wide, 2.6 per feature row, at least one row: sized so 8 / 9 pt fonts fit
      * ``region_grid_figsize_inches`` — every 2 x 4 per-region grid; 14 x 6.4
      * ``sign_flip_figsize_inches`` — 10 x 5.4
      * ``pag_gradient_figsize_inches`` — three projections side by side; 18 x 4.8
      * ``peth_timing_figsize_inches`` — the 2 x 7 grid; 18 x 6.6
      * ``behavioral_tier_figsize_inches`` — 7.5 x 3.5
      * ``overlap_venn_figsize_inches`` — 6.5 x 5.5
      * ``per_region_venn_figsize_inches`` — 18 x 9

      **Vocal page and category strip:**

      * ``bout_legend_fontsize`` — 9 pt
      * ``bout_legend_line_step`` — the axes fraction between the "bouts" / "unassigned USVs" labels; 0.075
      * ``peth_tick_seconds`` — PETH tick spacing, anchored on onset; 0.5 s
      * ``qlvm_category_label_fontsize`` — 14 pt
      * ``qlvm_label_inset`` — the fraction of the panel span a category label is held inside the border so it is not clipped; 0.06
      * ``strip_log_ratio_threshold`` — the category strip's max / min rate range above which the axis switches to symlog; 10
      * ``strip_symlog_linthresh`` — the category strip's linear region; 0.5 sp/s

      **Marker-size maps:**

      * ``stability_marker_size_range`` over ``stability_fr_log10_range`` — 4-70 square points over -1 to 1.5, i.e. 0.1-31.6 sp/s, the band the bulk of units fall in
      * ``consistency_marker_size_range`` over ``consistency_n_tested_range`` — 6-60 square points over 2-6 tested sessions, the span in this dataset
      * ``breadth_marker_size_range`` over ``breadth_z_range`` — 8-60 square points over peak z 3-20, so strong outliers do not dominate

      **Population-figure statistics:**

      * ``stability_min_units`` — fewer units leave a region's cross-condition r undrawn; 5
      * ``vmi_bootstrap_n`` — bootstrap resamples; 2000
      * ``vmi_bootstrap_quantiles`` — 0.025 / 0.975, a 95% CI
      * ``vmi_n_tested_min`` — tested sessions for the consistency and sign-flip figures; 2
      * ``pag_kde_grid_resolution`` — points per axis of the PAG density; 220
      * ``vmi_histogram_bins`` — bins over -1..1; 20
      * ``peth_consistency_tolerance_seconds`` — the full width within which a unit's peak times must agree; 0.1 s
      * ``peth_timing_bin_seconds`` — histogram bins; 0.05 s
      * ``consistency_k_min`` — agreeing significant sessions; 2
      * ``consistency_require_majority`` — true: those sessions must also be a strict majority of the significant ones
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)
   :Notes: the ``make_*`` population methods read these instead of taking them as arguments.

.. _vis-neuronal-decoding-figures:

``neuronal_decoding_figures``
   The vocal-bout occupancy decoding figure.

   :Options:
      **Inks:**

      * ``trace_color`` — marker edges and text; ``#1F1F1F``
      * ``page_color`` — the page the opacity ramp blends toward and the face of an unfilled marker; ``#FFFFFF``
      * ``opacity_range`` — the blend weight from the least to the most vocal session, so every curve stays legible; 0.45-1.0

      **Geometry in inches:**

      * ``main_panel_inches`` — 5.6 x 4.0
      * ``side_gap_inches`` — between the main panel and the comparison column; 0.44
      * ``panel_gap_inches`` — between the three stacked comparison panels; 0.2
      * ``page_margin_inches`` — left 0.78, right 0.14, bottom 0.6, top 0.74

      **Statistics drawn:**

      * ``ci_quantile`` — the Student-t quantile of a two-sided 99% error bar; 0.995
      * ``null_percentile`` — the percentile of a rung's own shift null its mean must exceed for a filled marker; 99.5
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)

.. _vis-anatomy-figures:

``anatomy_figures``
   ``excluded_mouse_ids``, mice left out of the anatomy figures (insufficient yield, used in no downstream analysis), and the rendering knobs of ``AnatomyFigureMaker``.

   :Options:
      * ``excluded_mouse_ids`` — a list of mouse ids; ``["147366"]``
      * ``recording_yield.fig_size_inches`` — 7.0 x 2.6 in, a Nature-style single-column band

      ``unit_positions`` — the 3D unit-positions still / builder:

      * ``fig_size_inches`` — 6.0 x 5.0 in
      * ``view_elev`` / ``view_azim`` — 18 / -60 degrees
      * ``shell_vertex_stride`` — every 6th root-mesh vertex in the brain-shell scatter
      * ``shell_point_size`` / ``shell_point_alpha`` — 0.7 square points, 0.5
      * ``region_alpha`` — face alpha of the PAG / MRN / VTA / CENT / SC volumes; 0.1
      * ``mb_alpha`` — kept low because MB is the parent volume and would otherwise hide PAG / MRN / VTA; 0.04
      * ``dot_size`` / ``dot_alpha`` — 9 square points, 1.0
      * ``legend_marker_scale`` — 3x the in-axes dot size for legible swatches

      ``unit_positions_video`` — the rotating video:

      * ``n_frames`` — 240, a full 360 degree sweep
      * ``fps`` — 30 frames/s, so 8 s
      * ``dpi`` — 600 per frame
      * ``bitrate`` — 4000 kbps for the ffmpeg writer

      ``unit_waveforms`` — the per-probe waveform figure:

      * ``lateral_jitter_um`` — 12 um half-range so units sharing a peak channel do not overlap
      * ``waveform_width_um`` — 12 um per waveform
      * ``peak_amplitude_target_um`` — 25 um, the axial height the largest unit's peak maps to
      * ``inter_shank_wspace`` — 0.05, the visible inter-shank break
      * ``opacity_sigma_um`` — 20 um Gaussian sigma of the opacity fall-off with distance from the peak channel
      * ``n_neighbors_each_side`` — 5 same-shank channels above and below the peak
      * ``ap_padding_um`` / ``dv_padding_um`` — 20 / 30 um axis padding
      * ``fig_size_inches`` — 4.5 x 7.0 in
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)
   :Notes: an argument the caller leaves at ``None`` falls back to the same-named key here.

.. _vis-behavioral-feature-distributions:

``behavioral_feature_distributions``
   The per-session feature-histogram PDF that ``compute_behavioral_features`` writes next to each behavioral-features CSV (``*_behavioral_features_histograms.pdf``).

   :Options:
      * ``figure_dpi`` — the resolution the figure is laid out and rasterised at; 600 (the PDF pages themselves are saved at ``figures.dpi``)
      * ``panels_per_row`` — feature panels per row of the grid; 6
      * ``figure_width_inches`` — 6.4
      * ``row_height_inches`` — per row of panels, so a page is ``6.4 x n_rows`` in; 1.0
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)

.. _vis-usv-summary-figures:

``usv_summary_figures``
   The figures of ``usv_summary_statistics``.

   :Options:
      * ``density_clip_percentile`` — every polar KDE is colour-capped at this percentile so a few dense pixels do not wash out the map; 98
      * ``colorbar_nbins`` — colorbar ticks on the polar KDE; 5
      * ``distance_kde_bw_adjust`` — the seaborn bandwidth multiplier of the distance-by-assignment KDE, smoother than Scott's rule for the sparse unassigned group; 1.5
      * ``anova_alpha`` — the ANOVA level that gates the Tukey HSD and the Tukey level itself; 0.05
      * ``polar_kde`` — ``grid_points`` 200 per axis and ``contour_levels`` 100 of the single-pair polar KDE
      * ``category_polar_kde_grid`` / ``estrous_category_kde_grid`` — ``grid_points`` 100, ``contour_levels`` 50 and ``panel_inches`` 4 x 4 per panel of the two grid figures
      * ``panel_inches`` — per-panel inches of the category x estrous rate grid, 6 x 4.5, and ratio grid, 5 x 5
      * ``figsizes`` — the figure size in inches of every fixed-layout figure, keyed by the plotting function's name without ``plot_``; ``category_estrous_ratio_grid_stages`` is the 2 x 2 per-stage companion of the ratio grid
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)

.. _vis-usv-interval-figures:

``usv_interval_figures``
   The figures of ``usv_interval_summary_statistics`` (the inter-USV interval notebook).

   :Options:
      * ``edge_color`` — histogram edges, the selected-K outline, component triangles and the LRT reference line; ``#202020``
      * ``log_histogram`` — ``figsize`` 5 x 5 in, ``alpha`` 0.5 so the two overlaid pools stay readable
      * ``ic_curves.figsize`` — 6 x 4 in
      * ``best_fit`` — ``figsize`` 5 x 5 in, square for the Q-Q inset; ``qq_quantiles`` 200 Q-Q points; ``density_grid_points`` 500, the grid the point-wise mixture curve of ``plot_gmm_fit`` is evaluated on; ``bin_mean_grid_points`` 4000, the denser grid its bin-averaged curve is smoothed on when ``density_as_bin_means`` is on; and ``component_grid_points`` 500 and ``component_linewidth`` 1.5 for the optional per-component curves. The histogram's bin count is the caller's, the notebook passing the per-sex ``bins_per_sex``.
      * ``qq_plot`` — the standalone ``plot_qq`` figure: ``figsize`` 5 x 5 in, ``n_quantiles`` 200 Q-Q points and ``line_color`` ``#202020`` for the y = x reference line
      * ``bootstrap_lrt`` — ``figsize_per_panel`` 4 x 3 in; ``break_gap_factor`` 1.5 (the axis is broken when the observed statistic sits more than 1.5 null-widths past the null); ``bins`` 50; ``null_color`` ``#a0a0a0`` and ``observed_color`` ``#cc0000``
      * ``serial_dependence`` — ``figsize`` 6.2 x 2.6 in; ``grid_bins`` 70 per axis of the 2-D histogram; ``lims_ms`` 15-20000 ms on both axes; and the inks ``density_dark_color`` ``#1F3A45`` (dark end of the density ramp), ``diagonal_color`` ``#8A8F98`` (next = current), ``boundary_color`` ``#B4404A`` (bout boundary), ``band_color`` ``#1F1F1F`` (bootstrap band) and ``line_color`` ``#000000`` (fitted line and bend)
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)

.. _vis-coactivity-figures:

``coactivity_figures``
   The figures of ``make_coactivity_figures`` (the coactivity notebook).

   :Options:
      * ``significance_alpha`` — the two-tailed level that colours an animal's slope line and puts a star over an amplitude bin; 0.05
      * ``acoustic_confound`` — ``bins`` 40 and ``alpha`` 0.5 of the overlaid feature histograms
      * ``null_distributions`` — the 3 x 2 chained-null grid: ``percentile`` 99 of the dashed threshold, ``bins`` 40, ``null_alpha`` 0.5, ``obs_linewidth`` 3.0, ``threshold_linewidth`` 1.0, ``legend_fontsize`` 8
      * ``per_session_pop_corr`` — the smaller per-session panels: ``percentile`` 99, ``bins`` 30, ``null_alpha`` 0.55, ``obs_linewidth`` 2.2, ``threshold_linewidth`` 0.8, ``legend_fontsize`` 7
   :Default: see the JSON above (each sub-key's shipped value is given in its bullet)
   :Notes: colours stay in ``coactivity_colors``.

Visualize 3D behavior (figure/video)
------------------------------------
Once 3D tracked data is available, you can visualize animal social behavior, either in figure or video. This GUI segment allows for a wide array of options in creating such visualizations. For example, you can choose whether you want to view the interaction from above or the side, and you can also choose to rotate the view as the behavior unfolds.

To obtain this visualization, you need to list the root directories of interest (it is best to stick with one), select the *Visualize 3D behavior (figure/video)* option in the GUI, insert the arena directory for that session, pick all desired figure features, click *Next* and then *Visualize*. It is important to point out that there are many more features available in the *visualizations_settings.json* file than are available in the GUI, and these options are explained in detail several sections below:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_2.png
   :align: center
   :alt: Visualize Step 2

.. raw:: html

   <br>

Running this function results in the creation of the *data_animation_examples* subdirectory (if it has not been created already), and the figure/video will be saved inside:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ...
    │   ├── **data_animation_examples**
    │   │   ├── **20250430_145017_3D_30045fr_dark_topview_17features_spectrogram_ch0_Bartul_20260701_133612.png**
    │   │   ├── **20250430_145017_3D_30045-30795fr_dark_topview_17features_spectrogram_ch0_Bartul.mp4**
    │   │   ...
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The */usv-playpen/_parameter_settings/visualizations_settings.json* file contains a section only partially modifiable in the GUI, but it can entirely be modified manually in the *visualizations_settings.json* file:

.. code-block:: json

    "make_behavioral_videos": {
        "arena_directory": "",
        "speaker_audio_file": "",
        "pitch_shifted_audio_bool": false,
        "pitch_shifted_audio_specs": {
            "fs_audio_dir": "hpss_filtered",
            "fs_device_id": "m",
            "fs_channel_id": 1,
            "fs_wav_sampling_rate": 250,
            "fs_octave_shift": -3,
            "fs_volume_adjustment": true,
            "fs_compand_transfer": "0.3,1 6:-70,-60,-20 -5 -90 0.2",
            "fs_noise_reduction_std_threshold": 3,
            "fs_sinc_upper_cutoff_hz": 25000,
            "fs_sox_timeout_s": 600
        },
        "animate_bool": false,
        "video_start_time": 37.0,
        "video_duration": 10.0,
        "plot_theme": "dark",
        "save_fig": true,
        "view_angle": "top",
        "side_azimuth_start": 45,
        "rotate_side_view_bool": false,
        "rotation_speed": 20,
        "history_bool": false,
        "speaker_bool": false,
        "spectrogram_bool": false,
        "spectrogram_ch": 0,
        "raster_plot_bool": false,
        "raster_selection_criteria": {
          "brain_areas": [],
          "other": [
            "good"
          ]
        },
        "raster_special_units": [
          ""
        ],
        "spike_sound_bool": false,
        "beh_features_bool": false,
        "beh_features_to_plot": [],
        "special_beh_features": [],
        "general_figure_specs": {
          "fig_format": "png",
          "animation_codec": "h264_nvenc",
          "animation_codec_preset_flag": "p5",
          "animation_codec_tune_flag": "hq",
          "animation_writer": "ffmpeg",
          "animation_format": "mp4"
        },
        "color_modes": {
          "light_mode": {
            "background_color": "#FFFFFF",
            "node_edge_color": "#8B8B8B",
            "body_edge_color": "#8B8B8B",
            "text_color": "#202020",
            "tick_color": "#202020",
            "arena_line_color": "#202020",
            "arena_mic_color": "#202020",
            "arena_mesh_color": "#202020",
            "spectrogram_text_color": "#202020",
            "faded_feature_color": "#00000033",
            "speaker_color": "#202020"
          },
          "dark_mode": {
            "background_color": "#202020",
            "node_edge_color": "#8B8B8B",
            "body_edge_color": "#8B8B8B",
            "text_color": "#FFFFFF",
            "tick_color": "#FFFFFF",
            "arena_line_color": "#FFFFFF",
            "arena_mic_color": "#FFFFFF",
            "arena_mesh_color": "#FFFFFF",
            "spectrogram_text_color": "#FFFFFF",
            "faded_feature_color": "#FFFFFF33",
            "speaker_color": "#FFFFFF"
          }
        },
        "arena_figure_specs": {
          "arena_node_connections_bool": false,
          "arena_axes_lw": 1.0,
          "arena_mics_lw": 0.75,
          "arena_mics_opacity": 0.25,
          "plot_corners_bool": false,
          "corner_size": 1.0,
          "corner_opacity": 1.0,
          "plot_mesh_walls_bool": true,
          "mesh_opacity": 0.1,
          "active_mic_bool": false,
          "inactive_mic_bool": true,
          "inactive_mic_color": "#898989",
          "text_fontsize": 10,
          "speaker_opacity": 1.0
        },
        "mouse_figure_specs": {
          "node_bool": true,
          "node_size": 3.5,
          "node_opacity": 1.0,
          "node_lw": 0.5,
          "node_connection_lw": 1.0,
          "body_opacity": 0.85,
          "history_point": "Head",
          "history_span_sec": 5,
          "history_ls": "-",
          "history_lw": 0.75
        },
        "subplot_specs": {
          "beh_features_window_size": 10,
          "raster_window_size": 1,
          "raster_lw": 0.3,
          "raster_ll": 10.9,
          "spectrogram_cbar_bool": true,
          "spectrogram_plot_window_size": 1,
          "spectrogram_power_limit": [
            -60,
            0
          ],
          "spectrogram_frequency_limit": [
            30000,
            125000
          ],
          "spectrogram_yticks": [
            50000,
            100000
          ],
          "spectrogram_stft_nfft": 512,
          "plot_usv_segments_bool": true,
          "usv_segments_ypos": 120000,
          "usv_segments_lw": 1.25
        }
    }

The top-level keys of ``make_behavioral_videos`` select the session data, the time window, the view and which panels are drawn:

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`arena_directory <vis-behavioral-videos-arena-directory>`
     - ``''``
     - path to a directory
   * - :ref:`speaker_audio_file <vis-behavioral-videos-speaker-audio-file>`
     - ``''``
     - path to an audio file
   * - :ref:`pitch_shifted_audio_bool <vis-behavioral-videos-pitch-shifted-audio-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`pitch_shifted_audio_specs <vis-behavioral-videos-pitch-shifted-audio-specs>`
     - see JSON above
     - object of ten ``fs_*`` keys
   * - :ref:`animate_bool <vis-behavioral-videos-animate-bool>`
     - ``false``
     - ``true`` (video), ``false`` (figure)
   * - :ref:`video_start_time <vis-behavioral-videos-video-start-time>`
     - ``37.0``
     - seconds
   * - :ref:`video_duration <vis-behavioral-videos-video-duration>`
     - ``10.0``
     - seconds
   * - :ref:`plot_theme <vis-behavioral-videos-plot-theme>`
     - ``'dark'``
     - ``'dark'``, ``'light'``
   * - :ref:`save_fig <vis-behavioral-videos-save-fig>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`view_angle <vis-behavioral-videos-view-angle>`
     - ``'top'``
     - ``'top'``, ``'side'``
   * - :ref:`side_azimuth_start <vis-behavioral-videos-side-azimuth-start>`
     - ``45``
     - degrees
   * - :ref:`rotate_side_view_bool <vis-behavioral-videos-rotate-side-view-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`rotation_speed <vis-behavioral-videos-rotation-speed>`
     - ``20``
     - degrees per second
   * - :ref:`history_bool <vis-behavioral-videos-history-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`speaker_bool <vis-behavioral-videos-speaker-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`spectrogram_bool <vis-behavioral-videos-spectrogram-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`spectrogram_ch <vis-behavioral-videos-spectrogram-ch>`
     - ``0``
     - audio channel index
   * - :ref:`raster_plot_bool <vis-behavioral-videos-raster-plot-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`raster_selection_criteria <vis-behavioral-videos-raster-selection-criteria>`
     - see JSON above
     - object with ``brain_areas`` and ``other`` lists
   * - :ref:`raster_special_units <vis-behavioral-videos-raster-special-units>`
     - ``[""]``
     - list of unit names
   * - :ref:`spike_sound_bool <vis-behavioral-videos-spike-sound-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`beh_features_bool <vis-behavioral-videos-beh-features-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`beh_features_to_plot <vis-behavioral-videos-beh-features-to-plot>`
     - ``[]``
     - list of behavioral feature names
   * - :ref:`special_beh_features <vis-behavioral-videos-special-beh-features>`
     - ``[]``
     - list of behavioral feature names

.. _vis-behavioral-videos-arena-directory:

``arena_directory``
   Path to the directory with the 3D tracked arena data.

   :Options: path (string) to a directory
   :Default: ``''``

.. _vis-behavioral-videos-speaker-audio-file:

``speaker_audio_file``
   Path to the audio file containing the playback speaker sound.

   :Options: path (string) to an audio file
   :Default: ``''``

.. _vis-behavioral-videos-pitch-shifted-audio-bool:

``pitch_shifted_audio_bool``
   If "Yes", automatically frequency-shift the session USVs over the chosen ``[video_start_time, video_start_time + video_duration]`` window into the human-audible range and mux the result onto the video.

   :Options:
      * ``true`` ("Yes") — pitch-shift the window's USVs and mux them onto the video
      * ``false`` ("No") — no pitch-shifted audio
   :Default: ``false``
   :Notes: the mux runs under a time limit; a failed mux is reported as a warning. The pitch-shift recipe is ``pitch_shifted_audio_specs``. This setting replaces the former manual ``sequence_audio_file`` path.

.. _vis-behavioral-videos-pitch-shifted-audio-specs:

``pitch_shifted_audio_specs``
   Pitch-shift recipe used when ``pitch_shifted_audio_bool`` is "Yes".

   :Options: an object with the keys ``fs_audio_dir``, ``fs_device_id``, ``fs_channel_id``, ``fs_wav_sampling_rate``, ``fs_octave_shift``, ``fs_volume_adjustment``, ``fs_compand_transfer``, ``fs_noise_reduction_std_threshold``, ``fs_sinc_upper_cutoff_hz``, ``fs_sox_timeout_s``; the same keys as the *Analyze* page's ``frequency_shift_audio_segment`` block
   :Default: see the JSON above
   :Notes: ``fs_sox_timeout_s`` is the upper bound in seconds on each SoX subprocess of the pitch shift (``600``): a SoX run on a video-length window takes seconds, so one running this long has hung and is stopped rather than blocking the render.

.. _vis-behavioral-videos-animate-bool:

``animate_bool``
   Boolean value indicating whether to animate the figure or not.

   :Options:
      * ``true`` ("Yes") — render a video
      * ``false`` ("No") — create a figure
   :Default: ``false``

.. _vis-behavioral-videos-video-start-time:

``video_start_time``
   Start time of the figure/video in seconds.

   :Options: a number of seconds
   :Default: ``37.0``

.. _vis-behavioral-videos-video-duration:

``video_duration``
   Duration of the video in seconds.

   :Options: a number of seconds
   :Default: ``10.0``

.. _vis-behavioral-videos-plot-theme:

``plot_theme``
   Plot background.

   :Options:
      * ``'dark'`` — dark plot background (ink palette ``color_modes.dark_mode``)
      * ``'light'`` — light plot background (ink palette ``color_modes.light_mode``)
   :Default: ``'dark'``

.. _vis-behavioral-videos-save-fig:

``save_fig``
   If True, the figure will be saved in the *data_animation_examples* subdirectory.

   :Options: ``true``, ``false``
   :Default: ``true``

.. _vis-behavioral-videos-view-angle:

``view_angle``
   View of social behavior in the playpen arena.

   :Options:
      * ``'top'`` — top view
      * ``'side'`` — side view
   :Default: ``'top'``

.. _vis-behavioral-videos-side-azimuth-start:

``side_azimuth_start``
   Azimuth angle of the side view (in deg).

   :Options: an angle in degrees
   :Default: ``45``

.. _vis-behavioral-videos-rotate-side-view-bool:

``rotate_side_view_bool``
   Rotate the side view or not.

   :Options: ``true``, ``false``
   :Default: ``false``
   :Notes: NB: angles wrap around.

.. _vis-behavioral-videos-rotation-speed:

``rotation_speed``
   Rotation speed of the side view (in deg/s).

   :Options: a number of degrees per second
   :Default: ``20``

.. _vis-behavioral-videos-history-bool:

``history_bool``
   Plot the location history of one body node.

   :Options: ``true``, ``false``
   :Default: ``false``
   :Notes: the node and the span are set by ``mouse_figure_specs.history_point`` and ``mouse_figure_specs.history_span_sec``.

.. _vis-behavioral-videos-speaker-bool:

``speaker_bool``
   Plot the playback speaker.

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-behavioral-videos-spectrogram-bool:

``spectrogram_bool``
   Plot the spectrogram of the audio segment.

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-behavioral-videos-spectrogram-ch:

``spectrogram_ch``
   Channel of the audio segment to plot.

   :Options: an audio channel index (integer)
   :Default: ``0``

.. _vis-behavioral-videos-raster-plot-bool:

``raster_plot_bool``
   Plot the live spiking raster of the neural data.

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-behavioral-videos-raster-selection-criteria:

``raster_selection_criteria``
   Criteria for selecting the neurons to plot in the raster.

   :Options:
      * ``brain_areas`` — list of brain areas to include in the raster plot
      * ``other`` — list of other criteria to include in the raster plot (e.g., "good" for unit type)
   :Default: ``brain_areas`` ``[]``, ``other`` ``["good"]``

.. _vis-behavioral-videos-raster-special-units:

``raster_special_units``
   Unit(s) to highlight in the raster plot (*e.g.*, "imec0_cl0000_ch361").

   :Options: a list of unit names
   :Default: ``[""]``

.. _vis-behavioral-videos-spike-sound-bool:

``spike_sound_bool``
   Make spike sound each time the highlighted unit spikes.

   :Options: ``true``, ``false``
   :Default: ``false``
   :Notes: the sound is muxed onto the video with ffmpeg under a time limit; a mux that fails raises and keeps the spike-sound WAV so it can be redone by hand.

.. _vis-behavioral-videos-beh-features-bool:

``beh_features_bool``
   Plot the behavioral features dynamics subplot.

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-behavioral-videos-beh-features-to-plot:

``beh_features_to_plot``
   List of behavioral features in the subplot.

   :Options: a list of behavioral feature names
   :Default: ``[]``

.. _vis-behavioral-videos-special-beh-features:

``special_beh_features``
   List of highlighted behavioral features in the subplot.

   :Options: a list of behavioral feature names
   :Default: ``[]``

Parameters controlling the figure output format and video encoding (``general_figure_specs``):

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`fig_format <vis-general-figure-specs-fig-format>`
     - ``'png'``
     - image format string
   * - :ref:`animation_codec <vis-general-figure-specs-animation-codec>`
     - ``'h264_nvenc'``
     - FFMPEG codec name
   * - :ref:`animation_codec_preset_flag <vis-general-figure-specs-animation-codec-preset-flag>`
     - ``'p5'``
     - codec preset string
   * - :ref:`animation_codec_tune_flag <vis-general-figure-specs-animation-codec-tune-flag>`
     - ``'hq'``
     - codec tune string
   * - :ref:`animation_writer <vis-general-figure-specs-animation-writer>`
     - ``'ffmpeg'``
     - matplotlib writer name
   * - :ref:`animation_format <vis-general-figure-specs-animation-format>`
     - ``'mp4'``
     - video container string

.. _vis-general-figure-specs-fig-format:

``fig_format``
   Output image format for still frames.

   :Options: an image format string (e.g. "png")
   :Default: ``'png'``

.. _vis-general-figure-specs-animation-codec:

``animation_codec``
   FFMPEG video codec for the rendered animation.

   :Options: an FFMPEG codec name (e.g. "h264_nvenc")
   :Default: ``'h264_nvenc'``

.. _vis-general-figure-specs-animation-codec-preset-flag:

``animation_codec_preset_flag``
   Codec preset / speed-quality flag.

   :Options: a codec preset string (e.g. "p5")
   :Default: ``'p5'``

.. _vis-general-figure-specs-animation-codec-tune-flag:

``animation_codec_tune_flag``
   Codec tuning flag.

   :Options: a codec tune string (e.g. "hq")
   :Default: ``'hq'``

.. _vis-general-figure-specs-animation-writer:

``animation_writer``
   Matplotlib animation writer.

   :Options: a matplotlib animation writer name (e.g. "ffmpeg")
   :Default: ``'ffmpeg'``

.. _vis-general-figure-specs-animation-format:

``animation_format``
   Output video container format.

   :Options: a video container format string (e.g. "mp4")
   :Default: ``'mp4'``

Parameters setting the ink palette of each ``plot_theme`` (``color_modes``; one map per theme, ``light_mode`` for "light" and ``dark_mode`` for "dark", every value a hex string):

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`background_color <vis-color-modes-background-color>`
     - ``#FFFFFF`` / ``#202020``
     - hex string (light / dark)
   * - :ref:`node_edge_color <vis-color-modes-edge-colors>`
     - ``#8B8B8B`` / ``#8B8B8B``
     - hex string (light / dark)
   * - :ref:`body_edge_color <vis-color-modes-edge-colors>`
     - ``#8B8B8B`` / ``#8B8B8B``
     - hex string (light / dark)
   * - :ref:`text_color <vis-color-modes-text-tick-colors>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)
   * - :ref:`tick_color <vis-color-modes-text-tick-colors>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)
   * - :ref:`arena_line_color <vis-color-modes-arena-colors>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)
   * - :ref:`arena_mic_color <vis-color-modes-arena-colors>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)
   * - :ref:`arena_mesh_color <vis-color-modes-arena-colors>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)
   * - :ref:`spectrogram_text_color <vis-color-modes-spectrogram-text-color>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)
   * - :ref:`faded_feature_color <vis-color-modes-faded-feature-color>`
     - ``#00000033`` / ``#FFFFFF33``
     - 8-digit hex string with alpha (light / dark)
   * - :ref:`speaker_color <vis-color-modes-speaker-color>`
     - ``#202020`` / ``#FFFFFF``
     - hex string (light / dark)

.. _vis-color-modes-background-color:

``background_color``
   Figure and panel background.

   :Options: a hex string
   :Default: ``#FFFFFF`` (light), ``#202020`` (dark)

.. _vis-color-modes-edge-colors:

``node_edge_color`` / ``body_edge_color``
   Edge colour of the body-node spheres and of the body polygons.

   :Options: a hex string
   :Default: ``#8B8B8B`` in both themes
   :Notes: a mid grey that reads on either background.

.. _vis-color-modes-text-tick-colors:

``text_color`` / ``tick_color``
   Text, tick and spine colour.

   :Options: a hex string
   :Default: ``#202020`` (light), ``#FFFFFF`` (dark)
   :Notes: the opposite of the background.

.. _vis-color-modes-arena-colors:

``arena_line_color`` / ``arena_mic_color`` / ``arena_mesh_color``
   Arena axes, microphone markers and mesh walls.

   :Options: a hex string
   :Default: ``#202020`` (light), ``#FFFFFF`` (dark)
   :Notes: same ink as the text.

.. _vis-color-modes-spectrogram-text-color:

``spectrogram_text_color``
   Spectrogram labels and colorbar text.

   :Options: a hex string
   :Default: ``#202020`` (light), ``#FFFFFF`` (dark)
   :Notes: same ink as the text.

.. _vis-color-modes-faded-feature-color:

``faded_feature_color``
   Trace colour of a partner-pair (dyadic) behavioral feature that is not among the highlighted ``special_beh_features``.

   :Options: an 8-digit hex string with an alpha channel
   :Default: ``#00000033`` (light), ``#FFFFFF33`` (dark)
   :Notes: 8-digit hex with alpha ``33``, about 20% opacity, so the highlighted traces stand out; dyadic features otherwise take ``text_color``.

.. _vis-color-modes-speaker-color:

``speaker_color``
   The playback speaker marker.

   :Options: a hex string
   :Default: ``#202020`` (light), ``#FFFFFF`` (dark)
   :Notes: same ink as the text.

Parameters specific to the arena figure include:

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`arena_node_connections_bool <vis-arena-figure-specs-arena-node-connections-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`arena_axes_lw <vis-arena-figure-specs-arena-axes-lw>`
     - ``1.0``
     - line width
   * - :ref:`arena_mics_lw <vis-arena-figure-specs-arena-mics-lw>`
     - ``0.75``
     - line width
   * - :ref:`arena_mics_opacity <vis-arena-figure-specs-arena-mics-opacity>`
     - ``0.25``
     - opacity
   * - :ref:`plot_corners_bool <vis-arena-figure-specs-plot-corners-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`corner_size <vis-arena-figure-specs-corner-size>`
     - ``1.0``
     - sphere size
   * - :ref:`corner_opacity <vis-arena-figure-specs-corner-opacity>`
     - ``1.0``
     - opacity
   * - :ref:`plot_mesh_walls_bool <vis-arena-figure-specs-plot-mesh-walls-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`mesh_opacity <vis-arena-figure-specs-mesh-opacity>`
     - ``0.1``
     - opacity
   * - :ref:`active_mic_bool <vis-arena-figure-specs-active-mic-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`inactive_mic_bool <vis-arena-figure-specs-inactive-mic-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`inactive_mic_color <vis-arena-figure-specs-inactive-mic-color>`
     - ``#898989``
     - hex string
   * - :ref:`text_fontsize <vis-arena-figure-specs-text-fontsize>`
     - ``10``
     - font size
   * - :ref:`speaker_opacity <vis-arena-figure-specs-speaker-opacity>`
     - ``1.0``
     - opacity

.. _vis-arena-figure-specs-arena-node-connections-bool:

``arena_node_connections_bool``
   Plots connections between corner and nearest microphones.

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-arena-figure-specs-arena-axes-lw:

``arena_axes_lw``
   Line width of the arena axes.

   :Options: a line width
   :Default: ``1.0``

.. _vis-arena-figure-specs-arena-mics-lw:

``arena_mics_lw``
   Line width of the microphones.

   :Options: a line width
   :Default: ``0.75``

.. _vis-arena-figure-specs-arena-mics-opacity:

``arena_mics_opacity``
   Opacity of the microphones.

   :Options: an opacity value
   :Default: ``0.25``

.. _vis-arena-figure-specs-plot-corners-bool:

``plot_corners_bool``
   Plot different color spheres in corners of the arena.

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-arena-figure-specs-corner-size:

``corner_size``
   Size of the corner spheres.

   :Options: a sphere size
   :Default: ``1.0``

.. _vis-arena-figure-specs-corner-opacity:

``corner_opacity``
   Opacity of the corner spheres.

   :Options: an opacity value
   :Default: ``1.0``

.. _vis-arena-figure-specs-plot-mesh-walls-bool:

``plot_mesh_walls_bool``
   Plot the mesh walls of the arena.

   :Options: ``true``, ``false``
   :Default: ``true``

.. _vis-arena-figure-specs-mesh-opacity:

``mesh_opacity``
   Opacity of the mesh walls.

   :Options: an opacity value
   :Default: ``0.1``

.. _vis-arena-figure-specs-active-mic-bool:

``active_mic_bool``
   Plots the active microphone (whose spectrogram is shown).

   :Options: ``true``, ``false``
   :Default: ``false``

.. _vis-arena-figure-specs-inactive-mic-bool:

``inactive_mic_bool``
   Plots the inactive microphones (whose spectrograms are not shown).

   :Options: ``true``, ``false``
   :Default: ``true``

.. _vis-arena-figure-specs-inactive-mic-color:

``inactive_mic_color``
   Color of the inactive microphones.

   :Options: a hex string
   :Default: ``#898989``

.. _vis-arena-figure-specs-text-fontsize:

``text_fontsize``
   Font size of the text in the arena figure.

   :Options: a font size
   :Default: ``10``

.. _vis-arena-figure-specs-speaker-opacity:

``speaker_opacity``
   Opacity of the playback speaker.

   :Options: an opacity value
   :Default: ``1.0``

Parameters specific to the mouse figure include:

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`node_bool <vis-mouse-figure-specs-node-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`node_size <vis-mouse-figure-specs-node-size>`
     - ``3.5``
     - sphere size
   * - :ref:`node_opacity <vis-mouse-figure-specs-node-opacity>`
     - ``1.0``
     - opacity
   * - :ref:`node_lw <vis-mouse-figure-specs-node-lw>`
     - ``0.5``
     - line width
   * - :ref:`node_connection_lw <vis-mouse-figure-specs-node-connection-lw>`
     - ``1.0``
     - line width
   * - :ref:`body_opacity <vis-mouse-figure-specs-body-opacity>`
     - ``0.85``
     - opacity
   * - :ref:`history_point <vis-mouse-figure-specs-history-point>`
     - ``'Head'``
     - body node name
   * - :ref:`history_span_sec <vis-mouse-figure-specs-history-span-sec>`
     - ``5``
     - seconds
   * - :ref:`history_ls <vis-mouse-figure-specs-history-ls>`
     - ``'-'``
     - ``'-'``, ``'--'``, ``'-.'``, ``':'``
   * - :ref:`history_lw <vis-mouse-figure-specs-history-lw>`
     - ``0.75``
     - line width

.. _vis-mouse-figure-specs-node-bool:

``node_bool``
   Plot mouse body nodes as spheres.

   :Options: ``true``, ``false``
   :Default: ``true``

.. _vis-mouse-figure-specs-node-size:

``node_size``
   Size of the body node spheres.

   :Options: a sphere size
   :Default: ``3.5``

.. _vis-mouse-figure-specs-node-opacity:

``node_opacity``
   Opacity of the body node spheres.

   :Options: an opacity value
   :Default: ``1.0``

.. _vis-mouse-figure-specs-node-lw:

``node_lw``
   Line width of the body node spheres.

   :Options: a line width
   :Default: ``0.5``

.. _vis-mouse-figure-specs-node-connection-lw:

``node_connection_lw``
   Plots connections between body nodes.

   :Options: a line width
   :Default: ``1.0``

.. _vis-mouse-figure-specs-body-opacity:

``body_opacity``
   Opacity of the body polygons connected with nodes.

   :Options: an opacity value
   :Default: ``0.85``

.. _vis-mouse-figure-specs-history-point:

``history_point``
   Plot history of particular body point.

   :Options: a body node name
   :Default: ``'Head'``
   :Notes: drawn when ``history_bool`` is on.

.. _vis-mouse-figure-specs-history-span-sec:

``history_span_sec``
   Time span of the history in seconds.

   :Options: a number of seconds
   :Default: ``5``
   :Notes: **will fail if history is set to start before tracking!**

.. _vis-mouse-figure-specs-history-ls:

``history_ls``
   Line style of the history plot.

   :Options: a line style (e.g., "-", "--", "-.", ":")
   :Default: ``'-'``

.. _vis-mouse-figure-specs-history-lw:

``history_lw``
   Line width of the history plot.

   :Options: a line width
   :Default: ``0.75``

Parameters specific to subplots include:

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`beh_features_window_size <vis-subplot-specs-beh-features-window-size>`
     - ``10``
     - seconds
   * - :ref:`raster_window_size <vis-subplot-specs-raster-window-size>`
     - ``1``
     - seconds
   * - :ref:`raster_lw <vis-subplot-specs-raster-lw>`
     - ``0.3``
     - line width
   * - :ref:`raster_ll <vis-subplot-specs-raster-ll>`
     - ``10.9``
     - line length
   * - :ref:`spectrogram_cbar_bool <vis-subplot-specs-spectrogram-cbar-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`spectrogram_plot_window_size <vis-subplot-specs-spectrogram-plot-window-size>`
     - ``1``
     - seconds
   * - :ref:`spectrogram_power_limit <vis-subplot-specs-spectrogram-power-limit>`
     - ``[-60, 0]``
     - ``[lower, upper]`` in dB
   * - :ref:`spectrogram_frequency_limit <vis-subplot-specs-spectrogram-frequency-limit>`
     - ``[30000, 125000]``
     - ``[lower, upper]`` in Hz
   * - :ref:`spectrogram_yticks <vis-subplot-specs-spectrogram-yticks>`
     - ``[50000, 100000]``
     - list of frequencies in Hz
   * - :ref:`spectrogram_stft_nfft <vis-subplot-specs-spectrogram-stft-nfft>`
     - ``512``
     - window size in samples
   * - :ref:`plot_usv_segments_bool <vis-subplot-specs-plot-usv-segments-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_segments_ypos <vis-subplot-specs-usv-segments-ypos>`
     - ``120000``
     - frequency in Hz
   * - :ref:`usv_segments_lw <vis-subplot-specs-usv-segments-lw>`
     - ``1.25``
     - line width

.. _vis-subplot-specs-beh-features-window-size:

``beh_features_window_size``
   Time window of the behavioral features subplot (in s).

   :Options: a number of seconds
   :Default: ``10``
   :Notes: **will fail if is set beyond tracking boundaries!**

.. _vis-subplot-specs-raster-window-size:

``raster_window_size``
   Time window of the raster subplot (in s).

   :Options: a number of seconds
   :Default: ``1``
   :Notes: **will fail if is set beyond tracking boundaries!**

.. _vis-subplot-specs-raster-lw:

``raster_lw``
   Horizontal line width of spikes in the raster plot.

   :Options: a line width
   :Default: ``0.3``

.. _vis-subplot-specs-raster-ll:

``raster_ll``
   Vertical line length of spikes in the raster plot.

   :Options: a line length
   :Default: ``10.9``

.. _vis-subplot-specs-spectrogram-cbar-bool:

``spectrogram_cbar_bool``
   Plot spectrogram colorbar.

   :Options: ``true``, ``false``
   :Default: ``true``

.. _vis-subplot-specs-spectrogram-plot-window-size:

``spectrogram_plot_window_size``
   Time window of the spectrogram subplot (in s).

   :Options: a number of seconds
   :Default: ``1``
   :Notes: **will fail if is set beyond tracking boundaries!**

.. _vis-subplot-specs-spectrogram-power-limit:

``spectrogram_power_limit``
   Lower and upper limits of the spectrogram colorbar (in dB).

   :Options: a two-element list ``[lower, upper]`` in dB
   :Default: ``[-60, 0]``

.. _vis-subplot-specs-spectrogram-frequency-limit:

``spectrogram_frequency_limit``
   Lower and upper limits of the spectrogram frequency axis (in Hz).

   :Options: a two-element list ``[lower, upper]`` in Hz
   :Default: ``[30000, 125000]``

.. _vis-subplot-specs-spectrogram-yticks:

``spectrogram_yticks``
   Y-axis ticks of the spectrogram (in Hz).

   :Options: a list of frequencies in Hz
   :Default: ``[50000, 100000]``

.. _vis-subplot-specs-spectrogram-stft-nfft:

``spectrogram_stft_nfft``
   Window size for the spectrogram calculation.

   :Options: a window size in samples (integer)
   :Default: ``512``

.. _vis-subplot-specs-plot-usv-segments-bool:

``plot_usv_segments_bool``
   Plot the Deep Audio Segmenter (DAS)-detected USV segments in the spectrogram.

   :Options: ``true``, ``false``
   :Default: ``true``

.. _vis-subplot-specs-usv-segments-ypos:

``usv_segments_ypos``
   Y-axis position of the USV segments in the spectrogram (in Hz).

   :Options: a frequency in Hz
   :Default: ``120000``

.. _vis-subplot-specs-usv-segments-lw:

``usv_segments_lw``
   Line width of the USV segments in the spectrogram.

   :Options: a line width
   :Default: ``1.25``

Render the QLVM torus-traversal
-------------------------------
The ``usv_playpen.visualizations.qlvm_torus_traversal_video`` module renders a two-panel toroidal (doughnut-shaped) "torus walkthrough" animation (the in-house, torch-free port of ``qmc_deep_gen``'s ``inference_latents_video.py``). The **left** panel is the QLVM (the in-house quasi-Monte Carlo latent variable model) latent map of the regular map — the QLVM category bundle's density heatmap with its R-1 … R-k category contours (no axes/ticks) and a recency-coloured trajectory trail (cyan at the current position, fading to white going back, built with ``create_colormap``); the **right** panel is a phase-specific spectrogram board. All spectrograms have their SAM2 (Segment Anything Model 2) mask applied (``apply_mask``) and the USV centred in its window with equal padding on both sides (so duration is preserved, not stretched). It runs in three parts, each introduced by a title card:

- **Part 1 — Category peaks**: one phase per category, the right panel showing the spectrogram nearest its peak (the category's label position in the bundle, the pixel farthest from its boundary) surrounded by its ``m`` nearest USVs in concentric rings; on the left the active category is outlined in a thick **pulsating cyan** contour with a cyan dot at its centre.
- **Part 2 — Peak-to-peak walks**: shortest-torus-path walks between random category peaks; the right 5×15 grid fills row-major with the nearest USV at each visited position (the current tile bordered in cyan).
- **Part 3 — Boundary crossings**: curved walks that wrap the torus edges/corners; the right grid's columns are trajectory positions and rows are nearest neighbours (the current trajectory tile bordered in cyan).

It is **cohort-level** — it reads the QLVM category bundle + the consolidated store, not a session directory — and is exposed both as the ``qlvm-torus-traversal-video`` CLI and in the GUI *Visualize* window (third column): "Render QLVM demo video" (Yes/No) and a **Video sampling rate (fps)** slider (the category grid has one level, so there is no clustering selector). A single **Spectrograms directory** **Browse** field lives in the left column under **Credentials directory** (the shared ``shared_resources.spectrograms_dir`` base, under which the consolidated store is resolved by convention).

To render it, set the *Spectrograms directory*, select *Render QLVM demo video*, choose the *Video sampling rate (fps)*, click *Next* and then *Visualize* (the video always uses the regular map and ignores the *QLVM map* selector):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_3.png
   :align: center
   :alt: Visualize Step 3

.. raw:: html

   <br>

**Which QLVM map.** The USV sequence figure and the embedding thumbnails draw ONE QLVM map, chosen by ``shared_resources.qlvm_map`` in */usv-playpen/_parameter_settings/visualizations_settings.json* (the GUI's **QLVM map (all QLVM figures)** selector at the top of the third column, whose tooltip says the same; the embedding explorer starts on it too). This video is the exception: it always uses the regular map ``qlvm`` and ignores ``shared_resources.qlvm_map`` (a conditional choice there is reported in a message and passed over, not refused), because the category bundle it draws exists on the regular map only. The maps are the production models of ``os_utils.QLVM_MAPS``: ``qlvm`` (the regular model, the default), ``qlvm_duration``, ``qlvm_entropy``, ``qlvm_bandwidth`` and ``qlvm_loudness`` (the duration-, spectral-entropy-, bandwidth- and loudness-conditional models; all five masked and time-stretched, ``os_utils.QLVM_PRODUCTION_MODEL_CELLS``). A map ``P`` places USVs at the summary columns ``P1`` / ``P2``. There is one category column, ``qlvm_category`` (R-1 … R-k, written by ``assign-qlvm-categories``): the categories are defined on the regular map and label the USV, so every map's figure colours or groups USVs by ``qlvm_category``.

**Where the category geometry comes from.** Every QLVM figure that draws category boundaries, places category centres or shows a density landscape — this video, the USV sequence figure, the embedding thumbnails, the embedding explorer, the neuronal tuning watersheds, the manifold filter atlas and the category-embedding panel — reads it from ONE place, the QLVM category bundle ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY`` (``/mnt/falkner/Bartul/spectrograms/categories``, written by ``build-qlvm-categories``; a module constant, not a setting, because the settings' experimenter folders are re-keyed to the active experimenter), through ``os_utils.load_qlvm_category_bundle``: its periodic ``label_grid`` (the partition the summaries' ``qlvm_category`` is assigned from), its smoothed corpus ``density`` and its category label positions (``category_nomenclature.json``). The bundle partitions the regular map's torus, so it is drawn on ``qlvm`` only: a figure of a conditional map draws no category boundaries (it colours the USVs by their ``qlvm_category`` instead) and says so, and no figure estimates boundaries from the data. The earlier ``<spectrograms_dir>/qlvm_v3/<map>/arrays_{coarse,fine}.npz`` reference arrays are no longer read.

Inputs of the video:

- the QLVM category bundle (above) — the ``density`` background, the ``label_grid`` contours and the category label positions (the peaks / path waypoints). It is defined on the regular map, so the video always renders ``qlvm`` whatever ``shared_resources.qlvm_map`` says.
- ``<spectrograms_dir>/spectrograms_*.h5`` (newest match) — the consolidated per-session spectrogram/SAM2 store. It supplies BOTH the per-USV latent coords (per-session ``qlvm/<key>/qlvm``, the regular map's ``qlvm1`` / ``qlvm2``) for the nearest-neighbour lookup AND the spectrograms (``spectrogram/<key>/spectrograms``) shown on the right. No latents pickle is read at render time, and coverage spans all sessions in the store.

**Prerequisite** — the consolidated H5 must carry the per-session ``qlvm/<key>/qlvm`` (n, 2) coordinates of the production regular cell the bundle is defined on. The video checks the store's provenance first (``check_store_map_provenance``): the per-map group ``qlvm_models/<map>`` (``qlvm_models/qlvm`` here) records the cell the coordinates come from — as the ``model_id`` attr in a production store, as ``package_root`` / ``cell`` attrs in a v3 store, from which the id is computed — and it must equal the production cell's model id ``spectrograms/qlvm/qlvm`` (``os_utils.qlvm_cell_model_id``, mount-independent). A production store (``consolidate-spectrogram-store`` in production mode) passes, while a store of another cell (the v3 archive the same command writes in v3 mode, whose regular map is the v3 phase 6 cell, see :doc:`Process`) places the USVs on another torus, where the bundle's regions would mislabel them, and is refused; a store without the group is refused as well, and a store without the coordinates raises a clear error naming the map.

Because it is cohort-level, the output is not written next to a session; the ``.mp4`` lands in the project-wide ``figures.save_directory`` with a render-timestamped name (``qlvm_torus_traversal_qlvm_<YYYYMMDD_HHMMSS>.mp4``; always the regular map) unless an explicit ``--output-path`` is given:

.. parsed-literal::

    ├── ...  (the ``figures.save_directory`` cohort output folder)
    │   ├── **qlvm_torus_traversal_qlvm_20250430_145017.mp4**
    │   ...

The render parameters live in the ``qlvm_torus_traversal_video`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*:

.. code-block:: json

    "qlvm_torus_traversal_video": {
        "fps": 20,
        "dpi": 100,
        "m": 36,
        "cluster_hold_frames": 60,
        "peak_traverse_frames": 200,
        "boundary_traverse_frames": 200,
        "title_card_frames": 45,
        "samples_per_trace": 75,
        "peak_jitter_sigma": 0.015,
        "boundary_curve_amplitude": 0.18,
        "boundary_positions_per_walk": 15,
        "boundary_neighbors": 5,
        "seed": 0,
        "peaks_only": false,
        "spec_cache_size": 2048,
        "apply_mask": true,
        "accent_color": "#00FFFF",
        "palette": {
            "white": "#FFFFFF",
            "black": "#000000",
            "gray": "#808080",
            "light_gray": "#D3D3D3",
            "subtitle": "#444444"
        },
        "heatmap_vmax_percentile": 95,
        "bitrate": 4000
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`fps <vis-qlvm-torus-traversal-video-fps>`
     - ``20``
     - frames per second > 0
   * - :ref:`dpi <vis-qlvm-torus-traversal-video-dpi>`
     - ``100``
     - dots per inch > 0
   * - :ref:`m <vis-qlvm-torus-traversal-video-m>`
     - ``36``
     - number of USVs
   * - :ref:`cluster_hold_frames <vis-qlvm-torus-traversal-video-cluster-hold-frames>`
     - ``60``
     - number of frames
   * - :ref:`peak_traverse_frames <vis-qlvm-torus-traversal-video-peak-traverse-frames>`
     - ``200``
     - number of frames
   * - :ref:`boundary_traverse_frames <vis-qlvm-torus-traversal-video-boundary-traverse-frames>`
     - ``200``
     - number of frames
   * - :ref:`title_card_frames <vis-qlvm-torus-traversal-video-title-card-frames>`
     - ``45``
     - number of frames
   * - :ref:`samples_per_trace <vis-qlvm-torus-traversal-video-samples-per-trace>`
     - ``75``
     - number of positions
   * - :ref:`peak_jitter_sigma <vis-qlvm-torus-traversal-video-peak-jitter-sigma>`
     - ``0.015``
     - torus units
   * - :ref:`boundary_curve_amplitude <vis-qlvm-torus-traversal-video-boundary-curve-amplitude>`
     - ``0.18``
     - curvature amplitude
   * - :ref:`boundary_positions_per_walk <vis-qlvm-torus-traversal-video-boundary-positions-per-walk>`
     - ``15``
     - number of positions
   * - :ref:`boundary_neighbors <vis-qlvm-torus-traversal-video-boundary-neighbors>`
     - ``5``
     - number of neighbours
   * - :ref:`seed <vis-qlvm-torus-traversal-video-seed>`
     - ``0``
     - integer RNG seed
   * - :ref:`peaks_only <vis-qlvm-torus-traversal-video-peaks-only>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`spec_cache_size <vis-qlvm-torus-traversal-video-spec-cache-size>`
     - ``2048``
     - number of spectrograms
   * - :ref:`apply_mask <vis-qlvm-torus-traversal-video-apply-mask>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`accent_color <vis-qlvm-torus-traversal-video-accent-color>`
     - ``"#00FFFF"``
     - hex colour
   * - :ref:`palette.white <vis-qlvm-torus-traversal-video-palette>`
     - ``"#FFFFFF"``
     - hex colour
   * - :ref:`palette.black <vis-qlvm-torus-traversal-video-palette>`
     - ``"#000000"``
     - hex colour
   * - :ref:`palette.gray <vis-qlvm-torus-traversal-video-palette>`
     - ``"#808080"``
     - hex colour
   * - :ref:`palette.light_gray <vis-qlvm-torus-traversal-video-palette>`
     - ``"#D3D3D3"``
     - hex colour
   * - :ref:`palette.subtitle <vis-qlvm-torus-traversal-video-palette>`
     - ``"#444444"``
     - hex colour
   * - :ref:`heatmap_vmax_percentile <vis-qlvm-torus-traversal-video-heatmap-vmax-percentile>`
     - ``95``
     - percentile
   * - :ref:`bitrate <vis-qlvm-torus-traversal-video-bitrate>`
     - ``4000``
     - kbit/s

.. _vis-qlvm-torus-traversal-video-fps:

``fps``
   Output frame rate.

   :Options: frames per second
   :Default: ``20``
   :Notes: set from the GUI by the **Video sampling rate (fps)** slider in the *Visualize* window (third column).

.. _vis-qlvm-torus-traversal-video-dpi:

``dpi``
   Raster resolution of each rendered frame.

   :Options: dots per inch
   :Default: ``100``

.. _vis-qlvm-torus-traversal-video-m:

``m``
   Number of nearest-neighbour USVs shown in the concentric rings around each category peak (Part 1).

   :Options: number of USVs
   :Default: ``36``

.. _vis-qlvm-torus-traversal-video-cluster-hold-frames:

``cluster_hold_frames``
   Frames each category peak is held on screen (Part 1).

   :Options: number of frames
   :Default: ``60``

.. _vis-qlvm-torus-traversal-video-peak-traverse-frames:

``peak_traverse_frames``
   Frames spent on each peak-to-peak walk (Part 2).

   :Options: number of frames
   :Default: ``200``

.. _vis-qlvm-torus-traversal-video-boundary-traverse-frames:

``boundary_traverse_frames``
   Frames spent on each boundary-crossing walk (Part 3).

   :Options: number of frames
   :Default: ``200``

.. _vis-qlvm-torus-traversal-video-title-card-frames:

``title_card_frames``
   Frames each of the three title cards is shown.

   :Options: number of frames
   :Default: ``45``

.. _vis-qlvm-torus-traversal-video-samples-per-trace:

``samples_per_trace``
   Number of positions revealed along each peak-to-peak walk (one nearest neighbour each).

   :Options: number of positions
   :Default: ``75``

.. _vis-qlvm-torus-traversal-video-peak-jitter-sigma:

``peak_jitter_sigma``
   Random jitter (torus units) applied to the peak-walk trajectory so repeated paths do not overlap.

   :Options: jitter in torus units
   :Default: ``0.015``

.. _vis-qlvm-torus-traversal-video-boundary-curve-amplitude:

``boundary_curve_amplitude``
   Curvature amplitude of the Part-3 boundary walks.

   :Options: curvature amplitude
   :Default: ``0.18``

.. _vis-qlvm-torus-traversal-video-boundary-positions-per-walk:

``boundary_positions_per_walk``
   Number of positions revealed along each boundary walk.

   :Options: number of positions
   :Default: ``15``

.. _vis-qlvm-torus-traversal-video-boundary-neighbors:

``boundary_neighbors``
   Nearest neighbours shown per boundary position (the Part-3 grid rows).

   :Options: number of neighbours
   :Default: ``5``

.. _vis-qlvm-torus-traversal-video-seed:

``seed``
   RNG (random-number-generator) seed for the walks / jitter (reproducible renders).

   :Options: an integer seed
   :Default: ``0``

.. _vis-qlvm-torus-traversal-video-peaks-only:

``peaks_only``
   Restrict the render to its first part.

   :Options:
      * ``true`` — render Part 1 (category peaks) only
      * ``false`` — render all three parts
   :Default: ``false``

.. _vis-qlvm-torus-traversal-video-spec-cache-size:

``spec_cache_size``
   Number of spectrograms held in the in-memory LRU cache during rendering.

   :Options: number of spectrograms
   :Default: ``2048``

.. _vis-qlvm-torus-traversal-video-apply-mask:

``apply_mask``
   Apply the SAM2 mask to each spectrogram.

   :Options:
      * ``true`` — the spectrograms are shown masked
      * ``false`` — the spectrograms are shown unmasked
   :Default: ``true``

.. _vis-qlvm-torus-traversal-video-accent-color:

``accent_color``
   Hex highlight color for the trail / head marker / cluster outline + dot / current-tile borders.

   :Options: a hex colour string
   :Default: ``"#00FFFF"``

.. _vis-qlvm-torus-traversal-video-palette:

``palette``
   The video's neutral hex colours; neutral so the accent colour stays the only highlight.

   :Options:
      * ``white`` — title-card face and the category contours on the density
      * ``black`` — title-card title and the marker edges
      * ``gray`` — borders of the shown tiles
      * ``light_gray`` — borders of the empty tiles
      * ``subtitle`` — the title-card subtitle
   :Default: ``white``: ``#FFFFFF``, ``black``: ``#000000``, ``gray``: ``#808080``, ``light_gray``: ``#D3D3D3``, ``subtitle``: ``#444444``

.. _vis-qlvm-torus-traversal-video-heatmap-vmax-percentile:

``heatmap_vmax_percentile``
   Percentile of the non-zero category-bundle density that tops the left panel's colour scale.

   :Options: a percentile
   :Default: ``95``
   :Notes: at 95 a few peak pixels do not wash the rest of the map out.

.. _vis-qlvm-torus-traversal-video-bitrate:

``bitrate``
   ffmpeg bitrate of an ``.mp4`` output, in kbit/s.

   :Options: bitrate in kbit/s
   :Default: ``4000``
   :Notes: a ``.gif`` is written with Pillow and ignores it.

Render a USV sequence figure
-----------------------------
``USVSpectrogramPlotter.plot_sequence`` (the ``'sequence'`` mode of ``make_usv_spectrograms``) renders a **per-session**, static two-panel figure of the USVs in a chosen ``[start, start + duration]`` window (seconds). It is wired into the GUI *Visualize* window (third column) under "Render USV sequence figure" and dispatched per session in ``visualize_data.py`` via ``make_usv_spectrograms_bool`` (enabling the GUI toggle sets ``make_usv_spectrograms.mode = 'sequence'``).

To render it, list a session root, set the *Spectrograms directory*, choose the *QLVM map*, select *Render USV sequence figure*, set the audio-sequence window and options, click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_4.png
   :align: center
   :alt: Visualize Step 4

.. raw:: html

   <br>

- **Left** — the precomputed cohort landscape of the QLVM map ``shared_resources.qlvm_map`` (see *Which QLVM map* above), drawn with no ticks/ticklabels. The window's USVs are placed at the map's ``P1`` / ``P2``, colored by emitter (``male_colors[0]`` / ``female_colors[0]`` / ``unassigned_colors[0]``), sized by USV duration, numbered ``1..n`` in time order, and joined by a connecting line whose color runs white → male color along the bout and whose per-segment width tracks the inter-USV silent gap (``start`` of the next minus ``stop`` of the previous; the line takes the short wrap-around route across a torus edge when that is closer). On the regular map the panel draws the QLVM category bundle's gray_r density heatmap and, when ``draw_boundaries`` is on, the bundle's black category boundaries (R-1 … R-k on the unit torus; see *Where the category geometry comes from* above). On a conditional map the bundle does not describe the torus, so the panel is a bare (tick-free) square with a note saying the categories are defined on the regular map. A session whose ``usv_summary`` lacks the map's ``P1`` / ``P2`` raises a clear error.
- **Right** — ONE continuous spectrogram over the same window: the per-USV averaged spectrograms are SAM2-masked (when ``apply_mask``) and stitched at their true times onto a **black** background, so only the USVs are lit and the gaps are black; an optional raw-audio trace can sit on top (``plot_raw_audio``), taken from the channel that is loudest across the window's USVs (the most-frequent per-USV ``peak_amp_ch``; raw waveforms are NOT averaged across mics — the per-mic phase delays would interfere destructively). Each USV can also be marked with a horizontal emitter-colored bar along the top of the spectrogram (``mark_usv_segments``). The left-panel numbers match the time order of the USVs along the right time axis.

The window's vocalizations never include noise: every segment the noise classifier flagged (``noise`` true; ``os_utils.drop_noise_usvs``, the shared noise rule) is dropped before the figure is drawn, and a summary without the ``noise`` column raises (run ``detect-usv-noise`` first). With the sequence setting ``exclude_squeaks`` on (the default) the squeak-bearing segments are left out too — the same rule as ``embedding_thumbnails.exclude_squeaks``: only the pure USVs (``usv`` true and ``squeak`` false) are placed on the map, numbered, stitched into the spectrogram and marked, while pure squeaks, segments holding both a squeak and a USV and unclassed rows are not; a summary without the ``usv`` / ``squeak`` booleans then raises (run ``detect-usv-squeaks`` first).

The GUI section leads with a **Save created figure in format** selector (writes ``make_usv_spectrograms.fig_format``), includes a **Draw embedding boundaries** toggle, and ends with an **Exclude squeaks** toggle (writes ``make_usv_spectrograms.sequence.exclude_squeaks``); the map itself is the shared **QLVM map** selector. The category boundaries exist on the regular map only, so the **Draw embedding boundaries** toggle is greyed out (disabled) whenever a conditional map is selected in the **QLVM map** box and enabled again for the regular map (``QLVM``); the stored ``draw_boundaries`` value is kept, so switching back restores it.

The consolidated store is resolved by convention from the single ``shared_resources.spectrograms_dir`` base (the newest ``spectrograms_*.h5``); the landscape is the category bundle. Being per-session, the figure is written to ``save_dir`` — or, when that is empty, to the ``data_animation_examples`` subdirectory of the session (the same per-session output folder the behavioral videos use) — and, when ``auto_open_figure`` is on AND running in a GUI context, opened in the OS default viewer (headless / batch runs never spawn a viewer; the plotter closes each figure after saving so a per-session run does not accumulate open figures):

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ...
    │   ├── **data_animation_examples**
    │   │   ├── **usv_spectrogram_..._sequence_qlvm_from_0.4s_to_2.4s_20250430_145017.png**
    │   │   ...
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

Settings live in the ``make_usv_spectrograms`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* — the shared top-level keys:

.. code-block:: json

    "make_usv_spectrograms": {
        "save_dir": "",
        "save_fig": true,
        "fig_format": "png",
        "fig_dpi": 300,
        "fig_size": [6, 2],
        "transparent_fig_bg": false,
        "mode": "sequence",
        "channel_of_interest": 11,
        "plot_raw_audio": true,
        "time_window": [0.4, 2.4],
        "freq_limits": [40, 95],
        "usv_amplitude_color": "#808080",
        "nfft": 512,
        "plot_cbar": true,
        "cbar_limits": [-70, 0],
        "apply_mask": true,
        "auto_open_figure": true,
        "stitched_edge_taper_alpha": 0.3,
        "sequence": {
            "draw_boundaries": true,
            "annotate_right": false,
            "mark_usv_segments": true,
            "exclude_squeaks": true,
            "marker_size_range": [35.0, 70.0],
            "line_width_range": [0.5, 3.0],
            "embedding_cmap": "gray_r",
            "embedding_vmax_percentile": 95
        }
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`save_dir <vis-make-usv-spectrograms-save-dir>`
     - ``""``
     - path to a directory, or empty
   * - :ref:`save_fig <vis-make-usv-spectrograms-save-fig>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`fig_format <vis-make-usv-spectrograms-fig-format>`
     - ``"png"``
     - ``png``, ``pdf``, ``svg``, ``jpg``
   * - :ref:`fig_dpi <vis-make-usv-spectrograms-fig-dpi>`
     - ``300``
     - dots per inch
   * - :ref:`fig_size <vis-make-usv-spectrograms-fig-size>`
     - ``[6, 2]``
     - ``[width, height]`` in inches
   * - :ref:`transparent_fig_bg <vis-make-usv-spectrograms-transparent-fig-bg>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`mode <vis-make-usv-spectrograms-mode>`
     - ``"sequence"``
     - pinned to ``sequence`` here; ``single``, ``all``, ``stitched`` in the notebook
   * - :ref:`channel_of_interest <vis-make-usv-spectrograms-channel-of-interest>`
     - ``11``
     - microphone channel index
   * - :ref:`plot_raw_audio <vis-make-usv-spectrograms-plot-raw-audio>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_amplitude_color <vis-make-usv-spectrograms-usv-amplitude-color>`
     - ``"#808080"``
     - hex colour
   * - :ref:`time_window <vis-make-usv-spectrograms-time-window>`
     - ``[0.4, 2.4]``
     - ``[start, end]`` in seconds
   * - :ref:`freq_limits <vis-make-usv-spectrograms-freq-limits>`
     - ``[40, 95]``
     - ``[low, high]`` in kHz
   * - :ref:`nfft <vis-make-usv-spectrograms-nfft>`
     - ``512``
     - FFT window size
   * - :ref:`plot_cbar <vis-make-usv-spectrograms-plot-cbar>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`cbar_limits <vis-make-usv-spectrograms-cbar-limits>`
     - ``[-70, 0]``
     - ``[vmin, vmax]`` in dB
   * - :ref:`apply_mask <vis-make-usv-spectrograms-apply-mask>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`auto_open_figure <vis-make-usv-spectrograms-auto-open-figure>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`stitched_edge_taper_alpha <vis-make-usv-spectrograms-stitched-edge-taper-alpha>`
     - ``0.3``
     - tapered fraction of the USV
   * - :ref:`sequence.draw_boundaries <vis-make-usv-spectrograms-sequence-draw-boundaries>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`sequence.annotate_right <vis-make-usv-spectrograms-sequence-annotate-right>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`sequence.mark_usv_segments <vis-make-usv-spectrograms-sequence-mark-usv-segments>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`sequence.exclude_squeaks <vis-make-usv-spectrograms-sequence-exclude-squeaks>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`sequence.marker_size_range <vis-make-usv-spectrograms-sequence-marker-size-range>`
     - ``[35.0, 70.0]``
     - ``[min, max]`` in points²
   * - :ref:`sequence.line_width_range <vis-make-usv-spectrograms-sequence-line-width-range>`
     - ``[0.5, 3.0]``
     - ``[min, max]`` in points
   * - :ref:`sequence.embedding_cmap <vis-make-usv-spectrograms-sequence-embedding-cmap>`
     - ``"gray_r"``
     - matplotlib colormap name
   * - :ref:`sequence.embedding_vmax_percentile <vis-make-usv-spectrograms-sequence-embedding-vmax-percentile>`
     - ``95``
     - percentile

.. _vis-make-usv-spectrograms-save-dir:

``save_dir``
   Output directory.

   :Options: path to a directory, or empty
   :Default: ``""`` (empty)
   :Notes: empty routes the figure to the session's ``data_animation_examples`` folder.

.. _vis-make-usv-spectrograms-save-fig:

``save_fig``
   Whether to write the figure to disk.

   :Options:
      * ``true`` — the figure is written
      * ``false`` — the figure is not written
   :Default: ``true``

.. _vis-make-usv-spectrograms-fig-format:

``fig_format``
   Output file format.

   :Options:
      * ``png``
      * ``pdf``
      * ``svg``
      * ``jpg``
   :Default: ``"png"``
   :Notes: set from the GUI by the **Save created figure in format** selector.

.. _vis-make-usv-spectrograms-fig-dpi:

``fig_dpi``
   Raster resolution.

   :Options: dots per inch
   :Default: ``300``

.. _vis-make-usv-spectrograms-fig-size:

``fig_size``
   Figure size in inches (``[width, height]``).

   :Options: two numbers, ``[width, height]`` in inches
   :Default: ``[6, 2]``

.. _vis-make-usv-spectrograms-transparent-fig-bg:

``transparent_fig_bg``
   Save the figure on a transparent background.

   :Options:
      * ``true`` — transparent background
      * ``false`` — opaque background
   :Default: ``false``

.. _vis-make-usv-spectrograms-mode:

``mode``
   The figure type the ``make_usv_spectrograms`` class renders.

   :Options:
      * ``sequence`` — this figure; for it the key is **pinned to** ``sequence``
      * ``single`` / ``all`` — raw per-channel spectrograms (notebook only)
      * ``stitched`` — the averaged session-timeline spectrogram (notebook only)
   :Default: ``"sequence"``
   :Notes: the GUI toggle overwrites it, so it is not a choice here. It becomes a free selection only when the plotter is driven from ``usv_general_analyses.ipynb`` (see :doc:`Notebooks`), where the other values render *different* figures.

.. _vis-make-usv-spectrograms-channel-of-interest:

``channel_of_interest``
   Microphone channel for the ``single`` mode.

   :Options: a microphone channel index
   :Default: ``11``
   :Notes: unused by ``sequence``, which auto-selects the loudest channel over the window.

.. _vis-make-usv-spectrograms-plot-raw-audio:

``plot_raw_audio``
   Overlay the raw-audio amplitude trace on the stitched spectrogram.

   :Options:
      * ``true`` — the trace is drawn
      * ``false`` — no trace
   :Default: ``true``

.. _vis-make-usv-spectrograms-usv-amplitude-color:

``usv_amplitude_color``
   Hex color of that raw-audio amplitude trace.

   :Options: a hex colour string
   :Default: ``"#808080"``

.. _vis-make-usv-spectrograms-time-window:

``time_window``
   ``[start, end]`` seconds of the window.

   :Options: two numbers, ``[start, end]`` in seconds
   :Default: ``[0.4, 2.4]``
   :Notes: the GUI presents it as a start + duration pair and writes it back.

.. _vis-make-usv-spectrograms-freq-limits:

``freq_limits``
   ``[low, high]`` kHz frequency axis of the spectrogram.

   :Options: two numbers, ``[low, high]`` in kHz
   :Default: ``[40, 95]``

.. _vis-make-usv-spectrograms-nfft:

``nfft``
   FFT window size for the spectrogram.

   :Options: an FFT window size in samples
   :Default: ``512``

.. _vis-make-usv-spectrograms-plot-cbar:

``plot_cbar``
   Whether to draw the colorbar.

   :Options:
      * ``true`` — the colorbar is drawn
      * ``false`` — no colorbar
   :Default: ``true``

.. _vis-make-usv-spectrograms-cbar-limits:

``cbar_limits``
   ``[vmin, vmax]`` colorbar (power) limits in dB.

   :Options: two numbers, ``[vmin, vmax]`` in dB
   :Default: ``[-70, 0]``

.. _vis-make-usv-spectrograms-apply-mask:

``apply_mask``
   Apply the SAM2 mask to each per-USV spectrogram before stitching.

   :Options:
      * ``true`` — masked before stitching
      * ``false`` — stitched unmasked
   :Default: ``true``

.. _vis-make-usv-spectrograms-auto-open-figure:

``auto_open_figure``
   Open the saved figure in the OS viewer when running in a GUI context.

   :Options:
      * ``true`` — opened in a GUI context
      * ``false`` — never opened
   :Default: ``true``

.. _vis-make-usv-spectrograms-stitched-edge-taper-alpha:

``stitched_edge_taper_alpha``
   Shape of the raised-cosine (Tukey) taper applied along each per-USV spectrogram's time axis before it is stitched onto the zero canvas (``stitched`` mode).

   :Options: the tapered fraction of the USV, start and end combined
   :Default: ``0.3`` (fades the first and last 15%)
   :Notes: each USV fades into the black gaps instead of stepping from a bright edge column.

plus the ``sequence`` sub-dict:

.. _vis-make-usv-spectrograms-sequence-draw-boundaries:

``sequence.draw_boundaries``
   Overlay the category bundle's boundaries on the left panel (regular map only).

   :Options:
      * ``true`` — the boundaries are drawn (on the regular map)
      * ``false`` — no boundaries
   :Default: ``true``
   :Notes: set from the GUI by the **Draw embedding boundaries** toggle.

.. _vis-make-usv-spectrograms-sequence-annotate-right:

``sequence.annotate_right``
   Annotate the stitched (right) spectrogram.

   :Options:
      * ``true`` — annotated
      * ``false`` — not annotated
   :Default: ``false``

.. _vis-make-usv-spectrograms-sequence-mark-usv-segments:

``sequence.mark_usv_segments``
   Draw an emitter-colored bar along the top of the spectrogram for each USV.

   :Options:
      * ``true`` — the bars are drawn
      * ``false`` — no bars
   :Default: ``true``

.. _vis-make-usv-spectrograms-sequence-exclude-squeaks:

``sequence.exclude_squeaks``
   Keep only the pure USVs (``usv`` true and ``squeak`` false) in the sequence, leaving out pure squeaks, segments holding both and unclassed rows.

   :Options:
      * ``true`` — pure USVs only
      * ``false`` — the squeak-bearing segments are not left out
   :Default: ``true``
   :Notes: same meaning as ``embedding_thumbnails.exclude_squeaks``. Noise segments are dropped regardless of this setting. Set from the GUI by the **Exclude squeaks** toggle.

.. _vis-make-usv-spectrograms-sequence-marker-size-range:

``sequence.marker_size_range``
   ``[min, max]`` marker area (points²) of the left-panel USV points, scaled linearly with USV duration from the shortest to the longest USV of the window.

   :Options: two numbers, ``[min, max]`` in points²
   :Default: ``[35.0, 70.0]``
   :Notes: modest, so longer USVs read bigger without dominating; a window of equal durations takes the midpoint.

.. _vis-make-usv-spectrograms-sequence-line-width-range:

``sequence.line_width_range``
   ``[min, max]`` width (points) of the connecting line's segments, scaled with the silent gap between consecutive USVs up to the Tukey upper fence of the gaps.

   :Options: two numbers, ``[min, max]`` in points
   :Default: ``[0.5, 3.0]``
   :Notes: a longer silence draws a thicker segment.

.. _vis-make-usv-spectrograms-sequence-embedding-cmap:

``sequence.embedding_cmap``
   Matplotlib colormap of the regular map's density heatmap behind the points.

   :Options: a matplotlib colormap name
   :Default: ``"gray_r"``
   :Notes: ``gray_r`` runs white to black, so the emitter colours stay legible on it.

.. _vis-make-usv-spectrograms-sequence-embedding-vmax-percentile:

``sequence.embedding_vmax_percentile``
   Percentile of the non-zero density pixels used as the heatmap's upper colour limit.

   :Options: a percentile
   :Default: ``95``
   :Notes: the few densest pixels saturate instead of washing out the rest of the map.

Render embedding thumbnails
---------------------------
The ``usv_playpen.visualizations.make_usv_spectrograms`` module's remaining cohort-level helper, ``plot_embedding_with_category_thumbnails``, is GUI-exposed (its pooled summary helpers — ``plot_usv_property_histograms``, ``plot_session_type_usv_counts``, ``plot_session_usv_timeline`` — are notebook-driven; see :doc:`Notebooks`):

- ``plot_embedding_with_category_thumbnails`` — a two-panel figure pairing a QLVM map scatter (the torus of ``shared_resources.qlvm_map``) — colored by the USVs' ``qlvm_category`` and, on the regular map, overlaid with the category bundle's boundaries — against a per-category grid of spectrogram thumbnails sampled from the consolidated SAM2 + spectrogram store. Unlike the helpers above it is **cohort-level** and exposed in the GUI: enabling *Render embedding thumbnails* in the *Visualize* window (third column, with **exclude squeaks** (``embedding_thumbnails.exclude_squeaks``), **thumbnails per category**, **thumbnail layout**, **draw cluster boundaries**, **apply SAM2 mask** and **per-cluster sampling** selectors) pools every cohort session list under ``shared_resources.input_files_directory`` (playback lists excluded, as in the embedding explorer) and resolves the store from ``shared_resources.spectrograms_dir``, then runs ONCE (the same run-once dispatch as the QLVM torus video, via ``render_embedding_thumbnails_for_cohort``). Its layout / sampling knobs all live in the ``embedding_thumbnails`` settings block (documented below); the figure DPI and the sampling seed are taken from the general ``figures`` block (``dpi`` / ``seed``); the category boundaries and centers (for cluster-ID labels / spiral centers) are the category bundle's ``label_grid`` and label positions (row ``i`` = category ``i + 1``; a pooled ``qlvm_category`` outside ``1..k`` raises, since the summaries and the bundle would be different partitions), on the regular map only; the categories themselves are the summaries' ``qlvm_category`` on every map, so a conditional ``qlvm_map`` colours and groups its USVs by ``qlvm_category`` with no boundaries, the spiral sampler walks without a boundary filter, and the centres are the per-category circular means of that map's USVs (a message says so); every QLVM map is a periodic unit torus, so centres, spiral radii and pick distances use wrapped displacements and a category straddling a seam is centred at the seam; and the cohort scatter is read from the precomputed pooled-embeddings cache ``<spectrograms_dir>/embeddings/pooled_embeddings_production.parquet`` (``os_utils.POOLED_EMBEDDINGS_CACHE_NAME``; one parquet holding every QLVM map's coordinates and the ``qlvm_category`` labels, fingerprinted against the summaries it was pooled from and rebuilt when they change) — built once on a fast mount via ``build_pooled_embeddings_df`` so the figure does not re-read every session's ``usv_summary.csv``.

To render it, set the *Spectrograms directory*, select *Render embedding thumbnails* in the *Visualize* window, choose the squeak filter / layout / sampling options (the map is the shared **QLVM map** selector), click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_5.png
   :align: center
   :alt: Visualize Step 5

.. raw:: html

   <br>

Being cohort-level, the figure is written to the project-wide ``figures.save_directory`` with a name built from the QLVM map alone, ``embedding_thumbnails_<qlvm_map>`` (``qlvm_category`` is the one category column, so the name carries no label-column part; plus a ``_YYYYMMDD_HHMMSS`` stamp when ``figures.timestamp_in_name`` is set):

.. parsed-literal::

    ├── ...  (the ``figures.save_directory`` cohort output folder)
    │   ├── **embedding_thumbnails_qlvm_20250430_145017.png**
    │   ...

The layout / sampling knobs live in the ``embedding_thumbnails`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (the figure DPI and the sampling seed come from the general ``figures`` block).

.. code-block:: json

    "embedding_thumbnails": {
        "exclude_squeaks": true,
        "n_samples_per_category": 8,
        "tile_orientation": "vertical",
        "apply_mask": true,
        "mask_excluded_categories": [],
        "category_colors": null,
        "sampling_method": "spiral",
        "draw_cluster_boundaries": true,
        "draw_spiral_overlay": false,
        "spiral_show_only_for": null,
        "spiral_color": "#000000",
        "spiral_linewidth": 1.0,
        "spiral_radius_scale": 0.1,
        "spiral_radius_abs": 0.03,
        "spiral_n_turns": 3,
        "spiral_random_phase": true,
        "annotate_picks_on_scatter": false,
        "pick_number_fontsize": 9,
        "annotate_cluster_ids": true,
        "cluster_id_fontsize": 20,
        "thumbnail_hspace": 0.03,
        "thumbnail_wspace": 0.04,
        "unstretched_specs": true,
        "scatter_max_points": 200000,
        "scatter_point_size": 4.0,
        "scatter_point_alpha": 0.5,
        "continuous_color_percentiles": [2, 98],
        "fig_size": [16, 12]
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`exclude_squeaks <vis-embedding-thumbnails-exclude-squeaks>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`n_samples_per_category <vis-embedding-thumbnails-n-samples-per-category>`
     - ``8``
     - number of thumbnails
   * - :ref:`tile_orientation <vis-embedding-thumbnails-tile-orientation>`
     - ``"vertical"``
     - ``vertical``, ``horizontal``
   * - :ref:`apply_mask <vis-embedding-thumbnails-apply-mask>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`mask_excluded_categories <vis-embedding-thumbnails-mask-excluded-categories>`
     - ``[]``
     - list of category numbers
   * - :ref:`category_colors <vis-embedding-thumbnails-category-colors>`
     - ``null``
     - ``null`` (auto), or ``{"1": "#RRGGBB", ...}``
   * - :ref:`scatter_max_points <vis-embedding-thumbnails-scatter-max-points>`
     - ``200000``
     - number of points
   * - :ref:`scatter_point_size <vis-embedding-thumbnails-scatter-point-size>`
     - ``4.0``
     - marker area in points²
   * - :ref:`scatter_point_alpha <vis-embedding-thumbnails-scatter-point-alpha>`
     - ``0.5``
     - opacity
   * - :ref:`continuous_color_percentiles <vis-embedding-thumbnails-continuous-color-percentiles>`
     - ``[2, 98]``
     - ``[lower, upper]`` percentiles
   * - :ref:`sampling_method <vis-embedding-thumbnails-sampling-method>`
     - ``"spiral"``
     - ``random``, ``nearest``, ``spread``, ``grid``, ``spiral``
   * - :ref:`draw_cluster_boundaries <vis-embedding-thumbnails-draw-cluster-boundaries>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`draw_spiral_overlay <vis-embedding-thumbnails-draw-spiral-overlay>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`spiral_show_only_for <vis-embedding-thumbnails-spiral-show-only-for>`
     - ``null``
     - ``null`` (all), or one category
   * - :ref:`spiral_color <vis-embedding-thumbnails-spiral-color>`
     - ``"#000000"``
     - hex colour
   * - :ref:`spiral_linewidth <vis-embedding-thumbnails-spiral-linewidth>`
     - ``1.0``
     - line width
   * - :ref:`spiral_radius_scale <vis-embedding-thumbnails-spiral-radius>`
     - ``0.1``
     - fraction of the farthest USV
   * - :ref:`spiral_radius_abs <vis-embedding-thumbnails-spiral-radius>`
     - ``0.03``
     - ``null``, or torus units
   * - :ref:`spiral_n_turns <vis-embedding-thumbnails-spiral-n-turns>`
     - ``3``
     - number of turns
   * - :ref:`spiral_random_phase <vis-embedding-thumbnails-spiral-random-phase>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`annotate_picks_on_scatter <vis-embedding-thumbnails-annotate-picks-on-scatter>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`pick_number_fontsize <vis-embedding-thumbnails-pick-number-fontsize>`
     - ``9``
     - font size
   * - :ref:`annotate_cluster_ids <vis-embedding-thumbnails-annotate-cluster-ids>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`cluster_id_fontsize <vis-embedding-thumbnails-cluster-id-fontsize>`
     - ``20``
     - font size
   * - :ref:`thumbnail_hspace <vis-embedding-thumbnails-thumbnail-spacing>`
     - ``0.03``
     - spacing
   * - :ref:`thumbnail_wspace <vis-embedding-thumbnails-thumbnail-spacing>`
     - ``0.04``
     - spacing
   * - :ref:`unstretched_specs <vis-embedding-thumbnails-unstretched-specs>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`fig_size <vis-embedding-thumbnails-fig-size>`
     - ``[16, 12]``
     - ``[width, height]`` in inches

Core keys:

.. _vis-embedding-thumbnails-exclude-squeaks:

``exclude_squeaks``
   Keep only the segments ``detect-usv-squeaks`` classed as pure USVs (``usv`` true and ``squeak`` false) in the scatter, the four small maps and the thumbnails, leaving out pure squeaks, segments holding both a squeak and a USV (``usv`` and ``squeak`` both true) and unclassed rows.

   :Options:
      * ``true`` — pure USVs only
      * ``false`` — squeak-bearing and unclassed segments are not left out
   :Default: ``true``
   :Notes: The USV QLVM models were trained on USVs only, so squeak-bearing segments sit wherever the decoder places them. A pooled embeddings table without the ``usv`` / ``squeak`` booleans (built from summaries ``detect-usv-squeaks`` has not classified) raises; a cache written before the booleans existed is rebuilt automatically. Set from the GUI by the **exclude squeaks** selector.

.. _vis-embedding-thumbnails-n-samples-per-category:

``n_samples_per_category``
   Number of thumbnail spectrograms sampled per category.

   :Options: number of thumbnails
   :Default: ``8``
   :Notes: set from the GUI by the **thumbnails per category** selector.

.. _vis-embedding-thumbnails-tile-orientation:

``tile_orientation``
   Thumbnail grid orientation.

   :Options:
      * ``vertical``
      * ``horizontal``
   :Default: ``"vertical"``
   :Notes: set from the GUI by the **thumbnail layout** selector.

.. _vis-embedding-thumbnails-apply-mask:

``apply_mask``
   Apply the SAM2 mask to each thumbnail spectrogram.

   :Options:
      * ``true`` — masked thumbnails (except ``mask_excluded_categories``)
      * ``false`` — unmasked thumbnails
   :Default: ``true``
   :Notes: set from the GUI by the **apply SAM2 mask** selector.

.. _vis-embedding-thumbnails-mask-excluded-categories:

``mask_excluded_categories``
   Categories whose thumbnails are shown WITHOUT the SAM2 mask even when ``apply_mask`` is on (for vocal types SAM2 masks poorly).

   :Options: a list of categories
   :Default: ``[]``
   :Notes: every category stays in the grid.

.. _vis-embedding-thumbnails-category-colors:

``category_colors``
   Optional explicit per-category color map, ``{"1": "#RRGGBB", ...}`` keyed by the ``qlvm_category`` number.

   :Options:
      * ``null`` — auto
      * ``{"1": "#RRGGBB", ...}`` — explicit colours
   :Default: ``null``

.. _vis-embedding-thumbnails-scatter-max-points:

``scatter_max_points``
   Downsampling cap for the embedding scatter.

   :Options: number of points
   :Default: ``200000``

.. _vis-embedding-thumbnails-scatter-point-size:

``scatter_point_size``
   Marker area of each scatter point (points²).

   :Options: a marker area in points²
   :Default: ``4.0``
   :Notes: the four small auxiliary maps use a quarter of it, at least ``0.5``; small because the cohort scatter holds up to ``scatter_max_points`` USVs.

.. _vis-embedding-thumbnails-scatter-point-alpha:

``scatter_point_alpha``
   Opacity of the scatter points.

   :Options: an opacity
   :Default: ``0.5``
   :Notes: so the dense cores of the map read darker than its fringes.

.. _vis-embedding-thumbnails-continuous-color-percentiles:

``continuous_color_percentiles``
   Lower / upper percentiles of the finite values that set the colour limits of the duration and bandwidth auxiliary maps.

   :Options: two percentiles, ``[lower, upper]``
   :Default: ``[2, 98]``
   :Notes: so a few extreme USVs do not compress the colour range of the rest.

Sampling / boundaries:

.. _vis-embedding-thumbnails-sampling-method:

``sampling_method``
   How thumbnails are drawn per category (e.g. ``spiral``).

   :Options:
      * ``random``
      * ``nearest``
      * ``spread``
      * ``grid``
      * ``spiral``
   :Default: ``"spiral"``
   :Notes: set from the GUI by the **per-cluster sampling** selector, which offers these five; any other value raises ``ValueError``.

.. _vis-embedding-thumbnails-draw-cluster-boundaries:

``draw_cluster_boundaries``
   Overlay the category bundle's boundaries on the scatter (regular map only).

   :Options:
      * ``true`` — the boundaries are drawn (on the regular map)
      * ``false`` — no boundaries
   :Default: ``true``
   :Notes: set from the GUI by the **draw cluster boundaries** selector.

Spiral overlay:

.. _vis-embedding-thumbnails-draw-spiral-overlay:

``draw_spiral_overlay``
   Draw the sampling-spiral overlay on the scatter.

   :Options:
      * ``true`` — the overlay is drawn
      * ``false`` — no overlay
   :Default: ``false``

.. _vis-embedding-thumbnails-spiral-show-only-for:

``spiral_show_only_for``
   Restrict the overlay to one category.

   :Options:
      * ``null`` — all
      * a category — the overlay is drawn for that category only
   :Default: ``null``

.. _vis-embedding-thumbnails-spiral-color:

``spiral_color``
   Hex color of the spiral.

   :Options: a hex colour string
   :Default: ``"#000000"``

.. _vis-embedding-thumbnails-spiral-linewidth:

``spiral_linewidth``
   Spiral line width.

   :Options: a line width
   :Default: ``1.0``

.. _vis-embedding-thumbnails-spiral-radius:

``spiral_radius_scale`` / ``spiral_radius_abs``
   Spiral radius — relative to each cluster's farthest USV, or (``spiral_radius_abs``, when not ``null``) an absolute radius in torus units shared by every cluster.

   :Options:
      * ``spiral_radius_scale`` — the radius relative to each cluster's farthest USV
      * ``spiral_radius_abs`` — ``null`` to use ``spiral_radius_scale``, or an absolute radius in torus units shared by every cluster
   :Default: ``0.1`` and ``0.03`` (shipped ``0.03``, so the thumbnails stay close to each cluster's peak)

.. _vis-embedding-thumbnails-spiral-n-turns:

``spiral_n_turns``
   Number of spiral turns.

   :Options: number of turns
   :Default: ``3``

.. _vis-embedding-thumbnails-spiral-random-phase:

``spiral_random_phase``
   Randomize the spiral's starting angle.

   :Options:
      * ``true`` — random starting angle
      * ``false`` — fixed starting angle
   :Default: ``true``

Annotations / layout:

.. _vis-embedding-thumbnails-annotate-picks-on-scatter:

``annotate_picks_on_scatter``
   Number each sampled pick on the scatter.

   :Options:
      * ``true`` — picks are numbered
      * ``false`` — no pick numbers
   :Default: ``false``

.. _vis-embedding-thumbnails-pick-number-fontsize:

``pick_number_fontsize``
   Font size of those pick numbers.

   :Options: a font size
   :Default: ``9``

.. _vis-embedding-thumbnails-annotate-cluster-ids:

``annotate_cluster_ids``
   Label each cluster with its id.

   :Options:
      * ``true`` — clusters are labelled
      * ``false`` — no cluster labels
   :Default: ``true``

.. _vis-embedding-thumbnails-cluster-id-fontsize:

``cluster_id_fontsize``
   Font size of the cluster-id labels.

   :Options: a font size
   :Default: ``20``

.. _vis-embedding-thumbnails-thumbnail-spacing:

``thumbnail_hspace`` / ``thumbnail_wspace``
   Vertical / horizontal spacing between thumbnails.

   :Options:
      * ``thumbnail_hspace`` — vertical spacing
      * ``thumbnail_wspace`` — horizontal spacing
   :Default: ``0.03`` and ``0.04``

.. _vis-embedding-thumbnails-unstretched-specs:

``unstretched_specs``
   How each thumbnail is fitted into its slot.

   :Options:
      * ``true`` — zero-pad every thumbnail to the longest pick's duration (USV centred, equal borders) so all thumbnails share one time scale
      * ``false`` — stretch each USV to fill its slot
   :Default: ``true``

.. _vis-embedding-thumbnails-fig-size:

``fig_size``
   Figure size in inches (``[width, height]``).

   :Options: two numbers, ``[width, height]`` in inches
   :Default: ``[16, 12]``

The three notebook-driven pooled summary helpers (``plot_usv_property_histograms``, ``plot_session_type_usv_counts``, ``plot_session_usv_timeline``; see :doc:`Notebooks`) take the whole settings dict as ``visualizations_parameter_dict``: their DPI is ``figures.dpi``, their sex / unassigned colours are ``male_colors[0]`` / ``female_colors[0]`` / ``unassigned_colors[0]``, and the rest lives in the ``usv_property_figures`` block:

.. code-block:: json

    "usv_property_figures": {
        "histograms": {
            "fig_size": [15.0, 3.0],
            "n_bins": 36,
            "face_color": "#808080",
            "edge_color": "#000000",
            "bandwidth_bimodal_split_khz": 30.0,
            "bandwidth_peak_smooth_sigma_bins": 3.0
        },
        "session_type_counts": {
            "fig_size": [6.0, 3.0],
            "male_female_color": "#DFB8B3"
        },
        "timeline": {
            "fig_size": [7.5, 1.6],
            "rectangle_height": 2.5,
            "rectangle_linewidth": 1.0
        }
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`histograms.fig_size <vis-usv-property-figures-histograms-fig-size>`
     - ``[15.0, 3.0]``
     - ``[width, height]`` in inches
   * - :ref:`histograms.n_bins <vis-usv-property-figures-histograms-n-bins>`
     - ``36``
     - number of bins
   * - :ref:`histograms.face_color <vis-usv-property-figures-histograms-colors>`
     - ``"#808080"``
     - hex colour
   * - :ref:`histograms.edge_color <vis-usv-property-figures-histograms-colors>`
     - ``"#000000"``
     - hex colour
   * - :ref:`histograms.bandwidth_bimodal_split_khz <vis-usv-property-figures-histograms-bandwidth-bimodal-split-khz>`
     - ``30.0``
     - kHz
   * - :ref:`histograms.bandwidth_peak_smooth_sigma_bins <vis-usv-property-figures-histograms-bandwidth-peak-smooth-sigma-bins>`
     - ``3.0``
     - bins
   * - :ref:`session_type_counts.fig_size <vis-usv-property-figures-session-type-counts-fig-size>`
     - ``[6.0, 3.0]``
     - ``[width, height]`` in inches
   * - :ref:`session_type_counts.male_female_color <vis-usv-property-figures-session-type-counts-male-female-color>`
     - ``"#DFB8B3"``
     - hex colour
   * - :ref:`timeline.fig_size <vis-usv-property-figures-timeline-fig-size>`
     - ``[7.5, 1.6]``
     - ``[width, height]`` in inches
   * - :ref:`timeline.rectangle_height <vis-usv-property-figures-timeline-rectangle-height>`
     - ``2.5``
     - data units
   * - :ref:`timeline.rectangle_linewidth <vis-usv-property-figures-timeline-rectangle-linewidth>`
     - ``1.0``
     - points

.. _vis-usv-property-figures-histograms-fig-size:

``histograms.fig_size``
   Figure size in inches of the five-panel property-histogram row.

   :Options: two numbers, ``[width, height]`` in inches
   :Default: ``[15.0, 3.0]`` (3 in per panel)

.. _vis-usv-property-figures-histograms-n-bins:

``histograms.n_bins``
   Linearly spaced bins per panel over the property's FeatureZoo range.

   :Options: number of bins
   :Default: ``36``

.. _vis-usv-property-figures-histograms-colors:

``histograms.face_color`` / ``histograms.edge_color``
   Histogram fill and outline.

   :Options:
      * ``histograms.face_color`` — the fill, a hex colour string
      * ``histograms.edge_color`` — the outline, a hex colour string
   :Default: ``#808080`` grey fill, ``#000000`` outline

.. _vis-usv-property-figures-histograms-bandwidth-bimodal-split-khz:

``histograms.bandwidth_bimodal_split_khz``
   The frequency bandwidth (kHz) splitting the bimodal ``freq_bandwidth_hz`` distribution; that panel marks the histogram peak on each side of it instead of the mean.

   :Options: a bandwidth in kHz
   :Default: ``30.0`` kHz
   :Notes: between the narrowband and broadband modes.

.. _vis-usv-property-figures-histograms-bandwidth-peak-smooth-sigma-bins:

``histograms.bandwidth_peak_smooth_sigma_bins``
   Gaussian smoothing (in bins, not kHz) applied before those two peaks are located.

   :Options: a smoothing width in bins
   :Default: ``3.0``
   :Notes: just enough to suppress sample-noise peaks near the split.

.. _vis-usv-property-figures-session-type-counts-fig-size:

``session_type_counts.fig_size``
   Figure size in inches of the session-type bar chart.

   :Options: two numbers, ``[width, height]`` in inches
   :Default: ``[6.0, 3.0]``

.. _vis-usv-property-figures-session-type-counts-male-female-color:

``session_type_counts.male_female_color``
   Colour of the male-female bar.

   :Options: a hex colour string
   :Default: ``#DFB8B3`` (the 50/50 blend of the male and female colours lightened 35% toward white)
   :Notes: the lone-male and female-female bars take ``male_colors[0]`` / ``female_colors[0]``.

.. _vis-usv-property-figures-timeline-fig-size:

``timeline.fig_size``
   Figure size in inches of the per-session USV timeline.

   :Options: two numbers, ``[width, height]`` in inches
   :Default: ``[7.5, 1.6]`` (a wide, short strip)

.. _vis-usv-property-figures-timeline-rectangle-height:

``timeline.rectangle_height``
   Vertical extent of each USV rectangle in data units.

   :Options: a height in data units
   :Default: ``2.5``
   :Notes: the y-axis is hidden, so it only sets the strip's visual thickness.

.. _vis-usv-property-figures-timeline-rectangle-linewidth:

``timeline.rectangle_linewidth``
   Outline width of each USV rectangle in points.

   :Options: a width in points
   :Default: ``1.0``

Render vocal-pose figures
-------------------------
The ``usv_playpen.visualizations.make_vocal_pose_figures`` module renders one session's **vocal-pose still** and the matching **video**: what the animals were doing while the male sang. It is **notebook-driven** (the ``usv_general_analyses.ipynb`` *Vocal-pose figures* cells, see :doc:`Notebooks`), not exposed in the GUI, so it has no ``visualize_booleans`` switch; it reads the session's 3D tracks (``*_points3d_translated_rotated_metric.h5``), ``*_usv_summary.csv`` and ``audio/cropped_to_video`` wavs, takes the animal colours from the ``male_colors`` / ``female_colors`` / ``unassigned_colors`` palette and the arena from ``make_behavioral_videos.arena_directory``.

**The still** (``VocalPoseStillMaker.make_vocal_pose_still``) stacks two panels. On top is the spectrogram of one microphone over the window ending at ``window.end_time`` (``"peak"`` picks the microphone most of the window's vocalizations peak on), every time column coloured by the emitter of the vocalization covering it — white to the male's colour, white to the female's, white to the unassigned colour outside any listed vocalization — with each emitter's USVs and squeaks scaled to their own loudest sound (a squeak is tens of dB louder than a USV) and a shared floor so silence stays white; the male ramp continues past his colour to a darker shade of the same hue (``male_deep``) so narrowband USVs stand out on white. Beneath it a time bar whose strength follows the fade of the pose trails. Below are both animals at the frame, drawn with the shared mouse renderer (``plot_mouse_data``) in one colour each with see-through bodies, over one silhouette per camera frame of their preceding movement that fades with age and ends after ``window.trail_end_seconds``; the skeleton's line widths scale with the panel's zoom so the animals look the same in every figure. The camera stands on the male's side of the pair when ``camera.azimuth`` is ``"male_side"`` (it looks along the line from the female's body centre to the male's, so he is nearest the viewer); a number read off the view picker fixes it instead. Optional: a scale mark of three perpendicular arms of ``scale_mark.length_cm`` beside the male, the arena floor and nearest wall as grey surfaces with a black junction line (``surfaces``), a two-swatch colour key (``layout.color_key``) and, with ``layout.align_pose``, the whole tracking panel moved left so its right edge sits under the spectrogram's. To the right, with ``traces.enabled``, a third panel shows what the pair's bodies were doing around the frame: one small strip per behavioural feature (``traces.egocentric`` for each animal, ``traces.dyadic`` per side, read from the session's ``*_behavioral_features.csv``) in two columns, male left and female right, each trace spanning ``traces.span_seconds`` on both sides of the frame, drawn in the animal's colour (``social_colors`` for the dyadic rows) and fading to white with ``(1 - |t| / span) ** fade_power`` while thinning from ``line_width_centre`` to ``line_width_end``; both strips of a row share one y-range whose extremes stand in the gap between the columns, and the row's name (``traces.labels``) sits at the left. The PNG is cropped to its content; ``output.save_svg`` also writes an SVG with live text (Helvetica Light), vector current poses and traces, and the trail, spectrogram and bar as embedded images.

**The video** (``VocalPoseVideoMaker.make_vocal_pose_video``) plays the ``video.duration_seconds`` before the same frame at ``video.speed`` (0.5 = half speed, every camera frame kept) with a ``video.trail.trail_seconds`` fading trail behind each animal, the camera fixed at the still's view, following the pair (``video.follow_seconds`` ≥ 0 centres the smoothed middle of their outline in every frame) or turning at ``video.degrees_per_second`` into the still's view on the last frame, and a scrolling spectrogram with the present fixed at its centre, ``reach_seconds`` visible on each side and sharp only within ``sharp_seconds`` — as an inset in a corner no animal ever enters (``layout`` ``"inset"``; the lower right when free, slid left by ``inset_shift`` of the free room) or a strip below the animals (``"below"``). Frames render in ``workers`` processes (0 = all cores but one) and are encoded by ffmpeg.

**Choosing a window** — ``find_vocal_pose_windows`` ranks the session's windows of ``candidates.window_seconds`` in which every vocalization is a non-noise, non-squeak USV of the male, no vocalization is cut by an edge, no noise detection falls inside and the animals stay within ``candidates.max_gap_cm`` (the smallest distance between any body keypoints of the two, tails excluded) on every frame; the score sums the z-scores of the vocal fraction, the duration-weighted bandwidth and the mean vocalization duration. ``plot_vocal_pose_window_candidates`` stacks the spectrograms of the best ``n_shown`` non-overlapping windows. **Choosing a camera** — ``vocal_pose_view_picker_html`` writes a page on which the frame turns by dragging (or two sliders), with presets for the male-side, heads-down, tails-down and top views, reporting the azimuth and elevation in the figure's convention.

Both outputs land in ``figures.save_directory`` as ``vocal_pose_<session>_<start>-<end>s`` (plus a ``_YYYYMMDD_HHMMSS`` stamp when ``figures.timestamp_in_name`` is set):

.. parsed-literal::

    ├── ...  (the ``figures.save_directory`` output folder)
    │   ├── **vocal_pose_20250516_122134_21.50-23.50s.png**
    │   ├── **vocal_pose_20250516_122134_21.50-23.50s.svg**
    │   ├── **vocal_pose_20250516_122134_19.50-23.50s.mp4**
    │   ...

Every tunable lives in the ``vocal_pose_figures`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*. The block as shipped:

.. code-block:: json

        "vocal_pose_figures": {
            "window": {
                "end_time": 23.5,
                "history_seconds": 2.0,
                "trail_end_seconds": 1.0
            },
            "spectrogram": {
                "microphone": "peak",
                "n_fft": 256,
                "hop_fraction": 4,
                "freq_range_khz": [
                    30.0,
                    110.0
                ],
                "freq_ticks_khz": [
                    40.0,
                    70.0,
                    100.0
                ],
                "db_floor": "adaptive",
                "adaptive_floor_above_median_db": 12.0,
                "ceiling_percentile": 99.9,
                "gamma": 0.8,
                "male_deep": 0.4,
                "female_deep": 1.0,
                "ramp_split": 0.5,
                "vocal_pad_seconds": 0.005
            },
            "camera": {
                "azimuth": "male_side",
                "elevation": 45.0,
                "top_sex": "male"
            },
            "trail": {
                "strongest": 0.6,
                "fade_seconds": 0.1,
                "floor": 0.05,
                "taper_fraction": 0.5
            },
            "skeleton": {
                "line_width": 4.5,
                "node_size": 56.0,
                "body_opacity": 0.4,
                "trail_body_opacity": 0.0,
                "reference_half_extent_m": 0.175,
                "margin_m": 0.02,
                "z_margin_m": 0.01,
                "history_point": "Head"
            },
            "scale_mark": {
                "length_cm": 2.0,
                "labels": [
                    "X",
                    "Y",
                    "Z"
                ],
                "signs": [
                    1,
                    -1,
                    1
                ],
                "turn_degrees": 90.0,
                "gap_cm": 4.0,
                "line_width": 2.5,
                "label_gap_fraction": 0.028,
                "label_separation_degrees": 35.0,
                "caption_gap_factor": 1.7,
                "font_size": 6,
                "color": "#000000"
            },
            "surfaces": {
                "enabled": false,
                "color": "#EEEEEE",
                "junction_color": "#000000",
                "junction_width": 3.0,
                "wall_height_m": 0.6,
                "content_tolerance": 14.0,
                "pad_inches": 0.3
            },
            "layout": {
                "figure_width_inches": 6.0,
                "top_block_inches": 1.85,
                "spectrogram_left": 0.13,
                "spectrogram_width": 0.6,
                "spectrogram_top_inches": 0.1,
                "spectrogram_height_inches": 1.25,
                "bar_gap_inches": 0.05,
                "bar_height_inches": 0.1,
                "bar_samples": 2048,
                "bar_label_offset": 0.6,
                "frequency_label": "Frequency (kHz)",
                "time_label": "Time prior to frame (s)",
                "tick_font_size": 8,
                "label_font_size": 9,
                "tick_pad": 3,
                "ink_color": "#202020",
                "background_color": "#FFFFFF",
                "content_threshold": 250,
                "pose_pad_inches": 0.05,
                "crop_edge_pixels": 8,
                "svg_pad_inches": 0.1,
                "box_zoom": 1.25,
                "fit_shrink": 0.93,
                "max_fit_passes": 8,
                "align_pose": true,
                "color_key": true,
                "key_position": "pose",
                "key_swatch_inches": 0.24,
                "key_wide_inches": 0.6,
                "key_spacing_inches": 0.12,
                "key_gap_inches": 0.4,
                "key_clear_inches": 0.07
            },
            "output": {
                "dpi": 300,
                "save_svg": false
            },
            "video": {
                "duration_seconds": 4.0,
                "speed": 0.5,
                "follow_seconds": 0.25,
                "degrees_per_second": 0.0,
                "inches": 6.0,
                "dpi": 300,
                "crop_edge_pixels": 12,
                "workers": 0,
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "crf": 18,
                "encode_timeout_seconds": 3600,
                "trail": {
                    "trail_seconds": 0.25,
                    "strongest": 0.85,
                    "fade_seconds": 0.05,
                    "floor": 0.15
                },
                "skeleton": {
                    "line_width": 3.5,
                    "node_size": 56.0,
                    "body_opacity": 0.4,
                    "trail_body_opacity": 0.0,
                    "margin_m": 0.02,
                    "z_margin_m": 0.01,
                    "history_point": "Head"
                },
                "spectrogram": {
                    "enabled": true,
                    "layout": "inset",
                    "reach_seconds": 0.1,
                    "sharp_seconds": 0.025,
                    "fade_seconds": 0.02,
                    "floor": 0.15,
                    "height_fraction": 0.5,
                    "inset_max_width": 0.5,
                    "inset_margin_pixels": 0,
                    "inset_gap_pixels": 20,
                    "inset_shift": 1.0,
                    "inset_corner": "auto",
                    "corner_preference": 0.8,
                    "panel_left_pixels": 150,
                    "panel_right_pixels": 6,
                    "panel_edge_pixels": 22,
                    "panel_full_text_pixels": 290
                }
            },
            "traces": {
                "enabled": true,
                "width_inches": 2.6,
                "left_gap_inches": 0.35,
                "right_gap_inches": 0.1,
                "top_inches": 0.1,
                "bottom_inches": 0.3,
                "span_seconds": 2.0,
                "columns": "both",
                "egocentric": [
                    "speed",
                    "neck_elevation",
                    "allo_roll",
                    "allo_pitch",
                    "ego_yaw",
                    "back_pitch",
                    "back_yaw",
                    "tail_curvature"
                ],
                "dyadic": {
                    "male": [
                        "allo_yaw-head",
                        "allo_pitch-head",
                        "head-head"
                    ],
                    "female": [
                        "head-allo_yaw",
                        "head-allo_pitch",
                        "head-head"
                    ]
                },
                "labels": {
                    "speed": "speed (cm/s)",
                    "neck_elevation": "neck elevation (cm)",
                    "allo_roll": "head roll (°)",
                    "allo_pitch": "head pitch (°)",
                    "ego_yaw": "head yaw (°)",
                    "back_pitch": "back pitch (°)",
                    "back_yaw": "back yaw (°)",
                    "tail_curvature": "tail curvature (a.u.)",
                    "head-head": "partner distance (cm)",
                    "allo_yaw-head": "partner yaw orientation (°)",
                    "head-allo_yaw": "partner yaw orientation (°)",
                    "allo_pitch-head": "partner pitch orientation (°)",
                    "head-allo_pitch": "partner pitch orientation (°)"
                },
                "fade_power": 3.0,
                "line_width_centre": 2.2,
                "line_width_end": 0.4,
                "y_pad_fraction": 0.08,
                "label_fraction": 0.4,
                "column_gap_fraction": 0.06,
                "row_gap_fraction": 0.12,
                "label_gap_fraction": 0.012,
                "value_decimals": {
                    "small": 2,
                    "medium": 1,
                    "large": 0
                },
                "value_decimal_thresholds": [
                    1.0,
                    10.0
                ],
                "value_font_scale": 0.8,
                "font_size": 6,
                "zero_tick": true,
                "zero_tick_width": 0.5,
                "zero_tick_alpha": 0.25,
                "time_label": "Time relative to frame (s)",
                "time_label_offset": 0.35
            },
            "candidates": {
                "window_seconds": 2.0,
                "step_seconds": 0.25,
                "n_shown": 12,
                "max_gap_cm": 5.0,
                "sheet_width_inches": 11.0,
                "row_height_inches": 1.55,
                "sheet_dpi": 200,
                "sheet_margins": {
                    "left": 0.055,
                    "right": 0.995,
                    "top": 0.975,
                    "bottom": 0.01,
                    "hspace": 0.42
                }
            },
            "view_picker": {
                "n_trail": 60,
                "strongest": 0.6,
                "fade_seconds": 0.1,
                "floor": 0.12
            }
        }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`window <vis-vocal-pose-figures-window>`
     - see JSON above
     - frame time, spectrogram span and trail span, in seconds
   * - :ref:`spectrogram <vis-vocal-pose-figures-spectrogram>`
     - see JSON above
     - microphone, band, STFT, floor and loudness scaling
   * - :ref:`camera <vis-vocal-pose-figures-camera>`
     - see JSON above
     - ``"male_side"`` or degrees; elevation; top animal
   * - :ref:`trail <vis-vocal-pose-figures-trail>`
     - see JSON above
     - fade of the still's pose trail
   * - :ref:`skeleton <vis-vocal-pose-figures-skeleton>`
     - see JSON above
     - line widths, node size, body opacity, margins
   * - :ref:`scale_mark <vis-vocal-pose-figures-scale-mark>`
     - see JSON above
     - three-arm scale mark; ``length_cm`` 0 = none
   * - :ref:`surfaces <vis-vocal-pose-figures-surfaces>`
     - see JSON above
     - arena floor and wall; ``enabled`` ``true`` / ``false``
   * - :ref:`layout <vis-vocal-pose-figures-layout>`
     - see JSON above
     - figure geometry, fonts, ink, crops, fitting, colour key
   * - :ref:`output <vis-vocal-pose-figures-output>`
     - see JSON above
     - ``dpi`` and ``save_svg``
   * - :ref:`video <vis-vocal-pose-figures-video>`
     - see JSON above
     - clip length, speed, camera motion, render, ffmpeg, its own trail / skeleton / spectrogram
   * - :ref:`traces <vis-vocal-pose-figures-traces>`
     - see JSON above
     - behavioural-feature trace panel
   * - :ref:`candidates <vis-vocal-pose-figures-candidates>`
     - see JSON above
     - window search and the candidate sheet
   * - :ref:`view_picker <vis-vocal-pose-figures-view-picker>`
     - see JSON above
     - trail silhouettes of the view-picker page

.. _vis-vocal-pose-figures-window:

``window``
   The frame the still shows and the spans that lead up to it.

   :Options:
      * ``end_time`` — the frame shown, in seconds from video start
      * ``history_seconds`` — the spectrogram / time-bar span before it, in seconds
      * ``trail_end_seconds`` — the age, in seconds, at which the trail has faded to nothing; ``0`` keeps it for the whole window
   :Default: ``end_time`` ``23.5``, ``history_seconds`` ``2.0``, ``trail_end_seconds`` ``1.0``

.. _vis-vocal-pose-figures-spectrogram:

``spectrogram``
   The spectrogram panel of the still: which microphone, which band, the STFT and the loudness scaling.

   :Options:
      * ``microphone`` — ``"peak"`` (the microphone most of the window's vocalizations peak on) or a device-channel name such as ``"s04"``
      * ``freq_range_khz`` / ``freq_ticks_khz`` — the band shown and its labels, in kHz
      * ``db_floor`` — ``"adaptive"`` (``adaptive_floor_above_median_db`` above the band's median power) or a dB value
      * ``adaptive_floor_above_median_db`` — how far above the band's median power the adaptive floor sits, in dB
      * ``n_fft`` / ``hop_fraction`` — the STFT (a shorter window widens narrowband sweeps); the hop is ``n_fft // hop_fraction`` samples
      * ``male_deep`` / ``female_deep`` / ``ramp_split`` — the darker end of each ramp
      * ``ceiling_percentile`` / ``gamma`` / ``vocal_pad_seconds`` — the loudness scaling
   :Default: ``microphone`` ``"peak"``, ``n_fft`` ``256``, ``hop_fraction`` ``4``, ``freq_range_khz`` ``[30.0, 110.0]``, ``freq_ticks_khz`` ``[40.0, 70.0, 100.0]``, ``db_floor`` ``"adaptive"``, ``adaptive_floor_above_median_db`` ``12.0``, ``ceiling_percentile`` ``99.9``, ``gamma`` ``0.8``, ``male_deep`` ``0.4``, ``female_deep`` ``1.0``, ``ramp_split`` ``0.5``, ``vocal_pad_seconds`` ``0.005``

.. _vis-vocal-pose-figures-camera:

``camera``
   Where the still's camera stands and which animal is drawn on top.

   :Options:
      * ``azimuth`` — ``"male_side"`` or degrees
      * ``elevation`` — degrees above the floor (90 is straight down)
      * ``top_sex`` — the animal drawn last, on top where they overlap
   :Default: ``azimuth`` ``"male_side"``, ``elevation`` ``45.0``, ``top_sex`` ``"male"``

.. _vis-vocal-pose-figures-trail:

``trail``
   The fade of the still's pose trail (the silhouettes of the preceding movement); the time bar under the spectrogram follows the same fade.

   :Options:
      * ``strongest`` — colour strength just behind the present
      * ``fade_seconds`` — time constant of the fade, in seconds
      * ``floor`` — strength old silhouettes settle to
      * ``taper_fraction`` — the last fraction of the trail over which it tapers to nothing
   :Default: ``strongest`` ``0.6``, ``fade_seconds`` ``0.1``, ``floor`` ``0.05``, ``taper_fraction`` ``0.5``

.. _vis-vocal-pose-figures-skeleton:

``skeleton``
   How the still draws the animals.

   :Options:
      * ``line_width`` / ``node_size`` — at the reference zoom (``reference_half_extent_m``)
      * ``reference_half_extent_m`` — the reference zoom, in metres
      * ``body_opacity`` / ``trail_body_opacity`` — of the filled bodies (current poses / trail silhouettes)
      * ``margin_m`` / ``z_margin_m`` — around the trail, in metres
      * ``history_point`` — the node passed to the shared mouse renderer as its history point
   :Default: ``line_width`` ``4.5``, ``node_size`` ``56.0``, ``body_opacity`` ``0.4``, ``trail_body_opacity`` ``0.0``, ``reference_half_extent_m`` ``0.175``, ``margin_m`` ``0.02``, ``z_margin_m`` ``0.01``, ``history_point`` ``"Head"``

.. _vis-vocal-pose-figures-scale-mark:

``scale_mark``
   The optional scale mark: three perpendicular arms of ``length_cm`` beside the male.

   :Options:
      * ``length_cm`` — arm length in cm; ``0`` = none
      * ``labels`` — the three arm labels
      * ``signs`` — one per arm; −1 reverses an arm
      * ``turn_degrees`` — turns the two floor arms about the vertical
      * ``gap_cm`` — the gap from the male, in cm
      * ``line_width`` — arm line width
      * ``label_gap_fraction`` / ``label_separation_degrees`` / ``caption_gap_factor`` — label and caption placement
      * ``font_size`` — label font size
      * ``color`` — hex colour of the mark
   :Default: ``length_cm`` ``2.0``, ``labels`` ``["X", "Y", "Z"]``, ``signs`` ``[1, -1, 1]``, ``turn_degrees`` ``90.0``, ``gap_cm`` ``4.0``, ``line_width`` ``2.5``, ``label_gap_fraction`` ``0.028``, ``label_separation_degrees`` ``35.0``, ``caption_gap_factor`` ``1.7``, ``font_size`` ``6``, ``color`` ``"#000000"``

.. _vis-vocal-pose-figures-surfaces:

``surfaces``
   The arena floor and nearest wall drawn as grey surfaces with a black junction line.

   :Options:
      * ``enabled`` — ``true`` draws them, ``false`` does not
      * ``color`` — hex colour of the surfaces
      * ``junction_color`` / ``junction_width`` — the floor–wall junction line
      * ``wall_height_m`` — wall height, in metres
      * ``content_tolerance`` — what counts as drawn against the surface colour when cropping
      * ``pad_inches`` — padding around the pose panel's content when the surfaces are drawn (``layout.pose_pad_inches`` applies otherwise)
   :Default: ``enabled`` ``false``, ``color`` ``"#EEEEEE"``, ``junction_color`` ``"#000000"``, ``junction_width`` ``3.0``, ``wall_height_m`` ``0.6``, ``content_tolerance`` ``14.0``, ``pad_inches`` ``0.3``

.. _vis-vocal-pose-figures-layout:

``layout``
   The still's geometry, text and crops.

   :Options:
      * ``figure_width_inches`` / ``top_block_inches`` — figure width, and the height of the block above the pose panel, in inches
      * ``spectrogram_left`` / ``spectrogram_width`` — the spectrogram's left edge and width, as fractions of the figure width
      * ``spectrogram_top_inches`` / ``spectrogram_height_inches`` — its top margin and height, in inches
      * ``bar_gap_inches`` / ``bar_height_inches`` / ``bar_samples`` / ``bar_label_offset`` — the time bar: its gap below the spectrogram, height, number of samples along it and the offset of its labels
      * ``frequency_label`` / ``time_label`` — the spectrogram's and the bar's axis labels
      * ``tick_font_size`` / ``label_font_size`` / ``tick_pad`` — fonts
      * ``ink_color`` / ``background_color`` — ink and page colours
      * ``content_threshold`` / ``crop_edge_pixels`` / ``svg_pad_inches`` — for the crops; ``pose_pad_inches`` pads the pose panel's content when the surfaces are off
      * ``box_zoom`` / ``fit_shrink`` / ``max_fit_passes`` — the pose box is shrunk until nothing is cut off
      * ``align_pose`` — ``true`` moves the whole tracking panel left so its right edge sits under the spectrogram's
      * ``color_key`` — ``true`` draws the two-swatch colour key
      * ``key_position`` — ``"pose"`` = wide swatches beside the animals, ``"spectrogram"`` = squares to its right
      * ``key_swatch_inches`` / ``key_wide_inches`` / ``key_spacing_inches`` / ``key_gap_inches`` / ``key_clear_inches`` — the swatch sizes and spacing
   :Default: ``figure_width_inches`` ``6.0``, ``top_block_inches`` ``1.85``, ``spectrogram_left`` ``0.13``, ``spectrogram_width`` ``0.6``, ``spectrogram_top_inches`` ``0.1``, ``spectrogram_height_inches`` ``1.25``, ``bar_gap_inches`` ``0.05``, ``bar_height_inches`` ``0.1``, ``bar_samples`` ``2048``, ``bar_label_offset`` ``0.6``, ``frequency_label`` ``"Frequency (kHz)"``, ``time_label`` ``"Time prior to frame (s)"``, ``tick_font_size`` ``8``, ``label_font_size`` ``9``, ``tick_pad`` ``3``, ``ink_color`` ``"#202020"``, ``background_color`` ``"#FFFFFF"``, ``content_threshold`` ``250``, ``pose_pad_inches`` ``0.05``, ``crop_edge_pixels`` ``8``, ``svg_pad_inches`` ``0.1``, ``box_zoom`` ``1.25``, ``fit_shrink`` ``0.93``, ``max_fit_passes`` ``8``, ``align_pose`` ``true``, ``color_key`` ``true``, ``key_position`` ``"pose"``, ``key_swatch_inches`` ``0.24``, ``key_wide_inches`` ``0.6``, ``key_spacing_inches`` ``0.12``, ``key_gap_inches`` ``0.4``, ``key_clear_inches`` ``0.07``

.. _vis-vocal-pose-figures-output:

``output``
   How the still is written.

   :Options:
      * ``dpi`` — resolution of the PNG (and of the SVG's embedded images)
      * ``save_svg`` — ``true`` also writes the SVG, ``false`` writes the PNG only
   :Default: ``dpi`` ``300``, ``save_svg`` ``false``

.. _vis-vocal-pose-figures-video:

``video``
   The vocal-pose video.

   :Options:
      * ``duration_seconds`` — the span played before the still's frame, in seconds
      * ``speed`` — playback speed (0.5 = half speed, every camera frame kept)
      * ``follow_seconds`` — ``≥ 0`` follows the pair (centres the smoothed middle of their outline in every frame); −1 = fixed camera
      * ``degrees_per_second`` — the camera turns at this rate into the still's view on the last frame
      * render ``inches`` / ``dpi`` / ``crop_edge_pixels`` / ``workers`` — frame size, resolution, crop margin and the number of render processes (0 = all cores but one)
      * the ffmpeg ``codec`` / ``pixel_format`` / ``crf`` / ``encode_timeout_seconds``
      * ``trail`` — its own sub-block: ``trail_seconds`` (the fading trail behind each animal) plus ``strongest`` / ``fade_seconds`` / ``floor`` as in the still's ``trail``
      * ``skeleton`` — its own sub-block, the keys of the still's ``skeleton`` except ``reference_half_extent_m``
      * ``spectrogram`` — ``enabled``, ``layout`` (``"inset"``, in a corner no animal ever enters, or ``"below"``, a strip below the animals), ``reach_seconds`` (visible on each side of the present), ``sharp_seconds`` (sharp only within it), ``fade_seconds``, ``floor``, the inset's size (``height_fraction``, ``inset_max_width``, ``inset_margin_pixels``, ``inset_gap_pixels``), corner (``inset_corner``, ``"auto"`` or a named corner; ``corner_preference``) and shift (``inset_shift``, the share of the free room the inset is slid left by), and the panel text room (``panel_left_pixels``, ``panel_right_pixels``, ``panel_edge_pixels``, ``panel_full_text_pixels``)
   :Default: ``duration_seconds`` ``4.0``, ``speed`` ``0.5``, ``follow_seconds`` ``0.25``, ``degrees_per_second`` ``0.0``, ``inches`` ``6.0``, ``dpi`` ``300``, ``crop_edge_pixels`` ``12``, ``workers`` ``0``, ``codec`` ``"libx264"``, ``pixel_format`` ``"yuv420p"``, ``crf`` ``18``, ``encode_timeout_seconds`` ``3600``; ``trail``, ``skeleton`` and ``spectrogram`` as in the JSON above

.. _vis-vocal-pose-figures-traces:

``traces``
   The third panel, to the right of the still, showing what the pair's bodies were doing around the frame.

   :Options:
      * ``enabled`` — ``true`` draws the panel
      * ``width_inches`` — width of the panel, with its ``left_gap_inches`` / ``right_gap_inches`` / ``top_inches`` / ``bottom_inches``
      * ``span_seconds`` — each side of the frame
      * ``columns`` — ``"both"``, ``"male"`` or ``"female"``
      * ``egocentric`` — feature names, top to bottom
      * ``dyadic`` — per sex, the same number of features, the name's order giving the side, e.g. ``allo_yaw-head`` for the male and ``head-allo_yaw`` for the female
      * ``labels`` — feature -> printed name
      * ``fade_power``, ``line_width_centre`` / ``line_width_end``, ``y_pad_fraction``
      * the panel's division into ``label_fraction``, ``column_gap_fraction``, ``row_gap_fraction``, ``label_gap_fraction``
      * the range numbers' ``value_decimals`` (``small`` / ``medium`` / ``large`` for magnitudes split at ``value_decimal_thresholds``) and ``value_font_scale``
      * ``font_size``
      * the frame line ``zero_tick`` / ``zero_tick_width`` / ``zero_tick_alpha``
      * ``time_label`` and its ``time_label_offset`` (in row heights below the last row)
   :Default: ``enabled`` ``true``, ``width_inches`` ``2.6``, ``span_seconds`` ``2.0``, ``columns`` ``"both"``, ``fade_power`` ``3.0``, ``line_width_centre`` ``2.2``, ``line_width_end`` ``0.4``, ``font_size`` ``6``, ``time_label`` ``"Time relative to frame (s)"``; the feature lists, labels and the other numbers as in the JSON above

.. _vis-vocal-pose-figures-candidates:

``candidates``
   The window search of ``find_vocal_pose_windows`` and the candidate sheet of ``plot_vocal_pose_window_candidates``.

   :Options:
      * ``window_seconds`` — the window length, in seconds
      * ``step_seconds`` — the step between candidate windows, in seconds
      * ``n_shown`` — how many non-overlapping best windows the sheet stacks
      * ``max_gap_cm`` — the largest allowed smallest distance between any body keypoints of the two (tails excluded), in cm
      * ``sheet_width_inches`` / ``row_height_inches`` / ``sheet_dpi`` / ``sheet_margins`` — the candidate sheet's size
   :Default: ``window_seconds`` ``2.0``, ``step_seconds`` ``0.25``, ``n_shown`` ``12``, ``max_gap_cm`` ``5.0``, ``sheet_width_inches`` ``11.0``, ``row_height_inches`` ``1.55``, ``sheet_dpi`` ``200``; ``sheet_margins`` as in the JSON above

.. _vis-vocal-pose-figures-view-picker:

``view_picker``
   How many trail silhouettes the page of ``vocal_pose_view_picker_html`` draws and their fade.

   :Options:
      * ``n_trail`` — the number of trail silhouettes
      * ``strongest`` / ``fade_seconds`` / ``floor`` — their fade, as in ``trail``
   :Default: ``n_trail`` ``60``, ``strongest`` ``0.6``, ``fade_seconds`` ``0.1``, ``floor`` ``0.12``

Render QLVM figures
-------------------
The ``usv_playpen.visualizations.qlvm_figures`` module renders three **cohort-level** figures of the regular QLVM map and its content-ridge categories, and ``usv_playpen.visualizations.qlvm_property_sweep_video`` the matching property-sweep video. Like the vocal-pose figures they are **notebook-driven** (the ``usv_general_analyses.ipynb`` cells *QLVM property-sweep video* and *QLVM figures*, see :doc:`Notebooks`), not exposed in the GUI, so they have no ``visualize_booleans`` switch; every setting the figures read lives in the ``qlvm_figures`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*, and every setting of the sweep video in its ``qlvm_property_sweep_video`` block (plus ``figures.sequential_cmap`` and ``figures.seed``).

**Data.** The USVs are the cohort's: ``make_usv_spectrograms.load_regular_map_cohort_usvs`` pools every non-playback ``*sessions_list.txt`` under ``shared_resources.input_files_directory`` (``read_cohort_session_roots`` — the same cohort definition as the embedding thumbnails and the embedding explorer) through ``build_pooled_embeddings_df`` and the pooled-embeddings cache ``<spectrograms_dir>/embeddings/pooled_embeddings_production.parquet`` (``os_utils.POOLED_EMBEDDINGS_CACHE_NAME``, rebuilt when the summaries change), and keeps the pure USVs (``usv`` true, ``squeak`` false; noise segments dropped) that have a regular-map position (``qlvm1`` / ``qlvm2``) and all six measured properties (``duration``, ``spectral_entropy``, ``freq_bandwidth_hz``, ``mean_freq_hz``, ``loudness_db``, ``mask_number``; ``make_usv_spectrograms.REGULAR_MAP_PROPERTY_COLUMNS``). The category geometry is the QLVM category bundle (``os_utils.load_qlvm_category_bundle``; see *Where the category geometry comes from* above), and ``check_categories_match_bundle`` first verifies that every pooled ``qlvm_category`` is the category the bundle's grid gives the USV's position (the pixel rule ``assign-qlvm-categories`` applies): summaries categorized with another bundle, or before their positions were last written, stop the figures with a message naming the sessions. The spectrogram-based panels use a per-session sample (``build_qlvm_figure_sample``: up to ``usvs_per_session`` random USVs of every session, read from the session's own ``audio/spectrograms/<session>_spectrograms.h5`` and masked by the union of the USV's SAM2 masks, kept both as the regular cell's decoder input — time-stretched to 128 × 128 and normalized by the production cell's training contract, exactly what ``infer-qlvm-latents`` embeds — and at native length), written to ``sample_cache_file`` and served again while the drawn USVs are the same.

- ``plot_qlvm_overview`` — the map at a glance: the USV density (``auxiliary_plot_functions.periodic_density``, a periodic Gaussian so the torus seams wrap), the bundle's content-change field with the R-1 … R-k boundaries and numbers, the USVs coloured by category with example spectrograms sampled along a spiral around each category's densest point, and the USVs coloured by duration, spectral entropy, bandwidth, mean frequency and loudness.
- ``compute_qlvm_quality_metrics`` / ``plot_qlvm_quality`` — the regular map against PCA and UMAP of the same decoder inputs: the acoustic readout from 2-D position (kNN rank R², whole sessions held out, against shuffled positions), neighborhood preservation (neighbor overlap, false-neighbor severity = 1 − trustworthiness, torn-apart severity = 1 − continuity), distance fidelity (distance correlation of map distances with property distances; for the QLVM the decoder pullback geodesic, the density geodesic and the flat-torus distance of ``modeling.modeling_torus_geodesics``) and the readout per training seed, the sample re-embedded with the cells of ``seed_cell_directories`` (``embed_with_cells``).
- ``compute_category_boundary_fields`` / ``compute_category_similarity`` / ``plot_category_boundaries`` — what the content-ridge boundaries follow (the density and the content change on the boundary pixels against the same pattern shifted rigidly around the torus; each property's share of the change; the change per property) and whether the categories group similar USVs (the aligned, per-USV-normalized spectrogram similarity between categories, each USV's attachment to its own category, where its closest matches fall, and the categories against the density-watershed clusters of ``watershed_levels_file``).
- ``render_qlvm_property_sweep_video`` — six panels (SAM-mask count, duration, bandwidth, spectral entropy, mean frequency, loudness) sweeping an equal-count percentile window of each property over the regular map: the window's USV density in the colour of its median value on black, the bundle's boundaries dotted in white, and a colour bar marking the window; a preview quantile renders one still instead.

Being cohort-level, the figures are written to ``figures.save_directory`` as ``qlvm_overview``, ``qlvm_quality`` and ``qlvm_categories`` (through ``figure_io.save_figure``, so in the ``figures`` block's format and DPI, with a ``_YYYYMMDD_HHMMSS`` stamp when ``figures.timestamp_in_name`` is set); the video's ``.mp4`` path is set in the notebook cell:

.. parsed-literal::

    ├── ...  (the ``figures.save_directory`` cohort output folder)
    │   ├── **qlvm_overview_20250430_145017.png**
    │   ├── **qlvm_quality_20250430_145017.png**
    │   ├── **qlvm_categories_20250430_145017.png**
    │   ├── **qlvm_property_sweep.mp4**
    │   ...

The ``qlvm_figures`` block (the seed is ``figures.seed`` and the colormap ``figures.sequential_cmap``):

.. code-block:: json

    "qlvm_figures": {
        "n_usvs": 100000,
        "usvs_per_session": 200,
        "sample_cache_file": "/mnt/falkner/Bartul/spectrograms/embeddings/qlvm_figure_sample.npz",
        "seed_cell_directories": [
            "/mnt/falkner/Bartul/spectrograms/qlvm/seeds/seed7",
            "/mnt/falkner/Bartul/spectrograms/qlvm/seeds/seed1234"
        ],
        "watershed_levels_file": "/mnt/falkner/Bartul/spectrograms/qlvm/watershed_levels.npz",
        "category_colors": ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"],
        "overview": {
            "n_examples": 10,
            "spiral_radius": 0.03,
            "spiral_turns": 3,
            "density_bandwidth": 0.01
        },
        "quality": {
            "n_pcs": 50,
            "umap_n_neighbors": 30,
            "umap_min_dist": 0.1,
            "readout_neighbors": 20,
            "readout_folds": 5,
            "neighborhood_k": 20,
            "n_neighborhood_subsets": 5,
            "neighborhood_subset_size": 9000,
            "distance_sample_size": 3000,
            "n_distance_samples": 5,
            "pullback_grid": 48,
            "embed_lattice_batch_size": 2048,
            "embed_data_batch_size": 1024
        },
        "categories": {
            "n_shifts": 1000,
            "histogram_bins": 30,
            "similarity_downsample": 2,
            "similarity_max_shift": 4,
            "balanced_per_category": 2000,
            "balanced_reference_per_category": 1000,
            "closest_matches": 100,
            "closest_log2_limit": 2.0,
            "partition_sample_size": 8000,
            "partition_reference_size": 4000,
            "average_size": 1000,
            "align_max_shift": 8,
            "align_iterations": 5,
            "average_time_ms": 150.0
        },
        "style": {
            "text_color": "#000000",
            "figure_face": "#FFFFFF",
            "boundary_color": "#FFFFFF",
            "property_boundary_color": "#000000",
            "map_colors": {
                "QLVM": "#000000",
                "PCA": "#9A9A9A",
                "UMAP": "#E69F00"
            },
            "distance_colors": {
                "QLVM pullback geodesic": "#000000",
                "QLVM density geodesic": "#555555",
                "QLVM flat torus": "#8C8C8C",
                "PCA Euclidean": "#BDBDBD",
                "UMAP Euclidean": "#E69F00"
            },
            "seed_colors": ["#000000", "#666666", "#A6A6A6", "#CCCCCC"],
            "null_color": "#D62728",
            "density_color": "#4E79A7",
            "change_color": "#E15759",
            "bar_color": "#59595B",
            "watershed_color": "#BDBDBD",
            "group_dot_color": "#FFFFFF",
            "title_size": 14,
            "label_size": 14,
            "tick_size": 12,
            "colorbar_tick_size": 13,
            "colorbar_label_size": 15,
            "quality_label_size": 16,
            "quality_tick_size": 14,
            "quality_title_size": 16,
            "color_percentiles": [1, 99],
            "overview_figsize": [30, 12.5],
            "quality_figsize": [20, 22.5],
            "categories_figsize": [32.5, 16.6]
        }
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`n_usvs <vis-qlvm-figures-n-usvs>`
     - ``100000``
     - integer > 0
   * - :ref:`usvs_per_session <vis-qlvm-figures-usvs-per-session>`
     - ``200``
     - integer > 0
   * - :ref:`sample_cache_file <vis-qlvm-figures-sample-cache-file>`
     - path
     - path to an ``.npz`` file, or empty
   * - :ref:`seed_cell_directories <vis-qlvm-figures-seed-cell-directories>`
     - two paths (see JSON above)
     - list of at most three model-cell directories, or empty
   * - :ref:`watershed_levels_file <vis-qlvm-figures-watershed-levels-file>`
     - path
     - path to an ``.npz`` file, or empty
   * - :ref:`category_colors <vis-qlvm-figures-category-colors>`
     - see JSON above
     - one hex colour per category
   * - :ref:`overview <vis-qlvm-figures-overview>`
     - see JSON above
     - example spectrograms, spiral, density bandwidth
   * - :ref:`quality <vis-qlvm-figures-quality>`
     - see JSON above
     - PCA / UMAP, readout, neighborhood, distance, geodesic grid, embedding batches
   * - :ref:`categories <vis-qlvm-figures-categories>`
     - see JSON above
     - along-boundary test, similarity, closest matches, partitions, aligned means
   * - :ref:`style <vis-qlvm-figures-style>`
     - see JSON above
     - colours, font sizes, colour percentiles, figure sizes

.. _vis-qlvm-figures-n-usvs:

``n_usvs``
   Most cohort USVs drawn for the map panels (the overview, and the density and content-change fields of the category figure).

   :Options: a positive integer
   :Default: ``100000``

.. _vis-qlvm-figures-usvs-per-session:

``usvs_per_session``
   Most USVs per session in the spectrogram sample.

   :Options: a positive integer
   :Default: ``200``

.. _vis-qlvm-figures-sample-cache-file:

``sample_cache_file``
   ``.npz`` cache of the spectrogram sample.

   :Options: path (string) to an ``.npz`` file; empty: no cache
   :Default: ``/mnt/falkner/Bartul/spectrograms/embeddings/qlvm_figure_sample.npz``
   :Notes: it is written to a temporary file and renamed into place, so an interrupted run never leaves a truncated cache to be served next time.

.. _vis-qlvm-figures-seed-cell-directories:

``seed_cell_directories``
   Model cells of other trainings of the regular map's recipe (other seeds), with which the quality figure's readout-per-seed panel re-embeds the sample.

   :Options: a list of at most three model-cell directories; empty skips the panel
   :Default: ``/mnt/falkner/Bartul/spectrograms/qlvm/seeds/seed7`` and ``/mnt/falkner/Bartul/spectrograms/qlvm/seeds/seed1234``
   :Notes: the panel has one colour per training (four, the production cell included), so at most three cells can be listed; more raises ``ValueError`` instead of leaving trainings out. These validation cells are kept outside the production package, so the shipped paths name them explicitly.

.. _vis-qlvm-figures-watershed-levels-file:

``watershed_levels_file``
   ``.npz`` of the density-watershed partitions of the regular map (its ``final_coarse``, ``k4`` and ``final_fine`` grids: 3, 4 and 6 clusters) that the category figure's partition panel compares the categories with.

   :Options: path (string) to an ``.npz`` file; empty compares the categories alone
   :Default: ``/mnt/falkner/Bartul/spectrograms/qlvm/watershed_levels.npz``

.. _vis-qlvm-figures-category-colors:

``category_colors``
   One hex colour per category (R-1 … R-k in order).

   :Options: a list of hex colour strings, one per category
   :Default: ``["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]``

.. _vis-qlvm-figures-overview:

``overview``
   The overview figure's example spectrograms and the density fields.

   :Options:
      * ``n_examples`` — spectrograms per category
      * ``spiral_radius`` (torus units) and ``spiral_turns`` — of the example spiral
      * ``density_bandwidth`` — bandwidth (torus units) of every density field of the figures
   :Default: ``n_examples`` ``10``, ``spiral_radius`` ``0.03``, ``spiral_turns`` ``3``, ``density_bandwidth`` ``0.01``

.. _vis-qlvm-figures-quality:

``quality``
   The quality figure's comparison maps and scores.

   :Options:
      * ``n_pcs`` — PCA components (the first two are the PCA map, all of them the UMAP input)
      * ``umap_n_neighbors`` / ``umap_min_dist`` — the UMAP
      * ``readout_neighbors`` / ``readout_folds`` — of the acoustic readout
      * ``neighborhood_k`` / ``n_neighborhood_subsets`` / ``neighborhood_subset_size`` — of the neighborhood scores
      * ``distance_sample_size`` / ``n_distance_samples`` — of the distance fidelity
      * ``pullback_grid`` — nodes per axis of the geodesic grids
      * ``embed_lattice_batch_size`` / ``embed_data_batch_size`` — of the seed-cell embedding
   :Default: ``n_pcs`` ``50``, ``umap_n_neighbors`` ``30``, ``umap_min_dist`` ``0.1``, ``readout_neighbors`` ``20``, ``readout_folds`` ``5``, ``neighborhood_k`` ``20``, ``n_neighborhood_subsets`` ``5``, ``neighborhood_subset_size`` ``9000``, ``distance_sample_size`` ``3000``, ``n_distance_samples`` ``5``, ``pullback_grid`` ``48``, ``embed_lattice_batch_size`` ``2048``, ``embed_data_batch_size`` ``1024``

.. _vis-qlvm-figures-categories:

``categories``
   The category figure's tests and panels.

   :Options:
      * ``n_shifts`` — of the along-boundary test, and ``histogram_bins`` of its histogram
      * ``similarity_downsample`` / ``similarity_max_shift`` — of the aligned correlation
      * ``balanced_per_category`` / ``balanced_reference_per_category`` — of the between-category panels
      * ``closest_matches`` and ``closest_log2_limit`` — of the closest-matches panel
      * ``partition_sample_size`` / ``partition_reference_size`` — of the partition comparison
      * ``average_size`` / ``align_max_shift`` / ``align_iterations`` / ``average_time_ms`` — of the aligned category means
   :Default: ``n_shifts`` ``1000``, ``histogram_bins`` ``30``, ``similarity_downsample`` ``2``, ``similarity_max_shift`` ``4``, ``balanced_per_category`` ``2000``, ``balanced_reference_per_category`` ``1000``, ``closest_matches`` ``100``, ``closest_log2_limit`` ``2.0``, ``partition_sample_size`` ``8000``, ``partition_reference_size`` ``4000``, ``average_size`` ``1000``, ``align_max_shift`` ``8``, ``align_iterations`` ``5``, ``average_time_ms`` ``150.0``

.. _vis-qlvm-figures-style:

``style``
   The look of the three figures.

   :Options:
      * ``text_color`` — all text and reference lines (``#000000``)
      * ``figure_face`` — page (``#FFFFFF``)
      * ``boundary_color`` — category boundaries and numbers on the fields, and the strong cells' text of tables (i) and (iii) (``#FFFFFF``)
      * ``property_boundary_color`` — boundaries over the property-coloured maps (``#000000``)
      * ``map_colors`` — one per compared map, keyed ``QLVM`` / ``PCA`` / ``UMAP``
      * ``distance_colors`` — one per distance measure of panel (b'), keyed by the measure's label
      * ``seed_colors`` — one per training in panel (c), the production cell included, so their count caps the trainings: four
      * ``null_color`` — shuffled-position readout (``#D62728``)
      * ``density_color`` / ``change_color`` — the two measures of the along-boundary test (``#4E79A7`` / ``#E15759``)
      * ``bar_color`` — property-share and category-partition bars (``#59595B``)
      * ``watershed_color`` — watershed-partition bars (``#BDBDBD``)
      * ``group_dot_color`` — partition group dots (``#FFFFFF``)
      * font sizes in points -- ``title_size`` / ``label_size`` / ``tick_size`` (14 / 14 / 12) and ``colorbar_tick_size`` / ``colorbar_label_size`` (13 / 15) of the overview and category figures, ``quality_title_size`` / ``quality_label_size`` / ``quality_tick_size`` (16 / 16 / 14) of the quality figure's summary panels
      * ``color_percentiles`` — the percentiles of every map's and field's colour range, so a few extreme USVs or pixels do not compress the scale (``[1, 99]``)
      * the figure sizes in inches, ``overview_figsize`` (30 × 12.5), ``quality_figsize`` (20 × 22.5) and ``categories_figsize`` (32.5 × 16.6), sized so the panels of each figure keep their aspect
   :Default: as given in brackets above; ``map_colors``, ``distance_colors`` and ``seed_colors`` as in the JSON above

The ``qlvm_property_sweep_video`` block (the seed of the tie order is ``figures.seed`` and the colormap ``figures.sequential_cmap``; the cohort directory, cache, output path and preview quantile are run choices set in the notebook cell):

.. code-block:: json

    "qlvm_property_sweep_video": {
        "window": 0.1,
        "step": 0.01,
        "hold": 8,
        "fps": 15,
        "dpi": 100,
        "density_bandwidth": 0.01,
        "colormap_floor": 0.2,
        "color_percentiles": [1, 99],
        "brightness_percentile": 99,
        "figure_size": [19.0, 11.4],
        "figure_face": "#FFFFFF",
        "boundary_color": "#FFFFFF",
        "median_mark_color": "#000000",
        "window_shade_color": "#FFFFFF",
        "colorbar_label_size": 17,
        "colorbar_tick_size": 14,
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "crf": 18
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`window <vis-qlvm-property-sweep-video-window>`
     - ``0.1``
     - share of the USVs in ``(0, 1]``
   * - :ref:`step <vis-qlvm-property-sweep-video-step>`
     - ``0.01``
     - share of the USVs > 0
   * - :ref:`hold <vis-qlvm-property-sweep-video-hold>`
     - ``8``
     - frames >= 0
   * - :ref:`fps <vis-qlvm-property-sweep-video-fps>`
     - ``15``
     - frames per second > 0
   * - :ref:`dpi <vis-qlvm-property-sweep-video-dpi>`
     - ``100``
     - integer > 0
   * - :ref:`density_bandwidth <vis-qlvm-property-sweep-video-density-bandwidth>`
     - ``0.01``
     - torus units > 0
   * - :ref:`colormap_floor <vis-qlvm-property-sweep-video-colormap-floor>`
     - ``0.2``
     - number in [0, 1)
   * - :ref:`color_percentiles <vis-qlvm-property-sweep-video-color-percentiles>`
     - ``[1, 99]``
     - two percentiles
   * - :ref:`brightness_percentile <vis-qlvm-property-sweep-video-brightness-percentile>`
     - ``99``
     - percentile
   * - :ref:`figure_size <vis-qlvm-property-sweep-video-figure-size>`
     - ``[19.0, 11.4]``
     - width, height in inches
   * - :ref:`figure_face <vis-qlvm-property-sweep-video-figure-face>`
     - ``#FFFFFF``
     - hex colour
   * - :ref:`boundary_color <vis-qlvm-property-sweep-video-boundary-color>`
     - ``#FFFFFF``
     - hex colour
   * - :ref:`median_mark_color <vis-qlvm-property-sweep-video-mark-shade>`
     - ``#000000``
     - hex colour
   * - :ref:`window_shade_color <vis-qlvm-property-sweep-video-mark-shade>`
     - ``#FFFFFF``
     - hex colour
   * - :ref:`colorbar_label_size <vis-qlvm-property-sweep-video-colorbar-sizes>`
     - ``17``
     - points
   * - :ref:`colorbar_tick_size <vis-qlvm-property-sweep-video-colorbar-sizes>`
     - ``14``
     - points
   * - :ref:`codec <vis-qlvm-property-sweep-video-ffmpeg>`
     - ``libx264``
     - ffmpeg encoder name
   * - :ref:`pixel_format <vis-qlvm-property-sweep-video-ffmpeg>`
     - ``yuv420p``
     - ffmpeg pixel format
   * - :ref:`crf <vis-qlvm-property-sweep-video-ffmpeg>`
     - ``18``
     - constant rate factor

.. _vis-qlvm-property-sweep-video-window:

``window``
   Share of the USVs in each equal-count window.

   :Options: a share of the USVs (0.1 = 10 %)
   :Default: ``0.1``

.. _vis-qlvm-property-sweep-video-step:

``step``
   Window-centre step between frames, as a share of the USVs.

   :Options: a positive share of the USVs
   :Default: ``0.01``, so the sweep moves one percentile per frame

.. _vis-qlvm-property-sweep-video-hold:

``hold``
   Frames held at the first and at the last window.

   :Options: a number of frames
   :Default: ``8``

.. _vis-qlvm-property-sweep-video-fps:

``fps``
   Video frame rate, frames per second.

   :Options: a positive number of frames per second
   :Default: ``15``

.. _vis-qlvm-property-sweep-video-dpi:

``dpi``
   Frame resolution.

   :Options: a positive integer
   :Default: ``100``
   :Notes: the frame is ``figure_size`` inches.

.. _vis-qlvm-property-sweep-video-density-bandwidth:

``density_bandwidth``
   Gaussian standard deviation of each window's density, in torus units.

   :Options: a positive number of torus units
   :Default: ``0.01`` (= 2 pixels of the 200 × 200 category grid)

.. _vis-qlvm-property-sweep-video-colormap-floor:

``colormap_floor``
   Lowest point of ``figures.sequential_cmap`` used.

   :Options: a number in [0, 1)
   :Default: ``0.2``
   :Notes: the density is drawn as brightness on black, so the black low end of a colormap such as inferno would hide the lowest windows.

.. _vis-qlvm-property-sweep-video-color-percentiles:

``color_percentiles``
   Percentiles of all USVs' values that span each property's colour scale.

   :Options: two percentiles, low and high
   :Default: ``[1, 99]``, so a few extreme USVs do not compress it

.. _vis-qlvm-property-sweep-video-brightness-percentile:

``brightness_percentile``
   Percentile of a window's own density that maps to full brightness.

   :Options: a percentile
   :Default: ``99``, so a single dense pixel does not dim the rest

.. _vis-qlvm-property-sweep-video-figure-size:

``figure_size``
   Frame size in inches.

   :Options: width and height in inches
   :Default: ``[19.0, 11.4]`` (19 × 11.4, the 2 × 3 panel grid with square maps)

.. _vis-qlvm-property-sweep-video-figure-face:

``figure_face``
   Hex page colour.

   :Options: a hex colour string
   :Default: ``#FFFFFF``

.. _vis-qlvm-property-sweep-video-boundary-color:

``boundary_color``
   Hex colour of the dotted category boundaries.

   :Options: a hex colour string
   :Default: ``#FFFFFF``, visible on the black density background

.. _vis-qlvm-property-sweep-video-mark-shade:

``median_mark_color`` / ``window_shade_color``
   Hex colours of the colour-bar mark at the window's median and of the shading over the window's range.

   :Options: a hex colour string each
   :Default: ``#000000`` (mark) and ``#FFFFFF`` (shading, drawn at 35 % opacity)

.. _vis-qlvm-property-sweep-video-colorbar-sizes:

``colorbar_label_size`` / ``colorbar_tick_size``
   Colour-bar label and tick font sizes, in points.

   :Options: a font size in points each
   :Default: 17 / 14

.. _vis-qlvm-property-sweep-video-ffmpeg:

``codec`` / ``pixel_format`` / ``crf``
   ffmpeg encoder settings.

   :Options:
      * ``codec`` — the ffmpeg encoder
      * ``pixel_format`` — the output pixel format
      * ``crf`` — the constant rate factor
   :Default: ``libx264``, ``yuv420p`` for players that only decode 4:2:0, and constant rate factor 18, visually lossless for x264


Modeling figures
--------------------------------
``visualizations/modeling_plots.py`` draws the modeling figures (univariate rankings and filters, multinomial performance and filters, the manifold filter atlas, the collinearity / timescale audits, the CNN results visualizer and the behavioural-response figures). It is not driven from the GUI; its plotters are called from the modeling notebook (see :doc:`Notebooks`). Besides the shared ``figures`` block (dpi, seed, format, colormaps) and the sex / social palettes, it reads its own tunables from the ``modeling_plots`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*. The plotters read these knobs from the settings when they are called, not as keyword arguments: the module functions read the shipped file (loaded once at import), and ``DeepResultsVisualizer`` reads the ``visualization_settings`` dict it is constructed with (the shipped file when ``None``). To draw with a different value, change it in the settings (or, for the CNN visualizer, in the dict passed to it):

.. code-block:: json

    "modeling_plots": {
      "colors": {
        "neutral": "#D3D3D3",
        "mean_line": "#DCB400",
        "text": "#202020",
        "reference_line": "#808080",
        "timescale_axis": "#000000",
        "timescale_null": "#808080"
      },
      "ci_percentiles": [0.5, 99.5],
      "raw_feature_difference": {
        "value_sanity_cap": 90.0,
        "n_bootstraps": 100,
        "heatmap_cmap": "binary",
        "heatmap_clip_percentiles": [1, 99]
      },
      "univariate_multinomial_performance": {
        "base_cmap": "mako"
      },
      "multinomial_filters": {
        "null_band_percentile": 95
      },
      "manifold_filter_atlas": {
        "n_time_slices": 16,
        "display_bins": 25,
        "smooth_sigma": 3.0,
        "atlas_grid_n": 12,
        "null_percentile": 95
      },
      "collinearity_audit": {
        "outline_threshold": 0.7,
        "outline_color": "#000000"
      },
      "timescale_audit": {
        "signal_smooth_window": 0,
        "acf_run_length": 15
      },
      "behavioral_response_forest": {
        "alpha": 0.01,
        "false_discovery_rate": 0.05,
        "axis_limits": [[-15.0, 15.0], [-5.0, 5.0], [-6.0, 6.0], [-2.0, 2.0]],
        "duration_sd_factor": 2.25
      },
      "matched_divergence": {
        "alpha": 0.01,
        "display_seconds": [-2.0, 2.0],
        "panel_columns": 4,
        "between_block_gap": 0.3,
        "onset_run_seconds": 1.0
      },
      "cnn_permutation_test": {
        "null_color": "#D3D3D3",
        "threshold_color": "#000000",
        "title_fontsize": 10,
        "label_fontsize": 8,
        "figsize": [3, 4],
        "n_bootstraps": 1000,
        "threshold_percentiles": [0.5, 99.5],
        "histogram_bins": 30,
        "skill_null_bins": 50,
        "skill_null_range": [-0.02, 0.02]
      },
      "cnn_feature_importance": {
        "title_fontsize": 10,
        "label_fontsize": 8,
        "figsize": [4, 6]
      },
      "cnn_spatial_precision_grid": {
        "n_patches": 20
      },
      "cnn_error_landscape": {
        "title_fontsize": 10,
        "label_fontsize": 8,
        "figsize": [12, 5]
      },
      "cnn_regional_saliency": {
        "null_color": "#D3D3D3",
        "figsize": [18, 9],
        "inset_histogram_bins": 15
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`colors <vis-modeling-plots-colors>`
     - see JSON above
     - six hex colours of the module palette
   * - :ref:`ci_percentiles <vis-modeling-plots-ci-percentiles>`
     - ``[0.5, 99.5]``
     - lower / upper percentile
   * - :ref:`raw_feature_difference <vis-modeling-plots-raw-feature-difference>`
     - see JSON above
     - sanity cap, bootstraps, heatmap colormap and clip
   * - :ref:`univariate_multinomial_performance <vis-modeling-plots-univariate-multinomial-performance>`
     - ``base_cmap`` ``mako``
     - a sequential colormap name
   * - :ref:`multinomial_filters <vis-modeling-plots-multinomial-filters>`
     - ``null_band_percentile`` 95
     - percentile
   * - :ref:`manifold_filter_atlas <vis-modeling-plots-manifold-filter-atlas>`
     - see JSON above
     - filmstrip slices, display bins, smoothing, atlas grid, null percentile
   * - :ref:`collinearity_audit <vis-modeling-plots-collinearity-audit>`
     - see JSON above
     - outline threshold and colour
   * - :ref:`timescale_audit <vis-modeling-plots-timescale-audit>`
     - see JSON above
     - smoothing window and ACF run length, in lag bins
   * - :ref:`behavioral_response_forest <vis-modeling-plots-behavioral-response-forest>`
     - see JSON above
     - error rate, FDR level, axis limits, duration SD factor
   * - :ref:`matched_divergence <vis-modeling-plots-matched-divergence>`
     - see JSON above
     - error rate, display window, panel layout, onset run
   * - :ref:`cnn_permutation_test <vis-modeling-plots-cnn-permutation-test>`
     - see JSON above
     - colours, fonts, size, bootstraps, thresholds, histograms
   * - :ref:`cnn_feature_importance <vis-modeling-plots-cnn-feature-importance>`
     - see JSON above
     - fonts and size
   * - :ref:`cnn_spatial_precision_grid <vis-modeling-plots-cnn-spatial-precision-grid>`
     - ``n_patches`` 20
     - integer > 0
   * - :ref:`cnn_error_landscape <vis-modeling-plots-cnn-error-landscape>`
     - see JSON above
     - fonts and size
   * - :ref:`cnn_regional_saliency <vis-modeling-plots-cnn-regional-saliency>`
     - see JSON above
     - null colour, size, inset histogram bins

.. _vis-modeling-plots-colors:

``colors``
   The module palette.

   :Options:
      * ``neutral`` — non-significant fills, separators and italic "not significant" notes
      * ``mean_line`` — the mean-line accent
      * ``text`` — axis and annotation ink
      * ``reference_line`` — dashed zero / chance lines
      * ``timescale_axis`` — zero lines and spines of the timescale audit
      * ``timescale_null`` — the circular-shift null fill / envelope of the timescale audit
   :Default: ``neutral`` ``#D3D3D3``, ``mean_line`` ``#DCB400``, ``text`` ``#202020``, ``reference_line`` ``#808080``, ``timescale_axis`` ``#000000``, ``timescale_null`` ``#808080``

.. _vis-modeling-plots-ci-percentiles:

``ci_percentiles``
   The lower / upper percentiles of every 99% confidence band drawn around a filter or a bootstrapped mean (the univariate filter figures and the raw feature-difference average).

   :Options: a ``[lower, upper]`` pair of percentiles
   :Default: ``[0.5, 99.5]``

.. _vis-modeling-plots-raw-feature-difference:

``raw_feature_difference``
   The raw feature-difference figures.

   :Options:
      * ``value_sanity_cap`` — per-frame values above it become NaN before the mean / CI so extreme outliers do not bias the estimate; in the feature's own unit, which suits angular features in degrees -- set a feature-appropriate cap for non-angular features, or ``null`` to disable it
      * ``n_bootstraps`` — bootstrap resamples of the averaged profile's 99% band
      * ``heatmap_cmap`` — the colormap of the sorted per-event heatmaps
      * ``heatmap_clip_percentiles`` — the percentiles of the pooled values that set the heatmap's colour limits, so a few extreme events do not wash out the rest
   :Default: ``value_sanity_cap`` 90.0, ``n_bootstraps`` 100 (the value the modeling notebook draws with), ``heatmap_cmap`` ``binary``, ``heatmap_clip_percentiles`` ``[1, 99]``

.. _vis-modeling-plots-univariate-multinomial-performance:

``univariate_multinomial_performance``
   The univariate multinomial performance figure.

   :Options: ``base_cmap`` — the sequential colormap of the Actual and Null proportion matrices
   :Default: ``base_cmap`` ``mako``

.. _vis-modeling-plots-multinomial-filters:

``multinomial_filters``
   The multinomial filter figures.

   :Options: ``null_band_percentile`` — the percentile of ``|w|`` over the permuted-target refits below which a class-by-lag cell is faded as no-signal
   :Default: ``null_band_percentile`` 95

.. _vis-modeling-plots-manifold-filter-atlas:

``manifold_filter_atlas``
   The manifold filter atlas.

   :Options:
      * ``n_time_slices`` — instants sampled per affinity filmstrip
      * ``display_bins`` — equal-width bins the magnitude line is averaged into
      * ``smooth_sigma`` — Gaussian smoothing of the magnitude line, in display bins; 0 disables it
      * ``atlas_grid_n`` — the vocal-space atlas holds ``atlas_grid_n ** 2`` real USV tiles
      * ``null_percentile`` — the dashed permuted-target magnitude line, as a percentile
   :Default: ``n_time_slices`` 16, ``display_bins`` 25, ``smooth_sigma`` 3.0, ``atlas_grid_n`` 12, ``null_percentile`` 95 (95th percentile)

.. _vis-modeling-plots-collinearity-audit:

``collinearity_audit``
   The collinearity audit.

   :Options:
      * ``outline_threshold`` — minimum ``|rho|`` for an off-diagonal cell to be outlined
      * ``outline_color`` — the outline's colour
   :Default: ``outline_threshold`` 0.7, ``outline_color`` ``#000000``

.. _vis-modeling-plots-timescale-audit:

``timescale_audit``
   The timescale audit.

   :Options:
      * ``signal_smooth_window`` — the rolling-mean window (lag bins) applied to each cross-correlation curve before the horizon is read; 0 is no smoothing
      * ``acf_run_length`` — the number of consecutive lag bins above the ACF null envelope that marks where the autocorrelation leaves the band, used by both timescale figures
   :Default: ``signal_smooth_window`` 0 (no smoothing), ``acf_run_length`` 15 (about 100 ms at 150 fps)

.. _vis-modeling-plots-behavioral-response-forest:

``behavioral_response_forest``
   The behavioural-response forest plot.

   :Options:
      * ``alpha`` — two-sided error rate of the plotted intervals
      * ``false_discovery_rate`` — Benjamini-Hochberg level deciding marker fill
      * ``axis_limits`` — four fixed ``[low, high]`` x-ranges -- contrast percent, contrast degrees, duration percent, duration degrees -- so the two questions read on comparable scales
      * ``duration_sd_factor`` — bout-length factor for one SD of log duration, used in the axis heading
   :Default: ``alpha`` 0.01, ``false_discovery_rate`` 0.05, ``axis_limits`` as in the JSON above, ``duration_sd_factor`` 2.25

.. _vis-modeling-plots-matched-divergence:

``matched_divergence``
   The matched-divergence figure.

   :Options:
      * ``alpha`` — two-sided error rate of the intervals and per-bin tests
      * ``display_seconds`` — window drawn, in seconds
      * ``panel_columns`` — features per block
      * ``between_block_gap`` — spacer between blocks, as a fraction of a panel
      * ``onset_run_seconds`` — consecutive significant time before an onset is reported, in seconds
   :Default: ``alpha`` 0.01, ``display_seconds`` ``[-2.0, 2.0]`` s, ``panel_columns`` 4, ``between_block_gap`` 0.3, ``onset_run_seconds`` 1.0 s

.. _vis-modeling-plots-cnn-permutation-test:

``cnn_permutation_test``
   The CNN permutation-test figure.

   :Options:
      * ``null_color`` / ``threshold_color`` — the null distribution and the threshold lines
      * ``title_fontsize`` / ``label_fontsize`` — in pt
      * ``figsize`` — in inches
      * ``n_bootstraps`` — resamples per distribution
      * ``threshold_percentiles`` — the 0.5th percentile of the null error and the 99.5th of the null skill, the one-sided 0.5% lines
      * ``histogram_bins`` / ``skill_null_bins`` — histogram bins
      * ``skill_null_range`` — the null skill's histogram range, centred on zero skill
   :Default: ``null_color`` ``#D3D3D3``, ``threshold_color`` ``#000000``, ``title_fontsize`` / ``label_fontsize`` 10 / 8 pt, ``figsize`` ``[3, 4]`` in, ``n_bootstraps`` 1000, ``threshold_percentiles`` ``[0.5, 99.5]``, ``histogram_bins`` 30, ``skill_null_bins`` 50, ``skill_null_range`` ``[-0.02, 0.02]``

.. _vis-modeling-plots-cnn-feature-importance:

``cnn_feature_importance``
   The CNN feature-importance figure.

   :Options: ``title_fontsize`` / ``label_fontsize`` (pt) and ``figsize`` (in)
   :Default: ``title_fontsize`` / ``label_fontsize`` 10 / 8 pt, ``figsize`` ``[4, 6]`` in

.. _vis-modeling-plots-cnn-spatial-precision-grid:

``cnn_spatial_precision_grid``
   The CNN spatial-precision grid.

   :Options: ``n_patches`` — the number of K-means neighbourhoods on a euclidean manifold
   :Default: ``n_patches`` 20
   :Notes: a torus run passes ``grid_shape`` instead.

.. _vis-modeling-plots-cnn-error-landscape:

``cnn_error_landscape``
   The CNN error-landscape figure.

   :Options: ``title_fontsize`` / ``label_fontsize`` (pt) and ``figsize`` (in)
   :Default: ``title_fontsize`` / ``label_fontsize`` 10 / 8 pt, ``figsize`` ``[12, 5]`` in

.. _vis-modeling-plots-cnn-regional-saliency:

``cnn_regional_saliency``
   The CNN regional-saliency figure.

   :Options:
      * ``null_color`` — the null colour
      * ``figsize`` — in inches
      * ``inset_histogram_bins`` — bins of the prediction-error inset
   :Default: ``null_color`` ``#D3D3D3``, ``figsize`` ``[18, 9]`` in, ``inset_histogram_bins`` 15


Neuronal decoding figures
--------------------------------
``visualizations/make_neuronal_decoding_figures.py`` draws the vocal-bout occupancy decode from
the per-session archives written by ``decode_usv_bout_occupancy`` (see :doc:`Analyze`). It is not
driven from the GUI; it is built in the notebook (see :doc:`Notebooks`).

``make_bout_decoding_figure`` draws one focal area's accuracy against unit count for every session
named, beside exactly three comparison panels. Opacity carries how much each animal vocalised, and
a filled marker means that rung cleared its **own** shift null — significance is per curve rather
than read off a shared band, which could not exist here, since the null's 99.5th percentile spans
0.05–0.10 across sessions.

``summarize`` gathers the per-session decoder tables into one frame, which is the step the analysis
deliberately leaves to the caller, since it pools nothing across sessions.
