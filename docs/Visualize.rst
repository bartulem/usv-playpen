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

* **visualize_booleans** : the on / off switch of every GUI-exposed visualization — ``make_neuronal_tuning_figures_bool``, ``make_behavioral_videos_bool``, ``make_usv_spectrograms_bool``, ``make_qlvm_torus_traversal_video_bool`` and ``make_embedding_thumbnails_bool``. The GUI's Yes / No toggles write them; ``Visualizer.visualize_data`` runs the first three per listed root directory and the two cohort-level ones (the torus-traversal video, the embedding thumbnails) once, after the per-session loop.
* **visualize_data.root_directories** : the session root directories of the *Root directories* field above (one per row), the default list ``Visualizer`` runs on when no list is passed in.
* **credentials_directory** : the directory holding ``email_config.ini``, the credentials of the start / completion e-mails.
* **send_email** : the e-mail notification block — ``visualizations_pc_list`` / ``visualizations_pc_choice`` (the lab PC named in the "PC is busy" / "PC is available again" messages), ``experimenter`` (named in the messages and passed to the behavioral videos as the experimenter id) and ``send_message.receivers`` (the recipients; with the list empty no e-mail is sent). The completion e-mail lists every visualization that failed.

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

* **save_directory** : default output directory for figures that aren't written next to their source data (e.g. cross-session anatomy plots). Per-figure code may override this with an explicit ``out_dir`` argument; per-cluster ratemaps and other session-bound figures always stay next to the data.
* **fig_format** : default output file format. For the per-cluster ratemap PDFs, ``pdf`` produces a single multi-page document; ``png`` / ``jpg`` / ``svg`` write one file per page.
* **dpi** : default raster resolution applied to every ``fig.savefig`` callsite that goes through ``visualizations.figure_io.save_figure``. ``fig_format``, ``dpi``, ``seed`` and ``sequential_cmap`` are read directly by the neuronal tuning figures: a settings file missing any of them raises ``KeyError`` instead of drawing at a substituted value.
* **timestamp_in_name** : when ``true``, ``_YYYYMMDD_HHMMSS`` is appended to figure stems by default. Session-bound figures opt out of this with ``timestamp_in_name=false`` since their filenames already embed a session id or unit id.
* **sequential_cmap** : the sequential (non-negative) colormap used by every heatmap / ratemap / spectrogram / density callsite (one of ``viridis``, ``cividis``, ``plasma``, ``inferno``, ``magma``) — the per-cluster ratemaps and neuronal-tuning densities, the ``make_behavioral_videos`` spectrogram subplot, the ``qlvm_torus_traversal_video`` spectrogram tiles, the USV-summary heatmaps, the vocal-space atlas of real USVs in ``plot_manifold_filter_atlas``, the QLVM figures of ``qlvm_figures`` (maps, fields and spectrograms) and the property colour scales of the ``qlvm_property_sweep_video`` (used from its colormap floor upward). This is the colormap the GUI's colormap dropdown edits.
* **diverging_cmap** : the white-centred diverging colormap for **signed** modeling-filter figures (``BrBG_r``: teal for negative, white at zero, yellow-brown for positive; chosen so neither side can be read as the male blue or the female red) — the selection-trajectory difference maps and the ``e(theta).W`` torus affinity fields in the manifold filter atlas (``plot_manifold_filter_atlas``). Any Matplotlib diverging map (``RdBu_r``, ``coolwarm``, ``bwr``, …).
* **seed** : random seed for figure-side stochastic operations (the embedding-thumbnail sampling, the QLVM figures' draws, folds and spirals), for reproducibility.

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

Colour palettes
~~~~~~~~~~~~~~~
Top-level ``*_colors`` blocks in *visualizations_settings.json* hold the semantic
(data-meaning) colours shared across figures; every entry is a hex string, so a
whole figure family can be recoloured in one place. Sex / emitter colours are
lists (index ``0`` primary, ``1`` secondary) while the grouped palettes are named
maps.

* **male_colors** / **female_colors** / **unassigned_colors** : per-emitter USV colours (male / female / no-emitter-assigned), used by the USV-timeline, spectrogram-overlay, embedding-scatter and modeling figures. An emitter's sex is always read from the ``Subjects`` block of the session's ``*_metadata.yaml`` (the ``subject_id`` matching the emitter / track name, ``yaml_utils.extract_animal_sexes``), never from the slot its track occupies, so same-sex sessions are coloured correctly; a tracked animal the metadata cannot resolve raises.
* **session_condition_styles** : per-condition ``color`` and ``label`` of the session types (courtship intact partners, courtship mute female, female-female, male-male, isolated male), used by the per-session squeak-timing heatmap (``plot_session_squeak_time_heatmap``): female-female takes ``female_colors[0]``, male-male ``male_colors[0]``, courtship intact partners the channel-wise mean of the two (``#CD928A``) and courtship mute female the same at 50 % opacity; isolated male is ``#2A9D8F``. Colours may carry an alpha channel (8-digit hex).
* **social_colors** : the single colour for social / dyadic features (e.g. the social-feature ratemaps and the timescale-audit "social" trace).
* **brain_area_colors** : per-brain-region palette (a seven-bucket map: ``PAG`` / ``MRN`` / ``VTA`` / ``SC`` / ``CENT`` / ``MB`` / ``other``) shared by the behavioral-video overlays and the anatomy / tuning figures.
* **cell_type_colors** : the mono-grey triad for the per-mouse cell-type stacked bars in the anatomy figures.
* **coactivity_colors** : the ``group_a`` / ``group_b`` / ``null`` / ``threshold`` colours for the neuronal-coactivity figures (also the defaults the coactivity notebook may override inline).
* **component_colors** : the per-mixture-component colour cycle for the inter-USV-interval mixture-model figures (paired in code with distinct line styles).
* **corner_colors** : the arena-corner marker colours (``North`` / ``West`` / ``South`` / ``East``) for the behavioral-video arena overlay.

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

These blocks hold the analysis choices, sizes, colours and statistical levels of particular figures (structural grid geometry and line-style cycles stay in the code):

* **neuronal_tuning_figures** : the bout panels of the per-unit vocal page -- ``bout_session_bin_seconds`` (the whole-session rate trace's bins, 1 s), ``bout_zoom_seconds`` / ``bout_zoom_bin_seconds`` (the densest-window zoom's width and bins, 20 s and 50 ms), ``bout_smooth_bins`` (the Gaussian smoothing of both traces, in bins, 2), ``bout_trigger_window_seconds`` / ``bout_trigger_bin_seconds`` / ``bout_trigger_smooth_bins`` (the bout-onset-triggered raster's window, bins and smoothing: -1 to 2 s, 20 ms, 2.5 bins), ``bout_density_grid`` (the first-versus-later scatter's density grid, 160 per side) and ``bout_density_levels`` (the bout shares its contours enclose, at most three: 0.5, 0.75, 0.9).
* **neuronal_tuning_figures** (styling and population figures) : the neutral inks shared by every tuning page and population figure -- ``ink_color`` (``#1F1F1F``, near-black text / traces / identity lines), ``band_color`` (``#D3D3D3``, the shuffle-CI band), ``dash_color`` (``#888888``, dashed references and "no data" notes), ``light_color`` (``#CCCCCC``, unassigned-USV bars and their legend) and ``hatch_color`` (``#BBBBBB``, hatch edges); figure sizes in inches -- ``vocal_page_figsize_inches`` (16 x 27, the one-page vocal layout), ``behavioral_page_width_inches`` / ``behavioral_row_height_inches`` (16 wide, 2.6 per feature row, at least one row: sized so 8 / 9 pt fonts fit), ``region_grid_figsize_inches`` (14 x 6.4, every 2 x 4 per-region grid), ``sign_flip_figsize_inches`` (10 x 5.4), ``pag_gradient_figsize_inches`` (18 x 4.8, three projections side by side), ``peth_timing_figsize_inches`` (18 x 6.6, the 2 x 7 grid), ``behavioral_tier_figsize_inches`` (7.5 x 3.5), ``overlap_venn_figsize_inches`` (6.5 x 5.5) and ``per_region_venn_figsize_inches`` (18 x 9); the vocal page's ``bout_legend_fontsize`` (9 pt) and ``bout_legend_line_step`` (0.075 axes fraction between the "bouts" / "unassigned USVs" labels), ``peth_tick_seconds`` (0.5 s PETH tick spacing, anchored on onset), ``qlvm_category_label_fontsize`` (14 pt) and ``qlvm_label_inset`` (0.06 of the panel span a category label is held inside the border so it is not clipped); the category strip's ``strip_log_ratio_threshold`` (10: a max / min rate range above it switches the axis to symlog) and ``strip_symlog_linthresh`` (0.5 sp/s linear region); marker-size maps -- ``stability_marker_size_range`` (4-70 square points) over ``stability_fr_log10_range`` (-1 to 1.5, i.e. 0.1-31.6 sp/s, the band the bulk of units fall in), ``consistency_marker_size_range`` (6-60 square points) over ``consistency_n_tested_range`` (2-6 tested sessions, the span in this dataset) and ``breadth_marker_size_range`` (8-60 square points) over ``breadth_z_range`` (peak z 3-20, so strong outliers do not dominate); and the population-figure statistics -- ``stability_min_units`` (5: fewer units leave a region's cross-condition r undrawn), ``vmi_bootstrap_n`` (2000 resamples) with ``vmi_bootstrap_quantiles`` (0.025 / 0.975, a 95% CI), ``vmi_n_tested_min`` (2 tested sessions for the consistency and sign-flip figures), ``pag_kde_grid_resolution`` (220 points per axis of the PAG density), ``vmi_histogram_bins`` (20 bins over -1..1), ``peth_consistency_tolerance_seconds`` (0.1 s full width within which a unit's peak times must agree), ``peth_timing_bin_seconds`` (0.05 s histogram bins), ``consistency_k_min`` (2 agreeing significant sessions) and ``consistency_require_majority`` (true: those sessions must also be a strict majority of the significant ones). The ``make_*`` population methods read these instead of taking them as arguments.
* **neuronal_decoding_figures** : the vocal-bout occupancy decoding figure -- ``trace_color`` (``#1F1F1F``, marker edges and text) and ``page_color`` (``#FFFFFF``, the page the opacity ramp blends toward and the face of an unfilled marker); ``opacity_range`` (0.45-1.0, the blend weight from the least to the most vocal session, so every curve stays legible); geometry in inches -- ``main_panel_inches`` (5.6 x 4.0), ``side_gap_inches`` (0.44, between the main panel and the comparison column), ``panel_gap_inches`` (0.2, between the three stacked comparison panels) and ``page_margin_inches`` (left 0.78, right 0.14, bottom 0.6, top 0.74); and the statistics drawn -- ``ci_quantile`` (0.995, the Student-t quantile of a two-sided 99% error bar) and ``null_percentile`` (99.5, the percentile of a rung's own shift null its mean must exceed for a filled marker).
* **anatomy_figures** : ``excluded_mouse_ids``, mice left out of the anatomy figures (insufficient yield, used in no downstream analysis), and the rendering knobs of ``AnatomyFigureMaker`` (an argument the caller leaves at ``None`` falls back to the same-named key here) -- ``recording_yield.fig_size_inches`` (7.0 x 2.6 in, a Nature-style single-column band); ``unit_positions`` for the 3D unit-positions still / builder: ``fig_size_inches`` (6.0 x 5.0 in), ``view_elev`` / ``view_azim`` (18 / -60 degrees), ``shell_vertex_stride`` (every 6th root-mesh vertex in the brain-shell scatter), ``shell_point_size`` / ``shell_point_alpha`` (0.7 square points, 0.5), ``region_alpha`` (0.1 face alpha of the PAG / MRN / VTA / CENT / SC volumes), ``mb_alpha`` (0.04, kept low because MB is the parent volume and would otherwise hide PAG / MRN / VTA), ``dot_size`` / ``dot_alpha`` (9 square points, 1.0) and ``legend_marker_scale`` (3x the in-axes dot size for legible swatches); ``unit_positions_video`` for the rotating video: ``n_frames`` (240, a full 360 degree sweep), ``fps`` (30 frames/s, so 8 s), ``dpi`` (600 per frame) and ``bitrate`` (4000 kbps for the ffmpeg writer); ``unit_waveforms`` for the per-probe waveform figure: ``lateral_jitter_um`` (12 um half-range so units sharing a peak channel do not overlap), ``waveform_width_um`` (12 um per waveform), ``peak_amplitude_target_um`` (25 um, the axial height the largest unit's peak maps to), ``inter_shank_wspace`` (0.05, the visible inter-shank break), ``opacity_sigma_um`` (20 um Gaussian sigma of the opacity fall-off with distance from the peak channel), ``n_neighbors_each_side`` (5 same-shank channels above and below the peak), ``ap_padding_um`` / ``dv_padding_um`` (20 / 30 um axis padding) and ``fig_size_inches`` (4.5 x 7.0 in).
* **behavioral_feature_distributions** : the per-session feature-histogram PDF that ``compute_behavioral_features`` writes next to each behavioral-features CSV (``*_behavioral_features_histograms.pdf``) -- ``figure_dpi`` (600, the resolution the figure is laid out and rasterised at; the PDF pages themselves are saved at ``figures.dpi``), ``panels_per_row`` (6 feature panels per row of the grid), ``figure_width_inches`` (6.4) and ``row_height_inches`` (1.0 per row of panels, so a page is ``6.4 x n_rows`` in).
* **usv_summary_figures** : the figures of ``usv_summary_statistics`` -- ``density_clip_percentile`` (98: every polar KDE is colour-capped at this percentile so a few dense pixels do not wash out the map), ``colorbar_nbins`` (5 colorbar ticks on the polar KDE), ``distance_kde_bw_adjust`` (1.5, the seaborn bandwidth multiplier of the distance-by-assignment KDE, smoother than Scott's rule for the sparse unassigned group), ``anova_alpha`` (0.05, the ANOVA level that gates the Tukey HSD and the Tukey level itself); ``polar_kde`` (``grid_points`` 200 per axis and ``contour_levels`` 100 of the single-pair polar KDE), ``category_polar_kde_grid`` / ``estrous_category_kde_grid`` (``grid_points`` 100, ``contour_levels`` 50 and ``panel_inches`` 4 x 4 per panel of the two grid figures); ``panel_inches`` (per-panel inches of the category x estrous rate grid, 6 x 4.5, and ratio grid, 5 x 5) and ``figsizes`` (the figure size in inches of every fixed-layout figure, keyed by the plotting function's name without ``plot_``; ``category_estrous_ratio_grid_stages`` is the 2 x 2 per-stage companion of the ratio grid).
* **usv_interval_figures** : the figures of ``usv_interval_summary_statistics`` (the inter-USV interval notebook) -- ``edge_color`` (``#202020``, histogram edges, the selected-K outline, component triangles and the LRT reference line); ``log_histogram`` (``figsize`` 5 x 5 in, ``alpha`` 0.5 so the two overlaid pools stay readable); ``ic_curves.figsize`` (6 x 4 in); ``best_fit`` (``figsize`` 5 x 5 in, square for the Q-Q inset, ``qq_quantiles`` 200 Q-Q points, ``density_grid_points`` 500 -- the grid the point-wise mixture curve of ``plot_gmm_fit`` is evaluated on -- ``bin_mean_grid_points`` 4000 -- the denser grid its bin-averaged curve is smoothed on when ``density_as_bin_means`` is on -- and ``component_grid_points`` 500 and ``component_linewidth`` 1.5 for the optional per-component curves; the histogram's bin count is the caller's, the notebook passing the per-sex ``bins_per_sex``); ``qq_plot`` (the standalone ``plot_qq`` figure: ``figsize`` 5 x 5 in, ``n_quantiles`` 200 Q-Q points and ``line_color`` ``#202020`` for the y = x reference line); ``bootstrap_lrt`` (``figsize_per_panel`` 4 x 3 in, ``break_gap_factor`` 1.5 -- the axis is broken when the observed statistic sits more than 1.5 null-widths past the null -- ``bins`` 50, ``null_color`` ``#a0a0a0`` and ``observed_color`` ``#cc0000``); ``serial_dependence`` (``figsize`` 6.2 x 2.6 in, ``grid_bins`` 70 per axis of the 2-D histogram, ``lims_ms`` 15-20000 ms on both axes, and the inks ``density_dark_color`` ``#1F3A45`` (dark end of the density ramp), ``diagonal_color`` ``#8A8F98`` (next = current), ``boundary_color`` ``#B4404A`` (bout boundary), ``band_color`` ``#1F1F1F`` (bootstrap band) and ``line_color`` ``#000000`` (fitted line and bend)).
* **coactivity_figures** : the figures of ``make_coactivity_figures`` (the coactivity notebook; colours stay in ``coactivity_colors``) -- ``significance_alpha`` (0.05, the two-tailed level that colours an animal's slope line and puts a star over an amplitude bin); ``acoustic_confound`` (``bins`` 40 and ``alpha`` 0.5 of the overlaid feature histograms); ``null_distributions`` (the 3 x 2 chained-null grid: ``percentile`` 99 of the dashed threshold, ``bins`` 40, ``null_alpha`` 0.5, ``obs_linewidth`` 3.0, ``threshold_linewidth`` 1.0, ``legend_fontsize`` 8) and ``per_session_pop_corr`` (the smaller per-session panels: ``percentile`` 99, ``bins`` 30, ``null_alpha`` 0.55, ``obs_linewidth`` 2.2, ``threshold_linewidth`` 0.8, ``legend_fontsize`` 7).

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

* **arena_directory** : path to the directory with the 3D tracked arena data
* **speaker_audio_file** : path to the audio file containing the playback speaker sound
* **pitch_shifted_audio_bool** : if "Yes", automatically frequency-shift the session USVs over the chosen ``[video_start_time, video_start_time + video_duration]`` window into the human-audible range and mux the result onto the video (under a time limit; a failed mux is reported as a warning) (replaces the former manual ``sequence_audio_file`` path)
* **pitch_shifted_audio_specs** : pitch-shift recipe used when ``pitch_shifted_audio_bool`` is "Yes" (``fs_audio_dir``, ``fs_device_id``, ``fs_channel_id``, ``fs_wav_sampling_rate``, ``fs_octave_shift``, ``fs_volume_adjustment``, ``fs_compand_transfer``, ``fs_noise_reduction_std_threshold``, ``fs_sinc_upper_cutoff_hz``, ``fs_sox_timeout_s``; the same keys as the *Analyze* page's ``frequency_shift_audio_segment`` block). ``fs_sox_timeout_s`` is the upper bound in seconds on each SoX subprocess of the pitch shift (``600``): a SoX run on a video-length window takes seconds, so one running this long has hung and is stopped rather than blocking the render
* **animate_bool** : boolean value indicating whether to animate the figure or not ("No" creates figure)
* **video_start_time** : start time of the figure/video in seconds
* **video_duration** : duration of the video in seconds
* **plot_theme** : "dark" or "light" plot background
* **save_fig** : if True, the figure will be saved in the *data_animation_examples* subdirectory
* **view_angle** : "top" or "side" view of social behavior in the playpen arena
* **side_azimuth_start** : azimuth angle of the side view (in deg)
* **rotate_side_view_bool** : rotate the side view or not (NB: angles wrap around)
* **rotation_speed** : rotation speed of the side view (in deg/s)
* **history_bool** : plot the location history of one body node
* **speaker_bool** : plot the playback speaker
* **spectrogram_bool** : plot the spectrogram of the audio segment
* **spectrogram_ch** : channel of the audio segment to plot
* **raster_plot_bool** : plot the live spiking raster of the neural data
* **raster_selection_criteria** : criteria for selecting the neurons to plot in the raster
* **raster_selection_criteria (brain_areas)** : list of brain areas to include in the raster plot
* **raster_selection_criteria (other)** : list of other criteria to include in the raster plot (e.g., "good" for unit type)
* **raster_special_units** : unit(s) to highlight in the raster plot (*e.g.*, "imec0_cl0000_ch361")
* **spike_sound_bool** : make spike sound each time the highlighted unit spikes (the sound is muxed onto the video with ffmpeg under a time limit; a mux that fails raises and keeps the spike-sound WAV so it can be redone by hand)
* **beh_features_bool** : plot the behavioral features dynamics subplot
* **beh_features_to_plot** : list of behavioral features in the subplot
* **special_beh_features** : list of highlighted behavioral features in the subplot

Parameters controlling the figure output format and video encoding (``general_figure_specs``):

* **fig_format** : output image format for still frames (e.g. "png").
* **animation_codec** : FFMPEG video codec for the rendered animation (e.g. "h264_nvenc").
* **animation_codec_preset_flag** : codec preset / speed-quality flag (e.g. "p5").
* **animation_codec_tune_flag** : codec tuning flag (e.g. "hq").
* **animation_writer** : matplotlib animation writer (e.g. "ffmpeg").
* **animation_format** : output video container format (e.g. "mp4").

Parameters setting the ink palette of each ``plot_theme`` (``color_modes``; one map per theme, ``light_mode`` for "light" and ``dark_mode`` for "dark", every value a hex string):

* **background_color** : figure and panel background ("#FFFFFF" light, "#202020" dark).
* **node_edge_color** / **body_edge_color** : edge colour of the body-node spheres and of the body polygons ("#8B8B8B" in both themes, a mid grey that reads on either background).
* **text_color** / **tick_color** : text, tick and spine colour (the opposite of the background: "#202020" light, "#FFFFFF" dark).
* **arena_line_color** / **arena_mic_color** / **arena_mesh_color** : arena axes, microphone markers and mesh walls (same ink as the text).
* **spectrogram_text_color** : spectrogram labels and colorbar text (same ink as the text).
* **faded_feature_color** : trace colour of a partner-pair (dyadic) behavioral feature that is not among the highlighted ``special_features`` (8-digit hex with alpha ``33``, about 20% opacity, so the highlighted traces stand out: "#00000033" light, "#FFFFFF33" dark); dyadic features otherwise take ``text_color``.
* **speaker_color** : the playback speaker marker (same ink as the text).

Parameters specific to the arena figure include:

* **arena_node_connections_bool** : plots connections between corner and nearest microphones
* **arena_axes_lw** : line width of the arena axes
* **arena_mics_lw** : line width of the microphones
* **arena_mics_opacity** : opacity of the microphones
* **plot_corners_bool** : plot different color spheres in corners of the arena
* **corner_size** : size of the corner spheres
* **corner_opacity** : opacity of the corner spheres
* **plot_mesh_walls_bool** : plot the mesh walls of the arena
* **mesh_opacity** : opacity of the mesh walls
* **active_mic_bool** : plots the active microphone (whose spectrogram is shown)
* **inactive_mic_bool** : plots the inactive microphones (whose spectrograms are not shown)
* **inactive_mic_color** : color of the inactive microphones
* **text_fontsize** : font size of the text in the arena figure
* **speaker_opacity** : opacity of the playback speaker

Parameters specific to the mouse figure include:

* **node_bool** : plot mouse body nodes as spheres
* **node_size** : size of the body node spheres
* **node_opacity** : opacity of the body node spheres
* **node_lw** : line width of the body node spheres
* **node_connection_lw** : plots connections between body nodes
* **body_opacity** : opacity of the body polygons connected with nodes
* **history_point** : plot history of particular body point
* **history_span_sec** : time span of the history in seconds (**will fail if history is set to start before tracking!**)
* **history_ls** : line style of the history plot (e.g., "-", "--", "-.", ":")
* **history_lw** : line width of the history plot

Parameters specific to subplots include:

* **beh_features_window_size** : time window of the behavioral features subplot (in s, **will fail if is set beyond tracking boundaries!**)
* **raster_window_size** : time window of the raster subplot (in s, **will fail if is set beyond tracking boundaries!**)
* **raster_lw** : horizontal line width of spikes in the raster plot
* **raster_ll** : vertical line length of spikes in the raster plot
* **spectrogram_cbar_bool** : plot spectrogram colorbar
* **spectrogram_plot_window_size** : time window of the spectrogram subplot (in s, **will fail if is set beyond tracking boundaries!**)
* **spectrogram_power_limit** : lower and upper limits of the spectrogram colorbar (in dB)
* **spectrogram_frequency_limit** : lower and upper limits of the spectrogram frequency axis (in Hz)
* **spectrogram_yticks** : y-axis ticks of the spectrogram (in Hz)
* **spectrogram_stft_nfft** : window size for the spectrogram calculation
* **plot_usv_segments_bool** : plot the Deep Audio Segmenter (DAS)-detected USV segments in the spectrogram
* **usv_segments_ypos** : y-axis position of the USV segments in the spectrogram (in Hz)
* **usv_segments_lw** : line width of the USV segments in the spectrogram

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

* **fps** : output frame rate.
* **dpi** : raster resolution of each rendered frame.
* **m** : number of nearest-neighbour USVs shown in the concentric rings around each category peak (Part 1).
* **cluster_hold_frames** : frames each category peak is held on screen (Part 1).
* **peak_traverse_frames** : frames spent on each peak-to-peak walk (Part 2).
* **boundary_traverse_frames** : frames spent on each boundary-crossing walk (Part 3).
* **title_card_frames** : frames each of the three title cards is shown.
* **samples_per_trace** : number of positions revealed along each peak-to-peak walk (one nearest neighbour each).
* **peak_jitter_sigma** : random jitter (torus units) applied to the peak-walk trajectory so repeated paths do not overlap.
* **boundary_curve_amplitude** : curvature amplitude of the Part-3 boundary walks.
* **boundary_positions_per_walk** : number of positions revealed along each boundary walk.
* **boundary_neighbors** : nearest neighbours shown per boundary position (the Part-3 grid rows).
* **seed** : RNG (random-number-generator) seed for the walks / jitter (reproducible renders).
* **peaks_only** : when ``true``, render Part 1 (category peaks) only.
* **spec_cache_size** : number of spectrograms held in the in-memory LRU cache during rendering.
* **apply_mask** : apply the SAM2 mask to each spectrogram.
* **accent_color** : hex highlight color for the trail / head marker / cluster outline + dot / current-tile borders.
* **palette** : the video's neutral hex colours -- ``white`` (title-card face and the category contours on the density, ``#FFFFFF``), ``black`` (title-card title and the marker edges, ``#000000``), ``gray`` (borders of the shown tiles, ``#808080``), ``light_gray`` (borders of the empty tiles, ``#D3D3D3``) and ``subtitle`` (the title-card subtitle, ``#444444``); neutral so the accent colour stays the only highlight.
* **heatmap_vmax_percentile** : percentile of the non-zero category-bundle density that tops the left panel's colour scale (95, so a few peak pixels do not wash the rest of the map out).
* **bitrate** : ffmpeg bitrate of an ``.mp4`` output, in kbit/s (4000; a ``.gif`` is written with Pillow and ignores it).

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

* **save_dir** : output directory; empty routes the figure to the session's ``data_animation_examples`` folder.
* **save_fig** : whether to write the figure to disk.
* **fig_format** : output file format (``png`` / ``pdf`` / ``svg`` / ``jpg``).
* **fig_dpi** : raster resolution.
* **fig_size** : figure size in inches (``[width, height]``).
* **transparent_fig_bg** : save the figure on a transparent background.
* **mode** : the figure type the ``make_usv_spectrograms`` class renders. For this figure it is **pinned to** ``sequence`` — the GUI toggle overwrites it, so it is not a choice here. It becomes a free selection only when the plotter is driven from ``usv_general_analyses.ipynb`` (see :doc:`Notebooks`), where the other values render *different* figures (``single`` / ``all`` = raw per-channel spectrograms; ``stitched`` = the averaged session-timeline spectrogram).
* **channel_of_interest** : microphone channel for the ``single`` mode (unused by ``sequence``, which auto-selects the loudest channel over the window).
* **plot_raw_audio** : overlay the raw-audio amplitude trace on the stitched spectrogram.
* **usv_amplitude_color** : hex color of that raw-audio amplitude trace.
* **time_window** : ``[start, end]`` seconds of the window (the GUI presents it as a start + duration pair and writes it back).
* **freq_limits** : ``[low, high]`` kHz frequency axis of the spectrogram.
* **nfft** : FFT window size for the spectrogram.
* **plot_cbar** : whether to draw the colorbar.
* **cbar_limits** : ``[vmin, vmax]`` colorbar (power) limits in dB.
* **apply_mask** : apply the SAM2 mask to each per-USV spectrogram before stitching.
* **auto_open_figure** : open the saved figure in the OS viewer when running in a GUI context.
* **stitched_edge_taper_alpha** : shape of the raised-cosine (Tukey) taper applied along each per-USV spectrogram's time axis before it is stitched onto the zero canvas (``stitched`` mode): the tapered fraction of the USV, start and end combined (``0.3`` fades the first and last 15%), so each USV fades into the black gaps instead of stepping from a bright edge column.

plus the ``sequence`` sub-dict:

* **draw_boundaries** : overlay the category bundle's boundaries on the left panel (regular map only).
* **annotate_right** : annotate the stitched (right) spectrogram.
* **mark_usv_segments** : draw an emitter-colored bar along the top of the spectrogram for each USV.
* **exclude_squeaks** : keep only the pure USVs (``usv`` true and ``squeak`` false) in the sequence, leaving out pure squeaks, segments holding both and unclassed rows (default ``true``; same meaning as ``embedding_thumbnails.exclude_squeaks``). Noise segments are dropped regardless of this setting.
* **marker_size_range** : ``[min, max]`` marker area (points²) of the left-panel USV points, scaled linearly with USV duration from the shortest to the longest USV of the window (``[35.0, 70.0]``: modest, so longer USVs read bigger without dominating; a window of equal durations takes the midpoint).
* **line_width_range** : ``[min, max]`` width (points) of the connecting line's segments, scaled with the silent gap between consecutive USVs up to the Tukey upper fence of the gaps (``[0.5, 3.0]``; a longer silence draws a thicker segment).
* **embedding_cmap** : matplotlib colormap of the regular map's density heatmap behind the points (``gray_r``, white to black, so the emitter colours stay legible on it).
* **embedding_vmax_percentile** : percentile of the non-zero density pixels used as the heatmap's upper colour limit (``95``, so the few densest pixels saturate instead of washing out the rest of the map).

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

The layout / sampling knobs live in the ``embedding_thumbnails`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (the figure DPI and the sampling seed come from the general ``figures`` block). Core keys:

* **exclude_squeaks** : keep only the segments ``detect-usv-squeaks`` classed as pure USVs (``usv`` true and ``squeak`` false) in the scatter, the four small maps and the thumbnails, leaving out pure squeaks, segments holding both a squeak and a USV (``usv`` and ``squeak`` both true) and unclassed rows (default ``true``). The USV QLVM models were trained on USVs only, so squeak-bearing segments sit wherever the decoder places them. A pooled embeddings table without the ``usv`` / ``squeak`` booleans (built from summaries ``detect-usv-squeaks`` has not classified) raises; a cache written before the booleans existed is rebuilt automatically.
* **n_samples_per_category** : number of thumbnail spectrograms sampled per category.
* **tile_orientation** : thumbnail grid orientation — ``vertical`` / ``horizontal``.
* **apply_mask** : apply the SAM2 mask to each thumbnail spectrogram.
* **mask_excluded_categories** : categories whose thumbnails are shown WITHOUT the SAM2 mask even when ``apply_mask`` is on (for vocal types SAM2 masks poorly); every category stays in the grid.
* **category_colors** : optional explicit per-category color map, ``{"1": "#RRGGBB", ...}`` keyed by the ``qlvm_category`` number (``null`` = auto).
* **scatter_max_points** : downsampling cap for the embedding scatter.
* **scatter_point_size** : marker area of each scatter point (points², ``4.0``; the four small auxiliary maps use a quarter of it, at least ``0.5``), small because the cohort scatter holds up to ``scatter_max_points`` USVs.
* **scatter_point_alpha** : opacity of the scatter points (``0.5``), so the dense cores of the map read darker than its fringes.
* **continuous_color_percentiles** : lower / upper percentiles of the finite values that set the colour limits of the duration and bandwidth auxiliary maps (``[2, 98]``, so a few extreme USVs do not compress the colour range of the rest).

Sampling / boundaries:

* **sampling_method** : how thumbnails are drawn per category (e.g. ``spiral``).
* **draw_cluster_boundaries** : overlay the category bundle's boundaries on the scatter (regular map only).

Spiral overlay:

* **draw_spiral_overlay** : draw the sampling-spiral overlay on the scatter.
* **spiral_show_only_for** : restrict the overlay to one category (``null`` = all).
* **spiral_color** : hex color of the spiral.
* **spiral_linewidth** : spiral line width.
* **spiral_radius_scale** / **spiral_radius_abs** : spiral radius — relative to each cluster's farthest USV, or (``spiral_radius_abs``, when not ``null``) an absolute radius in torus units shared by every cluster (shipped ``0.03``, so the thumbnails stay close to each cluster's peak).
* **spiral_n_turns** : number of spiral turns.
* **spiral_random_phase** : randomize the spiral's starting angle.

Annotations / layout:

* **annotate_picks_on_scatter** : number each sampled pick on the scatter.
* **pick_number_fontsize** : font size of those pick numbers.
* **annotate_cluster_ids** : label each cluster with its id.
* **cluster_id_fontsize** : font size of the cluster-id labels.
* **thumbnail_hspace** / **thumbnail_wspace** : vertical / horizontal spacing between thumbnails.
* **unstretched_specs** : zero-pad every thumbnail to the longest pick's duration (USV centred, equal borders) so all thumbnails share one time scale; ``false`` stretches each USV to fill its slot.
* **fig_size** : figure size in inches (``[width, height]``).

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

The three notebook-driven pooled summary helpers (``plot_usv_property_histograms``, ``plot_session_type_usv_counts``, ``plot_session_usv_timeline``; see :doc:`Notebooks`) take the whole settings dict as ``visualizations_parameter_dict``: their DPI is ``figures.dpi``, their sex / unassigned colours are ``male_colors[0]`` / ``female_colors[0]`` / ``unassigned_colors[0]``, and the rest lives in the ``usv_property_figures`` block:

* **histograms.fig_size** : figure size in inches of the five-panel property-histogram row (``[15.0, 3.0]``, 3 in per panel).
* **histograms.n_bins** : linearly spaced bins per panel over the property's FeatureZoo range (``36``).
* **histograms.face_color** / **histograms.edge_color** : histogram fill and outline (``#808080`` grey fill, ``#000000`` outline).
* **histograms.bandwidth_bimodal_split_khz** : the frequency bandwidth (kHz) splitting the bimodal ``freq_bandwidth_hz`` distribution; that panel marks the histogram peak on each side of it instead of the mean (``30.0`` kHz, between the narrowband and broadband modes).
* **histograms.bandwidth_peak_smooth_sigma_bins** : Gaussian smoothing (in bins, not kHz) applied before those two peaks are located (``3.0``, just enough to suppress sample-noise peaks near the split).
* **session_type_counts.fig_size** : figure size in inches of the session-type bar chart (``[6.0, 3.0]``).
* **session_type_counts.male_female_color** : colour of the male-female bar (``#DFB8B3``, the 50/50 blend of the male and female colours lightened 35% toward white); the lone-male and female-female bars take ``male_colors[0]`` / ``female_colors[0]``.
* **timeline.fig_size** : figure size in inches of the per-session USV timeline (``[7.5, 1.6]``, a wide, short strip).
* **timeline.rectangle_height** : vertical extent of each USV rectangle in data units (``2.5``; the y-axis is hidden, so it only sets the strip's visual thickness).
* **timeline.rectangle_linewidth** : outline width of each USV rectangle in points (``1.0``).

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

Every tunable lives in the ``vocal_pose_figures`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*. The keys a run normally changes:

* **window.end_time** : the frame shown, in seconds from video start; **window.history_seconds** the spectrogram / time-bar span before it; **window.trail_end_seconds** the age at which the trail has faded to nothing (0 keeps it for the whole window).
* **spectrogram.microphone** : ``"peak"`` or a device-channel name such as ``"s04"``; **spectrogram.freq_range_khz** / **freq_ticks_khz** the band shown and its labels; **db_floor** ``"adaptive"`` (``adaptive_floor_above_median_db`` above the band's median power) or a dB value; **n_fft** / **hop_fraction** the STFT (a shorter window widens narrowband sweeps); **male_deep** / **female_deep** / **ramp_split** the darker end of each ramp; **ceiling_percentile** / **gamma** / **vocal_pad_seconds** the loudness scaling.
* **camera.azimuth** : ``"male_side"`` or degrees; **camera.elevation** degrees above the floor (90 is straight down); **camera.top_sex** the animal drawn last, on top where they overlap.
* **trail** : ``strongest`` (colour strength just behind the present), ``fade_seconds`` (time constant), ``floor`` (strength old silhouettes settle to), ``taper_fraction`` (the last fraction of the trail over which it tapers to nothing).
* **skeleton** : ``line_width`` / ``node_size`` at the reference zoom (``reference_half_extent_m``), ``body_opacity`` / ``trail_body_opacity`` of the filled bodies, ``margin_m`` / ``z_margin_m`` around the trail, ``history_point``.
* **scale_mark** : ``length_cm`` (0 = none), ``labels``, ``signs`` (−1 reverses an arm), ``turn_degrees`` (turns the two floor arms about the vertical), ``gap_cm`` from the male, ``line_width``, label and caption placement, ``font_size``, ``color``.
* **surfaces** : ``enabled``, ``color``, ``junction_color`` / ``junction_width``, ``wall_height_m``, ``content_tolerance`` (what counts as drawn against the surface colour when cropping), ``pad_inches``.
* **layout** : figure width, the spectrogram's left / width / height, the bar, fonts and ink, ``content_threshold`` / ``crop_edge_pixels`` / ``svg_pad_inches`` for the crops, ``box_zoom`` / ``fit_shrink`` / ``max_fit_passes`` (the pose box is shrunk until nothing is cut off), ``align_pose``, ``color_key`` / ``key_position`` (``"pose"`` = wide swatches beside the animals, ``"spectrogram"`` = squares to its right) and the swatch sizes.
* **output** : ``dpi`` and ``save_svg``.
* **video** : ``duration_seconds``, ``speed``, ``follow_seconds`` (−1 = fixed camera), ``degrees_per_second``, render ``inches`` / ``dpi`` / ``crop_edge_pixels`` / ``workers``, the ffmpeg ``codec`` / ``pixel_format`` / ``crf`` / ``encode_timeout_seconds``, its own ``trail`` and ``skeleton`` sub-blocks, and ``spectrogram`` (``enabled``, ``layout``, ``reach_seconds``, ``sharp_seconds``, ``fade_seconds``, ``floor``, the inset's size, corner and shift, and the panel text room).
* **traces** : ``enabled``; ``width_inches`` of the panel and its ``left_gap_inches`` / ``right_gap_inches`` / ``top_inches`` / ``bottom_inches``; ``span_seconds`` each side of the frame; ``columns`` (``"both"``, ``"male"`` or ``"female"``); ``egocentric`` (feature names, top to bottom), ``dyadic`` (per sex, the same number of features, the name's order giving the side, e.g. ``allo_yaw-head`` for the male and ``head-allo_yaw`` for the female), ``labels`` (feature -> printed name); ``fade_power``, ``line_width_centre`` / ``line_width_end``, ``y_pad_fraction``; the panel's division into ``label_fraction``, ``column_gap_fraction``, ``row_gap_fraction``, ``label_gap_fraction``; the range numbers' ``value_decimals`` (``small`` / ``medium`` / ``large`` for magnitudes split at ``value_decimal_thresholds``) and ``value_font_scale``; ``font_size``; the frame line ``zero_tick`` / ``zero_tick_width`` / ``zero_tick_alpha``; ``time_label`` and its ``time_label_offset`` (in row heights below the last row).
* **candidates** : ``window_seconds``, ``step_seconds``, ``n_shown``, ``max_gap_cm`` and the candidate sheet's size.
* **view_picker** : how many trail silhouettes the page draws and their fade.

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

* **n_usvs** : most cohort USVs drawn for the map panels (the overview, and the density and content-change fields of the category figure).
* **usvs_per_session** : most USVs per session in the spectrogram sample.
* **sample_cache_file** : ``.npz`` cache of the spectrogram sample (empty: no cache). It is written to a temporary file and renamed into place, so an interrupted run never leaves a truncated cache to be served next time.
* **seed_cell_directories** : model cells of other trainings of the regular map's recipe (other seeds), with which the quality figure's readout-per-seed panel re-embeds the sample; empty skips the panel. The panel has one colour per training (four, the production cell included), so at most three cells can be listed; more raises ``ValueError`` instead of leaving trainings out. These validation cells are kept outside the production package, so the shipped paths name them explicitly.
* **watershed_levels_file** : ``.npz`` of the density-watershed partitions of the regular map (its ``final_coarse``, ``k4`` and ``final_fine`` grids: 3, 4 and 6 clusters) that the category figure's partition panel compares the categories with; empty compares the categories alone.
* **category_colors** : one hex colour per category (R-1 … R-k in order).
* **overview** : ``n_examples`` spectrograms per category; ``spiral_radius`` (torus units) and ``spiral_turns`` of the example spiral; ``density_bandwidth`` (torus units) of every density field of the figures.
* **quality** : ``n_pcs`` PCA components (the first two are the PCA map, all of them the UMAP input); ``umap_n_neighbors`` / ``umap_min_dist``; ``readout_neighbors`` / ``readout_folds`` of the acoustic readout; ``neighborhood_k`` / ``n_neighborhood_subsets`` / ``neighborhood_subset_size`` of the neighborhood scores; ``distance_sample_size`` / ``n_distance_samples`` of the distance fidelity; ``pullback_grid`` nodes per axis of the geodesic grids; ``embed_lattice_batch_size`` / ``embed_data_batch_size`` of the seed-cell embedding.
* **style** : the look of the three figures. ``text_color`` (all text and reference lines, ``#000000``), ``figure_face`` (page, ``#FFFFFF``), ``boundary_color`` (category boundaries and numbers on the fields, and the strong cells' text of tables (i) and (iii), ``#FFFFFF``), ``property_boundary_color`` (boundaries over the property-coloured maps, ``#000000``); ``map_colors`` (one per compared map, keyed ``QLVM`` / ``PCA`` / ``UMAP``), ``distance_colors`` (one per distance measure of panel (b'), keyed by the measure's label), ``seed_colors`` (one per training in panel (c), the production cell included, so their count caps the trainings: four), ``null_color`` (shuffled-position readout, ``#D62728``), ``density_color`` / ``change_color`` (the two measures of the along-boundary test, ``#4E79A7`` / ``#E15759``), ``bar_color`` (property-share and category-partition bars, ``#59595B``), ``watershed_color`` (watershed-partition bars, ``#BDBDBD``) and ``group_dot_color`` (partition group dots, ``#FFFFFF``); font sizes in points -- ``title_size`` / ``label_size`` / ``tick_size`` (14 / 14 / 12) and ``colorbar_tick_size`` / ``colorbar_label_size`` (13 / 15) of the overview and category figures, ``quality_title_size`` / ``quality_label_size`` / ``quality_tick_size`` (16 / 16 / 14) of the quality figure's summary panels; ``color_percentiles`` (``[1, 99]``: the percentiles of every map's and field's colour range, so a few extreme USVs or pixels do not compress the scale); and the figure sizes in inches, ``overview_figsize`` (30 × 12.5), ``quality_figsize`` (20 × 22.5) and ``categories_figsize`` (32.5 × 16.6), sized so the panels of each figure keep their aspect.
* **categories** : ``n_shifts`` of the along-boundary test and ``histogram_bins`` of its histogram; ``similarity_downsample`` / ``similarity_max_shift`` of the aligned correlation; ``balanced_per_category`` / ``balanced_reference_per_category`` of the between-category panels; ``closest_matches`` and ``closest_log2_limit`` of the closest-matches panel; ``partition_sample_size`` / ``partition_reference_size`` of the partition comparison; ``average_size`` / ``align_max_shift`` / ``align_iterations`` / ``average_time_ms`` of the aligned category means.

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

The ``qlvm_property_sweep_video`` block (the seed of the tie order is ``figures.seed`` and the colormap ``figures.sequential_cmap``; the cohort directory, cache, output path and preview quantile are run choices set in the notebook cell):

* **window** : share of the USVs in each equal-count window (0.1 = 10 %).
* **step** : window-centre step between frames, as a share of the USVs (0.01, so the sweep moves one percentile per frame).
* **hold** : frames held at the first and at the last window (8).
* **fps** : video frame rate, frames per second (15).
* **dpi** : frame resolution (100; the frame is ``figure_size`` inches).
* **density_bandwidth** : Gaussian standard deviation of each window's density, in torus units (0.01 = 2 pixels of the 200 × 200 category grid).
* **colormap_floor** : lowest point of ``figures.sequential_cmap`` used, in [0, 1) (0.2): the density is drawn as brightness on black, so the black low end of a colormap such as inferno would hide the lowest windows.
* **color_percentiles** : percentiles of all USVs' values that span each property's colour scale (``[1, 99]``, so a few extreme USVs do not compress it).
* **brightness_percentile** : percentile of a window's own density that maps to full brightness (99, so a single dense pixel does not dim the rest).
* **figure_size** : frame size in inches (19 × 11.4, the 2 × 3 panel grid with square maps).
* **figure_face** : hex page colour (``#FFFFFF``).
* **boundary_color** : hex colour of the dotted category boundaries (``#FFFFFF``, visible on the black density background).
* **median_mark_color** / **window_shade_color** : hex colours of the colour-bar mark at the window's median (``#000000``) and of the shading over the window's range (``#FFFFFF``, drawn at 35 % opacity).
* **colorbar_label_size** / **colorbar_tick_size** : colour-bar label and tick font sizes, in points (17 / 14).
* **codec** / **pixel_format** / **crf** : ffmpeg encoder settings (``libx264``, ``yuv420p`` for players that only decode 4:2:0, and constant rate factor 18, visually lossless for x264).

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


Modeling figures
--------------------------------
``visualizations/modeling_plots.py`` draws the modeling figures (univariate rankings and filters, multinomial performance and filters, the manifold filter atlas, the collinearity / timescale audits, the CNN results visualizer and the behavioural-response figures). It is not driven from the GUI; its plotters are called from the modeling notebook (see :doc:`Notebooks`). Besides the shared ``figures`` block (dpi, seed, format, colormaps) and the sex / social palettes, it reads its own tunables from the ``modeling_plots`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*. The plotters read these knobs from the settings when they are called, not as keyword arguments: the module functions read the shipped file (loaded once at import), and ``DeepResultsVisualizer`` reads the ``visualization_settings`` dict it is constructed with (the shipped file when ``None``). To draw with a different value, change it in the settings (or, for the CNN visualizer, in the dict passed to it):

* **colors** : the module palette -- ``neutral`` (``#D3D3D3``, non-significant fills, separators and italic "not significant" notes), ``mean_line`` (``#DCB400``, the mean-line accent), ``text`` (``#202020``, axis and annotation ink), ``reference_line`` (``#808080``, dashed zero / chance lines), ``timescale_axis`` (``#000000``, zero lines and spines of the timescale audit) and ``timescale_null`` (``#808080``, the circular-shift null fill / envelope of the timescale audit).
* **ci_percentiles** : the lower / upper percentiles of every 99% confidence band drawn around a filter or a bootstrapped mean (the univariate filter figures and the raw feature-difference average), ``[0.5, 99.5]``.
* **raw_feature_difference** : ``value_sanity_cap`` (per-frame values above it become NaN before the mean / CI so extreme outliers do not bias the estimate; 90.0, in the feature's own unit, which suits angular features in degrees -- set a feature-appropriate cap for non-angular features, or ``null`` to disable it), ``n_bootstraps`` (bootstrap resamples of the averaged profile's 99% band, 100 -- the value the modeling notebook draws with), ``heatmap_cmap`` (the colormap of the sorted per-event heatmaps, ``binary``) and ``heatmap_clip_percentiles`` (the percentiles of the pooled values that set the heatmap's colour limits, ``[1, 99]``, so a few extreme events do not wash out the rest).
* **univariate_multinomial_performance** : ``base_cmap``, the sequential colormap of the Actual and Null proportion matrices (``mako``).
* **multinomial_filters** : ``null_band_percentile``, the percentile of ``|w|`` over the permuted-target refits below which a class-by-lag cell is faded as no-signal (95).
* **manifold_filter_atlas** : ``n_time_slices`` (instants sampled per affinity filmstrip, 16), ``display_bins`` (equal-width bins the magnitude line is averaged into, 25), ``smooth_sigma`` (Gaussian smoothing of the magnitude line, in display bins, 3.0; 0 disables it), ``atlas_grid_n`` (the vocal-space atlas holds ``atlas_grid_n ** 2`` real USV tiles, 12) and ``null_percentile`` (the dashed permuted-target magnitude line, 95th percentile).
* **collinearity_audit** : ``outline_threshold`` (minimum ``|rho|`` for an off-diagonal cell to be outlined, 0.7) and ``outline_color`` (``#000000``).
* **timescale_audit** : ``signal_smooth_window``, the rolling-mean window (lag bins) applied to each cross-correlation curve before the horizon is read, 0 (no smoothing); ``acf_run_length``, the number of consecutive lag bins above the ACF null envelope that marks where the autocorrelation leaves the band, 15 (about 100 ms at 150 fps), used by both timescale figures.
* **behavioral_response_forest** : ``alpha`` (two-sided error rate of the plotted intervals, 0.01), ``false_discovery_rate`` (Benjamini-Hochberg level deciding marker fill, 0.05), ``axis_limits`` (four fixed ``[low, high]`` x-ranges -- contrast percent, contrast degrees, duration percent, duration degrees -- so the two questions read on comparable scales) and ``duration_sd_factor`` (bout-length factor for one SD of log duration, used in the axis heading, 2.25).
* **matched_divergence** : ``alpha`` (two-sided error rate of the intervals and per-bin tests, 0.01), ``display_seconds`` (window drawn, ``[-2.0, 2.0]`` s), ``panel_columns`` (features per block, 4), ``between_block_gap`` (spacer between blocks, as a fraction of a panel, 0.3) and ``onset_run_seconds`` (consecutive significant time before an onset is reported, 1.0 s).
* **cnn_permutation_test** : ``null_color`` (``#D3D3D3``), ``threshold_color`` (``#000000``), ``title_fontsize`` / ``label_fontsize`` (10 / 8 pt), ``figsize`` (``[3, 4]`` in), ``n_bootstraps`` (resamples per distribution, 1000), ``threshold_percentiles`` (``[0.5, 99.5]``: the 0.5th percentile of the null error and the 99.5th of the null skill, the one-sided 0.5% lines), ``histogram_bins`` (30), ``skill_null_bins`` (50) and ``skill_null_range`` (``[-0.02, 0.02]``, the null skill's histogram range, centred on zero skill).
* **cnn_feature_importance** : ``title_fontsize`` / ``label_fontsize`` (10 / 8 pt) and ``figsize`` (``[4, 6]`` in).
* **cnn_spatial_precision_grid** : ``n_patches``, the number of K-means neighbourhoods on a euclidean manifold (20; a torus run passes ``grid_shape`` instead).
* **cnn_error_landscape** : ``title_fontsize`` / ``label_fontsize`` (10 / 8 pt) and ``figsize`` (``[12, 5]`` in).
* **cnn_regional_saliency** : ``null_color`` (``#D3D3D3``), ``figsize`` (``[18, 9]`` in) and ``inset_histogram_bins`` (bins of the prediction-error inset, 15).

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
