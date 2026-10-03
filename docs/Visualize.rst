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

Plot neuronal tuning figures
----------------------------
Once the *Compute neuronal tuning curves* function from the *Analyze* section has completed, you have the ability to plot its results. Output is one combined multi-page document per cluster: a behavioral page per temporal offset and per plot-feature group (``individual.<mouse>`` and ``social``) followed by one vocal page per emitter: the bout raster + pooled peri-onset (ultrasonic vocalization) ``usv_peth`` on top, the ``usv_property_tuning`` continuous-property grid in the middle, and at the bottom the single ``usv_category_tuning`` row of the QLVM category ``qlvm_category`` (rate and occupancy maps over the category bundle's R-1 … R-k regions, ``os_utils.load_qlvm_category_bundle``, and the tuning-vs-shuffle strip), drawn on the property grid's columns so no page carries a lone figure row. The per-category PETH (peri-event time histogram, ``usv_category_peth``) stays in the pickle but is not drawn. 1D ratemaps are drawn as a line spanning the plot, colored by the per-mouse palette (or the social color for social features). The 99% CI (confidence interval) of the shuffled distribution is shown as a shaded band around the line.

To obtain this visualization, list the root directories of interest, select *Plot neuronal tuning figures* in the GUI and click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_1.png
   :align: center
   :alt: Visualize Step 1

.. raw:: html

   <br>

Running this function results in the population of the *tuning_curves* subdirectory with one combined output per cluster (PDF by default; configurable in the GUI / settings):

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
* **dpi** : default raster resolution applied to every ``fig.savefig`` callsite that goes through ``visualizations.figure_io.save_figure``.
* **timestamp_in_name** : when ``true``, ``_YYYYMMDD_HHMMSS`` is appended to figure stems by default. Session-bound figures opt out of this with ``timestamp_in_name=false`` since their filenames already embed a session id or unit id.
* **sequential_cmap** : the sequential (non-negative) colormap used by every heatmap / ratemap / spectrogram / density callsite (one of ``viridis``, ``cividis``, ``plasma``, ``inferno``, ``magma``) — the per-cluster ratemaps and neuronal-tuning densities, the ``make_behavioral_videos`` spectrogram subplot, the ``qlvm_torus_traversal_video`` spectrogram tiles, the USV-summary heatmaps, and the decoded-USV vocal-space atlas in ``plot_manifold_filter_atlas``. This is the colormap the GUI's colormap dropdown edits.
* **diverging_cmap** : the diverging (blue-white-red) colormap for **signed** modeling-filter figures — the selection-trajectory difference maps and the ``e(theta).W`` torus affinity fields in the manifold filter atlas (``plot_manifold_filter_atlas``). Any Matplotlib diverging map (``RdBu_r``, ``coolwarm``, ``bwr``, …).
* **seed** : random seed for figure-side stochastic operations, for reproducibility.

.. code-block:: json

    "figures": {
        "save_directory": "/mnt/falkner/Bartul/figures",
        "fig_format": "png",
        "dpi": 300,
        "timestamp_in_name": true,
        "diverging_cmap": "RdBu_r",
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

* **male_colors** / **female_colors** / **unassigned_colors** : per-emitter USV colours (male / female / no-emitter-assigned), used by the USV-timeline, spectrogram-overlay, embedding-scatter and modeling figures.
* **session_condition_styles** : per-condition ``color`` and ``label`` of the session types (courtship intact partners, courtship mute female, female-female, male-male, isolated male), used by the per-session squeak-timing heatmap (``plot_session_squeak_time_heatmap``): female-female takes ``female_colors[0]``, male-male ``male_colors[0]``, courtship intact partners the channel-wise mean of the two (``#CD928A``) and courtship mute female the same at 50 % opacity; isolated male is ``#2A9D8F``. Colours may carry an alpha channel (8-digit hex).
* **social_colors** : the single colour for social / dyadic features (e.g. the social-feature ratemaps and the timescale-audit "social" trace).
* **manifold_colors** : the two torus output-coordinate colours (manifold-x / manifold-y, index ``0`` / ``1``) for the last-bin manifold-filter bars; deliberately far from the male / female / social colours so a manifold axis never reads as an animal identity.
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
* **pitch_shifted_audio_bool** : if "Yes", automatically frequency-shift the session USVs over the chosen ``[video_start_time, video_start_time + video_duration]`` window into the human-audible range and mux the result onto the video (replaces the former manual ``sequence_audio_file`` path)
* **pitch_shifted_audio_specs** : pitch-shift recipe used when ``pitch_shifted_audio_bool`` is "Yes" (``fs_audio_dir``, ``fs_device_id``, ``fs_channel_id``, ``fs_wav_sampling_rate``, ``fs_octave_shift``, ``fs_volume_adjustment``, ``fs_compand_transfer``, ``fs_noise_reduction_std_threshold``, ``fs_sinc_upper_cutoff_hz``)
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
* **spike_sound_bool** : make spike sound each time the highlighted unit spikes
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
            "fs_sinc_upper_cutoff_hz": 25000
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
The ``usv_playpen.visualizations.qlvm_torus_traversal_video`` module renders a two-panel toroidal (doughnut-shaped) "torus walkthrough" animation (the in-house, torch-free port of ``qmc_deep_gen``'s ``inference_latents_video.py``). The **left** panel is the QLVM (the in-house quasi-Monte Carlo latent variable model) latent map of the regular map — the QLVM category bundle's density heatmap with its R-1 … R-k category contours (no axes/ticks) and a recency-coloured trajectory trail (cyan at the current position, fading to white going back, built with ``create_colormap``); the **right** panel is a phase-specific spectrogram board. All spectrograms have their SAM2 (Segment Anything Model 2) mask applied (``apply_mask``) and the call centred in its window with equal padding on both sides (so duration is preserved, not stretched). It runs in three parts, each introduced by a title card:

- **Part 1 — Category peaks**: one phase per category, the right panel showing the spectrogram nearest its peak (the category's label position in the bundle, the pixel farthest from its boundary) surrounded by its ``m`` nearest USVs in concentric rings; on the left the active category is outlined in a thick **pulsating cyan** contour with a cyan dot at its centre.
- **Part 2 — Peak-to-peak walks**: shortest-torus-path walks between random category peaks; the right 5×15 grid fills row-major with the nearest USV at each visited position (the current tile bordered in cyan).
- **Part 3 — Boundary crossings**: curved walks that wrap the torus edges/corners; the right grid's columns are trajectory positions and rows are nearest neighbours (the current trajectory tile bordered in cyan).

It is **cohort-level** — it reads the QLVM category bundle + the consolidated store, not a session directory — and is exposed both as the ``qlvm-torus-traversal-video`` CLI and in the GUI *Visualize* window (third column): "Render QLVM demo video" (Yes/No) and a **Video sampling rate (fps)** slider (the category grid has one level, so there is no clustering selector). A single **Spectrograms directory** **Browse** field lives in the left column under **Credentials directory** (the shared ``shared_resources.spectrograms_dir`` base, under which the consolidated store is resolved by convention).

To render it, list a root directory, set the *Spectrograms directory*, select *Render QLVM demo video*, choose the *Video sampling rate (fps)*, click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_3.png
   :align: center
   :alt: Visualize Step 3

.. raw:: html

   <br>

**Which QLVM map.** Every QLVM figure — this video, the USV sequence figure and the embedding thumbnails — draws ONE QLVM map, chosen by ``shared_resources.qlvm_map`` in */usv-playpen/_parameter_settings/visualizations_settings.json* (the GUI's **QLVM map (all QLVM figures)** selector at the top of the third column; the embedding explorer starts on it too). The maps are the production models of ``os_utils.QLVM_MAPS``: ``qlvm`` (the regular model, the default), ``qlvm_dur`` and ``qlvm_ent`` (the duration- and spectral-entropy-conditional models; all three masked and time-stretched, ``os_utils.QLVM_PRODUCTION_MODEL_CELLS``). A map ``P`` places calls at the summary columns ``P1`` / ``P2``. There is one category column, ``qlvm_category`` (R-1 … R-k, written by ``assign-qlvm-categories``): the categories are defined on the regular map and label the call, so every map's figure colours or groups calls by ``qlvm_category``.

**Where the category geometry comes from.** Every QLVM figure that draws category boundaries, places category centres or shows a density landscape — this video, the USV sequence figure, the embedding thumbnails, the embedding explorer, the neuronal tuning watersheds, the manifold filter atlas and the category-embedding panel — reads it from ONE place, the QLVM category bundle ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY`` (``/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/regions/clustering_clean/category_bundle``, written by ``build-qlvm-categories``; a module constant, not a setting, because the settings' experimenter folders are re-keyed to the active experimenter), through ``os_utils.load_qlvm_category_bundle``: its periodic ``label_grid`` (the partition the summaries' ``qlvm_category`` is assigned from), its smoothed corpus ``density`` and its category label positions (``category_nomenclature.json``). The bundle partitions the regular map's torus, so it is drawn on ``qlvm`` only: a figure of a conditional map draws no category boundaries (it colours the calls by their ``qlvm_category`` instead) and says so, and no figure estimates boundaries from the data. The earlier ``<spectrograms_dir>/qlvm_v3/<map>/arrays_{coarse,fine}.npz`` reference arrays are no longer read.

Inputs of the video:

- the QLVM category bundle (above) — the ``density`` background, the ``label_grid`` contours and the category label positions (the peaks / path waypoints). It is defined on the regular map, so ``shared_resources.qlvm_map`` must be ``qlvm`` for the video (any other map stops the run with a message).
- ``<spectrograms_dir>/spectrograms_*.h5`` (newest match) — the consolidated per-session spectrogram/SAM2 store. It supplies BOTH the per-USV latent coords (per-session ``qlvm/<key>/<qlvm_map>``, the map's ``P1`` / ``P2``) for the nearest-neighbour lookup AND the spectrograms (``spectrogram/<key>/spectrograms``) shown on the right. No latents pickle is read at render time, and coverage spans all sessions in the store.

**Prerequisite** — the consolidated H5 must carry the per-session ``qlvm/<key>/qlvm`` (n, 2) coordinates of the production regular cell the bundle is defined on, recorded in its ``qlvm_models/qlvm`` provenance (``package_root`` / ``cell``). The video checks the provenance first: a store of another cell (the current ``consolidate-spectrogram-store`` writes a v3 archive whose regular map is the v3 phase 6 cell, see :doc:`Process`) places the calls on another torus, where the bundle's regions would mislabel them, and is refused; a store without the coordinates raises a clear error naming the map.

Because it is cohort-level, the output is not written next to a session; the ``.mp4`` lands in the project-wide ``figures.save_directory`` with a render-timestamped name (``qlvm_torus_traversal_<qlvm_map>_<YYYYMMDD_HHMMSS>.mp4``) unless an explicit ``--output-path`` is given:

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
        "accent_color": "#00FFFF"
    }

Render a USV sequence figure
-----------------------------
``USVSpectrogramPlotter.plot_sequence`` (the ``'sequence'`` mode of ``make_usv_spectrograms``) renders a **per-session**, static two-panel figure of the USVs in a chosen ``[start, start + duration]`` window (seconds). It is wired into the GUI *Visualize* window (third column) under "Render USV sequence figure" and dispatched per session in ``visualize_data.py`` via ``make_usv_spectrograms_bool`` (enabling the GUI toggle sets ``make_usv_spectrograms.mode = 'sequence'``).

To render it, list a session root, select *Render USV sequence figure*, set the audio-sequence window and options, click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_4.png
   :align: center
   :alt: Visualize Step 4

.. raw:: html

   <br>

- **Left** — the precomputed cohort landscape of the QLVM map ``shared_resources.qlvm_map`` (see *Which QLVM map* above), drawn with no ticks/ticklabels. The window's USVs are placed at the map's ``P1`` / ``P2``, colored by emitter (``male_colors[0]`` / ``female_colors[0]`` / ``unassigned_colors[0]``), sized by call duration, numbered ``1..n`` in time order, and joined by a connecting line whose color runs white → male color along the bout and whose per-segment width tracks the inter-USV silent gap (``start`` of the next minus ``stop`` of the previous; the line takes the short wrap-around route across a torus edge when that is closer). On the regular map the panel draws the QLVM category bundle's gray_r density heatmap and, when ``draw_boundaries`` is on, the bundle's black category boundaries (R-1 … R-k on the unit torus; see *Where the category geometry comes from* above). On a conditional map the bundle does not describe the torus, so the panel is a bare (tick-free) square with a note saying the categories are defined on the regular map. A session whose ``usv_summary`` lacks the map's ``P1`` / ``P2`` raises a clear error.
- **Right** — ONE continuous spectrogram over the same window: the per-USV averaged spectrograms are SAM2-masked (when ``apply_mask``) and stitched at their true times onto a **black** background, so only the calls are lit and the gaps are black; an optional raw-audio trace can sit on top (``plot_raw_audio``), taken from the channel that is loudest across the window's USVs (the most-frequent per-USV ``peak_amp_ch``; raw waveforms are NOT averaged across mics — the per-mic phase delays would interfere destructively). Each USV can also be marked with a horizontal emitter-colored bar along the top of the spectrogram (``mark_usv_segments``). The left-panel numbers match the time order of the calls along the right time axis.

The GUI section leads with a **Save created figure in format** selector (writes ``make_usv_spectrograms.fig_format``) and a **Draw embedding boundaries** toggle; the map itself is the shared **QLVM map** selector.

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

plus the ``sequence`` sub-dict:

* **draw_boundaries** : overlay the category bundle's boundaries on the left panel (regular map only).
* **annotate_right** : annotate the stitched (right) spectrogram.
* **mark_usv_segments** : draw an emitter-colored bar along the top of the spectrogram for each USV.

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
        "sequence": {
            "draw_boundaries": true,
            "annotate_right": false,
            "mark_usv_segments": true
        }
    }

Render embedding thumbnails
---------------------------
The ``usv_playpen.visualizations.make_usv_spectrograms`` module's remaining cohort-level helper, ``plot_embedding_with_category_thumbnails``, is GUI-exposed (its pooled summary helpers — ``plot_usv_property_histograms``, ``plot_session_type_usv_counts``, ``plot_session_usv_timeline`` — are notebook-driven; see :doc:`Notebooks`):

- ``plot_embedding_with_category_thumbnails`` — a two-panel figure pairing a QLVM map scatter (the torus of ``shared_resources.qlvm_map``) — colored by the calls' ``qlvm_category`` and, on the regular map, overlaid with the category bundle's boundaries — against a per-category grid of spectrogram thumbnails sampled from the consolidated SAM2 + spectrogram store. Unlike the helpers above it is **cohort-level** and exposed in the GUI: enabling *Render embedding thumbnails* in the *Visualize* window (third column, with **clustering type borders** (fine: the one category level), **thumbnails per category**, **thumbnail layout**, **draw cluster boundaries**, **apply SAM2 mask** and **per-cluster sampling** selectors) pools every cohort session list under ``shared_resources.input_files_directory`` (playback lists excluded, as in the embedding explorer) and resolves the store from ``shared_resources.spectrograms_dir``, then runs ONCE (the same run-once dispatch as the QLVM torus video, via ``render_embedding_thumbnails_for_cohort``). Its layout / sampling knobs all live in the ``embedding_thumbnails`` settings block (documented below); the figure DPI and the sampling seed are taken from the general ``figures`` block (``dpi`` / ``seed``); the category boundaries and centers (for cluster-ID labels / spiral centers) are the category bundle's ``label_grid`` and label positions (row ``i`` = category ``i + 1``; a pooled ``qlvm_category`` outside ``1..k`` raises, since the summaries and the bundle would be different partitions), on the regular map only; the categories themselves are the summaries' ``qlvm_category`` on every map, so a conditional ``qlvm_map`` colours and groups its calls by ``qlvm_category`` with no boundaries, the spiral sampler walks without a boundary filter, and the centres are the per-category means of that map's calls (a message says so); and the cohort scatter is read from the precomputed pooled-embeddings cache ``<spectrograms_dir>/embeddings/pooled_embeddings_qlvmv3.parquet`` (``os_utils.POOLED_EMBEDDINGS_CACHE_NAME``; one parquet holding every QLVM map's coordinates and the ``qlvm_category`` labels, fingerprinted against the summaries it was pooled from and rebuilt when they change) — built once on a fast mount via ``build_pooled_embeddings_df`` so the figure does not re-read every session's ``usv_summary.csv``.

To render it, select *Render embedding thumbnails* in the *Visualize* window, choose the clustering / layout options (the map is the shared **QLVM map** selector), click *Next* and then *Visualize*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_5.png
   :align: center
   :alt: Visualize Step 5

.. raw:: html

   <br>

Being cohort-level, the figure is written to the project-wide ``figures.save_directory`` with a name built from the QLVM map and label column (plus a ``_YYYYMMDD_HHMMSS`` stamp when ``figures.timestamp_in_name`` is set):

.. parsed-literal::

    ├── ...  (the ``figures.save_directory`` cohort output folder)
    │   ├── **embedding_thumbnails_qlvm_category_20250430_145017.png**
    │   ...

The layout / sampling knobs live in the ``embedding_thumbnails`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (the figure DPI and the sampling seed come from the general ``figures`` block). Core keys:

* **category_col_suffix** : the label column that colors / groups the scatter and thumbnail rows — ``category`` (``qlvm_category``; the only level).
* **exclude_squeaks** : keep only the segments ``detect-usv-squeaks`` classed as pure USVs (``usv`` true and ``squeak`` false) in the scatter, the four small maps and the thumbnails, leaving out pure squeaks, segments holding both a squeak and a USV (``usv`` and ``squeak`` both true) and unclassed rows (default ``true``). The USV QLVM models were trained on USVs only, so squeak-bearing segments sit wherever the decoder places them. A pooled embeddings table without the ``usv`` / ``squeak`` booleans (built from summaries ``detect-usv-squeaks`` has not classified) raises; a cache written before the booleans existed is rebuilt automatically.
* **n_samples_per_category** : number of thumbnail spectrograms sampled per category.
* **tile_orientation** : thumbnail grid orientation — ``vertical`` / ``horizontal``.
* **apply_mask** : apply the SAM2 mask to each thumbnail spectrogram.
* **mask_excluded_categories** : categories to omit from the thumbnail grid.
* **category_colors** : optional explicit per-category color map (``null`` = auto).
* **scatter_max_points** : downsampling cap for the embedding scatter.

Sampling / boundaries:

* **sampling_method** : how thumbnails are drawn per category (e.g. ``spiral``).
* **draw_cluster_boundaries** : overlay the category bundle's boundaries on the scatter (regular map only).

Spiral overlay:

* **draw_spiral_overlay** : draw the sampling-spiral overlay on the scatter.
* **spiral_show_only_for** : restrict the overlay to one category (``null`` = all).
* **spiral_color** : hex color of the spiral.
* **spiral_linewidth** : spiral line width.
* **spiral_radius_scale** / **spiral_radius_abs** : spiral radius — relative to each cluster's farthest call, or (``spiral_radius_abs``, when not ``null``) an absolute radius in torus units shared by every cluster (shipped ``0.03``, so the thumbnails stay close to each cluster's peak).
* **spiral_n_turns** : number of spiral turns.
* **spiral_random_phase** : randomize the spiral's starting angle.

Annotations / layout:

* **annotate_picks_on_scatter** : number each sampled pick on the scatter.
* **pick_number_fontsize** : font size of those pick numbers.
* **annotate_cluster_ids** : label each cluster with its id.
* **cluster_id_fontsize** : font size of the cluster-id labels.
* **thumbnail_hspace** / **thumbnail_wspace** : vertical / horizontal spacing between thumbnails.
* **unstretched_specs** : keep thumbnails at their native aspect (no stretch).
* **fig_size** : figure size in inches (``[width, height]``).

.. code-block:: json

    "embedding_thumbnails": {
        "category_col_suffix": "category",
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
        "fig_size": [16, 12]
    }

Render vocal-pose figures
-------------------------
The ``usv_playpen.visualizations.make_vocal_pose_figures`` module renders one session's **vocal-pose still** and the matching **video**: what the animals were doing while the male sang. It is **notebook-driven** (the ``usv_general_analyses.ipynb`` *Vocal-pose figures* cells, see :doc:`Notebooks`), not exposed in the GUI, so it has no ``visualize_booleans`` switch; it reads the session's 3D tracks (``*_points3d_translated_rotated_metric.h5``), ``*_usv_summary.csv`` and ``audio/cropped_to_video`` wavs, takes the animal colours from the ``male_colors`` / ``female_colors`` / ``unassigned_colors`` palette and the arena from ``make_behavioral_videos.arena_directory``.

**The still** (``VocalPoseStillMaker.make_vocal_pose_still``) stacks two panels. On top is the spectrogram of one microphone over the window ending at ``window.end_time`` (``"peak"`` picks the microphone most of the window's calls peak on), every time column coloured by the emitter of the call covering it — white to the male's colour, white to the female's, white to the unassigned colour outside any listed call — with each emitter's USVs and squeaks scaled to their own loudest sound (a squeak is tens of dB louder than a USV) and a shared floor so silence stays white; the male ramp continues past his colour to a darker shade of the same hue (``male_deep``) so narrowband USVs stand out on white. Beneath it a time bar whose strength follows the fade of the pose trails. Below are both animals at the frame, drawn with the shared mouse renderer (``plot_mouse_data``) in one colour each with see-through bodies, over one silhouette per camera frame of their preceding movement that fades with age and ends after ``window.trail_end_seconds``; the skeleton's line widths scale with the panel's zoom so the animals look the same in every figure. The camera stands on the male's side of the pair when ``camera.azimuth`` is ``"male_side"`` (it looks along the line from the female's body centre to the male's, so he is nearest the viewer); a number read off the view picker fixes it instead. Optional: a scale mark of three perpendicular arms of ``scale_mark.length_cm`` beside the male, the arena floor and nearest wall as grey surfaces with a black junction line (``surfaces``), a two-swatch colour key (``layout.color_key``) and, with ``layout.align_pose``, the whole tracking panel moved left so its right edge sits under the spectrogram's. The PNG is cropped to its content; ``output.save_svg`` also writes an SVG with live text (Helvetica Light), vector current poses and the trail, spectrogram and bar as embedded images.

**The video** (``VocalPoseVideoMaker.make_vocal_pose_video``) plays the ``video.duration_seconds`` before the same frame at ``video.speed`` (0.5 = half speed, every camera frame kept) with a ``video.trail.trail_seconds`` fading trail behind each animal, the camera fixed at the still's view, following the pair (``video.follow_seconds`` ≥ 0 centres the smoothed middle of their outline in every frame) or turning at ``video.degrees_per_second`` into the still's view on the last frame, and a scrolling spectrogram with the present fixed at its centre, ``reach_seconds`` visible on each side and sharp only within ``sharp_seconds`` — as an inset in a corner no animal ever enters (``layout`` ``"inset"``; the lower right when free, slid left by ``inset_shift`` of the free room) or a strip below the animals (``"below"``). Frames render in ``workers`` processes (0 = all cores but one) and are encoded by ffmpeg.

**Choosing a window** — ``find_vocal_pose_windows`` ranks the session's windows of ``candidates.window_seconds`` in which every call is a non-noise, non-squeak USV of the male, no call is cut by an edge, no noise detection falls inside and the animals stay within ``candidates.max_gap_cm`` (the smallest distance between any body keypoints of the two, tails excluded) on every frame; the score sums the z-scores of the vocal fraction, the duration-weighted bandwidth and the mean call duration. ``plot_vocal_pose_window_candidates`` stacks the spectrograms of the best ``n_shown`` non-overlapping windows. **Choosing a camera** — ``vocal_pose_view_picker_html`` writes a page on which the frame turns by dragging (or two sliders), with presets for the male-side, heads-down, tails-down and top views, reporting the azimuth and elevation in the figure's convention.

Both outputs land in ``figures.save_directory`` as ``vocal_pose_<session>_<start>-<end>s`` (plus a ``_YYYYMMDD_HHMMSS`` stamp when ``figures.timestamp_in_name`` is set):

.. parsed-literal::

    ├── ...  (the ``figures.save_directory`` output folder)
    │   ├── **vocal_pose_20250516_122134_21.50-23.50s.png**
    │   ├── **vocal_pose_20250516_122134_21.50-23.50s.svg**
    │   ├── **vocal_pose_20250516_122134_19.50-23.50s.mp4**
    │   ...

Every tunable lives in the ``vocal_pose_figures`` block of */usv-playpen/_parameter_settings/visualizations_settings.json*. The keys a run normally changes:

* **window.end_time** : the frame shown, in seconds from video start; **window.history_seconds** the spectrogram / time-bar span before it; **window.trail_end_seconds** the age at which the trail has faded to nothing (0 keeps it for the whole window).
* **spectrogram.microphone** : ``"peak"`` or a device-channel name such as ``"s04"``; **spectrogram.freq_range_khz** / **freq_ticks_khz** the band shown and its labels; **db_floor** ``"adaptive"`` (``adaptive_floor_above_median_db`` above the band's median power) or a dB value; **n_fft** / **hop_fraction** the STFT (a shorter window widens narrowband sweeps); **male_deep** / **female_deep** / **ramp_split** the darker end of each ramp; **ceiling_percentile** / **gamma** / **call_pad_seconds** the loudness scaling.
* **camera.azimuth** : ``"male_side"`` or degrees; **camera.elevation** degrees above the floor (90 is straight down); **camera.top_sex** the animal drawn last, on top where they overlap.
* **trail** : ``strongest`` (colour strength just behind the present), ``fade_seconds`` (time constant), ``floor`` (strength old silhouettes settle to), ``taper_fraction`` (the last fraction of the trail over which it tapers to nothing).
* **skeleton** : ``line_width`` / ``node_size`` at the reference zoom (``reference_half_extent_m``), ``body_opacity`` / ``trail_body_opacity`` of the filled bodies, ``margin_m`` / ``z_margin_m`` around the trail, ``history_point``.
* **scale_mark** : ``length_cm`` (0 = none), ``labels``, ``signs`` (−1 reverses an arm), ``turn_degrees`` (turns the two floor arms about the vertical), ``gap_cm`` from the male, ``line_width``, label and caption placement, ``font_size``, ``color``.
* **surfaces** : ``enabled``, ``color``, ``junction_color`` / ``junction_width``, ``wall_height_m``, ``content_tolerance`` (what counts as drawn against the surface colour when cropping), ``pad_inches``.
* **layout** : figure width, the spectrogram's left / width / height, the bar, fonts and ink, ``content_threshold`` / ``crop_edge_pixels`` / ``svg_pad_inches`` for the crops, ``box_zoom`` / ``fit_shrink`` / ``max_fit_passes`` (the pose box is shrunk until nothing is cut off), ``align_pose``, ``color_key`` / ``key_position`` (``"pose"`` = wide swatches beside the animals, ``"spectrogram"`` = squares to its right) and the swatch sizes.
* **output** : ``dpi`` and ``save_svg``.
* **video** : ``duration_seconds``, ``speed``, ``follow_seconds`` (−1 = fixed camera), ``degrees_per_second``, render ``inches`` / ``dpi`` / ``crop_edge_pixels`` / ``workers``, the ffmpeg ``codec`` / ``pixel_format`` / ``crf`` / ``encode_timeout_seconds``, its own ``trail`` and ``skeleton`` sub-blocks, and ``spectrogram`` (``enabled``, ``layout``, ``reach_seconds``, ``sharp_seconds``, ``fade_seconds``, ``floor``, the inset's size, corner and shift, and the panel text room).
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
                "call_pad_seconds": 0.005
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
