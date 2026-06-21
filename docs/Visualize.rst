.. _Visualize:

Visualize
==================
This page explains how to use the data visualization functionalities in the *usv-playpen* GUI:

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

    F:\\Bartul\\Data\\20250430_145017
    F:\\Bartul\\Data\\20250430_165730
    F:\\Bartul\\Data\\20250430_182145

Plot neuronal tuning figures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the *Compute neuronal tuning curves* function from the *Analyze* section has completed, you have the ability to plot its results. Output is one combined multi-page document per cluster: a behavioral page per temporal offset and per plot-feature group (``individual.<mouse>`` and ``social``) followed by the vocal pages — Page 1 (bout raster + pooled pre-USV ``usv_peth`` on top, ``usv_property_tuning`` continuous-property grid below) and Page 2 (``usv_category_tuning`` watersheds + ``usv_category_peth`` per-category PETH grid). 1D ratemaps are drawn as a line spanning the plot, colored by the per-mouse palette (or the social color for social features). The 99% CI of the shuffled distribution is shown as a shaded band around the line.

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

For non-PDF formats, each page is written to a separate file with a ``_p{N}_{label}`` suffix (e.g. ``..._p1_behavioral_beh_offset=0s_individual.<mouse>.png``, ``..._p3_vocal_a_male.png``). PDF emits a single multi-page file.

An example of such tuning curves for one particular unit is shown below:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_tuning_1.png
   :align: center
   :alt: Example tuning 1

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_tuning_2.png
   :align: center
   :alt: Example tuning 2

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_tuning_3.png
   :align: center
   :alt: Example tuning 3

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_tuning_4.png
   :align: center
   :alt: Example tuning 4

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_tuning_5.png
   :align: center
   :alt: Example tuning 5

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_tuning_6.png
   :align: center
   :alt: Example tuning 6

.. raw:: html

   <br>

The rendering-side knobs live in the project-wide ``figures`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (compute-side knobs such as ``smoothing_sd`` and ``behavioral_min_occupancy_seconds`` live in *analyses_settings.json* under ``calculate_neuronal_tuning_curves`` — see the *Analyze* page):

* **save_directory** : default output directory for figures that aren't written next to their source data (e.g. cross-session anatomy plots). Per-figure code may override this with an explicit ``out_dir`` argument; per-cluster ratemaps and other session-bound figures always stay next to the data.
* **fig_format** : default output file format. For the per-cluster ratemap PDFs, ``pdf`` produces a single multi-page document; ``png`` / ``jpg`` / ``svg`` write one file per page.
* **dpi** : default raster resolution applied to every ``fig.savefig`` callsite that goes through ``visualizations.figure_io.save_figure``.
* **timestamp_in_name** : when ``true``, ``_YYYYMMDD_HHMMSS`` is appended to figure stems by default. Session-bound figures opt out of this with ``timestamp_in_name=false`` since their filenames already embed a session id or unit id.
* **cmap** : default matplotlib colormap used by every heatmap / ratemap callsite (one of ``viridis``, ``cividis``, ``plasma``, ``inferno``, ``magma``). ``make_behavioral_videos`` keeps its own ``general_figure_specs.cmap`` argument and is unaffected.

.. code-block:: json

    "figures": {
        "save_directory": "/mnt/falkner/Bartul/figures",
        "fig_format": "svg",
        "dpi": 300,
        "timestamp_in_name": true,
        "cmap": "inferno"
    }

Visualize 3D behavior (figure/video)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once 3D tracked data is available, you can visualize animal social behavior, either in figure or video. This GUI segment allows for a wide array of options in creating such visualizations. For example, you can choose whether you want to view the interaction from above or the side, and you can also choose to rotate the view as the behavior unfolds.

To obtain this visualization, you need to list the root directories of interest (it is best to stick with one), select the *Visualize 3D behavior (figure/video)* option in the GUI, insert the arena directory for that session, pick all desired figure features, click *Next* and then *Visualize*. It is important to point out that there are many more features available in the *visualization_settings.json* file than are available in the GUI, and these options are explained in detail several sections below:

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
    │   │   ├── **20250430_145017_3D_30045fr_dark_topview_Bartul.png**
    │   │   ├── **20250430_145017_3D_30045-30795fr_dark_topview_Bartul.mp4**
    │   │   ...
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

An example figure of male-female courtship behavior (as visualized from the top view with a light background) is shown below:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/behavior_light_mode_fig.png
   :align: center
   :alt: Visualization example 1

.. raw:: html

   <br>

Another example male-female courtship interaction with a live spectrogram subplot, with vocalizations labeled by color of animal they were assigned to:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/behavior_dark_mode_fig.png
   :align: center
   :alt: Visualization example 2

.. raw:: html

   <br>

An example side view of a male-female courtship interaction with spectrogram, raster plot and behavioral features subplots:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/behavior_dark_mode_side.png
   :align: center
   :alt: Visualization example 3

.. raw:: html

   <br>

An example of an animated male-female courtship interaction with a light background, side view and history of both animals' heads:

.. image:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/behavior_video_example1.gif
   :align: center
   :alt: Behavior video example 1

.. raw:: html

   <br>

An example of an animated male-female courtship interaction with a dark background, top view and spectrogram with assigned vocalizations:

.. image:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/behavior_video_example2.gif
   :align: center
   :alt: Behavior video example 2

.. raw:: html

   <br>

The */usv-playpen/_parameter_settings/visualization_settings.json* file contains a section only partially modifiable in the GUI, but it can entirely be modified manually in the *visualization_settings.json* file:

* **arena_directory** : path to the directory with the 3D tracked arena data
* **speaker_audio_file** : path to the audio file containing the playback speaker sound
* **sequence_audio_file** : path to the frequency-shifted audio file containing the audible vocalizations
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
* **plot_usv_segments_bool** : plot the DAS-detected USV segments in the spectrogram
* **usv_segments_ypos** : y-axis position of the USV segments in the spectrogram (in Hz)
* **usv_segments_lw** : line width of the USV segments in the spectrogram

.. code-block:: json

    "make_behavioral_videos": {
        "arena_directory": "",
        "speaker_audio_file": "",
        "sequence_audio_file": "",
        "animate_bool": false,
        "video_start_time": 567.19,
        "video_duration": 5.0,
        "plot_theme": "dark",
        "save_fig": true,
        "view_angle": "top",
        "side_azimuth_start": 45,
        "rotate_side_view_bool": false,
        "rotation_speed": 5,
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
          "fig_dpi": 600,
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
          "raster_lw": 0.1,
          "raster_ll": 0.9,
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

Render USV spectrograms and embedding maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Beyond the GUI functions above, the ``usv_playpen.visualizations.make_usv_spectrograms`` module renders publication figures for ultrasonic vocalizations (USVs) directly from the processed session artifacts. It is driven from the ``usv_spectrogram_analyses.ipynb`` notebook (embedded below) rather than the GUI, and exposes:

- ``USVSpectrogramPlotter`` — single-channel, all-channel and stitched session-timeline spectrograms read from a session's concatenated ``*_int16.mmap`` audio. The single / all modes show a dB amplitude scale over a user-defined ``time_window``; the stitched mode places the pre-computed ``[0, 1]``-normalized per-USV spectrograms from the consolidated HDF5 store at their true on-session times on a linear normalized-amplitude canvas.
- ``plot_usv_property_histograms`` — five pooled per-USV property histograms (duration, mean amplitude, mean frequency, frequency bandwidth, spectral entropy) across every session listed in a text file.
- ``plot_session_type_usv_counts`` — mean USVs per session compared across the male-female, female-female and lone-male session types, with SEM error bars on each bar.
- ``plot_session_usv_timeline`` — every non-noise USV in one session drawn as a colored interval on a horizontal strip, keyed to the male / female / unassigned emitter.
- ``plot_umap_with_category_thumbnails`` — a two-panel figure pairing a UMAP (VAE or QLVM) scatter — colored by call category and overlaid with kNN cluster boundaries — against a per-category grid of spectrogram thumbnails sampled from the consolidated SAM2 + spectrogram store.

The rendering knobs for the spectrogram plotter live in the ``make_usv_spectrograms`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (mode, channel, ``time_window``, ``freq_limits``, ``nfft``, colorbar limits and the save options); the module-level helpers take their inputs as function arguments, all surfaced in the notebook's single **Parameters** cell.

The ``usv_spectrogram_analyses.ipynb`` notebook is the recommended entry point: it imports every function above, collects all data paths and styling toggles in one **Parameters** cell near the top, and runs each figure in its own independent cell. The full notebook lives in the repository at `usv_spectrogram_analyses.ipynb <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/analyses_notebooks/usv_spectrogram_analyses.ipynb>`_.

Interactively explore USV embeddings (marimo app)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For interactive (rather than static) exploration of the embedding spaces, the repository ships a `marimo <https://marimo.io>`_ app, ``analyses_notebooks/usv_embedding_explorer.py``. It pools every selected session's ``*_usv_summary.csv`` into one scatter of the chosen embedding map (VAE UMAP or QLVM torus), lets you brush a region, and shows a grid of example spectrograms sampled from inside that region. Launch it from the repo root in either of two modes:

.. code-block:: bash

    # editable, reactive code view (for tweaking the notebook)
    uv run marimo edit src/usv_playpen/analyses_notebooks/usv_embedding_explorer.py

    # clean app view (just the controls + plot, no code)
    uv run marimo run  src/usv_playpen/analyses_notebooks/usv_embedding_explorer.py

Both open in the browser at ``http://localhost:2718``. The controls sit directly above the plot:

- **Session lists** — a fixed-width dropdown of every ``*.txt`` session list in the configured input-files directory (playback lists are excluded). Pick one, some, or all, then click the **Load** button beside it; the chosen lists are pooled (and cached to a per-selection parquet under ``~/.usv_playpen_cache`` so re-selecting the same set reloads in seconds). Picking does not rebuild — nothing builds until you click **Load**.
- **Map** — VAE (UMAP) or QLVM (torus) coordinates.
- **Color by** — either a **categorical** label (``category`` fine / ``supercategory`` coarse / ``session type`` / ``session (id)`` / ``emitter (sex)``) or a **continuous** metric rendered through the project colormap: ``density (counts)``, ``duration (ms)``, ``mean``/``peak frequency (kHz)``, ``frequency bandwidth (kHz)``, ``mean``/``max amplitude (a.u.)`` or ``spectral entropy (nats)``. Frequencies and duration are rescaled to the labelled units; the categorical legend is hidden automatically when there are more than ~24 categories (e.g. many sessions).
- **Boundaries** — optionally overlay cluster outlines for ``category`` or ``supercategory`` (kNN-predicted, drawn as uniform-width haloed contours), independent of the coloring.
- **Examples (spectrograms) plotted** — how many spectrograms the brush samples (5–50). They are picked along an Archimedean spiral from the centre of the brushed region outward and laid out as a square grid to the right of the scatter, each call's width preserving its true duration against the fixed spectrogram window.

The session-list directory and the consolidated spectrogram/SAM2 store are read from the ``usv_embedding`` block of */usv-playpen/_parameter_settings/visualizations_settings.json* (``input_files_directory`` and ``consolidated_h5_path``, both resolved per-host via ``configure_path``). The emitter-sex coloring is corrected per session type inferred from the list filename (``female_female`` / ``male_male`` / ``lone_male`` / ``courtship``), so a female in a female-female session is colored as female rather than by the raw track index. The app lives in the repository at `usv_embedding_explorer.py <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/analyses_notebooks/usv_embedding_explorer.py>`_.

Render the QLVM torus-traversal video
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``usv_playpen.visualizations.qlvm_torus_traversal_video`` module renders a two-panel animation: a QLVM density torus with a looped trajectory visiting each cluster center in turn (left), paired with the spectrogram of the USV nearest the moving marker (right). It is **cohort-level** — it reads one model's analysis arrays, not a session directory — and is exposed both as the ``qlvm-torus-traversal-video`` CLI and in the GUI *Visualize* window under the **QLVM / USV visualizations** column (third column). There, "Render QLVM torus video" (Yes/No), a **Clustering** selector (``coarse`` = 7 clusters / ``fine`` = 12), and a **Video fps** field are the exposed controls; the remaining parameters live in the ``qlvm_torus_traversal_video`` block of *visualizations_settings.json*.

Inputs (all in that settings block, resolved via ``configure_path``):

- ``arrays_npz_path_coarse`` / ``arrays_npz_path_fine`` — the QLVM analysis arrays (``latent_coords``, ``heatmap``, ``ws_labels_periodic``, ``centers``); the **Clustering** selector picks which one.
- ``provenance_pkl_path`` — the latents provenance pickle (a ``{session_name: DataFrame}`` dict with an ``original_index`` column). Its dict-order concatenation reproduces the ``latent_coords`` order, and is used to build a flat-index → (``<date>_<time>`` session key, spectrogram row) map.
- ``consolidated_h5_path`` — the consolidated per-session spectrogram/SAM2 store; each nearest-USV spectrogram is read on the fly from ``spectrogram/<key>/spectrograms`` rather than from a flat array.
- ``frames_per_segment``, ``fps``, ``dpi``, ``path_curvature``, ``trail_length`` — render parameters.

The output ``.mp4`` is written to ``figures.save_directory`` with a timestamped name (``qlvm_torus_traversal_<YYYYMMDD_HHMMSS>.mp4``) when no explicit ``--output-path`` is given. All text renders in Helvetica (titles in Helvetica Light) via the shared ``apply_plot_style`` helper. The module lives in the repository at `qlvm_torus_traversal_video.py <https://github.com/bartulem/usv-playpen/blob/main/src/usv_playpen/visualizations/qlvm_torus_traversal_video.py>`_.
