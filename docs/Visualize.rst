.. _Visualize:

Visualize
==================
This page explains how to use the data visualization functionalities in the *usv-playpen* GUI..

In order to run any of the functions detailed below, you first click the *Visualize* button on the GUI main display.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_0a.png
   :align: center
   :alt: Visualize Step 0a

.. raw:: html

   <br>

Clicking the *Visualize* button will open a new window with all the offered functionalities (see below).

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_0b.png
   :align: center
   :alt: Visualize Step 0b

.. raw:: html

   <br>

All the main functions are outlined in orange and black fields are specific options tunable by the user in the GUI. It is important to note that these are not necessarily *all* the options the user can set, and the full list of options can be found under each function in the */usv-playpen/_parameter_settings/visualizations_settings.json* file. Each time the user clicks the *Next* button in the window above, *visualizations_settings.json* is modified to the newest input configuration.

The *Root directories* field enables you to list the directories containing the data you want to visualize. Each root directory should be in its **own row**; for example, three sessions should be listed as follows:

.. parsed-literal::

    F:\\Bartul\\Data\\20250430_145017
    F:\\Bartul\\Data\\20250430_165730
    F:\\Bartul\\Data\\20250430_182145

Plot 3D behavioral tuning curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the *Compute 3D feature tuning curves* function from the *Analyze* section has completed, you have the ability to plot its results. The series of provided plots visualize the relationship between firing rate and each measured feature.
These tuning curves are denote by a line spanning the graph horizontally, usually in color (depending on the sex of the animal) or in black for social features.
The 99% CI of the shuffled distribution is shown as a shaded area around the tuning curve.

To obtain this visualization, you need to list the root directories of interest, select the *Plot 3D behavioral tuning curves* option in the GUI and click *Next* and then *Visualize*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/visualize_step_1.png
   :align: center
   :alt: Visualize Step 1

.. raw:: html

   <br>

Running this function results in the population of the *tuning_curves* subdirectory with *pdf files* containing tuning curves of each neuron for each feature:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ├── tuning_curves
    │   │   │   ├── **imec0_cl0000_ch361_good_tuning_curves_data.pdf**
    │   │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

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

The */usv-playpen/_parameter_settings/visualization_settings.json* file contains a section fully modifiable in the GUI, and it consists of the following parameters:

* **smoothing_sd** : standard deviation of the Gaussian kernel used for smoothing the tuning curves (unit is in number of bins)
* **occ_threshold** : minimum occupancy threshold for a bin to be considered in the tuning curve calculation (in s)

.. code-block:: json

    "neuronal_tuning_figures": {
        "smoothing_sd": 1.0,
        "occ_threshold": 1.0
    }

Visualize 3D behavior (figure/video)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    │   │   ├── **20250430_145017_3D_30045--30795fr_dark_topview_Bartul.mp4**
    │   │   ...
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The */usv-playpen/_parameter_settings/visualization_settings.json* file contains a section only partially modifiable in the GUI, but it can entirely be modified manually in the *visualization_settings.json* file:

* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx
* **xxx** : xxx

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