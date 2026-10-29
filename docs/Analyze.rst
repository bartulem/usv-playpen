.. _Analyze:

Analyze
==================
This page explains how to use the data analyses functionalities in the *usv-playpen* GUI.

In order to run any of the functions detailed below, select an experimenter name from the dropdown menu and click the *Analyze* button on the GUI main display:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_0a.png
   :align: center
   :alt: Analysis Step 0a

.. raw:: html

   <br>

Clicking the *Analyze* button will open a new window with all the offered functionalities (see below):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_0b.png
   :align: center
   :alt: Analysis Step 0b

.. raw:: html

   <br>

All the main functions are outlined in orange, and black fields are function-specific options tunable by the user in the GUI. It is important to note that these are not necessarily *all* the options the user can set, and the full list of options can be found under each function in the */usv-playpen/_parameter_settings/analyses_settings.json* file. Each time the user clicks the *Next* button in the window above, *analyses_settings.json* is modified to the newest input configuration.

The *Root directories* field enables you to list the directories containing the data you want to analyze. Each root directory should be in its **own row**; for example, three sessions should be listed as follows:

.. parsed-literal::

    /mnt/falkner/Bartul/20250430_145017
    /mnt/falkner/Bartul/20250430_165730
    /mnt/falkner/Bartul/20250430_182145

Shared settings
---------------
Besides the per-analysis blocks documented under each function below, *analyses_settings.json* carries the blocks the Analyze window and every analysis share. The GUI writes the first four from its fields; the ``data_roots`` block is edited in the file only. Every path in the file is stored with the shipped experimenter folder (``Bartul``) and is re-keyed to the experimenter selected on the GUI main display (or, for the CLI, to the ``experimenter`` of *behavioral_experiments_settings.toml* / the ``EXPERIMENTER_ID`` environment variable) and to the host OS mount before it is used:

.. code-block:: json

    "analyses_booleans": {
        "compute_behavioral_features_bool": false,
        "compute_neuronal_tuning_bool": false,
        "compute_inter_usv_interval_distributions_bool": false,
        "create_usv_playback_wav_bool": false,
        "create_naturalistic_usv_playback_wav_bool": false,
        "frequency_shift_audio_segment_bool": false,
        "decode_usv_bout_occupancy_bool": false
    },
    "analyze_data": {
        "root_directories": []
    },
    "credentials_directory": "",
    "data_roots": {
        "ephys_root": "/mnt/falkner/Bartul/EPHYS",
        "histology_root": "/mnt/falkner/Bartul/histology",
        "data_root": "/mnt/falkner/Bartul/Data",
        "catalog_path": "/mnt/falkner/Bartul/EPHYS/unit_catalog.csv",
        "aggregator_out_dir": "/mnt/falkner/Bartul/neuronal_tuning",
        "allen_meshes_dir": "/mnt/falkner/Bartul/EPHYS/allen_meshes",
        "converter_path": "/mnt/falkner/Bartul/EPHYS/neuropixels_sites_to_anatomy_converter.json",
        "naturalistic_usv_repository_dir": "/mnt/falkner/Bartul/usv_playback_experiments/usv_naturalistic_playback_repositories",
        "naturalistic_usv_playback_dir": "/mnt/falkner/Bartul/usv_playback_experiments/usv_naturalistic_playback_files"
    },
    "send_email": {
        "analyses_pc_list": ["A84E Windows", "A84I Linux", "A84I Windows", "165B Audio", "165B Neural"],
        "analyses_pc_choice": "A84I Linux",
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
   * - :ref:`analyses_booleans.compute_behavioral_features_bool <ana-analyses-booleans-compute-behavioral-features-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`analyses_booleans.compute_neuronal_tuning_bool <ana-analyses-booleans-compute-neuronal-tuning-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`analyses_booleans.compute_inter_usv_interval_distributions_bool <ana-analyses-booleans-compute-inter-usv-interval-distributions-bool>`
     - ``false``
     - ``true``, ``false`` (file only)
   * - :ref:`analyses_booleans.create_usv_playback_wav_bool <ana-analyses-booleans-create-usv-playback-wav-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`analyses_booleans.create_naturalistic_usv_playback_wav_bool <ana-analyses-booleans-create-naturalistic-usv-playback-wav-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`analyses_booleans.frequency_shift_audio_segment_bool <ana-analyses-booleans-frequency-shift-audio-segment-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`analyses_booleans.decode_usv_bout_occupancy_bool <ana-analyses-booleans-decode-usv-bout-occupancy-bool>`
     - ``false``
     - ``true``, ``false`` (file only)
   * - :ref:`analyze_data.root_directories <ana-analyze-data-root-directories>`
     - ``[]``
     - list of session-root paths
   * - :ref:`credentials_directory <ana-credentials-directory>`
     - ``""``
     - path to a directory
   * - :ref:`data_roots.ephys_root <ana-data-roots-ephys-root>`
     - path
     - path to a directory
   * - :ref:`data_roots.histology_root <ana-data-roots-histology-root>`
     - path
     - path to a directory
   * - :ref:`data_roots.data_root <ana-data-roots-data-root>`
     - path
     - path to a directory
   * - :ref:`data_roots.catalog_path <ana-data-roots-catalog-path>`
     - path
     - path to a CSV file
   * - :ref:`data_roots.aggregator_out_dir <ana-data-roots-aggregator-out-dir>`
     - path
     - path to a directory
   * - :ref:`data_roots.allen_meshes_dir <ana-data-roots-allen-meshes-dir>`
     - path
     - path to a directory
   * - :ref:`data_roots.converter_path <ana-data-roots-converter-path>`
     - path
     - path to a JSON file
   * - :ref:`data_roots.naturalistic_usv_repository_dir <ana-data-roots-naturalistic-usv-repository-dir>`
     - path
     - path to a directory
   * - :ref:`data_roots.naturalistic_usv_playback_dir <ana-data-roots-naturalistic-usv-playback-dir>`
     - path
     - path to a directory
   * - :ref:`send_email.analyses_pc_list <ana-send-email-analyses-pc-list>`
     - see JSON above
     - list of PC names
   * - :ref:`send_email.analyses_pc_choice <ana-send-email-analyses-pc-choice>`
     - ``"A84I Linux"``
     - one entry of ``analyses_pc_list``
   * - :ref:`send_email.experimenter <ana-send-email-experimenter>`
     - ``"Bartul"``
     - an experimenter name
   * - :ref:`send_email.send_message.receivers <ana-send-email-send-message-receivers>`
     - ``[]``
     - list of e-mail addresses

.. _ana-analyses-booleans-compute-behavioral-features-bool:

``analyses_booleans.compute_behavioral_features_bool``
   ``compute_behavioral_features_bool`` in ``analyses_booleans`` — the toggle of the *Compute 3D behavioral features* analysis (one toggle per analysis in ``analyses_booleans``).

   :Options:
      * ``true`` — run the analysis
      * ``false`` — skip it
   :Default: ``false``
   :Notes: set by the orange *Compute 3D behavioral features* field of the window.

.. _ana-analyses-booleans-compute-neuronal-tuning-bool:

``analyses_booleans.compute_neuronal_tuning_bool``
   ``compute_neuronal_tuning_bool`` in ``analyses_booleans`` — the toggle of the *Compute neuronal tuning curves* analysis.

   :Options:
      * ``true`` — run the analysis
      * ``false`` — skip it
   :Default: ``false``
   :Notes: set by the orange *Compute neuronal tuning curves* field of the window.

.. _ana-analyses-booleans-compute-inter-usv-interval-distributions-bool:

``analyses_booleans.compute_inter_usv_interval_distributions_bool``
   ``compute_inter_usv_interval_distributions_bool`` in ``analyses_booleans`` — the toggle of the inter-USV interval analysis.

   :Options:
      * ``true`` — run the analysis in a GUI session
      * ``false`` — skip it
   :Default: ``false``
   :Notes: has no GUI field and runs in a GUI session only when set to ``true`` in the file by hand; otherwise the analysis runs from the command line (``generate-usv-interval-distributions``, see :doc:`CLI`) or from its notebook (see :doc:`Notebooks`).

.. _ana-analyses-booleans-create-usv-playback-wav-bool:

``analyses_booleans.create_usv_playback_wav_bool``
   ``create_usv_playback_wav_bool`` in ``analyses_booleans`` — the toggle of the *Create artificial playback .WAV* analysis.

   :Options:
      * ``true`` — run the analysis
      * ``false`` — skip it
   :Default: ``false``
   :Notes: set by the orange *Create artificial playback .WAV* field of the window.

.. _ana-analyses-booleans-create-naturalistic-usv-playback-wav-bool:

``analyses_booleans.create_naturalistic_usv_playback_wav_bool``
   ``create_naturalistic_usv_playback_wav_bool`` in ``analyses_booleans`` — the toggle of the *Create naturalistic playback .WAV* analysis.

   :Options:
      * ``true`` — run the analysis
      * ``false`` — skip it
   :Default: ``false``
   :Notes: set by the orange *Create naturalistic playback .WAV* field of the window.

.. _ana-analyses-booleans-frequency-shift-audio-segment-bool:

``analyses_booleans.frequency_shift_audio_segment_bool``
   ``frequency_shift_audio_segment_bool`` in ``analyses_booleans`` — the toggle of the *Frequency shift audio segment* analysis.

   :Options:
      * ``true`` — run the analysis
      * ``false`` — skip it
   :Default: ``false``
   :Notes: set by the orange *Frequency shift audio segment* field of the window.

.. _ana-analyses-booleans-decode-usv-bout-occupancy-bool:

``analyses_booleans.decode_usv_bout_occupancy_bool``
   ``decode_usv_bout_occupancy_bool`` in ``analyses_booleans`` — the toggle of the per-session vocal-bout occupancy decoder (see *Decode vocal-bout occupancy* below).

   :Options:
      * ``true`` — run the decoder in a GUI session
      * ``false`` — skip it
   :Default: ``false``
   :Notes: has no GUI field and runs in a GUI session only when set to ``true`` in the file by hand.

.. _ana-analyze-data-root-directories:

``analyze_data.root_directories``
   ``root_directories`` in ``analyze_data`` — the *Root directories* field above (one session root per row).

   :Options: a list of session-root directory paths
   :Default: ``[]``
   :Notes: written by the GUI from its *Root directories* field.

.. _ana-credentials-directory:

``credentials_directory``
   The *Credentials directory* field — the directory holding *email_config.ini*, read to send the PC-usage e-mails.

   :Options: path to a directory holding *email_config.ini*
   :Default: ``""``
   :Notes: written by the GUI from its *Credentials directory* field.

.. _ana-data-roots-ephys-root:

``data_roots.ephys_root``
   ``ephys_root`` in ``data_roots`` — the ``EPHYS`` tree of spike-sorted recordings, sibling of the ``Data`` tree (``data_root``).

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/EPHYS``
   :Notes: like every ``data_roots`` entry, a canonical data location that every analysis, figure and notebook resolves through ``os_utils.resolve_data_root`` instead of hard-coding it; edited in the file only.

.. _ana-data-roots-histology-root:

``data_roots.histology_root``
   ``histology_root`` in ``data_roots`` — the histology stacks of the :doc:`Neuropixels` tools and the anatomy figures.

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/histology``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-data-root:

``data_roots.data_root``
   ``data_root`` in ``data_roots`` — the ``Data`` tree of session roots.

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/Data``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-catalog-path:

``data_roots.catalog_path``
   ``catalog_path`` in ``data_roots`` — the unit catalog CSV.

   :Options: path to a CSV file
   :Default: ``/mnt/falkner/Bartul/EPHYS/unit_catalog.csv``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-aggregator-out-dir:

``data_roots.aggregator_out_dir``
   ``aggregator_out_dir`` in ``data_roots`` — where the :ref:`unit-triage aggregator <unit-triage-aggregator>` writes.

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/neuronal_tuning``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-allen-meshes-dir:

``data_roots.allen_meshes_dir``
   ``allen_meshes_dir`` in ``data_roots`` — the Allen atlas meshes of the :doc:`Neuropixels` tools and the anatomy figures.

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/EPHYS/allen_meshes``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-converter-path:

``data_roots.converter_path``
   ``converter_path`` in ``data_roots`` — the sites-to-anatomy converter JSON of the :doc:`Neuropixels` tools and the anatomy figures.

   :Options: path to a JSON file
   :Default: ``/mnt/falkner/Bartul/EPHYS/neuropixels_sites_to_anatomy_converter.json``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-naturalistic-usv-repository-dir:

``data_roots.naturalistic_usv_repository_dir``
   ``naturalistic_usv_repository_dir`` in ``data_roots`` — the naturalistic playback repository root: the root directory holding the ``male`` / ``female`` / ``mixed`` subdirectories that the builds are written into and the playback step reads from (see *Build the naturalistic USV repository* and *Create naturalistic playback .WAV* below).

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/usv_playback_experiments/usv_naturalistic_playback_repositories``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-data-roots-naturalistic-usv-playback-dir:

``data_roots.naturalistic_usv_playback_dir``
   ``naturalistic_usv_playback_dir`` in ``data_roots`` — the playback output directory of *Create naturalistic playback .WAV* (see below).

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/usv_playback_experiments/usv_naturalistic_playback_files``
   :Notes: like every ``data_roots`` entry, resolved through ``os_utils.resolve_data_root`` instead of hard-coded; edited in the file only.

.. _ana-send-email-analyses-pc-list:

``send_email.analyses_pc_list``
   ``analyses_pc_list`` in ``send_email`` — the PC names offered in the *Analyses PC of choice* dropdown.

   :Options: a list of PC-name strings
   :Default: ``["A84E Windows", "A84I Linux", "A84I Windows", "165B Audio", "165B Neural"]``

.. _ana-send-email-analyses-pc-choice:

``send_email.analyses_pc_choice``
   ``analyses_pc_choice`` in ``send_email`` — the selected PC, named in the e-mail subject.

   :Options: one of the names in ``analyses_pc_list``
   :Default: ``"A84I Linux"``
   :Notes: written by the GUI from the *Analyses PC of choice* dropdown.

.. _ana-send-email-experimenter:

``send_email.experimenter``
   ``experimenter`` in ``send_email`` — the selected experimenter, named in the e-mail body.

   :Options: an experimenter name
   :Default: ``"Bartul"``
   :Notes: written by the GUI from the experimenter selected on the GUI main display.

.. _ana-send-email-send-message-receivers:

``send_email.send_message.receivers``
   ``send_message.receivers`` in ``send_email`` — the *Notify e-mail(s) of PC usage* field.

   :Options: a list of e-mail addresses
   :Default: ``[]``

The remaining blocks of the file belong to analyses that have no window in the GUI and are documented with them: ``compute_inter_usv_interval_distributions`` (:doc:`CLI`, ``generate-usv-interval-distributions``, and :doc:`Notebooks`), ``detect_interesting_tuning_neurons`` and ``neuronal_coactivity`` (:doc:`Notebooks`), and the ``npx_*`` blocks (:doc:`Neuropixels`).

Compute 3D behavioral features
------------------------------
Once 3D tracking data is available, you can compute behavioral features. These can be *individual features* specific to each mouse (*i.e.*, spatial location, speed, posture, *etc.*) or *social features* (assuming two or more mice) that describe the relationship between the mice (*i.e.*, distance, angle, *etc.*). The output of this analysis are two files: [1] CSV file containing each measured feature in each column, and [2] a PDF file containing graphs for the observed distribution of each feature. The CSV is written to a temporary file next to its final name and renamed into place, so an interrupted run never leaves a truncated features file for the modeling, tuning-curve and figure steps that read it. The features are computed by ``usv_playpen.analyses.compute_behavioral_features``; the experiment code a session carries (the manipulation, e.g. ephys, optogenetics or playback, the social setting, the lighting, the number, sex and housing of the animals and the female's estrous stage) is decoded by ``usv_playpen.analyses.decode_experiment_label``. To run this analysis in the GUI, you need to list the root directories of interest, select *Compute 3D behavioral features*, click *Next* and then *Analyze*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_1.png
   :align: center
   :alt: Analysis Step 1

.. raw:: html

   <br>

The analysis results in the creation of [1] a CSV file containing behavioral features, [2] a PDF file showing occupancy distributions for each feature, and both can be found as shown below:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   │
    │   └── video
    │       ├── 20250430_145027.21241563
    │       ...
    │       ├── 20250430145035_camera_frame_count_dict.json
    │       ├── 20250430145035
    │       │    ├── **20250430145035_points3d_translated_rotated_metric_behavioral_features.csv**
    │       │    ├── **20250430145035_points3d_translated_rotated_metric_behavioral_features_histograms.pdf**
    │       ...

The *behavioral_features.csv* file should look similar to an example table below:

.. parsed-literal::
    ┌─────────────────┬─────────────────┬─────────────────┬────────────────┬───┬
    │ 158114_2.spaceX ┆ 158114_2.spaceY ┆ 158114_2.spaceZ ┆ 158114_2.speed ┆ … ┆
    │ ---             ┆ ---             ┆ ---             ┆ ---            ┆   ┆
    │ f64             ┆ f64             ┆ f64             ┆ f64            ┆   ┆
    ╞═════════════════╪═════════════════╪═════════════════╪════════════════╪═══╪
    │ -26.841561      ┆ -23.796571      ┆ 2.922045        ┆ NaN            ┆ … ┆
    │ -26.844099      ┆ -23.798917      ┆ 2.923149        ┆ 0.426783       ┆ … ┆
    │ -26.848196      ┆ -23.802833      ┆ 2.925208        ┆ 0.505712       ┆ … ┆
    │ -26.85301       ┆ -23.807948      ┆ 2.927885        ┆ 0.598469       ┆ … ┆
    │ -26.859138      ┆ -23.813435      ┆ 2.930909        ┆ 0.692332       ┆ … ┆
    │ …               ┆ …               ┆ …               ┆ …              ┆ … ┆
    │ -4.515579       ┆ -28.340828      ┆ 3.667301        ┆ 11.337689      ┆ … ┆
    │ -4.583698       ┆ -28.336554      ┆ 3.668319        ┆ 9.594388       ┆ … ┆
    │ -4.638644       ┆ -28.332085      ┆ 3.668867        ┆ 7.809649       ┆ … ┆
    │ -4.678483       ┆ -28.327466      ┆ 3.668817        ┆ 6.153409       ┆ … ┆
    │ -4.699602       ┆ -28.324635      ┆ 3.668698        ┆ 4.805457       ┆ … ┆
    └─────────────────┴─────────────────┴─────────────────┴────────────────┴───┴

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section not modifiable in the GUI itself, but it can be modified manually:

.. code-block:: json

    "compute_behavioral_features": {
        "head_points": [
          "Head",
          "Ear_R",
          "Ear_L",
          "Nose"
        ],
        "tail_points": [
          "TTI",
          "Tail_0",
          "Tail_1",
          "Tail_2",
          "TailTip"
        ],
        "back_root_points": [
          "Neck",
          "Trunk",
          "TTI"
        ],
        "derivative_bins": 10,
        "speed_smoothing_time_window": 0.015,
        "feature_hist_num_bins": 36,
        "spatial_hist_min_val": -32,
        "spatial_hist_max_val": 32,
        "spatial_hist_num_bins": 196,
        "back_angles_optimizer_seed": 0,
        "sei_speed_percentile": 99
  }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`head_points <ana-compute-behavioral-features-head-points>`
     - see JSON above
     - ordered list of skeleton node names
   * - :ref:`tail_points <ana-compute-behavioral-features-tail-points>`
     - see JSON above
     - ordered list of skeleton node names
   * - :ref:`back_root_points <ana-compute-behavioral-features-back-root-points>`
     - see JSON above
     - ordered list of skeleton node names
   * - :ref:`derivative_bins <ana-compute-behavioral-features-derivative-bins>`
     - ``10``
     - integer > 0
   * - :ref:`speed_smoothing_time_window <ana-compute-behavioral-features-speed-smoothing-time-window>`
     - ``0.015``
     - seconds > 0
   * - :ref:`feature_hist_num_bins <ana-compute-behavioral-features-feature-hist-num-bins>`
     - ``36``
     - integer > 0
   * - :ref:`spatial_hist_min_val <ana-compute-behavioral-features-spatial-hist-min-val>`
     - ``-32``
     - cm
   * - :ref:`spatial_hist_max_val <ana-compute-behavioral-features-spatial-hist-max-val>`
     - ``32``
     - cm
   * - :ref:`spatial_hist_num_bins <ana-compute-behavioral-features-spatial-hist-num-bins>`
     - ``196``
     - integer > 0
   * - :ref:`back_angles_optimizer_seed <ana-compute-behavioral-features-back-angles-optimizer-seed>`
     - ``0``
     - integer seed
   * - :ref:`sei_speed_percentile <ana-compute-behavioral-features-sei-speed-percentile>`
     - ``99``
     - percentile in 0-100

.. _ana-compute-behavioral-features-head-points:

``head_points``
   Head skeleton node names (order matters!).

   :Options: an ordered list of skeleton node names
   :Default: ``["Head", "Ear_R", "Ear_L", "Nose"]``

.. _ana-compute-behavioral-features-tail-points:

``tail_points``
   Tail skeleton node names (order matters!).

   :Options: an ordered list of skeleton node names
   :Default: ``["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]``

.. _ana-compute-behavioral-features-back-root-points:

``back_root_points``
   Back skeleton node names (order matters!).

   :Options: an ordered list of skeleton node names
   :Default: ``["Neck", "Trunk", "TTI"]``

.. _ana-compute-behavioral-features-derivative-bins:

``derivative_bins``
   Number of bins to compute derivatives over.

   :Options: a positive integer number of bins
   :Default: ``10``

.. _ana-compute-behavioral-features-speed-smoothing-time-window:

``speed_smoothing_time_window``
   Gaussian speed-smoothing window (in seconds).

   :Options: a positive number of seconds
   :Default: ``0.015``
   :Notes: the Gaussian's stddev is ``floor(window * capture_framerate)`` frames.

.. _ana-compute-behavioral-features-feature-hist-num-bins:

``feature_hist_num_bins``
   Number of bins for the 1-D feature-distribution histograms.

   :Options: a positive integer number of bins
   :Default: ``36``

.. _ana-compute-behavioral-features-spatial-hist-min-val:

``spatial_hist_min_val``
   Lower edge (cm) of the 2-D spatial-occupancy histogram.

   :Options: a number in cm
   :Default: ``-32``

.. _ana-compute-behavioral-features-spatial-hist-max-val:

``spatial_hist_max_val``
   Upper edge (cm) of the 2-D spatial-occupancy histogram.

   :Options: a number in cm
   :Default: ``32``

.. _ana-compute-behavioral-features-spatial-hist-num-bins:

``spatial_hist_num_bins``
   Number of bins per axis for the 2-D spatial-occupancy histogram.

   :Options: a positive integer number of bins per axis
   :Default: ``196``

.. _ana-compute-behavioral-features-back-angles-optimizer-seed:

``back_angles_optimizer_seed``
   Seed of the random generator that draws the Nelder-Mead starting point (two angles uniform in (-π/4, π/4)) used to find the canonical back frame for back pitch / yaw.

   :Options: an integer seed
   :Default: ``0``
   :Notes: fixed (``0``) so the back angles are bit-exactly reproducible across runs.

.. _ana-compute-behavioral-features-sei-speed-percentile:

``sei_speed_percentile``
   Percentile (0-100) of each observer's speed trace used as the speed normaliser ``v_max`` of the Social Engagement Index (SEI).

   :Options: a percentile in 0-100
   :Default: ``99``
   :Notes: ``99`` so a handful of extreme-speed frames cannot compress every other frame's speed term.

Compute neuronal tuning curves
------------------------------
Having recorded unit activity, social behavior, and ultrasonic vocalizations (USVs), you might be interested whether individual units encode specific behavioral features and / or vocal properties. To get at this, you can compute session-averaged *tuning curves* capturing the relationship between the firing rate of each unit and (a) each 3D behavioral feature, and (b) USV-anchored quantities — a pooled peri-USV-onset PETH (peri-event time histogram; ``usv_peth``, by default from 2 s before to 0.5 s after each USV's onset), within-USV firing rate as a function of each continuous acoustic property (``usv_property_tuning`` over duration, mean / peak frequency, bandwidth, mean / max amplitude (relative to each USV's own peak), absolute loudness (``loudness_db``), spectral entropy, mask number), within-USV firing rate as a function of categorical USV labels (``usv_category_tuning`` over the QLVM (quasi-Monte Carlo latent variable model) category column of the summary, ``qlvm_category``: the content-ridge categories R-1 … R-k of the regular map; the conditional maps carry no category), and a per-category time-resolved peri-USV PETH (``usv_category_peth``; saved in the pickle, not plotted). Noise segments are always dropped. Squeaks are dropped from the self side's anchors by default (``exclude_squeaks_self``; ``exclude_squeaks_partner`` for the partner side, off by default) and never enter the QLVM category tuning — the QLVM maps were trained without squeaks, which get their own embedding. Behavioral features listed in ``excluded_behavioral_features`` (by default the nose-anchored dyadic features, whose head-anchored versions are tuned instead) are left out, derivatives included. Behavioral and vocal payloads are produced together and serialized into a single per-cluster pickle; a re-run replaces each part's keys whole, and a part that is skipped (missing inputs, no eligible emitter side) or fails for a cluster removes that part's previous results from the pickle rather than leaving them next to the new ones. A cluster whose vocal part fails is logged and the remaining clusters are still computed, but the session then ends with an error naming every failed cluster, so a partial run is never reported as finished. The computation lives in ``usv_playpen.analyses.compute_neuronal_tuning_curves``. To trigger this in the GUI, list the root directories of interest, select *Compute neuronal tuning curves*, click *Next* and then *Analyze* (a progress bar will appear in the terminal while the analysis is running):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_2.png
   :align: center
   :alt: Analysis Step 2

.. raw:: html

   <br>

The analysis results in the creation of a *tuning_curves* subdirectory containing a *pickle file* for each recorded unit. Each pkl carries (when the corresponding inputs exist) ``beh_offset=*s`` blocks for behavioral tuning, plus ``usv_peth`` (PETH), ``usv_property_tuning`` (continuous property tuning), ``usv_category_tuning`` (categorical), ``usv_category_peth`` (per-category PETH), and ``behavioral_metadata`` / ``usv_metadata`` blocks describing the compute config:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ├── **tuning_curves**
    │   │   │   ├── **imec0_cl0000_ch361_good_tuning_curves_data.pkl**
    │   │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section only partially modifiable in the GUI, but it can be modified manually:

.. code-block:: json

    "calculate_neuronal_tuning_curves": {
        "temporal_offsets": [0],
        "n_shuffles": 1000,
        "shuffle_seed": 0,
        "total_bin_num": 36,
        "n_spatial_bins": 196,
        "spatial_scale_cm": 32,
        "shuffle_seconds_range": [20, 60],
        "peth_window_seconds": [-2, 0.5],
        "peth_bin_seconds": 0.05,
        "bout_quiet_seconds": 2.0,
        "ramp_anchor_near_seconds": -0.1,
        "ramp_anchor_far_seconds": -1.5,
        "vmi_wilcoxon_exact_max_n": 25,
        "ibi_bout_onset": {
            "ibi_z_score": 2.58,
            "ibi_component_index": 0,
            "min_vocalizations": 3,
            "shuffle_guard_seconds": 20.0,
            "trigger_window_seconds": [-1.0, 2.0],
            "trigger_bin_seconds": 0.02,
            "smoothing_bins": 2.0,
            "gain_window_seconds": [0.0, 0.25],
            "pre_window_seconds": [-1.0, -0.5],
            "peak_search_seconds": [-0.25, 1.0],
            "response_window_seconds": 0.15,
            "response_kernel_seconds": 0.02
        },
        "vocal_require_clean_post_anchor": true,
        "vocal_require_clean_prior_anchor": false,
        "n_usv_min_self": 30,
        "n_usv_min_partner": 30,
        "n_usv_min_category": 20,
        "behavioral_min_occupancy_seconds": 1.0,
        "usv_property_min_occupancy_seconds": 0.25,
        "include_partner_vocalization_tuning_bool": false,
        "exclude_squeaks_self": true,
        "exclude_squeaks_partner": false,
        "excluded_behavioral_features": ["nose-nose", "allo_yaw-nose", "nose-allo_yaw", "allo_pitch-nose", "nose-allo_pitch"],
        "smoothing_sd": 1.0,
        "circular_features": ["allo_yaw", "body_dir"]
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`temporal_offsets <ana-calculate-neuronal-tuning-curves-temporal-offsets>`
     - ``[0]``
     - list of offsets in seconds
   * - :ref:`n_shuffles <ana-calculate-neuronal-tuning-curves-n-shuffles>`
     - ``1000``
     - integer > 0
   * - :ref:`shuffle_seed <ana-calculate-neuronal-tuning-curves-shuffle-seed>`
     - ``0``
     - integer seed
   * - :ref:`total_bin_num <ana-calculate-neuronal-tuning-curves-total-bin-num>`
     - ``36``
     - integer > 0
   * - :ref:`n_spatial_bins <ana-calculate-neuronal-tuning-curves-n-spatial-bins>`
     - ``196``
     - integer > 0
   * - :ref:`spatial_scale_cm <ana-calculate-neuronal-tuning-curves-spatial-scale-cm>`
     - ``32``
     - cm > 0
   * - :ref:`shuffle_seconds_range <ana-calculate-neuronal-tuning-curves-shuffle-seconds-range>`
     - ``[20, 60]``
     - ``[min, max]`` in seconds
   * - :ref:`peth_window_seconds <ana-calculate-neuronal-tuning-curves-peth-window-seconds>`
     - ``[-2, 0.5]``
     - ``[start, stop]`` in seconds
   * - :ref:`peth_bin_seconds <ana-calculate-neuronal-tuning-curves-peth-bin-seconds>`
     - ``0.05``
     - seconds > 0
   * - :ref:`bout_quiet_seconds <ana-calculate-neuronal-tuning-curves-bout-quiet-seconds>`
     - ``2.0``
     - seconds > 0
   * - :ref:`ramp_anchor_near_seconds <ana-calculate-neuronal-tuning-curves-ramp-anchor>`
     - ``-0.1``
     - seconds relative to USV onset
   * - :ref:`ramp_anchor_far_seconds <ana-calculate-neuronal-tuning-curves-ramp-anchor>`
     - ``-1.5``
     - seconds relative to USV onset
   * - :ref:`vmi_wilcoxon_exact_max_n <ana-calculate-neuronal-tuning-curves-vmi-wilcoxon-exact-max-n>`
     - ``25``
     - integer > 0
   * - :ref:`ibi_bout_onset <ana-calculate-neuronal-tuning-curves-ibi-bout-onset>`
     - see JSON above
     - object of the sub-keys below
   * - :ref:`ibi_bout_onset.ibi_z_score <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-threshold>`
     - ``2.58``
     - standard deviations > 0
   * - :ref:`ibi_bout_onset.ibi_component_index <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-threshold>`
     - ``0``
     - mixture component index
   * - :ref:`ibi_bout_onset.min_vocalizations <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-min-vocalizations>`
     - ``3``
     - integer >= 1
   * - :ref:`ibi_bout_onset.shuffle_guard_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-shuffle-guard-seconds>`
     - ``20.0``
     - seconds >= 0
   * - :ref:`ibi_bout_onset.trigger_window_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-trigger>`
     - ``[-1.0, 2.0]``
     - ``[start, stop]`` in seconds
   * - :ref:`ibi_bout_onset.trigger_bin_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-trigger>`
     - ``0.02``
     - seconds > 0
   * - :ref:`ibi_bout_onset.smoothing_bins <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-trigger>`
     - ``2.0``
     - bins >= 0
   * - :ref:`ibi_bout_onset.gain_window_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-gain-window-seconds>`
     - ``[0.0, 0.25]``
     - ``[start, stop]`` in seconds
   * - :ref:`ibi_bout_onset.pre_window_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-pre-window-seconds>`
     - ``[-1.0, -0.5]``
     - ``[start, stop]`` in seconds
   * - :ref:`ibi_bout_onset.peak_search_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-peak-search-seconds>`
     - ``[-0.25, 1.0]``
     - ``[start, stop]`` in seconds
   * - :ref:`ibi_bout_onset.response_window_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-response>`
     - ``0.15``
     - seconds > 0
   * - :ref:`ibi_bout_onset.response_kernel_seconds <ana-calculate-neuronal-tuning-curves-ibi-bout-onset-response>`
     - ``0.02``
     - seconds > 0
   * - :ref:`vocal_require_clean_post_anchor <ana-calculate-neuronal-tuning-curves-vocal-require-clean-post-anchor>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`vocal_require_clean_prior_anchor <ana-calculate-neuronal-tuning-curves-vocal-require-clean-prior-anchor>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`n_usv_min_self <ana-calculate-neuronal-tuning-curves-n-usv-min-self>`
     - ``30``
     - integer >= 0
   * - :ref:`n_usv_min_partner <ana-calculate-neuronal-tuning-curves-n-usv-min-partner>`
     - ``30``
     - integer >= 0
   * - :ref:`n_usv_min_category <ana-calculate-neuronal-tuning-curves-n-usv-min-category>`
     - ``20``
     - integer >= 0
   * - :ref:`behavioral_min_occupancy_seconds <ana-calculate-neuronal-tuning-curves-behavioral-min-occupancy-seconds>`
     - ``1.0``
     - seconds >= 0
   * - :ref:`usv_property_min_occupancy_seconds <ana-calculate-neuronal-tuning-curves-usv-property-min-occupancy-seconds>`
     - ``0.25``
     - seconds >= 0
   * - :ref:`include_partner_vocalization_tuning_bool <ana-calculate-neuronal-tuning-curves-include-partner-vocalization-tuning-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`exclude_squeaks_self <ana-calculate-neuronal-tuning-curves-exclude-squeaks-self>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`exclude_squeaks_partner <ana-calculate-neuronal-tuning-curves-exclude-squeaks-partner>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`excluded_behavioral_features <ana-calculate-neuronal-tuning-curves-excluded-behavioral-features>`
     - see JSON above
     - list of behavioral base-feature names
   * - :ref:`smoothing_sd <ana-calculate-neuronal-tuning-curves-smoothing-sd>`
     - ``1.0``
     - bins >= 0; ``0`` disables
   * - :ref:`circular_features <ana-calculate-neuronal-tuning-curves-circular-features>`
     - ``["allo_yaw", "body_dir"]``
     - list of feature suffixes

.. _ana-calculate-neuronal-tuning-curves-temporal-offsets:

``temporal_offsets``
   List of temporal offsets between spikes and behavior (in seconds) for which the tuning curves will be calculated.

   :Options: a list of offsets in seconds; negative values: spikes follow behavior, i.e. behavior precedes spikes
   :Default: ``[0]``
   :Notes: adding values to the list increases the time needed for analysis drastically.

.. _ana-calculate-neuronal-tuning-curves-n-shuffles:

``n_shuffles``
   Number of spike train shuffles.

   :Options: a positive integer
   :Default: ``1000``
   :Notes: increasing this number increases the time needed for analysis drastically.

.. _ana-calculate-neuronal-tuning-curves-shuffle-seed:

``shuffle_seed``
   Base seed for the spike-train shuffle null distribution.

   :Options: an integer seed
   :Default: ``0``
   :Notes: fixing it makes the significance verdicts reproducible across runs.

.. _ana-calculate-neuronal-tuning-curves-total-bin-num:

``total_bin_num``
   Total number of bins for a 1D behavioral / vocal-property feature.

   :Options: a positive integer number of bins
   :Default: ``36``

.. _ana-calculate-neuronal-tuning-curves-n-spatial-bins:

``n_spatial_bins``
   Number of spatial bins (2D behavioral feature).

   :Options: a positive integer number of bins
   :Default: ``196``

.. _ana-calculate-neuronal-tuning-curves-spatial-scale-cm:

``spatial_scale_cm``
   Maximum distance from center of arena to one edge (in cm).

   :Options: a positive distance in cm
   :Default: ``32``

.. _ana-calculate-neuronal-tuning-curves-shuffle-seconds-range:

``shuffle_seconds_range``
   ``[min, max]`` of the uniform circular shift (in s) used to build the null distribution.

   :Options: two numbers of seconds, ``[min, max]``
   :Default: ``[20, 60]``

.. _ana-calculate-neuronal-tuning-curves-peth-window-seconds:

``peth_window_seconds``
   ``[start, stop]`` of the PETH window relative to each USV's onset (in s).

   :Options: two numbers of seconds, ``[start, stop]``
   :Default: ``[-2, 0.5]``
   :Notes: bins before onset honour the two cleanliness rules below (:ref:`vocal_require_clean_post_anchor <ana-calculate-neuronal-tuning-curves-vocal-require-clean-post-anchor>` and :ref:`vocal_require_clean_prior_anchor <ana-calculate-neuronal-tuning-curves-vocal-require-clean-prior-anchor>`); a bin after onset is kept only while it ends before the next USV's onset (any emitter, squeak and overlap segments included), so the post-onset PETH covers the USV itself and the silence after it, never a following USV.

.. _ana-calculate-neuronal-tuning-curves-peth-bin-seconds:

``peth_bin_seconds``
   PETH bin width (in s).

   :Options: a positive number of seconds
   :Default: ``0.05``

.. _ana-calculate-neuronal-tuning-curves-bout-quiet-seconds:

``bout_quiet_seconds``
   Inter-bout silence required to define a new bout (in s).

   :Options: a positive number of seconds
   :Default: ``2.0``

.. _ana-calculate-neuronal-tuning-curves-ramp-anchor:

``ramp_anchor_near_seconds`` / ``ramp_anchor_far_seconds``
   The two PETH bins (in s, relative to USV onset) whose rates give ``triage_stats.usv_peth[emitter].ramp_index = (rate(near) - rate(far)) / (rate(near) + rate(far))``.

   :Options:
      * ``ramp_anchor_near_seconds`` — seconds relative to USV onset, the near anchor
      * ``ramp_anchor_far_seconds`` — seconds relative to USV onset, the far anchor
   :Default: ``-0.1`` (just before onset) and ``-1.5`` (inside the default 2 s pre-onset window, well before it)
   :Notes: the ramp index is NaN when ``peth_window_seconds`` does not span both anchors.

.. _ana-calculate-neuronal-tuning-curves-vmi-wilcoxon-exact-max-n:

``vmi_wilcoxon_exact_max_n``
   Largest number of valid per-bout ``(FR_baseline, FR_USV)`` pairs for which the VMI's Wilcoxon signed-rank p-value uses scipy's exact null distribution.

   :Options: a positive integer number of pairs
   :Default: ``25``
   :Notes: above it the normal approximation is used.

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset:

``ibi_bout_onset``
   Per-unit statistics around the onset of IBI bouts, the respiratory-tempo grouping (not the ``bout_quiet_seconds`` one), stored as ``usv_ibi_bout_onset`` in the pkl and in ``triage_stats``.

   :Options: an object holding the sub-keys below
   :Default: see the JSON above
   :Notes: each silence is measured from the latest stop of every earlier USV, so a USV nested inside a longer one does not split a bout, and a bout ends at the latest stop of its USVs. Each statistic is referenced to the unit's own session rate, never to a pre-onset baseline (the window before an onset is occupied by the previous bout on roughly half of all bouts), and gets a circular-shift null and a two-sided empirical p.

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-threshold:

``ibi_bout_onset.ibi_z_score`` / ``ibi_bout_onset.ibi_component_index``
   The inter-bout threshold, as the emitter sex's inter-syllable mixture component at this many standard deviations.

   :Options:
      * ``ibi_z_score`` — a positive number of standard deviations
      * ``ibi_component_index`` — the index of the emitter sex's inter-syllable mixture component
   :Default: ``2.58`` and ``0``
   :Notes: the construction ``decode_usv_bout_occupancy`` and ``build-naturalistic-usv-repository`` use; 115.5 ms for the male at 2.58.

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-min-vocalizations:

``ibi_bout_onset.min_vocalizations``
   Fewest USVs a run needs to count as a bout.

   :Options: a positive integer number of USVs
   :Default: ``3``

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-shuffle-guard-seconds:

``ibi_bout_onset.shuffle_guard_seconds``
   The null's circular shifts are drawn from ``[guard, duration - guard]`` (in s).

   :Options: a non-negative number of seconds
   :Default: ``20.0``

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-trigger:

``ibi_bout_onset.trigger_window_seconds`` / ``ibi_bout_onset.trigger_bin_seconds`` / ``ibi_bout_onset.smoothing_bins``
   The bout-triggered rate: its window around onset and bin width (in s), and its Gaussian smoothing (in bins).

   :Options:
      * ``trigger_window_seconds`` — ``[start, stop]`` around onset, in seconds
      * ``trigger_bin_seconds`` — a positive bin width, in seconds
      * ``smoothing_bins`` — the Gaussian smoothing, in bins
   :Default: ``[-1.0, 2.0]``, ``0.02`` and ``2.0``

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-gain-window-seconds:

``ibi_bout_onset.gain_window_seconds``
   Window after onset whose mean rate over the session rate is ``onset_gain`` (in s).

   :Options: ``[start, stop]`` in seconds
   :Default: ``[0.0, 0.25]``

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-pre-window-seconds:

``ibi_bout_onset.pre_window_seconds``
   Window before onset whose mean rate over the session rate is ``pre_onset_gain`` (in s).

   :Options: ``[start, stop]`` in seconds
   :Default: ``[-1.0, -0.5]``

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-peak-search-seconds:

``ibi_bout_onset.peak_search_seconds``
   Window in which the smoothed trace's maximum gives ``peak_latency`` (in s).

   :Options: ``[start, stop]`` in seconds
   :Default: ``[-0.25, 1.0]``

.. _ana-calculate-neuronal-tuning-curves-ibi-bout-onset-response:

``ibi_bout_onset.response_window_seconds`` / ``ibi_bout_onset.response_kernel_seconds``
   Each USV's response for ``first_over_later`` (the bout's first USV's response over the mean of its later USVs') and ``fraction_first_above`` (share of bouts whose first USV drew more): spikes in a window of this width after the USV onset, weighted through a Gaussian of this standard deviation rather than a hard edge, so the response is continuous rather than a small integer (in s).

   :Options:
      * ``response_window_seconds`` — the window width after the USV onset, in seconds
      * ``response_kernel_seconds`` — the Gaussian's standard deviation, in seconds
   :Default: ``0.15`` and ``0.02``

.. _ana-calculate-neuronal-tuning-curves-vocal-require-clean-post-anchor:

``vocal_require_clean_post_anchor``
   Whether the time after the USV onset must be free of contaminating USVs to keep the anchor.

   :Options:
      * ``true`` — the time after the USV onset must be free of contaminating USVs to keep the anchor
      * ``false`` — no such requirement
   :Default: ``true``

.. _ana-calculate-neuronal-tuning-curves-vocal-require-clean-prior-anchor:

``vocal_require_clean_prior_anchor``
   Whether the lookback window must also be free of contaminating USVs.

   :Options:
      * ``true`` — the lookback window must also be free of contaminating USVs
      * ``false`` — no such requirement
   :Default: ``false``

.. _ana-calculate-neuronal-tuning-curves-n-usv-min-self:

``n_usv_min_self``
   Minimum self-side USV count for the self plots to be computed.

   :Options: a non-negative integer number of USVs
   :Default: ``30``

.. _ana-calculate-neuronal-tuning-curves-n-usv-min-partner:

``n_usv_min_partner``
   Minimum partner-side USV count for the partner plots to be computed.

   :Options: a non-negative integer number of USVs
   :Default: ``30``

.. _ana-calculate-neuronal-tuning-curves-n-usv-min-category:

``n_usv_min_category``
   Minimum per-category USV count to retain that category in the categorical tuning layouts.

   :Options: a non-negative integer number of USVs
   :Default: ``20``

.. _ana-calculate-neuronal-tuning-curves-behavioral-min-occupancy-seconds:

``behavioral_min_occupancy_seconds``
   Minimum behavioral occupancy per bin (in s) to draw that bin in 1D feature plots.

   :Options: a non-negative number of seconds
   :Default: ``1.0``
   :Notes: persisted into ``behavioral_metadata``. GUI field: *Min occupancy behavior bin (s)*.

.. _ana-calculate-neuronal-tuning-curves-usv-property-min-occupancy-seconds:

``usv_property_min_occupancy_seconds``
   Minimum vocal-property occupancy per bin (in s) to keep the rate estimate finite.

   :Options: a non-negative number of seconds
   :Default: ``0.25``
   :Notes: GUI field: *Min occupancy vocal bin (s)*.

.. _ana-calculate-neuronal-tuning-curves-include-partner-vocalization-tuning-bool:

``include_partner_vocalization_tuning_bool``
   Also compute partner-side vocal tuning when its threshold is met.

   :Options:
      * ``true`` — also compute partner-side vocal tuning when its threshold (``n_usv_min_partner``) is met
      * ``false`` — self side only
   :Default: ``false``
   :Notes: the self side is the tracked animal with more pure USVs and the partner the other; each side's ``sex`` label comes from the session metadata's ``Subjects`` (matched to the track name), never from the track slot, so a female-female session has two female sides. GUI field: *Include partner vocal tuning*.

.. _ana-calculate-neuronal-tuning-curves-exclude-squeaks-self:

``exclude_squeaks_self``
   Restrict the self side's anchors to pure USVs (``usv`` true and ``squeak`` false), leaving its pure squeaks and its overlap segments (a squeak and a USV in one segment, treated like squeaks) out.

   :Options:
      * ``true`` — pure-USV anchors only on the self side
      * ``false`` — the self side's pure squeaks and overlap segments are anchors too
   :Default: ``true``
   :Notes: the self side is the emitter with more pure USVs. Squeak-bearing segments still count as other vocalizations in the overlap checks. Every anchor is the onset of its vocalization (``os_utils.vocal_event_times``): a pure squeak at its own ``squeak_start``, an overlap segment at the earlier of its segment's and its squeak's onset, a pure USV at its segment's ``start``. A summary without the ``usv`` / ``squeak`` or ``squeak_start`` / ``squeak_end`` columns raises.

.. _ana-calculate-neuronal-tuning-curves-exclude-squeaks-partner:

``exclude_squeaks_partner``
   The same as ``exclude_squeaks_self`` for the partner side.

   :Options:
      * ``true`` — pure-USV anchors only on the partner side
      * ``false`` — the partner side's pure squeaks and overlap segments are anchors too
   :Default: ``false``, since the partner is usually the female, who emits the squeaks
   :Notes: either way, only pure-USV anchors enter the QLVM category tuning (the QLVM label of a squeak or of an overlap segment is not a USV category).

.. _ana-calculate-neuronal-tuning-curves-excluded-behavioral-features:

``excluded_behavioral_features``
   Behavioral base features left out of the tuning, derivatives (``_1st_der`` / ``_2nd_der``) included.

   :Options: a list of behavioral base-feature names
   :Default: ``nose-nose``, ``allo_yaw-nose``, ``nose-allo_yaw``, ``allo_pitch-nose``, ``nose-allo_pitch``
   :Notes: their head-anchored versions, ``head-head``, ``allo_yaw-head``, ``head-allo_yaw``, ``allo_pitch-head``, ``head-allo_pitch``, are tuned instead.

.. _ana-calculate-neuronal-tuning-curves-smoothing-sd:

``smoothing_sd``
   Standard deviation of the Gaussian kernel (in bins) applied to ratemaps and shuffle distributions.

   :Options: a non-negative number of bins; ``0`` disables smoothing
   :Default: ``1.0``
   :Notes: GUI field: *Smoothing sigma (bins)*.

.. _ana-calculate-neuronal-tuning-curves-circular-features:

``circular_features``
   List of behavioral feature suffixes that are wrap-around in nature (e.g. ``allo_yaw``, ``body_dir``, binned over one full turn).

   :Options: a list of behavioral feature suffixes
   :Default: ``["allo_yaw", "body_dir"]``
   :Notes: their smoothed rate maps and shuffle bands wrap around the ends (the first and last bins are one direction), and the triage helpers detect divergence runs that span the bin-0 / bin-N boundary; their derivatives are linear.

Per-cluster ``triage_stats`` block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each per-cluster pkl also carries a ``triage_stats`` block — a flat collection of pre-computed scalar summaries that the downstream :ref:`unit-triage aggregator <unit-triage-aggregator>` consumes without re-touching spike or USV data. It covers every modality the rate payload does — the 3D behavioral features (kinematic and social), 2D spatial location, and the USV-anchored quantities:

* ``behavioral[offset_key][feature_key]`` — 1D tuning to every behavioral feature, kinematic (speed, posture, …) and social (distance, angle, …) alike. Per-direction (excitation / suppression) divergence-segment analysis: ``n_bins`` total above (or below) the shuffle band, ``max_run`` consecutive-bin run length, ``run_start_idx`` / ``run_end_idx`` (and the corresponding axis-value bounds), ``peak_idx`` / ``peak_z``, plus ``peak_abs_z``, ``peak_signed_z``, ``selectivity = (max−min)/(max+min)``, ``monotonicity`` (Spearman ρ between bin index and rate), and ``is_circular``.
* ``spatial[offset_key][feature_key]`` — 2D place-cell diagnostics for location: ``info_rate_bps`` (Skaggs information rate), ``sparsity``, ``coherence`` (Pearson correlation between each bin and the mean of its 8 neighbors), plus the unshuffled peak rate and its grid coordinates. The 2D spatial map is computed without shuffles, so peak Z is not defined; this block reports the rate / occupancy diagnostics instead.
* ``vmi[emitter]`` — Vocalization Modulation Index. For each emitter side: ``vmi`` in ``[-1, 1]``, paired Wilcoxon (the Wilcoxon signed-rank test) ``wilcoxon_statistic`` / ``wilcoxon_pvalue`` over the per-bout ``(FR_baseline, FR_USV)`` pairs, plus ``n_bouts``, ``fr_baseline_per_bout`` and ``fr_usv_per_bout`` arrays. ``VMI = (FR_USV − FR_baseline) / (FR_USV + FR_baseline)``, where the bouts are grouped from the side's own anchors (pure USVs only when ``exclude_squeaks_self`` / ``exclude_squeaks_partner`` is set for that side, as for the PETH and tuning anchors; the IBI bouts of ``ibi_bout_onset`` likewise), ``FR_baseline`` is the mean firing rate in the ``bout_quiet_seconds``-wide window before each bout and ``FR_USV`` is the mean over USVs in each bout of (spikes during USV) / (USV duration). Bouts whose baseline window starts before ``t = 0`` are NaN-baselined.
* ``usv_property_tuning[emitter][prop]`` — the same per-direction divergence-segment (run) analysis as the 1D behavioral axes above (``n_bins``, ``max_run``, ``run_start_idx`` / ``run_end_idx``, ``peak_idx``, ``peak_abs_z``, ``peak_signed_z``, ``selectivity``, ``monotonicity``, plus ``excit`` / ``suppress`` run segments).
* ``usv_peth[emitter]`` — a spike peri-event time histogram (PETH) around USV onset: ``peak_abs_z``, ``peak_signed_z``, ``peak_idx`` / ``peak_t``, ``ramp_index`` (a two-point pre-USV shape descriptor between ``ramp_anchor_far_seconds`` and ``ramp_anchor_near_seconds``), and ``excit`` / ``suppress`` run segments.
* ``usv_category_peth[emitter][cat_feat]`` — a per-category PETH: ``best_cat``, ``best_abs_z``, ``best_signed_z``, ``best_t_idx`` / ``best_t``, ``best_excit`` / ``best_suppress``, and ``per_category``.
* ``usv_ibi_bout_onset[emitter]`` — the ``ibi_bout_onset`` statistics (``onset_gain``, ``peak_latency``, ``pre_onset_gain``, ``first_over_later``, ``fraction_first_above``), each with its ``observed`` value, shuffle percentiles and ``p_two_sided``, plus ``n_bouts`` and ``ibi_threshold_seconds``. The aggregator labels a unit ``ibi_bout_onset_<role>_<excit|suppress>`` and ``ibi_bout_first_usv_<role>_<first|later>``, significant when ``p_two_sided`` is at most ``ibi_bout_alpha`` and the unit has at least ``ibi_bout_min_bouts`` bouts (``detect_interesting_tuning_neurons``).
* ``usv_category_tuning[emitter][cat_feat]`` — categorical (no run analysis): ``peak_abs_z``, ``peak_signed_z``, ``best_cat``, ``n_sig_categories`` (count of categories outside the [p0.5, p99.5] shuffle band), ``selectivity``.

Frequency shift audio segment
-----------------------------
For presentation purposes, one might want to play audio data of mouse USVs. Since these are beyond human audible range, the only way is to frequency-shift them several octaves down. To achieve this in the GUI, you need to list the root directories of interest, select *Frequency shift audio segment*, choose the start time and duration of the segment, click *Next* and then *Analyze*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_3.png
   :align: center
   :alt: Analysis Step 3

.. raw:: html

   <br>

The analysis results in the creation of a *frequency_shifted_audio_segments* subdirectory (if it is not already there) and a *wave* file in it containing the frequency-shifted segment. The volume and tempo passes run SoX with a time limit, and a SoX pass that fails or hangs raises instead of leaving a missing or stale intermediate file for the next step. This function, and both playback builders below, live in ``usv_playpen.analyses.generate_audio_files``:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ├── **frequency_shifted_audio_segments**
    │   │   │   ├── **m_20250430145035_ch01_cropped_to_video_hpss_filtered.wav_start=900.0s_duration=2.0s_octave_shift=-3_audible_denoised_tempo_adjusted.wav**
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

Below you can find an example of a brief sequence of frequency-shifted mouse vocalizations:

.. raw:: html

   <audio controls>
     <source src="https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/fs_example.wav" type="audio/wav">
     Your browser does not support the audio element.
   </audio>
   <br>
   <br>

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section only partially modifiable in the GUI, but it can be modified manually:

.. code-block:: json

    "frequency_shift_audio_segment": {
        "fs_audio_dir": "hpss_filtered",
        "fs_device_id": "m",
        "fs_channel_id": 1,
        "fs_wav_sampling_rate": 250,
        "fs_sequence_start": 900.0,
        "fs_sequence_duration": 2.0,
        "fs_octave_shift": -3,
        "fs_volume_adjustment": true,
        "fs_compand_transfer": "0.3,1 6:-70,-60,-20 -5 -90 0.2",
        "fs_noise_reduction_std_threshold": 3,
        "fs_sinc_upper_cutoff_hz": 25000,
        "fs_sox_timeout_s": 600
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`fs_audio_dir <ana-frequency-shift-audio-segment-fs-audio-dir>`
     - ``"hpss_filtered"``
     - ``"cropped_to_video"``, ``"hpss"``, ``"hpss_filtered"``
   * - :ref:`fs_device_id <ana-frequency-shift-audio-segment-fs-device-id>`
     - ``"m"``
     - ``"m"``, ``"s"``
   * - :ref:`fs_channel_id <ana-frequency-shift-audio-segment-fs-channel-id>`
     - ``1``
     - integer 1-12
   * - :ref:`fs_wav_sampling_rate <ana-frequency-shift-audio-segment-fs-wav-sampling-rate>`
     - ``250``
     - kHz
   * - :ref:`fs_sequence_start <ana-frequency-shift-audio-segment-fs-sequence-start>`
     - ``900.0``
     - seconds
   * - :ref:`fs_sequence_duration <ana-frequency-shift-audio-segment-fs-sequence-duration>`
     - ``2.0``
     - seconds
   * - :ref:`fs_octave_shift <ana-frequency-shift-audio-segment-fs-octave-shift>`
     - ``-3``
     - integer number of octaves
   * - :ref:`fs_volume_adjustment <ana-frequency-shift-audio-segment-fs-volume-adjustment>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`fs_compand_transfer <ana-frequency-shift-audio-segment-fs-compand-transfer>`
     - see JSON above
     - SoX ``compand`` argument string
   * - :ref:`fs_noise_reduction_std_threshold <ana-frequency-shift-audio-segment-fs-noise-reduction-std-threshold>`
     - ``3``
     - noise standard deviations
   * - :ref:`fs_sinc_upper_cutoff_hz <ana-frequency-shift-audio-segment-fs-sinc-upper-cutoff-hz>`
     - ``25000``
     - Hz
   * - :ref:`fs_sox_timeout_s <ana-frequency-shift-audio-segment-fs-sox-timeout-s>`
     - ``600``
     - seconds > 0

.. _ana-frequency-shift-audio-segment-fs-audio-dir:

``fs_audio_dir``
   Audio subdirectory where the audio files are stored.

   :Options:
      * ``"cropped_to_video"``
      * ``"hpss"``
      * ``"hpss_filtered"``
   :Default: ``"hpss_filtered"``
   :Notes: GUI field: *WAV audio subdirectory of choice* (a dropdown offering the three options above).

.. _ana-frequency-shift-audio-segment-fs-device-id:

``fs_device_id``
   USGH (Avisoft UltraSoundGate hardware) device ID.

   :Options:
      * ``"m"`` — main
      * ``"s"`` — secondary
   :Default: ``"m"``
   :Notes: GUI field: *Recording device identity (m|s)*.

.. _ana-frequency-shift-audio-segment-fs-channel-id:

``fs_channel_id``
   Microphone channel ID.

   :Options: an integer channel, 1-12
   :Default: ``1``
   :Notes: GUI field: *Recording device channel (1-12)*.

.. _ana-frequency-shift-audio-segment-fs-wav-sampling-rate:

``fs_wav_sampling_rate``
   Sampling rate of the audio devices in kHz.

   :Options: a sampling rate in kHz
   :Default: ``250``

.. _ana-frequency-shift-audio-segment-fs-sequence-start:

``fs_sequence_start``
   Start time of the audio segment in seconds.

   :Options: a number of seconds
   :Default: ``900.0``
   :Notes: GUI field: *Start of audio sequence (s)*.

.. _ana-frequency-shift-audio-segment-fs-sequence-duration:

``fs_sequence_duration``
   Duration of the audio segment in seconds.

   :Options: a positive number of seconds
   :Default: ``2.0``
   :Notes: GUI field: *Total duration of audio sequence (s)*.

.. _ana-frequency-shift-audio-segment-fs-octave-shift:

``fs_octave_shift``
   Octave shift of the audio segment.

   :Options: an integer number of octaves; e.g. -3 shifts down three octaves, to 1/8 the original frequency
   :Default: ``-3``
   :Notes: GUI field: *Octave shift (direction and quantity)*.

.. _ana-frequency-shift-audio-segment-fs-volume-adjustment:

``fs_volume_adjustment``
   Whether to automatically increase the volume of the audio segment.

   :Options:
      * ``true`` — automatically increase the volume (recommended since the vocalizations are faint)
      * ``false`` — leave the volume as it is
   :Default: ``true``
   :Notes: GUI field: *Volume-adjust audio segment*.

.. _ana-frequency-shift-audio-segment-fs-compand-transfer:

``fs_compand_transfer``
   SoX ``compand`` transfer-function argument string applied during the volume-adjustment pass (attack/decay + dB transfer points).

   :Options: a SoX ``compand`` argument string
   :Default: ``"0.3,1 6:-70,-60,-20 -5 -90 0.2"``

.. _ana-frequency-shift-audio-segment-fs-noise-reduction-std-threshold:

``fs_noise_reduction_std_threshold``
   ``noisereduce`` stationary noise-reduction threshold (in noise standard deviations).

   :Options: a positive number of noise standard deviations
   :Default: ``3``

.. _ana-frequency-shift-audio-segment-fs-sinc-upper-cutoff-hz:

``fs_sinc_upper_cutoff_hz``
   Upper cutoff (Hz) of the SoX ``sinc`` low-pass, scaled by the octave shift, applied to non-filtered source audio.

   :Options: a frequency in Hz
   :Default: ``25000``

.. _ana-frequency-shift-audio-segment-fs-sox-timeout-s:

``fs_sox_timeout_s``
   Upper bound (in s) on each SoX pass of the frequency shift.

   :Options: a positive number of seconds
   :Default: ``600``
   :Notes: the segments are presentation clips of seconds to minutes, which SoX processes in well under a second per minute of audio, so a pass that runs this long has hung (e.g. waiting on a broken pipe) and is stopped with an error rather than blocking the run.

Create artificial playback .WAV
-------------------------------
This function creates a .WAV file containing USV snippets. The snippets are randomly selected from the USV snippet repository in the specified directory and concatenated with inter-pulse intervals (IPIs) of a specified duration. The resulting .WAV file can be used for playback experiments. To achieve this in the GUI, select *Create artificial playback .WAV file* (no need to list root directories!), select total number of files to be created, number of vocalizations in each one, click *Next* and then *Analyze*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_4.png
   :align: center
   :alt: Analysis Step 4

.. raw:: html

   <br>

The analysis results in the creation of three files: [1] WAV file containing playback vocalizations, [2] a *spacing* text file informing you of the duration of each vocalization in order, and [3] a *usvids* text file containing the identity of each vocalization snippet if you need to go back and look at what it was:

.. parsed-literal::

    /mnt/falkner/Bartul/usv_playback_experiments/usv_playback_files
    ├── **usv_playback_n=10000_20250506_190808.wav**
    ├── **usv_playback_n=10000_20250506_190808_spacing.txt**
    ├── **usv_playback_n=10000_20250506_190808_usvids.txt**
    ...

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section only partially modifiable in the GUI, but it can be modified manually:

.. code-block:: json

    "create_usv_playback_wav": {
        "num_usv_files": 1,
        "total_usv_number": 10000,
        "ipi_duration": 0.015,
        "wav_sampling_rate": 250,
        "playback_snippets_dir": "usv_playback_snippets_loudness_corrected",
        "playback_seed": null
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`num_usv_files <ana-create-usv-playback-wav-num-usv-files>`
     - ``1``
     - integer > 0
   * - :ref:`total_usv_number <ana-create-usv-playback-wav-total-usv-number>`
     - ``10000``
     - integer > 0
   * - :ref:`ipi_duration <ana-create-usv-playback-wav-ipi-duration>`
     - ``0.015``
     - seconds
   * - :ref:`wav_sampling_rate <ana-create-usv-playback-wav-wav-sampling-rate>`
     - ``250``
     - kHz
   * - :ref:`playback_snippets_dir <ana-create-usv-playback-wav-playback-snippets-dir>`
     - see JSON above
     - subdirectory name
   * - :ref:`playback_seed <ana-create-usv-playback-wav-playback-seed>`
     - ``null``
     - ``null``, or an integer

.. _ana-create-usv-playback-wav-num-usv-files:

``num_usv_files``
   Number of artificial playback files to be created.

   :Options: a positive integer
   :Default: ``1``
   :Notes: GUI field: *Number of playback files*.

.. _ana-create-usv-playback-wav-total-usv-number:

``total_usv_number``
   Total number of USVs to be included in one playback file.

   :Options: a positive integer
   :Default: ``10000``
   :Notes: GUI field: *Total number of USVs per file*.

.. _ana-create-usv-playback-wav-ipi-duration:

``ipi_duration``
   Inter-pulse interval duration in seconds.

   :Options: a number of seconds
   :Default: ``0.015``
   :Notes: GUI field: *Fixed silence between USVs (s)*.

.. _ana-create-usv-playback-wav-wav-sampling-rate:

``wav_sampling_rate``
   Sampling rate of the playback .WAV file in kHz.

   :Options: a sampling rate in kHz
   :Default: ``250``

.. _ana-create-usv-playback-wav-playback-snippets-dir:

``playback_snippets_dir``
   Subdirectory where the USV snippets are stored.

   :Options: a subdirectory name
   :Default: ``"usv_playback_snippets_loudness_corrected"``

.. _ana-create-usv-playback-wav-playback-seed:

``playback_seed``
   Optional RNG seed for reproducible snippet selection and assembly.

   :Options:
      * ``null`` — draws a fresh random stimulus each run
      * an integer — reproduces the same stimulus
   :Default: ``null``

Build the naturalistic USV repository
-------------------------------------
Naturalistic playback replays **clean, real vocalizations**, so the vocalizations must first be reconstructed and stored in a **repository**. ``build-naturalistic-usv-repository`` reads one or more session lists, drops every segment the noise classifier flagged (the ``noise`` column ``detect-usv-noise`` writes; the number dropped is logged, and a session whose summary has no ``noise`` column is skipped and logged) **before** any bout is segmented — so noise neither enters the repository nor bridges or splits a bout; squeak and squeak+USV segments are not filtered — then segments each emitter's USVs into their natural **bouts**, reconstructs every USV of every *complete* bout directly from its masked spectrogram (an inverse STFT of the SAM mask — it does **not** slice the raw recording, so detector noise is excluded), and writes the clean audio together with each bout's membership, emission order, real within-bout and between-bout gaps, and every per-USV acoustic feature (including ``mask_number``).

Each run builds exactly **one (sex, social context) database**, chosen by ``context_label``. The **session lists you supply are the only filter** — the builder does not gate on experiment code, so you control coverage purely by which sessions you list. The output is written to::

    <naturalistic_usv_repository_dir>/<sex>/naturalistic_usv_repository_<context>_<datestring>.h5

where ``naturalistic_usv_repository_dir`` is configured once under the shared ``data_roots`` block, ``<sex>`` is ``male`` / ``female`` / ``mixed``, and ``<context>`` is the filename token (``courtship`` / ``same_sex`` / ``lone`` / ``mixed``). The datestring suffix means rebuilding a context never overwrites an earlier build, and each H5 also records the exact session lists it was built from (provenance). The H5 is written to a temporary file next to its final name and renamed into place, so an interrupted build never leaves a truncated repository for the playback step to read. A session whose inputs cannot be resolved (a missing or unreadable file, a missing column, or a failed consistency check) is skipped and logged, and every skipped session is listed again once the build finishes; an error while reconstructing a session's audio stops the build, since the repository would otherwise hold part of that session. To run it from the command line, see :doc:`CLI`.

The */usv-playpen/_parameter_settings/analyses_settings.json* ``build_naturalistic_usv_repository`` section:

.. code-block:: json

    "build_naturalistic_usv_repository": {
        "session_lists": [
            "/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt"
        ],
        "context_label": "courtship_female",
        "ibi_z_score": 2.58,
        "ibi_component_index": 0,
        "min_vocalizations": 2,
        "length_threshold": 128,
        "min_duration": 1,
        "mask_dilation": 0,
        "feather_sigma_time": 0.8,
        "fade_ms": 4.0,
        "peak_normalize": true,
        "peak_target_fraction": 0.85
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`session_lists <ana-build-naturalistic-usv-repository-session-lists>`
     - see JSON above
     - list of session-list text files
   * - :ref:`context_label <ana-build-naturalistic-usv-repository-context-label>`
     - ``"courtship_female"``
     - ``courtship_male``, ``courtship_female``, ``lone_male``, ``lone_female``, ``same_sex_male``, ``same_sex_female``, ``mixed``
   * - :ref:`ibi_z_score <ana-build-naturalistic-usv-repository-ibi-z-score>`
     - ``2.58``
     - standard deviations > 0
   * - :ref:`ibi_component_index <ana-build-naturalistic-usv-repository-ibi-component-index>`
     - ``0``
     - mixture component index
   * - :ref:`min_vocalizations <ana-build-naturalistic-usv-repository-min-vocalizations>`
     - ``2``
     - integer >= 1
   * - :ref:`length_threshold <ana-build-naturalistic-usv-repository-length-threshold>`
     - ``128``
     - time-bins
   * - :ref:`min_duration <ana-build-naturalistic-usv-repository-min-duration>`
     - ``1``
     - time-bins
   * - :ref:`mask_dilation <ana-build-naturalistic-usv-repository-mask-dilation>`
     - ``0``
     - bins >= 0
   * - :ref:`feather_sigma_time <ana-build-naturalistic-usv-repository-feather-sigma-time>`
     - ``0.8``
     - time-bins
   * - :ref:`fade_ms <ana-build-naturalistic-usv-repository-fade-ms>`
     - ``4.0``
     - ms
   * - :ref:`peak_normalize <ana-build-naturalistic-usv-repository-peak-normalize>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`peak_target_fraction <ana-build-naturalistic-usv-repository-peak-target-fraction>`
     - ``0.85``
     - fraction of the int16 ceiling

.. _ana-build-naturalistic-usv-repository-session-lists:

``session_lists``
   One or more text files (one session root directory per line) whose sessions this build draws from.

   :Options: a list of paths to session-list text files
   :Default: ``["/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt"]``
   :Notes: this list is the *sole* selection of what enters the repository.

.. _ana-build-naturalistic-usv-repository-context-label:

``context_label``
   Which (sex, social context) database to build.

   :Options:
      * ``courtship_male`` / ``courtship_female`` — keep only the target sex's attributed track
      * ``lone_male`` / ``lone_female`` — keep **every** USV labelled by that sex
      * ``same_sex_male`` / ``same_sex_female`` — keep **every** USV labelled by that sex
      * ``mixed`` — keep every USV with no sex split
   :Default: ``"courtship_female"``
   :Notes: it sets three things at once: (1) *emitter handling* — a ``courtship_*`` build keeps only the target sex's attributed track (the two animals differ in sex; the target animal is the one the session metadata's ``Subjects`` records with that sex, never a track slot, and a session whose metadata does not name exactly one such animal is skipped and logged), while ``same_sex_*`` / ``lone_*`` keep **every** USV labelled by that sex (no per-USV attribution needed), and ``mixed`` keeps every USV with no sex split; (2) the output *subdirectory* (``male`` / ``female`` / ``mixed``); and (3) the filename *context token*.

.. _ana-build-naturalistic-usv-repository-ibi-z-score:

``ibi_z_score``
   How many standard deviations above the mean (in log inter-USV-interval space) a silent gap must be to count as a **bout boundary**.

   :Options: a positive number of standard deviations
   :Default: ``2.58`` (≈ the 99.5th percentile of the one-sided cutoff)
   :Notes: the split threshold is ``exp(μ + z·σ)``, with the per-sex ``μ`` and ``σ`` taken from the modeling mixture model.

.. _ana-build-naturalistic-usv-repository-ibi-component-index:

``ibi_component_index``
   Which component of the per-sex log-inter-USV-interval mixture supplies that ``μ`` and ``σ``.

   :Options: a mixture component index; ``0`` = the first / fastest component
   :Default: ``0``

.. _ana-build-naturalistic-usv-repository-min-vocalizations:

``min_vocalizations``
   The fewest USVs a bout must contain to be kept.

   :Options: a positive integer number of USVs
   :Default: ``2`` (single-USV bouts are dropped)

.. _ana-build-naturalistic-usv-repository-length-threshold:

``length_threshold``
   The maximum USV duration in spectrogram **time-bins** — a bout is discarded if any of its USVs is longer than this.

   :Options: a number of time-bins
   :Default: ``128``
   :Notes: the stored mask is only 128 columns wide and would otherwise truncate the USV.

.. _ana-build-naturalistic-usv-repository-min-duration:

``min_duration``
   The minimum USV duration in time-bins — a bout is discarded if any of its USVs is shorter.

   :Options: a number of time-bins
   :Default: ``1``

.. _ana-build-naturalistic-usv-repository-mask-dilation:

``mask_dilation``
   Grow the SAM mask by this many bins before inversion.

   :Options: a non-negative number of bins; ``0`` keeps the mask **tight**
   :Default: ``0``
   :Notes: dilation pulls in neighbouring energy and muddies the reconstruction.

.. _ana-build-naturalistic-usv-repository-feather-sigma-time:

``feather_sigma_time``
   Gaussian sigma (in time-bins) of the **time-only** mask feather that softens each USV's onset/offset to remove edge clicks.

   :Options: a sigma in time-bins
   :Default: ``0.8``
   :Notes: feathering is applied in time only — a frequency feather bleeds a halo above and below the USV.

.. _ana-build-naturalistic-usv-repository-fade-ms:

``fade_ms``
   Length (ms) of the raised-cosine onset/offset fade applied to each reconstructed USV (capped at a third of that USV's length), removing any residual edge click.

   :Options: a length in ms
   :Default: ``4.0``

.. _ana-build-naturalistic-usv-repository-peak-normalize:

``peak_normalize``
   Whether every USV is peak-normalized to a uniform level.

   :Options:
      * ``true`` — peak-normalize every USV to a uniform level
      * ``false`` — preserve the USVs' relative amplitudes
   :Default: ``true``

.. _ana-build-naturalistic-usv-repository-peak-target-fraction:

``peak_target_fraction``
   The fraction of the int16 ceiling each peak-normalized snippet is scaled to.

   :Options: a fraction of the int16 ceiling
   :Default: ``0.85`` (leaves headroom below clipping)

The repository root itself lives under the shared ``data_roots`` block: :ref:`naturalistic_usv_repository_dir <ana-data-roots-naturalistic-usv-repository-dir>` is the root directory holding the ``male`` / ``female`` / ``mixed`` subdirectories that the builds are written into and the playback step reads from.

Create naturalistic playback .WAV
---------------------------------
This function creates a .WAV file that replays **real vocalization bouts** drawn from a naturalistic USV repository (built by ``build-naturalistic-usv-repository``, above). Instead of stitching randomly-chosen single snippets, it replays whole recorded bouts in their natural emission order (``1-2-3-4-5-6``) with their real timing, so the playback reproduces natural USV sequences. The resulting .WAV file can be used for playback experiments.
To achieve this in the GUI, select *Create naturalistic playback .WAV file* (no need to list root directories!), set the parameters described below — the **vocalization context** (a dropdown listing only the contexts that have actually been built; a context picked before that listing returns is kept when it is among them), the **playback duration** and **number of files**, the optional **complexity steering** (a Yes/No dropdown plus threshold / start / end / bandwidth sliders that grey out when it is off), and the **playback seed** — then click *Next* and *Analyze*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_5.png
   :align: center
   :alt: Analysis Step 5

.. raw:: html

   <br>

From ``context_label`` the generator opens the **newest** matching repository H5 under ``<naturalistic_usv_repository_dir>/<sex>/``, which holds the clean reconstructed audio plus the natural bout structure (which USVs belong to each bout, their order, and the real within-bout and between-bout gaps). It writes a fixed ``edge_silence_seconds`` lead-in silence, then repeats:

* draw a random **intact bout** and append its USVs in their natural emission order, separated by their real within-bout **inter-USV intervals (IUI)** of silence
* between bouts (not before the first), insert the drawn bout's real preceding **inter-sequence interval (ISI)** of silence — a session-first bout (no recorded pause) uses a sampled real ISI — **clipped to** ``max_isi_seconds`` so no single pause is huge

It keeps adding whole bouts, skipping any that (with the lead-out) would overrun the target, until ``max_consecutive_bout_skips`` draws in a row have failed to fit (the remaining time is too small); it then writes a fixed ``edge_silence_seconds`` lead-out silence. The within-bout and between-bout gaps are genuine recorded values (no interval model); a negative within-bout gap (two overlapping USVs, possible in a repository that keeps both animals' USVs unattributed) is played as no silence. Because the file is built of whole bouts framed by the fixed edge silences, it is **up to** ``total_acceptable_naturalistic_playback_time`` seconds — opening and closing on the edge silence and ending on a complete bout — rather than exactly that length.

The three files are written into the ``naturalistic_usv_playback_dir`` directory configured under the shared ``data_roots`` block. The analysis creates: [1] the WAV file, [2] a *spacing* text file with the sample count of each chunk in order, and [3] a *usvids* text file with the identity of each chunk — ``<session>_usv<row>`` for a vocalization (so you can trace it back to the source session and USV), or ``ISI`` / ``IUI`` for a silence gap:

.. parsed-literal::

    /mnt/falkner/Bartul/usv_playback_experiments/usv_naturalistic_playback_files
    ├── **female_usv_playback_1080s_20250506_190808.wav**
    ├── **female_usv_playback_1080s_20250506_190808_spacing.txt**
    ├── **female_usv_playback_1080s_20250506_190808_usvids.txt**
    ...

Every setting below is exposed in the GUI's *Create naturalistic playback .WAV* block: ``context_label`` is a dropdown (populated only with contexts that have a built repository); ``total_acceptable_naturalistic_playback_time``, ``num_naturalistic_usv_files`` and ``playback_seed`` are text fields; ``complexity_enabled`` is a Yes/No dropdown; and ``complexity_mask_threshold``, ``complexity_start_fraction``, ``complexity_end_fraction`` and ``complexity_bandwidth`` are sliders that grey out while complexity is disabled; ``edge_silence_seconds``, ``max_isi_seconds`` and ``max_consecutive_bout_skips`` are edited in the JSON only. The same values can equally be edited directly in *analyses_settings.json*:

.. code-block:: json

    "create_naturalistic_usv_playback_wav": {
        "num_naturalistic_usv_files": 1,
        "context_label": "courtship_male",
        "total_acceptable_naturalistic_playback_time": 60,
        "complexity_enabled": false,
        "complexity_mask_threshold": 2,
        "complexity_start_fraction": 0.0,
        "complexity_end_fraction": 1.0,
        "complexity_bandwidth": 0.15,
        "edge_silence_seconds": 1.0,
        "max_isi_seconds": 12.6,
        "max_consecutive_bout_skips": 100,
        "playback_seed": null
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`num_naturalistic_usv_files <ana-create-naturalistic-usv-playback-wav-num-naturalistic-usv-files>`
     - ``1``
     - integer > 0
   * - :ref:`context_label <ana-create-naturalistic-usv-playback-wav-context-label>`
     - ``"courtship_male"``
     - ``courtship_male``, ``courtship_female``, ``lone_male``, ``lone_female``, ``same_sex_male``, ``same_sex_female``, ``mixed``
   * - :ref:`total_acceptable_naturalistic_playback_time <ana-create-naturalistic-usv-playback-wav-total-acceptable-naturalistic-playback-time>`
     - ``60``
     - seconds > 0
   * - :ref:`complexity_enabled <ana-create-naturalistic-usv-playback-wav-complexity-enabled>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`complexity_mask_threshold <ana-create-naturalistic-usv-playback-wav-complexity-mask-threshold>`
     - ``2``
     - integer mask count
   * - :ref:`complexity_start_fraction <ana-create-naturalistic-usv-playback-wav-complexity-fractions>`
     - ``0.0``
     - fraction 0–1
   * - :ref:`complexity_end_fraction <ana-create-naturalistic-usv-playback-wav-complexity-fractions>`
     - ``1.0``
     - fraction 0–1
   * - :ref:`complexity_bandwidth <ana-create-naturalistic-usv-playback-wav-complexity-bandwidth>`
     - ``0.15``
     - complex-fraction units
   * - :ref:`edge_silence_seconds <ana-create-naturalistic-usv-playback-wav-edge-silence-seconds>`
     - ``1.0``
     - seconds
   * - :ref:`max_isi_seconds <ana-create-naturalistic-usv-playback-wav-max-isi-seconds>`
     - ``12.6``
     - seconds
   * - :ref:`max_consecutive_bout_skips <ana-create-naturalistic-usv-playback-wav-max-consecutive-bout-skips>`
     - ``100``
     - integer > 0
   * - :ref:`playback_seed <ana-create-naturalistic-usv-playback-wav-playback-seed>`
     - ``null``
     - ``null``, or an integer

.. _ana-create-naturalistic-usv-playback-wav-num-naturalistic-usv-files:

``num_naturalistic_usv_files``
   Number of naturalistic playback files to create in one run.

   :Options: a positive integer
   :Default: ``1``
   :Notes: with a fixed ``playback_seed`` the files differ from one another but the whole set is reproducible. GUI field: *Number of generated files* (a text field).

.. _ana-create-naturalistic-usv-playback-wav-context-label:

``context_label``
   Which repository to play back.

   :Options: one of ``courtship_male``, ``courtship_female``, ``lone_male``, ``lone_female``, ``same_sex_male``, ``same_sex_female``, ``mixed``
   :Default: ``"courtship_male"``
   :Notes: it selects the sex subdirectory and context token, and the playback always opens the **newest** matching build in that directory (no explicit file path); the chosen sex also prefixes the output filenames. GUI field: *Choose vocalization context* (a dropdown populated only with contexts that have a built repository).

.. _ana-create-naturalistic-usv-playback-wav-total-acceptable-naturalistic-playback-time:

``total_acceptable_naturalistic_playback_time``
   The target duration of the playback file (in s).

   :Options: a positive number of seconds
   :Default: ``60``
   :Notes: since the file is assembled from whole bouts framed by the fixed edge silences, the result is *up to* this length (a little under), ending on a complete bout. GUI field: *Playback duration (s)* (a text field).

.. _ana-create-naturalistic-usv-playback-wav-complexity-enabled:

``complexity_enabled``
   Master switch for complexity steering.

   :Options:
      * ``false`` — bouts are drawn **uniformly at random** (the natural mix) and the remaining ``complexity_*`` settings are inert
      * ``true`` — the draws are biased by USV complexity
   :Default: ``false``
   :Notes: GUI field: *Complexity enabled* (a Yes/No dropdown).

.. _ana-create-naturalistic-usv-playback-wav-complexity-mask-threshold:

``complexity_mask_threshold``
   With steering on, a USV is *complex* if its ``mask_number`` (count of SAM masks) is ``>=`` this value, else *simple*.

   :Options: an integer mask count (``2`` = multi-component)
   :Default: ``2``
   :Notes: GUI field: the *Complexity threshold* slider (2 to 6), greyed out while complexity is disabled.

.. _ana-create-naturalistic-usv-playback-wav-complexity-fractions:

``complexity_start_fraction`` / ``complexity_end_fraction``
   The target fraction of *complex* USVs at the **start** and **end** of the file (0–1), interpolated linearly across the file position.

   :Options:
      * equal values — hold a constant simple:complex ratio (e.g. ``1.0 / 1.0`` = all-complex throughout)
      * unequal values — a ramp; ``0.0 → 1.0`` starts simple and grows more complex
   :Default: ``0.0`` and ``1.0``
   :Notes: because bouts are drawn with replacement, a target is met by repeating complex-heavy bouts as needed (whole bouts, natural order preserved), bounded by what the available bouts contain; the achieved ratio and number of distinct bouts used are logged. GUI fields: the *Start fraction* and *End fraction* sliders, greyed out while complexity is disabled.

.. _ana-create-naturalistic-usv-playback-wav-complexity-bandwidth:

``complexity_bandwidth``
   The width (standard deviation, in complex-fraction units) of the Gaussian that weights each bout by how far its own complex-fraction sits from the current target.

   :Options: a width in complex-fraction units; **smaller** (e.g. ``0.05``) steers tightly to the target but from a narrow pool of bouts (more repetition); **larger** (e.g. ``0.5``) keeps more variety but tracks the target more loosely
   :Default: ``0.15``, a moderate middle
   :Notes: GUI field: the *Steering bandwidth* slider, greyed out while complexity is disabled.

.. _ana-create-naturalistic-usv-playback-wav-edge-silence-seconds:

``edge_silence_seconds``
   Fixed silence written at the very start and very end of the file (s), so it opens and closes on a short, constant gap rather than a variable (or truncated) real pause.

   :Options: a number of seconds
   :Default: ``1.0``
   :Notes: edited in the JSON only.

.. _ana-create-naturalistic-usv-playback-wav-max-isi-seconds:

``max_isi_seconds``
   Each inserted inter-bout pause (ISI) is clipped to at most this many seconds, so a single unusually long recorded gap cannot drop a giant silence into the file.

   :Options: a number of seconds
   :Default: ``12.6`` (a typical courtship inter-bout gap)
   :Notes: edited in the JSON only.

.. _ana-create-naturalistic-usv-playback-wav-max-consecutive-bout-skips:

``max_consecutive_bout_skips``
   How many bout draws in a row may fail to fit the remaining time (with the lead-out) before the file is closed.

   :Options: a positive integer number of draws
   :Default: ``100``
   :Notes: once that many random draws have all overrun, no bout in the repository is likely to fit, so the loop stops rather than drawing forever. Edited in the JSON only.

.. _ana-create-naturalistic-usv-playback-wav-playback-seed:

``playback_seed``
   RNG seed for the bout draws.

   :Options:
      * ``null`` — draws a fresh random stimulus every run (non-reproducible)
      * an integer — reproduces the exact same file(s)
   :Default: ``null``
   :Notes: GUI field: *Playback seed (blank=random)* (a text field; blank writes ``null``).

.. note::

   The generated ``.wav`` is written at the repository's native **250 kHz** sampling rate.
   The playback audio device expects **192 kHz**, so change the audio's sampling rate from
   250 kHz to 192 kHz before playback (resample the file to 192 kHz, or set the output rate
   on the playback command).

Decode vocal-bout occupancy
---------------------------------
This analysis asks whether a moment sits **inside a vocal bout** can be read off simultaneously
recorded spiking. It runs per session, alongside the neuronal tuning curves, and is switched on
with ``decode_usv_bout_occupancy_bool``.

A bout is found on the **USV sequence first**, as a maximal run of the emitter's USVs whose
consecutive silences all stay under the inter-bout threshold — the emitter sex's log
inter-syllable-interval mixture component at ``ibi_z_score`` standard deviations, the same
construction ``build-naturalistic-usv-repository`` uses, which for the male at z = 2.58 puts the
threshold at 115.5 ms. Runs shorter than ``min_vocalizations`` are not bouts.

The bins then follow the bouts rather than the other way round:

* a **positive** bin is laid from a bout's own first USV start, stepping forward by
  ``bin_seconds`` while it stays inside that bout, and contributes **only its silent time**, so
  no window the decoder reads ever contains a vocalization. It needs at least
  ``min_kept_seconds`` of silent time to count at all.
* a **negative** bin comes from the global lattice and must hold no vocalization and touch no bout.
* everything else is **dropped**. A bin holding a single USV cannot be assigned: a 120 ms USV
  followed by a 110 ms silence is one USV per bin and squarely inside a bout, so a USV count
  cannot say which side such a bin belongs to.

Bouts are grouped from the emitter's pure USVs, but silence is not class-specific: a positive
bin's kept time and a negative bin must both be free of every vocalization of the emitter (pure
USVs, pure squeaks, overlap segments and unscored segments alike) and of every vocalization
assigned to no animal, so a squeak, or an overlap segment, which holds a USV, never sits inside a
window labelled silent.

Each bin becomes a population vector of **raw** spike counts — nothing is divided by time and
nothing is standardised — three contiguous thirds rotate as train and test, and a test bin takes
the label of the training bin its vector correlates with most. Accuracy is balanced across the two
classes, because the negatives outnumber the positives several times over. The sweep repeats with
increasingly large random subsets of each area's units, so the curve says how much of the
population the signal needs. The null applies **one common circular shift** to every unit, which
preserves the correlations between them and breaks only their alignment to vocalization. Each
rung's observed accuracy is the mean over ``runs_per_count`` random unit subsets, and so is each of
its ``n_shuffles`` null draws (one shift, ``runs_per_count`` subsets): the p-value
``(1 + #{null >= observed}) / (n_shuffles + 1)`` compares like with like, at the cost of
``runs_per_count`` decodes per null draw.

.. warning::

   Per-unit z-scoring must not be introduced here. Under the shift null a positive bin's counts
   are a fixed fraction of a negative bin's for *every* unit, because it contributes less
   counting time, and Pearson correlation is blind to that uniform factor. Standardising each
   unit first turns it into a per-unit offset that varies across units, which correlation *can*
   read: measured on one session it lifted the 415-unit null from 0.509 to 0.751.

The session's units are split by brain area from the unit catalog on their own — no area list is
needed — and every area clearing ``min_units_per_area`` is decoded into its own compressed
archive, carrying every observed and every null draw. Areas seen but held back by the floor are
reported by name, so an absent area is never ambiguous. Sessions are independent and **nothing is
pooled**: assembling across them is a later step, done in the notebook (see :doc:`Notebooks`).

.. code-block:: json

    "decode_usv_bout_occupancy": {
      "emitter": "self",
      "ibi_z_score": 2.58,
      "ibi_component_index": 0,
      "min_vocalizations": 3,
      "bin_seconds": 0.25,
      "min_kept_seconds": 0.03,
      "brain_areas": null,
      "min_units_per_area": 50,
      "cluster_group": ["good"],
      "somatic_bool": true,
      "unit_ladder": [5, 10, 20, 50, 100, 200],
      "runs_per_count": 10,
      "n_folds": 3,
      "n_shuffles": 1000,
      "shuffle_guard_seconds": 20.0,
      "shuffle_seed": 0,
      "catalog_path": "/mnt/falkner/Bartul/EPHYS/unit_catalog.csv",
      "output_directory": "/mnt/falkner/Bartul/EPHYS/bout_decoding"
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`emitter <ana-decode-usv-bout-occupancy-emitter>`
     - ``"self"``
     - ``self``, ``partner``, ``all``
   * - :ref:`ibi_z_score <ana-decode-usv-bout-occupancy-threshold>`
     - ``2.58``
     - standard deviations > 0
   * - :ref:`ibi_component_index <ana-decode-usv-bout-occupancy-threshold>`
     - ``0``
     - mixture component index
   * - :ref:`min_vocalizations <ana-decode-usv-bout-occupancy-min-vocalizations>`
     - ``3``
     - integer >= 1
   * - :ref:`bin_seconds <ana-decode-usv-bout-occupancy-bin-seconds>`
     - ``0.25``
     - seconds > 0
   * - :ref:`min_kept_seconds <ana-decode-usv-bout-occupancy-min-kept-seconds>`
     - ``0.03``
     - seconds
   * - :ref:`brain_areas <ana-decode-usv-bout-occupancy-brain-areas>`
     - ``null``
     - ``null``, or a list of area names
   * - :ref:`min_units_per_area <ana-decode-usv-bout-occupancy-min-units-per-area>`
     - ``50``
     - integer > 0
   * - :ref:`cluster_group <ana-decode-usv-bout-occupancy-cluster-group>`
     - ``["good"]``
     - list of catalog ``cluster_group`` labels
   * - :ref:`somatic_bool <ana-decode-usv-bout-occupancy-somatic-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`unit_ladder <ana-decode-usv-bout-occupancy-unit-ladder>`
     - ``[5, 10, 20, 50, 100, 200]``
     - list of unit counts
   * - :ref:`runs_per_count <ana-decode-usv-bout-occupancy-runs-per-count>`
     - ``10``
     - integer > 0
   * - :ref:`n_folds <ana-decode-usv-bout-occupancy-n-folds>`
     - ``3``
     - integer > 1
   * - :ref:`n_shuffles <ana-decode-usv-bout-occupancy-n-shuffles>`
     - ``1000``
     - integer > 0
   * - :ref:`shuffle_guard_seconds <ana-decode-usv-bout-occupancy-shuffle-guard-seconds>`
     - ``20.0``
     - seconds
   * - :ref:`shuffle_seed <ana-decode-usv-bout-occupancy-shuffle-seed>`
     - ``0``
     - integer seed
   * - :ref:`catalog_path <ana-decode-usv-bout-occupancy-catalog-path>`
     - path
     - path to the unit catalog CSV
   * - :ref:`output_directory <ana-decode-usv-bout-occupancy-output-directory>`
     - path
     - path to a directory

.. _ana-decode-usv-bout-occupancy-emitter:

``emitter``
   Whose USVs define the bouts and the silence.

   :Options:
      * ``self`` — the recorded animal
      * ``partner`` — the other animal
      * ``all`` — no emitter filter
   :Default: ``"self"``
   :Notes: the inter-bout threshold follows the sex of whichever animal is vocalizing, so a female side is not segmented by a male rhythm. Under ``all`` the emitter set is mixed (and includes USVs assigned to no animal), so the recorded animal's sex is used.

.. _ana-decode-usv-bout-occupancy-threshold:

``ibi_z_score`` / ``ibi_component_index``
   The inter-bout threshold: the emitter sex's log inter-syllable-interval mixture component (``ibi_component_index``) at ``ibi_z_score`` standard deviations.

   :Options:
      * ``ibi_z_score`` — a positive number of standard deviations
      * ``ibi_component_index`` — the index of the emitter sex's mixture component
   :Default: ``2.58`` and ``0``
   :Notes: the same construction ``build-naturalistic-usv-repository`` uses, which for the male at z = 2.58 puts the threshold at 115.5 ms. A bout is a maximal run of the emitter's USVs whose consecutive silences all stay under this threshold.

.. _ana-decode-usv-bout-occupancy-min-vocalizations:

``min_vocalizations``
   Runs shorter than this many USVs are not bouts.

   :Options: a positive integer number of USVs
   :Default: ``3``

.. _ana-decode-usv-bout-occupancy-bin-seconds:

``bin_seconds``
   The bin width: a positive bin is laid from a bout's own first USV start, stepping forward by this much while it stays inside that bout, and negative bins come from the global lattice of this width.

   :Options: a positive number of seconds
   :Default: ``0.25``

.. _ana-decode-usv-bout-occupancy-min-kept-seconds:

``min_kept_seconds``
   The silent time a positive bin needs to count at all.

   :Options: a number of seconds
   :Default: ``0.03``

.. _ana-decode-usv-bout-occupancy-brain-areas:

``brain_areas``
   Which brain areas to decode.

   :Options:
      * ``null`` — every area clearing the unit floor (``min_units_per_area``)
      * a list of area names — only those areas; a requested area absent from the session is reported
   :Default: ``null``
   :Notes: the session's units are split by brain area from the unit catalog on their own, so no area list is needed.

.. _ana-decode-usv-bout-occupancy-min-units-per-area:

``min_units_per_area``
   The unit floor: every area clearing it is decoded into its own compressed archive.

   :Options: a positive integer number of units
   :Default: ``50``
   :Notes: areas seen but held back by the floor are reported by name.

.. _ana-decode-usv-bout-occupancy-cluster-group:

``cluster_group``
   Only catalog units whose ``cluster_group`` is in this list are decoded.

   :Options: a list of catalog ``cluster_group`` labels
   :Default: ``["good"]``

.. _ana-decode-usv-bout-occupancy-somatic-bool:

``somatic_bool``
   Whether only units the catalog marks as somatic are decoded.

   :Options:
      * ``true`` — somatic units only
      * ``false`` — no somatic filter
   :Default: ``true``

.. _ana-decode-usv-bout-occupancy-unit-ladder:

``unit_ladder``
   The unit-subset sizes of the sweep over increasingly large random subsets of each area's units.

   :Options: a list of unit counts
   :Default: ``[5, 10, 20, 50, 100, 200]``
   :Notes: only the counts smaller than an area's unit count are used, and the area's full unit count is always added as the last rung.

.. _ana-decode-usv-bout-occupancy-runs-per-count:

``runs_per_count``
   How many random unit subsets each rung's observed accuracy, and each of its null draws, is averaged over.

   :Options: a positive integer
   :Default: ``10``
   :Notes: costs ``runs_per_count`` decodes per null draw.

.. _ana-decode-usv-bout-occupancy-n-folds:

``n_folds``
   The number of contiguous time blocks that rotate as train and test.

   :Options: an integer of at least 2
   :Default: ``3`` (three contiguous thirds)

.. _ana-decode-usv-bout-occupancy-n-shuffles:

``n_shuffles``
   The number of null draws per rung, each one common circular shift applied to every unit.

   :Options: a positive integer
   :Default: ``1000``
   :Notes: the p-value is ``(1 + #{null >= observed}) / (n_shuffles + 1)``.

.. _ana-decode-usv-bout-occupancy-shuffle-guard-seconds:

``shuffle_guard_seconds``
   The null's circular shifts are drawn uniformly from ``[guard, duration - guard]`` (in s).

   :Options: a number of seconds
   :Default: ``20.0``

.. _ana-decode-usv-bout-occupancy-shuffle-seed:

``shuffle_seed``
   Seed of the random generator that draws the unit subsets, the circular shifts and the nearest-neighbour classifier's random choices.

   :Options: an integer seed
   :Default: ``0``
   :Notes: the generator is re-seeded for every area.

.. _ana-decode-usv-bout-occupancy-catalog-path:

``catalog_path``
   The unit catalog the brain areas and unit filters are read from.

   :Options: path to the unit catalog CSV
   :Default: ``/mnt/falkner/Bartul/EPHYS/unit_catalog.csv``
   :Notes: re-keyed to the experimenter in use.

.. _ana-decode-usv-bout-occupancy-output-directory:

``output_directory``
   Where each area's compressed archive is written.

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/EPHYS/bout_decoding``
   :Notes: re-keyed to the experimenter in use.

A unit's spike file is looked up by name under the session's ``ephys`` tree; a name matching no file,
or more than one, raises instead of decoding whichever file sorts first. The decoder is implemented
in ``usv_playpen.analyses.decode_usv_bout_occupancy``.

.. note::

   This bout grouping is **not** the ``bout_quiet_seconds`` of the neuronal tuning curves. That
   one is the VMI baseline window, and groups USVs only as a side effect of guaranteeing a
   silent pre-window; this one describes the respiratory rhythm. Both are correct for their own
   analysis, and the tuning curves now carry their own IBI-based ``ibi_bout_onset`` block for
   per-unit bout-onset statistics.
