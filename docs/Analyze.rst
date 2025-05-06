.. _Analyze:

Analyze
==================
This page explains how to use the data analyses functionalities in the *usv-playpen* GUI.

In order to run any of the functions detailed below, you first click the *Analyze* button on the GUI main display.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_0a.png
   :align: center
   :alt: Analysis Step 0a

.. raw:: html

   <br>

Clicking the *Analyze* button will open a new window with all the offered functionalities (see below).

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_0b.png
   :align: center
   :alt: Analysis Step 0b

.. raw:: html

   <br>

All the main functions are outlined in orange and black fields are specific options tunable by the user in the GUI. It is important to note that these are not necessarily *all* the options the user can set, and the full list of options can be found under each function in the */usv-playpen/_parameter_settings/analyses_settings.json* file. Each time the user clicks the *Next* button in the window above, *analyses_settings.json* is modified to the newest input configuration.

The *Root directories* field enables you to list the directories containing the data you want to analyze. Each root directory should be in its **own row**; for example, three sessions should be listed as follows:

.. parsed-literal::

    F:\\Bartul\\Data\\20250430_145017
    F:\\Bartul\\Data\\20250430_165730
    F:\\Bartul\\Data\\20250430_182145

Compute 3D behavioral features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once 3D tracking data is available, you can compute behavioral features. These can be *individual features* specific to each mouse (*i.e.*, spatial location, speed, posture, *etc.*) or *social features* (assuming two or more mice) that describe the relationship between the mice (*i.e.*, distance, angle, *etc.*). The code does not yet have the functionality to analyze multi-mouse features (>2 mice). The output of this analysis are two files: [1] CSV file containing each measured feature in each column, and [2] a PDF file containing graphs for the observed distribution of each feature. To run this analysis in the GUI, you need to list the root directories of interest, select *Compute 3D behavioral features*, click *Next* and then *Analyze*.

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

An example of typical individual and social feature distributions is shown below:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_behavioral_features_1.png
   :align: center
   :alt: Behavioral Features Example 1

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_behavioral_features_2.png
   :align: center
   :alt: Behavioral Features Example 2

.. raw:: html

   <br>

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/example_behavioral_features_3.png
   :align: center
   :alt: Behavioral Features Example 3

.. raw:: html

   <br>

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section not modifiable in the GUI itself, but it can be modified manually:

* **head_points** : head skeleton node names (order matters!)
* **tail_points** : tail skeleton node names (order matters!)
* **back_root_points** : back skeleton node names (order matters!)
* **derivative_bins** : number of bins to compute derivatives over

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
        "derivative_bins": 10
  }

Compute 3D feature tuning curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having recorded unit activity and social behavior, you might be interested whether individual units encode specific behavioral features. To get at this, you can compute session-averaged *tuning curves* capturing the relationship between the firing rate of each unit and each behavioral feature of interest. To achieve this in the GUI, you need to list the root directories of interest, select *Compute 3D feature tuning curves*, click *Next* and then *Analyze* (a progress bar will appear in the terminal while the analysis is running).

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_2.png
   :align: center
   :alt: Analysis Step 2

.. raw:: html

   <br>

The analysis results in the creation of a *tuning_curves* subdirectory containing a *pickle file* for each recorded unit:

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

* **temporal_offsets** : list of temporal offsets between spikes and behavior (in seconds, negative values: spikes precede behavior) for which the tuning curves will be calculated (adding values to the list will increase the time needed for analysis drastically)
* **n_shuffles** : number of spike train shuffles (increasing this number increases the time needed for analysis drastically)
* **total_bin_num** : total number of bins for a 1D behavioral feature
* **n_spatial_bins** : number of spatial bins (2D behavioral feature)
* **spatial_scale_cm** : maximum distance from center of arena to one edge (in cm)

.. code-block:: json

    "calculate_neuronal_tuning_curves": {
        "temporal_offsets": [
          0
        ],
        "n_shuffles": 1000,
        "total_bin_num": 36,
        "n_spatial_bins": 196,
        "spatial_scale_cm": 32
    }

Create USV playback .WAV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function creates a .WAV file containing USV snippets. The snippets are randomly selected from the USVs in the specified directory and concatenated with inter-pulse intervals (IPIs) of a specified duration. The resulting .WAV file can be used for playback experiments. To achieve this in the GUI, select *Create USV playback .WAV file* (no need to list root directories!), select total number of files to be created, number of vocalizations in each one, click *Next* and then *Analyze*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_4.png
   :align: center
   :alt: Analysis Step 4

.. raw:: html

   <br>

The analysis results in the creation of three files: [1] WAV file containing playback vocalizations, [2] a *spacing* text file informing you of the duration of each vocalization in order, and [3] a *usvids* text file containing the identity of each vocalization snippet if you need to go back and look at what it was:

.. parsed-literal::

    F:\\Bartul\\usv_playback_experiments\\usv_playback_files
    ├── **usv_playback_n=10000_20250506_190808.wav**
    ├── **usv_playback_n=10000_20250506_190808_spacing.txt**
    ├── **usv_playback_n=10000_20250506_190808_usvids.txt**
    ...

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section only partially modifiable in the GUI, but it can be modified manually:

* **num_usv_files** : number of USV files to be created
* **total_usv_number** : total number of USVs to be included in one playback file
* **ipi_duration** : inter-pulse interval duration in seconds
* **wav_sampling_rate** : sampling rate of the playback .WAV file in kHz
* **playback_snippets_dir** : subdirectory where the USV snippets are stored

.. code-block:: json

    "create_usv_playback_wav": {
        "num_usv_files": 1,
        "total_usv_number": 10000,
        "ipi_duration": 0.015,
        "wav_sampling_rate": 250,
        "playback_snippets_dir": "usv_playback_snippets_loudness_corrected"
    }

Frequency shift audio segment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For presentation purposes, one might want to play audio data of mouse USVs. Since these are beyond human audible range, the only way is to frequency-shift them several octaves down. To achieve this in the GUI, you need to list the root directories of interest, select *Frequency shift audio segment*, choose the start time and duration of the segment, click *Next* and then *Analyze*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_5.png
   :align: center
   :alt: Analysis Step 5

.. raw:: html

   <br>

The analysis results in the creation of a *frequency_shifted_audio_segments* subdirectory (if it is not already there) and a file *wave* in it containing the frequency-shifted segment:

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

* **fs_audio_dir** : audio subdirectory where the audio files are stored
* **fs_device_id** : USGH device ID (e.g. "m" for main, "s" for secondary)
* **fs_channel_id** : microphone channel ID (1-12)
* **fs_wav_sampling_rate** : sampling rate of the audio devices in kHz
* **fs_sequence_start** : start time of the audio segment in seconds
* **fs_sequence_duration** : duration of the audio segment in seconds
* **fs_octave_shift** : octave shift of the audio segment (e.g. -3 for 1/8 octave shift)
* **fs_volume_adjustment** : whether to automatically increase the volume of the audio segment; recommended since the vocalizations are faint

.. code-block:: json

    "frequency_shift_audio_segment": {
        "fs_audio_dir": "hpss_filtered",
        "fs_device_id": "m",
        "fs_channel_id": 1,
        "fs_wav_sampling_rate": 250,
        "fs_sequence_start": 900.0,
        "fs_sequence_duration": 2.0,
        "fs_octave_shift": -3,
        "fs_volume_adjustment": true
    }