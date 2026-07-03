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

Compute 3D behavioral features
------------------------------
Once 3D tracking data is available, you can compute behavioral features. These can be *individual features* specific to each mouse (*i.e.*, spatial location, speed, posture, *etc.*) or *social features* (assuming two or more mice) that describe the relationship between the mice (*i.e.*, distance, angle, *etc.*). The output of this analysis are two files: [1] CSV file containing each measured feature in each column, and [2] a PDF file containing graphs for the observed distribution of each feature. To run this analysis in the GUI, you need to list the root directories of interest, select *Compute 3D behavioral features*, click *Next* and then *Analyze*:

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

Compute neuronal tuning curves
------------------------------
Having recorded unit activity, social behavior, and ultrasonic vocalizations (USVs), you might be interested whether individual units encode specific behavioral features and / or vocal properties. To get at this, you can compute session-averaged *tuning curves* capturing the relationship between the firing rate of each unit and (a) each 3D behavioral feature, and (b) USV-anchored quantities — a pooled pre-USV PETH (peri-event time histogram; ``usv_peth``), within-USV firing rate as a function of each continuous acoustic property (``usv_property_tuning`` over duration, mean / peak frequency, bandwidth, amplitude, spectral entropy, mask number), within-USV firing rate as a function of categorical USV labels (``usv_category_tuning`` over VAE (variational autoencoder) / QLVM (in-house quasi-Monte Carlo latent variable model) ``category`` and ``supercategory``), and a per-category time-resolved peri-USV PETH (``usv_category_peth``). Behavioral and vocal payloads are produced together and serialized into a single per-cluster pickle. To trigger this in the GUI, list the root directories of interest, select *Compute neuronal tuning curves*, click *Next* and then *Analyze* (a progress bar will appear in the terminal while the analysis is running):

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

* **temporal_offsets** : list of temporal offsets between spikes and behavior (in seconds, negative values: spikes precede behavior) for which the tuning curves will be calculated (adding values to the list increases the time needed for analysis drastically)
* **n_shuffles** : number of spike train shuffles (increasing this number increases the time needed for analysis drastically)
* **shuffle_seed** : base seed for the spike-train shuffle null distribution; fixing it makes the significance verdicts reproducible across runs
* **total_bin_num** : total number of bins for a 1D behavioral / vocal-property feature
* **n_spatial_bins** : number of spatial bins (2D behavioral feature)
* **spatial_scale_cm** : maximum distance from center of arena to one edge (in cm)
* **shuffle_seconds_range** : ``[min, max]`` of the uniform circular shift (in s) used to build the null distribution
* **peth_window_seconds** : ``[start, stop]`` of the pre-USV PETH window (in s)
* **peth_bin_seconds** : PETH bin width (in s)
* **bout_quiet_seconds** : inter-bout silence required to define a new bout (in s)
* **vocal_require_clean_post_anchor** : if ``true``, the time after the USV onset must be free of contaminating USVs to keep the anchor
* **vocal_require_clean_prior_anchor** : if ``true``, the lookback window must also be free of contaminating USVs
* **n_usv_min_self** : minimum self-side USV count for the self plots to be computed
* **n_usv_min_partner** : minimum partner-side USV count for the partner plots to be computed
* **n_usv_min_category** : minimum per-category USV count to retain that category in the categorical tuning layouts
* **behavioral_min_occupancy_seconds** : minimum behavioral occupancy per bin (in s) to draw that bin in 1D feature plots; persisted into ``behavioral_metadata``
* **usv_property_min_occupancy_seconds** : minimum vocal-property occupancy per bin (in s) to keep the rate estimate finite
* **include_partner_vocalization_tuning_bool** : also compute partner-side vocal tuning when its threshold is met
* **shuffle_chunk_size** : how many shuffles to materialize at once (memory / speed knob)
* **smoothing_sd** : standard deviation of the Gaussian kernel (in bins) applied to ratemaps and shuffle distributions; ``0`` disables smoothing
* **circular_features** : list of behavioral feature suffixes that are wrap-around in nature (e.g. ``allo_yaw``, ``body_dir``); used by the triage helpers to detect divergence runs that span the bin-0 / bin-N boundary

.. code-block:: json

    "calculate_neuronal_tuning_curves": {
        "temporal_offsets": [0],
        "n_shuffles": 1000,
        "shuffle_seed": 0,
        "total_bin_num": 36,
        "n_spatial_bins": 196,
        "spatial_scale_cm": 32,
        "shuffle_seconds_range": [20, 60],
        "peth_window_seconds": [-2, 0],
        "peth_bin_seconds": 0.05,
        "bout_quiet_seconds": 2.0,
        "vocal_require_clean_post_anchor": true,
        "vocal_require_clean_prior_anchor": false,
        "n_usv_min_self": 30,
        "n_usv_min_partner": 30,
        "n_usv_min_category": 20,
        "behavioral_min_occupancy_seconds": 1.0,
        "usv_property_min_occupancy_seconds": 0.25,
        "include_partner_vocalization_tuning_bool": false,
        "shuffle_chunk_size": 50,
        "smoothing_sd": 1.0,
        "circular_features": ["allo_yaw", "body_dir"]
    }

Per-cluster ``triage_stats`` block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each per-cluster pkl also carries a ``triage_stats`` block — a flat collection of pre-computed scalar summaries that the downstream :ref:`unit-triage aggregator <unit-triage-aggregator>` consumes without re-touching spike or USV data. It covers every modality the rate payload does — the 3D behavioral features (kinematic and social), 2D spatial location, and the USV-anchored quantities:

* ``behavioral[offset_key][feature_key]`` — 1D tuning to every behavioral feature, kinematic (speed, posture, …) and social (distance, angle, …) alike. Per-direction (excitation / suppression) divergence-segment analysis: ``n_bins`` total above (or below) the shuffle band, ``max_run`` consecutive-bin run length, ``run_start_idx`` / ``run_end_idx`` (and the corresponding axis-value bounds), ``peak_idx`` / ``peak_z``, plus ``peak_abs_z``, ``peak_signed_z``, ``selectivity = (max−min)/(max+min)``, ``monotonicity`` (Spearman ρ between bin index and rate), and ``is_circular``.
* ``spatial[offset_key][feature_key]`` — 2D place-cell diagnostics for location: ``info_rate_bps`` (Skaggs information rate), ``sparsity``, ``coherence`` (Pearson correlation between each bin and the mean of its 8 neighbors), plus the unshuffled peak rate and its grid coordinates. The 2D spatial map is computed without shuffles, so peak Z is not defined; this block reports the rate / occupancy diagnostics instead.
* ``vmi[emitter]`` — Vocalization Modulation Index. For each emitter side: ``vmi`` in ``[-1, 1]``, paired Wilcoxon (the Wilcoxon signed-rank test) ``wilcoxon_statistic`` / ``wilcoxon_pvalue`` over the per-bout ``(FR_baseline, FR_USV)`` pairs, plus ``n_bouts``, ``fr_baseline_per_bout`` and ``fr_usv_per_bout`` arrays. ``VMI = (FR_USV − FR_baseline) / (FR_USV + FR_baseline)``, where ``FR_baseline`` is the mean firing rate in the ``bout_quiet_seconds``-wide window before each bout and ``FR_USV`` is the mean over USVs in each bout of (spikes during USV) / (USV duration). Bouts whose baseline window starts before ``t = 0`` are NaN-baselined.
* ``usv_peth[emitter]``, ``usv_property_tuning[emitter][prop]``, ``usv_category_peth[emitter][cat_feat]`` — the same per-direction divergence-segment analysis as the 1D behavioral axes above (``n_bins``, ``max_run``, ``run_start_idx`` / ``run_end_idx``, ``peak_idx`` / ``peak_z``, ``peak_abs_z``, ``peak_signed_z``, ``selectivity``, ``monotonicity``). The PETH variants additionally carry ``ramp_index`` (a two-point pre-USV shape descriptor).
* ``usv_category_tuning[emitter][cat_feat]`` — categorical (no run analysis): ``peak_abs_z``, ``best_cat``, ``n_sig_categories`` (count of categories outside the [p0.5, p99.5] shuffle band), ``selectivity``.

Frequency shift audio segment
-----------------------------
For presentation purposes, one might want to play audio data of mouse USVs. Since these are beyond human audible range, the only way is to frequency-shift them several octaves down. To achieve this in the GUI, you need to list the root directories of interest, select *Frequency shift audio segment*, choose the start time and duration of the segment, click *Next* and then *Analyze*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_3.png
   :align: center
   :alt: Analysis Step 3

.. raw:: html

   <br>

The analysis results in the creation of a *frequency_shifted_audio_segments* subdirectory (if it is not already there) and a *wave* file in it containing the frequency-shifted segment:

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
* **fs_device_id** : USGH (Avisoft UltraSoundGate hardware) device ID (e.g. "m" for main, "s" for secondary)
* **fs_channel_id** : microphone channel ID (1-12)
* **fs_wav_sampling_rate** : sampling rate of the audio devices in kHz
* **fs_sequence_start** : start time of the audio segment in seconds
* **fs_sequence_duration** : duration of the audio segment in seconds
* **fs_octave_shift** : octave shift of the audio segment (e.g. -3 shifts down three octaves, to 1/8 the original frequency)
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

* **num_usv_files** : number of artificial playback files to be created
* **total_usv_number** : total number of USVs to be included in one playback file
* **ipi_duration** : inter-pulse interval duration in seconds
* **wav_sampling_rate** : sampling rate of the playback .WAV file in kHz
* **playback_snippets_dir** : subdirectory where the USV snippets are stored
* **playback_seed** : optional RNG seed for reproducible snippet selection and assembly; ``null`` draws a fresh random stimulus each run, an integer reproduces the same stimulus

.. code-block:: json

    "create_usv_playback_wav": {
        "num_usv_files": 1,
        "total_usv_number": 10000,
        "ipi_duration": 0.015,
        "wav_sampling_rate": 250,
        "playback_snippets_dir": "usv_playback_snippets_loudness_corrected",
        "playback_seed": null
    }

Create naturalistic playback .WAV
---------------------------------
This function creates a .WAV file containing naturalistic sequences of USV snippets. The snippets are randomly selected from the female or male USV snippet repository in
the specified directory and assembled into sequences with empirically derived inter-event intervals. The resulting .WAV file can be used for playback experiments.
To achieve this in the GUI, select *Create naturalistic playback .WAV file* (no need to list root directories!), select total number of files to be created, number of vocalizations in each one, click *Next* and then *Analyze*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/analyze_step_5.png
   :align: center
   :alt: Analysis Step 5

.. raw:: html

   <br>

Inter-USV intervals (IUIs) and inter-sequence intervals (ISIs) are sampled from a sex-specific log-space Student-t mixture (mixture of Student's t-distributions) model fit to the empirical end-to-start (``e2s``) inter-USV interval distribution. The fitted model is read live from the HDF5 interval archive produced by the inter-USV interval analysis (see :doc:`Notebooks`) (the ``naturalistic_iui_archive_h5`` setting); the number of components ``K`` is the per-sex value the archive's bootstrap-LRT step-up procedure selected (``K_selected_<sex>``), so no component counts or parameters are hardcoded.
Sex is inferred automatically from the ``naturalistic_playback_snippets_dir_prefix`` setting (e.g. ``"female"`` or ``"male"``). The reconstructed mixture (components sorted ascending by log-mean) is split into two roles:

* **ISI** is drawn from the *slowest* (longest-interval) component only — the long quiet pause between sequences
* **IUI** is drawn from the pool of *all remaining (faster) components* (weights renormalised) — the short within-sequence gaps; pooling rather than using only the fastest breathing-expiration component keeps the full within-sequence interval mass (e.g. the female ~0.9 s component, which carries most of her mass)
* **Sequence length** is drawn from N(13, 5) clipped to [3, 23] USVs

Because the low-degrees-of-freedom Student-t components have heavy tails (a raw draw, once exponentiated, can exceed the entire playback file), every draw is reject-resampled to a per-sex ``[100 - clip_pct, clip_pct]`` percentile band (``naturalistic_interval_clip_pct``) before being exponentiated, so a single draw cannot emit an absurdly long silence.

The analysis results in the creation of three files: [1] WAV file containing playback vocalizations, [2] a *spacing* text file informing you of the duration of each vocalization in order, and [3] a *usvids* text file containing the identity of each vocalization snippet if you need to go back and look at what it was:

.. parsed-literal::

    /mnt/falkner/Bartul/usv_playback_experiments/naturalistic_usv_playback_files
    ├── **female_usv_playback_1080s_20250506_190808.wav**
    ├── **female_usv_playback_1080s_20250506_190808_spacing.txt**
    ├── **female_usv_playback_1080s_20250506_190808_usvids.txt**
    ...

The */usv-playpen/_parameter_settings/analyses_settings.json* file contains a section only partially modifiable in the GUI, but it can be modified manually:

* **num_naturalistic_usv_files** : number of naturalistic playback files to be created
* **naturalistic_wav_sampling_rate** : sampling rate of the playback .WAV file in kHz
* **naturalistic_playback_snippets_dir_prefix** : prefix of the subdirectory where the USV snippets are stored (the rest of the subdirectory name should be ``"_usv_playback_snippets"``); also determines which sex-specific Student-t model is used (``"female"`` or ``"male"``)
* **total_acceptable_naturalistic_playback_time** : total acceptable duration of the playback file (in s)
* **naturalistic_iui_archive_h5** : path to the HDF5 interval archive (``usv_interval_analysis_*.h5``) produced by the inter-USV interval analysis (see :doc:`Notebooks`); the per-sex Student-t model is reconstructed from it at generation time
* **naturalistic_interval_mode** : interval definition used for the gaps (``"e2s"`` = end-to-start, the physical silent gap between successive USVs)
* **naturalistic_interval_clip_pct** : per-sex upper percentile for heavy-tail clipping, a ``{"male": ..., "female": ...}`` dict; each draw is reject-resampled into the ``[100 - clip_pct, clip_pct]`` percentile band of its sub-mixture
* **playback_seed** : optional RNG seed for reproducible snippet selection and assembly; ``null`` draws a fresh random stimulus each run, an integer reproduces the same stimulus

.. code-block:: json

    "create_naturalistic_usv_playback_wav": {
        "num_naturalistic_usv_files": 1,
        "naturalistic_wav_sampling_rate": 250,
        "naturalistic_playback_snippets_dir_prefix": "female",
        "total_acceptable_naturalistic_playback_time": 1080,
        "naturalistic_iui_archive_h5": "/mnt/falkner/Bartul/modeling/usv_interval_results/usv_interval_analysis_20260501_193959.h5",
        "naturalistic_interval_mode": "e2s",
        "naturalistic_interval_clip_pct": { "male": 99.0, "female": 97.0 },
        "playback_seed": null
    }
