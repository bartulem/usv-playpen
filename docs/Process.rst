.. _Process:

Process
=======
This page explains how to use the data processing functionalities in the *usv-playpen* GUI.

In order to run any of the functions detailed below, select an experimenter name from the dropdown menu and click the *Process* button on the GUI main display:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_0a.png
   :align: center
   :alt: Processing Step 0

.. raw:: html

   <br>

Clicking the *Process* button will open a new window with all the processing functionalities (see below):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_0b.png
   :align: center
   :alt: Processing Step 0b

.. raw:: html

   <br>

All the main functions are outlined in orange, and black fields are function-specific options tunable by the user in the GUI. It is important to note that these are not necessarily *all* the options the user can set, and the full list of options can be found under each function in the */usv-playpen/_parameter_settings/processing_settings.json* file. Each time the user clicks the *Next* button in the window above, *processing_settings.json* is modified to the newest input configuration.

.. note::

   The shipped ``*_settings.json`` files store a literal default experimenter
   (by default ``Bartul``). Experimenter-scoped paths are re-keyed to the active
   experimenter automatically: in the GUI from the front-page experimenter
   selection, and for headless / CLI / cluster runs from the host
   ``behavioral_experiments_settings.toml`` ``experimenter`` key. You therefore
   set your experimenter once instead of editing every path; the example paths
   below show that shipped default.

It is relevant to note here, that just like in the *Record* section, you have the capability to *Notify e-mail(s) of PC usage*. This is useful if you are running a long processing job and want to be notified when it is finished. The e-mails about start and end of jobs will be sent to the addresses listed in the *Notify e-mail(s) of PC usage* field (**no space after comma for multiple e-mails**), and it requires you to choose what particular PC you are using for this job. Since the e-mails are sent from a Google account, the first e-mail you receive may end up in the Spam folder, so make sure to check that:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_email.png
   :align: center
   :alt: Processing Step e-mail

.. raw:: html

   <br>

The *Root directories* field enables you to list the directories containing the data you want to process. Each root directory should be in its **own row**; for example, three sessions should be listed as follows:

.. parsed-literal::

    /mnt/falkner/Bartul/Data/20250430_145017
    /mnt/falkner/Bartul/Data/20250430_165730
    /mnt/falkner/Bartul/Data/20250430_182145

Certain processing functions take all root sessions *together* when operating on data, and others process each session *separately*. Additionally, in both of these categories, there is a specific order in executing individual functions.

For **combined processing**, the order of processing steps is as follows:

    #. Concatenate e-phys files
    #. Split clusters to sessions
    #. Prepare SLEAP cluster job
    #. Build QLVM training set (CLI only)
    #. Train QLVM (CLI only)
    #. Export YOLO dataset (CLI only)
    #. Train (spectrogram) masks (CLI only)

The last four run only when (re)training the spectrogram-pipeline models: they aggregate a cohort of sessions and produce the QLVM (the in-house quasi-Monte Carlo latent variable model) decoder and You Only Look Once (YOLO) object detector weights that the per-session *Infer QLVM latents* / *Generate (spectrogram) masks* steps reload (see the *Render spectrograms and latents* section below).

On the other hand, for **processing sessions separately**, the order of processing steps is as follows:

    #. Run video concatenation
    #. Run video re-encoding
    #. Convert to single-ch files
    #. Crop AUDIO (to VIDEO)
    #. Run A/V sync check
    #. Run E/V sync check
    #. Run HPSS
    #. Filter audio files
    #. Concatenate to MEMMAP
    #. Run SLP-H5 conversion
    #. Run AP calibration
    #. Run AP triangulation
    #. Re-coordinate
    #. Run DAS inference
    #. Curate DAS outputs
    #. Detect noise
    #. Detect squeaks
    #. Prepare USV assignment
    #. Run USV assignment
    #. Generate spectrograms
    #. Generate (spectrogram) masks
    #. Compute USV features
    #. Infer QLVM latents

If you recorded a session with audio, e-phys and video data (imaginary example: 20250430_145017) and a calibration session (20250430_142022), the directory and file structure should look as follows:

.. parsed-literal::

    /mnt/falkner/Bartul/Data/:
    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── original (empty)
    │   │   ├── original_mc
    │   │       ├── m_250430145009.wav
    │   │           ...
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ├── imec0
    │   │   │   ├── 20250430_145017.imec0.ap.bin
    │   │   │   ├── 20250430_145017.imec0.ap.meta
    │   │   ├── imec1
    │   │       ├── 20250430_145017.imec1.ap.bin
    │   │       ├── 20250430_145017.imec1.ap.meta
    │   ├── sync
    │   │   ├── CoolTerm Capture (coolterm_config.stc) 2024-04-30-14-50-14-236.txt
    │   │   ├── 20250430_rec4_g0_t0.nidq.bin
    │   │   ├── 20250430_rec4_g0_t0.nidq.meta
    │   │
    │   └── video
    │       ├── 20250430_145027.21241563
    │           ├── 000000.mp4
    │           ├── 000000.npz
    │           ├── 000001.mp4
    │           ├── 000001.npz
    │           ├── metadata.yaml
    │       ...
    │
    ├── 20250430_142022
    │    ├── sync
    │    │   ...
    │    ├── video
    │        ├── 20250430_142022.21241563
    │        │   ...
    │        ├── 20250430142022
    │        │   ├── video
    │        │   │   ├── 21241563
    │        │   │   ...
    │        │   │   ├── 20250430142022_calibration.metadata.h5
    │        │   │   ├── 20250430142022_calibration.toml
    │        │   │   ├── 20250430142022_reprojection_histogram.png
    │        │   │   ...
    │        ├── calibration_20250430_141910.21241563
    │        │   ...

E-phys processing
-----------------
The processing of e-phys data passes several stages:

    #. Check e-phys data is synchronized with video
    #. Concatenate e-phys files of individual sessions for joint spike sorting
    #. Conduct spike sorting with `Kilosort4 <https://github.com/MouseLand/Kilosort/tree/main>`_ (not implemented in *usv-playpen*; reference runner: ``other/kilosort/run_kilosort.py``)
    #. Manually curate sorting outputs in `Phy <https://github.com/cortex-lab/phy>`_ (not implemented in *usv-playpen*)
    #. Split cluster spikes back to individual sessions
    #. Conduct light-sheet brain volume assembly, trace probe tracks in Allen atlas coordinates with `brainreg <https://github.com/brainglobe/brainreg-napari>`_ and `brainglobe-segmentation <https://github.com/brainglobe/brainglobe-segmentation>`_ to determine what brain regions individual channels were in using `iblapps <https://github.com/int-brain-lab/iblapps>`_, but IBL ephys-alignment functionality is provided (see :ref:`Neuropixels`)
    #. Compute unit quality metrics and categorize units with `SpikeInterface <https://github.com/SpikeInterface/spikeinterface>`_ (see :ref:`Neuropixels` for details on how this is implemented in *usv-playpen*)

Run E/V sync check
~~~~~~~~~~~~~~~~~~
To run the e-phys/video synchronization check, you need to list the root directories of interest, select *Run E/V sync check*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_1.png
   :align: center
   :alt: Processing Step 1

.. raw:: html

   <br>

Neural recording data is aligned to the start of video recording, which is identifiable by searching for a ~2.3 s break in Loopbio Triggerbox pulses, which are constantly being transmitted to the Neuropixels digital input channel. The code recursively finds all the *ap.bin* files in the root directory and saves the digital input channel data (385th or last channel) to a separate Numpy file (which ends with *_sync_ch_data.npy*), if it hasn't been saved already. After finding the tracking start and end (based on the largest Triggerbox break duration and total number of recording frames) in this Numpy file, the total video duration will then be compared to the total video-aligned neural recording, and you will get a report back whether that discrepancy is below 12 ms (in other words, less than 2 video frames, which is an acceptable level of distortion). Information at what Neuropixels sample the first and last video recording frame were detected will be saved to, for instance, */mnt/falkner/Bartul/EPHYS/20250430_imec0/changepoints_info_20250430_imec0.json*, as exemplified below:

.. parsed-literal::

    /mnt/falkner/Bartul/Data/:
    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ├── imec0
    │   │   │   ├── 20250430_145017.imec0.ap.bin
    │   │   │   ├── 20250430_145017.imec0.ap.meta
    │   │   │   ├── **20250430_145017_imec0_sync_ch_data.npy**
    │   │   ├── imec1
    │   │       ├── 20250430_145017.imec1.ap.bin
    │   │       ├── 20250430_145017.imec1.ap.meta
    │   │       ├── **20250430_145017_imec1_sync_ch_data.npy**
    │   ├── sync
    │   │   ...
    │   │
    │   └── video
    │       ...
    /mnt/falkner/Bartul/EPHYS:
    ├── 20250430_imec0
    │   ├── **changepoints_info_20250430_imec0.json**
    ├── 20250430_imec1
    │   ├── **changepoints_info_20250430_imec1.json**


In the *changepoints* JSON file, the E/V sync check process will save the *tracking_start_end* and *largest_camera_break_duration* values, and the latter, when divided with the Neuropixels sampling rate (should be ~30 kHz), should not be smaller than ~2.3 s.

.. code-block:: json

    "20250430_145017.imec0": {
        "session_start_end": [
            0,
            37825731
        ],
        "tracking_start_end": [
            850469,
            36867993
        ],
        "largest_camera_break_duration": 69341,
        "file_duration_samples": 37825731,
        "root_directory": "/mnt/falkner/Bartul/Data/20250430_145017",
        "total_num_channels": 385,
        "headstage_sn": "23280196",
        "imec_probe_sn": "22420015064"
    }

The */usv-playpen/_parameter_settings/processing_settings.json* file also contains a section not modifiable in the GUI itself, but it can be modified manually:

* **npx_file_type** : Neuropixels 1.0 had "lf" and "ap" files, this field allows you to switch between them
* **npx_ms_divergence_tolerance** : the maximum allowed difference between the video and e-phys recording duration in milliseconds; the default value is 12 ms but it can be tuned to whatever the user thinks is appropriate.
* **apply_phase_shift** : if ``true`` (the default), each session's AP binary is de-skewed for the Neuropixels ADC sample-time offset (see the note below) as part of the sync check; ``"ap"`` band only.

.. code-block:: json

    "validate_ephys_video_sync": {
            "npx_file_type": "ap",
            "npx_ms_divergence_tolerance": 12.0,
            "apply_phase_shift": true
    }

.. note::

   **Neuropixels ADC phase-shift correction (**\ ``apply_phase_shift``\ **).** A
   Neuropixels probe multiplexes its analog channels onto a small number of ADCs,
   so within a single sample period the channels are digitised at slightly
   staggered times (a fixed, per-channel fraction-of-a-sample delay). When
   ``apply_phase_shift`` is ``true`` (the default), the e-phys/video sync check
   additionally de-skews this offset on each session's AP binary with a Fourier
   fractional-sample shift (SpikeInterface's ``phase_shift``), so every channel is
   aligned to a common time base before any channel-combining step (referencing /
   whitening / drift correction) in the sorter reads the data. The trailing
   SpikeGLX sync channel is left bit-for-bit unchanged.

   The correction is applied **in place**: each raw ``*.ap.bin`` is replaced at its
   exact path (temporary file + atomic rename), so every downstream step that
   expects a single ``*.ap.bin`` per ``imecN`` directory is unaffected. A
   ``*_phase_shift_applied.json`` marker is written next to each corrected binary,
   and a session that already carries this marker is skipped, so re-running the
   check never double-shifts. **Because the raw binary is overwritten, keep an
   untouched copy of the raw SpikeGLX data elsewhere** (this is standard practice).

   **Run the e-phys/video sync check BEFORE concatenation.** The correction is done
   per session, so concatenation then stitches already-corrected binaries. If you
   concatenate first, the concatenated file is built from the raw, un-corrected
   sessions.


Concatenate e-phys files
~~~~~~~~~~~~~~~~~~~~~~~~
To run the concatenation of e-phys files (ap.bin), you need to list *all* the root directories of interest *in order you want them to be concatenated*, select *Concatenate e-phys files*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_2.png
   :align: center
   :alt: Processing Step 2

.. raw:: html

   <br>

The code will find all the *ap.bin* files for each probe and conduct the concatenation to save the files in the *EPHYS* directory with the *concatenated_* prefix:

.. parsed-literal::

    /mnt/falkner/Bartul/Data/:
    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ├── imec0
    │   │   │   ├── 20250430_145017.imec0.ap.bin
    │   │   │   ├── 20250430_145017.imec0.ap.meta
    │   │   │   ├── 20250430_145017_imec0_sync_ch_data.npy
    │   │   ├── imec1
    │   │       ├── 20250430_145017.imec1.ap.bin
    │   │       ├── 20250430_145017.imec1.ap.meta
    │   │       ├── 20250430_145017_imec1_sync_ch_data.npy
    │   ├── sync
    │   │   ...
    │   │
    │   └── video
    │       ...
    /mnt/falkner/Bartul/EPHYS:
    ├── 20250430_imec0
    │   ├── changepoints_info_20250430_imec0.json
    │   ├── **concatenated_20250430_imec0.ap.bin**
    ├── 20250430_imec1
    │   ├── changepoints_info_20250430_imec1.json
    │   ├── **concatenated_20250430_imec1.ap.bin**

In the *changepoints* JSON file, the concatenation process will modify all lines other than the ones described above for E/V sync.

.. code-block:: json

    "20250430_145017.imec0": {
        "session_start_end": [
            0,
            37825731
        ],
        "tracking_start_end": [
            850469,
            36867993
        ],
        "largest_camera_break_duration": 69341,
        "file_duration_samples": 37825731,
        "root_directory": "/mnt/falkner/Bartul/Data/20250430_145017",
        "total_num_channels": 385,
        "headstage_sn": "23280196",
        "imec_probe_sn": "22420015064"
    }


Split clusters to sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~
After spike sorting and post-sorting curation are complete, you can split the spikes of individual clusters back to the original sessions. To do this, even if you recorded multiple sessions in one day, **it is sufficient to put only one root directory for that day**, e.g., the first one. The script will find EPHYS root directory, and split spikes from all probes into sessions based on the inputs in the changepoints JSON file. Select *Split clusters to sessions*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_3.png
   :align: center
   :alt: Processing Step 3

.. raw:: html

   <br>

The code will create a *cluster_data* subdirectory in each session's *ephys/imec* directory and populate it with Numpy files containing spike times in the shape of (2, number_of_spikes), where the first row contains spike times in seconds relative to start of tracking and the second row spike times according to what tracking frame they occurred in. Each cluster is named in the following format: *probeID_clusterNumber_channelID_clusterType.npy*.

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ├── imec0
    │   │   │   ├── 20250430_145017.imec0.ap.bin
    │   │   │   ├── 20250430_145017.imec0.ap.meta
    │   │   │   ├── 20250430_145017_imec0_sync_ch_data.npy
    │   │   │   ├── **cluster_data**
    │   │   │   │   ├── **imec0_cl0000_ch361_good.npy**
    │   │   │   │       ...
    │   │   ├── imec1
    │   │       ├── 20250430_145017.imec1.ap.bin
    │   │       ├── 20250430_145017.imec1.ap.meta
    │   │       ├── 20250430_145017_imec1_sync_ch_data.npy
    │   │       ├── **cluster_data**
    │   │       │   ├── **imec1_cl0000_ch361_good.npy**
    │   │       │       ...
    │   ├── sync
    │   │   ...
    │   │
    │   └── video
    │       ...

The */usv-playpen/_parameter_settings/processing_settings.json* file also contains a section partially modifiable in the GUI, but it can entirely be modified manually:

* **min_spike_num** : eliminate clusters with fewer spikes than this (set 0 if you want to keep all)
* **kilosort_version** : Kilosort version in use
* **remove_duplicate_spikes** : if ``true`` (the default), drop near-coincident duplicate spikes per unit before splitting into sessions (see the note below)
* **duplicate_censored_period_ms** : two spikes of one unit closer than this (in ms) count as a duplicate (default ``0.3``)

.. code-block:: json

    "get_spike_times": {
        "min_spike_num": 100,
        "kilosort_version": "4",
        "remove_duplicate_spikes": true,
        "duplicate_censored_period_ms": 0.3
      },

.. note::

   **Duplicate-spike removal (**\ ``remove_duplicate_spikes``\ **).** When ``true``
   (the default), each unit's spike train is passed through SpikeInterface's
   ``find_duplicated_spikes`` (``keep_first_iterative``) before the split, dropping
   the second of any spike pair closer than ``duplicate_censored_period_ms``.

   This is **not** redundant with Kilosort's own duplicate removal. Kilosort
   de-duplicates **per cluster, at sort time** (its ``duplicate_spike_ms``); but a
   **Phy merge** combines two templates' detections of the *same* physical spike
   into a single unit, and because those spikes lived in *different* clusters when
   Kilosort ran, it never saw them as duplicates. They therefore survive into the
   curated sort. Measured on a real four-session sort, these merge-induced
   duplicates are ~0.03 % of all spikes but ~16 % of the sub-millisecond
   refractory-period violations.

   The removal cleans the per-session analysis spike files only. The spike-quality
   metrics (``isi_violations`` / ``rp_violations`` in :ref:`Neuropixels`) are
   deliberately computed on the **raw** curated trains: a duplicate-laden unit is a
   real quality signal that those metrics should flag, so the quality stage is
   intentionally left un-de-duplicated.

Video processing
----------------
The processing of video data passes multiple stages:

    #. Video concatenation and re-encoding (runs locally <20 min)
    #. SLEAP inference (runs on cluster)
    #. SLEAP proofreading (bottleneck step, requires extensive human curation)
    #. SLP-H5 conversion (runs locally <1 min)
    #. SLEAP-Anipose triangulation (runs locally <40 min)
    #. Translate, rotate and scale SLEAP coordinates to metric units (runs locally <1 min)

Video concatenation and re-encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before running this section, it is always a good idea to check that video files were copied to the file server correctly. These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run video concatenation and re-encoding, you need to list the root directories of interest, select *Run video concatenation* and *Run video re-encoding*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_4.png
   :align: center
   :alt: Processing Step 4

.. raw:: html

   <br>

The re-encoding step will also result in the creation of the *camera_frame_count_dict.json* file, which contains numbers of frames for each camera in the session, as well as the total number of frames and video time for the camera with the least number of frames. The file will be saved in the *video* subdirectory of each session, and it will look like this:

.. code-block:: json

    {
    "21241563": [
        180002,
        150.057
    ],
    "21369048": [
        180000,
        150.057
    ],
    "21372315": [
        180001,
        150.057
    ],
    "21372316": [
        180001,
        150.056
    ],
    "22085397": [
        180002,
        150.057
    ],
    "total_frame_number_least": 180000,
    "total_video_time_least": 1199.5477764606476,
    "median_empirical_camera_sr": 150.057
    }

These steps change videos and video directory structure from the native Loopbio format to one that is compatible with SLEAP-Anipose. Both rely on the usage of `ffmpeg <https://ffmpeg.org/download.html>`_ . After the steps are complete, the directory structure and file names should look as follows (displaying only one camera directory for brevity):

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
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
    │       ├── **20250430145035_camera_frame_count_dict.json**
    │       ├── **20250430145035**
    │       │    ├── **21241563**
    │       │    │   ├── **calibration_images**
    │       │    │   ├── **21241563-20250430145035.mp4**
    │       ...

The */usv-playpen/_parameter_settings/processing_settings.json* file also contains a section partially modifiable in the GUI, but it can entirely be modified manually:

* **concatenate_camera_serial_num** / **encode_camera_serial_num** : serial numbers of cameras used in the recording (for concatenation / re-encoding, respectively)
* **concatenate_video_extension** / **encode_video_extension** : video type (usually "mp4")
* **concatenated_video_name** : name of the concatenated video file
* **conversion_target_file** : name of the concatenated video file as target for re-encoding
* **constant_rate_factor** : FFMPEG constant rate factor for re-encoding
* **encoding_preset** : FFMPEG encoding preset for re-encoding
* **delete_old_file** : whether to delete the concatenated file after re-encoding

.. code-block:: json

    "concatenate_video_files": {
        "concatenate_camera_serial_num": [
          "21372315",
          "21372316",
          "21369048",
          "22085397",
          "21241563"
        ],
        "concatenate_video_extension": "mp4",
        "concatenated_video_name": "concatenated_temp"
      },
      "rectify_video_fps": {
        "encode_camera_serial_num": [
          "21372315",
          "21372316",
          "21369048",
          "22085397",
          "21241563"
        ],
        "conversion_target_file": "concatenated_temp",
        "encode_video_extension": "mp4",
        "constant_rate_factor": 16,
        "encoding_preset": "veryfast",
        "delete_old_file": true
      }

Prepare SLEAP cluster job
~~~~~~~~~~~~~~~~~~~~~~~~~
The *usv-playpen* GUI assumes usage of the SLEAP pose-tracking framework (`SLEAP <https://sleap.ai/>`_) for animal pose tracking. To do this, one first needs to train one or multiple models on the data of interest (*i.e.*, social interactions). Explaining how to do this is beyond the scope of this text, so we will assume you already have a *top-down centroid and centered instance model* ready for running inference.

Since the average office PC does not necessarily have GPU-capabilities, it is advised to run SLEAP inference on a high-performance computing cluster, as these usually have GPU-capabilities and allow for the parallelization of the inference process. The *usv-playpen* GUI helps you prepare the SLEAP cluster job, but you will need to run the job on the cluster yourself.

The preparation consists of creating a *job_list.txt* file which contains the paths to the video files and the model(s) to be used for inference. The job list can then be used by a shell script, such as the one in */usv-playpen/other/cluster/SLEAP/sleap_inference_global.sh* to execute inference on all video files of interest.

To run the SLEAP cluster job preparation, you need to list the root directories of interest (which will search for all videos recorded in those sessions), select the SLEAP conda environment name used **on the cluster**, select directories of centroid and centered instance models, select the output inference directory, select *Make SLEAP job list*, click *Next* and finally *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_5.png
   :align: center
   :alt: Processing Step 5

.. raw:: html

   <br>

This shouldn’t take longer than several seconds - it will create/update the *job_list.txt* file in, for example, */mnt/falkner/Bartul/SLEAP/inference* directory:

.. parsed-literal::

    /mnt/falkner/Bartul/SLEAP/inference:
    ├── **job_list.txt**
    │   ...

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section partially modifiable in the GUI, but it can entirely be modified manually:

* **camera_names** : camera serial numbers used in the recording
* **inference_root_dir** : directory where the inference job list will be saved
* **centroid_model_path** : path to the SLEAP centroid model
* **centered_instance_model_path** : path to the SLEAP centered instance model

.. code-block:: json

   "prepare_cluster_job": {
    "camera_names": [
      "21372315",
      "21372316",
      "21369048",
      "22085397",
      "21241563"
    ],
    "inference_root_dir": "/mnt/falkner/Bartul/SLEAP/inference",
    "centroid_model_path": "",
    "centered_instance_model_path": ""
  }

SLEAP inference and proofreading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The SLEAP inference and proofreading steps are not implemented in the *usv-playpen* GUI. However, you can run the inference job on the cluster using the shell script mentioned above. The proofreading step is done in the SLEAP GUI, where it is crucial to correct identity switches and to **keep the same animal identities across different video views**. By current convention, that means the male mouse is always assigned identity 0, and the female mouse is always assigned identity 1.

Run SLP-H5 conversion
~~~~~~~~~~~~~~~~~~~~~
After proofreading, you convert SLP to H5 files, which is the format SLEAP-Anipose operates on (*usv-playpen* runs this in parallel for all views). To do this, you need to list the root directories of interest, select *Run SLP-H5 conversion*, click *Next* and then *Process* (NB: using the SLEAP uvx functionality, it is no longer necessary to install SLEAP to run this step):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_6.png
   :align: center
   :alt: Processing Step 6

.. raw:: html

   <br>

This step shouldn’t take longer than two minutes to run; the directory structure and file names should look as follows (displaying only one camera directory for brevity):

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
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
    │       │    ├── 21241563
    │       │    │   ├── calibration_images
    │       │    │   ├── **21241563-20250430145035.h5**
    │       │    │   ├── 21241563-20250430145035.mp4
    │       │    │   ├── 21241563-20250430145035.slp
    │       ...


Run AP triangulation & Re-coordinate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once SLP files are converted to H5, you are ready to run triangulation. Triangulation is the process of estimating the 3D coordinates of the tracked items based on the 2D coordinates from multiple camera views.

SLEAP-Anipose triangulation can be run to obtain **3D arena points**, or **3D animal points**.

3D arena points
^^^^^^^^^^^^^^^

It was previously explained how to record a calibration session, and in that session you recorded a 1-minute video of the arena with visible microphones and IR-reflective markers in its corners. All the video views of this recording can be loaded into the SLEAP GUI, and **only on the first frame of each view**, you label the 24 microphones and 4 corners with a 28-node skeleton that can be found in */usv-playpen/_config/playpen_skeleton.json*. You label the microphones with the corresponding channel number, and corners with N, E, S and W, according to the following schematic:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/arena_mics_avisoft_devices.png
   :align: center
   :alt: Arena and microphones

.. raw:: html

   <br>

After labeling the first frame on each view, you can export the data as H5 files going to *File > Export Analysis HDF5*. You are now ready to run arena triangulation.

To do this, you need to list the root directories of interest, select the same root directory under *Tracking calibration / arena root directory*, select *Run AP triangulation* and *Re-coordinate*, select *Triangulate arena nodes*, select "arena" for *Save transformation type* and choose "No" for *Delete original .h5*. Finally, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_7.png
   :align: center
   :alt: Processing Step 7

.. raw:: html

   <br>

This shouldn’t take longer than one minute; the directory structure and file names should look as follows (note that you keep both the original and translated_rotated_metric H5 files!):

.. parsed-literal::

    ├── 20250430_145017
    │   ...
    │
    ├── 20250430_142022
    │    ├── sync
    │    │   ...
    │    ├── video
    │        ├── 20250430_142022.21241563
    │        │   ...
    │        ├── 20250430142022
    │        │   ├── **20250430142022_points3d.h5**
    │        │   ├── **20250430142022_points3d_translated_rotated_metric.h5**
    │        │   ...
    │        ├── calibration_20250430_141910.21241563
    │        │   ...

3D animal points
^^^^^^^^^^^^^^^^

To triangulate animal points, you need to list the root directories of interest, list their respective experimental codes, select the directory with the triangulated arena file, select *Run AP triangulation* and *Re-coordinate*, select "animal" for *Save transformation type* and choose "Yes" for *Delete original .h5*. Finally, click *Next* and then *Process* (a progress bar in the terminal will update you on the status of the process):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_8.png
   :align: center
   :alt: Processing Step 8

.. raw:: html

   <br>

The process results in the creation of an H5 file which ends in *_points3d_translated_rotated_metric.h5*, and can be found as shown below:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
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
    │       │    ├── **20250430145035_points3d_translated_rotated_metric.h5**
    │       ...

The */usv-playpen/_parameter_settings/processing_settings.json* file also contains a section partially modifiable in the GUI, but it can entirely be modified manually:

* **calibration_file_loc** : directory containing the _calibration.toml file relevant for the session
* **triangulate_arena_points_bool** : whether to triangulate arena or animal tracked nodes
* **frame_restriction** : range of frames to be triangulated; empty finds the least number of frames across all cameras and triangulates those
* **excluded_views** : list of camera serial numbers to be excluded from triangulation
* **display_progress_bool** : whether to display the progress bar in the terminal during execution
* **ransac_bool** : whether to use RANSAC for triangulation
* **rigid_body_constraints** : list of rigid body constraints to be used for triangulation
* **weak_body_constraints** : list of weak body constraints to be used for triangulation
* **smooth_scale** : scale of the smoothing kernel
* **weight_weak** : weight of the weak body constraints
* **weight_rigid** : weight of the rigid body constraints
* **reprojection_error_threshold** : threshold for reprojection error in pixels
* **regularization_function** : regularization function to be used for triangulation
* **n_deriv_smooth** : number of derivatives to be used for smoothing
* **original_arena_file_loc** : directory containing the original arena 3D file
* **save_transformed_data** : whether to save the transformed data as "animal" or "arena"
* **delete_original_h5** : whether to delete the original H5 file
* **static_reference_len** : length of the static reference in meters, defaults to distance between two outer rail edges of two arena corners
* **experimental_codes** : list of experimental codes associated with each session

.. code-block:: json

      "conduct_anipose_triangulation": {
        "calibration_file_loc": "",
        "triangulate_arena_points_bool": false,
        "frame_restriction": null,
        "excluded_views": [],
        "display_progress_bool": true,
        "ransac_bool": false,
        "rigid_body_constraints": [],
        "weak_body_constraints": [],
        "smooth_scale": 4,
        "weight_weak": 4,
        "weight_rigid": 1,
        "reprojection_error_threshold": 5,
        "regularization_function": "l2",
        "n_deriv_smooth": 1
      },
      "translate_rotate_metric": {
        "original_arena_file_loc": "",
        "save_transformed_data": "animal",
        "delete_original_h5": true,
        "static_reference_len": 0.615,
        "experimental_codes": []
      }

The experimental codes are used to identify the session and the type of experiment conducted. The decoding sheet can be found below:

.. parsed-literal::

   A - ablation
   E - ephys
   H - chemogenetics
   O - optogenetics
   P - playback
   B - behavior
   V - devocalization
   U - urine/bedding

   Q - alone
   C - courtship
   X - females
   Y - males

   L - light
   D - dark

   1,2,3 ... - number of animals

   F - female
   M - male

   S - single
   G - group

   p - proestrus
   e - estrus
   m - metestrus
   d - diestrus

Audio processing
----------------
The processing of audio data passes multiple stages:

    #. Split audio to single files and crop to video duration (runs locally <15 min)
    #. De-noise audio data with harmonic-percussive source separation (runs locally or on cluster)
    #. Band-pass filter audio files (runs locally <15 min)
    #. Concatenate all audio files to single MEMMAP file (runs locally <15 min)
    #. Run DAS inference (runs on cluster)
    #. Curate DAS outputs (runs locally <2 min)
    #. Detect noise (runs locally, GPU recommended)
    #. Detect squeaks (runs locally, GPU recommended)
    #. Prepare USV assignment (runs locally <1 min)
    #. Run USV assignment (runs locally <5 min)
    #. Generate per-USV spectrograms (runs on cluster)
    #. Generate USV masks — YOLO detection + SAM2 segmentation (runs on cluster)
    #. Compute USV acoustic features (runs on cluster)
    #. Infer QLVM latents and watershed categories (runs on cluster)

The QLVM decoder and mask detector that the last two steps rely on are trained separately, once per cohort — see *Train spectrogram-pipeline models* below.

Make mono and crop to video
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running this section, it is always a good idea to check that audio files were copied to the file server correctly. These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run these steps together, you need to list the root directories of interest, select *Convert to single-ch files* and *Crop AUDIO (to VIDEO)*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_9.png
   :align: center
   :alt: Processing Step 9

.. raw:: html

   <br>

If you used the SYNC recording mode (usghflags: 1574), the *Trgbox-USGH device(s)* needs to be set to **m**. If you, however, used the NO SYNC recording mode (usghflags: 1862), the *Trgbox-USGH device(s)* needs to be set to **both**:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_9b.png
   :align: center
   :alt: Processing Step 9b

.. raw:: html

   <br>

The *Convert to single-ch files* step populates the *original* directory with single channel files of the entire recording. The *Crop AUDIO (to VIDEO)* step will crop the audio files to the video duration, and save them in the *cropped_to_video* subdirectory. Both steps require the usage of `sox <https://sourceforge.net/projects/sox/>`_. The *original* directory is later removed by the *A/V sync check* step (once synchronization passes the divergence tolerance), not by this step; reduced to one channel below for brevity:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── **cropped_to_video**
    │   │   │   ├── **m_250430145009_ch01_cropped_to_video.wav**
    │   │   │       ...
    │   │   ├── original_mc
    │   │   │   ├── m_250430145009.wav
    │   │   │       ...
    │   │   ├── **audio_triggerbox_sync_info.json**
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ├── **m_video_frames_in_audio_samples.txt**
    │   │   ├── **s_video_frames_in_audio_samples.txt**
    │   └── video
    │       ...

The *Crop AUDIO (to VIDEO)* step will also result in the creation of a *audio_triggerbox_sync_info.json* file, which contains the sample number of first and last recorded video frame and the break duration detected prior to recording. It will also contain information about the total duration of the audio recording and its discrepancy with the duration of the video recording. In the *sync* subdirectory, the *m_video_frames_in_audio_samples.txt* and *s_video_frames_in_audio_samples.txt* files will be created, which contain the sample numbers of video frame starts in the audio recording. These files are useful should troubleshooting sync issues arise.

.. code-block:: json

    {
        "m": {
            "start_first_recorded_frame": 2654037,
            "end_last_recorded_frame": 302539204,
            "largest_break_duration": 578805,
            "duration_samples": 299885168,
            "duration_seconds": 1199.5407,
            "audio_tracking_diff_seconds": -0.0071,
            "num_dropouts": 0
        }
    }

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section fully modifiable in the GUI, with the following parameters:

* **device_receiving_input** : USGH device receiving Loopbio Triggerbox input (if using SYNC mode, this should be "m")
* **triggerbox_ch_receiving_input** : microphone channel receiving Loopbio Triggerbox input

.. code-block:: json

    "crop_wav_files_to_video": {
        "device_receiving_input": "both",
        "triggerbox_ch_receiving_input": 4
      }

Run HPSS
~~~~~~~~
You have the option to denoise audio data using harmonic-percussive source separation (HPSS; implemented with `librosa <https://librosa.org/doc/main/auto_examples/plot_hprss.html>`_). You can find materials that allow you to run this analysis on the cluster in: */usv-playpen/other/cluster/HPSS*. Alternatively, to run HPSS locally, you need to list the root directories of interest, select *Run HPSS*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_10.png
   :align: center
   :alt: Processing Step 10

.. raw:: html

   <br>

Below, you can see an example of an audio segment with mouse vocalizations before and after such denoising.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/hpss_example.png
   :align: center
   :width: 800
   :height: 600
   :alt: HPSS example

.. raw:: html

   <br>

The *Run HPSS* step populates the *hpss* directory with de-noised single channel files of the entire recording (reduced to one channel for brevity):

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── cropped_to_video
    │   │   │   ...
    │   │   ├── **hpss**
    │   │   │   ├── **m_250430145009_ch01_cropped_to_video_hpss.wav**
    │   │   ├── original_mc
    │   │   │   ...
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

These four parameters are no longer exposed in the GUI (only the *Run HPSS* toggle remains); they are edited directly in the */usv-playpen/_parameter_settings/processing_settings.json* file, under the following keys:

* **stft_window_length_hop_size** : STFT window length and hop size
* **kernel_size** : harmonic-percussive source separation kernel size
* **hpss_power** : harmonic-percussive source separation power
* **margin** : margin for harmonic-percussive source separation

.. code-block:: json

    "hpss_audio": {
        "stft_window_length_hop_size": [
          512,
          128
        ],
        "kernel_size": [
          5,
          60
        ],
        "hpss_power": 4.0,
        "margin": [
          4,
          1
        ]
    }

Filter and concatenate to MEMMAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run these steps together, you need to list the root directories of interest, select *Filter audio files* and *Concatenate to MEMMAP*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_11.png
   :align: center
   :alt: Processing Step 11

.. raw:: html

   <br>

The purpose of these two functions is to first high-pass filter each audio file (removing all lower frequencies) and then concatenate all channels into one `memory-mapped file  <https://docs.python.org/3/library/mmap.html>`_. The first step requires the usage of `sox <https://sourceforge.net/projects/sox/>`_. These processing steps populate the *hpss_filtered* directory with de-noised, high-pass filtered single channel files of the entire recording (reduced to one channel for brevity):

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── cropped_to_video
    │   │   │   ...
    │   │   ├── hpss
    │   │   │   ...
    │   │   ├── **hpss_filtered**
    │   │   │   ├── **250430145009_concatenated_audio_hpss_filtered_250000_299885168_24_int16.mmap**
    │   │   │   ├── **m_250430145009_ch01_cropped_to_video_hpss_filtered.wav**
    │   │   │   ...
    │   │   ├── original_mc
    │   │   │   ...
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section fully modifiable in the GUI, with the following parameters:

* **filter_audio_format** : audio file format (usually "wav")
* **filter_dirs** : list of directories to be filtered (usually "hpss")
* **filter_freq_bounds** : frequency bounds for filtering (usually [0, 30000])

.. code-block:: json

    "filter_audio_files": {
        "filter_audio_format": "wav",
        "filter_dirs": [
          "hpss"
        ],
        "filter_freq_bounds": [
          0,
          30000
        ]
    }

The *Concatenate to MEMMAP* step takes its parameters from the adjacent ``concatenate_audio_files`` block (also fully modifiable in the GUI):

* **concatenate_audio_format** : audio file format to concatenate (usually "wav")
* **concat_dirs** : list of directories whose single-channel files are concatenated into the memory-mapped file (usually "hpss_filtered")

.. code-block:: json

    "concatenate_audio_files": {
        "concatenate_audio_format": "wav",
        "concat_dirs": [
          "hpss_filtered"
        ]
    }

Run DAS inference
~~~~~~~~~~~~~~~~~
The *usv-playpen* GUI assumes usage of the Deep Audio Segmenter (`DAS <https://janclemenslab.org/das/>`_) for identifying vocalizations in audio recordings. To do this, one first needs to train a model on the data of interest (*i.e.*, social interactions with vocal output). Explaining how to do this is beyond the scope of this text, so we will assume you already have a *model* ready for running inference.

Since the average office PC does not necessarily have GPU-capabilities, it is advised to run DAS inference on a high-performance computing cluster, which allows for the parallelization of the inference process. The *usv-playpen* GUI allows you to run the process locally (which can be time consuming), and it provides you with a shell script you can modify for cluster usage (*/usv-playpen/other/cluster/DAS/das_inference_global.sh*). Note that the script does **not** request a GPU: unless the cluster's DAS environment is built against a CUDA TensorFlow, a reserved GPU sits idle, and for a workload of many independent single-channel jobs the queueing delay for a GPU generally outweighs the speed-up. Add ``#SBATCH --gres=gpu:1`` back only once ``python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"`` prints ``True`` inside that environment.

To run DAS inference, you need to list the root directories of interest, select the directory and base name of your DAS model, select *Run DAS inference*, click *Next* and finally *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_12.png
   :align: center
   :alt: Processing Step 12

.. raw:: html

   <br>

This will create a *das_annotations* subdirectory which will contain a CSV file for each recorded channel, denoting the start and end of each detected vocalization.

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── cropped_to_video
    │   │   │   ...
    │   │   ├── **das_annotations**
    │   │   │   ├── **m_250430145009_ch01_cropped_to_video_hpss_filtered_annotations.csv**
    │   │   │   ...
    │   │   ├── hpss
    │   │   │   ...
    │   │   ├── hpss_filtered
    │   │   │   ...
    │   │   ├── original_mc
    │   │   │   ...
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section partially modifiable in the GUI, but it can entirely be modified manually:

* **das_conda_env_name** : name of the local conda environment used for running DAS inference (settings / CLI only; not exposed in the GUI)
* **das_model_directory** : directory containing the trained DAS model
* **model_name_base** : base name (date) of the trained DAS model
* **output_file_type** : output file type ("csv" or "h5")
* **segment_confidence_threshold** : confidence threshold for segmenting vocalizations
* **segment_minlen** : minimum length of segments to be considered vocalizations
* **segment_fillgap** : maximum gap between segments to be joined into a single vocalization

.. code-block:: json

    "das_command_line_inference": {
        "das_conda_env_name": "das",
        "das_model_directory": "/mnt/falkner/Bartul/DAS/model_2024-03-25",
        "model_name_base": "20240325_073951",
        "output_file_type": "csv",
        "segment_confidence_threshold": 0.5,
        "segment_minlen": 0.015,
        "segment_fillgap": 0.015
      },

Curate DAS outputs
~~~~~~~~~~~~~~~~~~
As explained above, DAS is run on every channel separately, such that a need arises to systematize different channel detections in one singular table. This code identifies the same detections across different channels and creates a single CSV file with the start and end times of each detected vocalization. Detections are combined by a greedy interval union: segments are sorted by start time and any two that overlap are merged, transitively, so one USV spans the full extent of every per-channel detection that belongs to it. Boundaries therefore follow the outermost channel that heard the call, which keeps the faint onsets and offsets registered by only the nearest few microphones (Phase-4 noise rejection, below, decides which detections to keep). A coverage-watershed merge was used between August 2026 and this release; it trimmed each USV to the span where a fixed fraction of its peak channel count still agreed, which clipped exactly those faint edges, and it has been reverted. Merged intervals longer than ``max_usv_duration_s`` are rejected before the Phase-4 checks see them. Those checks ask whether a detection correlates across channels, which broadband noise satisfies easily: session ``20240229_163242`` carried six "USVs" with a median duration of 3,074 ms, on all 24 channels, and every one survived correlation and coherence. Duration is the discriminator correlation cannot supply -- no mouse USV lasts a second. With the gate in place that session yields no vocalizations at all, which is the correct reading of a noise-only recording. One failure mode of the union is corrected before the summary is written. Because the outermost edge wins, a single channel that fuses a run of calls into one long detection drags every other channel's boundaries out with it. In session ``20250919_145712`` twenty-three of twenty-four channels resolved seven separate calls between 0.9 s and 1.9 s while one channel emitted a single 897 ms detection, and the merged USV became 901 ms; the fused detection is in the raw annotation, so re-inference does not change it, and the channel is otherwise healthy (974 detections against a cohort median of 968, and only two of them anomalous) so excluding it session-wide would discard a good microphone. An interval is therefore re-merged from the remaining channels when it is longer than ``consensus_remerge_min_duration_s``, at most ``consensus_remerge_max_dissenting_channels`` of its channels have a segment exceeding ``consensus_remerge_span_factor`` times the median longest-segment, at least ``consensus_remerge_min_agreeing_channels`` remain, and that re-merge actually yields more than one interval. A channel set aside for the boundary is still credited in ``chs_detected`` wherever its own segment covers a sub-interval, since it did hear the call and only failed to separate it. The last condition is what keeps dense-but-genuine runs intact: where the long channel agrees with the others, the consensus re-merge reproduces one interval and nothing changes. A cut must also sit in a gap the segmenter could have produced. Because DAS closes gaps under 15 ms within a channel, anything finer in the merged output comes from taking the union across channels and cutting it again, where different channels' detections stop at slightly different instants; left alone it puts boundaries inside calls, as at ``20250928_172408`` 565.074 s, cut into 18 ms and 133 ms pieces divided by 0.1 ms. Matching ``consensus_remerge_min_gap_s`` to the segmenter's own resolution removed that class -- 54 of 232 split decisions across twelve sessions, each one a merge rather than a new cut -- while staying discriminating: at ``20250928_175135`` 303.130 s a 17.1 ms gap survives and a 4.9 ms one closes. Boundary errors above that width are not separable from correct splits on the annotations alone: a confirmed correct split at 16.4 ms and a confirmed false one at 17.1 ms come from the same session. The correction is applied recursively, because a single pass judges dissent against the median longest-segment of the whole interval: in ``20240311_143803`` at 300.816 s a 779 ms interval had a 96 ms median, so the channels fusing two 30-40 ms calls at 102-107 ms read as ordinary and their fusions survived. Re-testing each corrected sub-interval against its own median exposes them, which is why ``consensus_remerge_min_duration_s`` is 80 ms rather than the interval-level figure that would protect ordinary calls. Only the INTERNAL cuts come from the consensus: the outer edges stay the union's, since setting a channel aside also removes its contribution to the interval's own start and stop, and that contribution is the faint edge only the nearest microphone registered. Recursion is concentrated where calls are packed tightly -- across eight sessions, 736 of 895 corrected intervals came from one dense recording and most others contributed one or two. Reviewed against spectrograms over 13 sampled cases, one was a false split. Far narrower than the coverage-watershed merge, which reshaped every USV. Microphones recorded as hardware-compromised for a session -- the ``excluded_channels`` list under ``Equipment -> audio_Avisoft`` in the session's ``*_metadata.yaml`` -- are skipped outright: their annotation files never enter the merge. The field is absent on healthy sessions (no exclusions).

.. note::

   **Excluding a compromised microphone (**\ ``excluded_channels``\ **).** A
   microphone that was broken, unplugged or noisy for a given recording is
   excluded **per session, from that session's metadata** -- not from
   ``processing_settings.json``, which has no exclusion key. Add the channel
   names by hand under ``Equipment -> audio_Avisoft`` in the session's
   ``*_metadata.yaml``:

   .. code-block:: yaml

        Equipment:
          audio_Avisoft:
            excluded_channels:
              - m_ch02
              - m_ch08
              - s_ch11

   There is deliberately no GUI field or CLI flag for this: an exclusion
   describes the *recording*, not the analysis run, so it travels with the
   session rather than with the settings of whoever processes it. Two steps
   honour the list -- ``das-summarize`` (the excluded channels' annotation
   files never enter the merge) and
   ``generate-usv-spectrograms`` (they are dropped from the variance-weighted
   average) -- so a channel excluded after those steps have run only takes
   effect when they are re-run. Names must be ``m_chNN`` / ``s_chNN`` with
   ``NN`` in ``01``-``12``; a malformed entry fails loud rather than being
   silently ignored. The field is absent in healthy sessions, which simply
   means nothing is excluded.

To run, you need to list the root directories of interest, select *Curate DAS outputs*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_13.png
   :align: center
   :alt: Processing Step 13

.. raw:: html

   <br>

This process will create [1] a *20250430_145017_usv_summary.csv* file, and [2] a 20250430_145017_usv_signal_correlation_histogram.svg file, as shown below:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── cropped_to_video
    │   │   │   ...
    │   │   ├── das_annotations
    │   │   │   ...
    │   │   ├── hpss
    │   │   │   ...
    │   │   ├── hpss_filtered
    │   │   │   ...
    │   │   ├── original_mc
    │   │   │   ...
    │   │   ├── **20250430_145017_usv_summary.csv**
    │   │   ├── **20250430_145017_usv_signal_correlation_histogram.svg**
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The *usv_summary.csv* file should look similar to an example table below:

.. parsed-literal::
    ┌────────┬─────────────┬─────────────┬──────────┬───┬─────────────┬───────────┬─────────────────────────────────┬──────────┐
    │ usv_id ┆ start       ┆ stop        ┆ duration ┆ … ┆ mean_amp_ch ┆ chs_count ┆ chs_detected                    ┆ emitter  │
    │ ---    ┆ ---         ┆ ---         ┆ ---      ┆   ┆ ---         ┆ ---       ┆ ---                             ┆ ---      │
    ╞════════╪═════════════╪═════════════╪══════════╪═══╪═════════════╪═══════════╪═════════════════════════════════╪══════════╡
    │ 000000 ┆ 0.23296     ┆ 0.299388    ┆ 0.066428 ┆ … ┆ 17.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 000001 ┆ 0.36064     ┆ 0.42278     ┆ 0.06214  ┆ … ┆ 17.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 000002 ┆ 0.488896    ┆ 0.58534     ┆ 0.096444 ┆ … ┆ 2.0         ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 000003 ┆ 0.643392    ┆ 0.734588    ┆ 0.091196 ┆ … ┆ 2.0         ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 000004 ┆ 0.800192    ┆ 0.942972    ┆ 0.14278  ┆ … ┆ 11.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ …      ┆ …           ┆ …           ┆ …        ┆ … ┆ …           ┆ …         ┆ …                               ┆ …        │
    │ 002561 ┆ 1193.784896 ┆ 1193.828988 ┆ 0.044092 ┆ … ┆ 23.0        ┆ 20.0      ┆ [0, 1, 2, 3, 5, 6, 7, 8, 9, 11… ┆ null     │
    │ 002562 ┆ 1195.412544 ┆ 1195.433852 ┆ 0.021308 ┆ … ┆ 23.0        ┆ 1.0       ┆ [23]                            ┆ null     │
    │ 002563 ┆ 1195.531392 ┆ 1195.5639   ┆ 0.032508 ┆ … ┆ 23.0        ┆ 4.0       ┆ [0, 17, 21, 23]                 ┆ null     │
    │ 002564 ┆ 1195.775552 ┆ 1195.81926  ┆ 0.043708 ┆ … ┆ 23.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 002565 ┆ 1197.163712 ┆ 1197.196348 ┆ 0.032636 ┆ … ┆ 6.0         ┆ 2.0       ┆ [4, 6]                          ┆ null     │
    └────────┴─────────────┴─────────────┴──────────┴───┴─────────────┴───────────┴─────────────────────────────────┴──────────┘


The *usv_signal_correlation_histogram.svg* file contains a histogram of [1] mean spectrogram correlations across each candidate's detected channels and its absolute cutoff, and [2] the histogram of spatial coherences (mean pairwise correlation across the loudest in-band channels of the array) and its absolute cutoff (an example of which is shown below). A candidate must clear both cutoffs to be kept: noise correlates poorly across the detecting channels, and a localized artifact's pattern is absent from the array's loudest channels.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/usv_signal_correlation_histogram_example.png
   :align: center
   :alt: Correlation and variance of signal summary

.. raw:: html

   <br>

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section not modifiable in the GUI, but it can be modified manually:

* **filter_putative_noise_bool** : whether to run the Phase-4 amplitude/spectrogram noise rejection; when ``false``, every merged detection is kept and the summary CSV is written as-is (peak/mean amplitude channels left at 0)
* **len_win_signal** : STFT window length
* **low_freq_cutoff** : frequency cutoff for filtering (in Hz)
* **noise_corr_cutoff_min** : absolute cutoff on the cross-channel spectral correlation across a candidate's DAS-detected channels; a multi-channel candidate below it is dropped. Absolute rather than session-relative, because percentile rules overshoot on clean candidate pools and undershoot on noise-dominated ones (empirically, cohort-wide, noise sits at correlation ~0.25 and verified real calls at >=0.30)
* **coherence_cutoff_min** : absolute cutoff on the spatial coherence -- the mean pairwise spectral correlation across the loudest in-band channels of the whole array, detection status ignored. A genuine call dominates the in-band soundscape at its moment, so its loudest channels share one time-frequency pattern; a localized artifact's pattern appears nowhere else. Candidates below the cutoff are dropped; single-channel candidates (no defined detected-channel correlation) are gated by coherence alone
* **coherence_channel_count** : number of loudest in-band channels the spatial coherence is computed over
* **max_usv_duration_bool** : whether to reject merged intervals longer than **max_usv_duration_s** before the Phase-4 noise checks run (default ``true``)
* **max_usv_duration_s** : longest a merged interval may be (in s) before it is rejected as noise; measured over 439,301 USVs in 246 sessions the median is 86 ms, p99 371 ms and p99.99 896 ms, so the ``1.0`` default is past any plausible call or tightly-overlapping cluster and removes 0.005% of intervals
* **consensus_remerge_bool** : whether to re-merge an interval whose extent is set by a channel contradicting the majority (default ``true``); when ``false`` the union is taken exactly as detected
* **consensus_remerge_min_duration_s** : only merged intervals longer than this (in s) are examined, so ordinary calls are never reconsidered
* **consensus_remerge_span_factor** : a contributing channel counts as dissenting when its longest segment exceeds this multiple of the median longest-segment across the contributing channels
* **consensus_remerge_max_dissenting_channels** : fire only when at most this many channels dominate; more than that is not a lone dissenter
* **consensus_remerge_min_agreeing_channels** : fire only when at least this many channels remain once the dissenters are set aside
* **consensus_remerge_min_gap_s** : narrowest gap a cut may sit in (in s); DAS is run with ``--segment-fillgap 0.015`` and so closes anything under 15 ms within a channel (measured: 99.87% of 221,522 within-channel gaps are >= 15 ms, 1st percentile 21 ms), which means a finer gap in the merged output was manufactured by the union rather than found by the segmenter
* **consensus_remerge_rescue_dissenting_channels** : dissent limit used only for an interval that would otherwise be deleted for exceeding **max_usv_duration_s**; relaxing it everywhere is not justified (over twelve intervals judged against spectrograms a limit of 4 was better in six and worse in five), but applied on that path alone it changed 0 of 13,851 intervals across eleven sessions while rescuing 1,094 ms of genuine calls
* **consensus_remerge_max_depth** : how many times a corrected interval may itself be re-examined; ``0`` reproduces the single-pass behaviour
* **seam_repair_bool** : whether to run the post-summary seam check-and-repair (below); when ``false`` (the default), seam-snapped boundaries are left as detected. The artifact it corrects is produced by the legacy non-overlapping model (``nb_hist`` 8192, ``stride`` 8128); annotations from the overlap-tiled model carry no seam-ladder fingerprints, so the repair has nothing to act on
* **seam_repair_legacy_stride_samples** : window-stitching stride (in samples) of the legacy non-overlapping DAS model whose seam artifact is checked for (``8128`` for the lab's 2024-03-25 model at 250 kHz)
* **seam_repair_max_rung** : highest stride multiple tested for the seam-ladder fingerprint
* **seam_repair_ladder_tolerance_samples** : maximum deviation (in samples) from the exact ``k * stride + 1`` ladder fingerprint for a raw-annotation gap to count as seam-snapped
* **seam_repair_width_tolerance_below_ms** : merged-gap width gate reach below each stride multiple (in ms); merged gaps can sit slightly off the exact raw ladder values because the union takes the outermost edge across channels
* **seam_repair_width_tolerance_above_ms** : merged-gap width gate reach above each stride multiple (in ms)
* **seam_repair_raw_stop_tolerance_s** : maximum distance (in s) between a merged pair's stop and a raw ladder-gap stop for the pair to count as corroborated
* **seam_repair_snippet_margin_s** : audio margin (in s) excised around each flagged pair for re-detection
* **seam_repair_max_boundary_shift_s** : maximum outward correction (in s) permitted per call edge; mechanistically one legacy stride

.. code-block:: json

     "summarize_das_findings": {
        "filter_putative_noise_bool": true,
        "len_win_signal": 512,
        "low_freq_cutoff": 30000,
        "noise_corr_cutoff_min": 0.3,
        "coherence_cutoff_min": 0.2,
        "coherence_channel_count": 3,
        "seam_repair_bool": false,
        "seam_repair_legacy_stride_samples": 8128,
        "seam_repair_max_rung": 6,
        "seam_repair_ladder_tolerance_samples": 2,
        "seam_repair_width_tolerance_below_ms": 3.0,
        "seam_repair_width_tolerance_above_ms": 1.0,
        "seam_repair_raw_stop_tolerance_s": 0.01,
        "seam_repair_snippet_margin_s": 0.05,
        "seam_repair_max_boundary_shift_s": 0.034
     }

Detect noise
~~~~~~~~~~~~

The DAS segmenter keeps every interval a channel fired on, so a curated *usv_summary.csv* also holds segments with no vocalization in them: electrical clicks, cage knocks, broadband transients and faint smears. *Detect noise* scores every USV segment with an ensemble of five time-resolved multiple-instance classifiers (the bundle is derived from the *Spectrogram models directory* as ``noise/noise_timemil_ens5_n4680_20260926.pt``, whose name records the architecture, the ensemble size, the 4,680 training labels and the build date) and adds two columns to *usv_summary.csv*:

* ``noise`` -- ``true`` / ``false`` in every row; ``true`` **excludes** the segment, because the model is not confident it holds a vocalization (``noise_probability`` at or above the bundle's exclusion cut-off, 0.14);
* ``noise_probability`` -- the ensemble's probability in every row, so an analysis can tell confident noise (at or above 0.82) from the uncertain band, or re-threshold, without re-running the step.

A segment is *noise* only when it holds no vocalization at all -- neither a USV nor a squeak -- which is the rule its training labels follow, so a faint call heard on one microphone, or a call mixed with noise, is a vocalization, not noise.

The decision is fixed by the model bundle, not by a setting. The ensemble's probability splits the segments into confident vocalizations (below 0.14), an uncertain band (0.14-0.82) and confident noise (0.82 and above), and ``noise`` is ``true`` for the last two: uncertain segments are excluded with the noise rather than guessed. The cut-offs are the narrowest uncertain band whose confident decisions reach precision and recall of 0.95 on 1,118 segments drawn at random from all listed sessions and labelled by consensus (the majority of up to four blind passes), weighted to the cohort. Applied held out -- the cut-offs chosen on four session folds and scored on the fifth, each segment judged by an ensemble that never saw its session -- the confident decisions reach **precision 0.944 and recall 0.955**, with 0.9% of segments uncertain. The exclusion costs about 71 real calls per 10,000 segments (65 of them in the uncertain band, which holds mostly faint, short calls heard on one or two microphones). The step prints these numbers on every run.

An earlier bundle (``noise_timemil_ens5_n3562_20260916.pt``) described its calibration as measured on sessions the models never trained on; it was not -- 422 of its 633 calibration segments were training labels -- and on new, cohort-representative labels it reached precision 0.78 and recall 0.95 at its threshold, not the 0.987 / 0.973 it reported. The detector refuses bundles without the validated decision block.

Each segment's input is the two-band absolute-dB spectrogram (30-120 kHz and 3-30 kHz, 128 linear bins each) of the **unfiltered** per-channel *audio/hpss* wavs, averaged across channels by variance with metadata-excluded channels dropped. The audio window extends ~100 ms either side of the segment so the channel weights and STFT edges match the training inputs, and the spectrogram is then cropped back to the segment's own frames: the model judges the segment, not its neighbourhood.

Run *Detect noise* after *Curate DAS outputs* (re-curating rewrites *usv_summary.csv* with its base columns only) and before *Detect squeaks*, the order the summary's column layout follows. A GPU is used when present but is not required: a 424-USV session takes about 64 s on one and 90 s on the CPU, because most of the time goes on reading and transforming the audio rather than on the network.

Detect squeaks
~~~~~~~~~~~~~~

A *squeak* is a broadband harmonic stack (fundamental around 3-8 kHz) that the ultrasonic DAS segmenter picks up as part of a USV segment; a segment can hold an ultrasonic call alone, a squeak alone, or both. Once the curated *usv_summary.csv* exists and *Detect noise* has run, *Detect squeaks* classifies every segment that is not noise into one of three call classes with an ensemble of five time-resolved multiple-instance classifiers (the bundle is derived from the *Spectrogram models directory* as ``squeak/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt``) and adds eight columns to *usv_summary.csv*:

* ``call_class`` -- ``usv`` (an ultrasonic call only), ``squeak`` (a squeak only) or ``both`` (a squeak and an ultrasonic call in the same segment): the class of highest ensemble-mean probability; null on noise rows (and on the rare segment too short for one STFT window);
* ``p_usv`` / ``p_squeak`` / ``p_both`` -- the ensemble-mean class probabilities (they sum to 1), so an analysis can re-threshold without re-running the step; null where ``call_class`` is null;
* ``squeak_spans`` -- a JSON list of ``[start_s, end_s]`` pairs in session seconds (the clock of ``start`` / ``stop``), one per squeak, in time order; ``"[]"`` on ``usv`` rows; null where ``call_class`` is null;
* ``squeak_start`` / ``squeak_end`` -- the envelope of the spans (the first span's start, the last span's end); null when the row has no span;
* ``n_squeaks`` -- the number of spans (0 on ``usv`` rows); null where ``call_class`` is null.

The columns of the retired binary squeak detector (``squeak``, ``squeak_probability``, ``squeak_frame_runs``) are removed when a summary is rewritten; every other column is left exactly as it was.

The classifier reads the noise model's input (see *Detect noise*): two absolute-dB bands (30-120 kHz and 3-30 kHz, 128 linear bins each) of the variance-weighted average of the **unfiltered** per-channel *audio/hpss* wavs (*audio/hpss_filtered* is high-passed above 30 kHz and carries no squeak energy), with metadata-excluded channels dropped. The audio window is the noise model's too -- the segment plus 49 hops (100 ms) of context either side -- but the spectrogram is **not** cropped back to the segment, because squeaks often run past the segment the ultrasonic segmenter cut; a third input channel marks the segment's own frames, so the network knows which frames the class decision is about. The network is the noise model's trunk (its trained weights are the starting point of every member) with two heads: a three-class head attention-pooled over the segment's frames (plus a linear term in the log channel count and log duration), and a per-frame squeak head over every frame of the window. A squeak span is every run of frames whose ensemble-mean squeak probability exceeds 0.6 for at least 12 frames (24.6 ms), from half a hop before its first frame's centre to half a hop after its last; only spans that overlap the segment are kept (a span lying wholly in the context belongs to a neighbouring segment), and a kept span may extend past the segment into the context. Spans are written on ``squeak`` and ``both`` rows only; a ``squeak`` / ``both`` row whose track has no run passing the rule gets ``"[]"`` and no envelope. The rule, its constants and every input constant are read from the bundle.

Held out by session (5-fold grouped cross-fitting on the 2,476 non-unsure labels of two labelling rounds, the first drawn blind and stratified across the cohort), the three classes reach precision / recall of 0.999 / 0.993 (usv), 0.983 / 0.950 (squeak) and 0.915 / 0.985 (both) on the blind round's 1,981 segments; the commonest confusion is a usv segment called both (12 of 1,664), and squeak and usv are never confused with each other (the bundle's ``cv_summary`` records the tables). On three sessions (2,951 segments; 2023, 2025 and 2026 recordings, one with metadata-excluded microphones), this step reproduced the 49 labelled segments' inputs bit for bit against the inputs the bundle was trained on, its call class on all 49, its class probabilities to within 6.4e-6 (batching only) and its spans on all 49, and left every other column of the summaries unchanged. There, about 5 % of the ``squeak`` / ``both`` rows had no run passing the span rule (no envelope), 34 rows held more than one squeak, and two thirds of the spans extended past their segment.

Run *Detect squeaks* after *Curate DAS outputs* and *Detect noise*: re-curating rewrites *usv_summary.csv* with its base columns only, which removes the call-class columns, and the step needs the ``noise`` column (the processing run orders the steps accordingly). A GPU is recommended. The ensemble is trained by ``train-usv-squeak-model`` (see *Call-class model* below).

.. note::

   The ensemble was fitted on the labelled segments of 2,476 panels across the cohort (the label files the bundle's ``labels`` / ``label_overrides`` name). Its output on those segments is a fit, not a prediction: leave them out of any held-out accuracy claim.

Embed squeaks in a QLVM torus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Squeaks have their own QLVM models: the v3 model package's phase 3 broadband-vocalization cells (``/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/phase3_BBVs_qlvm/``: ``natural_lumped``, ``natural_session``, ``uniform_lumped`` and ``uniform_session``, each ``_N11000_nomask``), 2-D tori trained on sonic-band squeak crops and embedded over the package's 14,936-squeak corpus. ``infer-qlvm-squeak-latents`` (command line only; not a step of the processing run) places every segment of a session whose ``call_class`` is ``squeak`` or ``both`` and that is not noise (a null ``noise`` counts as not noise, the single noise rule the analyses use) on the torus of ONE of these cells and adds two columns to *usv_summary.csv*:

* ``qlvm_squeak1`` / ``qlvm_squeak2`` -- the squeak's torus coordinates in ``[0, 1)`` (the posterior mean, the same convention as ``qlvm1`` / ``qlvm2``), null on every other row. No cluster labels are written.

The input of each segment is rebuilt the way the cells' training sets were built (the reference builder ``build_bbv_dataset.py``, MMMmB): the cells' sonic spectrogram (3-30 kHz, 128 linear bins, absolute dB, from the unfiltered *audio/hpss* wavs; the front end of the retired squeak classifier the cells were trained on) is cropped to the frames whose centres lie inside the squeak envelope ``squeak_start`` .. ``squeak_end``, plus two frames of context either side, min-max normalized per crop, written into a 128-frame zero frame, centred without time stretching and min-max normalized once more, as the decoder's data loader did. Because a squeak often extends past its segment, the audio is read over the segment widened, where needed, to hold the envelope and its two context frames (the segment itself otherwise, as for the training crops), so a crop is never cut at the segment boundary. One row holds one pair of coordinates, so a segment with several squeaks (``n_squeaks`` > 1) is embedded once, over the envelope of all its spans, gaps included. Crops narrower than 8 frames (the training set's minimum) or wider than 128 frames (the decoder frame; the training crops were never compressed in time), and ``squeak`` / ``both`` rows without a span, get nulls. One deliberate difference: the training set cropped from a store that held only a segment's first 128 frames and dropped squeaks that ran past its end (right-censored), whereas this step crops from the full-length spectrogram, so such a squeak is embedded with its measured extent when its crop fits in 128 frames. The posterior mean is taken over the cell's 46,368-point Fibonacci lattice (``lattice_m`` 24 of its ``manifest.json``), built in float32 exactly as the torch driver that embedded the corpus built it: the exact (float64) lattice sits up to 2.3e-3 away from it and moves the embeddings by a median 6e-4.

Validated against the corpus embeddings the four cells ship (``posterior_cache.npz``, 14,936 squeaks of 239 sessions): the crop code reproduces every corpus input bit-exactly from the reference sonic spectrogram store, and on the reference inputs the embedding matches the package's corpus embedding to a median wrapped distance of 2.6e-5 to 3.1e-5 of the torus, with 99.5 % of squeaks within 1e-3, in all four cells. The remaining difference is numerical: the corpus embedding's GPU run computed the transposed convolutions in TF32 (torch's cuDNN default) and this step in full float32; a torch replica of the original torch driver in full float32 lands where this step does, and with TF32 on it reproduces the two lumped cells to a median 1.5e-6. Built from the session audio instead of the reference store, the inputs were bit-identical wherever the crop and the microphones agreed, with two exceptions by design. The reference store averaged all 24 microphones, while this step drops metadata-excluded channels like *Detect squeaks* does (13 corpus sessions, all from January and February 2023, exclude microphones; ``--no-exclude-metadata-audio-channels`` reproduces the reference inputs there bit-exactly). And a segment longer than 128 frames is cropped with its full-length extent (above). These checks were made with the extents of the retired binary classifier; the crop extents now come from the call-class model's squeak spans, so the squeaks (and their crops) differ from the corpus ones and the corpus comparison holds for the crop and embedding code, not for the extents.

The cell is the ``infer_qlvm_squeak_latents.model_cell_directory`` setting. The shipped settings leave it empty, and an empty setting is filled with the production squeak cell ``phase3_BBVs_qlvm/natural_session_N11000_nomask`` (``os_utils.QLVM_SQUEAK_PACKAGE_ROOT`` / ``QLVM_SQUEAK_PRODUCTION_CELL``: the natural draw over the duration bins with a per-session bin cap) whenever ``spectrograms_root`` is set; name another cell with ``--model-cell-directory``. The cells keep the old package layout (every file at the cell root, no ``training_contract.json``), so the step reads and checks what they do ship: the decoder weights of ``checkpoint.tar`` (read without torch), ``run_config.json`` (a 2-D, unmasked BBV run) and ``manifest.json`` (an unmasked, un-stretched 128 x 128 corpus and its ``lattice_m``). Run it after *Detect noise* and *Detect squeaks*; a GPU speeds it up but is not required.

Seam check-and-repair
~~~~~~~~~~~~~~~~~~~~~
DAS models inferring with non-overlapping window tiling (stride equal to the window length minus a 32-sample edge trim, i.e. ``stride: 8128`` / ``nb_hist: 8192`` at 250 kHz) judge each stitched span without acoustic context from its neighbours. The consequence is a boundary artifact on *real inter-call pauses*: the faint edges of the calls flanking a pause that straddles window seams are clipped exactly to the seam positions, so the pause is recorded with a width of exactly ``k`` stride multiples (a "ladder" at 32.5 / 65.0 / 97.5 ... ms, sample-exact in the raw annotations) and the flanking calls lose up to one stride of quiet edge material. Nothing is split or duplicated -- the audio between the recorded endpoints is genuinely silent -- but the affected gap widths are quantized (which corrupts inter-USV interval statistics) and the adjacent onsets/offsets are misplaced by up to ~32 ms.

.. note::
   This repair is **disabled by default** (``seam_repair_bool: false``) and exists for annotations that were already
   produced with non-overlapping tiling. It is a remedy of last resort, not the fix. The fix is to infer with
   overlapping windows in the first place: set ``stride`` to at most half of ``nb_hist`` and raise ``data_padding``
   to cover the model's receptive field (the lab's 2024-03-25 model moved from ``stride: 8128`` / ``data_padding: 32``
   to ``stride: 4096`` / ``data_padding: 2048`` in its ``*_params.yaml``). Overlapping windows let every sample be
   judged with acoustic context on both sides, so no boundary is ever forced onto a seam.

   Measured over six re-inferred sessions (97,011 vocalizations before, 104,333 after), taking the phase of every
   USV boundary within the old 32.512 ms window period over 32 bins -- where a flat distribution puts 0.0312 of the
   boundaries in each bin -- the non-overlapping annotations placed **0.2841 in the busiest bin (9.1x uniform)**,
   while the overlap-tiled ones placed **0.0324 (1.0x uniform)**. Median USV duration fell from 39.2 to 36.5 ms, the
   seam-clipped edge material being restored. Re-inference removes the artifact; the repair only patches it.

   Re-inference is expensive, though, and halving the stride roughly doubles the windows scored per channel (measured
   on this cohort: 342 s to ~505 s per 20-minute channel). Where re-running inference is not practical, this repair
   remains the way to recover usable inter-call intervals from legacy annotations -- enable it with
   ``--seam-repair``.

After every summary build (and standalone, by calling ``repair_seam_snapped_boundaries`` on an existing summary -- the standalone path never rebuilds the summary, so enrichment columns like ``emitter`` and the ``qlvm_*`` set survive), the tool:

#. flags consecutive summary pairs whose gap falls inside the width-gate window around a stride multiple **and** is corroborated by a sample-exact ladder gap in at least one channel's raw DAS annotations at that location -- the joint criterion keeps width-window innocents out, and any that do slip through are re-confirmed unchanged by construction;
#. excises a snippet around each flagged pair from the corroborating channel's HPSS-filtered WAV into ``audio/seam_repair_snippets`` (removed afterwards);
#. re-runs ``das predict`` once on the snippet folder with the model configured under ``das_command_line_inference`` -- the current model tiles with overlapping windows (``stride: 4096`` / ``data_padding: 2048``, i.e. >=8 ms of real context for every kept sample), so no seam is judged context-blind;
#. replaces each pair's facing edges with the re-detected ones, **outward-only** (the artifact only ever clips edges, so true edges lie inside the recorded gap; inward suggestions are single-channel-versus-merge convention differences and clamp to the recorded boundary) and capped at ``seam_repair_max_boundary_shift_s`` per edge, updating ``start`` / ``stop`` / ``duration``.

The summary CSV is rewritten atomically with every other column preserved, and a sidecar ``*_seam_repair_report.json`` in the session's ``audio`` directory records the settings used and the outcome for every flagged pair (``repaired`` / ``no_change`` / ``skipped:*``). Sessions inferred with the overlap-tiled model produce no ladder fingerprints, so the check is a no-op there by construction. Note that per-USV spectrograms, masks, and acoustic-feature columns generated *before* a repair reflect the old boundaries of repaired rows -- regenerate them for those rows (the report lists them) if an analysis depends on exact call edges.

Prepare and run USV assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You might also want to know which animal emitted which vocalization. To do this, *usv-playpen* relies on `vocalocator <https://github.com/neurostatslab/vocalocator>`_, a tool for localizing animal vocalizations in 3D space, and it assumes you already have a trained model. These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run these steps together, you need to list the root directories of interest, select the arena directory, select the directory of the vocalocator model, select *Prepare USV assignment* and *Run USV assignment*, select the *Assignment type* (``vcl`` or ``vcl-ssl``), click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_14.png
   :align: center
   :alt: Processing Step 14

.. raw:: html

   <br>

This will create a *sound_localization* subdirectory. With the default *vcl-ssl* backend, it contains a *dset.h5* file (all data relevant for sound localization) and a *model_predictions.npz* file, whose predictions are transferred to the "emitter" column of the *20250430_145017_usv_summary.csv* file. (With the older *vcl* backend it instead contains *dset.h5*, an *assessment.h5* file with 2D assessment data, and an *assessment_assn.npy* file with 6D assessment output that feeds the "emitter" column.)

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── cropped_to_video
    │   │   │   ...
    │   │   ├── das_annotations
    │   │   │   ...
    │   │   ├── hpss
    │   │   │   ...
    │   │   ├── hpss_filtered
    │   │   │   ...
    │   │   ├── original_mc
    │   │   │   ...
    │   │   ├── **sound_localization**
    │   │   │   ├── **model_predictions.npz**
    │   │   │   ├── **dset.h5**
    │   │   ├── **20250430_145017_usv_summary.csv**
    │   │   ├── 20250430_145017_usv_signal_correlation_histogram.svg
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The modified *usv_summary.csv* file now contains information in the last column for those vocalizations that have been attributed to specific animals:

.. parsed-literal::
    ┌────────┬─────────────┬─────────────┬──────────┬───┬─────────────┬───────────┬─────────────────────────────────┬──────────┐
    │ usv_id ┆ start       ┆ stop        ┆ duration ┆ … ┆ mean_amp_ch ┆ chs_count ┆ chs_detected                    ┆ emitter  │
    │ ---    ┆ ---         ┆ ---         ┆ ---      ┆   ┆ ---         ┆ ---       ┆ ---                             ┆ ---      │
    │ i64    ┆ f64         ┆ f64         ┆ f64      ┆   ┆ f64         ┆ f64       ┆ str                             ┆ str      │
    ╞════════╪═════════════╪═════════════╪══════════╪═══╪═════════════╪═══════════╪═════════════════════════════════╪══════════╡
    │ 000000 ┆ 0.23296     ┆ 0.299388    ┆ 0.066428 ┆ … ┆ 17.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 000001 ┆ 0.36064     ┆ 0.42278     ┆ 0.06214  ┆ … ┆ 17.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 000002 ┆ 0.488896    ┆ 0.58534     ┆ 0.096444 ┆ … ┆ 2.0         ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ 158114_2 │
    │ 000003 ┆ 0.643392    ┆ 0.734588    ┆ 0.091196 ┆ … ┆ 2.0         ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ 158114_2 │
    │ 000004 ┆ 0.800192    ┆ 0.942972    ┆ 0.14278  ┆ … ┆ 11.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ 158114_2 │
    │ …      ┆ …           ┆ …           ┆ …        ┆ … ┆ …           ┆ …         ┆ …                               ┆ …        │
    │ 002561 ┆ 1193.784896 ┆ 1193.828988 ┆ 0.044092 ┆ … ┆ 23.0        ┆ 20.0      ┆ [0, 1, 2, 3, 5, 6, 7, 8, 9, 11… ┆ null     │
    │ 002562 ┆ 1195.412544 ┆ 1195.433852 ┆ 0.021308 ┆ … ┆ 23.0        ┆ 1.0       ┆ [23]                            ┆ 156693_3 │
    │ 002563 ┆ 1195.531392 ┆ 1195.5639   ┆ 0.032508 ┆ … ┆ 23.0        ┆ 4.0       ┆ [0, 17, 21, 23]                 ┆ null     │
    │ 002564 ┆ 1195.775552 ┆ 1195.81926  ┆ 0.043708 ┆ … ┆ 23.0        ┆ 24.0      ┆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,… ┆ null     │
    │ 002565 ┆ 1197.163712 ┆ 1197.196348 ┆ 0.032636 ┆ … ┆ 6.0         ┆ 2.0       ┆ [4, 6]                          ┆ 156693_3 │
    └────────┴─────────────┴─────────────┴──────────┴───┴─────────────┴───────────┴─────────────────────────────────┴──────────┘


The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section partially modifiable in the GUI, but it can entirely be modified manually:

* **vcl_conda_env_name** : name of the local conda environment used for running Vocalocator (settings / CLI only; not exposed in the GUI)
* **vcl_model_directory** : directory containing the trained Vocalocator model
* **vcl_version** : version of the Vocalocator model (e.g., "vcl-ssl" for the SSL model)

.. code-block:: json

   "vocalocator": {
    "vcl_conda_env_name": "vcl-ssl-ss",
    "vcl_model_directory": "/mnt/falkner/Bartul/sound_localization/mouse_all_model_June2026",
    "vcl_version": "vcl-ssl"
   }

The ``assign_vocalizations`` block holds the confidence-set hyperparameters used
when turning the 6-D localizer output into per-mouse attributions (all exposed as
``vcl-assign`` CLI flags):

* **temperature** : covariance temperature scaling applied to the 6-D predictive covariance.
* **grid_resolution** : spatial ``(x_res, y_res)`` grid the confidence-set PDF is sampled on. The same grid drives both the PDF construction and the point-in-set lookup, so it is a single shared value.
* **n_angle_bins** : number of angular histogram bin edges (``-pi`` to ``pi``).
* **n_samples** : Monte-Carlo sample count for the per-vocalization angle PDF.
* **confidence_level** : confidence level (in ``[0, 1]``) for the extracted confidence sets.
* **angle_pdf_seed** : RNG seed for the reproducible angle-PDF sampling.

.. code-block:: json

   "assign_vocalizations": {
    "temperature": 1.0,
    "grid_resolution": [100, 100],
    "n_angle_bins": 46,
    "n_samples": 500,
    "confidence_level": 0.95,
    "angle_pdf_seed": 0
   }

Render spectrograms and latents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the curated *usv_summary.csv* exists (see *Curate DAS outputs* above), an in-house, self-contained pipeline turns every detected ultrasonic vocalization (USV) into a spectrogram, a USV mask, interpretable acoustic features, and toroidal **QLVM** latents. These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run them together, you need to list the root directories of interest, set the *Spectrogram models directory* (the single root from which the Segment Anything Model 2 (SAM2), YOLO, and QLVM model paths are derived), select *Generate spectrograms*, *Generate masks*, *Compute USV features* and *Infer QLVM latents*, click *Next* and then *Process* (GPU is required):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_15.png
   :align: center
   :alt: Processing Step 15

.. raw:: html

   <br>

The *Generate spectrograms* step computes a variance-weighted, multi-channel spectrogram of every USV (channels listed in the session metadata's ``excluded_channels`` are dropped from the average); *Generate masks* runs a YOLO box detector and prompts SAM2 to segment each call; *Compute USV features* derives per-USV spectral and amplitude features; and *Infer QLVM latents* embeds each spectrogram into the trained QLVM torus and assigns it a vocal category. The mask and latent steps run on the GPU and rely on two pre-trained models (see *Train spectrogram-pipeline models* below). The spectrogram and mask arrays are written to a new *spectrograms* subdirectory, while the acoustic features and QLVM latents are merged into *usv_summary.csv*. After a cohort of sessions has been processed, ``consolidate-spectrogram-store`` merges the per-session H5 files and the QLVM columns of their summaries into one multi-session store under ``spectrograms_root`` (layout below); the newest ``spectrograms_*.h5`` is picked up automatically by every consumer.

The acoustic-feature and QLVM columns:

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ├── cropped_to_video
    │   │   │   ...
    │   │   ├── das_annotations
    │   │   │   ...
    │   │   ├── hpss
    │   │   │   ...
    │   │   ├── hpss_filtered
    │   │   │   ...
    │   │   ├── original_mc
    │   │   │   ...
    │   │   ├── sound_localization
    │   │   │   ...
    │   │   ├── **spectrograms**
    │   │   │   ├── **20250430_145017_spectrograms.h5**
    │   │   ├── 20250430_145017_usv_summary.csv
    │   │   ├── audio_triggerbox_sync_info.json
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   └── video
    │       ...

The *20250430_145017_spectrograms.h5* file holds the spectrograms (created by *Generate spectrograms*) and the masks (appended by *Generate masks*), grouped per session:

.. code-block:: text

    20250430_145017_spectrograms.h5
    ├── frequency_bins (F,)                                  # Generate spectrograms
    ├── spectrogram/20250430_145017/spectrograms (N, F, T)   # Generate spectrograms
    ├── spectrogram/20250430_145017/durations (N,)           # Generate spectrograms
    ├── mask/20250430_145017/segmentations (M, F, T) bool    # Generate masks
    └── mask/20250430_145017/spectrogram_index (M,)          # Generate masks

The spectrogram rows are 1:1 with *usv_summary.csv*; each mask row carries a *spectrogram_index* pointing back to the spectrogram (and USV) it segments. Re-running a step overwrites only the group it owns and leaves the rest of the file intact.

**The consolidated store.** ``consolidate-spectrogram-store`` writes ``spectrograms_qlvmv3_<S>sessions_<N>vocalizations_<UTC timestamp>.h5`` to ``spectrograms_root``. It is meant for the corpus of the QLVM model package v3 (``QLVM_MODEL_PACKAGE_ROOT``): ``--package-corpus`` takes exactly the 402 sessions its ``corpus/SESSION_H5_BASELINE.tsv`` lists (each session root is the third parent of the listed H5 path under ``/mnt/falkner``), ``--root-directories`` any subset of them. Every session is checked before anything is written, and all problems are reported together: its spectrogram H5 must hash (SHA-256) to the baseline's (a rebuilt H5 renumbers the rows the package embedded), its summary must have one row per H5 row and carry every production QLVM column (``qlvm1`` … ``qlvm_loud_supercategory``), all sessions must share one ``frequency_bins`` axis, and the embedded calls must be exactly those the package's rules admit (status 0 exactly where ``qlvm1`` / ``qlvm2`` are non-null, each model's labels exactly where its coordinates are). Sessions are copied one at a time and the file is published atomically.

.. code-block:: text

    spectrograms_qlvmv3_402sessions_<N>vocalizations_<ts>.h5
    ├── frequency_bins (F,)                              # shared by every session
    ├── spectrogram/<session>/spectrograms (N, F, T)     # copied from the session H5
    ├── spectrogram/<session>/durations (N,)             # copied from the session H5
    ├── spectrogram/<session>/qlvm_dim (N, 2) float64    # regular model's qlvm1/qlvm2 (NaN = not embedded); kept for older readers
    ├── mask/<session>/segmentations (M, F, T) bool      # copied from the session H5
    ├── mask/<session>/spectrogram_index (M,)            # copied from the session H5
    ├── qlvm/<session>/<prefix> (N, 2) float64           # qlvm, qlvm_dur, qlvm_mf, qlvm_bw, qlvm_loud coordinates; NaN = not embedded
    ├── qlvm/<session>/<label column> (N,) int16         # <prefix>_category and <prefix>_supercategory of every model
    │                                                    # (qlvm_category, ..., qlvm_loud_supercategory); 0 = no label
    ├── qlvm/<session>/status (N,) int8                  # 0 embedded, 1 too long, 2 no SAM mask, 3 both
    ├── qlvm_models/<prefix>/                            # attrs: package root / name / version, cell, design, phase,
    │   │                                                #        condition, MANIFEST.sha256 SHA-256, training_contract (JSON)
    │   ├── clusters_fine, boundaries_fine               # the cell's clusters.csv / boundaries.csv (compound tables)
    │   ├── label_grid_fine (200, 200) int16
    │   └── clusters_coarse, boundaries_coarse,          # every model (both label levels are written)
    │       label_grid_coarse, fine_to_coarse
    └── sessions                                         # session_id, session_type, spectrogram_h5_sha256,
                                                         # usv_summary_sha256, n_rows, n_embedded

The status of a call is computed from the H5: bit 0 is set when its duration is at or above the regular cell contract's ``length_threshold`` (128 time bins), bit 1 when it has no SAM mask instance (the bincount of ``mask/<session>/spectrogram_index`` is 0). ``session_type`` comes from the regular cell's ``inference/recon_mse_breakdown.npz`` (``session_types``), the source the package names for it. The tables are compound datasets with a ``schema_json`` attr (read them back with ``usv_interval_archive._h5_to_polars``). A non-default label level (e.g. ``qlvm_dur_supercategory``) is stored, with its tables, when every session's summary carries it. The root attrs record ``created_by``, ``created_date`` (UTC), ``git_commit``, ``n_sessions``, ``n_vocalizations``, ``package_root``, ``package_manifest_sha256`` and ``generate_spectrograms_settings`` -- the ``generate_spectrograms`` processing settings in force at consolidation, as JSON; the settings each session's spectrograms were generated with were not recorded per session, as ``generate_spectrograms_settings_note`` says. Older ``spectrograms_sam2masks_*`` stores are still resolved; the newest ``spectrograms_*.h5`` of either name wins.

**The squeak spectrogram store.** The consolidated store's spectrograms span 30-125 kHz, which shows only the ultrasonic tail of a squeak (a broadband harmonic stack with its fundamental around 3-8 kHz). ``build-squeak-spectrogram-store`` (``build_squeak_spectrogram_store`` settings block) writes a second, squeak-only store to ``spectrograms_root`` that the embedding explorer's *Squeaks* map shows instead (see :doc:`Notebooks`): ``squeak_spectrograms_<F>logbins_<S>sessions_<N>squeaks_<UTC timestamp>.h5``, resolved (newest first) by ``os_utils.resolve_squeak_spectrogram_store_path``; the name never matches the consolidated store's ``spectrograms_*.h5`` pattern. Sessions come from ``--root-directories`` and/or ``--session-lists`` (``*.txt`` files, one session root per line) and are processed in ``n_workers`` parallel processes; a session that fails is reported (and recorded in the ``failed_sessions`` attr) and left out.

* **Rows**: per session, the rows whose ``call_class`` is ``squeak`` or ``both`` and whose ``noise`` is not true, the rows *Embed squeaks in a QLVM torus* considers, so every row that carries ``qlvm_squeak1`` / ``qlvm_squeak2`` is in the store, and the explorer's squeak-class filter (squeak / both / squeak + both) needs no rebuild.
* **Spectrogram**: rebuilt from the unfiltered ``audio/hpss`` wavs over the row's squeak crop window -- the segment, widened where needed to hold the squeak envelope plus two frames of context, the window *Embed squeaks in a QLVM torus* reads -- dropping the metadata-excluded channels (``exclude_metadata_audio_channels``) exactly as *Detect squeaks* does, with the sonic front end's STFT: Blackman-Harris, ``nperseg`` 2048 (8.19 ms, 122 Hz bins; the main lobe of about 1 kHz separates harmonics 3 kHz apart), hop 512 (2.048 ms frames, centred), so frame ``t`` sits at ``audio_start_s + t * 0.002048`` s like the embedding's frames. The linear bins are mapped onto ``n_frequency_bins`` (128) log-spaced bins with geometric edges from ``min_freq`` (2 kHz) to ``max_freq`` (125 kHz): a log bin averages the power of the linear bins whose centres fall inside it, and a log bin narrower than the 122 Hz linear spacing (below about 7.6 kHz at 128 bins) takes the power interpolated at its geometric centre. The power is converted to absolute dB (``ref`` 1.0, no ``top_db`` clamp; silence at -100 dB) and averaged across channels with audio-variance weights (the rule of *Generate spectrograms*).
* **Time extent**: every call sits from column 0 of a fixed ``window_frames`` window (256 frames, 524 ms), its frame count in ``durations``, so a thumbnail padded to the window shows its true duration, as in the consolidated store. An audio window longer than the display window keeps the ``window_frames`` frames centred on its squeak crop (the frames inside ``squeak_start`` .. ``squeak_end`` plus two, the embedding's crop), clipped to the audio window; ``window_first`` records the first kept frame and ``audio_start_s`` the session time of the audio window's frame 0.
* **Storage**: dB values clipped to ``[db_floor, db_ceil]`` (-100 to 60 dB) and quantized to uint8 (0.63 dB per step; code 0 = ``db_floor``, so the padding reads as silence), gzip-compressed in one chunk per call.

.. code-block:: text

    squeak_spectrograms_128logbins_<S>sessions_<N>squeaks_<ts>.h5
    ├── frequency_bins (F,) float64                      # geometric centres of the log bins (Hz)
    ├── frequency_bin_edges (F + 1,) float64             # their edges (Hz)
    ├── spectrogram/<session>/spectrograms (n, F, W) uint8   # W = window_frames; dB = db_floor + code * (db_ceil - db_floor) / 255
    ├── spectrogram/<session>/row_index (n,) int64       # 0-based usv_summary.csv rows, ascending
    ├── spectrogram/<session>/durations (n,) int32       # frames of the call in the window (0: shorter than one STFT window)
    ├── spectrogram/<session>/n_frames (n,) int32        # native frames of the whole segment
    ├── spectrogram/<session>/window_first (n,) int32    # first kept frame (0 unless the segment is longer than W)
    └── sessions                                         # session_id, root_directory, usv_summary_sha256,
                                                         # n_summary_rows, n_squeaks

The root attrs record ``created_by``, ``created_date`` (UTC), ``git_commit``, ``n_sessions``, ``n_squeaks``, the settings block (``settings``, JSON), the STFT (``stft``, JSON), ``frame_dt_s``, ``window_frames``, ``db_floor`` / ``db_ceil``, the quantization rule and ``failed_sessions``. Sessions without squeaks are listed in ``sessions`` but get no ``spectrogram`` group. Rebuild the store after *Detect squeaks* or *Detect noise* change a session's rows.

The *Compute USV features* and *Infer QLVM latents* steps add columns to *usv_summary.csv* in place. *Compute USV features* adds:

* **mean_freq_hz** : energy-weighted mean frequency of the USV (Hz)
* **peak_freq_hz** : frequency of peak energy (Hz)
* **freq_bandwidth_hz** : spectral bandwidth between the low/high cumulative-energy edges (Hz)
* **mean_amplitude** : mean spectrogram amplitude over the USV (a.u., on the call's own min-max normalized [0, 1] spectrogram, so relative to the call's peak)
* **max_amplitude** : maximum spectrogram amplitude over the USV (a.u., same normalized scale; close to 1 whenever the mask covers the call's peak)
* **loudness_db** : the call's ABSOLUTE loudness (dB): the image-level level of ``compute_usv_loudness`` — per eligible microphone, the mean over the call's SAM mask of ``10 log10`` of the absolute band power re-read from the session's ``hpss_filtered`` audio, combined across microphones with the spectrogram generator's variance weights. It is the value the QLVM loudness-conditional model is trained and decoded on (``infer-qlvm-latents`` reads it from here), and it reproduces the model package's corpus values. It needs the session's ``hpss_filtered`` audio and SAM masks; a call without a mask gets null
* **spectral_entropy** : spectral entropy of the USV (nats)
* **mask_number** : number of SAM mask instances of the USV

.. parsed-literal::

    ┌────────┬───┬──────────────┬──────────────┬───────────────────┬────────────────┬───────────────┬──────────────────┐
    │ usv_id ┆ … ┆ mean_freq_hz ┆ peak_freq_hz ┆ freq_bandwidth_hz ┆ mean_amplitude ┆ max_amplitude ┆ spectral_entropy │
    │ ---    ┆   ┆ ---          ┆ ---          ┆ ---               ┆ ---            ┆ ---           ┆ ---              │
    │ i64    ┆   ┆ f64          ┆ f64          ┆ f64               ┆ f64            ┆ f64           ┆ f64              │
    ╞════════╪═══╪══════════════╪══════════════╪═══════════════════╪════════════════╪═══════════════╪══════════════════╡
    │ 000000 ┆ … ┆ 68421.3      ┆ 71250.0      ┆ 24180.5           ┆ 0.182          ┆ 0.94          ┆ 0.61             │
    │ 000001 ┆ … ┆ 72980.1      ┆ 75000.0      ┆ 18640.2           ┆ 0.211          ┆ 0.88          ┆ 0.55             │
    │ …      ┆ … ┆ …            ┆ …            ┆ …                 ┆ …              ┆ …             ┆ …                │
    └────────┴───┴──────────────┴──────────────┴───────────────────┴────────────────┴───────────────┴──────────────────┘

*Detect noise* and *Detect squeaks* add their own columns, placed before the acoustic features:

* **noise** : ``true`` / ``false`` in every row -- the segment holds no vocalization at all
* **noise_probability** : the ensemble's probability in every row, so an analysis can re-threshold without re-running the step
* **call_class** / **p_usv** / **p_squeak** / **p_both** / **squeak_spans** / **squeak_start** / **squeak_end** / **n_squeaks** : see *Detect squeaks* above

*Infer QLVM latents* adds, with the production ``model_cells`` mapping (see *Several QLVM models in one run* below):

* **qlvm1** / **qlvm2** : the two torus (latent) coordinates of the regular (phase 6) model
* **qlvm_category** : its FINE cluster label (vocal category; 15 clusters in the v3 package)
* **qlvm_supercategory** : its COARSE cluster label (9 clusters in the v3 package)
* **qlvm_dur1** / **qlvm_dur2** / **qlvm_dur_category** / **qlvm_dur_supercategory**, **qlvm_mf1** / **qlvm_mf2** / **qlvm_mf_category** / **qlvm_mf_supercategory**, **qlvm_bw1** / **qlvm_bw2** / **qlvm_bw_category** / **qlvm_bw_supercategory**, **qlvm_loud1** / **qlvm_loud2** / **qlvm_loud_category** / **qlvm_loud_supercategory** : the coordinates and FINE and COARSE cluster labels of the duration, mean-frequency, bandwidth and loudness conditional models

Every label is an integer ``1 … k`` numbered by cluster size (``1`` = the largest cluster, as the package numbers them; the grids label every pixel, so there is no background / noise label ``0``), read off the cell's ``label_grid.npy`` of that level at the pixel of the call's written coordinates, and null where the call has no coordinates. Which levels each model writes is ``model_cell_label_levels``. No model-provenance column is written: summaries embedded by the retired single-model run (``model_cell_directory``, or decoder weights with reference arrays) may still carry a legacy **qlvm_model** column (the cell or weights path their ``qlvm_*`` columns came from), which the next *Infer QLVM latents* run removes. Labels from different models share the column names but not their meaning (a v2 cell has 8–18 fine and at most 9 coarse clusters, numbered by size), so embed every session of an analysis with the same cells.

*Embed squeaks in a QLVM torus* (``infer-qlvm-squeak-latents``) adds, last:

* **qlvm_squeak1** / **qlvm_squeak2** : the torus coordinates of each ``squeak`` / ``both`` segment that is not noise in a squeak (BBV) QLVM cell, null on every other row; see *Embed squeaks in a QLVM torus* above

Switching sessions to a QLVM model package cell also changes what these readers see:

* the QLVM visualizations (the sequence embedding map, the torus-traversal video, the thumbnails' cluster centres and the embedding explorer's boundaries) draw the map chosen by ``shared_resources.qlvm_map`` (``visualizations_settings.json``; one of ``os_utils.QLVM_MAPS``, ``qlvm`` by default) and read its ``<spectrograms_dir>/qlvm_v3/<qlvm_map>/arrays_{fine,coarse}.npz`` (``os_utils.QLVM_REFERENCE_ARRAYS_DIRECTORY_NAME``), the map's production cell clustering written by ``export-qlvm-reference-arrays --model-cell-directory <QLVM_MODEL_PACKAGE_ROOT>/<cell> --output-directory <spectrograms_dir>/qlvm_v3/<qlvm_map>`` (once per map; the manifold filter atlas always reads the regular map, ``qlvm_v3/qlvm/``, since it needs an unconditional decoder); the old in-house model's ``<spectrograms_dir>/qlvm/`` arrays and ``qlvm_clusters_*.h5`` are no longer read. Sessions embedded with another cell need that cell's arrays under a separate ``spectrograms_dir``
* the torus geodesic metrics decode with ``vocal_features.usv_manifold_geodesic_metrics.decoder_model_cell_directory`` (``modeling_settings.json``; shipped as the v3 phase 6 regular cell the production ``qlvm1`` / ``qlvm2`` come from, read without torch), the only decoder source (the legacy decoder ``.npz`` setting ``decoder_weights_npz_path`` is retired)
* the pooled-embedding parquet caches record a fingerprint of the summaries they were built from (each summary's path, size and modification time) and rebuild themselves when a summary changes; the cohort cache is ``<spectrograms_dir>/embeddings/pooled_embeddings_qlvmv3.parquet`` (``os_utils.POOLED_EMBEDDINGS_CACHE_NAME``), and it carries every map's coordinates and labels; the consolidated store's ``qlvm/<session>/<map>`` coordinates and labels keep the columns as they were when built, so rebuild it after re-embedding
* the tuning figures draw the ``qlvm_category`` / ``qlvm_supercategory`` region maps and size their per-category axes from the fine and coarse ``label_grid.npy`` of the production regular cell (``QLVM_PRODUCTION_MODEL_CELLS['qlvm']`` under ``QLVM_MODEL_PACKAGE_ROOT``), so sessions labelled by another cell do not match those maps; with the package unreachable the QLVM maps are placeholders and a message says why
* category ids chosen by meaning (e.g. the coactivity notebook's ``GROUP_A_IDS`` / ``GROUP_B_IDS``, set for the v3 regular model's coarse clusters: complex ``[4, 7, 8]``, simple ``[1, 6]``) have to be re-chosen for another cell: a cell numbers its clusters by size

.. parsed-literal::

    ┌────────┬───┬───────────┬───────────┬───────────────┬────────────────────┐
    │ usv_id ┆ … ┆ qlvm1 ┆ qlvm2 ┆ qlvm_category ┆ qlvm_supercategory │
    │ ---    ┆   ┆ ---       ┆ ---       ┆ ---           ┆ ---                │
    │ i64    ┆   ┆ f64       ┆ f64       ┆ i64           ┆ i64                │
    ╞════════╪═══╪═══════════╪═══════════╪═══════════════╪════════════════════╡
    │ 000000 ┆ … ┆ 0.4123    ┆ 0.8871    ┆ 7             ┆ 3                  │
    │ 000001 ┆ … ┆ 0.1902    ┆ 0.3320    ┆ 2             ┆ 1                  │
    │ …      ┆ … ┆ …         ┆ …         ┆ …             ┆ …                  │
    └────────┴───┴───────────┴───────────┴───────────────┴────────────────────┘

These columns are comparable across every session embedded into the same QLVM model, and are consumed by the categorical USV-tuning analysis in :ref:`Analyze <Analyze>` (*Compute neuronal tuning curves*). When a mask is present for a call, the acoustic features are computed over the true SAM mask region; otherwise they fall back to the signal time-window.

The */usv-playpen/_parameter_settings/processing_settings.json* file contains the settings for these steps, partially modifiable in the GUI but fully modifiable manually. The SAM2 / YOLO / squeak / noise model paths all derive from a single ``spectrograms_root`` (GUI: *Spectrogram models directory*): set that one directory and the paths below are filled as ``<root>/sam/...``, ``<root>/squeak/...`` and ``<root>/noise/...``. Set any individual path explicitly (in the JSON or via a CLI flag) to override its derived default; ``generate_masks.sam2_model_cfg`` is a config name, not a path, so it is never derived. The QLVM model is no longer derived from ``<root>/qlvm`` (the old in-house decoder and its watershed arrays, which would re-embed sessions on another torus and write ``qlvm_category`` / ``qlvm_supercategory`` back): when ``infer_qlvm_latents`` names no model (``model_cells`` empty), ``model_cells`` is filled with the production mapping of the v3 package ``/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3`` (``qlvm``, ``qlvm_dur``, ``qlvm_mf``, ``qlvm_bw``, ``qlvm_loud``; see *Several QLVM models in one run*) and ``masking_type`` is set to ``"none"``, the preprocessing those cells were trained with (the contract check would refuse the shipped ``"sam"``). The package root and the mapping are the constants ``QLVM_MODEL_PACKAGE_ROOT`` / ``QLVM_PRODUCTION_MODEL_CELLS`` in ``os_utils``, not settings, because the GUI and CLI re-key every experimenter name in the processing settings (a path under another experimenter's directory would be rewritten to the active experimenter's directory). An explicitly configured ``model_cells`` is left untouched, ``masking_type`` included; with no ``spectrograms_root`` and an empty ``model_cells`` the run stops, since model package cells are the only models *Infer QLVM latents* embeds with (the single-model keys ``model_cell_directory``, ``weights_npz_path``, ``reference_arrays_fine_npz_path`` / ``reference_arrays_coarse_npz_path`` and the lattice keys ``lattice_type``, ``n_points``, ``korobov_a``, ``fib_m`` are retired). The squeak QLVM cell (``infer_qlvm_squeak_latents.model_cell_directory``) is derived the same way: when it is empty it is filled with the production squeak cell ``QLVM_SQUEAK_PACKAGE_ROOT/QLVM_SQUEAK_PRODUCTION_CELL`` (``os_utils`` constants: ``.../qlvm_models_latest/phase3_BBVs_qlvm/natural_session_N11000_nomask``), for the same reason a constant rather than a setting; a configured cell is left alone. Name another with ``--model-cell-directory``, which is applied after the experimenter re-keying; a path under another experimenter's directory written into the JSON would be re-keyed like any other.

.. code-block:: json

    "spectrograms_root": "/mnt/falkner/Bartul/spectrograms"

*Generate spectrograms* (``generate_spectrograms``):

* **num_freq_bins** : number of spectrogram frequency bins (output height)
* **num_time_bins** : number of spectrogram time bins (output width)
* **nperseg** : STFT window length / n_fft (samples)
* **noverlap** : legacy scipy-style STFT overlap kept for parity with the QLVM training config; it is **not** used to compute the spectrogram (the hop is governed by ``hop_length``)
* **min_freq** : lower frequency cutoff (Hz)
* **max_freq** : upper frequency cutoff (Hz)
* **hop_length** : STFT hop length (samples; defaults to ``nperseg // 4`` when null)
* **window** : STFT window function
* **offset** : time padding added before each USV onset and after each offset (seconds)
* **normalize** : whether to min-max normalize each spectrogram to [0, 1]

.. code-block:: json

    "generate_spectrograms": {
        "num_freq_bins": 128,
        "num_time_bins": 128,
        "nperseg": 2048,
        "noverlap": 1792,
        "min_freq": 30000,
        "max_freq": 120000,
        "hop_length": 512,
        "window": "blackmanharris",
        "offset": 0.0,
        "normalize": true
      }

*Generate masks* (``generate_masks``):

* **method** : mask-generation method (``boxprompt`` = box-prompted SAM2 segmentation)
* **detector** : box detector backend (``yolo`` learned detector or ``cc`` connected-component baseline)
* **sam2_model_dir** : SAM2 model directory (config/checkpoint resolve against it; the step changes into it)
* **sam2_model_cfg** : SAM2 config name/path (resolved from the SAM2 install's Hydra search path)
* **sam2_model_path** : SAM2 checkpoint path
* **yolo_weights** : trained YOLO ``best.pt`` weights path
* **yolo_conf** : YOLO confidence threshold (lower → more recall)
* **yolo_iou** : YOLO NMS IoU (raise to keep stacked calls)
* **yolo_imgsz** : YOLO inference image size (px)
* **mask_cmap** : colormap used to render each spectrogram to RGB before detection
* **duration_min** : minimum USV duration to segment, in SPECTROGRAM TIME BINS (not milliseconds); shorter rows are skipped without any detection attempt and end up with ``mask_number`` 0. At the shipped spectrogram settings the grid runs at ~0.5 bins/ms, so a value of 10 silently excluded every USV under ~20 ms — well above DAS's own 15 ms ``segment_minlen``, so genuine calls were discarded. Set to 5 (~10 ms)
* **batch_size** : number of spectrograms per SAM2 batch
* **multimask_output** : whether SAM2 returns multiple candidate masks per box (the best is kept)
* **iou_floor** : SAM2 self-predicted-IoU threshold used by ``drop_below_iou``. Set to 0.30 (not 0.70): the mask population is strongly good-quality (median 0.925) and its low tail is continuous, so only the 0-0.30 band is unambiguously broken
* **drop_below_iou** : if true, drop masks whose SAM2 self-predicted IoU is below ``iou_floor`` (else keep them). Enabled, but paired with a low ``iou_floor``: measured over 1,209 masks the predicted-IoU distribution has median 0.925 with no gap in its low tail, so the original 0.70 floor cut through a continuum -- 28% of what it removed sat within 0.05 of the line. A 0.30 floor removes only the ~18% of low-IoU masks SAM2 rates as outright wrong and leaves the grey zone intact
* **split_disconnected** : split a mask with disconnected components into separate instances
* **max_iters** : maximum SAM2 prompt-refinement iterations per box
* **merge_instances** : merge overlapping per-box mask instances
* **merge_iou** : IoU above which two instances are merged
* **merge_containment** : containment fraction above which one instance is absorbed into another
* **mask_intensity_floor** : minimum normalized spectrogram intensity for a pixel to remain in a mask
* **tiny_mask_floor_px** : drop masks smaller than this many pixels
* **min_box_area** : drop detector boxes smaller than this area (px²; 0 disables)

.. code-block:: json

    "generate_masks": {
        "method": "boxprompt",
        "detector": "yolo",
        "sam2_model_dir": "",
        "sam2_model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2_model_path": "",
        "yolo_weights": "",
        "yolo_conf": 0.25,
        "yolo_iou": 0.7,
        "yolo_imgsz": 128,
        "mask_cmap": "viridis",
        "duration_min": 5,
        "batch_size": 12,
        "multimask_output": true,
        "iou_floor": 0.3,
        "drop_below_iou": true,
        "split_disconnected": true,
        "max_iters": 1,
        "merge_instances": true,
        "merge_iou": 0.5,
        "merge_containment": 0.8,
        "mask_intensity_floor": 0.0,
        "tiny_mask_floor_px": 12,
        "min_box_area": 0
      }

When left empty (the default) the SAM2/YOLO paths are derived from ``spectrograms_root`` above; whether derived or set explicitly, they are stored in the canonical ``/mnt/falkner/...`` lab-share form and translated to the host's mount root (e.g. ``/Volumes/falkner`` on macOS) via ``configure_path``, the same handling as the DAS / Vocalocator model paths. ``sam2_model_cfg`` is a config name resolved inside the SAM2 install, not a mount path. Both ``sam2`` and ``ultralytics`` (usv-playpen core dependencies) must be installed.

*Compute USV features* (``compute_usv_acoustic_features``):

* **low_energy_frac** : lower edge of the cumulative-energy band used for spectral bandwidth
* **high_energy_frac** : upper edge of the cumulative-energy band used for spectral bandwidth

.. code-block:: json

    "compute_usv_acoustic_features": {
        "low_energy_frac": 0.05,
        "high_energy_frac": 0.95
      }

*Detect noise* (``detect_usv_noise``):

* **noise_model_path** : path to the noise model bundle (``.pt``); left empty it is derived from *Spectrogram models directory* as ``noise/noise_timemil_ens5_n4680_20260926.pt``. The bundle also fixes the decision (exclusion cut-off 0.14; see *Detect noise* above), so there is no threshold setting
* **exclude_metadata_audio_channels** : drop channels the session metadata marks as excluded from the spectrogram average
* **batch_size** : segments per forward pass at the typical call length; a batch is budgeted at ``batch_size`` x 128 frame slots and padded to its longest call, so one long call never inflates a batch (raise it on a GPU with memory to spare, lower it when several sessions run in parallel)

.. code-block:: json

    "detect_usv_noise": {
        "noise_model_path": "",
        "exclude_metadata_audio_channels": true,
        "batch_size": 64
      }

*Train noise model* (``train_noise_model``, command line only; see *Noise model* below):

* **calibration_path** : JSON holding the ``calibration`` table (rows of ``threshold``, ``precision``, ``recall``, ...) and the ``decision`` block (``exclude_at_or_above``, ``noise_at_or_above``, ``held_out_precision``, ``held_out_recall``, ``real_calls_excluded_per_10000`` and any provenance) the bundle carries; ``calibration_source`` and ``calibration_population`` are copied too when present. Training does not measure these, so they must describe this labels set and recipe
* **exclude_metadata_audio_channels** : drop channels the session metadata marks as excluded from the spectrogram average; keep it as *Detect noise* runs
* **n_workers** : sessions whose audio is read concurrently (threads) while the inputs are built; reading short windows from 24 wavs on the lab share is latency-bound (~14 s per session serially)
* **seeds** : one seed per ensemble member (the production ensemble's five)
* **epochs** : training epochs per member
* **batch_size** : segments per training batch
* **learning_rate** : Adam learning rate, cosine-annealed to zero over the run
* **weight_decay** : Adam weight decay
* **label_smoothing** : targets are smoothed towards 0.5 by this amount

.. code-block:: json

    "train_noise_model": {
        "calibration_path": "",
        "exclude_metadata_audio_channels": true,
        "n_workers": 16,
        "seeds": [20269914, 20269915, 20269916, 20269917, 20269918],
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05
      }

*Infer QLVM latents* (``infer_qlvm_latents``):

The USV maps hold ultrasonic calls only. A segment whose ``call_class`` (written by *Detect squeaks*) is ``squeak`` -- a squeak with no ultrasonic call in it -- is skipped by every cell, on both routes (not embedded, and left out of a package's own coordinates), and gets null coordinates and null labels; squeaks have their own map (*Embed squeaks in a QLVM torus*). A ``both`` segment (a squeak and a USV together) is embedded like any other call, and rows with a null ``call_class`` (noise, or too short to score) are treated as before. The rule needs ``call_class``, so a summary without it stops the run: run *Detect noise* and *Detect squeaks* first.

* **model_cells** : column prefix → model package cell, e.g. ``{"qlvm_dur": ".../v3/phase11_cond_duration_floor/natural_5strata_N29000_unmasked_floor"}`` (default ``{}``: filled with the production mapping when ``spectrograms_root`` is set, see above; left empty, the run stops). One run places the session on the torus of **every** listed cell and writes, per prefix ``P``, the two float columns ``P1`` / ``P2`` (torus coordinates in ``[0, 1)``; nulls for calls a cell does not place) and the integer cluster-label columns of the prefix's ``model_cell_label_levels`` (by default both levels: ``qlvm_category`` and ``qlvm_supercategory`` for ``qlvm``, ``P_category`` and ``P_supercategory`` for every other prefix); no model column is written. The legacy ``qlvm_model`` column and the earlier ``P1`` / ``P2`` and label columns (both levels) of the listed prefixes are removed first (a ``qlvm1`` / ``qlvm2`` pair and its labels are kept unless ``qlvm`` is a listed prefix). A prefix must be a Python identifier (letters, digits, underscores, not starting with a digit), may appear once, and ``P1`` / ``P2`` may not be another column of the USV summary (the production coordinate and label columns are allowed). Each cell brings everything model-specific: the decoder from its ``checkpoint.tar`` (read without torch; the legacy or ReLU head is read from the weights), the input normalization and duration window from its ``training_contract.json``, the lattice from the contract's Fibonacci ``embedding_fib_m`` (``24``, 46,368 points; built in float32 exactly as the torch driver that embedded the package corpus built it, since the exact lattice, as JAX holds it in float32, sits up to ~1e-3 away and moved 8 % of 9,390 corpus calls by more than 1e-3), and the categories from its fine and coarse ``label_grid.npy`` (``inference/clusters_<level>/`` in v3, ``cluster/<level>/`` in v2 / v2.1). Each cell is loaded once per run and checked against ``masking_type``, ``target_shape``, ``time_stretch``, ``latent_dim`` and ``length_threshold``, so all listed cells must share them (``masking_type`` ``"sam"`` for phase 9 cells, ``"none"`` for phase 6, 10 and 11 cells). Package cells were trained on min-maxed spectrograms (phase 6 and 10 cells also with a 0.2 loudness floor), which is applied automatically. Conditional cells decode each USV at a value derived from its own conditioning value — the normalized duration; the mean frequency of its SAM-masked spectrogram, which needs the session's masks even though the input is unmasked; and, in phase 11 (``qlvm_models_latest/v3``), ``clip(freq_bandwidth_hz / 90000, 0, 1)`` from the summary (run *Generate USV acoustic features* first) or the loudness mapped through the contract's ``db_range`` (the masked, variance-weighted image-level dB, measured from the session's ``hpss_filtered`` audio with the spectrogram generator's own slice and STFT; about 40–90 ms per USV on one core); a cell ``train-qlvm`` trained on spectral entropy reads the summary's ``spectral_entropy`` (a summary without the column stops the run), scales it by the contract's ``entropy_min`` / ``entropy_max`` (the training split's range) and clamps it to ``[0, 1]``, the log counting the USVs outside that range. USVs without a value get null columns. Phase 10 cells (duration, mean frequency) decode at the frozen corpus bin mean of ``condition_bins.npz`` (the lattice is decoded once per distinct bin mean, up to 32 per session). Phase 11 cells (duration, mean frequency, bandwidth, loudness, and spectral entropy cells trained with ``train-qlvm``) follow the contract's ``condition.decode``: ``"exact"`` (duration, bandwidth) is the value clamped to the cell's training range ``[train_c_min, train_c_max]``, ``"grid"`` (mean frequency, loudness, spectral entropy) the nearest point of the cell's ``decode_grid`` (0.0025 apart); the log counts the USVs outside the range each rule applies (the training range for ``"exact"``, the ends of the decode grid for ``"grid"``). This is how the package embedded every corpus call, and the lattice is decoded once per distinct value (about 100–250 per ~900-USV session). See *Several QLVM models in one run* below for the production mapping and how each model's coordinates are obtained
* **model_cell_label_levels** : column prefix → list of cluster-label levels among ``"fine"`` and ``"coarse"`` a ``model_cells`` run writes for that prefix (default ``{}``: ``["fine", "coarse"]`` for every prefix). Column names: for ``qlvm``, fine → ``qlvm_category`` and coarse → ``qlvm_supercategory``; for any other prefix ``P``, fine → ``P_category`` and coarse → ``P_supercategory``. A listed prefix replaces its default (``[]`` writes coordinates only), a prefix not in ``model_cells``, an unknown or repeated level, or a value that is not a list stops the run before any cell is loaded, as does a label column that would overwrite another summary column
* **prefer_package_values** : take a corpus session's coordinates from each cell's own corpus embedding when the session's spectrogram H5 is verifiably the one the package was built from, and infer them otherwise (default ``true``); ``false`` infers every session with every cell
* **latent_dim** : torus latent dimensionality (must match every cell's training contract)
* **time_stretch** : whether to time-stretch spectrograms before embedding (must match every cell's training contract)
* **masking_type** : ``"none"`` (default) embeds raw spectrograms, as the phase 6, 10 and 11 cells (the production cells) were trained; ``"sam"`` masks each spectrogram by the union of its SAM regions before embedding, matching a decoder trained on masked spectrograms (phase 9 cells). It must match every cell's training contract. USVs without a mask instance get null columns rather than being embedded unmasked when the training contract has ``require_mask`` true (every v2 and v3 package cell) or the cell conditions on mean frequency or loudness (the spectral entropy is read from the summary and needs no mask); a ``"sam"`` decoder without ``require_mask`` (whose training set kept such calls under an all-ones mask) still embeds them. A session H5 without a ``mask/<session>`` group raises for any of these decoders
* **target_shape** : output spectrogram ``(freq, time)`` shape the embedder resizes to before inference; must match the ``target_shape`` used by *Build QLVM training set* (default ``[128, 128]``)
* **length_threshold** : embed only USVs with ``0 < duration < length_threshold`` (time bins), the window *Build QLVM training set* keeps; longer USVs get null columns. ``null`` (default) takes the value from each cell's training contract. A value set here must equal every contract's
* **lattice_batch_size** : lattice points decoded and scored per block (default ``4096``); each block holds its decoded images and their two logs, about ``3 * 16384 * 4`` bytes per point at ``128x128``
* **data_batch_size** : spectrograms whose lattice posteriors are computed together (default ``8192``); their likelihood matrix takes ``data_batch_size * <lattice points> * 4`` bytes (46,368 points for a package cell), and the lattice is decoded once per such batch

.. code-block:: json

    "infer_qlvm_latents": {
        "model_cells": {},
        "model_cell_label_levels": {},
        "prefer_package_values": true,
        "latent_dim": 2,
        "time_stretch": false,
        "masking_type": "sam",
        "target_shape": [128, 128],
        "length_threshold": null,
        "lattice_batch_size": 4096,
        "data_batch_size": 8192
      }

*Several QLVM models in one run* (``model_cells``). The production setting places every session on the regular (phase 6) torus and the four phase 11 conditional tori of ``qlvm_models_latest/v3``, all of the ``natural_5strata_N29000_unmasked_floor`` design, with ``masking_type`` ``"none"`` (the shipped setting stays ``{}``; with ``spectrograms_root`` set and no cells configured, ``derive_spectrogram_model_paths`` fills in exactly this mapping and ``masking_type`` ``"none"``; otherwise the paths are set per run or by a backfill):

.. code-block:: json

    "model_cells": {
        "qlvm": "/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3/phase6_USVs_unmasked_floor/natural_5strata_N29000_unmasked_floor",
        "qlvm_dur": "/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3/phase11_cond_duration_floor/natural_5strata_N29000_unmasked_floor",
        "qlvm_mf": "/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3/phase11_cond_mean_freq_floor/natural_5strata_N29000_unmasked_floor",
        "qlvm_bw": "/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3/phase11_cond_bandwidth_floor/natural_5strata_N29000_unmasked_floor",
        "qlvm_loud": "/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/v3/phase11_cond_loudness_floor/natural_5strata_N29000_unmasked_floor"
    }

This writes ``qlvm1`` / ``qlvm2`` / ``qlvm_category`` / ``qlvm_supercategory`` (regular), ``qlvm_dur1`` / ``qlvm_dur2`` / ``qlvm_dur_category`` / ``qlvm_dur_supercategory`` (duration), ``qlvm_mf1`` / ``qlvm_mf2`` / ``qlvm_mf_category`` / ``qlvm_mf_supercategory`` (mean frequency), ``qlvm_bw1`` / ``qlvm_bw2`` / ``qlvm_bw_category`` / ``qlvm_bw_supercategory`` (bandwidth) and ``qlvm_loud1`` / ``qlvm_loud2`` / ``qlvm_loud_category`` / ``qlvm_loud_supercategory`` (loudness), in that order after the acoustic features; the bandwidth model needs ``freq_bandwidth_hz`` in the summary and the loudness and mean-frequency models the session's SAM masks. Model provenance is not written to the summary.

For each model and session the coordinates come from one of two routes, and the log names the route and the reason (e.g. ``qlvm_dur (v3/phase11_cond_duration_floor/natural_5strata_N29000_unmasked_floor): package values (sha256 + 907 rows verified).``). The *package route* copies the package's own embedding of the call: the cell's ``posterior_cache.npz`` ``torus_weighted`` (turned into ``[0, 1)`` coordinates) joined to the summary row through ``recon_mse_breakdown.npz``'s ``spec_id`` (``<session>_<H5 row>``). Because a ``spec_id`` names a call only by its row in the session's spectrogram H5, it is taken only when all of these hold:

* ``prefer_package_values`` is ``true`` and a ``SESSION_H5_BASELINE.tsv`` is found in the cell directory or one of its parents (the package root);
* the session is listed in that baseline (it is one of the package's corpus sessions);
* the SHA-256 of the session's ``audio/spectrograms/<session>_spectrograms.h5`` equals the baseline's (the H5 was not rebuilt since; it is hashed once per session, not once per model);
* ``usv_summary.csv`` has as many rows as the H5;
* the cell holds rows of the session, all inside the H5, and each row's ``durations`` and ``mask_counts`` in ``recon_mse_breakdown.npz`` equal the H5's (``spectrogram/<session>/durations`` and the number of ``mask/<session>/spectrogram_index`` entries naming the row).

Otherwise the *inference route* embeds the session with that cell (the log says why: ``session not in the package corpus``, ``spectrogram H5 changed since the package: sha256 mismatch``, a row-count or per-row disagreement, no baseline above the cell, or ``prefer_package_values is false``). Both routes place the same calls of a verified corpus session (those inside the duration window with a SAM mask and, for conditional cells, a conditioning value), and inferred coordinates differ from the package's only where the package's TF32 embedding and the float32 embedding here round a spread-out posterior differently (see the phase 11 check above), so the columns are comparable across sessions whichever route filled them. Both routes label a call the same way: the cell's ``label_grid.npy`` at the pixel of the coordinates written to the summary. On the package's own posterior-mean coordinates this pixel rule reproduces the package's ``cluster_labels.csv``, so a package-route label equals the package's (on the whole v3 corpus, 445,742 calls, every fine label of the five production cells and every coarse label of the regular cell agreed except one ``qlvm_dur`` call, ``20230805_112702`` row 338, whose ``x * 200`` is 190.999997: a coordinate within float32 rounding of a pixel edge).

Train spectrogram-pipeline models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two per-session inference steps lean on learned models: *Generate (spectrogram) masks* uses a detector to find each USV, and *Infer QLVM latents* uses a decoder to embed it. Both are trained **once on a representative cohort** and then reused for every future session — so this is a setup / maintenance step, not part of routine per-experiment processing. Retrain only when something changes materially: a new or substantially expanded cohort, a different spectrogram representation, or revised category definitions.

Both train cross-session via CLI / cluster commands only (no GUI buttons): each is a two-command chain over a comma-separated list of session ``--root-directories`` writing standalone artifacts to ``--output-directory``. **SAM2 is used pretrained — it is not trained here.** Full flags for CLI can be found :ref:`here <usv-pipeline-cli>`.

QLVM training sets
^^^^^^^^^^^^^^^^^^

Two commands build the ``.npz`` sets ``train-qlvm`` trains on, in the format of the QLVM model packages' sets: ``build-qlvm-training-set`` for USVs and ``build-qlvm-squeak-training-set`` for squeaks. They are the in-house ports of the reference builders (MMMmB ``build_masked_usvs.py``, ``build_unmasked_usvs_floor.py`` and ``build_bbv_dataset.py``, together with the session-typing, quota, stratification and split helpers they imported from a local usv-playpen fork, which are reconstructed here), and with the same sessions, inputs and seed they draw the same rows, hold out the same sessions and write the same spectrograms. Both write ``train_data.npz`` + ``val_data.npz`` (and ``full_data.npz`` after them with ``--full-dataset``, which takes every eligible row instead of drawing) and ``metadata.npz`` (every setting, the per-type report and the train / validation session lists).

**USVs** (``build_qlvm_training_set`` settings block). For every session its ``audio/spectrograms/<session>_spectrograms.h5`` is read (durations and the SAM instance count of every row, from ``mask/<session>/spectrogram_index``), its type comes from the ``Subjects[].sex`` entries of its metadata YAML (``MF``, ``FF``, ``MM``, ``lone_male``, ``lone_female``, ``other``; ``unknown`` if unreadable), and the rows that are not ultrasonic calls alone are marked: with ``exclude_squeaks`` (on by default) every row whose *usv_summary.csv* ``call_class`` is ``squeak`` or ``both``, so a set holds ``call_class`` ``usv`` rows only (a null ``call_class`` -- noise, or a segment too short to score -- is not a squeak: noise is left out by ``exclude_noise``, the unscorable segments by the duration gate), and with ``exclude_noise`` every row flagged as noise. With ``strict_squeak_exclusion`` (off by default; needs ``exclude_squeaks``) the squeak rows come instead from the reference squeak index at ``reference_squeak_index_path`` (the shipped one: ``/mnt/falkner/Dexter/vocal_beh/models/bbv_classifier/indices/bbv_segment_index.csv``, keyed by ``session_id`` and ``seg_index``, the summary row), by its strict rule (``is_bbv`` true, or ``n_bouts_min3`` >= 1): the index is read once per build, and it replaces the ``call_class`` rule; a session it does not list, or a row it lists past the end of the session, stops the build. ``exclude_noise`` is on by default, so new sets leave out the rows *Detect noise* flags; the shipped v3 sets were drawn without it, so reproducing them needs ``--no-exclude-noise`` (and the reference squeak index's strict exclusions, see below). A row is eligible when ``0 < duration < length_threshold`` (128 time bins), it is not marked and, with ``require_mask``, it has at least one SAM mask instance.

* **Budgets.** ``session_type_targets`` maps each type to its number of rows; ``null`` takes the type whole and a type not listed is left out (the shipped sets: ``{"MF": N, "FF": N, "MM": null, "lone_male": null}``). A type's budget is split into per-session quotas by integer capped-even water-filling: every session gets ``min(capacity, L)`` at the largest level ``L`` that does not overshoot, and the remainder goes one row at a time to sessions with room left, in the order of a permutation drawn from ``random_state`` over the type's sessions sorted by id. The total is therefore met exactly, and no session dominates a type.
* **Within-session draw** (``draw_mode``). ``natural`` draws the quota uniformly over the session's eligible rows, keeping its own mask-count mix. ``uniform`` splits it equally across the session's mask-count strata (``mask_count_bin_edges`` are the inclusive lower bounds of the strata, the last open-ended: ``1,2,3,4,5`` gives 1/2/3/4/5+, ``1,2,3`` gives 1/2/3+), and its quotas are allocated against each session's uniform headroom (number of non-empty strata times the smallest), not its eligible count, so every quota can be met with exactly equal strata; the type's total uniform headroom is the largest budget that stays exactly uniform (the builder warns when a budget exceeds it). Types are drawn in sorted order and sessions in the order of ``--root-directories``, all from one generator seeded with ``random_state``.
* **Split.** Whole sessions are held out, type by type (sorted), each type's sessions (sorted by id) permuted and moved to validation until they hold ``validation_split`` of the type's rows, so no session straddles the boundary.
* **Spectrograms.** Each drawn spectrogram and its SAM mask union (binarized at 0.5 after the same resize) is resized to ``target_shape`` without time stretching. With ``apply_mask`` the mask is multiplied in (the phase 9 sets); without, the spectrogram stays unmasked and, with a ``floor``, is min-max normalized on its own and floored, ``clip((x - floor) / (1 - floor), 0, 1)`` (the phase 6 sets, floor 0.2). ``masks``, ``masks_len`` and ``mask_count`` are stored either way, so a masked set and its unmasked twin of the same draw differ only in ``spectrograms``. ``masking_type none`` reads no masks (and so allows neither ``apply_mask``, ``require_mask`` nor a ``uniform`` draw).
* **Conditioning values.** Every split also carries ``mean_freq_hz``, ``freq_bandwidth_hz``, ``loudness_db`` and ``spectral_entropy`` (float64), copied row for row from the session's *usv_summary.csv* (the mean frequency and bandwidth of the call's SAM-masked region, its absolute masked image-level loudness and the spectral entropy in nats of its normalized frequency power profile, all written by ``generate-usv-acoustic-features``; NaN where the summary has no value or no such column): the raw values a conditional ``train-qlvm`` run conditions on, captured with the rows they belong to.

The shipped settings draw the production phase 6 set's recipe (``natural_5strata_N29000_unmasked_floor``), with noise now left out; with ``--no-exclude-noise`` and the reference squeak index's strict exclusions they rebuild that set bit-exactly; ``--no-apply-mask --floor none`` would give a plain unmasked set, ``--apply-mask --floor none`` the phase 9 masked set, ``--draw-mode uniform`` the uniform cells, ``--mask-count-bin-edges 1,2,3 --session-type-targets '{"MF": 65000, "FF": 65000, "MM": null, "lone_male": null}'`` the 3-strata cells. Every session's spectrogram H5 is fingerprinted as it is read: ``SESSION_H5.sha256`` / ``SESSION_H5.tsv`` hold the SHA-256 and row counts of every session H5, so a later consumer can tell whether a session's spectrogram file has been rebuilt since (its ``spec_id`` row numbers would then point at different calls).

Reproduction (2026-09-30, the 324 sessions of the reference sets, seed 42): the phase 6 ``natural_5strata_N29000`` set (``draw-natural_bins-1-2-3-4-5_N29000_len128_seed42_bbvfree_floor0p2``) and the masked (phase 9 input) ``natural_3strata_N65000``, ``uniform_3strata_N65000`` and ``uniform_5strata_N29000`` sets were rebuilt bit for bit -- the same training / validation rows in the same order (49,411 / 12,629, 106,273 / 27,767, 106,860 / 27,180 and 49,287 / 12,753), the same session types, per-type reports and split sessions, and every ``spectrograms``, ``masks``, ``masks_len``, ``durations`` and ``mask_count`` value identical (maximum absolute difference 0). The reference sets left out broadband calls by a *strict* rule of the reference squeak index (segment probability >= 0.385, or at least one run of three frames above it): 2,597 rows of those sessions that the strict rule drops were not flagged by the *usv_summary*'s former binary ``squeak`` column, which was the segment rule alone. The rebuild therefore passed that index's exclusions through the Python-only ``row_exclusions`` argument of ``QLVMTrainingSetBuilder``; with the summary's own ``squeak`` column, the eligible pool grows by 2,388 rows (MF +1,866, FF +474, MM +27, lone_male +21), and since every quota and every draw position shifts with it, only 24,869 of the phase 6 set's 62,040 rows are drawn again (MF and FF still hold 29,000 rows each; the take-all types grow with their pools). A set built from the summary is therefore an equivalent draw, not the same one. (These counts compare the strict rule with the summary's old binary ``squeak`` column, which *Detect squeaks* no longer writes; a ``call_class`` build differs from the reference draw too.) ``--strict-squeak-exclusion --reference-squeak-index-path <index>`` applies the index's strict rule from the index itself, the same exclusions the rebuild passed through ``row_exclusions``, so with ``--no-exclude-noise`` it reproduces the shipped sets from the command line.

**Squeaks** (``build_qlvm_squeak_training_set`` settings block). The candidates are the rows whose ``call_class`` *Detect squeaks* set to ``squeak`` or ``both`` and that carry a squeak envelope (with ``exclude_noise`` also not noise; noise rows carry no class, so this is a guard), sessions in sorted order; a row with several squeaks is one candidate over their envelope, as *Embed squeaks in a QLVM torus* embeds it. ``exclude_noise`` is on by default; the shipped squeak sets were drawn without it (``--no-exclude-noise`` to reproduce them). Each candidate's audio window is the segment widened to hold the envelope and its context (the embedding's window), its first and last frames are the window frames whose centres lie inside ``squeak_start`` / ``squeak_end``; the extent is widened by ``context_frames`` (2) either side, clipped to the window, and crops narrower than ``min_trimmed_frames`` (8) or wider than the 128-frame frame are dropped. The written ``squeak_probability`` is ``p_squeak + p_both``, the probability that the segment holds a squeak. ``crop_window`` ``full_length`` (the shipped setting, and the rule *Embed squeaks in a QLVM torus* follows) crops from the whole segment; ``first_128_frames`` cuts every segment to its first 128 frames, as the reference 128-frame spectrogram store did, and drops the squeaks still going at frame 127 (right-censored). The crop width sets the duration stratum (``duration_bin_edges`` 26 / 40 / 62 frames, four strata). With one generator seeded with ``random_state``, ``per_session_bin_cap`` (48, the ``session`` cells; 0 for the ``lumped`` cells) first keeps at most that many squeaks of any one session in any one stratum, and ``n_total`` (11,000) squeaks are then drawn, ``natural`` (uniformly) or ``uniform`` (equal across strata). The drawn squeaks' sonic spectrograms are rebuilt from the session audio with the *Detect squeaks* front end (``exclude_metadata_audio_channels`` as there), each crop is min-max normalized (``(x - min) / (max - min + 1e-6)``, ``crop_normalization`` ``per_crop``; ``absolute`` is the classifier's fixed dB transform) and centred in a 128 x 128 frame; ``masks`` / ``masks_len`` are zero and ``apply_mask`` false. The split is the USV builder's. The shipped settings are those of the production squeak cell (``natural_session_N11000``) except the two rules squeak embedding follows: full-length crops and the metadata channel exclusion.

Reproduction (2026-09-30, the 324 sessions of the reference sets, 239 of which hold squeaks, seed 42): fed the candidates of the reference squeak index (segment probability >= 0.385 and the extents of its 128-frame pass) through ``QLVMSqueakTrainingSetBuilder.build_from_candidates`` with ``crop_window`` ``first_128_frames`` and ``exclude_metadata_audio_channels`` false, the builder passes the same 15,961 / 14,963 / 14,936 candidates through its gates as the reference builder and draws, crops and splits exactly the reference rows in all five phase 3 sets (natural / uniform x session / lumped, and the full set): the same ``spec_id`` order, ``crop_first`` / ``crop_last``, duration strata and split sessions. For ``bbv-natural_dur-26-40-62_session_N11000_seed42`` (the production cell's set) the spectrograms were rebuilt from the audio too, and all 8,767 / 2,233 training / validation spectrograms are bit-identical to the reference ones (maximum absolute difference 0). The summaries no longer carry the retired classifier's flags and extents (the call-class model replaced them), so a set drawn from the summaries is a new draw over different squeaks; reproducing the shipped sets needs the reference index's candidates fed through ``build_from_candidates`` (with each segment as its own audio window, ``start_s`` / ``stop_s`` = ``start`` / ``stop``).

QLVM decoder
^^^^^^^^^^^^

Defines the shared toroidal latent space (and watershed categories) that makes the ``qlvm_*`` columns comparable across every session embedded with the same model. ``build-qlvm-training-set`` (USVs) or ``build-qlvm-squeak-training-set`` (squeaks) builds the training set (see *QLVM training sets* above) → ``train_data.npz`` + ``val_data.npz`` (+ ``full_data.npz``) + ``metadata.npz``. ``train-qlvm`` then trains a decoder on such a set, or on a set in the same format built outside the repository (the QLVM model packages' training sets: ``train_data.npz`` / ``val_data.npz`` with ``spectrograms``, ``masks``, ``masks_len``, ``durations`` and an optional ``apply_mask`` scalar, plus ``metadata.npz``). It is the in-house JAX port of the recipe the shipped models were trained with (qmc_deep_gen's ``bartul_mouse.py`` under the v3 "shipped" protocol), using the same decoder and likelihood code ``infer-qlvm-latents`` embeds with: each spectrogram is min-max normalized on its own (``(x - min) / (max - min + 1e-8)``) and, when the set applies masks (its ``apply_mask``, else ``masking_type`` ``sam``), multiplied by its binarized SAM mask; a loudness floor (phase 6) is already baked into the stored spectrograms and is only recorded. The decoder (``relu`` or ``legacy`` head, torch's default initialization) is trained over a 610-point Fibonacci lattice (``training_fib_m`` 15) with one uniform random torus shift per batch, minimizing the negative QMC log evidence of the batch, with Adam at a constant learning rate of 1e-3, batches of 512, 300 epochs and seed 42; the last epoch's weights are kept. Every ``val_freq`` (10) epochs and after the last one the same loss is computed on a fixed validation subset -- at most ``val_samples_per_mask_count`` (100) calls per ``masks_len`` value, drawn exactly as the reference trainer draws it -- one call at a time against its own shift of a 6,765-point lattice (``validation_fib_m`` 20). The output directory is a v3 model package cell: ``checkpoint.tar`` (torch zip, the structure of a shipped cell's: ``decoder.<index>`` weights, a torch ``Adam`` state and every batch's loss), ``config/training_contract.json`` (``input_normalization`` ``minmax``, ``masking_type`` ``sam`` or ``none`` from the set's mask use, ``floor``, ``target_shape``, ``time_stretch``, ``length_threshold`` (null when the set records none), ``require_mask``, ``embedding_fib_m`` (24), the lattices, ``"condition": null`` for an unconditional decoder), ``config/run_config.json`` and ``metrics/val_diagnostics.npz`` (per-epoch training loss and seconds, the validation losses and subset). The recipes of the shipped cells: phases 6 (unmasked, floor 0.2) and 9 (SAM-masked) are the defaults; phase 3 (BBVs) is ``--decoder-head legacy --n-epochs 2500 --val-freq 80 --val-samples-per-mask-count 1600``. A short run (30 epochs) on the phase 6 ``natural_5strata_N29000`` set reproduced the shipped run's validation subset row for row, its per-epoch training loss to within 0.52 % and its validation loss to within 0.66 % from epoch 3 on, at 3.3 s per epoch (RTX 4080 SUPER); 240 epochs on the phase 3 ``natural_session_N11000`` set matched the shipped validation loss to within 0.07 % at epochs 80, 160 and 240. On a shared GPU, set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so JAX does not reserve most of its memory. The run refuses a set without ``metadata.npz``, and a ``train_data.npz`` or ``val_data.npz`` older than a ``full_data.npz`` in the same directory. Cluster submitter: ``train_qlvm_global.sh``. Clustering the trained torus into ``inference/clusters_<level>/label_grid.npy`` is a separate step, and ``infer-qlvm-latents`` needs it before it can use the cell.

**Conditional decoders (phase 11).** With ``conditional`` set to ``duration``, ``mean_freq``, ``bandwidth``, ``loudness`` or ``spectral_entropy`` (``--conditional``; null / ``none`` trains an unconditional decoder), ``train-qlvm`` trains a decoder that takes one conditioning value ``c`` per call, appended to the torus basis of every lattice point (``c_dim`` 1), with the recipe of the v3 package's phase 11 cells (qmc_deep_gen ``bartul_mouse_cond.py``, width-capped bins and a scaled loss). ``c`` per call: ``duration`` is ``(d - d_min) / (d_max - d_min + 1e-8)`` of the pre-resize duration in time bins, ``d_min`` / ``d_max`` the training split's (8 and 127 in the shipped cells); ``mean_freq`` is ``(f - 30000) * 127 / (128 * 90000)`` of the call's mean frequency over its SAM-masked region (the summary's ``mean_freq_hz``; equal to the energy-weighted row centroid of the resized SAM-masked spectrogram that ``infer-qlvm-latents`` computes, to 1.5e-7); ``bandwidth`` is ``clip(freq_bandwidth_hz / 90000, 0, 1)``; ``loudness`` is ``clip((dB - 28.83) / (95.85 - 28.83), 0, 1)`` of the call's absolute masked image-level loudness (the summary's ``loudness_db``, measured by ``compute_usv_loudness``); ``spectral_entropy`` is ``clip((H - H_min) / (H_max - H_min), 0, 1)`` of the call's spectral entropy in nats (the summary's ``spectral_entropy``), with ``H_min`` / ``H_max`` the smallest and largest entropy of the training split, so the training values span exactly ``[0, 1]`` and validation rows (and, at inference, new calls) outside that range are clamped; the two are recorded in the contract as ``entropy_min`` / ``entropy_max``, and a training split of constant entropy stops the run. The raw values come from the split's own ``mean_freq_hz`` / ``freq_bandwidth_hz`` / ``loudness_db`` / ``spectral_entropy`` columns (``build-qlvm-training-set`` copies them), or, with ``condition_table`` (``--condition-table``), from a per-call ``.npz`` with ``spec_id`` and the column (the package's ``corpus/cond_table.npz``, whose loudness column ``image_level_db`` is accepted too) for sets built outside the repository; a row without a finite value stops the run. The training values are cut into ``condition_n_bins`` (32) quantile bins, and with ``condition_bin_scheme`` ``quantile_capped`` (the default) every bin wider than the widest inner bin is split into equal-width pieces no wider than it (in practice the two end bins), kept however few rows they hold. Every batch is drawn from one bin and decoded at its rows' mean ``c`` (bins shuffled internally and batches shuffled together each epoch, the reference sampler's draws as the shipped multi-worker runs made them); with ``condition_scale_loss_by_batch`` (default on) each step's loss is multiplied by ``rows / batch_size``, so a small tail batch steps with its rows' share (short batches are padded with zero-weight rows, which changes neither the loss nor its gradient). Validation decodes each call at its own ``c``. The cell's ``training_contract.json`` then carries ``c_dim`` 1, ``conditional`` and a ``condition`` block in the form of the shipped phase 11 contracts (the constants ``infer-qlvm-latents`` computes ``c`` with, the ``decode`` rule: ``exact`` -- the call's own ``c`` clamped to the training range -- for duration and bandwidth, ``grid`` -- the nearest point of the decode grid -- for mean frequency, loudness and spectral entropy), and ``config/condition_bins.npz`` holds the bin ``edges``, ``group_ids`` / ``group_sizes`` / ``group_means``, ``train_c_min`` / ``train_c_max`` and the ``decode_grid`` (``condition_decode_grid_step`` apart, from ``train_c_min`` to at or just past ``train_c_max``; 0.0025 by default, the step of the shipped grid-decoded cells; the shipped duration and bandwidth cells, which never read it, carry 0.01). ``val_diagnostics.npz`` adds each validation call's ``c`` (``val_diag_c``). Validation (2026-09-30, 30 epochs, seed 42, the shipped phase 11 recipe on the ``natural_5strata_N29000`` floor set): the duration and loudness runs wrote ``condition_bins.npz`` bit-identical to the shipped cells' (edges, groups, means, training range and decode grid, with the step each shipped cell carries; the shipped files add only ``run_decode_grid_step``), reproduced the shipped validation subset row for row and the shipped batch order (from epoch 3 on, each epoch's per-batch losses correlate with the shipped run's same epoch at r >= 0.999), and from epoch 3 on matched the shipped per-epoch training loss to within 0.88 % (duration) and 0.87 % (loudness), and the validation loss at epochs 10, 20 and 30 to within 0.84 % and 0.20 % -- less than the port's own run-to-run spread (the duration run with every JAX key changed moved by up to 0.87 % in training and 1.14 % in validation loss). About 4.1-4.6 s per epoch (RTX 4080 SUPER).

QLVM model packages
^^^^^^^^^^^^^^^^^^^

A QLVM model package (the ``qlvm_models_latest/v2`` and ``v3`` layout; v3 is v2.1 plus the phase 11 cells) ships trained decoders together with everything inference needs, one folder per cell: ``<package>/<phase>/<cell>``. ``infer-qlvm-latents`` reads every cell of ``model_cells``. Two layouts are read. v3, restructured on 2026-09-28, keeps each cell's contract and bins in ``config/``, its per-call tables in ``inference/`` and each cluster level in ``inference/clusters_<level>/``, with ``checkpoint.tar`` at the top, and keeps the package's ``SESSION_H5_BASELINE.tsv`` in ``corpus/``; v2 / v2.1 keep every file at the cell's (or package's) top level and the clusters in ``cluster/<level>/``. The v3 location is tried first. The files it uses are:

* ``checkpoint.tar`` — a torch zip checkpoint whose ``"model"`` entry holds the decoder ``state_dict`` (``decoder.<index>.weight`` / ``.bias``); read without torch. The head follows from the keys: ``decoder.1.weight`` is the activation-free legacy head, ``decoder.2.weight`` the ReLU head
* ``training_contract.json`` — ``decoder_head``, ``latent_dim``, ``c_dim``, ``input_normalization`` (``"minmax"``) and ``normalization_epsilon``, ``masking_type``, ``floor``, ``target_shape``, ``time_stretch``, ``length_threshold``, ``require_mask`` (``true``: the corpus left out calls without a SAM mask, in every phase), ``embedding_lattice_type`` (``"fibonacci"``) and ``embedding_fib_m``, and the ``condition`` block of conditional cells (``"duration"``: ``duration_min``, ``duration_max``, ``epsilon``; ``"mean_freq"``: ``spectrogram``, ``epsilon``; ``"bandwidth"``: ``source``; ``"loudness"``: ``source``, ``db_range``; ``"spectral_entropy"`` (``train-qlvm`` cells): ``source``, ``entropy_min``, ``entropy_max``; in phase 11 also ``decode``, ``"exact"`` or ``"grid"``) with ``condition_bins``
* ``label_grid.npy`` of the fine and coarse cluster folders (``inference/clusters_fine`` / ``clusters_coarse``; ``cluster/fine`` / ``cluster/coarse`` in v2 / v2.1) — ``(200, 200)`` int16 label grids indexed ``[y, x]``, labels ``1 … k`` numbered by cluster size; the label of a USV is ``grid[floor(y * 200) mod 200, floor(x * 200) mod 200]`` for its coordinates ``(x, y)`` (e.g. ``qlvm1`` / ``qlvm2``), the pixel rule the package labels its corpus with, applied to the reference arrays too
* ``condition_bins.npz`` (conditional cells) — phase 10: ``edges`` (the corpus quantile-bin edges of the conditioning value) and ``bin_mean`` (the corpus mean value per bin); phase 11: ``train_c_min`` / ``train_c_max`` (the training range), ``decode_grid`` / ``decode_grid_step`` (the grid its corpus was decoded on) and the capped training bins (``edges``, ``group_ids``, ``group_sizes``, ``group_means``), with no ``bin_mean``. A cell whose bins do not match its contract's ``decode`` is refused
* ``posterior_cache.npz`` and ``recon_mse_breakdown.npz`` (``inference/``) — the package's per-call coordinates and the durations and mask counts they were computed from, read by the package route of ``model_cells`` and by ``export-qlvm-reference-arrays``
* ``clusters.csv`` and ``cluster_labels.csv`` of each cluster folder — read only by ``export-qlvm-reference-arrays``

The package's own ``code/selftest.py`` checks every file above against its sources and re-embeds corpus calls; run it on a copy before using it. Runtime is set by the 46,368-point lattice: decoding it takes about 20 s on CPU (JAX, 144 cores; decoding 4,096 points takes 1.5–1.8 s, and ``jax.jit`` barely changes that), so an unconditional cell embeds a session of ~700 USVs in about 25 s, and a phase 10 conditional cell, which decodes the lattice once per distinct bin mean, in about 11–12 minutes (a phase 11 cell decodes it roughly 3–8 times as often). The ``gpu`` extra (JAX with CUDA) shortens both. Embedding a session with a cell reproduces the package's corpus labels: on two sessions per cell (phase 6, 9 and 10 ``natural_3strata_N65000``), the model inputs matched the package's bitwise, and 99.0–100 % of fine labels matched ``cluster_labels.csv``, all of the rest being calls whose posterior mean crossed a pixel edge (conditional cells decode at the frozen bin mean rather than the corpus embedding's batch mean, which moves a few more calls). Phase 11 cells decode at the same value as the package's corpus embedding: on session ``20251003_143416`` (907 USVs, the four ``natural_3strata_N65000`` cells, one L40S with the ``gpu`` extra), the embedded calls were exactly the package's, their conditioning values matched ``condition_decode.npz`` to float32 rounding, 99.7–99.9 % of fine and coarse labels matched, and a session took about 100–115 s (190 s for loudness, which also measures every call's loudness from the audio). The remaining differences are calls with a spread-out posterior, where the float32 embedding here and the package's TF32 embedding round differently.

Mask detector
^^^^^^^^^^^^^

The YOLO box detector that localizes each call in its spectrogram so SAM2 can segment it; ``generate-usv-masks`` reloads its weights. ``export-yolo-dataset`` renders the cohort's spectrograms to an Ultralytics dataset (``images/`` + ``labels/`` + ``data.yaml``); ``train-masks`` fine-tunes YOLO → the run directory + ``best.pt``. Cluster submitter: ``train_masks_global.sh``.

Box labels are set by ``--label-source`` (or ``export_yolo_dataset.label_source``): ``cc`` (default — pseudo-labels from the connected-component detector; zero manual work, no GPU; the recommended start), ``manual`` (hand-verified ``{spec_id}.txt`` YOLO files in ``--manual-labels-directory``), or ``merge`` (``cc`` pseudo-labels overridden by manual files where present). ``manual`` / ``merge`` require ``--manual-labels-directory``; ``cc`` ignores it. The submitter exposes a ``LABEL_SOURCE`` knob and ``MANUAL_LABELS_DIRECTORY``. Both ``generate-usv-masks`` and ``train-masks`` need the ``sam2`` and ``ultralytics`` packages (usv-playpen core dependencies).

Noise model
^^^^^^^^^^^

The ensemble *Detect noise* scores with. ``train-noise-model`` trains it on a labels CSV and writes a bundle ``detect-usv-noise`` loads unchanged (set it as ``noise_model_path``). Drawing the segments to label, the labelling itself, the cross-fitted calibration and the choice of the decision cut-offs are not part of it: they are done once per labelling round and come in as the labels CSV and the ``calibration_path`` JSON.

* **Labels** -- one row per training example: ``sample_id``, ``session_dir``, ``start`` and ``stop`` (s, as in the session's *usv_summary.csv*), ``chs_count`` (the summary's channel count) and ``noise`` (``1``: the segment holds no vocalization at all, neither a USV nor a squeak; ``0``: it holds one). Unsure answers must be dropped or resolved first. The segment is defined by these columns, not by a summary row, so re-curating a session later cannot change what a label points at. A segment may appear twice (it is then trained on twice, as the production ensemble was); repeats and conflicting repeats are reported.
* **Inputs** -- built by the same function *Detect noise* scores with (same wavs, channel exclusion, ~100 ms context window, crop, dB transform), so a model is trained on exactly the input it will be run on. The scalars (log channel count, log duration) are standardized over the training set, and their mean and standard deviation go into the bundle.
* **Recipe** -- the production one: per seed, the network is initialized from ``torch.manual_seed(seed)`` and trained for 40 epochs in batches of 32 (a seeded permutation each epoch), with Adam (learning rate 0.001, weight decay 0.0001) under a cosine schedule stepped per batch, binary cross-entropy on targets smoothed by 0.05, and augmentation of the spectrogram channels (gain jitter of up to ±5 dB, a frequency roll of up to ±2 rows, and with probability 0.5 each a time mask of up to 15% of the frames and a frequency mask of 1-10 rows). cuDNN runs in deterministic mode, but GPU training is still not bit-reproducible, so a retrained ensemble matches an earlier one in its decisions, not its weights.
* **Bundle** -- the five ``state_dicts``, the input contract (bands, dB constants, frame cap, context), the scalar standardization, the calibration table and decision block from the JSON, and the provenance (labels CSV, recipe with every seed, build date). An existing bundle path is refused, never overwritten, and the written bundle is loaded back through the detector's loader before the run ends.

Retraining the production bundle from its 4,680 labels and five seeds (one RTX 4080, 23 min including the input build) reproduced it in its decisions. Every input matched the cached training inputs of the original run bit for bit, as did the scalar standardization, and on the CPU the training loop gives weights identical to the original code's. On the 1,118 consensus-labelled segments the cut-offs were chosen on (all of them training labels of both ensembles), weighted to the cohort, the confident decisions reach precision 0.988 / recall 0.993 (production 0.988 / 0.993) with 0.74% of segments uncertain (production 0.75%), and 99.85% of three-way decisions agree (99.91% of ``noise`` values). No segment moved between confident vocalization and confident noise. Unweighted, 94.8% of decisions agree (58 of 1,118, all between a confident decision and the uncertain band), because the set was drawn heavily from the uncertain score range (22% of it is uncertain).

Call-class model
^^^^^^^^^^^^^^^^

The ensemble *Detect squeaks* scores with. ``train-usv-squeak-model`` (``train_usv_squeak_model`` settings block) trains it on the labelling tool's files and writes a bundle ``detect-usv-squeaks`` loads unchanged (set it as ``squeak_model_path``). Drawing the segments to label, the labelling, the session-grouped cross-validation and the choice of the span rule are not part of it: they are done once per labelling round, outside the package, and come in as the label files and the ``span_threshold`` / ``span_min_run_frames`` settings.

* **Labels** -- one or more label sets (``label_sets``), each a labels CSV (``panel_id``, ``label``: 0 usv, 1 squeak, 2 both, 3 unsure; ``squeak_extents_s``: the labelled spans as a JSON list of ``[start_s, end_s]``, every squeak in view, the context included) with its sample CSV (``panel_id``, ``session_dir``, ``row_index``, ``start``, ``stop``). Sample ids are ``<set name>_<panel_id>``. Review CSVs in the labels format (``label_overrides``) replace the label and spans of the panels they list in the named set; the set's own CSV is never edited. Unsure answers are dropped; a ``usv`` label with spans, or a ``squeak`` / ``both`` label without one, stops the run. Each segment's ``chs_count`` is read from its session's *usv_summary.csv* row, which must still start where the sample says. The shipped settings list the production bundle's two rounds and review override.
* **Inputs** -- built by the same function *Detect squeaks* scores with (same wavs, channel exclusion, context window and segment indicator), so a model is trained on exactly the input it will be run on; frame targets are 1 on every frame whose centre lies inside a labelled span. The scalars (log channel count, log duration) are standardized over the training set, and their mean and standard deviation go into the bundle.
* **Recipe** -- per seed, the network is initialized from ``torch.manual_seed(seed)``, its trunk loaded from member ``seed mod 5`` of the noise ensemble (``pretrained``; the noise bundle of ``detect_usv_noise.noise_model_path``), and trained with the noise model's recipe (40 epochs, batch 32, Adam 0.001 / weight decay 0.0001 under a cosine schedule, the same augmentation) on the class cross-entropy (label smoothing 0.05) plus the frame binary cross-entropy (weight 1), which supervises every frame of a squeak / both segment but only the segment's own frames of a usv segment (its label says nothing about the context).
* **Bundle** -- the five ``state_dicts``, the input contract, the scalar standardization, the class names, the span rule (``extent_rule``), the recipe with every seed and the label files. An existing bundle path is refused, never overwritten, and the written bundle is loaded back through the detector's loader before the run ends.

A/V synchronization
-------------------
To run audio/video (A/V) synchronization, you need to list the root directories of interest, select *Run A/V sync check*, click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_16.png
   :align: center
   :alt: Processing Step 16

.. raw:: html

   <br>

The A/V synchronization procedure will first create a *sync_px* file for each input camera, recording pixel intensities of each LED position. The objective is to identify the start of each IPI event in camera time and on both audio devices. One can then compare, for each individual IPI event, what the discrepancy is between the clocks of both devices and that is captured in the *summary.svg* histograms.

.. parsed-literal::

    ├── 20250430_145017
    │   ├── 20250430_145017_metadata.yaml
    │   ├── audio
    │   │   ...
    │   ├── ephys
    │   │   ...
    │   ├── sync
    │   │   ...
    │   │   ├── **nidq_ipi_data.npy**
    │   │   ├── **sync_px_21372315-250430145009**
    │   │   ├── **20250430_145017_summary.svg**
    │   └── video
    │       ...

An example output of the A/V synchronization procedure is shown below.

Notice that the plot contains two columns, one for each Avisoft UltraSoundGate hardware (USGH) device (which can operate in NO SYNC mode). In the first row, you can observe the distribution of A-V IPI discrepancies, which is the difference between the IPI onsets detected in the video and audio data. In the example, you can see the discrepancy goes rarely beyond one camera frame, which is ~6 ms, an acceptable amount of jitter. One might also be interested in viewing how this discrepancy evolves over time. One thing we would want to avoid are drastic changes in sampling rates on any of the devices over time. In the second row, you can see the relationship between IPI onsets time (earlier-later in the session) and the A-V IPI discrepancy. Ideally, we would want to observe a *flat cloud* of points, which would indicate that the A/V IPI discrepancy is stable over time. If you observe a trend that goes beyond 2 tracking frames, it might be worth investigating further.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/sync_summary_example_noNIDQ.png
   :align: center
   :alt: Sync summary

.. raw:: html

   <br>

In case NIDQ was also used in the recording, the first of the device plots will have a subplot detailing the temporal relationship between the NIDQ IPI onsets and the video IPI onsets (in ms). This plot is informative in case there is a large A-V discrepancy, as it allows you to determine which device (A or V) is having issues. If the NIDQ-V discrepancy is small, the sync issue is likely related to the audio device. On the contrary, if the NIDQ-V discrepancy is large, the sync issue is likely related to the video device. Either way, this is a first step in investigating this further, which is highly recommended.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/sync_summary_example_NIDQ.png
   :align: center
   :alt: Sync summary NIDQ

.. raw:: html

   <br>

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section fully modifiable in the GUI, with the following parameters:

* **extra_data_camera** : serial number of the camera used to store phidget data
* **sync_ch_receiving_input** : microphone channel receiving Arduino digital input
* **extract_exact_video_frame_times_bool** : instead of using frame indices multiplied by empirical frame rate, use Loopbio times directly (which is less precise!)
* **nidq_sr** : sampling rate of the NIDQ device (in Hz)
* **nidq_num_channels** : number of channels on the NIDQ device (9 on BNC-2110)
* **nidq_bool** : whether NIDQ device received Triggerbox AND sync input
* **nidq_triggerbox_input_bit_position** : triggerbox input bit position on the NIDQ device digital channel (assumes last channel is digital!)
* **nidq_sync_input_bit_position** : sync input bit position on the NIDQ device digital channel (assumes last channel is digital!)
* **sync_camera_serial_num** : serial numbers of cameras that can detect flashing LEDs
* **led_px_version** : version of the LED pixel positions
* **led_px_dev** : maximal deviation (in px) of observed LED flashes relative to expected positions
* **sync_video_extension** : video type (usually "mp4")
* **relative_intensity_threshold** : top threshold (on 0-1 scale) for relative temporal change in pixel intensity
* **millisecond_divergence_tolerance** : maximal deviation of IPI onsets (in ms) between video detections and ground truth

.. code-block:: json

   "extract_phidget_data": {
    "Gatherer": {
      "prepare_data_for_analyses": {
        "extra_data_camera": "22085397"
      }
    }
   },
   "find_audio_sync_trains": {
        "sync_ch_receiving_input": 2,
        "extract_exact_video_frame_times_bool": false,
        "nidq_sr": 62500.72887,
        "nidq_num_channels": 9,
        "nidq_bool": false,
        "nidq_triggerbox_input_bit_position": 5,
        "nidq_sync_input_bit_position": 7
    },
   "find_video_sync_trains": {
        "sync_camera_serial_num": [
            "21372315"
        ],
        "led_px_version": "current",
        "led_px_dev": 10,
        "sync_video_extension": "mp4",
        "relative_intensity_threshold": 1.0,
        "millisecond_divergence_tolerance": 12
   }
