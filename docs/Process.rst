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
   below show that shipped default. Every name in the host config's
   ``experimenter_list`` must be non-empty: an empty name would match every
   path, so the re-keying refuses it.

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

The last four run only when (re)training the spectrogram-pipeline models: they aggregate a cohort of sessions and produce the QLVM (the in-house quasi-Monte Carlo latent variable model) decoder and You Only Look Once (YOLO) object detector weights that the per-session *Infer QLVM latents* / *Generate (spectrogram) masks* steps reload (see the *Label, render and embed USVs* section below).

On the other hand, for **processing sessions separately**, the order of processing steps is as follows:

    #. Run video concatenation
    #. Run video re-encoding
    #. Convert to single-ch files
    #. Crop AUDIO (to VIDEO)
    #. Run A/V sync check
    #. Run E/V sync check
    #. Run HPSS
    #. High-pass filter WAVs
    #. Ultrasonic MEMMAP
    #. Broadband MEMMAP
    #. Run SLP-H5 conversion
    #. Run AP calibration
    #. Run AP triangulation
    #. Re-coordinate
    #. Run DAS inference
    #. Curate DAS outputs
    #. Prepare USV assignment
    #. Run USV assignment
    #. Produce noise labels
    #. Produce squeak labels
    #. Generate spectrograms
    #. Generate (spectrogram) masks
    #. Compute USV features
    #. Infer QLVM latents

A step that cannot do its job raises rather than returning as if it had: a missing calibration for the vocalocator or for the triangulation, a failed vocalocator subprocess, and missing DAS annotations or a failed summary all stop that session, and the processing run lists it with the reason among the failures in its completion e-mail and log while the other sessions continue.

*Infer QLVM latents* writes each USV's coordinates to the summary row of the same position as its spectrogram H5 row, so a session whose *usv_summary.csv* and spectrogram H5 differ in length (one rebuilt without the other) is refused before anything is embedded. It also labels the USVs of the regular map with the QLVM categories (``qlvm_category``; the ``assign-qlvm-categories`` step), since re-inferring the latents drops that column, and then embeds the squeaks with the squeak QLVM (``qlvm_squeak1`` / ``qlvm_squeak2``; the ``infer-qlvm-squeak-latents`` step). The three always run together and have no separate toggles.

The same per-session order can be run on the Princeton cluster with the four SLURM scripts in *usv_playpen/other/cluster/usv_playpen*. Each one is a template: set the variables at its top (``EXPERIMENTER_ID``, ``WORK_DIR``, ``USV_PLAYPEN_PATH``, ``SESSION_ROOT_DIRECTORY``, the e-mail fields and, where present, the arena session, experimental code and model folders), then run it with ``bash <script>``; it writes a job file into ``WORK_DIR`` and submits it with ``sbatch``. Run the four steps in order, each after the previous one has finished:

    #. *process_data_step_one_inference_global.sh* (24 CPUs, 128 GB, 5 h): ``concatenate-video-files``, ``rectify-video-fps``, ``multichannel-to-single-ch``, ``crop-wav-files``, HPSS (submits the HPSS job array and waits for it), ``bp-filter-audio`` (*High-pass filter WAVs*), ``concatenate-audio-files`` (*Ultrasonic MEMMAP*), ``broadband-filter-audio`` (*Broadband MEMMAP*, using all the job's CPUs), DAS inference (submits the DAS job array and waits for it), ``das-summarize`` (*Curate DAS outputs*) and ``av-sync-check``
    #. *process_data_step_two_inference_global.sh* (6 CPUs, 48 GB, 2 h): ``sleap-to-h5``, ``anipose-triangulate`` and ``anipose-trm`` (*Re-coordinate*)
    #. *process_data_step_three_inference_global.sh* (2 CPUs, 18 GB, 2 h): ``prepare-vcl-assign`` and ``vcl-assign`` (*Prepare USV assignment* and *Run USV assignment*, in the vocalocator conda environment)
    #. *process_data_step_four_inference_global.sh* (1 GPU, requested as ``--gres=gpu:1``, 8 CPUs, 64 GB, 3 h): ``detect-usv-noise`` and ``detect-usv-squeaks`` (*Produce noise labels* and *Produce squeak labels*), ``generate-usv-spectrograms``, ``generate-usv-masks``, ``generate-usv-acoustic-features`` (*Compute USV features*), ``infer-qlvm-latents``, ``assign-qlvm-categories`` and ``infer-qlvm-squeak-latents`` (*Infer QLVM latents*)

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

The steps rely on this layout holding one file of each kind a step looks for. A step that looks for a file by pattern (a session's *usv_summary.csv*, its spectrogram H5, its 3D track, its camera frame-count JSON, the probe folder's changepoints JSON, the Anipose calibration TOML, the camera's recording folder, ...) stops with an error listing every match when more than one file matches, instead of reading (and possibly rewriting) whichever sorts first; remove or rename the stray copy and run the step again. Where several files are expected (the 24 per-channel WAVs, the master multichannel chunks), the step names its choice.

Every file that a later step reads back (USV summaries, spectrogram and mask H5 files, 3D tracks, spike files, changepoints and sync JSONs, rewritten WAVs, model bundles, training sets) is written to a temporary file in the same folder and renamed into place, so an interrupted run leaves the previous file whole. An external program that fails or runs past its time budget (sox, ffmpeg, DAS, SLEAP conversion) stops the step with its command and return code.

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

Neural recording data is aligned to the start of video recording, which is identifiable by searching for a ~2.3 s break in Loopbio Triggerbox pulses, which are constantly being transmitted to the Neuropixels digital input channel. The code recursively finds all the *ap.bin* files in the root directory and saves the digital input channel data (385th or last channel) to a separate Numpy file (which ends with *_sync_ch_data.npy*), if it hasn't been saved already.

After finding the tracking start and end (based on the largest Triggerbox break duration and total number of recording frames) in this Numpy file, the total video duration will then be compared to the total video-aligned neural recording, and you will get a report back whether that discrepancy is below 12 ms (in other words, less than 2 video frames, which is an acceptable level of distortion). Information at what Neuropixels sample the first and last video recording frame were detected will be saved to, for instance, */mnt/falkner/Bartul/EPHYS/20250430_imec0/changepoints_info_20250430_imec0.json*, as exemplified below:

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

.. code-block:: json

    "validate_ephys_video_sync": {
            "npx_file_type": "ap",
            "npx_ms_divergence_tolerance": 12.0,
            "apply_phase_shift": true
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`npx_file_type <proc-validate-ephys-video-sync-npx-file-type>`
     - ``'ap'``
     - ``'ap'``, ``'lf'``
   * - :ref:`npx_ms_divergence_tolerance <proc-validate-ephys-video-sync-npx-ms-divergence-tolerance>`
     - ``12.0``
     - milliseconds > 0
   * - :ref:`apply_phase_shift <proc-validate-ephys-video-sync-apply-phase-shift>`
     - ``true``
     - ``true``, ``false``

.. _proc-validate-ephys-video-sync-npx-file-type:

``npx_file_type``
   Which Neuropixels binary is read: Neuropixels 1.0 had "lf" and "ap" files, and this field allows you to switch between them.

   :Options:
      * ``'ap'`` — the AP (action-potential) band binary
      * ``'lf'`` — the LF (local-field-potential) band binary
   :Default: ``'ap'``
   :Notes: the e-phys concatenation step reads the same setting to pick which binaries to stitch. The ADC phase-shift correction (``apply_phase_shift``) runs on the ``"ap"`` band only.

.. _proc-validate-ephys-video-sync-npx-ms-divergence-tolerance:

``npx_ms_divergence_tolerance``
   The maximum allowed difference between the video and e-phys recording duration, in milliseconds.

   :Options: any positive number of milliseconds
   :Default: ``12.0``
   :Notes: the default value is 12 ms (less than 2 video frames, see above), but it can be tuned to whatever the user thinks is appropriate.

.. _proc-validate-ephys-video-sync-apply-phase-shift:

``apply_phase_shift``
   Whether each session's AP binary is de-skewed for the Neuropixels ADC sample-time offset (see the note below) as part of the sync check.

   :Options:
      * ``true`` — de-skew each session's AP binary (in place) during the sync check
      * ``false`` — leave the binaries as recorded
   :Default: ``true``
   :Notes: ``"ap"`` band only; with ``npx_file_type`` set to ``'lf'`` no correction is applied.

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

In the *changepoints* JSON file, the concatenation process will modify all lines other than the ones described above for E/V sync. A tracking window already stored there is moved onto the concatenated clock once: if no concatenation has run yet (no stored session start) the E/V sync step wrote it relative to its own file and the file's offset is added; otherwise the earlier concatenation's offset is replaced by this one's, so re-running the concatenation never shifts it again.

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

.. code-block:: json

    "get_spike_times": {
        "min_spike_num": 100,
        "kilosort_version": "4",
        "remove_duplicate_spikes": true,
        "duplicate_censored_period_ms": 0.3
      },

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`min_spike_num <proc-get-spike-times-min-spike-num>`
     - ``100``
     - integer >= 0
   * - :ref:`kilosort_version <proc-get-spike-times-kilosort-version>`
     - ``'4'``
     - Kilosort version string
   * - :ref:`remove_duplicate_spikes <proc-get-spike-times-remove-duplicate-spikes>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`duplicate_censored_period_ms <proc-get-spike-times-duplicate-censored-period-ms>`
     - ``0.3``
     - milliseconds > 0

.. _proc-get-spike-times-min-spike-num:

``min_spike_num``
   Eliminate clusters with this many spikes or fewer: a cluster is kept only when it has more than ``min_spike_num`` spikes.

   :Options: an integer ``>= 0``; set ``0`` if you want to keep all
   :Default: ``100``
   :Notes: the count is taken per session, i.e., over the cluster's spikes that fall into that session after the split.

.. _proc-get-spike-times-kilosort-version:

``kilosort_version``
   The Kilosort version in use.

   :Options: a version string, e.g. ``"4"``
   :Default: ``'4'``
   :Notes: the sort is read from the ``kilosort<version>`` directory (e.g. ``kilosort4``) in the probe's *EPHYS* directory (e.g. */mnt/falkner/Bartul/EPHYS/20250430_imec0*).

.. _proc-get-spike-times-remove-duplicate-spikes:

``remove_duplicate_spikes``
   Whether near-coincident duplicate spikes are dropped per unit before splitting into sessions (see the note below).

   :Options:
      * ``true`` — drop the second of any spike pair of one unit closer than ``duplicate_censored_period_ms``
      * ``false`` — keep every spike of the curated sort
   :Default: ``true``

.. _proc-get-spike-times-duplicate-censored-period-ms:

``duplicate_censored_period_ms``
   Two spikes of one unit closer than this (in ms) count as a duplicate.

   :Options: a positive number of milliseconds
   :Default: ``0.3``
   :Notes: used only when ``remove_duplicate_spikes`` is ``true``.

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
        "concatenated_video_name": "concatenated_temp",
        "subprocess_timeout_hours": 3
      },
      "rectify_video_fps": {
        "encode_camera_serial_num": [
          "21372315",
          "21372316",
          "21369048",
          "22085397",
          "21241563"
        ],
        "encode_video_extension": "mp4",
        "constant_rate_factor": 16,
        "encoding_preset": "veryfast",
        "delete_old_file": true,
        "subprocess_timeout_hours": 2
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`concatenate_video_files.concatenate_camera_serial_num <proc-video-camera-serial-num>`
     - see JSON above
     - list of camera serial numbers (strings)
   * - :ref:`concatenate_video_files.concatenate_video_extension <proc-video-extension>`
     - ``'mp4'``
     - video file extension
   * - :ref:`concatenate_video_files.concatenated_video_name <proc-concatenate-video-files-concatenated-video-name>`
     - ``'concatenated_temp'``
     - any string
   * - :ref:`concatenate_video_files.subprocess_timeout_hours <proc-video-subprocess-timeout-hours>`
     - ``3``
     - hours > 0
   * - :ref:`rectify_video_fps.encode_camera_serial_num <proc-video-camera-serial-num>`
     - see JSON above
     - list of camera serial numbers (strings)
   * - :ref:`rectify_video_fps.encode_video_extension <proc-video-extension>`
     - ``'mp4'``
     - video file extension
   * - :ref:`rectify_video_fps.constant_rate_factor <proc-rectify-video-fps-constant-rate-factor>`
     - ``16``
     - FFMPEG CRF, integer
   * - :ref:`rectify_video_fps.encoding_preset <proc-rectify-video-fps-encoding-preset>`
     - ``'veryfast'``
     - FFMPEG preset name
   * - :ref:`rectify_video_fps.delete_old_file <proc-rectify-video-fps-delete-old-file>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`rectify_video_fps.subprocess_timeout_hours <proc-video-subprocess-timeout-hours>`
     - ``2``
     - hours > 0

.. _proc-video-camera-serial-num:

``concatenate_camera_serial_num`` / ``encode_camera_serial_num``
   Serial numbers of cameras used in the recording (for concatenation / re-encoding, respectively).

   :Options: a list of camera serial-number strings
   :Default: the five serials in the JSON above, the same list in both blocks

.. _proc-video-extension:

``concatenate_video_extension`` / ``encode_video_extension``
   Video type (for concatenation / re-encoding, respectively).

   :Options: a video file extension, usually "mp4"
   :Default: ``'mp4'`` in both blocks

.. _proc-concatenate-video-files-concatenated-video-name:

``concatenated_video_name``
   Name of the concatenated video file; concatenation writes *<concatenated_video_name>_<serial>* inside each camera folder that holds more than one video chunk and then moves it up into the session's *video* directory.

   :Options: any string usable as a file name
   :Default: ``'concatenated_temp'``
   :Notes:
      * A file of that name already in the camera folder (left behind by a crashed run) is never treated as a chunk: it is deleted, with a logged message, and rebuilt, and ffmpeg always overwrites its output.
      * A camera folder with exactly one video has nothing to join and is left untouched (logged); a camera folder with no video is skipped with a logged message.
      * Re-encoding looks for this file as its input; see :ref:`how re-encoding finds its input <proc-rectify-video-fps-input>` below.

.. _proc-rectify-video-fps-constant-rate-factor:

``constant_rate_factor``
   FFMPEG constant rate factor for re-encoding.

   :Options: an integer; the GUI field is labelled *FFMPEG crf (0–51)*
   :Default: ``16``

.. _proc-rectify-video-fps-encoding-preset:

``encoding_preset``
   FFMPEG encoding preset for re-encoding.

   :Options: an FFMPEG speed preset (the command line accepts ``'ultrafast'``, ``'superfast'``, ``'veryfast'``, ``'faster'``, ``'fast'``, ``'medium'``, ``'slow'``, ``'slower'``, ``'veryslow'``)
   :Default: ``'veryfast'``
   :Notes: set in the GUI as *FFMPEG preset*.

.. _proc-rectify-video-fps-delete-old-file:

``delete_old_file``
   Whether to delete the concatenated file after re-encoding.

   :Options:
      * ``true`` — delete the concatenated *<concatenated_video_name>_<serial>* file once it has been re-encoded
      * ``false`` — keep it
   :Default: ``true``
   :Notes: a single video read from its camera folder is the raw recording and is never deleted, whatever this setting; neither is a calibration folder's *000000* video.

.. _proc-video-subprocess-timeout-hours:

``subprocess_timeout_hours`` (``concatenate_video_files`` / ``rectify_video_fps``)
   Hours the ffmpeg subprocesses of each step may run, all cameras together, before they are killed as hung and the step stops.

   :Options: a positive number of hours
   :Default: ``3`` for concatenation, ``2`` for re-encoding
   :Notes: the concatenation timeout is not in the GUI; the re-encoding one is set in the GUI as *FFMPEG timeout (h)*. Both are generous for a normal session, while still catching a wedged ffmpeg.

.. _proc-rectify-video-fps-input:

**How re-encoding finds its input.** Re-encoding finds each camera's input itself,
so no file name is set for it: the concatenated *<concatenated_video_name>_<serial>*
file in the *video* directory if concatenation made one (deleted after re-encoding
when ``delete_old_file`` is true); otherwise the single video in the camera folder,
whatever its name (e.g. *000000*), read in place and never deleted. A camera folder
with several videos and no concatenated file stops the step with an error asking to
run concatenation first (re-encoding one chunk would give a partial video), and one
with no video stops it too. Every camera is checked before any ffmpeg process starts.
Calibration folders always re-encode their *000000* video, which is never deleted.

Prepare SLEAP cluster job
~~~~~~~~~~~~~~~~~~~~~~~~~
The *usv-playpen* GUI assumes usage of the SLEAP pose-tracking framework (`SLEAP <https://sleap.ai/>`_) for animal pose tracking. To do this, one first needs to train one or multiple models on the data of interest (*i.e.*, social interactions). Explaining how to do this is beyond the scope of this text, so we will assume you already have a *top-down centroid and centered instance model* ready for running inference.

Since the average office PC does not necessarily have GPU-capabilities, it is advised to run SLEAP inference on a high-performance computing cluster, as these usually have GPU-capabilities and allow for the parallelization of the inference process. The *usv-playpen* GUI helps you prepare the SLEAP cluster job, but you will need to run the job on the cluster yourself.

The preparation consists of creating a *job_list.txt* file which contains the paths to the video files and the model(s) to be used for inference. The job list can then be used by a shell script, such as the one in */usv-playpen/other/cluster/SLEAP/sleap_inference_global.sh* to execute inference on all video files of interest.

To run the SLEAP cluster job preparation, you need to list the root directories of interest (which will search for all videos recorded in those sessions), select directories of centroid and centered instance models, select the output inference directory, select *Make SLEAP job list*, click *Next* and finally *Process*:

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

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`camera_names <proc-prepare-cluster-job-camera-names>`
     - see JSON above
     - list of camera serial numbers (strings)
   * - :ref:`inference_root_dir <proc-prepare-cluster-job-inference-root-dir>`
     - path
     - path to a directory
   * - :ref:`centroid_model_path <proc-prepare-cluster-job-centroid-model-path>`
     - ``''``
     - path to a SLEAP model
   * - :ref:`centered_instance_model_path <proc-prepare-cluster-job-centered-instance-model-path>`
     - ``''``
     - path to a SLEAP model

.. _proc-prepare-cluster-job-camera-names:

``camera_names``
   Camera serial numbers used in the recording.

   :Options: a list of camera serial-number strings
   :Default: the five serials in the JSON above
   :Notes: only the session's camera video directories whose name is one of these serials get their videos written into the job list.

.. _proc-prepare-cluster-job-inference-root-dir:

``inference_root_dir``
   Directory where the inference job list (*job_list.txt*) will be saved.

   :Options: path to a directory
   :Default: ``/mnt/falkner/Bartul/SLEAP/inference``
   :Notes: set in the GUI as the output inference directory (see above).

.. _proc-prepare-cluster-job-centroid-model-path:

``centroid_model_path``
   Path to the SLEAP centroid model.

   :Options: path to the centroid model directory
   :Default: ``''`` (empty; select it in the GUI)

.. _proc-prepare-cluster-job-centered-instance-model-path:

``centered_instance_model_path``
   Path to the SLEAP centered instance model.

   :Options: path to the centered instance model directory
   :Default: ``''`` (empty; select it in the GUI)

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

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section not modifiable in the GUI, but it can be modified manually:

.. code-block:: json

    "sleap_file_conversion": {
        "subprocess_timeout_hours": 3
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`subprocess_timeout_hours <proc-sleap-file-conversion-subprocess-timeout-hours>`
     - ``3``
     - hours > 0

.. _proc-sleap-file-conversion-subprocess-timeout-hours:

``subprocess_timeout_hours``
   Hours the parallel SLP-to-H5 conversion subprocesses may run, all views together, before they are killed as hung and the step stops.

   :Options: a positive number of hours
   :Default: ``3``
   :Notes: the conversion itself takes minutes, so the limit only catches a hung conversion.


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

To triangulate animal points, you need to list the root directories of interest, list their respective experimental codes, select the directory with the triangulated arena file, select *Run AP triangulation* and *Re-coordinate*, leave *Triangulate arena nodes* at "No", select "animal" for *Save transformation type* and choose "Yes" for *Delete original .h5*. Finally, click *Next* and then *Process* (a progress bar in the terminal will update you on the status of the process):

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

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`calibration_file_loc <proc-conduct-anipose-triangulation-calibration-file-loc>`
     - ``''``
     - path to a directory
   * - :ref:`triangulate_arena_points_bool <proc-conduct-anipose-triangulation-triangulate-arena-points-bool>`
     - ``false``
     - ``true`` (arena), ``false`` (animal)
   * - :ref:`frame_restriction <proc-conduct-anipose-triangulation-frame-restriction>`
     - ``null``
     - ``null``, or ``[first, last]``
   * - :ref:`excluded_views <proc-conduct-anipose-triangulation-excluded-views>`
     - ``[]``
     - list of camera serial numbers
   * - :ref:`display_progress_bool <proc-conduct-anipose-triangulation-display-progress-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`ransac_bool <proc-conduct-anipose-triangulation-ransac-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`rigid_body_constraints <proc-conduct-anipose-triangulation-rigid-body-constraints>`
     - ``[]``
     - list of rigid body constraints
   * - :ref:`weak_body_constraints <proc-conduct-anipose-triangulation-weak-body-constraints>`
     - ``[]``
     - list of weak body constraints
   * - :ref:`smooth_scale <proc-conduct-anipose-triangulation-smooth-scale>`
     - ``4``
     - number
   * - :ref:`weight_weak <proc-conduct-anipose-triangulation-weight-weak>`
     - ``4``
     - number
   * - :ref:`weight_rigid <proc-conduct-anipose-triangulation-weight-rigid>`
     - ``1``
     - number
   * - :ref:`reprojection_error_threshold <proc-conduct-anipose-triangulation-reprojection-error-threshold>`
     - ``5``
     - pixels
   * - :ref:`regularization_function <proc-conduct-anipose-triangulation-regularization-function>`
     - ``'l2'``
     - ``'l1'``, ``'l2'``
   * - :ref:`n_deriv_smooth <proc-conduct-anipose-triangulation-n-deriv-smooth>`
     - ``1``
     - integer
   * - :ref:`original_arena_file_loc <proc-translate-rotate-metric-original-arena-file-loc>`
     - ``''``
     - path to a directory
   * - :ref:`save_transformed_data <proc-translate-rotate-metric-save-transformed-data>`
     - ``'animal'``
     - ``'animal'``, ``'arena'``
   * - :ref:`delete_original_h5 <proc-translate-rotate-metric-delete-original-h5>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`static_reference_len <proc-translate-rotate-metric-static-reference-len>`
     - ``0.615``
     - meters
   * - :ref:`experimental_codes <proc-translate-rotate-metric-experimental-codes>`
     - ``[]``
     - list of experimental codes, one per session

.. _proc-conduct-anipose-triangulation-calibration-file-loc:

``calibration_file_loc``
   Directory containing the _calibration.toml file relevant for the session.

   :Options: path (string) to a directory
   :Default: ``''`` (empty)
   :Notes: GUI field *Tracking calibration / arena root directory*; the same field also sets ``original_arena_file_loc``.

.. _proc-conduct-anipose-triangulation-triangulate-arena-points-bool:

``triangulate_arena_points_bool``
   Whether to triangulate arena or animal tracked nodes.

   :Options:
      * ``true`` — triangulate the arena nodes (microphones and corners)
      * ``false`` — triangulate the animal nodes
   :Default: ``false``
   :Notes: GUI field *Triangulate arena nodes* ("Yes" / "No").

.. _proc-conduct-anipose-triangulation-frame-restriction:

``frame_restriction``
   Range of frames to be triangulated.

   :Options:
      * ``[first, last]`` — two integers (``--frame-restriction FIRST LAST`` on the command line)
      * ``null`` — finds the least number of frames across all cameras and triangulates those
   :Default: ``null``
   :Notes: GUI field *Frame restriction*.

.. _proc-conduct-anipose-triangulation-excluded-views:

``excluded_views``
   List of camera serial numbers to be excluded from triangulation.

   :Options: list of camera serial numbers; empty excludes none
   :Default: ``[]``
   :Notes: GUI field *Excluded camera views*.

.. _proc-conduct-anipose-triangulation-display-progress-bool:

``display_progress_bool``
   Whether to display the progress bar in the terminal during execution.

   :Options: ``true`` or ``false``
   :Default: ``true``
   :Notes: GUI field *Display progress*.

.. _proc-conduct-anipose-triangulation-ransac-bool:

``ransac_bool``
   Whether to use RANSAC for triangulation.

   :Options: ``true`` or ``false``
   :Default: ``false``
   :Notes: GUI field *Ransac*.

.. _proc-conduct-anipose-triangulation-rigid-body-constraints:

``rigid_body_constraints``
   List of rigid body constraints to be used for triangulation.

   :Options: list of rigid body constraints; empty uses none
   :Default: ``[]``
   :Notes: GUI field *Rigid body constraints*.

.. _proc-conduct-anipose-triangulation-weak-body-constraints:

``weak_body_constraints``
   List of weak body constraints to be used for triangulation.

   :Options: list of weak body constraints; empty uses none
   :Default: ``[]``
   :Notes: GUI field *Weak body constraints*.

.. _proc-conduct-anipose-triangulation-smooth-scale:

``smooth_scale``
   Scale of the smoothing kernel.

   :Options: a number
   :Default: ``4``
   :Notes: GUI field *Smoothing scale*.

.. _proc-conduct-anipose-triangulation-weight-weak:

``weight_weak``
   Weight of the weak body constraints.

   :Options: a number
   :Default: ``4``
   :Notes: GUI field *Weak constraints weight*.

.. _proc-conduct-anipose-triangulation-weight-rigid:

``weight_rigid``
   Weight of the rigid body constraints.

   :Options: a number
   :Default: ``1``
   :Notes: GUI field *Rigid constraints weight*.

.. _proc-conduct-anipose-triangulation-reprojection-error-threshold:

``reprojection_error_threshold``
   Threshold for reprojection error in pixels.

   :Options: a number of pixels
   :Default: ``5``
   :Notes: GUI field *Reproject error threshold*.

.. _proc-conduct-anipose-triangulation-regularization-function:

``regularization_function``
   Regularization function to be used for triangulation.

   :Options:
      * ``'l1'`` — L1 regularization
      * ``'l2'`` — L2 regularization
   :Default: ``'l2'``
   :Notes: GUI field *Regularization*; the command line (``--regularization``) accepts ``l1`` or ``l2``.

.. _proc-conduct-anipose-triangulation-n-deriv-smooth:

``n_deriv_smooth``
   Number of derivatives to be used for smoothing.

   :Options: an integer
   :Default: ``1``
   :Notes: GUI field *Derivation kernel order*.

.. _proc-translate-rotate-metric-original-arena-file-loc:

``original_arena_file_loc``
   Directory containing the original arena 3D file.

   :Options: path (string) to a directory
   :Default: ``''`` (empty)
   :Notes: set by the same GUI field as ``calibration_file_loc``, *Tracking calibration / arena root directory*.

.. _proc-translate-rotate-metric-save-transformed-data:

``save_transformed_data``
   Whether to save the transformed data as "animal" or "arena".

   :Options:
      * ``'animal'`` — save the transformed animal points
      * ``'arena'`` — save the transformed arena points
   :Default: ``'animal'``
   :Notes: GUI field *Save transformation type*.

.. _proc-translate-rotate-metric-delete-original-h5:

``delete_original_h5``
   Whether to delete the original H5 file.

   :Options: ``true`` or ``false``
   :Default: ``true``
   :Notes: GUI field *Delete original .h5* ("Yes" / "No").

.. _proc-translate-rotate-metric-static-reference-len:

``static_reference_len``
   Length of the static reference in meters.

   :Options: a length in meters
   :Default: ``0.615``, the distance between two outer rail edges of two arena corners
   :Notes: GUI field *Static reference (m)*.

.. _proc-translate-rotate-metric-experimental-codes:

``experimental_codes``
   List of experimental codes associated with each session.

   :Options: list of experimental-code strings (decoding sheet below)
   :Default: ``[]``
   :Notes: entered next to each root directory in the GUI (*ExCode*).

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
    #. Write the broadband (2 kHz+) MEMMAP file for squeaks (runs locally or on cluster)
    #. Run DAS inference (runs on cluster)
    #. Curate DAS outputs (runs locally <2 min)
    #. Prepare USV assignment (runs locally <1 min)
    #. Run USV assignment (runs locally <5 min)
    #. Produce noise labels (runs locally, GPU recommended)
    #. Produce squeak labels (runs locally, GPU recommended)
    #. Generate per-USV spectrograms (runs on cluster)
    #. Generate USV masks — YOLO detection + SAM2 segmentation (runs on cluster)
    #. Compute USV acoustic features (runs on cluster)
    #. Infer QLVM latents, QLVM categories and squeak QLVM latents (runs on cluster)

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

The *Convert to single-ch files* step populates the *original* directory with single channel files of the entire recording. The *Crop AUDIO (to VIDEO)* step will crop the audio files to the video duration, and save them in the *cropped_to_video* subdirectory. Both steps require the usage of `sox <https://sourceforge.net/projects/sox/>`_. The *original* directory is later removed by the *A/V sync check* step (once synchronization passes the divergence tolerance for every audio device, each compared against the first sync camera after the cameras are checked to carry identical pulse sequences; if any comparison fails or cannot be made it is kept), not by this step; reduced to one channel below for brevity:

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

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section partially modifiable in the GUI, with the following parameters:

.. code-block:: json

    "multichannel_to_channel_audio": {
        "subprocess_timeout_hours": 2
      },
    "crop_wav_files_to_video": {
        "device_receiving_input": "both",
        "triggerbox_ch_receiving_input": 4,
        "subprocess_timeout_hours": 2
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`multichannel_to_channel_audio.subprocess_timeout_hours <proc-multichannel-to-channel-audio-subprocess-timeout-hours>`
     - ``2``
     - hours > 0
   * - :ref:`device_receiving_input <proc-crop-wav-files-to-video-device-receiving-input>`
     - ``'both'``
     - ``'m'``, ``'s'``, ``'both'``
   * - :ref:`triggerbox_ch_receiving_input <proc-crop-wav-files-to-video-triggerbox-ch-receiving-input>`
     - ``4``
     - microphone channel, ``1``–``12``
   * - :ref:`crop_wav_files_to_video.subprocess_timeout_hours <proc-crop-wav-files-to-video-subprocess-timeout-hours>`
     - ``2``
     - hours > 0

.. _proc-multichannel-to-channel-audio-subprocess-timeout-hours:

``multichannel_to_channel_audio.subprocess_timeout_hours``
   Hours each batch of sox channel-separation subprocesses (one per multichannel file, then the per-channel concatenation) may run before it is killed as hung and the step stops.

   :Options: a positive number of hours
   :Default: ``2``
   :Notes: used by *Convert to single-ch files*; not in the GUI.

.. _proc-crop-wav-files-to-video-device-receiving-input:

``device_receiving_input``
   USGH device receiving Loopbio Triggerbox input.

   :Options:
      * ``'m'`` — the master device (if using SYNC mode, this should be "m")
      * ``'s'`` — the slave device
      * ``'both'`` — both devices (the NO SYNC recording mode, see above)
   :Default: ``'both'``
   :Notes: GUI field *Trgbox-USGH device(s)*.

.. _proc-crop-wav-files-to-video-triggerbox-ch-receiving-input:

``triggerbox_ch_receiving_input``
   Microphone channel receiving Loopbio Triggerbox input.

   :Options: a channel number, ``1``–``12``
   :Default: ``4``
   :Notes: GUI field *Trgbox-USGH ch (1-12)*.

.. _proc-crop-wav-files-to-video-subprocess-timeout-hours:

``crop_wav_files_to_video.subprocess_timeout_hours``
   Hours the sox "cut to video" subprocesses may run before they are killed as hung and the step stops.

   :Options: a positive number of hours
   :Default: ``2``
   :Notes: used by *Crop AUDIO (to VIDEO)*; not in the GUI. Long sessions take a while, anything beyond this almost certainly means a hang.

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

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`stft_window_length_hop_size <proc-hpss-audio-stft-window-length-hop-size>`
     - ``[512, 128]``
     - two integers, ``[window length, hop size]``
   * - :ref:`kernel_size <proc-hpss-audio-kernel-size>`
     - ``[5, 60]``
     - two integers
   * - :ref:`hpss_power <proc-hpss-audio-hpss-power>`
     - ``4.0``
     - number > 0
   * - :ref:`margin <proc-hpss-audio-margin>`
     - ``[4, 1]``
     - two numbers

.. _proc-hpss-audio-stft-window-length-hop-size:

``stft_window_length_hop_size``
   STFT window length and hop size.

   :Options: two integers, ``[window length, hop size]`` in samples
   :Default: ``[512, 128]``
   :Notes: the first value is the STFT's ``n_fft`` / window length, the second its ``hop_length``.

.. _proc-hpss-audio-kernel-size:

``kernel_size``
   Harmonic-percussive source separation kernel size.

   :Options: two integers, passed to librosa's HPSS as ``kernel_size``
   :Default: ``[5, 60]``

.. _proc-hpss-audio-hpss-power:

``hpss_power``
   Harmonic-percussive source separation power.

   :Options: a positive number, passed to librosa's HPSS as ``power``
   :Default: ``4.0``

.. _proc-hpss-audio-margin:

``margin``
   Margin for harmonic-percussive source separation.

   :Options: two numbers, passed to librosa's HPSS as ``margin``
   :Default: ``[4, 1]``

Filter and concatenate to MEMMAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run these steps together, you need to list the root directories of interest, select *High-pass filter WAVs* and *Ultrasonic MEMMAP* (and *Broadband MEMMAP* for the 2 kHz+ file), click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_11.png
   :align: center
   :alt: Processing Step 11

.. raw:: html

   <br>

The purpose of the first two (*High-pass filter WAVs* and *Ultrasonic MEMMAP*) is to first high-pass filter each audio file (removing all lower frequencies) and then concatenate all channels into one `memory-mapped file  <https://docs.python.org/3/library/mmap.html>`_. The first step requires the usage of `sox <https://sourceforge.net/projects/sox/>`_. These processing steps populate the *hpss_filtered* directory with de-noised, high-pass filtered single channel files of the entire recording (reduced to one channel for brevity):

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

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section with the following parameters; none of them is in the GUI (which shows only the *High-pass filter WAVs* toggle), they are set in the settings file:

.. code-block:: json

    "filter_audio_files": {
        "filter_audio_format": "wav",
        "filter_dirs": [
          "hpss"
        ],
        "filter_freq_bounds": [
          0,
          30000
        ],
        "subprocess_timeout_hours": 2
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`filter_audio_format <proc-filter-audio-files-filter-audio-format>`
     - ``'wav'``
     - audio file extension
   * - :ref:`filter_dirs <proc-filter-audio-files-filter-dirs>`
     - ``["hpss"]``
     - list of folders under *audio*
   * - :ref:`filter_freq_bounds <proc-filter-audio-files-filter-freq-bounds>`
     - ``[0, 30000]``
     - ``[low, high]`` in Hz
   * - :ref:`subprocess_timeout_hours <proc-filter-audio-files-subprocess-timeout-hours>`
     - ``2``
     - hours > 0

.. _proc-filter-audio-files-filter-audio-format:

``filter_audio_format``
   Audio file format.

   :Options: an audio file extension (usually "wav")
   :Default: ``'wav'``

.. _proc-filter-audio-files-filter-dirs:

``filter_dirs``
   List of directories to be filtered.

   :Options: list of folder names (usually "hpss")
   :Default: ``["hpss"]``

.. _proc-filter-audio-files-filter-freq-bounds:

``filter_freq_bounds``
   Band filtered out, as [low, high] in Hz.

   :Options: two numbers, ``[low, high]`` in Hz
   :Default: ``[0, 30000]``, i.e. a 30 kHz high-pass

.. _proc-filter-audio-files-subprocess-timeout-hours:

``subprocess_timeout_hours``
   Hours the sox filtering subprocesses of one directory may run, all channels together, before they are killed as hung and the step stops.

   :Options: a positive number of hours
   :Default: ``2``

The *Ultrasonic MEMMAP* step takes its parameters from the adjacent ``concatenate_audio_files`` block (set in the settings file only; the GUI shows just the step's toggle):

.. code-block:: json

    "concatenate_audio_files": {
        "concatenate_audio_format": "wav",
        "concat_dirs": [
          "hpss_filtered"
        ]
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`concatenate_audio_format <proc-concatenate-audio-files-concatenate-audio-format>`
     - ``'wav'``
     - audio file extension
   * - :ref:`concat_dirs <proc-concatenate-audio-files-concat-dirs>`
     - ``["hpss_filtered"]``
     - list of folders under *audio*

.. _proc-concatenate-audio-files-concatenate-audio-format:

``concatenate_audio_format``
   Audio file format to concatenate.

   :Options: an audio file extension (usually "wav")
   :Default: ``'wav'``

.. _proc-concatenate-audio-files-concat-dirs:

``concat_dirs``
   List of directories whose single-channel files are concatenated into the memory-mapped file.

   :Options: list of folder names (usually "hpss_filtered")
   :Default: ``["hpss_filtered"]``

Broadband MEMMAP
~~~~~~~~~~~~~~~~
The *hpss_filtered* memory-mapped file above keeps only the band above 30 kHz, which is what DAS, the USV spectrograms/masks, loudness, the vocalocator and the figures were built and trained on. Lower-frequency vocalizations (squeaks, 2-30 kHz) need a second, broadband file. Every squeak-side step reads it: *Produce noise labels*, *Produce squeak labels*, the squeak QLVM embedding, the squeak spectrogram store and the training of the noise and vocal-class models, so it must exist before they run. Selecting *Broadband MEMMAP* (or running ``broadband-filter-audio``) writes it straight from the full-band HPSS files in *audio/hpss* (the ``source_dir`` setting, set in the settings file only), without storing any per-channel files. The GUI shows only the step's toggle; its removed band ``filter_freq_bounds`` (usually [0, 2000]) is set in the settings file. For every channel it:

1. finds the narrow electrical line-noise tones in a few search bands (a line near 8 kHz and its 16 kHz harmonic, a line near 1.65 kHz, and a comb of lines about 30 Hz apart around 2.09-2.15 kHz, some of them close doublets). The exact frequencies follow each recording device's clock (e.g. 8000.17 Hz on the master and 8000.67 Hz on the slave device of one session; 2091.7 / 2122.5 / 2152.9 Hz on the master and 2090.05 / 2120.06 / 2150.05 Hz on the slave channels of the same session), so they are estimated per channel. The spectrum is whitened by its running median, and up to six peaks per band are kept when they stand at least 8 dB above that local floor;
2. subtracts each kept tone as a sinusoid whose amplitude and phase may drift slowly (complex demodulation over 0.5 s blocks, median-smoothed over 9 blocks);
3. high-passes the result with a linear-phase (zero-delay) Kaiser FIR equivalent to sox ``sinc -t 1000 2k`` (-6 dB at 2 kHz, at least 120 dB down at and below 1.5 kHz, flat from 2.5 kHz), in floating point and without dither, then rounds to int16.

The channels go into one memory-mapped file whose columns follow the sorted file names (the same order as the *hpss_filtered* file), next to a *line_noise.json* report (per channel: the tones searched, found and removed, with frequency, height and amplitude; the filter and its measured response; the source files and their sizes; each column's low-band variance, see below; the code version):

.. parsed-literal::

    ├── 20250430_145017
    │   ├── audio
    │   │   ├── **broadband_filtered**
    │   │   │   ├── **250430145009_concatenated_audio_broadband_filtered_250000_299885168_24_int16.mmap**
    │   │   │   ├── **line_noise.json**
    │   │   ├── hpss
    │   │   │   ...
    │   │   ├── hpss_filtered
    │   │   │   ...

Every reader of the concatenated audio asks for its band explicitly (``os_utils.find_audio_mmap(root, 'ultrasonic')`` or ``find_audio_mmap(root, 'broadband')``): the file is looked up in its exact folder with its exact name, and the lookup fails unless exactly one file matches, so no USV step can read the broadband file by accident. The step works in short time chunks across all channels (memory stays at a few hundred MB), writes to a hidden temporary file that is renamed only when complete, and skips a session whose broadband file and report already exist and still match the source files and settings, so it can be re-run or resumed safely. To backfill many sessions, ``broadband-filter-audio-batch`` runs it over a list of sessions in parallel, with a log and a per-session report table.

The step takes its parameters from the ``broadband_filter_audio`` block of */usv-playpen/_parameter_settings/processing_settings.json*:

.. code-block:: json

    "broadband_filter_audio": {
        "source_dir": "hpss",
        "source_glob": "*_cropped_to_video_hpss.wav",
        "filter_freq_bounds": [0, 2000],
        "transition_width_hz": 1000,
        "stopband_attenuation_db": 120,
        "line_noise_search_bands_hz": [[7990, 8010], [15980, 16020], [1640, 1665], [2080, 2165]],
        "line_noise_min_height_db": 8.0,
        "line_noise_max_tones_per_band": 6,
        "line_noise_min_separation_hz": 0.25,
        "line_noise_floor_window_hz": 4.0,
        "line_noise_estimation_windows": 8,
        "line_noise_estimation_window_s": 20.0,
        "line_noise_block_s": 0.5,
        "line_noise_smoothing_blocks": 9,
        "chunk_s": 10.0,
        "n_threads": 4,
        "low_band_variance_windows": 25,
        "low_band_variance_window_s": 0.25,
        "low_band_variance_seed": 0
    }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`source_dir <proc-broadband-filter-audio-source-dir>`
     - ``'hpss'``
     - folder under *audio*
   * - :ref:`source_glob <proc-broadband-filter-audio-source-glob>`
     - ``'*_cropped_to_video_hpss.wav'``
     - file pattern
   * - :ref:`filter_freq_bounds <proc-broadband-filter-audio-filter-freq-bounds>`
     - ``[0, 2000]``
     - ``[0, high]`` in Hz
   * - :ref:`transition_width_hz <proc-broadband-filter-audio-transition-width-hz>`
     - ``1000``
     - Hz > 0
   * - :ref:`stopband_attenuation_db <proc-broadband-filter-audio-stopband-attenuation-db>`
     - ``120``
     - dB > 0
   * - :ref:`line_noise_search_bands_hz <proc-broadband-filter-audio-line-noise-search-bands-hz>`
     - see JSON above
     - list of ``[lo, hi]`` bands in Hz, each at most 100 Hz wide
   * - :ref:`line_noise_min_height_db <proc-broadband-filter-audio-line-noise-min-height-db>`
     - ``8.0``
     - dB
   * - :ref:`line_noise_max_tones_per_band <proc-broadband-filter-audio-line-noise-max-tones-per-band>`
     - ``6``
     - integer
   * - :ref:`line_noise_min_separation_hz <proc-broadband-filter-audio-line-noise-min-separation-hz>`
     - ``0.25``
     - Hz
   * - :ref:`line_noise_floor_window_hz <proc-broadband-filter-audio-line-noise-floor-window-hz>`
     - ``4.0``
     - Hz
   * - :ref:`line_noise_estimation_windows <proc-broadband-filter-audio-line-noise-estimation>`
     - ``8``
     - integer
   * - :ref:`line_noise_estimation_window_s <proc-broadband-filter-audio-line-noise-estimation>`
     - ``20.0``
     - seconds
   * - :ref:`line_noise_block_s <proc-broadband-filter-audio-line-noise-demodulation>`
     - ``0.5``
     - seconds
   * - :ref:`line_noise_smoothing_blocks <proc-broadband-filter-audio-line-noise-demodulation>`
     - ``9``
     - positive odd integer
   * - :ref:`chunk_s <proc-broadband-filter-audio-chunk-s>`
     - ``10.0``
     - seconds
   * - :ref:`n_threads <proc-broadband-filter-audio-n-threads>`
     - ``4``
     - integer
   * - :ref:`low_band_variance_windows <proc-broadband-filter-audio-low-band-variance>`
     - ``25``
     - integer
   * - :ref:`low_band_variance_window_s <proc-broadband-filter-audio-low-band-variance>`
     - ``0.25``
     - seconds
   * - :ref:`low_band_variance_seed <proc-broadband-filter-audio-low-band-variance-seed>`
     - ``0``
     - integer seed

.. _proc-broadband-filter-audio-source-dir:

``source_dir``
   Folder under *audio* holding the full-band single-channel files.

   :Options: a folder name under *audio* (usually "hpss")
   :Default: ``'hpss'``

.. _proc-broadband-filter-audio-source-glob:

``source_glob``
   File pattern of the source files.

   :Options: a glob pattern (usually "\*_cropped_to_video_hpss.wav")
   :Default: ``'*_cropped_to_video_hpss.wav'``
   :Notes: stray files in the folder are ignored.

.. _proc-broadband-filter-audio-filter-freq-bounds:

``filter_freq_bounds``
   Band filtered out, as [low, high] in Hz, the same convention as ``filter_audio_files``.

   :Options: ``[low, high]`` in Hz; the filter is a high-pass, so low must be 0, and high is its -6 dB point
   :Default: ``[0, 2000]``
   :Notes: ``line_noise.json`` records the high value as ``highpass_cutoff_hz``.

.. _proc-broadband-filter-audio-transition-width-hz:

``transition_width_hz``
   Width of the high-pass transition band, centred on the cutoff.

   :Options: a width in Hz
   :Default: ``1000`` (usually 1000)

.. _proc-broadband-filter-audio-stopband-attenuation-db:

``stopband_attenuation_db``
   Stopband attenuation of the Kaiser design.

   :Options: an attenuation in dB
   :Default: ``120`` (usually 120)

.. _proc-broadband-filter-audio-line-noise-search-bands-hz:

``line_noise_search_bands_hz``
   Frequency bands searched for line-noise tones.

   :Options: a list of ``[lo, hi]`` bands in Hz, each at most 100 Hz wide
   :Default: ``[[7990, 8010], [15980, 16020], [1640, 1665], [2080, 2165]]``

.. _proc-broadband-filter-audio-line-noise-min-height-db:

``line_noise_min_height_db``
   Minimum tone height above the local floor (dB) for the tone to be removed.

   :Options: a height in dB
   :Default: ``8.0``

.. _proc-broadband-filter-audio-line-noise-max-tones-per-band:

``line_noise_max_tones_per_band``
   Maximum number of tones removed per search band.

   :Options: a positive integer
   :Default: ``6``

.. _proc-broadband-filter-audio-line-noise-min-separation-hz:

``line_noise_min_separation_hz``
   Minimum distance (Hz) between two removed tones of a band.

   :Options: a distance in Hz
   :Default: ``0.25``

.. _proc-broadband-filter-audio-line-noise-floor-window-hz:

``line_noise_floor_window_hz``
   Width (Hz) of the running median that estimates the local floor.

   :Options: a width in Hz
   :Default: ``4.0``

.. _proc-broadband-filter-audio-line-noise-estimation:

``line_noise_estimation_windows`` / ``line_noise_estimation_window_s``
   Number and length (s) of the windows the tones are estimated from.

   :Options:
      * ``line_noise_estimation_windows`` — the number of windows, a positive integer
      * ``line_noise_estimation_window_s`` — the length of each window, in seconds
   :Default: ``8`` and ``20.0``

.. _proc-broadband-filter-audio-line-noise-demodulation:

``line_noise_block_s`` / ``line_noise_smoothing_blocks``
   Demodulation block length (s) and running-median length (blocks, odd) of the tone subtraction.

   :Options:
      * ``line_noise_block_s`` — the demodulation block length, in seconds
      * ``line_noise_smoothing_blocks`` — the running-median length, in blocks; must be odd
   :Default: ``0.5`` and ``9``

.. _proc-broadband-filter-audio-chunk-s:

``chunk_s``
   Length of the processing chunks (s).

   :Options: a length in seconds
   :Default: ``10.0``
   :Notes: does not change the result.

.. _proc-broadband-filter-audio-n-threads:

``n_threads``
   Threads the channels of a chunk are spread over.

   :Options: a positive integer
   :Default: ``4``
   :Notes: does not change the result.

.. _proc-broadband-filter-audio-low-band-variance:

``low_band_variance_windows`` / ``low_band_variance_window_s``
   Number and length (s) of the random windows the per-column *low-band variance* is estimated from.

   :Options:
      * ``low_band_variance_windows`` — the number of random windows, a positive integer
      * ``low_band_variance_window_s`` — the length of each window, in seconds
   :Default: ``25`` and ``0.25`` (default 25 windows of 0.25 s)
   :Notes:
      **Why it is stored:** the audio classifiers (*Produce noise labels*, *Produce squeak labels*, the squeak QLVM embedding and the squeak spectrogram store) average the channels with weights equal to each channel's **full-band** audio variance, which is what their models were trained on; 88 % of that variance lies below the 2 kHz high-pass, so the broadband audio alone gives different weights, and the noise model's recall falls from 0.99 to 0.73 on it (retraining on the broadband audio does not recover the separation).

      **How it is used:** since full-band variance = low-band variance + high-band variance and the high band is the broadband audio, the report stores each column's typical low-band variance (the median over the random windows of the source wav's variance minus the broadband column's), and the classifiers add it to a window's broadband variance, which reproduces the full-band weights (Spearman 0.997).

      **Effect:** through this path the noise model keeps its precision and recall (0.988 / 0.993 on 1,118 labelled segments, scores r 0.984 with the wav input) at a cost of about 11 more real vocalizations per 10,000 segments, 9 of them from the tone removal alone, and the vocal-class and squeak QLVM outputs agree with the wav input at 99.98 % or better.

      **Older reports:** reports written before this field existed get it from ``add-broadband-low-band-variance`` (a few seconds of audio per session; the memmap is not rewritten), and a classifier refuses a report without it.

.. _proc-broadband-filter-audio-low-band-variance-seed:

``low_band_variance_seed``
   Seed of the random window positions the low-band variance is estimated from.

   :Options: an integer seed
   :Default: ``0``
   :Notes: recorded in the report as ``seed``; it keeps the estimate reproducible run to run (independent draws of the windows agree at Spearman 0.997, values within about 5 %).

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

.. code-block:: json

    "das_command_line_inference": {
        "das_conda_env_name": "das",
        "das_model_directory": "/mnt/falkner/Bartul/DAS/model_2024-03-25",
        "model_name_base": "20240325_073951",
        "output_file_type": "csv",
        "segment_confidence_threshold": 0.5,
        "segment_minlen": 0.015,
        "segment_fillgap": 0.015,
        "subprocess_timeout_hours": 12
      },

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`das_conda_env_name <proc-das-command-line-inference-das-conda-env-name>`
     - ``"das"``
     - any conda environment name
   * - :ref:`das_model_directory <proc-das-command-line-inference-das-model-directory>`
     - path (see JSON above)
     - path to a directory
   * - :ref:`model_name_base <proc-das-command-line-inference-model-name-base>`
     - ``"20240325_073951"``
     - model base name (date)
   * - :ref:`output_file_type <proc-das-command-line-inference-output-file-type>`
     - ``"csv"``
     - ``"csv"``, ``"h5"``
   * - :ref:`segment_confidence_threshold <proc-das-command-line-inference-segment-confidence-threshold>`
     - ``0.5``
     - confidence threshold
   * - :ref:`segment_minlen <proc-das-command-line-inference-segment-minlen>`
     - ``0.015``
     - seconds
   * - :ref:`segment_fillgap <proc-das-command-line-inference-segment-fillgap>`
     - ``0.015``
     - seconds
   * - :ref:`subprocess_timeout_hours <proc-das-command-line-inference-subprocess-timeout-hours>`
     - ``12``
     - hours

.. _proc-das-command-line-inference-das-conda-env-name:

``das_conda_env_name``
   Name of the local conda environment used for running DAS inference.

   :Options: any string naming a local conda environment that has DAS installed
   :Default: ``"das"``
   :Notes: settings / CLI only; not exposed in the GUI.

.. _proc-das-command-line-inference-das-model-directory:

``das_model_directory``
   Directory containing the trained DAS model.

   :Options: path (string) to a directory
   :Default: ``/mnt/falkner/Bartul/DAS/model_2024-03-25``
   :Notes: set in the GUI through the *DAS model directory* field; the model path handed to DAS joins this directory with ``model_name_base``.

.. _proc-das-command-line-inference-model-name-base:

``model_name_base``
   Base name (date) of the trained DAS model.

   :Options: any string; the base name of the model files inside ``das_model_directory``
   :Default: ``"20240325_073951"``
   :Notes: GUI field *DAS model base (timestamp)*.

.. _proc-das-command-line-inference-output-file-type:

``output_file_type``
   Output file type of the DAS annotations.

   :Options:
      * ``"csv"`` — CSV annotation files
      * ``"h5"`` — HDF5 annotation files
   :Default: ``"csv"``
   :Notes: GUI field *Inference output file type*.

.. _proc-das-command-line-inference-segment-confidence-threshold:

``segment_confidence_threshold``
   Confidence threshold for segmenting vocalizations.

   :Options: a number (passed to ``das predict`` as ``--segment-thres``)
   :Default: ``0.5``
   :Notes: GUI field *DAS confidence threshold*.

.. _proc-das-command-line-inference-segment-minlen:

``segment_minlen``
   Minimum length of segments to be considered vocalizations.

   :Options: a number of seconds (passed to ``das predict`` as ``--segment-minlen``)
   :Default: ``0.015``
   :Notes: GUI field *USV min duration (s)*.

.. _proc-das-command-line-inference-segment-fillgap:

``segment_fillgap``
   Maximum gap between segments to be joined into a single vocalization.

   :Options: a number of seconds (passed to ``das predict`` as ``--segment-fillgap``)
   :Default: ``0.015``
   :Notes: GUI field *Fill gaps shorter than (s)*.

.. _proc-das-command-line-inference-subprocess-timeout-hours:

``subprocess_timeout_hours``
   Hours one per-file DAS inference subprocess may run before it is killed as hung and the step stops.

   :Options: a number of hours
   :Default: ``12``
   :Notes: inference on a long recording can take hours on CPU, so the budget is generous, but a lost GPU or a deadlock is still caught; settings / CLI only.

Curate DAS outputs
~~~~~~~~~~~~~~~~~~
As explained above, DAS is run on every channel separately, such that a need arises to systematize different channel detections in one singular table. This code identifies the same detections across different channels and creates a single CSV file with the start and end times of each detected vocalization.

Detections are combined by a greedy interval union: segments are sorted by start time and any two that overlap are merged, transitively, so one USV spans the full extent of every per-channel detection that belongs to it. Boundaries therefore follow the outermost channel that heard the USV, which keeps the faint onsets and offsets registered by only the nearest few microphones (Phase-4 noise rejection, below, decides which detections to keep). A coverage-watershed merge was used between August 2026 and this release; it trimmed each USV to the span where a fixed fraction of its peak channel count still agreed, which clipped exactly those faint edges, and it has been reverted.

Merged intervals longer than ``max_usv_duration_s`` are rejected before the Phase-4 checks see them. Those checks ask whether a detection correlates across channels, which broadband noise satisfies easily: session ``20240229_163242`` carried six "USVs" with a median duration of 3,074 ms, on all 24 channels, and every one survived correlation and coherence. Duration is the discriminator correlation cannot supply -- no mouse USV lasts a second. With the gate in place that session yields no vocalizations at all, which is the correct reading of a noise-only recording.

One failure mode of the union is corrected before the summary is written. Because the outermost edge wins, a single channel that fuses a run of USVs into one long detection drags every other channel's boundaries out with it. In session ``20250919_145712`` twenty-three of twenty-four channels resolved seven separate USVs between 0.9 s and 1.9 s while one channel emitted a single 897 ms detection, and the merged USV became 901 ms; the fused detection is in the raw annotation, so re-inference does not change it, and the channel is otherwise healthy (974 detections against a cohort median of 968, and only two of them anomalous) so excluding it session-wide would discard a good microphone.

An interval is therefore re-merged from the remaining channels when it is longer than ``consensus_remerge_min_duration_s``, at most ``consensus_remerge_max_dissenting_channels`` of its channels have a segment exceeding ``consensus_remerge_span_factor`` times the median longest-segment, at least ``consensus_remerge_min_agreeing_channels`` remain, and that re-merge actually yields more than one interval. A channel set aside for the boundary is still credited in ``chs_detected`` wherever its own segment covers a sub-interval, since it did hear the USV and only failed to separate it. The last condition is what keeps dense-but-genuine runs intact: where the long channel agrees with the others, the consensus re-merge reproduces one interval and nothing changes.

A cut must also sit in a gap the segmenter could have produced. Because DAS closes gaps under 15 ms within a channel, anything finer in the merged output comes from taking the union across channels and cutting it again, where different channels' detections stop at slightly different instants; left alone it puts boundaries inside USVs, as at ``20250928_172408`` 565.074 s, cut into 18 ms and 133 ms pieces divided by 0.1 ms. Matching ``consensus_remerge_min_gap_s`` to the segmenter's own resolution removed that class -- 54 of 232 split decisions across twelve sessions, each one a merge rather than a new cut -- while staying discriminating: at ``20250928_175135`` 303.130 s a 17.1 ms gap survives and a 4.9 ms one closes. Boundary errors above that width are not separable from correct splits on the annotations alone: a confirmed correct split at 16.4 ms and a confirmed false one at 17.1 ms come from the same session.

The correction is applied recursively, because a single pass judges dissent against the median longest-segment of the whole interval: in ``20240311_143803`` at 300.816 s a 779 ms interval had a 96 ms median, so the channels fusing two 30-40 ms USVs at 102-107 ms read as ordinary and their fusions survived. Re-testing each corrected sub-interval against its own median exposes them, which is why ``consensus_remerge_min_duration_s`` is 80 ms rather than the interval-level figure that would protect ordinary USVs.

Only the INTERNAL cuts come from the consensus: the outer edges stay the union's, since setting a channel aside also removes its contribution to the interval's own start and stop, and that contribution is the faint edge only the nearest microphone registered. Recursion is concentrated where USVs are packed tightly -- across eight sessions, 736 of 895 corrected intervals came from one dense recording and most others contributed one or two. Reviewed against spectrograms over 13 sampled cases, one was a false split. Far narrower than the coverage-watershed merge, which reshaped every USV.

Microphones recorded as hardware-compromised for a session -- the ``excluded_channels`` list under ``Equipment -> audio_Avisoft`` in the session's ``*_metadata.yaml`` -- are skipped outright: their annotation files never enter the merge. The field is absent on healthy sessions (no exclusions).

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

.. code-block:: json

     "summarize_das_findings": {
        "filter_putative_noise_bool": true,
        "len_win_signal": 512,
        "low_freq_cutoff": 30000,
        "noise_corr_cutoff_min": 0.3,
        "coherence_cutoff_min": 0.2,
        "coherence_channel_count": 3,
        "consensus_remerge_bool": true,
        "consensus_remerge_min_duration_s": 0.08,
        "consensus_remerge_span_factor": 2.0,
        "consensus_remerge_max_dissenting_channels": 2,
        "consensus_remerge_min_agreeing_channels": 3,
        "consensus_remerge_max_depth": 4,
        "consensus_remerge_min_gap_s": 0.015,
        "consensus_remerge_rescue_dissenting_channels": 4,
        "max_usv_duration_bool": true,
        "max_usv_duration_s": 1.0,
        "seam_repair_bool": false,
        "seam_repair_legacy_stride_samples": 8128,
        "seam_repair_max_rung": 6,
        "seam_repair_ladder_tolerance_samples": 2,
        "seam_repair_width_tolerance_below_ms": 3.0,
        "seam_repair_width_tolerance_above_ms": 1.0,
        "seam_repair_raw_stop_tolerance_s": 0.01,
        "seam_repair_snippet_margin_s": 0.05,
        "seam_repair_max_boundary_shift_s": 0.034,
        "seam_repair_subprocess_timeout_hours": 4,
        "noise_histogram_bins": 20,
        "noise_histogram_figure_size": [24, 4],
        "noise_histogram_face_color": "#BBD5E8",
        "noise_histogram_edge_color": "#202020",
        "noise_histogram_cutoff_line_color": "#202020"
     }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`filter_putative_noise_bool <proc-summarize-das-findings-filter-putative-noise-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`len_win_signal <proc-summarize-das-findings-len-win-signal>`
     - ``512``
     - STFT window length
   * - :ref:`low_freq_cutoff <proc-summarize-das-findings-low-freq-cutoff>`
     - ``30000``
     - Hz
   * - :ref:`noise_corr_cutoff_min <proc-summarize-das-findings-noise-corr-cutoff-min>`
     - ``0.3``
     - absolute correlation cutoff
   * - :ref:`coherence_cutoff_min <proc-summarize-das-findings-coherence-cutoff-min>`
     - ``0.2``
     - absolute coherence cutoff
   * - :ref:`coherence_channel_count <proc-summarize-das-findings-coherence-channel-count>`
     - ``3``
     - number of channels
   * - :ref:`consensus_remerge_bool <proc-summarize-das-findings-consensus-remerge-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`consensus_remerge_min_duration_s <proc-summarize-das-findings-consensus-remerge-min-duration-s>`
     - ``0.08``
     - seconds
   * - :ref:`consensus_remerge_span_factor <proc-summarize-das-findings-consensus-remerge-span-factor>`
     - ``2.0``
     - multiple of the median longest-segment
   * - :ref:`consensus_remerge_max_dissenting_channels <proc-summarize-das-findings-consensus-remerge-max-dissenting-channels>`
     - ``2``
     - number of channels
   * - :ref:`consensus_remerge_min_agreeing_channels <proc-summarize-das-findings-consensus-remerge-min-agreeing-channels>`
     - ``3``
     - number of channels
   * - :ref:`consensus_remerge_max_depth <proc-summarize-das-findings-consensus-remerge-max-depth>`
     - ``4``
     - integer; ``0`` = single pass
   * - :ref:`consensus_remerge_min_gap_s <proc-summarize-das-findings-consensus-remerge-min-gap-s>`
     - ``0.015``
     - seconds
   * - :ref:`consensus_remerge_rescue_dissenting_channels <proc-summarize-das-findings-consensus-remerge-rescue-dissenting-channels>`
     - ``4``
     - number of channels
   * - :ref:`max_usv_duration_bool <proc-summarize-das-findings-max-usv-duration-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`max_usv_duration_s <proc-summarize-das-findings-max-usv-duration-s>`
     - ``1.0``
     - seconds
   * - :ref:`seam_repair_bool <proc-summarize-das-findings-seam-repair-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`seam_repair_legacy_stride_samples <proc-summarize-das-findings-seam-repair-legacy-stride-samples>`
     - ``8128``
     - samples
   * - :ref:`seam_repair_max_rung <proc-summarize-das-findings-seam-repair-max-rung>`
     - ``6``
     - stride multiple
   * - :ref:`seam_repair_ladder_tolerance_samples <proc-summarize-das-findings-seam-repair-ladder-tolerance-samples>`
     - ``2``
     - samples
   * - :ref:`seam_repair_width_tolerance_below_ms <proc-summarize-das-findings-seam-repair-width-tolerance-below-ms>`
     - ``3.0``
     - ms
   * - :ref:`seam_repair_width_tolerance_above_ms <proc-summarize-das-findings-seam-repair-width-tolerance-above-ms>`
     - ``1.0``
     - ms
   * - :ref:`seam_repair_raw_stop_tolerance_s <proc-summarize-das-findings-seam-repair-raw-stop-tolerance-s>`
     - ``0.01``
     - seconds
   * - :ref:`seam_repair_snippet_margin_s <proc-summarize-das-findings-seam-repair-snippet-margin-s>`
     - ``0.05``
     - seconds
   * - :ref:`seam_repair_max_boundary_shift_s <proc-summarize-das-findings-seam-repair-max-boundary-shift-s>`
     - ``0.034``
     - seconds
   * - :ref:`seam_repair_subprocess_timeout_hours <proc-summarize-das-findings-seam-repair-subprocess-timeout-hours>`
     - ``4``
     - hours
   * - :ref:`noise_histogram_bins <proc-summarize-das-findings-noise-histogram-bins>`
     - ``20``
     - number of bins
   * - :ref:`noise_histogram_figure_size <proc-summarize-das-findings-noise-histogram-figure-size>`
     - ``[24, 4]``
     - width, height in inches
   * - :ref:`noise_histogram_face_color <proc-summarize-das-findings-noise-histogram-colors>`
     - ``#BBD5E8``
     - hex colour
   * - :ref:`noise_histogram_edge_color <proc-summarize-das-findings-noise-histogram-colors>`
     - ``#202020``
     - hex colour
   * - :ref:`noise_histogram_cutoff_line_color <proc-summarize-das-findings-noise-histogram-cutoff-line-color>`
     - ``#202020``
     - hex colour

.. _proc-summarize-das-findings-filter-putative-noise-bool:

``filter_putative_noise_bool``
   Whether to run the Phase-4 amplitude/spectrogram noise rejection.

   :Options:
      * ``true`` — run the noise rejection
      * ``false`` — every merged detection is kept and the summary CSV is written as-is (peak/mean amplitude channels left at 0)
   :Default: ``true``

.. _proc-summarize-das-findings-len-win-signal:

``len_win_signal``
   STFT window length.

   :Options: a positive integer
   :Default: ``512``

.. _proc-summarize-das-findings-low-freq-cutoff:

``low_freq_cutoff``
   Frequency cutoff for filtering (in Hz).

   :Options: a number of Hz
   :Default: ``30000``

.. _proc-summarize-das-findings-noise-corr-cutoff-min:

``noise_corr_cutoff_min``
   Absolute cutoff on the cross-channel spectral correlation across a candidate's DAS-detected channels; a multi-channel candidate below it is dropped.

   :Options: a correlation value
   :Default: ``0.3``
   :Notes: absolute rather than session-relative, because percentile rules overshoot on clean candidate pools and undershoot on noise-dominated ones (empirically, cohort-wide, noise sits at correlation ~0.25 and verified real USVs at >=0.30).

.. _proc-summarize-das-findings-coherence-cutoff-min:

``coherence_cutoff_min``
   Absolute cutoff on the spatial coherence -- the mean pairwise spectral correlation across the loudest in-band channels of the whole array, detection status ignored. Candidates below the cutoff are dropped.

   :Options: a coherence value
   :Default: ``0.2``
   :Notes: a genuine USV dominates the in-band soundscape at its moment, so its loudest channels share one time-frequency pattern; a localized artifact's pattern appears nowhere else. Single-channel candidates (no defined detected-channel correlation) are gated by coherence alone.

.. _proc-summarize-das-findings-coherence-channel-count:

``coherence_channel_count``
   Number of loudest in-band channels the spatial coherence is computed over.

   :Options: a positive integer
   :Default: ``3``

.. _proc-summarize-das-findings-consensus-remerge-bool:

``consensus_remerge_bool``
   Whether to re-merge an interval whose extent is set by a channel contradicting the majority.

   :Options:
      * ``true`` — re-merge such intervals from the remaining channels
      * ``false`` — the union is taken exactly as detected
   :Default: ``true``

.. _proc-summarize-das-findings-consensus-remerge-min-duration-s:

``consensus_remerge_min_duration_s``
   Only merged intervals longer than this (in s) are examined, so ordinary USVs are never reconsidered.

   :Options: a number of seconds
   :Default: ``0.08``

.. _proc-summarize-das-findings-consensus-remerge-span-factor:

``consensus_remerge_span_factor``
   A contributing channel counts as dissenting when its longest segment exceeds this multiple of the median longest-segment across the contributing channels.

   :Options: a positive number (a multiple)
   :Default: ``2.0``

.. _proc-summarize-das-findings-consensus-remerge-max-dissenting-channels:

``consensus_remerge_max_dissenting_channels``
   Fire only when at most this many channels dominate; more than that is not a lone dissenter.

   :Options: a number of channels
   :Default: ``2``

.. _proc-summarize-das-findings-consensus-remerge-min-agreeing-channels:

``consensus_remerge_min_agreeing_channels``
   Fire only when at least this many channels remain once the dissenters are set aside.

   :Options: a number of channels
   :Default: ``3``

.. _proc-summarize-das-findings-consensus-remerge-max-depth:

``consensus_remerge_max_depth``
   How many times a corrected interval may itself be re-examined.

   :Options: a non-negative integer; ``0`` reproduces the single-pass behaviour
   :Default: ``4``

.. _proc-summarize-das-findings-consensus-remerge-min-gap-s:

``consensus_remerge_min_gap_s``
   Narrowest gap a cut may sit in (in s).

   :Options: a number of seconds
   :Default: ``0.015``
   :Notes: DAS is run with ``--segment-fillgap 0.015`` and so closes anything under 15 ms within a channel (measured: 99.87% of 221,522 within-channel gaps are >= 15 ms, 1st percentile 21 ms), which means a finer gap in the merged output was manufactured by the union rather than found by the segmenter.

.. _proc-summarize-das-findings-consensus-remerge-rescue-dissenting-channels:

``consensus_remerge_rescue_dissenting_channels``
   Dissent limit used only for an interval that would otherwise be deleted for exceeding ``max_usv_duration_s``.

   :Options: a number of channels
   :Default: ``4``
   :Notes: relaxing it everywhere is not justified (over twelve intervals judged against spectrograms a limit of 4 was better in six and worse in five), but applied on that path alone it changed 0 of 13,851 intervals across eleven sessions while rescuing 1,094 ms of genuine USVs.

.. _proc-summarize-das-findings-max-usv-duration-bool:

``max_usv_duration_bool``
   Whether to reject merged intervals longer than ``max_usv_duration_s`` before the Phase-4 noise checks run.

   :Options: ``true``, ``false``
   :Default: ``true``

.. _proc-summarize-das-findings-max-usv-duration-s:

``max_usv_duration_s``
   Longest a merged interval may be (in s) before it is rejected as noise.

   :Options: a number of seconds
   :Default: ``1.0``
   :Notes: measured over 439,301 USVs in 246 sessions the median is 86 ms, p99 371 ms and p99.99 896 ms, so the ``1.0`` default is past any plausible USV or tightly-overlapping cluster and removes 0.005% of intervals.

.. _proc-summarize-das-findings-seam-repair-bool:

``seam_repair_bool``
   Whether to run the post-summary seam check-and-repair (below).

   :Options:
      * ``true`` — run the seam check-and-repair
      * ``false`` — seam-snapped boundaries are left as detected
   :Default: ``false``
   :Notes: the artifact it corrects is produced by the legacy non-overlapping model (``nb_hist`` 8192, ``stride`` 8128); annotations from the overlap-tiled model carry no seam-ladder fingerprints, so the repair has nothing to act on.

.. _proc-summarize-das-findings-seam-repair-legacy-stride-samples:

``seam_repair_legacy_stride_samples``
   Window-stitching stride (in samples) of the legacy non-overlapping DAS model whose seam artifact is checked for.

   :Options: a number of samples
   :Default: ``8128`` (the lab's 2024-03-25 model at 250 kHz)

.. _proc-summarize-das-findings-seam-repair-max-rung:

``seam_repair_max_rung``
   Highest stride multiple tested for the seam-ladder fingerprint.

   :Options: a positive integer
   :Default: ``6``

.. _proc-summarize-das-findings-seam-repair-ladder-tolerance-samples:

``seam_repair_ladder_tolerance_samples``
   Maximum deviation (in samples) from the exact ``k * stride + 1`` ladder fingerprint for a raw-annotation gap to count as seam-snapped.

   :Options: a number of samples
   :Default: ``2``

.. _proc-summarize-das-findings-seam-repair-width-tolerance-below-ms:

``seam_repair_width_tolerance_below_ms``
   Merged-gap width gate reach below each stride multiple (in ms).

   :Options: a number of ms
   :Default: ``3.0``
   :Notes: merged gaps can sit slightly off the exact raw ladder values because the union takes the outermost edge across channels.

.. _proc-summarize-das-findings-seam-repair-width-tolerance-above-ms:

``seam_repair_width_tolerance_above_ms``
   Merged-gap width gate reach above each stride multiple (in ms).

   :Options: a number of ms
   :Default: ``1.0``

.. _proc-summarize-das-findings-seam-repair-raw-stop-tolerance-s:

``seam_repair_raw_stop_tolerance_s``
   Maximum distance (in s) between a merged pair's stop and a raw ladder-gap stop for the pair to count as corroborated.

   :Options: a number of seconds
   :Default: ``0.01``

.. _proc-summarize-das-findings-seam-repair-snippet-margin-s:

``seam_repair_snippet_margin_s``
   Audio margin (in s) excised around each flagged pair for re-detection.

   :Options: a number of seconds
   :Default: ``0.05``

.. _proc-summarize-das-findings-seam-repair-max-boundary-shift-s:

``seam_repair_max_boundary_shift_s``
   Maximum outward correction (in s) permitted per USV edge.

   :Options: a number of seconds
   :Default: ``0.034``
   :Notes: mechanistically one legacy stride.

.. _proc-summarize-das-findings-seam-repair-subprocess-timeout-hours:

``seam_repair_subprocess_timeout_hours``
   Hours the seam-repair snippet re-detection subprocess may run before it is killed as hung.

   :Options: a number of hours
   :Default: ``4``
   :Notes: snippets are sub-second, so even hundreds finish in minutes on CPU; its progress, failure and timeout messages are reported through the same GUI / CLI log as the rest of the step.

.. _proc-summarize-das-findings-noise-histogram-bins:

``noise_histogram_bins``
   Number of bins over [-1, 1] of the two histograms (detected-channel correlation, spatial coherence) in the *<session>_usv_signal_correlation_histogram* figure the noise rejection saves under *audio*.

   :Options: a positive integer
   :Default: ``20``

.. _proc-summarize-das-findings-noise-histogram-figure-size:

``noise_histogram_figure_size``
   Width and height of that figure (in inches).

   :Options: a two-element list ``[width, height]``
   :Default: ``[24, 4]``
   :Notes: its file format and dpi come from the visualization settings.

.. _proc-summarize-das-findings-noise-histogram-colors:

``noise_histogram_face_color`` / ``noise_histogram_edge_color``
   Hex fill and edge colours of the two histograms.

   :Options: hex colour strings
   :Default: ``#BBD5E8`` / ``#202020``

.. _proc-summarize-das-findings-noise-histogram-cutoff-line-color:

``noise_histogram_cutoff_line_color``
   Hex colour of the dash-dotted lines marking ``noise_corr_cutoff_min`` and ``coherence_cutoff_min``.

   :Options: a hex colour string
   :Default: ``#202020``

Seam check-and-repair
~~~~~~~~~~~~~~~~~~~~~
DAS models inferring with non-overlapping window tiling (stride equal to the window length minus a 32-sample edge trim, i.e. ``stride: 8128`` / ``nb_hist: 8192`` at 250 kHz) judge each stitched span without acoustic context from its neighbours. The consequence is a boundary artifact on *real inter-USV pauses*: the faint edges of the USVs flanking a pause that straddles window seams are clipped exactly to the seam positions, so the pause is recorded with a width of exactly ``k`` stride multiples (a "ladder" at 32.5 / 65.0 / 97.5 ... ms, sample-exact in the raw annotations) and the flanking USVs lose up to one stride of quiet edge material. Nothing is split or duplicated -- the audio between the recorded endpoints is genuinely silent -- but the affected gap widths are quantized (which corrupts inter-USV interval statistics) and the adjacent onsets/offsets are misplaced by up to ~32 ms.

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
   remains the way to recover usable inter-USV intervals from legacy annotations -- enable it with
   ``--seam-repair``.

After every summary build (and standalone, by calling ``repair_seam_snapped_boundaries`` on an existing summary -- the standalone path never rebuilds the summary, so enrichment columns like ``emitter`` and the ``qlvm_*`` set survive), the tool:

#. flags consecutive summary pairs whose gap falls inside the width-gate window around a stride multiple **and** is corroborated by a sample-exact ladder gap in at least one channel's raw DAS annotations at that location -- the joint criterion keeps width-window innocents out, and any that do slip through are re-confirmed unchanged by construction;
#. excises a snippet around each flagged pair from the corroborating channel's HPSS-filtered WAV into ``audio/seam_repair_snippets`` (removed afterwards);
#. re-runs ``das predict`` once on the snippet folder with the model configured under ``das_command_line_inference`` -- the current model tiles with overlapping windows (``stride: 4096`` / ``data_padding: 2048``, i.e. >=8 ms of real context for every kept sample), so no seam is judged context-blind;
#. replaces each pair's facing edges with the re-detected ones, **outward-only** (the artifact only ever clips edges, so true edges lie inside the recorded gap; inward suggestions are single-channel-versus-merge convention differences and clamp to the recorded boundary) and capped at ``seam_repair_max_boundary_shift_s`` per edge, updating ``start`` / ``stop`` / ``duration``; a pair whose corrected edges would meet or cross (the second USV starting at or before the first one's new stop) is skipped as unrepairable rather than written as two overlapping USVs.

The summary CSV is rewritten atomically with every other column preserved, and a sidecar ``*_seam_repair_report.json`` in the session's ``audio`` directory records the settings used and the outcome for every flagged pair (``repaired`` / ``no_change`` / ``skipped:*``). Sessions inferred with the overlap-tiled model produce no ladder fingerprints, so the check is a no-op there by construction. Note that per-USV spectrograms, masks, and acoustic-feature columns generated *before* a repair reflect the old boundaries of repaired rows -- regenerate them for those rows (the report lists them) if an analysis depends on exact USV edges.

Prepare and run USV assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You might also want to know which animal emitted which vocalization. To do this, *usv-playpen* relies on `vocalocator <https://github.com/neurostatslab/vocalocator>`_, a tool for localizing animal vocalizations in 3D space, and it assumes you already have a trained model. These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run these steps together, you need to list the root directories of interest, select the arena directory, select the directory of the vocalocator model, select *Prepare USV assignment* and *Run USV assignment*, select the *Assignment type* (``vcl`` or ``vcl-ssl``), click *Next* and then *Process*:

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_14.png
   :align: center
   :alt: Processing Step 14

.. raw:: html

   <br>

This will create a *sound_localization* subdirectory. With the default *vcl-ssl* backend, it contains a *dset.h5* file (all data relevant for sound localization) and a *model_predictions.npz* file, whose predictions are transferred to the "emitter" column of the *20250430_145017_usv_summary.csv* file. The *dset.h5* ``animal_id`` field holds one sex code per tracked animal (0 = male, 1 = female), read from the ``Subjects`` block of the session's *_metadata.yaml* (the ``subject_id`` matching the track name), never from the track order; a track without a matching subject raises. (With the older *vcl* backend it instead contains *dset.h5*, an *assessment.h5* file with 2D assessment data, and an *assessment_assn.npy* file with 6D assessment output that feeds the "emitter" column.)

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

.. code-block:: json

   "vocalocator": {
    "vcl_audio_band": "ultrasonic",
    "vcl_conda_env_name": "vcl-ssl-ss",
    "vcl_model_directory": "/mnt/falkner/Bartul/sound_localization/mouse_all_model_June2026",
    "vcl_version": "vcl-ssl",
    "subprocess_timeout_hours": null
   }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`vcl_audio_band <proc-vocalocator-vcl-audio-band>`
     - ``"ultrasonic"``
     - ``"ultrasonic"``, ``"broadband"``
   * - :ref:`vcl_conda_env_name <proc-vocalocator-vcl-conda-env-name>`
     - ``"vcl-ssl-ss"``
     - any conda environment name
   * - :ref:`vcl_model_directory <proc-vocalocator-vcl-model-directory>`
     - path (see JSON above)
     - path to a directory
   * - :ref:`vcl_version <proc-vocalocator-vcl-version>`
     - ``"vcl-ssl"``
     - ``"vcl-ssl"``, ``"vcl"``
   * - :ref:`subprocess_timeout_hours <proc-vocalocator-subprocess-timeout-hours>`
     - ``null``
     - ``null``, or hours

.. _proc-vocalocator-vcl-audio-band:

``vcl_audio_band``
   The concatenated audio the vocalocator dataset (``dset.h5``) is cut from, which must be the band the model in ``vcl_model_directory`` was trained on.

   :Options:
      * ``"ultrasonic"`` — the 30 kHz high-passed memmap in *audio/hpss_filtered* (*Ultrasonic MEMMAP*), the band the current models were trained on
      * ``"broadband"`` — the 2 kHz high-passed memmap in *audio/broadband_filtered* (*Broadband MEMMAP*), for a model trained on 2-125 kHz audio, e.g. one that localizes squeaks
   :Default: ``"ultrasonic"``
   :Notes: settings only; not exposed in the GUI. Any other value raises ``ValueError`` naming the allowed bands.

.. _proc-vocalocator-vcl-conda-env-name:

``vcl_conda_env_name``
   Name of the local conda environment used for running Vocalocator.

   :Options: any string naming a local conda environment that has Vocalocator installed
   :Default: ``"vcl-ssl-ss"``
   :Notes: settings / CLI only; not exposed in the GUI.

.. _proc-vocalocator-vcl-model-directory:

``vcl_model_directory``
   Directory containing the trained Vocalocator model.

   :Options: path (string) to a directory
   :Default: ``/mnt/falkner/Bartul/sound_localization/mouse_all_model_June2026``
   :Notes: set in the GUI through the *Vocalocator model directory* field.

.. _proc-vocalocator-vcl-version:

``vcl_version``
   Version of the Vocalocator model.

   :Options:
      * ``"vcl-ssl"`` — the SSL model
      * ``"vcl"`` — the older *vcl* backend
   :Default: ``"vcl-ssl"``
   :Notes: chosen in the GUI through the *Assignment type* field.

.. _proc-vocalocator-subprocess-timeout-hours:

``subprocess_timeout_hours``
   Time budget, in hours, of each Vocalocator subprocess (``vocalocator.assess``; ``vocalocatorssl`` prediction and assignment).

   :Options:
      * ``null`` — no budget, so the step waits however long the subprocess runs
      * a number of hours — a subprocess that runs longer is killed and the step stops with an error naming the command
   :Default: ``null``
   :Notes: a subprocess that exits with an error always stops the step. Settings only; not exposed in the GUI.

The ``assign_vocalizations`` block holds the confidence-set hyperparameters used
when turning the 6-D localizer output into per-mouse attributions (all exposed as
``vcl-assign`` CLI flags):

.. code-block:: json

   "assign_vocalizations": {
    "temperature": 1.0,
    "grid_resolution": [100, 100],
    "n_angle_bins": 46,
    "n_samples": 500,
    "confidence_level": 0.95,
    "angle_pdf_seed": 0
   }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`temperature <proc-assign-vocalizations-temperature>`
     - ``1.0``
     - number
   * - :ref:`grid_resolution <proc-assign-vocalizations-grid-resolution>`
     - ``[100, 100]``
     - ``(x_res, y_res)`` integers
   * - :ref:`n_angle_bins <proc-assign-vocalizations-n-angle-bins>`
     - ``46``
     - integer
   * - :ref:`n_samples <proc-assign-vocalizations-n-samples>`
     - ``500``
     - integer
   * - :ref:`confidence_level <proc-assign-vocalizations-confidence-level>`
     - ``0.95``
     - number in ``[0, 1]``
   * - :ref:`angle_pdf_seed <proc-assign-vocalizations-angle-pdf-seed>`
     - ``0``
     - integer

.. _proc-assign-vocalizations-temperature:

``temperature``
   Covariance temperature scaling applied to the 6-D predictive covariance.

   :Options: a number
   :Default: ``1.0``
   :Notes: CLI flag ``--temperature``.

.. _proc-assign-vocalizations-grid-resolution:

``grid_resolution``
   Spatial ``(x_res, y_res)`` grid the confidence-set PDF is sampled on.

   :Options: a two-element list of integers
   :Default: ``[100, 100]``
   :Notes: the same grid drives both the PDF construction and the point-in-set lookup, so it is a single shared value. CLI flag ``--grid-resolution``.

.. _proc-assign-vocalizations-n-angle-bins:

``n_angle_bins``
   Number of angular histogram bin edges (``-pi`` to ``pi``).

   :Options: a positive integer
   :Default: ``46``
   :Notes: CLI flag ``--n-angle-bins``.

.. _proc-assign-vocalizations-n-samples:

``n_samples``
   Monte-Carlo sample count for the per-vocalization angle PDF.

   :Options: a positive integer
   :Default: ``500``
   :Notes: CLI flag ``--n-samples``.

.. _proc-assign-vocalizations-confidence-level:

``confidence_level``
   Confidence level for the extracted confidence sets.

   :Options: a number in ``[0, 1]``
   :Default: ``0.95``
   :Notes: CLI flag ``--confidence-level``.

.. _proc-assign-vocalizations-angle-pdf-seed:

``angle_pdf_seed``
   RNG seed for the reproducible angle-PDF sampling.

   :Options: an integer
   :Default: ``0``
   :Notes: CLI flag ``--angle-pdf-seed``.

Label, render and embed USVs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the curated *usv_summary.csv* exists (see *Curate DAS outputs* above) and the session's broadband memmap has been written (see *Broadband MEMMAP* above), an in-house, self-contained pipeline first labels every detected segment -- noise, or a vocalization holding a USV, a squeak or both -- and then turns every ultrasonic vocalization (USV) into a spectrogram, a USV mask, interpretable acoustic features, and toroidal **QLVM** latents, and every squeak into squeak QLVM latents.

These steps can be run separately (still in sequence, though), but for the sake of simplicity, they will be described jointly. To run them together, you need to list the root directories of interest, set the *Spectrogram models directory* (the single root from which the Segment Anything Model 2 (SAM2), YOLO, noise and squeak model paths are derived; setting it also fills in the production QLVM cells and the QLVM category bundle, which are code constants rather than paths under it), select *Produce noise labels*, *Produce squeak labels*, *Generate spectrograms*, *Generate masks*, *Compute USV features* and *Infer QLVM latents*, click *Next* and then *Process* (GPU is required):

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/processing_step_15.png
   :align: center
   :alt: Processing Step 15

.. raw:: html

   <br>

*Produce noise labels* flags the segments that hold no vocalization at all, and *Produce squeak labels* gives every other segment its vocal class (USV, squeak or both) and, where it holds a squeak, the squeak's extent; both are described in their own sections below.

The *Generate spectrograms* step computes a variance-weighted, multi-channel spectrogram of every USV (channels listed in the session metadata's ``excluded_channels`` are dropped from the average); *Generate masks* runs a YOLO box detector and prompts SAM2 to segment each USV; *Compute USV features* derives per-USV spectral and amplitude features; and *Infer QLVM latents* embeds each spectrogram into the regular QLVM torus and the four conditional tori (duration, spectral entropy, bandwidth, loudness), labels it with its QLVM category (``qlvm_category``, defined on the regular map) and embeds the squeaks into the squeak QLVM torus.

The mask and latent steps run on the GPU and rely on two pre-trained models (see *Train spectrogram-pipeline models* below). The spectrogram and mask arrays are written to a new *spectrograms* subdirectory, while the acoustic features and QLVM latents are merged into *usv_summary.csv*.

After a cohort of sessions has been processed, ``consolidate-spectrogram-store`` (production mode) merges the per-session H5 files and the QLVM columns of their summaries into one multi-session store under ``spectrograms_root`` (layout below); the newest ``spectrograms_*.h5`` is picked up automatically by every consumer.

Produce noise labels
^^^^^^^^^^^^^^^^^^^^

The DAS segmenter keeps every interval a channel fired on, so a curated *usv_summary.csv* also holds segments with no vocalization in them: electrical clicks, cage knocks, broadband transients and faint smears. *Produce noise labels* scores every USV segment with an ensemble of five time-resolved multiple-instance classifiers (the bundle is derived from the *Spectrogram models directory* as ``noise/noise_timemil_ens5_n4680_20260926.pt``, whose name records the architecture, the ensemble size, the 4,680 training labels and the build date) and adds two columns to *usv_summary.csv*:

* ``noise`` -- ``true`` / ``false`` in every row; ``true`` **excludes** the segment, because the model is not confident it holds a vocalization (``noise_probability`` at or above the bundle's exclusion cut-off, 0.14);
* ``noise_probability`` -- the ensemble's probability in every row, so an analysis can tell confident noise (at or above 0.82) from the uncertain band, or re-threshold, without re-running the step.

On a re-run, a row that becomes noise loses the vocal class (``usv`` / ``squeak``, their probabilities and the squeak extent) an earlier *Produce squeak labels* gave it, since those columns are empty on noise rows and the class masks rely on that, and it loses its squeak-map coordinates (``qlvm_squeak1`` / ``qlvm_squeak2`` and any category read off them), which the squeak embedding never writes for noise rows; a row that stops being noise has no class until *Produce squeak labels* runs again. The summary is written atomically.

A segment is *noise* only when it holds no vocalization at all -- neither a USV nor a squeak -- which is the rule its training labels follow, so a faint vocalization heard on one microphone, or a vocalization mixed with noise, is a vocalization, not noise.

The decision is fixed by the model bundle, not by a setting. The ensemble's probability splits the segments into confident vocalizations (below 0.14), an uncertain band (0.14-0.82) and confident noise (0.82 and above), and ``noise`` is ``true`` for the last two: uncertain segments are excluded with the noise rather than guessed. The cut-offs are the narrowest uncertain band whose confident decisions reach precision and recall of 0.95 on 1,118 segments drawn at random from all listed sessions and labelled by consensus (the majority of up to four blind passes), weighted to the cohort.

Applied held out -- the cut-offs chosen on four session folds and scored on the fifth, each segment judged by an ensemble that never saw its session -- the confident decisions reach **precision 0.944 and recall 0.955**, with 0.9% of segments uncertain. The exclusion costs about 71 real vocalizations per 10,000 segments (65 of them in the uncertain band, which holds mostly faint, short vocalizations heard on one or two microphones). The step prints these numbers on every run.

An earlier bundle (``noise_timemil_ens5_n3562_20260916.pt``) described its calibration as measured on sessions the models never trained on; it was not -- 422 of its 633 calibration segments were training labels -- and on new, cohort-representative labels it reached precision 0.78 and recall 0.95 at its threshold, not the 0.987 / 0.973 it reported. The detector refuses bundles without the validated decision block.

Each segment's input is the two-band absolute-dB spectrogram (30-120 kHz and 3-30 kHz, 128 linear bins each) of the channels of the session's broadband memmap (*audio/broadband_filtered*: the per-channel HPSS audio high-passed at 2 kHz with the line-noise tones removed, written by *Broadband MEMMAP* (``broadband-filter-audio``); its ``line_noise.json`` names the channel of every column), averaged across channels by variance -- each channel's broadband variance over the window plus its stored low-band variance from ``line_noise.json``, which equals the full-band variance the model was trained with (see *Broadband MEMMAP*) -- with metadata-excluded channels dropped. The audio window extends ~100 ms either side of the segment so the channel weights and STFT edges match the training inputs, and the spectrogram is then cropped back to the segment's own frames: the model judges the segment, not its neighbourhood.

Run *Produce noise labels* after *Curate DAS outputs* (re-curating rewrites *usv_summary.csv* with its base columns only) and before *Produce squeak labels*; the processing run places both after USV assignment, which neither reads nor drops their columns. The summary's column layout is fixed regardless of the order. A GPU is used when present but is not required: a 424-USV session takes about 64 s on one and 90 s on the CPU, because most of the time goes on reading and transforming the audio rather than on the network.

Produce squeak labels
^^^^^^^^^^^^^^^^^^^^^

A *squeak* is a broadband harmonic stack (fundamental around 3-8 kHz) that the ultrasonic DAS segmenter picks up as part of a USV segment; a segment can hold a USV alone, a squeak alone, or both. Once the curated *usv_summary.csv* exists and *Produce noise labels* has run, *Produce squeak labels* classifies every segment that is not noise into one of three vocal classes (``usv``, ``squeak`` and ``overlap``, a squeak and a USV in one segment) with an ensemble of five time-resolved multiple-instance classifiers (the bundle is derived from the *Spectrogram models directory* as ``squeak/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt``) and adds seven columns to *usv_summary.csv*, placed right after the noise columns:

* ``usv`` / ``squeak`` -- two booleans from the class of highest ensemble-mean probability: a pure USV (a USV only) is ``usv`` true and ``squeak`` false, a pure squeak is ``usv`` false and ``squeak`` true, and an overlap segment (a squeak and a USV together) has both true; both are empty on noise rows (and on the rare segment too short for one STFT window). ``usv`` alone does not mean "USV only": a pure-USV selection is ``usv & ~squeak`` (``os_utils.pure_usv_mask``; ``pure_squeak_mask``, ``overlap_mask`` and ``squeak_bearing_mask`` give the others), and every consumer in the package selects through these helpers;
* ``usv_probability`` / ``squeak_probability`` / ``overlap_probability`` -- the ensemble-mean probabilities of the three classes, pure USV, pure squeak and overlap (they sum to 1), so an analysis can re-threshold without re-running the step; empty where the booleans are. The probability that a segment holds a squeak at all is ``squeak_probability + overlap_probability``;
* ``squeak_start`` / ``squeak_end`` -- ONE squeak extent per segment, in session seconds (the clock of ``start`` / ``stop``), on rows with ``squeak`` true only (empty otherwise): the envelope from the first to the last frame whose ensemble-mean squeak probability exceeds the bundle's threshold (0.6), counting only the above-threshold runs that touch the segment's own frames (a run lying wholly in the context belongs to a neighbouring segment) and with no minimum run length, from half a hop before the first frame's centre to half a hop after the last. A segment holding several squeaks gets their envelope, gaps included. The extent may reach into the context past the segment. A ``squeak`` row with no above-threshold frame touching the segment falls back to the one segment frame of highest squeak probability (a one-frame extent); the step reports how many rows needed the fallback.

The columns of earlier encodings (``squeak_frame_runs``, ``call_class``, ``squeak_spans``, ``n_squeaks``, and ``p_usv`` / ``p_squeak`` / ``p_both``, the first names of the three class probabilities) are removed and an older ``squeak`` column is replaced when a summary is rewritten. The squeak map's columns (``qlvm_squeak1`` / ``qlvm_squeak2`` and any category read off them) are dropped, because they were embedded from the previous classification's squeak rows and extents (run *infer-qlvm-squeak-latents* again), and a row that becomes a pure squeak loses its USV-map coordinates and category, which every USV map leaves empty on pure squeaks (a row that stops being a pure squeak has none until *infer-qlvm-latents* runs again); every other column is left exactly as it was.

The classifier reads the noise model's input (see *Produce noise labels*): two absolute-dB bands (30-120 kHz and 3-30 kHz, 128 linear bins each) of the variance-weighted average of the channels of the session's broadband memmap (*audio/broadband_filtered*; *audio/hpss_filtered* is high-passed above 30 kHz and carries no squeak energy), with metadata-excluded channels dropped.

The audio window is the noise model's too -- the segment plus 49 hops (100 ms) of context either side -- but the spectrogram is **not** cropped back to the segment, because squeaks often run past the segment the ultrasonic segmenter cut; a third input channel marks the segment's own frames, so the network knows which frames the class decision is about.

The network is the noise model's trunk (its trained weights are the starting point of every member) with two heads: a three-class head attention-pooled over the segment's frames (plus a linear term in the log channel count and log duration), and a per-frame squeak head over every frame of the window, whose mean track gives the squeak extent above. The labelled training spans (several per segment where the labeller marked several squeaks, the context included) are the squeak head's targets; the envelope rule is applied at inference only. The threshold and every input constant are read from the bundle.

Held out by session (5-fold grouped cross-fitting on the 2,476 non-unsure labels of two labelling rounds, the first drawn blind and stratified across the cohort), the three classes reach precision / recall of 0.999 / 0.993 (usv), 0.983 / 0.950 (squeak) and 0.915 / 0.985 (overlap) on the blind round's 1,981 segments; the commonest confusion is a usv segment classed as overlap (12 of 1,664), and squeak and usv are never confused with each other (the bundle's ``cv_summary`` records the tables).

On three sessions (2,951 segments; 2023, 2025 and 2026 recordings, one with metadata-excluded microphones), this step reproduced the 49 labelled segments' inputs bit for bit against the inputs the bundle was trained on, their class on all 49, their class probabilities to within 6.4e-6 (batching only) and their extents on all 49, and left every other column of the summaries unchanged.

On the 42 labelled squeak / both segments among them, the extent's onset and offset lay a median 1.6 ms and 2.3 ms (90th percentile 8.0 and 7.7 ms) from the envelope of the labelled spans; over all 812 labelled squeak / both segments of the training labels (in-sample) the medians are 1.9 and 2.6 ms, and 2.8 and 3.2 ms on the 63 segments labelled with several squeaks. There, 22 of the 1,034 squeak-bearing segments (2.1 %) needed the one-frame fallback (none of the labelled ones), and 708 extents reached past their segment.

Run *Produce squeak labels* after *Curate DAS outputs* and *Produce noise labels*: re-curating rewrites *usv_summary.csv* with its base columns only, which removes the vocal-class columns, and the step needs the ``noise`` column (the processing run orders the steps accordingly). A GPU is recommended. The ensemble is trained by ``train-usv-squeak-model`` (see *Vocal-class model* below).

.. note::

   The ensemble was fitted on the labelled segments of 2,476 panels across the cohort (the label files the bundle's ``labels`` / ``label_overrides`` name). Its output on those segments is a fit, not a prediction: leave them out of any held-out accuracy claim.

Spectrograms, masks, features and QLVM latents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

**The consolidated store.** ``consolidate-spectrogram-store`` builds the store in one of two modes, chosen by ``--store-mode`` or the ``consolidate_spectrogram_store.store_mode`` setting (``production``, the shipped value, or ``v3``). Both group-copy every session's ``spectrogram/<session>`` and ``mask/<session>`` groups next to one shared ``frequency_bins`` axis, validate every session before anything is written (all problems are reported together), copy one session at a time and publish the file atomically. The ``consolidate_spectrogram_store`` settings block holds the one setting, the mode a run without ``--store-mode`` uses:

.. code-block:: json

    "consolidate_spectrogram_store": {
        "store_mode": "production"
      }

*Production mode* writes ``spectrograms_production_<S>sessions_<N>vocalizations_<UTC timestamp>.h5`` from the current production QLVM maps: the five USV maps of ``os_utils.QLVM_MAPS``, the squeak map ``os_utils.QLVM_SQUEAK_MAP`` (cells resolved by ``os_utils.qlvm_map_cell_directory``) and the category column ``qlvm_category`` of the category bundle (``os_utils.load_qlvm_category_bundle``).

Sessions come from ``--root-directories`` and / or ``--session-lists`` (``*.txt`` files, one session root per line, ``#`` comments and blank lines skipped; each root once, in first-seen order) or from ``--package-corpus`` (the v3 package's session list; no SHA-256 check in this mode), so ``consolidate-spectrogram-store --store-mode production --package-corpus`` rebuilds the store over the 402 sessions of the v3 corpus, and ``--session-lists`` with the cohort's session lists builds it over every embedded session.

A session is refused when its summary does not have one row per H5 row; lacks a coordinate column of any map, ``qlvm_category`` or the ``usv`` / ``squeak`` flags; when the regular map does not place a USV exactly where its status is 0 (the rule ``infer-qlvm-latents`` embeds by: USV frames, duration below the regular cell's ``length_threshold``, a SAM mask, not a pure squeak), a conditional map places a USV the status excludes, or the squeak map places a USV without a squeak; or when ``qlvm_category`` is not present exactly where ``qlvm1`` / ``qlvm2`` are or differs from the bundle's category of the USV's pixel (the summary was labelled with another bundle: run ``assign-qlvm-categories`` on it). The regular cell's contract must set ``require_mask``, which the status rule assumes.

.. code-block:: text

    spectrograms_production_<S>sessions_<N>vocalizations_<ts>.h5
    ├── frequency_bins (F,)                              # shared by every session
    ├── spectrogram/<session>/...                        # copied from the session H5 (spectrograms, durations)
    ├── mask/<session>/...                               # copied from the session H5 (segmentations, spectrogram_index)
    ├── qlvm/<session>/<map> (N, 2) float64              # qlvm, qlvm_duration, qlvm_entropy, qlvm_bandwidth,
    │                                                    # qlvm_loudness, qlvm_squeak coordinates; NaN = not embedded
    ├── qlvm/<session>/status (N,) int8                  # bit field: 0 embedded, 1 too long, 2 no SAM mask,
    │                                                    # 4 pure squeak, 8 no USV frames
    ├── qlvm/<session>/qlvm_category (N,) int16          # 1..k (R-1..R-k); 0 = not on the regular map
    ├── qlvm/<session>/qlvm_category_agreement (N,) float64  # the bundle's agreement of the USV's pixel; NaN = not on the map
    ├── qlvm/<session>/qlvm_category_uncertain (N,) bool # agreement below the bundle's uncertain_agreement
    ├── qlvm_models/<map>/                               # attrs: map, cell_directory, model_id, checkpoint_sha256,
    │                                                    #        condition, training_contract (JSON)
    ├── qlvm_categories/label_grid, agreement            # the bundle's grids; attrs: column, map, model_id, identity,
    │                                                    # directory, uncertain_agreement, names, descriptions,
    │                                                    # nomenclature, build_config
    └── sessions                                         # session_id, spectrogram_h5_sha256, usv_summary_sha256,
                                                         # n_rows, n_embedded, n_squeak_embedded

The per-USV agreement and uncertain flag are not in the summaries: they are computed from the bundle and each USV's ``qlvm1`` / ``qlvm2`` with the assigner's pixel rule (``qlvm_categories.assign_categories``). The root attrs record ``created_by``, ``created_date`` (UTC), ``git_commit``, ``store_layout`` (``spectrograms_production``), ``n_sessions``, ``n_vocalizations``, ``category_bundle_identity`` and ``generate_spectrograms_settings`` (with its note, as in v3 mode). The ``sessions`` table's SHA-256 of each spectrogram H5 and summary fingerprints the inputs the store was built from.

*V3 mode* writes ``spectrograms_qlvmv3_<S>sessions_<N>vocalizations_<UTC timestamp>.h5`` to ``spectrograms_root``. It is meant for the corpus of the QLVM model package v3 (``consolidate_spectrogram_store.V3_PACKAGE_ROOT`` and its cells ``V3_MODEL_CELLS``; the store stays pinned to that package, since the production cells of ``os_utils.QLVM_PRODUCTION_MODEL_CELLS`` have no package baseline, manifest or cluster tables, and summaries in the canonical layout no longer carry the v3 ``qlvm_mf*`` / ``qlvm_bw*`` / ``qlvm_loud*`` columns, so they are refused): ``--package-corpus`` takes exactly the 402 sessions its ``corpus/SESSION_H5_BASELINE.tsv`` lists (each session root is the third parent of the listed H5 path under ``/mnt/falkner``), ``--root-directories`` any subset of them.

Every session is checked before anything is written, and all problems are reported together: its spectrogram H5 must hash (SHA-256) to the baseline's (a rebuilt H5 renumbers the rows the package embedded), its summary must have one row per H5 row and carry the coordinate columns of every v3 model (``qlvm1`` … ``qlvm_loud2``), all sessions must share one ``frequency_bins`` axis, and the embedded USVs must be exactly those the package's rules admit (status 0 exactly where ``qlvm1`` / ``qlvm2`` are non-null, each model's labels exactly where its coordinates are). Sessions are copied one at a time and the file is published atomically.

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

The status of a USV is computed from the H5: bit 0 is set when its duration is at or above the regular cell contract's ``length_threshold`` (128 time bins; a cell trained without a duration bound, contract ``length_threshold`` null, never sets it, and the store's ``qlvm`` ``length_threshold`` attr is then NaN, in both modes), bit 1 when it has no SAM mask instance (the bincount of ``mask/<session>/spectrogram_index`` is 0).

``session_type`` comes from the regular cell's ``inference/recon_mse_breakdown.npz`` (``session_types``), the source the package names for it. The tables are compound datasets with a ``schema_json`` attr (read them back with ``usv_interval_archive._h5_to_polars``). A non-default label level (e.g. ``qlvm_dur_supercategory``) is stored, with its tables, when every session's summary carries it.

The root attrs record ``created_by``, ``created_date`` (UTC), ``git_commit``, ``n_sessions``, ``n_vocalizations``, ``package_root``, ``package_manifest_sha256`` and ``generate_spectrograms_settings`` -- the ``generate_spectrograms`` processing settings in force at consolidation, as JSON; the settings each session's spectrograms were generated with were not recorded per session, as ``generate_spectrograms_settings_note`` says. Older ``spectrograms_sam2masks_*`` stores are still resolved; the newest ``spectrograms_*.h5`` of any name (production, v3 or ``sam2masks``) wins.

**The squeak spectrogram store.** The consolidated store's spectrograms span 30-125 kHz, which shows only the ultrasonic tail of a squeak (a broadband harmonic stack with its fundamental around 3-8 kHz). ``build-squeak-spectrogram-store`` (``build_squeak_spectrogram_store`` settings block) writes a second, squeak-only store to ``spectrograms_root`` that the embedding explorer's *Squeaks* map shows instead (see :doc:`Notebooks`): ``squeak_spectrograms_<F>logbins_<S>sessions_<N>squeaks_<UTC timestamp>.h5``, resolved (newest first) by ``os_utils.resolve_squeak_spectrogram_store_path``; the name never matches the consolidated store's ``spectrograms_*.h5`` pattern. Sessions come from ``--root-directories`` and/or ``--session-lists`` (``*.txt`` files, one session root per line) and are processed in ``n_workers`` parallel processes; a session that fails is reported (and recorded in the ``failed_sessions`` attr) and left out.

* **Rows**: per session, the rows with ``squeak`` true (pure squeaks and segments holding both) and ``noise`` not true, the rows *Embed squeaks in a QLVM torus* considers, so every row that carries ``qlvm_squeak1`` / ``qlvm_squeak2`` is in the store, and the explorer's squeak-class filter (squeak / overlap / squeak + overlap) needs no rebuild.
* **Spectrogram**: rebuilt from the session's broadband memmap (``audio/broadband_filtered``) over the row's squeak crop window -- the segment, widened where needed to hold the squeak envelope plus two frames of context, the window *Embed squeaks in a QLVM torus* reads -- dropping the metadata-excluded channels (``exclude_metadata_audio_channels``) exactly as *Produce squeak labels* does, with the sonic front end's STFT: Blackman-Harris, ``nperseg`` 2048 (8.19 ms, 122 Hz bins; the main lobe of about 1 kHz separates harmonics 3 kHz apart), hop 512 (2.048 ms frames, centred), so frame ``t`` sits at ``audio_start_s + t * 0.002048`` s like the embedding's frames. The linear bins are mapped onto ``n_frequency_bins`` (128) log-spaced bins with geometric edges from ``min_freq`` (2 kHz) to ``max_freq`` (125 kHz): a log bin averages the power of the linear bins whose centres fall inside it, and a log bin narrower than the 122 Hz linear spacing (below about 7.6 kHz at 128 bins) takes the power interpolated at its geometric centre. The power is converted to absolute dB (``ref`` 1.0, no ``top_db`` clamp; silence at -100 dB) and averaged across channels with audio-variance weights (the rule of *Generate spectrograms*).
* **Time extent**: every USV sits from column 0 of a fixed ``window_frames`` window (256 frames, 524 ms), its frame count in ``durations``, so a thumbnail padded to the window shows its true duration, as in the consolidated store. An audio window longer than the display window keeps the ``window_frames`` frames centred on its squeak crop (the frames inside ``squeak_start`` .. ``squeak_end`` plus two, the embedding's crop), clipped to the audio window; ``window_first`` records the first kept frame and ``audio_start_s`` the session time of the audio window's frame 0.
* **Storage**: dB values clipped to ``[db_floor, db_ceil]`` (-100 to 60 dB) and quantized to uint8 (0.63 dB per step; code 0 = ``db_floor``, so the padding reads as silence), gzip-compressed in one chunk per USV.

.. code-block:: text

    squeak_spectrograms_128logbins_<S>sessions_<N>squeaks_<ts>.h5
    ├── frequency_bins (F,) float64                      # geometric centres of the log bins (Hz)
    ├── frequency_bin_edges (F + 1,) float64             # their edges (Hz)
    ├── spectrogram/<session>/spectrograms (n, F, W) uint8   # W = window_frames; dB = db_floor + code * (db_ceil - db_floor) / 255
    ├── spectrogram/<session>/row_index (n,) int64       # 0-based usv_summary.csv rows, ascending
    ├── spectrogram/<session>/durations (n,) int32       # frames of the USV in the window (0: shorter than one STFT window)
    ├── spectrogram/<session>/n_frames (n,) int32        # native frames of the whole segment
    ├── spectrogram/<session>/window_first (n,) int32    # first kept frame (0 unless the segment is longer than W)
    └── sessions                                         # session_id, root_directory, usv_summary_sha256,
                                                         # n_summary_rows, n_squeaks

The root attrs record ``created_by``, ``created_date`` (UTC), ``git_commit``, ``n_sessions``, ``n_squeaks``, the settings block (``settings``, JSON), the STFT (``stft``, JSON), ``frame_dt_s``, ``window_frames``, ``db_floor`` / ``db_ceil``, the quantization rule and ``failed_sessions``. Sessions without squeaks are listed in ``sessions`` but get no ``spectrogram`` group. Rebuild the store after *Produce squeak labels* or *Produce noise labels* change a session's rows. The ``build_squeak_spectrogram_store`` settings block (every key is also a CLI flag, see the CLI reference):

.. code-block:: json

    "build_squeak_spectrogram_store": {
        "min_freq": 2000.0,
        "max_freq": 125000.0,
        "n_frequency_bins": 128,
        "window_frames": 256,
        "db_floor": -100.0,
        "db_ceil": 60.0,
        "exclude_metadata_audio_channels": true,
        "n_workers": 6
      }

Embed squeaks in a QLVM torus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Squeaks have their own QLVM models, 2-D tori trained on sonic-band squeak crops. The production model is the ``train-qlvm`` cell ``stretch_nofloor`` of the squeak package, copied to ``/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_squeak``: ReLU head, trained on 11,000 class-balanced squeak crops (4,905 pure squeaks, 6,095 segments holding both; 8,728 training / 2,272 validation rows, whole sessions held out by type) time-stretched to fill the 128-frame decoder frame, unmasked and with no loudness floor.

It replaces the v3 model package's phase 3 broadband-vocalization cells (``/mnt/falkner/Dexter/vocal_beh/models/qlvm_models/qlvm_models_latest/phase3_BBVs_qlvm/``: ``natural_lumped``, ``natural_session``, ``uniform_lumped`` and ``uniform_session``, each ``_N11000_nomask``; zero-padded, un-stretched crops), which can still be named explicitly.

``infer-qlvm-squeak-latents`` (the last part of *Infer QLVM latents* in the processing run, and a command of its own) places every segment of a session with ``squeak`` true (pure squeaks and segments holding both) that is not noise (a null ``noise`` counts as not noise, the single noise rule the analyses use) on the torus of ONE squeak cell and adds two columns to *usv_summary.csv*:

* ``qlvm_squeak1`` / ``qlvm_squeak2`` -- the squeak's torus coordinates in ``[0, 1)`` (the posterior mean, the same convention as ``qlvm1`` / ``qlvm2``), null on every other row. No cluster labels are written.

The input of each segment is rebuilt the way the cell's training set was built (the squeak package's ``scripts/s01_build_sets.py`` for the ``train-qlvm`` cells, the reference builder ``build_bbv_dataset.py`` for the phase 3 cells): the cells' sonic spectrogram (3-30 kHz, 128 linear bins, absolute dB, from the session's broadband memmap *audio/broadband_filtered*; the front end of the retired squeak classifier) is cropped to the frames whose centres lie inside the squeak extent ``squeak_start`` .. ``squeak_end``, plus two frames of context either side, min-max normalized per crop and written into a 128-frame zero frame, then time-stretched to fill all 128 frames (the production cell, ``time_stretch`` true in its contract) or centred without stretching (the phase 3 cells), and normalized once more as the decoder's data loader did (per-spectrogram min-max; the contract's loudness floor, when a cell has one, then a second min-max).

Because a squeak often extends past its segment, the audio is read over the segment widened, where needed, to hold the envelope and its two context frames (the segment itself otherwise, as for the training crops), so a crop is never cut at the segment boundary.

The frame rule is closed (a frame is kept when its centre lies in ``[squeak_start, squeak_end]``, ends included), and this makes the crop one frame wider when the squeak starts before its segment: the detector writes extent edges half a hop from frame centres, so an ``n``-frame squeak spans exactly ``n`` hops; with the window starting at the segment start its edges fall between frame centres and ``n`` frames are kept, but with the window starting two context frames before ``squeak_start`` both edges fall exactly on frame centres and ``n + 1`` frames are kept (the extra one centred on ``squeak_end``). The time-stretched production cell therefore stretches the same squeak very slightly differently depending on where the segmenter cut.

The rule is kept until the next retrain of the squeak cell, because every stored ``qlvm_squeak1`` / ``qlvm_squeak2`` was embedded with it and which rule the shipped cell's training crops used is not verified; the half-open rule belongs with that retrain, after which every session is re-embedded.

One row holds one extent and one pair of coordinates, so a segment with several squeaks is embedded once, over the envelope of all of them, gaps included. Crops narrower than 8 frames (the training set's minimum; a one-frame fallback extent always is) or wider than 128 frames (the decoder frame; no training set, stretched or not, held a wider crop) get nulls. One deliberate difference of the phase 3 cells: their training set cropped from a store that held only a segment's first 128 frames and dropped squeaks that ran past its end (right-censored), whereas this step crops from the full-length spectrogram, so such a squeak is embedded with its measured extent when its crop fits in 128 frames.

The posterior mean is taken over the cell's 46,368-point Fibonacci lattice (``embedding_fib_m`` 24 of a ``train-qlvm`` cell's contract, ``lattice_m`` 24 of a phase 3 cell's ``manifest.json``), built in float32 exactly as the torch driver that embedded the corpus built it: the exact (float64) lattice sits up to 2.3e-3 away from it and moves the embeddings by a median 6e-4.

The phase 3 cells were validated against the corpus embeddings the four cells ship (``posterior_cache.npz``, 14,936 squeaks of 239 sessions): the crop code reproduces every corpus input bit-exactly from the reference sonic spectrogram store, and on the reference inputs the embedding matches the package's corpus embedding to a median wrapped distance of 2.6e-5 to 3.1e-5 of the torus, with 99.5 % of squeaks within 1e-3, in all four cells.

The remaining difference is numerical: the corpus embedding's GPU run computed the transposed convolutions in TF32 (torch's cuDNN default) and this step in full float32; a torch replica of the original torch driver in full float32 lands where this step does, and with TF32 on it reproduces the two lumped cells to a median 1.5e-6.

Built from the session audio instead of the reference store, the inputs were bit-identical wherever the crop and the microphones agreed, with two exceptions by design. The reference store averaged all 24 microphones, while this step drops metadata-excluded channels like *Produce squeak labels* does (13 corpus sessions, all from January and February 2023, exclude microphones; ``--no-exclude-metadata-audio-channels`` reproduces the reference inputs there bit-exactly). And a segment longer than 128 frames is cropped with its full-length extent (above).

These checks were made with the extents of the retired binary classifier; the crop extents now come from the vocal-class model's squeak extent, so the squeaks (and their crops) differ from the corpus ones and the corpus comparison holds for the crop and embedding code, not for the extents.

The cell is the ``infer_qlvm_squeak_latents.model_cell_directory`` setting. The shipped settings leave it empty, and an empty setting is filled with the production squeak cell ``qlvm/qlvm_squeak`` (trained as ``squeaks/cell/stretch_nofloor``; ``os_utils.QLVM_SQUEAK_PACKAGE_ROOT`` / ``QLVM_SQUEAK_PRODUCTION_CELL``) whenever ``spectrograms_root`` is set; name another cell with ``--model-cell-directory``.

Two cell layouts are read. A ``train-qlvm`` cell (``checkpoint.tar``, ``config/training_contract.json``, ``config/run_config.json``, ``metrics/``) states its input rule in its contract, and the step reads and checks it there: ``masking_type`` ``"none"`` and ``require_mask`` false (squeaks have no SAM masks), ``time_stretch`` (stretched or centred crops), ``floor`` (null or in ``[0, 1)``), ``input_normalization`` / ``normalization_epsilon``, a ``[128, 128]`` ``target_shape``, an unconditional 2-D decoder whose ``decoder_head`` is the weights' head, and the ``embedding_fib_m`` of its Fibonacci lattice.

The phase 3 cells keep the old package layout (every file at the cell root, no ``training_contract.json``), so the step reads and checks what they do ship: the decoder weights of ``checkpoint.tar`` (read without torch), ``run_config.json`` (a 2-D, unmasked BBV run) and ``manifest.json`` (an unmasked, un-stretched 128 x 128 corpus and its ``lattice_m``).

The log names the cell, its layout and whether its inputs are time-stretched and floored. Changing the squeak cell moves every squeak to a different torus, so re-embed every session (the cohort's ``qlvm_squeak1`` / ``qlvm_squeak2`` must all come from one cell). Run it after *Produce noise labels* and *Produce squeak labels*; a GPU speeds it up but is not required.

Summary columns
^^^^^^^^^^^^^^^

The *Compute USV features* and *Infer QLVM latents* steps add columns to *usv_summary.csv* in place. *Compute USV features* adds:

* **mean_freq_hz** : energy-weighted mean frequency of the USV (Hz)
* **peak_freq_hz** : frequency of peak energy (Hz)
* **freq_bandwidth_hz** : spectral bandwidth between the low/high cumulative-energy edges (Hz)
* **mean_amplitude** : mean spectrogram amplitude over the USV (a.u., on the USV's own min-max normalized [0, 1] spectrogram, so relative to the USV's peak)
* **max_amplitude** : maximum spectrogram amplitude over the USV (a.u., same normalized scale; close to 1 whenever the mask covers the USV's peak)
* **loudness_db** : the USV's ABSOLUTE loudness (dB): the image-level level of ``compute_usv_loudness`` — per eligible microphone, the mean over the USV's SAM mask of ``10 log10`` of the absolute band power re-read from the session's ``hpss_filtered`` audio, combined across microphones with the spectrogram generator's variance weights. It is the value the QLVM loudness-conditional model is trained and decoded on (``infer-qlvm-latents`` reads it from here), and it reproduces the model package's corpus values. It needs the session's ``hpss_filtered`` audio and SAM masks; a USV without a mask gets null
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

*Produce noise labels* and *Produce squeak labels* add their own columns, placed right after the DAS event (``usv_id``, ``start``, ``stop``, ``duration``) and before ``emitter`` and the DAS channel statistics:

* **noise** : ``true`` / ``false`` in every row -- the segment holds no vocalization at all
* **usv** / **squeak** : the vocal class; see *Produce squeak labels* above
* **noise_probability** : the noise ensemble's probability in every row, so an analysis can re-threshold without re-running the step
* **usv_probability** / **squeak_probability** / **overlap_probability** : the probabilities of the three vocal classes behind ``usv`` / ``squeak``
* **squeak_start** / **squeak_end** : the squeak extent on ``squeak`` rows; see *Produce squeak labels* above

*Infer QLVM latents* (``infer-qlvm-latents`` followed by ``assign-qlvm-categories``) adds the QLVM maps (see *Several QLVM models in one run* below):

* **qlvm1** / **qlvm2** : the two torus (latent) coordinates of the regular map
* **qlvm_category** : its content-ridge category, ``1 … k`` meaning ``R-1 … R-k`` (written by ``assign-qlvm-categories`` from ``qlvm1`` / ``qlvm2``; the per-USV category agreement and uncertain flag are not stored in the summary)
* **qlvm_duration1** / **qlvm_duration2**, **qlvm_entropy1** / **qlvm_entropy2**, **qlvm_bandwidth1** / **qlvm_bandwidth2**, **qlvm_loudness1** / **qlvm_loudness2** : the coordinates of the duration, spectral-entropy, bandwidth and loudness conditional maps (no category columns of their own: the categories are defined on the regular map, so every analysis or figure of a conditional map labels a USV with its ``qlvm_category``)

The full column order of a session's summary is fixed in one place, ``os_utils.USV_SUMMARY_COLUMN_ORDER``, and every step that writes the summary puts its columns in that order (``os_utils.order_usv_summary_columns``): ``usv_id``, ``start``, ``stop``, ``duration``, ``noise``, ``usv``, ``squeak``, ``noise_probability``, ``usv_probability``, ``squeak_probability``, ``overlap_probability``, ``squeak_start``, ``squeak_end``, ``emitter``, ``peak_amp_ch``, ``mean_amp_ch``, ``chs_count``, ``chs_detected``, ``mean_freq_hz``, ``peak_freq_hz``, ``freq_bandwidth_hz``, ``mean_amplitude``, ``max_amplitude``, ``loudness_db``, ``spectral_entropy``, ``mask_number``, ``qlvm1``, ``qlvm2``, ``qlvm_category``, ``qlvm_duration1``, ``qlvm_duration2``, ``qlvm_entropy1``, ``qlvm_entropy2``, ``qlvm_bandwidth1``, ``qlvm_bandwidth2``, ``qlvm_loudness1``, ``qlvm_loudness2``, ``qlvm_squeak1``, ``qlvm_squeak2``.

A column a session does not have yet is simply absent; a column the order does not list is kept after the listed ones, in its existing order.

Summaries written before this layout may still carry ``qlvm_supercategory``, the short-prefix duration and spectral-entropy maps ``qlvm_dur1`` / ``qlvm_dur2`` (with ``qlvm_dur_category`` / ``qlvm_dur_supercategory``) and ``qlvm_ent1`` / ``qlvm_ent2`` that ``qlvm_duration*`` / ``qlvm_entropy*`` replace, the retired v3 ``qlvm_mf*`` / ``qlvm_bw*`` / ``qlvm_loud*`` maps and their labels (obsolete names are matched exactly, so the current ``qlvm_bandwidth*`` / ``qlvm_loudness*`` columns are never dropped), the retired squeak-detector columns ``squeak_frame_runs`` / ``call_class`` / ``squeak_spans`` / ``n_squeaks``, or ``*_category_agreement`` / ``*_category_uncertain`` columns (``os_utils.USV_SUMMARY_OBSOLETE_COLUMNS``), and the class probabilities under their first names ``p_usv`` / ``p_squeak`` / ``p_both`` (``os_utils.USV_SUMMARY_RENAMED_COLUMNS``); ``tidy-usv-summary-columns`` (see the CLI reference) removes the obsolete columns, renames the old-name ones and reorders the rest.

A ``model_cells`` run writes torus coordinates only: a package cell's own cluster ``label_grid.npy`` files are not read, and the summary's one category column, ``qlvm_category``, comes from ``assign-qlvm-categories``. No model-provenance column is written: summaries embedded by the retired single-model run (``model_cell_directory``, or decoder weights with reference arrays) may still carry a legacy **qlvm_model** column (the cell or weights path their ``qlvm_*`` columns came from), which the next *Infer QLVM latents* run removes. Coordinates from different models share the column names but not their meaning, so embed every session of an analysis with the same cells.

*Embed squeaks in a QLVM torus* (``infer-qlvm-squeak-latents``, the last part of *Infer QLVM latents*) adds, last:

* **qlvm_squeak1** / **qlvm_squeak2** : the torus coordinates of each ``squeak`` / ``overlap`` segment that is not noise in a squeak (BBV) QLVM cell, null on every other row; see *Embed squeaks in a QLVM torus* above

Switching sessions to a QLVM model package cell also changes what these readers see:

* the QLVM visualizations draw the map chosen by ``shared_resources.qlvm_map`` (``visualizations_settings.json``; one of ``os_utils.QLVM_MAPS``, ``qlvm`` by default), and every category grid, boundary, centre or density they draw comes from ONE place, the category bundle ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY`` (``/mnt/falkner/Bartul/spectrograms/categories``, written by ``build-qlvm-categories``), loaded through ``os_utils.load_qlvm_category_bundle``: its ``label_grid`` (R-1 … R-k), the partition the summaries' ``qlvm_category`` is assigned from (``assign-qlvm-categories`` defaults to the same bundle), its smoothed corpus ``density`` and its category label positions. The bundle is a partition of the regular map's torus, so the neuronal tuning watersheds, the sequence figure's landscape, the embedding thumbnails' boundaries and centres, the embedding explorer's boundaries, the torus-traversal video, the manifold filter atlas and the category-embedding panel draw it on the regular map only; on a conditional map they draw no category boundaries and show each USV's category by its ``qlvm_category`` colour instead (each says so), and nothing estimates boundaries from the data. A module constant, like the production cells, because the settings' experimenter folders are re-keyed to the active experimenter. The ``<spectrograms_dir>/qlvm_v3/<map>/arrays_{fine,coarse}.npz`` reference arrays, their writer ``export-qlvm-reference-arrays`` and the old in-house model's ``<spectrograms_dir>/qlvm/`` arrays and ``qlvm_clusters_*.h5`` are no longer read or written
* the torus geodesic pullback metric (``vocal_features.usv_manifold_geodesic_metrics.pullback_metric`` in ``modeling_settings.json``) and the manifold filter atlas decode with the production cell of the map the run's coordinates come from, ``os_utils.qlvm_map_cell_directory(<map>)`` (``qlvm`` -> ``.../spectrograms/qlvm/qlvm``, the conditional maps -> ``.../spectrograms/qlvm/<map>``, ``qlvm_squeak`` -> ``.../spectrograms/qlvm/qlvm_squeak``; read without torch), a conditional cell at one fixed training-corpus quantile of its conditioning value (``pullback_condition_quantile`` / the atlas's ``condition_quantile``, default ``0.5``), taken from the code constants rather than a settings path the experimenter re-keying could rewrite (the earlier ``decoder_model_cell_directory`` and the legacy ``decoder_weights_npz_path`` settings are ignored)
* the pooled-embedding parquet caches record a fingerprint of the summaries they were built from (each summary's path, size and modification time) and rebuild themselves when a summary changes; the cohort cache is ``<spectrograms_dir>/embeddings/pooled_embeddings_production.parquet`` (``os_utils.POOLED_EMBEDDINGS_CACHE_NAME``), and it carries every map's coordinates and the ``qlvm_category`` labels; the consolidated store's ``qlvm/<session>/<map>`` coordinates and labels keep the columns as they were when built, so rebuild it after re-embedding
* the tuning figures tune to ``qlvm_category`` only and draw its region map and size its per-category axis from the category bundle's ``label_grid``; an unreadable bundle stops the figures (no placeholder panels)
* the region / category label of every map's analyses (the manifold position models and their torus macro score, the equal-region reweighting, model selection, CNN saliency segments, GLM-HMM targets, tuning, coactivity) is the regular map's ``qlvm_category`` for ``qlvm`` and the four conditional maps (``qlvm_duration``, ``qlvm_entropy``, ``qlvm_bandwidth``, ``qlvm_loudness``) alike; a summary whose categories are not assigned yet gives the label-free fallbacks it always did
* category ids chosen by meaning (e.g. the coactivity notebook's ``GROUP_A_IDS`` / ``GROUP_B_IDS``, now complex ``[4]`` and simple ``[1]`` of ``qlvm_category`` after the shipped ``category_order`` and ``category_descriptions``) have to be re-checked against the category directory's ``category_nomenclature.json``: categories are numbered by the build's ``category_order`` (by USV count when it is empty), so a rebuilt directory can number them differently

.. parsed-literal::

    ┌────────┬───┬───────────┬───────────┬───────────────┐
    │ usv_id ┆ … ┆ qlvm1     ┆ qlvm2     ┆ qlvm_category │
    │ ---    ┆   ┆ ---       ┆ ---       ┆ ---           │
    │ i64    ┆   ┆ f64       ┆ f64       ┆ i64           │
    ╞════════╪═══╪═══════════╪═══════════╪═══════════════╡
    │ 000000 ┆ … ┆ 0.4123    ┆ 0.8871    ┆ 3             │
    │ 000001 ┆ … ┆ 0.1902    ┆ 0.3320    ┆ 1             │
    │ …      ┆ … ┆ …         ┆ …         ┆ …             │
    └────────┴───┴───────────┴───────────┴───────────────┘

These columns are comparable across every session embedded into the same QLVM model, and are consumed by the categorical USV-tuning analysis in :ref:`Analyze <Analyze>` (*Compute neuronal tuning curves*). When a mask is present for a USV, the acoustic features are computed over the true SAM mask region; otherwise they fall back to the signal time-window.

Settings
^^^^^^^^

**Settings file and derived paths.** The */usv-playpen/_parameter_settings/processing_settings.json* file contains the settings for these steps, partially modifiable in the GUI but fully modifiable manually. The SAM2 / YOLO / squeak / noise model paths all derive from a single ``spectrograms_root`` (GUI: *Spectrogram models directory*): set that one directory and the paths below are filled as ``<root>/sam/...``, ``<root>/squeak/...`` and ``<root>/noise/...``. Set any individual path explicitly (in the JSON or via a CLI flag) to override its derived default; ``generate_masks.sam2_model_cfg`` is a config name, not a path, so it is never derived.

**QLVM cells.** The QLVM model is no longer derived from ``<root>/qlvm`` (the old in-house decoder and its watershed arrays, which would re-embed sessions on another torus and write ``qlvm_category`` / ``qlvm_supercategory`` back): when ``infer_qlvm_latents`` names no model (``model_cells`` empty), ``model_cells`` is filled with the production mapping of the masked, time-stretched cells under ``/mnt/falkner/Bartul/spectrograms/qlvm`` (``qlvm``, ``qlvm_duration``, ``qlvm_entropy``, ``qlvm_bandwidth``, ``qlvm_loudness``; see *Several QLVM models in one run*), ``masking_type`` is set to ``"sam"`` and ``time_stretch`` to ``true``, the preprocessing those cells were trained with (the contract check refuses anything else).

The package root and the mapping are the constants ``QLVM_MODEL_PACKAGE_ROOT`` / ``QLVM_PRODUCTION_MODEL_CELLS`` in ``os_utils``, not settings, because the GUI and CLI re-key every experimenter name in the processing settings (a path under another experimenter's directory would be rewritten to the active experimenter's directory).

An explicitly configured ``model_cells`` is left untouched, ``masking_type`` and ``time_stretch`` included; with no ``spectrograms_root`` and an empty ``model_cells`` the run stops, since model package cells are the only models *Infer QLVM latents* embeds with (the single-model keys ``model_cell_directory``, ``weights_npz_path``, ``reference_arrays_fine_npz_path`` / ``reference_arrays_coarse_npz_path`` and the lattice keys ``lattice_type``, ``n_points``, ``korobov_a``, ``fib_m`` are retired).

**Squeak QLVM cell.** The squeak QLVM cell (``infer_qlvm_squeak_latents.model_cell_directory``) is derived the same way: when it is empty it is filled with the production squeak cell ``QLVM_SQUEAK_PACKAGE_ROOT/QLVM_SQUEAK_PRODUCTION_CELL`` (``os_utils`` constants: ``/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_squeak``), for the same reason a constant rather than a setting; a configured cell is left alone. Name another with ``--model-cell-directory``, which is applied after the experimenter re-keying; a path under another experimenter's directory written into the JSON would be re-keyed like any other, except a path inside ``QLVM_MODEL_PACKAGE_ROOT`` or ``QLVM_CATEGORY_BUNDLE_DIRECTORY``, which the re-keying always leaves as it is.

**Run-time copy.** The derived paths are filled into the run's own copy of the settings: the GUI hands the processing run a copy of the settings it saves, so none of them is written back into *processing_settings.json*, whose derived keys stay empty.

.. code-block:: json

    "spectrograms_root": "/mnt/falkner/Bartul/spectrograms"

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`spectrograms_root <proc-spectrograms-root>`
     - see JSON above
     - path to a directory

.. _proc-spectrograms-root:

``spectrograms_root``
   The single root from which the SAM2 / YOLO / squeak / noise model paths are derived (GUI: *Spectrogram models directory*).

   :Options: path (string) to a directory
   :Default: ``/mnt/falkner/Bartul/spectrograms``
   :Notes: see the paragraphs above for what is derived from it and what is not.

*Generate spectrograms* (``generate_spectrograms``):

.. code-block:: json

    "generate_spectrograms": {
        "num_freq_bins": 128,
        "num_time_bins": 128,
        "nperseg": 2048,
        "min_freq": 30000,
        "max_freq": 120000,
        "hop_length": 512,
        "window": "blackmanharris",
        "offset": 0.0,
        "normalize": true,
        "top_db": 80.0
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`num_freq_bins <proc-generate-spectrograms-num-freq-bins>`
     - ``128``
     - number of bins
   * - :ref:`num_time_bins <proc-generate-spectrograms-num-time-bins>`
     - ``128``
     - number of bins
   * - :ref:`nperseg <proc-generate-spectrograms-nperseg>`
     - ``2048``
     - samples
   * - :ref:`min_freq <proc-generate-spectrograms-min-freq>`
     - ``30000``
     - Hz
   * - :ref:`max_freq <proc-generate-spectrograms-max-freq>`
     - ``120000``
     - Hz
   * - :ref:`hop_length <proc-generate-spectrograms-hop-length>`
     - ``512``
     - samples, or ``null``
   * - :ref:`window <proc-generate-spectrograms-window>`
     - ``'blackmanharris'``
     - STFT window name
   * - :ref:`offset <proc-generate-spectrograms-offset>`
     - ``0.0``
     - seconds
   * - :ref:`normalize <proc-generate-spectrograms-normalize>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`top_db <proc-generate-spectrograms-top-db>`
     - ``80.0``
     - dB

.. _proc-generate-spectrograms-num-freq-bins:

``num_freq_bins``
   Number of spectrogram frequency bins (output height).

   :Options: a number of bins
   :Default: ``128``

.. _proc-generate-spectrograms-num-time-bins:

``num_time_bins``
   Number of spectrogram time bins (output width).

   :Options: a number of bins
   :Default: ``128``

.. _proc-generate-spectrograms-nperseg:

``nperseg``
   STFT window length / n_fft (samples).

   :Options: a number of samples
   :Default: ``2048``

.. _proc-generate-spectrograms-min-freq:

``min_freq``
   Lower frequency cutoff (Hz).

   :Options: a frequency in Hz
   :Default: ``30000``

.. _proc-generate-spectrograms-max-freq:

``max_freq``
   Upper frequency cutoff (Hz).

   :Options: a frequency in Hz
   :Default: ``120000``

.. _proc-generate-spectrograms-hop-length:

``hop_length``
   STFT hop length (samples).

   :Options:
      * a number of samples
      * ``null`` — defaults to ``nperseg // 4``
   :Default: ``512``

.. _proc-generate-spectrograms-window:

``window``
   STFT window function.

   :Options: a window function name
   :Default: ``'blackmanharris'``

.. _proc-generate-spectrograms-offset:

``offset``
   Time padding added before each USV onset and after each offset (seconds).

   :Options: a number of seconds
   :Default: ``0.0``

.. _proc-generate-spectrograms-normalize:

``normalize``
   Whether to min-max normalize each spectrogram to [0, 1].

   :Options:
      * ``true`` — min-max normalize each spectrogram to [0, 1]
      * ``false`` — leave it unnormalized
   :Default: ``true``

.. _proc-generate-spectrograms-top-db:

``top_db``
   Dynamic range kept by the dB conversion (``librosa.power_to_db``): each channel spectrogram is clipped at this many dB below its own peak before the channels are averaged.

   :Options: a number of dB
   :Default: ``80.0`` (librosa's own default)
   :Notes: part of the input the QLVM and mask models were trained on, so a different value feeds them spectrograms they have not seen.

*Generate masks* (``generate_masks``):

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
        "deterministic": true,
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

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`method <proc-generate-masks-method>`
     - ``'boxprompt'``
     - ``'boxprompt'``
   * - :ref:`detector <proc-generate-masks-detector>`
     - ``'yolo'``
     - ``'yolo'``, ``'cc'``
   * - :ref:`sam2_model_dir <proc-generate-masks-sam2-model-dir>`
     - ``''`` (derived)
     - path to a directory
   * - :ref:`sam2_model_cfg <proc-generate-masks-sam2-model-cfg>`
     - see JSON above
     - SAM2 config name / path
   * - :ref:`sam2_model_path <proc-generate-masks-sam2-model-path>`
     - ``''`` (derived)
     - path to a checkpoint
   * - :ref:`yolo_weights <proc-generate-masks-yolo-weights>`
     - ``''`` (derived)
     - path to ``best.pt`` weights
   * - :ref:`yolo_conf <proc-generate-masks-yolo-conf>`
     - ``0.25``
     - confidence threshold
   * - :ref:`yolo_iou <proc-generate-masks-yolo-iou>`
     - ``0.7``
     - NMS IoU
   * - :ref:`yolo_imgsz <proc-generate-masks-yolo-imgsz>`
     - ``128``
     - px
   * - :ref:`deterministic <proc-generate-masks-deterministic>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`mask_cmap <proc-generate-masks-mask-cmap>`
     - ``'viridis'``
     - colormap name
   * - :ref:`duration_min <proc-generate-masks-duration-min>`
     - ``5``
     - spectrogram time bins
   * - :ref:`batch_size <proc-generate-masks-batch-size>`
     - ``12``
     - spectrograms per batch
   * - :ref:`multimask_output <proc-generate-masks-multimask-output>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`iou_floor <proc-generate-masks-iou-floor>`
     - ``0.3``
     - predicted-IoU threshold
   * - :ref:`drop_below_iou <proc-generate-masks-drop-below-iou>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`split_disconnected <proc-generate-masks-split-disconnected>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`max_iters <proc-generate-masks-max-iters>`
     - ``1``
     - iterations per box
   * - :ref:`merge_instances <proc-generate-masks-merge-instances>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`merge_iou <proc-generate-masks-merge-iou>`
     - ``0.5``
     - IoU
   * - :ref:`merge_containment <proc-generate-masks-merge-containment>`
     - ``0.8``
     - containment fraction
   * - :ref:`mask_intensity_floor <proc-generate-masks-mask-intensity-floor>`
     - ``0.0``
     - normalized intensity
   * - :ref:`tiny_mask_floor_px <proc-generate-masks-tiny-mask-floor-px>`
     - ``12``
     - pixels
   * - :ref:`min_box_area <proc-generate-masks-min-box-area>`
     - ``0``
     - px², ``0`` disables

.. _proc-generate-masks-method:

``method``
   Mask-generation method (``boxprompt`` = box-prompted SAM2 segmentation).

   :Options: ``'boxprompt'`` — box-prompted SAM2 segmentation
   :Default: ``'boxprompt'``

.. _proc-generate-masks-detector:

``detector``
   Box detector backend (``yolo`` learned detector or ``cc`` connected-component baseline).

   :Options:
      * ``'yolo'`` — learned detector
      * ``'cc'`` — connected-component baseline
   :Default: ``'yolo'``

.. _proc-generate-masks-sam2-model-dir:

``sam2_model_dir``
   SAM2 model directory.

   :Options: path (string) to a directory; empty derives it from ``spectrograms_root``
   :Default: ``''`` (derived from ``spectrograms_root``)
   :Notes: config/checkpoint resolve against it; the step changes into it.

.. _proc-generate-masks-sam2-model-cfg:

``sam2_model_cfg``
   SAM2 config name/path.

   :Options: a SAM2 config name/path
   :Default: ``configs/sam2.1/sam2.1_hiera_b+.yaml``
   :Notes: resolved from the SAM2 install's Hydra search path.

.. _proc-generate-masks-sam2-model-path:

``sam2_model_path``
   SAM2 checkpoint path.

   :Options: path (string) to a checkpoint; empty derives it from ``spectrograms_root``
   :Default: ``''`` (derived from ``spectrograms_root``)

.. _proc-generate-masks-yolo-weights:

``yolo_weights``
   Trained YOLO ``best.pt`` weights path.

   :Options: path (string) to the weights; empty derives it from ``spectrograms_root``
   :Default: ``''`` (derived from ``spectrograms_root``)

.. _proc-generate-masks-yolo-conf:

``yolo_conf``
   YOLO confidence threshold.

   :Options: a confidence threshold (lower → more recall)
   :Default: ``0.25``

.. _proc-generate-masks-yolo-iou:

``yolo_iou``
   YOLO NMS IoU.

   :Options: an IoU (raise to keep stacked USVs)
   :Default: ``0.7``

.. _proc-generate-masks-yolo-imgsz:

``yolo_imgsz``
   YOLO inference image size (px).

   :Options: an image size in px
   :Default: ``128``

.. _proc-generate-masks-deterministic:

``deterministic``
   Disable cuDNN's autotuner so the masks reproduce across processes.

   :Options:
      * ``true`` — autotuner disabled, masks reproduce across processes
      * ``false`` — autotuner on
   :Default: ``true``
   :Notes: with the autotuner on, the algorithm choice varies with the GPU state at process start and borderline faint USVs fall on either side of the detector's confidence gate.

.. _proc-generate-masks-mask-cmap:

``mask_cmap``
   Colormap used to render each spectrogram to RGB before detection.

   :Options: a colormap name
   :Default: ``'viridis'``

.. _proc-generate-masks-duration-min:

``duration_min``
   Minimum USV duration to segment, in SPECTROGRAM TIME BINS (not milliseconds).

   :Options: a number of spectrogram time bins
   :Default: ``5`` (~10 ms)
   :Notes: shorter rows are skipped without any detection attempt and end up with ``mask_number`` 0. At the shipped spectrogram settings the grid runs at ~0.5 bins/ms, so a value of 10 silently excluded every USV under ~20 ms — well above DAS's own 15 ms ``segment_minlen``, so genuine USVs were discarded. Set to 5 (~10 ms).

.. _proc-generate-masks-batch-size:

``batch_size``
   Number of spectrograms per SAM2 batch.

   :Options: a number of spectrograms
   :Default: ``12``

.. _proc-generate-masks-multimask-output:

``multimask_output``
   Whether SAM2 returns multiple candidate masks per box (the best is kept).

   :Options:
      * ``true`` — multiple candidate masks per box, the best is kept
      * ``false`` — one mask per box
   :Default: ``true``

.. _proc-generate-masks-iou-floor:

``iou_floor``
   SAM2 self-predicted-IoU threshold used by ``drop_below_iou``.

   :Options: a predicted-IoU threshold
   :Default: ``0.3``
   :Notes: set to 0.30 (not 0.70): the mask population is strongly good-quality (median 0.925) and its low tail is continuous, so only the 0-0.30 band is unambiguously broken.

.. _proc-generate-masks-drop-below-iou:

``drop_below_iou``
   Whether to drop masks whose SAM2 self-predicted IoU is below ``iou_floor``.

   :Options:
      * ``true`` — drop masks whose SAM2 self-predicted IoU is below ``iou_floor``
      * ``false`` — keep them
   :Default: ``true``
   :Notes: enabled, but paired with a low ``iou_floor``: measured over 1,209 masks the predicted-IoU distribution has median 0.925 with no gap in its low tail, so the original 0.70 floor cut through a continuum -- 28% of what it removed sat within 0.05 of the line. A 0.30 floor removes only the ~18% of low-IoU masks SAM2 rates as outright wrong and leaves the grey zone intact.

.. _proc-generate-masks-split-disconnected:

``split_disconnected``
   Split a mask with disconnected components into separate instances.

   :Options:
      * ``true`` — split a mask with disconnected components into separate instances
      * ``false`` — keep it as one instance
   :Default: ``true``

.. _proc-generate-masks-max-iters:

``max_iters``
   Maximum SAM2 prompt-refinement iterations per box.

   :Options: a number of iterations
   :Default: ``1``

.. _proc-generate-masks-merge-instances:

``merge_instances``
   Merge overlapping per-box mask instances.

   :Options:
      * ``true`` — merge overlapping per-box mask instances
      * ``false`` — keep them separate
   :Default: ``true``

.. _proc-generate-masks-merge-iou:

``merge_iou``
   IoU above which two instances are merged.

   :Options: an IoU
   :Default: ``0.5``

.. _proc-generate-masks-merge-containment:

``merge_containment``
   Containment fraction above which one instance is absorbed into another.

   :Options: a containment fraction
   :Default: ``0.8``

.. _proc-generate-masks-mask-intensity-floor:

``mask_intensity_floor``
   Minimum normalized spectrogram intensity for a pixel to remain in a mask.

   :Options: a normalized spectrogram intensity
   :Default: ``0.0``

.. _proc-generate-masks-tiny-mask-floor-px:

``tiny_mask_floor_px``
   Drop masks smaller than this many pixels.

   :Options: a number of pixels
   :Default: ``12``

.. _proc-generate-masks-min-box-area:

``min_box_area``
   Drop detector boxes smaller than this area (px²).

   :Options: an area in px²; ``0`` disables
   :Default: ``0``

When left empty (the default) the SAM2/YOLO paths are derived from ``spectrograms_root`` above; whether derived or set explicitly, they are stored in the canonical ``/mnt/falkner/...`` lab-share form and translated to the host's mount root (e.g. ``/Volumes/falkner`` on macOS) via ``configure_path``, the same handling as the DAS / Vocalocator model paths. ``sam2_model_cfg`` is a config name resolved inside the SAM2 install, not a mount path. Both ``sam2`` and ``ultralytics`` (usv-playpen core dependencies) must be installed.

*Compute USV features* (``compute_usv_acoustic_features``):

.. code-block:: json

    "compute_usv_acoustic_features": {
        "low_energy_frac": 0.05,
        "high_energy_frac": 0.95
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`low_energy_frac <proc-compute-usv-acoustic-features-low-energy-frac>`
     - ``0.05``
     - cumulative-energy fraction
   * - :ref:`high_energy_frac <proc-compute-usv-acoustic-features-high-energy-frac>`
     - ``0.95``
     - cumulative-energy fraction

.. _proc-compute-usv-acoustic-features-low-energy-frac:

``low_energy_frac``
   Lower edge of the cumulative-energy band used for spectral bandwidth.

   :Options: a cumulative-energy fraction
   :Default: ``0.05``

.. _proc-compute-usv-acoustic-features-high-energy-frac:

``high_energy_frac``
   Upper edge of the cumulative-energy band used for spectral bandwidth.

   :Options: a cumulative-energy fraction
   :Default: ``0.95``

*Produce noise labels* (``detect_usv_noise``):

.. code-block:: json

    "detect_usv_noise": {
        "noise_model_path": "",
        "exclude_metadata_audio_channels": true,
        "batch_size": 64
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`noise_model_path <proc-detect-usv-noise-noise-model-path>`
     - ``''`` (derived)
     - path to a ``.pt`` bundle
   * - :ref:`exclude_metadata_audio_channels <proc-detect-usv-noise-exclude-metadata-audio-channels>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`batch_size <proc-detect-usv-noise-batch-size>`
     - ``64``
     - segments per forward pass

.. _proc-detect-usv-noise-noise-model-path:

``noise_model_path``
   Path to the noise model bundle (``.pt``).

   :Options: path (string) to a bundle; empty derives it
   :Default: ``''`` — derived from *Spectrogram models directory* as ``noise/noise_timemil_ens5_n4680_20260926.pt``
   :Notes: the bundle also fixes the decision (exclusion cut-off 0.14; see *Produce noise labels* above), so there is no threshold setting.

.. _proc-detect-usv-noise-exclude-metadata-audio-channels:

``exclude_metadata_audio_channels``
   Drop channels the session metadata marks as excluded from the spectrogram average.

   :Options:
      * ``true`` — drop the metadata-excluded channels
      * ``false`` — keep them in the average
   :Default: ``true``

.. _proc-detect-usv-noise-batch-size:

``batch_size``
   Segments per forward pass at the typical vocalization length.

   :Options: a number of segments
   :Default: ``64``
   :Notes: a batch is budgeted at ``batch_size`` x 128 frame slots and padded to its longest vocalization, so one long vocalization never inflates a batch (raise it on a GPU with memory to spare, lower it when several sessions run in parallel).

*Train noise model* (``train_noise_model``, command line only; see *Noise model* below):

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

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`calibration_path <proc-train-noise-model-calibration-path>`
     - ``''``
     - path to a JSON file
   * - :ref:`exclude_metadata_audio_channels <proc-train-noise-model-exclude-metadata-audio-channels>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`n_workers <proc-train-noise-model-n-workers>`
     - ``16``
     - number of threads
   * - :ref:`seeds <proc-train-noise-model-seeds>`
     - five seeds, see JSON above
     - list of integer seeds
   * - :ref:`epochs <proc-train-noise-model-epochs>`
     - ``40``
     - epochs per member
   * - :ref:`batch_size <proc-train-noise-model-batch-size>`
     - ``32``
     - segments per batch
   * - :ref:`learning_rate <proc-train-noise-model-learning-rate>`
     - ``0.001``
     - Adam learning rate
   * - :ref:`weight_decay <proc-train-noise-model-weight-decay>`
     - ``0.0001``
     - Adam weight decay
   * - :ref:`label_smoothing <proc-train-noise-model-label-smoothing>`
     - ``0.05``
     - smoothing amount

.. _proc-train-noise-model-calibration-path:

``calibration_path``
   JSON holding the ``calibration`` table and the ``decision`` block the bundle carries.

   :Options: path (string) to a JSON file
   :Default: ``''``
   :Notes: the ``calibration`` table holds rows of ``threshold``, ``precision``, ``recall``, ...; the ``decision`` block holds ``exclude_at_or_above``, ``noise_at_or_above``, ``held_out_precision``, ``held_out_recall``, ``real_vocalizations_excluded_per_10000`` (a file written before the rename may name it ``real_calls_excluded_per_10000``; it is read under the new name) and any provenance. ``calibration_source`` and ``calibration_population`` are copied too when present. Training does not measure these, so they must describe this labels set and recipe.

.. _proc-train-noise-model-exclude-metadata-audio-channels:

``exclude_metadata_audio_channels``
   Drop channels the session metadata marks as excluded from the spectrogram average.

   :Options:
      * ``true`` — drop the metadata-excluded channels
      * ``false`` — keep them in the average
   :Default: ``true``
   :Notes: keep it as *Produce noise labels* runs.

.. _proc-train-noise-model-n-workers:

``n_workers``
   Sessions whose audio is read concurrently (threads) while the inputs are built.

   :Options: a number of threads
   :Default: ``16``
   :Notes: reading short windows of the broadband memmap on the lab share is latency-bound (measured with the earlier 24-wav input: ~14 s per session serially).

.. _proc-train-noise-model-seeds:

``seeds``
   One seed per ensemble member.

   :Options: a list of integer seeds, one per ensemble member
   :Default: ``[20269914, 20269915, 20269916, 20269917, 20269918]`` (the production ensemble's five)

.. _proc-train-noise-model-epochs:

``epochs``
   Training epochs per member.

   :Options: a number of epochs
   :Default: ``40``

.. _proc-train-noise-model-batch-size:

``batch_size``
   Segments per training batch.

   :Options: a number of segments
   :Default: ``32``

.. _proc-train-noise-model-learning-rate:

``learning_rate``
   Adam learning rate, cosine-annealed to zero over the run.

   :Options: a learning rate
   :Default: ``0.001``

.. _proc-train-noise-model-weight-decay:

``weight_decay``
   Adam weight decay.

   :Options: a weight decay
   :Default: ``0.0001``

.. _proc-train-noise-model-label-smoothing:

``label_smoothing``
   Targets are smoothed towards 0.5 by this amount.

   :Options: a smoothing amount
   :Default: ``0.05``

*Produce squeak labels* (``detect_usv_squeaks``):

.. code-block:: json

    "detect_usv_squeaks": {
        "squeak_model_path": "",
        "exclude_metadata_audio_channels": true,
        "batch_size": 64
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`squeak_model_path <proc-detect-usv-squeaks-squeak-model-path>`
     - ``""``
     - path to a ``.pt`` bundle, or empty (derived)
   * - :ref:`exclude_metadata_audio_channels <proc-detect-usv-squeaks-exclude-metadata-audio-channels>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`batch_size <proc-detect-usv-squeaks-batch-size>`
     - ``64``
     - integer > 0

.. _proc-detect-usv-squeaks-squeak-model-path:

``squeak_model_path``
   Path to the vocal-class (usv / squeak / overlap) model bundle (``.pt``).

   :Options: a path to a ``.pt`` bundle, or empty
   :Default: ``""`` — left empty it is derived from *Spectrogram models directory* as ``squeak/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt``
   :Notes: the bundle also fixes the squeak-extent threshold (0.6; see *Produce squeak labels* above) and every input constant, so there is no threshold setting.

.. _proc-detect-usv-squeaks-exclude-metadata-audio-channels:

``exclude_metadata_audio_channels``
   Drop channels the session metadata marks as excluded from the spectrogram average.

   :Options:
      * ``true`` — drop the excluded channels
      * ``false`` — keep every channel
   :Default: ``true``

.. _proc-detect-usv-squeaks-batch-size:

``batch_size``
   Segments per forward pass at the typical window length.

   :Options: a positive integer
   :Default: ``64``
   :Notes: a batch is budgeted at ``batch_size`` x 128 frame slots and padded to its longest window, so one long window never inflates a batch.

*Train vocal-class model* (``train_usv_squeak_model``, command line only; see *Vocal-class model* below):

.. code-block:: json

    "train_usv_squeak_model": {
        "label_sets": [
          {
            "name": "r1",
            "labels_csv": "/mnt/falkner/Bartul/PC_transfer/usv_squeak_labelling/usv_squeak_labels.csv",
            "sample_csv": "/mnt/falkner/Bartul/PC_transfer/usv_squeak_labelling/usv_squeak_sample.csv"
          },
          {
            "name": "r2",
            "labels_csv": "/mnt/falkner/Bartul/PC_transfer/usv_squeak_labelling_r2/usv_squeak_labels_r2.csv",
            "sample_csv": "/mnt/falkner/Bartul/PC_transfer/usv_squeak_labelling_r2/usv_squeak_sample_r2.csv"
          }
        ],
        "label_overrides": [
          {
            "name": "r1",
            "override_csv": "/mnt/falkner/Bartul/PC_transfer/usv_squeak_labelling/usv_squeak_review_labels.csv"
          }
        ],
        "pretrained": true,
        "span_threshold": 0.6,
        "exclude_metadata_audio_channels": true,
        "n_workers": 16,
        "seeds": [0, 1, 2, 3, 4],
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05,
        "frame_loss_weight": 1.0,
        "class_weighted": false
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`label_sets <proc-train-usv-squeak-model-label-sets>`
     - two rounds, see JSON above
     - list of ``name`` / ``labels_csv`` / ``sample_csv`` objects
   * - :ref:`label_overrides <proc-train-usv-squeak-model-label-overrides>`
     - one review, see JSON above
     - list of ``name`` / ``override_csv`` objects
   * - :ref:`pretrained <proc-train-usv-squeak-model-pretrained>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`span_threshold <proc-train-usv-squeak-model-span-threshold>`
     - ``0.6``
     - probability threshold
   * - :ref:`exclude_metadata_audio_channels <proc-train-usv-squeak-model-exclude-metadata-audio-channels>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`n_workers <proc-train-usv-squeak-model-n-workers>`
     - ``16``
     - integer > 0
   * - :ref:`seeds <proc-train-usv-squeak-model-seeds>`
     - ``[0, 1, 2, 3, 4]``
     - list of integers, one per member
   * - :ref:`epochs <proc-train-usv-squeak-model-epochs>`
     - ``40``
     - integer > 0
   * - :ref:`batch_size <proc-train-usv-squeak-model-batch-size>`
     - ``32``
     - integer > 0
   * - :ref:`learning_rate <proc-train-usv-squeak-model-learning-rate>`
     - ``0.001``
     - number > 0
   * - :ref:`weight_decay <proc-train-usv-squeak-model-weight-decay>`
     - ``0.0001``
     - number >= 0
   * - :ref:`label_smoothing <proc-train-usv-squeak-model-label-smoothing>`
     - ``0.05``
     - number >= 0
   * - :ref:`frame_loss_weight <proc-train-usv-squeak-model-frame-loss-weight>`
     - ``1.0``
     - number >= 0
   * - :ref:`class_weighted <proc-train-usv-squeak-model-class-weighted>`
     - ``false``
     - ``true``, ``false``

.. _proc-train-usv-squeak-model-label-sets:

``label_sets``
   The label sets the ensemble is trained on.

   :Options: a list of label sets, each with a ``name``, its ``labels_csv`` (``panel_id``, ``label`` 0 usv / 1 squeak / 2 overlap / 3 unsure, ``squeak_extents_s``) and its ``sample_csv`` (``panel_id``, ``session_dir``, ``row_index``, ``start``, ``stop``)
   :Default: the production bundle's two labelling rounds (see the JSON above)

.. _proc-train-usv-squeak-model-label-overrides:

``label_overrides``
   Review CSVs in the labels format whose panels replace those of the named set.

   :Options: a list of overrides, each with a ``name`` (the set it applies to) and an ``override_csv``
   :Default: the production bundle's review of the first round (see the JSON above)

.. _proc-train-usv-squeak-model-pretrained:

``pretrained``
   Initialize every member's trunk from the noise ensemble (``detect_usv_noise.noise_model_path``; member ``seed mod 5``).

   :Options:
      * ``true`` — load each member's trunk from the noise ensemble
      * ``false`` — train from the plain seeded initialization
   :Default: ``true``

.. _proc-train-usv-squeak-model-span-threshold:

``span_threshold``
   Frame squeak-probability threshold of the squeak-extent rule the bundle carries.

   :Options: a probability threshold
   :Default: ``0.6``
   :Notes: chosen on session-grouped cross-fitted predictions outside the package; not tuned here.

.. _proc-train-usv-squeak-model-exclude-metadata-audio-channels:

``exclude_metadata_audio_channels``
   Drop channels the session metadata marks as excluded from the spectrogram average.

   :Options:
      * ``true`` — drop the excluded channels
      * ``false`` — keep every channel
   :Default: ``true``
   :Notes: keep it as *Produce squeak labels* runs.

.. _proc-train-usv-squeak-model-n-workers:

``n_workers``
   Sessions whose audio is read concurrently (threads) while the inputs are built.

   :Options: a positive integer
   :Default: ``16``

.. _proc-train-usv-squeak-model-seeds:

``seeds``
   One seed per ensemble member.

   :Options: a list of integers; its length is the number of members
   :Default: ``[0, 1, 2, 3, 4]`` (the production ensemble's five)

.. _proc-train-usv-squeak-model-epochs:

``epochs``
   Training epochs per member.

   :Options: a positive integer
   :Default: ``40``

.. _proc-train-usv-squeak-model-batch-size:

``batch_size``
   Segments per training batch.

   :Options: a positive integer
   :Default: ``32``

.. _proc-train-usv-squeak-model-learning-rate:

``learning_rate``
   Adam learning rate, cosine-annealed to zero over the run.

   :Options: a positive number
   :Default: ``0.001``

.. _proc-train-usv-squeak-model-weight-decay:

``weight_decay``
   Adam weight decay.

   :Options: a non-negative number
   :Default: ``0.0001``

.. _proc-train-usv-squeak-model-label-smoothing:

``label_smoothing``
   Label smoothing of the class cross-entropy.

   :Options: a non-negative number
   :Default: ``0.05``

.. _proc-train-usv-squeak-model-frame-loss-weight:

``frame_loss_weight``
   Weight of the frame squeak loss relative to the class loss.

   :Options: a non-negative number
   :Default: ``1.0``

.. _proc-train-usv-squeak-model-class-weighted:

``class_weighted``
   Weight the class loss by inverse training class frequency.

   :Options:
      * ``true`` — inverse-frequency class weights
      * ``false`` — unweighted class loss
   :Default: ``false``
   :Notes: training stops when a class has no label (its weight would be infinite).

*Embed squeaks in a QLVM torus* (``infer_qlvm_squeak_latents``; runs as the last part of *Infer QLVM latents*, with no GUI field of its own):

.. code-block:: json

    "infer_qlvm_squeak_latents": {
        "model_cell_directory": "",
        "exclude_metadata_audio_channels": true,
        "lattice_batch_size": 2048,
        "data_batch_size": 512
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`model_cell_directory <proc-infer-qlvm-squeak-latents-model-cell-directory>`
     - ``""``
     - path to a squeak QLVM cell, or empty (production cell)
   * - :ref:`exclude_metadata_audio_channels <proc-infer-qlvm-squeak-latents-exclude-metadata-audio-channels>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`lattice_batch_size <proc-infer-qlvm-squeak-latents-lattice-batch-size>`
     - ``2048``
     - integer > 0
   * - :ref:`data_batch_size <proc-infer-qlvm-squeak-latents-data-batch-size>`
     - ``512``
     - integer > 0

.. _proc-infer-qlvm-squeak-latents-model-cell-directory:

``model_cell_directory``
   The squeak QLVM cell.

   :Options: a ``train-qlvm`` cell with a ``training_contract.json``, or an old-layout ``phase3_BBVs_qlvm`` cell
   :Default: ``""`` — left empty it is filled with the production cell ``QLVM_SQUEAK_PACKAGE_ROOT/QLVM_SQUEAK_PRODUCTION_CELL`` (``/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_squeak``) whenever ``spectrograms_root`` is set (see *Embed squeaks in a QLVM torus* above)

.. _proc-infer-qlvm-squeak-latents-exclude-metadata-audio-channels:

``exclude_metadata_audio_channels``
   Drop channels the session metadata marks as excluded from the spectrogram average.

   :Options:
      * ``true`` — drop the excluded channels
      * ``false`` — keep every channel
   :Default: ``true``
   :Notes: keep it equal to the *Produce squeak labels* run.

.. _proc-infer-qlvm-squeak-latents-lattice-batch-size:

``lattice_batch_size``
   Lattice points decoded and scored per block.

   :Options: a positive integer
   :Default: ``2048``
   :Notes: lower it to cut memory.

.. _proc-infer-qlvm-squeak-latents-data-batch-size:

``data_batch_size``
   Squeaks whose lattice posteriors are computed together.

   :Options: a positive integer
   :Default: ``512``
   :Notes: memory grows with this times the lattice size.

*Infer QLVM latents* (``infer_qlvm_latents``):

The USV maps hold USVs only. A pure squeak (``squeak`` true and ``usv`` false, written by *Produce squeak labels*) -- a squeak with no USV in it -- is skipped by every cell, on both routes (not embedded, and left out of a package's own coordinates), and gets null coordinates and null labels; squeaks have their own map (*Embed squeaks in a QLVM torus*). A segment holding both (``usv`` and ``squeak`` true) is embedded like any other USV, and rows with empty booleans (noise, or too short to score) are treated as before. The rule needs the two booleans, so a summary without them stops the run: run *Produce noise labels* and *Produce squeak labels* first.

.. code-block:: json

    "infer_qlvm_latents": {
        "model_cells": {},
        "prefer_package_values": true,
        "latent_dim": 2,
        "time_stretch": true,
        "masking_type": "sam",
        "target_shape": [128, 128],
        "length_threshold": null,
        "lattice_batch_size": 4096,
        "data_batch_size": 8192
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`model_cells <proc-infer-qlvm-latents-model-cells>`
     - ``{}``
     - column prefix → model package cell (empty: production mapping)
   * - :ref:`prefer_package_values <proc-infer-qlvm-latents-prefer-package-values>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`latent_dim <proc-infer-qlvm-latents-latent-dim>`
     - ``2``
     - must match every cell's contract
   * - :ref:`time_stretch <proc-infer-qlvm-latents-time-stretch>`
     - ``true``
     - ``true``, ``false`` (must match every cell's contract)
   * - :ref:`masking_type <proc-infer-qlvm-latents-masking-type>`
     - ``"sam"``
     - ``"sam"``, ``"none"`` (must match every cell's contract)
   * - :ref:`target_shape <proc-infer-qlvm-latents-target-shape>`
     - ``[128, 128]``
     - ``(freq, time)`` shape
   * - :ref:`length_threshold <proc-infer-qlvm-latents-length-threshold>`
     - ``null``
     - ``null`` (from the contract), or time bins equal to every contract's
   * - :ref:`lattice_batch_size <proc-infer-qlvm-latents-lattice-batch-size>`
     - ``4096``
     - integer > 0
   * - :ref:`data_batch_size <proc-infer-qlvm-latents-data-batch-size>`
     - ``8192``
     - integer > 0

.. _proc-infer-qlvm-latents-model-cells:

``model_cells``
   Column prefix → model package cell, e.g. ``{"qlvm_duration": ".../spectrograms/qlvm/qlvm_duration"}``. One run places the session on the torus of **every** listed cell.

   :Options: a mapping from column prefix to model package cell directory
   :Default: ``{}`` — filled with the production mapping when ``spectrograms_root`` is set, see above; left empty, the run stops
   :Notes:
      **Columns written:** per prefix ``P``, the two float columns ``P1`` / ``P2`` (torus coordinates in ``[0, 1)``; nulls for USVs a cell does not place) and nothing else (the regular map's ``qlvm_category`` is written afterwards by ``assign-qlvm-categories``); no model column is written. The legacy ``qlvm_model`` column and the earlier ``P1`` / ``P2``, category (``P_category``, and the legacy coarse-level ``P_supercategory``) and category confidence (``P_category_agreement`` / ``P_category_uncertain``) columns of the listed prefixes are removed first, so ``assign-qlvm-categories`` has to run again after a re-embedding (a ``qlvm1`` / ``qlvm2`` pair and its ``qlvm_category`` are kept unless ``qlvm`` is a listed prefix).

      **Validation:** a prefix must be a Python identifier (letters, digits, underscores, not starting with a digit), may appear once, and ``P1`` / ``P2`` may not be another column of the USV summary (the production coordinate columns are allowed).

      **What each cell brings:** everything model-specific: the decoder from its ``checkpoint.tar`` (read without torch; the legacy or ReLU head is read from the weights), the input normalization and duration window from its ``training_contract.json``, and the lattice from the contract's Fibonacci ``embedding_fib_m`` (``24``, 46,368 points; built in float32 exactly as the torch driver that embedded the package corpus built it, since the exact lattice, as JAX holds it in float32, sits up to ~1e-3 away and moved 8 % of 9,390 corpus USVs by more than 1e-3).

      Each cell is loaded once per run and checked against ``masking_type``, ``target_shape``, ``time_stretch``, ``latent_dim`` and ``length_threshold``, so all listed cells must share them (``masking_type`` ``"sam"`` for phase 9 cells, ``"none"`` for phase 6, 10 and 11 cells). Package cells were trained on min-maxed spectrograms (phase 6 and 10 cells also with a 0.2 loudness floor), which is applied automatically.

      **Conditional cells:** they decode each USV at a value derived from its own conditioning value — the normalized duration; the mean frequency of its SAM-masked spectrogram, which needs the session's masks even though the input is unmasked; and, in phase 11 (``qlvm_models_latest/v3``), ``clip(freq_bandwidth_hz / 90000, 0, 1)`` from the summary (run *Generate USV acoustic features* first) or the loudness mapped through the contract's ``db_range`` (the masked, variance-weighted image-level dB, measured from the session's ``hpss_filtered`` audio with the spectrogram generator's own slice and STFT; about 40–90 ms per USV on one core); a cell ``train-qlvm`` trained on spectral entropy reads the summary's ``spectral_entropy`` (a summary without the column stops the run), scales it by the contract's ``entropy_min`` / ``entropy_max`` (the training split's range) and clamps it to ``[0, 1]``, the log counting the USVs outside that range. USVs without a value get null columns.

      Phase 10 cells (duration, mean frequency) decode at the frozen corpus bin mean of ``condition_bins.npz`` (the lattice is decoded once per distinct bin mean, up to 32 per session).

      Phase 11 cells (duration, mean frequency, bandwidth, loudness, and spectral entropy cells trained with ``train-qlvm``) follow the contract's ``condition.decode``: ``"exact"`` (duration, bandwidth) is the value clamped to the cell's training range ``[train_c_min, train_c_max]``, ``"grid"`` (mean frequency, loudness, spectral entropy) the nearest point of the cell's ``decode_grid`` (0.0025 apart); the log counts the USVs outside the range each rule applies (the training range for ``"exact"``, the ends of the decode grid for ``"grid"``). This is how the package embedded every corpus USV, and the lattice is decoded once per distinct value (about 100–250 per ~900-USV session).

      See *Several QLVM models in one run* below for the production mapping and how each model's coordinates are obtained.

.. _proc-infer-qlvm-latents-prefer-package-values:

``prefer_package_values``
   Take a corpus session's coordinates from each cell's own corpus embedding when the session's spectrogram H5 is verifiably the one the package was built from, and infer them otherwise.

   :Options:
      * ``true`` — use the package's own coordinates where verified, infer the rest
      * ``false`` — infer every session with every cell
   :Default: ``true``

.. _proc-infer-qlvm-latents-latent-dim:

``latent_dim``
   Torus latent dimensionality.

   :Options: must match every cell's training contract
   :Default: ``2``

.. _proc-infer-qlvm-latents-time-stretch:

``time_stretch``
   Whether to time-stretch spectrograms before embedding.

   :Options:
      * ``true`` — time-stretch, as the production cells were trained
      * ``false`` — plain resize
   :Default: ``true``
   :Notes: must match every cell's training contract.

.. _proc-infer-qlvm-latents-masking-type:

``masking_type``
   How each spectrogram is masked before embedding.

   :Options:
      * ``"sam"`` — masks each spectrogram by the union of its SAM regions exactly as a masked training set did (spectrogram and mask resized or time-stretched separately, the resized mask binarized at 0.5 and multiplied in, then a min-max and the binarized mask once more), matching a decoder trained on masked spectrograms (the production cells, phase 9 cells, and ``train-qlvm`` cells trained on a ``build-qlvm-training-set --apply-mask`` set)
      * ``"none"`` — embeds raw spectrograms, as the v3 phase 6, 10 and 11 cells were trained
   :Default: ``"sam"`` (as the production cells were trained)
   :Notes: it must match every cell's training contract. USVs without a mask instance get null columns rather than being embedded unmasked when the training contract has ``require_mask`` true (every v2 and v3 package cell) or the cell conditions on mean frequency or loudness (the spectral entropy is read from the summary and needs no mask); a ``"sam"`` decoder without ``require_mask`` (whose training set kept such USVs under an all-ones mask) still embeds them. A session H5 without a ``mask/<session>`` group raises for any of these decoders.

.. _proc-infer-qlvm-latents-target-shape:

``target_shape``
   Output spectrogram ``(freq, time)`` shape the embedder resizes to before inference.

   :Options: a ``(freq, time)`` shape; must match the ``target_shape`` used by *Build QLVM training set*
   :Default: ``[128, 128]``

.. _proc-infer-qlvm-latents-length-threshold:

``length_threshold``
   Embed only USVs with ``0 < duration < length_threshold`` (time bins), the window *Build QLVM training set* keeps; longer USVs get null columns.

   :Options:
      * ``null`` — takes the value from each cell's training contract
      * a number of time bins — must equal every contract's
   :Default: ``null``
   :Notes: a cell trained on a set with no duration bound (contract ``length_threshold`` null, which ``train-qlvm`` records for such a set) embeds every USV of positive duration and accepts only ``null`` here.

.. _proc-infer-qlvm-latents-lattice-batch-size:

``lattice_batch_size``
   Lattice points decoded and scored per block.

   :Options: a positive integer
   :Default: ``4096``
   :Notes: each block holds its decoded images and their two logs, about ``3 * 16384 * 4`` bytes per point at ``128x128``.

.. _proc-infer-qlvm-latents-data-batch-size:

``data_batch_size``
   Spectrograms whose lattice posteriors are computed together.

   :Options: a positive integer
   :Default: ``8192``
   :Notes: their likelihood matrix takes ``data_batch_size * <lattice points> * 4`` bytes (46,368 points for a package cell), and the lattice is decoded once per such batch.

*Several QLVM models in one run* (``model_cells``). The production setting places every session on the regular torus and the duration, spectral-entropy, bandwidth and loudness conditional tori of the masked, time-stretched cells under ``/mnt/falkner/Bartul/spectrograms/qlvm`` (``os_utils.QLVM_MODEL_PACKAGE_ROOT`` and ``os_utils.QLVM_PRODUCTION_MODEL_CELLS``), with ``masking_type`` ``"sam"`` and ``time_stretch`` ``true`` (the shipped setting stays ``{}``; with ``spectrograms_root`` set and no cells configured, ``derive_spectrogram_model_paths`` fills in exactly this mapping, ``masking_type`` ``"sam"`` and ``time_stretch`` ``true``; otherwise the paths are set per run or by a backfill).

The cells live in the lab's spectrograms directory, one folder per map named by its column prefix, but they are module constants, not settings and not derived from ``spectrograms_root``: the settings' experimenter paths are re-keyed to the active experimenter, which would move them:

.. code-block:: json

    "model_cells": {
        "qlvm": "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm",
        "qlvm_duration": "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_duration",
        "qlvm_entropy": "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_entropy",
        "qlvm_bandwidth": "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_bandwidth",
        "qlvm_loudness": "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_loudness"
    }

This writes ``qlvm1`` / ``qlvm2`` (regular), ``qlvm_duration1`` / ``qlvm_duration2`` (duration), ``qlvm_entropy1`` / ``qlvm_entropy2`` (spectral entropy), ``qlvm_bandwidth1`` / ``qlvm_bandwidth2`` (bandwidth) and ``qlvm_loudness1`` / ``qlvm_loudness2`` (loudness), after the acoustic features, which is the canonical layout (``os_utils.USV_SUMMARY_COLUMN_ORDER``) together with the regular map's ``qlvm_category`` that ``assign-qlvm-categories`` writes afterwards (the processing run's *Infer QLVM latents* runs both).

The production cells' folder holds no ``SESSION_H5_BASELINE.tsv``, so every session is inferred (the package route below applies to full model packages such as ``qlvm_models_latest/v3``). The spectral-entropy model needs ``spectral_entropy`` in the summary, the bandwidth model ``freq_bandwidth_hz``, the loudness model ``loudness_db`` (all written by *Generate USV acoustic features*), and every model the session's SAM masks. Model provenance is not written to the summary.

For each model and session the coordinates come from one of two routes, and the log names the route and the reason (e.g. ``qlvm_duration (v3/phase11_cond_duration_floor/natural_5strata_N29000_unmasked_floor): package values (sha256 + 907 rows verified).``). The *package route* copies the package's own embedding of the USV: the cell's ``posterior_cache.npz`` ``torus_weighted`` (turned into ``[0, 1)`` coordinates) joined to the summary row through ``recon_mse_breakdown.npz``'s ``spec_id`` (``<session>_<H5 row>``). Because a ``spec_id`` names a USV only by its row in the session's spectrogram H5, it is taken only when all of these hold:

* ``prefer_package_values`` is ``true`` and a ``SESSION_H5_BASELINE.tsv`` is found in the cell directory or one of its parents (the package root);
* the session is listed in that baseline (it is one of the package's corpus sessions);
* the SHA-256 of the session's ``audio/spectrograms/<session>_spectrograms.h5`` equals the baseline's (the H5 was not rebuilt since; it is hashed once per session, not once per model);
* ``usv_summary.csv`` has as many rows as the H5;
* the cell holds rows of the session, all inside the H5, and each row's ``durations`` and ``mask_counts`` in ``recon_mse_breakdown.npz`` equal the H5's (``spectrogram/<session>/durations`` and the number of ``mask/<session>/spectrogram_index`` entries naming the row).

Otherwise the *inference route* embeds the session with that cell (the log says why: ``session not in the package corpus``, ``spectrogram H5 changed since the package: sha256 mismatch``, a row-count or per-row disagreement, no baseline above the cell, or ``prefer_package_values is false``).

Both routes place the same USVs of a verified corpus session (those inside the duration window with a SAM mask and, for conditional cells, a conditioning value), and inferred coordinates differ from the package's only where the package's TF32 embedding and the float32 embedding here round a spread-out posterior differently (see the phase 11 check above), so the columns are comparable across sessions whichever route filled them.

Both routes label a USV the same way: the cell's ``label_grid.npy`` at the pixel of the coordinates written to the summary. On the package's own posterior-mean coordinates this pixel rule reproduces the package's ``cluster_labels.csv``, so a package-route label equals the package's (on the whole v3 corpus, 445,742 USVs, every fine label of the five v3 cells and every coarse label of the v3 regular cell agreed except one ``qlvm_dur`` USV, ``20230805_112702`` row 338, whose ``x * 200`` is 190.999997: a coordinate within float32 rounding of a pixel edge).

Train spectrogram-pipeline models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two per-session inference steps lean on learned models: *Generate (spectrogram) masks* uses a detector to find each USV, and *Infer QLVM latents* uses a decoder to embed it. Both are trained **once on a representative cohort** and then reused for every future session — so this is a setup / maintenance step, not part of routine per-experiment processing. Retrain only when something changes materially: a new or substantially expanded cohort, a different spectrogram representation, or revised category definitions.

Both train cross-session via CLI / cluster commands only (no GUI buttons): each is a two-command chain over a comma-separated list of session ``--root-directories`` writing standalone artifacts to ``--output-directory``. **SAM2 is used pretrained — it is not trained here.** Full flags for CLI can be found :ref:`here <usv-pipeline-cli>`.

QLVM training sets
^^^^^^^^^^^^^^^^^^

Two commands build the ``.npz`` sets ``train-qlvm`` trains on, in the format of the QLVM model packages' sets: ``build-qlvm-training-set`` for USVs and ``build-qlvm-squeak-training-set`` for squeaks. They are the in-house ports of the reference builders (MMMmB ``build_masked_usvs.py``, ``build_unmasked_usvs_floor.py`` and ``build_bbv_dataset.py``, together with the session-typing, quota, stratification and split helpers they imported from a local usv-playpen fork, which are reconstructed here), and with the same sessions, inputs and seed they draw the same rows, hold out the same sessions and write the same spectrograms. Both write ``train_data.npz`` + ``val_data.npz`` (and ``full_data.npz`` after them with ``--full-dataset``, which takes every eligible row instead of drawing) and ``metadata.npz`` (every setting, the per-type report and the train / validation session lists).

**USVs** (``build_qlvm_training_set`` settings block). For every session its ``audio/spectrograms/<session>_spectrograms.h5`` is read (durations and the SAM instance count of every row, from ``mask/<session>/spectrogram_index``), its type comes from the ``Subjects[].sex`` entries of its metadata YAML (``MF``, ``FF``, ``MM``, ``lone_male``, ``lone_female``, ``other``; ``unknown`` if unreadable), and the rows that are not USVs alone are marked: with ``exclude_squeaks`` (on by default) every row whose *usv_summary.csv* ``squeak`` is true (pure squeaks and segments holding both), so a set holds pure USVs only (``usv & ~squeak``; empty booleans -- noise, or a segment too short to score -- do not mark a squeak: noise is left out by ``exclude_noise``, the unscorable segments by the duration gate), and with ``exclude_noise`` every row flagged as noise.

``squeak_exclusion_source`` (needs ``exclude_squeaks`` unless ``summary``) says where the squeak rows come from: ``summary``, the summary ``squeak`` flag described above; ``reference``, the rows the reference squeak index (keyed by ``session_id`` and ``seg_index``, the summary row) marks by its strict rule (``is_bbv`` true, or ``n_bouts_min3`` >= 1), INSTEAD of the flag (to reproduce the shipped sets); ``union`` (the shipped setting), both -- every row the flag marks and every row the index's strict rule marks. With the shipped ``exclude_noise`` and ``exclude_squeaks``, ``union`` leaves out the union of the current noise flag, the current squeak flag and the reference strict squeak rows, the exclusions every QLVM training or evaluation set of USVs needs.

The index is read once per build from ``reference_squeak_index_path``; empty (the shipped value) reads the lab's index, the code constant ``REFERENCE_SQUEAK_INDEX_PATH`` (``/mnt/falkner/Dexter/vocal_beh/models/bbv_classifier/indices/bbv_segment_index.csv``), a constant rather than a setting because the GUI and CLI re-key every experimenter folder in the settings. With ``reference`` a session the index does not list stops the build; with ``union`` such a session (recorded after the index was written) keeps its summary flags alone, is named in the log and is listed in ``metadata.npz`` (``sessions_not_in_reference_index``); in both, a row the index lists past the end of the session stops the build.

``exclude_noise`` is on by default, so new sets leave out the rows *Produce noise labels* flags; the shipped v3 sets were drawn without it, so reproducing them needs ``--no-exclude-noise`` (and the reference squeak index's strict exclusions, see below). A row is eligible when ``0 < duration < length_threshold`` (128 time bins), it is not marked and, with ``require_mask``, it has at least one SAM mask instance.

* **Budgets.** ``session_type_targets`` maps each type to its number of rows; ``null`` takes the type whole and a type not listed is left out (the shipped sets: ``{"MF": N, "FF": N, "MM": null, "lone_male": null}``). A type's budget is split into per-session quotas by integer capped-even water-filling: every session gets ``min(capacity, L)`` at the largest level ``L`` that does not overshoot, and the remainder goes one row at a time to sessions with room left, in the order of a permutation drawn from ``random_state`` over the type's sessions sorted by id. The total is therefore met exactly, and no session dominates a type.
* **Within-session draw** (``draw_mode``). ``natural`` draws the quota uniformly over the session's eligible rows, keeping its own mask-count mix. ``uniform`` splits it equally across the session's mask-count strata (``mask_count_bin_edges`` are the inclusive lower bounds of the strata, the last open-ended: ``1,2,3,4,5`` gives 1/2/3/4/5+, ``1,2,3`` gives 1/2/3+), and its quotas are allocated against each session's uniform headroom (number of non-empty strata times the smallest), not its eligible count, so every quota can be met with exactly equal strata; the type's total uniform headroom is the largest budget that stays exactly uniform (the builder warns when a budget exceeds it). Types are drawn in sorted order and sessions in the order of ``--root-directories``, all from one generator seeded with ``random_state``.
* **Split.** Whole sessions are held out, type by type (sorted), each type's sessions (sorted by id) permuted and moved to validation until they hold ``validation_split`` of the type's rows, so no session straddles the boundary. A type with a single session (e.g. ``MM`` or ``lone_male`` in a small build) cannot be split and goes to training whole, so the model sees every recording type; the build log names each such type, and it has no validation rows (its permutation is still drawn, so every other type's split is the one the seed gives). The squeak training-set builder shares this split.
* **Spectrograms.** Each drawn spectrogram and its SAM mask union (binarized at 0.5 after the same resize) is resized to ``target_shape`` -- time-stretched to fill it with ``time_stretch`` (the production set), plainly resized without. With ``apply_mask`` the mask is multiplied in (the phase 9 sets); without, the spectrogram stays unmasked and, with a ``floor``, is min-max normalized on its own and floored, ``clip((x - floor) / (1 - floor), 0, 1)`` (the phase 6 sets, floor 0.2). ``masks``, ``masks_len`` and ``mask_count`` are stored either way, so a masked set and its unmasked twin of the same draw differ only in ``spectrograms``. ``masking_type none`` reads no masks (and so allows neither ``apply_mask``, ``require_mask`` nor a ``uniform`` draw).
* **Conditioning values.** Every split also carries ``mean_freq_hz``, ``freq_bandwidth_hz``, ``loudness_db`` and ``spectral_entropy`` (float64), copied row for row from the session's *usv_summary.csv* (the mean frequency and bandwidth of the USV's SAM-masked region, its absolute masked image-level loudness and the spectral entropy in nats of its normalized frequency power profile, all written by ``generate-usv-acoustic-features``; NaN where the summary has no value or no such column): the raw values a conditional ``train-qlvm`` run conditions on, captured with the rows they belong to.

The shipped settings are those of the production training set, ``masked_clean/trainset/masked`` (the set the production regular cell ``masked_clean/cell/masked`` was trained on; its ``metadata.npz`` and build script record them): ``session_type_targets`` ``{"MF": 29000, "FF": 29000, "MM": null, "lone_male": null}``, ``draw_mode`` ``natural``, ``mask_count_bin_edges`` ``1,2,3,4,5``, ``length_threshold`` 128, ``require_mask``, ``exclude_squeaks`` and ``exclude_noise`` on, ``squeak_exclusion_source`` ``union`` with ``reference_squeak_index_path`` empty (the lab's index), ``row_exclusion_table`` null, ``masking_type`` ``sam`` with ``apply_mask`` on and ``floor`` null, ``time_stretch`` on, ``target_shape`` 128 x 128, ``validation_split`` 0.2 and ``random_state`` 42.

The production build also left out the reference squeak index's strict rows, through the Python ``row_exclusions`` argument. They are kept as a table outside the repository, */mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean/trainset/reference_strict_squeak_rows.csv* (18,558 rows; on the current summaries 17,295 of them are already left out by the summary squeak or noise flag, so the table adds 1,263 rows caught only by the strict rule). The shipped ``union`` setting reads those strict rows from the index itself, so the shipped settings no longer need the table; it can still be passed with ``--row-exclusion-table``.

A row-exclusion table's ``session_id`` and ``row`` are stripped of surrounding blanks, and a session it names that the build does not hold (not among the session roots, or a misspelled id) stops the build with the list of such sessions, so a table never silently excludes nothing; a table made for a larger cohort has to be cut to the build's sessions.

A build first removes the data files (``train_data.npz``, ``val_data.npz``, ``full_data.npz``, ``metadata.npz``) an earlier build left in the output directory and writes every file to a temporary sibling renamed into place, so a rebuild never leaves an earlier ``full_data.npz`` next to the new split and a crash never leaves a partial file (the squeak builder below does the same).

That set was drawn on 2026-10-01 from the summaries as they were then, and every summary of its 324 sessions has been rewritten since: on the current summaries the summary flags plus the table leave out 433 more rows than that build did (MF 17,908 vs 17,551, FF 3,770 vs 3,710, MM 1,771 vs 1,757, lone_male 209 vs 207; 23,658 vs 23,225), and the summary flags alone (``squeak_exclusion_source`` ``summary``, no table) leave out 22,395, 830 fewer, so a rebuild today is an equivalent draw, not the same one.

``--no-apply-mask --floor 0.2 --no-time-stretch`` gives the phase 6 recipe (``natural_5strata_N29000_unmasked_floor``; with ``--no-exclude-noise`` and the reference squeak index's strict exclusions it rebuilds that set bit-exactly), ``--no-apply-mask --floor none`` a plain unmasked set, ``--no-time-stretch`` the phase 9 masked set recipe, ``--draw-mode uniform`` the uniform cells, ``--mask-count-bin-edges 1,2,3 --session-type-targets '{"MF": 65000, "FF": 65000, "MM": null, "lone_male": null}'`` the 3-strata cells.

Every session's spectrogram H5 is fingerprinted as it is read: ``SESSION_H5.sha256`` / ``SESSION_H5.tsv`` hold the SHA-256 and row counts of every session H5, so a later consumer can tell whether a session's spectrogram file has been rebuilt since (its ``spec_id`` row numbers would then point at different rows).

Reproduction (2026-09-30, the 324 sessions of the reference sets, seed 42): the phase 6 ``natural_5strata_N29000`` set (``draw-natural_bins-1-2-3-4-5_N29000_len128_seed42_bbvfree_floor0p2``) and the masked (phase 9 input) ``natural_3strata_N65000``, ``uniform_3strata_N65000`` and ``uniform_5strata_N29000`` sets were rebuilt bit for bit -- the same training / validation rows in the same order (49,411 / 12,629, 106,273 / 27,767, 106,860 / 27,180 and 49,287 / 12,753), the same session types, per-type reports and split sessions, and every ``spectrograms``, ``masks``, ``masks_len``, ``durations`` and ``mask_count`` value identical (maximum absolute difference 0).

The reference sets left out broadband vocalizations by a *strict* rule of the reference squeak index (segment probability >= 0.385, or at least one run of three frames above it): 2,597 rows of those sessions that the strict rule drops were not flagged by the *usv_summary*'s former binary ``squeak`` column, which was the segment rule alone.

The rebuild therefore passed that index's exclusions through the Python-only ``row_exclusions`` argument of ``QLVMTrainingSetBuilder``; with the summary's own ``squeak`` column, the eligible pool grows by 2,388 rows (MF +1,866, FF +474, MM +27, lone_male +21), and since every quota and every draw position shifts with it, only 24,869 of the phase 6 set's 62,040 rows are drawn again (MF and FF still hold 29,000 rows each; the take-all types grow with their pools). A set built from the summary is therefore an equivalent draw, not the same one. (These counts compare the strict rule with the summary's old binary ``squeak`` column, which *Produce squeak labels* no longer writes; a build from the new booleans differs from the reference draw too.)

``--squeak-exclusion-source reference --reference-squeak-index-path <index>`` applies the index's strict rule from the index itself, the same exclusions the rebuild passed through ``row_exclusions``, so with ``--no-exclude-noise`` it reproduces the shipped sets from the command line.

The ``build_qlvm_training_set`` settings block (every key is also a ``build-qlvm-training-set`` flag; ``full_dataset`` takes every eligible row instead of drawing and also writes ``full_data.npz``):

.. code-block:: json

    "build_qlvm_training_set": {
        "session_type_targets": {
          "MF": 29000,
          "FF": 29000,
          "MM": null,
          "lone_male": null
        },
        "draw_mode": "natural",
        "mask_count_bin_edges": [1, 2, 3, 4, 5],
        "length_threshold": 128.0,
        "require_mask": true,
        "exclude_squeaks": true,
        "squeak_exclusion_source": "union",
        "reference_squeak_index_path": "",
        "exclude_noise": true,
        "row_exclusion_table": null,
        "masking_type": "sam",
        "apply_mask": true,
        "floor": null,
        "validation_split": 0.2,
        "random_state": 42,
        "full_dataset": false,
        "target_shape": [128, 128],
        "time_stretch": true
      }

**Squeaks** (``build_qlvm_squeak_training_set`` settings block). The candidates are the rows *Produce squeak labels* marked ``squeak`` true (pure squeaks and segments holding both) that carry a squeak extent (with ``exclude_noise`` also not noise; noise rows carry empty booleans, so this is a guard), sessions in sorted order; a row with several squeaks is one candidate over their envelope (its one extent), as *Embed squeaks in a QLVM torus* embeds it. ``exclude_noise`` is on by default; the shipped squeak sets were drawn without it (``--no-exclude-noise`` to reproduce them).

Each candidate's audio window is the segment widened to hold the envelope and its context (the embedding's window), its first and last frames are the window frames whose centres lie inside ``squeak_start`` / ``squeak_end``; the extent is widened by ``context_frames`` (2) either side, clipped to the window, and crops narrower than ``min_trimmed_frames`` (8) or wider than the 128-frame frame are dropped. The written ``squeak_probability`` is ``p_squeak + p_both``, the probability that the segment holds a squeak. ``crop_window`` ``full_length`` (the shipped setting, and the rule *Embed squeaks in a QLVM torus* follows) crops from the whole segment; ``first_128_frames`` cuts every segment to its first 128 frames, as the reference 128-frame spectrogram store did, and drops the squeaks still going at frame 127 (right-censored).

The crop width sets the duration stratum (``duration_bin_edges`` 26 / 40 / 62 frames, four strata). With one generator seeded with ``random_state``, ``per_session_bin_cap`` (48, the ``session`` cells; 0 for the ``lumped`` cells) first keeps at most that many squeaks of any one session in any one stratum, and ``n_total`` (11,000) squeaks are then drawn, ``natural`` (uniformly) or ``uniform`` (equal across strata).

The drawn squeaks' sonic spectrograms are rebuilt from the session audio with the *Produce squeak labels* front end (``exclude_metadata_audio_channels`` as there), each crop is min-max normalized (``(x - min) / (max - min + 1e-6)``, ``crop_normalization`` ``per_crop``; ``absolute`` is the classifier's fixed dB transform) and centred in a 128 x 128 frame; ``masks`` / ``masks_len`` are zero and ``apply_mask`` false. The split is the USV builder's, and so are the removal of an earlier build's data files and the atomic writes. When no candidate passes the gates (and the cap) the build stops before writing anything, with the count each gate left.

The shipped settings are those of the production squeak cell (``natural_session_N11000``) except the two rules squeak embedding follows: full-length crops and the metadata channel exclusion.

Reproduction (2026-09-30, the 324 sessions of the reference sets, 239 of which hold squeaks, seed 42): fed the candidates of the reference squeak index (segment probability >= 0.385 and the extents of its 128-frame pass) through ``QLVMSqueakTrainingSetBuilder.build_from_candidates`` with ``crop_window`` ``first_128_frames`` and ``exclude_metadata_audio_channels`` false, the builder passes the same 15,961 / 14,963 / 14,936 candidates through its gates as the reference builder and draws, crops and splits exactly the reference rows in all five phase 3 sets (natural / uniform x session / lumped, and the full set): the same ``spec_id`` order, ``crop_first`` / ``crop_last``, duration strata and split sessions.

For ``bbv-natural_dur-26-40-62_session_N11000_seed42`` (the production cell's set) the spectrograms were rebuilt from the audio too, and all 8,767 / 2,233 training / validation spectrograms are bit-identical to the reference ones (maximum absolute difference 0).

The summaries no longer carry the retired classifier's flags and extents (the vocal-class model replaced them), so a set drawn from the summaries is a new draw over different squeaks; reproducing the shipped sets needs the reference index's candidates fed through ``build_from_candidates`` (with each segment as its own audio window, ``start_s`` / ``stop_s`` = ``start`` / ``stop``).

The ``build_qlvm_squeak_training_set`` settings block (every key is also a ``build-qlvm-squeak-training-set`` flag; ``full_dataset`` takes every squeak that passes the gates, with no cap and no draw, and also writes ``full_data.npz``):

.. code-block:: json

    "build_qlvm_squeak_training_set": {
        "draw_mode": "natural",
        "n_total": 11000,
        "per_session_bin_cap": 48,
        "duration_bin_edges": [26, 40, 62],
        "context_frames": 2,
        "min_trimmed_frames": 8,
        "crop_window": "full_length",
        "crop_normalization": "per_crop",
        "exclude_noise": true,
        "exclude_metadata_audio_channels": true,
        "validation_split": 0.2,
        "random_state": 42,
        "full_dataset": false,
        "target_shape": [128, 128]
      }

QLVM decoder
^^^^^^^^^^^^

Defines the shared toroidal latent space (on whose regular map ``build-qlvm-categories`` defines the content-ridge categories) that makes the ``qlvm_*`` columns comparable across every session embedded with the same model. ``build-qlvm-training-set`` (USVs) or ``build-qlvm-squeak-training-set`` (squeaks) builds the training set (see *QLVM training sets* above) → ``train_data.npz`` + ``val_data.npz`` (+ ``full_data.npz``) + ``metadata.npz``. ``train-qlvm`` then trains a decoder on such a set, or on a set in the same format built outside the repository (the QLVM model packages' training sets: ``train_data.npz`` / ``val_data.npz`` with ``spectrograms``, ``masks``, ``masks_len``, ``durations`` and an optional ``apply_mask`` scalar, plus ``metadata.npz``).

It is the in-house JAX port of the recipe the shipped models were trained with (qmc_deep_gen's ``bartul_mouse.py`` under the v3 "shipped" protocol), using the same decoder and likelihood code ``infer-qlvm-latents`` embeds with: each spectrogram is min-max normalized on its own (``(x - min) / (max - min + 1e-8)``) and, when the set applies masks (its ``apply_mask``, else ``masking_type`` ``sam``), multiplied by its binarized SAM mask; a loudness floor (phase 6) is already baked into the stored spectrograms and is only recorded.

The decoder (``relu`` or ``legacy`` head, torch's default initialization) is trained over a 610-point Fibonacci lattice (``training_fib_m`` 15) with one uniform random torus shift per batch, minimizing the negative QMC log evidence of the batch, with Adam at a constant learning rate of 1e-3, batches of 512, 300 epochs and seed 42; the last epoch's weights are kept. Every ``val_freq`` (10) epochs and after the last one the same loss is computed on a fixed validation subset -- at most ``val_samples_per_mask_count`` (100) USVs per ``masks_len`` value, drawn exactly as the reference trainer draws it -- one USV at a time against its own shift of a 6,765-point lattice (``validation_fib_m`` 20).

The output directory is a v3 model package cell: ``checkpoint.tar`` (torch zip, the structure of a shipped cell's: ``decoder.<index>`` weights, a torch ``Adam`` state and every batch's loss), ``config/training_contract.json`` (``input_normalization`` ``minmax``, ``masking_type`` ``sam`` or ``none`` from the set's mask use, ``floor``, ``target_shape``, ``time_stretch``, ``length_threshold`` (null when the set records none), ``require_mask``, ``embedding_fib_m`` (24), the lattices, ``"condition": null`` for an unconditional decoder), ``config/run_config.json`` and ``metrics/val_diagnostics.npz`` (per-epoch training loss and seconds, the validation losses and subset).

The recipes of the shipped cells: phases 6 (unmasked, floor 0.2) and 9 (SAM-masked) are the defaults; phase 3 (BBVs) is ``--decoder-head legacy --n-epochs 2500 --val-freq 80 --val-samples-per-mask-count 1600``. A short run (30 epochs) on the phase 6 ``natural_5strata_N29000`` set reproduced the shipped run's validation subset row for row, its per-epoch training loss to within 0.52 % and its validation loss to within 0.66 % from epoch 3 on, at 3.3 s per epoch (RTX 4080 SUPER); 240 epochs on the phase 3 ``natural_session_N11000`` set matched the shipped validation loss to within 0.07 % at epochs 80, 160 and 240. On a shared GPU, set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so JAX does not reserve most of its memory.

The run refuses a set without ``metadata.npz``, and any split whose row count differs from the count ``metadata.npz`` records for it (``n_train`` / ``n_val`` / ``n_full``): a split left over from an earlier build. (File times cannot tell: one build writes ``train_data.npz`` and ``val_data.npz`` before ``full_data.npz``.) A set built with ``full_dataset`` (``metadata.npz`` says so) is trained on ``full_data.npz``, every eligible row, the point of such a build; its validation rows are then training rows too, so the validation loss only tracks convergence, and ``config/run_config.json`` records ``validation_in_sample: true``. Any other set is trained on ``train_data.npz`` (``full_data.npz`` when that is the only split).

Cluster submitter: ``train_qlvm_global.sh``. ``infer-qlvm-latents`` embeds with such a cell as with any other (it writes coordinates only), and ``build-qlvm-categories`` / ``assign-qlvm-categories`` (see the CLI reference) label the USVs with the torus's content-ridge categories.

**Conditional decoders (phase 11).** With ``conditional`` set to ``duration``, ``mean_freq``, ``bandwidth``, ``loudness`` or ``spectral_entropy`` (``--conditional``; null / ``none`` trains an unconditional decoder), ``train-qlvm`` trains a decoder that takes one conditioning value ``c`` per USV, appended to the torus basis of every lattice point (``c_dim`` 1), with the recipe of the v3 package's phase 11 cells (qmc_deep_gen ``bartul_mouse_cond.py``, width-capped bins and a scaled loss).

``c`` per USV: ``duration`` is ``(d - d_min) / (d_max - d_min + 1e-8)`` of the pre-resize duration in time bins, ``d_min`` / ``d_max`` the training split's (8 and 127 in the shipped cells); ``mean_freq`` is ``(f - 30000) * 127 / (128 * 90000)`` of the USV's mean frequency over its SAM-masked region (the summary's ``mean_freq_hz``; equal to the energy-weighted row centroid of the resized SAM-masked spectrogram that ``infer-qlvm-latents`` computes, to 1.5e-7); ``bandwidth`` is ``clip(freq_bandwidth_hz / 90000, 0, 1)``; ``loudness`` is ``clip((dB - 28.83) / (95.85 - 28.83), 0, 1)`` of the USV's absolute masked image-level loudness (the summary's ``loudness_db``, measured by ``compute_usv_loudness``); ``spectral_entropy`` is ``clip((H - H_min) / (H_max - H_min), 0, 1)`` of the USV's spectral entropy in nats (the summary's ``spectral_entropy``), with ``H_min`` / ``H_max`` the smallest and largest entropy of the training split, so the training values span exactly ``[0, 1]`` and validation rows (and, at inference, new USVs) outside that range are clamped; the two are recorded in the contract as ``entropy_min`` / ``entropy_max``, and a training split of constant entropy stops the run.

The raw values come from the split's own ``mean_freq_hz`` / ``freq_bandwidth_hz`` / ``loudness_db`` / ``spectral_entropy`` columns (``build-qlvm-training-set`` copies them), or, with ``condition_table`` (``--condition-table``), from a per-USV ``.npz`` with ``spec_id`` and the column (the package's ``corpus/cond_table.npz``, whose loudness column ``image_level_db`` is accepted too) for sets built outside the repository; a row without a finite value stops the run.

The training values are cut into ``condition_n_bins`` (32) quantile bins, and with ``condition_bin_scheme`` ``quantile_capped`` (the default) every bin wider than the widest inner bin is split into equal-width pieces no wider than it (in practice the two end bins), kept however few rows they hold. Every batch is drawn from one bin and decoded at its rows' mean ``c`` (bins shuffled internally and batches shuffled together each epoch, the reference sampler's draws as the shipped multi-worker runs made them); with ``condition_scale_loss_by_batch`` (default on) each step's loss is multiplied by ``rows / batch_size``, so a small tail batch steps with its rows' share (short batches are padded with zero-weight rows, which changes neither the loss nor its gradient). Validation decodes each USV at its own ``c``.

The cell's ``training_contract.json`` then carries ``c_dim`` 1, ``conditional`` and a ``condition`` block in the form of the shipped phase 11 contracts (the constants ``infer-qlvm-latents`` computes ``c`` with, the ``decode`` rule: ``exact`` -- the USV's own ``c`` clamped to the training range -- for duration and bandwidth, ``grid`` -- the nearest point of the decode grid -- for mean frequency, loudness and spectral entropy), and ``config/condition_bins.npz`` holds the bin ``edges``, ``group_ids`` / ``group_sizes`` / ``group_means`` (one entry per bin between consecutive edges: ``group_ids`` is ``0 .. n - 1``, and a bin with no training row -- an equal-width piece of a sparse tail the capped scheme keeps -- has size 0 and a NaN mean; cells written before this listed their occupied bins only, which the readers place by ``group_ids``), ``train_c_min`` / ``train_c_max`` and the ``decode_grid`` (``condition_decode_grid_step`` apart, from ``train_c_min`` to at or just past ``train_c_max``; 0.0025 by default, the step of the shipped grid-decoded cells; the shipped duration and bandwidth cells, which never read it, carry 0.01). ``val_diagnostics.npz`` adds each validation USV's ``c`` (``val_diag_c``).

``condition_bins.npz`` is written only after training has succeeded, with the checkpoint, the diagnostics and the contract, so a run that fails (divergence, an interrupt) into an existing cell directory leaves that cell's earlier bins in place instead of pairing new bins with the old checkpoint.

Validation (2026-09-30, 30 epochs, seed 42, the shipped phase 11 recipe on the ``natural_5strata_N29000`` floor set): the duration and loudness runs wrote ``condition_bins.npz`` bit-identical to the shipped cells' (edges, groups, means, training range and decode grid, with the step each shipped cell carries; the shipped files add only ``run_decode_grid_step``), reproduced the shipped validation subset row for row and the shipped batch order (from epoch 3 on, each epoch's per-batch losses correlate with the shipped run's same epoch at r >= 0.999), and from epoch 3 on matched the shipped per-epoch training loss to within 0.88 % (duration) and 0.87 % (loudness), and the validation loss at epochs 10, 20 and 30 to within 0.84 % and 0.20 % -- less than the port's own run-to-run spread (the duration run with every JAX key changed moved by up to 0.87 % in training and 1.14 % in validation loss). About 4.1-4.6 s per epoch (RTX 4080 SUPER).

The ``train_qlvm`` settings block (every key is also a ``train-qlvm`` flag): ``n_epochs`` training epochs, ``decoder_head`` (``relu`` or ``legacy``), the Fibonacci indices of the training, validation and embedding lattices (``training_fib_m``, ``validation_fib_m``, ``embedding_fib_m``), ``batch_size``, the constant Adam ``learning_rate``, the validation cadence ``val_freq`` and subset size ``val_samples_per_mask_count``, the ``seed`` of the initialization, shuffling, lattice shifts and validation subset, and the conditional-decoder keys described above (``conditional``, ``condition_n_bins``, ``condition_bin_scheme``, ``condition_scale_loss_by_batch``, ``condition_decode_grid_step``, ``condition_table``):

.. code-block:: json

    "train_qlvm": {
        "n_epochs": 300,
        "decoder_head": "relu",
        "training_fib_m": 15,
        "validation_fib_m": 20,
        "embedding_fib_m": 24,
        "batch_size": 512,
        "learning_rate": 0.001,
        "val_freq": 10,
        "val_samples_per_mask_count": 100,
        "seed": 42,
        "conditional": null,
        "condition_n_bins": 32,
        "condition_bin_scheme": "quantile_capped",
        "condition_scale_loss_by_batch": true,
        "condition_decode_grid_step": 0.0025,
        "condition_table": null
      }

QLVM model packages
^^^^^^^^^^^^^^^^^^^

A QLVM model package (the ``qlvm_models_latest/v2`` and ``v3`` layout; v3 is v2.1 plus the phase 11 cells) ships trained decoders together with everything inference needs, one folder per cell: ``<package>/<phase>/<cell>``. ``infer-qlvm-latents`` reads every cell of ``model_cells``. Two layouts are read. v3, restructured on 2026-09-28, keeps each cell's contract and bins in ``config/``, its per-USV tables in ``inference/`` and each cluster level in ``inference/clusters_<level>/``, with ``checkpoint.tar`` at the top, and keeps the package's ``SESSION_H5_BASELINE.tsv`` in ``corpus/``; v2 / v2.1 keep every file at the cell's (or package's) top level and the clusters in ``cluster/<level>/``. The v3 location is tried first. The files it uses are:

* ``checkpoint.tar`` — a torch zip checkpoint whose ``"model"`` entry holds the decoder ``state_dict`` (``decoder.<index>.weight`` / ``.bias``); read without torch. The head follows from the keys: ``decoder.1.weight`` is the activation-free legacy head, ``decoder.2.weight`` the ReLU head
* ``training_contract.json`` — ``decoder_head``, ``latent_dim``, ``c_dim``, ``input_normalization`` (``"minmax"``) and ``normalization_epsilon``, ``masking_type``, ``floor``, ``target_shape``, ``time_stretch``, ``length_threshold``, ``require_mask`` (``true``: the corpus left out USVs without a SAM mask, in every phase), ``embedding_lattice_type`` (``"fibonacci"``) and ``embedding_fib_m``, and the ``condition`` block of conditional cells (``"duration"``: ``duration_min``, ``duration_max``, ``epsilon``; ``"mean_freq"``: ``spectrogram``, ``epsilon``; ``"bandwidth"``: ``source``; ``"loudness"``: ``source``, ``db_range``; ``"spectral_entropy"`` (``train-qlvm`` cells): ``source``, ``entropy_min``, ``entropy_max``; in phase 11 also ``decode``, ``"exact"`` or ``"grid"``) with ``condition_bins``
* ``label_grid.npy`` of the fine and coarse cluster folders (``inference/clusters_fine`` / ``clusters_coarse``; ``cluster/fine`` / ``cluster/coarse`` in v2 / v2.1) — ``(200, 200)`` int16 label grids indexed ``[y, x]``, labels ``1 … k`` numbered by cluster size; the label of a USV is ``grid[floor(y * 200) mod 200, floor(x * 200) mod 200]`` for its coordinates ``(x, y)`` (e.g. ``qlvm1`` / ``qlvm2``), the pixel rule the package labels its corpus with, applied to the reference arrays too
* ``condition_bins.npz`` (conditional cells) — phase 10: ``edges`` (the corpus quantile-bin edges of the conditioning value) and ``bin_mean`` (the corpus mean value per bin); phase 11: ``train_c_min`` / ``train_c_max`` (the training range), ``decode_grid`` / ``decode_grid_step`` (the grid its corpus was decoded on) and the capped training bins (``edges``, ``group_ids``, ``group_sizes``, ``group_means``; each size is read into the bin its ``group_ids`` entry names, so a table that lists only its occupied bins reads like one listing every bin, and a short ``group_sizes`` with no ``group_ids`` is refused), with no ``bin_mean``. A cell whose bins do not match its contract's ``decode`` is refused
* ``posterior_cache.npz`` and ``recon_mse_breakdown.npz`` (``inference/``) — the package's per-USV coordinates and the durations and mask counts they were computed from, read by the package route of ``model_cells``
* ``clusters.csv`` and ``cluster_labels.csv`` of each cluster folder — read only by ``consolidate-spectrogram-store`` (its v3 mode; the reference-arrays exporter that also read them is retired: category geometry comes from the category bundle)

The package's own ``code/selftest.py`` checks every file above against its sources and re-embeds corpus USVs; run it on a copy before using it. Runtime is set by the 46,368-point lattice: decoding it takes about 20 s on CPU (JAX, 144 cores; decoding 4,096 points takes 1.5–1.8 s, and ``jax.jit`` barely changes that), so an unconditional cell embeds a session of ~700 USVs in about 25 s, and a phase 10 conditional cell, which decodes the lattice once per distinct bin mean, in about 11–12 minutes (a phase 11 cell decodes it roughly 3–8 times as often). The ``gpu`` extra (JAX with CUDA) shortens both.

Embedding a session with a cell reproduces the package's corpus labels: on two sessions per cell (phase 6, 9 and 10 ``natural_3strata_N65000``), the model inputs matched the package's bitwise, and 99.0–100 % of fine labels matched ``cluster_labels.csv``, all of the rest being USVs whose posterior mean crossed a pixel edge (conditional cells decode at the frozen bin mean rather than the corpus embedding's batch mean, which moves a few more USVs).

Phase 11 cells decode at the same value as the package's corpus embedding: on session ``20251003_143416`` (907 USVs, the four ``natural_3strata_N65000`` cells, one L40S with the ``gpu`` extra), the embedded USVs were exactly the package's, their conditioning values matched ``condition_decode.npz`` to float32 rounding, 99.7–99.9 % of fine and coarse labels matched, and a session took about 100–115 s (190 s for loudness, which also measures every USV's loudness from the audio). The remaining differences are USVs with a spread-out posterior, where the float32 embedding here and the package's TF32 embedding round differently.

QLVM categories
^^^^^^^^^^^^^^^

The content-ridge categories of the regular map, built once from a corpus of embedded USVs and then assigned to every session's summary.

``build-qlvm-categories`` (``build_qlvm_categories`` settings block) writes a category directory from a per-USV positions file (``spec_id``, ``x``, ``y``) and a per-USV properties file (``spec_id``, ``session_id`` and the property columns, e.g. a ``build-qlvm-training-set`` split): each property becomes its rank over all USVs, each rank's Gaussian kernel average on a periodic grid is a content field, the root sum of squares of the fields' periodic central differences is the content change, a marker watershed of it gives basins bounded by its ridges (a marker crossing a torus seam is one marker, so it gives one basin), neighbouring basins are joined (under-sized regions first, then the pair across the lowest ridge) down to ``n_categories``, and session resamples rebuild the categories to vote a consensus and a per-pixel agreement.

``assign-qlvm-categories`` (``assign_qlvm_categories`` settings block; the last part of the processing run's *Infer QLVM latents*) labels a session's USVs from that directory by the pixel under their coordinates, writing ``<prefix>_category`` only. The production bundle is ``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY`` (``/mnt/falkner/Bartul/spectrograms/categories``), the one partition every QLVM figure draws; see the CLI reference for the full description of both commands and *Label, render and embed USVs* above for the column.

.. code-block:: json

    "build_qlvm_categories": {
        "properties": [
          "mask_count",
          "durations",
          "freq_bandwidth_hz",
          "spectral_entropy",
          "mean_freq_hz",
          "loudness_db"
        ],
        "grid_resolution": 200,
        "field_sigma": 8.0,
        "change_span": 8,
        "marker_sigma": 1.5,
        "marker_distance": 5,
        "size_floor": 0.05,
        "n_categories": 4,
        "n_bootstraps": 200,
        "bootstrap_seed": 7,
        "uncertain_agreement": 0.6,
        "category_order": [1, 4, 2, 3],
        "category_descriptions": ["simple", "biphones", "intermediate", "complex"],
        "n_jobs": 20
      },
    "assign_qlvm_categories": {
        "category_directory": "",
        "coordinate_prefix": ""
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`properties <proc-build-qlvm-categories-properties>`
     - six columns, see JSON above
     - list of property columns
   * - :ref:`grid_resolution <proc-build-qlvm-categories-grid-resolution>`
     - ``200``
     - pixels per side
   * - :ref:`field_sigma <proc-build-qlvm-categories-field-sigma>`
     - ``8.0``
     - pixels
   * - :ref:`change_span <proc-build-qlvm-categories-change-span>`
     - ``8``
     - pixels
   * - :ref:`marker_sigma <proc-build-qlvm-categories-marker-sigma>`
     - ``1.5``
     - pixels
   * - :ref:`marker_distance <proc-build-qlvm-categories-marker-distance>`
     - ``5``
     - pixels
   * - :ref:`size_floor <proc-build-qlvm-categories-size-floor>`
     - ``0.05``
     - share of the USVs
   * - :ref:`n_categories <proc-build-qlvm-categories-n-categories>`
     - ``4``
     - integer > 0
   * - :ref:`n_bootstraps <proc-build-qlvm-categories-n-bootstraps>`
     - ``200``
     - integer > 0
   * - :ref:`bootstrap_seed <proc-build-qlvm-categories-bootstrap-seed>`
     - ``7``
     - integer
   * - :ref:`uncertain_agreement <proc-build-qlvm-categories-uncertain-agreement>`
     - ``0.6``
     - pixel agreement threshold
   * - :ref:`category_order <proc-build-qlvm-categories-category-order>`
     - ``[1, 4, 2, 3]``
     - size ranks, ``R-1`` first, or empty
   * - :ref:`category_descriptions <proc-build-qlvm-categories-category-descriptions>`
     - see JSON above
     - one string per category, ``R-1`` first
   * - :ref:`n_jobs <proc-build-qlvm-categories-n-jobs>`
     - ``20``
     - integer > 0

.. _proc-build-qlvm-categories-properties:

``properties``
   The property columns the categories are built on.

   :Options: a list of per-USV property columns of the properties file
   :Default: ``mask_count``, ``durations``, ``freq_bandwidth_hz``, ``spectral_entropy``, ``mean_freq_hz``, ``loudness_db``

.. _proc-build-qlvm-categories-grid-resolution:

``grid_resolution``
   Pixels per side of the periodic grid.

   :Options: a positive integer number of pixels
   :Default: ``200``

.. _proc-build-qlvm-categories-field-sigma:

``field_sigma``
   Gaussian smoothing of the property fields (pixels).

   :Options: a positive number of pixels
   :Default: ``8.0``

.. _proc-build-qlvm-categories-change-span:

``change_span``
   Offset of the central differences of the content-change field (pixels).

   :Options: a positive integer number of pixels
   :Default: ``8``

.. _proc-build-qlvm-categories-marker-sigma:

``marker_sigma``
   Blur of the content-change field before the watershed (pixels).

   :Options: a positive number of pixels
   :Default: ``1.5``

.. _proc-build-qlvm-categories-marker-distance:

``marker_distance``
   Half-width of the window a watershed marker is the minimum of (pixels).

   :Options: a positive integer number of pixels
   :Default: ``5``

.. _proc-build-qlvm-categories-size-floor:

``size_floor``
   Smallest share of the USVs a category may hold.

   :Options: a fraction of the USVs
   :Default: ``0.05``

.. _proc-build-qlvm-categories-n-categories:

``n_categories``
   Number of categories.

   :Options: a positive integer
   :Default: ``4``

.. _proc-build-qlvm-categories-n-bootstraps:

``n_bootstraps``
   Session resamples of the consensus vote.

   :Options: a positive integer
   :Default: ``200``

.. _proc-build-qlvm-categories-bootstrap-seed:

``bootstrap_seed``
   Seed of the first session resample (resample ``i`` uses ``seed + i``).

   :Options: an integer
   :Default: ``7``

.. _proc-build-qlvm-categories-uncertain-agreement:

``uncertain_agreement``
   A USV whose pixel agreement is below this is flagged uncertain.

   :Options: a pixel-agreement threshold
   :Default: ``0.6``
   :Notes: the flag is written in ``category_usv_labels.csv``; the summaries carry no agreement.

.. _proc-build-qlvm-categories-category-order:

``category_order``
   The size rank (1 = most USVs) of each category, ``R-1`` first.

   :Options: a list of size ranks, one per category; empty keeps the size order
   :Default: ``[1, 4, 2, 3]``
   :Notes: the shipped order numbers the production bundle 1 simple, 2 biphones, 3 intermediate, 4 complex.

.. _proc-build-qlvm-categories-category-descriptions:

``category_descriptions``
   A short description per category, ``R-1`` first.

   :Options: a list of strings, one per category
   :Default: ``["simple", "biphones", "intermediate", "complex"]``

.. _proc-build-qlvm-categories-n-jobs:

``n_jobs``
   Parallel workers of the session resamples.

   :Options: a positive integer
   :Default: ``20``

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`category_directory <proc-assign-qlvm-categories-category-directory>`
     - ``""``
     - a ``build-qlvm-categories`` directory, or empty (production bundle)
   * - :ref:`coordinate_prefix <proc-assign-qlvm-categories-coordinate-prefix>`
     - ``""``
     - a column prefix, or empty (``qlvm``)

.. _proc-assign-qlvm-categories-category-directory:

``category_directory``
   A ``build-qlvm-categories`` directory.

   :Options: a path to a ``build-qlvm-categories`` directory, or empty
   :Default: ``""`` — left empty it is filled with the production bundle

.. _proc-assign-qlvm-categories-coordinate-prefix:

``coordinate_prefix``
   The prefix ``P`` of the summary columns ``P1`` / ``P2`` the categories were built on.

   :Options: a column prefix, or empty
   :Default: ``""`` — left empty it is filled with ``qlvm``, the regular map

Mask detector
^^^^^^^^^^^^^

The YOLO box detector that localizes each USV in its spectrogram so SAM2 can segment it; ``generate-usv-masks`` reloads its weights. ``export-yolo-dataset`` renders the cohort's spectrograms to an Ultralytics dataset (``images/`` + ``labels/`` + ``data.yaml``); ``train-masks`` fine-tunes YOLO → the run directory + ``best.pt``. Cluster submitter: ``train_masks_global.sh``.

Box labels are set by ``--label-source`` (or ``export_yolo_dataset.label_source``): ``cc`` (default — pseudo-labels from the connected-component detector; zero manual work, no GPU; the recommended start), ``manual`` (hand-verified ``{spec_id}.txt`` YOLO files in ``--manual-labels-directory``), or ``merge`` (``cc`` pseudo-labels overridden by manual files where present). ``manual`` / ``merge`` require ``--manual-labels-directory``; ``cc`` ignores it. The submitter exposes a ``LABEL_SOURCE`` knob and ``MANUAL_LABELS_DIRECTORY``. Both ``generate-usv-masks`` and ``train-masks`` need the ``sam2`` and ``ultralytics`` packages (usv-playpen core dependencies).

The two settings blocks (every key is also a flag of its command):

.. code-block:: json

    "export_yolo_dataset": {
        "label_source": "cc",
        "validation_split": 0.2,
        "random_state": 42,
        "colormap": "viridis",
        "manual_labels_directory": ""
      },
    "train_masks": {
        "base_weights": "yolo11n.pt",
        "n_epochs": 100,
        "imgsz": 128,
        "batch_size": 16,
        "device": null,
        "run_name": "usv_yolo_detector"
      }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`label_source <proc-export-yolo-dataset-label-source>`
     - ``"cc"``
     - ``cc``, ``manual``, ``merge``
   * - :ref:`validation_split <proc-export-yolo-dataset-validation-split>`
     - ``0.2``
     - fraction of images
   * - :ref:`random_state <proc-export-yolo-dataset-random-state>`
     - ``42``
     - integer seed
   * - :ref:`colormap <proc-export-yolo-dataset-colormap>`
     - ``"viridis"``
     - a Matplotlib colormap name
   * - :ref:`manual_labels_directory <proc-export-yolo-dataset-manual-labels-directory>`
     - ``""``
     - path to a directory

.. _proc-export-yolo-dataset-label-source:

``label_source``
   Box label source.

   :Options:
      * ``cc`` — pseudo-labels
      * ``manual`` — hand-verified files
      * ``merge`` — ``cc`` pseudo-labels overridden by manual files where present
   :Default: ``"cc"``
   :Notes: also ``--label-source``; see above.

.. _proc-export-yolo-dataset-validation-split:

``validation_split``
   Fraction of images held out for validation.

   :Options: a fraction
   :Default: ``0.2``

.. _proc-export-yolo-dataset-random-state:

``random_state``
   Seed of the reproducible train / validation split permutation.

   :Options: an integer
   :Default: ``42``

.. _proc-export-yolo-dataset-colormap:

``colormap``
   Matplotlib colormap the spectrogram images are rendered with.

   :Options: a Matplotlib colormap name; must match the detector's ``generate_masks.mask_cmap``
   :Default: ``"viridis"``

.. _proc-export-yolo-dataset-manual-labels-directory:

``manual_labels_directory``
   Directory of hand-verified ``{spec_id}.txt`` YOLO labels.

   :Options: a path to a directory
   :Default: ``""``
   :Notes: used by ``manual`` / ``merge``.

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`base_weights <proc-train-masks-base-weights>`
     - ``"yolo11n.pt"``
     - a YOLO checkpoint
   * - :ref:`n_epochs <proc-train-masks-n-epochs>`
     - ``100``
     - integer > 0
   * - :ref:`imgsz <proc-train-masks-imgsz>`
     - ``128``
     - square image size (px)
   * - :ref:`batch_size <proc-train-masks-batch-size>`
     - ``16``
     - integer > 0
   * - :ref:`device <proc-train-masks-device>`
     - ``null``
     - a GPU index, ``"cpu"``, ``null``
   * - :ref:`run_name <proc-train-masks-run-name>`
     - ``"usv_yolo_detector"``
     - any string

.. _proc-train-masks-base-weights:

``base_weights``
   Base YOLO checkpoint to fine-tune from.

   :Options: a YOLO checkpoint
   :Default: ``"yolo11n.pt"``

.. _proc-train-masks-n-epochs:

``n_epochs``
   Training epochs.

   :Options: a positive integer
   :Default: ``100``

.. _proc-train-masks-imgsz:

``imgsz``
   Square image size (px) the detector trains at.

   :Options: a positive integer number of pixels
   :Default: ``128``
   :Notes: 128 is the native spectrogram size.

.. _proc-train-masks-batch-size:

``batch_size``
   Training batch size (images per batch).

   :Options: a positive integer
   :Default: ``16``

.. _proc-train-masks-device:

``device``
   Compute device.

   :Options:
      * a GPU index (e.g. ``"0"``)
      * ``"cpu"``
      * ``null`` — Ultralytics' auto-select
   :Default: ``null``

.. _proc-train-masks-run-name:

``run_name``
   Ultralytics run name (the subdirectory under the output directory holding the run artifacts).

   :Options: any string
   :Default: ``"usv_yolo_detector"``

Noise model
^^^^^^^^^^^

The ensemble *Produce noise labels* scores with. ``train-noise-model`` trains it on a labels CSV and writes a bundle ``detect-usv-noise`` loads unchanged (set it as ``noise_model_path``). Drawing the segments to label, the labelling itself, the cross-fitted calibration and the choice of the decision cut-offs are not part of it: they are done once per labelling round and come in as the labels CSV and the ``calibration_path`` JSON.

* **Labels** -- one row per training example: ``sample_id``, ``session_dir``, ``start`` and ``stop`` (s, as in the session's *usv_summary.csv*), ``chs_count`` (the summary's channel count) and ``noise`` (``1``: the segment holds no vocalization at all, neither a USV nor a squeak; ``0``: it holds one). Unsure answers must be dropped or resolved first. The segment is defined by these columns, not by a summary row, so re-curating a session later cannot change what a label points at. A segment may appear twice (it is then trained on twice, as the production ensemble was); repeats and conflicting repeats are reported.
* **Inputs** -- built by the same function *Produce noise labels* scores with (same broadband memmap channels with their stored low-band variance weights, channel exclusion, ~100 ms context window, crop, dB transform), so a model is trained on exactly the input it will be run on. The scalars (log channel count, log duration) are standardized over the training set, and their mean and standard deviation go into the bundle.
* **Recipe** -- the production one: per seed, the network is initialized from ``torch.manual_seed(seed)`` and trained for 40 epochs in batches of 32 (a seeded permutation each epoch), with Adam (learning rate 0.001, weight decay 0.0001) under a cosine schedule stepped per batch, binary cross-entropy on targets smoothed by 0.05, and augmentation of the spectrogram channels (gain jitter of up to ±5 dB, a frequency roll of up to ±2 rows, and with probability 0.5 each a time mask of up to 15% of the frames and a frequency mask of 1-10 rows). cuDNN runs in deterministic mode, but GPU training is still not bit-reproducible, so a retrained ensemble matches an earlier one in its decisions, not its weights.
* **Bundle** -- the five ``state_dicts``, the input contract (bands, dB constants, frame cap, context), the scalar standardization, the calibration table and decision block from the JSON, and the provenance (labels CSV, recipe with every seed, build date). An existing bundle path is refused, never overwritten, and the written bundle is loaded back through the detector's loader before the run ends.

Retraining the production bundle from its 4,680 labels and five seeds (one RTX 4080, 23 min including the input build) reproduced it in its decisions. Every input matched the cached training inputs of the original run bit for bit (that check read the full-band HPSS wavs, the input the step took then; on the broadband memmap the inputs differ by the tone removal and the reconstructed channel weights, see *Broadband MEMMAP*), as did the scalar standardization, and on the CPU the training loop gives weights identical to the original code's.

On the 1,118 consensus-labelled segments the cut-offs were chosen on (all of them training labels of both ensembles), weighted to the cohort, the confident decisions reach precision 0.988 / recall 0.993 (production 0.988 / 0.993) with 0.74% of segments uncertain (production 0.75%), and 99.85% of three-way decisions agree (99.91% of ``noise`` values). No segment moved between confident vocalization and confident noise. Unweighted, 94.8% of decisions agree (58 of 1,118, all between a confident decision and the uncertain band), because the set was drawn heavily from the uncertain score range (22% of it is uncertain).

Vocal-class model
^^^^^^^^^^^^^^^^^

The ensemble *Produce squeak labels* scores with. ``train-usv-squeak-model`` (``train_usv_squeak_model`` settings block) trains it on the labelling tool's files and writes a bundle ``detect-usv-squeaks`` loads unchanged (set it as ``squeak_model_path``). Drawing the segments to label, the labelling, the session-grouped cross-validation and the choice of the span rule are not part of it: they are done once per labelling round, outside the package, and come in as the label files and the ``span_threshold`` setting.

* **Labels** -- one or more label sets (``label_sets``), each a labels CSV (``panel_id``, ``label``: 0 usv, 1 squeak, 2 overlap, 3 unsure; ``squeak_extents_s``: the labelled spans as a JSON list of ``[start_s, end_s]``, every squeak in view, the context included; several spans per segment stay separate training targets of the squeak head, the envelope rule being an inference step) with its sample CSV (``panel_id``, ``session_dir``, ``row_index``, ``start``, ``stop``). Sample ids are ``<set name>_<panel_id>``. Review CSVs in the labels format (``label_overrides``) replace the label and spans of the panels they list in the named set (an empty spans cell in a review row means no span, so a review can turn a squeak / overlap panel into a usv one); the set's own CSV is never edited, and a review that lists a panel more than once stops the run, naming the repeated panels. Unsure answers are dropped; a ``usv`` label with spans, or a ``squeak`` / ``both`` label without one, stops the run. Each segment's ``chs_count`` is read from its session's *usv_summary.csv* row, which must still start where the sample says. The shipped settings list the production bundle's two rounds and review override.
* **Inputs** -- built by the same function *Produce squeak labels* scores with (same broadband memmap channels with their stored low-band variance weights, channel exclusion, context window and segment indicator), so a model is trained on exactly the input it will be run on; frame targets are 1 on every frame whose centre lies inside a labelled span. The scalars (log channel count, log duration) are standardized over the training set, and their mean and standard deviation go into the bundle.
* **Recipe** -- per seed, the network is initialized from ``torch.manual_seed(seed)``, its trunk loaded from member ``seed mod 5`` of the noise ensemble (``pretrained``; the noise bundle of ``detect_usv_noise.noise_model_path``), and trained with the noise model's recipe (40 epochs, batch 32, Adam 0.001 / weight decay 0.0001 under a cosine schedule, the same augmentation) on the class cross-entropy (label smoothing 0.05) plus the frame binary cross-entropy (weight 1), which supervises every frame of a squeak / both segment but only the segment's own frames of a usv segment (its label says nothing about the context).
* **Bundle** -- the five ``state_dicts``, the input contract, the scalar standardization, the class names, the extent threshold (``extent_rule``), the recipe with every seed and the label files. An existing bundle path is refused, never overwritten, and the written bundle is loaded back through the detector's loader before the run ends.

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

Notice that the plot contains two columns, one for each Avisoft UltraSoundGate hardware (USGH) device (which can operate in NO SYNC mode). In the first row, you can observe the distribution of A-V IPI discrepancies, which is the difference between the IPI onsets detected in the video and audio data. In the example, you can see the discrepancy goes rarely beyond one camera frame, which is ~6 ms, an acceptable amount of jitter.

One might also be interested in viewing how this discrepancy evolves over time. One thing we would want to avoid are drastic changes in sampling rates on any of the devices over time. In the second row, you can see the relationship between IPI onsets time (earlier-later in the session) and the A-V IPI discrepancy. Ideally, we would want to observe a *flat cloud* of points, which would indicate that the A/V IPI discrepancy is stable over time. If you observe a trend that goes beyond 2 tracking frames, it might be worth investigating further.

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

The */usv-playpen/_parameter_settings/processing_settings.json* file contains a section partially modifiable in the GUI (only *Arduino-USGH ch (1-12)*, ``sync_ch_receiving_input``, and *Sync camera serial num(s)*, ``sync_camera_serial_num``, are in the GUI; the rest are set in the file), with the following parameters:

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
        "relative_intensity_threshold_floor": 0.2,
        "relative_intensity_threshold_step": 0.01,
        "min_led_event_frames": 35,
        "millisecond_divergence_tolerance": 12,
        "unpaired_off_event_threshold_ceiling": 0.35
   }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`extra_data_camera <proc-extract-phidget-data-extra-data-camera>`
     - ``"22085397"``
     - camera serial number (string)
   * - :ref:`sync_ch_receiving_input <proc-find-audio-sync-trains-sync-ch-receiving-input>`
     - ``2``
     - microphone channel (GUI: 1-12)
   * - :ref:`extract_exact_video_frame_times_bool <proc-find-audio-sync-trains-extract-exact-video-frame-times-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`nidq_sr <proc-find-audio-sync-trains-nidq-sr>`
     - ``62500.72887``
     - Hz > 0
   * - :ref:`nidq_num_channels <proc-find-audio-sync-trains-nidq-num-channels>`
     - ``9``
     - integer > 0
   * - :ref:`nidq_bool <proc-find-audio-sync-trains-nidq-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`nidq_triggerbox_input_bit_position <proc-find-audio-sync-trains-nidq-triggerbox-input-bit-position>`
     - ``5``
     - bit position on the digital channel
   * - :ref:`nidq_sync_input_bit_position <proc-find-audio-sync-trains-nidq-sync-input-bit-position>`
     - ``7``
     - bit position on the digital channel
   * - :ref:`sync_camera_serial_num <proc-find-video-sync-trains-sync-camera-serial-num>`
     - ``["21372315"]``
     - list of camera serial numbers
   * - :ref:`led_px_version <proc-find-video-sync-trains-led-px-version>`
     - ``"current"``
     - LED pixel-position version
   * - :ref:`led_px_dev <proc-find-video-sync-trains-led-px-dev>`
     - ``10``
     - pixels > 0
   * - :ref:`sync_video_extension <proc-find-video-sync-trains-sync-video-extension>`
     - ``"mp4"``
     - video file extension
   * - :ref:`relative_intensity_threshold <proc-find-video-sync-trains-relative-intensity-threshold>`
     - ``1.0``
     - number on a 0-1 scale
   * - :ref:`relative_intensity_threshold_floor <proc-find-video-sync-trains-relative-intensity-threshold-sweep>`
     - ``0.2``
     - number below ``relative_intensity_threshold``
   * - :ref:`relative_intensity_threshold_step <proc-find-video-sync-trains-relative-intensity-threshold-sweep>`
     - ``0.01``
     - step size > 0
   * - :ref:`min_led_event_frames <proc-find-video-sync-trains-min-led-event-frames>`
     - ``35``
     - frames > 0
   * - :ref:`millisecond_divergence_tolerance <proc-find-video-sync-trains-millisecond-divergence-tolerance>`
     - ``12``
     - ms > 0
   * - :ref:`unpaired_off_event_threshold_ceiling <proc-find-video-sync-trains-unpaired-off-event-threshold-ceiling>`
     - ``0.35``
     - sweep-threshold value

.. _proc-extract-phidget-data-extra-data-camera:

``extra_data_camera``
   Serial number of the camera used to store phidget data.

   :Options: a camera serial number, as a string
   :Default: ``"22085397"``
   :Notes: settings file only (``extract_phidget_data`` → ``Gatherer`` → ``prepare_data_for_analyses``); not in the GUI.

.. _proc-find-audio-sync-trains-sync-ch-receiving-input:

``sync_ch_receiving_input``
   Microphone channel receiving Arduino digital input.

   :Options: a microphone channel number
   :Default: ``2``
   :Notes: set in the GUI as *Arduino-USGH ch (1-12)*.

.. _proc-find-audio-sync-trains-extract-exact-video-frame-times-bool:

``extract_exact_video_frame_times_bool``
   Instead of using frame indices multiplied by empirical frame rate, use Loopbio times directly (which is less precise!).

   :Options:
      * ``true`` — use the Loopbio frame times directly (less precise)
      * ``false`` — use frame indices multiplied by the empirical frame rate
   :Default: ``false``
   :Notes: settings file only.

.. _proc-find-audio-sync-trains-nidq-sr:

``nidq_sr``
   Sampling rate of the NIDQ device (in Hz).

   :Options: a positive sampling rate, in Hz
   :Default: ``62500.72887``
   :Notes: settings file only.

.. _proc-find-audio-sync-trains-nidq-num-channels:

``nidq_num_channels``
   Number of channels on the NIDQ device (9 on BNC-2110).

   :Options: a positive integer, the device's channel count
   :Default: ``9``
   :Notes: settings file only.

.. _proc-find-audio-sync-trains-nidq-bool:

``nidq_bool``
   Whether NIDQ device received Triggerbox AND sync input.

   :Options:
      * ``true`` — the NIDQ device received both the Triggerbox and the sync input
      * ``false`` — it did not
   :Default: ``false``
   :Notes: settings file only.

.. _proc-find-audio-sync-trains-nidq-triggerbox-input-bit-position:

``nidq_triggerbox_input_bit_position``
   Triggerbox input bit position on the NIDQ device digital channel.

   :Options: a bit position on the NIDQ digital channel
   :Default: ``5``
   :Notes: assumes last channel is digital! Settings file only.

.. _proc-find-audio-sync-trains-nidq-sync-input-bit-position:

``nidq_sync_input_bit_position``
   Sync input bit position on the NIDQ device digital channel.

   :Options: a bit position on the NIDQ digital channel
   :Default: ``7``
   :Notes: assumes last channel is digital! Settings file only.

.. _proc-find-video-sync-trains-sync-camera-serial-num:

``sync_camera_serial_num``
   Serial numbers of cameras that can detect flashing LEDs.

   :Options: a list of camera serial numbers, as strings
   :Default: ``["21372315"]``
   :Notes: set in the GUI as *Sync camera serial num(s)*.

.. _proc-find-video-sync-trains-led-px-version:

``led_px_version``
   Version of the LED pixel positions.

   :Options: a version key of the LED pixel-position table
   :Default: ``"current"``
   :Notes: settings file only.

.. _proc-find-video-sync-trains-led-px-dev:

``led_px_dev``
   Maximal deviation (in px) of observed LED flashes relative to expected positions.

   :Options: a positive number of pixels
   :Default: ``10``
   :Notes: settings file only.

.. _proc-find-video-sync-trains-sync-video-extension:

``sync_video_extension``
   Video type (usually "mp4").

   :Options: a video file extension
   :Default: ``"mp4"``
   :Notes: settings file only.

.. _proc-find-video-sync-trains-relative-intensity-threshold:

``relative_intensity_threshold``
   Top threshold (on 0-1 scale) for relative temporal change in pixel intensity.

   :Options: a number on a 0-1 scale
   :Default: ``1.0``
   :Notes: the starting point of the threshold sweep set by ``relative_intensity_threshold_floor`` / ``relative_intensity_threshold_step``. Settings file only.

.. _proc-find-video-sync-trains-relative-intensity-threshold-sweep:

``relative_intensity_threshold_floor`` / ``relative_intensity_threshold_step``
   The threshold is swept down from ``relative_intensity_threshold`` to this floor in these steps, stopping at the first threshold that yields the Arduino IPI sequence.

   :Options:
      * ``relative_intensity_threshold_floor`` — the lowest threshold tried, on the same 0-1 scale
      * ``relative_intensity_threshold_step`` — the decrement between successive thresholds
   :Default: ``0.2`` and ``0.01``
   :Notes: settings file only.

.. _proc-find-video-sync-trains-min-led-event-frames:

``min_led_event_frames``
   LED on / off events shorter than this many frames are discarded as glitches.

   :Options: a positive number of frames
   :Default: ``35``
   :Notes: settings file only.

.. _proc-find-video-sync-trains-millisecond-divergence-tolerance:

``millisecond_divergence_tolerance``
   Maximal deviation of IPI onsets (in ms) between video detections and ground truth.

   :Options: a positive number of milliseconds
   :Default: ``12``
   :Notes: settings file only.

.. _proc-find-video-sync-trains-unpaired-off-event-threshold-ceiling:

``unpaired_off_event_threshold_ceiling``
   At sweep thresholds below this value an event train with one more OFF than ON event is also accepted (its first OFF, which precedes the first ON, is dropped); above it the ON and OFF counts must pair up.

   :Options: a threshold value on the sweep's 0-1 scale
   :Default: ``0.35``
   :Notes: not in the GUI.

The look of the *summary* figure is set in a section not modifiable in the GUI, but it can be modified manually (the figure's file format and dpi come from the visualization settings):

.. code-block:: json

   "preprocessing_plot": {
    "SummaryPlotter": {
      "preprocessing_summary": {
        "figure_size": [12.8, 9.6],
        "discrepancy_face_color": "#BBD5E826",
        "discrepancy_edge_color": "#00000040",
        "zero_line_color": "#202020",
        "median_color": "#FF6347",
        "mean_color": "#00C78C",
        "humidity_color": "#8EE5EE",
        "illumination_color": "#EEB422",
        "temperature_color": "#FF7F50"
      }
    }
   }

.. list-table::
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`figure_size <proc-preprocessing-summary-figure-size>`
     - ``[12.8, 9.6]``
     - ``[width, height]`` in inches
   * - :ref:`discrepancy_face_color <proc-preprocessing-summary-discrepancy-colors>`
     - ``#BBD5E826``
     - hex colour with alpha
   * - :ref:`discrepancy_edge_color <proc-preprocessing-summary-discrepancy-colors>`
     - ``#00000040``
     - hex colour with alpha
   * - :ref:`zero_line_color <proc-preprocessing-summary-zero-line-color>`
     - ``#202020``
     - hex colour
   * - :ref:`median_color <proc-preprocessing-summary-median-mean-colors>`
     - ``#FF6347``
     - hex colour
   * - :ref:`mean_color <proc-preprocessing-summary-median-mean-colors>`
     - ``#00C78C``
     - hex colour
   * - :ref:`humidity_color <proc-preprocessing-summary-phidget-colors>`
     - ``#8EE5EE``
     - hex colour
   * - :ref:`illumination_color <proc-preprocessing-summary-phidget-colors>`
     - ``#EEB422``
     - hex colour
   * - :ref:`temperature_color <proc-preprocessing-summary-phidget-colors>`
     - ``#FF7F50``
     - hex colour

.. _proc-preprocessing-summary-figure-size:

``figure_size``
   Width and height of the figure (in inches).

   :Options: a two-element list ``[width, height]``, in inches
   :Default: ``[12.8, 9.6]``

.. _proc-preprocessing-summary-discrepancy-colors:

``discrepancy_face_color`` / ``discrepancy_edge_color``
   Hex fill and edge colours (with alpha) of the A-V IPI discrepancy histograms and scatter plots, and of the NIDQ-V inset.

   :Options:
      * ``discrepancy_face_color`` — the fill colour, a hex string with alpha
      * ``discrepancy_edge_color`` — the edge colour, a hex string with alpha
   :Default: ``#BBD5E826`` / ``#00000040``

.. _proc-preprocessing-summary-zero-line-color:

``zero_line_color``
   Hex colour of the zero-discrepancy line.

   :Options: a hex colour string
   :Default: ``#202020``

.. _proc-preprocessing-summary-median-mean-colors:

``median_color`` / ``mean_color``
   Hex colours of the median and mean discrepancy lines and their labels.

   :Options:
      * ``median_color`` — the median line and its label, a hex colour string
      * ``mean_color`` — the mean line and its label, a hex colour string
   :Default: ``#FF6347`` / ``#00C78C``

.. _proc-preprocessing-summary-phidget-colors:

``humidity_color`` / ``illumination_color`` / ``temperature_color``
   Hex colours of the phidget humidity, illumination and temperature panels.

   :Options:
      * ``humidity_color`` — the humidity panel, a hex colour string
      * ``illumination_color`` — the illumination panel, a hex colour string
      * ``temperature_color`` — the temperature panel, a hex colour string
   :Default: ``#8EE5EE`` / ``#EEB422`` / ``#FF7F50``
