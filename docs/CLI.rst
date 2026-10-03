.. _CLI:

Command Line Interfaces (CLI)
=============================
This page explains how to use the *usv-playpen* CLI (command line interfaces).

Record
------

``conduct-calibration``
``conduct-calibration`` is the command-line interface for performing a tracking camera calibration.

.. code-block:: text

    usage: conduct-calibration [-h] [--set KEY.PATH=VALUE]...

    optional arguments:
      -h, --help            Show this help message and exit.
      --set                 Override a specific setting using a dot-path. This option
                            can be used multiple times. For example:
                            --set calibration_duration=10
                            --set video.general.calibration_frame_rate=20

``conduct-recording``
``conduct-recording`` is the command-line interface for conducting a recording session.

.. code-block:: text

    usage: conduct-recording [-h] [--set KEY.PATH=VALUE]...

    optional arguments:
      -h, --help            Show this help message and exit.
      --set                 Override a specific setting using a dot-path. This option
                            can be used multiple times. For example:
                            --set video_session_duration=20
                            --set audio.general.fftlength=512
                            --set arduino_sync_port=COM7

Process
-------

``concatenate-ephys-files``
``concatenate-ephys-files`` is the command-line interface for concatenating electrophysiology (ephys) binary files across multiple sessions.

.. code-block:: text

    usage: concatenate-ephys-files [-h] --root-directories TEXT,TEXT,...

    required arguments:
      --root-directories    A comma-separated string of session root directory paths.

    optional arguments:
      -h, --help            Show this help message and exit.

``split-clusters``
``split-clusters`` is the command-line interface for splitting curated ephys clusters into individual session files.

.. code-block:: text

    usage: split-clusters [-h] --root-directories TEXT,TEXT,...
                          [--min-spikes INTEGER] [--kilosort-version TEXT]
                          [--remove-duplicate-spikes | --no-remove-duplicate-spikes]
                          [--censored-period-ms FLOAT]

    required arguments:
      --root-directories    A comma-separated string of session root directory paths.

    optional arguments:
      -h, --help            Show this help message and exit.
      --min-spikes          Minimum number of spikes for a cluster to be saved.
      --kilosort-version    Version of Kilosort used for spike sorting.
      --remove-duplicate-spikes / --no-remove-duplicate-spikes
                            Drop near-coincident duplicate spikes (e.g. from Phy merges) per unit.
      --censored-period-ms  Censored period (in ms) for duplicate-spike removal.

``concatenate-video-files``
``concatenate-video-files`` is the command-line interface for concatenating video files.

.. code-block:: text

    usage: concatenate-video-files  [-h] --root-directory PATH
                                    [--camera-serial TEXT]
                                    [--extension TEXT]
                                    [--output-name TEXT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --camera-serial       Camera serial number(s).
      --extension           Video file extension.
      --output-name         Name of the concatenated file.

``rectify-video-fps``
``rectify-video-fps`` is the command-line interface for re-encoding videos to a correct frame rate.

.. code-block:: text

    usage: rectify-video-fps [-h] --root-directory PATH [--camera-serial TEXT...]
                             [--target-file TEXT] [--extension TEXT]
                             [--crf INTEGER]
                             [--preset {ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow}]
                             [--delete-old-file | --no-delete-old-file]
                             [--conduct-concat | --no-conduct-concat]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --camera-serial       Camera serial number(s).
      --target-file         Name of the target video file.
      --extension           Video file extension.
      --crf                 FFMPEG -crf (e.g., 16).
      --preset              FFMPEG encoding speed preset.
      --delete-old-file / --no-delete-old-file
                            Deletes the original file after encoding.
      --conduct-concat / --no-conduct-concat
                            Indicate if prior concatenation was performed

``multichannel-to-single-ch``
``multichannel-to-single-ch`` is the command-line interface for splitting multichannel audio files into single-channel files.

.. code-block:: text

    usage: multichannel-to-single-ch [-h] --root-directory PATH

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.

``crop-wav-files``
``crop-wav-files`` is the command-line interface for cropping audio WAV files to match video length.

.. code-block:: text

    usage: crop-wav-files [-h] --root-directory PATH [--trigger-device {both,m,r}]
                          [--trigger-channel INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --trigger-device      USGH device(s) receiving triggerbox input.
      --trigger-channel     USGH channel receiving triggerbox input.

``av-sync-check``
``av-sync-check`` is the command-line interface for checking audio-video synchronization and generating a summary figure.

.. code-block:: text

    usage: av-sync-check [-h] --root-directory PATH [--extra-camera TEXT]
                         [--audio-sync-ch INTEGER]
                         [--exact-frame-times | --no-exact-frame-times]
                         [--nidq-sr FLOAT] [--nidq-channels INTEGER]
                         [--nidq-trigger-bit INTEGER] [--nidq-sync-bit INTEGER]
                         [--video-sync-camera TEXT...] [--led-version TEXT]
                         [--led-dev INTEGER] [--video-extension TEXT]
                         [--intensity-thresh FLOAT] [--ms-tolerance INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --extra-camera        Camera serial number for extra data.
      --audio-sync-ch       Audio channel receiving sync input.
      --exact-frame-times / --no-exact-frame-times
                            Extract exact video frame times.
      --nidq-sr             NI-DAQ sampling rate (Hz).
      --nidq-channels       Number of NI-DAQ channels.
      --nidq-trigger-bit    NI-DAQ triggerbox input bit position.
      --nidq-sync-bit       NI-DAQ sync input bit position.
      --video-sync-camera   Camera serial number for video sync.
      --led-version         Version of the LED pixel used for sync.
      --led-dev             LED pixel deviation value.
      --video-extension     Video extension for sync files.
      --intensity-thresh    Relative intensity threshold for LED detection.
      --ms-tolerance        Divergence tolerance (in ms).

``ev-sync-check``
``ev-sync-check`` is the command-line interface for validating ephys-video synchronization.

.. code-block:: text

    usage: ev-sync-check [-h] --root-directory PATH [--file-type {ap,lf}]
                         [--tolerance FLOAT]
                         [--apply-phase-shift | --no-apply-phase-shift]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --file-type           Neuropixels file type (ap or lf).
      --tolerance           Divergence tolerance (in ms).
      --apply-phase-shift / --no-apply-phase-shift
                            De-skew each AP binary for the Neuropixels ADC sample-time offset (in place).

``hpss-audio``
``hpss-audio`` is the command-line interface for performing Harmonic-Percussive Source Separation (HPSS) on audio files.

.. code-block:: text

    usage: hpss-audio [-h] --root-directory PATH [--stft-params INTEGER INTEGER]
                      [--kernel-size INTEGER INTEGER] [--power FLOAT]
                      [--margin INTEGER INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --stft-params         STFT window length and hop size.
      --kernel-size         Median filter kernel size (harmonic, percussive).
      --power               HPSS power parameter.
      --margin              HPSS margin (harmonic, percussive).

``bp-filter-audio``
``bp-filter-audio`` is the command-line interface for band-pass filtering audio files.

.. code-block:: text

    usage: bp-filter-audio [-h] --root-directory PATH [--format TEXT]
                           [--dirs TEXT...] [--freq-bounds INTEGER INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --format              Audio file format.
      --dirs                Directory/ies containing files to filter.
      --freq-bounds         Frequency bounds for the band-pass filter (Hz).

``concatenate-audio-files``
``concatenate-audio-files`` is the command-line interface for vertically stacking audio files into a single memmap file.

.. code-block:: text

    usage: concatenate-audio-files [-h] --root-directory PATH
                                   [--format TEXT] [--dirs TEXT...]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --format              Audio file format.
      --dirs                Directory/ies to search for files to concatenate.

``sleap-to-h5``
``sleap-to-h5`` is the command-line interface for converting SLEAP (the SLEAP pose-tracking framework) ``.slp`` files to hierarchical data format (HDF5) ``.h5`` files.

.. code-block:: text

    usage: sleap-to-h5 [-h] --root-directory PATH

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.

``anipose-calibrate``
``anipose-calibrate`` is the command-line interface for conducting Anipose camera calibration.

.. code-block:: text

    usage: anipose-calibrate [-h] --root-directory PATH
                             [--board-provided]
                             [--board-dims INTEGER INTEGER] [--square-len INTEGER]
                             [--marker-params FLOAT FLOAT] [--dict-size INTEGER]
                             [--img-dims INTEGER INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --board-provided      Indicate that the calibration board is provided.
      --board-dims          Checkerboard dimensions (squares_x, squares_y).
      --square-len          Length of a checkerboard square (mm).
      --marker-params       ArUco marker length (mm) and dictionary bits.
      --dict-size           Size of the ArUco dictionary.
      --img-dims            Image dimensions (width, height) in pixels.

``anipose-triangulate``
``anipose-triangulate`` is the command-line interface for conducting Anipose 3D triangulation.

.. code-block:: text

    usage: anipose-triangulate [-h] --root-directory PATH --cal-directory PATH
                               [--arena-points | --no-arena-points]
                               [--frame-restriction INTEGER]
                               [--exclude-views TEXT...]
                               [--display-progress | --no-display-progress]
                               [--use-ransac | --no-use-ransac]
                               [--rigid-constraint "TEXT,TEXT"...]
                               [--weak-constraint "TEXT,TEXT"...] [--smooth-scale FLOAT]
                               [--weight-weak INTEGER] [--weight-rigid INTEGER]
                               [--reprojection-threshold INTEGER] [--regularization {l1,l2}]
                               [--n-deriv-smooth INTEGER]

    required arguments:
      --root-directory           Session root directory path.
      --cal-directory            Path to the Anipose calibration session.

    optional arguments:
      -h, --help                 Show this help message and exit.
      --arena-points / --no-arena-points
                                 Triangulate arena points instead of animal points.
      --frame-restriction        Restrict triangulation to a specific number of frames.
      --exclude-views            Camera views to exclude from triangulation.
      --display-progress / --no-display-progress
                                 Display the progress bar during triangulation.
      --use-ransac / --no-use-ransac
                                 Use RANSAC for robust triangulation.
      --rigid-constraint        Pair(s) of nodes for a rigid constraint.
      --weak-constraint         Pair(s) of nodes for a weak constraint.
      --smooth-scale             Scaling factor for smoothing.
      --weight-weak              Weight for weak constraints.
      --weight-rigid             Weight for rigid constraints.
      --reprojection-threshold   Reprojection error threshold.
      --regularization           Regularization function to use.
      --n-deriv-smooth           Number of derivatives to use for smoothing.

``anipose-trm``
``anipose-trm`` is the command-line interface for translating, rotating, and scaling 3D point data.

.. code-block:: text

    usage: anipose-trm [-h] --root-directory PATH --exp-code TEXT --arena-directory PATH
                       [--save-data-for {animal,arena}]
                       [--delete-original | --no-delete-original]
                       [--ref-len FLOAT]

    required arguments:
      --root-directory      Session root directory path.
      --exp-code            Experimental code.
      --arena-directory     Path to the original arena session.

    optional arguments:
      -h, --help            Show this help message and exit.
      --save-data-for       Data to save after transformation.
      --delete-original / --no-delete-original
                            Delete the original data after transformation.
      --ref-len             Length of the static reference object.

``das-infer``
``das-infer`` is the command-line interface for running Deep Audio Segmenter (DAS) inference on audio files.

.. code-block:: text

    usage: das-infer [-h] --root-directory PATH [--env-name TEXT] [--model-dir PATH]
                     [--model-name TEXT] [--output-type {csv,hdf5}]
                     [--confidence-thresh FLOAT] [--min-len FLOAT] [--fill-gap FLOAT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --env-name            Name of the DAS conda environment.
      --model-dir           Directory of the DAS model.
      --model-name          Base name of the DAS model.
      --output-type         Output file type for DAS predictions.
      --confidence-thresh   Confidence threshold for segment detection.
      --min-len             Minimum length for a detected segment (s).
      --fill-gap            Gap duration to fill between segments (s).

``das-summarize``
``das-summarize`` is the command-line interface for summarizing DAS inference findings.

.. code-block:: text

    usage: das-summarize [-h] --root-directory PATH
                         [--filter-putative-noise | --no-filter-putative-noise]
                         [--win-len INTEGER] [--freq-cutoff INTEGER]
                         [--corr-cutoff FLOAT] [--coherence-cutoff FLOAT]
                         [--coherence-channel-count INTEGER]
                         [--max-usv-duration | --no-max-usv-duration]
                         [--max-usv-duration-s FLOAT]
                         [--consensus-remerge | --no-consensus-remerge]
                         [--consensus-remerge-min-duration-s FLOAT]
                         [--consensus-remerge-span-factor FLOAT]
                         [--consensus-remerge-max-dissenting-channels INTEGER]
                         [--consensus-remerge-min-agreeing-channels INTEGER]
                         [--consensus-remerge-max-depth INTEGER]
                         [--consensus-remerge-min-gap-s FLOAT]
                         [--consensus-remerge-rescue-dissenting-channels INTEGER]
                         [--seam-repair | --no-seam-repair]
                         [--seam-repair-legacy-stride-samples INTEGER]
                         [--seam-repair-max-rung INTEGER]
                         [--seam-repair-ladder-tolerance-samples INTEGER]
                         [--seam-repair-width-tolerance-below-ms FLOAT]
                         [--seam-repair-width-tolerance-above-ms FLOAT]
                         [--seam-repair-raw-stop-tolerance-s FLOAT]
                         [--seam-repair-snippet-margin-s FLOAT]
                         [--seam-repair-max-boundary-shift-s FLOAT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --filter-putative-noise / --no-filter-putative-noise
                            Run the Phase-4 amplitude/spectrogram noise rejection (default: enabled); pass --no-filter-putative-noise to keep every merged detection.
      --win-len             Window length of the signal.
      --freq-cutoff         Low frequency cutoff (Hz).
      --corr-cutoff         Absolute detected-channel correlation cutoff (drop below).
      --coherence-cutoff    Absolute top-K spatial-coherence cutoff (drop below).
      --coherence-channel-count
                            Number of loudest in-band channels the spatial coherence is computed over.
      --max-usv-duration / --no-max-usv-duration
                            Reject merged intervals longer than --max-usv-duration-s before the noise checks run (default: enabled).
      --max-usv-duration-s
                            Longest a merged interval may be (s) before it is rejected as noise; no mouse USV lasts this long.
      --consensus-remerge / --no-consensus-remerge
                            Re-merge intervals whose extent is set by one channel contradicting the majority (default: enabled). Pass --no-consensus-remerge to take the union exactly as detected.
      --consensus-remerge-min-duration-s
                            Consensus re-merge: only merged intervals longer than this (s) are examined, so ordinary calls are never reconsidered.
      --consensus-remerge-span-factor
                            Consensus re-merge: a contributing channel counts as dissenting when its longest segment exceeds this multiple of the median longest-segment across contributing channels.
      --consensus-remerge-max-dissenting-channels
                            Consensus re-merge: fire only when at most this many channels dissent; more than that is not a lone dissenter.
      --consensus-remerge-min-agreeing-channels
                            Consensus re-merge: fire only when at least this many channels remain once dissenters are set aside.
      --consensus-remerge-max-depth
                            Consensus re-merge: how many times a corrected interval may itself be re-examined; 0 reproduces the single-pass behaviour.
      --consensus-remerge-min-gap-s
                            Consensus re-merge: narrowest gap (s) a cut may sit in; below the segmenter's own --segment-fillgap the gap was manufactured by the union, not found in the audio.
      --consensus-remerge-rescue-dissenting-channels
                            Consensus re-merge: dissent limit used only for an interval about to be rejected for exceeding --max-usv-duration-s.
      --seam-repair / --no-seam-repair
                            Run the post-summary seam check-and-repair (default: disabled). Only meaningful for annotations from the legacy non-overlapping model (stride 8128); pass --seam-repair to enable.
      --seam-repair-legacy-stride-samples
                            Seam repair: window-stitching stride (samples) of the legacy DAS model whose seam artifact is checked for.
      --seam-repair-max-rung
                            Seam repair: highest stride multiple tested for the seam-ladder fingerprint.
      --seam-repair-ladder-tolerance-samples
                            Seam repair: maximum deviation (samples) from the exact ladder fingerprint in raw annotations.
      --seam-repair-width-tolerance-below-ms
                            Seam repair: merged-gap width gate reach below each stride multiple (ms).
      --seam-repair-width-tolerance-above-ms
                            Seam repair: merged-gap width gate reach above each stride multiple (ms).
      --seam-repair-raw-stop-tolerance-s
                            Seam repair: maximum distance (s) between a merged stop and a raw ladder-gap stop for corroboration.
      --seam-repair-snippet-margin-s
                            Seam repair: audio margin (s) excised around each flagged pair for re-detection.
      --seam-repair-max-boundary-shift-s
                            Seam repair: maximum outward correction (s) permitted per call edge.

``detect-usv-noise``
``detect-usv-noise`` flags the USV segments of a session that hold no vocalization at all and merges two columns into ``usv_summary.csv``: ``noise`` (true / false in every row; true excludes the segment) and ``noise_probability`` (the ensemble's probability in every row, so an analysis can separate confident noise from the uncertain band, or re-threshold, without re-running). The decision comes from the model bundle, not from an option: ``noise`` is true for confident noise and for the uncertain band (``noise_probability`` >= 0.14), whose confident decisions reach a held-out precision of 0.944 and recall of 0.955 (0.9% of segments uncertain; see *Detect noise* in :doc:`Process`). The step prints these numbers on every run. The input spectrogram is rebuilt from the unfiltered ``audio/hpss`` wavs (30-120 kHz and 3-30 kHz, absolute dB) over a window that extends ~100 ms either side of the segment and is then cropped back to it; run it after ``das-summarize``, which rewrites the summary without these columns, and before ``detect-usv-squeaks``. A GPU is used when present but is not required (about 64 s against 90 s on the CPU for a 424-USV session).

.. code-block:: text

    usage: detect-usv-noise [-h] --root-directory PATH
                            [--noise-model-path TEXT]
                            [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                            [--batch-size INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --noise-model-path    Path to the noise model bundle (.pt); derived from spectrograms_root when empty.
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                            Drop channels the session metadata marks as excluded from the spectrogram average.
      --batch-size          Segments per forward pass at the typical call length; a batch is budgeted at batch-size x 128 frame slots, so one long call never inflates it.

``train-noise-model``
``train-noise-model`` trains the ``detect-usv-noise`` ensemble (one model per seed) on a labels CSV and writes a bundle that ``detect-usv-noise`` loads unchanged. The CSV holds one row per training example: ``sample_id``, ``session_dir``, ``start``, ``stop``, ``chs_count`` and ``noise`` (``1`` = no vocalization at all, ``0`` = a vocalization; drop unsure answers first). Every input is rebuilt from the session's unfiltered ``audio/hpss`` wavs by the same function ``detect-usv-noise`` scores with, and the models are trained with the production recipe (40 epochs, batch 32, Adam at 0.001 with a cosine schedule, label smoothing 0.05, spectrogram augmentation). The calibration table and the decision block the detector reads are not measured here: they come from the ``--calibration-path`` JSON and are written into the bundle as given. An existing bundle path is refused. GPU recommended (see *Noise model* in :doc:`Process`).

.. code-block:: text

    usage: train-noise-model [-h] --labels-csv FILE --bundle-path FILE
                            [--calibration-path TEXT]
                            [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                            [--n-workers INTEGER] [--seed INTEGER ...]
                            [--epochs INTEGER] [--batch-size INTEGER]
                            [--learning-rate FLOAT] [--weight-decay FLOAT]
                            [--label-smoothing FLOAT]

    required arguments:
      --labels-csv          Labels CSV: sample_id, session_dir, start, stop, chs_count, noise (1 = no vocalization, 0 = vocalization).
      --bundle-path         Output bundle (.pt); must not exist yet.

    optional arguments:
      -h, --help            Show this help message and exit.
      --calibration-path    JSON with the calibration table and the decision block the bundle carries.
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                            Drop channels the session metadata marks as excluded from the spectrogram average (keep it as detect-usv-noise runs).
      --n-workers           Sessions whose audio is read concurrently (threads) while the training inputs are built.
      --seed                Seed of one ensemble member; repeat once per member (replaces the seeds setting).
      --epochs              Training epochs per member.
      --batch-size          Segments per training batch.
      --learning-rate       Adam learning rate (cosine-annealed over the run).
      --weight-decay        Adam weight decay.
      --label-smoothing     Label smoothing towards 0.5.

``detect-usv-squeaks``
``detect-usv-squeaks`` classifies every USV segment of a session that is not noise as a pure USV, a pure squeak, or a segment holding both, and merges seven columns into ``usv_summary.csv`` right after the noise columns: the booleans ``usv`` and ``squeak`` (pure USV = ``usv`` true, ``squeak`` false; pure squeak = ``usv`` false, ``squeak`` true; both = both true; empty on noise rows -- a USV-only selection must require ``squeak`` false too), the ensemble-mean class probabilities ``p_usv`` / ``p_squeak`` / ``p_both``, and on ``squeak`` rows ONE squeak extent ``squeak_start`` / ``squeak_end`` (session seconds): the envelope of the frames whose squeak probability exceeds the bundle's threshold (0.6), counting only above-threshold runs that touch the segment and with no minimum run, so it may reach past the segment; a squeak row with no such frame falls back to its highest-scoring segment frame, and the step reports how often. The step needs the ``noise`` column, and removes the columns of earlier encodings (``squeak_probability``, ``squeak_frame_runs``, ``call_class``, ``squeak_spans``, ``n_squeaks``). Each segment's input is the noise model's two-band spectrogram (30-120 and 3-30 kHz, absolute dB, from the unfiltered ``audio/hpss`` wavs) of the segment plus ~100 ms of context either side, with a channel that marks the segment's own frames. The bundle (derived from ``spectrograms_root`` as ``squeak/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt``) supplies the threshold and every input constant. Run it after ``das-summarize`` and ``detect-usv-noise``. GPU recommended.

.. code-block:: text

    usage: detect-usv-squeaks [-h] --root-directory PATH
                            [--squeak-model-path TEXT]
                            [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                            [--batch-size INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --squeak-model-path   Path to the call-class (usv / squeak / both) model bundle (.pt); derived from spectrograms_root when empty.
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                            Drop channels the session metadata marks as excluded from the spectrogram average.
      --batch-size          Segments per forward pass at the typical window length; a batch is budgeted at batch-size x 128 frame slots, so one long window never inflates it.

``train-usv-squeak-model``
``train-usv-squeak-model`` trains the ``detect-usv-squeaks`` ensemble (one member per seed) on the labelling tool's files and writes a bundle ``detect-usv-squeaks`` loads unchanged. A label set is a labels CSV (``panel_id``, ``label`` -- 0 usv, 1 squeak, 2 both, 3 unsure -- and ``squeak_extents_s``, the labelled spans as JSON) with its sample CSV (``panel_id``, ``session_dir``, ``row_index``, ``start``, ``stop``); a label override is a review CSV in the labels format whose panels replace those of the named set. Unsure answers are dropped; each segment's ``chs_count`` is read from its session's summary row (which must still start where the sample says). Every input is rebuilt by the same function ``detect-usv-squeaks`` scores with; each member's trunk starts from a noise-model member (``--pretrained``, the shipped setting) and is trained with the noise model's recipe on the joint class + frame loss. The extent threshold is not tuned here: ``--span-threshold`` (0.6, chosen on session-grouped cross-fitted predictions outside the package) is written into the bundle as given. The labelled spans (several per segment where marked) are the squeak head's training targets; the one-extent envelope rule is applied at inference only. The shipped settings list the two labelling rounds and the review override of the production bundle. An existing bundle path is refused. GPU recommended (see *Call-class model* in :doc:`Process`).

.. code-block:: text

    usage: train-usv-squeak-model [-h] --bundle-path FILE
                            [--label-set NAME LABELS_CSV SAMPLE_CSV ...]
                            [--label-override NAME OVERRIDE_CSV ...]
                            [--pretrained | --no-pretrained]
                            [--span-threshold FLOAT]
                            [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                            [--n-workers INTEGER] [--seed INTEGER ...]
                            [--epochs INTEGER] [--batch-size INTEGER]
                            [--learning-rate FLOAT] [--weight-decay FLOAT]
                            [--label-smoothing FLOAT] [--frame-loss-weight FLOAT]
                            [--class-weighted | --no-class-weighted]

    required arguments:
      --bundle-path         Output bundle (.pt); must not exist yet.

    optional arguments:
      -h, --help            Show this help message and exit.
      --label-set           NAME LABELS_CSV SAMPLE_CSV of one label set (labelling-tool labels and their sample); repeat per set (replaces the label_sets setting).
      --label-override      NAME OVERRIDE_CSV: a review CSV (labels format) whose panels replace those of label set NAME; repeat per file (replaces the label_overrides setting).
      --pretrained / --no-pretrained
                            Initialize every member's trunk from the noise ensemble (detect_usv_noise.noise_model_path).
      --span-threshold      Frame squeak-probability threshold of the squeak-extent rule the bundle carries.
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                            Drop channels the session metadata marks as excluded from the spectrogram average (keep it as detect-usv-squeaks runs).
      --n-workers           Sessions whose audio is read concurrently (threads) while the training inputs are built.
      --seed                Seed of one ensemble member; repeat once per member (replaces the seeds setting).
      --epochs              Training epochs per member.
      --batch-size          Segments per training batch.
      --learning-rate       Adam learning rate (cosine-annealed over the run).
      --weight-decay        Adam weight decay.
      --label-smoothing     Label smoothing of the class cross-entropy.
      --frame-loss-weight   Weight of the frame squeak loss relative to the class loss.
      --class-weighted / --no-class-weighted
                            Weight the class loss by inverse training class frequency.

``infer-qlvm-squeak-latents``
``infer-qlvm-squeak-latents`` places every segment of a session with ``squeak`` true (pure squeaks and segments holding both) that is not noise on the torus of one of the phase 3 broadband-vocalization QLVM cells (``qlvm_models_latest/phase3_BBVs_qlvm/<cell>``) and merges two float columns into ``usv_summary.csv``: ``qlvm_squeak1`` / ``qlvm_squeak2``, the torus coordinates in ``[0, 1)`` (null on every other row; no cluster labels). Each segment's input is the cells' sonic spectrogram (3-30 kHz, absolute dB) cropped to the frames inside its squeak extent ``squeak_start`` .. ``squeak_end`` plus two frames either side, min-max normalized per crop and centred in a 128-frame frame, exactly as the cells' training sets were built; the audio is read over the segment widened to hold the envelope and its context, because a squeak often runs past its segment. A segment with several squeaks is embedded once, over its one extent (their envelope). Crops narrower than 8 frames (a one-frame fallback extent always is) or wider than 128 frames get nulls. The cell is the ``infer_qlvm_squeak_latents.model_cell_directory`` setting; left empty (the shipped value), it is filled with the production cell ``phase3_BBVs_qlvm/natural_session_N11000_nomask`` (``os_utils.QLVM_SQUEAK_PRODUCTION_CELL``) whenever ``spectrograms_root`` is set, and ``--model-cell-directory`` names another. Run it after ``detect-usv-noise`` and ``detect-usv-squeaks``. A GPU speeds it up but is not required.

.. code-block:: text

    usage: infer-qlvm-squeak-latents [-h] --root-directory PATH
                            [--model-cell-directory TEXT]
                            [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                            [--lattice-batch-size INTEGER] [--data-batch-size INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --model-cell-directory
                            A squeak (BBV) QLVM cell; filled with the production phase3_BBVs_qlvm/natural_session_N11000_nomask cell when empty and spectrograms_root is set.
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                            Drop channels the session metadata marks as excluded from the spectrogram average (keep it equal to the detect-usv-squeaks run).
      --lattice-batch-size  Lattice points decoded and scored per block; lower it to cut memory.
      --data-batch-size     Squeaks whose lattice posteriors are computed together; memory grows with this times the lattice size.

``prepare-vcl-assign``
``prepare-vcl-assign`` is the command-line interface for preparing data for vocalization assignment using the Vocalocator sound-source localizer.

.. code-block:: text

    usage: prepare-vcl-assign [-h] --root-directory PATH --arena-directory PATH

    required arguments:
      --root-directory      Session root directory path.
      --arena-directory     Arena session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.


``vcl-assign``
``vcl-assign`` is the command-line interface for assigning vocalizations to specific animals using Vocalocator.

.. code-block:: text

    usage: vcl-assign        [-h] --root-directory PATH [--vcl-version {vcl,vcl-ssl}]
                             [--env-name TEXT] [--model-dir PATH]
                             [--temperature FLOAT]
                             [--grid-resolution INTEGER INTEGER]
                             [--n-angle-bins INTEGER] [--n-samples INTEGER]
                             [--confidence-level FLOAT] [--angle-pdf-seed INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --vcl-version         Version of Vocalocator to use ('vcl' or 'vcl-ssl').
      --env-name            Name of the Vocalocator conda environment.
      --model-dir           Directory of the Vocalocator model.
      --temperature         Covariance temperature scaling for the 6D confidence sets.
      --grid-resolution     Spatial (x_res, y_res) grid the confidence-set PDF is sampled on, e.g. --grid-resolution 100 100.
      --n-angle-bins        Number of angular histogram bin edges (-pi..pi) for the confidence-set PDF.
      --n-samples           Monte-Carlo sample count for the per-vocalization angle PDF.
      --confidence-level    Confidence level (0..1) for the extracted confidence sets.
      --angle-pdf-seed      RNG seed for the reproducible angle-PDF sampling.

.. _usv-pipeline-cli:

These commands form the in-house, self-contained pipeline that turns segmented USVs into spectrograms, call masks, interpretable acoustic features, and toroidal QLVM latents — and that trains the two models the pipeline relies on (the Segment Anything Model 2 (SAM2) box-prompt **You Only Look Once (YOLO) object detector** and the **QLVM decoder**). Every step is a `click <https://click.palletsprojects.com>`_ CLI whose full option set lives under its block in */usv-playpen/_parameter_settings/processing_settings.json*; the flags below are the common overrides. The per-session steps read/write each session's ``audio/spectrograms/<session>_spectrograms.h5``; the cross-session steps aggregate a list of those files.

Inference flow (per session): ``generate-usv-spectrograms`` → ``generate-usv-masks`` → ``generate-usv-acoustic-features`` and/or ``infer-qlvm-latents``. Training flow (cross-session, run once on a cohort): build a dataset, then train. The YOLO detector and QLVM decoder are trained in-house; **SAM2 is used pretrained** (it is not fine-tuned here).

``generate-usv-spectrograms``
``generate-usv-spectrograms`` computes the variance-weighted, multi-channel average spectrogram of every USV in a session and writes the consolidated ``spectrogram/<session>`` group (``spectrograms`` (N, 128, 128), ``durations`` (N,)) into ``audio/spectrograms/<session>_spectrograms.h5``. Rows are 1:1 with ``usv_summary.csv``.

.. code-block:: text

    usage: generate-usv-spectrograms [-h] --root-directory PATH
                            [--num-freq-bins INTEGER] [--num-time-bins INTEGER]
                            [--nperseg INTEGER] [--min-freq FLOAT] [--max-freq FLOAT]
                            [--noverlap INTEGER] [--hop-length INTEGER]
                            [--window TEXT] [--offset FLOAT]
                            [--normalize | --no-normalize]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --num-freq-bins       Number of spectrogram frequency bins.
      --num-time-bins       Number of spectrogram time bins.
      --nperseg             STFT window length (n_fft).
      --min-freq            Lower frequency cutoff (Hz).
      --max-freq            Upper frequency cutoff (Hz).
      --noverlap            STFT segment overlap in samples (nperseg - hop_length); legacy scipy-style overlap kept in the block for parity with the QLVM training config.
      --hop-length          STFT hop length in samples (frames advance); falls back to nperseg // 4 when unset.
      --window              STFT window function name passed to librosa.stft (e.g. blackmanharris).
      --offset              Symmetric padding in seconds added to each USV before/after its start/stop bounds when slicing the audio.
      --normalize / --no-normalize
                            Whether to min-max normalize each averaged spectrogram to [0, 1].

``generate-usv-masks``
``generate-usv-masks`` segments each USV's calls with the box-prompt detector → SAM2 path (default ``detector=yolo``; ``cc`` connected-component fallback) and writes the instance masks back into the SAME spectrogram H5 under a ``mask/<session>`` group (``segmentations`` (M, 128, 128) bool, ``spectrogram_index`` (M,) int). Requires a pretrained SAM2 checkpoint and trained YOLO weights configured in settings (a missing path raises a clear error). GPU recommended.

.. code-block:: text

    usage: generate-usv-masks [-h] --root-directory PATH
                            [--detector {yolo,cc}]
                            [--sam2-model-dir TEXT] [--sam2-model-cfg TEXT]
                            [--sam2-model-path TEXT] [--yolo-weights TEXT]
                            [--yolo-conf FLOAT] [--yolo-iou FLOAT]
                            [--method TEXT] [--yolo-imgsz INTEGER]
                            [--deterministic | --no-deterministic]
                            [--mask-cmap TEXT] [--duration-min INTEGER]
                            [--batch-size INTEGER]
                            [--multimask-output | --no-multimask-output]
                            [--iou-floor FLOAT]
                            [--drop-below-iou | --no-drop-below-iou]
                            [--split-disconnected | --no-split-disconnected]
                            [--max-iters INTEGER]
                            [--merge-instances | --no-merge-instances]
                            [--merge-iou FLOAT] [--merge-containment FLOAT]
                            [--mask-intensity-floor FLOAT]
                            [--tiny-mask-floor-px INTEGER] [--min-box-area INTEGER]

    required arguments:
      --root-directory            Session root directory path.

    optional arguments:
      -h, --help                  Show this help message and exit.
      --detector                  Box detector backend (yolo learned detector or cc baseline).
      --sam2-model-dir            SAM2 model directory (config/checkpoint resolve against it).
      --sam2-model-cfg            SAM2 model config name/path (resolvable from sam2_model_dir).
      --sam2-model-path           SAM2 checkpoint path.
      --yolo-weights              Trained YOLO best.pt weights path.
      --yolo-conf                 YOLO confidence threshold (lower => more recall).
      --yolo-iou                  YOLO NMS IoU (raise to keep stacked calls).
      --method                    Mask-generation method; only 'boxprompt' (SAM2 box-prompt path) is supported.
      --yolo-imgsz                YOLO detector input image size in px (native spectrogram size is 128).
      --deterministic / --no-deterministic
                                  Disable cuDNN's autotuner so masks reproduce across
                                  processes (default: enabled). With it on, algorithm
                                  selection depends on the GPU state at process start and
                                  borderline faint calls fall on either side of the
                                  detector's confidence gate.
      --mask-cmap                 Matplotlib colormap used to render each spectrogram to RGB before SAM2 prompting.
      --duration-min              Minimum USV duration (time bins) to segment; shorter/placeholder (duration==0) rows are skipped.
      --batch-size                Number of spectrograms processed per batch before a memory-cleanup pass.
      --multimask-output / --no-multimask-output
                                  Let SAM2 emit multiple candidate masks per box and keep the highest-IoU one (vs a single mask).
      --iou-floor                 Predicted-IoU threshold below which a mask is flagged low-IoU (see --drop-below-iou).
      --drop-below-iou / --no-drop-below-iou
                                  Discard masks whose SAM2 predicted IoU is below --iou-floor (default keeps them).
      --split-disconnected / --no-split-disconnected
                                  Split a SAM2 mask with multiple 8-connected components into separate instances.
      --max-iters                 Residual re-prompting passes for the cc detector (the yolo detector is always single-pass).
      --merge-instances / --no-merge-instances
                                  Post-merge near-duplicate / contained instances to correct over-segmentation.
      --merge-iou                 IoU above which two overlapping instances are fused in the post-merge step.
      --merge-containment         Containment fraction above which a smaller instance is merged into a larger enclosing one.
      --mask-intensity-floor      Normalized-intensity floor; keep only mask pixels at/above it (drops faint harmonics/tails). 0 disables.
      --tiny-mask-floor-px        Minimum mask area in px; smaller masks / split components are dropped.
      --min-box-area              Minimum detector box area in px before SAM2 prompting; 0 disables the gate.

``generate-usv-acoustic-features``
``generate-usv-acoustic-features`` computes interpretable per-USV spectral/amplitude features and merges them into the session's ``usv_summary.csv``. When a ``mask/<session>`` group is present it restricts each feature to the true SAM mask region (``np.any`` union of the call's segmentations); otherwise it falls back to the signal time-window.

.. code-block:: text

    usage: generate-usv-acoustic-features [-h] --root-directory PATH
                            [--low-energy-frac FLOAT] [--high-energy-frac FLOAT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --low-energy-frac     Lower edge of the bandwidth energy band.
      --high-energy-frac    Upper edge of the bandwidth energy band.

``consolidate-spectrogram-store``
``consolidate-spectrogram-store`` merges per-session spectrogram/mask H5 files and the QLVM v3 columns of their USV summaries into one multi-session store under ``spectrograms_root``: the shared ``frequency_bins`` axis, per-session ``spectrogram/<session>`` and ``mask/<session>`` groups (plus ``spectrogram/<session>/qlvm_dim``, the regular model's ``qlvm1``/``qlvm2``), per-session ``qlvm/<session>/`` coordinates of all five production models (NaN = not embedded), int16 labels (0 = no label) and an int8 ``status`` (0 embedded, 1 too long, 2 no SAM mask, 3 both), a ``qlvm_models/<prefix>/`` group of provenance attrs and cluster tables / label grids per model, and a ``sessions`` table (session type, H5 and summary SHA-256, row and embedded counts). ``--package-corpus`` takes exactly the package's corpus sessions (its ``corpus/SESSION_H5_BASELINE.tsv``); every session's H5 must match the baseline's SHA-256 and its summary must carry the QLVM columns, or nothing is written. The store is written atomically as ``spectrograms_qlvmv3_<S>sessions_<N>vocalizations_<timestamp>.h5``; consumers resolve the newest ``spectrograms_*.h5`` (this or an older ``spectrograms_sam2masks_*`` store) automatically. The full layout is in *Process* (*The consolidated store*).

.. code-block:: text

    usage: consolidate-spectrogram-store [-h] (--root-directories TEXT | --package-corpus)
                                         [--package-root PATH] [--spectrograms-root PATH]

    required arguments (exactly one):
      --root-directories    Comma-separated string of session root directory paths, in store order.
      --package-corpus      Consolidate exactly the QLVM model package's corpus sessions, in the order of its SESSION_H5_BASELINE.tsv.

    optional arguments:
      -h, --help            Show this help message and exit.
      --package-root        QLVM model package root whose models' columns and tables the store carries; defaults to the v3 package (qlvm_models_latest/v3).
      --spectrograms-root   Output directory the consolidated store is written to.

``build-squeak-spectrogram-store``
``build-squeak-spectrogram-store`` writes the squeak spectrogram store the embedding explorer's *Squeaks* map reads: for every row with ``squeak`` true (pure squeaks and segments holding both) that is not noise, of every given session (so the explorer's squeak-class filter needs no rebuild), a 2-125 kHz spectrogram on log-spaced frequency bins, rebuilt from the unfiltered ``audio/hpss`` wavs over the segment widened to hold its squeak envelope (a squeak often runs past its segment) with the sonic front end's STFT (Blackman-Harris, ``nperseg`` 2048, hop 512, centred; metadata-excluded channels dropped, variance-weighted channel average, absolute dB), placed in a fixed window of ``--window-frames`` frames (a longer audio window keeps the display window centred on its squeak) and quantized to uint8 over ``[--db-floor, --db-ceil]``. Sessions come from ``--root-directories`` and/or ``--session-lists`` and are processed in ``--n-workers`` parallel processes; a failing session is reported and left out. The store is written atomically to ``spectrograms_root`` as ``squeak_spectrograms_<F>logbins_<S>sessions_<N>squeaks_<timestamp>.h5``; the explorer resolves the newest one automatically. The full layout is in *Process* (*The squeak spectrogram store*).

.. code-block:: text

    usage: build-squeak-spectrogram-store [-h] [--root-directories TEXT,TEXT,...] [--session-lists TEXT,TEXT,...]
                                          [--spectrograms-root PATH]
                                          [--min-freq FLOAT] [--max-freq FLOAT]
                                          [--n-frequency-bins INT] [--window-frames INT]
                                          [--db-floor FLOAT] [--db-ceil FLOAT]
                                          [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                                          [--n-workers INT]

    required arguments (at least one):
      --root-directories    Comma-separated string of session root directory paths.
      --session-lists       Comma-separated string of session-list .txt files (one session root per line).

    optional arguments:
      -h, --help            Show this help message and exit.
      --spectrograms-root   Output directory the store is written to (the spectrograms_root setting when not given).
      --min-freq            Lower edge (Hz) of the lowest log-frequency bin.
      --max-freq            Upper edge (Hz) of the highest log-frequency bin (at most 125000, the Nyquist frequency).
      --n-frequency-bins    Number of log-spaced frequency bins.
      --window-frames       Width (2.048 ms frames) of the fixed window every call is stored in.
      --db-floor            Absolute dB mapped to uint8 code 0 (values below are clipped).
      --db-ceil             Absolute dB mapped to uint8 code 255 (values above are clipped).
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                            Drop channels the session metadata marks as excluded from the spectrogram average
                            (keep it equal to the detect-usv-squeaks run).
      --n-workers           Sessions processed in parallel worker processes (1: in this process).

``build-qlvm-training-set``
``build-qlvm-training-set`` builds a QLVM training set of USV spectrograms (``train_data.npz`` + ``val_data.npz``, plus ``full_data.npz`` with ``--full-dataset``, and ``metadata.npz``) from a list of session root directories, drawn the way the v3 model packages' sets were (the port of the reference builders ``build_masked_usvs.py`` / ``build_unmasked_usvs_floor.py``, MMMmB). Each session's type (``MF``, ``FF``, ``MM``, ``lone_male``, ...) is read from the subjects' sexes in its metadata YAML; ``--session-type-targets`` gives each type's row budget (``null``: take the type whole; unlisted types are left out). A row is eligible when its duration is below ``--length-threshold``, it has a SAM mask instance (``--require-mask``) and it is an ultrasonic call alone: with ``--exclude-squeaks`` (the default) rows whose USV summary ``squeak`` is true are left out, so only pure USVs (``usv`` true, ``squeak`` false) are drawn (``--strict-squeak-exclusion`` takes the squeak rows instead from the reference squeak index named by ``--reference-squeak-index-path``, by the strict rule ``is_bbv`` or ``n_bouts_min3`` >= 1 by which the reference sets left broadband calls out; off by default, and it reproduces the shipped sets together with ``--no-exclude-noise``), with ``--exclude-noise`` rows flagged as noise are left out, and rows listed in ``--row-exclusion-table`` (a CSV of ``session_id`` + spectrogram-H5 ``row`` for exclusions kept outside the summaries, added on top of the others; the table's path, SHA-256 and the rows it added go into ``metadata.npz``) are left out. A type's budget is split into per-session quotas by capped-even water-filling and drawn per session, either uniformly over the eligible rows (``--draw-mode natural``) or equally across the mask-count strata of ``--mask-count-bin-edges`` (``uniform``, quotas allocated against each session's exactly-uniform headroom). Whole sessions are held out for validation per type. The drawn spectrograms (and their binarized SAM mask unions) are resized to ``--target-shape``, and then masked (``--apply-mask``, the phase 9 sets) or kept unmasked, optionally with a loudness floor baked in (``--floor``, the phase 6 sets). Every split also carries the rows' ``mean_freq_hz``, ``freq_bandwidth_hz``, ``loudness_db`` and ``spectral_entropy`` from the USV summary (NaN where missing), the raw values a conditional ``train-qlvm`` run conditions on. The defaults rebuild the production phase 6 set (``natural_5strata_N29000_unmasked_floor``). Every session's spectrogram H5 is fingerprinted as it is read: its SHA-256, row count and the number of its rows that entered the set go into ``metadata.npz`` and into two sidecars, ``SESSION_H5.sha256`` (checkable with ``sha256sum -c``) and ``SESSION_H5.tsv``. ``spec_id`` is a row number in that H5, so a consumer should compare the hash before joining on it: a rebuilt H5 renumbers its rows.

.. code-block:: text

    usage: build-qlvm-training-set [-h] --root-directories TEXT,TEXT,... --output-directory PATH
                            [--session-type-targets JSON]
                            [--draw-mode {natural,uniform}]
                            [--mask-count-bin-edges INT,INT,...]
                            [--length-threshold FLOAT]
                            [--require-mask | --no-require-mask]
                            [--exclude-squeaks | --no-exclude-squeaks]
                            [--strict-squeak-exclusion | --no-strict-squeak-exclusion]
                            [--reference-squeak-index-path PATH]
                            [--exclude-noise | --no-exclude-noise]
                            [--row-exclusion-table PATH|none]
                            [--masking-type {sam,none}]
                            [--apply-mask | --no-apply-mask]
                            [--floor FLOAT|none]
                            [--validation-split FLOAT] [--random-state INTEGER]
                            [--full-dataset | --no-full-dataset]
                            [--target-shape INTEGER INTEGER]
                            [--time-stretch | --no-time-stretch]

    required arguments:
      --root-directories              Comma-separated string of session root directory paths (the order sessions are drawn in).
      --output-directory              Directory to write the .npz training set.

    optional arguments:
      -h, --help                      Show this help message and exit.
      --session-type-targets          JSON object of session type -> rows (null takes the type whole), e.g. '{"MF": 29000, "FF": 29000, "MM": null, "lone_male": null}'; sessions of unlisted types are left out.
      --draw-mode                     Within-session draw: uniform over rows ("natural") or equal across mask-count strata ("uniform").
      --mask-count-bin-edges          Comma-separated inclusive lower bounds of the mask-count strata above 0, the last open-ended, e.g. 1,2,3,4,5 (strata 0/1/2/3/4/5+).
      --length-threshold              Drop spectrograms with duration >= threshold (time bins).
      --require-mask / --no-require-mask
                                      Leave out calls without a SAM mask instance.
      --exclude-squeaks / --no-exclude-squeaks
                                      Leave out rows whose USV summary squeak flag is true (pure squeaks and squeak + USV segments; detect-usv-squeaks), so the set holds pure USVs (usv true, squeak false) only.
      --strict-squeak-exclusion / --no-strict-squeak-exclusion
                                      With --exclude-squeaks, take the squeak rows from the reference squeak index (--reference-squeak-index-path) by its strict rule (is_bbv or n_bouts_min3 >= 1) instead of the summary squeak flag, to reproduce the shipped sets.
      --reference-squeak-index-path   The reference squeak index CSV (session_id, seg_index, is_bbv, n_bouts_min3) read by --strict-squeak-exclusion.
      --exclude-noise / --no-exclude-noise
                                      Leave out rows the USV summary flags as noise (detect-usv-noise).
      --row-exclusion-table           CSV with columns session_id and row (spectrogram-H5 row) of further calls to leave out, on top of the summary flags (e.g. a hand-curated list of calls to drop), or "none".
      --masking-type                  Read SAM masks from the mask/<session> groups ("sam") or none ("none").
      --apply-mask / --no-apply-mask  Multiply the binarized SAM mask into the stored spectrograms (masked set) or keep them unmasked.
      --floor                         Loudness floor baked into unmasked spectrograms after a per-spectrogram min-max (e.g. 0.2), or "none".
      --validation-split              Fraction of each session type's rows held out (as whole sessions) for validation.
      --random-state                  Seed of the quota tie-breaks, the draw and the session split.
      --full-dataset / --no-full-dataset
                                      Take every eligible row (no draw) and also write full_data.npz.
      --target-shape                  Output spectrogram (freq, time) shape as two ints, e.g. --target-shape 128 128.
      --time-stretch / --no-time-stretch
                                      Time-warp the signal window instead of center-resizing.

``build-qlvm-squeak-training-set``
``build-qlvm-squeak-training-set`` builds a QLVM training set of squeak crops (the port of the reference builder ``build_bbv_dataset.py`` (MMMmB), which built the phase 3 squeak cells' sets) from a list of session root directories. The candidates are the rows ``detect-usv-squeaks`` marked ``squeak`` true -- pure squeaks and segments holding both -- (optionally not noise; noise rows carry empty booleans) with a squeak extent ``squeak_start`` / ``squeak_end``; a row with several squeaks is one candidate over their envelope (its one extent), as ``infer-qlvm-squeak-latents`` embeds it. Each is read over the segment widened to hold the envelope plus its context and cropped to the frames inside the envelope plus ``--context-frames`` either side, crops narrower than ``--min-trimmed-frames`` (or wider than 128 frames) are dropped, and the crop width sets its duration stratum (``--duration-bin-edges``). ``--per-session-bin-cap`` keeps at most that many squeaks of one session in one stratum (0: no cap), and ``--n-total`` squeaks are then drawn uniformly (``--draw-mode natural``) or equally across strata (``uniform``); ``--full-dataset`` takes every squeak instead. The sonic spectrograms are rebuilt from the session audio exactly as ``infer-qlvm-squeak-latents`` builds them, each crop is min-max normalized (``--crop-normalization per_crop``) and centred in a 128 x 128 frame. ``--crop-window full_length`` (default) crops from the whole segment, the rule ``infer-qlvm-squeak-latents`` embeds with; ``first_128_frames`` cuts every segment to its first 128 frames and drops squeaks still going at frame 127 (the phase 3 sets). Sessions are taken in sorted order. The defaults are those of the production ``natural_session_N11000`` cell's set, except the crop window and the channel exclusion.

.. code-block:: text

    usage: build-qlvm-squeak-training-set [-h] --root-directories TEXT,TEXT,... --output-directory PATH
                            [--draw-mode {natural,uniform}] [--n-total INTEGER]
                            [--per-session-bin-cap INTEGER]
                            [--duration-bin-edges INT,INT,...]
                            [--context-frames INTEGER] [--min-trimmed-frames INTEGER]
                            [--crop-window {full_length,first_128_frames}]
                            [--crop-normalization {per_crop,absolute}]
                            [--exclude-noise | --no-exclude-noise]
                            [--exclude-metadata-audio-channels | --no-exclude-metadata-audio-channels]
                            [--validation-split FLOAT] [--random-state INTEGER]
                            [--full-dataset | --no-full-dataset]
                            [--target-shape INTEGER INTEGER]

    required arguments:
      --root-directories              Comma-separated string of session root directory paths.
      --output-directory              Directory to write the .npz training set.

    optional arguments:
      -h, --help                      Show this help message and exit.
      --draw-mode                     Draw uniformly over squeaks ("natural") or equally across duration strata ("uniform").
      --n-total                       Squeaks to draw.
      --per-session-bin-cap           Most squeaks any one session may give any one duration stratum before the draw (0: no cap, the "lumped" sets).
      --duration-bin-edges            Comma-separated crop-width edges (frames) of the duration strata, e.g. 26,40,62.
      --context-frames                Frames added either side of the squeak extent.
      --min-trimmed-frames            Narrowest crop (frames) kept.
      --crop-window                   Crop from the full-length segment ("full_length") or from its first 128 frames, dropping right-censored squeaks ("first_128_frames", the phase 3 sets).
      --crop-normalization            Min-max each crop ("per_crop") or apply the fixed dB transform ("absolute").
      --exclude-noise / --no-exclude-noise
                                      Leave out squeak rows the USV summary flags as noise.
      --exclude-metadata-audio-channels / --no-exclude-metadata-audio-channels
                                      Drop channels the session metadata marks as excluded from the spectrogram average.
      --validation-split              Fraction of each session type's squeaks held out (as whole sessions) for validation.
      --random-state                  Seed of the cap, the draw and the session split.
      --full-dataset / --no-full-dataset
                                      Take every squeak that passes the gates (no cap, no draw) and also write full_data.npz.
      --target-shape                  Output spectrogram (freq, time) shape as two ints, e.g. --target-shape 128 128.

``train-qlvm``
``train-qlvm`` trains a QLVM decoder on a prebuilt ``.npz`` training set (``build-qlvm-training-set`` output or a QLVM model package's training set: ``train_data.npz`` or ``full_data.npz``, optionally ``val_data.npz``, and ``metadata.npz``) with the recipe the shipped v3 models were trained with (JAX; per-spectrogram min-max, masks applied when the set applies them, a Fibonacci training lattice with one random torus shift per batch, the QMC log-evidence loss, Adam at a constant learning rate, the last epoch's weights kept) and writes a model package cell: ``checkpoint.tar`` (torch zip, readable by ``infer-qlvm-latents`` without torch), ``config/training_contract.json``, ``config/run_config.json`` and ``metrics/val_diagnostics.npz``. The defaults are the phase 6 / 9 recipe; phase 3 (BBVs) is ``--decoder-head legacy --n-epochs 2500 --val-freq 80 --val-samples-per-mask-count 1600``. ``--conditional duration|mean_freq|bandwidth|loudness|spectral_entropy`` trains a phase 11 conditional decoder (one conditioning value per call appended to the decoder input; batches drawn one width-capped quantile bin at a time and decoded at their mean value, the step loss scaled by rows / batch size) and adds ``config/condition_bins.npz`` (bins, training range, decode grid) and a ``condition`` block to the training contract, which ``infer-qlvm-latents`` reads; the raw values come from the set's ``mean_freq_hz`` / ``freq_bandwidth_hz`` / ``loudness_db`` / ``spectral_entropy`` columns or from ``--condition-table`` (e.g. a package's ``corpus/cond_table.npz``). Spectral entropy is min-max scaled by the training split's own range (written to the contract as ``entropy_min`` / ``entropy_max``; values outside it are clamped) and decoded on the grid. The run refuses a set without ``metadata.npz``, and a ``train_data.npz`` or ``val_data.npz`` older than a ``full_data.npz`` in the same directory. GPU recommended (about 3.3 s per epoch on 49,411 calls on an RTX 4080 SUPER, plus about 25 s per validation); on a shared GPU set ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.

.. code-block:: text

    usage: train-qlvm [-h] --dataset-directory PATH --output-directory PATH
                            [--n-epochs INTEGER] [--decoder-head {relu,legacy}]
                            [--training-fib-m INTEGER] [--validation-fib-m INTEGER]
                            [--embedding-fib-m INTEGER] [--batch-size INTEGER]
                            [--learning-rate FLOAT] [--val-freq INTEGER]
                            [--val-samples-per-mask-count INTEGER] [--seed INTEGER]
                            [--conditional {none,duration,mean_freq,bandwidth,loudness,spectral_entropy}]
                            [--condition-n-bins INTEGER]
                            [--condition-bin-scheme {quantile,quantile_capped}]
                            [--condition-scale-loss-by-batch | --no-condition-scale-loss-by-batch]
                            [--condition-decode-grid-step FLOAT] [--condition-table TEXT]

    required arguments:
      --dataset-directory   Directory holding the .npz training set (train_data.npz or full_data.npz, val_data.npz, metadata.npz).
      --output-directory    Directory the model package cell (checkpoint.tar, config/, metrics/) is written to.

    optional arguments:
      -h, --help            Show this help message and exit.
      --n-epochs            Number of training epochs.
      --decoder-head        Decoder head: relu (ReLU between the two Linear layers) or legacy (none).
      --training-fib-m      Fibonacci index of the training lattice (fib(m) points).
      --validation-fib-m    Fibonacci index of the validation lattice (fib(m) points).
      --embedding-fib-m     Fibonacci index of the embedding lattice recorded in the training contract.
      --batch-size          Training batch size.
      --learning-rate       Constant Adam learning rate.
      --val-freq            Compute the validation loss every N epochs (and after the last).
      --val-samples-per-mask-count
                            Validation subset: at most this many spectrograms per masks_len value.
      --seed                Seed of the initialization, shuffling, lattice shifts and validation subset.
      --conditional         Train a decoder conditioned on this per-call value (the phase 11 recipe), or none for an unconditional decoder.
      --condition-n-bins    Quantile bins the training values are cut into (conditional runs).
      --condition-bin-scheme
                            quantile, or quantile_capped (bins wider than the widest inner bin split into equal-width pieces).
      --condition-scale-loss-by-batch / --no-condition-scale-loss-by-batch
                            Scale each step's loss by rows / batch size (conditional runs).
      --condition-decode-grid-step
                            Spacing of the decode grid written to condition_bins.npz.
      --condition-table     Per-call .npz (spec_id + mean_freq_hz / freq_bandwidth_hz / loudness_db or image_level_db / spectral_entropy) to read the raw values from instead of the splits' columns, or none.

``infer-qlvm-latents``
``infer-qlvm-latents`` embeds a session's spectrograms into the torus of every QLVM model package cell of the ``model_cells`` setting (column prefix → cell; one ``--model-cell PREFIX CELL`` per model replaces it) and merges, per model, the ``PREFIX1`` / ``PREFIX2`` float torus coordinates (null where the call has no coordinates) into ``usv_summary.csv``, in the canonical column order (``os_utils.USV_SUMMARY_COLUMN_ORDER``). By default no cluster label is written: the regular map's ``qlvm_category`` comes from ``assign-qlvm-categories``, run afterwards. ``--model-cell-labels PREFIX LEVELS`` (the ``model_cell_label_levels`` setting; repeat once per prefix) sets a prefix's comma-separated levels among ``fine`` and ``coarse`` (``fine`` → ``PREFIX_category``, ``coarse`` → ``PREFIX_supercategory``, with the ``qlvm`` names for prefix ``qlvm``; an empty string writes no label) to write the package's own cluster labels, read off the cell's ``label_grid.npy`` at the pixel of the coordinates (``1 … k``, 1 the largest cluster). No model column is written; the legacy ``qlvm_model`` column (written by the retired single-model run) and the earlier coordinate, label and category confidence (``PREFIX_category_agreement`` / ``PREFIX_category_uncertain``) columns of the listed prefixes are removed. A prefix must be an identifier given once. Pure squeaks (``squeak`` true and ``usv`` false, written by ``detect-usv-squeaks``) are skipped by every cell and get null coordinates and labels; segments holding both are embedded as usual; a summary without the two booleans stops the run (run ``detect-usv-noise`` and ``detect-usv-squeaks`` first). The production mapping (``os_utils.QLVM_PRODUCTION_MODEL_CELLS`` under ``os_utils.QLVM_MODEL_PACKAGE_ROOT``, ``/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean``) is ``qlvm`` (the regular cell ``cell/masked``), ``qlvm_dur`` (``conditionals/cell/duration``) and ``qlvm_ent`` (``conditionals/cell/spectral_entropy``), all trained on SAM-masked, time-stretched spectrograms, so ``masking_type`` ``sam`` and ``time_stretch`` true (the shipped defaults); when ``model_cells`` is empty and ``spectrograms_root`` is set, the run fills it with that mapping (and ``masking_type`` / ``time_stretch`` with ``sam`` / true), and an empty ``model_cells`` otherwise stops the run. The production cells are module constants rather than settings and are not derived from ``spectrograms_root``: the settings' experimenter paths are re-keyed to the active experimenter, which would move this path. The cells carry no cluster label grid and their folder no package baseline, so every session is inferred and no label level can be asked of them. Model package cells are the only models it embeds with: the single-model route (``model_cell_directory``, or a ``train-qlvm`` ``qmc_decoder_weights.npz`` with fine and coarse reference ``arrays.npz``) is retired.

Each cell supplies everything model-specific (``qlvm_models_latest/v2/<phase>/<cell>`` or ``qlvm_models_latest/v3/<phase>/<cell>``): ``checkpoint.tar`` (read without torch; legacy or ReLU head), ``training_contract.json`` (input min-max and loudness floor, duration window), its Fibonacci embedding lattice (46,368 points), and the fine and coarse ``label_grid.npy`` (``inference/clusters_<level>/`` in v3, ``cluster/<level>/`` in v2 / v2.1; in v3 the contract and bins sit in ``config/``, the per-call tables in ``inference/`` and the package's ``SESSION_H5_BASELINE.tsv`` in ``corpus/``, and both layouts are read). The settings are checked against every cell's contract before anything is embedded and the run stops on any disagreement; ``--masking-type`` must match it (``sam`` for phase 9 cells, ``none`` for phase 6, 10 and 11 cells). With ``--masking-type sam`` each call is prepared exactly as a masked training set fed its decoder: the spectrogram and the union of its SAM mask regions (``mask/<session>`` group) are resized (or time-stretched) separately, the resized mask is binarized at 0.5 and multiplied in (``build-qlvm-training-set --apply-mask``), and the result is min-maxed and multiplied by the binarized mask again (``train-qlvm``); ``--masking-type none`` embeds raw spectrograms. A cell that has not been clustered yet (a ``train-qlvm`` cell without ``inference/clusters_<level>/label_grid.npy``) embeds too, since by default no label level is asked of it; asking it for a level stops the run before anything is written. USVs with a duration at or above the contract's ``length_threshold`` get null columns. When the contract has ``require_mask`` true (its training set left out calls without a SAM mask, as in every v2 and v3 package cell) or the cell conditions on mean frequency or loudness, USVs without a mask instance also get null columns instead of being embedded unmasked. Whenever the decoder needs SAM masks (either case, or ``--masking-type sam``), a session H5 without a ``mask/<session>`` group stops the run. Conditional cells are decoded at one conditioning value per USV, computed from its own: the normalized duration, the mean frequency of its SAM-masked spectrogram (so the session H5 needs its ``mask/<session>`` group even though the decoder is fed unmasked spectrograms), and, for phase 11 (v3) cells, its ``freq_bandwidth_hz`` from the summary (written by ``generate-usv-acoustic-features``; a summary without the column stops the run) or its loudness, the summary's ``loudness_db`` (the masked image-level dB ``generate-usv-acoustic-features`` measures from the session's ``hpss_filtered`` audio; a summary without the column stops the run), and, for a cell ``train-qlvm`` trained on spectral entropy, the summary's ``spectral_entropy`` scaled by the contract's ``entropy_min`` / ``entropy_max`` and clamped to ``[0, 1]`` (a summary without the column stops the run). USVs without a value get null columns. Phase 10 cells replace the value with the frozen corpus bin mean of the cell's ``condition_bins.npz`` (the lattice is decoded once per distinct bin mean, up to 32 times per session). Phase 11 cells follow the contract's ``condition.decode``: ``exact`` (duration, bandwidth) decodes at the value itself clamped to the cell's training range, ``grid`` (mean frequency, loudness, spectral entropy) at the nearest point of its 0.0025-step decode grid, exactly as the package embedded its corpus; the lattice is then decoded once per distinct value, about 100-250 times on a ~900-USV session. The summary CSV is rewritten atomically.

For each model, a session of the package's corpus takes the package's own coordinates (``posterior_cache.npz`` joined on ``spec_id``) when the package's ``SESSION_H5_BASELINE.tsv`` lists it with the SHA-256 of its current spectrogram H5, the summary has as many rows as the H5, and the package's ``durations`` and ``mask_counts`` equal the H5's on every package row; every other session is inferred with the cell, and the log names the route and the reason per model. ``--no-prefer-package-values`` infers every session.

.. code-block:: text

    infer-qlvm-latents --root-directory /mnt/falkner/Bartul/Data/20251003_143416 \
        --model-cell qlvm     .../qlvm_time_stretch/masked_clean/cell/masked \
        --model-cell qlvm_dur .../qlvm_time_stretch/masked_clean/conditionals/cell/duration \
        --model-cell qlvm_ent .../qlvm_time_stretch/masked_clean/conditionals/cell/spectral_entropy

.. code-block:: text

    usage: infer-qlvm-latents [-h] --root-directory PATH
                            [--model-cell TEXT TEXT]...
                            [--model-cell-labels TEXT TEXT]...
                            [--prefer-package-values | --no-prefer-package-values]
                            [--latent-dim INTEGER]
                            [--target-shape INTEGER INTEGER]
                            [--time-stretch | --no-time-stretch]
                            [--masking-type {sam,none}]
                            [--length-threshold FLOAT]
                            [--lattice-batch-size INTEGER]
                            [--data-batch-size INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --model-cell          A column prefix and a QLVM model package cell (e.g. --model-cell qlvm_dur .../qlvm_time_stretch/masked_clean/conditionals/cell/duration); repeat once per model. When given, these pairs replace the model_cells setting: the session is placed on the torus of every listed cell and <prefix>1/<prefix>2 plus the cluster-label columns asked of each prefix (see --model-cell-labels; none by default) are written, and earlier coordinate, label and category columns of each listed prefix are removed. Without it, the model_cells setting is used (by default the production cells).
      --model-cell-labels   A model_cells column prefix and the comma-separated cluster-label levels it writes, among fine and coarse (e.g. --model-cell-labels qlvm_dur fine,coarse writes qlvm_dur_category and qlvm_dur_supercategory; an empty string writes none); repeat once per prefix. When given, these pairs replace the model_cell_label_levels setting; prefixes not listed keep the default (no label column: the coordinates only; qlvm_category of the regular map comes from assign-qlvm-categories).
      --prefer-package-values / --no-prefer-package-values
                            With model cells: take a corpus session's coordinates from the package's own embedding when its spectrogram H5 is unchanged since the package (SHA-256, row count, durations and mask counts verified), else infer them; --no-prefer-package-values infers every session.
      --latent-dim          Dimensionality of the toroidal latent space; must equal every model cell's training contract.
      --target-shape        Output spectrogram (freq, time) shape as two ints, matching the training preprocessing, e.g. --target-shape 128 128.
      --time-stretch / --no-time-stretch
                            Whether to time-stretch each spectrogram to the fixed size (matching training preprocessing; true for the production cells) instead of a plain resize; must match every cell's training contract.
      --masking-type        Apply SAM mask regions as a masked training set did ("sam", the default; the production cells, phase 9 cells and masked train-qlvm cells) or embed raw spectrograms ("none", the v3 phase 6 and 11 cells); must match every cell's training contract. With "sam", a session without a mask group raises.
      --length-threshold    Embed only USVs with duration below this (time bins); when set, must equal every model cell's training contract. Unset in the settings (null), each cell's contract sets it.
      --lattice-batch-size  Lattice points decoded and scored per block; lower it to cut memory on large lattices.
      --data-batch-size     Spectrograms whose lattice posteriors are computed together; memory grows with this times the lattice size.

``export-qlvm-reference-arrays``
``export-qlvm-reference-arrays`` writes one QLVM model package cell's clustering as ``arrays_fine.npz`` and ``arrays_coarse.npz``, the reference-arrays layout the QLVM visualizations read (``qlvm-torus-traversal-video``, the sequence embedding map, the manifold atlas). Each file holds the cell's ``label_grid.npy`` as ``ws_labels_periodic`` and ``ws_labels``, the ``clusters.csv`` peaks as ``centers`` (row ``i`` for label ``i + 1``), the corpus calls' torus coordinates and labels as ``latent_coords`` / ``sample_ws`` / ``sample_ws_periodic``, a ``heatmap`` of the aggregated posterior over the cell's embedding lattice (summing to the number of corpus calls; not the reference arrays' smoothing), and ``model_id``. Write each production cell to its map's folder, ``<spectrograms_dir>/qlvm_v3/<map>`` (``qlvm`` for the phase 6 regular cell ``phase6_USVs_unmasked_floor/natural_5strata_N29000_unmasked_floor``; ``qlvm_dur``, ``qlvm_mf``, ``qlvm_bw``, ``qlvm_loud`` for the phase 11 cells), the folders the QLVM visualizations read for ``shared_resources.qlvm_map`` (``os_utils.QLVM_REFERENCE_ARRAYS_DIRECTORY_NAME``, ``os_utils.QLVM_MAPS``); write any other cell under a separate ``spectrograms_dir`` so the models never overwrite each other.

.. code-block:: text

    usage: export-qlvm-reference-arrays [-h] --model-cell-directory PATH --output-directory PATH

    required arguments:
      --model-cell-directory
                            A QLVM model package cell, e.g. .../qlvm_models_latest/v2/phase9_USVs_masked_relu/natural_3strata_N65000_masked.
      --output-directory    Directory to write arrays_fine.npz and arrays_coarse.npz into (created if missing), e.g. <spectrograms_dir>/qlvm_v3/<map> (the folder the QLVM visualizations read for that map; qlvm for the production regular cell).

    optional arguments:
      -h, --help            Show this help message and exit.

``build-qlvm-categories``
``build-qlvm-categories`` builds the content-ridge categories of a QLVM torus from a corpus of embedded calls and writes them as a category directory. Its corpus is the rows of ``--properties-file`` (a ``.npz`` such as a ``build-qlvm-training-set`` split, or a ``.parquet`` / ``.csv``, with ``spec_id``, ``session_id`` and the ``properties`` columns, by default ``mask_count``, ``durations``, ``freq_bandwidth_hz``, ``spectral_entropy``, ``mean_freq_hz`` and ``loudness_db``), each placed by its ``x`` / ``y`` in ``--positions-file`` (joined on ``spec_id``). Every property becomes its rank over all calls; each rank's Gaussian kernel average (``field_sigma`` 8 px) on a periodic ``grid_resolution`` 200 x 200 grid is a content field; the content change is the root sum of squares of the fields' periodic central differences at ``change_span`` 8 px over x, y and the properties; a marker watershed of it on a 3 x 3 tiling (markers: local minima over an 11 px window after a 1.5 px blur) gives basins whose boundaries run along its ridges; neighbouring basins are joined, a region under ``size_floor`` (5 % of the calls) first (smallest first, to the neighbour across its lowest mean ridge), then the pair across the lowest mean ridge, down to ``n_categories`` (4). ``n_bootstraps`` (200) session resamples (seed ``bootstrap_seed`` + i; calls weighted by their session's draw count) rebuild the categories, each is matched to the full-data categories by pixel overlap (Hungarian assignment), and every pixel takes the majority category with its ``agreement`` (the winning share of the votes). Categories are ``R-1`` ... ``R-k`` by call count, described by ``category_descriptions``. Written: ``category_grids.npz`` (``label_grid``, ``agreement``, ``reference_grid``, ``content_change``, ``basins``, ``density``), ``category_nomenclature.json`` (names, descriptions, counts, label positions, ``uncertain_agreement``), ``category_call_labels.csv`` (each corpus call's category, agreement and ``uncertain`` flag: agreement below ``uncertain_agreement``, 0.6) and ``build_config.json``. On the masked time-stretched map's 61,378-call set and its embedding, the defaults reproduce the four consensus categories (simple, mixed, complex, wide-band two-mask) pixel for pixel, with every call's category, agreement and uncertainty flag.

.. code-block:: text

    usage: build-qlvm-categories [-h] --positions-file PATH --properties-file PATH --output-directory PATH
                                 [--properties TEXT] [--grid-resolution INTEGER] [--field-sigma FLOAT]
                                 [--change-span INTEGER] [--marker-sigma FLOAT] [--marker-distance INTEGER]
                                 [--size-floor FLOAT] [--n-categories INTEGER] [--n-bootstraps INTEGER]
                                 [--bootstrap-seed INTEGER] [--uncertain-agreement FLOAT]
                                 [--category-descriptions TEXT] [--n-jobs INTEGER]

    required arguments:
      --positions-file      Per-call torus positions (.npz / .parquet / .csv with spec_id, x, y in [0, 1)).
      --properties-file     Per-call properties (.npz / .parquet / .csv with spec_id, session_id and every property column); its rows are the corpus.
      --output-directory    Directory to write the category files into.

    optional arguments:
      -h, --help            Show this help message and exit.
      --properties          Comma-separated property columns the categories are built on.
      --grid-resolution     Pixels per side of the periodic grid.
      --field-sigma         Gaussian smoothing of the property fields, in pixels.
      --change-span         Offset of the central differences of the content-change field, in pixels.
      --marker-sigma        Blur of the content-change field before the watershed, in pixels.
      --marker-distance     Half-width of the window a watershed marker is the minimum of, in pixels.
      --size-floor          Smallest share of the calls a category may hold.
      --n-categories        Number of categories.
      --n-bootstraps        Session resamples of the consensus vote.
      --bootstrap-seed      Seed of the first session resample (resample i uses seed + i).
      --uncertain-agreement A call whose pixel agreement is below this is flagged uncertain.
      --category-descriptions
                            Comma-separated short description of each category, R-1 first.
      --n-jobs              Parallel workers of the session resamples.

``assign-qlvm-categories``
``assign-qlvm-categories`` labels a session's USVs with a ``build-qlvm-categories`` directory (``--category-directory``), from their torus coordinates ``P1`` / ``P2`` in its ``usv_summary.csv`` (``P`` = ``--coordinate-prefix``, the ``infer-qlvm-latents --model-cell`` prefix of the map the categories were built on): each call takes the category of the grid pixel under its position (the QLVM pixel rule). It writes ``P_category`` only (``1 … k``, ``R-k`` in the directory's ``category_nomenclature.json``), null for calls without coordinates, replacing an earlier column of that name; the summary is rewritten atomically, in the canonical column order. The per-call agreement (the pixel's share of the consensus votes) and the uncertain flag (agreement below the directory's ``uncertain_agreement``) are not written to the summary -- the log counts the uncertain calls, and both stay computable from the directory and the coordinates (``qlvm_categories.assign_categories``) -- and ``P_category_agreement`` / ``P_category_uncertain`` columns an earlier version wrote are removed. Run it after ``infer-qlvm-latents``, which drops ``P_category`` (and those confidence columns) of every prefix it embeds.

.. code-block:: text

    usage: assign-qlvm-categories [-h] --root-directory PATH [--category-directory PATH] [--coordinate-prefix TEXT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --category-directory  A build-qlvm-categories output directory.
      --coordinate-prefix   Prefix P of the summary columns P1 / P2 holding the torus coordinates of the map the categories were built on.

``tidy-usv-summary-columns``
``tidy-usv-summary-columns`` migrates existing ``usv_summary.csv`` files to the canonical column layout (``os_utils.USV_SUMMARY_COLUMN_ORDER``, see *Process*): it drops the obsolete columns (``os_utils.USV_SUMMARY_OBSOLETE_COLUMNS``: ``qlvm_supercategory``, ``qlvm_dur_category``, ``qlvm_dur_supercategory``, ``qlvm_mf1`` / ``qlvm_mf2`` / ``qlvm_mf_category`` / ``qlvm_mf_supercategory`` and the same for ``qlvm_bw`` and ``qlvm_loud``, the legacy ``qlvm_model``, the retired squeak-detector columns ``squeak_probability``, ``squeak_frame_runs``, ``call_class``, ``squeak_spans`` and ``n_squeaks``, and every ``*_category_agreement`` / ``*_category_uncertain`` column) and puts the rest in canonical order, any column the order does not list kept last in its existing order. No column is created and no value changes: every column is read and written back as text. A summary already in the canonical layout is not rewritten. ``--dry-run`` only reports, per session, the columns it would drop and the resulting order. ``--backup-directory`` copies each original summary to ``<backup-directory>/<session>/<file name>`` before rewriting it and refuses (that session fails) when a backup is already there. Every session is attempted; failed sessions are listed at the end and make the command exit with an error.

.. code-block:: text

    usage: tidy-usv-summary-columns [-h] [--root-directory PATH ...] [--sessions-file PATH] [--dry-run] [--backup-directory PATH]

    optional arguments:
      -h, --help            Show this help message and exit.
      --root-directory      Session root directory path; repeat once per session.
      --sessions-file       Text file with one session root directory per line (blank lines and # lines skipped).
      --dry-run             Only report what would change; nothing is written.
      --backup-directory    Copy each original summary to <backup-directory>/<session>/ before rewriting it.

``export-yolo-dataset``
``export-yolo-dataset`` renders USV spectrograms to images (exactly as the detector renders them at inference) and writes an Ultralytics-format YOLO dataset (``images/{train,val}``, ``labels/{train,val}``, ``data.yaml``). ``--label-source cc`` (default) pseudo-labels boxes with the unlearned connected-component detector (no annotation needed); ``manual`` ingests hand-verified ``{spec_id}.txt`` labels; ``merge`` uses cc overridden by manual where present.

.. code-block:: text

    usage: export-yolo-dataset [-h] --root-directories TEXT,TEXT,... --output-directory PATH
                            [--label-source {cc,manual,merge}]
                            [--validation-split FLOAT] [--random-state INTEGER]
                            [--colormap TEXT]
                            [--manual-labels-directory TEXT]

    required arguments:
      --root-directories    Comma-separated string of session root directory paths.
      --output-directory    Directory to write the YOLO dataset.

    optional arguments:
      -h, --help            Show this help message and exit.
      --label-source        Box label source: cc pseudo-labels, manual files, or merge.
      --validation-split    Fraction of images held out for validation.
      --random-state        Random seed (RNG seed) for the reproducible train/val split permutation.
      --colormap            Matplotlib colormap name the spectrogram images are rendered with (must match the detector colormap).
      --manual-labels-directory
                            Directory of hand-verified {spec_id}.txt YOLO labels (manual/merge).

``train-masks``
``train-masks`` fine-tunes the Ultralytics YOLO box detector on an ``export-yolo-dataset`` dataset (from a COCO-pretrained ``yolo11n.pt`` by default) and copies the resulting ``best.pt`` to ``<output-directory>/best.pt`` — the path to set as ``generate-usv-masks``' ``yolo_weights``. GPU recommended.

.. code-block:: text

    usage: train-masks [-h] --dataset-directory PATH --output-directory PATH
                            [--base-weights TEXT] [--n-epochs INTEGER]
                            [--imgsz INTEGER] [--batch-size INTEGER]
                            [--device TEXT] [--run-name TEXT]

    required arguments:
      --dataset-directory   YOLO dataset directory (export-yolo-dataset output, with data.yaml).
      --output-directory    Directory for the Ultralytics run + copied best.pt.

    optional arguments:
      -h, --help            Show this help message and exit.
      --base-weights        Base YOLO checkpoint to fine-tune from (e.g. yolo11n.pt).
      --n-epochs            Number of training epochs.
      --imgsz               Square image size (px) the detector trains at; 128 is the native spectrogram size.
      --batch-size          Training batch size (imgs/batch).
      --device              Compute device: a GPU index (e.g. "0"), "cpu", or omit for Ultralytics auto-select (null).
      --run-name            Ultralytics run name (subdir under the output directory holding the run artifacts).

Analyze
-------

``generate-beh-features``
``generate-beh-features`` is the command-line interface for calculating 3D behavioral features.

.. code-block:: text

    usage: generate-beh-features  [-h] --root-directory PATH
                                  [--head-points TEXT TEXT TEXT TEXT]
                                  [--tail-points TEXT TEXT TEXT TEXT TEXT]
                                  [--back-root-points TEXT TEXT TEXT]
                                  [--derivative-bins TEXT...]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --head-points         Skeleton head nodes.
      --tail-points         Skeleton tail nodes.
      --back-root-points    Skeleton back nodes.
      --derivative-bins     Number of bins for derivative calculation.


``generate-usv-playback``
``generate-usv-playback`` is the command-line interface for generating artificial ultrasonic vocalization (USV) playback files.

.. code-block:: text

    usage: generate-usv-playback [-h] --exp-id TEXT [--num-usv-files INTEGER]
                                 [--total-usv-number INTEGER] [--ipi-duration FLOAT]
                                 [--wav-sampling-rate INTEGER]
                                 [--playback-snippets-dir TEXT]

    required arguments:
        --exp-id                     Experimenter ID.

    optional arguments:
        -h, --help                   Show this help message and exit.
        --num-usv-files              Number of WAV files to create.
        --total-usv-number           Total number of USVs to distribute across files.
        --ipi-duration               Inter-USV-interval duration (in s).
        --wav-sampling-rate          Sampling rate for the output WAV file (in kHz).
        --playback-snippets-dir      Directory of USV playback snippets.

``build-naturalistic-usv-repository``
``build-naturalistic-usv-repository`` is the command-line interface for building one naturalistic USV repository — the clean, reconstructed vocalizations that naturalistic playback replays (see the *Build the naturalistic USV repository* section of :doc:`Analyze` for the full explanation of every parameter).

.. code-block:: text

    usage: build-naturalistic-usv-repository [-h] [--session-list PATH]
                                             [--context-label {courtship_male,courtship_female,lone_male,lone_female,same_sex_male,same_sex_female,mixed}]
                                             [--ibi-z-score FLOAT] [--ibi-component-index INTEGER]
                                             [--min-vocalizations INTEGER] [--length-threshold INTEGER]
                                             [--min-duration INTEGER] [--mask-dilation INTEGER]
                                             [--feather-sigma-time FLOAT] [--fade-ms FLOAT]
                                             [--peak-normalize | --no-peak-normalize] [--peak-target-fraction FLOAT]

    optional arguments:
        -h, --help                                    Show this help message and exit.
        --session-list                                Text file of session root directories (one per line); repeatable. These lists are the sole selection of what enters the repository.
        --context-label                               Which (sex, social context) database to build; drives emitter handling, output subdirectory, and the filename token.
        --ibi-z-score                                 z-score for the bout-boundary threshold exp(mu + z*sd).
        --ibi-component-index                         Per-sex mixture component index used for the bout threshold.
        --min-vocalizations                           Minimum USVs for a bout to be kept.
        --length-threshold                            Drop a bout if any USV is longer than this (spectrogram time-bins).
        --min-duration                                Drop a bout if any USV is shorter than this (time-bins).
        --mask-dilation                               Grow the SAM mask by this many bins before inversion (0 = tight).
        --feather-sigma-time                          Gaussian sigma (time-bins) of the time-only mask feather.
        --fade-ms                                     Raised-cosine onset/offset fade length (ms).
        --peak-normalize/--no-peak-normalize          Peak-normalize each USV to a uniform level, or preserve relative amplitude.
        --peak-target-fraction                        Fraction of the int16 ceiling each peak-normalized snippet is scaled to.

The output directory is not a CLI option: it is ``naturalistic_usv_repository_dir`` under the ``data_roots`` block of *analyses_settings.json*, and each run writes ``<dir>/<sex>/naturalistic_usv_repository_<context>_<datestring>.h5``.

``generate-naturalistic-usv-playback``
``generate-naturalistic-usv-playback`` is the command-line interface for generating naturalistic USV playback files.

.. code-block:: text

    usage: generate-naturalistic-usv-playback [-h] --exp-id TEXT [--num-naturalistic-usv-files INTEGER]
                                              [--context-label {courtship_male,courtship_female,lone_male,lone_female,same_sex_male,same_sex_female,mixed}]
                                              [--total-playback-time INTEGER]
                                              [--complexity-enabled | --no-complexity-enabled]
                                              [--complexity-mask-threshold INTEGER]
                                              [--complexity-start-fraction FLOAT]
                                              [--complexity-end-fraction FLOAT]
                                              [--complexity-bandwidth FLOAT]
                                              [--edge-silence-seconds FLOAT]
                                              [--max-isi-seconds FLOAT]

    required arguments:
        --exp-id                                      Experimenter ID.

    optional arguments:
        -h, --help                                    Show this help message and exit.
        --num-naturalistic-usv-files                  Number of naturalistic playback files to be created.
        --context-label                               Which (sex, social context) repository to play back; the newest matching build is used.
        --total-playback-time                         Total acceptable duration of the playback file (in s).
        --complexity-enabled/--no-complexity-enabled  Steer bout draws toward a target call complexity (else uniform).
        --complexity-mask-threshold                   A USV is complex if its mask_number is >= this (default 2).
        --complexity-start-fraction                   Target complex-USV fraction at the START of the file (0-1).
        --complexity-end-fraction                     Target complex-USV fraction at the END of the file (0-1); differs from start = ramp.
        --complexity-bandwidth                        Gaussian bandwidth (complex-fraction units) for complexity steering; smaller = tighter to target.
        --edge-silence-seconds                        Fixed lead-in/lead-out silence at the start and end of the file (s).
        --max-isi-seconds                             Clip each inter-bout pause (ISI) to at most this many seconds.

The repository is not selected by an explicit file path: ``context_label`` picks the sex subdirectory + context, and the newest matching build in ``<naturalistic_usv_repository_dir>/<sex>/`` is used. ``playback_seed`` (for a reproducible stimulus) is the one parameter without a command-line flag — set it in the ``create_naturalistic_usv_playback_wav`` block of *analyses_settings.json*.

``generate-usv-interval-distributions``
``generate-usv-interval-distributions`` is the command-line interface for computing inter-USV interval distributions across one or more session-list text files and modelling them. It runs the same code as the ``inter_usv_interval_analyses.ipynb`` compute cell and writes the same archive.

**Pools and pairing.** The analysis works on the pools declared in ``interval_pools`` (JSON only): each names an emitter sex, a call type (``usv`` or ``squeak``), an adjacency rule (``filtered`` or ``strict``) and whether it is fitted. Each animal's sex is read from the ``Subjects`` block of the session's ``*_metadata.yaml``, not from its track slot, and a session whose metadata does not record an animal's sex raises. Intervals are measured only between two consecutive calls of the *same* animal, so a pool holds every animal of its sex in a session -- one in courtship, both in a female-female session -- without ever pairing the two. Each session-list text file contains one session root directory per line; paths are run through ``configure_path`` so Mac/Linux/Windows entries resolve correctly on the host platform. ``--session-list`` may be passed multiple times to merge multiple cohorts.

Both interval definitions are computed unconditionally on every run: ``s2s`` = ``start[i+1] - start[i]`` (literature standard), and ``e2s`` = ``start[i+1] - stop[i]`` (can be negative for overlapping calls and is dropped via the ``> 0`` filter, with the drop count reported per pool). Both definitions share the same per-session pass over the noise-filtered USV table.

**Models.** On every fitted pool with at least ``min_intervals_for_fitting`` intervals, three analyses run as the JSON enables them:

* ``fit_tied_model`` -- the tied-scale Student-t peak model (``tied_peak_grid`` peak counts, each with ``tied_n_background`` free background components) and its step-up peak-count test, every rung's likelihood ratio divided by its session design effect (``design_bootstrap`` session resamples) before it is scored against the parametric-bootstrap null. This is the model the male courtship analysis reports.
* ``fit_mixture_model`` -- the unconstrained mixture sweep (``n_components_min`` to ``n_components_max``, ``--model-class``) and its step-up LRT, session-corrected the same way. This is the model the female-female analysis reports.
* ``fit_serial_dependence`` -- median regressions of the next interval on the current one over all consecutive same-animal pairs: a spline and a bent line whose bend estimates the bout boundary from serial dependence alone, with session-bootstrap bands (``serial_dependence_*`` keys).

The options below override the JSON for the extraction and the unconstrained sweep; the pools, the tied model and the serial-dependence knobs are set in the JSON (see :doc:`Notebooks`, *Inter-USV interval analyses*, for every key and the archive layout).

**Output.** A single self-describing HDF5 archive ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5`` in ``--output-directory``. Per interval type it holds the tidy one-row-per-interval table (with each interval's ``emitter_id``), per-pool drop counts and descriptive summaries, and the tables of every analysis that ran; the root ``/attrs`` record every parameter that drove the run, the ``git_sha``, the list files (``source_lists``) and the session directories they resolved to (``session_roots``).

.. code-block:: text

    usage: generate-usv-interval-distributions [--session-list FILE...] [--output-directory DIRECTORY]
                            [--exclude-noise-usvs | --no-exclude-noise-usvs]
                            [--fit-mixture-model | --no-fit-mixture-model]
                            [--n-components-min INTEGER] [--n-components-max INTEGER]
                            [--n-repeats INTEGER] [--max-modes-reported INTEGER]
                            [--random-seed-base INTEGER]
                            [--cv-n-folds INTEGER] [--cv-n-init INTEGER]
                            [--mixture-model-n-init INTEGER] [--mixture-model-reg-covar FLOAT]
                            [--tau FLOAT] [--figures-directory DIRECTORY]
                            [--model-class {gauss,t,ig}]
                            [--bootstrap-lrt-B INTEGER]
                            [--bootstrap-lrt-n-subsample INTEGER]
                            [--bootstrap-lrt-alpha FLOAT]
                            [--bootstrap-lrt-n-init INTEGER]
                            [--bootstrap-lrt-n-jobs INTEGER]
                            [--bootstrap-lrt-bonferroni | --no-bootstrap-lrt-bonferroni]
                            [--help]

    optional arguments:
      --session-list              Path to a text file containing session root
                                  directories (one per line). Repeatable.
      --output-directory          Directory in which to write the consolidated
                                  usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5 archive.
      --exclude-noise-usvs / --no-exclude-noise-usvs
                                  Drop the USV segments ``detect-usv-noise``
                                  flagged as holding no vocalization (default:
                                  enabled). A session whose summary has no
                                  ``noise`` column raises rather than passing its
                                  detections through unfiltered; run
                                  ``detect-usv-noise`` on it, or pass
                                  ``--no-exclude-noise-usvs``.
      --fit-mixture-model / --no-fit-mixture-model
                                  Whether to run the unconstrained mixture sweep
                                  and its session-corrected LRT on the fitted pools.
      --n-components-min          Minimum number of mixture components. Default 2.
      --n-components-max          Maximum number of mixture components. Default 6.
      --n-repeats                 Number of EM-init repeats per (pool, n_components).
                                  Default 10.
      --max-modes-reported        Maximum number of mixture modes recorded per fit.
                                  Default 6.
      --random-seed-base          Base seed; rep r uses random_seed_base + r. It
                                  also seeds the subsamples, the bootstrap nulls
                                  and the session resamples. Default 0.
      --cv-n-folds                Number of K-fold splits for CV log-likelihood.
                                  Default 5.
      --cv-n-init                 EM restarts per fold during CV. Default 5.
      --mixture-model-n-init      EM restarts per in-sample mixture-model fit.
                                  Default 10.
      --mixture-model-reg-covar   Variance floor of the EM solver (also used by
                                  the tied model). Default 1e-4.
      --tau                       Posterior threshold for the LEFT component
                                  when computing inter-component decision
                                  boundaries. Default 0.5 (standard Bayes
                                  boundary).
      --figures-directory         Directory the inter-USV interval notebook uses to save
                                  rendered figures (not used by the analysis itself).
      --model-class               Mixture class of the unconstrained sweep. 't' =
                                  Student-t mixture in log space (default;
                                  heavy-tailed components absorb the long-pause
                                  tail). 'gauss' = log-Gaussian mixture
                                  (classical). 'ig' = inverse-Gaussian mixture in
                                  linear time (first-passage-time family for
                                  waiting times).
      --bootstrap-lrt-B           Number of parametric bootstrap replicates per
                                  rung, for the unconstrained LRT and the tied
                                  peak test alike. Default 1000.
      --bootstrap-lrt-n-subsample Subsample size for both observed and bootstrap
                                  fits. Default 10000. This is the effective
                                  sample size of the whole test, not just of the
                                  null: the observed statistic is computed on the
                                  same subsample, so raising it increases the
                                  test's power rather than sharpening the null
                                  against a fixed value.
      --bootstrap-lrt-alpha       Significance threshold for the step-up rule,
                                  before any Bonferroni correction. Default 0.01.
      --bootstrap-lrt-n-init      EM restarts for the bootstrap REFITS of the
                                  unconstrained LRT, separate from
                                  --mixture-model-n-init which governs the observed
                                  fit. Default 10. Restarts dominate the cost of
                                  this test and barely move its answer: measured on
                                  one male end-to-start K=4 vs K=5 comparison,
                                  going 3 -> 10 -> 20 moved the failed-refit rate
                                  75% -> 70% -> 65% and p 0.0090 -> 0.0080 ->
                                  0.0070, at 38 and 57 minutes for a SINGLE pair.
                                  A high failure rate is not necessarily an
                                  optimiser problem: it also arises when the
                                  larger K is simply not identifiable on the data.
      --bootstrap-lrt-n-jobs      Number of parallel workers for the bootstrap
                                  replicates and the serial-dependence resamples
                                  (1 = sequential). Default 24.
      --bootstrap-lrt-bonferroni / --no-bootstrap-lrt-bonferroni
                                  Divide alpha by the number of rungs before
                                  applying the step-up rule (default: enabled).
                                  With the shipped grids the unconstrained sweep
                                  has four rungs (2 vs 3 ... 5 vs 6, per-rung
                                  0.01 / 4 = 0.0025) and the tied peak test three
                                  (1 vs 2 ... 3 vs 4 peaks, 0.01 / 3 = 0.0033).

``generate-rm``
``generate-rm`` is the command-line interface for calculating per-cluster neuronal tuning curves (behavioral + vocal in one pass). Behavioral tuning runs when the session's ``*_behavioral_features.csv`` exists; vocal tuning runs when the ``*_usv_summary.csv`` and synced spike data exist. Sessions missing both inputs return cleanly without producing any tuning files.

.. code-block:: text

    usage: generate-rm [-h] --root-directory PATH [--temporal-offsets INTEGER...]
                       [--n-shuffles INTEGER] [--total-bin-num INTEGER]
                       [--n-spatial-bins INTEGER] [--spatial-scale-cm INTEGER]
                       [--peth-window-seconds FLOAT FLOAT] [--peth-bin-seconds FLOAT]
                       [--bout-quiet-seconds FLOAT]
                       [--n-usv-min-self INTEGER] [--n-usv-min-partner INTEGER]
                       [--n-usv-min-category INTEGER]
                       [--include-partner-tuning | --no-include-partner-tuning]
                       [--behavioral-min-occupancy-seconds FLOAT]
                       [--smoothing-sd FLOAT]

    required arguments:
      --root-directory                       Session root directory path.

    optional arguments:
      -h, --help                             Show this help message and exit.
      --temporal-offsets                     Spike-behavior offset(s) to consider (in s).
      --n-shuffles                           Number of shuffles.
      --total-bin-num                        Total number of bins for 1D tuning curves.
      --n-spatial-bins                       Number of spatial bins.
      --spatial-scale-cm                     Spatial extent of the arena (in cm).
      --peth-window-seconds                  Peri-USV-onset PETH window [start stop] relative to each call onset (in s); default -2 0.5.
      --peth-bin-seconds                     PETH bin width (in s).
      --bout-quiet-seconds                   Inter-bout silence required to define a new bout (in s).
      --n-usv-min-self                       Minimum self-side USV count to compute self plots.
      --n-usv-min-partner                    Minimum partner-side USV count to compute partner plots.
      --n-usv-min-category                   Minimum per-category USV count to retain that category.
      --include-partner-tuning /
        --no-include-partner-tuning          Also compute partner-side vocal tuning when partner threshold is met.
      --exclude-squeaks-self /
        --keep-squeaks-self                  Leave the self side's squeak and both (squeak + USV) segments out of
                                             its vocal tuning anchors (default: exclude). QLVM category tuning
                                             uses pure USVs only.
      --exclude-squeaks-partner /
        --keep-squeaks-partner               Leave the partner side's squeak and both (squeak + USV) segments out
                                             of its vocal tuning anchors (default: keep). QLVM category tuning
                                             uses pure USVs only.
      --excluded-behavioral-features         Behavioral base features (derivatives included) left out of
                                             tuning; repeat once per feature (default: nose-nose,
                                             allo_yaw-nose, nose-allo_yaw, allo_pitch-nose, nose-allo_pitch).
      --behavioral-min-occupancy-seconds     Minimum behavioral occupancy per bin (in s) for that bin
                                             to be rendered in the 1D feature line plots; persisted
                                             into ``behavioral_metadata`` of each cluster pkl.
      --smoothing-sd                         Standard deviation (in bins) of the Gaussian smoothing
                                             applied to ratemaps and shuffle distributions; ``0`` disables.

Visualize
---------

``generate-rm-figs``
``generate-rm-figs`` is the command-line interface for rendering the per-cluster neuronal tuning figures from existing pkls. Each cluster gets one combined output (behavioral pages + vocal Page 1 / Page 2). The output file format and ratemap colormap are read from ``visualizations_settings.json`` under the shared ``figures`` block; compute-time knobs (``smoothing_sd``, ``behavioral_min_occupancy_seconds``) live on the analyses side and are read from each pkl's ``behavioral_metadata`` block at render time.

.. code-block:: text

    usage: generate-rm-figs [-h] --root-directory PATH

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.


``generate-viz``
``generate-viz`` is the command-line interface for making plots/animations of 3D tracked mice.

.. code-block:: text

    usage: generate-viz [-h] --root-directory PATH --arena-directory PATH --exp-id TEXT
                        [--speaker-audio-file PATH] [--pitch-shifted-audio | --no-pitch-shifted-audio]
                        [--animate | --no-animate] [--video-start-time FLOAT]
                        [--video-duration FLOAT] [--plot-theme TEXT]
                        [--save-fig | --no-save-fig]
                        [--view-angle TEXT] [--side-azimuth-start FLOAT]
                        [--rotate-side-view | --no-rotate-side-view]
                        [--rotation-speed FLOAT]
                        [--history | --no-history] [--speaker | --no-speaker]
                        [--spectrogram | --no-spectrogram] [--spectrogram-ch INTEGER]
                        [--raster-plot | --no-raster-plot] [--brain-areas TEXT...]
                        [--other TEXT...] [--raster-special-units TEXT...]
                        [--spike-sound | --no-spike-sound]
                        [--beh-features | --no-beh-features]
                        [--beh-features-to-plot TEXT...]
                        [--special-beh-features TEXT...]
                        [--fig-format TEXT] [--fig-dpi INTEGER]
                        [--animation-codec TEXT]
                        [--animation-codec-preset TEXT]
                        [--animation-codec-tune TEXT]
                        [--animation-writer TEXT]
                        [--animation-format TEXT]
                        [--arena-node-connections | --no-arena-node-connections]
                        [--arena-axes-lw FLOAT] [--arena-mics-lw FLOAT]
                        [--arena-mics-opacity FLOAT]
                        [--plot-corners | --no-plot-corners]
                        [--corner-size FLOAT] [--corner-opacity FLOAT]
                        [--plot-mesh-walls | --no-plot-mesh-walls]
                        [--mesh-opacity FLOAT]
                        [--active-mic | --no-active-mic]
                        [--inactive-mic | --no-inactive-mic]
                        [--inactive-mic-color TEXT] [--text-fontsize INTEGER]
                        [--speaker-opacity FLOAT] [--nodes | --no-nodes]
                        [--node-size FLOAT] [--node-opacity FLOAT]
                        [--node-lw FLOAT]
                        [--node-connection-lw FLOAT] [--body-opacity FLOAT]
                        [--history-point TEXT] [--history-span-sec INTEGER]
                        [--history-ls TEXT] [--history-lw FLOAT]
                        [--beh-features-window-size INTEGER]
                        [--raster-window-size INTEGER] [--raster-lw FLOAT]
                        [--raster-ll FLOAT]
                        [--spectrogram-cbar | --no-spectrogram-cbar]
                        [--spectrogram-plot-window-size INTEGER]
                        [--spectrogram-power-limit INTEGER INTEGER]
                        [--spectrogram-frequency-limit INTEGER INTEGER]
                        [--spectrogram-yticks INTEGER...]
                        [--spectrogram-stft-nfft INTEGER]
                        [--plot-usv-segments | --no-plot-usv-segments]
                        [--usv-segments-ypos INTEGER] [--usv-segments-lw FLOAT]

    required arguments:
      --root-directory                 Session root directory path.
      --arena-directory                Arena session path.
      --exp-id                         Experimenter ID.

    optional arguments:
      -h, --help                       Show this help message and exit.
      --speaker-audio-file             Speaker audio file path.
      --pitch-shifted-audio / --no-pitch-shifted-audio
                                       Auto-produce and mux pitch-shifted (audible) USV audio onto the video.
      --animate / --no-animate         Animate visualization.
      --video-start-time               Video start time (in s).
      --video-duration                 Video duration (in s).
      --plot-theme                     Plot background theme (light or dark).
      --save-fig / --no-save-fig       Save plot as figure to file.
      --view-angle                     View angle for 3D visualization ("top" or "side").
      --side-azimuth-start             Azimuth angle for side view (in degrees).
      --rotate-side-view / --no-rotate-side-view
                                       Rotate side view in animation.
      --rotation-speed                 Speed of rotation for side view (in degrees/s).
      --history / --no-history         Display history of single mouse node.
      --speaker / --no-speaker         Display speaker node in visualization.
      --spectrogram / --no-spectrogram
                                       Display spectrogram of audio sequence.
      --spectrogram-ch                 Spectrogram channel (0-23).
      --raster-plot / --no-raster-plot
                                       Display spike raster plot in visualization.
      --brain-areas                    Brain areas to display in raster plot.
      --other                          Other spike cluster features to use for filtering.
      --raster-special-units           Clusters to accentuate in raster plot.
      --spike-sound / --no-spike-sound
                                       Play sound each time the cluster spikes.
      --beh-features / --no-beh-features
                                       Display behavioral feature dynamics.
      --beh-features-to-plot           Behavioral feature(s) to display.
      --special-beh-features           Behavioral feature(s) to accentuate in display.
      --fig-format                     Figure format.
      --fig-dpi                        Figure resolution in dots per inch.
      --animation-codec                The video codec for the animation writer.
      --animation-codec-preset         The preset flag for the animation codec.
      --animation-codec-tune           The tune flag for the animation codec.
      --animation-writer               Animation writer backend.
      --animation-format               Video format.
      --arena-node-connections / --no-arena-node-connections
                                       Display connections between arena nodes.
      --arena-axes-lw                  Line width for the arena axes.
      --arena-mics-lw                  Line width for the microphone markers.
      --arena-mics-opacity             Opacity for the microphone markers.
      --plot-corners / --no-plot-corners
                                       Display arena corner markers.
      --corner-size                    Size of the arena corner markers.
      --corner-opacity                 Opacity of the arena corner markers.
      --plot-mesh-walls / --no-plot-mesh-walls
                                       Display arena walls as a mesh.
      --mesh-opacity                   Opacity of the arena wall mesh.
      --active-mic / --no-active-mic   Display the active microphone marker.
      --inactive-mic / --no-inactive-mic
                                       Display inactive microphone markers.
      --inactive-mic-color             Color for inactive microphone markers.
      --text-fontsize                  Font size for text elements in the plot.
      --speaker-opacity                Opacity of the speaker node.
      --nodes / --no-nodes             Display mouse nodes.
      --node-size                      Size of the mouse nodes.
      --node-opacity                   Opacity of the mouse nodes.
      --node-lw                        Line width (edge) for the mouse node markers.
      --node-connection-lw             Line width for mouse node connections.
      --body-opacity                   Opacity of the mouse body.
      --history-point                  Node to use for the history trail.
      --history-span-sec               Duration of the history trail (s).
      --history-ls                     Line style for the history trail.
      --history-lw                     Line width for the history trail.
      --beh-features-window-size       Window size for behavioral features (s).
      --raster-window-size             Window size for the raster plot (s).
      --raster-lw                      Line width for spikes in the raster plot.
      --raster-ll                      Line length for spikes in the raster plot.
      --spectrogram-cbar / --no-spectrogram-cbar
                                       Display the color bar for the spectrogram.
      --spectrogram-plot-window-size   Window size for the spectrogram plot (s).
      --spectrogram-power-limit        Power (min/max) for spectrogram color scale.
      --spectrogram-frequency-limit    Freq. (min/max) for spectrogram y-axis (Hz).
      --spectrogram-yticks             Y-tick position for spectrogram.
      --spectrogram-stft-nfft          NFFT for the spectrogram STFT calculation.
      --plot-usv-segments / --no-plot-usv-segments
                                       Display USV assignments on the spectrogram.
      --usv-segments-ypos              Y-axis position for USV segment markers (Hz).
      --usv-segments-lw                Line width for USV segment markers.

``qlvm-torus-traversal-video``
``qlvm-torus-traversal-video`` renders a demo video that traverses the toroidal QLVM (the in-house quasi-Monte Carlo latent variable model) latent space.

.. code-block:: text

    usage: qlvm-torus-traversal-video [-h] [--output-path TEXT]
                                      [--clustering TEXT] [--fps INTEGER]

    optional arguments:
      -h, --help            Show this help message and exit.
      --output-path         Output .mp4 / .gif path (default: figures.save_directory + timestamp).
      --clustering          Clustering type borders (coarse / fine).
      --fps                 Video frames per second.

Neuropixels
-----------

``npx-meta-to-coords``
``npx-meta-to-coords`` converts a SpikeGLX (the SpikeGLX acquisition software) ``*.ap.meta`` file into a probe-geometry artifact: a Kilosort (the Kilosort spike-sorter) ``chanMap.mat``, plain-text or ``.npy`` site coordinates, JRClust (the JRClust spike sorter) ``.prm`` strings, or an in-place upgrade of a legacy (pre-SpikeGLX 032623) meta file. Pass ``--meta-file`` to run **headless** (no GUI, so the conversion can be scripted next to the spike-sorting step); with **no arguments** it launches an interactive Qt GUI whose three dialogs pick the meta file, the output format, and the destination / optional probe-layout plot. The related ``python -m usv_playpen.neuropixels.anatomy_converter`` utility (below) and the programmatic API are documented on the :ref:`Neuropixels` page.

.. code-block:: text

    usage: npx-meta-to-coords [-h] [--meta-file META_FILE]
                              [--output-format {text,kilosort_mat,jrclust_strings,npy,legacy_meta_augment}]
                              [--plot] [--save-plot SAVE_PLOT]

    optional arguments:
      -h, --help          Show this help message and exit.
      --meta-file         Path to the SpikeGLX *.ap.meta file. When given, runs headlessly (no GUI).
      --output-format     Output artefact format (headless mode; default: kilosort_mat).
      --plot              After a headless conversion, show the probe-layout plot interactively.
      --save-plot         After a headless conversion, write the probe-layout plot to this path.

    With no --meta-file, an interactive Qt GUI runs three dialogs instead:
      1. select a SpikeGLX *.ap.meta file;
      2. choose the output format (defaults to the Kilosort chanMap);
      3. confirm the destination, optionally showing the probe-layout plot.

``python -m usv_playpen.neuropixels.anatomy_converter``
Unlike ``npx-meta-to-coords``, the channel-brain area converter runs fully headless. It updates ``neuropixels_sites_to_anatomy_converter.json`` with Kilosort-row-keyed per-region channel ranges: pass ``--regenerate-all`` to rewrite every triple already in the file, or ``--mouse`` / ``--session`` / ``--probe`` to add just one; with no action it prints help and writes nothing (the full workflow is on the :ref:`Neuropixels` page).

.. code-block:: text

    usage: python -m usv_playpen.neuropixels.anatomy_converter [-h]
             [--converter-path PATH] [--ephys-root PATH] [--histology-root PATH]
             [--probe-to-hemisphere PROBE=HEMISPHERE [PROBE=HEMISPHERE ...]]
             [--regenerate-all] [--mouse TEXT] [--session TEXT] [--probe TEXT]
             [--force] [--dry-run]

    optional arguments:
      -h, --help            Show this help message and exit.
      --converter-path      Path to the converter JSON to update.
      --ephys-root          Root directory containing per-probe Kilosort outputs.
      --histology-root      Root directory containing per-mouse IBL histology output.
      --probe-to-hemisphere
                            Override the per-probe hemisphere map as space-separated
                            PROBE=HEMISPHERE pairs (e.g. imec0=R imec1=L); defaults to
                            analyses_settings.json.
      --regenerate-all      Bulk-regenerate EVERY triple already in the converter
                            (mutually exclusive with --mouse/--session/--probe).
      --mouse               Mouse id for single-triple mode (requires --session/--probe).
      --session             Session id (YYYYMMDD...) for single-triple mode.
      --probe               Probe id ('imec0'/'imec1') for single-triple mode.
      --force               Re-regenerate the single triple even if already present.
      --dry-run             Print the summary without writing the converter to disk.

CLI *Spock* cluster usage
-------------------------

In order to exploit the full functionality of *usv-playpen*, one should install subsidiary uv (sleap) or conda packages (das, vcl-ssl or vcl-ssl-ss). To install these on the *Spock* cluster, you can use the commands below (NB: the conda version is arbitrary, but you should note down which one you used):

.. code-block:: bash

    $ uv tool install --python 3.11 "sleap-nn[torch]==0.1.2" --torch-backend cu118

.. code-block:: bash

    $ module load anacondapy/2024.02
    $ conda init bash

.. code-block:: bash

    $ conda create python=3.10 das=0.32.2 -c conda-forge -c nvidia -c ncb -n das -y

.. code-block:: bash

    $ conda create --name vcl-ssl python=3.10 torchaudio packaging -y
    $ git clone https://github.com/Aramist/vocalocator-ssl.git && cd vocalocator-ssl
    $ conda activate vcl-ssl && pip install -e .

The shipped default ``vcl_conda_env_name`` is ``vcl-ssl-ss``, which uses the
`separate-scorers <https://github.com/Aramist/vocalocator-ssl/tree/separate-scorers>`_
branch of the same repository (this branch may become the default in the future). To
set that environment up instead of (or alongside) ``vcl-ssl``:

.. code-block:: bash

    $ conda create --name vcl-ssl-ss python=3.10 torchaudio packaging -y
    $ git clone -b separate-scorers https://github.com/Aramist/vocalocator-ssl.git vocalocator-ssl-ss && cd vocalocator-ssl-ss
    $ conda activate vcl-ssl-ss && pip install -e .

Having set up these environments, you can set up directories with bash scripts in /src/other/DAS, /src/other/HPSS, /src/other/SLEAP and /src/other/USV_PLAYPEN and run them to expedite your data processing or analysis.
