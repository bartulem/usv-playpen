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

    required arguments:
      --root-directories    A comma-separated string of session root directory paths.

    optional arguments:
      -h, --help            Show this help message and exit.
      --min-spikes          Minimum number of spikes for a cluster to be saved.
      --kilosort-version    Version of Kilosort used for spike sorting.

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

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --file-type           Neuropixels file type (ap or lf).
      --tolerance           Divergence tolerance (in ms).

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
                         [--corr-cutoff FLOAT] [--var-cutoff FLOAT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --filter-putative-noise / --no-filter-putative-noise
                            Run the Phase-4 amplitude/spectrogram noise rejection (default: enabled); pass --no-filter-putative-noise to keep every merged detection.
      --win-len             Window length of the signal.
      --freq-cutoff         Low frequency cutoff (Hz).
      --corr-cutoff         Minimum noise correlation cutoff.
      --var-cutoff          Maximum noise variance cutoff.

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

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --vcl-version         Version of Vocalocator to use ('vcl' or 'vcl-ssl').
      --env-name            Name of the Vocalocator conda environment.
      --model-dir           Directory of the Vocalocator model.

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

``build-qlvm-training-set``
``build-qlvm-training-set`` aggregates a list of session root directories into a single curated ``.npz`` training set (``train_data.npz`` + ``val_data.npz``, or ``full_data.npz``) for the QLVM. With ``--masking-type sam`` (default), each kept spectrogram is masked by the union of its SAM mask regions from the ``mask/<session>`` group (background zeroed; a call with no detected mask keeps an all-ones mask); ``--masking-type none`` keeps raw spectrograms.

.. code-block:: text

    usage: build-qlvm-training-set [-h] --root-directories TEXT,TEXT,... --output-directory PATH
                            [--length-threshold FLOAT]
                            [--dataset-size-constraint INTEGER]
                            [--validation-split FLOAT] [--random-state INTEGER]
                            [--target-shape INTEGER INTEGER]
                            [--full-dataset | --no-full-dataset]
                            [--time-stretch | --no-time-stretch]
                            [--masking-type {sam,none}]

    required arguments:
      --root-directories              Comma-separated string of session root directory paths.
      --output-directory              Directory to write the .npz training set.

    optional arguments:
      -h, --help                      Show this help message and exit.
      --length-threshold              Drop spectrograms with duration >= threshold (time bins).
      --dataset-size-constraint       Optional cap on the total number of kept spectrograms (absolute count if > 1, proportion if in (0, 1]); omit for no cap (null = all data).
      --validation-split              Fraction held out for validation.
      --random-state                  RNG (random number generator) seed for reproducible subsampling and the train/val split.
      --target-shape                  Output spectrogram (freq, time) shape as two ints, e.g. --target-shape 128 128.
      --full-dataset / --no-full-dataset
                                      Write a single full_data.npz (no train/val split).
      --time-stretch / --no-time-stretch
                                      Time-warp the signal window instead of center-resizing.
      --masking-type                  Apply SAM mask regions from the mask/<session> groups ("sam") or keep raw spectrograms ("none").

``train-qlvm``
``train-qlvm`` trains the QLVM decoder on a ``build-qlvm-training-set`` ``.npz`` set (fixed quasi-random torus lattice + ConvTranspose decoder, Bernoulli evidence objective) and writes ``qmc_train_qlvm.tar`` (full checkpoint) plus ``qmc_decoder_weights.npz``. That ``.npz`` is the train→inference bridge: point ``infer-qlvm-latents``' ``weights_npz_path`` at it (the torch-free JAX (the JAX numerical-computing library) inference reloads exactly these decoder weights). GPU recommended.

.. code-block:: text

    usage: train-qlvm [-h] --dataset-directory PATH --output-directory PATH
                            [--n-epochs INTEGER] [--latent-dim INTEGER]
                            [--lattice-type {korobov,roberts,fibonacci}]
                            [--korobov-a INTEGER] [--train-n-points INTEGER]
                            [--test-n-points INTEGER] [--fib-m INTEGER]
                            [--batch-size INTEGER] [--learning-rate FLOAT]
                            [--val-freq INTEGER] [--seed INTEGER]
                            [--num-workers INTEGER]

    required arguments:
      --dataset-directory   Directory holding the .npz training set (build-qlvm-training-set output).
      --output-directory    Directory to write the checkpoint + decoder-weights .npz.

    optional arguments:
      -h, --help            Show this help message and exit.
      --n-epochs            Number of training epochs.
      --latent-dim          Torus latent dimensionality.
      --lattice-type        Quasi-random lattice generator.
      --korobov-a           Korobov generating integer.
      --train-n-points      Number of quasi-random lattice points used during training (korobov/roberts).
      --test-n-points       Number of quasi-random lattice points used at evaluation/validation (korobov/roberts).
      --fib-m               Fibonacci lattice order (used only when lattice-type=fibonacci; 2D only).
      --batch-size          Training batch size.
      --learning-rate       Adam learning rate.
      --val-freq            Run validation-evidence evaluation every N epochs (must be >= 1).
      --seed                Global RNG seed (torch + numpy) for reproducible shuffling / per-batch torus shifts.
      --num-workers         DataLoader worker processes (0 = load in the main process).

``infer-qlvm-latents``
``infer-qlvm-latents`` embeds a session's spectrograms into the trained QLVM toroidal latent space (loading the ``qmc_decoder_weights.npz`` written by ``train-qlvm``) and merges four columns into ``usv_summary.csv``: the torus coordinates ``qlvm_dim1`` / ``qlvm_dim2``, plus ``qlvm_category`` (fine cluster) and ``qlvm_supercategory`` (coarse cluster), each looked up in the ``ws_labels_periodic`` grid of a fine and a coarse reference ``arrays.npz``. With ``--masking-type sam`` (default) each spectrogram is masked by the union of its SAM mask regions from the ``mask/<session>`` group before embedding -- matching how the decoder was trained by ``build-qlvm-training-set`` (embedding raw spectrograms into a masked-trained decoder is out-of-distribution); ``--masking-type none`` embeds raw spectrograms.

.. code-block:: text

    usage: infer-qlvm-latents [-h] --root-directory PATH
                            [--weights-npz-path TEXT]
                            [--reference-arrays-fine-npz-path TEXT]
                            [--reference-arrays-coarse-npz-path TEXT]
                            [--lattice-type {korobov,roberts,fibonacci}]
                            [--latent-dim INTEGER] [--n-points INTEGER]
                            [--korobov-a INTEGER] [--fib-m INTEGER]
                            [--time-stretch | --no-time-stretch]
                            [--masking-type {sam,none}]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --weights-npz-path    Path to the converted decoder weights .npz.
      --reference-arrays-fine-npz-path
                            Path to the FINE reference arrays.npz (ws_labels_periodic -> qlvm_category).
      --reference-arrays-coarse-npz-path
                            Path to the COARSE reference arrays.npz (ws_labels_periodic -> qlvm_supercategory).
      --lattice-type        Quasi-random lattice generator used to rebuild the fixed QLVM (quasi-Monte Carlo latent variable model) lattice at inference.
      --latent-dim          Dimensionality of the toroidal latent space.
      --n-points            Number of lattice points used at inference.
      --korobov-a           Korobov generating integer (used when lattice-type=korobov).
      --fib-m               Fibonacci lattice order m (used when lattice-type=fibonacci).
      --time-stretch / --no-time-stretch
                            Whether to time-stretch each spectrogram to the fixed size (matching training preprocessing) instead of a plain resize.
      --masking-type        Apply SAM mask regions from the mask/<session> groups before embedding ("sam", matching how the decoder was trained) or embed raw spectrograms ("none").

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
``generate-usv-interval-distributions`` is the command-line interface for computing inter-vocalization-interval (inter-USV interval) distributions across one or more session-list text files and (optionally) sweeping a 1D mixture model (Gaussian or Student-t) on the pooled log-inter-USV intervals.

By convention, ``track_names[0]`` is treated as the male and ``track_names[1]`` as the female. Each session-list text file contains one session root directory per line; paths are run through ``configure_path`` so Mac/Linux/Windows entries resolve correctly on the host platform. ``--session-list`` may be passed multiple times to merge multiple cohorts.

Both interval definitions are computed unconditionally on every run: ``s2s`` = ``start[i+1] - start[i]`` (literature standard), and ``e2s`` = ``start[i+1] - stop[i]`` (alternate; can be negative for overlapping calls and is dropped via the ``> 0`` filter, with the drop count reported per session per mode). Both definitions share the same per-session pass over the noise-filtered USV table, so there is no compute saving from omitting one.

The command writes a single self-describing HDF5 archive ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5`` to ``--output-directory``. Per interval mode it holds the tidy one-row-per-interval table, the per-sex drop counts, and (when ``--fit-mixture-model``) the full Gaussian / Student-t mixture sweep — all four information criteria plus per-component parameters — and the bootstrap-LRT results; the root ``/attrs`` records every parameter that drove the run (plus ``git_sha`` and the source lists), so the archive is fully self-describing. See :doc:`Notebooks` for the complete archive schema and the plotting notebook that reads it.

.. code-block:: text

    usage: generate-usv-interval-distributions [-h] [--session-list PATH...] [--output-directory PATH]
                            [--noise-col-id TEXT] [--noise-categories INTEGER...]
                            [--fit-mixture-model | --no-fit-mixture-model]
                            [--n-components-min INTEGER] [--n-components-max INTEGER]
                            [--n-repeats INTEGER] [--max-modes-reported INTEGER]
                            [--random-seed-base INTEGER]
                            [--cv-n-folds INTEGER] [--cv-n-init INTEGER]
                            [--mixture-model-n-init INTEGER] [--mixture-model-reg-covar FLOAT]
                            [--tau FLOAT] [--figures-directory PATH]
                            [--model-class {gauss,t}]
                            [--bootstrap-lrt-B INTEGER]
                            [--bootstrap-lrt-n-subsample INTEGER]
                            [--bootstrap-lrt-alpha FLOAT]
                            [--bootstrap-lrt-bonferroni | --no-bootstrap-lrt-bonferroni]

    optional arguments:
      -h, --help                  Show this help message and exit.
      --session-list              Path to a text file containing session root
                                  directories (one per line). Repeatable.
      --output-directory          Directory in which to write the consolidated
                                  usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5 archive.
      --noise-col-id              Name of the noise classification column in the
                                  USV summary CSV.
      --noise-categories          Integer label(s) in noise_col_id that mark a
                                  USV as noise.
      --fit-mixture-model / --no-fit-mixture-model    Whether to run the mixture-model sweep after inter-USV interval extraction.
      --n-components-min          Minimum number of mixture components.
      --n-components-max          Maximum number of mixture components.
      --n-repeats                 Number of EM-init repeats per (key, n_components).
      --max-modes-reported        Maximum number of mixture modes recorded per fit.
      --random-seed-base          Base seed; rep r uses random_seed_base + r.
      --cv-n-folds                Number of K-fold splits for CV log-likelihood.
                                  Default 5.
      --cv-n-init                 EM restarts per fold during CV. Default 5.
      --mixture-model-n-init      EM restarts per in-sample mixture-model fit. Default 10.
      --mixture-model-reg-covar   Covariance regularisation passed to sklearn's
                                  GaussianMixture. Default 1e-4.
      --tau                       Posterior threshold for the LEFT component
                                  when computing inter-component decision
                                  boundaries. Default 0.5 (standard Bayes
                                  boundary).
      --figures-directory         Directory the inter-USV interval notebook uses to save
                                  rendered figures.
      --model-class               Mixture class. 't' = Student-t mixture
                                  (default; one heavy-tailed component
                                  absorbs the long-pause tail). 'gauss' =
                                  log-Gaussian mixture (classical).
      --bootstrap-lrt-B           Number of parametric bootstrap replicates
                                  per pairwise LRT. Default 1000.
      --bootstrap-lrt-n-subsample Subsample size for both observed and
                                  bootstrap fits. Default 15000.
      --bootstrap-lrt-alpha       Significance threshold for the step-up
                                  rule. Default 0.05.
      --bootstrap-lrt-bonferroni  Divide alpha by the number of pairwise
                                  tests before applying the step-up rule.

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
      --peth-window-seconds                  Pre-USV PETH window [start stop] (in s).
      --peth-bin-seconds                     PETH bin width (in s).
      --bout-quiet-seconds                   Inter-bout silence required to define a new bout (in s).
      --n-usv-min-self                       Minimum self-side USV count to compute self plots.
      --n-usv-min-partner                    Minimum partner-side USV count to compute partner plots.
      --n-usv-min-category                   Minimum per-category USV count to retain that category.
      --include-partner-tuning /
        --no-include-partner-tuning          Also compute partner-side vocal tuning when partner threshold is met.
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

``build-vae-density``
``build-vae-density`` builds the pooled variational autoencoder (VAE)-Uniform Manifold Approximation and Projection (UMAP) density / category-label maps (coarse + fine) used as the embedding-space background for the USV spectrogram figures.

.. code-block:: text

    usage: build-vae-density [-h] --sessions-txt TEXT --out-coarse TEXT
                             --out-fine TEXT [--cache-path TEXT]
                             [--grid INTEGER] [--smooth-sigma FLOAT] [--knn INTEGER]

    required arguments:
      --sessions-txt        Text file listing one session root per line.
      --out-coarse          Output .npz path for the COARSE map (vae_supercategory).
      --out-fine            Output .npz path for the FINE map (vae_category).

    optional arguments:
      -h, --help            Show this help message and exit.
      --cache-path          Optional parquet cache for the pooled DataFrame.
      --grid                Density / label grid side length (default 300).
      --smooth-sigma        Gaussian sigma in grid cells; 0 = raw histogram (default 1.5).
      --knn                 Neighbours for the grid label classifier (default 15).

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
             [--regenerate-all] [--mouse TEXT] [--session TEXT] [--probe TEXT]
             [--force] [--dry-run]

    optional arguments:
      -h, --help            Show this help message and exit.
      --converter-path      Path to the converter JSON to update.
      --ephys-root          Root directory containing per-probe Kilosort outputs.
      --histology-root      Root directory containing per-mouse IBL histology output.
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
