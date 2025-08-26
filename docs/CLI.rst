.. _CLI:

Command Line Interfaces (CLI)
=============================
This page explains how to use the *usv-playpen* CLI (command line interfaces).

Record
^^^^^^

``conduct-calibration``
-----------------------
``conduct-calibration`` is the command-line interface for performing a tracking camera calibration.

.. code-block:: plaintext

    usage: conduct-calibration [-h] [--set KEY.PATH=VALUE]...

    optional arguments:
      -h, --help            Show this help message and exit.
      --set                 Override a specific setting using a dot-path. This option
                            can be used multiple times. For example:
                            --set calibration_duration=10
                            --set video.general.calibration_frame_rate=20

``conduct-recording``
---------------------
``conduct-recording`` is the command-line interface for conducting a recording session.

.. code-block:: plaintext

    usage: conduct-recording [-h] [--set KEY.PATH=VALUE]...

    optional arguments:
      -h, --help            Show this help message and exit.
      --set                 Override a specific setting using a dot-path. This option
                            can be used multiple times. For example:
                            --set video_session_duration=25
                            --set audio.general.fftlength=512
                            --set video.metadata.notes="This is a special run."

Process
^^^^^^^

``concatenate-ephys-files``
---------------------------
``concatenate-ephys-files`` is the command-line interface for concatenating ephys binary files across multiple sessions.

.. code-block:: plaintext

    usage: concatenate-ephys-files [-h] --root-directories TEXT,TEXT,...

    required arguments:
      --root-directories    A comma-separated string of session root directory paths.

    optional arguments:
      -h, --help            Show this help message and exit.

``split-clusters``
------------------
``split-clusters`` is the command-line interface for splitting curated ephys clusters into individual session files.

.. code-block:: plaintext

    usage: split-clusters [-h] --root-directories TEXT,TEXT,...
                          [--min-spikes INTEGER] [--kilosort-version TEXT]

    required arguments:
      --root-directories    A comma-separated string of session root directory paths.

    optional arguments:
      -h, --help            Show this help message and exit.
      --min-spikes          Minimum number of spikes for a cluster to be saved.
      --kilosort-version    Version of Kilosort used for spike sorting.

``concatenate-video-files``
-------------------------
``concatenate-video-files`` is the command-line interface for concatenating video files.

.. code-block:: plaintext

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
-------------------------
``rectify-video-fps`` iis the command-line interface for re-encoding videos to a correct frame rate.

.. code-block:: plaintext

    usage: rectify-video-fps [-h] --root-directory PATH [--camera-serial TEXT...]
                             [--target-file TEXT] [--extension TEXT]
                             [--crf INTEGER] [--preset TEXT]
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
-----------------------------
``multichannel-to-single-ch`` is the command-line interface for splitting multichannel audio files into single-channel files.

.. code-block:: plaintext

    usage: multichannel-to-single-ch [-h] --root-directory PATH

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.

``crop-wav-files``
------------------
``crop-wav-files`` is the command-line interface for cropping audio WAV files to match video length.

.. code-block:: plaintext

    usage: crop-wav-files [-h] --root-directory PATH [--trigger-device TEXT]
                          [--trigger-channel INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --trigger-device      USGH device(s) receiving triggerbox input.
      --trigger-channel     USGH channel receiving triggerbox input.

``av-sync-check``
-----------------
``av-sync-check`` is the command-line interface for checking audio-video synchronization and generating a summary figure.

.. code-block:: plaintext

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
-----------------
``ev-sync-check`` is the command-line interface for validating ephys-video synchronization.

.. code-block:: plaintext

    usage: ev-sync-check [-h] --root-directory PATH [--file-type TEXT]
                         [--tolerance FLOAT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --file-type           Neuropixels file type (ap or lf).
      --tolerance           Divergence tolerance (in ms).

``hpss-audio``
--------------
``hpss-audio`` is the command-line interface for performing Harmonic-Percussive Source Separation (HPSS) on audio files.

.. code-block:: plaintext

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
-------------------
``bp-filter-audio`` is the command-line interface for band-pass filtering audio files.

.. code-block:: plaintext

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
---------------------------
``concatenate-audio-files`` is the command-line interface for vertically stacking audio files into a single memmap file.

.. code-block:: plaintext

    usage: concatenate-audio-files [-h] --root-directory PATH
                                   [--format TEXT] [--dirs TEXT...]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --format              Audio file format.
      --dirs                Directory/ies to search for files to concatenate.

``sleap-to-h5``
---------------
``sleap-to-h5`` is the command-line interface for converting SLEAP (.slp) files to HDF5 (.h5) files.

.. code-block:: plaintext

    usage: sleap-to-h5 [-h] --root-directory PATH [--env-name TEXT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --env-name            SLEAP conda environment.

``anipose-calibrate``
---------------------
``anipose-calibrate`` is the command-line interface for conducting Anipose camera calibration.

.. code-block:: plaintext

    usage: anipose-calibrate [-h] --root-directory PATH
                             [--board-provided | --no-board-provided]
                             [--board-dims INTEGER INTEGER] [--square-len INTEGER]
                             [--marker-params FLOAT FLOAT] [--dict-size INTEGER]
                             [--img-dims INTEGER INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --board-provided / --no-board-provided
                            Indicate that the calibration board is provided.
      --board-dims          Checkerboard dimensions (squares_x, squares_y).
      --square-len          Length of a checkerboard square (mm).
      --marker-params       ArUco marker length (mm) and dictionary bits.
      --dict-size           Size of the ArUco dictionary.
      --img-dims            Image dimensions (width, height) in pixels.

``anipose-triangulate``
-----------------------
``anipose-triangulate`` is the command-line interface for conducting Anipose 3D triangulation.

.. code-block:: plaintext

    usage: anipose-triangulate [-h] --root-directory PATH --calibration-file PATH
                               [--arena-points | --no-arena-points]
                               [--frame-restriction INTEGER]
                               [--exclude-views TEXT...]
                               [--display-progress | --no-display-progress]
                               [--use-ransac | --no-use-ransac]
                               [--rigid-constraints "TEXT,TEXT"...]
                               [--weak-constraints "TEXT,TEXT"...] [--smooth-scale FLOAT]
                               [--weight-weak INTEGER] [--weight-rigid INTEGER]
                               [--reprojection-threshold INTEGER] [--regularization TEXT]
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
      --rigid-constraints        Pair(s) of nodes for a rigid constraint.
      --weak-constraints         Pair(s) of nodes for a weak constraint.
      --smooth-scale             Scaling factor for smoothing.
      --weight-weak              Weight for weak constraints.
      --weight-rigid             Weight for rigid constraints.
      --reprojection-threshold   Reprojection error threshold.
      --regularization           Regularization function to use.
      --n-deriv-smooth           Number of derivatives to use for smoothing.

``anipose-trm``
---------------
``anipose-trm`` is the command-line interface for translating, rotating, and scaling 3D point data.

.. code-block:: plaintext

    usage: anipose-trm [-h] --root-directory PATH --exp-code TEXT --arena-file PATH
                       [--save-data-for TEXT]
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
-------------
``das-infer`` is the command-line interface for running Deep Audio Segmenter (DAS) inference on audio files.

.. code-block:: plaintext

    usage: das-infer [-h] --root-directory PATH [--env-name TEXT] [--model-dir PATH]
                     [--model-name TEXT] [--output-type TEXT]
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
-----------------
``das-summarize`` is the command-line interface for summarizing DAS inference findings.

.. code-block:: plaintext

    usage: das-summarize [-h] --root-directory PATH [--win-len INTEGER]
                         [--freq-cutoff INTEGER] [--corr-cutoff FLOAT]
                         [--var-cutoff FLOAT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --win-len             Window length of the signal.
      --freq-cutoff         Low frequency cutoff (Hz).
      --corr-cutoff         Minimum noise correlation cutoff.
      --var-cutoff          Maximum noise variance cutoff.

Analyze
^^^^^^^

``generate-beh-features``
-------------------------
``generate-beh-features`` is the command-line interface for calculating 3D behavioral features.

.. code-block:: plaintext

    usage: generate-beh-features  [-h] --root-directory PATH
                                  [--head-points TEXT TEXT TEXT TEXT]
                                  [--tail-points TEXT TEXT TEXT TEXT TEXT]
                                  [--back-root-points TEXT TEXT TEXT]
                                  [--derivative-bins TEXT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --head-points         Skeleton head nodes.
      --tail-points         Skeleton tail nodes.
      --back-root-points    Skeleton back nodes.
      --derivative-bins     Number of bins for derivative calculation.


``generate-usv-playback``
-------------------------
``generate-usv-playback`` is the command-line interface for generating artificial USV playback files.

.. code-block:: plaintext

    usage: generate-usv-playback [-h] --exp-id TEXT [--num-usv-files INTEGER]
                                 [--total-usv-number INTEGER] [--ipi-duration FLOAT]
                                 [--wav-sampling-rate INTEGER]
                                 [--playback-snippets-dir TEXT]

    required arguments:
        --exp-id                     Experimenter ID.

    optional arguments:
        -h, --help                   Show this help message and exit.
        --num-usv-files              Number of WAV files to create.
        --total-usv-number           Total number of USVs to distribute across file.
        --ipi-duration               Inter-USV-interval duration (in s).
        --wav-sampling-rate          Sampling rate for the output WAV file (in Hz).
        --playback-snippets-dir      Directory of USV playback snippets.

``generate-naturalistic-usv-playback``
--------------------------------------
``generate-naturalistic-usv-playback`` is the command-line interface for generating naturalistic USV playback files.

.. code-block:: plaintext

    usage: generate-usv-playback [-h] --exp-id TEXT [--num-naturalistic-usv-files INTEGER]
                                 [--naturalistic-wav-sampling-rate INTEGER]
                                 [--total-playback-time INTEGER]
                                 [--naturalistic-playback-snippets-dir-prefix TEXT]
                                 [--inter-seq-interval-dist TEXT]
                                 [--usv-seq-length-dist TEXT]
                                 [--inter-usv-interval-dist TEXT]

    required arguments:
        --exp-id                                      Experimenter ID.

    optional arguments:
        -h, --help                                    Show this help message and exit.
        --num-naturalistic-usv-files                  Number of naturalistic playback files to be created.
        --naturalistic-wav-sampling-rate              Sampling rate of the naturalistic playback .WAV file in kHz.
        --naturalistic-playback-snippets-dir-prefix   Prefix of the snippet subdirectory (the rest of its name should be "_usv_playback_snippets".
        --total-playback-time                         Total acceptable time of the playback time (in s).
        --inter-seq-interval-dist                     Distribution of inter-sequence intervals (time (s) : probability (sums to 1).
        --usv-seq-length-dist                         Distribution of USV sequence lengths (time (s) : probability (sums to 1).
        --inter-usv-interval-dist                     Distribution of inter-USV intervals (time (s) : probability (sums to 1).

``generate-rm``
---------------
``generate-rm`` is the command-line interface for calculating neural-behavioral tuning curves.

.. code-block:: plaintext

    usage: generate-rm [-h] --root-directory PATH [--temporal-offsets INTEGER...]
                       [--n-shuffles INTEGER] [--total-bin-num INTEGER]
                       [--n-spatial-bins INTEGER] [--spatial-scale-cm INTEGER]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --temporal-offsets    Spike-behavior offset(s) to consider (in s).
      --n-shuffles          Number of shuffles.
      --total-bin-num       Total number of bins for 1D tuning curves.
      --n-spatial-bins      Number of spatial bins.
      --spatial-scale-cm    Spatial extent of the arena (in cm).

Visualize
^^^^^^^^^

``generate-rm-figs``
--------------------
``generate-rm-figs`` is the command-line interface for making neural-behavioral tuning curve figures.

.. code-block:: plaintext

    usage: generate-rm-figs [-h] --root-directory PATH [--smoothing-sd FLOAT]
                            [--occ-threshold FLOAT]
    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --smoothing-sd        Standard deviation of smoothing (in bins).
      --occ-threshold       Minimum acceptable occupancy (in s).


``generate-viz``
----------------
``generate-viz`` is the command-line interface for making plots/animations of 3D tracked mice.

.. code-block:: plaintext

    usage: generate-viz [-h] --root-directory PATH --arena-directory PATH --exp-id TEXT
                        [--speaker-audio-file PATH] [--sequence-audio-file PATH]
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
      --sequence-audio-file            Audible audio sequence file path.
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
      --node-lw                        Line width for the mouse node connections.
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
