.. _CLI:

Command Line Interfaces (CLI)
=============================
This page explains how to use the *usv-playpen* CLI (command line interfaces).

Record
^^^^^^

``conduct-calibration``
-----------------------
``conduct-calibration`` is the command-line interface for performing a tracking camera calibration.

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

    usage: concatenate-ephys-files [-h] --root-directories TEXT,TEXT,...

    required arguments:
      --root-directories    A comma-separated string of session root directory paths.

    optional arguments:
      -h, --help            Show this help message and exit.

``split-clusters``
------------------
``split-clusters`` is the command-line interface for splitting curated ephys clusters into individual session files.

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

    usage: rectify-video-fps [-h] --root-directory PATH [--camera-serial TEXT...]
                             [--target-file TEXT] [--extension TEXT]
                             [--crf INTEGER] [--preset TEXT]
                             [--delete-old-file | --no-delete-old-file]
                             [--conduct-concat | --no-conduct-concat]

    required arguments:
      --root_directory      Session root directory path.

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

.. code-block:: bash

    usage: multichannel-to-single-ch [-h] --root-directory PATH

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.

``crop-wav-files``
------------------
``crop-wav-files`` is the command-line interface for cropping audio WAV files to match video length.

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

    usage: sleap-to-h5 [-h] --root-directory PATH [--env-name TEXT]

    required arguments:
      --root-directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --env-name            SLEAP conda environment.

``anipose-calibrate``
---------------------
``anipose-calibrate`` is the command-line interface for conducting Anipose camera calibration.

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

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

.. code-block:: bash

    usage: generate-beh-features  [-h] --root_directory PATH
                                  [--head_points TEXT TEXT TEXT TEXT]
                                  [--tail_points TEXT TEXT TEXT TEXT TEXT]
                                  [--back_root_points TEXT TEXT TEXT]
                                  [--derivative_bins TEXT]

    required arguments:
      --root_directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --head_points         Skeleton head nodes.
      --tail_points         Skeleton tail nodes.
      --back_root_points    Skeleton back nodes.
      --derivative_bins     Number of bins for derivative calculation.


``generate-usv-playback``
-------------------------
``generate-usv-playback`` is the command-line interface for generating USV playback files.

.. code-block:: bash

    usage: generate-usv-playback [-h] --exp_id TEXT [--num_usv_files INTEGER]
                                 [--total_usv_number INTEGER] [--ipi_duration FLOAT]
                                 [--wav_sampling_rate INTEGER]
                                 [--playback_snippets_dir TEXT]
    required arguments:
      --exp_id                     Experimenter ID.

    optional arguments:
      -h, --help                   Show this help message and exit.
      --num_usv_files              Number of WAV files to create.
      --total_usv_number           Total number of USVs to distribute across file.
      --ipi_duration               Inter-USV-interval duration (in s).
      --wav_sampling_rate          Sampling rate for the output WAV file (in Hz).
      --playback_snippets_dir      Directory of USV playback snippets.

``generate-rm``
---------------
``generate-rm`` is the command-line interface for calculating neural-behavioral tuning curves.

.. code-block:: bash

    usage: generate-rm [-h] --root_directory PATH [--temporal_offsets INTEGER...]
                       [--n_shuffles INTEGER] [--total_bin_num INTEGER]
                       [--n_spatial_bins INTEGER] [--spatial_scale_cm INTEGER]

    required arguments:
      --root_directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --temporal_offsets    Spike-behavior offset(s) to consider (in s).
      --n_shuffles          Number of shuffles.
      --total_bin_num       Total number of bins for 1D tuning curves.
      --n_spatial_bins      Number of spatial bins.
      --spatial_scale_cm    Spatial extent of the arena (in cm).

Visualize
^^^^^^^^^

``generate-rm-figs``
--------------------
``generate-rm-figs`` is the command-line interface for making neural-behavioral tuning curve figures.

.. code-block:: bash

    usage: generate-rm-figs [-h] --root_directory PATH [--smoothing_sd FLOAT]
                            [--occ_threshold FLOAT]
    required arguments:
      --root_directory      Session root directory path.

    optional arguments:
      -h, --help            Show this help message and exit.
      --smoothing_sd        Standard deviation of smoothing (in bins).
      --occ_threshold       Minimum acceptable occupancy (in s).


``generate-viz``
----------------
``generate-viz`` is the command-line interface for making plots/animations of 3D tracked mice.

.. code-block:: bash

    usage: generate-viz [-h] --root_directory PATH --arena_directory PATH --exp_id TEXT
                        [--speaker_audio_file PATH] [--sequence_audio_file PATH]
                        [--animate | --no-animate] [--video_start_time INTEGER]
                        [--video_duration FLOAT] [--plot_theme TEXT]
                        [--save-fig | --no-save-fig]
                        [--view_angle TEXT] [--side_azimuth_start FLOAT]
                        [--rotate-side-view | --no-rotate-side-view]
                        [--rotation_speed FLOAT]
                        [--history | --no-history] [--speaker | --no-speaker]
                        [--spectrogram | --no-spectrogram] [--spectrogram_ch INTEGER]
                        [--raster-plot | --no-raster-plot] [--brain_areas TEXT...]
                        [--other TEXT...] [--raster_special_units TEXT...]
                        [--spike-sound | --no-spike-sound]
                        [--beh-features | --no-beh-features]
                        [--beh_features_to_plot TEXT...]
                        [--special_beh_features TEXT...]
                        [--fig_format TEXT] [--fig_dpi INTEGER]
                        [--animation_writer TEXT]
                        [--animation_format TEXT]
                        [--arena-node-connections | --no-arena-node-connections]
                        [--arena_axes_lw FLOAT] [--arena_mics_lw FLOAT]
                        [--arena_mics_opacity FLOAT]
                        [--plot-corners | --no-plot-corners]
                        [--corner_size FLOAT] [--corner_opacity FLOAT]
                        [--plot-mesh-walls | --no-plot-mesh-walls]
                        [--mesh_opacity FLOAT]
                        [--active-mic | --no-active-mic]
                        [--inactive-mic | --no-inactive-mic]
                        [--inactive_mic_color TEXT] [--text_fontsize INTEGER]
                        [--speaker_opacity FLOAT] [--nodes | --no-nodes]
                        [--node_size FLOAT] [--node_opacity FLOAT]
                        [--node_lw FLOAT]
                        [--node_connection_lw FLOAT] [--body_opacity FLOAT]
                        [--history_point TEXT] [--history_span_sec INTEGER]
                        [--history_ls TEXT] [--history_lw FLOAT]
                        [--beh_features_window_size INTEGER]
                        [--raster_window_size INTEGER] [--raster_lw FLOAT]
                        [--raster_ll FLOAT]
                        [--spectrogram-cbar | --no-spectrogram-cbar]
                        [--spectrogram_plot_window_size INTEGER]
                        [--spectrogram_power_limit INTEGER INTEGER]
                        [--spectrogram_frequency_limit INTEGER INTEGER]
                        [--spectrogram_yticks INTEGER...]
                        [--spectrogram_stft_nfft INTEGER]
                        [--plot-usv-segments | --no-plot-usv-segments]
                        [--usv_segments_ypos INTEGER] [--usv_segments_lw FLOAT]

    required arguments:
      --root_directory                 Session root directory path.
      --arena_directory                Arena session path.
      --exp_id                         Experimenter ID.

    optional arguments:
      -h, --help                       Show this help message and exit.
      --speaker_audio_file             Speaker audio file path.
      --sequence_audio_file            Audible audio sequence file path.
      --animate / --no-animate         Animate visualization.
      --video_start_time               Video start time (in s).
      --video_duration                 Video duration (in s).
      --plot_theme                     Plot background theme (light or dark).
      --save-fig / --no-save-fig       Save plot as figure to file.
      --view_angle                     View angle for 3D visualization ("top" or "side").
      --side_azimuth_start             Azimuth angle for side view (in degrees).
      --rotate-side-view / --no-rotate-side-view
                                       Rotate side view in animation.
      --rotation_speed                 Speed of rotation for side view (in degrees/s).
      --history / --no-history         Display history of single mouse node.
      --speaker / --no-speaker         Display speaker node in visualization.
      --spectrogram / --no-spectrogram
                                       Display spectrogram of audio sequence.
      --spectrogram_ch                 Spectrogram channel (0-23).
      --raster-plot / --no-raster-plot
                                       Display spike raster plot in visualization.
      --brain_areas                    Brain areas to display in raster plot.
      --other                          Other spike cluster features to use for filtering.
      --raster_special_units           Clusters to accentuate in raster plot.
      --spike-sound / --no-spike-sound
                                       Play sound each time the cluster spikes.
      --beh-features / --no-beh-features
                                       Display behavioral feature dynamics.
      --beh_features_to_plot           Behavioral feature(s) to display.
      --special_beh_features           Behavioral feature(s) to accentuate in display.
      --fig_format                     Figure format.
      --fig_dpi                        Figure resolution in dots per inch.
      --animation_writer               Animation writer backend.
      --animation_format               Video format.
      --arena-node-connections / --no-arena-node-connections
                                       Display connections between arena nodes.
      --arena_axes_lw                  Line width for the arena axes.
      --arena_mics_lw                  Line width for the microphone markers.
      --arena_mics_opacity             Opacity for the microphone markers.
      --plot-corners / --no-plot-corners
                                       Display arena corner markers.
      --corner_size                    Size of the arena corner markers.
      --corner_opacity                 Opacity of the arena corner markers.
      --plot-mesh-walls / --no-plot-mesh-walls
                                       Display arena walls as a mesh.
      --mesh_opacity                   Opacity of the arena wall mesh.
      --active-mic / --no-active-mic   Display the active microphone marker.
      --inactive-mic / --no-inactive-mic
                                       Display inactive microphone markers.
      --inactive_mic_color             Color for inactive microphone markers.
      --text_fontsize                  Font size for text elements in the plot.
      --speaker_opacity                Opacity of the speaker node.
      --nodes / --no-nodes             Display mouse nodes.
      --node_size                      Size of the mouse nodes.
      --node_opacity                   Opacity of the mouse nodes.
      --node_lw                        Line width for the mouse node connections.
      --node_connection_lw             Line width for mouse node connections.
      --body_opacity                   Opacity of the mouse body.
      --history_point                  Node to use for the history trail.
      --history_span_sec               Duration of the history trail (s).
      --history_ls                     Line style for the history trail.
      --history_lw                     Line width for the history trail.
      --beh_features_window_size       Window size for behavioral features (s).
      --raster_window_size             Window size for the raster plot (s).
      --raster_lw                      Line width for spikes in the raster plot.
      --raster_ll                      Line length for spikes in the raster plot.
      --spectrogram-cbar / --no-spectrogram-cbar
                                       Display the color bar for the spectrogram.
      --spectrogram_plot_window_size   Window size for the spectrogram plot (s).
      --spectrogram_power_limit        Power (min/max) for spectrogram color scale.
      --spectrogram_frequency_limit    Freq. (min/max) for spectrogram y-axis (Hz).
      --spectrogram_yticks             Y-tick position for spectrogram.
      --spectrogram_stft_nfft          NFFT for the spectrogram STFT calculation.
      --plot-usv-segments / --no-plot-usv-segments
                                       Display USV assignments on the spectrogram.
      --usv_segments_ypos              Y-axis position for USV segment markers (Hz).
      --usv_segments_lw                Line width for USV segment markers.