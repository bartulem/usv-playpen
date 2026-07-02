"""
Standalone Kilosort4 spike-sorting runner (separate conda environment).

Description
-----------
A reference runner for spike-sorting a concatenated SpikeGLX AP binary with
Kilosort4. It is NOT part of the ``usv-playpen`` package and is never imported by
it: Kilosort4 (with a CUDA build of PyTorch) and the manual-curation GUI Phy2
(PyQt) have heavy, version-specific dependencies that conflict with the main
environment's pins, so each lives in its own conda environment. Create the
Kilosort one, for example::

    conda create -n kilosort python=3.11
    conda activate kilosort
    python -m pip install kilosort[gui]

    # GPU is strongly recommended. The line above pulls in a CPU PyTorch; replace
    # it with a CUDA build (otherwise stop here and sort on CPU):
    pip uninstall torch
    pip3 install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8

To find the CUDA version your machine supports, run ``nvidia-smi``: the
``CUDA Version: XX.X`` printed in the top-right is the highest CUDA the installed
NVIDIA driver supports. The same command works on Windows; on an HPC cluster run
it on a GPU node (login nodes usually have none), e.g. via
``srun --gres=gpu:1 --pty nvidia-smi``. Pick a PyTorch build at or below that and
swap the ``cu118`` tag in the index URL to match — e.g. newer cards such as the GeForce RTX
5000 series work best with CUDA 12.8, i.e. ``cu128`` (``.../whl/cu128``). The
PyTorch "Get Started" page lists the currently available tags.

Where it fits in the pipeline
-----------------------------
``probe_path`` below is exactly the ``*_kilosortChanMap.mat`` produced by
Neuropixels **Step 0** (``sglx_meta_to_coords`` / the ``npx-meta-to-coords`` GUI).
The ``kilosort4/`` directory this script writes (typically after manual Phy2
curation) is what the ``usv-playpen`` Neuropixels pipeline then consumes
(Steps 3-6 and ``SpikeQualityMetricsExtractor``).

Usage
-----
Edit the USER CONFIGURATION block, activate the ``kilosort`` env, then run::

    python run_kilosort.py

The ``settings`` dictionary below carries Kilosort4's defaults with inline notes,
so it doubles as a documented template — adjust per dataset as needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from kilosort.run_kilosort import run_kilosort

# =============================================================================
# --- 1. USER CONFIGURATION: EDIT THESE ---
# =============================================================================

# Root directory holding one ``{session_date}_{probe_id}/`` folder per recording.
EPHYS_DIR = Path("/path/to/ephys")

# Probe ID (SpikeGLX label, e.g. 'imec0' / 'imec1').
probe_id = 'imec1'

# Session date (YYYYMMDD).
session_date = '20251011'

# REQUIRED: The total number of channels in your binary file.
n_chan_bin = 385

# bad channels (list if any)
bad_channels_ = None

# Sampling rate of the headstage (the probe's calibrated rate, e.g. from
# _config/calibrated_sample_rates_imec.ini).
fs = 30000.207531380755

# REQUIRED: The full path to your raw binary data file (.bin, .dat, etc.)
data_file = EPHYS_DIR / f'{session_date}_{probe_id}' / f'concatenated_{session_date}_{probe_id}.ap.bin'

# REQUIRED: Channel map for your binary file (the *_kilosortChanMap.mat from Step 0).
probe_path = EPHYS_DIR / f'{session_date}_{probe_id}' / f'concatenated_{session_date}_{probe_id}.ap_kilosortChanMap.mat'

# OPTIONAL: The directory where Kilosort will save the results.
results_dir = data_file.parent / 'kilosort4'

# =============================================================================
# --- 2. KILOSORT SETTINGS DICTIONARY ---
# =============================================================================

settings = {
    # # # --- Main Parameters ---

    # Total number of channels in the binary file. MUST BE SPECIFIED BY THE USER.
    'n_chan_bin': n_chan_bin,

    # Channel map for probe
    'probe_path': probe_path,

    # Sampling frequency of the probe in Hz.
    'fs': fs,

    # Number of samples included in each batch of data for processing.
    'batch_size': 60000,

    # Number of non-overlapping blocks for drift correction.
    'nblocks': 1,

    # Spike detection threshold for universal templates (pre-learned, generic shapes).
    'Th_universal': 9.0,

    # Spike detection threshold for templates learned from the data.
    'Th_learned': 8.0,

    # Time in seconds to start sorting. Default is 0 (the beginning of the file).
    'tmin': 0.0,

    # Time in seconds to end sorting. Default is np.inf (the end of the file).
    'tmax': np.inf,

    # # # --- Extra Parameters ---

    # Number of samples per waveform.
    'nt': 61,

    # Scalar shift to apply to data (e.g., for float32 data). `data = data*scale + shift`.
    'shift': None,

    # Scaling factor to apply to data (e.g., for float32 data). `data = data*scale + shift`.
    'scale': None,

    # Subsampling factor for batches. e.g., 10 means only every 10th batch is used. Default is 1 (use all batches).
    'batch_downsampling': 1,

    # --- Preprocessing ---
    # Batches with absolute values above this threshold will be zeroed out as artifacts. Default is infinite (no artifact removal).
    'artifact_threshold': np.inf,

    # Batch stride for computing the whitening matrix.
    'nskip': 25,

    # Number of nearby channels used to estimate the whitening matrix.
    'whitening_range': 32,

    # High-pass filter cutoff frequency in Hz.
    'highpass_cutoff': 300.0,

    # Vertical bin size in microns for 2D histogram in drift correction.
    'binning_depth': 5.0,

    # Approximate spatial smoothness scale in microns for drift correction.
    'sig_interp': 20.0,

    # Gaussian smoothing for drift estimation [correlation, time, y-axis].
    'drift_smoothing': [0.5, 0.5, 0.5],

    # --- Spike Detection ---
    # Sample index for aligning waveforms. If None, it's determined automatically.
    'nt0min': None,

    # Vertical spacing of template centers in microns. If None, it's determined automatically.
    'dmin': None,

    # Horizontal spacing of template centers in microns.
    'dminx': 32.0,

    # Standard deviation of the smallest Gaussian for universal templates (in microns).
    'min_template_size': 10.0,

    # Number of sizes for universal spike templates.
    'template_sizes': 5,

    # Number of nearest channels for finding local maxima during spike detection.
    'nearest_chans': 10,

    # Number of nearest templates for finding local maxima during spike detection.
    'nearest_templates': 100,

    # Templates farther than this (in microns) from their nearest channel will be ignored.
    'max_channel_distance': 32.0,

    # Number of matching pursuit iterations per batch. More peels can find more overlapping spikes.
    'max_peels': 100,

    # If True, universal templates are estimated from the data.
    'templates_from_data': True,

    # Number of single-channel templates to learn (if templates_from_data is True).
    'n_templates': 6,

    # Number of principal components for spike features (if templates_from_data is True).
    'n_pcs': 6,

    # Threshold in standard deviations for single-channel crossings to compute universal templates.
    'Th_single_ch': 6.0,

    # --- Clustering ---
    # Allowed fraction of refractory violations in ACG for a unit to be labeled "good".
    'acg_threshold': 0.2,

    # Allowed fraction of refractory violations in CCG for performing splits/merges.
    'ccg_threshold': 0.25,

    # Number of nearest spike neighbors to use for building the clustering graph.
    'cluster_neighbors': 10,

    # Inverse fraction of spikes used as landmarks during clustering. Default uses all up to a max.
    'cluster_downsampling': 1,

    # Maximum number of spikes to use for building the clustering graph.
    'max_cluster_subset': 25000,

    # Number of x-positions for template groupings. If None, determined automatically.
    'x_centers': None,

    # Random seed for the kmeans++ clustering initialization.
    'cluster_init_seed': 5,

    # --- Postprocessing ---
    # Time in ms to consider subsequent spikes from the same cluster as duplicates.
    'duplicate_spike_ms': 0.25,

    # Maximum distance in microns between channels used for estimating spike positions.
    'position_limit': 100.0,
}


# =============================================================================
# --- 3. RUN KILOSORT ---
# =============================================================================

if __name__ == '__main__':

    results_dir.mkdir(exist_ok=True, parents=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("No GPU found, using CPU.")  # noqa: T201

    run_kilosort(
        settings,
        filename=data_file,
        data_dtype='int16',
        bad_channels=bad_channels_,
        results_dir=results_dir,
        verbose_console=True,
        verbose_log=True,
        clear_cache=False,
        device=device,
    )
