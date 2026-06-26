"""
@author: bartulem
Enables analysis of neuronal coactivity during vocalization events, including
pairwise spike count correlations and population vector similarity, with
bootstrapping and circular shuffle controls for statistical validation.
"""

import pathlib
import re
from collections import defaultdict
from typing import Any

import h5py
import librosa
import numpy as np
import polars as pls
from sklearn.metrics.pairwise import cosine_similarity

from ..os_utils import first_match_or_raise


def extract_snippet_matrix(
    onsets: np.ndarray,
    neural_data: dict[str, np.ndarray],
    window_s: float
) -> np.ndarray:
    """
    Extracts a Neurons x Trials matrix of spike counts for a fixed
    temporal window relative to event onsets.

    Parameters
    ----------
    onsets : np.ndarray
        1D array of event start times in seconds.
    neural_data : Dict[str, np.ndarray]
        Dictionary of spike times per neuron.
    window_s : float
        The length of the analysis window in seconds.

    Returns
    -------
    np.ndarray
        Matrix of shape (N_neurons, N_trials) containing spike counts.
    """

    neuron_keys = list(neural_data.keys())
    num_neurons = len(neuron_keys)
    num_trials = len(onsets)
    count_matrix = np.zeros((num_neurons, num_trials), dtype=np.float64)
    if num_trials == 0:
        return count_matrix
    onsets_arr = np.asarray(onsets, dtype=np.float64)
    ends_arr = onsets_arr + window_s

    # Spike arrays are sorted on disk (Kilosort output) and
    # `apply_circular_shift` re-sorts after wraparound, so we can use
    # binary search to count spikes in each window in O(log N_spikes)
    # per onset, vectorised across the whole onset vector at once.
    # The previous double-loop was O(N_neurons * N_trials * N_spikes)
    # which dominated chained-shuffle / bootstrap runtimes.
    for n_idx, n_key in enumerate(neuron_keys):
        spikes = neural_data[n_key]
        lo = np.searchsorted(spikes, onsets_arr, side="left")
        hi = np.searchsorted(spikes, ends_arr, side="right")
        count_matrix[n_idx, :] = hi - lo

    return count_matrix


def extract_snippet_acoustics(
    session_root: str,
    onsets: np.ndarray,
    peak_channels: np.ndarray,
    window_s: float,
    *,
    nperseg: int = 2048,
    hop_length: int = 512,
    window: str = "blackmanharris",
    min_freq: float = 30000.0,
    max_freq: float = 120000.0,
    low_energy_frac: float = 0.05,
    high_energy_frac: float = 0.95,
) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    Per-USV acoustic features over a FIXED ``window_s`` window from each call's
    onset -- the audio analogue of :func:`extract_snippet_matrix` (which counts
    spikes in the same window). For each onset the loudest-channel waveform
    snippet ``[onset, onset + window_s)`` is read from the session's concatenated
    ``*_int16.mmap`` audio and reduced to:

    - ``rms``             : absolute loudness, ``sqrt(mean(x**2))`` of the int16 ->
                            float (``/32767``) samples. This is an ABSOLUTE measure,
                            unlike the cohort spectrogram features, which are per-USV
                            ``power_to_db(ref=max)`` + min-max normalized and so
                            cannot express loudness.
    - ``mean_freq_hz``    : energy-weighted mean frequency (spectral centroid).
    - ``peak_freq_hz``    : frequency of the loudest spectrogram bin.
    - ``freq_bandwidth_hz``: span between the ``low_energy_frac`` and
                            ``high_energy_frac`` cumulative-energy frequency
                            crossings.

    The frequency features are energy-weighted (the textbook spectral centroid /
    spread) over the snippet's LINEAR power STFT, masked to ``[min_freq, max_freq]``.
    This deliberately differs from the cohort feature extractor
    (``compute_usv_acoustic_features.compute_acoustic_features``), which weights a
    ``power_to_db(ref=np.max)`` + min-max normalized spectrogram: that form relies on
    a per-USV region mask to suppress the noise floor, and these maskless 30 ms
    snippets have none, so linear power gives a sharper, physically meaningful
    measure (``peak_freq`` is identical either way). Amplitude is taken from the raw
    waveform because the cohort spectrogram normalization cannot express absolute
    loudness.

    Used to test (and later control for) acoustic confounds between vocal-category
    groups in the coactivity analysis: the fixed-duration window already equalizes
    call DURATION; these features expose any residual AMPLITUDE / FREQUENCY
    differences over the identical window.

    Parameters
    ----------
    session_root : str
        Session root directory; the ``*_int16.mmap`` audio is found recursively
        beneath it.
    onsets : np.ndarray
        1D array of call onset times in seconds.
    peak_channels : np.ndarray
        1D array (length matching ``onsets``) of the loudest 0-indexed audio
        channel per call (the ``peak_amp_ch`` column; floats are cast to int,
        non-finite -> channel 0).
    window_s : float
        Window length in seconds (e.g. 0.030), measured from each onset.
    nperseg : int
        STFT window length (samples). Default 2048 (generate-spectrograms config).
    hop_length : int
        STFT hop (samples). Default 512.
    window : str
        STFT window name. Default 'blackmanharris'.
    min_freq : float
        Lower frequency bound (Hz) the STFT is masked to. Default 30000.
    max_freq : float
        Upper frequency bound (Hz). Default 120000.
    low_energy_frac : float
        Lower cumulative-energy fraction for the bandwidth crossing. Default 0.05.
    high_energy_frac : float
        Upper cumulative-energy fraction for the bandwidth crossing. Default 0.95.

    Returns
    -------
    dict[str, np.ndarray]
        ``{'rms', 'mean_freq_hz', 'peak_freq_hz', 'freq_bandwidth_hz'}``, each a
        length-N float64 array (NaN where a snippet is empty, or too short for one
        STFT window in the case of the frequency features).
    """

    onsets_arr = np.asarray(onsets, dtype=np.float64)
    chans_arr = np.asarray(peak_channels, dtype=np.float64)
    num_trials = onsets_arr.shape[0]
    out = {
        key: np.full(num_trials, np.nan, dtype=np.float64)
        for key in ("rms", "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz")
    }
    if num_trials == 0:
        return out
    if chans_arr.shape[0] != num_trials:
        msg = (
            f"peak_channels length ({chans_arr.shape[0]}) must match onsets length "
            f"({num_trials})."
        )
        raise ValueError(msg)

    # Locate + memmap the concatenated int16 audio (searched recursively under the
    # session root, matching the spectrogram pipeline); the per-OS sample rate,
    # sample count and channel count are encoded in the trailing
    # ``_<sr>_<n_samples>_<n_ch>_int16.mmap`` filename segment.
    audio_path = first_match_or_raise(
        root=pathlib.Path(session_root),
        pattern="*_int16.mmap*",
        recursive=True,
        label="concatenated int16 audio memmap",
    )
    meta = re.search(r"_(?P<sr>\d+)_(?P<n_samples>\d+)_(?P<n_ch>\d+)_int16\.mmap", audio_path.name)
    if meta is None:
        msg = (
            f"Could not parse the '_<sr>_<n_samples>_<n_ch>_int16.mmap' segment from "
            f"audio memmap name {audio_path.name!r}."
        )
        raise ValueError(msg)
    sampling_rate = int(meta["sr"])
    n_samples = int(meta["n_samples"])
    n_channels = int(meta["n_ch"])
    handle = np.memmap(audio_path, dtype=np.int16, mode="r", shape=(n_samples, n_channels), order="C")

    window_samples = round(window_s * sampling_rate)
    freqs_full = librosa.fft_frequencies(sr=sampling_rate, n_fft=nperseg)
    band = (freqs_full >= min_freq) & (freqs_full <= max_freq)
    band_freqs = freqs_full[band]
    n_band = band_freqs.shape[0]
    eps = 1e-8

    for trial_idx in range(num_trials):
        s0 = round(float(onsets_arr[trial_idx]) * sampling_rate)
        s1 = min(s0 + window_samples, n_samples)
        if s0 < 0 or s1 <= s0:
            continue  # out of range -> leave NaN
        ch_val = chans_arr[trial_idx]
        ch = round(float(ch_val)) if np.isfinite(ch_val) else 0
        ch = min(max(ch, 0), n_channels - 1)
        seg = handle[s0:s1, ch].astype(np.float64) / 32767.0
        if seg.size == 0:
            continue

        # ABSOLUTE amplitude from the raw waveform.
        out["rms"][trial_idx] = float(np.sqrt(np.mean(seg ** 2)))

        # FREQUENCY features need at least one full STFT window of real signal.
        if seg.size < nperseg:
            continue
        power = np.abs(
            librosa.stft(
                seg, n_fft=nperseg, hop_length=hop_length,
                win_length=nperseg, window=window, center=True,
            )
        ) ** 2
        power = power[band]
        if power.size == 0 or power.shape[1] == 0:
            continue
        total = power.sum()
        if total <= eps:
            continue

        # Energy-weighted (linear-power) spectral features. These are the textbook
        # spectral centroid / spread. We deliberately do NOT use the cohort's
        # dB(ref=max) + min-max normalization here: that form relies on a per-USV
        # region mask to suppress the noise floor, and these maskless snippets have
        # none, so a lifted floor would dominate the cumulative energy. Linear power
        # gives a sharper, physically meaningful measure (peak_freq is identical).
        #
        # peak frequency: freq row of the loudest (freq, time) power bin
        out["peak_freq_hz"][trial_idx] = float(band_freqs[int(power.argmax() // power.shape[1])])
        # mean frequency: energy-weighted spectral centroid
        out["mean_freq_hz"][trial_idx] = float((power * band_freqs[:, None]).sum() / (total + eps))
        # bandwidth: span between the low/high cumulative-energy crossings of the
        # per-frequency (time-summed) linear-power profile
        row_power = power.sum(axis=1)
        row_norm = row_power / (row_power.sum() + eps)
        cumsum = np.cumsum(row_norm)
        low_bin = min(int((cumsum < low_energy_frac).sum()), n_band - 1)
        high_bin = min(int((cumsum < high_energy_frac).sum()), n_band - 1)
        out["freq_bandwidth_hz"][trial_idx] = float(band_freqs[high_bin] - band_freqs[low_bin])

    return out


def compute_coactivity_metrics(count_matrix: np.ndarray) -> dict[str, Any]:
    """
    Calculates three metrics of ensemble coordination:
    1. r_sc: Mean pairwise correlation (neurons across trials).
    2. similarity: Mean cosine similarity (population vectors across trials).
    3. pop_corr: Mean Pearson correlation (population vectors across trials).

    Parameters
    ----------
    count_matrix : np.ndarray
        Matrix of shape (N_neurons, N_trials).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'r_sc', 'similarity', and 'pop_corr'.
    """

    num_neurons, num_trials = count_matrix.shape
    if num_neurons < 2:
        return {"r_sc": np.nan, "similarity": np.nan, "pop_corr": np.nan}

    # Pairwise spike count correlation (r_sc)
    # Correlation between neurons (rows) across trials
    with np.errstate(divide='ignore', invalid='ignore'):
        neuron_corr_mat = np.corrcoef(count_matrix)

    r_sc_values = neuron_corr_mat[np.triu_indices(num_neurons, k=1)]
    mean_r_sc = np.nanmean(r_sc_values)

    # Population vector similarity (cosine similarity)
    # Angle between raw population vectors (columns)
    # Transpose to (trials, neurons) for cosine_similarity
    sim_matrix = cosine_similarity(count_matrix.T)
    pop_sim_values = sim_matrix[np.triu_indices(num_trials, k=1)]
    mean_pop_sim = np.nanmean(pop_sim_values)

    # Correlation of population vectors (Pearson correlation)
    # This is the "mean-centered" version of cosine similarity.
    # It removes the average firing rate (magnitude) of each trial.
    with np.errstate(divide='ignore', invalid='ignore'):
        # rowvar=False correlates the columns (trials)
        trial_corr_mat = np.corrcoef(count_matrix, rowvar=False)

    pop_corr_values = trial_corr_mat[np.triu_indices(num_trials, k=1)]
    mean_pop_corr = np.nanmean(pop_corr_values)

    return {
        "r_sc": mean_r_sc,
        "similarity": mean_pop_sim,
        "pop_corr": mean_pop_corr
    }

def bootstrap_coactivity_distribution(
    count_matrix: np.ndarray,
    n_target: int,
    n_iterations: int = 1000,
    seed: int | None = None
) -> dict[str, np.ndarray]:
    """
    Performs trial-level bootstrap resampling of a coactivity count
    matrix to build an empirical distribution of each metric at a
    fixed `n_target`. Used to equalise trial counts between two
    vocalization groups before comparing their coactivity, so the
    statistical power for both is identical.

    For each iteration the function draws `n_target` trial indices
    with replacement, slices the input matrix to those columns, and
    evaluates `compute_coactivity_metrics` on the resample. The
    per-iteration metric values are accumulated into per-metric
    arrays which the caller can summarise as means, percentiles or
    confidence intervals.

    Parameters
    ----------
    count_matrix : np.ndarray
        Neural spike-count matrix for the group, shape
        (N_neurons x N_trials). Earlier revisions named this
        argument `simple_counts` (back when the analysis only
        bootstrapped the "simple" call group); the function has
        always been group-agnostic and works on any count matrix.
    n_target : int
        The target trial count per resample. Typically the size of
        the smaller of the two groups being compared, so the larger
        group is sampled down to match.
    n_iterations : int, optional
        Number of bootstrap iterations. Defaults to 1000.
    seed : int | None, optional
        Seed for the NumPy random generator driving the trial
        resampling. Pass a fixed integer to make the bootstrap
        distribution reproducible across runs; leave as `None`
        (the default) for fresh entropy on every call.

    Returns
    -------
    dict[str, np.ndarray]
        Per-metric bootstrap distributions:
        * `r_sc`        — mean pairwise neuron spike-count correlations
        * `similarity`  — mean population-vector cosine similarities
        * `pop_corr`    — mean population-vector Pearson correlations
    """

    boot_rsc = np.zeros(n_iterations)
    boot_sim = np.zeros(n_iterations)
    boot_pop = np.zeros(n_iterations)

    num_trials = count_matrix.shape[1]
    rng = np.random.default_rng(seed)

    for i in range(n_iterations):
        # Sample trial indices with replacement to match n_target
        idx = rng.choice(num_trials, n_target, replace=True)
        resampled_matrix = count_matrix[:, idx]

        # Calculate coactivity metrics for the resampled matrix
        metrics = compute_coactivity_metrics(resampled_matrix)

        # Store results for all three metrics
        boot_rsc[i] = metrics["r_sc"]
        boot_sim[i] = metrics["similarity"]
        boot_pop[i] = metrics["pop_corr"]

    return {
        "r_sc": boot_rsc,
        "similarity": boot_sim,
        "pop_corr": boot_pop
    }

def apply_circular_shift(
    neural_data: dict[str, np.ndarray],
    shift_s: float,
    total_duration_s: float
) -> dict[str, np.ndarray]:
    """
    Applies a uniform circular temporal shift to a population of neurons.

    This function shifts every spike time in the population by a constant
    offset and wraps any spikes that exceed the session duration back to
    the beginning. This 'joint' shift preserves the relative timing (and
    thus the correlations) between neurons while decoupling the neural
    activity from the external behavioral timestamps.

    Parameters
    ----------
    neural_data : dict[str, np.ndarray]
        Dictionary where keys are unique neuron identifiers and values
        are 1D NumPy arrays containing spike times in seconds.
    shift_s : float
        The temporal offset in seconds to be added to every spike time.
    total_duration_s : float
        The total recording session length in seconds, used as the
        modulus for the circular wrap-around.

    Returns
    -------
    dict[str, np.ndarray]
        A new dictionary containing the shifted and wrapped spike times.
        Each spike array is re-sorted to maintain chronological order,
        ensuring compatibility with window-extraction algorithms.
    """

    shifted_data = {}
    # The input spike arrays are already sorted (Kilosort output, see the binary-
    # search helpers above), so a circular shift is a ROTATION: the spikes that
    # wrap past the session end (>= total_duration_s - shift_s) move to the front
    # in [0, shift_s), and the rest move to [shift_s, total_duration_s). Splitting
    # at the wrap point and concatenating the two already-sorted halves is O(log N)
    # per neuron instead of an O(N log N) np.sort -- and this runs n_shuffles
    # (default 1000) times per session.
    shift_s = shift_s % total_duration_s
    for n_id, spikes in neural_data.items():
        split = np.searchsorted(spikes, total_duration_s - shift_s, side='left')
        shifted_data[n_id] = np.concatenate([
            (spikes[split:] + shift_s) - total_duration_s,
            spikes[:split] + shift_s,
        ])
    return shifted_data

def perform_circular_shuffle(
    onsets: np.ndarray,
    neural_data: dict[str, np.ndarray],
    total_duration: float,
    window_s: float,
    min_shift_s: float = 20.0,
    max_shift_s: float = 60.0,
    n_shuffles: int = 1000,
    seed: int | None = None
) -> dict[str, np.ndarray]:
    """
    Orchestrates a joint circular temporal shuffle to generate null
    distributions for coactivity metrics.

    By shifting the entire neural population as a single unit, this
    method maintains the intrinsic 'idling' correlation structure of
    the network. It tests whether the observed alignment during
    behavioral onsets is significantly greater than what would be
    observed if the same neural patterns occurred at random times
    within the session.

    Parameters
    ----------
    onsets : np.ndarray
        1D array of behavioral event start times (e.g., USV onsets)
        in seconds.
    neural_data : dict[str, np.ndarray]
        Dictionary of neuron IDs and their corresponding spike time arrays.
    total_duration : float
        The total duration of the recording session in seconds.
    window_s : float
        The duration of the analysis window in seconds (e.g., 0.030 for 30ms).
    min_shift_s : float, optional
        The minimum allowable random time shift, by default 20.0.
    max_shift_s : float, optional
        The maximum allowable random time shift, by default 60.0.
    n_shuffles : int, optional
        The number of shuffle iterations to perform, by default 1000.
    seed : int | None, optional
        Seed for the NumPy random generator that draws the per-shuffle
        circular offsets. Pass a fixed integer to make the null
        distribution reproducible across runs; leave as `None` (the
        default) for fresh entropy on every call.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where each key corresponds to a metric returned
        by compute_coactivity_metrics (e.g., 'r_sc', 'similarity',
        'pop_corr'). Values are NumPy arrays of length n_shuffles,
        representing the empirical null distribution for that metric.
    """

    # Initialize storage based on keys from a test calculation
    # We run a dummy calculation to dynamically identify available metrics
    sample_matrix = extract_snippet_matrix(onsets, neural_data, window_s)
    metric_keys = compute_coactivity_metrics(sample_matrix).keys()

    results = {key: np.zeros(n_shuffles) for key in metric_keys}

    rng = np.random.default_rng(seed)

    for i in range(n_shuffles):
        # 1. Generate a single random shift for the entire population
        shift = rng.uniform(min_shift_s, max_shift_s)

        # 2. Shift the spikes
        shifted_neural_data = apply_circular_shift(
            neural_data,
            shift,
            total_duration
        )

        # 3. Extract matrix and compute all metrics for this shuffle
        shuffled_matrix = extract_snippet_matrix(
            onsets,
            shifted_neural_data,
            window_s
        )
        metrics = compute_coactivity_metrics(shuffled_matrix)

        # 4. Record the results
        for key in results.keys():
            results[key][i] = metrics[key]

    return results

def perform_chained_circular_shuffle(
    session_onsets: list[np.ndarray],
    session_neural_data: list[dict[str, np.ndarray]],
    session_durations: list[float],
    window_s: float,
    min_shift_s: float = 20.0,
    max_shift_s: float = 60.0,
    n_shuffles: int = 1000,
    seed: int | None = None
) -> dict[str, np.ndarray]:
    """
    Performs a circular shuffle across multiple independent sessions.

    In each iteration, every session is shifted by its own random offset
    to preserve internal timing. The resulting shuffled matrices from all
    sessions are concatenated before coactivity metrics are calculated,
    providing a global null distribution for the chained dataset.

    Parameters
    ----------
    session_onsets : list[np.ndarray]
        List of onset arrays, one per session.
    session_neural_data : list[dict]
        List of neural data dictionaries, one per session.
    session_durations : list[float]
        List of total durations (seconds) for each session.
    window_s : float
        Analysis window size in seconds.
    min_shift_s : float, optional
        The minimum allowable random time shift, by default 20.0.
    max_shift_s : float, optional
        The maximum allowable random time shift, by default 60.0.
    n_shuffles : int, optional
        Number of global shuffle iterations, by default 1000.
    seed : int | None, optional
        Seed for the NumPy random generator that draws each session's
        per-shuffle circular offset. Pass a fixed integer to make the
        chained null distribution reproducible across runs; leave as
        `None` (the default) for fresh entropy on every call.

    Returns
    -------
    dict[str, np.ndarray]
        Empirical null distributions for the concatenated sessions.
    """
    # 1. Initialize results by checking metric keys from a dummy run
    dummy_matrices = []
    for onsets, neural, _duration in zip(session_onsets, session_neural_data, session_durations):
        # We only need a small slice to get the keys (durations are unused here;
        # they are only needed in the real shuffle loop below).
        dummy_matrices.append(extract_snippet_matrix(onsets[:2], neural, window_s))

    combined_dummy = np.hstack(dummy_matrices)
    metric_keys = compute_coactivity_metrics(combined_dummy).keys()
    results = {key: np.zeros(n_shuffles) for key in metric_keys}

    rng = np.random.default_rng(seed)

    for i in range(n_shuffles):
        shuffled_mats_to_combine = []

        for onsets, neural, duration in zip(session_onsets, session_neural_data, session_durations):
            # 2. Shift and extract for THIS session
            shift = rng.uniform(min_shift_s, max_shift_s)
            shifted_neural = apply_circular_shift(neural, shift, duration)
            shuff_mat = extract_snippet_matrix(onsets, shifted_neural, window_s)
            shuffled_mats_to_combine.append(shuff_mat)

        # 3. Concatenate all session shuffles into one global matrix
        global_shuffled_matrix = np.hstack(shuffled_mats_to_combine)

        # 4. Compute metrics on the aggregate
        metrics = compute_coactivity_metrics(global_shuffled_matrix)
        for key in results.keys():
            results[key][i] = metrics[key]

    return results

def perform_label_permutation_test(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Direct two-group comparison of coactivity via trial-label permutation.

    Pools the trial columns of `counts_a` and `counts_b` into a single
    matrix, then on each iteration randomly partitions them back into
    pseudo-groups of the original sizes and computes
    `metric(pseudo_a) - metric(pseudo_b)`. This builds a null
    distribution of the group-difference under the hypothesis that
    the two trial labels are interchangeable. The observed difference
    `metric(a) - metric(b)` is then placed against this null to test
    whether the two groups differ in coactivity (right-tailed `a > b`
    p-value plus a two-tailed p-value).

    Use this when the bootstrap-vs-shuffle test only gives per-group
    significance against a within-group null — it does NOT directly
    test whether the two groups differ from each other.

    Parameters
    ----------
    counts_a, counts_b : np.ndarray
        Per-group spike-count matrices, both shape (N_neurons,
        N_trials_group). The neuron axis must agree (same population
        of cells); only the trial axis differs.
    n_permutations : int, optional
        Number of trial-label permutations. Defaults to 1000.
    seed : int | None, optional
        Seed for the NumPy random generator that permutes the pooled
        trial labels. Pass a fixed integer to make the permutation
        null (and therefore the p-values / z-scores) reproducible
        across runs; leave as `None` (the default) for fresh entropy
        on every call.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping `metric -> {observed_delta, null, null_mean, null_std,
        p_a_gt_b, p_two_tailed, z_score}`. `metric` covers every key returned
        by `compute_coactivity_metrics` (`r_sc`, `similarity`,
        `pop_corr`).
    """

    n_a = counts_a.shape[1]
    combined = np.hstack([counts_a, counts_b])
    n_total = combined.shape[1]

    # Observed per-group metrics + delta.
    m_a = compute_coactivity_metrics(counts_a)
    m_b = compute_coactivity_metrics(counts_b)
    observed_delta = {k: m_a[k] - m_b[k] for k in m_a}

    # Null distribution: shuffle trial labels, recompute delta.
    null_dists = {k: np.zeros(n_permutations) for k in m_a}
    rng = np.random.default_rng(seed)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        idx_a = perm[:n_a]
        idx_b = perm[n_a:]
        pa = compute_coactivity_metrics(combined[:, idx_a])
        pb = compute_coactivity_metrics(combined[:, idx_b])
        for k in null_dists:
            null_dists[k][i] = pa[k] - pb[k]

    results: dict[str, dict[str, Any]] = {}
    for k, null in null_dists.items():
        obs = observed_delta[k]
        null_mean = float(np.nanmean(null))
        null_std = float(np.nanstd(null))
        # Monte-Carlo permutation p-values with the standard +1/(n+1) bias
        # correction: the observed statistic is itself one valid arrangement
        # of the labels, so the count and total each gain 1. This keeps the
        # p-value from ever being exactly 0 for a finite permutation count
        # (which would be anticonservative).
        n_perm = int(null.size)
        p_a_gt_b = float((np.sum(null >= obs) + 1) / (n_perm + 1))   # right-tailed: a > b
        p_two = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
        z = (obs - null_mean) / null_std if null_std > 0 else 0.0
        results[k] = {
            "observed_delta": float(obs),
            "null": null,
            "null_mean": null_mean,
            "null_std": null_std,
            "p_a_gt_b": p_a_gt_b,
            "p_two_tailed": p_two,
            "z_score": float(z),
        }
    return results


def compute_sliding_coactivity(
    onsets: np.ndarray,
    neural_data: dict[str, np.ndarray],
    window_s: float,
    step_s: float,
    n_steps: int
) -> dict[str, np.ndarray]:
    """
    Computes coactivity metrics across a series of sliding windows relative
    to event onsets to track temporal dynamics.

    Parameters
    ----------
    onsets : np.ndarray
        1D array of event start times.
    neural_data : dict[str, np.ndarray]
        Spike times per neuron.
    window_s : float
        Width of each analysis window.
    step_s : float
        Step size between windows (determines overlap).
    n_steps : int
        Number of steps to slide forward.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with 'time_bins' (per-step onset offsets in seconds),
        'r_sc', and 'similarity', each a length-n_steps array.
    """

    sliding_rsc = np.zeros(n_steps)
    sliding_sim = np.zeros(n_steps)
    time_bins = np.arange(n_steps) * step_s

    for i, offset in enumerate(time_bins):
        # Shift the onsets forward for this specific bin
        current_onsets = onsets + offset

        # Reuse the same snippet extraction and metric computation per bin.
        matrix = extract_snippet_matrix(current_onsets, neural_data, window_s)
        metrics = compute_coactivity_metrics(matrix)

        sliding_rsc[i] = metrics["r_sc"]
        sliding_sim[i] = metrics["similarity"]

    return {
        "time_bins": time_bins,
        "r_sc": sliding_rsc,
        "similarity": sliding_sim
    }

def sample_onsets_across_sessions(
    sessions_list: list[dict[str, Any]],
    category_key: str,
    n_total: int,
    seed: int | None = None
) -> list[np.ndarray]:
    """
    Samples a total of N onsets across multiple sessions, maintaining
    session identity for subsequent circular shifting.

    This function pools all available onsets from a specific category
    (e.g., 'group_a_df') across all provided sessions, draws a random
    subset of size n_total, and returns them as a list of arrays
    corresponding to the original session order.

    Parameters
    ----------
    sessions_list : list[dict]
        The list of session data dictionaries created during loading.
    category_key : str
        The key in the session dictionary to pull onsets from
        (e.g., 'group_a_df' or 'group_b_df').
    n_total : int
        The total number of onsets to sample across the entire dataset.
    seed : int | None, optional
        Seed for the NumPy random generator that draws the onset
        subset from the global pool. Pass a fixed integer to make the
        sampled onset set reproducible across runs; leave as `None`
        (the default) for fresh entropy on every call.

    Returns
    -------
    list[np.ndarray]
        A list of onset arrays, one per session. If a session contributed
        no onsets to the sample, its entry will be an empty array.
    """
    all_indices = []

    # 1. Create a global pool of (session_index, onset_time)
    for s_idx, sess in enumerate(sessions_list):
        onsets = sess[category_key]['start'].to_numpy()
        for t in onsets:
            all_indices.append((s_idx, t))

    if len(all_indices) < n_total:
        msg = (f"Total available {category_key} onsets ({len(all_indices)}) "
               f"is less than target N ({n_total}).")
        raise ValueError(msg)

    # 2. Sample N unique onsets from the global pool
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(all_indices), size=n_total, replace=False)
    sampled_points = [all_indices[i] for i in selected_indices]

    # 3. Redistribute sampled onsets back into session buckets
    session_buckets = [[] for _ in range(len(sessions_list))]
    for s_idx, t in sampled_points:
        session_buckets[s_idx].append(t)

    # 4. Convert to sorted numpy arrays
    return [np.sort(np.array(bucket)) for bucket in session_buckets]


def load_unit_catalog(catalog_path: str | pathlib.Path) -> dict[tuple[str, str, str], dict]:
    """
    Description
    -----------
    Reads the unit catalog CSV into a ``(mouse_id, rec_date, unit_id) -> row``
    lookup so per-session unit filtering is an O(1) dictionary access. The catalog
    is keyed by recording date, so every session recorded on a given date for a
    mouse shares the same row set.

    Parameters
    ----------
    catalog_path : str | pathlib.Path
        Path to ``unit_catalog.csv``.

    Returns
    -------
    catalog (dict[tuple[str, str, str], dict])
        Lookup keyed by ``(mouse_id, rec_date, unit_id)`` (all strings) with the raw
        CSV row dict as value.
    """

    # `infer_schema_length=0` reads every column as a string, so the
    # `(mouse_id, rec_date, unit_id)` keys match the string file stems and the
    # downstream string comparisons in `filter_units_by_catalog` hold.
    catalog_df = pls.read_csv(catalog_path, infer_schema_length=0)
    return {
        (row["mouse_id"], row["rec_date"], row["unit_id"]): row
        for row in catalog_df.iter_rows(named=True)
    }


def filter_units_by_catalog(
    mouse_id: str,
    rec_date: str,
    file_stems: set[str],
    catalog: dict[tuple[str, str, str], dict],
    *,
    cluster_group: str,
    require_somatic: bool,
    brain_areas: set[str],
) -> set[str]:
    """
    Description
    -----------
    Keeps only those ``file_stems`` whose catalog row passes the three configured
    filters: ``cluster_group``, ``somatic`` (when ``require_somatic``), and
    ``brain_area`` (when ``brain_areas`` is non-empty). Stems without a catalog row
    are silently dropped.

    Parameters
    ----------
    mouse_id : str
        Focal-mouse ID.
    rec_date : str
        Recording date in ``YYYYMMDD`` form.
    file_stems : set[str]
        Cluster file stems found in the session directory.
    catalog : dict[tuple[str, str, str], dict]
        Output of :func:`load_unit_catalog`.
    cluster_group : str
        Required Kilosort/Phy ``cluster_group`` label (e.g. ``"good"``).
    require_somatic : bool
        If True, keep only rows whose ``somatic`` column is truthy.
    brain_areas : set[str]
        Allowed ``brain_area`` labels; an empty set disables the brain-area filter.

    Returns
    -------
    kept (set[str])
        Subset of ``file_stems`` passing every filter.
    """

    kept: set[str] = set()
    for stem in file_stems:
        row = catalog.get((mouse_id, rec_date, stem))
        if row is None:
            continue
        if row["cluster_group"] != cluster_group:
            continue
        if require_somatic and str(row["somatic"]).strip().lower() != "true":
            continue
        if brain_areas and row["brain_area"] not in brain_areas:
            continue
        kept.add(stem)
    return kept


def load_animal_sessions(
    animal_id: str,
    session_names: list[str],
    *,
    data_root: pathlib.Path,
    catalog: dict[tuple[str, str, str], dict],
    category_column: str,
    group_a_ids: list,
    group_b_ids: list,
    cluster_group: str,
    require_somatic: bool,
    brain_areas: set[str],
    message_output: Any = None,
) -> list[dict]:
    """
    Description
    -----------
    Builds the ``sessions_data`` list for one focal mouse, restricted to the SINGLE
    recording day with the largest catalog-filtered unit pool -- so the neural
    population is fixed across the sessions actually analysed AND the probe is in a
    requested brain area. Kilosort is run per day, so cluster IDs are not stable
    across days; the loader therefore never mixes days. Each candidate date is scored
    by the size of the catalog-filtered unit set on a representative (first) session
    of that date; the largest wins, ties broken by session count.

    Each returned entry carries the session id + root, the recording frame rate, the
    total session duration, per-call onsets split into the two category groups (as
    polars dataframes), and spike-time arrays for the filtered unit set common to all
    of the chosen day's sessions. The tracks array is not materialised -- only its
    leading dimension is read so ``total_duration = n_frames / fs`` is cheap.

    The per-session ``*_good.npy`` tree is globbed ONCE and the resulting
    stem -> path map is reused for both filtering and spike loading; a recursive glob
    per unit would otherwise dominate runtime over network mounts.

    Parameters
    ----------
    animal_id : str
        Focal-mouse ID (matches ``track_names[0]`` in the tracking H5).
    session_names : list[str]
        Session directory names under ``data_root``; may span multiple days.
    data_root : pathlib.Path
        Root directory holding the session folders.
    catalog : dict[tuple[str, str, str], dict]
        Output of :func:`load_unit_catalog`.
    category_column : str
        ``usv_summary`` column used to split calls into groups
        (e.g. ``"qlvm_supercategory"``).
    group_a_ids, group_b_ids : list
        Category id values defining group A and group B.
    cluster_group : str
        Required ``cluster_group`` label, forwarded to :func:`filter_units_by_catalog`.
    require_somatic : bool
        Somatic-only flag, forwarded to :func:`filter_units_by_catalog`.
    brain_areas : set[str]
        Allowed brain areas, forwarded to :func:`filter_units_by_catalog`.
    message_output : Any, optional
        Diagnostic sink; defaults to None, in which case the built-in ``print`` is used.

    Returns
    -------
    sessions_data (list[dict])
        One dict per session of the chosen day, each with keys ``session_id``,
        ``session_root``, ``fs``, ``group_a_df``, ``group_b_df``, ``neural_data``,
        ``total_duration``.
    """

    log = message_output or print

    by_date: dict[str, list[str]] = defaultdict(list)
    for session_name in session_names:
        by_date[session_name.split("_")[0]].append(session_name)
    if not by_date:
        return []

    # One good-unit tree-walk per session; reused for filtering AND spike loading.
    good_maps: dict[str, dict[str, pathlib.Path]] = {
        session_name: {f.stem: f for f in (data_root / session_name).glob("**/*_good.npy")}
        for session_name in session_names
    }

    def _score(date: str) -> tuple[int, int]:
        first_session = sorted(by_date[date])[0]
        filtered = filter_units_by_catalog(
            animal_id, date, set(good_maps[first_session]), catalog,
            cluster_group=cluster_group, require_somatic=require_somatic, brain_areas=brain_areas,
        )
        return (len(filtered), len(by_date[date]))

    scored = {date: _score(date) for date in by_date}
    chosen_date = max(scored, key=lambda date: scored[date])
    chosen_session_names = by_date[chosen_date]
    log(
        f"  {animal_id}: picked day {chosen_date} "
        f"({scored[chosen_date][0]} filtered units, {scored[chosen_date][1]} sessions); "
        f"all days = " + ", ".join(f"{d}({s[0]}u/{s[1]}s)" for d, s in sorted(scored.items()))
    )

    per_session_sets = [
        filter_units_by_catalog(
            animal_id, session_name.split("_")[0], set(good_maps[session_name]), catalog,
            cluster_group=cluster_group, require_somatic=require_somatic, brain_areas=brain_areas,
        )
        for session_name in chosen_session_names
    ]
    common_unit_ids = set.intersection(*per_session_sets) if per_session_sets else set()
    log(f"  {animal_id}: common filtered units = {len(common_unit_ids)}")

    sessions_data = []
    for session_name in chosen_session_names:
        directory = data_root / session_name
        tracking_file = next(directory.glob("**/*_translated_rotated_metric.h5"))
        with h5py.File(name=tracking_file, mode="r") as track_file:
            mouse_track_names = [t.decode("utf-8") for t in list(track_file["track_names"])]
            recording_frame_rate = float(track_file["recording_frame_rate"][()])
            n_frames = int(track_file["tracks"].shape[0])

        usv_summary_file = next(directory.glob("**/*_usv_summary.csv"))
        usv_summary_data = pls.read_csv(usv_summary_file)
        focal_usvs = usv_summary_data.filter(pls.col("emitter") == mouse_track_names[0])
        group_a_df = focal_usvs.filter(pls.col(category_column).is_in(group_a_ids))
        group_b_df = focal_usvs.filter(pls.col(category_column).is_in(group_b_ids))

        good = good_maps[session_name]
        session_neural_data = {unit_id: np.load(good[unit_id])[0, :] for unit_id in common_unit_ids}

        sessions_data.append({
            "session_id":     directory.name,
            "session_root":   str(directory),
            "fs":             recording_frame_rate,
            "group_a_df":     group_a_df,
            "group_b_df":     group_b_df,
            "neural_data":    session_neural_data,
            "total_duration": n_frames / recording_frame_rate,
        })
    return sessions_data


def compute_group_acoustics(
    session: dict,
    group_key: str,
    window_s: float,
    message_output: Any = None,
) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    Convenience wrapper that runs :func:`extract_snippet_acoustics` for one category
    group of one session-data entry, reading onsets and the loudest channel
    (``peak_amp_ch``) straight from the group's ``usv_summary`` dataframe. Falls back
    to channel 0 (with a diagnostic message) when the summary predates the
    ``peak_amp_ch`` column.

    Parameters
    ----------
    session : dict
        A ``sessions_data`` entry (see :func:`load_animal_sessions`); must carry
        ``session_root`` and the group dataframe.
    group_key : str
        Either ``"group_a_df"`` or ``"group_b_df"``.
    window_s : float
        Snippet length in seconds, measured from each onset.
    message_output : Any, optional
        Diagnostic sink; defaults to None, in which case the built-in ``print`` is used.

    Returns
    -------
    features (dict[str, np.ndarray])
        :func:`extract_snippet_acoustics` output for the group's onsets.
    """

    log = message_output or print
    group_df = session[group_key]
    onsets = group_df["start"].to_numpy()
    if "peak_amp_ch" in group_df.columns:
        peak_channels = group_df["peak_amp_ch"].to_numpy()
    else:
        log(f"  {session['session_id']}: no 'peak_amp_ch' column -> using channel 0.")
        peak_channels = np.zeros(onsets.shape[0])
    return extract_snippet_acoustics(session["session_root"], onsets, peak_channels, window_s)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Description
    -----------
    Cohen's d (pooled-SD standardized mean difference) between two 1D samples,
    ignoring non-finite entries.

    Parameters
    ----------
    x, y : np.ndarray
        The two samples.

    Returns
    -------
    d (float)
        ``(mean(x) - mean(y)) / pooled_sd``, or NaN if either sample has fewer than
        two finite values or the pooled SD is zero.
    """

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    n_x, n_y = x.size, y.size
    pooled_sd = np.sqrt(((n_x - 1) * x.var(ddof=1) + (n_y - 1) * y.var(ddof=1)) / (n_x + n_y - 2))
    return float((x.mean() - y.mean()) / pooled_sd) if pooled_sd > 0 else np.nan


def bootstrap_vs_null_stats(
    boot_data: np.ndarray,
    null_data: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Description
    -----------
    Right-tailed empirical comparison of a bootstrap distribution against a null
    distribution: the (``+1``-bias-corrected) p-value is the fraction of null samples
    at or above the bootstrap mean, and the Z-score standardizes the bootstrap mean by
    the null's mean and standard deviation.

    Parameters
    ----------
    boot_data : np.ndarray
        Bootstrap distribution of the observed metric.
    null_data : np.ndarray
        Null distribution of the same metric.

    Returns
    -------
    boot_mean, null_mean, p_val, z (tuple[float, float, float, float])
        Bootstrap mean, null mean, ``(#{null >= boot_mean} + 1) / (n_null + 1)``, and
        ``(boot_mean - null_mean) / null_std`` (0.0 when the null has zero spread).
    """

    boot_mean = float(np.mean(boot_data))
    null_mean = float(np.mean(null_data))
    null_std = float(np.std(null_data))
    n_null = int(np.size(null_data))
    p_val = float((np.sum(null_data >= boot_mean) + 1) / (n_null + 1))
    z = (boot_mean - null_mean) / null_std if null_std > 0 else 0.0
    return boot_mean, null_mean, p_val, z
