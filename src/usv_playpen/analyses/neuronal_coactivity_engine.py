"""
@author: bartulem
Enables analysis of neuronal coactivity during vocalization events, including
pairwise spike count correlations and population vector similarity, with
bootstrapping and circular shuffle controls for statistical validation.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any

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
    count_matrix = np.zeros((num_neurons, num_trials))

    for t_idx, t_start in enumerate(onsets):
        t_end = t_start + window_s
        for n_idx, n_key in enumerate(neuron_keys):
            spikes = neural_data[n_key]
            count_matrix[n_idx, t_idx] = np.sum((spikes >= t_start) & (spikes <= t_end))

    return count_matrix

def compute_coactivity_metrics(count_matrix: np.ndarray) -> dict[str, Any]:
    """
    Calculates pairwise spike count correlations (r_sc) and population
    vector cosine similarity.

    Parameters
    ----------
    count_matrix : np.ndarray
        Matrix of shape (N_neurons, N_trials).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'r_sc' (mean of unique pairs) and 'similarity'
        (mean trial-to-trial cosine similarity).
    """

    num_neurons, num_trials = count_matrix.shape
    if num_neurons < 2:
        return {"r_sc": np.nan, "similarity": np.nan}

    # Pairwise spike count correlation (r_sc)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_mat = np.corrcoef(count_matrix)

    # Extract unique pairs (upper triangle)
    r_sc_values = corr_mat[np.triu_indices(num_neurons, k=1)]
    mean_r_sc = np.nanmean(r_sc_values)

    # Population vector similarity (cosine similarity)
    # Transpose to (trials, neurons) for sklearn
    sim_matrix = cosine_similarity(count_matrix.T)

    # Mean similarity across unique trial pairs
    pop_sim_values = sim_matrix[np.triu_indices(num_trials, k=1)]
    mean_pop_sim = np.nanmean(pop_sim_values)

    return {
        "r_sc": mean_r_sc,
        "similarity": mean_pop_sim
    }

def get_bootstrapped_simple_calls(
    simple_counts: np.ndarray,
    n_target: int,
    n_iterations: int = 1000
) -> dict[str, np.ndarray]:
    """
    Performs bootstrapping on simple vocalization trials to equalize sample size
    with the complex vocalization group.

    Parameters
    ----------
    simple_counts : np.ndarray
        Count matrix for simple USVs (Neurons x N_simple).
    n_target : int
        The number of trials in the complex group (target size).
    n_iterations : int
        Number of bootstrap iterations.

    Returns
    -------
    Dict[str, np.ndarray]
        Arrays of bootstrapped r_sc and similarity values.
    """

    boot_rsc = np.zeros(n_iterations)
    boot_sim = np.zeros(n_iterations)
    num_trials = simple_counts.shape[1]
    rng = np.random.default_rng()

    for i in range(n_iterations):
        idx = rng.choice(num_trials, n_target, replace=True)
        resampled_matrix = simple_counts[:, idx]
        metrics = compute_coactivity_metrics(resampled_matrix)
        boot_rsc[i] = metrics["r_sc"]
        boot_sim[i] = metrics["similarity"]

    return {"r_sc": boot_rsc, "similarity": boot_sim}

def perform_circular_shuffle(
    onsets: np.ndarray,
    neural_data: dict[str, np.ndarray],
    total_duration: float,
    window_s: float,
    min_shift_s: float = 20.0,
    max_shift_s: float = 60.0,
    n_shuffles: int = 1000
) -> dict[str, np.ndarray]:
    """
    Performs a circular temporal shuffle by shifting each neuron's spike
    train by a random offset and wrapping around the recording duration.

    Parameters
    ----------
    onsets : np.ndarray
        Array of vocalization start times.
    neural_data : dict[str, np.ndarray]
        Dictionary of original spike times.
    total_duration : float
        Total recording duration in seconds (used for wrapping).
    window_s : float
        Analysis window length.
    min_shift_s : float, optional
        Minimum time shift in seconds. Default is 20.0.
    max_shift_s : float, optional
        Maximum time shift in seconds. Default is 60.0.
    n_shuffles : int, optional
        Number of global shuffles to perform. Default is 1000.

    Returns
    -------
    dict[str, np.ndarray]
        Distribution of metrics under the null hypothesis.
    """

    shuff_rsc = np.zeros(n_shuffles)
    shuff_sim = np.zeros(n_shuffles)
    neuron_keys = list(neural_data.keys())

    rng = np.random.default_rng()
    for i in range(n_shuffles):
        # Create a shifted version of the entire neural dictionary
        shifted_neural_data = {}
        for n_key in neuron_keys:
            # Shift by a random value between min_shift_s and max_shift_s seconds
            shift = rng.uniform(min_shift_s, max_shift_s)
            shifted_spikes = (neural_data[n_key] + shift) % total_duration
            shifted_neural_data[n_key] = np.sort(shifted_spikes)

        # Extract matrix and compute metrics for this shuffled iteration
        shuffled_matrix = extract_snippet_matrix(onsets, shifted_neural_data, window_s)
        metrics = compute_coactivity_metrics(shuffled_matrix)
        shuff_rsc[i] = metrics["r_sc"]
        shuff_sim[i] = metrics["similarity"]

    return {"r_sc": shuff_rsc, "similarity": shuff_sim}

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
        Arrays of 'r_sc' and 'similarity' for each time bin.
    """

    sliding_rsc = np.zeros(n_steps)
    sliding_sim = np.zeros(n_steps)
    time_bins = np.arange(n_steps) * step_s

    for i, offset in enumerate(time_bins):
        # Shift the onsets forward for this specific bin
        current_onsets = onsets + offset

        # Reuse your existing extraction and computation logic
        matrix = extract_snippet_matrix(current_onsets, neural_data, window_s)
        metrics = compute_coactivity_metrics(matrix)

        sliding_rsc[i] = metrics["r_sc"]
        sliding_sim[i] = metrics["similarity"]

    return {
        "time_bins": time_bins,
        "r_sc": sliding_rsc,
        "similarity": sliding_sim
    }
