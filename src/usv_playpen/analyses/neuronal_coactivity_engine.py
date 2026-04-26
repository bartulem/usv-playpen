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

def get_bootstrapped_simple_calls(
    simple_counts: np.ndarray,
    n_target: int,
    n_iterations: int = 1000
) -> dict[str, np.ndarray]:
    """
    Performs bootstrapping on simple vocalization trials to equalize sample size
    with the complex vocalization group, ensuring statistical symmetry.

    This function resamples the simple vocalization count matrix with replacement
    to match the trial count of the complex group. It calculates three distinct
    coactivity metrics—r_sc, population similarity (cosine), and population
    correlation—to build an empirical distribution of the mean for the
    subsampled group.

    Parameters
    ----------
    simple_counts : np.ndarray
        The neural count matrix for simple USVs, shape (Neurons x N_simple).
    n_target : int
        The target sample size to match (typically the N of the complex group).
    n_iterations : int, optional
        The number of bootstrap iterations to perform, by default 1000.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the bootstrap distributions for:
        - 'r_sc': Mean pairwise spike count correlations.
        - 'similarity': Mean population vector cosine similarities.
        - 'pop_corr': Mean population vector Pearson correlations.
    """

    boot_rsc = np.zeros(n_iterations)
    boot_sim = np.zeros(n_iterations)
    boot_pop = np.zeros(n_iterations)

    num_trials = simple_counts.shape[1]
    rng = np.random.default_rng()

    for i in range(n_iterations):
        # Sample trial indices with replacement to match n_target
        idx = rng.choice(num_trials, n_target, replace=True)
        resampled_matrix = simple_counts[:, idx]

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
    for n_id, spikes in neural_data.items():
        # Apply shift and wrap using modulo
        shifted_spikes = (spikes + shift_s) % total_duration_s
        # Sort is required for subsequent snippet extraction
        shifted_data[n_id] = np.sort(shifted_spikes)
    return shifted_data

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

    rng = np.random.default_rng()

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
    n_shuffles: int = 1000
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
    n_shuffles : int
        Number of global shuffle iterations.

    Returns
    -------
    dict[str, np.ndarray]
        Empirical null distributions for the concatenated sessions.
    """
    # 1. Initialize results by checking metric keys from a dummy run
    dummy_matrices = []
    for onsets, neural, duration in zip(session_onsets, session_neural_data, session_durations):
        # We only need a small slice to get the keys
        dummy_matrices.append(extract_snippet_matrix(onsets[:2], neural, window_s))

    combined_dummy = np.hstack(dummy_matrices)
    metric_keys = compute_coactivity_metrics(combined_dummy).keys()
    results = {key: np.zeros(n_shuffles) for key in metric_keys}

    rng = np.random.default_rng()

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

def sample_onsets_across_sessions(
    sessions_list: list[dict[str, Any]],
    category_key: str,
    n_total: int
) -> list[np.ndarray]:
    """
    Samples a total of N onsets across multiple sessions, maintaining
    session identity for subsequent circular shifting.

    This function pools all available onsets from a specific category
    (e.g., 'complex_df') across all provided sessions, draws a random
    subset of size n_total, and returns them as a list of arrays
    corresponding to the original session order.

    Parameters
    ----------
    sessions_list : list[dict]
        The list of session data dictionaries created during loading.
    category_key : str
        The key in the session dictionary to pull onsets from
        (e.g., 'complex_df' or 'simple_df').
    n_total : int
        The total number of onsets to sample across the entire dataset.

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
    rng = np.random.default_rng()
    selected_indices = rng.choice(len(all_indices), size=n_total, replace=False)
    sampled_points = [all_indices[i] for i in selected_indices]

    # 3. Redistribute sampled onsets back into session buckets
    session_buckets = [[] for _ in range(len(sessions_list))]
    for s_idx, t in sampled_points:
        session_buckets[s_idx].append(t)

    # 4. Convert to sorted numpy arrays
    return [np.sort(np.array(bucket)) for bucket in session_buckets]
