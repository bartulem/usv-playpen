"""
@author: bartulem
Module for deep non-linear USV manifold prediction using pure JAX.

This module replaces the legacy Dual-Stream MLP with a 1D Depthwise-Separable
Convolutional Neural Network (CNN) featuring Residual Connections (ResNet-1D).
It maps high-dimensional, temporal behavioral kinematics (X) to the continuous
topological space of the vocal repertoire (Y).

Key Architectural Choices:
1.  Depthwise Separable Convolutions: Isolates temporal motif extraction per kinematic
    feature before mixing synergies, vastly reducing parameter bloat and overfitting.
2.  Residual Pooling Blocks: Combines a Pointwise Shortcut (1x1 Conv, Stride 2) with
    a Heavy Path (Depthwise -> Pointwise -> MaxPool) to smoothly halve the temporal
    dimension while preserving gradient flow.
3.  Translation Variance (Flattening): Preserves the absolute timing of behavioral
    motifs by flattening the final temporal matrix instead of globally averaging it.
4.  Pure Functional JAX: Explicitly manages all convolutional and Batch Normalization
    states without relying on high-level OOP wrappers like Flax or Haiku.
"""

import os

# Force efficient memory allocation to prevent OOMs
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Disable aggressive XLA auto-tuning to prevent cuDNN/Triton compilation hangs
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

# NEW: Silence non-critical XLA, Triton, and Abseil optimization warnings (E0324... timestamps)
# TF_CPP_MIN_LOG_LEVEL suppresses JAX info/warning logs.
# GLOG_minloglevel suppresses C++ level logs (0=INFO, 1=WARN, 2=ERROR, 3=FATAL).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import json
import optax
import numpy as np
import pathlib
import pickle
from datetime import datetime
from functools import partial
from typing import Dict, Any, List, Tuple

from .cluster_geometry import (
    derive_cluster_centers_empirically,
    derive_cluster_geometry,
    usv_in_circle,
)
from .manifold_metric import (
    pairwise_distance,
    signed_diff_jax,
    resolve_manifold_metric,
)
from .modeling_usv_manifold_position import (
    get_stratified_spatial_splits_stable as _manifold_spatial_splits,
)


class HashableDict(dict):
    """A dictionary that can be hashed for JAX static arguments, handling nested lists."""

    def __hash__(self):
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(make_hashable(e) for e in v)
            elif isinstance(v, dict):
                return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
            return v

        return hash(tuple(sorted((k, make_hashable(v)) for k, v in self.items())))

# =============================================================================
# PHASE 1: UTILITIES & DATA AUGMENTATION
# =============================================================================

def apply_kinematic_masking(x_seq: np.ndarray, mask_prob: float, mask_length: int,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Applies 1D Cutout (Kinematic Masking) to a batch of behavioral sequences.

    Acts as aggressive regularization to prevent the network from over-relying
    on a single dominant feature (like speed or distance). It forces the network
    to construct the continuous vocal manifold using secondary kinematic cues.

    Algorithmic Logic:
    For every single sequence in the batch, and for every individual feature channel,
    it rolls a probability check. If the check passes, a contiguous chunk of length
    `mask_length` is overwritten with 0.0 (the neutral z-scored mean).

    Parameters
    ----------
    x_seq : np.ndarray
        A 3D array of shape (Batch, Features, Time_Bins) representing the kinematics.
    mask_prob : float
        The probability (0.0 to 1.0) that any specific feature channel gets blinded.
    mask_length : int
        The duration (in frames) of the masked chunk.
    rng : np.random.Generator, optional
        Seeded NumPy generator used for both the mask-decision draws and the start-index
        draws. If None, a fresh default-seeded generator is created (non-reproducible).

    Returns
    -------
    masked_batch : np.ndarray
        The augmented 3D tensor with random temporal chunks zeroed out.
    """
    if rng is None:
        rng = np.random.default_rng()

    batch_size, n_feats, n_bins = x_seq.shape
    masked_batch = x_seq.copy()

    # Generate a boolean matrix determining which (batch, feature) pairs get masked
    mask_decisions = rng.random((batch_size, n_feats)) < mask_prob

    for i in range(batch_size):
        for f in range(n_feats):
            if mask_decisions[i, f]:
                # Choose a random start point, ensuring we don't slice out of bounds
                max_start = max(1, n_bins - mask_length)
                start_idx = int(rng.integers(0, max_start))
                end_idx = min(start_idx + mask_length, n_bins)

                # Zero out the temporal chunk (0.0 represents the mean in standardized data)
                masked_batch[i, f, start_idx:end_idx] = 0.0

    return masked_batch

def apply_temporal_warping(x_seq: np.ndarray, warp_factors: np.ndarray) -> np.ndarray:
    """
    Applies dynamic temporal warping to a batch of kinematic sequences.

    Biological behaviors rarely unfold at the exact same speed. A "lunge" might
    take 0.5 seconds in one instance and 0.6 seconds in another. To prevent the
    network from overfitting to absolute durations, this function acts as a data
    augmentation step, randomly stretching or squeezing the time-series matrices
    via linear interpolation.

    Mathematical Operation:
    Uses a center-anchored warp so the middle frame is a fixed point and the
    distortion accumulates symmetrically toward both edges of the window:
    $t_{query} = t_{center} + (t_{orig} - t_{center}) * w_i$,
    with $t_{query}$ clipped to the valid index range before linear interpolation.
    This eliminates the one-sided boundary clamp a left-anchored warp would
    otherwise produce (where warp_factor < 1 forces a constant-value tail and
    warp_factor > 1 silently truncates the end of the sequence).

    Parameters
    ----------
    x_seq : np.ndarray
        A 3D array of shape (Batch, Features, Time_Bins) representing the raw
        temporal kinematics.
    warp_factors : np.ndarray
        A 1D array of shape (Batch, ) containing the specific scaling factor for
        each sequence (e.g., 0.90 for a 10% squeeze, 1.10 for a 10% stretch).

    Returns
    -------
    warped_batch : np.ndarray
        A 3D array of shape (Batch, Features, Time_Bins) containing the
        temporally warped kinematics. Any out-of-range queries at the left and
        right edges are clipped symmetrically to the first / last input frame.
    """

    batch_size, n_feats, n_bins = x_seq.shape
    input_t = np.arange(n_bins)
    center = (n_bins - 1) / 2.0
    warped_batch = np.zeros_like(x_seq)

    for i in range(batch_size):
        t_query = center + (input_t - center) * warp_factors[i]
        t_query = np.clip(t_query, 0.0, n_bins - 1)
        for f in range(n_feats):
            warped_batch[i, f, :] = np.interp(t_query, input_t, x_seq[i, f, :])

    return warped_batch


def get_grid_balanced_indices(Y_vals: np.ndarray, grid_size: int = 25,
                              base_samples: int = 40, alpha: float = 0.5,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generates training indices that uniformly sample the 2D continuous acoustic manifold.

    The UMAP projection of vocalizations typically features an overwhelmingly dense
    central core and sparse "satellite" clusters. Standard random mini-batching
    would cause the network to exclusively optimize for the dense core, entirely
    ignoring rare vocal states. This function imposes spatial fairness.

    Algorithmic Logic:
    1. Divides the empirical 2D continuous space into a `grid_size` x `grid_size` mesh.
    2. Maps every $(x, y)$ coordinate in `Y_vals` to its discrete grid cell.
    3. For every populated cell, draws a density-scaled number of indices
       (with replacement). The per-cell draw is
       `ceil(base_samples * (true_count ** alpha) / (base_samples ** alpha))`,
       so `alpha = 0` enforces a flat quota per cell (perfect grid balance),
       `alpha = 1` recovers uniform sampling proportional to occupancy (no
       balancing), and intermediate values interpolate between the two.

    Parameters
    ----------
    Y_vals : np.ndarray
        A 2D array of shape (N, 2) containing the continuous UMAP targets.
    grid_size : int, default 25
        The number of bins to divide both the X and Y spatial axes into.
    base_samples : int, default 40
        Baseline number of indices drawn from a cell with occupancy equal to
        `base_samples`. The effective per-cell draw scales with the true cell
        occupancy via the `alpha` exponent below.
    alpha : float, default 0.5
        Exponent of the density-scaling rule described above.
    rng : np.random.Generator, optional
        Seeded NumPy generator used for the per-cell `choice` draws. If None,
        a fresh default-seeded generator is created (non-reproducible).

    Returns
    -------
    balanced_indices : np.ndarray
        A 1D array of randomly selected data indices. The final length is the
        sum of the density-scaled per-cell draws.
    """

    if rng is None:
        rng = np.random.default_rng()

    x_bins = np.linspace(Y_vals[:, 0].min(), Y_vals[:, 0].max(), grid_size)
    y_bins = np.linspace(Y_vals[:, 1].min(), Y_vals[:, 1].max(), grid_size)

    x_idx = np.digitize(Y_vals[:, 0], x_bins)
    y_idx = np.digitize(Y_vals[:, 1], y_bins)

    cells = {}
    for i, cell_coord in enumerate(zip(x_idx, y_idx)):
        if cell_coord not in cells:
            cells[cell_coord] = []
        cells[cell_coord].append(i)

    sampled_indices = []
    for c in cells:
        true_count = len(cells[c])
        # Scale the sampling target based on the cell's actual density
        target_samples = int(np.ceil(base_samples * (true_count ** alpha) / (base_samples ** alpha)))

        # Ensure we pull at least 1 sample from every populated cell
        target_samples = max(1, target_samples)

        sampled_indices.append(
            rng.choice(cells[c], size=target_samples, replace=True)
        )

    return np.concatenate(sampled_indices)


# =============================================================================
# PHASE 2: PURE JAX RESNET-1D ARCHITECTURE
# =============================================================================

def init_cnn_params_and_state(key: jax.Array, in_channels: int,
                              time_steps: int, hp: Dict[str, Any]) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    r"""
    Initializes the trainable weights, biases, and non-trainable Batch Normalization
    states for the Depthwise-Separable 1D ResNet.

    This refactored version pulls all structural configuration from the 'hp' dictionary,
    enabling Multi-Scale Inception branching in Block 0 and dynamic Attention resolution.

    Initialization Strategy:
    Uses He Normal initialization ($W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}}$)
    for all convolutional and dense layers to preserve variance through the non-linear activations.
    Batch Norm Gamma ($\gamma$) is initialized to 1.0, Beta ($\beta$) to 0.0.

    Multi-Scale logic (Block 0):
    If 'use_inception_kernels' is True, initializes parallel depthwise kernels defined by
    'inception_kernel_sizes'. This allows the network to learn high-frequency and
    low-frequency behavioral motifs simultaneously in the first layer. If False,
    Block 0 acts as a standard single-scale layer.

    Parameters
    ----------
    key : jax.Array
        A JAX PRNG key to ensure deterministic, reproducible weight generation.
    in_channels : int
        The initial number of kinematic features (e.g., 20).
    time_steps : int
        The initial length of the temporal sequence (e.g., 600 frames).
    hp : dict
        Hyperparameter dictionary containing 'kernel_size', 'inception_kernel_sizes',
        'use_inception_kernels', 'se_reduction', and 'hidden_dim'.

    Returns
    -------
    params : dict
        All trainable network weights (Conv kernels, dense matrices, BN gamma/beta).
    state : dict
        All non-trainable network states (BN moving means and moving variances).
    """

    k = jax.random.split(key, 60)
    params = {}
    state = {}

    channels = [in_channels, 32, 64, 128, 256]

    # Read structural inputs directly from dict
    std_kernel = hp['kernel_size']
    inc_kernels = hp['inception_kernel_sizes']
    se_reduction = hp['se_reduction']
    hidden_dim = hp['hidden_dim']
    use_inception = hp['use_inception_kernels']

    current_time = time_steps

    for i in range(4):
        c_in = channels[i]
        c_out = channels[i + 1]

        if i == 0 and use_inception:
            # === Multi-Scale Inception Block 0 ===
            # Initialize N parallel depthwise kernels for different temporal fields
            for j, k_size in enumerate(inc_kernels):
                params[f'b0_dw_w_{j}'] = jax.random.normal(k[j], (c_in, 1, k_size)) * jnp.sqrt(2.0 / k_size)

            # Pointwise mixes parallel scales (N_kernels * in_channels -> c_out)
            pw_in_dim = c_in * len(inc_kernels)
            params[f'b0_pw_w'] = jax.random.normal(k[10], (c_out, pw_in_dim, 1)) * jnp.sqrt(2.0 / pw_in_dim)
            params[f'b0_pw_b'] = jnp.zeros((1, c_out, 1))

            # Shortcut matches scale and channel expansion
            params[f'b0_sc_w'] = jax.random.normal(k[11], (c_out, c_in, 1)) * jnp.sqrt(2.0 / c_in)
            params[f'b0_sc_b'] = jnp.zeros((1, c_out, 1))
        else:
            # === Standard Blocks (Or Block 0 if Inception is False) ===
            # 1. Depthwise Conv Weights
            params[f'b{i}_dw_w'] = jax.random.normal(k[i * 10], (c_in, 1, std_kernel)) * jnp.sqrt(2.0 / std_kernel)

            # 2. Pointwise Conv Weights
            params[f'b{i}_pw_w'] = jax.random.normal(k[i * 10 + 1], (c_out, c_in, 1)) * jnp.sqrt(2.0 / c_in)
            params[f'b{i}_pw_b'] = jnp.zeros((1, c_out, 1))

            # 3. Shortcut Pointwise (1x1, stride 2)
            params[f'b{i}_sc_w'] = jax.random.normal(k[i * 10 + 2], (c_out, c_in, 1)) * jnp.sqrt(2.0 / c_in)
            params[f'b{i}_sc_b'] = jnp.zeros((1, c_out, 1))

        # 4. BatchNorm States & Params (Main & Shortcut)
        params[f'b{i}_bn_main_gamma'] = jnp.ones((1, c_out, 1))
        params[f'b{i}_bn_main_beta'] = jnp.zeros((1, c_out, 1))
        state[f'b{i}_bn_main_mean'] = jnp.zeros((1, c_out, 1))
        state[f'b{i}_bn_main_var'] = jnp.ones((1, c_out, 1))

        params[f'b{i}_bn_sc_gamma'] = jnp.ones((1, c_out, 1))
        params[f'b{i}_bn_sc_beta'] = jnp.zeros((1, c_out, 1))
        state[f'b{i}_bn_sc_mean'] = jnp.zeros((1, c_out, 1))
        state[f'b{i}_bn_sc_var'] = jnp.ones((1, c_out, 1))

        # === 5. Squeeze-and-Excitation (SE) Attention Weights ===
        se_hidden = max(1, c_out // se_reduction)

        params[f'b{i}_se_w1'] = jax.random.normal(k[i * 10 + 3], (c_out, se_hidden)) * jnp.sqrt(2.0 / c_out)
        params[f'b{i}_se_b1'] = jnp.zeros(se_hidden)

        params[f'b{i}_se_w2'] = jax.random.normal(k[i * 10 + 4], (se_hidden, c_out)) * jnp.sqrt(2.0 / se_hidden)
        params[f'b{i}_se_b2'] = jnp.zeros(c_out)

        current_time = current_time // 2

    # === Final Dense Head (Bottleneck) ===
    flattened_size = channels[-1] * current_time

    # If Hybrid Flatten is active, we add the Global Average Pooling dimension (c_out)
    if hp['use_hybrid_flatten']:
        flattened_size += channels[-1]

    padded_flat_size = 1 << (flattened_size - 1).bit_length()
    hidden_dim = hp['hidden_dim']

    params['dense1_w'] = jax.random.normal(k[-2], (padded_flat_size, hidden_dim)) * jnp.sqrt(2.0 / padded_flat_size)
    params['dense1_b'] = jnp.zeros(hidden_dim)

    params['dense2_w'] = jax.random.normal(k[-1], (hidden_dim, 2)) * jnp.sqrt(2.0 / hidden_dim)
    params['dense2_b'] = jnp.zeros(2)

    return params, state


def _batch_norm_1d(x: jax.Array, gamma: jax.Array, beta: jax.Array,
                   mean: jax.Array, var: jax.Array, is_training: bool,
                   momentum: float = 0.99, eps: float = 1e-5) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Internal helper function for 1D Batch Normalization over the NCW tensor format.

    Batch Normalization solves internal covariate shift by re-centering and scaling
    the activations of a layer.

    Mathematical Operation:
    $$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \gamma + \beta $$

    Parameters
    ----------
    x : jax.Array
        The input tensor of shape (Batch, Channels, Width).
    gamma : jax.Array
        The learned scaling parameter of shape (1, Channels, 1).
    beta : jax.Array
        The learned shifting parameter of shape (1, Channels, 1).
    mean : jax.Array
        The current exponential moving average of the mean.
    var : jax.Array
        The current exponential moving average of the variance.
    is_training : bool
        Flag to determine whether to update moving averages or freeze them.
    momentum : float, default 0.99
        The decay rate for the exponential moving average update.
    eps : float, default 1e-5
        A small constant added to the variance to prevent division by zero.

    Returns
    -------
    x_norm : jax.Array
        The normalized and scaled output tensor.
    new_mean : jax.Array
        The updated moving mean.
    new_var : jax.Array
        The updated moving variance.
    """

    if is_training:
        batch_mean = jnp.mean(x, axis=(0, 2), keepdims=True)
        batch_var = jnp.var(x, axis=(0, 2), keepdims=True)

        new_mean = momentum * mean + (1.0 - momentum) * batch_mean
        new_var = momentum * var + (1.0 - momentum) * batch_var

        x_norm = (x - batch_mean) / jnp.sqrt(batch_var + eps)
        return x_norm * gamma + beta, new_mean, new_var
    else:
        x_norm = (x - mean) / jnp.sqrt(var + eps)
        return x_norm * gamma + beta, mean, var


@partial(jax.jit, static_argnames=['is_training', 'hp'])
def cnn_forward(params: Dict[str, jax.Array],
                state: Dict[str, jax.Array],
                x: jax.Array,
                Y_center: jax.Array,
                Y_scale: jax.Array,
                hp: Dict[str, Any],
                rng_key: jax.Array = None,
                is_training: bool = False) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Executes the pure JAX forward pass of the Depthwise-Separable ResNet-1D.

    This refactored version implements:
    1. Multi-Scale Input Branching: If active, Block 0 processes kinematics via parallel
       kernels and concatenates them before feature mixing.
    2. Dynamic Physics: Activations (ReLU/GELU) and SE-Attention gates (Sigmoid/Hard-Sigmoid)
       are read directly from the 'hp' dictionary.
    3. Regularization: Dropout is pulled from hp['dropout_rate'].

    Parameters
    ----------
    params : dict
        Trainable weights (Conv kernels, dense weights, BN gamma/beta).
    state : dict
        Non-trainable states (BN moving mean/var).
    x : jax.Array
        Input tensor of shape (Batch, Features, Time).
    Y_center : jax.Array
        The 2D spatial center of the training manifold for `tanh` scaling.
    Y_scale : jax.Array
        The 2D spatial half-width of the training manifold for `tanh` scaling.
    hp : dict
        Hyperparameter dictionary containing configuration for physics and architecture.
    rng_key : jax.RNGKey, optional
        A JAX random key for stochastic dropout.
    is_training : bool, default False
        Signals the BN layers to compute batch statistics and update moving averages.

    Returns
    -------
    predictions : jax.Array
        Predicted 2D coordinates of shape (Batch, 2).
    new_state : dict
        Updated BN moving averages (only modified if is_training=True).
    """

    new_state = {}
    out = x
    dimension_numbers = ('NCW', 'OIW', 'NCW')

    # Mapping configuration flags directly from dict to JAX functions
    act = jax.nn.relu if hp['act_func'] == 'relu' else jax.nn.gelu
    se_gate = jax.nn.sigmoid if hp['se_activation'] == 'sigmoid' else jax.nn.hard_sigmoid

    # Read stability and regularization dials
    bn_momentum = hp['bn_momentum']
    bn_eps = hp['bn_eps']
    dropout_rate = hp['dropout_rate']
    use_inception = hp['use_inception_kernels']
    num_inc_kernels = len(hp['inception_kernel_sizes'])

    for i in range(4):
        in_channels = out.shape[1]

        # === PATH A: Main Transform (Multi-Scale vs Standard) ===
        if i == 0 and use_inception:
            # Parallel Multi-Scale Branching
            branches = [jax.lax.conv_general_dilated(
                lhs=out, rhs=params[f'b0_dw_w_{j}'], window_strides=(1,), padding='SAME',
                dimension_numbers=dimension_numbers, feature_group_count=in_channels
            ) for j in range(num_inc_kernels)]

            path_a = jnp.concatenate(branches, axis=1)

            path_a = jax.lax.conv_general_dilated(
                lhs=path_a, rhs=params[f'b0_pw_w'], window_strides=(1,), padding='VALID',
                dimension_numbers=dimension_numbers, feature_group_count=1
            ) + params[f'b0_pw_b']
        else:
            # Standard single-scale Depthwise -> Pointwise
            path_a = jax.lax.conv_general_dilated(
                lhs=out, rhs=params[f'b{i}_dw_w'], window_strides=(1,), padding='SAME',
                dimension_numbers=dimension_numbers, feature_group_count=in_channels
            )
            path_a = jax.lax.conv_general_dilated(
                lhs=path_a, rhs=params[f'b{i}_pw_w'], window_strides=(1,), padding='VALID',
                dimension_numbers=dimension_numbers, feature_group_count=1
            ) + params[f'b{i}_pw_b']

        # BatchNorm & Activation
        path_a, nm, nv = _batch_norm_1d(
            path_a, params[f'b{i}_bn_main_gamma'], params[f'b{i}_bn_main_beta'],
            state[f'b{i}_bn_main_mean'], state[f'b{i}_bn_main_var'],
            is_training, momentum=bn_momentum, eps=bn_eps
        )
        new_state[f'b{i}_bn_main_mean'], new_state[f'b{i}_bn_main_var'] = nm, nv
        path_a = act(path_a)

        # Smooth temporal gradient flow
        path_a_sum = jax.lax.reduce_window(
            path_a, 0.0, jax.lax.add, window_dimensions=(1, 1, 2),
            window_strides=(1, 1, 2), padding='VALID'
        )
        path_a = path_a_sum * 0.5  # Average pooling

        # === 5. SQUEEZE-AND-EXCITATION (SE) ATTENTION ===
        se_squeeze = jnp.max(path_a, axis=2)  # Peak-isolating global pool
        se_hidden = act(jnp.dot(se_squeeze, params[f'b{i}_se_w1']) + params[f'b{i}_se_b1'])
        se_weight = se_gate(jnp.dot(se_hidden, params[f'b{i}_se_w2']) + params[f'b{i}_se_b2'])
        path_a = path_a * se_weight[:, :, jnp.newaxis]

        # === PATH B: The Shortcut ===
        sc_w_key = f'b{i}_sc_w' if i > 0 else 'b0_sc_w'
        sc_b_key = f'b{i}_sc_b' if i > 0 else 'b0_sc_b'

        path_b = jax.lax.conv_general_dilated(
            lhs=out, rhs=params[sc_w_key], window_strides=(2,), padding='VALID',
            dimension_numbers=dimension_numbers, feature_group_count=1
        ) + params[sc_b_key]

        path_b, nm_sc, nv_sc = _batch_norm_1d(
            path_b, params[f'b{i}_bn_sc_gamma'], params[f'b{i}_bn_sc_beta'],
            state[f'b{i}_bn_sc_mean'], state[f'b{i}_bn_sc_var'],
            is_training, momentum=bn_momentum, eps=bn_eps
        )
        new_state[f'b{i}_bn_sc_mean'], new_state[f'b{i}_bn_sc_var'] = nm_sc, nv_sc

        # Residual Merge: Add and Re-Activate
        min_time = min(path_a.shape[2], path_b.shape[2])
        out = act(path_a[:, :, :min_time] + path_b[:, :, :min_time])

    # === FLATTEN & NON-LINEAR REASONING HEAD ===
    out_flat = out.reshape(out.shape[0], -1)

    # Hybrid Flatten: Concatenate the strictly timed motifs with the time-invariant averages
    if hp['use_hybrid_flatten']:
        out_gap = jnp.mean(out, axis=2)  # Global average over the temporal dimension
        out_flat = jnp.concatenate([out_flat, out_gap], axis=1)

    # Pad the tensor with zeros to match the perfect power-of-2 weights
    pad_len = params['dense1_w'].shape[0] - out_flat.shape[1]
    out_flat = jnp.pad(out_flat, ((0, 0), (0, pad_len)), mode='constant')

    # 1. Hidden Reasoning Layer + ReLU (using dict-defined activation)
    hidden = jnp.dot(out_flat, params['dense1_w']) + params['dense1_b']
    hidden = act(hidden)

    # 2. Pure JAX Dropout
    if is_training and dropout_rate > 0.0 and rng_key is not None:
        keep_prob = 1.0 - dropout_rate
        mask = jax.random.bernoulli(rng_key, p=keep_prob, shape=hidden.shape)
        hidden = jnp.where(mask, hidden / keep_prob, 0.0)

    # 3. Final Coordinate Projection
    logits = jnp.dot(hidden, params['dense2_w']) + params['dense2_b']

    # Map predictions strictly inside the empirical physical boundaries
    predictions = Y_center + jnp.tanh(logits) * Y_scale

    return predictions, new_state


# =============================================================================
# PHASE 3: LEARNING RATE SCHEDULER
# =============================================================================

def build_lr_schedule(peak_lr: float, total_epochs: int, steps_per_epoch: int) -> optax.Schedule:
    """
    Constructs a Warmup Cosine Decay learning rate schedule.

    Initializing deep 1D Convolutions with He normal variance can lead to massive
    activation spikes in the first few batches. A high initial learning rate would
    shatter the gradients and cause the loss to diverge instantly.

    This schedule acts as a safety buffer. It starts the learning rate at 0.0,
    linearly warms it up to the `peak_lr` over the first 10% of total training steps,
    and then decays it following a cosine curve down to 0.0 over the remaining 90%.
    This allows the optimizer to gently orient the fresh kernels before making large
    updates, and eventually forces the model to settle smoothly into the local minimum.

    Parameters
    ----------
    peak_lr : float
        The maximum learning rate to reach after the warmup phase is complete.
    total_epochs : int
        Total number of training epochs defined in the hyperparameters.
    steps_per_epoch : int
        Number of batches processed per epoch (calculated as total_samples // batch_size).

    Returns
    -------
    schedule : optax.Schedule
        A callable Optax schedule that takes the current step count and returns the
        exact learning rate for that specific gradient update.
    """

    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_steps)

    # Ensure at least 1 warmup step to prevent Optax errors on extremely short runs
    warmup_steps = max(1, warmup_steps)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0
    )
    return schedule


# =============================================================================
# PHASE 4: THE TRAINING ENGINE
# =============================================================================

class NeuralContinuousCNNRunner:
    """
    Orchestrates the data fusion, training, and statistical validation of the
    Depthwise-Separable ResNet-1D for continuous USV manifold prediction.

    This class serves as the deep-learning execution engine. It natively handles:
    1. Multivariate Tensor Construction: Bins and stacks 2D behavioral DataFrames
       into the 3D `(Batch, Channels, Time)` tensors required for 1D convolutions.
    2. Spatial K-Means Stratification: Ensures that highly dense manifold cores and
       rare acoustic "satellites" are proportionally represented in cross-validation.
    3. Tri-Strategy Experimental Control: Evaluates kinematic mapping ('actual')
       against a target-shuffled control ('null') and an empirical density draw
       ('null_model_free').
    4. Post-Hoc Permutation Importance: Decouples specific kinematic channels
       during inference to quantify their individual physical significance via
       Delta Euclidean Error.
    """

    def __init__(self, modeling_settings: dict | None) -> None:
        """
        Initializes the CNN runner with strict parameter lookups.

        Parameters
        ----------
        modeling_settings : dict
            A nested dictionary containing IO paths, filter histories, split strategies,
            and JAX/Optax hyperparameters. If None is provided, it attempts to load
            from the default JSON configuration file.

        Raises
        ------
        FileNotFoundError
            If the settings dictionary is None and the JSON file cannot be located.
        KeyError
            If expected parameters are missing, enforcing fail-fast safety.
        """
        if modeling_settings is None:
            settings_path = pathlib.Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
            try:
                with open(settings_path, 'r') as settings_json_file:
                    self.modeling_settings = json.load(settings_json_file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
        else:
            self.modeling_settings = modeling_settings

        self.history_frames = int(np.floor(self.modeling_settings['model_params']['filter_history'] * self.modeling_settings['io']['camera_sampling_rate']))
        self.split_strategy = self.modeling_settings['model_params']['split_strategy']
        self.random_seed = self.modeling_settings['model_params']['random_seed']

        # Manifold-metric configuration. Threaded through the spatial
        # splitter, the training loss / RMSE evaluations, and the
        # region-saliency centroid distance — so a settings flip from
        # `'euclidean'` to `'torus'` produces consistent wrap-aware
        # behaviour across training and post-hoc analyses.
        self.manifold_metric, self.manifold_period = resolve_manifold_metric(self.modeling_settings)

        # Hyperparameter block read directly as a dict
        self.hp = HashableDict(self.modeling_settings['hyperparameters']['deep_learning']['cnn_continuous'])

    def get_stratified_spatial_splits_stable(self, groups: np.ndarray,
                                             Y: np.ndarray,
                                             split_strategy: str = 'session',
                                             n_clusters: int = 15,
                                             test_prop: float = 0.2,
                                             n_splits: int = 100,
                                             tolerance: float = 0.05,
                                             random_seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Thin delegator to the canonical splitter in
        `modeling_usv_manifold_position.get_stratified_spatial_splits_stable`.

        The CNN runner used to carry its own near-identical copy of the
        K-means stratifier; the standalone helper now owns the
        Mode-A-and-B coverage guards, the metric-aware K-means embedding,
        and the precomputed `session_to_rows` indexing path. Delegating
        keeps a single source of truth for fold construction so any
        future fix propagates to both pipelines automatically. The
        method-on-self surface is preserved for backwards compatibility
        with existing call sites.

        Parameters
        ----------
        groups : np.ndarray
            Array of session IDs.
        Y : np.ndarray
            Array of shape `(N, 2)` containing continuous UMAP coordinates.
        split_strategy : str, default 'session'
            `'session'` (whole-session holdout) or `'mixed'` (epoch-level
            stratified shuffling).
        n_clusters : int, default 15
            Number of K-means proxy clusters.
        test_prop : float, default 0.2
            Test-fold proportion (sessions for `'session'`, samples for
            `'mixed'`).
        n_splits : int, default 100
            Number of independent fold iterations.
        tolerance : float, default 0.05
            Initial spatial-distribution tolerance for the rejection
            sampler (only consulted when `split_strategy='session'`).
        random_seed : int, default 0
            Fixed seed for K-means and the per-iteration shuffle.

        Returns
        -------
        cv_folds : list of tuples
            A list of length `n_splits`, where each tuple contains
            `(train_indices, test_indices)`.
        """

        return _manifold_spatial_splits(
            groups=groups,
            Y=Y,
            n_clusters=n_clusters,
            split_strategy=split_strategy,
            test_prop=test_prop,
            n_splits=n_splits,
            tolerance=tolerance,
            random_seed=random_seed,
            metric=self.manifold_metric,
            period=self.manifold_period,
        )

    def load_multivariate_data_blocks(self, pkl_path: str) -> Dict[str, Any]:
        """
        Loads extracted feature data and constructs the 3D tensor required for the 1D CNN.

        This method reads the unrolled, session-nested 2D feature matrices from the
        source `.pkl` file and dynamically stacks them into a single multivariate
        3D tensor. Because the ResNet-1D architecture relies entirely on its temporal
        receptive field to understand global context, this function discards any
        static global stream calculations (Mean/Std).

        Parameters
        ----------
        pkl_path : str
            Full absolute path to the `.pkl` file containing the extracted (X, Y, w)
            dictionaries produced by the ContinuousModelingPipeline.

        Returns
        -------
        data_blocks : dict
            A dictionary configured for the JAX engine, containing:
            - 'X_seq': (Batch, Features, Bins) array of stacked kinematic sequences.
            - 'Y': (Batch, 2) array of continuous UMAP coordinates.
            - 'w': (Batch, ) array of KDE inverse-density sample weights.
            - 'groups': (Batch, ) array of session IDs.
            - 'features': Sorted list of the kinematic feature names.
            - 'num_bins': The finalized length of the temporal dimension after binning.
        """
        print(f"Loading and fusing multivariate data from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Strip top-level metadata keys (e.g. `_input_metadata`,
        # `_run_metadata`); only feature dicts are session-keyed and carry
        # the X/Y/w arrays. Without this filter, `_input_metadata` sorts
        # to the front and the loop below tries to index an int-valued
        # metadata field as if it were a session dict.
        features = sorted(k for k in raw_data.keys() if not str(k).startswith('_'))

        # The CNN consumes the raw history frames directly
        num_frames = self.history_frames

        X_seq_list, Y_list, w_list, groups_list = [], [], [], []
        super_list, cat_list = [], []
        sessions = sorted(list(raw_data[features[0]].keys()))

        for sess in sessions:
            sess_dict = raw_data[features[0]][sess]
            Y_sess = sess_dict['Y']
            w_sess = sess_dict['w']
            sess_seq = []

            for feat in features:
                raw_X = raw_data[feat][sess]['X']

                # Strictly enforce the frame limit (e.g., 600) without binning
                truncated_X = raw_X[:, :num_frames]
                sess_seq.append(truncated_X)

            # Stack along feature axis (Axis 1) to create (N, Features, Time)
            X_seq_list.append(np.stack(sess_seq, axis=1))
            Y_list.append(Y_sess)
            w_list.append(w_sess)
            groups_list.append(np.full(len(Y_sess), sess))

            # Optional per-USV cluster labels (supercategory + category).
            # Persisted by the extract-pipeline when the source USV CSV
            # carried them; absent on legacy pickles built before that
            # change shipped, in which case the saliency phase will raise
            # a clear "re-extract with the updated pipeline" message.
            if 'supercategory' in sess_dict:
                super_list.append(sess_dict['supercategory'])
            if 'category' in sess_dict:
                cat_list.append(sess_dict['category'])

        block = {
            'X_seq': np.vstack(X_seq_list).astype(np.float32),
            'Y': np.vstack(Y_list).astype(np.float32),
            'w': np.concatenate(w_list).astype(np.float32),
            'groups': np.concatenate(groups_list),
            'features': features,
            'num_bins': num_frames,  # Passed downstream to dynamically build network
            'source_pkl_path': pkl_path,
        }
        # Surface labels only when every session provided them — partial
        # coverage would corrupt the alignment to Y / X_seq.
        if super_list and len(super_list) == len(sessions):
            block['supercategory'] = np.concatenate(super_list)
        if cat_list and len(cat_list) == len(sessions):
            block['category'] = np.concatenate(cat_list)
        return block

    def compute_centroid_saliency(self, params: Dict[str, jax.Array], state: Dict[str, jax.Array],
                                  X_te: jax.Array, Y_center: jax.Array, Y_scale: jax.Array,
                                  polygon_centroid: Tuple[float, float]) -> Dict[str, np.ndarray]:
        r"""
        Extracts kinematic drivers for a specific manifold region via Contrastive Centroid-Gradient Saliency.

        This method identifies the precise, millisecond-resolution behavioral motifs that causally
        drive the network's prediction into a specific acoustic cluster on the continuous UMAP manifold.
        It adapts the legacy directional MLP gradient attribution to a point-attractor framework,
        suitable for the 1D-CNN.

        Theoretical Background & Mathematics:
        To extract gradients using JAX autodiff, we require a smooth, differentiable scalar objective.
        Instead of a binary polygon boundary (which yields a gradient of zero), we define the target
        as the centroid of the biological cluster: $C = (C_X, C_Y)$.

        1. Region Objective (The Point-Attractor):
           We define the region scalar as the negative Euclidean distance from the prediction to the centroid:
           $D_{region} = -\sqrt{(\hat{Y}_X - C_X)^2 + (\hat{Y}_Y - C_Y)^2}$
           Maximizing this scalar (following the positive gradient) pulls the prediction directly
           into the heart of the cluster from any starting position on the map.

        2. Global Objective (The Postural Baseline):
           To isolate features unique to the cluster, we must filter out the generic kinematic effort
           required to be active on the manifold. We define the baseline scalar as the L1 magnitude
           from the origin (assuming the origin represents a resting/neutral behavioral state):
           $D_{glob} = |\hat{Y}_X| + |\hat{Y}_Y|$

        3. Input x Gradient Attribution:
           Raw gradients only tell us the network's mathematical sensitivity. To ensure we only highlight
           behavioral motifs physically executed by the animal, we scale the absolute gradient by the
           raw kinematic input values:
           $S_{region} = X \odot \nabla_X D_{region}$

        4. Contrastive Subtraction:
           By subtracting the trial-averaged global effort template from the region-specific saliency,
           we cancel out non-specific background kinematics (e.g., general postural maintenance),
           revealing the unique, causal motor programs for that specific acoustic state:
           $S_{contrastive} = S_{region} - \frac{1}{N}\sum_{i=1}^N S_{glob}^{(i)}$

        Parameters
        ----------
        params : Dict[str, jax.Array]
            Trainable network weights (Conv kernels, dense matrices, BN gamma/beta).
        state : Dict[str, jax.Array]
            Non-trainable network states (BN moving means and variances).
        X_te : jax.Array
            The 3D temporal input tensor for the test fold, shape (Batch, Features, Time_Bins).
        Y_center : jax.Array
            The 2D spatial center of the training manifold, used for bound-limited tanh scaling.
        Y_scale : jax.Array
            The 2D spatial half-width of the training manifold.
        polygon_centroid : Tuple[float, float]
            The (X, Y) coordinate representing the center of mass for the target acoustic cluster.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing a single NumPy array of shape
            (Batch, Features, Time_Bins):
            - 'contrastive_saliency': The finalized, baseline-subtracted causal
              driver map (`region_saliency - mean(global_saliency)` over trials).
        """

        print(f"   > Extracting drivers for centroid {polygon_centroid}...")

        # 1. Define the differentiable region objective. The negative
        #    distance is the wrap-aware Euclidean norm of the signed
        #    diff between the prediction and the polygon centroid; on
        #    `metric='euclidean'` this reduces to the original
        #    flat-space distance, on torus the diff folds into the
        #    shortest-path representation before squaring.
        polygon_centroid_jax = jnp.asarray(polygon_centroid)

        def region_scalar_fn(x_single):
            preds, _ = cnn_forward(params, state, x_single[jnp.newaxis, ...],
                                   Y_center, Y_scale, self.hp, is_training=False)

            diff = signed_diff_jax(
                preds[0], polygon_centroid_jax,
                metric=self.manifold_metric, period=self.manifold_period,
            )
            dist_to_centroid = jnp.sqrt(jnp.sum(diff ** 2))
            return -dist_to_centroid

        # 2. Define the global baseline objective
        def glob_scalar_fn(x_single):
            preds, _ = cnn_forward(params, state, x_single[jnp.newaxis, ...],
                                   Y_center, Y_scale, self.hp, is_training=False)

            # L1 Magnitude from the origin (0,0)
            return jnp.abs(preds[0, 0]) + jnp.abs(preds[0, 1])

        # 3. Transform scalars into Gradient functions
        compute_region_grad = jax.grad(region_scalar_fn)
        compute_glob_grad = jax.grad(glob_scalar_fn)

        # 4. JIT compile the vectorized functions for fast batched execution
        @jax.jit
        def get_batched_grads(x_batch):
            return jax.vmap(compute_region_grad)(x_batch), jax.vmap(compute_glob_grad)(x_batch)

        # 5. Execute in memory-safe mini-batches
        batch_size = self.hp['batch_size']
        n_samples = len(X_te)

        region_grads_list = []
        glob_grads_list = []

        for i in range(0, n_samples, batch_size):
            batch_x = X_te[i:i + batch_size]
            actual_size = len(batch_x)

            # JAX requires strict static shapes. Pad the final remainder batch.
            if actual_size < batch_size:
                pad_width = ((0, batch_size - actual_size), (0, 0), (0, 0))
                batch_x = jnp.pad(batch_x, pad_width, mode='constant')

            batch_r_grad, batch_g_grad = get_batched_grads(batch_x)

            # Truncate any padding and pull directly to CPU RAM as NumPy arrays
            region_grads_list.append(np.array(batch_r_grad[:actual_size]))
            glob_grads_list.append(np.array(batch_g_grad[:actual_size]))

        # Reconstruct the full dataset gradients on the CPU
        region_grads = np.concatenate(region_grads_list, axis=0)
        glob_grads = np.concatenate(glob_grads_list, axis=0)

        # 6. Input * Gradient Saliency (Computed on CPU to save VRAM)
        X_te_np = np.array(X_te)
        region_saliency = X_te_np * np.abs(region_grads)
        glob_saliency = X_te_np * np.abs(glob_grads)

        # 7. Contrastive Subtraction
        global_template = np.mean(glob_saliency, axis=0)
        contrastive_saliency = region_saliency - global_template

        return {
            'contrastive_saliency': contrastive_saliency.astype(np.float32)
        }

    def run_cnn_training(self, data_blocks: Dict[str, Any]) -> None:
        """
        Executes the Deep 1D-CNN pipeline: Tri-strategy validation and Permutation Importance.

        This refactored version pulls all structural and physics dials directly from
        the 'hp' dictionary, enabling Multi-Scale Inception, Dynamic Attention resolution,
        and Configurable Physics (activations and BN momentum).

        Parameters
        ----------
        data_blocks : dict
            The output of `load_multivariate_data_blocks()`.

        Returns
        -------
        None
            Serializes a comprehensive "Deep Storage" `.pkl` file containing metrics,
            learned weights, true targets, and raw coordinate predictions for all strategies.
        """
        print("=" * 75)
        print(" EXECUTING 1D-CNN TRAINING PIPELINE (RESIDUAL & PURE JAX)")
        print("=" * 75)

        X_seq = data_blocks['X_seq']
        Y = data_blocks['Y']
        w = data_blocks['w']
        groups = data_blocks['groups']
        features = list(data_blocks['features'])

        # Per-USV cluster labels surfaced by the modeling pickle when it
        # carries them (the extract pipeline persists supercategory and
        # category alongside X/Y/w for every USV). The saliency phase
        # below consumes whichever the user selected via
        # `settings['hyperparameters']['deep_learning']['cnn_continuous']['saliency']['segmentation']`.
        cluster_labels_super = data_blocks['supercategory'] if 'supercategory' in data_blocks else None
        cluster_labels_cat = data_blocks['category'] if 'category' in data_blocks else None

        n_feats = len(features)
        n_bins = data_blocks['num_bins']

        # Center/Scale boundaries to bind final tanh predictions
        Y_center = jnp.array((np.max(Y, 0) + np.min(Y, 0)) / 2.0)
        Y_scale = jnp.array((np.max(Y, 0) - np.min(Y, 0)) / 2.0 * 1.1)

        # Pull training dials directly from dict (Passivity Check: No .get())
        n_folds = self.hp['n_folds']
        batch_size = self.hp['batch_size']
        total_epochs = self.hp['epochs']
        grid_size = self.hp['grid_size']
        samples_per_cell = self.hp['samples_per_cell']
        warp_range = self.hp['warp_range']
        perm_iters = self.hp['permutation_iterations']

        cv_settings = self.modeling_settings['model_params']

        folds = self.get_stratified_spatial_splits_stable(
            groups=groups, Y=Y, n_clusters=cv_settings['spatial_cluster_num'],
            test_prop=cv_settings['test_proportion'], split_strategy=self.split_strategy,
            n_splits=n_folds, random_seed=self.random_seed
        )

        deep_storage = {
            'metadata': {
                'hyperparameters': self.hp,
                'features_list': features,
                'n_time_bins': n_bins,
                'split_strategy': self.split_strategy,
                # Manifold-metric provenance — downstream
                # visualisation code reads these to compute wrap-aware
                # distances on the saved Y_pred / Y_true arrays.
                'manifold_metric': self.manifold_metric,
                'manifold_period': self.manifold_period,
            },
            'cross_validation': [],
            'feature_importance': {}
        }

        # Helper for efficient, memory-safe evaluation (splits test set to avoid OOM)
        def evaluate_batched(p, s, x_s):
            preds = []

            for i in range(0, len(x_s), batch_size):
                batch_x = x_s[i:i + batch_size]
                actual_size = len(batch_x)

                # JAX requires strict static shapes. If this is the remainder batch, pad it.
                if actual_size < batch_size:
                    pad_width = ((0, batch_size - actual_size), (0, 0), (0, 0))
                    batch_x = jnp.pad(batch_x, pad_width, mode='constant')

                # is_training=False prevents BN stats from updating during eval
                # Passing self.hp enables the dynamic physics defined in Phase 2
                pred_batch, _ = cnn_forward(
                    p, s, batch_x, Y_center, Y_scale, self.hp,
                    rng_key=None, is_training=False
                )

                # Discard the padded dummy predictions and keep only the real ones
                preds.append(pred_batch[:actual_size])

            return jnp.concatenate(preds, axis=0)

        # ---------------------------------------------------------
        # TRI-STRATEGY EXECUTION
        # ---------------------------------------------------------
        strategies = ['null_model_free', 'null', 'actual']

        # Single seeded NumPy generator threaded through every stochastic path in
        # the training loop (batch sampling, warping, masking, permutation test).
        # This replaces the previous mix of a local RandomState and unseeded global
        # `np.random` calls, so fold / epoch-level randomness is fully reproducible
        # from `self.random_seed`.
        rng = np.random.default_rng(self.random_seed)

        # We only persist the heavy parameters for the actual strategy
        best_actual_params_list = []
        best_actual_states_list = []
        actual_errors = []

        for fold, (train_idx, test_idx) in enumerate(folds):
            print(f"\n--- FOLD {fold + 1}/{n_folds} ---")

            X_tr, X_te = X_seq[train_idx], X_seq[test_idx]
            Y_tr_base, Y_te = Y[train_idx], Y[test_idx]
            w_tr, w_te = w[train_idx], w[test_idx]

            fold_results = {'fold_idx': fold, 'test_indices': test_idx, 'Y_true': Y_te}

            for strategy in strategies:
                Y_tr = Y_tr_base.copy()

                if strategy == 'null_model_free':
                    # 1. Empirical Density Draw (Simulated Dart Throw)
                    draw_indices = rng.choice(len(Y_tr), size=len(Y_te), replace=True)
                    Y_pred = Y_tr[draw_indices]

                    err = float(np.mean(pairwise_distance(
                        Y_te, Y_pred,
                        metric=self.manifold_metric, period=self.manifold_period,
                    )))
                    fold_results['Y_pred_null_model_free'] = Y_pred
                    fold_results['error_null_model_free'] = err
                    print(f"  > [Model-free prior] Euclidean Err: {err:.4f}")
                    continue

                elif strategy == 'null':
                    # 2. Strict Session-Isolated Shuffling
                    unique_groups = np.unique(groups[train_idx])
                    groups_tr = groups[train_idx]
                    for g in unique_groups:
                        g_idx = np.where(groups_tr == g)[0]
                        Y_tr[g_idx] = Y_tr[rng.permutation(g_idx)]

                # 3. JAX Network Initialization using the full hyperparameter dictionary
                params, state = init_cnn_params_and_state(
                    jax.random.PRNGKey(fold + (100 if strategy == 'null' else 0)),
                    n_feats, n_bins, self.hp
                )

                # Initialize a dedicated PRNG key just for Dropout
                dropout_rng = jax.random.PRNGKey(self.random_seed + fold)

                if self.hp['use_kde_weights']:
                    steps_per_epoch = len(Y_tr) // self.hp['batch_size']
                else:
                    steps_per_epoch = len(get_grid_balanced_indices(Y_tr, self.hp['grid_size'], self.hp['samples_per_cell'], rng=rng)) // self.hp['batch_size']

                lr_schedule = build_lr_schedule(self.hp['learning_rate'], self.hp['epochs'], steps_per_epoch) if self.hp['use_scheduler'] else self.hp['learning_rate']

                optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=self.hp['weight_decay'])
                opt_state = optimizer.init(params)

                # Pure Functional Update Step (Now handles splitting the dropout key)
                @jax.jit
                def update_step(p, s, o_state, xs, yt, rng_key):
                    # Split the key: use one for this step's dropout, keep the other for the next step
                    rng_key, drop_key = jax.random.split(rng_key)

                    def loss_fn(weights, current_state):
                        preds, new_state = cnn_forward(
                            weights, current_state, xs, Y_center, Y_scale, self.hp,
                            rng_key=drop_key, is_training=True
                        )

                        # Wrap-aware residual on torus, plain `yt - preds`
                        # on euclidean. Captured at trace time so the JIT
                        # graph specialises on the metric tag.
                        residual = signed_diff_jax(
                            yt, preds,
                            metric=self.manifold_metric, period=self.manifold_period,
                        )

                        # Evaluate strictly based on the dictionary configuration
                        if self.hp['loss_function'] == 'huber':
                            delta = self.hp['huber_delta']
                            abs_diff = jnp.abs(residual)

                            # Piecewise Huber: 0.5 * x^2 for small errors, linear for outliers
                            huber_elements = jnp.where(
                                abs_diff <= delta,
                                0.5 * (abs_diff ** 2),
                                delta * (abs_diff - 0.5 * delta)
                            )
                            loss = jnp.mean(jnp.sum(huber_elements, axis=-1))
                        else:
                            # Standard Mean Squared Error
                            loss = jnp.mean(jnp.sum(residual ** 2, axis=-1))

                        return loss, new_state

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss, s_new), grads = grad_fn(p, s)
                    updates, o_state_new = optimizer.update(grads, o_state, p)
                    p_new = optax.apply_updates(p, updates)

                    return p_new, s_new, o_state_new, rng_key

                best_err = float('inf')
                best_params, best_state = None, None
                patience_counter = 0

                # Active Patience limit read from dict
                patience_limit = self.hp['null_patience'] if strategy == 'null' else self.hp['patience']

                for epoch in range(self.hp['epochs']):
                    if self.hp['use_kde_weights']:
                        # Normalize KDE inverse-density weights to sum to 1.0
                        p_weights = w_tr / np.sum(w_tr)
                        # Draw a full epoch of samples using true continuous probabilities
                        b_idx = rng.choice(len(Y_tr), size=len(Y_tr), p=p_weights, replace=True)
                    else:
                        b_idx = get_grid_balanced_indices(Y_tr, self.hp['grid_size'], self.hp['samples_per_cell'], rng=rng)
                        rng.shuffle(b_idx)

                    for b in range(len(b_idx) // self.hp['batch_size']):
                        idx = b_idx[b * self.hp['batch_size']:(b + 1) * self.hp['batch_size']]

                        # 1. Temporal Warping
                        warps = rng.uniform(1.0 - warp_range, 1.0 + warp_range, len(idx))
                        X_batch = apply_temporal_warping(X_tr[idx], warps)

                        # 2. Kinematic Masking (Dynamic Interception)
                        if self.hp['use_kinematic_masking']:
                            X_batch = apply_kinematic_masking(
                                X_batch,
                                mask_prob=self.hp['masking_prob'],
                                mask_length=self.hp['masking_length_frames'],
                                rng=rng
                            )

                        # 3. Pass to JAX Update Step
                        params, state, opt_state, dropout_rng = update_step(
                            params, state, opt_state, jnp.array(X_batch), jnp.array(Y_tr[idx]), dropout_rng
                        )

                    if epoch % 5 == 0:
                        Y_pred_te = evaluate_batched(params, state, jnp.array(X_te))
                        err = float(jnp.mean(jnp.sqrt(jnp.sum(signed_diff_jax(
                            jnp.array(Y_te), Y_pred_te,
                            metric=self.manifold_metric, period=self.manifold_period,
                        ) ** 2, axis=-1))))

                        if err < best_err - 0.001:
                            best_err = err
                            # JAX arrays are immutable, so a shallow dict copy is
                            # sufficient to snapshot the best weights / BN state;
                            # deepcopy would waste allocations on the array payload.
                            best_params = dict(params)
                            best_state = dict(state)
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience_limit:
                            break

                Y_pred_final = evaluate_batched(best_params, best_state, jnp.array(X_te))
                fold_results[f'Y_pred_{strategy}'] = Y_pred_final
                fold_results[f'error_{strategy}'] = best_err

                print(f"  > [{strategy.capitalize():<6}] Euclidean Err: {best_err:.4f}")

                if strategy == 'actual':
                    actual_errors.append(best_err)
                    best_actual_params_list.append(best_params)
                    best_actual_states_list.append(best_state)

            deep_storage['cross_validation'].append(fold_results)

        # ---------------------------------------------------------
        # PHASE 2: POST-HOC PERMUTATION FEATURE IMPORTANCE
        # ---------------------------------------------------------
        print("\n" + "=" * 50)
        print(" PHASE 2: POST-HOC PERMUTATION FEATURE IMPORTANCE")
        print("=" * 50)

        best_fold_idx = int(np.argmin(actual_errors))
        best_params = best_actual_params_list[best_fold_idx]
        best_state = best_actual_states_list[best_fold_idx]
        test_idx_final = folds[best_fold_idx][1]

        X_te_base = np.array(X_seq[test_idx_final])
        Y_te_final = np.array(Y[test_idx_final])
        base_err = actual_errors[best_fold_idx]

        importance_means, importance_stds, raw_importance = {}, {}, {}
        feature_snrs = {}
        significant_features = []

        for f_idx, feat_name in enumerate(features):
            feat_scores = []
            for k in range(perm_iters):
                X_perm = X_te_base.copy()
                perm_idx = rng.permutation(len(Y_te_final))

                # Decouple the specific feature across the trial dimension
                X_perm[:, f_idx, :] = X_perm[perm_idx, f_idx, :]

                Y_pred_perm = evaluate_batched(best_params, best_state, jnp.array(X_perm))
                err_perm = float(jnp.mean(jnp.sqrt(jnp.sum(signed_diff_jax(
                    jnp.array(Y_te_final), Y_pred_perm,
                    metric=self.manifold_metric, period=self.manifold_period,
                ) ** 2, axis=-1))))

                delta_e = err_perm - base_err
                feat_scores.append(delta_e)

            mu, sigma = np.mean(feat_scores), np.std(feat_scores)

            # Compute Signal-to-Noise Ratio (safeguard against zero variance)
            snr = mu / sigma if sigma > 1e-9 else 0.0

            raw_importance[feat_name] = feat_scores
            importance_means[feat_name] = mu
            importance_stds[feat_name] = sigma
            feature_snrs[feat_name] = snr

            # Apply the strict SNR > 3 threshold defined in the Methods
            sig_flag = "*" if snr > 3.0 else ""
            if snr > 3.0:
                significant_features.append(feat_name)

            print(f"   [{feat_name:<25}] Delta E: {mu:+.4f} (±{sigma:.4f}) | SNR: {snr:.2f} {sig_flag}")

        sorted_features = sorted(importance_means.keys(), key=lambda x: importance_means[x], reverse=True)

        deep_storage['feature_importance'] = {
            'best_fold_idx': best_fold_idx,
            'raw_scores': raw_importance,
            'means': importance_means,
            'stds': importance_stds,
            'snrs': feature_snrs,
            'ranked_features': sorted_features,
            'significant_features': significant_features
        }

        # Resolve the save path BEFORE Phase 3 so we can checkpoint between
        # phases. The same `save_path` is rewritten at the end with the
        # fully-populated `deep_storage`; a Phase-3 failure therefore no
        # longer destroys the training + feature-importance work that
        # already lives in memory at this point.
        source_file = pathlib.Path(data_blocks['source_pkl_path']).name

        # Explicit Multi-line parsing to avoid summarizing logic
        if "male_mute_partner" in source_file:
            sex_mod = "male_mute_partner"
        elif "female" in source_file:
            sex_mod = "female"
        else:
            sex_mod = "male"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cnn_manifold_integrated_predictions_{sex_mod}_{timestamp}.pkl"

        save_dir = pathlib.Path(self.modeling_settings['io']['save_directory'])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        def _checkpoint_deep_storage(stage_label: str) -> None:
            """Defensive serialisation hook called between phases.

            Converts every JAX device array currently held in `deep_storage`
            into NumPy, then writes the resulting object to `save_path` via
            pickle. The intent is to guarantee that, even if the very next
            phase of `run_cnn_training` raises, the user keeps every byte of
            the work produced up to this point on disk. The final write at
            the end of this method overwrites the same path with the
            fully-populated structure.

            Parameters
            ----------
            stage_label : str
                Human-readable identifier of the phase whose output is being
                persisted (e.g. "post-Phase-2"). Used only for the console
                breadcrumb line.
            """

            host_snapshot = jax.device_get(deep_storage)
            with save_path.open('wb') as fh:
                pickle.dump(host_snapshot, fh)
            print(f"[CHECKPOINT::{stage_label}] saved deep_storage to {save_path}")

        # Persist Phase-1 + Phase-2 results before Phase 3 begins.
        _checkpoint_deep_storage("post-Phase-2")

        # ---------------------------------------------------------
        # PHASE 3: INPUT-GRADIENT SALIENCY EXTRACTION
        # ---------------------------------------------------------
        print("\n" + "=" * 50)
        print(" PHASE 3: INPUT-GRADIENT SALIENCY EXTRACTION")
        print("=" * 50)

        saliency_cfg = self.hp['saliency']
        deep_storage['saliency_maps'] = {}

        if not saliency_cfg['enable']:
            print("  [skip] saliency.enable=False; leaving saliency_maps empty")
        else:
            segmentation = saliency_cfg['segmentation']
            if segmentation == 'supercategory':
                labels_all = cluster_labels_super
            elif segmentation == 'category':
                labels_all = cluster_labels_cat
            else:
                raise ValueError(
                    f"saliency.segmentation must be 'supercategory' or 'category'; "
                    f"got {segmentation!r}"
                )

            if labels_all is None:
                raise RuntimeError(
                    f"saliency.segmentation='{segmentation}' but the modeling pickle does "
                    f"not carry per-USV {segmentation} labels. Re-extract the data with "
                    f"the updated pipeline (extract_and_save_continuous_data) before "
                    f"training, or set saliency.enable=false."
                )

            labels_all_np = np.asarray(labels_all)
            labels_test = labels_all_np[test_idx_final]

            # Empirical centres from ALL Y + labels. The centres are a
            # property of the data labelling, not the trained model, so no
            # train/test leakage is introduced by pooling here.
            centres = derive_cluster_centers_empirically(
                np.asarray(Y),
                labels_all_np,
                drop_label=saliency_cfg['noise_label'],
                metric=self.manifold_metric,
                period=self.manifold_period,
            )

            if len(centres) < 2:
                raise RuntimeError(
                    f"saliency.segmentation='{segmentation}' produced only {len(centres)} "
                    f"cluster centre(s); need at least 2 for the alpha-gap radius rule. "
                    f"Check label coverage in the modeling pickle."
                )

            geometry = derive_cluster_geometry(
                centres,
                alpha=saliency_cfg['alpha'],
                mode=saliency_cfg['radius_mode'],
                metric=self.manifold_metric,
                period=self.manifold_period,
            )

            # Persist the derived geometry alongside the saliency tensors so
            # downstream plotting code can reconstruct each cluster's
            # circle without reading the modeling pickle a second time.
            deep_storage['cluster_geometry'] = {
                'segmentation': segmentation,
                'alpha': float(saliency_cfg['alpha']),
                'radius_mode': str(saliency_cfg['radius_mode']),
                'noise_label': saliency_cfg['noise_label'],
                'metric': self.manifold_metric,
                'period': float(self.manifold_period),
                'clusters': {
                    str(int(k) if float(k).is_integer() else k): {
                        'centroid': np.asarray(v['centroid'], dtype=np.float64),
                        'radius': float(v['radius']),
                        'nearest_neighbour_distance': float(v['nearest_neighbour_distance']),
                    }
                    for k, v in geometry.items()
                },
            }

            for label, g in geometry.items():
                centroid = np.asarray(g['centroid'], dtype=np.float64)
                radius = float(g['radius'])
                label_int = int(label) if float(label).is_integer() else label
                cluster_name = f"{segmentation}_{label_int}"

                # 1. Compute the saliency for the whole test batch toward
                #    this cluster's empirically-derived centroid.
                saliency_maps = self.compute_centroid_saliency(
                    best_params, best_state,
                    jnp.array(X_te_base), Y_center, Y_scale,
                    polygon_centroid=tuple(centroid),
                )

                # 2. Dual filter: a USV is a true positive for this cluster
                #    iff its ground-truth Y lies inside the (wrap-aware)
                #    circle AND its label matches the cluster's label.
                in_circle = usv_in_circle(
                    Y_te_final, centroid, radius,
                    metric=self.manifold_metric,
                    period=self.manifold_period,
                )
                label_match = labels_test == label
                keep = in_circle & label_match

                # 3. Slice the saliency tensor down to the dual-filter set.
                true_positive_saliency = saliency_maps['contrastive_saliency'][keep]

                print(
                    f"      -> {cluster_name}: r={radius:.4f}  "
                    f"in_circle={int(in_circle.sum())}  "
                    f"label_match={int(label_match.sum())}  "
                    f"both={int(keep.sum())}"
                )

                deep_storage['saliency_maps'][cluster_name] = {
                    'contrastive_saliency': true_positive_saliency,
                    'centroid': centroid,
                    'radius': radius,
                    'n_inside_circle': int(in_circle.sum()),
                    'n_label_match': int(label_match.sum()),
                    'n_dual_filter_pass': int(keep.sum()),
                }

        # ---------------------------------------------------------
        # SERIALIZATION
        # ---------------------------------------------------------
        print("\nConverting JAX device arrays to NumPy and saving Deep Storage...")
        numpy_storage = jax.device_get(deep_storage)

        with save_path.open('wb') as f:
            pickle.dump(numpy_storage, f)

        print(f"Success. Deep Storage saved to: {save_path}")
