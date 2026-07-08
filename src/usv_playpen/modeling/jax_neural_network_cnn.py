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
    a Heavy Path (Depthwise -> Pointwise -> AvgPool) to smoothly halve the temporal
    dimension while preserving gradient flow.
3.  Translation Variance (Flattening): Preserves the absolute timing of behavioral
    motifs by flattening the final temporal matrix instead of globally averaging it.
    When `use_hybrid_flatten` is active, the flattened features are additionally
    concatenated with a global-average-pooled summary so both the strictly-timed
    motifs and the time-invariant averages reach the dense head.
4.  Pure Functional JAX: Explicitly manages all convolutional and Batch Normalization
    states without relying on high-level OOP wrappers like Flax or Haiku.
"""

import os

# Force efficient memory allocation to prevent OOMs
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Disable aggressive XLA auto-tuning to prevent cuDNN/Triton compilation hangs
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

# Silence non-critical XLA, Triton, and Abseil optimization warnings (E0324... timestamps)
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

from .acoustic_manifold_geometry import (
    derive_cluster_centers_empirically,
    derive_cluster_geometry,
    usv_in_circle,
)
from .manifold_metric import (
    pairwise_distance,
    signed_diff_jax,
    resolve_manifold_metric,
    sin_cos_encode_jax,
    angle_decode_jax,
)
from .modeling_usv_manifold_position import (
    get_stratified_spatial_splits_stable as _manifold_spatial_splits,
)
from ..os_utils import resolve_modeling_setting

# Initial spatial-CV session-split matching tolerance, read from the settings
# block (it auto-widens at runtime; this is the starting value) rather than a
# bare 0.05 default.
_SESSION_SPLIT_INITIAL_TOLERANCE = resolve_modeling_setting('model_params', 'session_split_initial_tolerance')


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

# PHASE 1: UTILITIES & DATA AUGMENTATION

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
        draws. If None, a deterministic generator seeded with 0 is created so standalone
        calls are reproducible; the production training loop always threads its own live
        seeded generator (derived from ``self.random_seed``), which advances across batches.

    Returns
    -------
    masked_batch : np.ndarray
        The augmented 3D tensor with random temporal chunks zeroed out.
    """
    if rng is None:
        rng = np.random.default_rng(0)

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

    # Per-sequence query times, shared across feature channels:
    # `t_query[i] = center + (input_t - center) * warp_factors[i]`, clipped to
    # the valid index range. Shape (Batch, Time).
    t_query = center + (input_t[None, :] - center) * warp_factors[:, None]
    t_query = np.clip(t_query, 0.0, n_bins - 1)

    # Vectorised linear interpolation replacing the per-(batch, feature)
    # `np.interp` loop. Because every query lies in `[0, n_bins - 1]`, the
    # left index `t0 = floor(t_query)` and right index `t1 = min(t0 + 1,
    # n_bins - 1)` bracket it, and the result is the lerp
    # `x[t0] * (1 - frac) + x[t1] * frac` with `frac = t_query - t0`. At an
    # exact integer / right-edge query `frac == 0` (and `t1 == t0` at the
    # edge), reproducing `np.interp`'s endpoint clamp. For the pipeline's
    # float32 kinematics this is bit-identical to the previous per-channel
    # `np.interp` (verified on random data); in float64 it differs only by
    # sub-epsilon (~1e-16) reduction-order noise.
    t0 = np.clip(np.floor(t_query).astype(int), 0, n_bins - 1)
    t1 = np.minimum(t0 + 1, n_bins - 1)
    frac = (t_query - t0)[:, None, :]                       # (Batch, 1, Time)

    b_idx = np.arange(batch_size)[:, None, None]
    f_idx = np.arange(n_feats)[None, :, None]
    x0 = x_seq[b_idx, f_idx, t0[:, None, :]]                # (Batch, Feats, Time)
    x1 = x_seq[b_idx, f_idx, t1[:, None, :]]
    # `frac` is float64 (derived from the float64 `t_query`), so the lerp
    # promotes to float64; cast back to the input dtype to match the original
    # loop, which wrote float64 `np.interp` results into a `np.zeros_like(x_seq)`
    # buffer (downcasting to `x_seq.dtype` on assignment). For float32 input this
    # round-trip is bit-identical to that per-element downcast.
    warped_batch = (x0 * (1.0 - frac) + x1 * frac).astype(x_seq.dtype)

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
        a deterministic generator seeded with 0 is created so standalone calls are
        reproducible; the production training loop always threads its own live seeded
        generator (derived from ``self.random_seed``), which advances across batches.

    Returns
    -------
    balanced_indices : np.ndarray
        A 1D array of randomly selected data indices. The final length is the
        sum of the density-scaled per-cell draws.
    """

    if rng is None:
        rng = np.random.default_rng(0)

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


# PHASE 2: PURE JAX RESNET-1D ARCHITECTURE

# Number of manifold axes the CNN regresses. The whole pipeline (Y,
# Y_center, Y_scale, fold metrics, saliency centroids) is hard-coded
# around two-axis manifolds (`(umap1, umap2)`), but pulling the value
# through a single helper keeps the output-head plumbing explicit and
# leaves a single place to revisit if a future modality ever ships a
# D != 2 manifold.
def _output_axes_count(hp: Dict[str, Any]) -> int:
    """
    Returns the number of manifold axes the CNN predicts. The CNN
    pipeline assumes a 2-D acoustic manifold (`vae_umap{1,2}` /
    `qlvm_dim{1,2}`); this helper centralises that constant so the
    output-head sizing logic doesn't sprinkle bare `2`s through
    `init_cnn_params_and_state`, `cnn_forward`, and the loss block.

    Parameters
    ----------
    hp : dict
        Hyperparameter dictionary. Currently unused (reserved): the
        axis count is a hard-coded constant (`2`) and is NOT derived
        from any `hp` key. The argument is kept so a future modality
        shipping a `D != 2` manifold can switch this helper to a
        config-driven lookup without touching the call sites in
        `init_cnn_params_and_state`.

    Returns
    -------
    int
        The number of manifold axes the CNN regresses (always 2).
    """

    return 2


def _use_sin_cos_torus_output(hp: Dict[str, Any]) -> bool:
    """
    Returns True if the CNN should use the per-axis `(sin, cos)`
    output encoding on the torus.

    Two-key gate, evaluated at Python compile time so the JIT graph
    specialises once: the manifold metric must be `'torus'` AND
    `cnn_torus_output_encoding` must be the string `'sin_cos'`. The
    encoding flag accepts `'sin_cos'` (the principled fix) or
    `'raw'` (the legacy `tanh * Y_scale + Y_center` head, kept
    available as an ablation knob). On `'euclidean'` the encoding
    flag is ignored — euclidean targets have no wrap-aware loss
    degeneracy and the original head is preserved verbatim.

    A missing `cnn_torus_output_encoding` key is treated as a
    settings-file bug rather than silently defaulting; the strict
    `hp[...]` lookup mirrors the project convention for hp access.

    Parameters
    ----------
    hp : dict
        Hyperparameter dictionary. Must contain `manifold_metric`
        and, on torus runs, `cnn_torus_output_encoding`.

    Returns
    -------
    bool
        True iff the network should produce per-axis `(sin, cos)`
        outputs and be trained with chord-distance MSE.
    """

    if hp['manifold_metric'] != 'torus':
        return False
    encoding = hp['cnn_torus_output_encoding']
    if encoding not in ('sin_cos', 'raw'):
        raise ValueError(
            f"cnn_torus_output_encoding must be 'sin_cos' or 'raw'; "
            f"got {encoding!r}"
        )
    return encoding == 'sin_cos'


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

    block_channels = hp['block_channels']
    channels = [in_channels, *block_channels]

    # Read structural inputs directly from dict
    std_kernel = hp['kernel_size']
    inc_kernels = hp['inception_kernel_sizes']
    se_reduction = hp['se_reduction']
    hidden_dim = hp['hidden_dim']
    use_inception = hp['use_inception_kernels']

    current_time = time_steps

    for i in range(len(block_channels)):
        c_in = channels[i]
        c_out = channels[i + 1]

        if i == 0 and use_inception:
            # === Multi-Scale Inception Block 0 ===
            # Initialize N parallel depthwise kernels for different temporal fields.
            #
            # The per-kernel PRNG keys are drawn from a dedicated, disjoint
            # offset (`inc_dw_key_offset`) rather than from `k[j]`. The previous
            # `k[j]` indexing collided with the per-block reserved key layout
            # used everywhere else (block `i` consumes `k[i*10 .. i*10+4]`): for
            # block 0 those are the depthwise/pointwise/shortcut keys `k[0..2]`
            # and the SE keys `k[3]`, `k[4]`. With four or more inception
            # kernels, `k[3]` / `k[4]` would be reused for both a depthwise
            # kernel and an SE matrix, correlating their initial weights. The
            # offset of 40 sits clear of every per-block slot (`k[0..34]` for
            # blocks 0-3) and of the dense-head keys (`k[-2]`, `k[-1]`), so the
            # kernels are independently seeded for any supported kernel count.
            inc_dw_key_offset = 40
            if inc_dw_key_offset + len(inc_kernels) > len(k):
                raise ValueError(
                    f"Too many inception kernels ({len(inc_kernels)}) for the reserved "
                    f"PRNG-key budget: keys {inc_dw_key_offset}..{inc_dw_key_offset + len(inc_kernels) - 1} "
                    f"would exceed the {len(k)} split keys. Reduce "
                    f"`inception_kernel_sizes` or widen the `jax.random.split` count."
                )
            for j, k_size in enumerate(inc_kernels):
                params[f'b0_dw_w_{j}'] = jax.random.normal(k[inc_dw_key_offset + j], (c_in, 1, k_size)) * jnp.sqrt(2.0 / k_size)

            # Pointwise mixes parallel scales (N_kernels * in_channels -> c_out)
            pw_in_dim = c_in * len(inc_kernels)
            params['b0_pw_w'] = jax.random.normal(k[10], (c_out, pw_in_dim, 1)) * jnp.sqrt(2.0 / pw_in_dim)
            params['b0_pw_b'] = jnp.zeros((1, c_out, 1))

            # Shortcut matches scale and channel expansion
            params['b0_sc_w'] = jax.random.normal(k[11], (c_out, c_in, 1)) * jnp.sqrt(2.0 / c_in)
            params['b0_sc_b'] = jnp.zeros((1, c_out, 1))
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

    params['dense1_w'] = jax.random.normal(k[-2], (padded_flat_size, hidden_dim)) * jnp.sqrt(2.0 / padded_flat_size)
    params['dense1_b'] = jnp.zeros(hidden_dim)

    # Output-head dimensionality. On the torus path with `sin_cos`
    # encoding the network predicts the per-axis `(sin 2pi y, cos 2pi y)`
    # pair instead of the raw scalar, so the final dense layer doubles
    # in width. Every other manifold-metric / encoding combination
    # keeps the original 2-output projection. The branch is resolved at
    # init time (Python-side string compare) and produces a `params`
    # dict whose shapes match exactly the cnn_forward path selected by
    # the same flags, so no runtime conditional inside the JIT loop.
    if _use_sin_cos_torus_output(hp):
        out_dim = 2 * _output_axes_count(hp)
    else:
        out_dim = _output_axes_count(hp)

    params['dense2_w'] = jax.random.normal(k[-1], (hidden_dim, out_dim)) * jnp.sqrt(2.0 / hidden_dim)
    params['dense2_b'] = jnp.zeros(out_dim)

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
        Predicted manifold output of shape `(Batch, 2)` on euclidean
        and on the legacy `'raw'` torus head; `(Batch, 4)` on the
        torus `'sin_cos'` head (per-axis `(sin, cos)` interleaving).
        Downstream consumers wanting a 2-D angle vector should call
        `angle_decode_jax(predictions, period)` from `manifold_metric`.
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

    for i in range(len(hp['block_channels'])):
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
                lhs=path_a, rhs=params['b0_pw_w'], window_strides=(1,), padding='VALID',
                dimension_numbers=dimension_numbers, feature_group_count=1
            ) + params['b0_pw_b']
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

    if _use_sin_cos_torus_output(hp):
        # Torus `sin_cos` head: emit raw (sin, cos) per axis, no tanh
        # squash and no Y_center / Y_scale rescale. A linear output is
        # the standard choice for (sin, cos) regression — tanh would
        # saturate gradients exactly when the network wants to commit
        # to a confident `(±1, 0)` direction, and Y_center / Y_scale
        # would map the natural [-1, 1] target range onto the original
        # [0, period) data range and re-introduce a seam attractor.
        predictions = logits
    else:
        # Original euclidean head (also the legacy torus `'raw'`
        # head): squash through tanh and rescale into the empirical
        # data bounding box. Required on euclidean to keep predictions
        # inside the manifold's physical support; required on the
        # legacy torus run for backward-compat with previously-saved
        # artifacts.
        predictions = Y_center + jnp.tanh(logits) * Y_scale

    return predictions, new_state


# PHASE 3: LEARNING RATE SCHEDULER

def build_lr_schedule(peak_lr: float, total_epochs: int, steps_per_epoch: int, warmup_fraction: float = 0.10) -> optax.Schedule:
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
    warmup_fraction : float
        Fraction of total training steps spent linearly warming the learning rate up
        to ``peak_lr`` before the cosine decay begins. Defaults to 0.10.

    Returns
    -------
    schedule : optax.Schedule
        A callable Optax schedule that takes the current step count and returns the
        exact learning rate for that specific gradient update.
    """

    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(warmup_fraction * total_steps)

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


# PHASE 4: THE TRAINING ENGINE

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
       Delta manifold-distance error (Euclidean or wrap-aware torus, per
       manifold_metric).
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

        # Promote the resolved manifold metric / period into the `hp`
        # dict so the module-level `init_cnn_params_and_state` and
        # `cnn_forward` can route on the torus-vs-euclidean branch
        # without a second positional argument. Both keys are strings
        # / floats (hashable), so the `HashableDict.__hash__` contract
        # is preserved and the JIT cache key for the forward / update
        # functions still flips deterministically when the metric
        # changes between runs.
        self.hp['manifold_metric'] = self.manifold_metric
        self.hp['manifold_period'] = float(self.manifold_period)

        # Optional knob: when set, `run_cnn_training` only actually
        # trains the listed fold indices and emits placeholder
        # entries (with `error = +inf`) for the rest. Used by the
        # Phase-3 recovery script when the only need is to regenerate
        # the best fold's weights without paying for all 10. `None`
        # (the default) trains every fold as usual.
        self.restrict_to_fold_indices: list[int] | None = None

    def get_stratified_spatial_splits_stable(self, groups: np.ndarray,
                                             Y: np.ndarray,
                                             split_strategy: str = 'session',
                                             n_clusters: int = 15,
                                             test_prop: float = 0.2,
                                             n_splits: int = 100,
                                             tolerance: float = _SESSION_SPLIT_INITIAL_TOLERANCE,
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

    def _compute_global_saliency_template(self, params: Dict[str, jax.Array], state: Dict[str, jax.Array],
                                          X_te: jax.Array, Y_center: jax.Array, Y_scale: jax.Array) -> np.ndarray:
        """
        Description
        -----------
        Compute the cluster-invariant global ("postural baseline") saliency template
        used by :meth:`compute_centroid_saliency` for its contrastive subtraction.

        The global objective is the wrap-aware L1 magnitude of the prediction from the
        origin -- it never references any cluster centroid -- so the trial-averaged
        Input x Gradient template it produces is identical for every cluster. Factoring
        it out lets the per-cluster saliency loop compute it once instead of re-running
        a full-batch backward pass per cluster (K redundant passes -> 1). The body is
        the exact global path previously inlined in compute_centroid_saliency, so the
        returned template is byte-identical.

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

        Returns
        -------
        np.ndarray
            The trial-averaged global saliency template of shape (Features, Time_Bins),
            i.e. ``mean(X_te * |grad_glob|, axis=0)``.
        """

        torus_sin_cos = _use_sin_cos_torus_output(self.hp)

        def _decode(preds_raw):
            if torus_sin_cos:
                return angle_decode_jax(preds_raw, jnp.asarray(self.manifold_period))
            return preds_raw

        # L1 "distance from origin" baseline (wrap-aware on torus); the origin is the
        # 2-D zero vector, matching jnp.zeros_like(polygon_centroid_jax) in the
        # original inlined path.
        origin_jax = jnp.zeros(2)

        def glob_scalar_fn(x_single):
            preds_raw, _ = cnn_forward(params, state, x_single[jnp.newaxis, ...],
                                       Y_center, Y_scale, self.hp, is_training=False)
            preds = _decode(preds_raw)

            diff_origin = signed_diff_jax(
                preds[0], origin_jax,
                metric=self.manifold_metric, period=self.manifold_period,
            )
            return jnp.sum(jnp.abs(diff_origin))

        compute_glob_grad = jax.grad(glob_scalar_fn)

        @jax.jit
        def get_batched_glob_grads(x_batch):
            return jax.vmap(compute_glob_grad)(x_batch)

        batch_size = self.hp['batch_size']
        n_samples = len(X_te)
        glob_grads_list = []

        for i in range(0, n_samples, batch_size):
            batch_x = X_te[i:i + batch_size]
            actual_size = len(batch_x)

            # JAX requires strict static shapes. Pad the final remainder batch.
            if actual_size < batch_size:
                pad_width = ((0, batch_size - actual_size), (0, 0), (0, 0))
                batch_x = jnp.pad(batch_x, pad_width, mode='constant')

            batch_g_grad = get_batched_glob_grads(batch_x)
            glob_grads_list.append(np.array(batch_g_grad[:actual_size]))

        glob_grads = np.concatenate(glob_grads_list, axis=0)
        X_te_np = np.array(X_te)
        glob_saliency = X_te_np * np.abs(glob_grads)
        return np.mean(glob_saliency, axis=0)

    def compute_centroid_saliency(self, params: Dict[str, jax.Array], state: Dict[str, jax.Array],
                                  X_te: jax.Array, Y_center: jax.Array, Y_scale: jax.Array,
                                  polygon_centroid: Tuple[float, float],
                                  global_template: np.ndarray) -> Dict[str, np.ndarray]:
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
        global_template : np.ndarray
            The cluster-invariant global saliency template of shape (Features, Time_Bins),
            precomputed once by :meth:`_compute_global_saliency_template` and subtracted
            from the region-specific saliency. Passed in (rather than recomputed per
            cluster) because the global baseline does not depend on the centroid.

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

        # On the torus `'sin_cos'` head `cnn_forward` returns a raw 4-D
        # `(sin, cos)` vector per axis; the saliency objectives below
        # need to operate on the 2-D angle representation so the
        # wrap-aware distance to the polygon centroid keeps the same
        # geometric meaning it had on euclidean / the legacy `'raw'`
        # head. The `_decode` closure folds the raw output back to a
        # 2-D angle via `atan2`. `atan2` is smooth and JAX-
        # differentiable, so the gradient flows back through the
        # decoding step into the network parameters; saliency for the
        # `sin_cos` head is therefore the gradient of "angle distance
        # to centroid" w.r.t. the input, exactly what the euclidean /
        # raw heads already compute.
        torus_sin_cos = _use_sin_cos_torus_output(self.hp)

        def _decode(preds_raw):
            if torus_sin_cos:
                # Build the period array only on the torus `sin_cos` path; on
                # euclidean / legacy `'raw'` runs the decode is the identity and
                # the period is never consulted.
                return angle_decode_jax(preds_raw, jnp.asarray(self.manifold_period))
            return preds_raw

        def region_scalar_fn(x_single):
            preds_raw, _ = cnn_forward(params, state, x_single[jnp.newaxis, ...],
                                       Y_center, Y_scale, self.hp, is_training=False)
            preds = _decode(preds_raw)

            diff = signed_diff_jax(
                preds[0], polygon_centroid_jax,
                metric=self.manifold_metric, period=self.manifold_period,
            )
            dist_to_centroid = jnp.sqrt(jnp.sum(diff ** 2))
            return -dist_to_centroid

        # The global ("postural baseline") saliency template is cluster-invariant
        # (its objective is the L1 distance from the origin, not from this centroid),
        # so it is computed once by `_compute_global_saliency_template` and passed in
        # via `global_template` instead of being recomputed here for every cluster.

        # 2. Transform the region scalar into its gradient function.
        compute_region_grad = jax.grad(region_scalar_fn)

        # 3. JIT compile the vectorized region-gradient function for fast batched execution
        @jax.jit
        def get_batched_grads(x_batch):
            return jax.vmap(compute_region_grad)(x_batch)

        # 4. Execute in memory-safe mini-batches
        batch_size = self.hp['batch_size']
        n_samples = len(X_te)

        region_grads_list = []

        for i in range(0, n_samples, batch_size):
            batch_x = X_te[i:i + batch_size]
            actual_size = len(batch_x)

            # JAX requires strict static shapes. Pad the final remainder batch.
            if actual_size < batch_size:
                pad_width = ((0, batch_size - actual_size), (0, 0), (0, 0))
                batch_x = jnp.pad(batch_x, pad_width, mode='constant')

            batch_r_grad = get_batched_grads(batch_x)

            # Truncate any padding and pull directly to CPU RAM as NumPy arrays
            region_grads_list.append(np.array(batch_r_grad[:actual_size]))

        # Reconstruct the full dataset gradients on the CPU
        region_grads = np.concatenate(region_grads_list, axis=0)

        # 5. Input * Gradient Saliency (Computed on CPU to save VRAM)
        X_te_np = np.array(X_te)
        region_saliency = X_te_np * np.abs(region_grads)

        # 6. Contrastive Subtraction against the precomputed global template
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

        # The per-fold training loop tracks `best_params` across epochs,
        # starting at `None` and only assigning inside the epoch loop. With
        # `epochs == 0` the loop never runs, `best_params` stays `None`, and the
        # post-loop `evaluate_batched(best_params, ...)` crashes with an opaque
        # error deep in the JIT trace. Fail fast and explicitly here instead.
        # `raise` rather than `assert` so the check survives under `python -O`.
        if int(self.hp['epochs']) < 1:
            raise ValueError(
                f"hyperparameters.deep_learning.cnn_continuous.epochs must be >= 1; "
                f"got {self.hp['epochs']!r}. With zero epochs no parameters are ever "
                f"trained and the final evaluation has no model to score."
            )

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

        # Pre-flight: if Phase 3 (saliency) is enabled, validate the
        # requested labels are present in the modeling pickle AND that
        # they resolve to at least two cluster centres BEFORE running
        # any folds. Without this the runner happily completes Phase 1
        # (10 folds of CNN training) and Phase 2 (permutation
        # importance) and only dies in Phase 3 — burning ~all of the
        # job's wall-clock time on work that was destined to fail.
        saliency_cfg = self.hp['saliency']
        if saliency_cfg['enable']:
            preflight_seg = saliency_cfg['segmentation']
            if preflight_seg not in ('supercategory', 'category'):
                raise ValueError(
                    f"saliency.segmentation must be 'supercategory' or 'category'; "
                    f"got {preflight_seg!r}"
                )
            preflight_labels = (
                cluster_labels_super if preflight_seg == 'supercategory'
                else cluster_labels_cat
            )
            if preflight_labels is None:
                raise RuntimeError(
                    f"saliency.enable=true and saliency.segmentation="
                    f"'{preflight_seg}', but the modeling pickle does not carry "
                    f"per-USV {preflight_seg} labels. Re-extract the data with "
                    f"the updated pipeline (extract_and_save_continuous_data) "
                    f"before training, or set saliency.enable=false. Failing "
                    f"now (before Phase 1) to avoid burning training time on a "
                    f"run that will die in Phase 3."
                )
            preflight_centres = derive_cluster_centers_empirically(
                np.asarray(Y),
                np.asarray(preflight_labels),
                drop_label=saliency_cfg['noise_label'],
                metric=self.manifold_metric,
                period=self.manifold_period,
            )
            if len(preflight_centres) < 2:
                raise RuntimeError(
                    f"saliency.segmentation='{preflight_seg}' produces only "
                    f"{len(preflight_centres)} cluster centre(s) on this "
                    f"modeling pickle; need at least 2 for the alpha-gap "
                    f"radius rule. Check the label coverage / noise_label "
                    f"setting. Failing now (before Phase 1)."
                )

        n_feats = len(features)
        n_bins = data_blocks['num_bins']

        # Center/Scale boundaries that bind the tanh prediction head are computed
        # PER FOLD from the training split only (see the fold loop below). Deriving
        # them from the full Y here would let each fold's held-out test coordinates
        # co-define its prediction bounding box -- a mild train/test leakage that
        # optimistically biases the cross-validated generalization estimate.

        # Pull training dials directly from dict (Passivity Check: No .get())
        n_folds = self.hp['n_folds']
        batch_size = self.hp['batch_size']
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
                # Source-pickle provenance — pinned in metadata so any
                # checkpoint produced by this run can be paired back to
                # the modeling pickle it consumed without grepping
                # filesystem timestamps.
                'source_pkl_path': str(data_blocks['source_pkl_path']),
                # Manifold-metric provenance — downstream
                # visualisation code reads these to compute wrap-aware
                # distances on the saved Y_pred / Y_true arrays.
                'manifold_metric': self.manifold_metric,
                'manifold_period': self.manifold_period,
                # CNN output-head encoding provenance. On torus runs
                # this records whether the network produced raw 2-D
                # coords with `tanh * Y_scale + Y_center` ('raw',
                # legacy) or per-axis `(sin, cos)` decoded via atan2
                # ('sin_cos', principled fix). Euclidean runs always
                # use the 2-D tanh head and carry 'raw' for symmetry.
                # Downstream readers can branch on this to interpret
                # any per-fold diagnostics that depend on the encoding.
                'output_encoding': (
                    'sin_cos' if _use_sin_cos_torus_output(self.hp) else 'raw'
                ),
            },
            'cross_validation': [],
            'feature_importance': {}
        }

        # Resolve the persistent save path and the checkpoint helper
        # BEFORE Phase 1 starts so every phase boundary can checkpoint
        # to disk. Earlier this lived between Phase 2 and Phase 3,
        # which meant a Phase-2 failure (or any Phase-1 OOM) lost the
        # trained-weight tensors that only existed on the device. The
        # final write at the end of `run_cnn_training` rewrites the
        # same path with the fully-populated `deep_storage`.
        source_file = pathlib.Path(data_blocks['source_pkl_path']).name
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
            """
            Description
            -----------
            Defensive serialisation hook called between phases.
            Converts every JAX device array currently held in
            `deep_storage` into NumPy via `jax.device_get`, then
            writes the resulting object to `save_path` via pickle.
            The intent is to guarantee that, even if the very next
            phase of `run_cnn_training` raises, the user keeps every
            byte of the work produced up to this point on disk —
            including the per-fold trained weights, which are
            persisted into `deep_storage['cross_validation']` as
            `params_actual` / `state_actual` during Phase 1. The
            final write at the end of this method overwrites the
            same path with the fully-populated structure.

            Parameters
            ----------
            stage_label : str
                Human-readable identifier of the phase whose output
                is being persisted (e.g. "post-Phase-1"). Used only
                for the console breadcrumb line.

            Returns
            -------
            None
            """

            host_snapshot = jax.device_get(deep_storage)
            with save_path.open('wb') as fh:
                pickle.dump(host_snapshot, fh)
            print(f"[CHECKPOINT::{stage_label}] saved deep_storage to {save_path}")

        # `cnn_forward` returns raw network output: 2-D on euclidean
        # and on the legacy torus `'raw'` head, 4-D per-axis (sin, cos)
        # on the torus `'sin_cos'` head. The `_decode_predictions_for_eval`
        # closure folds the sin/cos representation back to a 2-D angle
        # vector so every consumer downstream of `evaluate_batched`
        # (early-stopping RMSE, permutation importance, saved
        # `Y_pred_*` arrays, plotting code) sees the same `(N, 2)`
        # shape regardless of which output encoding the run chose.
        torus_sin_cos = _use_sin_cos_torus_output(self.hp)
        manifold_period_jax = jnp.asarray(self.manifold_period)

        def _decode_predictions_for_eval(preds_raw):
            if torus_sin_cos:
                return angle_decode_jax(preds_raw, manifold_period_jax)
            return preds_raw

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

            preds_raw = jnp.concatenate(preds, axis=0)
            return _decode_predictions_for_eval(preds_raw)

        # Forward + backward pass, hoisted ABOVE the fold / strategy loops and
        # jitted once per `fit` call. It closes over only fit-invariant state
        # (`Y_center` / `Y_scale`, the hyperparameter dict, and the torus
        # output / period config) and is always called on fixed-size
        # `batch_size` batches, so this conv-heavy graph is traced and compiled
        # a SINGLE time and reused across every fold and strategy. Only the
        # small optimiser-update graph (`_apply_update` below), which depends on
        # the per-fold LR schedule, is re-jitted per fold — avoiding a full
        # forward / backward recompile on every fold and strategy.
        @jax.jit
        def _compute_grads(p, s, xs, yt, rng_key):
            # Split the key: use one half for this step's dropout, keep the
            # other for the next step.
            rng_key, drop_key = jax.random.split(rng_key)

            def loss_fn(weights, current_state):
                preds, new_state = cnn_forward(
                    weights, current_state, xs, Y_center, Y_scale, self.hp,
                    rng_key=drop_key, is_training=True
                )

                if torus_sin_cos:
                    # Torus `sin_cos` head: the network emits raw
                    # per-axis `(sin, cos)` and the loss is the
                    # plain Euclidean residual against the
                    # `(sin 2pi y, cos 2pi y)` encoding of the
                    # target. There is no wrap to apply — the
                    # `(sin, cos)` representation is intrinsically
                    # periodic, which is the whole point of
                    # the encoding (it removes the wrap-aware
                    # MSE degeneracy where every constant
                    # prediction has identical loss on a uniform
                    # torus target).
                    target_sc = sin_cos_encode_jax(yt, manifold_period_jax)
                    residual = preds - target_sc
                else:
                    # Wrap-aware residual on the legacy torus
                    # `'raw'` head and on euclidean: plain
                    # `yt - preds` on euclidean, folded into
                    # `(-period/2, period/2]` on torus. Captured
                    # at trace time so the JIT graph specialises
                    # on the metric tag.
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
            # value_and_grad with has_aux returns ((loss, new_state), grads); the
            # scalar loss is unused here (early-stopping RMSE is recomputed in the
            # epoch loop), so it is bound to `_loss` to mark it intentionally dead.
            (_loss, s_new), grads = grad_fn(p, s)

            return grads, s_new, rng_key

        # TRI-STRATEGY EXECUTION
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
            # Skip non-target folds when the runner has been configured
            # to only retrain a specific subset (Phase-3 recovery
            # workflow). Per-fold list positions are still filled so
            # downstream `best_fold_idx = argmin(actual_errors)`
            # naturally selects the only trained fold.
            if (self.restrict_to_fold_indices is not None
                    and fold not in self.restrict_to_fold_indices):
                print(f"\n--- FOLD {fold + 1}/{n_folds}  (skipped: restrict_to_fold_indices) ---")
                fold_results = {
                    'fold_idx':       fold,
                    'test_indices':   test_idx,
                    'Y_true':         Y[test_idx],
                    'skipped':        True,
                }
                deep_storage['cross_validation'].append(fold_results)
                actual_errors.append(float('inf'))
                best_actual_params_list.append(None)
                best_actual_states_list.append(None)
                _checkpoint_deep_storage(f"post-fold-{fold + 1}-skipped")
                continue

            print(f"\n--- FOLD {fold + 1}/{n_folds} ---")

            X_tr, X_te = X_seq[train_idx], X_seq[test_idx]
            Y_tr_base, Y_te = Y[train_idx], Y[test_idx]
            w_tr = w[train_idx]

            # Center/Scale boundaries that bind the tanh prediction head, derived
            # from THIS fold's training split only (Y_tr_base) so the held-out test
            # coordinates do not leak into the prediction bounding box.
            Y_center = jnp.array((np.max(Y_tr_base, 0) + np.min(Y_tr_base, 0)) / 2.0)
            Y_scale = jnp.array((np.max(Y_tr_base, 0) - np.min(Y_tr_base, 0)) / 2.0 * 1.1)

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
                    print(f"  > [Model-free prior] {self.manifold_metric.capitalize()} Err: {err:.4f}")
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
                    # Size `steps_per_epoch` from the grid-balanced index count
                    # WITHOUT consuming the shared training `rng`. The number of
                    # indices `get_grid_balanced_indices` returns depends only on
                    # the per-cell densities (each cell contributes a
                    # deterministic `max(1, ceil(...))` rows), not on the random
                    # draw, so a throwaway sizing rng yields the same length while
                    # leaving the shared `rng` state untouched for the epoch loop
                    # below. The previous code drew through `rng` here, advancing
                    # its state by one full grid-balanced sample before the first
                    # epoch's own `get_grid_balanced_indices(..., rng=rng)` draw.
                    sizing_rng = np.random.default_rng(self.random_seed + fold)
                    steps_per_epoch = len(get_grid_balanced_indices(Y_tr, self.hp['grid_size'], self.hp['samples_per_cell'], rng=sizing_rng)) // self.hp['batch_size']

                lr_schedule = build_lr_schedule(self.hp['learning_rate'], self.hp['epochs'], steps_per_epoch, self.hp['warmup_fraction']) if self.hp['use_scheduler'] else self.hp['learning_rate']

                # Optional weight-decay mask: when
                # `weight_decay_exclude_output_head` is True, the AdamW
                # decoupled-wd step skips `dense2_w` / `dense2_b`. The
                # output head is the only layer whose collapse to zero
                # produces a degenerate constant prediction on the
                # torus `sin_cos` head — leaving it un-decayed prevents
                # the "weight-decay drives weights to zero -> output
                # parks at constant -> no escape" failure mode that
                # motivated the principled-fix rewrite. On euclidean
                # the same exclusion is harmless: the head is small and
                # the empirical-bound tanh squash already constrains
                # its output range.
                if self.hp['weight_decay_exclude_output_head']:
                    wd_mask = jax.tree_util.tree_map(
                        lambda _: True, params,
                    )
                    wd_mask = {**wd_mask, 'dense2_w': False, 'dense2_b': False}
                else:
                    wd_mask = None

                optimizer = optax.adamw(
                    learning_rate=lr_schedule,
                    weight_decay=self.hp['weight_decay'],
                    mask=wd_mask,
                )
                opt_state = optimizer.init(params)

                # Optimiser-update step. A small graph (AdamW update + apply)
                # that depends on the per-fold LR schedule baked into
                # `optimizer`, so it is cheap to re-jit per fold / strategy; the
                # expensive conv forward / backward lives in the once-compiled
                # `_compute_grads` hoisted above the fold loop. Closes over the
                # per-fold `optimizer`, so it must be defined here (after the
                # optimiser is rebuilt) rather than hoisted.
                @jax.jit
                def _apply_update(p, o_state, grads):
                    updates, o_state_new = optimizer.update(grads, o_state, p)
                    p_new = optax.apply_updates(p, updates)
                    return p_new, o_state_new

                best_err = float('inf')
                best_params, best_state = None, None
                patience_counter = 0

                # Active Patience limit read from dict
                patience_limit = self.hp['null_patience'] if strategy == 'null' else self.hp['patience']

                # Normalize KDE inverse-density weights to sum to 1.0. w_tr is
                # fixed per fold/strategy, so this is epoch-invariant and hoisted
                # out of the epoch loop.
                if self.hp['use_kde_weights']:
                    p_weights = w_tr / np.sum(w_tr)

                for epoch in range(self.hp['epochs']):
                    if self.hp['use_kde_weights']:
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

                        # 2. Kinematic Masking (1D Cutout: randomly blind feature channels)
                        if self.hp['use_kinematic_masking']:
                            X_batch = apply_kinematic_masking(
                                X_batch,
                                mask_prob=self.hp['masking_prob'],
                                mask_length=self.hp['masking_length_frames'],
                                rng=rng
                            )

                        # 3. Forward / backward (once-compiled) then optimiser step
                        grads, state, dropout_rng = _compute_grads(
                            params, state, jnp.array(X_batch), jnp.array(Y_tr[idx]), dropout_rng
                        )
                        params, opt_state = _apply_update(params, opt_state, grads)

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

                print(f"  > [{strategy.capitalize():<6}] {self.manifold_metric.capitalize()} Err: {best_err:.4f}")

                if strategy == 'actual':
                    actual_errors.append(best_err)
                    best_actual_params_list.append(best_params)
                    best_actual_states_list.append(best_state)
                    # Persist the per-fold "actual" weights into
                    # `fold_results` so the post-Phase-1 checkpoint
                    # captures every byte the saliency phase needs.
                    # Without this, a Phase-2/Phase-3 crash drops the
                    # only in-process copy of the trained weights —
                    # the failure mode that motivated this rewrite.
                    fold_results['params_actual'] = best_params
                    fold_results['state_actual'] = best_state

            deep_storage['cross_validation'].append(fold_results)
            # Per-fold checkpoint — a Phase-1 mid-loop failure
            # (OOM during fold 7, say) now preserves folds 1..6
            # on disk instead of dropping everything.
            _checkpoint_deep_storage(f"post-fold-{fold + 1}")

        # Persist the per-fold predictions, errors, AND trained
        # weights before Phase 2 begins. A Phase-2 (permutation) or
        # Phase-3 (saliency) failure from here on no longer destroys
        # the ~hour of CNN training that just finished.
        _checkpoint_deep_storage("post-Phase-1")

        # PHASE 2: POST-HOC PERMUTATION FEATURE IMPORTANCE
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

            print(f"   [{feat_name:<25}] Delta Err: {mu:+.4f} (±{sigma:.4f}) | SNR: {snr:.2f} {sig_flag}")

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

        # Persist Phase-1 (predictions + per-fold weights) + Phase-2
        # (permutation importance) results before Phase 3 begins.
        # `save_path` and `_checkpoint_deep_storage` were hoisted to
        # the top of this method so the post-Phase-1 checkpoint
        # captured the trained weights too.
        _checkpoint_deep_storage("post-Phase-2")

        # PHASE 3: INPUT-GRADIENT SALIENCY EXTRACTION
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

            # The global saliency baseline does not depend on the cluster centroid, so
            # compute it once here instead of once per cluster inside the loop below.
            global_saliency_template = self._compute_global_saliency_template(
                best_params, best_state,
                jnp.array(X_te_base), Y_center, Y_scale,
            )

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
                    global_template=global_saliency_template,
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

        # SERIALIZATION
        print("\nConverting JAX device arrays to NumPy and saving Deep Storage...")
        numpy_storage = jax.device_get(deep_storage)

        with save_path.open('wb') as f:
            pickle.dump(numpy_storage, f)

        print(f"Success. Deep Storage saved to: {save_path}")
