"""
@author: bartulem
JAX (torch-free) re-implementation of the QLVM (QMC latent-variable model)
*inference* path, for embedding USV spectrograms into the trained model's fixed
toroidal latent space.

The model has no encoder: the torus is defined by a fixed quasi-random lattice
and a frozen ConvTranspose decoder. Embedding a new spectrogram is a forward
operation — decode every lattice point once (an "atlas"), score each new
spectrogram against the atlas under the Bernoulli likelihood the model was
trained with, and read off the posterior-weighted torus coordinate. Because the
architecture and weights are fixed, any new session embeds into the SAME torus.

This is a faithful port of ``qmc_deep_gen``'s ``models/qmc_base.py``
(``QMCLVM`` / ``TorusBasis``), ``models/sampling.py`` (lattice generators) and
``train/losses.py`` (``binary_lp``). The decoder weights are loaded from a
``.npz`` produced once (externally, where torch lives) from the training
checkpoint's ``state_dict``; usv-playpen never imports torch.

PARITY: the JAX ``conv_transpose2d`` reproduces ``torch.nn.ConvTranspose2d``'s
exact definition (validated against a pure-numpy reference in the tests). Before
trusting embeddings in production, confirm the parity test passes against a
torch-generated reference for the actual checkpoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Pixel clamp from the training loss (`binary_lp`) that keeps the logs finite.
_BINARY_LP_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Lattice generators (port of models/sampling.py)
# --------------------------------------------------------------------------- #
def _fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (``fib(0)=0``, ``fib(1)=1``)."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def gen_korobov_basis(a: int, num_dims: int, num_points: int) -> jnp.ndarray:
    """
    Description
    -----------
    Korobov rank-1 lattice over ``[0, 1)^num_dims``: row ``i`` is
    ``i * z / num_points mod 1`` with ``z_k = a**k mod num_points``.

    Parameters
    ----------
    a (int)
        Korobov generator (e.g. 76 for 1021 points).
    num_dims (int)
        Latent dimensionality.
    num_points (int)
        Number of lattice points.

    Returns
    -------
    lattice (jnp.ndarray)
        A ``(num_points, num_dims)`` lattice (not yet reduced mod 1).
    """
    z = np.array([a**k % num_points for k in range(num_dims)], dtype=np.float64)
    base_pts = np.arange(0, num_points)[:, None] * z[None, :] / num_points
    return jnp.asarray(base_pts)


def gen_fib_basis(m: int) -> jnp.ndarray:
    """
    Description
    -----------
    2D Fibonacci lattice: ``n = fib(m)`` points with generator ``[1, fib(m-1)]``.

    Parameters
    ----------
    m (int)
        Fibonacci index (``m >= 3``).

    Returns
    -------
    lattice (jnp.ndarray)
        A ``(fib(m), 2)`` lattice.
    """
    n = _fibonacci(m)
    z = np.array([1.0, _fibonacci(m - 1)], dtype=np.float64)
    return jnp.asarray(np.arange(0, n)[:, None] * z[None, :] / n)


def roberts_sequence(num_points: int, num_dims: int, root_iters: int = 10_000) -> jnp.ndarray:
    """
    Description
    -----------
    Roberts low-discrepancy sequence over ``[0, 1)^num_dims`` (not reduced mod 1).

    Parameters
    ----------
    num_points (int)
        Number of points.
    num_dims (int)
        Dimensionality.
    root_iters (int)
        Newton iterations for the generalized-golden-ratio root.

    Returns
    -------
    sequence (jnp.ndarray)
        A ``(num_points, num_dims)`` array.
    """
    x = 1.0
    for _ in range(root_iters):
        x = x - (x ** (num_dims + 1) - x - 1) / ((num_dims + 1) * (x**num_dims) - 1)
    basis = 1 - (1 / x ** (1 + np.arange(0, num_dims)))
    return jnp.asarray(np.arange(0, num_points)[:, None] * basis[None, :])


# --------------------------------------------------------------------------- #
# TorusBasis (port of models/qmc_base.py)
# --------------------------------------------------------------------------- #
def torus_basis_forward(data: jnp.ndarray) -> jnp.ndarray:
    """
    Description
    -----------
    Maps latent coordinates to the torus embedding
    ``[cos(2*pi*data), sin(2*pi*data)]`` (concatenated on the last axis), so a
    ``(N, d)`` input becomes ``(N, 2d)``.

    Parameters
    ----------
    data (jnp.ndarray)
        Latent coordinates, shape ``(N, d)``.

    Returns
    -------
    embedding (jnp.ndarray)
        Torus embedding, shape ``(N, 2d)``.
    """
    return jnp.concatenate([jnp.cos(2 * jnp.pi * data), jnp.sin(2 * jnp.pi * data)], axis=-1)


def torus_basis_reverse(data: jnp.ndarray) -> jnp.ndarray:
    """
    Description
    -----------
    Inverse of :func:`torus_basis_forward`: recovers latent coordinates in
    ``[0, 1)`` from a ``(N, 2d)`` ``[cos, sin]`` embedding via ``atan2``.

    Parameters
    ----------
    data (jnp.ndarray)
        Torus embedding, shape ``(N, 2d)`` ordered ``[cos..., sin...]``.

    Returns
    -------
    coords (jnp.ndarray)
        Latent coordinates in ``[0, 1)``, shape ``(N, d)``.
    """
    d = data.shape[-1] // 2
    angles = jnp.arctan2(data[:, d:], data[:, :d])
    angles = jnp.where(angles < 0, 2 * jnp.pi + angles, angles)
    return angles / (2 * jnp.pi)


# --------------------------------------------------------------------------- #
# Likelihood (port of train/losses.py:binary_lp)
# --------------------------------------------------------------------------- #
def binary_lp(samples: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """
    Description
    -----------
    Bernoulli log-likelihood ``log p(x_b | z_s)`` of each data spectrogram under
    each decoded lattice point, matching ``train/losses.py:binary_lp``. Returns a
    ``(B_data, K_grid)`` matrix where entry ``[b, s]`` sums, over all pixels,
    ``data_b * log(sample_s) + (1 - data_b) * log(1 - sample_s)``.

    Parameters
    ----------
    samples (jnp.ndarray)
        Decoded reconstructions, shape ``(K, C, H, W)``, values in ``[0, 1]``.
    data (jnp.ndarray)
        Data spectrograms, shape ``(B, C, H, W)``, values in ``[0, 1]``.

    Returns
    -------
    log_likelihood (jnp.ndarray)
        A ``(B, K)`` log-likelihood matrix.
    """
    samples = jnp.clip(samples, _BINARY_LP_EPS, 1 - _BINARY_LP_EPS)
    t1 = jnp.einsum("bjdl,sjdl->bs", data, jnp.log(samples))
    t2 = jnp.einsum("bjdl,sjdl->bs", 1 - data, jnp.log(1 - samples))
    return t1 + t2


# --------------------------------------------------------------------------- #
# Decoder forward (torch ConvTranspose2d / Linear, in JAX)
# --------------------------------------------------------------------------- #
def _linear(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Apply ``torch.nn.Linear``: ``y = x @ weight.T + bias`` (weight is (out, in))."""
    return x @ weight.T + bias


def conv_transpose2d(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    stride: int,
    padding: int,
    output_padding: int,
) -> jnp.ndarray:
    """
    Description
    -----------
    JAX implementation of ``torch.nn.ConvTranspose2d`` (groups=1, dilation=1).
    Reproduces torch's exact definition: input dilation by ``stride`` (fractional
    stride), transposed-conv padding ``(k - 1 - padding)`` plus ``output_padding``
    on the trailing edge, and a spatially-flipped, in/out-swapped kernel.

    Parameters
    ----------
    x (jnp.ndarray)
        Input, shape ``(N, C_in, H, W)``.
    weight (jnp.ndarray)
        torch weight, shape ``(C_in, C_out, kH, kW)``.
    bias (jnp.ndarray)
        Bias, shape ``(C_out,)``.
    stride (int)
        Stride.
    padding (int)
        torch ``padding``.
    output_padding (int)
        torch ``output_padding``.

    Returns
    -------
    out (jnp.ndarray)
        Output, shape ``(N, C_out, H_out, W_out)`` with
        ``H_out = (H - 1) * stride - 2 * padding + kH + output_padding``.
    """
    kh, kw = weight.shape[2], weight.shape[3]
    # OIHW kernel for a regular cross-correlation conv that equals the transpose:
    # swap in/out (I<->O) and flip the spatial axes.
    kernel = jnp.flip(jnp.transpose(weight, (1, 0, 2, 3)), axis=(2, 3))
    pad_h = (kh - 1 - padding, kh - 1 - padding + output_padding)
    pad_w = (kw - 1 - padding, kw - 1 - padding + output_padding)
    out = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=(pad_h, pad_w),
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return out + bias[None, :, None, None]


# The QLVM mouse decoder architecture (from build_qmc_decoder): two Linear layers,
# reshape to (64, 8, 8), then four ConvTranspose2d(stride=2, padding=1,
# output_padding=1) blocks 64->32->16->8->1, ReLU between, Sigmoid at the end.
# Indices match the nn.Sequential state_dict keys ("<idx>.weight"/"<idx>.bias").
_DECODER_CONV_INDICES = (3, 5, 7, 9)
_DECODER_RESHAPE = (64, 8, 8)
_CONV_STRIDE, _CONV_PADDING, _CONV_OUTPUT_PADDING = 2, 1, 1


def decoder_forward(latent_embeddings: jnp.ndarray, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Description
    -----------
    Runs the frozen QLVM decoder on torus embeddings, returning reconstructed
    spectrograms in ``[0, 1]``. ``params`` maps the decoder ``state_dict`` keys
    (``"0.weight"``, ``"0.bias"``, ``"1.weight"``, ..., ``"9.weight"``) to arrays
    (``decoder.`` prefixes, if present, are accepted and stripped by the loader).

    Parameters
    ----------
    latent_embeddings (jnp.ndarray)
        TorusBasis embeddings of latent coords, shape ``(N, 2 * latent_dim)``.
    params (dict[str, jnp.ndarray])
        Decoder weights keyed by ``"<layer_idx>.weight"`` / ``"<layer_idx>.bias"``.

    Returns
    -------
    reconstructions (jnp.ndarray)
        Spectrogram reconstructions, shape ``(N, 1, 128, 128)``, in ``[0, 1]``.
    """
    h = _linear(latent_embeddings, params["0.weight"], params["0.bias"])
    h = _linear(h, params["1.weight"], params["1.bias"])
    h = h.reshape(h.shape[0], *_DECODER_RESHAPE)
    for n_block, idx in enumerate(_DECODER_CONV_INDICES):
        h = conv_transpose2d(
            h, params[f"{idx}.weight"], params[f"{idx}.bias"],
            _CONV_STRIDE, _CONV_PADDING, _CONV_OUTPUT_PADDING,
        )
        # ReLU after every conv except the last, which is followed by Sigmoid.
        h = jax.nn.relu(h) if n_block < len(_DECODER_CONV_INDICES) - 1 else jax.nn.sigmoid(h)
    return h


# --------------------------------------------------------------------------- #
# Posterior + embedding (port of QMCLVM.posterior_probability / embed_data)
# --------------------------------------------------------------------------- #
def decode_lattice_atlas(lattice: jnp.ndarray, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Description
    -----------
    Decodes every lattice point once into a reconstruction "atlas" — the fixed
    map of the torus that new data is scored against. Computed once and reused
    across all data batches.

    Parameters
    ----------
    lattice (jnp.ndarray)
        Lattice points, shape ``(K, latent_dim)`` (reduced mod 1 internally).
    params (dict[str, jnp.ndarray])
        Decoder weights.

    Returns
    -------
    atlas (jnp.ndarray)
        Decoded reconstructions, shape ``(K, 1, 128, 128)``.
    """
    basis = torus_basis_forward(lattice % 1)
    return decoder_forward(basis, params)


def posterior_over_lattice(atlas: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """
    Description
    -----------
    Posterior ``P(lattice point | spectrogram)`` for each data spectrogram, via
    the Bernoulli likelihood against the decoded atlas, the log-mean-exp evidence
    and a softmax over lattice points — matching ``QMCLVM.posterior_probability``.

    Parameters
    ----------
    atlas (jnp.ndarray)
        Decoded lattice reconstructions, shape ``(K, 1, 128, 128)``.
    data (jnp.ndarray)
        Data spectrograms, shape ``(B, 1, 128, 128)``.

    Returns
    -------
    posterior (jnp.ndarray)
        A ``(B, K)`` posterior over lattice points.
    """
    lls = binary_lp(atlas, data)                                     # (B, K)
    # `evidence` is a per-row constant (log-mean-exp of the row); subtracting a
    # per-row constant from the softmax logits leaves the softmax unchanged
    # (softmax is shift-invariant). It is retained here only for exact
    # line-by-line parity with QMCLVM.posterior_probability.
    evidence = jax.scipy.special.logsumexp(lls, axis=1, keepdims=True) - jnp.log(atlas.shape[0])
    return jax.nn.softmax(lls - evidence, axis=1)


def embed_data(
    lattice: jnp.ndarray,
    data: jnp.ndarray,
    params: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Description
    -----------
    Embeds data spectrograms into the trained torus: posterior over the lattice,
    then the posterior-weighted average of the lattice's torus embeddings mapped
    back to latent coordinates (the ``embed_type='posterior'`` path of
    ``QMCLVM.embed_data``).

    Parameters
    ----------
    lattice (jnp.ndarray)
        Lattice points, shape ``(K, latent_dim)``.
    data (jnp.ndarray)
        Data spectrograms, shape ``(B, 1, 128, 128)`` in ``[0, 1]``.
    params (dict[str, jnp.ndarray])
        Decoder weights.

    Returns
    -------
    latent_coords (jnp.ndarray)
        Torus coordinates in ``[0, 1)``, shape ``(B, latent_dim)``.
    """
    atlas = decode_lattice_atlas(lattice, params)
    posterior = posterior_over_lattice(atlas, data)                 # (B, K)
    weighted = posterior @ torus_basis_forward(lattice)             # (B, 2*latent_dim)
    return torus_basis_reverse(weighted)
