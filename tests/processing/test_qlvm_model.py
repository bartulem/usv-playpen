"""
@author: bartulem
Tests for processing/qlvm_model — the JAX (torch-free) QLVM inference port.

The load-bearing test is ``conv_transpose2d`` parity: the JAX transposed
convolution is checked against a pure-numpy implementation of
``torch.nn.ConvTranspose2d``'s exact definition (dilate input -> pad -> flipped
cross-correlation), so we match torch without importing it. The rest cover the
lattice generators, the TorusBasis round-trip, the Bernoulli likelihood, the
posterior known-answer, and an end-to-end embed shape/range check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from usv_playpen.processing import qlvm_model as qm


def _ct2d_numpy(x, weight, bias, stride, padding, output_padding):
    """Reference torch.nn.ConvTranspose2d (groups=1, dilation=1) in pure numpy:
    dilate the input by ``stride``, pad by ``(k-1-padding)`` plus
    ``output_padding`` on the trailing edge, then cross-correlate with the
    in/out-swapped, spatially-flipped kernel."""
    n, c_in, h, w = x.shape
    _, c_out, kh, kw = weight.shape
    hd, wd = (h - 1) * stride + 1, (w - 1) * stride + 1
    xd = np.zeros((n, c_in, hd, wd))
    xd[:, :, ::stride, ::stride] = x
    pt, pb = kh - 1 - padding, kh - 1 - padding + output_padding
    pl, pr = kw - 1 - padding, kw - 1 - padding + output_padding
    xp = np.pad(xd, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    keff = np.flip(weight.transpose(1, 0, 2, 3), axis=(2, 3))  # (c_out, c_in, kh, kw)
    h_out, w_out = xp.shape[2] - kh + 1, xp.shape[3] - kw + 1
    out = np.zeros((n, c_out, h_out, w_out))
    for ni in range(n):
        for co in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    out[ni, co, i, j] = np.sum(xp[ni, :, i:i + kh, j:j + kw] * keff[co]) + bias[co]
    return out


def test_conv_transpose2d_matches_torch_reference():
    """JAX conv_transpose2d == the numpy reference of torch's ConvTranspose2d,
    for the same (stride=2, padding=1, output_padding=1) config the QLVM uses."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 5, 5))
    weight = rng.standard_normal((3, 4, 3, 3))  # (C_in, C_out, kH, kW)
    bias = rng.standard_normal(4)
    ref = _ct2d_numpy(x, weight, bias, stride=2, padding=1, output_padding=1)
    got = np.asarray(qm.conv_transpose2d(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias), 2, 1, 1))
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-5)


def test_gen_korobov_basis_values():
    """Korobov lattice row i is i * z / n with z_k = a**k mod n."""
    lat = np.asarray(qm.gen_korobov_basis(a=3, num_dims=2, num_points=5))
    assert lat.shape == (5, 2)
    # z = [3**0 % 5, 3**1 % 5] = [1, 3]; row 2 = 2*[1,3]/5 = [0.4, 1.2]
    assert np.allclose(lat[2], [0.4, 1.2])


def test_fib_and_roberts_shapes():
    """Fibonacci and Roberts lattices have the documented sizes and live in a
    sensible range before the mod-1 reduction."""
    fib = np.asarray(qm.gen_fib_basis(8))   # fib(8) = 21 points
    assert fib.shape == (21, 2)
    rob = np.asarray(qm.roberts_sequence(50, 2))
    assert rob.shape == (50, 2)
    assert rob[0].tolist() == [0.0, 0.0]


def test_torus_basis_roundtrip():
    """reverse(forward(z)) recovers z in [0, 1)."""
    z = jnp.asarray(np.array([[0.1, 0.25], [0.9, 0.5], [0.0, 0.75]]))
    back = np.asarray(qm.torus_basis_reverse(qm.torus_basis_forward(z)))
    assert np.allclose(back, np.asarray(z), atol=1e-6)


def test_torus_basis_reverse_never_returns_one():
    """A tiny negative angle rounds to 2*pi in float32; without the mod-1 wrap the
    coordinate is exactly 1.0 and the label lookup clips it to the far grid edge."""
    embedding = jnp.asarray(np.array([[1.0, 1.0, -1e-9, 0.5]], dtype=np.float32))
    coords = np.asarray(qm.torus_basis_reverse(embedding))
    assert coords.dtype == np.float32
    assert np.all(coords >= 0.0)
    assert np.all(coords < 1.0)
    assert coords[0, 0] == 0.0


def test_binary_lp_shape_and_peaks_at_self():
    """binary_lp returns (B, K) and is maximized when a sample equals the data."""
    rng = np.random.default_rng(1)
    atlas = jnp.asarray(rng.uniform(0.1, 0.9, size=(4, 1, 6, 6)))
    data = atlas[2][None]  # one data point equal to atlas entry 2
    lls = np.asarray(qm.binary_lp(atlas, data))
    assert lls.shape == (1, 4)
    assert int(lls.argmax(axis=1)[0]) == 2


def test_posterior_peaks_and_embed_recovers_lattice_point():
    """When data equals a lattice point's reconstruction, the posterior peaks at
    that point and the embedded coordinate matches that lattice point."""
    lattice = qm.gen_korobov_basis(a=3, num_dims=2, num_points=12)
    rng = np.random.default_rng(2)
    atlas = jnp.asarray(rng.uniform(0.1, 0.9, size=(12, 1, 8, 8)))
    data = atlas[5][None]
    posterior = np.asarray(qm.posterior_over_lattice(atlas, data))
    assert int(posterior.argmax(axis=1)[0]) == 5
    # Embed tail: posterior-weighted torus average -> lattice point 5 (mod 1).
    weighted = jnp.asarray(posterior) @ qm.torus_basis_forward(lattice)
    coord = np.asarray(qm.torus_basis_reverse(weighted))[0]
    assert np.allclose(coord, np.asarray(lattice[5] % 1), atol=1e-3)


def _random_decoder_params(rng, latent_dim=2):
    """Random decoder weights matching the QLVM mouse architecture shapes."""
    return {
        "0.weight": jnp.asarray(rng.standard_normal((2048, 2 * latent_dim)) * 0.05),
        "0.bias": jnp.asarray(rng.standard_normal(2048) * 0.05),
        "1.weight": jnp.asarray(rng.standard_normal((64 * 8 * 8, 2048)) * 0.01),
        "1.bias": jnp.asarray(rng.standard_normal(64 * 8 * 8) * 0.05),
        "3.weight": jnp.asarray(rng.standard_normal((64, 32, 3, 3)) * 0.05),
        "3.bias": jnp.asarray(rng.standard_normal(32) * 0.05),
        "5.weight": jnp.asarray(rng.standard_normal((32, 16, 3, 3)) * 0.05),
        "5.bias": jnp.asarray(rng.standard_normal(16) * 0.05),
        "7.weight": jnp.asarray(rng.standard_normal((16, 8, 3, 3)) * 0.05),
        "7.bias": jnp.asarray(rng.standard_normal(8) * 0.05),
        "9.weight": jnp.asarray(rng.standard_normal((8, 1, 3, 3)) * 0.05),
        "9.bias": jnp.asarray(rng.standard_normal(1) * 0.05),
    }


def test_decoder_forward_and_embed_end_to_end():
    """The decoder maps (N, 2*latent_dim) -> (N, 1, 128, 128) in [0, 1], and
    embed_data returns torus coordinates in [0, 1) with the right shape."""
    rng = np.random.default_rng(3)
    params = _random_decoder_params(rng)
    lattice = qm.gen_korobov_basis(a=3, num_dims=2, num_points=16)

    recon = qm.decode_lattice_atlas(lattice, params)
    assert recon.shape == (16, 1, 128, 128)
    recon_np = np.asarray(recon)
    assert recon_np.min() >= 0.0
    assert recon_np.max() <= 1.0

    data = jnp.asarray(rng.uniform(0.0, 1.0, size=(3, 1, 128, 128)))
    coords = np.asarray(qm.embed_data(lattice, data, params, lattice_batch_size=16, data_batch_size=3))
    assert coords.shape == (3, 2)
    assert coords.min() >= 0.0
    assert coords.max() < 1.0


def _sharp_decoder_params(rng, latent_dim=2):
    """Random decoder weights large enough that lattice points decode to clearly
    different images, so a decoded image has a peaked posterior. With the small
    weights of ``_random_decoder_params`` every image is ~0.5 and the posterior is
    flat, which makes the posterior-mean angle ill-conditioned."""
    layers = {
        "0": ((2048, 2 * latent_dim), 1.0),
        "1": ((64 * 8 * 8, 2048), 1.0 / np.sqrt(2048)),
        "3": ((64, 32, 3, 3), 0.5),
        "5": ((32, 16, 3, 3), 0.5),
        "7": ((16, 8, 3, 3), 0.5),
        "9": ((8, 1, 3, 3), 0.5),
    }
    params = {}
    for idx, (shape, scale) in layers.items():
        params[f"{idx}.weight"] = jnp.asarray(rng.standard_normal(shape) * scale)
        params[f"{idx}.bias"] = jnp.asarray(np.zeros(shape[1] if len(shape) == 4 else shape[0]))
    return params


def test_embed_data_chunking_matches_one_block():
    """Chunking the lattice and the data must not reorder columns or rows: decoded
    images of lattice points spread across every block embed onto their own lattice
    points for block sizes that split both axes unevenly. Their posteriors are one-hot,
    so the answer is exact up to float32 rounding on any device; a reference computed
    with another JAX product would share its rounding (TensorFloat-32 on a CUDA GPU
    put both 4.5e-5 off and they still agreed)."""
    rng = np.random.default_rng(5)
    params = _sharp_decoder_params(rng)
    lattice = qm.gen_korobov_basis(a=5, num_dims=2, num_points=23)
    points = np.array([0, 4, 9, 13, 17, 20, 22])
    data = qm.decode_lattice_atlas(lattice, params)[points]

    reference = np.asarray(lattice[points] % 1)
    for lattice_batch_size, data_batch_size in ((23, 7), (5, 2), (1, 3), (100, 100)):
        coords = np.asarray(qm.embed_data(lattice, data, params, lattice_batch_size, data_batch_size))
        torus_gap = np.abs(coords - reference)
        assert np.all(np.minimum(torus_gap, 1.0 - torus_gap) < 1e-6)


@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="TensorFloat-32 matrix products only happen on a CUDA GPU",
)
def test_decoder_and_likelihood_products_keep_float32_precision_on_gpu():
    """On a CUDA GPU JAX multiplies matrices in TensorFloat-32 unless asked not to, which
    left a decoder Linear layer 3.6e-4 off float64; the decoder and the likelihood must
    stay within float32 rounding of the float64 answer."""
    rng = np.random.default_rng(8)
    x, weight, bias = rng.standard_normal((64, 4)), rng.standard_normal((2048, 4)), rng.standard_normal(2048)
    linear = np.asarray(qm._linear(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias)))
    expected_linear = x @ weight.T + bias
    assert np.max(np.abs(linear - expected_linear)) / np.max(np.abs(expected_linear)) < 1e-6
    samples, data = rng.uniform(0.05, 0.95, size=(8, 1, 32, 32)), rng.uniform(0.0, 1.0, size=(6, 1, 32, 32))
    lls = np.asarray(qm.binary_lp(jnp.asarray(samples), jnp.asarray(data)))
    expected_lls = (np.einsum("bjdl,sjdl->bs", data, np.log(samples))
                    + np.einsum("bjdl,sjdl->bs", 1 - data, np.log(1 - samples)))
    assert np.max(np.abs(lls - expected_lls)) / np.max(np.abs(expected_lls)) < 1e-6


def _lattice_shift_decoder_params(rng, lattice, shift, low, high):
    """
    Description
    -----------
    A ReLU-head decoder conditioned on one value c whose two values decode the lattice
    ``shift`` points apart: the image it draws for lattice point ``i`` at ``c = low`` is
    the image it draws for point ``i + shift`` at ``c = high``. Both the right answer and
    the wrong one are then known in advance, which is what lets a test see whether each
    spectrogram was scored against the lattice decoded at its OWN value.

    It is built, not found, from three facts:

    1. A rank-1 lattice is closed under addition, so point ``i + shift`` sits at point
       ``i``'s coordinates plus point ``shift``'s: every point moves by the SAME angle
       (hence the angle here comes from ``lattice[shift]``).
    2. Adding a fixed angle is linear in the ``[cos, sin]`` embedding (the angle-sum
       identities), so it is a rotation ``R`` with ``R e_i = e_{i+shift}``. The first
       layer is a matmul, so ``(reads @ R) e_i == reads e_{i+shift}``: a half whose
       weights carry ``R`` reads the point ``shift`` steps along.
    3. c enters the first Linear layer additively, so it can never rotate anything, but
       an offset in front of the ReLU can switch a unit off.

    So the first layer's ``2 * half`` units split into one half reading ``reads`` and one
    reading ``reads @ R``, and the c column adds ``+gate * (c - middle)`` to the first and
    ``-gate * (c - middle)`` to the second. ``open_gate`` exceeds
    ``|reads_u . e| <= ||reads_u|| * sqrt(2)`` for every unit and lattice point (``R`` is
    orthogonal, so both halves share the bound), so exactly one half survives the ReLU at
    either value: the first at ``high``, the second at ``low``. The second Linear layer
    applies the same ``V`` to both halves and its bias subtracts the survivor's constant
    ``open_gate``, leaving ``V (reads e_i)`` at ``high`` and ``V (reads e_{i+shift})`` at
    ``low``, so the two images are equal by construction (in float32: ~3e-4 per pixel,
    against ~0.24 between different lattice points, with a ~43000-nat best-match margin).
    Dropping that bias subtraction keeps the equality but drives the sigmoid to 0/1 nearly
    everywhere, flattening the posteriors this relies on. The conv stack is
    ``_sharp_decoder_params``'s, so every image's posterior sits on one lattice point.

    Parameters
    ----------
    rng (np.random.Generator)
        Seeded generator for the layer weights.
    lattice (jnp.ndarray)
        The lattice being embedded into, shape ``(K, latent_dim)``.
    shift (int)
        How many lattice points apart the two conditioning values decode.
    low (np.float32)
        The conditioning value whose lattice is the shifted one.
    high (np.float32)
        The conditioning value whose lattice is read as-is.

    Returns
    -------
    params (dict[str, jnp.ndarray])
        ReLU-head decoder weights (Linear layers 0 and 2, convs 4/6/8/10).
    """
    half = 1024
    angle = 2 * np.pi * np.asarray(lattice[shift])
    cos, sin = np.diag(np.cos(angle)), np.diag(np.sin(angle))
    rotation = np.block([[cos, -sin], [sin, cos]])                  # acts on [cos..., sin...]
    reads = rng.standard_normal((half, 4))
    # |reads @ embedding| <= |reads| * sqrt(2) for any torus embedding, so this margin
    # keeps the open half positive and the closed half negative at every lattice point.
    open_gate = np.sqrt(2) * np.linalg.norm(reads, axis=1).max() + 1.0
    gate = open_gate / ((high - low) / 2)
    middle = (low + high) / 2
    second = rng.standard_normal((64 * 8 * 8, half)) / np.sqrt(half)
    params = {
        "0.weight": np.block([[reads, np.full((half, 1), gate)], [reads @ rotation, np.full((half, 1), -gate)]]),
        "0.bias": np.concatenate([np.full(half, -gate * middle), np.full(half, gate * middle)]),
        "2.weight": np.concatenate([second, second], axis=1),
        "2.bias": -open_gate * second.sum(axis=1),
    }
    convs = _sharp_decoder_params(rng)
    for idx in (3, 5, 7, 9):                                        # the ReLU head shifts conv indices by one
        params[f"{idx + 1}.weight"], params[f"{idx + 1}.bias"] = convs[f"{idx}.weight"], convs[f"{idx}.bias"]
    return {key: jnp.asarray(value) for key, value in params.items()}


def test_embed_data_conditional_decodes_each_value_with_c_appended():
    """A conditional decoder sees one c for the whole lattice, so every spectrogram must
    be scored against the lattice decoded at its own value: with the two values decoding
    the lattice ``shift`` points apart, the right grouping recovers each image's lattice
    point and a swapped one lands every row ``shift`` points away. The decoder width
    must match the conditioning."""
    rng = np.random.default_rng(7)
    lattice = qm.gen_korobov_basis(a=5, num_dims=2, num_points=23)
    low, high, shift = np.float32(0.2), np.float32(0.7), 4
    params = _lattice_shift_decoder_params(rng, lattice, shift, low, high)
    values = np.array([high, low, high, low, high], dtype=np.float32)
    points = np.array([3, 9, 15, 1, 20])
    decoder_input = jnp.concatenate([qm.torus_basis_forward(lattice[points] % 1), jnp.asarray(values)[:, None]], axis=1)
    data = qm.decoder_forward(decoder_input, params)
    swapped_points = np.where(values == low, points + shift, points - shift) % 23
    for condition_values, expected_points in ((values, points), (np.where(values == low, high, low), swapped_points)):
        coords = np.asarray(qm.embed_data(lattice, data, params, 7, 2, condition_values=condition_values))
        gap = np.abs(coords - np.asarray(lattice[expected_points] % 1))
        # One lattice point is 1/23 away, so a loose tolerance still tells the right
        # grouping from a wrong one; the precision checks are separate tests.
        assert np.all(np.minimum(gap, 1.0 - gap) < 1e-3)
    with pytest.raises(ValueError, match="conditioning input"):
        qm.embed_data(lattice, data, params, 7, 2)
    with pytest.raises(ValueError, match="conditioning input"):
        qm.embed_data(lattice, data, _sharp_decoder_params(rng), 7, 2, condition_values=values)


def test_embed_data_rejects_empty_blocks():
    """A zero block size would loop forever or embed nothing, so it must fail loudly."""
    rng = np.random.default_rng(6)
    params = _random_decoder_params(rng)
    lattice = qm.gen_korobov_basis(a=3, num_dims=2, num_points=4)
    data = jnp.asarray(rng.uniform(0.0, 1.0, size=(1, 1, 128, 128)))
    with pytest.raises(ValueError, match="must be >= 1"):
        qm.embed_data(lattice, data, params, lattice_batch_size=0, data_batch_size=1)
