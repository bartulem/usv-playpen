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
    images of lattice points spread across every block embed to the one-block answer
    for block sizes that split both axes unevenly."""
    rng = np.random.default_rng(5)
    params = _sharp_decoder_params(rng)
    lattice = qm.gen_korobov_basis(a=5, num_dims=2, num_points=23)
    atlas = qm.decode_lattice_atlas(lattice, params)
    data = atlas[np.array([0, 4, 9, 13, 17, 20, 22])]

    posterior = qm.posterior_over_lattice(atlas, data)
    reference = np.asarray(qm.torus_basis_reverse(posterior @ qm.torus_basis_forward(lattice)))
    for lattice_batch_size, data_batch_size in ((23, 7), (5, 2), (1, 3), (100, 100)):
        coords = np.asarray(qm.embed_data(lattice, data, params, lattice_batch_size, data_batch_size))
        torus_gap = np.abs(coords - reference)
        assert np.all(np.minimum(torus_gap, 1.0 - torus_gap) < 1e-5)


def test_embed_data_conditional_decodes_each_value_with_c_appended():
    """A conditional decoder sees one c for the whole lattice: grouping the
    spectrograms by value must give, for every row, the embedding its own value's
    lattice gives, and the decoder width must match the conditioning."""
    rng = np.random.default_rng(7)
    params = _sharp_decoder_params(rng)
    params["0.weight"] = jnp.asarray(rng.standard_normal((2048, 5)))           # 2 * latent_dim + c_dim
    lattice = qm.gen_korobov_basis(a=5, num_dims=2, num_points=23)
    wrapped = qm.torus_basis_forward(lattice % 1)
    values = np.array([0.2, 0.7, 0.2, 0.7, 0.2], dtype=np.float32)
    low, high = (float(value) for value in np.unique(values))
    atlases = {
        value: qm.decoder_forward(jnp.concatenate([wrapped, jnp.full((23, 1), value)], axis=1), params)
        for value in (low, high)
    }
    data = jnp.stack([atlases[low][3], atlases[high][9], atlases[low][15], atlases[high][1], atlases[low][20]])
    coords = np.asarray(qm.embed_data(lattice, data, params, 7, 2, condition_values=values))
    for row, value in enumerate(values):
        posterior = qm.posterior_over_lattice(atlases[float(value)], data[row:row + 1])
        expected = np.asarray(qm.torus_basis_reverse(posterior @ qm.torus_basis_forward(lattice)))[0]
        gap = np.abs(coords[row] - expected)
        assert np.all(np.minimum(gap, 1.0 - gap) < 1e-5)
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
