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
    coords = np.asarray(qm.embed_data(lattice, data, params))
    assert coords.shape == (3, 2)
    assert coords.min() >= 0.0
    assert coords.max() < 1.0 + 1e-6
