"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.jax_neural_network_cnn`` — the pure
JAX ResNet-1D building blocks plus the data-augmentation / sampling
helpers that feed the continuous-manifold CNN.

This module also locks in the **M3** fix: ``apply_kinematic_masking`` and
``get_grid_balanced_indices`` now default their ``rng`` fallback to a
deterministic seed-0 generator, so a standalone call with ``rng=None`` is
reproducible (production always threads a live seeded rng). The
architecture coverage is end-to-end ("maximal"): deterministic parameter
init, a forward pass on both the euclidean (2-output) and torus sin/cos
(4-output) heads, batch-norm normalisation, the warmup-cosine schedule,
and a short gradient-descent smoke that must drive an MSE loss down on a
small synthetic batch — all reproducible from a fixed PRNG key.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The CNN module imports optax, whose import emits a one-time JAX
# DeprecationWarning that `filterwarnings = ["error"]` would promote to a
# collection error (see the regression tests for the same guard).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from usv_playpen.modeling.jax_neural_network_cnn import (
        HashableDict,
        _batch_norm_1d,
        _output_axes_count,
        _use_sin_cos_torus_output,
        apply_kinematic_masking,
        build_lr_schedule,
        cnn_forward,
        get_grid_balanced_indices,
        init_cnn_params_and_state,
    )


def _load_cnn_hp(metric='euclidean', period=1.0):
    """Load the shipped CNN hyperparameter block and promote the manifold
    metric/period in, exactly as ``NeuralContinuousCNNRunner.__init__``
    does. Reading the settings file is side-effect-free (no mutation)."""

    settings_path = (pathlib.Path(__file__).resolve().parents[2]
                     / 'src' / 'usv_playpen' / '_parameter_settings'
                     / 'modeling_settings.json')
    with settings_path.open() as fh:
        settings = json.load(fh)
    # `cnn_forward` is JIT-compiled with `hp` as a static argument, so it
    # must be hashable — the runner wraps it in `HashableDict` for exactly
    # this reason.
    hp = HashableDict(settings['hyperparameters']['deep_learning']['cnn_continuous'])
    hp['manifold_metric'] = metric
    hp['manifold_period'] = float(period)
    return hp


# M3 — apply_kinematic_masking


class TestApplyKinematicMasking:

    def test_seeded_rng_is_reproducible(self):
        """Passing the same seeded generator yields identical masks."""

        x = np.ones((4, 6, 20), dtype=np.float64)
        out_a = apply_kinematic_masking(x, mask_prob=0.5, mask_length=5,
                                        rng=np.random.default_rng(123))
        out_b = apply_kinematic_masking(x, mask_prob=0.5, mask_length=5,
                                        rng=np.random.default_rng(123))
        np.testing.assert_array_equal(out_a, out_b)

    def test_none_fallback_is_deterministic_after_m3(self):
        """M3 fix: with ``rng=None`` the masking now falls back to a
        deterministic seed-0 generator, so two standalone calls match."""

        x = np.ones((4, 6, 20), dtype=np.float64)
        out_a = apply_kinematic_masking(x, mask_prob=0.5, mask_length=5)
        out_b = apply_kinematic_masking(x, mask_prob=0.5, mask_length=5)
        np.testing.assert_array_equal(out_a, out_b)

    def test_masking_zeros_chunks_and_preserves_shape(self):
        """With ``mask_prob=1`` every channel is masked, so the output has
        zeros where the all-ones input was blanked; shape is preserved."""

        x = np.ones((3, 4, 16), dtype=np.float64)
        out = apply_kinematic_masking(x, mask_prob=1.0, mask_length=4,
                                      rng=np.random.default_rng(0))
        assert out.shape == x.shape
        assert (out == 0.0).any()
        # Untouched entries remain exactly 1.0 (no scaling applied).
        assert set(np.unique(out)).issubset({0.0, 1.0})


# M3 — get_grid_balanced_indices


class TestGetGridBalancedIndices:

    def test_seeded_rng_is_reproducible(self):
        """The per-cell draws are reproducible under a seeded rng."""

        rng = np.random.default_rng(7)
        Y = rng.uniform(0.0, 1.0, size=(500, 2))
        idx_a = get_grid_balanced_indices(Y, grid_size=10, base_samples=20, alpha=0.5,
                                          rng=np.random.default_rng(1))
        idx_b = get_grid_balanced_indices(Y, grid_size=10, base_samples=20, alpha=0.5,
                                          rng=np.random.default_rng(1))
        np.testing.assert_array_equal(idx_a, idx_b)

    def test_none_fallback_is_deterministic_after_m3(self):
        """M3 fix: the ``rng=None`` path is now seed-0 deterministic."""

        rng = np.random.default_rng(7)
        Y = rng.uniform(0.0, 1.0, size=(300, 2))
        idx_a = get_grid_balanced_indices(Y, grid_size=8, base_samples=15)
        idx_b = get_grid_balanced_indices(Y, grid_size=8, base_samples=15)
        np.testing.assert_array_equal(idx_a, idx_b)

    def test_indices_within_bounds(self):
        """All returned indices reference valid rows of ``Y``."""

        rng = np.random.default_rng(2)
        Y = rng.uniform(0.0, 1.0, size=(400, 2))
        idx = get_grid_balanced_indices(Y, grid_size=10, base_samples=20,
                                        rng=np.random.default_rng(3))
        assert idx.ndim == 1
        assert idx.min() >= 0
        assert idx.max() < Y.shape[0]

    def test_alpha_zero_flattens_more_than_alpha_one(self):
        """A flat quota (``alpha=0``) over-samples sparse cells relative to
        proportional sampling (``alpha=1``); on a density-skewed cloud the
        flat regime therefore draws strictly more indices than the
        proportional one."""

        rng = np.random.default_rng(4)
        # Dense core + sparse tail to make density balancing matter.
        core = rng.normal(0.5, 0.02, size=(900, 2))
        tail = rng.uniform(0.0, 1.0, size=(100, 2))
        Y = np.vstack([core, tail])
        n_flat = get_grid_balanced_indices(Y, grid_size=12, base_samples=30, alpha=0.0,
                                           rng=np.random.default_rng(5)).size
        n_prop = get_grid_balanced_indices(Y, grid_size=12, base_samples=30, alpha=1.0,
                                           rng=np.random.default_rng(5)).size
        assert n_flat > n_prop


# Output-head gates


class TestOutputHeadGates:

    def test_output_axes_count_is_two(self):
        """The manifold is 2-D, so the axis count is fixed at 2."""

        assert _output_axes_count({}) == 2

    @pytest.mark.parametrize("metric,encoding,expected", [
        ('euclidean', 'sin_cos', False),
        ('euclidean', 'raw', False),
        ('torus', 'raw', False),
        ('torus', 'sin_cos', True),
    ])
    def test_sin_cos_gate_truth_table(self, metric, encoding, expected):
        """The sin/cos head is used iff metric is torus AND encoding is
        ``sin_cos``."""

        hp = {'manifold_metric': metric, 'cnn_torus_output_encoding': encoding}
        assert _use_sin_cos_torus_output(hp) is expected

    def test_invalid_encoding_raises_on_torus(self):
        """An unrecognised encoding on the torus path is rejected."""

        with pytest.raises(ValueError):
            _use_sin_cos_torus_output({'manifold_metric': 'torus',
                                       'cnn_torus_output_encoding': 'polar'})


# build_lr_schedule


class TestBuildLrSchedule:

    def test_warmup_and_decay_shape(self):
        """The schedule starts at 0, peaks near ``peak_lr`` after warmup,
        and decays back toward 0 by the final step."""

        peak = 0.01
        total_epochs, steps_per_epoch = 10, 10
        sched = build_lr_schedule(peak, total_epochs, steps_per_epoch)
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = int(0.10 * total_steps)
        assert float(sched(0)) == pytest.approx(0.0, abs=1e-9)
        assert float(sched(warmup_steps)) == pytest.approx(peak, rel=1e-3)
        assert float(sched(total_steps)) == pytest.approx(0.0, abs=1e-6)

    def test_short_run_gets_at_least_one_warmup_step(self):
        """A short run (where ``0.10 * total_steps`` rounds to 0) still
        gets the ``max(1, ...)`` warmup floor and builds a valid schedule
        starting at 0."""

        sched = build_lr_schedule(0.01, total_epochs=5, steps_per_epoch=1)
        assert float(sched(0)) == pytest.approx(0.0, abs=1e-9)
        assert float(sched(1)) > 0.0


# _batch_norm_1d


class TestBatchNorm1d:

    def test_training_normalises_per_channel(self):
        """In training mode with gamma=1/beta=0 the output is per-channel
        (axes 0 and 2) zero-mean, unit-variance."""

        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(5.0, 3.0, size=(8, 4, 20)))
        gamma = jnp.ones((1, 4, 1))
        beta = jnp.zeros((1, 4, 1))
        mean = jnp.zeros((1, 4, 1))
        var = jnp.ones((1, 4, 1))
        out, _, _ = _batch_norm_1d(x, gamma, beta, mean, var, is_training=True)
        out = np.asarray(out)
        np.testing.assert_allclose(out.mean(axis=(0, 2)), np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(out.var(axis=(0, 2)), np.ones(4), atol=1e-3)

    def test_ema_state_update(self):
        """The moving mean/var follow the documented EMA update."""

        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.normal(2.0, 1.0, size=(8, 3, 10)))
        gamma, beta = jnp.ones((1, 3, 1)), jnp.zeros((1, 3, 1))
        mean, var = jnp.zeros((1, 3, 1)), jnp.ones((1, 3, 1))
        momentum = 0.99
        _, new_mean, _new_var = _batch_norm_1d(x, gamma, beta, mean, var,
                                               is_training=True, momentum=momentum)
        batch_mean = np.asarray(jnp.mean(x, axis=(0, 2), keepdims=True))
        expected_mean = momentum * np.asarray(mean) + (1 - momentum) * batch_mean
        np.testing.assert_allclose(np.asarray(new_mean), expected_mean, atol=1e-6)


# init_cnn_params_and_state + cnn_forward (end-to-end architecture)


class TestCnnArchitecture:

    def _init(self, metric='euclidean', in_channels=4, time_steps=32, seed=0):
        hp = _load_cnn_hp(metric=metric)
        params, state = init_cnn_params_and_state(
            jax.random.PRNGKey(seed), in_channels, time_steps, hp)
        return hp, params, state

    def test_init_is_deterministic_for_same_key(self):
        """Initialising twice from the same key yields bitwise-identical
        weights."""

        _, p1, _ = self._init(seed=42)
        _, p2, _ = self._init(seed=42)
        for key in p1:
            np.testing.assert_array_equal(np.asarray(p1[key]), np.asarray(p2[key]))

    def test_output_head_width_euclidean_vs_torus(self):
        """The final dense layer is 2-wide on euclidean and 4-wide on the
        torus sin/cos head."""

        _, p_euc, _ = self._init(metric='euclidean')
        assert p_euc['dense2_w'].shape[1] == 2
        hp_torus, p_torus, _ = self._init(metric='torus')
        assert _use_sin_cos_torus_output(hp_torus)
        assert p_torus['dense2_w'].shape[1] == 4

    def test_forward_output_shape_and_eval_reproducible(self):
        """A forward pass in eval mode produces ``(batch, 2)`` on euclidean
        and is reproducible (no dropout state) for the same inputs."""

        hp, params, state = self._init(metric='euclidean')
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(8, 4, 32)))
        y_center = jnp.zeros(2)
        y_scale = jnp.ones(2)
        out1, _ = cnn_forward(params, state, x, y_center, y_scale, hp, is_training=False)
        out2, _ = cnn_forward(params, state, x, y_center, y_scale, hp, is_training=False)
        assert np.asarray(out1).shape == (8, 2)
        np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    def test_gradient_descent_reduces_mse(self):
        """A short hand-rolled SGD loop over the real forward pass must
        drive an MSE loss down — the end-to-end 'does the network learn'
        smoke, made reproducible by a fixed init key."""

        hp, params, state = self._init(metric='euclidean', seed=1)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(16, 4, 32)))
        target = jnp.asarray(rng.normal(scale=0.3, size=(16, 2)))
        y_center = jnp.zeros(2)
        y_scale = jnp.ones(2)

        def loss_fn(p):
            pred, _ = cnn_forward(p, state, x, y_center, y_scale, hp,
                                  rng_key=jax.random.PRNGKey(0), is_training=True)
            return jnp.mean((pred - target) ** 2)

        loss_before = float(loss_fn(params))
        grad_fn = jax.jit(jax.grad(loss_fn))
        lr = 1e-2
        for _ in range(25):
            grads = grad_fn(params)
            params = {k: params[k] - lr * grads[k] for k in params}
        loss_after = float(loss_fn(params))
        assert loss_after < loss_before
