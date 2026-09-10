"""
@author: bartulem
Unit tests for ``usv_playpen.neural_modeling.neural_vocal_decoding``.

Coverage: the torus geometry, the two bases, the Poisson tuning fit, and the LOSO Bayesian decode. The
central test is a synthetic unit with a KNOWN tuning surface -- it must decode with a positive gain while
an untuned unit of the same firing rate scores about zero, which is the only check that distinguishes a
working decoder from one that rewards firing rate.
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

from usv_playpen.neural_modeling.neural_significance import (
    permutation_null,
    within_session_permutation,
)
from usv_playpen.neural_modeling.neural_vocal_decoding import (
    PERIOD,
    bump_basis,
    bump_basis_gain,
    bump_nodes,
    decode_gain,
    decoding_context,
    flagged_descriptors,
    fourier_basis,
    left_tail_anti_alignment,
    poisson_tuning_fit,
    score_events,
    session_exposure_offsets,
    torus_grid,
    torus_occupancy,
    wrapped_square_distance,
)

SETTINGS_PATH = (pathlib.Path(__file__).resolve().parents[2] / "src" / "usv_playpen"
                 / "_parameter_settings" / "neural_modeling_settings.json")


def _settings(**surface_overrides) -> dict:
    """The SHIPPED vocal_decoding block with only the tiny-run knobs overridden.

    Building from the real file rather than a hand-written literal is what makes these tests notice a
    settings-versus-code drift: a key the decoder starts reading, or stops honouring, shows up here
    instead of only at run time. The grid is coarsened and the bandwidth widened so the synthetic units
    stay fast; nothing that changes the decoder's structure is touched."""
    with SETTINGS_PATH.open() as handle:
        block = copy.deepcopy(json.load(handle)["vocal_decoding"])
    block["tuning_surface"].update({"grid_n": 24, "kde_bandwidth": 0.10, **surface_overrides})
    return block


SETTINGS = _settings()


def _fixed_alpha(alpha):
    """SETTINGS with the amplitude calibration pinned to one value, for tests about the surface."""
    return _settings(amplitude_calibration={"enabled": True, "alpha_grid": [alpha], "alpha_min": 0.0,
                                            "mode": "train_tuned_inner_loso"})


def _tuned_unit(n_sessions=3, per_session=200, depth=1.4, seed=0, tuned=True):
    """Events uniform on the torus; counts Poisson with a rate that either does or does not depend on
    position, at MATCHED mean rate so the comparison isolates tuning from firing rate."""
    generator = np.random.default_rng(seed)
    positions = generator.random((n_sessions * per_session, 2))
    sessions = np.repeat(np.arange(n_sessions), per_session)
    shape = np.cos(2 * np.pi * positions[:, 0]) + np.cos(2 * np.pi * positions[:, 1])
    log_rate = (depth * shape if tuned else np.zeros(positions.shape[0]))
    rate = np.exp(log_rate - log_rate.max() + np.log(3.0))
    return generator.poisson(rate).astype(float), positions, sessions


class TestTorusGeometry:

    def test_distance_wraps(self):
        near_edges = wrapped_square_distance(np.array([[0.98, 0.5]]), np.array([[0.02, 0.5]]))
        assert near_edges[0, 0] == pytest.approx(0.04 ** 2)

    def test_distance_is_zero_to_itself(self):
        points = np.random.default_rng(0).random((5, 2))
        assert np.allclose(np.diag(wrapped_square_distance(points, points)), 0.0)

    def test_grid_cells_tile_the_torus(self):
        grid, area = torus_grid(24)
        assert grid.shape == (576, 2)
        assert area * 576 == pytest.approx(PERIOD ** 2)


class TestBases:

    def test_fourier_is_periodic_across_the_seam(self):
        """A seam in the basis would put an artificial discontinuity in every fitted surface."""
        left = fourier_basis(np.array([[0.0, 0.3]]), 2)
        right = fourier_basis(np.array([[1.0, 0.3]]), 2)
        assert np.allclose(left, right, atol=1e-12)

    def test_fourier_width_matches_k(self):
        assert fourier_basis(np.random.default_rng(0).random((4, 2)), 2).shape == (4, 12)

    def test_bump_nodes_are_gated_by_occupancy(self):
        """Events confined to one corner must not license nodes on the far side of the torus."""
        corner = np.random.default_rng(0).random((300, 2)) * 0.2
        kept = bump_nodes(corner, 10, 0.1, 5.0)
        assert kept.shape[0] < 100
        # gating is on the TORUS, so nodes near 0.95 survive by wrapping round to events near 0.05;
        # what must be excluded is the genuinely far side, around 0.5
        far = kept[(kept[:, 0] > 0.4) & (kept[:, 0] < 0.7) & (kept[:, 1] > 0.4) & (kept[:, 1] < 0.7)]
        assert far.shape[0] == 0

    def test_bump_basis_peaks_at_its_node(self):
        nodes = np.array([[0.5, 0.5]])
        values = bump_basis(np.array([[0.5, 0.5], [0.9, 0.9]]), nodes, 0.1)
        assert values[0, 0] == pytest.approx(1.0)
        assert values[1, 0] < 0.05


class TestPoissonFit:

    def test_it_recovers_a_known_surface(self):
        counts, positions, _sessions = _tuned_unit(n_sessions=1, per_session=4000, depth=1.0)
        basis = fourier_basis(positions, 2)
        beta = poisson_tuning_fit(basis, counts, ridge=1.0, n_steps=50)
        predicted = np.exp(basis @ beta[:-1] + beta[-1])

        assert np.corrcoef(predicted, np.exp(np.cos(2 * np.pi * positions[:, 0])
                                             + np.cos(2 * np.pi * positions[:, 1])))[0, 1] > 0.9

    def test_the_intercept_is_unpenalized(self):
        """A heavy ridge must flatten the surface onto the mean rate, not shrink the rate itself."""
        counts, positions, _sessions = _tuned_unit(n_sessions=1, per_session=1500)
        basis = fourier_basis(positions, 2)
        beta = poisson_tuning_fit(basis, counts, ridge=1e8, n_steps=50)

        assert np.abs(beta[:-1]).max() < 1e-3
        assert np.exp(beta[-1]) == pytest.approx(counts.mean(), rel=0.05)


class TestOccupancyAndMask:

    def test_occupancy_is_kde_weighted_not_a_cell_count(self):
        """Stated because the mask admits cells holding ZERO training calls -- two calls a bandwidth
        away sum above 1.0 -- and that is a property to report, not to present as conservative."""
        grid, _area = torus_grid(24)
        occupancy = torus_occupancy(np.array([[0.5, 0.5], [0.52, 0.5]]), grid, 0.10)
        empty_cell = np.argmin(np.sum((grid - np.array([0.55, 0.5])) ** 2, axis=1))

        assert occupancy[empty_cell] > 1.0

    def test_masked_events_are_excluded_and_counted(self):
        """Events far from any training call are dropped rather than scored against an invented prior."""
        counts, positions, sessions = _tuned_unit()
        loose = decoding_context(positions, sessions, SETTINGS)
        strict = decoding_context(positions, sessions,
                                  _settings(decode_grid_min_occupancy=20.0))


        n_loose = decode_gain(counts, loose, SETTINGS)["n_masked"]
        n_strict = decode_gain(counts, strict, _settings(decode_grid_min_occupancy=20.0))["n_masked"]

        assert n_strict > n_loose            # a stricter mask excludes more events
        assert n_strict < counts.size        # but not all of them


class TestDecodeGain:

    def test_a_tuned_unit_gains_and_an_untuned_one_does_not(self):
        """The test that matters: matched firing rate, only the position dependence differs."""
        tuned = _tuned_unit(depth=1.4, seed=0, tuned=True)
        flat = _tuned_unit(depth=1.4, seed=0, tuned=False)

        tuned_gain = decode_gain(tuned[0], decoding_context(tuned[1], tuned[2], SETTINGS),
                                 SETTINGS)["gain"]
        flat_gain = decode_gain(flat[0], decoding_context(flat[1], flat[2], SETTINGS),
                                SETTINGS)["gain"]

        assert tuned_gain > 0.05
        assert abs(flat_gain) < 0.02
        assert tuned_gain > flat_gain

    def test_alpha_zero_reduces_the_decoder_to_the_prior(self):
        """With the shape switched off the posterior IS the prior, so the gain must vanish. This is what
        makes `alpha >= 0` a guarantee that a reversed map can be ignored but never inverted."""
        counts, positions, sessions = _tuned_unit()
        folds = decoding_context(positions, sessions, SETTINGS)
        gain = decode_gain(counts, folds, _fixed_alpha(0.0))["gain"]

        assert abs(gain) < 1e-9

    def test_permuting_the_target_destroys_the_gain(self):
        """End-to-end with the real null: the same counts against re-paired positions must score about
        zero, which is what the exact permutation is for."""
        counts, positions, sessions = _tuned_unit(depth=1.4)
        folds = decoding_context(positions, sessions, SETTINGS)
        observed = decode_gain(counts, folds, SETTINGS)["gain"]

        def statistic(permutation):
            # the context is built ONCE from the real positions -- the prior, the mask and the
            # occupancy are properties of the events, not of the pairing, and the null is meant to
            # destroy the count-to-position pairing and nothing else. `decode_gain` applies the
            # permutation to the basis rows and truth cells, leaving counts (and so the per-fold
            # ridge) exactly where they were.
            return decode_gain(counts, folds, SETTINGS, permutation)["gain"]

        null = permutation_null(statistic, sessions, 12, seed=0)

        assert observed > np.nanmax(null)
        # the null centres BELOW zero, not at it: a surface fitted to re-paired counts is noise, and a
        # noisy surface carried to a held-out event is worse than the prior alone. Same phenomenon as
        # the raw-D2 null in claim 1. The statistic is therefore conservative, not centred.
        assert np.nanmean(null) < 0.0
        assert observed - np.nanmean(null) > 0.1

    def test_the_permutation_keeps_events_within_their_session(self):
        _counts, _positions, sessions = _tuned_unit()
        permutation = within_session_permutation(sessions, np.random.default_rng(0))
        assert np.array_equal(sessions[permutation], sessions)


class TestSessionExposureOffsets:

    def test_the_offset_is_the_log_of_the_session_mean(self):
        counts = np.array([0.0, 2.0, 4.0, 10.0, 20.0])
        sessions = np.array([0, 0, 0, 1, 1])
        offsets = session_exposure_offsets(counts, sessions)
        assert offsets[:3] == pytest.approx(np.log(2.0))
        assert offsets[3:] == pytest.approx(np.log(15.0))

    def test_a_silent_session_is_floored_rather_than_infinite(self):
        offsets = session_exposure_offsets(np.zeros(4), np.array([0, 0, 1, 1]))
        assert np.isfinite(offsets).all()
        assert offsets[0] == pytest.approx(np.log(1e-3))

    def test_the_offsets_stop_session_rate_drift_reading_as_position_tuning(self):
        """The reason the offsets exist. Positions cluster by session and the rate drifts 20x across
        them, with NO position dependence at all. Fitted with one pooled intercept, the surface has to
        explain that drift with the only thing it has -- position -- and invents ~3 log units of
        modulation. Per-session exposure leaves position explaining within-session variation only."""
        generator = np.random.default_rng(7)
        centres = np.array([[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]])
        positions = np.mod(np.repeat(centres, 400, axis=0)
                           + generator.normal(scale=0.05, size=(1200, 2)), 1.0)
        sessions = np.repeat(np.arange(3), 400)
        counts = generator.poisson(np.array([0.3, 1.0, 6.0])[sessions]).astype(float)

        basis = fourier_basis(positions, 2)
        grid, _area = torus_grid(24)
        grid_basis = fourier_basis(grid, 2)
        ridge = 0.02 * counts.sum()

        def modulation_depth(beta):
            shape = grid_basis @ beta[:-1]
            return float(shape.max() - shape.min())

        pooled = modulation_depth(poisson_tuning_fit(basis, counts, ridge, 50))
        with_offsets = modulation_depth(poisson_tuning_fit(
            basis, counts, ridge, 50, session_exposure_offsets(counts, sessions)))

        assert pooled > 2.0
        assert with_offsets < 1.0
        assert pooled > 4.0 * with_offsets


class TestAmplitudeCalibration:

    def test_alpha_scales_depth_and_leaves_the_prior_weighted_level_alone(self):
        """Alpha must move the surface's modulation DEPTH only. The shape is centred on its
        prior-weighted mean first, so the exposure term keeps setting the level -- without the centring
        alpha would rescale the offset too and fight the term it is meant to complement."""
        counts, positions, sessions = _tuned_unit(depth=1.2)
        folds = decoding_context(positions, sessions, SETTINGS)
        fold = folds[0]
        beta = poisson_tuning_fit(fold["event_basis"][fold["train_rows"]],
                                  counts[fold["train_rows"]], 0.02 * counts.sum(), 50,
                                  session_exposure_offsets(counts[fold["train_rows"]],
                                                           fold["train_session_index"]))

        shape = fold["grid_basis"] @ beta[:-1]
        weights = fold["masked_prior"] * fold["cell_area"]
        centre = float(np.sum(shape * weights) / np.sum(weights))
        level = np.log(counts[fold["test_rows"]].mean())

        for alpha in (0.25, 1.0, 1.5):
            log_rate = alpha * (shape - centre) + level
            weighted_mean = float(np.sum(log_rate * weights) / np.sum(weights))
            assert weighted_mean == pytest.approx(level)

    def test_alpha_zero_flattens_the_surface_completely(self):
        counts, positions, sessions = _tuned_unit()
        folds = decoding_context(positions, sessions, SETTINGS)
        fold = folds[0]
        beta = poisson_tuning_fit(fold["event_basis"][fold["train_rows"]],
                                  counts[fold["train_rows"]], 1.0, 50)
        gains = score_events(beta, fold["test_rows"], counts, fold, fold["truth_cell"], 0.0)
        assert np.nanmax(np.abs(gains)) < 1e-9

    def test_the_tuned_alpha_is_recorded_per_fold(self):
        """Alpha is a per-unit descriptor whose values carry meaning -- passers land at 0.75-1.5 and
        null units rail to the floor -- so it is kept per fold, not averaged away."""
        counts, positions, sessions = _tuned_unit(depth=1.4)
        result = decode_gain(counts, decoding_context(positions, sessions, SETTINGS), SETTINGS)
        assert len(result["alphas"]) == 3
        assert len(result["ridges"]) == 3
        assert all(a in SETTINGS["tuning_surface"]["amplitude_calibration"]["alpha_grid"]
                   for a in result["alphas"])
        assert [f["held_out"] for f in result["per_fold"]] == [0, 1, 2]


class TestConfigurationGuards:
    """A settings key that the code silently ignores is worse than one that is missing: the run looks
    configured and is not. Each of these switches is either honoured or refused."""

    def test_a_non_poisson_likelihood_is_refused(self):
        counts, positions, sessions = _tuned_unit()
        settings = _settings()
        settings["tuning_surface"]["likelihood_family"] = "gaussian"
        with pytest.raises(ValueError, match="Poisson"):
            decode_gain(counts, decoding_context(positions, sessions, SETTINGS), settings)

    def test_a_different_ridge_mode_is_refused(self):
        counts, positions, sessions = _tuned_unit()
        settings = _settings()
        settings["tuning_surface"]["ridge_selection"]["mode"] = "nested_inner_loso"
        with pytest.raises(ValueError, match="ridge"):
            decode_gain(counts, decoding_context(positions, sessions, SETTINGS), settings)

    def test_an_alpha_grid_below_alpha_min_is_refused(self):
        """`alpha >= 0` is what stops the decoder inverting a reversed map instead of declining to use
        it, so a grid that reaches below the floor has to fail loudly rather than quietly widen."""
        counts, positions, sessions = _tuned_unit()
        settings = _settings()
        settings["tuning_surface"]["amplitude_calibration"]["alpha_grid"] = [-0.5, 1.0]
        with pytest.raises(ValueError, match="alpha_min"):
            decode_gain(counts, decoding_context(positions, sessions, SETTINGS), settings)

    def test_a_different_calibration_mode_is_refused(self):
        counts, positions, sessions = _tuned_unit()
        settings = _settings()
        settings["tuning_surface"]["amplitude_calibration"]["mode"] = "held_out_mle"
        with pytest.raises(ValueError, match="inner LOSO"):
            decode_gain(counts, decoding_context(positions, sessions, SETTINGS), settings)

    def test_session_rate_offsets_can_be_switched_off(self):
        """The switch has to change the answer, or it is decoration."""
        counts, positions, sessions = _tuned_unit(depth=1.4)
        folds = decoding_context(positions, sessions, SETTINGS)
        off = _settings()
        off["tuning_surface"]["session_rate_offsets"] = False
        assert decode_gain(counts, folds, SETTINGS)["gain"] != decode_gain(counts, folds, off)["gain"]


class TestSaveRoster:
    """The pooled gain is a summary of n fold values, and what it can hide is one session carrying the
    whole result. The plan requires the folds themselves to be persisted, not just the mean."""

    def test_the_per_fold_detail_is_returned(self):
        counts, positions, sessions = _tuned_unit(depth=1.4)
        result = decode_gain(counts, decoding_context(positions, sessions, SETTINGS), SETTINGS)
        assert len(result["per_fold"]) == 3
        for entry in result["per_fold"]:
            assert set(entry) == {"held_out", "n_events", "alpha", "ridge", "n_masked",
                                  "coefficients", "gain"}
            # 12 Fourier columns at k=2 plus the intercept: the map itself, so a figure or a
            # population decode can be rebuilt without refitting
            assert entry["coefficients"].size == 13

    def test_the_leave_one_fold_out_minimum_tracks_the_weakest_fold(self):
        counts, positions, sessions = _tuned_unit(depth=1.4)
        result = decode_gain(counts, decoding_context(positions, sessions, SETTINGS), SETTINGS)
        assert result["n_folds_positive"] == 3
        assert result["min_leave_one_fold_out"] > 0.0
        # dropping the STRONGEST fold gives the smallest pooled mean, so the minimum sits at or
        # below the full pooled gain -- and staying positive is what says no one session carries it
        assert result["min_leave_one_fold_out"] <= result["gain"] + 1e-12

    def test_masked_events_are_counted_per_fold_and_in_total(self):
        counts, positions, sessions = _tuned_unit()
        strict = _settings(decode_grid_min_occupancy=20.0)
        result = decode_gain(counts, decoding_context(positions, sessions, strict), strict)
        assert sum(f["n_masked"] for f in result["per_fold"]) == result["n_masked"]


class TestFlaggedDescriptors:
    """The three `record_*` flags named real quantities the plan rules should be persisted, and were
    read by nothing -- so a run could declare them true and record none of them."""

    def test_the_overdispersion_index_follows_its_flag(self):
        counts, positions, sessions = _tuned_unit(seed=5)
        folds = decoding_context(positions, sessions, SETTINGS)
        on = decode_gain(counts, folds, SETTINGS, with_detail=True)
        off_settings = _settings()
        off_settings["record_overdispersion_index"] = False
        off = decode_gain(counts, folds, off_settings, with_detail=True)
        assert "overdispersion_index" in on
        assert "overdispersion_index" not in off
        # and it is a real measurement, not a placeholder: Poisson counts sit near 1
        assert 0.2 < on["overdispersion_index"] < 5.0

    def test_the_bump_gain_is_a_genuine_refit_under_the_local_basis(self):
        """Not a rescoring of the Fourier surface -- the basis is chosen when the context is built, so
        a bump gain equal to the Fourier gain would mean the override never took."""
        counts, positions, sessions = _tuned_unit(seed=6)
        fourier = decode_gain(counts, decoding_context(positions, sessions, SETTINGS),
                              SETTINGS)["gain"]
        bump = bump_basis_gain(counts, positions, sessions, SETTINGS)
        assert np.isfinite(bump)
        assert bump != fourier
        # both bases should find a strongly tuned unit
        assert bump > 0
        assert fourier > 0

    def test_the_bump_refit_does_not_mutate_the_caller_settings(self):
        counts, positions, sessions = _tuned_unit(seed=7)
        settings = _settings()
        bump_basis_gain(counts, positions, sessions, settings)
        assert settings["tuning_surface"]["tuning_basis"] == "fourier"

    def test_anti_alignment_reads_the_left_tail_not_the_right(self):
        """A reliably BACKWARDS decoder is extreme in the wrong direction: interesting, never a pass."""
        null = np.linspace(-0.01, 0.01, 1001)
        anti = left_tail_anti_alignment(null, -0.02)
        assert anti["anti_aligned"]
        assert anti["p_left"] < 0.01
        aligned = left_tail_anti_alignment(null, +0.02)
        assert not aligned["anti_aligned"]
        assert aligned["p_left"] > 0.99
        middle = left_tail_anti_alignment(null, 0.0)
        assert not middle["anti_aligned"]

    def test_the_flags_gate_what_is_recorded(self):
        counts, positions, sessions = _tuned_unit(seed=8)
        null = np.linspace(-0.01, 0.01, 51)
        both = flagged_descriptors(counts, positions, sessions, SETTINGS, 0.05, null)
        assert "bump_basis_gain" in both
        assert "p_left" in both

        off = _settings()
        off["tuning_surface"]["record_bump_basis_gain"] = False
        off["discrimination_null"]["record_left_tail_anti_alignment"] = False
        neither = flagged_descriptors(counts, positions, sessions, off, 0.05, null)
        assert neither == {}

    def test_the_shipped_settings_turn_all_three_on(self):
        """If a flag ships false, the cohort run silently produces a thinner artifact than the plan's
        save-everything roster promises."""
        with SETTINGS_PATH.open() as handle:
            block = json.load(handle)["vocal_decoding"]
        assert block["record_overdispersion_index"] is True
        assert block["tuning_surface"]["record_bump_basis_gain"] is True
        assert block["discrimination_null"]["record_left_tail_anti_alignment"] is True
