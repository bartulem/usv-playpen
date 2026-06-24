"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.monopolar_triangulation``.

The triangulation is verified by the round-trip property: place a
synthetic monopolar source at a known 3D location, generate the exact
``alpha / distance`` feature vector it would produce over a cloud of
contacts, and confirm both optimisers recover the known location. The
helper functions are checked directly against hand-computed values.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize

from usv_playpen.neuropixels.monopolar_triangulation import (
    _data_at_3d,
    make_initial_guess_and_bounds_3d,
    estimate_distance_error_with_log_3d,
    solve_monopolar_triangulation_3d,
)


def _contact_cloud():
    """
    Description
    -----------
    Build a small 3D cloud of contact coordinates loosely resembling a
    sparse Neuropixels channel set: two lateral columns, eight rows
    along the shank, two depth offsets — 32 contacts spanning a compact
    volume.

    Returns
    -------
    numpy.ndarray
        ``(32, 3)`` array of contact coordinates.
    """
    xs = np.array([0.0, 32.0])
    ys = np.linspace(0.0, 210.0, 8)
    zs = np.array([0.0, 15.0])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _synthetic_waveform(contacts, source, alpha):
    """
    Description
    -----------
    Generate the exact monopolar feature vector a source at ``source``
    with magnitude ``alpha`` would produce over ``contacts``, using the
    same ``alpha / distance`` model (and the same near-zero distance
    clamp) as the module under test.

    Parameters
    ----------
    contacts : numpy.ndarray
        ``(n_contacts, 3)`` contact coordinates.
    source : numpy.ndarray
        Length-3 source location.
    alpha : float
        Source magnitude.

    Returns
    -------
    numpy.ndarray
        ``(n_contacts,)`` synthetic per-contact feature vector.
    """
    distances = np.linalg.norm(contacts - np.asarray(source), axis=1)
    distances[distances < 1e-6] = 1e-6
    return alpha / distances


def test_data_at_3d_matches_hand_computed_values():
    """
    Description
    -----------
    :func:`_data_at_3d` must return ``alpha / distance`` per contact,
    with sub-epsilon distances clamped to ``1e-6``. A contact 5 units
    from the source with ``alpha = 10`` gives ``2.0``; a contact
    coincident with the source gives ``10 / 1e-6``.
    """
    contacts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    v = _data_at_3d(0.0, 0.0, 0.0, 10.0, contacts)
    assert v[1] == pytest.approx(2.0)
    assert v[0] == pytest.approx(10.0 / 1e-6)


def test_initial_guess_is_weighted_centre_of_mass_and_bounds_expand_by_max_distance():
    """
    Description
    -----------
    With a uniform feature vector the initial ``(x, y, z)`` guess must
    equal the plain centroid of the contact cloud, ``x0`` must be
    length 4, and the per-axis bounds must be the contact-cloud extent
    expanded by ``max_distance_um`` on each side, with the ``alpha``
    lower bound at 0.
    """
    contacts = _contact_cloud()
    wf = np.ones(contacts.shape[0])
    max_distance_um = 500.0
    x0, (low_bounds, high_bounds) = make_initial_guess_and_bounds_3d(wf, contacts, max_distance_um)

    assert x0.shape == (4,)
    np.testing.assert_allclose(x0[:3], contacts.mean(axis=0), atol=1e-3)
    assert len(low_bounds) == 4 and len(high_bounds) == 4
    assert low_bounds[0] == pytest.approx(contacts[:, 0].min() - max_distance_um)
    assert high_bounds[0] == pytest.approx(contacts[:, 0].max() + max_distance_um)
    assert low_bounds[3] == 0


def test_zero_feature_vector_falls_back_to_uniform_weights():
    """
    Description
    -----------
    A degenerate all-zero feature vector must not crash: the initial
    guess falls back to uniform weights (so the guess is the plain
    centroid), and the optimiser returns a length-3 position rather
    than raising.
    """
    contacts = _contact_cloud()
    wf = np.zeros(contacts.shape[0])
    x0, _ = make_initial_guess_and_bounds_3d(wf, contacts, 500.0)
    np.testing.assert_allclose(x0[:3], contacts.mean(axis=0), atol=1e-3)

    result = solve_monopolar_triangulation_3d(wf, contacts, 500.0, 'minimize_with_log_penality')
    assert len(result) == 3
    assert all(np.isfinite(c) for c in result)


@pytest.mark.parametrize("source", [
    [16.0, 100.0, 7.0],
    [16.0, 30.0, 7.0],
    [10.0, 160.0, 5.0],
])
def test_least_square_recovers_known_source(source):
    """
    Description
    -----------
    The ``"least_square"`` optimiser must recover a synthetic source
    placed inside the contact cloud to within 1 µm, given the exact
    (noise-free) monopolar feature vector that source would produce.
    """
    contacts = _contact_cloud()
    source = np.array(source)
    wf = _synthetic_waveform(contacts, source, alpha=1000.0)

    x, y, z, alpha = solve_monopolar_triangulation_3d(wf, contacts, 1000.0, 'least_square')
    np.testing.assert_allclose([x, y, z], source, atol=1.0)
    assert alpha == pytest.approx(1000.0, rel=1e-3)


@pytest.mark.parametrize("source", [
    [16.0, 100.0, 7.0],
    [16.0, 30.0, 7.0],
    [10.0, 160.0, 5.0],
])
def test_minimize_with_log_penality_recovers_known_source(source):
    """
    Description
    -----------
    The ``"minimize_with_log_penality"`` optimiser (the one used by the
    spike-quality-metrics pipeline) must recover a synthetic source
    placed inside the contact cloud to within 2 µm, given the exact
    (noise-free) monopolar feature vector that source would produce.
    """
    contacts = _contact_cloud()
    source = np.array(source)
    wf = _synthetic_waveform(contacts, source, alpha=1000.0)

    x, y, z = solve_monopolar_triangulation_3d(wf, contacts, 1000.0, 'minimize_with_log_penality')
    np.testing.assert_allclose([x, y, z], source, atol=2.0)


def test_log_objective_is_zero_at_the_true_source():
    """
    Description
    -----------
    The scalar objective minimised by ``"minimize_with_log_penality"``
    must be (numerically) zero when evaluated at the true source
    location with a noise-free feature vector — there is a perfect
    monopolar fit there, so the mean squared error vanishes.
    """
    contacts = _contact_cloud()
    source = np.array([16.0, 100.0, 7.0])
    wf = _synthetic_waveform(contacts, source, alpha=1000.0)
    max_data = float(np.max(np.abs(wf)))

    err = estimate_distance_error_with_log_3d(source, wf, contacts, max_data)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_least_square_returns_nan_sentinel_on_optimizer_failure(monkeypatch):
    """
    Description
    -----------
    When ``scipy.optimize.least_squares`` raises, the solver swallows
    the exception (logging it) and returns the four-NaN sentinel so a
    single failed unit doesn't abort a batch.
    """

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced optimiser failure")

    monkeypatch.setattr(scipy.optimize, "least_squares", _boom)
    contacts = _contact_cloud()
    wf = _synthetic_waveform(contacts, np.array([16.0, 100.0, 7.0]), alpha=1000.0)
    result = solve_monopolar_triangulation_3d(wf, contacts, 1000.0, "least_square")
    assert len(result) == 4
    assert all(np.isnan(c) for c in result)


def test_minimize_with_log_penality_returns_nan_sentinel_on_optimizer_failure(monkeypatch):
    """
    Description
    -----------
    When ``scipy.optimize.minimize`` raises, the
    ``minimize_with_log_penality`` branch returns the three-NaN
    sentinel rather than propagating the exception.
    """

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced optimiser failure")

    monkeypatch.setattr(scipy.optimize, "minimize", _boom)
    contacts = _contact_cloud()
    wf = _synthetic_waveform(contacts, np.array([16.0, 100.0, 7.0]), alpha=1000.0)
    result = solve_monopolar_triangulation_3d(
        wf, contacts, 1000.0, "minimize_with_log_penality"
    )
    assert len(result) == 3
    assert all(np.isnan(c) for c in result)
