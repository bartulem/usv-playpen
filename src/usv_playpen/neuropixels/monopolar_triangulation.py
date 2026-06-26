"""
@author: bartulem
3D monopolar-source triangulation for Neuropixels unit localization.

Given a unit's per-channel waveform feature (typically the peak-to-peak
amplitude on each channel within the unit's sparse channel set) and the
3D physical coordinates of those channels, this module estimates the 3D
location of the current source that best explains the observed
amplitudes under a monopolar point-source model.

Monopolar model
---------------
The extracellular potential contributed by a single point current
source of magnitude ``alpha`` at Euclidean distance ``d`` from a
recording contact is modelled as ``V = alpha / d``. Fitting the
observed per-channel feature vector ``wf_data`` to this model over the
contact coordinates yields the source position ``(x, y, z)`` (and,
implicitly, ``alpha``).

This is a 3D generalisation of the 2D monopolar triangulation used in
SpikeInterface's ``compute_unit_locations`` — the third spatial axis is
fit explicitly rather than collapsed, which matters here because the
channel coordinates fed in are anatomical (Allen CCF mlapdv) rather than
flat probe-plane coordinates.

Two optimisers are exposed via :func:`solve_monopolar_triangulation_3d`:

* ``"least_square"`` — fits ``(x, y, z, alpha)`` jointly by non-linear
  least squares on the raw residual ``wf_data - V``.
* ``"minimize_with_log_penality"`` — fits ``(x, y, z)`` only; ``alpha``
  is solved analytically at each step as the closed-form least-squares
  scale between the unit-amplitude model and the (max-normalised)
  observed data. The objective is the mean squared error of the
  normalised model fit.

The ``minimize_with_log_penality`` optimiser is the one used by the
spike-quality-metrics pipeline; the ``least_square`` variant is kept for
comparison and is numerically less stable when the source lies far
outside the contact cloud.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize


def _data_at_3d(x: float, y: float, z: float, alpha: float,
                local_contact_locations: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Evaluate the monopolar point-source model at every recording
    contact: the modelled potential at a contact is ``alpha`` divided by
    the Euclidean distance from the source ``(x, y, z)`` to that
    contact.

    Distances below ``1e-6`` are clamped to ``1e-6`` so the division is
    numerically stable when the candidate source coincides with a
    contact.

    Parameters
    ----------
    x : float
        Source x-coordinate (same units as ``local_contact_locations``).
    y : float
        Source y-coordinate.
    z : float
        Source z-coordinate.
    alpha : float
        Source magnitude (the numerator of the ``alpha / distance``
        model).
    local_contact_locations : numpy.ndarray
        ``(n_contacts, 3)`` array of contact coordinates.

    Returns
    -------
    numpy.ndarray
        ``(n_contacts,)`` array of modelled potentials, one per contact.
    """
    source_pos = np.array([x, y, z])
    # Euclidean distances between the 3D source and the 3D contacts.
    distances = np.linalg.norm(local_contact_locations - source_pos, axis=1)
    # Clamp to a small epsilon so the division stays finite.
    distances[distances < 1e-6] = 1e-6
    return alpha / distances


def make_initial_guess_and_bounds_3d(wf_data: np.ndarray,
                                     local_contact_locations: np.ndarray,
                                     max_distance_um: float) -> tuple:
    """
    Description
    -----------
    Build a robust initial guess and data-driven bounds for the
    optimiser, for all three spatial axes plus the source magnitude.

    The initial ``(x, y, z)`` is the amplitude-weighted centre of mass
    of the contact cloud; the initial ``alpha`` is the distance from
    that centre of mass to the contact with the largest absolute
    amplitude, multiplied by that contact's (signed) amplitude (i.e. the
    ``alpha`` that would reproduce the peak amplitude under the monopolar
    model). When the feature vector has effectively zero total weight,
    uniform weights are used so the centre of mass is still defined.

    Bounds on ``x``, ``y`` and ``z`` are the per-axis extent of the
    contact cloud expanded by ``max_distance_um`` on each side; the
    ``alpha`` bound runs from ``0`` to the peak amplitude times
    ``max_distance_um``.

    Parameters
    ----------
    wf_data : numpy.ndarray
        ``(n_contacts,)`` per-contact waveform feature (e.g. peak-to-peak
        amplitude).
    local_contact_locations : numpy.ndarray
        ``(n_contacts, 3)`` array of contact coordinates.
    max_distance_um : float
        How far beyond the contact cloud (per axis) the source is
        allowed to lie, and the multiplier used to bound ``alpha``.

    Returns
    -------
    numpy.ndarray
        ``x0`` — the length-4 initial guess ``[x, y, z, alpha]``
        (float32).
    tuple[list, list]
        ``(low_bounds, high_bounds)`` — each a length-4 list bounding
        ``[x, y, z, alpha]``.
    """
    weights = np.abs(wf_data)
    if np.sum(weights) < 1e-9:
        weights = np.ones_like(wf_data)

    ind_max = np.argmax(weights)
    max_ptp = wf_data[ind_max]
    max_alpha = np.abs(max_ptp) * max_distance_um

    # Amplitude-weighted 3D centre of mass as the starting point.
    com_3d = np.sum(weights[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(weights)

    x0 = np.zeros(4, dtype="float32")
    x0[:3] = com_3d

    dist_to_max = np.linalg.norm(x0[:3] - local_contact_locations[ind_max, :])
    x0[3] = dist_to_max * max_ptp

    # Bounds are the contact-cloud extent expanded by max_distance_um per axis.
    min_coords = local_contact_locations.min(axis=0)
    max_coords = local_contact_locations.max(axis=0)

    low_bounds = [
        min_coords[0] - max_distance_um,
        min_coords[1] - max_distance_um,
        min_coords[2] - max_distance_um,
        0
    ]
    high_bounds = [
        max_coords[0] + max_distance_um,
        max_coords[1] + max_distance_um,
        max_coords[2] + max_distance_um,
        max_alpha
    ]

    return x0, (low_bounds, high_bounds)


def _estimate_distance_error_3d(vec: np.ndarray, wf_data: np.ndarray,
                                local_contact_locations: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Residual function for the ``"least_square"`` optimiser: the
    element-wise difference between the observed per-contact feature
    vector and the monopolar model evaluated at the candidate
    ``(x, y, z, alpha)``.

    Parameters
    ----------
    vec : numpy.ndarray
        Candidate parameter vector ``[x, y, z, alpha]``.
    wf_data : numpy.ndarray
        ``(n_contacts,)`` observed per-contact waveform feature.
    local_contact_locations : numpy.ndarray
        ``(n_contacts, 3)`` array of contact coordinates.

    Returns
    -------
    numpy.ndarray
        ``(n_contacts,)`` residual vector ``wf_data - model``.
    """
    x, y, z, alpha = vec
    data_estimated = _data_at_3d(x, y, z, alpha, local_contact_locations)
    err = wf_data - data_estimated
    return err


def estimate_distance_error_with_log_3d(vec: np.ndarray, wf_data: np.ndarray,
                                        local_contact_locations: np.ndarray,
                                        max_data: float) -> float:
    """
    Description
    -----------
    Scalar objective for the ``"minimize_with_log_penality"`` optimiser.

    Only ``(x, y, z)`` are free parameters here. At each candidate
    position the source magnitude is solved in closed form: with the
    unit-magnitude model ``q = _data_at_3d(x, y, z, 1.0, ...)``, the
    least-squares scale between ``q`` and the max-normalised observed
    data is ``alpha_norm = sum(q * wf_data / max_data) / sum(q * q)``.
    The objective is the mean squared error between the normalised
    observed data and the normalised model at that scale.

    Despite the name, no log penalty term is applied — the invalid log
    penalty present in the original 2D formulation was removed because
    it is not well-defined in this 3D variant.

    Parameters
    ----------
    vec : numpy.ndarray
        Candidate position ``[x, y, z]``.
    wf_data : numpy.ndarray
        ``(n_contacts,)`` observed per-contact waveform feature.
    local_contact_locations : numpy.ndarray
        ``(n_contacts, 3)`` array of contact coordinates.
    max_data : float
        Normalisation constant — the maximum absolute value of
        ``wf_data`` (or ``1.0`` when that is degenerate).

    Returns
    -------
    float
        Mean squared error of the normalised monopolar fit.
    """
    x, y, z = vec
    q = _data_at_3d(x, y, z, 1.0, local_contact_locations)

    alpha_norm = (q * wf_data / max_data).sum() / (q * q).sum()
    model_data_norm = _data_at_3d(x, y, z, alpha_norm, local_contact_locations)

    err = np.square(wf_data / max_data - model_data_norm).mean()

    return err


def solve_monopolar_triangulation_3d(wf_data: np.ndarray,
                                     local_contact_locations: np.ndarray,
                                     max_distance_um: float,
                                     optimizer: str) -> tuple:
    """
    Description
    -----------
    Estimate the 3D location of a unit's monopolar current source from
    its per-contact waveform feature vector and the 3D coordinates of
    those contacts.

    Two optimisers are available:

    * ``"least_square"`` — jointly fits ``(x, y, z, alpha)`` by
      non-linear least squares on the raw residual; returns the full
      length-4 solution.
    * ``"minimize_with_log_penality"`` — fits ``(x, y, z)`` only, with
      ``alpha`` solved analytically at each step; returns the length-3
      position.

    On optimiser failure — either a hard exception or a result whose
    ``success`` flag is ``False`` (non-convergence / maximum iterations
    reached) — the function returns a tuple of NaNs of the appropriate
    length, so a single unit's failure does not abort a batch. Hard
    exceptions are additionally printed.

    Parameters
    ----------
    wf_data : numpy.ndarray
        ``(n_contacts,)`` observed per-contact waveform feature (e.g.
        peak-to-peak amplitude over the unit's sparse channel set).
    local_contact_locations : numpy.ndarray
        ``(n_contacts, 3)`` array of contact coordinates, in the same
        spatial units throughout.
    max_distance_um : float
        How far beyond the contact cloud the source is allowed to lie
        (passed to :func:`make_initial_guess_and_bounds_3d`).
    optimizer : {'least_square', 'minimize_with_log_penality'}
        Which optimiser to use.

    Returns
    -------
    tuple
        For ``"least_square"``: ``(x, y, z, alpha)``.
        For ``"minimize_with_log_penality"``: ``(x, y, z)``.
        A tuple of NaNs of the matching length on optimiser failure.
    None
        If ``optimizer`` is not one of the two recognised values, neither
        branch executes and the function returns ``None``.
    """
    x0, bounds = make_initial_guess_and_bounds_3d(wf_data, local_contact_locations, max_distance_um)

    if optimizer == "least_square":
        args = (wf_data, local_contact_locations)
        try:
            output = scipy.optimize.least_squares(
                _estimate_distance_error_3d,
                x0=x0,
                bounds=bounds,
                args=args
            )
            # scipy does not raise on non-convergence; honour the documented
            # NaN-on-failure contract by gating on the success flag.
            if not output.success:
                return (np.nan, np.nan, np.nan, np.nan)
            return tuple(output["x"])
        except Exception as e:
            print(f"scipy.optimize.least_squares error: {e}")
            return (np.nan, np.nan, np.nan, np.nan)

    if optimizer == "minimize_with_log_penality":
        x0_3d = x0[:3]
        bounds_3d = [(bounds[0][i], bounds[1][i]) for i in range(3)]

        max_data = np.max(np.abs(wf_data))
        if max_data < 1e-9:
            max_data = 1.0

        args = (wf_data, local_contact_locations, max_data)
        try:
            output = scipy.optimize.minimize(
                estimate_distance_error_with_log_3d,
                x0=x0_3d,
                bounds=bounds_3d,
                args=args
            )
            # scipy does not raise on non-convergence; honour the documented
            # NaN-on-failure contract by gating on the success flag.
            if not output.success:
                return (np.nan, np.nan, np.nan)
            x_opt, y_opt, z_opt = output["x"]
            return (x_opt, y_opt, z_opt)
        except Exception as e:
            print(f"scipy.optimize.minimize error: {e}")
            return (np.nan, np.nan, np.nan)
