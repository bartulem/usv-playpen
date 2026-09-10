"""
@author: bartulem
The WHAT axis: does a unit's prevocal firing predict WHICH call is coming?

The target is the upcoming call's position on the QLVM torus, and the decoder is a Bayesian inversion of
a per-unit Poisson tuning surface: fit ``lambda(x)`` on the training sessions' events, then for a
held-out event invert it against an occupancy prior to get a posterior density over the torus, and read
that density at the call's true position. The statistic is the GAIN -- how much better that posterior is
than the prior alone -- in nats per event, which is a cross-validated estimate of mutual information.

Bayesian inversion rather than a regression from count to position because it composes: the same
per-unit surfaces multiply together for a population decode, which is where this is eventually going.

Four things are locked, each after a measurement that is worth not repeating.

The likelihood is Poisson on RAW counts with per-session exposure offsets. A Gaussian on z-scored counts
is not a count likelihood and lost the bake-off on both currencies, missing one unit entirely.

The tuning basis is Fourier k=2. A local bump basis wins on strong units by ~10% but HALVES the faintest
unit's gain, and passes are decided at the faint margin. The bump gain is still computed as a persisted
descriptor so the question stays answerable.

The ridge is ``0.02 x total training spike count``, fixed. Scale-free, because the ridge competes with a
data term whose Fisher weight is the training spike count, so the same number means the same thing for
every unit and window. Nested per-unit selection costs ~6x the compute for no gain and its inner
criterion chases amplitude shrinkage.

An amplitude calibration is applied and is NOT optional. Exposure offsets fix the mean of ``lambda`` but
leave its modulation DEPTH free, and because the score is a strictly proper log density, a surface
carried into a session at the wrong amplitude is punished hard enough to drive the gain negative while
the map still points the right way. A scalar ``alpha >= 0`` multiplies the surface's shape, tuned on the
TRAINING sessions by inner LOSO. The non-negativity is load-bearing: the decoder may temper its
confidence but can never invert a reversed map.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from ..modeling.manifold_metric import signed_diff
from ..modeling.modeling_torus_geodesics import torus_grid as flat_torus_grid
from .deviance_metrics import finite_mean

PERIOD = 1.0


def wrapped_square_distance(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Squared distance on the flat torus, taking the shorter way round in each dimension.

    The QLVM space is periodic, so a point at 0.98 and one at 0.02 are 0.04 apart, not 0.96. Every
    distance in this module goes through here for that reason.

    The wrapping itself is delegated to ``manifold_metric.signed_diff``, which is the project's one
    wrapping convention -- the geodesic machinery routes through it for the same reason. What this adds is
    the CROSS-SET shape: the existing helpers give pairwise distances within a single point cloud, while
    every use here is events against grid cells or against basis nodes.

    Parameters
    ----------
    points (np.ndarray)
        ``(n_points, 2)`` positions.
    nodes (np.ndarray)
        ``(n_nodes, 2)`` positions.

    Returns
    -------
    distances (np.ndarray)
        ``(n_points, n_nodes)`` squared distances.
    """

    difference = signed_diff(np.asarray(points, dtype=np.float64)[:, None, :],
                             np.asarray(nodes, dtype=np.float64)[None, :, :],
                             metric="torus", period=PERIOD)
    return np.sum(difference ** 2, axis=-1)


def fourier_basis(positions: np.ndarray, k_max: int) -> np.ndarray:
    """
    Description
    -----------
    Global periodic basis on the torus: harmonics in each dimension plus the two diagonal terms.

    Periodic by construction, so a surface fitted with it cannot have a seam at the wrap. At ``k_max=2``
    this is 12 columns -- few enough to defend and to make a permutation null cheap, and broad enough
    that a faint unit's surface is not shredded across many parameters.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n_events, 2)`` torus positions.
    k_max (int)
        Highest harmonic.

    Returns
    -------
    basis (np.ndarray)
        ``(n_events, 4 * k_max + 4)`` design.
    """

    x = 2.0 * np.pi * positions[:, 0] / PERIOD
    y = 2.0 * np.pi * positions[:, 1] / PERIOD
    columns = []
    for k in range(1, k_max + 1):
        columns += [np.sin(k * x), np.cos(k * x), np.sin(k * y), np.cos(k * y)]
    columns += [np.sin(x + y), np.cos(x + y), np.sin(x - y), np.cos(x - y)]
    return np.column_stack(columns)


def bump_nodes(train_positions: np.ndarray, grid_n: int, sigma: float,
               min_occupancy: float) -> np.ndarray:
    """
    Description
    -----------
    Node centres for a local bump basis, kept only where training events actually live.

    The occupancy gate is what stops a local basis inventing tuning in empty regions of the torus: a node
    with no events near it has nothing to constrain its weight, and an ungated basis would extrapolate
    there.

    Parameters
    ----------
    train_positions (np.ndarray)
        ``(n_events, 2)`` training positions.
    grid_n (int)
        Nodes per dimension before gating.
    sigma (float)
        Bump width, also the occupancy kernel width.
    min_occupancy (float)
        Smallest kernel-weighted event count a node must carry.

    Returns
    -------
    nodes (np.ndarray)
        ``(n_kept, 2)`` surviving node centres.
    """

    nodes, _cell_area = torus_grid(grid_n)
    occupancy = np.exp(-wrapped_square_distance(train_positions, nodes) / (2.0 * sigma ** 2)).sum(axis=0)
    return nodes[occupancy >= min_occupancy]


def bump_basis(positions: np.ndarray, nodes: np.ndarray, sigma: float) -> np.ndarray:
    """
    Description
    -----------
    Local wrapped-Gaussian basis: one column per node.

    Retained as the ruled-out alternative to Fourier, because its per-unit gain is persisted as a
    robustness descriptor.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n_events, 2)`` positions.
    nodes (np.ndarray)
        ``(n_nodes, 2)`` centres.
    sigma (float)
        Bump width.

    Returns
    -------
    basis (np.ndarray)
        ``(n_events, n_nodes)`` design.
    """

    return np.exp(-wrapped_square_distance(positions, nodes) / (2.0 * sigma ** 2))


def torus_occupancy(train_positions: np.ndarray, grid_positions: np.ndarray,
                    bandwidth: float) -> np.ndarray:
    """
    Description
    -----------
    Wrapped-Gaussian kernel occupancy of the training events over a grid.

    This is the decoder's prior and also its mask. Note the occupancy is a KDE-WEIGHTED sum, so a cell
    can clear a threshold of 1.0 while containing no training call at all -- two calls one bandwidth away
    sum to 1.21. That is a real property of the mask and is reported rather than presented as a
    conservative count.

    Parameters
    ----------
    train_positions (np.ndarray)
        ``(n_events, 2)`` training positions.
    grid_positions (np.ndarray)
        ``(n_cells, 2)`` grid cell centres.
    bandwidth (float)
        Kernel bandwidth.

    Returns
    -------
    occupancy (np.ndarray)
        ``(n_cells,)`` weighted counts.
    """

    return np.exp(-wrapped_square_distance(grid_positions, train_positions)
                  / (2.0 * bandwidth ** 2)).sum(axis=1)


def torus_grid(grid_n: int) -> tuple[np.ndarray, float]:
    """
    Description
    -----------
    Cell centres of a regular ``grid_n x grid_n`` grid on the torus, and the area of one cell.

    The centres come from the geodesic module's grid so the decode grid and the geodesic node set are the
    same points; what is added here is the cell AREA, which turns a summed posterior into a density, so
    the gain is in nats per event rather than per cell and does not move when the grid is refined.

    Parameters
    ----------
    grid_n (int)
        Cells per dimension.

    Returns
    -------
    grid_positions (np.ndarray)
        ``(grid_n ** 2, 2)`` centres.
    cell_area (float)
        Area of one cell.
    """

    return flat_torus_grid(grid_n, period=PERIOD), (PERIOD / grid_n) ** 2


def poisson_tuning_fit(basis: np.ndarray, counts: np.ndarray, ridge: float, n_steps: int,
                       offset: np.ndarray = None) -> np.ndarray:
    """
    Description
    -----------
    Poisson GLM with a log link, fitted by IRLS, with an unpenalized intercept and a fixed exposure
    offset.

    The intercept carries the unit's overall rate and must stay free; penalizing it would push rate
    differences into the surface's shape, which is the part being tested. The linear predictor is clipped
    before exponentiating so a sparse unit cannot produce an overflow on the way to a near-zero rate.

    The offset enters the linear predictor but not the working response, which is what makes it an
    EXPOSURE rather than a covariate: it shifts each session's expected rate by a known amount and is not
    fitted. Supplying per-session offsets is what stops session rate drift being absorbed into the
    surface as if it were position tuning.

    Parameters
    ----------
    basis (np.ndarray)
        ``(n_events, n_basis)`` design.
    counts (np.ndarray)
        Spike count per event.
    ridge (float)
        Penalty on the basis columns.
    n_steps (int)
        IRLS iteration cap; exits early once coefficients settle.
    offset (np.ndarray)
        Fixed exposure per event, or None for a single pooled intercept.

    Returns
    -------
    beta (np.ndarray)
        ``n_basis + 1`` coefficients, intercept last.
    """

    n_events, n_basis = basis.shape
    exposure = np.zeros(n_events, dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    augmented = np.hstack([basis, np.ones((n_events, 1))])
    beta = np.zeros(n_basis + 1, dtype=np.float64)
    beta[-1] = np.log(max(float(counts.mean()), 1e-3)) - float(exposure.mean())
    penalty = ridge * np.eye(n_basis + 1)
    penalty[-1, -1] = 0.0

    for _ in range(n_steps):
        eta = np.clip(augmented @ beta + exposure, -10.0, 5.0)
        mean = np.exp(eta)
        working = (eta - exposure) + (counts - mean) / np.clip(mean, 1e-6, None)
        updated = np.linalg.solve((augmented.T * mean) @ augmented + penalty,
                                  (augmented.T * mean) @ working)
        step = float(np.max(np.abs(updated - beta)))
        beta = updated
        if step < 1e-9:
            break
    return beta



def session_exposure_offsets(counts: np.ndarray, session_index: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Per-session Poisson exposure offsets: each event gets ``log`` of its own session's mean count.

    This is the count-likelihood analogue of per-session z-scoring, and it is not cosmetic. A unit's
    prevocal rate can more than double across a day's sessions, and a tuning surface fitted on pooled raw
    counts with a single intercept absorbs that drift as if it were position tuning -- session rate leaks
    into ``lambda(x)`` and reads as content. With the offsets in place, position explains only WITHIN-
    session variation.

    The floor at 1e-3 keeps a session in which the unit never fired from sending the offset to negative
    infinity.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    session_index (np.ndarray)
        Session id per event.

    Returns
    -------
    offsets (np.ndarray)
        One offset per event.
    """

    offsets = np.zeros(counts.size, dtype=np.float64)
    for session in np.unique(session_index):
        in_session = session_index == session
        offsets[in_session] = np.log(max(float(counts[in_session].mean()), 1e-3))
    return offsets


def decoding_context(positions: np.ndarray, session_index: np.ndarray, settings: dict) -> list:
    """
    Description
    -----------
    Precompute everything about a unit's LOSO folds that does NOT depend on spike counts.

    The bases, the kernel occupancy, the masked log-prior and the truth-cell indices are functions of the
    positions and the configuration alone, so they are identical for the observed fit and for every
    permutation draw. Building them once is what makes an exact permutation null affordable; rebuilding
    them per draw would repeat the KDE thousands of times.

    The basis and truth cells are stored for ALL events, not only the fold's own, because a permutation
    re-pairs counts with positions and the scoring then has to index them in the permuted order. The
    prior, the mask and the occupancy stay built from the REAL training positions: the null destroys the
    count-to-position pairing and nothing else, so the geometry an event is scored against must not move
    with the draw.

    Each fold also carries the inner leave-one-session-out splits over its own training sessions, which
    is where the amplitude calibration is tuned. They are index sets rather than a nested context, so the
    inner scoring reuses this fold's prior instead of rebuilding a KDE per draw.

    The occupancy mask is a DECODE-TIME GATE, not a display convention: it stops the posterior placing
    mass where no training call ever occurred. Events whose true position falls in a masked cell are
    excluded and counted, never silently scored.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n_events, 2)`` torus positions.
    session_index (np.ndarray)
        Session id per event.
    settings (dict)
        The ``vocal_decoding`` block; ``tuning_surface`` supplies grid, bandwidth, basis and mask knobs.

    Returns
    -------
    folds (list)
        One dict per held-out session.
    """

    surface = settings["tuning_surface"]
    grid_positions, cell_area = torus_grid(surface["grid_n"])
    grid_n = surface["grid_n"]
    sessions = np.unique(session_index)
    if sessions.size < 2:
        # leave-one-session-out needs something to leave in; a one-session unit would build a fold with
        # an empty training set and fail deep inside the solver instead of here
        msg = f"the decoder needs at least two sessions; got {sessions.size}."
        raise ValueError(msg)
    rows_by_session = {int(s): np.flatnonzero(session_index == s) for s in sessions}

    ix = np.clip(np.floor(positions[:, 0] / PERIOD * grid_n), 0, grid_n - 1).astype(int)
    iy = np.clip(np.floor(positions[:, 1] / PERIOD * grid_n), 0, grid_n - 1).astype(int)
    truth_cell = ix * grid_n + iy

    folds = []
    for held_out in sessions:
        train_sessions = [int(s) for s in sessions if s != held_out]
        train_rows = (np.concatenate([rows_by_session[s] for s in train_sessions])
                      if train_sessions else np.empty(0, dtype=np.int64))
        train_positions = positions[train_rows]

        if surface["tuning_basis"] == "fourier":
            event_basis = fourier_basis(positions, surface["fourier_k"])
            grid_basis = fourier_basis(grid_positions, surface["fourier_k"])
        else:
            nodes = bump_nodes(train_positions, surface["bump_grid_n"], surface["bump_sigma"],
                               surface["min_basis_occupancy"])
            event_basis = bump_basis(positions, nodes, surface["bump_sigma"])
            grid_basis = bump_basis(grid_positions, nodes, surface["bump_sigma"])

        occupancy = torus_occupancy(train_positions, grid_positions, surface["kde_bandwidth"])
        prior = occupancy / (occupancy.sum() * cell_area)
        mask = occupancy >= surface["decode_grid_min_occupancy"]
        # A mask that excludes the whole grid leaves nothing to normalise. It cannot arise from the
        # shipped threshold, but a caller sweeping the mask can reach it, and dividing by zero here
        # would emit a NaN prior that then silently poisons every score rather than failing.
        masked_prior = np.where(mask, prior, 0.0)
        total = float(masked_prior.sum())
        masked_prior = masked_prior / (total * cell_area) if total > 0.0 else masked_prior

        inner_splits = []
        for inner_held_out in train_sessions:
            rest = [s for s in train_sessions if s != inner_held_out]
            if rest:
                inner_splits.append((np.concatenate([rows_by_session[s] for s in rest]),
                                     rows_by_session[inner_held_out]))

        folds.append({"held_out": int(held_out), "train_sessions": train_sessions,
                      "train_rows": train_rows, "test_rows": rows_by_session[int(held_out)],
                      "train_session_index": np.asarray(session_index)[train_rows],
                      "event_basis": event_basis, "grid_basis": grid_basis,
                      "log_prior": np.log(np.clip(masked_prior, 1e-300, None)),
                      "masked_prior": masked_prior, "mask": mask,
                      "truth_cell": truth_cell, "cell_area": cell_area,
                      "grid_positions": grid_positions,
                      "inner_splits": inner_splits})
    return folds


def score_events(beta: np.ndarray, rows: np.ndarray, counts: np.ndarray, fold: dict,
                 truth_cell: np.ndarray, alpha: float, with_detail: bool = False):
    """
    Description
    -----------
    Per-event decode gain -- log posterior minus log prior at the event's true position, in nats.

    The surface is decomposed into SHAPE and LEVEL before ``alpha`` is applied, which is the whole point
    of the calibration. The shape is centred on its PRIOR-WEIGHTED mean, so scaling it by ``alpha``
    changes the modulation depth and leaves the average height alone; the level is then supplied by the
    scored session's own mean count. Without the centring, ``alpha`` would rescale the offset as well and
    fight the exposure term it is meant to complement.

    The level being read from the scored session is a rate calibration carrying no position information
    whatever -- it is one number, the session's mean count -- and is the exact analogue of recalibrating
    a held-out intercept on the WHEN axis.

    Events whose true position lies in a masked cell return NaN, so they are excluded from the mean
    rather than scored against a prior of zero.

    Parameters
    ----------
    beta (np.ndarray)
        Fitted surface coefficients, intercept last.
    rows (np.ndarray)
        Event indices to score.
    counts (np.ndarray)
        Spike count per event.
    fold (dict)
        From :func:`decoding_context`.
    truth_cell (np.ndarray)
        Grid cell of every event, in the order counts are paired with.
    alpha (float)
        Amplitude scaling of the surface shape; ``>= 0``.
    with_detail (bool)
        Also return the per-event decode error, posterior entropy and decoded position. Read off the
        posterior already built, so free here -- but this function runs inside every permutation
        draw, so the detail is taken only on the observed pass.

    Returns
    -------
    gains (np.ndarray)
        One value per row; NaN where the true position is masked. With ``with_detail``, a
        ``(gains, detail)`` pair.
    """

    shape = fold["grid_basis"] @ beta[:-1]
    weights = fold["masked_prior"] * fold["cell_area"]
    centre = float(np.sum(shape * weights) / max(float(np.sum(weights)), 1e-12))

    scored_counts = counts[rows]
    level = np.log(max(float(scored_counts.mean()), 1e-3))
    rate = np.exp(np.clip(alpha * (shape - centre) + level, -10.0, 5.0))
    log_rate = np.log(np.clip(rate, 1e-12, None))

    loglik = (scored_counts[:, None] * log_rate[None, :] - rate[None, :]
              - gammaln(scored_counts + 1.0)[:, None])
    posterior = fold["log_prior"][None, :] + loglik
    posterior -= posterior.max(axis=1, keepdims=True)
    density = np.exp(posterior)
    density /= density.sum(axis=1, keepdims=True) * fold["cell_area"]

    cells = truth_cell[rows]
    at_truth = density[np.arange(scored_counts.size), cells]
    prior_at_truth = fold["masked_prior"][cells]
    outside = ~fold["mask"][cells]

    posterior_log = np.log(np.clip(at_truth, 1e-300, None))
    prior_log = np.log(np.clip(prior_at_truth, 1e-300, None))
    gains = posterior_log - prior_log
    gains[outside] = np.nan
    if not with_detail:
        return gains

    # Everything below is read off the posterior that has ALREADY been built, so it is free here and
    # ruinous anywhere else -- this function runs inside every permutation draw, which is why the
    # detail is opt-in and taken only on the observed pass.
    mass = density * fold["cell_area"]
    angles = 2.0 * np.pi * fold["grid_positions"] / PERIOD
    decoded = np.empty((scored_counts.size, 2), dtype=np.float64)
    for dimension in range(2):
        cosine = mass @ np.cos(angles[:, dimension])
        sine = mass @ np.sin(angles[:, dimension])
        decoded[:, dimension] = np.mod(np.arctan2(sine, cosine) / (2.0 * np.pi) * PERIOD, PERIOD)
    truth = fold["grid_positions"][cells]
    error = np.sqrt(np.sum(signed_diff(decoded, truth, metric="torus", period=PERIOD) ** 2, axis=1))
    entropy = -np.sum(mass * np.log(np.clip(mass, 1e-300, None)), axis=1)
    error[outside] = np.nan
    entropy[outside] = np.nan
    return gains, {"decode_error": error, "posterior_entropy": entropy, "decoded": decoded}


def tuned_amplitude(counts: np.ndarray, fold: dict, basis: np.ndarray, truth_cell: np.ndarray,
                    ridge: float, alpha_grid: list, n_steps: int,
                    session_rate_offsets: bool = True) -> float:
    """
    Description
    -----------
    Choose this fold's amplitude scaling on its TRAINING sessions alone, by inner leave-one-session-out.

    The training sessions are rotated among themselves; for every candidate alpha the surface fitted on
    the inner training set is scored on the held-in session, and the alpha with the best mean is kept for
    the outer held-out session. No held-out information enters at any point, and the identical procedure
    runs inside every permutation draw, so the extra parameter buys the null exactly what it buys the
    observed value.

    The inner scoring reuses the OUTER fold's prior and mask rather than rebuilding a KDE per inner
    split. That is deliberate and load-bearing for cost: the geometry is a function of positions and
    configuration only, and rebuilding it per draw would repeat the expensive part of the pipeline
    thousands of times for a quantity that does not change.

    A unit with fewer than three sessions has no inner rotation to tune on, and falls back to the grid
    point nearest 1.0 -- the uncalibrated surface -- rather than tuning on data it does not have.

    The grid floor matters and must not be lowered. Allowed below 0.10, null units select 0.01-0.025 and
    the positive-gain count FALLS, because the inner criterion overfits the held-in fold. The floor
    regularizes the calibration itself.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    fold (dict)
        From :func:`decoding_context`.
    basis (np.ndarray)
        Event basis in the order counts are paired with (permuted for a null draw).
    truth_cell (np.ndarray)
        Grid cell per event, in the same order.
    ridge (float)
        Basis penalty for this fold.
    alpha_grid (list)
        Candidate amplitudes, all ``>= 0``.
    n_steps (int)
        IRLS iteration cap.
    session_rate_offsets (bool)
        Whether the inner fits carry per-session exposure offsets; must match the outer fit.

    Returns
    -------
    alpha (float)
        The chosen amplitude.
    """

    if not fold["inner_splits"]:
        return float(min(alpha_grid, key=lambda a: abs(a - 1.0)))

    session_labels = np.asarray(fold["train_session_index"])
    inner_betas = []
    for inner_train, inner_test in fold["inner_splits"]:
        offsets = (session_exposure_offsets(counts[inner_train],
                                            session_labels[np.isin(fold["train_rows"], inner_train)])
                   if session_rate_offsets else None)
        inner_betas.append((poisson_tuning_fit(basis[inner_train], counts[inner_train], ridge,
                                               n_steps, offsets), inner_test))

    best_alpha, best_gain = float(alpha_grid[0]), -np.inf
    for candidate in alpha_grid:
        scores = [finite_mean(score_events(beta, inner_test, counts, fold, truth_cell,
                                           float(candidate)))
                  for beta, inner_test in inner_betas]
        mean_gain = finite_mean(np.asarray(scores))
        if mean_gain > best_gain:
            best_gain, best_alpha = mean_gain, float(candidate)
    return best_alpha


def decode_gain(counts: np.ndarray, folds: list, settings: dict,
                permutation: np.ndarray = None, with_detail: bool = False) -> dict:
    """
    Description
    -----------
    The frozen WHAT statistic: mean held-out decode gain in nats per event, with everything the
    configuration fixes applied in one pass.

    Per outer fold the ridge is set from that fold's own training spike total, the amplitude is tuned by
    inner leave-one-session-out on the training sessions alone, the surface is refitted with per-session
    exposure offsets, and the held-out session's events are scored. The IDENTICAL procedure runs on a
    permuted draw, which is what keeps the exact permutation test exact: any advantage the fitting gives
    the observed value it gives the null too.

    The gain is the effect size, not merely a test statistic -- it is how much this unit contributes to
    decoding a given call. A pass requires it to be positive AND the permutation p to be small, because a
    statistic that gains significance by tightening its null rather than by predicting better does not
    qualify.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event; NOT permuted -- the draw re-pairs positions, leaving counts in place.
    folds (list)
        From :func:`decoding_context`.
    settings (dict)
        The ``vocal_decoding`` block.
    permutation (np.ndarray)
        Event reordering applied to positions (basis rows and truth cells), or None for the observed fit.
    with_detail (bool)
        Also collect the per-event gain, decode error, posterior entropy and the overdispersion index.
        Read off quantities already computed, but this function IS the permutation draw, so the detail
        is taken once on the observed pass and never inside the null.

    Returns
    -------
    result (dict)
        ``gain``, ``gain_uncalibrated`` (alpha fixed at 1), ``nats_per_spike``, ``n_masked``,
        ``n_folds_positive``, ``min_leave_one_fold_out`` (the worst pooled mean with any one fold
        dropped -- positive means no single session carries the result), ``per_fold`` (held-out
        session, gain, n events, alpha, ridge, n masked, and the fitted surface COEFFICIENTS, so a
        figure or a population decode can be rebuilt without refitting), and ``alphas`` / ``ridges``.
    """

    surface = settings["tuning_surface"]
    if surface["likelihood_family"] != "poisson":
        msg = (f"decode_gain implements the Poisson tuning inversion only; "
               f"likelihood_family is {surface['likelihood_family']!r}.")
        raise ValueError(msg)
    if surface["ridge_selection"]["mode"] != "count_normalized_fixed":
        msg = (f"decode_gain implements the fixed count-normalized ridge only; "
               f"ridge_selection.mode is {surface['ridge_selection']['mode']!r}.")
        raise ValueError(msg)

    ridge_fraction = surface["ridge_selection"]["ridge_frac"]
    n_steps = surface["solver"]["irls_n_steps"]
    calibration = surface["amplitude_calibration"]
    if calibration["enabled"] and calibration["mode"] != "train_tuned_inner_loso":
        msg = (f"the amplitude calibration is tuned by inner LOSO on the training sessions only; "
               f"amplitude_calibration.mode is {calibration['mode']!r}.")
        raise ValueError(msg)
    alpha_grid = ([float(a) for a in calibration["alpha_grid"]]
                  if calibration["enabled"] else [1.0])
    if min(alpha_grid) < calibration["alpha_min"]:
        # a negative amplitude would let the decoder INVERT a reversed map rather than decline to use
        # it, which is the one thing the calibration must never be able to do
        msg = (f"alpha_grid contains {min(alpha_grid)}, below alpha_min "
               f"{calibration['alpha_min']}.")
        raise ValueError(msg)

    order = np.arange(counts.size) if permutation is None else np.asarray(permutation)
    calibrated = np.full(counts.size, np.nan, dtype=np.float64)
    uncalibrated = np.full(counts.size, np.nan, dtype=np.float64)
    decode_error = np.full(counts.size, np.nan, dtype=np.float64)
    posterior_entropy = np.full(counts.size, np.nan, dtype=np.float64)
    per_fold, n_masked, dispersion = [], 0, []

    for fold in folds:
        basis = fold["event_basis"][order]
        truth_cell = fold["truth_cell"][order]
        train_rows, test_rows = fold["train_rows"], fold["test_rows"]
        ridge = ridge_fraction * float(counts[train_rows].sum())

        alpha = tuned_amplitude(counts, fold, basis, truth_cell, ridge, alpha_grid, n_steps,
                                surface["session_rate_offsets"])
        offsets = (session_exposure_offsets(counts[train_rows], fold["train_session_index"])
                   if surface["session_rate_offsets"] else None)
        beta = poisson_tuning_fit(basis[train_rows], counts[train_rows], ridge, n_steps, offsets)

        if with_detail:
            fold_gains, fold_detail = score_events(beta, test_rows, counts, fold, truth_cell, alpha,
                                                   with_detail=True)
            calibrated[test_rows] = fold_gains
            decode_error[test_rows] = fold_detail["decode_error"]
            posterior_entropy[test_rows] = fold_detail["posterior_entropy"]
            # the fitted surface's Pearson dispersion on the held-out counts: a diagnostic on whether
            # the Poisson likelihood is defensible for THIS unit, which nothing else reports
            rate = np.exp(np.clip(alpha * (fold["grid_basis"] @ beta[:-1]), -10.0, 5.0))
            expected = float(np.mean(rate)) * np.ones(test_rows.size)
            dispersion.append(float(np.mean((counts[test_rows] - expected) ** 2
                                            / np.clip(expected, 1e-9, None))))
        else:
            calibrated[test_rows] = score_events(beta, test_rows, counts, fold, truth_cell, alpha)
        uncalibrated[test_rows] = score_events(beta, test_rows, counts, fold, truth_cell, 1.0)
        fold_masked = int(np.sum(~fold["mask"][truth_cell[test_rows]]))
        n_masked += fold_masked
        per_fold.append({"held_out": fold["held_out"], "n_events": int(test_rows.size),
                         "alpha": alpha, "ridge": ridge, "n_masked": fold_masked,
                         "coefficients": beta,
                         "gain": finite_mean(calibrated[test_rows])})

    total_spikes = float(counts.sum())
    gain = finite_mean(calibrated)

    # The pooled mean is a summary of n fold values, and the failure it can hide is one session
    # carrying the whole result. Dropping each fold in turn and taking the WORST pooled mean answers
    # that directly; it costs nothing, needs no threshold, and is a descriptor rather than a gate.
    dropped = []
    for fold in folds:
        rest = np.setdiff1d(np.arange(counts.size), fold["test_rows"], assume_unique=False)
        dropped.append(finite_mean(calibrated[rest]))

    return {"gain": gain,
            "gain_uncalibrated": finite_mean(uncalibrated),
            "nats_per_spike": gain * counts.size / total_spikes if total_spikes > 0 else float("nan"),
            "n_masked": n_masked,
            "n_folds_positive": int(sum(1 for f in per_fold if f["gain"] > 0)),
            "min_leave_one_fold_out": float(np.min(dropped)) if dropped else float("nan"),
            "per_fold": per_fold,
            "alphas": [f["alpha"] for f in per_fold],
            "ridges": [f["ridge"] for f in per_fold],
            **({"per_event_gain": calibrated, "decode_error": decode_error,
                "posterior_entropy": posterior_entropy,
                "mean_decode_error": finite_mean(decode_error),
                "mean_posterior_entropy": finite_mean(posterior_entropy),
                "overdispersion_index": finite_mean(np.asarray(dispersion))} if with_detail else {})}
