"""
@author: bartulem
Vocalization identity: does a unit's firing around a USV's onset carry information about which USV it is?

The target is the USV's position on the QLVM torus. The decoder is a Bayesian inversion of a per-unit
Poisson tuning surface: ``lambda(x)`` is fitted on the training sessions' events (Fourier k=2 basis,
per-session exposure offsets, ridge set as a fraction of the training spike count), and for each
held-out event it is inverted against a kernel occupancy prior to give a posterior density over the
torus. The statistic is the gain, the log posterior minus the log prior at the USV's true position, in
nats per event. A non-negative amplitude calibration, tuned by inner leave-one-session-out on the
training sessions, scales the depth of the surface.

Exports the bases (:func:`fourier_basis`, :func:`bump_basis`), the fold context
(:func:`decoding_context`, :func:`basis_context`), the fit and scorer (:func:`poisson_tuning_fit`,
:func:`score_events`, :func:`tuned_amplitude`), the statistic :func:`decode_gain`, and the descriptors
(:func:`bump_basis_gain`, :func:`left_tail_anti_alignment`, :func:`flagged_descriptors`).
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.special import gammaln

from ..modeling.manifold_metric import signed_diff
from ..modeling.modeling_torus_geodesics import torus_grid as flat_torus_grid
from .deviance_metrics import finite_mean, finite_min

#: The tuning bases the decoder implements. Fourier is the shipped default; the local bump basis
#: is kept because its per-unit gain is recorded as a robustness descriptor.
TUNING_BASES = frozenset({"fourier", "bumps"})


def wrapped_square_distance(points: np.ndarray, nodes: np.ndarray, period: float) -> np.ndarray:
    """
    Description
    -----------
    Squared distance on the flat torus, taking the shorter way round in each dimension.

    The wrapping is delegated to ``manifold_metric.signed_diff``; this adds the cross-set shape
    (events against grid cells or basis nodes).

    Parameters
    ----------
    points (np.ndarray)
        ``(n_points, 2)`` positions.
    nodes (np.ndarray)
        ``(n_nodes, 2)`` positions.
    period (float)
        Per-axis wrap period of the torus (``vocalization_settings.torus_period``).

    Returns
    -------
    distances (np.ndarray)
        ``(n_points, n_nodes)`` squared distances.
    """

    difference = signed_diff(np.asarray(points, dtype=np.float64)[:, None, :],
                             np.asarray(nodes, dtype=np.float64)[None, :, :],
                             metric="torus", period=period)
    return np.sum(difference ** 2, axis=-1)


def fourier_basis(positions: np.ndarray, k_max: int, period: float) -> np.ndarray:
    """
    Description
    -----------
    Global periodic basis on the torus: harmonics in each dimension plus the two diagonal terms.

    Periodic by construction, so the fitted surface has no seam at the wrap. At ``k_max=2`` this is
    12 columns.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n_events, 2)`` torus positions.
    k_max (int)
        Highest harmonic.
    period (float)
        Per-axis wrap period of the torus.

    Returns
    -------
    basis (np.ndarray)
        ``(n_events, 4 * k_max + 4)`` design.
    """

    x = 2.0 * np.pi * positions[:, 0] / period
    y = 2.0 * np.pi * positions[:, 1] / period
    columns = []
    for k in range(1, k_max + 1):
        columns += [np.sin(k * x), np.cos(k * x), np.sin(k * y), np.cos(k * y)]
    columns += [np.sin(x + y), np.cos(x + y), np.sin(x - y), np.cos(x - y)]
    return np.column_stack(columns)


def bump_nodes(train_positions: np.ndarray, grid_n: int, sigma: float,
               min_occupancy: float, period: float) -> np.ndarray:
    """
    Description
    -----------
    Node centres for a local bump basis, kept only where training events actually live.

    The occupancy gate stops the basis placing unconstrained weights in empty regions of the torus.

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
    period (float)
        Per-axis wrap period of the torus.

    Returns
    -------
    nodes (np.ndarray)
        ``(n_kept, 2)`` surviving node centres.
    """

    nodes, _cell_area = torus_grid(grid_n, period)
    occupancy = np.exp(-wrapped_square_distance(train_positions, nodes, period)
                       / (2.0 * sigma ** 2)).sum(axis=0)
    return nodes[occupancy >= min_occupancy]


def bump_basis(positions: np.ndarray, nodes: np.ndarray, sigma: float, period: float) -> np.ndarray:
    """
    Description
    -----------
    Local wrapped-Gaussian basis: one column per node.

    The alternative to the Fourier basis; its per-unit gain is recorded as a robustness descriptor.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n_events, 2)`` positions.
    nodes (np.ndarray)
        ``(n_nodes, 2)`` centres.
    sigma (float)
        Bump width.
    period (float)
        Per-axis wrap period of the torus.

    Returns
    -------
    basis (np.ndarray)
        ``(n_events, n_nodes)`` design.
    """

    return np.exp(-wrapped_square_distance(positions, nodes, period) / (2.0 * sigma ** 2))


def torus_occupancy(train_positions: np.ndarray, grid_positions: np.ndarray,
                    bandwidth: float, period: float) -> np.ndarray:
    """
    Description
    -----------
    Wrapped-Gaussian kernel occupancy of the training events over a grid.

    This is the decoder's prior and also its mask. The occupancy is a kernel-weighted sum, so a cell
    can clear a threshold of 1.0 without containing a training USV.

    Parameters
    ----------
    train_positions (np.ndarray)
        ``(n_events, 2)`` training positions.
    grid_positions (np.ndarray)
        ``(n_cells, 2)`` grid cell centres.
    bandwidth (float)
        Kernel bandwidth.
    period (float)
        Per-axis wrap period of the torus.

    Returns
    -------
    occupancy (np.ndarray)
        ``(n_cells,)`` weighted counts.
    """

    return np.exp(-wrapped_square_distance(grid_positions, train_positions, period)
                  / (2.0 * bandwidth ** 2)).sum(axis=1)


def torus_grid(grid_n: int, period: float) -> tuple[np.ndarray, float]:
    """
    Description
    -----------
    Cell centres of a regular ``grid_n x grid_n`` grid on the torus, and the area of one cell.

    The centres come from the geodesic module's grid. The cell area turns a summed posterior into a
    density, so the gain does not depend on the grid resolution.

    Parameters
    ----------
    grid_n (int)
        Cells per dimension.
    period (float)
        Per-axis wrap period of the torus.

    Returns
    -------
    grid_positions (np.ndarray)
        ``(grid_n ** 2, 2)`` centres.
    cell_area (float)
        Area of one cell.
    """

    return flat_torus_grid(grid_n, period=period), (period / grid_n) ** 2


def poisson_tuning_fit(basis: np.ndarray, counts: np.ndarray, ridge: float, n_steps: int,
                       offset: np.ndarray = None) -> np.ndarray:
    """
    Description
    -----------
    Poisson GLM with a log link, fitted by IRLS, with an unpenalized intercept and a fixed exposure
    offset.

    The intercept carries the unit's overall rate and is not penalized, so rate differences are not
    pushed into the surface's shape. The offset enters the linear predictor but not the working
    response, so it is an exposure rather than a fitted covariate. The linear predictor is clipped
    before exponentiating to avoid overflow.

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

    With the offsets in place, position explains only within-session variation, so drift in the
    unit's rate across sessions is not read as tuning. The floor at 1e-3 keeps a session without
    spikes from sending the offset to negative infinity.

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


def inner_rotation_splits(train_sessions: list, rows_by_session: dict, event_times: np.ndarray | None,
                          single_session_blocks: int) -> list:
    """
    Description
    -----------
    The inner splits a fold tunes its calibration on, using its TRAINING sessions alone.

    With two or more training sessions the split is leave-one-training-session-out. With a single
    training session (a unit with two vocal sessions, or two vocal blocks) there is no session to hold
    out, so that session's events are ordered in time and cut into ``single_session_blocks`` contiguous
    blocks of equal event count, rotated leave-one-block-out -- the same fallback the kinematic encoding
    uses for a pool of one session. Contiguous blocks keep neighbouring events, which share a bout and a
    behavioural state, on the same side of a split. Without event times or with fewer than two blocks
    there are no inner splits, and the caller falls back to its uncalibrated default.

    Parameters
    ----------
    train_sessions (list)
        Session ids of the fold's training sessions.
    rows_by_session (dict)
        ``{session id: row indices of that session's events}``.
    event_times (np.ndarray)
        Time of each event in seconds (onset, or window left edge), used only to order the events of a
        single training session; ``None`` disables the single-session fallback.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

    Returns
    -------
    splits (list)
        ``(inner_train_rows, inner_test_rows)`` pairs.
    """

    if len(train_sessions) >= 2:
        splits = []
        for inner_held_out in train_sessions:
            rest = [s for s in train_sessions if s != inner_held_out]
            splits.append((np.concatenate([rows_by_session[s] for s in rest]), rows_by_session[inner_held_out]))
        return splits
    if len(train_sessions) == 0 or event_times is None or single_session_blocks < 2:
        return []
    rows = np.asarray(rows_by_session[train_sessions[0]])
    ordered = rows[np.argsort(np.asarray(event_times, dtype=np.float64)[rows], kind="stable")]
    blocks = [block for block in np.array_split(ordered, int(single_session_blocks)) if block.size]
    if len(blocks) < 2:
        return []
    return [(np.sort(np.concatenate([other for index, other in enumerate(blocks) if index != held])),
             np.sort(blocks[held])) for held in range(len(blocks))]


def decoding_context(positions: np.ndarray, session_index: np.ndarray, settings: dict,
                     period: float, event_times: np.ndarray | None = None,
                     single_session_blocks: int = 0) -> list:
    """
    Description
    -----------
    Precompute everything about a unit's LOSO folds that does NOT depend on spike counts.

    The bases, kernel occupancy, masked log-prior and truth cells depend only on positions and
    configuration, so they are built once and shared by the observed fit and every permutation draw.
    The basis and truth cells are stored for all events, because a draw re-pairs counts with
    positions; the prior and mask stay built from the real training positions. Each fold also carries
    the inner splits used to tune the amplitude calibration (:func:`inner_rotation_splits`):
    leave-one-training-session-out, or contiguous blocks of a fold's single training session.

    The occupancy mask is a decode-time gate: the posterior places no mass where no training USV
    occurred, and events whose true position is masked are excluded and counted.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n_events, 2)`` torus positions.
    session_index (np.ndarray)
        Session id per event.
    settings (dict)
        The ``vocal_decoding`` block; ``tuning_surface`` supplies grid, bandwidth, basis and mask knobs.
    period (float)
        Per-axis wrap period of the torus (``vocalization_settings.torus_period``). It is stored on
        each fold so the scorer uses the period the geometry was built with.
    event_times (np.ndarray)
        Onset time of each event in seconds, used to cut a fold's single training session into blocks
        for its inner splits. The per-unit dispatcher always passes it; ``None`` leaves such a fold
        without inner splits.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

    Returns
    -------
    folds (list)
        One dict per held-out session.
    """

    surface = settings["tuning_surface"]
    # The basis branch below is `fourier or else`, so an unknown name must fail here rather than
    # silently select the bump basis.
    if surface["tuning_basis"] not in TUNING_BASES:
        msg = (f"tuning_basis must be one of {sorted(TUNING_BASES)}; got "
               f"{surface['tuning_basis']!r}.")
        raise ValueError(msg)
    grid_positions, cell_area = torus_grid(surface["grid_n"], period)
    grid_n = surface["grid_n"]
    sessions = np.unique(session_index)
    if sessions.size < 2:
        # leave-one-session-out needs at least one training session per fold
        msg = f"the decoder needs at least two sessions; got {sessions.size}."
        raise ValueError(msg)
    rows_by_session = {int(s): np.flatnonzero(session_index == s) for s in sessions}

    ix = np.clip(np.floor(positions[:, 0] / period * grid_n), 0, grid_n - 1).astype(int)
    iy = np.clip(np.floor(positions[:, 1] / period * grid_n), 0, grid_n - 1).astype(int)
    truth_cell = ix * grid_n + iy

    folds = []
    for held_out in sessions:
        train_sessions = [int(s) for s in sessions if s != held_out]
        train_rows = (np.concatenate([rows_by_session[s] for s in train_sessions])
                      if train_sessions else np.empty(0, dtype=np.int64))
        train_positions = positions[train_rows]

        if surface["tuning_basis"] == "fourier":
            event_basis = fourier_basis(positions, surface["fourier_k"], period)
            grid_basis = fourier_basis(grid_positions, surface["fourier_k"], period)
        else:
            nodes = bump_nodes(train_positions, surface["bump_grid_n"], surface["bump_sigma"],
                               surface["min_basis_occupancy"], period)
            event_basis = bump_basis(positions, nodes, surface["bump_sigma"], period)
            grid_basis = bump_basis(grid_positions, nodes, surface["bump_sigma"], period)

        occupancy = torus_occupancy(train_positions, grid_positions, surface["kde_bandwidth"], period)
        prior = occupancy / (occupancy.sum() * cell_area)
        mask = occupancy >= surface["decode_grid_min_occupancy"]
        # A mask that excludes the whole grid leaves nothing to normalise; skip the division
        # rather than emit a NaN prior.
        masked_prior = np.where(mask, prior, 0.0)
        total = float(masked_prior.sum())
        masked_prior = masked_prior / (total * cell_area) if total > 0.0 else masked_prior

        inner_splits = inner_rotation_splits(train_sessions, rows_by_session, event_times,
                                             single_session_blocks)

        folds.append({"held_out": int(held_out), "train_sessions": train_sessions,
                      "train_rows": train_rows, "test_rows": rows_by_session[int(held_out)],
                      "train_session_index": np.asarray(session_index)[train_rows],
                      "event_basis": event_basis, "grid_basis": grid_basis,
                      "log_prior": np.log(np.clip(masked_prior, 1e-300, None)),
                      "masked_prior": masked_prior, "mask": mask,
                      "truth_cell": truth_cell, "cell_area": cell_area,
                      "grid_positions": grid_positions, "period": float(period),
                      "inner_splits": inner_splits})
    return folds


def surface_rate(beta: np.ndarray, fold: dict, alpha: float, scored_counts: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    The calibrated tuning surface's expected count in every grid cell for one scored session: the
    surface SHAPE centred on its occupancy-prior-weighted mean and scaled by ``alpha``, plus the LEVEL
    read from the scored session's own mean count.

    This is the single definition of the predicted rate, used by :func:`score_events` and by the
    overdispersion diagnostic.

    Parameters
    ----------
    beta (np.ndarray)
        Fitted surface coefficients, intercept last.
    fold (dict)
        From :func:`decoding_context`.
    alpha (float)
        Amplitude scaling of the surface shape; ``>= 0``.
    scored_counts (np.ndarray)
        Spike counts of the events being scored; their mean sets the level.

    Returns
    -------
    rate (np.ndarray)
        ``(n_cells,)`` expected count per grid cell.
    """

    shape = fold["grid_basis"] @ beta[:-1]
    # centred on the occupancy prior whichever prior the posterior uses: the centre fixes what
    # `alpha` scales
    weights = fold["masked_prior"] * fold["cell_area"]
    centre = float(np.sum(shape * weights) / max(float(np.sum(weights)), 1e-12))
    level = np.log(max(float(np.mean(scored_counts)), 1e-3))
    return np.exp(np.clip(alpha * (shape - centre) + level, -10.0, 5.0))


def score_events(beta: np.ndarray, rows: np.ndarray, counts: np.ndarray, fold: dict,
                 truth_cell: np.ndarray, alpha: float, with_detail: bool = False,
                 log_prior_events: np.ndarray | None = None):
    """
    Description
    -----------
    Per-event decode gain -- log posterior minus log prior at the event's true position, in nats.

    By default every event shares the fold's occupancy prior and the gain is the vocalization-identity
    statistic. Given ``log_prior_events`` (a per-event prior, such as a von Mises around the behaviour
    control's prediction), the same posterior is formed against that prior and the gain becomes what the
    neuron adds beyond it; nothing else changes.

    The surface is split into a shape, centred on its prior-weighted mean and scaled by ``alpha``, and
    a level, read from the scored session's own mean count (a single number with no position
    information). Events whose true position is masked return NaN.

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
        Also return the per-event decode error, posterior entropy and decoded position. Taken on the
        observed pass only.
    log_prior_events (np.ndarray)
        ``(n_events, n_cells)`` log prior density per event over the fold's grid, already masked and
        normalised, indexed in the order counts are paired with; or None for the fold's shared
        occupancy prior. The gain is measured against the prior the posterior was formed with.

    Returns
    -------
    gains (np.ndarray)
        One value per row; NaN where the true position is masked. With ``with_detail``, a
        ``(gains, detail)`` pair.
    """

    scored_counts = counts[rows]
    rate = surface_rate(beta, fold, alpha, scored_counts)
    log_rate = np.log(np.clip(rate, 1e-12, None))

    loglik = (scored_counts[:, None] * log_rate[None, :] - rate[None, :]
              - gammaln(scored_counts + 1.0)[:, None])
    log_prior = (fold["log_prior"][None, :] if log_prior_events is None
                 else np.asarray(log_prior_events)[rows])
    posterior = log_prior + loglik
    posterior -= posterior.max(axis=1, keepdims=True)
    density = np.exp(posterior)
    density /= density.sum(axis=1, keepdims=True) * fold["cell_area"]

    cells = truth_cell[rows]
    at_truth = density[np.arange(scored_counts.size), cells]
    prior_at_truth = (fold["masked_prior"][cells] if log_prior_events is None
                      else np.exp(log_prior[np.arange(scored_counts.size), cells]))
    outside = ~fold["mask"][cells]

    posterior_log = np.log(np.clip(at_truth, 1e-300, None))
    prior_log = np.log(np.clip(prior_at_truth, 1e-300, None))
    gains = posterior_log - prior_log
    gains[outside] = np.nan
    if not with_detail:
        return gains

    # Detail is read off the posterior already built; opt-in because this runs inside every draw.
    period = fold["period"]
    mass = density * fold["cell_area"]
    angles = 2.0 * np.pi * fold["grid_positions"] / period
    decoded = np.empty((scored_counts.size, 2), dtype=np.float64)
    for dimension in range(2):
        cosine = mass @ np.cos(angles[:, dimension])
        sine = mass @ np.sin(angles[:, dimension])
        decoded[:, dimension] = np.mod(np.arctan2(sine, cosine) / (2.0 * np.pi) * period, period)
    truth = fold["grid_positions"][cells]
    error = np.sqrt(np.sum(signed_diff(decoded, truth, metric="torus", period=period) ** 2, axis=1))
    entropy = -np.sum(mass * np.log(np.clip(mass, 1e-300, None)), axis=1)
    error[outside] = np.nan
    entropy[outside] = np.nan
    return gains, {"decode_error": error, "posterior_entropy": entropy, "decoded": decoded}


def tuned_amplitude(counts: np.ndarray, fold: dict, basis: np.ndarray, truth_cell: np.ndarray,
                    ridge: float, alpha_grid: list, n_steps: int,
                    session_rate_offsets: bool = True,
                    log_prior_events: np.ndarray | None = None) -> float:
    """
    Description
    -----------
    Choose this fold's amplitude scaling on its TRAINING sessions alone, by inner leave-one-session-out.

    The training sessions are rotated among themselves; each candidate alpha is scored on the inner
    held-in session, and the best mean is kept. The same procedure runs inside every permutation draw,
    so the extra parameter gives the observed value no advantage over the null. The inner scoring
    reuses the outer fold's prior and mask.

    A fold whose single training session yields no inner blocks has no inner rotation and falls back to
    the grid point nearest 1.0. The grid floor of 0.1 regularizes the calibration itself.

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
    log_prior_events (np.ndarray)
        Per-event log prior in the same order as ``basis``, or None for the fold's occupancy prior.
        The calibration is tuned under the same prior the statistic is scored with.

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
                                           float(candidate), log_prior_events=log_prior_events))
                  for beta, inner_test in inner_betas]
        mean_gain = finite_mean(np.asarray(scores))
        if mean_gain > best_gain:
            best_gain, best_alpha = mean_gain, float(candidate)
    return best_alpha


def decode_gain(counts: np.ndarray, folds: list, settings: dict,
                permutation: np.ndarray = None, with_detail: bool = False,
                log_prior_events: list | None = None) -> dict:
    """
    Description
    -----------
    The frozen vocalization-identity statistic: mean held-out decode gain in nats per event, with everything the
    configuration fixes applied in one pass.

    Per outer fold the ridge is set from that fold's training spike total, the amplitude is tuned by
    inner leave-one-session-out, the surface is refitted with per-session exposure offsets, and the
    held-out session's events are scored. The identical procedure runs on every permuted draw. The
    gain is also the effect size; the pass rule is applied at the cohort step.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event; not permuted (a draw re-pairs positions, leaving counts in place).
    folds (list)
        From :func:`decoding_context`.
    settings (dict)
        The ``vocal_decoding`` block.
    permutation (np.ndarray)
        Event reordering applied to positions (basis rows and truth cells), or None for the observed fit.
    with_detail (bool)
        Also collect the per-event gain, decode error, posterior entropy and the overdispersion index.
        Taken on the observed pass only.
    log_prior_events (list)
        One ``(n_events, n_cells)`` per-event log prior per fold, in the folds' order and in the
        unpermuted event order, or None for each fold's occupancy prior. A prior row belongs to its
        event's position, so under a draw it is reordered with the positions.

    Returns
    -------
    result (dict)
        ``gain``, ``gain_uncalibrated`` (alpha fixed at 1), ``nats_per_spike`` (summed finite gain over
        the spikes of those same events), ``n_masked``, ``n_folds_positive``,
        ``min_leave_one_fold_out`` (the worst pooled mean with any one fold dropped, over the folds
        whose left-out mean is finite -- ``np.nanmin`` semantics; positive means no single session
        carries the result), ``per_fold`` (held-out
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
        # a negative amplitude would let the decoder invert a reversed map
        msg = (f"alpha_grid contains {min(alpha_grid)}, below alpha_min "
               f"{calibration['alpha_min']}.")
        raise ValueError(msg)

    order = np.arange(counts.size) if permutation is None else np.asarray(permutation)
    calibrated = np.full(counts.size, np.nan, dtype=np.float64)
    uncalibrated = np.full(counts.size, np.nan, dtype=np.float64)
    decode_error = np.full(counts.size, np.nan, dtype=np.float64)
    posterior_entropy = np.full(counts.size, np.nan, dtype=np.float64)
    per_fold, n_masked, dispersion = [], 0, []

    for fold_index, fold in enumerate(folds):
        basis = fold["event_basis"][order]
        truth_cell = fold["truth_cell"][order]
        # the prior row travels with its position under a draw, never with the count
        prior = (None if log_prior_events is None
                 else np.asarray(log_prior_events[fold_index])[order])
        train_rows, test_rows = fold["train_rows"], fold["test_rows"]
        ridge = ridge_fraction * float(counts[train_rows].sum())

        alpha = tuned_amplitude(counts, fold, basis, truth_cell, ridge, alpha_grid, n_steps,
                                surface["session_rate_offsets"], log_prior_events=prior)
        offsets = (session_exposure_offsets(counts[train_rows], fold["train_session_index"])
                   if surface["session_rate_offsets"] else None)
        beta = poisson_tuning_fit(basis[train_rows], counts[train_rows], ridge, n_steps, offsets)

        if with_detail:
            fold_gains, fold_detail = score_events(beta, test_rows, counts, fold, truth_cell, alpha,
                                                   with_detail=True, log_prior_events=prior)
            calibrated[test_rows] = fold_gains
            decode_error[test_rows] = fold_detail["decode_error"]
            posterior_entropy[test_rows] = fold_detail["posterior_entropy"]
            # Pearson dispersion of the held-out counts around the calibrated rate at each event's
            # cell: a check on the Poisson likelihood (about 1 for an untuned Poisson unit)
            expected = surface_rate(beta, fold, alpha, counts[test_rows])[truth_cell[test_rows]]
            dispersion.append(float(np.mean((counts[test_rows] - expected) ** 2
                                            / np.clip(expected, 1e-9, None))))
        else:
            calibrated[test_rows] = score_events(beta, test_rows, counts, fold, truth_cell, alpha,
                                                 log_prior_events=prior)
        uncalibrated[test_rows] = score_events(beta, test_rows, counts, fold, truth_cell, 1.0,
                                               log_prior_events=prior)
        fold_masked = int(np.sum(~fold["mask"][truth_cell[test_rows]]))
        n_masked += fold_masked
        per_fold.append({"held_out": fold["held_out"], "n_events": int(test_rows.size),
                         "alpha": alpha, "ridge": ridge, "n_masked": fold_masked,
                         "coefficients": beta,
                         "gain": finite_mean(calibrated[test_rows])})

    gain = finite_mean(calibrated)
    # Nats per spike over the scored events only: their summed gain over their spikes.
    scored = np.isfinite(calibrated)
    scored_spikes = float(counts[scored].sum())

    # The worst pooled mean with any one fold dropped: a descriptor of whether one session carries
    # the result.
    dropped = []
    for fold in folds:
        rest = np.setdiff1d(np.arange(counts.size), fold["test_rows"], assume_unique=False)
        dropped.append(finite_mean(calibrated[rest]))

    return {"gain": gain,
            "gain_uncalibrated": finite_mean(uncalibrated),
            "nats_per_spike": (float(calibrated[scored].sum()) / scored_spikes if scored_spikes > 0
                               else float("nan")),
            "n_masked": n_masked,
            "n_folds_positive": int(sum(1 for f in per_fold if f["gain"] > 0)),
            "min_leave_one_fold_out": finite_min(dropped),
            "per_fold": per_fold,
            "alphas": [f["alpha"] for f in per_fold],
            "ridges": [f["ridge"] for f in per_fold],
            **({"per_event_gain": calibrated, "decode_error": decode_error,
                "posterior_entropy": posterior_entropy,
                "mean_decode_error": finite_mean(decode_error),
                "mean_posterior_entropy": finite_mean(posterior_entropy),
                **({"overdispersion_index": finite_mean(np.asarray(dispersion))}
                   if settings["record_overdispersion_index"] else {})} if with_detail else {})}


def bump_basis_gain(counts: np.ndarray, positions: np.ndarray, session_index: np.ndarray,
                    settings: dict, period: float, event_times: np.ndarray | None = None,
                    single_session_blocks: int = 0) -> float:
    """
    Description
    -----------
    Refit the whole decoder under the LOCAL bump basis and return its gain.

    A robustness descriptor, taken on the observed pass only, so the local-basis comparison can be
    made from a completed run. The basis is chosen inside :func:`decoding_context`, so
    this builds its own context.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    positions (np.ndarray)
        ``(n, 2)`` torus positions of the events.
    session_index (np.ndarray)
        Session slot per event, defining the LOSO folds.
    settings (dict)
        The ``vocal_decoding`` block. Its ``tuning_surface.tuning_basis`` is overridden locally; the
        caller's dict is not mutated.
    period (float)
        Per-axis wrap period of the torus.
    event_times (np.ndarray)
        Onset time of each event in seconds, passed to :func:`decoding_context` for the inner splits
        of a fold with a single training session; ``None`` leaves such a fold without them.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

    Returns
    -------
    gain (float)
        Pooled event-weighted gain in nats per event under the bump basis, or NaN if it cannot be fit.
    """

    return float(basis_context(positions, session_index, settings, "bumps", period, event_times,
                               single_session_blocks)[1](counts)["gain"])


def basis_context(positions: np.ndarray, session_index: np.ndarray, settings: dict,
                  basis: str, period: float, event_times: np.ndarray | None = None,
                  single_session_blocks: int = 0) -> tuple:
    """
    Description
    -----------
    Build the LOSO folds for one tuning basis and return them with a scorer bound to them.

    A descriptor uses the scorer once; a permutation null reuses the folds across draws and drives
    :func:`decode_gain` itself, so the bump basis can be tested against its own null. The basis
    override is applied to a deep copy of the settings.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n, 2)`` torus positions of the events.
    session_index (np.ndarray)
        Session slot per event, defining the folds.
    settings (dict)
        The ``vocal_decoding`` block.
    basis (str)
        One of :data:`TUNING_BASES`.
    period (float)
        Per-axis wrap period of the torus.
    event_times (np.ndarray)
        Onset time of each event in seconds, passed to :func:`decoding_context` for the inner splits
        of a fold with a single training session; ``None`` leaves such a fold without them.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

    Returns
    -------
    folds, score (tuple)
        The fold list from :func:`decoding_context`, and a callable
        ``score(counts, permutation=None, with_detail=False)`` returning :func:`decode_gain`'s dict.
    """

    if basis not in TUNING_BASES:
        msg = f"basis must be one of {sorted(TUNING_BASES)}; got {basis!r}."
        raise ValueError(msg)
    local = copy.deepcopy(settings)
    local["tuning_surface"]["tuning_basis"] = basis
    folds = decoding_context(positions, session_index, local, period, event_times, single_session_blocks)

    def score(counts: np.ndarray, permutation: np.ndarray = None,
              with_detail: bool = False) -> dict:
        """Score counts against these folds under the chosen basis."""
        return decode_gain(counts, folds, local, permutation, with_detail)

    return folds, score


def left_tail_anti_alignment(null: np.ndarray, observed: float) -> dict:
    """
    Description
    -----------
    The LEFT tail of the permutation null: is this unit significantly ANTI-aligned?

    A unit can decode reliably backwards, its surface pointing the wrong way more consistently than
    chance allows. This is recorded as a descriptor, never a pass, and separates a unit with no
    content from one whose content the decoder reads inverted.

    Parameters
    ----------
    null (np.ndarray)
        The permutation null distribution of the gain.
    observed (float)
        The observed gain.

    Returns
    -------
    descriptor (dict)
        ``p_left`` (fraction of draws at or below the observed, the anti-alignment exceedance),
        ``exceed_count_left`` and ``anti_aligned`` (observed below the null mean AND ``p_left`` at or
        under 0.01 -- a descriptor flag, never a verdict).
    """

    null = np.asarray(null, dtype=np.float64)
    below = int(np.sum(null <= observed))
    p_left = (1.0 + below) / (1.0 + null.size)
    return {"p_left": p_left, "exceed_count_left": below,
            "anti_aligned": bool(observed < float(null.mean()) and p_left <= 0.01)}


def flagged_descriptors(counts: np.ndarray, positions: np.ndarray, session_index: np.ndarray,
                        settings: dict, period: float, observed_gain: float, null: np.ndarray,
                        event_times: np.ndarray | None = None, single_session_blocks: int = 0) -> dict:
    """
    Description
    -----------
    Compute the vocalization-identity descriptors the settings ask to record, honouring each flag.

    Reads the ``record_bump_basis_gain`` and ``record_left_tail_anti_alignment`` flags; the
    overdispersion flag is read in :func:`decode_gain`.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    positions (np.ndarray)
        ``(n, 2)`` torus positions.
    session_index (np.ndarray)
        Session slot per event.
    settings (dict)
        The ``vocal_decoding`` block.
    period (float)
        Per-axis wrap period of the torus.
    observed_gain (float)
        The observed Fourier-basis gain.
    null (np.ndarray)
        The permutation null of that gain.
    event_times (np.ndarray)
        Onset time of each event in seconds, passed to :func:`decoding_context` for the inner splits
        of a fold with a single training session; ``None`` leaves such a fold without them.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

    Returns
    -------
    descriptors (dict)
        ``bump_basis_gain`` when ``tuning_surface.record_bump_basis_gain``, and the left-tail keys
        when ``discrimination_null.record_left_tail_anti_alignment``. Absent keys mean the flag is off,
        never that the quantity was unavailable.
    """

    out: dict = {}
    if settings["tuning_surface"]["record_bump_basis_gain"]:
        out["bump_basis_gain"] = bump_basis_gain(counts, positions, session_index, settings, period, event_times,
                                                 single_session_blocks)
    if settings["discrimination_null"]["record_left_tail_anti_alignment"]:
        out.update(left_tail_anti_alignment(null, observed_gain))
    return out
