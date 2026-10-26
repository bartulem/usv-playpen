"""
@author: bartulem
Vocal occurrence: does a unit's firing distinguish the window around a vocalization's onset from silence?

Positives are the unit's spike count in a peri-onset window around each clean onset of the recorded
animal's USVs; negatives are same-width counts in quiet tiles, which share the kinematic encoding's
definition of silence and are capped per session upstream. The decoder is the two-class counterpart of
the identity decoders: one Poisson rate per class, relative to each session's quiet level, fitted on the
training sessions; an amplitude calibration tuned on the training sessions alone; and a posterior over
the two classes under a fixed equal prior. The statistic is the held-out gain, the log posterior of the
true class minus log 1/2, averaged over each class separately and then across the two, in nats per
window. The null circularly shifts each session's spike train, recounts both classes and refits.

Exports: ``counts_in_windows``, ``quiet_tile_edges``, ``occurrence_context``, ``occurrence_rates``,
``occurrence_log_posterior``, ``held_out_quiet_levels``, ``tuned_occurrence_amplitude``,
``vocal_occurrence_statistic`` and ``vocal_occurrence_shift_null``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .deviance_metrics import area_under_roc, contiguous_blocks
from .neural_vocal_decoding import inner_rotation_splits
from .shift_null_inference import shift_range_seconds

#: The occurrence prior is fixed at one half for each class. The number of quiet tiles is set by the
#: subsampling cap, so their observed share is arbitrary and would otherwise set the prior.
EQUAL_PRIOR_LOG = float(np.log(0.5))


def counts_in_windows(spike_seconds: np.ndarray, left_edges: np.ndarray, width: float) -> np.ndarray:
    """
    Description
    -----------
    Spike count in each half-open window ``[left, left + width)``.

    Uses two searchsorted lookups per window rather than binning to a raster, since the windows are
    short against the session and a raster would be almost entirely empty.

    Parameters
    ----------
    spike_seconds (np.ndarray)
        Spike times in seconds, ascending.
    left_edges (np.ndarray)
        Window start times in seconds.
    width (float)
        Window width in seconds.

    Returns
    -------
    counts (np.ndarray)
        Integer count per window.
    """

    times = np.asarray(spike_seconds, dtype=np.float64)
    left = np.asarray(left_edges, dtype=np.float64)
    return (np.searchsorted(times, left + width, side="left")
            - np.searchsorted(times, left, side="left")).astype(np.float64)


def quiet_tile_edges(vocalization_starts: np.ndarray, vocalization_stops: np.ndarray, duration: float,
                     width: float, history_pre_seconds: float, clean_post_seconds: float) -> np.ndarray:
    """
    Description
    -----------
    Left edges of non-overlapping quiet tiles, tiled through the gaps between guard-banded USVs.

    The guard band is the kinematic encoding's quiet definition -- a USV forbids
    ``[start - clean_post, stop + history_pre]`` -- applied to USVs from every emitter, so a tile is
    silent of the partner as well as the recorded animal. Tiles are contiguous and
    non-overlapping within each gap, and a gap shorter than one tile contributes nothing.

    Parameters
    ----------
    vocalization_starts (np.ndarray)
        Vocalization onsets in seconds, ALL emitters.
    vocalization_stops (np.ndarray)
        Vocalization offsets in seconds, ALL emitters.
    duration (float)
        Session duration in seconds.
    width (float)
        Tile width; must equal the peri-onset window width so the counts are comparable.
    history_pre_seconds (float)
        Guard after each USV's stop; equal to the kinematic history length, so no vocalization falls
        inside the history of a quiet tile.
    clean_post_seconds (float)
        Guard before each USV's start (the forward look).

    Returns
    -------
    edges (np.ndarray)
        Tile start times in seconds.
    """

    starts = np.asarray(vocalization_starts, dtype=np.float64) - clean_post_seconds
    stops = np.asarray(vocalization_stops, dtype=np.float64) + history_pre_seconds
    if starts.size == 0:
        n_tiles = int(np.floor(duration / width))
        return width * np.arange(max(n_tiles, 0), dtype=np.float64)

    order = np.argsort(starts)
    starts, stops = starts[order], stops[order]

    merged = []
    for begin, end in zip(starts, stops, strict=True):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])

    edges, previous_end = [], 0.0
    for begin, end in [*merged, [duration, duration]]:
        clipped = min(max(begin, 0.0), duration)
        if clipped > previous_end:
            n_tiles = int(np.floor((clipped - previous_end) / width))
            if n_tiles > 0:
                edges.append(previous_end + width * np.arange(n_tiles))
        previous_end = max(previous_end, min(max(end, 0.0), duration))
    return np.concatenate(edges) if edges else np.empty(0, dtype=np.float64)


def occurrence_context(labels: np.ndarray, session_index: np.ndarray, event_times: np.ndarray,
                       single_session_blocks: int) -> list:
    """
    Description
    -----------
    The leave-one-session-out folds of the occurrence decoder, with everything that does not depend on
    spike counts.

    Each fold holds out one session and carries the inner splits its amplitude calibration is tuned on
    (:func:`neural_vocal_decoding.inner_rotation_splits`): the other training sessions in turn, or
    contiguous blocks of a single training session. The folds are shared by the observed fit and every
    null draw, since a circular shift changes the counts and nothing else.

    Parameters
    ----------
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    session_index (np.ndarray)
        Session id per window.
    event_times (np.ndarray)
        Left edge of each window in seconds, used to order a single training session into blocks.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

    Returns
    -------
    folds (list)
        One dict per held-out session: ``held_out``, ``train_rows``, ``test_rows`` and ``inner_splits``.
    """

    sessions = np.unique(session_index)
    if sessions.size < 2:
        msg = f"the occurrence decoder needs at least two sessions; got {sessions.size}."
        raise ValueError(msg)
    if not np.isin(np.asarray(labels), (0.0, 1.0)).all():
        msg = "occurrence labels must be 1 (peri-onset window) or 0 (quiet tile)."
        raise ValueError(msg)
    rows_by_session = {int(s): np.flatnonzero(session_index == s) for s in sessions}
    folds = []
    for held_out in sessions:
        train_sessions = [int(s) for s in sessions if s != held_out]
        folds.append({"held_out": int(held_out),
                      "train_rows": np.concatenate([rows_by_session[s] for s in train_sessions]),
                      "test_rows": rows_by_session[int(held_out)],
                      "inner_splits": inner_rotation_splits(train_sessions, rows_by_session, event_times,
                                                            single_session_blocks)})
    return folds


def session_quiet_levels(counts: np.ndarray, labels: np.ndarray, session_index: np.ndarray,
                         rows: np.ndarray, mean_count_floor: float) -> np.ndarray:
    """
    Description
    -----------
    Each window's session quiet level: the mean count over that session's quiet tiles among ``rows``.

    This is the occurrence decoder's exposure. Expressing each class's rate relative to the session's
    quiet level lets one pair of class rates serve sessions that fire at different speeds, in the same
    way the identity decoder's per-session exposure offsets do. A session with no quiet tile among the
    rows falls back to the mean over all its windows; the floor keeps a session silent in quiet tiles
    finite.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per window.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    session_index (np.ndarray)
        Session id per window.
    rows (np.ndarray)
        The windows to compute levels for (a fold's training rows).
    mean_count_floor (float)
        Lower bound on a level (``vocal_occurrence.mean_count_floor``).

    Returns
    -------
    levels (np.ndarray)
        Quiet level per window in ``rows``, in the same order.
    """

    rows = np.asarray(rows)
    levels = np.empty(rows.size, dtype=np.float64)
    for session in np.unique(session_index[rows]):
        in_session = session_index[rows] == session
        session_rows = rows[in_session]
        quiet = session_rows[labels[session_rows] < 0.5]
        reference = quiet if quiet.size else session_rows
        levels[in_session] = max(float(np.mean(counts[reference])), mean_count_floor)
    return levels


def held_out_quiet_levels(counts: np.ndarray, labels: np.ndarray, event_times: np.ndarray, rows: np.ndarray,
                          n_blocks: int, mean_count_floor: float) -> np.ndarray:
    """
    Description
    -----------
    The quiet level of a held-out session, CROSS-FITTED: each window's level comes from the quiet tiles
    of the session's other contiguous blocks.

    The level is the one quantity the held-out session supplies about itself, and it is computed from its
    quiet tiles, so a tile would otherwise help set the baseline it is then scored against. Following the
    cross-fitting rule used for every frozen-model score in this analysis (``kinematic_encoding.
    calibration``), the held-out windows are ordered in time and cut into ``n_blocks`` contiguous blocks,
    and the level of each block is the mean count of the quiet tiles in the others. A block whose other
    blocks hold no quiet tile takes the mean over all the session's quiet tiles, and a session with none
    takes the mean over all its windows.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per window.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    event_times (np.ndarray)
        Left edge of each window in seconds.
    rows (np.ndarray)
        The held-out session's windows.
    n_blocks (int)
        Cross-fitting blocks (``kinematic_encoding.calibration.n_blocks``).
    mean_count_floor (float)
        Lower bound on a level.

    Returns
    -------
    levels (np.ndarray)
        Quiet level per window in ``rows``, in the same order.
    """

    rows = np.asarray(rows)
    order = np.argsort(np.asarray(event_times, dtype=np.float64)[rows], kind="stable")
    block_of = np.empty(rows.size, dtype=np.int64)
    block_of[order] = contiguous_blocks(np.zeros(rows.size, dtype=np.int64), int(n_blocks))
    quiet = labels[rows] < 0.5
    fallback = (float(np.mean(counts[rows][quiet])) if quiet.any()
                else (float(np.mean(counts[rows])) if rows.size else 0.0))
    levels = np.empty(rows.size, dtype=np.float64)
    for block in np.unique(block_of):
        in_block = block_of == block
        reference = quiet & ~in_block
        level = float(np.mean(counts[rows][reference])) if reference.any() else fallback
        levels[in_block] = max(level, mean_count_floor)
    return levels


def occurrence_rates(counts: np.ndarray, labels: np.ndarray, levels: np.ndarray, ridge: float) -> np.ndarray:
    """
    Description
    -----------
    The two class rates, each relative to the quiet level, shrunk toward their pooled value.

    Under ``E[n] = level * r_class`` the maximum-likelihood rate of a class is its total count over its
    total exposure. The shrinkage adds ``ridge`` pseudo-windows at the mean exposure and the pooled rate,
    as the categorical decoder does, with ``ridge`` passed in as a fraction of the training spike count.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per training window.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    levels (np.ndarray)
        Quiet level (exposure) per training window.
    ridge (float)
        Shrinkage weight, in pseudo-windows.

    Returns
    -------
    rates (np.ndarray)
        ``(2,)`` strictly positive rates, quiet first.
    """

    exposure_mean = float(np.mean(levels)) if levels.size else 1.0
    pooled = float(counts.sum()) / max(float(levels.sum()), 1e-12)
    rates = np.full(2, pooled, dtype=np.float64)
    for code in (0, 1):
        rows = (labels > 0.5) if code == 1 else (labels < 0.5)
        if rows.any():
            rates[code] = ((counts[rows].sum() + ridge * pooled * exposure_mean)
                           / (levels[rows].sum() + ridge * exposure_mean))
    return np.maximum(rates, 1e-9)


def occurrence_log_posterior(counts: np.ndarray, rates: np.ndarray, levels: np.ndarray,
                             alpha: float) -> np.ndarray:
    """
    Description
    -----------
    Log posterior of the two classes for each window, under the fixed equal prior.

    ``alpha`` scales each class's log-rate deviation from the mean of the two, tempering the contrast
    without changing its level or its direction (0 returns the prior, 1 the training fit as is); the
    level is the window's own quiet level.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per scored window.
    rates (np.ndarray)
        ``(2,)`` class rates relative to the quiet level, quiet first.
    levels (np.ndarray)
        Quiet level per scored window.
    alpha (float)
        Amplitude calibration, ``>= 0``.

    Returns
    -------
    log_posterior (np.ndarray)
        ``(n_windows, 2)`` log posterior, quiet first, normalised per row.
    """

    log_rates = np.log(rates)
    centre = float(log_rates.mean())
    tilted = np.exp(centre + alpha * (log_rates - centre))
    expected = levels[:, None] * tilted[None, :]
    log_likelihood = counts[:, None] * np.log(expected) - expected
    log_joint = log_likelihood + EQUAL_PRIOR_LOG
    return log_joint - logsumexp(log_joint, axis=1, keepdims=True)


def class_balanced_mean(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Description
    -----------
    Mean over each class separately, then the mean of the two.

    With the prior fixed at one half, this is the gain the decoder earns with the two classes weighted
    equally, so the statistic does not depend on how many quiet tiles were drawn. A class with no
    window leaves the result undefined.

    Parameters
    ----------
    values (np.ndarray)
        Per-window values.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.

    Returns
    -------
    mean (float)
        The class-balanced mean, or NaN when a class is missing.
    """

    positive, negative = labels > 0.5, labels < 0.5
    if not positive.any() or not negative.any():
        return float("nan")
    return 0.5 * float(np.mean(values[positive])) + 0.5 * float(np.mean(values[negative]))


def score_split(counts: np.ndarray, labels: np.ndarray, session_index: np.ndarray, event_times: np.ndarray,
                train_rows: np.ndarray, test_rows: np.ndarray, alpha: float, ridge_fraction: float,
                mean_count_floor: float, crossfit_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Fit the class rates on ``train_rows`` and score the windows of ``test_rows``.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per window.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    session_index (np.ndarray)
        Session id per window.
    event_times (np.ndarray)
        Left edge of each window in seconds.
    train_rows (np.ndarray)
        Training windows.
    test_rows (np.ndarray)
        Scored windows (one session, or one block of a session).
    alpha (float)
        Amplitude calibration.
    ridge_fraction (float)
        Ridge as a fraction of the training spike count.
    mean_count_floor (float)
        Lower bound on a quiet level.
    crossfit_blocks (int)
        Cross-fitting blocks for the scored windows' quiet level.

    Returns
    -------
    gain (np.ndarray)
        Per-window gain over the prior, in nats, aligned with ``test_rows``.
    probability_vocal (np.ndarray)
        Posterior probability of the peri-onset class per scored window.
    rates (np.ndarray)
        The fitted ``(2,)`` class rates, quiet first.
    """

    train_levels = session_quiet_levels(counts, labels, session_index, train_rows, mean_count_floor)
    ridge = ridge_fraction * float(counts[train_rows].sum())
    rates = occurrence_rates(counts[train_rows], labels[train_rows], train_levels, ridge)
    test_levels = held_out_quiet_levels(counts, labels, event_times, test_rows, crossfit_blocks,
                                        mean_count_floor)
    log_posterior = occurrence_log_posterior(counts[test_rows], rates, test_levels, alpha)
    truth = (labels[test_rows] > 0.5).astype(np.int64)
    gain = log_posterior[np.arange(truth.size), truth] - EQUAL_PRIOR_LOG
    return gain, np.exp(log_posterior[:, 1]), rates


def tuned_occurrence_amplitude(counts: np.ndarray, labels: np.ndarray, session_index: np.ndarray,
                               event_times: np.ndarray, fold: dict, axis: dict, crossfit_blocks: int) -> float:
    """
    Description
    -----------
    Choose the fold's amplitude calibration on its TRAINING sessions alone, by its inner splits.

    Each candidate is scored by the class-balanced gain on each inner held-out session (or block) and
    the best mean is kept. The same search runs inside every null draw, so the calibration gives the
    observed value no advantage over the null. A fold without inner splits uses 1.0, the training fit as
    is.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per window.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    session_index (np.ndarray)
        Session id per window.
    event_times (np.ndarray)
        Left edge of each window in seconds.
    fold (dict)
        One entry of :func:`occurrence_context`.
    axis (dict)
        The ``vocal_decoding.vocal_occurrence`` block.
    crossfit_blocks (int)
        Cross-fitting blocks for a scored session's quiet level.

    Returns
    -------
    alpha (float)
        The selected calibration.
    """

    calibration = axis["amplitude_calibration"]
    if not calibration["enabled"] or not fold["inner_splits"]:
        return 1.0
    best_alpha, best_score = 1.0, -np.inf
    for alpha in (float(a) for a in calibration["alpha_grid"]):
        scores = []
        for inner_train, inner_test in fold["inner_splits"]:
            gain, _probability, _rates = score_split(counts, labels, session_index, event_times, inner_train,
                                                     inner_test, alpha, axis["ridge_frac"],
                                                     axis["mean_count_floor"], crossfit_blocks)
            score = class_balanced_mean(gain, labels[inner_test])
            if np.isfinite(score):
                scores.append(score)
        if scores and float(np.mean(scores)) > best_score:
            best_alpha, best_score = alpha, float(np.mean(scores))
    return best_alpha


def vocal_occurrence_statistic(counts: np.ndarray, labels: np.ndarray, session_index: np.ndarray,
                               event_times: np.ndarray, folds: list, axis: dict, crossfit_blocks: int) -> dict:
    """
    Description
    -----------
    The vocal-occurrence statistic: held-out class-balanced decoding gain, in nats per window.

    Each fold fits the two class rates on its training sessions, tunes the amplitude on them, and scores
    the held-out session's windows against a fixed equal prior, with that session's quiet level
    cross-fitted. The per-window gains of all folds are pooled and averaged within each class, then
    across the two, so the statistic is bounded above by ``log 2`` and does not depend on how many quiet
    tiles were drawn. It is sign-free: a unit suppressed around onsets decodes as well as one excited.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per window, peri-onset windows and quiet tiles together.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    session_index (np.ndarray)
        Session id per window.
    event_times (np.ndarray)
        Left edge of each window in seconds.
    folds (list)
        Output of :func:`occurrence_context`.
    axis (dict)
        The ``vocal_decoding.vocal_occurrence`` block.
    crossfit_blocks (int)
        Cross-fitting blocks for a held-out session's quiet level.

    Returns
    -------
    result (dict)
        ``gain``, ``gain_uncalibrated``, ``auroc``, ``response_sign``, ``log_rate_ratio``,
        ``n_folds_positive``, ``min_leave_one_fold_out``, ``n_windows``, ``alphas`` and ``per_fold``.
    """

    per_window = np.full(counts.size, np.nan)
    uncalibrated = np.full(counts.size, np.nan)
    probability = np.full(counts.size, np.nan)
    per_fold, log_ratios = [], []
    for fold in folds:
        alpha = tuned_occurrence_amplitude(counts, labels, session_index, event_times, fold, axis,
                                           crossfit_blocks)
        test_rows = fold["test_rows"]
        gain, fold_probability, rates = score_split(counts, labels, session_index, event_times, fold["train_rows"],
                                                    test_rows, alpha, axis["ridge_frac"],
                                                    axis["mean_count_floor"], crossfit_blocks)
        raw, _probability, _rates = score_split(counts, labels, session_index, event_times, fold["train_rows"],
                                                test_rows, 1.0, axis["ridge_frac"], axis["mean_count_floor"],
                                                crossfit_blocks)
        per_window[test_rows], uncalibrated[test_rows], probability[test_rows] = gain, raw, fold_probability
        log_ratio = float(np.log(rates[1] / rates[0]))
        log_ratios.append(log_ratio)
        per_fold.append({"session": fold["held_out"], "alpha": float(alpha), "rates": rates.tolist(),
                         "log_rate_ratio": log_ratio,
                         "gain": class_balanced_mean(gain, labels[test_rows]),
                         "auroc": area_under_roc(fold_probability, labels[test_rows]),
                         "n_vocal_windows": int((labels[test_rows] > 0.5).sum()),
                         "n_quiet_windows": int((labels[test_rows] < 0.5).sum())})

    scored = np.isfinite(per_window)
    gain = class_balanced_mean(per_window[scored], labels[scored])
    dropped = []
    for fold in folds:
        rest = scored & ~np.isin(np.arange(counts.size), fold["test_rows"])
        dropped.append(class_balanced_mean(per_window[rest], labels[rest]))
    mean_log_ratio = float(np.mean(log_ratios)) if log_ratios else float("nan")
    return {"gain": gain,
            "gain_uncalibrated": class_balanced_mean(uncalibrated[scored], labels[scored]),
            "auroc": area_under_roc(probability[scored], labels[scored]),
            "log_rate_ratio": mean_log_ratio,
            "response_sign": "elevated" if mean_log_ratio >= 0 else "suppressed",
            "n_folds_positive": int(sum(1 for entry in per_fold if entry["gain"] > 0)),
            "min_leave_one_fold_out": float(np.nanmin(dropped)) if np.isfinite(dropped).any() else float("nan"),
            "n_windows": int(counts.size),
            "alphas": [entry["alpha"] for entry in per_fold],
            "per_fold": per_fold}


def vocal_occurrence_shift_null(occurrence_windows: dict, spike_seconds: dict, durations: dict, start: int,
                                n_draws: int, seed: int, guard_seconds: float, folds: list, axis: dict,
                                crossfit_blocks: int) -> np.ndarray:
    """
    Description
    -----------
    Null draws of the vocal-occurrence statistic: circularly shift each session's spike train, recount BOTH
    classes at the same window positions, and refit the decoder.

    Both classes are recounted from the shifted train, so the shift breaks only the alignment of spikes
    to vocal onsets while the unit's bursting and slow rate structure travel with it.
    Each session is shifted independently, by an amount drawn uniformly from ``shift_range_seconds``,
    and the shifted times wrap around the session. The amplitude search and the cross-fitted quiet
    levels are redone in every draw.

    Each draw has its own generator, seeded ``seed + start + draw``, so draws ``[start, start + n_draws)``
    are the same however the range is partitioned across workers.

    Parameters
    ----------
    occurrence_windows (dict)
        The ``occurrence_windows`` block of ``assemble_unit_vocal_events``: ``counts``, ``labels``,
        ``session_index``, ``edges`` (each window's left edge, seconds) and ``width``.
    spike_seconds (dict)
        ``{session slot: sorted spike times in seconds}``, slots as in ``occurrence_windows["session_index"]``.
    durations (dict)
        ``{session slot: session duration in seconds}``, the wrap-around length.
    start (int)
        Global index of the first draw, so a continuation never repeats earlier draws.
    n_draws (int)
        How many draws to make.
    seed (int)
        Base seed.
    guard_seconds (float)
        Minimum shift away from either end of the session.
    folds (list)
        Output of :func:`occurrence_context`.
    axis (dict)
        The ``vocal_decoding.vocal_occurrence`` block.
    crossfit_blocks (int)
        Cross-fitting blocks for a held-out session's quiet level.

    Returns
    -------
    null (np.ndarray)
        ``n_draws`` values of ``gain``.
    """

    values = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        generator = np.random.default_rng(seed + start + draw)
        counts = np.empty_like(occurrence_windows["counts"])
        for slot, train in spike_seconds.items():
            low, high = shift_range_seconds(durations[slot], guard_seconds)
            shifted = np.sort((train + generator.uniform(low, high)) % durations[slot])
            mask = occurrence_windows["session_index"] == slot
            counts[mask] = counts_in_windows(shifted, occurrence_windows["edges"][mask], occurrence_windows["width"])
        values[draw] = vocal_occurrence_statistic(counts, occurrence_windows["labels"],
                                                  occurrence_windows["session_index"], occurrence_windows["edges"],
                                                  folds, axis, crossfit_blocks)["gain"]
    return values
