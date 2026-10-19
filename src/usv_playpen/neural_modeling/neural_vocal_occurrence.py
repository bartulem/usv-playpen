"""
@author: bartulem
Vocal occurrence: does a unit's firing distinguish the window around a vocalization's onset from silence?

Positives are the unit's spike count in a peri-onset window around each clean onset of the recorded
animal's USVs; negatives are same-width counts in quiet tiles, which share the kinematic encoding's
definition of silence and are capped per session upstream. Counts are standardized per session against
the quiet tiles. The statistic is the leave-one-session-out added deviance of a logistic with one shared,
ridge-penalised slope and per-session intercepts: the slope is frozen for the held-out session and only
its intercept is recalibrated. The null circularly shifts each session's spike train and recounts both
classes.

Exports: ``counts_in_windows``, ``quiet_tile_edges``, ``standardize_per_session``,
``vocal_occurrence_statistic`` and ``vocal_occurrence_shift_null``.
"""

from __future__ import annotations

import numpy as np

from .deviance_metrics import (
    area_under_roc,
    bernoulli_deviance,
    calibrate_intercept,
    newton_logistic,
)
from .shift_null_inference import shift_range_seconds


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


def standardize_per_session(counts: np.ndarray, session_index: np.ndarray, labels: np.ndarray,
                            sigma_floor: float) -> tuple[np.ndarray, int]:
    """
    Description
    -----------
    Express counts in SDs of the unit's own QUIET-tile distribution within each session.

    Scaling from the quiet tiles makes the predictor read "SDs above this unit's resting variability in
    this session"; a session with no quiet tiles is scaled on all its windows. ``sigma_floor`` bounds
    the SD from below, so a unit silent during quiet tiles does not produce unbounded values.

    Parameters
    ----------
    counts (np.ndarray)
        Raw window counts, positives then negatives.
    session_index (np.ndarray)
        Session id per window.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    sigma_floor (float)
        Lower bound on the per-session SD.

    Returns
    -------
    standardized (np.ndarray)
        Counts in per-session SD units.
    n_floored (int)
        How many sessions hit the floor.
    """

    values = np.asarray(counts, dtype=np.float64).copy()
    sessions = np.asarray(session_index)
    negative = np.asarray(labels) < 0.5

    n_floored = 0
    for session in np.unique(sessions):
        in_session = sessions == session
        baseline = values[in_session & negative]
        if baseline.size == 0:
            baseline = values[in_session]
        centre = float(np.mean(baseline)) if baseline.size else 0.0
        spread = float(np.std(baseline)) if baseline.size else 0.0
        if spread < sigma_floor:
            spread = sigma_floor
            n_floored += 1
        values[in_session] = (values[in_session] - centre) / spread
    return values, n_floored


def vocal_occurrence_statistic(counts: np.ndarray, labels: np.ndarray, session_index: np.ndarray,
                   ridge_fraction: float, sigma_floor: float, irls_steps: int,
                   calibration_steps: int) -> dict:
    """
    Description
    -----------
    The vocal-occurrence statistic: leave-one-session-out added deviance of a shared-slope transfer logistic.

    Per fold the slope and the training sessions' intercepts are fitted on the other sessions, then the
    slope is FROZEN and only the held-out session's intercept is recalibrated on its own windows. The
    score is the deviance that frozen slope explains over an intercept-only model of the same session,
    summed across folds.

    The statistic is sign-free, so suppressed and elevated units are treated alike; the sign is
    reported separately as a descriptor. The slope ridge scales with the training-set size
    (``ridge_fraction * n_train``) so that it keeps near-silent units from separating at any sample size.

    Parameters
    ----------
    counts (np.ndarray)
        Raw window counts, positives and quiet tiles together.
    labels (np.ndarray)
        1 for a peri-onset window, 0 for a quiet tile.
    session_index (np.ndarray)
        Session id per window.
    ridge_fraction (float)
        Ridge on the slope as a fraction of the training-set size.
    sigma_floor (float)
        Lower bound on the per-session standardizing SD.
    irls_steps (int)
        IRLS iteration cap.
    calibration_steps (int)
        Held-out intercept iteration cap.

    Returns
    -------
    result (dict)
        ``nats_per_window``, ``d2``, ``auroc``, ``slope``, ``response_sign``, ``n_floored_sessions``,
        ``n_windows`` and ``per_fold``.
    """

    standardized, n_floored = standardize_per_session(counts, session_index, labels, sigma_floor)
    sessions = np.unique(session_index)

    added, null_total, slopes, aurocs, per_fold = 0.0, 0.0, [], [], []
    for held_out in sessions:
        train = session_index != held_out
        test = session_index == held_out
        train_sessions = np.unique(session_index[train])

        design = np.zeros((int(train.sum()), 1 + train_sessions.size), dtype=np.float64)
        design[:, 0] = standardized[train]
        for column, session in enumerate(train_sessions):
            design[session_index[train] == session, 1 + column] = 1.0

        ridge = np.zeros(design.shape[1], dtype=np.float64)
        ridge[0] = ridge_fraction * design.shape[0]
        beta = newton_logistic(design, labels[train], ridge, irls_steps)

        slope = float(beta[0])
        offset = slope * standardized[test]
        intercept = calibrate_intercept(offset, labels[test], calibration_steps)
        fitted = bernoulli_deviance(offset + intercept, labels[test])

        flat = calibrate_intercept(np.zeros(int(test.sum())), labels[test], calibration_steps)
        null = bernoulli_deviance(np.full(int(test.sum()), flat), labels[test])

        fold_auroc = area_under_roc(standardized[test], labels[test])
        added += null - fitted
        null_total += null
        slopes.append(slope)
        aurocs.append(fold_auroc)
        per_fold.append({"session": held_out, "slope": slope, "auroc": fold_auroc,
                         "added_deviance": float(null - fitted), "n_windows": int(test.sum())})

    mean_slope = float(np.mean(slopes)) if slopes else float("nan")
    return {"nats_per_window": added / (2.0 * counts.size),
            "d2": added / max(null_total, 1e-12),
            "auroc": float(np.nanmean(aurocs)) if aurocs else float("nan"),
            "slope": mean_slope,
            "response_sign": "elevated" if mean_slope >= 0 else "suppressed",
            "n_floored_sessions": n_floored,
            "n_windows": int(counts.size),
            "per_fold": per_fold}


def vocal_occurrence_shift_null(occurrence_windows: dict, spike_seconds: dict, durations: dict, start: int, n_draws: int,
                    seed: int, guard_seconds: float, ridge_fraction: float, sigma_floor: float,
                    irls_steps: int, calibration_steps: int) -> np.ndarray:
    """
    Description
    -----------
    Null draws of the vocal-occurrence statistic: circularly shift each session's spike train, recount BOTH classes
    at the same window positions, and refit.

    Both classes are recounted from the shifted train, so the shift breaks only the alignment of spikes
    to vocal onsets while the unit's bursting and slow rate structure travel with it.
    Each session is shifted independently, by an amount drawn uniformly from ``shift_range_seconds``,
    and the shifted times wrap around the session.

    Each draw has its own generator, seeded ``seed + start + draw``, so draws ``[start, start + n_draws)``
    are the same however the range is partitioned across workers.

    Parameters
    ----------
    occurrence_windows (dict)
        The ``occurrence_windows`` block of ``assemble_unit_vocal_events``: ``counts``, ``labels``, ``session_index``,
        ``edges`` (each window's left edge, seconds) and ``width``.
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
    ridge_fraction (float)
        As in :func:`vocal_occurrence_statistic`.
    sigma_floor (float)
        As in :func:`vocal_occurrence_statistic`.
    irls_steps (int)
        As in :func:`vocal_occurrence_statistic`.
    calibration_steps (int)
        As in :func:`vocal_occurrence_statistic`.

    Returns
    -------
    null (np.ndarray)
        ``n_draws`` values of ``nats_per_window``.
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
        values[draw] = vocal_occurrence_statistic(counts, occurrence_windows["labels"], occurrence_windows["session_index"], ridge_fraction,
                                      sigma_floor, irls_steps, calibration_steps)["nats_per_window"]
    return values
