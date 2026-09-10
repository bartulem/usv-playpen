"""
@author: bartulem
The WHEN axis: does a unit's firing predict that a vocalization is imminent?

The statistic is a leave-one-session-out transfer logistic, not a rank test. Positives are the unit's
spike count in a window before each clean focal onset; negatives are the same-width count in QUIET tiles,
using the claim-1 quiet definition so one notion of silence serves the whole project. The model is a
single SHARED slope across sessions with a per-session intercept, and the transfer is what makes it a
test of the relationship rather than of a level: the slope is frozen for the held-out session and only
its intercept is recalibrated, so a unit may sit at a different baseline there but its count-to-onset
relationship must carry over.

Three numerical choices are locked because each was arrived at the hard way.

The slope is SHARED and the intercepts are per-session. A per-session slope quasi-separates and diverges
on small sessions -- one 361-call session sent it to 1e8 and wrecked the pooled score -- while
per-session intercepts are needed to absorb rate drift between sessions.

The predictor is standardized PER SESSION. An intercept shifts the linear predictor but cannot rescale
it, so on raw counts one extra spike means very different evidence in a 10 Hz session than in a 30 Hz
one, and no single shared slope can serve both. Standardizing expresses the count in SDs of the unit's
own distribution within that session -- the count analogue of the per-session exposure offsets the WHAT
axis uses.

Negatives are subsampled without importance weights. Logistic slopes are consistent under
outcome-dependent sampling and the base-rate distortion is absorbed entirely by the intercept; weights
would inflate the effective n and destabilize the fit.

No amplitude calibration is applied here, and that asymmetry with the WHAT axis is deliberate rather
than an omission: because the predictor is already per-session standardized, scale is normalised
upstream of the model. It was tested -- a scalar gain on the frozen slope, tuned by inner LOSO -- and
bought nothing: 4 of 16 units improved, median delta exactly 0.000000, tuned gain 1.0 at the median.
"""

from __future__ import annotations

import numpy as np

from .deviance_metrics import (
    area_under_roc,
    bernoulli_deviance,
    calibrate_intercept,
    newton_logistic,
)


def counts_in_windows(spike_seconds: np.ndarray, left_edges: np.ndarray, width: float) -> np.ndarray:
    """
    Description
    -----------
    Spike count in each half-open window ``[left, left + width)``.

    Uses two searchsorted lookups per window rather than binning to a raster: the windows here are 50 ms
    against sessions of ~20 minutes, so a raster would be almost entirely empty.

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


def quiet_tile_edges(call_starts: np.ndarray, call_stops: np.ndarray, duration: float, width: float,
                     history_pre_seconds: float, clean_post_seconds: float) -> np.ndarray:
    """
    Description
    -----------
    Left edges of non-overlapping quiet tiles, tiled through the gaps between guard-banded calls.

    The guard band is the claim-1 quiet definition -- a call forbids
    ``[start - clean_post, stop + history_pre]`` -- applied to calls from EVERY emitter, so a tile is
    silent of the partner as well as the focal animal. Sharing the definition matters twice over: one
    code path across claim 1, the WHEN negatives and the claim-2 baselines, and a forward guard that
    makes the negative class "no call for seconds" rather than merely "not during a call".

    Tiles are contiguous and non-overlapping within each gap, so no spike is counted twice, and a gap
    shorter than one tile contributes nothing rather than a partial window of a different width.

    Parameters
    ----------
    call_starts (np.ndarray)
        Call onsets in seconds, ALL emitters.
    call_stops (np.ndarray)
        Call offsets in seconds, ALL emitters.
    duration (float)
        Session duration in seconds.
    width (float)
        Tile width, which must equal the prevocal window width or the counts are not comparable.
    history_pre_seconds (float)
        Guard after each call's stop. The same number is the kinematic filter's length, which is why
        it appears here: a quiet window shorter than the predictor history would admit a vocalization
        into the history of a frame called silent.
    clean_post_seconds (float)
        Guard before each call's start.

    Returns
    -------
    edges (np.ndarray)
        Tile start times in seconds.
    """

    starts = np.asarray(call_starts, dtype=np.float64) - clean_post_seconds
    stops = np.asarray(call_stops, dtype=np.float64) + history_pre_seconds
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

    Scaling from the negatives -- the quiet tiles -- rather than from all windows makes the predictor
    read "SDs above this unit's resting variability in this session", and on a held-out session it uses
    only that session's negative-class statistics, which is the same rate-calibration allowance already
    granted by recalibrating the intercept.

    ``sigma_floor`` guards a real failure, not a hypothetical one: a unit that is silent during quiet
    tiles has zero SD there, and the unstandardized ratio produced z-values around 2e10 for exactly the
    most vocal-exclusive units. The floor is calibrated rather than guessed -- across 240 unit-sessions
    the quiet-tile SD has a median far above it, so it binds only in the degenerate case.

    Parameters
    ----------
    counts (np.ndarray)
        Raw window counts, positives then negatives.
    session_index (np.ndarray)
        Session id per window.
    labels (np.ndarray)
        1 for a prevocal window, 0 for a quiet tile.
    sigma_floor (float)
        Lower bound on the per-session SD.

    Returns
    -------
    standardized (np.ndarray)
        Counts in per-session SD units.
    n_floored (int)
        How many sessions hit the floor, recorded so a degenerate unit is visible rather than silent.
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


def when_statistic(counts: np.ndarray, labels: np.ndarray, session_index: np.ndarray,
                   ridge_fraction: float, sigma_floor: float, irls_steps: int,
                   calibration_steps: int) -> dict:
    """
    Description
    -----------
    The WHEN statistic: leave-one-session-out added deviance of a shared-slope transfer logistic.

    Per fold the slope and the training sessions' intercepts are fitted on the other sessions, then the
    slope is FROZEN and only the held-out session's intercept is recalibrated on its own windows. The
    score is the deviance that frozen slope explains over an intercept-only model of the same session,
    summed across folds.

    The statistic is sign-free, so suppressed units compete on equal footing with elevated ones -- about
    a fifth of responders are suppressed, and the sign is reported separately as a descriptor rather than
    being built into the score.

    The ridge scales with the training-set size (``ridge_fraction * n_train``) rather than being a fixed
    small constant: a constant is roughly a thousand times too weak at ~50k rows, and the penalty exists
    to stop separation blowups on near-silent units.

    Parameters
    ----------
    counts (np.ndarray)
        Raw window counts, positives and quiet tiles together.
    labels (np.ndarray)
        1 for a prevocal window, 0 for a quiet tile.
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
        ``per_fold`` -- and ``n_windows``.
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
