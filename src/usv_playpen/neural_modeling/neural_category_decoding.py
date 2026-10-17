"""
@author: bartulem

Does the peri-onset spike count say WHICH acoustic CATEGORY is emitted?

The categorical twin of ``neural_vocal_decoding``. Same events, same window, same leave-one-session-out
folds, same currency -- held-out gain in nats per event over the prior -- and the same exact
within-session permutation null, so a unit's categorical and continuous statistics are directly
comparable and can be drawn against ONE set of permutations.

What differs is the target. Continuous decoding asks where on the torus a call sits and needs a basis
expansion over a grid, an occupancy mask and a kernel prior. A category is one of a handful of discrete
labels, so the encoding model is one Poisson rate per category and the prior is their training
frequency. There is no grid, no mask and no bandwidth, which removes three knobs and the events those
knobs would have discarded.

The amplitude calibration is KEPT, and measurement rather than symmetry is the reason. A fitted rate
profile can carry into a held-out session at the wrong modulation depth exactly as a fitted surface
can, and on 1,231 units it bought 7 points of pass rate at p < 0.05 (29.1% against 21.9%), with each
arm scored against its own null and alpha re-tuned inside every draw. Its grid is floored at 0.1 for
the reason the continuous decoder floors it: allowing zero lets the inner loop switch the neuron off,
and a wider grid downward REDUCES the positive-gain count by overfitting the held-in fold.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

# A session in which the unit never fired would send its exposure to zero, and `0 * log(0)` is NaN
# rather than the limit the maths gives -- with no spikes the count cannot discriminate, so the
# posterior IS the prior and the gain is exactly zero. The continuous decoder floors its own
# per-session offsets at the same value for the same reason; this module lacked the guard, and on the
# first cohort pass it cost 48 of 1,231 units a NaN, every one of them near-silent (median 17 spikes
# across ~1,548 events).
MEAN_COUNT_FLOOR = 1e-3


def decoding_spike_sufficiency(counts: np.ndarray, settings: dict) -> dict:
    """
    Description
    -----------
    Is a unit firing enough across its scored events for a decode to mean anything?

    The permutation null handles a near-silent unit correctly on its own -- its null is correspondingly
    wide, so it does not become significant -- but every analysis that reads the GAIN rather than the
    p-value inherits the instability. Measured over 1,231 units: below 50 spikes the gain's standard
    deviation is 0.029 against 0.014 above it, and a unit with under 25 spikes returned a gain of 0.70,
    more than three times the strongest genuinely tuned unit in the cohort. Those magnitudes come from
    almost no data and contaminate both tails.

    The count is returned whether or not a threshold is set, and a unit below the threshold is FLAGGED
    rather than dropped: its statistic becomes NaN with a stated reason, so the cohort denominator stays
    visible instead of quietly shrinking.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per scored event.
    settings (dict)
        Full neural-modeling settings; reads ``data_sufficiency.min_decoding_spikes``.

    Returns
    -------
    verdict (dict)
        ``n_spikes``, ``threshold``, ``sufficient`` and a ``reason`` when it is not.
    """

    threshold = settings["data_sufficiency"]["min_decoding_spikes"]
    total = int(np.asarray(counts).sum())
    if threshold is None:
        return {"n_spikes": total, "threshold": None, "sufficient": True, "reason": None}
    sufficient = total >= int(threshold)
    return {"n_spikes": total, "threshold": int(threshold), "sufficient": sufficient,
            "reason": None if sufficient
                      else f"{total} spikes across scored events, below min_decoding_spikes={threshold}"}


def category_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Map raw category labels onto contiguous codes, dropping events with no label.

    The labels come off the USV table and need neither be zero-based nor gap-free, while every array
    downstream indexes by code. Doing this once, here, is what stops a missing category silently
    shifting every index after it.

    Parameters
    ----------
    labels (np.ndarray)
        Raw per-event category labels; non-finite entries mark an unlabelled call.

    Returns
    -------
    codes (np.ndarray)
        Contiguous integer code per KEPT event.
    kept (np.ndarray)
        Boolean mask of events carrying a usable label.
    values (np.ndarray)
        The distinct raw labels, in code order, so a result can be reported in the table's own terms.
    """

    raw = np.asarray(labels, dtype=np.float64)
    kept = np.isfinite(raw)
    values = np.unique(raw[kept])
    codes = np.searchsorted(values, raw[kept]).astype(np.int64)
    return codes, kept, values


def category_rates(counts: np.ndarray, codes: np.ndarray, n_categories: int,
                   ridge: float) -> np.ndarray:
    """
    Description
    -----------
    One Poisson rate per category, shrunk toward the unit's pooled rate.

    The ridge scales with the training fold's total spike count, which is the convention the continuous
    decoder uses and for the same reason: the penalty competes against a data term whose Fisher weight
    IS that count, so a normalised rule means the same thing for every unit, window and day. A category
    the training fold never contains falls back to the pooled rate rather than to zero, which would make
    its log-likelihood negative infinity for any event that lands there.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per training event.
    codes (np.ndarray)
        Category code per training event.
    n_categories (int)
        Number of categories.
    ridge (float)
        Shrinkage weight, in units of events.

    Returns
    -------
    rates (np.ndarray)
        ``(n_categories,)`` strictly positive rates.
    """

    pooled = float(counts.mean()) if counts.size else 1e-9
    rates = np.full(n_categories, pooled, dtype=np.float64)
    for code in range(n_categories):
        rows = codes == code
        if rows.any():
            rates[code] = (counts[rows].sum() + ridge * pooled) / (rows.sum() + ridge)
    return np.maximum(rates, 1e-9)


def category_posterior(counts: np.ndarray, rates: np.ndarray, prior: np.ndarray,
                       alpha: float, exposure: float) -> np.ndarray:
    """
    Description
    -----------
    Posterior over categories for each event, from its spike count.

    ``alpha`` scales each category's log-rate DEVIATION from the unit's mean log-rate, so it tempers the
    profile's depth without moving its overall level or its ordering: at 0 every category shares one
    rate and the posterior collapses to the prior, at 1 the training fit is used as is. It can therefore
    decline to use a neuron but can never invert a profile, which is the property that makes the
    calibration one-sided in the same way the continuous decoder's is.

    ``exposure`` carries the held-out session's own mean count, the discrete analogue of the per-session
    rate offsets: it corrects the LEVEL that session drifted to, while alpha corrects the DEPTH.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per scored event.
    rates (np.ndarray)
        Per-category rates from the training fold.
    prior (np.ndarray)
        Category prior from the training fold.
    alpha (float)
        Amplitude calibration.
    exposure (float)
        Held-out level correction.

    Returns
    -------
    posterior (np.ndarray)
        ``(n_events, n_categories)`` log-posterior, normalised per row.
    """

    log_rates = np.log(rates)
    tilted = (np.exp(log_rates.mean() + alpha * (log_rates - log_rates.mean()))
              * max(float(exposure), MEAN_COUNT_FLOOR))
    log_likelihood = (counts[:, None] * np.log(tilted)[None, :] - tilted[None, :]
                      - gammaln(counts + 1.0)[:, None])
    posterior = log_likelihood + np.log(prior)[None, :]
    return posterior - logsumexp(posterior, axis=1, keepdims=True)


def category_context(labels: np.ndarray, session_index: np.ndarray) -> list:
    """
    Description
    -----------
    Precompute the leave-one-session-out folds, with everything that does not depend on spike counts.

    The priors and the inner splits are functions of the labels and the session structure alone, so they
    are identical for the observed fit and for every permutation draw. Building them once is what keeps
    an exact permutation null affordable.

    The prior is built from the fold's TRAINING labels only. A permutation re-pairs counts with event
    identity and nothing else, so the label distribution an event is scored against must not move with
    the draw.

    Parameters
    ----------
    labels (np.ndarray)
        Contiguous category codes, one per event.
    session_index (np.ndarray)
        Session id per event.

    Returns
    -------
    folds (list)
        One dict per held-out session, carrying its row indices, prior and inner splits.
    """

    sessions = np.unique(session_index)
    if sessions.size < 2:
        msg = f"the decoder needs at least two sessions; got {sessions.size}."
        raise ValueError(msg)
    n_categories = int(labels.max()) + 1
    rows_by_session = {int(s): np.flatnonzero(session_index == s) for s in sessions}

    folds = []
    for held_out in sessions:
        train_sessions = [int(s) for s in sessions if s != held_out]
        train_rows = np.concatenate([rows_by_session[s] for s in train_sessions])
        # A category absent from the training fold still needs non-zero prior mass, or an event of that
        # category in the held-out session scores at negative infinity and the fold's mean is destroyed.
        prior = np.maximum(np.bincount(labels[train_rows], minlength=n_categories).astype(np.float64),
                           0.5)
        inner_splits = []
        for inner_held_out in train_sessions:
            rest = [s for s in train_sessions if s != inner_held_out]
            if rest:
                inner_splits.append((np.concatenate([rows_by_session[s] for s in rest]),
                                     rows_by_session[inner_held_out]))
        folds.append({"held_out": int(held_out), "train_sessions": train_sessions,
                      "train_rows": train_rows, "test_rows": rows_by_session[int(held_out)],
                      "prior": prior / prior.sum(), "n_categories": n_categories,
                      "inner_splits": inner_splits})
    return folds


def tuned_category_amplitude(counts: np.ndarray, labels: np.ndarray, fold: dict,
                             settings: dict) -> float:
    """
    Description
    -----------
    Choose the amplitude calibration on the TRAINING sessions, by inner leave-one-session-out.

    No held-out information enters, and the identical search runs inside every permutation draw, so the
    extra fitted parameter cannot buy the observed value an advantage the null does not also get.

    A fold whose training pool is a single session has no inner rotation to search, and the calibration
    then falls back to 1.0 rather than being tuned. That is a real limitation, not a silent one: the
    continuous decoder has the same floor, and on two-session units it is what made the uncalibrated
    profile carry straight through and drove the gain systematically negative.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    labels (np.ndarray)
        Category code per event.
    fold (dict)
        One entry of ``category_context``.
    settings (dict)
        The ``vocal_decoding`` block.

    Returns
    -------
    alpha (float)
        The selected calibration.
    """

    surface = settings["category_surface"]
    grid = [float(a) for a in surface["amplitude_calibration"]["alpha_grid"]]
    if not surface["amplitude_calibration"]["enabled"] or not fold["inner_splits"]:
        return 1.0
    ridge_frac = surface["ridge_selection"]["ridge_frac"]
    best_alpha, best_score = 1.0, -np.inf
    for alpha in grid:
        scores = []
        for train_rows, test_rows in fold["inner_splits"]:
            if train_rows.size < 20 or test_rows.size < 5:
                continue
            prior = np.maximum(np.bincount(labels[train_rows],
                                           minlength=fold["n_categories"]).astype(np.float64), 0.5)
            prior = prior / prior.sum()
            rates = category_rates(counts[train_rows], labels[train_rows], fold["n_categories"],
                                   ridge_frac * counts[train_rows].sum())
            exposure = (max(float(counts[test_rows].mean()), MEAN_COUNT_FLOOR)
                        / max(float(counts[train_rows].mean()), MEAN_COUNT_FLOOR))
            posterior = category_posterior(counts[test_rows], rates, prior, alpha, exposure)
            scores.append(float(np.mean(posterior[np.arange(test_rows.size), labels[test_rows]]
                                        - np.log(prior)[labels[test_rows]])))
        if scores and np.mean(scores) > best_score:
            best_alpha, best_score = alpha, float(np.mean(scores))
    return best_alpha


def category_gain(counts: np.ndarray, labels: np.ndarray, folds: list, settings: dict,
                  permutation: np.ndarray = None, with_detail: bool = False) -> dict:
    """
    Description
    -----------
    The frozen categorical statistic: mean held-out decode gain in nats per event.

    ``permutation`` re-pairs counts with events and is how the exact null is drawn. It permutes the
    COUNTS, leaving labels, priors and folds untouched, so the only thing destroyed is the pairing under
    test. The same permutation array can be handed to the continuous decoder, which is what lets both
    targets share one set of draws and be compared against identical shuffles.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    labels (np.ndarray)
        Category code per event.
    folds (list)
        Output of ``category_context``.
    settings (dict)
        The ``vocal_decoding`` block.
    permutation (np.ndarray)
        Row order applied to the counts; ``None`` scores the observed pairing.
    with_detail (bool)
        Whether to return the per-event companions.

    Returns
    -------
    result (dict)
        The same key roster the continuous decoder returns, so downstream code is target-agnostic.
    """

    surface = settings["category_surface"]
    ridge_frac = surface["ridge_selection"]["ridge_frac"]
    scored = counts if permutation is None else counts[permutation]
    per_event = np.full(counts.size, np.nan)
    entropy = np.full(counts.size, np.nan)
    uncalibrated = np.full(counts.size, np.nan)
    per_fold = []

    for fold in folds:
        train_rows, test_rows = fold["train_rows"], fold["test_rows"]
        if train_rows.size < 20 or test_rows.size < 5:
            continue
        ridge = ridge_frac * float(scored[train_rows].sum())
        alpha = tuned_category_amplitude(scored, labels, fold, settings)
        rates = category_rates(scored[train_rows], labels[train_rows], fold["n_categories"], ridge)
        exposure = (max(float(scored[test_rows].mean()), MEAN_COUNT_FLOOR)
                    / max(float(scored[train_rows].mean()), MEAN_COUNT_FLOOR))
        log_prior = np.log(fold["prior"])
        truth = labels[test_rows]

        posterior = category_posterior(scored[test_rows], rates, fold["prior"], alpha, exposure)
        per_event[test_rows] = posterior[np.arange(truth.size), truth] - log_prior[truth]
        entropy[test_rows] = -np.sum(np.exp(posterior) * posterior, axis=1)
        raw = category_posterior(scored[test_rows], rates, fold["prior"], 1.0, exposure)
        uncalibrated[test_rows] = raw[np.arange(truth.size), truth] - log_prior[truth]

        per_fold.append({"held_out": fold["held_out"], "n_events": int(test_rows.size),
                         "gain": float(np.nanmean(per_event[test_rows])), "alpha": float(alpha),
                         "ridge": float(ridge), "rates": rates.tolist()})

    finite = np.isfinite(per_event)
    gain = float(np.mean(per_event[finite])) if finite.any() else float("nan")
    total_spikes = float(counts.sum())
    dropped = []
    for fold in folds:
        rest = np.setdiff1d(np.arange(counts.size), fold["test_rows"], assume_unique=False)
        vals = per_event[rest][np.isfinite(per_event[rest])]
        dropped.append(float(np.mean(vals)) if vals.size else float("nan"))

    result = {"gain": gain,
              "gain_uncalibrated": (float(np.mean(uncalibrated[np.isfinite(uncalibrated)]))
                                    if np.isfinite(uncalibrated).any() else float("nan")),
              "nats_per_spike": (gain * counts.size / total_spikes if total_spikes > 0
                                 else float("nan")),
              "n_masked": int((~finite).sum()),
              "n_folds_positive": int(sum(1 for f in per_fold if f["gain"] > 0)),
              "min_leave_one_fold_out": float(np.nanmin(dropped)) if dropped else float("nan"),
              "per_fold": per_fold,
              "alphas": [f["alpha"] for f in per_fold],
              "ridges": [f["ridge"] for f in per_fold]}
    if with_detail:
        result.update({"per_event_gain": per_event, "posterior_entropy": entropy,
                       "mean_posterior_entropy": (float(np.mean(entropy[np.isfinite(entropy)]))
                                                  if np.isfinite(entropy).any() else float("nan"))})
    return result
