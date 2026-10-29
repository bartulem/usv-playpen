"""
@author: bartulem
Categorical decoding: does the peri-onset spike count say which acoustic category is emitted?

The categorical version of ``neural_vocal_decoding``. It uses the same events, spike window,
leave-one-session-out folds, gain statistic (held-out nats per event over the prior) and exact
within-session permutation null, so the categorical and continuous statistics are directly
comparable. The encoding model is one Poisson rate per category, shrunk toward the unit's pooled
rate, with the training-fold category frequencies as the prior; the amplitude calibration is tuned
by inner leave-one-session-out as in the continuous decoder.

Exports :func:`category_codes`, :func:`category_context`, :func:`category_gain`,
:func:`category_rates`, :func:`category_posterior`, :func:`tuned_category_amplitude` and
:func:`decoding_spike_sufficiency`.

The nested decoding's categorical form also lives here: :func:`category_behaviour_tilts` turns the
categorical behaviour control's evidence into a per-USV tilt of the category prior, and
:func:`category_nested_gain` is :func:`category_gain` under that behaviour-conditioned prior, so the
two report one statistic under two priors, as the torus forms do.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

from .neural_vocal_decoding import inner_rotation_splits


def decoding_spike_sufficiency(counts: np.ndarray, settings: dict) -> dict:
    """
    Description
    -----------
    Is a unit firing enough across its scored events for a decode to mean anything?

    The permutation null already protects the p-value of a near-silent unit, but its gain is unstable,
    so analyses reading the gain need a floor. The count is returned whether or not a threshold is set,
    and a unit below it is flagged rather than dropped, so the cohort denominator stays visible.

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

    The raw labels need be neither zero-based nor gap-free, while every array downstream indexes by
    code.

    Parameters
    ----------
    labels (np.ndarray)
        Raw per-event category labels; non-finite entries mark an unlabelled USV.

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

    The ridge is passed in as a multiple of the training fold's spike count, as in the continuous
    decoder. A category absent from the training fold falls back to the pooled rate, since a zero rate
    would give its events a log-likelihood of negative infinity.

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
                       alpha: float, exposure: float, mean_count_floor: float) -> np.ndarray:
    """
    Description
    -----------
    Posterior over categories for each event, from its spike count.

    ``alpha`` scales each category's log-rate deviation from the unit's mean log-rate, tempering the
    profile's depth without changing its level or ordering (0 collapses the posterior to the prior, 1
    uses the training fit as is). ``exposure`` rescales to the held-out session's own mean count,
    correcting its level.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per scored event.
    rates (np.ndarray)
        Per-category rates from the training fold.
    prior (np.ndarray)
        Category prior: ``(n_categories,)`` shared by every event, or ``(n_events, n_categories)`` one
        row per event (the behaviour-conditioned prior of the nested decoding).
    alpha (float)
        Amplitude calibration.
    exposure (float)
        Held-out level correction.
    mean_count_floor (float)
        Lower bound on ``exposure`` (``category_surface.mean_count_floor``). A held-out session with no
        spikes would otherwise give ``0 * log(0)`` = NaN, where the correct limit is a posterior equal to
        the prior (zero gain).

    Returns
    -------
    posterior (np.ndarray)
        ``(n_events, n_categories)`` log-posterior, normalised per row.
    """

    log_rates = np.log(rates)
    tilted = (np.exp(log_rates.mean() + alpha * (log_rates - log_rates.mean()))
              * max(float(exposure), mean_count_floor))
    log_likelihood = (counts[:, None] * np.log(tilted)[None, :] - tilted[None, :]
                      - gammaln(counts + 1.0)[:, None])
    log_prior = np.log(prior)
    posterior = log_likelihood + (log_prior[None, :] if log_prior.ndim == 1 else log_prior)
    return posterior - logsumexp(posterior, axis=1, keepdims=True)


def tilted_category_prior(frequencies: np.ndarray, tilt: np.ndarray | None) -> np.ndarray:
    """
    Description
    -----------
    A category prior: the given frequencies, optionally tilted per event by the behaviour evidence.

    With ``tilt`` the prior of event ``i`` is ``frequencies * exp(tilt[i])``, renormalised, which is
    Bayes' rule with ``exp(tilt[i])`` as the behaviour's likelihood of each category; ``tilt = 0`` is the
    frequencies themselves.

    Parameters
    ----------
    frequencies (np.ndarray)
        ``(n_categories,)`` category frequencies, strictly positive, summing to one.
    tilt (np.ndarray | None)
        ``(n_events, n_categories)`` log-evidence per event, or None.

    Returns
    -------
    prior (np.ndarray)
        ``frequencies`` unchanged when ``tilt`` is None, otherwise ``(n_events, n_categories)`` rows
        summing to one.
    """

    if tilt is None:
        return frequencies
    log_prior = np.log(frequencies)[None, :] + tilt
    return np.exp(log_prior - logsumexp(log_prior, axis=1, keepdims=True))


def training_frequencies(labels: np.ndarray, rows: np.ndarray, n_categories: int) -> np.ndarray:
    """
    Description
    -----------
    Category frequencies of the given rows, with a floor of half an event so a category absent from
    them keeps non-zero prior mass.

    Parameters
    ----------
    labels (np.ndarray)
        Category code per event.
    rows (np.ndarray)
        Rows to count.
    n_categories (int)
        Number of categories.

    Returns
    -------
    frequencies (np.ndarray)
        ``(n_categories,)`` frequencies summing to one.
    """

    counts = np.maximum(np.bincount(labels[rows], minlength=n_categories).astype(np.float64), 0.5)
    return counts / counts.sum()


def category_context(labels: np.ndarray, session_index: np.ndarray, event_times: np.ndarray | None = None,
                     single_session_blocks: int = 0) -> list:
    """
    Description
    -----------
    Precompute the leave-one-session-out folds, with everything that does not depend on spike counts.

    The priors (from the fold's training labels only) and the inner splits depend on the labels and
    sessions alone, so they are shared by the observed fit and every permutation draw.

    Parameters
    ----------
    labels (np.ndarray)
        Contiguous category codes, one per event.
    session_index (np.ndarray)
        Session id per event.
    event_times (np.ndarray)
        Onset time of each event in seconds, used to cut a fold's single training session into blocks
        for its inner splits (:func:`neural_vocal_decoding.inner_rotation_splits`). The per-unit
        dispatcher always passes it; ``None`` leaves such a fold without inner splits.
    single_session_blocks (int)
        Blocks a single training session is cut into (``data_sufficiency.single_session_blocks``).

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
        # A category absent from the training fold still needs non-zero prior mass.
        prior = training_frequencies(labels, train_rows, n_categories)
        inner_splits = inner_rotation_splits(train_sessions, rows_by_session, event_times,
                                             single_session_blocks)
        folds.append({"held_out": int(held_out), "train_sessions": train_sessions,
                      "train_rows": train_rows, "test_rows": rows_by_session[int(held_out)],
                      "prior": prior, "n_categories": n_categories,
                      "inner_splits": inner_splits})
    return folds


def tuned_category_amplitude(counts: np.ndarray, labels: np.ndarray, fold: dict,
                             settings: dict, tilt: np.ndarray | None = None) -> float:
    """
    Description
    -----------
    Choose the amplitude calibration on the TRAINING sessions, by the fold's inner splits.

    The same search runs inside every permutation draw, so the extra parameter gives the observed
    value no advantage over the null. The inner splits rotate the training sessions, or contiguous blocks
    of a single training session; a fold with no inner split falls back to 1.0.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    labels (np.ndarray)
        Category code per event.
    fold (dict)
        One entry of ``category_context``.
    settings (dict)
        The ``vocal_decoding`` block; reads ``category_surface`` (ridge, amplitude calibration and the
        ``min_train_events`` / ``min_test_events`` floors below which a split is skipped, and the
        ``mean_count_floor`` under each session's mean count).
    tilt (np.ndarray | None)
        ``(n_events, n_categories)`` behaviour tilt of the prior (:func:`category_behaviour_tilts`),
        applied to each inner split's own frequencies; None for the unconditioned prior.

    Returns
    -------
    alpha (float)
        The selected calibration.
    """

    surface = settings["category_surface"]
    mean_count_floor = surface["mean_count_floor"]
    grid = [float(a) for a in surface["amplitude_calibration"]["alpha_grid"]]
    if not surface["amplitude_calibration"]["enabled"] or not fold["inner_splits"]:
        return 1.0
    ridge_frac = surface["ridge_selection"]["ridge_frac"]
    best_alpha, best_score = 1.0, -np.inf
    for alpha in grid:
        scores = []
        for train_rows, test_rows in fold["inner_splits"]:
            if (train_rows.size < surface["min_train_events"]
                    or test_rows.size < surface["min_test_events"]):
                continue
            prior = tilted_category_prior(training_frequencies(labels, train_rows, fold["n_categories"]),
                                          None if tilt is None else tilt[test_rows])
            rates = category_rates(counts[train_rows], labels[train_rows], fold["n_categories"],
                                   ridge_frac * counts[train_rows].sum())
            exposure = (max(float(counts[test_rows].mean()), mean_count_floor)
                        / max(float(counts[train_rows].mean()), mean_count_floor))
            posterior = category_posterior(counts[test_rows], rates, prior, alpha, exposure,
                                           mean_count_floor)
            prior_at_truth = (prior[labels[test_rows]] if prior.ndim == 1
                              else prior[np.arange(test_rows.size), labels[test_rows]])
            scores.append(float(np.mean(posterior[np.arange(test_rows.size), labels[test_rows]]
                                        - np.log(prior_at_truth))))
        if scores and np.mean(scores) > best_score:
            best_alpha, best_score = alpha, float(np.mean(scores))
    return best_alpha


def category_gain(counts: np.ndarray, labels: np.ndarray, folds: list, settings: dict,
                  permutation: np.ndarray = None, with_detail: bool = False,
                  prior_tilts: list | None = None) -> dict:
    """
    Description
    -----------
    The frozen categorical statistic: mean held-out decode gain in nats per event.

    ``permutation`` reorders the counts, leaving labels, priors and folds untouched, and is how the
    exact null is drawn; the same array can be handed to the continuous decoder so both targets share
    one set of draws.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    labels (np.ndarray)
        Category code per event.
    folds (list)
        Output of ``category_context``.
    settings (dict)
        The ``vocal_decoding`` block; reads ``category_surface`` (ridge, amplitude calibration and the
        ``min_train_events`` / ``min_test_events`` floors below which a split is skipped, and the
        ``mean_count_floor`` under each session's mean count).
    permutation (np.ndarray)
        Row order applied to the counts; ``None`` scores the observed pairing.
    with_detail (bool)
        Whether to return the per-event companions.
    prior_tilts (list | None)
        One ``(n_events, n_categories)`` behaviour tilt per fold (:func:`category_behaviour_tilts`),
        which conditions every prior on the behaviour control, in the outer fold and in its amplitude
        tuning alike. The tilt stays with its event's label, so a permutation of the counts moves only
        the neuron. None scores the unconditioned prior (vocalization identity).

    Returns
    -------
    result (dict)
        The same key roster the continuous decoder returns, so downstream code is target-agnostic.
    """

    surface = settings["category_surface"]
    mean_count_floor = surface["mean_count_floor"]
    ridge_frac = surface["ridge_selection"]["ridge_frac"]
    scored = counts if permutation is None else counts[permutation]
    per_event = np.full(counts.size, np.nan)
    entropy = np.full(counts.size, np.nan)
    uncalibrated = np.full(counts.size, np.nan)
    per_fold = []

    for fold_index, fold in enumerate(folds):
        train_rows, test_rows = fold["train_rows"], fold["test_rows"]
        if train_rows.size < surface["min_train_events"] or test_rows.size < surface["min_test_events"]:
            continue
        tilt = None if prior_tilts is None else np.asarray(prior_tilts[fold_index])
        ridge = ridge_frac * float(scored[train_rows].sum())
        alpha = tuned_category_amplitude(scored, labels, fold, settings, tilt)
        rates = category_rates(scored[train_rows], labels[train_rows], fold["n_categories"], ridge)
        exposure = (max(float(scored[test_rows].mean()), mean_count_floor)
                    / max(float(scored[train_rows].mean()), mean_count_floor))
        truth = labels[test_rows]
        prior = tilted_category_prior(fold["prior"], None if tilt is None else tilt[test_rows])
        log_prior_at_truth = np.log(prior[truth] if prior.ndim == 1
                                    else prior[np.arange(truth.size), truth])

        posterior = category_posterior(scored[test_rows], rates, prior, alpha, exposure,
                                       mean_count_floor)
        per_event[test_rows] = posterior[np.arange(truth.size), truth] - log_prior_at_truth
        entropy[test_rows] = -np.sum(np.exp(posterior) * posterior, axis=1)
        raw = category_posterior(scored[test_rows], rates, prior, 1.0, exposure,
                                 mean_count_floor)
        uncalibrated[test_rows] = raw[np.arange(truth.size), truth] - log_prior_at_truth

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


def category_behaviour_tilts(folds: list, labels: np.ndarray, log_evidence: np.ndarray,
                             kappa_grid: list) -> tuple[list, list]:
    """
    Description
    -----------
    The behaviour control's evidence as a per-USV tilt of each fold's category prior, with one tuned
    concentration per fold.

    The categorical behaviour control gives, for each USV, base-rate-free category probabilities
    ``e_i(c)``. The behaviour-conditioned prior is ``p_i(c) ~ freq(c) * e_i(c) ** kappa``: the
    training frequencies, sharpened by the behaviour's evidence. ``kappa = 0`` is the frequencies
    themselves (the identity decoder's prior), ``kappa = 1`` is Bayes' rule with the control's
    probabilities taken at face value. The control is fitted with a focal loss, so its probabilities
    are not calibrated, and the concentration is tuned rather than fixed at 1.

    The concentration is chosen per outer fold from ``kappa_grid`` by the fold's inner splits, on the
    training sessions alone: for each candidate, the mean over inner held-out events of the log
    conditioned prior at the true category minus the log inner-training frequency at it, each inner
    split using its own training frequencies. The grid is non-negative, so the evidence can be
    declined but never inverted, and ties go to the first (smallest) candidate. A fold with no inner
    split takes the grid's first value.

    Parameters
    ----------
    folds (list)
        From :func:`category_context`.
    labels (np.ndarray)
        Category code per event.
    log_evidence (np.ndarray)
        ``(n_events, n_categories)`` log of the control's base-rate-free probabilities, columns in code
        order.
    kappa_grid (list)
        Candidate concentrations, all ``>= 0`` (``nested_vocal_decoding.behaviour_prior.kappa_grid``).

    Returns
    -------
    kappas, tilts (tuple)
        One concentration per fold, and one ``(n_events, n_categories)`` tilt ``kappa * log_evidence``
        per fold.
    """

    grid = [float(k) for k in kappa_grid]
    if any(k < 0 for k in grid):
        msg = f"kappa_grid must be non-negative; got {grid}."
        raise ValueError(msg)
    log_evidence = np.asarray(log_evidence, dtype=np.float64)
    kappas, tilts = [], []
    for fold in folds:
        kappa = grid[0]
        if fold["inner_splits"]:
            best = -np.inf
            for candidate in grid:
                held_in = []
                for train_rows, test_rows in fold["inner_splits"]:
                    frequencies = training_frequencies(labels, train_rows, fold["n_categories"])
                    prior = tilted_category_prior(frequencies, candidate * log_evidence[test_rows])
                    truth = labels[test_rows]
                    held_in.append(float(np.mean(np.log(prior[np.arange(truth.size), truth])
                                                 - np.log(frequencies[truth]))))
                score = float(np.mean(held_in))
                if score > best:
                    best, kappa = score, candidate
        kappas.append(kappa)
        tilts.append(kappa * log_evidence)
    return kappas, tilts


def category_nested_gain(counts: np.ndarray, labels: np.ndarray, folds: list, settings: dict,
                         tilts: list, kappas: list, permutation: np.ndarray = None,
                         with_detail: bool = False) -> dict:
    """
    Description
    -----------
    The nested statistic for the categorical target: what the neuron adds, in nats per event, once
    behaviour has already set the category prior.

    It is :func:`category_gain` with each fold's prior conditioned on the behaviour control
    (:func:`category_behaviour_tilts`), so ``added`` is log posterior(truth | neuron, behaviour) minus
    log prior(truth | behaviour), averaged over held-out events. Reported beside it:
    ``behaviour_over_no_behaviour`` (log prior(truth | behaviour) minus log frequency(truth), what
    behaviour alone buys) and ``full_over_no_behaviour`` (the sum). A permutation reorders the counts
    only, so the first is invariant under the null and only ``added`` is tested.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event.
    labels (np.ndarray)
        Category code per event.
    folds (list)
        From :func:`category_context`.
    settings (dict)
        The ``vocal_decoding`` block.
    tilts (list)
        From :func:`category_behaviour_tilts`.
    kappas (list)
        The concentration each tilt was built with.
    permutation (np.ndarray)
        Row order applied to the counts; ``None`` scores the observed pairing.
    with_detail (bool)
        Carry the per-event detail through, observed pass only.

    Returns
    -------
    result (dict)
        ``added``, ``added_uncalibrated``, ``nats_per_spike``, ``behaviour_over_no_behaviour``,
        ``full_over_no_behaviour``, ``n_masked``, ``n_folds_positive``, ``min_leave_one_fold_out``,
        ``per_fold`` (each with its ``kappa``), ``alphas``, ``ridges``, ``kappas``, and the detail.
    """

    full = category_gain(counts, labels, folds, settings, permutation, with_detail, prior_tilts=tilts)

    surface = settings["category_surface"]
    behaviour_over = np.full(counts.size, np.nan, dtype=np.float64)
    fold_kappa = {}
    for fold, tilt, kappa in zip(folds, tilts, kappas, strict=True):
        train_rows, test_rows = fold["train_rows"], fold["test_rows"]
        if train_rows.size < surface["min_train_events"] or test_rows.size < surface["min_test_events"]:
            continue
        truth = labels[test_rows]
        prior = tilted_category_prior(fold["prior"], np.asarray(tilt)[test_rows])
        behaviour_over[test_rows] = (np.log(prior[np.arange(truth.size), truth])
                                     - np.log(fold["prior"][truth]))
        fold_kappa[fold["held_out"]] = kappa
    finite = np.isfinite(behaviour_over)
    behaviour_gain = float(np.mean(behaviour_over[finite])) if finite.any() else float("nan")

    per_fold = []
    for fold_result in full["per_fold"]:
        entry = {key: value for key, value in fold_result.items() if key != "gain"}
        entry.update({"added": fold_result["gain"], "kappa": fold_kappa[fold_result["held_out"]]})
        per_fold.append(entry)
    result = {"added": full["gain"], "added_uncalibrated": full["gain_uncalibrated"],
              "nats_per_spike": full["nats_per_spike"], "n_masked": full["n_masked"],
              "n_folds_positive": full["n_folds_positive"],
              "min_leave_one_fold_out": full["min_leave_one_fold_out"],
              "per_fold": per_fold, "alphas": full["alphas"], "ridges": full["ridges"],
              "kappas": [entry["kappa"] for entry in per_fold],
              "behaviour_over_no_behaviour": behaviour_gain,
              "full_over_no_behaviour": full["gain"] + behaviour_gain}
    if with_detail:
        result.update({key: full[key] for key in ("per_event_gain", "posterior_entropy",
                                                  "mean_posterior_entropy")})
    return result
