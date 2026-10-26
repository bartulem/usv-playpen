"""
@author: bartulem
Vocal gating: is a unit's vocal response scaled by a specific behavioural feature?

A unit that fires during the animal's own courtship USVs but not during solo USVs is socially gated
rather than vocal-motor, and a courtship-only analysis cannot separate the two directly. This
sub-analysis asks instead whether within-courtship variation in a feature scales the vocal response.
Per feature, contemporaneously and with no history filter:

    logit P(spike_t) = b0 + beta_V * V(t) + alpha_f * S_f(t) + delta_f * (S_f(t) * V(t))

so the quiet slope is ``alpha_f``, the vocal slope is ``alpha_f + delta_f``, and ``delta_f`` is the
gate.

Frames are vocal (inside a focal USV) or quiet (the kinematic encoding's guard-banded anchors); the
peri-vocal frames in between are excluded. ``alpha_f`` is fitted and scored on quiet frames only;
``beta_V`` is scored over the feature alone, without the interaction in its base; ``delta_f`` is
scored over feature plus vocal indicator. Every effect is a held-out (leave-one-session-out) added
deviance with a ridge proportional to the training row count, fitted on a quiet subsample and scored
on every row; the sign of ``delta_f`` comes from a strongly ridged fit. Significance is an empirical
p against a circular-shift null (:func:`gating_null`), and :func:`gating_unit_pvalue` reduces a unit's
features to one p for the cohort correction.
"""

from __future__ import annotations

import numpy as np

from .deviance_metrics import bernoulli_deviance, calibrate_intercept, newton_logistic
from .neural_design_assembly import (
    emitter_names,
    filter_vocalizations,
    frames_in_window,
    load_session_usvs,
    session_timebase,
    source_session,
    spike_labels_at_frames,
    vocal_span_frames,
)
from .shift_null_inference import sample_circular_shift, shifted_spike_frames

#: The only value implemented for each option `vocal_gating` names. ``vocal_window`` is every frame of
#: the vocalization; ``content_gating_compute`` (feature x torus position) is not implemented.
IMPLEMENTED_OPTIONS = {
    "vocal_window": ("during_vocalization",),
    "content_gating_compute": (False,),
}


def check_settings(settings: dict) -> None:
    """
    Description
    -----------
    Refuse a ``vocal_gating`` block that declares an option this module does not implement.

    Parameters
    ----------
    settings (dict)
        The ``vocal_gating`` block.

    Returns
    -------
    """

    for key, allowed in IMPLEMENTED_OPTIONS.items():
        if settings[key] not in allowed:
            others = ", ".join(repr(a) for a in allowed)
            msg = (f"vocal_gating.{key} = {settings[key]!r} is not implemented; this module supports "
                   f"{others}.")
            raise ValueError(msg)


def should_run_gating(vocal_occurrence_significant: bool, vocalization_identity_significant: bool,
                      n_vocal_spikes: int, feature_iqr_vocal: float, settings: dict) -> tuple:
    """
    Description
    -----------
    Whether this unit gets the gating test, and if not, why not.

    Two reasons to skip are reported separately:

    - ``no_vocal_tuning`` (``require_vocal_tuning``): the unit has neither vocal-occurrence nor
      vocalization-identity tuning, so there is no vocal response for a gate to modulate. Those tests
      use a 100 ms peri-onset window while gating spans the whole USV, so a unit responding only late in
      long USVs can be skipped.
    - ``not_testable`` (``min_vocal_spikes``, ``min_feature_iqr_vocal``): too few vocal spikes, or too
      little feature spread during vocal frames, to identify ``delta_f``. This is kept distinct from
      ``ns`` so missing power is never read as absence. Both thresholds may be null (off).

    Parameters
    ----------
    vocal_occurrence_significant, vocalization_identity_significant (bool)
        Whether the unit's vocal-occurrence and vocalization-identity p-values clear the activation threshold.
    n_vocal_spikes (int)
        Spikes inside focal USVs.
    feature_iqr_vocal (float)
        Interquartile spread of the feature during vocal frames.
    settings (dict)
        The ``vocal_gating`` block.

    Returns
    -------
    run, reason (tuple)
        Whether to run, and ``None`` or one of ``"no_vocal_tuning"`` / ``"not_testable"``.
    """

    minimum_spikes = settings["min_vocal_spikes"]
    minimum_spread = settings["min_feature_iqr_vocal"]
    if minimum_spikes is not None and n_vocal_spikes < minimum_spikes:
        return False, "not_testable"
    if minimum_spread is not None and feature_iqr_vocal < minimum_spread:
        return False, "not_testable"
    if settings["require_vocal_tuning"] and not (vocal_occurrence_significant or vocalization_identity_significant):
        return False, "no_vocal_tuning"
    return True, None


def loso_added_deviance(full: np.ndarray, base: np.ndarray, spikes: np.ndarray,
                        session_index: np.ndarray, ridge_fraction: float, irls_steps: int,
                        calibration_steps: int, fit_rows: np.ndarray = None) -> dict:
    """
    Description
    -----------
    Held-out added deviance of a FULL design over a nested BASE design, summed across sessions.

    Per fold the slopes of both models are fitted on the training sessions and frozen, and only the
    intercept is recalibrated on the held-out session, so the statistic tests the relationship rather
    than the level. Deviance is used rather than the coefficient, which can quasi-separate, and the
    ridge scales with the training row count.

    ``fit_rows`` subsamples the rows the models are fitted on while every row is still scored, and the
    ridge is taken from the full training count rather than the subsample's, so the quiet cap changes
    only how well the statistic is estimated, not what it measures.

    Parameters
    ----------
    full (np.ndarray)
        ``(n, p_full)`` design; the base columns must be a subset, in the same order.
    base (np.ndarray)
        ``(n, p_base)`` nested design; ``p_base`` of 0 means an intercept-only base.
    spikes (np.ndarray)
        Binary spike label per row.
    session_index (np.ndarray)
        Session slot per row, defining the folds.
    ridge_fraction (float)
        Ridge as a fraction of the training row count.
    irls_steps (int)
        Newton steps for the slope fits.
    calibration_steps (int)
        Newton steps for the held-out intercept recalibration.
    fit_rows (np.ndarray)
        Boolean mask of rows the models are fitted on; ``None`` fits on every row. Scoring and the ridge
        always use every row, so this changes only how much data the fit sees.

    Returns
    -------
    result (dict)
        ``added`` (summed held-out deviance reduction, higher means the full model predicts better),
        ``baseline`` (the intercept-only deviance of the same rows, so ``added / baseline`` is the
        fraction of what there was to explain), ``per_fold`` and ``n_folds``.
    """

    sessions = sorted({int(s) for s in session_index})
    if len(sessions) < 2:
        msg = f"gating needs at least two sessions for leave-one-session-out; got {len(sessions)}."
        raise ValueError(msg)

    fitted = np.ones(spikes.size, dtype=bool) if fit_rows is None else np.asarray(fit_rows, dtype=bool)
    per_fold, total, baseline = [], 0.0, 0.0
    for held_out in sessions:
        test = np.flatnonzero(session_index == held_out)
        train = np.flatnonzero((session_index != held_out) & fitted)
        calibration_rows = np.flatnonzero((session_index == held_out) & fitted)
        if test.size == 0 or train.size == 0 or calibration_rows.size == 0:
            continue
        # From the FULL training count, not the subsample's, so the shrinkage does not move with the cap.
        ridge = ridge_fraction * float(np.count_nonzero(session_index != held_out))

        deviances = []
        for design in (base, full):
            if design.shape[1] == 0:
                # an intercept-only base: nothing is frozen, the held-out rate IS the model
                offset = np.zeros(test.size, dtype=np.float64)
            else:
                # The solver fits no intercept of its own, so an unpenalised one is prepended; without
                # it the slopes distort to compensate for the base rate.
                with_intercept = np.hstack([np.ones((design.shape[0], 1)), design])
                ridge_vector = np.concatenate([[0.0], np.full(design.shape[1], ridge)])
                coefficients = newton_logistic(with_intercept[train], spikes[train], ridge_vector,
                                               irls_steps)
                # Only the slopes are carried to the held-out session; its intercept is refitted.
                offset = design[test] @ coefficients[1:]
            # Recalibrated on the held-out session's fit rows, so the scored set's composition cannot
            # decide how much of an effect the intercept absorbs.
            if design.shape[1] == 0:
                calibration_offset = np.zeros(calibration_rows.size, dtype=np.float64)
            else:
                calibration_offset = design[calibration_rows] @ coefficients[1:]
            intercept = calibrate_intercept(calibration_offset, spikes[calibration_rows], calibration_steps)
            deviances.append(bernoulli_deviance(offset + intercept, spikes[test]))

        added = float(deviances[0] - deviances[1])
        per_fold.append({"held_out": held_out, "added": added, "n_test": int(test.size)})
        total += added
        # What there was to explain in this fold: a constant at the held-out rate, calibrated the same
        # way, so a term divided by it is on the same scale as the encoding's D2.
        flat = calibrate_intercept(np.zeros(calibration_rows.size, dtype=np.float64),
                                   spikes[calibration_rows], calibration_steps)
        baseline += bernoulli_deviance(np.full(test.size, flat), spikes[test])

    return {"added": total, "baseline": baseline, "per_fold": per_fold, "n_folds": len(per_fold)}


def gating_terms(feature: np.ndarray, vocal: np.ndarray) -> dict:
    """
    Description
    -----------
    The ``(full, base)`` design pair each effect is scored over, built in one place so no caller can
    pair an effect with the wrong base.

    Parameters
    ----------
    feature (np.ndarray)
        Instantaneous z-scored feature value per row.
    vocal (np.ndarray)
        1.0 inside a focal USV, 0.0 on a guard-banded quiet frame.

    Returns
    -------
    blocks (dict)
        For each of ``feature_main``, ``vocal_main``, ``interaction`` and ``additive``, a
        ``(full, base)`` pair.
    """

    column = np.asarray(feature, dtype=np.float64)[:, None]
    indicator = np.asarray(vocal, dtype=np.float64)[:, None]
    product = column * indicator
    empty = np.empty((column.shape[0], 0), dtype=np.float64)
    return {
        # scored on quiet rows only by the caller; the base is an intercept alone
        "feature_main": (column, empty),
        # V at full strength: the interaction is not in this base
        "vocal_main": (np.hstack([column, indicator]), column),
        # the gate: the product over the feature and vocal main effects
        "interaction": (np.hstack([column, indicator, product]), np.hstack([column, indicator])),
        # the additive model over an intercept alone, the comparison the interaction is weighed against
        "additive": (np.hstack([column, indicator]), empty),
    }


def feature_gating_statistic(feature_vocal: np.ndarray, feature_quiet: np.ndarray,
                             vocal: np.ndarray, spikes_universe: np.ndarray,
                             spikes_quiet: np.ndarray, session_universe: np.ndarray,
                             session_quiet: np.ndarray, settings: dict,
                             fit_rows: np.ndarray = None) -> dict:
    """
    Description
    -----------
    The effects for one feature, each on its own frame set and over its own base.

    ``alpha_f`` runs on the quiet rows alone; ``beta_V`` and ``delta_f`` run on the vocal-plus-quiet
    universe. The two row sets are separate arguments so the feature main effect cannot be scored on
    the full universe by accident.

    Parameters
    ----------
    feature_vocal (np.ndarray)
        Feature value on the universe rows (vocal and quiet together).
    feature_quiet (np.ndarray)
        Feature value on the quiet-only rows.
    vocal (np.ndarray)
        Vocal indicator aligned with the universe rows.
    spikes_universe, spikes_quiet (np.ndarray)
        Binary spike labels for the two row sets.
    session_universe, session_quiet (np.ndarray)
        Session slot for the two row sets.
    settings (dict)
        The ``vocal_gating`` block.
    fit_rows (np.ndarray)
        Boolean mask of universe rows the universe-scored terms are fitted on; every row is scored
        regardless. ``None`` fits on everything.

    Returns
    -------
    statistic (dict)
        ``feature_main``, ``vocal_main``, ``interaction`` and ``additive`` (each an added deviance),
        the difference ``interaction_minus_vocal``, ``gating_sign``, ``baseline`` and ``quiet_baseline``
        (the intercept-only deviances of the universe and quiet rows, which put any term on the scale of
        the kinematic encoding's D2), and ``per_fold``.
    """

    solver = settings["solver"]
    ridge, steps = settings["ridge_frac"], solver["irls_n_steps"]
    calibration = solver["calibration_steps"]

    blocks = gating_terms(feature_vocal, vocal)
    quiet_blocks = gating_terms(feature_quiet, np.zeros_like(feature_quiet))

    main = loso_added_deviance(*quiet_blocks["feature_main"], spikes_quiet, session_quiet,
                               ridge, steps, calibration)
    vocal_main = loso_added_deviance(*blocks["vocal_main"], spikes_universe, session_universe,
                                     ridge, steps, calibration, fit_rows)
    interaction = loso_added_deviance(*blocks["interaction"], spikes_universe, session_universe,
                                      ridge, steps, calibration, fit_rows)
    additive = loso_added_deviance(*blocks["additive"], spikes_universe, session_universe,
                                   ridge, steps, calibration, fit_rows)

    # The sign comes from a strongly ridged fit and the magnitude from the deviance, since an unridged
    # interaction coefficient can quasi-separate.
    full, _base = blocks["interaction"]
    with_intercept = np.hstack([np.ones((full.shape[0], 1)), full])
    heavy = np.concatenate([[0.0], np.full(full.shape[1],
                                           settings["sign_ridge_frac"] * float(full.shape[0]))])
    coefficients = newton_logistic(with_intercept, spikes_universe, heavy, steps)
    return {"feature_main": main["added"],
            "vocal_main": vocal_main["added"],
            "interaction": interaction["added"],
            "additive": additive["added"],
            "baseline": additive["baseline"],
            "quiet_baseline": main["baseline"],
            "interaction_minus_vocal": interaction["added"] - vocal_main["added"],
            "gating_sign": float(np.sign(coefficients[-1])),
            "per_fold": {"feature_main": main["per_fold"], "vocal_main": vocal_main["per_fold"],
                         "interaction": interaction["per_fold"], "additive": additive["per_fold"]}}


def gating_verdict(observed: dict, interaction_null: np.ndarray, difference_null: np.ndarray,
                   main_null: np.ndarray, settings: dict, n_features: int) -> dict:
    """
    Description
    -----------
    The gating label for one feature, from three conditions.

    (i) ``delta_f`` significant: its empirical p clears the within-unit Bonferroni bar
    ``screen_alpha / n_features`` (:func:`settings_bonferroni_bar`).

    (ii) ``delta_f > beta_V`` by a paired null: the difference is scored against shuffled differences
    from the same draws for both terms (``interaction_gt_vocal_alpha``), when
    ``require_interaction_gt_vocal`` is on.

    (iii) ``alpha_f`` present at an uncorrected ``feature_main_alpha``, deliberately generous, so the
    clean ``GATE`` label is reserved for convincingly absent main effects.

    A significant ``beta_V`` is expected, since the screen runs only on vocal-responsive units.

    Parameters
    ----------
    observed (dict)
        From :func:`feature_gating_statistic`.
    interaction_null, difference_null, main_null (np.ndarray)
        Null distributions of the interaction, of ``interaction - vocal_main``, and of the quiet-only
        feature main effect.
    settings (dict)
        The ``vocal_gating`` block.
    n_features (int)
        Features screened, for the within-unit Bonferroni bar.

    Returns
    -------
    verdict (dict)
        ``label`` in {GATE, GATE+silentME, int<vocal, ns}, the three empirical p-values, the sigma
        margin, and a ``borderline`` flag.
    """

    def empirical(null: np.ndarray, value: float) -> float:
        """One-sided upper empirical p with the standard +1, so it can never report zero."""
        null = np.asarray(null, dtype=np.float64)
        return float((1.0 + np.sum(null >= value)) / (1.0 + null.size))

    p_interaction = empirical(interaction_null, observed["interaction"])
    p_difference = empirical(difference_null, observed["interaction_minus_vocal"])
    p_main = empirical(main_null, observed["feature_main"])

    spread = float(np.std(np.asarray(interaction_null, dtype=np.float64)))
    margin = (observed["interaction"] - float(np.mean(interaction_null))) / (spread + 1e-12)

    bonferroni = settings_bonferroni_bar(settings, n_features)
    interaction_ok = p_interaction <= bonferroni
    # Three separate within-unit alphas, none of which is the cohort FDR level `significance.fdr_q`.
    difference_ok = ((p_difference <= settings["interaction_gt_vocal_alpha"])
                     if settings["require_interaction_gt_vocal"] else True)
    main_present = p_main <= settings["feature_main_alpha"]

    if not interaction_ok:
        label = "ns"
    elif not difference_ok:
        label = "int<vocal"
    elif main_present:
        label = "GATE+silentME"
    else:
        label = "GATE"

    return {"label": label,
            "p_interaction": p_interaction,
            "p_interaction_gt_vocal": p_difference,
            "p_feature_main_quiet": p_main,
            "sigma_margin": float(margin),
            "borderline": bool(interaction_ok and margin < settings["borderline_sigma_flag"]),
            "bonferroni_bar": bonferroni}


def settings_bonferroni_bar(settings: dict, n_features: int) -> float:
    """
    Description
    -----------
    The within-unit Bonferroni bar, and a check that the shuffle count can actually reach it.

    The per-feature screen is the within-unit correction, so the bar is ``screen_alpha / n_features``.
    An empirical p cannot fall below ``1 / (n_shuffles + 1)``, so a shuffle count too small to reach the
    bar raises instead of making every feature silently unpassable.

    Parameters
    ----------
    settings (dict)
        The ``vocal_gating`` block.
    n_features (int)
        Features screened.

    Returns
    -------
    bar (float)
        The Bonferroni-corrected alpha the interaction p must clear.
    """

    bar = settings["screen_alpha"] / float(n_features)
    floor = 1.0 / (float(settings["gating_screen_n_shuffles"]) + 1.0)
    if floor > bar:
        msg = (f"gating_screen_n_shuffles = {settings['gating_screen_n_shuffles']} floors the "
               f"empirical p at {floor:.2e}, above the Bonferroni bar {bar:.2e} for {n_features} "
               f"features -- no feature could pass however strong. Raise it to at least "
               f"{int(np.ceil(1.0 / bar))}.")
        raise ValueError(msg)
    return bar


def gating_survivors(per_feature: dict, feature_values_vocal: dict) -> dict:
    """
    Description
    -----------
    The features that passed, with their pairwise correlations.

    All survivors are reported rather than a forward-selected subset, since under collinearity which
    of several correlated features a greedy search keeps is unstable. The correlations are measured
    during vocal frames, where ``delta_f`` is identified.

    Parameters
    ----------
    per_feature (dict)
        ``{feature: verdict dict}`` from :func:`gating_verdict`.
    feature_values_vocal (dict)
        ``{feature: values on VOCAL frames}``, aligned across features.

    Returns
    -------
    summary (dict)
        ``survivors`` (labelled GATE or GATE+silentME, strongest first by sigma margin),
        ``correlations`` (pairwise, among survivors only), and ``max_abs_correlation``.
    """

    survivors = [name for name, verdict in per_feature.items()
                 if verdict["label"] in ("GATE", "GATE+silentME")]
    survivors.sort(key=lambda name: -per_feature[name]["sigma_margin"])

    correlations: dict = {}
    strongest = 0.0
    for i, first in enumerate(survivors):
        for second in survivors[i + 1:]:
            a = np.asarray(feature_values_vocal[first], dtype=np.float64)
            b = np.asarray(feature_values_vocal[second], dtype=np.float64)
            if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
                value = float("nan")
            else:
                value = float(np.corrcoef(a, b)[0, 1])
            correlations[f"{first}|{second}"] = value
            if np.isfinite(value):
                strongest = max(strongest, abs(value))

    return {"survivors": survivors,
            "n_survivors": len(survivors),
            "correlations": correlations,
            "max_abs_correlation": strongest if correlations else float("nan")}


def quiet_subsample_rows(quiet_labels: np.ndarray, zeros_per_spike: int, rng) -> np.ndarray:
    """
    Description
    -----------
    The quiet-only rows the feature main effect is fitted on: every quiet frame carrying a spike, plus
    ``zeros_per_spike`` spike-free quiet frames per spike drawn without replacement (all of them when
    fewer exist; at least ``zeros_per_spike`` when there is no spike at all).

    The same rule is applied to the observed labels and to every null draw's shifted labels, so each
    draw has the observed fit's class balance.

    Parameters
    ----------
    quiet_labels (np.ndarray)
        Binary spike label per quiet frame of one session.
    zeros_per_spike (int)
        Spike-free rows kept per spike (``vocal_gating.quiet_zeros_per_spike``).
    rng (np.random.Generator)
        Generator for the zero draw.

    Returns
    -------
    rows (np.ndarray)
        Sorted indices into ``quiet_labels``.
    """

    positives = np.flatnonzero(quiet_labels > 0)
    negatives = np.flatnonzero(quiet_labels == 0)
    keep_zeros = min(negatives.size, zeros_per_spike * max(positives.size, 1))
    if negatives.size > keep_zeros:
        negatives = np.sort(rng.choice(negatives, keep_zeros, replace=False))
    return np.sort(np.concatenate([positives, negatives]))


def gating_universe(unit: dict, data_root: str, per_session: dict, settings: dict,
                    vocal_settings: dict, seed: int = 0) -> dict:
    """
    Description
    -----------
    The two frame sets gating runs on, with the peri-vocal frames excluded from both.

    Vocal frames sit inside a focal USV. Quiet frames are the kinematic encoding's guard-banded anchors
    (no USV from any emitter within ``history_pre_seconds`` before or ``clean_post_seconds`` after).
    Frames in neither set are peri-vocal and are never assembled, so they cannot enter the quiet class.

    Two subsamples, neither with importance weights: the vocal-plus-quiet universe marks at most
    ``max_quiet_frames_per_session`` quiet frames per session for fitting (every frame is still
    scored), and the quiet-only rows keep ``quiet_zeros_per_spike`` zeros per quiet spike
    (:func:`quiet_subsample_rows`). Features are instantaneous, the value at the frame with no lags.

    Parameters
    ----------
    unit (dict)
        Cohort record; ``vocal_sessions`` are used, since gating needs USVs.
    data_root (str)
        The ``Data`` root.
    per_session (dict)
        From ``assemble_unit_sessions``: feature time series, names, quiet anchors, spike frames and the
        ``window`` each session or block covers; a block's vocal frames are clipped to its window.
    settings (dict)
        The ``vocal_gating`` block.
    vocal_settings (dict)
        The ``vocalization_settings`` block, for the emitter.
    seed (int)
        Subsampling seed.

    Returns
    -------
    universe (dict)
        ``features`` / ``vocal`` / ``spikes`` / ``session_index`` / ``frames`` / ``fit_rows`` over the
        vocal-plus-quiet rows (the frames are carried so the null can relabel at fixed positions);
        ``features_quiet`` / ``spikes_quiet`` / ``session_quiet`` / ``frames_quiet`` over the observed
        quiet-only subsample; ``frames_quiet_all`` / ``features_quiet_all`` / ``session_quiet_all`` over
        every quiet frame, from which the null redraws its own subsample; ``spike_frames``, ``n_frames``
        and ``fps`` per session slot; ``feature_names``; and ``per_session`` counts of frames and spikes
        per category.
    """

    check_settings(settings)
    rng = np.random.default_rng(seed)
    cap = settings["max_quiet_frames_per_session"]
    zeros_per_spike = settings["quiet_zeros_per_spike"]

    feature_parts, vocal_parts, spike_parts, index_parts = [], [], [], []
    frame_parts, quiet_frame_parts, fit_parts = [], [], []
    quiet_feature_parts, quiet_spike_parts, quiet_index_parts = [], [], []
    quiet_all_frame_parts, quiet_all_feature_parts, quiet_all_index_parts = [], [], []
    bookkeeping: dict = {}
    feature_names = None

    for slot, session_id in enumerate(unit["vocal_sessions"]):
        session = per_session[session_id]
        series = np.asarray(session["feature_time_series"], dtype=np.float64)
        if feature_names is None:
            feature_names = list(session["feature_names"])
        elif list(session["feature_names"]) != feature_names:
            msg = (f"{session_id}: feature names differ from the first session. They are "
                   f"canonicalised upstream, so this means the assembler changed under us.")
            raise ValueError(msg)

        # A block loads its source session and keeps the frames in its window.
        track_names, fps, n_frames = session_timebase(data_root, source_session(session_id))
        usv = filter_vocalizations(load_session_usvs(data_root, source_session(session_id)), vocal_settings,
                                   "analysed")
        # 'all' resolves to an empty list, meaning no emitter filter.
        focal_names = emitter_names(track_names, unit["mouse_id"], vocal_settings["vocal_emitter"])
        focal = usv.filter(usv["emitter"].is_in(focal_names)) if focal_names else usv
        starts = focal["start"].to_numpy().astype(np.float64)
        stops = focal["stop"].to_numpy().astype(np.float64)
        # n_lags of 1: gating needs no history. Only the flat frames of every span are used.
        _kept, vocal_frames, _pointer = vocal_span_frames(starts, stops, fps, n_frames, 1)
        vocal_frames = frames_in_window(vocal_frames, session["window"])
        quiet_frames = np.asarray(session["quiet"], dtype=np.int64)
        spike_frames = np.asarray(session["spike_frames"], dtype=np.int64)

        # Peri-vocal frames are in neither set and are never assembled.
        vocal_labels = spike_labels_at_frames(spike_frames, vocal_frames, n_frames)
        quiet_labels = spike_labels_at_frames(spike_frames, quiet_frames, n_frames)

        # The universe keeps every quiet frame; the cap marks which of them the fit may use (see
        # `loso_added_deviance`).
        universe_frames = np.concatenate([vocal_frames, quiet_frames])
        universe_vocal = np.concatenate([np.ones(vocal_frames.size), np.zeros(quiet_frames.size)])
        universe_labels = np.concatenate([vocal_labels, quiet_labels])
        universe_fit = np.ones(universe_frames.size, dtype=bool)
        if cap is not None and quiet_frames.size > cap:
            dropped = rng.choice(quiet_frames.size, quiet_frames.size - cap, replace=False)
            universe_fit[vocal_frames.size + dropped] = False

        # quiet-only: a rare-event logistic, so the zero count is bounded
        quiet_rows = quiet_subsample_rows(quiet_labels, zeros_per_spike, rng)

        frame_parts.append(universe_frames)
        quiet_frame_parts.append(quiet_frames[quiet_rows])
        feature_parts.append(series[universe_frames])
        vocal_parts.append(universe_vocal)
        spike_parts.append(universe_labels)
        index_parts.append(np.full(universe_frames.size, slot))
        fit_parts.append(universe_fit)
        quiet_feature_parts.append(series[quiet_frames[quiet_rows]])
        quiet_spike_parts.append(quiet_labels[quiet_rows])
        quiet_index_parts.append(np.full(quiet_rows.size, slot))
        quiet_all_frame_parts.append(quiet_frames)
        quiet_all_feature_parts.append(series[quiet_frames])
        quiet_all_index_parts.append(np.full(quiet_frames.size, slot))

        bookkeeping[session_id] = {
            "n_vocal_frames": int(vocal_frames.size),
            "n_quiet_frames_available": int(quiet_frames.size),
            "n_quiet_frames_fitted": int(universe_fit.sum() - vocal_frames.size),
            "n_peri_vocal_frames_excluded": int(n_frames - vocal_frames.size - quiet_frames.size),
            "n_vocal_spikes": int(vocal_labels.sum()),
            "n_quiet_spikes": int(quiet_labels.sum()),
            "n_focal_vocalizations": int(starts.size)}

    def stack(parts: list) -> np.ndarray:
        """Concatenate, tolerating a unit that contributed no rows at all."""
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def rows(parts: list, n_columns: int) -> np.ndarray:
        """Stack row blocks, keeping the column count when there are none."""
        return np.vstack(parts) if parts else np.empty((0, n_columns), dtype=np.float64)

    width = len(feature_names) if feature_names else 0
    return {"frames": stack(frame_parts).astype(np.int64),
            "frames_quiet": stack(quiet_frame_parts).astype(np.int64),
            "spike_frames": {slot: np.asarray(per_session[sid]["spike_frames"], dtype=np.int64)
                             for slot, sid in enumerate(unit["vocal_sessions"])},
            "n_frames": {slot: int(per_session[sid]["n_frames"])
                         for slot, sid in enumerate(unit["vocal_sessions"])},
            "fps": {slot: float(per_session[sid]["fps"])
                    for slot, sid in enumerate(unit["vocal_sessions"])},
            "features": rows(feature_parts, width),
            "vocal": stack(vocal_parts),
            "spikes": stack(spike_parts),
            "session_index": stack(index_parts).astype(np.int64),
            "fit_rows": stack(fit_parts).astype(bool),
            "features_quiet": rows(quiet_feature_parts, width),
            "spikes_quiet": stack(quiet_spike_parts),
            "session_quiet": stack(quiet_index_parts).astype(np.int64),
            # every quiet frame, so the null can redraw the quiet-only subsample from shifted labels
            "frames_quiet_all": stack(quiet_all_frame_parts).astype(np.int64),
            "features_quiet_all": rows(quiet_all_feature_parts, width),
            "session_quiet_all": stack(quiet_all_index_parts).astype(np.int64),
            "feature_names": feature_names or [],
            "per_session": bookkeeping}


def gating_null(universe: dict, feature_index: int, settings: dict, guard_seconds: float,
                n_draws: int, seed: int = 0) -> dict:
    """
    Description
    -----------
    The circular-shift null for one feature, driving every statistic from the same draws.

    Per draw each session's spike train is circularly shifted. The vocal-plus-quiet
    universe is relabelled at its fixed positions; the quiet-only rows are redrawn from the shifted
    labels by :func:`quiet_subsample_rows`, as the observed rows were. Because all statistics come from
    the same draw, ``difference`` is a paired null for the ``delta_f > beta_V`` condition.

    Parameters
    ----------
    universe (dict)
        From :func:`gating_universe`.
    feature_index (int)
        Which column of ``features`` to test.
    settings (dict)
        The ``vocal_gating`` block; ``quiet_zeros_per_spike`` sets the quiet-only redraw.
    guard_seconds (float)
        Guard band at each end of the shift range.
    n_draws (int)
        Shuffles.
    seed (int)
        Base seed; each draw advances it.

    Returns
    -------
    null (dict)
        ``interaction``, ``vocal_main``, ``feature_main`` and ``difference`` -- four arrays of
        ``n_draws`` values, aligned draw by draw so the difference is paired.
    """

    feature = universe["features"][:, feature_index]
    feature_quiet_all = universe["features_quiet_all"][:, feature_index]
    vocal = universe["vocal"]
    frames, frames_quiet_all = universe["frames"], universe["frames_quiet_all"]
    index, index_quiet_all = universe["session_index"], universe["session_quiet_all"]
    zeros_per_spike = settings["quiet_zeros_per_spike"]

    values = {key: np.empty(n_draws, dtype=np.float64)
              for key in ("interaction", "vocal_main", "feature_main", "difference")}
    for draw in range(n_draws):
        rng = np.random.default_rng(seed + draw)
        spikes = np.empty(frames.size, dtype=np.float64)
        quiet_feature_parts, quiet_spike_parts, quiet_index_parts = [], [], []
        for slot, train in universe["spike_frames"].items():
            n_frames, fps = universe["n_frames"][slot], universe["fps"][slot]
            shift = sample_circular_shift(rng, n_frames, fps, guard_seconds)
            shifted = shifted_spike_frames(train, shift, fps, n_frames)
            occupancy = np.zeros(n_frames, dtype=bool)
            occupancy[np.clip(shifted, 0, n_frames - 1)] = True
            mask = index == slot
            spikes[mask] = occupancy[frames[mask]]
            # Redraw the quiet-only rows from this draw's shifted labels, by the observed rule.
            session_quiet = np.flatnonzero(index_quiet_all == slot)
            quiet_labels = occupancy[frames_quiet_all[session_quiet]].astype(np.float64)
            chosen = session_quiet[quiet_subsample_rows(quiet_labels, zeros_per_spike, rng)]
            quiet_feature_parts.append(feature_quiet_all[chosen])
            quiet_spike_parts.append(occupancy[frames_quiet_all[chosen]].astype(np.float64))
            quiet_index_parts.append(index_quiet_all[chosen])

        drawn = feature_gating_statistic(feature, np.concatenate(quiet_feature_parts), vocal, spikes,
                                         np.concatenate(quiet_spike_parts), index,
                                         np.concatenate(quiet_index_parts), settings,
                                         universe["fit_rows"])
        values["interaction"][draw] = drawn["interaction"]
        values["vocal_main"][draw] = drawn["vocal_main"]
        values["feature_main"][draw] = drawn["feature_main"]
        values["difference"][draw] = drawn["interaction_minus_vocal"]
    return values


def gating_unit_pvalue(per_feature: dict, settings: dict) -> dict:
    """
    Description
    -----------
    The one p-value per tested unit that the cohort correction uses, and the feature it came from.

    Each feature's p is the conjunction ``max(p_delta, p_{delta > beta_V})``, since both conditions must
    hold (``p_delta`` alone when ``require_interaction_gt_vocal`` is off). The unit p is the within-unit
    Bonferroni adjustment of the best of them:

        p_unit = min(1, n_features * min_f p_f)

    with ``n_features`` the number of features tested for this unit. Every tested unit carries its p,
    survivor or not, so it enters the cohort correction's denominator; NaN marks a unit with no feature
    tested. The reported feature is the one with the smallest conjunction p (ties broken by the larger
    sigma margin). Strongly gated units sit at the shuffle floor and tie; ``at_floor`` flags them.

    Parameters
    ----------
    per_feature (dict)
        ``{feature: verdict dict}`` from :func:`gating_verdict`, for the features TESTED.
    settings (dict)
        The ``vocal_gating`` block.

    Returns
    -------
    result (dict)
        ``p_unit`` (NaN only when no feature was tested), ``feature`` (or None), ``label`` and
        ``at_floor`` (the best conjunction p sits at the shuffle floor ``1 / (n + 1)``).
    """

    if not per_feature:
        return {"p_unit": float("nan"), "feature": None, "label": "not_testable", "at_floor": False}

    def conjunction(verdict: dict) -> float:
        """The feature's conjunction p: the weaker of the two conditions that must both hold."""
        if settings["require_interaction_gt_vocal"]:
            return float(max(verdict["p_interaction"], verdict["p_interaction_gt_vocal"]))
        return float(verdict["p_interaction"])

    best = min(per_feature, key=lambda name: (conjunction(per_feature[name]),
                                              -per_feature[name]["sigma_margin"]))
    best_p = conjunction(per_feature[best])
    floor = 1.0 / (float(settings["gating_screen_n_shuffles"]) + 1.0)
    return {"p_unit": float(min(1.0, len(per_feature) * best_p)),
            "feature": best,
            "label": per_feature[best]["label"],
            "at_floor": bool(np.isclose(best_p, floor))}
