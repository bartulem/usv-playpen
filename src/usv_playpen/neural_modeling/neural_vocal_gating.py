"""
@author: bartulem
Is a unit's vocal response GATED on a specific behavioural feature?

The question this exists to answer is one the base taxonomy cannot: a unit that fires to the animal's
own courtship USVs but is SILENT to its solo USVs is socially gated, not vocal-motor, and every
courtship vocalization is socially embedded, so nothing in a courtship-only analysis separates the
two. The calibration unit is `imec0_cl0633_ch278_good`: ~8.8 Hz during its own courtship calls
against 0.05 Hz over 966 male-alone calls. This sub-analysis finds that WITHIN courtship, with no
lone-male control, by asking whether within-courtship variation in a feature scales the response --
solo silence then being the same gate at its limit, with no partner present.

Per feature, contemporaneously, with NO history filter (gating asks whether the CURRENT context
scales the response, which neither needs nor uses claim 1's 600 lags):

    logit P(spike_t) = b0 + beta_V * V(t) + alpha_f * S_f(t) + delta_f * (S_f(t) * V(t))

so the quiet slope is ``alpha_f``, the vocal slope is ``alpha_f + delta_f``, and ``delta_f`` -- the
vocal-specific modulation -- is the gate.

THREE FRAME CATEGORIES, AND THE MODEL USES TWO. Vocal frames are strictly inside a focal USV. Quiet
frames are guard-banded by the claim-1 definition: no USV from any emitter within
`history_pre_seconds` before or `clean_post_seconds` after. **Everything between -- the peri-vocal
zone -- is DROPPED from the universe entirely**, and that is load-bearing rather than tidy: the
peri-vocal bout ramp holds MOST of the firing (on cl0633, 58% of spikes against 35% vocal and 7%
quiet), so defining "non-vocal" as merely "not inside a USV" sweeps those frames into quiet and
MANUFACTURES a feature main effect. That happened, and it cost a day.

EACH EFFECT IS MEASURED IN THE RIGHT FRAME SET AND OVER THE RIGHT BASE. Both errors below happened
and both flatter the result:

- ``alpha_f``, the feature main effect, is fitted AND evaluated on the guard-banded QUIET frames
  ONLY. Read from the joint model on the full universe it is dominated by vocal-frame reshuffling,
  because the alpha column also moves predictions where ~5x more spikes live -- that produced a
  spurious "+219, 21.6 sigma main effect" where the clean quiet-only test gives -38, null, -5 sigma.
- ``beta_V`` is scored over ``[S_f]`` at FULL strength, with the interaction NOT in its base.
  Marginality runs one way: V does not require the interaction, the interaction requires V. Putting
  the interaction in beta_V's base shrank it 410 -> 85 and inflated the delta/beta ratio to a fake 11x.
- ``delta_f`` is scored over ``[S_f + V]``. It is identified purely from vocal frames, since the
  product is identically zero on quiet, so no quiet contamination is possible.

FIVE NUMERICAL REQUIREMENTS, every one of which is a failure that occurred. Held-out DEVIANCE, never
the raw coefficient, which quasi-separates and diverges to 1e5-1e6. Ridge scaling with n
(``ridge_frac * n_train``) -- a fixed small ridge is ~1000x too weak at ~50k rows. NO importance
weights: logistic slopes are consistent under outcome-dependent sampling, and weights inflate the
effective n and destabilise the interaction. LOSO, never a random split, which leaks across
autocorrelation and sessions. And the gating SIGN read from a strongly-ridged (finite) fit while the
MAGNITUDE comes from the deviance.
"""

from __future__ import annotations

import numpy as np

from .deviance_metrics import bernoulli_deviance, calibrate_intercept, newton_logistic
from .neural_design_assembly import (
    emitter_names,
    load_session_usvs,
    session_timebase,
    spike_labels_at_frames,
    vocal_span_frames,
)
from .shift_null_inference import sample_circular_shift, shifted_spike_frames

#: The only value implemented for each option `vocal_gating` names.
#:
#: ``vocal_window`` is ``"during_call"`` -- every frame of the call. The nameable alternative is a
#: fixed ``"peri_onset"`` window, and the plan rejects it: gating asks about VOCALIZING, not
#: peri-vocalizing, so the whole call is the right span even though on the calibration unit it
#: captures only ~35% of the spikes. ``content_gating_compute`` stays false because the content
#: branch (feature x torus position, a 4-df group) is DEFERRED and must be gated on a claim-2 pass --
#: run blind on a non-tuned unit it produces false positives.
IMPLEMENTED_OPTIONS = {
    "vocal_window": ("during_call",),
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


def should_run_gating(claim2_when_significant: bool, claim2_what_significant: bool,
                      n_vocal_spikes: int, feature_iqr_vocal: float, settings: dict) -> tuple:
    """
    Description
    -----------
    Whether this unit gets the gating test, and if not, why not.

    Two independent reasons to skip, and they mean different things -- which is why they are reported
    separately rather than collapsed into one boolean.

    ACTIVATION (``require_claim2_tuning``, user-ruled 2026-09-11): run only on units with WHEN or WHAT
    tuning. A unit with no vocal tuning at all has nothing for a gate to modulate, and the screen is
    19 features x 2,000 shuffles, so skipping them is the saving that keeps gating negligible against
    the rest of the pipeline. Note the two are measured on OVERLAPPING but not identical frames: since
    2026-09-11 claim 2's window straddles onset and covers the call's first 50 ms, while gating spans
    the WHOLE call -- so a unit responding only late in long calls could in principle be skipped. That
    is the accepted cost of the filter.

    IDENTIFIABILITY (``min_vocal_spikes``, ``min_feature_iqr_vocal``): a feature with no spread DURING
    vocal frames leaves ``delta_f`` unidentifiable, and a unit with too few vocal spikes leaves it
    unestimable. Either way the verdict must read ``not_testable`` -- which says "we could not test
    this" -- and NOT ``ns``, which says "we tested it and found nothing". Collapsing the two would
    quietly convert missing power into evidence of absence. Both gates default to null (off), matching
    every other ``min_*`` in this project: the quantities are recorded regardless.

    Parameters
    ----------
    claim2_when_significant, claim2_what_significant (bool)
        The unit's claim-2 verdicts.
    n_vocal_spikes (int)
        Spikes inside focal USVs.
    feature_iqr_vocal (float)
        Interquartile spread of the feature during vocal frames.
    settings (dict)
        The ``vocal_gating`` block.

    Returns
    -------
    run, reason (tuple)
        Whether to run, and ``None`` or one of ``"no_claim2_tuning"`` / ``"not_testable"``.
    """

    minimum_spikes = settings["min_vocal_spikes"]
    minimum_spread = settings["min_feature_iqr_vocal"]
    if minimum_spikes is not None and n_vocal_spikes < minimum_spikes:
        return False, "not_testable"
    if minimum_spread is not None and feature_iqr_vocal < minimum_spread:
        return False, "not_testable"
    if settings["require_claim2_tuning"] and not (claim2_when_significant or claim2_what_significant):
        return False, "no_claim2_tuning"
    return True, None


def loso_added_deviance(full: np.ndarray, base: np.ndarray, spikes: np.ndarray,
                        session_index: np.ndarray, ridge_fraction: float, irls_steps: int,
                        calibration_steps: int) -> dict:
    """
    Description
    -----------
    Held-out added deviance of a FULL design over a nested BASE design, summed across sessions.

    Per fold the slopes of both models are fitted on the training sessions and then FROZEN, and only
    the per-session baseline rate -- the intercept -- is recalibrated on the held-out session. That
    recalibration is what makes the statistic about the RELATIONSHIP rather than the level: a unit may
    sit at a different baseline rate in the held-out session, but its predictor-to-probability slope
    has to carry over unchanged for the effect to count.

    Deviance rather than the coefficient, because the coefficient quasi-separates: on the calibration
    unit the raw interaction term diverged to 1e5-1e6 while the held-out deviance stayed finite and
    interpretable. The ridge scales with the training row count for the same reason -- at ~50k rows a
    fixed small ridge is about a thousand times too weak to restrain it.

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

    Returns
    -------
    result (dict)
        ``added`` (summed held-out deviance reduction, higher means the full model predicts better),
        ``per_fold`` and ``n_folds``.
    """

    sessions = sorted({int(s) for s in session_index})
    if len(sessions) < 2:
        msg = f"gating needs at least two sessions for leave-one-session-out; got {len(sessions)}."
        raise ValueError(msg)

    per_fold, total = [], 0.0
    for held_out in sessions:
        test = np.flatnonzero(session_index == held_out)
        train = np.flatnonzero(session_index != held_out)
        if test.size == 0 or train.size == 0:
            continue
        ridge = ridge_fraction * float(train.size)

        deviances = []
        for design in (base, full):
            if design.shape[1] == 0:
                # an intercept-only base: nothing is frozen, the held-out rate IS the model
                offset = np.zeros(test.size, dtype=np.float64)
            else:
                # The solver fits no intercept of its own, so one is prepended here -- without it the
                # fit is forced through zero and, at a base rate near logit -2, the slopes distort
                # wildly to compensate (it drove the vocal main effect NEGATIVE on a synthetic whose
                # vocal drive was strongly positive). It is left UNPENALISED: ridging an intercept
                # pulls the predicted rate toward one half, which is a bias, not regularisation.
                with_intercept = np.hstack([np.ones((design.shape[0], 1)), design])
                ridge_vector = np.concatenate([[0.0], np.full(design.shape[1], ridge)])
                coefficients = newton_logistic(with_intercept[train], spikes[train], ridge_vector,
                                               irls_steps)
                # Only the SLOPES are carried to the held-out session; its intercept is refitted, so
                # the statistic tests the relationship rather than the level.
                offset = design[test] @ coefficients[1:]
            intercept = calibrate_intercept(offset, spikes[test], calibration_steps)
            deviances.append(bernoulli_deviance(offset + intercept, spikes[test]))

        added = float(deviances[0] - deviances[1])
        per_fold.append({"held_out": held_out, "added": added, "n_test": int(test.size)})
        total += added

    return {"added": total, "per_fold": per_fold, "n_folds": len(per_fold)}


def gating_terms(feature: np.ndarray, vocal: np.ndarray) -> dict:
    """
    Description
    -----------
    The design blocks each of the three effects is scored over, built once so no caller can pair an
    effect with the wrong base.

    The bases are not interchangeable and getting them wrong flatters the result in both directions --
    see the module docstring for the two measured failures. This function exists so the pairing is
    stated once, in code, rather than reconstructed at each call site.

    Parameters
    ----------
    feature (np.ndarray)
        Instantaneous z-scored feature value per row.
    vocal (np.ndarray)
        1.0 inside a focal USV, 0.0 on a guard-banded quiet frame.

    Returns
    -------
    blocks (dict)
        For each of ``feature_main``, ``vocal_main`` and ``interaction``, a ``(full, base)`` pair.
    """

    column = np.asarray(feature, dtype=np.float64)[:, None]
    indicator = np.asarray(vocal, dtype=np.float64)[:, None]
    product = column * indicator
    empty = np.empty((column.shape[0], 0), dtype=np.float64)
    return {
        # scored on QUIET rows only by the caller -- the base is an intercept alone
        "feature_main": (column, empty),
        # V at FULL strength: the interaction is deliberately NOT in this base
        "vocal_main": (np.hstack([column, indicator]), column),
        # the gate: the product over feature AND vocal main effects
        "interaction": (np.hstack([column, indicator, product]), np.hstack([column, indicator])),
    }


def feature_gating_statistic(feature_vocal: np.ndarray, feature_quiet: np.ndarray,
                             vocal: np.ndarray, spikes_universe: np.ndarray,
                             spikes_quiet: np.ndarray, session_universe: np.ndarray,
                             session_quiet: np.ndarray, settings: dict) -> dict:
    """
    Description
    -----------
    The three effects for ONE feature, each on its own frame set and over its own base.

    ``alpha_f`` runs on the QUIET rows alone; ``beta_V`` and ``delta_f`` run on the vocal-plus-quiet
    universe. Keeping the two row sets as separate arguments is deliberate -- it makes it impossible to
    score the feature main effect on the full universe by accident, which is the mistake that produced
    a spurious 21.6 sigma main effect where the correct test gives -5 sigma.

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

    Returns
    -------
    statistic (dict)
        ``feature_main``, ``vocal_main``, ``interaction`` (each an added deviance), their difference
        ``interaction_minus_vocal``, and ``gating_sign``.
    """

    solver = settings["solver"]
    ridge, steps = settings["ridge_frac"], solver["irls_n_steps"]
    calibration = solver["calibration_steps"]

    blocks = gating_terms(feature_vocal, vocal)
    quiet_blocks = gating_terms(feature_quiet, np.zeros_like(feature_quiet))

    main = loso_added_deviance(*quiet_blocks["feature_main"], spikes_quiet, session_quiet,
                               ridge, steps, calibration)
    vocal_main = loso_added_deviance(*blocks["vocal_main"], spikes_universe, session_universe,
                                     ridge, steps, calibration)
    interaction = loso_added_deviance(*blocks["interaction"], spikes_universe, session_universe,
                                      ridge, steps, calibration)

    # The SIGN comes from a strongly-ridged fit and the MAGNITUDE from the deviance: an unridged
    # interaction coefficient quasi-separates and its sign is then read off a diverged number.
    full, _base = blocks["interaction"]
    with_intercept = np.hstack([np.ones((full.shape[0], 1)), full])
    heavy = np.concatenate([[0.0], np.full(full.shape[1],
                                           settings["sign_ridge_frac"] * float(full.shape[0]))])
    coefficients = newton_logistic(with_intercept, spikes_universe, heavy, steps)
    return {"feature_main": main["added"],
            "vocal_main": vocal_main["added"],
            "interaction": interaction["added"],
            "interaction_minus_vocal": interaction["added"] - vocal_main["added"],
            "gating_sign": float(np.sign(coefficients[-1])),
            "per_fold": {"feature_main": main["per_fold"], "vocal_main": vocal_main["per_fold"],
                         "interaction": interaction["per_fold"]}}


def gating_verdict(observed: dict, interaction_null: np.ndarray, difference_null: np.ndarray,
                   main_null: np.ndarray, settings: dict, n_features: int) -> dict:
    """
    Description
    -----------
    The gating label, from three conditions that must ALL hold.

    (i) ``delta_f`` significant: no null draw reaches the observed interaction. At the ruled 2,000
    shuffles that empirical p is 1/2001 = 5.0e-4, which clears the within-unit Bonferroni bar of
    ``0.01 / n_features`` = 5.3e-4 -- and this is why the gating screen overrides the shared 1,000:
    significance here is read PURELY empirically, because at 100 shuffles a GPD extrapolated ~3 sigma
    borderline features into false positives, and an empirical p floors at 1/(n+1).

    (ii) ``alpha_f`` present at an UNCORRECTED p <= 0.01, deliberately generous toward DETECTING a
    main effect, so that the clean ``GATE`` label is reserved for convincingly-absent mains and a unit
    with a main effect is labelled rather than quietly promoted.

    (iii) ``delta_f > beta_V`` by a PAIRED null -- the difference must beat 99% of paired shuffled
    differences, using the same draws for both terms. A raw point comparison is not enough; that is
    what the superseded pilot did. This is the condition that separates a socially-gated unit, whose
    response IS the gated part, from a vocal-motor unit with mild modulation: on the calibration unit
    it isolated exactly 1 of 19 features.

    A significant ``beta_V`` is expected and fine -- the screen runs only on vocal-responsive units.

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
    difference_ok = (p_difference <= 0.01) if settings["require_interaction_gt_vocal"] else True
    main_present = p_main <= 0.01

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

    The per-feature screen IS the within-unit correction, so the bar is ``0.01 / n_features``. An
    empirical p cannot fall below ``1 / (n_shuffles + 1)``, so a shuffle count too small to reach the
    bar would make every feature unpassable no matter how strong -- silently, since nothing else in the
    pipeline would complain.

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

    bar = 0.01 / float(n_features)
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
    The features that passed, with the pairwise correlations that say how much they are one story.

    RULED 2026-09-11 (user): report ALL survivors rather than forward-selecting a minimal set. The
    plan had specified greedy forward selection by LOSO gain, and the argument against it is the same
    one that removed group-lasso from claim 1: under collinearity a greedy search is knife-edge, and
    which of several correlated posture proxies it keeps is unstable rather than meaningful. Since
    this project's standing position is that FEATURE IDENTITY IS DESCRIPTION and not result, a stable
    description beats an unstable selection.

    So the collinearity is reported instead of resolved. A unit gated on four features that correlate
    at 0.9 is one story told four ways; a unit gated on two that correlate at 0.1 is two stories. The
    reader can see which without the pipeline having committed to an answer -- and the correlations
    are measured DURING VOCAL FRAMES, since that is where ``delta_f`` is identified and therefore the
    only place their collinearity can confound it.

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


def gating_universe(unit: dict, data_root: str, per_session: dict, settings: dict,
                    vocal_settings: dict, seed: int = 0) -> dict:
    """
    Description
    -----------
    The two frame sets gating runs on, with the peri-vocal zone DROPPED from both.

    Three categories exist and the model uses two. VOCAL frames sit strictly inside a focal USV.
    QUIET frames are the claim-1 guard-banded anchors -- no USV from any emitter within
    ``history_pre_seconds`` before or ``clean_post_seconds`` after -- which is why this reuses the
    anchors the kinematic assembler already built rather than defining quiet a second way. Everything
    between is PERI-VOCAL and is discarded.

    **That discard is the single most consequential line here.** The peri-vocal bout ramp holds most
    of the firing -- on the calibration unit 58% of spikes, against 35% vocal and 7% quiet -- so
    treating "not inside a USV" as quiet sweeps those frames into the quiet class and manufactures a
    feature main effect out of nothing. It happened, and it cost a day.

    TWO SUBSAMPLES, for two different universes and two different reasons. The vocal-plus-quiet
    universe caps quiet at ``max_quiet_frames_per_session`` so the fit is not swamped by the ~64% of
    the session that is quiet. The quiet-ONLY universe, where ``alpha_f`` is measured, instead keeps
    ``quiet_zeros_per_spike`` zeros per quiet spike, because there the question is a rare-event
    logistic and the class balance is what needs bounding. Neither carries importance weights:
    logistic slopes are consistent under outcome-dependent sampling, and weights inflate the
    effective n and destabilise the interaction.

    Features are INSTANTANEOUS -- the value at the frame, no lags. Gating asks whether the CURRENT
    context scales the response, which neither needs nor uses claim 1's 600-lag history.

    Parameters
    ----------
    unit (dict)
        Cohort record; ``vocal_sessions`` are used, since gating needs calls.
    data_root (str)
        The ``Data`` root.
    per_session (dict)
        From ``assemble_unit_sessions``: feature time series, names, quiet anchors, spike frames.
    settings (dict)
        The ``vocal_gating`` block.
    vocal_settings (dict)
        The ``vocalization_settings`` block, for the emitter.
    seed (int)
        Subsampling seed.

    Returns
    -------
    universe (dict)
        ``features`` / ``vocal`` / ``spikes`` / ``session_index`` / ``frames`` over the
        vocal-plus-quiet rows (the FRAMES are carried so the null can relabel at fixed positions);
        ``features_quiet`` / ``spikes_quiet`` / ``session_quiet`` over the quiet-only rows;
        ``feature_names``; and ``per_session`` counts including how many frames each category
        contributed and how many spikes fell in each.
    """

    check_settings(settings)
    rng = np.random.default_rng(seed)
    cap = settings["max_quiet_frames_per_session"]
    zeros_per_spike = settings["quiet_zeros_per_spike"]

    feature_parts, vocal_parts, spike_parts, index_parts = [], [], [], []
    frame_parts, quiet_frame_parts = [], []
    quiet_feature_parts, quiet_spike_parts, quiet_index_parts = [], [], []
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

        track_names, fps, n_frames = session_timebase(data_root, session_id)
        usv = load_session_usvs(data_root, session_id)
        focal_name = emitter_names(track_names, unit["mouse_id"],
                                   vocal_settings["vocal_emitter"])[0]
        focal = usv.filter(usv["emitter"] == focal_name)
        starts = focal["start"].to_numpy().astype(np.float64)
        stops = focal["stop"].to_numpy().astype(np.float64)
        # n_lags of 1: gating needs no history, so a call is usable wherever it lies in the session.
        # The helper returns (kept mask, flat frames, span pointer) -- the flat frames are the middle
        # element, and gating wants every frame of every span, so the pointer is not needed here.
        _kept, vocal_frames, _pointer = vocal_span_frames(starts, stops, fps, n_frames, 1)
        quiet_frames = np.asarray(session["quiet"], dtype=np.int64)
        spike_frames = np.asarray(session["spike_frames"], dtype=np.int64)

        # PERI-VOCAL is whatever is in neither set; it is never assembled, which is the point.
        vocal_labels = spike_labels_at_frames(spike_frames, vocal_frames, n_frames)
        quiet_labels = spike_labels_at_frames(spike_frames, quiet_frames, n_frames)

        # universe: every vocal frame, quiet capped so the fit is not swamped by silence
        if cap is not None and quiet_frames.size > cap:
            chosen = np.sort(rng.choice(quiet_frames.size, cap, replace=False))
        else:
            chosen = np.arange(quiet_frames.size)
        universe_frames = np.concatenate([vocal_frames, quiet_frames[chosen]])
        universe_vocal = np.concatenate([np.ones(vocal_frames.size), np.zeros(chosen.size)])
        universe_labels = np.concatenate([vocal_labels, quiet_labels[chosen]])

        # quiet-only: a rare-event logistic, so the ZERO count is what needs bounding
        positives = np.flatnonzero(quiet_labels > 0)
        negatives = np.flatnonzero(quiet_labels == 0)
        keep_zeros = min(negatives.size, zeros_per_spike * max(positives.size, 1))
        if negatives.size > keep_zeros:
            negatives = np.sort(rng.choice(negatives, keep_zeros, replace=False))
        quiet_rows = np.sort(np.concatenate([positives, negatives]))

        frame_parts.append(universe_frames)
        quiet_frame_parts.append(quiet_frames[quiet_rows])
        feature_parts.append(series[universe_frames])
        vocal_parts.append(universe_vocal)
        spike_parts.append(universe_labels)
        index_parts.append(np.full(universe_frames.size, slot))
        quiet_feature_parts.append(series[quiet_frames[quiet_rows]])
        quiet_spike_parts.append(quiet_labels[quiet_rows])
        quiet_index_parts.append(np.full(quiet_rows.size, slot))

        bookkeeping[session_id] = {
            "n_vocal_frames": int(vocal_frames.size),
            "n_quiet_frames_available": int(quiet_frames.size),
            "n_quiet_frames_used": int(chosen.size),
            "n_peri_vocal_frames_excluded": int(n_frames - vocal_frames.size - quiet_frames.size),
            "n_vocal_spikes": int(vocal_labels.sum()),
            "n_quiet_spikes": int(quiet_labels.sum()),
            "n_focal_calls": int(starts.size)}

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
            "features_quiet": rows(quiet_feature_parts, width),
            "spikes_quiet": stack(quiet_spike_parts),
            "session_quiet": stack(quiet_index_parts).astype(np.int64),
            "feature_names": feature_names or [],
            "per_session": bookkeeping}


def gating_null(universe: dict, feature_index: int, settings: dict, guard_seconds: float,
                n_draws: int, seed: int = 0) -> dict:
    """
    Description
    -----------
    The circular-shift null for one feature, driving all three statistics from the SAME draws.

    Per draw the spike train of each session is circularly shifted and BOTH frame sets are relabelled
    at their fixed positions -- the features, the vocal indicator and the frame membership never move,
    only which frames carry a spike. The shift preserves the unit's rate, bursting and slow drift, so
    a null train looks like a real one in every respect except its alignment to behaviour and to
    vocalization, which is exactly what the interaction claims to be about.

    **The three statistics are recomputed together, not separately, and the paired difference is what
    makes the delta > beta_V condition a test rather than a comparison.** Scoring the difference
    against its own paired null -- the same shifted train behind both terms on every draw -- is the
    correction the superseded pilot lacked: it compared the two point estimates directly, which says
    nothing about whether the gap could have arisen by chance.

    Significance here is read PURELY empirically. At 100 shuffles a GPD extrapolated ~3 sigma
    borderline features into false positives (tail_curvature, allo_roll), and an empirical p floors at
    1/(n+1), which is why the ruled count is 2,000: it puts the floor at 5.0e-4, just under the
    within-unit Bonferroni bar of 0.01/19 = 5.3e-4.

    Parameters
    ----------
    universe (dict)
        From :func:`gating_universe`.
    feature_index (int)
        Which column of ``features`` to test.
    settings (dict)
        The ``vocal_gating`` block.
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
        ``n_draws`` values, aligned draw by draw so the difference is genuinely paired.
    """

    feature = universe["features"][:, feature_index]
    feature_quiet = universe["features_quiet"][:, feature_index]
    vocal = universe["vocal"]
    frames, frames_quiet = universe["frames"], universe["frames_quiet"]
    index, index_quiet = universe["session_index"], universe["session_quiet"]

    values = {key: np.empty(n_draws, dtype=np.float64)
              for key in ("interaction", "vocal_main", "feature_main", "difference")}
    for draw in range(n_draws):
        rng = np.random.default_rng(seed + draw)
        spikes = np.empty(frames.size, dtype=np.float64)
        spikes_quiet = np.empty(frames_quiet.size, dtype=np.float64)
        for slot, train in universe["spike_frames"].items():
            n_frames, fps = universe["n_frames"][slot], universe["fps"][slot]
            shift = sample_circular_shift(rng, n_frames, fps, guard_seconds)
            shifted = shifted_spike_frames(train, shift, fps, n_frames)
            occupancy = np.zeros(n_frames, dtype=bool)
            occupancy[np.clip(shifted, 0, n_frames - 1)] = True
            mask, mask_quiet = index == slot, index_quiet == slot
            spikes[mask] = occupancy[frames[mask]]
            spikes_quiet[mask_quiet] = occupancy[frames_quiet[mask_quiet]]

        drawn = feature_gating_statistic(feature, feature_quiet, vocal, spikes, spikes_quiet,
                                         index, index_quiet, settings)
        values["interaction"][draw] = drawn["interaction"]
        values["vocal_main"][draw] = drawn["vocal_main"]
        values["feature_main"][draw] = drawn["feature_main"]
        values["difference"][draw] = drawn["interaction_minus_vocal"]
    return values
