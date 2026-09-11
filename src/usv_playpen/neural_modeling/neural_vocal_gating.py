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
