"""
@author: bartulem
The vocal analyses for one unit, in one job: vocal occurrence, vocalization identity, nested vocal-manifold
position decoding, then vocal gating.

Each analysis is a single held-out computation per unit whose cost sits in its null, and they run in
order: occurrence and identity score the same USV-grain event set, the nested decoding scores that set
again with behaviour as the prior, and gating runs only on a unit whose occurrence or identity p-value
shows tuning. Running them in one job also keeps their read-modify-write section writes from
overlapping; the kinematic-encoding combine step for the same unit must not run at the same time.

Each section stores observed statistics, p-values and full nulls, and no decisions. Every null draw is
seeded by its global index, so draws split across spawned worker processes match a serial run.
Workers are spawned rather than forked because the parent has already run JAX.

Exports :func:`run_vocal` and one ``run_*`` function per analysis, plus the command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

from .behaviour_control_model import apply_behaviour_control, load_control_model
from .main_neural_encoding_dispatcher import build_unit_record, load_settings
from .neural_artifacts import read_unit_artifact, write_unit_section
from .neural_design_assembly import (
    assemble_unit_sessions,
    assemble_unit_vocal_events,
    load_unit_spike_frames,
    worker_process_context,
)
from .neural_nested_decoding import (
    behaviour_priors,
    nested_design,
    nested_scores,
    nested_settings,
    no_behaviour_predictions,
    pooled_macro_score,
    posterior_nested_gain,
    reduced_model_features,
    reduced_predictions,
)
from .neural_significance import within_session_permutation
from .neural_vocal_decoding import decode_gain, decoding_context, flagged_descriptors
from .neural_vocal_gating import check_settings as check_gating_settings
from .neural_vocal_gating import (
    feature_gating_statistic,
    gating_null,
    gating_unit_pvalue,
    gating_universe,
    gating_verdict,
    should_run_gating,
)
from .neural_vocal_occurrence import (
    vocal_occurrence_shift_null,
    vocal_occurrence_statistic,
)
from .shift_null_inference import escalated_empirical_pvalue

VOCAL_STEPS = ("vocal_occurrence", "vocalization_identity", "nested_vocal_manifold_position_decoding",
               "vocal_gating")

# State handed to each worker once, by the pool initializer, rather than pickled with every task.
_WORKER: dict = {}


def initialise_worker(state: dict) -> None:
    """
    Description
    -----------
    Install a null's inputs in a freshly spawned worker.

    Parameters
    ----------
    state (dict)
        Everything the worker's chunk function reads.

    Returns
    -------
    None
    """

    _WORKER.clear()
    _WORKER.update(state)


def spawn_pool(n_workers: int, state: dict) -> ProcessPoolExecutor:
    """
    Description
    -----------
    A pool of spawned workers, each initialised with ``state``.

    Spawned, not forked: the parent has already run JAX, and forking a process with live threads can
    deadlock the child.

    Parameters
    ----------
    n_workers (int)
        Worker processes.
    state (dict)
        Inputs installed in every worker; must pickle.

    Returns
    -------
    pool (ProcessPoolExecutor)
        The pool, to be used as a context manager.
    """

    return ProcessPoolExecutor(max_workers=n_workers, mp_context=worker_process_context(),
                               initializer=initialise_worker, initargs=(state,))


def parallel_draw_more(pool: ProcessPoolExecutor, chunk_function, n_workers: int):
    """
    Description
    -----------
    A ``draw_more(n)`` that splits draws across a pool and keeps counting.

    Draws are handed out as contiguous index ranges and reassembled in order; a chunk function seeds
    each draw by its global index, so the values match a serial run however the range is split. The
    counter persists across invocations, so a further request (an escalation step, when the ladder is not
    empty) continues where the last one stopped.

    Parameters
    ----------
    pool (ProcessPoolExecutor)
        An initialised pool.
    chunk_function (Callable)
        ``chunk_function(start, n_draws) -> np.ndarray`` of null values.
    n_workers (int)
        Worker count, used only to size chunks.

    Returns
    -------
    draw_more (Callable)
        ``draw_more(n_draws)`` returning the next ``n_draws`` null values.
    """

    drawn = {"count": 0}

    def draw_more(n_draws: int) -> np.ndarray:
        total, start = int(n_draws), drawn["count"]
        if total <= 0:
            return np.empty(0, dtype=np.float64)
        size = max(1, -(-total // (4 * n_workers)))
        starts = list(range(start, start + total, size))
        sizes = [min(size, start + total - first) for first in starts]
        values = np.concatenate(list(pool.map(chunk_function, starts, sizes)))
        drawn["count"] += total
        return values

    return draw_more


def null_z(observed: float, null: np.ndarray) -> float:
    """
    Description
    -----------
    How many null standard deviations the observed value sits above the null mean; NaN for a flat null.

    Parameters
    ----------
    observed (float)
        The observed statistic.
    null (np.ndarray)
        Its null distribution.

    Returns
    -------
    z (float)
        The standardized distance.
    """

    spread = float(np.std(null))
    return float((observed - float(np.mean(null))) / spread) if spread > 0 else float("nan")


def vocalization_identity_null_chunk(start: int, n_draws: int) -> np.ndarray:
    """
    Description
    -----------
    Vocalization-identity null draws ``[start, start + n_draws)``: the exact within-session position
    permutation, seeded ``seed + draw`` exactly as ``permutation_null`` seeds it.

    Parameters
    ----------
    start (int)
        Global index of the first draw.
    n_draws (int)
        How many draws.

    Returns
    -------
    null (np.ndarray)
        Decode gains under permuted positions.
    """

    state = _WORKER
    values = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        permutation = within_session_permutation(state["session_index"],
                                                 np.random.default_rng(state["seed"] + start + draw))
        values[draw] = decode_gain(state["counts"], state["folds"], state["decoding"], permutation)["gain"]
    return values


def vocal_occurrence_null_chunk(start: int, n_draws: int) -> np.ndarray:
    """
    Description
    -----------
    Vocal-occurrence null draws ``[start, start + n_draws)``; see
    ``neural_vocal_occurrence.vocal_occurrence_shift_null``.

    Parameters
    ----------
    start (int)
        Global index of the first draw.
    n_draws (int)
        How many draws.

    Returns
    -------
    null (np.ndarray)
        ``nats_per_window`` under circularly shifted spike trains.
    """

    state = _WORKER
    return vocal_occurrence_shift_null(state["occurrence_windows"], state["spike_seconds"], state["durations"],
                                       start, n_draws, state["seed"], state["guard_seconds"],
                                       state["ridge_fraction"], state["sigma_floor"], state["irls_steps"],
                                       state["calibration_steps"])


def posterior_nested_null_chunk(start: int, n_draws: int) -> np.ndarray:
    """
    Description
    -----------
    One worker's share of the nested statistic's null: the exact within-session permutation, with the
    behaviour prior travelling with its position.

    A draw re-pairs positions, and the priors built on them, against the counts within each session,
    seeded ``seed + start + draw`` as in the identity chunk; only the neuron's pairing moves.

    Parameters
    ----------
    start (int)
        Global index of this chunk's first draw.
    n_draws (int)
        Draws to produce.

    Returns
    -------
    values (np.ndarray)
        Added scores under permuted pairings.
    """

    state = _WORKER
    values = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        permutation = within_session_permutation(state["session_index"],
                                                 np.random.default_rng(state["seed"] + start + draw))
        values[draw] = posterior_nested_gain(state["counts"], state["folds"], state["decoding"],
                                             state["priors"], state["kappas"], permutation)["added"]
    return values


def posterior_nested_payload(posterior: dict, regression: dict, no_behaviour: float, session_ids: list,
                             null: np.ndarray, p_value: float, at_floor: bool,
                             behaviour_control: dict) -> dict:
    """
    Description
    -----------
    The nested section: the posterior statistic with its null and p-value, the point-regression form
    as a descriptor, and the control's provenance. Metrics and p-values only, no verdict.

    Parameters
    ----------
    posterior (dict)
        From ``posterior_nested_gain`` on the observed pass, with detail.
    regression (dict)
        From ``nested_scores``: the point-regression form, observed only.
    no_behaviour (float)
        The regression ladder's first rung, pooled macro score of the no-behaviour model.
    session_ids (list)
        Session id per slot, to name each fold's held-out session.
    null (np.ndarray)
        The permutation null of ``added``.
    p_value, at_floor (float, bool)
        From the empirical test.
    behaviour_control (dict)
        From ``apply_behaviour_control``: the control's features, held-out day and settings hash.

    Returns
    -------
    payload (dict)
        The section as written.
    """

    per_fold = [{"held_out_session": session_ids[int(fold["held_out"])], "n_events": fold["n_events"],
                 "added": fold["added"], "alpha": fold["alpha"], "ridge": fold["ridge"],
                 "kappa": fold["kappa"], "n_masked": fold["n_masked"],
                 "coefficients": fold["coefficients"]}
                for fold in posterior["per_fold"]]
    detail = {key: posterior[key] for key in ("per_event_gain", "decode_error", "posterior_entropy",
                                               "mean_decode_error", "mean_posterior_entropy",
                                               "overdispersion_index") if key in posterior}
    return {"added": posterior["added"], "added_uncalibrated": posterior["added_uncalibrated"],
            "nats_per_spike": posterior["nats_per_spike"],
            "behaviour_over_no_behaviour": posterior["behaviour_over_no_behaviour"],
            "full_over_no_behaviour": posterior["full_over_no_behaviour"],
            "n_events": int(sum(fold["n_events"] for fold in per_fold)),
            "n_masked": posterior["n_masked"], "n_folds_positive": posterior["n_folds_positive"],
            "min_leave_one_fold_out": posterior["min_leave_one_fold_out"],
            "per_fold": per_fold, "alphas": posterior["alphas"], "ridges": posterior["ridges"],
            "kappas": posterior["kappas"], **detail,
            "null": null, "p": p_value, "at_floor": at_floor, "z": null_z(posterior["added"], null),
            "n_permutations": int(null.size),
            # the point-regression form: a fine-position descriptor, observed only, with no null
            "point_regression": {"added": regression["added"], "reduced": regression["reduced"],
                                 "full": regression["full"], "no_behaviour": no_behaviour,
                                 "behaviour_over_no_behaviour": regression["reduced"] - no_behaviour,
                                 "n_events": regression["n_events"],
                                 "n_regions_scored": regression["n_regions_scored"],
                                 "min_leave_one_fold_out": regression["min_leave_one_fold_out"],
                                 "per_fold": [{"held_out_session": session_ids[int(fold["held_out"])],
                                               "n_events": fold["n_events"], "added": fold["added"]}
                                              for fold in regression["per_fold"]]},
            "behaviour_control": {"features": list(behaviour_control["features"]),
                                  "held_out_day": behaviour_control["held_out_day"],
                                  "settings_sha256": behaviour_control["settings_sha256"]}}


def vocal_gating_feature_job(feature_index: int) -> tuple:
    """
    Description
    -----------
    One feature's gating statistic and its null, in a worker.

    Each feature's draws use seeds ``seed + feature_index * n_draws + draw``, so no two features share
    a shift.

    Parameters
    ----------
    feature_index (int)
        Column of the feature in the universe.

    Returns
    -------
    result (tuple)
        ``(feature_index, observed, null)``, as returned by ``feature_gating_statistic`` and
        ``gating_null``.
    """

    state = _WORKER
    universe, settings = state["universe"], state["settings"]
    observed = feature_gating_statistic(universe["features"][:, feature_index],
                                        universe["features_quiet"][:, feature_index], universe["vocal"],
                                        universe["spikes"], universe["spikes_quiet"],
                                        universe["session_index"], universe["session_quiet"], settings,
                                        universe["fit_rows"])
    null = gating_null(universe, feature_index, settings, state["guard_seconds"], state["n_draws"],
                       seed=state["seed"] + feature_index * state["n_draws"])
    return feature_index, observed, null


def session_spike_seconds(events: dict, data_root: str, unit: dict) -> dict:
    """
    Description
    -----------
    Each vocal session's spike times in seconds, sorted, keyed by the events' session slot.

    Parameters
    ----------
    events (dict)
        Output of ``assemble_unit_vocal_events``.
    data_root (str)
        The ``Data`` root.
    unit (dict)
        The unit record.

    Returns
    -------
    spike_seconds (dict)
        ``{slot: sorted spike times}``.
    """

    return {slot: np.sort(load_unit_spike_frames(data_root, session_id, unit["unit_id"])[0])
            for slot, session_id in enumerate(events["session_ids"])}


def session_durations(events: dict) -> dict:
    """
    Description
    -----------
    Each vocal session's duration in seconds, keyed by the events' session slot: the wrap-around length
    of a circular shift.

    Parameters
    ----------
    events (dict)
        Output of ``assemble_unit_vocal_events``.

    Returns
    -------
    durations (dict)
        ``{slot: seconds}``.
    """

    return {slot: events["per_session"][session_id]["duration_seconds"]
            for slot, session_id in enumerate(events["session_ids"])}


def vocalization_identity_payload(identity: dict, null: np.ndarray, p_value: float, at_floor: bool,
                                  events: dict) -> dict:
    """
    Description
    -----------
    The vocalization-identity section as written: the decoder's own result plus its null and the numbers
    derived from it.

    Parameters
    ----------
    identity (dict)
        ``decode_gain(..., with_detail=True)``, updated with the flagged descriptors.
    null (np.ndarray)
        The permutation null.
    p_value (float)
        Empirical p.
    at_floor (bool)
        Whether ``p_value`` sits at its resolution floor.
    events (dict)
        Output of ``assemble_unit_vocal_events``.

    Returns
    -------
    payload (dict)
        The section.
    """

    exceed = int(np.sum(null >= identity["gain"]))
    return {**identity, "null": null, "p": p_value, "at_floor": at_floor, "n_permutations": int(null.size),
            "exceed_count": exceed, "mid_p": (0.5 + exceed) / (1.0 + null.size),
            "z": null_z(identity["gain"], null), "n_events": int(events["counts"].size),
            "per_session_counts": {session_id: int(events["counts"][events["session_index"] == slot].sum())
                                   for slot, session_id in enumerate(events["session_ids"])}}


def nested_decoding_payload(scores: dict, session_ids: list, null: np.ndarray, p_value: float, at_floor: bool,
                            no_behaviour: float, behaviour_control: dict) -> dict:
    """
    Description
    -----------
    The point-regression nested section. The per-unit run writes :func:`posterior_nested_payload`
    instead; this form is not used by the dispatcher.

    The held-out session is recorded by name rather than by slot.

    Parameters
    ----------
    scores (dict)
        ``nested_scores`` on the observed neural column.
    session_ids (list)
        The events' sessions, indexed by slot.
    null (np.ndarray)
        The null distribution of ``added``.
    p_value (float)
        Empirical p.
    at_floor (bool)
        Whether ``p_value`` sits at its resolution floor.
    no_behaviour (float)
        The pooled score of the no-behaviour model on the same USVs; see ``no_behaviour_predictions``.
    behaviour_control (dict)
        Which cohort control produced the reduced model: its features, the day held out of its fit,
        and the settings hash it was fitted under.

    Returns
    -------
    payload (dict)
        The section.
    """

    return {"added": scores["added"], "reduced": scores["reduced"], "full": scores["full"],
            "n_events": scores["n_events"], "n_regions_scored": scores["n_regions_scored"],
            "min_leave_one_fold_out": scores["min_leave_one_fold_out"],
            "per_fold": [{"held_out_session": session_ids[int(fold["held_out"])],
                          "n_events": fold["n_events"], "added": fold["added"]}
                         for fold in scores["per_fold"]],
            "null": null, "p": p_value, "at_floor": at_floor, "z": null_z(scores["added"], null),
            "n_draws": int(null.size), "no_behaviour": no_behaviour,
            "behaviour_over_no_behaviour": scores["reduced"] - no_behaviour,
            "behaviour_control": {"features": list(behaviour_control["features"]),
                                  "held_out_day": behaviour_control["held_out_day"],
                                  "settings_sha256": behaviour_control["settings_sha256"]}}


def vocal_gating_payload(feature_names: list, observed: dict, nulls: dict, verdicts: dict, unit_p: dict,
                   reasons: dict, n_draws: int) -> dict:
    """
    Description
    -----------
    The gating section as written: statistics, nulls and p-values, and no decisions.

    The per-feature label, the borderline flag and the unit's best label are dropped; the verdicts
    are read only for their p-values and sigma margin. Untested features carry their reason.

    Parameters
    ----------
    feature_names (list)
        Every feature considered.
    observed (dict)
        ``{feature: feature_gating_statistic}`` for the features tested.
    nulls (dict)
        ``{feature: gating_null}`` for the features tested.
    verdicts (dict)
        ``{feature: gating_verdict}`` for the features tested.
    unit_p (dict)
        ``gating_unit_pvalue``.
    reasons (dict)
        ``{feature: reason}`` for the features NOT tested.
    n_draws (int)
        Shuffles per feature.

    Returns
    -------
    payload (dict)
        The section.
    """

    per_feature = {}
    for name in feature_names:
        if name in reasons:
            per_feature[name] = {"tested": False, "reason": reasons[name]}
            continue
        statistic, null, verdict = observed[name], nulls[name], verdicts[name]
        per_feature[name] = {"tested": True,
                             "feature_main": statistic["feature_main"],
                             "vocal_main": statistic["vocal_main"],
                             "interaction": statistic["interaction"],
                             # the additive model and the baseline, so either can be put on the
                             # encoding's D2 scale
                             "additive": statistic["additive"],
                             "baseline": statistic["baseline"],
                             "quiet_baseline": statistic["quiet_baseline"],
                             "interaction_minus_vocal": statistic["interaction_minus_vocal"],
                             "gating_sign": statistic["gating_sign"],
                             "null_interaction": null["interaction"], "null_vocal_main": null["vocal_main"],
                             "null_feature_main": null["feature_main"], "null_difference": null["difference"],
                             "p_interaction": verdict["p_interaction"],
                             "p_interaction_gt_vocal": verdict["p_interaction_gt_vocal"],
                             "p_feature_main_quiet": verdict["p_feature_main_quiet"],
                             "sigma_margin": verdict["sigma_margin"]}
    return {"ran": True, "reason": None, "n_draws": int(n_draws), "per_feature": per_feature,
            "p_unit": unit_p["p_unit"], "p_unit_feature": unit_p["feature"],
            "p_unit_at_floor": unit_p["at_floor"]}


def run_vocal_occurrence(unit: dict, events: dict, settings: dict, data_root: str, output_directory: str,
                         n_workers: int, message_output=print) -> dict:
    """
    Description
    -----------
    Vocal occurrence for one unit: does its count distinguish the window around a USV's onset from
    silence, in a way that transfers across sessions? Written as its own section.

    The statistic is the leave-one-session-out transfer logistic, tested against circular shifts of
    the spike train that recount both classes.

    Parameters
    ----------
    unit (dict)
        The unit record.
    events (dict)
        Output of ``assemble_unit_vocal_events``.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Where the unit's result file lives.
    n_workers (int)
        Worker processes for the null.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    payload (dict)
        The section as written.
    """

    axis = settings["vocal_decoding"]["vocal_occurrence"]
    windows = events["occurrence_windows"]
    occurrence = vocal_occurrence_statistic(windows["counts"], windows["labels"], windows["session_index"],
                                            axis["ridge_frac"], axis["sigma_floor"],
                                            axis["solver"]["irls_n_steps"], axis["solver"]["calibration_steps"])
    state = {"occurrence_windows": windows, "spike_seconds": session_spike_seconds(events, data_root, unit),
             "durations": session_durations(events), "seed": int(settings["null"]["shuffle_seed"]),
             "guard_seconds": settings["null"]["shuffle_guard_seconds"], "ridge_fraction": axis["ridge_frac"],
             "sigma_floor": axis["sigma_floor"], "irls_steps": axis["solver"]["irls_n_steps"],
             "calibration_steps": axis["solver"]["calibration_steps"]}
    with spawn_pool(n_workers, state) as pool:
        draw_more = parallel_draw_more(pool, vocal_occurrence_null_chunk, n_workers)
        null = draw_more(settings["null"]["n_shuffles"])
        p_value, at_floor, null = escalated_empirical_pvalue(occurrence["nats_per_window"], draw_more,
                                                             settings["significance"]["escalation_ladder"],
                                                             null, message_output)
    # Sensitivity beside the discrimination: whether a unit fires weakly on most USVs or strongly on
    # a few, which the averaged statistic and AUROC cannot tell apart.
    vocal_rows = windows["labels"] > 0.5
    occurrence.update({"null": null, "p": p_value, "at_floor": at_floor, "n_shuffles": int(null.size),
                       "z": null_z(occurrence["nats_per_window"], null),
                       "fraction_vocal_windows_with_spike":
                           float(np.mean(windows["counts"][vocal_rows] > 0)),
                       "fraction_quiet_windows_with_spike":
                           float(np.mean(windows["counts"][~vocal_rows] > 0)),
                       "median_spikes_per_vocal_window":
                           float(np.median(windows["counts"][vocal_rows]))})
    message_output(f"  VOCAL OCCURRENCE  nats {occurrence['nats_per_window']:+.6f} "
                   f"| AUROC {occurrence['auroc']:.3f} | z {occurrence['z']:+.1f} | p {p_value:.4e}"
                   f"{' (at floor)' if at_floor else ''} | {null.size:,} draws")
    write_unit_section(output_directory, unit, "vocal_occurrence", occurrence, settings)
    message_output(f"    wrote [vocal_occurrence] for {unit['unit_uid']}")
    return occurrence


def run_vocalization_identity(unit: dict, events: dict, settings: dict, output_directory: str,
                              n_workers: int, message_output=print) -> dict:
    """
    Description
    -----------
    Vocalization identity for one unit: does its count carry information about which USV it is, as a
    position on the vocal manifold? Written as its own section.

    The statistic is the held-out prior-corrected decode gain, tested against the exact within-session
    position permutation.

    Parameters
    ----------
    unit (dict)
        The unit record.
    events (dict)
        Output of ``assemble_unit_vocal_events``.
    settings (dict)
        Full neural-modeling settings.
    output_directory (str)
        Where the unit's result file lives.
    n_workers (int)
        Worker processes for the null.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    payload (dict)
        The section as written.
    """

    decoding = settings["vocal_decoding"]
    period = settings["vocalization_settings"]["torus_period"]
    folds = decoding_context(events["positions"], events["session_index"], decoding, period)
    identity = decode_gain(events["counts"], folds, decoding, with_detail=True)
    state = {"counts": events["counts"], "folds": folds, "decoding": decoding,
             "session_index": events["session_index"], "seed": int(settings["null"]["shuffle_seed"])}
    with spawn_pool(n_workers, state) as pool:
        draw_more = parallel_draw_more(pool, vocalization_identity_null_chunk, n_workers)
        null = draw_more(decoding["discrimination_null"]["n_permutations"])
        p_value, at_floor, null = escalated_empirical_pvalue(identity["gain"], draw_more,
                                                             settings["significance"]["escalation_ladder"],
                                                             null, message_output)
    identity.update(flagged_descriptors(events["counts"], events["positions"], events["session_index"],
                                        decoding, period, identity["gain"], null))
    identity = vocalization_identity_payload(identity, null, p_value, at_floor, events)
    message_output(f"  VOCALIZATION IDENTITY  gain {identity['gain']:+.5f} | z {identity['z']:+.1f} "
                   f"| p {p_value:.4e}{' (at floor)' if at_floor else ''} | {null.size:,} draws")
    write_unit_section(output_directory, unit, "vocalization_identity", identity, settings)
    message_output(f"    wrote [vocalization_identity] for {unit['unit_uid']}")
    return identity


def run_nested_vocal_manifold_position_decoding(unit: dict, events: dict, settings: dict, data_root: str, output_directory: str,
               n_workers: int, message_output=print) -> dict:
    """
    Description
    -----------
    Nested vocal-manifold position decoding for one unit: does the neuron add vocalization-identity
    information beyond the behaviour control?

    The behaviour control's own held-out improvement over a no-behaviour model on the same USVs is
    stored beside the result, since beating the control only matters where it predicts.
    The kinematic data are released before the pool starts.

    Parameters
    ----------
    unit (dict)
        The unit record.
    events (dict)
        Output of ``assemble_unit_vocal_events`` -- the same USVs the vocalization-identity decoding
        scores.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Where the unit's result file lives.
    n_workers (int)
        Worker processes for the null.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    payload (dict)
        The section as written.
    """

    nested = nested_settings(settings)
    encoding = settings["kinematic_encoding"]
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"],
                                         settings["vocalization_settings"]["clean_against"],
                                         settings["vocalization_settings"])
    fps = per_session[unit["courtship_sessions"][0]]["fps"]
    history_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    reduced = reduced_model_features(settings)
    design = nested_design(events, per_session, reduced, history_lags, nested["sigma_floor"])
    # The behaviour control is fitted at cohort scale and enters as its prediction.
    # `n_lags` changes with the block, which is why the swap returns both.
    control = load_control_model(settings)
    design, n_lags = apply_behaviour_control(design, control, str(unit["rec_date"]), settings)
    del per_session

    # The point-regression form: a fine-position descriptor, observed only, with no null.
    reduced_predicted = reduced_predictions(design, nested, n_lags)
    regression = nested_scores(design, nested, n_lags, reduced_predicted=reduced_predicted)
    no_behaviour = pooled_macro_score(no_behaviour_predictions(design, nested, n_lags), design["positions"],
                                      design["region_labels"], nested)

    # The statistic: identity's decoder with behaviour as the prior, on the kept events (those with a
    # full kinematic history).
    decoding = settings["vocal_decoding"]
    counts = np.asarray(events["counts"], dtype=np.float64)[design["kept"]]
    folds = decoding_context(design["positions"], design["session_index"], decoding,
                             settings["vocalization_settings"]["torus_period"])
    kappas, priors = behaviour_priors(folds, design["behaviour_control"]["predicted_positions"],
                                      design["positions"], nested)
    posterior = posterior_nested_gain(counts, folds, decoding, priors, kappas, with_detail=True)
    message_output(f"  NESTED DECODING  added {posterior['added']:+.5f} nats/event over {counts.size} events "
                   f"| behaviour over no behaviour {posterior['behaviour_over_no_behaviour']:+.5f} "
                   f"| point regression {regression['added']:+.5f}")

    state = {"counts": counts, "folds": folds, "decoding": decoding, "priors": priors, "kappas": kappas,
             "session_index": design["session_index"], "seed": int(settings["null"]["shuffle_seed"])}
    with spawn_pool(n_workers, state) as pool:
        draw_more = parallel_draw_more(pool, posterior_nested_null_chunk, n_workers)
        null = draw_more(nested["discrimination_null"]["n_permutations"])
        p_value, at_floor, null = escalated_empirical_pvalue(posterior["added"], draw_more,
                                                             settings["significance"]["escalation_ladder"],
                                                             null, message_output)
    payload = posterior_nested_payload(posterior, regression, no_behaviour, events["session_ids"], null,
                                       p_value, at_floor, design["behaviour_control"])
    message_output(f"    null z {payload['z']:+.1f} | p {p_value:.4e}{' (at floor)' if at_floor else ''} "
                   f"| {null.size:,} draws")
    write_unit_section(output_directory, unit, "nested_vocal_manifold_position_decoding", payload, settings)
    message_output(f"    wrote [nested_vocal_manifold_position_decoding] for {unit['unit_uid']}")
    return payload


def run_vocal_gating(unit: dict, settings: dict, data_root: str, output_directory: str, n_workers: int,
               vocal_occurrence_p: float, vocalization_identity_p: float, message_output=print) -> dict:
    """
    Description
    -----------
    Vocal gating for one unit: every feature's feature x vocalization interaction, with its
    null, written without the per-feature decisions.

    Whether to run is decided first, from the vocal-occurrence and vocalization-identity p-values.
    With both identifiability gates off (the shipped default) an untuned unit is skipped without
    assembling its kinematics; with either gate on, the universe is built first so identifiability
    is checked before tuning. The Bonferroni family for the printed labels and the unit-level p is
    the features actually tested.

    Parameters
    ----------
    unit (dict)
        The unit record.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Where the unit's result file lives.
    n_workers (int)
        Worker processes, one feature each.
    vocal_occurrence_p (float)
        The unit's vocal-occurrence p-value.
    vocalization_identity_p (float)
        The unit's vocalization-identity p-value.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    payload (dict)
        The section as written.
    """

    gating = settings["vocal_gating"]
    check_gating_settings(gating)
    alpha = gating["vocal_tuning_alpha"]
    occurrence_tuned = bool(vocal_occurrence_p < alpha)
    identity_tuned = bool(vocalization_identity_p < alpha)
    gates_off = gating["min_vocal_spikes"] is None and gating["min_feature_iqr_vocal"] is None
    if gates_off:
        run, reason = should_run_gating(occurrence_tuned, identity_tuned, 0, 0.0, gating)
        if not run:
            payload = {"ran": False, "reason": reason}
            write_unit_section(output_directory, unit, "vocal_gating", payload, settings)
            message_output(f"  VOCAL GATING  not run: {reason}")
            return payload

    encoding = settings["kinematic_encoding"]
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"],
                                         settings["vocalization_settings"]["clean_against"],
                                         settings["vocalization_settings"])
    seed = int(settings["null"]["shuffle_seed"])
    universe = gating_universe(unit, data_root, per_session, gating, settings["vocalization_settings"],
                               seed=seed)
    del per_session
    names = list(universe["feature_names"])
    n_vocal_spikes = int(np.sum(universe["spikes"][universe["vocal"] > 0]))
    reasons, tested = {}, []
    for index, name in enumerate(names):
        values = universe["features"][universe["vocal"] > 0, index]
        spread = float(np.subtract(*np.percentile(values, [75, 25]))) if values.size else 0.0
        run, reason = should_run_gating(occurrence_tuned, identity_tuned, n_vocal_spikes, spread, gating)
        if run:
            tested.append(index)
        else:
            reasons[name] = reason
    if not tested:
        unit_reason = "not_testable" if "not_testable" in reasons.values() else "no_vocal_tuning"
        payload = {"ran": False, "reason": unit_reason, "per_feature": {name: {"tested": False,
                                                                                "reason": reasons[name]}
                                                                         for name in names}}
        write_unit_section(output_directory, unit, "vocal_gating", payload, settings)
        message_output(f"  VOCAL GATING  not run: {unit_reason}")
        return payload

    n_draws = int(gating["gating_screen_n_shuffles"])
    state = {"universe": universe, "settings": gating, "seed": seed, "n_draws": n_draws,
             "guard_seconds": settings["null"]["shuffle_guard_seconds"]}
    message_output(f"  VOCAL GATING  {len(tested)} features x {n_draws:,} shuffles")
    observed, nulls, verdicts = {}, {}, {}
    with spawn_pool(n_workers, state) as pool:
        for index, statistic, null in pool.map(vocal_gating_feature_job, tested):
            name = names[index]
            observed[name], nulls[name] = statistic, null
            verdicts[name] = gating_verdict(statistic, null["interaction"], null["difference"],
                                            null["feature_main"], gating, len(tested))
    unit_p = gating_unit_pvalue(verdicts, gating)
    for name in sorted(verdicts, key=lambda feature: -verdicts[feature]["sigma_margin"]):
        verdict = verdicts[name]
        message_output(f"    {name:<24} int {observed[name]['interaction']:+9.1f} "
                       f"vocal {observed[name]['vocal_main']:+9.1f} p_int {verdict['p_interaction']:.1e} "
                       f"p_d>b {verdict['p_interaction_gt_vocal']:.1e} sigma {verdict['sigma_margin']:+6.1f} "
                       f"| {verdict['label']} (uncorrected, not persisted)")
    payload = vocal_gating_payload(names, observed, nulls, verdicts, unit_p, reasons, n_draws)
    write_unit_section(output_directory, unit, "vocal_gating", payload, settings)
    message_output(f"    unit p {unit_p['p_unit']:.4e} from {unit_p['feature']} | wrote [vocal_gating]")
    return payload


def check_tuning_threshold_is_reachable(alpha: float, null_sizes: dict) -> None:
    """
    Description
    -----------
    Refuse a vocal-gating threshold that the p-values it reads can never clear.

    An empirical p-value cannot fall below ``1 / (n_draws + 1)``. If that floor is not below
    ``vocal_gating.vocal_tuning_alpha``, every unit would be skipped as ``no_vocal_tuning``. The fewest
    draws that can clear ``alpha`` is ``floor(1 / alpha)``: 100 at the shipped 0.01.

    Parameters
    ----------
    alpha (float)
        ``vocal_gating.vocal_tuning_alpha``.
    null_sizes (dict)
        ``{section: draws its p-value was computed from}``.

    Returns
    -------
    None
    """

    for section, n_draws in null_sizes.items():
        floor = 1.0 / (float(n_draws) + 1.0)
        if floor >= alpha:
            msg = (f"{section} used {n_draws} draws, so its p-value cannot fall below {floor:.3g}, which never "
                   f"clears vocal_gating.vocal_tuning_alpha = {alpha}: vocal gating would be skipped for every "
                   f"unit. Use at least {int(np.floor(1.0 / alpha))} draws, or raise the threshold.")
            raise ValueError(msg)


def run_vocal(unit: dict, settings: dict, data_root: str, output_directory: str, n_workers: int,
              steps: tuple = VOCAL_STEPS, message_output=print) -> None:
    """
    Description
    -----------
    Run the requested vocal analyses for one unit, in order.

    The USV-grain event set is assembled once and shared by occurrence, identity and the nested
    decoding. Vocal gating uses the occurrence and identity p-values computed in this job, or reads
    them from the unit's result file, and refuses to run if one is missing or was computed from too
    few draws to clear the activation threshold. Categorical decoding is not run here.

    Parameters
    ----------
    unit (dict)
        The unit record.
    settings (dict)
        Full neural-modeling settings.
    data_root (str)
        The ``Data`` root.
    output_directory (str)
        Where the unit's result file lives.
    n_workers (int)
        Worker processes for the nulls.
    steps (tuple)
        Any of ``VOCAL_STEPS``; run in that order regardless of how they are listed.
    message_output (Callable)
        Where progress is reported.

    Returns
    -------
    None
    """

    unknown = sorted(set(steps) - set(VOCAL_STEPS))
    if unknown:
        msg = f"unknown vocal step(s) {unknown}; choose from {list(VOCAL_STEPS)}"
        raise ValueError(msg)
    results: dict = {}
    events = None
    if any(step in steps for step in VOCAL_STEPS[:3]):
        window = settings["vocalization_settings"]["spike_window"]
        events = assemble_unit_vocal_events(unit, data_root, settings, window["pre_offset_seconds"],
                                            window["width_seconds"])
        message_output(f"  {events['counts'].size} events over {len(events['session_ids'])} sessions")
    if "vocal_occurrence" in steps:
        occurrence = run_vocal_occurrence(unit, events, settings, data_root, output_directory, n_workers,
                                          message_output)
        results["vocal_occurrence"] = (occurrence["p"], occurrence["n_shuffles"])
    if "vocalization_identity" in steps:
        identity = run_vocalization_identity(unit, events, settings, output_directory, n_workers, message_output)
        results["vocalization_identity"] = (identity["p"], identity["n_permutations"])
    if "nested_vocal_manifold_position_decoding" in steps:
        run_nested_vocal_manifold_position_decoding(unit, events, settings, data_root, output_directory,
                                                    n_workers, message_output)
    if "vocal_gating" in steps:
        stored = read_unit_artifact(output_directory, unit["unit_uid"])
        for section, draw_key in (("vocal_occurrence", "n_shuffles"), ("vocalization_identity", "n_permutations")):
            if section in results:
                continue
            if section not in stored:
                msg = (f"vocal gating needs {section}'s p-value, and {unit['unit_uid']} has no {section} "
                       f"section in {output_directory}; run {section} first")
                raise ValueError(msg)
            results[section] = (stored[section]["p"], stored[section][draw_key])
        check_tuning_threshold_is_reachable(settings["vocal_gating"]["vocal_tuning_alpha"],
                                            {section: n_draws for section, (_p, n_draws) in results.items()})
        run_vocal_gating(unit, settings, data_root, output_directory, n_workers, results["vocal_occurrence"][0],
                         results["vocalization_identity"][0], message_output)


def dispatch(args: argparse.Namespace) -> int:
    """
    Description
    -----------
    Run one unit's vocal analyses, reporting the full traceback on failure so a pre-empted or I/O-starved
    cluster node leaves something diagnosable behind.

    Parameters
    ----------
    args (argparse.Namespace)
        Parsed command-line arguments.

    Returns
    -------
    status (int)
        Process exit status.
    """

    settings = load_settings(args.settings_path)
    unit = build_unit_record(args.unit_id, args.mouse_id, args.rec_date, args.sessions, args.data_root,
                             settings)
    started = datetime.now()
    try:
        run_vocal(unit, settings, args.data_root, args.output_directory, args.n_workers,
                  tuple(args.only) if args.only else VOCAL_STEPS)
    except Exception:
        traceback.print_exc()
        return 1
    # bare print rather than an injected `message_output`: this is the task's own exit line.
    print(f"finished in {(datetime.now() - started).total_seconds() / 60:.1f} min")
    return 0


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description="Vocal occurrence, vocalization identity, nested vocal-manifold position "
                                                 "decoding and vocal gating for one unit, in one job.")
    parser.add_argument("--unit-id", dest="unit_id", required=True)
    parser.add_argument("--mouse-id", dest="mouse_id", required=True)
    parser.add_argument("--rec-date", dest="rec_date", type=int, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--data-root", dest="data_root", required=True)
    parser.add_argument("--output-directory", dest="output_directory", required=True)
    parser.add_argument("--settings-path", dest="settings_path", default=None)
    parser.add_argument("--n-workers", dest="n_workers", type=int, required=True)
    parser.add_argument("--only", nargs="+", choices=VOCAL_STEPS, default=None)
    return dispatch(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
