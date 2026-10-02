"""
@author: bartulem
The vocal analyses for one unit, in one job: vocal occurrence, vocalization identity, nested vocal-manifold
position decoding, then vocal gating.

The kinematic encoding is a fold ARRAY -- one job per held-out session, combined afterwards -- and has
its own dispatcher. These analyses are not: each is a single held-out computation per unit whose cost
sits in its null, and they must run in ORDER. Vocal occurrence and vocalization identity score the same
call-grain event set, the nested decoding scores that set again with behaviour conditioned on, and vocal
gating runs only on a unit whose occurrence or identity shows tuning, so it needs both p-values first.

ONE JOB PER UNIT, for a second reason. Every analysis writes its section into the unit's single result
file by read-modify-write, so two jobs writing the same unit at once would silently lose one of them.
Running the vocal analyses in one job makes that impossible among them. It is still possible between this
job and the kinematic-encoding combine step for the SAME unit, which must therefore not overlap with it.

STORAGE follows two rulings of 2026-09-14. Only what a figure or the cohort step cannot rebuild
cheaply is kept: observed statistics, p-values and full nulls, and anything needing a refit or a
selection; quantities that rebuild in seconds are left out. And no decisions: vocal gating's per-feature
label, borderline flag and survivor list apply thresholds one unit at a time, so they are uncorrected by
construction. They are computed and printed here for eyes on a single unit, and the artifact writer
refuses them.

NULLS ARE PARALLEL AND EXACT. Every null in the package seeds each draw by its own global index, so draws
can be split across worker processes and come back identical to a serial run, and the escalation ladder
tops a null up without repeating a draw. Workers are SPAWNED rather than forked: the parent has run JAX
before any pool starts, and forking a process with live threads can deadlock the child.

The vocal-occurrence and nested-decoding nulls both shift the unit's spike trains from the same base seed,
so draw ``d`` of one uses the same shifts as draw ``d`` of the other. Each p-value is valid on its own;
the two nulls are simply draw-paired rather than independent.
"""

from __future__ import annotations

import argparse
import multiprocessing
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
)
from .neural_nested_decoding import (
    nested_design,
    nested_scores,
    nested_settings,
    no_behaviour_predictions,
    pooled_macro_score,
    reduced_model_features,
    reduced_predictions,
    shifted_neural_column,
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

# State handed to each worker ONCE, by the pool initializer, rather than pickled with every task: the
# nested decoding's design alone is tens of megabytes.
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
    A pool of SPAWNED workers, each initialised with ``state``.

    Spawned, not forked: the parent has already run JAX, and forking a process with live threads can
    deadlock the child. A spawned worker re-imports the package instead, about a second each.

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

    return ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=initialise_worker, initargs=(state,))


def parallel_draw_more(pool: ProcessPoolExecutor, chunk_function, n_workers: int):
    """
    Description
    -----------
    A ``draw_more(n)`` for the escalation ladder that splits draws across a pool and keeps counting.

    Draws are handed out as contiguous index ranges and reassembled in order, and a chunk function seeds
    each draw by its GLOBAL index -- so the values are identical to a serial run however the range is
    partitioned. The counter persists across calls, so each escalation continues from where the last
    stopped instead of repeating draws already taken. About four chunks per worker is a scheduling
    choice only; it cannot change a value.

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


def nested_decoding_null_chunk(start: int, n_draws: int) -> np.ndarray:
    """
    Description
    -----------
    Nested vocal-manifold position decoding null draws ``[start, start + n_draws)``, seeded ``seed + draw``
    exactly as the package's serial ``null_draw_factory`` seeds them, so the two agree value for value.

    Only the neural column changes between draws; the behaviour block and the reduced model's held-out
    predictions are fixed and reused.

    Parameters
    ----------
    start (int)
        Global index of the first draw.
    n_draws (int)
        How many draws.

    Returns
    -------
    null (np.ndarray)
        Added ``vm_logscore`` under a circularly shifted neural column.
    """

    state = _WORKER
    values = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        column = shifted_neural_column(state["design"], state["window_edges"], state["spike_seconds"],
                                       state["durations"], state["width"],
                                       np.random.default_rng(state["seed"] + start + draw),
                                       state["guard_seconds"], state["sigma_floor"])
        values[draw] = nested_scores(state["design"], state["settings"], state["n_lags"], neural=column,
                                     reduced_predicted=state["reduced_predicted"])["added"]
    return values


def vocal_gating_feature_job(feature_index: int) -> tuple:
    """
    Description
    -----------
    One feature's gating statistic and its null, in a worker.

    Each feature's draws use seeds ``seed + feature_index * n_draws + draw``, so no two features share a
    shift within the screen's fixed draw count.

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
        The permutation null, escalated.
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
    The nested vocal-manifold position decoding section as written (roster ruled 2026-09-14).

    The held-out session is recorded by NAME. ``nested_scores`` records it as the session's slot, which
    a figure would otherwise have to map back through an event set it no longer has.

    Parameters
    ----------
    scores (dict)
        ``nested_scores`` on the observed neural column.
    session_ids (list)
        The events' sessions, indexed by slot.
    null (np.ndarray)
        The circular-shift null, escalated.
    p_value (float)
        Empirical p.
    at_floor (bool)
        Whether ``p_value`` sits at its resolution floor.
    no_behaviour (float)
        The pooled score of the no-behaviour model on the same calls; see ``no_behaviour_predictions``.
    behaviour_control (dict)
        Which cohort control produced the reduced model -- its features, the day held out of its fit,
        and the settings hash it was fitted under. Without this an added score cannot be traced to the
        control that made it, and the control is now an external artifact rather than a refit.

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
    The gating section as written (roster ruled 2026-09-14): statistics, nulls and p-values, and NO
    decisions.

    The per-feature label, the borderline flag and the unit's best label are deliberately dropped here --
    the verdicts are consulted only for their p-values and sigma margin. Features that could not be
    tested carry their reason instead.

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
                             # The additive model entire, and what there was to explain: the comparison
                             # the sub-analysis is for is additive against multiplicative, and dividing
                             # either by the baseline puts it on the encoding's D2 scale.
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
    Vocal occurrence for one unit: does its count distinguish the window around a call's onset from
    silence, in a way that transfers across sessions? Written as its own section.

    The statistic is the leave-one-session-out transfer logistic, tested against circular shifts of the
    spike train that recount both classes, escalated along the shared ladder.

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
    # SENSITIVITY, kept beside the discrimination. The statistic and the AUROC both average over every
    # window, so a unit that fires on only part of its calls scores lower -- but neither says which of the
    # two it is, and two units at the same score can be "fires weakly on nearly every call" or "fires hard
    # on half of them". These two fractions separate them, and cost nothing: on the pilot units they run
    # from 44% of calls with a spike (a unit nearly silent otherwise) to 94%.
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
    Vocalization identity for one unit: does its count carry information about WHICH call it is, as a
    position on the vocal manifold? Written as its own section.

    The statistic is the held-out prior-corrected decode gain, tested against the exact within-session
    position permutation, escalated along the shared ladder.

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
    folds = decoding_context(events["positions"], events["session_index"], decoding)
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
                                        decoding, identity["gain"], null))
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

    What is tested is whether the neuron beats the BEHAVIOUR MODEL, which is only worth claiming when that
    model predicts -- so its held-out improvement over a no-behaviour model on the same calls is stored beside
    the result (ruled 2026-09-15). The kinematic data are released before the pool starts, so the largest
    arrays are never held while workers are alive.

    Parameters
    ----------
    unit (dict)
        The unit record.
    events (dict)
        Output of ``assemble_unit_vocal_events`` -- the same calls the vocalization-identity decoding
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
    window = settings["vocalization_settings"]["spike_window"]
    per_session = assemble_unit_sessions(unit, data_root, settings["kinematic_features"],
                                         encoding["history_pre_seconds"], encoding["clean_post_seconds"],
                                         settings["vocalization_settings"]["clean_against"],
                                         settings["vocalization_settings"]["vocal_emitter"],
                                         settings["vocalization_settings"])
    fps = per_session[unit["courtship_sessions"][0]]["fps"]
    history_lags = int(np.floor(encoding["history_pre_seconds"] * fps))
    reduced = reduced_model_features(settings)
    design = nested_design(events, per_session, reduced, history_lags, nested["sigma_floor"])
    # The behaviour control is FITTED AT COHORT SCALE and carried in as its prediction. Refitting it
    # here is the failure it exists to fix: this unit has ~2,000 calls against the thousands of columns
    # the raw block carries, and such a control scores BELOW a constant fitted on the same calls. As
    # four columns the behaviour information comes from the whole cohort and only four coefficients are
    # estimated on the unit. `n_lags` changes with the block, which is why the swap returns both.
    control = load_control_model(settings)
    design, n_lags = apply_behaviour_control(design, control, str(unit["rec_date"]), settings)
    reduced_predicted = reduced_predictions(design, nested, n_lags)
    scores = nested_scores(design, nested, n_lags, reduced_predicted=reduced_predicted)
    message_output(f"  NESTED DECODING  added {scores['added']:+.5f} over {scores['n_events']} events")

    no_behaviour = pooled_macro_score(no_behaviour_predictions(design, nested, n_lags), design["positions"],
                                      design["region_labels"], nested)
    message_output(f"    behaviour model over no behaviour: {scores['reduced'] - no_behaviour:+.5f}")
    del per_session

    state = {"design": design, "settings": nested, "n_lags": n_lags,
             "window_edges": events["call_start"][design["kept"]] - window["pre_offset_seconds"],
             "spike_seconds": session_spike_seconds(events, data_root, unit),
             "durations": session_durations(events), "width": window["width_seconds"],
             "seed": int(settings["null"]["shuffle_seed"]),
             "guard_seconds": settings["null"]["shuffle_guard_seconds"],
             "sigma_floor": nested["sigma_floor"], "reduced_predicted": reduced_predicted}
    with spawn_pool(n_workers, state) as pool:
        draw_more = parallel_draw_more(pool, nested_decoding_null_chunk, n_workers)
        null = draw_more(settings["null"]["n_shuffles"])
        p_value, at_floor, null = escalated_empirical_pvalue(scores["added"], draw_more,
                                                             settings["significance"]["escalation_ladder"],
                                                             null, message_output)
    payload = nested_decoding_payload(scores, events["session_ids"], null, p_value, at_floor,
                                      no_behaviour, design["behaviour_control"])
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

    Whether to run at all is decided first, from the vocal-occurrence and vocalization-identity p-values. With both identifiability gates switched off
    (their shipped default) the counts ``should_run_gating`` would check cannot change its answer, so an
    untuned unit is skipped WITHOUT assembling its kinematics. With either gate on, the universe is
    built first so identifiability can be checked before tuning, which is the order the plan requires:
    a unit both untestable and untuned reports the stronger fact.

    The Bonferroni family for the printed labels and the unit-level p is the features actually tested.

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
                                         settings["vocalization_settings"]["vocal_emitter"],
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
    ``vocal_gating.vocal_tuning_alpha``, no unit can ever count as tuned, and vocal gating is skipped for
    EVERY unit with the reason ``no_vocal_tuning`` -- a statement about the draw count masquerading as one
    about the unit. Found on a 20-draw smoke run of a strongly tuned unit, whose p-values sat at their
    1/21 floor of 0.048 against a threshold of 0.01. The fewest draws that can clear ``alpha`` is
    ``floor(1 / alpha)``: 100 at the shipped 0.01.

    The same kind of guard vocal gating already applies to its own shuffle count against its Bonferroni
    bar.

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

    The call-grain event set is assembled ONCE and shared by every analysis that scores calls, so vocal
    occurrence, vocalization identity and the nested decoding all see exactly the same events. Vocal
    gating needs the vocal-occurrence and vocalization-identity p-values: it uses the ones computed in
    this job, reads any it did not compute from the unit's result file, and refuses to run if one is in
    neither place -- or if either was computed from too few draws ever to clear the activation threshold.

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
    print(f"finished in {(datetime.now() - started).total_seconds() / 60:.1f} min")  # noqa: T201
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
