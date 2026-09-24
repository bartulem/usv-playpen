"""
@author: bartulem
The nested decoding's behaviour control, fitted once at COHORT scale instead of once per unit.

WHY THIS EXISTS. The control used to be refitted inside every unit: five features over six hundred lags
is three thousand columns, against a unit's ~2,000 calls. Measured 2026-09-22, that control scored BELOW
a constant fitted on the same calls, so the added score it produced was the neuron measured against
nothing. The shortfall is the call count and not the recordings -- P1's own data cut to that size
reproduces it exactly -- and a model fitted on the whole ephys cohort predicts call position perfectly
well. So the control is estimated once, here, and carried into each unit as its PREDICTION.

TWO STAGES, sharing one assembly because the assembly is the expensive half.
``select_control_features`` runs forward selection over the full kinematic set; ``fit_control_models``
fits the chosen features once per held-out recording day. Both write an artifact that the per-unit run
READS and checks, never recomputes.

HELD OUT BY DAY, and the reason is narrower than it looks. The cohort spans more days than animals, so a
day holdout does not hold out the animal -- and holding out the animal would be WRONG. This model never
sees spikes, and units are single-date, so its training set cannot carry information about the neuron
under test. It is a nuisance model: ``added = full - reduced`` uses the same prediction on both sides, so
a stronger control only SHRINKS the added score, and a weaker one makes the claim easier to pass. What
the day holdout buys is the one thing needed -- the unit's own calls stay out of the training set, so the
reduced score is genuinely held out.

THE TORUS COLUMNS ARE NOT IN THE DATA TREE. Its ``usv_summary`` files carry no QLVM columns at all; they
live under ``data_roots.vocal_summary_root``, which is why that key exists.
"""

from __future__ import annotations

import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from ..modeling.modeling_metadata import compute_settings_sha256
from .neural_design_assembly import (
    build_zscored_feature_frames,
    emitter_names,
    lagged_design,
    session_timebase,
)
from .neural_nested_decoding import (
    NestedTorusRegression,
    inverse_region_frequency_weights,
    macro_von_mises_logscore,
    nested_settings,
)


def torus_embedding(positions: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    A torus point as four coordinates, so a predicted POSITION can serve as a predictor.

    A torus coordinate is circular, so a raw coordinate would put a discontinuity at the wrap and make
    0.99 and 0.01 maximally distant. The sine/cosine pair is the representation the estimator already
    uses internally for its target.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n, 2)`` torus coordinates on the unit period.

    Returns
    -------
    embedded (np.ndarray)
        ``(n, 4)`` cosine and sine of each dimension.
    """

    angles = 2.0 * np.pi * positions
    return np.hstack([np.cos(angles[:, [0]]), np.sin(angles[:, [0]]),
                      np.cos(angles[:, [1]]), np.sin(angles[:, [1]])])


def control_estimator(n_features: int, n_lags: int, settings: dict) -> NestedTorusRegression:
    """
    Description
    -----------
    The nested decoding's estimator at the requested shape, carrying the run's own penalties.

    Parameters
    ----------
    n_features (int)
        Behaviour features in the design.
    n_lags (int)
        Columns per feature.
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    model (NestedTorusRegression)
        An unfitted estimator.
    """

    nested = nested_settings(settings)
    return NestedTorusRegression(
        n_behaviour_features=n_features, n_lags=n_lags, n_neural_columns=0,
        lambda_smooth=nested["lambda_smooth"], l2_reg=nested["l2_reg"],
        smoothness_derivative_order=nested["smoothness_derivative_order"],
        period=nested["torus_period"])


def session_call_design(session_directory: str, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    One session's calls, their acoustic regions, and the kinematic history behind each.

    The calls kept are the ones the vocal decoding would see: the recorded animal's own, dropped when
    another call intrudes on the pre-window, when the torus position or region label is missing, or when
    the history a prediction needs runs off the start of the session. The history window ENDS at the
    frame before onset, so it is strictly pre-onset.

    Parameters
    ----------
    session_directory (str)
        Full path to the session directory.
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.

    Returns
    -------
    block (dict)
        ``session_id``, ``design``, ``positions``, ``regions``, ``feature_names``, ``n_kept`` and
        ``call_start``; or ``session_id`` and ``reason`` when the session contributes nothing.
    """

    session_id = pathlib.Path(session_directory).name
    data_root = settings["data_roots"]["data_root"]
    position_columns = settings["vocal_decoding"]["usv_manifold_column_names"]
    region_column = settings["nested_vocal_manifold_position_decoding"]["region_label_column"]
    window = settings["vocalization_settings"]["spike_window"]
    minimum = settings["data_sufficiency"]["min_emitter_usvs_per_session"]

    track_names, _frame_rate, _n_frames = session_timebase(data_root, session_id)
    if len(track_names) != 2:
        return {"session_id": session_id, "reason": f"{len(track_names)} tracked animals"}
    recorded = track_names[0]

    usv_path = (pathlib.Path(settings["data_roots"]["vocal_summary_root"]) / session_id
                / f"{session_id}_usv_summary.csv")
    if not usv_path.exists():
        return {"session_id": session_id, "reason": f"no usv_summary at {usv_path}"}
    usv = pl.read_csv(usv_path, separator=",", infer_schema_length=5000)
    absent = [column for column in (*position_columns, region_column) if column not in usv.columns]
    if absent:
        return {"session_id": session_id, "reason": f"summary lacks {absent}"}

    focal_name = emitter_names(track_names, recorded,
                               settings["vocalization_settings"]["vocal_emitter"])[0]
    all_starts = usv["start"].to_numpy().astype(np.float64)
    all_stops = usv["stop"].to_numpy().astype(np.float64)
    focal = usv.filter(usv["emitter"] == focal_name)
    if focal.height < minimum:
        return {"session_id": session_id, "reason": f"{focal.height} emitter calls"}

    starts = focal["start"].to_numpy().astype(np.float64)
    positions = focal.select(position_columns).to_numpy().astype(np.float64)
    regions = focal[region_column].to_numpy().astype(np.float64)

    frames, camera, _names, _suffixes = build_zscored_feature_frames(
        session_dirs=[session_directory], kinematic_features=settings["kinematic_features"],
        recorded_mouse_id=recorded)
    frame_df = frames[session_id]
    feature_series = np.nan_to_num(frame_df.to_numpy().astype(np.float64), nan=0.0)

    anchors = np.round(starts * camera[session_id]).astype(int) - 1
    window_open = starts - window["pre_offset_seconds"]
    intrudes = np.array([bool(np.any((all_stops > open_at) & (all_starts < start)))
                         for open_at, start in zip(window_open, starts, strict=True)])
    keep = ((~intrudes) & np.isfinite(positions).all(axis=1) & np.isfinite(regions)
            & (anchors >= n_lags - 1) & (anchors < feature_series.shape[0]))
    if not np.any(keep):
        return {"session_id": session_id, "reason": "no call survives the filters"}

    rows = np.flatnonzero(keep)
    return {"session_id": session_id,
            "design": lagged_design(feature_series, anchors[rows], n_lags).astype(np.float32),
            "positions": positions[rows], "regions": regions[rows],
            "feature_names": list(frame_df.columns), "n_kept": int(rows.size),
            "call_start": starts[rows]}


def assemble_cohort(session_directories: list, settings: dict, n_lags: int,
                    n_workers: int) -> dict:
    """
    Description
    -----------
    Every listed session's calls and histories, stacked, with the recording day recorded per call.

    Parameters
    ----------
    session_directories (list)
        Full paths to the session directories.
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.
    n_workers (int)
        Worker processes for the per-session assembly.

    Returns
    -------
    cohort (dict)
        ``design``, ``positions``, ``regions``, ``days``, ``session_ids``, ``feature_names`` and
        ``skipped`` (session id to reason).
    """

    blocks, skipped = [], {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for block in pool.map(_assemble_one, [(directory, settings, n_lags)
                                              for directory in session_directories]):
            if "reason" in block:
                skipped[block["session_id"]] = block["reason"]
            else:
                blocks.append(block)
    if not blocks:
        msg = "no session contributed calls; the cohort control cannot be fitted."
        raise ValueError(msg)

    feature_names = blocks[0]["feature_names"]
    for block in blocks:
        if block["feature_names"] != feature_names:
            msg = (f"{block['session_id']} resolved a different feature set; features must be "
                   f"canonical across sessions or the design columns do not line up.")
            raise ValueError(msg)

    return {"design": np.vstack([block["design"] for block in blocks]),
            "positions": np.vstack([block["positions"] for block in blocks]),
            "regions": np.concatenate([block["regions"] for block in blocks]),
            "days": np.concatenate([np.full(block["n_kept"], block["session_id"][:8])
                                    for block in blocks]),
            "session_ids": [block["session_id"] for block in blocks],
            "feature_names": feature_names, "skipped": skipped}


def _assemble_one(arguments: tuple) -> dict:
    """
    Description
    -----------
    Worker shim so the per-session assembly can be mapped over a process pool.

    Parameters
    ----------
    arguments (tuple)
        ``(session_directory, settings, n_lags)``.

    Returns
    -------
    block (dict)
        As :func:`session_call_design`.
    """

    session_directory, settings, n_lags = arguments
    return session_call_design(session_directory, settings, n_lags)


def held_out_by_day(design: np.ndarray, positions: np.ndarray, regions: np.ndarray,
                    days: np.ndarray, settings: dict, n_lags: int) -> np.ndarray:
    """
    Description
    -----------
    Predicted positions for every call, each day predicted by a model fitted on the other days.

    Parameters
    ----------
    design (np.ndarray)
        ``(n, n_features * n_lags)`` behaviour histories.
    positions (np.ndarray)
        ``(n, 2)`` true torus positions.
    regions (np.ndarray)
        Acoustic region label per call.
    days (np.ndarray)
        Recording day per call.
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.

    Returns
    -------
    predicted (np.ndarray)
        ``(n, 2)`` held-out predicted positions.
    """

    weights = inverse_region_frequency_weights(regions)
    n_features = design.shape[1] // n_lags
    predicted = np.full_like(positions, np.nan)
    for held_out in np.unique(days):
        test = np.flatnonzero(days == held_out)
        train = np.flatnonzero(days != held_out)
        model = control_estimator(n_features, n_lags, settings)
        model.fit(design[train].astype(np.float64), positions[train], sample_weight=weights[train])
        predicted[test] = model.predict(design[test].astype(np.float64), snap=False)
    return predicted


def pooled_macro(predicted: np.ndarray, positions: np.ndarray, regions: np.ndarray,
                 settings: dict) -> float:
    """
    Description
    -----------
    The nested decoding's own macro von Mises log-score, pooled over the given calls.

    Parameters
    ----------
    predicted (np.ndarray)
        Predicted torus positions.
    positions (np.ndarray)
        True torus positions.
    regions (np.ndarray)
        Acoustic region label per call.
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    score (float)
        Macro von Mises log-likelihood, NaN when no region clears the minimum.
    """

    nested = nested_settings(settings)
    return float(macro_von_mises_logscore(predicted, positions, regions, metric="torus",
                                          period=nested["torus_period"],
                                          min_region_events=nested["min_region_events"]))


def fit_control_models(cohort: dict, feature_names: list, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    Fit the control once per held-out recording day and return the artifact payload.

    Only ``coef_`` and ``intercept_`` are kept: those are all ``predict`` consumes, so the artifact
    survives any change to the estimator class that leaves its linear form alone. Pickling fitted
    objects would tie every banked result to one version of that class.

    Parameters
    ----------
    cohort (dict)
        From :func:`assemble_cohort`.
    feature_names (list)
        The control's features, a subset of the cohort's canonical feature list.
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.

    Returns
    -------
    artifact (dict)
        ``features``, ``n_lags``, ``coefficients`` (day to ``{coef, intercept}``), ``held_out_score``,
        ``no_behaviour_score``, per-day scores, counts and ``settings_sha256``.
    """

    positions, regions, days = cohort["positions"], cohort["regions"], cohort["days"]
    columns = feature_columns(cohort["feature_names"], feature_names, n_lags)
    design = cohort["design"][:, columns]
    weights = inverse_region_frequency_weights(regions)

    coefficients, per_day = {}, {}
    predicted = np.full_like(positions, np.nan)
    for held_out in np.unique(days):
        test = np.flatnonzero(days == held_out)
        train = np.flatnonzero(days != held_out)
        model = control_estimator(len(feature_names), n_lags, settings)
        model.fit(design[train].astype(np.float64), positions[train], sample_weight=weights[train])
        predicted[test] = model.predict(design[test].astype(np.float64), snap=False)
        coefficients[str(held_out)] = {"coef": model.coef_.copy(),
                                       "intercept": model.intercept_.copy()}

    blank = np.zeros_like(design[:, :n_lags])
    no_behaviour = held_out_by_day(blank, positions, regions, days, settings, n_lags)
    for held_out in np.unique(days):
        rows = np.flatnonzero(days == held_out)
        per_day[str(held_out)] = {
            "n_calls": int(rows.size),
            "behaviour": pooled_macro(predicted[rows], positions[rows], regions[rows], settings),
            "no_behaviour": pooled_macro(no_behaviour[rows], positions[rows], regions[rows],
                                         settings)}

    return {"features": list(feature_names), "n_lags": int(n_lags), "coefficients": coefficients,
            "held_out_score": pooled_macro(predicted, positions, regions, settings),
            "no_behaviour_score": pooled_macro(no_behaviour, positions, regions, settings),
            "per_day": per_day, "n_calls": int(positions.shape[0]),
            "session_ids": list(cohort["session_ids"]), "skipped": dict(cohort["skipped"]),
            "settings_sha256": compute_settings_sha256(settings)}


def feature_columns(canonical: list, wanted: list, n_lags: int) -> np.ndarray:
    """
    Description
    -----------
    Design columns belonging to the wanted features, in the order they are asked for.

    Parameters
    ----------
    canonical (list)
        The design's full feature order.
    wanted (list)
        Features to select.
    n_lags (int)
        Columns per feature.

    Returns
    -------
    columns (np.ndarray)
        Column indices.
    """

    missing = [name for name in wanted if name not in canonical]
    if missing:
        msg = f"features {missing} are not in the assembled design, which carries {canonical}."
        raise KeyError(msg)
    return np.concatenate([np.arange(canonical.index(name) * n_lags,
                                     (canonical.index(name) + 1) * n_lags) for name in wanted])


def write_control_model(artifact: dict, path: str) -> None:
    """
    Description
    -----------
    Persist the fitted control.

    Parameters
    ----------
    artifact (dict)
        From :func:`fit_control_models`.
    path (str)
        Destination file.

    Returns
    -------
    None
    """

    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_control_model(settings: dict) -> dict:
    """
    Description
    -----------
    Read the fitted control and CHECK it against the run it is about to serve.

    A control silently out of step with its settings would change every added score without changing a
    single reported configuration value, which is the failure mode this check exists for. The features
    are checked too, since those are what "beyond behaviour" means.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    artifact (dict)
        As written by :func:`fit_control_models`.
    """

    path = pathlib.Path(
        settings["nested_vocal_manifold_position_decoding"]["behaviour_control_model_path"])
    if not path.exists():
        msg = (f"behaviour control model not found at {path}; fit it once with the behaviour-control "
               f"dispatcher before running the nested decoding.")
        raise FileNotFoundError(msg)
    with path.open("rb") as handle:
        artifact = pickle.load(handle)

    frozen = list(settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"])
    if list(artifact["features"]) != frozen:
        msg = (f"behaviour control drift: {path.name} was fitted on {list(artifact['features'])}, but "
               f"settings freeze {frozen}. Refit the control or update the settings deliberately.")
        raise ValueError(msg)
    return artifact


def control_predictions(artifact: dict, held_out_day: str, design: np.ndarray,
                        settings: dict) -> np.ndarray:
    """
    Description
    -----------
    Predicted torus positions for a unit's calls, from the fold that did NOT see that unit's day.

    Parameters
    ----------
    artifact (dict)
        From :func:`load_control_model`.
    held_out_day (str)
        The unit's recording date, as the eight-character day key.
    design (np.ndarray)
        ``(n, n_features * n_lags)`` histories for the unit's calls, in the artifact's feature order.
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    predicted (np.ndarray)
        ``(n, 2)`` predicted torus positions.
    """

    if held_out_day not in artifact["coefficients"]:
        msg = (f"the behaviour control carries no fold for day {held_out_day}; it was fitted over "
               f"{sorted(artifact['coefficients'])}. A unit's own day must be held out.")
        raise KeyError(msg)
    fold = artifact["coefficients"][held_out_day]
    model = control_estimator(len(artifact["features"]), artifact["n_lags"], settings)
    model.coef_ = fold["coef"]
    model.intercept_ = fold["intercept"]
    return model.predict(design.astype(np.float64), snap=False)


def paired_gain(candidate: np.ndarray, incumbent: np.ndarray) -> tuple:
    """
    Description
    -----------
    The PAIRED per-day improvement of a candidate over the incumbent, and its own standard error.

    Pairing is load-bearing and its absence is a defect this project has now found TWICE. Recording days
    differ enormously in absolute score -- roughly -3.42 to -3.67 -- so the standard error of the raw
    scores measures how much DAYS differ, not how reliably the candidate improves on them. A per-step
    gain is an order of magnitude smaller than that between-day spread, so comparing the two rejects
    every feature after the first, and the selected model then scores WORSE than an unselected one.
    Differencing within day removes the day effect, which is what the acceptance rule is testing.

    Parameters
    ----------
    candidate (np.ndarray)
        Per-day scores of the candidate model.
    incumbent (np.ndarray)
        Per-day scores of the current model, same days in the same order.

    Returns
    -------
    gain, error (tuple)
        Mean paired improvement, and the standard error of the paired differences.
    """

    difference = candidate - incumbent
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size < 2:
        return float(finite.mean()), 0.0
    return float(finite.mean()), float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def _score_candidate(arguments: tuple) -> tuple:
    """
    Description
    -----------
    Held-out-by-day scores of one candidate feature set, reading the design by MEMORY MAP.

    The design runs to gigabytes, so it is neither pickled to each task nor inherited through a fork --
    the first is prohibitive and the second silently fails wherever processes are spawned rather than
    forked. Mapping the file costs nothing and behaves the same on every platform.

    Parameters
    ----------
    arguments (tuple)
        ``(design_path, columns, positions, regions, days, settings, n_lags, n_features, candidate)``.

    Returns
    -------
    result (tuple)
        The candidate, its per-day scores, and its pooled score.
    """

    (design_path, columns, positions, regions, days, settings, n_lags, n_features,
     candidate) = arguments
    design = np.load(design_path, mmap_mode="r")[:, columns]
    weights = inverse_region_frequency_weights(regions)
    predicted = np.full_like(positions, np.nan)
    for held_out in np.unique(days):
        test = np.flatnonzero(days == held_out)
        train = np.flatnonzero(days != held_out)
        model = control_estimator(n_features, n_lags, settings)
        model.fit(np.asarray(design[train], dtype=np.float64), positions[train],
                  sample_weight=weights[train])
        predicted[test] = model.predict(np.asarray(design[test], dtype=np.float64), snap=False)
    per_day = np.asarray([pooled_macro(predicted[days == day], positions[days == day],
                                       regions[days == day], settings) for day in np.unique(days)])
    return candidate, per_day, pooled_macro(predicted, positions, regions, settings)


def select_control_features(cohort: dict, settings: dict, n_lags: int, n_workers: int,
                            design_path: str, external_reference: list | None = None) -> dict:
    """
    Description
    -----------
    Forward selection of the control's features on the cohort, held out by recording day.

    Greedy by held-out macro score, accepting a candidate only when its PAIRED gain less that gain's own
    standard error still clears zero, and stopping at the first rejection -- the codebase's rule, with
    the pairing it is supposed to have.

    An ``external_reference`` feature set is scored under the identical setup and recorded beside the
    result. That is not decoration: the selection sees every day, including the days units are later
    scored on, so it carries some optimism. A feature set chosen ELSEWHERE cannot, and the gap between
    the two BOUNDS the optimism instead of leaving it to be argued about.

    Parameters
    ----------
    cohort (dict)
        From :func:`assemble_cohort`.
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.
    n_workers (int)
        Worker processes.
    design_path (str)
        Where the cohort design is stored as ``.npy`` for the workers to map.
    external_reference (list)
        An independently chosen feature set to score alongside, or None.

    Returns
    -------
    selection (dict)
        ``selected_features``, ``per_step``, ``rejected``, ``pooled_score``, ``no_behaviour_score``,
        ``external_reference``, counts and ``settings_sha256``.
    """

    canonical = cohort["feature_names"]
    positions, regions, days = cohort["positions"], cohort["regions"], cohort["days"]
    shared = (design_path, positions, regions, days, settings, n_lags)

    def score(names: list) -> tuple:
        columns = feature_columns(canonical, names, n_lags)
        return _score_candidate((shared[0], columns, *shared[1:], len(names), (*names,)))

    blank = np.zeros_like(cohort["design"][:, :n_lags])
    no_behaviour = held_out_by_day(blank, positions, regions, days, settings, n_lags)
    incumbent = np.asarray([pooled_macro(no_behaviour[days == day], positions[days == day],
                                         regions[days == day], settings)
                            for day in np.unique(days)])
    no_behaviour_score = pooled_macro(no_behaviour, positions, regions, settings)

    selected: list = []
    remaining = list(canonical)
    per_step: list = []
    rejected = None
    pooled_score = no_behaviour_score
    while remaining:
        tasks = [(design_path, feature_columns(canonical, [*selected, name], n_lags), positions,
                  regions, days, settings, n_lags, len(selected) + 1, (*selected, name))
                 for name in remaining]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_score_candidate, tasks))
        ranked = sorted(((paired_gain(per_day, incumbent), pooled, per_day, candidate)
                         for candidate, per_day, pooled in results),
                        key=lambda row: -row[0][0] if np.isfinite(row[0][0]) else np.inf)
        (gain, error), pooled, per_day, candidate = ranked[0]
        if not (gain - error) > 0.0:
            rejected = {"feature": candidate[-1], "paired_gain": gain, "standard_error": error}
            break
        selected = list(candidate)
        remaining = [name for name in remaining if name != candidate[-1]]
        incumbent, pooled_score = per_day, pooled
        per_step.append({"feature": candidate[-1], "paired_gain": gain, "standard_error": error,
                         "pooled": pooled})

    reference = None
    if external_reference is not None:
        _candidate, _per_day, reference_pooled = score(list(external_reference))
        reference = {"features": list(external_reference), "pooled": reference_pooled}

    return {"selected_features": selected, "per_step": per_step, "rejected": rejected,
            "pooled_score": pooled_score, "no_behaviour_score": no_behaviour_score,
            "external_reference": reference, "days": sorted({str(day) for day in days}),
            "n_calls": int(positions.shape[0]), "session_ids": list(cohort["session_ids"]),
            "skipped": dict(cohort["skipped"]), "settings_sha256": compute_settings_sha256(settings)}


EMBEDDING_COLUMNS = ("control.cos1", "control.sin1", "control.cos2", "control.sin2")


def apply_behaviour_control(design: dict, artifact: dict, held_out_day: str,
                            settings: dict) -> tuple:
    """
    Description
    -----------
    Replace a unit's raw behaviour block with the cohort control's PREDICTION, and say what n_lags the
    result must now be scored at.

    The nesting is unchanged -- reduced is behaviour, full is behaviour plus the neuron's one column --
    and so is the statistic. What changes is the representation. The raw block is five hundred to three
    thousand columns refit on the unit's own ~2,000 calls, which is the failure this control exists to
    fix; carrying the control in raw would reintroduce it. As four columns, the behaviour information
    comes from the whole cohort while only four coefficients are estimated on the unit.

    It returns the lag count with the design ON PURPOSE. Every scorer takes ``n_lags`` separately and
    derives the design's shape from it, so a caller that swapped the block and forgot to switch to one
    lag would be describing a different matrix. The estimator would raise on the column count rather
    than fit something wrong, but pairing the two here means it cannot come up.

    Parameters
    ----------
    design (dict)
        From ``nested_design``, its behaviour block carrying the control's features at full lag.
    artifact (dict)
        From :func:`load_control_model`.
    held_out_day (str)
        The unit's recording date as the eight-character day key; its fold must not have seen it.
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    controlled, n_lags (tuple)
        The design with its behaviour block replaced, and the lag count to score it at (always 1).
    """

    expected = len(artifact["features"]) * artifact["n_lags"]
    if design["behaviour"].shape[1] != expected:
        msg = (f"the unit's behaviour block has {design['behaviour'].shape[1]} columns but the control "
               f"was fitted on {len(artifact['features'])} features x {artifact['n_lags']} lags = "
               f"{expected}. The block must be built from the control's own features.")
        raise ValueError(msg)

    predicted = control_predictions(artifact, held_out_day, design["behaviour"], settings)
    controlled = dict(design)
    controlled["behaviour"] = torus_embedding(predicted)
    controlled["feature_names"] = list(EMBEDDING_COLUMNS)
    controlled["behaviour_control"] = {"features": list(artifact["features"]),
                                       "held_out_day": held_out_day,
                                       "predicted_positions": predicted,
                                       "settings_sha256": artifact["settings_sha256"]}
    return controlled, 1
