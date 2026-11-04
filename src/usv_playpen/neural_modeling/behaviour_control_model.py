"""
@author: bartulem
The nested decoding's behaviour control, fitted once on the whole cohort rather than per unit.

A per-unit control (several features over 600 lags) has thousands of columns against a unit's few
thousand USVs and cannot be estimated, so the control is fitted once on the cohort's USVs and enters
each unit as its predicted torus position.

Two stages share one assembly (:func:`assemble_cohort`):

- :func:`select_control_features` -- forward selection over the kinematic set, held out by recording
  day, with a paired one-standard-error acceptance rule;
- :func:`fit_control_models` -- fits the selected features once per held-out recording day.

Both write an artifact that the per-unit run reads and checks (:func:`load_control_model`) and never
recomputes. :func:`apply_behaviour_control` swaps a unit's raw behaviour block for the four
sine/cosine columns of the control's prediction from the fold that did not see the unit's day. Torus
positions and region labels are read from the ``Data`` tree summaries through ``load_session_usvs``.

The onset axis has its own control too (:func:`fit_onset_control_models`): the same two-class
model, fitted on the cohort's clean USV onsets against quiet tiles placed by the per-unit occurrence
decoder's rule (the same tile width, guard bands and per-session cap; which tiles a capped session keeps is
drawn per session here, so it need not be the same subset a unit's run keeps) (:func:`assemble_onset_cohort`),
held out by recording day, and
entering each unit as the base-rate-free log-odds of onset against quiet at the unit's own windows
(:func:`onset_control_log_odds`).

The categorical target has its own control (:func:`fit_category_control_models`): the behavioural
modelling's multinomial logistic regression (``SmoothMultinomialLogisticRegression`` on the same
B-spline lag basis, penalties and focal loss as that pipeline, from
``nested_vocal_decoding.category_control``), fitted on the same cohort assembly and held out by the
same recording day. It enters each unit as the base-rate-free category evidence of each USV
(:func:`category_control_evidence`), which the per-unit run turns into a per-USV category prior.
"""

from __future__ import annotations

import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..modeling.jax_multinomial_logistic_regression import (
    SmoothMultinomialLogisticRegression,
)
from ..modeling.modeling_bases_functions import (
    project_history_onto_basis,
    resolve_temporal_basis,
)
from ..modeling.modeling_metadata import compute_settings_sha256
from ..os_utils import atomic_output_path, configure_path
from .deviance_metrics import area_under_roc
from .neural_artifacts import build_provenance
from .neural_design_assembly import (
    build_zscored_feature_frames,
    clean_window_mask,
    emitter_names,
    filter_vocalizations,
    lagged_design,
    load_session_usvs,
    onset_history_anchor_frames,
    session_timebase,
    worker_process_context,
)
from .neural_nested_decoding import (
    NestedTorusRegression,
    check_control_target,
    inverse_region_frequency_weights,
    macro_von_mises_logscore,
    nested_settings,
)
from .neural_vocal_occurrence import quiet_tile_edges


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
        period=nested["torus_period"], huber_delta=nested["huber_delta"],
        learning_rate=nested["learning_rate"], max_iter=nested["max_iter"], tol=nested["tol"],
        random_state=nested["random_state"])


def behaviour_history_lags(settings: dict, session_directories: list) -> int:
    """
    Description
    -----------
    History length of the behaviour design, in tracking frames: ``floor(history_pre_seconds * fps)``.

    The behaviour control is applied to histories the per-unit side builds at exactly this length
    (``kinematic_encoding.history_pre_seconds`` at the session's camera rate), so the length is derived
    from the same setting rather than declared a second time. Each session's measured camera rate differs
    slightly (about 150.06-150.11 fps), so the lag count is computed per session, as the unit side does,
    and every listed session must give the same count; otherwise one design would hold histories of
    different lengths.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict; ``kinematic_encoding.history_pre_seconds`` and
        ``paths.data_root`` are read.
    session_directories (list)
        Full paths of the sessions the control is built from.

    Returns
    -------
    n_lags (int)
        History length in tracking frames (600 for 4 s at 150 fps).

    Raises
    ------
    ValueError
        If no session is listed, or the sessions' camera rates give different lag counts.
    """

    if not session_directories:
        msg = "no session is listed, so the camera rate behind the behaviour history is unknown."
        raise ValueError(msg)
    data_root = settings["paths"]["data_root"]
    history_seconds = settings["kinematic_encoding"]["history_pre_seconds"]
    lags = {}
    for directory in session_directories:
        session_id = pathlib.Path(directory).name
        lags[session_id] = int(np.floor(history_seconds * session_timebase(data_root, session_id)[1]))
    distinct = sorted(set(lags.values()))
    if len(distinct) != 1:
        msg = (f"the listed sessions' camera rates give history lengths of {distinct} frames for "
               f"{history_seconds} s; one design cannot hold histories of different lengths.")
        raise ValueError(msg)
    return distinct[0]


def session_vocalization_design(session_directory: str, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    One session's vocalizations, their acoustic regions, and the kinematic history behind each.

    The vocalizations kept are the rows the per-unit analysis scores, so the control is trained on the
    population it is later applied to. Two filters, with the two scopes the unit side uses:

    - the analysed set is ``filter_vocalizations(usv, vocalizations, "analysed")`` restricted to
      ``vocalizations.vocal_emitter`` (``'self'``, ``'partner'``, or ``'all'`` for no emitter
      filter) -- noise rows are never vocalizations and squeaks are a different vocal type with their
      own coordinates, so neither is a training row;
    - the intrusion guard is ``filter_vocalizations(usv, vocalizations, "guard")`` over every
      emitter -- noise never intrudes, and a squeak intrudes when ``squeaks_block_quiet`` says it does.

    A vocalization is then dropped when another one from the guard set intrudes on its pre-window
    (``vocalizations.spike_window.pre_offset_seconds``), when its torus position or region
    label is missing, or when the history a prediction needs runs off the start of the session. The
    history window ends at the frame before onset (``onset_history_anchor_frames``), the same anchor the
    nested decoding uses for the unit-side histories.

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
        ``vocalization_start``; or ``session_id`` and ``reason`` when the session contributes nothing.
    """

    session_id = pathlib.Path(session_directory).name
    data_root = settings["paths"]["data_root"]
    position_columns = settings["vocalizations"]["usv_manifold_column_names"]
    region_column = settings["vocalizations"]["category_column"]
    vocal_settings = settings["vocalizations"]
    window = vocal_settings["spike_window"]
    minimum = settings["cohort"]["min_emitter_usvs_per_session"]

    track_names, _frame_rate, _n_frames = session_timebase(data_root, session_id)
    if len(track_names) != 2:
        return {"session_id": session_id, "reason": f"{len(track_names)} tracked animals"}
    recorded = track_names[0]

    try:
        usv = load_session_usvs(data_root, session_id)
    except FileNotFoundError as error:
        return {"session_id": session_id, "reason": str(error)}
    # A session with no summary is skipped. A summary missing the configured columns affects every
    # session alike, so it raises instead of being skipped silently.
    absent = [column for column in (*position_columns, region_column) if column not in usv.columns]
    if absent:
        msg = (f"{session_id}: usv_summary lacks {absent}. Reconcile "
               f"vocalizations.usv_manifold_column_names and "
               f"vocalizations.category_column with the summary's "
               f"columns.")
        raise KeyError(msg)

    # The same two scopes the unit side uses: the guard (every emitter) decides intrusion, the
    # analysed set restricted to the configured emitter is what is trained on.
    guard = filter_vocalizations(usv, vocal_settings, "guard")
    all_starts = guard["start"].to_numpy().astype(np.float64)
    all_stops = guard["stop"].to_numpy().astype(np.float64)
    analysed = filter_vocalizations(usv, vocal_settings, "analysed")
    focal_names = emitter_names(track_names, recorded, vocal_settings["vocal_emitter"])
    focal = analysed.filter(analysed["emitter"].is_in(focal_names)) if focal_names else analysed
    if focal.height < minimum:
        return {"session_id": session_id, "reason": f"{focal.height} emitter vocalizations"}

    starts = focal["start"].to_numpy().astype(np.float64)
    positions = focal.select(position_columns).to_numpy().astype(np.float64)
    regions = focal[region_column].to_numpy().astype(np.float64)

    frames, camera, _names, _suffixes = build_zscored_feature_frames(
        session_dirs=[session_directory], kinematic_features=settings["kinematic_features"],
        recorded_mouse_id=recorded)
    frame_df = frames[session_id]
    feature_series = np.nan_to_num(frame_df.to_numpy().astype(np.float64), nan=0.0)

    anchors = onset_history_anchor_frames(starts, camera[session_id])
    window_open = starts - window["pre_offset_seconds"]
    intrudes = np.array([bool(np.any((all_stops > open_at) & (all_starts < start)))
                         for open_at, start in zip(window_open, starts, strict=True)])
    keep = ((~intrudes) & np.isfinite(positions).all(axis=1) & np.isfinite(regions)
            & (anchors >= n_lags - 1) & (anchors < feature_series.shape[0]))
    if not np.any(keep):
        return {"session_id": session_id, "reason": "no vocalization survives the filters"}

    rows = np.flatnonzero(keep)
    return {"session_id": session_id,
            "design": lagged_design(feature_series, anchors[rows], n_lags).astype(np.float32),
            "positions": positions[rows], "regions": regions[rows],
            "feature_names": list(frame_df.columns), "n_kept": int(rows.size),
            "vocalization_start": starts[rows]}


def assemble_cohort(session_directories: list, settings: dict, n_lags: int,
                    n_workers: int) -> dict:
    """
    Description
    -----------
    Every listed session's USVs and histories, stacked, with the recording day recorded per USV.

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
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=worker_process_context()) as pool:
        for block in pool.map(_assemble_one, [(directory, settings, n_lags)
                                              for directory in session_directories]):
            if "reason" in block:
                skipped[block["session_id"]] = block["reason"]
            else:
                blocks.append(block)
    if not blocks:
        msg = "no session contributed USVs; the cohort control cannot be fitted."
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
            "sessions": np.concatenate([np.full(block["n_kept"], block["session_id"]) for block in blocks]),
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
        As :func:`session_vocalization_design`.
    """

    session_directory, settings, n_lags = arguments
    return session_vocalization_design(session_directory, settings, n_lags)


def held_out_by_day(design: np.ndarray, positions: np.ndarray, regions: np.ndarray,
                    days: np.ndarray, settings: dict, n_lags: int) -> np.ndarray:
    """
    Description
    -----------
    Predicted positions for every USV, each day predicted by a model fitted on the other days.

    Parameters
    ----------
    design (np.ndarray)
        ``(n, n_features * n_lags)`` behaviour histories.
    positions (np.ndarray)
        ``(n, 2)`` true torus positions.
    regions (np.ndarray)
        Acoustic region label per USV.
    days (np.ndarray)
        Recording day per USV.
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
    The nested decoding's own macro von Mises log-score, pooled over the given USVs.

    Parameters
    ----------
    predicted (np.ndarray)
        Predicted torus positions.
    positions (np.ndarray)
        True torus positions.
    regions (np.ndarray)
        Acoustic region label per USV.
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


def cohort_sessions(cohort: dict, days: np.ndarray) -> np.ndarray | None:
    """
    Description
    -----------
    The session of every cohort row, for the stored held-out predictions; None for a cohort assembled
    without per-row sessions (only the day is then known).

    Parameters
    ----------
    cohort (dict)
        From one of the cohort assemblers.
    days (np.ndarray)
        The cohort's per-row day, whose length the sessions must match.

    Returns
    -------
    sessions (np.ndarray | None)
        Per-row session ids, or None.
    """

    if "sessions" not in cohort:
        return None
    sessions = np.asarray(cohort["sessions"])
    if sessions.size != np.asarray(days).size:
        msg = f"the cohort carries {sessions.size} row sessions for {np.asarray(days).size} rows."
        raise ValueError(msg)
    return sessions


def fit_control_models(cohort: dict, feature_names: list, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    Fit the control once per held-out recording day and return the artifact payload.

    Only ``coef_`` and ``intercept_`` are kept, since those are all ``predict`` consumes; the artifact
    therefore does not depend on the estimator class's pickled form.

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
        ``target`` (``"manifold"``), ``features``, ``n_lags``, ``coefficients`` (day to
        ``{coef, intercept}``), ``held_out_score``, ``no_behaviour_score``, per-day scores, counts and
        ``settings_sha256``.
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
            "n_vocalizations": int(rows.size),
            "behaviour": pooled_macro(predicted[rows], positions[rows], regions[rows], settings),
            "no_behaviour": pooled_macro(no_behaviour[rows], positions[rows], regions[rows],
                                         settings)}

    return {"target": "manifold", "features": list(feature_names), "n_lags": int(n_lags),
            "coefficients": coefficients,
            "held_out_score": pooled_macro(predicted, positions, regions, settings),
            "no_behaviour_score": pooled_macro(no_behaviour, positions, regions, settings),
            "per_day": per_day, "n_vocalizations": int(positions.shape[0]),
            "session_ids": list(cohort["session_ids"]), "skipped": dict(cohort["skipped"]),
            # every held-out row, so the control's calibration and its per-region and per-session scores can be
            # audited without refitting
            "held_out_predictions": {"predicted_positions": predicted, "no_behaviour": no_behaviour,
                                     "positions": positions, "regions": regions, "days": days,
                                     "sessions": cohort_sessions(cohort, days)},
            "settings_sha256": compute_settings_sha256(settings), "provenance": build_provenance(settings)}


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
    # Published atomically, as the unit files are: a fit interrupted mid-write never leaves a truncated
    # control where a complete one stood.
    with atomic_output_path(destination) as temporary, pathlib.Path(temporary).open("wb") as handle:
        pickle.dump(artifact, handle)


def upgrade_legacy_counts(record: dict) -> dict:
    """
    Description
    -----------
    Read an artifact that stores its counts under the legacy key ``n_calls`` under the current name.

    Renames ``n_calls`` to ``n_vocalizations`` at the top level and inside every ``per_day`` entry, and
    tags a fitted control that predates the categorical control with ``target = "manifold"``, the only
    control that existed then. A record that already carries the current names passes through
    unchanged, and the input is not mutated.

    Parameters
    ----------
    record (dict)
        A control artifact (:func:`fit_control_models`) or selection result
        (:func:`select_control_features`), old or new.

    Returns
    -------
    upgraded (dict)
        A shallow copy with every ``n_calls`` renamed to ``n_vocalizations``.
    """

    upgraded = dict(record)
    # Artifacts fitted before the categorical control existed were all torus controls.
    if "target" not in upgraded and "coefficients" in upgraded:
        upgraded["target"] = "manifold"
    if "n_calls" in upgraded:
        upgraded["n_vocalizations"] = upgraded.pop("n_calls")
    if "per_day" in upgraded:
        per_day = {}
        for day, stored_scores in upgraded["per_day"].items():
            scores = dict(stored_scores)
            if "n_calls" in scores:
                scores["n_vocalizations"] = scores.pop("n_calls")
            per_day[day] = scores
        upgraded["per_day"] = per_day
    return upgraded


def load_control_model(settings: dict, target: str) -> dict:
    """
    Description
    -----------
    Read the fitted control for one target and check it against the settings it is about to serve.

    A control out of step with the settings would change every added score without changing any
    reported configuration value, so a feature or target mismatch raises. The path and the frozen
    features are read at the target's key.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.
    target (str)
        ``"manifold"``, ``"multinomial"`` or ``"onset"``.

    Returns
    -------
    artifact (dict)
        As written by :func:`fit_control_models`, with legacy ``n_calls`` keys read as
        ``n_vocalizations`` (:func:`upgrade_legacy_counts`).
    """

    check_control_target(target)
    path = pathlib.Path(configure_path(
        settings["paths"]["behaviour_control_model_path"][target]))
    if not path.exists():
        msg = (f"behaviour control model not found at {path}; fit it once with the behaviour-control "
               f"dispatcher before running the nested decoding.")
        raise FileNotFoundError(msg)
    with path.open("rb") as handle:
        artifact = upgrade_legacy_counts(pickle.load(handle))

    if artifact["target"] != target:
        msg = (f"behaviour control target mismatch: {path.name} is a {artifact['target']!r} control, but "
               f"it was asked for as the {target!r} control.")
        raise ValueError(msg)
    frozen = list(settings["nested_vocal_decoding"]["reduced_model_features"][target])
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
    Predicted torus positions for a unit's USVs, from the fold that did not see that unit's day.

    Parameters
    ----------
    artifact (dict)
        From :func:`load_control_model`.
    held_out_day (str)
        The unit's recording date, as the eight-character day key.
    design (np.ndarray)
        ``(n, n_features * n_lags)`` histories for the unit's USVs, in the artifact's feature order.
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
    The paired per-day improvement of a candidate over the incumbent, and its standard error.

    Days differ far more in absolute score than a candidate improves on them, so the standard error is
    taken over the within-day differences, which removes the day effect.

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
    Held-out-by-day scores of one candidate feature set, reading the design by memory map.

    The design runs to gigabytes, so workers map the ``.npy`` file instead of receiving a pickled copy;
    this behaves the same whether processes are spawned or forked.

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

    Greedy by held-out macro score: a candidate is accepted only when its paired gain minus that gain's
    standard error is above zero (:func:`paired_gain`), and selection stops at the first rejection.

    An ``external_reference`` feature set is scored under the identical setup and recorded beside the
    result. The selection sees every day, so it carries some optimism; a feature set chosen elsewhere
    does not, and the gap between the two bounds that optimism.

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
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=worker_process_context()) as pool:
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
            "n_vocalizations": int(positions.shape[0]), "session_ids": list(cohort["session_ids"]),
            "skipped": dict(cohort["skipped"]), "settings_sha256": compute_settings_sha256(settings)}


EMBEDDING_COLUMNS = ("control.cos1", "control.sin1", "control.cos2", "control.sin2")


def apply_behaviour_control(design: dict, artifact: dict, held_out_day: str,
                            settings: dict) -> tuple:
    """
    Description
    -----------
    Replace a unit's raw behaviour block with the cohort control's prediction, and return the lag count
    the result must be scored at.

    The nesting is unchanged (reduced is behaviour, full is behaviour plus the neuron's column); only
    the representation changes, to the four sine/cosine columns of the predicted position, so only four
    behaviour coefficients are estimated on the unit. The lag count (always 1) is returned
    with the design because every scorer derives the design's shape from ``n_lags``.

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


def category_control_basis(settings: dict, n_lags: int) -> tuple[np.ndarray, dict]:
    """
    Description
    -----------
    The categorical control's lag basis and the penalty settings its fit uses.

    Resolved by the behavioural modelling's own ``resolve_temporal_basis`` from
    ``nested_vocal_decoding.category_control``, a copy of that pipeline's multinomial settings block,
    so the control is the same model the categorical feature selection scored: each feature's history
    is projected onto the same B-splines with the same lag-varying ridge.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.

    Returns
    -------
    basis, block (tuple)
        ``(n_lags, n_splines)`` projection basis, and the effective settings block
        (``lambda_smooth_fixed``, ``l2_reg_fixed``, ``smoothness_derivative_order``,
        ``smoothness_reflective_edges`` and the solver settings).
    """

    basis, block = resolve_temporal_basis(settings["nested_vocal_decoding"]["category_control"], int(n_lags))
    if basis is None:
        msg = ("nested_vocal_decoding.category_control.temporal_basis.type must be 'bspline'; the "
               "categorical control is the behavioural modelling's spline-basis model.")
        raise ValueError(msg)
    return basis, block


def project_category_design(design: np.ndarray, n_features: int, basis: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Project a feature-major lagged design onto the categorical control's basis, feature by feature,
    in the column order the behavioural modelling uses.

    Parameters
    ----------
    design (np.ndarray)
        ``(n, n_features * n_lags)`` histories, feature-major.
    n_features (int)
        Number of features.
    basis (np.ndarray)
        ``(n_lags, n_splines)`` basis.

    Returns
    -------
    projected (np.ndarray)
        ``(n, n_features * n_splines)`` projected design.
    """

    n_lags = basis.shape[0]
    if design.shape[1] != n_features * n_lags:
        msg = (f"design has {design.shape[1]} columns, expected {n_features} features x {n_lags} lags.")
        raise ValueError(msg)
    return np.hstack([project_history_onto_basis(design[:, index * n_lags:(index + 1) * n_lags], basis)
                      for index in range(n_features)])


def category_control_estimator(n_features: int, basis: np.ndarray, block: dict,
                               random_state: int) -> SmoothMultinomialLogisticRegression:
    """
    Description
    -----------
    An unfitted categorical control, configured as the behavioural modelling's multinomial fit.

    Parameters
    ----------
    n_features (int)
        Number of behaviour features.
    basis (np.ndarray)
        The projection basis; its column count is the number of coefficients per feature.
    block (dict)
        The effective settings block from :func:`category_control_basis`.
    random_state (int)
        Seed of the fit.

    Returns
    -------
    model (SmoothMultinomialLogisticRegression)
        The unfitted estimator.
    """

    return SmoothMultinomialLogisticRegression(
        n_features=int(n_features), n_time_bins=int(basis.shape[1]),
        lambda_smooth=float(block["lambda_smooth_fixed"]), l2_reg=float(block["l2_reg_fixed"]),
        smoothness_derivative_order=int(block["smoothness_derivative_order"]),
        smoothness_reflective_edges=bool(block["smoothness_reflective_edges"]),
        focal_gamma=float(block["focal_loss_gamma"]), uniform_class_weights=False,
        learning_rate=float(block["learning_rate"]), max_iter=int(block["max_iter"]),
        tol=float(block["tol"]), random_state=int(random_state),
        _use_lax_loop=bool(block["use_lax_loop"]))


def category_labels(cohort: dict) -> np.ndarray:
    """
    Description
    -----------
    The categorical control's target: the cohort's acoustic category per USV.

    The cohort assembly carries the ``vocalizations.category_column`` labels, the same column
    the per-unit categorical decoder reads, so the control predicts exactly the category system the
    neuron is decoded on.

    Parameters
    ----------
    cohort (dict)
        From :func:`assemble_cohort`.

    Returns
    -------
    labels (np.ndarray)
        Integer category label per USV.
    """

    return np.asarray(cohort["regions"]).astype(np.int64)


def macro_one_versus_rest_auc(probabilities: np.ndarray, labels: np.ndarray,
                              classes: np.ndarray) -> float:
    """
    Description
    -----------
    Mean over categories of the one-versus-rest AUROC, the behavioural modelling's selection metric.

    Parameters
    ----------
    probabilities (np.ndarray)
        ``(n, n_classes)`` predicted probabilities, columns in ``classes`` order.
    labels (np.ndarray)
        True category per USV.
    classes (np.ndarray)
        Category of each column.

    Returns
    -------
    auc (float)
        Macro AUROC over the categories present with both positives and negatives; NaN if none.
    """

    values = []
    for column, category in enumerate(classes):
        positive = labels == category
        if positive.all() or not positive.any():
            continue
        scores = probabilities[:, column]
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(scores.size, dtype=np.float64)
        ranks[order] = np.arange(1, scores.size + 1)
        # Average ranks over ties, so the AUROC is exact for tied scores.
        _unique, inverse = np.unique(scores, return_inverse=True)
        ranks = np.bincount(inverse, weights=ranks)[inverse] / np.bincount(inverse)[inverse]
        n_positive = int(positive.sum())
        n_negative = scores.size - n_positive
        values.append((ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
                      / (n_positive * n_negative))
    return float(np.mean(values)) if values else float("nan")


def fit_category_control_models(cohort: dict, feature_names: list, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    Fit the categorical control once per held-out recording day and return the artifact payload.

    The estimator is the behavioural modelling's multinomial logistic regression on the B-spline lag
    basis (:func:`category_control_estimator`), fitted on every day but the held-out one. Only each
    fold's ``coef_``, ``intercept_``, ``log_priors_``, ``class_weights_``, ``effective_log_priors_`` and
    ``classes_`` are kept, since those are all a prediction needs.

    Two held-out scores are reported, per day and pooled, neither of which the per-unit statistic
    uses: the macro one-versus-rest AUROC of the base-rate-free probabilities (the behavioural
    modelling's metric, chance 0.5), and the mean log-probability of the true category under the
    fitted probabilities minus that under the training days' category frequencies (nats per USV,
    zero for a control that knows nothing). The model is fitted with a focal loss, so its
    probabilities are not calibrated and the second score is only indicative; the per-unit prior
    tunes its own concentration (``neural_category_decoding.category_behaviour_tilts``).

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
        ``target`` (``"multinomial"``), ``features``, ``n_lags``, ``classes``, ``coefficients`` (day to
        ``{coef, intercept, log_priors}``), ``held_out_auc``, ``held_out_log_score_gain``, ``per_day``,
        counts, ``converged`` per day and ``settings_sha256``.
    """

    labels = category_labels(cohort)
    days = cohort["days"]
    classes = np.unique(labels)
    columns = feature_columns(cohort["feature_names"], feature_names, n_lags)
    basis, block = category_control_basis(settings, n_lags)
    design = project_category_design(cohort["design"][:, columns].astype(np.float64),
                                     len(feature_names), basis).astype(np.float32)

    coefficients, per_day = {}, {}
    balanced = np.full((labels.size, classes.size), np.nan)
    gain = np.full(labels.size, np.nan)
    for held_out in np.unique(days):
        test = np.flatnonzero(days == held_out)
        train = np.flatnonzero(days != held_out)
        missing = np.setdiff1d(classes, np.unique(labels[train]))
        if missing.size:
            msg = (f"categories {missing.tolist()} occur only on day {held_out}; the control held out "
                   f"by that day cannot predict them.")
            raise ValueError(msg)
        model = category_control_estimator(len(feature_names), basis, block, int(block["random_state"]))
        model.fit(design[train], labels[train])
        if not np.array_equal(model.classes_, classes):
            msg = f"fold {held_out} fitted classes {model.classes_.tolist()}, expected {classes.tolist()}."
            raise ValueError(msg)
        balanced[test] = model.predict_proba(design[test], balanced=True)
        fitted = model.predict_proba(design[test], balanced=False)
        frequency = np.bincount(np.searchsorted(classes, labels[train]),
                                minlength=classes.size) / float(train.size)
        truth = np.searchsorted(classes, labels[test])
        gain[test] = (np.log(np.maximum(fitted[np.arange(test.size), truth], 1e-300))
                      - np.log(frequency[truth]))
        coefficients[str(held_out)] = {"coef": np.asarray(model.coef_).copy(),
                                       "intercept": np.asarray(model.intercept_).copy(),
                                       "log_priors": np.asarray(model.log_priors_).copy(),
                                       "class_weights": np.asarray(model.class_weights_).copy(),
                                       "effective_log_priors": np.asarray(model.effective_log_priors_).copy(),
                                       "converged": bool(model.converged_),
                                       "n_iter": int(model.n_iter_)}
        per_day[str(held_out)] = {
            "n_vocalizations": int(test.size),
            "auc": macro_one_versus_rest_auc(balanced[test], labels[test], classes),
            "log_score_gain": float(np.mean(gain[test])),
            "class_counts": np.bincount(np.searchsorted(classes, labels[test]), minlength=classes.size).tolist(),
            # one-versus-rest AUROC per category (NaN where the day holds none of it, or only it)
            "auc_per_class": [area_under_roc(balanced[test][:, index], (labels[test] == value).astype(np.float64))
                              if np.any(labels[test] == value) and not np.all(labels[test] == value)
                              else float("nan") for index, value in enumerate(classes)]}

    return {"target": "multinomial", "features": list(feature_names), "n_lags": int(n_lags),
            "classes": classes, "coefficients": coefficients,
            "held_out_auc": macro_one_versus_rest_auc(balanced, labels, classes),
            "held_out_log_score_gain": float(np.mean(gain)),
            "per_day": per_day, "n_vocalizations": int(labels.size),
            "session_ids": list(cohort["session_ids"]), "skipped": dict(cohort["skipped"]),
            "held_out_predictions": {"probabilities": balanced, "log_score_gain": gain, "labels": labels,
                                     "days": days, "sessions": cohort_sessions(cohort, days)},
            "settings_sha256": compute_settings_sha256(settings), "provenance": build_provenance(settings)}


def restore_fitted_control(model: SmoothMultinomialLogisticRegression, fold: dict, classes: np.ndarray) -> None:
    """
    Description
    -----------
    Put a stored fold's fitted state back on an unfitted control estimator, so it predicts as the fold did.

    A fold record carries ``coef``, ``intercept``, ``log_priors``, ``class_weights`` and
    ``effective_log_priors``; the last is what ``predict_proba(balanced=True)`` subtracts, so a record
    written before it was stored (a fit whose balanced prediction subtracted the plain class frequencies)
    is refused rather than applied with a tilted prior.

    Parameters
    ----------
    model (SmoothMultinomialLogisticRegression)
        The unfitted estimator of the fold's target.
    fold (dict)
        The stored fold record (``artifact["coefficients"][day]``).
    classes (np.ndarray)
        The artifact's class labels, in the layout of ``coef``.

    Returns
    -------
    None
    """

    if "effective_log_priors" not in fold:
        msg = ("the control's fold record has no 'effective_log_priors': it was fitted before the balanced "
               "prediction subtracted the prior a class-weighted fit absorbs; refit the control.")
        raise KeyError(msg)
    model.coef_ = np.asarray(fold["coef"])
    model.intercept_ = np.asarray(fold["intercept"])
    model.log_priors_ = np.asarray(fold["log_priors"])
    model.class_weights_ = np.asarray(fold["class_weights"])
    model.effective_log_priors_ = np.asarray(fold["effective_log_priors"])
    model.classes_ = classes


def category_control_evidence(artifact: dict, held_out_day: str, design: np.ndarray,
                              settings: dict) -> np.ndarray:
    """
    Description
    -----------
    The categorical control's base-rate-free category probabilities for a unit's USVs, from the fold
    that did not see that unit's day.

    ``predict_proba(balanced=True)`` removes the prior the fit absorbed from the logits (the training
    category frequencies as weighted by the class weights, ``effective_log_priors_``), so the result
    carries the behaviour's evidence about the category and not the cohort's category frequencies;
    the per-unit prior supplies its own frequencies.

    Parameters
    ----------
    artifact (dict)
        From :func:`load_control_model`, a ``"multinomial"`` control.
    held_out_day (str)
        The unit's recording date, as the eight-character day key.
    design (np.ndarray)
        ``(n, n_features * n_lags)`` histories for the unit's USVs, in the artifact's feature order.
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    evidence (np.ndarray)
        ``(n, n_classes)`` probabilities, columns in ``artifact["classes"]`` order.
    """

    if artifact["target"] != "multinomial":
        msg = f"category_control_evidence needs a multinomial control; got {artifact['target']!r}."
        raise ValueError(msg)
    if held_out_day not in artifact["coefficients"]:
        msg = (f"the behaviour control carries no fold for day {held_out_day}; it was fitted over "
               f"{sorted(artifact['coefficients'])}. A unit's own day must be held out.")
        raise KeyError(msg)
    n_features = len(artifact["features"])
    expected = n_features * artifact["n_lags"]
    if design.shape[1] != expected:
        msg = (f"the unit's behaviour block has {design.shape[1]} columns but the control was fitted on "
               f"{n_features} features x {artifact['n_lags']} lags = {expected}.")
        raise ValueError(msg)
    basis, block = category_control_basis(settings, artifact["n_lags"])
    fold = artifact["coefficients"][held_out_day]
    model = category_control_estimator(n_features, basis, block, int(block["random_state"]))
    restore_fitted_control(model, fold, np.asarray(artifact["classes"]))
    projected = project_category_design(np.asarray(design, dtype=np.float64), n_features, basis)
    return model.predict_proba(projected.astype(np.float32), balanced=True)


def session_onset_design(session_directory: str, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    One session's onset-control rows: the recorded animal's clean USV onsets (label 1) and the quiet
    tiles (label 0), each with the kinematic history behind it.

    Both classes are the per-unit occurrence decoder's own (``neural_design_assembly.assemble_unit_vocal_events``):
    a USV is kept when nothing from the guard set intrudes on its pre-onset window
    (``vocalizations.spike_window.pre_offset_seconds``), and the tiles are the windows of the spike
    window's width tiled through the gaps between guard-banded vocalizations of every emitter
    (:func:`neural_vocal_occurrence.quiet_tile_edges`, with the kinematic encoding's
    ``history_pre_seconds`` and ``clean_post_seconds``), capped per session at
    ``vocal_decoding.max_quiet_tiles_per_session`` by a draw seeded with ``inference.shuffle_seed``.
    A USV's history ends at the frame before its onset (``onset_history_anchor_frames``); a tile's
    history ends at the frame before ``tile_edge + pre_offset_seconds``, the tile's centre, so every
    window's history ends at its centre, positive or tile, and the classes differ only in the USV at
    the end. Rows whose history runs off the start of the session are dropped.

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
        ``session_id``, ``design`` (float32, USVs first then tiles), ``labels`` (1 / 0),
        ``feature_names``, ``n_kept``, ``n_positive``, ``n_tiles`` and ``n_tiles_before_cap``; or
        ``session_id`` and ``reason`` when the session contributes nothing.
    """

    session_id = pathlib.Path(session_directory).name
    data_root = settings["paths"]["data_root"]
    vocal_settings = settings["vocalizations"]
    window = vocal_settings["spike_window"]
    encoding = settings["kinematic_encoding"]
    minimum = settings["cohort"]["min_emitter_usvs_per_session"]
    tiles_cap = settings["vocal_decoding"]["max_quiet_tiles_per_session"]
    track_names, frame_rate, n_frames = session_timebase(data_root, session_id)
    if len(track_names) != 2:
        return {"session_id": session_id, "reason": f"{len(track_names)} tracked animals"}
    recorded = track_names[0]
    try:
        usv = load_session_usvs(data_root, session_id)
    except FileNotFoundError as error:
        return {"session_id": session_id, "reason": str(error)}
    guard = filter_vocalizations(usv, vocal_settings, "guard")
    all_starts = guard["start"].to_numpy().astype(np.float64)
    all_stops = guard["stop"].to_numpy().astype(np.float64)
    analysed = filter_vocalizations(usv, vocal_settings, "analysed")
    focal_names = emitter_names(track_names, recorded, vocal_settings["vocal_emitter"])
    focal = analysed.filter(analysed["emitter"].is_in(focal_names)) if focal_names else analysed
    if focal.height < minimum:
        return {"session_id": session_id, "reason": f"{focal.height} emitter vocalizations"}
    starts = focal["start"].to_numpy().astype(np.float64)
    frames, camera, _names, _suffixes = build_zscored_feature_frames(
        session_dirs=[session_directory], kinematic_features=settings["kinematic_features"],
        recorded_mouse_id=recorded)
    frame_df = frames[session_id]
    feature_series = np.nan_to_num(frame_df.to_numpy().astype(np.float64), nan=0.0)
    fps = camera[session_id]
    n_series = feature_series.shape[0]
    clean = (clean_window_mask(starts, all_starts, all_stops, window["pre_offset_seconds"])
             if settings["vocal_decoding"]["require_clean_prevocal_bool"]
             else np.ones(starts.size, dtype=bool))
    # The unit side builds its USV windows from its USV events, which drop a USV without a torus position,
    # so the control drops the same USVs: it is trained on exactly the onsets the units score.
    has_position = np.isfinite(focal.select(vocal_settings["usv_manifold_column_names"]).to_numpy()
                               .astype(np.float64)).all(axis=1)
    onset_anchors = onset_history_anchor_frames(starts, fps)
    keep_onsets = clean & has_position & (onset_anchors >= n_lags - 1) & (onset_anchors < n_series)
    duration = float(n_frames) / float(frame_rate)
    tile_edges = quiet_tile_edges(all_starts, all_stops, duration, window["width_seconds"],
                                  encoding["history_pre_seconds"], encoding["clean_post_seconds"])
    n_tiles_before_cap = int(tile_edges.size)
    if tiles_cap is not None and tile_edges.size > tiles_cap:
        generator = np.random.default_rng(int(settings["inference"]["shuffle_seed"]))
        tile_edges = np.sort(generator.choice(tile_edges, int(tiles_cap), replace=False))
    tile_anchors = onset_history_anchor_frames(tile_edges + window["pre_offset_seconds"], fps)
    keep_tiles = (tile_anchors >= n_lags - 1) & (tile_anchors < n_series)
    n_positive, n_tiles = int(keep_onsets.sum()), int(keep_tiles.sum())
    if n_positive == 0 or n_tiles == 0:
        return {"session_id": session_id,
                "reason": f"{n_positive} clean onsets and {n_tiles} quiet tiles with a full history"}
    design = np.vstack([lagged_design(feature_series, onset_anchors[keep_onsets], n_lags),
                        lagged_design(feature_series, tile_anchors[keep_tiles], n_lags)]).astype(np.float32)
    return {"session_id": session_id, "design": design,
            "labels": np.concatenate([np.ones(n_positive), np.zeros(n_tiles)]),
            "feature_names": list(frame_df.columns), "n_kept": n_positive + n_tiles,
            "n_positive": n_positive, "n_tiles": n_tiles, "n_tiles_before_cap": n_tiles_before_cap}


def _assemble_one_onset(arguments: tuple) -> dict:
    """
    Description
    -----------
    Worker shim so the per-session onset assembly can be mapped over a process pool.

    Parameters
    ----------
    arguments (tuple)
        ``(session_directory, settings, n_lags)``.

    Returns
    -------
    block (dict)
        As :func:`session_onset_design`.
    """

    session_directory, settings, n_lags = arguments
    return session_onset_design(session_directory, settings, n_lags)


def assemble_onset_cohort(session_directories: list, settings: dict, n_lags: int,
                          n_workers: int) -> dict:
    """
    Description
    -----------
    Every listed session's clean USV onsets and quiet tiles with their histories, stacked, with the
    recording day recorded per row.

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
        ``design``, ``labels``, ``days``, ``session_ids``, ``feature_names``, ``per_session`` (the
        positive and tile counts per session) and ``skipped`` (session id to reason).

    Raises
    ------
    ValueError
        No session contributes, or two sessions disagree on the feature set.
    """

    blocks, skipped = [], {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=worker_process_context()) as pool:
        for block in pool.map(_assemble_one_onset, [(directory, settings, n_lags)
                                                    for directory in session_directories]):
            if "reason" in block:
                skipped[block["session_id"]] = block["reason"]
            else:
                blocks.append(block)
    if not blocks:
        msg = f"no session contributes to the onset control; skipped: {skipped}"
        raise ValueError(msg)
    feature_names = blocks[0]["feature_names"]
    for block in blocks[1:]:
        if block["feature_names"] != feature_names:
            msg = (f"{block['session_id']} assembled features {block['feature_names']}, "
                   f"{blocks[0]['session_id']} {feature_names}.")
            raise ValueError(msg)
    return {"design": np.vstack([block["design"] for block in blocks]),
            "labels": np.concatenate([block["labels"] for block in blocks]),
            "days": np.concatenate([np.full(block["n_kept"], block["session_id"][:8]) for block in blocks]),
            "sessions": np.concatenate([np.full(block["n_kept"], block["session_id"]) for block in blocks]),
            "session_ids": [block["session_id"] for block in blocks],
            "feature_names": feature_names,
            "per_session": {block["session_id"]: {"n_positive": block["n_positive"], "n_tiles": block["n_tiles"],
                                                  "n_tiles_before_cap": block["n_tiles_before_cap"]}
                            for block in blocks},
            "skipped": skipped}


def onset_control_basis(settings: dict, n_lags: int) -> tuple[np.ndarray, dict]:
    """
    Description
    -----------
    The onset control's B-spline lag basis and effective penalty block, resolved by the behavioural
    modelling's ``resolve_temporal_basis`` from ``nested_vocal_decoding.onset_control``.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.

    Returns
    -------
    basis (np.ndarray)
        ``(n_lags, n_splines)`` projection basis.
    block (dict)
        The effective settings block.
    """

    basis, block = resolve_temporal_basis(settings["nested_vocal_decoding"]["onset_control"], int(n_lags))
    if basis is None:
        msg = "nested_vocal_decoding.onset_control.temporal_basis.type must be 'bspline'."
        raise ValueError(msg)
    return basis, block


def onset_control_estimator(n_features: int, basis: np.ndarray, block: dict,
                            random_state: int) -> SmoothMultinomialLogisticRegression:
    """
    Description
    -----------
    The unfitted two-class estimator of the onset control: the category control's model with two
    classes, so a smooth logistic regression on the basis coefficients. Class imbalance (one tile cap
    against the session's USVs) is handled inside the loss by the model's class weights; the
    predictions the per-unit test reads are base-rate-free in any case.

    Parameters
    ----------
    n_features (int)
        Number of behaviour features.
    basis (np.ndarray)
        The projection basis; its column count is the number of coefficients per feature.
    block (dict)
        The effective settings block from :func:`onset_control_basis`.
    random_state (int)
        Seed of the fit.

    Returns
    -------
    model (SmoothMultinomialLogisticRegression)
        The unfitted estimator.
    """

    return SmoothMultinomialLogisticRegression(
        n_features=int(n_features), n_time_bins=int(basis.shape[1]),
        lambda_smooth=float(block["lambda_smooth_fixed"]), l2_reg=float(block["l2_reg_fixed"]),
        smoothness_derivative_order=int(block["smoothness_derivative_order"]),
        smoothness_reflective_edges=bool(block["smoothness_reflective_edges"]),
        focal_gamma=float(block["focal_loss_gamma"]), uniform_class_weights=False,
        learning_rate=float(block["learning_rate"]), max_iter=int(block["max_iter"]),
        tol=float(block["tol"]), random_state=int(random_state),
        _use_lax_loop=bool(block["use_lax_loop"]))


def balanced_log_score_gain(log_odds: np.ndarray, labels: np.ndarray) -> float:
    """
    Description
    -----------
    The class-balanced log-score of base-rate-free onset log-odds over the equal prior: the mean,
    over the two classes, of each class's mean ``log p(true class) - log 1/2``, in nats per window;
    zero for a control that knows nothing.

    Parameters
    ----------
    log_odds (np.ndarray)
        ``(n,)`` log-odds of onset against quiet.
    labels (np.ndarray)
        ``(n,)`` labels, 1 for a USV onset and 0 for a quiet tile.

    Returns
    -------
    gain (float)
        The class-balanced gain; NaN when a class is absent.
    """

    log_odds = np.asarray(log_odds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64) > 0.5
    log_p1 = -np.logaddexp(0.0, -log_odds)
    log_p0 = -np.logaddexp(0.0, log_odds)
    if not labels.any() or labels.all():
        return float("nan")
    positive = float(np.mean(log_p1[labels]) - np.log(0.5))
    negative = float(np.mean(log_p0[~labels]) - np.log(0.5))
    return 0.5 * (positive + negative)


def fit_onset_control_models(cohort: dict, feature_names: list, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    Fit the onset control once per held-out recording day and return the artifact payload.

    The estimator is the two-class model of :func:`onset_control_estimator`, fitted on every day but
    the held-out one on the cohort's clean USV onsets against quiet tiles. Only each fold's ``coef_``,
    ``intercept_``, ``log_priors_``, ``class_weights_``, ``effective_log_priors_`` and ``classes_`` are
    kept. Two held-out scores are reported, per
    day and pooled, neither of which the per-unit statistic uses: the AUROC of the base-rate-free
    onset probability (chance 0.5) and the class-balanced log-score gain over the equal prior
    (:func:`balanced_log_score_gain`). The per-unit prior tunes its own concentration on the control's
    log-odds (``neural_vocal_occurrence.onset_behaviour_tilts``), so neither score is a calibration.

    Parameters
    ----------
    cohort (dict)
        From :func:`assemble_onset_cohort`.
    feature_names (list)
        The control's features, a subset of the cohort's canonical feature list.
    settings (dict)
        The whole neural-modelling settings dict.
    n_lags (int)
        History length in frames.

    Returns
    -------
    artifact (dict)
        ``target`` (``"onset"``), ``features``, ``n_lags``, ``classes`` (``[0, 1]``), ``coefficients``
        (day to ``{coef, intercept, log_priors, converged, n_iter}``), ``held_out_auc``,
        ``held_out_log_score_gain``, ``per_day``, counts, ``session_ids``, ``skipped`` and
        ``settings_sha256``.

    Raises
    ------
    ValueError
        A held-out day leaves a training set with one class only.
    """

    labels = np.asarray(cohort["labels"], dtype=np.int64)
    days = cohort["days"]
    classes = np.array([0, 1])
    columns = feature_columns(cohort["feature_names"], feature_names, n_lags)
    basis, block = onset_control_basis(settings, n_lags)
    design = project_category_design(cohort["design"][:, columns].astype(np.float64),
                                     len(feature_names), basis).astype(np.float32)
    coefficients, per_day = {}, {}
    log_odds = np.full(labels.size, np.nan)
    for held_out in np.unique(days):
        test = np.flatnonzero(days == held_out)
        train = np.flatnonzero(days != held_out)
        if np.unique(labels[train]).size != 2:
            msg = f"holding out day {held_out} leaves training rows of one class only."
            raise ValueError(msg)
        model = onset_control_estimator(len(feature_names), basis, block, int(block["random_state"]))
        model.fit(design[train], labels[train])
        if not np.array_equal(model.classes_, classes):
            msg = f"fold {held_out} fitted classes {model.classes_.tolist()}, expected [0, 1]."
            raise ValueError(msg)
        balanced = model.predict_proba(design[test], balanced=True)
        log_odds[test] = (np.log(np.maximum(balanced[:, 1], 1e-300))
                          - np.log(np.maximum(balanced[:, 0], 1e-300)))
        coefficients[str(held_out)] = {"coef": np.asarray(model.coef_).copy(),
                                       "intercept": np.asarray(model.intercept_).copy(),
                                       "log_priors": np.asarray(model.log_priors_).copy(),
                                       "class_weights": np.asarray(model.class_weights_).copy(),
                                       "effective_log_priors": np.asarray(model.effective_log_priors_).copy(),
                                       "converged": bool(model.converged_),
                                       "n_iter": int(model.n_iter_)}
        per_day[str(held_out)] = {
            "n_vocalizations": int((labels[test] == 1).sum()), "n_tiles": int((labels[test] == 0).sum()),
            "auc": area_under_roc(balanced[:, 1], labels[test].astype(np.float64)),
            "log_score_gain": balanced_log_score_gain(log_odds[test], labels[test])}
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    return {"target": "onset", "features": list(feature_names), "n_lags": int(n_lags),
            "classes": classes, "coefficients": coefficients,
            "held_out_auc": area_under_roc(probability, labels.astype(np.float64)),
            "held_out_log_score_gain": balanced_log_score_gain(log_odds, labels),
            "per_day": per_day, "n_vocalizations": int((labels == 1).sum()), "n_tiles": int((labels == 0).sum()),
            "per_session": dict(cohort["per_session"]),
            "session_ids": list(cohort["session_ids"]), "skipped": dict(cohort["skipped"]),
            "held_out_predictions": {"log_odds": log_odds, "labels": labels, "days": days,
                                     "sessions": cohort_sessions(cohort, days)},
            "settings_sha256": compute_settings_sha256(settings), "provenance": build_provenance(settings)}


def onset_control_log_odds(artifact: dict, held_out_day: str, design: np.ndarray, settings: dict) -> np.ndarray:
    """
    Description
    -----------
    The onset control's base-rate-free log-odds of onset against quiet for a unit's windows, from the
    fold that did not see that unit's day.

    ``predict_proba(balanced=True)`` removes the prior the fit absorbed from the logits (the training
    class frequencies as weighted by the class weights, ``effective_log_priors_``; under the softened
    inverse-frequency weights that is half the log USV-to-tile ratio, not all of it), so the log-odds
    carry the behaviour's evidence and not the cohort's USV-to-tile ratio; the per-unit test supplies
    its own equal prior and tunes how strongly the evidence tilts it.

    Parameters
    ----------
    artifact (dict)
        From :func:`load_control_model`, an ``"onset"`` control.
    held_out_day (str)
        The unit's recording date, as the eight-character day key.
    design (np.ndarray)
        ``(n, n_features * n_lags)`` histories for the unit's windows, in the artifact's feature order.
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    log_odds (np.ndarray)
        ``(n,)`` log-odds of onset against quiet.

    Raises
    ------
    ValueError
        The artifact is not an onset control, or the design's width disagrees with it.
    KeyError
        The control carries no fold for the unit's day.
    """

    if artifact["target"] != "onset":
        msg = f"onset_control_log_odds needs an onset control; got {artifact['target']!r}."
        raise ValueError(msg)
    if held_out_day not in artifact["coefficients"]:
        msg = (f"the behaviour control carries no fold for day {held_out_day}; it was fitted over "
               f"{sorted(artifact['coefficients'])}. A unit's own day must be held out.")
        raise KeyError(msg)
    n_features = len(artifact["features"])
    expected = n_features * artifact["n_lags"]
    if design.shape[1] != expected:
        msg = (f"the unit's behaviour block has {design.shape[1]} columns but the control was fitted on "
               f"{n_features} features x {artifact['n_lags']} lags = {expected}.")
        raise ValueError(msg)
    basis, block = onset_control_basis(settings, artifact["n_lags"])
    fold = artifact["coefficients"][held_out_day]
    model = onset_control_estimator(n_features, basis, block, int(block["random_state"]))
    restore_fitted_control(model, fold, np.asarray(artifact["classes"]))
    projected = project_category_design(np.asarray(design, dtype=np.float64), n_features, basis)
    balanced = model.predict_proba(projected.astype(np.float32), balanced=True)
    return (np.log(np.maximum(balanced[:, 1], 1e-300)) - np.log(np.maximum(balanced[:, 0], 1e-300)))
