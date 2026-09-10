"""
@author: bartulem
Claim 3: does a neuron carry vocal output type BEYOND the behaviour that predicts it?

Two nested decoders of the same target, the upcoming call's position on the QLVM torus:

    reduced:  position ~ kinematics
    full:     position ~ kinematics + prevocal spike count

and the statistic is the out-of-sample added ``vm_logscore``, full minus reduced. A PURE RELAY, whose
spiking is a function of the behaviour, adds nothing the kinematics did not already carry and fails.
A unit that adds something passes -- and what it adds is either genuinely internal, or behaviour that
was not measured. Claim 3 cannot separate those two, which is why the paper says "beyond MEASURED
behaviour" rather than "computes the transformation", and why measuring behaviour more completely
makes this test STRONGER rather than weaker.

The asymmetry is deliberate and runs conservative. The behaviour block is a rich 4 s history of five
features -- 3,000 columns -- and the neuron is ONE column, its 50 ms prevocal count. The neuron must
earn its place against a control as complete as we can make it, because a weak control would let a
unit pass by proxying behaviour the control missed. The cost is power, not validity: contribution
that is nonlinear in the count, or that lives in the time course within the window, cannot be
expressed, so the transformation count is a lower bound.

Three things are settled and should not be re-derived.

The reduced model is P1's SELECTED five features, not all nineteen. The selection MEASURED that the
other fourteen add no position information, and replacing a measurement with an assumption is the
wrong direction. Its `final_model_features` is read from P1's own result file and asserted against
the frozen list in settings, so a re-run of P1 fails loudly here instead of silently redefining what
"beyond behaviour" means after the counts are banked.

It is scored under MACRO von Mises -- the objective the control was SELECTED under. Micro would leave
the control mildly suboptimal for the test (its first-selected feature, `self.neck_elevation`, HURTS
the micro score), which biases claim 3 toward passing: the wrong direction for a positive claim.

The neuron enters as a spike COUNT, not 600 lags of binary train. A 4 s spike history spans ~20 prior
calls at the 195 ms median inter-USV interval, so it is largely the neuron's RESPONSES to earlier
calls -- and since calls within a bout correlate in type, claim 3 would pass on bout structure. The
behaviour block has the same 4 s window and the same bout structure, and that is accepted there
because behaviour is the CONTROL: a longer window makes it stronger and the test harder. The neuron
is the TESTED predictor, so its window is kept short.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
from scipy.linalg import block_diag

from ..modeling.manifold_metric import (
    inverse_region_frequency_weights,
    macro_von_mises_logscore,
)
from ..modeling.manifold_torus_regression import SmoothTorusManifoldRegression
from .neural_design_assembly import lagged_design
from .neural_when import counts_in_windows
from .shift_null_inference import shift_range_seconds


class NestedTorusRegression(SmoothTorusManifoldRegression):
    """
    Description
    -----------
    The torus regressor with a BLOCK-DIAGONAL smoothness penalty, so a design that mixes a temporal
    behaviour block with a handful of non-temporal neural columns is penalised correctly.

    The parent applies one identical ``D^T D`` block per feature across ``n_time_bins`` lags, which
    assumes every feature is a temporal filter. Claim 3's design is not like that: five behaviour
    features carry 600 lags each and want smoothing along that axis, while the neural column is a
    single number with no time axis at all and must receive ridge ONLY -- smoothing a one-element
    filter is meaningless, and the parent's operator is not even defined for it.

    Two constraints from the parent, both worked around here rather than fought:

    ``fit`` hard-checks that ``X`` has exactly ``n_features * n_time_bins`` columns, so this class
    reports the design as a single "feature" of ``total_columns`` bins. Those two numbers are used by
    nothing except the penalty builder, which is overridden.

    One scalar ``lambda_smooth`` multiplies the WHOLE penalty matrix, so any relative weighting
    between blocks has to be baked into the matrix returned here. The neural block is given a zero
    block, which is exactly the ruled policy: uniform ridge (``l2_reg`` reaches every column through
    the parent's ``+ l2_reg * I``), per-block smoothness.

    NOTE the penalty asymmetry this leaves, which is a KNOWN and accepted design choice rather than
    an oversight: a behaviour filter pays ``l2_reg`` plus ``lambda_smooth * D^T D`` over 600
    coefficients while the neuron pays ``l2_reg`` alone. The planned check is to report the added
    score at several neural-ridge values, turning the judgement call into a measurement.

    Parameters
    ----------
    n_behaviour_features (int)
        Number of behavioural features in the design; each contributes ``n_lags`` columns.
    n_lags (int)
        Lags per behavioural feature (the history length in modelling bins).
    n_neural_columns (int)
        Neural columns appended after the behaviour block; 0 gives the REDUCED model.
    lambda_smooth (float)
        Temporal-smoothness strength, applied to the behaviour blocks only.
    l2_reg (float)
        Ridge, applied uniformly to every column by the parent.
    smoothness_derivative_order (int)
        1 or 2; passed through to the parent's operator so the boundary handling matches P1 exactly.
    period (float)
        Torus period per axis.

    Returns
    -------
    estimator (NestedTorusRegression)
        An unfitted estimator.
    """

    def __init__(
            self,
            n_behaviour_features: int = 1,
            n_lags: int = 1,
            n_neural_columns: int = 0,
            lambda_smooth: float = 1.0,
            l2_reg: float = 0.01,
            smoothness_derivative_order: int = 1,
            period: float = 1.0,
    ):
        total_columns = int(n_behaviour_features) * int(n_lags) + int(n_neural_columns)
        super().__init__(n_features=1, n_time_bins=total_columns, lambda_smooth=lambda_smooth,
                         l2_reg=l2_reg, smoothness_derivative_order=smoothness_derivative_order,
                         metric="torus", period=period)
        self.n_behaviour_features = int(n_behaviour_features)
        self.n_lags = int(n_lags)
        self.n_neural_columns = int(n_neural_columns)

    def _smoothness_penalty(self) -> np.ndarray:
        """
        Description
        -----------
        Block-diagonal penalty: one temporal block per behavioural feature, zeros for the neural
        columns.

        The per-feature block is taken from a bare PARENT instance rather than rebuilt here, so the
        finite-difference operator -- including the reflective (Neumann) boundary rows the parent adds
        at order 2 -- stays identical to P1's by construction. Rebuilding it would be a second copy
        able to drift from the one the control was selected under.

        Parameters
        ----------

        Returns
        -------
        penalty (np.ndarray)
            ``(total_columns, total_columns)`` matrix, zero on the neural diagonal block.
        """

        template = SmoothTorusManifoldRegression(
            n_features=1, n_time_bins=self.n_lags,
            smoothness_derivative_order=self.smoothness_derivative_order,
            metric="torus", period=self.period)
        block = template._smoothness_penalty()   # the parent's documented subclass hook
        blocks = [block] * self.n_behaviour_features
        if self.n_neural_columns:
            blocks.append(np.zeros((self.n_neural_columns, self.n_neural_columns)))
        return block_diag(*blocks) if len(blocks) > 1 else blocks[0]


def reduced_model_features(settings: dict, require_selection_file: bool = True) -> list:
    """
    Description
    -----------
    The behaviour control's feature set: P1's selected model, READ from P1's own result file and
    ASSERTED against the frozen list in settings.

    Reading alone would let a re-run of P1 silently redefine what "beyond behaviour" means after
    transformation counts are banked. Freezing alone would let the control drift out of step with the
    selection it claims to be. Doing both catches the drift instead of absorbing it, and is the same
    "check rather than assume" the emitter and dyad resolutions use.

    The file is an INPUT to the run, so it lives in the ``data`` block beside the session lists.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict; ``data.behaviour_selection_result_path`` and
        ``nested_position_decoding.reduced_model_features`` are read.
    require_selection_file (bool)
        When False, a missing selection file is tolerated and the frozen list is returned unchecked --
        for running where the lab share is not mounted. A file that IS present is always checked.

    Returns
    -------
    features (list)
        The behaviour control's feature names, in P1's selection order.
    """

    frozen = list(settings["nested_position_decoding"]["reduced_model_features"])
    path = pathlib.Path(settings["data"]["behaviour_selection_result_path"])
    if not path.exists():
        if require_selection_file:
            msg = (f"behaviour selection result not found at {path}; it defines claim 3's reduced "
                   f"model. Pass require_selection_file=False to run on the frozen list alone.")
            raise FileNotFoundError(msg)
        return frozen

    with path.open("rb") as handle:
        selection = pickle.load(handle)
    selected = list(selection["steps"][-1]["final_model_features"])
    if selected != frozen:
        msg = (f"behaviour selection drift: {path.name} ends at {selected}, but settings freeze "
               f"{frozen}. Claim 3's control would silently change meaning; update the settings "
               f"deliberately if P1 was re-run.")
        raise ValueError(msg)
    return selected


def nested_design(events: dict, per_session: dict, feature_names: list, n_lags: int,
                  sigma_floor: float = 0.05) -> dict:
    """
    Description
    -----------
    Build claim 3's design: each event's 4 s behavioural history, plus the neuron as ONE column.

    Rows are CALLS, not frames. Each behavioural feature contributes ``n_lags`` columns -- its history
    ending at the call's onset frame -- so a row is one call with ``len(feature_names) * n_lags``
    behaviour columns followed by a single neural column. The common misstatement to avoid is that
    "every predictor is summarised to one value per call": the behaviour block is NOT summarised, it
    is 600 columns per feature. What differs between predictors is how many columns each contributes.

    **The neuron is a z-scored spike COUNT, not 600 lags of binary train.** The symmetric alternative
    was considered and rejected: a 4 s spike history spans about twenty prior calls at the 195 ms
    median inter-USV interval, so it is largely the neuron's RESPONSES to earlier calls, and since
    calls within a bout correlate in type the model would predict this call's position from previous
    calls' positions via the neuron -- claim 3 would pass on bout structure. The behaviour block has
    the same 4 s window and the same bout structure and is accepted there because behaviour is the
    CONTROL, where a longer window makes the test harder. Within the 50 ms window a count rather than
    per-frame binary, because 50 ms at 150 fps is 7.5 frames (so per-frame needs the window rounded or
    the last bin ragged), the data are sparse (cl0401: 2.16 spikes per window over 8 frame-columns),
    and decisively because CLAIM 2 ALREADY ENCODES THE NEURON AS A COUNT -- encoding it differently in
    the two claims of one conjunction would break the nesting.

    The count is z-scored PER SESSION, the same correction claim 2's WHEN axis applies, because this
    is a least-squares fit on a continuous predictor where standardising the input is available. (The
    WHAT axis cannot do that -- a Poisson probability is not evaluable at "2.4 SD" -- which is why it
    corrects on the fitted rate instead. Same quantity, each normalised the way its likelihood
    allows.)

    **Events without a full history are DROPPED, and that is why claim 2 and claim 3 see different
    event counts.** A call in a session's first 4 s is perfectly usable for claim 2, whose 50 ms count
    needs no history, and unusable here. Both counts are returned so the difference is reported rather
    than discovered.

    Parameters
    ----------
    events (dict)
        From ``assemble_unit_vocal_events``: ``call_start``, ``session_index``, ``counts``,
        ``positions``, ``region_labels``, ``session_ids``.
    per_session (dict)
        From ``assemble_unit_sessions``: per session ``feature_time_series``, ``feature_names``,
        ``fps`` and ``n_frames``.
    feature_names (list)
        The behavioural features, in order. Selected BY NAME -- per-session column order differs, so
        positional access reads a different feature in different sessions.
    n_lags (int)
        History length in frames.
    sigma_floor (float)
        Floor on the per-session count SD, so a session in which the unit is near-silent cannot divide
        a near-zero spread into an enormous z.

    Returns
    -------
    design (dict)
        ``behaviour`` ``(n_kept, n_features * n_lags)``, ``neural`` ``(n_kept, 1)``, ``positions``,
        ``session_index``, ``region_labels``, ``kept`` (index into the claim-2 event set), and
        ``per_session`` counts of events offered versus kept.
    """

    if not feature_names:
        msg = "nested_design needs at least one behavioural feature."
        raise ValueError(msg)

    behaviour_parts, neural_parts, keep_parts = [], [], []
    bookkeeping: dict = {}
    for slot, session_id in enumerate(events["session_ids"]):
        session = per_session[session_id]
        available = list(session["feature_names"])
        missing = [name for name in feature_names if name not in available]
        if missing:
            msg = (f"{session_id}: behaviour control features {missing} are not in the assembled "
                   f"feature set. Available: {available}.")
            raise ValueError(msg)
        # BY NAME, never by position: per-session column order differs, so a positional read would
        # silently take a different feature in different sessions.
        columns = [available.index(name) for name in feature_names]
        series = np.asarray(session["feature_time_series"], dtype=np.float64)[:, columns]

        rows = np.flatnonzero(events["session_index"] == slot)
        frames = np.rint(events["call_start"][rows] * session["fps"]).astype(np.int64)
        usable = (frames >= n_lags - 1) & (frames < session["n_frames"])
        kept_rows, kept_frames = rows[usable], frames[usable]

        counts = np.asarray(events["counts"], dtype=np.float64)[kept_rows]
        centre = float(counts.mean()) if counts.size else 0.0
        spread = max(float(counts.std()), sigma_floor)

        behaviour_parts.append(lagged_design(series, kept_frames, n_lags)
                               if kept_frames.size else np.empty((0, len(columns) * n_lags)))
        neural_parts.append(((counts - centre) / spread)[:, None])
        keep_parts.append(kept_rows)
        bookkeeping[session_id] = {"n_events_claim2": int(rows.size),
                                   "n_events_claim3": int(kept_rows.size),
                                   "n_dropped_short_history": int(rows.size - kept_rows.size),
                                   "count_mean": centre, "count_sd": spread}

    kept = (np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64))
    stack = (lambda parts, width: np.vstack(parts) if parts else np.empty((0, width)))  # noqa: E731
    return {"behaviour": stack(behaviour_parts, len(feature_names) * n_lags),
            "neural": stack(neural_parts, 1),
            "positions": np.asarray(events["positions"], dtype=np.float64)[kept],
            "session_index": np.asarray(events["session_index"])[kept],
            "region_labels": np.asarray(events["region_labels"], dtype=np.float64)[kept],
            "kept": kept,
            "feature_names": list(feature_names),
            "per_session": bookkeeping}


def _fit_predict(behaviour: np.ndarray, neural: np.ndarray, positions: np.ndarray,
                 weights: np.ndarray, train: np.ndarray, test: np.ndarray,
                 n_lags: int, n_features: int, settings: dict) -> np.ndarray:
    """
    Description
    -----------
    Fit one model on the training rows and predict the held-out ones.

    ``neural`` of None gives the REDUCED model (behaviour alone); passing the column gives the FULL
    model. Both go through the same estimator and the same penalty policy, so the pair differs in
    exactly one column, which is what makes the added score interpretable.

    Parameters
    ----------
    behaviour (np.ndarray)
        ``(n_events, n_features * n_lags)`` behaviour block.
    neural (np.ndarray)
        ``(n_events, 1)`` neural column, or None for the reduced model.
    positions (np.ndarray)
        ``(n_events, 2)`` torus targets.
    weights (np.ndarray)
        Per-row fit weights (equal-region reweighting).
    train, test (np.ndarray)
        Row indices.
    n_lags, n_features (int)
        Design shape.
    settings (dict)
        The ``nested_position_decoding`` block.

    Returns
    -------
    predictions (np.ndarray)
        ``(len(test), 2)`` predicted torus coordinates.
    """

    design = behaviour if neural is None else np.hstack([behaviour, neural])
    estimator = NestedTorusRegression(
        n_behaviour_features=n_features, n_lags=n_lags,
        n_neural_columns=0 if neural is None else neural.shape[1],
        lambda_smooth=settings["lambda_smooth"], l2_reg=settings["l2_reg"],
        smoothness_derivative_order=settings["smoothness_derivative_order"])
    estimator.fit(design[train], positions[train], sample_weight=weights[train])
    return estimator.predict(design[test], snap=False)


def reduced_predictions(design: dict, settings: dict, n_lags: int) -> np.ndarray:
    """
    Description
    -----------
    The reduced model's pooled held-out predictions -- IDENTICAL in every null draw, so they are
    computed once and reused.

    The circular shift moves only the spike train. Behaviour, positions and the equal-region weights
    (which depend on region labels) are untouched, so the reduced fit cannot change between draws.
    Recomputing it per draw would double the null's cost for a guaranteed-identical answer.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`.
    settings (dict)
        The ``nested_position_decoding`` block.
    n_lags (int)
        History length in frames.

    Returns
    -------
    predictions (np.ndarray)
        ``(n_events, 2)`` held-out predicted torus coordinates.
    """

    positions, session_index = design["positions"], design["session_index"]
    weights = inverse_region_frequency_weights(design["region_labels"])
    n_features = len(design["feature_names"])
    predicted = np.full_like(positions, np.nan)
    for held_out in sorted({int(s) for s in session_index}):
        test = np.flatnonzero(session_index == held_out)
        train = np.flatnonzero(session_index != held_out)
        predicted[test] = _fit_predict(design["behaviour"], None, positions, weights, train, test,
                                       n_lags, n_features, settings)
    return predicted


def shifted_neural_column(design: dict, window_edges: np.ndarray, spikes: dict, durations: dict,
                          width: float, rng, guard_seconds: float,
                          sigma_floor: float = 0.05) -> np.ndarray:
    """
    Description
    -----------
    One null draw's neural column: recount the neuron in each event's window from a CIRCULARLY
    SHIFTED spike train, then z-score per session exactly as the observed column is.

    The shift is the right null here because the question is purely the neuron's contribution. It
    preserves the unit's rate, bursting and slow drift -- so the null column looks like a real spike
    train in every respect except its alignment to the calls -- while behaviour and position keep
    their real relationship, which a target permutation would destroy along with the thing being
    conditioned on.

    Counts are RECOUNTED from the shifted train rather than reordered, because a reordering would
    preserve the exact multiset of counts and so test something narrower than "this neuron, decoupled
    in time".

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`; its ``session_index`` assigns events to sessions.
    window_edges (np.ndarray)
        Left edge (seconds) of each event's spike window, aligned with the design's rows.
    spikes (dict)
        ``{session slot: sorted spike times in seconds}``.
    durations (dict)
        ``{session slot: session duration in seconds}``, the circular wrap length.
    width (float)
        Window width in seconds.
    rng (np.random.Generator)
        Draw source.
    guard_seconds (float)
        Guard band at each end of the shift range; lags near 0 or T leave the train near register.
    sigma_floor (float)
        Floor on the per-session count SD.

    Returns
    -------
    column (np.ndarray)
        ``(n_events, 1)`` z-scored shifted count per event.
    """

    session_index = design["session_index"]
    column = np.empty((session_index.size, 1), dtype=np.float64)
    for slot in sorted({int(s) for s in session_index}):
        mask = session_index == slot
        duration = durations[slot]
        low, high = shift_range_seconds(duration, guard_seconds)
        shifted = np.sort((spikes[slot] + rng.uniform(low, high)) % duration)
        counts = counts_in_windows(shifted, window_edges[mask], width).astype(np.float64)
        spread = max(float(counts.std()), sigma_floor)
        column[mask, 0] = (counts - float(counts.mean())) / spread
    return column


def nested_scores(design: dict, settings: dict, n_lags: int,
                  neural: np.ndarray = None, reduced_predicted: np.ndarray = None) -> dict:
    """
    Description
    -----------
    Claim 3's statistic: the POOLED added ``vm_logscore`` of ``position ~ kinematics + neuron`` over
    ``position ~ kinematics``, across leave-one-session-out folds.

    **Pooled, not fold-averaged, and this is ruled rather than stylistic.** Two reasons specific to
    this score. First, the von Mises concentration is FITTED at scoring time, so fold-averaging fits a
    separate one per fold on a thin fold's residuals. Second, the macro score drops acoustic regions
    below ``min_region_events`` and returns NaN when none clear -- measured over the cohort's 70
    vocal-eligible sessions, at the ruled threshold of 20 events, 31.4% carry a region that would be
    dropped at fold grain, so per-fold macro scores would be averages over DIFFERENT region sets, with
    a 1-event region weighted equally against a 500-event one. Pooled, cl0401's thinnest region holds
    198 events and cl0499's 454.

    Each model fits its OWN concentration on the pooled residuals, which is the ordinary nested
    likelihood comparison: a model with tighter residuals earns both a higher concentration and a
    higher score, and that is the score being what it claims to be rather than a nuisance advantage.

    Training rows carry EQUAL-REGION REWEIGHTING, exactly as P1 fits it, so common regions do not
    dominate. Those weights depend only on the region labels, which no shuffle moves.

    The guard against one session carrying the result is the LEAVE-ONE-FOLD-OUT MINIMUM -- the worst
    pooled added score with any single fold's events dropped. Threshold-free, free to compute, and a
    descriptor rather than a gate.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`.
    settings (dict)
        The ``nested_position_decoding`` block.
    n_lags (int)
        History length in frames.
    neural (np.ndarray)
        Neural column to use instead of ``design['neural']`` -- the null passes a shifted one. None
        uses the observed column.

    Returns
    -------
    scores (dict)
        ``added``, ``reduced``, ``full``, ``per_fold``, ``min_leave_one_fold_out``, ``n_events``,
        ``n_regions_scored``.
    """

    behaviour = design["behaviour"]
    neural_column = design["neural"] if neural is None else np.asarray(neural, dtype=np.float64)
    positions, regions = design["positions"], design["region_labels"]
    session_index = design["session_index"]
    n_features = len(design["feature_names"])
    weights = inverse_region_frequency_weights(regions)

    sessions = sorted({int(s) for s in session_index})
    if len(sessions) < 2:
        msg = (f"nested_scores needs at least two sessions for leave-one-session-out; "
               f"got {len(sessions)}.")
        raise ValueError(msg)

    # The reduced pass is identical in every null draw, so a caller running a null hands it in once.
    predicted_reduced = (np.full_like(positions, np.nan) if reduced_predicted is None
                         else np.asarray(reduced_predicted, dtype=np.float64))
    predicted_full = np.full_like(positions, np.nan)
    per_fold = []
    for held_out in sessions:
        test = np.flatnonzero(session_index == held_out)
        train = np.flatnonzero(session_index != held_out)
        if reduced_predicted is None:
            predicted_reduced[test] = _fit_predict(behaviour, None, positions, weights, train, test,
                                                   n_lags, n_features, settings)
        predicted_full[test] = _fit_predict(behaviour, neural_column, positions, weights, train,
                                            test, n_lags, n_features, settings)
        per_fold.append({"held_out": held_out, "n_events": int(test.size)})

    def macro(predictions: np.ndarray, rows: np.ndarray) -> float:
        """Pooled macro score over the given rows; each model fits its own concentration."""
        return float(macro_von_mises_logscore(
            predictions[rows], positions[rows], regions[rows], metric="torus", period=1.0,
            min_region_events=settings["min_region_events"]))

    everything = np.arange(positions.shape[0])
    reduced_score, full_score = macro(predicted_reduced, everything), macro(predicted_full, everything)

    dropped = []
    for fold in per_fold:
        rest = np.flatnonzero(session_index != fold["held_out"])
        fold_rows = np.flatnonzero(session_index == fold["held_out"])
        fold["added"] = macro(predicted_full, fold_rows) - macro(predicted_reduced, fold_rows)
        dropped.append(macro(predicted_full, rest) - macro(predicted_reduced, rest))

    labelled = regions[np.isfinite(regions)]
    _values, counts = np.unique(labelled, return_counts=True)
    return {"added": full_score - reduced_score,
            "reduced": reduced_score,
            "full": full_score,
            "per_fold": per_fold,
            "min_leave_one_fold_out": float(np.min(dropped)) if dropped else float("nan"),
            "n_events": int(positions.shape[0]),
            "n_regions_scored": int(np.sum(counts >= settings["min_region_events"]))}


def next_feature_control(events: dict, per_session: dict, reduced_features: list,
                         candidates: list, settings: dict, n_lags: int) -> dict:
    """
    Description
    -----------
    ONE MORE STEP of the behaviour selection: what would a sixth kinematic feature have added?

    This calibrates claim 3's added score against a realistic alternative, and it answers a question
    the permutation null CANNOT. The null asks whether the neuron's alignment IN TIME is real; a
    relay's alignment is genuinely real, so a neuron that merely re-encodes behaviour passes it. What
    that pathway needs instead is a benchmark: how much does adding a REAL extra behavioural
    predictor -- out-of-fold, with realistic noise -- improve the same score? If the neuron's
    contribution is unremarkable beside the best remaining feature, its added score is telling us
    about the control's estimation error rather than about the neuron.

    This is why a synthetic placebo is the weaker instrument. The behaviour block's own fitted
    projection is a noiseless, optimally weighted scalar that no real predictor resembles, so it
    bounds the pathway rather than calibrating it.

    It also settles a residual the behaviour selection left open: it stopped under a 1SE rule, which
    is deliberately conservative about ADDING features, so a feature carrying a little position
    information could have failed that bar and still be worth conditioning on in a control. This
    measures how much such a feature would have contributed.

    Every candidate is scored on the SAME events and folds as the neuron, so the numbers are directly
    comparable.

    Parameters
    ----------
    events (dict)
        From ``assemble_unit_vocal_events``.
    per_session (dict)
        From ``assemble_unit_sessions``.
    reduced_features (list)
        The behaviour control's features.
    candidates (list)
        Features to try as a sixth; typically every assembled feature not already in the control.
    settings (dict)
        The ``nested_position_decoding`` block.
    n_lags (int)
        History length in frames.

    Returns
    -------
    control (dict)
        ``per_candidate`` (feature -> added score), ``best_feature``, ``best_added``, and
        ``n_candidates``. Added scores are computed exactly as the neuron's is, against the same
        reduced model.
    """

    overlap = [name for name in candidates if name in reduced_features]
    if overlap:
        msg = f"candidates must not already be in the behaviour control; got {overlap}."
        raise ValueError(msg)

    per_candidate: dict = {}
    for candidate in candidates:
        design = nested_design(events, per_session, [*reduced_features, candidate], n_lags)
        # The sixth feature IS the extra block, so the "reduced" model here is the five-feature
        # control and the "full" model is six features -- scored by dropping the neural column and
        # letting the candidate's own lags be the addition.
        five = nested_design(events, per_session, reduced_features, n_lags)
        wide = dict(design)
        wide["neural"] = np.zeros((design["behaviour"].shape[0], 0))
        narrow = dict(five)
        narrow["neural"] = np.zeros((five["behaviour"].shape[0], 0))
        per_candidate[candidate] = float(
            _score_block(wide, settings, n_lags) - _score_block(narrow, settings, n_lags))

    best = max(per_candidate, key=per_candidate.get) if per_candidate else None
    return {"per_candidate": per_candidate,
            "best_feature": best,
            "best_added": per_candidate[best] if best is not None else float("nan"),
            "n_candidates": len(per_candidate)}


def _score_block(design: dict, settings: dict, n_lags: int) -> float:
    """
    Description
    -----------
    Pooled macro score of a behaviour-only design, over the same folds ``nested_scores`` uses.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`, with a zero-width neural block.
    settings (dict)
        The ``nested_position_decoding`` block.
    n_lags (int)
        History length in frames.

    Returns
    -------
    score (float)
        Pooled macro von Mises log-likelihood.
    """

    positions, regions = design["positions"], design["region_labels"]
    session_index = design["session_index"]
    weights = inverse_region_frequency_weights(regions)
    n_features = len(design["feature_names"])
    predicted = np.full_like(positions, np.nan)
    for held_out in sorted({int(s) for s in session_index}):
        test = np.flatnonzero(session_index == held_out)
        train = np.flatnonzero(session_index != held_out)
        predicted[test] = _fit_predict(design["behaviour"], None, positions, weights, train, test,
                                       n_lags, n_features, settings)
    return float(macro_von_mises_logscore(predicted, positions, regions, metric="torus", period=1.0,
                                          min_region_events=settings["min_region_events"]))
