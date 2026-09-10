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

from ..modeling.manifold_torus_regression import SmoothTorusManifoldRegression
from .neural_design_assembly import lagged_design


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
