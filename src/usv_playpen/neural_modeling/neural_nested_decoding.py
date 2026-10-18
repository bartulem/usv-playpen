"""
@author: bartulem
Nested vocal-manifold position decoding: does a neuron carry vocal output type BEYOND the behaviour that
predicts it?

Two nested models of the same target, the upcoming USV's position on the QLVM torus, differing in
exactly the neuron:

    reduced:  behaviour
    full:     behaviour + the neuron's window count

and the statistic is what the neuron ADDS, held out, full minus reduced. A PURE RELAY, whose spiking
is a function of the behaviour, adds nothing the behaviour did not already carry and fails. A unit
that adds something passes -- and what it adds is either genuinely internal, or behaviour that was
not measured. This test cannot separate those two, which is why the paper says "beyond MEASURED
behaviour" rather than "computes the transformation", and why measuring behaviour more completely
makes this test STRONGER rather than weaker.

THE TESTED STATISTIC IS THE POSTERIOR FORM (ruled 2026-10-12): identity's own Bayes decoder with the
behaviour control entering as the PRIOR -- a von Mises on the torus grid around the control's predicted
position, with a concentration fitted on training events -- so ``added`` is log posterior(truth |
neuron, behaviour) minus log prior(truth | behaviour), in nats per event. Identity's gain is the same
quantity under the unconditioned occupancy prior, which makes the two claims one statistic under two
priors. The reason is measured, not aesthetic: cl0401, whose counts run 3.4 to 5.7 across the four
acoustic regions and decode +0.049 nats/event categorically, added NOTHING (-0.012) to the earlier
point-regression form, because "probably this region" can reweight a distribution but cannot move a
single predicted point without spreading it, and a concentration-fitting score punishes the spread.
The regression form is RETAINED as a fine-position DESCRIPTOR (``point_regression`` in the section),
never a route to passing.

The point-regression form, for the record: ``reduced`` is the control's predicted position as four
sine/cosine columns, ``full`` adds the neuron's z-scored count as one column, both fit by the torus
ridge regression and scored by the macro von Mises log-score. Its asymmetry was deliberate and ran
conservative -- the neuron earns its place as ONE column -- and its cost was power, not validity:
contribution that is nonlinear in the count cannot be expressed. The posterior form removes that
limitation for coarse tuning; the transformation count remains a lower bound.

Three things are settled and should not be re-derived.

The reduced model is P1's SELECTED five features, not all nineteen. The selection MEASURED that the
other fourteen add no position information, and replacing a measurement with an assumption is the
wrong direction. Its `final_model_features` is read from P1's own result file and asserted against
the frozen list in settings, so a re-run of P1 fails loudly here instead of silently redefining what
the neuron is measured against, after the counts are banked.

It is scored under MACRO von Mises -- the objective the control was SELECTED under. Micro would leave
the control mildly suboptimal for the test (its first-selected feature, `self.neck_elevation`, HURTS
the micro score), which biases the test toward passing: the wrong direction for a positive result.

The neuron enters as a spike COUNT, not 600 lags of binary train. A 4 s spike history spans ~20 prior
USVs at the 195 ms median inter-USV interval, so it is largely the neuron's RESPONSES to earlier
USVs -- and since USVs within a bout correlate in type, the test would pass on bout structure. The
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
    _fit_von_mises_kappa,
    inverse_region_frequency_weights,
    macro_von_mises_logscore,
    signed_diff,
)
from ..modeling.manifold_torus_regression import SmoothTorusManifoldRegression
from .deviance_metrics import finite_mean, finite_min
from .neural_design_assembly import lagged_design, onset_history_anchor_frames
from .neural_vocal_decoding import decode_gain
from .neural_vocal_occurrence import counts_in_windows
from .shift_null_inference import shift_range_seconds


class NestedTorusRegression(SmoothTorusManifoldRegression):
    """
    Description
    -----------
    The torus regressor with a BLOCK-DIAGONAL smoothness penalty, so a design that mixes a temporal
    behaviour block with a handful of non-temporal neural columns is penalised correctly.

    The parent applies one identical ``D^T D`` block per feature across ``n_time_bins`` lags, which
    assumes every feature is a temporal filter. The nested design is not like that: five behaviour
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
    huber_delta (float)
        Huber transition point of the parent's residual loss, in torus units; production passes
        ``nested_vocal_manifold_position_decoding.huber_delta``. The default equals the shipped value.
    learning_rate (float)
        Initial Adam step size of the parent's optimiser (cosine-decayed to zero over ``max_iter``);
        production passes ``nested_vocal_manifold_position_decoding.learning_rate``. The default equals the
        shipped value.
    max_iter (int)
        Optimiser step cap; production passes ``nested_vocal_manifold_position_decoding.max_iter``. The
        default equals the shipped value.
    tol (float)
        Convergence tolerance on the parameter-change norm; production passes
        ``nested_vocal_manifold_position_decoding.tol``. The default equals the shipped value.
    random_state (int)
        Seed of the weight initialisation; production passes
        ``nested_vocal_manifold_position_decoding.random_state``. The default equals the shipped value.

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
            huber_delta: float = 1.0,
            learning_rate: float = 1e-3,
            max_iter: int = 5000,
            tol: float = 1e-4,
            random_state: int = 0,
    ):
        total_columns = int(n_behaviour_features) * int(n_lags) + int(n_neural_columns)
        super().__init__(n_features=1, n_time_bins=total_columns, lambda_smooth=lambda_smooth,
                         l2_reg=l2_reg, smoothness_derivative_order=smoothness_derivative_order,
                         huber_delta=huber_delta, learning_rate=learning_rate, max_iter=max_iter,
                         tol=tol, random_state=random_state, metric="torus", period=period)
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
    The behaviour control's feature set: READ from the cohort selection's result file and ASSERTED
    against the frozen list in settings.

    Reading alone would let a re-run of the selection silently redefine the behaviour the neuron is
    measured against
    after transformation counts are banked. Freezing alone would let the control drift out of step with
    the selection it is meant to be. Doing both catches the drift instead of absorbing it, and is the
    same "check rather than assume" the emitter and dyad resolutions use.

    WHAT MAY BE ADOPTED FROM A BEHAVIOURAL SELECTION, AND WHAT MAY NOT. Its FITTED MAP does not
    transfer to the ephys preparation -- measured at -0.00113 over 41,087 USVs -- while the same
    estimator refitted in domain predicts perfectly well. That is a statement about COEFFICIENTS, not
    about which features to use, so a behavioural selection's feature set is read here while the
    control itself is always refitted in domain by ``fit_control_models``. An externally chosen set is
    in one respect preferable: those features were never selected on the preparation the claim is made
    about, so selection optimism is zero rather than merely bounded.

    Both artifact shapes are therefore accepted -- a cohort selection's ``selected_features``, or a
    behavioural selection's ``steps`` (its last step's ``current_features``) -- and either way the
    result is ASSERTED against the frozen list, so a re-run of a selection cannot quietly redefine the
    behaviour the neuron is measured against.

    The file is an INPUT to the run, so it lives in the ``data`` block beside the session lists.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict; ``data.behaviour_selection_result_path`` and
        ``nested_vocal_manifold_position_decoding.reduced_model_features`` are read.
    require_selection_file (bool)
        When False, a missing selection file is tolerated and the frozen list is returned unchecked --
        for running where the lab share is not mounted. A file that IS present is always checked.

    Returns
    -------
    features (list)
        The behaviour control's feature names, in P1's selection order.
    """

    frozen = list(settings["nested_vocal_manifold_position_decoding"]["reduced_model_features"])
    path = pathlib.Path(settings["data"]["behaviour_selection_result_path"])
    if not path.exists():
        if require_selection_file:
            msg = (f"behaviour selection result not found at {path}; it defines the nested decoding's reduced "
                   f"model. Pass require_selection_file=False to run on the frozen list alone.")
            raise FileNotFoundError(msg)
        return frozen

    with path.open("rb") as handle:
        selection = pickle.load(handle)
    if "selected_features" in selection:
        selected = list(selection["selected_features"])
    elif "steps" in selection:
        # A behavioural-modelling selection, whose FEATURE SET may be adopted while its COEFFICIENTS
        # may not. Only the names are taken here; the control is still fitted in domain by
        # `fit_control_models`, which is the guard that matters -- the fitted map is what failed to
        # transfer (-0.00113 over 41,087 USVs), not the choice of features. Adopting an external set
        # also removes selection optimism outright instead of bounding it, since those features were
        # never chosen on the preparation the claim is made about.
        selected = list(selection["steps"][-1]["current_features"])
    else:
        msg = (f"{path.name} is neither a cohort selection result ('selected_features') nor a "
               f"behavioural-modelling selection ('steps'). Point "
               f"data.behaviour_selection_result_path at the artifact written by "
               f"behaviour_control_model.select_control_features, or at a behavioural selection "
               f"whose feature set is being adopted.")
        raise ValueError(msg)
    if selected != frozen:
        msg = (f"behaviour selection drift: {path.name} selected {selected}, but settings freeze "
               f"{frozen}. The behaviour control would silently change meaning; update the settings "
               f"deliberately if the selection was re-run.")
        raise ValueError(msg)
    return selected


def nested_design(events: dict, per_session: dict, feature_names: list, n_lags: int,
                  sigma_floor: float) -> dict:
    """
    Description
    -----------
    Build the nested decoding's design: each event's 4 s behavioural history, plus the neuron as ONE column.

    Rows are USVs, not frames. Each behavioural feature contributes ``n_lags`` columns -- its history
    ending at the USV's onset frame -- so a row is one USV with ``len(feature_names) * n_lags``
    behaviour columns followed by a single neural column. The common misstatement to avoid is that
    "every predictor is summarised to one value per USV": the behaviour block is NOT summarised, it
    is 600 columns per feature. What differs between predictors is how many columns each contributes.

    **The neuron is a z-scored spike COUNT, not 600 lags of binary train.** The symmetric alternative
    was considered and rejected: a 4 s spike history spans about twenty prior USVs at the 195 ms
    median inter-USV interval, so it is largely the neuron's RESPONSES to earlier USVs, and since
    USVs within a bout correlate in type the model would predict this USV's position from previous
    USVs' positions via the neuron -- the test would pass on bout structure. The behaviour block has
    the same 4 s window and the same bout structure and is accepted there because behaviour is the
    CONTROL, where a longer window makes the test harder. Within the 50 ms window a count rather than
    per-frame binary, because 50 ms at 150 fps is 7.5 frames (so per-frame needs the window rounded or
    the last bin ragged), the data are sparse (cl0401: 2.16 spikes per window over 8 frame-columns),
    and decisively because THE VOCAL DECODING ALREADY ENCODES THE NEURON AS A COUNT -- encoding it differently in
    the two analyses of one conjunction would break the nesting.

    The count is z-scored PER SESSION, the same correction the vocal-occurrence axis applies, because this
    is a least-squares fit on a continuous predictor where standardising the input is available. (The
    vocalization-identity axis cannot do that -- a Poisson probability is not evaluable at "2.4 SD" -- which is why it
    corrects on the fitted rate instead. Same quantity, each normalised the way its likelihood
    allows.)

    **Events without a full history are DROPPED, and that is why the vocal decoding and the nested
    decoding see different event counts.** A USV in a session's first 4 s is perfectly usable for the vocal
    decoding, whose spike count
    needs no history, and unusable here. Both counts are returned so the difference is reported rather
    than discovered.

    Parameters
    ----------
    events (dict)
        From ``assemble_unit_vocal_events``: ``vocalization_start``, ``session_index``, ``counts``,
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
        ``session_index``, ``region_labels``, ``kept`` (index into the vocal-decoding event set), and
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
        # The frame before onset, the convention the cohort behaviour control is fitted at.
        frames = onset_history_anchor_frames(events["vocalization_start"][rows], session["fps"])
        usable = (frames >= n_lags - 1) & (frames < session["n_frames"])
        kept_rows, kept_frames = rows[usable], frames[usable]

        counts = np.asarray(events["counts"], dtype=np.float64)[kept_rows]
        centre = float(counts.mean()) if counts.size else 0.0
        spread = max(float(counts.std()), sigma_floor)

        behaviour_parts.append(lagged_design(series, kept_frames, n_lags)
                               if kept_frames.size else np.empty((0, len(columns) * n_lags)))
        neural_parts.append(((counts - centre) / spread)[:, None])
        keep_parts.append(kept_rows)
        bookkeeping[session_id] = {"n_events_vocal_decoding": int(rows.size),
                                   "n_events_with_full_history": int(kept_rows.size),
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


def behaviour_prior_log_density(fold: dict, predicted_positions: np.ndarray, positions: np.ndarray,
                                period: float, settings: dict) -> tuple[float, np.ndarray]:
    """
    Description
    -----------
    Behaviour as a PRIOR over the torus grid: for each event, the fold's occupancy prior MULTIPLIED by
    the von Mises evidence the behaviour control supplies for that USV, with a concentration measured
    on the training events alone.

    This is how the nested statistic conditions on behaviour without asking the neuron to compete for a
    point. The regression form forced every model to answer "where exactly", one point per USV, and a
    neuron whose knowledge is "probably this region" cannot help with that -- measured on cl0401, whose
    counts run 3.4 to 5.7 across regions and decode +0.049 nats/event categorically, yet add nothing to
    a point. A prior can be reweighted without being moved, so the same knowledge is worth nats here.

    THE PRODUCT FORM IS BAYES, AND THE FLAT FORM WAS A BIAS -- found before it produced a number. The
    control gives a point ``x_hat`` whose residual around the truth is von Mises with concentration
    ``kappa``; by symmetry ``p(x_hat | x) = vM(x; x_hat, kappa)``, so with the occupancy base rate
    ``p(x)`` the conditioned prior is ``p(x | x_hat) ~ occupancy(x) * vM(x; x_hat, kappa)``. The first
    draft used the von Mises ALONE, the flat-prior special case, and on cl0401 that scored 0.22 nats
    WORSE at the truth than identity's occupancy prior -- not because behaviour misled, but because the
    prior had forgotten where USVs live. A neuron is then credited for restoring that, which inflates
    ``added``. In the product form a control that knows nothing (``kappa`` to 0) reduces to the
    occupancy prior, the ladder's first step goes to zero rather than -0.22, and a control that knows
    something sharpens the base rate instead of replacing it.

    The concentration is ONE shared kappa per fold, set on the training sessions alone, two ways.
    ``train_residuals`` fits it on the training events' wrap-aware angular residuals around the
    control's prediction -- the macro scorer's own convention (``_fit_von_mises_kappa`` over both
    dimensions ravelled). ``train_tuned_inner_loso`` chooses it from ``kappa_grid`` by inner
    leave-one-session-out on the training sessions, maximising the held-in log prior at the truth over
    the occupancy prior -- the same discipline as the amplitude calibration, and for the same reason:
    a residual fit is MIS-SPECIFIED for a shrunk predictor (see the branch comment) and only a
    held-out criterion can tell a control that knows nothing from one that knows something. Either
    way the evidence is only as sharp as behaviour has earned on data the scored session never sees.
    The occupancy term already carries the fold's mask, so the product can put no mass where no
    training USV ever sat; renormalising over the masked grid absorbs ``log I0(kappa)`` and no second
    copy of the density formula is needed.

    Parameters
    ----------
    fold (dict)
        From ``neural_vocal_decoding.decoding_context``; its ``train_rows``, ``grid_positions``,
        ``mask`` and ``cell_area`` are read.
    predicted_positions (np.ndarray)
        ``(n_events, 2)`` the behaviour control's predicted torus position per event, unpermuted.
    positions (np.ndarray)
        ``(n_events, 2)`` true torus positions, unpermuted.
    period (float)
        Torus period.
    settings (dict)
        The nested block; ``behaviour_prior.form``, ``behaviour_prior.concentration`` and, for the
        tuned concentration, ``behaviour_prior.kappa_grid`` (candidates, all ``>= 0``) are read.

    Returns
    -------
    kappa, log_density (tuple)
        The fitted concentration, and ``(n_events, n_cells)`` log prior density per event over the
        grid, masked and normalised, floored at identity's own ``log(1e-300)``.
    """

    prior = settings["behaviour_prior"]
    if prior["form"] != "von_mises":
        msg = f"behaviour_prior.form must be 'von_mises'; got {prior['form']!r}. Ruled, not missing."
        raise ValueError(msg)
    scale = 2.0 * np.pi / period

    def tilted(kappa: float) -> np.ndarray:
        # Bayes: the occupancy base rate times the behaviour evidence. Starting from the fold's own
        # (masked) log occupancy prior is what makes a know-nothing control reduce to identity's prior.
        grid = fold["grid_positions"]
        log_density = np.repeat(fold["log_prior"][None, :], predicted_positions.shape[0], axis=0)
        for dimension in range(2):
            delta = scale * (grid[None, :, dimension] - predicted_positions[:, None, dimension])
            log_density += kappa * np.cos(delta)
        log_density = np.where(fold["mask"][None, :], log_density, -np.inf)
        peak = log_density.max(axis=1, keepdims=True)
        log_norm = peak + np.log(np.sum(np.exp(log_density - peak), axis=1, keepdims=True)
                                 * fold["cell_area"])
        return np.maximum(log_density - log_norm, np.log(1e-300))

    if prior["concentration"] == "train_residuals":
        train = fold["train_rows"]
        residual = signed_diff(positions[train], predicted_positions[train], metric="torus",
                               period=period) * scale
        kappa = float(_fit_von_mises_kappa(residual.ravel()))
    elif prior["concentration"] == "train_tuned_inner_loso":
        # MEASURED on cl0401 (2026-10-12): residual kappa (0.62-0.85) scored the prior 0.10 nats BELOW
        # occupancy at the truth, because a shrunk predictor's residual spread is mostly the truth's
        # own spread, which occupancy already carries -- the evidence double-counts the base rate.
        # Tuning kappa on the training rotation, exactly as the amplitude is tuned, lets a control
        # that knows nothing settle at kappa = 0 (the inner curve was monotone decreasing from zero
        # on every fold) and a control that knows something sharpen the base rate by what it has
        # earned. kappa >= 0 by construction: the evidence can be declined, never inverted.
        #
        # What this measures is evidence beyond the occupancy prior AS FITTED. If that prior is
        # over-smoothed -- a KDE bandwidth too wide for a tight cluster -- a control that merely
        # returns the cluster's centre earns a positive kappa by re-sharpening the base rate (seen on
        # the coarse test grid: 0.5 for a constant predictor). That is CONSERVATIVE for `added`, since
        # a sharper prior leaves the neuron less to add, but it is misattributed on the ladder's first
        # rung. The shipped bandwidth is selected by held-out position likelihood, under which cl0401
        # tuned to exactly 0, so it does not arise there; it is the reason the bandwidth must stay
        # held-out-selected and never be widened for speed on real data.
        kappa = float(prior["kappa_grid"][0])
        if fold["inner_splits"]:
            best = -np.inf
            for candidate in (float(k) for k in prior["kappa_grid"]):
                log_density = tilted(candidate)
                held_in = []
                for _inner_train, inner_test in fold["inner_splits"]:
                    cells = fold["truth_cell"][inner_test]
                    value = log_density[inner_test, cells] - fold["log_prior"][cells]
                    value[~fold["mask"][cells]] = np.nan
                    held_in.append(finite_mean(value))
                score = finite_mean(np.asarray(held_in))
                if score > best:
                    best, kappa = score, candidate
    else:
        msg = (f"behaviour_prior.concentration must be 'train_residuals' or "
               f"'train_tuned_inner_loso'; got {prior['concentration']!r}. Ruled, not missing.")
        raise ValueError(msg)
    return kappa, tilted(kappa)


def behaviour_priors(folds: list, predicted_positions: np.ndarray, positions: np.ndarray,
                     nested: dict) -> tuple[list, list]:
    """
    Description
    -----------
    The behaviour prior for every fold, built ONCE.

    A prior depends only on positions, the control's predictions and the fold's geometry -- none of
    which a null draw changes -- so it is computed here and handed to every draw, rather than rebuilt
    a thousand times inside the null for a quantity that does not move. The grid matrices are the
    largest arrays this statistic holds (events x cells per fold), so building them once is also what
    keeps the worker processes' memory flat.

    Parameters
    ----------
    folds (list)
        From ``decoding_context``.
    predicted_positions (np.ndarray)
        The behaviour control's predicted position per event, unpermuted.
    positions (np.ndarray)
        True positions per event, unpermuted.
    nested (dict)
        The nested block (``nested_settings``); its ``torus_period`` and ``behaviour_prior`` are read.

    Returns
    -------
    kappas, priors (tuple)
        One fitted concentration per fold, and one ``(n_events, n_cells)`` log prior per fold.
    """

    kappas, priors = [], []
    for fold in folds:
        kappa, log_prior = behaviour_prior_log_density(fold, predicted_positions, positions,
                                                       nested["torus_period"], nested)
        kappas.append(kappa)
        priors.append(log_prior)
    return kappas, priors


def posterior_nested_gain(counts: np.ndarray, folds: list, decoding: dict, priors: list,
                          kappas: list, permutation: np.ndarray = None,
                          with_detail: bool = False) -> dict:
    """
    Description
    -----------
    The nested statistic in identity's currency: what the neuron adds, in nats per event, once
    behaviour has already set the prior.

    It IS ``decode_gain`` -- same tuning surface, ridge, exposure offsets, amplitude calibration, grid
    and mask -- with the fold's occupancy prior replaced by :func:`behaviour_prior_log_density`. So
    ``added`` is log posterior(truth | neuron, behaviour) minus log prior(truth | behaviour), averaged
    over held-out events, and identity's gain is the same quantity under the unconditioned prior. The
    two claims are one statistic under two priors, which is what combining their p-values assumes.

    Reported beside it, the LADDER the regression form also drew: ``behaviour_over_no_behaviour`` is
    log prior(truth | behaviour) minus log occupancy prior(truth) -- what behaviour alone buys -- and
    ``full_over_no_behaviour`` is the sum. The ladder's first step is INVARIANT under the permutation
    null because a prior row travels with its position; only ``added`` is tested.

    Parameters
    ----------
    counts (np.ndarray)
        Spike count per event; never permuted.
    folds (list)
        From ``decoding_context`` built on these events' positions and sessions.
    decoding (dict)
        The ``vocal_decoding`` block, whose tuning-surface settings the shared decoder reads.
    priors (list)
        From :func:`behaviour_priors`: one ``(n_events, n_cells)`` log prior per fold, unpermuted.
    kappas (list)
        From :func:`behaviour_priors`: the concentration each prior was built with.
    permutation (np.ndarray)
        Event reordering applied to positions AND their priors together, or None.
    with_detail (bool)
        Carry the per-event detail through, observed pass only.

    Returns
    -------
    result (dict)
        ``added``, ``added_uncalibrated``, ``nats_per_spike``, ``behaviour_over_no_behaviour``,
        ``full_over_no_behaviour``, ``n_masked``, ``n_folds_positive``, ``min_leave_one_fold_out``,
        ``per_fold`` (each with its ``kappa``), ``alphas``, ``ridges``, ``kappas``, and the detail.
    """

    full = decode_gain(counts, folds, decoding, permutation, with_detail, log_prior_events=priors)

    order = np.arange(counts.size) if permutation is None else np.asarray(permutation)
    behaviour_over = np.full(counts.size, np.nan, dtype=np.float64)
    for fold, log_prior in zip(folds, priors, strict=True):
        test = fold["test_rows"]
        cells = fold["truth_cell"][order][test]
        value = log_prior[order][test, cells] - fold["log_prior"][cells]
        value[~fold["mask"][cells]] = np.nan
        behaviour_over[test] = value
    behaviour_gain = finite_mean(behaviour_over)

    per_fold = [{**fold_result, "kappa": kappa, "added": fold_result["gain"]}
                for fold_result, kappa in zip(full["per_fold"], kappas, strict=True)]
    for fold_result in per_fold:
        del fold_result["gain"]
    renamed = {"added": full["gain"], "added_uncalibrated": full["gain_uncalibrated"],
               "nats_per_spike": full["nats_per_spike"], "n_masked": full["n_masked"],
               "n_folds_positive": full["n_folds_positive"],
               "min_leave_one_fold_out": full["min_leave_one_fold_out"],
               "per_fold": per_fold, "alphas": full["alphas"], "ridges": full["ridges"],
               "kappas": kappas,
               "behaviour_over_no_behaviour": behaviour_gain,
               "full_over_no_behaviour": full["gain"] + behaviour_gain}
    detail = {key: full[key] for key in ("per_event_gain", "decode_error", "posterior_entropy",
                                          "mean_decode_error", "mean_posterior_entropy",
                                          "overdispersion_index") if key in full}
    return {**renamed, **detail}


#: The only value implemented for each option `nested_vocal_manifold_position_decoding` names. Every one of these
#: keys described a real alternative the plan discusses, and every one was read by NOTHING -- so a run
#: could declare `vm_score_mode: "micro"`, be reported as micro, and score macro. The plan RULED macro
#: for this test (micro leaves the behaviour control suboptimal for it, which biases it toward
#: passing), RULED the rich behaviour window, and RULED the neuron as ONE z-scored count -- so
#: these are settled decisions, and the guard makes a deviation from them loud instead of silent.
#: `statistic` is the 2026-10-12 ruling: the posterior form is the tested statistic and the point
#: regression is kept as a fine-position DESCRIPTOR, never a route to passing.
IMPLEMENTED_OPTIONS = {
    "vm_score_mode": ("macro",),
    "behaviour_control": ("rich",),
    "prevocal_window_n_bins": (1,),
    "rate_transform": ("none",),
    "rate_basis": ("linear",),
    "statistic": ("posterior",),
}

#: Nested-key options, ``(outer key, inner key) -> implemented values``, guarded the same way.
IMPLEMENTED_NESTED_OPTIONS = {
    ("behaviour_prior", "form"): ("von_mises",),
    ("behaviour_prior", "concentration"): ("train_residuals", "train_tuned_inner_loso"),
    ("discrimination_null", "method"): ("within_session_pair_permutation",),
}


def nested_settings(settings: dict) -> dict:
    """
    Description
    -----------
    The ``nested_vocal_manifold_position_decoding`` block with the shared manifold period merged in, validated.

    The torus period is a property of the QLVM MANIFOLD, not of this test, so it lives in the shared
    ``vocalization_settings`` block where every analysis can see one value. This module's functions take a
    single block, so the caller merges here rather than the key being duplicated into two blocks that
    could drift apart -- which is the defect this whole audit was about.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    block (dict)
        A copy of ``nested_vocal_manifold_position_decoding`` carrying ``torus_period``, refused if it declares an
        option this module does not implement.
    """

    block = dict(settings["nested_vocal_manifold_position_decoding"])
    block["torus_period"] = settings["vocalization_settings"]["torus_period"]
    check_settings(block)
    return block


def check_settings(settings: dict) -> None:
    """
    Description
    -----------
    Refuse a ``nested_vocal_manifold_position_decoding`` block that declares an option this module does not implement.

    Parameters
    ----------
    settings (dict)
        The ``nested_vocal_manifold_position_decoding`` block.

    Returns
    -------
    """

    for key, allowed in IMPLEMENTED_OPTIONS.items():
        value = settings[key]
        if value not in allowed:
            others = ", ".join(repr(a) for a in allowed)
            msg = (f"nested_vocal_manifold_position_decoding.{key} = {value!r} is not implemented; this module supports "
                   f"{others}. This is a ruled decision, not a missing feature -- see the plan.")
            raise ValueError(msg)
    for (outer, inner), allowed in IMPLEMENTED_NESTED_OPTIONS.items():
        value = settings[outer][inner]
        if value not in allowed:
            others = ", ".join(repr(a) for a in allowed)
            msg = (f"nested_vocal_manifold_position_decoding.{outer}.{inner} = {value!r} is not "
                   f"implemented; this module supports {others}. This is a ruled decision, not a "
                   f"missing feature -- see the plan.")
            raise ValueError(msg)


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
        The ``nested_vocal_manifold_position_decoding`` block.

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
        smoothness_derivative_order=settings["smoothness_derivative_order"],
        period=settings["torus_period"], huber_delta=settings["huber_delta"],
        learning_rate=settings["learning_rate"], max_iter=settings["max_iter"], tol=settings["tol"],
        random_state=settings["random_state"])
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
        The ``nested_vocal_manifold_position_decoding`` block.
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
                          sigma_floor: float) -> np.ndarray:
    """
    Description
    -----------
    One null draw's neural column: recount the neuron in each event's window from a CIRCULARLY
    SHIFTED spike train, then z-score per session exactly as the observed column is.

    The shift is the right null here because the question is purely the neuron's contribution. It
    preserves the unit's rate, bursting and slow drift -- so the null column looks like a real spike
    train in every respect except its alignment to the USVs -- while behaviour and position keep
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
    The nested decoding's statistic: the POOLED added ``vm_logscore`` of ``position ~ kinematics + neuron`` over
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
    higher score, and that is the score being what it is meant to be rather than a nuisance advantage.

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
        The ``nested_vocal_manifold_position_decoding`` block.
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

    check_settings(settings)
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
        return pooled_macro_score(predictions[rows], positions[rows], regions[rows], settings)

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
            "min_leave_one_fold_out": finite_min(dropped),
            "n_events": int(positions.shape[0]),
            "n_regions_scored": int(np.sum(counts >= settings["min_region_events"]))}


def pooled_macro_score(predictions: np.ndarray, positions: np.ndarray, regions: np.ndarray,
                       settings: dict) -> float:
    """
    Description
    -----------
    The pooled macro von Mises log-score of torus predictions, the ONE scoring rule every model in this module
    is compared under.

    Pooled over whatever rows are passed, macro over acoustic regions (each clearing ``min_region_events``
    weighted equally), with the concentration fitted to these predictions at scoring time -- so each model is
    scored at its own best concentration rather than one borrowed from another model.

    Parameters
    ----------
    predictions (np.ndarray)
        ``(n, 2)`` predicted torus positions.
    positions (np.ndarray)
        ``(n, 2)`` true torus positions.
    regions (np.ndarray)
        ``(n,)`` acoustic region label per USV.
    settings (dict)
        From :func:`nested_settings`.

    Returns
    -------
    score (float)
        Mean log-likelihood per USV, macro over regions; NaN when no region clears the minimum.
    """

    return float(macro_von_mises_logscore(predictions, positions, regions, metric="torus",
                                          period=settings["torus_period"],
                                          min_region_events=settings["min_region_events"]))


def no_behaviour_predictions(design: dict, settings: dict, n_lags: int) -> np.ndarray:
    """
    Description
    -----------
    Held-out predictions of a model given NO behaviour: the behaviour model's own estimator, weights and
    leave-one-session-out folds, fit on an all-zero design, so all it can learn is each fold's weighted mean
    position.

    It exists for one condition the analysis rests on (ruled 2026-09-15). What is claimed is that the neuron
    beats the BEHAVIOUR MODEL -- not that it carries information beyond behaviour itself -- and that is only
    worth claiming when the behaviour model predicts. The behaviour model's pooled score minus this model's is
    its held-out improvement over having no behaviour at all. It is stored per unit and judged at the cohort
    step.

    MEASURED 2026-09-22, and the condition FAILS for the behaviour model as it is built above: the per-unit
    refit scores BELOW this baseline on both example recording days (-0.0090 and -0.0062). Five features over
    six hundred lags is three thousand columns, and a unit carries about two thousand USVs; P1's own data
    cut to that size reproduces the failure exactly (median +0.00000 over eight draws against +0.06356 on all
    of it), so the shortfall is the USV count and not the ephys sessions. A behaviour model fitted on the
    ephys cohort and held out by recording day DOES predict (+0.042 pooled over 41,087 USVs), and the fix is
    to carry that model in as its predicted position rather than to refit its columns here. An earlier
    docstring quoted "about +0.04" for both days; that figure came from a script whose constant was fitted on
    P1's USVs and applied to ephys ones, which is a baseline at the wrong centre.

    No null is attached, deliberately. The behaviour features' ability to predict USV position was already
    tested when they were selected, against a within-session permutation of USV positions on independent
    animals. What remains per unit is only whether that carries over to these animals and USVs, and the
    held-out improvement shows that directly.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`.
    settings (dict)
        From :func:`nested_settings`.
    n_lags (int)
        History length in frames.

    Returns
    -------
    predictions (np.ndarray)
        ``(n_events, 2)`` held-out predicted torus positions.
    """

    positions, session_index = design["positions"], design["session_index"]
    weights = inverse_region_frequency_weights(design["region_labels"])
    n_features = len(design["feature_names"])
    blank = np.zeros_like(design["behaviour"])
    predicted = np.full_like(positions, np.nan)
    for held_out in sorted({int(s) for s in session_index}):
        test = np.flatnonzero(session_index == held_out)
        train = np.flatnonzero(session_index != held_out)
        predicted[test] = _fit_predict(blank, None, positions, weights, train, test, n_lags, n_features, settings)
    return predicted


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
        The ``nested_vocal_manifold_position_decoding`` block.
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
    return float(macro_von_mises_logscore(predicted, positions, regions, metric="torus",
                                          period=settings["torus_period"],
                                          min_region_events=settings["min_region_events"]))


def null_draw_factory(design: dict, settings: dict, n_lags: int, window_edges: np.ndarray,
                      spikes: dict, durations: dict, width: float, guard_seconds: float,
                      reduced_predicted: np.ndarray, seed: int,
                      sigma_floor: float):
    """
    Description
    -----------
    A ``draw_more(n)`` callable for :func:`shift_null_inference.escalated_empirical_pvalue`.

    The escalation ladder ACCUMULATES -- draws already taken are kept and topped up rather than
    discarded -- which is valid because the draws are exchangeable and the decision to continue
    depends only on whether the floor was reached, never on where the observed value falls among the
    new draws. For that to hold, each batch must use FRESH shifts: the factory advances its own
    counter across calls, so no shift sequence is ever repeated between decades.

    The reduced pass is passed in rather than recomputed. It cannot change between draws, and
    verifying that once is far cheaper than paying for it a hundred thousand times.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`.
    settings (dict)
        The ``nested_vocal_manifold_position_decoding`` block.
    n_lags (int)
        History length in frames.
    window_edges (np.ndarray)
        Left edge (seconds) of each event's spike window, aligned with the design's rows.
    spikes, durations (dict)
        Per session slot: sorted spike times, and the session duration used as the wrap length.
    width (float)
        Spike-window width in seconds.
    guard_seconds (float)
        Guard band at each end of the shift range.
    reduced_predicted (np.ndarray)
        From :func:`reduced_predictions`.
    seed (int)
        Base seed; each draw uses ``seed + running index``.
    sigma_floor (float)
        Floor on the per-session count SD.

    Returns
    -------
    draw_more (Callable)
        ``draw_more(n)`` returning ``n`` fresh null values of the added score.
    """

    state = {"drawn": 0}

    def draw_more(n_draws: int) -> np.ndarray:
        values = np.empty(int(n_draws), dtype=np.float64)
        for i in range(int(n_draws)):
            rng = np.random.default_rng(seed + state["drawn"] + i)
            column = shifted_neural_column(design, window_edges, spikes, durations, width, rng,
                                           guard_seconds, sigma_floor)
            values[i] = nested_scores(design, settings, n_lags, neural=column,
                                      reduced_predicted=reduced_predicted)["added"]
        state["drawn"] += int(n_draws)
        return values

    return draw_more
