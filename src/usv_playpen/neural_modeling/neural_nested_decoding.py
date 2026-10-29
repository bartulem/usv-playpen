"""
@author: bartulem
Nested vocal-manifold position decoding: does a neuron carry vocal output type beyond the behaviour
that predicts it?

Two nested models of the same target, the upcoming USV's position on the QLVM torus, differ in exactly
the neuron (reduced: behaviour; full: behaviour plus the neuron's spike count in the USV's window), and
the statistic is what the neuron adds, held out. A unit whose spiking is a function of the measured
behaviour adds nothing; a unit that adds something carries information beyond the measured behaviour.

The tested statistic is the posterior form (:func:`posterior_nested_gain`): the vocalization-identity
Bayes decoder with the behaviour control entering as the prior, a von Mises around the control's
predicted position multiplied into the occupancy prior (:func:`behaviour_prior_log_density`). The
point-regression form (:func:`nested_scores`, :class:`NestedTorusRegression`) is kept as a
fine-position descriptor and is never a route to passing.

The behaviour control's features are read from the selection result for the run's target
(``vocalization_settings.vocal_target``) and checked against the frozen list in settings
(:func:`reduced_model_features`). Scoring uses the macro von Mises log-score. The categorical
counterpart of the posterior statistic lives in ``neural_category_decoding``.
The neuron enters as one per-session z-scored count rather than a lagged spike history, so the test
cannot pass on the neuron's responses to earlier USVs in the bout.
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
from ..os_utils import configure_path
from .deviance_metrics import finite_mean, finite_min
from .neural_design_assembly import lagged_design, onset_history_anchor_frames
from .neural_vocal_decoding import decode_gain


class NestedTorusRegression(SmoothTorusManifoldRegression):
    """
    Description
    -----------
    The torus regressor with a block-diagonal smoothness penalty, so a design that mixes temporal
    behaviour blocks with non-temporal neural columns is penalised correctly.

    Each behaviour feature receives the parent's ``D^T D`` smoothness block over its lags; the neural
    columns have no time axis and receive ridge only (``l2_reg`` reaches every column through the
    parent's ``+ l2_reg * I``). The parent's ``fit`` checks that ``X`` has ``n_features * n_time_bins``
    columns, so the design is reported as one "feature" of ``total_columns`` bins; those numbers are
    used only by the penalty builder, which is overridden here.

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
        1 or 2; passed through to the parent's operator so the boundary handling is the parent's.
    period (float)
        Torus period per axis.
    huber_delta (float)
        Huber transition point of the parent's residual loss, in torus units; production passes
        ``nested_vocal_decoding.huber_delta``. The default equals the shipped value.
    learning_rate (float)
        Initial Adam step size of the parent's optimiser (cosine-decayed to zero over ``max_iter``);
        production passes ``nested_vocal_decoding.learning_rate``. The default equals the
        shipped value.
    max_iter (int)
        Optimiser step cap; production passes ``nested_vocal_decoding.max_iter``. The
        default equals the shipped value.
    tol (float)
        Convergence tolerance on the parameter-change norm; production passes
        ``nested_vocal_decoding.tol``. The default equals the shipped value.
    random_state (int)
        Seed of the weight initialisation; production passes
        ``nested_vocal_decoding.random_state``. The default equals the shipped value.

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

        The per-feature block is taken from a bare parent instance rather than rebuilt here, so the
        finite-difference operator, including the reflective boundary rows the parent adds at order 2,
        is the parent's by construction.

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


#: The two decoded targets. Every unit is decoded on both, each against its own behaviour control, so
#: the neuron is always tested on the target the behavioural model predicts: the torus position, or the
#: category. ``vocalization_settings.vocal_target`` names the one the claims use.
VOCAL_TARGETS = ("manifold", "multinomial")


def check_target(target: str) -> str:
    """
    Description
    -----------
    Refuse a target that is not one of :data:`VOCAL_TARGETS`.

    Parameters
    ----------
    target (str)
        The target to check.

    Returns
    -------
    target (str)
        The same target, once checked.
    """

    if target not in VOCAL_TARGETS:
        msg = f"vocal target must be one of {list(VOCAL_TARGETS)}; got {target!r}."
        raise ValueError(msg)
    return target


def vocal_target(settings: dict) -> str:
    """
    Description
    -----------
    The target the claims use, ``vocalization_settings.vocal_target``, refused unless it is one of
    :data:`VOCAL_TARGETS`.

    Every unit is decoded on both targets regardless; this setting picks which one's identity and
    nested p-values enter the cohort correction and which identity p-value decides whether vocal gating
    runs. Changing it later needs only the cohort consolidation re-run, not the units.

    Parameters
    ----------
    settings (dict)
        Full neural-modeling settings.

    Returns
    -------
    target (str)
        ``"manifold"`` or ``"multinomial"``.
    """

    target = settings["vocalization_settings"]["vocal_target"]
    if target not in VOCAL_TARGETS:
        msg = f"vocalization_settings.vocal_target must be one of {list(VOCAL_TARGETS)}; got {target!r}."
        raise ValueError(msg)
    return target


def target_section(step: str, target: str) -> str:
    """
    Description
    -----------
    The result-file section holding one target's result of a target-dependent analysis.

    Identity and the nested decoding run once per target, and each target's result is its own
    section, ``<step>_<target>`` (e.g. ``vocalization_identity_manifold``), written as soon as it is
    done.

    Parameters
    ----------
    step (str)
        ``"vocalization_identity"`` or ``"nested_vocal_decoding"``.
    target (str)
        One of :data:`VOCAL_TARGETS`.

    Returns
    -------
    section (str)
        The section name.
    """

    return f"{step}_{check_target(target)}"


def reduced_model_features(settings: dict, target: str, require_selection_file: bool = True) -> list:
    """
    Description
    -----------
    The behaviour control's feature set for one target: read from the selection result file and
    checked against the frozen list in settings, so a re-run of the selection cannot silently redefine
    the behaviour the neuron is measured against. Both settings are dicts keyed by target.

    Two artifact shapes are accepted: a cohort selection's ``selected_features``, or a behavioural
    selection's ``steps`` (its last step's ``current_features``). Only feature names are taken from a
    behavioural selection; the control itself is always fitted in domain by ``fit_control_models``.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict; ``data.behaviour_selection_result_path`` and
        ``nested_vocal_decoding.reduced_model_features`` are read at the target's key.
    target (str)
        One of :data:`VOCAL_TARGETS`.
    require_selection_file (bool)
        When False, a missing selection file is tolerated and the frozen list is returned unchecked --
        for running where the lab share is not mounted. A file that IS present is always checked.

    Returns
    -------
    features (list)
        The behaviour control's feature names, in selection order.
    """

    check_target(target)
    frozen = list(settings["nested_vocal_decoding"]["reduced_model_features"][target])
    path = pathlib.Path(configure_path(settings["data"]["behaviour_selection_result_path"][target]))
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
        # A behavioural-modelling selection: only its feature names are adopted, never its
        # coefficients, since the control is fitted in domain by `fit_control_models`.
        selected = list(selection["steps"][-1]["current_features"])
    else:
        msg = (f"{path.name} is neither a cohort selection result ('selected_features') nor a "
               f"behavioural-modelling selection ('steps'). Point "
               f"data.behaviour_selection_result_path[{target!r}] at the artifact written by "
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
    Build the nested decoding's design: each event's behavioural history, plus the neuron as one column.

    Rows are USVs. Each behavioural feature contributes ``n_lags`` columns, its history ending at the
    frame before onset, followed by a single neural column: the unit's spike count in the USV's
    window, z-scored per session. The neuron enters as a count, as in the
    vocalization-identity decoder, rather than as a lagged spike history, which would mostly carry its
    responses to earlier USVs in the bout.

    Events without a full behavioural history are dropped, so this design can hold fewer events than
    the identity decoder; both counts are returned per session.

    Parameters
    ----------
    events (dict)
        From ``assemble_unit_vocal_events``: ``vocalization_start``, ``session_index``, ``counts``,
        ``positions``, ``region_labels``, ``session_ids``.
    per_session (dict)
        From ``assemble_unit_sessions``: per session ``feature_time_series``, ``feature_names``,
        ``fps`` and ``n_frames``.
    feature_names (list)
        The behavioural features, in order. Selected by name, since per-session column order differs.
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
        # By name, never by position: per-session column order differs.
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
    Behaviour as a prior over the torus grid: for each event, the fold's occupancy prior multiplied by
    the von Mises evidence the behaviour control supplies for that USV.

    The control gives a point ``x_hat``; with the occupancy base rate ``p(x)`` the conditioned prior is
    ``p(x | x_hat) ~ occupancy(x) * vM(x; x_hat, kappa)``, so a control that knows nothing
    (``kappa = 0``) reduces to the occupancy prior. The product is renormalised over
    the fold's masked grid, which absorbs ``log I0(kappa)``.

    One shared ``kappa`` per fold is set on the training sessions alone, in one of two ways:
    ``train_residuals`` fits it to the training events' wrap-aware residuals around the control's
    prediction; ``train_tuned_inner_loso`` chooses it from ``kappa_grid`` by inner
    leave-one-session-out, maximising the held-in log prior at the truth over the occupancy prior.

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
        msg = f"behaviour_prior.form must be 'von_mises'; got {prior['form']!r}. No other form is supported."
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
        # A residual fit double-counts the base rate for a shrunk predictor, so kappa is tuned by
        # inner leave-one-session-out; a control that knows nothing settles at kappa = 0. The grid is
        # non-negative, so the evidence can be declined but never inverted. An over-smoothed occupancy
        # prior would let a constant predictor earn kappa > 0, which is why the KDE bandwidth must stay
        # held-out-selected.
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
               f"'train_tuned_inner_loso'; got {prior['concentration']!r}. No other option is supported.")
        raise ValueError(msg)
    return kappa, tilted(kappa)


def behaviour_priors(folds: list, predicted_positions: np.ndarray, positions: np.ndarray,
                     nested: dict) -> tuple[list, list]:
    """
    Description
    -----------
    The behaviour prior for every fold, built once.

    A prior depends only on positions, the control's predictions and the fold's geometry, none of
    which a null draw changes, so it is computed here and handed to every draw.

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

    It is ``decode_gain`` (same tuning surface, ridge, exposure offsets, amplitude calibration, grid
    and mask) with the fold's occupancy prior replaced by :func:`behaviour_prior_log_density`, so
    ``added`` is log posterior(truth | neuron, behaviour) minus log prior(truth | behaviour), averaged
    over held-out events. The identity gain is the same quantity under the unconditioned prior.

    Reported beside it: ``behaviour_over_no_behaviour`` (log prior(truth | behaviour) minus log
    occupancy prior(truth), what behaviour alone buys) and ``full_over_no_behaviour`` (the sum). The
    first is invariant under the permutation null, since a prior row travels with its position; only
    ``added`` is tested.

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


#: The only values implemented for each option `nested_vocal_decoding` names, so a
#: settings file that declares any other value is refused rather than silently scored as these.
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
    The ``nested_vocal_decoding`` block with the shared manifold period merged in, validated.

    The torus period is a property of the QLVM manifold and lives in the shared
    ``vocalization_settings`` block; it is merged here so this module's functions can take one block.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.

    Returns
    -------
    block (dict)
        A copy of ``nested_vocal_decoding`` carrying ``torus_period``, refused if it declares an
        option this module does not implement.
    """

    block = dict(settings["nested_vocal_decoding"])
    block["torus_period"] = settings["vocalization_settings"]["torus_period"]
    check_settings(block)
    return block


def check_settings(settings: dict) -> None:
    """
    Description
    -----------
    Refuse a ``nested_vocal_decoding`` block that declares an option this module does not implement.

    Parameters
    ----------
    settings (dict)
        The ``nested_vocal_decoding`` block.

    Returns
    -------
    """

    for key, allowed in IMPLEMENTED_OPTIONS.items():
        value = settings[key]
        if value not in allowed:
            others = ", ".join(repr(a) for a in allowed)
            msg = (f"nested_vocal_decoding.{key} = {value!r} is not implemented; this module supports "
                   f"{others}.")
            raise ValueError(msg)
    for (outer, inner), allowed in IMPLEMENTED_NESTED_OPTIONS.items():
        value = settings[outer][inner]
        if value not in allowed:
            others = ", ".join(repr(a) for a in allowed)
            msg = (f"nested_vocal_decoding.{outer}.{inner} = {value!r} is not "
                   f"implemented; this module supports {others}.")
            raise ValueError(msg)


def _fit_predict(behaviour: np.ndarray, neural: np.ndarray, positions: np.ndarray,
                 weights: np.ndarray, train: np.ndarray, test: np.ndarray,
                 n_lags: int, n_features: int, settings: dict) -> np.ndarray:
    """
    Description
    -----------
    Fit one model on the training rows and predict the held-out ones.

    ``neural`` of None gives the reduced model (behaviour alone); passing the column gives the full
    model. Both use the same estimator and penalty policy, so the pair differs in exactly the neural
    column.

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
        The ``nested_vocal_decoding`` block.

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
    The reduced model's pooled held-out predictions, computed once and handed to
    :func:`nested_scores`.

    The reduced fit depends only on behaviour, positions and region labels, none of which the neural
    column changes.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`.
    settings (dict)
        The ``nested_vocal_decoding`` block.
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


def nested_scores(design: dict, settings: dict, n_lags: int,
                  neural: np.ndarray = None, reduced_predicted: np.ndarray = None) -> dict:
    """
    Description
    -----------
    The point-regression descriptor: the pooled added macro von Mises log-score of
    ``position ~ behaviour + neuron`` over ``position ~ behaviour``, across leave-one-session-out folds.

    Scores are pooled over folds rather than fold-averaged, because the concentration is fitted at
    scoring time and the macro score drops regions below ``min_region_events``, which a thin fold
    would do unevenly. Each model fits its own concentration. Training rows carry equal-region
    weights. The leave-one-fold-out minimum (the worst pooled added score with any one fold dropped)
    is reported as a descriptor.

    Parameters
    ----------
    design (dict)
        From :func:`nested_design`.
    settings (dict)
        The ``nested_vocal_decoding`` block.
    n_lags (int)
        History length in frames.
    neural (np.ndarray)
        Neural column to use instead of ``design['neural']``; None uses the observed column.
    reduced_predicted (np.ndarray)
        Precomputed reduced-model predictions (:func:`reduced_predictions`); None fits them here.

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
    The pooled macro von Mises log-score of torus predictions, the scoring rule every model in this
    module is compared under.

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
    Held-out predictions of a model given no behaviour: the behaviour model's own estimator, weights and
    leave-one-session-out folds, fitted on an all-zero design, so all it can learn is each fold's weighted
    mean position.

    Beating the behaviour control is only worth claiming where that control predicts, so the control's
    pooled score minus this model's is stored per unit as its held-out improvement over no behaviour and
    judged at the cohort step. No null is attached to this improvement.

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
