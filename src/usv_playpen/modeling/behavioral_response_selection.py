"""
@author: bartulem
Module for the nested block comparison behind the behavioral-response analysis.

Where ``model_selection.py`` asks which individual *feature* best predicts a
vocal target, this module asks a block question: does the partner's vocal trace
explain anything about a behavioral variable that the kinematic and social
features do not already explain? That is a comparison of two nested models fit
on identical folds — a baseline of everything except the vocal block, and the
same model with the vocal block added — and it has no counterpart among the five
existing selectors, none of which carries a notion of a feature block.

Key scientific and computational components:

1.  A paired, per-fold increment. The two models see the same rows in the same
    folds, so their scores are paired and the fold-wise difference is the
    statistic; a bootstrap over folds gives its interval. This mirrors
    ``model_selection._fold_paired_margin_bootstrap``, which does the same thing
    for 2-D manifold coordinates, with a scalar deviance score in place of the
    von Mises / distance-correlation ones.
2.  A null that destroys vocal *timing* and nothing else -- DEFERRED, and not
    part of the pipeline. ``circular_shift_rows_within_session`` and
    ``paired_fold_margin`` implement it and are unit-tested, but NOTHING in the
    selection path calls them: every acceptance, including the vocal step's, uses
    the paired 1SE rule, because held-out cross-validation across sessions already
    establishes that a block improving prediction on unseen data carries
    information. The shift null answers the narrower question of whether that
    information lies in the TIMING of the calls, which is a follow-up on an effect
    you already have rather than a criterion for deciding whether you have one.
    Run it once, by hand, only if the vocal step is accepted; the settings keys
    ``n_shift_draws`` and ``shift_null_min_seconds`` exist for that run. (Note
    ``shift_null_min_seconds`` is ALSO read by
    ``BehavioralResponsePipeline._null_target`` for a different purpose -- a
    whole-fold roll of the response in the univariate screen.) The block is
    circularly shifted relative to behavior, leaving its bout structure, rate and
    marginal distribution exactly as observed. Permuting instead would produce a
    trace that is no longer bursty, so "any sparse bursty regressor would have
    done this" would never be represented in the null — which is precisely the
    alternative the test exists to exclude.
3.  Shifting by whole rows is exact, not an approximation. Anchors are tiled at a
    stride equal to the history length, so a circular shift of the underlying
    trace by ``k`` strides maps anchor ``i``'s window exactly onto anchor
    ``i - k``'s window. Rolling the block's rows within a session therefore keeps
    every window a real, contiguous slice of the real trace; the only consequence
    is that offsets are quantized to the stride.
4.  A minimum offset, symmetric at both ends. Small shifts leave real alignment
    intact, and under wraparound a shift close to the session length is nearly the
    identity — so offsets are drawn from ``[min, n_rows - min]`` with the single
    ``shift_null_min_seconds`` setting fixing both ends.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pygam import GAM, te
from sklearn.metrics import mean_gamma_deviance
from sklearn.model_selection import StratifiedGroupKFold

from ..os_utils import atomic_output_path
from .model_selection import get_unrolled_X_for_multivariate
from .modeling_metadata import (
    build_selection_metadata,
    derive_experimental_condition,
    extract_metadata_blocks,
    inject_metadata,
)
from .modeling_utils import (
    format_run_header,
    format_run_summary,
    format_selection_step,
    mean_absolute_error_1d,
    paired_one_se_improvement,
    pearson_r_safe,
    root_mean_squared_error,
    spearman_r_safe,
)


def circular_shift_rows_within_session(block_data: np.ndarray,
                                       groups: np.ndarray,
                                       rng: np.random.Generator,
                                       min_shift_rows: int) -> np.ndarray:
    """
    Circularly shifts a predictor block within each session to destroy its timing.

    Rows belonging to one session are rolled as a unit by a random offset, so the
    block's values keep their internal structure and their session membership but
    no longer line up with the behavior they were recorded alongside. Because the
    anchors are tiled at a stride equal to the history length, rolling by ``k``
    rows is an exact circular shift of the underlying continuous trace by ``k``
    strides — every window remains a genuine contiguous slice of real data rather
    than a synthetic reassembly.

    The offset is drawn from ``[min_shift_rows, n_session_rows - min_shift_rows]``.
    The lower bound keeps the shift past the longest behavioral autocorrelation;
    the upper bound is its mirror, because a roll close to the full session length
    wraps almost all the way round and is nearly the identity.

    Parameters
    ----------
    block_data : np.ndarray
        Array of shape ``(n_rows, ...)`` whose first axis is row-aligned with
        ``groups``. Any trailing shape is preserved.
    groups : np.ndarray
        1-D array of session identifiers, one per row.
    rng : np.random.Generator
        Seeded generator; passed in rather than created here so a whole null
        distribution is reproducible from one seed.
    min_shift_rows : int
        Minimum (and mirrored maximum) offset, in rows.

    Returns
    -------
    shifted : np.ndarray
        A new array of the same shape and dtype, rolled per session. Sessions with
        too few rows to admit a legal offset are returned unshifted, which makes
        the null conservative rather than silently wrong.
    """

    if min_shift_rows < 1:
        msg = f"`min_shift_rows` must be >= 1, got {min_shift_rows}."
        raise ValueError(msg)
    if block_data.shape[0] != groups.shape[0]:
        msg = (
            f"`block_data` has {block_data.shape[0]} rows but `groups` has "
            f"{groups.shape[0]}; they must be row-aligned."
        )
        raise ValueError(msg)

    shifted = block_data.copy()
    for session_id in np.unique(groups):
        session_rows = np.flatnonzero(groups == session_id)
        n_session_rows = session_rows.size
        highest_legal = n_session_rows - min_shift_rows
        if highest_legal < min_shift_rows:
            continue
        offset = int(rng.integers(min_shift_rows, highest_legal + 1))
        shifted[session_rows] = block_data[np.roll(session_rows, offset)]

    return shifted


def gamma_explained_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes explained Gamma deviance against an intercept-only null.

    Matches the ``D^2`` definition already inlined in the bout-parameter pipeline
    and its selector: the null predicts the mean of the evaluation set itself, so
    the score is a genuine out-of-sample explained-deviance fraction rather than
    one referenced to a training-set constant.

    Parameters
    ----------
    y_true : np.ndarray
        Observed targets, strictly positive.
    y_pred : np.ndarray
        Model predictions for the same rows.

    Returns
    -------
    d2 : float
        Explained deviance; ``0.0`` when the null deviance is zero (a constant
        evaluation set), which leaves no deviance to explain.
    """

    y_true_safe = np.maximum(y_true, 1e-6)
    y_pred_safe = np.maximum(y_pred, 1e-6)
    null_pred = np.full_like(y_true_safe, float(np.mean(y_true_safe)))

    residual_deviance = mean_gamma_deviance(y_true_safe, y_pred_safe)
    null_deviance = mean_gamma_deviance(y_true_safe, null_pred)
    if null_deviance == 0:
        return 0.0
    return float(1.0 - (residual_deviance / null_deviance))


def gaussian_explained_variance(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    Computes explained variance on the log scale against a mean-only null.

    The counterpart of :func:`gamma_explained_deviance` for the Gaussian-on-log
    arm. Both are referenced to the mean of the evaluation set itself, so each is
    an out-of-sample explained fraction within its own likelihood; the two are
    deliberately **not** on a common scale, and only the *increments* they produce
    are compared across arms.

    Parameters
    ----------
    y_true_log : np.ndarray
        Observed targets on the log scale.
    y_pred_log : np.ndarray
        Model predictions on the log scale.

    Returns
    -------
    r2 : float
        Explained variance; ``0.0`` when the evaluation set is constant.
    """

    residual_ss = float(np.sum((y_true_log - y_pred_log) ** 2))
    total_ss = float(np.sum((y_true_log - np.mean(y_true_log)) ** 2))
    if total_ss == 0:
        return 0.0
    return float(1.0 - (residual_ss / total_ss))


def paired_fold_margin(baseline_scores: np.ndarray,
                       full_scores: np.ndarray,
                       rng: np.random.Generator,
                       n_bootstrap: int,
                       ci_level: float) -> dict[str, Any]:
    """
    Summarizes the per-fold paired improvement of a full model over its baseline.

    The two models are fit on identical folds, so their scores are paired and the
    fold-wise difference removes the between-fold variation that dominates an
    unpaired comparison. The interval comes from resampling folds, which is the
    grain the models actually vary at; comparing a paired improvement against an
    unpaired standard error — as the repo's older 1-SE selectors do — is
    substantially over-conservative.

    Parameters
    ----------
    baseline_scores : np.ndarray
        1-D per-fold scores for the reduced model (higher is better).
    full_scores : np.ndarray
        1-D per-fold scores for the full model, fold-aligned with the baseline.
    rng : np.random.Generator
        Seeded generator for the fold bootstrap.
    n_bootstrap : int
        Number of fold resamples.
    ci_level : float
        Two-sided confidence level, e.g. ``0.99``.

    Returns
    -------
    summary : dict
        ``mean_margin``, ``ci_low``, ``ci_high``, ``p_value`` (fraction of
        bootstrap means at or below zero), ``n_folds`` and ``folds_positive``.
    """

    if baseline_scores.shape != full_scores.shape:
        msg = (
            f"Fold-score arrays must be aligned; got {baseline_scores.shape} "
            f"and {full_scores.shape}."
        )
        raise ValueError(msg)

    margins = np.asarray(full_scores, dtype=float) - np.asarray(baseline_scores, dtype=float)
    finite_margins = margins[np.isfinite(margins)]
    if finite_margins.size == 0:
        return {
            'mean_margin': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'),
            'p_value': float('nan'), 'n_folds': 0, 'folds_positive': 0,
        }

    boot_means = np.array([
        float(np.mean(finite_margins[rng.integers(0, finite_margins.size, finite_margins.size)]))
        for _ in range(n_bootstrap)
    ])
    tail = (1.0 - ci_level) / 2.0

    return {
        'mean_margin': float(np.mean(finite_margins)),
        'ci_low': float(np.percentile(boot_means, 100.0 * tail)),
        'ci_high': float(np.percentile(boot_means, 100.0 * (1.0 - tail))),
        'p_value': float(np.mean(boot_means <= 0.0)),
        'n_folds': int(finite_margins.size),
        'folds_positive': int(np.sum(finite_margins > 0.0)),
    }


def fit_block_across_folds(feature_arrays: list[np.ndarray],
                           y_global: np.ndarray,
                           cv_folds: list[tuple[np.ndarray, np.ndarray]],
                           history_frames: int,
                           n_splines_value: int,
                           n_splines_time: int,
                           gam_kwargs: dict[str, Any],
                           likelihood: str = 'gamma',
                           feature_names: list[str] | None = None,
                           collect_diagnostics: bool = False) -> dict[str, Any]:
    """
    Fits one multivariate Gamma GAM per fold and scores it out of sample.

    Each feature contributes a ``te(value, lag)`` tensor term over its own history
    window, matching the engine the rest of ``modeling/`` uses. Per-frame
    predictions are averaged on the linear-predictor scale before the inverse link
    (``exp(mean(eta))``, not ``mean(exp(eta))``) so no Jensen bias is introduced by
    the aggregation — the same correction the bout-parameter selector applies.

    Parameters
    ----------
    feature_arrays : list of np.ndarray
        One ``(n_rows, history_frames)`` history matrix per feature, in the order
        the tensor terms should be built.
    y_global : np.ndarray
        1-D target vector, row-aligned with every feature matrix.
    cv_folds : list of tuple
        ``(train_index, test_index)`` pairs; the same folds must be used for the
        baseline and full models so their scores are paired.
    history_frames : int
        Number of lags per feature.
    n_splines_value : int
        Spline count along the feature-value axis of each tensor term.
    n_splines_time : int
        Spline count along the lag axis of each tensor term.
    gam_kwargs : dict
        Extra keyword arguments forwarded to ``pygam.GAM`` (``lam``, ``max_iter``,
        ``tol``).
    likelihood : str, optional
        ``'gamma'`` (default) fits a Gamma GAM with a log link to ``y`` in
        native units and scores explained Gamma deviance. ``'lognormal'`` fits a
        Gaussian GAM with an identity link to ``log(y)`` -- which is the
        lognormal likelihood by definition, hence the name; calling it
        "gaussian_log" would wrongly suggest a log *link*, and the link here is
        the identity -- and scores explained variance on that scale. The two are internally consistent but on
        different scales, so only the *increments* they yield are comparable
        across arms -- back-transforming a log-scale fit would give a geometric
        mean rather than ``E[y]``, the very fit/score mismatch the bout-parameter
        pipeline was rewritten to remove.

    Returns
    -------
    result : dict
        ``d2`` (per-fold explained deviance or variance, per ``likelihood``),
        ``fit_time`` (per fold), and
        ``failed_folds`` naming any fold whose fit raised, which scores ``NaN``
        rather than aborting the whole comparison.
    """

    if not feature_arrays:
        msg = "`feature_arrays` is empty; a model needs at least one feature."
        raise ValueError(msg)
    if likelihood not in ('gamma', 'lognormal'):
        msg = f"`likelihood` must be 'gamma' or 'lognormal', got '{likelihood}'."
        raise ValueError(msg)

    tensor_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
    for feature_position in range(1, len(feature_arrays)):
        tensor_terms = tensor_terms + te(
            2 * feature_position,
            2 * feature_position + 1,
            n_splines=[n_splines_value, n_splines_time],
        )

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(feature_arrays))]

    fold_d2: list[float] = []
    fold_fit_time: list[float] = []
    failed_folds: list[int] = []
    diagnostics: dict[str, list[Any]] = {
        'y_true': [], 'y_pred': [], 'test_indices': [], 'filter_shapes': [],
        'spearman_r': [], 'pearson_r': [], 'mae': [], 'rmse': [], 'residual_deviance': [],
    }
    time_indices = np.arange(history_frames, dtype=float)

    def _record_diagnostics(gam: GAM, y_observed: np.ndarray,
                            y_predicted: np.ndarray, test_index: np.ndarray) -> None:
        """
        Stores this fold's predictions, descriptive metrics and filter shapes.

        Filters come from the partial-dependence trick the onset selector
        already uses: predict on an all-zero grid, then re-predict with one
        feature's value column set to 1.0 and difference the two. Features are
        pooled-z-scored, so the curve reads as the effect of a +1 SD increase in
        that feature at each lag -- which is what makes the response latency
        legible rather than inferred.

        Parameters
        ----------
        gam : GAM
            The fitted model for this fold, queried for the filter shapes.
        y_observed : np.ndarray
            Held-out targets for this fold.
        y_predicted : np.ndarray
            Model predictions aligned with ``y_observed``.
        test_index : np.ndarray
            Row indices of this fold's held-out rows, kept so the per-fold
            arrays can be reassembled against the full design.

        Returns
        -------
        None
        """

        diagnostics['y_true'].append(np.asarray(y_observed, dtype=float))
        diagnostics['y_pred'].append(np.asarray(y_predicted, dtype=float))
        diagnostics['test_indices'].append(np.asarray(test_index, dtype=int))
        diagnostics['spearman_r'].append(spearman_r_safe(y_observed, y_predicted))
        diagnostics['pearson_r'].append(pearson_r_safe(y_observed, y_predicted))
        diagnostics['mae'].append(mean_absolute_error_1d(y_observed, y_predicted))
        diagnostics['rmse'].append(root_mean_squared_error(y_observed, y_predicted))
        if likelihood == 'gamma':
            diagnostics['residual_deviance'].append(
                float(mean_gamma_deviance(np.maximum(y_observed, 1e-6),
                                          np.maximum(y_predicted, 1e-6))),
            )
        else:
            diagnostics['residual_deviance'].append(
                float(np.mean((y_observed - y_predicted) ** 2)),
            )

        base_grid = np.zeros((history_frames, 2 * len(feature_arrays)))
        for feature_position in range(len(feature_arrays)):
            base_grid[:, feature_position * 2 + 1] = time_indices
        base_prediction = gam.predict_mu(base_grid)

        fold_filters = {}
        for feature_position, feature_name in enumerate(feature_names):
            test_grid = base_grid.copy()
            test_grid[:, feature_position * 2] = 1.0
            fold_filters[feature_name] = np.asarray(
                gam.predict_mu(test_grid) - base_prediction, dtype=float,
            ).flatten()
        diagnostics['filter_shapes'].append(fold_filters)

    for fold_index, (train_index, test_index) in enumerate(cv_folds):
        try:
            train_unrolled = get_unrolled_X_for_multivariate(
                feature_data_dict_list=[arr[train_index] for arr in feature_arrays],
                history_frames=history_frames,
            )
            test_unrolled = get_unrolled_X_for_multivariate(
                feature_data_dict_list=[arr[test_index] for arr in feature_arrays],
                history_frames=history_frames,
            )

            y_train = y_global[train_index]
            y_test = y_global[test_index]

            if likelihood == 'gamma':
                gam = GAM(tensor_terms, distribution='gamma', link='log', **gam_kwargs)
                fit_start = time.perf_counter()
                gam.fit(train_unrolled, np.repeat(y_train + 1e-6, history_frames))
                fold_fit_time.append(float(time.perf_counter() - fit_start))

                # Aggregate the per-frame predictions on the linear-predictor
                # (eta = log mu) scale before inverting the link, so no Jensen
                # bias is introduced by averaging on the natural scale.
                eta_test = np.log(gam.predict_mu(test_unrolled)).reshape(len(y_test), history_frames)
                y_pred = np.exp(np.mean(eta_test, axis=1))
                fold_d2.append(gamma_explained_deviance(y_true=y_test, y_pred=y_pred))
                if collect_diagnostics:
                    _record_diagnostics(gam, y_test, y_pred, test_index)
            else:
                # Gaussian on log(y): the link is the identity, so the per-frame
                # predictions already live on the scale the score is computed on
                # and a plain mean is the correct aggregation.
                gam = GAM(tensor_terms, distribution='normal', link='identity', **gam_kwargs)
                fit_start = time.perf_counter()
                gam.fit(train_unrolled, np.repeat(np.log(y_train), history_frames))
                fold_fit_time.append(float(time.perf_counter() - fit_start))

                pred_test = gam.predict(test_unrolled).reshape(len(y_test), history_frames)
                y_pred_log = np.mean(pred_test, axis=1)
                fold_d2.append(gaussian_explained_variance(
                    y_true_log=np.log(y_test), y_pred_log=y_pred_log,
                ))
                if collect_diagnostics:
                    # Scored on the log scale, so the descriptive metrics are
                    # reported there too rather than back-transformed: exp() of a
                    # mean log prediction is a geometric mean, not E[y].
                    _record_diagnostics(gam, np.log(y_test), y_pred_log, test_index)
        except Exception:
            # A fold that fails to converge must not abort the paired comparison:
            # it is scored NaN, dropped by `paired_fold_margin`, and named here so
            # a run with many failures is visible rather than quietly thinned.
            failed_folds.append(fold_index)
            fold_d2.append(float('nan'))
            fold_fit_time.append(float('nan'))
            if collect_diagnostics:
                # Placeholders keep every diagnostics list one-entry-per-FOLD.
                # Without them a failed fold shortens these lists while `d2` keeps
                # its NaN, so `per_session_scores` -- which pairs the two models
                # by fold POSITION -- would silently attribute one model's fold k
                # to the other's fold k+1 whenever the two disagree about which
                # folds failed. Equal-sized test sides make that misalignment
                # produce numbers rather than an IndexError.
                diagnostics['y_true'].append(np.empty(0, dtype=float))
                diagnostics['y_pred'].append(np.empty(0, dtype=float))
                diagnostics['test_indices'].append(np.empty(0, dtype=int))
                diagnostics['filter_shapes'].append({})
                for metric_name in ('spearman_r', 'pearson_r', 'mae', 'rmse',
                                    'residual_deviance'):
                    diagnostics[metric_name].append(float('nan'))

    result: dict[str, Any] = {
        'd2': np.array(fold_d2, dtype=float),
        'fit_time': np.array(fold_fit_time, dtype=float),
        'failed_folds': failed_folds,
        'score_scale': 'native' if likelihood == 'gamma' else 'log',
    }
    if collect_diagnostics:
        result.update(diagnostics)
    return result


def fraction_of_remaining_deviance(baseline_scores: np.ndarray,
                                   full_scores: np.ndarray) -> dict[str, Any]:
    """
    Expresses the increment as a share of what the baseline left unexplained.

    An absolute margin in D2 is not comparable across configurations, because the
    baseline absorbs a different share of the deviance in each one: the same
    margin means something very different against a baseline at 0.05 than against
    one at 0.60. Dividing by ``1 - baseline`` answers the question actually being
    asked -- of what the kinematics could NOT explain, how much does the vocal
    block account for?

    Computed per fold and then averaged, rather than as a ratio of fold means, so
    a single fold with an unusual baseline cannot distort the summary through the
    denominator.

    Parameters
    ----------
    baseline_scores : np.ndarray
        1-D per-fold scores for the reduced model.
    full_scores : np.ndarray
        1-D per-fold scores for the full model, fold-aligned with the baseline.

    Returns
    -------
    summary : dict
        ``per_fold`` (the fold-wise fractions), ``mean`` and ``n_folds``. Folds
        whose baseline sits at exactly 1.0 leave no deviance to explain and are
        dropped rather than dividing by zero.
    """

    baseline = np.asarray(baseline_scores, dtype=float)
    full = np.asarray(full_scores, dtype=float)
    remaining = 1.0 - baseline

    usable = np.isfinite(baseline) & np.isfinite(full) & (remaining != 0.0)
    per_fold = np.full(baseline.shape, np.nan, dtype=float)
    per_fold[usable] = (full[usable] - baseline[usable]) / remaining[usable]

    finite = per_fold[np.isfinite(per_fold)]
    return {
        'per_fold': per_fold,
        'mean': float(np.mean(finite)) if finite.size else float('nan'),
        'n_folds': int(finite.size),
    }


def per_session_scores(baseline_diagnostics: dict[str, Any],
                       full_diagnostics: dict[str, Any],
                       groups_global: np.ndarray,
                       likelihood: str) -> dict[str, dict[str, float]]:
    """
    Breaks the fold-level scores down to one number per session.

    A fold's test side holds several whole sessions pooled together, so a
    fold-level score cannot distinguish an effect present in most of the cohort
    from one carried by a handful of unusual pairs -- a strong session simply
    lifts whichever fold it landed in. Scoring each session separately, on the
    fold where it was held out, makes "is this general, or is it a few animals?"
    a lookup instead of a re-analysis.

    Nothing is refit: this re-scores predictions already stored per fold, split by
    the session each test row belongs to.

    Parameters
    ----------
    baseline_diagnostics : dict
        Fold diagnostics for the reduced model, carrying ``y_true``, ``y_pred``
        and ``test_indices``.
    full_diagnostics : dict
        The same for the full model, from the same folds.
    groups_global : np.ndarray
        1-D session identifier per row, indexed by ``test_indices``.
    likelihood : str
        Which scorer to apply -- the two arms score on different scales.

    Returns
    -------
    per_session : dict
        ``session_id -> {'baseline_d2', 'full_d2', 'margin', 'n_rows'}``. A session
        appearing in more than one test fold is scored on its pooled rows. Folds
        are Monte Carlo shuffle-splits rather than exhaustive leave-one-session-out,
        so a session that never lands in a test side simply has no entry -- the
        breakdown covers the sessions that were actually held out, not the whole
        cohort by construction.
    """

    score = gamma_explained_deviance if likelihood == 'gamma' else gaussian_explained_variance

    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    for fold_index, test_index in enumerate(full_diagnostics['test_indices']):
        session_labels = groups_global[test_index]
        for session_id in np.unique(session_labels):
            rows = session_labels == session_id
            entry = pooled.setdefault(str(session_id), {'y': [], 'base': [], 'full': []})
            entry['y'].append(full_diagnostics['y_true'][fold_index][rows])
            entry['base'].append(baseline_diagnostics['y_pred'][fold_index][rows])
            entry['full'].append(full_diagnostics['y_pred'][fold_index][rows])

    per_session: dict[str, dict[str, float]] = {}
    for session_id, arrays in pooled.items():
        y_observed = np.concatenate(arrays['y'])
        baseline_d2 = score(y_observed, np.concatenate(arrays['base']))
        full_d2 = score(y_observed, np.concatenate(arrays['full']))
        per_session[session_id] = {
            'baseline_d2': float(baseline_d2),
            'full_d2': float(full_d2),
            'margin': float(full_d2 - baseline_d2),
            'n_rows': int(y_observed.size),
        }
    return per_session


def build_session_folds(y_global: np.ndarray,
                        groups_global: np.ndarray,
                        held_out_session_ids: list[str],
                        n_splits: int,
                        test_proportion: float,
                        random_seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Builds Monte Carlo session-grouped folds over the development rows only.

    Whole sessions are held out per fold, so no session straddles train and test
    and the score is a genuine cross-animal generalization estimate. Folds are
    stratified on quantile-binned targets so every fold spans the range of the
    response rather than concentrating on slow or fast periods. The carve-once
    held-out reserve is excluded from every fold; a leak guard raises if any of
    its rows reach one, because that silently converts the final estimate from an
    honest last look into a second validation set.

    Parameters
    ----------
    y_global : np.ndarray
        1-D target vector for all rows.
    groups_global : np.ndarray
        1-D session identifier per row.
    held_out_session_ids : list of str
        Sessions carved off at extraction time; excluded from every fold.
    n_splits : int
        Number of Monte Carlo folds to generate.
    test_proportion : float
        Fraction of development rows in each fold's test side.
    random_seed : int
        Base seed; fold ``i`` uses ``random_seed + i`` so folds differ but the
        whole set is reproducible.

    Returns
    -------
    cv_folds : list of tuple
        ``(train_index, test_index)`` pairs indexing the FULL arrays.
    """

    held_out = set(held_out_session_ids)
    dev_mask = np.array([g not in held_out for g in groups_global], dtype=bool)
    dev_positions = np.flatnonzero(dev_mask)
    held_positions = np.flatnonzero(~dev_mask)

    if dev_positions.size == 0:
        msg = "Every row belongs to a held-out session; no development data remains to fold."
        raise ValueError(msg)

    n_bins = max(2, min(10, dev_positions.size // 20))
    bin_edges = np.percentile(y_global[dev_mask], np.linspace(0, 100, n_bins + 1))
    if np.unique(bin_edges).size < 3:
        y_binned = np.zeros(dev_positions.size)
    else:
        y_binned = np.digitize(y_global[dev_mask], bin_edges[1:-1])

    groups_dev = groups_global[dev_mask]
    # Clamp to the number of development sessions, matching
    # `BoutParameterPipeline.create_data_splits`. Without it a small cohort makes
    # the univariate screen succeed (it clamps) while selection hard-fails, so the
    # two stages would disagree about whether the data can be folded at all.
    n_sessions_available = int(np.unique(groups_dev).size)
    n_folds_mc = max(2, min(round(1.0 / test_proportion), n_sessions_available))

    cv_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for split_index in range(n_splits):
        splitter = StratifiedGroupKFold(
            n_splits=n_folds_mc,
            shuffle=True,
            random_state=random_seed + split_index,
        )
        try:
            train_dev, test_dev = next(
                splitter.split(np.zeros(y_binned.size), y_binned, groups=groups_dev)
            )
        except (StopIteration, ValueError):
            continue
        cv_folds.append((dev_positions[train_dev], dev_positions[test_dev]))

    if not cv_folds:
        msg = (
            f"No usable folds were generated from {dev_positions.size} development rows "
            f"across {np.unique(groups_dev).size} sessions; check `n_cv_folds` and "
            f"`cv_validation_proportion`."
        )
        raise RuntimeError(msg)

    if held_positions.size > 0:
        held_set = set(held_positions.tolist())
        for train_index, test_index in cv_folds:
            leaked = held_set.intersection(train_index.tolist()) or held_set.intersection(test_index.tolist())
            if leaked:
                msg = (
                    "Held-out session rows leaked into a behavioral-response CV fold; "
                    "the development / held-out split is inconsistent."
                )
                raise AssertionError(msg)

    return cv_folds


def score_on_held_out(all_feature_data: dict[str, dict[str, Any]],
                      baseline_features: list[str],
                      vocal_features: list[str],
                      y_global: np.ndarray,
                      groups_global: np.ndarray,
                      held_out_session_ids: list[str],
                      history_frames: int,
                      gam_settings: dict[str, Any],
                      likelihood: str) -> dict[str, Any]:
    """
    Refits on the development sessions and scores the reserve once.

    The carve-once held-out sessions are excluded from every CV fold and from
    the entire selection search, so they are the only estimate untouched by
    feature nomination, screening or greedy acceptance. Carving them and then
    never scoring them would throw that away; this is the single last look.

    Both models are refit on the same development rows and evaluated on the same
    reserve, so their scores stay paired and their difference is the held-out
    counterpart of the cross-validated increment. It is one fit per model, not a
    fold loop, and the increment it yields has no null attached -- it is a
    corroborating number, not a second test.

    Parameters
    ----------
    all_feature_data : dict
        ``feature -> {'X', 'y', 'groups'}`` as written by the extraction pipeline.
    baseline_features : list of str
        The forward-selected baseline.
    vocal_features : list of str
        The block under test.
    y_global : np.ndarray
        Shared 1-D target vector.
    groups_global : np.ndarray
        Shared 1-D session identifier per row.
    held_out_session_ids : list of str
        Sessions reserved at extraction time.
    history_frames : int
        Number of lags per feature.
    gam_settings : dict
        The ``pygam`` settings block.
    likelihood : str
        Which likelihood arm to fit.

    Returns
    -------
    held_out : dict
        ``baseline_score`` / ``full_score`` on the reserve, their ``margin``, the
        row and session counts behind them, and the per-model diagnostics
        (predictions, descriptive metrics, filter shapes). Every field is
        ``None`` when no sessions were reserved.
    """

    if not held_out_session_ids:
        return {
            'baseline_score': None, 'full_score': None, 'margin': None,
            'n_held_out_rows': 0, 'n_held_out_sessions': 0,
            'baseline': None, 'full': None,
        }

    held_out = set(held_out_session_ids)
    held_mask = np.array([g in held_out for g in groups_global], dtype=bool)
    development_index = np.flatnonzero(~held_mask)
    held_out_index = np.flatnonzero(held_mask)

    if held_out_index.size == 0 or development_index.size == 0:
        msg = (
            f"Held-out scoring needs both development and reserve rows; got "
            f"{development_index.size} and {held_out_index.size}."
        )
        raise ValueError(msg)

    # One "fold" whose train side is every development row and whose test side is
    # the reserve, so the same fitter serves both the CV loop and the last look.
    single_fold = [(development_index, held_out_index)]
    gam_kwargs = {
        'lam': gam_settings['lam_penalty'],
        'max_iter': gam_settings['max_iterations'],
        'tol': gam_settings['tol_val'],
    }
    full_features = [*baseline_features, *vocal_features]

    baseline_fit = fit_block_across_folds(
        feature_arrays=[all_feature_data[f]['X'] for f in baseline_features],
        y_global=y_global, cv_folds=single_fold, history_frames=history_frames,
        n_splines_value=gam_settings['n_splines_value'],
        n_splines_time=gam_settings['n_splines_time'],
        gam_kwargs=gam_kwargs, likelihood=likelihood,
        feature_names=baseline_features, collect_diagnostics=True,
    )
    full_fit = fit_block_across_folds(
        feature_arrays=[all_feature_data[f]['X'] for f in full_features],
        y_global=y_global, cv_folds=single_fold, history_frames=history_frames,
        n_splines_value=gam_settings['n_splines_value'],
        n_splines_time=gam_settings['n_splines_time'],
        gam_kwargs=gam_kwargs, likelihood=likelihood,
        feature_names=full_features, collect_diagnostics=True,
    )

    baseline_score = float(baseline_fit['d2'][0])
    full_score = float(full_fit['d2'][0])
    return {
        'baseline_score': baseline_score,
        'full_score': full_score,
        'margin': full_score - baseline_score,
        'n_held_out_rows': int(held_out_index.size),
        # Sessions actually PRESENT in the data, not the reserved id list: a
        # reserved session that contributed no rows would otherwise inflate the
        # count and overstate what the last look was measured on.
        'n_held_out_sessions': int(np.unique(groups_global[held_mask]).size),
        'baseline': baseline_fit,
        'full': full_fit,
    }


def screen_from_univariate(univariate_results_path: str | Path,
                           candidate_features: list[str]) -> dict[str, Any]:
    """
    Screens candidates from the consolidated univariate artifact.

    The per-feature fits happen on the cluster, one feature per job-array task,
    and are merged by ``consolidate_univariate_results``; this reads that merged
    pickle rather than refitting. Each feature arrives with an ``actual`` and a
    ``null`` branch scored on the same folds, so the screen is the same paired
    1SE test used everywhere else in this module: keep a feature when its mean
    per-fold improvement over its own null exceeds one standard error of that
    improvement.

    Screening on the response is what makes the later pruning safe -- a variable
    unassociated with the response cannot confound it, since confounding needs
    association with both. The bar is deliberately lenient, because a feature
    kept out of the baseline is not controlled for, it is merely absent, and the
    vocal block would inherit whatever it would have explained.

    Parameters
    ----------
    univariate_results_path : str or pathlib.Path
        Consolidated univariate pickle: ``feature -> {'actual': {...}, 'null': {...}}``
        plus the reserved metadata blocks.
    candidate_features : list of str
        Features eligible for the baseline. The vocal block is excluded by the
        caller, since it is the thing under test rather than a candidate.

    Returns
    -------
    screen_results : dict
        ``passed`` (ranked by improvement, best first), ``per_feature`` (the
        improvement, its SE, both mean scores and the verdict) and ``skipped``
        (always empty on return, since any skip aborts).

    Raises
    ------
    ValueError
        If any candidate is absent from the artifact or arrives without the
        branches the screen needs. Such a candidate never enters the pool the
        forward selection searches, so continuing would silently answer a
        narrower question; the usual cause is a job array sized below
        ``len(candidate_features) - 1``. Ruled fatal on 2026-09-03. Excluding a
        feature on purpose is done by leaving it out of ``candidate_features``,
        which records the exclusion instead of hiding it.
    """

    with Path(univariate_results_path).open('rb') as univariate_file:
        univariate_data = pickle.load(univariate_file)
    for reserved in ('_run_metadata', '_input_metadata', '_univariate_metadata',
                     '_consolidation_metadata'):
        univariate_data.pop(reserved, None)

    per_feature: dict[str, Any] = {}
    skipped: list[str] = []
    skip_reasons: dict[str, str] = {}
    for feature_name in candidate_features:
        if feature_name not in univariate_data:
            skipped.append(feature_name)
            skip_reasons[feature_name] = 'absent from the artifact (never swept)'
            continue

        payload = univariate_data[feature_name]
        # Consolidated entries are sometimes a `(meta, results)` tuple.
        results = payload[1] if isinstance(payload, tuple) and len(payload) == 2 else payload
        if 'actual' not in results or 'null' not in results:
            skipped.append(feature_name)
            skip_reasons[feature_name] = "present but missing an 'actual' or 'null' branch"
            continue
        if 'explained_deviance' not in results['actual'] or 'explained_deviance' not in results['null']:
            skipped.append(feature_name)
            skip_reasons[feature_name] = "present but missing 'explained_deviance'"
            continue

        actual_folds = np.asarray(results['actual']['explained_deviance'], dtype=float)
        null_folds = np.asarray(results['null']['explained_deviance'], dtype=float)
        improvement, improvement_se = paired_one_se_improvement(
            actual_folds, null_folds, higher_is_better=True,
        )
        passes = bool(np.isfinite(improvement) and improvement > improvement_se)

        per_feature[feature_name] = {
            'paired_improvement': float(improvement),
            'paired_improvement_se': float(improvement_se),
            'mean_d2_actual': float(np.nanmean(actual_folds)),
            'mean_d2_null': float(np.nanmean(null_folds)),
            'passed': passes,
        }
        print(format_selection_step(
            'Screen', feature=feature_name,
            metrics={'D2': per_feature[feature_name]['mean_d2_actual'],
                     'D2_null': per_feature[feature_name]['mean_d2_null'],
                     'improvement': improvement, 'se': improvement_se},
            decision='PASS' if passes else 'DROP',
        ))

    if skipped:
        # Fatal by ruling (2026-09-03). Every skip shrinks the candidate pool the
        # forward selection searches, so a run that proceeds here answers a
        # different question than the one asked -- and does so while looking
        # entirely normal in the logs. The usual cause is a `--array` upper bound
        # below (number of features - 1), which leaves the surplus features
        # unswept and therefore absent from the consolidated artifact.
        print(format_selection_step(
            'Screen', decision='SKIPPED',
            detail=f"{len(skipped)} candidate(s) absent or incomplete in the univariate "
                   f"artifact: {sorted(skipped)}",
        ))
        reason_lines = '\n'.join(f"    {name}: {skip_reasons[name]}" for name in sorted(skipped))
        error_message = (
            f"Screen aborted: {len(skipped)} of {len(candidate_features)} candidate features "
            f"are absent or incomplete in the consolidated univariate artifact "
            f"{Path(univariate_results_path).name}, so the candidate pool would be truncated:\n"
            f"{reason_lines}\n"
            f"  If the cause is a short job array, resubmit with --array=0-{len(candidate_features) - 1} "
            f"and re-run the consolidator. To proceed deliberately without a feature, drop it from "
            f"`candidate_features` so the exclusion is explicit and recorded."
        )
        raise ValueError(error_message)

    passed = sorted(
        (f for f in per_feature if per_feature[f]['passed']),
        key=lambda f: per_feature[f]['paired_improvement'],
        reverse=True,
    )
    return {'passed': passed, 'per_feature': per_feature, 'skipped': skipped}


def _restore_last_step(output_directory: Path, step_prefix: str) -> dict[str, Any] | None:
    """
    Reads back the highest-numbered step checkpoint, if any.

    Selection runs on the cluster and can be interrupted, so each accepted step
    is persisted and a restart continues from the last one rather than refitting
    everything. A checkpoint whose payload is unusable is treated as absent, so a
    stale or truncated file makes the run start fresh instead of resuming into a
    corrupt state.

    Parameters
    ----------
    output_directory : pathlib.Path
        Directory holding the per-step pickles.
    step_prefix : str
        Filename prefix identifying this run's steps.

    Returns
    -------
    last_step : dict or None
        The restored payload, or ``None`` when there is nothing usable to resume.
    """

    if not output_directory.is_dir():
        return None

    step_numbers = []
    for candidate_path in output_directory.iterdir():
        name = candidate_path.name
        if name.startswith(step_prefix) and name.endswith('.pkl'):
            try:
                step_numbers.append(int(name[len(step_prefix):-len('.pkl')]))
            except ValueError:
                continue
    if not step_numbers:
        return None

    last_path = output_directory / f"{step_prefix}{max(step_numbers)}.pkl"
    try:
        with last_path.open('rb') as step_file:
            payload = pickle.load(step_file)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None

    if not isinstance(payload, dict) or 'current_features' not in payload:
        return None
    if 'vocal_block_features' in payload:
        # The vocal step is TERMINAL, not a forward step. Its payload carries
        # `current_features` and a `selected_feature`, so resuming from it would
        # silently seed the baseline with the block under test. The run is
        # complete; there is nothing to continue.
        return None
    return payload


def forward_select_features(all_feature_data: dict[str, dict[str, Any]],
                            screened_features: list[str],
                            y_global: np.ndarray,
                            cv_folds: list[tuple[np.ndarray, np.ndarray]],
                            history_frames: int,
                            gam_settings: dict[str, Any],
                            output_directory: Path,
                            step_prefix: str,
                            wrap_step: Callable[[dict[str, Any]], dict[str, Any]],
                            likelihood: str = 'gamma',
                            use_top_rank_as_anchor: bool = False) -> dict[str, Any]:
    """
    Greedily grows the baseline model, persisting one checkpoint per step.

    At each step every remaining candidate is added to the current model and
    scored on the same folds, and the winner is accepted when its per-fold paired
    improvement over the incumbent exceeds one standard error OF THAT IMPROVEMENT
    -- the rule ``model_selection`` uses, via the shared
    :func:`paired_one_se_improvement`. Pairing matters because the incumbent and
    the candidate are nested and scored on identical folds, so the fold-difficulty
    term they share cancels in the difference but dominates either score alone;
    an unpaired standard error sets the bar far too high and truncates the search.

    The bar is deliberately lenient. This loop builds the CONTROL, and a feature
    kept out of the baseline is not controlled for -- it is simply absent, and
    the vocal block added afterwards would inherit whatever it would have
    explained.

    Every accepted step is written to its own pickle before the next begins, so a
    run killed by a wall-clock limit resumes from the last accepted feature
    instead of restarting.

    Parameters
    ----------
    all_feature_data : dict
        ``feature -> {'X', 'y', 'groups'}`` as written by the extraction pipeline.
    screened_features : list of str
        Candidates that survived the screen, best first.
    y_global : np.ndarray
        Shared 1-D target vector.
    cv_folds : list of tuple
        Folds shared by every fit so all comparisons stay paired.
    history_frames : int
        Number of lags per feature.
    gam_settings : dict
        The ``pygam`` settings block.
    output_directory : pathlib.Path
        Where the per-step pickles are written.
    step_prefix : str
        Filename prefix for this run's steps; must be unique per likelihood arm.
    wrap_step : callable
        Injects the provenance blocks into a step payload before it is written.
    likelihood : str, optional
        Which likelihood arm to fit; see :func:`fit_block_across_folds`.
    use_top_rank_as_anchor : bool, optional
        When ``True``, the highest-ranked screened feature is taken as step 0
        without evaluating the alternatives, matching the other selectors'
        ``--anchor``. It is fitted once and accepted unconditionally, which skips
        roughly ``n_candidates`` fits on the most expensive step. The trade is
        that the starting feature comes from the SCREEN's univariate ranking
        rather than from a multivariate comparison at step 0, so a feature that
        only looks best on its own can anchor the search.

    Returns
    -------
    selection : dict
        ``selected`` (in acceptance order), ``steps`` (the payload of each step)
        and ``final_scores`` (per-fold scores of the accepted model, reused as the
        increment baseline so it is not refit).
    """

    gam_kwargs = {
        'lam': gam_settings['lam_penalty'],
        'max_iter': gam_settings['max_iterations'],
        'tol': gam_settings['tol_val'],
    }
    output_directory.mkdir(parents=True, exist_ok=True)

    selected: list[str] = []
    current_scores: np.ndarray | None = None
    steps: list[dict[str, Any]] = []

    resumed = _restore_last_step(output_directory, step_prefix)
    if resumed is not None:
        # Any checkpoint above the one we resume from belongs to a previous run.
        # Leaving them in place lets `consolidate_model_selection_results` merge a
        # mixture of two runs into one "selection path" without complaint.
        for stale_path in output_directory.glob(f'{step_prefix}*.pkl'):
            try:
                stale_index = int(stale_path.name[len(step_prefix):-len('.pkl')])
            except ValueError:
                continue
            if stale_index > resumed['step_index']:
                print(format_selection_step(
                    'Resume', decision='DISCARD',
                    detail=f'stale checkpoint from an earlier run: {stale_path.name}',
                ))
                stale_path.unlink()

    converged_on_resume = False
    if resumed is not None:
        if resumed['selected_feature'] is not None:
            selected = [*resumed['current_features'], resumed['selected_feature']]
            current_scores = np.asarray(resumed['selected_feature_folds'], dtype=float)
            steps.append(resumed)
            print(format_selection_step(
                'Resume', detail=f"restored {len(selected)} accepted feature(s): {selected}",
            ))
        else:
            # The last checkpoint is a REJECTED step, which means the search had
            # already converged. Its `current_features` IS the converged set and
            # its `baseline_folds` are that model's per-fold scores, so the run is
            # finished. Treating this as "nothing to restore" would silently throw
            # away every accepted feature and repeat the whole search.
            selected = list(resumed['current_features'])
            current_scores = np.asarray(resumed['baseline_folds'], dtype=float)
            steps.append(resumed)
            converged_on_resume = True
            print(format_selection_step(
                'Resume', decision='CONVERGED',
                detail=f"last checkpoint is a rejected step; {len(selected)} accepted "
                       f"feature(s) restored, forward search complete: {selected}",
            ))

    remaining = [] if converged_on_resume else [f for f in screened_features if f not in set(selected)]

    # Anchor: accept the top-ranked screened feature outright as step 0. Skipped
    # entirely on resume, where step 0 is already on disk.
    if use_top_rank_as_anchor and resumed is None and remaining:
        anchor_feature = remaining[0]
        anchor_fit = fit_block_across_folds(
            feature_arrays=[all_feature_data[anchor_feature]['X']],
            y_global=y_global, cv_folds=cv_folds, history_frames=history_frames,
            n_splines_value=gam_settings['n_splines_value'],
            n_splines_time=gam_settings['n_splines_time'],
            gam_kwargs=gam_kwargs,
            likelihood=likelihood,
        )
        anchor_improvement, anchor_improvement_se = paired_one_se_improvement(
            anchor_fit['d2'], np.zeros(len(cv_folds)), higher_is_better=True,
        )
        print(format_selection_step(
            'Anchor', feature=anchor_feature,
            metrics={'D2': float(np.nanmean(anchor_fit['d2'])),
                     'improvement': anchor_improvement, 'se': anchor_improvement_se},
            decision='ACCEPT', detail='top-ranked; forced without testing alternatives',
        ))

        anchor_payload = {
            'step_index': 0,
            'current_features': [],
            'baseline_folds': np.zeros(len(cv_folds)),
            'baseline_score': 0.0,
            'candidates': {anchor_feature: {
                'folds': anchor_fit['d2'],
                'mean_d2': float(np.nanmean(anchor_fit['d2'])),
                'paired_improvement': float(anchor_improvement),
                'paired_improvement_se': float(anchor_improvement_se)}},
            'selected_feature': anchor_feature,
            'selected_feature_folds': anchor_fit['d2'],
            'forced_anchor': True,
        }
        steps.append(anchor_payload)
        anchor_path = output_directory / f"{step_prefix}0.pkl"
        with atomic_output_path(anchor_path) as tmp_path, tmp_path.open('wb') as step_file:
            pickle.dump(wrap_step(anchor_payload), step_file)

        selected.append(anchor_feature)
        current_scores = anchor_fit['d2']
        remaining.remove(anchor_feature)

    while remaining:
        # 0-based: `consolidate_model_selection_results` checks the merged indices
        # against `range(len(steps))` and warns when they do not match.
        step_index = len(selected)
        print(format_selection_step(f"Step {step_index:02d}", detail=f"{len(remaining)} candidates remaining"))

        reference = current_scores if current_scores is not None else np.zeros(len(cv_folds))
        candidate_scores: dict[str, np.ndarray] = {}
        candidate_improvements: dict[str, tuple[float, float]] = {}
        for candidate in remaining:
            trial_features = [*selected, candidate]
            trial = fit_block_across_folds(
                feature_arrays=[all_feature_data[f]['X'] for f in trial_features],
                y_global=y_global, cv_folds=cv_folds, history_frames=history_frames,
                n_splines_value=gam_settings['n_splines_value'],
                n_splines_time=gam_settings['n_splines_time'],
                gam_kwargs=gam_kwargs,
                likelihood=likelihood,
            )
            candidate_scores[candidate] = trial['d2']
            candidate_improvements[candidate] = paired_one_se_improvement(
                trial['d2'], reference, higher_is_better=True,
            )
            print(format_selection_step(
                f"Step {step_index:02d}", feature=candidate,
                metrics={'D2': float(np.nanmean(trial['d2'])),
                         'improvement': candidate_improvements[candidate][0],
                         'se': candidate_improvements[candidate][1]},
            ))

        # `max` seeds on remaining[0] and only replaces when the comparison is
        # True; every comparison against NaN is False, so a candidate whose folds
        # all failed would stay "best", fail acceptance, and STOP the search with
        # every genuinely improving candidate behind it silently discarded.
        scorable = [f for f in remaining if np.isfinite(candidate_improvements[f][0])]
        if not scorable:
            print(format_selection_step(
                f"Step {step_index:02d}", decision='STOP',
                detail=f"no candidate produced a finite improvement "
                       f"({len(remaining)} candidate(s) all failed to fit)",
            ))
            break
        if len(scorable) < len(remaining):
            print(format_selection_step(
                f"Step {step_index:02d}", decision='WARN',
                detail=f"{len(remaining) - len(scorable)} candidate(s) scored NaN and "
                       f"were excluded from this step: "
                       f"{sorted(set(remaining) - set(scorable))}",
            ))
        best_candidate = max(scorable, key=lambda f: candidate_improvements[f][0])
        best_improvement, best_improvement_se = candidate_improvements[best_candidate]
        accepted = bool(np.isfinite(best_improvement) and best_improvement > best_improvement_se)

        step_payload = {
            'step_index': step_index,
            'current_features': list(selected),
            'baseline_folds': np.asarray(reference, dtype=float),
            'baseline_score': float(np.nanmean(reference)),
            'candidates': {
                f: {'folds': candidate_scores[f],
                    'mean_d2': float(np.nanmean(candidate_scores[f])),
                    'paired_improvement': float(candidate_improvements[f][0]),
                    'paired_improvement_se': float(candidate_improvements[f][1])}
                for f in remaining
            },
            'selected_feature': best_candidate if accepted else None,
            'selected_feature_folds': candidate_scores[best_candidate] if accepted else None,
        }
        steps.append(step_payload)
        print(format_selection_step(
            f"Step {step_index:02d}", feature=best_candidate,
            metrics={'improvement': best_improvement, 'se': best_improvement_se},
            decision='ACCEPT' if accepted else 'STOP',
        ))

        # The rejected step is persisted too: it records WHY the search stopped,
        # which is otherwise lost, and lets a resume see convergence rather than
        # re-testing the same losing candidates.
        step_path = output_directory / f"{step_prefix}{step_index}.pkl"
        with atomic_output_path(step_path) as tmp_path, tmp_path.open('wb') as step_file:
            pickle.dump(wrap_step(step_payload), step_file)

        if not accepted:
            break

        selected.append(best_candidate)
        current_scores = candidate_scores[best_candidate]
        remaining.remove(best_candidate)

    if current_scores is None or not selected:
        msg = (
            "Forward selection accepted no feature, so there is no baseline model to test the "
            "vocal block against. A vocal increment measured against an empty baseline would be "
            "uninterpretable; investigate the screen before proceeding."
        )
        raise RuntimeError(msg)

    return {'selected': selected, 'steps': steps, 'final_scores': current_scores}


def vocal_block_final_step(all_feature_data: dict[str, dict[str, Any]],
                           baseline_features: list[str],
                           vocal_features: list[str],
                           baseline_scores: np.ndarray,
                           y_global: np.ndarray,
                           groups_global: np.ndarray,
                           cv_folds: list[tuple[np.ndarray, np.ndarray]],
                           held_out_session_ids: list[str],
                           history_frames: int,
                           gam_settings: dict[str, Any],
                           step_index: int,
                           output_directory: Path,
                           step_prefix: str,
                           wrap_step: Callable[[dict[str, Any]], dict[str, Any]],
                           likelihood: str = 'gamma') -> dict[str, Any]:
    """
    Adds the vocal block as the final selection step and scores what it buys.

    Structurally this is one more forward step: the block is appended to the
    converged baseline, both models are scored on the same folds, and acceptance
    is the same paired 1SE rule every earlier step used. Holding the vocal block
    to a different standard than the features it is being compared against would
    make the comparison incoherent, so it is not given one.

    No null is computed here. Held-out cross-validation already establishes that
    a block improving prediction on sessions it never saw carries real
    information; a shift null answers the narrower question of whether that
    information is in the *timing* of the calls rather than the shape of the
    trace, which is a follow-up on an effect you already have rather than a
    criterion for deciding whether you have one. Run
    :func:`circular_shift_rows_within_session` with
    :func:`paired_fold_margin` afterwards, and only if this step is accepted.

    Parameters
    ----------
    all_feature_data : dict
        ``feature -> {'X', 'y', 'groups'}`` as written by the extraction pipeline.
    baseline_features : list of str
        The converged baseline.
    vocal_features : list of str
        The block under test.
    baseline_scores : np.ndarray
        Per-fold scores of the baseline, reused rather than refit for the margin.
    y_global : np.ndarray
        Shared 1-D target vector.
    groups_global : np.ndarray
        Shared 1-D session identifier per row.
    cv_folds : list of tuple
        The same folds the baseline was scored on.
    held_out_session_ids : list of str
        The carve-once reserve, scored separately as the honest last look.
    history_frames : int
        Number of lags per feature.
    gam_settings : dict
        The ``pygam`` settings block.
    step_index : int
        Step number this becomes in the checkpoint sequence.
    output_directory : pathlib.Path
        Where the step pickle is written.
    step_prefix : str
        Filename prefix for this run's steps.
    wrap_step : callable
        Injects the provenance blocks into the payload before it is written.
    likelihood : str, optional
        Which likelihood arm to fit; see :func:`fit_block_across_folds`.

    Returns
    -------
    final_step : dict
        The accept/reject verdict, the paired improvement and its SE, per-fold
        baseline and full scores, the share of remaining deviance the block
        explains, the per-session breakdown, the held-out reserve scores, and the
        full model's per-fold predictions, descriptive metrics and filter shapes.
    """

    # The vocal block must never have been a selection candidate: if it reached
    # the baseline, the increment would be measured against a model that already
    # contains the thing under test, and the step would silently report nothing.
    # This can only happen if a caller passed the full feature list to the screen,
    # so it fails loudly rather than returning a meaningless zero.
    leaked = sorted(set(baseline_features) & set(vocal_features))
    if leaked:
        msg = (
            f"Vocal feature(s) {leaked} are already in the selected baseline. The vocal "
            f"block is the quantity under test and must be withheld from the screen and "
            f"from forward selection; check that `baseline_block_features` was used as "
            f"the candidate list."
        )
        raise ValueError(msg)

    gam_kwargs = {
        'lam': gam_settings['lam_penalty'],
        'max_iter': gam_settings['max_iterations'],
        'tol': gam_settings['tol_val'],
    }
    full_features = [*baseline_features, *vocal_features]

    # The baseline is refit once WITH diagnostics: its stored scores came from
    # forward selection without predictions, and a per-session margin needs both
    # models' predictions on the same rows.
    baseline_refit = fit_block_across_folds(
        feature_arrays=[all_feature_data[f]['X'] for f in baseline_features],
        y_global=y_global, cv_folds=cv_folds, history_frames=history_frames,
        n_splines_value=gam_settings['n_splines_value'],
        n_splines_time=gam_settings['n_splines_time'],
        gam_kwargs=gam_kwargs, likelihood=likelihood,
        feature_names=baseline_features, collect_diagnostics=True,
    )
    full = fit_block_across_folds(
        feature_arrays=[all_feature_data[f]['X'] for f in full_features],
        y_global=y_global, cv_folds=cv_folds, history_frames=history_frames,
        n_splines_value=gam_settings['n_splines_value'],
        n_splines_time=gam_settings['n_splines_time'],
        gam_kwargs=gam_kwargs, likelihood=likelihood,
        feature_names=full_features, collect_diagnostics=True,
    )

    improvement, improvement_se = paired_one_se_improvement(
        full['d2'], baseline_scores, higher_is_better=True,
    )
    accepted = bool(np.isfinite(improvement) and improvement > improvement_se)

    remaining_fraction = fraction_of_remaining_deviance(
        baseline_scores=baseline_scores, full_scores=full['d2'],
    )
    session_breakdown = per_session_scores(
        baseline_diagnostics=baseline_refit, full_diagnostics=full,
        groups_global=groups_global, likelihood=likelihood,
    )
    session_margins = np.array([s['margin'] for s in session_breakdown.values()], dtype=float)
    held_out = score_on_held_out(
        all_feature_data=all_feature_data, baseline_features=baseline_features,
        vocal_features=vocal_features, y_global=y_global, groups_global=groups_global,
        held_out_session_ids=held_out_session_ids, history_frames=history_frames,
        gam_settings=gam_settings, likelihood=likelihood,
    )

    print(format_selection_step(
        f"Step {step_index:02d}", feature='+vocal block',
        metrics={'D2_base': float(np.nanmean(baseline_scores)),
                 'D2_full': float(np.nanmean(full['d2'])),
                 'improvement': improvement, 'se': improvement_se,
                 'frac_remaining': remaining_fraction['mean']},
        decision='ACCEPT' if accepted else 'REJECT',
        detail=f"sessions+ {int(np.sum(session_margins > 0.0))}/{session_margins.size}"
               + (f", held-out margin {held_out['margin']:+.4f}"
                  if held_out['margin'] is not None else ''),
    ))

    payload = {
        'step_index': step_index,
        'current_features': list(baseline_features),
        'baseline_folds': np.asarray(baseline_scores, dtype=float),
        'baseline_score': float(np.nanmean(baseline_scores)),
        'vocal_block_features': list(vocal_features),
        'accepted': accepted,
        'paired_improvement': float(improvement),
        'paired_improvement_se': float(improvement_se),
        'full_scores': full['d2'],
        'baseline_refit_scores': baseline_refit['d2'],
        'fraction_of_remaining': remaining_fraction,
        'per_session': session_breakdown,
        'per_session_summary': {
            'n_sessions': int(session_margins.size),
            'sessions_positive': int(np.sum(session_margins > 0.0)),
            'median_margin': float(np.median(session_margins)) if session_margins.size else float('nan'),
            'min_margin': float(np.min(session_margins)) if session_margins.size else float('nan'),
            'max_margin': float(np.max(session_margins)) if session_margins.size else float('nan'),
        },
        'held_out': held_out,
        'full_fold_diagnostics': {
            key: full[key] for key in
            ('y_true', 'y_pred', 'test_indices', 'filter_shapes', 'spearman_r',
             'pearson_r', 'mae', 'rmse', 'residual_deviance', 'score_scale')
        },
        'selected_feature': vocal_features[0] if accepted else None,
        'selected_feature_folds': full['d2'] if accepted else None,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    step_path = output_directory / f"{step_prefix}{step_index}.pkl"
    with atomic_output_path(step_path) as tmp_path, tmp_path.open('wb') as step_file:
        pickle.dump(wrap_step(payload), step_file)
    return payload


def behavioral_response_model_selection(input_pickle_path: str | Path,
                                        univariate_results_path: str | Path,
                                        output_directory: str | Path,
                                        *,
                                        settings_path: str | Path | None = None,
                                        modeling_settings_dict: dict[str, Any] | None = None,
                                        use_top_rank_as_anchor: bool = False) -> dict[str, Any]:
    """
    Runs the screen, the forward selection and the vocal step, checkpointing each.

    Consumes two upstream artifacts -- the extraction pickle and the consolidated
    per-feature univariate pickle produced by the cluster job array -- and writes
    one pickle per selection step, which ``consolidate_model_selection_results``
    merges. The per-feature fits are NOT repeated here; the screen reads the
    scores the array already computed.

    Every acceptance in the run, including the vocal block's, uses the same
    paired 1SE rule. No null is computed: held-out cross-validation establishes
    that a block improving prediction on unseen sessions carries information, and
    the shift null answers the narrower timing question as a conditional
    follow-up (see :func:`vocal_block_final_step`).

    When ``likelihood`` is ``'both'`` the whole sequence runs once per arm, into
    its own checkpoint prefix, because a Gamma fit on native units and a
    lognormal fit are scored on different scales and only their increments are
    comparable.

    Parameters
    ----------
    input_pickle_path : str or pathlib.Path
        Modeling-input pickle from ``BehavioralResponsePipeline``.
    univariate_results_path : str or pathlib.Path
        Consolidated per-feature univariate pickle feeding the screen.
    output_directory : str or pathlib.Path
        Directory the per-step pickles are written to.
    settings_path : str or pathlib.Path, optional
        Keyword-only, as is ``modeling_settings_dict``: the two are easy to
        transpose positionally and a settings dict landing in the path slot fails
        with an unhelpful ``TypeError`` deep inside ``pathlib``.
        Settings JSON to load. This is what the SLURM dispatcher passes, matching
        the other selectors; ignored when ``modeling_settings_dict`` is supplied.
    modeling_settings_dict : dict, optional
        Configuration dictionary, for programmatic callers. When both this and
        ``settings_path`` are ``None`` the shipped
        ``_parameter_settings/modeling_settings.json`` is loaded.
    use_top_rank_as_anchor : bool, optional
        Forwarded to :func:`forward_select_features`; the dispatcher's
        ``--anchor`` flag, matching the other selectors.

    Returns
    -------
    results : dict
        One entry per likelihood arm, each with ``screen``, ``selection`` and
        ``final_step``.

    Notes
    -----
    With ``likelihood = 'both'`` this writes TWO checkpoint prefixes into
    ``output_directory``. ``consolidate_model_selection_results`` infers a single
    prefix and refuses to proceed when it finds more than one, so consolidate each
    arm separately by passing its ``prefix`` explicitly, or point each arm at its
    own directory.
    """

    if modeling_settings_dict is None:
        resolved_settings_path = (
            Path(settings_path) if settings_path is not None
            else Path(__file__).parent.parent / '_parameter_settings' / 'modeling_settings.json'
        )
        with resolved_settings_path.open('r') as settings_file:
            modeling_settings_dict = json.load(settings_file)

    response_settings = modeling_settings_dict['behavioral_response']
    validation_settings = modeling_settings_dict['model_validation']
    gam_settings = modeling_settings_dict['hyperparameters']['classical']['pygam']

    input_pickle_path = Path(input_pickle_path)
    output_directory = Path(output_directory)
    with input_pickle_path.open('rb') as input_file:
        artifact = pickle.load(input_file)
    all_feature_data, metadata_blocks = extract_metadata_blocks(artifact)
    input_metadata = metadata_blocks['_input_metadata']

    vocal_features = input_metadata['analysis_specific']['vocal_block_features']
    baseline_candidates = input_metadata['analysis_specific']['baseline_block_features']
    history_frames = int(input_metadata['filter_history_frames'])
    held_out_session_ids = list(input_metadata['held_out_session_ids'])

    reference_feature = next(iter(all_feature_data))
    y_global = all_feature_data[reference_feature]['y']
    groups_global = all_feature_data[reference_feature]['groups']

    requested = response_settings['likelihood']
    likelihood_arms = ['gamma', 'lognormal'] if requested == 'both' else [requested]

    print(format_run_header(
        task='behavioral-response selection',
        engine='pygam',
        feature=f"slot {response_settings['response_mouse_index']} '{response_settings['response_feature']}'",
        split_strategy=validation_settings['split_strategy'],
        n_splits=validation_settings['n_cv_folds'],
        input_files={'input data': str(input_pickle_path),
                     'univariate results': str(univariate_results_path)},
        output_directory=str(output_directory),
    ))

    cv_folds = build_session_folds(
        y_global=y_global, groups_global=groups_global,
        held_out_session_ids=held_out_session_ids,
        n_splits=validation_settings['n_cv_folds'],
        test_proportion=validation_settings['cv_validation_proportion'],
        random_seed=validation_settings['random_seed'],
    )

    # The screen reads scores the univariate array already computed, so it does
    # not depend on the likelihood arm -- computing it once avoids re-parsing the
    # artifact per arm and makes explicit that BOTH arms screen on the same
    # candidate set. Note those scores come from whichever engine produced the
    # univariate run (Gamma by default), so a `lognormal` arm is screened on
    # Gamma-derived rankings; acceptable because the screen's bar is deliberately
    # lenient, but recorded in `extra_knobs` rather than left implicit.
    screen = screen_from_univariate(
        univariate_results_path=univariate_results_path,
        candidate_features=baseline_candidates,
    )

    results: dict[str, Any] = {}
    for likelihood in likelihood_arms:
        print(format_selection_step('Arm', detail=f"likelihood = {likelihood}"))

        run_metadata = build_selection_metadata(
            modeling_settings=modeling_settings_dict,
            selection_function='behavioral_response_model_selection',
            selection_metric='explained_deviance',
            n_splits_selection=len(cv_folds),
            test_proportion=validation_settings['cv_validation_proportion'],
            split_strategy=validation_settings['split_strategy'],
            random_seed=validation_settings['random_seed'],
            one_se_rule_used=True,
            aic_termination_used=False,
            n_anchor_features=1 if use_top_rank_as_anchor else 0,
            anchor_feature=(screen['passed'][0]
                            if use_top_rank_as_anchor and screen['passed'] else ''),
            gam_kwargs={'lam': gam_settings['lam_penalty'],
                        'max_iter': gam_settings['max_iterations'],
                        'tol': gam_settings['tol_val']},
            extra_knobs={
                'likelihood': likelihood,
                'vocal_block_features': vocal_features,
                'held_out_session_ids': held_out_session_ids,
                'screen_passed': screen['passed'],
                'screen_skipped': screen['skipped'],
                'screen_scores_from_engine': 'univariate artifact (arm-independent)',
            },
        )

        def wrap_step(payload: dict[str, Any],
                      _run_md: dict[str, Any] = run_metadata) -> dict[str, Any]:
            """
            Injects the provenance blocks so each step file stands alone.

            Parameters
            ----------
            payload : dict[str, Any]
                The step's own results, copied before the blocks are added.
            _run_md : dict[str, Any]
                Run-provenance block; bound as a default argument so each step
                captures the metadata as it stood when the step was written.

            Returns
            -------
            step_payload : dict[str, Any]
                ``payload`` plus the run and input provenance blocks.
            """
            return inject_metadata(dict(payload), _run_metadata=_run_md,
                                   _input_metadata=input_metadata)

        # `consolidate_model_selection_results` infers the prefix by requiring the
        # `model_selection_..._step_` shape, so the name must follow that convention
        # or the merge finds nothing. Cohort, feature and arm are all in the stem so
        # two arms never collide in one output directory.
        cohort_condition = derive_experimental_condition(modeling_settings_dict)
        step_prefix = (
            f"model_selection_behavioral_response_"
            f"{response_settings['response_feature']}_{likelihood}_"
            f"{cohort_condition}_{validation_settings['split_strategy']}_step_"
        )
        selection = forward_select_features(
            all_feature_data=all_feature_data, screened_features=screen['passed'],
            y_global=y_global, cv_folds=cv_folds, history_frames=history_frames,
            gam_settings=gam_settings, output_directory=output_directory,
            step_prefix=step_prefix, wrap_step=wrap_step, likelihood=likelihood,
            use_top_rank_as_anchor=use_top_rank_as_anchor,
        )
        final_step = vocal_block_final_step(
            all_feature_data=all_feature_data, baseline_features=selection['selected'],
            vocal_features=vocal_features, baseline_scores=selection['final_scores'],
            y_global=y_global, groups_global=groups_global, cv_folds=cv_folds,
            held_out_session_ids=held_out_session_ids, history_frames=history_frames,
            # NOT len(steps): on resume `steps` is seeded with only the restored
            # checkpoint, so its length no longer tracks how many step files exist
            # and the vocal step would overwrite an accepted one -- recording a
            # vocal feature as a baseline feature in the merged selection path.
            gam_settings=gam_settings,
            step_index=selection['steps'][-1]['step_index'] + 1,
            output_directory=output_directory, step_prefix=step_prefix,
            wrap_step=wrap_step, likelihood=likelihood,
        )
        results[likelihood] = {'screen': screen, 'selection': selection, 'final_step': final_step}

        print(format_run_summary(
            label=f"behavioral response [{likelihood}], {len(selection['selected'])} baseline feature(s)",
            metrics_by_strategy={
                'BASELINE': {'D2': float(np.nanmean(final_step['baseline_folds']))},
                'WITH VOCAL': {'D2': float(np.nanmean(final_step['full_scores'])),
                               'improvement': final_step['paired_improvement']},
            },
            out_path=str(output_directory / f"{step_prefix}*.pkl"),
        ))

    return results
