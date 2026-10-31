"""
@author: bartulem
Forward stepwise model selection for predicting bout and USV dynamics.
"""

from __future__ import annotations

import contextlib
import gc
import json
import pickle
import re
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pygam import GAM, LogisticGAM, te
from scipy.stats import spearmanr
from sklearn.linear_model import GammaRegressor, LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_gamma_deviance,
    mean_squared_log_error,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)

from ..os_utils import atomic_output_path
from .jax_multinomial_logistic_regression import SmoothMultinomialLogisticRegression
from .load_input_files import load_pickle_modeling_data, onset_target_tag
from .manifold_metric import (
    _fit_von_mises_kappa,
    circular_mean,
    dcor_prediction_truth,
    inverse_region_frequency_weights,
    macro_von_mises_logscore,
    manifold_prediction_metrics,
    resolve_manifold_metric,
    resolve_manifold_selection_score_key,
    signed_diff,
    warn_if_no_region_labels,
)
from .manifold_torus_regression import resolve_manifold_regressor_cls
from .modeling_bases_functions import (
    _normalizecols,
    basis_coefficients_to_frames,
    bsplines,
    identity,
    laplacian_pyramid,
    project_history_onto_basis,
    raised_cosine,
    resolve_temporal_basis,
)
from .modeling_metadata import (
    RESERVED_METADATA_KEYS,
    build_selection_metadata,
    inject_metadata,
)
from .modeling_torus_geodesics import (
    build_torus_geodesic_context,
    geodesic_mae_columns,
    make_qlvm_decode_fn_from_source,
    resolve_geodesic_decoder_source,
    resolve_manifold_column_names,
)
from .modeling_usv_manifold_position import (
    _log_spaced_grid,
    _tune_manifold_regularization,
    get_stratified_spatial_splits_stable,
)
from .modeling_utils import (
    align_probs_to_canonical,
    bounded_test_proportion,
    brier_score_multi,
    classification_inner_cv_splits,
    cut_history_windows,
    development_heldout_masks,
    expected_calibration_error,
    format_run_header,
    format_run_summary,
    format_selection_step,
    held_out_session_ids_from_metadata,
    macro_log_likelihood,
    mean_absolute_error_1d,
    null_target_permutation,
    pearson_r_safe,
    pool_session_arrays,
    pool_worker_count,
    read_feature_trace,
    read_window_starts,
    root_mean_squared_error,
    safe_confusion_matrix,
    safe_matthews_corrcoef,
    session_grouped_inner_cv,
    verify_windows_against_traces,
)
from .modeling_vocal_categories_multinomial import (
    _balance_multinomial_train_indices,
    _log_spaced_grid_multinomial,
    _tune_multinomial_regularization,
    get_stratified_group_splits_stable,
)
from .modeling_vocal_onsets import VocalOnsetModelingPipeline
from .step_shift_test import (
    STEP_TEST_RULE,
    SelectionPool,
    pool_context,
    run_step_shift_test,
    set_pool_context,
)

# Human-readable names of the manifold Step-0 strategies (console lines and the run log).
_STEP0_GATE_LABELS = {
    'null_model_intercept': 'intercept-only (training-fold circular mean)',
    'null_model_free': 'empirical-density draw',
}


def _harvest_upstream_metadata(univariate_data: dict,
                               input_data: dict) -> tuple:
    """
    Pulls the upstream metadata blocks out of the loaded Level-2
    consolidated univariate file and the Level-1 modeling input pickle.

    The function pops the reserved keys (`_input_metadata`,
    `_run_metadata`, `_consolidation_metadata`) from each dict so the
    downstream selection loop can iterate `*.items()` over feature data
    only. It returns the harvested `_input_metadata` (preferring the
    Level-2 value if both files supply one — the consolidator already
    asserted Level-2's matched every per-feature file) and the Level-2
    `_run_metadata` (re-tagged as `_univariate_metadata` for embedding
    into the selection step file).

    Parameters
    ----------
    univariate_data : dict
        The dict returned by `pickle.load` on the consolidated
        univariate artifact.
    input_data : dict
        The dict returned by `pickle.load` on the Level-1 modeling
        input pickle.

    Returns
    -------
    tuple
        `(input_metadata, univariate_metadata)`. Either may be `None`
        if the upstream file was a legacy artifact without metadata.
    """

    in_md = None
    if '_input_metadata' in univariate_data:
        in_md = univariate_data.pop('_input_metadata')
    if '_input_metadata' in input_data:
        # Prefer the Level-2 copy (already consolidator-asserted) but
        # fall back to the Level-1 copy if Level-2 was a legacy run.
        if in_md is None:
            in_md = input_data.pop('_input_metadata')
        else:
            input_data.pop('_input_metadata', None)

    univ_md = None
    if '_run_metadata' in univariate_data:
        univ_md = univariate_data.pop('_run_metadata')

    # Drop other reserved keys from both dicts so the iteration is clean.
    for k in ('_consolidation_metadata',):
        univariate_data.pop(k, None)
        input_data.pop(k, None)

    return in_md, univ_md


def _require_analysis_type(loaded: dict, expected: str, source: str) -> None:
    """
    Description
    -----------
    Refuses an upstream artifact that was built for a different analysis than
    the selection reading it. Every modeling-input pickle records the pipeline
    that built it in ``_input_metadata.analysis_type`` (``'onset'``,
    ``'category'``, ``'params'``, ``'multinomial'`` or ``'continuous'``, the
    vocabulary of the dispatchers' ``--analysis_type``), and the consolidated
    univariate file carries that block verbatim. A selection fed the other
    pipeline's files (e.g. a bout-offset input, built by the onset pipeline,
    sent to the bout-parameter selection) would otherwise fail deep inside the
    fit on a missing key, or run on data of the wrong layout; this check stops
    it at load time with both names. It runs before the reserved blocks are
    popped, once on the univariate file and once on the input pickle, so a
    mismatch in either is caught. A legacy artifact without an
    ``_input_metadata`` block, or with one that predates the
    ``analysis_type`` field, cannot be checked and passes.

    Parameters
    ----------
    loaded (dict)
        The dict ``pickle.load`` returned for the univariate or input file,
        still holding its reserved metadata keys.
    expected (str)
        The analysis type of the calling selection.
    source (str)
        The file's path, named in the error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the recorded ``analysis_type`` differs from ``expected``.
    """

    if '_input_metadata' not in loaded or 'analysis_type' not in loaded['_input_metadata']:
        return
    recorded = loaded['_input_metadata']['analysis_type']
    if recorded != expected:
        msg = (f"{source} was built for analysis_type '{recorded}' "
               f"(analysis_tag '{loaded['_input_metadata']['analysis_tag']}'), but this is the "
               f"'{expected}' model selection. Rerun with --analysis_type {recorded}, or pass the "
               f"'{expected}' files.")
        raise ValueError(msg)


def _dump_step_pickle(path: str | Path, payload: Any) -> None:
    """
    Description
    -----------
    Pickles one selection artifact (a step file or the finalized last step) to ``path``
    atomically: the bytes go to a temporary sibling that ``os_utils.atomic_output_path``
    renames onto ``path`` only after the dump finished, so a crash, kill or full disk
    mid-write leaves the previous file intact instead of a truncated pickle that a
    resume or the consolidator would then fail to read (or, for the finalization
    rewrite of the last step, the loss of the very checkpoint a resume needs).

    Parameters
    ----------
    path (str | Path)
        Destination pickle path; its parent directory must exist.
    payload (Any)
        The object to pickle (normally a ``_wrap_step`` dict).

    Returns
    -------
    None
    """
    with atomic_output_path(path) as tmp_path, Path(tmp_path).open('wb') as handle:
        pickle.dump(payload, handle)


def _make_step_wrapper(input_md: dict, univariate_md: dict, run_md: dict):
    """
    Returns a closure `wrap(payload) -> dict` that injects the three
    metadata blocks into a step-pickle payload before serialization.

    Each selection function loads its upstream metadata once, builds
    its own `_run_metadata`, and then uses the closure to attach all
    three blocks at every `pickle.dump` site. The closure short-
    circuits cleanly when an upstream block is `None` (legacy
    artifact); the run-level block is always emitted.

    Parameters
    ----------
    input_md : dict or None
        `_input_metadata` block to embed.
    univariate_md : dict or None
        `_univariate_metadata` block to embed.
    run_md : dict
        `_run_metadata` block to embed (always present for new runs).

    Returns
    -------
    callable
        `wrap(payload)` returning a new dict with the metadata blocks
        injected. Never mutates the input payload.
    """

    def wrap(payload: dict) -> dict:
        """
        Description
        -----------
        Attaches the metadata blocks captured by `_make_step_wrapper` to one step-pickle
        payload: `_run_metadata` always, `_input_metadata` and
        `_univariate_metadata` only when the corresponding upstream block
        is not `None`.

        Parameters
        ----------
        payload : dict
            The step result to be pickled.

        Returns
        -------
        dict
            A new dict produced by `inject_metadata` holding `payload`'s
            entries plus the metadata blocks; `payload` itself is not mutated.
        """
        blocks = {'_run_metadata': run_md}
        if input_md is not None:
            blocks['_input_metadata'] = input_md
        if univariate_md is not None:
            blocks['_univariate_metadata'] = univariate_md
        return inject_metadata(payload, **blocks)

    return wrap


def _frozen_selection_kappa(y_dev: np.ndarray, w_dev: np.ndarray, *, metric: str, period: float) -> float:
    """
    Description
    -----------
    The torus selection's frozen von Mises concentration: fit ONCE on the
    development events from the residuals of an intercept-only (circular-mean)
    predictor -- the marginal dispersion of the USV positions -- and unweighted,
    matching the unweighted per-point von Mises scoring. Every torus score of a
    selection run (the univariate screen, the baseline, the candidates, the
    acceptance gate and the held-out refit) reuses it when
    ``vocal_features.freeze_selection_kappa`` is on, so a fold whose predictions
    are worse than the marginal scores BELOW the flat-torus density instead of
    being clamped to it (a self-refit kappa on that fold's own residuals would be
    0) and every score is one comparable, signed rule.

    Parameters
    ----------
    y_dev (np.ndarray)
        ``(n, 2)`` torus positions of the development events.
    w_dev (np.ndarray)
        ``(n,)`` event weights, used only to place the circular mean.
    metric (str)
        Manifold geometry (``'torus'``).
    period (float)
        Per-axis wrap period.

    Returns
    -------
    kappa (float)
        The frozen concentration.
    """
    y_dev = np.asarray(y_dev, dtype=np.float64)
    w_dev = np.asarray(w_dev, dtype=np.float64)
    w_cov = w_dev / (np.sum(w_dev) + 1e-12)
    mu_dev = circular_mean(y_dev, metric=metric, period=period, weights=w_cov)
    r_dev = signed_diff(y_dev, mu_dev[None, :], metric=metric, period=period) * (2.0 * np.pi / period)
    return _fit_von_mises_kappa(r_dev.ravel())


def _restore_selection_checkpoint(
    last_res: dict,
    score_key: str,
    se_key: str,
    folds_of: Callable[[dict], Any],
) -> dict | None:
    """
    Description
    -----------
    Rebuilds the state a forward model selection held right after the step a checkpoint
    records, from the decision written in it, never by re-deciding with another rule. A
    step file's ``selected_feature`` is what the live gate accepted (``'null_model_free'``
    for a model-free Step-0 baseline, the anchor for the anchor step, the step winner for a
    forward step) or None when the step rejected its winner and the search finished, which
    is a valid record of convergence (a step at which no candidate could be scored writes
    no file: the live run stops there with an error). The anchor file stores
    the features after acceptance and a forward-step file the features before it, so the
    selected feature is appended only when missing. The incumbent's per-fold scores become
    the reference the next gate compares against. A manifold Step-0 record keeps the
    ``'null_model_free'`` marker but gates against the strategy it names in
    ``baseline_strategy`` (the intercept-only predictor on the torus), so that strategy's
    block is the restored incumbent.

    Parameters
    ----------
    last_res (dict)
        The loaded step pickle (``step_idx``, ``current_features``, ``baseline_score``,
        ``selected_feature``, ``candidates_summary``; ``baseline_folds`` on a converged
        step; ``baseline_strategy`` on a manifold Step-0 record).
    score_key (str)
        The candidate entry's fold-mean score key (e.g. ``'mean_auc'``).
    se_key (str)
        The candidate entry's standard-error key (e.g. ``'se_auc'``).
    folds_of (Callable[[dict], Any])
        Returns a candidate entry's per-fold reference (the folds bundle or the per-fold
        score list the selector's gate consumes).

    Returns
    -------
    state (dict | None)
        None when the step accepted a feature that has no scored folds (a broken
        checkpoint: the caller starts fresh). Otherwise ``features`` (list),
        ``score`` (float), ``se`` (float), ``folds`` (the incumbent's per-fold
        reference; on a converged step the step's ``baseline_folds`` when stored, else
        None), ``next_step`` (int: the step to run next, or the converged step itself)
        and ``converged`` (bool).
    """

    candidates = last_res['candidates_summary']
    selected = last_res['selected_feature']
    features = list(last_res['current_features'])
    step_idx = int(last_res['step_idx'])
    if selected is None:
        return {
            'features': features,
            'score': last_res['baseline_score'],
            'se': 0.0,
            'folds': last_res['baseline_folds'],
            'next_step': step_idx,
            'converged': True,
        }
    if selected not in candidates or not isinstance(candidates[selected], dict) \
            or score_key not in candidates[selected] or not np.isfinite(candidates[selected][score_key]):
        return None
    entry = candidates[selected]
    if selected == 'null_model_free' and 'baseline_strategy' in last_res:
        # Manifold Step-0 record: the marker stays 'null_model_free', the incumbent is
        # the gate baseline it names.
        entry = candidates[last_res['baseline_strategy']]
    if selected != 'null_model_free' and selected not in features:
        features.append(selected)
    return {
        'features': features,
        'score': entry[score_key],
        'se': entry[se_key],
        'folds': folds_of(entry),
        'next_step': step_idx + 1,
        'converged': False,
    }


def _quarantine_stale_step_files(model_selection_dir: Path, prefix: str) -> Path | None:
    """
    Description
    -----------
    Moves every step file a FRESH selection run would collide with out of the way
    before it writes its Step 0. A fresh start happens with step files on disk when
    the resume logic rejects the last checkpoint (broken, unreadable or on another
    score scale); the old files would otherwise survive next to the new run's, and
    the consolidator and the GLM-HMM stage, which read the highest step index as
    the final one, would pick up a stale step from the earlier run. The files are
    moved, never deleted, into ``stale_<timestamp>/`` inside the selection
    directory, and the move is logged.

    Parameters
    ----------
    model_selection_dir (pathlib.Path)
        The selection output directory.
    prefix (str)
        The run's step-file prefix; a step file is ``<prefix><int>.pkl``.

    Returns
    -------
    stale_dir (pathlib.Path | None)
        The folder the files were moved into, or None when there was nothing to move.
    """

    if not model_selection_dir.is_dir():
        return None
    stale_files = []
    for path in model_selection_dir.iterdir():
        name = path.name
        if not (path.is_file() and name.startswith(prefix) and name.endswith('.pkl')):
            continue
        stem = name[len(prefix):-len('.pkl')]
        # Step files are `<prefix><int>.pkl`; a step's interrupted shift-test null is
        # `<prefix><int>_null_partial.pkl` and belongs to the same run.
        if stem.endswith('_null_partial'):
            stem = stem[:-len('_null_partial')]
        if stem.isdigit():
            stale_files.append(path)
    if not stale_files:
        return None
    stale_dir = model_selection_dir / f"stale_{time.strftime('%Y%m%d_%H%M%S')}"
    stale_dir.mkdir(exist_ok=True)
    for path in stale_files:
        path.replace(stale_dir / path.name)
    print(f"[FRESH START] Moved {len(stale_files)} stale step file(s) '{prefix}*.pkl' from an earlier "
          f"run into {stale_dir} so they cannot be read as this run's steps.")
    return stale_dir


def _write_empty_selection(last_file: Path,
                           wrap_step: Callable[[dict], dict],
                           univariate_results_path: str,
                           input_data_path: str,
                           label: str) -> None:
    """
    Description
    -----------
    Finalizes a selection that accepted no feature (no anchor forced and the first
    forward step rejected every candidate). There is no model to refit, so the
    per-fold filter-shape refit and the held-out refit are skipped, and the last
    step file records the empty result explicitly: ``final_model_features = []``,
    ``filter_shapes = []`` and ``heldout = None``. A rerun resumes that converged
    checkpoint and lands here again, writing the same result instead of crashing.

    Parameters
    ----------
    last_file (pathlib.Path)
        The step file the run wrote last (the rejecting step).
    wrap_step (Callable[[dict], dict])
        The selector's metadata wrapper (re-injects the reserved metadata blocks).
    univariate_results_path (str)
        The univariate ranking the run screened, recorded for provenance.
    input_data_path (str)
        The modeling-input pickle, recorded for provenance.
    label (str)
        The selector's name in the log line (e.g. ``'onset'``).

    Returns
    -------
    None
    """

    with Path(last_file).open('rb') as f:
        data = pickle.load(f)
    for _k in RESERVED_METADATA_KEYS:
        data.pop(_k, None)
    data.update({
        'final_model_features': [],
        'filter_shapes': [],
        'heldout': None,
        'univariate_results_path': univariate_results_path,
        'input_data_path': input_data_path,
    })
    _dump_step_pickle(last_file, wrap_step(data))
    print(f"\n[EMPTY SELECTION] {label} model selection accepted no feature: no candidate passed the "
          f"per-step shift test. Recorded final_model_features=[], filter_shapes=[], "
          f"heldout=None in {last_file}; no final or held-out refit was run.")


def _sign_flip_p_value(margins: np.ndarray, *, random_state: int, exact_max_folds: int,
                       monte_carlo_draws: int) -> float:
    """
    Description
    -----------
    One-sided sign-flip permutation p-value that the mean of paired per-fold margins
    exceeds zero.

    Under the null that each fold's ``actual - baseline`` margin is symmetric about
    zero, every assignment of signs to the observed margins is equally likely. The
    statistic is the margin sum (equivalently the mean); the p-value is the fraction of
    sign patterns whose sum is at least the observed sum. With ``n`` folds there are
    ``2**n`` patterns, the identity among them, so writing ``k`` for the number of the
    ``B = 2**n - 1`` non-identity patterns with a sum at least the observed one gives
    the usual ``p = (1 + k) / (1 + B)``; its floor is ``1 / 2**n`` (``9.54e-07`` at 20
    folds), reached when no other pattern ties or beats the observed sum. Up to
    ``exact_max_folds`` folds all patterns are enumerated (the test is exact; 2**22
    sums are ~32 MB of float64, a fraction of a second); above it
    ``B = monte_carlo_draws`` random patterns are drawn with ``random_state`` and the
    same ``(1 + k) / (1 + B)`` form is returned.

    The pattern sums are built by doubling: starting from ``[0]``, each margin ``m``
    turns the current sums ``S`` into ``[S + m, S - m]``, so index 0 is the identity
    (all signs positive) and the whole set costs ``2**n`` additions. Sums that differ
    from the observed one only by floating-point reassociation (a margin of exactly
    zero, or a tie) count as ties, through a tolerance of ``1e-12`` times the sum of
    absolute margins.

    Parameters
    ----------
    margins (np.ndarray)
        ``(n,)`` finite per-fold paired margins (``n >= 1``).
    random_state (int)
        Seed for the Monte-Carlo branch (unused when the patterns are enumerated).
    exact_max_folds (int)
        Largest fold count whose sign patterns are enumerated exactly
        (``model_params.selection_sign_flip_exact_max_folds``).
    monte_carlo_draws (int)
        Number of random sign patterns drawn above ``exact_max_folds``
        (``model_params.selection_sign_flip_monte_carlo_draws``).

    Returns
    -------
    p_value (float)
        The one-sided sign-flip p-value, in ``[1 / 2**n, 1]`` when exact.
    """
    margins = np.asarray(margins, dtype=np.float64).ravel()
    n_folds = margins.size
    tolerance = 1e-12 * max(1.0, float(np.sum(np.abs(margins))))
    if n_folds <= exact_max_folds:
        pattern_sums = np.zeros(1, dtype=np.float64)
        for margin in margins:
            pattern_sums = np.concatenate((pattern_sums + margin, pattern_sums - margin))
        observed = pattern_sums[0]
        return float(np.count_nonzero(pattern_sums >= observed - tolerance)) / float(pattern_sums.size)
    observed = float(np.sum(margins))
    rng = np.random.default_rng(random_state)
    n_at_least = 0
    remaining = monte_carlo_draws
    chunk_size = 2 ** 14
    while remaining > 0:
        n_draw = min(chunk_size, remaining)
        signs = rng.integers(0, 2, size=(n_draw, n_folds)) * 2.0 - 1.0
        n_at_least += int(np.count_nonzero(signs @ margins >= observed - tolerance))
        remaining -= n_draw
    return (1.0 + n_at_least) / (1.0 + monte_carlo_draws)


def _fold_paired_margin_bootstrap(
        actual_folds: dict,
        null_folds: dict,
        *,
        metric: str,
        period: float,
        n_bootstrap: int,
        ci_level: float,
        random_state: int,
        event_to_region: np.ndarray = None,
        min_region_events: int,
        min_fold_events: int,
        sign_flip_exact_max_folds: int,
        sign_flip_monte_carlo_draws: int,
        dcor_n_sub: int,
        dcor_n_rep: int,
        kappa: float | None = None) -> dict:
    """
    Description
    -----------
    Fold-grain significance of a manifold predictor via a paired score margin.

    This is the operative statistic for the manifold selection gate. Each CV
    fold yields one predictive score for the ``actual`` fit and one for the
    target-permutation ``null`` fit, computed on that fold's genuine
    out-of-sample test predictions; the per-fold ``actual - null`` difference is
    the *within-fold* behaviour->position signal, with the structure both share
    cancelled by the pairing. **Folds -- not sessions, not individual
    vocalizations -- are the unit**. Significance is an exact one-sided
    sign-flip permutation test over the per-fold margins
    (:func:`_sign_flip_p_value`; floor ``1 / 2**n_folds``), and the reported
    interval is a percentile CI at ``ci_level`` from bootstrapping the folds.

    Why fold-grain. On the torus the per-fold score is the macro (per-region)
    von Mises log-likelihood (:func:`manifold_metric.macro_von_mises_logscore`),
    which equal-weights acoustic regions so that a feature rescuing a rare,
    badly-predicted region is rewarded. A macro average is only meaningful when
    every region is populated well enough for a stable per-region mean; pooling
    all sessions within a fold guarantees that, whereas scoring one session at a
    time starves most regions below any sensible per-region floor and collapses
    the macro signal. The concentration ``kappa`` is fit per fold on that fold's
    pooled angular residuals (each fold carries thousands of events, so the
    estimate is stable).

    On euclidean the per-fold score is the wrap-aware distance correlation
    (:func:`manifold_metric.dcor_prediction_truth`), unchanged. Distance
    correlation is non-negative and finite-sample biased, so an absolute dcor is
    uninterpretable; the paired margin is the meaningful quantity and it cancels
    the bias empirically (``actual`` and ``null`` share each fold's events, hence
    the same bias, which subtracts out). No analytical bias correction is needed.

    Parameters
    ----------
    actual_folds (dict)
        The ``actual`` block's ``folds`` dict; must carry per-fold
        ``test_indices``, ``y_pred_xy`` and ``y_true`` lists.
    null_folds (dict)
        The target-permuted ``null`` block's ``folds`` dict (same keys).
    metric (str)
        ``'torus'`` or ``'euclidean'`` -- the manifold geometry for the score.
    period (float)
        Per-axis wrap period (used on the torus; ignored on euclidean).
    n_bootstrap (int)
        Number of fold bootstrap resamples (for the reported CI only; the
        p-value is the sign-flip test).
    ci_level (float)
        Central confidence level for the reported percentile interval (e.g.
        ``0.99``).
    random_state (int)
        Seed for the deterministic fold bootstrap (and for the sign-flip
        test's Monte-Carlo branch, used only above
        ``sign_flip_exact_max_folds`` usable folds).
    event_to_region (np.ndarray)
        Per-event acoustic-region (QLVM category) label, aligned to the same
        full-event index space as each fold's ``test_indices``. Required for the
        torus macro von Mises score; pass ``None`` to score with the pooled
        (label-free) von Mises log-likelihood (a run whose selection score is the
        pooled ``vm_logscore_pooled``). Ignored on euclidean.
    min_region_events (int)
        Minimum events a region must contribute within a fold to enter that
        fold's torus macro average (``vocal_features.usv_manifold_min_region_events``).
        Ignored on euclidean and when ``event_to_region`` is ``None`` (pooled
        scoring has a single region).
    min_fold_events (int)
        Folds contributing fewer than this many test events are skipped (too few
        points for a stable per-fold score); ``model_params.selection_min_fold_events``.
    sign_flip_exact_max_folds (int)
        Largest usable-fold count whose sign patterns are enumerated exactly
        (``model_params.selection_sign_flip_exact_max_folds``).
    sign_flip_monte_carlo_draws (int)
        Random sign patterns drawn above that count
        (``model_params.selection_sign_flip_monte_carlo_draws``).
    dcor_n_sub (int)
        Subsample size per repeat of the euclidean per-fold distance correlation
        (``vocal_features.usv_manifold_dcor.n_subsample``); unused on the torus.
    dcor_n_rep (int)
        Subsample repeats averaged per fold by that score
        (``vocal_features.usv_manifold_dcor.n_repeats_fold_gate``); unused on the torus.
    kappa (float)
        Pre-fit global von Mises concentration to reuse for every fold's torus
        score (see the ``kappa`` note in the "Why fold-grain" paragraph). ``None``
        (default) preserves the historical per-fold self-refit. Passing one frozen
        concentration removes the per-fold-refit floor that can corrupt the margin
        on weak folds and keeps ``actual`` / ``null`` on one dispersion scale.
        Ignored on euclidean.

    Returns
    -------
    result (dict)
        Keys: ``mean_margin`` (mean per-fold margin), ``ci_low`` / ``ci_high``
        (bootstrap percentile interval at ``ci_level``), ``p_value`` (one-sided
        exact sign-flip permutation p-value that the mean margin exceeds 0,
        ``(1 + k) / (1 + B)`` over the ``B = 2**n_folds - 1`` non-identity sign
        patterns, so never 0), and ``n_folds``
        (number of folds with a usable margin). If fewer than two folds are
        usable, ``mean_margin`` is ``nan`` and ``p_value`` is ``1.0`` so the
        feature cannot pass.
    """
    # One score PER FOLD. On the TORUS it is the macro (per-region) von Mises
    # log-likelihood with a per-fold concentration `kappa` (each fold pools all
    # sessions, so every region is well populated and the estimate is stable); on
    # EUCLIDEAN it is the wrap-aware distance correlation. Both are reduced to one
    # number per fold; the paired actual-minus-null margin is tested by an exact
    # sign-flip permutation over folds and its CI is bootstrapped over folds.
    def _fold_scores(folds: dict) -> np.ndarray:
        """
        Description
        -----------
        Scores every fold of one strategy's per-fold predictions: the torus
        macro (per-region) von Mises log-likelihood
        (:func:`macro_von_mises_logscore`, with the enclosing ``kappa``,
        ``min_region_events`` and the regions of ``event_to_region`` at the
        fold's test indices, or pooled scoring when ``event_to_region`` is
        ``None``) or, on euclidean, the wrap-aware distance correlation
        (:func:`dcor_prediction_truth`, ``dcor_n_sub`` / ``dcor_n_rep``). A fold with fewer than
        ``min_fold_events`` test predictions scores NaN.

        Parameters
        ----------
        folds (dict)
            The strategy's ``folds`` block, holding the parallel per-fold lists
            ``test_indices`` (full-event indices), ``y_pred_xy`` (``(n, 2)``
            predictions) and ``y_true`` (``(n, 2)`` targets).

        Returns
        -------
        scores (np.ndarray)
            ``(n_folds,)`` float array of per-fold scores, NaN for skipped folds
            (and for a torus fold where no region clears ``min_region_events``).
        """
        scores = []
        for fold_test_indices, fold_y_pred_xy, fold_y_true in zip(
                folds['test_indices'], folds['y_pred_xy'], folds['y_true'], strict=False):
            test_indices = np.asarray(fold_test_indices)
            y_pred_xy = np.asarray(fold_y_pred_xy)
            y_true = np.asarray(fold_y_true)
            if y_pred_xy.shape[0] < min_fold_events:
                # Too few test points in this fold for a stable score; drop it so
                # it neither enters the margin nor destabilises the per-fold kappa.
                scores.append(float('nan'))
                continue
            if metric == 'torus':
                regions = (event_to_region[test_indices]
                           if event_to_region is not None else None)
                score = macro_von_mises_logscore(
                    y_pred_xy, y_true, regions,
                    metric=metric, period=period,
                    min_region_events=min_region_events, kappa=kappa,
                )
            else:
                score = dcor_prediction_truth(
                    y_pred_xy, y_true,
                    metric=metric, period=period,
                    # A single fold's events are well under the subsample cap, so
                    # no subsampling occurs and repeat averaging would only
                    # recompute the identical full-sample dcor -- the shipped
                    # fold-gate repeat count is 1 to avoid that waste.
                    n_sub=dcor_n_sub, n_rep=dcor_n_rep,
                )
            scores.append(score)
        return np.asarray(scores, dtype=float)

    actual_scores = _fold_scores(actual_folds)
    null_scores = _fold_scores(null_folds)
    n_paired = min(actual_scores.size, null_scores.size)
    margins = actual_scores[:n_paired] - null_scores[:n_paired]
    # A fold whose macro-vM has no region clearing the per-region floor returns
    # NaN; drop it so it neither enters the margin nor crashes the bootstrap
    # (dcor is always finite, so euclidean is unaffected by this guard).
    margins = margins[np.isfinite(margins)]

    if margins.size < 2:
        return {
            'mean_margin': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'p_value': 1.0,
            'n_folds': int(margins.size),
        }

    rng = np.random.default_rng(random_state)
    boot_means = np.array([
        margins[rng.integers(0, margins.size, margins.size)].mean()
        for _ in range(n_bootstrap)
    ])
    tail = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.percentile(boot_means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return {
        'mean_margin': float(margins.mean()),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        # One-sided exact sign-flip permutation p over the folds (floor 1 / 2**n).
        'p_value': _sign_flip_p_value(margins, random_state=random_state,
                                      exact_max_folds=sign_flip_exact_max_folds,
                                      monte_carlo_draws=sign_flip_monte_carlo_draws),
        'n_folds': int(margins.size),
    }


def get_unrolled_X_for_multivariate(feature_data_dict_list: list | None = None,
                                    history_frames: int | None = None) -> np.ndarray:
    """
    Prepares the 'unrolled' X matrix for a multivariate pyGAM by stacking
    multiple features side-by-side in the specific (Value, Time) format required
    for tensor product splines.

    For a single feature, pyGAM's `te(feature, time)` term expects an input with
    two columns: [Feature_Value, Time_Index].

    For a multivariate model with N features (e.g., `te(A, time) + te(B, time)`),
    we need to construct a matrix where each feature gets its own pair of columns.
    This function transforms a list of (n_samples, history_frames) matrices into
    a single tall, wide matrix suitable for fitting.

    Parameters
    ----------
    feature_data_dict_list : list of np.ndarray
        A list of numpy arrays, where each array contains the history data for
        one feature.
        - Each array must have shape (n_samples, history_frames).
        - The order of arrays in this list determines the column order in the output.
        - Example: [X_nose_nose, X_speed, X_yaw]
    history_frames : int
        The number of time lags (columns) in each input array. Used for validation
        and generating the time index column.

    Returns
    -------
    X_unrolled : np.ndarray
        A 2D numpy array of shape (n_samples * history_frames, 2 * n_features).

        The columns are organized as pairs for each feature:
        - Col 0: Feature 1 Value
        - Col 1: Feature 1 Time Index (0, 1, ... history_frames-1)
        - Col 2: Feature 2 Value
        - Col 3: Feature 2 Time Index
        - ... and so on.

        This format allows constructing a GAM model with terms like:
        `te(0, 1) + te(2, 3) + ...`
    """

    if not feature_data_dict_list:
        msg = "feature_data_dict_list is empty."
        raise ValueError(msg)

    n_samples, n_frames = feature_data_dict_list[0].shape
    if n_frames != history_frames:
        msg = f"Frame mismatch: Data has {n_frames}, expected {history_frames}"
        raise ValueError(msg)

    n_features = len(feature_data_dict_list)
    n_total_rows = n_samples * n_frames

    X_unrolled = np.empty((n_total_rows, 2 * n_features), dtype=np.float32)

    time_indices = np.arange(history_frames, dtype=np.float32)
    tiled_time = np.tile(time_indices, n_samples)

    for i, X_feat in enumerate(feature_data_dict_list):
        if X_feat.shape[0] != n_samples:
            msg = f"Sample count mismatch at feature index {i}. Expected {n_samples}, got {X_feat.shape[0]}"
            raise ValueError(msg)

        col_val_idx = i * 2
        col_time_idx = i * 2 + 1

        X_unrolled[:, col_val_idx] = X_feat.ravel()
        X_unrolled[:, col_time_idx] = tiled_time

    return X_unrolled


def compute_filter_shapes_per_fold_vocal_onset(
        *,
        cv_folds: list[dict],
        current_model_features: list[str],
        all_feature_data: dict,
        pooled_feature_cache: dict,
        history_frames: int,
        n_splines_value: int,
        n_splines_time: int,
        gam_kwargs: dict,
        random_seed: int,
        time_indices: np.ndarray,
) -> list[dict]:
    """
    Description
    -----------
    Refit the final multivariate ``LogisticGAM`` once per CV fold
    and extract the per-feature temporal-filter shape for that fold.

    The bout-onset selector's final-refit block is this helper's
    single caller; the loop is factored out here purely for
    readability, isolating the per-fold fitting, balancing, and
    partial-dependence-extraction logic from the selector body.

    For each fold:
      1. Pull this fold's train pos/neg arrays for every feature in
         ``current_model_features``. Session folds re-pool via
         ``pool_session_arrays``; mixed folds slice the cached
         ``X_full`` arrays by the fold's ``train_idx``.
      2. Balance the train fold 50/50 with a fold-specific RNG
         seed; one RNG draw is shared across features so the same
         pos / neg row indices are taken for every feature (design
         matrix stays column-aligned).
      3. Fit a multivariate LogisticGAM.
      4. Extract per-feature partial dependence via the
         ``predict_mu(test_grid) - predict_mu(base_grid)`` trick.
      5. Append the fold's per-feature dict to the returned list.

    Per-fold exceptions are caught and logged but do NOT abort the
    loop; a subset of bad folds shouldn't lose the others. The
    caller is responsible for checking the returned list length
    against ``len(cv_folds)`` and deciding whether to raise (see
    the bout-onset selector for the canonical ``RuntimeError`` on
    fully-empty result + warning on <2 folds).

    Parameters
    ----------
    cv_folds : list of dict
        Fold definitions. Each must have a ``'type'`` key of either
        ``'session'`` (with ``'train_sessions'``) or ``'mixed'``
        (with ``'train_idx'``, ``'n_pos_total'``, ``'n_neg_total'``).
    current_model_features : list of str
        Features to include in the multivariate model. Must all be
        present in ``all_feature_data`` (for session folds) and in
        ``pooled_feature_cache`` (for mixed folds).
    all_feature_data : dict
        Per-feature, per-session raw data. Consumed only for
        ``'session'`` folds.
    pooled_feature_cache : dict
        Pre-pooled balanced design matrices keyed by feature.
        Values are dicts with ``'X_full'``. Consumed only for
        ``'mixed'`` folds.
    history_frames : int
        Number of time frames per filter.
    n_splines_value, n_splines_time : int
        pyGAM spline counts for the value and time axes,
        respectively.
    gam_kwargs : dict
        Keyword args forwarded to ``LogisticGAM(**gam_kwargs)``.
    random_seed : int
        Base RNG seed. Each fold uses ``random_seed + fold_idx``
        so the balancing draws are reproducible across reruns.
    time_indices : np.ndarray
        Vector ``np.arange(history_frames, dtype=float)`` used to
        construct the partial-dependence base grid. Passed in to
        avoid re-allocating it per call.

    Returns
    -------
    list of dict
        One entry per successfully-fit fold. Each dict maps every
        feature in ``current_model_features`` to a 1-D numpy
        array of length ``history_frames``. Length may be less
        than ``len(cv_folds)`` if any folds raised; check the
        length at the call site.
    """

    final_fold_shapes = []
    for fold_idx, fold_info in enumerate(cv_folds):
        try:
            X_p_per_feat = []
            X_n_per_feat = []
            if fold_info['type'] == 'session':
                train_sess = fold_info['train_sessions']
                for feat in current_model_features:
                    _X_p, _X_n = pool_session_arrays(
                        all_feature_data[feat],
                        train_sess,
                        pos_key="usv_feature_arr",
                        neg_key="no_usv_feature_arr",
                        n_frames=history_frames,
                    )
                    X_p_per_feat.append(_X_p)
                    X_n_per_feat.append(_X_n)
            elif fold_info['type'] == 'mixed':
                train_ix = fold_info['train_idx']
                n_pos_total = fold_info['n_pos_total']
                n_neg_total = fold_info['n_neg_total']
                y_full = np.concatenate(
                    (np.ones(n_pos_total), np.zeros(n_neg_total))
                )
                y_tr_all = y_full[train_ix]
                for feat in current_model_features:
                    X_full_feat = pooled_feature_cache[feat]['X_full']
                    X_tr_all = X_full_feat[train_ix]
                    X_p_per_feat.append(X_tr_all[y_tr_all == 1])
                    X_n_per_feat.append(X_tr_all[y_tr_all == 0])
            else:
                print(
                    f"    [!] Unknown fold type {fold_info['type']!r} "
                    f"(fold {fold_idx}); skipping."
                )
                continue

            n_pos = X_p_per_feat[0].shape[0]
            n_neg = X_n_per_feat[0].shape[0]
            n_k = min(n_pos, n_neg)
            if n_k == 0:
                print(
                    f"    [!] Empty class in training fold {fold_idx} "
                    f"(pos={n_pos}, neg={n_neg}); skipping."
                )
                continue

            # One RNG per fold; same idx_p / idx_n applied across
            # every feature so rows line up column-wise.
            fold_rng = np.random.default_rng(random_seed + fold_idx)
            idx_p = fold_rng.choice(n_pos, n_k, replace=False)
            idx_n = fold_rng.choice(n_neg, n_k, replace=False)

            X_list_fold = [
                np.concatenate((X_p[idx_p], X_n[idx_n]))
                for X_p, X_n in zip(X_p_per_feat, X_n_per_feat, strict=False)
            ]
            y_fold = np.concatenate((np.ones(n_k), np.zeros(n_k)))

            X_gam_tr = get_unrolled_X_for_multivariate(
                X_list_fold, history_frames
            )
            y_gam_tr = np.repeat(y_fold.astype(float), history_frames)

            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(current_model_features)):
                gam_terms += te(
                    i * 2, i * 2 + 1,
                    n_splines=[n_splines_value, n_splines_time],
                )

            gam_fold = LogisticGAM(gam_terms, **gam_kwargs).fit(
                X_gam_tr, y_gam_tr
            )

            base_grid = np.zeros(
                (history_frames, 2 * len(current_model_features))
            )
            for k in range(len(current_model_features)):
                base_grid[:, k * 2 + 1] = time_indices
            base_prob = gam_fold.predict_mu(base_grid)

            fold_res = {}
            for k, f_name in enumerate(current_model_features):
                test_grid = base_grid.copy()
                test_grid[:, k * 2] = 1.0
                fold_res[f_name] = (
                    gam_fold.predict_mu(test_grid) - base_prob
                ).flatten()

            final_fold_shapes.append(fold_res)

            del gam_fold, X_gam_tr, y_gam_tr
            gc.collect()
        except Exception as e:
            # Per-fold failure is non-fatal; a subset of bad folds
            # shouldn't kill the visualisation. The caller's
            # post-loop assertion catches the "every fold failed"
            # case.
            print(
                f"    [!] Filter-shape fit failed for fold {fold_idx}: {e}"
            )
            continue
    return final_fold_shapes


#: Per-fold metric keys every onset / category candidate fit records, in the order the
#: step pickles store them.
_BINARY_FOLD_METRIC_KEYS = ('ll', 'auc', 'score', 'f1', 'recall', 'brier', 'ece', 'mcc',
                            'confusion_matrix', 'n_iter', 'converged', 'fit_time')

#: Start-frame slot -> pickle array key of the onset pipeline (the trace sidecar lists
#: the starts under the slot names; the windows live under the array keys).
_ONSET_WINDOW_SLOTS = {'usv': 'usv_feature_arr', 'no_usv': 'no_usv_feature_arr'}


def _step_test_settings(settings: dict, camera_fps: float) -> dict:
    """
    Description
    -----------
    Reads the per-step shift test's knobs from ``model_params`` (none of them is
    hard-coded anywhere) and converts the lag floor to frames at the cohort's camera
    rate.

    Parameters
    ----------
    settings (dict)
        The loaded modeling settings.
    camera_fps (float)
        ``io.camera_sampling_rate``; every session's stored frame rate must equal it
        (checked where the window starts are loaded).

    Returns
    -------
    cfg (dict)
        ``min_lag_seconds``, ``min_lag_frames``, ``checkpoint_every``, ``pool_workers``,
        ``pool_memory_gb``, ``fit_peak_gb_base``, ``fit_peak_gb_per_term``.
    """

    model_params = settings['model_params']
    min_lag_seconds = float(model_params['selection_step_null_min_lag_seconds'])
    return {
        'min_lag_seconds': min_lag_seconds,
        'min_lag_frames': round(min_lag_seconds * float(camera_fps)),
        'checkpoint_every': int(model_params['selection_step_null_checkpoint_every']),
        'pool_workers': int(model_params['selection_pool_workers']),
        'pool_memory_gb': float(model_params['selection_pool_memory_gb']),
        'fit_peak_gb_base': float(model_params['selection_fit_peak_gb_base']),
        'fit_peak_gb_per_term': float(model_params['selection_fit_peak_gb_per_term']),
    }


def _require_shift_test_inputs(input_md: dict | None, input_data_path: str, camera_fps: float) -> tuple[str, dict]:
    """
    Description
    -----------
    The trace sidecar the per-step shift test needs, named in the modeling pickle's
    ``_input_metadata``, and the per-session window start frames read back from it. A
    pickle written before the extraction stored them cannot be used under this rule:
    the run stops here with the instruction to rebuild it, instead of silently falling
    back to another test. Every session's recorded frame rate must equal the cohort
    rate the lag floor is converted at.

    Parameters
    ----------
    input_md (dict | None)
        The pickle's ``_input_metadata`` block.
    input_data_path (str)
        The pickle, named in the error.
    camera_fps (float)
        ``io.camera_sampling_rate``.

    Returns
    -------
    traces_path (str)
        The HDF5 sidecar with the processed per-session traces and the window starts.
    window_starts (dict)
        ``{session_id: {slot_name: starts, 'n_frames': T, 'fps': fps}}`` read from it.

    Raises
    ------
    ValueError
        If the path is missing from the metadata, the sidecar is not on disk or holds
        no window starts, or a session's frame rate differs from ``camera_fps`` by more
        than 0.5 frames per second.
    """

    if input_md is None or 'traces_path' not in input_md:
        msg = (f"{input_data_path} was extracted before the per-step shift test existed: its "
               f"_input_metadata has no 'traces_path'. Rebuild the modeling input pickle with the "
               f"current extraction (a one-minute step) and rerun the selection.")
        raise ValueError(msg)
    traces_path = str(input_md['traces_path'])
    if not Path(traces_path).is_file():
        msg = (f"The trace sidecar recorded in {input_data_path} is not on disk: {traces_path}. "
               f"Rebuild the modeling input pickle or restore the sidecar next to it.")
        raise ValueError(msg)
    window_starts = read_window_starts(traces_path)
    for session_id, entry in window_starts.items():
        # The stored rate is the session's recorded frame rate; it may differ from the
        # nominal cohort rate by a fraction of a frame per second, which moves a 20 s lag
        # floor by less than a frame. A rate off by more than half a frame per second is
        # another cohort or a corrupted recording.
        if abs(float(entry['fps']) - float(camera_fps)) > 0.5:
            msg = (f"Session {session_id!r} was extracted at {entry['fps']} fps but io.camera_sampling_rate "
                   f"is {camera_fps}; the shift test's lag floor is converted at one rate.")
            raise ValueError(msg)
    return traces_path, window_starts


def _require_scored_candidate(step_label: str, n_candidates: int, n_scored: int) -> None:
    """
    Description
    -----------
    Stops a selection at a step that cannot be decided: candidates were fitted but not
    one produced a finite cross-validated score (every fold of every candidate failed or
    diverged), so there is no winner to test. That is a failure of the fits (the per-fold
    error lines above say which), not a rejection by the data; recording it as "no feature
    improves prediction" and finishing as an empty selection would hide it. The forced
    anchor is one candidate.

    Parameters
    ----------
    step_label (str)
        ``'Anchor'`` or ``'Step <k>'``, named in the error.
    n_candidates (int)
        Candidates fitted at the step.
    n_scored (int)
        Candidates (or, for the anchor, folds) with a finite score.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        When candidates were fitted and none has a finite score.
    """

    if n_candidates > 0 and n_scored == 0:
        msg = (f"{step_label}: none of the {n_candidates} candidate(s) produced a valid fold score "
               f"(every fit failed or diverged; see the per-fold error lines above). The step cannot be "
               f"decided, so the selection stops here instead of recording a rejection.")
        raise RuntimeError(msg)


def _require_step_test_rule(last_results: dict, checkpoint_path: Path) -> None:
    """
    Description
    -----------
    Refuses to resume from a step file written under another acceptance rule: a run
    must be one rule from its first step to its last.

    Parameters
    ----------
    last_results (dict)
        The loaded checkpoint.
    checkpoint_path (Path)
        Its path, named in the error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the checkpoint carries no ``step_test`` tag or another rule.
    """

    run_md = last_results['_run_metadata'] if '_run_metadata' in last_results else {}
    knobs = run_md['extra_knobs'] if 'extra_knobs' in run_md else {}
    recorded = knobs['step_test']['rule'] if 'step_test' in knobs and 'rule' in knobs['step_test'] else None
    if recorded != STEP_TEST_RULE:
        msg = (f"{checkpoint_path} was written under step rule {recorded!r}, not {STEP_TEST_RULE!r}. "
               f"A selection run keeps one rule from start to end: move the old step files out of the "
               f"output directory (or point the run at an empty one) and start the selection again.")
        raise ValueError(msg)


def _assemble_fold_metrics(fold_results: list, feature: str, keys: tuple,
                           placeholders: dict | None = None) -> dict:
    """
    Description
    -----------
    Turns the per-fold worker results of one candidate into the per-fold metric lists
    the step pickles store, keeping every list aligned across folds: a failed fold is
    logged and contributes a placeholder, NaN by default (a NaN-filled ``(2, 2)``
    matrix for ``confusion_matrix``), exactly as the former inline loops did.

    Parameters
    ----------
    fold_results (list)
        One worker result per fold, in fold order (``{'ok': True, <key>: value}`` or
        ``{'ok': False, 'error': str}``).
    feature (str)
        The candidate, for the log line.
    keys (tuple)
        Metric keys to collect.
    placeholders (dict | None)
        ``{key: callable}`` returning the placeholder of a failed fold for keys whose
        placeholder is not NaN (the bout-parameter pipeline's empty ``y_true`` /
        ``y_pred`` arrays and ``converged = False``).

    Returns
    -------
    metrics (dict)
        ``{key: [per-fold values]}``.
    """

    metrics = {key: [] for key in keys}
    for fold_i, result in enumerate(fold_results):
        if result['ok']:
            for key in keys:
                metrics[key].append(result[key])
            continue
        print(f"    [!] Error fitting {feature} (Fold {fold_i}): {result['error']}")
        for key in keys:
            if placeholders is not None and key in placeholders:
                metrics[key].append(placeholders[key]())
            elif key == 'confusion_matrix':
                metrics[key].append(np.full((2, 2), np.nan))
            else:
                metrics[key].append(np.nan)
    return metrics


def _onset_candidate_fold(feat: str, fold_i: int) -> dict:
    """
    Description
    -----------
    One (candidate, fold) fit of the onset selector, run in a pool worker: fits the
    incumbent features plus ``feat`` on the fold's balanced training rows and scores
    the fold's test rows. The data comes from the ``'onset_step'`` pool-context entry
    the selector stores before creating the pool (``cv_folds``, ``session_fold_base``,
    ``pooled_feature_cache``, ``all_feature_data``, ``current_model_features``, the
    GAM knobs, ``history_frames``, ``random_seed``, ``decision_threshold``,
    ``ece_n_bins``). The operations are those of the former inline loop, so the
    fold scores are the same numbers.

    Parameters
    ----------
    feat (str)
        The candidate feature.
    fold_i (int)
        Fold index.

    Returns
    -------
    result (dict)
        ``{'ok': True, <metric key>: value, ...}`` with the keys of
        ``_BINARY_FOLD_METRIC_KEYS``, or ``{'ok': False, 'error': str}``.
    """

    ctx = pool_context('onset_step')
    fold_info = ctx['cv_folds'][fold_i]
    all_feature_data = ctx['all_feature_data']
    history_frames = ctx['history_frames']
    current_model_features = ctx['current_model_features']
    n_splines_value, n_splines_time = ctx['n_splines_value'], ctx['n_splines_time']
    gam_kwargs = ctx['gam_kwargs']
    trial_features = [*current_model_features, feat]
    gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
    for i in range(1, len(trial_features)):
        gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
    try:
        X_train_list, X_test_list = [], []
        if fold_info['type'] == 'session':
            train_sess, test_sess, idx_p, idx_n, y_tr_fold, y_te_fold, base_train_list, base_test_list = ctx['session_fold_base'][fold_i]
            X_train_list = list(base_train_list)
            X_test_list = list(base_test_list)
            f_p_tr, f_n_tr = pool_session_arrays(all_feature_data[feat], train_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            f_p_te, f_n_te = pool_session_arrays(all_feature_data[feat], test_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            X_train_list.append(np.concatenate((f_p_tr[idx_p], f_n_tr[idx_n])))
            X_test_list.append(np.concatenate((f_p_te, f_n_te)))
        elif fold_info['type'] == 'mixed':
            train_ix, test_ix = fold_info['train_idx'], fold_info['test_idx']
            n_pos_total = fold_info['n_pos_total']
            n_neg_total = fold_info['n_neg_total']
            y_full = np.concatenate((np.ones(n_pos_total), np.zeros(n_neg_total)))
            y_tr_all = y_full[train_ix]
            y_te_fold = y_full[test_ix]
            pos_positions = np.where(y_tr_all == 1)[0]
            neg_positions = np.where(y_tr_all == 0)[0]
            n_tr_keep = min(pos_positions.size, neg_positions.size)
            if n_tr_keep == 0:
                msg_0 = "Training fold has no samples in one class after split."
                raise ValueError(msg_0)
            fs_rng = np.random.default_rng(ctx['random_seed'] + fold_i)
            sel_pos = fs_rng.choice(pos_positions.size, n_tr_keep, replace=False)
            sel_neg = fs_rng.choice(neg_positions.size, n_tr_keep, replace=False)
            bal_train_local = np.concatenate((pos_positions[sel_pos], neg_positions[sel_neg]))
            y_tr_fold = np.concatenate((np.ones(n_tr_keep), np.zeros(n_tr_keep)))
            for f_name in trial_features:
                X_full = ctx['pooled_feature_cache'][f_name]['X_full']
                X_tr_all = X_full[train_ix]
                X_train_list.append(X_tr_all[bal_train_local])
                X_test_list.append(X_full[test_ix])
        else:
            msg = f"Unknown fold type {fold_info['type']!r}."
            raise ValueError(msg)

        X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
        X_te_gam = get_unrolled_X_for_multivariate(X_test_list, history_frames)
        fit_start = time.perf_counter()
        gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr_fold.astype(float), history_frames))
        fit_time = float(time.perf_counter() - fit_start)

        y_proba_tiled = gam.predict_proba(X_te_gam)
        y_proba_mean = np.mean(y_proba_tiled.reshape(len(y_te_fold), history_frames), axis=1)
        result = {'ok': True}
        result['ll'] = log_loss(y_te_fold.astype(int), np.clip(y_proba_mean, 1e-15, 1 - 1e-15))
        y_pred_mean = (y_proba_mean >= ctx['decision_threshold']).astype(int)
        result['score'] = balanced_accuracy_score(y_te_fold, y_pred_mean)
        result['f1'] = f1_score(y_te_fold, y_pred_mean, zero_division=0)
        result['recall'] = recall_score(y_te_fold, y_pred_mean, zero_division=0)
        result['auc'] = roc_auc_score(y_te_fold, y_proba_mean) if len(np.unique(y_te_fold)) > 1 else np.nan
        result['brier'] = float(brier_score_loss(y_te_fold.astype(int), y_proba_mean))
        try:
            y_proba_2d = np.column_stack([1.0 - y_proba_mean, y_proba_mean])
            result['ece'] = expected_calibration_error(y_te_fold.astype(int), y_pred_mean, y_proba_2d, n_bins=ctx['ece_n_bins'])
        except Exception:
            result['ece'] = np.nan
        result['mcc'] = safe_matthews_corrcoef(y_te_fold.astype(int), y_pred_mean)
        result['confusion_matrix'] = safe_confusion_matrix(y_te_fold.astype(int), y_pred_mean, labels=np.array([0, 1]))
        gam_diffs = gam.logs_['diffs']
        result['n_iter'] = float(len(gam_diffs))
        result['converged'] = bool(len(gam_diffs) > 0 and len(gam_diffs) < gam_kwargs['max_iter'])
        result['fit_time'] = fit_time
        del gam, X_tr_gam, X_te_gam
        gc.collect()
        return result
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


def vocal_onset_model_selection(univariate_results_path: str,
                               input_data_path: str,
                               output_directory: str,
                               settings_path: str | None = None,
                               use_top_rank_as_anchor: bool = False,
                               *,
                               p_val: float) -> None:
    """
    Performs Forward Stepwise Selection to identify the "minimal sufficient set"
    of behavioral features that predict vocal (bout) onset.

    This function implements a greedy search strategy using Generalized
    Additive Models (GAMs) with tensor product interactions. At every step the
    cross-validated winner among the remaining candidates is tested with the
    per-step shift test (``step_shift_test.run_step_shift_test``): the incumbent and
    the incumbent plus the winner are refit on the balanced development draw, the
    statistic is the in-sample Bernoulli log-likelihood gain per event, and its null
    comes from refits after circularly shifting the winner's trace within every
    development session and re-cutting its history windows at the stored start
    frames, ``floor(m / alpha)`` times with Bonferroni correction over the ``m``
    candidates (``alpha = model_params.selection_p_val``). The first step is tested
    the same way against the constant model. To ensure memory stability,
    filter shapes are NOT calculated during the selection loop. Instead, a single
    refit is performed on the final winning model to extract shapes for visualization,
    which are then appended to the results file of the last *successful* step.

    Splitting & balancing invariant:
    --------------------------------
    Across both 'session' and 'mixed' strategies, the training fold is down-sampled
    to a 50/50 class balance (Bout vs. No-Bout), while the test fold preserves the
    natural class prior of the source data. The 'mixed' strategy achieves this by
    running `StratifiedShuffleSplit` on the *unbalanced* pool and balancing the
    training half inside the per-fold loop. Reported metrics should therefore be
    imbalance-robust: the `score` field stores `balanced_accuracy`, paired with
    AUC and log-loss.

    Crucially, this version preserves full-resolution metric data. For every
    candidate tested at every step, the raw per-fold results are saved for:
    Log-Likelihood (LL), AUC, Recall, F1, Balanced Accuracy (`score`), Brier
    score, Expected Calibration Error (ECE), Matthews correlation coefficient
    (MCC), and the (2, 2) confusion matrix. Per-fold optimizer diagnostics —
    `n_iter`, `converged`, `fit_time` — are stored alongside so folds that
    terminated at `max_iter` without meeting the tolerance can be audited
    after the fact. Precision is intentionally not stored: it is recoverable
    from the saved confusion matrices and macro-F1 already summarizes the
    precision / recall trade-off. This allows plotting scripts to
    reconstruct model selection trajectories with accurate error bars and
    individual fold data points.

    Stability features:
    - Safe cleanup: Prevents `UnboundLocalError` during garbage collection if a
      fold fails early.
    - Crash reporting: Logs detailed error messages for failed fits.
    - Resource management: Explicit garbage collection triggers to prevent
      memory fragmentation.

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file.
    input_data_path : str
        Path to the raw extracted feature data .pkl file.
    output_directory : str
        Directory to save the step-wise state dictionaries.
    settings_path : str, optional
        Path to the JSON settings file.
    use_top_rank_as_anchor : bool, default False
        If True, initializes the search by forcing the highest-ranked univariate
        feature as the baseline model.
    p_val : float
        Keyword-only, no default: the dispatcher passes ``model_params.selection_p_val``.
        The alpha level used for significance screening against shuffled controls.

    Returns
    -------
    None
        Writes one pickle per selection step and the final consolidated
        selection pickle into ``output_directory``.
    """

    chance_ll = np.log(2)  # mean log-loss of a balanced 50/50 coin: the binary chance floor

    if settings_path is None:
        settings_path_obj = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with Path(settings_path).open() as f:
            settings = json.load(f)
    except FileNotFoundError as err:
        msg = f"Settings file not found at {settings_path}"
        raise FileNotFoundError(msg) from err

    model_selection_dir = Path(output_directory)
    model_selection_dir.mkdir(parents=True, exist_ok=True)

    with Path(univariate_results_path).open('rb') as f:
        univariate_data = pickle.load(f)
    all_feature_data = load_pickle_modeling_data(input_data_path)

    # Strip upstream metadata blocks before iterating either dict — the
    # reserved-key blocks (`_input_metadata` etc.) must not be mistaken
    # for feature entries. The harvested blocks are embedded into every
    # step pickle below via `_wrap_step`.
    _require_analysis_type(univariate_data, 'onset', univariate_results_path)
    _require_analysis_type(all_feature_data, 'onset', input_data_path)
    _input_md, _univariate_md = _harvest_upstream_metadata(univariate_data, all_feature_data)

    # Filter feature-level candidates *after* metadata extraction so the
    # iteration only sees behavioral-feature keys.
    candidates = []
    for feat_name, results in univariate_data.items():
        if 'actual' not in results or 'll' not in results['actual']:
            continue
        actual_ll = results['actual']['ll']
        null_ll = results['null']['ll']
        valid_actual = actual_ll[~np.isnan(actual_ll)]
        valid_null = null_ll[~np.isnan(null_ll)]
        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue
        mean_actual_ll = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=(p_val / len(univariate_data)) * 100)
        if mean_actual_ll < null_threshold:
            candidates.append({'feature': feat_name, 'mean_ll': mean_actual_ll})

    candidates.sort(key=lambda x: x['mean_ll'])
    ranked_features = [x['feature'] for x in candidates]
    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
    if not hasattr(pipeline, 'history_frames'):
        pipeline.history_frames = int(np.floor(settings['io']['camera_sampling_rate'] * settings['model_params']['filter_history']))
    history_frames = pipeline.history_frames

    camera_fps = float(settings['io']['camera_sampling_rate'])
    step_cfg = _step_test_settings(settings, camera_fps)
    traces_path, window_starts = _require_shift_test_inputs(_input_md, input_data_path, camera_fps)

    common_sessions = set(all_feature_data[ranked_features[0]].keys())
    for feat in ranked_features[1:]:
        common_sessions = common_sessions.intersection(set(all_feature_data[feat].keys()))
    all_sessions = sorted(common_sessions)

    # Held-out reserve carved ONCE at extraction time and shipped in the
    # univariate `_input_metadata.held_out_session_ids`. Restrict every selection
    # fold and the pooled-feature cache below to DEVELOPMENT sessions; the held-out
    # sessions are scored exactly once at finalization by refitting the accepted
    # model on all development data (the honest, selection-leakage-free number).
    # An empty list -- holdout disabled or a pre-v2 pickle -- makes
    # `dev_sessions == all_sessions` and `held_sessions` empty, an exact no-op.
    _held_out_session_ids = held_out_session_ids_from_metadata(_input_md if _input_md is not None else {})
    _held_set = set(_held_out_session_ids)
    dev_sessions = [s for s in all_sessions if s not in _held_set]
    held_sessions = [s for s in all_sessions if s in _held_set]
    if held_sessions:
        print(
            f"Held-out reserve: {len(held_sessions)} session(s) excluded from "
            f"selection; scored once at finalization."
        )

    # Lag-zero self-check of the shift test's inputs: every stored window of every
    # screened feature must re-cut exactly from the trace sidecar at its stored start.
    verify_windows_against_traces(
        all_feature_data, traces_path, window_starts, _ONSET_WINDOW_SLOTS, history_frames,
        features=ranked_features, sessions=dev_sessions,
    )
    print(f"Shift-test inputs verified: {len(ranked_features)} feature(s) x {len(dev_sessions)} development "
          f"session(s) re-cut exactly from {Path(traces_path).name}; lag floor "
          f"{step_cfg['min_lag_seconds']} s = {step_cfg['min_lag_frames']} frames.")

    pygam_params = settings['hyperparameters']['classical']['pygam']
    n_splines_time = pygam_params['n_splines_time']
    n_splines_value = pygam_params['n_splines_value']
    lam_penalty = pygam_params['lam_penalty']

    gam_kwargs = {
        'max_iter': int(pygam_params['max_iterations']),
        'tol': float(pygam_params['tol_val']),
        'lam': lam_penalty
    }

    split_strategy = settings['model_validation']['split_strategy']
    n_splits_selection = settings['model_validation']['n_cv_folds']
    test_prop = settings['model_validation']['cv_validation_proportion']

    random_seed = settings['model_validation']['random_seed']
    print(format_run_header(
        task='SELECTION:ONSET',
        engine=settings['model_params']['model_engine'],
        feature=Path(univariate_results_path).stem,
        input_files={'univariate results': univariate_results_path,
                     'input data': input_data_path,
                     'settings': settings_path},
        output_directory=str(model_selection_dir),
        split_strategy=split_strategy,
        n_splits=n_splits_selection,
    ))
    anchor_feature = ranked_features[0]

    # Build the selection-level `_run_metadata` block. Frozen here so
    # every step pickle written by this function carries an identical
    # block (the consolidator asserts equality across step files).
    _run_md = build_selection_metadata(
        modeling_settings=settings,
        selection_function='vocal_onset_model_selection',
        selection_metric='log_likelihood',
        n_splits_selection=int(n_splits_selection),
        test_proportion=float(test_prop),
        split_strategy=split_strategy,
        random_seed=int(random_seed),
        aic_termination_used=False,
        n_anchor_features=len(ranked_features),
        anchor_feature=anchor_feature,
        gam_kwargs=gam_kwargs,
        extra_knobs={
            'use_top_rank_as_anchor': bool(use_top_rank_as_anchor),
            'p_val': float(p_val),
            'n_splines_time': int(n_splines_time),
            'n_splines_value': int(n_splines_value),
            # Held-out provenance (also in the embedded `_input_metadata` sibling);
            # duplicated here so the selection run-metadata is self-describing.
            'held_out_test_proportion': float(settings['model_validation']['held_out_test_proportion']),
            'held_out_session_ids': list(_held_out_session_ids),
            'n_held_out_sessions': len(held_sessions),
            # The per-step acceptance rule and its knobs (a resume refuses another rule).
            'step_test': {
                'rule': STEP_TEST_RULE,
                'screened_features': list(ranked_features),
                'statistic': 'bernoulli_log_likelihood_per_event',
                'alpha': float(p_val),
                'min_lag_seconds': step_cfg['min_lag_seconds'],
                'min_lag_frames': step_cfg['min_lag_frames'],
                'checkpoint_every': step_cfg['checkpoint_every'],
                'pool_workers': step_cfg['pool_workers'],
                'refit_rows': 'balanced_development_draw',
                'traces_path': traces_path,
            },
        },
        settings_path=settings_path,
    )
    _wrap_step = _make_step_wrapper(_input_md, _univariate_md, _run_md)

    cv_folds = []
    if split_strategy == 'session':
        all_sessions_arr = np.array(dev_sessions)
        ss = ShuffleSplit(n_splits=n_splits_selection, test_size=bounded_test_proportion(test_prop, len(dev_sessions)), random_state=random_seed)
        for train_idx, test_idx in ss.split(all_sessions_arr):
            cv_folds.append({'train_sessions': all_sessions_arr[train_idx], 'test_sessions': all_sessions_arr[test_idx], 'type': 'session'})
    elif split_strategy == 'mixed':
        # Stratified split over the *unbalanced* pool: preserves the natural class
        # ratio in both train and test indices. Per-fold, training will be balanced
        # 50/50 inside the loop; test retains the natural class prior. Pool over
        # DEVELOPMENT sessions only so the mixed-fold event indices never reference
        # a held-out row (the pooled-feature cache below is keyed to the same set).
        X_p_all, X_n_all = pool_session_arrays(all_feature_data[anchor_feature], dev_sessions, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=pipeline.history_frames)
        n_pos_total = X_p_all.shape[0]
        n_neg_total = X_n_all.shape[0]
        y_full = np.concatenate((np.ones(n_pos_total), np.zeros(n_neg_total)))
        sss = StratifiedShuffleSplit(n_splits=n_splits_selection, test_size=test_prop, random_state=random_seed)
        for train_ix, test_ix in sss.split(np.zeros(len(y_full)), y_full):
            cv_folds.append({'train_idx': train_ix, 'test_idx': test_ix, 'n_pos_total': n_pos_total, 'n_neg_total': n_neg_total, 'type': 'mixed'})
    else:
        msg = (
            f"Unknown split_strategy '{split_strategy}' for bout-onset model selection. "
            f"Supported strategies: 'session', 'mixed'."
        )
        raise ValueError(msg)

    # Pre-pool each candidate feature's full (positive, negative) design matrix
    # once up front. The forward-selection loop re-evaluates every feature for
    # every fold and every accepted step; without this cache the inner body
    # would otherwise call `pool_session_arrays` O(n_steps * n_features * n_folds)
    # times, re-concatenating the same per-session arrays at every iteration.
    pooled_feature_cache = {}
    _anchor_n_pos, _anchor_n_neg = None, None
    for _feat_name in ranked_features:
        _X_p, _X_n = pool_session_arrays(
            all_feature_data[_feat_name],
            dev_sessions,
            pos_key="usv_feature_arr",
            neg_key="no_usv_feature_arr",
            n_frames=pipeline.history_frames,
        )
        if _anchor_n_pos is None:
            _anchor_n_pos, _anchor_n_neg = _X_p.shape[0], _X_n.shape[0]
        # The 'mixed' fold loop builds train/test indices against the
        # anchor's `(n_pos, n_neg)` and then applies them to every other
        # feature's `X_full`. This requires every feature's pooled
        # (positive, negative) sample counts to match the anchor's; if
        # they ever drift (e.g., feature-specific NaN drops upstream)
        # the indices silently land on wrong rows. Fail loudly here.
        # ``raise`` rather than ``assert`` so the check survives
        # under ``python -O``.
        elif _X_p.shape[0] != _anchor_n_pos or _X_n.shape[0] != _anchor_n_neg:
            msg = (
                f"pooled_feature_cache misalignment: feature {_feat_name!r} has "
                f"({_X_p.shape[0]}, {_X_n.shape[0]}) pos/neg samples but the "
                f"anchor has ({_anchor_n_pos}, {_anchor_n_neg}). The 'mixed' "
                "fold indices would silently misindex this feature."
            )
            raise ValueError(msg)
        pooled_feature_cache[_feat_name] = {
            'X_pos': _X_p,
            'X_neg': _X_n,
            'X_full': np.concatenate((_X_p, _X_n), axis=0),
        }

    # The development refit the per-step shift test scores on: all development
    # sessions pooled, the majority class subsampled to the minority count with the
    # finalization's seed, so the test's models are the models the held-out refit
    # fits. `pooled_feature_cache` is pooled over `dev_sessions` in that order, which is
    # what the held-out refit pools too.
    _dev_n_keep = min(_anchor_n_pos, _anchor_n_neg)
    _dev_rng = np.random.default_rng(random_seed + n_splits_selection + 1)
    _dev_idx_p = _dev_rng.choice(_anchor_n_pos, _dev_n_keep, replace=False)
    _dev_idx_n = _dev_rng.choice(_anchor_n_neg, _dev_n_keep, replace=False)
    _dev_y = np.concatenate((np.ones(_dev_n_keep), np.zeros(_dev_n_keep)))

    def _onset_development_score(features: list, winner_override: dict | None = None) -> float:
        """
        Description
        -----------
        In-sample Bernoulli log-likelihood per event (the negative log-loss of the
        frame-averaged probability) of the model with ``features`` refit on the
        balanced development draw. With ``winner_override`` the LAST feature's windows
        are taken from that per-session dict (a shifted null refit) instead of the
        stored windows. The empty model is the constant p = 0.5, log-likelihood
        ``-log 2``.

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            ``{session_id: {'usv_feature_arr': ..., 'no_usv_feature_arr': ...}}``.

        Returns
        -------
        score (float)
            Mean log-likelihood per event on the training rows.
        """

        if not features:
            return -float(chance_ll)
        X_train_list = []
        for position, f_name in enumerate(features):
            if winner_override is not None and position == len(features) - 1:
                X_p, X_n = pool_session_arrays(winner_override, dev_sessions, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            else:
                X_p, X_n = pooled_feature_cache[f_name]['X_pos'], pooled_feature_cache[f_name]['X_neg']
            X_train_list.append(np.concatenate((X_p[_dev_idx_p], X_n[_dev_idx_n])))
        gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
        for i in range(1, len(features)):
            gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
        X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
        gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, np.repeat(_dev_y.astype(float), history_frames))
        proba = np.mean(gam.predict_proba(X_tr_gam).reshape(len(_dev_y), history_frames), axis=1)
        return -float(log_loss(_dev_y.astype(int), np.clip(proba, 1e-15, 1 - 1e-15)))

    # Read-only data every pool worker of this run inherits (`_onset_candidate_fold`);
    # the step-specific entries are added per step before the pool forks.
    _onset_pool_base = {
        'cv_folds': cv_folds,
        'all_feature_data': all_feature_data,
        'pooled_feature_cache': pooled_feature_cache,
        'history_frames': history_frames,
        'n_splines_value': n_splines_value,
        'n_splines_time': n_splines_time,
        'gam_kwargs': gam_kwargs,
        'random_seed': random_seed,
        'decision_threshold': settings['diagnostics']['binary_decision_threshold'],
        'ece_n_bins': settings['diagnostics']['ece_n_bins'],
    }

    def _onset_session_fold_base(incumbent: list) -> dict:
        """
        Description
        -----------
        The candidate-invariant pieces of every 'session' fold for one step: the
        anchor pooling, the fold's balanced subsample indices (seeded by fold only, so
        identical for every candidate) and the incumbent features' pooled and
        subsampled design columns. ``pool_session_arrays`` is a pure per-feature
        concatenate, so ``pool(base + [new]) == pool(base) + pool([new])``.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.

        Returns
        -------
        base (dict)
            ``{fold_i: (train_sess, test_sess, idx_p, idx_n, y_tr, y_te, base_train_list,
            base_test_list)}`` for the 'session' folds.
        """

        base = {}
        for fold_i, fold_info in enumerate(cv_folds):
            if fold_info['type'] != 'session':
                continue
            train_sess, test_sess = fold_info['train_sessions'], fold_info['test_sessions']
            anc_data = all_feature_data[anchor_feature]
            X_p_tr, X_n_tr = pool_session_arrays(anc_data, train_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            trial_rng = np.random.default_rng(random_seed + fold_i)
            n_k = min(X_p_tr.shape[0], X_n_tr.shape[0])
            idx_p = trial_rng.choice(X_p_tr.shape[0], n_k, replace=False)
            idx_n = trial_rng.choice(X_n_tr.shape[0], n_k, replace=False)
            y_tr_fold = np.concatenate((np.ones(n_k), np.zeros(n_k)))
            X_p_te_anc, X_n_te_anc = pool_session_arrays(anc_data, test_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            y_te_fold = np.concatenate((np.ones(X_p_te_anc.shape[0]), np.zeros(X_n_te_anc.shape[0])))
            base_train_list, base_test_list = [], []
            for f_name in incumbent:
                f_p_tr, f_n_tr = pool_session_arrays(all_feature_data[f_name], train_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
                f_p_te, f_n_te = pool_session_arrays(all_feature_data[f_name], test_sess, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
                base_train_list.append(np.concatenate((f_p_tr[idx_p], f_n_tr[idx_n])))
                base_test_list.append(np.concatenate((f_p_te, f_n_te)))
            base[fold_i] = (train_sess, test_sess, idx_p, idx_n, y_tr_fold, y_te_fold, base_train_list, base_test_list)
        return base

    def _onset_fit_candidates(incumbent: list, candidate_features: list) -> dict:
        """
        Description
        -----------
        Fits every candidate on every fold on top of ``incumbent`` through the
        selection pool and returns each candidate's per-fold metric lists.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.
        candidate_features (list)
            Candidates to score.

        Returns
        -------
        per_candidate (dict)
            ``{feature: {metric key: [per-fold values]}}``.
        """

        set_pool_context(
            onset_step={**_onset_pool_base,
                        'current_model_features': list(incumbent),
                        'session_fold_base': _onset_session_fold_base(incumbent)},
            candidate_fold=_onset_candidate_fold,
        )
        n_workers = pool_worker_count(step_cfg['pool_workers'], len(incumbent) + 1,
                                      step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                      step_cfg['fit_peak_gb_per_term'])
        n_folds = len(cv_folds)
        tasks = [(feat, fold_i) for feat in candidate_features for fold_i in range(n_folds)]
        pool = SelectionPool(n_workers)
        try:
            results = pool.map('candidate_fold', tasks)
        finally:
            pool.close()
        return {feat: _assemble_fold_metrics(results[i * n_folds:(i + 1) * n_folds], feat, _BINARY_FOLD_METRIC_KEYS)
                for i, feat in enumerate(candidate_features)}

    def _onset_step_test(winner: str, step_index: int, m_candidates: int) -> dict:
        """
        Description
        -----------
        The per-step shift test of ``winner`` on top of the current incumbent.

        Parameters
        ----------
        winner (str)
            The cross-validated winner.
        step_index (int)
            The step.
        m_candidates (int)
            Candidates the winner was chosen from (the Bonferroni factor).

        Returns
        -------
        step_test (dict)
            The record of ``run_step_shift_test``.
        """

        incumbent = list(current_model_features)
        trial = [*incumbent, winner]
        incumbent_score = _onset_development_score(incumbent)
        observed_score = _onset_development_score(trial)
        winner_traces = {s: read_feature_trace(traces_path, winner, s)
                         for s in dev_sessions if s in all_feature_data[winner]}
        n_workers = pool_worker_count(step_cfg['pool_workers'], len(trial),
                                      step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                      step_cfg['fit_peak_gb_per_term'])
        return run_step_shift_test(
            n_workers=n_workers,
            refit_and_score=lambda override: _onset_development_score(trial, override),
            incumbent_features=incumbent,
            incumbent_score=incumbent_score,
            winner=winner,
            observed_score=observed_score,
            statistic_name='bernoulli_log_likelihood_per_event',
            winner_traces=winner_traces,
            window_starts={s: window_starts[s] for s in winner_traces},
            slots=_ONSET_WINDOW_SLOTS,
            history_frames=history_frames,
            m_candidates=m_candidates,
            alpha=p_val,
            min_lag_frames=step_cfg['min_lag_frames'],
            random_seed=random_seed,
            step_index=step_index,
            checkpoint_every=step_cfg['checkpoint_every'],
            partial_path=model_selection_dir / f"{prefix}{step_index}_null_partial.pkl",
            label=f"Step {step_index}",
        )

    current_model_features = []
    best_current_score = chance_ll
    # Per-fold scores of the incumbent model, stored as each step record's
    # `baseline_folds`. Before Step 0 the incumbent is the constant chance floor.
    best_current_folds = chance_ll
    time_indices = np.arange(history_frames, dtype=float)
    step_counter = 0
    # True when the checkpoint a resume restores ends a finished search: the anchor and
    # the forward loop are skipped and finalization runs.
    resume_converged = False

    fname = Path(univariate_results_path).name
    cond_match = re.search(r'((?:ephys_)?(?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"
    prediction_mode = settings['model_params']['model_target_vocal_type']

    # Mirror the Level-1 analysis_tag when a single onset target category is
    # active ('individual' mode only): embed the category column name and index
    # so step files for different categories (or label columns of different QLVM maps) never
    # collide in the same model-selection directory.
    onset_cat = settings['model_params']['onset_target_category']
    if onset_cat is not None and prediction_mode == 'individual':
        cat_col = settings['vocal_features']['usv_category_column_name']
        cat_seg = f"_cat_{cat_col}_{onset_cat}"
    else:
        cat_seg = ""
    # The onset target classes join the prefix when they are not the default ["usv"],
    # as in the Level-1 analysis_tag, so squeak-onset and USV-onset step files never collide.
    target_type_seg = onset_target_tag(settings['model_params']['onset_target_classes'])

    prefix = f"model_selection_{target_condition}_{prediction_mode}{target_type_seg}{cat_seg}_{split_strategy}_step_"

    existing_steps = []
    if model_selection_dir.is_dir():
        for f_name in (p.name for p in model_selection_dir.iterdir()):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                with contextlib.suppress(ValueError):
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))

    if existing_steps:
        last_step = max(existing_steps)
        print(f"\n[RESUME] Detected previous run. Checking Step {last_step}...")
        try:
            with Path(model_selection_dir / f"{prefix}{last_step}.pkl").open('rb') as f:
                last_results = pickle.load(f)
            _require_step_test_rule(last_results, model_selection_dir / f"{prefix}{last_step}.pkl")
            # Rebuild the state right after `last_step` from its recorded decision
            # (`_restore_selection_checkpoint`); the incumbent's per-fold log-losses are
            # kept as the step record's baseline.
            state = _restore_selection_checkpoint(
                last_results, 'mean_ll', 'se_ll', lambda entry: entry['ll'],
            )
            if state is None:
                print(
                    f"[RESUME] Step {last_step} accepted a feature without scored folds "
                    "(broken checkpoint); ignoring and starting fresh."
                )
                existing_steps = []
            else:
                current_model_features = state['features']
                best_current_score = state['score']
                best_current_folds = state['folds']
                step_counter = state['next_step']
                # A search that had accepted every ranked feature stopped on that, with no
                # rejecting step; it is converged too.
                resume_converged = state['converged'] or len(current_model_features) == len(ranked_features)
                # Finalization reads the step file the loop last wrote (`fname`); a converged
                # resume skips the loop, so it points at the restored checkpoint.
                fname = f"{prefix}{last_step}.pkl"
                if resume_converged:
                    print(f"[RESUME] Step {last_step}: selection already converged on "
                          f"{current_model_features}; finalizing.")
                else:
                    print(f"[RESUME] Continuing from Step {step_counter} with {current_model_features}.")
        except (OSError, pickle.UnpicklingError, KeyError, EOFError) as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # A fresh start (no checkpoint, or one the resume logic rejected) moves any step
    # files of an earlier run aside before Step 0 is written.
    if not existing_steps:
        _quarantine_stale_step_files(model_selection_dir, prefix)

    if use_top_rank_as_anchor and step_counter == 0 and not resume_converged:
        anchor_to_force = ranked_features[0]
        print(format_selection_step('Anchor', feature=anchor_to_force,
                                    detail='top-ranked; forced as the first candidate, tested against the constant model'))
        metrics = _onset_fit_candidates([], [anchor_to_force])[anchor_to_force]
        valid_ll = [x for x in metrics['ll'] if np.isfinite(x)]
        _require_scored_candidate('Anchor', 1, len(valid_ll))
        step_test = None
        if valid_ll:
            # The forced anchor is tested like every winner, with Bonferroni over every
            # screened feature it was forced ahead of.
            step_test = _onset_step_test(anchor_to_force, 0, len(ranked_features))
        anchor_accepted = step_test is not None and step_test['accepted']
        mean_ll = float(np.mean(valid_ll)) if valid_ll else float('nan')
        se_ll = float(np.std(valid_ll, ddof=1) / np.sqrt(len(valid_ll))) if len(valid_ll) > 1 else 0.0
        step_0_metadata = {
            'step_idx': 0,
            'current_features': [anchor_to_force] if anchor_accepted else [],
            'baseline_score': chance_ll, 'baseline_folds': chance_ll,
            'selected_feature': anchor_to_force if anchor_accepted else None,
            'candidates_summary': {
                anchor_to_force: {
                    **{key: metrics[key] for key in _BINARY_FOLD_METRIC_KEYS},
                    'mean_ll': mean_ll, 'se_ll': se_ll,
                }
            } if valid_ll else {},
            'step_test': step_test,
        }
        s0_name = f"{prefix}0.pkl"
        _dump_step_pickle(model_selection_dir / s0_name, _wrap_step(step_0_metadata))
        fname = s0_name
        if anchor_accepted:
            print(format_selection_step('Anchor', feature=anchor_to_force, decision='ACCEPT',
                                        metrics={'LL': mean_ll}))
            current_model_features = [anchor_to_force]
            best_current_score, best_current_folds = mean_ll, metrics['ll']
            step_counter = 1
        else:
            print(format_selection_step('Anchor', feature=anchor_to_force, decision='REJECT',
                                        detail='the forced anchor failed the shift test; selection finished'))
            resume_converged = True

    # A resume that restored a converged search skips the loop and finalizes directly.
    while not resume_converged:
        print(format_selection_step(f"Step {step_counter}",
                                    detail=f"best so far LL={best_current_score:.4f}"))
        step_results_metadata = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'baseline_folds': best_current_folds,
            'candidates_summary': {},
            'selected_feature': None,
            'step_test': None,
        }

        candidates_this_step = [feat for feat in ranked_features if feat not in current_model_features]
        m_candidates = len(candidates_this_step)
        per_candidate = _onset_fit_candidates(current_model_features, candidates_this_step)

        best_candidate, best_candidate_score = None, float('inf')
        best_candidate_folds = None
        for feat in candidates_this_step:
            metrics = per_candidate[feat]
            valid = [x for x in metrics['ll'] if np.isfinite(x)]
            if valid:
                # `np.std([x], ddof=1)` is NaN for a single finite fold (zero divisor);
                # guard with len > 1 so a one-fold success keeps a finite SE.
                mean_ll = np.mean(valid)
                se_ll = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
                print(format_selection_step(f"Step {step_counter}", feature=feat,
                                            metrics={'LL': mean_ll},
                                            detail=f"range {min(valid):.4f}-{max(valid):.4f}"))
                step_results_metadata['candidates_summary'][feat] = {
                    **{key: metrics[key] for key in _BINARY_FOLD_METRIC_KEYS},
                    'mean_ll': mean_ll, 'se_ll': se_ll,
                }
                if mean_ll < best_candidate_score:
                    best_candidate_score, best_candidate = mean_ll, feat
                    best_candidate_folds = metrics['ll']
            else:
                print(format_selection_step(f"Step {step_counter}", feature=feat,
                                            decision='FAILED (no valid scores)'))

        # The cross-validated winner is tested with the per-step shift test, Bonferroni
        # over the m candidates of this step; the step is accepted only when no shifted
        # refit reaches the observed in-sample gain.
        _require_scored_candidate(f"Step {step_counter}", m_candidates, int(best_candidate is not None))
        step_test = (_onset_step_test(best_candidate, step_counter, m_candidates)
                     if best_candidate is not None else None)
        step_results_metadata['step_test'] = step_test

        if step_test is not None and step_test['accepted']:
            print(format_selection_step(f"Step {step_counter}", feature=best_candidate,
                                        decision='ACCEPT',
                                        metrics={'LL': best_candidate_score,
                                                 'p_adj': step_test['p_adjusted']}))
            step_results_metadata['selected_feature'] = best_candidate
            current_model_features.append(best_candidate)

            fname = f"{prefix}{step_counter}.pkl"
            _dump_step_pickle(model_selection_dir / fname, _wrap_step(step_results_metadata))

            best_current_score, step_counter = best_candidate_score, step_counter + 1
            best_current_folds = best_candidate_folds
        else:
            if best_candidate is None:
                reject_detail = 'no candidate had valid scores; selection finished'
            else:
                reject_detail = (f"winner {best_candidate!r} failed the shift test "
                                 f"(p_adj {step_test['p_adjusted']:.4f} vs alpha {p_val}); selection finished")
            print(format_selection_step(f"Step {step_counter}", decision='REJECT', detail=reject_detail))
            step_results_metadata['selected_feature'] = None
            fname = f"{prefix}{step_counter}.pkl"
            _dump_step_pickle(model_selection_dir / fname, _wrap_step(step_results_metadata))
            break

        if len(current_model_features) == len(ranked_features):
            break

    print("\n--- Final Model Fit for Visualization (per-fold) ---")
    last_file = model_selection_dir / fname

    if not current_model_features:
        _write_empty_selection(last_file, _wrap_step, univariate_results_path, input_data_path, 'onset')
        return

    # Per-fold refit of the final multivariate GAM so the consumer
    # gets one filter-shape dict per CV fold. The downstream plotter
    # stacks the folds and renders a percentile band; a single-entry
    # list collapses the band to a zero-width line. The previous
    # version of this block did exactly that (one global-balanced
    # fit wrapped in a silent ``except Exception: print('Final fit
    # failed: ...')`` that hid every subsequent failure), and the
    # lost per-fold filter data was only spotted after a 6-day run.
    #
    # The loop body is factored into
    # ``compute_filter_shapes_per_fold_vocal_onset`` purely for
    # readability (this is its single caller).
    # Safety nets are layered here (caller side) rather than inside
    # the helper:
    #   * Per-fold exceptions are caught inside the helper and
    #     logged but do NOT abort the loop -- a few bad folds
    #     shouldn't lose the others.
    #   * If every fold fails, this caller RAISES (was silently
    #     swallowed before).
    #   * If only one fold succeeds, a warning is printed because
    #     the plotter's percentile band needs >= 2 folds to render.
    #
    # No outer try/except wraps this block on purpose. The silent
    # ``except Exception`` was the previous failure-hiding vector;
    # any unexpected error now propagates so the regression is
    # impossible to miss.
    print(f"  Calculating filter shapes across {len(cv_folds)} fold(s)...")
    final_fold_shapes = compute_filter_shapes_per_fold_vocal_onset(
        cv_folds=cv_folds,
        current_model_features=current_model_features,
        all_feature_data=all_feature_data,
        pooled_feature_cache=pooled_feature_cache,
        history_frames=pipeline.history_frames,
        n_splines_value=n_splines_value,
        n_splines_time=n_splines_time,
        gam_kwargs=gam_kwargs,
        random_seed=random_seed,
        time_indices=time_indices,
    )

    if not final_fold_shapes:
        msg = (
            f"All {len(cv_folds)} fold(s) failed during the final-refit "
            "filter-shape loop. The consolidated artifact would carry "
            "no usable per-fold filter data, which silently breaks the "
            "downstream plotter; raising here so the failure is visible "
            "at the source. Inspect the per-fold error lines above."
        )
        raise RuntimeError(msg)

    if len(final_fold_shapes) < 2:
        print(
            f"  WARNING: only {len(final_fold_shapes)} fold(s) produced "
            "filter shapes; the plotter's percentile band will collapse "
            f"to a single line (expected {len(cv_folds)} folds)."
        )

    # Honest held-out score (onset). Refit the accepted feature set on ALL
    # development sessions (balanced training pool) and score it once on the
    # reserved held-out sessions at their natural class prior (LL), mirroring the
    # session-strategy fold fit, plus a shuffled-label null control for the
    # held-out chance floor. This is the only number untouched by screening,
    # greedy acceptance, or the per-fold balancing. Skipped when holdout is
    # disabled (`held_sessions` empty), no feature was accepted, or the pooled
    # data is degenerate -- leaving `heldout` None so no-holdout runs match before.
    heldout_result = None
    if held_sessions and len(current_model_features) > 0:
        p_dev_by_feat, n_dev_by_feat, test_by_feat = [], [], []
        n_p_held = n_n_held = None
        for f_name in current_model_features:
            p_dev, n_dev = pool_session_arrays(all_feature_data[f_name], dev_sessions, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            p_held, n_held = pool_session_arrays(all_feature_data[f_name], held_sessions, pos_key="usv_feature_arr", neg_key="no_usv_feature_arr", n_frames=history_frames)
            p_dev_by_feat.append(p_dev)
            n_dev_by_feat.append(n_dev)
            test_by_feat.append(np.concatenate((p_held, n_held)))
            if n_p_held is None:
                n_p_held, n_n_held = p_held.shape[0], n_held.shape[0]
        n_keep = min(p_dev_by_feat[0].shape[0], n_dev_by_feat[0].shape[0])
        heldout_result = {
            'features': list(current_model_features),
            'n_held_out_sessions': len(held_sessions),
            'n_held_out_events': int(n_p_held + n_n_held),
        }
        if n_keep > 0 and (n_p_held + n_n_held) > 0:
            gam_terms_final = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(current_model_features)):
                gam_terms_final += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
            y_held_test = np.concatenate((np.ones(n_p_held), np.zeros(n_n_held)))
            y_te_int = y_held_test.astype(int)
            X_te_gam = get_unrolled_X_for_multivariate(test_by_feat, history_frames)
            # Distinct-but-deterministic seed for the balanced development draw and
            # the null permutation; offset the fold seeds by `n_splits_selection`.
            held_rng = np.random.default_rng(random_seed + n_splits_selection + 1)
            idx_p = held_rng.choice(p_dev_by_feat[0].shape[0], n_keep, replace=False)
            idx_n = held_rng.choice(n_dev_by_feat[0].shape[0], n_keep, replace=False)
            X_train_list = [
                np.concatenate((p_dev_by_feat[k][idx_p], n_dev_by_feat[k][idx_n]))
                for k in range(len(current_model_features))
            ]
            X_tr_gam = get_unrolled_X_for_multivariate(X_train_list, history_frames)
            y_tr_bal = np.concatenate((np.ones(n_keep), np.zeros(n_keep)))
            for branch in ('actual', 'null'):
                # The held-out null shuffles labels across the development set.
                y_tr_use = (y_tr_bal if branch == 'actual'
                            else y_tr_bal[null_target_permutation(y_tr_bal.shape[0], held_rng)])
                rec = {'ll': np.nan, 'auc': np.nan, 'score': np.nan,
                       'deviance_explained': np.nan, 'brier': np.nan,
                       'heldout_fit_failed': False, 'heldout_fit_error': None}
                try:
                    gam = LogisticGAM(gam_terms_final, **gam_kwargs).fit(
                        X_tr_gam, np.repeat(y_tr_use.astype(float), history_frames)
                    )
                    y_proba_tiled = gam.predict_proba(X_te_gam)
                    y_proba_mean = np.mean(y_proba_tiled.reshape(len(y_held_test), history_frames), axis=1)
                    y_pred_mean = (y_proba_mean >= settings['diagnostics']['binary_decision_threshold']).astype(int)
                    rec['ll'] = log_loss(y_te_int, np.clip(y_proba_mean, 1e-15, 1 - 1e-15))
                    rec['deviance_explained'] = 1.0 - rec['ll'] / np.log(2)
                    rec['score'] = balanced_accuracy_score(y_te_int, y_pred_mean)
                    rec['brier'] = float(brier_score_loss(y_te_int, y_proba_mean))
                    if len(np.unique(y_te_int)) > 1:
                        rec['auc'] = roc_auc_score(y_te_int, y_proba_mean)
                except Exception as e:
                    # The held-out number is touched once; a failed refit is recorded as such
                    # (flag + error text beside the NaN metrics) and warned about, never left
                    # as a silent NaN.
                    rec['heldout_fit_failed'] = True
                    rec['heldout_fit_error'] = f"{type(e).__name__}: {e}"
                    warnings.warn(f"Held-out refit failed (onset {branch}): {rec['heldout_fit_error']}",
                                  RuntimeWarning, stacklevel=2)
                heldout_result[branch] = rec

    with Path(last_file).open('rb') as f:
        data = pickle.load(f)

    # Strip existing metadata blocks so `_wrap_step` can re-inject
    # them cleanly without colliding with the prior write's keys.
    for _k in RESERVED_METADATA_KEYS:
        data.pop(_k, None)

    data.update({
        'final_model_features': current_model_features,
        'filter_shapes': final_fold_shapes,
        'heldout': heldout_result,
        'univariate_results_path': univariate_results_path,
        'input_data_path': input_data_path,
    })

    _dump_step_pickle(last_file, _wrap_step(data))
    metrics_by_strategy = {'Final': {'LL': best_current_score},
                           'Chance': {'LL': chance_ll}}
    # Held-out LL (development-refit, scored once) -- the honest number alongside
    # the optimistic development-CV `Final`.
    if heldout_result is not None and 'actual' in heldout_result:
        metrics_by_strategy['Held-out'] = {'LL': heldout_result['actual']['ll']}
    print(format_run_summary(
        label=f"onset model-selection ({len(current_model_features)} features)",
        metrics_by_strategy=metrics_by_strategy,
        out_path=str(last_file),
    ))


def _pool_category_features(
        all_feature_data: dict,
        features_to_load: list,
        session_list: np.ndarray,
        history_frames: int
) -> tuple[list, list]:
    """
    Extracts and concatenates Target and Other arrays for a given list of features
    across specified sessions.

    Unlike older implementations, this function strictly handles pooling without
    enforcing balancing. This separation of concerns allows the upstream
    cross-validation logic to seamlessly support both 'session' and 'mixed'
    splitting strategies before the final downsampling occurs.

    Parameters
    ----------
    all_feature_data : dict
        The nested dictionary containing raw data for all features.
        Structure: `dict[feature_name][session_id]['target_feature_arr' | 'other_feature_arr']`.
    features_to_load : list of str
        The list of feature names (keys in `all_feature_data`) to extract.
    session_list : np.ndarray
        Array of session IDs (strings) to pool data from.
    history_frames : int
        The number of time columns in the feature matrices.

    Returns
    -------
    tuple[list, list]
        - X_target_list: List of pooled (n_samples, history_frames) arrays for the Target class.
        - X_other_list: List of pooled (n_samples, history_frames) arrays for the Other class.
    """

    raw_target = {feat: [] for feat in features_to_load}
    raw_other = {feat: [] for feat in features_to_load}

    for sess in session_list:
        for feat in features_to_load:
            if sess in all_feature_data[feat]:
                raw_target[feat].append(all_feature_data[feat][sess]['target_feature_arr'])
                raw_other[feat].append(all_feature_data[feat][sess]['other_feature_arr'])

    X_target_list = [np.concatenate(raw_target[f], axis=0) if raw_target[f] else np.empty((0, history_frames)) for f in features_to_load]
    X_other_list = [np.concatenate(raw_other[f], axis=0) if raw_other[f] else np.empty((0, history_frames)) for f in features_to_load]

    return X_target_list, X_other_list


def _category_row_sessions(
        all_feature_data: dict,
        feature: str,
        session_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The session of every row `_pool_category_features` pools for one feature, in the
    same order (sessions in `session_list` order, a session's rows contiguous), for the
    Target and the Other class. Every feature carries the same rows per session, so
    one feature's labels serve the whole feature list; they let the selection split
    its inner cross-validation by session.

    Parameters
    ----------
    all_feature_data (dict)
        ``dict[feature_name][session_id]['target_feature_arr' | 'other_feature_arr']``.
    feature (str)
        The feature whose rows are labelled.
    session_list (np.ndarray)
        The sessions pooled, in pooling order.

    Returns
    -------
    groups_target, groups_other (tuple[np.ndarray, np.ndarray])
        ``(n_target,)`` and ``(n_other,)`` session ids.
    """

    target, other = [], []
    for sess in session_list:
        if sess in all_feature_data[feature]:
            target.append(np.full(all_feature_data[feature][sess]['target_feature_arr'].shape[0], sess, dtype=object))
            other.append(np.full(all_feature_data[feature][sess]['other_feature_arr'].shape[0], sess, dtype=object))
    empty = np.empty(0, dtype=object)
    return (np.concatenate(target) if target else empty), (np.concatenate(other) if other else empty)


def _balance_multivariate_arrays(
        X_targ_list: list,
        X_other_list: list,
        random_seed: int,
        groups_target: np.ndarray | None = None,
        groups_other: np.ndarray | None = None
) -> tuple:
    """
    Downsamples the majority class to enforce a strict 50/50 balance across all
    feature arrays simultaneously.

    This ensures that row-alignment is maintained across the multivariate feature
    lists, preventing misalignment between predictors during classification.

    Parameters
    ----------
    X_targ_list : list of np.ndarray
        The aligned list of Target class arrays.
    X_other_list : list of np.ndarray
        The aligned list of Other class arrays.
    random_seed : int
        Seed for the random number generator used for downsampling indices.
    groups_target : np.ndarray | None
        Session label of every Target row; when given (with `groups_other`) the
        balanced labels are returned too.
    groups_other : np.ndarray | None
        Session label of every Other row.

    Returns
    -------
    tuple
        1. X_tr_t_bal (list of np.ndarray): Balanced Target arrays.
        2. X_tr_o_bal (list of np.ndarray): Balanced Other arrays.
        3. y_target (np.ndarray): Array of ones.
        4. y_other (np.ndarray): Array of zeros.
        With groups, 5. and 6. are the balanced Target and Other session labels.
        Returns ([], [], None, None) (plus None, None with groups) if data is insufficient.
    """

    with_groups = groups_target is not None and groups_other is not None
    n_target = X_targ_list[0].shape[0]
    n_other = X_other_list[0].shape[0]

    if n_target == 0 or n_other == 0:
        return ([], [], None, None, None, None) if with_groups else ([], [], None, None)

    limit = min(n_target, n_other)
    rng = np.random.default_rng(random_seed)
    idx_target = rng.choice(n_target, limit, replace=False)
    idx_other = rng.choice(n_other, limit, replace=False)

    X_tr_t_bal = [x[idx_target] for x in X_targ_list]
    X_tr_o_bal = [x[idx_other] for x in X_other_list]

    y_target = np.ones(limit)
    y_other = np.zeros(limit)

    if with_groups:
        return X_tr_t_bal, X_tr_o_bal, y_target, y_other, groups_target[idx_target], groups_other[idx_other]
    return X_tr_t_bal, X_tr_o_bal, y_target, y_other


#: Start-frame slot -> pickle array key of the binary category pipeline.
_CATEGORY_WINDOW_SLOTS = {'target': 'target_feature_arr', 'other': 'other_feature_arr'}


def _category_candidate_fold(feat: str, fold_i: int) -> dict:
    """
    Description
    -----------
    One (candidate, fold) fit of the binary category selector, run in a pool worker:
    pools the fold's training and test rows of the incumbent features plus ``feat``
    ('session' strategy: the incumbent block comes pre-pooled per fold from
    ``fold_base_pooled``, only the candidate is pooled here; 'mixed' strategy: the
    rows come from ``pooled_category_cache`` sliced by the fold's indices), balances
    the training rows 50/50 with the fold's seed, fits the configured engine
    (``LogisticRegressionCV`` on the basis projection, or ``LogisticGAM`` on the
    unrolled tensor terms) and scores the test rows at their natural class prior.
    The data comes from the ``'category_step'`` pool-context entry the selector
    stores before creating the pool. The operations are those of the former inline
    loop; a fold the loop used to skip silently (no balanced training rows, or no
    test rows) is now reported as a failed fold, so every per-fold list keeps one
    entry per fold.

    Parameters
    ----------
    feat (str)
        The candidate feature.
    fold_i (int)
        Fold index.

    Returns
    -------
    result (dict)
        ``{'ok': True, <metric key>: value, ...}`` with the keys of
        ``_BINARY_FOLD_METRIC_KEYS``, or ``{'ok': False, 'error': str}``.
    """

    ctx = pool_context('category_step')
    tr_idx, te_idx = ctx['cv_folds'][fold_i]
    all_feature_data = ctx['all_feature_data']
    all_sessions_arr = ctx['all_sessions_arr']
    history_frames = ctx['history_frames']
    current_model_features = ctx['current_model_features']
    model_type = ctx['model_type']
    random_seed = ctx['random_seed']
    gam_kwargs = ctx['gam_kwargs']
    lr_params = ctx['lr_params']
    basis_matrix = ctx['basis_matrix']
    n_targ_total = ctx['n_targ_total']
    trial_feats = [*current_model_features, feat]
    try:
        if ctx['split_strategy'] == 'session':
            # Pool only the new candidate feature; the base features were
            # pooled once per fold by the selector. List order matches trial_feats
            # (base first, then the new feat), so this is byte-identical.
            b_tr_t, b_tr_o, b_te_t, b_te_o = ctx['fold_base_pooled'][fold_i]
            n_tr_t, n_tr_o = _pool_category_features(all_feature_data, [feat], all_sessions_arr[tr_idx], history_frames)
            n_te_t, n_te_o = _pool_category_features(all_feature_data, [feat], all_sessions_arr[te_idx], history_frames)
            X_tr_t_raw, X_tr_o_raw = b_tr_t + n_tr_t, b_tr_o + n_tr_o
            X_te_t_raw, X_te_o_raw = b_te_t + n_te_t, b_te_o + n_te_o
            g_tr_t_raw, g_tr_o_raw = _category_row_sessions(all_feature_data, feat, all_sessions_arr[tr_idx])
        else:
            # Reassemble the per-trial feature list from the pre-pooled cache so
            # every fold reuses the same dense (target, other) arrays.
            pooled_category_cache = ctx['pooled_category_cache']
            X_all_t_raw = [pooled_category_cache[f]['target'] for f in trial_feats]
            X_all_o_raw = [pooled_category_cache[f]['other'] for f in trial_feats]

            tr_targ_idx = tr_idx[tr_idx < n_targ_total]
            tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total
            te_targ_idx = te_idx[te_idx < n_targ_total]
            te_other_idx = te_idx[te_idx >= n_targ_total] - n_targ_total

            X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
            X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]
            X_te_t_raw = [x[te_targ_idx] for x in X_all_t_raw]
            X_te_o_raw = [x[te_other_idx] for x in X_all_o_raw]
            g_tr_t_raw, g_tr_o_raw = ctx['g_all_targ'][tr_targ_idx], ctx['g_all_other'][tr_other_idx]

        X_tr_t, X_tr_o, y_tr_t, y_tr_o, g_tr_t, g_tr_o = _balance_multivariate_arrays(
            X_tr_t_raw, X_tr_o_raw, random_seed + fold_i, groups_target=g_tr_t_raw, groups_other=g_tr_o_raw)

        # Test set: natural class prior (no balancing).
        if not X_tr_t or len(X_te_t_raw) == 0:
            msg = "fold skipped: no balanced training rows"
            raise ValueError(msg)
        n_te_targ = X_te_t_raw[0].shape[0]
        n_te_other = X_te_o_raw[0].shape[0]
        if (n_te_targ + n_te_other) == 0:
            msg = "fold skipped: no test rows"
            raise ValueError(msg)

        X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o, strict=False)]
        y_tr = np.concatenate([y_tr_t, y_tr_o])
        perm = np.random.default_rng(random_seed + fold_i).permutation(len(y_tr))
        groups_tr = np.concatenate([g_tr_t, g_tr_o])

        X_te_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_te_t_raw, X_te_o_raw, strict=False)]
        y_te = np.concatenate([np.ones(n_te_targ), np.zeros(n_te_other)]).astype(int)

        if model_type == 'sklearn':
            X_tr_stacked = np.hstack([np.dot(x[perm], basis_matrix) for x in X_tr_list])
            X_te_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_te_list])

            model = LogisticRegressionCV(
                penalty=lr_params['penalty'],
                Cs=lr_params['cs'],
                # Inner folds hold out whole training sessions, as the outer folds do.
                cv=classification_inner_cv_splits(y_tr[perm], groups_tr[perm], lr_params['cv'], random_seed),
                class_weight='balanced',
                solver=lr_params['solver'],
                max_iter=lr_params['max_iter'],
                random_state=random_seed
            )
            fit_start = time.perf_counter()
            model.fit(X_tr_stacked, y_tr[perm])
            fit_time = float(time.perf_counter() - fit_start)
            y_proba = model.predict_proba(X_te_stacked)[:, 1]
            y_pred = model.predict(X_te_stacked)
            try:
                fold_n_iter = float(np.max(model.n_iter_))
                fold_converged = bool(fold_n_iter < lr_params['max_iter'])
            except Exception:
                fold_n_iter, fold_converged = np.nan, False
        else:
            gam_terms = te(0, 1, n_splines=[ctx['n_splines_value'], ctx['n_splines_time']])
            for i in range(1, len(trial_feats)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[ctx['n_splines_value'], ctx['n_splines_time']])
            X_tr_gam = get_unrolled_X_for_multivariate([x[perm] for x in X_tr_list], history_frames)
            y_tr_gam = np.repeat(y_tr[perm].astype(float), history_frames)
            fit_start = time.perf_counter()
            gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_tr_gam, y_tr_gam)
            fit_time = float(time.perf_counter() - fit_start)

            X_te_gam = get_unrolled_X_for_multivariate(X_te_list, history_frames)
            y_proba_tiled = gam.predict_proba(X_te_gam)
            y_proba = np.mean(y_proba_tiled.reshape(len(y_te), history_frames), axis=1)
            y_pred = (y_proba >= ctx['decision_threshold']).astype(int)
            gam_diffs = gam.logs_['diffs']
            fold_n_iter = float(len(gam_diffs))
            fold_converged = bool(len(gam_diffs) > 0 and len(gam_diffs) < gam_kwargs['max_iter'])

        result = {'ok': True}
        result['ll'] = log_loss(y_te, np.clip(y_proba, 1e-15, 1 - 1e-15))
        result['auc'] = roc_auc_score(y_te, y_proba)
        result['f1'] = f1_score(y_te, y_pred)
        result['recall'] = recall_score(y_te, y_pred, zero_division=0)
        result['score'] = balanced_accuracy_score(y_te, y_pred)
        result['brier'] = float(brier_score_loss(y_te, y_proba))
        try:
            y_proba_2d = np.column_stack([1.0 - y_proba, y_proba])
            result['ece'] = expected_calibration_error(y_te, y_pred, y_proba_2d, n_bins=ctx['ece_n_bins'])
        except Exception:
            result['ece'] = np.nan
        result['mcc'] = safe_matthews_corrcoef(y_te, y_pred)
        result['confusion_matrix'] = safe_confusion_matrix(y_te, y_pred, labels=np.array([0, 1]))
        result['n_iter'] = fold_n_iter
        result['converged'] = fold_converged
        result['fit_time'] = fit_time
        gc.collect()
        return result
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


def vocal_category_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str | None = None,
        use_top_rank_as_anchor: bool = False,
        *,
        p_val: float
) -> None:
    """
    Performs Forward Stepwise Selection for Vocal Category prediction using strict
    mixed or session-based splitting, class balancing, and engine toggling (sklearn/pygam).

    This function identifies the optimal subset of behavioral features that predict
    a specific USV category (one-vs-rest). It adapts the standard forward selection
    algorithm to handle the specific data structure of vocal categories ('target' vs 'other')
    and enforces a consistent splitting & balancing invariant (see below) across
    all cross-validation folds.

    Splitting & balancing invariant:
    --------------------------------
    Across both 'session' and 'mixed' strategies, the training fold is down-sampled
    to a 50/50 class balance (Target vs. Other), while the test fold preserves the
    natural class prior of the source data. The 'mixed' strategy achieves this by
    running `StratifiedShuffleSplit` on the *unbalanced* pool and balancing the
    training half inside the per-fold loop. Reported metrics should therefore be
    imbalance-robust: the `score` field stores `balanced_accuracy`, paired with
    AUC and log-loss.

    Engine Flexibility:
    -------------------
    Reads the user-defined `model_engine` from the JSON settings to dynamically build models:
    - 'sklearn': Utilizes `LogisticRegressionCV` combined with linear basis projection
                 (e.g., raised_cosines, b-splines) to rapidly isolate temporal predictors.
    - 'pygam': Utilizes `LogisticGAM` with unrolled tensor product splines to capture
               non-linear log-odds surfaces.

    The process follows these steps:
    1.  Extracts the target category and sex directly from the univariate path.
    2.  Loads univariate results and ranks candidate features by Log-Likelihood (LL).
    3.  Evaluates 'split_strategy'. If 'session', isolates whole recording days.
        If 'mixed', pools all data and uses StratifiedShuffleSplit to ensure proportional
        epoch-level splitting across all folds.
    4.  Forward selection loop:
        - Starts with an empty model (or the top-ranked anchor).
        - Iteratively tests adding every remaining candidate feature.
        - Calculates and saves raw lists of metrics (Log-Loss, AUC, F1,
          Balanced Accuracy, Recall) for every fold for every candidate.
    5.  Decision rule: the cross-validated winner is tested with the per-step
        shift test (``step_shift_test.run_step_shift_test``): the incumbent and
        the incumbent plus the winner are refit on the balanced development draw
        (the held-out refit's draw), the statistic is the in-sample Bernoulli
        log-likelihood gain per event, its null comes from refits after circularly
        shifting the winner's trace within every development session and
        re-cutting its windows at the stored start frames, ``floor(m / alpha)``
        times with Bonferroni correction over the ``m`` candidates
        (``alpha = model_params.selection_p_val``). The first step is tested the
        same way against the constant model.
    6.  Final Refit: Computes filter shapes for visualization depending on the chosen
        engine (back-projection for sklearn, partial dependence for pygam).

    Parameters
    ----------
    univariate_results_path : str
        The absolute path to the .pkl file containing univariate modeling results.
    input_data_path : str
        The absolute path to the .pkl file containing the raw feature data.
    output_directory : str
        The specific directory path where the resulting step files (.pkl) will be saved.
    settings_path : str, optional
        The path to the 'modeling_settings.json' configuration file.
    use_top_rank_as_anchor : bool, default False
        If True, the selection process initializes with the highest-ranked univariate feature.
    p_val : float
        The alpha level used to determine significance against the shuffled null.
        Keyword-only and required — callers must specify it explicitly (no default).

    Returns
    -------
    None
    """

    chance_ll = np.log(2)  # mean log-loss of a balanced 50/50 coin: the binary chance floor

    if settings_path is None:
        settings_path_obj = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with Path(settings_path).open() as f:
            settings = json.load(f)
    except FileNotFoundError as err:
        msg = f"Settings file not found at {settings_path}"
        raise FileNotFoundError(msg) from err

    model_selection_dir = Path(output_directory)
    model_selection_dir.mkdir(parents=True, exist_ok=True)

    # Extract target metadata safely
    fname = Path(univariate_results_path).name

    cat_match = re.search(r'category_(\d+)', fname)
    target_category = f"category_{cat_match.group(1)}" if cat_match else "category_unknown"
    cond_match = re.search(r'((?:ephys_)?(?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"

    with Path(univariate_results_path).open('rb') as f:
        univariate_data = pickle.load(f)

    # Filter candidates safely
    candidates = []
    # Defer the upstream-input metadata harvest until `all_feature_data`
    # is loaded below. Strip just the univariate-side blocks here so the
    # `for feat_name, payload in univariate_data.items()` loop only sees
    # behavioral-feature entries.
    _univ_pre_md = univariate_data.pop('_run_metadata', None)
    _require_analysis_type(univariate_data, 'category', univariate_results_path)
    _input_pre_md_from_univ = univariate_data.pop('_input_metadata', None)
    univariate_data.pop('_consolidation_metadata', None)
    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results or 'll' not in results['actual']:
            continue

        actual_ll = results['actual']['ll']
        null_ll = results['null']['ll']

        valid_actual = actual_ll[~np.isnan(actual_ll)]
        valid_null = null_ll[~np.isnan(null_ll)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        mean_actual_ll = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=(p_val / len(univariate_data)) * 100)

        if mean_actual_ll < null_threshold:
            candidates.append({'feature': feat_name, 'mean_ll': mean_actual_ll})

    candidates.sort(key=lambda x: x['mean_ll'])
    ranked_features = [x['feature'] for x in candidates]
    if not ranked_features:
        print("No significant features found. Aborting.")
        return
    print(f"Identified {len(ranked_features)} significant candidates. Top: {ranked_features[0]}")

    print("Loading raw input data...")
    all_feature_data = load_pickle_modeling_data(input_data_path)

    # Final upstream-metadata harvest. Prefer the value the consolidator
    # asserted across every per-feature pickle; fall back to the
    # Level-1 input pickle's copy for legacy runs.
    _require_analysis_type(all_feature_data, 'category', input_data_path)
    _input_md = _input_pre_md_from_univ
    if _input_md is None and '_input_metadata' in all_feature_data:
        _input_md = all_feature_data.pop('_input_metadata')
    else:
        all_feature_data.pop('_input_metadata', None)
    all_feature_data.pop('_run_metadata', None)
    all_feature_data.pop('_consolidation_metadata', None)
    _univariate_md = _univ_pre_md

    common_sessions = set(all_feature_data[ranked_features[0]].keys())
    for feat in ranked_features[1:]:
        common_sessions = common_sessions.intersection(set(all_feature_data[feat].keys()))

    valid_sessions = []
    for sess in common_sessions:
        try:
            if (all_feature_data[ranked_features[0]][sess]['target_feature_arr'].shape[0] > 0 and
                    all_feature_data[ranked_features[0]][sess]['other_feature_arr'].shape[0] > 0):
                valid_sessions.append(sess)
        except KeyError:
            continue

    all_sessions_arr = np.array(sorted(valid_sessions))
    if len(all_sessions_arr) < 2:
        msg = "Not enough common sessions for cross-validation."
        raise ValueError(msg)

    # Held-out reserve carved ONCE at extraction time and shipped in the
    # univariate `_input_metadata.held_out_session_ids`. Restrict every selection
    # fold (and the 'mixed' pooled cache) to DEVELOPMENT sessions by shrinking
    # `all_sessions_arr`; the held-out sessions are kept aside in
    # `held_sessions_arr` and scored exactly once at finalization by refitting the
    # accepted model on all development data. An empty list -- holdout disabled or
    # a pre-v2 pickle -- leaves `all_sessions_arr` unchanged and `held_sessions_arr`
    # empty, an exact no-op.
    _held_out_session_ids = held_out_session_ids_from_metadata(_input_md if _input_md is not None else {})
    _held_set = set(_held_out_session_ids)
    held_sessions_arr = np.array([s for s in all_sessions_arr if s in _held_set])
    all_sessions_arr = np.array([s for s in all_sessions_arr if s not in _held_set])
    if held_sessions_arr.size > 0:
        print(
            f"Held-out reserve: {held_sessions_arr.size} session(s) excluded from "
            f"selection; scored once at finalization."
        )
    if len(all_sessions_arr) < 2:
        msg = "Not enough development sessions for cross-validation after reserving the held-out block."
        raise ValueError(msg)

    # Strict Dictionary Access Only (No .get())
    camera_rate = settings['io']['camera_sampling_rate']
    filter_history_sec = settings['model_params']['filter_history']
    history_frames = int(np.floor(camera_rate * filter_history_sec))

    step_cfg = _step_test_settings(settings, float(camera_rate))
    traces_path, window_starts = _require_shift_test_inputs(_input_md, input_data_path, float(camera_rate))
    # Lag-zero self-check of the shift test's inputs: every stored window of every
    # screened feature must re-cut exactly from the trace sidecar at its stored start.
    verify_windows_against_traces(
        all_feature_data, traces_path, window_starts, _CATEGORY_WINDOW_SLOTS, history_frames,
        features=ranked_features, sessions=list(all_sessions_arr),
    )
    print(f"Shift-test inputs verified: {len(ranked_features)} feature(s) x {len(all_sessions_arr)} development "
          f"session(s) re-cut exactly from {Path(traces_path).name}; lag floor "
          f"{step_cfg['min_lag_seconds']} s = {step_cfg['min_lag_frames']} frames.")

    model_ops = settings['model_params']
    model_type = model_ops['model_engine']
    n_splits = settings['model_validation']['n_cv_folds']
    test_prop = settings['model_validation']['cv_validation_proportion']
    split_strategy = settings['model_validation']['split_strategy']
    random_seed = settings['model_validation']['random_seed']

    basis_matrix = None
    gam_kwargs = {}
    n_splines_time = 0
    n_splines_value = 0
    lr_params = {}

    # Initialize Engine Logic
    if model_type == 'sklearn':
        basis_type = model_ops['model_basis_function']
        if basis_type == 'raised_cosine':
            p = settings['hyperparameters']['basis_functions']['raised_cosine']
            kp = int(np.floor(history_frames * p['kpeaks_proportion']))
            basis_matrix = raised_cosine(neye=p['neye'], ncos=p['ncos'], kpeaks=[0, kp], b=p['b'], w=history_frames)
        elif basis_type == 'bspline':
            p = settings['hyperparameters']['basis_functions']['bspline']
            deg = p['degree']
            max_k = max(0, history_frames - deg)
            knots = np.linspace(0, max_k, p['n_splines'] - deg + 1).astype(int)
            basis_matrix = _normalizecols(bsplines(width=history_frames, positions=knots, degree=deg))
        elif basis_type == 'laplacian_pyramid':
            p = settings['hyperparameters']['basis_functions']['laplacian_pyramid']
            basis_matrix = _normalizecols(laplacian_pyramid(width=history_frames, levels=p['levels'], step=p['step'], fwhm=p['fwhm']))
        elif basis_type == 'identity':
            basis_matrix = identity(width=history_frames)

        lr_params = settings['hyperparameters']['classical']['logistic_regression']
        print(f"Engine: Sklearn LogisticRegressionCV with '{basis_type}' projection.")

    elif model_type == 'pygam':
        pygam_params = settings['hyperparameters']['classical']['pygam']
        n_splines_time = pygam_params['n_splines_time']
        n_splines_value = pygam_params['n_splines_value']
        gam_kwargs = {
            'max_iter': int(pygam_params['max_iterations']),
            'tol': float(pygam_params['tol_val']),
            'lam': pygam_params['lam_penalty']
        }
        print("Engine: PyGAM LogisticGAM with Tensor Splines.")
    else:
        msg = "model_engine in settings must be either 'sklearn' or 'pygam'."
        raise ValueError(msg)

    print(format_run_header(
        task='SELECTION:CATEGORY',
        engine=settings['model_params']['model_engine'],
        feature=f"{target_category} ({target_condition})",
        input_files={'univariate results': univariate_results_path,
                     'input data': input_data_path,
                     'settings': settings_path},
        output_directory=str(model_selection_dir),
        split_strategy=split_strategy,
        n_splits=n_splits,
    ))

    # Establish CV indices globally to maintain fold consistency across features
    cv_folds = []
    n_targ_total, n_other_total = 0, 0

    # Pre-pool each candidate feature's full (target, other) arrays once for
    # the 'mixed' strategy. Because the pooled result depends only on the
    # feature (session list is always `all_sessions_arr`), the anchor and
    # forward loops read from this cache instead of re-running
    # `_pool_category_features` per fold per trial feature — the dominant cost
    # of the inner loop at ~ (n_steps * n_features * n_folds) pool calls.
    pooled_category_cache = {}
    if split_strategy == 'session':
        ss = ShuffleSplit(n_splits=n_splits, test_size=bounded_test_proportion(test_prop, len(all_sessions_arr)), random_state=random_seed)
        cv_folds = list(ss.split(all_sessions_arr))
    elif split_strategy == 'mixed':
        for _feat_name in ranked_features:
            _t_list, _o_list = _pool_category_features(
                all_feature_data, [_feat_name], all_sessions_arr, history_frames
            )
            pooled_category_cache[_feat_name] = {
                'target': _t_list[0],
                'other': _o_list[0],
            }
        ref_targ = pooled_category_cache[ranked_features[0]]['target']
        ref_other = pooled_category_cache[ranked_features[0]]['other']
        # Session of every cached row (the same for every feature), sliced per fold
        # like the arrays, for the grouped inner CV.
        g_all_targ, g_all_other = _category_row_sessions(all_feature_data, ranked_features[0], all_sessions_arr)
        n_targ_total = ref_targ.shape[0]
        n_other_total = ref_other.shape[0]

        # The 'mixed' fold loop builds train/test indices against
        # `(n_targ_total, n_other_total)` derived from the anchor and then
        # applies them to every trial feature's cached arrays. If a feature
        # ever has a different per-class sample count, the indices land on
        # wrong rows. Fail loudly here. ``raise`` rather than ``assert``
        # so the check survives under ``python -O``.
        for _feat_name, _entry in pooled_category_cache.items():
            if (
                _entry['target'].shape[0] != n_targ_total
                or _entry['other'].shape[0] != n_other_total
            ):
                msg = (
                    f"pooled_category_cache misalignment: feature {_feat_name!r} has "
                    f"({_entry['target'].shape[0]}, {_entry['other'].shape[0]}) target/other "
                    f"samples but the anchor has ({n_targ_total}, {n_other_total})."
                )
                raise ValueError(msg)

        y_dummy = np.hstack([np.ones(n_targ_total), np.zeros(n_other_total)])
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        cv_folds = list(sss.split(np.zeros(len(y_dummy)), y_dummy))
    else:
        msg_0 = "split_strategy in settings must be either 'session' or 'mixed'."
        raise ValueError(msg_0)

    # Build the selection-level `_run_metadata` block + step-wrapper once.
    _run_md = build_selection_metadata(
        modeling_settings=settings,
        selection_function='vocal_category_model_selection',
        selection_metric='log_likelihood',
        n_splits_selection=int(n_splits),
        test_proportion=float(test_prop),
        split_strategy=split_strategy,
        random_seed=int(random_seed),
        aic_termination_used=False,
        n_anchor_features=len(ranked_features),
        anchor_feature=ranked_features[0],
        gam_kwargs=gam_kwargs,
        extra_knobs={
            'use_top_rank_as_anchor': bool(use_top_rank_as_anchor),
            'p_val': float(p_val),
            'model_type': model_type,
            'n_splines_time': int(n_splines_time),
            'n_splines_value': int(n_splines_value),
            'target_category': target_category,
            'target_condition': target_condition,
            # Held-out provenance (also in the embedded `_input_metadata` sibling);
            # duplicated here so the selection run-metadata is self-describing.
            'held_out_test_proportion': float(settings['model_validation']['held_out_test_proportion']),
            'held_out_session_ids': list(_held_out_session_ids),
            'n_held_out_sessions': int(held_sessions_arr.size),
            # The per-step acceptance rule and its knobs (a resume refuses another rule).
            'step_test': {
                'rule': STEP_TEST_RULE,
                'screened_features': list(ranked_features),
                'statistic': 'bernoulli_log_likelihood_per_event',
                'alpha': float(p_val),
                'min_lag_seconds': step_cfg['min_lag_seconds'],
                'min_lag_frames': step_cfg['min_lag_frames'],
                'checkpoint_every': step_cfg['checkpoint_every'],
                'pool_workers': step_cfg['pool_workers'],
                'refit_rows': 'balanced_development_draw',
                'traces_path': traces_path,
            },
        },
        settings_path=settings_path,
    )
    _wrap_step = _make_step_wrapper(_input_md, _univariate_md, _run_md)

    # Pin which USV category column generated the binary target so the
    # per-step prefix (and downstream consolidation) carries the choice
    # forward in the filename — e.g. `qlvm_category`.
    _column_name_cats = _input_md['analysis_specific']['usv_category_column_name']
    # `target_category` is the human-readable `category_<idx>` (kept for
    # metadata + console output); strip the redundant prefix when
    # building the step filename so it reads `category_<col>_<idx>`
    # instead of `category_<col>_category_<idx>`.
    _cat_idx = target_category.replace('category_', '', 1)
    prefix = f"model_selection_category_{_column_name_cats}_{_cat_idx}_{target_condition}_step_"
    existing_steps = []
    if model_selection_dir.is_dir():
        for f_name in (p.name for p in model_selection_dir.iterdir()):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                with contextlib.suppress(ValueError):
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))

    step_counter, current_model_features = 0, []
    best_current_score = chance_ll
    # Per-fold scores of the incumbent, stored as each step record's
    # `baseline_folds`. Before Step 0 the incumbent is the constant chance floor.
    best_current_folds = chance_ll
    # True when the checkpoint a resume restores ends a finished search: the anchor and
    # the forward loop are skipped and finalization runs.
    resume_converged = False

    if existing_steps:
        last_step = max(existing_steps)
        print(f"\n[RESUME] Detected previous run. Checking Step {last_step}...")
        try:
            with Path(model_selection_dir / f"{prefix}{last_step}.pkl").open('rb') as f:
                last_results = pickle.load(f)
            _require_step_test_rule(last_results, model_selection_dir / f"{prefix}{last_step}.pkl")
            # Rebuild the state right after `last_step` from its recorded decision
            # (`_restore_selection_checkpoint`); the incumbent's per-fold log-losses are
            # kept as the step record's baseline.
            state = _restore_selection_checkpoint(
                last_results, 'mean_ll', 'se_ll', lambda entry: entry['ll'],
            )
            if state is None:
                print(
                    f"[RESUME] Step {last_step} accepted a feature without scored folds "
                    "(broken checkpoint); ignoring and starting fresh."
                )
                existing_steps = []
            else:
                current_model_features = state['features']
                best_current_score = state['score']
                best_current_folds = state['folds']
                step_counter = state['next_step']
                # A search that had accepted every ranked feature stopped on that, with no
                # rejecting step; it is converged too.
                resume_converged = state['converged'] or len(current_model_features) == len(ranked_features)
                if resume_converged:
                    print(f"[RESUME] Step {last_step}: selection already converged on "
                          f"{current_model_features}; finalizing.")
                else:
                    print(f"[RESUME] Continuing from Step {step_counter} with {current_model_features}.")
        except (OSError, pickle.UnpicklingError, KeyError, EOFError) as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # A fresh start (no checkpoint, or one the resume logic rejected) moves any step
    # files of an earlier run aside before Step 0 is written.
    if not existing_steps:
        _quarantine_stale_step_files(model_selection_dir, prefix)

    # The development refit the per-step shift test scores on: every development
    # session pooled, the majority class subsampled to the minority count with the
    # held-out refit's seed and draw (`_balance_multivariate_arrays` draws the target
    # indices first, then the other indices, from one generator), so the test's models
    # are the models the held-out refit fits. Every feature carries the same rows per
    # session, so one feature's pooled counts and row sessions serve all.
    _dev_seed = random_seed + n_splits + 1
    _dev_pool_cache = {}

    def _category_dev_pool(feature: str) -> tuple:
        """
        Description
        -----------
        The feature's (target, other) windows pooled over the development sessions,
        cached per feature.

        Parameters
        ----------
        feature (str)
            Feature key.

        Returns
        -------
        X_t, X_o (tuple)
            ``(n_target, history_frames)`` and ``(n_other, history_frames)`` arrays.
        """

        if feature not in _dev_pool_cache:
            t_list, o_list = _pool_category_features(all_feature_data, [feature], all_sessions_arr, history_frames)
            _dev_pool_cache[feature] = (t_list[0], o_list[0])
        return _dev_pool_cache[feature]

    _dev_X_t0, _dev_X_o0 = _category_dev_pool(ranked_features[0])
    _dev_limit = min(_dev_X_t0.shape[0], _dev_X_o0.shape[0])
    _dev_rng = np.random.default_rng(_dev_seed)
    _dev_idx_t = _dev_rng.choice(_dev_X_t0.shape[0], _dev_limit, replace=False)
    _dev_idx_o = _dev_rng.choice(_dev_X_o0.shape[0], _dev_limit, replace=False)
    _dev_y = np.concatenate([np.ones(_dev_limit), np.zeros(_dev_limit)])
    _dev_g_t, _dev_g_o = _category_row_sessions(all_feature_data, ranked_features[0], all_sessions_arr)
    _dev_groups = np.concatenate([_dev_g_t[_dev_idx_t], _dev_g_o[_dev_idx_o]])

    def _category_development_score(features: list, winner_override: dict | None = None) -> float:
        """
        Description
        -----------
        In-sample Bernoulli log-likelihood per event (the negative log-loss of the
        per-event probability) of the model with ``features`` refit on the balanced
        development draw with the configured engine, exactly as the held-out refit
        fits it. With ``winner_override`` the LAST feature's windows are taken from
        that per-session dict (a shifted null refit) instead of the stored windows.
        The empty model is the constant p = 0.5, log-likelihood ``-log 2``.

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            ``{session_id: {'target_feature_arr': ..., 'other_feature_arr': ...}}``.

        Returns
        -------
        score (float)
            Mean log-likelihood per event on the training rows.
        """

        if not features:
            return -float(chance_ll)
        X_tr_list = []
        for position, f_name in enumerate(features):
            if winner_override is not None and position == len(features) - 1:
                t_list, o_list = _pool_category_features({f_name: winner_override}, [f_name], all_sessions_arr, history_frames)
                X_t, X_o = t_list[0], o_list[0]
            else:
                X_t, X_o = _category_dev_pool(f_name)
            if X_t.shape[0] != _dev_X_t0.shape[0] or X_o.shape[0] != _dev_X_o0.shape[0]:
                msg = (f"development pool misalignment for {f_name!r}: ({X_t.shape[0]}, {X_o.shape[0]}) rows "
                       f"vs ({_dev_X_t0.shape[0]}, {_dev_X_o0.shape[0]}) of the anchor.")
                raise ValueError(msg)
            X_tr_list.append(np.concatenate([X_t[_dev_idx_t], X_o[_dev_idx_o]], axis=0))
        if model_type == 'sklearn':
            X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_tr_list])
            model = LogisticRegressionCV(
                penalty=lr_params['penalty'],
                Cs=lr_params['cs'],
                cv=classification_inner_cv_splits(_dev_y, _dev_groups, lr_params['cv'], random_seed),
                class_weight='balanced',
                solver=lr_params['solver'],
                max_iter=lr_params['max_iter'],
                random_state=random_seed
            ).fit(X_tr_stacked, _dev_y)
            proba = model.predict_proba(X_tr_stacked)[:, 1]
        else:
            X_gam_tr = get_unrolled_X_for_multivariate(X_tr_list, history_frames)
            y_gam_tr = np.repeat(_dev_y.astype(float), history_frames)
            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(features)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
            gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_gam_tr, y_gam_tr)
            proba = np.mean(gam.predict_proba(X_gam_tr).reshape(len(_dev_y), history_frames), axis=1)
        return -float(log_loss(_dev_y.astype(int), np.clip(proba, 1e-15, 1 - 1e-15)))

    # Read-only data every pool worker of this run inherits (`_category_candidate_fold`);
    # the step-specific entries are added per step before the pool forks.
    _category_pool_base = {
        'cv_folds': cv_folds,
        'all_feature_data': all_feature_data,
        'all_sessions_arr': all_sessions_arr,
        'split_strategy': split_strategy,
        'pooled_category_cache': pooled_category_cache,
        'n_targ_total': n_targ_total,
        'g_all_targ': g_all_targ if split_strategy == 'mixed' else None,
        'g_all_other': g_all_other if split_strategy == 'mixed' else None,
        'history_frames': history_frames,
        'model_type': model_type,
        'basis_matrix': basis_matrix,
        'lr_params': lr_params,
        'n_splines_value': n_splines_value,
        'n_splines_time': n_splines_time,
        'gam_kwargs': gam_kwargs,
        'random_seed': random_seed,
        'decision_threshold': settings['diagnostics']['binary_decision_threshold'],
        'ece_n_bins': settings['diagnostics']['ece_n_bins'],
    }

    def _category_fold_base(incumbent: list) -> list:
        """
        Description
        -----------
        'session' strategy: the incumbent features pooled once per fold (identical
        across the candidates of a step), so each worker pools only its candidate.
        `_pool_category_features` pools each feature independently in list order, so
        pool(base + [new]) is byte-identical to pool(base) + pool([new]). Empty lists
        when the incumbent is empty; an empty list for the 'mixed' strategy.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.

        Returns
        -------
        fold_base_pooled (list)
            One ``(b_tr_t, b_tr_o, b_te_t, b_te_o)`` per fold.
        """

        fold_base_pooled = []
        if split_strategy == 'session':
            for tr_idx, te_idx in cv_folds:
                if incumbent:
                    b_tr_t, b_tr_o = _pool_category_features(all_feature_data, incumbent, all_sessions_arr[tr_idx], history_frames)
                    b_te_t, b_te_o = _pool_category_features(all_feature_data, incumbent, all_sessions_arr[te_idx], history_frames)
                else:
                    b_tr_t, b_tr_o, b_te_t, b_te_o = [], [], [], []
                fold_base_pooled.append((b_tr_t, b_tr_o, b_te_t, b_te_o))
        return fold_base_pooled

    def _category_fit_candidates(incumbent: list, candidate_features: list) -> dict:
        """
        Description
        -----------
        Fits every candidate on every fold on top of ``incumbent`` through the
        selection pool and returns each candidate's per-fold metric lists.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.
        candidate_features (list)
            Candidates to score.

        Returns
        -------
        per_candidate (dict)
            ``{feature: {metric key: [per-fold values]}}``.
        """

        set_pool_context(
            category_step={**_category_pool_base,
                           'current_model_features': list(incumbent),
                           'fold_base_pooled': _category_fold_base(incumbent)},
            candidate_fold=_category_candidate_fold,
        )
        n_workers = pool_worker_count(step_cfg['pool_workers'], len(incumbent) + 1,
                                      step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                      step_cfg['fit_peak_gb_per_term'])
        n_folds = len(cv_folds)
        tasks = [(feat, fold_i) for feat in candidate_features for fold_i in range(n_folds)]
        pool = SelectionPool(n_workers)
        try:
            results = pool.map('candidate_fold', tasks)
        finally:
            pool.close()
        return {feat: _assemble_fold_metrics(results[i * n_folds:(i + 1) * n_folds], feat, _BINARY_FOLD_METRIC_KEYS)
                for i, feat in enumerate(candidate_features)}

    def _category_step_test(winner: str, step_index: int, m_candidates: int) -> dict:
        """
        Description
        -----------
        The per-step shift test of ``winner`` on top of the current incumbent.

        Parameters
        ----------
        winner (str)
            The cross-validated winner.
        step_index (int)
            The step.
        m_candidates (int)
            Candidates the winner was chosen from (the Bonferroni factor).

        Returns
        -------
        step_test (dict)
            The record of ``run_step_shift_test``.
        """

        incumbent = list(current_model_features)
        trial = [*incumbent, winner]
        incumbent_score = _category_development_score(incumbent)
        observed_score = _category_development_score(trial)
        winner_traces = {s: read_feature_trace(traces_path, winner, s)
                         for s in all_sessions_arr if s in all_feature_data[winner]}
        n_workers = pool_worker_count(step_cfg['pool_workers'], len(trial),
                                      step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                      step_cfg['fit_peak_gb_per_term'])
        return run_step_shift_test(
            n_workers=n_workers,
            refit_and_score=lambda override: _category_development_score(trial, override),
            incumbent_features=incumbent,
            incumbent_score=incumbent_score,
            winner=winner,
            observed_score=observed_score,
            statistic_name='bernoulli_log_likelihood_per_event',
            winner_traces=winner_traces,
            window_starts={s: window_starts[s] for s in winner_traces},
            slots=_CATEGORY_WINDOW_SLOTS,
            history_frames=history_frames,
            m_candidates=m_candidates,
            alpha=p_val,
            min_lag_frames=step_cfg['min_lag_frames'],
            random_seed=random_seed,
            step_index=step_index,
            checkpoint_every=step_cfg['checkpoint_every'],
            partial_path=model_selection_dir / f"{prefix}{step_index}_null_partial.pkl",
            label=f"Step {step_index}",
        )

    def _category_candidate_record(metrics: dict, mean_ll: float, se_ll: float) -> dict:
        """
        Description
        -----------
        The per-candidate block of a category step pickle: the per-fold metric lists
        in the order the pipeline has always stored them, plus the fold-mean and SE.

        Parameters
        ----------
        metrics (dict)
            ``{metric key: [per-fold values]}``.
        mean_ll (float)
            Mean held-out log-loss over the finite folds.
        se_ll (float)
            Its standard error.

        Returns
        -------
        record (dict)
            The candidate block.
        """

        return {
            'll': metrics['ll'], 'auc': metrics['auc'],
            'f1': metrics['f1'], 'recall': metrics['recall'],
            'score': metrics['score'],
            'brier': metrics['brier'], 'ece': metrics['ece'], 'mcc': metrics['mcc'],
            'confusion_matrix': metrics['confusion_matrix'],
            'n_iter': metrics['n_iter'], 'converged': metrics['converged'],
            'fit_time': metrics['fit_time'],
            'mean_ll': mean_ll, 'se_ll': se_ll
        }

    if use_top_rank_as_anchor and step_counter == 0 and not resume_converged:
        anchor = ranked_features[0]
        print(format_selection_step('Anchor', feature=anchor,
                                    detail='top-ranked; forced as the first candidate, tested against the constant model'))
        metrics = _category_fit_candidates([], [anchor])[anchor]
        valid = [x for x in metrics['ll'] if np.isfinite(x)]
        _require_scored_candidate('Anchor', 1, len(valid))
        step_test = None
        if valid:
            # The forced anchor is tested like every winner, with Bonferroni over every
            # screened feature it was forced ahead of.
            step_test = _category_step_test(anchor, 0, len(ranked_features))
        anchor_accepted = step_test is not None and step_test['accepted']
        mean_ll = float(np.mean(valid)) if valid else float('nan')
        se_ll = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
        step_results = {
            'step_idx': 0,
            'current_features': [anchor] if anchor_accepted else [],
            'baseline_score': chance_ll, 'baseline_folds': chance_ll,
            'selected_feature': anchor if anchor_accepted else None,
            'candidates_summary': {anchor: _category_candidate_record(metrics, mean_ll, se_ll)} if valid else {},
            'step_test': step_test,
        }
        _dump_step_pickle(model_selection_dir / f"{prefix}0.pkl", _wrap_step(step_results))
        if anchor_accepted:
            print(format_selection_step('Anchor', feature=anchor, decision='ACCEPT', metrics={'LL': mean_ll}))
            current_model_features = [anchor]
            best_current_score, best_current_folds = mean_ll, metrics['ll']
            step_counter = 1
        else:
            print(format_selection_step('Anchor', feature=anchor, decision='REJECT',
                                        detail='the forced anchor failed the shift test; selection finished'))
            resume_converged = True

    print("\n--- Starting Forward Selection ---")
    # A resume that restored a converged search skips the loop and finalizes directly.
    while not resume_converged:
        print(format_selection_step(f"Step {step_counter}",
                                    detail=f"best so far LL={best_current_score:.4f}"))
        step_results_metadata = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'baseline_folds': best_current_folds,
            'candidates_summary': {},
            'selected_feature': None,
            'step_test': None,
        }

        candidates_this_step = [feat for feat in ranked_features if feat not in current_model_features]
        m_candidates = len(candidates_this_step)
        per_candidate = _category_fit_candidates(current_model_features, candidates_this_step)

        best_cand_name, best_cand_score = None, float('inf')
        best_cand_folds = None
        for feat in candidates_this_step:
            metrics = per_candidate[feat]
            valid = [x for x in metrics['ll'] if np.isfinite(x)]
            if not valid:
                print(format_selection_step(f"Step {step_counter}", feature=feat,
                                            decision='FAILED (no finite folds)'))
                continue

            mean_ll = np.mean(valid)
            # `np.std([x], ddof=1)` is NaN for a single finite fold (zero divisor);
            # guard with len > 1 so a one-fold success keeps a finite SE.
            se_ll = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            print(format_selection_step(f"Step {step_counter}", feature=feat,
                                        metrics={'LL': mean_ll,
                                                 'AUC': float(np.nanmean(metrics['auc']))}))

            step_results_metadata['candidates_summary'][feat] = _category_candidate_record(metrics, mean_ll, se_ll)
            if mean_ll < best_cand_score:
                best_cand_score, best_cand_name = mean_ll, feat
                best_cand_folds = metrics['ll']

        # The cross-validated winner is tested with the per-step shift test, Bonferroni
        # over the m candidates of this step; the step is accepted only when no shifted
        # refit reaches the observed in-sample gain.
        _require_scored_candidate(f"Step {step_counter}", m_candidates, int(best_cand_name is not None))
        step_test = (_category_step_test(best_cand_name, step_counter, m_candidates)
                     if best_cand_name is not None else None)
        step_results_metadata['step_test'] = step_test

        if step_test is not None and step_test['accepted']:
            print(format_selection_step(f"Step {step_counter}", feature=best_cand_name,
                                        decision='ACCEPT',
                                        metrics={'LL': best_cand_score,
                                                 'p_adj': step_test['p_adjusted']}))
            current_model_features.append(best_cand_name)
            step_results_metadata['selected_feature'] = best_cand_name
            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results_metadata))
            best_current_score, step_counter = best_cand_score, step_counter + 1
            best_current_folds = best_cand_folds
            if len(current_model_features) == len(ranked_features):
                break
        else:
            if best_cand_name is None:
                reject_detail = 'no candidate had finite folds; selection finished'
            else:
                reject_detail = (f"winner {best_cand_name!r} failed the shift test "
                                 f"(p_adj {step_test['p_adjusted']:.4f} vs alpha {p_val}); selection finished")
            print(format_selection_step(f"Step {step_counter}", decision='REJECT', detail=reject_detail))
            step_results_metadata['selected_feature'] = None
            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results_metadata))
            break

    print("\n--- Final Refit for Visualization (CV-based) ---")
    # No blanket except: a failed final refit or save must stop the run loudly, not
    # pass as a finished selection with no final model.
    last_file = model_selection_dir / f"{prefix}{step_counter}.pkl"
    if not last_file.exists():
        last_file = model_selection_dir / f"{prefix}{step_counter - 1}.pkl"

    if not current_model_features:
        _write_empty_selection(last_file, _wrap_step, univariate_results_path, input_data_path, 'category')
        return

    final_fold_shapes = []
    print(f"  Calculating filter shapes across {len(cv_folds)} folds...")

    for fold_idx, (tr_idx, _te_idx) in enumerate(cv_folds):

        if split_strategy == 'session':
            X_tr_t_raw, X_tr_o_raw = _pool_category_features(all_feature_data, current_model_features, all_sessions_arr[tr_idx], history_frames)
            g_tr_t_raw, g_tr_o_raw = _category_row_sessions(all_feature_data, current_model_features[0], all_sessions_arr[tr_idx])
        else:
            # Reuse the pre-pooled target / other arrays for the final
            # refit so the visualisation loop doesn't re-pool the full
            # dataset per fold.
            X_all_t_raw = [pooled_category_cache[f]['target'] for f in current_model_features]
            X_all_o_raw = [pooled_category_cache[f]['other'] for f in current_model_features]

            tr_targ_idx = tr_idx[tr_idx < n_targ_total]
            tr_other_idx = tr_idx[tr_idx >= n_targ_total] - n_targ_total

            X_tr_t_raw = [x[tr_targ_idx] for x in X_all_t_raw]
            X_tr_o_raw = [x[tr_other_idx] for x in X_all_o_raw]
            g_tr_t_raw, g_tr_o_raw = g_all_targ[tr_targ_idx], g_all_other[tr_other_idx]

        X_tr_t, X_tr_o, y_tr_t, y_tr_o, g_tr_t, g_tr_o = _balance_multivariate_arrays(
            X_tr_t_raw, X_tr_o_raw, random_seed + fold_idx, groups_target=g_tr_t_raw, groups_other=g_tr_o_raw)

        if not X_tr_t:
            continue

        X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o, strict=False)]
        y_tr = np.concatenate([y_tr_t, y_tr_o])
        groups_tr = np.concatenate([g_tr_t, g_tr_o])

        if model_type == 'sklearn':
            X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_tr_list])

            model = LogisticRegressionCV(
                penalty=lr_params['penalty'],
                Cs=lr_params['cs'],
                cv=classification_inner_cv_splits(y_tr, groups_tr, lr_params['cv'], random_seed),
                class_weight='balanced',
                solver=lr_params['solver'],
                max_iter=lr_params['max_iter'],
                random_state=random_seed
            )
            model.fit(X_tr_stacked, y_tr)

            coefs = model.coef_.flatten()
            n_bases = basis_matrix.shape[1]

            fold_res = {}
            for k, f_name in enumerate(current_model_features):
                feat_coefs = coefs[k * n_bases: (k + 1) * n_bases]
                fold_res[f_name] = np.dot(feat_coefs, basis_matrix.T).flatten()

            final_fold_shapes.append(fold_res)

        else:
            X_gam_tr = get_unrolled_X_for_multivariate(X_tr_list, history_frames)
            y_gam_tr = np.repeat(y_tr.astype(float), history_frames)

            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(current_model_features)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

            gam_fold = LogisticGAM(gam_terms, **gam_kwargs).fit(X_gam_tr, y_gam_tr)

            time_idx = np.arange(history_frames)
            base_grid = np.zeros((history_frames, 2 * len(current_model_features)))
            for k in range(len(current_model_features)):
                base_grid[:, k * 2 + 1] = time_idx

            base_prob = gam_fold.predict_mu(base_grid)

            fold_res = {}
            for k, f_name in enumerate(current_model_features):
                test_grid = base_grid.copy()
                test_grid[:, k * 2] = 1.0
                fold_res[f_name] = (gam_fold.predict_mu(test_grid) - base_prob).flatten()

            final_fold_shapes.append(fold_res)

            del gam_fold, X_gam_tr, y_gam_tr

        gc.collect()

    # Honest held-out score (category). Refit the accepted feature set on ALL
    # development sessions (balanced target-vs-other training pool) and score
    # it once on the reserved held-out sessions at their natural class prior
    # (LL), mirroring the finalization fold fit for the active engine, plus a
    # shuffled-label null control for the held-out chance floor. This is the
    # only number untouched by screening, greedy acceptance, or the per-fold
    # balancing. Skipped when holdout is disabled (`held_sessions_arr` empty),
    # no feature was accepted, or the pooled data is degenerate -- leaving
    # `heldout` None so no-holdout runs are byte-identical to before.
    heldout_result = None
    if held_sessions_arr.size > 0 and len(current_model_features) > 0:
        dev_t_raw, dev_o_raw = _pool_category_features(
            all_feature_data, current_model_features, all_sessions_arr, history_frames
        )
        held_t_raw, held_o_raw = _pool_category_features(
            all_feature_data, current_model_features, held_sessions_arr, history_frames
        )
        # Distinct-but-deterministic seed for the balanced development draw and
        # the null permutation; offset the fold seeds by `n_splits`.
        held_seed = random_seed + n_splits + 1
        dev_g_t, dev_g_o = _category_row_sessions(all_feature_data, current_model_features[0], all_sessions_arr)
        X_tr_t, X_tr_o, y_tr_t, y_tr_o, g_tr_t, g_tr_o = _balance_multivariate_arrays(
            dev_t_raw, dev_o_raw, held_seed, groups_target=dev_g_t, groups_other=dev_g_o)
        n_held_targ = held_t_raw[0].shape[0] if held_t_raw else 0
        n_held_other = held_o_raw[0].shape[0] if held_o_raw else 0
        heldout_result = {
            'features': list(current_model_features),
            'n_held_out_sessions': int(held_sessions_arr.size),
            'n_held_out_events': int(n_held_targ + n_held_other),
        }
        if X_tr_t and n_held_targ > 0 and n_held_other > 0:
            X_tr_list = [np.concatenate([t, o], axis=0) for t, o in zip(X_tr_t, X_tr_o, strict=False)]
            y_tr_bal = np.concatenate([y_tr_t, y_tr_o])
            groups_dev = np.concatenate([g_tr_t, g_tr_o])
            X_te_list = [np.concatenate([t, o], axis=0) for t, o in zip(held_t_raw, held_o_raw, strict=False)]
            y_test_held = np.concatenate([np.ones(n_held_targ), np.zeros(n_held_other)])
            y_te_int = y_test_held.astype(int)
            held_rng = np.random.default_rng(held_seed)
            for branch in ('actual', 'null'):
                # The held-out null shuffles labels across the development set.
                y_tr = (y_tr_bal if branch == 'actual'
                        else y_tr_bal[null_target_permutation(y_tr_bal.shape[0], held_rng)])
                rec = {'ll': np.nan, 'auc': np.nan, 'score': np.nan,
                       'deviance_explained': np.nan, 'brier': np.nan,
                       'heldout_fit_failed': False, 'heldout_fit_error': None}
                try:
                    if model_type == 'sklearn':
                        X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_tr_list])
                        X_te_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_te_list])
                        model = LogisticRegressionCV(
                            penalty=lr_params['penalty'],
                            Cs=lr_params['cs'],
                            cv=classification_inner_cv_splits(y_tr, groups_dev, lr_params['cv'], random_seed),
                            class_weight='balanced',
                            solver=lr_params['solver'],
                            max_iter=lr_params['max_iter'],
                            random_state=random_seed
                        ).fit(X_tr_stacked, y_tr)
                        y_prob = model.predict_proba(X_te_stacked)[:, 1]
                        y_pred = model.predict(X_te_stacked)
                    else:
                        X_gam_tr = get_unrolled_X_for_multivariate(X_tr_list, history_frames)
                        y_gam_tr = np.repeat(y_tr.astype(float), history_frames)
                        gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
                        for i in range(1, len(current_model_features)):
                            gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
                        gam = LogisticGAM(gam_terms, **gam_kwargs).fit(X_gam_tr, y_gam_tr)
                        X_gam_te = get_unrolled_X_for_multivariate(X_te_list, history_frames)
                        y_prob_frame = gam.predict_proba(X_gam_te)
                        y_prob = np.mean(y_prob_frame.reshape(len(y_test_held), history_frames), axis=1)
                        y_pred = (y_prob >= settings['diagnostics']['binary_decision_threshold']).astype(int)
                    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                    rec['ll'] = log_loss(y_te_int, y_prob_clipped)
                    rec['deviance_explained'] = 1.0 - rec['ll'] / np.log(2)
                    rec['score'] = balanced_accuracy_score(y_te_int, y_pred)
                    rec['brier'] = float(brier_score_loss(y_te_int, y_prob))
                    if len(np.unique(y_te_int)) > 1:
                        rec['auc'] = roc_auc_score(y_te_int, y_prob)
                except Exception as e:
                    # The held-out number is touched once; a failed refit is recorded as such
                    # (flag + error text beside the NaN metrics) and warned about, never left
                    # as a silent NaN.
                    rec['heldout_fit_failed'] = True
                    rec['heldout_fit_error'] = f"{type(e).__name__}: {e}"
                    warnings.warn(f"Held-out refit failed (category {branch}): {rec['heldout_fit_error']}",
                                  RuntimeWarning, stacklevel=2)
                heldout_result[branch] = rec

    final_res = {
        'final_model_features': current_model_features,
        'filter_shapes': final_fold_shapes,
        'heldout': heldout_result,
        'univariate_results_path': univariate_results_path,
        'input_data_path': input_data_path
    }

    with Path(last_file).open('rb') as f:
        data = pickle.load(f)

    for _k in RESERVED_METADATA_KEYS:
        data.pop(_k, None)
    data.update(final_res)

    _dump_step_pickle(last_file, _wrap_step(data))

    metrics_by_strategy = {'Final': {'LL': best_current_score},
                           'Chance': {'LL': chance_ll}}
    # Held-out LL (development-refit, scored once) -- the honest number
    # alongside the optimistic development-CV `Final`.
    if heldout_result is not None and 'actual' in heldout_result:
        metrics_by_strategy['Held-out'] = {'LL': heldout_result['actual']['ll']}
    print(format_run_summary(
        label=f"category model-selection ({len(current_model_features)} features)",
        metrics_by_strategy=metrics_by_strategy,
        out_path=str(last_file),
    ))



#: Per-fold metric keys every bout-parameter candidate fit records (aligned with the
#: univariate runner), and the placeholders of a failed fold that are not NaN.
_PARAMS_FOLD_METRIC_KEYS = ('explained_deviance', 'residual_deviance', 'spearman_r', 'pearson_r',
                            'msle', 'mae', 'rmse', 'y_true', 'y_pred', 'n_iter', 'converged', 'fit_time')
_PARAMS_FOLD_PLACEHOLDERS = {
    'y_true': lambda: np.empty((0,), dtype=np.float64),
    'y_pred': lambda: np.empty((0,), dtype=np.float64),
    'converged': lambda: False,
}

#: Start-frame slot -> pickle array key of the bout-parameter pipeline (one global row
#: array per feature; the starts of a session are those of its rows in row order).
_PARAMS_WINDOW_SLOTS = {'X': 'X'}


def _verify_global_windows_against_traces(feature_data: dict,
                                          traces_path: str | Path,
                                          window_starts: dict,
                                          history_frames: int,
                                          features: list,
                                          sessions: list) -> None:
    """
    Description
    -----------
    The lag-zero self-check of the per-step shift test for the global-row layout of
    the bout-parameter pipeline (``feature_data[feature]['X']`` holds every session's
    windows stacked, ``['groups']`` the session of every row): re-cuts every session's
    rows from the trace sidecar at the stored start frames and requires bit-exact
    equality, and the same number of rows as stored starts. A mismatch means the
    pickle, its start frames and the sidecar do not describe one extraction, and the
    run aborts before any model is fit.

    Parameters
    ----------
    feature_data (dict)
        The modeling pickle's feature dict (metadata blocks stripped).
    traces_path (str | Path)
        The HDF5 sidecar.
    window_starts (dict)
        ``{session: {'X': (n,) start frames, ...}}`` from ``_input_metadata``.
    history_frames (int)
        Window length in frames.
    features (list)
        Features to check.
    sessions (list)
        Sessions to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        On the first session whose rows do not reproduce.
    """

    with h5py.File(Path(traces_path), 'r') as handle:
        for feature in features:
            X_all = feature_data[feature]['X']
            groups = feature_data[feature]['groups']
            for session_id in sessions:
                rows = np.where(groups == session_id)[0]
                starts = np.asarray(window_starts[session_id]['X'], dtype=int)
                if rows.size != starts.size:
                    msg = (f"_verify_global_windows_against_traces: {feature!r} / {session_id!r}: "
                           f"{rows.size} stored rows but {starts.size} start frames.")
                    raise ValueError(msg)
                if rows.size == 0:
                    continue
                column = handle[feature][str(session_id)][()]
                rebuilt = cut_history_windows(column, starts, history_frames)
                stored = X_all[rows]
                if rebuilt.shape != stored.shape or not np.array_equal(rebuilt, stored):
                    msg = (f"_verify_global_windows_against_traces: {feature!r} / {session_id!r}: rows re-cut "
                           f"from the trace sidecar at lag 0 do not match the stored rows (shapes "
                           f"{rebuilt.shape} vs {stored.shape}). The pickle, its start frames and "
                           f"{traces_path} are not from one extraction.")
                    raise ValueError(msg)


def _params_candidate_fold(feat: str, fold_idx: int) -> dict:
    """
    Description
    -----------
    One (candidate, fold) fit of the bout-parameter selector, run in a pool worker:
    stacks the incumbent features' fold rows (pre-sliced per fold by the selector in
    ``fold_base_X_tr`` / ``fold_base_X_te``, and for the sklearn engine pre-projected
    in ``fold_base_proj_tr`` / ``fold_base_proj_te``) with the candidate's, fits the
    configured engine (``GammaRegressor`` under an inner-CV alpha search on the basis
    projection, or the Gamma ``GAM`` on the unrolled tensor terms) and scores the
    test rows with the univariate runner's metric set. The data comes from the
    ``'params_step'`` pool-context entry the selector stores before creating the
    pool. The operations are those of the former inline loop.

    Parameters
    ----------
    feat (str)
        The candidate feature.
    fold_idx (int)
        Fold index.

    Returns
    -------
    result (dict)
        ``{'ok': True, <metric key>: value, ...}`` with the keys of
        ``_PARAMS_FOLD_METRIC_KEYS``, or ``{'ok': False, 'error': str}``.
    """

    ctx = pool_context('params_step')
    tr_idx, te_idx = ctx['cv_folds'][fold_idx]
    all_feature_data = ctx['all_feature_data']
    y_global = ctx['y_global']
    groups_global = ctx['groups_global']
    history_frames = ctx['history_frames']
    model_type = ctx['model_type']
    basis_matrix = ctx['basis_matrix']
    lr_params = ctx['lr_params']
    gam_kwargs = ctx['gam_kwargs']
    current_model_features = ctx['current_model_features']
    try:
        trial_tr = ctx['fold_base_X_tr'][fold_idx] + [all_feature_data[feat]['X'][tr_idx]]
        trial_te = ctx['fold_base_X_te'][fold_idx] + [all_feature_data[feat]['X'][te_idx]]
        y_tr, y_te = y_global[tr_idx], y_global[te_idx]

        if model_type == 'sklearn':
            # Project only the new candidate feature; the fixed base projection was
            # precomputed once per fold by the selector and hstacked on.
            new_proj_tr = np.dot(all_feature_data[feat]['X'][tr_idx], basis_matrix)
            new_proj_te = np.dot(all_feature_data[feat]['X'][te_idx], basis_matrix)
            base_proj_tr = ctx['fold_base_proj_tr'][fold_idx]
            base_proj_te = ctx['fold_base_proj_te'][fold_idx]
            X_tr_stacked = np.hstack([base_proj_tr, new_proj_tr]) if base_proj_tr is not None else new_proj_tr
            X_te_stacked = np.hstack([base_proj_te, new_proj_te]) if base_proj_te is not None else new_proj_te

            # Gamma GLM (log link), matching the anchor fit and the pyGAM branch:
            # fit and Gamma-deviance score share one likelihood.
            y_tr_pos = np.maximum(y_tr, 1e-6)
            fit_start = time.perf_counter()
            model = GridSearchCV(
                GammaRegressor(),
                {'alpha': list(lr_params['alphas'])},
                cv=session_grouped_inner_cv(groups_global[tr_idx], lr_params['cv']),
                scoring='neg_mean_gamma_deviance',
            ).fit(X_tr_stacked, y_tr_pos, groups=groups_global[tr_idx])
            fit_time = float(time.perf_counter() - fit_start)

            best = model.best_estimator_
            y_pred = best.predict(X_te_stacked)
            fold_n_iter = int(np.max(np.atleast_1d(best.n_iter_)))
            fold_converged = bool(fold_n_iter < best.max_iter)
        else:
            gam_terms = te(0, 1, n_splines=[ctx['n_splines_value'], ctx['n_splines_time']])
            for i in range(1, len(current_model_features) + 1):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[ctx['n_splines_value'], ctx['n_splines_time']])
            X_tr_gam = get_unrolled_X_for_multivariate(trial_tr, history_frames)
            X_te_gam = get_unrolled_X_for_multivariate(trial_te, history_frames)

            fit_start = time.perf_counter()
            gam = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_tr_gam, np.repeat(y_tr + 1e-6, history_frames))
            fit_time = float(time.perf_counter() - fit_start)

            # Aggregate the H per-frame predictions on the linear-predictor scale.
            eta_te = np.log(gam.predict_mu(X_te_gam)).reshape(len(y_te), history_frames)
            y_pred = np.exp(np.mean(eta_te, axis=1))
            gam_diffs = gam.logs_['diffs']
            fold_n_iter = float(len(gam_diffs))
            fold_converged = bool(len(gam_diffs) > 0 and len(gam_diffs) < gam_kwargs['max_iter'])

        y_te_safe = np.maximum(y_te, 1e-6)
        y_pred_safe = np.maximum(y_pred, 1e-6)
        y_null_pred = np.full_like(y_te_safe, np.mean(y_te_safe))

        res_dev = mean_gamma_deviance(y_te_safe, y_pred_safe)
        null_dev = mean_gamma_deviance(y_te_safe, y_null_pred)

        d2 = 0.0 if null_dev == 0 else 1 - (res_dev / null_dev)

        result = {'ok': True}
        result['residual_deviance'] = res_dev
        result['explained_deviance'] = d2
        result['spearman_r'] = spearmanr(y_te, y_pred)[0]
        result['pearson_r'] = pearson_r_safe(y_te, y_pred)
        result['msle'] = mean_squared_log_error(y_te_safe, y_pred_safe)
        result['mae'] = mean_absolute_error_1d(y_te, y_pred)
        result['rmse'] = root_mean_squared_error(y_te, y_pred)
        result['y_true'] = y_te_safe
        result['y_pred'] = y_pred_safe
        result['n_iter'] = fold_n_iter
        result['converged'] = fold_converged
        result['fit_time'] = fit_time
        gc.collect()
        return result
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


def bout_parameter_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str | None = None,
        target_variable: str = 'bout_durations',
        use_top_rank_as_anchor: bool = False,
        *,
        p_val: float
) -> None:
    """
    Performs forward stepwise selection for continuous bout parameter regression.

    Identifies the minimal sufficient set of behavioral and vocal syntax features
    that predict a continuous target (e.g., duration, complexity). The framework
    accommodates strictly positive, right-skewed biological data.

    Engine Flexibility:
    -------------------
    Reads `model_engine` from JSON settings to dynamically construct models:
    - 'sklearn': Utilizes an L2-penalized `GammaRegressor` (Gamma GLM, log link)
                 combined with linear basis projection. It fits under the SAME Gamma
                 likelihood the model is later scored with (`mean_gamma_deviance`), so
                 there is no log-normal-fit / Gamma-score mismatch, and — because the
                 log link targets the conditional mean E[y] directly — no Jensen
                 back-transform correction is needed. The L2 penalty `alpha` is chosen
                 by an inner-CV Gamma-deviance search (`GridSearchCV` over the
                 `ridge_regression.alphas` grid, `cv` scheme from settings).
    - 'pygam': Utilizes `GAM` with a Gamma distribution and Log-link function. High-dimensional
               features are unrolled into tensor product splines (te) to capture non-linear surfaces.
               For each test trial the `H` per-frame predictions produced by the tensor-product
               unroll are aggregated on the linear-predictor (eta = log mu) scale before applying
               the inverse link (i.e. `y_pred = exp(mean(eta))`, not `mean(exp(eta))`), which
               avoids the Jensen-inequality bias that natural-scale averaging would introduce
               whenever the per-frame eta have any spread.

    Known caveat (tile-and-repeat unroll):
    --------------------------------------
    The tensor-product unroll duplicates each trial's scalar target `y_tr` across `H` history
    frames via `np.repeat(y_tr, H)` and fits as if those `N * H` rows were independent
    observations. The smoothness penalty is fixed (`lam_penalty`, never tuned: pyGAM's
    `gridsearch()` is not called), while the data term grows with the `N * H` duplicated rows,
    so the fixed penalty weighs less than it would on `N` truly i.i.d. observations, nudging
    the fit toward under-smoothing. Test-time aggregation is performed per-trial so held-out metrics remain on
    the correct `N` scale, and cross-validation is the practical safeguard against the
    resulting optimism; the bias is shared by both engines' multivariate paths and so does
    not confound the sklearn-vs-pyGAM comparison.

    Splitting Strategies:
    ---------------------
    Continuous target variables are discretized into quantile bins (`y_binned`) to enable stratification.
    - 'session': Uses `StratifiedGroupKFold` (re-seeded per fold) to isolate whole recording days
                 while preserving the target distribution.
    - 'mixed': Uses `StratifiedShuffleSplit` to pool all data and split at the epoch level,
               guaranteeing identical continuous distributions across train and test sets.

    Selection Metric:
    -----------------
    Explained Deviance (D^2) ranks the candidates of a step (cross-validated fold
    mean). The winner is then tested with the per-step shift test
    (``step_shift_test.run_step_shift_test``): the incumbent and the incumbent plus
    the winner are refit on all development rows (the held-out refit's rows), the
    statistic is the in-sample gain in negative mean Gamma deviance per event, its
    null comes from refits after circularly shifting the winner's trace within every
    development session and re-cutting its windows at the stored start frames,
    ``floor(m / alpha)`` times with Bonferroni correction over the ``m`` candidates
    (``alpha = model_params.selection_p_val``). The first step is tested the same way
    against the intercept-only model (the training mean).

    Data Persistence:
    -----------------
    Preserves full-resolution metric data. For every candidate tested at
    every step, the raw per-fold results are saved to match the metric set
    emitted by the univariate runner: Explained Deviance (D^2), Residual
    Gamma Deviance, Spearman's rho, Pearson's r, Mean Squared Logarithmic
    Error (MSLE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    the raw test-fold `y_true` / `y_pred` arrays, and per-fold optimizer
    diagnostics (`n_iter`, `converged`, `fit_time`). `converged=False`
    flags folds that terminated at `max_iter` without meeting the tolerance,
    surfacing the main silent-failure mode of the pyGAM path.

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file.
    input_data_path : str
        Path to the raw extracted feature data .pkl file containing (X, y, groups).
    output_directory : str
        The directory path where the resulting step files (.pkl) will be saved.
    settings_path : str, optional
        Path to the 'modeling_settings.json' configuration file.
    target_variable : str, default 'bout_durations'
        The name of the target variable to pull from the input data.
    use_top_rank_as_anchor : bool, default False
        If True, starts the search with the #1 ranked univariate feature.
    p_val : float
        Keyword-only, no default: the dispatcher passes ``model_params.selection_p_val``.
        The alpha level used for significance screening against shuffled controls.

    Returns
    -------
    None
        Saves step-wise results and final CV-based filter shapes to disk.
    """

    if settings_path is None:
        settings_path_obj = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with Path(settings_path).open() as f:
            settings = json.load(f)
    except FileNotFoundError as err:
        msg = f"Settings file not found at {settings_path}"
        raise FileNotFoundError(msg) from err

    model_selection_dir = Path(output_directory)
    model_selection_dir.mkdir(parents=True, exist_ok=True)

    with Path(univariate_results_path).open('rb') as f:
        univariate_data = pickle.load(f)

    # Strip the upstream metadata blocks from `univariate_data` before
    # iterating its `.items()`. The same pattern is applied to
    # `all_feature_data` further down once it is loaded.
    _univ_pre_md = univariate_data.pop('_run_metadata', None)
    _require_analysis_type(univariate_data, 'params', univariate_results_path)
    _input_pre_md_from_univ = univariate_data.pop('_input_metadata', None)
    univariate_data.pop('_consolidation_metadata', None)

    # 1. Feature Screening & Ranking
    candidates = []
    num_features = len(univariate_data)

    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if 'actual' not in results or 'explained_deviance' not in results['actual']:
            continue
        if 'null' not in results or 'explained_deviance' not in results['null']:
            continue

        actual_deviance = results['actual']['explained_deviance']
        null_deviance = results['null']['explained_deviance']

        valid_actual = actual_deviance[~np.isnan(actual_deviance)]
        valid_null = null_deviance[~np.isnan(null_deviance)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        mean_actual_deviance = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=100 - (p_val / num_features) * 100)

        if mean_actual_deviance > null_threshold and mean_actual_deviance > 0:
            candidates.append({'feature': feat_name, 'mean_explained_deviance': mean_actual_deviance})
        else:
            reason = "Negative D^2" if mean_actual_deviance <= 0 else "Not Significant"
            print(format_selection_step(
                'Screen', feature=feat_name, decision='DROP',
                detail=f"{reason} (Mean Dev {mean_actual_deviance:.4f} vs Null {null_threshold:.4f})"))

    candidates.sort(key=lambda x: x['mean_explained_deviance'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    print(f"Identified {len(ranked_features)} significant candidates. Top: {ranked_features[0]}")

    # 2. Raw Data Preparation
    print("Loading raw input data...")
    all_feature_data = load_pickle_modeling_data(input_data_path)

    # Final upstream-metadata harvest: prefer the value the consolidator
    # asserted against every per-feature univariate pickle, then fall
    # back to the Level-1 pickle's own block for legacy runs.
    _require_analysis_type(all_feature_data, 'params', input_data_path)
    _input_md = _input_pre_md_from_univ
    if _input_md is None and '_input_metadata' in all_feature_data:
        _input_md = all_feature_data.pop('_input_metadata')
    else:
        all_feature_data.pop('_input_metadata', None)
    all_feature_data.pop('_run_metadata', None)
    all_feature_data.pop('_consolidation_metadata', None)
    _univariate_md = _univ_pre_md

    y_global = all_feature_data[ranked_features[0]]['y']
    groups_global = all_feature_data[ranked_features[0]]['groups']

    camera_rate = settings['io']['camera_sampling_rate']
    filter_history_sec = settings['model_params']['filter_history']
    history_frames = int(np.floor(camera_rate * filter_history_sec))

    # Strict Dictionary Lookup
    model_ops = settings['model_params']
    model_type = model_ops['model_engine']
    split_strategy = settings['model_validation']['split_strategy']
    n_splits = settings['model_validation']['n_cv_folds']
    test_prop = settings['model_validation']['cv_validation_proportion']
    random_seed = settings['model_validation']['random_seed']

    # Reserve the held-out sessions carved ONCE at extraction time and shipped in
    # the univariate `_input_metadata.held_out_session_ids`. They are dropped from
    # every greedy cv_fold and every candidate fit; the accepted feature set is
    # then refit on the development sessions and scored on this block once in the
    # finalization stage (the honest, selection-leakage-free number). An empty
    # list -- holdout disabled or a pre-v2 pickle -- makes `_dev_mask` all-True
    # and `_held_positions` empty, so the path is an exact no-op.
    _held_out_session_ids = held_out_session_ids_from_metadata(_input_md if _input_md is not None else {})
    _dev_mask, _held_mask = development_heldout_masks(groups_global, _held_out_session_ids)
    _dev_positions = np.where(_dev_mask)[0]
    _held_positions = np.where(_held_mask)[0]
    if _held_positions.size > 0:
        print(
            f"Held-out reserve: {len(_held_out_session_ids)} session(s), "
            f"{_held_positions.size} event(s) excluded from selection; scored once at finalization."
        )
    _dev_session_ids = [str(s_id) for s_id in np.unique(groups_global[_dev_positions])]

    step_cfg = _step_test_settings(settings, float(camera_rate))
    traces_path, window_starts = _require_shift_test_inputs(_input_md, input_data_path, float(camera_rate))
    # Lag-zero self-check of the shift test's inputs: every stored row of every
    # screened feature must re-cut exactly from the trace sidecar at its stored start.
    _verify_global_windows_against_traces(
        all_feature_data, traces_path, window_starts, history_frames,
        features=ranked_features, sessions=_dev_session_ids,
    )
    print(f"Shift-test inputs verified: {len(ranked_features)} feature(s) x {len(_dev_session_ids)} development "
          f"session(s) re-cut exactly from {Path(traces_path).name}; lag floor "
          f"{step_cfg['min_lag_seconds']} s = {step_cfg['min_lag_frames']} frames.")

    print(format_run_header(
        task='SELECTION:PARAMS',
        engine=model_type,
        feature=target_variable,
        input_files={'univariate results': univariate_results_path,
                     'input data': input_data_path,
                     'settings': settings_path},
        output_directory=str(model_selection_dir),
        split_strategy=split_strategy,
        n_splits=n_splits,
    ))

    # Engine-Specific Setup
    basis_matrix = None
    gam_kwargs = {}
    n_splines_time = 0
    n_splines_value = 0
    lr_params = {}

    if model_type == 'sklearn':
        basis_type = model_ops['model_basis_function']
        if basis_type == 'raised_cosine':
            p = settings['hyperparameters']['basis_functions']['raised_cosine']
            kp = int(np.floor(history_frames * p['kpeaks_proportion']))
            basis_matrix = raised_cosine(neye=p['neye'], ncos=p['ncos'], kpeaks=[0, kp], b=p['b'], w=history_frames)
        elif basis_type == 'bspline':
            p = settings['hyperparameters']['basis_functions']['bspline']
            deg = p['degree']
            max_k = max(0, history_frames - deg)
            knots = np.linspace(0, max_k, p['n_splines'] - deg + 1).astype(int)
            basis_matrix = _normalizecols(bsplines(width=history_frames, positions=knots, degree=deg))
        elif basis_type == 'laplacian_pyramid':
            p = settings['hyperparameters']['basis_functions']['laplacian_pyramid']
            basis_matrix = _normalizecols(laplacian_pyramid(width=history_frames, levels=p['levels'], step=p['step'], fwhm=p['fwhm']))
        elif basis_type == 'identity':
            basis_matrix = identity(width=history_frames)

        lr_params = settings['hyperparameters']['classical']['ridge_regression']

    elif model_type == 'pygam':
        pygam_params = settings['hyperparameters']['classical']['pygam']
        n_splines_time = pygam_params['n_splines_time']
        n_splines_value = pygam_params['n_splines_value']
        gam_kwargs = {
            'max_iter': int(pygam_params['max_iterations']),
            'tol': float(pygam_params['tol_val']),
            'lam': pygam_params['lam_penalty']
        }
    else:
        msg = "model_engine in settings must be either 'sklearn' or 'pygam'."
        raise ValueError(msg)

    # 3. Target Quantile Binning for Stratification
    # At most `model_validation.stratification_max_bins` quantile strata; 2 is the
    # structural floor (one stratum would not stratify anything).
    n_bins = max(2, min(settings['model_validation']['stratification_max_bins'], len(y_global) // n_splits))
    bins = np.unique(np.percentile(y_global, np.linspace(0, 100, n_bins + 1)))
    if len(bins) < 2:
        # Right-skewed targets with many tied values can collapse every
        # percentile edge to a single distinct value, leaving fewer than two
        # bins. The all-zeros fallback below makes StratifiedShuffleSplit /
        # StratifiedGroupKFold silently degrade to unstratified splitting, so
        # surface the loss of stratification in the run log (it is otherwise
        # invisible: the metadata/console still report the configured strategy).
        print("  WARNING: quantile binning collapsed to a single bin; "
              "stratification is disabled and splits degrade to unstratified.")
        y_binned = np.zeros(len(y_global))
    else:
        y_binned = np.digitize(y_global, bins[1:-1])

    # 4. Generate Splits
    #
    # Fold the DEVELOPMENT rows only, then remap the dev-relative indices each
    # splitter returns back into full-array index space via `_dev_positions`, so
    # every fold index stays a valid row of the full `y_global`/`groups_global`/
    # `binned_data` arrays while no held-out row enters a train or validation
    # split. With holdout disabled `_dev_positions` is the identity map, so this
    # is byte-identical to the pre-holdout behaviour.
    cv_folds = []
    n_folds_mc = max(2, int(round(1.0 / test_prop)))  # noqa: RUF046  test_prop may be a NumPy scalar; int() pins the type
    _y_binned_dev = y_binned[_dev_mask]
    _groups_dev = groups_global[_dev_mask]

    if split_strategy == 'session':
        print(f"  > Generating {n_splits} Monte Carlo Session Splits (Test Prop: {test_prop})...")
        for i in range(n_splits):
            current_seed = random_seed + i
            stratified_group_k_fold = StratifiedGroupKFold(n_splits=n_folds_mc, shuffle=True, random_state=current_seed)
            try:
                tr_dev, te_dev = next(stratified_group_k_fold.split(np.zeros(len(_y_binned_dev)), _y_binned_dev, groups=_groups_dev))
                cv_folds.append((_dev_positions[tr_dev], _dev_positions[te_dev]))
            except StopIteration:
                pass
    elif split_strategy == 'mixed':
        print(f"  > Generating {n_splits} Monte Carlo Mixed Splits (Test Prop: {test_prop})...")
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_prop, random_state=random_seed)
        for tr_dev, te_dev in sss.split(np.zeros(len(_y_binned_dev)), _y_binned_dev):
            cv_folds.append((_dev_positions[tr_dev], _dev_positions[te_dev]))
    else:
        msg = "split_strategy in settings must be either 'session' or 'mixed'."
        raise ValueError(msg)

    # Leak guard: no held-out event may appear in any greedy fold (train or test).
    if _held_positions.size > 0:
        _held_set = set(_held_positions.tolist())
        for _tr_idx, _te_idx in cv_folds:
            if _held_set.intersection(_tr_idx.tolist()) or _held_set.intersection(_te_idx.tolist()):
                msg = (
                    "Held-out session events leaked into a bout-parameter selection "
                    "CV fold; the development/held-out split is inconsistent."
                )
                raise AssertionError(msg)

    # Resume Logic
    current_model_features = []
    best_current_score = 0.0
    # Per-fold scores of the incumbent, stored as each step record's
    # `baseline_folds`. Before Step 0 the incumbent is the constant zero-D^2 floor.
    best_current_folds = 0.0
    # True when the checkpoint a resume restores is a step that rejected its winner: the
    # search had already finished, so the anchor and the forward loop are skipped.
    resume_converged = False
    step_counter = 0

    fname = Path(univariate_results_path).name
    cond_match = re.search(r'((?:ephys_)?(?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"

    prefix = f"model_selection_{target_variable}_{target_condition}_{split_strategy}_step_"

    # Selection-level run metadata + step-wrapper, frozen here so every
    # step file carries an identical block.
    _run_md = build_selection_metadata(
        modeling_settings=settings,
        selection_function='bout_parameter_model_selection',
        selection_metric='explained_deviance',
        n_splits_selection=int(n_splits),
        test_proportion=float(test_prop),
        split_strategy=split_strategy,
        random_seed=int(random_seed),
        aic_termination_used=False,
        n_anchor_features=len(ranked_features),
        anchor_feature=ranked_features[0],
        gam_kwargs=gam_kwargs,
        extra_knobs={
            'target_variable': target_variable,
            'use_top_rank_as_anchor': bool(use_top_rank_as_anchor),
            'p_val': float(p_val),
            'model_type': model_type,
            'n_splines_time': int(n_splines_time),
            'n_splines_value': int(n_splines_value),
            'target_condition': target_condition,
            # Held-out provenance (also in the embedded `_input_metadata` sibling);
            # duplicated here so the selection run-metadata is self-describing.
            'held_out_test_proportion': float(settings['model_validation']['held_out_test_proportion']),
            'held_out_session_ids': list(_held_out_session_ids),
            'n_held_out_sessions': len(_held_out_session_ids),
            'n_held_out_events': int(_held_positions.size),
            # The per-step acceptance rule and its knobs (a resume refuses another rule).
            'step_test': {
                'rule': STEP_TEST_RULE,
                'screened_features': list(ranked_features),
                'statistic': 'negative_gamma_deviance_per_event',
                'alpha': float(p_val),
                'min_lag_seconds': step_cfg['min_lag_seconds'],
                'min_lag_frames': step_cfg['min_lag_frames'],
                'checkpoint_every': step_cfg['checkpoint_every'],
                'pool_workers': step_cfg['pool_workers'],
                'refit_rows': 'all_development_rows',
                'traces_path': traces_path,
            },
        },
        settings_path=settings_path,
    )
    _wrap_step = _make_step_wrapper(_input_md, _univariate_md, _run_md)

    existing_steps = []
    if model_selection_dir.is_dir():
        for f_name in (p.name for p in model_selection_dir.iterdir()):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                with contextlib.suppress(ValueError):
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))

    if existing_steps:
        last_step = max(existing_steps)
        print(f"[RESUME] Restoring from Step {last_step}...")
        try:
            with Path(model_selection_dir / f"{prefix}{last_step}.pkl").open('rb') as f:
                last_res = pickle.load(f)
            _require_step_test_rule(last_res, model_selection_dir / f"{prefix}{last_step}.pkl")

            # Rebuild the state right after `last_step` from its recorded decision
            # (`_restore_selection_checkpoint`); the incumbent's per-fold deviances are
            # kept as the step record's baseline.
            state = _restore_selection_checkpoint(
                last_res, 'mean_explained_deviance', 'se_explained_deviance',
                lambda entry: entry['explained_deviance'],
            )
            if state is None:
                print(
                    f"[RESUME] Step {last_step} accepted a feature without scored folds "
                    "(broken checkpoint); ignoring and starting fresh."
                )
                existing_steps = []
            else:
                current_model_features = state['features']
                best_current_score = state['score']
                best_current_folds = state['folds']
                step_counter = state['next_step']
                # A search that had accepted every ranked feature stopped on that, with no
                # rejecting step; it is converged too.
                resume_converged = state['converged'] or len(current_model_features) == len(ranked_features)
                if resume_converged:
                    print(f"[RESUME] Step {last_step}: selection already converged on "
                          f"{current_model_features}; finalizing.")
                else:
                    print(f"[RESUME] Continuing from Step {step_counter} with {current_model_features}.")
        except (OSError, pickle.UnpicklingError, KeyError, EOFError) as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # A fresh start (no checkpoint, or one the resume logic rejected) moves any step
    # files of an earlier run aside before Step 0 is written.
    if not existing_steps:
        _quarantine_stale_step_files(model_selection_dir, prefix)

    # The development refit the per-step shift test scores on: every development row
    # (the held-out refit's rows), with the engine's own fit. The statistic is the
    # negative mean Gamma deviance per event on those rows; the intercept-only model
    # (the empty incumbent) predicts the training mean.
    _y_dev = y_global[_dev_positions]
    _y_dev_safe = np.maximum(_y_dev, 1e-6)
    _groups_dev = groups_global[_dev_positions]

    def _params_development_design(features: list, winner_override: dict | None = None) -> list:
        """
        Description
        -----------
        The development rows of every feature of the model, in feature order; with
        ``winner_override`` the LAST feature's rows are replaced session by session by
        the windows in that per-session dict (a shifted null refit), scattered into a
        copy of the stored development design by the rows' sessions.

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            ``{session_id: {'X': (n_rows_of_session, history_frames) windows}}``.

        Returns
        -------
        design (list)
            One ``(n_dev_rows, history_frames)`` array per feature.
        """

        design = [all_feature_data[f]['X'][_dev_positions] for f in features]
        if winner_override is not None:
            X_winner = design[-1].copy()
            for session_id, entry in winner_override.items():
                rows = np.where(_groups_dev == session_id)[0]
                if rows.size != entry['X'].shape[0]:
                    msg = (f"shift override for {features[-1]!r} / {session_id!r} has {entry['X'].shape[0]} "
                           f"rows but the development design holds {rows.size}.")
                    raise ValueError(msg)
                X_winner[rows] = entry['X']
            design[-1] = X_winner
        return design

    def _params_development_score(features: list, winner_override: dict | None = None) -> float:
        """
        Description
        -----------
        Negative in-sample mean Gamma deviance per event of the model with
        ``features`` refit on all development rows with the configured engine,
        exactly as the held-out refit fits it (per-event prediction on the link
        scale for the GAM). The empty model is the intercept-only model predicting
        the training mean.

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            See ``_params_development_design``.

        Returns
        -------
        score (float)
            ``-mean_gamma_deviance`` on the development rows (higher is better).
        """

        if not features:
            y_null_pred = np.full_like(_y_dev_safe, np.mean(_y_dev_safe))
            return -float(mean_gamma_deviance(_y_dev_safe, y_null_pred))
        design = _params_development_design(features, winner_override)
        if model_type == 'sklearn':
            X_dev_stacked = np.hstack([np.dot(x, basis_matrix) for x in design])
            dev_model = GridSearchCV(
                GammaRegressor(),
                {'alpha': list(lr_params['alphas'])},
                cv=session_grouped_inner_cv(_groups_dev, lr_params['cv']),
                scoring='neg_mean_gamma_deviance',
            ).fit(X_dev_stacked, _y_dev_safe, groups=_groups_dev)
            y_pred = dev_model.best_estimator_.predict(X_dev_stacked)
        else:
            X_dev_gam = get_unrolled_X_for_multivariate(design, history_frames)
            y_dev_gam = np.repeat(_y_dev + 1e-6, history_frames)
            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(features)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
            dev_gam = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_dev_gam, y_dev_gam)
            eta_dev = np.log(dev_gam.predict_mu(X_dev_gam)).reshape(len(_y_dev), history_frames)
            y_pred = np.exp(np.mean(eta_dev, axis=1))
        return -float(mean_gamma_deviance(_y_dev_safe, np.maximum(y_pred, 1e-6)))

    # Read-only data every pool worker of this run inherits (`_params_candidate_fold`);
    # the step-specific entries are added per step before the pool forks.
    _params_pool_base = {
        'cv_folds': cv_folds,
        'all_feature_data': all_feature_data,
        'y_global': y_global,
        'groups_global': groups_global,
        'history_frames': history_frames,
        'model_type': model_type,
        'basis_matrix': basis_matrix,
        'lr_params': lr_params,
        'n_splines_value': n_splines_value,
        'n_splines_time': n_splines_time,
        'gam_kwargs': gam_kwargs,
    }

    def _params_fit_candidates(incumbent: list, candidate_features: list) -> dict:
        """
        Description
        -----------
        Fits every candidate on every fold on top of ``incumbent`` through the
        selection pool and returns each candidate's per-fold metric lists. The
        incumbent's fold rows (and, for the sklearn engine, their basis projection)
        are sliced once per fold here, identically for every candidate of the step.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.
        candidate_features (list)
            Candidates to score.

        Returns
        -------
        per_candidate (dict)
            ``{feature: {metric key: [per-fold values]}}``.
        """

        fold_base_X_tr, fold_base_X_te = [], []
        for tr_idx, te_idx in cv_folds:
            fold_base_X_tr.append([all_feature_data[f]['X'][tr_idx] for f in incumbent])
            fold_base_X_te.append([all_feature_data[f]['X'][te_idx] for f in incumbent])
        # For the sklearn path the basis projection of the already-selected features is
        # identical across every candidate in this step, so project the fixed base once
        # per fold here. `None` marks the empty base (np.hstack of [] would raise).
        fold_base_proj_tr, fold_base_proj_te = [], []
        if model_type == 'sklearn':
            for fold_idx in range(len(cv_folds)):
                base_tr = fold_base_X_tr[fold_idx]
                base_te = fold_base_X_te[fold_idx]
                fold_base_proj_tr.append(np.hstack([np.dot(x, basis_matrix) for x in base_tr]) if base_tr else None)
                fold_base_proj_te.append(np.hstack([np.dot(x, basis_matrix) for x in base_te]) if base_te else None)
        set_pool_context(
            params_step={**_params_pool_base,
                         'current_model_features': list(incumbent),
                         'fold_base_X_tr': fold_base_X_tr, 'fold_base_X_te': fold_base_X_te,
                         'fold_base_proj_tr': fold_base_proj_tr, 'fold_base_proj_te': fold_base_proj_te},
            candidate_fold=_params_candidate_fold,
        )
        n_workers = pool_worker_count(step_cfg['pool_workers'], len(incumbent) + 1,
                                      step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                      step_cfg['fit_peak_gb_per_term'])
        n_folds = len(cv_folds)
        tasks = [(feat, fold_idx) for feat in candidate_features for fold_idx in range(n_folds)]
        pool = SelectionPool(n_workers)
        try:
            results = pool.map('candidate_fold', tasks)
        finally:
            pool.close()
        return {feat: _assemble_fold_metrics(results[i * n_folds:(i + 1) * n_folds], feat,
                                             _PARAMS_FOLD_METRIC_KEYS, _PARAMS_FOLD_PLACEHOLDERS)
                for i, feat in enumerate(candidate_features)}

    def _params_step_test(winner: str, step_index: int, m_candidates: int) -> dict:
        """
        Description
        -----------
        The per-step shift test of ``winner`` on top of the current incumbent.

        Parameters
        ----------
        winner (str)
            The cross-validated winner.
        step_index (int)
            The step.
        m_candidates (int)
            Candidates the winner was chosen from (the Bonferroni factor).

        Returns
        -------
        step_test (dict)
            The record of ``run_step_shift_test``.
        """

        incumbent = list(current_model_features)
        trial = [*incumbent, winner]
        incumbent_score = _params_development_score(incumbent)
        observed_score = _params_development_score(trial)
        winner_traces = {s: read_feature_trace(traces_path, winner, s) for s in _dev_session_ids}
        n_workers = pool_worker_count(step_cfg['pool_workers'], len(trial),
                                      step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                      step_cfg['fit_peak_gb_per_term'])
        return run_step_shift_test(
            n_workers=n_workers,
            refit_and_score=lambda override: _params_development_score(trial, override),
            incumbent_features=incumbent,
            incumbent_score=incumbent_score,
            winner=winner,
            observed_score=observed_score,
            statistic_name='negative_gamma_deviance_per_event',
            winner_traces=winner_traces,
            window_starts={s: window_starts[s] for s in winner_traces},
            slots=_PARAMS_WINDOW_SLOTS,
            history_frames=history_frames,
            m_candidates=m_candidates,
            alpha=p_val,
            min_lag_frames=step_cfg['min_lag_frames'],
            random_seed=random_seed,
            step_index=step_index,
            checkpoint_every=step_cfg['checkpoint_every'],
            partial_path=model_selection_dir / f"{prefix}{step_index}_null_partial.pkl",
            label=f"Step {step_index}",
        )

    # 5. Auto-Anchor Logic
    if use_top_rank_as_anchor and step_counter == 0 and not resume_converged:
        anchor = ranked_features[0]
        print(format_selection_step('Anchor', feature=anchor,
                                    detail='top-ranked; forced as the first candidate, tested against the intercept-only model'))
        metrics = _params_fit_candidates([], [anchor])[anchor]
        valid_dev = [m for m in metrics['explained_deviance'] if np.isfinite(m)]
        _require_scored_candidate('Anchor', 1, len(valid_dev))
        step_test = None
        if valid_dev:
            # The forced anchor is tested like every winner, with Bonferroni over every
            # screened feature it was forced ahead of.
            step_test = _params_step_test(anchor, 0, len(ranked_features))
        anchor_accepted = step_test is not None and step_test['accepted']
        mean_anchor_score = float(np.mean(valid_dev)) if valid_dev else float('nan')
        se_anchor_score = float(np.std(valid_dev, ddof=1) / np.sqrt(len(valid_dev))) if len(valid_dev) > 1 else 0.0
        step_0_res = {
            'step_idx': 0,
            'current_features': [anchor] if anchor_accepted else [],
            'baseline_score': 0.0, 'baseline_folds': 0.0,
            'selected_feature': anchor if anchor_accepted else None,
            'candidates_summary': {
                anchor: {
                    **{k: metrics[k] for k in _PARAMS_FOLD_METRIC_KEYS},
                    'mean_explained_deviance': mean_anchor_score,
                    'se_explained_deviance': se_anchor_score
                }
            } if valid_dev else {},
            'step_test': step_test,
        }
        _dump_step_pickle(model_selection_dir / f"{prefix}0.pkl", _wrap_step(step_0_res))
        if anchor_accepted:
            print(format_selection_step('Anchor', feature=anchor, decision='ACCEPT',
                                        metrics={'D^2': mean_anchor_score}))
            current_model_features = [anchor]
            best_current_score, best_current_folds = mean_anchor_score, metrics['explained_deviance']
            step_counter = 1
        else:
            print(format_selection_step('Anchor', feature=anchor, decision='REJECT',
                                        detail='the forced anchor failed the shift test; selection finished'))
            resume_converged = True

    # 6. Forward Selection Loop
    print("\n--- Starting Forward Selection ---")
    # A resume that restored a converged search skips the loop and finalizes directly.
    while not resume_converged:
        step_results = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'baseline_folds': best_current_folds,
            'candidates_summary': {},
            'selected_feature': None,
            'step_test': None,
        }

        candidates_this_step = [feat for feat in ranked_features if feat not in current_model_features]
        m_candidates = len(candidates_this_step)
        per_candidate = _params_fit_candidates(current_model_features, candidates_this_step)

        best_cand, best_cand_score = None, -float('inf')
        best_cand_folds = None
        for feat in candidates_this_step:
            metrics = per_candidate[feat]
            valid = [m for m in metrics['explained_deviance'] if np.isfinite(m)]
            if not valid:
                print(format_selection_step(f"Step {step_counter}", feature=feat,
                                            decision='FAILED (no finite folds)'))
                continue

            # `np.std([x], ddof=1)` is NaN for a single finite fold (zero divisor);
            # guard with len > 1 so a one-fold success keeps a finite SE.
            m_dev = np.mean(valid)
            s_dev = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            print(format_selection_step(f"Step {step_counter}", feature=feat,
                                        metrics={'D^2': m_dev}))

            step_results['candidates_summary'][feat] = {
                **{k: metrics[k] for k in _PARAMS_FOLD_METRIC_KEYS},
                'mean_explained_deviance': m_dev,
                'se_explained_deviance': s_dev
            }

            if m_dev > best_cand_score:
                best_cand_score, best_cand = m_dev, feat
                best_cand_folds = metrics['explained_deviance']

        # The cross-validated winner is tested with the per-step shift test, Bonferroni
        # over the m candidates of this step; the step is accepted only when no shifted
        # refit reaches the observed in-sample gain.
        _require_scored_candidate(f"Step {step_counter}", m_candidates, int(best_cand is not None))
        step_test = (_params_step_test(best_cand, step_counter, m_candidates)
                     if best_cand is not None else None)
        step_results['step_test'] = step_test

        if step_test is not None and step_test['accepted']:
            print(format_selection_step(f"Step {step_counter}", feature=best_cand,
                                        decision='ACCEPT',
                                        metrics={'D^2': best_cand_score,
                                                 'p_adj': step_test['p_adjusted']}))
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)

            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results))

            best_current_score = best_cand_score
            best_current_folds = best_cand_folds
            step_counter += 1
        else:
            if best_cand is None:
                reject_detail = 'no candidate had finite folds; selection finished'
            else:
                reject_detail = (f"winner {best_cand!r} failed the shift test "
                                 f"(p_adj {step_test['p_adjusted']:.4f} vs alpha {p_val}); selection finished")
            print(format_selection_step(f"Step {step_counter}", decision='REJECT', detail=reject_detail))
            step_results['selected_feature'] = None
            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results))
            break

        if len(current_model_features) == len(ranked_features):
            break

    # 7. Final Model Fit for Visualization
    print("\n--- Final Model Fit for Visualization (CV-based) ---")
    # No blanket except: a failed final refit or save must stop the run loudly, not
    # pass as a finished selection with no final model.
    last_file = model_selection_dir / f"{prefix}{step_counter}.pkl"
    if not last_file.exists():
        last_file = model_selection_dir / f"{prefix}{step_counter - 1}.pkl"

    if not current_model_features:
        _write_empty_selection(last_file, _wrap_step, univariate_results_path, input_data_path, 'params')
        return

    final_fold_shapes = []
    print(f"  Calculating filter shapes across {len(cv_folds)} folds...")

    for _fold_idx, (tr_idx, _te_idx) in enumerate(cv_folds):

        X_list_tr = [all_feature_data[f]['X'][tr_idx] for f in current_model_features]
        y_tr = y_global[tr_idx]

        fold_res = {}
        if model_type == 'sklearn':
            X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in X_list_tr])
            y_tr_pos = np.maximum(y_tr, 1e-6)

            model = GridSearchCV(
                GammaRegressor(),
                {'alpha': list(lr_params['alphas'])},
                cv=session_grouped_inner_cv(groups_global[tr_idx], lr_params['cv']),
                scoring='neg_mean_gamma_deviance',
            ).fit(X_tr_stacked, y_tr_pos, groups=groups_global[tr_idx])
            coefs = model.best_estimator_.coef_.flatten()
            n_bases = basis_matrix.shape[1]

            for k, f_name in enumerate(current_model_features):
                feat_coefs = coefs[k * n_bases: (k + 1) * n_bases]
                fold_res[f_name] = np.dot(feat_coefs, basis_matrix.T).flatten()
        else:
            X_tr_gam = get_unrolled_X_for_multivariate(X_list_tr, history_frames)
            y_tr_gam = np.repeat(y_tr + 1e-6, history_frames)

            gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
            for i in range(1, len(current_model_features)):
                gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])

            gam_fold = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_tr_gam, y_tr_gam)

            time_idx = np.arange(history_frames)
            base_grid = np.zeros((history_frames, 2 * len(current_model_features)))
            for k in range(len(current_model_features)):
                base_grid[:, k * 2 + 1] = time_idx

            pred_base = gam_fold.predict(base_grid)

            for k, feat in enumerate(current_model_features):
                grid = base_grid.copy()
                grid[:, k * 2] = 1.0
                pred_feat = gam_fold.predict(grid)
                fold_res[feat] = (pred_feat - pred_base).flatten()

            del gam_fold, X_tr_gam, y_tr_gam

        final_fold_shapes.append(fold_res)
        gc.collect()

    # Honest held-out score (bout parameters). Refit the accepted feature set
    # on ALL development rows and score it once on the reserved held-out rows
    # (D^2 = explained Gamma deviance), mirroring the finalization fold fit for
    # the active engine, plus a shuffled-label null control for the held-out
    # chance floor. This is the only number untouched by screening, greedy
    # acceptance, or inner-CV alpha selection. Skipped when holdout is disabled
    # (`_held_positions` empty) or no feature was accepted, leaving `heldout`
    # None so no-holdout runs are byte-identical to before.
    heldout_result = None
    if _held_positions.size > 0 and len(current_model_features) > 0:
        y_dev = y_global[_dev_positions]
        y_held = y_global[_held_positions]
        # Distinct-but-deterministic seed for the held-out null permutation;
        # offset the fold seeds by `n_splits` so it never collides with a fold.
        held_rng = np.random.default_rng(random_seed + n_splits + 1)

        heldout_result = {
            'engine': model_type,
            'features': list(current_model_features),
            'n_held_out_sessions': len(np.unique(groups_global[_held_positions])),
            'n_held_out_events': int(_held_positions.size),
        }
        for branch in ('actual', 'null'):
            # The held-out null shuffles targets across the development set.
            y_dev_use = (y_dev if branch == 'actual'
                         else y_dev[null_target_permutation(y_dev.shape[0], held_rng)])
            rec = {'explained_deviance': np.nan, 'residual_deviance': np.nan,
                   'heldout_fit_failed': False, 'heldout_fit_error': None}
            try:
                if model_type == 'sklearn':
                    X_dev_stacked = np.hstack(
                        [np.dot(all_feature_data[f]['X'][_dev_positions], basis_matrix)
                         for f in current_model_features]
                    )
                    X_held_stacked = np.hstack(
                        [np.dot(all_feature_data[f]['X'][_held_positions], basis_matrix)
                         for f in current_model_features]
                    )
                    y_dev_pos = np.maximum(y_dev_use, 1e-6)
                    held_model = GridSearchCV(
                        GammaRegressor(),
                        {'alpha': list(lr_params['alphas'])},
                        cv=session_grouped_inner_cv(groups_global[_dev_positions], lr_params['cv']),
                        scoring='neg_mean_gamma_deviance',
                    ).fit(X_dev_stacked, y_dev_pos, groups=groups_global[_dev_positions])
                    y_pred = held_model.best_estimator_.predict(X_held_stacked)
                else:
                    X_dev_gam = get_unrolled_X_for_multivariate(
                        [all_feature_data[f]['X'][_dev_positions] for f in current_model_features],
                        history_frames,
                    )
                    X_held_gam = get_unrolled_X_for_multivariate(
                        [all_feature_data[f]['X'][_held_positions] for f in current_model_features],
                        history_frames,
                    )
                    y_dev_gam = np.repeat(y_dev_use + 1e-6, history_frames)
                    gam_terms = te(0, 1, n_splines=[n_splines_value, n_splines_time])
                    for i in range(1, len(current_model_features)):
                        gam_terms += te(i * 2, i * 2 + 1, n_splines=[n_splines_value, n_splines_time])
                    held_gam = GAM(gam_terms, distribution='gamma', link='log', **gam_kwargs).fit(X_dev_gam, y_dev_gam)
                    # Same per-event reduction as the CV folds the selection ranked on:
                    # average the frame predictions on the LINK (log) scale, then apply
                    # the inverse link, exp(mean(eta)), not mean(exp(eta)), so the held-out
                    # D^2 is comparable with the CV D^2.
                    eta_held = np.log(held_gam.predict_mu(X_held_gam)).reshape(len(y_held), history_frames)
                    y_pred = np.exp(np.mean(eta_held, axis=1))

                y_held_safe = np.maximum(y_held, 1e-6)
                y_pred_safe = np.maximum(y_pred, 1e-6)
                y_null_pred = np.full_like(y_held_safe, np.mean(y_held_safe))
                res_dev = mean_gamma_deviance(y_held_safe, y_pred_safe)
                null_dev = mean_gamma_deviance(y_held_safe, y_null_pred)
                rec['residual_deviance'] = res_dev
                rec['explained_deviance'] = 0.0 if null_dev == 0 else 1 - (res_dev / null_dev)
            except Exception as e:
                # The held-out number is touched once; a failed refit is recorded as such
                # (flag + error text beside the NaN metrics) and warned about, never left
                # as a silent NaN.
                rec['heldout_fit_failed'] = True
                rec['heldout_fit_error'] = f"{type(e).__name__}: {e}"
                warnings.warn(f"Held-out refit failed (params {branch}): {rec['heldout_fit_error']}",
                              RuntimeWarning, stacklevel=2)
            heldout_result[branch] = rec

    final_res = {
        'final_model_features': current_model_features,
        'filter_shapes': final_fold_shapes,
        'heldout': heldout_result,
        'univariate_results_path': univariate_results_path,
        'input_data_path': input_data_path
    }

    with Path(last_file).open('rb') as open_pkl_file:
        data = pickle.load(open_pkl_file)

    for _k in RESERVED_METADATA_KEYS:
        data.pop(_k, None)
    data.update(final_res)

    _dump_step_pickle(last_file, _wrap_step(data))

    metrics_by_strategy = {'Final': {'D^2': best_current_score},
                           'Baseline': {'D^2': 0.0}}
    # Held-out D^2 (development-refit, scored once) -- the honest number
    # alongside the optimistic development-CV `Final`.
    if heldout_result is not None:
        metrics_by_strategy['Held-out'] = {'D^2': heldout_result['actual']['explained_deviance']}
    print(format_run_summary(
        label=f"params model-selection ({len(current_model_features)} features)",
        metrics_by_strategy=metrics_by_strategy,
        out_path=str(last_file),
    ))



#: Per-fold scalar metric keys the multinomial selector stores for every candidate.
_MULTINOMIAL_FOLD_METRIC_KEYS = ('auc', 'score', 'recall', 'f1', 'll', 'brier', 'ece', 'mcc')
#: Per-fold list keys of a multinomial candidate record besides the metrics.
_MULTINOMIAL_FOLD_LIST_KEYS = ('weights', 'basis_coefficients', 'intercepts', 'y_true', 'y_pred', 'y_probs',
                               'test_indices', 'p_train', 'p_test', 'confusion_matrix', 'n_iter', 'converged',
                               'fit_time', 'selected_lambda_smooth', 'selected_l2_reg', 'hyperparam_grid_audit',
                               'hyperparams_tuned')
#: Start-frame slot -> pickle array key of the per-session single-array pipelines
#: (multinomial, manifold): the starts of a session are those of its ``X`` rows in order.
_SESSION_WINDOW_SLOTS = {'X': 'X'}


def _verify_session_windows_against_traces(feature_data: dict,
                                           traces_path: str | Path,
                                           window_starts: dict,
                                           history_frames: int,
                                           features: list,
                                           sessions: list,
                                           cast_dtype) -> None:
    """
    Description
    -----------
    The lag-zero self-check of the per-step shift test for the per-session
    single-array layout of the multinomial and manifold pipelines
    (``feature_data[feature][session]['X']`` holds the session's windows, cast to
    ``cast_dtype`` by the extraction): re-cuts every session's windows from the
    trace sidecar at the stored start frames, casts them the same way and requires
    bit-exact equality and the same row count as stored starts. A mismatch means
    the pickle, its start frames and the sidecar do not describe one extraction, and
    the run aborts before any model is fit.

    Parameters
    ----------
    feature_data (dict)
        The modeling pickle's feature dict (metadata blocks stripped).
    traces_path (str | Path)
        The HDF5 sidecar.
    window_starts (dict)
        ``{session: {'X': (n,) start frames, ...}}`` from ``_input_metadata``.
    history_frames (int)
        Window length in frames (the stored, unbinned window).
    features (list)
        Features to check.
    sessions (list)
        Sessions to check.
    cast_dtype
        The dtype the extraction cast the windows to (``np.float32``).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        On the first session whose windows do not reproduce.
    """

    with h5py.File(Path(traces_path), 'r') as handle:
        for feature in features:
            for session_id in sessions:
                if session_id not in feature_data[feature]:
                    continue
                stored = feature_data[feature][session_id]['X']
                starts = np.asarray(window_starts[session_id]['X'], dtype=int)
                if stored.shape[0] != starts.size:
                    msg = (f"_verify_session_windows_against_traces: {feature!r} / {session_id!r}: "
                           f"{stored.shape[0]} stored rows but {starts.size} start frames.")
                    raise ValueError(msg)
                if starts.size == 0:
                    continue
                column = handle[feature][str(session_id)][()]
                rebuilt = cut_history_windows(column, starts, history_frames).astype(cast_dtype)
                if rebuilt.shape != stored.shape or not np.array_equal(rebuilt, stored):
                    msg = (f"_verify_session_windows_against_traces: {feature!r} / {session_id!r}: windows "
                           f"re-cut from the trace sidecar at lag 0 do not match the stored windows (shapes "
                           f"{rebuilt.shape} vs {stored.shape}). The pickle, its start frames and "
                           f"{traces_path} are not from one extraction.")
                    raise ValueError(msg)


def _multinomial_empty_cand_data(canonical_classes: np.ndarray, balance_train: bool) -> dict:
    """
    Description
    -----------
    A fresh, empty results entry for one multinomial candidate: the ``folds`` block of
    empty per-fold lists (the scalar metrics of ``_MULTINOMIAL_FOLD_METRIC_KEYS`` and
    the lists of ``_MULTINOMIAL_FOLD_LIST_KEYS``), the ``balanced_train`` flag, the
    fitted class order (filled by the first successful fold) and the cohort's
    canonical class order.

    Parameters
    ----------
    canonical_classes (np.ndarray)
        Sorted unique labels of the whole cohort.
    balance_train (bool)
        Whether the training folds were class-balanced.

    Returns
    -------
    cand_data (dict)
        ``{'folds': {...}, 'classes': None, 'canonical_classes': ...}``.
    """

    folds = {'metrics': {m: [] for m in _MULTINOMIAL_FOLD_METRIC_KEYS}}
    for key in _MULTINOMIAL_FOLD_LIST_KEYS:
        folds[key] = []
    folds['balanced_train'] = bool(balance_train)
    return {'folds': folds, 'classes': None, 'canonical_classes': canonical_classes}


def _multinomial_assemble_cand_data(fold_results: list, feature: str, canonical_classes: np.ndarray,
                                    balance_train: bool, n_splits: int) -> dict:
    """
    Description
    -----------
    Turns the per-fold worker results of one multinomial candidate into its results
    entry, keeping every per-fold list aligned: a failed fold is logged and
    contributes the shape-preserving placeholders the former inline loop appended
    (NaN metrics, None weights / coefficients / intercepts, empty arrays, NaN class
    proportions and a NaN ``(K, K)`` confusion matrix, unconverged, NaN
    hyperparameters and an empty grid audit).

    Parameters
    ----------
    fold_results (list)
        One ``_multinomial_candidate_fold`` result per fold, in fold order.
    feature (str)
        The candidate, for the log line.
    canonical_classes (np.ndarray)
        Sorted unique labels of the whole cohort.
    balance_train (bool)
        Whether the training folds were class-balanced.
    n_splits (int)
        Number of folds, for the log line.

    Returns
    -------
    cand_data (dict)
        The candidate's results entry.
    """

    cand_data = _multinomial_empty_cand_data(canonical_classes, balance_train)
    folds = cand_data['folds']
    K = len(canonical_classes)
    for fold_idx, result in enumerate(fold_results):
        if result['ok']:
            fold = result['fold']
            for key in _MULTINOMIAL_FOLD_METRIC_KEYS:
                folds['metrics'][key].append(fold['metrics'][key])
            for key in _MULTINOMIAL_FOLD_LIST_KEYS:
                folds[key].append(fold[key])
            if cand_data['classes'] is None:
                cand_data['classes'] = result['classes']
            continue
        print(
            f"\n    [!] Fold {fold_idx + 1:02d}/{n_splits:02d} failed on +{feature}: {result['error']}",
            flush=True,
        )
        for key in _MULTINOMIAL_FOLD_METRIC_KEYS:
            folds['metrics'][key].append(np.nan)
        folds['weights'].append(None)
        folds['basis_coefficients'].append(None)
        folds['intercepts'].append(None)
        folds['y_true'].append(np.empty((0,), dtype=np.int32))
        folds['y_pred'].append(np.empty((0,), dtype=np.int32))
        folds['y_probs'].append(np.empty((0, K), dtype=np.float32))
        folds['test_indices'].append(np.empty((0,), dtype=np.int64))
        folds['p_train'].append(np.full(K, np.nan, dtype=np.float32))
        folds['p_test'].append(np.full(K, np.nan, dtype=np.float32))
        folds['confusion_matrix'].append(np.full((K, K), np.nan))
        folds['n_iter'].append(np.nan)
        folds['converged'].append(False)
        folds['fit_time'].append(np.nan)
        folds['selected_lambda_smooth'].append(float('nan'))
        folds['selected_l2_reg'].append(float('nan'))
        folds['hyperparam_grid_audit'].append({
            'grid_scores': {}, 'grid_ses': {},
            'argmax_pair': None,
            'one_se_applied': False, 'one_se_threshold': None,
        })
        folds['hyperparams_tuned'].append(False)
    return cand_data


def _multinomial_candidate_fold(feat: str, fold_idx: int) -> dict:
    """
    Description
    -----------
    One (candidate, fold) fit of the multinomial selector: fits the incumbent features
    plus ``feat`` on the fold's (optionally class-balanced) training rows with the
    JAX ``SmoothMultinomialLogisticRegression`` and scores the fold's test rows. The
    data comes from the ``'multinomial_step'`` pool-context entry the selector stores
    before running the fits (``cv_folds``, ``binned_data``, ``y_global``,
    ``groups_global``, the per-fold cached ``fold_tr_idx_model`` / ``fold_base_tr`` /
    ``fold_base_te``, ``current_model_features``, the hyperparameter closure
    ``pick_hyperparams``, ``class_proportions``, the model knobs). The operations are
    those of the former inline fold body, so the fold scores are the same numbers.

    Parameters
    ----------
    feat (str)
        The candidate feature.
    fold_idx (int)
        Fold index.

    Returns
    -------
    result (dict)
        ``{'ok': True, 'fold': {...}, 'classes': model.classes_}`` with one entry per
        per-fold list of the candidate record, or ``{'ok': False, 'error': str}``.
    """

    ctx = pool_context('multinomial_step')
    _, te_idx = ctx['cv_folds'][fold_idx]
    tr_idx_model = ctx['fold_tr_idx_model'][fold_idx]
    binned_data = ctx['binned_data']
    y_global = ctx['y_global']
    groups_global = ctx['groups_global']
    hp = ctx['hp']
    temporal_basis = ctx['temporal_basis']
    canonical_classes = ctx['canonical_classes']
    trial_feats = [*ctx['current_model_features'], feat]
    n_trial_feats = len(trial_feats)
    try:
        new_tr = binned_data[feat][tr_idx_model]
        new_te = binned_data[feat][te_idx]
        base_tr = ctx['fold_base_tr'][fold_idx]
        base_te = ctx['fold_base_te'][fold_idx]
        X_tr_stacked = np.hstack([base_tr, new_tr]) if base_tr is not None else new_tr
        X_te_stacked = np.hstack([base_te, new_te]) if base_te is not None else new_te
        y_tr, y_te = y_global[tr_idx_model], y_global[te_idx]

        fold_lambda_smooth, fold_l2_reg, fold_grid_audit, fold_tuned_flag = (
            ctx['pick_hyperparams'](
                X_tr_stacked, y_tr, groups_global[tr_idx_model],
                n_feats_=n_trial_feats,
                fold_idx_=fold_idx,
            )
        )

        model = SmoothMultinomialLogisticRegression(
            n_features=n_trial_feats, n_time_bins=ctx['n_time_bins'],
            lambda_smooth=fold_lambda_smooth,
            l2_reg=fold_l2_reg,
            smoothness_derivative_order=ctx['smoothness_order'],
            smoothness_reflective_edges=ctx['reflective_edges'],
            focal_gamma=ctx['model_focal_gamma'],
            uniform_class_weights=ctx['model_uniform_weights'],
            learning_rate=hp['learning_rate'],
            max_iter=hp['max_iter'],
            tol=hp['tol'],
            random_state=hp['random_state'] + fold_idx,
            _use_lax_loop=ctx['use_lax_loop'],
        )
        model.fit(X_tr_stacked, y_tr)

        y_proba = model.predict_proba(X_te_stacked, balanced=hp['balance_predictions_bool'])
        y_pred = model.predict(X_te_stacked, balanced=hp['balance_predictions_bool'])

        eps = 1e-15
        y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

        metrics = {}
        metrics['auc'] = roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro', labels=model.classes_)
        metrics['score'] = balanced_accuracy_score(y_te, y_pred)
        metrics['f1'] = f1_score(y_te, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_te, y_pred, average='macro', zero_division=0)
        metrics['ll'] = log_loss(y_te, y_proba_clipped, labels=model.classes_)

        # Canonical-ordered probability matrix so Brier / ECE are computed against a
        # stable column ordering across folds even when a rare class is absent from
        # `model.classes_`.
        probs_canonical = align_probs_to_canonical(y_proba, model.classes_, canonical_classes)
        try:
            metrics['brier'] = brier_score_multi(y_te, probs_canonical, canonical_classes)
        except Exception:
            metrics['brier'] = np.nan
        try:
            metrics['ece'] = expected_calibration_error(y_te, y_pred, probs_canonical, n_bins=ctx['ece_n_bins'])
        except Exception:
            metrics['ece'] = np.nan
        metrics['mcc'] = safe_matthews_corrcoef(y_te, y_pred)

        fold = {
            'metrics': metrics,
            'weights': (basis_coefficients_to_frames(model.coef_, temporal_basis, n_features=n_trial_feats, axis=1)
                        if temporal_basis is not None else model.coef_),
            'basis_coefficients': np.asarray(model.coef_) if temporal_basis is not None else None,
            'intercepts': model.intercept_,
            'y_true': y_te,
            'y_pred': y_pred,
            'y_probs': y_proba,
            'test_indices': te_idx,
            'p_train': ctx['class_proportions'](y_tr),
            'p_test': ctx['class_proportions'](y_te),
            'confusion_matrix': safe_confusion_matrix(y_te, y_pred, labels=canonical_classes),
            'n_iter': int(model.n_iter_),
            'converged': bool(model.converged_),
            'fit_time': float(model.fit_time_),
            'selected_lambda_smooth': fold_lambda_smooth,
            'selected_l2_reg': fold_l2_reg,
            'hyperparam_grid_audit': fold_grid_audit,
            'hyperparams_tuned': fold_tuned_flag,
        }
        return {'ok': True, 'fold': fold, 'classes': model.classes_}
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


def multinomial_vocal_category_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str | None = None,
        use_top_rank_as_anchor: bool = False,
        *,
        p_val: float
) -> None:
    """
    Performs forward stepwise selection for multinomial USV category prediction.

    This function identifies the optimal subset of behavioral features that
    predict the full repertoire of USV categories simultaneously. It uses the
    JAX-accelerated SmoothMultinomialLogisticRegression model.

    Splitting & balancing invariant:
    --------------------------------
    The test fold always preserves the natural class prior of the source data
    (stratified in 'mixed' mode, within-tolerance of global in 'session' mode).
    The training fold follows one of two paths, selected by
    `hyperparameters.linear_models.multinomial_logistic.balance_train_bool`:

    - `false` (default): the training fold retains the natural class prior,
      and class imbalance is handled inside the JAX loss through softened
      inverse-frequency alpha weights combined with focal-gamma modulation
      (`focal_loss_gamma`).
    - `true`: every learned-model training fold (anchor + forward-selection
      trials) is sample-level down-sampled so each class contributes
      `min(class_count)` rows. The JAX fit is then invoked with
      `focal_gamma=0` and uniform `1 / n_classes` class weights to avoid
      double-correcting an already balanced batch. The `null_model_free`
      baseline keeps the natural-rate training distribution so its empirical
      prior floor remains an informative reference.

    Reported metrics are imbalance-robust (balanced accuracy, log-loss,
    macro OvR AUC).

    Algorithm adaptations for multinomial multi-feature data:
    1.  Data stacking: Instead of unrolling into (Value, Time) columns for pyGAM,
        features are horizontally concatenated (hstack) into a wide matrix of
        shape (N_samples, N_features * N_time_bins).
    2.  Dynamic architecture: The JAX model is re-instantiated at each step with
        `n_features = len(trial_features)` to ensure the temporal smoothing
        penalty (the configured `smoothness_derivative_order`) is applied
        strictly within each feature's bins.
    3.  Direct filter extraction: Because the model is linear in logit space,
        filter shapes do not require synthetic grid predictions. The learned
        `coef_` matrix is sliced directly to yield the exact temporal filters
        for every USV category. Under `temporal_basis.type = 'bspline'` the
        coefficients are mapped back onto the frame axis first
        (`basis_coefficients_to_frames`), so the stored `weights` keep the
        full-resolution layout and the raw spline coefficients are stored
        beside them.
    4.  Model-Free Baseline (Step 0): Computes the Marginal Class Prior (the
        empirical frequency of each USV category in the training set) as the
        absolute baseline. Features must prove they offer predictive power
        above and beyond simply guessing the majority class.
    5.  Per-step shift test: at every step the cross-validated winner (best mean
        macro AUC) is tested with ``step_shift_test.run_step_shift_test``. The
        incumbent and the incumbent plus the winner are refit on the development
        rows exactly as the held-out refit fits them; the statistic is the gain in
        the in-sample macro (class-balanced) log-likelihood per event
        (``modeling_utils.macro_log_likelihood``); its null comes from refits after
        circularly shifting the winner's processed trace within every development
        session and re-cutting (and re-binning / re-projecting) its history windows
        at the stored start frames, ``floor(m / alpha)`` times with Bonferroni
        correction over the ``m`` candidates of the step. The first feature is
        tested the same way against the marginal class prior of the refit rows.

    Data Persistence:
    -----------------
    At each step of the forward sweep, a .pkl file is saved containing the complete
    experimental state. This deep storage preserves all raw fold-level data to enable
    post-hoc statistical testing and confusion matrix generation without refitting.

    The saved dictionary at each step contains:
    - 'step_idx' (int): The current iteration number.
    - 'current_features' (list): Features selected prior to this step.
    - 'selected_feature' (str or None): The winning feature added in this step.
    - 'baseline_score' (float): The macro OvR AUC of the current_features model;
       at Step 0 current_features is the empty list and this is instead the
       model-free marginal-prior baseline AUC (the `null_model_free` entry).
    - 'candidates_summary' (dict): Maps each tested feature to its detailed results:
        - 'mean_auc', 'se_auc': Aggregated AUC metrics (reported; the ranking
          uses the mean, the acceptance is the per-step shift test stored under
          the step's 'step_test').
        - 'classes': Array of original USV category string labels.
        - 'folds': A nested dictionary containing lists (length = n_splits) of:
            * 'metrics': dict of the following per-fold scalars —
                - `ll` : multinomial log-loss; strictly proper probabilistic score.
                - `auc` : macro one-vs-rest ROC-AUC; threshold-free ranking quality.
                - `f1` : macro F1; imbalance-robust harmonic mean of precision / recall.
                - `recall` : macro recall (precision is derivable from the saved
                  confusion matrix, so it is intentionally not stored).
                - `score` : balanced accuracy (mean of per-class recall).
                - `brier` : multiclass Brier score; quadratic counterpart to log-loss.
                - `ece` : top-label expected calibration error (10 bins); gap
                  between predicted confidence and empirical accuracy.
                - `mcc` : Matthews correlation coefficient; chance-corrected
                  [-1, +1] summary of the confusion matrix.
            * 'weights': The learned JAX coefficient matrix, on the frame axis
              (`n_classes x n_features * n_time_bins`) whatever the temporal basis.
            * 'basis_coefficients': The raw spline coefficients when a B-spline
              `temporal_basis` is in use (`n_classes x n_features * n_splines`),
              else None.
            * 'intercepts': The learned JAX biases.
            * 'y_true', 'y_pred', 'y_probs': Ground truth, hard predictions, and softmax arrays.
            * 'test_indices': The specific dataset rows used in the validation fold.
            * 'confusion_matrix': (K, K) matrix with canonical class ordering.
            * 'p_train', 'p_test': Realized per-class proportions in the fold.
            * 'n_iter', 'converged', 'fit_time': JAX optimizer diagnostics.

    *Note: The final step's file is additionally appended with 'final_model_features'
    and 'weights_reshaped' (the extracted temporal kernels for all included features).*

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file.
    input_data_path : str
        Path to the raw extracted feature data .pkl file.
    output_directory : str
        Directory to save the step-wise state dictionaries.
    settings_path : str, optional
        Path to the JSON settings file.
    use_top_rank_as_anchor : bool, default False
        If True, initializes the search by forcing the highest-ranked univariate
        feature as the baseline model.
    p_val : float
        Keyword-only, no default: the dispatcher passes ``model_params.selection_p_val``.
        The alpha level used for significance screening against shuffled controls.

    Returns
    -------
    None
        Writes one pickle per selection step and the final consolidated
        selection pickle into ``output_directory``.
    """

    if settings_path is None:
        settings_path_obj = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with Path(settings_path).open() as f:
            settings = json.load(f)
    except FileNotFoundError as err:
        msg = f"Settings file not found at {settings_path}"
        raise FileNotFoundError(msg) from err

    model_selection_dir = Path(output_directory)
    model_selection_dir.mkdir(parents=True, exist_ok=True)

    # Standardized run header (files used + config). The multinomial task always
    # runs the JAX estimator, so the engine label is 'jax'.
    print(format_run_header(
        task='SELECTION:MULTINOMIAL',
        engine='jax',
        feature=Path(univariate_results_path).stem,
        input_files={'univariate results': univariate_results_path,
                     'input data': input_data_path,
                     'settings': settings_path},
        output_directory=str(model_selection_dir),
        split_strategy=settings['model_validation']['split_strategy'],
        n_splits=int(settings['model_validation']['n_cv_folds']),
    ))

    with Path(univariate_results_path).open('rb') as f:
        univariate_data = pickle.load(f)

    # Strip upstream metadata blocks before iterating feature entries.
    _univ_pre_md = univariate_data.pop('_run_metadata', None)
    _require_analysis_type(univariate_data, 'multinomial', univariate_results_path)
    _input_pre_md_from_univ = univariate_data.pop('_input_metadata', None)
    univariate_data.pop('_consolidation_metadata', None)

    candidates = []
    num_features = len(univariate_data)

    for feat_name, payload in univariate_data.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            _, results = payload
        else:
            results = payload

        if ('actual' not in results or
                'auc' not in results['actual']['folds']['metrics']):
            continue

        actual_auc = np.array(results['actual']['folds']['metrics']['auc'])
        null_auc = np.array(results['null']['folds']['metrics']['auc'])

        valid_actual = actual_auc[~np.isnan(actual_auc)]
        valid_null = null_auc[~np.isnan(null_auc)]

        if len(valid_actual) == 0 or len(valid_null) == 0:
            continue

        mean_actual_auc = np.mean(valid_actual)
        null_threshold = np.percentile(valid_null, q=100 - ((p_val / num_features) * 100))

        if mean_actual_auc > null_threshold:
            candidates.append({'feature': feat_name, 'mean_auc': mean_actual_auc})

    candidates.sort(key=lambda x: x['mean_auc'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print("No significant features found. Aborting.")
        return

    print("Loading and binning raw input data...")
    with Path(input_data_path).open('rb') as f:
        raw_data = pickle.load(f)

    # Final upstream-metadata harvest. The univariate-side blocks were
    # captured above; the input-side block falls back to the Level-1
    # pickle if the consolidator did not provide one (legacy run).
    _require_analysis_type(raw_data, 'multinomial', input_data_path)
    _input_md = _input_pre_md_from_univ
    if _input_md is None and '_input_metadata' in raw_data:
        _input_md = raw_data.pop('_input_metadata')
    else:
        raw_data.pop('_input_metadata', None)
    raw_data.pop('_run_metadata', None)
    raw_data.pop('_consolidation_metadata', None)
    _univariate_md = _univ_pre_md

    # The temporal representation (full-resolution frames or a B-spline basis) decides the
    # penalty settings the fit uses; `hp` is that effective block. The basis spans the
    # history window of the input pickle (every feature and session shares its length).
    _first_feature = raw_data[ranked_features[0]]
    history_frames = _first_feature[sorted(_first_feature)[0]]['X'].shape[1]
    temporal_basis, hp = resolve_temporal_basis(
        settings['hyperparameters']['linear_models']['multinomial_logistic'], history_frames)
    bin_size = hp['bin_resizing_factor']

    binned_data = {}
    y_global, groups_global = None, None
    n_time_bins = None

    def _multinomial_bin_and_project(X_s: np.ndarray, feat: str) -> np.ndarray:
        """
        Description
        -----------
        The temporal representation of one session's history windows: the optional
        `bin_resizing_factor` binning (keeping the MOST RECENT `new_T * bin_size`
        columns and dropping the oldest `T % bin_size`, exactly as the univariate
        loader bins: the last column is the frame before onset, so truncating the
        tail would drop the frames nearest onset and shift the design against the
        filters) followed by the B-spline projection when a temporal basis is in
        use. The binning loop below and the per-step shift test's shifted refits
        both go through this one function, so a shifted window is represented
        exactly as the stored window it replaces.

        Parameters
        ----------
        X_s (np.ndarray)
            ``(n_rows, history_frames)`` windows of one session.
        feat (str)
            The feature, named in the error.

        Returns
        -------
        X_s (np.ndarray)
            ``(n_rows, n_time_bins)`` design rows (before the float32 cast).
        """

        N, T = X_s.shape
        if bin_size > 1:
            new_T = T // bin_size
            if new_T < 1:
                msg = f"bin_size={bin_size} exceeds the history window length T={T} for feature '{feat}'."
                raise ValueError(msg)
            X_s = X_s[:, T - new_T * bin_size:].reshape(N, new_T, bin_size).mean(axis=2)
        if temporal_basis is not None:
            X_s = project_history_onto_basis(X_s, temporal_basis)
        return X_s

    for feat in ranked_features:
        X_list, y_list, g_list = [], [], []
        sessions = sorted(raw_data[feat].keys())
        for sess_id in sessions:
            X_s = raw_data[feat][sess_id]['X']
            y_s = raw_data[feat][sess_id]['y']

            X_s = _multinomial_bin_and_project(X_s, feat)

            X_list.append(X_s)
            y_list.append(y_s)
            g_list.append(np.full(len(y_s), sess_id))

        binned_data[feat] = np.vstack(X_list).astype(np.float32)
        if y_global is None:
            y_global = np.concatenate(y_list).astype(np.int32)
            groups_global = np.concatenate(g_list)
            n_time_bins = binned_data[feat].shape[1]

    # The per-step shift test's inputs and their lag-zero self-check: every stored
    # window of every screened feature and session must re-cut exactly (float32, as
    # stored) from the trace sidecar at its stored start frame.
    camera_fps = float(settings['io']['camera_sampling_rate'])
    step_cfg = _step_test_settings(settings, camera_fps)
    traces_path, window_starts = _require_shift_test_inputs(_input_md, input_data_path, camera_fps)
    _all_session_ids = sorted({str(s_id) for feat in ranked_features for s_id in raw_data[feat]})
    _verify_session_windows_against_traces(
        raw_data, traces_path, window_starts, history_frames,
        features=ranked_features, sessions=_all_session_ids, cast_dtype=np.float32,
    )
    print(f"Shift-test inputs verified: {len(ranked_features)} feature(s) x {len(_all_session_ids)} "
          f"session(s) re-cut exactly from {Path(traces_path).name}; lag floor "
          f"{step_cfg['min_lag_seconds']} s = {step_cfg['min_lag_frames']} frames.")
    print("JAX selectors run their candidate fits and null refits serially inside the job: XLA's thread "
          "pool is not fork-safe once the backend has run, and one fit already uses every core.")

    del raw_data
    gc.collect()

    split_strategy = settings['model_validation']['split_strategy']
    n_splits = settings['model_validation']['n_cv_folds']
    test_prop = settings['model_validation']['cv_validation_proportion']
    random_seed = settings['model_validation']['random_seed']
    balance_train = hp['balance_train_bool']

    # Reserve the held-out sessions carved ONCE at extraction time and shipped in
    # the univariate `_input_metadata.held_out_session_ids`. They are dropped from
    # every greedy cv_fold and every candidate fit; the accepted feature set is
    # then refit on the development sessions and scored on this block once in the
    # finalization stage (the honest, selection-leakage-free number). An empty
    # list -- holdout disabled or a pre-v2 pickle -- makes `_dev_mask` all-True
    # and `_held_positions` empty, so the path is an exact no-op.
    _held_out_session_ids = held_out_session_ids_from_metadata(_input_md if _input_md is not None else {})
    _dev_mask, _held_mask = development_heldout_masks(groups_global, _held_out_session_ids)
    _dev_positions = np.where(_dev_mask)[0]
    _held_positions = np.where(_held_mask)[0]
    _dev_session_ids = [str(s_id) for s_id in np.unique(groups_global[_dev_positions])]
    if _held_positions.size > 0:
        print(
            f"Held-out reserve: {len(_held_out_session_ids)} session(s), "
            f"{_held_positions.size} event(s) excluded from selection; scored once at finalization."
        )

    # Both `'session'` and `'mixed'` go through the shared splitter so
    # the cohort-wide and per-fold class-coverage guards apply
    # uniformly. `n_categories` is read back from the Level-1
    # `_input_metadata` block harvested above — the extractor
    # auto-derives this value from the cohort-pooled labels after the
    # `exclude_noise_usvs` filter (see
    # `extract_and_save_multinomial_input_data`), so there is no
    # hand-set JSON literal that could go stale if
    # `usv_category_column_name` or `exclude_noise_usvs` change.
    #
    # Fold the DEVELOPMENT sessions only, then remap the dev-relative indices the
    # splitter returns back into full-array index space via `_dev_positions`, so
    # every fold index stays a valid row of the full arrays while no held-out row
    # enters a train or validation split. With holdout disabled `_dev_positions`
    # is the identity map, so this is byte-identical to the pre-holdout behaviour.
    cv_folds_dev, _ = get_stratified_group_splits_stable(
        groups=groups_global[_dev_mask],
        y=y_global[_dev_mask],
        split_strategy=split_strategy,
        test_prop=test_prop,
        n_splits=n_splits,
        random_seed=random_seed,
        n_categories=int(_input_md['analysis_specific']['usv_category_number']),
        tolerance=settings['model_validation']['session_split_initial_tolerance'],
        max_total_attempts=settings['model_validation']['session_split_max_attempts'],
        widen_step=settings['model_validation']['session_split_widen_step'],
        widen_every=settings['model_validation']['session_split_widen_every'],
    )
    cv_folds = [
        (_dev_positions[tr_idx], _dev_positions[te_idx])
        for (tr_idx, te_idx) in cv_folds_dev
    ]
    if _held_positions.size > 0:
        _held_set = set(_held_positions.tolist())
        for _tr_idx, _te_idx in cv_folds:
            if _held_set.intersection(_tr_idx.tolist()) or _held_set.intersection(_te_idx.tolist()):
                msg_0 = (
                    "Held-out session events leaked into a multinomial selection "
                    "CV fold; the development/held-out split is inconsistent."
                )
                raise AssertionError(msg_0)

    # When balance_train_bool is active the learned-model paths (anchor,
    # forward-selection trials) down-sample each training fold per-class and
    # the JAX fit is switched to focal_gamma=0 + uniform class weights so the
    # already balanced batch is not double-corrected. The null_model_free
    # baseline intentionally keeps the natural-rate training distribution so
    # its empirical-prior floor remains an informative baseline.
    def _fold_train_indices_for_model(tr_idx: np.ndarray, fold_idx: int) -> np.ndarray:
        """
        Description
        -----------
        The training rows a learned model fits on in one outer fold. With
        ``balance_train`` off these are the fold's training rows unchanged;
        with it on they are down-sampled so every class contributes the same
        number of rows (:func:`_balance_multinomial_train_indices`), drawn with
        a generator seeded ``random_seed + fold_idx + 1`` so every candidate in
        the same fold sees the same balanced subset.

        Parameters
        ----------
        tr_idx (np.ndarray)
            The fold's training row indices into the full arrays.
        fold_idx (int)
            The outer fold's 0-based index (offsets the balancing seed).

        Returns
        -------
        train_idx (np.ndarray)
            ``tr_idx`` itself when balancing is off, otherwise a shuffled
            class-balanced subset of it.
        """
        if not balance_train:
            return tr_idx
        fold_rng = np.random.default_rng(random_seed + fold_idx + 1)
        return _balance_multinomial_train_indices(tr_idx, y_global, fold_rng)

    if balance_train:
        model_focal_gamma = 0.0
        model_uniform_weights = True
    else:
        model_focal_gamma = hp['focal_loss_gamma']
        model_uniform_weights = False

    # Joint-tuning configuration. Mirrors the multinomial runner: fixed
    # `lambda_smooth_fixed` / `l2_reg_fixed` centres are used when tuning
    # is off; when on, every outer fold / candidate runs an inner CV over
    # log-spaced grids centred on those values, with the 1-SE
    # interpretability rule optionally promoting the smoothest pair in
    # the SE band. `focal_loss_gamma` is intentionally not tuned — it is
    # treated as a data-modelling choice.
    smoothness_order = hp['smoothness_derivative_order']
    reflective_edges = hp['smoothness_reflective_edges']
    tune_regularization_bool = hp['tune_regularization_bool']
    tune_params = hp['tune_regularization_params']
    lambda_smooth_grid_mn = _log_spaced_grid_multinomial(
        center=hp['lambda_smooth_fixed'],
        decades_each_side=tune_params['lambda_smooth_decades_each_side'],
    )
    l2_reg_grid_mn = _log_spaced_grid_multinomial(
        center=hp['l2_reg_fixed'],
        decades_each_side=tune_params['l2_reg_decades_each_side'],
    )
    inner_cv_folds_mn = tune_params['inner_cv_folds']
    inner_cv_scoring_metric_mn = tune_params['inner_cv_scoring_metric']
    inner_cv_use_one_se_rule_mn = tune_params['inner_cv_use_one_se_rule']
    inner_max_iter_mn = tune_params['inner_max_iter']
    use_lax_loop_mn = hp['use_lax_loop']

    if tune_regularization_bool:
        print(
            f"  Joint per-fold tuning ENABLED: |λ_smooth grid|={len(lambda_smooth_grid_mn)}, "
            f"|l2_reg grid|={len(l2_reg_grid_mn)}, inner CV folds={inner_cv_folds_mn}, "
            f"scoring={inner_cv_scoring_metric_mn}, 1SE rule={'ON' if inner_cv_use_one_se_rule_mn else 'OFF'}"
        )
    else:
        print(
            f"  Joint per-fold tuning DISABLED (fixed λ_smooth={hp['lambda_smooth_fixed']}, "
            f"fixed l2_reg={hp['l2_reg_fixed']})"
        )

    def _pick_multinomial_fold_hyperparams(X_tr_, y_tr_, groups_tr_, n_feats_, fold_idx_):
        """
        Returns `(lambda_smooth, l2_reg, grid_audit, tuned_flag)` for the
        current outer fold and trial feature set. When tuning is off the
        fixed centres are used with the standard `1 / sqrt(n_features)`
        rescale on l2 so larger trial models don't silently
        over-regularise. When tuning is on the inner-CV search rescales
        the l2 grid the same way.

        Parameters
        ----------
        X_tr_ (np.ndarray)
            The outer fold's training design, ``(n_rows, n_features * n_time_bins)``.
        y_tr_ (np.ndarray)
            The training rows' category labels.
        groups_tr_ (np.ndarray)
            The training rows' session labels, so the inner CV splits by session.
        n_feats_ (int)
            Number of features in the trial model (sets the l2 rescale).
        fold_idx_ (int)
            The outer fold's index, for the log.

        Returns
        -------
        lambda_smooth, l2_reg, grid_audit, tuned (tuple[float, float, dict, bool])
            The smoothness and l2 penalties used, the inner-CV grid record (empty
            when not tuned) and whether the inner-CV search ran.
        """
        if not tune_regularization_bool:
            lam_sm = float(hp['lambda_smooth_fixed'])
            lam_l2 = float(hp['l2_reg_fixed'] / np.sqrt(n_feats_))
            return lam_sm, lam_l2, {
                'grid_scores': {}, 'grid_ses': {},
                'argmax_pair': None,
                'one_se_applied': False, 'one_se_threshold': None,
            }, False

        rescaled_l2_grid = l2_reg_grid_mn / np.sqrt(n_feats_)
        lam_sm_win, lam_l2_win, grid_audit_ = _tune_multinomial_regularization(
            X_train=X_tr_,
            y_train=y_tr_,
            groups_train=groups_tr_,
            lambda_smooth_grid=lambda_smooth_grid_mn,
            l2_reg_grid=rescaled_l2_grid,
            inner_cv_folds=inner_cv_folds_mn,
            inner_cv_scoring_metric=inner_cv_scoring_metric_mn,
            inner_cv_use_one_se_rule=inner_cv_use_one_se_rule_mn,
            n_features=n_feats_,
            n_time_bins=n_time_bins,
            smoothness_derivative_order=smoothness_order,
            smoothness_reflective_edges=reflective_edges,
            focal_gamma=model_focal_gamma,
            uniform_class_weights=model_uniform_weights,
            learning_rate=hp['learning_rate'],
            inner_max_iter=inner_max_iter_mn,
            tol=hp['tol'],
            random_state=hp['random_state'] + fold_idx_,
            verbose=False,
            # Force-disable the lax_loop on the inner-CV tuner path. Same
            # rationale as the manifold variant: per-instance JIT cache
            # makes the tuner's ~175 short-lived estimators per outer fold
            # each pay a fresh compile, which stalls for hours on wide
            # graphs. The outer-fit call sites still honour `use_lax_loop_mn`.
            use_lax_loop=False,
            regressor_cls=SmoothMultinomialLogisticRegression,
            ece_n_bins=settings['diagnostics']['ece_n_bins'],
        )
        return lam_sm_win, lam_l2_win, grid_audit_, True

    # Project-wide canonical class order so every per-fold p_train / p_test
    # vector has identical length and index-to-class mapping, enabling direct
    # cross-fold comparison even if a fold happens to miss a rare class.
    canonical_classes = np.unique(y_global)

    def _class_proportions(labels: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        The share of each class of ``canonical_classes`` (the sorted unique
        labels of the whole cohort) among ``labels``, so every fold's
        proportion vector has the same length and class order even when a
        fold lacks a class.

        Parameters
        ----------
        labels (np.ndarray)
            Class labels of one set of rows (a fold's training or test rows).

        Returns
        -------
        proportions (np.ndarray)
            ``(n_classes,)`` float32 proportions summing to 1; all zeros when
            ``labels`` is empty.
        """
        if len(labels) == 0:
            return np.zeros(len(canonical_classes), dtype=np.float32)
        counts = np.array([(labels == c).sum() for c in canonical_classes], dtype=np.float64)
        return (counts / counts.sum()).astype(np.float32)

    current_model_features = []
    best_current_score = 0.0
    # Per-fold scores of the incumbent, stored as each step record's `baseline_folds`.
    # Replaced by the model-free marginal baseline's per-fold AUCs as soon as that
    # block runs.
    best_current_folds = 0.0
    # True when the checkpoint a resume restores is a step that rejected its winner: the
    # search had already finished, so the anchor and the forward loop are skipped.
    resume_converged = False
    step_counter = 0

    fname = Path(univariate_results_path).name
    # Match the binomial / bout-onset regex grouping: `(?:...)` non-capture on
    # the alternation so conditions like `male_mute_partner` are captured in
    # full, together with an `ephys_` recording-modality prefix when present. The older `(male|female.*?)` form captured only the literal `male`
    # for any `male_...` filename and truncated the prefix.
    cond_match = re.search(r'((?:ephys_)?(?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"
    # Pin which USV category column generated the multinomial labels so
    # the per-step prefix (and downstream consolidation) carries the
    # choice forward in the filename — e.g. `qlvm_category`.
    _column_name_cats = _input_md['analysis_specific']['usv_category_column_name']
    prefix = f"model_selection_multinomial_{_column_name_cats}_{target_condition}_{split_strategy}_step_"

    _run_md = build_selection_metadata(
        modeling_settings=settings,
        selection_function='multinomial_vocal_category_model_selection',
        selection_metric='auc',
        n_splits_selection=int(n_splits),
        test_proportion=float(test_prop),
        split_strategy=split_strategy,
        random_seed=int(random_seed),
        aic_termination_used=False,
        n_anchor_features=len(ranked_features),
        anchor_feature=ranked_features[0],
        gam_kwargs={},
        extra_knobs={
            'use_top_rank_as_anchor': bool(use_top_rank_as_anchor),
            'p_val': float(p_val),
            'bin_resizing_factor': int(bin_size),
            'temporal_basis': dict(hp['temporal_basis']),
            'target_condition': target_condition,
            # Held-out provenance (also in the embedded `_input_metadata` sibling);
            # duplicated here so the selection run-metadata is self-describing.
            'held_out_test_proportion': float(settings['model_validation']['held_out_test_proportion']),
            'held_out_session_ids': list(_held_out_session_ids),
            'n_held_out_sessions': len(_held_out_session_ids),
            'n_held_out_events': int(_held_positions.size),
            # The per-step acceptance rule and its knobs (a resume refuses another rule).
            'step_test': {
                'rule': STEP_TEST_RULE,
                'screened_features': list(ranked_features),
                'statistic': 'macro_log_likelihood_per_event',
                'alpha': float(p_val),
                'min_lag_seconds': step_cfg['min_lag_seconds'],
                'min_lag_frames': step_cfg['min_lag_frames'],
                'checkpoint_every': step_cfg['checkpoint_every'],
                'pool_workers': step_cfg['pool_workers'],
                'refit_rows': 'development_rows_as_held_out_refit',
                'traces_path': traces_path,
            },
        },
        settings_path=settings_path,
    )
    _wrap_step = _make_step_wrapper(_input_md, _univariate_md, _run_md)

    existing_steps = []
    if model_selection_dir.is_dir():
        for f_name in (p.name for p in model_selection_dir.iterdir()):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                with contextlib.suppress(ValueError):
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))

    if existing_steps:
        last_step = max(existing_steps)
        print(f"[RESUME] Restoring from Step {last_step}...")
        try:
            with Path(model_selection_dir / f"{prefix}{last_step}.pkl").open('rb') as f:
                last_res = pickle.load(f)
            _require_step_test_rule(last_res, model_selection_dir / f"{prefix}{last_step}.pkl")

            # Rebuild the state right after `last_step` from its recorded decision
            # (`_restore_selection_checkpoint`); the incumbent's per-fold AUCs are
            # kept as the step record's baseline.
            state = _restore_selection_checkpoint(
                last_res, 'mean_auc', 'se_auc',
                lambda entry: entry['folds']['metrics']['auc'],
            )
            if state is None:
                print(
                    f"[RESUME] Step {last_step} accepted a feature without scored folds "
                    "(broken checkpoint); ignoring and starting fresh."
                )
                existing_steps = []
            else:
                current_model_features = state['features']
                best_current_score = state['score']
                best_current_folds = state['folds']
                step_counter = state['next_step']
                # A search that had accepted every ranked feature stopped on that, with no
                # rejecting step; it is converged too.
                resume_converged = state['converged'] or len(current_model_features) == len(ranked_features)
                if resume_converged:
                    print(f"[RESUME] Step {last_step}: selection already converged on "
                          f"{current_model_features}; finalizing.")
                else:
                    print(f"[RESUME] Continuing from Step {step_counter} with {current_model_features}.")
        except (OSError, pickle.UnpicklingError, KeyError, EOFError) as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # A fresh start (no checkpoint, or one the resume logic rejected) moves any step
    # files of an earlier run aside before Step 0 is written.
    if not existing_steps:
        _quarantine_stale_step_files(model_selection_dir, prefix)

    # Calculate Model-Free Baseline (Step 0) if starting fresh
    if not existing_steps:
        print("\n--- Establishing Absolute Baseline (Model-Free Marginal Prior) ---")

        baseline_data = {
            'folds': {
                # `precision` is not stored (derivable from the saved
                # confusion matrix); Brier / ECE / MCC are added as
                # calibration and chance-corrected summary scores.
                'metrics': {m: [] for m in
                            ['auc', 'score', 'recall', 'f1', 'll',
                             'brier', 'ece', 'mcc']},
                'y_true': [], 'y_pred': [], 'y_probs': [], 'test_indices': [],
                'confusion_matrix': [],
                # Hyperparameter audit fields — the centroid baseline has
                # no model to tune, so these are all NaN / empty / False.
                # Kept so the per-strategy schema stays uniform with the
                # anchor and forward-selection cand_data blocks.
                'selected_lambda_smooth': [],
                'selected_l2_reg': [],
                'hyperparam_grid_audit': [],
                'hyperparams_tuned': [],
            },
            'classes': None
        }

        for tr_idx, te_idx in cv_folds:
            y_tr, y_te = y_global[tr_idx], y_global[te_idx]

            unique_classes, counts = np.unique(y_tr, return_counts=True)
            class_priors = counts / float(np.sum(counts))

            y_proba = np.tile(class_priors, (len(y_te), 1))
            majority_class = unique_classes[np.argmax(class_priors)]
            y_pred = np.full(len(y_te), majority_class)

            eps = 1e-15
            y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

            f_met = baseline_data['folds']['metrics']

            try:
                f_auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro', labels=unique_classes)
            except ValueError:
                f_auc = np.nan

            f_met['auc'].append(f_auc)
            f_met['score'].append(balanced_accuracy_score(y_te, y_pred))
            f_met['f1'].append(f1_score(y_te, y_pred, average='macro', zero_division=0))
            f_met['recall'].append(recall_score(y_te, y_pred, average='macro', zero_division=0))

            try:
                f_ll = log_loss(y_te, y_proba_clipped, labels=unique_classes)
            except ValueError:
                f_ll = np.nan
            f_met['ll'].append(f_ll)

            # Canonical-ordered probability matrix for Brier / ECE.
            probs_canonical = align_probs_to_canonical(
                y_proba, unique_classes, canonical_classes
            )
            try:
                f_met['brier'].append(brier_score_multi(y_te, probs_canonical, canonical_classes))
            except Exception:
                f_met['brier'].append(np.nan)
            try:
                f_met['ece'].append(expected_calibration_error(y_te, y_pred, probs_canonical, n_bins=settings['diagnostics']['ece_n_bins']))
            except Exception:
                f_met['ece'].append(np.nan)
            f_met['mcc'].append(safe_matthews_corrcoef(y_te, y_pred))

            baseline_data['folds']['y_true'].append(y_te)
            baseline_data['folds']['y_pred'].append(y_pred)
            baseline_data['folds']['y_probs'].append(y_proba)
            baseline_data['folds']['test_indices'].append(te_idx)
            baseline_data['folds']['confusion_matrix'].append(
                safe_confusion_matrix(y_te, y_pred, labels=canonical_classes)
            )
            baseline_data['folds']['selected_lambda_smooth'].append(float('nan'))
            baseline_data['folds']['selected_l2_reg'].append(float('nan'))
            baseline_data['folds']['hyperparam_grid_audit'].append({
                'grid_scores': {}, 'grid_ses': {},
                'argmax_pair': None,
                'one_se_applied': False, 'one_se_threshold': None,
            })
            baseline_data['folds']['hyperparams_tuned'].append(False)
            if baseline_data['classes'] is None:
                baseline_data['classes'] = unique_classes

        valid_auc = [m for m in baseline_data['folds']['metrics']['auc'] if not np.isnan(m)]
        # Guard against the all-NaN / single-fold cases the way the manifold
        # baseline does: an empty `valid_auc` would make `np.mean([])` NaN and a
        # single finite fold would make `np.std([x], ddof=1)` NaN (ddof drives the
        # divisor to zero); the logged running best must stay finite.
        best_current_score = float(np.mean(valid_auc)) if valid_auc else float('-inf')
        baseline_se_auc = float(np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))) if len(valid_auc) > 1 else 0.0
        best_current_folds = baseline_data['folds']['metrics']['auc']

        print(format_selection_step('Baseline', detail='model-free marginal prior',
                                    metrics={'AUC': best_current_score}))

        baseline_data['mean_auc'] = best_current_score
        baseline_data['se_auc'] = baseline_se_auc

        step_0_res = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': best_current_score,
            'selected_feature': 'null_model_free',
            'candidates_summary': {'null_model_free': baseline_data},
            'step_test': None,
        }

        _dump_step_pickle(model_selection_dir / f"{prefix}0.pkl", _wrap_step(step_0_res))

        step_counter = 1

    # The development refit the per-step shift test scores on: exactly the held-out
    # refit's rows and fit (every development row, class-balanced with the seed
    # `hp['random_state'] + n_splits + 1` when `balance_train` is on, the same
    # hyperparameter choice and model kwargs). The statistic is the in-sample macro
    # (class-balanced) log-likelihood per event on those rows; the empty incumbent is
    # the marginal class prior of the same rows.
    _dev_seed_mn = int(hp['random_state']) + n_splits + 1
    _y_dev_all = y_global[_dev_positions]
    _groups_dev_all = groups_global[_dev_positions]
    if balance_train:
        _dev_train_idx = _balance_multinomial_train_indices(
            np.arange(len(_y_dev_all)), _y_dev_all, np.random.default_rng(_dev_seed_mn)
        )
    else:
        _dev_train_idx = np.arange(len(_y_dev_all))
    _y_dev_fit = _y_dev_all[_dev_train_idx]
    _groups_dev_fit = _groups_dev_all[_dev_train_idx]

    def _multinomial_development_design(features: list, winner_override: dict | None = None) -> np.ndarray:
        """
        Description
        -----------
        The development design of the model with ``features`` (every development row,
        features side by side, before the balanced-row selection); with
        ``winner_override`` the LAST feature's rows are replaced session by session by
        the raw shifted windows in that per-session dict, binned and projected by
        ``_multinomial_bin_and_project`` exactly as the stored windows were, and
        scattered into a copy of the stored development rows by the rows' sessions
        (a session's development rows are its stored rows in start-frame order).

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            ``{session_id: {'X': (n_rows_of_session, history_frames) windows}}``.

        Returns
        -------
        X_dev (np.ndarray)
            ``(n_dev_rows, n_features * n_time_bins)`` float32 design.
        """

        blocks = [binned_data[f][_dev_positions] for f in features]
        if winner_override is not None:
            X_winner = blocks[-1].copy()
            for session_id, entry in winner_override.items():
                rows = np.where(_groups_dev_all == session_id)[0]
                shifted = _multinomial_bin_and_project(entry['X'].astype(np.float32), features[-1]).astype(np.float32)
                if rows.size != shifted.shape[0]:
                    msg = (f"shift override for {features[-1]!r} / {session_id!r} has {shifted.shape[0]} "
                           f"rows but the development design holds {rows.size}.")
                    raise ValueError(msg)
                X_winner[rows] = shifted
            blocks[-1] = X_winner
        return np.hstack(blocks)

    def _multinomial_development_score(features: list, winner_override: dict | None = None) -> float:
        """
        Description
        -----------
        In-sample macro (class-balanced) log-likelihood per event of the model with
        ``features`` refit on the development rows exactly as the held-out refit fits
        it. The empty model is the marginal class prior of the refit rows.

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            See ``_multinomial_development_design``.

        Returns
        -------
        score (float)
            Mean over present classes of the class-mean log-probability of the true
            class on the refit rows (higher is better).
        """

        unique_classes, counts = np.unique(_y_dev_fit, return_counts=True)
        if not features:
            class_priors = counts / float(np.sum(counts))
            return macro_log_likelihood(_y_dev_fit, np.tile(class_priors, (len(_y_dev_fit), 1)), unique_classes)
        X_dev_fit = _multinomial_development_design(features, winner_override)[_dev_train_idx]
        dev_lambda_smooth, dev_l2_reg, _, _ = _pick_multinomial_fold_hyperparams(
            X_dev_fit, _y_dev_fit, _groups_dev_fit, n_feats_=len(features), fold_idx_=n_splits,
        )
        dev_model = SmoothMultinomialLogisticRegression(
            n_features=len(features), n_time_bins=n_time_bins,
            lambda_smooth=dev_lambda_smooth,
            l2_reg=dev_l2_reg,
            smoothness_derivative_order=smoothness_order,
            smoothness_reflective_edges=reflective_edges,
            focal_gamma=model_focal_gamma,
            uniform_class_weights=model_uniform_weights,
            learning_rate=hp['learning_rate'],
            max_iter=hp['max_iter'],
            tol=hp['tol'],
            random_state=_dev_seed_mn,
            _use_lax_loop=use_lax_loop_mn,
        )
        dev_model.fit(X_dev_fit, _y_dev_fit)
        y_proba = dev_model.predict_proba(X_dev_fit, balanced=hp['balance_predictions_bool'])
        return macro_log_likelihood(_y_dev_fit, y_proba, dev_model.classes_)

    # Read-only data the candidate fits read (`_multinomial_candidate_fold`); the
    # step-specific entries are added per step.
    _multinomial_pool_base = {
        'cv_folds': cv_folds,
        'binned_data': binned_data,
        'y_global': y_global,
        'groups_global': groups_global,
        'n_time_bins': n_time_bins,
        'hp': hp,
        'temporal_basis': temporal_basis,
        'canonical_classes': canonical_classes,
        'smoothness_order': smoothness_order,
        'reflective_edges': reflective_edges,
        'model_focal_gamma': model_focal_gamma,
        'model_uniform_weights': model_uniform_weights,
        'use_lax_loop': use_lax_loop_mn,
        'ece_n_bins': settings['diagnostics']['ece_n_bins'],
        'pick_hyperparams': _pick_multinomial_fold_hyperparams,
        'class_proportions': _class_proportions,
    }

    def _multinomial_fit_candidates(incumbent: list, candidate_features: list) -> dict:
        """
        Description
        -----------
        Fits every candidate on every fold on top of ``incumbent`` and returns each
        candidate's results entry. The fold's balanced-train index selection and the
        incumbent's stacked base block are built once per fold here (they depend on
        the fold, not on the candidate); the candidate rows are then stacked on.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.
        candidate_features (list)
            Candidates to score.

        Returns
        -------
        per_candidate (dict)
            ``{feature: cand_data}``.
        """

        fold_tr_idx_model, fold_base_tr, fold_base_te = [], [], []
        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            tr_idx_model = _fold_train_indices_for_model(tr_idx, fold_idx)
            fold_tr_idx_model.append(tr_idx_model)
            fold_base_tr.append(
                np.hstack([binned_data[f][tr_idx_model] for f in incumbent])
                if incumbent else None
            )
            fold_base_te.append(
                np.hstack([binned_data[f][te_idx] for f in incumbent])
                if incumbent else None
            )
        set_pool_context(
            multinomial_step={**_multinomial_pool_base,
                              'current_model_features': list(incumbent),
                              'fold_tr_idx_model': fold_tr_idx_model,
                              'fold_base_tr': fold_base_tr, 'fold_base_te': fold_base_te},
            candidate_fold=_multinomial_candidate_fold,
        )
        # The worker count is derived as in the other selectors but the pool runs with
        # ONE worker: JAX (XLA's thread pool) is not fork-safe once the backend has run
        # in this process, and a single fit already uses every core.
        n_workers_requested = pool_worker_count(step_cfg['pool_workers'], len(incumbent) + 1,
                                                step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                                step_cfg['fit_peak_gb_per_term'])
        del n_workers_requested
        n_folds = len(cv_folds)
        tasks = [(feat, fold_idx) for feat in candidate_features for fold_idx in range(n_folds)]
        pool = SelectionPool(1)
        try:
            results = pool.map('candidate_fold', tasks)
        finally:
            pool.close()
        return {feat: _multinomial_assemble_cand_data(results[i * n_folds:(i + 1) * n_folds], feat,
                                                      canonical_classes, balance_train, n_splits)
                for i, feat in enumerate(candidate_features)}

    def _multinomial_step_test(winner: str, step_index: int, m_candidates: int) -> dict:
        """
        Description
        -----------
        The per-step shift test of ``winner`` on top of the current incumbent.

        Parameters
        ----------
        winner (str)
            The cross-validated winner.
        step_index (int)
            The step.
        m_candidates (int)
            Candidates the winner was chosen from (the Bonferroni factor).

        Returns
        -------
        step_test (dict)
            The record of ``run_step_shift_test``.
        """

        incumbent = list(current_model_features)
        trial = [*incumbent, winner]
        incumbent_score = _multinomial_development_score(incumbent)
        observed_score = _multinomial_development_score(trial)
        winner_traces = {s: read_feature_trace(traces_path, winner, s) for s in _dev_session_ids}
        # JAX is not fork-safe once the backend has run in this process: the null
        # refits run serially (one worker), see `_multinomial_fit_candidates`.
        return run_step_shift_test(
            n_workers=1,
            refit_and_score=lambda override: _multinomial_development_score(trial, override),
            incumbent_features=incumbent,
            incumbent_score=incumbent_score,
            winner=winner,
            observed_score=observed_score,
            statistic_name='macro_log_likelihood_per_event',
            winner_traces=winner_traces,
            window_starts={s: window_starts[s] for s in winner_traces},
            slots=_SESSION_WINDOW_SLOTS,
            history_frames=history_frames,
            m_candidates=m_candidates,
            alpha=p_val,
            min_lag_frames=step_cfg['min_lag_frames'],
            random_seed=random_seed,
            step_index=step_index,
            checkpoint_every=step_cfg['checkpoint_every'],
            partial_path=model_selection_dir / f"{prefix}{step_index}_null_partial.pkl",
            label=f"Step {step_index}",
        )

    # Auto-anchor logic
    if use_top_rank_as_anchor and step_counter == 1 and not resume_converged:
        anchor = ranked_features[0]
        print(format_selection_step('Anchor', feature=anchor,
                                    detail='top-ranked; forced as the first candidate, tested against the marginal prior'))
        cand_data = _multinomial_fit_candidates([], [anchor])[anchor]
        valid_auc = [m for m in cand_data['folds']['metrics']['auc'] if np.isfinite(m)]
        # Surface PARTIAL fold failure: the anchor's mean is quietly averaged over
        # only the surviving folds and the run still reports success. A large
        # bad-fold fraction signals numerical instability (e.g. a rare, heavily
        # inverse-frequency-weighted class fit under near-zero regularisation).
        _n_anchor_folds = len(cand_data['folds']['metrics']['auc'])
        _n_anchor_bad = _n_anchor_folds - len(valid_auc)
        if 0 < _n_anchor_bad < _n_anchor_folds:
            print(
                f"  [!] WARNING: anchor '{anchor}' produced a non-finite AUC on "
                f"{_n_anchor_bad}/{_n_anchor_folds} folds (fit failed / diverged); those folds "
                f"are dropped from its mean AUC. A large fraction indicates numerical "
                f"instability — check class balance and the l2 grid floor."
            )
        _require_scored_candidate('Anchor', 1, len(valid_auc))
        step_test = None
        if valid_auc:
            mean_anc_auc = float(np.mean(valid_auc))
            se_anc_auc = float(np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))) if len(valid_auc) > 1 else 0.0
            cand_data['mean_auc'] = mean_anc_auc
            cand_data['se_auc'] = se_anc_auc
            # The forced anchor is tested like every winner, with Bonferroni over every
            # screened feature it was forced ahead of.
            step_test = _multinomial_step_test(anchor, 1, len(ranked_features))
        anchor_accepted = step_test is not None and step_test['accepted']
        step_1_res = {
            'step_idx': 1,
            'current_features': [anchor] if anchor_accepted else [],
            'baseline_score': best_current_score,
            'baseline_folds': best_current_folds,
            'selected_feature': anchor if anchor_accepted else None,
            'candidates_summary': {anchor: cand_data} if valid_auc else {},
            'step_test': step_test,
        }
        _dump_step_pickle(model_selection_dir / f"{prefix}1.pkl", _wrap_step(step_1_res))
        if anchor_accepted:
            print(format_selection_step('Anchor', feature=anchor, decision='ACCEPT',
                                        metrics={'AUC': cand_data['mean_auc'], 'p_adj': step_test['p_adjusted']}))
            best_current_score = cand_data['mean_auc']
            best_current_folds = cand_data['folds']['metrics']['auc']
            current_model_features = [anchor]
            step_counter = 2
        else:
            print(format_selection_step('Anchor', feature=anchor, decision='REJECT',
                                        detail=('every fold errored out' if not valid_auc
                                                else 'the forced anchor failed the shift test')
                                               + '; selection finished'))
            resume_converged = True

    # Main Forward Selection Loop
    # A resume that restored a converged search skips the loop and finalizes directly.
    while not resume_converged:
        print(format_selection_step(f"Step {step_counter}",
                                    detail=f"best so far AUC={best_current_score:.4f}"))
        step_results = {
            'step_idx': step_counter,
            'current_features': list(current_model_features),
            'baseline_score': best_current_score,
            'baseline_folds': best_current_folds,
            'candidates_summary': {},
            'selected_feature': None,
            'step_test': None,
        }
        candidates_this_step = [feat for feat in ranked_features if feat not in current_model_features]
        m_candidates = len(candidates_this_step)
        per_candidate = _multinomial_fit_candidates(current_model_features, candidates_this_step)

        best_cand, best_cand_score = None, 0.0
        best_cand_folds = None
        for feat in candidates_this_step:
            cand_data = per_candidate[feat]
            valid_auc = [x for x in cand_data['folds']['metrics']['auc'] if np.isfinite(x)]
            # Surface PARTIAL fold failure for this candidate (some folds
            # diverged, others survived): otherwise the candidate's mean AUC is
            # silently averaged over only the surviving folds. See the matching
            # anchor guard above for the rationale.
            _n_cand_folds = len(cand_data['folds']['metrics']['auc'])
            _n_cand_bad = _n_cand_folds - len(valid_auc)
            if 0 < _n_cand_bad < _n_cand_folds:
                print(
                    f"  [!] WARNING: candidate '{feat}' produced a non-finite AUC on "
                    f"{_n_cand_bad}/{_n_cand_folds} folds (fit failed / diverged); those folds "
                    f"are dropped from its mean AUC."
                )
            if not valid_auc:
                print(format_selection_step(f"Step {step_counter}", feature=feat,
                                            decision='FAILED (no finite folds)'))
                continue

            mean_auc = np.mean(valid_auc)
            # `np.std([x], ddof=1)` is NaN for a single finite fold (zero divisor);
            # guard with len > 1 so a one-fold success keeps a finite SE.
            se_auc = np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc)) if len(valid_auc) > 1 else 0.0
            print(format_selection_step(f"Step {step_counter}", feature=feat,
                                        metrics={'AUC': mean_auc}))
            cand_data['mean_auc'], cand_data['se_auc'] = mean_auc, se_auc
            step_results['candidates_summary'][feat] = cand_data
            if mean_auc > best_cand_score:
                best_cand_score, best_cand = mean_auc, feat
                best_cand_folds = cand_data['folds']['metrics']['auc']

        # The cross-validated winner is tested with the per-step shift test, Bonferroni
        # over the m candidates of this step; the step is accepted only when no shifted
        # refit reaches the observed in-sample gain.
        _require_scored_candidate(f"Step {step_counter}", m_candidates, int(best_cand is not None))
        step_test = (_multinomial_step_test(best_cand, step_counter, m_candidates)
                     if best_cand is not None else None)
        step_results['step_test'] = step_test

        if step_test is not None and step_test['accepted']:
            print(format_selection_step(f"Step {step_counter}", feature=best_cand,
                                        decision='ACCEPT',
                                        metrics={'AUC': best_cand_score, 'p_adj': step_test['p_adjusted']}))
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)
            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results))
            best_current_score, step_counter = best_cand_score, step_counter + 1
            best_current_folds = best_cand_folds
        else:
            if best_cand is None:
                reject_detail = 'no candidate had finite folds; selection finished'
            else:
                reject_detail = (f"winner {best_cand!r} failed the shift test "
                                 f"(p_adj {step_test['p_adjusted']:.4f} vs alpha {p_val}); selection finished")
            print(format_selection_step(f"Step {step_counter}", decision='REJECT', detail=reject_detail))
            step_results['selected_feature'] = None
            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results))
            break

        if len(current_model_features) == len(ranked_features):
            break

    # 7. Final Data Promotion for Visualization
    print("\n--- Finalizing Results for Visualization ---")
    # No blanket except: a failed final refit or save must stop the run loudly, not
    # pass as a finished selection with no final model.
    last_file_path = model_selection_dir / f"{prefix}{max(0, step_counter - 1)}.pkl"
    with Path(last_file_path).open('rb') as f:
        step_data = pickle.load(f)

    winner = step_data['selected_feature'] or current_model_features[-1]
    winning_cand_data = step_data['candidates_summary'][winner]

    # Only process reshaping if the winning candidate was a real model with weights
    if winner != 'null_model_free':
        # CV folds that hit a fitting exception are recorded as
        # `None` in `cand_data['folds']['weights']`. `np.array` on
        # a mixed list of `None` + 2D arrays raises an
        # inhomogeneous-shape error under numpy>=1.24, so swap
        # each failed-fold `None` for a same-shape NaN block
        # taken from the first successful fold. Downstream
        # consumers of `weights_reshaped` should already treat
        # NaNs as missing folds.
        weights_list = list(winning_cand_data['folds']['weights'])
        first_valid = next((w for w in weights_list if w is not None), None)
        if first_valid is None:
            reshaped_weights = None
        else:
            nan_block = np.full_like(first_valid, np.nan, dtype=float)
            weights_list = [
                nan_block if w is None else w for w in weights_list
            ]
            raw_weights = np.array(weights_list)
            n_folds, n_classes, _ = raw_weights.shape
            n_final_feats = len(current_model_features)
            # The stored weights are on the frame axis (B-spline coefficients are
            # converted back at storage), so the per-feature length is derived
            # from them rather than from `n_time_bins` (the fitted column count).
            reshaped_weights = raw_weights.reshape(
                n_folds, n_classes, n_final_feats, -1,
            )
    else:
        reshaped_weights = None

    # Honest held-out score: refit the accepted feature set on ALL
    # development sessions and score it once on the reserved held-out block,
    # mirroring the per-fold candidate fit (train := development, test :=
    # held-out). This is the only number untouched by screening, greedy
    # acceptance, or lambda-tuning -- every one of those saw development data
    # only. When the winner is `null_model_free` the honest model is the
    # empirical development prior. Skipped when holdout is disabled
    # (`_held_positions` empty), leaving `heldout` None so no-holdout runs are
    # byte-identical to before.
    heldout_result = None
    if _held_positions.size > 0:
        y_dev = y_global[_dev_positions]
        y_held = y_global[_held_positions]
        held_seed_mn = int(hp['random_state']) + n_splits + 1

        if winner != 'null_model_free' and len(current_model_features) > 0:
            n_final_feats = len(current_model_features)
            X_dev = np.hstack([binned_data[f][_dev_positions] for f in current_model_features])
            X_held = np.hstack([binned_data[f][_held_positions] for f in current_model_features])

            # Mirror the fold body's optional class balancing on the training
            # (development) rows.
            if balance_train:
                balance_rng = np.random.default_rng(held_seed_mn)
                dev_train_idx = _balance_multinomial_train_indices(
                    np.arange(len(y_dev)), y_dev, balance_rng
                )
                X_dev_fit = X_dev[dev_train_idx]
                y_dev_fit = y_dev[dev_train_idx]
            else:
                X_dev_fit = X_dev
                y_dev_fit = y_dev

            final_lambda_smooth, final_l2_reg, final_grid_audit, final_tuned_flag = (
                _pick_multinomial_fold_hyperparams(
                    X_dev_fit, y_dev_fit,
                    (groups_global[_dev_positions][dev_train_idx] if balance_train
                     else groups_global[_dev_positions]),
                    n_feats_=n_final_feats,
                    fold_idx_=n_splits,
                )
            )
            held_model = SmoothMultinomialLogisticRegression(
                n_features=n_final_feats, n_time_bins=n_time_bins,
                lambda_smooth=final_lambda_smooth,
                l2_reg=final_l2_reg,
                smoothness_derivative_order=smoothness_order,
                smoothness_reflective_edges=reflective_edges,
                focal_gamma=model_focal_gamma,
                uniform_class_weights=model_uniform_weights,
                learning_rate=hp['learning_rate'],
                max_iter=hp['max_iter'],
                tol=hp['tol'],
                random_state=held_seed_mn,
                _use_lax_loop=use_lax_loop_mn,
            )
            held_model.fit(X_dev_fit, y_dev_fit)
            y_proba = held_model.predict_proba(X_held, balanced=hp['balance_predictions_bool'])
            y_pred = held_model.predict(X_held, balanced=hp['balance_predictions_bool'])
            model_classes = held_model.classes_
            held_weights = (basis_coefficients_to_frames(held_model.coef_, temporal_basis, n_features=n_final_feats, axis=1) if temporal_basis is not None else held_model.coef_)
            held_basis_coefficients = (np.asarray(held_model.coef_) if temporal_basis is not None else None)
            held_intercepts = held_model.intercept_
            held_n_iter = int(held_model.n_iter_)
            held_converged = bool(held_model.converged_)
            held_fit_time = float(held_model.fit_time_)
            p_train_labels = y_dev_fit
            model_type = 'multinomial_logistic'
            # No-signal filter band: the same model and hyperparameters refit on
            # the development rows with the category labels permuted across the
            # development set; the filter figures draw the final filters against it.
            groups_dev_fit = (groups_global[_dev_positions][dev_train_idx] if balance_train
                              else groups_global[_dev_positions])
            n_null_permutations = int(settings['model_params']['filter_null_permutations'])
            null_rng = np.random.default_rng(held_seed_mn)
            held_null_weights = []
            for _ in range(n_null_permutations):
                perm = null_target_permutation(groups_dev_fit.shape[0], null_rng)
                null_model = SmoothMultinomialLogisticRegression(
                    n_features=n_final_feats, n_time_bins=n_time_bins,
                    lambda_smooth=final_lambda_smooth,
                    l2_reg=final_l2_reg,
                    smoothness_derivative_order=smoothness_order,
                    smoothness_reflective_edges=reflective_edges,
                    focal_gamma=model_focal_gamma,
                    uniform_class_weights=model_uniform_weights,
                    learning_rate=hp['learning_rate'],
                    max_iter=hp['max_iter'],
                    tol=hp['tol'],
                    random_state=held_seed_mn,
                    _use_lax_loop=use_lax_loop_mn,
                )
                null_model.fit(X_dev_fit, y_dev_fit[perm])
                held_null_weights.append(basis_coefficients_to_frames(null_model.coef_, temporal_basis, n_features=n_final_feats, axis=1)
                                         if temporal_basis is not None else np.asarray(null_model.coef_))
            held_null_weights = np.stack(held_null_weights).astype(np.float32) if held_null_weights else None
        else:
            # No feature accepted: the empirical development prior is the model.
            unique_classes, counts = np.unique(y_dev, return_counts=True)
            class_priors = counts / float(np.sum(counts))
            y_proba = np.tile(class_priors, (len(y_held), 1))
            majority_class = unique_classes[np.argmax(class_priors)]
            y_pred = np.full(len(y_held), majority_class)
            model_classes = unique_classes
            held_weights = None
            held_basis_coefficients = None
            held_intercepts = None
            held_n_iter = 0
            held_converged = True
            held_fit_time = 0.0
            held_null_weights = None
            final_lambda_smooth = float('nan')
            final_l2_reg = float('nan')
            final_grid_audit = {
                'grid_scores': {}, 'grid_ses': {},
                'argmax_pair': None,
                'one_se_applied': False, 'one_se_threshold': None,
            }
            final_tuned_flag = False
            p_train_labels = y_dev
            model_type = 'null_model_free'

        eps = 1e-15
        y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
        h_score = balanced_accuracy_score(y_held, y_pred)
        try:
            h_ll = log_loss(y_held, y_proba_clipped, labels=model_classes)
        except ValueError:
            h_ll = np.nan
        try:
            h_auc = roc_auc_score(y_held, y_proba, multi_class='ovr', average='macro', labels=model_classes)
        except ValueError:
            h_auc = np.nan
        probs_canonical = align_probs_to_canonical(y_proba, model_classes, canonical_classes)
        try:
            h_brier = brier_score_multi(y_held, probs_canonical, canonical_classes)
        except Exception:
            h_brier = np.nan
        try:
            h_ece = expected_calibration_error(y_held, y_pred, probs_canonical, n_bins=settings['diagnostics']['ece_n_bins'])
        except Exception:
            h_ece = np.nan
        h_mcc = safe_matthews_corrcoef(y_held, y_pred)

        heldout_result = {
            'model_type': model_type,
            'features': list(current_model_features),
            'metrics': {
                'score': h_score, 'll': h_ll, 'auc': h_auc,
                'recall': recall_score(y_held, y_pred, average='macro', zero_division=0),
                'f1': f1_score(y_held, y_pred, average='macro', zero_division=0),
                'brier': h_brier, 'ece': h_ece, 'mcc': h_mcc,
            },
            'weights': held_weights,
            'null_weights': held_null_weights,
            'basis_coefficients': held_basis_coefficients,
            'intercepts': held_intercepts,
            'y_true': y_held,
            'y_pred': y_pred,
            'y_probs': y_proba,
            'test_indices': _held_positions,
            'p_train': _class_proportions(p_train_labels),
            'p_test': _class_proportions(y_held),
            'confusion_matrix': safe_confusion_matrix(y_held, y_pred, labels=canonical_classes),
            'n_iter': held_n_iter,
            'converged': held_converged,
            'fit_time': held_fit_time,
            'selected_lambda_smooth': final_lambda_smooth,
            'selected_l2_reg': final_l2_reg,
            'hyperparam_grid_audit': final_grid_audit,
            'hyperparams_tuned': final_tuned_flag,
            'n_held_out_sessions': len(_held_out_session_ids),
            'n_held_out_events': int(_held_positions.size),
        }

    step_data.update({
        'final_model_features': current_model_features,
        'weights_reshaped': reshaped_weights,
        'classes': winning_cand_data['classes'],
        'heldout': heldout_result,
    })
    for _k in RESERVED_METADATA_KEYS:
        step_data.pop(_k, None)
    _dump_step_pickle(last_file_path, _wrap_step(step_data))

    baseline_score = float('nan')
    step0_path = model_selection_dir / f"{prefix}0.pkl"
    if step0_path.exists():
        with Path(step0_path).open('rb') as f:
            _step0 = pickle.load(f)
        if 'baseline_score' in _step0:
            baseline_score = _step0['baseline_score']
    metrics_by_strategy = {'Final': {'AUC': best_current_score},
                           'Baseline': {'AUC': baseline_score}}
    # Held-out AUC (development-refit, scored once) -- the honest number
    # alongside the optimistic development-CV `Final`.
    if heldout_result is not None:
        metrics_by_strategy['Held-out'] = {'AUC': heldout_result['metrics']['auc']}
    print(format_run_summary(
        label=f"multinomial model-selection ({len(current_model_features)} features)",
        metrics_by_strategy=metrics_by_strategy,
        out_path=str(last_file_path),
    ))

def _manifold_candidate_fold(feat: str, fold_idx: int) -> dict:
    """
    Description
    -----------
    One (candidate, fold) fit of the continuous manifold selector: fits the incumbent
    features plus ``feat`` on the fold's training rows with the geometry's JAX
    regressor (equal-region reweighting on the torus) and scores the fold's test
    rows with the full metric bundle. The data comes from the ``'manifold_step'``
    pool-context entry the selector stores before running the fits (``cv_folds``,
    ``binned_data``, ``y_global``, ``w_global``, ``groups_global``, ``region_global``,
    the per-fold cached ``fold_base_tr`` / ``fold_base_te``,
    ``current_model_features``, the hyperparameter closure ``pick_hyperparams``, the
    regressor class, the scoring knobs and the geodesic context). The operations are
    those of the former inline fold body, so the fold scores are the same numbers.

    Parameters
    ----------
    feat (str)
        The candidate feature.
    fold_idx (int)
        Fold index.

    Returns
    -------
    result (dict)
        ``{'ok': True, 'fold': {...}}`` with one entry per per-fold list of the
        candidate record, or ``{'ok': False, 'error': str}``.
    """

    ctx = pool_context('manifold_step')
    tr_idx, te_idx = ctx['cv_folds'][fold_idx]
    binned_data = ctx['binned_data']
    hp = ctx['hp']
    temporal_basis = ctx['temporal_basis']
    manifold_metric = ctx['manifold_metric']
    trial_feats = [*ctx['current_model_features'], feat]
    n_trial_feats = len(trial_feats)
    try:
        new_tr = binned_data[feat][tr_idx]
        new_te = binned_data[feat][te_idx]
        base_tr = ctx['fold_base_tr'][fold_idx]
        base_te = ctx['fold_base_te'][fold_idx]
        X_tr_stacked = np.hstack([base_tr, new_tr]) if base_tr is not None else new_tr
        X_te_stacked = np.hstack([base_te, new_te]) if base_te is not None else new_te
        Y_tr, Y_te = ctx['y_global'][tr_idx], ctx['y_global'][te_idx]
        w_tr, w_te = ctx['w_global'][tr_idx], ctx['w_global'][te_idx]
        groups_tr = ctx['groups_global'][tr_idx]
        region_tr, region_te = ctx['region_global'][tr_idx], ctx['region_global'][te_idx]

        # Equal-region fit reweighting (torus only): weight training rows by inverse
        # acoustic-region frequency so common regions don't dominate the fit.
        # Euclidean keeps the raw weights.
        if manifold_metric == 'torus':
            w_tr_fit = w_tr * inverse_region_frequency_weights(region_tr)
        else:
            w_tr_fit = w_tr

        fold_lambda_smooth, fold_l2_reg, fold_grid_audit, fold_tuned_flag = (
            ctx['pick_hyperparams'](
                X_tr_stacked, Y_tr, w_tr, groups_tr,
                n_feats_=n_trial_feats,
                fold_idx_=fold_idx,
                region_tr_=region_tr,
            )
        )

        model = ctx['regressor_cls'](
            n_features=n_trial_feats, n_time_bins=ctx['n_time_bins'],
            lambda_smooth=fold_lambda_smooth, l2_reg=fold_l2_reg,
            smoothness_derivative_order=ctx['smoothness_order'],
            smoothness_reflective_edges=ctx['reflective_edges'],
            huber_delta=hp['huber_delta'],
            learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
            tol=hp['tol'], random_state=hp['random_state'] + fold_idx,
            _use_lax_loop=ctx['use_lax_loop'],
            metric=manifold_metric, period=ctx['manifold_period'],
        )
        model.fit(X_tr_stacked, Y_tr, sample_weight=w_tr_fit)

        metrics = model.evaluate_metrics(
            X_te_stacked, Y_te, weights=w_te,
            region_labels=region_te, min_region_events=ctx['min_region_events'],
            kappa=ctx['kappa_frozen'], dcor_n_sub=ctx['dcor_n_sub'], dcor_n_rep=ctx['dcor_n_rep'],
        )
        y_pred_xy = model.predict(X_te_stacked, snap=True).astype(np.float32)
        metrics.update(geodesic_mae_columns(
            y_pred_xy, Y_te, ctx['geodesic_ctx'],
            region_labels=region_te, min_region_events=ctx['min_region_events']))

        fold = {
            'metrics': {_mk: metrics[_mk] for _mk in ctx['metric_keys']},
            'weights': (basis_coefficients_to_frames(model.coef_, temporal_basis, n_features=n_trial_feats, axis=0)
                        if temporal_basis is not None else model.coef_),
            'basis_coefficients': np.asarray(model.coef_) if temporal_basis is not None else None,
            'intercepts': model.intercept_,
            'test_indices': te_idx,
            'y_true': Y_te,
            'w_test': w_te,
            'y_pred_xy': y_pred_xy,
            'n_iter': int(model.n_iter_),
            'converged': bool(model.converged_),
            'fit_time': float(model.fit_time_),
            'selected_lambda_smooth': fold_lambda_smooth,
            'selected_l2_reg': fold_l2_reg,
            'hyperparam_grid_audit': fold_grid_audit,
            'hyperparams_tuned': fold_tuned_flag,
        }
        del model, X_tr_stacked, X_te_stacked
        gc.collect()
        return {'ok': True, 'fold': fold}
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


def continuous_vocal_manifold_model_selection(
        univariate_results_path: str,
        input_data_path: str,
        output_directory: str,
        settings_path: str | None = None,
        use_top_rank_as_anchor: bool = False,
        *,
        p_val: float
) -> None:
    """
    Performs forward stepwise selection for continuous manifold-position
    prediction using the geometry-resolved estimator
    (`SmoothBivariateRegression` on Euclidean manifolds,
    `SmoothTorusManifoldRegression` on the torus).

    The selector identifies the minimal set of behavioural features that
    jointly predict the `(x, y)` manifold coordinates of upcoming USVs.
    Candidates are ranked by the geometry-appropriate selection score
    (`SELECTION_SCORE_KEY`, `manifold_metric.resolve_manifold_selection_score_key`):
    the von Mises log-score on the TORUS manifold — the region-balanced macro
    `vm_logscore` by default, or the event-weighted `vm_logscore_pooled` twin
    when `vocal_features.usv_manifold_selection_score` is `'micro'` — and the
    wrap-aware distance correlation `dcor_xy` on Euclidean manifolds. The
    centroid-referenced `r2_spatial` is structurally inverted on the
    near-uniform periodic torus and is a reported descriptor only.

    Key algorithmic choices
    -----------------------
    1. Manifold-bounded predictions. On Euclidean manifolds every
       candidate prediction is snapped to the nearest observed training
       point before metrics are computed, so the active model and the
       baselines compete on the same support and the magnitudes are
       directly comparable (see `SmoothBivariateRegression.predict`).
       The torus estimator decodes to a coordinate that is always on the
       manifold, so it deliberately does not snap.
    2. Fold-grain paired-margin screening (higher-is-better). For every
       feature the per-fold margin of `actual` over the Y-permutation
       `null` (`SCREEN_BASELINE_STRATEGY`) on the selection
       score is tested over folds (`_fold_paired_margin_bootstrap`): the
       one-sided exact sign-flip permutation p-values (`_sign_flip_p_value`,
       floor `1 / 2**manifold_n_cv_folds`, so 20 folds let a single feature clear the
       BH rank-1 threshold `q / m`) are Benjamini-Hochberg controlled at
       `q = p_val`, and a surviving feature must also clear a relative
       effect floor (`selection_effect_floor` x the top surviving margin).
       The permutation across the training fold destroys every
       behaviour->position pairing, within and between sessions. On the torus the screen scores
       every fold with the same rule as the greedy gate -- the selection
       score's own averaging (the region-balanced macro score by default,
       matching the equal-region fit reweighting) and, with
       `freeze_selection_kappa`, the one frozen concentration -- so the
       input pickle is loaded before the screen there (once; the binning
       reuses it); on euclidean the screen still runs before any load.
    3. Spatial stratification. Folds come from
       `get_stratified_spatial_splits_stable`, which uses deterministic
       K-Means geographic clustering so rare acoustic satellites are
       proportionally represented across train / test halves under both
       `session` and `mixed` strategies.
    4. Inverse-density weighting. The pre-computed KDE spatial weights
       `w` are forwarded to the JAX optimiser so gradient updates give
       satellite vocalisations the same pull as dense-core bouts.
    5. Step-0 gate baseline and chance floor. On the TORUS the anchor / first
       forward feature must beat `null_model_intercept`, a training-fold
       INTERCEPT-ONLY predictor (every test event predicted at the circular
       mean of the fold's training `Y`, weighted like the candidates' fit),
       scored on the test folds with the same bundle and gated with the same
       per-fold margin as every candidate, so a model without behavioural
       information ties it and is rejected. `null_model_free` predicts a
       uniform random draw from the training `Y` for every test trial — the
       chance floor (a sample from the observed repertoire, ignoring
       kinematics), stored in the Step-0 file (`chance_floor_score`) and
       reported, never gating on the torus. On EUCLIDEAN the selection score
       is distance correlation, which is 0 for any constant prediction while
       non-constant predictions carry its positive bias, so the bias-matched
       draw stays the gate baseline there. The Step-0 record keeps
       `selected_feature = 'null_model_free'` as its marker and names the gate
       baseline in `baseline_strategy`.
    6. Per-step shift test. Candidates are RANKED by their mean selection score
       (higher-is-better on both geometries); the step winner is then tested with
       ``step_shift_test.run_step_shift_test``: the incumbent and the incumbent plus
       the winner are refit on the development rows exactly as the held-out refit
       fits them, the statistic is the gain in the selection score computed
       in-sample on those rows (the macro von Mises log-score on the torus, the
       distance correlation on euclidean), and its null comes from refits after
       circularly shifting the winner's processed trace within every development
       session and re-cutting (and re-binning / re-projecting) its history windows
       at the stored start frames, ``floor(m / alpha)`` times with Bonferroni
       correction over the ``m`` candidates of the step. The first feature is
       tested the same way against the Step-0 gate baseline refit on the
       development rows (the intercept-only circular mean on the torus, the
       empirical-density draw on euclidean).

    Joint per-fold hyperparameter tuning
    ------------------------------------
    `hyperparameters.linear_models.manifold_regression.tune_regularization_bool`
    is honoured on BOTH geometries. The inner-CV objective is the geometry's
    own selection score (`vm_logscore` on the torus, in its pooled label-free
    form; `dcor_xy` on euclidean), so it responds to the smoothness penalty
    and the tuner picks a meaningful pair rather than fitting noise on a
    flat surface. When it is `true`, every candidate fold (anchor + every
    forward-selection trial) runs its own joint inner CV over the log-spaced
    `(lambda_smooth, l2_reg)` grids before the outer fit; under a B-spline
    `temporal_basis` the grid is the basis's own `lambda_smooth` decades with
    the L2 grid pinned to the basis's `l2_reg_fixed` (`resolve_temporal_basis`). The l2 grid is
    rescaled by `1 / sqrt(n_trial_features)` so the search window tracks
    the same effective regularisation strength as the fixed fallback
    (which is rescaled the same way). The winning pair, the full grid of
    inner-CV scores, and a `hyperparams_tuned` flag are persisted per
    candidate per fold alongside the filter weights. Setting the flag to
    `false` preserves the legacy single-fixed-centre behaviour and
    writes the same schema.

    Deep storage for post-hoc visualisation
    ---------------------------------------
    Every step persists, for each candidate and fold: the full
    `evaluate_metrics` bundle, the learned `coef_` / `intercept_`
    matrices (`weights`, always on the frame axis; under a B-spline
    `temporal_basis` the raw spline coefficients are stored beside them as
    `basis_coefficients`, else None), the test coordinates `y_true`, the
    manifold-snapped predictions `y_pred_xy`, the test-fold weights `w_test`, the per-
    fold JAX optimiser diagnostics (`n_iter`, `converged`, `fit_time`),
    and the hyperparameter audit trail (`selected_lambda_smooth`,
    `selected_l2_reg`, `hyperparam_grid_audit`, `hyperparams_tuned`).
    The audit dict contains per-fold `grid_scores` / `grid_ses` maps,
    the raw performance `argmax_pair`, and flags recording whether the
    1-SE interpretability rule softened the final choice.

    Parameters
    ----------
    univariate_results_path : str
        Path to the univariate regression results pickle file containing the
        paired actual / null per-fold metric arrays.
    input_data_path : str
        Path to the extracted manifold data containing X (history), Y (manifold position),
        and w (KDE spatial weights).
    output_directory : str
        Directory to save the step-wise state dictionaries.
    settings_path : str, optional
        Path to the JSON settings file.
    use_top_rank_as_anchor : bool, default False
        If True, initialises the search by forcing the single highest-ranked
        screened feature (largest fold-mean paired margin) as Step 1.
    p_val : float
        Keyword-only, no default: the dispatcher passes ``model_params.selection_p_val``.
        The Benjamini-Hochberg false-discovery rate `q` of the screen's
        one-sided sign-flip permutation p-values.

    Returns
    -------
    None
        Writes one pickle per selection step and the final consolidated
        selection pickle into ``output_directory``.
    """

    if settings_path is None:
        settings_path_obj = Path(__file__).resolve().parent.parent / '_parameter_settings/modeling_settings.json'
        settings_path = str(settings_path_obj)

    try:
        with Path(settings_path).open() as f:
            settings = json.load(f)
    except FileNotFoundError as err:
        msg = f"Settings file not found at {settings_path}"
        raise FileNotFoundError(msg) from err

    model_selection_dir = Path(output_directory)
    model_selection_dir.mkdir(parents=True, exist_ok=True)

    # Standardized run header (files used + config). The continuous/manifold
    # task always runs the JAX estimators, so the engine label is 'jax'.
    print(format_run_header(
        task='SELECTION:CONTINUOUS',
        engine='jax',
        feature=Path(univariate_results_path).stem,
        input_files={'univariate results': univariate_results_path,
                     'input data': input_data_path,
                     'settings': settings_path},
        output_directory=str(model_selection_dir),
        split_strategy=settings['model_validation']['split_strategy'],
        n_splits=int(settings['model_validation']['manifold_n_cv_folds']),
    ))

    with Path(univariate_results_path).open('rb') as f:
        univariate_data = pickle.load(f)

    # Strip upstream metadata blocks before iterating feature entries.
    _univ_pre_md = univariate_data.pop('_run_metadata', None)
    _require_analysis_type(univariate_data, 'continuous', univariate_results_path)
    _input_pre_md_from_univ = univariate_data.pop('_input_metadata', None)
    univariate_data.pop('_consolidation_metadata', None)

    # Per-region event floor for the torus macro von Mises score. On EUCLIDEAN
    # runs no input pickle is touched before the screen, so it can still abort or
    # raise without ever loading `input_data_path`; on the TORUS the input is
    # loaded just before the screen, which needs its per-event regions and the
    # development events' marginal dispersion (see below).
    _min_region_events = settings['vocal_features']['usv_manifold_min_region_events']
    # Fold-grain selection-gate knobs (bootstrap CI, sign-flip p-value, per-fold
    # event floor and the euclidean distance-correlation subsampling), read from
    # this run's settings so a non-default settings file drives every gate.
    _selection_effect_floor = settings['model_params']['selection_effect_floor']
    _gate_kwargs = {
        'n_bootstrap': settings['model_params']['selection_n_bootstrap'],
        'ci_level': settings['model_params']['selection_ci_level'],
        'min_fold_events': settings['model_params']['selection_min_fold_events'],
        'sign_flip_exact_max_folds': settings['model_params']['selection_sign_flip_exact_max_folds'],
        'sign_flip_monte_carlo_draws': settings['model_params']['selection_sign_flip_monte_carlo_draws'],
        'dcor_n_sub': settings['vocal_features']['usv_manifold_dcor']['n_subsample'],
        'dcor_n_rep': settings['vocal_features']['usv_manifold_dcor']['n_repeats_fold_gate'],
    }
    _dcor_n_sub = settings['vocal_features']['usv_manifold_dcor']['n_subsample']
    _dcor_n_rep = settings['vocal_features']['usv_manifold_dcor']['n_repeats']

    # Geometry-conditional selection score, carried under the reused numeric key
    # `dcor_xy` (schema stability) but with geometry-dependent CONTENTS:
    #   - torus: the macro (per-region) von Mises log-likelihood
    #     (`macro_von_mises_logscore`). The QLVM torus is a near-uniform,
    #     fully-covered periodic latent space, so the centroid-referenced
    #     `r2_spatial` is structurally inverted there; distance correlation
    #     recovers signal but is non-negative, has no true zero, and is flat in
    #     the ridge/smoothness strength (so it can be neither tuned nor compared
    #     to a real chance level). The von Mises log-lik is signed, has a genuine
    #     zero, rewards rescuing badly-predicted rare regions (macro averaging),
    #     and responds to the smoothness penalty (so regularisation tuning and the
    #     reflective-boundary smoothing are now meaningful).
    #   - euclidean: the wrap-aware distance correlation, unchanged;
    #     `r2_spatial` is a reported descriptor only.
    # The score drives the greedy candidate/baseline ranking and (via the
    # per-fold paired margin) the accept gate; on both geometries it is
    # screened against the target-permuted `null` (see SCREEN_BASELINE_STRATEGY).
    _selection_manifold_metric, _selection_manifold_period = resolve_manifold_metric(settings)
    # Explicitly named, geometry-specific selection key: the torus selects on a
    # von Mises log-score, euclidean on distance correlation (`dcor_xy`). On the
    # torus the `vocal_features.usv_manifold_selection_score` knob chooses the
    # region-balanced `vm_logscore` (macro, default) or the event-weighted
    # `vm_logscore_pooled` (micro) twin; both are always present in the bundle, so
    # switching only changes which column drives the greedy ranking and the
    # accept gate (the candidate pool and the region-reweighted fit are
    # unchanged). The inactive key is NaN in every metric bundle.
    SELECTION_SCORE_KEY = resolve_manifold_selection_score_key(
        settings, _selection_manifold_metric
    )
    # The screen tests `actual` against the target-permuted `null` baseline on
    # BOTH geometries: the permutation across the training fold destroys every
    # X->Y pairing, so beating it shows a behaviour->position relationship that
    # generalises to held-out sessions. The `null_model_free`
    # empirical-density draw is the reported chance floor, not the baseline.
    SCREEN_BASELINE_STRATEGY = 'null'

    num_features = len(univariate_data)
    candidates = []

    # SCREENING_METRIC is retained for the reported metadata; the gate itself
    # recomputes dcor from the stored predictions rather than reading this key.
    SCREENING_METRIC = SELECTION_SCORE_KEY

    # Fold-grain paired-margin screen
    # The gate recomputes the paired `actual` vs `null` score margin PER CV FOLD
    # (the macro von Mises log-likelihood on the torus; the wrap-aware distance
    # correlation on euclidean) and tests it over the folds with an exact sign-flip
    # permutation (the CI is a fold bootstrap): a feature is kept only
    # if its per-fold margin is consistently > 0 across folds (BH-FDR corrected)
    # AND is at least `model_params.selection_effect_floor` of the top surviving driver's
    # margin (a relative effect floor that kills trivial-but-consistent passes
    # and ports across manifolds). Fold-grain -- not session-grain -- is used
    # because the macro average equal-weights acoustic regions, which is only
    # stable when every region is well populated; pooling all sessions within a
    # fold guarantees that, whereas scoring one session at a time starves most
    # regions below the per-region floor and collapses the macro signal.

    # Schema guard: the gate recomputes the score from stored PREDICTIONS, so it needs
    # per-fold `y_pred_xy` / `y_true` / `test_indices` in BOTH `actual` and
    # `null`. A ranking that predates prediction-saving lacks these; fail loudly
    # instead of mis-reporting the schema mismatch as "no significant features".
    _required_fold_keys = ('y_pred_xy', 'y_true', 'test_indices')

    def _has_pred_keys(res):
        """
        Description
        -----------
        Whether one univariate result carries the per-fold predictions the
        fold-grain screen recomputes its score from: ``y_pred_xy``, ``y_true``
        and ``test_indices`` in the ``folds`` block of both its ``actual`` and
        its ``SCREEN_BASELINE_STRATEGY`` entries.

        Parameters
        ----------
        res (Any)
            One feature's univariate result (normally a dict); anything that is
            not a dict yields ``False``.

        Returns
        -------
        has_keys (bool)
            True when every required per-fold key is present in both entries.
        """
        return (isinstance(res, dict) and 'actual' in res
                and SCREEN_BASELINE_STRATEGY in res
                and all(k in res['actual']['folds'] for k in _required_fold_keys)
                and all(k in res[SCREEN_BASELINE_STRATEGY]['folds'] for k in _required_fold_keys))

    _pred_keys_present = any(
        _has_pred_keys(_p[1] if isinstance(_p, tuple) and len(_p) == 2 else _p)
        for _p in univariate_data.values()
    )
    if num_features > 0 and not _pred_keys_present:
        msg = (
            "The manifold fold-grain screen needs per-fold predictions "
            f"({', '.join(_required_fold_keys)}) in both 'actual' and "
            f"'{SCREEN_BASELINE_STRATEGY}', but the univariate results do not "
            "carry them. Re-run the univariate stage with prediction-saving on."
        )
        raise ValueError(msg)

    # Fold-count guard: the screen re-tests the univariate stage's STORED per-fold predictions
    # with a sign-flip permutation whose floor is 1 / 2**n_folds, and the greedy stage below
    # re-splits the same events into `manifold_n_cv_folds` folds. A univariate pickle written
    # with a different fold count (an older 10-fold run) would be screened silently on a
    # different fold layout, so it is refused here, naming both counts.
    _expected_n_folds = int(settings['model_validation']['manifold_n_cv_folds'])
    _stored_fold_counts = {}
    for _feat_name, _payload in univariate_data.items():
        _results = _payload[1] if isinstance(_payload, tuple) and len(_payload) == 2 else _payload
        if _has_pred_keys(_results):
            _stored_fold_counts[_feat_name] = len(_results['actual']['folds']['test_indices'])
    _mismatched_folds = {_f: _n for _f, _n in _stored_fold_counts.items() if _n != _expected_n_folds}
    if _mismatched_folds:
        msg = (
            f"The univariate results were fitted with a different number of outer folds than "
            f"model_validation.manifold_n_cv_folds = {_expected_n_folds}: "
            f"{dict(sorted(_mismatched_folds.items()))}. Re-run the univariate stage with the "
            f"current settings (or point the selection at settings with the matching fold count)."
        )
        raise ValueError(msg)

    print(f"Screening {num_features} features (fold-grain paired score margin "
          f"[torus: {SELECTION_SCORE_KEY} von Mises log-lik; euclidean: dcor], "
          f"BH-FDR q={p_val:.2g}, effect floor={_selection_effect_floor:.2g} x top driver)...")

    _screen_seed = settings['model_validation']['random_seed']

    # On the TORUS the screen scores each fold with the same rule as the greedy
    # gate: the selection score's own averaging (the region-balanced macro score
    # by default, matching the equal-region fit reweighting) and, with
    # `freeze_selection_kappa`, the one frozen concentration. Scoring the
    # region-reweighted fits with the pooled score and a per-fold self-fit kappa
    # instead put half the folds of a real cohort at the kappa=0 flat-torus floor
    # for every feature and its null (a zero margin there), so only features
    # positive on every remaining fold survived. This needs the per-event regions
    # and the development events, so the input pickle is loaded here (once; the
    # binning below reuses it). The univariate folds' `test_indices` index the
    # same sorted-session event array.
    raw_data = None
    screen_event_to_region = None
    kappa_frozen = None
    _vf = settings['vocal_features']
    freeze_selection_kappa = bool(_vf['freeze_selection_kappa'])
    if _selection_manifold_metric == 'torus':
        print("Loading raw continuous input data (the torus screen needs per-event regions)...")
        with Path(input_data_path).open('rb') as f:
            raw_data = pickle.load(f)
        _screen_feature = next(k for k in raw_data if not k.startswith('_'))
        _screen_sessions = sorted(raw_data[_screen_feature].keys())
        _screen_y = np.vstack([raw_data[_screen_feature][sess_id]['Y'] for sess_id in _screen_sessions]).astype(np.float32).astype(np.float64)
        _screen_w = np.concatenate([raw_data[_screen_feature][sess_id]['w'] for sess_id in _screen_sessions]).astype(np.float32).astype(np.float64)
        _screen_groups = np.concatenate([np.full(len(raw_data[_screen_feature][sess_id]['Y']), sess_id) for sess_id in _screen_sessions])
        _screen_regions = np.concatenate([
            np.asarray(raw_data[_screen_feature][sess_id]['category'], dtype=np.float32)
            if 'category' in raw_data[_screen_feature][sess_id]
            else np.full(len(raw_data[_screen_feature][sess_id]['Y']), np.nan, dtype=np.float32)
            for sess_id in _screen_sessions])
        if SELECTION_SCORE_KEY == 'vm_logscore':
            screen_event_to_region = _screen_regions
        if freeze_selection_kappa:
            if _input_pre_md_from_univ is not None:
                _screen_md = _input_pre_md_from_univ
            elif '_input_metadata' in raw_data:
                _screen_md = raw_data['_input_metadata']
            else:
                _screen_md = {}
            _screen_dev, _ = development_heldout_masks(_screen_groups, held_out_session_ids_from_metadata(_screen_md))
            kappa_frozen = _frozen_selection_kappa(_screen_y[_screen_dev], _screen_w[_screen_dev],
                                                   metric=_selection_manifold_metric, period=_selection_manifold_period)
            print(f"  Frozen selection kappa = {kappa_frozen:.4f} "
                  f"(dev events, marginal residuals, unweighted)")

    # PASS 1: per-feature fold-margin bootstrap.
    screen_rows = []
    for feat_name, payload in univariate_data.items():
        results = payload[1] if isinstance(payload, tuple) and len(payload) == 2 else payload
        if not _has_pred_keys(results):
            continue
        stat = _fold_paired_margin_bootstrap(
            results['actual']['folds'],
            results[SCREEN_BASELINE_STRATEGY]['folds'],
            metric=_selection_manifold_metric,
            period=_selection_manifold_period,
            **_gate_kwargs,
            random_state=_screen_seed,
            # Torus: the selection score's averaging and the frozen kappa (see
            # above); euclidean: both None (dcor needs neither).
            event_to_region=screen_event_to_region,
            min_region_events=_min_region_events,
            kappa=kappa_frozen,
        )
        if stat['n_folds'] < 2:
            continue
        screen_rows.append({'feature': feat_name, **stat})

    # BH-FDR on the one-sided sign-flip p-values -- the multiplicity-correct form
    # of "the per-fold margin is consistently > 0". `ci_low` / `ci_high` are
    # reported for transparency; the significance decision uses this FDR test.
    bh_significant = set()
    if screen_rows:
        _pvals = np.array([r['p_value'] for r in screen_rows])
        _order = np.argsort(_pvals)
        _m = _pvals.size
        _thresh_rank = 0
        for _rank, _idx in enumerate(_order):
            if _pvals[_idx] <= (_rank + 1) / _m * p_val:
                _thresh_rank = _rank + 1
        for _rank in range(_thresh_rank):
            bh_significant.add(screen_rows[_order[_rank]]['feature'])

    # Relative effect floor = a fraction of the top BH-significant driver's
    # margin (kills trivial-but-consistent passes; adapts to each manifold's dcor
    # scale). If nothing is BH-significant the floor stays 0 and the
    # `_effect_floor > 0` requirement below rejects every feature.
    _sig_margins = [r['mean_margin'] for r in screen_rows
                    if r['feature'] in bh_significant and r['mean_margin'] > 0]
    _top_margin = max(_sig_margins) if _sig_margins else 0.0
    _effect_floor = _selection_effect_floor * _top_margin

    # PASS 2: keep BH-significant features whose margin clears the effect floor.
    for r in screen_rows:
        keep = (r['feature'] in bh_significant
                and _effect_floor > 0.0
                and r['mean_margin'] >= _effect_floor)
        if keep:
            candidates.append({
                'feature': r['feature'],
                # `mean_r2` is the downstream ranking key; it carries the
                # fold-mean paired score margin (macro von Mises log-lik on the
                # torus, distance correlation on euclidean).
                'mean_r2': r['mean_margin'],
                'p_val': r['p_value'],
                'ci_low': r['ci_low'],
                'ci_high': r['ci_high'],
                'n_folds': r['n_folds'],
            })
        else:
            reason = ('not BH-significant' if r['feature'] not in bh_significant
                      else f"margin < floor {_effect_floor:.4f}")
            print(format_selection_step(
                'Screen', feature=r['feature'], decision='DROP',
                detail=(f"{reason} (margin {r['mean_margin']:+.4f}, "
                        f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], p={r['p_value']:.3g}, "
                        f"folds={r['n_folds']})")))

    # Rank by descending margin (best features first).
    candidates.sort(key=lambda x: x['mean_r2'], reverse=True)
    ranked_features = [x['feature'] for x in candidates]

    if not ranked_features:
        print("No significant features found (fold-grain paired margin + effect floor). Aborting.")
        return

    print(f"Identified {len(ranked_features)} significant candidates "
          f"(effect floor {_effect_floor:.4f} = {_selection_effect_floor:.2g} x top margin "
          f"{_top_margin:.4f}). Top: {ranked_features[0]}")

    # Load the raw input pickle only NOW, after the screen has confirmed there is
    # something to fit -- so the abort / stale-ranking-raise paths above never
    # touch `input_data_path` (the contract two abort tests rely on).
    if raw_data is None:
        print("Loading and binning raw continuous input data...")
        with Path(input_data_path).open('rb') as f:
            raw_data = pickle.load(f)

    # Final upstream-metadata harvest: prefer the univariate copy, fall back to
    # the input pickle's own block for legacy runs.
    _require_analysis_type(raw_data, 'continuous', input_data_path)
    _input_md = _input_pre_md_from_univ
    if _input_md is None and '_input_metadata' in raw_data:
        _input_md = raw_data.pop('_input_metadata')
    else:
        raw_data.pop('_input_metadata', None)
    raw_data.pop('_run_metadata', None)
    raw_data.pop('_consolidation_metadata', None)
    _univariate_md = _univ_pre_md

    # The temporal representation (full-resolution frames or a B-spline basis) decides the
    # penalty settings the fit uses; `hp` is that effective block. The basis spans the
    # history window of the input pickle (every feature and session shares its length).
    _first_feature = raw_data[ranked_features[0]]
    history_frames = _first_feature[sorted(_first_feature)[0]]['X'].shape[1]
    temporal_basis, hp = resolve_temporal_basis(
        settings['hyperparameters']['linear_models']['manifold_regression'], history_frames)
    bin_size = hp['bin_resizing_factor']

    binned_data = {}
    y_global, w_global, groups_global = None, None, None
    region_global = None
    n_time_bins = None

    def _manifold_bin_and_project(X_s: np.ndarray, feat: str) -> np.ndarray:
        """
        Description
        -----------
        The temporal representation of one session's history windows: the optional
        `bin_resizing_factor` binning (keeping the MOST RECENT `new_T * bin_size`
        columns and dropping the oldest `T % bin_size`, exactly as the univariate
        loader bins: the last column is the frame before onset, so truncating the
        tail would drop the frames nearest onset and shift the design against the
        filters) followed by the B-spline projection when a temporal basis is in
        use. The binning loop below and the per-step shift test's shifted refits
        both go through this one function, so a shifted window is represented
        exactly as the stored window it replaces.

        Parameters
        ----------
        X_s (np.ndarray)
            ``(n_rows, history_frames)`` windows of one session.
        feat (str)
            The feature, named in the error.

        Returns
        -------
        X_s (np.ndarray)
            ``(n_rows, n_time_bins)`` design rows (before the float32 cast).
        """

        N, T = X_s.shape
        if bin_size > 1:
            new_T = T // bin_size
            if new_T < 1:
                msg = f"bin_size={bin_size} exceeds the history window length T={T} for feature '{feat}'."
                raise ValueError(msg)
            X_s = X_s[:, T - new_T * bin_size:].reshape(N, new_T, bin_size).mean(axis=2)
        if temporal_basis is not None:
            X_s = project_history_onto_basis(X_s, temporal_basis)
        return X_s

    for feat in ranked_features:
        X_list, y_list, w_list, g_list, region_list = [], [], [], [], []
        sessions = sorted(raw_data[feat].keys())
        for sess_id in sessions:
            X_s = raw_data[feat][sess_id]['X']
            y_s = raw_data[feat][sess_id]['Y']
            w_s = raw_data[feat][sess_id]['w']
            # Per-event acoustic-region labels (the QLVM category, packet key
            # 'category'), row-aligned to Y; NaN when a pickle carried no labels
            # (torus score then degrades to the pooled form, reweighting to uniform).
            if 'category' in raw_data[feat][sess_id]:
                region_s = np.asarray(raw_data[feat][sess_id]['category'], dtype=np.float32)
            else:
                region_s = np.full(len(y_s), np.nan, dtype=np.float32)

            X_s = _manifold_bin_and_project(X_s, feat)

            X_list.append(X_s)
            y_list.append(y_s)
            w_list.append(w_s)
            g_list.append(np.full(len(y_s), sess_id))
            region_list.append(region_s)

        binned_data[feat] = np.vstack(X_list).astype(np.float32)
        if y_global is None:
            y_global = np.vstack(y_list).astype(np.float32)
            w_global = np.concatenate(w_list).astype(np.float32)
            groups_global = np.concatenate(g_list)
            region_global = np.concatenate(region_list)
            n_time_bins = binned_data[feat].shape[1]

    # The per-step shift test's inputs and their lag-zero self-check: every stored
    # window of every screened feature and session must re-cut exactly (float32, as
    # stored) from the trace sidecar at its stored start frame.
    camera_fps = float(settings['io']['camera_sampling_rate'])
    step_cfg = _step_test_settings(settings, camera_fps)
    traces_path, window_starts = _require_shift_test_inputs(_input_md, input_data_path, camera_fps)
    _all_session_ids = sorted({str(s_id) for feat in ranked_features for s_id in raw_data[feat]})
    _verify_session_windows_against_traces(
        raw_data, traces_path, window_starts, history_frames,
        features=ranked_features, sessions=_all_session_ids, cast_dtype=np.float32,
    )
    print(f"Shift-test inputs verified: {len(ranked_features)} feature(s) x {len(_all_session_ids)} "
          f"session(s) re-cut exactly from {Path(traces_path).name}; lag floor "
          f"{step_cfg['min_lag_seconds']} s = {step_cfg['min_lag_frames']} frames.")
    print("JAX selectors run their candidate fits and null refits serially inside the job: XLA's thread "
          "pool is not fork-safe once the backend has run, and one fit already uses every core.")

    del raw_data
    gc.collect()

    # `region_global` is aligned to the same full-event index space as
    # `groups_global` (sorted-session order); the fold scoring and the per-step
    # shift test index it by rows (the torus screen above built the same map from
    # the same pickle).
    warn_if_no_region_labels(region_global, metric=_selection_manifold_metric,
                             context="continuous manifold model selection")

    n_splits = settings['model_validation']['manifold_n_cv_folds']
    test_prop = settings['model_validation']['cv_validation_proportion']
    n_clusters = settings['model_validation']['spatial_cluster_num']
    split_strategy = settings['model_validation']['split_strategy']
    random_seed = settings['model_validation']['random_seed']

    # Reserve the held-out sessions carved ONCE at extraction time and shipped
    # verbatim in the univariate `_input_metadata.held_out_session_ids`. They are
    # dropped from every greedy cv_fold, every candidate fit, and the per-step shift
    # test (which refits on the development rows only); the accepted feature set is then refit on
    # the development sessions and scored on this block exactly once in the
    # finalization stage (the honest, selection-leakage-free number). An empty
    # list -- holdout disabled (`held_out_test_proportion=0`) or a pre-v2 pickle
    # without the field -- makes `_dev_mask` all-True and `_held_positions` empty,
    # so the whole path is an exact no-op and legacy inputs still run.
    #
    # PASS-1 screen (above) needs no masking: it recomputes margins from the
    # univariate stage's stored fold predictions, whose `test_indices` already
    # exclude held-out sessions (carved there too), so `region_global` is only ever
    # indexed at development events by the screen and the fold scoring.
    _held_out_session_ids = held_out_session_ids_from_metadata(_input_md if _input_md is not None else {})
    _dev_mask, _held_mask = development_heldout_masks(groups_global, _held_out_session_ids)
    _dev_positions = np.where(_dev_mask)[0]
    _held_positions = np.where(_held_mask)[0]
    _dev_session_ids = [str(s_id) for s_id in np.unique(groups_global[_dev_positions])]
    if _held_positions.size > 0:
        print(
            f"Held-out reserve: {len(_held_out_session_ids)} session(s), "
            f"{_held_positions.size} event(s) excluded from selection; scored once at finalization."
        )

    # The screen's fold-grain gate bootstraps the PER-FOLD score margin of the
    # univariate stage's stored predictions, and the candidate ranking uses each
    # fold's held-out test predictions, so both are out-of-sample regardless of the
    # split strategy. Under `split_strategy='session'` whole sessions are held out
    # per fold, so there is no within-session leakage. Under `'mixed'` a session's
    # events are scattered across train and test, so within-session temporal
    # autocorrelation leaks into every fold's score; the leakage inflates the
    # `actual` and `null` sides alike and the paired margin cancels most of it, but
    # the screen's CI is still mildly optimistic. Warn (do NOT block: 'mixed' stays
    # a valid option), so the choice is deliberate and visible in the log.
    if split_strategy == 'mixed':
        warnings.warn(
            "continuous_vocal_manifold_model_selection: split_strategy='mixed' "
            "with the fold-grain screen. Each fold's score is still "
            "out-of-sample, but under 'mixed' a session's events span train and "
            "test, so within-session autocorrelation leaks into both the 'actual' "
            "and 'null' sides; the paired per-fold margin cancels most of it yet "
            "the bootstrap CI stays mildly optimistic. Prefer "
            "split_strategy='session' for a leakage-free test.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Joint-tuning configuration. Mirrors the runner: fixed fallback values
    # are the grid centres; the `tune_regularization_bool` flag flips
    # whether each outer fold / candidate runs an inner CV to pick the
    # `(λ_smooth, l2_reg)` pair. Keeping both regularisers in the same
    # search guarantees the selector behaves consistently — the current
    # active model, the anchor, and every forward-selection trial all use
    # hyperparameters chosen by their own inner CV on their own training
    # fold.
    tune_regularization_bool = hp['tune_regularization_bool']
    tune_params = hp['tune_regularization_params']
    lambda_smooth_grid = _log_spaced_grid(
        center=hp['lambda_smooth_fixed'],
        decades_each_side=tune_params['lambda_smooth_decades_each_side'],
    )
    l2_reg_grid = _log_spaced_grid(
        center=hp['l2_reg_fixed'],
        decades_each_side=tune_params['l2_reg_decades_each_side'],
    )
    inner_cv_folds = tune_params['inner_cv_folds']
    # Regularisation tuning is honored on BOTH geometries. The inner-CV objective
    # is the geometry's own selection score (`SELECTION_SCORE_KEY`): `vm_logscore`
    # (the POOLED, label-free von Mises log-lik, via evaluate_metrics with no
    # region_labels) on the torus, `dcor_xy` on euclidean -- so it responds to the
    # smoothness penalty and the tuner selects a meaningful (lambda_smooth,
    # l2_reg) rather than fitting noise on a flat surface. The outer ranking / gate
    # use the per-region MACRO form. This is geometry-determined science, not a
    # settings knob. `tune_regularization_bool` follows the settings flag as-is.
    inner_cv_scoring_metric = SELECTION_SCORE_KEY
    inner_cv_use_one_se_rule = tune_params['inner_cv_use_one_se_rule']
    inner_max_iter = tune_params['inner_max_iter']
    use_lax_loop = hp['use_lax_loop']
    smoothness_order = hp['smoothness_derivative_order']
    reflective_edges = hp['smoothness_reflective_edges']

    print(f"Random Seed: {random_seed} | Num Splits: {n_splits} | Split Strategy: Spatial Proxy ({split_strategy.upper()})")
    if tune_regularization_bool:
        print(
            f"  Joint per-fold tuning ENABLED: |λ_smooth grid|={len(lambda_smooth_grid)}, "
            f"|l2_reg grid|={len(l2_reg_grid)}, inner CV folds={inner_cv_folds}, "
            f"scoring={inner_cv_scoring_metric}, 1SE rule={'ON' if inner_cv_use_one_se_rule else 'OFF'}"
        )
    else:
        print(
            f"  Joint per-fold tuning DISABLED (fixed λ_smooth={hp['lambda_smooth_fixed']}, "
            f"fixed l2_reg={hp['l2_reg_fixed']})"
        )

    # Manifold-metric configuration. On `metric='torus'` the K-means
    # behind the spatial splitter operates on the canonical 4-D
    # embedding so cluster boundaries respect the wrap.
    manifold_metric, manifold_period = resolve_manifold_metric(settings)
    # Geometry selects the estimator for every candidate fit and the inner-CV
    # tuner: torus runs use the convex closed-form sin-cos embedding ridge
    # (wound-aware), euclidean runs keep the unchanged coordinate model.
    manifold_regressor_cls = resolve_manifold_regressor_cls(manifold_metric)

    # Freeze the von Mises concentration ONCE on the development set when
    # `freeze_selection_kappa` is on (torus only). Every torus score below --
    # the empirical-density baseline, the fitted `actual`/`null`/candidate
    # strategies, the held-out refit, and the acceptance-gate bootstrap -- then
    # reuses this single `kappa_frozen` instead of self-refitting kappa on its
    # own residuals. Self-refit floors the concentration on weak folds (the
    # kappa->0 clamp gives the -2*log(2*pi) score floor), which corrupts the
    # paired per-fold margin the gate bootstraps; a shared frozen kappa makes
    # baseline, fitted and gate scores one comparable scoring rule. It is fit on
    # DEVELOPMENT events only (never the held-out reserve) from the residuals of
    # an intercept-only (circular-mean) predictor -- the marginal / null-predictor
    # dispersion -- and UNWEIGHTED, matching the unweighted per-point von Mises
    # scoring. `None` (the default / flag off) preserves the historical
    # per-call self-refit at every site, so euclidean and legacy runs are
    # byte-identical.
    # `freeze_selection_kappa` / `kappa_frozen` were resolved before the screen
    # (torus only; `None` otherwise), from the same development events, so the
    # screen and every score below share the one frozen concentration.

    # Generate the greedy CV folds on the DEVELOPMENT sessions only, then remap
    # the dev-relative indices the splitter returns back into full-array index
    # space via `_dev_positions`. This keeps every `cv_folds` index a valid row of
    # the full `binned_data`/`y_global`/`groups_global` arrays (so the candidate
    # fits and the acceptance bootstrap still index them directly) while
    # guaranteeing no held-out row ever enters a train or validation split. With
    # holdout disabled `_dev_mask` is all-True and `_dev_positions` is the
    # identity map, so this is byte-identical to the pre-holdout behaviour.
    _cv_folds_dev = get_stratified_spatial_splits_stable(
        groups=groups_global[_dev_mask],
        Y=y_global[_dev_mask],
        n_clusters=n_clusters,
        test_prop=test_prop,
        n_splits=n_splits,
        split_strategy=split_strategy,
        random_seed=random_seed,
        tolerance=settings['model_validation']['session_split_initial_tolerance'],
        max_total_attempts=settings['model_validation']['session_split_max_attempts'],
        widen_step=settings['model_validation']['session_split_widen_step'],
        widen_every=settings['model_validation']['session_split_widen_every'],
        metric=manifold_metric,
        period=manifold_period,
    )
    cv_folds = [
        (_dev_positions[tr_idx], _dev_positions[te_idx])
        for (tr_idx, te_idx) in _cv_folds_dev
    ]
    # Leak guard: no held-out event may appear in any greedy fold (train or test).
    if _held_positions.size > 0:
        _held_set = set(_held_positions.tolist())
        for _tr_idx, _te_idx in cv_folds:
            if _held_set.intersection(_tr_idx.tolist()) or _held_set.intersection(_te_idx.tolist()):
                msg_0 = (
                    "Held-out session events leaked into a selection CV fold; "
                    "the development/held-out split is inconsistent."
                )
                raise AssertionError(msg_0)

    # The Step-0 strategy the anchor / first forward feature is gated against: the
    # training-fold intercept-only predictor on the torus, the empirical-density draw on
    # euclidean (where dcor of a constant prediction is degenerate; see Step 0 below).
    step1_gate_baseline = 'null_model_intercept' if manifold_metric == 'torus' else 'null_model_free'
    # The per-step shift test's statistic: the selection score computed in-sample on
    # the development refit (the region-balanced or pooled von Mises log-score on
    # the torus, the distance correlation on euclidean).
    _step_statistic_name = {
        'vm_logscore': 'macro_von_mises_logscore',
        'vm_logscore_pooled': 'pooled_von_mises_logscore',
        'dcor_xy': 'distance_correlation',
    }[SELECTION_SCORE_KEY]

    current_model_features = []
    # R^2 is higher-is-better, so the pre-Step-0 "best" is `-inf` (any finite
    # baseline beats it). `best_current_score` only RANKS candidates here; the
    # ACCEPT decision is the per-fold paired-margin bootstrap below.
    best_current_score = float('-inf')
    step_counter = 0
    # True when the checkpoint a resume restores is a step that rejected its winner: the
    # search had already finished, so the forward loop is skipped and finalization runs.
    resume_converged = False

    fname = Path(univariate_results_path).name
    # Use a non-capturing alternation so conditions like `male_mute_partner`
    # are captured in full rather than truncated to `male`, together with an
    # `ephys_` recording-modality prefix when present.
    cond_match = re.search(r'((?:ephys_)?(?:male|female).*?)(?=_splits|_lam|_gmm|\.pkl)', fname)
    target_condition = cond_match.group(1) if cond_match else "unknown"

    # The per-step prefix (and so the consolidated artifact) carries the input
    # pickle's analysis tag verbatim: the extraction pipeline minted it from the
    # QLVM map the manifold targets come from (`manifold_qlvm`,
    # `manifold_qlvm_duration`, ...; `modeling_utils.manifold_tag_segment`), so
    # the selector and the extraction always agree on the name, and a pickle
    # written before the map-prefix tag (`manifold_qlvm_category`) keeps the
    # step-file name it always had.
    _analysis_tag = _input_md['analysis_tag']
    prefix = f"model_selection_continuous_{_analysis_tag}_{target_condition}_{split_strategy}_step_"

    _run_md = build_selection_metadata(
        modeling_settings=settings,
        selection_function='continuous_vocal_manifold_model_selection',
        selection_metric=SELECTION_SCORE_KEY,
        n_splits_selection=int(n_splits),
        test_proportion=float(test_prop),
        split_strategy=split_strategy,
        random_seed=int(random_seed),
        aic_termination_used=False,
        n_anchor_features=len(ranked_features),
        anchor_feature=ranked_features[0],
        gam_kwargs={},
        extra_knobs={
            'use_top_rank_as_anchor': bool(use_top_rank_as_anchor),
            'p_val': float(p_val),
            # The inner-CV regularisation tuner's one-standard-error rule (a parsimony
            # rule inside the tuner, unrelated to the per-step acceptance test).
            'inner_cv_use_one_se_rule': bool(inner_cv_use_one_se_rule),
            # The screen's p-value test and the Step-0 strategy the first feature is
            # gated against; a resume refuses a checkpoint written under other rules.
            'screen_p_value_test': 'exact_sign_flip',
            'step1_gate_baseline': step1_gate_baseline,
            'manifold_n_cv_folds': int(n_splits),
            'bin_resizing_factor': int(bin_size),
            'temporal_basis': dict(hp['temporal_basis']),
            'screening_metric': SCREENING_METRIC,
            'target_condition': target_condition,
            # Selection-score provenance. On the torus the numeric key `dcor_xy`
            # holds the macro von Mises log-likelihood (not distance correlation);
            # record the human-readable descriptor + the per-region floor so a
            # reader of the saved pickle is not misled by the reused key name.
            'selection_score': (
                'von_mises_macro_logscore' if _selection_manifold_metric == 'torus'
                else 'distance_correlation'
            ),
            'usv_manifold_min_region_events': int(_min_region_events),
            # Frozen-kappa provenance: whether the torus von Mises concentration
            # was frozen once on the development set (vs self-refit per call) and
            # the fitted value, so a reader knows the dispersion scale every score
            # in this run was computed against. `None` on euclidean / when off.
            'freeze_selection_kappa': bool(freeze_selection_kappa),
            'frozen_selection_kappa': (float(kappa_frozen) if kappa_frozen is not None else None),
            # Held-out provenance. `held_out_session_ids` also lives verbatim in
            # the embedded `_input_metadata` sibling; duplicating the summary here
            # makes the selection run-metadata self-describing about whether this
            # run reserved a honest test block (and, if so, how large).
            'held_out_test_proportion': float(settings['model_validation']['held_out_test_proportion']),
            'held_out_session_ids': list(_held_out_session_ids),
            'n_held_out_sessions': len(_held_out_session_ids),
            'n_held_out_events': int(_held_positions.size),
            # The per-step acceptance rule and its knobs (a resume refuses another rule).
            'step_test': {
                'rule': STEP_TEST_RULE,
                'screened_features': list(ranked_features),
                'statistic': _step_statistic_name,
                'selection_score_key': SELECTION_SCORE_KEY,
                'alpha': float(p_val),
                'min_lag_seconds': step_cfg['min_lag_seconds'],
                'min_lag_frames': step_cfg['min_lag_frames'],
                'checkpoint_every': step_cfg['checkpoint_every'],
                'pool_workers': step_cfg['pool_workers'],
                'refit_rows': 'development_rows_as_held_out_refit',
                'traces_path': traces_path,
            },
        },
        settings_path=settings_path,
    )
    _wrap_step = _make_step_wrapper(_input_md, _univariate_md, _run_md)

    existing_steps = []
    if model_selection_dir.is_dir():
        for f_name in (p.name for p in model_selection_dir.iterdir()):
            if f_name.startswith(prefix) and f_name.endswith(".pkl"):
                with contextlib.suppress(ValueError):
                    existing_steps.append(int(f_name.replace(prefix, "").replace(".pkl", "")))

    if existing_steps:
        last_step = max(existing_steps)
        print(f"[RESUME] Restoring from Step {last_step}...")
        try:
            with Path(model_selection_dir / f"{prefix}{last_step}.pkl").open('rb') as f:
                last_res = pickle.load(f)
            _require_step_test_rule(last_res, model_selection_dir / f"{prefix}{last_step}.pkl")

            # Metric-compatibility guard: if this checkpoint was scored on a
            # different selection metric (usv_manifold_metric was flipped
            # between runs, e.g. euclidean `r2_spatial` <-> torus `dcor_xy`)
            # its scores live on a different scale and must not be compared
            # against this run's `best_current_score`
            # / accept rule. Discard and fall through to the fresh-start path.
            _run_meta = last_res['_run_metadata']
            _saved_metric = _run_meta['selection_metric']
            _saved_knobs = _run_meta['extra_knobs']
            # Gate-rule guard: a checkpoint written before the Step-0 gate baseline, the
            # screen's sign-flip p-value and the manifold fold count were recorded (or
            # under different ones) accepted its features under another rule or on other
            # folds, so it cannot be continued.
            _saved_gate = (_saved_knobs['step1_gate_baseline']
                           if 'step1_gate_baseline' in _saved_knobs else None)
            _saved_p_test = (_saved_knobs['screen_p_value_test']
                             if 'screen_p_value_test' in _saved_knobs else None)
            _saved_n_folds = (_saved_knobs['manifold_n_cv_folds']
                              if 'manifold_n_cv_folds' in _saved_knobs else None)
            if _saved_metric != SELECTION_SCORE_KEY:
                print(
                    f"[RESUME] Step {last_step} was scored on '{_saved_metric}' but this "
                    f"run uses '{SELECTION_SCORE_KEY}' (usv_manifold_metric changed); the "
                    "scores are on different scales. Ignoring checkpoint and starting fresh."
                )
                existing_steps = []
            elif (_saved_gate != step1_gate_baseline or _saved_p_test != 'exact_sign_flip'
                  or _saved_n_folds != int(n_splits)):
                print(
                    f"[RESUME] Step {last_step} was selected with Step-0 gate baseline "
                    f"'{_saved_gate}', screen p-value test '{_saved_p_test}' and "
                    f"{_saved_n_folds} folds, but this run uses '{step1_gate_baseline}', "
                    f"'exact_sign_flip' and {int(n_splits)} folds. Ignoring checkpoint "
                    "and starting fresh."
                )
                existing_steps = []
            else:
                # Rebuild the state right after `last_step` from its recorded decision
                # (`_restore_selection_checkpoint`); the incumbent's folds are kept as
                # the step record's baseline.
                state = _restore_selection_checkpoint(
                    last_res, 'mean_r2', 'se_r2', lambda entry: entry['folds'],
                )
                if state is None:
                    print(
                        f"[RESUME] Step {last_step} accepted a feature without scored folds "
                        "(broken checkpoint); ignoring and starting fresh."
                    )
                    existing_steps = []
                else:
                    current_model_features = state['features']
                    best_current_score = state['score']
                    current_model_folds = state['folds']
                    step_counter = state['next_step']
                    # A search that had accepted every ranked feature stopped on that, with no
                    # rejecting step; it is converged too.
                    resume_converged = state['converged'] or len(current_model_features) == len(ranked_features)
                    if resume_converged:
                        print(f"[RESUME] Step {last_step}: selection already converged on "
                              f"{current_model_features}; finalizing.")
                    else:
                        print(f"[RESUME] Continuing from Step {step_counter} with {current_model_features}.")
        except (OSError, pickle.UnpicklingError, KeyError, EOFError) as e:
            print(f"Resume failed: {e}. Starting fresh.")
            existing_steps = []

    # A fresh start (no checkpoint, or one the resume logic rejected) moves any step
    # files of an earlier run aside before Step 0 is written.
    if not existing_steps:
        _quarantine_stale_step_files(model_selection_dir, prefix)

    # Canonical ordering of the metric keys so the dict initialiser and the
    # downstream aggregation stay in lockstep with the estimator output.
    # `r2_spatial` is first because it is the selection score.
    MANIFOLD_METRIC_KEYS = [
        'r2_spatial',
        'euclidean_mae',
        'euclidean_rmse',
        'euclidean_mae_weighted',
        'euclidean_mae_raw',
        'mahalanobis_mae',
        'mae_x',
        'mae_y',
        'pearson_x',
        'pearson_y',
        'spearman_x',
        'spearman_y',
        'dcor_xy',
        'vm_logscore',
        'vm_logscore_pooled',
        'density_geodesic_mae',
        'pullback_geodesic_mae',
        # Region-balanced (macro) twins of the three distance columns: the
        # per-region mean error averaged with equal weight per acoustic region
        # (`manifold_metric.macro_region_mean`), matching the region-reweighted
        # fit and the macro selection score.
        'euclidean_mae_macro',
        'density_geodesic_mae_macro',
        'pullback_geodesic_mae_macro',
    ]

    # Precompute the torus geodesic "reference map" ONCE (fold-independent, like
    # the flat metric): the density-ratio and decoder-Jacobian pullback geodesic
    # geometries over a regular grid, from all embedded `Y` plus the frozen
    # decoder of the map `Y` comes from. Per fold, each event's (prediction, truth) pair snaps to the grid
    # and looks up its geodesic distance, giving the `density_geodesic_mae` /
    # `pullback_geodesic_mae` columns. Torus-only; a disabled metric, an unavailable
    # decoder (OSError / KeyError / ValueError / RuntimeError / ImportError) or a
    # degenerate reference geometry (ValueError / LinAlgError / MemoryError /
    # FloatingPointError / RuntimeError) degrades to NaN columns; any other error is a
    # bug and propagates (the same scopes as the univariate runner).
    geodesic_ctx = None
    _vf_settings = settings['vocal_features']
    if ('usv_manifold_geodesic_metrics' in _vf_settings
            and manifold_metric == 'torus' and y_global is not None):
        _geo_cfg = _vf_settings['usv_manifold_geodesic_metrics']
        if _geo_cfg['compute']:
            # Resolved outside the soft-failure block: a settings block without
            # pullback_metric is a settings error, not a NaN column. The decoder
            # is the production cell of the map the input pickle's manifold
            # columns come from (os_utils constants), never a path.
            _geo_decoder_source = resolve_geodesic_decoder_source(
                _geo_cfg, resolve_manifold_column_names(_input_md, _vf_settings))
            try:
                _geo_decode_fn = None
                if _geo_decoder_source is not None:
                    try:
                        _geo_decode_fn = make_qlvm_decode_fn_from_source(_geo_decoder_source)
                    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as _decode_err:
                        print(f"    [geodesic] decoder unavailable ({_decode_err}); "
                              f"pullback_geodesic_mae -> NaN")
                # Built from the DEVELOPMENT targets only: the reserved held-out
                # sessions never shape the geometry they are scored on.
                geodesic_ctx = build_torus_geodesic_context(
                    np.asarray(y_global[_dev_positions], dtype=np.float64), decode_fn=_geo_decode_fn,
                    n_per_dim=int(_geo_cfg['grid_n_per_dim']), period=manifold_period,
                    k=int(_geo_cfg['graph_k']),
                    density_exponent=float(_geo_cfg['density_exponent']),
                )
                print(f"    [geodesic] reference map built: "
                      f"{int(_geo_cfg['grid_n_per_dim'])}^2 grid, "
                      f"pullback={'on' if _geo_decode_fn is not None else 'off'}")
            except (ValueError, np.linalg.LinAlgError, MemoryError, FloatingPointError, RuntimeError) as _geo_err:
                print(f"    [geodesic] geometry build failed ({_geo_err}); "
                      f"geodesic MAE columns -> NaN")
                geodesic_ctx = None

    # Step 0 (fresh start only): the incumbent the anchor / first forward feature must beat,
    # plus the reported chance floor.
    #   - Gate baseline. On the TORUS it is a training-fold INTERCEPT-ONLY predictor: every
    #     test event of a fold is predicted at the circular mean of that fold's TRAINING
    #     targets (weighted exactly as the candidates are fitted: the KDE spatial weights
    #     times the equal-region `inverse_region_frequency_weights`, i.e. what the
    #     candidates' own estimator returns with no feature; weighting it by the KDE weights
    #     alone gave a pure-noise feature a non-zero margin, because a candidate's fit tends
    #     to the region-reweighted mean as its coefficients go to zero), scored on the
    #     fold's test events by the same `manifold_prediction_metrics` bundle, and gated by
    #     the same per-fold macro von Mises margin (frozen kappa) as every candidate. A model
    #     carrying no behavioural information therefore ties it (margin 0, CI not above 0)
    #     and is rejected; beating it requires the predictions to move with behaviour, not
    #     merely to sit where the repertoire is dense.
    #     On EUCLIDEAN the selection score is distance correlation, which is exactly 0 for
    #     any constant prediction while every non-constant prediction carries dcor's
    #     positive finite-sample bias, so an intercept-only baseline would let any fit pass
    #     on bias alone; there the bias-matched empirical-density draw stays the gate
    #     baseline (no intercept-only block is computed).
    #   - Chance floor. `null_model_free` predicts a uniform random draw from the training
    #     `Y` for every test event ("vocalise from the observed repertoire, ignoring
    #     kinematics"), the chance floor shared with the univariate runner and the CNN. On
    #     the torus it is stored and reported only (`chance_floor_score`); it never gates.
    # The step record keeps `selected_feature = 'null_model_free'` as the Step-0 marker every
    # reader of the step files keys on, names the gate baseline in `baseline_strategy`, and
    # lists the gate baseline FIRST in `candidates_summary`, so a figure that anchors on the
    # first Step-0 block anchors on the gate baseline.
    if not existing_steps:
        print(f"\n--- Establishing Step-0 Baseline (gate: {_STEP0_GATE_LABELS[step1_gate_baseline]}; "
              f"chance floor: {_STEP0_GATE_LABELS['null_model_free']}) ---")

        def _empty_step0_block():
            """
            Description
            -----------
            A fresh, empty Step-0 strategy entry (the intercept-only gate baseline or the
            empirical-density chance floor): a ``folds`` block of empty per-fold lists for
            every metric of ``MANIFOLD_METRIC_KEYS``, the test indices, targets, weights and
            predictions, and the hyperparameter audit fields (a Step-0 strategy has no model,
            so these are filled with NaN / empty / False to keep the saved schema uniform
            across strategies).

            Parameters
            ----------

            Returns
            -------
            block (dict)
                ``{'folds': {...}}`` with every per-fold list empty.
            """
            return {
                'folds': {
                    'metrics': {m: [] for m in MANIFOLD_METRIC_KEYS},
                    'test_indices': [],
                    'y_true': [],
                    'w_test': [],
                    # `(x, y)` predictions per fold, one per test event.
                    'y_pred_xy': [],
                    'selected_lambda_smooth': [],
                    'selected_l2_reg': [],
                    'hyperparam_grid_audit': [],
                    'hyperparams_tuned': [],
                }
            }

        step0_blocks = {'null_model_free': _empty_step0_block()}
        if step1_gate_baseline == 'null_model_intercept':
            step0_blocks = {'null_model_intercept': _empty_step0_block(), **step0_blocks}

        for fold_idx, (tr_idx, te_idx) in enumerate(cv_folds):
            Y_tr, Y_te = y_global[tr_idx], y_global[te_idx]
            w_tr, w_te = w_global[tr_idx], w_global[te_idx]

            # The metric-aware training centroid (the intercept-only prediction) and
            # covariance; the covariance supplies the `mahalanobis_mae` inverse-covariance
            # (matching the regressor's convention so baselines and fitted models compete
            # on the same distance metric). `manifold_prediction_metrics` produces exactly
            # the same metric bundle as the fitted strategies; on `metric='euclidean'`
            # every wrap-aware term reduces to the ordinary plane distance, on
            # `metric='torus'` to the shortest-wrap geodesic.
            w_cov = w_tr / (np.sum(w_tr) + 1e-12)
            # The intercept-only centre carries the candidates' own fit weights: on the torus
            # the KDE weights times the equal-region reweighting (euclidean: the KDE weights).
            if manifold_metric == 'torus':
                w_mu = w_tr * inverse_region_frequency_weights(region_global[tr_idx])
            else:
                w_mu = w_tr
            mu = circular_mean(
                Y_tr, metric=manifold_metric, period=manifold_period,
                weights=w_mu / (np.sum(w_mu) + 1e-12)
            )
            diff_tr = signed_diff(
                Y_tr, mu[None, :], metric=manifold_metric, period=manifold_period
            )
            cov_tr = (w_cov[:, None] * diff_tr).T @ diff_tr
            cov_inv = np.linalg.pinv(cov_tr)

            draw_rng = np.random.default_rng(random_seed + fold_idx + 2)
            draw_idx = draw_rng.choice(len(Y_tr), size=len(Y_te), replace=True)
            fold_predictions = {
                'null_model_free': Y_tr[draw_idx].astype(np.float32),
                'null_model_intercept': np.tile(np.asarray(mu, dtype=np.float32)[None, :], (len(Y_te), 1)),
            }

            for strategy_name, block in step0_blocks.items():
                y_pred_xy = fold_predictions[strategy_name]
                bundle = manifold_prediction_metrics(
                    Y_te, y_pred_xy, w_te,
                    metric=manifold_metric, period=manifold_period,
                    train_cov_inv=cov_inv, random_state=random_seed + fold_idx,
                    region_labels=region_global[te_idx], min_region_events=_min_region_events,
                    kappa=kappa_frozen, dcor_n_sub=_dcor_n_sub, dcor_n_rep=_dcor_n_rep,
                )
                bundle.update(geodesic_mae_columns(
                    y_pred_xy, Y_te, geodesic_ctx,
                    region_labels=region_global[te_idx], min_region_events=_min_region_events))
                f_met = block['folds']['metrics']
                for _k, _v in bundle.items():
                    f_met[_k].append(_v)

                block['folds']['test_indices'].append(te_idx)
                block['folds']['y_true'].append(Y_te)
                block['folds']['w_test'].append(w_te)
                block['folds']['y_pred_xy'].append(y_pred_xy)
                block['folds']['selected_lambda_smooth'].append(float('nan'))
                block['folds']['selected_l2_reg'].append(float('nan'))
                # Shape-matched empty audit so the persisted schema is uniform
                # with the `actual` and `null` strategies.
                block['folds']['hyperparam_grid_audit'].append({
                    'grid_scores': {},
                    'grid_ses': {},
                    'argmax_pair': None,
                    'one_se_applied': False,
                    'one_se_threshold': None,
                })
                block['folds']['hyperparams_tuned'].append(False)

        for block in step0_blocks.values():
            block_scores = np.asarray(block['folds']['metrics'][SELECTION_SCORE_KEY], dtype=float)
            valid_scores = block_scores[~np.isnan(block_scores)]
            block['mean_r2'] = float(np.mean(valid_scores)) if valid_scores.size else float('-inf')
            block['se_r2'] = (
                float(np.std(valid_scores, ddof=1) / np.sqrt(valid_scores.size))
                if valid_scores.size > 1 else 0.0
            )

        gate_block = step0_blocks[step1_gate_baseline]
        best_current_score = gate_block['mean_r2']
        chance_floor_score = step0_blocks['null_model_free']['mean_r2']

        print(format_selection_step('Baseline', detail=f"gate: {_STEP0_GATE_LABELS[step1_gate_baseline]}",
                                    metrics={SELECTION_SCORE_KEY: best_current_score}))
        print(format_selection_step('Chance floor', detail=f"reported only: {_STEP0_GATE_LABELS['null_model_free']}",
                                    metrics={SELECTION_SCORE_KEY: chance_floor_score}))

        # The "current model" begins as the Step-0 gate baseline; its folds are stored
        # as the first forward step's `baseline_folds`. `best_current_score` is kept
        # purely for the logged running-best figure; the accept decision is the
        # per-step shift test.
        current_model_folds = gate_block['folds']

        step_0_res = {
            'step_idx': 0,
            'current_features': [],
            'baseline_score': best_current_score,
            # Step-0 marker every step-file reader keys on (not the gate baseline's name).
            'selected_feature': 'null_model_free',
            # The strategy the anchor / first forward feature is gated against.
            'baseline_strategy': step1_gate_baseline,
            # The empirical-density draw's fold-mean score: the reported chance floor.
            'chance_floor_strategy': 'null_model_free',
            'chance_floor_score': chance_floor_score,
            'candidates_summary': step0_blocks,
            'step_test': None,
        }

        _dump_step_pickle(model_selection_dir / f"{prefix}0.pkl", _wrap_step(step_0_res))

        step_counter = 1

    def _make_manifold_cand_data():
        """
        Description
        -----------
        A fresh, empty results entry for one manifold candidate: a ``folds``
        block of empty per-fold lists (one list per metric of
        ``MANIFOLD_METRIC_KEYS``, the fitted weights, spline coefficients and
        intercepts, the test indices, targets, weights and predictions, the fit
        diagnostics and the per-fold hyperparameter audit trail), appended to
        fold by fold.

        Parameters
        ----------

        Returns
        -------
        cand_data (dict)
            ``{'folds': {...}}`` with every per-fold list empty.
        """
        return {
            'folds': {
                'metrics': {m: [] for m in MANIFOLD_METRIC_KEYS},
                'weights': [],
                # Raw spline coefficients when a temporal basis is used (None
                # otherwise); `weights` holds them mapped onto the frame axis.
                'basis_coefficients': [],
                'intercepts': [],
                'test_indices': [],
                'y_true': [],
                'w_test': [],
                'y_pred_xy': [],
                'n_iter': [],
                'converged': [],
                'fit_time': [],
                # Mirror of the runner's schema so the selection-step pickles
                # and the univariate pickles carry the same hyperparameter
                # audit trail. Both tuned and fixed runs populate these
                # keys; `null_model_free` writes NaN.
                'selected_lambda_smooth': [],
                'selected_l2_reg': [],
                'hyperparam_grid_audit': [],
                'hyperparams_tuned': [],
            }
        }

    def _empty_grid_audit():
        """
        Description
        -----------
        The hyperparameter grid audit recorded for a fold whose regularization
        was not tuned (fixed centres) or whose fit failed: empty score and SE
        grids, no argmax pair and no 1-SE rule applied.

        Parameters
        ----------

        Returns
        -------
        grid_audit (dict)
            ``{'grid_scores': {}, 'grid_ses': {}, 'argmax_pair': None,
            'one_se_applied': False, 'one_se_threshold': None}``.
        """
        return {
            'grid_scores': {},
            'grid_ses': {},
            'argmax_pair': None,
            'one_se_applied': False,
            'one_se_threshold': None,
        }

    def _append_failed_fold(cand_data_):
        """
        Description
        -----------
        Appends a shape-preserving placeholder for a fold whose fit failed to every
        per-fold list of a candidate (NaN metrics, None weights / basis coefficients /
        intercepts, empty test arrays, unconverged, NaN hyperparameters and an empty
        grid audit), so the lists stay aligned fold by fold and the filter
        reconstruction never pairs one fold's coefficients with another fold's index.

        Parameters
        ----------
        cand_data_ (dict)
            The candidate's results entry; its ``folds`` lists are appended to in place.

        Returns
        -------
        None
        """
        for _mk in cand_data_['folds']['metrics']:
            cand_data_['folds']['metrics'][_mk].append(np.nan)
        cand_data_['folds']['weights'].append(None)
        cand_data_['folds']['basis_coefficients'].append(None)
        cand_data_['folds']['intercepts'].append(None)
        cand_data_['folds']['test_indices'].append(np.empty((0,), dtype=np.int64))
        cand_data_['folds']['y_true'].append(np.empty((0, 2), dtype=np.float32))
        cand_data_['folds']['w_test'].append(np.empty((0,), dtype=np.float32))
        cand_data_['folds']['y_pred_xy'].append(np.empty((0, 2), dtype=np.float32))
        cand_data_['folds']['n_iter'].append(np.nan)
        cand_data_['folds']['converged'].append(False)
        cand_data_['folds']['fit_time'].append(np.nan)
        cand_data_['folds']['selected_lambda_smooth'].append(float('nan'))
        cand_data_['folds']['selected_l2_reg'].append(float('nan'))
        cand_data_['folds']['hyperparam_grid_audit'].append(_empty_grid_audit())
        cand_data_['folds']['hyperparams_tuned'].append(False)

    def _pick_fold_hyperparams(X_tr_, Y_tr_, w_tr_, groups_tr_, n_feats_, fold_idx_, region_tr_=None):
        """
        Returns `(lambda_smooth, l2_reg, grid_audit, tuned_flag)` for this
        fold and trial feature set.

        When `tune_regularization_bool` is False the fixed centres are used
        directly (with the standard `1 / sqrt(n_features)` rescale on the
        l2 centre so larger trial models don't silently over-regularise).
        When tuning is on, the inner-CV routine searches the Cartesian
        product of the settings-level grids, with the l2 grid *also*
        rescaled by `1 / sqrt(n_features)` so the search window tracks
        the same effective regularisation strength as the fixed fallback.
        The 1-SE interpretability rule is forwarded via
        `inner_cv_use_one_se_rule`. `region_tr_` (per-row acoustic-region
        labels for the training rows) forwards the equal-region fit reweighting
        into the inner-CV fits on the torus so the tuned pair is chosen under the
        same reweighted objective the outer fit uses; `None` on euclidean.

        Parameters
        ----------
        X_tr_ (np.ndarray)
            The outer fold's training design, ``(n_rows, n_features * n_time_bins)``.
        Y_tr_ (np.ndarray)
            The training rows' ``(n_rows, 2)`` manifold-position targets.
        w_tr_ (np.ndarray)
            The training rows' ``(n_rows,)`` sample weights (before any
            equal-region reweighting; the inner CV applies that via ``region_tr_``).
        groups_tr_ (np.ndarray)
            The training rows' session labels, so the inner CV splits by session.
        n_feats_ (int)
            Number of features in the trial model (sets the l2 rescale).
        fold_idx_ (int)
            The outer fold's index (offsets the tuner's ``random_state``).
        region_tr_ (np.ndarray or None)
            Per-row acoustic-region labels of the training rows, forwarded to the
            inner-CV fits as ``region_train``; ``None`` (default) on euclidean.

        Returns
        -------
        lambda_smooth, l2_reg, grid_audit, tuned (tuple[float, float, dict, bool])
            The smoothness and l2 penalties used, the inner-CV grid record
            (:func:`_empty_grid_audit` when not tuned) and whether the inner-CV
            search ran.
        """
        if not tune_regularization_bool:
            lam_sm = float(hp['lambda_smooth_fixed'])
            lam_l2 = float(hp['l2_reg_fixed'] / np.sqrt(n_feats_))
            return lam_sm, lam_l2, _empty_grid_audit(), False

        rescaled_l2_grid = l2_reg_grid / np.sqrt(n_feats_)
        lam_sm_win, lam_l2_win, grid_audit_ = _tune_manifold_regularization(
            X_train=X_tr_,
            Y_train=Y_tr_,
            w_train=w_tr_,
            groups_train=groups_tr_,
            lambda_smooth_grid=lambda_smooth_grid,
            l2_reg_grid=rescaled_l2_grid,
            inner_cv_folds=inner_cv_folds,
            inner_cv_scoring_metric=inner_cv_scoring_metric,
            inner_cv_use_one_se_rule=inner_cv_use_one_se_rule,
            n_features=n_feats_,
            n_time_bins=n_time_bins,
            spatial_cluster_num=n_clusters,
            smoothness_derivative_order=smoothness_order,
            smoothness_reflective_edges=reflective_edges,
            huber_delta=hp['huber_delta'],
            learning_rate=hp['learning_rate'],
            inner_max_iter=inner_max_iter,
            tol=hp['tol'],
            random_state=hp['random_state'] + fold_idx_,
            verbose=False,
            # Force-disable the lax_loop on the inner-CV tuner path. Per the
            # SmoothBivariateRegression docstring, the lax_loop cache is keyed
            # per-instance, so the joint tuner's ~175 short-lived estimators
            # per outer fold each pay a fresh JIT compile and the run stalls
            # for hours on wide-feature graphs (bin=1 with 600 time bins).
            # The Python-loop path has no compile issues at any scale. The
            # outer-fit call below still honours the user setting where
            # lax_loop is appropriate.
            use_lax_loop=False,
            regressor_cls=manifold_regressor_cls,
            metric=manifold_metric,
            period=manifold_period,
            split_tolerance=settings['model_validation']['session_split_initial_tolerance'],
            split_max_attempts=settings['model_validation']['session_split_max_attempts'],
            split_widen_step=settings['model_validation']['session_split_widen_step'],
            split_widen_every=settings['model_validation']['session_split_widen_every'],
            dcor_n_sub=_dcor_n_sub,
            dcor_n_rep=_dcor_n_rep,
            min_region_events=_min_region_events,
            region_train=region_tr_,
            kappa=kappa_frozen,
        )
        return lam_sm_win, lam_l2_win, grid_audit_, True

    # The development refit the per-step shift test scores on: exactly the held-out
    # refit's rows and fit (every development row, the KDE weights times the
    # equal-region reweighting on the torus, the same hyperparameter choice, the seed
    # `hp['random_state'] + n_splits + 1`). The statistic is the selection score
    # computed in-sample on those rows; the empty incumbent is the Step-0 gate
    # baseline refit on them (the intercept-only circular mean on the torus, the
    # empirical-density draw on euclidean), scored the same way.
    _dev_seed = int(hp['random_state']) + n_splits + 1
    _Y_dev = y_global[_dev_positions]
    _w_dev = w_global[_dev_positions]
    _region_dev = region_global[_dev_positions]
    _groups_dev = groups_global[_dev_positions]
    if manifold_metric == 'torus':
        _w_dev_fit = _w_dev * inverse_region_frequency_weights(_region_dev)
    else:
        _w_dev_fit = _w_dev
    _baseline_dev_score = {}

    def _manifold_development_design(features: list, winner_override: dict | None = None) -> np.ndarray:
        """
        Description
        -----------
        The development design of the model with ``features`` (every development row,
        features side by side); with ``winner_override`` the LAST feature's rows are
        replaced session by session by the raw shifted windows in that per-session
        dict, binned and projected by ``_manifold_bin_and_project`` exactly as the
        stored windows were, and scattered into a copy of the stored development
        rows by the rows' sessions (a session's development rows are its stored rows
        in start-frame order).

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            ``{session_id: {'X': (n_rows_of_session, history_frames) windows}}``.

        Returns
        -------
        X_dev (np.ndarray)
            ``(n_dev_rows, n_features * n_time_bins)`` float32 design.
        """

        blocks = [binned_data[f][_dev_positions] for f in features]
        if winner_override is not None:
            X_winner = blocks[-1].copy()
            for session_id, entry in winner_override.items():
                rows = np.where(_groups_dev == session_id)[0]
                shifted = _manifold_bin_and_project(entry['X'].astype(np.float32), features[-1]).astype(np.float32)
                if rows.size != shifted.shape[0]:
                    msg = (f"shift override for {features[-1]!r} / {session_id!r} has {shifted.shape[0]} "
                           f"rows but the development design holds {rows.size}.")
                    raise ValueError(msg)
                X_winner[rows] = shifted
            blocks[-1] = X_winner
        return np.hstack(blocks)

    def _manifold_development_score(features: list, winner_override: dict | None = None) -> float:
        """
        Description
        -----------
        The selection score (``SELECTION_SCORE_KEY``) computed in-sample on the
        development rows of the model with ``features`` refit on them exactly as the
        held-out refit fits it. The empty model is the Step-0 gate baseline refit on
        the development rows (computed once): the intercept-only circular mean on the
        torus, the empirical-density draw on euclidean, scored with the same bundle.

        Parameters
        ----------
        features (list)
            Features of the model, the winner last.
        winner_override (dict | None)
            See ``_manifold_development_design``.

        Returns
        -------
        score (float)
            The in-sample selection score (higher is better).
        """

        if not features:
            if 'score' not in _baseline_dev_score:
                w_cov = _w_dev / (np.sum(_w_dev) + 1e-12)
                mu = circular_mean(
                    _Y_dev, metric=manifold_metric, period=manifold_period,
                    weights=_w_dev_fit / (np.sum(_w_dev_fit) + 1e-12)
                )
                diff_dev = signed_diff(_Y_dev, mu[None, :], metric=manifold_metric, period=manifold_period)
                cov_dev = (w_cov[:, None] * diff_dev).T @ diff_dev
                cov_inv = np.linalg.pinv(cov_dev)
                if step1_gate_baseline == 'null_model_intercept':
                    dev_pred_xy = np.tile(np.asarray(mu, dtype=np.float32)[None, :], (len(_Y_dev), 1))
                else:
                    draw_rng = np.random.default_rng(_dev_seed)
                    draw_idx = draw_rng.choice(len(_Y_dev), size=len(_Y_dev), replace=True)
                    dev_pred_xy = _Y_dev[draw_idx].astype(np.float32)
                bundle = manifold_prediction_metrics(
                    _Y_dev, dev_pred_xy, _w_dev,
                    metric=manifold_metric, period=manifold_period,
                    train_cov_inv=cov_inv, random_state=_dev_seed,
                    region_labels=_region_dev, min_region_events=_min_region_events,
                    kappa=kappa_frozen, dcor_n_sub=_dcor_n_sub, dcor_n_rep=_dcor_n_rep,
                )
                _baseline_dev_score['score'] = float(bundle[SELECTION_SCORE_KEY])
            return _baseline_dev_score['score']
        X_dev = _manifold_development_design(features, winner_override)
        dev_lambda_smooth, dev_l2_reg, _, _ = _pick_fold_hyperparams(
            X_dev, _Y_dev, _w_dev, _groups_dev,
            n_feats_=len(features),
            fold_idx_=n_splits,
            region_tr_=_region_dev,
        )
        dev_model = manifold_regressor_cls(
            n_features=len(features), n_time_bins=n_time_bins,
            lambda_smooth=dev_lambda_smooth, l2_reg=dev_l2_reg,
            smoothness_derivative_order=smoothness_order,
            smoothness_reflective_edges=reflective_edges,
            huber_delta=hp['huber_delta'],
            learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
            tol=hp['tol'], random_state=_dev_seed,
            _use_lax_loop=use_lax_loop,
            metric=manifold_metric, period=manifold_period,
        )
        dev_model.fit(X_dev, _Y_dev, sample_weight=_w_dev_fit)
        dev_metrics = dev_model.evaluate_metrics(
            X_dev, _Y_dev, weights=_w_dev,
            region_labels=_region_dev, min_region_events=_min_region_events,
            kappa=kappa_frozen, dcor_n_sub=_dcor_n_sub, dcor_n_rep=_dcor_n_rep,
        )
        return float(dev_metrics[SELECTION_SCORE_KEY])

    # Read-only data the candidate fits read (`_manifold_candidate_fold`); the
    # step-specific entries are added per step.
    _manifold_pool_base = {
        'cv_folds': cv_folds,
        'binned_data': binned_data,
        'y_global': y_global,
        'w_global': w_global,
        'groups_global': groups_global,
        'region_global': region_global,
        'n_time_bins': n_time_bins,
        'hp': hp,
        'temporal_basis': temporal_basis,
        'manifold_metric': manifold_metric,
        'manifold_period': manifold_period,
        'regressor_cls': manifold_regressor_cls,
        'smoothness_order': smoothness_order,
        'reflective_edges': reflective_edges,
        'use_lax_loop': use_lax_loop,
        'kappa_frozen': kappa_frozen,
        'min_region_events': _min_region_events,
        'dcor_n_sub': _dcor_n_sub,
        'dcor_n_rep': _dcor_n_rep,
        'geodesic_ctx': geodesic_ctx,
        'metric_keys': list(MANIFOLD_METRIC_KEYS),
        'pick_hyperparams': _pick_fold_hyperparams,
    }

    def _manifold_assemble_cand_data(fold_results: list, feature: str) -> dict:
        """
        Description
        -----------
        Turns the per-fold worker results of one manifold candidate into its results
        entry, keeping every per-fold list aligned: a failed fold is logged and
        contributes the shape-preserving placeholder of ``_append_failed_fold``.

        Parameters
        ----------
        fold_results (list)
            One ``_manifold_candidate_fold`` result per fold, in fold order.
        feature (str)
            The candidate, for the log line.

        Returns
        -------
        cand_data (dict)
            The candidate's results entry.
        """

        cand_data = _make_manifold_cand_data()
        for fold_idx, result in enumerate(fold_results):
            if not result['ok']:
                print(f"    [!] Error fitting {feature} (Fold {fold_idx}): {result['error']}")
                _append_failed_fold(cand_data)
                continue
            fold = result['fold']
            for _mk in cand_data['folds']['metrics']:
                cand_data['folds']['metrics'][_mk].append(fold['metrics'][_mk])
            for key in ('weights', 'basis_coefficients', 'intercepts', 'test_indices', 'y_true', 'w_test',
                        'y_pred_xy', 'n_iter', 'converged', 'fit_time', 'selected_lambda_smooth',
                        'selected_l2_reg', 'hyperparam_grid_audit', 'hyperparams_tuned'):
                cand_data['folds'][key].append(fold[key])
        return cand_data

    def _manifold_fit_candidates(incumbent: list, candidate_features: list) -> dict:
        """
        Description
        -----------
        Fits every candidate on every fold on top of ``incumbent`` and returns each
        candidate's results entry. The incumbent's stacked base block is built once
        per fold here (it depends on the fold, not on the candidate); the candidate
        rows are then stacked on.

        Parameters
        ----------
        incumbent (list)
            The incumbent features.
        candidate_features (list)
            Candidates to score.

        Returns
        -------
        per_candidate (dict)
            ``{feature: cand_data}``.
        """

        fold_base_tr, fold_base_te = [], []
        for tr_idx, te_idx in cv_folds:
            fold_base_tr.append(
                np.hstack([binned_data[f][tr_idx] for f in incumbent])
                if incumbent else None
            )
            fold_base_te.append(
                np.hstack([binned_data[f][te_idx] for f in incumbent])
                if incumbent else None
            )
        set_pool_context(
            manifold_step={**_manifold_pool_base,
                           'current_model_features': list(incumbent),
                           'fold_base_tr': fold_base_tr, 'fold_base_te': fold_base_te},
            candidate_fold=_manifold_candidate_fold,
        )
        # The worker count is derived as in the other selectors but the pool runs with
        # ONE worker: JAX (XLA's thread pool) is not fork-safe once the backend has run
        # in this process, and a single fit already uses every core.
        n_workers_requested = pool_worker_count(step_cfg['pool_workers'], len(incumbent) + 1,
                                                step_cfg['pool_memory_gb'], step_cfg['fit_peak_gb_base'],
                                                step_cfg['fit_peak_gb_per_term'])
        del n_workers_requested
        n_folds = len(cv_folds)
        tasks = [(feat, fold_idx) for feat in candidate_features for fold_idx in range(n_folds)]
        pool = SelectionPool(1)
        try:
            results = pool.map('candidate_fold', tasks)
        finally:
            pool.close()
        return {feat: _manifold_assemble_cand_data(results[i * n_folds:(i + 1) * n_folds], feat)
                for i, feat in enumerate(candidate_features)}

    def _manifold_step_test(winner: str, step_index: int, m_candidates: int) -> dict:
        """
        Description
        -----------
        The per-step shift test of ``winner`` on top of the current incumbent.

        Parameters
        ----------
        winner (str)
            The cross-validated winner.
        step_index (int)
            The step.
        m_candidates (int)
            Candidates the winner was chosen from (the Bonferroni factor).

        Returns
        -------
        step_test (dict)
            The record of ``run_step_shift_test``.
        """

        incumbent = list(current_model_features)
        trial = [*incumbent, winner]
        incumbent_score = _manifold_development_score(incumbent)
        observed_score = _manifold_development_score(trial)
        winner_traces = {s: read_feature_trace(traces_path, winner, s) for s in _dev_session_ids}
        # JAX is not fork-safe once the backend has run in this process: the null
        # refits run serially (one worker), see `_manifold_fit_candidates`.
        return run_step_shift_test(
            n_workers=1,
            refit_and_score=lambda override: _manifold_development_score(trial, override),
            incumbent_features=incumbent,
            incumbent_score=incumbent_score,
            winner=winner,
            observed_score=observed_score,
            statistic_name=_step_statistic_name,
            winner_traces=winner_traces,
            window_starts={s: window_starts[s] for s in winner_traces},
            slots=_SESSION_WINDOW_SLOTS,
            history_frames=history_frames,
            m_candidates=m_candidates,
            alpha=p_val,
            min_lag_frames=step_cfg['min_lag_frames'],
            random_seed=random_seed,
            step_index=step_index,
            checkpoint_every=step_cfg['checkpoint_every'],
            partial_path=model_selection_dir / f"{prefix}{step_index}_null_partial.pkl",
            label=f"Step {step_index}",
        )

    # Auto-anchor logic
    if use_top_rank_as_anchor and step_counter == 1 and not resume_converged:
        anchor = ranked_features[0]
        print(format_selection_step('Anchor', feature=anchor,
                                    detail=f"top-ranked; forced as the first candidate, tested against "
                                           f"{_STEP0_GATE_LABELS[step1_gate_baseline]}"))
        cand_data = _manifold_fit_candidates([], [anchor])[anchor]
        _all_anchor_scores = np.asarray(cand_data['folds']['metrics'][SELECTION_SCORE_KEY], dtype=float)
        valid_scores = _all_anchor_scores[np.isfinite(_all_anchor_scores)]
        # Surface PARTIAL fold failure (some folds non-finite, others fine): the
        # anchor's mean is quietly averaged over only the surviving folds while the
        # run still reports success. On the euclidean path this signals iterative-fit
        # divergence (the closed-form torus estimator does not trip it).
        _n_anchor_folds = int(_all_anchor_scores.size)
        _n_anchor_bad = _n_anchor_folds - int(valid_scores.size)
        if 0 < _n_anchor_bad < _n_anchor_folds:
            print(
                f"  [!] WARNING: anchor '{anchor}' produced a non-finite "
                f"{SELECTION_SCORE_KEY} on {_n_anchor_bad}/{_n_anchor_folds} folds "
                f"(fit failed / diverged); those folds are dropped from its mean. A large "
                f"fraction indicates numerical instability — check the l2 grid floor."
            )
        _require_scored_candidate('Anchor', 1, int(valid_scores.size))
        step_test = None
        if valid_scores.size:
            cand_data['mean_r2'] = float(np.mean(valid_scores))
            cand_data['se_r2'] = (
                float(np.std(valid_scores, ddof=1) / np.sqrt(valid_scores.size))
                if valid_scores.size > 1 else 0.0
            )
            # The forced anchor is tested like every winner, with Bonferroni over every
            # screened feature it was forced ahead of.
            step_test = _manifold_step_test(anchor, 1, len(ranked_features))
        anchor_accepted = step_test is not None and step_test['accepted']
        step_1_res = {
            'step_idx': 1,
            'current_features': [anchor] if anchor_accepted else [],
            'baseline_score': best_current_score,
            'baseline_folds': current_model_folds,
            'selected_feature': anchor if anchor_accepted else None,
            'candidates_summary': {anchor: cand_data} if valid_scores.size else {},
            'step_test': step_test,
        }
        _dump_step_pickle(model_selection_dir / f"{prefix}1.pkl", _wrap_step(step_1_res))
        if anchor_accepted:
            print(format_selection_step(
                'Anchor', feature=anchor, decision='ACCEPT',
                metrics={SELECTION_SCORE_KEY: cand_data['mean_r2'], 'p_adj': step_test['p_adjusted']}))
            best_current_score = cand_data['mean_r2']
            current_model_folds = cand_data['folds']
            current_model_features = [anchor]
            step_counter = 2
        else:
            print(format_selection_step(
                'Anchor', feature=anchor, decision='REJECT',
                detail=('every fold errored out' if not valid_scores.size
                        else 'the forced anchor failed the shift test') + '; selection finished'))
            resume_converged = True

    # Forward stepwise selection loop
    print("\n--- Starting Forward Selection ---")
    # A resume that restored a converged search skips the loop and finalizes directly.
    while not resume_converged:
        print(f"\n=== Step {step_counter} === Best {SELECTION_SCORE_KEY}: {best_current_score:.5f}")
        step_results = {
            'step_idx': step_counter, 'current_features': list(current_model_features),
            'baseline_score': best_current_score, 'baseline_folds': current_model_folds,
            'candidates_summary': {},
            'selected_feature': None,
            'step_test': None,
        }

        candidates_this_step = [feat for feat in ranked_features if feat not in current_model_features]
        m_candidates = len(candidates_this_step)
        per_candidate = _manifold_fit_candidates(current_model_features, candidates_this_step)

        # The score is higher-is-better; initialise the per-step best score at
        # `-inf` so any finite candidate score is an improvement.
        best_cand, best_cand_score = None, float('-inf')
        for feat in candidates_this_step:
            cand_data = per_candidate[feat]
            _all_cand_scores = np.asarray(cand_data['folds']['metrics'][SELECTION_SCORE_KEY], dtype=float)
            valid_scores = _all_cand_scores[np.isfinite(_all_cand_scores)]
            # Surface PARTIAL fold failure for this candidate (some folds
            # diverged, others survived): otherwise the candidate's mean is
            # silently averaged over only the surviving folds. See the matching
            # anchor guard above for the rationale (euclidean path only; the
            # closed-form torus estimator does not diverge).
            _n_cand_folds = int(_all_cand_scores.size)
            _n_cand_bad = _n_cand_folds - int(valid_scores.size)
            if 0 < _n_cand_bad < _n_cand_folds:
                print(
                    f"  [!] WARNING: candidate '{feat}' produced a non-finite "
                    f"{SELECTION_SCORE_KEY} on {_n_cand_bad}/{_n_cand_folds} folds "
                    f"(fit failed / diverged); those folds are dropped from its mean."
                )
            if not valid_scores.size:
                print(format_selection_step(f"Step {step_counter}", feature=feat,
                                            decision='FAILED (no finite folds)'))
                continue

            mean_score = float(np.mean(valid_scores))
            se_score = (
                float(np.std(valid_scores, ddof=1) / np.sqrt(valid_scores.size))
                if valid_scores.size > 1 else 0.0
            )
            mean_mae = float(np.nanmean(cand_data['folds']['metrics']['euclidean_mae']))
            print(format_selection_step(f"Step {step_counter}", feature=feat,
                                        metrics={SELECTION_SCORE_KEY: mean_score, 'MAE': mean_mae}))

            cand_data['mean_r2'] = mean_score
            cand_data['se_r2'] = se_score
            step_results['candidates_summary'][feat] = cand_data

            if mean_score > best_cand_score:
                best_cand_score, best_cand = mean_score, feat

        # The cross-validated winner is tested with the per-step shift test, Bonferroni
        # over the m candidates of this step; the step is accepted only when no shifted
        # refit reaches the observed in-sample gain.
        _require_scored_candidate(f"Step {step_counter}", m_candidates, int(best_cand is not None))
        step_test = (_manifold_step_test(best_cand, step_counter, m_candidates)
                     if best_cand is not None else None)
        step_results['step_test'] = step_test

        if step_test is not None and step_test['accepted']:
            print(format_selection_step(
                f"Step {step_counter}", feature=best_cand, decision='ACCEPT',
                metrics={SELECTION_SCORE_KEY: best_cand_score, 'p_adj': step_test['p_adjusted']}))
            step_results['selected_feature'] = best_cand
            current_model_features.append(best_cand)
            current_model_folds = step_results['candidates_summary'][best_cand]['folds']

            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results))

            best_current_score = best_cand_score
            step_counter += 1
        else:
            if best_cand is None:
                reject_detail = 'no candidate had finite folds; selection finished'
            else:
                reject_detail = (f"winner {best_cand!r} failed the shift test "
                                 f"(p_adj {step_test['p_adjusted']:.4f} vs alpha {p_val}); selection finished")
            print(format_selection_step(f"Step {step_counter}", decision='REJECT', detail=reject_detail))
            step_results['selected_feature'] = None
            _dump_step_pickle(model_selection_dir / f"{prefix}{step_counter}.pkl", _wrap_step(step_results))
            break

        if len(current_model_features) == len(ranked_features):
            break

    print("\n--- Finalizing Results ---")
    # No blanket except: a failed final refit or save must stop the run loudly, not
    # pass as a finished selection with no final model.
    last_file_path = model_selection_dir / f"{prefix}{step_counter}.pkl"
    if not last_file_path.exists():
        last_file_path = model_selection_dir / f"{prefix}{step_counter - 1}.pkl"

    with Path(last_file_path).open('rb') as f:
        step_data = pickle.load(f)

    final_res = {
        'final_model_features': current_model_features,
        'univariate_results_path': univariate_results_path,
        'input_data_path': input_data_path
    }

    # Honest held-out score: refit the accepted feature set on ALL development
    # sessions and score it once on the reserved held-out block. This is the
    # only number in the run untouched by feature nomination, screening,
    # greedy acceptance, or lambda-tuning -- every one of those saw
    # development data only. Hyperparameters are chosen on the development set
    # via the same `_pick_fold_hyperparams` the folds use (with torus tuning
    # now re-enabled, this runs the inner-CV pooled-vM search on development).
    # Equal-region reweighting is applied to the development fit on the torus,
    # and the held-out score is the per-region macro von Mises log-lik. When no
    # feature was accepted the honest model is the Step-0 gate baseline refit on
    # development Y (the intercept-only circular mean on the torus, the
    # empirical-density draw on euclidean). When holdout
    # is disabled (`_held_positions` empty, i.e. `held_out_test_proportion=0`
    # or a pre-v2 pickle) the block is skipped and `heldout` stays None, so
    # legacy / no-holdout runs are byte-identical to before.
    heldout_result = None
    if _held_positions.size > 0:
        Y_dev = y_global[_dev_positions]
        w_dev = w_global[_dev_positions]
        Y_held = y_global[_held_positions]
        w_held = w_global[_held_positions]
        region_dev = region_global[_dev_positions]
        region_held = region_global[_held_positions]
        held_seed = int(hp['random_state']) + n_splits + 1

        if manifold_metric == 'torus':
            w_dev_fit = w_dev * inverse_region_frequency_weights(region_dev)
        else:
            w_dev_fit = w_dev

        if len(current_model_features) > 0:
            X_dev = np.hstack([binned_data[f][_dev_positions] for f in current_model_features])
            X_held = np.hstack([binned_data[f][_held_positions] for f in current_model_features])
            groups_dev = groups_global[_dev_positions]
            n_feats_final = len(current_model_features)

            final_lambda_smooth, final_l2_reg, final_grid_audit, final_tuned_flag = (
                _pick_fold_hyperparams(
                    X_dev, Y_dev, w_dev, groups_dev,
                    n_feats_=n_feats_final,
                    fold_idx_=n_splits,
                    region_tr_=region_dev,
                )
            )
            held_model = manifold_regressor_cls(
                n_features=n_feats_final, n_time_bins=n_time_bins,
                lambda_smooth=final_lambda_smooth, l2_reg=final_l2_reg,
                smoothness_derivative_order=smoothness_order,
                smoothness_reflective_edges=reflective_edges,
                huber_delta=hp['huber_delta'],
                learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                tol=hp['tol'], random_state=held_seed,
                _use_lax_loop=use_lax_loop,
                metric=manifold_metric, period=manifold_period,
            )
            held_model.fit(X_dev, Y_dev, sample_weight=w_dev_fit)
            held_metrics = held_model.evaluate_metrics(
                X_held, Y_held, weights=w_held,
                region_labels=region_held, min_region_events=_min_region_events,
                kappa=kappa_frozen, dcor_n_sub=_dcor_n_sub, dcor_n_rep=_dcor_n_rep,
            )
            held_pred_xy = held_model.predict(X_held, snap=True).astype(np.float32)
            held_metrics.update(geodesic_mae_columns(
                held_pred_xy, Y_held, geodesic_ctx,
                region_labels=region_held, min_region_events=_min_region_events))
            heldout_result = {
                'model_type': 'manifold_regressor',
                'features': list(current_model_features),
                'metrics': {_mk: float(held_metrics[_mk]) for _mk in MANIFOLD_METRIC_KEYS},
                'weights': (basis_coefficients_to_frames(held_model.coef_, temporal_basis, n_features=n_feats_final, axis=0) if temporal_basis is not None else held_model.coef_),
                'basis_coefficients': (np.asarray(held_model.coef_) if temporal_basis is not None else None),
                'intercept': held_model.intercept_,
                'test_indices': _held_positions,
                'y_true': Y_held,
                'w_test': w_held,
                'y_pred_xy': held_pred_xy,
                'n_iter': int(held_model.n_iter_),
                'converged': bool(held_model.converged_),
                'fit_time': float(held_model.fit_time_),
                'selected_lambda_smooth': final_lambda_smooth,
                'selected_l2_reg': final_l2_reg,
                'hyperparam_grid_audit': final_grid_audit,
                'hyperparams_tuned': final_tuned_flag,
                'n_held_out_sessions': len(_held_out_session_ids),
                'n_held_out_events': int(_held_positions.size),
            }
            # No-signal filter band: the same model, hyperparameters and
            # reweighting refit on the development sessions with the positions
            # (with their weights and regions) permuted across the development set;
            # the filter figures draw the final filters against it.
            n_null_permutations = int(settings['model_params']['filter_null_permutations'])
            null_rng = np.random.default_rng(held_seed)
            null_weights = []
            for _ in range(n_null_permutations):
                perm = null_target_permutation(groups_dev.shape[0], null_rng)
                w_perm = w_dev[perm]
                w_perm_fit = (w_perm * inverse_region_frequency_weights(region_dev[perm])
                              if manifold_metric == 'torus' else w_perm)
                null_model = manifold_regressor_cls(
                    n_features=n_feats_final, n_time_bins=n_time_bins,
                    lambda_smooth=final_lambda_smooth, l2_reg=final_l2_reg,
                    smoothness_derivative_order=smoothness_order,
                    smoothness_reflective_edges=reflective_edges,
                    huber_delta=hp['huber_delta'],
                    learning_rate=hp['learning_rate'], max_iter=hp['max_iter'],
                    tol=hp['tol'], random_state=held_seed,
                    _use_lax_loop=use_lax_loop,
                    metric=manifold_metric, period=manifold_period,
                )
                null_model.fit(X_dev, Y_dev[perm], sample_weight=w_perm_fit)
                null_weights.append(basis_coefficients_to_frames(null_model.coef_, temporal_basis, n_features=n_feats_final, axis=0)
                                    if temporal_basis is not None else np.asarray(null_model.coef_))
            heldout_result['null_weights'] = (np.stack(null_weights).astype(np.float32) if null_weights else None)
        else:
            # No feature survived selection: the honest held-out model is the
            # Step-0 gate baseline built from development Y and scored on the
            # held-out block -- the intercept-only circular mean on the torus, the
            # empirical-density draw on euclidean (see Step 0).
            w_cov = w_dev / (np.sum(w_dev) + 1e-12)
            # Same weighting as the Step-0 gate baseline: `w_dev_fit` is the KDE weights times
            # the equal-region reweighting on the torus (euclidean: the KDE weights).
            mu = circular_mean(
                Y_dev, metric=manifold_metric, period=manifold_period,
                weights=w_dev_fit / (np.sum(w_dev_fit) + 1e-12)
            )
            diff_dev = signed_diff(
                Y_dev, mu[None, :], metric=manifold_metric, period=manifold_period
            )
            cov_dev = (w_cov[:, None] * diff_dev).T @ diff_dev
            cov_inv = np.linalg.pinv(cov_dev)
            if step1_gate_baseline == 'null_model_intercept':
                held_pred_xy = np.tile(np.asarray(mu, dtype=np.float32)[None, :], (len(Y_held), 1))
            else:
                draw_rng = np.random.default_rng(held_seed)
                draw_idx = draw_rng.choice(len(Y_dev), size=len(Y_held), replace=True)
                held_pred_xy = Y_dev[draw_idx].astype(np.float32)
            held_bundle = manifold_prediction_metrics(
                Y_held, held_pred_xy, w_held,
                metric=manifold_metric, period=manifold_period,
                train_cov_inv=cov_inv, random_state=held_seed,
                region_labels=region_held, min_region_events=_min_region_events,
                kappa=kappa_frozen, dcor_n_sub=_dcor_n_sub, dcor_n_rep=_dcor_n_rep,
            )
            held_bundle.update(geodesic_mae_columns(
                held_pred_xy, Y_held, geodesic_ctx,
                region_labels=region_held, min_region_events=_min_region_events))
            heldout_result = {
                'model_type': step1_gate_baseline,
                'features': [],
                'metrics': {_mk: float(held_bundle[_mk]) for _mk in MANIFOLD_METRIC_KEYS},
                'y_pred_xy': held_pred_xy,
                'y_true': Y_held,
                'w_test': w_held,
                'test_indices': _held_positions,
                'n_held_out_sessions': len(_held_out_session_ids),
                'n_held_out_events': int(_held_positions.size),
            }
        final_res['heldout'] = heldout_result

    for _k in RESERVED_METADATA_KEYS:
        step_data.pop(_k, None)
    step_data.update(final_res)
    _dump_step_pickle(last_file_path, _wrap_step(step_data))

    # Standardized end summary: selected-feature count + final vs the Step-0 gate
    # baseline (intercept-only on the torus) + the empirical-density chance floor
    # (reported only) + the full save path. Both are read back from the Step-0
    # pickle so the comparison is correct on both fresh and resume runs (where
    # `best_current_score` no longer holds the baseline). A resume always runs on
    # a checkpoint of this rule set (see the gate-rule guard), whose Step-0 file
    # carries both fields.
    step0_path = model_selection_dir / f"{prefix}0.pkl"
    with Path(step0_path).open('rb') as f:
        _step0 = pickle.load(f)
    metrics_by_strategy = {
        'Final': {SELECTION_SCORE_KEY: best_current_score},
        f"Baseline ({_step0['baseline_strategy']})": {SELECTION_SCORE_KEY: _step0['baseline_score']},
        'Chance floor (null_model_free)': {SELECTION_SCORE_KEY: _step0['chance_floor_score']},
    }
    # The held-out row (development-refit, scored once on the reserved block)
    # is the honest, selection-leakage-free number; it sits alongside the
    # optimistic development-CV `Final` for a direct at/below sanity check.
    if heldout_result is not None:
        metrics_by_strategy['Held-out'] = {
            SELECTION_SCORE_KEY: heldout_result['metrics'][SELECTION_SCORE_KEY]
        }
    print(format_run_summary(
        label=f"continuous model-selection ({len(current_model_features)} features)",
        metrics_by_strategy=metrics_by_strategy,
        out_path=str(last_file_path),
    ))
