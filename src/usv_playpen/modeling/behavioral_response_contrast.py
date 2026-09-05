"""
@author: bartulem
Module for testing whether male vocal bouts change female behavior.

Consumes the anchor table written by
:class:`~usv_playpen.modeling.modeling_behavioral_response.BehavioralResponsePipeline`
and answers two questions from one fit:

1.  **Does a bout, versus comparable silence, change her behavior?** The
    coefficient on a 0/1 ``vocal`` indicator, contrasting bout offsets against
    anchors drawn from inter-bout silence.
2.  **Does bout duration matter?** Duration terciles interacted with ``vocal``, so
    each duration band gets its own step against silence.

The model is a GLM, not a nested predictive comparison. That is deliberate. A
nested comparison answers "does the vocal block improve out-of-sample
prediction" and reports a ``dD^2`` -- a number that conflates effect size,
timing and nonlinearity, and that came back at 0.0039 on the tiled precursor:
reproducible across every split, and uninterpretable. A coefficient answers the
question actually asked, in the feature's own units, with an interval.

Three choices worth stating:

*   **Duration enters as terciles, not a slope.** Bout duration is heavily skewed
    (median 0.43 s with a long tail), so a single slope would be dominated by a
    handful of long bouts. Terciles show the shape of the dose-response and let
    non-monotonicity be visible rather than averaged away.
*   **Standard errors are clustered on session.** Rows within a session share an
    animal, an arena and a day; treating ~13,000 anchors from ~120 sessions as
    independent would shrink the intervals by roughly the square root of the
    within-session count.
*   **The same model is refit per time bin.** The predictors never change; only
    the target does. That turns the single contrast into a time course of when
    the response appears, which is the event-triggered average with the
    pre-anchor state regressed out.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm

from ..os_utils import atomic_output_path
from .modeling_utils import format_run_header, format_run_summary, format_selection_step

RESERVED_ARTIFACT_KEYS = ('_run_metadata', '_input_metadata', '_univariate_metadata',
                          '_consolidation_metadata')


def duration_tercile_labels(bout_duration: np.ndarray,
                            is_vocal: np.ndarray,
                            n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits vocal rows into equal-count duration bands.

    Cut points come from the vocal rows only -- quiet rows have no duration -- so
    the bands hold equal numbers of BOUTS rather than equal numbers of anchors.

    Parameters
    ----------
    bout_duration : np.ndarray
        Per-row bout duration in seconds; ``nan`` on quiet rows.
    is_vocal : np.ndarray
        Per-row 1.0 at bout offsets, 0.0 at quiet anchors.
    n_bins : int
        Number of equal-count bands.

    Returns
    -------
    band_index, edges : tuple of np.ndarray
        Per-row band index (``-1`` on quiet rows), and the ``n_bins + 1`` duration
        cut points defining the bands.

    Raises
    ------
    ValueError
        If the vocal rows carry fewer distinct durations than requested bands, so
        the bands could not be populated.
    """

    vocal_mask = is_vocal > 0.0
    durations = bout_duration[vocal_mask]
    if np.unique(durations).size < n_bins:
        msg = (
            f"Only {np.unique(durations).size} distinct bout durations are available, "
            f"which cannot fill {n_bins} equal-count bands; lower "
            f"`behavioral_response.duration_n_bins`."
        )
        raise ValueError(msg)

    edges = np.quantile(durations, np.linspace(0.0, 1.0, n_bins + 1))
    # `np.digitize` on the interior edges puts each duration in [0, n_bins - 1].
    band_index = np.full(bout_duration.shape[0], -1, dtype=int)
    band_index[vocal_mask] = np.clip(np.digitize(durations, edges[1:-1]), 0, n_bins - 1)
    return band_index, edges


def build_design_matrix(covariates: np.ndarray,
                        is_vocal: np.ndarray,
                        band_index: np.ndarray,
                        n_bins: int,
                        covariate_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Assembles the design: intercept, one step per duration band, then covariates.

    The vocal terms are ``vocal x band`` indicators rather than a ``vocal`` main
    effect plus a duration slope. Each band's coefficient is then read directly
    as that band's step against silence, and question (1) is the joint statement
    of those steps while question (2) is how they differ from one another. Coding
    quiet rows as duration zero was rejected: it would assert that "no bout at
    all" lies on the same line as "a very short bout", which is the assumption
    under test.

    Any row carrying a non-finite covariate is left in place here; callers drop
    such rows before fitting.

    Parameters
    ----------
    covariates : np.ndarray
        ``(n_rows, n_covariates)`` pre-anchor history summaries.
    is_vocal : np.ndarray
        Per-row 1.0 at bout offsets, 0.0 at quiet anchors.
    band_index : np.ndarray
        Per-row duration band, ``-1`` on quiet rows.
    n_bins : int
        Number of duration bands.
    covariate_labels : list of str
        Column names for ``covariates``.

    Returns
    -------
    design, labels : tuple
        ``(n_rows, 1 + n_bins + n_covariates)`` design matrix and its column names.
    """

    n_rows = covariates.shape[0]
    band_columns = np.zeros((n_rows, n_bins), dtype=float)
    for band in range(n_bins):
        band_columns[:, band] = ((is_vocal > 0.0) & (band_index == band)).astype(float)

    design = np.column_stack([np.ones(n_rows), band_columns, covariates])
    labels = (['intercept']
              + [f'vocal_duration_band_{band}' for band in range(n_bins)]
              + list(covariate_labels))
    return design, labels


def fit_contrast(target: np.ndarray,
                 design: np.ndarray,
                 labels: list[str],
                 session_ids: np.ndarray,
                 likelihood: str) -> dict[str, Any]:
    """
    Fits one GLM with session-clustered standard errors.

    Rows carrying a non-finite target or covariate are dropped before fitting;
    the count is reported so a silently shrinking sample is visible.

    Parameters
    ----------
    target : np.ndarray
        ``(n_rows,)`` response in native units, strictly positive for ``'gamma'``.
    design : np.ndarray
        ``(n_rows, n_terms)`` design matrix including the intercept.
    labels : list of str
        Column names for ``design``.
    session_ids : np.ndarray
        ``(n_rows,)`` clustering unit for the robust covariance.
    likelihood : str
        ``'gamma'`` (log link) or ``'gaussian'`` (identity on the raw response).

    Returns
    -------
    fit_results : dict
        Per-term ``coefficient``, ``std_error``, ``z``, ``p_value`` and 95%
        interval, the fitted row count, the number dropped, and
        ``non_finite_by_term`` attributing the loss to individual columns.

    Raises
    ------
    ValueError
        If ``likelihood`` is unrecognised, no row survives the finite filter, or
        the surviving design is rank-deficient.
    """

    if likelihood not in ('gamma', 'gaussian'):
        msg = f"`likelihood` must be 'gamma' or 'gaussian'; got '{likelihood}'."
        raise ValueError(msg)

    usable = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    if likelihood == 'gamma':
        usable &= target > 0.0
    n_dropped = int((~usable).sum())
    # A single non-finite covariate drops the whole row, so one bad feature can
    # decimate the sample without saying so. Attribute the loss per column.
    non_finite_by_term = {
        labels[i]: int(np.sum(~np.isfinite(design[:, i])))
        for i in range(design.shape[1])
        if np.any(~np.isfinite(design[:, i]))
    }
    if not np.any(usable):
        worst = sorted(non_finite_by_term.items(), key=lambda kv: -kv[1])[:5]
        msg = (
            f"No row survives the finite-value filter, so the contrast cannot be fitted. "
            f"{int(np.sum(~np.isfinite(target)))} of {target.size} rows have a non-finite "
            f"target; the covariates losing the most rows are {worst or 'none'}."
        )
        raise ValueError(msg)

    # A rank-deficient design surfaces from statsmodels as a bare
    # `LinAlgError: Singular matrix` raised inside the sandwich estimator, which
    # says nothing about which columns caused it. The usual causes here are two
    # covariate summaries of a slow feature being near-identical, or a duration
    # band left empty after the finite filter.
    fitted_design = design[usable]
    rank = int(np.linalg.matrix_rank(fitted_design))
    if rank < fitted_design.shape[1]:
        constant_columns = [labels[i] for i in range(fitted_design.shape[1])
                            if np.ptp(fitted_design[:, i]) == 0.0 and labels[i] != 'intercept']
        msg = (
            f"Design is rank-deficient: rank {rank} for {fitted_design.shape[1]} terms on "
            f"{fitted_design.shape[0]} rows, so the coefficients are not identifiable. "
            f"Constant columns: {constant_columns or 'none'}. Usual causes are covariate "
            f"summaries that are near-identical for a slow feature, a duration band left "
            f"empty after the finite filter, or too few rows for the term count."
        )
        raise ValueError(msg)

    family = (sm.families.Gamma(link=sm.families.links.Log()) if likelihood == 'gamma'
              else sm.families.Gaussian())
    model = sm.GLM(target[usable], fitted_design, family=family)
    fitted = model.fit(cov_type='cluster',
                       cov_kwds={'groups': session_ids[usable], 'use_correction': True})

    intervals = fitted.conf_int()
    terms = {}
    for position, name in enumerate(labels):
        terms[name] = {
            'coefficient': float(fitted.params[position]),
            'std_error': float(fitted.bse[position]),
            'z': float(fitted.tvalues[position]),
            'p_value': float(fitted.pvalues[position]),
            'ci_low': float(intervals[position, 0]),
            'ci_high': float(intervals[position, 1]),
        }
    return {
        'terms': terms,
        'n_rows_fitted': int(usable.sum()),
        'n_rows_dropped': n_dropped,
        'non_finite_by_term': non_finite_by_term,
        'n_sessions': int(np.unique(session_ids[usable]).size),
        'likelihood': likelihood,
    }


def behavioral_response_contrast(input_pickle_path: str | Path,
                                 output_directory: str | Path,
                                 settings_path: str | Path | None = None) -> dict[str, Any]:
    """
    Runs the vocal-versus-silence contrast and its time course, and saves them.

    Fits the same design twice over: once against the response averaged over the
    whole forward window, giving the headline numbers, and once per time bin,
    giving the adjusted time course.

    Parameters
    ----------
    input_pickle_path : str or pathlib.Path
        Anchor table written by ``BehavioralResponsePipeline``.
    output_directory : str or pathlib.Path
        Directory to publish the results pickle into.
    settings_path : str or pathlib.Path, optional
        Recorded in the run header for provenance; the analysis knobs themselves
        come from the artifact's ``_input_metadata``, so the extraction and the
        fit can never disagree about them.

    Returns
    -------
    results : dict
        ``per_feature`` (each with its ``window`` fit, ``time_course`` and derived
        ``likelihood``), ``duration_edges``, ``term_labels`` and the provenance
        block.
    """

    input_path = Path(input_pickle_path)
    with input_path.open('rb') as handle:
        artifact = pickle.load(handle)

    metadata = artifact['_input_metadata']
    analysis = metadata['analysis_specific']
    response_features = list(artifact['response_features'])
    response_likelihoods = dict(artifact['response_likelihoods'])
    n_duration_bins = int(analysis['duration_n_bins'])

    covariates = np.asarray(artifact['covariates'], dtype=float)
    covariate_labels = list(artifact['covariate_labels'])
    target = np.asarray(artifact['target'], dtype=float)
    target_bins = np.asarray(artifact['target_bins'], dtype=float)
    is_vocal = np.asarray(artifact['is_vocal'], dtype=float)
    bout_duration = np.asarray(artifact['bout_duration'], dtype=float)
    session_ids = np.asarray(artifact['session_ids'])

    print(format_run_header(
        task='BEHAVIORAL_RESPONSE_CONTRAST',
        engine='glm',
        feature=f'{len(response_features)} response feature(s)',
        split_strategy='cluster-robust (session)',
        n_splits=int(np.unique(session_ids).size),
        input_files={'input data': str(input_path), 'settings': str(settings_path)},
        output_directory=str(output_directory),
    ))

    # The design is identical for every feature and every time bin -- only the
    # target changes -- so it is built once. That is also what makes the time
    # course readable as one contrast resolved over time rather than a series of
    # separate analyses.
    band_index, duration_edges = duration_tercile_labels(
        bout_duration=bout_duration, is_vocal=is_vocal, n_bins=n_duration_bins)
    design, labels = build_design_matrix(
        covariates=covariates, is_vocal=is_vocal, band_index=band_index,
        n_bins=n_duration_bins, covariate_labels=covariate_labels)

    per_feature: dict[str, Any] = {}
    for feature_index, feature in enumerate(response_features):
        likelihood = response_likelihoods[feature]
        window_fit = fit_contrast(target=target[:, feature_index], design=design, labels=labels,
                                  session_ids=session_ids, likelihood=likelihood)
        for band in range(n_duration_bins):
            term = window_fit['terms'][f'vocal_duration_band_{band}']
            print(format_selection_step(
                'Contrast',
                feature=f'{feature} | band {band} '
                        f'[{duration_edges[band]:.2f}-{duration_edges[band + 1]:.2f}s]',
                metrics={'beta': term['coefficient'], 'se': term['std_error'],
                         'p': term['p_value']},
                decision='SIG' if term['p_value'] < 0.05 else 'ns',
            ))

        time_course = []
        for bin_index in range(target_bins.shape[2]):
            bin_fit = fit_contrast(target=target_bins[:, feature_index, bin_index],
                                   design=design, labels=labels,
                                   session_ids=session_ids, likelihood=likelihood)
            bin_fit['bin_index'] = bin_index
            time_course.append(bin_fit)

        per_feature[feature] = {'window': window_fit, 'time_course': time_course,
                                'likelihood': likelihood}

    results = {
        'per_feature': per_feature,
        'response_features': response_features,
        'response_likelihoods': response_likelihoods,
        'duration_edges': duration_edges,
        'term_labels': labels,
        'n_duration_bins': n_duration_bins,
        '_input_metadata': metadata,
    }

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = output_path / f"behavioral_response_contrast_{timestamp}.pkl"
    with atomic_output_path(save_path) as temporary_path:
        with Path(temporary_path).open('wb') as handle:
            pickle.dump(results, handle)

    print(format_run_summary(
        label=f"behavioral response contrast, {len(response_features)} feature(s) / "
              f"{window_fit['n_sessions']} sessions",
        metrics_by_strategy={
            feature: {
                'beta': per_feature[feature]['window']['terms']['vocal_duration_band_0']['coefficient'],
                'p': per_feature[feature]['window']['terms']['vocal_duration_band_0']['p_value'],
            }
            for feature in response_features
        },
        out_path=str(save_path),
    ))
    return results
