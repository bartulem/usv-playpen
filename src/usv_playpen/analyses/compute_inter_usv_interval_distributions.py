"""
@author: bartulem
Computes inter-vocalization interval (inter-USV interval) distributions across one or
more lists of session root directories, and (optionally) sweeps a 1D
mixture model -- either a Gaussian or a Student-t mixture, selected via
``model_class`` -- over a range of component counts on the pooled
log-inter-USV interval samples.

Convention
Within each session, ``track_names[0]`` is treated as the male and
``track_names[1]`` as the female. Inter-vocalization intervals are
computed only between consecutive USVs emitted by the *same* animal.
Two interval definitions are supported:

* ``s2s``: ``start[i+1] - start[i]`` (literature standard).
* ``e2s``: ``start[i+1] - stop[i]``  (alternate; can be negative for
  overlapping calls and is dropped via the strict ``> 0`` filter).

The number of dropped non-positive intervals is reported per session
per mode so a user can detect bias from overlapping calls in ``e2s``.
Non-positive intervals are most common in ``e2s`` (overlapping calls),
but ``s2s`` can also drop an interval when two same-animal calls share
an identical ``start`` timestamp (interval == 0), so the drop count is
meaningful -- if rarely nonzero -- in both modes.
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import numpy as np
import polars as pls

from ..os_utils import configure_path
from ._usv_io import (
    extract_session_metadata,
    load_and_filter_usv_data,
)
from .mixture_model_utils import (
    bootstrap_lrt,
    fit_log_gmm,
    fit_log_ig_mixture,
    fit_log_t_mixture,
    fit_nested_by_splitting,
    fit_tied_nested_by_splitting,
    fit_tied_scale_t_mixture,
    gmm_boundaries_logspace,
    gmm_cv_neg_loglik,
    ig_mixture_cv_neg_loglik,
    gmm_icl,
    ig_mixture_icl,
    report_gmm_stats,
    report_ig_mixture_stats,
    report_t_mixture_stats,
    select_n_components_step_up_lrt,
    t_mixture_cv_neg_loglik,
    t_mixture_icl,
    t_mixture_modes,
    tied_peak_bootstrap_lrt,
    tied_scale_n_parameters,
)
from .usv_interval_archive import git_sha_for_provenance, write_ivi_h5


def _read_session_lists(session_lists: list[str], message_output) -> list[str]:
    """
    Description
    -----------
    Reads one or more session-list text files (one session root per
    line, blank lines ignored) and returns the de-duplicated union of
    sessions, preserving first-seen order. Each path is run through
    ``configure_path`` so Mac/Linux/Windows paths in the input file all
    resolve correctly on the host platform.

    Parameters
    ----------
    session_lists (list[str])
        List of text file paths, each containing session roots.
    message_output (callable)
        Logging callable (typically ``print``).

    Returns
    -------
    sessions (list[str])
        De-duplicated, order-preserving list of session root paths
        (after platform conversion).
    """

    seen = set()
    sessions: list[str] = []
    for txt in session_lists:
        txt_path = pathlib.Path(configure_path(str(txt)))
        if not txt_path.exists():
            message_output(f"Session list file not found: {txt_path}")
            continue
        n_added = 0
        with txt_path.open('r') as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                resolved = configure_path(s)
                if resolved in seen:
                    continue
                seen.add(resolved)
                sessions.append(resolved)
                n_added += 1
        message_output(f"Loaded {n_added} session(s) from {txt_path}.")
    return sessions


def _session_source_map(session_lists: list[str]) -> dict[str, str]:
    """
    Description
    -----------
    Builds a session-root -> source-file-name mapping so the master
    DataFrame can carry a ``source_list`` column for multi-cohort
    comparisons. The first list a session appears in wins (mirroring
    the de-duplication policy in :func:`_read_session_lists`).

    Parameters
    ----------
    session_lists (list[str])
        List of text file paths.

    Returns
    -------
    mapping (dict)
        Mapping of resolved session root path -> stem of the text file
        it came from.
    """

    mapping: dict[str, str] = {}
    for txt in session_lists:
        txt_path = pathlib.Path(configure_path(str(txt)))
        if not txt_path.exists():
            continue
        with txt_path.open('r') as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                resolved = configure_path(s)
                if resolved not in mapping:
                    mapping[resolved] = txt_path.stem
    return mapping


def compute_session_usv_intervals(
    session_root: str,
    interval_type: str,
    exclude_noise_usvs: bool,
    call_type: str | None = None,
    adjacency: str = "filtered",
) -> dict:
    """
    Description
    -----------
    Computes inter-vocalization intervals for one session under one
    interval-definition mode.

    For each consecutive pair of USVs in the session's noise-filtered
    summary CSV, an interval is recorded only if both vocalizations
    are emitted by the same animal. The "current pointer" advances on
    every row regardless of whether an interval was recorded — this is
    intentional: it preserves chronological order of the conversation
    so a male->female->male triplet does not record a male-male
    interval that skips over the female call. Intervals strictly
    greater than zero are kept; non-positive intervals are counted and
    reported. These arise most often in ``e2s`` mode (overlapping
    calls), but ``s2s`` can also yield a zero interval when two
    same-animal calls share an identical ``start`` timestamp.

    Parameters
    ----------
    session_root (str)
        Absolute path to the session directory.
    interval_type (str)
        Either ``'s2s'`` (start-to-start) or ``'e2s'`` (end-to-start).
    exclude_noise_usvs (bool)
        Whether to drop the segments ``detect_usv_noise`` flagged as holding no
        vocalization before the intervals are measured.
    call_type (str or None)
        Restrict to ``'usv'`` (ultrasonic calls) or ``'squeak'``, or None for both.
        Defaults to None. An inter-USV interval analysis wants ``'usv'``: dropping
        noise alone leaves squeaks in the record, and a squeak between two ultrasonic
        calls suppresses the long interval those calls would have formed and
        contributes two short ones instead.
    adjacency (str)
        How "consecutive" is defined, and the two are not interchangeable:

        * ``'filtered'`` (default) removes the other call types first, so two calls of
          the target type are paired even when something else occurred between them.
          For a dense caller this measures bout structure.
        * ``'strict'`` requires both members of a pair to be adjacent in the session's
          full record, so any intervening vocalization by either animal breaks it.

        The distinction is immaterial for the male's ultrasonic calls, which are dense
        enough that his neighbours are almost always his own, but decisive for the
        female's squeaks: filtered gives her a 5.2 s median and a 1,080 s maximum
        because the definition spans arbitrary stretches of his song, while strict
        gives a 0.32 s median and a recognisable bout distribution.

    Returns
    -------
    out (dict)
        Keys: ``'male'``, ``'female'`` (np.ndarray of intervals in
        seconds), ``'n_dropped_male'``, ``'n_dropped_female'`` (int),
        ``'male_id'``, ``'female_id'``, ``'interval_type'``.
        Returns an empty dict ``{}`` if the session is missing
        tracking or USV files.
    """

    if interval_type not in ('s2s', 'e2s'):
        msg = f"Unknown interval_type '{interval_type}'; expected 's2s' or 'e2s'."
        raise ValueError(msg)
    if adjacency not in ('filtered', 'strict'):
        msg = f"Unknown adjacency '{adjacency}'; expected 'filtered' or 'strict'."
        raise ValueError(msg)

    try:
        metadata = extract_session_metadata(session_root)
    except (FileNotFoundError, IndexError):
        return {}

    raw_male_id = metadata['male_id']
    raw_female_id = metadata['female_id']
    male_id = str(raw_male_id).strip('\x00').strip()
    female_id = str(raw_female_id).strip('\x00').strip()

    try:
        # Under 'filtered' the other call types are removed before pairing, so the loader
        # does the restriction. Under 'strict' the full record has to stay intact -- an
        # intervening call must be able to break a pair -- so the restriction is applied
        # as a mask during pairing instead.
        usv_info = load_and_filter_usv_data(
            session_root=session_root,
            frame_rate=metadata['frame_rate'],
            exclude_noise_usvs=exclude_noise_usvs,
            call_type=call_type if adjacency == "filtered" else None,
        )
    except FileNotFoundError:
        return {}

    # column lookup for the two interval modes
    usv0_tag, usv1_tag = ("start", "start") if interval_type == "s2s" else ("stop", "start")

    if usv_info.height == 0:
        empty = np.array([], dtype=float)
        return {
            "male": empty, "female": empty,
            "n_dropped_male": 0, "n_dropped_female": 0,
            "male_id": male_id, "female_id": female_id,
            "interval_type": interval_type,
        }

    # build a polars frame with start, stop, sex (mirroring the
    # `with_columns(when().then()...)` pattern used in
    # ``visualizations.usv_summary_statistics.build_master_usv_dataframe``).
    # We accept either the raw H5-decoded ID or the stripped variant in
    # the emitter column, since the CSV's emitter values can lack the
    # null-byte / whitespace padding that ``track_names`` carry; comparing
    # only the raw form silently routes every row to "unassigned" when
    # the CSV is clean, which is what produced the previous M=0, F=0 result.
    sex_expr = (
        pls.when(pls.col("emitter") == raw_male_id).then(pls.lit("male"))
        .when(pls.col("emitter") == raw_female_id).then(pls.lit("female"))
        .when(pls.col("emitter") == male_id).then(pls.lit("male"))
        .when(pls.col("emitter") == female_id).then(pls.lit("female"))
        .otherwise(pls.lit("unassigned"))
        .alias("sex")
    )
    # Under 'strict' the target-type flag has to survive into the pairing loop, so it is
    # carried alongside start/stop/sex rather than applied as a row filter.
    target_expr = (
        pls.lit(True).alias("is_target") if (call_type is None or adjacency == "filtered")
        else (pls.col("squeak") if call_type == "squeak" else ~pls.col("squeak")).alias("is_target")
    )
    if "stop" in usv_info.columns:
        sub = usv_info.with_columns([sex_expr, target_expr]).select(
            ["start", "stop", "sex", "is_target"])
    else:
        sub = usv_info.with_columns([
            (pls.col("start") + pls.col("duration")).alias("stop"),
            sex_expr,
            target_expr,
        ]).select(["start", "stop", "sex", "is_target"])

    # extract to numpy arrays for the streaming pointer iteration; this avoids
    # per-row Polars overhead and keeps the same-emitter gating readable
    start_arr = sub["start"].to_numpy()
    stop_arr = sub["stop"].to_numpy()
    sex_arr = sub["sex"].to_numpy()
    target_arr = sub["is_target"].to_numpy()

    col_for_tag = {"start": start_arr, "stop": stop_arr}
    usv0_col = col_for_tag[usv0_tag]
    usv1_col = col_for_tag[usv1_tag]

    intervals = {"male": [], "female": []}
    n_dropped = {"male": 0, "female": 0}

    usv0_time = usv0_col[0]
    usv0_sex = sex_arr[0]
    usv0_target = target_arr[0]

    for r in range(1, len(sex_arr)):
        usv1_time = usv1_col[r]
        usv1_sex = sex_arr[r]
        usv1_target = target_arr[r]

        # same identified emitter (skip unassigned-unassigned pairs), and under 'strict'
        # both members must also be the requested call type -- anything else between them
        # breaks the pair rather than being skipped over
        if (usv0_sex == usv1_sex) and (usv0_sex in ("male", "female")) \
                and usv0_target and usv1_target:
            interval = usv1_time - usv0_time
            if interval > 0:
                intervals[usv0_sex].append(interval)
            else:
                n_dropped[usv0_sex] += 1

        usv0_time = usv0_col[r]
        usv0_sex = usv1_sex
        usv0_target = usv1_target

    male_arr = np.asarray(intervals["male"], dtype=float)
    female_arr = np.asarray(intervals["female"], dtype=float)

    # boundary safety: log() is taken downstream and requires strictly positive input
    assert np.all(male_arr > 0) and np.all(female_arr > 0), \
        "Non-positive intervals leaked past the > 0 filter; refusing to log."

    return {
        "male": male_arr,
        "female": female_arr,
        "n_dropped_male": int(n_dropped["male"]),
        "n_dropped_female": int(n_dropped["female"]),
        "male_id": male_id,
        "female_id": female_id,
        "interval_type": interval_type,
    }


def fit_mixture_model_sweep(
    intervals_by_key: dict[str, np.ndarray],
    n_components_min: int,
    n_components_max: int,
    n_repeats: int,
    max_modes_reported: int,
    random_seed_base: int,
    tau: float = 0.5,
    cv_n_folds: int = 5,
    cv_n_init: int = 5,
    mixture_model_n_init: int = 10,
    mixture_model_reg_covar: float = 1e-4,
    model_class: str = "gauss",
) -> pls.DataFrame:
    """
    Description
    -----------
    Sweeps mixture models of size ``n_components_min`` through
    ``n_components_max`` on each pooled inter-USV interval array
    (typically ``{'male': ..., 'female': ...}``), repeating each fit
    ``n_repeats`` times under different seeds. Despite the ``mixture_model`` in
    the name and in the ``mixture_model_fits`` archive group, this function is
    model-class agnostic: ``model_class`` selects a Gaussian mixture
    (``'gauss'``) or a Student-t mixture (``'t'``); the shipped config
    default is ``'t'``. Selection across reps and across K is
    delegated to the bootstrap-LRT step-up procedure in
    :func:`mixture_model_utils.bootstrap_lrt` /
    :func:`mixture_model_utils.select_n_components_step_up_lrt`; the IC columns
    in the returned table (``bic`` / ``aic`` / ``icl`` /
    ``cv_neg_loglik``) are diagnostic only.

    The returned tidy DataFrame is the model-parameter store consumed
    by the HDF5 archive layer: per-component log-mean / log-std /
    weight / nu (NaN for Gaussian) are recorded per row, plus
    inter-component decision boundaries in log-space and seconds at
    posterior threshold ``tau`` (Gaussian model class only; NaN-padded
    for the Student-t path, where boundaries lack a closed form).

    Parameters
    ----------
    intervals_by_key (dict)
        Mapping ``key -> np.ndarray`` of strictly positive intervals
        (``key`` typically ``'male'`` / ``'female'``).
    n_components_min (int)
        Minimum number of components in the sweep.
    n_components_max (int)
        Maximum number of components in the sweep.
    n_repeats (int)
        Number of EM-init repetitions per ``n_components`` value.
    max_modes_reported (int)
        Up to this many mixture modes are recorded per fit.
    random_seed_base (int)
        Base seed; rep ``r`` uses ``random_seed_base + r``.
    tau (float)
        Posterior threshold for inter-component boundaries; defaults
        to 0.5.
    cv_n_folds (int)
        K-fold splits for the cross-validated negative log-likelihood
        column.
    cv_n_init (int)
        EM restarts per CV fold.
    mixture_model_n_init (int)
        EM restarts for each in-sample fit.
    mixture_model_reg_covar (float)
        Variance floor passed to the EM solver.
    model_class (str)
        ``'gauss'`` (sklearn ``GaussianMixture``) or ``'t'``
        (:class:`mixture_model_utils.TMixture`).

    Returns
    -------
    df_results (pls.DataFrame)
        One row per ``(key, n_comp, rep)`` with: ``bic``, ``aic``,
        ``icl``, ``cv_neg_loglik`` (constant across reps for a given
        K), per-component ``logmean_k`` / ``logsd_k`` /
        ``median_sec_k`` / ``weight_k`` / ``nu_k`` (NaN-padded to
        ``n_components_max`` and NaN for ``nu_k`` under
        ``gauss``), the first ``max_modes_reported`` mixture modes
        (``mode_sec_k`` / ``density_k``), kept in ascending-location
        order, and Gaussian-only adjacent-component boundaries
        (``boundary_log_k`` / ``boundary_sec_k``). For the Student-t
        path (``model_class='t'``), ``mode_sec_k`` / ``density_k``
        carry per-component peak locations and the mixture density at
        each component peak rather than distinct Gaussian-style mixture
        modes.
    """

    if model_class not in ("gauss", "t", "ig"):
        msg = f"fit_mixture_model_sweep: model_class must be 'gauss', 't' or 'ig', got {model_class!r}."
        raise ValueError(
            msg
        )

    n_comps = list(range(n_components_min, n_components_max + 1))

    # Cross-validated negative log-likelihood is independent of the rep
    # dimension (KFold averages out EM seed noise within each fold), so
    # we compute it once per (key, n_comp) and broadcast it across reps.
    # The CV implementation dispatches on `model_class`.
    cv_per_key_n: dict[str, dict[int, float]] = {}
    for key, iui in intervals_by_key.items():
        if len(iui) < cv_n_folds:
            continue
        cv_per_key_n[key] = {}
        for n_components in n_comps:
            if model_class == "gauss":
                cv_val = gmm_cv_neg_loglik(
                    intervals_sec=iui,
                    n_components=n_components,
                    seed=random_seed_base,
                    n_folds=cv_n_folds,
                    n_init=cv_n_init,
                    reg_covar=mixture_model_reg_covar,
                )
            elif model_class == "ig":
                cv_val = ig_mixture_cv_neg_loglik(
                    intervals_sec=iui,
                    n_components=n_components,
                    seed=random_seed_base,
                    n_folds=cv_n_folds,
                    n_init=cv_n_init,
                    reg_covar=mixture_model_reg_covar,
                )
            else:  # t-mixture
                cv_val = t_mixture_cv_neg_loglik(
                    intervals_sec=iui,
                    n_components=n_components,
                    seed=random_seed_base,
                    n_folds=cv_n_folds,
                    n_init=max(1, cv_n_init - 2),  # t-mix EM is heavier; trim per-fold inits
                    reg_covar=mixture_model_reg_covar,
                )
            cv_per_key_n[key][n_components] = cv_val

    results: dict = {
        "sex": [], "n_comp": [], "rep": [], "model_class": [],
        "bic": [], "aic": [], "icl": [], "cv_neg_loglik": [],
    }

    for i in range(1, n_components_max + 1):
        results[f"logmean_{i}"] = []
        results[f"logsd_{i}"] = []
        results[f"median_sec_{i}"] = []
        results[f"weight_{i}"] = []
        # per-component degrees of freedom; populated for t-mixtures only,
        # NaN-filled for Gaussian mixtures so the schema is class-agnostic.
        results[f"nu_{i}"] = []
        # per-component inverse-Gaussian shape; populated for model_class='ig'
        # only, NaN-filled otherwise. For IG rows, logmean_k = log(mu_k) (log
        # of the component MEAN) and logsd_k is NaN (an IG has no log-sd).
        results[f"lambda_{i}"] = []

    for i in range(1, max_modes_reported + 1):
        results[f"mode_sec_{i}"] = []
        results[f"density_{i}"] = []

    for i in range(1, n_components_max):
        results[f"boundary_log_{i}"] = []
        results[f"boundary_sec_{i}"] = []

    for key, iui in intervals_by_key.items():
        if len(iui) < 2:
            continue
        log_iui = np.log(iui).reshape(-1, 1)

        for r in range(n_repeats):
            # Student-t mixtures are fitted up a LADDER within each repeat: the K+1 model starts
            # from the K solution instead of from scratch. Random restarts find a K=5 solution that
            # scores 39.7 nats better and places a 0.7%-weight component at 28.8 ms, below the
            # dominant 63 ms mode, in the region the segmenter's 15 ms gap-fill and the 16.384 ms
            # DAS seam control rather than the animal. Since component 0 is what the modeling
            # pipeline reads for its inter-bout threshold, that solution yields 36 ms in place of
            # 122 ms. The ladder lands on the interpretable solution at every K from 2 to 6, and
            # n_comps is ascending so the previous fit is always available.
            ladder_model = None
            for n_components in n_comps:
                seed = random_seed_base + r
                if model_class == "gauss":
                    model, model_order = fit_log_gmm(
                        iui, n_components=n_components, seed=seed,
                        n_init=mixture_model_n_init, reg_covar=mixture_model_reg_covar,
                    )
                    logmeans, logsds, modes_log, densities = report_gmm_stats(model, model_order)
                    weights = model.weights_.flatten()[model_order]
                    nus = np.full(n_components, np.nan, dtype=float)  # not applicable for gauss
                    icl = gmm_icl(model, log_iui)
                elif model_class == "ig":
                    model, model_order = fit_log_ig_mixture(
                        iui, n_components=n_components, seed=seed,
                        n_init=mixture_model_n_init, reg_covar=mixture_model_reg_covar,
                    )
                    logmeans, lambdas, weights, modes_log, densities = report_ig_mixture_stats(model, model_order)
                    logsds = np.full(n_components, np.nan, dtype=float)  # no log-sd for IG
                    nus = np.full(n_components, np.nan, dtype=float)     # not applicable for IG
                    icl = ig_mixture_icl(model, log_iui)
                else:  # t-mixture
                    if ladder_model is None:
                        model, model_order = fit_log_t_mixture(
                            iui, n_components=n_components, seed=seed,
                            n_init=mixture_model_n_init, reg_covar=mixture_model_reg_covar,
                        )
                    else:
                        model, _ = fit_nested_by_splitting(
                            log_iui.ravel(), ladder_model, n_init=mixture_model_n_init,
                            seed=seed, reg_covar=mixture_model_reg_covar,
                        )
                        model_order = np.argsort(np.asarray(model.means_, dtype=float).ravel())
                    ladder_model = model
                    logmeans, logsds, nus, weights, _ = report_t_mixture_stats(model, model_order)
                    # Modes are the local maxima of the mixture density, not the
                    # per-component means: components overlap, so a K-component fit
                    # routinely has fewer than K modes, with the extra components
                    # tiling a skewed shoulder rather than marking a peak. This row
                    # previously stored the component means under the mode columns,
                    # which reported structure the density does not have -- male e2s
                    # K=6 lists components at 23 ms and 63 ms but the density has a
                    # single maximum at 63 ms and no local maximum below it.
                    modes_arr, densities_arr = t_mixture_modes(model)
                    mode_order = np.argsort(modes_arr)
                    modes_log = modes_arr[mode_order]
                    densities = densities_arr[mode_order]
                    icl = t_mixture_icl(model, log_iui)

                bic = float(model.bic(log_iui))
                aic = float(model.aic(log_iui))
                cv_nll = float(cv_per_key_n.get(key, {}).get(n_components, np.nan))

                results["sex"].append(key)
                results["n_comp"].append(n_components)
                results["rep"].append(r)
                results["model_class"].append(model_class)
                results["bic"].append(bic)
                results["aic"].append(aic)
                results["icl"].append(icl)
                results["cv_neg_loglik"].append(cv_nll)

                # per-component (filled) and NaN-padded slots
                for k in range(n_components):
                    results[f"logmean_{k+1}"].append(float(logmeans[k]))
                    results[f"logsd_{k+1}"].append(float(logsds[k]))
                    results[f"median_sec_{k+1}"].append(float(np.exp(logmeans[k])))
                    results[f"weight_{k+1}"].append(float(weights[k]))
                    results[f"nu_{k+1}"].append(float(nus[k]))
                    results[f"lambda_{k+1}"].append(
                        float(lambdas[k]) if model_class == "ig" else np.nan
                    )
                for k in range(n_components, n_components_max):
                    results[f"logmean_{k+1}"].append(np.nan)
                    results[f"logsd_{k+1}"].append(np.nan)
                    results[f"median_sec_{k+1}"].append(np.nan)
                    results[f"weight_{k+1}"].append(np.nan)
                    results[f"nu_{k+1}"].append(np.nan)
                    results[f"lambda_{k+1}"].append(np.nan)

                # mixture modes (Gaussian) or per-component peaks (t), kept in
                # ascending-location order; first max_modes_reported recorded
                # (no density-based selection -- report_gmm_stats re-sorts by location)
                modes_sec = np.exp(modes_log) if modes_log.size else modes_log
                n_modes = len(modes_sec)
                for k in range(min(max_modes_reported, n_modes)):
                    results[f"mode_sec_{k+1}"].append(float(modes_sec[k]))
                    results[f"density_{k+1}"].append(float(densities[k]))
                for k in range(n_modes, max_modes_reported):
                    results[f"mode_sec_{k+1}"].append(np.nan)
                    results[f"density_{k+1}"].append(np.nan)

                # adjacent-component boundaries (Gaussian-only; NaN-fill for t / ig)
                if model_class == "gauss":
                    boundaries_log, boundaries_sec = gmm_boundaries_logspace(model, tau=tau)
                    for k in range(n_components - 1):
                        results[f"boundary_log_{k+1}"].append(float(boundaries_log[k]))
                        results[f"boundary_sec_{k+1}"].append(float(boundaries_sec[k]))
                    for k in range(n_components - 1, n_components_max - 1):
                        results[f"boundary_log_{k+1}"].append(np.nan)
                        results[f"boundary_sec_{k+1}"].append(np.nan)
                else:
                    for k in range(n_components_max - 1):
                        results[f"boundary_log_{k+1}"].append(np.nan)
                        results[f"boundary_sec_{k+1}"].append(np.nan)

    return pls.DataFrame(results)



def summarize_interval_pool(values: np.ndarray, n_sessions: int) -> dict:
    """
    Description
    -----------
    Descriptive summary of one interval pool, written for every pool whether or not it is fitted.

    Pools too small for mixture modelling -- the female's squeaks give 237 end-to-start
    intervals across 70 sessions -- are reported by these numbers alone, so they have to be
    computed identically for the pools that are fitted, where they sit beside the model.

    Parameters
    ----------
    values (np.ndarray)
        A (n_intervals,) ndarray of intervals in seconds.
    n_sessions (int)
        Number of sessions contributing at least one interval.

    Returns
    -------
    summary (dict)
        Count, session count, and the 5th, 25th, 50th, 75th and 95th percentiles in seconds
        (NaN when the pool is empty).
    """

    quantiles = (np.percentile(values, [5, 25, 50, 75, 95]) if values.size
                 else np.full(5, np.nan))
    return {
        "n_intervals": int(values.size),
        "n_sessions": int(n_sessions),
        "p05_s": float(quantiles[0]),
        "p25_s": float(quantiles[1]),
        "median_s": float(quantiles[2]),
        "p75_s": float(quantiles[3]),
        "p95_s": float(quantiles[4]),
    }


def fit_tied_peak_ladder_and_lrt(
    values: np.ndarray,
    session_labels: np.ndarray,
    pool_identity: dict,
    peak_grid: list[int],
    n_background: int,
    n_init: int,
    n_init_boot: int,
    reg_covar: float,
    B: int,
    n_subsample: int,
    alpha: float,
    bonferroni: bool,
    n_design_bootstrap: int,
    seed: int,
    n_jobs: int,
    message,
) -> tuple[pls.DataFrame, pls.DataFrame, pls.DataFrame, pls.DataFrame, int, float]:
    """
    Description
    -----------
    Fits the tied-scale peak ladder on one pool and runs the session-corrected peak-count test.

    The model is a Student-t mixture with ``n_peak`` components sharing one fitted width and
    ``n_background`` components with free widths (:func:`fit_tied_scale_t_mixture`). The
    unconstrained mixture has no stopping point on the male end-to-start pool because every
    extra component tiles a little more of the misfit around the second peak; separating the
    two roles removes the incentive to split a peak, and the peak count is the quantity a
    timescale claim is about.

    The ladder is fitted on the full pool, each peak count split-initialised from the one below
    (:func:`fit_tied_nested_by_splitting`), for every count in ``peak_grid`` plus one. The test
    runs every rung ``n -> n+1`` for ``n`` in ``peak_grid`` through
    :func:`tied_peak_bootstrap_lrt` with session labels, so each observed statistic is divided
    by its session design effect before being scored against the parametric null (Rao & Scott,
    1981). Intervals are nested in sessions and the iid null has no session structure; without
    the correction the male end-to-start ladder was non-monotonic (2v3 kept, 3v4 rejected), with
    it every rung past 1v2 keeps. All rungs are run rather than stopping at the first keep, so
    later rungs are reported as tested rather than merely unreached; selection is still the
    step-up rule applied to the corrected p-values.

    Parameters
    ----------
    values (np.ndarray)
        A (n_intervals,) ndarray of intervals in seconds.
    session_labels (np.ndarray)
        A (n_intervals,) ndarray of session identifiers aligned with ``values``.
    pool_identity (dict)
        ``{'sex', 'call_type', 'adjacency'}`` copied onto every output row.
    peak_grid (list[int])
        Null peak counts to test; the ladder is fitted up to ``max(peak_grid) + 1``.
    n_background (int)
        Background component count, fixed under every hypothesis.
    n_init (int)
        Restarts for the observed fits.
    n_init_boot (int)
        Restarts per bootstrap refit.
    reg_covar (float)
        Component variance floor.
    B (int)
        Bootstrap replicates per rung.
    n_subsample (int)
        Test subsample size.
    alpha (float)
        Family-wise significance level.
    bonferroni (bool)
        Whether ``alpha`` is divided across the rungs.
    n_design_bootstrap (int)
        Session-resampling replicates for each design effect.
    seed (int)
        Seed for fits, subsample and replicates.
    n_jobs (int)
        Parallel workers for the replicates.
    message (callable)
        Logging callable.

    Returns
    -------
    tied_fits (pls.DataFrame)
        One row per (peak count, component): role, location, scale, nu, weight, plus fit-level
        log-likelihood, parameter count, BIC and shared scale.
    tied_modes (pls.DataFrame)
        One row per (peak count, density mode): mode location in seconds and density.
    peak_lrt (pls.DataFrame)
        One row per rung: raw and corrected statistics, design effect, p-values, threshold.
    peak_lrt_null (pls.DataFrame)
        Long-form null draws, one row per (rung, replicate).
    selected_n_peak (int)
        Step-up selection on the corrected p-values.
    alpha_effective (float)
        The per-rung significance level used.
    """

    log_values = np.log(values)
    fit_rows: list[dict] = []
    mode_rows: list[dict] = []
    model = None
    for n_peak in range(min(peak_grid), max(peak_grid) + 2):
        if model is None:
            model, log_likelihood, shared = fit_tied_scale_t_mixture(
                log_values, n_peak, n_background, seed=seed, n_init=n_init, reg_covar=reg_covar)
        else:
            model, log_likelihood, shared = fit_tied_nested_by_splitting(
                log_values, model, n_peak - 1, n_background, seed=seed, n_init=n_init,
                reg_covar=reg_covar)
        n_parameters = tied_scale_n_parameters(n_peak, n_background)
        bic = n_parameters * np.log(values.size) - 2.0 * log_likelihood
        means = np.asarray(model.means_, dtype=float).ravel()
        scales = np.sqrt(np.asarray(model.covariances_, dtype=float).ravel())
        nus = np.asarray(model.nus_, dtype=float).ravel()
        weights = np.asarray(model.weights_, dtype=float).ravel()
        for component in range(means.size):
            fit_rows.append({
                **pool_identity,
                "n_peak": int(n_peak),
                "n_background": int(n_background),
                "component": int(component),
                "role": "peak" if component < n_peak else "background",
                "logmean": float(means[component]),
                "median_sec": float(np.exp(means[component])),
                "logscale": float(scales[component]),
                "nu": float(nus[component]),
                "weight": float(weights[component]),
                "log_likelihood": float(log_likelihood),
                "n_parameters": int(n_parameters),
                "bic": float(bic),
                "shared_peak_scale": float(shared),
            })
        modes_log, densities = t_mixture_modes(model)
        order = np.argsort(modes_log)
        for rank, index in enumerate(order):
            mode_rows.append({
                **pool_identity,
                "n_peak": int(n_peak),
                "mode_index": int(rank),
                "mode_sec": float(np.exp(modes_log[index])),
                "density": float(densities[index]),
            })
        message(f"    tied {n_peak} peak(s) + {n_background} bkgd: logL={log_likelihood:.1f} "
                f"BIC={bic:.1f} shared sigma={shared:.3f} modes="
                + ", ".join(f"{np.exp(modes_log[i]):.4f}" for i in order))

    alpha_effective = alpha / len(peak_grid) if bonferroni else alpha
    lrt_rows: list[dict] = []
    null_rows: list[dict] = []
    selected_n_peak = None
    for n_peak in peak_grid:
        result = tied_peak_bootstrap_lrt(
            intervals_sec=values, n_peak_null=n_peak, n_background=n_background, B=B,
            n_subsample=n_subsample, n_init_obs=n_init, n_init_boot=n_init_boot,
            reg_covar=reg_covar, seed=seed, n_jobs=n_jobs, session_labels=session_labels,
            n_design_bootstrap=n_design_bootstrap)
        threshold = float(np.percentile(result["lr_null"], 100.0 * (1.0 - alpha_effective)))
        rejected = bool(result["p_value_corrected"] < alpha_effective)
        if not rejected and selected_n_peak is None:
            selected_n_peak = n_peak
        message(f"    peaks {n_peak}v{n_peak + 1}: LR={result['lr_obs']:.2f} "
                f"deff={result['design_effect']:.2f} LR_corr={result['lr_corrected']:.2f} "
                f"thr={threshold:.2f} p={result['p_value']:.4f} "
                f"p_corr={result['p_value_corrected']:.4f} -> "
                f"{'reject' if rejected else 'keep'}")
        lrt_rows.append({
            **pool_identity,
            "n_peak_null": int(n_peak),
            "n_peak_alt": int(n_peak + 1),
            "n_background": int(n_background),
            "lr_obs": float(result["lr_obs"]),
            "design_effect": float(result["design_effect"]),
            "design_effect_raw": float(result["design_effect_raw"]),
            "effective_n": float(result["effective_n"]),
            "lr_corrected": float(result["lr_corrected"]),
            "null_p95": float(result["null_p95"]),
            "threshold": threshold,
            "p_value": float(result["p_value"]),
            "p_value_corrected": float(result["p_value_corrected"]),
            "negative_fraction": float(result["negative_fraction"]),
            "B": int(result["B"]),
            "n_subsample": int(result["n_subsample"]),
            "alpha_used": float(alpha_effective),
            "rejected": rejected,
        })
        for b_index, lr_b in enumerate(result["lr_null"]):
            null_rows.append({**pool_identity, "n_peak_null": int(n_peak),
                              "b": int(b_index), "lr_b": float(lr_b)})
    if selected_n_peak is None:
        selected_n_peak = max(peak_grid) + 1
    message(f"    step-up selection (session-corrected, alpha_eff={alpha_effective:.4g}): "
            f"{selected_n_peak} peak(s) + {n_background} background")
    return (pls.DataFrame(fit_rows), pls.DataFrame(mode_rows), pls.DataFrame(lrt_rows),
            pls.DataFrame(null_rows), int(selected_n_peak), float(alpha_effective))


class InterUSVIntervalCalculator:
    """
    Cross-session inter-vocalization-interval driver. Reads one or more
    session-list text files, computes per-session inter-USV intervals in each
    interval-definition mode (``s2s`` and ``e2s``), runs the optional
    Gaussian / t-mixture sweep and bootstrap LRT, and consolidates the
    whole run into a single ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5``
    archive (see :mod:`usv_playpen.analyses.usv_interval_archive`).
    """

    def __init__(self, **kwargs):
        """
        Description
        -----------
        Initialises the InterUSVIntervalCalculator. The keyword arguments are
        validated against the keys the class consumes (an unknown key raises
        ``TypeError``) and then captured into ``self.__dict__``, matching the
        convention used by :class:`FeatureZoo`.

        Parameters
        ----------
        input_parameter_dict (dict)
            Full ``analyses_settings`` dictionary; the
            ``compute_inter_usv_interval_distributions`` block is read from it.
        message_output (callable)
            Logging callable (typically ``print``).

        Returns
        -------
        None
        """

        expected_kwargs = {'input_parameter_dict', 'message_output'}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword argument(s) "
                            f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                            f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def save_inter_usv_interval_distributions_to_file(self) -> None:
        """
        Description
        -----------
        Reads the configured session lists, computes per-session inter-USV intervals
        in each requested ``interval_type``, pools them into a master
        DataFrame, and writes a single self-describing HDF5 archive
        ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5`` to
        ``output_directory``. The timestamp in the filename is
        captured at the start of this routine and is also stored in
        the archive's ``created_at_iso`` root attribute, so file name
        and provenance metadata stay coherent.

        The archive structure (see :mod:`usv_playpen.analyses.usv_interval_archive`
        for the full schema):

        * Root ``/attrs`` -- every JSON parameter that drove the run,
          plus ``created_at_iso``, ``git_sha``, ``source_lists`` and
          ``n_sessions_loaded``.
        * ``/<mode>/intervals`` -- tidy one-row-per-inter-USV interval table with
          ``session_id``, ``source_list``, ``interval_type``, ``sex``,
          ``interval_s``, ``log_interval``, ``male_id``, ``female_id``.
        * ``/<mode>/drop_counts`` -- per-sex count of dropped
          non-positive intervals (only meaningful for ``e2s`` mode).
        * ``/<mode>/mixture_model_fits`` (only when ``fit_mixture_model`` is true) -- the
          full Gaussian / t-mixture sweep with all four ICs (``bic``,
          ``aic``, ``icl``, ``cv_neg_loglik``) and per-component
          parameters (``logmean_k``, ``logsd_k``, ``weight_k``,
          ``nu_k``) per ``(sex, n_comp, rep)`` row. This table doubles
          as the model-parameter store; downstream plot helpers pick
          the best-rep row to rebuild the fitted mixture without
          refitting.
        * ``/<mode>/bootstrap_lrt`` -- parametric bootstrap LRT per
          ``(sex, K_null, K_alt)`` pair plus the per-sex step-up
          selection in the constant ``K_selected_step_up`` column.
        * ``/<mode>/bootstrap_lrt_null`` -- long-form null
          distribution: one row per ``(sex, K_null, K_alt, b)``, used
          to re-render the panel plot without re-running the test.
        * ``/<mode>/attrs`` -- ``alpha_effective`` and the per-sex
          step-up selected K (``K_selected_male``, ``K_selected_female``).

        Parameters
        ----------

        Returns
        -------
        None
        """

        cfg = self.input_parameter_dict['compute_inter_usv_interval_distributions']

        session_lists = cfg['session_lists']
        output_directory = cfg['output_directory']
        # Both interval definitions are computed unconditionally; the cost
        # is dominated by the per-session USV CSV pass which is shared.
        interval_types = ("s2s", "e2s")
        exclude_noise_usvs = cfg['exclude_noise_usvs']
        fit_mixture_model = cfg['fit_mixture_model']
        n_components_min = cfg['n_components_min']
        n_components_max = cfg['n_components_max']
        n_repeats = cfg['n_repeats']
        max_modes_reported = cfg['max_modes_reported']
        random_seed_base = cfg['random_seed_base']
        cv_n_folds = cfg['cv_n_folds']
        cv_n_init = cfg['cv_n_init']
        mixture_model_n_init = cfg['mixture_model_n_init']
        mixture_model_reg_covar = cfg['mixture_model_reg_covar']
        tau = cfg['tau']
        model_class = cfg['model_class']
        bootstrap_lrt_B = cfg['bootstrap_lrt_B']
        bootstrap_lrt_n_subsample = cfg['bootstrap_lrt_n_subsample']
        bootstrap_lrt_alpha = cfg['bootstrap_lrt_alpha']
        bootstrap_lrt_n_jobs = cfg['bootstrap_lrt_n_jobs']
        # EM restarts for the bootstrap REFITS, separate from the observed fit's
        # mixture_model_n_init. Previously derived as max(1, mixture_model_n_init - 7),
        # which pinned it to 3 and could not be raised without also changing the
        # observed fit. Restarts dominate this test's cost while barely moving the
        # answer: measured on male e2s K=4 vs K=5, going 3 -> 10 -> 20 moved the
        # failed-refit rate 75% -> 70% -> 65% and p 0.0090 -> 0.0080 -> 0.0070, at
        # 38 and 57 minutes for a SINGLE pair against ~56 minutes for the whole
        # analysis at 3. The failures are not local optima more starts would clear:
        # the larger K is not identifiable there, which is itself the finding.
        bootstrap_lrt_n_init = cfg['bootstrap_lrt_n_init']
        bootstrap_lrt_bonferroni = cfg['bootstrap_lrt_bonferroni']
        # Pools are declared in settings rather than hardcoded: each names the emitter, the call
        # type (ultrasonic calls or squeaks) and the adjacency rule, and whether it is fitted.
        # Male USVs and female squeaks are different measurements of different vocalizations;
        # the adjacency rule matters for the sparse caller (see compute_session_usv_intervals).
        interval_pools = cfg['interval_pools']
        min_intervals_for_fitting = cfg['min_intervals_for_fitting']
        fit_tied_model = cfg['fit_tied_model']
        tied_n_background = cfg['tied_n_background']
        tied_peak_grid = list(cfg['tied_peak_grid'])
        tied_n_init = cfg['tied_n_init']
        tied_n_init_boot = cfg['tied_n_init_boot']
        tied_design_bootstrap = cfg['tied_design_bootstrap']

        if model_class not in ('gauss', 't', 'ig'):
            msg = f"compute_inter_usv_interval_distributions: model_class must be 'gauss', 't' or 'ig', got {model_class!r}."
            raise ValueError(
                msg
            )

        message = self.message_output

        if not session_lists:
            message("compute_inter_usv_interval_distributions: no session_lists configured; skipping.")
            return

        out_dir = pathlib.Path(configure_path(str(output_directory))) if output_directory else None
        if out_dir is None:
            msg = "compute_inter_usv_interval_distributions: output_directory must be set."
            raise ValueError(msg)
        out_dir.mkdir(parents=True, exist_ok=True)

        sessions = _read_session_lists(session_lists, message)
        source_map = _session_source_map(session_lists)

        if not sessions:
            message("compute_inter_usv_interval_distributions: zero sessions resolved from the session_lists.")
            return

        run_started_at = datetime.now()
        # Filename-friendly timestamp at second resolution; same value
        # is propagated to the HDF5 ``created_at_iso`` attribute so the
        # file's name and its provenance metadata stay in lock-step.
        run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")

        message(
            f"compute_inter_usv_interval_distributions: {len(sessions)} session(s) resolved across "
            f"{len(session_lists)} list file(s); started at "
            f"{run_started_at.hour:02d}:{run_started_at.minute:02d}:{run_started_at.second:02d}."
        )

        # Per-mode artifact dict assembled across the loop, written
        # once at the end via a single HDF5 archive call. No per-mode
        # CSV side-effects.
        per_mode: dict[str, dict] = {}
        sessions_with_data: set[str] = set()

        usv_interval_schema = {
            "session_id": pls.Utf8,
            "source_list": pls.Utf8,
            "interval_type": pls.Utf8,
            "sex": pls.Utf8,
            "call_type": pls.Utf8,
            "adjacency": pls.Utf8,
            "interval_s": pls.Float64,
            "log_interval": pls.Float64,
            "male_id": pls.Utf8,
            "female_id": pls.Utf8,
        }

        for interval_type in interval_types:
            tidy_frames: list[pls.DataFrame] = []
            drop_rows: list[dict] = []
            summary_rows: list[dict] = []
            pool_values: dict[tuple, np.ndarray] = {}
            pool_sessions: dict[tuple, np.ndarray] = {}
            tied_tables: dict[str, list[pls.DataFrame]] = {
                "tied_fits": [], "tied_modes": [], "peak_lrt": [], "peak_lrt_null": []}
            mode_attrs: dict = {}

            for spec in interval_pools:
                sex = spec['sex']
                call_type = spec['call_type']
                adjacency = spec['adjacency']
                identity = {"sex": sex, "call_type": call_type, "adjacency": adjacency}
                key = (sex, call_type, adjacency)
                arrays: list[np.ndarray] = []
                labels: list[np.ndarray] = []
                n_dropped = 0
                rows: list[dict] = []

                for session_root in sessions:
                    usv_interval = compute_session_usv_intervals(
                        session_root=session_root,
                        interval_type=interval_type,
                        exclude_noise_usvs=exclude_noise_usvs,
                        call_type=call_type,
                        adjacency=adjacency,
                    )
                    if not usv_interval:
                        continue
                    sessions_with_data.add(session_root)
                    arr = usv_interval[sex]
                    n_dropped += usv_interval[f"n_dropped_{sex}"]
                    if arr.size == 0:
                        continue
                    session_id = pathlib.Path(session_root).name
                    arrays.append(arr)
                    labels.append(np.full(arr.size, session_id, dtype=object))
                    rows.append({
                        "session_id": [session_id] * arr.size,
                        "source_list": [source_map.get(session_root, "")] * arr.size,
                        "interval_type": [interval_type] * arr.size,
                        "sex": [sex] * arr.size,
                        "call_type": [call_type] * arr.size,
                        "adjacency": [adjacency] * arr.size,
                        "interval_s": arr.astype(float),
                        "log_interval": np.log(arr),
                        "male_id": [usv_interval["male_id"]] * arr.size,
                        "female_id": [usv_interval["female_id"]] * arr.size,
                    })

                values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
                session_labels = (np.concatenate(labels) if labels
                                  else np.array([], dtype=object))
                pool_values[key] = values
                pool_sessions[key] = session_labels
                for block in rows:
                    tidy_frames.append(pls.DataFrame(block, schema=usv_interval_schema))
                drop_rows.append({**identity, "n_dropped": int(n_dropped)})
                summary = summarize_interval_pool(values, len(labels))
                summary_rows.append({**identity, **summary, "fitted": False})
                message(
                    f"  [{interval_type}] {sex} {call_type} ({adjacency}): "
                    f"n={values.size} from {len(labels)} sessions, "
                    f"median={summary['median_s']:.4f} s (dropped non-positive: {n_dropped})"
                )

                if fit_tied_model and spec['fit']:
                    if values.size < min_intervals_for_fitting:
                        message(f"    below min_intervals_for_fitting ({min_intervals_for_fitting}); "
                                "archived for description only.")
                        continue
                    fits, modes, lrt, lrt_null, selected, alpha_eff = fit_tied_peak_ladder_and_lrt(
                        values=values, session_labels=session_labels, pool_identity=identity,
                        peak_grid=tied_peak_grid, n_background=tied_n_background,
                        n_init=tied_n_init, n_init_boot=tied_n_init_boot,
                        reg_covar=mixture_model_reg_covar, B=bootstrap_lrt_B,
                        n_subsample=bootstrap_lrt_n_subsample, alpha=bootstrap_lrt_alpha,
                        bonferroni=bootstrap_lrt_bonferroni,
                        n_design_bootstrap=tied_design_bootstrap, seed=random_seed_base,
                        n_jobs=bootstrap_lrt_n_jobs, message=message,
                    )
                    tied_tables["tied_fits"].append(fits)
                    tied_tables["tied_modes"].append(modes)
                    tied_tables["peak_lrt"].append(lrt)
                    tied_tables["peak_lrt_null"].append(lrt_null)
                    summary_rows[-1]["fitted"] = True
                    mode_attrs[f"selected_n_peak_{sex}_{call_type}"] = int(selected)
                    mode_attrs[f"alpha_effective_{sex}_{call_type}"] = float(alpha_eff)

            tidy_df = (pls.concat(tidy_frames) if tidy_frames
                       else pls.DataFrame(schema=usv_interval_schema))
            mode_payload: dict = {
                "attrs": mode_attrs,
                "intervals": tidy_df,
                "drop_counts": pls.DataFrame(drop_rows),
                "pool_summary": pls.DataFrame(summary_rows),
                "mixture_model_fits": None,
                "bootstrap_lrt": None,
                "bootstrap_lrt_null": None,
            }
            for table_name, frames in tied_tables.items():
                mode_payload[table_name] = pls.concat(frames) if frames else None

            # The unconstrained K sweep and its LRT are kept available behind fit_mixture_model
            # but are off by default: the tied peak ladder above is the model and the test this
            # analysis reports. When enabled they run on the fitted pools only, keyed by sex.
            if fit_mixture_model:
                fit_pools = {spec['sex']: pool_values[(spec['sex'], spec['call_type'], spec['adjacency'])]
                             for spec in interval_pools if spec['fit']}
                fit_pools = {k: v for k, v in fit_pools.items() if v.size >= 2}
                if fit_pools:
                    mode_payload["mixture_model_fits"] = fit_mixture_model_sweep(
                        intervals_by_key=fit_pools,
                        n_components_min=n_components_min,
                        n_components_max=n_components_max,
                        n_repeats=n_repeats,
                        max_modes_reported=max_modes_reported,
                        random_seed_base=random_seed_base,
                        tau=tau,
                        cv_n_folds=cv_n_folds,
                        cv_n_init=cv_n_init,
                        mixture_model_n_init=mixture_model_n_init,
                        mixture_model_reg_covar=mixture_model_reg_covar,
                        model_class=model_class,
                    )
                    lrt_rows: list[dict] = []
                    lrt_null_rows: list[dict] = []
                    lrt_pairs = list(range(n_components_min, n_components_max))
                    for sex, pool in fit_pools.items():
                        pair_results: dict = {}
                        for K_n in lrt_pairs:
                            res = bootstrap_lrt(
                                intervals_sec=pool, K_null=K_n, K_alt=K_n + 1,
                                B=bootstrap_lrt_B, n_subsample=bootstrap_lrt_n_subsample,
                                model_class=model_class, n_init_obs=mixture_model_n_init,
                                n_init_boot=bootstrap_lrt_n_init,
                                reg_covar=mixture_model_reg_covar, seed=random_seed_base,
                                n_jobs=bootstrap_lrt_n_jobs,
                            )
                            pair_results[(K_n, K_n + 1)] = res
                        n_tests = len(pair_results)
                        alpha_eff = (bootstrap_lrt_alpha / n_tests
                                     if (bootstrap_lrt_bonferroni and n_tests > 0)
                                     else bootstrap_lrt_alpha)
                        K_sel = select_n_components_step_up_lrt(pair_results, alpha=alpha_eff)
                        mode_payload["attrs"][f"K_selected_{sex}"] = int(K_sel)
                        for (K_n, K_a), res in pair_results.items():
                            lrt_rows.append({
                                "sex": sex, "K_null": res["K_null"], "K_alt": res["K_alt"],
                                "lr_obs": res["lr_obs"], "null_mean": res["null_mean"],
                                "null_p95": res["null_p95"], "null_max": res["null_max"],
                                "p_value": res["p_value"], "B": res["B"],
                                "n_subsample": res["n_subsample"],
                                "model_class": res["model_class"],
                                "alpha_used": float(alpha_eff),
                                "K_selected_step_up": int(K_sel),
                            })
                            for b_idx, lr_b in enumerate(res["lr_null"]):
                                lrt_null_rows.append({"sex": sex, "K_null": int(K_n),
                                                      "K_alt": int(K_a), "b": int(b_idx),
                                                      "lr_b": float(lr_b)})
                    mode_payload["bootstrap_lrt"] = pls.DataFrame(lrt_rows) if lrt_rows else None
                    mode_payload["bootstrap_lrt_null"] = (pls.DataFrame(lrt_null_rows)
                                                          if lrt_null_rows else None)

            per_mode[interval_type] = mode_payload

        # Single archive write -- one HDF5 file containing both modes, every pool's intervals and
        # summary, the tied ladder, the session-corrected peak test and its null draws, and a
        # complete root-level provenance attribute set so a months-later reader has every
        # parameter that drove the run.
        analysis_attrs: dict = {
            "created_at_iso": run_started_at.isoformat(timespec="seconds"),
            "git_sha": git_sha_for_provenance(pathlib.Path(__file__).resolve().parent),
            "source_lists": [str(p) for p in session_lists],
            "n_sessions_loaded": int(len(sessions_with_data)),
            "exclude_noise_usvs": bool(exclude_noise_usvs),
            "interval_pools": interval_pools,
            "min_intervals_for_fitting": int(min_intervals_for_fitting),
            "fit_tied_model": bool(fit_tied_model),
            "tied_n_background": int(tied_n_background),
            "tied_peak_grid": tied_peak_grid,
            "tied_n_init": int(tied_n_init),
            "tied_n_init_boot": int(tied_n_init_boot),
            "tied_design_bootstrap": int(tied_design_bootstrap),
            "fit_mixture_model": bool(fit_mixture_model),
            "n_components_min": int(n_components_min),
            "n_components_max": int(n_components_max),
            "n_repeats": int(n_repeats),
            "max_modes_reported": int(max_modes_reported),
            "random_seed_base": int(random_seed_base),
            "cv_n_folds": int(cv_n_folds),
            "cv_n_init": int(cv_n_init),
            "mixture_model_n_init": int(mixture_model_n_init),
            "mixture_model_reg_covar": float(mixture_model_reg_covar),
            "tau": float(tau),
            "model_class": str(model_class),
            "bootstrap_lrt_B": int(bootstrap_lrt_B),
            "bootstrap_lrt_n_subsample": int(bootstrap_lrt_n_subsample),
            "bootstrap_lrt_alpha": float(bootstrap_lrt_alpha),
            "bootstrap_lrt_bonferroni": bool(bootstrap_lrt_bonferroni),
        }
        h5_path = out_dir / f"usv_interval_analysis_{run_ts}.h5"
        write_ivi_h5(h5_path, analysis_attrs=analysis_attrs, per_mode=per_mode)
        message(f"compute_inter_usv_interval_distributions: archive -> {h5_path}")

        message(
            f"compute_inter_usv_interval_distributions: finished at "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
