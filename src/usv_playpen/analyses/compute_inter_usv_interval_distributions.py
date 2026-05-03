"""
@author: bartulem
Computes inter-vocalization interval (inter-USV interval) distributions across one or
more lists of session root directories, and (optionally) sweeps a 1D
Gaussian Mixture Model over a range of component counts on the pooled
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
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import numpy as np
import polars as pls

from ..os_utils import configure_path
from ..visualizations.usv_summary_statistics import (
    extract_session_metadata,
    load_and_filter_usv_data,
)
from .gmm_utils import (
    bootstrap_lrt,
    fit_log_gmm,
    fit_log_t_mixture,
    gmm_boundaries_logspace,
    gmm_cv_neg_loglik,
    gmm_icl,
    report_gmm_stats,
    report_t_mixture_stats,
    select_n_components_step_up_lrt,
    t_mixture_cv_neg_loglik,
    t_mixture_icl,
)
from .usv_interval_archive import git_sha_for_provenance, write_ivi_h5


def _read_session_lists(session_lists: list[str], message_output) -> list[str]:
    """
    Description
    Reads one or more session-list text files (one session root per
    line, blank lines ignored) and returns the de-duplicated union of
    sessions, preserving first-seen order. Each path is run through
    ``configure_path`` so Mac/Linux/Windows paths in the input file all
    resolve correctly on the host platform.

    Parameters
    session_lists (list[str])
        List of text file paths, each containing session roots.
    message_output (callable)
        Logging callable (typically ``print``).

    Returns
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
    Builds a session-root -> source-file-name mapping so the master
    DataFrame can carry a ``source_list`` column for multi-cohort
    comparisons. The first list a session appears in wins (mirroring
    the de-duplication policy in :func:`_read_session_lists`).

    Parameters
    session_lists (list[str])
        List of text file paths.

    Returns
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
    noise_col_id: str,
    noise_categories: list[int],
) -> dict:
    """
    Description
    Computes inter-vocalization intervals for one session under one
    interval-definition mode.

    For each consecutive pair of USVs in the session's noise-filtered
    summary CSV, an interval is recorded only if both vocalizations
    are emitted by the same animal. The "current pointer" advances on
    every row regardless of whether an interval was recorded — this is
    intentional: it preserves chronological order of the conversation
    so a male->female->male triplet does not record a male-male
    interval that skips over the female call. Intervals strictly
    greater than zero are kept; non-positive intervals (only possible
    in ``e2s`` mode for overlapping calls) are counted and reported.

    Parameters
    session_root (str)
        Absolute path to the session directory.
    interval_type (str)
        Either ``'s2s'`` (start-to-start) or ``'e2s'`` (end-to-start).
    noise_col_id (str)
        Name of the noise classification column in the USV summary CSV.
    noise_categories (list[int])
        Integer labels in ``noise_col_id`` that identify a row as
        noise to be excluded.

    Returns
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

    try:
        metadata = extract_session_metadata(session_root)
    except (FileNotFoundError, IndexError):
        return {}

    raw_male_id = metadata['male_id']
    raw_female_id = metadata['female_id']
    male_id = str(raw_male_id).strip('\x00').strip()
    female_id = str(raw_female_id).strip('\x00').strip()

    try:
        usv_info = load_and_filter_usv_data(
            session_root=session_root,
            frame_rate=metadata['frame_rate'],
            noise_col_id=noise_col_id,
            noise_categories=noise_categories,
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
    if "stop" in usv_info.columns:
        sub = usv_info.with_columns(sex_expr).select(["start", "stop", "sex"])
    else:
        sub = usv_info.with_columns([
            (pls.col("start") + pls.col("duration")).alias("stop"),
            sex_expr,
        ]).select(["start", "stop", "sex"])

    # extract to numpy arrays for the streaming pointer iteration; this avoids
    # per-row Polars overhead and keeps the same-emitter gating readable
    start_arr = sub["start"].to_numpy()
    stop_arr = sub["stop"].to_numpy()
    sex_arr = sub["sex"].to_numpy()

    col_for_tag = {"start": start_arr, "stop": stop_arr}
    usv0_col = col_for_tag[usv0_tag]
    usv1_col = col_for_tag[usv1_tag]

    intervals = {"male": [], "female": []}
    n_dropped = {"male": 0, "female": 0}

    usv0_time = usv0_col[0]
    usv0_sex = sex_arr[0]

    for r in range(1, len(sex_arr)):
        usv1_time = usv1_col[r]
        usv1_sex = sex_arr[r]

        # same identified emitter (skip unassigned-unassigned pairs)
        if (usv0_sex == usv1_sex) and (usv0_sex in ("male", "female")):
            interval = usv1_time - usv0_time
            if interval > 0:
                intervals[usv0_sex].append(interval)
            else:
                n_dropped[usv0_sex] += 1

        usv0_time = usv0_col[r]
        usv0_sex = usv1_sex

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


def fit_gmm_sweep(
    intervals_by_key: dict[str, np.ndarray],
    n_components_min: int,
    n_components_max: int,
    n_repeats: int,
    max_modes_reported: int,
    random_seed_base: int,
    tau: float = 0.5,
    cv_n_folds: int = 5,
    cv_n_init: int = 5,
    gmm_n_init: int = 10,
    gmm_reg_covar: float = 1e-4,
    model_class: str = "gauss",
) -> pls.DataFrame:
    """
    Description
    Sweeps GMMs of size ``n_components_min`` through ``n_components_max``
    on each pooled inter-USV interval array (typically
    ``{'male': ..., 'female': ...}``), repeating each fit ``n_repeats``
    times under different seeds. Selection across reps and across K is
    delegated to the bootstrap-LRT step-up procedure in
    :func:`gmm_utils.bootstrap_lrt` /
    :func:`gmm_utils.select_n_components_step_up_lrt`; the IC columns
    in the returned table (``bic`` / ``aic`` / ``icl`` /
    ``cv_neg_loglik``) are diagnostic only.

    The returned tidy DataFrame is the model-parameter store consumed
    by the HDF5 archive layer: per-component log-mean / log-std /
    weight / nu (NaN for Gaussian) are recorded per row, plus
    inter-component decision boundaries in log-space and seconds at
    posterior threshold ``tau`` (Gaussian model class only; NaN-padded
    for the Student-t path, where boundaries lack a closed form).

    Parameters
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
    gmm_n_init (int)
        EM restarts for each in-sample fit.
    gmm_reg_covar (float)
        Variance floor passed to the EM solver.
    model_class (str)
        ``'gauss'`` (sklearn ``GaussianMixture``) or ``'t'``
        (:class:`gmm_utils.TMixture`).

    Returns
    df_results (pls.DataFrame)
        One row per ``(key, n_comp, rep)`` with: ``bic``, ``aic``,
        ``icl``, ``cv_neg_loglik`` (constant across reps for a given
        K), per-component ``logmean_k`` / ``logsd_k`` /
        ``median_sec_k`` / ``weight_k`` / ``nu_k`` (NaN-padded to
        ``n_components_max`` and NaN for ``nu_k`` under
        ``gauss``), the top ``max_modes_reported`` mixture modes
        (``mode_sec_k`` / ``density_k``), and Gaussian-only
        adjacent-component boundaries (``boundary_log_k`` /
        ``boundary_sec_k``).
    """

    if model_class not in ("gauss", "t"):
        msg = f"fit_gmm_sweep: model_class must be 'gauss' or 't', got {model_class!r}."
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
                    reg_covar=gmm_reg_covar,
                )
            else:  # t-mixture
                cv_val = t_mixture_cv_neg_loglik(
                    intervals_sec=iui,
                    n_components=n_components,
                    seed=random_seed_base,
                    n_folds=cv_n_folds,
                    n_init=max(1, cv_n_init - 2),  # t-mix EM is heavier; trim per-fold inits
                    reg_covar=gmm_reg_covar,
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
            for n_components in n_comps:
                seed = random_seed_base + r
                if model_class == "gauss":
                    model, model_order = fit_log_gmm(
                        iui, n_components=n_components, seed=seed,
                        n_init=gmm_n_init, reg_covar=gmm_reg_covar,
                    )
                    logmeans, logsds, modes_log, densities = report_gmm_stats(model, model_order)
                    weights = model.weights_.flatten()[model_order]
                    nus = np.full(n_components, np.nan, dtype=float)  # not applicable for gauss
                    icl = gmm_icl(model, log_iui)
                else:  # t-mixture
                    model, model_order = fit_log_t_mixture(
                        iui, n_components=n_components, seed=seed,
                        n_init=gmm_n_init, reg_covar=gmm_reg_covar,
                    )
                    logmeans, logsds, nus, weights, mode_dens = report_t_mixture_stats(model, model_order)
                    # For the rep-row schema, "modes" are the per-component
                    # means in 1D log-space; their densities are the
                    # mixture density at each mean.
                    modes_log = logmeans.copy()
                    densities = mode_dens
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
                for k in range(n_components, n_components_max):
                    results[f"logmean_{k+1}"].append(np.nan)
                    results[f"logsd_{k+1}"].append(np.nan)
                    results[f"median_sec_{k+1}"].append(np.nan)
                    results[f"weight_{k+1}"].append(np.nan)
                    results[f"nu_{k+1}"].append(np.nan)

                # mixture modes (top-N by density, sorted ascending in seconds)
                modes_sec = np.exp(modes_log) if modes_log.size else modes_log
                n_modes = len(modes_sec)
                for k in range(min(max_modes_reported, n_modes)):
                    results[f"mode_sec_{k+1}"].append(float(modes_sec[k]))
                    results[f"density_{k+1}"].append(float(densities[k]))
                for k in range(n_modes, max_modes_reported):
                    results[f"mode_sec_{k+1}"].append(np.nan)
                    results[f"density_{k+1}"].append(np.nan)

                # adjacent-component boundaries (Gaussian-only; NaN-fill for t-mix)
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


class InterUSVIntervalCalculator:
    """
    Cross-session inter-vocalization-interval driver. Reads one or more
    session-list text files, computes per-session inter-USV intervals in each
    interval-definition mode (``s2s`` and ``e2s``), runs the optional
    GMM / t-mixture sweep and bootstrap LRT, and consolidates the
    whole run into a single ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5``
    archive (see :mod:`usv_playpen.analyses.usv_interval_archive`).
    """

    def __init__(self, **kwargs):
        """
        Description
        Initialises the InterUSVIntervalCalculator. All keyword arguments are
        captured into ``self.__dict__`` verbatim so the instance
        exposes every supplied kwarg as an attribute (no whitelisting),
        matching the convention used by :class:`FeatureZoo`.

        Parameters
        input_parameter_dict (dict)
            Full ``analyses_settings`` dictionary; the
            ``compute_inter_usv_interval_distributions`` block is read from it.
        message_output (callable)
            Logging callable (typically ``print``).

        Returns
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def save_inter_usv_interval_distributions_to_file(self) -> None:
        """
        Description
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
        * ``/<mode>/gmm_fits`` (only when ``fit_gmm`` is true) -- the
          full GMM / t-mixture sweep with all four ICs (``bic``,
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

        Returns
        """

        cfg = self.input_parameter_dict['compute_inter_usv_interval_distributions']

        session_lists = cfg['session_lists']
        output_directory = cfg['output_directory']
        # Both interval definitions are computed unconditionally; the cost
        # is dominated by the per-session USV CSV pass which is shared.
        interval_types = ("s2s", "e2s")
        noise_col_id = cfg['noise_col_id']
        noise_categories = cfg['noise_categories']
        fit_gmm = cfg['fit_gmm']
        n_components_min = cfg['n_components_min']
        n_components_max = cfg['n_components_max']
        n_repeats = cfg['n_repeats']
        max_modes_reported = cfg['max_modes_reported']
        random_seed_base = cfg['random_seed_base']
        cv_n_folds = cfg['cv_n_folds']
        cv_n_init = cfg['cv_n_init']
        gmm_n_init = cfg['gmm_n_init']
        gmm_reg_covar = cfg['gmm_reg_covar']
        tau = cfg['tau']
        model_class = cfg['model_class']
        bootstrap_lrt_B = cfg['bootstrap_lrt_B']
        bootstrap_lrt_n_subsample = cfg['bootstrap_lrt_n_subsample']
        bootstrap_lrt_alpha = cfg['bootstrap_lrt_alpha']
        bootstrap_lrt_bonferroni = cfg['bootstrap_lrt_bonferroni']

        if model_class not in ('gauss', 't'):
            msg = f"compute_inter_usv_interval_distributions: model_class must be 'gauss' or 't', got {model_class!r}."
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

        for interval_type in interval_types:
            pooled = {"male": [], "female": []}
            n_dropped_total = {"male": 0, "female": 0}
            tidy_rows: list[dict] = []

            for session_root in sessions:
                usv_interval = compute_session_usv_intervals(
                    session_root=session_root,
                    interval_type=interval_type,
                    noise_col_id=noise_col_id,
                    noise_categories=noise_categories,
                )
                if not usv_interval:
                    continue

                session_id = pathlib.Path(session_root).name
                source_list = source_map.get(session_root, "")

                sessions_with_data.add(session_root)
                pooled["male"].append(usv_interval["male"])
                pooled["female"].append(usv_interval["female"])
                n_dropped_total["male"] += usv_interval["n_dropped_male"]
                n_dropped_total["female"] += usv_interval["n_dropped_female"]

                for sex_label, arr in (("male", usv_interval["male"]), ("female", usv_interval["female"])):
                    for v in arr:
                        tidy_rows.append({
                            "session_id": session_id,
                            "source_list": source_list,
                            "interval_type": interval_type,
                            "sex": sex_label,
                            "interval_s": float(v),
                            "log_interval": float(np.log(v)),
                            "male_id": usv_interval["male_id"],
                            "female_id": usv_interval["female_id"],
                        })

            male_pool = np.concatenate(pooled["male"]) if pooled["male"] else np.array([], dtype=float)
            female_pool = np.concatenate(pooled["female"]) if pooled["female"] else np.array([], dtype=float)

            # Explicit tidy schema so empty modes still archive a frame
            # the loader can introspect (matches build_master_usv_interval_dataframe).
            usv_interval_schema = {
                "session_id": pls.Utf8,
                "source_list": pls.Utf8,
                "interval_type": pls.Utf8,
                "sex": pls.Utf8,
                "interval_s": pls.Float64,
                "log_interval": pls.Float64,
                "male_id": pls.Utf8,
                "female_id": pls.Utf8,
            }
            tidy_df = (
                pls.from_dicts(tidy_rows, schema=usv_interval_schema)
                if tidy_rows else pls.DataFrame(schema=usv_interval_schema)
            )
            drop_df = pls.DataFrame([
                {"sex": "male",   "n_dropped": int(n_dropped_total["male"])},
                {"sex": "female", "n_dropped": int(n_dropped_total["female"])},
            ])
            message(
                f"  [{interval_type}] male={male_pool.size} female={female_pool.size} "
                f"(dropped non-positive: male={n_dropped_total['male']}, "
                f"female={n_dropped_total['female']})"
            )

            mode_payload: dict = {
                "attrs": {},
                "intervals": tidy_df,
                "drop_counts": drop_df,
                "gmm_fits": None,
                "bootstrap_lrt": None,
                "bootstrap_lrt_null": None,
            }

            if fit_gmm and (male_pool.size >= 2 or female_pool.size >= 2):
                df_results = fit_gmm_sweep(
                    intervals_by_key={"male": male_pool, "female": female_pool},
                    n_components_min=n_components_min,
                    n_components_max=n_components_max,
                    n_repeats=n_repeats,
                    max_modes_reported=max_modes_reported,
                    random_seed_base=random_seed_base,
                    tau=tau,
                    cv_n_folds=cv_n_folds,
                    cv_n_init=cv_n_init,
                    gmm_n_init=gmm_n_init,
                    gmm_reg_covar=gmm_reg_covar,
                    model_class=model_class,
                )
                mode_payload["gmm_fits"] = df_results
                message(f"  [{interval_type}] GMM sweep ({df_results.height} rows) recorded.")

                # Parametric bootstrap LRT for K-selection (McLachlan 1987;
                # McLachlan & Peel 2000 Ch. 6). Step-up rule: stop at the
                # first (K, K+1) test that fails to reject H0.
                lrt_pairs = list(range(n_components_min, n_components_max))
                if lrt_pairs:
                    message(
                        f"  [{interval_type}] running bootstrap LRT "
                        f"(B={bootstrap_lrt_B}, N_sub={bootstrap_lrt_n_subsample}, "
                        f"model={model_class})..."
                    )
                    lrt_rows: list[dict] = []
                    lrt_null_rows: list[dict] = []  # long-form lr_null for replotting
                    selected_per_sex: dict[str, int] = {}
                    alpha_eff_for_attr: float = float(bootstrap_lrt_alpha)
                    for sex, pool in (("male", male_pool), ("female", female_pool)):
                        if pool.size < 2:
                            continue
                        pair_results: dict = {}
                        for K_n in lrt_pairs:
                            K_a = K_n + 1
                            res = bootstrap_lrt(
                                intervals_sec=pool,
                                K_null=K_n,
                                K_alt=K_a,
                                B=bootstrap_lrt_B,
                                n_subsample=bootstrap_lrt_n_subsample,
                                model_class=model_class,
                                n_init_obs=gmm_n_init,
                                n_init_boot=max(1, gmm_n_init - 7),
                                reg_covar=gmm_reg_covar,
                                seed=random_seed_base,
                            )
                            pair_results[(K_n, K_a)] = res
                            message(
                                f"    [{sex}] K={K_n} vs K={K_a}: "
                                f"LR_obs={res['lr_obs']:.2f}, "
                                f"null_95%={res['null_p95']:.2f}, "
                                f"p={res['p_value']:.3f}"
                            )

                        n_tests = len(pair_results)
                        alpha_eff = (
                            bootstrap_lrt_alpha / n_tests
                            if (bootstrap_lrt_bonferroni and n_tests > 0)
                            else bootstrap_lrt_alpha
                        )
                        alpha_eff_for_attr = float(alpha_eff)
                        K_sel = select_n_components_step_up_lrt(pair_results, alpha=alpha_eff)
                        selected_per_sex[sex] = int(K_sel)
                        message(
                            f"  [{interval_type}] [{sex}] step-up LRT selection "
                            f"(alpha_eff={alpha_eff:.4g}): K = {K_sel}"
                        )

                        for (K_n, K_a), res in pair_results.items():
                            lrt_rows.append({
                                "sex": sex,
                                "K_null": res["K_null"],
                                "K_alt": res["K_alt"],
                                "lr_obs": res["lr_obs"],
                                "null_mean": res["null_mean"],
                                "null_p95": res["null_p95"],
                                "null_max": res["null_max"],
                                "p_value": res["p_value"],
                                "B": res["B"],
                                "n_subsample": res["n_subsample"],
                                "model_class": res["model_class"],
                                "alpha_used": float(alpha_eff),
                                "K_selected_step_up": int(K_sel),
                            })
                            for b_idx, lr_b in enumerate(res["lr_null"]):
                                lrt_null_rows.append({
                                    "sex": sex,
                                    "K_null": int(K_n),
                                    "K_alt": int(K_a),
                                    "b": int(b_idx),
                                    "lr_b": float(lr_b),
                                })

                    if lrt_rows:
                        mode_payload["bootstrap_lrt"] = pls.DataFrame(lrt_rows)
                    if lrt_null_rows:
                        mode_payload["bootstrap_lrt_null"] = pls.DataFrame(lrt_null_rows)
                    mode_payload["attrs"]["alpha_effective"] = alpha_eff_for_attr
                    mode_payload["attrs"]["K_selected_male"] = selected_per_sex.get("male", -1)
                    mode_payload["attrs"]["K_selected_female"] = selected_per_sex.get("female", -1)

            per_mode[interval_type] = mode_payload

        # Single archive write -- one HDF5 file containing both modes,
        # all intervals, full sweep, LRT summary + null draws, and a
        # complete root-level provenance attribute set so a months-later
        # reader has every parameter that drove the run.
        analysis_attrs: dict = {
            "created_at_iso": run_started_at.isoformat(timespec="seconds"),
            "git_sha": git_sha_for_provenance(pathlib.Path(__file__).resolve().parent),
            "source_lists": [str(p) for p in session_lists],
            "n_sessions_loaded": int(len(sessions_with_data)),
            "noise_col_id": noise_col_id,
            "noise_categories": list(noise_categories),
            "fit_gmm": bool(fit_gmm),
            "n_components_min": int(n_components_min),
            "n_components_max": int(n_components_max),
            "n_repeats": int(n_repeats),
            "max_modes_reported": int(max_modes_reported),
            "random_seed_base": int(random_seed_base),
            "cv_n_folds": int(cv_n_folds),
            "cv_n_init": int(cv_n_init),
            "gmm_n_init": int(gmm_n_init),
            "gmm_reg_covar": float(gmm_reg_covar),
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
