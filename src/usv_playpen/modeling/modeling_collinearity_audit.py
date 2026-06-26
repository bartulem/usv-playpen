"""
@author: bartulem
Predictor diagnostics computed at modeling-input-pickle creation time.

This module exposes two free-standing audits that every modeling pipeline
(`vocal_onsets`, `vocal_categories_binomial`, `vocal_categories_multinomial`,
`vocal_bout_parameters`, `usv_manifold_position`) calls just before slicing
the per-event history matrix:

1. `audit_predictor_collinearity` — answers "are any of the kept predictors
   redundant with each other?"  Computes pairwise Spearman / Pearson ρ on
   the per-event-onset history-window mean of every generic feature column,
   plus per-feature Variance Inflation Factors and the design-matrix
   condition number.

2. `audit_predictor_timescales` — characterises how each predictor
   relates to the binary USV train, in time. Reports two complementary
   summaries:
     * Predictor autocorrelation (ACF) → how long each feature holds
       memory of itself; a structural property of the predictor.
     * Signal correlation → temporal cross-correlation between every
       predictor and the per-frame binary bout-onset trace, at lags
       spanning `[-max_lag_seconds, +max_lag_seconds]`. Negative lags
       mean the bout precedes the feature; positive lags mean the
       feature precedes the bout. A within-session circular-shift
       null is computed alongside as a per-lag noise floor.
   The response-side IBI thresholds and empirical IBI percentiles are
   bundled in for completeness.

Both audits write a side-by-side `.pkl` artifact (never modifies the
modeling pickle itself) and emit a stdout summary so cluster logs surface
the headline numbers without the user needing to re-load anything.

The audits are diagnostic — they do not feed back into the model fit. A
failure inside either function should warn-and-continue rather than abort
the extraction pipeline; the wrapper `run_predictor_audits` in
`modeling_utils` enforces that policy.
"""

import pickle
import time
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr, rankdata

def _build_event_summary_matrix(processed_beh_dict: dict,
                                event_times_per_session: dict,
                                mouse_names_dict: dict,
                                target_idx: int,
                                predictor_idx: int,
                                history_frames: int,
                                camera_fps_dict: dict) -> tuple:
    """
    Builds the (n_events_total, n_features) per-event-onset summary matrix
    used as the input to the collinearity audit.

    For every kept event onset across every session, the value of each
    generic feature column (renamed to `self.*` / `other.*` / `dyad-*`) is
    averaged over its `history_frames`-frame pre-event window. Events whose
    full window does not fit inside the recording (`s < 0` or
    `e > session_duration`) are skipped, mirroring the per-event slicing
    policy used by the modeling pipelines.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping `session_id -> polars.DataFrame` of z-scored per-session
        feature traces.
    event_times_per_session : dict
        Mapping `session_id -> np.ndarray` of event onset times (in
        seconds). The audit pools rows from every session.
    mouse_names_dict : dict
        Mapping `session_id -> list[mouse_name]`. Used to translate
        `{mouse}.{feat}` columns into generic `self.*` / `other.*` keys.
    target_idx, predictor_idx : int
        The target / predictor mouse slot indices.
    history_frames : int
        Number of frames in the pre-event history window.
    camera_fps_dict : dict
        Mapping `session_id -> camera_sampling_rate_hz`. Used to convert
        event times (seconds) into frame indices.

    Returns
    -------
    tuple
        `(generic_feature_names, summary_matrix)` where
        `generic_feature_names` is the sorted list of generic feature
        identifiers (e.g. `'self.speed'`, `'other.usv_rate'`,
        `'nose-nose'`) and `summary_matrix` is a `(n_events_total,
        n_features)` `float32` array of per-event window means. Sessions
        with no in-bounds events contribute zero rows.
    """

    # Two-pass build. First pass: walk every session, decide which
    # features it has, and stash the per-session window-mean blocks per
    # feature. Track each contributing session and its valid-event count
    # so we can later restrict the output to features that contributed
    # *for every contributing session* — the column_stack at the end
    # requires equal per-feature row totals, and that only holds if a
    # feature was present in every session whose events landed in the
    # pool. (The legacy single-pass build silently dropped sessions
    # that lacked some features and produced a ragged dict that
    # crashed `column_stack`; see the per-feature row-count
    # mismatch reported on dense-USV cohorts.)
    per_feature_session_blocks = {}   # generic_key -> {sess_id: ndarray}
    contributing_sessions = []        # list[(sess_id, n_valid_events)]

    for sess_id, sess_df in processed_beh_dict.items():
        if sess_id not in event_times_per_session:
            continue
        ev_times = event_times_per_session[sess_id]
        if ev_times is None or len(ev_times) == 0:
            continue

        if sess_id not in mouse_names_dict:
            continue
        t_name = mouse_names_dict[sess_id][target_idx]
        p_name = mouse_names_dict[sess_id][predictor_idx]

        fps = camera_fps_dict[sess_id]
        n_frames = sess_df.height

        ends = np.round(np.asarray(ev_times) * fps).astype(int)
        starts = ends - history_frames
        valid_mask = (starts >= 0) & (ends <= n_frames)
        starts = starts[valid_mask]
        ends = ends[valid_mask]

        if starts.size == 0:
            continue

        contributing_sessions.append((sess_id, int(starts.size)))

        for col_name in sess_df.columns:
            suffix = col_name.split('.')[-1]
            if suffix.isdigit():
                continue

            if col_name.startswith(f"{t_name}."):
                generic_key = f"self.{suffix}"
            elif col_name.startswith(f"{p_name}."):
                generic_key = f"other.{suffix}"
            else:
                generic_key = col_name

            col_values = sess_df[col_name].to_numpy()
            window_means = np.empty(starts.size, dtype=np.float32)
            for i, (s, e) in enumerate(zip(starts, ends)):
                chunk = col_values[s:e]
                if np.isnan(chunk).any():
                    chunk = np.nan_to_num(chunk, nan=0.0)
                window_means[i] = float(chunk.mean())

            per_feature_session_blocks.setdefault(generic_key, {})[sess_id] = window_means

    if not per_feature_session_blocks or not contributing_sessions:
        return [], np.empty((0, 0), dtype=np.float32)

    # Second pass: keep only features that contributed a block for
    # *every* contributing session. Drop the rest with a single
    # consolidated warning so the operator can see which features were
    # removed (typically rare-category vocal channels that only fire
    # in some sessions).
    contributing_session_ids = {sid for sid, _ in contributing_sessions}
    kept_features = []
    dropped_features = []
    for f, sess_to_block in per_feature_session_blocks.items():
        if contributing_session_ids.issubset(sess_to_block.keys()):
            kept_features.append(f)
        else:
            dropped_features.append(f)

    if dropped_features:
        n_drop = len(dropped_features)
        sample = ', '.join(sorted(dropped_features)[:6])
        more = f" (+ {n_drop - 6} more)" if n_drop > 6 else ''
        print(f"[audit] collinearity: dropped {n_drop} feature(s) absent from "
              f"some sessions: {sample}{more}.")

    if not kept_features:
        return [], np.empty((0, 0), dtype=np.float32)

    feature_names = sorted(kept_features)
    # Concatenate each feature's per-session blocks in the same order
    # so column_stack is rectangular by construction.
    ordered_session_ids = [sid for sid, _ in contributing_sessions]
    aligned_columns = [
        np.concatenate(
            [per_feature_session_blocks[f][sid] for sid in ordered_session_ids],
            axis=0,
        )
        for f in feature_names
    ]
    summary_matrix = np.column_stack(aligned_columns).astype(np.float32)
    return feature_names, summary_matrix


def _vif_from_design(X: np.ndarray) -> np.ndarray:
    """
    Computes the per-column Variance Inflation Factor of the design matrix.

    For each column `j`, regresses `X[:, j]` on the remaining columns via
    ordinary least squares (closed-form pseudoinverse) and reports
    `VIF_j = 1 / (1 - R²_j)`. `VIF = 1` means perfect orthogonality;
    `VIF > 5` is a concern; `VIF > 10` is a serious collinearity problem
    (Belsley/Kuh/Welsch convention). Constant columns and columns whose
    OLS fit is degenerate are returned as `inf` so they surface clearly in
    the summary table.

    Inlined here so the module does not pull `statsmodels` as a hard
    dependency just for this single computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape `(n_samples, n_features)`. Rows with any
        NaN are dropped before fitting; constant columns are reported as
        `inf` rather than fitted.

    Returns
    -------
    np.ndarray
        Per-feature VIF vector of shape `(n_features,)`.
    """

    X = np.asarray(X, dtype=np.float64)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    n_samples, n_features = X.shape
    vif = np.full(n_features, np.nan, dtype=np.float64)

    if n_samples <= n_features + 1:
        # Under-determined system — VIF is undefined.
        return vif

    # Constant columns: VIF is infinite by definition.
    col_std = X.std(axis=0)
    constant_cols = col_std == 0

    for j in range(n_features):
        if constant_cols[j]:
            vif[j] = float('inf')
            continue
        y = X[:, j]
        keep_cols = np.ones(n_features, dtype=bool)
        keep_cols[j] = False
        Xj = X[:, keep_cols]
        # Add an intercept column so VIF matches the standard textbook
        # definition (regress feature on the rest with intercept).
        Xj_aug = np.column_stack([np.ones(n_samples), Xj])
        try:
            beta, *_ = np.linalg.lstsq(Xj_aug, y, rcond=None)
            y_pred = Xj_aug @ beta
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
            r2 = min(r2, 1.0 - 1e-12)
            vif[j] = 1.0 / (1.0 - r2)
        except np.linalg.LinAlgError:
            vif[j] = float('inf')

    return vif


def _flagged_pairs(rho: np.ndarray,
                   names: list,
                   concern_thresh: float = 0.7,
                   exclude_thresh: float = 0.85) -> list:
    """
    Returns the list of off-diagonal pairs whose absolute correlation
    exceeds `concern_thresh`, sorted by descending magnitude.

    Each entry is a `(name_i, name_j, rho_value, tier)` tuple where
    `tier` is the string `'exclude'` if `|rho| > exclude_thresh` else
    `'concern'`. Used to populate the stdout summary and the artifact's
    `flagged_pairs` field.

    Parameters
    ----------
    rho : np.ndarray
        Square correlation matrix of shape `(n_features, n_features)`.
    names : list of str
        Feature labels matching the row/column ordering of `rho`.
    concern_thresh : float, default 0.7
        Absolute-correlation threshold above which a pair is flagged as a
        concern (selection-stability worry).
    exclude_thresh : float, default 0.85
        Absolute-correlation threshold above which the pair is treated
        as effectively redundant.

    Returns
    -------
    list of tuple
        Sorted list of `(name_i, name_j, rho_value, tier)` tuples.
    """

    n = rho.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = rho[i, j]
            if not np.isfinite(r):
                continue
            mag = abs(r)
            if mag >= concern_thresh:
                tier = 'exclude' if mag >= exclude_thresh else 'concern'
                pairs.append((names[i], names[j], float(r), tier))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs

def audit_predictor_collinearity(processed_beh_dict: dict,
                                 event_times_per_session: dict,
                                 mouse_names_dict: dict,
                                 target_idx: int,
                                 predictor_idx: int,
                                 history_frames: int,
                                 camera_fps_dict: dict,
                                 save_path: str,
                                 source_pickle: str,
                                 concern_thresh: float = 0.7,
                                 exclude_thresh: float = 0.85,
                                 input_metadata: dict = None) -> dict:
    """
    Computes pairwise predictor correlations and per-feature VIFs on the
    per-event-onset summary matrix and persists the result to disk.

    The audit answers the question "are any of the kept predictors
    redundant with each other to a degree that would destabilise forward
    stepwise selection?"  It computes both Spearman ρ (primary; robust to
    monotonic non-linearities and not assuming Gaussian features) and
    Pearson ρ (secondary; matches the published 0.7 / 0.85 thresholds),
    Variance Inflation Factors per feature, and the design matrix
    condition number. Pairs above `concern_thresh` are flagged for review;
    pairs above `exclude_thresh` are treated as effectively redundant.

    Operating point
    ---------------
    The summary matrix is built at the per-event level (one row per event
    onset, one column per generic feature, value = mean of the feature's
    history window). Pooling at this granularity matches what the model
    actually sees per epoch, rather than including off-task continuous
    frames that would inflate or dilute correlations relative to the
    inferential target.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping `session_id -> polars.DataFrame` after z-scoring; same
        object the per-event slicer in each pipeline consumes.
    event_times_per_session : dict
        Mapping `session_id -> np.ndarray` of pooled event onset times
        (seconds). Pipelines that have multiple event classes (e.g.
        positive + negative) should pool them all here so the summary
        matrix reflects every row that will ever enter the model.
    mouse_names_dict : dict
        Mapping `session_id -> list[mouse_name]`.
    target_idx, predictor_idx : int
        Target / predictor mouse slot indices.
    history_frames : int
        Pre-event window length in frames.
    camera_fps_dict : dict
        Mapping `session_id -> camera_sampling_rate_hz`.
    save_path : str
        Absolute path to the artifact `.pkl`. The directory is created if
        missing.
    source_pickle : str
        Basename of the modeling input pickle this audit is paired with.
        Stored in the artifact for provenance.
    concern_thresh : float, default 0.7
        See `_flagged_pairs`.
    exclude_thresh : float, default 0.85
        See `_flagged_pairs`.
    input_metadata : dict, optional
        Pre-built `_input_metadata` block from the calling pipeline.
        When supplied, embedded under the reserved key
        `_input_metadata` of the saved payload so this artifact is
        independently provenance-complete.

    Returns
    -------
    dict
        The artifact payload (also written to `save_path`).
    """

    print("[audit] computing collinearity diagnostics...")
    feature_names, X = _build_event_summary_matrix(
        processed_beh_dict=processed_beh_dict,
        event_times_per_session=event_times_per_session,
        mouse_names_dict=mouse_names_dict,
        target_idx=target_idx,
        predictor_idx=predictor_idx,
        history_frames=history_frames,
        camera_fps_dict=camera_fps_dict,
    )

    n_events, n_features = X.shape
    if n_events == 0 or n_features == 0:
        print("[audit] collinearity: empty summary matrix — nothing to audit.")
        payload = {
            'features': feature_names,
            'spearman_rho': np.empty((0, 0)),
            'pearson_rho': np.empty((0, 0)),
            'vif': np.empty((0,)),
            'condition_number': float('nan'),
            'flagged_pairs': [],
            'concern_threshold': concern_thresh,
            'exclude_threshold': exclude_thresh,
            'n_events': 0,
            'source_pickle': source_pickle,
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        if input_metadata is not None:
            payload['_input_metadata'] = dict(input_metadata)
    else:
        # Drop zero-variance columns before the correlation / VIF /
        # condition-number computations. A constant column produces
        # division-by-zero in `np.corrcoef`'s
        # `c /= stddev[:, None]` step (RuntimeWarning) and a
        # `ConstantInputWarning` from `scipy.stats.spearmanr`, with
        # the offending row/column then becoming NaN — which
        # subsequently corrupts the condition number and VIF
        # computations. Surface the dropped feature names explicitly
        # and re-run the audit on the survivors only.
        col_var = X.var(axis=0, ddof=0)
        constant_mask = col_var == 0
        n_constant = int(constant_mask.sum())
        if n_constant > 0:
            const_names = [feature_names[i] for i in np.where(constant_mask)[0]]
            sample = ', '.join(const_names[:6])
            more = f" (+ {n_constant - 6} more)" if n_constant > 6 else ''
            print(f"[audit] collinearity: dropped {n_constant} zero-variance "
                  f"feature(s) before correlation/VIF: {sample}{more}.")
            keep_mask = ~constant_mask
            X = X[:, keep_mask]
            feature_names = [feature_names[i]
                             for i in range(len(feature_names))
                             if not constant_mask[i]]
            n_features = X.shape[1]

        if n_features == 0:
            print("[audit] collinearity: every feature is zero-variance — "
                  "nothing left to audit.")
            sp_full = np.empty((0, 0), dtype=np.float32)
            pe_full = np.empty((0, 0), dtype=np.float32)
            vif = np.empty((0,), dtype=np.float64)
            cond_num = float('nan')
            flagged = []
        else:
            # Spearman matrix on (n_events, n_features). scipy returns a square
            # matrix when given a 2-D array.
            sp_full = spearmanr(X, axis=0).correlation
            # `spearmanr` returns a 0-D scalar when n_features == 2; coerce.
            sp_full = np.atleast_2d(np.asarray(sp_full))
            if sp_full.shape != (n_features, n_features):
                sp_full = np.corrcoef(np.apply_along_axis(rankdata, 0, X), rowvar=False)

            pe_full = np.corrcoef(X, rowvar=False)

            vif = _vif_from_design(X)

            # Condition number on the column-standardized design — comparable
            # across runs regardless of feature scale.
            col_std = X.std(axis=0, ddof=1)
            col_std[col_std == 0] = 1.0
            X_std = (X - X.mean(axis=0)) / col_std
            try:
                cond_num = float(np.linalg.cond(X_std))
            except np.linalg.LinAlgError:
                cond_num = float('inf')

            flagged = _flagged_pairs(sp_full, feature_names,
                                     concern_thresh=concern_thresh,
                                     exclude_thresh=exclude_thresh)

        payload = {
            'features': feature_names,
            'spearman_rho': sp_full.astype(np.float32),
            'pearson_rho': pe_full.astype(np.float32),
            'vif': vif.astype(np.float64),
            'condition_number': cond_num,
            'flagged_pairs': flagged,
            'concern_threshold': concern_thresh,
            'exclude_threshold': exclude_thresh,
            'n_events': int(n_events),
            'source_pickle': source_pickle,
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        if input_metadata is not None:
            payload['_input_metadata'] = dict(input_metadata)

        # Stdout summary
        print("\n" + "=" * 72)
        print(f"COLLINEARITY AUDIT  ({n_events} events × {n_features} features)")
        print("=" * 72)
        print(f"  Condition number (z-scored design): {cond_num:.2f}")

        # Top-10 VIFs
        sorted_idx = np.argsort(-np.where(np.isfinite(vif), vif, -np.inf))
        print(f"\n  Top {min(10, n_features)} VIFs:")
        for k in sorted_idx[:10]:
            v = vif[k]
            tag = '  '
            if np.isfinite(v):
                if v > 10:
                    tag = '!!'
                elif v > 5:
                    tag = '! '
            print(f"    {tag} {feature_names[k]:<35s}  VIF = {v:8.2f}")

        # Flagged pairs
        n_concern = sum(1 for *_, t in flagged if t == 'concern')
        n_exclude = sum(1 for *_, t in flagged if t == 'exclude')
        print(f"\n  Flagged pairs: {n_exclude} exclude (|ρ|>{exclude_thresh}), "
              f"{n_concern} concern ({concern_thresh}<|ρ|≤{exclude_thresh})")
        for f1, f2, r, tier in flagged[:15]:
            tag = '!!' if tier == 'exclude' else '! '
            print(f"    {tag} {f1:<30s}  <->  {f2:<30s}  ρ = {r:+.3f}")
        if len(flagged) > 15:
            print(f"    ... and {len(flagged) - 15} more (see artifact).")
        print("=" * 72 + "\n")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(save_path).open('wb') as fh:
        pickle.dump(payload, fh)
    print(f"[audit] collinearity artifact written: {save_path}")

    return payload

def _per_session_acf(trace: np.ndarray, max_lag_frames: int) -> np.ndarray:
    """
    Computes the biased sample ACF of a 1-D time series up to
    `max_lag_frames` lags using the FFT-based estimator.

    The trace is mean-centered before computing the ACF. NaN values are
    replaced by zero so the FFT remains well-defined. The returned array
    has length `max_lag_frames + 1` (lag 0 through `max_lag_frames`) and
    is normalised so that `acf[0] == 1`.

    Parameters
    ----------
    trace : np.ndarray
        1-D time series.
    max_lag_frames : int
        Maximum lag to return (in frames).

    Returns
    -------
    np.ndarray
        ACF values of shape `(max_lag_frames + 1,)`. `acf[0] == 1` for any
        non-constant input; for constant input, returns `nan`.
    """

    x = np.asarray(trace, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x = x - x.mean()
    n = x.size
    if n < 2 or x.std() == 0:
        return np.full(max_lag_frames + 1, np.nan)

    # FFT-based ACF: pad to next power of two >= 2*n to avoid circular wrap.
    n_pad = 1 << int(np.ceil(np.log2(2 * n)))
    fx = np.fft.rfft(x, n=n_pad)
    acf_full = np.fft.irfft(fx * np.conj(fx), n=n_pad)[:n]
    acf = acf_full / acf_full[0]
    if acf.size < max_lag_frames + 1:
        # Series shorter than requested window — pad with NaN.
        out = np.full(max_lag_frames + 1, np.nan)
        out[:acf.size] = acf
        return out
    return acf[:max_lag_frames + 1]


def _integrated_autocorr_time(acf: np.ndarray) -> float:
    """
    Computes Sokal's integrated autocorrelation time `τ_int = 1 + 2 Σ ρ(k)`
    summed up to the first non-positive crossing of the ACF.

    This is the "effective independent sample" timescale: a time series of
    length `N` carries roughly `N / τ_int` independent observations. More
    robust than threshold-crossing estimators (`τ_1/e`, `τ_0.2`) because
    it integrates the entire usable portion of the ACF rather than
    relying on a single point.

    Parameters
    ----------
    acf : np.ndarray
        Lag-indexed ACF starting at lag 0. Should have `acf[0] == 1` for
        a sensibly normalised input.

    Returns
    -------
    float
        `τ_int` in frames. Returns `nan` if `acf` is degenerate.
    """

    if not np.isfinite(acf).all() or acf.size < 2:
        return float('nan')
    # First lag at which ACF goes <= 0; sum positive lags only.
    sign_change = np.where(acf[1:] <= 0)[0]
    cutoff = sign_change[0] + 1 if sign_change.size else acf.size
    return float(1.0 + 2.0 * np.sum(acf[1:cutoff]))


def _first_crossing_below(acf: np.ndarray, threshold: float) -> float:
    """
    Returns the first lag (in frames) at which `acf` falls below
    `threshold`, or `nan` if the ACF stays above the threshold within
    the supplied window.

    Parameters
    ----------
    acf : np.ndarray
        Lag-indexed ACF starting at lag 0.
    threshold : float
        Crossing threshold (e.g. `1/e` or `0.2`).

    Returns
    -------
    float
        First lag below `threshold`, in frames, or `nan` if no crossing
        is observed.
    """

    if not np.isfinite(acf).all():
        return float('nan')
    below = np.where(acf < threshold)[0]
    return float(below[0]) if below.size else float('nan')


def _binary_event_trace(event_times: np.ndarray, n_frames: int, fps: float) -> np.ndarray:
    """
    Renders a sparse list of event timestamps into a binary occupancy
    trace at the camera frame rate.

    Parameters
    ----------
    event_times : np.ndarray
        Event onset times in seconds.
    n_frames : int
        Length of the output trace (in frames).
    fps : float
        Camera sampling rate.

    Returns
    -------
    np.ndarray
        `float32` binary trace of shape `(n_frames,)`. `1.0` at event
        onset frames, `0.0` elsewhere.
    """

    trace = np.zeros(n_frames, dtype=np.float32)
    if event_times is None or len(event_times) == 0:
        return trace
    idx = np.floor(np.asarray(event_times) * fps).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    trace[idx] = 1.0
    return trace


def audit_predictor_timescales(processed_beh_dict: dict,
                               mouse_names_dict: dict,
                               target_idx: int,
                               predictor_idx: int,
                               configured_filter_history: float,
                               camera_fps_dict: dict,
                               max_lag_seconds: float,
                               n_shuffles: int,
                               ibi_thresholds: dict,
                               save_path: str,
                               source_pickle: str,
                               random_seed: int = 0,
                               input_metadata: dict = None,
                               shuffle_range_seconds: tuple = (20.0, 60.0),
                               event_intervals_per_session: dict = None,
                               bout_onset_times_per_session: dict = None,
                               signal_floor_seconds: float = 0.5,
                               signal_min_run_seconds: float = 0.2) -> dict:
    """
    Computes per-feature ACF and signal correlation between every kept
    predictor and the per-frame binary model-event-onset trace,
    alongside the response-side IBI distribution, and persists the
    result to disk.

    Binary `Y` definition
    ---------------------
    `Y(t) = 1` at every frame index that is the start of a model
    event (single-frame impulse, sparse), and `Y(t) = 0` everywhere
    else. Model-event times come from `bout_onset_times_per_session`,
    whose granularity is set by the calling pipeline:

      - **Bout-level pipelines** — vocal_onsets in `bout` mode and
        bout-parameters — use bout starts (`usv_data_dict[sess][target]
        ['positive_events']` and `[...]['bout_onsets']` respectively),
        because each bout corresponds to one model prediction.
      - **Per-USV pipelines** — the binomial / multinomial Category
        pipelines and the continuous manifold pipeline — use per-USV
        starts (one impulse at every USV onset), because each
        individual USV corresponds to one model prediction. Within
        a bout that holds three USVs the bout-level `Y` would mark
        only the first; the per-USV `Y` marks all three, so the
        cross-correlation reflects the timing of features relative
        to every event the model actually sees.

    The kwarg names retain the historical "bout-onset" wording from
    when the audit was first written; treat them all as the
    **generalized model-event impulse source**, with the per-pipeline
    granularity documented above. Only `bout_onset_times_per_session`
    is a parameter of this function; `bout_onset_event_key` and
    `precomputed_bout_onset_times` are wrapper-side kwargs of
    `run_predictor_audits` (in `modeling_utils`) that ultimately
    resolve into the `bout_onset_times_per_session` dict passed here.
    The trace is built via `_binary_event_trace`, which marks the
    integer frame index of each onset (one `1.0` per event, no
    within-event duty cycle).

    The pooled `positive_events ∪ negative_events` definition was
    explicitly removed because it mixed bout-start frames with
    silence-sample frames into a single `1.0` marker, which is not
    a valid vocal indicator.

    Event-onset Y is sharper than is-vocalizing Y for the audit's
    primary question ("how far back does feature history inform an
    upcoming event?") because: (1) each event contributes one clean
    sample of `x` at lag `k` rather than a duration-blurred window,
    so lag-specific peaks survive; and (2) event onsets have narrow
    autocorrelation at the lag scales we care about (±10 s),
    keeping the circular-shift null tight.

    The ACF profile characterises each feature's own memory (how long
    its values stay correlated with themselves). The signal-correlation
    profile characterises how that feature aligns in time with `Y`, at
    lags from `-max_lag_seconds` to `+max_lag_seconds`. Lag-sign
    convention:

        ρ_signal(k) = corr( feature[t], Y[t + k] )

    so positive `k` ⇒ feature leads bout (behaviour precedes
    vocalisation); negative `k` ⇒ bout leads feature (vocalisation
    precedes behaviour).

    Signal-correlation implementation
    ---------------------------------
    For each (session × feature) pair the FFT cross-correlation curve
    is computed once at `n_pad_sess` lags. Two views of that single
    curve are then read off: the symmetric `[-L_max, +L_max]` window
    (the per-session actual ρ_session(k)) and `n_shuffles` per-session
    null windows centred at random shifts `S ∈ [shuffle_min,
    shuffle_max]` seconds.

    The per-session actuals are then averaged across sessions (mean +
    SEM): the plotted line is `mean_s ρ_session(k)` and the SEM band
    is `std_s ρ_session(k) / √n_sessions`.

    The null is reported on the *same* cohort-mean scale: shuffles
    are paired by index across sessions (valid because shifts within
    each session are i.i.d.), the cohort mean of `ρ_session(k)` is
    computed for each shuffle index, and the resulting `n_shuffles`
    cohort-mean curves are reduced to per-feature, per-lag mean and
    0.5 / 99.5 percentile envelopes. Width of the null band is
    therefore ~`σ_session/√n_sessions` — comparable to the SEM band
    on the actual line — rather than the much wider per-session
    spread the previous (`(session × shuffle)`-pooled) implementation
    reported. With `S` chosen well past the slowest feature's
    autocorrelation timescale, the sampled null lags lie in the tail
    where true ρ ≈ 0, giving a Bartlett-style honest null that
    preserves the autocorrelation structure of both `x` and `y`.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping `session_id -> polars.DataFrame` after z-scoring.
    mouse_names_dict : dict
        Mapping `session_id -> list[mouse_name]`.
    target_idx, predictor_idx : int
        Mouse slot indices.
    configured_filter_history : float
        The `filter_history` value (in seconds) currently configured in
        `modeling_settings.json`. Stored in the artifact for downstream
        plot annotations only; not used to gate any computation here.
    camera_fps_dict : dict
        Mapping `session_id -> camera_sampling_rate_hz`.
    max_lag_seconds : float
        Half-width of the signal-correlation lag axis (in seconds). The
        signal lag grid spans `[-max_lag_seconds, +max_lag_seconds]`;
        the ACF lag grid spans `[0, max_lag_seconds]`.
    n_shuffles : int
        Number of within-session circular-shift shuffles used to build
        the per-lag signal-correlation null.
    ibi_thresholds : dict
        Pre-computed `{'male': float, 'female': float}` IBI thresholds
        from `_calculate_ibi_threshold`. Stored in the artifact for
        downstream consumers.
    save_path : str
        Absolute path to the artifact `.pkl`. Directory is created if
        missing.
    source_pickle : str
        Basename of the paired modeling input pickle, for provenance.
    random_seed : int, default 0
        Seed for the within-session shift null.
    input_metadata : dict, optional
        Pre-built `_input_metadata` block from the calling pipeline.
        When supplied, embedded under the reserved key
        `_input_metadata` of the saved payload so this artifact is
        independently provenance-complete.
    shuffle_range_seconds : tuple, default (20.0, 60.0)
        `(shuffle_min_seconds, shuffle_max_seconds)` bounds for the
        within-session circular shifts that build both the ACF and the
        signal-correlation nulls. Both bounds are positive and chosen
        well past the slowest feature's autocorrelation timescale so
        the sampled null lags sit in the ACF tail where the true
        correlation has decayed (a Bartlett-style honest null).
    event_intervals_per_session : dict, optional but required at runtime
        Mapping `session_id -> (starts, stops)` of per-USV `[start,
        stop)` arrays (in seconds), used solely for the empirical
        IBI-percentile report block (inter-USV gaps `gap_i =
        start[i+1] - stop[i]`, directly comparable to the GMM-derived
        `ibi_threshold`). Not a source of `Y`. Passing `None` raises a
        `ValueError`; pipelines that expose bout onsets but no per-USV
        `[start, stop)` arrays should supply an empty dict, in which
        case the IBI percentiles come back NaN.
    bout_onset_times_per_session : dict, optional but required at runtime
        Mapping `session_id -> np.ndarray` of per-session model-event
        onset times (in seconds). This is the audit's sole source of
        the binary `Y(t)` impulse trace (see the "Binary `Y`
        definition" section above). Passing `None` raises a
        `ValueError`. Sessions absent from this dict are excluded from
        the signal-correlation computation.
    signal_floor_seconds : float, default 0.5
        Recorded into the payload for downstream plot / provenance use
        only; does not gate any computation in this function (it sets a
        marker threshold in `modeling_plots.py`).
    signal_min_run_seconds : float, default 0.2
        Recorded into the payload for downstream plot / provenance use
        only; does not gate any computation in this function (it sets a
        marker threshold in `modeling_plots.py`).

    Returns
    -------
    dict
        The artifact payload (also written to `save_path`).
    """

    print("[audit] computing timescale diagnostics...")

    # 1. Build per-session continuous matrices in generic naming. We
    #    re-use the same generic-key derivation as the collinearity audit.
    session_blocks = {}  # session_id -> {generic_key: ndarray}
    feature_set = set()

    for sess_id, sess_df in processed_beh_dict.items():
        if sess_id not in mouse_names_dict:
            continue
        t_name = mouse_names_dict[sess_id][target_idx]
        p_name = mouse_names_dict[sess_id][predictor_idx]

        per_feature = {}
        for col_name in sess_df.columns:
            suffix = col_name.split('.')[-1]
            if suffix.isdigit():
                continue
            if col_name.startswith(f"{t_name}."):
                generic_key = f"self.{suffix}"
            elif col_name.startswith(f"{p_name}."):
                generic_key = f"other.{suffix}"
            else:
                generic_key = col_name
            per_feature[generic_key] = sess_df[col_name].to_numpy().astype(np.float32)
            feature_set.add(generic_key)
        session_blocks[sess_id] = per_feature

    feature_names = sorted(feature_set)
    n_features = len(feature_names)

    if n_features == 0 or not session_blocks:
        print("[audit] timescales: empty input — nothing to audit.")
        payload = {
            'features': feature_names,
            'acf_lags_frames': np.empty((0,), dtype=np.int32),
            'acf_lags_seconds': np.empty((0,), dtype=np.float32),
            'acf_median': np.empty((0, 0), dtype=np.float32),
            'acf_p25': np.empty((0, 0), dtype=np.float32),
            'acf_p75': np.empty((0, 0), dtype=np.float32),
            'acf_null_mean': np.empty((0, 0), dtype=np.float32),
            'acf_null_p0_5': np.empty((0, 0), dtype=np.float32),
            'acf_null_p99_5': np.empty((0, 0), dtype=np.float32),
            'tau_acf_1_over_e': np.empty((0,), dtype=np.float32),
            'tau_acf_0_2': np.empty((0,), dtype=np.float32),
            'tau_acf_integrated': np.empty((0,), dtype=np.float32),
            'signal_lags_frames': np.empty((0,), dtype=np.int32),
            'signal_lags_seconds': np.empty((0,), dtype=np.float32),
            'rho_signal': np.empty((0, 0), dtype=np.float32),
            'rho_signal_per_session_mean': np.empty((0, 0), dtype=np.float32),
            'rho_signal_per_session_sem': np.empty((0, 0), dtype=np.float32),
            'rho_signal_null_mean': np.empty((0, 0), dtype=np.float32),
            'rho_signal_null_p0_5': np.empty((0, 0), dtype=np.float32),
            'rho_signal_null_p99_5': np.empty((0, 0), dtype=np.float32),
            'ibi_thresholds': dict(ibi_thresholds),
            'ibi_empirical_pcts': {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')},
            'configured_filter_history': float(configured_filter_history),
            'signal_floor_seconds': float(signal_floor_seconds),
            'signal_min_run_seconds': float(signal_min_run_seconds),
            'n_events': 0, 'n_bouts': 0, 'n_usvs': 0, 'n_sessions': 0,
            'source_pickle': source_pickle,
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        if input_metadata is not None:
            payload['_input_metadata'] = dict(input_metadata)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(save_path).open('wb') as fh:
            pickle.dump(payload, fh)
        return payload

    # All sessions are recorded at the same fps in this project; pull the
    # canonical value and warn loudly if heterogeneity ever appears. The
    # reported `lags_seconds` axis uses a single fps; `_per_session_acf`
    # still uses each session's own fps internally, but a mixed-fps run
    # would render the artifact's `lags_seconds` axis ambiguous. Treat
    # as a known limitation.
    fps_values = {camera_fps_dict[s] for s in session_blocks if s in camera_fps_dict}
    fps = float(next(iter(fps_values)))
    if len(fps_values) > 1:
        print(f"[audit] timescales: WARNING — fps varies across sessions {fps_values}; "
              f"reporting the lag axis at {fps} fps. Per-session lag-in-seconds may differ.")

    max_lag_frames = int(np.ceil(max_lag_seconds * fps))
    # ACF axis: positive lags only [0, L_max]
    acf_lag_grid_frames = np.arange(0, max_lag_frames + 1, dtype=np.int32)
    acf_lag_grid_seconds = acf_lag_grid_frames.astype(np.float32) / fps
    # Signal-correlation axis: symmetric [-L_max, +L_max]
    signal_lag_grid_frames = np.arange(-max_lag_frames, max_lag_frames + 1, dtype=np.int32)
    signal_lag_grid_seconds = signal_lag_grid_frames.astype(np.float32) / fps

    # Shift-bound conversion (used by both nulls). The user-facing
    # range is in seconds; convert here once. Both bounds are positive,
    # `shuffle_min_frames < shuffle_max_frames`, and large enough that
    # `ACF_x(S) ≈ 0` for the slowest feature in the cohort.
    shuffle_min_seconds, shuffle_max_seconds = (
        float(shuffle_range_seconds[0]),
        float(shuffle_range_seconds[1]),
    )
    shuffle_min_frames = int(np.floor(shuffle_min_seconds * fps))
    shuffle_max_frames = int(np.floor(shuffle_max_seconds * fps))

    # ACF + circular-shift null
    # The circular-shift null preserves the autocorrelation structure
    # of `x` (we shift, we don't permute) while breaking the alignment
    # at lag 0. Because `xcorr(x, x_shifted_by_S)[k] = ACF(k + S)`, we
    # can pre-compute one extended-length ACF per (session, feature)
    # and read off null samples by indexing. With shifts drawn from
    # `[shuffle_min_seconds, shuffle_max_seconds]` (≈ 20–60 s, well
    # past the slowest feature's τ_int), every sampled value lies in
    # the ACF's "tail" where the true autocorrelation has decayed,
    # giving a Bartlett-style honest null reflecting the actual
    # estimator variance under the data's spectral properties.
    n_sess_total = len(session_blocks)
    print(f"[audit]   ACF: {n_features} features × {n_sess_total} sessions × "
          f"{max_lag_frames + 1} lags ({max_lag_seconds:.1f} s @ {fps:.1f} fps)")

    acf_extended_max_lag = max_lag_frames + shuffle_max_frames
    acf_long_stack = np.full(
        (n_features, n_sess_total, acf_extended_max_lag + 1),
        np.nan, dtype=np.float32,
    )
    acf_t0 = time.monotonic()
    for s_i, (sess_id, per_feature) in enumerate(session_blocks.items()):
        for f_i, fname in enumerate(feature_names):
            if fname not in per_feature:
                continue
            acf_long_stack[f_i, s_i, :] = _per_session_acf(
                per_feature[fname], acf_extended_max_lag
            ).astype(np.float32)
        if (s_i + 1) % 10 == 0 or (s_i + 1) == n_sess_total:
            elapsed = time.monotonic() - acf_t0
            rate = (s_i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_sess_total - (s_i + 1)) / rate if rate > 0 else float('inf')
            print(f"[audit]   ACF: session {s_i + 1}/{n_sess_total} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    # Display ACF (lags 0..L) is just the leading slice of the long
    # ACF. Median / IQR across sessions give the central / spread
    # bands shown on the plot.
    acf_stack = acf_long_stack[:, :, :max_lag_frames + 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        acf_median = np.nanmedian(acf_stack, axis=1)
        acf_p25 = np.nanpercentile(acf_stack, 25, axis=1)
        acf_p75 = np.nanpercentile(acf_stack, 75, axis=1)

    tau_1e = np.array([_first_crossing_below(acf_median[i], 1.0 / np.e)
                       for i in range(n_features)], dtype=np.float32) / fps
    tau_02 = np.array([_first_crossing_below(acf_median[i], 0.2)
                       for i in range(n_features)], dtype=np.float32) / fps
    tau_int = np.array([_integrated_autocorr_time(acf_median[i])
                        for i in range(n_features)], dtype=np.float32) / fps

    # Circular-shift ACF null: stream by feature, draw `n_shuffles`
    # random shifts per session in [shuffle_min_frames, shuffle_max_frames],
    # collect `acf_long_stack[f, s, S : S+L+1]` for each, pool into a
    # `(n_sess × n_shuffles, L+1)` matrix, reduce to per-lag mean +
    # 0.5 / 99.5 percentile per feature.
    print(f"[audit]   ACF null: {n_shuffles} circular shifts × {n_sess_total} "
          f"sessions per feature, S ∈ [{shuffle_min_seconds:.1f}, "
          f"{shuffle_max_seconds:.1f}] s ({shuffle_min_frames}–{shuffle_max_frames} frames)...")
    acf_null_mean = np.full((n_features, max_lag_frames + 1), np.nan, dtype=np.float32)
    acf_null_p0_5 = np.full((n_features, max_lag_frames + 1), np.nan, dtype=np.float32)
    acf_null_p99_5 = np.full((n_features, max_lag_frames + 1), np.nan, dtype=np.float32)
    rng_acf = np.random.default_rng(random_seed + 1)
    acf_null_t0 = time.monotonic()
    for f_i in range(n_features):
        pool = np.empty((n_sess_total * n_shuffles, max_lag_frames + 1),
                        dtype=np.float32)
        pool_idx = 0
        for s_i in range(n_sess_total):
            if not np.isfinite(acf_long_stack[f_i, s_i, 0]):
                continue
            shifts = rng_acf.integers(shuffle_min_frames, shuffle_max_frames + 1,
                                      size=n_shuffles)
            for S in shifts:
                pool[pool_idx, :] = acf_long_stack[f_i, s_i, int(S):int(S) + max_lag_frames + 1]
                pool_idx += 1
        if pool_idx > 0:
            pool = pool[:pool_idx]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                # Use nan-aware reductions: a session shorter than
                # `acf_extended_max_lag` passes the finite-`acf[0]` gate
                # above but carries a NaN tail from `_per_session_acf`,
                # so its window rows contain NaN at the longer lags.
                # Plain `np.mean` / `np.percentile` would collapse the
                # entire null to NaN at any such lag; `nan*` variants
                # keep the contributions of the full-length sessions,
                # matching the display-ACF (`acf_median`/`p25`/`p75`)
                # and signal-correlation-null reductions elsewhere in
                # this module.
                acf_null_mean[f_i, :] = np.nanmean(pool, axis=0)
                acf_null_p0_5[f_i, :] = np.nanpercentile(pool, 0.5, axis=0)
                acf_null_p99_5[f_i, :] = np.nanpercentile(pool, 99.5, axis=0)
        del pool
        if (f_i + 1) % 5 == 0 or (f_i + 1) == n_features:
            elapsed = time.monotonic() - acf_null_t0
            rate = (f_i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_features - (f_i + 1)) / rate if rate > 0 else float('inf')
            print(f"[audit]   ACF null: feature {f_i + 1}/{n_features} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    del acf_long_stack

    # Signal correlation
    print(f"[audit]   Signal correlation: pooling sessions, "
          f"{n_shuffles} shuffles for null, lags ±{max_lag_seconds:.1f} s")

    # Determine session length per session by picking the first feature
    # we have data for; require a bout-onset times entry for the
    # session — that is the audit's sole source of `Y`. Sessions
    # without bout onsets are excluded (no bouts means no signal
    # correlation to compute and no within-session shuffle null to
    # draw). The `event_intervals_per_session` dict is also required,
    # but only for the IBI-percentile reporting (it is the inter-USV
    # gap source, not a `Y` source).
    if bout_onset_times_per_session is None:
        raise ValueError(
            "audit_predictor_timescales requires `bout_onset_times_per_session`. "
            "The binary `Y(t) = bout-onset-at-frame-t` trace is built "
            "exclusively from per-session bout-onset arrays "
            "(typically `usv_data_dict[sess][target]['positive_events']` "
            "from the vocal_onsets pipeline in `bout` mode)."
        )
    if event_intervals_per_session is None:
        raise ValueError(
            "audit_predictor_timescales requires `event_intervals_per_session` "
            "for the IBI-percentile report block (per-USV `[start, stop)` "
            "arrays — used to compute inter-USV gaps that are directly "
            "comparable to the GMM-derived `ibi_threshold`)."
        )
    valid_sessions = []
    for sess_id, per_feature in session_blocks.items():
        if sess_id not in bout_onset_times_per_session:
            continue
        if sess_id not in camera_fps_dict:
            # The phase-1 loop below indexes ``camera_fps_dict[sess_id]``
            # directly; exclude any session missing a recorded fps here
            # rather than letting it raise a KeyError mid-loop.
            continue
        if not per_feature:
            continue
        n_frames_sess = next(iter(per_feature.values())).size
        valid_sessions.append((sess_id, n_frames_sess))

    n_signal_lags = signal_lag_grid_frames.size
    if not valid_sessions:
        print("[audit]   Signal correlation: no valid sessions with bout onsets — skipping.")
        rho_signal_per_session_mean = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_per_session_sem = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_null_mean = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_null_p0_5 = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_null_p99_5 = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        empirical_pcts = {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')}
        n_total_bouts = 0
        n_total_usvs = 0
    else:
        # Per-session signal correlation (actual + circular-shift null)
        # For every (session × feature) pair we compute a single
        # FFT-based cross-correlation at all positive true lags
        # `[0, n_pad_sess − 1]` once. Two views of that single curve
        # are then read off:
        #
        #   - actual : symmetric lag window `[−L_max, +L_max]` —
        #              this is the per-session ρ_session(k) curve
        #              that gets averaged across sessions for the
        #              displayed mean and SEM band.
        #   - null   : per-shuffle window `[S − L_max, S + L_max]`
        #              for `S ∈ [shuffle_min_frames, shuffle_max_frames]`
        #              uniformly. Equivalent to circularly shifting
        #              the binary bout-onset trace by S and re-cross-
        #              correlating, but obtained for free by indexing
        #              the same precomputed curve. With S ≫ τ_xy the
        #              sampled lags lie in the tail where the true
        #              ρ_xy ≈ 0, giving an honest null that preserves
        #              the autocorrelation structure of both x and y.
        #
        # Aggregation is per feature (streaming): collect per-session
        # per-shuffle null curves into a 3-D `(n_sessions, n_shuffles,
        # n_lags)` array, take cohort-mean per shuffle (`nanmean` over
        # the session axis) to get `n_shuffles` cohort-mean null
        # curves, then per-feature, per-lag mean and 0.5 / 99.5
        # percentiles across those cohort-mean curves. The null band
        # is therefore on the same cohort-mean scale as the plotted
        # line, with width ~`σ_session/√n_sessions`.
        rho_signal_per_session_mean = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_per_session_sem = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_null_mean = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_null_p0_5 = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)
        rho_signal_null_p99_5 = np.full((n_features, n_signal_lags), np.nan, dtype=np.float32)

        n_valid = len(valid_sessions)

        # Pre-cache per-session Y FFTs and norms (one per session,
        # reused across features). Per-session pad length must hold
        # both the actual symmetric window and the longest null
        # window, i.e. `n_frames_sess + max_lag_frames + shuffle_max_frames`.
        print(f"[audit]   Signal correlation: phase 1/2 — caching per-session "
              f"binary-USV FFTs ({n_valid} sessions)...")
        y_t0 = time.monotonic()
        per_sess_n_pad = []
        per_sess_y_fft = []
        per_sess_y_norm = []
        for s_i, (sess_id, n_frames_sess) in enumerate(valid_sessions):
            n_pad_sess = 1
            while n_pad_sess < n_frames_sess + max_lag_frames + shuffle_max_frames + 1:
                n_pad_sess *= 2
            # Build the per-session bout-onset trace: a single `1.0`
            # at the integer frame index of each bout's first USV,
            # zero everywhere else. `valid_sessions` is already gated
            # on having bout onsets so no fallback is needed.
            _bout_times = bout_onset_times_per_session[sess_id]
            Y_sess = _binary_event_trace(
                _bout_times, n_frames_sess, camera_fps_dict[sess_id]
            )
            Y_sess_r = rankdata(Y_sess).astype(np.float32)
            Y_sess_c = Y_sess_r - Y_sess_r.mean()
            y_norm = float(np.sqrt(np.sum(np.square(Y_sess_c, dtype=np.float64))))
            y_fft = np.fft.rfft(Y_sess_c, n=n_pad_sess).astype(np.complex64)
            per_sess_n_pad.append(n_pad_sess)
            per_sess_y_fft.append(y_fft)
            per_sess_y_norm.append(y_norm)
            del Y_sess, Y_sess_r, Y_sess_c
            if (s_i + 1) % 25 == 0 or (s_i + 1) == n_valid:
                elapsed = time.monotonic() - y_t0
                rate = (s_i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_valid - (s_i + 1)) / rate if rate > 0 else float('inf')
                print(f"[audit]   Signal correlation: phase 1 — "
                      f"session {s_i + 1}/{n_valid} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        # Per-feature streaming: compute per-session full cross-
        # correlation, sample actual + null windows, reduce, free.
        print(f"[audit]   Signal correlation: phase 2/2 — per-feature "
              f"per-session cross-correlation + {n_shuffles} circular-shift "
              f"shuffles per session, S ∈ [{shuffle_min_seconds:.1f}, "
              f"{shuffle_max_seconds:.1f}] s...")
        rng_xc = np.random.default_rng(random_seed)
        feat_t0 = time.monotonic()
        for f_i, fname in enumerate(feature_names):
            rho_per_sess = np.full((n_valid, n_signal_lags), np.nan, dtype=np.float32)
            # Null storage is now 3-D: (session, shuffle, lag). Pre-fill
            # with NaN so sessions that fail the `denom > 0` check (or
            # are missing the feature) leave their slab as NaN rather
            # than uninitialized memory. The downstream cohort-mean
            # null then averages across the session axis with `nanmean`,
            # which handles the gaps correctly. Memory footprint is the
            # same as the previous flat `(n_sessions × n_shuffles, n_lags)`
            # pool.
            null_per_sess_shuffle = np.full(
                (n_valid, n_shuffles, n_signal_lags), np.nan, dtype=np.float32
            )

            for s_i, (sess_id, n_frames_sess) in enumerate(valid_sessions):
                per_feature = session_blocks[sess_id]
                if fname not in per_feature:
                    continue

                n_pad_sess = per_sess_n_pad[s_i]
                y_fft = per_sess_y_fft[s_i]
                y_norm = per_sess_y_norm[s_i]

                x_sess = per_feature[fname]
                x_sess = np.where(np.isfinite(x_sess), x_sess, 0.0).astype(np.float32)
                x_sess_r = rankdata(x_sess).astype(np.float32)
                x_sess_c = x_sess_r - x_sess_r.mean()
                x_norm = float(np.sqrt(np.sum(np.square(x_sess_c, dtype=np.float64))))
                x_fft_conj = np.conj(np.fft.rfft(x_sess_c, n=n_pad_sess))

                xcorr_full = np.fft.irfft(x_fft_conj * y_fft, n=n_pad_sess)
                denom = x_norm * y_norm
                # Always advance the RNG by `n_shuffles` draws even when
                # the session is skipped, so that downstream sessions
                # see the same shift sequence regardless of which
                # earlier sessions happened to fail. This keeps results
                # deterministic w.r.t. `random_seed` independent of the
                # set of valid sessions.
                shifts = rng_xc.integers(shuffle_min_frames, shuffle_max_frames + 1,
                                         size=n_shuffles)
                if denom <= 0:
                    # Slab stays NaN; nothing to write.
                    continue

                # Actual: symmetric lag window [-L_max, +L_max] read
                # from the standard FFT lag layout (positive lags at
                # the front, negative lags wrapped to the back).
                xcorr_actual_raw = np.concatenate([
                    xcorr_full[n_pad_sess - max_lag_frames:],
                    xcorr_full[:max_lag_frames + 1],
                ])
                rho_per_sess[s_i, :] = (xcorr_actual_raw / denom).astype(np.float32)

                # Null: n_shuffles random shifts per session. Each shift
                # S samples a contiguous window centred at lag S.
                # `null_per_sess_shuffle[s_i, j, :]` holds the j-th
                # shuffle's lag window for this session.
                for j, S in enumerate(shifts):
                    S_int = int(S)
                    null_per_sess_shuffle[s_i, j, :] = (
                        xcorr_full[S_int - max_lag_frames:S_int + max_lag_frames + 1]
                        / denom
                    ).astype(np.float32)

                del x_sess, x_sess_r, x_sess_c, x_fft_conj, xcorr_full

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                rho_signal_per_session_mean[f_i, :] = np.nanmean(rho_per_sess, axis=0).astype(np.float32)
                rho_per_sess_std = np.nanstd(rho_per_sess, axis=0, ddof=1).astype(np.float32)
                n_valid_per_lag = np.sum(np.isfinite(rho_per_sess), axis=0)
                rho_signal_per_session_sem[f_i, :] = (
                    rho_per_sess_std / np.sqrt(np.maximum(n_valid_per_lag, 1))
                ).astype(np.float32)

                # Cohort-mean null: pair shuffles by index across
                # sessions (valid since within each session the shifts
                # are i.i.d. uniform draws), average across sessions
                # for each shuffle index, then take percentiles across
                # the resulting `n_shuffles` cohort-mean curves. This
                # makes the null band match the scale of the plotted
                # cohort-mean ρ — about `1/√n_sessions` narrower than
                # the per-session null pool the previous implementation
                # reported.
                if np.any(np.isfinite(null_per_sess_shuffle)):
                    null_cohort_means = np.nanmean(null_per_sess_shuffle, axis=0)
                    rho_signal_null_mean[f_i, :] = np.nanmean(
                        null_cohort_means, axis=0
                    ).astype(np.float32)
                    rho_signal_null_p0_5[f_i, :] = np.nanpercentile(
                        null_cohort_means, 0.5, axis=0
                    ).astype(np.float32)
                    rho_signal_null_p99_5[f_i, :] = np.nanpercentile(
                        null_cohort_means, 99.5, axis=0
                    ).astype(np.float32)
                    del null_cohort_means
            del rho_per_sess, rho_per_sess_std, n_valid_per_lag, null_per_sess_shuffle

            if (f_i + 1) % 5 == 0 or (f_i + 1) == n_features:
                elapsed = time.monotonic() - feat_t0
                rate = (f_i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_features - (f_i + 1)) / rate if rate > 0 else float('inf')
                print(f"[audit]   Signal correlation: phase 2 — "
                      f"feature {f_i + 1}/{n_features} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        del per_sess_n_pad, per_sess_y_fft, per_sess_y_norm

        # Empirical IBI percentiles, computed as inter-USV gaps using
        # the same definition the loader applies against the
        # GMM-derived `ibi_threshold`: `gap_i = start[i+1] - stop[i]`.
        # Per-USV starts and stops come straight from
        # `event_intervals_per_session[sess]` *when available*. Pipelines
        # whose data dict carries bout onsets but no per-USV
        # `[start, stop)` arrays (e.g. the bout-parameters pipeline,
        # whose `bout_data_dict[sess][target]` exposes `bout_onsets`
        # but no `start` / `stop`) supply an empty
        # `event_intervals_per_session` dict. For those sessions we
        # skip the IBI gap computation (the percentiles end up NaN
        # for the cohort) but still count bouts from
        # `bout_onset_times_per_session`, which is the audit's `Y`-
        # event source and is populated for every session in
        # `valid_sessions`.
        all_ibis = []
        n_total_usvs = 0
        n_total_bouts = 0
        for sess_id, _ in valid_sessions:
            if sess_id in event_intervals_per_session:
                starts, stops = event_intervals_per_session[sess_id]
                starts = np.asarray(starts, dtype=np.float64)
                stops = np.asarray(stops, dtype=np.float64)
                order = np.argsort(starts)
                starts = starts[order]
                stops = stops[order]
                n_total_usvs += int(starts.size)
                if starts.size > 1:
                    gaps = starts[1:] - stops[:-1]
                    gaps = gaps[np.isfinite(gaps)]
                    if gaps.size > 0:
                        all_ibis.append(gaps)
            n_total_bouts += int(np.asarray(
                bout_onset_times_per_session[sess_id]
            ).size)
        if all_ibis:
            ibis = np.concatenate(all_ibis)
            empirical_pcts = {
                'p50': float(np.percentile(ibis, 50)),
                'p90': float(np.percentile(ibis, 90)),
                'p99': float(np.percentile(ibis, 99)),
            }
        else:
            empirical_pcts = {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')}

    payload = {
        'features': feature_names,
        # ACF (positive lags only). `acf_null_*` are the per-feature,
        # per-lag mean and 0.5/99.5 percentiles of the circular-shift
        # null (shifts in [shuffle_min, shuffle_max] seconds, n_shuffles
        # per session, pooled across (session, shuffle)).
        'acf_lags_frames': acf_lag_grid_frames,
        'acf_lags_seconds': acf_lag_grid_seconds,
        'acf_median': acf_median.astype(np.float32),
        'acf_p25': acf_p25.astype(np.float32),
        'acf_p75': acf_p75.astype(np.float32),
        'acf_null_mean': acf_null_mean,
        'acf_null_p0_5': acf_null_p0_5,
        'acf_null_p99_5': acf_null_p99_5,
        'tau_acf_1_over_e': tau_1e,
        'tau_acf_0_2': tau_02,
        'tau_acf_integrated': tau_int,
        # Signal correlation (symmetric lags, ρ vs bout-onset indicator).
        # `rho_signal` is the per-session mean across the cohort;
        # `rho_signal_per_session_sem` is the SEM around that mean.
        # `rho_signal_null_*` are the per-feature, per-lag mean and
        # 0.5/99.5 percentiles of the circular-shift null on the
        # cohort-mean scale (shuffles paired by index across sessions,
        # cohort-mean computed per shuffle, percentiles across the
        # n_shuffles cohort-mean curves) — matches the SEM scale of
        # the line.
        'signal_lags_frames': signal_lag_grid_frames,
        'signal_lags_seconds': signal_lag_grid_seconds,
        'rho_signal': rho_signal_per_session_mean,
        # Self-documenting alias of `rho_signal` for external readers;
        # in-repo plotting consumers read only `rho_signal`.
        'rho_signal_per_session_mean': rho_signal_per_session_mean,
        'rho_signal_per_session_sem': rho_signal_per_session_sem,
        'rho_signal_null_mean': rho_signal_null_mean,
        'rho_signal_null_p0_5': rho_signal_null_p0_5,
        'rho_signal_null_p99_5': rho_signal_null_p99_5,
        # Response side
        'ibi_thresholds': dict(ibi_thresholds),
        'ibi_empirical_pcts': empirical_pcts,
        # Context. `n_events` is retained as an alias for `n_bouts`
        # (the audit's `Y` event count) so older readers don't break;
        # `n_usvs` is the underlying per-USV count for reference.
        'configured_filter_history': float(configured_filter_history),
        'signal_floor_seconds': float(signal_floor_seconds),
        'signal_min_run_seconds': float(signal_min_run_seconds),
        'n_events': int(n_total_bouts),
        'n_bouts': int(n_total_bouts),
        'n_usvs': int(n_total_usvs),
        'n_sessions': len(session_blocks),
        'source_pickle': source_pickle,
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    if input_metadata is not None:
        payload['_input_metadata'] = dict(input_metadata)

    # Stdout headline
    finite_int = tau_int[np.isfinite(tau_int)]
    max_int = float(np.max(finite_int)) if finite_int.size else float('nan')
    arg_int = feature_names[int(np.nanargmax(tau_int))] if finite_int.size else '-'

    # Signal-correlation peak: per-feature argmax|ρ|, then global max.
    rho_for_peak = (
        rho_signal_per_session_mean
        if rho_signal_per_session_mean.size
        else np.empty((0, 0), dtype=np.float32)
    )
    signal_abs = np.abs(rho_for_peak)
    if signal_abs.size and np.any(np.isfinite(signal_abs)):
        # A feature held constant across *every* session has an all-NaN signal
        # row; `np.nanargmax(..., axis=1)` raises `ValueError: All-NaN slice`
        # on such a row even when other features are fine, which would crash
        # the whole audit. Fill non-finite entries with -inf so the per-row
        # argmax never sees an all-NaN slice (those rows resolve to lag-index 0
        # and carry an all-NaN `peak_abs`, which the nan-aware global argmax
        # below skips). `np.nanmax` on an all-NaN row only warns (not raises),
        # so keep the RuntimeWarning suppression for it.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            safe_signal_abs = np.where(np.isfinite(signal_abs), signal_abs, -np.inf)
            peak_lag_idx_per_feature = np.argmax(safe_signal_abs, axis=1)
            peak_abs_per_feature = np.nanmax(signal_abs, axis=1)
        peak_signed_per_feature = np.array([
            rho_for_peak[i, peak_lag_idx_per_feature[i]] for i in range(n_features)
        ], dtype=np.float32)
        global_max_idx = int(np.nanargmax(peak_abs_per_feature))
        peak_feat = feature_names[global_max_idx]
        peak_rho = float(peak_signed_per_feature[global_max_idx])
        peak_lag_s = float(signal_lag_grid_seconds[peak_lag_idx_per_feature[global_max_idx]])
        if peak_lag_s < 0:
            direction = 'bout leads feature'
        elif peak_lag_s > 0:
            direction = 'feature leads bout'
        else:
            direction = 'simultaneous'
    else:
        peak_feat = '-'
        peak_rho = float('nan')
        peak_lag_s = float('nan')
        direction = '-'

    print("\n" + "=" * 72)
    print(f"TIMESCALE AUDIT  ({len(session_blocks)} sessions × {n_features} features, "
          f"{n_total_bouts} bouts / {n_total_usvs} USVs)")
    print("=" * 72)
    print(f"  ACF                : max τ_int = {max_int:5.2f} s   (feature: {arg_int})")
    print(f"  Signal correlation : peak ρ = {peak_rho:+.4f} at lag = {peak_lag_s:+5.2f} s   "
          f"(feature: {peak_feat}, {direction})")
    p90 = empirical_pcts['p90']
    print(f"  Response (IBI 90th-pct) = {p90:5.2f} s")
    print(f"  Configured filter_history = {configured_filter_history:5.2f} s")
    print("=" * 72 + "\n")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(save_path).open('wb') as fh:
        pickle.dump(payload, fh)
    print(f"[audit] timescale artifact written: {save_path}")

    return payload
