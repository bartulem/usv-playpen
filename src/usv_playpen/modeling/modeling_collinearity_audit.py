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

2. `audit_predictor_timescales` — answers "is the configured
   `filter_history` window long enough to capture all useful predictor
   information?"  Reports two complementary summaries:
     * Predictor autocorrelation (ACF) → lower bound on the window
       (anything shorter throws away in-feature memory).
     * Event-locked predictive Spearman ρ at varying lead-times → upper
       bound on the window (anything longer no longer carries event-locked
       signal above a within-session circular-shift null).
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


def _spearman_at_lag(X_ranks: np.ndarray,
                     Y_ranks: np.ndarray,
                     lag: int) -> np.ndarray:
    """
    Computes Spearman ρ between every column of `X_ranks` shifted by
    `lag` frames and `Y_ranks`, in a single vectorised pass.

    Operates on rank-transformed arrays so the per-call cost reduces to
    a centred dot product divided by the norms — vastly faster than
    calling `scipy.stats.spearmanr` once per (feature, lag) pair, which
    re-ranks both arrays every time.

    Parameters
    ----------
    X_ranks : np.ndarray
        Rank-transformed feature matrix of shape `(n_frames, n_features)`.
    Y_ranks : np.ndarray
        Rank-transformed event indicator of shape `(n_frames,)`.
    lag : int
        Number of frames to shift `X` *backward* relative to `Y`. The
        comparison aligns `X[:-lag]` with `Y[lag:]`, so a positive lag
        measures `corr(X(t-lag), Y(t))`.

    Returns
    -------
    np.ndarray
        Per-feature Spearman ρ at the supplied lag, shape `(n_features,)`.
    """

    if lag == 0:
        Xs = X_ranks
        Ys = Y_ranks
    else:
        Xs = X_ranks[:-lag]
        Ys = Y_ranks[lag:]

    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Ys = Ys - Ys.mean()
    denom_x = np.sqrt(np.sum(Xs ** 2, axis=0))
    denom_y = float(np.sqrt(np.sum(Ys ** 2)))
    denom = denom_x * denom_y
    rho = np.zeros(Xs.shape[1], dtype=np.float64)
    valid = denom > 0
    rho[valid] = (Xs[:, valid].T @ Ys) / denom[valid]
    return rho


def audit_predictor_timescales(processed_beh_dict: dict,
                               event_times_per_session: dict,
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
                               input_metadata: dict = None) -> dict:
    """
    Computes ACF (lower bound) and event-locked predictive ρ (upper bound)
    timescales for every kept predictor, alongside the response-side IBI
    distribution, and persists the result to disk.

    The ACF profile answers "below what window length am I throwing away
    a feature's own memory?"  The predictive-ρ profile answers "above what
    window length is no kept feature still carrying event-locked signal
    above the within-session shuffle null?"  Together they bracket the
    defensible range for the configured `filter_history`.

    Predictive-ρ implementation
    ---------------------------
    For each feature `f` the kept session traces are concatenated into a
    single rank-transformed array `X_f`, and the per-session binary event
    indicator is concatenated into `Y`. Spearman ρ is then computed at
    every lag Δ ∈ {0, 1, …, L_max_frames} via the centred-dot-product
    shortcut on the pre-ranked arrays (one sort per feature, one per
    shuffle). The within-session circular-shift null shifts each
    session's `Y` by a uniformly drawn offset before re-pooling — this
    preserves `Y`'s autocorrelation structure (so the null is not
    degenerately tight) while destroying the temporal alignment with
    `X`. `τ_predictive(f)` is the largest lag at which the actual ρ
    magnitude exceeds the per-lag 95th percentile of |ρ| under the null.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping `session_id -> polars.DataFrame` after z-scoring.
    event_times_per_session : dict
        Mapping `session_id -> np.ndarray` of pooled event onset times
        (seconds). Used both to derive `Y` and to anchor the ACF window
        to the same set of sessions the model will be trained on.
    mouse_names_dict : dict
        Mapping `session_id -> list[mouse_name]`.
    target_idx, predictor_idx : int
        Mouse slot indices.
    configured_filter_history : float
        The `filter_history` value (in seconds) currently configured in
        `modeling_settings.json`. Used only for the recommendation line
        and the plot vertical-line annotation.
    camera_fps_dict : dict
        Mapping `session_id -> camera_sampling_rate_hz`.
    max_lag_seconds : float
        Upper bound of the lag axis (in seconds).
    n_shuffles : int
        Number of within-session circular-shift shuffles used to build the
        per-lag predictive-ρ null.
    ibi_thresholds : dict
        Pre-computed `{'male': float, 'female': float}` IBI thresholds
        from `_calculate_ibi_threshold`. Stored in the artifact for the
        recommendation line; not recomputed here.
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
            'lags_frames': np.empty((0,), dtype=np.int32),
            'lags_seconds': np.empty((0,), dtype=np.float32),
            'acf_median': np.empty((0, 0), dtype=np.float32),
            'acf_p25': np.empty((0, 0), dtype=np.float32),
            'acf_p75': np.empty((0, 0), dtype=np.float32),
            'tau_acf_1_over_e': np.empty((0,), dtype=np.float32),
            'tau_acf_0_2': np.empty((0,), dtype=np.float32),
            'tau_acf_integrated': np.empty((0,), dtype=np.float32),
            'rho_predictive': np.empty((0, 0), dtype=np.float32),
            'rho_predictive_null_p95': np.empty((0, 0), dtype=np.float32),
            'tau_predictive': np.empty((0,), dtype=np.float32),
            'ibi_thresholds': dict(ibi_thresholds),
            'ibi_empirical_pcts': {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')},
            'configured_filter_history': float(configured_filter_history),
            'n_events': 0, 'n_sessions': 0,
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
    # and `_binary_event_trace` still use each session's own fps
    # internally, but a mixed-fps run would render the artifact's
    # `lags_seconds` axis ambiguous. Treat as a known limitation.
    fps_values = {camera_fps_dict[s] for s in session_blocks if s in camera_fps_dict}
    fps = float(next(iter(fps_values)))
    if len(fps_values) > 1:
        print(f"[audit] timescales: WARNING — fps varies across sessions {fps_values}; "
              f"reporting the lag axis at {fps} fps. Per-session lag-in-seconds may differ.")

    max_lag_frames = int(np.ceil(max_lag_seconds * fps))
    lag_grid_frames = np.arange(0, max_lag_frames + 1, dtype=np.int32)
    lag_grid_seconds = lag_grid_frames.astype(np.float32) / fps

    # ----------------- ACF (lower bound) -----------------
    print(f"[audit]   ACF: {n_features} features × {len(session_blocks)} sessions × "
          f"{max_lag_frames + 1} lags ({max_lag_seconds:.1f} s @ {fps:.1f} fps)")

    acf_stack = np.full((n_features, len(session_blocks), max_lag_frames + 1),
                        np.nan, dtype=np.float32)
    n_sess_total = len(session_blocks)
    acf_t0 = time.monotonic()
    for s_i, (sess_id, per_feature) in enumerate(session_blocks.items()):
        for f_i, fname in enumerate(feature_names):
            if fname not in per_feature:
                continue
            acf_stack[f_i, s_i, :] = _per_session_acf(per_feature[fname],
                                                     max_lag_frames).astype(np.float32)
        if (s_i + 1) % 10 == 0 or (s_i + 1) == n_sess_total:
            elapsed = time.monotonic() - acf_t0
            rate = (s_i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_sess_total - (s_i + 1)) / rate if rate > 0 else float('inf')
            print(f"[audit]   ACF: session {s_i + 1}/{n_sess_total} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    # `np.nanmedian` / `np.nanpercentile` emit a "All-NaN slice
    # encountered" warning whenever a (feature, lag) cell is NaN for
    # every session — common for features that fail to compute on
    # constant traces or that the loader stripped from some sessions.
    # The NaNs propagate cleanly through the rest of the audit; the
    # warning is just stdout noise.
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

    # ----------------- Predictive ρ (upper bound) -----------------
    print(f"[audit]   Predictive ρ: pooling sessions, {n_shuffles} shuffles for null")

    # Determine session length per session by picking the first feature
    # we have data for; require an event_times entry for the session.
    # Sessions without an event-times entry are excluded (the predictive-ρ
    # null shuffles per-session indicators, so each session needs at
    # least one event); sessions where a feature is absent receive a
    # zero-filled column further below so per-feature column shapes are
    # preserved.
    valid_sessions = []
    for sess_id, per_feature in session_blocks.items():
        if sess_id not in event_times_per_session:
            continue
        if not per_feature:
            continue
        n_frames_sess = next(iter(per_feature.values())).size
        valid_sessions.append((sess_id, n_frames_sess))

    if not valid_sessions:
        print("[audit]   Predictive ρ: no valid sessions with events — skipping.")
        rho_actual = np.full((n_features, len(lag_grid_frames)), np.nan, dtype=np.float32)
        rho_null_p95 = np.full((n_features, len(lag_grid_frames)), np.nan, dtype=np.float32)
        tau_pred = np.zeros(n_features, dtype=np.float32)
        empirical_pcts = {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')}
        n_total_events = 0
    else:
        # ----- FFT-based predictive-ρ -----
        # We compute Spearman ρ between every feature's rank-vector and
        # both the actual event indicator's ranks and `n_shuffles`
        # within-session circular-shifted indicator ranks, at every lag
        # in `lag_grid_frames`.
        #
        # The naive implementation evaluates each (feature, shuffle,
        # lag) triple as a separate centred dot product in a Python
        # loop. With this cohort's scale (~500 features × ~100 sessions
        # × ~10^5 frames × ~10^3 lags × 50 shuffles ≈ 4 × 10^14
        # element-pair ops dispatched as ~4 × 10^7 small NumPy calls
        # of ~10^7 elements each) the per-call bandwidth-limited
        # overhead made the streaming Python loop infeasible —
        # estimated ~10^4 hours wall-clock on the actual workload.
        #
        # The fundamental observation is that for fixed feature `f` and
        # fixed Y variant `v`, the per-lag cross-correlation
        #     xcorr_fv(k) = Σ_t x_centered[t] * y_v_centered[t + k]
        # for all k = 0..L_max can be computed in `O(N log N)` via the
        # cross-correlation theorem (FFT of the centred rank vectors,
        # multiply, inverse FFT) instead of `O(N · L_max)` via per-lag
        # dot products. With `N ≈ 10^7`, `L_max ≈ 1500`, that's a ~50×
        # algorithmic speedup on top of pocketfft's native vectorised
        # throughput.
        #
        # The Y FFT (one per shuffle variant) is amortised across all
        # `n_features` features because it does not depend on which
        # feature is being scored. Per-feature work is one rank-transform,
        # one rfft, plus `n_shuffles + 1` complex-multiply-and-irfft
        # passes — each a single bandwidth-limited operation on
        # `O(n_pad)` elements rather than `n_lags × n_shuffles` separate
        # passes.
        #
        # Numerical note: we use the global-mean / global-norm Pearson
        # form (the standard ACF/CCF estimator) rather than the
        # windowed-mean form used by the legacy code. For
        # `L_max / N ≈ 10^-4` the per-lag means and variances are
        # numerically indistinguishable from the global ones; the
        # global form additionally avoids catastrophic cancellation in
        # the rank-product subtraction (the windowed `Σ x·y - N·μ_x·μ_y`
        # involved subtracting two ~10^14 quantities to recover a ~10^7
        # difference, which loses ~6 digits of float32 precision; the
        # centred form `Σ x_c · y_c` directly recovers the small
        # difference).
        total_frames = sum(n for _, n in valid_sessions)
        ses_starts = []
        cursor = 0
        for _, n_frames_sess in valid_sessions:
            ses_starts.append(cursor)
            cursor += n_frames_sess
        ses_starts.append(cursor)

        # FFT pad length: must satisfy `n_pad >= total_frames + L_max`
        # so the circular-correlation wrap-around does not contaminate
        # any lag in `[0, L_max]`. We round up to the next power of two
        # because pocketfft is fastest on power-of-two lengths.
        n_pad = 1
        while n_pad < total_frames + max_lag_frames + 1:
            n_pad *= 2

        # Build the pooled binary event trace once.
        Y_pooled = np.zeros(total_frames, dtype=np.float32)
        for k, (sess_id, n_frames_sess) in enumerate(valid_sessions):
            start = ses_starts[k]
            ev_times = event_times_per_session[sess_id]
            Y_pooled[start:start + n_frames_sess] = _binary_event_trace(
                ev_times, n_frames_sess, camera_fps_dict[sess_id]
            )

        # Pre-compute centred-rank FFTs for actual + every shuffle. The
        # ρ formula collapses to `Σ x_c · y_c / (||x_c|| · ||y_c||)` so we
        # store the per-variant `||y_c||` alongside the FFT.
        rng = np.random.default_rng(random_seed)
        n_variants = n_shuffles + 1   # index 0 = actual, 1..n_shuffles = shuffles
        Y_ffts = []                    # length n_variants, each (n_pad // 2 + 1) complex64
        Y_norms = np.zeros(n_variants, dtype=np.float64)

        print(f"[audit]   Predictive ρ: pre-computing {n_variants} "
              f"Y-variant rank FFTs (1 actual + {n_shuffles} shuffles, "
              f"n_pad={n_pad}, total_frames={total_frames})...")
        y_t0 = time.monotonic()
        for vi in range(n_variants):
            if vi == 0:
                Y_v = Y_pooled
            else:
                Y_v = np.empty_like(Y_pooled)
                for k in range(len(valid_sessions)):
                    start = ses_starts[k]
                    end = ses_starts[k + 1]
                    length = end - start
                    shift = int(rng.integers(0, length))
                    Y_v[start:end] = np.roll(Y_pooled[start:end], shift)

            Y_r = rankdata(Y_v).astype(np.float32)
            Y_c = Y_r - Y_r.mean()
            Y_ffts.append(np.fft.rfft(Y_c, n=n_pad))
            Y_norms[vi] = float(np.sqrt(np.sum(np.square(Y_c, dtype=np.float64))))
            del Y_v, Y_r, Y_c
            if (vi + 1) % 10 == 0 or (vi + 1) == n_variants:
                elapsed = time.monotonic() - y_t0
                rate = (vi + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_variants - (vi + 1)) / rate if rate > 0 else float('inf')
                print(f"[audit]   Predictive ρ: Y-variant {vi + 1}/{n_variants} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        rho_actual = np.zeros((n_features, len(lag_grid_frames)), dtype=np.float32)
        null_abs_max = np.zeros((n_shuffles, n_features, len(lag_grid_frames)),
                                dtype=np.float32)

        # Feature loop. Per feature: build pooled column, rank, centre,
        # rfft, compute its norm, then `n_variants` complex-multiply +
        # irfft passes for the cross-correlations and divide by the
        # paired norms.
        print(f"[audit]   Predictive ρ: scoring {n_features} features "
              f"× {n_variants} Y-variants × {max_lag_frames + 1} lags...")
        feat_t0 = time.monotonic()
        for f_i, fname in enumerate(feature_names):
            x_col = np.zeros(total_frames, dtype=np.float32)
            for k, (sess_id, n_frames_sess) in enumerate(valid_sessions):
                per_feature = session_blocks[sess_id]
                if fname not in per_feature:
                    continue
                arr = per_feature[fname]
                arr = np.where(np.isfinite(arr), arr, 0.0)
                x_col[ses_starts[k]:ses_starts[k] + n_frames_sess] = arr

            x_ranks = rankdata(x_col).astype(np.float32)
            x_centered = x_ranks - x_ranks.mean()
            x_norm = float(np.sqrt(np.sum(np.square(x_centered, dtype=np.float64))))
            x_fft = np.fft.rfft(x_centered, n=n_pad)
            x_fft_conj = np.conj(x_fft)

            for vi in range(n_variants):
                # Cross-correlation: xcorr(k) = Σ_t x_c[t] * y_c[t + k]
                #                            = irfft(conj(X) * Y)[k]
                # for k = 0..max_lag_frames; padding to `n_pad`
                # eliminates circular wrap-around.
                xcorr = np.fft.irfft(x_fft_conj * Y_ffts[vi], n=n_pad)
                xcorr_lags = xcorr[:max_lag_frames + 1]

                denom = x_norm * Y_norms[vi]
                if denom > 0:
                    rho_k = (xcorr_lags / denom).astype(np.float32)
                else:
                    rho_k = np.zeros(max_lag_frames + 1, dtype=np.float32)

                if vi == 0:
                    rho_actual[f_i, :] = rho_k
                else:
                    null_abs_max[vi - 1, f_i, :] = np.abs(rho_k)

            del x_col, x_ranks, x_centered, x_fft, x_fft_conj

            if (f_i + 1) % 25 == 0 or (f_i + 1) == n_features:
                elapsed = time.monotonic() - feat_t0
                rate = (f_i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_features - (f_i + 1)) / rate if rate > 0 else float('inf')
                print(f"[audit]   Predictive ρ: feature {f_i + 1}/{n_features} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        rho_null_p95 = np.percentile(null_abs_max, 95, axis=0).astype(np.float32)

        tau_pred_frames = np.zeros(n_features, dtype=np.float32)
        for f_i in range(n_features):
            above = np.where(np.abs(rho_actual[f_i]) > rho_null_p95[f_i])[0]
            tau_pred_frames[f_i] = float(above.max()) if above.size else 0.0
        tau_pred = tau_pred_frames / fps

        # Empirical IBI percentiles from the pooled event onsets.
        all_ibis = []
        for sess_id, _ in valid_sessions:
            ev = np.sort(np.asarray(event_times_per_session[sess_id]))
            if ev.size > 1:
                all_ibis.append(np.diff(ev))
        if all_ibis:
            ibis = np.concatenate(all_ibis)
            empirical_pcts = {
                'p50': float(np.percentile(ibis, 50)),
                'p90': float(np.percentile(ibis, 90)),
                'p99': float(np.percentile(ibis, 99)),
            }
        else:
            empirical_pcts = {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')}
        n_total_events = int(sum(np.asarray(event_times_per_session[s]).size
                                  for s, _ in valid_sessions))

    payload = {
        'features': feature_names,
        'lags_frames': lag_grid_frames,
        'lags_seconds': lag_grid_seconds,
        # ACF (lower bound)
        'acf_median': acf_median.astype(np.float32),
        'acf_p25': acf_p25.astype(np.float32),
        'acf_p75': acf_p75.astype(np.float32),
        'tau_acf_1_over_e': tau_1e,
        'tau_acf_0_2': tau_02,
        'tau_acf_integrated': tau_int,
        # Predictive ρ (upper bound)
        'rho_predictive': rho_actual,
        'rho_predictive_null_p95': rho_null_p95,
        'tau_predictive': tau_pred,
        # Response side
        'ibi_thresholds': dict(ibi_thresholds),
        'ibi_empirical_pcts': empirical_pcts,
        # Context
        'configured_filter_history': float(configured_filter_history),
        'n_events': int(n_total_events),
        'n_sessions': len(session_blocks),
        'source_pickle': source_pickle,
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    if input_metadata is not None:
        payload['_input_metadata'] = dict(input_metadata)

    # Stdout headline
    finite_int = tau_int[np.isfinite(tau_int)]
    finite_pred = tau_pred[np.isfinite(tau_pred)]
    max_int = float(np.max(finite_int)) if finite_int.size else float('nan')
    arg_int = feature_names[int(np.nanargmax(tau_int))] if finite_int.size else '-'
    max_pred = float(np.max(finite_pred)) if finite_pred.size else float('nan')
    arg_pred = feature_names[int(np.nanargmax(tau_pred))] if finite_pred.size else '-'

    print("\n" + "=" * 72)
    print(f"TIMESCALE AUDIT  ({len(session_blocks)} sessions × {n_features} features)")
    print("=" * 72)
    print(f"  ACF lower bound  : max τ_int    = {max_int:5.2f} s   (feature: {arg_int})")
    print(f"  Predictive upper : max τ_pred   = {max_pred:5.2f} s   (feature: {arg_pred})")
    p90 = empirical_pcts['p90']
    print(f"  Response (IBI 90th-pct)         = {p90:5.2f} s")
    print(f"  Configured filter_history       = {configured_filter_history:5.2f} s", end='')
    bounds = [b for b in (max_int, max_pred, p90) if np.isfinite(b)]
    if bounds and configured_filter_history < max(bounds):
        print("   ** WARNING: below at least one bound **")
    else:
        print("   OK")
    print("=" * 72 + "\n")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(save_path).open('wb') as fh:
        pickle.dump(payload, fh)
    print(f"[audit] timescale artifact written: {save_path}")

    return payload
