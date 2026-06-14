"""
@author: bartulem
Pools data from different sessions and normalizes them together.
"""

import polars as pls


def zscore_different_sessions_together(data_dict: dict,
                                       feature_lst: list,
                                       feature_bounds: dict,
                                       abs_features: list | None = None,
                                       smooth_abs_features: dict | None = None) -> dict:
    """
    Computes z-scored behavioral data when different sessions need to
    be pooled together (z-scores are computed on the pooled data,
    instead of on each session separately).

    Per-feature pre-processing branches
    -----------------------------------
    Before z-scoring, every column matching `feature_lst` is routed
    through exactly one of three branches, in priority order:

    1. **`smooth_abs_features` (smooth-magnitude fold)** — for
       features whose suffix is a key in `smooth_abs_features`,
       values are replaced by `sqrt(x² + ε²)` where `ε` is the
       per-feature epsilon read from the dict (in the feature's
       native units, e.g. degrees for angles). For `|x| ≫ ε` this
       function is indistinguishable from `|x|` to sub-percent
       accuracy; near `x = 0` it rounds the V-shaped corner of
       `|x|` into a smooth bowl with finite curvature `1/ε`. This
       branch exists specifically because pygam's penalised B-spline
       fit (with the second-difference smoothness penalty) cannot
       reconcile a hard corner in the modelled relationship with
       its smoothness assumption when the data is also concentrated
       at that corner — IRLS conditioning collapses, the
       weighted-normal-equations system becomes near-singular, and
       the solver grinds toward `max_iter` without converging. Use
       this branch for features where the *signed* distribution
       peaks sharply at zero (e.g. `ego_yaw`, `back_yaw` — head
       held aligned with body most of the time), so that after
       folding the bulk of mass would sit right on the kink and
       poison the fit. Recommended ε is small relative to
       measurement noise: `ε = 1.0°` for ego_yaw (range ±180°),
       `ε = 0.5°` for back_yaw (range ±36°). The smooth-abs branch
       skips theoretical-bounds clipping (the transformed values
       are non-negative and the `+ε²` shifts the upper end
       infinitesimally past the theoretical maximum, which the
       clip would mistakenly null out).
    2. **`abs_features` (plain magnitude fold)** — for features
       whose suffix is in this list AND not in
       `smooth_abs_features`, values are replaced by `|x|` exactly.
       Use this for features whose signed distribution is
       *spread out* across the range (e.g. dyadic angles
       `allo_yaw-nose`, where the focal mouse occupies many
       bearings relative to the partner): folding gives a
       reasonably uniform `[0, max]` distribution where almost no
       observations sit on the kink, so the spline fit is
       well-conditioned even with the corner.
    3. **Theoretical-bounds clip** — for the rest (excluding
       `usv_*` columns), values outside `feature_bounds[base]` are
       set to null. Outlier guard.

    After this routing, every kept feature is z-scored against the
    pooled across-session mean and std (NaN -> null coercion before
    the aggregation, so a single NaN frame in one session doesn't
    poison the global statistics).

    Parameters
    ----------
    data_dict : dict
        Original behavioral feature data for each session
        (`{session_id: polars.DataFrame}`).
    feature_lst : list
        Relevant behavioral features (suffixes only — column
        prefixes such as `self.` / `other.` are stripped before the
        comparison).
    feature_bounds : dict
        Dictionary with theoretical boundaries
        `{feature: (min, max)}` per feature (such that outliers can
        be excluded). Only consulted in the bounds-clip branch.
    abs_features : list, optional
        Feature suffixes whose values should be replaced by their
        absolute value (`|x|`) *prior* to z-scoring. Use for
        features whose signed distribution is spread out enough
        across the range that the resulting kink at zero carries
        little data mass. When None, no features are folded with
        plain abs.
    smooth_abs_features : dict, optional
        Mapping `{feature_suffix: epsilon}` for features that
        should be folded onto magnitude *smoothly* via
        `sqrt(x² + ε²)` rather than via plain `|x|`. Use for
        features whose signed distribution has a sharp peak at
        zero (so that after plain folding the bulk of mass would
        sit on the kink). When None, no features take the
        smooth-abs branch. A feature listed in both
        `smooth_abs_features` and `abs_features` is treated as
        smooth-abs (the dict key wins).

    Returns
    -------
    data_dict : dict
        Behavioral data (z-scored). The same dict object is
        mutated in-place AND returned, for compatibility with
        existing callers.
    """

    if abs_features is None:
        abs_features = []
    if smooth_abs_features is None:
        smooth_abs_features = {}

    abs_features_set = set(abs_features)
    smooth_abs_set = set(smooth_abs_features.keys())

    # Clean the data *in place* in the DataFrames
    for one_beh_session, df in data_dict.items():
        cols_to_update = []
        for column in df.columns:
            base_feature = column.split('.')[-1]
            if base_feature not in feature_lst:
                continue

            if base_feature in smooth_abs_set:
                # Smooth-abs branch: `sqrt(x² + ε²)`. ε comes from
                # the per-feature mapping. Like the plain-abs branch
                # below, this skips theoretical-bounds clipping —
                # justified because the transformed values are
                # non-negative by construction and we don't want a
                # value at e.g. `back_yaw = 36°` (ε = 0.5°) mapping
                # to `36.003°` and then being nulled by a strict
                # `≤ 36` upper-bound check.
                eps = float(smooth_abs_features[base_feature])
                cols_to_update.append(
                    (pls.col(column).pow(2) + (eps * eps)).sqrt().alias(column)
                )
            elif base_feature in abs_features_set:
                cols_to_update.append(pls.col(column).abs().alias(column))
            elif 'usv_' not in base_feature:
                theoretical_min, theoretical_max = feature_bounds[base_feature]
                cols_to_update.append(
                    pls.when((pls.col(column) >= theoretical_min) & (pls.col(column) <= theoretical_max))
                    .then(pls.col(column))
                    .otherwise(pls.lit(None))
                    .alias(column)
                )

        if cols_to_update:
            data_dict[one_beh_session] = data_dict[one_beh_session].with_columns(cols_to_update)

    # Build pooled series for each feature and get global stats
    for one_feature in feature_lst:
        pooled_series_list = []

        for dd_df in data_dict.values():
            for column in dd_df.columns:
                if column.split('.')[-1] == one_feature:
                    pooled_series_list.append(dd_df[column].cast(pls.Float32))
                    break

        if not pooled_series_list:
            continue

        base_ds = pls.concat(pooled_series_list, how='vertical')

        # Calculate global mean and std. Polars ignores nulls by
        # default, but float NaN values propagate through aggregations
        # (mean/std become NaN if any element is NaN). The bounds-clip
        # branch converts NaN -> null implicitly via the failed
        # comparison, but the abs() and smooth-abs branches (used for
        # `abs_features` / `smooth_abs_features`) preserve raw NaNs
        # untouched. Coerce NaN -> null before pooling so a single
        # NaN in any session does not poison the global statistics
        # for that feature.
        base_ds_for_stats = base_ds.fill_nan(None) if base_ds.dtype.is_float() else base_ds
        global_mean = base_ds_for_stats.mean()
        global_std = base_ds_for_stats.std()

        # Handle case of zero std (constant data)
        if global_std is None or global_std == 0:
            global_std = 1.0  # Avoid division by zero
            if global_mean is None:
                global_mean = 0.0  # Avoid (null - 0) / 1

        # Apply z-score expression using global stats
        for one_beh_session, df in data_dict.items():
            cols_to_zscore = []
            for column in df.columns:
                if column.split('.')[-1] == one_feature:
                    # Apply z-score as a Polars expression
                    # This operates in-place and preserves height
                    cols_to_zscore.append(
                        ((pls.col(column) - global_mean) / global_std).alias(column)
                    )

            if cols_to_zscore:
                data_dict[one_beh_session] = df.with_columns(cols_to_zscore)

    return data_dict
