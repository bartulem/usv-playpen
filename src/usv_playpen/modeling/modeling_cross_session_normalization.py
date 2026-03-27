"""
@author: bartulem
Pools data from different sessions and normalizes them together.
"""

import polars as pls

def cap_series_values_to_nan(series: pls.Series,
                             theoretical_min: float,
                             theoretical_max: float) -> pls.Series:
    """
    Replaces values in a Polars Series that are outside theoretical min/max
    bounds with NaN, returning a Series.

    Args:
        series: The input Polars Series.
        theoretical_min: The minimum allowed value.
        theoretical_max: The maximum allowed value.

    Returns:
        A new Polars Series with values outside bounds replaced by NaN.
    """

    temp_df = series.to_frame(name=series.name)

    capped_df = temp_df.with_columns(
        pls.when((pls.col(series.name) >= theoretical_min) & (pls.col(series.name) <= theoretical_max))
        .then(pls.col(series.name))
        .otherwise(pls.lit(None))
        .alias(series.name)
    )

    return capped_df[series.name]


def zscore_different_sessions_together(data_dict: dict = None,
                                       feature_lst: list = None,
                                       feature_bounds: dict = None) -> dict:
    """
    Computes z-scored behavioral data when different sessions need to
    be pooled together (z-scores are computed on the pooled data,
    instead of on each session separately).

    Parameters
    ----------
    data_dict : dict
        Original behavioral feature data for each session.
    feature_lst : list
        Relevant behavioral features.
    feature_bounds : dict
        Dictionary with theoretical boundaries
        for each feature (such that outliers can be excluded).

    Returns
    -------
    data_dict : dict
        Behavioral data (z-scored).
    """

    # Clean the data *in place* in the DataFrames
    for one_beh_session in data_dict.keys():
        cols_to_update = []
        for column in data_dict[one_beh_session].columns:
            base_feature = column.split('.')[-1]
            if base_feature not in feature_lst:
                continue

            if base_feature in ['allo_roll',
                                'allo_yaw-nose', 'nose-allo_yaw',
                                'allo_yaw-TTI', 'TTI-allo_yaw']:
                cols_to_update.append(pls.col(column).abs().alias(column))
            else:
                if 'usv_' not in base_feature:
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
    for feature_idx, one_feature in enumerate(feature_lst):
        pooled_series_list = []

        for one_beh_session in data_dict.keys():
            for column in data_dict[one_beh_session].columns:
                if column.split('.')[-1] == one_feature:
                    pooled_series_list.append(data_dict[one_beh_session][column].cast(pls.Float32))
                    break

        if not pooled_series_list:
            continue

        base_ds = pls.concat(pooled_series_list, how='vertical')

        # Calculate global mean and std, (Polars ignores nulls by default)
        global_mean = base_ds.mean()
        global_std = base_ds.std()

        # Handle case of zero std (constant data)
        if global_std is None or global_std == 0:
            global_std = 1.0  # Avoid division by zero
            if global_mean is None:
                global_mean = 0.0  # Avoid (null - 0) / 1

        # Apply z-score expression using global stats
        for one_beh_session in data_dict.keys():
            cols_to_zscore = []
            for column in data_dict[one_beh_session].columns:
                if column.split('.')[-1] == one_feature:
                    # Apply z-score as a Polars expression
                    # This operates in-place and preserves height
                    cols_to_zscore.append(
                        ((pls.col(column) - global_mean) / global_std).alias(column)
                    )

            if cols_to_zscore:
                data_dict[one_beh_session] = data_dict[one_beh_session].with_columns(cols_to_zscore)

    return data_dict
