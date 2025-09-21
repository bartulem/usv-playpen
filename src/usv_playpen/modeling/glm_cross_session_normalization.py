"""
@author: bartulem
Pools data from different sessions and normalizes them together.
"""

import numpy as np
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
        .otherwise(pls.lit(np.nan))
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

    for feature_idx, one_feature in enumerate(feature_lst):
        demarcation_indices = {'start_indices':[], 'end_indices': []}
        for session_idx, one_beh_session in enumerate(data_dict.keys()):
            for column in data_dict[one_beh_session].columns:
                if column.split('.')[-1] == one_feature:
                    if session_idx == 0:
                        if column.split('.')[-1] in ['allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']:
                            base_ds = data_dict[one_beh_session][column].abs()
                        else:
                            theoretical_min, theoretical_max = feature_bounds[column.split('.')[-1]]
                            base_ds = cap_series_values_to_nan(series=data_dict[one_beh_session][column],
                                                               theoretical_min=theoretical_min,
                                                               theoretical_max=theoretical_max)

                        demarcation_indices['start_indices'].append(0)
                        demarcation_indices['end_indices'].append(data_dict[one_beh_session][column].shape[0])
                    else:
                        demarcation_indices['start_indices'].append(base_ds.shape[0])
                        if column.split('.')[-1] in ['allo_yaw-nose', 'nose-allo_yaw', 'allo_yaw-TTI', 'TTI-allo_yaw']:
                            base_ds = pls.concat(items=[base_ds, data_dict[one_beh_session][column].abs()], how='vertical')
                        else:
                            theoretical_min, theoretical_max = feature_bounds[column.split('.')[-1]]
                            base_ds_added = cap_series_values_to_nan(series=data_dict[one_beh_session][column],
                                                                     theoretical_min=theoretical_min,
                                                                     theoretical_max=theoretical_max)
                            base_ds = pls.concat(items=[base_ds, base_ds_added], how='vertical')
                        demarcation_indices['end_indices'].append(base_ds.shape[0])

        z_scores_series = (base_ds - base_ds.fill_nan(None).mean()) / base_ds.fill_nan(None).std()

        for session_idx, one_beh_session in enumerate(data_dict.keys()):
            for column in data_dict[one_beh_session].columns:
                if column.split('.')[-1] == one_feature:
                    start_idx = demarcation_indices['start_indices'][session_idx]
                    end_idx = demarcation_indices['end_indices'][session_idx]
                    data_dict[one_beh_session] = data_dict[one_beh_session].with_columns(z_scores_series[start_idx:end_idx].alias(column))
                    break

    return data_dict
