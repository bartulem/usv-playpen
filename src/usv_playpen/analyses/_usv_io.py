"""
@author: bartulem
Shared USV/session input loaders used by both the analyses and the
visualizations layers.

These loaders previously lived in ``visualizations/usv_summary_statistics.py``
and were imported back into ``analyses/compute_inter_usv_interval_distributions``
-- an analyses->visualizations dependency that, together with
``visualizations/usv_interval_summary_statistics`` importing from
``compute_inter``, formed a near-cycle between the two layers. Hosting them here
lets both layers depend "downward" on ``analyses`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import polars as pls


def extract_session_metadata(session_root: str) -> dict[str, Any]:
    """
    Description
    -----------
    This method extracts core experimental metadata from a session directory, including
    animal identity strings (male_id, female_id), recording frame rate, and the experimental code.

    It searches for the metric H5 tracking file within the provided
    directory and extracts identity strings for the animals involved. It is
    specifically designed for social interaction sessions (male-female).

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory containing the .h5 tracking files.

    Returns
    -------
    metadata (dict)
        Contains 'male_id', 'female_id', 'frame_rate', 'experiment_code', and 'tracking_file'.
    """

    session_path = Path(session_root)
    tracking_file = next(iter(sorted(session_path.glob('**/*_points3d_translated_rotated_metric.h5'))), None)

    if tracking_file is None:
        msg = f"No tracking file found in {session_root}"
        raise FileNotFoundError(msg)

    with h5py.File(name=str(tracking_file), mode='r') as h5_file:
        track_names = [item.decode('utf-8') for item in list(h5_file['track_names'])]
        if len(track_names) < 2:
            msg = f"Session {session_root} does not contain two animal tracks."
            raise IndexError(msg)

        return {
            'male_id': track_names[0],
            'female_id': track_names[1],
            'frame_rate': float(h5_file['recording_frame_rate'][()]),
            'experiment_code': h5_file['experimental_code'][()].decode("utf-8"),
            'tracking_file': tracking_file
        }

def load_and_filter_usv_data(
    session_root: str,
    frame_rate: float,
    noise_col_id: str,
    noise_categories: list[int]
) -> pls.DataFrame:
    """
    Description
    -----------
    This method loads USV summary CSV data using Polars and appends calculated frame
    indices based on the provided recording frame rate.

    The function filters the entire dataset to remove noise based on the provided
    noise column and a list of noise categories. The remaining valid vocalizations
    (male, female, and unassigned) are retained and returned.

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory.
    frame_rate (float)
        The sampling rate of the video recording used to synchronize USVs with behavioral frames.
    noise_col_id (str)
        The name of the column in the CSV that dictates the noise classification.
    noise_categories (list[int])
        A list of specific integer values in the noise column that identify a row as noise to be excluded.

    Returns
    -------
    usv_info (pls.DataFrame)
        All columns from the USV summary CSV with noise rows removed, plus a newly
        calculated 'frame_index' column.
    """

    session_path = Path(session_root)
    usv_file = next(iter(sorted(session_path.glob('**/*_usv_summary.csv'))), None)

    if usv_file is None:
        msg = f"USV summary file missing in {session_root}"
        raise FileNotFoundError(msg)

    usv_info = pls.read_csv(str(usv_file))

    # Remove noise across all categories provided in the list; rows whose noise value is
    # null are not in noise_categories, so fill_null(True) retains them rather than letting
    # the three-valued (~null -> null) logic silently drop them via filter()
    usv_info_clean = usv_info.filter(
        pls.col(noise_col_id).is_in(noise_categories).not_().fill_null(True)
    )

    return usv_info_clean.with_columns(
        (pls.col("start") * frame_rate).floor().cast(pls.UInt32).alias("frame_index")
    )
