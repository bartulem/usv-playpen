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

from ..os_utils import drop_noise_usvs


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
    exclude_noise_usvs: bool,
    call_type: str | None = None
) -> pls.DataFrame:
    """
    Description
    -----------
    This method loads USV summary CSV data using Polars and appends calculated frame
    indices based on the provided recording frame rate.

    When ``exclude_noise_usvs`` is set, the segments ``detect_usv_noise`` flagged as holding no
    vocalization are dropped (:func:`os_utils.drop_noise_usvs`), leaving the valid vocalizations
    (male, female and unassigned). A session whose summary has no ``noise`` column raises there, so a
    missing classification can never be mistaken for a clean session.

    ``call_type`` selects among the vocalizations that survive. Dropping noise leaves BOTH
    ultrasonic calls and squeaks, and the two are different vocalizations: a squeak is a broadband
    call with a 3-8 kHz fundamental, detected by ``detect_usv_squeaks``, while a USV is ultrasonic.
    Callers that want one and not the other must say so, because the union is rarely what an
    analysis means. In the cohort the distinction is large -- 7.6% of the male's segments and 48.0%
    of the female's are squeaks -- and treating them as one class puts a squeak between two
    ultrasonic calls, which suppresses the long interval those calls would have formed and
    contributes two short ones in its place.

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory.
    frame_rate (float)
        The sampling rate of the video recording used to synchronize USVs with behavioral frames.
    exclude_noise_usvs (bool)
        Whether to drop the segments flagged as noise.
    call_type (str or None)
        Which vocalizations to keep: ``'usv'`` for ultrasonic calls only (squeaks dropped),
        ``'squeak'`` for squeaks only, or None to keep both. Defaults to None, which preserves
        the behaviour of every caller written before the squeak classifier existed.

    Returns
    -------
    usv_info (pls.DataFrame)
        All columns from the USV summary CSV with noise rows removed, restricted to ``call_type``,
        plus a newly calculated 'frame_index' column.
    """

    session_path = Path(session_root)
    usv_file = next(iter(sorted(session_path.glob('**/*_usv_summary.csv'))), None)

    if usv_file is None:
        msg = f"USV summary file missing in {session_root}"
        raise FileNotFoundError(msg)

    usv_info = pls.read_csv(str(usv_file))

    usv_info_clean = drop_noise_usvs(usv_info, usv_file.name)[0] if exclude_noise_usvs else usv_info

    if call_type is not None:
        if call_type not in ("usv", "squeak"):
            msg = f"load_and_filter_usv_data: call_type must be 'usv', 'squeak' or None, got {call_type!r}."
            raise ValueError(msg)
        if "squeak" not in usv_info_clean.columns:
            msg = (f"{usv_file.name} has no 'squeak' column, so call_type={call_type!r} cannot be "
                   "honoured; run detect_usv_squeaks on this session first.")
            raise KeyError(msg)
        usv_info_clean = usv_info_clean.filter(
            pls.col("squeak") if call_type == "squeak" else ~pls.col("squeak")
        )

    return usv_info_clean.with_columns(
        (pls.col("start") * frame_rate).floor().cast(pls.UInt32).alias("frame_index")
    )
