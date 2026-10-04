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

from ..os_utils import call_class_mask, drop_noise_usvs
from ..yaml_utils import extract_animal_sexes

__all__ = [
    'emitter_sex_expression',
    'extract_animal_sexes',
    'extract_session_metadata',
    'load_and_filter_usv_data',
    'sex_track_ids',
]


def extract_session_metadata(session_root: str) -> dict[str, Any]:
    """
    Description
    -----------
    This method extracts core experimental metadata from a session directory, including
    the animal identity strings (the track names), recording frame rate, and the experimental code.

    It searches for the metric H5 tracking file within the provided
    directory and extracts identity strings for the animals involved. It is
    designed for social interaction sessions (two or more tracked animals).

    The track names are returned in file order but carry no sex: the slot a track
    occupies says nothing about the animal's sex (track 0 is a female in a
    female-female session), so callers that need sexes resolve them from the session
    metadata with :func:`extract_animal_sexes`. Names are stripped of null-byte padding
    and whitespace, so they compare equal to the metadata ``subject_id`` and the USV
    summary ``emitter`` strings.

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory containing the .h5 tracking files.

    Returns
    -------
    metadata (dict)
        Contains 'track_names' (list of stripped str, file order), 'frame_rate',
        'experiment_code', and 'tracking_file'.

    Raises
    ------
    FileNotFoundError
        No tracking file in the session.
    IndexError
        The tracking file holds fewer than two animal tracks.
    """

    session_path = Path(session_root)
    tracking_file = next(iter(sorted(session_path.glob('**/*_points3d_translated_rotated_metric.h5'))), None)

    if tracking_file is None:
        msg = f"No tracking file found in {session_root}"
        raise FileNotFoundError(msg)

    with h5py.File(name=str(tracking_file), mode='r') as h5_file:
        track_names = [item.decode('utf-8').strip('\x00').strip() for item in list(h5_file['track_names'])]
        if len(track_names) < 2:
            msg = f"Session {session_root} does not contain two animal tracks."
            raise IndexError(msg)

        return {
            'track_names': track_names,
            'frame_rate': float(h5_file['recording_frame_rate'][()]),
            'experiment_code': h5_file['experimental_code'][()].decode("utf-8"),
            'tracking_file': tracking_file
        }


def emitter_sex_expression(animal_sex: dict[str, str], emitter_column: str = 'emitter') -> pls.Expr:
    """
    Description
    -----------
    Builds the polars expression that maps a USV summary's emitter column to the
    emitter's sex, with the sexes taken from the session metadata
    (:func:`extract_animal_sexes`), never from the order of the tracks.

    The emitter is cast to string and stripped of null bytes and whitespace before it is
    compared, matching the stripping :func:`extract_animal_sexes` applies to the track
    names, so a padded name on either side cannot silently send a call to
    ``'unassigned'``. A row whose emitter is not one of the resolved animals -- an
    empty / null emitter, i.e. a call Vocalocator could not attribute -- is
    ``'unassigned'``; that is a property of the call, not a failed sex lookup (an animal
    whose sex cannot be resolved has already raised in :func:`extract_animal_sexes`).

    Parameters
    ----------
    animal_sex (dict)
        ``{stripped track name: 'male' | 'female'}`` from :func:`extract_animal_sexes`.
    emitter_column (str)
        Name of the emitter column; defaults to ``'emitter'``.

    Returns
    -------
    sex_expression (pls.Expr)
        A string expression aliased ``'sex'`` holding ``'male'``, ``'female'`` or
        ``'unassigned'`` for every row.
    """

    emitter_norm = pls.col(emitter_column).cast(pls.Utf8).str.strip_chars('\x00').str.strip_chars()
    if not animal_sex:
        # No resolved animal: every row is unassigned. The always-false condition keeps the
        # expression tied to the emitter column, so it has one value per row in a select too.
        never = emitter_norm.is_null() & emitter_norm.is_not_null()
        return pls.when(never).then(pls.lit('unassigned')).otherwise(pls.lit('unassigned')).alias('sex')
    names = list(animal_sex)
    chain = pls.when(emitter_norm == names[0]).then(pls.lit(animal_sex[names[0]]))
    for name in names[1:]:
        chain = chain.when(emitter_norm == name).then(pls.lit(animal_sex[name]))
    return chain.otherwise(pls.lit('unassigned')).alias('sex')


def sex_track_ids(animal_sex: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Description
    -----------
    Names the male and the female of a session from its metadata-resolved sexes.

    Some outputs carry one ``male_id`` and one ``female_id`` per session (the master USV
    table, the courtship playback repository). Those identities are only defined when
    the session holds exactly one animal of that sex; this returns the track of the
    single male and the track of the single female, and ``None`` for a sex that has no
    animal or more than one (a female-female session has no ``male_id`` and no single
    ``female_id``). Nothing is inferred from the track order.

    Parameters
    ----------
    animal_sex (dict)
        ``{stripped track name: 'male' | 'female'}`` from :func:`extract_animal_sexes`.

    Returns
    -------
    male_id, female_id (tuple of str or None)
        The single male's and the single female's track names, each ``None`` when the
        session does not hold exactly one animal of that sex.
    """

    males = [name for name, sex in animal_sex.items() if sex == 'male']
    females = [name for name, sex in animal_sex.items() if sex == 'female']
    return (males[0] if len(males) == 1 else None,
            females[0] if len(females) == 1 else None)


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

    ``call_type`` selects among the vocalizations that survive. Dropping noise leaves ultrasonic
    calls AND squeaks, and the two are different vocalizations: a squeak is a broadband call with a
    3-8 kHz fundamental, while a USV is ultrasonic. ``detect_usv_squeaks`` writes two booleans for
    every non-noise segment, ``usv`` and ``squeak``: a pure USV is ``usv & ~squeak``, a pure squeak
    ``squeak & ~usv``, and a segment holding both has both true. Callers that want
    one kind and not the other must say so, because the union is rarely what an analysis means. In
    the cohort the distinction is large -- with the earlier binary squeak detector, 7.6% of the
    male's segments and 48.0% of the female's were squeaks -- and treating them as one class puts a
    squeak between two ultrasonic calls, which suppresses the long interval those calls would have
    formed and contributes two short ones in its place. A segment holding both is neither a clean
    USV nor a clean squeak, so it belongs to neither ``'usv'`` nor ``'squeak'`` and is removed from
    both sequences; null booleans (a segment too short to score) are likewise in neither.

    Parameters
    ----------
    session_root (str)
        The absolute path to the session directory.
    frame_rate (float)
        The sampling rate of the video recording used to synchronize USVs with behavioral frames.
    exclude_noise_usvs (bool)
        Whether to drop the segments flagged as noise.
    call_type (str or None)
        Which vocalizations to keep: ``'usv'`` for pure USVs only (``usv & ~squeak``),
        ``'squeak'`` for pure squeaks only (``squeak & ~usv``; segments holding both and null
        booleans are in neither), or None to keep every row that survives the noise filter.
        Defaults to None, which preserves the behaviour of every caller written before the call
        classifier existed.

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
        usv_info_clean = usv_info_clean.filter(call_class_mask(usv_info_clean, (call_type,), usv_file.name))

    return usv_info_clean.with_columns(
        (pls.col("start") * frame_rate).floor().cast(pls.UInt32).alias("frame_index")
    )
