"""
@author: bartulem
Cohort selection: which units, and which of their sessions, each analysis is allowed to use.

Builds the per-unit manifest the pipelines iterate over, from ``unit_catalog.csv`` for unit
properties and the session-list ``.txt`` files for the courtship session set.

A unit is kept when its area, quality and somatic flag pass and it appears in at least
``min_courtship_sessions`` courtship sessions. Each kept unit carries two session sets:

- ``courtship_sessions`` — every courtship session the unit was recorded in, used by the kinematic
  encoding, which is fitted on silence.
- ``vocal_sessions``     — the subset with at least ``min_emitter_usvs_per_session`` analysed USVs
  (noise and squeaks excluded) from the emitter ``vocalization_settings.vocal_emitter`` names, used
  by everything scored on vocalizations.

A ``unit_uid`` is single-``rec_date``, so its sessions are same-day blocks and every
leave-one-session-out rotation is within-day.

- `load_courtship_session_ids`  — session IDs from the session-list files.
- `recorded_mouse_by_session`   — the recorded mouse per session, from the catalog.
- `emitter_usv_counts`          — analysed emitter USVs per session.
- `select_cohort`               — the unit filters and both session sets.
- `build_cohort_manifest`       — the whole manifest from a settings dict.
- `summarize_cohort`            — counts to print before a cohort run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pls

from ..os_utils import configure_path
from .neural_design_assembly import (
    emitter_names,
    filter_vocalizations,
    load_session_usvs,
    session_timebase,
)


def load_courtship_session_ids(session_list_files: list[str]) -> set[str]:
    """
    Description
    -----------
    Read the session-list ``.txt`` files and return the set of courtship session IDs (the session
    directory basenames), resolving each stored path through ``configure_path``.

    Unions over several lists, since the cohort spans the intact-partner and mute-female conditions.

    Parameters
    ----------
    session_list_files (list[str])
        Full paths to the session-list ``.txt`` files.

    Returns
    -------
    session_ids (set[str])
        Session-directory basenames.
    """

    session_ids: set[str] = set()
    for list_file in session_list_files:
        with Path(configure_path(list_file)).open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    session_ids.add(Path(configure_path(line)).name)
    return session_ids


def _area_pass(brain_area: str, brain_areas: list[str], area_mode: str) -> bool:
    """
    Description
    -----------
    Apply the brain-area scheme: ``'include'`` keeps the listed areas, ``'exclude'`` keeps everything
    else (the pooled non-target contrast), ``'all'`` keeps every area.

    Parameters
    ----------
    brain_area (str)
        The unit's catalog ``brain_area``.
    brain_areas (list[str])
        Named areas.
    area_mode (str)
        ``'include'`` | ``'exclude'`` | ``'all'``.

    Returns
    -------
    keep (bool)
        Whether the area passes.
    """

    if area_mode == "all":
        return True
    if area_mode == "include":
        return brain_area in brain_areas
    if area_mode == "exclude":
        return brain_area not in brain_areas
    msg = f"area_mode must be 'include' | 'exclude' | 'all'; got {area_mode!r}."
    raise ValueError(msg)


def recorded_mouse_by_session(catalog_path: str) -> dict:
    """
    Description
    -----------
    Which mouse each session was recorded from, read off the catalog rather than assumed.

    Every unit in a session belongs to the same animal, so the mapping is well defined, and reading it
    from the catalog avoids assuming the recorded animal's track slot.

    Parameters
    ----------
    catalog_path (str)
        Path to ``unit_catalog.csv``.

    Returns
    -------
    mapping (dict)
        ``{session_id: mouse_id}``.
    """

    catalog = pls.read_csv(catalog_path, infer_schema_length=5000)
    mapping: dict = {}
    for row in catalog.iter_rows(named=True):
        for session in (ast.literal_eval(row["rec_sessions"]) if row["rec_sessions"] else []):
            mapping[session] = str(row["mouse_id"])
    return mapping


def emitter_usv_counts(session_ids, data_root: str, mouse_by_session: dict,
                     vocal_emitter: str, vocal_settings: dict) -> dict:
    """
    Description
    -----------
    Count the recorded animal's own vocalizations in each session.

    Computed once per session and shared across the cohort. The recorded animal is resolved by name
    from the session's ``track_names``, and only analysed USVs (noise and squeaks excluded) are
    counted, so a session cannot qualify on events no analysis scores. A session whose tracking or
    summary file is missing counts as zero rather than raising.

    Parameters
    ----------
    session_ids (Iterable[str])
        Session-directory basenames.
    data_root (str)
        The ``Data`` root.
    mouse_by_session (dict)
        ``{session_id: mouse_id}`` from :func:`recorded_mouse_by_session`.
    vocal_emitter (str)
        Whose USVs are the events: ``'self'`` | ``'partner'`` | ``'all'``.
    vocal_settings (dict)
        The ``vocalization_settings`` block, which defines the analysed USVs.

    Returns
    -------
    counts (dict)
        ``{session_id: n_focal_usvs}``.
    """

    counts: dict = {}
    for session_id in sorted(session_ids):
        try:
            track_names, _frame_rate, _n_frames = session_timebase(data_root, session_id)
            # Count only the USVs the analyses score (noise and squeaks excluded).
            usv = filter_vocalizations(load_session_usvs(data_root, session_id), vocal_settings,
                                       "analysed")
        except FileNotFoundError:
            counts[session_id] = 0
            continue
        names = emitter_names(track_names, mouse_by_session[session_id], vocal_emitter)
        counts[session_id] = (int(usv["emitter"].is_in(names).sum())
                              if usv.height and names else usv.height)
    return counts


def select_cohort(
        catalog_path: str,
        courtship_session_ids: set[str],
        brain_areas: list[str],
        area_mode: str,
        quality: list[str],
        somatic: str,
        min_courtship_sessions: int,
        emitter_counts: dict,
        min_emitter_usvs_per_session: int,
        min_vocal_sessions: int,
) -> list[dict]:
    """
    Description
    -----------
    Filter ``unit_catalog.csv`` into the per-unit cohort, annotating each unit with BOTH session sets.

    Unit-level filters (area, quality, somatic) run first, then the unit must appear in at least
    ``min_courtship_sessions`` courtship sessions. ``vocal_sessions`` is then the subset of those carrying at
    least ``min_emitter_usvs_per_session`` USVs from the selected emitter, and ``vocal_testable`` records whether enough of
    them survive for a leave-one-session-out rotation on the vocal side.

    A unit that fails the vocal floor is kept, flagged ``vocal_testable = False``, so it can still be
    tested for kinematic encoding and every cohort fraction keeps the same denominator.

    Parameters
    ----------
    catalog_path (str)
        Path to ``unit_catalog.csv``.
    courtship_session_ids (set[str])
        Courtship session IDs.
    brain_areas (list[str])
        Named brain areas.
    area_mode (str)
        ``'include'`` | ``'exclude'`` | ``'all'``.
    quality (list[str])
        Allowed ``cluster_group`` labels.
    somatic (str)
        ``'somatic'`` | ``'non_somatic'`` | ``'both'``.
    min_courtship_sessions (int)
        Minimum courtship sessions the unit must appear in.
    emitter_counts (dict)
        ``{session_id: n_usvs}`` from :func:`emitter_usv_counts`.
    min_emitter_usvs_per_session (int)
        Focal USVs a session needs before anything scored on vocalizations may use it.
    min_vocal_sessions (int)
        Vocal-eligible sessions needed for the vocal-side rotation.

    Returns
    -------
    cohort (list[dict])
        One record per kept unit, carrying ``courtship_sessions`` and ``vocal_sessions`` alongside
        the unit's catalog properties.
    """

    if somatic not in ("somatic", "non_somatic", "both"):
        msg = f"somatic must be 'somatic' | 'non_somatic' | 'both'; got {somatic!r}."
        raise ValueError(msg)

    catalog = pls.read_csv(catalog_path, infer_schema_length=5000)
    cohort: list[dict] = []
    for row in catalog.iter_rows(named=True):
        if not _area_pass(row["brain_area"], brain_areas, area_mode):
            continue
        if row["cluster_group"] not in quality:
            continue
        is_somatic = str(row["somatic"]).strip().lower() == "true"
        if somatic == "somatic" and not is_somatic:
            continue
        if somatic == "non_somatic" and is_somatic:
            continue

        rec_sessions = ast.literal_eval(row["rec_sessions"]) if row["rec_sessions"] else []
        courtship_sessions = sorted(set(rec_sessions) & courtship_session_ids)
        if len(courtship_sessions) < min_courtship_sessions:
            continue

        vocal_sessions = [s for s in courtship_sessions
                          if emitter_counts[s] >= min_emitter_usvs_per_session]
        cohort.append({
            "unit_uid": f"{row['mouse_id']}_{row['rec_date']}_{row['unit_id']}",
            "mouse_id": str(row["mouse_id"]),
            "rec_date": int(row["rec_date"]),
            "unit_id": row["unit_id"],
            "brain_area": row["brain_area"],
            "cluster_group": row["cluster_group"],
            "somatic": is_somatic,
            "courtship_sessions": courtship_sessions,
            "n_courtship_sessions": len(courtship_sessions),
            "vocal_sessions": vocal_sessions,
            "n_vocal_sessions": len(vocal_sessions),
            "vocal_testable": len(vocal_sessions) >= min_vocal_sessions,
            "emitter_usvs_per_session": {s: emitter_counts[s] for s in courtship_sessions},
        })
    return cohort


def build_cohort_manifest(settings: dict) -> list[dict]:
    """
    Description
    -----------
    Build the cohort manifest from a settings dictionary, resolving the session lists and counting
    the focal USVs per session on the way.

    Parameters
    ----------
    settings (dict)
        The whole neural-modeling settings dict; the ``data``, ``data_sufficiency``,
        ``vocalization_settings`` and ``data_roots``
        blocks are read.

    Returns
    -------
    cohort (list[dict])
        The manifest, one record per unit.
    """

    # `data` says which data the run draws on, `data_sufficiency` how much of it a unit must carry.
    data_settings = settings["data"]
    sufficiency = settings["data_sufficiency"]
    session_ids = load_courtship_session_ids(data_settings["session_list_files"])
    catalog_path = settings["data_roots"]["catalog_path"]
    mouse_by_session = recorded_mouse_by_session(catalog_path)
    counts = emitter_usv_counts(session_ids & set(mouse_by_session),
                                settings["data_roots"]["data_root"], mouse_by_session,
                                settings["vocalization_settings"]["vocal_emitter"],
                                settings["vocalization_settings"])
    return select_cohort(
        catalog_path=catalog_path,
        courtship_session_ids=session_ids,
        brain_areas=data_settings["brain_areas"],
        area_mode=data_settings["area_mode"],
        quality=data_settings["quality"],
        somatic=data_settings["somatic"],
        min_courtship_sessions=sufficiency["min_courtship_sessions"],
        emitter_counts=counts,
        min_emitter_usvs_per_session=sufficiency["min_emitter_usvs_per_session"],
        min_vocal_sessions=sufficiency["min_vocal_sessions"],
    )


def summarize_cohort(cohort: list[dict]) -> dict:
    """
    Description
    -----------
    Counts worth printing before a cohort run: how many units there are, how many can be asked the
    vocal questions, and how much the session gate actually removed.

    Reported so that the effect of the session floor on the cohort denominator is visible.

    Parameters
    ----------
    cohort (list[dict])
        The manifest.

    Returns
    -------
    summary (dict)
        ``n_units``, ``n_vocal_testable``, ``n_units_losing_a_session``, ``session_slots``,
        ``vocal_session_slots``, ``focal_usvs_total``, ``focal_usvs_retained``.
    """

    slots = sum(u["n_courtship_sessions"] for u in cohort)
    vocal_slots = sum(u["n_vocal_sessions"] for u in cohort)
    total = sum(sum(u["emitter_usvs_per_session"].values()) for u in cohort)
    retained = sum(u["emitter_usvs_per_session"][s] for u in cohort for s in u["vocal_sessions"])
    return {
        "n_units": len(cohort),
        "n_vocal_testable": sum(1 for u in cohort if u["vocal_testable"]),
        "n_units_losing_a_session": sum(1 for u in cohort
                                        if u["n_vocal_sessions"] < u["n_courtship_sessions"]),
        "session_slots": slots,
        "vocal_session_slots": vocal_slots,
        "focal_usvs_total": int(total),
        "focal_usvs_retained": int(retained),
    }
