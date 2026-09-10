"""
@author: bartulem
Cohort selection: which units, and which of their sessions, each claim is allowed to use.

Builds the per-unit manifest the pipelines iterate over, from the canonical sources rather than
reinventions -- ``unit_catalog.csv`` for unit properties, and the session-list ``.txt`` files for the
courtship session set.

A unit is kept when its area, quality and somatic flag pass and it appears in at least
``min_courtship_sessions`` courtship sessions. A ``unit_uid`` is single-``rec_date``, so its "sessions" are
same-day replicate blocks and the cross-validation is within-day by construction.

TWO SESSION SETS COME BACK PER UNIT, AND THE DIFFERENCE IS THE POINT.

``courtship_sessions`` is every courtship session the cluster was recorded in. ``vocal_sessions`` is
the subset carrying at least ``min_emitter_usvs_per_session`` calls from the emitter
``anchors.vocal_emitter`` names -- the recorded animal by default.

They differ because the claims need different things from a session. The encoding claim is fitted on
QUIET anchors, and a session in which the male barely called is not impoverished for that purpose --
it has MORE silence, not less. Everything scored on vocalizations is the opposite: a session with two
calls contributes a leave-one-session-out fold whose statistic rests on two events.

That is not a hypothetical. Across the 73 courtship sessions the median is ~508 focal calls but the
minimum is 2, and two sessions with 2 calls apiece are recorded alongside 116 and 67 cohort units.
Measured on a pilot unit by subsampling one session and refitting 12 times, the fold's decode gain
holds a standard deviation of 0.019 at 200 events and 0.130 at 2 -- a sevenfold spread, ranging from
strongly anti-aligned to five times the true effect. The exact permutation null degrades in the same
place for a sharper reason: a within-session permutation of n events has n! orderings and the
identity is one of them, so at n = 2 half of all "null" draws leave that session exactly where it
started, against 2.8e-7 at n = 10.

The threshold is 20 (user-ruled 2026-09-08). It sits past the measured variance knee -- the fold-gain
spread halves between 10 and 20 and then stops improving -- clears the permutation floor with a
factor of two in hand, and costs almost nothing: the sessions it removes are the ones with no data in
them, so 99.8% of units and 99.9% of all focal calls survive. Ten and twenty select the IDENTICAL
cohort, there being a gap in the distribution between them, which is the property to want in a number
that is about to be frozen.

The count is deliberately the RAW focal total rather than the cleaned or torus-positioned one, so
that cohort membership does not move when an analysis window or an embedding is changed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl

from ..os_utils import configure_path
from .neural_design_assembly import emitter_names, load_session_usvs, session_timebase


def load_courtship_session_ids(session_list_files: list[str]) -> set[str]:
    """
    Description
    -----------
    Read the session-list ``.txt`` files and return the set of courtship session IDs (the session
    directory basenames), resolving each stored path through ``configure_path``.

    Mirrors the modeling pipeline's ``prepare_modeling_sessions`` but unions over MULTIPLE lists,
    since the cohort spans the intact-partner and mute-female conditions by default.

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

    Every unit in a session belongs to the same animal, so the mapping is well defined -- and taking
    it from the catalog avoids the slot assumption that "the recorded animal is track_names[0]". That
    assumption holds in this cohort (73 of 73 session-mouse pairs) but it is a property of the
    recordings, not a guarantee, and it is the kind of thing that fails silently.

    Parameters
    ----------
    catalog_path (str)
        Path to ``unit_catalog.csv``.

    Returns
    -------
    mapping (dict)
        ``{session_id: mouse_id}``.
    """

    catalog = pl.read_csv(catalog_path, infer_schema_length=5000)
    mapping: dict = {}
    for row in catalog.iter_rows(named=True):
        for session in (ast.literal_eval(row["rec_sessions"]) if row["rec_sessions"] else []):
            mapping[session] = str(row["mouse_id"])
    return mapping


def emitter_usv_counts(session_ids, data_root: str, mouse_by_session: dict,
                     vocal_emitter: str = "self") -> dict:
    """
    Description
    -----------
    Count the recorded animal's own vocalizations in each session.

    This is a property of the SESSION, not of any unit, so it is computed once and shared across the
    whole cohort rather than recomputed per unit. The recorded animal is resolved by NAME from the
    session's ``track_names``, the same mechanism the design assembler uses, and calls are matched on
    the emitter label exactly.

    A session whose tracking or summary file is missing counts as zero rather than raising: cohort
    building sees every listed session, including ones that were never processed, and a missing file
    is a reason to exclude a session from vocal analyses rather than to abort the manifest.

    Parameters
    ----------
    session_ids (Iterable[str])
        Session-directory basenames.
    data_root (str)
        The ``Data`` root.
    mouse_by_session (dict)
        ``{session_id: mouse_id}`` from :func:`recorded_mouse_by_session`.
    vocal_emitter (str)
        Whose calls are the events: ``'self'`` | ``'partner'`` | ``'all'``.

    Returns
    -------
    counts (dict)
        ``{session_id: n_focal_usvs}``.
    """

    counts: dict = {}
    for session_id in sorted(session_ids):
        try:
            track_names, _frame_rate, _n_frames = session_timebase(data_root, session_id)
            usv = load_session_usvs(data_root, session_id)
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
    least ``min_emitter_usvs_per_session`` calls from the selected emitter, and ``vocal_testable`` records whether enough of
    them survive for a leave-one-session-out rotation on the vocal side.

    A unit that fails the vocal floor is KEPT, not dropped: it still has an encoding claim to answer,
    and dropping it here would silently shrink the denominator the transformation fraction is reported
    against. The flag says what it can be asked, and the counts say why.

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
        Focal calls a session needs before anything scored on vocalizations may use it.
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

    catalog = pl.read_csv(catalog_path, infer_schema_length=5000)
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
    the focal calls per session on the way.

    Parameters
    ----------
    settings (dict)
        The whole neural-modeling settings dict; the ``cohort``, ``anchors`` and ``data_roots``
        blocks are read.

    Returns
    -------
    cohort (list[dict])
        The manifest, one record per unit.
    """

    cohort_settings = settings["cohort"]
    session_ids = load_courtship_session_ids(cohort_settings["session_list_files"])
    catalog_path = settings["data_roots"]["catalog_path"]
    mouse_by_session = recorded_mouse_by_session(catalog_path)
    counts = emitter_usv_counts(session_ids & set(mouse_by_session),
                              settings["data_roots"]["data_root"], mouse_by_session,
                              settings["anchors"]["vocal_emitter"])
    return select_cohort(
        catalog_path=catalog_path,
        courtship_session_ids=session_ids,
        brain_areas=cohort_settings["brain_areas"],
        area_mode=cohort_settings["area_mode"],
        quality=cohort_settings["quality"],
        somatic=cohort_settings["somatic"],
        min_courtship_sessions=cohort_settings["min_courtship_sessions"],
        emitter_counts=counts,
        min_emitter_usvs_per_session=cohort_settings["min_emitter_usvs_per_session"],
        min_vocal_sessions=cohort_settings["min_vocal_sessions"],
    )


def summarize_cohort(cohort: list[dict]) -> dict:
    """
    Description
    -----------
    Counts worth printing before a cohort run: how many units there are, how many can be asked the
    vocal questions, and how much the session gate actually removed.

    Reported rather than merely applied, because a gate whose effect is not stated is a silent
    change to the denominator every fraction in the paper is quoted against.

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
