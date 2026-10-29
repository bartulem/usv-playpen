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

A unit recorded in a single courtship session has no session to hold out, so that session is split
into ``data_sufficiency.single_session_blocks`` contiguous blocks of equal duration, separated by
``kinematic_encoding.history_pre_seconds``, and each block stands in for a session in both sets
(``<session>#block<i>``). The record's ``block_split`` says which regime a unit is tested in.

- `load_courtship_session_ids`  — session IDs from the session-list files.
- `recorded_mouse_by_session`   — the recorded mouse per session, from the catalog.
- `focal_usv_onsets`            — onsets of the analysed emitter USVs per session, with its timebase.
- `emitter_usv_counts`          — analysed emitter USVs per session.
- `block_usv_counts`            — analysed emitter USVs per block of each session.
- `unit_session_sets`           — a unit's two session sets, as sessions or as blocks.
- `select_cohort`               — the unit filters and both session sets.
- `build_cohort_manifest`       — the whole manifest from a settings dict.
- `summarize_cohort`            — counts to print before a cohort run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pls

from ..os_utils import configure_path
from .neural_design_assembly import (
    block_window_frames,
    emitter_names,
    filter_vocalizations,
    load_session_usvs,
    session_block_ids,
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

    catalog = pls.read_csv(configure_path(catalog_path), infer_schema_length=5000)
    mapping: dict = {}
    for row in catalog.iter_rows(named=True):
        for session in (ast.literal_eval(row["rec_sessions"]) if row["rec_sessions"] else []):
            mapping[session] = str(row["mouse_id"])
    return mapping


def focal_usv_onsets(session_ids, data_root: str, mouse_by_session: dict,
                     vocal_emitter: str, vocal_settings: dict) -> dict:
    """
    Description
    -----------
    The onsets of the analysed USVs from the selected emitter in each session, with the session's frame
    rate and frame count, so they can be counted per session or per block.

    The recorded animal is resolved by name from the session's ``track_names``, and only analysed USVs
    (noise and squeaks excluded) are kept, so a session cannot qualify on events no analysis scores. A
    session whose tracking or summary file is missing maps to ``None`` rather than raising.

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
    onsets (dict)
        ``{session_id: (onsets_seconds, frame_rate, n_frames) | None}``.
    """

    onsets: dict = {}
    for session_id in sorted(session_ids):
        try:
            track_names, frame_rate, n_frames = session_timebase(data_root, session_id)
            # Only the USVs the analyses score (noise and squeaks excluded).
            usv = filter_vocalizations(load_session_usvs(data_root, session_id), vocal_settings,
                                       "analysed")
        except FileNotFoundError:
            onsets[session_id] = None
            continue
        names = emitter_names(track_names, mouse_by_session[session_id], vocal_emitter)
        focal = usv.filter(usv["emitter"].is_in(names)) if usv.height and names else usv
        onsets[session_id] = (focal["start"].to_numpy().astype(np.float64), float(frame_rate), int(n_frames))
    return onsets


def block_usv_counts(onsets: dict, n_blocks: int, gap_seconds: float) -> dict:
    """
    Description
    -----------
    Count each session's analysed emitter USVs per block, with the blocks the block-tested units use
    (:func:`neural_design_assembly.block_window_frames`): a USV belongs to a block when its onset lies in
    the block's window, so USVs in the gaps between blocks count for none.

    Parameters
    ----------
    onsets (dict)
        From :func:`focal_usv_onsets`.
    n_blocks (int)
        Blocks per session (``data_sufficiency.single_session_blocks``).
    gap_seconds (float)
        Separation between neighbouring blocks (``kinematic_encoding.history_pre_seconds``).

    Returns
    -------
    counts (dict)
        ``{block_id: n_usvs}``; a session without files gives zero for each of its blocks.
    """

    counts: dict = {}
    for session_id, entry in onsets.items():
        block_ids = session_block_ids(session_id, n_blocks)
        if entry is None:
            counts.update(dict.fromkeys(block_ids, 0))
            continue
        starts, frame_rate, n_frames = entry
        for block_id, (low, high) in zip(block_ids, block_window_frames(n_frames, frame_rate, n_blocks,
                                                                         gap_seconds), strict=True):
            counts[block_id] = int(((starts >= low / frame_rate) & (starts < high / frame_rate)).sum())
    return counts


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

    onsets = focal_usv_onsets(session_ids, data_root, mouse_by_session, vocal_emitter, vocal_settings)
    return {session_id: (0 if entry is None else int(entry[0].size)) for session_id, entry in onsets.items()}


def unit_session_sets(sessions: list, emitter_counts: dict, n_blocks: int,
                      min_emitter_usvs_per_session: int) -> dict:
    """
    Description
    -----------
    A unit's two session sets. A unit recorded in two or more courtship sessions is tested across them; a
    unit recorded in one is tested across ``n_blocks`` contiguous blocks of it, which then stand in for
    sessions everywhere, including the vocal floor, which applies to each block.

    Parameters
    ----------
    sessions (list)
        The unit's courtship sessions, in time order.
    emitter_counts (dict)
        Analysed emitter USVs per session and, for a one-session unit, per block
        (:func:`emitter_usv_counts`, :func:`block_usv_counts`).
    n_blocks (int)
        Blocks a single session is split into (``data_sufficiency.single_session_blocks``).
    min_emitter_usvs_per_session (int)
        USVs a session or block needs before anything scored on vocalizations may use it.

    Returns
    -------
    sets (dict)
        ``courtship_sessions`` and ``vocal_sessions`` (session or block identifiers),
        ``block_split`` (``{"n_blocks": n_blocks}`` for a one-session unit, else ``None``) and
        ``emitter_usvs_per_session`` over the courtship set.
    """

    if len(sessions) == 1:
        courtship = session_block_ids(sessions[0], n_blocks)
        block_split = {"n_blocks": int(n_blocks)}
    else:
        courtship = list(sessions)
        block_split = None
    return {"courtship_sessions": courtship,
            "vocal_sessions": [s for s in courtship if emitter_counts[s] >= min_emitter_usvs_per_session],
            "block_split": block_split,
            "emitter_usvs_per_session": {s: emitter_counts[s] for s in courtship}}


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
        single_session_blocks: int,
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

    A unit recorded in a single courtship session is split into ``single_session_blocks`` blocks, which
    take the place of its sessions in both sets (:func:`unit_session_sets`).

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
        ``{session_id: n_usvs}`` from :func:`emitter_usv_counts`, merged with ``{block_id: n_usvs}`` from
        :func:`block_usv_counts`.
    min_emitter_usvs_per_session (int)
        Focal USVs a session (or block) needs before anything scored on vocalizations may use it.
    min_vocal_sessions (int)
        Vocal-eligible sessions (or blocks) needed for the vocal-side rotation.
    single_session_blocks (int)
        Blocks a one-session unit's session is split into.

    Returns
    -------
    cohort (list[dict])
        One record per kept unit, carrying ``courtship_sessions``, ``vocal_sessions`` and
        ``block_split`` alongside the unit's catalog properties. ``n_courtship_sessions`` counts recorded
        sessions, not blocks.
    """

    if somatic not in ("somatic", "non_somatic", "both"):
        msg = f"somatic must be 'somatic' | 'non_somatic' | 'both'; got {somatic!r}."
        raise ValueError(msg)

    catalog = pls.read_csv(configure_path(catalog_path), infer_schema_length=5000)
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

        sets = unit_session_sets(courtship_sessions, emitter_counts, single_session_blocks,
                                 min_emitter_usvs_per_session)
        cohort.append({
            "unit_uid": f"{row['mouse_id']}_{row['rec_date']}_{row['unit_id']}",
            "mouse_id": str(row["mouse_id"]),
            "rec_date": int(row["rec_date"]),
            "unit_id": row["unit_id"],
            "brain_area": row["brain_area"],
            "cluster_group": row["cluster_group"],
            "somatic": is_somatic,
            "courtship_sessions": sets["courtship_sessions"],
            "n_courtship_sessions": len(courtship_sessions),
            "block_split": sets["block_split"],
            "vocal_sessions": sets["vocal_sessions"],
            "n_vocal_sessions": len(sets["vocal_sessions"]),
            "vocal_testable": len(sets["vocal_sessions"]) >= min_vocal_sessions,
            "emitter_usvs_per_session": sets["emitter_usvs_per_session"],
        })
    return cohort


def build_cohort_manifest(settings: dict) -> list[dict]:
    """
    Description
    -----------
    Build the cohort manifest from a settings dictionary, resolving the session lists and counting
    the focal USVs per session, and per block for the units tested on blocks, on the way.

    Parameters
    ----------
    settings (dict)
        The whole neural-modeling settings dict; the ``data``, ``data_sufficiency``,
        ``vocalization_settings``, ``data_roots`` and ``kinematic_encoding`` (for the block gap)
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
    onsets = focal_usv_onsets(session_ids & set(mouse_by_session),
                              settings["data_roots"]["data_root"], mouse_by_session,
                              settings["vocalization_settings"]["vocal_emitter"],
                              settings["vocalization_settings"])
    # Per-block counts for every session, since which sessions hold a one-session unit is decided below.
    counts = {**{session_id: (0 if entry is None else int(entry[0].size)) for session_id, entry in onsets.items()},
              **block_usv_counts(onsets, sufficiency["single_session_blocks"],
                                 settings["kinematic_encoding"]["history_pre_seconds"])}
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
        single_session_blocks=sufficiency["single_session_blocks"],
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
        ``n_units``, ``n_block_units`` (tested on blocks of one session), ``n_vocal_testable``,
        ``n_units_losing_a_session``, ``session_slots``, ``vocal_session_slots``, ``focal_usvs_total``,
        ``focal_usvs_retained``. Slots count sessions, or blocks for a block-tested unit.
    """

    slots = sum(len(u["courtship_sessions"]) for u in cohort)
    vocal_slots = sum(u["n_vocal_sessions"] for u in cohort)
    total = sum(sum(u["emitter_usvs_per_session"].values()) for u in cohort)
    retained = sum(u["emitter_usvs_per_session"][s] for u in cohort for s in u["vocal_sessions"])
    return {
        "n_units": len(cohort),
        "n_block_units": sum(1 for u in cohort if u["block_split"] is not None),
        "n_vocal_testable": sum(1 for u in cohort if u["vocal_testable"]),
        "n_units_losing_a_session": sum(1 for u in cohort
                                        if u["n_vocal_sessions"] < len(u["courtship_sessions"])),
        "session_slots": slots,
        "vocal_session_slots": vocal_slots,
        "focal_usvs_total": int(total),
        "focal_usvs_retained": int(retained),
    }
