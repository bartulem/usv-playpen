"""
@author: bartulem
The per-unit result file: one merged pickle carrying metrics and p-values, and no decisions.

Every claim writes its own SECTION into a single file per unit. What goes in is what a rule would
consume -- effect sizes, p-values, the full null distributions, per-fold detail, and the data-
sufficiency counts that say how much to trust any of it. What stays out is what a rule PRODUCES: no
verdicts, no pass flags, no FDR marks, and no labels -- including the gating sub-analysis's
per-feature GATE / int<vocal / ns label, its borderline flag and its survivor list, which apply
thresholds one unit at a time exactly as a verdict does.

That division is not tidiness. A per-unit file cannot know the cohort, and false-discovery control is
a cohort operation, so any verdict written here would be uncorrected by construction and would invite
being read as though it were not. The rules themselves have also moved twice in a week -- the verdict
became three-way, then transformation came to require the WHEN axis -- and each change would have
invalidated stored decisions while leaving stored metrics perfectly good. Deciding later is free;
re-fitting is not.

CONCURRENCY. Sections are written by read-modify-write, so two jobs must not write sections of the
SAME unit at the same time. This is why the claim-1 per-fold artifacts stay separate files: folds are
an array job and do run concurrently. They are intermediates. The merged file is written once per
claim, by the step that finishes it, and the claims run at different times.
"""

from __future__ import annotations

import pathlib
import pickle

from ..modeling.modeling_metadata import (
    compute_settings_sha256,
    get_git_commit_info,
    get_package_version,
)
from ..os_utils import atomic_output_path

DECISION_KEYS = ("verdict", "passed", "significant", "fdr_flag", "rejected", "is_transformation",
                 "label", "borderline", "survivors")


def decision_key_paths(value, path: str = "") -> list:
    """
    Description
    -----------
    Every place a decision-shaped key appears in a payload, at ANY depth, as a readable path.

    Checking only the top level would let decisions through wherever a claim's results are nested,
    and the gating sub-analysis is nested by construction: its per-feature label sits inside a
    per-feature entry, never at the top of the payload. A guard that cannot see there would refuse
    ``verdict`` while quietly storing nineteen of them under another name.

    Matching is on the exact key, so a data array named ``labels`` -- claim 1 stores the 0/1 spike
    labels of every vocal frame under that name -- is not mistaken for a decision. Checked against
    the real claim-1 and claim-2 payloads on disk before this was made recursive: 126-129 distinct
    nested keys, no collision.

    Dicts and lists are descended into; arrays and scalars are leaves.

    Parameters
    ----------
    value (Any)
        A payload, or any part of one.
    path (str)
        Where ``value`` sits, for the report; empty at the top.

    Returns
    -------
    paths (list)
        Dotted paths of every decision-shaped key found, e.g. ``per_feature.nose-nose.label``.
    """

    found = []
    if isinstance(value, dict):
        for key, inner in value.items():
            here = f"{path}.{key}" if path else str(key)
            if key in DECISION_KEYS:
                found.append(here)
            found.extend(decision_key_paths(inner, here))
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            found.extend(decision_key_paths(inner, f"{path}[{index}]"))
    return found


def build_provenance(settings: dict) -> dict:
    """
    Description
    -----------
    The provenance block every per-unit file carries: what code and what configuration produced it.

    Reuses the modeling pipeline's own helpers rather than reimplementing them, so a neural artifact
    and a behavioural one can be traced the same way. The settings hash is what makes "the frozen
    config" checkable after the fact instead of asserted.

    Parameters
    ----------
    settings (dict)
        The whole neural-modeling settings dict.

    Returns
    -------
    provenance (dict)
        ``settings_sha256``, ``git_commit``, ``git_dirty``, ``package_version``.
    """

    git = get_git_commit_info()
    return {"settings_sha256": compute_settings_sha256(settings),
            "git_commit": git.get("commit") if isinstance(git, dict) else None,
            "git_dirty": git.get("dirty") if isinstance(git, dict) else None,
            "package_version": get_package_version()}


def unit_artifact_path(output_directory: str, unit_uid: str) -> pathlib.Path:
    """
    Description
    -----------
    Where a unit's merged result file lives.

    Parameters
    ----------
    output_directory (str)
        Run output directory.
    unit_uid (str)
        The unit's uid.

    Returns
    -------
    path (pathlib.Path)
        ``<output_directory>/<unit_uid>.pkl``.
    """

    return pathlib.Path(output_directory) / f"{unit_uid}.pkl"


def read_unit_artifact(output_directory: str, unit_uid: str) -> dict:
    """
    Description
    -----------
    Read a unit's merged file, or an empty dict when it does not exist yet.

    Parameters
    ----------
    output_directory (str)
        Run output directory.
    unit_uid (str)
        The unit's uid.

    Returns
    -------
    artifact (dict)
        The stored artifact, or ``{}``.
    """

    path = unit_artifact_path(output_directory, unit_uid)
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_unit_section(output_directory: str, unit: dict, section: str, payload: dict,
                       settings: dict) -> pathlib.Path:
    """
    Description
    -----------
    Merge one claim's results into the unit's file, leaving the other claims' sections untouched.

    Read-modify-write, published atomically, so a crash mid-write cannot leave a half-file where a
    complete one used to be. The identity and provenance blocks are refreshed on every write, so the
    file always records the configuration that produced its most recent section.

    Payloads are checked for decision-shaped keys AT ANY DEPTH and refused (see
    :func:`decision_key_paths`). The file is for what a rule consumes,
    not what it produces: a verdict stored here would be uncorrected by construction, since false
    discovery control needs the cohort and a per-unit file has never seen it.

    Parameters
    ----------
    output_directory (str)
        Run output directory.
    unit (dict)
        The cohort record; identity fields are copied from it.
    section (str)
        Which claim's results these are, e.g. ``'claim1'``, ``'claim2_when'``, ``'claim2_what'``.
    payload (dict)
        Metrics, p-values, nulls and per-fold detail.
    settings (dict)
        The whole neural-modeling settings dict, for provenance.

    Returns
    -------
    path (pathlib.Path)
        Where it was written.
    """

    offending = sorted(decision_key_paths(payload))
    if offending:
        msg = (f"per-unit files carry metrics and p-values, not decisions; refusing {offending}. "
               f"A verdict written here would be uncorrected by construction -- false-discovery "
               f"control is a cohort operation. Let the consolidator decide.")
        raise ValueError(msg)

    artifact = read_unit_artifact(output_directory, unit["unit_uid"])
    artifact[section] = payload
    # The cohort record IS the identity: unit_uid, mouse, area, quality, both session sets and the
    # per-session focal-call counts the vocal gate was applied on.
    artifact["identity"] = dict(unit)
    artifact["provenance"] = build_provenance(settings)

    path = unit_artifact_path(output_directory, unit["unit_uid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(path) as temporary, pathlib.Path(temporary).open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path
