"""
@author: bartulem
The per-unit result file: one merged pickle carrying metrics and p-values, and no decisions.

Every analysis writes its own section into a single file per unit: effect sizes, p-values, the full
null distributions, per-fold detail and the data-sufficiency counts. Verdicts, pass flags, FDR marks
and labels are refused, because false-discovery control is a cohort operation that a per-unit file
cannot perform.

Sections are written by read-modify-write, so two jobs must not write sections of the same unit at
the same time; per-fold intermediates therefore stay in separate files.

Exports :func:`write_unit_section`, :func:`read_unit_artifact`, :func:`unit_artifact_path`,
:func:`build_provenance` and :func:`decision_key_paths`.
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

    The search is recursive because analyses nest their results (the gating label sits inside a
    per-feature entry). Matching is on the exact key, so a data array named ``labels`` (the 0/1 spike
    labels of the kinematic encoding) is not mistaken for a decision. Dicts and lists are descended
    into; arrays and scalars are leaves.

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

    Reuses the modeling pipeline's helpers, so neural and behavioural artifacts are traced the same
    way; the settings hash makes the configuration checkable after the fact.

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
    Merge one analysis's results into the unit's file, leaving the other analyses' sections untouched.

    Read-modify-write, published atomically, so a crash mid-write cannot leave a half-file. The
    identity and provenance blocks are refreshed on every write. Payloads carrying decision-shaped
    keys at any depth are refused (see :func:`decision_key_paths`).

    Parameters
    ----------
    output_directory (str)
        Run output directory.
    unit (dict)
        The cohort record; identity fields are copied from it.
    section (str)
        Which analysis these results belong to, e.g. ``'kinematic_encoding'``, ``'vocal_occurrence'``, ``'vocalization_identity'``.
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
    # The cohort record is the identity: uid, mouse, area, quality, both session sets and the
    # per-session focal-USV counts.
    artifact["identity"] = dict(unit)
    artifact["provenance"] = build_provenance(settings)

    path = unit_artifact_path(output_directory, unit["unit_uid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(path) as temporary, pathlib.Path(temporary).open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path
