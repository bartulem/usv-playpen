"""
@author: bartulem
Build a QLVM training set (``.npz``) of USV spectrograms from a list of session
root directories, drawn the way the training sets of the QLVM model packages
(``qlvm_models_latest/v3``) were drawn.

This is the in-house port of the builders of those sets
(``build_masked_usvs.py`` and ``build_unmasked_usvs_floor.py`` in the MMMmB
repository, together with the session-typing, quota and stratification helpers
they imported from a local usv-playpen fork). Given the same sessions, inputs
and seed it selects the same rows, splits them into the same train and validation
sessions and writes the same spectrograms (see ``docs/Process.rst``). The steps:

1. **Pool.** For every session its ``audio/spectrograms/<session>_spectrograms.h5``
   is read (``durations`` and, unless ``masking_type`` is ``"none"``, the SAM
   ``mask/<session>/spectrogram_index``, from which each row's mask count is
   taken), its type is read from the ``Subjects[].sex`` entries of its metadata
   YAML (:func:`session_type_from_metadata`: ``MF``, ``FF``, ``MM``,
   ``lone_male``, ...), and the rows that are not ultrasonic calls alone are
   marked: with ``exclude_squeaks`` every row whose ``*_usv_summary.csv``
   ``call_class`` (written by ``detect-usv-squeaks``) is ``"squeak"`` or
   ``"both"``, so the set holds ``call_class`` ``"usv"`` rows only (with
   ``strict_squeak_exclusion`` the squeak rows come instead from the reference
   squeak index at ``reference_squeak_index_path``, by its strict rule, to
   reproduce the shipped sets), and with ``exclude_noise`` every row flagged as
   noise. A row is
   eligible when ``0 < duration < length_threshold``, it is not marked, and, with
   ``require_mask``, it has at least one SAM mask instance
   (:func:`eligible_rows`).
2. **Draw.** Sessions are grouped by type; a type missing from
   ``session_type_targets`` is left out, a type whose target is ``null`` is taken
   whole, and a type with a numeric target gets that many rows in total, split into
   per-session quotas by integer capped-even water-filling
   (:func:`allocate_even_quotas`). Within a session the quota is drawn either
   uniformly over its eligible rows (``draw_mode`` ``"natural"``: the session's
   own mask-count distribution) or equally across its mask-count strata
   (``"uniform"``, strata set by ``mask_count_bin_edges``), in which case the
   quotas are allocated against each session's uniform headroom
   (:func:`uniform_sampling_headroom`) rather than its eligible count, so every
   quota can be met exactly (:func:`select_rows_stratified`,
   :func:`compute_selected_rows_by_type`).
3. **Split.** Whole sessions are held out for validation, separately within each
   type, until the held-out sessions hold ``validation_split`` of that type's
   rows (:func:`split_sessions_by_type`), so no session straddles the boundary.
4. **Write.** Each drawn spectrogram (and its SAM mask union, binarized at 0.5
   after the same resize) is resized to ``target_shape`` (:func:`stretch_specs`),
   then either masked (``apply_mask``: ``spectrogram * mask``, phase 9) or left
   unmasked, optionally with a loudness floor baked in (``floor``: per-spectrogram
   min-max ``(x - min) / (max - min + 1e-8)`` followed by
   ``clip((x - floor) / (1 - floor), 0, 1)``, phase 6). A row without a SAM
   instance (possible only without ``require_mask``) keeps an all-ones mask
   (:func:`build_session_masks`), so masking leaves it unchanged rather than
   zeroed.

``full_dataset`` skips the draw and takes every eligible row of every session of a
listed type (the population the models are embedded in); the train/validation
pair is still written, and ``full_data.npz`` is written after it.

Each split ``.npz`` is row-aligned on dim 0 = N samples and holds
``spectrograms`` (N, F, T) float32, ``masks`` (N, F, T) float32 (the binarized
SAM region; all zero under ``masking_type`` ``"none"``), ``masks_len`` (N,) int64
(SAM instance count), ``durations`` (N,) int64 (native time bins), ``spec_id``
(N,) str (``{session}_{row}``), ``session_id`` (N,) str, ``session_type`` (N,)
str, ``mask_count`` (N,) int64, the scalar ``apply_mask`` that tells
``train-qlvm`` whether to multiply the masks in, and ``mean_freq_hz``,
``freq_bandwidth_hz``, ``loudness_db`` and ``spectral_entropy`` (N,) float64,
copied row for row from the session's USV summary (NaN where it has no value;
:func:`usv_summary_condition_values`) -- the raw values a conditional
``train-qlvm`` run conditions on, captured with the rows they belong to so a later
rewrite of the summary cannot shift them. ``metadata.npz`` records every
setting, the per-type report (available / target / drawn / per-stratum counts),
the session types and the train/validation session lists.

Session fingerprints. ``spec_id`` is ``{session}_{row}``, a row number in the
session's spectrogram H5, so it points at the right call only while that file is
unchanged; a rebuilt H5 renumbers its rows and every consumer joining on
``spec_id`` silently attaches the wrong calls (session 20251004_201051 of the v2
package went stale exactly this way). The builder therefore records, for every
session H5 it reads, the SHA-256 of the file's bytes, its row count and how many of
its rows entered the set -- in ``metadata.npz`` and in two sidecars laid out like the
v2 package's baseline: ``SESSION_H5.sha256`` (``sha256sum -c`` format) and
``SESSION_H5.tsv`` (session, h5_rows, corpus_rows, bytes, sha256, path). A consumer
compares the hash before joining; the summary CSV is not hashed, because it is
legitimately rewritten (columns added) while its rows stay aligned.

Squeaks (broadband vocalizations) get their own builder,
:mod:`build_qlvm_squeak_training_set`, which shares :func:`split_sessions_by_type`,
:func:`allocate_even_quotas` and :func:`stretch_specs` with this one.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import numpy as np
import polars as pls
from click.core import ParameterSource
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    CALL_CLASS_COLUMN,
    call_class_mask,
    configure_path,
    first_match_or_raise,
)
from ..time_utils import is_gui_context, smart_wait
from ..yaml_utils import load_session_metadata

# Epsilon of the per-spectrogram min-max that precedes the loudness floor; the
# same one the QLVM data loader and train-qlvm use (qmc_deep_gen data/mouse_data.py).
FLOOR_MINMAX_EPSILON = 1e-8

# The USV summary columns copied into every split, row for row: the raw per-call
# values a conditional train-qlvm run conditions on (mean frequency and bandwidth
# of the SAM-masked call, written by generate-usv-acoustic-features, and the
# absolute loudness it measures with compute_usv_loudness.session_image_level_db,
# and the spectral entropy in nats of the call's normalized frequency power profile).
SUMMARY_CONDITION_COLUMNS = ("mean_freq_hz", "freq_bandwidth_hz", "loudness_db", "spectral_entropy")


def file_sha256(path: str | pathlib.Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """
    Description
    -----------
    SHA-256 of a file's bytes, read in chunks so a multi-GB spectrogram H5 never
    has to fit in memory. Identical to ``sha256sum`` on the same file.

    Parameters
    ----------
    path (str | pathlib.Path)
        The file to hash.
    chunk_bytes (int)
        Read size; defaults to 8 MiB.

    Returns
    -------
    digest (str)
        The 64-character lowercase hexadecimal digest.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_type_from_subject_sexes(subject_sexes: list[str]) -> str:
    """
    Description
    -----------
    Classifies a session by the sexes of its subjects: two subjects give ``"MF"``
    (one male, one female), ``"MM"`` or ``"FF"``; one subject gives
    ``"lone_male"`` or ``"lone_female"``; anything else (three or more subjects,
    none, or a sex other than ``"male"`` / ``"female"``) gives ``"other"``. These
    are the session types the QLVM model packages' training sets were balanced
    over (``MF`` and ``FF`` budgeted, ``MM`` and ``lone_male`` taken whole).

    Parameters
    ----------
    subject_sexes (list[str])
        The ``sex`` entry of every subject, in any order and case.

    Returns
    -------
    session_type (str)
        ``"MF"``, ``"MM"``, ``"FF"``, ``"lone_male"``, ``"lone_female"`` or ``"other"``.
    """

    sexes = sorted(str(sex).strip().lower() for sex in subject_sexes)
    if any(sex not in ("male", "female") for sex in sexes):
        return "other"
    if sexes == ["female", "male"]:
        return "MF"
    if sexes == ["male", "male"]:
        return "MM"
    if sexes == ["female", "female"]:
        return "FF"
    if sexes == ["male"]:
        return "lone_male"
    if sexes == ["female"]:
        return "lone_female"
    return "other"


def session_type_from_metadata(root_directory: str, message_output: Callable) -> str:
    """
    Description
    -----------
    Reads a session's type from the ``Subjects[].sex`` entries of its
    ``*_metadata.yaml`` (:func:`session_type_from_subject_sexes`). A session
    whose metadata cannot be read, or has no ``Subjects`` or a subject without a
    ``sex``, is ``"unknown"`` rather than an error, so one defective file does not
    stop a cohort build; the builder then leaves the session out unless
    ``"unknown"`` is itself a listed type.

    Parameters
    ----------
    root_directory (str)
        Session root directory holding ``<session>_metadata.yaml``.
    message_output (Callable)
        Logging callback.

    Returns
    -------
    session_type (str)
        The session type, or ``"unknown"``.
    """

    metadata, _ = load_session_metadata(root_directory=root_directory, logger=message_output)
    if metadata is None or 'Subjects' not in metadata or not metadata['Subjects']:
        return "unknown"
    if any(not isinstance(subject, dict) or 'sex' not in subject for subject in metadata['Subjects']):
        return "unknown"
    return session_type_from_subject_sexes([subject['sex'] for subject in metadata['Subjects']])


def session_mask_counts(h5_file: h5py.File, session_id: str, n_rows: int) -> np.ndarray:
    """
    Description
    -----------
    Number of SAM mask instances of every row of a session's spectrogram H5,
    counted from ``mask/<session>/spectrogram_index`` alone (the large
    ``segmentations`` array is not read). A session without a
    ``mask/<session>`` group has zero instances on every row. The count equals
    the ``masks_len`` :func:`build_session_masks` returns for the same row.

    Parameters
    ----------
    h5_file (h5py.File)
        Open per-session spectrogram H5.
    session_id (str)
        Session id naming the ``mask/<session>`` group.
    n_rows (int)
        Number of spectrogram rows of the session.

    Returns
    -------
    mask_counts (np.ndarray)
        ``(n_rows,)`` int64 instance counts.
    """

    mask_group_key = f"mask/{session_id}"
    if mask_group_key not in h5_file:
        return np.zeros(n_rows, dtype=np.int64)
    spectrogram_index = h5_file[mask_group_key]["spectrogram_index"][:].astype(np.int64)
    return np.bincount(spectrogram_index, minlength=n_rows)[:n_rows].astype(np.int64)


def mask_count_bins(mask_counts: np.ndarray, bin_edges: list[int]) -> np.ndarray:
    """
    Description
    -----------
    Assigns each mask count to its stratum. ``bin_edges`` are the inclusive lower
    bounds of the strata above stratum 0, and the last stratum is open-ended:
    ``[1, 2, 3, 4, 5]`` gives strata 0 (no mask), 1, 2, 3, 4 and 5+ (five or
    more), ``[1, 2, 3]`` gives 0, 1, 2 and 3+.

    Parameters
    ----------
    mask_counts (np.ndarray)
        ``(N,)`` mask instance counts.
    bin_edges (list[int])
        Strictly increasing stratum lower bounds.

    Returns
    -------
    bins (np.ndarray)
        ``(N,)`` int64 stratum indices in ``0 .. len(bin_edges)``.
    """

    return np.digitize(np.asarray(mask_counts), np.asarray(bin_edges)).astype(np.int64)


def mask_count_bin_labels(bin_edges: list[int]) -> list[str]:
    """
    Description
    -----------
    Human-readable names of the strata of :func:`mask_count_bins`: ``"0"``,
    ``"1"``, ..., and ``"<last edge>+"`` for the open top stratum (a stratum
    spanning more than one count is named ``"<low>-<high>"``).

    Parameters
    ----------
    bin_edges (list[int])
        Strictly increasing stratum lower bounds.

    Returns
    -------
    labels (list[str])
        ``len(bin_edges) + 1`` labels, stratum 0 first.
    """

    lower_bounds = [0, *[int(edge) for edge in bin_edges]]
    labels = []
    for position, low in enumerate(lower_bounds):
        if position == len(lower_bounds) - 1:
            labels.append(f"{low}+")
        elif lower_bounds[position + 1] - low == 1:
            labels.append(f"{low}")
        else:
            labels.append(f"{low}-{lower_bounds[position + 1] - 1}")
    return labels


def eligible_rows(
    durations: np.ndarray,
    mask_counts: np.ndarray,
    excluded: np.ndarray,
    length_threshold: float,
    require_mask: bool,
) -> np.ndarray:
    """
    Description
    -----------
    The rows of a session that may enter the set: ``0 < duration <
    length_threshold`` (``duration == 0`` rows are the all-zero placeholders of
    invalid USVs), not ``excluded`` (squeak / noise rows of the USV summary), and,
    with ``require_mask``, at least one SAM mask instance.

    Parameters
    ----------
    durations (np.ndarray)
        ``(N,)`` native spectrogram durations (time bins).
    mask_counts (np.ndarray)
        ``(N,)`` SAM mask instance counts.
    excluded (np.ndarray)
        ``(N,)`` boolean, True for rows to leave out.
    length_threshold (float)
        Rows with ``duration >= length_threshold`` are left out.
    require_mask (bool)
        Leave out rows without a SAM mask instance.

    Returns
    -------
    rows (np.ndarray)
        Ascending int64 row indices.
    """

    keep = (durations > 0) & (durations < length_threshold) & ~np.asarray(excluded, dtype=bool)
    if require_mask:
        keep &= np.asarray(mask_counts) > 0
    return np.flatnonzero(keep).astype(np.int64)


def uniform_sampling_headroom(bins: np.ndarray) -> int:
    """
    Description
    -----------
    The largest number of rows a session can supply with its non-empty strata
    represented exactly equally: ``n_non_empty_strata * min(non-empty stratum
    size)``. A ``"uniform"`` draw allocates its per-session quotas against this
    headroom rather than the session's eligible count; allocating against the
    eligible count would hand some sessions more rows than they can supply
    equally, and the draw would silently fall back to a best-effort mix.

    Parameters
    ----------
    bins (np.ndarray)
        ``(N,)`` stratum of each eligible row of the session.

    Returns
    -------
    headroom (int)
        The uniform headroom (0 for a session without eligible rows).
    """

    if bins.size == 0:
        return 0
    counts = np.bincount(bins)
    counts = counts[counts > 0]
    return int(counts.size * counts.min())


def waterfill_level(available: np.ndarray, target: int) -> int:
    """
    Description
    -----------
    The largest integer level ``L`` whose capped allocation
    ``sum(min(available, L))`` does not exceed ``target`` (binary search). When
    ``target`` covers everything available, ``L`` is ``max(available)``.

    Parameters
    ----------
    available (np.ndarray)
        ``(K,)`` non-negative integer capacities.
    target (int)
        Total to allocate.

    Returns
    -------
    level (int)
        The water-filling level.
    """

    available = np.asarray(available, dtype=np.int64)
    if available.size == 0:
        return 0
    low, high = 0, int(available.max())
    while low < high:
        middle = (low + high + 1) // 2
        if int(np.minimum(available, middle).sum()) <= target:
            low = middle
        else:
            high = middle - 1
    return low


def allocate_even_quotas(available: np.ndarray, target: int, random_state: int) -> np.ndarray:
    """
    Description
    -----------
    Integer capped-even water-filling: every entry gets ``min(available, L)`` at
    the level ``L`` of :func:`waterfill_level`, and the remainder
    ``target - sum`` is handed out one unit at a time to the entries with room
    left (``available > L``), in the order of a permutation of all entries drawn
    from a fresh ``np.random.default_rng(random_state)``. The quotas therefore sum
    to ``min(target, sum(available))`` exactly and never exceed ``available``.
    Callers pass the entries in a fixed order (the builders sort session ids), so
    the tie-breaks do not depend on the order of the session list.

    Parameters
    ----------
    available (np.ndarray)
        ``(K,)`` non-negative integer capacities.
    target (int)
        Total to allocate.
    random_state (int)
        Seed of the tie-break permutation.

    Returns
    -------
    quotas (np.ndarray)
        ``(K,)`` int64 quotas.
    """

    available = np.asarray(available, dtype=np.int64)
    if target >= int(available.sum()):
        return available.copy()
    level = waterfill_level(available, target)
    quotas = np.minimum(available, level)
    remainder = int(target - quotas.sum())
    for position in np.random.default_rng(random_state).permutation(available.size):
        if remainder == 0:
            break
        if available[position] > level:
            quotas[position] += 1
            remainder -= 1
    return quotas


def select_rows_stratified(
    rows: np.ndarray,
    bins: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    draw_mode: str,
) -> np.ndarray:
    """
    Description
    -----------
    Draws ``n_target`` of a session's eligible rows without replacement. When the
    quota covers every row, all rows are kept and ``rng`` is not used.
    ``"natural"`` draws uniformly over the rows (``rng.choice`` over their
    positions), so the session's own mask-count mix is kept. ``"uniform"``
    water-fills the quota across the session's non-empty strata (ascending),
    handing any remainder, one row per stratum, to the strata with rows left in
    the order of ``rng.permutation`` over those strata, and then draws each
    stratum's share with ``rng.choice`` (a stratum taken whole uses no draw).

    Parameters
    ----------
    rows (np.ndarray)
        ``(N,)`` ascending eligible row indices of the session.
    bins (np.ndarray)
        ``(N,)`` stratum of each of those rows.
    n_target (int)
        Rows to draw.
    rng (np.random.Generator)
        Generator the draw consumes (shared across the sessions of a build).
    draw_mode (str)
        ``"natural"`` or ``"uniform"``.

    Returns
    -------
    selected (np.ndarray)
        Ascending int64 selected row indices.
    """

    if n_target >= rows.size:
        return rows.copy()
    if draw_mode == "natural":
        return np.sort(rows[rng.choice(rows.size, size=int(n_target), replace=False)])
    if draw_mode != "uniform":
        error_message = f"draw_mode must be 'natural' or 'uniform', got {draw_mode!r}."
        raise ValueError(error_message)

    strata = np.unique(bins)
    members = [rows[bins == stratum] for stratum in strata]
    counts = np.array([member.size for member in members], dtype=np.int64)
    level = waterfill_level(counts, n_target)
    quotas = np.minimum(counts, level)
    remainder = int(n_target - quotas.sum())
    if remainder > 0:
        for position in rng.permutation(strata.size):
            if remainder == 0:
                break
            if counts[position] > level:
                quotas[position] += 1
                remainder -= 1
    chosen = []
    for member, quota in zip(members, quotas, strict=True):
        if quota == 0:
            continue
        if quota >= member.size:
            chosen.append(member)
        else:
            chosen.append(member[rng.choice(member.size, size=int(quota), replace=False)])
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def compute_selected_rows_by_type(
    eligible_by_session: dict[str, np.ndarray],
    bins_by_session: dict[str, np.ndarray],
    session_type_by_key: dict[str, str],
    session_type_targets: dict[str, int | None],
    draw_mode: str,
    random_state: int,
    bin_labels: list[str],
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Description
    -----------
    The type-budgeted draw. Types are processed in sorted order; within a type the
    sessions keep the order of ``eligible_by_session`` (the session list's). A
    type whose target is ``None`` is taken whole. For a numeric target the
    per-session quotas are :func:`allocate_even_quotas` of each session's
    capacity -- its eligible count under ``"natural"``, its
    :func:`uniform_sampling_headroom` under ``"uniform"`` -- with the sessions in
    sorted id order, and each session's quota is then drawn with
    :func:`select_rows_stratified` from one generator,
    ``np.random.default_rng(random_state)``, shared by every session of every type.
    This order of operations is what reproduces the QLVM model packages' sets.

    Parameters
    ----------
    eligible_by_session (dict[str, np.ndarray])
        Ascending eligible rows per session, in session-list order; only sessions
        of a listed type.
    bins_by_session (dict[str, np.ndarray])
        Stratum of each eligible row, aligned with ``eligible_by_session``.
    session_type_by_key (dict[str, str])
        Type of every session.
    session_type_targets (dict[str, int | None])
        Rows per type (``None``: take the type whole).
    draw_mode (str)
        ``"natural"`` or ``"uniform"``.
    random_state (int)
        Seed of the allocation tie-breaks and of the draw.
    bin_labels (list[str])
        Stratum names (:func:`mask_count_bin_labels`) for the report.

    Returns
    -------
    selected (dict[str, np.ndarray])
        Ascending selected rows per session, in the input order.
    type_report (dict)
        Per type: ``n_sessions``, ``n_sessions_drawn``, ``available`` (eligible
        rows), ``capacity`` (the allocation basis summed; the exact-uniform ceiling
        under ``"uniform"``), ``target`` (``"all"`` for a whole type), ``drawn`` and
        ``per_bin`` (stratum label -> drawn rows).
    """

    draw_rng = np.random.default_rng(random_state)
    selected: dict[str, np.ndarray] = {}
    type_report: dict = {}
    session_types_present = sorted({session_type_by_key[key] for key in eligible_by_session})
    for session_type in session_types_present:
        sessions = [key for key in eligible_by_session if session_type_by_key[key] == session_type]
        target = session_type_targets[session_type]
        if draw_mode == "uniform":
            capacity = {key: uniform_sampling_headroom(bins_by_session[key]) for key in sessions}
        else:
            capacity = {key: int(eligible_by_session[key].size) for key in sessions}
        if target is None:
            for key in sessions:
                selected[key] = eligible_by_session[key].copy()
        else:
            sorted_sessions = sorted(sessions)
            quotas = allocate_even_quotas(
                np.array([capacity[key] for key in sorted_sessions], dtype=np.int64), int(target), random_state
            )
            quota_by_key = dict(zip(sorted_sessions, quotas.tolist(), strict=True))
            for key in sessions:
                selected[key] = select_rows_stratified(
                    eligible_by_session[key], bins_by_session[key], quota_by_key[key], draw_rng, draw_mode
                )
        per_bin = np.zeros(len(bin_labels), dtype=np.int64)
        for key in sessions:
            chosen_bins = bins_by_session[key][np.isin(eligible_by_session[key], selected[key])]
            per_bin += np.bincount(chosen_bins, minlength=len(bin_labels))[:len(bin_labels)]
        type_report[session_type] = {
            "n_sessions": len(sessions),
            "n_sessions_drawn": int(sum(selected[key].size > 0 for key in sessions)),
            "available": int(sum(eligible_by_session[key].size for key in sessions)),
            "capacity": int(sum(capacity.values())),
            "target": "all" if target is None else int(target),
            "drawn": int(sum(selected[key].size for key in sessions)),
            "per_bin": dict(zip(bin_labels, per_bin.tolist(), strict=True)),
        }
    return selected, type_report


def split_sessions_by_type(
    session_ids: list[str],
    session_type_by_key: dict[str, str],
    selected_counts: dict[str, int],
    validation_split: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """
    Description
    -----------
    Holds out whole sessions for validation, stratified by type. With one
    generator ``np.random.default_rng(random_state)``, the types are visited in
    sorted order; each type's sessions (sorted by id) are permuted and moved into
    validation one at a time until the moved sessions hold at least
    ``validation_split`` of that type's selected rows. No session straddles the
    boundary, and every type contributes about ``validation_split`` of its rows
    (at least that, by up to one session's rows).

    Parameters
    ----------
    session_ids (list[str])
        Sessions that contributed rows.
    session_type_by_key (dict[str, str])
        Type of every session.
    selected_counts (dict[str, int])
        Rows each session contributed.
    validation_split (float)
        Target validation fraction of each type's rows, in ``(0, 1)``.
    random_state (int)
        Seed of the permutations.

    Returns
    -------
    train_sessions (list[str])
        Sorted training sessions.
    val_sessions (list[str])
        Sorted validation sessions.
    """

    rng = np.random.default_rng(random_state)
    val_sessions: set[str] = set()
    for session_type in sorted({session_type_by_key[key] for key in session_ids}):
        keys = sorted(key for key in session_ids if session_type_by_key[key] == session_type)
        target = validation_split * sum(selected_counts[key] for key in keys)
        held_out = 0
        for position in rng.permutation(len(keys)):
            if held_out >= target:
                break
            val_sessions.add(keys[position])
            held_out += selected_counts[keys[position]]
    return sorted(set(session_ids) - val_sessions), sorted(val_sessions)


def apply_loudness_floor(spectrograms: np.ndarray, floor: float) -> np.ndarray:
    """
    Description
    -----------
    Bakes an AVA-style loudness floor into spectrograms: each is min-max
    normalized on its own, ``x = (s - min) / (max - min + 1e-8)``, and then
    ``clip((x - floor) / (1 - floor), 0, 1)``, so everything below ``floor`` of
    its range is silenced and the rest is stretched back to ``[0, 1]`` (the
    phase 6 recipe; a later per-spectrogram min-max, as ``train-qlvm`` applies,
    leaves the result unchanged).

    Parameters
    ----------
    spectrograms (np.ndarray)
        ``(N, F, T)`` float32 spectrograms.
    floor (float)
        Floor in ``[0, 1)``.

    Returns
    -------
    floored (np.ndarray)
        ``(N, F, T)`` float32 spectrograms in ``[0, 1]``.
    """

    low = spectrograms.min(axis=(1, 2), keepdims=True)
    high = spectrograms.max(axis=(1, 2), keepdims=True)
    normalized = (spectrograms - low) / (high - low + FLOOR_MINMAX_EPSILON)
    return np.clip((normalized - floor) / (1.0 - floor), 0.0, 1.0).astype(np.float32)


def read_reference_squeak_index(reference_squeak_index_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Description
    -----------
    Reads the reference squeak index (the per-segment table of the classifier the
    shipped USV training sets were curated with, ``bbv_segment_index.csv``) and
    applies its strict squeak rule: a segment is a squeak when its segment flag
    ``is_bbv`` is true OR it holds at least one run of three above-threshold frames
    (``n_bouts_min3 >= 1``). By this rule the reference USV training sets left
    broadband calls out, so a rebuild of those sets has to exclude exactly these
    rows. Only the four needed columns are read, ``session_id`` as a string.

    Parameters
    ----------
    reference_squeak_index_path (str)
        Path to the index CSV (columns ``session_id``, ``seg_index`` -- the 0-based
        row of the session's USV summary and spectrogram H5 --, ``is_bbv`` and
        ``n_bouts_min3``, among others).

    Returns
    -------
    index (dict[str, tuple[np.ndarray, np.ndarray]])
        Session id -> (``seg_index`` (int64), strict squeak flag (bool)), row-aligned.

    Raises
    ------
    FileNotFoundError
        The index does not exist.
    ValueError
        A needed column is missing.
    """

    path = pathlib.Path(configure_path(reference_squeak_index_path))
    if not path.is_file():
        error_message = f"Reference squeak index not found: {path}."
        raise FileNotFoundError(error_message)
    needed = ["session_id", "seg_index", "is_bbv", "n_bouts_min3"]
    header = pls.read_csv(str(path), n_rows=0).columns
    missing = [column for column in needed if column not in header]
    if missing:
        error_message = f"{path} lacks the column(s) {missing}; the strict squeak rule reads {needed}."
        raise ValueError(error_message)
    table = pls.read_csv(
        str(path),
        columns=needed,
        schema_overrides={"session_id": pls.String, "seg_index": pls.Int64, "is_bbv": pls.String, "n_bouts_min3": pls.Int64},
    )
    strict = table["is_bbv"].str.to_lowercase().eq("true").fill_null(False) | (table["n_bouts_min3"].fill_null(0) >= 1)
    table = table.with_columns(strict.alias("strict"))
    return {
        str(session_id): (group["seg_index"].to_numpy().astype(np.int64), group["strict"].to_numpy().astype(bool))
        for (session_id,), group in table.group_by("session_id", maintain_order=True)
    }


def reference_strict_squeak_rows(index: dict[str, tuple[np.ndarray, np.ndarray]], session_id: str, n_rows: int) -> np.ndarray:
    """
    Description
    -----------
    One session's rows that the reference squeak index's strict rule marks as
    squeaks (:func:`read_reference_squeak_index`), as a boolean over the
    session's spectrogram H5 rows. A row the index does not list is not marked.

    Parameters
    ----------
    index (dict[str, tuple[np.ndarray, np.ndarray]])
        The read index.
    session_id (str)
        The session (its spectrogram H5 group name).
    n_rows (int)
        Row count of the session's spectrogram H5.

    Returns
    -------
    squeak (np.ndarray)
        ``(n_rows,)`` boolean.

    Raises
    ------
    ValueError
        The session is not in the index (its strict exclusions are unknown), or the
        index lists a row outside ``0 .. n_rows - 1`` (the session's rows changed
        since the index was written).
    """

    if session_id not in index:
        error_message = (
            f"Session {session_id} is not in the reference squeak index, so its strict squeak exclusions are "
            f"unknown; leave it out or turn strict_squeak_exclusion off."
        )
        raise ValueError(error_message)
    rows, strict = index[session_id]
    if rows.size and (rows.min() < 0 or rows.max() >= n_rows):
        error_message = (
            f"The reference squeak index lists rows {int(rows.min())}..{int(rows.max())} for session {session_id}, "
            f"whose spectrogram H5 has {n_rows} rows; the index does not describe this session's rows."
        )
        raise ValueError(error_message)
    squeak = np.zeros(n_rows, dtype=bool)
    squeak[rows[strict]] = True
    return squeak


def usv_summary_exclusions(
    root_directory: str,
    n_rows: int,
    exclude_squeaks: bool,
    exclude_noise: bool,
    reference_squeak_rows: np.ndarray | None = None,
) -> np.ndarray:
    """
    Description
    -----------
    The rows of a session to leave out of a USV training set, which holds
    ultrasonic calls alone. With ``exclude_squeaks`` every row whose
    ``*_usv_summary.csv`` ``call_class`` (written by ``detect-usv-squeaks``) is
    ``"squeak"`` (a squeak alone) or ``"both"`` (a squeak and a USV in one segment)
    is left out, so only ``call_class`` ``"usv"`` rows can be drawn; a null
    ``call_class`` (a noise row, or a segment too short to score) is not a squeak
    -- noise is left out by ``exclude_noise`` and the unscorable segments by the
    duration gate (:func:`eligible_rows`). With ``exclude_noise`` every row whose
    ``noise`` (written by ``detect-usv-noise``) is true is left out; a null counts
    as false. When ``reference_squeak_rows`` is given (``strict_squeak_exclusion``,
    which needs ``exclude_squeaks``) it REPLACES the ``call_class`` rule: the
    squeak rows are those the reference squeak index marks by its strict rule
    (:func:`reference_strict_squeak_rows`), the exclusions the shipped USV training
    sets were drawn with. The summary is read only when a rule needs it, and its
    rows must be 1:1 with the spectrogram H5 rows, which is checked.

    Parameters
    ----------
    root_directory (str)
        Session root directory.
    n_rows (int)
        Row count of the session's spectrogram H5.
    exclude_squeaks (bool)
        Leave out squeak and both rows.
    exclude_noise (bool)
        Leave out noise rows.
    reference_squeak_rows (np.ndarray | None)
        ``(n_rows,)`` boolean squeak rows of the reference squeak index, used
        instead of ``call_class``; None uses ``call_class``.

    Returns
    -------
    excluded (np.ndarray)
        ``(n_rows,)`` boolean.

    Raises
    ------
    ValueError
        The summary's row count differs from the H5's, a needed column is missing
        (``call_class`` or ``noise``), or ``reference_squeak_rows`` is given without
        ``exclude_squeaks`` or with the wrong shape.
    """

    if reference_squeak_rows is not None and not exclude_squeaks:
        error_message = "strict_squeak_exclusion replaces the squeak exclusion, so it needs exclude_squeaks."
        raise ValueError(error_message)

    excluded = np.zeros(n_rows, dtype=bool)
    if reference_squeak_rows is not None:
        reference_squeak_rows = np.asarray(reference_squeak_rows, dtype=bool)
        if reference_squeak_rows.shape != (n_rows,):
            error_message = f"reference_squeak_rows has shape {reference_squeak_rows.shape}, the H5 has {n_rows} rows."
            raise ValueError(error_message)
        excluded |= reference_squeak_rows
    use_call_class = exclude_squeaks and reference_squeak_rows is None
    if not use_call_class and not exclude_noise:
        return excluded
    usv_summary_path = first_match_or_raise(
        root=pathlib.Path(root_directory) / "audio",
        pattern="*_usv_summary.csv",
        recursive=True,
        label="USV summary CSV",
    )
    usv_summary = pls.read_csv(source=str(usv_summary_path), schema_overrides={"usv_id": pls.String})
    if usv_summary.height != n_rows:
        error_message = (
            f"{usv_summary_path} has {usv_summary.height} rows but the session's spectrogram H5 has {n_rows}; "
            f"the summary flags cannot be joined to the spectrograms by row."
        )
        raise ValueError(error_message)
    if use_call_class:
        if CALL_CLASS_COLUMN not in usv_summary.columns:
            error_message = f"{usv_summary_path} has no '{CALL_CLASS_COLUMN}' column; run detect-usv-squeaks on the session first."
            raise ValueError(error_message)
        excluded |= call_class_mask(usv_summary, ("squeak", "both"), usv_summary_path.name).to_numpy()
    if exclude_noise:
        if "noise" not in usv_summary.columns:
            error_message = f"{usv_summary_path} has no 'noise' column; run detect-usv-noise on the session first."
            raise ValueError(error_message)
        excluded |= usv_summary["noise"].cast(pls.Boolean).fill_null(False).to_numpy()
    return excluded


def usv_summary_condition_values(root_directory: str, n_rows: int) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    The raw per-call values a conditional QLVM conditions on, read from the
    session's ``*_usv_summary.csv`` (rows 1:1 with the spectrogram H5 rows, which
    is checked): ``mean_freq_hz`` and ``freq_bandwidth_hz`` (the energy-weighted
    mean frequency and the bandwidth of the call's SAM-masked region) and
    ``loudness_db`` (the absolute image-level loudness over the same mask region,
    :func:`compute_usv_loudness.session_image_level_db`) and ``spectral_entropy``
    (the entropy in nats of the call's normalized frequency power profile), all
    four written by ``generate-usv-acoustic-features``. A column the summary lacks, a null or NaN
    value, and every row of a session without a summary are NaN: the values are
    only needed by a conditional ``train-qlvm`` run, which refuses a training row
    without one.

    Parameters
    ----------
    root_directory (str)
        Session root directory.
    n_rows (int)
        Row count of the session's spectrogram H5.

    Returns
    -------
    values (dict[str, np.ndarray])
        ``SUMMARY_CONDITION_COLUMNS`` name -> ``(n_rows,)`` float64.

    Raises
    ------
    ValueError
        The summary's row count differs from the H5's.
    """

    values = {column: np.full(n_rows, np.nan) for column in SUMMARY_CONDITION_COLUMNS}
    summary_paths = sorted((pathlib.Path(root_directory) / "audio").rglob("*_usv_summary.csv"))
    if not summary_paths:
        return values
    usv_summary = pls.read_csv(source=str(summary_paths[0]), schema_overrides={"usv_id": pls.String})
    if usv_summary.height != n_rows:
        error_message = (
            f"{summary_paths[0]} has {usv_summary.height} rows but the session's spectrogram H5 has {n_rows}; "
            f"the summary values cannot be joined to the spectrograms by row."
        )
        raise ValueError(error_message)
    for column in SUMMARY_CONDITION_COLUMNS:
        if column in usv_summary.columns:
            values[column] = usv_summary[column].cast(pls.Float64).fill_null(np.nan).to_numpy()
    return values


def _apply_time_stretching(spec: np.ndarray, duration: int, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Description
    -----------
    Time-stretches a spectrogram's signal window ``[0, duration]`` to fill the
    full target time axis via linear interpolation (the QLVM time-warp option).

    Parameters
    ----------
    spec (np.ndarray)
        A ``(F, T)`` spectrogram.
    duration (int)
        Native signal length in time bins.
    target_shape (tuple[int, int])
        Output ``(freq, time)`` shape.

    Returns
    -------
    spec_resized (np.ndarray)
        The stretched ``target_shape`` spectrogram.
    """

    freq_orig = np.arange(spec.shape[0])
    time_orig = np.arange(duration)
    spec_signal = spec[:, :duration]
    interpolator = RegularGridInterpolator(
        (freq_orig, time_orig), spec_signal, method="linear", bounds_error=False, fill_value=0.0
    )
    freq_new = np.linspace(0, spec.shape[0] - 1, target_shape[0])
    time_new = np.linspace(0, duration - 1, target_shape[1])
    freq_grid, time_grid = np.meshgrid(freq_new, time_new, indexing="ij")
    points = np.stack([freq_grid.ravel(), time_grid.ravel()], axis=-1)
    return interpolator(points).reshape(target_shape)


def _apply_simple_resize(spec: np.ndarray, duration: int, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Description
    -----------
    Zoom-resizes a spectrogram to ``target_shape`` and center-pads the signal
    window (the QLVM non-warp option).

    Parameters
    ----------
    spec (np.ndarray)
        A ``(F, T)`` spectrogram.
    duration (int)
        Native signal length in time bins.
    target_shape (tuple[int, int])
        Output ``(freq, time)`` shape.

    Returns
    -------
    spec_centered (np.ndarray)
        The resized, centered ``target_shape`` spectrogram.
    """

    zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
    spec_interp = zoom(spec, zoom_factors, order=1)
    # The signal window occupies `duration` columns in NATIVE coordinates, but
    # after the zoom it spans `duration * zoom_factors[1]` columns in the resized
    # array. Slice in zoomed coordinates so time-upsampling (target wider than the
    # native spec) keeps the whole signal instead of truncating it to the first
    # `duration` columns.
    signal_length = min(int(round(duration * zoom_factors[1])), target_shape[1])
    signal_portion = spec_interp[:, :signal_length]
    left_pad = (target_shape[1] - signal_length) // 2
    right_pad = target_shape[1] - signal_length - left_pad
    return np.pad(signal_portion, ((0, 0), (left_pad, right_pad)), mode="constant", constant_values=0.0)


def stretch_specs(
    spectrograms: np.ndarray,
    durations: np.ndarray,
    target_shape: tuple[int, int],
    time_stretch: bool,
) -> np.ndarray:
    """
    Description
    -----------
    Resizes every spectrogram to ``target_shape``, either time-stretching the
    signal window to fill the frame or center-resizing it. Zeros the padded tail
    beyond each spectrogram's native ``duration`` before resizing.

    Parameters
    ----------
    spectrograms (np.ndarray)
        ``(N, F, T)`` spectrograms.
    durations (np.ndarray)
        Native time-bin counts, shape ``(N,)``.
    target_shape (tuple[int, int])
        Output ``(freq, time)`` shape.
    time_stretch (bool)
        Time-warp if True, else center-resize.

    Returns
    -------
    resized (np.ndarray)
        ``(N, *target_shape)`` float32 array.
    """

    n = spectrograms.shape[0]
    resized = np.empty((n, *target_shape), dtype=np.float32)
    for idx in range(n):
        spec_work = spectrograms[idx].copy()
        duration = int(min(max(int(durations[idx]), 1), spec_work.shape[1]))
        spec_work[:, duration:] = 0
        if time_stretch:
            resized[idx] = _apply_time_stretching(spec_work, duration, target_shape)
        else:
            resized[idx] = _apply_simple_resize(spec_work, duration, target_shape)
    return resized


def build_session_masks(
    h5_file: h5py.File,
    session_id: str,
    selected_indices: np.ndarray,
    n_freq: int,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Builds per-spectrogram 2D region masks and instance counts for one session's
    selected rows from its ``mask/<session>`` group (written by
    :mod:`generate_masks`). Each selected row's mask is the boolean union
    (``np.any``) of every ``segmentations`` row whose ``spectrogram_index`` equals
    that row; a row with no mask instances (the detector found none) falls back to
    an all-ones mask so its spectrogram is later kept unchanged rather than zeroed.
    When the session has no ``mask/<session>`` group, every selected row gets an
    all-ones mask and a zero instance count.

    Parameters
    ----------
    h5_file (h5py.File)
        Open per-session spectrogram H5.
    session_id (str)
        Session id naming the ``spectrogram/<session>`` / ``mask/<session>`` groups.
    selected_indices (np.ndarray)
        Sorted usv_summary row indices kept for this session.
    n_freq (int)
        Frequency-bin count ``F`` of each mask (mask height).
    n_time (int)
        Time-bin count ``T`` of each mask (mask width).

    Returns
    -------
    masks (np.ndarray)
        A ``(len(selected_indices), F, T)`` float32 array (1.0 inside the region,
        0.0 outside; all-ones for rows that fall back).
    masks_len (np.ndarray)
        A ``(len(selected_indices),)`` int64 array of per-row instance counts.
    """

    n_selected = selected_indices.shape[0]
    masks = np.ones((n_selected, n_freq, n_time), dtype=np.float32)
    masks_len = np.zeros(n_selected, dtype=np.int64)

    mask_group_key = f"mask/{session_id}"
    if mask_group_key not in h5_file:
        return masks, masks_len

    mask_group = h5_file[mask_group_key]
    segmentations = mask_group["segmentations"][:]
    spectrogram_index = mask_group["spectrogram_index"][:]
    # Group the mask rows by their spectrogram_index once (argsort + searchsorted)
    # instead of rescanning the whole spectrogram_index array per selected row
    # (the previous O(n_selected x n_masks) flatnonzero loop). For each selected
    # summary row, the contiguous block of the sort order gives exactly the same
    # set of mask rows the equality scan produced; np.any over axis 0 and the row
    # count are both order-independent, so the result is byte-for-byte identical.
    sort_order = np.argsort(spectrogram_index, kind="stable")
    sorted_index = spectrogram_index[sort_order]
    summary_rows = selected_indices.astype(sorted_index.dtype, copy=False)
    lo = np.searchsorted(sorted_index, summary_rows, side="left")
    hi = np.searchsorted(sorted_index, summary_rows, side="right")
    for position in range(n_selected):
        if hi[position] > lo[position]:
            mask_rows = sort_order[lo[position]:hi[position]]
            masks[position] = np.any(segmentations[mask_rows], axis=0).astype(np.float32)
            masks_len[position] = int(hi[position] - lo[position])
    return masks, masks_len


def parse_session_type_targets(value: str) -> dict:
    """
    Description
    -----------
    Decodes the ``--session-type-targets`` JSON object (session type -> rows,
    ``null`` to take the type whole), so the settings override stores a
    dictionary, not a string.

    Parameters
    ----------
    value (str)
        The raw option value, e.g. ``'{"MF": 29000, "FF": 29000, "MM": null, "lone_male": null}'``.

    Returns
    -------
    targets (dict)
        The decoded targets.

    Raises
    ------
    click.BadParameter
        The value is not a JSON object of non-negative integers or nulls.
    """

    try:
        targets = json.loads(value)
    except json.JSONDecodeError as error:
        error_message = f"--session-type-targets is not valid JSON ({error})."
        raise click.BadParameter(error_message) from error
    if not isinstance(targets, dict) or any(
            not (target is None or (isinstance(target, int) and target >= 0)) for target in targets.values()):
        error_message = (
            '--session-type-targets expects a JSON object of session type -> non-negative integer or null, '
            'e.g. \'{"MF": 29000, "FF": 29000, "MM": null, "lone_male": null}\'.'
        )
        raise click.BadParameter(error_message)
    return targets


def parse_optional_float(value: str) -> float | None:
    """
    Description
    -----------
    Decodes an option that takes a float or ``none`` (``--floor``): ``"none"`` /
    ``"null"`` become None (written to the settings as JSON null), anything else
    a float.

    Parameters
    ----------
    value (str)
        The raw option value.

    Returns
    -------
    parsed (float | None)
        The float, or None.

    Raises
    ------
    click.BadParameter
        The value is neither a number nor ``none``.
    """

    if value.strip().lower() in ("none", "null"):
        return None
    try:
        return float(value)
    except ValueError as error:
        error_message = f"expected a number or 'none', got {value!r}."
        raise click.BadParameter(error_message) from error


def parse_int_list(value: str) -> list[int]:
    """
    Description
    -----------
    Decodes a comma-separated integer-list option (``--mask-count-bin-edges 1,2,3,4,5``).

    Parameters
    ----------
    value (str)
        The raw option value.

    Returns
    -------
    parsed (list[int])
        The integers.

    Raises
    ------
    click.BadParameter
        An item is not an integer.
    """

    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        error_message = f"expected comma-separated integers, got {value!r}."
        raise click.BadParameter(error_message) from error


class QLVMTrainingSetBuilder:
    """
    Description
    -----------
    Builds a QLVM ``.npz`` training set of USV spectrograms from a list of
    session root directories (see the module docstring).
    """

    def __init__(
        self,
        root_directories: list[str] | None = None,
        output_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
        row_exclusions: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the QLVMTrainingSetBuilder.

        Parameters
        ----------
        root_directories (list[str])
            Session root directories to combine; each session's
            ``audio/spectrograms/<session>_spectrograms.h5`` is read. Their order is
            the order sessions are drawn in (and written in).
        output_directory (str)
            Directory to write the ``.npz`` outputs + metadata.
        input_parameter_dict (dict)
            Processing settings; the ``build_qlvm_training_set`` block supplies
            every parameter.
        message_output (Callable)
            Logging callback; defaults to ``print``.
        row_exclusions (dict[str, np.ndarray] | None)
            Python-API only: per session id, a boolean ``(n_rows,)`` array of rows to
            leave out that REPLACES the ``exclude_squeaks`` / ``exclude_noise`` /
            ``strict_squeak_exclusion`` rules for that session (e.g. exclusions computed
            elsewhere, to rebuild a set whose exclusions came from outside the summary
            and the reference squeak index). None (the default) applies the settings'
            rules to every session.

        Returns
        -------
        None
        """

        self.root_directories = root_directories if root_directories is not None else []
        self.output_directory = output_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.row_exclusions = row_exclusions
        self.app_context_bool = is_gui_context()

    def build(self) -> None:
        """
        Description
        -----------
        Runs the build: reads every session's durations, mask counts, type and
        summary exclusions (fingerprinting its spectrogram H5), draws the rows
        (:func:`compute_selected_rows_by_type`, or every eligible row under
        ``full_dataset``), reads and resizes the drawn spectrograms and masks
        session by session (with the drawn rows' raw conditioning values from the
        USV summary, :func:`usv_summary_condition_values`), splits whole sessions into train and validation
        (:func:`split_sessions_by_type`), masks them or bakes in the floor, and
        writes ``train_data.npz``, ``val_data.npz`` (then ``full_data.npz`` under
        ``full_dataset``), ``metadata.npz``, ``SESSION_H5.sha256`` and
        ``SESSION_H5.tsv`` to ``output_directory``.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(
            f"QLVM training-set build started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['build_qlvm_training_set']
        session_type_targets = cfg['session_type_targets']
        draw_mode = cfg['draw_mode']
        bin_edges = [int(edge) for edge in cfg['mask_count_bin_edges']]
        length_threshold = float(cfg['length_threshold'])
        require_mask = cfg['require_mask']
        exclude_squeaks = cfg['exclude_squeaks']
        exclude_noise = cfg['exclude_noise']
        strict_squeak_exclusion = cfg['strict_squeak_exclusion']
        reference_squeak_index_path = cfg['reference_squeak_index_path']
        masking_type = cfg['masking_type']
        apply_mask = cfg['apply_mask']
        floor = None if cfg['floor'] is None else float(cfg['floor'])
        validation_split = cfg['validation_split']
        random_state = cfg['random_state']
        full_dataset = cfg['full_dataset']
        target_shape = tuple(int(v) for v in cfg['target_shape'])
        time_stretch = cfg['time_stretch']

        problems = []
        if not 0.0 < validation_split < 1.0:
            problems.append(f"validation_split must be in the open interval (0, 1), got {validation_split}")
        if draw_mode not in ("natural", "uniform"):
            problems.append(f"draw_mode must be 'natural' or 'uniform', got {draw_mode!r}")
        if masking_type not in ("sam", "none"):
            problems.append(f"masking_type must be 'sam' or 'none', got {masking_type!r}")
        if masking_type == "none" and (apply_mask or require_mask or draw_mode == "uniform"):
            problems.append(
                "masking_type 'none' reads no SAM masks, so it cannot apply them (apply_mask), require them "
                "(require_mask) or stratify by their count (draw_mode 'uniform'); use masking_type 'sam'"
            )
        if floor is not None and apply_mask:
            problems.append("a loudness floor is baked into unmasked sets only; set apply_mask false or floor null")
        if floor is not None and not 0.0 <= floor < 1.0:
            problems.append(f"floor must be in [0, 1), got {floor}")
        if not bin_edges or any(later <= earlier for earlier, later in itertools.pairwise(bin_edges)) or bin_edges[0] < 1:
            problems.append(f"mask_count_bin_edges must be strictly increasing integers >= 1, got {bin_edges}")
        if not session_type_targets:
            problems.append("session_type_targets is empty; list at least one session type")
        if strict_squeak_exclusion and not exclude_squeaks:
            problems.append("strict_squeak_exclusion replaces the squeak exclusion, so it needs exclude_squeaks")
        if strict_squeak_exclusion and not reference_squeak_index_path:
            problems.append("strict_squeak_exclusion reads the reference squeak index, but reference_squeak_index_path is empty")
        if problems:
            error_message = "build_qlvm_training_set settings are inconsistent:\n  " + "\n  ".join(problems)
            raise ValueError(error_message)

        output_dir = pathlib.Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        bin_labels = mask_count_bin_labels(bin_edges)
        # The reference squeak index is one cohort-wide table: read once, looked up per session.
        reference_index = read_reference_squeak_index(reference_squeak_index_path) if strict_squeak_exclusion else None

        # Phase 1: per-session metadata. The H5 pattern is session-keyed rather
        # than "*_spectrograms.h5" because a session can hold other files ending in
        # that suffix (e.g. a sonic-band "<session>_3_30khz_spectrograms.h5").
        # Each file is fingerprinted before any row is taken from it (see the
        # module docstring, "Session fingerprints").
        sessions: dict[str, dict] = {}
        skipped_by_type: dict[str, int] = {}
        excluded_by_type: dict[str, int] = {}
        for root_directory in self.root_directories:
            h5_path = str(first_match_or_raise(
                root=pathlib.Path(root_directory) / "audio" / "spectrograms",
                pattern=f"{pathlib.Path(root_directory).name}_spectrograms.h5",
                label="per-session spectrogram H5",
            ))
            session_type = session_type_from_metadata(root_directory, self.message_output)
            if session_type not in session_type_targets:
                skipped_by_type[session_type] = skipped_by_type.get(session_type, 0) + 1
                continue
            with h5py.File(h5_path, "r") as h5_file:
                session_id = next(iter(h5_file["spectrogram"].keys()))
                durations = h5_file[f"spectrogram/{session_id}"]["durations"][:].astype(np.int64)
                if masking_type == "sam":
                    mask_counts = session_mask_counts(h5_file, session_id, durations.size)
                else:
                    mask_counts = np.zeros(durations.size, dtype=np.int64)
            if self.row_exclusions is not None and session_id in self.row_exclusions:
                excluded = np.asarray(self.row_exclusions[session_id], dtype=bool)
                if excluded.shape != durations.shape:
                    error_message = f"row_exclusions[{session_id!r}] has shape {excluded.shape}, the H5 has {durations.size} rows."
                    raise ValueError(error_message)
            else:
                reference_rows = None if reference_index is None else reference_strict_squeak_rows(reference_index, session_id, durations.size)
                excluded = usv_summary_exclusions(root_directory, durations.size, exclude_squeaks, exclude_noise, reference_rows)
            rows = eligible_rows(durations, mask_counts, excluded, length_threshold, require_mask)
            without_exclusion = eligible_rows(durations, mask_counts, np.zeros_like(excluded), length_threshold, require_mask)
            excluded_by_type[session_type] = excluded_by_type.get(session_type, 0) + int(without_exclusion.size - rows.size)
            sessions[session_id] = {
                "root": root_directory,
                "h5_path": h5_path,
                "sha256": file_sha256(h5_path),
                "type": session_type,
                "durations": durations,
                "mask_counts": mask_counts,
                "eligible": rows,
                "bins": mask_count_bins(mask_counts[rows], bin_edges),
            }
        if skipped_by_type:
            self.message_output(
                "Left out sessions of types not in session_type_targets: "
                + ", ".join(f"{session_type}={count}" for session_type, count in sorted(skipped_by_type.items())) + "."
            )
        self.message_output(
            f"{len(sessions)} sessions; eligible rows removed by the summary exclusions: "
            + (", ".join(f"{session_type}={count:,}" for session_type, count in sorted(excluded_by_type.items())) or "none") + "."
        )
        if not sessions:
            self.message_output("No session of a listed type; nothing written.")
            return
        session_type_by_key = {session_id: session['type'] for session_id, session in sessions.items()}

        # Phase 2: the type-budgeted draw (or every eligible row).
        eligible_by_session = {session_id: session['eligible'] for session_id, session in sessions.items()}
        bins_by_session = {session_id: session['bins'] for session_id, session in sessions.items()}
        if full_dataset:
            selected, type_report = compute_selected_rows_by_type(
                eligible_by_session, bins_by_session, session_type_by_key,
                dict.fromkeys(session_type_targets), draw_mode, random_state, bin_labels,
            )
        else:
            selected, type_report = compute_selected_rows_by_type(
                eligible_by_session, bins_by_session, session_type_by_key,
                session_type_targets, draw_mode, random_state, bin_labels,
            )
        for session_type, report in type_report.items():
            shortfall = "" if report['target'] == "all" or report['drawn'] == report['target'] else (
                f" -- SHORT of the target {report['target']:,}: the {draw_mode} capacity of this type is {report['capacity']:,}"
            )
            self.message_output(
                f"  {session_type}: {report['drawn']:,} rows from {report['n_sessions_drawn']}/{report['n_sessions']} sessions "
                f"(available {report['available']:,}, target {report['target']}), per stratum {report['per_bin']}{shortfall}."
            )

        # Phase 3: read and resize the drawn rows session by session (every
        # transform is per row, so resizing here equals resizing the stacked set).
        specs_list, masks_list, masks_len_list, durations_list = [], [], [], []
        spec_id_list, session_list, type_list, count_list = [], [], [], []
        condition_lists: dict[str, list[np.ndarray]] = {column: [] for column in SUMMARY_CONDITION_COLUMNS}
        for session_id, session in sessions.items():
            idx = selected[session_id]
            if idx.size == 0:
                continue
            summary_values = usv_summary_condition_values(session['root'], session['durations'].size)
            for column in SUMMARY_CONDITION_COLUMNS:
                condition_lists[column].append(summary_values[column][idx])
            with h5py.File(session['h5_path'], "r") as h5_file:
                session_group = h5_file[f"spectrogram/{session_id}"]
                native_specs = session_group["spectrograms"][idx]
                n_freq, n_time = session_group["spectrograms"].shape[1:]
                if masking_type == "sam":
                    native_masks, native_masks_len = build_session_masks(h5_file, session_id, idx, n_freq, n_time)
            native_durations = session['durations'][idx]
            specs_list.append(stretch_specs(native_specs, native_durations, target_shape, time_stretch))
            if masking_type == "sam":
                resized_masks = stretch_specs(native_masks, native_durations, target_shape, time_stretch)
                masks_list.append((resized_masks >= 0.5).astype(np.float32))
                masks_len_list.append(native_masks_len)
            else:
                masks_list.append(np.zeros((idx.size, *target_shape), dtype=np.float32))
                masks_len_list.append(np.zeros(idx.size, dtype=np.int64))
            durations_list.append(native_durations)
            spec_id_list.append(np.array([f"{session_id}_{int(i)}" for i in idx]))
            session_list.append(np.full(idx.size, session_id))
            type_list.append(np.full(idx.size, session['type']))
            count_list.append(session['mask_counts'][idx])

        all_specs = np.concatenate(specs_list)
        all_masks = np.concatenate(masks_list)
        all_masks_len = np.concatenate(masks_len_list)
        all_durations = np.concatenate(durations_list)
        all_spec_ids = np.concatenate(spec_id_list)
        all_sessions = np.concatenate(session_list)
        all_types = np.concatenate(type_list)
        all_counts = np.concatenate(count_list)
        all_condition_values = {column: np.concatenate(condition_lists[column]) for column in SUMMARY_CONDITION_COLUMNS}
        if apply_mask:
            all_specs = (all_specs * all_masks).astype(np.float32)
        elif floor is not None:
            all_specs = apply_loudness_floor(all_specs, floor)
        self.message_output(f"Loaded and resized {all_specs.shape[0]:,} spectrograms.")

        # Phase 4: split whole sessions by type, then write.
        selected_counts = {session_id: int(selected[session_id].size) for session_id in sessions if selected[session_id].size}
        train_sessions, val_sessions = split_sessions_by_type(
            list(selected_counts), session_type_by_key, selected_counts, validation_split, random_state
        )
        is_val = np.isin(all_sessions, val_sessions)
        splits = [("train_data.npz", np.flatnonzero(~is_val)), ("val_data.npz", np.flatnonzero(is_val))]
        if full_dataset:
            splits.append(("full_data.npz", np.arange(all_specs.shape[0])))
        written: dict[str, int] = {}
        for filename, split_rows in splits:
            np.savez(
                output_dir / filename,
                spectrograms=all_specs[split_rows],
                masks=all_masks[split_rows],
                masks_len=all_masks_len[split_rows].astype(np.int64),
                durations=all_durations[split_rows].astype(np.int64),
                spec_id=all_spec_ids[split_rows],
                session_id=all_sessions[split_rows],
                session_type=all_types[split_rows],
                mask_count=all_counts[split_rows].astype(np.int64),
                apply_mask=np.array(bool(apply_mask)),
                **{column: all_condition_values[column][split_rows] for column in SUMMARY_CONDITION_COLUMNS},
            )
            written[filename] = int(split_rows.size)
            self.message_output(f"  Wrote {written[filename]:,} samples -> {output_dir / filename}.")

        session_ids = list(sessions)
        h5_rows = [int(sessions[session_id]['durations'].size) for session_id in session_ids]
        corpus_rows = [int(selected[session_id].size) for session_id in session_ids]
        h5_bytes = [pathlib.Path(sessions[session_id]['h5_path']).stat().st_size for session_id in session_ids]
        with (output_dir / "SESSION_H5.sha256").open("w") as sha_file:
            for session_id in session_ids:
                sha_file.write(f"{sessions[session_id]['sha256']}  {sessions[session_id]['h5_path']}\n")
        with (output_dir / "SESSION_H5.tsv").open("w") as tsv_file:
            tsv_file.write("session\th5_rows\tcorpus_rows\tbytes\tsha256\tpath\n")
            for session_id, n_rows, n_corpus, n_bytes in zip(session_ids, h5_rows, corpus_rows, h5_bytes, strict=True):
                tsv_file.write(
                    f"{session_id}\t{n_rows}\t{n_corpus}\t{n_bytes}\t{sessions[session_id]['sha256']}\t{sessions[session_id]['h5_path']}\n"
                )

        floor_fields = {} if floor is None else {
            "floor": floor,
            "floor_rule": "x = minmax(spectrogram); x = clip((x - floor) / (1 - floor), 0, 1); baked into spectrograms",
        }
        np.savez(
            output_dir / "metadata.npz",
            root_directories=np.array([sessions[session_id]['root'] for session_id in session_ids]),
            spectrogram_h5_paths=np.array([sessions[session_id]['h5_path'] for session_id in session_ids]),
            session_ids=np.array(session_ids),
            spectrogram_h5_sha256=np.array([sessions[session_id]['sha256'] for session_id in session_ids]),
            spectrogram_h5_rows=np.array(h5_rows, dtype=np.int64),
            spectrogram_h5_corpus_rows=np.array(corpus_rows, dtype=np.int64),
            length_threshold=length_threshold,
            validation_split=validation_split,
            random_state=random_state,
            full_dataset=full_dataset,
            target_shape=np.array(target_shape),
            time_stretch=time_stretch,
            masking_type=masking_type,
            apply_mask=bool(apply_mask),
            **floor_fields,
            session_type_targets=json.dumps(session_type_targets),
            allocation="even",
            mask_sampling_mode=draw_mode,
            mask_count_bin_edges=np.array(bin_edges),
            require_mask=require_mask,
            split_by_session=True,
            session_type_by_key=json.dumps(session_type_by_key),
            type_report=json.dumps(type_report),
            split_sessions=json.dumps({"train": train_sessions, "validation": val_sessions}),
            exclude_squeaks=exclude_squeaks,
            exclude_noise=exclude_noise,
            strict_squeak_exclusion=strict_squeak_exclusion,
            reference_squeak_index_path=str(reference_squeak_index_path) if strict_squeak_exclusion else "",
            row_exclusions_override=self.row_exclusions is not None,
            excluded_by_type=json.dumps(excluded_by_type),
            n_train=written["train_data.npz"],
            n_val=written["val_data.npz"],
            n_full=written["full_data.npz"] if full_dataset else 0,
        )

        self.message_output(
            f"QLVM training-set build ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="build-qlvm-training-set")
@click.option('--root-directories', type=str, required=True, help='Comma-separated string of session root directory paths (the order sessions are drawn in).')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write the .npz training set.')
@click.option('--session-type-targets', 'session_type_targets', type=str, default=None, required=False, help='JSON object of session type -> rows (null takes the type whole), e.g. \'{"MF": 29000, "FF": 29000, "MM": null, "lone_male": null}\'; sessions of unlisted types are left out.')
@click.option('--draw-mode', 'draw_mode', type=click.Choice(['natural', 'uniform']), default=None, required=False, help='Within-session draw: uniform over rows ("natural") or equal across mask-count strata ("uniform").')
@click.option('--mask-count-bin-edges', 'mask_count_bin_edges', type=str, default=None, required=False, help='Comma-separated inclusive lower bounds of the mask-count strata above 0, the last open-ended, e.g. 1,2,3,4,5 (strata 0/1/2/3/4/5+).')
@click.option('--length-threshold', 'length_threshold', type=float, default=None, required=False, help='Drop spectrograms with duration >= threshold (time bins).')
@click.option('--require-mask/--no-require-mask', 'require_mask', default=None, required=False, help='Leave out calls without a SAM mask instance.')
@click.option('--exclude-squeaks/--no-exclude-squeaks', 'exclude_squeaks', default=None, required=False, help='Leave out rows whose USV summary call_class is squeak or both (detect-usv-squeaks), so the set holds call_class usv rows only.')
@click.option('--strict-squeak-exclusion/--no-strict-squeak-exclusion', 'strict_squeak_exclusion', default=None, required=False, help='With --exclude-squeaks, take the squeak rows from the reference squeak index (--reference-squeak-index-path) by its strict rule (is_bbv or n_bouts_min3 >= 1) instead of call_class, to reproduce the shipped sets.')
@click.option('--reference-squeak-index-path', 'reference_squeak_index_path', type=str, default=None, required=False, help='The reference squeak index CSV (session_id, seg_index, is_bbv, n_bouts_min3) read by --strict-squeak-exclusion.')
@click.option('--exclude-noise/--no-exclude-noise', 'exclude_noise', default=None, required=False, help='Leave out rows the USV summary flags as noise (detect-usv-noise).')
@click.option('--masking-type', 'masking_type', type=click.Choice(['sam', 'none']), default=None, required=False, help='Read SAM masks from the mask/<session> groups ("sam") or none ("none").')
@click.option('--apply-mask/--no-apply-mask', 'apply_mask', default=None, required=False, help='Multiply the binarized SAM mask into the stored spectrograms (masked set) or keep them unmasked.')
@click.option('--floor', 'floor', type=str, default=None, required=False, help='Loudness floor baked into unmasked spectrograms after a per-spectrogram min-max (e.g. 0.2), or "none".')
@click.option('--validation-split', 'validation_split', type=float, default=None, required=False, help='Fraction of each session type\'s rows held out (as whole sessions) for validation.')
@click.option('--random-state', 'random_state', type=int, default=None, required=False, help='Seed of the quota tie-breaks, the draw and the session split.')
@click.option('--full-dataset/--no-full-dataset', 'full_dataset', default=None, required=False, help='Take every eligible row (no draw) and also write full_data.npz.')
@click.option('--target-shape', 'target_shape', nargs=2, type=int, default=None, required=False, help='Output spectrogram (freq, time) shape as two ints, e.g. --target-shape 128 128.')
@click.option('--time-stretch/--no-time-stretch', 'time_stretch', default=None, required=False, help='Time-warp the signal window instead of center-resizing.')
@click.pass_context
def build_qlvm_training_set_cli(ctx, root_directories, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to build a QLVM USV training set (``.npz``) from a list
    of session root directories.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]
    for key, parser in (('session_type_targets', parse_session_type_targets), ('mask_count_bin_edges', parse_int_list), ('floor', parse_optional_float)):
        if key in provided_params:
            ctx.params[key] = parser(ctx.params[key])

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
        parameters_lists=['target_shape'],
        block='build_qlvm_training_set',
    )

    root_dirs = [p.strip() for p in root_directories.split(",") if p.strip()]

    QLVMTrainingSetBuilder(
        root_directories=root_dirs,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).build()
