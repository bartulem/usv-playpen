"""
@author: bartulem
Shared data-preparation helpers for the USV modeling pipelines.

This module consolidates two layers shared across the modeling pipelines:
(1) the recurring data-preparation prologue of every `extract_and_save_*`
method across the five pipeline classes (`VocalOnsetModelingPipeline`,
`BoutParameterPipeline`, `VocalCategoryModelingPipeline`,
`MultinomialModelingPipeline`, `ContinuousModelingPipeline`), and (2) the
cross-pipeline evaluation-metric helpers plus the audit-orchestration wrapper
used at modeling-input-pickle creation time. The helpers are module-level
free functions — they take the settings dictionary and the raw per-session
inputs explicitly, so they can be invoked from any pipeline without class
coupling.

Each prologue helper targets one distinct stage of the shared prologue:

A. `prepare_modeling_sessions`    — seed the RNG and load the session-path list.
C. `resolve_mouse_roles`          — derive predictor/target indices and names.
D. `select_kinematic_columns`     — pick the kinematic columns for a session
                                     using the three-bucket schema
                                     (egocentric / dyadic_pose /
                                     dyadic_engagement) with optional
                                     directional filtering for the pose bucket.
E. `build_vocal_signal_columns`   — materialize the per-mouse vocal signal
                                     columns (rate/event/categories) subject to
                                     the partner-only toggle and the
                                     self-autocorrelation guard.
F. `identify_empty_event_sessions`— list sessions where the target mouse has
                                     zero events for a given event key.
G. `harmonize_session_columns`    — dyad-rename + project-wide existence-map
                                     USV gating + zero-fill, used by the
                                     Category/Multinomial/Continuous pipelines
                                     (returns the harmonized dict and its
                                     unified suffix list).
H. `zscore_features_across_sessions`   — thin wrapper around
                                     `zscore_different_sessions_together`
                                     for API symmetry with the other helpers.
I. `pool_session_arrays`          — concatenate two-class per-session arrays
                                     (positive/negative) across a list of
                                     sessions, parameterized by the pair of
                                     dict keys (used by Onset and Category).
J. `balance_two_class_arrays`     — down-sample the majority class of a
                                     two-class dataset to match the minority
                                     class size (used by Onset and Category).
K. `unroll_history_matrix`        — reshape a `(n_samples, n_frames)` feature-
                                     history matrix into the two-column
                                     `(n_samples * n_frames, 2)` layout
                                     consumed by the pygam tensor-product
                                     spline fits (used by Onset, Bout, and
                                     Category).
L. `concat_two_class_with_labels` — vertically stack positive/negative feature
                                     arrays and emit the matching label vector
                                     (1.0 for positives, 0.0 for negatives).
                                     Shared by the Onset and Category
                                     splitters.
M. `shuffle_train_test_arrays`    — apply independent NumPy permutations to
                                     the train and test blocks at the final
                                     yield step of a split generator. Shared
                                     by the Onset and Category splitters.
N. `bounded_test_proportion`      — clamp `test_proportion` up to the minimum
                                     fraction required to keep
                                     `min_test_sessions` sessions in the test
                                     fold, used when the session count is
                                     small. Used by the Onset splitter.

Block B (`load_behavioral_feature_data`) is intentionally NOT wrapped here —
callers should import it directly from `load_input_files`.

Metrics & audit orchestration
-----------------------------
The remaining helpers are the cross-pipeline evaluation-metric functions and
the single audit-orchestration entry point:

- `brier_score_multi`          — multiclass Brier score (strictly proper
                                 scoring rule for calibration + sharpness).
- `expected_calibration_error` — top-label Expected Calibration Error (ECE)
                                 over equal-width confidence bins.
- `safe_matthews_corrcoef`     — NaN-safe wrapper around sklearn's MCC for
                                 degenerate (single-class) folds.
- `safe_confusion_matrix`      — confusion matrix with explicit `labels` so
                                 every fold yields the canonical (K, K) shape.
- `align_probs_to_canonical`   — reorder a per-fold probability matrix onto
                                 the project-wide canonical class ordering.
- `pearson_r_safe`             — Pearson's r with a zero-variance NaN guard.
- `root_mean_squared_error`    — RMSE between two 1-D arrays.
- `mean_absolute_error_1d`     — MAE between two 1-D arrays.
- `run_predictor_audits`       — non-fatal collinearity + timescale audit
                                 orchestration at modeling-input-pickle
                                 creation time.
"""

from __future__ import annotations

import itertools
import os
import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import polars as pls
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ..os_utils import configure_path
from .load_input_files import (
    is_pooled_vocal_signal,
    is_vocal_signal,
    mixture_component_count,
    mixture_ibi_threshold,
)
from .modeling_collinearity_audit import (
    audit_predictor_collinearity,
    audit_predictor_timescales,
)
from .modeling_cross_session_normalization import zscore_different_sessions_together


def prepare_modeling_sessions(modeling_settings: dict) -> list:
    """
    Loads the session-path list.

    Reads the `io.session_list_file` entry from the supplied settings
    dictionary, opens the session-list text file, and returns a list of
    configured absolute paths. Blank lines in the session-list file are
    skipped. NumPy's global RNG is not touched: every stochastic step of the
    modeling stack draws from its own generator seeded with
    `model_validation.random_seed`, so nothing depends on global state.

    Parameters
    ----------
    modeling_settings : dict
        The modeling settings dictionary. Must contain
        `modeling_settings['io']['session_list_file']`.

    Returns
    -------
    list of str
        The list of configured session-directory paths (one per line in the
        session-list text file). Guaranteed non-empty on successful return.

    Raises
    ------
    FileNotFoundError
        If the session-list file does not exist at the configured path.
    ValueError
        If the session-list file is readable but contains no non-blank lines.
    RuntimeError
        For any other I/O failure while reading the session-list file.
    """

    txt_modeling_sessions = []
    sessions_file = modeling_settings['io']['session_list_file']
    try:
        with Path(configure_path(sessions_file)).open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    txt_modeling_sessions.append(configure_path(line))
        if not txt_modeling_sessions:
            msg = "No valid session paths found."
            raise ValueError(msg)
        print(f"Loaded {len(txt_modeling_sessions)} session paths.")
    except FileNotFoundError as err:
        msg = f"Sessions list file not found: {sessions_file}"
        raise FileNotFoundError(msg) from err
    except ValueError:
        raise
    except Exception as e:
        msg = f"Error reading session paths: {e}"
        raise RuntimeError(msg) from e

    return txt_modeling_sessions


def manifold_tag_segment(manifold_column_names: list) -> str:
    """
    Description
    -----------
    Names the source of a continuous-manifold run in its analysis tag
    (``manifold_<segment>``), which the model-selection step-file prefix and the
    consolidated artifact's name carry on. The segment is the QLVM map the target
    coordinates come from: the first manifold column without its trailing digits
    (``['qlvm1', 'qlvm2']`` -> ``'qlvm'``, ``['qlvm_duration1', 'qlvm_duration2']``
    -> ``'qlvm_duration'``), so runs on different maps never share a name. It is
    NOT the category column: ``qlvm_category`` is the regular map's category for
    every map, so naming runs after it gave a ``qlvm_duration`` run and a ``qlvm``
    run the same tag. The label column, when configured, is still recorded in the
    run's ``_input_metadata.analysis_specific.usv_category_column_name``.

    Parameters
    ----------
    manifold_column_names (list)
        The configured ``vocal_features.usv_manifold_column_names``.

    Returns
    -------
    segment (str)
        The map prefix of the manifold columns.
    """

    return re.sub(r'\d+$', '', manifold_column_names[0])


def seeded_session_holdout(session_ids: list,
                           held_out_test_proportion: float,
                           random_seed: int) -> list:
    """
    Deterministically reserves a session-grain held-out test set.

    Selects a fixed subset of whole sessions to be reserved as the held-out
    test set for honest final-model evaluation. The reserved sessions are never
    used for feature nomination, cross-validation, selection, or regularisation
    tuning at any modeling stage; they are scored exactly once by the final
    (development-refit) model. Selection is deterministic: the same
    ``session_ids``, ``held_out_test_proportion``, and ``random_seed`` always
    yield the same reserved set, so every stage (univariate + selection) that
    reads the recorded set operates on an identical partition and no leakage
    can occur.

    The count of reserved sessions is ``floor(len(session_ids) *
    held_out_test_proportion)``. A proportion of ``0`` reserves nothing (the
    held-out mechanism is disabled and the pipeline behaves exactly as before).
    The reserved sessions are drawn without replacement from the sorted session
    list using a ``numpy.random.default_rng(random_seed)`` permutation, so the
    result is invariant to the input order and to the global NumPy RNG state.

    Parameters
    ----------
    session_ids : list
        The full list of session identifiers in play for this run (order
        irrelevant; the function sorts internally for reproducibility).
    held_out_test_proportion : float
        Fraction of whole sessions to reserve as the held-out test set, in
        ``[0, 1)``. ``0`` disables the holdout (returns an empty list).
    random_seed : int
        Seed for the deterministic draw. ``None`` is treated as ``0``.

    Returns
    -------
    list of str
        The sorted list of reserved (held-out) session identifiers. Empty when
        ``held_out_test_proportion`` is ``0``.

    Raises
    ------
    ValueError
        If ``held_out_test_proportion`` is not in ``[0, 1)``, or if a positive
        proportion would reserve either zero or all sessions (leaving no valid
        development set).
    """

    if not (0.0 <= held_out_test_proportion < 1.0):
        msg = (
            f"`held_out_test_proportion` must be in [0, 1); got "
            f"{held_out_test_proportion!r}."
        )
        raise ValueError(msg)

    if held_out_test_proportion == 0.0:
        return []

    sorted_sessions = sorted(session_ids)
    n_sessions = len(sorted_sessions)
    n_held_out = int(np.floor(n_sessions * held_out_test_proportion))

    if n_held_out <= 0 or n_held_out >= n_sessions:
        msg = (
            f"`held_out_test_proportion`={held_out_test_proportion} reserves "
            f"{n_held_out}/{n_sessions} sessions; it must reserve at least one "
            f"session and leave at least one development session."
        )
        raise ValueError(msg)

    rng = np.random.default_rng(int(random_seed) if random_seed is not None else 0)
    held_out_positions = rng.permutation(n_sessions)[:n_held_out]
    return sorted(sorted_sessions[pos] for pos in held_out_positions)



def held_out_session_ids_from_metadata(input_metadata: dict) -> list:
    """
    Extracts the reserved held-out session list from an ``_input_metadata`` block.

    Returns the ``held_out_session_ids`` recorded by the data-extraction stage
    (see :func:`seeded_session_holdout`). This is the single source of truth for
    the held-out partition: every downstream stage (univariate + selection) reads
    it from here so all stages share an identical reserved set and no leakage can
    occur. Backward-compatible with schema-v1 input pickles that predate the
    held-out mechanism: if the field is absent, an empty list is returned, so the
    caller simply behaves as before (no session is reserved).

    Parameters
    ----------
    input_metadata : dict
        The ``_input_metadata`` block from a modeling input / univariate pickle.

    Returns
    -------
    list
        The sorted list of reserved session identifiers, or an empty list when
        the field is absent (schema-v1) or the holdout was disabled.
    """

    if not isinstance(input_metadata, dict) or 'held_out_session_ids' not in input_metadata:
        return []
    held_out = input_metadata['held_out_session_ids']
    return list(held_out) if held_out is not None else []


def development_heldout_masks(groups, held_out_session_ids) -> tuple:
    """
    Splits a per-event session-id array into development vs held-out boolean masks.

    Given the per-row session identifiers (``groups``, one entry per event/row of
    the pooled ``X`` / ``Y``) and the reserved ``held_out_session_ids``, returns
    two boolean masks over the rows: the development mask (rows whose session is
    NOT reserved — used for all cross-validation, screening, selection, and
    regularisation tuning) and the held-out mask (rows whose session IS reserved —
    scored exactly once by the final development-refit model). When
    ``held_out_session_ids`` is empty the development mask is all-True and the
    held-out mask is all-False, so the caller behaves exactly as it did before the
    held-out mechanism existed.

    Parameters
    ----------
    groups : array-like
        Per-row session identifiers, length equal to the number of pooled rows.
    held_out_session_ids : list
        The reserved session identifiers (see :func:`seeded_session_holdout`).

    Returns
    -------
    tuple of numpy.ndarray
        ``(development_mask, held_out_mask)`` — boolean arrays of shape
        ``(len(groups),)`` that are exact complements of each other.
    """

    groups_arr = np.asarray(groups)
    if not held_out_session_ids:
        held_out_mask = np.zeros(groups_arr.shape[0], dtype=bool)
    else:
        held_out_mask = np.isin(groups_arr, np.asarray(list(held_out_session_ids)))
    development_mask = ~held_out_mask
    return development_mask, held_out_mask


def resolve_mouse_roles(modeling_settings: dict,
                        mouse_names_dict: dict,
                        session_id: str) -> tuple:
    """
    Resolves predictor/target mouse indices and names for a single session.

    The predictor mouse index is read directly from the modeling settings
    (`model_params.model_predictor_mouse_index`). The target index is the
    opposite mouse slot (i.e. `abs(predictor_idx - 1)` under the two-mouse
    assumption). The corresponding names are looked up from
    `mouse_names_dict[session_id]`.

    Parameters
    ----------
    modeling_settings : dict
        The modeling settings dictionary. Must contain
        `modeling_settings['model_params']['model_predictor_mouse_index']`.
    mouse_names_dict : dict
        Mapping from `session_id` to an ordered list of mouse track names
        (index 0 = male, index 1 = female by convention).
    session_id : str
        The session identifier used as the key into `mouse_names_dict`.

    Returns
    -------
    tuple of (int, int, str, str)
        `(predictor_idx, target_idx, predictor_name, target_name)`.
    """

    predictor_idx = modeling_settings['model_params']['model_predictor_mouse_index']
    target_idx = abs(predictor_idx - 1)
    predictor_name = mouse_names_dict[session_id][predictor_idx]
    target_name = mouse_names_dict[session_id][target_idx]
    return predictor_idx, target_idx, predictor_name, target_name


def select_kinematic_columns(session_df_columns: list,
                             target_name: str,
                             predictor_name: str,
                             kin_settings: dict,
                             predictor_idx: int) -> list:
    """
    Selects the kinematic columns to retain for one session using the
    three-bucket feature schema.

    Columns are classified by the (explicit) bucket assignment in
    `kin_settings` rather than by inferring prefix/suffix patterns at runtime.
    This avoids the ambiguity of the old `split('-')` heuristic when feature
    names themselves contain a dash (e.g. `orofacial-sei`).

    Buckets
    -------
    - `egocentric`:
        Per-mouse scalar features of the form `{mouse_id}.{base_feature}`.
        Both the target and predictor mouse columns are kept when present.

    - `dyadic_pose`:
        Directional two-mouse pose features of the form
        `{male_id}-{female_id}.{feat_male}-{feat_female}`. When
        `kin_settings['dyadic_pose_symmetric']` is False, a directional rule
        is applied to drop one of the two symmetric halves based on the
        predictor-mouse convention. The convention is "drop the
        predictor's view of allo angles, keep the predictor's view of
        TTI", which folds the male/female-symmetric pair into a single
        directional feature per dyadic-pose suffix:
          * `predictor_idx == 0`: drop if `feat_parts[0]` in
                                  (`'allo_yaw'`, `'allo_pitch'`) or
                                  `feat_parts[1] == 'TTI'`.
          * `predictor_idx != 0`: drop if `feat_parts[1]` in
                                  (`'allo_yaw'`, `'allo_pitch'`) or
                                  `feat_parts[0] == 'TTI'`.
        When `dyadic_pose_symmetric` is True, both halves are retained.

    - `dyadic_engagement`:
        Two-mouse interaction features of the form
        `{tracker_id}-{tracked_id}.{feature}` (e.g. Social Engagement Index
        suffixes such as `orofacial-sei`). Because the first mouse in the
        dyad prefix is the observer (actor), this bucket keeps only the
        orientation where the observer is the *target* mouse — i.e.
        `{target_name}-{predictor_name}.{feature}`. The reverse
        `{predictor_name}-{target_name}.{feature}` orientation is dropped.
        This preserves the "subject's (focal target's) engagement toward
        partner" reading, which is the direction expected to track the
        focal mouse's vocal output (e.g. the focal vocalizes when it is
        oriented toward and close to the partner). The
        previously-kept partner→subject direction was dropped because it
        measures the partner's gaze, which can decouple from the focal's
        gaze and produce sign-inverted correlations during chase /
        anogenital-sniffing geometries (where the partner is facing
        away while the focal is closely tracking it). The column-name
        collision concern is moot because only one direction is kept.

    Derivatives
    -----------
    For every base feature kept, the corresponding `_1st_der` and `_2nd_der`
    columns are additionally kept when the respective `include_*_derivatives`
    flag is True and the derivative column actually exists in
    `session_df_columns`. Derivatives are never added for `speed` or
    `acceleration` base features (their derivatives are not meaningful here).

    Parameters
    ----------
    session_df_columns : list of str
        The list of column names present in the session's behavioral DataFrame.
    target_name : str
        The target (self) mouse identifier.
    predictor_name : str
        The predictor (other) mouse identifier.
    kin_settings : dict
        Must contain the keys `egocentric`, `dyadic_pose`, `dyadic_engagement`
        (each a list of base-feature suffixes), `dyadic_pose_symmetric` (bool),
        `include_1st_derivatives` (bool), and `include_2nd_derivatives` (bool).
    predictor_idx : int
        The predictor mouse slot (0 for male, 1 for female).

    Returns
    -------
    list of str
        The sorted, deduplicated list of column names to retain for this
        session.
    """

    columns_to_keep = []
    session_cols_set = set(session_df_columns)

    egocentric = kin_settings['egocentric']
    dyadic_pose = kin_settings['dyadic_pose']
    dyadic_engagement = kin_settings['dyadic_engagement']
    pose_symmetric = kin_settings['dyadic_pose_symmetric']
    include_1st = kin_settings['include_1st_derivatives']
    include_2nd = kin_settings['include_2nd_derivatives']

    def _maybe_add_derivatives(feature: str, base_feature: str) -> None:
        """
        Appends the `_1st_der` / `_2nd_der` columns of `feature` to
        `columns_to_keep` when the matching `include_*_derivatives` flag is
        True and the derivative column exists in the session. Nothing is
        added for the `speed` and `acceleration` base features.

        Parameters
        ----------
        feature : str
            Full column name of the kept base feature (e.g.
            `"mouse_a.speed"`); the derivative columns are named
            `{feature}_1st_der` / `{feature}_2nd_der`.
        base_feature : str
            The base-feature suffix of `feature`, used for the
            `speed` / `acceleration` exclusion.

        Returns
        -------
        None
        """
        if base_feature in ('speed', 'acceleration'):
            return
        der_1st = f'{feature}_1st_der'
        der_2nd = f'{feature}_2nd_der'
        if include_1st and der_1st in session_cols_set:
            columns_to_keep.append(der_1st)
        if include_2nd and der_2nd in session_cols_set:
            columns_to_keep.append(der_2nd)

    for base_feature in egocentric:
        for m_name in (target_name, predictor_name):
            ego_col = f"{m_name}.{base_feature}"
            if ego_col in session_cols_set:
                columns_to_keep.append(ego_col)
                _maybe_add_derivatives(ego_col, base_feature)

    for base_feature in dyadic_pose:
        matching = [c for c in session_df_columns if c.split('.')[-1] == base_feature]
        for feature in matching:
            if not pose_symmetric:
                feat_parts = base_feature.split('-')
                if len(feat_parts) == 2:
                    if predictor_idx == 0:
                        if feat_parts[0] in ('allo_yaw', 'allo_pitch') or feat_parts[1] == 'TTI':
                            continue
                    elif feat_parts[1] in ('allo_yaw', 'allo_pitch') or feat_parts[0] == 'TTI':
                        continue
            columns_to_keep.append(feature)
            _maybe_add_derivatives(feature, base_feature)

    engagement_dyad_prefix = f"{target_name}-{predictor_name}"
    for base_feature in dyadic_engagement:
        matching = [c for c in session_df_columns if c.split('.')[-1] == base_feature]
        for feature in matching:
            dyad_prefix = feature.split('.')[0]
            if dyad_prefix != engagement_dyad_prefix:
                continue
            columns_to_keep.append(feature)
            _maybe_add_derivatives(feature, base_feature)

    return sorted(set(columns_to_keep))


def build_vocal_signal_columns(usv_data_dict: dict,
                               session_id: str,
                               target_name: str,
                               predictor_name: str,
                               voc_settings: dict,
                               usv_self_exclude: tuple = ()) -> tuple:
    """
    Constructs the list of per-mouse vocal signal columns for one session.

    Reads the continuous vocal signals previously attached to the USV data
    dictionary (by `find_onset_epochs`, `find_usv_categories`, or
    `find_variable_length_bouts`) and turns them into polars Series ready to
    be `.with_columns`-attached to a behavioral DataFrame.

    Behavior
    --------
    - When `voc_settings['vocal_predictor_type']` is falsy, returns empty lists.
    - When `voc_settings['vocal_predictor_partner_only']` is True, only the
      predictor (partner) mouse's signals are emitted.
    - For the target (self) mouse, every POOLED vocal trace ('<name>_rate' /
      '<name>_event' for any trace name, `load_input_files.is_pooled_vocal_signal`)
      is skipped, because including the subject's own aggregate rate/event trace
      would induce trivial autocorrelation and drown out behavioral predictors;
      so is any extra key in `usv_self_exclude`. Per-category signals
      ('<name>_cat_*') are otherwise allowed because they carry syntax
      information, not mere presence.

    Parameters
    ----------
    usv_data_dict : dict
        Nested dictionary from the USV loader (`session_id -> mouse_name -> ...`),
        where each mouse entry contains a `continuous_vocal_signals` sub-dict.
    session_id : str
        The session identifier.
    target_name : str
        The target (self) mouse identifier.
    predictor_name : str
        The predictor (other) mouse identifier.
    voc_settings : dict
        Must contain the keys `vocal_predictor_type` (str or None) and
        `vocal_predictor_partner_only` (bool).
    usv_self_exclude : tuple of str, optional
        Further signal keys that must NEVER be emitted for the target mouse, on
        top of its pooled traces (e.g. the binomial model's target-category
        traces). Default: `()`.

    Returns
    -------
    tuple of (list of pls.Series, list of str)
        `(new_voc_cols, new_voc_col_names)`. The Series list is suitable for
        `df.with_columns(new_voc_cols)`; the name list mirrors it so callers
        can union-extend `columns_to_keep`.
    """

    new_voc_cols = []
    new_voc_col_names = []

    voc_out_type = voc_settings['vocal_predictor_type']
    partner_only = voc_settings['vocal_predictor_partner_only']

    if not voc_out_type:
        return new_voc_cols, new_voc_col_names

    mice_to_process = [predictor_name] if partner_only else [target_name, predictor_name]

    for m_name in mice_to_process:
        is_target = (m_name == target_name)

        if m_name not in usv_data_dict[session_id]:
            continue

        vocal_signals = usv_data_dict[session_id][m_name]['continuous_vocal_signals']

        for sig_key, sig_arr in vocal_signals.items():
            if is_target and (is_pooled_vocal_signal(sig_key) or sig_key in usv_self_exclude):
                continue
            col_name = f"{m_name}.{sig_key}"
            new_voc_cols.append(pls.Series(col_name, sig_arr))
            new_voc_col_names.append(col_name)

    return new_voc_cols, new_voc_col_names


def identify_empty_event_sessions(usv_data_dict: dict,
                                  mouse_names_dict: dict,
                                  target_idx: int,
                                  event_key: str,
                                  warn_label: str = 'sessions') -> list:
    """
    Identifies sessions whose target mouse has no events for a given key.

    A session is flagged for removal when any of the following hold:
      * the session id is absent from `usv_data_dict`, or
      * the target mouse is absent from `usv_data_dict[session_id]`, or
      * `usv_data_dict[session_id][target_name][event_key]` is empty.

    Parameters
    ----------
    usv_data_dict : dict
        The USV data dictionary, keyed by `session_id` then `mouse_name`.
    mouse_names_dict : dict
        Mapping from `session_id` to an ordered list of mouse track names.
    target_idx : int
        The target mouse slot.
    event_key : str
        The event-array key under each mouse entry to check for non-emptiness
        (e.g. `'positive_events'` for onset models, `'bout_onsets'` for bout
        parameter models, `'target_events'` for category models).
    warn_label : str, optional
        A short label used in the printed warning for context (default:
        `'sessions'`).

    Returns
    -------
    list of str
        The list of session ids flagged for removal. Deterministic order
        (matches iteration order of `mouse_names_dict`).
    """

    sessions_to_remove = []
    for session_id, track_names in mouse_names_dict.items():
        target_name = track_names[target_idx]

        if session_id not in usv_data_dict:
            sessions_to_remove.append(session_id)
            continue

        if target_name not in usv_data_dict[session_id]:
            sessions_to_remove.append(session_id)
            continue

        # Explicit membership check + strict lookup, no `.get()`
        # default. A missing `event_key` here means the loader
        # wrote a different event-key set for this session (e.g.
        # different prediction-mode), in which case we want the
        # same "skip session" behavior as a present-but-empty
        # event list — record it and move on.
        if event_key not in usv_data_dict[session_id][target_name]:
            print(f"Skipping {warn_label} {session_id}: no `{event_key}` events for {target_name}.")
            sessions_to_remove.append(session_id)
            continue

        events = usv_data_dict[session_id][target_name][event_key]
        if len(events) == 0:
            print(f"Skipping {warn_label} {session_id}: 0 valid events for {target_name} ({event_key}).")
            sessions_to_remove.append(session_id)

    return sessions_to_remove


def harmonize_session_columns(processed_beh_dict: dict,
                              mouse_names_dict: dict,
                              target_idx: int,
                              predictor_idx: int) -> tuple:
    """
    Renames dyadic columns, zero-fills missing columns with project-wide
    existence gating for USV features, and returns the unified suffix list.

    This helper consolidates the column-harmonization logic shared by the
    Category, Multinomial, and Continuous pipelines. It performs three
    sequential operations across the per-session DataFrames, mutating
    `processed_beh_dict` in place:

    1. Dyad-rename: Any column whose prefix contains a dash (e.g.
       `{male_id}-{female_id}.nose-nose`) is renamed to its suffix-only
       form (`nose-nose`). After this step dyadic features appear as
       standalone columns, which is the convention these pipelines'
       downstream epoch-slicing code assumes.

    2. Existence-map construction: In the same pass, builds a
       project-wide set of `"self.{suffix}"` / `"other.{suffix}"` keys
       recording which (role, suffix) combinations were actually
       populated in at least one session. This gate is used for USV
       zero-filling so missing roles that never had any vocalizations
       do not receive spurious zero channels.

    3. Zero-fill missing columns per session:
       - Dyadic suffix (contains a dash) missing: fill as a standalone
         zero column. Ensures every session has the full dyadic
         feature set, even sessions where some dyadic pairs were empty.
       - Ego suffix + USV: fill `{mouse_id}.{suffix}` for both self and
         other roles only when the corresponding generic
         `"{role}.{suffix}"` key is present in the existence map. The
         gate prevents zero-filling USV columns for a role that never
         vocalized anywhere in the project.
       - Ego suffix + non-USV: always fill `{mouse_id}.{suffix}` for
         both self and other roles when missing. Non-USV kinematic
         absence is treated as missing-data, filled with zero so every
         session has a consistent column set for downstream z-scoring
         and epoch slicing.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping from `session_id` to a polars DataFrame of kept features.
        Modified in place.
    mouse_names_dict : dict
        Mapping from `session_id` to an ordered list of mouse track names
        (index 0 = male, index 1 = female by convention).
    target_idx : int
        The target mouse slot. Used to build `"self.{suffix}"` existence
        keys and to resolve the per-session target mouse name.
    predictor_idx : int
        The predictor mouse slot. Used to build `"other.{suffix}"`
        existence keys and to resolve the per-session predictor mouse
        name.

    Returns
    -------
    tuple of (dict, list of str)
        `(processed_beh_dict, revised_predictor_suffixes)`. The suffixes
        are the sorted union of all non-numeric column suffixes observed
        across the harmonized sessions, suitable for immediate passage
        to `zscore_features_across_sessions`.
    """

    final_suffixes = set()
    generic_existence_map = set()

    for sess_id, session_df in processed_beh_dict.items():
        df = session_df
        t_name = mouse_names_dict[sess_id][target_idx]
        p_name = mouse_names_dict[sess_id][predictor_idx]

        dyad_renames = {c: c.split('.')[-1] for c in df.columns if '-' in c.split('.')[0]}
        if dyad_renames:
            df = df.rename(dyad_renames)
            processed_beh_dict[sess_id] = df

        for col in df.columns:
            suffix = col.split('.')[-1]
            if not suffix.isdigit():
                final_suffixes.add(suffix)

            if col.startswith(f"{t_name}."):
                generic_existence_map.add(f"self.{suffix}")
            elif col.startswith(f"{p_name}."):
                generic_existence_map.add(f"other.{suffix}")

    for sess_id, df in processed_beh_dict.items():
        existing_cols = set(df.columns)
        new_zeros = []
        t_name = mouse_names_dict[sess_id][target_idx]
        p_name = mouse_names_dict[sess_id][predictor_idx]

        for suffix in final_suffixes:
            if '-' in suffix:
                if suffix not in existing_cols:
                    new_zeros.append(pls.Series(suffix, np.zeros(df.height, dtype=np.float32)))
            else:
                for prefix, m_name in [('self', t_name), ('other', p_name)]:
                    expected_col = f"{m_name}.{suffix}"
                    generic_key = f"{prefix}.{suffix}"

                    if expected_col not in existing_cols:
                        # Gate vocal suffixes ('<name>_rate' / '_event' /
                        # '_cat_<k>' for every trace name,
                        # `load_input_files.is_vocal_signal`), so a non-vocal
                        # suffix is never gated and a squeak trace is.
                        if is_vocal_signal(suffix):
                            if generic_key in generic_existence_map:
                                new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))
                        else:
                            new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))

        if new_zeros:
            processed_beh_dict[sess_id] = df.with_columns(new_zeros)

    return processed_beh_dict, sorted(final_suffixes)


def zscore_features_across_sessions(processed_beh_dict: dict,
                                    suffixes: list,
                                    feature_bounds: dict,
                                    abs_features: list | None = None,
                                    smooth_abs_features: dict | None = None,
                                    stats_out: dict | None = None,
                                    reference_sessions: list | None = None) -> dict:
    """
    Z-scores every session's feature columns using pooled cross-session statistics.

    Thin wrapper around `zscore_different_sessions_together` that fixes the
    argument shape (`data_dict`, `feature_lst`, `feature_bounds`) and makes
    the modeling_utils import surface complete. Supplied for API symmetry
    with the other helpers in this module.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping from `session_id` to a polars DataFrame of kept features.
    suffixes : list of str
        The union of feature suffixes (typically from
        `harmonize_session_columns`).
    feature_bounds : dict
        Optional per-feature clipping bounds (may be empty).
    abs_features : list of str, optional
        Feature names whose values should be folded to their absolute
        magnitude prior to z-scoring via plain `|x|`. Forwarded
        directly to `zscore_different_sessions_together`. Use for
        features whose signed distribution is spread out across the
        range; the kink at zero introduced by `|x|` is then carrying
        little data mass and pygam's penalised B-spline can fit
        cleanly. See `zscore_different_sessions_together` for the
        full rationale.
    smooth_abs_features : dict of {str: float}, optional
        Mapping `{feature_suffix: epsilon}` for features that need
        the *smooth* magnitude fold `sqrt(x² + ε²)` instead of plain
        `|x|`. Use for features whose signed distribution is sharply
        peaked at zero (e.g. `ego_yaw`, `back_yaw` — head/back held
        aligned with body most of the time): plain `|x|` would pile
        the bulk of mass right on the kink, where pygam's IRLS
        cannot reconcile the corner with its smoothness penalty and
        grinds toward `max_iter` without converging. Smooth-abs
        replaces the kink with a finite-curvature bowl while
        leaving values `|x| ≫ ε` essentially unchanged. Recommended
        ε is small relative to measurement noise: ε = 1.0° for
        ego_yaw (range ±180°), ε = 0.5° for back_yaw (range ±36°).
        Forwarded to `zscore_different_sessions_together`.

    stats_out : dict, optional
        When supplied, populated in-place with the pooled mean and
        standard deviation applied to each feature, so a caller can
        recover the native scale after the frames are overwritten.
        Forwarded to `zscore_different_sessions_together`.
    reference_sessions : list of str, optional
        The sessions the pooled statistics come from (the extraction's
        development sessions, so held-out sessions never shape the
        scale); None pools every session. Forwarded to
        `zscore_different_sessions_together`.

    Returns
    -------
    dict
        The normalized mapping (per `zscore_different_sessions_together`).
    """

    return zscore_different_sessions_together(
        data_dict=processed_beh_dict,
        feature_lst=suffixes,
        feature_bounds=feature_bounds,
        abs_features=abs_features,
        smooth_abs_features=smooth_abs_features,
        stats_out=stats_out,
        reference_sessions=reference_sessions,
    )


def session_grouped_inner_cv(groups: np.ndarray, n_folds: int) -> GroupKFold | int:
    """
    Description
    -----------
    The inner cross-validation splitter a hyperparameter search runs inside an outer
    training fold: a ``GroupKFold`` over the training rows' sessions, so every inner
    validation fold holds out whole sessions, as the outer folds do, and a penalty is
    never scored on events from sessions it was fit on. The fold count is ``n_folds``
    capped at the number of sessions. A training set drawn from a single session
    cannot be split by session; it falls back to ``n_folds`` event folds (the int
    ``GridSearchCV`` turns into ``KFold``). Pass ``groups`` to the search's ``fit``.

    Parameters
    ----------
    groups (np.ndarray)
        ``(n_rows,)`` session label of every training row.
    n_folds (int)
        The configured inner fold count.

    Returns
    -------
    cv (GroupKFold | int)
        The splitter for ``GridSearchCV(cv=...)``.
    """

    n_sessions = int(np.unique(groups).size)
    if n_sessions < 2:
        return int(n_folds)
    return GroupKFold(n_splits=int(min(n_folds, n_sessions)))


def pool_session_arrays(feature_data: dict,
                        session_list,
                        pos_key: str,
                        neg_key: str,
                        n_frames: int,
                        return_groups: bool = False) -> tuple:
    """
    Pools per-session two-class arrays for a single feature across a list of sessions.

    This helper generalizes the two-class pooling pattern used by the binary
    classification pipelines (Onset: `usv_feature_arr` / `no_usv_feature_arr`,
    Category: `target_feature_arr` / `other_feature_arr`). For every session in
    `session_list` that is present in `feature_data`, the arrays stored under
    `pos_key` and `neg_key` are collected and concatenated along axis 0. Entries
    that are `None` or empty (size 0) are silently skipped, so sessions with a
    missing class still contribute their populated class.

    Parameters
    ----------
    feature_data : dict
        The per-session data dictionary for a single feature. Each value is a
        dict that must contain `pos_key` and `neg_key` as NumPy arrays (either
        of which may be `None` or size 0).
    session_list : iterable of str
        Session identifiers to pool from. Sessions absent from `feature_data`
        are skipped without error. Any iterable is accepted (including NumPy
        arrays of session ids).
    pos_key : str
        Key under each session entry holding the positive-class array
        (e.g. `'usv_feature_arr'` for the Onset pipeline, `'target_feature_arr'`
        for the Category pipeline).
    neg_key : str
        Key under each session entry holding the negative-class array
        (e.g. `'no_usv_feature_arr'` for the Onset pipeline,
        `'other_feature_arr'` for the Category pipeline).
    n_frames : int
        The history-window length (number of frames per epoch). Used to shape
        the empty `(0, n_frames)` placeholder returned for a class that has no
        usable data across the requested sessions.
    return_groups : bool, default False
        Also return the session of every pooled row, so the caller can split its
        inner cross-validation by session.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        `(X_pos, X_neg)`, each of shape `(n_epochs, n_frames)`, followed with
        `return_groups` by `(groups_pos, groups_neg)`, the session id of every
        row of each. Either class may have `n_epochs == 0` when no non-empty
        contribution was found for it.
    """

    pos_list = []
    neg_list = []
    pos_groups = []
    neg_groups = []
    for sess_id in session_list:
        if sess_id not in feature_data:
            continue
        p_arr = feature_data[sess_id][pos_key]
        n_arr = feature_data[sess_id][neg_key]
        if p_arr is not None and p_arr.size > 0:
            pos_list.append(p_arr)
            pos_groups.append(np.full(p_arr.shape[0], sess_id, dtype=object))
        if n_arr is not None and n_arr.size > 0:
            neg_list.append(n_arr)
            neg_groups.append(np.full(n_arr.shape[0], sess_id, dtype=object))

    X_pos = np.concatenate(pos_list, axis=0) if pos_list else np.empty((0, n_frames))
    X_neg = np.concatenate(neg_list, axis=0) if neg_list else np.empty((0, n_frames))
    if not return_groups:
        return X_pos, X_neg
    groups_pos = np.concatenate(pos_groups) if pos_groups else np.empty(0, dtype=object)
    groups_neg = np.concatenate(neg_groups) if neg_groups else np.empty(0, dtype=object)
    return X_pos, X_neg, groups_pos, groups_neg


def balance_two_class_arrays(X_pos: np.ndarray,
                             X_neg: np.ndarray,
                             rng: np.random.Generator,
                             groups_pos: np.ndarray | None = None,
                             groups_neg: np.ndarray | None = None) -> tuple:
    """
    Down-samples the majority class so both classes have equal sample counts.

    Finds the minority class size `n = min(|X_pos|, |X_neg|)` and randomly
    subsamples the majority class (without replacement) down to `n` rows,
    drawing from the caller's seeded generator `rng` (never NumPy's global
    state, so a run's draws do not depend on what else consumed it). When
    either input is empty, both outputs are returned as empty arrays
    preserving their original column count. Session labels given in
    `groups_pos` / `groups_neg` follow their rows.

    Parameters
    ----------
    X_pos : np.ndarray
        Positive-class samples of shape `(n_pos, n_features)`.
    X_neg : np.ndarray
        Negative-class samples of shape `(n_neg, n_features)`.
    rng : np.random.Generator
        The seeded generator the subsample is drawn from.
    groups_pos : np.ndarray | None
        Session label of every positive row; when given (with `groups_neg`)
        the balanced labels are returned too.
    groups_neg : np.ndarray | None
        Session label of every negative row.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        `(X_pos_balanced, X_neg_balanced)`, followed with groups by
        `(groups_pos_balanced, groups_neg_balanced)`. Both classes have
        `min(n_pos, n_neg)` rows. If either input was empty the outputs are
        `(0, X_pos.shape[1])` and `(0, X_neg.shape[1])` (and empty labels).
    """

    with_groups = groups_pos is not None and groups_neg is not None
    n_pos = X_pos.shape[0]
    n_neg = X_neg.shape[0]

    if n_pos == 0 or n_neg == 0:
        empty = (np.empty((0, X_pos.shape[1])), np.empty((0, X_neg.shape[1])))
        return (*empty, np.empty(0, dtype=object), np.empty(0, dtype=object)) if with_groups else empty

    n_samples = min(n_pos, n_neg)

    if n_pos > n_samples:
        pos_indices = rng.choice(n_pos, n_samples, replace=False)
        X_pos = X_pos[pos_indices]
        if with_groups:
            groups_pos = groups_pos[pos_indices]

    if n_neg > n_samples:
        neg_indices = rng.choice(n_neg, n_samples, replace=False)
        X_neg = X_neg[neg_indices]
        if with_groups:
            groups_neg = groups_neg[neg_indices]

    if with_groups:
        return X_pos, X_neg, groups_pos, groups_neg
    return X_pos, X_neg


def unroll_history_matrix(X: np.ndarray,
                          time_indices: np.ndarray = None) -> np.ndarray:
    """
    Reshapes a feature-history matrix into the two-column layout consumed by
    pygam tensor-product splines.

    Every pygam-based runner in the modeling pipelines (the Onset, Bout, and
    Category pipelines) fits a 2-D tensor-product spline over
    `(feature_value, time_lag)`. That fit expects samples as rows, with the
    first column holding the per-lag feature value and the second column
    holding the integer (or float) time-lag index. This helper takes an input
    of shape `(n_samples, n_frames)` — where each row is the feature history
    for one epoch — and expands it into a `(n_samples * n_frames, 2)` array
    with `X[:, 0] = X_in.ravel()` (row-major, so each epoch's lags appear
    contiguously) and `X[:, 1] = tile(time_indices, n_samples)`. The output
    is `float32` to match the downstream pygam fit dtype.

    Three identical inline copies of this function previously lived in the
    pipeline modules (see the Onset, Bout, and Category `_run_model_for_*`
    methods); they have been replaced with a call to this helper.

    Parameters
    ----------
    X : np.ndarray
        Feature-history matrix of shape `(n_samples, n_frames)`. `n_frames`
        corresponds to the number of history lags used by the pipeline.
    time_indices : np.ndarray or None, optional
        Optional array of length `n_frames` providing the explicit lag index
        values placed in the second output column. When `None` (the default),
        a contiguous range `np.arange(n_frames)` is used. Callers that want
        to preserve a particular dtype (e.g. `float32` in the Onset pipeline)
        can pass the pre-built index array.

    Returns
    -------
    np.ndarray
        `float32` array of shape `(n_samples * n_frames, 2)`. Column 0 holds
        the row-major-flattened feature values; column 1 holds the tiled
        time-lag indices, repeated `n_samples` times.
    """

    n_samples, n_frames = X.shape
    if time_indices is None:
        time_indices = np.arange(n_frames)

    X_out = np.zeros((n_samples * n_frames, 2), dtype=np.float32)
    X_out[:, 0] = X.ravel()
    X_out[:, 1] = np.tile(time_indices, n_samples)
    return X_out


def concat_two_class_with_labels(X_pos: np.ndarray,
                                 X_neg: np.ndarray) -> tuple:
    """
    Vertically stacks two-class feature arrays and emits the matching label
    vector.

    This is a tiny but ubiquitous helper in the binary-classification
    splitters (Onset + Category). Given `X_pos` (shape `(n_pos, n_features)`)
    and `X_neg` (shape `(n_neg, n_features)`), it returns `X` of shape
    `(n_pos + n_neg, n_features)` together with `y` of shape
    `(n_pos + n_neg,)`, where `y = 1.0` on the first `n_pos` rows and
    `y = 0.0` on the remaining `n_neg` rows. The ordering matches the input
    vertical stack so downstream shuffle steps can be applied as a single
    permutation. No copying of `X_pos` / `X_neg` data beyond what
    `np.concatenate` performs is done by this helper.

    Parameters
    ----------
    X_pos : np.ndarray
        Positive-class samples of shape `(n_pos, n_features)`.
    X_neg : np.ndarray
        Negative-class samples of shape `(n_neg, n_features)`. The column
        count must match `X_pos`.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        `(X, y)` — `X` is the vertical concatenation of `(X_pos, X_neg)`, and
        `y` is a 1-D `float64` array with ones for the positive block
        followed by zeros for the negative block.
    """

    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    X = np.concatenate((X_pos, X_neg), axis=0)
    return X, y


def shuffle_train_test_arrays(X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              rng: np.random.Generator,
                              groups_train: np.ndarray | None = None) -> tuple:
    """
    Randomly permutes the train and test blocks of a split independently.

    After a binary-classification split generator has assembled the balanced
    train and unbalanced-or-balanced test arrays, the rows of both blocks
    are still ordered as `[positives, negatives]`. Downstream models that do
    not perform their own mini-batch shuffling (or that take the data order
    as the optimization order) benefit from a final permutation step. This
    helper draws one permutation index per block from the caller's seeded
    generator `rng` and applies it to `X` and `y` jointly (and to the
    training rows' session labels, when given) so every alignment is
    preserved.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape `(n_train, n_features)`.
    y_train : np.ndarray
        Training label vector of shape `(n_train,)`.
    X_test : np.ndarray
        Test feature matrix of shape `(n_test, n_features)`.
    y_test : np.ndarray
        Test label vector of shape `(n_test,)`.
    rng : np.random.Generator
        The seeded generator the permutations are drawn from.
    groups_train : np.ndarray | None
        Session label of every training row; when given it is permuted with
        the training rows and returned last.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray) or with a fifth np.ndarray
        The permuted `(X_train, y_train, X_test, y_test)` quadruple, followed
        by the permuted `groups_train` when it was given. Shapes are unchanged
        relative to the inputs.
    """

    train_shuffle_idx = rng.permutation(X_train.shape[0])
    test_shuffle_idx = rng.permutation(X_test.shape[0])
    shuffled = (X_train[train_shuffle_idx],
                y_train[train_shuffle_idx],
                X_test[test_shuffle_idx],
                y_test[test_shuffle_idx])
    if groups_train is None:
        return shuffled
    return (*shuffled, groups_train[train_shuffle_idx])


def null_target_permutation(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """
    Description
    -----------
    Index permutation for the label-destroying null of every behavioral modeling
    pipeline: the training rows' targets (onset labels, category labels, bout
    parameters, torus positions with their weights and regions) are shuffled
    across the WHOLE training set, pooled over its sessions, while the inputs stay
    in place. This destroys every pairing between behavior and target, both the
    moment-to-moment pairing inside a session and any difference between sessions
    (e.g. sessions with a faster partner also holding more bout onsets), so the
    null model learns nothing and a feature beats it by carrying any relationship
    to vocal output that generalizes to held-out sessions. Sessions are still kept
    whole on either side of every train / test split; only the null's target is
    pooled. The test rows are never permuted: the null is always scored against
    the true test targets.

    Parameters
    ----------
    n_rows (int)
        Number of training rows whose targets are permuted.
    rng (np.random.Generator)
        Seeded random generator the permutation is drawn from.

    Returns
    -------
    permutation (np.ndarray)
        ``(n_rows,)`` row indices; ``target[permutation]`` is the permuted target.
    """

    return rng.permutation(n_rows)


def classification_inner_cv_splits(y: np.ndarray,
                                   groups: np.ndarray,
                                   n_folds: int,
                                   random_state: int) -> list | int:
    """
    Description
    -----------
    The inner folds a classifier's regularization search (``LogisticRegressionCV``)
    runs on inside an outer training fold: ``StratifiedGroupKFold`` over the
    training rows' sessions, so every inner validation fold holds out whole sessions
    (as the outer folds do) while keeping both classes in every fold, precomputed as
    a list of ``(train, validation)`` index pairs because ``LogisticRegressionCV.fit``
    takes no ``groups``. The fold count is ``n_folds`` capped at the number of
    sessions. A training set from a single session cannot be split by session, and
    session folds that would leave a class out of an inner training or validation
    fold cannot be scored; both fall back to ``n_folds`` stratified event folds (the
    int ``cv`` argument).

    Parameters
    ----------
    y (np.ndarray)
        ``(n_rows,)`` class label of every training row.
    groups (np.ndarray)
        ``(n_rows,)`` session label of every training row.
    n_folds (int)
        The configured inner fold count.
    random_state (int)
        Seed of the fold assignment.

    Returns
    -------
    cv (list | int)
        The ``cv`` argument for ``LogisticRegressionCV``.
    """

    n_sessions = int(np.unique(groups).size)
    if n_sessions < 2:
        return int(n_folds)
    splitter = StratifiedGroupKFold(n_splits=int(min(n_folds, n_sessions)), shuffle=True,
                                    random_state=int(random_state))
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    n_classes = np.unique(y).size
    if any(np.unique(y[train]).size < n_classes or np.unique(y[validation]).size < n_classes
           for train, validation in splits):
        return int(n_folds)
    return splits

def bounded_test_proportion(test_proportion: float,
                            n_sessions: int,
                            min_test_sessions: int = 1) -> float:
    """
    Clamps `test_proportion` up to the minimum fraction required to retain
    at least `min_test_sessions` sessions in the test fold.

    In session-held-out splitting (used by the Onset `session` strategy),
    the raw `test_proportion` read from
    the modeling settings can correspond to a test fold size below one
    session when the total session count is small — e.g. `test_proportion
    = 0.2` across 3 sessions rounds down to 0 test sessions under
    `ShuffleSplit`. This helper raises the effective test proportion to
    `min_test_sessions / n_sessions` when necessary. When `n_sessions == 0`
    the helper returns the raw `test_proportion` unchanged (the caller is
    expected to short-circuit in that case).

    Parameters
    ----------
    test_proportion : float
        The nominal test proportion read from the modeling settings
        (must be in `(0, 1)`).
    n_sessions : int
        The total number of sessions available for splitting.
    min_test_sessions : int, optional
        The minimum number of sessions that must end up in the test fold.
        Defaults to `1`.

    Returns
    -------
    float
        The effective test proportion to pass to `ShuffleSplit`. This is
        `max(test_proportion, min_test_sessions / n_sessions)` when
        `n_sessions > 0`, or `test_proportion` itself when
        `n_sessions == 0`.
    """

    if n_sessions <= 0:
        return test_proportion
    return max(test_proportion, min_test_sessions / n_sessions)


def paired_one_se_improvement(candidate_fold_scores,
                              incumbent_fold_scores,
                              higher_is_better: bool = True) -> tuple[float, float]:
    """
    Computes the mean and standard error of a candidate model's PER-FOLD
    improvement over the incumbent, for use in the forward-selection 1SE rule.

    The 1SE acceptance rule asks whether a candidate improves on the incumbent
    by more than one standard error. The quantity under test is therefore a
    *difference* of two scores measured on the same folds, and its uncertainty
    is the standard error of the per-fold differences — not the standard error
    of either score on its own.

    That distinction is large, not cosmetic. Fold-to-fold spread in an absolute
    score is dominated by how hard each test fold is (its event count, its
    animal, its base rate), and that difficulty term is shared by the incumbent
    and the candidate because they are scored on the same folds. Since the two
    models are nested — the candidate is the incumbent plus one feature — their
    per-fold scores are strongly correlated, and

        Var(d) = Var(a) + Var(b) - 2 * Cov(a, b) = 2 * sigma^2 * (1 - rho)

    so the paired standard error is smaller than the unpaired one by a factor of
    `sqrt(2 * (1 - rho))`: about 7x at `rho = 0.99` and about 70x at
    `rho = 0.9999`. Using an unpaired standard error therefore sets the
    acceptance bar far too high and truncates the forward search early. A
    feature that improves every single fold by an identical amount — the most
    reliable improvement obtainable — is rejected under the unpaired rule purely
    because the sessions differ in difficulty.

    Folds are paired by position, so both sequences must be ordered by fold and
    have the same length. A fold is used only when BOTH models scored it
    finitely; folds where either model failed carry no information about the
    difference and are dropped as a pair.

    A constant baseline (a chance floor, or a checkpoint that predates per-fold
    baseline storage) may be passed as a scalar. Pairing against a constant is
    exactly the unpaired case — `d_k = a_k - c` has the same spread as `a_k` —
    so the scalar branch is the correct answer for those callers rather than a
    fallback.

    Parameters
    ----------
    candidate_fold_scores (Sequence[float] | np.ndarray)
        The candidate model's score on each CV fold, ordered by fold.
    incumbent_fold_scores (Sequence[float] | np.ndarray | float)
        The incumbent model's score on each CV fold, ordered identically, or a
        float for a constant baseline.
    higher_is_better (bool)
        Whether a larger score is a better score. `True` for D^2, AUC and R^2;
        `False` for a log-loss / negative-log-likelihood score. Defaults to
        `True`.

    Returns
    -------
    improvement (float)
        The mean per-fold improvement of the candidate over the incumbent,
        signed so that a positive value always means the candidate is better.
        `nan` when no fold is usable.
    standard_error (float)
        The standard error of that per-fold improvement,
        `std(d, ddof=1) / sqrt(n)`. `0.0` when fewer than two folds are usable,
        matching the existing guard against `np.std([x], ddof=1)` returning
        `nan` and poisoning every subsequent acceptance test.
    """

    candidate = np.asarray(candidate_fold_scores, dtype=float).ravel()

    if np.ndim(incumbent_fold_scores) == 0:
        incumbent = np.full(candidate.shape, float(incumbent_fold_scores))
    else:
        incumbent = np.asarray(incumbent_fold_scores, dtype=float).ravel()

    if candidate.size != incumbent.size:
        msg = (f"paired 1SE needs one score per fold from both models, "
               f"got {candidate.size} candidate and {incumbent.size} incumbent folds")
        raise ValueError(msg)

    usable = np.isfinite(candidate) & np.isfinite(incumbent)
    if not np.any(usable):
        return float('nan'), 0.0

    difference = candidate[usable] - incumbent[usable]
    if not higher_is_better:
        difference = -difference

    improvement = float(np.mean(difference))
    standard_error = (float(np.std(difference, ddof=1) / np.sqrt(difference.size))
                      if difference.size > 1 else 0.0)
    return improvement, standard_error


def valid_window_starts(event_ends: np.ndarray,
                        history_frames: int,
                        n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Start frame of the history window preceding every event, keeping only the
    events whose full window lies inside the recording. A window spans the
    ``history_frames`` frames ending at the event frame (exclusive), so it starts at
    ``end - history_frames`` and must satisfy ``0 <= start`` and ``end <= n_frames``.
    Every extraction cuts its windows with this rule and stores the kept starts, and
    the per-step shift test re-cuts the same windows from a shifted trace at the
    same starts, so the rule lives here once.

    Parameters
    ----------
    event_ends (np.ndarray)
        ``(n_events,)`` integer frame index at which each window ends (exclusive):
        ``round(time_seconds * fps)`` for a time-stamped event, or the onset frame.
    history_frames (int)
        Window length in frames.
    n_frames (int)
        Length of the feature trace the windows are cut from.

    Returns
    -------
    starts (np.ndarray)
        ``(n_valid,)`` int start frames of the kept windows, in event order.
    valid_mask (np.ndarray)
        ``(n_events,)`` bool, True for the events whose window was kept.
    """

    ends = np.asarray(event_ends).astype(int)
    starts = ends - int(history_frames)
    valid_mask = (starts >= 0) & (ends <= int(n_frames))
    return starts[valid_mask].astype(int), valid_mask


def cut_history_windows(column: np.ndarray,
                        starts: np.ndarray,
                        history_frames: int) -> np.ndarray:
    """
    Description
    -----------
    Cuts the ``history_frames``-long window starting at every start frame out of one
    feature trace, with NaN replaced by 0 (the rule every extraction applies; ``inf``
    is left as it is, as the onset and manifold extractions always did). Row
    ``i`` equals ``column[starts[i] : starts[i] + history_frames]`` after the NaN fill;
    the windows are gathered in one fancy-index on a sliding view and copied, so the
    result is identical to a per-event slice loop. The extractions store the windows
    this function returns, and the per-step shift test re-cuts them from a circularly
    shifted copy of the same trace at the same starts; a lag of zero therefore
    reproduces the stored windows exactly (``verify_windows_against_traces``).

    Parameters
    ----------
    column (np.ndarray)
        ``(n_frames,)`` feature trace of one session (any float dtype).
    starts (np.ndarray)
        ``(n_windows,)`` int start frames; every ``start + history_frames`` must be
        ``<= n_frames`` and every start ``>= 0`` (``valid_window_starts``).
    history_frames (int)
        Window length in frames.

    Returns
    -------
    windows (np.ndarray)
        ``(n_windows, history_frames)`` array in the trace's dtype; ``(0,
        history_frames)`` when there is no start.

    Raises
    ------
    ValueError
        If a start lies outside the trace.
    """

    column = np.asarray(column)
    starts = np.asarray(starts, dtype=int)
    history_frames = int(history_frames)
    if starts.size == 0:
        return np.empty((0, history_frames), dtype=column.dtype)
    if starts.min() < 0 or starts.max() + history_frames > column.shape[0]:
        msg = (f"cut_history_windows: a window start lies outside the trace "
               f"(starts in [{starts.min()}, {starts.max()}], history {history_frames}, "
               f"trace length {column.shape[0]}).")
        raise ValueError(msg)
    clean = np.nan_to_num(column, nan=0.0, posinf=np.inf, neginf=-np.inf)
    windows = sliding_window_view(clean, history_frames)
    return windows[starts].copy()


def circular_shift_trace(column: np.ndarray, lag_frames: int) -> np.ndarray:
    """
    Description
    -----------
    Circularly shifts one session's feature trace by ``lag_frames``: frame ``t`` of the
    result holds frame ``t - lag`` of the input, and the segment pushed past the end
    wraps to the start. The trace's marginal distribution and autocorrelation are
    preserved, its alignment with the session's event times is broken, and one
    discontinuity is introduced at the wrap point. This is the null of the per-step
    shift test (Harris 2020; Nevjen & Dunn 2024).

    Parameters
    ----------
    column (np.ndarray)
        ``(n_frames,)`` feature trace.
    lag_frames (int)
        Shift in frames (positive).

    Returns
    -------
    shifted (np.ndarray)
        The shifted trace, same shape and dtype.
    """

    return np.roll(np.asarray(column), int(lag_frames))


def draw_session_shift_lags(rng: np.random.Generator,
                            session_lengths: dict,
                            min_lag_frames: int) -> dict:
    """
    Description
    -----------
    One circular-shift lag per session for one null draw of the per-step shift test,
    drawn uniformly from ``{min_lag, ..., T_s - min_lag}`` frames where ``T_s`` is the
    session's trace length, independently across sessions (sessions are independent
    recordings). The lower bound keeps the shifted trace at least ``min_lag`` frames
    away from its true position in both directions of the circle.

    Parameters
    ----------
    rng (np.random.Generator)
        Seeded generator of this draw.
    session_lengths (dict)
        ``{session_id: n_frames}`` for every session that is shifted, in the order the
        lags are drawn (sorted session ids give a reproducible draw).
    min_lag_frames (int)
        Smallest admissible lag in frames.

    Returns
    -------
    lags (dict)
        ``{session_id: lag_frames}``.

    Raises
    ------
    ValueError
        If a session is shorter than ``2 * min_lag_frames`` (no admissible lag).
    """

    min_lag_frames = int(min_lag_frames)
    if min_lag_frames < 1:
        msg = f"draw_session_shift_lags: min_lag_frames must be >= 1, got {min_lag_frames}."
        raise ValueError(msg)
    lags = {}
    for session_id, session_frames in session_lengths.items():
        n_frames = int(session_frames)
        high = n_frames - min_lag_frames
        if high < min_lag_frames:
            msg = (f"draw_session_shift_lags: session {session_id!r} has {n_frames} frames, "
                   f"shorter than 2 x min_lag ({2 * min_lag_frames}); no admissible lag.")
            raise ValueError(msg)
        lags[session_id] = int(rng.integers(min_lag_frames, high + 1))
    return lags


def shift_refit_count(m_candidates: int, alpha: float) -> int:
    """
    Description
    -----------
    Number of shifted refits ``B`` the per-step test runs: the smallest ``B`` for which
    a Bonferroni-corrected permutation p-value over ``m`` candidates can fall below
    ``alpha`` at all. With ``p = (1 + k) / (B + 1)`` and ``p_adj = m * p``, passing
    needs ``m * (1 + k) / (B + 1) < alpha``; ``B = floor(m / alpha)`` makes that
    possible exactly when ``k = 0`` (no shifted refit reaches the observed statistic),
    so the step is decided by the first exceedance and accepted steps run all ``B``.

    Parameters
    ----------
    m_candidates (int)
        Number of candidates the step chose its winner from (``>= 1``).
    alpha (float)
        Level of the test (``0 < alpha < 1``).

    Returns
    -------
    n_refits (int)
        ``floor(m / alpha)``.
    """

    m_candidates = int(m_candidates)
    alpha = float(alpha)
    if m_candidates < 1 or not (0.0 < alpha < 1.0):
        msg = f"shift_refit_count: need m >= 1 and 0 < alpha < 1, got m={m_candidates}, alpha={alpha}."
        raise ValueError(msg)
    return int(np.floor(m_candidates / alpha))


def bonferroni_permutation_p(k_exceed: int, n_draws: int, m_candidates: int) -> tuple[float, float]:
    """
    Description
    -----------
    Monte-Carlo permutation p-value ``(1 + k) / (n + 1)`` (the observed statistic
    counted among the draws, exact at every ``n``) and its Bonferroni correction over
    the ``m`` candidates the winner was chosen from, ``min(1, m * p)``.

    Parameters
    ----------
    k_exceed (int)
        Number of draws whose statistic reached the observed one (``>=``).
    n_draws (int)
        Number of draws run.
    m_candidates (int)
        Number of candidates at the step.

    Returns
    -------
    p (float)
        The uncorrected permutation p-value.
    p_adjusted (float)
        ``min(1, m * p)``.
    """

    p = (1.0 + int(k_exceed)) / (int(n_draws) + 1.0)
    return float(p), float(min(1.0, int(m_candidates) * p))


def macro_log_likelihood(y_true: np.ndarray,
                         y_proba: np.ndarray,
                         classes: np.ndarray) -> float:
    """
    Description
    -----------
    Class-balanced (macro) mean log-likelihood per event of a categorical predictor:
    the mean over the classes present in ``y_true`` of each class's mean
    ``log p(true class)``. Every class weighs the same whatever its prevalence, which
    is what the multinomial pipeline's macro AUC also does; unlike AUC it is a proper
    scoring rule on the likelihood scale, so it is the statistic of the multinomial
    pipeline's per-step shift test. Probabilities are clipped to ``[1e-15, 1 -
    1e-15]`` before the log, as the pipeline's log-loss is.

    Parameters
    ----------
    y_true (np.ndarray)
        ``(n,)`` true class labels.
    y_proba (np.ndarray)
        ``(n, n_classes)`` predicted probabilities, columns in the order of
        ``classes``.
    classes (np.ndarray)
        ``(n_classes,)`` class labels matching the probability columns.

    Returns
    -------
    macro_ll (float)
        Mean over present classes of the class-mean log-probability of the true
        class (``<= 0``; higher is better).
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    classes = np.asarray(classes)
    if y_proba.ndim != 2 or y_proba.shape[1] != classes.shape[0]:
        msg = (f"macro_log_likelihood: y_proba must be (n, {classes.shape[0]}), "
               f"got {y_proba.shape}.")
        raise ValueError(msg)
    clipped = np.clip(y_proba, 1e-15, 1.0 - 1e-15)
    per_class = []
    for column, label in enumerate(classes):
        rows = y_true == label
        if not np.any(rows):
            continue
        per_class.append(float(np.mean(np.log(clipped[rows, column]))))
    if not per_class:
        msg = "macro_log_likelihood: no row of y_true matches any class."
        raise ValueError(msg)
    return float(np.mean(per_class))


def pool_worker_count(requested_workers: int,
                      n_terms: int,
                      memory_gb: float,
                      peak_gb_base: float,
                      peak_gb_per_term: float) -> int:
    """
    Description
    -----------
    Number of worker processes the selection pool may run at once: the requested
    count (``0`` = every CPU of the SLURM task, or every CPU of the machine off the
    cluster), capped by the memory one fit of the current model size peaks at.
    The peak is modelled as ``peak_gb_base + peak_gb_per_term * n_terms`` from the
    measured pygam peaks (13 GB at 3 terms, 36 GB at 9 terms); the memory budget is
    ``memory_gb`` (``0`` = the SLURM job's memory, or no cap off the cluster).

    Parameters
    ----------
    requested_workers (int)
        ``model_params.selection_pool_workers`` (0 = all available CPUs).
    n_terms (int)
        Number of tensor terms of the models the pool is about to fit.
    memory_gb (float)
        ``model_params.selection_pool_memory_gb`` (0 = read ``SLURM_MEM_PER_NODE``,
        or the machine's physical memory when that is unset).
    peak_gb_base (float)
        ``model_params.selection_fit_peak_gb_base``.
    peak_gb_per_term (float)
        ``model_params.selection_fit_peak_gb_per_term``.

    Returns
    -------
    n_workers (int)
        At least 1.
    """

    requested_workers = int(requested_workers)
    if requested_workers <= 0:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        requested_workers = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    memory_gb = float(memory_gb)
    if memory_gb <= 0.0:
        slurm_mem = os.environ.get('SLURM_MEM_PER_NODE')
        if slurm_mem:
            memory_gb = float(slurm_mem) / 1024.0
        else:
            # Off the cluster the budget is the machine's physical memory: with no
            # cap, every CPU would get a worker and a 13-36 GB fit each.
            memory_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024.0 ** 3
    peak = float(peak_gb_base) + float(peak_gb_per_term) * max(int(n_terms), 1)
    by_memory = int(np.floor(memory_gb / peak))
    return max(1, min(requested_workers, by_memory))


#: Reserved group of the trace sidecar that holds the per-session window start frames
#: (no behavioral feature key starts with an underscore).
WINDOW_STARTS_GROUP = '_window_starts'

#: Keys of a window-start entry that are session attributes, not start-frame slots.
_WINDOW_STARTS_ATTRS = ('n_frames', 'fps')


def write_feature_traces(path: str | Path,
                         traces: dict,
                         window_starts: dict) -> None:
    """
    Description
    -----------
    Writes the per-step shift test's inputs into an HDF5 sidecar next to the modeling
    pickle: the processed per-session feature traces an extraction cut its windows
    from, one dataset per ``/<feature>/<session>`` in the trace's own dtype, and the
    start frame of every kept window, one dataset per
    ``/_window_starts/<session>/<slot>`` under the pipeline's slot names, with the
    trace length and the camera frame rate as attributes of every trace dataset and of
    every session's start group. The windows themselves stay in the pickle, so every
    reader of the pickle is unchanged, and the pickle's ``_input_metadata`` records
    only the sidecar's path: that block stays JSON-like, as the consolidators compare
    it leaf by leaf with ``!=`` and an array-valued leaf would break them. The
    per-step shift test reads one feature's traces back (``read_feature_trace``) and
    re-cuts shifted windows at the stored starts (``read_window_starts``).

    Parameters
    ----------
    path (str | Path)
        Destination ``.h5`` file (overwritten).
    traces (dict)
        ``{feature: {session_id: (n_frames,) np.ndarray}}``.
    window_starts (dict)
        ``{session_id: {slot_name: (n,) start frames, 'n_frames': int, 'fps': float}}``
        for every session a trace is written for (``usv`` / ``no_usv``, ``target`` /
        ``other`` or ``X`` slots).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a feature key equals the reserved start group's name, or a trace's session
        has no window-start entry.
    """

    path = Path(path)
    if WINDOW_STARTS_GROUP in traces:
        msg = f"write_feature_traces: the feature key {WINDOW_STARTS_GROUP!r} is reserved for the window starts."
        raise ValueError(msg)
    with h5py.File(path, 'w') as handle:
        for feature, by_session in traces.items():
            group = handle.create_group(feature)
            for session_id, column in by_session.items():
                if session_id not in window_starts:
                    msg = f"write_feature_traces: {feature!r} has a trace for {session_id!r} but no window starts."
                    raise ValueError(msg)
                dataset = group.create_dataset(str(session_id), data=np.asarray(column))
                dataset.attrs['n_frames'] = int(window_starts[session_id]['n_frames'])
                dataset.attrs['fps'] = float(window_starts[session_id]['fps'])
        starts_group = handle.create_group(WINDOW_STARTS_GROUP)
        for session_id, entry in window_starts.items():
            session_group = starts_group.create_group(str(session_id))
            session_group.attrs['n_frames'] = int(entry['n_frames'])
            session_group.attrs['fps'] = float(entry['fps'])
            for slot_name, starts in entry.items():
                if slot_name in _WINDOW_STARTS_ATTRS:
                    continue
                session_group.create_dataset(str(slot_name), data=np.asarray(starts, dtype=np.int64))


def read_window_starts(path: str | Path) -> dict:
    """
    Description
    -----------
    Reads the per-session window start frames back from the HDF5 sidecar written by
    ``write_feature_traces``, in the layout the extractions built them in.

    Parameters
    ----------
    path (str | Path)
        The sidecar file.

    Returns
    -------
    window_starts (dict)
        ``{session_id: {slot_name: (n,) int64 start frames, 'n_frames': int, 'fps': float}}``.

    Raises
    ------
    ValueError
        If the sidecar holds no window-start group (written before the starts moved
        into it).
    """

    window_starts = {}
    with h5py.File(Path(path), 'r') as handle:
        if WINDOW_STARTS_GROUP not in handle:
            msg = (f"{path} holds no {WINDOW_STARTS_GROUP!r} group: the sidecar predates the window starts "
                   f"being stored in it. Rebuild the modeling input pickle with the current extraction.")
            raise ValueError(msg)
        for session_id, session_group in handle[WINDOW_STARTS_GROUP].items():
            entry = {slot_name: session_group[slot_name][()] for slot_name in session_group}
            entry['n_frames'] = int(session_group.attrs['n_frames'])
            entry['fps'] = float(session_group.attrs['fps'])
            window_starts[session_id] = entry
    return window_starts


def read_feature_trace(path: str | Path, feature: str, session_id: str) -> np.ndarray:
    """
    Description
    -----------
    Reads one session's processed trace of one feature from the HDF5 sidecar written by
    ``write_feature_traces``.

    Parameters
    ----------
    path (str | Path)
        The sidecar file.
    feature (str)
        Generic feature key (``'self.speed'``, ``'head-head'``, ...).
    session_id (str)
        Session id.

    Returns
    -------
    column (np.ndarray)
        ``(n_frames,)`` trace.
    """

    with h5py.File(Path(path), 'r') as handle:
        return handle[feature][str(session_id)][()]


def verify_windows_against_traces(feature_data: dict,
                                  traces_path: str | Path,
                                  window_starts: dict,
                                  slots: dict,
                                  history_frames: int,
                                  features: list,
                                  sessions: list) -> None:
    """
    Description
    -----------
    The lag-zero self-check of the per-step shift test: re-cuts every stored window of
    every listed feature and session from the trace sidecar at the stored start
    frames and requires bit-exact equality with the windows in the pickle. A mismatch
    means the sidecar, the starts and the windows do not describe the same extraction
    and the run aborts before any model is fit.

    Parameters
    ----------
    feature_data (dict)
        The modeling pickle's feature dict ``{feature: {session: {slot: windows}}}``
        (metadata blocks stripped).
    traces_path (str | Path)
        The HDF5 sidecar.
    window_starts (dict)
        ``{session: {slot_name: (n,) start frames}}`` from the sidecar (``read_window_starts``).
    slots (dict)
        ``{slot_name: array_key}`` mapping each start list to the pickle key holding
        its windows (``{'usv': 'usv_feature_arr', 'no_usv': 'no_usv_feature_arr'}``).
    history_frames (int)
        Window length in frames.
    features (list)
        Features to check.
    sessions (list)
        Sessions to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        On the first window that does not reproduce.
    """

    with h5py.File(Path(traces_path), 'r') as handle:
        for feature in features:
            for session_id in sessions:
                if session_id not in feature_data[feature]:
                    continue
                column = handle[feature][str(session_id)][()]
                for slot_name, array_key in slots.items():
                    stored = feature_data[feature][session_id][array_key]
                    starts = np.asarray(window_starts[session_id][slot_name], dtype=int)
                    if stored is None or stored.size == 0:
                        if starts.size != 0:
                            msg = (f"verify_windows_against_traces: {feature!r} / {session_id!r} / "
                                   f"{slot_name!r} stores no windows but {starts.size} start frames.")
                            raise ValueError(msg)
                        continue
                    rebuilt = cut_history_windows(column, starts, history_frames)
                    if rebuilt.shape != stored.shape or not np.array_equal(rebuilt, stored):
                        msg = (f"verify_windows_against_traces: {feature!r} / {session_id!r} / "
                               f"{slot_name!r}: windows re-cut from the trace sidecar at lag 0 do not "
                               f"match the stored windows (shapes {rebuilt.shape} vs {stored.shape}). "
                               f"The pickle, its start frames and {traces_path} are not from one extraction.")
                        raise ValueError(msg)


def brier_score_multi(y_true: np.ndarray,
                      y_proba: np.ndarray,
                      classes: np.ndarray) -> float:
    """
    Computes the multiclass Brier score — a strictly proper scoring rule that
    measures both calibration and sharpness of predicted class probabilities.

    The Brier score is the mean over samples of the squared L2 distance between
    the predicted probability vector and the one-hot encoded true label:

        BS = (1 / N) * sum_i || y_proba_i - onehot(y_true_i) ||_2^2

    It evaluates to `0` for perfectly calibrated and sharp predictions and is
    upper-bounded by `2` in the multiclass case. Lower is better. Unlike
    log-loss, the Brier score remains finite for zero-probability predictions
    on observed classes, which makes it more robust when the estimator is
    occasionally overconfident. Use it as a complement to log-loss: log-loss
    rewards sharp correct predictions harshly, whereas Brier penalizes large
    errors quadratically and smaller ones linearly.

    The helper supports binary problems too: pass `y_proba` of shape (N, 2)
    and `classes` of length 2. It computes an identical quantity to
    `sklearn.metrics.brier_score_loss` on the positive-class column (up to a
    factor of 2 for the two-column parameterization).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels of shape (N,). Values must appear in
        `classes`.
    y_proba : np.ndarray
        Predicted class probabilities of shape (N, K), one row per sample and
        one column per class, in the same order as `classes`.
    classes : np.ndarray
        Ordered array of the K class labels, matching the column ordering of
        `y_proba`.

    Returns
    -------
    float
        The mean Brier score across samples.
    """

    y_true_arr = np.asarray(y_true)
    classes_arr = np.asarray(classes)
    onehot = (y_true_arr[:, None] == classes_arr[None, :]).astype(np.float64)
    return float(np.mean(np.sum((np.asarray(y_proba, dtype=np.float64) - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: np.ndarray,
                               *,
                               n_bins: int) -> float:
    """
    Computes the top-label Expected Calibration Error (ECE).

    ECE partitions the predicted top-class confidences into `n_bins`
    equal-width bins on [0, 1]. Within each bin it compares the empirical
    accuracy (fraction of samples whose predicted label matches the true
    label) against the mean predicted confidence, and reports the weighted
    mean absolute gap:

        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    A perfectly calibrated classifier has `ECE = 0`: samples it predicts with
    70% confidence are correct 70% of the time. Values > 0 indicate
    systematic miscalibration (typically overconfidence). Use ECE alongside
    log-loss and Brier to distinguish calibration error (what ECE measures)
    from raw predictive accuracy.

    This is the "top-label" variant (aka "ECE-MaxProb"); for multiclass
    problems it ignores the confidence assigned to non-predicted classes.
    Binary classifiers can be scored by passing `y_proba` of shape (N, 2)
    (e.g., `np.column_stack([1 - p_pos, p_pos])`).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels of shape (N,).
    y_pred : np.ndarray
        Predicted class labels of shape (N,) — typically `argmax(y_proba)`.
    y_proba : np.ndarray
        Predicted class probabilities of shape (N, K). The top-class
        confidence is taken as `np.max(y_proba, axis=1)`.
    n_bins : int
        Number of equal-width confidence bins. Keyword-only, no default:
        callers pass `diagnostics.ece_n_bins` from the run's settings.

    Returns
    -------
    float
        The top-label expected calibration error on [0, 1], or `NaN` when
        `y_true` is empty (`n_total == 0`).
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim != 2:
        msg = f"expected_calibration_error requires y_proba of shape (N, K); got {y_proba.shape}."
        raise ValueError(msg)

    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = float(len(y_true))
    if n_total == 0:
        return float('nan')

    ece = 0.0
    for lo, hi in itertools.pairwise(bin_edges):
        if hi == 1.0:
            mask = (confidences > lo) & (confidences <= hi + 1e-9)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        n_bin = int(np.sum(mask))
        if n_bin == 0:
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(confidences[mask]))
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)
    return float(ece)


def safe_matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Wraps `sklearn.metrics.matthews_corrcoef` with NaN-safe degenerate-case
    handling.

    Matthews Correlation Coefficient (MCC) is a chance-corrected, imbalance-
    robust summary of a confusion matrix. It returns values in [-1, +1]:
    `+1` for perfect prediction, `0` for performance indistinguishable from
    random, and negative values for systematic disagreement. MCC is
    recommended by the machine-learning literature as the best single
    number to summarize multi-class imbalanced classifiers because it
    cannot be inflated by a model that only predicts the majority class.

    If both `y_true` and `y_pred` collapse to a single class (i.e., every
    entry of the confusion matrix is zero except one cell), MCC is
    mathematically undefined; scikit-learn already returns `0.0` in that
    degenerate case (and for empty inputs) without raising, and that `0.0`
    is passed through unchanged. The behavior this wrapper actually *adds*
    is to catch the `ValueError` that scikit-learn raises for inputs it
    rejects outright and return `NaN` instead, so a single bad fold yields
    a missing value rather than aborting the surrounding metric loop.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels of shape (N,).
    y_pred : np.ndarray
        Predicted class labels of shape (N,).

    Returns
    -------
    float
        The Matthews correlation coefficient, or `NaN` when scikit-learn
        raises a `ValueError` on the supplied inputs.
    """

    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        return float('nan')


def safe_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: np.ndarray) -> np.ndarray:
    """
    Wraps `sklearn.metrics.confusion_matrix` with explicit `labels` so the
    returned matrix always has the canonical shape (K, K) even when a fold
    happens to not observe every class.

    Storing the per-fold confusion matrix is cheap and makes the saved
    `.pkl` self-sufficient for downstream diagnostics: reviewers can
    re-derive precision, recall, F1, sensitivity, and specificity on demand,
    and quickly spot pathological failure modes (e.g., class collapse onto
    the majority label) without re-running the model.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels of shape (N,).
    y_pred : np.ndarray
        Predicted class labels of shape (N,).
    labels : np.ndarray
        Canonical class ordering to enforce on the rows and columns of the
        returned matrix.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (K, K) with `labels` ordering. Rows are
        true classes; columns are predicted classes.
    """

    return confusion_matrix(y_true, y_pred, labels=labels)


def align_probs_to_canonical(probabilities: np.ndarray,
                             model_classes: np.ndarray,
                             canonical_classes: np.ndarray) -> np.ndarray:
    """
    Reorders a probability matrix so its columns line up with the
    project-wide canonical class ordering.

    When a fold happens to miss a rare class, `model.classes_` is a
    subset of `canonical_classes`, and naïvely stacking the per-fold
    `model.predict_proba` outputs would silently shift column indices
    across folds. This helper builds a `(N, K_canonical)` matrix where
    every column `k` holds the probability for class
    `canonical_classes[k]` if the model trained on it, and zero
    otherwise — enabling downstream Brier / ECE / cross-fold stacking
    to compare apples to apples.

    Parameters
    ----------
    probabilities : np.ndarray
        Per-row class probabilities as returned by the estimator, shape
        `(n_samples, len(model_classes))`.
    model_classes : np.ndarray
        The classes the estimator actually trained on, in the column
        order of `probabilities`. Typically `model.classes_`.
    canonical_classes : np.ndarray
        Project-wide class ordering. Defines the column layout of the
        returned matrix.

    Returns
    -------
    probs_canonical : np.ndarray
        Reordered probability matrix of shape
        `(n_samples, len(canonical_classes))`, same dtype as
        `probabilities`. Columns for classes absent from
        `model_classes` are filled with zeros.
    """

    n_samples = probabilities.shape[0]
    canonical_arr = np.asarray(canonical_classes)
    # `np.searchsorted` is only correct when `canonical_classes` is sorted
    # ascending. Every current caller passes `np.unique(y_global)` which
    # satisfies this; assert it explicitly so a future caller that supplies
    # a custom (unsorted) class ordering fails loudly rather than silently
    # returning a column-shuffled probability matrix.
    if canonical_arr.size > 1 and not np.all(np.diff(canonical_arr) > 0):
        msg = (
            "align_probs_to_canonical requires `canonical_classes` to be "
            "strictly ascending; got an unsorted ordering."
        )
        raise ValueError(msg)
    probs_canonical = np.zeros(
        (n_samples, len(canonical_arr)), dtype=probabilities.dtype
    )
    # `np.searchsorted` finds each model class's position in the sorted
    # canonical ordering in O((K_model + K_canon) log K_canon), replacing
    # the explicit Python for-loop in the callers.
    model_classes_arr = np.asarray(model_classes)
    target_cols = np.searchsorted(canonical_arr, model_classes_arr)
    # `np.searchsorted` only returns an insertion index; it does NOT verify
    # that each model class is actually present in `canonical_classes`. A
    # model class that falls between two canonical values would silently
    # write probabilities to the wrong canonical column, and a value above
    # all canonical values would yield `index == len(canonical_arr)` and an
    # out-of-bounds assignment. The docstring asserts subset membership, so
    # enforce it loudly here (mirroring the ascending-order check above)
    # rather than corrupting the probability matrix silently.
    if np.any(target_cols >= len(canonical_arr)) or not np.all(canonical_arr[target_cols] == model_classes_arr):
        msg = (
            "align_probs_to_canonical requires every `model_classes` value "
            "to be present in `canonical_classes`; got a model class absent "
            "from the canonical ordering."
        )
        raise ValueError(msg)
    probs_canonical[:, target_cols] = probabilities
    return probs_canonical


def pearson_r_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient between `y_true` and
    `y_pred`, returning NaN when either input has zero variance.

    Pearson's r complements Spearman's rho by measuring *linear* (rather
    than monotonic) agreement on the original scale. For back-transformed
    predictions from a log-link model, a high Spearman but low Pearson
    indicates the model has ranked trials correctly but compressed or
    expanded the magnitudes.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth continuous target of shape (N,).
    y_pred : np.ndarray
        Predicted continuous values of shape (N,).

    Returns
    -------
    float
        Pearson's r on [-1, +1], or NaN if either input is constant.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float('nan')
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except ValueError:
        return float('nan')


def spearman_r_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Spearman rank correlation coefficient between `y_true`
    and `y_pred`, returning NaN when either input has zero variance.

    Spearman's rho measures *monotonic* (rank-based) agreement and is the
    complement to :func:`pearson_r_safe`. On a constant `y_pred` fold,
    `scipy.stats.spearmanr` emits a `ConstantInputWarning` and returns NaN;
    under a strict `filterwarnings=error` run that warning is promoted to an
    error and swallowed by the fold's outer `except`, silently NaN-filling the
    whole actual/null branch. Guarding on the input variance first (mirroring
    :func:`pearson_r_safe`) avoids emitting the warning at all.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth continuous target of shape (N,).
    y_pred : np.ndarray
        Predicted continuous values of shape (N,).

    Returns
    -------
    float
        Spearman's rho on [-1, +1], or NaN if either input is constant.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float('nan')
    try:
        return float(spearmanr(y_true, y_pred)[0])
    except ValueError:
        return float('nan')


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between two 1-D arrays.

    RMSE returns an error magnitude in the native units of the target,
    making it directly interpretable (e.g., "predictions are off by X
    seconds on average"). It penalizes large deviations quadratically and
    is therefore the natural complement to MAE: RMSE grows faster than MAE
    under heavy-tailed residuals, so a large RMSE / MAE ratio signals a few
    outlier folds or trials driving the error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth continuous target.
    y_pred : np.ndarray
        Predicted continuous values.

    Returns
    -------
    float
        RMSE in the units of the target.
    """

    diff = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(diff ** 2)))


def mean_absolute_error_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE) between two 1-D arrays.

    MAE is a robust linear-scale error magnitude: it penalizes each residual
    by its absolute value, so a single outlier trial cannot dominate the
    score the way it would in RMSE. Use MAE as the interpretable baseline
    error ("the model's typical miss on a held-out bout is X units") and
    RMSE as the heavy-tail sensitive diagnostic.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth continuous target.
    y_pred : np.ndarray
        Predicted continuous values.

    Returns
    -------
    float
        MAE in the units of the target.
    """

    return float(np.mean(np.abs(
        np.asarray(y_true, dtype=np.float64).ravel()
        - np.asarray(y_pred, dtype=np.float64).ravel()
    )))


def _render_metric_group(metrics: dict, *, kv_sep: str = '=', join: str = ' ') -> str:
    """
    Renders an ordered ``{metric_name: value}`` mapping into a single string.

    This is the shared low-level formatter behind the standardized modeling
    console output (`format_split_line`, `format_run_summary`). It keeps the
    numeric rendering identical everywhere: four-decimal fixed-point for finite
    values and the literal ``nan`` token for non-finite ones (so an all-NaN
    fold or a degenerate metric is visible rather than crashing the format
    call). The mapping's iteration order is preserved, so callers control the
    left-to-right metric order by the order in which they insert keys.

    Parameters
    ----------
    metrics : dict
        Ordered mapping of metric name -> scalar value (float-coercible).
    kv_sep : str
        Separator between a metric's name and its value (``'='`` for the
        compact per-split line, ``': '`` for the end-of-run summary).
    join : str
        Separator inserted between successive ``name<kv_sep>value`` tokens
        (a single space for per-split, ``' / '`` for the multi-metric summary
        groups).

    Returns
    -------
    str
        The rendered ``name<kv_sep>value`` tokens joined by ``join``.
    """

    tokens = []
    for name, value in metrics.items():
        val = float(value)
        tokens.append(f"{name}{kv_sep}{val:.4f}" if np.isfinite(val)
                      else f"{name}{kv_sep}nan")
    return join.join(tokens)


def format_run_header(*, task: str, engine: str, feature: str,
                      input_files: dict, output_directory: str,
                      split_strategy: str, n_splits: int) -> str:
    """
    Builds the standardized top-of-run header block for a modeling run.

    Emitted once at the start of every univariate feature run and every
    model-selection run so the log opens with an unambiguous statement of WHAT
    is being modelled and WHICH files feed it. Replaces the ad-hoc, per-pipeline
    ``Loading ... from: ...`` / ``Results will be saved to: ...`` lines that
    differ across task types today, giving one scannable block in a fixed order.

    Parameters
    ----------
    task : str
        Task-type label (e.g. ``'ONSET'``, ``'PARAMS'``, ``'MULTINOMIAL'``).
    engine : str
        Fitting engine / framework (e.g. ``'pygam'``, ``'sklearn'``, ``'jax'``).
    feature : str
        The feature (univariate) or target (model selection) being modelled.
    input_files : dict
        Ordered mapping of human-readable label -> file path for every input
        the run consumes (raw input pickle, univariate-results pickle, settings
        file, ...). Rendered one per line in insertion order.
    output_directory : str
        Directory (or file) the run's results are written to.
    split_strategy : str
        Cross-validation split strategy in force (e.g. ``'session'``, ``'mixed'``).
    n_splits : int
        Number of cross-validation splits.

    Returns
    -------
    str
        A multi-line header block bracketed by ``=`` rules.
    """

    rule = '=' * 70
    lines = [rule,
             f"MODELING RUN | task={task} | engine={engine} | feature={feature}",
             f"  split_strategy={split_strategy} | n_splits={n_splits}"]
    for label, path in input_files.items():
        lines.append(f"  {label}: {path}")
    lines.append(f"  output_directory: {output_directory}")
    lines.append(rule)
    return "\n".join(lines)


def format_split_line(split_number: int, n_splits: int,
                      metrics_by_strategy: dict) -> str:
    """
    Builds one standardized per-split console line.

    Every cross-validation split, in every pipeline, prints exactly this format
    so the ACTUAL and NULL (and, where present, the second ``NULL-MF`` model-free
    baseline) metrics for the split sit side by side on one line. Replaces the
    divergent per-task split prints (metric-rich fold lines in some pipelines,
    bare ``> Processing Split N...`` in others).

    Parameters
    ----------
    split_number : int
        1-based split index for display (caller passes ``fold_idx + 1``).
    n_splits : int
        Total number of splits.
    metrics_by_strategy : dict
        Ordered mapping of strategy label (e.g. ``'ACTUAL'``, ``'NULL'``,
        ``'NULL-MF'``) -> an ordered ``{metric_name: value}`` mapping for that
        strategy on this split. Group order and metric order are preserved.

    Returns
    -------
    str
        e.g. ``"Split 03/10 | ACTUAL AUC=0.7200 LL=0.6500 | NULL AUC=0.5000 LL=0.6900"``.
    """

    groups = [f"{label} {_render_metric_group(metrics)}"
              for label, metrics in metrics_by_strategy.items()]
    return f"Split {split_number:02d}/{n_splits:02d} | " + " | ".join(groups)


def format_run_summary(*, label: str, metrics_by_strategy: dict,
                       out_path: str, timestamp=None) -> str:
    """
    Builds the standardized two-line end-of-run summary.

    Printed once when a run finishes: line one states the headline result
    (ACTUAL vs NULL, and NULL-MF where present) and line two states the full
    path the ``.pkl`` was written to, timestamped with ``datetime.now()`` (the
    format the univariate dispatcher already uses for its save line).

    Parameters
    ----------
    label : str
        Run label, including the engine tag, e.g. ``'self.speed [pygam]'`` for a
        univariate feature or ``'multinomial (selected 6 features)'`` for a
        model-selection run.
    metrics_by_strategy : dict
        Ordered mapping of group label (e.g. ``'Mean Actual'``, ``'Mean Null'``,
        ``'Mean Null-MF'``) -> an ordered ``{metric_name: value}`` mapping. A
        group's metrics are joined with ``' / '`` and the groups with ``', '``,
        reproducing ``"Mean Actual AUC: 0.7468, Mean Null AUC: 0.5174"`` for a
        single metric and the ``bal-acc / AUC`` pair for the multinomial task.
    out_path : str
        Full path of the saved results file.
    timestamp : datetime, optional
        Timestamp for line two; defaults to ``datetime.now()``. Passed
        explicitly in tests to make the line deterministic.

    Returns
    -------
    str
        The two summary lines joined by a newline.
    """

    groups = [f"{group} {_render_metric_group(metrics, kv_sep=': ', join=' / ')}"
              for group, metrics in metrics_by_strategy.items()]
    ts = timestamp if timestamp is not None else datetime.now()
    line_one = f"--- Finished {label}. " + ", ".join(groups)
    line_two = f"[{ts}] Success. Results saved to: {out_path}"
    return line_one + "\n" + line_two


def format_selection_step(stage: str, *, feature: str | None = None,
                          metrics: dict | None = None, decision: str | None = None,
                          detail: str | None = None) -> str:
    """
    Builds one standardized forward-feature-selection console line.

    Gives every stage of a ``*_model_selection`` run (screening, the Step-0
    baseline, the auto-anchor, and each forward step) one consistent, pipe-
    delimited shape instead of the divergent ad-hoc prints each function used.
    Only the non-empty fields are rendered, so the same helper serves a bare
    stage banner, a candidate test with a metric, and an accept/reject verdict.

    Parameters
    ----------
    stage : str
        Stage label, e.g. ``'Screen'``, ``'Baseline'``, ``'Anchor'``,
        ``'Step 03'``.
    feature : str, optional
        The feature under consideration; rendered as ``+feature`` when given.
    metrics : dict, optional
        Ordered ``{metric_name: value}`` rendered as ``name=value`` tokens
        (four-decimal, nan-safe -- shared with the per-split formatter).
    decision : str, optional
        The verdict for this line, e.g. ``'ACCEPT'``, ``'REJECT'``, ``'DROP'``,
        ``'FAILED'``.
    detail : str, optional
        Free-form trailing detail (e.g. a bootstrap CI or a drop reason).

    Returns
    -------
    str
        e.g. ``"Step 03 | +self.speed | dcor_xy=0.3100 | ACCEPT | CI [+0.01, +0.05]"``.
    """

    parts = [stage]
    if feature is not None:
        parts.append(f"+{feature}")
    if metrics:
        parts.append(_render_metric_group(metrics))
    if decision:
        parts.append(decision)
    if detail:
        parts.append(detail)
    return " | ".join(parts)


_STRATEGY_SUMMARY_LABELS = {
    'actual': 'Mean Actual',
    'null': 'Mean Null',
    'null_model_free': 'Mean Null-MF',
}


def _safe_fold_mean(values) -> float:
    """
    NaN-safe mean over a per-fold metric list (empty / all-NaN -> ``nan``).

    Guards the end-of-run summary against `RuntimeWarning`-to-error under the
    test suite's ``filterwarnings = error`` policy: a bare ``np.nanmean`` on an
    all-NaN or empty fold list would raise, so the finite-value check is done
    first (the same idiom the runners already use for their own aggregates).

    Parameters
    ----------
    values : array-like
        Per-fold metric values (may contain NaNs from failed folds).

    Returns
    -------
    float
        The nan-ignoring mean, or ``nan`` when nothing finite is present.
    """

    arr = np.asarray(values, dtype=np.float64).ravel()
    return float(np.nanmean(arr)) if arr.size and np.any(np.isfinite(arr)) else float('nan')


def extract_univariate_headline(analysis_type: str, res: dict) -> dict:
    """
    Builds the end-of-run headline ``metrics_by_strategy`` for a univariate run.

    Reads the per-fold metrics a runner already stored in its per-feature result
    dict and reduces them to the mean-over-folds headline for each strategy that
    is present (``actual`` -> "Mean Actual", ``null`` -> "Mean Null",
    ``null_model_free`` -> "Mean Null-MF"). The return value is shaped for
    `format_run_summary` (an ordered ``{group_label: {metric: value}}`` mapping).
    Only the headline metric(s) go into the summary; the fuller per-metric set
    remains on the per-split lines and in the saved pickle.

    Headline metric per task
    ------------------------
    - ``onset`` / ``category`` : AUC (from ``res[strategy]['auc']``).
    - ``params``               : D^2 (from ``res[strategy]['explained_deviance']``).
    - ``multinomial``          : balanced accuracy + macro AUC
                                 (``res[strategy]['folds']['metrics']['score']``
                                 and ``['auc']``).
    - ``continuous``           : ``vm_logscore`` when finite on the ``actual``
                                 strategy (the torus geometry), else ``dcor_xy``
                                 (euclidean), else ``r2_spatial``; read from
                                 ``res[strategy]['folds']['metrics']``.

    Parameters
    ----------
    analysis_type : str
        One of ``'onset'``, ``'category'``, ``'params'``, ``'multinomial'``,
        ``'continuous'``.
    res : dict
        A single feature's result dict, keyed by strategy.

    Returns
    -------
    dict
        Ordered ``{group_label: {metric_name: mean_value}}`` for the strategies
        present, ready to hand to `format_run_summary`.
    """

    out = {}
    if analysis_type in ('onset', 'category'):
        for strategy in ('actual', 'null'):
            if strategy in res:
                out[_STRATEGY_SUMMARY_LABELS[strategy]] = {
                    'AUC': _safe_fold_mean(res[strategy]['auc'])
                }
    elif analysis_type == 'params':
        for strategy in ('actual', 'null'):
            if strategy in res:
                out[_STRATEGY_SUMMARY_LABELS[strategy]] = {
                    'D^2': _safe_fold_mean(res[strategy]['explained_deviance'])
                }
    elif analysis_type == 'multinomial':
        for strategy in ('actual', 'null', 'null_model_free'):
            if strategy in res:
                metrics = res[strategy]['folds']['metrics']
                out[_STRATEGY_SUMMARY_LABELS[strategy]] = {
                    'bal-acc': _safe_fold_mean(metrics['score']),
                    'AUC': _safe_fold_mean(metrics['auc']),
                }
    elif analysis_type == 'continuous' and 'actual' in res and 'folds' in res['actual'] and 'metrics' in res['actual']['folds']:
        # Geometry selects the headline: `vm_logscore` is finite only on the
        # torus (NaN on euclidean), `dcor_xy` only on euclidean (NaN on the
        # torus), else `r2_spatial`. Guard the unconditional `actual` read so a
        # malformed / minimal result dict (e.g. a mocked dispatcher test) yields
        # an empty summary rather than raising.
        actual_metrics = res['actual']['folds']['metrics']

        def _finite(metric_key):
            """
            Reports whether the actual-strategy fold metrics hold at
            least one finite value for `metric_key`.

            Parameters
            ----------
            metric_key : str
                Key into `res['actual']['folds']['metrics']`.

            Returns
            -------
            bool
                True when the key is present and its per-fold values
                contain at least one finite number; False otherwise
                (absent key, empty array or all non-finite).
            """
            if metric_key not in actual_metrics:
                return False
            vals = np.asarray(actual_metrics[metric_key], dtype=np.float64)
            return bool(vals.size and np.any(np.isfinite(vals)))

        if _finite('vm_logscore'):
            key = 'vm_logscore'
        elif _finite('dcor_xy'):
            key = 'dcor_xy'
        else:
            key = 'r2_spatial'
        for strategy in ('actual', 'null', 'null_model_free'):
            if strategy in res:
                out[_STRATEGY_SUMMARY_LABELS[strategy]] = {
                    key: _safe_fold_mean(res[strategy]['folds']['metrics'][key])
                }
    return out


def run_predictor_audits(processed_beh_dict: dict,
                         usv_data_dict: dict,
                         mouse_names_dict: dict,
                         camera_fps_dict: dict,
                         target_idx: int,
                         predictor_idx: int,
                         history_frames: int,
                         event_keys: list,
                         settings: dict,
                         save_dir: str,
                         pickle_basename: str,
                         precomputed_event_times: dict | None = None,
                         input_metadata: dict | None = None,
                         balance_event_keys: bool = False,
                         onset_event_key: str = 'positive_events',
                         precomputed_onset_times: dict | None = None) -> None:
    """
    Runs the collinearity and timescale audits as a single non-fatal
    diagnostic step at modeling-input-pickle creation time.

    This wrapper is the only entry point each `extract_and_save_*`
    pipeline needs to call. It:

    1. Reads the `diagnostics` block from `settings`. When neither audit
       is enabled the wrapper returns immediately without touching disk.
    2. Pools per-session event time arrays from the supplied USV data
       dict by walking the per-pipeline `event_keys` (e.g.
       `['positive_events']` for the Onset pipeline,
       `['target_events', 'other_events']` for the binomial Category
       pipeline). The union of all listed keys is used both as the row
       basis of the collinearity audit's per-event summary matrix and as
       the binary event indicator `Y(t)` of the timescale audit's
       predictive-ρ profile.
    3. Computes the sex-specific IBI thresholds via
       `_calculate_ibi_threshold` so the timescale audit can report them
       alongside the configured `filter_history` for the headline
       recommendation line.
    4. Calls each audit inside its own `try/except` — any failure is
       logged with a warning but does not abort the calling pipeline.
       The audits are diagnostic only; downstream modeling does not
       depend on their artifacts existing.

    The two artifacts are written next to the modeling input pickle
    using the basename suffixes `_collinearity.pkl` and `_timescales.pkl`
    so they are trivially co-locatable for downstream plotting.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping `session_id -> polars.DataFrame` of z-scored, harmonized
        per-session feature traces. Must already have been through
        `harmonize_session_columns` (or equivalent) and
        `zscore_features_across_sessions`.
    usv_data_dict : dict
        Nested USV data dictionary
        (`session_id -> mouse_name -> {event_key: ndarray, ...}`)
        produced by the loaders in `load_input_files.py`.
    mouse_names_dict : dict
        Mapping `session_id -> list[mouse_name]`.
    camera_fps_dict : dict
        Mapping `session_id -> camera_sampling_rate_hz`.
    target_idx, predictor_idx : int
        Mouse slot indices used to translate per-mouse columns into
        generic `self.*` / `other.*` keys.
    history_frames : int
        Pre-event window length in frames (used by the collinearity
        audit's per-event summary matrix).
    event_keys : list of str
        Per-pipeline list of keys under each `usv_data_dict[session][mouse]`
        entry whose stored arrays should be unioned to form the audit's
        event time set. Pass every event class the pipeline will actually
        train on (e.g. `['positive_events']` for Onset,
        `['target_events', 'other_events']` for binomial Category).
    settings : dict
        The modeling settings dictionary. Reads
        `settings['diagnostics']` (toggle flags + `timescale_max_lag_seconds`
        / `timescale_n_shuffles`), `settings['model_params']['filter_history']`
        (recommendation line), `settings['model_params']['mixture_model_component_index']`
        and `settings['model_params']['mixture_model_z_score']` (IBI thresholds), and
        `settings['mixture_model_params']` (per-sex mixture-model components).
    save_dir : str
        Output directory for the audit artifacts (typically the modeling
        save directory).
    pickle_basename : str
        Filename of the paired modeling input pickle. Used to derive the
        artifact basenames and as a provenance string written into each
        artifact.
    precomputed_event_times : dict, optional
        Pre-built `session_id -> np.ndarray` mapping of pooled event onset
        times (seconds). When supplied, the wrapper skips the per-mouse
        `event_keys` extraction step. Use this when the calling pipeline
        stores its events in a non-standard shape (e.g. the multinomial
        pipeline's `events_by_category` dict).
    input_metadata : dict, optional
        The fully built `_input_metadata` block from the calling
        pipeline. Forwarded verbatim to both audit functions, which
        embed it inside their on-disk payloads under the reserved key
        `_input_metadata`. This makes each audit artifact independently
        provenance-complete (no need to consult the paired modeling
        input pickle to learn which cohort / settings produced the
        diagnostic).
    balance_event_keys : bool, default False
        When True (and `event_keys` resolves to two-or-more non-empty
        per-mouse arrays for a session), the wrapper subsamples each
        non-smallest array down to the smallest's size *per session*,
        without replacement, before concatenating. The resulting
        `event_times_per_session` then has equal counts across the
        listed event classes — matching the balanced design that
        binary-classification pipelines (e.g. the bout-onset GAM)
        actually train on. Subsampling is deterministic via a NumPy
        `default_rng` seeded with `settings['model_validation']['random_seed']`
        (or `0` when that field is None). When False (default), the
        wrapper concatenates every per-key array verbatim, which is
        appropriate for pipelines whose event classes have
        independent semantics (e.g. the binomial Category pipeline's
        `target_events` / `other_events`, which describe two
        different mice rather than two outcome classes for the same
        decision). The timescale audit's binary `Y` trace and the
        empirical IBI block are unaffected — they read
        `onset_event_key` / `start` / `stop` directly, not the
        balanced pool.
    onset_event_key : str, default 'positive_events'
        Per-mouse key under `usv_data_dict[sess][target_name]` whose
        stored array is the event time set used to build the
        timescale audit's binary `Y(t)` impulse trace. The kwarg
        name retains the historical "bout-onset" wording from when
        the audit was first written for the bout-onsets pipeline;
        in practice it is the **generalized model-event impulse
        source**, and the granularity should match what the pipeline
        actually predicts. Pipelines whose model fits one prediction
        per bout pass a bout-start key (e.g. the vocal-onsets
        pipeline's default `'positive_events'`, or the bout-
        parameters pipeline's `'bout_onsets'`); pipelines whose
        model fits one prediction per individual USV pass a
        per-USV-start key (e.g. the binomial Category pipeline's
        `'target_events'`). Without this override `Y(t)` ends up
        empty, every cross-correlation row becomes NaN, and no XC
        marker can be located. Ignored when
        `precomputed_onset_times` is supplied.
    precomputed_onset_times : dict, optional
        Pre-built `session_id -> np.ndarray` mapping of model-event
        impulse times in seconds, used as the timescale audit's
        `Y(t)` source. When supplied the wrapper bypasses the
        `onset_event_key` string-key lookup entirely. Use this
        when the calling pipeline stores its events in a non-flat
        shape (e.g. the multinomial pipeline's `events_by_category`
        nested dict, or the continuous manifold pipeline's
        `continuous_targets` packets). Like
        `onset_event_key`, the kwarg name retains the
        historical "bout-onset" wording but the actual semantic is
        **per-prediction event times** — for the multinomial /
        manifold pipelines those are per-USV starts (one impulse
        per USV across all kept categories), not bout starts.
        Typically the same dict already passed as
        `precomputed_event_times` is appropriate here too, since
        both ultimately describe the per-session events the model
        is trained on.

    Returns
    -------
    None
        The audit artifacts are written to `save_dir` as a side effect;
        nothing is returned (also when both audits are disabled).
    """

    diagnostics_cfg = settings['diagnostics']
    do_collinearity = diagnostics_cfg['collinearity_audit']
    do_timescale = diagnostics_cfg['timescale_audit']

    if not (do_collinearity or do_timescale):
        return

    # Pool per-session event time arrays across the requested event keys.
    # The audit treats every listed key as a "model row" so the summary
    # matrix and the binary event indicator span every epoch the model
    # will ever see. Callers with non-standard event storage can pass
    # `precomputed_event_times` to bypass this loop. When
    # `balance_event_keys` is True, each non-smallest per-key array is
    # subsampled down to the smallest's size *per session* before
    # concatenation, so the audit sees the balanced design the
    # downstream classifier actually trains on (avoids majority-class
    # domination of the audit's correlations / VIFs / cond(X̃)).
    if precomputed_event_times is not None:
        event_times_per_session = precomputed_event_times
    else:
        # Seeded RNG for reproducible balanced subsampling. Created
        # lazily inside the loop only when actually needed; the seed
        # comes from the same `random_seed` field every other
        # stochastic step in the modeling pipeline reads, with `0` as
        # the fallback (matching the timescale audit's convention
        # below).
        balance_rng = None
        event_times_per_session = {}
        for sess_id, track_names in mouse_names_dict.items():
            if sess_id not in usv_data_dict:
                continue
            target_name = track_names[target_idx]
            if target_name not in usv_data_dict[sess_id]:
                continue
            per_mouse = usv_data_dict[sess_id][target_name]

            per_key_arrays = []
            for key in event_keys:
                arr = per_mouse[key] if key in per_mouse else None
                if arr is None:
                    continue
                arr = np.asarray(arr).ravel()
                if arr.size > 0:
                    per_key_arrays.append(arr)
            if not per_key_arrays:
                continue

            if balance_event_keys and len(per_key_arrays) >= 2:
                if balance_rng is None:
                    seed_raw = settings['model_validation']['random_seed']
                    balance_rng = np.random.default_rng(
                        int(seed_raw) if seed_raw is not None else 0
                    )
                min_n = min(a.size for a in per_key_arrays)
                balanced = []
                for a in per_key_arrays:
                    if a.size > min_n:
                        pick = balance_rng.choice(a.size, size=min_n,
                                                  replace=False)
                        balanced.append(a[pick])
                    else:
                        balanced.append(a)
                per_key_arrays = balanced

            event_times_per_session[sess_id] = np.sort(
                np.concatenate(per_key_arrays)
            )

    # Per-USV start/stop arrays for the timescale audit's empirical
    # IBI-percentile reporting. Read directly from the loader's
    # `usv_data_dict[sess][target]['start' | 'stop']` arrays. The audit
    # computes inter-USV gaps (`gap_i = start[i+1] − stop[i]`) for the
    # IBI percentile block; the mixture-model-derived `ibi_threshold` is on the
    # same gaps so the percentiles are directly comparable to the
    # threshold.
    event_intervals_per_session = {}
    for sess_id, track_names in mouse_names_dict.items():
        if sess_id not in usv_data_dict:
            continue
        target_name = track_names[target_idx]
        if target_name not in usv_data_dict[sess_id]:
            continue
        per_mouse = usv_data_dict[sess_id][target_name]
        if 'start' not in per_mouse or 'stop' not in per_mouse:
            continue
        starts = np.asarray(per_mouse['start'])
        stops = np.asarray(per_mouse['stop'])
        if starts.size == 0 or stops.size == 0:
            continue
        event_intervals_per_session[sess_id] = (starts, stops)

    base_no_ext = Path(pickle_basename).stem
    save_dir_p = Path(save_dir)
    coll_path = str(save_dir_p / f"{base_no_ext}_collinearity.pkl")
    ts_path = str(save_dir_p / f"{base_no_ext}_timescales.pkl")

    if do_collinearity:
        try:
            audit_predictor_collinearity(
                processed_beh_dict=processed_beh_dict,
                event_times_per_session=event_times_per_session,
                mouse_names_dict=mouse_names_dict,
                target_idx=target_idx,
                predictor_idx=predictor_idx,
                history_frames=history_frames,
                camera_fps_dict=camera_fps_dict,
                save_path=coll_path,
                source_pickle=pickle_basename,
                concern_thresh=diagnostics_cfg['collinearity_concern_threshold'],
                exclude_thresh=diagnostics_cfg['collinearity_exclude_threshold'],
                input_metadata=input_metadata,
            )
        except Exception as exc:
            print(f"[audit] collinearity audit failed (non-fatal): {exc}")

    if do_timescale:
        try:
            # Sex-specific IBI thresholds — same recipe used by the loaders
            # so the audit's headline matches what gates bout detection.
            mixture_model_idx = settings['model_params']['mixture_model_component_index']
            mixture_model_z = settings['model_params']['mixture_model_z_score']
            mixture_model_params = settings['mixture_model_params']
            ibi_thresholds = {}
            for sex in ('male', 'female'):
                params = mixture_model_params[sex]
                # Lower-bound the index too: a negative `mixture_model_idx` would pass a
                # bare `< n` guard and then index the component arrays from the end,
                # silently selecting the wrong mixture-model component. Require a
                # non-negative in-range index so any out-of-range value (negative or
                # too large) falls to the NaN branch instead.
                if 0 <= mixture_model_idx < mixture_component_count(params):
                    ibi_thresholds[sex] = mixture_ibi_threshold(params, mixture_model_idx, mixture_model_z)
                else:
                    ibi_thresholds[sex] = float('nan')

            max_lag = float(diagnostics_cfg['timescale_max_lag_seconds'])
            n_shuffles = int(diagnostics_cfg['timescale_n_shuffles'])
            shuffle_range_raw = diagnostics_cfg['timescale_shuffle_range']
            if not (isinstance(shuffle_range_raw, (list, tuple)) and len(shuffle_range_raw) == 2):
                msg = (
                    f"`diagnostics.timescale_shuffle_range` must be a 2-element list of "
                    f"positive numbers (got {shuffle_range_raw!r})."
                )
                raise ValueError(msg)
            shuffle_range_seconds = (float(shuffle_range_raw[0]), float(shuffle_range_raw[1]))
            if not (0 < shuffle_range_seconds[0] < shuffle_range_seconds[1]):
                msg = (
                    f"`diagnostics.timescale_shuffle_range` must satisfy "
                    f"0 < min < max (got {shuffle_range_seconds!r})."
                )
                raise ValueError(msg)
            # Lag floor for the cross-correlation right-side significance
            # marker. Crossings at lags below this floor are ignored when
            # locating the outer-run marker; this prevents the very-near-zero
            # null dips (e.g. `other.speed`) from anchoring the marker too
            # close to lag 0.
            signal_floor_seconds = float(diagnostics_cfg['timescale_signal_floor_seconds'])
            if signal_floor_seconds < 0:
                msg = (
                    f"`diagnostics.timescale_signal_floor_seconds` must be "
                    f"non-negative (got {signal_floor_seconds!r})."
                )
                raise ValueError(msg)
            # Minimum sustained-run length for the cross-correlation
            # right-side significance marker. The marker is placed at
            # the end of the *longest* sign-consistent run on the
            # positive lag axis whose length is at least this many
            # seconds. Set above the typical noise-fragment scale
            # (~0.1 s for a few-bin run) so that scattered short
            # excursions (e.g. `other.usv_rate` with ~28-bin
            # fragments) are not promoted to markers, while
            # preserving real sustained runs (≥ 213 bins / 1.4 s on
            # `self.back_yaw`, 636 bins / 4.2 s on `self.ego_yaw`).
            signal_min_run_seconds = float(diagnostics_cfg['timescale_signal_min_run_seconds'])
            if signal_min_run_seconds <= 0:
                msg = (
                    f"`diagnostics.timescale_signal_min_run_seconds` must be "
                    f"strictly positive (got {signal_min_run_seconds!r})."
                )
                raise ValueError(msg)
            random_seed = settings['model_validation']['random_seed'] if settings['model_validation']['random_seed'] is not None else 0

            # Bout-onset times for the timescale audit's binary `Y`
            # trace. Built here (inside the timescale audit's
            # try/except) rather than at the top of the wrapper so a
            # mis-specified `onset_event_key` raises into the
            # non-fatal warning path instead of crashing the calling
            # pipeline. The vocal_onsets pipeline in `bout` mode
            # writes onsets to `usv_data_dict[sess][target]
            # ['positive_events']`; the bout-parameters pipeline
            # writes them to `[...]['bout_onsets']`; the binomial
            # Category pipeline writes them to
            # `[...]['target_events']`. The per-pipeline key is
            # parameterised via `onset_event_key`. Pipelines
            # whose events are stored in a non-flat shape (e.g. the
            # multinomial `events_by_category` nested dict, or the
            # continuous manifold's per-session `continuous_targets`
            # packets) bypass the string-key lookup by passing
            # `precomputed_onset_times` directly. Either way,
            # the audit's `Y` is exactly the set of bout onsets the
            # model is trained to predict — no duplicate
            # bout-detection logic in the audit.
            if precomputed_onset_times is not None:
                onset_times_per_session = {
                    sess_id: np.sort(np.asarray(arr).ravel())
                    for sess_id, arr in precomputed_onset_times.items()
                    if np.asarray(arr).size > 0
                }
            else:
                onset_times_per_session = {}
                n_sessions_with_target = 0
                for sess_id, track_names in mouse_names_dict.items():
                    if sess_id not in usv_data_dict:
                        continue
                    target_name = track_names[target_idx]
                    if target_name not in usv_data_dict[sess_id]:
                        continue
                    n_sessions_with_target += 1
                    per_mouse = usv_data_dict[sess_id][target_name]
                    if onset_event_key not in per_mouse:
                        continue
                    bout_starts = np.asarray(per_mouse[onset_event_key])
                    if bout_starts.size == 0:
                        continue
                    onset_times_per_session[sess_id] = np.sort(bout_starts.ravel())

                # Universal-absence guard. If every session that had a
                # target-mouse entry was silently skipped because none
                # carried `onset_event_key`, the key is wrong (typo,
                # or the calling pipeline stores its onsets under a
                # different name). A degenerate audit pickle with
                # `n_bouts=0` and all-NaN ρ rows would otherwise be
                # written silently — fail loudly so the caller fixes the
                # key rather than misreading the empty result. The
                # outer `try/except` converts this into the documented
                # `[audit] timescale audit failed (non-fatal): ...`
                # warning instead of crashing the pipeline. Sessions
                # that legitimately have *some* onsets but not all
                # continue to skip silently as before.
                if n_sessions_with_target > 0 and not onset_times_per_session:
                    msg = (
                        f"`onset_event_key={onset_event_key!r}` was not "
                        f"found in any of the {n_sessions_with_target} session(s) "
                        f"that carried a target-mouse entry in `usv_data_dict`. "
                        f"Either the key is a typo, or the calling pipeline "
                        f"stores its onsets under a different key — pass "
                        f"`onset_event_key=...` or "
                        f"`precomputed_onset_times=...` explicitly. The "
                        f"audit refuses to write a pickle whose `Y(t)` is empty, "
                        f"since every cross-correlation row would collapse to "
                        f"NaN."
                    )
                    raise ValueError(msg)

            audit_predictor_timescales(
                processed_beh_dict=processed_beh_dict,
                mouse_names_dict=mouse_names_dict,
                target_idx=target_idx,
                predictor_idx=predictor_idx,
                configured_filter_history=float(settings['model_params']['filter_history']),
                camera_fps_dict=camera_fps_dict,
                max_lag_seconds=max_lag,
                n_shuffles=n_shuffles,
                ibi_thresholds=ibi_thresholds,
                save_path=ts_path,
                source_pickle=pickle_basename,
                random_seed=int(random_seed),
                input_metadata=input_metadata,
                shuffle_range_seconds=shuffle_range_seconds,
                event_intervals_per_session=event_intervals_per_session,
                onset_times_per_session=onset_times_per_session,
                signal_floor_seconds=signal_floor_seconds,
                signal_min_run_seconds=signal_min_run_seconds,
            )
        except Exception as exc:
            print(f"[audit] timescale audit failed (non-fatal): {exc}")
