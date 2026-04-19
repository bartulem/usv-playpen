"""
@author: bartulem
Shared data-preparation helpers for the USV modeling pipelines.

This module consolidates the recurring prologue of every `extract_and_save_*`
method across the five pipeline classes (`VocalOnsetModelingPipeline`,
`BoutParameterPipeline`, `VocalCategoryModelingPipeline`,
`MultinomialModelingPipeline`, `ContinuousModelingPipeline`). The helpers are
module-level free functions — they take the settings dictionary and the raw
per-session inputs explicitly, so they can be invoked from any pipeline without
class coupling.

Each helper targets one distinct stage of the shared prologue:

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
G. `collect_predictor_suffixes`   — compute the union of non-numeric column
                                     suffixes across per-session DataFrames.
H. `zero_fill_missing_feature_columns` — fill in missing columns across
                                     sessions with zeros so every session has
                                     the same feature set, respecting vocal
                                     exclusion rules for target/partner mice.
I. `harmonize_session_columns`    — dyad-rename + project-wide existence-map
                                     USV gating + zero-fill, used by the
                                     Category/Multinomial/Continuous pipelines
                                     (returns the harmonized dict and its
                                     unified suffix list).
J. `zscore_features_across_sessions`   — thin wrapper around
                                     `zscore_different_sessions_together`
                                     for API symmetry with the other helpers.

Block B (`load_behavioral_feature_data`) is intentionally NOT wrapped here —
callers should import it directly from `load_input_files`.
"""

import os
import numpy as np
import polars as pls

from .load_input_files import load_behavioral_feature_data
from .modeling_cross_session_normalization import zscore_different_sessions_together
from ..os_utils import configure_path


def prepare_modeling_sessions(modeling_settings: dict) -> list:
    """
    Seeds NumPy's global RNG (if requested) and loads the session-path list.

    Reads the `model_params.random_seed` and `io.session_list_file` entries
    from the supplied settings dictionary, applies the seed (or resets it to
    a non-deterministic state when the seed is `None`), opens the session-list
    text file, and returns a list of configured absolute paths. Blank lines
    in the session-list file are skipped.

    Parameters
    ----------
    modeling_settings : dict
        The modeling settings dictionary. Must contain
        `modeling_settings['model_params']['random_seed']` and
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

    if modeling_settings['model_params']['random_seed'] is not None:
        np.random.seed(modeling_settings['model_params']['random_seed'])
        print(f"Random seed set to: {modeling_settings['model_params']['random_seed']}")
    else:
        np.random.seed(None)
        print("Random seed not set.")

    txt_modeling_sessions = []
    sessions_file = modeling_settings['io']['session_list_file']
    try:
        with open(configure_path(sessions_file)) as f:
            for line in f:
                line = line.strip()
                if line:
                    txt_modeling_sessions.append(configure_path(line))
        if not txt_modeling_sessions:
            raise ValueError("No valid session paths found.")
        print(f"Loaded {len(txt_modeling_sessions)} session paths.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Sessions list file not found: {sessions_file}")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading session paths: {e}")

    return txt_modeling_sessions


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
        predictor-mouse convention:
          * `predictor_idx == 0`: drop if `feat_parts[0] == 'allo_yaw'` or
                                  `feat_parts[1] == 'TTI'`.
          * `predictor_idx != 0`: drop if `feat_parts[1] == 'allo_yaw'` or
                                  `feat_parts[0] == 'TTI'`.
        When `dyadic_pose_symmetric` is True, both halves are retained.

    - `dyadic_engagement`:
        Two-mouse interaction features of the form
        `{tracker_id}-{tracked_id}.{feature}` (e.g. Social Engagement Index
        suffixes such as `orofacial-sei`). Because the first mouse in the
        dyad prefix is the observer (actor), this bucket keeps only the
        orientation where the observer is the *predictor* mouse — i.e.
        `{predictor_name}-{target_name}.{feature}`. The reverse
        `{target_name}-{predictor_name}.{feature}` orientation is dropped.
        This preserves the "partner's engagement toward subject" reading,
        which is consistent with the partner-only vocal predictor scheme,
        and avoids the column-name collision that would otherwise occur
        when the downstream dyad-rename step strips the `{m1-m2}.` prefix.

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
                        if feat_parts[0] == 'allo_yaw' or feat_parts[1] == 'TTI':
                            continue
                    else:
                        if feat_parts[1] == 'allo_yaw' or feat_parts[0] == 'TTI':
                            continue
            columns_to_keep.append(feature)
            _maybe_add_derivatives(feature, base_feature)

    engagement_dyad_prefix = f"{predictor_name}-{target_name}"
    for base_feature in dyadic_engagement:
        matching = [c for c in session_df_columns if c.split('.')[-1] == base_feature]
        for feature in matching:
            dyad_prefix = feature.split('.')[0]
            if dyad_prefix != engagement_dyad_prefix:
                continue
            columns_to_keep.append(feature)
            _maybe_add_derivatives(feature, base_feature)

    return sorted(list(set(columns_to_keep)))


def build_vocal_signal_columns(usv_data_dict: dict,
                               session_id: str,
                               target_name: str,
                               predictor_name: str,
                               voc_settings: dict,
                               usv_self_exclude: tuple = ('usv_rate', 'usv_event')) -> tuple:
    """
    Constructs the list of per-mouse vocal signal columns for one session.

    Reads the continuous vocal signals previously attached to the USV data
    dictionary (by `find_bout_epochs`, `find_usv_categories`, or
    `find_variable_length_bouts`) and turns them into polars Series ready to
    be `.with_columns`-attached to a behavioral DataFrame.

    Behavior
    --------
    - When `voc_settings['usv_predictor_type']` is falsy, returns empty lists.
    - When `voc_settings['usv_predictor_partner_only']` is True, only the
      predictor (partner) mouse's signals are emitted.
    - For the target (self) mouse, any signal key in `usv_self_exclude`
      (default: `('usv_rate', 'usv_event')`) is skipped, because including the
      subject's own aggregate rate/event trace would induce trivial
      autocorrelation and drown out behavioral predictors. Per-category
      signals (`usv_cat_*`) are always allowed because they carry syntax
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
        Must contain the keys `usv_predictor_type` (str or None) and
        `usv_predictor_partner_only` (bool).
    usv_self_exclude : tuple of str, optional
        Signal keys that must NEVER be emitted for the target mouse.
        Default: `('usv_rate', 'usv_event')`.

    Returns
    -------
    tuple of (list of pls.Series, list of str)
        `(new_voc_cols, new_voc_col_names)`. The Series list is suitable for
        `df.with_columns(new_voc_cols)`; the name list mirrors it so callers
        can union-extend `columns_to_keep`.
    """

    new_voc_cols = []
    new_voc_col_names = []

    voc_out_type = voc_settings['usv_predictor_type']
    partner_only = voc_settings['usv_predictor_partner_only']

    if not voc_out_type:
        return new_voc_cols, new_voc_col_names

    mice_to_process = [predictor_name] if partner_only else [target_name, predictor_name]

    for m_name in mice_to_process:
        is_target = (m_name == target_name)

        if m_name not in usv_data_dict[session_id]:
            continue

        vocal_signals = usv_data_dict[session_id][m_name]['continuous_vocal_signals']

        for sig_key, sig_arr in vocal_signals.items():
            if is_target and sig_key in usv_self_exclude:
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

        events = usv_data_dict[session_id][target_name].get(event_key, [])
        if len(events) == 0:
            print(f"Skipping {warn_label} {session_id}: 0 valid events for {target_name} ({event_key}).")
            sessions_to_remove.append(session_id)

    return sessions_to_remove


def collect_predictor_suffixes(processed_beh_dict: dict) -> list:
    """
    Collects the union of non-numeric column suffixes across sessions.

    The suffix is the segment after the last `.` in a column name. Purely
    numeric suffixes are skipped (they correspond to vocal-category integer
    labels attached elsewhere and are not generic predictor suffixes).

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping from `session_id` to a polars DataFrame of kept features.

    Returns
    -------
    list of str
        Sorted list of unique feature suffixes.
    """

    suffixes = set()
    for sess_df in processed_beh_dict.values():
        for col in sess_df.columns:
            suffix = col.split('.')[-1]
            if suffix.isdigit():
                continue
            suffixes.add(suffix)
    return sorted(list(suffixes))


def zero_fill_missing_feature_columns(processed_beh_dict: dict,
                                      mouse_names_dict: dict,
                                      target_idx: int,
                                      predictor_idx: int,
                                      suffixes: list,
                                      voc_settings: dict,
                                      session_list_file: str = None,
                                      skip_dyadic_suffixes: bool = True,
                                      usv_self_exclude: tuple = ('usv_rate', 'usv_event')) -> dict:
    """
    Standardizes the column set across sessions by filling missing columns with zeros.

    For every `(mouse, suffix)` combination that is expected based on the
    supplied `suffixes` list but not already present in a session DataFrame,
    this function appends a zero-filled float32 column. The result is that
    every session DataFrame has the same egocentric column set after the
    call.

    Vocal exclusion rules
    ---------------------
    When the suffix starts with `usv_` (vocal signal), additional rules apply:

    - For the *target* mouse, never add `usv_*` zero columns when
      `voc_settings['usv_predictor_partner_only']` is True OR when the
      suffix is in `usv_self_exclude` (default: `('usv_rate', 'usv_event')`).
      These are the same self-autocorrelation guards enforced by
      `build_vocal_signal_columns`.

    - For the *partner* mouse, when a `session_list_file` is supplied and its
      basename contains the token `'mute'`, never add `usv_*` zero columns —
      the partner is muted in that experimental condition, so zero-filling
      would falsely suggest vocal activity channels existed.

    Dyadic suffixes
    ---------------
    When `skip_dyadic_suffixes` is True (default), suffixes containing a `-`
    are skipped entirely: dyadic columns carry a `{male}-{female}` prefix,
    not a single-mouse prefix, so the `{mouse}.{dyadic-suffix}` construction
    would produce meaningless column names. Set this flag to False only to
    preserve legacy behavior of callers that used to do this unconditionally.

    The function mutates `processed_beh_dict` in place and also returns it.

    Parameters
    ----------
    processed_beh_dict : dict
        Mapping from `session_id` to a polars DataFrame. Modified in place.
    mouse_names_dict : dict
        Mapping from `session_id` to an ordered list of mouse track names.
    target_idx : int
        The target mouse slot.
    predictor_idx : int
        The predictor mouse slot.
    suffixes : list of str
        The union of predictor suffixes (typically from
        `collect_predictor_suffixes`).
    voc_settings : dict
        Must contain `usv_predictor_partner_only` (bool).
    session_list_file : str, optional
        The path (or basename) of the session-list file; consulted only to
        decide the partner-mute rule. Ignored when None.
    skip_dyadic_suffixes : bool, optional
        If True (default), suffixes containing `-` are skipped.
    usv_self_exclude : tuple of str, optional
        Signal keys that must NEVER receive a target-mouse zero column.
        Default: `('usv_rate', 'usv_event')`.

    Returns
    -------
    dict
        The same `processed_beh_dict` (returned for call-chaining convenience).
    """

    partner_only = voc_settings['usv_predictor_partner_only']
    mute_partner = (session_list_file is not None
                    and 'mute' in os.path.basename(session_list_file))

    for sess_id in processed_beh_dict:
        df = processed_beh_dict[sess_id]
        existing_cols = set(df.columns)
        new_zeros = []

        t_name = mouse_names_dict[sess_id][target_idx]
        p_name = mouse_names_dict[sess_id][predictor_idx]

        for pred_suffix in suffixes:
            if skip_dyadic_suffixes and '-' in pred_suffix:
                continue

            for m_name in (t_name, p_name):
                expected_col = f"{m_name}.{pred_suffix}"
                if expected_col in existing_cols:
                    continue

                is_vocal = 'usv_' in pred_suffix
                if is_vocal:
                    if m_name == t_name:
                        if partner_only or pred_suffix in usv_self_exclude:
                            continue
                    else:
                        if mute_partner:
                            continue

                new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))

        if new_zeros:
            processed_beh_dict[sess_id] = df.with_columns(new_zeros)

    return processed_beh_dict


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

    for sess_id, df in processed_beh_dict.items():
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
                        if 'usv' in suffix:
                            if generic_key in generic_existence_map:
                                new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))
                        else:
                            new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))

        if new_zeros:
            processed_beh_dict[sess_id] = df.with_columns(new_zeros)

    return processed_beh_dict, sorted(list(final_suffixes))


def zscore_features_across_sessions(processed_beh_dict: dict,
                                    suffixes: list,
                                    feature_bounds: dict) -> dict:
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
        `collect_predictor_suffixes`).
    feature_bounds : dict
        Optional per-feature clipping bounds (may be empty).

    Returns
    -------
    dict
        The normalized mapping (per `zscore_different_sessions_together`).
    """

    return zscore_different_sessions_together(
        data_dict=processed_beh_dict,
        feature_lst=suffixes,
        feature_bounds=feature_bounds
    )
