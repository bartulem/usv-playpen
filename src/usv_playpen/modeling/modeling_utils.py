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
K. `pool_session_arrays`          — concatenate two-class per-session arrays
                                     (positive/negative) across a list of
                                     sessions, parameterized by the pair of
                                     dict keys (used by Onset and Category).
L. `balance_two_class_arrays`     — down-sample the majority class of a
                                     two-class dataset to match the minority
                                     class size (used by Onset and Category).
M. `unroll_history_matrix`        — reshape a `(n_samples, n_frames)` feature-
                                     history matrix into the two-column
                                     `(n_samples * n_frames, 2)` layout
                                     consumed by the pygam tensor-product
                                     spline fits (used by Onset, Bout, and
                                     Category).
N. `concat_two_class_with_labels` — vertically stack positive/negative feature
                                     arrays and emit the matching label vector
                                     (1.0 for positives, 0.0 for negatives).
                                     Shared by the Onset and Category
                                     splitters.
O. `shuffle_train_test_arrays`    — apply independent NumPy permutations to
                                     the train and test blocks at the final
                                     yield step of a split generator. Shared
                                     by the Onset and Category splitters.
P. `bounded_test_proportion`      — clamp `test_proportion` up to the minimum
                                     fraction required to keep
                                     `min_test_sessions` sessions in the test
                                     fold, used when the session count is
                                     small. Used by the Onset splitter.

Block B (`load_behavioral_feature_data`) is intentionally NOT wrapped here —
callers should import it directly from `load_input_files`.
"""

from pathlib import Path
import numpy as np
import polars as pls
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from .load_input_files import load_behavioral_feature_data, _calculate_ibi_threshold
from .modeling_collinearity_audit import (
    audit_predictor_collinearity,
    audit_predictor_timescales,
)
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
                    and 'mute' in Path(session_list_file).name)

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
                        # Use the same `'usv_'` substring test as
                        # `zero_fill_missing_feature_columns` so the two
                        # helpers gate vocal suffixes identically (no risk
                        # of a non-vocal suffix that happens to contain the
                        # bare `'usv'` substring being silently gated).
                        if 'usv_' in suffix:
                            if generic_key in generic_existence_map:
                                new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))
                        else:
                            new_zeros.append(pls.Series(expected_col, np.zeros(df.height, dtype=np.float32)))

        if new_zeros:
            processed_beh_dict[sess_id] = df.with_columns(new_zeros)

    return processed_beh_dict, sorted(list(final_suffixes))


def zscore_features_across_sessions(processed_beh_dict: dict,
                                    suffixes: list,
                                    feature_bounds: dict,
                                    abs_features: list | None = None) -> dict:
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
    abs_features : list of str, optional
        Feature names whose values should be folded to their absolute
        magnitude prior to z-scoring (forwarded to
        `zscore_different_sessions_together`).

    Returns
    -------
    dict
        The normalized mapping (per `zscore_different_sessions_together`).
    """

    return zscore_different_sessions_together(
        data_dict=processed_beh_dict,
        feature_lst=suffixes,
        feature_bounds=feature_bounds,
        abs_features=abs_features
    )


def pool_session_arrays(feature_data: dict,
                        session_list,
                        pos_key: str,
                        neg_key: str,
                        n_frames: int) -> tuple:
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

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        `(X_pos, X_neg)`, each of shape `(n_epochs, n_frames)`. Either array
        may have `n_epochs == 0` when no non-empty contribution was found for
        that class.
    """

    pos_list = []
    neg_list = []
    for sess_id in session_list:
        if sess_id not in feature_data:
            continue
        p_arr = feature_data[sess_id][pos_key]
        n_arr = feature_data[sess_id][neg_key]
        if p_arr is not None and p_arr.size > 0:
            pos_list.append(p_arr)
        if n_arr is not None and n_arr.size > 0:
            neg_list.append(n_arr)

    X_pos = np.concatenate(pos_list, axis=0) if pos_list else np.empty((0, n_frames))
    X_neg = np.concatenate(neg_list, axis=0) if neg_list else np.empty((0, n_frames))
    return X_pos, X_neg


def balance_two_class_arrays(X_pos: np.ndarray,
                             X_neg: np.ndarray) -> tuple:
    """
    Down-samples the majority class so both classes have equal sample counts.

    Finds the minority class size `n = min(|X_pos|, |X_neg|)` and randomly
    subsamples the majority class (without replacement) down to `n` rows. The
    random draw uses NumPy's global RNG state, so reproducibility is governed
    by the seed set at the start of the pipeline (see
    `prepare_modeling_sessions`). When either input is empty, both outputs are
    returned as empty arrays preserving their original column count.

    Parameters
    ----------
    X_pos : np.ndarray
        Positive-class samples of shape `(n_pos, n_features)`.
    X_neg : np.ndarray
        Negative-class samples of shape `(n_neg, n_features)`.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        `(X_pos_balanced, X_neg_balanced)`. Both arrays have the same number
        of rows equal to `min(n_pos, n_neg)`. If either input was empty the
        outputs are `(0, X_pos.shape[1])` and `(0, X_neg.shape[1])`
        respectively.
    """

    n_pos = X_pos.shape[0]
    n_neg = X_neg.shape[0]

    if n_pos == 0 or n_neg == 0:
        return np.empty((0, X_pos.shape[1])), np.empty((0, X_neg.shape[1]))

    n_samples = min(n_pos, n_neg)

    if n_pos > n_samples:
        pos_indices = np.random.choice(n_pos, n_samples, replace=False)
        X_pos = X_pos[pos_indices]

    if n_neg > n_samples:
        neg_indices = np.random.choice(n_neg, n_samples, replace=False)
        X_neg = X_neg[neg_indices]

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
                              y_test: np.ndarray) -> tuple:
    """
    Randomly permutes the train and test blocks of a split independently.

    After a binary-classification split generator has assembled the balanced
    train and unbalanced-or-balanced test arrays, the rows of both blocks
    are still ordered as `[positives, negatives]`. Downstream models that do
    not perform their own mini-batch shuffling (or that take the data order
    as the optimization order) benefit from a final permutation step. This
    helper generates one permutation index per block (sampled via NumPy's
    global RNG, so reproducibility is governed by the pipeline-level seed
    set in `prepare_modeling_sessions`) and applies it to `X` and `y`
    jointly so the label alignment is preserved.

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

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The permuted `(X_train, y_train, X_test, y_test)` quadruple. Shapes
        are unchanged relative to the inputs.
    """

    train_shuffle_idx = np.random.permutation(X_train.shape[0])
    test_shuffle_idx = np.random.permutation(X_test.shape[0])
    return (X_train[train_shuffle_idx],
            y_train[train_shuffle_idx],
            X_test[test_shuffle_idx],
            y_test[test_shuffle_idx])


def bounded_test_proportion(test_proportion: float,
                            n_sessions: int,
                            min_test_sessions: int = 1) -> float:
    """
    Clamps `test_proportion` up to the minimum fraction required to retain
    at least `min_test_sessions` sessions in the test fold.

    In session-held-out splitting (used by the Onset `session` and
    `session_null_control` strategies), the raw `test_proportion` read from
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
                               n_bins: int = 10) -> float:
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
    n_bins : int, optional
        Number of equal-width confidence bins. Defaults to `10`.

    Returns
    -------
    float
        The top-label expected calibration error on [0, 1].
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim != 2:
        raise ValueError(f"expected_calibration_error requires y_proba of shape (N, K); got {y_proba.shape}.")

    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = float(len(y_true))
    if n_total == 0:
        return float('nan')

    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
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
    mathematically undefined; scikit-learn returns `0.0` in that case and
    this wrapper preserves that behavior.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels of shape (N,).
    y_pred : np.ndarray
        Predicted class labels of shape (N,).

    Returns
    -------
    float
        The Matthews correlation coefficient.
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
        raise ValueError(
            "align_probs_to_canonical requires `canonical_classes` to be "
            "strictly ascending; got an unsorted ordering."
        )
    probs_canonical = np.zeros(
        (n_samples, len(canonical_arr)), dtype=probabilities.dtype
    )
    # `np.searchsorted` finds each model class's position in the sorted
    # canonical ordering in O((K_model + K_canon) log K_canon), replacing
    # the explicit Python for-loop in the callers.
    target_cols = np.searchsorted(canonical_arr, model_classes)
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
                         precomputed_event_times: dict = None,
                         input_metadata: dict = None) -> None:
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
        (recommendation line), `settings['model_params']['gmm_component_index']`
        and `settings['model_params']['gmm_z_score']` (IBI thresholds), and
        `settings['gmm_params']` (per-sex GMM components).
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
    """

    diagnostics_cfg = settings['diagnostics'] if 'diagnostics' in settings else {}
    do_collinearity = diagnostics_cfg['collinearity_audit'] if 'collinearity_audit' in diagnostics_cfg else True
    do_timescale = diagnostics_cfg['timescale_audit'] if 'timescale_audit' in diagnostics_cfg else True

    if not (do_collinearity or do_timescale):
        return

    # Pool per-session event time arrays across the requested event keys.
    # The audit treats every listed key as a "model row" so the summary
    # matrix and the binary event indicator span every epoch the model
    # will ever see. Callers with non-standard event storage can pass
    # `precomputed_event_times` to bypass this loop.
    if precomputed_event_times is not None:
        event_times_per_session = precomputed_event_times
    else:
        event_times_per_session = {}
        for sess_id, track_names in mouse_names_dict.items():
            if sess_id not in usv_data_dict:
                continue
            target_name = track_names[target_idx]
            if target_name not in usv_data_dict[sess_id]:
                continue
            per_mouse = usv_data_dict[sess_id][target_name]

            pooled = []
            for key in event_keys:
                arr = per_mouse[key] if key in per_mouse else None
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.size > 0:
                    pooled.append(arr.ravel())
            if pooled:
                event_times_per_session[sess_id] = np.sort(np.concatenate(pooled))

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
                input_metadata=input_metadata,
            )
        except Exception as exc:
            print(f"[audit] collinearity audit failed (non-fatal): {exc}")

    if do_timescale:
        try:
            # Sex-specific IBI thresholds — same recipe used by the loaders
            # so the audit's headline matches what gates bout detection.
            gmm_idx = settings['model_params']['gmm_component_index']
            gmm_z = settings['model_params']['gmm_z_score']
            gmm_params = settings['gmm_params']
            ibi_thresholds = {}
            for sex in ('male', 'female'):
                params = gmm_params[sex]
                if gmm_idx < len(params['means']):
                    ibi_thresholds[sex] = float(_calculate_ibi_threshold(
                        params['means'][gmm_idx], params['sds'][gmm_idx], gmm_z
                    ))
                else:
                    ibi_thresholds[sex] = float('nan')

            max_lag = float(diagnostics_cfg['timescale_max_lag_seconds']) if 'timescale_max_lag_seconds' in diagnostics_cfg else 10.0
            n_shuffles = int(diagnostics_cfg['timescale_n_shuffles']) if 'timescale_n_shuffles' in diagnostics_cfg else 50
            random_seed = settings['model_params']['random_seed'] if settings['model_params']['random_seed'] is not None else 0

            audit_predictor_timescales(
                processed_beh_dict=processed_beh_dict,
                event_times_per_session=event_times_per_session,
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
            )
        except Exception as exc:
            print(f"[audit] timescale audit failed (non-fatal): {exc}")
