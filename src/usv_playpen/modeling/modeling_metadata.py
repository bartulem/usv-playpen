"""
@author: bartulem
Centralized provenance / metadata builders for the three-level modeling
pipeline.

The modeling stack writes three separate kinds of artifacts, each one
downstream of the previous:

    Level 1 — modeling input pickle
        Output of `extract_and_save_*` in each per-pipeline module.
        Produced once per (analysis_type, experimental cohort, target
        sex) combination. Carries the cohort, the kept feature zoo, the
        temporal frame, and analysis-specific knobs like target_category
        / target_variable / target_vocal_type.

    Level 2 — consolidated univariate result
        Output of `consolidate_univariate_results.py`, fed by N
        per-feature pickles written by `main_univariate_dispatcher.py`.
        Carries everything from Level 1 plus the run-level
        regularization / optimizer / inner-CV configuration that was
        actually used for fitting.

    Level 3 — consolidated model-selection result
        Output of `consolidate_model_selection_results.py`, fed by the
        per-step pickles written by `bout_onset_model_selection`,
        `vocal_category_model_selection`, and `bout_param_model_selection`.
        Carries everything from Level 1 and Level 2 plus the
        forward-stepwise selection knobs.

This module exposes one builder per level (`build_input_metadata`,
`build_run_metadata`, `build_selection_metadata`) plus the shared
provenance helpers (`derive_experimental_condition`,
`compute_settings_sha256`, `get_git_commit_info`, `get_package_version`)
that every builder calls. It also defines the reserved-key vocabulary
(`_input_metadata`, `_run_metadata`, `_univariate_metadata`) used to
embed metadata blocks alongside the actual data inside each artifact.

Design choices baked in here:

- **Schema versioning**: every metadata block carries
  `_schema_version: int`. Consolidators warn on a known older version,
  abort on an unknown one. This is the only field whose format is
  forward-compatible — bump the version whenever the rest of the schema
  changes incompatibly.

- **Embedding policy**: per-feature univariate pickles carry both the
  upstream `_input_metadata` and the just-built `_run_metadata`, so each
  per-feature file is independently provenance-complete. The
  consolidator hoists both blocks to the top of the consolidated artifact
  after asserting equality across files.

- **Experimental-condition derivation**: parsed from the session-list
  filename (basename of `modeling_settings['io']['session_list_file']`).
  The label is **target-centric**, matching the `male_mute_partner`
  convention. For the intact-partner cohorts the target sex is computed
  as `1 - model_predictor_mouse_index` (target is the opposite of the
  predictor). See `derive_experimental_condition` for the exact rules.

- **Reserved keys**: writers must call `assert_no_reserved_keys` on
  their payload before injecting metadata, so a behavioral feature key
  can never silently collide with one of the metadata keys.

- **Best-effort provenance**: `get_git_commit_info` and
  `get_package_version` swallow every exception and return placeholder
  values instead — provenance metadata must never break a fit run.
"""

import hashlib
import json
import pickle
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: Reserved top-level keys inside any modeling artifact. Behavioral feature
#: names are forbidden from starting with `_` so a feature key can never
#: collide with one of these metadata blocks.
RESERVED_METADATA_KEYS = ('_input_metadata', '_run_metadata',
                          '_univariate_metadata', '_consolidation_metadata')

#: Schema version per metadata block. Bump the corresponding entry whenever
#: the on-disk shape of that block changes incompatibly.
SCHEMA_VERSIONS = {
    'input': 1,
    'run': 1,
    'selection': 1,
    'consolidation': 1,
}


def get_package_version() -> str:
    """
    Returns the installed `usv_playpen` package version, or `'unknown'`
    if the version file cannot be imported (e.g., when running from an
    in-place source tree without `hatch-vcs` having stamped a version).

    The version is read once per call from `usv_playpen.__version__`,
    which `pyproject.toml` configures via `hatch-vcs`. Any import or
    attribute error yields the string `'unknown'`; this function
    therefore never raises.

    Returns
    -------
    str
        Version string, or `'unknown'` on failure.
    """

    try:
        from .. import __version__
        return str(__version__)
    except Exception:
        return 'unknown'


def get_git_commit_info(repo_root: str = None) -> dict:
    """
    Returns a `{'commit': str, 'dirty': bool}` dictionary describing the
    git HEAD of the repo containing this file (or `repo_root` if
    supplied), best-effort.

    The function shells out to `git rev-parse HEAD` and `git status
    --porcelain`. Any failure (non-git directory, missing `git` binary,
    permission error, timeout) returns the placeholder pair
    `{'commit': 'unknown', 'dirty': False}` rather than raising — the
    metadata block must never break a pipeline run.

    Parameters
    ----------
    repo_root : str, optional
        Directory to treat as the git working tree. Defaults to the
        directory containing this module, which is the project package
        root for any source-checkout install.

    Returns
    -------
    dict
        `{'commit': '<7-or-40-char-sha-or-unknown>', 'dirty': bool}`.
    """

    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parent)

    info = {'commit': 'unknown', 'dirty': False}
    try:
        commit = subprocess.run(
            ['git', '-C', repo_root, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if commit.returncode == 0:
            info['commit'] = commit.stdout.strip()
        status = subprocess.run(
            ['git', '-C', repo_root, 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0:
            info['dirty'] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def compute_settings_sha256(settings_source) -> str:
    """
    Hashes the modeling-settings configuration so two artifacts with
    distinct settings can be told apart even when the exposed builder
    fields happen to coincide.

    Two input forms are supported:

    - A path-like string pointing at the `modeling_settings.json` file
      that the pipeline actually read. The file's bytes are hashed
      verbatim, so whitespace / key ordering / trailing-newline
      differences all surface as different SHAs (which is what we want
      — the hash represents the on-disk artifact, not the in-memory
      dict).
    - A `dict` already loaded into memory. The dict is serialized with
      `json.dumps(..., sort_keys=True, default=str)` to produce a
      canonical byte form before hashing. Use this form when the
      settings dict was constructed in-memory rather than read from
      disk.

    Parameters
    ----------
    settings_source : str or dict
        Either an absolute path to the JSON file or the loaded dict.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest, or `'unknown'` if reading /
        serializing the source raised.
    """

    h = hashlib.sha256()
    try:
        if isinstance(settings_source, dict):
            payload = json.dumps(settings_source, sort_keys=True,
                                 default=str).encode('utf-8')
            h.update(payload)
        else:
            with Path(settings_source).open('rb') as fh:
                for chunk in iter(lambda: fh.read(65536), b''):
                    h.update(chunk)
        return h.hexdigest()
    except Exception:
        return 'unknown'


def assert_no_reserved_keys(payload: dict, reserved: tuple = None) -> None:
    """
    Raises `ValueError` if `payload` contains any of the reserved
    metadata keys. Writers should call this on the data dict *before*
    injecting metadata, so a behavioral feature key whose name happens
    to start with `_` is detected loudly rather than silently
    overwritten.

    Parameters
    ----------
    payload : dict
        The data dict the writer is about to embed metadata into.
    reserved : tuple of str, optional
        Override the default `RESERVED_METADATA_KEYS` tuple. Pass a
        narrower tuple when only a subset of the metadata blocks will
        be embedded by the caller.

    Raises
    ------
    ValueError
        If any reserved key already exists in `payload`.
    """

    if reserved is None:
        reserved = RESERVED_METADATA_KEYS
    collisions = [k for k in payload.keys() if k in reserved]
    if collisions:
        raise ValueError(
            "Refusing to inject metadata: payload already contains "
            f"reserved keys {collisions}. Reserved key set: {reserved}."
        )


def _utcnow_iso() -> str:
    """
    Returns the current UTC time in ISO-8601 format with seconds
    precision and a trailing `Z`, e.g. `'2026-04-27T13:42:11Z'`.

    The format is chosen to be unambiguous across timezones and
    locale-stable. Used as the timestamp in every metadata block.

    Returns
    -------
    str
        Current UTC timestamp string.
    """

    return (datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace('+00:00', 'Z'))


def derive_experimental_condition(modeling_settings: dict) -> str:
    """
    Determines the experimental-cohort label for the current run.

    The label is **target-centric**, matching the `male_mute_partner` /
    `female_mute_partner` folder convention used by the upstream session
    builder. Two cohort families are supported:

    1. **Intact-partner cohorts** — the session-list filename contains
       the substring `'intact_partners'` (case-insensitive). The target
       sex is the opposite of `model_params['model_predictor_mouse_index']`
       (mouse index 0 is conventionally male, 1 is conventionally
       female across this project). The label returned is
       `'intact_partners_male'` or `'intact_partners_female'`.

    2. **Mute-partner cohorts** — the session-list filename contains
       either `'male_mute_partner'` or `'female_mute_partner'` directly.
       The substring is returned verbatim.

    If the filename matches none of the above patterns, the function
    returns `'unspecified'` rather than raising — the label is purely
    descriptive provenance and an unrecognised cohort should not abort
    a pipeline run.

    Parameters
    ----------
    modeling_settings : dict
        The fully loaded `modeling_settings.json` dictionary. Must
        contain `'io' -> 'session_list_file'` (path string) and, for
        the intact-partner branch, `'model_params' ->
        'model_predictor_mouse_index'` (int).

    Returns
    -------
    str
        One of `'intact_partners_male'`, `'intact_partners_female'`,
        `'male_mute_partner'`, `'female_mute_partner'`, or
        `'unspecified'`.
    """

    session_list_path = modeling_settings['io']['session_list_file']
    fname = Path(session_list_path).name.lower()

    # Mute-partner labels appear verbatim in the filename — return them
    # directly without consulting the settings dict. The female check
    # MUST come first because `'male_mute_partner' in 'female_mute_partner'`
    # is True (substring overlap).
    if 'female_mute_partner' in fname:
        return 'female_mute_partner'
    if 'male_mute_partner' in fname:
        return 'male_mute_partner'

    # Intact cohorts: target is the *opposite* of the predictor mouse,
    # since the project convention is mouse_idx 0 = male, 1 = female.
    if 'intact_partners' in fname:
        pred_idx = modeling_settings['model_params']['model_predictor_mouse_index']
        target_sex = 'male' if pred_idx == 1 else 'female'
        return f"intact_partners_{target_sex}"

    return 'unspecified'


def build_input_metadata(modeling_settings: dict,
                         analysis_type: str,
                         analysis_tag: str,
                         pipeline_class: str,
                         target_idx: int,
                         predictor_idx: int,
                         n_sessions_used: int,
                         session_ids: list,
                         n_events_per_session: dict,
                         feature_zoo_full: list,
                         feature_zoo_kept: list,
                         dyadic_engagement_features_used: list,
                         dyadic_pose_symmetric_features_used: bool,
                         noise_vocal_categories_excluded: list,
                         vocal_signal_columns_added: list,
                         filter_history_seconds: float,
                         filter_history_frames: int,
                         camera_sampling_rate_hz,
                         ibi_thresholds: dict,
                         analysis_specific: dict,
                         settings_path: str = None) -> dict:
    """
    Builds the Level-1 (`_input_metadata`) provenance block for a
    modeling input pickle.

    The block is intentionally self-contained: every downstream artifact
    (consolidated univariate, per-step selection, consolidated
    selection) carries a verbatim copy so a single artifact at any
    level can be interpreted without consulting the original input
    pickle. Every field is explicit — no `.get()` defaults, no
    inference at read time.

    Field groups
    ------------
    - **Cohort / experimental scope** (`experimental_condition`,
      `target_idx`, `target_mouse_sex`, `predictor_idx`,
      `predictor_mouse_sex`, `n_sessions_used`, `session_ids`,
      `n_events_per_session`).
    - **Behavioral feature provenance** (`feature_zoo_full`,
      `feature_zoo_kept`, `dyadic_engagement_features_used`,
      `dyadic_pose_symmetric_features_used`,
      `noise_vocal_categories_excluded`).
    - **Vocal-input shape** (`usv_predictor_type`,
      `usv_predictor_partner_only`, `usv_predictor_smoothing_sd`,
      `vocal_signal_columns_added`).
    - **Temporal frame** (`filter_history_seconds`,
      `filter_history_frames`, `camera_sampling_rate_hz`,
      `gmm_component_index`, `gmm_z_score`, `ibi_thresholds`).
    - **Analysis-specific knobs** (whatever the caller passes through
      `analysis_specific`).
    - **Provenance** (`pipeline_class`, `analysis_type`,
      `analysis_tag`, `git_commit`, `git_dirty`, `created_utc`,
      `package_version`, `settings_sha256`, `_schema_version`).

    Parameters
    ----------
    modeling_settings : dict
        Fully loaded settings JSON.
    analysis_type : str
        Internal pipeline tag — one of `'onset'`, `'category'`,
        `'params'`, `'multinomial'`, `'continuous'`. Used by the
        dispatcher to route to the right runner.
    analysis_tag : str
        Filename-friendly analysis identifier — e.g. `'onsets'`,
        `'category-vocalization-2'`, `'boutparam-mean_mask_complexity'`,
        `'multinomial'`, `'manifold'`. Used to build the Level-1 / -2 /
        -3 filenames.
    pipeline_class : str
        Class name of the pipeline that wrote this artifact (e.g.
        `'VocalOnsetModelingPipeline'`).
    target_idx, predictor_idx : int
        Mouse-slot indices (0 or 1).
    n_sessions_used : int
        Number of sessions that survived per-pipeline filtering and
        contributed at least one event.
    session_ids : list of str
        Basenames of the sessions that contributed.
    n_events_per_session : dict
        Mapping `session_id -> int` of pooled event counts. The
        downstream consolidators do not use this field; it is provided
        purely for inspection / re-balancing.
    feature_zoo_full : list of str
        Generic feature names in the project zoo *before* any
        per-pipeline filtering.
    feature_zoo_kept : list of str
        Generic feature names actually present in the saved input
        pickle (after `harmonize_session_columns`). Always a subset of
        `feature_zoo_full`.
    dyadic_engagement_features_used : list of str
        Names of the dyadic engagement features (from
        `kinematic_features.dyadic_engagement`) that survived per-
        pipeline filtering — usually `['orofacial-sei', 'anogenital-sei']`
        or a subset. Empty list when the pipeline disabled them.
    dyadic_pose_symmetric_features_used : bool
        The `kinematic_features.dyadic_pose_symmetric` flag that was
        active during extraction.
    noise_vocal_categories_excluded : list of int
        GMM-supercategory codes stripped at load time
        (`vocal_features.usv_noise_categories`).
    vocal_signal_columns_added : list of str
        Vocal-history column names injected into the per-session DFs by
        `build_vocal_signal_columns`. Empty when `usv_predictor_type`
        is `null`.
    filter_history_seconds : float
        Pre-event window length in seconds.
    filter_history_frames : int
        Same window in frames.
    camera_sampling_rate_hz : float or dict
        Either a single float (uniform fps across sessions, the project
        default) or a `{session_id: float}` map (heterogeneous fps).
    ibi_thresholds : dict
        Pre-computed `{'male': float, 'female': float}` IBI-gap
        thresholds derived from the GMM means / SDs at the configured
        z-score. Stored once here so downstream consumers do not
        recompute them.
    analysis_specific : dict
        Per-analysis knobs. Suggested keys:
        - onsets: `{'model_target_vocal_type', 'usv_count_threshold'}`
        - category: `{'target_category'}`
        - multinomial: `{'categories_kept', 'class_counts'}`
        - bout-params: `{'target_variable'}`
        - manifold: `{'usv_manifold_column_names'}`
    settings_path : str, optional
        Filesystem path to the settings JSON. When supplied the
        SHA-256 is computed on the file's bytes; otherwise it is
        computed on the in-memory dict's canonical JSON serialization.

    Returns
    -------
    dict
        The `_input_metadata` block, ready to be embedded under the
        reserved key `'_input_metadata'` of the saved artifact.
    """

    target_sex = 'male' if target_idx == 0 else 'female'
    predictor_sex = 'male' if predictor_idx == 0 else 'female'

    # Vocal-feature provenance — the three knobs that determine what the
    # vocal predictor channel actually looks like.
    voc_settings = modeling_settings['vocal_features']
    usv_predictor_type = voc_settings['usv_predictor_type']
    usv_predictor_partner_only = voc_settings['usv_predictor_partner_only']
    usv_predictor_smoothing_sd = voc_settings['usv_predictor_smoothing_sd']

    git_info = get_git_commit_info()
    settings_source = settings_path if settings_path is not None else modeling_settings

    metadata = {
        '_schema_version': SCHEMA_VERSIONS['input'],

        # Cohort / experimental scope
        'experimental_condition': derive_experimental_condition(modeling_settings),
        'target_idx': int(target_idx),
        'target_mouse_sex': target_sex,
        'predictor_idx': int(predictor_idx),
        'predictor_mouse_sex': predictor_sex,
        'n_sessions_used': int(n_sessions_used),
        'session_ids': list(session_ids),
        'n_events_per_session': dict(n_events_per_session),

        # Behavioral feature provenance
        'feature_zoo_full': list(feature_zoo_full),
        'feature_zoo_kept': list(feature_zoo_kept),
        'dyadic_engagement_features_used': list(dyadic_engagement_features_used),
        'dyadic_pose_symmetric_features_used': bool(dyadic_pose_symmetric_features_used),
        'noise_vocal_categories_excluded': list(noise_vocal_categories_excluded),

        # Vocal-input shape
        'usv_predictor_type': usv_predictor_type,
        'usv_predictor_partner_only': bool(usv_predictor_partner_only),
        'usv_predictor_smoothing_sd': usv_predictor_smoothing_sd,
        'vocal_signal_columns_added': list(vocal_signal_columns_added),

        # Temporal frame
        'filter_history_seconds': float(filter_history_seconds),
        'filter_history_frames': int(filter_history_frames),
        'camera_sampling_rate_hz': camera_sampling_rate_hz,
        'gmm_component_index': int(modeling_settings['model_params']['gmm_component_index']),
        'gmm_z_score': float(modeling_settings['model_params']['gmm_z_score']),
        'ibi_thresholds': dict(ibi_thresholds),

        # Analysis-specific knobs (passed through verbatim)
        'analysis_specific': dict(analysis_specific),

        # Provenance
        'analysis_type': analysis_type,
        'analysis_tag': analysis_tag,
        'pipeline_class': pipeline_class,
        'session_list_file': modeling_settings['io']['session_list_file'],
        'git_commit': git_info['commit'],
        'git_dirty': git_info['dirty'],
        'created_utc': _utcnow_iso(),
        'package_version': get_package_version(),
        'settings_sha256': compute_settings_sha256(settings_source),
    }

    return metadata


def build_run_metadata(modeling_settings: dict,
                       analysis_type: str,
                       null_strategy: str,
                       n_outer_folds: int,
                       split_strategy: str,
                       settings_path: str = None) -> dict:
    """
    Builds the Level-2 (`_run_metadata`) provenance block for a single
    per-feature univariate fit.

    The block captures everything about *how* the fit was configured —
    the engine, the regularization knobs, the inner-CV grid, the
    optimizer hyperparameters, the null strategy, and the outer fold
    layout. Two per-feature pickles whose `_run_metadata` differ on any
    field cannot be merged into the same consolidated artifact;
    `consolidate_univariate_results.py` enforces this by structural
    equality.

    The block is **timestamp-free** by design: the per-feature pickle's
    own filesystem mtime, and the consolidated artifact's own
    `_consolidation_metadata.consolidated_at_utc`, are sufficient time
    references. Embedding a per-file timestamp inside `_run_metadata`
    would cause every per-feature pickle to disagree on a field whose
    only purpose is bookkeeping.

    Field groups
    ------------
    - **Engine** (`analysis_type`, `model_engine`, `basis_function`).
    - **JAX hyperparameters** — only populated for analyses that use
      the JAX path (multinomial, continuous): `bin_resizing_factor`,
      `lambda_smooth_fixed`, `l2_reg_fixed`, `smoothness_derivative_order`,
      `learning_rate`, `max_iter`, `tol`, `random_state`,
      `use_lax_loop`, `tune_regularization_bool`. Plus
      `focal_loss_gamma` for multinomial only.
    - **Inner-CV grid** — populated when `tune_regularization_bool` is
      true: `lambda_smooth_decades_each_side`,
      `l2_reg_decades_each_side`, `inner_cv_folds`,
      `inner_cv_scoring_metric`, `inner_cv_use_one_se_rule`,
      `inner_max_iter`.
    - **Outer-loop layout** (`null_strategy`, `n_outer_folds`,
      `split_strategy`).
    - **Provenance** (`git_commit`, `git_dirty`, `package_version`,
      `settings_sha256`, `_schema_version`).

    Parameters
    ----------
    modeling_settings : dict
        Fully loaded settings JSON.
    analysis_type : str
        Same `analysis_type` as in the input metadata: `'onset'`,
        `'category'`, `'params'`, `'multinomial'`, `'continuous'`.
    null_strategy : str
        Description of how the null distribution is built — typically
        `'x_history_shuffle'` for the JAX paths.
    n_outer_folds : int
        Number of outer CV splits used for the actual / null
        comparison. Pulled from the dispatcher / pipeline at run time.
    split_strategy : str
        `'mixed'`, `'session'`, etc., as resolved by the dispatcher.
    settings_path : str, optional
        Filesystem path to the settings JSON for SHA-256 hashing. See
        `compute_settings_sha256`.

    Returns
    -------
    dict
        The `_run_metadata` block, ready to be embedded under the
        reserved key `'_run_metadata'` of the per-feature pickle.
    """

    model_params = modeling_settings['model_params']
    hp_root = modeling_settings['hyperparameters']

    metadata = {
        '_schema_version': SCHEMA_VERSIONS['run'],
        'analysis_type': analysis_type,
        'model_engine': model_params['model_engine'],
        'basis_function': model_params['model_basis_function'],
        'null_strategy': null_strategy,
        'n_outer_folds': int(n_outer_folds),
        'split_strategy': split_strategy,
        'random_seed_outer': int(model_params['random_seed']),
        'spatial_cluster_num': int(model_params['spatial_cluster_num']),
        'test_proportion': float(model_params['test_proportion']),
        'session_split_max_attempts': int(model_params['session_split_max_attempts']),
        'session_split_widen_step': float(model_params['session_split_widen_step']),
        'session_split_widen_every': int(model_params['session_split_widen_every']),
    }

    # JAX-path knobs — populated for analyses that use the GPU runner.
    # The bivariate (continuous) and multinomial blocks differ only in
    # `focal_loss_gamma` and the `balance_*` flags; both share the same
    # core hyperparameter names so we copy through whichever block
    # matches the analysis_type.
    jax_root = hp_root['jax_linear']
    if analysis_type == 'multinomial':
        jax_block = jax_root['multinomial_logistic']
        jax_kind = 'multinomial_logistic'
    elif analysis_type == 'continuous':
        jax_block = jax_root['bivariate']
        jax_kind = 'bivariate'
    else:
        jax_block = None
        jax_kind = None

    if jax_block is not None:
        metadata['jax_hyperparameters'] = {
            'jax_block_kind': jax_kind,
            'bin_resizing_factor': int(jax_block['bin_resizing_factor']),
            'lambda_smooth_fixed': float(jax_block['lambda_smooth_fixed']),
            'l2_reg_fixed': float(jax_block['l2_reg_fixed']),
            'smoothness_derivative_order': int(jax_block['smoothness_derivative_order']),
            'learning_rate': float(jax_block['learning_rate']),
            'max_iter': int(jax_block['max_iter']),
            'tol': float(jax_block['tol']),
            'random_state': int(jax_block['random_state']),
            'use_lax_loop': bool(jax_block['use_lax_loop']),
            'tune_regularization_bool': bool(jax_block['tune_regularization_bool']),
        }
        if jax_kind == 'multinomial_logistic':
            metadata['jax_hyperparameters']['focal_loss_gamma'] = float(jax_block['focal_loss_gamma'])
            metadata['jax_hyperparameters']['balance_predictions_bool'] = bool(jax_block['balance_predictions_bool'])
            metadata['jax_hyperparameters']['balance_train_bool'] = bool(jax_block['balance_train_bool'])
        if jax_block['tune_regularization_bool']:
            tp = jax_block['tune_regularization_params']
            metadata['jax_hyperparameters']['tune_regularization_params'] = {
                'lambda_smooth_decades_each_side': int(tp['lambda_smooth_decades_each_side']),
                'l2_reg_decades_each_side': int(tp['l2_reg_decades_each_side']),
                'inner_cv_folds': int(tp['inner_cv_folds']),
                'inner_cv_scoring_metric': tp['inner_cv_scoring_metric'],
                'inner_cv_use_one_se_rule': bool(tp['inner_cv_use_one_se_rule']),
                'inner_max_iter': int(tp['inner_max_iter']),
            }

    # CPU-path knobs — populated for sklearn / pygam paths (onset,
    # category, params). The pygam block carries n_splines_*, lam,
    # max_iterations, tol, distribution, link.
    if analysis_type in ('onset', 'category', 'params'):
        if model_params['model_engine'] == 'pygam':
            pgm = hp_root['classical']['pygam']
            metadata['pygam_hyperparameters'] = {
                'n_splines_time': int(pgm['n_splines_time']),
                'n_splines_value': int(pgm['n_splines_value']),
                'lam_penalty': float(pgm['lam_penalty']),
                'max_iterations': int(pgm['max_iterations']),
                'tol_val': float(pgm['tol_val']),
                'distribution': pgm['distribution'],
                'link': pgm['link'],
            }
        elif model_params['model_engine'] == 'sklearn':
            basis = model_params['model_basis_function']
            metadata['sklearn_hyperparameters'] = {
                'basis_function': basis,
                'basis_function_params': dict(hp_root['basis_functions'][basis]),
            }
            if analysis_type == 'onset' or analysis_type == 'category':
                metadata['sklearn_hyperparameters']['logistic_regression'] = dict(
                    hp_root['classical']['logistic_regression']
                )
            elif analysis_type == 'params':
                metadata['sklearn_hyperparameters']['ridge_regression'] = dict(
                    hp_root['classical']['ridge_regression']
                )

    # Provenance footer
    git_info = get_git_commit_info()
    settings_source = settings_path if settings_path is not None else modeling_settings
    metadata['git_commit'] = git_info['commit']
    metadata['git_dirty'] = git_info['dirty']
    metadata['package_version'] = get_package_version()
    metadata['settings_sha256'] = compute_settings_sha256(settings_source)

    return metadata


def build_selection_metadata(modeling_settings: dict,
                             selection_function: str,
                             selection_metric: str,
                             n_splits_selection: int,
                             test_proportion: float,
                             split_strategy: str,
                             random_seed: int,
                             one_se_rule_used: bool,
                             aic_termination_used: bool,
                             n_anchor_features: int,
                             anchor_feature: str,
                             gam_kwargs: dict,
                             extra_knobs: dict = None,
                             settings_path: str = None) -> dict:
    """
    Builds the Level-3 (`_run_metadata` for selection artifacts)
    provenance block for a forward-stepwise model-selection run.

    The block captures the selection-side configuration only; the
    matching `_input_metadata` and `_univariate_metadata` blocks are
    embedded as siblings inside the selection artifact (the writer
    copies them verbatim from the input modeling pickle and the
    consolidated univariate file respectively).

    Field groups
    ------------
    - **Selection function** (`selection_function`, `selection_metric`).
    - **Outer CV layout** (`n_splits_selection`, `test_proportion`,
      `split_strategy`, `random_seed`).
    - **Termination policy** (`one_se_rule_used`,
      `aic_termination_used`).
    - **Anchor / GAM config** (`n_anchor_features`, `anchor_feature`,
      `gam_kwargs`).
    - **Caller-supplied extras** (`extra_knobs` — any selector-specific
      flags the caller wants frozen with the artifact).
    - **Provenance** (`git_commit`, `git_dirty`, `package_version`,
      `settings_sha256`, `_schema_version`).

    Parameters
    ----------
    modeling_settings : dict
        Fully loaded settings JSON.
    selection_function : str
        Name of the entry-point function that built this artifact —
        `'bout_onset_model_selection'`, `'vocal_category_model_selection'`,
        or `'bout_param_model_selection'`.
    selection_metric : str
        Primary scoring metric used to rank features at each step
        (`'AUC'`, `'Brier'`, `'r2_spatial'`, etc.).
    n_splits_selection : int
        Number of outer CV splits used per step.
    test_proportion : float
        Fraction of samples / sessions in the test fold.
    split_strategy : str
        `'mixed'` or `'session'`.
    random_seed : int
        Seed for the outer splitter.
    one_se_rule_used : bool
        Whether the 1-SE rule trimmed the chosen step.
    aic_termination_used : bool
        Whether AIC-based early termination was active.
    n_anchor_features : int
        Number of features that survived the anchor-set construction.
    anchor_feature : str
        Generic name of the feature whose pooled `(n_pos, n_neg)` /
        `(n_targ, n_other)` shape governed the mixed-fold construction.
    gam_kwargs : dict
        Keyword arguments handed to `LogisticGAM` / `LinearGAM` at fit
        time (n_splines, lam, distribution, link).
    extra_knobs : dict, optional
        Any selector-specific knobs the caller wants persisted.
    settings_path : str, optional
        Filesystem path to the settings JSON for SHA-256 hashing.

    Returns
    -------
    dict
        The selection-level `_run_metadata` block.
    """

    git_info = get_git_commit_info()
    settings_source = settings_path if settings_path is not None else modeling_settings

    metadata = {
        '_schema_version': SCHEMA_VERSIONS['selection'],
        'selection_function': selection_function,
        'selection_metric': selection_metric,
        'n_splits_selection': int(n_splits_selection),
        'test_proportion': float(test_proportion),
        'split_strategy': split_strategy,
        'random_seed': int(random_seed),
        'one_se_rule_used': bool(one_se_rule_used),
        'aic_termination_used': bool(aic_termination_used),
        'n_anchor_features': int(n_anchor_features),
        'anchor_feature': anchor_feature,
        'gam_kwargs': dict(gam_kwargs),
        'extra_knobs': dict(extra_knobs) if extra_knobs is not None else {},
        'git_commit': git_info['commit'],
        'git_dirty': git_info['dirty'],
        'package_version': get_package_version(),
        'settings_sha256': compute_settings_sha256(settings_source),
    }

    return metadata


def build_consolidation_metadata(n_files_merged: int,
                                 individual_file_paths: list,
                                 individual_file_timestamps: list,
                                 consolidator_name: str,
                                 consolidator_version: int) -> dict:
    """
    Builds the `_consolidation_metadata` block written by the Level-2
    and Level-3 consolidators alongside the hoisted `_input_metadata` /
    `_run_metadata` / `_univariate_metadata` blocks.

    Unlike the upstream metadata blocks, this one is timestamped (with
    the moment of consolidation) and lists every per-feature / per-step
    file that was merged, so a downstream consumer can audit what went
    into a consolidated artifact.

    Parameters
    ----------
    n_files_merged : int
        Number of per-feature / per-step pickles consolidated.
    individual_file_paths : list of str
        Absolute paths of the consolidated files, in the order they
        were merged.
    individual_file_timestamps : list of str
        ISO-8601 modification timestamps of the consolidated files,
        same ordering as `individual_file_paths`.
    consolidator_name : str
        Identifier of the consolidator that produced this artifact
        (e.g. `'consolidate_univariate_results'`,
        `'consolidate_model_selection_results'`).
    consolidator_version : int
        Internal version number of the consolidator. Bump whenever the
        consolidator's behavior changes incompatibly so old consolidated
        artifacts can be filtered out by readers.

    Returns
    -------
    dict
        The `_consolidation_metadata` block.
    """

    return {
        '_schema_version': SCHEMA_VERSIONS['consolidation'],
        'consolidator_name': consolidator_name,
        'consolidator_version': int(consolidator_version),
        'consolidated_at_utc': _utcnow_iso(),
        'n_files_merged': int(n_files_merged),
        'individual_file_paths': list(individual_file_paths),
        'individual_file_timestamps': list(individual_file_timestamps),
    }


def extract_metadata_blocks(modeling_data: dict) -> tuple:
    """
    Splits a loaded modeling artifact into its data payload and its
    metadata blocks.

    All Level-1 / Level-2 / Level-3 artifacts share a flat dict layout
    where reserved-key blocks (`_input_metadata`, `_run_metadata`,
    `_univariate_metadata`, `_consolidation_metadata`) sit alongside
    the actual feature data. This helper pops every reserved block
    into a separate dict so callers can iterate the data dict's
    feature keys without contamination.

    Behavioral feature names are forbidden from starting with `_` by
    convention; the helper warns (does not raise) if any non-reserved
    underscore-prefixed key is encountered, since that almost certainly
    indicates a future schema field that should be added to
    `RESERVED_METADATA_KEYS`.

    Parameters
    ----------
    modeling_data : dict
        The full dict returned by `pickle.load` on a Level-1 / -2 / -3
        artifact.

    Returns
    -------
    tuple
        `(clean_data, metadata_blocks)` where `clean_data` is a shallow
        copy of `modeling_data` with reserved keys removed, and
        `metadata_blocks` is a dict containing only the reserved blocks
        actually present (an empty dict for legacy artifacts).
    """

    metadata_blocks = {}
    clean_data = {}
    for k, v in modeling_data.items():
        if k in RESERVED_METADATA_KEYS:
            metadata_blocks[k] = v
        elif isinstance(k, str) and k.startswith('_'):
            print(f"[metadata] WARNING: top-level key {k!r} starts with `_` "
                  "but is not in RESERVED_METADATA_KEYS — treating as data.")
            clean_data[k] = v
        else:
            clean_data[k] = v
    return clean_data, metadata_blocks


def inject_metadata(payload: dict, **metadata_blocks) -> dict:
    """
    Returns a new dict that combines the data payload with the supplied
    metadata blocks under their reserved keys.

    The function asserts no collision between the payload's existing
    keys and the metadata blocks (via `assert_no_reserved_keys`), and
    rejects metadata-block names that are not in
    `RESERVED_METADATA_KEYS`. The original payload is never mutated.

    Typical usage at write time:

        artifact = inject_metadata(
            modeling_final_data_dict,
            _input_metadata=md_in,
        )
        pickle.dump(artifact, fh)

    Parameters
    ----------
    payload : dict
        The data dict (Level-1: feature → session → arrays;
        Level-2: feature → result; Level-3: dict of step results).
    **metadata_blocks
        Keyword arguments whose names must be drawn from
        `RESERVED_METADATA_KEYS` (passed without the leading underscore
        is *not* permitted — the caller writes the underscore name
        explicitly so the embedded key matches the spec).

    Returns
    -------
    dict
        A new dict combining payload and metadata. Order is not
        guaranteed but underscore-prefixed keys conventionally sort
        last in Python dict iteration when iterated in insertion
        order.

    Raises
    ------
    ValueError
        If any supplied block name is not reserved, or any reserved
        key already exists in `payload`.
    """

    for name in metadata_blocks.keys():
        if name not in RESERVED_METADATA_KEYS:
            raise ValueError(
                f"Refusing to inject metadata under non-reserved key "
                f"{name!r}. Permitted names: {RESERVED_METADATA_KEYS}."
            )
    assert_no_reserved_keys(payload, reserved=tuple(metadata_blocks.keys()))
    out = dict(payload)
    out.update(metadata_blocks)
    return out


def derive_feature_zoo_full(modeling_settings: dict,
                            include_egocentric_per_mouse: bool = True) -> list:
    """
    Returns the *full* generic feature-name list implied by the
    project's kinematic settings, before any per-pipeline filtering.

    The egocentric block in `kinematic_features.egocentric` lists
    bare suffixes (e.g. `'speed'`, `'neck_elevation'`) that are
    instantiated once per mouse as `self.<suffix>` and `other.<suffix>`.
    The dyadic_pose and dyadic_engagement blocks list pre-formatted
    pair / engagement names that appear once per dyad.

    Parameters
    ----------
    modeling_settings : dict
        Fully loaded settings JSON. Must contain
        `kinematic_features.egocentric`,
        `kinematic_features.dyadic_pose`, and
        `kinematic_features.dyadic_engagement`.
    include_egocentric_per_mouse : bool, default True
        When True, each egocentric suffix is expanded into its
        `self.<suffix>` and `other.<suffix>` pair. Pass False to
        return the bare suffix list (useful for debug printing).

    Returns
    -------
    list of str
        Sorted list of generic feature names available before any
        pipeline-specific filtering.
    """

    kin = modeling_settings['kinematic_features']
    full = []
    if include_egocentric_per_mouse:
        for f in kin['egocentric']:
            full.append(f"self.{f}")
            full.append(f"other.{f}")
    else:
        full.extend(kin['egocentric'])
    full.extend(kin['dyadic_pose'])
    full.extend(kin['dyadic_engagement'])
    return sorted(full)


def derive_camera_fps_field(camera_fr_dict: dict):
    """
    Reduces the per-session `{session_id: float}` fps map to either a
    single float (when every session shares the same fps, the project
    default) or the original dict (when fps is heterogeneous across
    the cohort).

    Stored in `_input_metadata['camera_sampling_rate_hz']` so a
    downstream consumer can do `if isinstance(field, float)` to
    short-circuit the per-session path.

    Parameters
    ----------
    camera_fr_dict : dict
        Mapping `session_id -> camera_sampling_rate_hz`.

    Returns
    -------
    float or dict
        Single float when all sessions match; original dict otherwise.
    """

    if not camera_fr_dict:
        return {}
    values = set(float(v) for v in camera_fr_dict.values())
    if len(values) == 1:
        return float(next(iter(values)))
    return {k: float(v) for k, v in camera_fr_dict.items()}


def load_selection_results(selection_results_dir) -> tuple:
    """
    Loads a model-selection result set from disk, transparently
    handling both the new consolidated single-file artifact and the
    legacy per-step directory layout.

    Lookup policy
    -------------
    1. The directory is first scanned for any `selection_*.pkl` file
       (the output of `consolidate_model_selection_results`). When one
       or more such files exist, the most-recently-modified one is
       loaded and its `'steps'` list is returned in step order. The
       artifact's basename is returned as `display_name` so downstream
       plotters can keep their existing substring-based sex / cohort
       inference (`'_male_' in display_name` etc.). The consolidated
       artifact's `_input_metadata` / `_run_metadata` /
       `_univariate_metadata` blocks are returned alongside the steps
       list under `metadata` so plotters can short-circuit any
       per-step metadata harvesting they used to do.
    2. When no `selection_*.pkl` exists in the directory, the function
       falls back to the legacy `*_step_*.pkl` glob: every per-step
       pickle is loaded, its reserved metadata blocks are stripped
       (via `extract_metadata_blocks`), and the cleaned step dicts are
       returned in step-index order. The first per-step pickle's
       basename is returned as `display_name`. The metadata block
       returned in that case is harvested from the *first* per-step
       file, mirroring the consolidator's hoist policy.

    Parameters
    ----------
    selection_results_dir : str or pathlib.Path
        Directory containing either the consolidated `selection_*.pkl`
        or the per-step `*_step_*.pkl` files.

    Returns
    -------
    tuple
        `(steps, display_name, metadata)`:

        - `steps` : list of dict
            Per-step result dicts in step order, with reserved
            metadata keys removed. May be empty when the directory
            has neither a consolidated artifact nor any per-step
            files.
        - `display_name` : str
            Basename of the loaded artifact (consolidated file or
            first per-step file). Empty string when nothing was
            loaded. Plotters use this for sex / experimental-condition
            substring matching, which keeps working unchanged because
            the substring `'_male_'` / `'_female_'` appears in both
            the consolidated and the per-step filenames.
        - `metadata` : dict
            The reserved metadata blocks harvested from whichever
            file was loaded. Possible keys are
            `'_input_metadata'`, `'_run_metadata'`,
            `'_univariate_metadata'`, `'_consolidation_metadata'`.
            Empty dict for legacy artifacts that lack metadata.
    """

    sel_dir = Path(selection_results_dir)

    # 1. Consolidated artifact path. Multiple `selection_*.pkl` files
    #    can land in the same directory if the user re-runs the
    #    consolidator across re-runs of the upstream selector; pick the
    #    most-recently-modified one as the canonical artifact.
    cons_candidates = sorted(
        sel_dir.glob('selection_*.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cons_candidates:
        chosen = cons_candidates[0]
        with chosen.open('rb') as fh:
            cons = pickle.load(fh)
        if 'steps' in cons and isinstance(cons['steps'], list):
            metadata = {
                k: v for k, v in cons.items()
                if k in RESERVED_METADATA_KEYS
            }
            return list(cons['steps']), chosen.name, metadata

    # 2. Legacy per-step path.
    step_files = sorted(
        sel_dir.glob('*_step_*.pkl'),
        key=lambda p: int(re.search(r'_step_(\d+)', p.name).group(1)),
    )
    if not step_files:
        return [], '', {}

    steps = []
    metadata = {}
    for fp in step_files:
        with fp.open('rb') as fh:
            payload = pickle.load(fh)
        clean, md = extract_metadata_blocks(payload)
        steps.append(clean)
        if not metadata and md:
            # Capture the first non-empty metadata bundle so the
            # plotters can rely on a consistent provenance source
            # without needing to re-load any per-step file.
            metadata = md
    return steps, step_files[0].name, metadata


def metadata_blocks_equal(a: dict, b: dict, ignore_keys: tuple = ()) -> bool:
    """
    Structural equality check between two metadata blocks, optionally
    ignoring a tuple of top-level keys.

    Two metadata blocks compare equal iff the union of their keys is
    identical and every value compares equal under `==`. Lists and
    dicts compare element-wise; numpy scalars are compared after a
    `float()` / `int()` cast where applicable. Used by the consolidators
    to assert that every per-feature / per-step pickle in a directory
    agrees on the run-level configuration before merging.

    Parameters
    ----------
    a, b : dict
        Metadata blocks to compare.
    ignore_keys : tuple of str
        Top-level keys to skip during comparison (typically per-file
        timestamps).

    Returns
    -------
    bool
        True iff the blocks agree on every non-ignored key.
    """

    keys_a = set(a.keys()) - set(ignore_keys)
    keys_b = set(b.keys()) - set(ignore_keys)
    if keys_a != keys_b:
        return False
    for k in keys_a:
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            if not metadata_blocks_equal(va, vb, ignore_keys=()):
                return False
        else:
            if va != vb:
                return False
    return True
