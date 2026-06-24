"""
@author: bartulem
Reusable synthetic-fixture builders for the USV modeling test-suite.

This module is deliberately NOT a test file (the leading underscore keeps it
out of pytest's collection). It exports a small set of builder functions that
manufacture, on disk and in memory, the *exact* data contract the real
modeling pipeline and HPC dispatchers consume, so end-to-end smoke tests can
drive the production code paths without any real recordings.

The three artifacts the pipeline expects are:

1.  **A session-directory tree** — one directory per session, each containing:
      - ``<session>/video/**/*_points3d_translated_rotated_metric_behavioral_features.csv``
        the per-frame behavioral feature table (polars-readable CSV). Column
        names follow the project's ``{mouse_id}.{base_feature}`` convention for
        egocentric features and ``{m1}-{m2}.{feat}`` for dyadic features.
      - ``<session>/video/**/[!speaker]*_points3d_translated_rotated_metric.h5``
        an HDF5 track file (named with a numeric date-stamp prefix, mirroring
        the real ``YYYYMMDDhhmmss_...`` convention, so it clears the loader's
        ``[!speaker]`` exclusion class) carrying two datasets the loader reads:
        ``recording_frame_rate`` (scalar float) and ``track_names`` (list of
        byte-strings, index 0 == male, index 1 == female by convention).
      - ``<session>/audio/**/*_usv_summary.csv`` the USV summary table with at
        least ``emitter``, ``start``, ``stop`` columns plus the configurable
        category / supercategory / manifold columns.

2.  **A ``modeling_settings`` dict** — a trimmed, hyperparameter-shrunk copy of
    the canonical ``_parameter_settings/modeling_settings.json`` with ``io``
    paths repointed at a caller-supplied scratch directory and the tiny-data
    knobs (few features, few folds, low ``max_iter``) applied.

3.  **A "modeling input pickle"** — the nested
    ``{generic_feature: {session_id: {usv_feature_arr, no_usv_feature_arr}}}``
    dict (plus a reserved ``_input_metadata`` block) that the univariate
    dispatcher and model-selection orchestrators load. Most tests obtain this
    by *running* the real extraction, but ``build_modeling_input_pickle`` is
    provided for tests that want to bypass extraction and inject a controlled
    design matrix directly.

All builders take explicit arguments (no hidden globals) and return the paths
or objects they create, so they compose cleanly inside ``tmp_path``-scoped
tests.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import polars as pls


# Canonical location of the shipped settings JSON, resolved relative to the
# installed package so the synthetic settings inherit every real key.
_SETTINGS_JSON = (
    Path(__file__).resolve().parents[2]
    / 'src' / 'usv_playpen' / '_parameter_settings' / 'modeling_settings.json'
)

# Numeric date-stamp file prefix mirroring the lab's real ``YYYYMMDDhhmmss_...``
# naming. The loaders' track-file glob uses the ``[!speaker]`` exclusion class,
# which (as a single-character negated set) would reject any basename starting
# with one of s/p/e/a/k/r — a digit prefix sidesteps that entirely.
_FILE_PREFIX = '20240101120000'


def build_behavioral_features_csv(
        session_root: Path,
        mouse_names: list[str],
        n_frames: int,
        egocentric_features: list[str],
        dyadic_features: list[str] | None = None,
        engagement_features: list[str] | None = None,
        seed: int = 0,
        csv_sep: str = ',',
) -> Path:
    """
    Description
    -----------
    Writes a synthetic per-frame behavioral-feature CSV under
    ``<session_root>/video`` matching the glob the loader uses
    (``video/**/*_points3d_translated_rotated_metric_behavioral_features.csv``).

    Column naming mirrors the production convention so the kinematic-column
    selector keeps them:
      - egocentric  -> ``{mouse_id}.{base_feature}`` for every mouse.
      - dyadic pose -> ``{m1}-{m2}.{base_feature}`` (single direction only;
        the selector folds the symmetric pair).
      - engagement  -> ``{m1}-{m2}.{base_feature}`` where ``m1`` is the
        observer; both directions are written so the selector can keep the
        ``target-predictor`` orientation it expects.

    All feature values are smooth low-frequency sinusoids plus light noise, so
    the per-event history windows carry non-degenerate, finite variance that
    z-scoring and basis projection can act on.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session ROOT directory; the ``video`` subtree is created if absent.
    mouse_names (list of str)
        Ordered mouse track names (index 0 == male, index 1 == female).
    n_frames (int)
        Number of rows (frames) in the feature table.
    egocentric_features (list of str)
        Base names written per mouse as ``{mouse_id}.{name}``.
    dyadic_features (list of str or None)
        Base names written once as ``{m1}-{m2}.{name}``; ``None`` -> none.
    engagement_features (list of str or None)
        Base names written in both dyad orientations; ``None`` -> none.
    seed (int)
        Seed for the per-column noise generator (reproducible tables).
    csv_sep (str)
        Field separator passed straight to ``polars.write_csv``.

    Returns
    -------
    csv_path (pathlib.Path)
        Absolute path to the CSV that was written.
    """

    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=float)
    columns: dict[str, np.ndarray] = {}

    def _signal(freq: float, phase: float) -> np.ndarray:
        return np.sin(2.0 * np.pi * freq * t / max(n_frames, 1) + phase) + 0.05 * rng.standard_normal(n_frames)

    col_counter = 0
    for base in egocentric_features:
        for m_name in mouse_names:
            col_counter += 1
            columns[f"{m_name}.{base}"] = _signal(freq=col_counter, phase=0.3 * col_counter)

    if dyadic_features:
        dyad_prefix = f"{mouse_names[0]}-{mouse_names[1]}"
        for base in dyadic_features:
            col_counter += 1
            columns[f"{dyad_prefix}.{base}"] = _signal(freq=col_counter, phase=0.1 * col_counter)

    if engagement_features:
        for base in engagement_features:
            for prefix in (f"{mouse_names[0]}-{mouse_names[1]}", f"{mouse_names[1]}-{mouse_names[0]}"):
                col_counter += 1
                columns[f"{prefix}.{base}"] = _signal(freq=col_counter, phase=0.2 * col_counter)

    video_dir = session_root / 'video'
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_path = video_dir / f"{_FILE_PREFIX}_points3d_translated_rotated_metric_behavioral_features.csv"
    pls.DataFrame(columns).write_csv(file=csv_path, separator=csv_sep)
    return csv_path


def build_track_h5(
        session_root: Path,
        mouse_names: list[str],
        camera_fps: float,
) -> Path:
    """
    Description
    -----------
    Writes the HDF5 track-metadata file the loader reads to learn the camera
    frame-rate and the per-session mouse track names. The filename matches the
    loader glob ``video/**/[!speaker]*_points3d_translated_rotated_metric.h5``
    (the leading ``[!speaker]`` exclusion class means the basename must not
    start with the literal ``speaker``).

    Parameters
    ----------
    session_root (pathlib.Path)
        Session ROOT directory; the ``video`` subtree is created if absent.
    mouse_names (list of str)
        Ordered mouse track names stored as UTF-8 byte-strings under
        ``track_names`` (index 0 == male, index 1 == female).
    camera_fps (float)
        Scalar written to the ``recording_frame_rate`` dataset.

    Returns
    -------
    h5_path (pathlib.Path)
        Absolute path to the HDF5 file that was written.
    """

    video_dir = session_root / 'video'
    video_dir.mkdir(parents=True, exist_ok=True)
    h5_path = video_dir / f"{_FILE_PREFIX}_points3d_translated_rotated_metric.h5"
    with h5py.File(name=h5_path, mode='w') as h5_file:
        h5_file.create_dataset('recording_frame_rate', data=float(camera_fps))
        h5_file.create_dataset(
            'track_names',
            data=np.array([m.encode('utf-8') for m in mouse_names], dtype='S64'),
        )
    return h5_path


def build_usv_summary_csv(
        session_root: Path,
        target_mouse: str,
        partner_mouse: str,
        camera_fps: float,
        n_frames: int,
        filter_history: float,
        n_bouts: int = 6,
        usv_per_bout: int = 3,
        category_column: str = 'vae_supercategory',
        manifold_columns: tuple[str, str] = ('vae_umap1', 'vae_umap2'),
        seed: int = 0,
        csv_sep: str = ',',
) -> Path:
    """
    Description
    -----------
    Writes a synthetic ``*_usv_summary.csv`` under ``<session_root>/audio`` that
    yields a controllable number of valid vocal *bouts* for the target mouse
    when run through ``find_bout_epochs`` in ``'bout'`` prediction mode.

    Construction guarantees, per bout:
      - ``usv_per_bout`` syllables packed tightly (gap << IBI threshold) so they
        cluster into a single bout (meets ``usv_per_bout_floor``).
      - bouts placed far apart (gap >> IBI threshold and >> ``filter_history``)
        so each is a distinct, clean-history-eligible positive event.
      - the early portion of the session (the first ``filter_history`` seconds
        plus generous slack) is left silent, so the clean-epoch tiler can carve
        out negative (No-USV) events too.

    A non-noise integer category, two manifold coordinates, a ``vae_category``
    column and a ``mask_number`` column are attached to every row so the
    category / multinomial / continuous / bout-parameter loaders all find what
    they need (even though the onset smoke path only consumes onsets).

    Parameters
    ----------
    session_root (pathlib.Path)
        Session ROOT directory; the ``audio`` subtree is created if absent.
    target_mouse (str)
        Emitter name for all biological USV rows (the focal/target mouse).
    partner_mouse (str)
        Recorded as a second emitter on a couple of rows so partner-only vocal
        predictors are exercised; kept sparse so they do not pollute the
        target's clean negative epochs near its bouts.
    camera_fps (float)
        Camera frame-rate (used only to keep timestamps within the session).
    n_frames (int)
        Total session length in frames (defines the session duration).
    filter_history (float)
        Pre-event history length in seconds; the first window of the session is
        kept silent so early events are never clipped.
    n_bouts (int)
        Number of distinct target-mouse bouts to synthesize.
    usv_per_bout (int)
        Syllables packed into each bout.
    category_column (str)
        Name of the integer category column written (default matches the JSON's
        ``usv_category_column_name`` / ``usv_noise_column``).
    manifold_columns (tuple of str)
        Two column names holding synthetic acoustic-manifold coordinates.
    seed (int)
        Seed for the manifold-coordinate noise.
    csv_sep (str)
        Field separator passed straight to ``polars.write_csv``.

    Returns
    -------
    csv_path (pathlib.Path)
        Absolute path to the CSV that was written.
    """

    rng = np.random.default_rng(seed)
    session_duration_sec = n_frames / camera_fps

    # Leave a silent warm-up region at the start so early bouts are not clipped,
    # then spread the bouts across the remaining session with wide spacing.
    warmup = filter_history * 2.0 + 1.0
    usable = session_duration_sec - warmup - filter_history * 2.0
    if usable <= 0 or n_bouts <= 0:
        raise ValueError("Session too short to host the requested number of bouts.")
    bout_spacing = usable / n_bouts

    emitters: list[str | None] = []
    starts: list[float] = []
    stops: list[float] = []

    syllable_gap = 0.02      # << IBI threshold -> syllables fuse into one bout
    syllable_dur = 0.01

    for b in range(n_bouts):
        bout_origin = warmup + b * bout_spacing
        for s in range(usv_per_bout):
            start = bout_origin + s * (syllable_dur + syllable_gap)
            emitters.append(target_mouse)
            starts.append(round(start, 6))
            stops.append(round(start + syllable_dur, 6))

    # A couple of sparse partner USVs late in the session (after the last
    # target bout) so partner-only vocal predictors have content without
    # contaminating the target's clean negative windows.
    partner_anchor = warmup + (n_bouts - 0.5) * bout_spacing
    for s in range(2):
        start = partner_anchor + s * 0.5
        emitters.append(partner_mouse)
        starts.append(round(start, 6))
        stops.append(round(start + syllable_dur, 6))

    n_rows = len(starts)
    rows = {
        'emitter': emitters,
        'start': starts,
        'stop': stops,
        category_column: [1] * n_rows,            # non-noise category (noise == 0)
        'vae_category': [1] * n_rows,
        'mask_number': [2] * n_rows,
        manifold_columns[0]: (rng.standard_normal(n_rows)).round(6).tolist(),
        manifold_columns[1]: (rng.standard_normal(n_rows)).round(6).tolist(),
    }

    audio_dir = session_root / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audio_dir / f"{session_root.name}_usv_summary.csv"
    pls.DataFrame(rows).write_csv(file=csv_path, separator=csv_sep)
    return csv_path


def build_session_tree(
        base_dir: Path,
        n_sessions: int = 3,
        n_frames: int = 1200,
        camera_fps: float = 150.0,
        filter_history: float = 1.0,
        egocentric_features: list[str] | None = None,
        dyadic_features: list[str] | None = None,
        engagement_features: list[str] | None = None,
        mouse_name_stub: tuple[str, str] = ('m_male', 'm_female'),
        n_bouts: int = 8,
        usv_per_bout: int = 3,
        category_column: str = 'vae_supercategory',
        manifold_columns: tuple[str, str] = ('vae_umap1', 'vae_umap2'),
        csv_sep: str = ',',
) -> list[Path]:
    """
    Description
    -----------
    Builds a complete multi-session synthetic recording tree under
    ``base_dir`` and returns the list of session ROOT directories. Every
    session gets the three on-disk artifacts the loaders require: the
    behavioral-feature CSV, the HDF5 track file, and the USV-summary CSV. Mouse
    track names are uniquified per session (e.g. ``s0_m_male``) so cross-session
    role resolution and the dyad-rename logic are genuinely exercised.

    Parameters
    ----------
    base_dir (pathlib.Path)
        Directory under which ``session_0``, ``session_1`` ... are created.
    n_sessions (int)
        Number of session directories to build.
    n_frames (int)
        Frame count per session feature table / session duration.
    camera_fps (float)
        Camera frame-rate stored in every track file.
    filter_history (float)
        Pre-event history length in seconds; also used to size the silent
        warm-up region in each USV summary.
    egocentric_features (list of str or None)
        Egocentric base features; defaults to ``['speed', 'neck_elevation']``.
    dyadic_features (list of str or None)
        Dyadic-pose base features; defaults to none (kept tiny).
    engagement_features (list of str or None)
        Engagement base features; defaults to none (kept tiny).
    mouse_name_stub (tuple of str)
        Two-element (male, female) stub; each session prefixes it with its id.
    n_bouts (int)
        Target-mouse bouts per session (passed to the USV-summary builder).
    usv_per_bout (int)
        Syllables per bout.
    category_column (str)
        Integer category column name in the USV summary.
    manifold_columns (tuple of str)
        Acoustic-manifold coordinate column names.
    csv_sep (str)
        Field separator for both CSVs.

    Returns
    -------
    session_roots (list of pathlib.Path)
        Absolute session ROOT directories, in creation order.
    """

    if egocentric_features is None:
        egocentric_features = ['speed', 'neck_elevation']

    base_dir.mkdir(parents=True, exist_ok=True)
    session_roots: list[Path] = []

    for s_idx in range(n_sessions):
        session_root = base_dir / f"session_{s_idx}"
        session_root.mkdir(parents=True, exist_ok=True)
        mouse_names = [f"s{s_idx}_{mouse_name_stub[0]}", f"s{s_idx}_{mouse_name_stub[1]}"]

        build_behavioral_features_csv(
            session_root=session_root,
            mouse_names=mouse_names,
            n_frames=n_frames,
            egocentric_features=egocentric_features,
            dyadic_features=dyadic_features,
            engagement_features=engagement_features,
            seed=s_idx,
            csv_sep=csv_sep,
        )
        build_track_h5(
            session_root=session_root,
            mouse_names=mouse_names,
            camera_fps=camera_fps,
        )
        # Predictor index 1 (female) is the partner; target index 0 (male) is
        # the focal/emitter by the project convention used in the smoke tests.
        build_usv_summary_csv(
            session_root=session_root,
            target_mouse=mouse_names[0],
            partner_mouse=mouse_names[1],
            camera_fps=camera_fps,
            n_frames=n_frames,
            filter_history=filter_history,
            n_bouts=n_bouts,
            usv_per_bout=usv_per_bout,
            category_column=category_column,
            manifold_columns=manifold_columns,
            seed=s_idx,
            csv_sep=csv_sep,
        )
        session_roots.append(session_root)

    return session_roots


def write_session_list_file(session_roots: list[Path], list_path: Path) -> Path:
    """
    Description
    -----------
    Writes a newline-delimited session-list text file (one absolute session
    ROOT path per line) — the exact format ``prepare_modeling_sessions`` reads
    from ``modeling_settings['io']['session_list_file']``.

    Parameters
    ----------
    session_roots (list of pathlib.Path)
        Session ROOT directories to enumerate.
    list_path (pathlib.Path)
        Destination text file path (parent dirs created if absent).

    Returns
    -------
    list_path (pathlib.Path)
        The path that was written (absolute).
    """

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text('\n'.join(str(p) for p in session_roots) + '\n')
    return list_path


def build_modeling_settings(
        session_list_file: Path,
        save_directory: Path,
        camera_sampling_rate: float = 150.0,
        filter_history: float = 1.0,
        model_engine: str = 'sklearn',
        egocentric_features: list[str] | None = None,
        dyadic_features: list[str] | None = None,
        engagement_features: list[str] | None = None,
        usv_predictor_type=None,
        split_strategy: str = 'mixed',
        split_num: int = 2,
        test_proportion: float = 0.3,
        random_seed: int = 0,
        csv_separator: str = ',',
) -> dict:
    """
    Description
    -----------
    Loads the canonical ``modeling_settings.json`` shipped with the package and
    returns a deep-copied, *trimmed and shrunk* variant suitable for a tiny
    end-to-end smoke run. The real JSON is the source of truth for every key
    (so no schema drift), and only the following are overridden:

      - ``io.session_list_file`` / ``io.save_directory`` -> caller's tmp paths.
      - ``io.camera_sampling_rate`` / ``io.csv_separator``.
      - ``model_params``: ``filter_history`` (short), ``model_engine`` (default
        ``sklearn`` -> fast univariate path), ``split_strategy``, ``split_num``
        (few folds), ``test_proportion``, ``random_seed``.
      - ``kinematic_features``: feature buckets shrunk to the tiny synthetic
        set; derivatives off.
      - ``vocal_features.usv_predictor_type`` -> ``None`` by default so no
        vocal predictor columns are generated (keeps the design matrix tiny).
      - ``hyperparameters.classical.logistic_regression``: a single tiny ``cs``
        value, ``cv=2``, low ``max_iter`` so ``LogisticRegressionCV`` is fast.
      - ``hyperparameters.classical.pygam``: few splines and few iterations so
        the GAM-based model-selection path runs quickly.
      - ``hyperparameters.basis_functions.*.plot_bool`` -> ``False`` so basis
        construction never spawns matplotlib figures.
      - ``diagnostics``: collinearity / timescale audits left off (the input
        extraction still invokes the audit wrapper, which no-ops cleanly).

    Parameters
    ----------
    session_list_file (pathlib.Path)
        Path to the session-list text file (written separately).
    save_directory (pathlib.Path)
        Directory where the pipeline writes all artifacts (must be under
        ``tmp_path``; never the package tree).
    camera_sampling_rate (float)
        Camera frame-rate; must match the track files for frame/second
        conversions to line up.
    filter_history (float)
        Pre-event history length in seconds (kept short for tiny windows).
    model_engine (str)
        ``'sklearn'`` (LogisticRegressionCV) or ``'pygam'`` (LogisticGAM) for
        the univariate onset path.
    egocentric_features (list of str or None)
        Egocentric bucket; defaults to ``['speed', 'neck_elevation']``.
    dyadic_features (list of str or None)
        Dyadic-pose bucket; defaults to ``[]``.
    engagement_features (list of str or None)
        Engagement bucket; defaults to ``[]``.
    usv_predictor_type (str or None)
        Vocal predictor mode; ``None`` disables vocal predictor columns.
    split_strategy (str)
        ``'mixed'`` or ``'session'`` (the two strategies model selection
        supports).
    split_num (int)
        Number of CV folds / splits.
    test_proportion (float)
        Test fraction per split.
    random_seed (int)
        Global / estimator seed.
    csv_separator (str)
        CSV field separator the loaders use.

    Returns
    -------
    settings (dict)
        A ready-to-use ``modeling_settings`` dictionary.
    """

    with open(_SETTINGS_JSON, 'r') as fh:
        settings = json.load(fh)
    settings = copy.deepcopy(settings)

    if egocentric_features is None:
        egocentric_features = ['speed', 'neck_elevation']
    if dyadic_features is None:
        dyadic_features = []
    if engagement_features is None:
        engagement_features = []

    settings['io']['session_list_file'] = str(session_list_file)
    settings['io']['save_directory'] = str(save_directory)
    settings['io']['camera_sampling_rate'] = camera_sampling_rate
    settings['io']['csv_separator'] = csv_separator

    mp = settings['model_params']
    mp['filter_history'] = filter_history
    mp['model_engine'] = model_engine
    mp['split_strategy'] = split_strategy
    mp['split_num'] = split_num
    mp['test_proportion'] = test_proportion
    mp['random_seed'] = random_seed
    mp['model_target_vocal_type'] = 'bout'
    mp['model_predictor_mouse_index'] = 1
    mp['usv_per_bout_floor'] = 2

    kin = settings['kinematic_features']
    kin['egocentric'] = list(egocentric_features)
    kin['dyadic_pose'] = list(dyadic_features)
    kin['dyadic_engagement'] = list(engagement_features)
    kin['dyadic_pose_symmetric'] = False
    kin['include_1st_derivatives'] = False
    kin['include_2nd_derivatives'] = False
    kin['smooth_abs_features'] = {}

    settings['vocal_features']['usv_predictor_type'] = usv_predictor_type

    settings['diagnostics']['collinearity_audit'] = False
    settings['diagnostics']['timescale_audit'] = False

    lr = settings['hyperparameters']['classical']['logistic_regression']
    lr['cs'] = [0.1]
    lr['cv'] = 2
    lr['max_iter'] = 200

    gam = settings['hyperparameters']['classical']['pygam']
    gam['n_splines_time'] = 4
    gam['n_splines_value'] = 4
    gam['max_iterations'] = 20
    gam['lam_penalty'] = 0.6

    for basis_cfg in settings['hyperparameters']['basis_functions'].values():
        if 'plot_bool' in basis_cfg:
            basis_cfg['plot_bool'] = False

    return settings


def build_modeling_input_pickle(
        save_path: Path,
        feature_names: list[str],
        session_ids: list[str],
        history_frames: int,
        n_usv: int = 40,
        n_no_usv: int = 80,
        input_metadata: dict | None = None,
        seed: int = 0,
) -> Path:
    """
    Description
    -----------
    Serializes a controlled "modeling input pickle" directly, bypassing the
    on-disk extraction. The artifact matches the schema the univariate
    dispatcher and the model-selection orchestrators load:

        {
          '<generic_feature>': {
              '<session_id>': {
                  'usv_feature_arr':    np.ndarray (n_usv,    history_frames),
                  'no_usv_feature_arr': np.ndarray (n_no_usv, history_frames),
              }, ...
          }, ...,
          '_input_metadata': {...}   # reserved block (optional)
        }

    Per-session event counts are identical across features (the alignment
    invariant the real extractor enforces). Signal arrays carry feature- and
    class-dependent means so a logistic model can find above-chance structure
    on the ``usv`` vs ``no_usv`` contrast.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Generic feature keys (e.g. ``['self.speed', 'other.speed']``).
    session_ids (list of str)
        Session identifiers to populate under every feature.
    history_frames (int)
        Number of temporal lags (columns) per event window.
    n_usv (int)
        Positive (USV) events per session.
    n_no_usv (int)
        Negative (No-USV) events per session.
    input_metadata (dict or None)
        Optional ``_input_metadata`` block; omitted when ``None``.
    seed (int)
        Base seed for the per-cell RNG.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)
    artifact: dict = {}

    for f_idx, feature in enumerate(feature_names):
        artifact[feature] = {}
        for sess in session_ids:
            usv_arr = (
                0.5 + 0.1 * f_idx
                + rng.standard_normal((n_usv, history_frames))
            ).astype(float)
            no_usv_arr = (
                -0.5 - 0.1 * f_idx
                + rng.standard_normal((n_no_usv, history_frames))
            ).astype(float)
            artifact[feature][sess] = {
                'usv_feature_arr': usv_arr,
                'no_usv_feature_arr': no_usv_arr,
            }

    if input_metadata is not None:
        artifact['_input_metadata'] = input_metadata

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path
