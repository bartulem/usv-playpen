"""
@author: bartulem
End-to-end smoke tests for the multinomial USV-category modeling pipeline and
its JAX-accelerated model-selection path, driven entirely on tiny synthetic
data built by the shared ``_synth`` builders plus a handful of multinomial-
specific local builders defined in this module.

These tests deliberately walk the *production* code paths rather than testing
isolated helpers:

* ``TestMultinomialInputExtraction`` runs the real
  ``MultinomialModelingPipeline.extract_and_save_multinomial_input_data``
  against a synthetic session tree whose USV-summary CSVs carry at least three
  distinct (non-noise) vocal categories. This lights up
  ``load_behavioral_feature_data``, ``find_usv_categories`` in multinomial mode
  (``target_category=None`` -> ``events_by_category``), the kinematic-column
  selector, the vocal-column builder, cross-session harmonization / z-scoring,
  ``run_predictor_audits``, and the ``modeling_metadata`` block. The single-pass
  integer-label target vector ``y`` (>= 3 classes) is the multinomial contract
  the binomial onset pipeline never exercises.

* ``TestMultinomialSplitters`` covers the pure-NumPy splitter and balancing
  helpers (``get_stratified_group_splits_stable`` for both 'mixed' and
  'session' strategies plus their hard coverage guards,
  ``_balance_multinomial_train_indices``, and ``_log_spaced_grid_multinomial``)
  without touching JAX, so the cheap branches are green regardless of solver
  speed.

* ``TestMultinomialModelSelection`` runs the real
  ``multinomial_vocal_category_model_selection`` forward-selection orchestrator
  on a strong-signal synthetic three-class input pickle plus a matching
  hand-built univariate ranking, with the JAX knobs shrunk to the bone (one
  binned time column, ``cv=2``, tiny ``max_iter`` / inner ``max_iter``, tuning
  off). It asserts the Step-0 model-free-prior baseline pickle and any
  forward-selection step pickles carry the documented structure.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so the
top-level modeling imports below are wrapped in a ``warnings.catch_warnings``
block that ignores ``DeprecationWarning`` during import. At run time the JAX /
optax / sklearn stack and the tiny synthetic folds emit assorted benign
``RuntimeWarning`` / ``UserWarning`` / ``DeprecationWarning`` instances (e.g.
mean-of-empty-slice, ill-defined macro metrics on degenerate folds, JAX export
deprecations); the JAX-driven tests demote those with narrow per-test
``@pytest.mark.filterwarnings`` markers. ``matplotlib`` is forced onto the
headless ``Agg`` backend defensively in case any imported module pulls
``pyplot`` at import time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import polars as pls
import pytest

matplotlib.use('Agg')

from tests.modeling._synth import (
    build_behavioral_features_csv,
    build_modeling_settings,
    build_track_h5,
    write_session_list_file,
)

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level imports so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import usv_playpen.modeling.main_model_selection_dispatcher as ms_dispatcher
    from usv_playpen.modeling.model_selection import (
        multinomial_vocal_category_model_selection,
    )
    from usv_playpen.modeling.jax_multinomial_logistic_regression import (
        SmoothMultinomialLogisticRegression,
    )
    from usv_playpen.modeling.modeling_vocal_categories_multinomial import (
        MultinomialModelingPipeline,
        MultinomialModelRunner,
        _balance_multinomial_train_indices,
        _log_spaced_grid_multinomial,
        _tune_multinomial_regularization,
        get_stratified_group_splits_stable,
    )


# Tiny-data geometry shared across tests. Sized so the multinomial extraction
# yields a meaningful number of bout events spread across at least three vocal
# categories while keeping every array small. ``HISTORY_FRAMES`` is the derived
# ``floor(CAMERA_FPS * FILTER_HISTORY)`` column count of every per-event window.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
N_FRAMES = 7200       # 120 s sessions -> plenty of room for many spread events
N_SESSIONS = 4
N_CATEGORIES = 3      # non-noise vocal categories: labels 1, 2, 3
NOISE_CATEGORY = 0    # filtered out by ``usv_noise_categories`` (default [0])

# Multinomial-specific settings overrides. The shipped JSON's multinomial knobs
# (max_iter=20000, tuning on, inner_max_iter=2500, smoothness order 2) are far
# too heavy for a smoke test; these shrink them to the bone while keeping the
# estimator numerically well-posed on the tiny folds.
_MULTINOMIAL_HP_OVERRIDES = {
    'balance_predictions_bool': False,
    'balance_train_bool': False,
    'bin_resizing_factor': max(HISTORY_FRAMES, 1),   # collapse history -> 1 bin
    'lambda_smooth_fixed': 1.0,
    'l2_reg_fixed': 0.01,
    'smoothness_derivative_order': 1,
    'learning_rate': 0.05,
    'max_iter': 60,
    'tol': 0.01,
    'random_state': 0,
    'verbose': False,
    'use_lax_loop': False,
    'focal_loss_gamma': 0.0,
    'tune_regularization_bool': False,
    'tune_regularization_params': {
        'lambda_smooth_decades_each_side': 0,
        'l2_reg_decades_each_side': 0,
        'inner_cv_folds': 2,
        'inner_cv_scoring_metric': 'auc',
        'inner_cv_use_one_se_rule': False,
        'inner_max_iter': 20,
    },
}


def build_multinomial_usv_summary_csv(
        session_root: Path,
        target_mouse: str,
        partner_mouse: str,
        camera_fps: float,
        n_frames: int,
        filter_history: float,
        n_categories: int = N_CATEGORIES,
        events_per_category: int = 10,
        category_column: str = 'vae_supercategory',
        manifold_columns: tuple[str, str] = ('vae_umap1', 'vae_umap2'),
        seed: int = 0,
        csv_sep: str = ',',
) -> Path:
    """
    Description
    -----------
    Writes a synthetic ``*_usv_summary.csv`` under ``<session_root>/audio`` that
    carries at least ``n_categories`` *distinct non-noise* vocal categories for
    the target mouse, so the multinomial extractor's per-session
    ``events_by_category`` dict — and the cohort-pooled label set the splitter's
    class-coverage gate validates against — spans every theoretical class slot.

    Construction guarantees:
      - ``events_per_category`` isolated single-syllable events per non-noise
        category (labels ``1 .. n_categories``), each event well past the
        ``filter_history`` warm-up region so its pre-event window never clips
        before frame 0, and all events spread across the usable session so the
        per-class onset frame indices are distinct.
      - a handful of noise rows (category ``0``) that the
        ``usv_noise_categories`` filter strips before label extraction, so the
        noise-filter branch in ``find_usv_categories`` is genuinely exercised.
      - a couple of sparse partner-mouse rows late in the session so partner
        vocal predictors have content without polluting the target's events.

    Unlike ``_synth.build_usv_summary_csv`` (which packs syllables into bouts
    and stamps every row with the single category ``1``), this builder spreads
    events and varies the category label, which is exactly what the multinomial
    pipeline needs and the shared builder does not provide.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session ROOT directory; the ``audio`` subtree is created if absent.
    target_mouse (str)
        Emitter name for all biological target USV rows (focal/target mouse).
    partner_mouse (str)
        Emitter name for the sparse partner rows.
    camera_fps (float)
        Camera frame-rate (used only to keep timestamps within the session).
    n_frames (int)
        Total session length in frames (defines the session duration).
    filter_history (float)
        Pre-event history length in seconds; the session's opening window is
        kept silent so early events are never clipped.
    n_categories (int)
        Number of distinct non-noise categories to synthesize (labels start
        at 1; ``0`` is the noise label).
    events_per_category (int)
        Isolated target events written per non-noise category.
    category_column (str)
        Integer category column name (also the noise column by default).
    manifold_columns (tuple of str)
        Two synthetic acoustic-manifold coordinate column names.
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

    # Silent warm-up at the start so early events are never clipped; spread the
    # events across the remaining session with generous spacing.
    warmup = filter_history * 2.0 + 1.0
    usable = session_duration_sec - warmup - filter_history * 2.0
    if usable <= 0:
        raise ValueError("Session too short to host the requested events.")

    total_target_events = n_categories * events_per_category
    spacing = usable / total_target_events
    syllable_dur = 0.01

    emitters: list[str] = []
    starts: list[float] = []
    stops: list[float] = []
    categories: list[int] = []

    # Interleave categories across the timeline so per-class onset frames are
    # distinct and no single class clusters into one corner of the session.
    event_counter = 0
    for ev in range(events_per_category):
        for cat in range(1, n_categories + 1):
            start = warmup + event_counter * spacing
            emitters.append(target_mouse)
            starts.append(round(start, 6))
            stops.append(round(start + syllable_dur, 6))
            categories.append(cat)
            event_counter += 1

    # A few noise rows (category 0) for the target, late but inside the session,
    # so the ``usv_noise_categories`` filter has something to strip.
    noise_anchor = warmup + (total_target_events + 1) * spacing
    for s in range(3):
        start = noise_anchor + s * 0.25
        if start + syllable_dur >= session_duration_sec:
            break
        emitters.append(target_mouse)
        starts.append(round(start, 6))
        stops.append(round(start + syllable_dur, 6))
        categories.append(NOISE_CATEGORY)

    # Sparse partner USVs (non-noise) so partner vocal predictors have content.
    partner_anchor = warmup + (total_target_events - 0.5) * spacing
    for s in range(2):
        start = partner_anchor + s * 0.4
        emitters.append(partner_mouse)
        starts.append(round(start, 6))
        stops.append(round(start + syllable_dur, 6))
        categories.append(1)

    n_rows = len(starts)
    rows = {
        'emitter': emitters,
        'start': starts,
        'stop': stops,
        category_column: categories,
        'vae_category': categories,
        'mask_number': [2] * n_rows,
        manifold_columns[0]: (rng.standard_normal(n_rows)).round(6).tolist(),
        manifold_columns[1]: (rng.standard_normal(n_rows)).round(6).tolist(),
    }

    audio_dir = session_root / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audio_dir / f"{session_root.name}_usv_summary.csv"
    pls.DataFrame(rows).write_csv(file=csv_path, separator=csv_sep)
    return csv_path


def build_multinomial_session_tree(
        base_dir: Path,
        n_sessions: int = N_SESSIONS,
        n_frames: int = N_FRAMES,
        camera_fps: float = CAMERA_FPS,
        filter_history: float = FILTER_HISTORY,
        n_categories: int = N_CATEGORIES,
        events_per_category: int = 10,
        egocentric_features: list[str] | None = None,
        mouse_name_stub: tuple[str, str] = ('m_male', 'm_female'),
        csv_sep: str = ',',
) -> list[Path]:
    """
    Description
    -----------
    Builds a complete multi-session synthetic recording tree for the multinomial
    pipeline and returns the session ROOT directories. Each session reuses the
    shared ``_synth`` behavioral-feature and HDF5 track builders, but its
    USV-summary CSV is produced by ``build_multinomial_usv_summary_csv`` so the
    target mouse vocalizes across ``n_categories`` distinct categories.

    Mouse track names are uniquified per session (e.g. ``s0_m_male``) so
    cross-session role resolution and the dyad-rename logic are exercised. The
    target (emitter) is mouse index 0 (male) by the project convention the
    extractor uses (``target_idx = abs(predictor_idx - 1)`` with predictor
    index 1).

    Parameters
    ----------
    base_dir (pathlib.Path)
        Directory under which ``session_0`` ... are created.
    n_sessions (int)
        Number of session directories to build.
    n_frames (int)
        Frame count per session feature table / session duration.
    camera_fps (float)
        Camera frame-rate stored in every track file.
    filter_history (float)
        Pre-event history length in seconds.
    n_categories (int)
        Distinct non-noise vocal categories per session.
    events_per_category (int)
        Isolated target events per category per session.
    egocentric_features (list of str or None)
        Egocentric base features; defaults to ``['speed', 'neck_elevation']``.
    mouse_name_stub (tuple of str)
        Two-element (male, female) stub; each session prefixes it with its id.
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
            dyadic_features=None,
            engagement_features=None,
            seed=s_idx,
            csv_sep=csv_sep,
        )
        build_track_h5(
            session_root=session_root,
            mouse_names=mouse_names,
            camera_fps=camera_fps,
        )
        build_multinomial_usv_summary_csv(
            session_root=session_root,
            target_mouse=mouse_names[0],
            partner_mouse=mouse_names[1],
            camera_fps=camera_fps,
            n_frames=n_frames,
            filter_history=filter_history,
            n_categories=n_categories,
            events_per_category=events_per_category,
            seed=s_idx,
            csv_sep=csv_sep,
        )
        session_roots.append(session_root)

    return session_roots


def _apply_multinomial_overrides(settings: dict) -> dict:
    """
    Description
    -----------
    Mutates ``settings`` in place so the multinomial JAX estimator runs in a
    smoke-test-tiny regime: the
    ``hyperparameters.jax_linear.multinomial_logistic`` block is replaced with
    ``_MULTINOMIAL_HP_OVERRIDES`` (one binned column, few iters, tuning off),
    the vocal-predictor mode is disabled to keep the design matrix to the four
    kinematic features, and the per-session split rejection-sampler ceilings are
    left at their JSON defaults.

    Parameters
    ----------
    settings (dict)
        A ``modeling_settings`` dict (typically from ``build_modeling_settings``).

    Returns
    -------
    settings (dict)
        The same dict, mutated, for call-site chaining.
    """

    settings['hyperparameters']['jax_linear']['multinomial_logistic'] = dict(
        _MULTINOMIAL_HP_OVERRIDES
    )
    settings['vocal_features']['usv_predictor_type'] = None
    return settings


def build_multinomial_input_pickle(
        save_path: Path,
        feature_names: list[str],
        session_ids: list[str],
        history_frames: int,
        n_categories: int = N_CATEGORIES,
        n_per_class_per_session: int = 18,
        seed: int = 0,
) -> Path:
    """
    Description
    -----------
    Serializes a controlled multinomial "modeling input pickle" directly,
    bypassing on-disk extraction. The artifact matches the schema
    ``MultinomialModelRunner.load_univariate_data_blocks`` and
    ``multinomial_vocal_category_model_selection`` consume:

        {
          '<feature>': {
              '<session_id>': {
                  'X': np.ndarray (n_samples, history_frames),
                  'y': np.ndarray (n_samples,)  # integer class labels 0..K-1
              }, ...
          }, ...,
          '_input_metadata': {
              'analysis_specific': {
                  'usv_category_number': <K>,
                  'usv_category_column_name': 'vae_supercategory',
              }, ...
          }
        }

    Every session carries ``n_per_class_per_session`` rows of *each* of the
    ``n_categories`` classes (so the cohort-wide and per-fold class-coverage
    guards in ``get_stratified_group_splits_stable`` are always satisfiable),
    and the per-feature window means are shifted per class so a multinomial
    logistic fit can find above-chance structure.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Generic feature keys (e.g. ``['self.speed', 'other.speed']``).
    session_ids (list of str)
        Session identifiers populated under every feature.
    history_frames (int)
        Number of temporal lags (columns) per event window.
    n_categories (int)
        Number of integer classes (labels ``0 .. n_categories - 1``).
    n_per_class_per_session (int)
        Rows of each class written per session.
    seed (int)
        Base seed for the per-cell RNG.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)

    # One label vector per session, shared verbatim across every feature so the
    # intra-session alignment invariant (identical event counts / labels across
    # features) holds exactly. Labels are 0..K-1 contiguous.
    labels_per_session = {
        sess: np.repeat(np.arange(n_categories, dtype=np.int64), n_per_class_per_session)
        for sess in session_ids
    }

    artifact: dict = {}
    for f_idx, feature in enumerate(feature_names):
        artifact[feature] = {}
        for sess in session_ids:
            y_sess = labels_per_session[sess]
            n_rows = y_sess.size
            # Per-class mean shift (scaled by feature index) so distinct
            # features carry distinct, class-separating structure.
            class_shift = (y_sess.astype(float) - (n_categories - 1) / 2.0)
            base = (0.4 + 0.1 * f_idx) * class_shift
            X_sess = (
                base[:, None]
                + rng.standard_normal((n_rows, history_frames))
            ).astype(np.float32)
            artifact[feature][sess] = {'X': X_sess, 'y': y_sess.copy()}

    artifact['_input_metadata'] = {
        'analysis_tag': 'multinomial_vae_supercategory',
        'analysis_specific': {
            'usv_category_number': int(n_categories),
            'usv_category_column_name': 'vae_supercategory',
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def build_univariate_ranking_pickle(
        save_path: Path,
        feature_names: list[str],
        n_splits: int,
        seed: int = 0,
) -> Path:
    """
    Description
    -----------
    Serializes a minimal consolidated univariate-results pickle in the exact
    shape ``multinomial_vocal_category_model_selection`` screens: a
    ``{feature: {'actual': {...}, 'null': {...}}}`` dict where each strategy
    exposes ``['folds']['metrics']['auc']`` as a length-``n_splits`` list. The
    ``actual`` AUCs are pinned high and the ``null`` AUCs low so every feature
    survives the per-feature significance screen (mean actual AUC above the
    null percentile threshold) and enters the ranked candidate pool.

    This bypasses a full univariate JAX run (which the smoke test does not need
    to re-derive) while still feeding the selection orchestrator a real,
    schema-correct ranking input.

    Parameters
    ----------
    save_path (pathlib.Path)
        Destination ``.pkl`` path (parent dirs created if absent).
    feature_names (list of str)
        Feature keys to rank; AUCs descend slightly per feature so the ranking
        order is deterministic.
    n_splits (int)
        Number of folds each metric list carries (matches ``split_num``).
    seed (int)
        Seed for the tiny per-fold AUC jitter.

    Returns
    -------
    save_path (pathlib.Path)
        The path the pickle was written to (absolute).
    """

    rng = np.random.default_rng(seed)
    ranking: dict = {}
    for f_idx, feature in enumerate(feature_names):
        actual_center = 0.85 - 0.02 * f_idx
        actual_auc = (actual_center + 0.005 * rng.standard_normal(n_splits)).tolist()
        null_auc = (0.50 + 0.01 * rng.standard_normal(n_splits)).tolist()
        ranking[feature] = {
            'actual': {'folds': {'metrics': {'auc': actual_auc}}},
            'null': {'folds': {'metrics': {'auc': null_auc}}},
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(ranking, fh)
    return save_path


def _build_extraction_settings(tmp_path: Path, **overrides):
    """
    Description
    -----------
    Builds the synthetic multinomial session tree, the session-list file, and a
    trimmed ``modeling_settings`` dict for an extraction smoke run — all rooted
    under ``tmp_path`` so nothing is ever written into the package tree. The
    multinomial JAX-block / vocal-predictor overrides are applied on top of the
    shared shrunk settings.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test scratch directory.
    overrides (dict)
        Extra keyword arguments forwarded to ``build_modeling_settings``.

    Returns
    -------
    settings (dict)
        The ready-to-use ``modeling_settings`` dictionary.
    save_dir (pathlib.Path)
        The pipeline output directory (``tmp_path / 'out'``).
    """

    session_roots = build_multinomial_session_tree(base_dir=tmp_path / 'sessions')
    list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
    save_dir = tmp_path / 'out'
    save_dir.mkdir(parents=True, exist_ok=True)

    settings = build_modeling_settings(
        session_list_file=list_file,
        save_directory=save_dir,
        camera_sampling_rate=CAMERA_FPS,
        filter_history=FILTER_HISTORY,
        **overrides,
    )
    _apply_multinomial_overrides(settings)
    return settings, save_dir


class TestMultinomialInputExtraction:
    """End-to-end extraction of the multinomial input pickle from a synthetic tree."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extraction_produces_multiclass_input_pickle(self, tmp_path):
        """
        The real ``extract_and_save_multinomial_input_data`` writes a single
        ``modeling_multinomial_*.pkl`` whose structure matches the documented
        contract: a nested ``{feature: {session: {X, y}}}`` dict carrying a
        reserved ``_input_metadata`` block. Each per-event window is
        ``HISTORY_FRAMES`` wide, the integer-label target ``y`` spans at least
        ``N_CATEGORIES`` distinct (non-noise) classes, the per-session
        positive-count alignment holds across features, and the metadata records
        the auto-derived ``usv_category_number`` >= ``N_CATEGORIES``.
        """

        settings, save_dir = _build_extraction_settings(tmp_path)
        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_multinomial_input_data()

        pkls = list(save_dir.glob('modeling_multinomial_*.pkl'))
        assert len(pkls) == 1, f"expected exactly one input pickle, got {pkls}"

        with pkls[0].open('rb') as fh:
            artifact = pickle.load(fh)

        assert '_input_metadata' in artifact
        feature_keys = sorted(k for k in artifact if not k.startswith('_'))
        # Egocentric ['speed', 'neck_elevation'] expand to self.* and other.*.
        assert feature_keys == [
            'other.neck_elevation', 'other.speed',
            'self.neck_elevation', 'self.speed',
        ]

        anchor = feature_keys[0]
        sessions = sorted(artifact[anchor].keys())
        assert len(sessions) >= 1

        pooled_labels = []
        for sess in sessions:
            X = artifact[anchor][sess]['X']
            y = artifact[anchor][sess]['y']
            assert X.shape[1] == HISTORY_FRAMES
            assert X.shape[0] == y.shape[0]
            assert np.isfinite(X).all()
            pooled_labels.append(y)

            # Intra-session alignment: every feature shares this session's
            # sample count and label vector.
            for feat in feature_keys[1:]:
                assert artifact[feat][sess]['X'].shape[0] == X.shape[0]
                np.testing.assert_array_equal(artifact[feat][sess]['y'], y)

        all_y = np.concatenate(pooled_labels)
        observed_classes = np.unique(all_y)
        assert observed_classes.size >= N_CATEGORIES
        # The noise category (0) must have been filtered out before extraction.
        assert NOISE_CATEGORY not in observed_classes.tolist()

        md = artifact['_input_metadata']
        assert md['analysis_type'] == 'multinomial'
        assert md['analysis_tag'] == 'multinomial_vae_supercategory'
        assert sorted(md['feature_zoo_kept']) == feature_keys
        spec = md['analysis_specific']
        assert spec['usv_category_number'] >= N_CATEGORIES
        assert spec['usv_category_number'] == observed_classes.size
        assert spec['usv_category_column_name'] == 'vae_supercategory'


class TestMultinomialSplitters:
    """Pure-NumPy splitter / balancing / grid helpers (no JAX involved)."""

    def test_log_spaced_grid_shapes_and_degenerate_case(self):
        """
        ``_log_spaced_grid_multinomial`` returns a sorted, log-spaced grid of
        length ``2 * decades_each_side + 1`` centred on the supplied value, and
        collapses to the single fixed value when ``decades_each_side == 0``.
        Invalid (negative half-width / non-positive centre) inputs raise.
        """

        grid = _log_spaced_grid_multinomial(center=1.0, decades_each_side=3)
        assert grid.shape == (7,)
        np.testing.assert_allclose(grid, [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

        degenerate = _log_spaced_grid_multinomial(center=0.5, decades_each_side=0)
        np.testing.assert_allclose(degenerate, [0.5])

        with pytest.raises(ValueError, match='decades_each_side must be >= 0'):
            _log_spaced_grid_multinomial(center=1.0, decades_each_side=-1)
        with pytest.raises(ValueError, match='center must be positive'):
            _log_spaced_grid_multinomial(center=0.0, decades_each_side=1)

    def test_balance_train_indices_equalizes_classes(self):
        """
        ``_balance_multinomial_train_indices`` down-samples to ``min(class
        count)`` per class, returning a shuffled index subset drawn from the
        supplied training indices with every class equally represented.
        """

        y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=int)
        train_idx = np.arange(y.size)
        rng = np.random.default_rng(0)
        balanced = _balance_multinomial_train_indices(train_idx, y, rng)

        balanced_classes, balanced_counts = np.unique(y[balanced], return_counts=True)
        np.testing.assert_array_equal(balanced_classes, [0, 1, 2])
        # Minimum class count is 2 (class 1), so every class contributes 2.
        np.testing.assert_array_equal(balanced_counts, [2, 2, 2])
        assert set(balanced.tolist()).issubset(set(train_idx.tolist()))

    def test_mixed_strategy_splits_cover_all_classes(self):
        """
        With ``split_strategy='mixed'`` the splitter delegates to
        ``StratifiedShuffleSplit``, returns ``n_splits`` train/test index pairs
        whose union is disjoint, reports all-zero fold tolerances, and (by the
        symmetric coverage guard) guarantees every class appears in both halves
        of every fold.
        """

        rng = np.random.default_rng(1)
        y = np.repeat([0, 1, 2], 40)
        groups = rng.integers(0, 5, size=y.size)

        cv_folds, tolerances = get_stratified_group_splits_stable(
            groups=groups, y=y, n_categories=3, split_strategy='mixed',
            test_prop=0.3, n_splits=3, random_seed=0,
        )
        assert len(cv_folds) == 3
        assert tolerances == [0.0, 0.0, 0.0]
        for tr_idx, te_idx in cv_folds:
            assert set(tr_idx.tolist()).isdisjoint(te_idx.tolist())
            assert np.unique(y[tr_idx]).size == 3
            assert np.unique(y[te_idx]).size == 3

    def test_cohort_coverage_guard_raises_on_missing_class(self):
        """
        The cohort-wide coverage invariant raises ``ValueError`` when the pooled
        label set carries fewer distinct classes than ``n_categories`` — the
        guard that protects the downstream classifier from an empty class slot.
        """

        y = np.repeat([0, 1], 20)   # only two classes present
        groups = np.zeros(y.size, dtype=int)
        with pytest.raises(ValueError, match="distinct classes"):
            get_stratified_group_splits_stable(
                groups=groups, y=y, n_categories=3, split_strategy='mixed',
                test_prop=0.3, n_splits=2, random_seed=0,
            )

    def test_session_strategy_keeps_sessions_disjoint(self):
        """
        With ``split_strategy='session'`` whole sessions are the atomic sampling
        unit: the rejection sampler accepts folds whose test set holds entire
        sessions never seen in train, every class is covered on both sides, and
        a per-fold tolerance is recorded. Asserts session-level disjointness.
        """

        rng = np.random.default_rng(2)
        # Six sessions, each carrying all three classes, so the session-level
        # rejection sampler can find folds that cover every class on both sides.
        groups_list, y_list = [], []
        for sess in range(6):
            groups_list.append(np.full(30, sess))
            y_list.append(np.repeat([0, 1, 2], 10))
        groups = np.concatenate(groups_list)
        y = np.concatenate(y_list)
        # Shuffle within to avoid any accidental ordering coupling.
        perm = rng.permutation(y.size)
        groups, y = groups[perm], y[perm]

        cv_folds, tolerances = get_stratified_group_splits_stable(
            groups=groups, y=y, n_categories=3, split_strategy='session',
            test_prop=0.5, n_splits=2, random_seed=0, tolerance=0.5,
        )
        assert len(cv_folds) == 2
        assert len(tolerances) == 2
        for tr_idx, te_idx in cv_folds:
            tr_sessions = set(groups[tr_idx].tolist())
            te_sessions = set(groups[te_idx].tolist())
            assert tr_sessions.isdisjoint(te_sessions)
            assert np.unique(y[tr_idx]).size == 3
            assert np.unique(y[te_idx]).size == 3

    def test_invalid_strategy_raises(self):
        """An unsupported ``split_strategy`` value raises ``ValueError``."""

        y = np.repeat([0, 1, 2], 10)
        groups = np.zeros(y.size, dtype=int)
        with pytest.raises(ValueError, match="split_strategy"):
            get_stratified_group_splits_stable(
                groups=groups, y=y, n_categories=3, split_strategy='bogus',
                test_prop=0.3, n_splits=1, random_seed=0,
            )


class TestMultinomialUnivariateRunner:
    """The JAX-driven per-feature univariate runner and its inner-CV tuner."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_run_univariate_training_tri_strategy(self, tmp_path):
        """
        ``MultinomialModelRunner.run_univariate_training`` drives the full
        tri-strategy (actual / null / null_model_free) JAX univariate pass on a
        single feature of a strong-signal three-class input pickle with tiny
        knobs. Asserts every strategy is present, exposes the documented per-fold
        metric key set with ``split_num`` entries each, persists the
        ``canonical_classes`` ordering, and that the ``actual`` strategy yields at
        least one finite (fitted) AUC fold.
        """

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
            test_proportion=0.4,
        )

        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=18,
        ))

        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        runner = MultinomialModelRunner(pipeline_instance=pipeline)
        feat_name, results = runner.run_univariate_training(
            pkl_path=input_pkl, feat_name='self.speed'
        )

        assert feat_name == 'self.speed'
        n_splits = settings['model_params']['split_num']
        metric_keys = {'auc', 'score', 'recall', 'f1', 'll', 'brier', 'ece', 'mcc'}
        for strategy in ('actual', 'null', 'null_model_free'):
            assert strategy in results
            folds = results[strategy]['folds']
            assert metric_keys.issubset(folds['metrics'].keys())
            for metric in metric_keys:
                assert len(folds['metrics'][metric]) == n_splits
            assert results[strategy]['canonical_classes'].size == N_CATEGORIES

        actual_auc = np.asarray(results['actual']['folds']['metrics']['auc'], dtype=float)
        assert np.isfinite(actual_auc).any()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_tune_multinomial_regularization_picks_pair(self):
        """
        ``_tune_multinomial_regularization`` runs an inner stratified CV over the
        ``(lambda_smooth, l2_reg)`` Cartesian grid using the injected estimator
        class and returns a concrete pair drawn from the grids plus a populated
        audit dict (grid scores / SEs / argmax pair). Driven on a tiny strongly
        class-separable design with single-iteration fits to stay fast.
        """

        rng = np.random.default_rng(0)
        n_per_class, n_time = 24, 1
        y = np.repeat([0, 1, 2], n_per_class).astype(np.int32)
        shift = (y.astype(float) - 1.0)[:, None]
        X = (1.5 * shift + 0.2 * rng.standard_normal((y.size, n_time))).astype(np.float32)

        lambda_grid = np.array([0.1, 1.0])
        l2_grid = np.array([0.01, 0.1])
        best_lam_sm, best_l2, audit = _tune_multinomial_regularization(
            X, y,
            lambda_smooth_grid=lambda_grid,
            l2_reg_grid=l2_grid,
            inner_cv_folds=2,
            inner_cv_scoring_metric='auc',
            inner_cv_use_one_se_rule=False,
            n_features=1,
            n_time_bins=n_time,
            smoothness_derivative_order=1,
            focal_gamma=0.0,
            uniform_class_weights=False,
            learning_rate=0.05,
            inner_max_iter=20,
            tol=0.01,
            random_state=0,
            verbose=False,
            use_lax_loop=False,
            regressor_cls=SmoothMultinomialLogisticRegression,
        )

        assert best_lam_sm in lambda_grid.tolist()
        assert best_l2 in l2_grid.tolist()
        assert set(audit.keys()) == {
            'grid_scores', 'grid_ses', 'argmax_pair',
            'one_se_applied', 'one_se_threshold',
        }
        assert len(audit['grid_scores']) == lambda_grid.size * l2_grid.size


class TestMultinomialModelSelection:
    """The real JAX-driven forward-stepwise multinomial selection orchestrator."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_multinomial_selection_writes_step_pickles(self, tmp_path):
        """
        Running ``multinomial_vocal_category_model_selection`` on a strong-signal
        three-class synthetic input pickle (with a matching hand-built univariate
        ranking and tiny JAX knobs) establishes the Step-0 model-free-prior
        baseline and runs the anchored forward search. Every step pickle carries
        the documented ``step_idx`` / ``current_features`` / ``baseline_score`` /
        ``candidates_summary`` structure, the Step-0 baseline exposes the
        ``null_model_free`` candidate with a per-fold AUC list of length
        ``split_num``, and the accepted feature count never shrinks across steps.
        """

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
            test_proportion=0.4,
        )

        feature_names = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=18,
        ))
        univ_pkl = str(build_univariate_ranking_pickle(
            save_path=tmp_path / 'univariate_combined.pkl',
            feature_names=feature_names,
            n_splits=settings['model_params']['split_num'],
        ))

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        multinomial_vocal_category_model_selection(
            univariate_results_path=univ_pkl,
            input_data_path=input_pkl,
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.05,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_multinomial_*_step_*.pkl'))
        assert len(step_pkls) >= 1, "expected at least the Step-0 baseline pickle"

        n_splits = settings['model_params']['split_num']
        accepted_counts = []
        saw_baseline = False
        for p in step_pkls:
            with p.open('rb') as fh:
                step = pickle.load(fh)
            assert 'step_idx' in step
            assert 'current_features' in step
            assert 'baseline_score' in step
            assert 'candidates_summary' in step
            accepted_counts.append(len(step['current_features']))

            if step['step_idx'] == 0:
                saw_baseline = True
                assert step['selected_feature'] == 'null_model_free'
                baseline = step['candidates_summary']['null_model_free']
                assert len(baseline['folds']['metrics']['auc']) == n_splits

        assert saw_baseline, "Step-0 model-free baseline pickle was not written"
        # The forward search never drops an already-accepted feature.
        assert accepted_counts == sorted(accepted_counts)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_multinomial_selection_dispatcher_routes(self, tmp_path, monkeypatch):
        """
        ``dispatch_model_selection`` validates the three required paths and routes
        the 'multinomial' task into ``multinomial_vocal_category_model_selection``.
        The dispatcher auto-resolves the package settings JSON, so the real
        selection call is wrapped to inject the synthetic settings path instead
        of editing ``src/``. Reaching the end without an uncaught exception means
        the dispatcher's validation + routing executed; step pickles are written
        once the baseline / anchor blocks run.
        """

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
            test_proportion=0.4,
        )

        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=16,
        ))
        univ_pkl = str(build_univariate_ranking_pickle(
            save_path=tmp_path / 'univariate_combined.pkl',
            feature_names=feature_names,
            n_splits=settings['model_params']['split_num'],
        ))

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        # The dispatcher resolves and passes the package settings path; redirect
        # the selection call to the synthetic settings instead of editing src/.
        real_selection = multinomial_vocal_category_model_selection

        def _wrapped(**kwargs):
            kwargs['settings_path'] = str(settings_json)
            return real_selection(**kwargs)

        monkeypatch.setattr(
            ms_dispatcher, 'multinomial_vocal_category_model_selection', _wrapped
        )

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        ms_dispatcher.dispatch_model_selection(
            argparse.Namespace(
                analysis_type='multinomial',
                univariate_path=univ_pkl,
                input_path=input_pkl,
                output_dir=str(ms_dir),
                anchor=True,
                pval=0.05,
                target_variable='bout_durations',
            )
        )

        # The dispatcher swallows downstream exceptions and prints a traceback;
        # reaching here means validation + routing executed. The baseline /
        # anchor blocks write at least the Step-0 pickle on a successful run.
        step_pkls = list(ms_dir.glob('model_selection_multinomial_*_step_*.pkl'))
        assert len(step_pkls) >= 1


class TestMultinomialTunerBranches:
    """
    Exercises the alternate / error / degenerate branches of
    ``_tune_multinomial_regularization`` that the strong-signal happy-path
    tuner test never reaches: the unsupported-metric guard, the
    too-few-samples / single-class early bail-out, the non-AUC scoring
    metrics, the all-NaN-grid fallback, and the one-SE interpretability
    rule. These are pure-NumPy / tiny-JAX paths so they stay cheap.
    """

    def test_unsupported_scoring_metric_raises(self):
        """
        A scoring metric outside the supported higher-/lower-is-better sets
        raises ``ValueError`` before any inner CV is attempted, so the guard
        cannot be silently bypassed by a typo in the settings block.
        """

        X = np.zeros((6, 1), dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="Unsupported inner_cv_scoring_metric"):
            _tune_multinomial_regularization(
                X, y,
                lambda_smooth_grid=np.array([1.0]),
                l2_reg_grid=np.array([0.01]),
                inner_cv_folds=2,
                inner_cv_scoring_metric='not_a_metric',
                inner_cv_use_one_se_rule=False,
                n_features=1,
                n_time_bins=1,
                smoothness_derivative_order=1,
                focal_gamma=0.0,
                uniform_class_weights=False,
                learning_rate=0.05,
                inner_max_iter=5,
                tol=0.01,
                random_state=0,
                verbose=False,
                use_lax_loop=False,
                regressor_cls=SmoothMultinomialLogisticRegression,
            )

    def test_degenerate_fold_returns_grid_centre_and_empty_audit(self):
        """
        When the training fold is too small to inner-split (fewer than
        ``inner_cv_folds * 2`` rows) or carries a single class, the tuner
        short-circuits to the median grid value of each grid and an empty
        audit payload — never touching the JAX estimator.
        """

        X = np.zeros((3, 1), dtype=np.float32)
        y = np.array([0, 0, 0], dtype=np.int32)   # single class -> early bail
        lambda_grid = np.array([0.1, 1.0, 10.0])
        l2_grid = np.array([0.001, 0.01, 0.1])
        best_lam, best_l2, audit = _tune_multinomial_regularization(
            X, y,
            lambda_smooth_grid=lambda_grid,
            l2_reg_grid=l2_grid,
            inner_cv_folds=2,
            inner_cv_scoring_metric='auc',
            inner_cv_use_one_se_rule=False,
            n_features=1,
            n_time_bins=1,
            smoothness_derivative_order=1,
            focal_gamma=0.0,
            uniform_class_weights=False,
            learning_rate=0.05,
            inner_max_iter=5,
            tol=0.01,
            random_state=0,
            verbose=False,
            use_lax_loop=False,
            regressor_cls=SmoothMultinomialLogisticRegression,
        )
        assert best_lam == 1.0          # median of the 3-point lambda grid
        assert best_l2 == 0.01          # median of the 3-point l2 grid
        assert audit == {
            'grid_scores': {}, 'grid_ses': {},
            'argmax_pair': None, 'one_se_applied': False, 'one_se_threshold': None,
        }

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("scoring_metric", ['score', 'f1', 'recall', 'mcc', 'll', 'brier', 'ece'])
    def test_non_auc_scoring_metrics_with_one_se_rule(self, scoring_metric):
        """
        Each supported non-AUC scoring metric (the balanced-accuracy /
        macro-F1 / macro-recall / MCC higher-is-better branch and the
        log-loss / Brier / ECE lower-is-better branch of ``_compute_score``)
        runs end-to-end, and the one-SE interpretability rule fires its
        threshold / in-band selection path. The returned pair is drawn from
        the supplied grids and the audit records whether the rule softened
        the choice.
        """

        rng = np.random.default_rng(1)
        n_per_class, n_time = 20, 1
        y = np.repeat([0, 1, 2], n_per_class).astype(np.int32)
        shift = (y.astype(float) - 1.0)[:, None]
        X = (1.2 * shift + 0.3 * rng.standard_normal((y.size, n_time))).astype(np.float32)

        lambda_grid = np.array([0.5, 1.0])
        l2_grid = np.array([0.01, 0.1])
        best_lam, best_l2, audit = _tune_multinomial_regularization(
            X, y,
            lambda_smooth_grid=lambda_grid,
            l2_reg_grid=l2_grid,
            inner_cv_folds=2,
            inner_cv_scoring_metric=scoring_metric,
            inner_cv_use_one_se_rule=True,
            n_features=1,
            n_time_bins=n_time,
            smoothness_derivative_order=1,
            focal_gamma=0.0,
            uniform_class_weights=False,
            learning_rate=0.05,
            inner_max_iter=15,
            tol=0.01,
            random_state=0,
            verbose=False,
            use_lax_loop=False,
            regressor_cls=SmoothMultinomialLogisticRegression,
        )
        assert best_lam in lambda_grid.tolist()
        assert best_l2 in l2_grid.tolist()
        assert set(audit.keys()) == {
            'grid_scores', 'grid_ses', 'argmax_pair',
            'one_se_applied', 'one_se_threshold',
        }
        assert isinstance(audit['one_se_applied'], bool)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_all_nan_grid_falls_back_to_grid_centre(self, monkeypatch):
        """
        When every inner fit raises (monkeypatched estimator), every
        ``(lambda_smooth, l2_reg)`` pair scores NaN, the per-fold exception
        handler is exercised, and the tuner returns the median grid values
        with a populated (NaN-scored) grid audit rather than crashing.
        """

        class _AlwaysRaises:
            """Stub regressor whose ``fit`` always raises to force the NaN path."""

            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                raise RuntimeError("forced inner-fit failure")

        rng = np.random.default_rng(2)
        y = np.repeat([0, 1, 2], 18).astype(np.int32)
        X = rng.standard_normal((y.size, 1)).astype(np.float32)
        lambda_grid = np.array([0.1, 1.0, 10.0])
        l2_grid = np.array([0.01, 0.1])

        best_lam, best_l2, audit = _tune_multinomial_regularization(
            X, y,
            lambda_smooth_grid=lambda_grid,
            l2_reg_grid=l2_grid,
            inner_cv_folds=2,
            inner_cv_scoring_metric='auc',
            inner_cv_use_one_se_rule=False,
            n_features=1,
            n_time_bins=1,
            smoothness_derivative_order=1,
            focal_gamma=0.0,
            uniform_class_weights=False,
            learning_rate=0.05,
            inner_max_iter=5,
            tol=0.01,
            random_state=0,
            verbose=True,   # also lights up the verbose exception-print branch
            use_lax_loop=False,
            regressor_cls=_AlwaysRaises,
        )
        assert best_lam == 1.0          # median of the 3-point lambda grid
        assert best_l2 == 0.1           # median of the 2-point l2 grid -> index 1
        # The grid shape is preserved even though every pair is NaN-scored.
        assert len(audit['grid_scores']) == lambda_grid.size * l2_grid.size
        assert all(not np.isfinite(v) for v in audit['grid_scores'].values())
        assert audit['argmax_pair'] is None


class TestMultinomialSplitterSessionStrategy:
    """The 'session' rejection-sampler branch of the splitter under stress."""

    def test_session_strategy_raises_when_no_valid_fold(self):
        """
        When every theoretical class cannot be covered on both sides of any
        session partition (here a class lives in a single session, so any
        split that holds it out on one side empties it on the other), the
        rejection sampler exhausts ``max_total_attempts`` — driving the
        tolerance-widening print branch — and raises ``RuntimeError``.
        """

        # Three sessions; classes 0/1 are everywhere but class 2 lives only
        # in session 2, so no session split can keep all three classes on
        # both sides simultaneously.
        groups = np.concatenate([np.full(20, 0), np.full(20, 1), np.full(20, 2)])
        y = np.concatenate([
            np.repeat([0, 1], 10),       # session 0: classes 0,1
            np.repeat([0, 1], 10),       # session 1: classes 0,1
            np.repeat([0, 1, 2], [3, 3, 14]),   # session 2: classes 0,1,2
        ])
        with pytest.raises(RuntimeError, match="valid splits"):
            get_stratified_group_splits_stable(
                groups=groups, y=y, n_categories=3, split_strategy='session',
                test_prop=1.0 / 3.0, n_splits=2, random_seed=0,
                tolerance=0.0, max_total_attempts=1500, widen_every=1000,
                widen_step=0.0,
            )


class TestMultinomialPipelineInit:
    """The ``__init__`` configuration-loading branches of the pipeline."""

    def test_init_loads_default_settings_when_none(self):
        """
        Constructing with ``modeling_settings_dict=None`` loads the shipped
        package JSON, derives ``history_frames`` from its camera rate /
        filter-history, and (when present) caches ``feature_boundaries``.
        Exercises the default-load arm of the constructor and the
        history-frame calculation.
        """

        pipeline = MultinomialModelingPipeline(modeling_settings_dict=None)
        assert isinstance(pipeline.modeling_settings, dict)
        assert isinstance(pipeline.history_frames, int)
        assert pipeline.history_frames >= 0

    def test_init_kwargs_are_set_as_attributes(self):
        """Extra keyword arguments are attached verbatim as instance attributes."""

        minimal = {
            'io': {'camera_sampling_rate': 60.0},
            'model_params': {'filter_history': 0.5},
            # Present so the optional `feature_boundaries` caching arm runs.
            'feature_boundaries': {'speed': [0.0, 10.0]},
        }
        pipeline = MultinomialModelingPipeline(
            modeling_settings_dict=minimal, custom_marker=42
        )
        assert pipeline.custom_marker == 42
        assert pipeline.history_frames == 30
        assert pipeline.feature_boundaries == {'speed': [0.0, 10.0]}

    def test_init_missing_history_keys_raises(self):
        """
        A settings dict lacking the camera-rate / filter-history keys raises
        ``KeyError`` from the history-frame calculation guard.
        """

        with pytest.raises(KeyError, match="Critical setting missing"):
            MultinomialModelingPipeline(modeling_settings_dict={'io': {}, 'model_params': {}})


class TestMultinomialRunnerExtraBranches:
    """
    The ``MultinomialModelRunner`` paths the strong-signal tri-strategy test
    leaves cold: the balance-train + in-loop regularisation-tuning fold path
    (with the cross-fold hyperparameter summary), the feature-not-found
    guard, and the binning passthrough on a multi-feature pickle.
    """

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_balanced_train_with_inner_tuning(self, tmp_path):
        """
        With ``balance_train_bool=True`` and ``tune_regularization_bool=True``
        each modelled fold (a) down-samples the training fold to equal
        per-class counts and switches the JAX fit to ``focal_gamma=0`` with
        uniform class weights, and (b) runs the inner-CV regularisation tuner
        before the outer fit. The cross-fold hyperparameter-selection summary
        then runs over the ``actual`` / ``null`` strategies. Asserts every
        modelled fold records a tuned flag and a finite selected
        ``lambda_smooth``, and that ``null_model_free`` keeps its NaN
        placeholder (it has no model to tune).
        """

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
            test_proportion=0.4,
        )
        hp = settings['hyperparameters']['jax_linear']['multinomial_logistic']
        hp['balance_train_bool'] = True
        hp['tune_regularization_bool'] = True
        hp['tune_regularization_params'] = dict(hp['tune_regularization_params'])
        hp['tune_regularization_params']['lambda_smooth_decades_each_side'] = 1
        hp['tune_regularization_params']['l2_reg_decades_each_side'] = 0
        hp['tune_regularization_params']['inner_cv_folds'] = 2
        hp['tune_regularization_params']['inner_max_iter'] = 12

        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=16,
        ))

        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        runner = MultinomialModelRunner(pipeline_instance=pipeline)
        _, results = runner.run_univariate_training(
            pkl_path=input_pkl, feat_name='self.speed'
        )

        n_splits = settings['model_params']['split_num']
        for strategy in ('actual', 'null'):
            folds = results[strategy]['folds']
            assert folds['balanced_train'] is True
            assert folds['hyperparams_tuned'] == [True] * n_splits
            assert all(np.isfinite(folds['selected_lambda_smooth']))
        mf_folds = results['null_model_free']['folds']
        assert all(not np.isfinite(v) for v in mf_folds['selected_lambda_smooth'])

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_run_univariate_training_unknown_feature_raises(self, tmp_path):
        """
        Requesting a feature absent from the loaded pickle raises ``KeyError``
        — the runner's not-found guard before any fold loop executes.
        """

        settings, _ = _build_extraction_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
        )
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=['self.speed'],
            session_ids=[f'session_{i}' for i in range(N_SESSIONS)],
            history_frames=HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=14,
        ))
        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        runner = MultinomialModelRunner(pipeline_instance=pipeline)
        with pytest.raises(KeyError, match="not found"):
            runner.run_univariate_training(pkl_path=input_pkl, feat_name='does.not.exist')

    def test_load_blocks_bins_all_features_when_no_filter(self, tmp_path):
        """
        ``load_univariate_data_blocks`` with ``feature_filter=None`` bins and
        returns *every* feature in the pickle (the legacy whole-pickle path),
        stripping the reserved ``_input_metadata`` key. Driven directly on a
        controlled two-feature pickle so the un-filtered branch is exercised
        without a full JAX run.
        """

        feature_names = ['self.speed', 'other.speed']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = str(build_multinomial_input_pickle(
            save_path=tmp_path / 'modeling_multinomial_input.pkl',
            feature_names=feature_names,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_categories=N_CATEGORIES,
            n_per_class_per_session=10,
        ))
        blocks = MultinomialModelRunner.load_univariate_data_blocks(
            input_pkl, bin_size=max(HISTORY_FRAMES, 1), feature_filter=None,
        )
        assert sorted(blocks.keys()) == sorted(feature_names)
        assert '_input_metadata' not in blocks
        for feat in feature_names:
            assert blocks[feat]['X'].shape[0] == blocks[feat]['y'].shape[0]
            assert blocks[feat]['n_time_bins'] == 1


class TestMultinomialExtractionEdgeCases:
    """Extraction guards: the out-of-range mixture-model IBI fallback and no-target abort."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extraction_out_of_range_mixture_model_index_writes_nan_ibi(self, tmp_path):
        """
        When ``mixture_model_component_index`` exceeds the per-sex mixture-model means length, the
        metadata IBI-threshold computation takes its NaN fallback arm for both
        sexes; extraction still completes and writes a valid pickle.
        """

        settings, save_dir = _build_extraction_settings(tmp_path)
        # Push the component index past the shipped mixture-model length so the
        # `mixture_model_idx_md < len(params['means'])` guard fails for both sexes.
        settings['model_params']['mixture_model_component_index'] = 999

        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_multinomial_input_data()

        pkls = list(save_dir.glob('modeling_multinomial_*.pkl'))
        assert len(pkls) == 1
        with pkls[0].open('rb') as fh:
            artifact = pickle.load(fh)
        ibi = artifact['_input_metadata']['ibi_thresholds']
        assert np.isnan(ibi['male']) and np.isnan(ibi['female'])

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extraction_all_noise_aborts_without_pickle(self, tmp_path):
        """
        A session tree whose target USVs are *all* the noise category yields
        no non-noise multinomial targets after the noise filter, so the
        extractor reaches the ``No valid data extracted. Aborting save.``
        guard and writes no input pickle.
        """

        # Build the tree, then overwrite every target USV row's category with
        # the noise label so `events_by_category` carries no usable class.
        session_roots = build_multinomial_session_tree(base_dir=tmp_path / 'sessions')
        for root in session_roots:
            csv_path = next((root / 'audio').glob('*_usv_summary.csv'))
            df = pls.read_csv(csv_path)
            df = df.with_columns(
                pls.lit(NOISE_CATEGORY).alias('vae_supercategory'),
                pls.lit(NOISE_CATEGORY).alias('vae_category'),
            )
            df.write_csv(csv_path)

        list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
        save_dir = tmp_path / 'out'
        save_dir.mkdir(parents=True, exist_ok=True)
        settings = build_modeling_settings(
            session_list_file=list_file,
            save_directory=save_dir,
            camera_sampling_rate=CAMERA_FPS,
            filter_history=FILTER_HISTORY,
        )
        _apply_multinomial_overrides(settings)

        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_multinomial_input_data()

        assert list(save_dir.glob('modeling_multinomial_*.pkl')) == []

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extraction_with_vocal_predictors_adds_usv_columns(self, tmp_path):
        """
        With ``usv_predictor_type='categories_rate'`` the extractor builds
        partner-side vocal-signal predictor columns
        (``build_vocal_signal_columns``), attaches them with
        ``with_columns`` (the vocal-column injection arm), and slices them
        into per-event windows under their own non-mouse-prefixed feature
        keys. Asserts at least one ``usv_*`` feature key lands in the output
        pickle alongside the kinematic features.
        """

        settings, save_dir = _build_extraction_settings(tmp_path)
        # Re-enable vocal predictors (the multinomial overrides disable them).
        settings['vocal_features']['usv_predictor_type'] = 'categories_rate'

        pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
        pipeline.extract_and_save_multinomial_input_data()

        pkls = list(save_dir.glob('modeling_multinomial_*.pkl'))
        assert len(pkls) == 1
        with pkls[0].open('rb') as fh:
            artifact = pickle.load(fh)
        feature_keys = [k for k in artifact if not k.startswith('_')]
        assert any(
            tok in k for k in feature_keys
            for tok in ('usv_rate', 'usv_cat_', 'usv_event')
        ), f"expected at least one vocal-signal feature key, got {feature_keys}"
