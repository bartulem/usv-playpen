"""
@author: bartulem
End-to-end smoke tests for the binomial (one-vs-rest) USV vocal-category
modeling pipeline and its forward-stepwise model-selection path, driven
entirely on tiny synthetic data.

These tests deliberately exercise the *production* code paths of two target
modules rather than isolated helpers:

* ``modeling_vocal_categories_binomial.VocalCategoryModelingPipeline`` — the
  full ``extract_and_save_category_input_data`` extraction (loaders, role
  resolution, kinematic-column selection, vocal-column building,
  harmonization, cross-session z-scoring, predictor audits, epoch slicing,
  and the ``{feature: {session: {target_feature_arr, other_feature_arr}}}``
  serialization), plus the univariate ``_run_modeling_category`` fit (both
  the ``create_category_splits`` 'mixed' and 'session' strategies, the
  actual / null_other conditions, and the sklearn basis-projected
  ``LogisticRegressionCV`` engine).

* ``model_selection.vocal_category_model_selection`` — the binary
  target-category-vs-other forward-selection orchestrator: univariate
  candidate ranking against the size-matched null, the 'mixed' and 'session'
  CV-fold construction, the auto-anchor Step-0 sweep, the greedy forward
  search, and the final CV-based filter-shape refit, with every per-step
  pickle written under ``tmp_path``.

The univariate ``category`` dispatch is driven through
``main_univariate_dispatcher.dispatch_univariate_job`` so the dispatcher's
routing into ``VocalCategoryModelingPipeline._run_modeling_category`` is
covered too; the consolidated per-feature pickles are then fed to
``vocal_category_model_selection``.

Building the target-vs-other contrast
--------------------------------------
The shared ``_synth`` USV-summary builder assigns *every* row the same
category integer, which would leave the one-vs-rest extraction with an empty
"other" class. This module therefore defines a *local* USV-summary builder
(``_write_category_usv_summary``) that interleaves two non-noise categories
(target == 1, other == 2) within every bout while keeping category 0 reserved
as noise, so both the positive (target) and negative (other) pools are
non-empty in every session. Everything else (behavioral CSV, track H5,
session tree, settings) reuses ``_synth`` unchanged.

Warning policy
--------------
The project runs pytest with ``filterwarnings = ["error"]``. The modeling
import chain pulls ``optax`` -> a one-time JAX ``DeprecationWarning``, so all
top-level modeling imports below are wrapped in a ``warnings.catch_warnings``
block that ignores ``DeprecationWarning`` during import. At run time the
sklearn ``LogisticRegressionCV`` on tiny synthetic folds can emit convergence
and undefined-metric warnings, and ``astropy``'s Gaussian smoothing emits an
``AstropyUserWarning`` for the tiny synthetic traces; both are demoted with
narrow per-test ``@pytest.mark.filterwarnings`` markers. ``matplotlib`` is
forced onto the headless ``Agg`` backend because the dispatcher imports
``pyplot`` for basis-verification plotting.
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
    import usv_playpen.modeling.main_univariate_dispatcher as univ_dispatcher
    import usv_playpen.modeling.modeling_vocal_categories_binomial as category_module
    from usv_playpen.modeling.model_selection import vocal_category_model_selection
    from usv_playpen.modeling.modeling_vocal_categories_binomial import (
        VocalCategoryModelingPipeline,
        _collect_category_windows,
    )


# Tiny-data geometry shared across tests. Chosen so the one-vs-rest extraction
# yields a workable number of target (category 1) and other (category 2)
# events per session while keeping every array small. ``HISTORY_FRAMES`` is the
# derived ``floor(CAMERA_FPS * FILTER_HISTORY)`` and is the column count of
# every per-event window; any directly-injected input pickle must match it.
CAMERA_FPS = 60.0
FILTER_HISTORY = 0.5
HISTORY_FRAMES = int(np.floor(CAMERA_FPS * FILTER_HISTORY))
N_FRAMES = 7200       # 120 s sessions -> plenty of room for spaced events
N_SESSIONS = 4
N_BOUTS = 16
USV_PER_BOUT = 4
TARGET_CATEGORY = 1
OTHER_CATEGORY = 2
NOISE_CATEGORY = 0

# The USV category column the synthetic summaries are labelled with; mirrors
# the shipped JSON default so the loaders / metadata route through it.
CATEGORY_COLUMN = 'vae_supercategory'


def _write_category_usv_summary(
        session_root: Path,
        target_mouse: str,
        partner_mouse: str,
        camera_fps: float,
        n_frames: int,
        filter_history: float,
        n_bouts: int,
        usv_per_bout: int,
        category_column: str,
        manifold_columns: tuple[str, str],
        seed: int,
        csv_sep: str = ',',
) -> Path:
    """
    Description
    -----------
    Writes a synthetic ``*_usv_summary.csv`` under ``<session_root>/audio`` that
    yields a *non-empty* target (category ``TARGET_CATEGORY``) and other
    (category ``OTHER_CATEGORY``) pool for the focal/target mouse when consumed
    by ``find_usv_categories`` in one-vs-rest mode.

    The shared ``_synth.build_usv_summary_csv`` assigns every row the same
    integer category, which would collapse the one-vs-rest negative class to
    zero rows. This builder instead interleaves the two non-noise categories
    syllable-by-syllable inside every bout, so each session contributes both
    classes. Category 0 stays reserved as noise (never emitted here) so the
    ``usv_noise_categories=[0]`` filter is a no-op on this data.

    Construction guarantees, per session:
      - The first ``filter_history`` seconds (plus generous slack) are silent,
        so no event is clipped by the start-of-session history filter or by the
        per-event history-window bounds check at extraction time.
      - Events are spread across the remaining session with wide spacing.
      - Each event alternates between ``TARGET_CATEGORY`` and ``OTHER_CATEGORY``
        so both pools are populated and roughly balanced.
      - A couple of sparse partner-mouse events are added late in the session so
        partner-only vocal predictors have content.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session ROOT directory; the ``audio`` subtree is created if absent.
    target_mouse (str)
        Emitter name for all focal/target USV rows.
    partner_mouse (str)
        Emitter recorded on a couple of sparse late rows so partner vocal
        predictors are exercised.
    camera_fps (float)
        Camera frame-rate (used to size the session duration in seconds).
    n_frames (int)
        Total session length in frames (defines the session duration).
    filter_history (float)
        Pre-event history length in seconds; the start of the session is kept
        silent so early events are never clipped.
    n_bouts (int)
        Number of spaced event clusters to synthesize for the target mouse.
    usv_per_bout (int)
        Syllables per cluster (alternating target / other categories).
    category_column (str)
        Name of the integer category column written.
    manifold_columns (tuple of str)
        Two column names holding synthetic acoustic-manifold coordinates.
    seed (int)
        Seed for the manifold-coordinate noise generator (reproducible tables).
    csv_sep (str)
        Field separator passed straight to ``polars.write_csv``.

    Returns
    -------
    csv_path (pathlib.Path)
        Absolute path to the CSV that was written.
    """

    rng = np.random.default_rng(seed)
    session_duration_sec = n_frames / camera_fps

    warmup = filter_history * 2.0 + 1.0
    usable = session_duration_sec - warmup - filter_history * 2.0
    if usable <= 0 or n_bouts <= 0:
        raise ValueError("Session too short to host the requested number of events.")
    bout_spacing = usable / n_bouts

    syllable_gap = 0.3
    syllable_dur = 0.01

    emitters: list[str] = []
    starts: list[float] = []
    stops: list[float] = []
    categories: list[int] = []

    # One in every four syllables is the target category; the rest are "other".
    # The deliberate ~1:3 target:other imbalance keeps the natural-rate test
    # folds non-trivial AND gives the size-matched ``null_other`` condition
    # enough surplus "other" rows to draw its pseudo-classes from (it needs the
    # other pool to be at least twice the balanced training size, which a 50/50
    # split could never satisfy).
    flip = 0
    for b in range(n_bouts):
        bout_origin = warmup + b * bout_spacing
        for s in range(usv_per_bout):
            start = bout_origin + s * (syllable_dur + syllable_gap)
            emitters.append(target_mouse)
            starts.append(round(start, 6))
            stops.append(round(start + syllable_dur, 6))
            categories.append(TARGET_CATEGORY if (flip % 4 == 0) else OTHER_CATEGORY)
            flip += 1

    # Sparse partner events late in the session (after the last target event)
    # so partner-only vocal predictors have content without contaminating the
    # target's clean windows.
    partner_anchor = warmup + (n_bouts - 0.5) * bout_spacing
    for s in range(2):
        start = partner_anchor + s * 0.5
        emitters.append(partner_mouse)
        starts.append(round(start, 6))
        stops.append(round(start + syllable_dur, 6))
        categories.append(TARGET_CATEGORY if (s % 2 == 0) else OTHER_CATEGORY)

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


def _build_category_session_tree(
        base_dir: Path,
        n_sessions: int,
        manifold_columns: tuple[str, str] = ('vae_umap1', 'vae_umap2'),
) -> list[Path]:
    """
    Description
    -----------
    Builds a complete multi-session synthetic recording tree under ``base_dir``
    for the one-vs-rest category pipeline and returns the session ROOT
    directories. Each session gets the three on-disk artifacts the loaders
    require: the behavioral-feature CSV and the HDF5 track file (reused verbatim
    from ``_synth``), plus a *mixed-category* USV-summary CSV produced by the
    local ``_write_category_usv_summary`` so both target and other classes are
    populated. Mouse track names are uniquified per session so cross-session
    role resolution and the dyad-rename logic are genuinely exercised.

    Parameters
    ----------
    base_dir (pathlib.Path)
        Directory under which ``session_0``, ``session_1`` ... are created.
    n_sessions (int)
        Number of session directories to build.
    manifold_columns (tuple of str)
        Acoustic-manifold coordinate column names written into each summary.

    Returns
    -------
    session_roots (list of pathlib.Path)
        Absolute session ROOT directories, in creation order.
    """

    base_dir.mkdir(parents=True, exist_ok=True)
    egocentric_features = ['speed', 'neck_elevation']
    session_roots: list[Path] = []

    for s_idx in range(n_sessions):
        session_root = base_dir / f"session_{s_idx}"
        session_root.mkdir(parents=True, exist_ok=True)
        mouse_names = [f"s{s_idx}_m_male", f"s{s_idx}_m_female"]

        build_behavioral_features_csv(
            session_root=session_root,
            mouse_names=mouse_names,
            n_frames=N_FRAMES,
            egocentric_features=egocentric_features,
            seed=s_idx,
        )
        build_track_h5(
            session_root=session_root,
            mouse_names=mouse_names,
            camera_fps=CAMERA_FPS,
        )
        # Predictor index 1 (female) is the partner; target index 0 (male) is
        # the focal/emitter by the project convention used in the smoke tests.
        _write_category_usv_summary(
            session_root=session_root,
            target_mouse=mouse_names[0],
            partner_mouse=mouse_names[1],
            camera_fps=CAMERA_FPS,
            n_frames=N_FRAMES,
            filter_history=FILTER_HISTORY,
            n_bouts=N_BOUTS,
            usv_per_bout=USV_PER_BOUT,
            category_column=CATEGORY_COLUMN,
            manifold_columns=manifold_columns,
            seed=s_idx,
        )
        session_roots.append(session_root)

    return session_roots


def _write_category_input_pickle(
        save_path: Path,
        feature_names: list[str],
        session_ids: list[str],
        history_frames: int,
        n_target: int = 24,
        n_other: int = 72,
        input_metadata: dict | None = None,
        seed: int = 0,
) -> Path:
    """
    Description
    -----------
    Serializes a controlled one-vs-rest category "modeling input pickle"
    directly, bypassing the on-disk extraction. The artifact matches the schema
    ``vocal_category_model_selection`` and the univariate category dispatch
    load:

        {
          '<generic_feature>': {
              '<session_id>': {
                  'target_feature_arr': np.ndarray (n_target, history_frames),
                  'other_feature_arr':  np.ndarray (n_other,  history_frames),
              }, ...
          }, ...,
          '_input_metadata': {...}   # reserved category-provenance block
        }

    Unlike the extraction-driven fixtures, the target and other windows here
    carry a *strong, class-dependent mean offset* so a logistic model finds
    clearly above-chance structure on the target-vs-other contrast — this is
    what lets the model-selection candidate screen (which keeps features whose
    actual log-loss beats the size-matched ``null_other`` null) admit
    candidates and run the forward search. The deliberate ``n_other`` >>
    ``n_target`` imbalance leaves the ``null_other`` condition enough surplus
    "other" rows to draw its size-matched pseudo-classes from.

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
    n_target (int)
        Positive (target-category) events per session.
    n_other (int)
        Negative (other-category) events per session.
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
            target_arr = (
                0.9 + 0.1 * f_idx
                + rng.standard_normal((n_target, history_frames))
            ).astype(float)
            other_arr = (
                -0.9 - 0.1 * f_idx
                + rng.standard_normal((n_other, history_frames))
            ).astype(float)
            artifact[feature][sess] = {
                'target_feature_arr': target_arr,
                'other_feature_arr': other_arr,
            }

    if input_metadata is not None:
        artifact['_input_metadata'] = input_metadata

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as fh:
        pickle.dump(artifact, fh)
    return save_path


def _category_input_metadata():
    """
    Description
    -----------
    Returns the minimal ``_input_metadata`` block the category univariate
    dispatch and ``vocal_category_model_selection`` consult on a directly-injected
    input pickle: the ``analysis_tag`` (so per-feature output filenames group
    correctly) and the ``analysis_specific.usv_category_column_name`` (which the
    selection orchestrator reads to build the per-step filename prefix).

    Returns
    -------
    metadata (dict)
        The reserved-block dict to embed under ``_input_metadata``.
    """

    return {
        'analysis_tag': f"category_{CATEGORY_COLUMN}_{TARGET_CATEGORY}",
        'analysis_specific': {
            'target_category': TARGET_CATEGORY,
            'usv_category_column_name': CATEGORY_COLUMN,
        },
    }


def _build_category_settings(tmp_path: Path, **overrides):
    """
    Description
    -----------
    Builds the mixed-category synthetic session tree, the session-list file, and
    the trimmed ``modeling_settings`` dict for a category smoke run, all rooted
    under ``tmp_path``.

    Beyond the shrink ``_synth.build_modeling_settings`` already applies, this
    helper sets the two knobs the category path specifically needs:
      - ``model_params.model_type`` — ``vocal_category_model_selection`` reads
        ``model_type`` (the shipped JSON default is ``None``), distinct from the
        ``model_engine`` key the univariate extraction / dispatch consult; both
        are pinned to the same engine so the two stages agree.
      - ``vocal_features.usv_predictor_type`` defaults to ``None`` here (kept
        tiny); callers may override via ``overrides``.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        The pytest-provided per-test scratch directory; every artifact lives
        below it so nothing is ever written into the package tree.
    overrides (dict)
        Extra keyword arguments forwarded to ``build_modeling_settings`` (e.g.
        ``model_engine``, ``usv_predictor_type``, ``split_strategy``,
        ``split_num``, ``test_proportion``). ``model_type`` is popped out and
        applied to ``model_params`` directly.

    Returns
    -------
    settings (dict)
        The ready-to-use ``modeling_settings`` dictionary.
    save_dir (pathlib.Path)
        The pipeline output directory (``tmp_path / 'out'``).
    """

    n_sessions = overrides.pop('n_sessions', N_SESSIONS)
    model_type = overrides.pop('model_type', 'sklearn')

    session_roots = _build_category_session_tree(
        base_dir=tmp_path / 'sessions', n_sessions=n_sessions
    )
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
    # ``vocal_category_model_selection`` reads ``model_type`` (not the
    # ``model_engine`` the univariate dispatch / extraction read); the shipped
    # JSON leaves it as ``None`` so it must be pinned explicitly here.
    settings['model_params']['model_type'] = model_type
    return settings, save_dir


def _extract_category_input(settings, save_dir, target_category=TARGET_CATEGORY):
    """
    Description
    -----------
    Runs the real ``VocalCategoryModelingPipeline.extract_and_save_category_input_data``
    once for ``target_category`` and returns both the on-disk pickle path and
    the loaded artifact dict, asserting exactly one ``modeling_*.pkl`` was
    produced under ``save_dir``.

    Parameters
    ----------
    settings (dict)
        The synthetic ``modeling_settings`` dict.
    save_dir (pathlib.Path)
        The pipeline output directory; the input pickle is written here.
    target_category (int)
        The positive-class category index to extract (one-vs-rest).

    Returns
    -------
    pkl_path (pathlib.Path)
        Path to the single extracted modeling input pickle.
    artifact (dict)
        The deserialized pickle contents.
    """

    pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
    pipeline.extract_and_save_category_input_data(target_category=target_category)

    pkls = list(save_dir.glob('modeling_*.pkl'))
    assert len(pkls) == 1, f"expected exactly one input pickle, got {pkls}"
    with pkls[0].open('rb') as fh:
        artifact = pickle.load(fh)
    return pkls[0], artifact


def _run_univariate_category_and_consolidate(
        settings, input_pkl, feature_names, out_dir, monkeypatch, target_category=TARGET_CATEGORY
):
    """
    Description
    -----------
    Runs the univariate ``category`` dispatcher once per feature against the
    extracted input pickle, then consolidates the per-feature result pickles
    into a single ``{feature: results}`` dict written to a path whose basename
    carries the ``category_<col>_<idx>`` tag ``vocal_category_model_selection``
    parses for target/condition metadata.

    The dispatcher hard-codes loading the *package* settings JSON; this helper
    monkeypatches ``json.load`` inside the dispatcher module so the synthetic,
    shrunk settings are used instead — without ever touching the package file.
    The combined pickle re-embeds the upstream ``_input_metadata`` block so the
    selection orchestrator can read ``analysis_specific.usv_category_column_name``.

    Parameters
    ----------
    settings (dict)
        The synthetic settings the dispatcher should consume.
    input_pkl (str)
        Path to the extracted category modeling input pickle.
    feature_names (list of str)
        Behavioral-feature keys to run (one dispatcher invocation each).
    out_dir (pathlib.Path)
        Directory for the per-feature and consolidated univariate pickles.
    monkeypatch (pytest.MonkeyPatch)
        Used to redirect the dispatcher's settings load.
    target_category (int)
        The positive-class category index (used to name the combined pickle).

    Returns
    -------
    combined_path (pathlib.Path)
        Path to the consolidated univariate-results pickle.
    per_feature_paths (list of pathlib.Path)
        The individual per-feature univariate pickles that were produced.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

    # Pull the upstream input metadata so the consolidated pickle stays
    # provenance-complete for the selection orchestrator.
    with open(input_pkl, 'rb') as fh:
        loaded = pickle.load(fh)
    input_md = loaded['_input_metadata']
    feature_keys = [k for k in loaded if not k.startswith('_')]
    del loaded

    name_to_idx = {name: idx for idx, name in enumerate(sorted(feature_keys))}
    for feat in feature_names:
        univ_dispatcher.dispatch_univariate_job(
            argparse.Namespace(
                analysis_type='category',
                feature_idx=name_to_idx[feat],
                input_data=input_pkl,
                output_dir=str(out_dir),
            )
        )

    per_feature_paths = sorted(out_dir.glob('univariate_*.pkl'))
    combined = {}
    for p in per_feature_paths:
        with p.open('rb') as fh:
            payload = pickle.load(fh)
        for key, value in payload.items():
            if not key.startswith('_'):
                combined[key] = value
    combined['_input_metadata'] = input_md

    # The selection function parses ``category_(\d+)`` out of the univariate
    # filename for the target/condition metadata, so name the combined pickle
    # with the canonical ``category_<idx>_<condition>`` token.
    combined_path = out_dir / f"univariate_category_{target_category}_male-female_splits.pkl"
    with combined_path.open('wb') as fh:
        pickle.dump(combined, fh)
    return combined_path, per_feature_paths


class TestCategoryInputExtraction:
    """End-to-end extraction of the one-vs-rest category input pickle."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_produces_target_vs_other_pickle(self, tmp_path):
        """
        The real ``extract_and_save_category_input_data`` writes a single
        ``modeling_category_<col>_<idx>_*.pkl`` whose structure matches the
        documented contract: a nested ``{generic_feature: {session:
        {target_feature_arr, other_feature_arr}}}`` dict carrying a reserved
        ``_input_metadata`` block. Every per-event window is ``HISTORY_FRAMES``
        wide, both the target and other pools are non-empty in aggregate, and
        the metadata pins ``analysis_type == 'category'`` with the configured
        ``target_category`` / ``usv_category_column_name``.
        """

        settings, save_dir = _build_category_settings(tmp_path, model_engine='sklearn')
        pkl_path, artifact = _extract_category_input(settings, save_dir)

        assert pkl_path.name.startswith(f"modeling_category_{CATEGORY_COLUMN}_{TARGET_CATEGORY}_")
        assert '_input_metadata' in artifact

        feature_keys = sorted(k for k in artifact if not k.startswith('_'))
        # Egocentric ['speed', 'neck_elevation'] expand to self.* and other.*.
        assert feature_keys == [
            'other.neck_elevation', 'other.speed',
            'self.neck_elevation', 'self.speed',
        ]

        anchor = feature_keys[0]
        sessions = sorted(artifact[anchor].keys())
        assert len(sessions) >= 2

        total_target = 0
        total_other = 0
        for sess in sessions:
            targ = artifact[anchor][sess]['target_feature_arr']
            other = artifact[anchor][sess]['other_feature_arr']
            assert targ.shape[1] == HISTORY_FRAMES
            assert other.shape[1] == HISTORY_FRAMES
            assert np.isfinite(targ).all() and np.isfinite(other).all()
            total_target += targ.shape[0]
            total_other += other.shape[0]

            # Intra-session alignment: every feature shares this session's
            # target / other event counts.
            for feat in feature_keys[1:]:
                assert artifact[feat][sess]['target_feature_arr'].shape[0] == targ.shape[0]
                assert artifact[feat][sess]['other_feature_arr'].shape[0] == other.shape[0]

        assert total_target > 0
        assert total_other > 0

        md = artifact['_input_metadata']
        assert md['analysis_type'] == 'category'
        assert md['analysis_tag'] == f"category_{CATEGORY_COLUMN}_{TARGET_CATEGORY}"
        assert md['analysis_specific']['target_category'] == TARGET_CATEGORY
        assert md['analysis_specific']['usv_category_column_name'] == CATEGORY_COLUMN
        assert sorted(md['feature_zoo_kept']) == feature_keys

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    def test_extraction_with_partner_vocal_predictors(self, tmp_path):
        """
        With ``usv_predictor_type='categories_rate'`` the extraction additionally
        materializes a partner per-category vocal predictor column
        (``other.usv_cat_2``), exercising ``build_vocal_signal_columns`` plus the
        identity-guard that excludes the target category's own
        ``usv_cat_<target>`` self-predictor. ``other.usv_cat_1`` (the target
        category index) is therefore present (partner side is not guarded) while
        the *self* side excludes it.
        """

        settings, save_dir = _build_category_settings(
            tmp_path, model_engine='sklearn', usv_predictor_type='categories_rate'
        )
        _pkl_path, artifact = _extract_category_input(settings, save_dir)

        feature_keys = {k for k in artifact if not k.startswith('_')}
        # The partner (other) side ingests the per-category vocal density traces;
        # category 2 (a non-target category) must surface as a predictor.
        assert 'other.usv_cat_2' in feature_keys


class TestUnivariateCategoryDispatcher:
    """The univariate category dispatcher on the extracted input pickle."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_dispatch_category_sklearn_writes_per_feature_pickles(self, tmp_path, monkeypatch):
        """
        ``dispatch_univariate_job`` (analysis 'category', sklearn engine) routes
        into ``VocalCategoryModelingPipeline._run_modeling_category`` and writes
        one per-feature pickle per feature index, each carrying ``_run_metadata``
        and ``_input_metadata`` blocks plus the actual / null results branches
        with the full scalar-metric key set and the per-fold ``filter_shapes``
        array. This exercises both the 'mixed' split strategy and the actual /
        null_other conditions inside ``create_category_splits``.
        """

        settings, save_dir = _build_category_settings(
            tmp_path, model_engine='sklearn', split_strategy='mixed', split_num=2,
        )
        input_pkl, artifact = _extract_category_input(settings, save_dir)
        feature_keys = sorted(k for k in artifact if not k.startswith('_'))

        out_dir = tmp_path / 'univariate'
        out_dir.mkdir()
        monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

        for feature_idx in range(len(feature_keys)):
            univ_dispatcher.dispatch_univariate_job(
                argparse.Namespace(
                    analysis_type='category',
                    feature_idx=feature_idx,
                    input_data=str(input_pkl),
                    output_dir=str(out_dir),
                )
            )

        per_feature = sorted(out_dir.glob('univariate_*.pkl'))
        assert len(per_feature) == len(feature_keys)

        with per_feature[0].open('rb') as fh:
            payload = pickle.load(fh)
        assert '_run_metadata' in payload
        assert '_input_metadata' in payload

        feat_key = next(k for k in payload if not k.startswith('_'))
        for branch_name in ('actual', 'null'):
            branch = payload[feat_key][branch_name]
            for metric in ('ll', 'auc', 'score', 'brier', 'ece', 'mcc', 'f1', 'recall'):
                assert metric in branch
                assert branch[metric].shape == (settings['model_params']['split_num'],)
        assert payload[feat_key]['actual']['filter_shapes'].shape == (
            settings['model_params']['split_num'], HISTORY_FRAMES,
        )
        # At least one fold across actual + null is finite (fitted), proving the
        # category split / fit path actually ran rather than failing every fold.
        finite_any = (
            np.isfinite(payload[feat_key]['actual']['ll']).any()
            or np.isfinite(payload[feat_key]['null']['ll']).any()
        )
        assert finite_any

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_dispatch_category_sklearn_session_strategy(self, tmp_path, monkeypatch):
        """
        Re-runs the univariate category dispatch with ``split_strategy='session'``
        so the ``create_category_splits`` 'session' branch (whole-session
        ``ShuffleSplit`` + ``pool_session_arrays``) is exercised distinctly from
        the 'mixed' branch above. Asserts a per-feature pickle is produced for
        the anchor feature with the expected metric arrays.
        """

        settings, save_dir = _build_category_settings(
            tmp_path, model_engine='sklearn', split_strategy='session', split_num=2,
            test_proportion=0.5, n_sessions=4,
        )
        input_pkl, _artifact = _extract_category_input(settings, save_dir)

        out_dir = tmp_path / 'univariate'
        out_dir.mkdir()
        monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

        univ_dispatcher.dispatch_univariate_job(
            argparse.Namespace(
                analysis_type='category',
                feature_idx=0,
                input_data=str(input_pkl),
                output_dir=str(out_dir),
            )
        )

        per_feature = sorted(out_dir.glob('univariate_*.pkl'))
        assert len(per_feature) == 1
        with per_feature[0].open('rb') as fh:
            payload = pickle.load(fh)
        feat_key = next(k for k in payload if not k.startswith('_'))
        assert payload[feat_key]['actual']['auc'].shape == (
            settings['model_params']['split_num'],
        )

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    def test_run_modeling_category_pygam_engine(self, tmp_path):
        """
        Drives ``VocalCategoryModelingPipeline._run_modeling_category`` directly
        with the ``pygam`` engine (and ``basis_matrix=None``) on a strong-signal
        injected category input, exercising the tensor-product-spline
        ``LogisticGAM`` branch: ``unroll_history_matrix``, per-frame
        probability averaging, partial-dependence filter-shape extraction, and
        the GAM convergence-diagnostic bookkeeping. The GAM fit under Python 3.13
        emits a ``DeprecationWarning: Bitwise inversion`` that is demoted here so
        the per-fold try/except does not silently mark every fold failed. Asserts
        the actual branch produced the expected metric arrays with at least one
        finite fold.
        """

        settings, _ = _build_category_settings(
            tmp_path, model_engine='pygam', split_strategy='mixed', split_num=2,
        )
        feature_data = {}
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        rng = np.random.default_rng(0)
        for sess in session_ids:
            feature_data[sess] = {
                'target_feature_arr': (0.9 + rng.standard_normal((20, HISTORY_FRAMES))).astype(float),
                'other_feature_arr': (-0.9 + rng.standard_normal((60, HISTORY_FRAMES))).astype(float),
            }

        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        fn, res = pipeline._run_modeling_category('self.speed', feature_data, basis_matrix=None)

        assert fn == 'self.speed'
        for metric in ('ll', 'auc', 'score', 'brier', 'mcc', 'f1', 'recall'):
            assert res['actual'][metric].shape == (settings['model_params']['split_num'],)
        assert res['actual']['filter_shapes'].shape == (
            settings['model_params']['split_num'], HISTORY_FRAMES,
        )
        # The pygam branch fitted at least one fold (proves the GAM path ran).
        assert np.isfinite(res['actual']['ll']).any()


class TestVocalCategoryModelSelection:
    """The forward-stepwise ``vocal_category_model_selection`` orchestrator."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_category_selection_mixed_anchored(self, tmp_path, monkeypatch):
        """
        Runs ``vocal_category_model_selection`` end-to-end on a strong-signal
        injected category input plus a freshly-computed univariate ranking, with
        the 'mixed' split strategy and the auto-anchor enabled. Exercises the
        pre-pooled ``pooled_category_cache`` path, the stratified-shuffle fold
        construction, the anchored Step-0 sweep, the greedy forward search, and
        the final CV-based filter-shape refit. Asserts the per-step pickles are
        written with the documented structure and that the accepted feature set
        never shrinks across steps.
        """

        settings, _ = _build_category_settings(
            tmp_path, model_engine='sklearn', model_type='sklearn',
            split_strategy='mixed', split_num=2,
        )
        feature_keys = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = _write_category_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_keys,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            input_metadata=_category_input_metadata(),
        )

        combined_path, per_feature = _run_univariate_category_and_consolidate(
            settings=settings,
            input_pkl=str(input_pkl),
            feature_names=feature_keys,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )
        assert len(per_feature) == len(feature_keys)

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_category_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=str(input_pkl),
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_category_*_step_*.pkl'))
        # If no univariate candidate clears the (Bonferroni-adjusted) null
        # screen the orchestrator aborts before any step pickle; the synthetic
        # signal is strong + p_val loose so at least the anchored Step-0 lands.
        assert len(step_pkls) >= 1, "expected at least one forward-selection step pickle"

        accepted_counts = []
        for p in step_pkls:
            with p.open('rb') as fh:
                step = pickle.load(fh)
            assert 'current_features' in step
            assert 'baseline_score' in step
            assert 'candidates_summary' in step
            accepted_counts.append(len(step['current_features']))

        assert min(accepted_counts) >= 1
        assert accepted_counts == sorted(accepted_counts)

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    def test_category_selection_mixed_anchored_pygam(self, tmp_path, monkeypatch):
        """
        Runs ``vocal_category_model_selection`` with the ``pygam`` engine and the
        auto-anchor enabled, exercising the tensor-product-spline ``LogisticGAM``
        branches of the anchored Step-0 sweep, the forward search, and the final
        CV-based partial-dependence filter-shape refit — code paths the sklearn
        runs above never touch. The univariate screen is still computed with the
        sklearn engine (fast) by feeding the GAM selection a sklearn-fitted
        ranking; only the selection engine is switched to ``pygam`` so the GAM
        multivariate fit / unroll / predict path is what gets covered. Asserts at
        least one per-step pickle is written.
        """

        # Univariate ranking uses the fast sklearn engine; selection uses pygam.
        univ_settings, _ = _build_category_settings(
            tmp_path, model_engine='sklearn', model_type='sklearn',
            split_strategy='mixed', split_num=2,
        )
        sel_settings = json.loads(json.dumps(univ_settings))
        sel_settings['model_params']['model_engine'] = 'pygam'
        sel_settings['model_params']['model_type'] = 'pygam'

        feature_keys = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = _write_category_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_keys,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            input_metadata=_category_input_metadata(),
        )

        combined_path, _ = _run_univariate_category_and_consolidate(
            settings=univ_settings,
            input_pkl=str(input_pkl),
            feature_names=feature_keys,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(sel_settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_category_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=str(input_pkl),
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=True,
            p_val=0.5,
        )

        step_pkls = sorted(ms_dir.glob('model_selection_category_*_step_*.pkl'))
        assert len(step_pkls) >= 1, "expected at least one pygam forward-selection step pickle"

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_category_selection_session_unanchored(self, tmp_path, monkeypatch):
        """
        Re-runs ``vocal_category_model_selection`` with ``split_strategy='session'``
        and *without* the auto-anchor, exercising the whole-session
        ``ShuffleSplit`` CV-fold construction and the non-anchored Step-0
        candidate sweep — code paths distinct from the 'mixed'/anchored run
        above. Asserts the orchestrator completes and emits at least one step
        pickle (or, if every candidate is screened out, aborts cleanly without
        raising).
        """

        settings, _ = _build_category_settings(
            tmp_path, model_engine='sklearn', model_type='sklearn',
            split_strategy='session', split_num=2, test_proportion=0.5,
        )
        # Six sessions so the session-level ShuffleSplit has train/test room.
        feature_keys = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(6)]
        input_pkl = _write_category_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_keys,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            n_target=16,
            n_other=48,
            input_metadata=_category_input_metadata(),
        )

        combined_path, _ = _run_univariate_category_and_consolidate(
            settings=settings,
            input_pkl=str(input_pkl),
            feature_names=feature_keys,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        vocal_category_model_selection(
            univariate_results_path=str(combined_path),
            input_data_path=str(input_pkl),
            settings_path=str(settings_json),
            output_directory=str(ms_dir),
            use_top_rank_as_anchor=False,
            p_val=0.5,
        )

        # Whether or not candidates survive the screen, the orchestrator must
        # not raise; on this synthetic data the non-anchored sweep completes and
        # at least the Step-0 pickle lands.
        step_pkls = list(ms_dir.glob('model_selection_category_*_step_*.pkl'))
        assert len(step_pkls) >= 1


class TestModelSelectionDispatcherCategory:
    """The model-selection dispatcher routing for the 'category' task."""

    @pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_dispatch_category_runs_through_validation_and_routing(self, tmp_path, monkeypatch):
        """
        ``dispatch_model_selection`` validates the three required paths and routes
        the 'category' task into ``vocal_category_model_selection``. The
        dispatcher auto-resolves the package settings JSON, so the real selection
        function is wrapped to inject the synthetic settings path; this still
        exercises the dispatcher's own validation and routing statements
        end-to-end without raising.
        """

        settings, _ = _build_category_settings(
            tmp_path, model_engine='sklearn', model_type='sklearn',
            split_strategy='mixed', split_num=2,
        )
        feature_keys = ['self.speed', 'other.speed', 'self.neck_elevation']
        session_ids = [f'session_{i}' for i in range(N_SESSIONS)]
        input_pkl = _write_category_input_pickle(
            save_path=tmp_path / 'modeling_input.pkl',
            feature_names=feature_keys,
            session_ids=session_ids,
            history_frames=HISTORY_FRAMES,
            input_metadata=_category_input_metadata(),
        )

        combined_path, _ = _run_univariate_category_and_consolidate(
            settings=settings,
            input_pkl=str(input_pkl),
            feature_names=feature_keys,
            out_dir=tmp_path / 'univariate',
            monkeypatch=monkeypatch,
        )

        settings_json = tmp_path / 'settings.json'
        settings_json.write_text(json.dumps(settings))

        # The dispatcher resolves and passes the package settings path; redirect
        # the selection call to the synthetic settings instead of editing src/.
        real_selection = vocal_category_model_selection

        def _wrapped(**kwargs):
            kwargs['settings_path'] = str(settings_json)
            return real_selection(**kwargs)

        monkeypatch.setattr(ms_dispatcher, 'vocal_category_model_selection', _wrapped)

        ms_dir = tmp_path / 'model_selection'
        ms_dir.mkdir()
        ms_dispatcher.dispatch_model_selection(
            argparse.Namespace(
                analysis_type='category',
                univariate_path=str(combined_path),
                input_path=str(input_pkl),
                output_dir=str(ms_dir),
                anchor=True,
                pval=0.5,
            )
        )

        # The dispatcher swallows downstream exceptions and prints a traceback;
        # reaching here without an uncaught exception means validation + routing
        # executed. On this synthetic data the wrapped selection runs the screen
        # to completion, so at least the Step-0 pickle is written.
        step_pkls = list(ms_dir.glob('model_selection_category_*_step_*.pkl'))
        assert len(step_pkls) >= 1


def _minimal_category_settings(
        split_strategy: str = 'mixed',
        split_num: int = 2,
        test_proportion: float = 0.3,
        random_seed: int = 0,
        model_engine: str = 'sklearn',
):
    """
    Description
    -----------
    Hand-rolls the smallest ``modeling_settings`` dict that lets
    ``VocalCategoryModelingPipeline.__init__`` succeed and that
    ``create_category_splits`` / ``_run_modeling_category`` can read end-to-end,
    without manufacturing an on-disk session tree. Only the keys those two
    methods (and the constructor's ``history_frames`` derivation) consult are
    populated.

    Parameters
    ----------
    split_strategy (str)
        ``'mixed'``, ``'session'`` (or a deliberately-unknown value to drive the
        ``create_category_splits`` terminal ``ValueError``).
    split_num (int)
        Number of CV folds / splits.
    test_proportion (float)
        Per-split test fraction.
    random_seed (int)
        Global / estimator seed.
    model_engine (str)
        ``'sklearn'`` or ``'pygam'`` — the per-feature fit engine selector.

    Returns
    -------
    settings (dict)
        The ready-to-use minimal ``modeling_settings`` dictionary.
    """

    return {
        'io': {'camera_sampling_rate': CAMERA_FPS, 'csv_separator': ','},
        'model_params': {
            'filter_history': FILTER_HISTORY,
            'split_strategy': split_strategy,
            'split_num': split_num,
            'test_proportion': test_proportion,
            'random_seed': random_seed,
            'model_engine': model_engine,
        },
        'hyperparameters': {
            'classical': {
                'logistic_regression': {
                    'penalty': 'l2', 'cs': [0.1], 'cv': 2,
                    'solver': 'lbfgs', 'max_iter': 200,
                },
                'pygam': {
                    'n_splines_value': 4, 'n_splines_time': 4,
                    'lam_penalty': 0.6, 'tol_val': 1e-4, 'max_iterations': 20,
                },
            }
        },
    }


def _category_feature_data(history_frames: int, n_target=24, n_other=72, n_sessions=4, seed=0):
    """
    Description
    -----------
    Manufactures a single-feature ``{session: {target_feature_arr,
    other_feature_arr}}`` dict in the shape ``create_category_splits`` and
    ``_run_modeling_category`` consume, with a strong class-dependent mean
    offset so a logistic / GAM fit finds clearly above-chance structure on the
    target-vs-other contrast. The deliberate ``n_other`` >> ``n_target``
    imbalance leaves the size-matched ``null_other`` condition surplus rows to
    draw its pseudo-classes from.

    Parameters
    ----------
    history_frames (int)
        Columns (temporal lags) per event window.
    n_target (int)
        Positive (target-category) events per session.
    n_other (int)
        Negative (other-category) events per session.
    n_sessions (int)
        Number of session groups.
    seed (int)
        RNG seed.

    Returns
    -------
    feature_data (dict)
        The ``{session: {target_feature_arr, other_feature_arr}}`` dict.
    """

    rng = np.random.default_rng(seed)
    feature_data = {}
    for i in range(n_sessions):
        feature_data[f'session_{i}'] = {
            'target_feature_arr': (0.9 + rng.standard_normal((n_target, history_frames))).astype(float),
            'other_feature_arr': (-0.9 + rng.standard_normal((n_other, history_frames))).astype(float),
        }
    return feature_data


class TestCategoryInitBranches:
    """Constructor branches: file-load, feature_boundaries, history KeyError, kwargs."""

    def test_init_loads_package_settings_when_none(self):
        """
        Passing ``modeling_settings_dict=None`` exercises the settings-file load
        path: the constructor resolves and reads the shipped package
        ``modeling_settings.json`` (read-only) and derives ``history_frames``
        from it.
        """

        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=None)
        assert isinstance(pipeline.history_frames, int)
        assert pipeline.history_frames > 0

    def test_init_attaches_feature_boundaries_and_kwargs(self):
        """
        A ``feature_boundaries`` block in the settings is copied onto the
        instance (the membership-gated assignment), and any extra ``**kwargs``
        are set as instance attributes (the trailing kwargs loop).
        """

        settings = _minimal_category_settings()
        settings['feature_boundaries'] = {'speed': [0.0, 1.0]}
        pipeline = VocalCategoryModelingPipeline(
            modeling_settings_dict=settings, extra_marker='sentinel'
        )
        assert pipeline.feature_boundaries == {'speed': [0.0, 1.0]}
        assert pipeline.extra_marker == 'sentinel'

    def test_init_raises_on_missing_history_keys(self):
        """
        Dropping ``model_params.filter_history`` makes the ``history_frames``
        derivation raise ``KeyError``, which the constructor re-wraps into a
        ``KeyError`` carrying the critical-setting message.
        """

        settings = _minimal_category_settings()
        del settings['model_params']['filter_history']
        with pytest.raises(KeyError, match="Critical setting missing"):
            VocalCategoryModelingPipeline(modeling_settings_dict=settings)


class TestCollectCategoryWindows:
    """The module-level ``_collect_category_windows`` slicer edge case."""

    def test_no_valid_events_returns_empty(self):
        """
        When every requested window falls off the start of the recording (each
        ``start`` index is negative), the slicer returns an empty
        ``(0, history_frames)`` array via the ``not np.any(valid_mask)`` guard
        rather than fabricating NaN rows.
        """

        col = np.arange(50, dtype=float)
        # Timestamps of 0.0 -> end index 0 -> start index -history_frames < 0.
        out = _collect_category_windows(
            times=np.zeros(5), column_data=col, sampling_rate=CAMERA_FPS,
            history_frames=HISTORY_FRAMES, max_frame_idx=col.size,
        )
        assert out.shape == (0, HISTORY_FRAMES)


class TestCreateCategorySplitsBranches:
    """Guard / strategy / null_other branches in ``create_category_splits``."""

    def test_fewer_than_two_valid_sessions_returns_empty(self):
        """
        A feature dict in which fewer than two sessions carry both a non-empty
        target and other pool (here: one good session, one degenerate session
        missing the ``other`` key) trips the ``len(valid_sessions) < 2`` early
        return AND the per-session ``except (KeyError, AttributeError)`` skip.
        """

        settings = _minimal_category_settings(split_strategy='mixed')
        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        feature_data = {
            'session_0': {
                'target_feature_arr': np.ones((5, HISTORY_FRAMES)),
                'other_feature_arr': np.ones((5, HISTORY_FRAMES)),
            },
            # Degenerate: missing 'other_feature_arr' -> KeyError -> skipped.
            'session_1': {
                'target_feature_arr': np.ones((5, HISTORY_FRAMES)),
            },
        }
        assert list(pipeline.create_category_splits(feature_data, strategy='actual')) == []

    def test_unknown_split_strategy_raises(self):
        """
        An unrecognized ``split_strategy`` falls through both the 'session' and
        'mixed' branches into the terminal ``ValueError``.
        """

        settings = _minimal_category_settings(split_strategy='nonsense')
        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        feature_data = _category_feature_data(HISTORY_FRAMES)
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            list(pipeline.create_category_splits(feature_data, strategy='actual'))

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_null_other_draw_branches_yield_pseudo_classes(self):
        """
        The ``strategy='null_other'`` condition exercises the seeded pseudo-class
        draw helpers: ``draw_pseudo_train`` (balanced 50/50 from the Other pool)
        and ``draw_pseudo_test`` (size- and ratio-matched to the actual test).
        With a generous Other surplus every split yields a valid pseudo-fold, so
        the generator produces at least one ``(X_tr, y_tr, X_te, y_te)`` tuple.
        """

        settings = _minimal_category_settings(split_strategy='mixed', split_num=2)
        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        feature_data = _category_feature_data(HISTORY_FRAMES, n_target=12, n_other=120)
        folds = list(pipeline.create_category_splits(feature_data, strategy='null_other'))
        assert len(folds) >= 1
        X_tr, y_tr, _X_te, _y_te = folds[0]
        # Pseudo-train is balanced 50/50 across the two pseudo-classes.
        assert set(np.unique(y_tr)).issubset({0, 1})
        assert X_tr.shape[1] == HISTORY_FRAMES

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_null_other_starved_other_pool_skips_split(self):
        """
        When the Other pool is too small to source a size-matched pseudo-train
        (its per-split count is below ``2 * n_tr_limit``), ``draw_pseudo_train``
        returns ``(None, None)`` and the per-split ``if X_tr_A is None: continue``
        skips that fold. With every split starved this way the ``null_other``
        generator yields nothing.
        """

        settings = _minimal_category_settings(split_strategy='mixed', split_num=2)
        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        # A near-balanced, small Other pool: after the stratified split the
        # train-Other count cannot reach 2 * n_tr_limit, so every pseudo-train
        # draw bails out.
        feature_data = _category_feature_data(HISTORY_FRAMES, n_target=8, n_other=9)
        folds = list(pipeline.create_category_splits(feature_data, strategy='null_other'))
        assert folds == []


class TestRunModelingCategoryHandlers:
    """Fit-error / metric-except handlers in ``_run_modeling_category`` (sklearn)."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    def test_fit_error_leaves_all_nan(self, monkeypatch):
        """
        Monkeypatching ``LogisticRegressionCV`` so every ``.fit`` raises drives
        the per-fold ``except`` handler (which prints and moves on): no metric is
        written, so every pre-filled ``np.nan`` slot survives and the headline
        mean-AUC computation takes its all-NaN guarded branch without raising
        under ``filterwarnings=error``.
        """

        settings = _minimal_category_settings(split_strategy='mixed', split_num=2)
        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        feature_data = _category_feature_data(HISTORY_FRAMES)
        basis = np.eye(HISTORY_FRAMES, 3, dtype=float)

        class _ExplodingLR:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced logistic fit failure")

        monkeypatch.setattr(category_module, 'LogisticRegressionCV', _ExplodingLR)

        fn, res = pipeline._run_modeling_category('self.speed', feature_data, basis)
        assert fn == 'self.speed'
        assert not np.isfinite(res['actual']['auc']).any()
        assert not np.isfinite(res['null']['auc']).any()

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_ece_except_path_swallows_failure(self, monkeypatch):
        """
        With the sklearn fits otherwise succeeding, forcing
        ``expected_calibration_error`` to raise exercises the ECE ``try/except``
        ``pass`` clause: the ECE slot is never written (stays NaN) while the
        remaining metrics (AUC, log-loss, MCC, ...) are still recorded for the
        fitted folds.
        """

        settings = _minimal_category_settings(split_strategy='mixed', split_num=2)
        pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
        feature_data = _category_feature_data(HISTORY_FRAMES)
        basis = np.eye(HISTORY_FRAMES, 3, dtype=float)

        def _ece_boom(*a, **k):
            raise RuntimeError("forced ece failure")

        monkeypatch.setattr(category_module, 'expected_calibration_error', _ece_boom)

        fn, res = pipeline._run_modeling_category('self.speed', feature_data, basis)
        assert fn == 'self.speed'
        # ECE never recorded (its except swallowed every write), but AUC did.
        assert not np.isfinite(res['actual']['ece']).any()
        assert np.isfinite(res['actual']['auc']).any()
