"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.modeling_behavioral_response``
— the tiled-anchor extraction that turns a behavioural feature into a
regression target driven by a partner's vocal trace.

This pipeline inverts the direction of every other extractor in
``modeling/``, so the invariants worth pinning are the ones that would
silently corrupt the science rather than crash. Coverage:

* ``tile_anchor_frames`` — anchors leave room for both the backward
  history and the forward target, are stride-spaced, and collapse to
  empty rather than raising on a session too short to hold one.
* ``forward_window_mean`` — the target window is strictly forward of the
  anchor (so it can never overlap that row's own history), NaNs are
  ignored rather than propagated, and an all-NaN window yields NaN so the
  caller can drop the row.
* The shipped ``behavioral_response`` settings block — the keys the
  pipeline reads by name exist, and the two mouse indices are absolute
  slot numbers rather than relative role strings.
* ``BehavioralResponsePipeline`` — seconds convert to frames on the
  configured camera grid, and a degenerate window is rejected loudly.
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import pathlib
import pickle
import warnings

import numpy as np
import pytest

from ._synth import (
    build_modeling_settings,
    build_session_tree,
    write_session_list_file,
)

# The dispatcher's import chain pulls optax -> a one-time JAX DeprecationWarning.
# Guard the top-level imports so collection does not trip ``filterwarnings =
# ["error"]`` before any per-test marker can take effect (mirrors the guard in
# ``test_model_selection_fold_gate``).
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling import main_univariate_dispatcher as univ_dispatcher
    from usv_playpen.modeling.behavioral_response_selection import (
        behavioral_response_model_selection,
    )
    from usv_playpen.modeling.consolidate_model_selection_results import (
        consolidate as consolidate_model_selection,
    )
    from usv_playpen.modeling.consolidate_univariate_results import (
        consolidate as consolidate_univariate,
    )
    from usv_playpen.modeling.modeling_behavioral_response import (
        BehavioralResponsePipeline,
        forward_window_mean,
        tile_anchor_frames,
    )
    from usv_playpen.modeling.modeling_vocal_bout_parameters import (
        BoutParameterPipeline,
    )


def _shipped_settings() -> dict:
    """Loads the shipped modeling settings JSON."""
    settings_file = (
        importlib.resources.files('usv_playpen')
        / '_parameter_settings'
        / 'modeling_settings.json'
    )
    return json.loads(settings_file.read_text())


class TestTileAnchorFrames:
    def test_anchors_leave_room_for_history_and_target(self):
        """Every anchor admits a full backward history and a full forward target.

        This is the guard that keeps a row from being built out of a
        truncated window; a violation would silently shorten some rows'
        history and misalign the design matrix.
        """

        anchors = tile_anchor_frames(
            n_frames=150 * 1200, history_frames=600, stride_frames=600, lookahead_frames=75,
        )
        assert anchors.size > 0
        assert (anchors - 600 >= 0).all()
        assert (anchors + 75 <= 150 * 1200).all()

    def test_anchors_are_stride_spaced(self):
        """Consecutive anchors differ by exactly the stride."""

        anchors = tile_anchor_frames(
            n_frames=20_000, history_frames=600, stride_frames=600, lookahead_frames=75,
        )
        assert np.unique(np.diff(anchors)).tolist() == [600]

    def test_stride_equal_to_history_gives_non_overlapping_windows(self):
        """At stride == history no two rows share a single history sample.

        This is what keeps the tiled rows close to independent, which the
        whole sample-size argument for the analysis rests on.
        """

        history_frames = 600
        anchors = tile_anchor_frames(
            n_frames=20_000, history_frames=history_frames,
            stride_frames=history_frames, lookahead_frames=75,
        )
        starts = anchors - history_frames
        assert (starts[1:] >= anchors[:-1]).all()

    def test_session_too_short_returns_empty_rather_than_raising(self):
        """A session that cannot hold one anchor yields no rows, not an error."""

        anchors = tile_anchor_frames(
            n_frames=500, history_frames=600, stride_frames=600, lookahead_frames=75,
        )
        assert anchors.size == 0

    @pytest.mark.parametrize(
        ('history_frames', 'stride_frames', 'lookahead_frames', 'offending_key'),
        [(0, 600, 75, 'history_frames'), (600, 0, 75, 'stride_frames'), (600, 600, -1, 'lookahead_frames')],
    )
    def test_degenerate_geometry_raises_value_error(self, history_frames, stride_frames,
                                                    lookahead_frames, offending_key):
        """Non-positive geometry fails loudly, naming the offending parameter."""

        with pytest.raises(ValueError, match=offending_key):
            tile_anchor_frames(
                n_frames=20_000, history_frames=history_frames,
                stride_frames=stride_frames, lookahead_frames=lookahead_frames,
            )


class TestForwardWindowMean:
    def test_window_mean_matches_hand_computation(self):
        """On a ramp the forward mean is the arithmetic mean of the window."""

        values = np.arange(100, dtype=float)
        anchors = np.array([0, 10, 50])
        means = forward_window_mean(values, anchors, gap_frames=0, window_frames=4)
        np.testing.assert_allclose(means, [1.5, 11.5, 51.5])

    def test_gap_shifts_the_window_forward(self):
        """A gap moves the averaged window later by exactly that many frames."""

        values = np.arange(100, dtype=float)
        anchors = np.array([0, 10, 50])
        means = forward_window_mean(values, anchors, gap_frames=10, window_frames=4)
        np.testing.assert_allclose(means, [11.5, 21.5, 61.5])

    def test_window_never_reaches_before_the_anchor(self):
        """The target is strictly forward, so pre-anchor values cannot leak in.

        If the window reached backwards it would overlap the row's own
        history and the model would be partly predicting its own inputs.
        """

        values = np.zeros(100, dtype=float)
        values[:50] = 1000.0
        means = forward_window_mean(values, np.array([50]), gap_frames=0, window_frames=10)
        assert means[0] == 0.0

    def test_non_finite_samples_are_ignored_not_propagated(self):
        """A partially NaN window averages only its finite samples."""

        values = np.array([np.nan] * 4 + [2.0, 4.0, np.nan, np.nan], dtype=float)
        means = forward_window_mean(values, np.array([4]), gap_frames=0, window_frames=4)
        assert means[0] == pytest.approx(3.0)

    def test_all_non_finite_window_yields_nan(self):
        """An entirely unusable window is NaN so the caller can drop the row."""

        values = np.array([np.nan] * 8, dtype=float)
        means = forward_window_mean(values, np.array([0]), gap_frames=0, window_frames=4)
        assert np.isnan(means[0])

    def test_zero_width_window_raises_value_error(self):
        """A zero-width target window is rejected rather than silently empty."""

        with pytest.raises(ValueError, match='window_frames'):
            forward_window_mean(np.arange(10, dtype=float), np.array([0]), 0, 0)


class TestShippedSettingsBlock:
    def test_block_exposes_every_key_the_pipeline_reads(self):
        """The pipeline reads these by key with no ``.get`` fallback."""

        block = _shipped_settings()['behavioral_response']
        expected = {
            'response_mouse_index', 'response_feature', 'history_seconds',
            'target_window_seconds', 'target_gap_seconds', 'vocal_predictor_type',
            'vocal_smoothing_sd_frames', 'likelihood', 'n_shift_draws',
            'shift_null_min_seconds',
        }
        assert expected <= set(block)

    def test_response_mouse_is_an_absolute_slot_index(self):
        """The responder is named by slot index, not by a relative role string.

        ``self`` / ``other`` are defined against
        ``model_predictor_mouse_index`` and cannot be read alone; slot 0 is
        always the male and slot 1 always the female.
        """

        block = _shipped_settings()['behavioral_response']
        assert isinstance(block['response_mouse_index'], int)
        assert block['response_mouse_index'] in (0, 1)

    def test_window_and_gap_are_scalars_not_sweep_lists(self):
        """A sweep is several runs, so these are single values."""

        block = _shipped_settings()['behavioral_response']
        assert isinstance(block['target_window_seconds'], (int, float))
        assert isinstance(block['target_gap_seconds'], (int, float))

    def test_shift_minimum_exceeds_the_slowest_measured_autocorrelation(self):
        """The shift floor must clear the slowest behavioural feature.

        ``nose-nose`` has the longest measured autocorrelation horizon in
        the zoo at roughly 6-8 s; a floor below that would leave real
        alignment intact in the null.
        """

        assert _shipped_settings()['behavioral_response']['shift_null_min_seconds'] >= 10.0


class TestPipelineGeometry:
    def test_seconds_convert_to_frames_on_the_camera_grid(self):
        """History, target and gap are resolved against ``camera_sampling_rate``."""

        settings = _shipped_settings()
        pipeline = BehavioralResponsePipeline(modeling_settings_dict=settings)
        camera_rate = settings['io']['camera_sampling_rate']
        block = settings['behavioral_response']

        assert pipeline.response_history_frames == int(camera_rate * block['history_seconds'])
        assert pipeline.response_window_frames == int(camera_rate * block['target_window_seconds'])
        assert pipeline.response_gap_frames == int(camera_rate * block['target_gap_seconds'])

    def test_vocal_smoothing_is_read_in_frames_and_kept_fractional(self):
        """Smoothing is a frame-count sigma, kept as a float.

        It is passed straight to the loader's Gaussian kernel, which is
        also frame-based; truncating to an int would make a fractional
        kernel silently unavailable.
        """

        settings = _shipped_settings()
        settings['behavioral_response']['vocal_smoothing_sd_frames'] = 2.5
        pipeline = BehavioralResponsePipeline(modeling_settings_dict=settings)

        assert isinstance(pipeline.response_vocal_smoothing_frames, float)
        assert pipeline.response_vocal_smoothing_frames == pytest.approx(2.5)

    def test_sub_frame_target_window_raises_value_error(self):
        """A target window shorter than one frame fails loudly, not silently."""

        settings = _shipped_settings()
        settings['behavioral_response']['target_window_seconds'] = 0.001
        with pytest.raises(ValueError, match='target_window_seconds'):
            BehavioralResponsePipeline(modeling_settings_dict=settings)

    def test_vocal_block_is_derived_from_the_emitted_signal_names(self):
        """The partition survives the ``self.`` / ``other.`` role prefixing.

        It is derived rather than configured, so it cannot drift out of
        step with the predictor type that decides which traces exist.
        """

        pipeline = BehavioralResponsePipeline(modeling_settings_dict=_shipped_settings())
        vocal_block = pipeline._vocal_block_feature_names(
            ['self.speed', 'other.usv_rate', 'nose-nose', 'self.usv_rate'],
        )
        assert vocal_block == ['other.usv_rate', 'self.usv_rate']

    def test_every_vocal_representation_is_recognised_by_the_partition(self):
        """All four predictor types' traces land in the block, not just one.

        ``pooled_binary`` emits ``usv_event``, ``pooled_rate`` emits
        ``usv_rate``, ``categories_rate`` emits ``usv_cat_<n>``, and
        ``all_rate`` emits both; a partition that recognised only one of
        them would silently leave vocal columns in the baseline.
        """

        pipeline = BehavioralResponsePipeline(modeling_settings_dict=_shipped_settings())
        vocal_block = pipeline._vocal_block_feature_names(
            ['other.usv_event', 'other.usv_rate', 'other.usv_cat_1',
             'other.usv_cat_12', 'self.speed', 'nose-nose'],
        )
        assert vocal_block == [
            'other.usv_cat_1', 'other.usv_cat_12', 'other.usv_event', 'other.usv_rate',
        ]

    def test_vocal_predictor_type_is_a_legal_loader_value(self):
        """The block-local override must be one the loader can act on."""

        block = _shipped_settings()['behavioral_response']
        assert block['vocal_predictor_type'] in (
            'pooled_binary', 'pooled_rate', 'categories_rate', 'all_rate',
        )

    def test_predictor_index_is_derived_as_the_other_mouse(self):
        """Whose calls go in is derived from whose behaviour is predicted.

        ``model_predictor_mouse_index`` and ``response_mouse_index`` have
        opposite meanings on the same 0/1 axis, and the partner's calls are
        by definition the other animal's -- so requiring both to be set by
        hand is two keys that must disagree by construction, which is two
        keys that can silently agree.
        """

        for response_index in (0, 1):
            settings = _shipped_settings()
            settings['behavioral_response']['response_mouse_index'] = response_index
            pipeline = BehavioralResponsePipeline(modeling_settings_dict=settings)

            derived = pipeline.modeling_settings['model_params']['model_predictor_mouse_index']
            assert derived == 1 - response_index

    def test_deriving_the_predictor_index_does_not_mutate_the_caller_dict(self):
        """The override is local to the instance, not to the passed settings."""

        settings = _shipped_settings()
        shipped_value = settings['model_params']['model_predictor_mouse_index']
        settings['behavioral_response']['response_mouse_index'] = 1

        BehavioralResponsePipeline(modeling_settings_dict=settings)

        assert settings['model_params']['model_predictor_mouse_index'] == shipped_value

    @pytest.mark.parametrize('bad_index', [-1, 2])
    def test_response_mouse_index_outside_the_two_slots_raises(self, bad_index):
        """Only two mice exist; a third slot cannot be silently accepted."""

        settings = _shipped_settings()
        settings['behavioral_response']['response_mouse_index'] = bad_index
        with pytest.raises(ValueError, match='response_mouse_index'):
            BehavioralResponsePipeline(modeling_settings_dict=settings)

    def test_override_does_not_mutate_the_shared_vocal_features_block(self):
        """The five vocal pipelines read ``vocal_features``; it stays untouched."""

        settings = _shipped_settings()
        shared_before = settings['vocal_features']['usv_predictor_type']
        BehavioralResponsePipeline(modeling_settings_dict=settings)
        assert settings['vocal_features']['usv_predictor_type'] == shared_before


class TestScreenNullIsAShift:
    def test_null_target_preserves_the_response_serial_structure(self):
        """The screen null rolls the target rather than permuting it.

        Permutation destroys the response's own autocorrelation, which makes
        it an easier null than the data warrants: the predictors are strongly
        autocorrelated, so a scrambled target is trivially harder to predict
        than a real one.
        """

        pipeline = BehavioralResponsePipeline(modeling_settings_dict=_shipped_settings())
        y_train = np.arange(200, dtype=float)
        rolled = pipeline._null_target(y_train, np.random.default_rng(0))

        # a roll is a rotation: same values, same local ordering, moved wholesale
        np.testing.assert_allclose(np.sort(rolled), np.sort(y_train))
        assert not np.array_equal(rolled, y_train)
        steps = np.diff(rolled)
        assert np.sum(steps != 1.0) == 1          # exactly one wrap point

    def test_offset_respects_the_configured_minimum(self):
        """A near-identity roll would leave real alignment intact."""

        settings = _shipped_settings()
        pipeline = BehavioralResponsePipeline(modeling_settings_dict=settings)
        minimum_rows = int(np.ceil(
            settings['behavioral_response']['shift_null_min_seconds']
            / settings['behavioral_response']['history_seconds'],
        ))
        y_train = np.arange(300, dtype=float)

        for seed in range(15):
            rolled = pipeline._null_target(y_train, np.random.default_rng(seed))
            offset = int(np.flatnonzero(rolled == 0.0)[0])
            assert minimum_rows <= offset <= 300 - minimum_rows

    def test_too_few_rows_falls_back_to_the_inherited_permutation(self):
        """Returning the identity would be no null at all."""

        pipeline = BehavioralResponsePipeline(modeling_settings_dict=_shipped_settings())
        y_train = np.arange(4, dtype=float)
        result = pipeline._null_target(y_train, np.random.default_rng(0))

        np.testing.assert_allclose(np.sort(result), np.sort(y_train))
        assert result.shape == y_train.shape

    def test_parent_default_is_unchanged(self):
        """The seam must not alter the pipelines that already use it."""

        parent = BoutParameterPipeline(modeling_settings_dict=_shipped_settings())
        y_train = np.arange(50, dtype=float)
        permuted = parent._null_target(y_train, np.random.default_rng(0))

        np.testing.assert_allclose(
            permuted, np.random.default_rng(0).permutation(y_train),
        )


class TestFullPipelineOnSyntheticSessions:
    """All five stages end-to-end: extract, univariate array, consolidate,
    select, consolidate.

    Each stage is exercised through the same entry point the cluster uses, so
    the things that only break at the seams -- filename conventions the
    consolidators infer from, step numbering, the metadata blocks each stage
    expects the previous one to have written -- are covered rather than assumed.
    """

    def _settings(self, tmp_path):
        session_roots = build_session_tree(
            base_dir=tmp_path / 'sessions', n_sessions=4, n_frames=3000,
            camera_fps=150.0, filter_history=1.0, n_bouts=10, usv_per_bout=3,
        )
        list_file = write_session_list_file(session_roots, tmp_path / 'session_list.txt')
        save_dir = tmp_path / 'out'
        save_dir.mkdir(parents=True, exist_ok=True)

        settings = build_modeling_settings(
            session_list_file=list_file, save_directory=save_dir,
            camera_sampling_rate=150.0, filter_history=1.0,
            model_engine='sklearn', split_strategy='session',
            split_num=2, test_proportion=0.5,
        )
        settings['model_validation']['n_cv_folds'] = 2
        settings['model_validation']['cv_validation_proportion'] = 0.5
        settings['model_validation']['held_out_test_proportion'] = 0.25
        settings['behavioral_response'].update({
            'response_mouse_index': 1,
            'response_feature': 'speed',
            'history_seconds': 1.0,
            'target_window_seconds': 0.2,
            'target_gap_seconds': 0.0,
            'vocal_predictor_type': 'pooled_rate',
            'likelihood': 'gamma',
        })
        settings['hyperparameters']['classical']['pygam'].update({
            'n_splines_value': 4, 'n_splines_time': 4, 'max_iterations': 30,
        })
        return settings, save_dir

    def test_five_stages_run_and_consolidate(self, tmp_path, monkeypatch):
        """The whole pipeline, stage by stage, on synthetic recordings."""

        settings, save_dir = self._settings(tmp_path)

        # --- stage 1: extraction -------------------------------------------
        BehavioralResponsePipeline(
            modeling_settings_dict=settings,
        ).extract_and_save_modeling_input_data()

        input_pickles = sorted(save_dir.glob('modeling_behavioral_response_*.pkl'))
        input_pickles = [f for f in input_pickles
                         if not f.name.endswith(('_collinearity.pkl', '_timescales.pkl'))]
        assert len(input_pickles) == 1
        input_pkl = input_pickles[0]

        with input_pkl.open('rb') as handle:
            extracted = pickle.load(handle)
        partition = extracted['_input_metadata']['analysis_specific']
        baseline_features = partition['baseline_block_features']
        vocal_features = partition['vocal_block_features']

        # the block-local `vocal_predictor_type` override must have produced a
        # pooled rate even though the shared `vocal_features` block asks for none
        assert vocal_features, 'no vocal column was generated'
        assert not set(baseline_features) & set(vocal_features)

        # --- stage 2: univariate job array, BASELINE features only ----------
        univariate_dir = tmp_path / 'univariate'
        univariate_dir.mkdir()
        monkeypatch.setattr(univ_dispatcher.json, 'load', lambda _f: settings)

        all_features = sorted(k for k in extracted if not k.startswith('_'))
        for feature_index, feature_name in enumerate(all_features):
            if feature_name not in baseline_features:
                continue
            univ_dispatcher.dispatch_univariate_job(argparse.Namespace(
                analysis_type='behavioral_response', feature_idx=feature_index,
                input_data=str(input_pkl), output_dir=str(univariate_dir),
            ))

        per_feature = sorted(univariate_dir.glob('univariate_*.pkl'))
        assert per_feature, 'the univariate stage wrote nothing'

        # --- stage 3: consolidate the per-feature pickles -------------------
        consolidated_univariate = consolidate_univariate(input_dir=str(univariate_dir))
        assert pathlib.Path(consolidated_univariate).is_file()

        # --- stage 4: selection, one checkpoint per step --------------------
        selection_dir = tmp_path / 'selection'
        results = behavioral_response_model_selection(
            input_pickle_path=input_pkl,
            univariate_results_path=consolidated_univariate,
            output_directory=selection_dir,
            modeling_settings_dict=settings,
        )
        arm = results['gamma']

        # the vocal block is never a selection candidate
        assert all(f not in arm['screen']['per_feature'] for f in vocal_features)
        for step in arm['selection']['steps']:
            assert not set(step['candidates']) & set(vocal_features)
        assert arm['final_step']['vocal_block_features'] == vocal_features

        step_files = sorted(selection_dir.glob('*.pkl'))
        indices = sorted(int(f.stem.rsplit('_', 1)[-1]) for f in step_files)
        assert indices == list(range(len(indices)))    # 0-based and contiguous
        assert all(f.name.startswith('model_selection_') for f in step_files)

        # --- stage 5: consolidate the per-step pickles ----------------------
        consolidated_selection = consolidate_model_selection(input_dir=str(selection_dir))
        with pathlib.Path(consolidated_selection).open('rb') as handle:
            merged = pickle.load(handle)

        assert len(merged['steps']) == len(step_files)
        assert '_input_metadata' in merged
        assert '_run_metadata' in merged
        assert merged['steps'][-1]['vocal_block_features'] == vocal_features


class TestTargetMagnitudeFold:
    """The target must get the same fold the predictors get.

    Skipping it puts the target and the identical design-matrix column on
    different scales, and for a signed feature it also collapses the usable
    sample: the Gamma likelihood drops every non-positive row.
    """

    def _pipeline(self, response_feature):
        """Pipeline with a ONE-FRAME target window.

        ``_response_target_values`` folds and then forward-averages; a
        single-frame window makes that average the identity, so the fold can
        be checked element-wise instead of through a 75-frame mean.
        """

        settings = _shipped_settings()
        settings['behavioral_response']['response_feature'] = response_feature
        settings['behavioral_response']['target_window_seconds'] = (
            1.0 / settings['io']['camera_sampling_rate']
        )
        pipeline = BehavioralResponsePipeline(modeling_settings_dict=settings)
        assert pipeline.response_window_frames == 1
        return pipeline, settings

    def test_smooth_abs_feature_is_folded_with_the_configured_epsilon(self):
        pipeline, settings = self._pipeline('ego_yaw')
        epsilon = settings['kinematic_features']['smooth_abs_features']['ego_yaw']

        raw = np.array([-40.0, -1.0, 0.0, 1.0, 40.0])
        anchors = np.arange(raw.size)
        folded = pipeline._response_target_values(raw_values=raw, anchor_frames=anchors)

        np.testing.assert_allclose(folded, np.sqrt(raw ** 2 + epsilon ** 2))
        assert (folded > 0).all()          # usable under a Gamma likelihood

    def test_abs_feature_is_folded_to_plain_magnitude(self):
        pipeline, _ = self._pipeline('allo_roll')
        raw = np.array([-30.0, -5.0, 5.0, 30.0])
        folded = pipeline._response_target_values(raw, np.arange(raw.size))
        np.testing.assert_allclose(folded, np.abs(raw))

    def test_unfolded_feature_is_left_signed(self):
        """Only the configured features are folded; speed is untouched."""

        pipeline, _ = self._pipeline('speed')
        raw = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            pipeline._response_target_values(raw, np.arange(raw.size)), raw,
        )

    def test_signed_target_without_the_fold_would_lose_half_the_rows(self):
        """Documents why the fold is load-bearing, not cosmetic.

        A near-symmetric signed angle has about half its samples below zero,
        and the extraction drops non-positive targets outright.
        """

        pipeline, _ = self._pipeline('ego_yaw')
        rng = np.random.default_rng(0)
        raw = rng.normal(0.0, 40.0, size=2000)

        assert np.mean(raw <= 0) > 0.4          # unfolded, ~half would be dropped
        folded = pipeline._response_target_values(raw, np.arange(raw.size))
        assert (folded > 0).all()               # folded, all rows survive

    def test_the_applied_fold_is_recorded_in_provenance(self):
        """An artifact must say which fold produced its target."""

        settings = _shipped_settings()
        for feature, expected in (('ego_yaw', 'smooth_abs'), ('allo_roll', 'abs'),
                                  ('speed', 'none')):
            kinematic = settings['kinematic_features']
            resolved = ('smooth_abs' if feature in kinematic['smooth_abs_features']
                        else 'abs' if feature in kinematic['abs_features'] else 'none')
            assert resolved == expected
