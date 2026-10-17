"""
@author: bartulem
Unit tests for ``usv_playpen.modeling.modeling_cross_session_normalization``
— the pooled (across-session) z-scoring of behavioural features and the
theoretical-bounds capping helper.

The function pools every session's copy of a feature, computes a single
global mean/std, and z-scores in place. The invariants tested here are:
the pooled result has ~zero mean and ~unit std across the concatenation
(not per session); out-of-bounds values are nulled before pooling; the
plain-abs branch folds the sign; and the smooth-abs branch applies
``sqrt(x^2 + eps^2)`` so the kink at zero is rounded.
"""

from __future__ import annotations

import importlib.resources
import json

import numpy as np
import polars as pls
import pytest

from usv_playpen.modeling.modeling_cross_session_normalization import (
    zscore_different_sessions_together,
)


def test_shipped_settings_expose_abs_features_list():
    """The five modeling pipelines read the plain-abs feature list from
    ``modeling_settings.json['kinematic_features']['abs_features']`` rather than
    hard-coding it, so the shipped block must carry a non-empty list of string
    feature suffixes (its absence would raise ``KeyError`` at fit time)."""

    settings_traversable = importlib.resources.files('usv_playpen').joinpath(
        '_parameter_settings', 'modeling_settings.json'
    )
    kinematic_features = json.loads(settings_traversable.read_text())['kinematic_features']
    abs_features = kinematic_features['abs_features']
    assert isinstance(abs_features, list) and len(abs_features) > 0
    assert all(isinstance(feature, str) for feature in abs_features)
    # `smooth_abs_features` must remain a sibling key (the two folds are distinct).
    assert 'smooth_abs_features' in kinematic_features


def test_shipped_settings_expose_binary_decision_threshold():
    """Every binary label is assigned by thresholding the predicted probability
    at ``diagnostics.binary_decision_threshold`` (with a single unified ``>=``
    operator across all selectors and pipelines), so the shipped block must
    carry the threshold as a probability in ``[0, 1]``; its absence would raise
    ``KeyError`` where the prediction is thresholded."""

    settings_traversable = importlib.resources.files('usv_playpen').joinpath(
        '_parameter_settings', 'modeling_settings.json'
    )
    threshold = json.loads(settings_traversable.read_text())['diagnostics']['binary_decision_threshold']
    assert isinstance(threshold, (int, float))
    assert 0.0 <= float(threshold) <= 1.0


def test_shipped_settings_expose_borderline_tunables():
    """The former hard-coded modeling defaults are now settings-driven, so the
    shipped blocks must expose the session-split initial tolerance
    (``model_validation``) and the forward-selection significance level
    (``model_params``) and the ECE histogram bin count (``diagnostics``) with
    sane types / ranges."""

    settings = json.loads(
        importlib.resources.files('usv_playpen').joinpath(
            '_parameter_settings', 'modeling_settings.json'
        ).read_text()
    )
    model_params = settings['model_params']
    model_validation = settings['model_validation']
    diagnostics = settings['diagnostics']
    assert 0.0 <= float(model_validation['session_split_initial_tolerance']) <= 1.0
    assert 0.0 < float(model_params['selection_p_val']) < 1.0
    assert int(diagnostics['ece_n_bins']) >= 2


def test_modeling_constants_resolve_from_settings():
    """The module-level constants that replaced the bare literals are read from
    the settings file via ``resolve_modeling_setting``, so they must equal the
    shipped values (guards the wiring end-to-end)."""

    import warnings

    from usv_playpen.os_utils import resolve_modeling_setting
    # These modules pull the JAX / optax stack, whose import emits a one-time
    # DeprecationWarning that the suite's warnings-as-errors would promote.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from usv_playpen.modeling import jax_neural_network_cnn, model_selection, modeling_utils

    assert model_selection._SELECTION_P_VAL == resolve_modeling_setting('model_params', 'selection_p_val')
    assert model_selection._ECE_N_BINS == resolve_modeling_setting('diagnostics', 'ece_n_bins')
    assert modeling_utils._ECE_N_BINS == resolve_modeling_setting('diagnostics', 'ece_n_bins')
    assert jax_neural_network_cnn._SESSION_SPLIT_INITIAL_TOLERANCE == resolve_modeling_setting(
        'model_validation', 'session_split_initial_tolerance'
    )


# zscore_different_sessions_together


class TestZscorePooled:

    def test_pooled_zscore_has_zero_mean_unit_std(self):
        """The pooled (concatenated) feature across both sessions has mean
        ~0 and std ~1 after z-scoring; the statistics are global, not
        per-session."""

        rng = np.random.default_rng(0)
        s1 = rng.normal(10.0, 2.0, size=200)
        s2 = rng.normal(10.0, 2.0, size=300)
        data = {
            'sess1': pls.DataFrame({'self.speed': s1}),
            'sess2': pls.DataFrame({'self.speed': s2}),
        }
        out = zscore_different_sessions_together(
            data,
            feature_lst=['speed'],
            feature_bounds={'speed': (-100.0, 100.0)},
        )
        pooled = np.concatenate([
            out['sess1']['self.speed'].to_numpy(),
            out['sess2']['self.speed'].to_numpy(),
        ])
        # ddof=1 sample std matches polars' default std used internally.
        assert np.nanmean(pooled) == pytest.approx(0.0, abs=1e-5)
        assert np.nanstd(pooled, ddof=1) == pytest.approx(1.0, abs=1e-3)

    def test_returns_same_object_mutated_in_place(self):
        """The helper mutates and returns the same dict object (caller
        compatibility contract)."""

        data = {'sess1': pls.DataFrame({'self.speed': [1.0, 2.0, 3.0]})}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (-10.0, 10.0)},
        )
        assert out is data

    def test_bounds_clip_nulls_outliers_before_pooling(self):
        """A value outside ``feature_bounds`` is nulled before the global
        statistics are computed, so it cannot poison the pooled mean/std;
        the nulled cell stays null after z-scoring."""

        data = {
            'sess1': pls.DataFrame({'self.speed': [0.0, 1.0, 2.0, 1000.0]}),
        }
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (-10.0, 10.0)},
        )
        col = out['sess1']['self.speed'].to_list()
        # The out-of-bounds 1000.0 is nulled out.
        assert col[3] is None
        # Remaining finite entries are z-scored about the (small) pooled mean.
        finite = np.array([c for c in col if c is not None])
        assert np.mean(finite) == pytest.approx(0.0, abs=1e-6)

    def test_plain_abs_branch_folds_sign(self):
        """A feature in ``abs_features`` is folded to ``|x|`` before
        z-scoring, so two opposite-sign inputs map to the same z-score."""

        data = {
            'sess1': pls.DataFrame({'self.allo_yaw': [-3.0, -1.0, 1.0, 3.0]}),
        }
        out = zscore_different_sessions_together(
            data,
            feature_lst=['allo_yaw'],
            feature_bounds={'allo_yaw': (-180.0, 180.0)},
            abs_features=['allo_yaw'],
        )
        z = out['sess1']['self.allo_yaw'].to_numpy()
        # |−3|==|3| and |−1|==|1| -> symmetric z-scores.
        assert z[0] == pytest.approx(z[3])
        assert z[1] == pytest.approx(z[2])

    def test_smooth_abs_branch_applies_sqrt_x2_eps2(self):
        """A feature in ``smooth_abs_features`` is mapped through
        ``sqrt(x^2 + eps^2)`` (pre-z-score). With a single session we can
        invert the z-score to recover the transformed magnitudes and check
        the smooth fold against the analytic formula."""

        eps = 1.0
        x = np.array([-2.0, 0.0, 2.0])
        data = {'sess1': pls.DataFrame({'self.ego_yaw': x})}
        out = zscore_different_sessions_together(
            data,
            feature_lst=['ego_yaw'],
            feature_bounds={'ego_yaw': (-180.0, 180.0)},
            smooth_abs_features={'ego_yaw': eps},
        )
        z = out['sess1']['self.ego_yaw'].to_numpy()
        transformed = np.sqrt(x ** 2 + eps ** 2)
        expected_z = (transformed - transformed.mean()) / transformed.std(ddof=1)
        np.testing.assert_allclose(z, expected_z, atol=1e-5)
        # The smooth fold has no hard corner: the zero input maps to eps,
        # strictly above 0 (unlike plain |0| == 0).
        assert transformed[1] == pytest.approx(eps)

    def test_unprefixed_and_multi_prefix_columns_match_by_last_suffix(self):
        """The feature suffix is the text after the LAST dot: a bare column
        (``speed``, no prefix) and a doubly-prefixed one (``a.b.speed``) are
        both clipped, pooled and z-scored. Values [1, 3] and [5, 7] pool to
        mean 4 and sample std sqrt(20/3); the out-of-bounds 99 is nulled."""

        data = {
            'sess1': pls.DataFrame({'speed': [1.0, 3.0, 99.0]}),
            'sess2': pls.DataFrame({'a.b.speed': [5.0, 7.0, 99.0]}),
        }
        stats = {}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 10.0)}, stats_out=stats,
        )
        std = np.sqrt(20.0 / 3.0)
        assert stats == {'speed': {'mean': pytest.approx(4.0), 'std': pytest.approx(std, rel=1e-6)}}
        s1 = out['sess1']['speed'].to_list()
        s2 = out['sess2']['a.b.speed'].to_list()
        assert s1[:2] == pytest.approx([-3.0 / std, -1.0 / std], rel=1e-6)
        assert s1[2] is None
        assert s2[:2] == pytest.approx([1.0 / std, 3.0 / std], rel=1e-6)
        assert s2[2] is None

    def test_non_feature_column_before_a_feature_column_is_skipped_not_fatal(self):
        """A column that is not a listed feature (``time``, placed FIRST) is
        skipped and left untouched; the feature column after it is still
        clipped (99 nulled) and z-scored ([0, 2] -> [-1/sqrt(2), 1/sqrt(2)])."""

        data = {'sess1': pls.DataFrame({'time': [10.0, 20.0, 30.0], 'self.speed': [0.0, 2.0, 99.0]})}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 10.0)},
        )
        assert out['sess1']['time'].to_list() == [10.0, 20.0, 30.0]
        z = out['sess1']['self.speed'].to_list()
        assert z[:2] == pytest.approx([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], rel=1e-6)
        assert z[2] is None

    def test_smooth_abs_uses_eps_squared(self):
        """With eps = 2 the smooth fold adds eps^2 = 4 (not eps/eps = 1): inputs
        [0, 0, 0] (constant, std 0 -> std 1) map to sqrt(4) = 2, so the reported
        pooled mean is exactly 2."""

        data = {'sess1': pls.DataFrame({'self.ego_yaw': [0.0, 0.0, 0.0]})}
        stats = {}
        zscore_different_sessions_together(
            data, feature_lst=['ego_yaw'], feature_bounds={'ego_yaw': (-180.0, 180.0)},
            smooth_abs_features={'ego_yaw': 2.0}, stats_out=stats,
        )
        assert stats['ego_yaw']['mean'] == pytest.approx(2.0)

    def test_bounds_clip_keeps_values_exactly_at_the_bounds(self):
        """The theoretical bounds are inclusive: values exactly at the minimum
        and the maximum are kept, only values strictly outside are nulled.
        [-1, 0, 10, 11] with bounds (0, 10) keeps 0 and 10 -> z = [-1, 1]
        times 1/sqrt(2)."""

        data = {'sess1': pls.DataFrame({'self.speed': [-1.0, 0.0, 10.0, 11.0]})}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 10.0)},
        )
        z = out['sess1']['self.speed'].to_list()
        assert z[0] is None
        assert z[3] is None
        assert z[1:3] == pytest.approx([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], rel=1e-6)

    def test_reference_sessions_set_the_scale_for_every_session(self):
        """With ``reference_sessions`` only the listed sessions shape the pooled
        mean/std (dev [0, 2] -> mean 1, std sqrt(2)); the held-out session
        [100] is z-scored with that scale, (100 - 1) / sqrt(2)."""

        data = {
            'dev': pls.DataFrame({'self.speed': [0.0, 2.0]}),
            'held': pls.DataFrame({'self.speed': [100.0]}),
        }
        stats = {}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 1000.0)},
            stats_out=stats, reference_sessions=['dev'],
        )
        assert stats['speed']['mean'] == pytest.approx(1.0)
        assert stats['speed']['std'] == pytest.approx(np.sqrt(2.0), rel=1e-6)
        assert out['held']['self.speed'].to_list() == pytest.approx([99.0 / np.sqrt(2.0)], rel=1e-6)

    def test_reference_session_missing_from_the_data_raises(self):
        """A listed reference session absent from ``data_dict`` raises a KeyError
        naming the missing session(s), sorted."""

        data = {'dev': pls.DataFrame({'self.speed': [0.0, 2.0]})}
        with pytest.raises(KeyError, match=r"reference session\(s\) \['gone_a', 'gone_b'\] are not in the data"):
            zscore_different_sessions_together(
                data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 10.0)},
                reference_sessions=['gone_b', 'dev', 'gone_a'],
            )

    def test_absent_feature_does_not_stop_later_features(self):
        """A feature listed first but present in no session is skipped; the
        feature listed after it is still pooled and z-scored."""

        data = {'sess1': pls.DataFrame({'self.speed': [0.0, 2.0]})}
        stats = {}
        out = zscore_different_sessions_together(
            data, feature_lst=['absent', 'speed'],
            feature_bounds={'absent': (0.0, 1.0), 'speed': (0.0, 10.0)}, stats_out=stats,
        )
        assert list(stats) == ['speed']
        assert out['sess1']['self.speed'].to_list() == pytest.approx([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], rel=1e-6)

    def test_constant_feature_is_centred_with_unit_scale(self):
        """A constant feature (std 0) keeps its pooled mean and uses std 1, so
        every value z-scores to exactly 0 and the reported stats are
        {'mean': 5, 'std': 1}."""

        data = {'sess1': pls.DataFrame({'self.speed': [5.0, 5.0, 5.0]})}
        stats = {}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 10.0)}, stats_out=stats,
        )
        assert out['sess1']['self.speed'].to_list() == [0.0, 0.0, 0.0]
        assert stats == {'speed': {'mean': 5.0, 'std': 1.0}}

    def test_all_null_feature_reports_zero_mean_unit_std(self):
        """A feature whose every value is nulled (all out of bounds) has no mean
        and no std; the stats fall back to {'mean': 0, 'std': 1} and the column
        stays all-null."""

        data = {'sess1': pls.DataFrame({'self.speed': [50.0, 60.0]})}
        stats = {}
        out = zscore_different_sessions_together(
            data, feature_lst=['speed'], feature_bounds={'speed': (0.0, 10.0)}, stats_out=stats,
        )
        assert stats == {'speed': {'mean': 0.0, 'std': 1.0}}
        assert out['sess1']['self.speed'].to_list() == [None, None]
