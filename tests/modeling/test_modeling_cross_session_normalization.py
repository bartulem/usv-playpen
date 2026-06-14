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

import numpy as np
import polars as pls
import pytest

from usv_playpen.modeling.modeling_cross_session_normalization import (
    zscore_different_sessions_together,
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
