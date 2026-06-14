"""
@author: bartulem
Unit tests for ``FeatureZoo`` display-label resolution in
``usv_playpen.analyses.compute_behavioral_features``.

``FeatureZoo.feature_display_names`` is the single source of truth mapping
generic (sex-neutral) behavioral-feature keys to human-readable labels, and
``FeatureZoo.resolve_feature_label`` fills in the per-cohort sexes. The
modeling figures consume this so labels stay consistent and
cohort-correct (male- vs female-target) without per-figure override dicts.

The tests pin: the per-mouse ``self.``/``other.`` sex prefixing, the
dyadic forward/reverse angle phrasing, the derivative suffix, the
no-sex and unknown-key fallbacks, and — crucially — that EVERY kinematic
base feature in ``feature_boundaries`` has a display label (so no feature
silently falls back to its raw key).
"""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")

from usv_playpen.analyses.compute_behavioral_features import FeatureZoo


class TestResolveFeatureLabel:

    def test_egocentric_prefixes_get_sex_word(self):
        """``self.<x>`` / ``other.<x>`` are prefixed with the target /
        predictor sex word."""

        assert FeatureZoo.resolve_feature_label('self.speed', 'male', 'female') == 'male speed'
        assert FeatureZoo.resolve_feature_label('other.speed', 'male', 'female') == 'female speed'
        assert FeatureZoo.resolve_feature_label('self.neck_elevation', 'female', 'male') == 'female neck elevation'

    def test_head_angle_relabelling(self):
        """``allo_roll``/``allo_pitch``/``allo_yaw``/``ego_yaw`` render as
        head roll/pitch/yaw; ego_yaw and allo_yaw share 'head yaw'."""

        assert FeatureZoo.resolve_feature_label('self.allo_roll', 'male') == 'male head roll'
        assert FeatureZoo.resolve_feature_label('self.allo_pitch', 'male') == 'male head pitch'
        assert FeatureZoo.resolve_feature_label('self.allo_yaw', 'male') == 'male head yaw'
        assert FeatureZoo.resolve_feature_label('self.ego_yaw', 'male') == 'male head yaw'

    def test_dyadic_distance_is_sex_neutral(self):
        """Dyadic distances carry no sex word."""

        assert FeatureZoo.resolve_feature_label('nose-nose', 'male', 'female') == 'nose-nose distance'
        assert FeatureZoo.resolve_feature_label('nose-TTI', 'male', 'female') == 'nose-TTI distance'

    def test_dyadic_angle_forward_and_reverse(self):
        """Forward angle is ``{self}-partner``; reverse is ``partner-{self}``;
        the ``-TTI`` variant shares wording with the ``-nose`` variant."""

        assert FeatureZoo.resolve_feature_label('allo_yaw-nose', 'male', 'female') == 'male-partner yaw'
        assert FeatureZoo.resolve_feature_label('allo_yaw-TTI', 'male', 'female') == 'male-partner yaw'
        assert FeatureZoo.resolve_feature_label('nose-allo_yaw', 'male', 'female') == 'partner-male yaw'
        assert FeatureZoo.resolve_feature_label('allo_pitch-nose', 'female', 'male') == 'female-partner pitch'

    def test_sei_engagement(self):
        """SEI engagement features keep the literal 'SEI' tag."""

        assert FeatureZoo.resolve_feature_label('orofacial-sei') == 'orofacial SEI'
        assert FeatureZoo.resolve_feature_label('anogenital-sei') == 'anogenital SEI'

    def test_derivative_suffix_appended(self):
        """A ``_1st_der`` / ``_2nd_der`` suffix is re-appended after the
        base label is resolved (prefix + sex still applied)."""

        assert FeatureZoo.resolve_feature_label('self.speed_1st_der', 'male') == 'male speed (1st derivative)'
        assert FeatureZoo.resolve_feature_label('nose-nose_2nd_der') == 'nose-nose distance (2nd derivative)'

    def test_missing_sex_falls_back_to_role_word(self):
        """With no sexes supplied, the literal 'self'/'other'/'partner'
        wording is used."""

        assert FeatureZoo.resolve_feature_label('self.speed') == 'self speed'
        assert FeatureZoo.resolve_feature_label('other.speed') == 'other speed'
        assert FeatureZoo.resolve_feature_label('allo_yaw-nose') == 'self-partner yaw'

    def test_unknown_key_returns_verbatim(self):
        """A key outside the zoo (e.g. a vocal predictor) is returned
        unchanged so the caller never crashes."""

        assert FeatureZoo.resolve_feature_label('usv_rate', 'male', 'female') == 'usv_rate'
        assert FeatureZoo.resolve_feature_label('self.not_a_feature', 'male') == 'self.not_a_feature'


class TestFeatureDisplayNamesCompleteness:

    def test_every_boundary_base_has_a_display_label(self):
        """Every kinematic base feature in ``feature_boundaries`` (with the
        derivative suffix stripped) must have a ``feature_display_names``
        entry, so no behavioral feature ever renders as its raw key."""

        bases = []
        for key in FeatureZoo.feature_boundaries:
            base = re.sub(r'_(1st|2nd)_der$', '', key)
            if base not in bases:
                bases.append(base)
        missing = [b for b in bases if b not in FeatureZoo.feature_display_names]
        assert missing == [], f"feature_display_names missing labels for: {missing}"
