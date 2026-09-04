"""
@author: bartulem
Module for modeling a continuous behavioral response to partner vocalization.

This module inverts the direction of every other pipeline in ``modeling/``. The
five vocal pipelines predict something about a *vocalization* from a window of
preceding *behavior*; this one predicts a window-averaged *behavioral* variable
(e.g. the female's locomotor speed) and asks whether the partner's vocal trace
explains anything the kinematic and social features do not.

Key scientific and computational components:

1.  Tiled anchors rather than vocal events. Every other extractor anchors its
    rows on a USV or a bout, and ``_get_clean_tiled_epochs`` deliberately tiles
    only USV-*free* stretches. Here the anchors tile the whole session on a
    regular grid with no forbidden zones, because a history window that
    *contains* calls is precisely the observation under test. The stride equals
    the history length, so no two rows share history and the rows stay close to
    independent.
2.  A behavioral target. ``y`` is read from a behavioral feature column of a
    nominated mouse, clipped to its theoretical bounds and averaged over a short
    forward window ``[t + gap, t + gap + W)``. Averaging suppresses the
    frame-to-frame differentiation noise a single 6.7 ms sample carries, and
    breaks the near-determinism that would otherwise tie ``y`` to the last frame
    of its own history.
3.  A separable vocal block. The predictors are partitioned into a kinematic /
    social block and a vocal block, and the partition is recorded in the saved
    metadata so downstream selection can fit one model with the vocal block and
    one without, on identical folds.
4.  Two mouse indices, both absolute. ``model_predictor_mouse_index`` decides
    *whose calls* are ingested (``build_vocal_signal_columns`` honours
    ``usv_predictor_partner_only``), while ``behavioral_response.response_mouse_index``
    decides *whose behavior* is predicted. Index 0 is always the male and index 1
    always the female, so neither key has to be read relative to the other.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from ..os_utils import atomic_output_path
from .load_input_files import (
    _calculate_ibi_threshold,
    find_variable_length_bouts,
    load_behavioral_feature_data,
)
from .modeling_metadata import (
    build_input_metadata,
    derive_camera_fps_field,
    derive_experimental_condition,
    derive_feature_zoo_full,
    inject_metadata,
)
from .modeling_utils import (
    build_vocal_signal_columns,
    harmonize_session_columns,
    identify_empty_event_sessions,
    prepare_modeling_sessions,
    resolve_mouse_roles,
    run_predictor_audits,
    seeded_session_holdout,
    select_kinematic_columns,
    zscore_features_across_sessions,
)
from .modeling_vocal_bout_parameters import BoutParameterPipeline


def tile_anchor_frames(n_frames: int,
                       history_frames: int,
                       stride_frames: int,
                       lookahead_frames: int) -> np.ndarray:
    """
    Builds the regular grid of anchor frames for one session.

    An anchor ``t`` is legal when its full history window ``[t - history_frames, t)``
    lies inside the session AND its forward target window, which extends
    ``lookahead_frames`` past ``t``, also lies inside the session. Anchors are
    spaced ``stride_frames`` apart; when the stride equals ``history_frames`` no
    two rows share any history sample, which is the intended configuration.

    Unlike ``load_input_files._get_clean_tiled_epochs`` this tiler applies no
    forbidden zones around vocalizations — a history window containing calls is
    the observation of interest, not a contaminant.

    Parameters
    ----------
    n_frames : int
        Number of frames in the session.
    history_frames : int
        Length of the backward history window, in frames.
    stride_frames : int
        Spacing between consecutive anchors, in frames.
    lookahead_frames : int
        Number of frames after the anchor that the target window occupies
        (i.e. gap + averaging window), used to bound the last legal anchor.

    Returns
    -------
    anchor_frames : np.ndarray
        Sorted 1-D int array of anchor frame indices; empty when the session is
        too short to hold a single legal anchor.
    """

    if history_frames < 1:
        msg = f"`history_frames` must be >= 1, got {history_frames}."
        raise ValueError(msg)
    if stride_frames < 1:
        msg = f"`stride_frames` must be >= 1, got {stride_frames}."
        raise ValueError(msg)
    if lookahead_frames < 0:
        msg = f"`lookahead_frames` must be >= 0, got {lookahead_frames}."
        raise ValueError(msg)

    # `t` is legal when its target window ENDS at or before the last frame, so
    # `t = n_frames - lookahead_frames` qualifies and the range is inclusive.
    last_legal = n_frames - lookahead_frames
    if last_legal < history_frames:
        return np.empty(0, dtype=int)

    return np.arange(history_frames, last_legal + 1, stride_frames, dtype=int)


def forward_window_mean(values: np.ndarray,
                        anchor_frames: np.ndarray,
                        gap_frames: int,
                        window_frames: int) -> np.ndarray:
    """
    Averages a behavioral trace over a forward window at each anchor.

    Computes the mean of ``values`` over ``[t + gap_frames, t + gap_frames + window_frames)``
    for every anchor ``t``, ignoring non-finite samples. Anchors whose whole
    target window is non-finite yield ``NaN`` so the caller can drop them.

    The window is strictly forward of the anchor, so in INDEX terms it never
    overlaps that row's backward history.

    That is not the whole story for ``speed``, which
    ``compute_behavioral_features.calculate_speed`` smooths with a CENTRED
    ``Gaussian1DKernel`` (stddev ``floor(0.015 * fps)`` = 2 frames at 150 fps,
    17-sample support, ~40% of its weight at positive lags). The last history
    sample ``speed[t-1]`` therefore already carries weight from frames ``t`` to
    ``t+7`` -- the first ~53 ms of the target window. With
    ``target_gap_seconds = 0`` nothing absorbs that overlap. It inflates the
    BASELINE (which contains the response feature's own history), so the vocal
    increment is if anything conservative; but the reported ``baseline_d2`` and
    the ``1 - baseline`` denominator of ``fraction_of_remaining_deviance`` are
    both affected. A ``target_gap_seconds`` above the kernel half-width (8 frames,
    ~53 ms) removes it entirely, at negligible cost to a response whose measured
    peak sits at ~2.7 s.

    Consecutive anchors' targets do fall inside the *next* anchor's history when
    the stride is short; that couples neighbouring rows but is not leakage, and
    session-grouped folds keep such rows on the same side of any split.

    Parameters
    ----------
    values : np.ndarray
        1-D float trace of the response feature, already clipped to its
        theoretical bounds with out-of-range samples set to ``NaN``.
    anchor_frames : np.ndarray
        1-D int array of anchor frame indices.
    gap_frames : int
        Frames between the anchor and the start of the target window; ``0`` puts
        the target immediately after the history.
    window_frames : int
        Width of the averaging window, in frames.

    Returns
    -------
    window_means : np.ndarray
        1-D float array, one entry per anchor; ``NaN`` where the target window
        held no finite sample.
    """

    if window_frames < 1:
        msg = f"`window_frames` must be >= 1, got {window_frames}."
        raise ValueError(msg)

    finite_mask = np.isfinite(values)
    value_cumsum = np.concatenate([[0.0], np.cumsum(np.where(finite_mask, values, 0.0))])
    count_cumsum = np.concatenate([[0.0], np.cumsum(finite_mask)])

    window_starts = anchor_frames + gap_frames
    window_stops = window_starts + window_frames

    finite_counts = count_cumsum[window_stops] - count_cumsum[window_starts]
    value_sums = value_cumsum[window_stops] - value_cumsum[window_starts]

    window_means = np.full(anchor_frames.size, np.nan, dtype=float)
    populated = finite_counts > 0
    window_means[populated] = value_sums[populated] / finite_counts[populated]
    return window_means


class BehavioralResponsePipeline(BoutParameterPipeline):
    """
    Pipeline for predicting a continuous behavioral feature from a history window.

    Inherits the fitting, splitting, cross-validation and held-out-reserve
    machinery from ``BoutParameterPipeline`` — the target is a continuous,
    strictly positive scalar in both cases — and overrides only the extraction,
    which differs in three ways:

    1.  **Anchors** are tiled across the whole session instead of taken at vocal
        events, and carry no clean-history requirement.
    2.  **``y``** comes from a behavioral feature column of the mouse named by
        ``behavioral_response.response_mouse_index``, averaged forward over
        ``target_window_seconds`` starting ``target_gap_seconds`` after the
        anchor, rather than from the USV table.
    3.  **The predictor set is partitioned** into a kinematic/social block and a
        vocal block, with the partition written into the saved metadata so the
        nested block comparison downstream can fit with and without the vocal
        block on identical folds.
    """

    def __init__(self, modeling_settings_dict: dict[str, Any] | None = None) -> None:
        """
        Initializes the pipeline and resolves its block-local temporal geometry.

        Delegates settings loading to the parent chain, then converts this
        pipeline's own ``behavioral_response.history_seconds`` /
        ``target_window_seconds`` / ``target_gap_seconds`` into frame counts on
        the ``io.camera_sampling_rate`` grid. The history length is read from
        this pipeline's own block rather than ``model_params.filter_history`` so
        the block is self-contained, matching how ``glm_hmm`` carries its own
        ``history_frames``.

        Parameters
        ----------
        modeling_settings_dict : dict, optional
            Configuration dictionary. When ``None`` the parent chain loads
            ``_parameter_settings/modeling_settings.json``.

        Returns
        -------
        None
        """

        super().__init__(modeling_settings_dict=modeling_settings_dict)  # type: ignore[no-untyped-call]

        response_settings = self.modeling_settings['behavioral_response']
        camera_rate = self.modeling_settings['io']['camera_sampling_rate']

        self.response_history_frames = int(np.floor(camera_rate * response_settings['history_seconds']))
        self.response_window_frames = int(np.floor(camera_rate * response_settings['target_window_seconds']))
        self.response_gap_frames = int(np.floor(camera_rate * response_settings['target_gap_seconds']))

        # Already in frames, matching the loader's Gaussian sigma; kept as a float
        # so a fractional-frame kernel stays expressible.
        self.response_vocal_smoothing_frames = float(response_settings['vocal_smoothing_sd_frames'])

        # The inherited univariate runner and `get_basis_matrix_standardized` both
        # read `self.history_frames`, which the parent derives from
        # `model_params.filter_history`. Our windows are built at
        # `history_seconds`, so leaving the two independent means the univariate
        # arm reads windows of the wrong width the moment either value is changed.
        # They agree in the shipped JSON only by coincidence.
        self.history_frames = self.response_history_frames

        # `model_predictor_mouse_index` decides whose CALLS are ingested (via
        # `usv_predictor_partner_only`), while `response_mouse_index` decides whose
        # BEHAVIOUR is predicted -- opposite meanings on the same 0/1 axis. Since the
        # partner's calls are by definition the other animal's, the predictor index is
        # `1 - response_mouse_index` and is DERIVED here rather than set by hand: two
        # keys that must disagree by construction are two keys that can silently agree.
        # Copied at both levels so a caller-supplied settings dict is never mutated.
        response_idx = int(response_settings['response_mouse_index'])
        if response_idx not in (0, 1):
            msg = (
                f"`behavioral_response.response_mouse_index` must be 0 (male) or 1 (female), "
                f"got {response_idx}."
            )
            raise ValueError(msg)

        self.modeling_settings = dict(self.modeling_settings)
        self.modeling_settings['model_params'] = dict(self.modeling_settings['model_params'])
        self.modeling_settings['model_params']['model_predictor_mouse_index'] = 1 - response_idx

        if self.response_history_frames < 1:
            msg = (
                f"`behavioral_response.history_seconds` "
                f"({response_settings['history_seconds']}) yields "
                f"{self.response_history_frames} frames at "
                f"{camera_rate} fps; must be >= 1."
            )
            raise ValueError(msg)
        if self.response_window_frames < 1:
            msg = (
                f"`behavioral_response.target_window_seconds` "
                f"({response_settings['target_window_seconds']}) yields "
                f"{self.response_window_frames} frames at "
                f"{camera_rate} fps; must be >= 1."
            )
            raise ValueError(msg)

    def _null_target(self, y_train: np.ndarray, null_rng: np.random.Generator) -> np.ndarray:
        """
        Builds the screen's null target by circularly SHIFTING, not permuting.

        Permutation destroys the response's own serial structure, which makes it
        an easier null than the data warrants: the predictors are strongly
        autocorrelated, so a fully scrambled target is trivially harder to
        predict than a real one. A circular roll leaves the target's temporal
        structure exactly as observed and destroys only its alignment with the
        predictors, which is the relationship under test.

        The roll is over the fold's training rows as a block rather than within
        session, because the splitter yields ``(X_tr, y_tr, X_te, y_te)`` without
        the group labels. That is acceptable here: it breaks alignment at least
        as thoroughly as the permutation it replaces, and the cross-session
        mixing it introduces is present in that permutation too. The offset is
        kept away from both ends so the roll is never near the identity.

        Parameters
        ----------
        y_train : np.ndarray
            The fold's training targets.
        null_rng : np.random.Generator
            Seeded generator for this fold's null.

        Returns
        -------
        y_null : np.ndarray
            The targets rolled by a random offset.
        """

        n_rows = len(y_train)
        minimum_offset = max(
            1,
            int(np.ceil(
                self.modeling_settings['behavioral_response']['shift_null_min_seconds']
                / self.modeling_settings['behavioral_response']['history_seconds'],
            )),
        )
        if n_rows <= 2 * minimum_offset:
            # Too few rows to roll legally; fall back to the inherited permutation
            # rather than returning the identity, which would be no null at all.
            return super()._null_target(y_train, null_rng)

        offset = int(null_rng.integers(minimum_offset, n_rows - minimum_offset + 1))
        return np.roll(np.asarray(y_train), offset)

    def _resolve_response_column(self,
                                 session_df_columns: list[str],
                                 mouse_names: list[str]) -> str:
        """
        Names the raw feature column holding this session's response variable.

        The responder is identified by absolute slot index
        (``behavioral_response.response_mouse_index``; 0 is always the male, 1
        always the female), deliberately not by the relative ``self.`` /
        ``other.`` role keys, which can only be read against
        ``model_params.model_predictor_mouse_index``.

        Parameters
        ----------
        session_df_columns : list of str
            Column names of the raw per-session behavioral feature frame.
        mouse_names : list of str
            Ordered mouse track names for the session, slot 0 first.

        Returns
        -------
        response_column : str
            The ``{mouse_name}.{response_feature}`` column name.
        """

        response_settings = self.modeling_settings['behavioral_response']
        response_idx = response_settings['response_mouse_index']
        response_feature = response_settings['response_feature']

        if not 0 <= response_idx < len(mouse_names):
            msg = (
                f"`behavioral_response.response_mouse_index` = {response_idx} is "
                f"outside the {len(mouse_names)} mouse slots available for this session."
            )
            raise ValueError(msg)

        response_column = f"{mouse_names[response_idx]}.{response_feature}"
        if response_column not in session_df_columns:
            msg = (
                f"Response column '{response_column}' is absent from the session's "
                f"behavioral feature table; check "
                f"`behavioral_response.response_feature` ('{response_feature}')."
            )
            raise KeyError(msg)

        return response_column

    def _response_target_values(self,
                                raw_values: np.ndarray,
                                anchor_frames: np.ndarray) -> np.ndarray:
        """
        Turns a raw response trace into one forward-averaged target per anchor.

        Applies the theoretical-bounds clip from ``FeatureZoo.feature_boundaries``
        (out-of-range samples become ``NaN``, matching what
        ``zscore_different_sessions_together`` does to the predictors), then the
        same magnitude fold that helper applies — ``sqrt(x^2 + eps^2)`` for a
        feature in ``smooth_abs_features``, ``|x|`` for one in ``abs_features`` —
        and finally averages forward over the configured gap and window.

        The fold matters for signed features such as ``ego_yaw``, ``back_yaw`` and
        ``allo_roll``, whose distributions are near-symmetric about zero: their
        signed mean is a small residual of large opposing excursions, so it
        carries no stable direction, and it would additionally be dropped wholesale
        by the Gamma likelihood's positivity requirement. Skipping it would also
        leave the target and the identical column in the design matrix on
        different scales.

        The target is otherwise kept in **native units** — deliberately not
        z-scored, because the link function, not standardisation, is what handles
        a response's scale.

        Parameters
        ----------
        raw_values : np.ndarray
            1-D unclipped trace of the response feature for this session.
        anchor_frames : np.ndarray
            1-D int array of anchor frame indices.

        Returns
        -------
        target_values : np.ndarray
            1-D float array of per-anchor targets; ``NaN`` where the target
            window contained no in-bounds sample.
        """

        response_feature = self.modeling_settings['behavioral_response']['response_feature']
        kinematic_settings = self.modeling_settings['kinematic_features']

        values = np.asarray(raw_values, dtype=float)
        # `FeatureZoo` always defines `feature_boundaries`, so a missing entry means
        # the configured response feature is not a known feature -- fail rather than
        # silently skipping the clip, which would let the raw CSV's out-of-range
        # excursions (speed up to ~1e7) into the target.
        if response_feature not in self.feature_boundaries:
            msg = (
                f"`behavioral_response.response_feature` = '{response_feature}' has no entry in "
                f"`FeatureZoo.feature_boundaries`, so its theoretical range is unknown and the "
                f"raw trace cannot be clipped."
            )
            raise KeyError(msg)

        feature_bounds = np.asarray(self.feature_boundaries[response_feature], dtype=float)
        lower_bound, upper_bound = float(feature_bounds[0]), float(feature_bounds[1])
        out_of_range = ~np.isfinite(values) | (values < lower_bound) | (values > upper_bound)
        values = np.where(out_of_range, np.nan, values)

        # Apply the SAME magnitude fold the predictors get in
        # `zscore_different_sessions_together`. Without it a signed feature would
        # enter as a raw signed target while the identical column enters the design
        # matrix folded -- and a signed target additionally breaks the Gamma
        # likelihood, whose non-positive rows are dropped, silently discarding
        # roughly half the data. `smooth_abs_features` wins over `abs_features`
        # when a feature appears in both, matching the z-scoring helper.
        if response_feature in kinematic_settings['smooth_abs_features']:
            epsilon = float(kinematic_settings['smooth_abs_features'][response_feature])
            values = np.sqrt(np.square(values) + epsilon ** 2)
        elif response_feature in kinematic_settings['abs_features']:
            values = np.abs(values)

        return forward_window_mean(
            values=values,
            anchor_frames=anchor_frames,
            gap_frames=self.response_gap_frames,
            window_frames=self.response_window_frames,
        )

    def _vocal_block_feature_names(self, feature_names: list[str]) -> list[str]:
        """
        Selects the generic feature keys that constitute the vocal block.

        The block is **derived**, not configured: whichever vocal traces
        ``build_vocal_signal_columns`` emitted for the chosen
        ``behavioral_response.vocal_predictor_type`` are the block under test, and
        everything else is the kinematic / social baseline. A second setting
        listing the column names would have to be kept in step with the predictor
        type by hand, and any disagreement between them silently changes what is
        being tested -- so there is deliberately only one knob.

        The three markers below are the complete set of continuous vocal-signal
        names the loader can attach (``load_input_files``: ``usv_event`` for
        ``'pooled_binary'``, ``usv_rate`` for ``'pooled_rate'``, ``usv_cat_<n>``
        for ``'categories_rate'``, and both for ``'all_rate'``), matched after the
        ``self.`` / ``other.`` role prefixing the extractor applies.

        Parameters
        ----------
        feature_names : list of str
            All generic feature keys present in the extracted data dictionary.

        Returns
        -------
        vocal_block : list of str
            Sorted subset naming the block under test; may be empty, which the
            caller is expected to treat as a fatal misconfiguration.
        """

        vocal_markers = ('usv_rate', 'usv_event', 'usv_cat_')
        return sorted(
            name for name in feature_names
            if any(name.split('.')[-1].startswith(marker) for marker in vocal_markers)
        )

    def extract_and_save_modeling_input_data(self) -> None:
        """
        Extracts and saves the tiled (X, y, groups) triples for the response analysis.

        Orchestrates the whole extraction: loads behavior, attaches the partner's
        continuous vocal traces, harmonizes and z-scores the predictors, tiles
        anchors across each session, reads the forward-averaged behavioral target,
        and writes one pickle carrying the per-feature history windows plus an
        ``_input_metadata`` block that records the kinematic / vocal block split.

        Three departures from the inherited bout-parameter extraction are
        deliberate and load-bearing:

        - anchors tile the entire session with **no** clean-history requirement,
          because a history window containing calls is the observation under test;
        - ``y`` is a behavioral column of the mouse at
          ``behavioral_response.response_mouse_index``, kept in native units so a
          Gamma likelihood remains usable, never z-scored;
        - sessions are dropped when the **predictor** mouse has no bouts, not when
          the target mouse has none — with the male as predictor the female is the
          role-target and usually silent, so the inherited check would discard
          almost the whole cohort.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        response_settings = self.modeling_settings['behavioral_response']
        response_feature = response_settings['response_feature']
        response_idx = response_settings['response_mouse_index']
        predictor_mouse_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']

        print(f"--- Extracting behavioral-response data for: mouse slot {response_idx}, '{response_feature}' ---")

        txt_modeling_sessions = prepare_modeling_sessions(self.modeling_settings)

        mixture_model_idx = self.modeling_settings['model_params']['mixture_model_component_index']
        mixture_model_z = self.modeling_settings['model_params']['mixture_model_z_score']
        # Shallow copy: `vocal_features` is read by the five vocal pipelines too, so
        # this analysis overrides the representation locally rather than in place.
        voc_settings = dict(self.modeling_settings['vocal_features'])
        voc_settings['usv_predictor_type'] = response_settings['vocal_predictor_type']
        voc_settings['usv_predictor_smoothing_sd'] = self.response_vocal_smoothing_frames
        kin_settings = self.modeling_settings['kinematic_features']

        print("Loading behavioral feature data...")
        beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict = load_behavioral_feature_data(
            behavior_file_paths=txt_modeling_sessions,
            csv_sep=self.modeling_settings['io']['csv_separator'],
        )

        print(
            f"Generating vocal signals "
            f"(type: {voc_settings['usv_predictor_type']}, "
            f"partner only: {voc_settings['usv_predictor_partner_only']})..."
        )
        bout_data_dict = find_variable_length_bouts(
            root_directories=txt_modeling_sessions,
            mouse_ids_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            features_dict=beh_feature_data_dict,
            csv_sep=self.modeling_settings['io']['csv_separator'],
            mixture_model_component_index=mixture_model_idx,
            mixture_model_z_score=mixture_model_z,
            mixture_model_params=self.modeling_settings['mixture_model_params'],
            min_vocalizations=self.modeling_settings['model_params']['usv_per_bout_floor'],
            filter_history=response_settings['history_seconds'],
            proportion_smoothing_sd=voc_settings['usv_predictor_smoothing_sd'],
            vocal_output_type=voc_settings['usv_predictor_type'],
            noise_vocal_categories=voc_settings['usv_noise_categories'],
            category_column=voc_settings['usv_category_column_name'],
            noise_column=voc_settings['usv_noise_column'],
        )

        # The inherited pipeline drops sessions whose *target* mouse has no bouts.
        # Here the vocal block belongs to the PREDICTOR mouse, so it is the
        # predictor's silence that makes a session uninformative (a constant-zero
        # vocal column is degenerate under z-scoring); the role-target is usually
        # the near-silent female and must not be the criterion.
        sessions_to_remove = identify_empty_event_sessions(
            usv_data_dict=bout_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            target_idx=predictor_mouse_idx,
            event_key='bout_onsets',
            warn_label='session',
        )
        for sess in sessions_to_remove:
            if sess in beh_feature_data_dict:
                del beh_feature_data_dict[sess]

        print(f"Proceeding with {len(beh_feature_data_dict)} sessions after filtering vocally-empty ones.")

        # The response target is read from the RAW loaded frame, before column
        # selection and before z-scoring: it must stay in native units (a Gamma
        # likelihood needs y > 0) and it need not belong to the predictor zoo.
        raw_response_traces: dict[str, np.ndarray] = {}
        predictor_bout_onsets: dict[str, np.ndarray] = {}
        processed_beh_feature_data_dict = {}
        for sess_id, session_df in beh_feature_data_dict.items():
            if sess_id not in mouse_track_names_dict:
                continue

            (predictor_mouse_idx,
             target_mouse_idx,
             p_name,
             t_name) = resolve_mouse_roles(
                modeling_settings=self.modeling_settings,
                mouse_names_dict=mouse_track_names_dict,
                session_id=sess_id,
            )

            session_df_cols = list(session_df.columns)
            response_column = self._resolve_response_column(
                session_df_columns=session_df_cols,
                mouse_names=mouse_track_names_dict[sess_id],
            )
            raw_response_traces[sess_id] = session_df[response_column].to_numpy().astype(float)

            # The audit's Y(t) must be the PREDICTOR mouse's calls -- the block under
            # test -- not the role-target's. The role-target here is the responder,
            # who is typically near-silent, so the default per-mouse lookup would
            # cross-correlate every feature against the wrong animal's vocalizations.
            # Supplying the times directly leaves `target_idx` / `predictor_idx`
            # untouched, so the audit's self./other. naming still matches the main
            # pickle (the multinomial pipeline uses this same escape hatch).
            if p_name in bout_data_dict[sess_id]:
                predictor_bout_onsets[sess_id] = np.asarray(
                    bout_data_dict[sess_id][p_name]['bout_onsets'], dtype=float,
                )

            columns_to_keep_session = select_kinematic_columns(
                session_df_columns=session_df_cols,
                target_name=t_name,
                predictor_name=p_name,
                kin_settings=kin_settings,
                predictor_idx=predictor_mouse_idx,
            )

            new_voc_cols, new_voc_col_names = build_vocal_signal_columns(
                usv_data_dict=bout_data_dict,
                session_id=sess_id,
                target_name=t_name,
                predictor_name=p_name,
                voc_settings=voc_settings,
            )

            columns_to_keep_session = sorted(set(columns_to_keep_session) | set(new_voc_col_names))
            existing_cols = [c for c in columns_to_keep_session if c in session_df_cols]
            current_df = session_df.select(existing_cols)
            if new_voc_cols:
                current_df = current_df.with_columns(new_voc_cols)

            processed_beh_feature_data_dict[sess_id] = current_df

        print("Standardizing columns ...")
        processed_beh_feature_data_dict, revised_behavioral_predictors = harmonize_session_columns(
            processed_beh_dict=processed_beh_feature_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
        )

        if hasattr(self, 'feature_boundaries'):
            feature_bounds = self.feature_boundaries
        else:
            feature_bounds = {}

        processed_beh_feature_data_dict = zscore_features_across_sessions(
            processed_beh_dict=processed_beh_feature_data_dict,
            suffixes=revised_behavioral_predictors,
            feature_bounds=feature_bounds,
            abs_features=kin_settings['abs_features'],
            smooth_abs_features=kin_settings['smooth_abs_features'],
        )

        cohort_condition = derive_experimental_condition(self.modeling_settings)
        analysis_tag = f"behavioral_response_m{response_idx}_{response_feature}"
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"modeling_{analysis_tag}_{cohort_condition}_{ts}.pkl"

        ibi_thresholds_md = {}
        mixture_model_params_md = self.modeling_settings['mixture_model_params']
        for sex in ('male', 'female'):
            params = mixture_model_params_md[sex]
            if 0 <= mixture_model_idx < len(params['means']):
                ibi_thresholds_md[sex] = float(_calculate_ibi_threshold(
                    params['means'][mixture_model_idx],
                    params['sds'][mixture_model_idx],
                    mixture_model_z,
                ))
            else:
                ibi_thresholds_md[sex] = float('nan')

        if not processed_beh_feature_data_dict:
            msg = (
                "No session survived loading and column selection, so there is nothing to "
                "extract. Check the session list, that each session has both a behavioral "
                "feature CSV and a USV summary, and that the predictor mouse actually "
                "vocalizes in them."
            )
            raise RuntimeError(msg)

        first_sess_id = next(iter(processed_beh_feature_data_dict))
        kept_columns_first_sess = list(processed_beh_feature_data_dict[first_sess_id].columns)
        vocal_columns_md = sorted({
            c for c in kept_columns_first_sess
            if any(tok in c for tok in ('usv_rate', 'usv_cat_', 'usv_event'))
        })

        _sorted_session_ids = sorted(processed_beh_feature_data_dict.keys())
        held_out_session_ids = seeded_session_holdout(
            session_ids=_sorted_session_ids,
            held_out_test_proportion=self.modeling_settings['model_validation']['held_out_test_proportion'],
            random_seed=self.modeling_settings['model_validation']['random_seed'],
        )

        input_metadata = build_input_metadata(
            modeling_settings=self.modeling_settings,
            analysis_type='behavioral_response',
            analysis_tag=analysis_tag,
            pipeline_class=type(self).__name__,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            n_sessions_used=len(processed_beh_feature_data_dict),
            session_ids=_sorted_session_ids,
            n_events_per_session={},
            held_out_session_ids=held_out_session_ids,
            feature_zoo_full=derive_feature_zoo_full(self.modeling_settings),
            feature_zoo_kept=[],
            dyadic_engagement_features_used=list(kin_settings['dyadic_engagement']),
            dyadic_pose_symmetric_features_used=kin_settings['dyadic_pose_symmetric'],
            noise_vocal_categories_excluded=list(voc_settings['usv_noise_categories']),
            vocal_signal_columns_added=vocal_columns_md,
            filter_history_seconds=float(response_settings['history_seconds']),
            filter_history_frames=int(self.response_history_frames),
            camera_sampling_rate_hz=derive_camera_fps_field(camera_fr_dict),
            ibi_thresholds=ibi_thresholds_md,
            analysis_specific={
                'response_mouse_index': response_idx,
                'derived_predictor_mouse_index': predictor_mouse_idx,
                'response_feature': response_feature,
                'response_fold': (
                    'smooth_abs' if response_feature in kin_settings['smooth_abs_features']
                    else 'abs' if response_feature in kin_settings['abs_features']
                    else 'none'
                ),
                'target_window_seconds': response_settings['target_window_seconds'],
                'target_window_frames': int(self.response_window_frames),
                'target_gap_seconds': response_settings['target_gap_seconds'],
                'target_gap_frames': int(self.response_gap_frames),
                'anchor_stride_frames': int(self.response_history_frames),
                'vocal_predictor_type': response_settings['vocal_predictor_type'],
                'vocal_smoothing_sd_frames': self.response_vocal_smoothing_frames,
                'likelihood': response_settings['likelihood'],
                'n_shift_draws': response_settings['n_shift_draws'],
                'shift_null_min_seconds': response_settings['shift_null_min_seconds'],
            },
        )

        # NOTE: `input_metadata` is still the pre-extraction block here -- the
        # per-session counts, kept-feature list and block partition are filled in
        # by `_save_extracted_data` after the tiling loop. The audit artifacts
        # therefore carry the cohort/temporal provenance but empty event counts;
        # the paired modeling-input pickle is the complete record.
        run_predictor_audits(
            processed_beh_dict=processed_beh_feature_data_dict,
            usv_data_dict=bout_data_dict,
            mouse_names_dict=mouse_track_names_dict,
            camera_fps_dict=camera_fr_dict,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            history_frames=self.response_history_frames,
            event_keys=['bout_onsets'],
            settings=self.modeling_settings,
            save_dir=self.modeling_settings['io']['save_directory'],
            pickle_basename=fname,
            input_metadata=input_metadata,
            onset_event_key='bout_onsets',
            precomputed_event_times=predictor_bout_onsets,
            precomputed_onset_times=predictor_bout_onsets,
        )

        lookahead_frames = self.response_gap_frames + self.response_window_frames
        final_data_dict: dict[str, dict[str, Any]] = {}
        anchors_per_session: dict[str, int] = {}
        # Same marker set `_vocal_block_feature_names` uses to identify vocal
        # columns; kept local rather than promoted to a module constant.
        vocal_markers = ('usv_rate', 'usv_event', 'usv_cat_')
        vocal_occupancy_rows: list[float] = []
        informative_fraction_per_session: dict[str, float] = {}

        for sess_id, df in tqdm(processed_beh_feature_data_dict.items(), desc="Tiling anchors"):
            t_name = mouse_track_names_dict[sess_id][target_mouse_idx]
            p_name = mouse_track_names_dict[sess_id][predictor_mouse_idx]

            n_frames = df.height
            anchor_frames = tile_anchor_frames(
                n_frames=n_frames,
                history_frames=self.response_history_frames,
                stride_frames=self.response_history_frames,
                lookahead_frames=lookahead_frames,
            )
            if anchor_frames.size == 0:
                continue

            target_values = self._response_target_values(
                raw_values=raw_response_traces[sess_id],
                anchor_frames=anchor_frames,
            )
            # A Gamma likelihood needs a strictly positive response; anchors whose
            # target window held no in-bounds sample, or averaged to exactly zero,
            # cannot be modelled and are dropped rather than floored. Dropping
            # rather than flooring is deliberate -- a floored zero would enter the
            # fit as a real observation -- but note it is a systematic exclusion of
            # the completely immobile animal, not a random one. Note also that a
            # window surviving on ONE in-bounds sample is kept, so a target can be
            # a single 6.7 ms sample rather than a 500 ms average; those cluster on
            # tracking failures and on excursions past the feature's upper bound.
            usable = np.isfinite(target_values) & (target_values > 0.0)
            anchor_frames = anchor_frames[usable]
            target_values = target_values[usable]
            if anchor_frames.size == 0:
                continue

            window_starts = anchor_frames - self.response_history_frames
            n_valid = int(anchor_frames.size)
            anchors_per_session[sess_id] = n_valid

            # Per-anchor call content of the predictor mouse, from the RAW binary
            # occupancy rather than the smoothed `usv_rate`: smoothing spreads a
            # call over its neighbours, and the design matrix is pooled z-scored,
            # so neither can be thresholded to recover "was he calling here".
            # This drives the occupancy ladder in the selection step, which exists
            # because the acceptance gate is a global deviance over EVERY row
            # while only ~30.6% of anchors carry any call at all (measured over
            # this cohort: 121 sessions, 36,200 anchors).
            # Source is `continuous_vocal_signals`, the predictor traces themselves,
            # rather than a raw USV table: the ladder asks where the trace the model
            # actually sees is active, so defining it from that trace keeps the
            # diagnostic self-consistent. `usv_count` lives on `find_onset_epochs`
            # output and is NOT present here. The design matrix cannot be used
            # either -- it is pooled z-scored, so "silent" is a constant negative
            # value rather than zero.
            vocal_signals = bout_data_dict[sess_id][p_name]['continuous_vocal_signals']
            active_traces = [
                np.asarray(vocal_signals[key], dtype=float) for key in vocal_signals
                if any(marker in key for marker in vocal_markers)
            ]
            if active_traces:
                trace_length = min(trace.size for trace in active_traces)
                any_active = np.zeros(trace_length, dtype=float)
                for trace in active_traces:
                    any_active = np.maximum(any_active, np.abs(trace[:trace_length]))
                active_frames = (any_active > 0.0).astype(float)
                cumulative_active = np.concatenate([[0.0], np.cumsum(active_frames)])
                window_ends = np.clip(anchor_frames, 0, cumulative_active.size - 1)
                window_begins = np.clip(window_starts, 0, cumulative_active.size - 1)
                session_occupancy = (
                    (cumulative_active[window_ends] - cumulative_active[window_begins])
                    / float(self.response_history_frames)
                )
            else:
                session_occupancy = np.zeros(n_valid, dtype=float)
            vocal_occupancy_rows.extend(session_occupancy.tolist())
            informative_fraction_per_session[sess_id] = float(
                np.mean(session_occupancy > 0.0)) if n_valid else float('nan')

            for col in df.columns:
                base_feature = col.split('.')[-1]
                if base_feature.isdigit():
                    continue

                if '-' in base_feature:
                    generic_key = base_feature
                elif col.startswith(f"{t_name}."):
                    generic_key = f"self.{base_feature}"
                elif col.startswith(f"{p_name}."):
                    generic_key = f"other.{base_feature}"
                else:
                    generic_key = base_feature

                if generic_key not in final_data_dict:
                    final_data_dict[generic_key] = {'X': [], 'y': [], 'groups': []}

                col_data = np.nan_to_num(df[col].to_numpy(), nan=0.0)
                windows = sliding_window_view(col_data, self.response_history_frames)
                final_data_dict[generic_key]['X'].extend(windows[window_starts].copy())
                final_data_dict[generic_key]['y'].extend(target_values)
                final_data_dict[generic_key]['groups'].extend([sess_id] * n_valid)

        for feature_arrays in final_data_dict.values():
            for key in ('X', 'y', 'groups'):
                feature_arrays[key] = np.array(feature_arrays[key])

        input_metadata['informative_fraction_per_session'] = informative_fraction_per_session
        input_metadata['vocal_occupancy_pooled_fraction'] = (
            float(np.mean(np.asarray(vocal_occupancy_rows) > 0.0))
            if vocal_occupancy_rows else float('nan')
        )

        self._save_extracted_data(
            final_data_dict=final_data_dict,
            input_metadata=input_metadata,
            anchors_per_session=anchors_per_session,
            vocal_occupancy=np.asarray(vocal_occupancy_rows, dtype=float),
            fname=fname,
        )

    def _save_extracted_data(self,
                             final_data_dict: dict[str, dict[str, Any]],
                             input_metadata: dict[str, Any],
                             anchors_per_session: dict[str, int],
                             vocal_occupancy: np.ndarray,
                             fname: str) -> None:
        """
        Validates alignment, prints the extraction summary and publishes the pickle.

        Every generic feature must carry the same number of rows in the same
        session order, because the rows are positionally paired with one shared
        target vector; a mismatch means predictors are lined up against the wrong
        anchors. That is a silent scientific corruption rather than a crash, so it
        raises instead of warning, exactly as the inherited bout-parameter
        extraction does.

        The kinematic / vocal block partition is resolved here and written into
        ``analysis_specific`` so the downstream nested comparison never has to
        re-derive which columns are under test.

        Parameters
        ----------
        final_data_dict : dict
            Mapping ``generic_feature -> {'X', 'y', 'groups'}`` of aligned arrays.
        input_metadata : dict
            The provenance block to embed; mutated in place with the per-session
            anchor counts, the kept feature list and the block partition.
        vocal_occupancy : np.ndarray
            Per-anchor fraction of history frames containing a predictor-mouse
            call, row-aligned with every feature's ``X``. Stored so the selection
            step can report the occupancy ladder without recomputing it from the
            pooled z-scored design, where "no call" is a constant negative value
            rather than zero and so cannot be recovered by thresholding.
        anchors_per_session : dict
            Mapping ``session_id -> number of retained anchors``.
        fname : str
            Basename of the pickle to publish under ``io.save_directory``.

        Returns
        -------
        None
        """

        final_features = sorted(final_data_dict.keys())
        total_covariates = len(final_features)
        if total_covariates == 0:
            msg = (
                "No features survived behavioral-response extraction; every session was "
                "either too short for a single anchor or had no usable target values."
            )
            raise RuntimeError(msg)

        reference_feature = final_features[0]
        reference_y = final_data_dict[reference_feature]['y']
        reference_groups = final_data_dict[reference_feature]['groups']

        mismatched_features = []
        for feat in final_features[1:]:
            if len(final_data_dict[feat]['y']) != len(reference_y):
                mismatched_features.append(feat)
                continue
            if not np.array_equal(final_data_dict[feat]['groups'], reference_groups):
                mismatched_features.append(feat)
        alignment_passed = not mismatched_features

        vocal_block = self._vocal_block_feature_names(final_features)
        baseline_block = [f for f in final_features if f not in set(vocal_block)]

        total_n = len(reference_y)
        history_length = final_data_dict[reference_feature]['X'].shape[1]

        print("\n" + "=" * 105)
        print(
            f"BEHAVIORAL-RESPONSE DATA SUMMARY: slot "
            f"{self.modeling_settings['behavioral_response']['response_mouse_index']} "
            f"'{self.modeling_settings['behavioral_response']['response_feature']}'"
        )
        print("=" * 105)
        print(f"{'#':<4} {'Generic Feature Name':<45} | {'Block':<9} | {'Anchors (N)':<12} | {'Status'}")
        print("-" * 105)
        for i, feat in enumerate(final_features, 1):
            block_label = "VOCAL" if feat in set(vocal_block) else "baseline"
            is_zero = bool(np.all(final_data_dict[feat]['X'] == 0))
            status = "ZERO-FILLED" if is_zero else "DATA-PRESENT"
            print(f"{i:3}. {feat:<45} | {block_label:<9} | {len(final_data_dict[feat]['y']):<12} | {status}")
        print("-" * 105)
        print("PROJECT-WIDE TALLY:")
        print(f"  > Total Unique Covariates:      {total_covariates}")
        print(f"  > Baseline-block Covariates:    {len(baseline_block)}")
        print(f"  > Vocal-block Covariates:       {len(vocal_block)}")
        print(f"  > Total Sessions Included:      {len(anchors_per_session)}")
        print(f"  > Total Anchors Across Project: {total_n}")
        print(f"  > History Window Length:        {history_length} frames")
        print(f"  > Target Window Length:         {self.response_window_frames} frames")
        print(f"  > INTRA-SESSION ALIGNMENT:      {'PASSED (True)' if alignment_passed else 'FAILED (False)'}")
        if not alignment_passed:
            print(f"  [!] ALERT: Dimensional or grouping mismatch in: {mismatched_features}")
        print("=" * 105 + "\n")

        if not alignment_passed:
            msg = (
                f"Intra-session alignment FAILED for behavioral-response extraction: "
                f"predictors are misaligned with targets for feature(s) {mismatched_features} "
                f"(mismatched anchor count or session grouping vs the reference feature "
                f"'{reference_feature}'). Refusing to write a known-misaligned artifact; fix the "
                f"upstream extraction and re-run."
            )
            raise RuntimeError(msg)

        if not vocal_block:
            msg = (
                f"`behavioral_response.vocal_predictor_type` = "
                f"'{self.modeling_settings['behavioral_response']['vocal_predictor_type']}' produced "
                f"no vocal columns, so the block under test is empty and the nested comparison has "
                f"nothing to add. Check that the predictor mouse actually vocalizes and that "
                f"`vocal_features.usv_predictor_partner_only` selects the intended animal."
            )
            raise RuntimeError(msg)

        # `session_ids` / `n_sessions_used` were built before the tiling loop, which
        # can skip a session (no legal anchor, or no usable target). Re-derive both
        # from the sessions that actually contributed rows so the provenance block
        # cannot disagree with the per-session counts beside it.
        contributing_sessions = sorted(anchors_per_session)
        input_metadata['session_ids'] = contributing_sessions
        input_metadata['n_sessions_used'] = len(contributing_sessions)
        input_metadata['n_events_per_session'] = {
            str(sess_id): {'anchors': int(count)}
            for sess_id, count in sorted(anchors_per_session.items())
        }
        # Reserved sessions that contributed no rows cannot be scored later, so
        # record the intersection rather than the requested list.
        input_metadata['held_out_session_ids'] = [
            s for s in input_metadata['held_out_session_ids'] if s in anchors_per_session
        ]
        input_metadata['feature_zoo_kept'] = final_features
        n_rows_reference = int(np.asarray(final_data_dict[reference_feature]['y']).shape[0])
        if vocal_occupancy.shape[0] != n_rows_reference:
            msg = (
                f"vocal occupancy carries {vocal_occupancy.shape[0]} rows but the design "
                f"has {n_rows_reference}; the occupancy ladder would be scored against "
                f"misaligned rows."
            )
            raise ValueError(msg)
        # Stored as a list, not an ndarray: `metadata_blocks_equal` compares
        # metadata values with `!=`, which is ambiguous for an array and aborts
        # consolidation. A list compares to a plain bool.
        input_metadata['analysis_specific']['vocal_occupancy'] = vocal_occupancy.tolist()
        input_metadata['analysis_specific']['vocal_block_features'] = vocal_block
        input_metadata['analysis_specific']['baseline_block_features'] = baseline_block

        save_dir = Path(self.modeling_settings['io']['save_directory'])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / fname

        artifact = inject_metadata(final_data_dict, _input_metadata=input_metadata)
        with atomic_output_path(save_path) as tmp_path, tmp_path.open('wb') as f:
            pickle.dump(artifact, f)
        print(f"[+] Saved: {save_path}")
