"""
@author: bartulem
Module for extracting a female behavioral response to male vocal bouts.

This module inverts the direction of every other pipeline in ``modeling/``. The
five vocal pipelines predict something about a *vocalization* from a window of
preceding *behavior*; this one predicts a *behavioral* variable and asks whether
the partner's calling changed it.

It sets up two questions, both answered from one extraction:

1.  **Does a male vocal bout, versus comparable silence, change her behavior?**
    Rows are male bout offsets contrasted against anchors drawn from the silence
    *between* bouts.
2.  **Does bout duration matter?** Among the bout rows, longer versus shorter.

Key design decisions, each ruled rather than defaulted:

1.  Anchors are **events, not tiles**. Every row sits either at a bout offset or
    inside an inter-bout gap. An earlier tiled design spent ~70% of its rows on
    windows containing no calls at all, which diluted the very contrast it was
    meant to measure.
2.  The silent comparison is **inter-bout silence**, not globally quiet stretches.
    A point inside a gap is a moment where he *could* have called and did not,
    with both animals demonstrably still interacting. Globally quiet periods are a
    different behavioral regime, and covariate adjustment would then be
    extrapolating between two separated groups rather than comparing within one.
3.  Covariates are **summaries, not lag histories**. Each feature's pre-anchor
    window collapses to a mean over the last ``0.5 s`` and a mean over the full
    ``4 s``. Their job is to hold pre-anchor state fixed, not to model a temporal
    filter; and these features are slow (autocorrelation horizons from ~0.75 s for
    speed to ~6.8 s for ``nose-nose``), so finer sub-windows would be collinear
    rather than informative.
4.  The response is kept in **native units, never z-scored** -- a Gamma likelihood
    needs ``y > 0``, and the effect is then readable in the feature's own units.
    Covariates *are* pooled z-scored, so their coefficients stay comparable.
5.  Two targets are written per anchor: one mean over the whole window, and a
    series of short bins across it, so the same contrast yields both a single
    number and a time course of when the response appears.

The contrast itself lives here too, in
:func:`behavioral_response_contrast`. It is a GLM, not a nested predictive
comparison: a nested comparison answers "does the vocal block improve
out-of-sample prediction" and reports a ``dD^2``, which conflates effect size,
timing and nonlinearity -- on the tiled precursor it came back at 0.0039,
reproducible across every split and uninterpretable. A coefficient answers the
question actually asked, in the feature's own units, with an interval.

Settings
--------
Everything is read from the ``behavioral_response`` block of
``modeling_settings.json``: ``response_mouse_index``, ``response_feature``,
``history_seconds``, ``target_window_seconds``, ``target_bin_seconds``,
``post_bout_silence_seconds``, ``covariate_summary_seconds``, ``duration_n_bins``
and ``likelihood``. The anchor RNG is seeded from
``model_validation.random_seed``.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

from ..os_utils import atomic_output_path, configure_path
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
    format_run_header,
    format_run_summary,
    format_selection_step,
    harmonize_session_columns,
    identify_empty_event_sessions,
    prepare_modeling_sessions,
    resolve_mouse_roles,
    run_predictor_audits,
    select_kinematic_columns,
    zscore_features_across_sessions,
)
from .modeling_vocal_bout_parameters import BoutParameterPipeline


def bout_offset_anchors(bout_onsets: np.ndarray,
                        bout_durations: np.ndarray,
                        camera_fps: float,
                        n_frames: int,
                        history_frames: int,
                        lookahead_frames: int,
                        silence_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Places one anchor at the offset of every qualifying male vocal bout.

    A bout offset qualifies when the next bout starts at least
    ``silence_seconds`` later, so the forward window genuinely follows the bout
    rather than straddling the next one, and when the anchor leaves room for both
    the pre-anchor history and the forward window inside the recording.

    The silence requirement is deliberately tied to the forward window rather than
    set independently: anything beyond the window is discarded data bought for
    cleanliness the analysis never uses. It does select on the future, though --
    bigger bouts are followed sooner by the next bout (measured Spearman
    ``rho = -0.12`` between syllable count and gap), so a longer requirement
    preferentially discards long bouts and truncates the very predictor question 2
    asks about.

    Parameters
    ----------
    bout_onsets : np.ndarray
        Bout start times in seconds, ascending.
    bout_durations : np.ndarray
        Bout durations in seconds, aligned with ``bout_onsets``.
    camera_fps : float
        Tracking frame rate.
    n_frames : int
        Number of frames in the session.
    history_frames : int
        Pre-anchor window width, in frames.
    lookahead_frames : int
        Forward window width, in frames.
    silence_seconds : float
        Required quiet interval after the bout offset.

    Returns
    -------
    anchor_frames, kept_durations : tuple of np.ndarray
        Frame index of each qualifying offset, and that bout's duration in
        seconds. Both empty when the session has no qualifying bout.
    """

    if bout_onsets.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    order = np.argsort(bout_onsets)
    onsets, durations = bout_onsets[order], bout_durations[order]
    offsets = onsets + durations

    # A bout with no successor is limited only by the end of the recording.
    next_onset = np.concatenate([onsets[1:], [np.inf]])
    has_silence = (next_onset - offsets) >= silence_seconds

    anchor_frames = np.round(offsets * camera_fps).astype(int)
    fits = (anchor_frames - history_frames >= 0) & (anchor_frames + lookahead_frames <= n_frames)

    keep = has_silence & fits
    return anchor_frames[keep], durations[keep]


def inter_bout_quiet_anchors(bout_onsets: np.ndarray,
                             bout_durations: np.ndarray,
                             camera_fps: float,
                             n_frames: int,
                             history_frames: int,
                             lookahead_frames: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Draws one silent anchor from each sufficiently long gap between bouts.

    The anchor is placed uniformly at random inside the sub-interval where its
    whole pre-anchor history and forward window fall within the gap, so no male
    call touches either. Random placement rather than the midpoint avoids
    systematically sampling the quietest instant of every gap.

    One anchor per gap, not several: tiling long gaps would buy rows that are
    strongly correlated with each other and would let a handful of long silences
    dominate the silent condition.

    Parameters
    ----------
    bout_onsets : np.ndarray
        Bout start times in seconds, ascending.
    bout_durations : np.ndarray
        Bout durations in seconds, aligned with ``bout_onsets``.
    camera_fps : float
        Tracking frame rate.
    n_frames : int
        Number of frames in the session.
    history_frames : int
        Pre-anchor window width, in frames.
    lookahead_frames : int
        Forward window width, in frames.
    rng : np.random.Generator
        Seeded generator for the within-gap placement.

    Returns
    -------
    anchor_frames : np.ndarray
        Frame index of one silent anchor per qualifying gap; empty when no gap is
        long enough to hold a full history plus forward window.
    """

    if bout_onsets.size < 2:
        return np.empty(0, dtype=int)

    order = np.argsort(bout_onsets)
    onsets, durations = bout_onsets[order], bout_durations[order]
    offsets = onsets + durations

    history_seconds = history_frames / camera_fps
    lookahead_seconds = lookahead_frames / camera_fps

    earliest = offsets[:-1] + history_seconds
    latest = onsets[1:] - lookahead_seconds
    usable = latest > earliest
    if not np.any(usable):
        return np.empty(0, dtype=int)

    drawn = rng.uniform(earliest[usable], latest[usable])
    anchor_frames = np.round(drawn * camera_fps).astype(int)
    fits = (anchor_frames - history_frames >= 0) & (anchor_frames + lookahead_frames <= n_frames)
    return anchor_frames[fits]


def summarise_history(values: np.ndarray,
                      anchor_frames: np.ndarray,
                      summary_frames: list[int]) -> np.ndarray:
    """
    Collapses each anchor's pre-anchor history into one mean per summary width.

    Means are NaN-aware: out-of-bounds samples are nulled upstream by
    ``zscore_features_across_sessions``, so a window is averaged over whatever
    finite samples it holds, and returns ``nan`` only when it holds none.

    Parameters
    ----------
    values : np.ndarray
        One feature's per-frame trace for the session, already z-scored.
    anchor_frames : np.ndarray
        Anchor frame indices; the history for anchor ``t`` is ``[t - w, t)``.
    summary_frames : list of int
        Window widths in frames, one summary produced per width.

    Returns
    -------
    summaries : np.ndarray
        ``(n_anchors, len(summary_frames))`` array of window means.
    """

    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    value_cumsum = np.concatenate([[0.0], np.cumsum(filled)])
    count_cumsum = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])

    summaries = np.empty((anchor_frames.size, len(summary_frames)), dtype=float)
    for column, width in enumerate(summary_frames):
        starts = np.maximum(anchor_frames - width, 0)
        total = value_cumsum[anchor_frames] - value_cumsum[starts]
        count = count_cumsum[anchor_frames] - count_cumsum[starts]
        with np.errstate(invalid='ignore', divide='ignore'):
            summaries[:, column] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return summaries


def forward_window_mean(values: np.ndarray,
                        anchor_frames: np.ndarray,
                        window_frames: int,
                        lower_bound: float,
                        upper_bound: float) -> np.ndarray:
    """
    Averages a raw feature over the forward window following each anchor.

    Samples outside ``[lower_bound, upper_bound]`` are dropped before averaging
    rather than clamped: the on-disk CSVs carry excursions (speed to ~1e7) that
    would otherwise dominate a mean, and clamping would enter a fabricated
    boundary value as a real observation.

    Parameters
    ----------
    values : np.ndarray
        The response feature's per-frame trace, in native units.
    anchor_frames : np.ndarray
        Anchor frame indices; the window for anchor ``t`` is ``[t, t + w)``.
    window_frames : int
        Forward window width, in frames.
    lower_bound, upper_bound : float
        Theoretical bounds for the feature, from ``FeatureZoo.feature_boundaries``.

    Returns
    -------
    window_means : np.ndarray
        One mean per anchor; ``nan`` where the window held no in-bounds sample.
    """

    in_bounds = np.isfinite(values) & (values >= lower_bound) & (values <= upper_bound)
    filled = np.where(in_bounds, values, 0.0)
    value_cumsum = np.concatenate([[0.0], np.cumsum(filled)])
    count_cumsum = np.concatenate([[0], np.cumsum(in_bounds.astype(np.int64))])

    ends = np.minimum(anchor_frames + window_frames, values.size)
    total = value_cumsum[ends] - value_cumsum[anchor_frames]
    count = count_cumsum[ends] - count_cumsum[anchor_frames]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


class BehavioralResponsePipeline(BoutParameterPipeline):
    """
    Pipeline for contrasting female behavior after male bouts against silence.

    Subclasses ``BoutParameterPipeline`` for its settings loading, session
    preparation and provenance machinery, and replaces the anchor and target
    construction entirely.

    Two mouse indices are in play and both are absolute (0 is always the male, 1
    always the female). Only ``behavioral_response.response_mouse_index`` is set
    by hand; ``model_params.model_predictor_mouse_index`` is DERIVED as
    ``1 - response_mouse_index`` on a private copy of the settings, so the shared
    block the five vocal pipelines read is never disturbed.
    """

    def __init__(self, modeling_settings_dict: dict[str, Any] | None = None) -> None:
        """
        Initializes the pipeline and resolves its temporal geometry.

        Converts every duration in the ``behavioral_response`` block into frames
        at the configured camera rate, and derives the predictor mouse index from
        the response index on a private settings copy.

        Parameters
        ----------
        modeling_settings_dict : dict or None
            Full modeling settings; loaded from JSON by the parent chain when
            ``None``.

        Returns
        -------
        None
        """

        super().__init__(modeling_settings_dict=modeling_settings_dict)

        response_settings = self.modeling_settings['behavioral_response']
        camera_rate = self.modeling_settings['io']['camera_sampling_rate']

        self.response_history_frames = int(np.floor(camera_rate * response_settings['history_seconds']))
        self.response_window_frames = int(np.floor(camera_rate * response_settings['target_window_seconds']))
        self.response_bin_frames = int(np.floor(camera_rate * response_settings['target_bin_seconds']))
        self.response_silence_frames = int(np.floor(camera_rate * response_settings['post_bout_silence_seconds']))
        self.covariate_summary_frames = [
            int(np.floor(camera_rate * seconds))
            for seconds in response_settings['covariate_summary_seconds']
        ]
        # The parent's fold machinery reads `self.history_frames`.
        self.history_frames = self.response_history_frames

        response_idx = int(response_settings['response_mouse_index'])
        if response_idx not in (0, 1):
            msg = (
                f"`behavioral_response.response_mouse_index` must be 0 (male) or 1 "
                f"(female); got {response_idx}."
            )
            raise ValueError(msg)

        self.modeling_settings = dict(self.modeling_settings)
        self.modeling_settings['model_params'] = dict(self.modeling_settings['model_params'])
        self.modeling_settings['model_params']['model_predictor_mouse_index'] = 1 - response_idx

        for label, frames, seconds in (
            ('history_seconds', self.response_history_frames, response_settings['history_seconds']),
            ('target_window_seconds', self.response_window_frames, response_settings['target_window_seconds']),
            ('target_bin_seconds', self.response_bin_frames, response_settings['target_bin_seconds']),
        ):
            if frames < 1:
                msg = (
                    f"`behavioral_response.{label}` ({seconds}) yields {frames} frames at "
                    f"{camera_rate} fps; it must cover at least one frame."
                )
                raise ValueError(msg)

        if self.response_bin_frames > self.response_window_frames:
            msg = (
                f"`behavioral_response.target_bin_seconds` "
                f"({response_settings['target_bin_seconds']}) exceeds "
                f"`target_window_seconds` ({response_settings['target_window_seconds']}); "
                f"the time course must fit inside the window it resolves."
            )
            raise ValueError(msg)

        # Bin EDGES rather than a fixed width: 50 ms is 7.5 frames at 150 fps, so a
        # floored width of 7 would leave the last 5 frames of the window outside the
        # time course. Rounded edges tile the window exactly, at the cost of bins
        # alternating between 7 and 8 frames.
        self.n_response_bins = max(1, int(round(
            response_settings['target_window_seconds'] / response_settings['target_bin_seconds'])))
        self.response_bin_edges = np.linspace(
            0, self.response_window_frames, self.n_response_bins + 1).round().astype(int)

    def _resolve_response_column(self,
                                 session_df_columns: list[str],
                                 mouse_names: list[str],
                                 response_feature: str) -> str:
        """
        Names the raw feature column holding this session's response variable.

        The responder is identified by absolute slot index
        (``behavioral_response.response_mouse_index``), deliberately not by the
        relative ``self.`` / ``other.`` role keys, which can only be read against
        ``model_params.model_predictor_mouse_index``.

        Parameters
        ----------
        session_df_columns : list of str
            Column names of the raw per-session behavioral feature frame.
        mouse_names : list of str
            Ordered mouse track names for the session, slot 0 first.
        response_feature : str
            Feature whose column is wanted.

        Returns
        -------
        response_column : str
            The ``{mouse_name}.{response_feature}`` column name.
        """

        response_idx = self.modeling_settings['behavioral_response']['response_mouse_index']

        if not 0 <= response_idx < len(mouse_names):
            msg = (
                f"`behavioral_response.response_mouse_index` = {response_idx} is "
                f"outside the {len(mouse_names)} mouse slots available for this session."
            )
            raise ValueError(msg)

        response_column = f"{mouse_names[response_idx]}.{response_feature}"
        if response_column not in session_df_columns:
            available_mice = sorted({column.partition('.')[0]
                                     for column in session_df_columns
                                     if '-' not in column.partition('.')[0]})
            msg = (
                f"Response column '{response_column}' is absent from the session's "
                f"behavioral feature table. Mouse identity reaches this code from two "
                f"places: the tracking H5 gives '{mouse_names[response_idx]}', while the "
                f"feature CSV's columns give {available_mice}. When those disagree the "
                f"CSV labels are stale and must be corrected on disk -- the feature "
                f"VALUES are fine. Otherwise check "
                f"`behavioral_response.response_features` ('{response_feature}')."
            )
            raise KeyError(msg)

        return response_column

    def _response_target_values(self,
                                raw_values: np.ndarray,
                                anchor_frames: np.ndarray,
                                response_feature: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads the response over the forward window, as one mean and as bins.

        Both targets come from the same raw trace and the same anchors, so the
        single number and the time course describe exactly the same rows.

        The magnitude fold the predictors receive is applied here too when the
        response feature is a signed angle: ``sqrt(x**2 + eps**2)`` for a
        ``smooth_abs_features`` entry, ``|x|`` for an ``abs_features`` one. For a
        near-symmetric angle the signed mean is a small residual left after large
        opposing excursions cancel, and a signed target additionally breaks the
        Gamma likelihood, which needs strictly positive values.

        Parameters
        ----------
        raw_values : np.ndarray
            The response feature's per-frame trace, in native units.
        anchor_frames : np.ndarray
            Anchor frame indices.
        response_feature : str
            Which feature ``raw_values`` holds, for its fold and bounds.

        Returns
        -------
        window_means, bin_means : tuple of np.ndarray
            ``(n_anchors,)`` means over the whole forward window, and
            ``(n_anchors, n_bins)`` means over successive bins across it.
        """

        kinematic_settings = self.modeling_settings['kinematic_features']

        values = np.asarray(raw_values, dtype=float)
        if response_feature in kinematic_settings['smooth_abs_features']:
            epsilon = float(kinematic_settings['smooth_abs_features'][response_feature])
            values = np.sqrt(np.square(values) + epsilon ** 2)
        elif response_feature in kinematic_settings['abs_features']:
            values = np.abs(values)

        lower_bound, upper_bound = self.feature_boundaries[response_feature]

        window_means = forward_window_mean(
            values=values,
            anchor_frames=anchor_frames,
            window_frames=self.response_window_frames,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        bin_means = np.empty((anchor_frames.size, self.n_response_bins), dtype=float)
        for bin_index in range(self.n_response_bins):
            start_offset = int(self.response_bin_edges[bin_index])
            width = int(self.response_bin_edges[bin_index + 1] - start_offset)
            bin_means[:, bin_index] = forward_window_mean(
                values=values,
                anchor_frames=anchor_frames + start_offset,
                window_frames=width,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        return window_means, bin_means

    def _response_fold_label(self, response_feature: str) -> str:
        """
        Names which magnitude fold a response feature receives.

        Parameters
        ----------
        response_feature : str
            Feature to classify.

        Returns
        -------
        fold_label : str
            ``'smooth_abs'``, ``'abs'`` or ``'none'``.
        """

        kinematic_settings = self.modeling_settings['kinematic_features']
        if response_feature in kinematic_settings['smooth_abs_features']:
            return 'smooth_abs'
        if response_feature in kinematic_settings['abs_features']:
            return 'abs'
        return 'none'

    def _response_likelihood(self, response_feature: str) -> str:
        """
        Derives the likelihood family from the feature's post-fold support.

        This is a property of the feature, not a preference, so it is computed
        rather than configured: a signed feature simply cannot use a Gamma
        likelihood, and a settings key would let someone assert otherwise and
        silently discard every non-positive row.

        Both magnitude folds map onto ``[0, inf)``, so a folded feature is always
        Gamma-usable. An unfolded feature is Gamma-usable when its lower bound is
        already non-negative. Exact zeros are still possible for a feature bounded
        at zero; those rows are dropped at fit time and the count reported, rather
        than the whole feature being demoted for them.

        Parameters
        ----------
        response_feature : str
            Feature to classify.

        Returns
        -------
        likelihood : str
            ``'gamma'`` when the feature cannot be negative, else ``'gaussian'``.
        """

        if self._response_fold_label(response_feature) != 'none':
            return 'gamma'
        lower_bound, _ = self.feature_boundaries[response_feature]
        return 'gamma' if lower_bound >= 0.0 else 'gaussian'

    def extract_and_save_modeling_input_data(self) -> None:
        """
        Builds the anchor table and writes the modeling-input pickle.

        Loads behavior, finds the predictor mouse's bouts, harmonizes and z-scores
        the covariate features, places bout-offset and inter-bout-quiet anchors,
        summarizes each anchor's history, reads the response over the forward
        window, and writes one pickle holding a flat row table.

        Sessions are dropped when the **predictor** mouse has no bouts, not when
        the response mouse has none: with the male as predictor the female is the
        role-target and is usually silent, so the inherited check would discard
        almost the whole cohort.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        response_settings = self.modeling_settings['behavioral_response']
        response_features = list(response_settings['response_features'])
        response_idx = response_settings['response_mouse_index']
        predictor_mouse_idx = self.modeling_settings['model_params']['model_predictor_mouse_index']
        kin_settings = self.modeling_settings['kinematic_features']

        unknown = [f for f in response_features if f not in kin_settings['egocentric']]
        if unknown:
            msg = (
                f"`behavioral_response.response_features` names {unknown}, which are not in "
                f"`kinematic_features.egocentric` ({sorted(kin_settings['egocentric'])})."
            )
            raise ValueError(msg)

        likelihoods = {f: self._response_likelihood(f) for f in response_features}
        print(f"--- Extracting behavioral-response data for mouse slot {response_idx}, "
              f"{len(response_features)} feature(s) ---")
        for feature in response_features:
            print(f"      {feature:18s} fold={self._response_fold_label(feature):11s} "
                  f"likelihood={likelihoods[feature]}")

        txt_modeling_sessions = prepare_modeling_sessions(self.modeling_settings)

        mixture_model_idx = self.modeling_settings['model_params']['mixture_model_component_index']
        mixture_model_z = self.modeling_settings['model_params']['mixture_model_z_score']

        print("Loading behavioral feature data...")
        beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict = load_behavioral_feature_data(
            behavior_file_paths=txt_modeling_sessions,
            csv_sep=self.modeling_settings['io']['csv_separator'],
        )

        print("Finding male vocal bouts...")
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
            proportion_smoothing_sd=None,
            vocal_output_type=self.modeling_settings['vocal_features']['usv_predictor_type'],
            noise_vocal_categories=self.modeling_settings['vocal_features']['usv_noise_categories'],
            category_column=self.modeling_settings['vocal_features']['usv_category_column_name'],
            noise_column=self.modeling_settings['vocal_features']['usv_noise_column'],
        )

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

        print(f"Proceeding with {len(beh_feature_data_dict)} sessions after filtering "
              f"vocally-empty ones.")

        # The response is read from the RAW frame, before column selection and
        # before z-scoring: it must stay in native units for the Gamma likelihood
        # and it need not belong to the covariate zoo.
        raw_response_traces: dict[str, dict[str, np.ndarray]] = {}
        predictor_bout_onsets: dict[str, np.ndarray] = {}
        predictor_bout_durations: dict[str, np.ndarray] = {}
        processed_beh_feature_data_dict = {}
        session_roles: dict[str, tuple[str, str]] = {}
        target_mouse_idx = 1 - predictor_mouse_idx

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
            raw_response_traces[sess_id] = {}
            for feature in response_features:
                response_column = self._resolve_response_column(
                    session_df_columns=session_df_cols,
                    mouse_names=mouse_track_names_dict[sess_id],
                    response_feature=feature,
                )
                raw_response_traces[sess_id][feature] = (
                    session_df[response_column].to_numpy().astype(float))

            if p_name in bout_data_dict[sess_id]:
                predictor_bout_onsets[sess_id] = np.asarray(
                    bout_data_dict[sess_id][p_name]['bout_onsets'], dtype=float)
                predictor_bout_durations[sess_id] = np.asarray(
                    bout_data_dict[sess_id][p_name]['bout_durations'], dtype=float)

            columns_to_keep_session = select_kinematic_columns(
                session_df_columns=session_df_cols,
                target_name=t_name,
                predictor_name=p_name,
                kin_settings=kin_settings,
                predictor_idx=predictor_mouse_idx,
            )
            existing_cols = [c for c in columns_to_keep_session if c in session_df_cols]
            processed_beh_feature_data_dict[sess_id] = session_df.select(existing_cols)
            session_roles[sess_id] = (t_name, p_name)

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

        if not processed_beh_feature_data_dict:
            msg = (
                "No session survived loading and column selection, so there is nothing to "
                "extract. Check the session list, that each session has both a behavioral "
                "feature CSV and a USV summary, and that the predictor mouse actually "
                "vocalizes."
            )
            raise RuntimeError(msg)

        cohort_condition = derive_experimental_condition(self.modeling_settings)
        analysis_tag = f"behavioral_response_m{response_idx}_allfeatures"
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

        # Counts and the kept-feature list are placeholders here and are filled in by
        # `_save_extracted_data` once the anchors exist; the audits below only need
        # the cohort and temporal provenance.
        input_metadata = build_input_metadata(
            modeling_settings=self.modeling_settings,
            analysis_type='behavioral_response',
            analysis_tag=analysis_tag,
            pipeline_class=type(self).__name__,
            target_idx=target_mouse_idx,
            predictor_idx=predictor_mouse_idx,
            n_sessions_used=len(processed_beh_feature_data_dict),
            session_ids=sorted(processed_beh_feature_data_dict),
            n_events_per_session={},
            feature_zoo_full=derive_feature_zoo_full(self.modeling_settings),
            feature_zoo_kept=sorted(revised_behavioral_predictors),  # suffixes, not columns
            dyadic_engagement_features_used=list(kin_settings['dyadic_engagement']),
            dyadic_pose_symmetric_features_used=kin_settings['dyadic_pose_symmetric'],
            noise_vocal_categories_excluded=list(
                self.modeling_settings['vocal_features']['usv_noise_categories']),
            vocal_signal_columns_added=[],
            filter_history_seconds=float(response_settings['history_seconds']),
            filter_history_frames=int(self.response_history_frames),
            camera_sampling_rate_hz=derive_camera_fps_field(camera_fr_dict),
            ibi_thresholds=ibi_thresholds_md,
            analysis_specific={
                'response_mouse_index': response_idx,
                'derived_predictor_mouse_index': predictor_mouse_idx,
                'response_features': response_features,
                'response_folds': {f: self._response_fold_label(f) for f in response_features},
                'response_likelihoods': likelihoods,
                'target_window_seconds': response_settings['target_window_seconds'],
                'target_window_frames': int(self.response_window_frames),
                'target_bin_seconds': response_settings['target_bin_seconds'],
                'target_bin_frames': int(self.response_bin_frames),
                'target_bin_edges_frames': [int(e) for e in self.response_bin_edges],
                'n_response_bins': int(self.n_response_bins),
                'post_bout_silence_seconds': response_settings['post_bout_silence_seconds'],
                'covariate_summary_seconds': list(response_settings['covariate_summary_seconds']),
                'covariate_summary_frames': list(self.covariate_summary_frames),
                'duration_n_bins': int(response_settings['duration_n_bins']),
            },
        )

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
            save_dir=configure_path(self.modeling_settings['io']['save_directory']),
            pickle_basename=fname,
            input_metadata=input_metadata,
            onset_event_key='bout_onsets',
            precomputed_event_times=predictor_bout_onsets,
            precomputed_onset_times=predictor_bout_onsets,
        )

        anchor_rng = np.random.default_rng(self.modeling_settings['model_validation']['random_seed'])
        # Columns still carry raw mouse names, so each is mapped to a session-neutral
        # key -- `self.` for the responder, `other.` for the caller, bare for dyadic
        # features -- exactly as the inherited extractors do. Without this the
        # sessions share no column names at all.
        rows_covariates: list[dict[str, np.ndarray]] = []
        rows_target: list[np.ndarray] = []
        rows_bins: list[np.ndarray] = []
        rows_vocal: list[np.ndarray] = []
        rows_duration: list[np.ndarray] = []
        rows_session: list[np.ndarray] = []
        anchors_per_session: dict[str, dict[str, int]] = {}

        for sess_id, df in tqdm(processed_beh_feature_data_dict.items(), desc="Placing anchors"):
            if sess_id not in predictor_bout_onsets:
                continue
            camera_fps = float(camera_fr_dict[sess_id])
            n_frames = df.height

            vocal_frames, vocal_durations = bout_offset_anchors(
                bout_onsets=predictor_bout_onsets[sess_id],
                bout_durations=predictor_bout_durations[sess_id],
                camera_fps=camera_fps,
                n_frames=n_frames,
                history_frames=self.response_history_frames,
                lookahead_frames=self.response_window_frames,
                silence_seconds=response_settings['post_bout_silence_seconds'],
            )
            quiet_frames = inter_bout_quiet_anchors(
                bout_onsets=predictor_bout_onsets[sess_id],
                bout_durations=predictor_bout_durations[sess_id],
                camera_fps=camera_fps,
                n_frames=n_frames,
                history_frames=self.response_history_frames,
                lookahead_frames=self.response_window_frames,
                rng=anchor_rng,
            )
            anchor_frames = np.concatenate([vocal_frames, quiet_frames])
            if anchor_frames.size == 0:
                continue

            is_vocal = np.concatenate([
                np.ones(vocal_frames.size, dtype=float),
                np.zeros(quiet_frames.size, dtype=float),
            ])
            durations = np.concatenate([
                vocal_durations,
                np.full(quiet_frames.size, np.nan, dtype=float),
            ])

            # One target per feature. Rows are NOT filtered on any single feature's
            # validity here: different features fail on different anchors, so a
            # shared filter would silently shrink every feature to the intersection
            # of all of them. Non-finite and non-positive targets are dropped per
            # feature at fit time, where the count is reported.
            window_stack = np.empty((anchor_frames.size, len(response_features)), dtype=float)
            bin_stack = np.empty(
                (anchor_frames.size, len(response_features), self.n_response_bins), dtype=float)
            for feature_index, feature in enumerate(response_features):
                window_means, bin_means = self._response_target_values(
                    raw_values=raw_response_traces[sess_id][feature],
                    anchor_frames=anchor_frames,
                    response_feature=feature,
                )
                window_stack[:, feature_index] = window_means
                bin_stack[:, feature_index, :] = bin_means

            # An anchor with no usable target for ANY feature carries no information.
            usable = np.any(np.isfinite(window_stack), axis=1)
            if not np.any(usable):
                continue

            anchor_frames = anchor_frames[usable]
            t_name, p_name = session_roles[sess_id]
            session_covariates: dict[str, np.ndarray] = {}
            for column in df.columns:
                base_feature = column.split('.')[-1]
                if '-' in base_feature:
                    generic_key = base_feature
                elif column.startswith(f"{t_name}."):
                    generic_key = f"self.{base_feature}"
                elif column.startswith(f"{p_name}."):
                    generic_key = f"other.{base_feature}"
                else:
                    generic_key = base_feature
                session_covariates[generic_key] = summarise_history(
                    values=df[column].to_numpy().astype(float),
                    anchor_frames=anchor_frames,
                    summary_frames=self.covariate_summary_frames,
                )

            rows_covariates.append(session_covariates)
            rows_target.append(window_stack[usable])
            rows_bins.append(bin_stack[usable])
            rows_vocal.append(is_vocal[usable])
            rows_duration.append(durations[usable])
            rows_session.append(np.full(int(usable.sum()), sess_id, dtype=object))
            anchors_per_session[sess_id] = {
                'vocal': int(is_vocal[usable].sum()),
                'quiet': int((1.0 - is_vocal[usable]).sum()),
            }

        if not rows_target:
            msg = (
                "No anchor survived placement in any session. Check "
                "`behavioral_response.post_bout_silence_seconds` and "
                "`history_seconds` against the session durations, and that the "
                "predictor mouse produces bouts."
            )
            raise RuntimeError(msg)

        # Only keys present in every session enter the design, so it stays rectangular.
        shared_keys = sorted(set.intersection(*(set(d) for d in rows_covariates)))
        dropped_keys = sorted(set.union(*(set(d) for d in rows_covariates)) - set(shared_keys))
        if dropped_keys:
            print(f"[warn] {len(dropped_keys)} feature(s) absent from at least one session "
                  f"and dropped from the design: {dropped_keys}")
        covariate_matrix = np.vstack([
            np.column_stack([session_covariates[key] for key in shared_keys])
            for session_covariates in rows_covariates
        ])
        summary_labels = [
            f"{key}__mean_{seconds:g}s"
            for key in shared_keys
            for seconds in response_settings['covariate_summary_seconds']
        ]

        self._save_extracted_data(
            covariates=covariate_matrix,
            covariate_labels=summary_labels,
            target=np.vstack(rows_target),
            target_bins=np.concatenate(rows_bins, axis=0),
            response_features=response_features,
            response_likelihoods=likelihoods,
            is_vocal=np.concatenate(rows_vocal),
            bout_duration=np.concatenate(rows_duration),
            session_ids=np.concatenate(rows_session),
            input_metadata=input_metadata,
            anchors_per_session=anchors_per_session,
            fname=fname,
        )

    def _save_extracted_data(self,
                             covariates: np.ndarray,
                             covariate_labels: list[str],
                             target: np.ndarray,
                             target_bins: np.ndarray,
                             response_features: list[str],
                             response_likelihoods: dict[str, str],
                             is_vocal: np.ndarray,
                             bout_duration: np.ndarray,
                             session_ids: np.ndarray,
                             input_metadata: dict[str, Any],
                             anchors_per_session: dict[str, dict[str, int]],
                             fname: str) -> None:
        """
        Validates row alignment, prints the summary and publishes the pickle.

        Every array must carry the same number of rows in the same order, because
        they are positionally paired into one design; a mismatch is raised rather
        than warned about, since a silently misaligned covariate would produce a
        plausible but meaningless contrast.

        Parameters
        ----------
        covariates : np.ndarray
            ``(n_rows, n_covariates)`` history summaries.
        covariate_labels : list of str
            Column names for ``covariates``.
        target : np.ndarray
            ``(n_rows, n_features)`` response averaged over the whole forward window.
        target_bins : np.ndarray
            ``(n_rows, n_features, n_bins)`` response averaged over successive bins.
        response_features : list of str
            Column order of ``target`` and the middle axis of ``target_bins``.
        response_likelihoods : dict
            Derived likelihood family per response feature.
        is_vocal : np.ndarray
            ``(n_rows,)`` 1.0 at bout offsets, 0.0 at inter-bout quiet anchors.
        bout_duration : np.ndarray
            ``(n_rows,)`` bout duration in seconds; ``nan`` on quiet rows.
        session_ids : np.ndarray
            ``(n_rows,)`` session identifier, the clustering unit for inference.
        input_metadata : dict
            Provenance block, completed here with the row counts.
        anchors_per_session : dict
            Per-session vocal / quiet anchor counts.
        fname : str
            Basename of the pickle to publish under ``io.save_directory``.

        Returns
        -------
        None
        """

        n_rows = int(target.shape[0])
        if target.shape[1] != len(response_features):
            msg = (
                f"`target` has {target.shape[1]} feature columns but "
                f"{len(response_features)} features were requested."
            )
            raise ValueError(msg)
        for label, array in (('covariates', covariates), ('target_bins', target_bins),
                             ('is_vocal', is_vocal), ('bout_duration', bout_duration),
                             ('session_ids', session_ids)):
            if array.shape[0] != n_rows:
                msg = (
                    f"Row-count mismatch: `target` has {n_rows} rows but `{label}` has "
                    f"{array.shape[0]}; the design would be misaligned."
                )
                raise ValueError(msg)
        if covariates.shape[1] != len(covariate_labels):
            msg = (
                f"`covariates` has {covariates.shape[1]} columns but "
                f"{len(covariate_labels)} labels were built for it."
            )
            raise ValueError(msg)

        n_vocal = int(np.sum(is_vocal > 0.0))
        n_quiet = n_rows - n_vocal
        print("=" * 70)
        print(f"  > Sessions contributing:        {len(anchors_per_session)}")
        print(f"  > Anchors (vocal / quiet):      {n_vocal} / {n_quiet}")
        print(f"  > Covariate columns:            {covariates.shape[1]}")
        print(f"  > Response bins:                {target_bins.shape[2]}")
        print(f"  > Bout duration (s), median:    {np.nanmedian(bout_duration):.3f}")
        print(f"  > Response features:            {len(response_features)}")
        for feature_index, feature in enumerate(response_features):
            column = target[:, feature_index]
            finite = int(np.sum(np.isfinite(column)))
            positive = int(np.sum(np.isfinite(column) & (column > 0.0)))
            note = ('' if response_likelihoods[feature] == 'gaussian'
                    else f", {finite - positive} non-positive dropped at fit")
            print(f"      {feature:18s} {response_likelihoods[feature]:9s} "
                  f"{finite}/{n_rows} finite{note}")
        print("=" * 70)

        contributing_sessions = sorted(anchors_per_session)
        input_metadata['session_ids'] = contributing_sessions
        input_metadata['n_sessions_used'] = len(contributing_sessions)
        input_metadata['n_events_per_session'] = anchors_per_session
        input_metadata['n_rows'] = n_rows
        input_metadata['n_vocal_rows'] = n_vocal
        input_metadata['n_quiet_rows'] = n_quiet
        input_metadata['analysis_specific']['covariate_labels'] = list(covariate_labels)
        input_metadata['analysis_specific']['response_features'] = list(response_features)
        input_metadata['analysis_specific']['response_likelihoods'] = dict(response_likelihoods)

        artifact = inject_metadata(
            {
                'covariates': covariates,
                'covariate_labels': list(covariate_labels),
                'target': target,
                'target_bins': target_bins,
                'response_features': list(response_features),
                'response_likelihoods': dict(response_likelihoods),
                'is_vocal': is_vocal,
                'bout_duration': bout_duration,
                'session_ids': session_ids,
            },
            _input_metadata=input_metadata,
        )

        # Through `configure_path`, not `Path` directly: the settings carry the
        # cluster's `/mnt/...` root, which does not exist on macOS, and writing
        # there fails only at the very end -- after the whole cohort has loaded.
        save_dir = Path(configure_path(self.modeling_settings['io']['save_directory']))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / fname
        with atomic_output_path(save_path) as temporary_path:
            with Path(temporary_path).open('wb') as handle:
                pickle.dump(artifact, handle)
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Success. Results saved to: {save_path}")


def duration_tercile_labels(bout_duration: np.ndarray,
                            is_vocal: np.ndarray,
                            n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits vocal rows into equal-count duration bands.

    Cut points come from the vocal rows only -- quiet rows have no duration -- so
    the bands hold equal numbers of BOUTS rather than equal numbers of anchors.

    Parameters
    ----------
    bout_duration : np.ndarray
        Per-row bout duration in seconds; ``nan`` on quiet rows.
    is_vocal : np.ndarray
        Per-row 1.0 at bout offsets, 0.0 at quiet anchors.
    n_bins : int
        Number of equal-count bands.

    Returns
    -------
    band_index, edges : tuple of np.ndarray
        Per-row band index (``-1`` on quiet rows), and the ``n_bins + 1`` duration
        cut points defining the bands.

    Raises
    ------
    ValueError
        If the vocal rows carry fewer distinct durations than requested bands, so
        the bands could not be populated.
    """

    vocal_mask = is_vocal > 0.0
    durations = bout_duration[vocal_mask]
    if np.unique(durations).size < n_bins:
        msg = (
            f"Only {np.unique(durations).size} distinct bout durations are available, "
            f"which cannot fill {n_bins} equal-count bands; lower "
            f"`behavioral_response.duration_n_bins`."
        )
        raise ValueError(msg)

    edges = np.quantile(durations, np.linspace(0.0, 1.0, n_bins + 1))
    # `np.digitize` on the interior edges puts each duration in [0, n_bins - 1].
    band_index = np.full(bout_duration.shape[0], -1, dtype=int)
    band_index[vocal_mask] = np.clip(np.digitize(durations, edges[1:-1]), 0, n_bins - 1)
    return band_index, edges


def build_design_matrix(covariates: np.ndarray,
                        is_vocal: np.ndarray,
                        band_index: np.ndarray,
                        n_bins: int,
                        covariate_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Assembles the design: intercept, one step per duration band, then covariates.

    The vocal terms are ``vocal x band`` indicators rather than a ``vocal`` main
    effect plus a duration slope. Each band's coefficient is then read directly
    as that band's step against silence, and question (1) is the joint statement
    of those steps while question (2) is how they differ from one another. Coding
    quiet rows as duration zero was rejected: it would assert that "no bout at
    all" lies on the same line as "a very short bout", which is the assumption
    under test.

    Any row carrying a non-finite covariate is left in place here; callers drop
    such rows before fitting.

    Parameters
    ----------
    covariates : np.ndarray
        ``(n_rows, n_covariates)`` pre-anchor history summaries.
    is_vocal : np.ndarray
        Per-row 1.0 at bout offsets, 0.0 at quiet anchors.
    band_index : np.ndarray
        Per-row duration band, ``-1`` on quiet rows.
    n_bins : int
        Number of duration bands.
    covariate_labels : list of str
        Column names for ``covariates``.

    Returns
    -------
    design, labels : tuple
        ``(n_rows, 1 + n_bins + n_covariates)`` design matrix and its column names.
    """

    n_rows = covariates.shape[0]
    band_columns = np.zeros((n_rows, n_bins), dtype=float)
    for band in range(n_bins):
        band_columns[:, band] = ((is_vocal > 0.0) & (band_index == band)).astype(float)

    design = np.column_stack([np.ones(n_rows), band_columns, covariates])
    labels = (['intercept']
              + [f'vocal_duration_band_{band}' for band in range(n_bins)]
              + list(covariate_labels))
    return design, labels


def fit_contrast(target: np.ndarray,
                 design: np.ndarray,
                 labels: list[str],
                 session_ids: np.ndarray,
                 likelihood: str) -> dict[str, Any]:
    """
    Fits one GLM with session-clustered standard errors.

    Rows carrying a non-finite target or covariate are dropped before fitting;
    the count is reported so a silently shrinking sample is visible.

    Parameters
    ----------
    target : np.ndarray
        ``(n_rows,)`` response in native units, strictly positive for ``'gamma'``.
    design : np.ndarray
        ``(n_rows, n_terms)`` design matrix including the intercept.
    labels : list of str
        Column names for ``design``.
    session_ids : np.ndarray
        ``(n_rows,)`` clustering unit for the robust covariance.
    likelihood : str
        ``'gamma'`` (log link) or ``'gaussian'`` (identity on the raw response).

    Returns
    -------
    fit_results : dict
        Per-term ``coefficient``, ``std_error``, ``z``, ``p_value`` and 95%
        interval, the fitted row count, the number dropped, and
        ``non_finite_by_term`` attributing the loss to individual columns.

    Raises
    ------
    ValueError
        If ``likelihood`` is unrecognised, no row survives the finite filter, or
        the surviving design is rank-deficient.
    """

    if likelihood not in ('gamma', 'gaussian'):
        msg = f"`likelihood` must be 'gamma' or 'gaussian'; got '{likelihood}'."
        raise ValueError(msg)

    usable = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    if likelihood == 'gamma':
        usable &= target > 0.0
    n_dropped = int((~usable).sum())
    # A single non-finite covariate drops the whole row, so one bad feature can
    # decimate the sample without saying so. Attribute the loss per column.
    non_finite_by_term = {
        labels[i]: int(np.sum(~np.isfinite(design[:, i])))
        for i in range(design.shape[1])
        if np.any(~np.isfinite(design[:, i]))
    }
    if not np.any(usable):
        worst = sorted(non_finite_by_term.items(), key=lambda kv: -kv[1])[:5]
        msg = (
            f"No row survives the finite-value filter, so the contrast cannot be fitted. "
            f"{int(np.sum(~np.isfinite(target)))} of {target.size} rows have a non-finite "
            f"target; the covariates losing the most rows are {worst or 'none'}."
        )
        raise ValueError(msg)

    # A rank-deficient design surfaces from statsmodels as a bare
    # `LinAlgError: Singular matrix` raised inside the sandwich estimator, which
    # says nothing about which columns caused it. The usual causes here are two
    # covariate summaries of a slow feature being near-identical, or a duration
    # band left empty after the finite filter.
    fitted_design = design[usable]
    rank = int(np.linalg.matrix_rank(fitted_design))
    if rank < fitted_design.shape[1]:
        constant_columns = [labels[i] for i in range(fitted_design.shape[1])
                            if np.ptp(fitted_design[:, i]) == 0.0 and labels[i] != 'intercept']
        msg = (
            f"Design is rank-deficient: rank {rank} for {fitted_design.shape[1]} terms on "
            f"{fitted_design.shape[0]} rows, so the coefficients are not identifiable. "
            f"Constant columns: {constant_columns or 'none'}. Usual causes are covariate "
            f"summaries that are near-identical for a slow feature, a duration band left "
            f"empty after the finite filter, or too few rows for the term count."
        )
        raise ValueError(msg)

    family = (sm.families.Gamma(link=sm.families.links.Log()) if likelihood == 'gamma'
              else sm.families.Gaussian())
    model = sm.GLM(target[usable], fitted_design, family=family)
    fitted = model.fit(cov_type='cluster',
                       cov_kwds={'groups': session_ids[usable], 'use_correction': True})

    intervals = fitted.conf_int()
    terms = {}
    for position, name in enumerate(labels):
        terms[name] = {
            'coefficient': float(fitted.params[position]),
            'std_error': float(fitted.bse[position]),
            'z': float(fitted.tvalues[position]),
            'p_value': float(fitted.pvalues[position]),
            'ci_low': float(intervals[position, 0]),
            'ci_high': float(intervals[position, 1]),
        }
    return {
        'terms': terms,
        'n_rows_fitted': int(usable.sum()),
        'n_rows_dropped': n_dropped,
        'non_finite_by_term': non_finite_by_term,
        'n_sessions': int(np.unique(session_ids[usable]).size),
        'likelihood': likelihood,
    }


def behavioral_response_contrast(input_pickle_path: str | Path,
                                 output_directory: str | Path,
                                 settings_path: str | Path | None = None) -> dict[str, Any]:
    """
    Runs the vocal-versus-silence contrast and its time course, and saves them.

    Fits the same design twice over: once against the response averaged over the
    whole forward window, giving the headline numbers, and once per time bin,
    giving the adjusted time course.

    Parameters
    ----------
    input_pickle_path : str or pathlib.Path
        Anchor table written by ``BehavioralResponsePipeline``.
    output_directory : str or pathlib.Path
        Directory to publish the results pickle into.
    settings_path : str or pathlib.Path, optional
        Recorded in the run header for provenance; the analysis knobs themselves
        come from the artifact's ``_input_metadata``, so the extraction and the
        fit can never disagree about them.

    Returns
    -------
    results : dict
        ``per_feature`` (each with its ``window`` fit, ``time_course`` and derived
        ``likelihood``), ``duration_edges``, ``term_labels`` and the provenance
        block.
    """

    input_path = Path(input_pickle_path)
    with input_path.open('rb') as handle:
        artifact = pickle.load(handle)

    metadata = artifact['_input_metadata']
    analysis = metadata['analysis_specific']
    response_features = list(artifact['response_features'])
    response_likelihoods = dict(artifact['response_likelihoods'])
    n_duration_bins = int(analysis['duration_n_bins'])

    covariates = np.asarray(artifact['covariates'], dtype=float)
    covariate_labels = list(artifact['covariate_labels'])
    target = np.asarray(artifact['target'], dtype=float)
    target_bins = np.asarray(artifact['target_bins'], dtype=float)
    is_vocal = np.asarray(artifact['is_vocal'], dtype=float)
    bout_duration = np.asarray(artifact['bout_duration'], dtype=float)
    session_ids = np.asarray(artifact['session_ids'])

    print(format_run_header(
        task='BEHAVIORAL_RESPONSE_CONTRAST',
        engine='glm',
        feature=f'{len(response_features)} response feature(s)',
        split_strategy='cluster-robust (session)',
        n_splits=int(np.unique(session_ids).size),
        input_files={'input data': str(input_path), 'settings': str(settings_path)},
        output_directory=str(output_directory),
    ))

    # The design is identical for every feature and every time bin -- only the
    # target changes -- so it is built once. That is also what makes the time
    # course readable as one contrast resolved over time rather than a series of
    # separate analyses.
    band_index, duration_edges = duration_tercile_labels(
        bout_duration=bout_duration, is_vocal=is_vocal, n_bins=n_duration_bins)
    design, labels = build_design_matrix(
        covariates=covariates, is_vocal=is_vocal, band_index=band_index,
        n_bins=n_duration_bins, covariate_labels=covariate_labels)

    per_feature: dict[str, Any] = {}
    for feature_index, feature in enumerate(response_features):
        likelihood = response_likelihoods[feature]
        window_fit = fit_contrast(target=target[:, feature_index], design=design, labels=labels,
                                  session_ids=session_ids, likelihood=likelihood)
        for band in range(n_duration_bins):
            term = window_fit['terms'][f'vocal_duration_band_{band}']
            print(format_selection_step(
                'Contrast',
                feature=f'{feature} | band {band} '
                        f'[{duration_edges[band]:.2f}-{duration_edges[band + 1]:.2f}s]',
                metrics={'beta': term['coefficient'], 'se': term['std_error'],
                         'p': term['p_value']},
                decision='SIG' if term['p_value'] < 0.05 else 'ns',
            ))

        time_course = []
        for bin_index in range(target_bins.shape[2]):
            bin_fit = fit_contrast(target=target_bins[:, feature_index, bin_index],
                                   design=design, labels=labels,
                                   session_ids=session_ids, likelihood=likelihood)
            bin_fit['bin_index'] = bin_index
            time_course.append(bin_fit)

        per_feature[feature] = {'window': window_fit, 'time_course': time_course,
                                'likelihood': likelihood}

    results = {
        'per_feature': per_feature,
        'response_features': response_features,
        'response_likelihoods': response_likelihoods,
        'duration_edges': duration_edges,
        'term_labels': labels,
        'n_duration_bins': n_duration_bins,
        '_input_metadata': metadata,
    }

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = output_path / f"behavioral_response_contrast_{timestamp}.pkl"
    with atomic_output_path(save_path) as temporary_path:
        with Path(temporary_path).open('wb') as handle:
            pickle.dump(results, handle)

    print(format_run_summary(
        label=f"behavioral response contrast, {len(response_features)} feature(s) / "
              f"{window_fit['n_sessions']} sessions",
        metrics_by_strategy={
            feature: {
                'beta': per_feature[feature]['window']['terms']['vocal_duration_band_0']['coefficient'],
                'p': per_feature[feature]['window']['terms']['vocal_duration_band_0']['p_value'],
            }
            for feature in response_features
        },
        out_path=str(save_path),
    ))
    return results
