"""
@author: bartulem
Per-unit figures of the neural modeling results: one page per unit drawn from its merged result
file (``<unit_uid>.pkl``, written by the neural modeling dispatchers), plus the per-session trace
pages for a unit whose kinematic encoding holds on held-out data.

Every number a panel reports is read from the stored sections; what the storage rule leaves out
(event assemblies, designs, predictions) is rebuilt from the session data and cached, and each
rebuild is checked against the stored values before it is drawn. A result file whose recorded
settings hash differs from the settings in use is skipped rather than drawn.

Layout for a unit with a model:
    top row     forward selection on the representative fold's training sessions; held-out QUIET and pooled
                VOCAL (transfer) scores, mean across folds +/- SE, representative fold marked, beside its null
    bottom row  the fitted quiet filter of every selected feature (resampling SD band), the whole model's
                tuning on quiet versus vocal frames, and each selected feature's own tuning

A tuning curve is the spike rate in bins of a predictor, divided by the rate expected from each frame's
session mean in that regime -- so 1 means no modulation, and quiet (~2% of frames spike) and vocal (~30%) share
one axis. Its slope is what the transfer tests: rising in both regimes means the tuning carries over, falling on
vocal frames means it flips. Bands are block-bootstrap SEs over contiguous history-length blocks.

A unit with no model gets the screen instead: every feature's score per fold, with its p-value against the
Bonferroni bar.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

from usv_playpen.analyses.compute_behavioral_features import FeatureZoo
from usv_playpen.modeling.load_input_files import load_behavioral_feature_data
from usv_playpen.modeling.modeling_metadata import compute_settings_sha256
from usv_playpen.neural_modeling.behaviour_control_model import (
    apply_behaviour_control,
    category_control_evidence,
    load_control_model,
)
from usv_playpen.neural_modeling.deviance_metrics import (
    contiguous_blocks,
    finite_mean,
    frozen_model_score,
    newton_logistic,
)
from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    focal_vocal_frames,
    read_fold_artifact,
)
from usv_playpen.neural_modeling.neural_category_decoding import (
    category_behaviour_tilts,
    category_codes,
    category_context,
    category_posterior,
    tilted_category_prior,
)
from usv_playpen.neural_modeling.neural_design_assembly import (
    assemble_unit_sessions,
    assemble_unit_vocal_events,
    filter_vocalizations,
    lagged_design,
    load_session_usvs,
    load_unit_spike_frames,
    spike_labels_at_frames,
    vocal_span_frames,
)
from usv_playpen.neural_modeling.neural_nested_decoding import (
    behaviour_priors,
    nested_design,
    nested_settings,
    reduced_model_features,
    target_section,
)
from usv_playpen.neural_modeling.neural_vocal_decoding import (
    decoding_context,
    fourier_basis,
    torus_grid,
    torus_occupancy,
)
from usv_playpen.neural_modeling.quiet_to_vocal_transfer import transfer_frames
from usv_playpen.neural_modeling.shift_null_inference import sample_circular_shift
from usv_playpen.os_utils import configure_path, load_qlvm_category_bundle
from usv_playpen.visualizations.auxiliary_plot_functions import draw_category_outlines
from usv_playpen.visualizations.plot_style import apply_plot_style

SETTINGS_DIRECTORY = pathlib.Path(__file__).resolve().parent.parent / "_parameter_settings"
NEURAL_SETTINGS_PATH = SETTINGS_DIRECTORY / "neural_modeling_settings.json"


@dataclasses.dataclass(frozen=True)
class FigurePaths:
    """
    Description
    -----------
    Where a figure run reads and writes: the neural modeling results (the merged ``<unit_uid>.pkl``
    files and the encoding's ``<unit_id>_fold<i>.npz`` artifacts), the session data tree the rebuilds
    read, and the cache the rebuilds are kept in.

    The cache is keyed to the settings hash by :func:`plot_neural_modeling_units`, so a rebuild made
    under other settings is never read back.

    Parameters
    ----------
    results_directory (pathlib.Path)
        Directory holding the unit result files.
    data_root (pathlib.Path)
        Root of the session data tree.
    cache_directory (pathlib.Path)
        Directory the rebuilt per-unit arrays are cached in.

    Returns
    -------
    None
    """

    results_directory: pathlib.Path
    data_root: pathlib.Path
    cache_directory: pathlib.Path


SELF_SEX, PARTNER_SEX = "male", "female"

SELF_COLOR = "#9AC0CD"
OTHER_COLOR = "#FF6347"
DYADIC_COLOR = "#5A6470"
REJECTED_COLOR = "#C0C0C0"
NULL_COLOR = "#A0A0A0"
QUIET_COLOR = "#1D3557"
VOCAL_COLOR = "#2A9D8F"
STA_COLOR = "#B0B0B0"
REFERENCE_COLOR = "#808080"
TEXT_COLOR = "#202020"

STA_CHUNK_ROWS = 20000
N_TUNING_BINS = 10
OCCURRENCE_MAX_SPIKES = 10
# The predicted-versus-observed rate rows. RATE_SMOOTHING_MS is the scoring knob (user-ruled: start at
# 30-50 ms, the spikes being per frame) and RATE_DRAW_SMOOTHING_MS the drawing one (user-ruled 2026-10-17:
# at 50 ms the observed trace is single spikes that hide the prediction; both are reported); both traces are smoothed with the same kernel restricted to the scored frame class, so
# smoothing never mixes quiet and vocal frames. The inset shows a fixed-length stretch chosen by RULE -- the
# quiet window at the median concordance -- never by eye.
RATE_SMOOTHING_MS = 50.0
RATE_DRAW_SMOOTHING_MS = 250.0
RATE_INSET_SECONDS = 20.0
RATE_INSET_MIN_QUIET_FRACTION = 0.7
RATE_NULL_DRAWS = 200
RATE_NULL_SEED = 0
# The thin pad, in inches, around the traces page's content (the page is sized to its content, user-ruled
# 2026-10-20), and how many sessions share a column.
SESSION_PAGE_MARGIN = 0.2
SESSIONS_PER_PAGE = 3
# Sessions beyond the first column's three go to a second column to its right, same geometry, the page
# widened by one column (user-ruled 2026-10-20).
SESSION_COLUMNS = 2
SESSION_COLUMN_GAP = 0.85
# The whole-session trace in inches (a third smaller in both directions than the four-per-page layout,
# user-ruled 2026-10-18); its best-period panels sit underneath at the same height.
SESSION_TRACE_SIZE = (4.6, 0.95)
# Best periods drawn under each session trace, sharing exactly its width (user-ruled 2026-10-18: four).
SESSION_ZOOM_COUNT = 4
# The emitter colors of the project's visualization settings: the male's bout windows and the unassigned
# vocalizations are drawn in them (user-ruled 2026-10-18).
with (SETTINGS_DIRECTORY / "visualizations_settings.json").open() as _visual:
    _VISUAL_SETTINGS = json.load(_visual)
MALE_COLOR = _VISUAL_SETTINGS["male_colors"][0]
UNASSIGNED_COLOR = _VISUAL_SETTINGS["unassigned_colors"][0]
# One colour per acoustic category, in category order (category 1 first), from the QLVM figures.
CATEGORY_COLORS = _VISUAL_SETTINGS["qlvm_figures"]["category_colors"]
# One marker per session wherever sessions are told apart by symbol, in the order of the unit's sessions.
SESSION_MARKERS = ["o", "s", "^", "D", "v", "P"]
# Borders between USV-category regions on the torus scatter.
CATEGORY_BORDER_COLOR = "#000000"
CATEGORY_BORDER_WIDTH = 0.5
# Largest allowed difference between the rebuilt and the stored per-USV category gain.
CATEGORY_REBUILD_TOLERANCE = 1e-9
# How much the confusion matrix is enlarged beyond its column, into the gaps on either side.
CONFUSION_GROWTH = 1.32
# Space between the page title and the first row of panels, in inches.
TITLE_CLEARANCE_INCHES = 0.7
# One title size for every panel in the figure, so no panel reads as more important than its neighbours.
PANEL_TITLE_SIZE = 7
# The width of the TWO-feature figure (2.6 in per column: two filters, the model's tuning, two tunings).
# Every panel on every unit keeps the size it has at this width; a unit with more features gets a wider
# figure, never narrower panels or stretched rows.
REFERENCE_FIGURE_WIDTH = 2.6 * 5
# Where a filter panel's spike-triggered-average inset sits, per unit and feature. An override rather than
# a heuristic because the corner is an aesthetic call the data cannot settle: on the VTA unit's speed
# filter the upper right and upper left are BOTH completely free, so any automatic rule picks one of them
# arbitrarily. Anything not listed falls back to `freest_corner`.
# Where the session legend sits in the vocalizing panel, per unit; anything unlisted uses the upper left.
OCCURRENCE_LEGEND_AT = {"181316_0_20250919_imec0_cl0633_ch278_good": "center right"}
INSET_CORNERS = {
    "181316_0_20250919_imec1_cl0401_ch220_good": {"head-head": "lower left", "allo_yaw-head": "lower left"},
    "181316_0_20250923_imec1_cl0499_ch322_good": {"self.speed": "upper left", "head-head": "lower left"},
}
N_GATE_BINS = 8
ADDITIVE_COLOR = "#9A9A9A"
GATE_SIGNIFICANT_COLOR = "#8B1A1A"
GATE_BELOW_COLOR = "#F4A582"
HEATMAP_SHADING_POINTS = 400
INSET_CROWDING_FRACTION = 0.15
INSET_MARGIN = 0.12
INSET_RIGHT_INSET = 0.22
INSET_TOP_INSET = 0.06
# Which ruler every gating panel uses. The panels do not name it -- both are the distance between the
# animals, and the label would read as a distinction the figure is not making. `nose` is the model's own feature, the one the gating statistic was
# computed on; `head` describes proximity without the head-orientation confound. One or the other, never a
# mixture, or the panels describe different quantities side by side.
GATING_RULER = "head"
SEQUENTIAL_CMAP = "inferno"
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 0


def subplot_span() -> float:
    """
    Description
    -----------
    The fraction of a figure's width that its subplots span under the current rcParams -- the ruler that
    turns a width in inches into a GridSpec share, so the layout can be stated in inches.

    Parameters
    ----------
    None

    Returns
    -------
    span (float)
        ``figure.subplot.right - figure.subplot.left``.
    """

    return float(plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"])


def gridspec_widths(total: float, ratios: list, wspace: float) -> tuple[list, float]:
    """
    Description
    -----------
    The panel widths and the gap that a GridSpec with these width ratios and ``wspace`` lays out across
    ``total`` -- matplotlib's own arithmetic, so a row rebuilt from them in absolute units is exactly the
    row the ratios draw at that width.

    Parameters
    ----------
    total (float)
        The row's width (inches here).
    ratios (list)
        The GridSpec ``width_ratios``.
    wspace (float)
        The GridSpec ``wspace``, a fraction of the average panel width.

    Returns
    -------
    widths (list)
        One width per panel, in the units of ``total``.
    gap (float)
        The space between neighbouring panels, in the same units.
    """

    n = len(ratios)
    cell = total / (n + wspace * (n - 1))
    scale = cell * n / sum(ratios)
    return [ratio * scale for ratio in ratios], wspace * cell


def row_cells(cell, widths: list, gaps: list, total: float) -> list:
    """
    Description
    -----------
    Slots of ABSOLUTE width inside a row: ``widths`` inches each, separated by ``gaps`` inches, left-aligned
    across ``total`` inches with whatever remains left empty. This is what keeps a panel the same size on
    every unit: a row that spans the figure stretches with it, a row stated in inches does not.

    Parameters
    ----------
    cell (matplotlib.gridspec.SubplotSpec)
        The row's slot, ``total`` inches wide.
    widths (list)
        Panel widths in inches, left to right.
    gaps (list)
        The gap after each panel but the last, in inches (one fewer than ``widths``).
    total (float)
        The slot's width in inches.

    Returns
    -------
    cells (list)
        One SubplotSpec per panel.
    """

    ratios = [widths[0]]
    for gap, width in zip(gaps, widths[1:], strict=True):
        ratios.extend([gap, width])
    slack = total - sum(ratios)
    if slack > 1e-6:
        ratios.append(slack)
    grid = GridSpecFromSubplotSpec(1, len(ratios), subplot_spec=cell, width_ratios=ratios, wspace=0.0)
    return [grid[2 * k] for k in range(len(widths))]


def bottom_row_layout(reference: float, layout_features: int, n_features: int, show_gating: bool) -> dict:
    """
    Description
    -----------
    Absolute widths, in inches, for the bottom row -- the filters, the model's tuning and each feature's
    tuning, or the filters and the gating panels -- at the panel sizes of the two-feature reference figure,
    so that extra features add panels of the same size instead of shrinking every panel to fit the row.

    Parameters
    ----------
    reference (float)
        The reference figure's subplot span in inches (the two-feature layout).
    layout_features (int)
        Filter and tuning slots laid out: at least two, so a one-feature unit keeps the two-feature panel size.
    n_features (int)
        Filters actually drawn; sizes the filter block of the gating variant, which has no empty slot.
    show_gating (bool)
        Whether the gating panels replace the tuning panels.

    Returns
    -------
    layout (dict)
        ``blocks`` (the row's cell widths), ``gaps`` (between them), ``filters`` (slot widths inside the
        filter block), ``filter_gap``, ``tunings``, ``tuning_gap`` and ``total`` (the row's width).
    """

    if show_gating:
        widths, gap = gridspec_widths(reference, [1.0, 1.15, 1.0, 1.0, 1.1], 0.6)
        filter_gap = 0.35 * widths[0]
        filters = [widths[0]] * n_features
        blocks = [sum(filters) + filter_gap * (n_features - 1), *widths[1:]]
        gaps = [gap] * 4
        tunings, tuning_gap = [], 0.0
    else:
        widths, gap = gridspec_widths(reference, [2.0, 1.1, 2.0], 0.28)
        slot, filter_gap = gridspec_widths(widths[0], [1.0, 1.0], 0.35)
        tuning_slot, tuning_gap = gridspec_widths(widths[2], [1.0, 1.0], 0.3)
        filters = [slot[0]] * layout_features
        tunings = [tuning_slot[0]] * layout_features
        blocks = [sum(filters) + filter_gap * (layout_features - 1), widths[1],
                  sum(tunings) + tuning_gap * (layout_features - 1)]
        gaps = [gap] * 2
    return {"blocks": blocks, "gaps": gaps, "filters": filters, "filter_gap": filter_gap, "tunings": tunings,
            "tuning_gap": tuning_gap, "total": sum(blocks) + sum(gaps)}


def feature_color(feature_name: str) -> str:
    """
    Description
    -----------
    The self / partner / dyadic colour of a kinematic feature, matching the behavioural selection figures.

    Parameters
    ----------
    feature_name (str)
        Canonical feature name, e.g. ``self.speed``, ``other.neck_elevation`` or ``nose-nose``.

    Returns
    -------
    color (str)
        Hex colour.
    """

    if feature_name.startswith("self."):
        return SELF_COLOR
    if feature_name.startswith("other."):
        return OTHER_COLOR
    return DYADIC_COLOR


def screened_only(encoding: dict) -> bool:
    """
    Description
    -----------
    Whether this unit's panels describe the best SCREENED feature rather than a selected model.

    Parameters
    ----------
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.

    Returns
    -------
    flag (bool)
        True when no fold selected anything and the folds fitted their best screened feature instead.
    """

    return "best_screened_only" in encoding and bool(encoding["best_screened_only"])


def kinematic_encoding_holds(encoding: dict, settings: dict) -> bool:
    """
    Description
    -----------
    Whether the unit's kinematic encoding is significant on HELD-OUT data -- the quiet test of the
    representative model against its shift null, at the uncorrected level the pipeline's per-unit reading
    uses (``significance.fdr_q``). This, not whether some fold's screen let a feature through, decides how
    the unit is drawn: a feature that passes one fold's screen inside its training sessions and then fails
    the held-out session is not a kinematic feature of the unit.

    Parameters
    ----------
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.

    Returns
    -------
    holds (bool)
        True when the held-out quiet p is below the level.
    """

    return bool(np.isfinite(encoding["quiet_p"]) and encoding["quiet_p"] < settings["significance"]["fdr_q"])


def feature_label(feature_name: str) -> str:
    """
    Description
    -----------
    The repository's display label for a feature, from ``FeatureZoo.resolve_feature_label``.

    Parameters
    ----------
    feature_name (str)
        Canonical feature name.

    Returns
    -------
    label (str)
        e.g. ``male speed``, ``partner distance``, ``self-partner yaw``.
    """

    return FeatureZoo.resolve_feature_label(feature_name, SELF_SEX, PARTNER_SEX)


def standard_error(values) -> float:
    """
    Description
    -----------
    Standard error of the mean of the finite values, ``std(ddof=1) / sqrt(n)``; NaN below two values.

    Parameters
    ----------
    values (array-like)
        Values.

    Returns
    -------
    se (float)
        Standard error.
    """

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size >= 2 else np.nan


def predictor_outputs(per_session: dict, frames_by_session: dict, feature_indices: list, filters: np.ndarray,
                      intercept: float, n_lags: int) -> dict:
    """
    Description
    -----------
    The frozen model's output and each feature's own filter output at the given frames, with labels.

    ``feature_output[f]`` is feature f's lagged history weighted by its own filter, so the model's linear
    predictor is the sum of the feature outputs plus the intercept.

    Parameters
    ----------
    per_session (dict)
        Assembled session data from ``assemble_unit_sessions``.
    frames_by_session (dict)
        Session id to the frames to evaluate.
    feature_indices (list)
        The model's feature columns.
    filters (np.ndarray)
        ``(n_features, n_lags)`` fitted filter.
    intercept (float)
        Fitted intercept.
    n_lags (int)
        History length in frames.

    Returns
    -------
    outputs (dict)
        ``eta``, ``feature_output`` ``(n_features, n_frames)``, ``labels``, ``session_index``, ``block`` (a
        contiguous history-length block id, unique across sessions), ``session_ids``.
    """

    etas, features, labels, sessions, blocks = [], [], [], [], []
    block_offset = 0
    session_ids = list(frames_by_session)
    for index, session_id in enumerate(session_ids):
        session = per_session[session_id]
        frames = np.asarray(frames_by_session[session_id], dtype=np.int64)
        if frames.size == 0:
            continue
        outputs = np.stack([lagged_design(session["feature_time_series"][:, [column]], frames, n_lags) @ filters[k]
                            for k, column in enumerate(feature_indices)])
        features.append(outputs)
        etas.append(outputs.sum(axis=0) + intercept)
        labels.append(spike_labels_at_frames(session["spike_frames"], frames, session["n_frames"]))
        sessions.append(np.full(frames.size, index))
        blocks.append(frames // n_lags + block_offset)
        block_offset += session["n_frames"] // n_lags + 1
    return {"eta": np.concatenate(etas), "feature_output": np.concatenate(features, axis=1),
            "labels": np.concatenate(labels).astype(np.float64), "session_index": np.concatenate(sessions),
            "block": np.concatenate(blocks), "session_ids": session_ids}


def block_resampled_sta(per_session: dict, session_ids: list, feature_indices: list, n_lags: int, rng) -> dict:
    """
    Description
    -----------
    Spike-triggered average on quiet anchors with a block-bootstrap SD band.

    The STA is the mean history before a spike minus the mean history over all anchors, lag by lag, in SDs of
    the feature. Its uncertainty is resampled over CONTIGUOUS blocks one history window long, never over
    individual spikes: frames a few tens of milliseconds apart share almost their whole history, so an
    independent resample of spikes reports a precision the data do not have -- measured at 17.8x too narrow
    for the filter band, which faces the same autocorrelation.

    Each block's triggered sum, anchor sum and both counts are accumulated once; a draw is then a weighted
    sum over resampled blocks, so the design is never rebuilt per draw.

    Parameters
    ----------
    per_session (dict)
        Assembled session data.
    session_ids (list)
        Sessions to pool over.
    feature_indices (list)
        The model's feature columns.
    n_lags (int)
        History length in frames, and the resampling block length.
    rng (np.random.Generator)
        Bootstrap generator.

    Returns
    -------
    sta (dict)
        ``difference`` and ``sd``, each ``(n_features, n_lags)``, plus ``n_spikes`` and ``n_anchors``.
    """

    columns = list(feature_indices)
    triggered, anchored, spike_counts, anchor_counts = [], [], [], []
    for session_id in session_ids:
        session = per_session[session_id]
        anchors = np.sort(session["quiet"])
        labels = spike_labels_at_frames(session["spike_frames"], anchors, session["n_frames"])
        blocks, block_index = np.unique(anchors // n_lags, return_inverse=True)
        session_triggered = np.zeros((blocks.size, len(columns), n_lags))
        session_anchored = np.zeros((blocks.size, len(columns), n_lags))
        session_spikes = np.zeros(blocks.size)
        session_anchors = np.zeros(blocks.size)
        for start in range(0, anchors.size, STA_CHUNK_ROWS):
            stop = min(start + STA_CHUNK_ROWS, anchors.size)
            design = lagged_design(session["feature_time_series"][:, columns], anchors[start:stop], n_lags)
            design = design.reshape(stop - start, len(columns), n_lags)
            block_slice = block_index[start:stop]
            label_slice = labels[start:stop]
            np.add.at(session_anchored, block_slice, design)
            np.add.at(session_triggered, block_slice, design * label_slice[:, None, None])
            np.add.at(session_anchors, block_slice, 1.0)
            np.add.at(session_spikes, block_slice, label_slice)
        triggered.append(session_triggered)
        anchored.append(session_anchored)
        spike_counts.append(session_spikes)
        anchor_counts.append(session_anchors)

    def difference(chosen_per_session):
        """The pooled STA for one selection of blocks per session."""
        spikes = sum(count[chosen].sum() for count, chosen in zip(spike_counts, chosen_per_session, strict=True))
        frames = sum(count[chosen].sum() for count, chosen in zip(anchor_counts, chosen_per_session, strict=True))
        spike_sum = sum(block[chosen].sum(axis=0) for block, chosen in zip(triggered, chosen_per_session, strict=True))
        anchor_sum = sum(block[chosen].sum(axis=0) for block, chosen in zip(anchored, chosen_per_session, strict=True))
        return spike_sum / max(spikes, 1.0) - anchor_sum / max(frames, 1.0), spikes, frames

    everything = [np.arange(count.size) for count in spike_counts]
    observed, n_spikes, n_anchors = difference(everything)
    draws = np.stack([difference([rng.choice(index, size=index.size, replace=True) for index in everything])[0]
                      for _ in range(N_BOOTSTRAP)])
    return {"difference": observed, "sd": draws.std(axis=0, ddof=1), "n_spikes": float(n_spikes),
            "n_anchors": float(n_anchors)}


def transfer_block(encoding: dict, settings: dict) -> dict:
    """
    Description
    -----------
    The transfer settings an artifact was scored under, so the figure rebuilds the SAME frames.

    Artifacts written before the 2026-10-14 switch carry no ``transfer_universe`` and were scored on the
    USVs' own spans; the shipped settings now say bout window. Rebuilding a span artifact on the window
    (or the reverse) would make the stored and rebuilt transfer disagree by construction.

    Parameters
    ----------
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.

    Returns
    -------
    block (dict)
        The ``kinematic_encoding.transfer`` block with ``universe`` set to the artifact's own.
    """

    universe = encoding["transfer_universe"] if "transfer_universe" in encoding else "usv_span"
    return {**settings["kinematic_encoding"]["transfer"], "universe": universe}


def tuning_data(unit_uid: str, paths: FigurePaths, identity: dict, encoding: dict, settings: dict) -> dict:
    """
    Description
    -----------
    Model and per-feature outputs on the representative fold's held-out quiet anchors and on every vocal
    frame, cached beside the figures.

    Quiet anchors come from the HELD-OUT session only, so nothing plotted on the quiet side was seen by the
    fit. Vocal frames come from every vocal session, since no vocal frame enters any model; that is the set
    the transfer is scored on. The vocal model output is checked against the stored transfer predictions.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    data (dict)
        ``quiet`` and ``vocal``, each as returned by :func:`predictor_outputs`.
    """

    cache_path = paths.cache_directory / f"{unit_uid}_tuning_{transfer_block(encoding, settings)['universe']}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)

    kinematic = settings["kinematic_encoding"]
    vocalization = settings["vocalization_settings"]
    per_session = assemble_unit_sessions(identity, str(paths.data_root), settings["kinematic_features"],
                                         kinematic["history_pre_seconds"], kinematic["clean_post_seconds"],
                                         vocalization["clean_against"],
                                         vocalization)
    n_lags = int(encoding["n_lags"])
    fps = float(encoding["fps"])
    features = [int(index) for index in encoding["selected_indices"]]
    filters = np.asarray(encoding["filter"]).reshape(len(features), n_lags)
    intercept = float(encoding["filter_intercept"])

    held_out = encoding["representative_session"]
    quiet = predictor_outputs(per_session, {held_out: per_session[held_out]["quiet"]}, features, filters,
                              intercept, n_lags)
    vocal_frames = {}
    block = transfer_block(encoding, settings)
    for session_id in encoding["transfer_session_ids"]:
        frames, pointer, _ = focal_vocal_frames(per_session[session_id], session_id, str(paths.data_root),
                                                identity["mouse_id"], fps, n_lags, vocalization)
        vocal_frames[session_id] = transfer_frames(frames, pointer, fps, per_session[session_id]["n_frames"],
                                                   n_lags, per_session[session_id]["quiet"],
                                                   settings["kinematic_encoding"]["clean_post_seconds"], block)
    vocal = predictor_outputs(per_session, vocal_frames, features, filters, intercept, n_lags)

    stored = np.concatenate([np.asarray(entry["eta"]) for entry in encoding["transfer_per_session"]])
    if stored.size != vocal["eta"].size:
        msg = f"{unit_uid}: rebuilt {vocal['eta'].size} vocal frames, stored transfer has {stored.size}"
        raise ValueError(msg)
    print(f"  {unit_uid}: vocal model output vs stored transfer, max |diff| {np.max(np.abs(stored - vocal['eta'])):.2e}")

    # The STA is pooled over the sessions the representative model was FITTED on, matching the stored one,
    # so the panel shows the raw counterpart of the filter drawn beside it.
    sta = block_resampled_sta(per_session, [s for s in identity["courtship_sessions"] if s != held_out],
                              features, n_lags, np.random.default_rng(BOOTSTRAP_SEED + 11))
    stored_sta = np.asarray(encoding["spike_triggered_average"]["difference"])
    print(f"  {unit_uid}: STA vs stored, max |diff| {np.max(np.abs(stored_sta - sta['difference'])):.2e}")

    data = {"quiet": quiet, "vocal": vocal, "sta": sta}
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(data, cache_file)
    return data


def crossfitted_probability(eta: np.ndarray, labels: np.ndarray, group: np.ndarray, n_blocks: int,
                            steps: int) -> np.ndarray:
    """
    Description
    -----------
    Per-frame spike probability of a frozen predictor, calibrated the way the tested statistic calibrates
    it: per-group intercepts and one shared scale on the standardised predictor, fitted for every
    contiguous block on the OTHER blocks of all groups.

    Parameters
    ----------
    eta (np.ndarray)
        Frozen predictor per frame (time order within each group).
    labels (np.ndarray)
        0/1 spike per frame.
    group (np.ndarray)
        Integer group (session) per frame.
    n_blocks (int)
        Contiguous blocks per group.
    steps (int)
        Maximum Newton iterations.

    Returns
    -------
    probability (np.ndarray)
        Cross-fitted spike probability per frame.
    """

    z = (eta - eta.mean()) / eta.std()
    groups = np.unique(group)
    position = np.searchsorted(groups, group)
    design = np.column_stack([z, (position[:, None] == np.arange(groups.size)[None, :]).astype(np.float64)])
    blocks = contiguous_blocks(position, n_blocks)
    probability = np.empty(eta.size)
    for block in range(n_blocks):
        train, test = blocks != block, blocks == block
        beta = newton_logistic(design[train], labels[train], np.zeros(design.shape[1]), steps)
        probability[test] = 1.0 / (1.0 + np.exp(-(design[test] @ beta)))
    return probability


def rate_trace_data(unit_uid: str, paths: FigurePaths, identity: dict, encoding: dict, settings: dict) -> dict:
    """
    Description
    -----------
    The frozen representative model's per-frame spike probability, calibrated exactly as the tested
    statistics calibrate it (``kinematic_encoding.calibration``), for the two traces of the rate row.

    QUIET: the held-out session only -- the one session this model never saw -- with its baseline and scale
    cross-fitted on that session's quiet frames, as the quiet score does. VOCAL: every session's transfer
    frames (the bout window), which no fit ever touched, with per-session intercepts and one shared scale
    cross-fitted across all sessions' blocks, as the pooled transfer does (at frame rather than bin
    resolution, for drawing). Cached beside the figures.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    data (dict)
        ``fps``, ``held_out``, ``quiet`` (``n_frames``, ``spikes``, ``frames``, ``probability`` over the
        held-out session) and ``vocal`` (per session in order: ``session_id``, ``frames``, ``spikes`` at
        them, ``probability``, and the session's frame count).
    """

    calibration = settings["kinematic_encoding"]["calibration"]
    cache_path = paths.cache_directory / f"{unit_uid}_rate_trace_v2_{calibration['scheme']}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)
    kinematic = settings["kinematic_encoding"]
    vocalization = settings["vocalization_settings"]
    per_session = assemble_unit_sessions(identity, str(paths.data_root), settings["kinematic_features"],
                                         kinematic["history_pre_seconds"], kinematic["clean_post_seconds"],
                                         vocalization["clean_against"],
                                         vocalization)
    n_lags = int(encoding["n_lags"])
    fps = float(encoding["fps"])
    features = [int(index) for index in encoding["selected_indices"]]
    weights = np.asarray(encoding["filter"], dtype=np.float64).ravel()
    n_blocks = int(calibration["n_blocks"])
    steps = kinematic["solver"]["calibration_steps"]
    held_out = encoding["representative_session"]

    def predictor(session, frames):
        """The frozen filter's linear predictor at the frames (no intercept: calibration supplies it)."""
        return np.concatenate([lagged_design(session["feature_time_series"][:, features], frames[lo:lo + 20000],
                                             n_lags) @ weights for lo in range(0, frames.size, 20000)])

    def spike_vector(session):
        """0/1 spike per frame over the whole session."""
        spikes = np.zeros(int(session["n_frames"]))
        spike_frames = np.asarray(session["spike_frames"], dtype=np.int64)
        spikes[spike_frames[(spike_frames >= 0) & (spike_frames < spikes.size)]] = 1.0
        return spikes

    session = per_session[held_out]
    spikes = spike_vector(session)
    quiet = np.sort(np.asarray(session["quiet"], dtype=np.int64))
    quiet_probability = crossfitted_probability(predictor(session, quiet), spikes[quiet],
                                                np.zeros(quiet.size, dtype=np.int64), n_blocks, steps)
    vocal, etas, labels, groups = [], [], [], []
    for index, session_id in enumerate(identity["courtship_sessions"]):
        other = per_session[session_id]
        span, pointer, _ = focal_vocal_frames(other, session_id, str(paths.data_root), identity["mouse_id"], fps, n_lags,
                                              vocalization)
        frames = np.sort(transfer_frames(span, pointer, fps, int(other["n_frames"]), n_lags,
                                         np.asarray(other["quiet"]),
                                         settings["kinematic_encoding"]["clean_post_seconds"],
                                         transfer_block(encoding, settings)))
        session_spikes = spike_vector(other)[frames]
        vocal.append({"session_id": session_id, "frames": frames, "spikes": session_spikes,
                      "n_frames": int(other["n_frames"])})
        etas.append(predictor(other, frames))
        labels.append(session_spikes)
        groups.append(np.full(frames.size, index))
    vocal_probability = crossfitted_probability(np.concatenate(etas), np.concatenate(labels), np.concatenate(groups),
                                                n_blocks, steps)
    start = 0
    for entry in vocal:
        entry["probability"] = vocal_probability[start:start + entry["frames"].size]
        start += entry["frames"].size
    data = {"fps": fps, "held_out": held_out,
            "quiet": {"n_frames": int(session["n_frames"]), "spikes": spikes, "frames": quiet,
                      "probability": quiet_probability},
            "vocal": vocal}
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(data, cache_file)
    return data


def masked_smooth(values: np.ndarray, mask: np.ndarray, sigma_frames: float) -> np.ndarray:
    """
    Description
    -----------
    Gaussian smoothing restricted to the frames in ``mask`` (normalised convolution), NaN elsewhere, so a
    trace never borrows from frames outside its class (quiet from vocal, one bout window from the gap
    before it).

    Parameters
    ----------
    values (np.ndarray)
        Per-frame values.
    mask (np.ndarray)
        Boolean per frame: which frames belong to the class.
    sigma_frames (float)
        Kernel SD in frames.

    Returns
    -------
    smoothed (np.ndarray)
        Smoothed values on the class's frames, NaN elsewhere.
    """

    weight = gaussian_filter1d(mask.astype(np.float64), sigma_frames)
    total = gaussian_filter1d(np.where(mask, values, 0.0), sigma_frames)
    out = np.full(values.size, np.nan)
    out[mask] = total[mask] / np.clip(weight[mask], 1e-12, None)
    return out


def overlap_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """
    Description
    -----------
    How closely two rate traces lie on top of each other: bias (mean predicted minus observed, Hz), the slope
    of observed on predicted (1 when the swings match), Pearson r, and Lin's concordance -- 1 only when the
    lines coincide, penalised for offset and scale as well as scatter.

    Parameters
    ----------
    predicted (np.ndarray)
        Predicted rate (Hz) at the scored frames.
    observed (np.ndarray)
        Observed rate (Hz) at the same frames.

    Returns
    -------
    metrics (dict)
        ``bias``, ``slope``, ``r``, ``ccc``.
    """

    mp, mo = predicted.mean(), observed.mean()
    vp, vo = predicted.var(), observed.var()
    covariance = np.mean((predicted - mp) * (observed - mo))
    return {"bias": float(mp - mo), "slope": float(covariance / vp) if vp > 0 else float("nan"),
            "r": float(covariance / np.sqrt(vp * vo)) if vp > 0 and vo > 0 else float("nan"),
            "ccc": float(2.0 * covariance / (vp + vo + (mp - mo) ** 2))}


def class_traces(spikes: np.ndarray, frames: np.ndarray, probability: np.ndarray, n_frames: int, fps: float,
                 sigma_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Observed and predicted rate (Hz) over a session, each smoothed with the same kernel restricted to the
    class's frames.

    Parameters
    ----------
    spikes (np.ndarray)
        0/1 spike per frame at ``frames`` (or over the whole session when its size is ``n_frames``).
    frames (np.ndarray)
        The class's frames.
    probability (np.ndarray)
        Predicted spike probability at ``frames``.
    n_frames (int)
        Session frame count.
    fps (float)
        Frame rate.
    sigma_ms (float)
        Kernel SD in milliseconds.

    Returns
    -------
    mask (np.ndarray)
        Boolean class membership per session frame.
    observed (np.ndarray)
        Smoothed observed rate, NaN off-class.
    predicted (np.ndarray)
        Smoothed predicted rate, NaN off-class.
    """

    mask = np.zeros(n_frames, dtype=bool)
    mask[frames] = True
    full_spikes = np.zeros(n_frames)
    full_spikes[frames] = spikes if spikes.size == frames.size else spikes[frames]
    full_probability = np.zeros(n_frames)
    full_probability[frames] = probability
    sigma = sigma_ms / 1000.0 * fps
    return (mask, masked_smooth(full_spikes * fps, mask, sigma), masked_smooth(full_probability * fps, mask, sigma))


def overlap_with_null(spikes: np.ndarray, frames: np.ndarray, probability: np.ndarray, n_frames: int, fps: float,
                      sigma_ms: float, shifts: list) -> dict:
    """
    Description
    -----------
    :func:`overlap_metrics` on one session's class frames at one smoothing, with the concordance's
    circular-shift null (the WHOLE spike train shifted, then relabelled at the class's frames).

    Parameters
    ----------
    spikes (np.ndarray)
        0/1 spike per frame over the WHOLE session.
    frames (np.ndarray)
        The class's frames.
    probability (np.ndarray)
        Predicted spike probability at ``frames``.
    n_frames (int)
        Session frame count.
    fps (float)
        Frame rate.
    sigma_ms (float)
        Kernel SD in milliseconds.
    shifts (list)
        Circular shifts in FRAMES.

    Returns
    -------
    result (dict)
        The metrics plus ``null_ccc`` (array) and ``p_ccc``.
    """

    mask, observed, predicted = class_traces(spikes, frames, probability, n_frames, fps, sigma_ms)
    metrics = overlap_metrics(predicted[mask], observed[mask])
    null = []
    for shift in shifts:
        _mask, shifted, _predicted = class_traces(np.roll(spikes, shift), frames, probability, n_frames, fps, sigma_ms)
        null.append(overlap_metrics(predicted[mask], shifted[mask])["ccc"])
    null = np.asarray(null)
    return {**metrics, "null_ccc": null, "p_ccc": float((1 + np.sum(null >= metrics["ccc"])) / (1 + null.size))}


def draw_rate_traces(quiet_axis, vocal_axis, data: dict, settings: dict) -> None:
    """
    Description
    -----------
    The frozen kinematic model's predicted firing rate against the observed one.

    TOP: the whole held-out session, quiet frames scored, with an inset on the quiet window at the MEDIAN
    concordance over all windows with enough quiet frames (a rule, so the stretch is typical rather than the
    best). BOTTOM: every session's vocal (bout-window) frames, concatenated with the gaps between windows
    removed, one session after another -- the frames the pooled transfer scores. Both traces are what the
    tested statistics score (see :func:`rate_trace_data`), never a refit. The drawing uses a kernel of
    ``RATE_DRAW_SMOOTHING_MS``; overlap is reported at both that and ``RATE_SMOOTHING_MS``, each against a
    circular shift of the spike train: concordance (which an offset or a scale mismatch cannot fake) with its
    null, Pearson r, bias and slope.

    Parameters
    ----------
    quiet_axis (matplotlib.axes.Axes)
        Axis for the held-out session's quiet trace.
    vocal_axis (matplotlib.axes.Axes)
        Axis for the concatenated vocal frames of every session.
    data (dict)
        Output of :func:`rate_trace_data`.
    settings (dict)
        The run's settings.

    Returns
    -------
    None
    """

    fps = data["fps"]
    guard = float(settings["null"]["shuffle_guard_seconds"])
    rng = np.random.default_rng(RATE_NULL_SEED)

    def summary(label, spikes, frames, probability, n_frames):
        """One text line: concordance at both smoothings with its null, plus r, bias and slope at the coarse one."""
        shifts = [int(np.rint(sample_circular_shift(rng, n_frames, fps, guard) * fps)) for _ in range(RATE_NULL_DRAWS)]
        fine = overlap_with_null(spikes, frames, probability, n_frames, fps, RATE_SMOOTHING_MS, shifts)
        coarse = overlap_with_null(spikes, frames, probability, n_frames, fps, RATE_DRAW_SMOOTHING_MS, shifts)
        return (f"{label}: CCC {coarse['ccc']:.2f} (null {coarse['null_ccc'].mean():+.2f}, p {coarse['p_ccc']:.3f}) "
                f"@{RATE_DRAW_SMOOTHING_MS:.0f} ms | {fine['ccc']:.2f} (null {fine['null_ccc'].mean():+.2f}, "
                f"p {fine['p_ccc']:.3f}) @{RATE_SMOOTHING_MS:.0f} ms | r {coarse['r']:.2f} | bias "
                f"{coarse['bias']:+.2f} Hz | slope {coarse['slope']:.2f}")

    # Top: the held-out session's quiet frames.
    quiet = data["quiet"]
    n_frames = quiet["n_frames"]
    minutes = np.arange(n_frames) / fps / 60.0
    mask, observed, predicted = class_traces(quiet["spikes"], quiet["frames"], quiet["probability"], n_frames, fps,
                                             RATE_DRAW_SMOOTHING_MS)
    quiet_axis.plot(minutes[::3], observed[::3], color=TEXT_COLOR, linewidth=0.5)
    quiet_axis.plot(minutes[::3], predicted[::3], color=QUIET_COLOR, linewidth=0.9)
    quiet_axis.set_xlim(0.0, minutes[-1])
    quiet_axis.set_xlabel("time in held-out session (min)")
    quiet_axis.set_ylabel(f"firing rate (Hz, {RATE_DRAW_SMOOTHING_MS:.0f} ms kernel)")
    quiet_axis.set_title(f"Quiet frames, held-out session {data['held_out'].split('_')[-1]}: frozen kinematic model "
                         f"(baseline and scale cross-fitted) vs observed", loc="left", fontsize=PANEL_TITLE_SIZE)
    quiet_axis.legend(handles=[Line2D([0], [0], color=TEXT_COLOR, linewidth=1.0, label="observed"),
                               Line2D([0], [0], color=QUIET_COLOR, linewidth=1.2, label="predicted")],
                      loc="upper left", fontsize=5.5, frameon=False, ncol=2)
    quiet_axis.text(0.995, 0.98, summary("quiet", quiet["spikes"], quiet["frames"], quiet["probability"], n_frames),
                    transform=quiet_axis.transAxes, ha="right", va="top", fontsize=5.0, color=TEXT_COLOR,
                    bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    width = int(np.rint(RATE_INSET_SECONDS * fps))
    candidates = []
    for start in range(0, n_frames - width + 1, width):
        window = slice(start, start + width)
        if mask[window].mean() < RATE_INSET_MIN_QUIET_FRACTION:
            continue
        keep = mask[window]
        candidates.append((overlap_metrics(predicted[window][keep], observed[window][keep])["ccc"], start))
    if candidates:
        candidates.sort()
        ccc, start = candidates[len(candidates) // 2]
        window = slice(start, start + width)
        inset = quiet_axis.inset_axes([0.62, 0.36, 0.37, 0.42])
        seconds = np.arange(width) / fps
        inset.plot(seconds, observed[window], color=TEXT_COLOR, linewidth=0.7)
        inset.plot(seconds, predicted[window], color=QUIET_COLOR, linewidth=1.1)
        inset.set_title(f"{RATE_INSET_SECONDS:.0f} s from {start / fps / 60:.1f} min (median-concordance quiet "
                        f"window, CCC {ccc:.2f})", fontsize=5.0, loc="left")
        inset.tick_params(labelsize=4.5, length=1.5)
        inset.set_xlabel("s", fontsize=5.0, labelpad=1)
        # The inset runs in seconds and the panel in minutes, so the window is marked by a span, not a zoom box.
        quiet_axis.axvspan(start / fps / 60.0, (start + width) / fps / 60.0, color="#E8E8E8", zorder=0)

    # Bottom: every session's vocal frames, concatenated.
    offset, lines, boundaries = 0.0, [], []
    for entry in data["vocal"]:
        frames = entry["frames"]
        mask, observed, predicted = class_traces(entry["spikes"], frames, entry["probability"], entry["n_frames"], fps,
                                                 RATE_DRAW_SMOOTHING_MS)
        x = offset + np.arange(frames.size) / fps
        vocal_axis.plot(x, observed[frames], color=TEXT_COLOR, linewidth=0.5)
        vocal_axis.plot(x, predicted[frames], color=VOCAL_COLOR, linewidth=0.9)
        full_spikes = np.zeros(entry["n_frames"])
        full_spikes[frames] = entry["spikes"]
        lines.append(summary(entry["session_id"].split("_")[-1], full_spikes, frames, entry["probability"],
                             entry["n_frames"]))
        boundaries.append((offset, entry["session_id"].split("_")[-1]))
        offset = x[-1] + 1.0 / fps
    for left, label in boundaries:
        vocal_axis.axvline(left, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
        vocal_axis.text(left, 1.0, f" {label}", transform=vocal_axis.get_xaxis_transform(), fontsize=5.5,
                        va="bottom", ha="left", color=TEXT_COLOR)
    vocal_axis.set_xlim(0.0, offset)
    vocal_axis.set_xlabel("time within bout windows, sessions concatenated (s)")
    vocal_axis.set_ylabel(f"firing rate (Hz, {RATE_DRAW_SMOOTHING_MS:.0f} ms kernel)")
    vocal_axis.set_title("Vocal frames (bout windows) of every session: same frozen model, per-session baselines and "
                         "shared scale cross-fitted, as the transfer test scores them", loc="left",
                         fontsize=PANEL_TITLE_SIZE, pad=10)
    vocal_axis.legend(handles=[Line2D([0], [0], color=TEXT_COLOR, linewidth=1.0, label="observed"),
                               Line2D([0], [0], color=VOCAL_COLOR, linewidth=1.2, label="predicted")],
                      loc="upper left", fontsize=5.5, frameon=False, ncol=2, bbox_to_anchor=(0.0, 0.93))
    vocal_axis.text(0.995, 0.97, "\n".join(lines), transform=vocal_axis.transAxes, ha="right", va="top", fontsize=5.0,
                    color=TEXT_COLOR, bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})


def session_trace_data(unit_uid: str, paths: FigurePaths, identity: dict, settings: dict) -> dict:
    """
    Description
    -----------
    Every session's predicted spike probability over ALL its frames, each session from the fold model that
    held it out -- so the whole trace, quiet and vocal alike, is out of sample: that model's filters never
    saw the session. Frames are split into three classes -- quiet anchors, the transfer's bout-window frames,
    and every other frame with a full history -- and each class is calibrated the way the tested statistics
    calibrate a frozen model (``kinematic_encoding.calibration``): its own baseline and scale on the
    standardised predictor, cross-fitted on the class's other contiguous blocks. The level therefore follows
    the regime (the unit fires several times faster inside bouts), while the moment-to-moment modulation is
    the kinematic filter's alone. Cached beside the figures.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    data (dict)
        ``fps`` and ``sessions``: one entry per session in order, with ``session_id``, ``features`` (the
        fold model's), ``n_frames``, ``spikes`` (0/1 per frame), ``probability`` (NaN before the first full
        history), ``vocal`` (boolean per frame: inside the bout window) and ``unassigned`` (boolean per frame:
        inside a vocalization no animal was credited with).
    """

    calibration = settings["kinematic_encoding"]["calibration"]
    cache_path = paths.cache_directory / f"{unit_uid}_session_traces_v2_{calibration['scheme']}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)
    kinematic = settings["kinematic_encoding"]
    vocalization = settings["vocalization_settings"]
    per_session = assemble_unit_sessions(identity, str(paths.data_root), settings["kinematic_features"],
                                         kinematic["history_pre_seconds"], kinematic["clean_post_seconds"],
                                         vocalization["clean_against"],
                                         vocalization)
    n_blocks = int(calibration["n_blocks"])
    steps = kinematic["solver"]["calibration_steps"]
    sessions = []
    for fold_index in range(len(identity["courtship_sessions"])):
        fold = stored_fold(paths, identity, fold_index)
        if bool(fold["no_model"]) and not ("best_screened" in fold and bool(fold["best_screened"])):
            continue
        session_id = str(fold["test_id"])
        session = per_session[session_id]
        features = [int(index) for index in fold["selected_indices"]]
        weights = np.asarray(fold["coefficients"], dtype=np.float64).ravel()
        n_lags = weights.size // len(features)
        n_frames = int(session["n_frames"])
        fps = float(session["fps"])
        spikes = np.zeros(n_frames)
        spike_frames = np.asarray(session["spike_frames"], dtype=np.int64)
        spikes[spike_frames[(spike_frames >= 0) & (spike_frames < n_frames)]] = 1.0
        scored = np.arange(n_lags - 1, n_frames)
        eta = np.concatenate([lagged_design(session["feature_time_series"][:, features], scored[lo:lo + 20000], n_lags)
                              @ weights for lo in range(0, scored.size, 20000)])
        span, pointer, _ = focal_vocal_frames(session, session_id, str(paths.data_root), identity["mouse_id"], fps, n_lags,
                                              vocalization)
        quiet = np.asarray(session["quiet"], dtype=np.int64)
        vocal_frames = transfer_frames(span, pointer, fps, n_frames, n_lags, quiet,
                                       settings["kinematic_encoding"]["clean_post_seconds"], transfer_block(
            {"transfer_universe": settings["kinematic_encoding"]["transfer"]["universe"]}, settings))
        label = np.full(n_frames, 2)
        label[quiet] = 0
        label[vocal_frames] = 1
        probability = np.full(n_frames, np.nan)
        for code in (0, 1, 2):
            rows = scored[label[scored] == code]
            if rows.size < 10 * n_blocks or spikes[rows].sum() == 0:
                continue
            probability[rows] = crossfitted_probability(eta[rows - scored[0]], spikes[rows],
                                                        np.zeros(rows.size, dtype=np.int64), n_blocks, steps)
        vocal = np.zeros(n_frames, dtype=bool)
        vocal[vocal_frames] = True
        # Vocalizations no animal was credited with, over the same analysed set the focal ones come from.
        table = filter_vocalizations(load_session_usvs(str(paths.data_root), session_id), vocalization, "analysed")
        unassigned_table = table.filter(table["emitter"].is_null())
        unassigned = np.zeros(n_frames, dtype=bool)
        if unassigned_table.height:
            _kept, unassigned_frames, _pointer = vocal_span_frames(unassigned_table["start"].to_numpy(),
                                                                   unassigned_table["stop"].to_numpy(), fps,
                                                                   n_frames, 1)
            unassigned[unassigned_frames] = True
        sessions.append({"session_id": session_id, "features": [str(name) for name in fold["selected"]],
                         "n_frames": n_frames, "spikes": spikes, "probability": probability, "vocal": vocal,
                         "unassigned": unassigned, "fps": fps})
    data = {"fps": sessions[0]["fps"], "sessions": sessions}
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(data, cache_file)
    return data


def session_overlap(entry: dict, settings: dict, rng) -> dict:
    """
    Description
    -----------
    One session's smoothed observed and predicted traces and every overlap number the figure reports (in its
    metrics table, never on the axes): concordance over the whole session at the drawing kernel and at
    ``RATE_SMOOTHING_MS``, each against a circular shift of the spike train, plus r, bias and slope, and
    concordance within the frames outside bouts and within the bout windows.

    Parameters
    ----------
    entry (dict)
        One session of :func:`session_trace_data`.
    settings (dict)
        The run's settings.
    rng (np.random.Generator)
        Generator for the null's shifts.

    Returns
    -------
    result (dict)
        ``observed`` and ``predicted`` (drawing kernel, NaN where no prediction), ``has`` (frames carrying a
        prediction) and ``metrics`` (a flat dict for the table).
    """

    fps, n_frames = entry["fps"], entry["n_frames"]
    has = np.isfinite(entry["probability"])
    probability = np.where(has, entry["probability"], 0.0)

    def traces(spikes, sigma_ms):
        """Observed and predicted rate (Hz) smoothed over the predicted frames."""
        sigma = sigma_ms / 1000.0 * fps
        return masked_smooth(spikes * fps, has, sigma), masked_smooth(probability * fps, has, sigma)

    guard = float(settings["null"]["shuffle_guard_seconds"])
    shifts = [int(np.rint(sample_circular_shift(rng, n_frames, fps, guard) * fps)) for _ in range(RATE_NULL_DRAWS)]
    metrics = {"session_id": entry["session_id"], "features": " + ".join(entry["features"])}
    drawn = None
    for sigma_ms in (RATE_DRAW_SMOOTHING_MS, RATE_SMOOTHING_MS):
        observed, predicted = traces(entry["spikes"], sigma_ms)
        overlap = overlap_metrics(predicted[has], observed[has])
        null = np.array([overlap_metrics(predicted[has], traces(np.roll(entry["spikes"], shift), sigma_ms)[0][has])["ccc"]
                         for shift in shifts])
        tag = f"{sigma_ms:.0f}ms"
        metrics.update({f"ccc_{tag}": overlap["ccc"], f"ccc_null_mean_{tag}": float(null.mean()),
                        f"ccc_null_sd_{tag}": float(null.std()),
                        f"ccc_p_{tag}": float((1 + np.sum(null >= overlap["ccc"])) / (1 + null.size)),
                        f"r_{tag}": overlap["r"], f"bias_hz_{tag}": overlap["bias"], f"slope_{tag}": overlap["slope"]})
        if drawn is None:
            drawn = (observed, predicted)
    observed, predicted = drawn
    for name, rows in (("outside_bouts", has & ~entry["vocal"]), ("bout_windows", has & entry["vocal"])):
        metrics[f"ccc_{name}_{RATE_DRAW_SMOOTHING_MS:.0f}ms"] = (overlap_metrics(predicted[rows], observed[rows])["ccc"]
                                                                if rows.sum() > 100 else float("nan"))
    return {"observed": observed, "predicted": predicted, "has": has, "metrics": metrics}


def best_windows(observed: np.ndarray, predicted: np.ndarray, has: np.ndarray, fps: float, count: int) -> list:
    """
    Description
    -----------
    The ``count`` non-overlapping ``RATE_INSET_SECONDS`` windows with the highest concordance, among windows
    that are fully predicted and whose observed rate actually varies (window SD at or above the median) --
    the best periods, by rule, labelled as such on the figure. The median window's concordance is returned
    beside them for the table, so best and typical can be compared.

    Parameters
    ----------
    observed (np.ndarray)
        Smoothed observed rate.
    predicted (np.ndarray)
        Smoothed predicted rate.
    has (np.ndarray)
        Frames carrying a prediction.
    fps (float)
        Frame rate.
    count (int)
        Windows wanted.

    Returns
    -------
    windows (list)
        ``(start_frame, ccc)`` per chosen window, best first, followed by nothing else; the median window
        concordance is appended as a final ``(None, median)`` entry.
    """

    width = int(np.rint(RATE_INSET_SECONDS * fps))
    candidates = []
    for start in range(0, observed.size - width + 1, width // 4):
        window = slice(start, start + width)
        if has[window].mean() < 0.99:
            continue
        keep = has[window]
        candidates.append((overlap_metrics(predicted[window][keep], observed[window][keep])["ccc"],
                           float(np.std(observed[window][keep])), start))
    if not candidates:
        return [(None, float("nan"))]
    spread_floor = float(np.median([spread for _ccc, spread, _start in candidates]))
    chosen = []
    for ccc, spread, start in sorted(candidates, reverse=True):
        if spread < spread_floor or any(abs(start - other) < width for other, _value in chosen):
            continue
        chosen.append((start, ccc))
        if len(chosen) == count:
            break
    return [*chosen, (None, float(np.median([ccc for ccc, _spread, _start in candidates])))]


def draw_session_trace(axis, zoom_axes: list, entry: dict, overlap: dict) -> list:
    """
    Description
    -----------
    One session's observed against predicted firing over the whole session (all frames, bout windows shaded),
    the best periods marked A-D on it, and those periods drawn in the panels below. No numbers on
    the axes: they go to the metrics table.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        The whole-session axis.
    zoom_axes (list)
        The ``SESSION_ZOOM_COUNT`` axes below it, one per best window (A, B, ...).
    entry (dict)
        One session of :func:`session_trace_data`.
    overlap (dict)
        Output of :func:`session_overlap` for this session.

    Returns
    -------
    windows (list)
        :func:`best_windows`' result, for the table.
    """

    fps, n_frames = entry["fps"], entry["n_frames"]
    observed, predicted, has = overlap["observed"], overlap["predicted"], overlap["has"]
    minutes = np.arange(n_frames) / fps / 60.0

    def event_strips(target, window, scale, top):
        """ONE strip under the traces, below zero rate (user-ruled 2026-10-18): the male's bout windows as solid
        blocks, the unassigned vocalizations drawn over them; the y axis then ticks positive rates only."""
        strip = 0.08 * top
        for zorder, (mask, color) in enumerate(((entry["vocal"][window], MALE_COLOR),
                                                (entry["unassigned"][window], UNASSIGNED_COLOR))):
            edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
            spans = [(left * scale, (right - left) * scale) for left, right in zip(edges[::2], edges[1::2], strict=True)]
            target.broken_barh(spans, (-1.3 * strip, strip), facecolors=color, linewidth=0, zorder=1 + zorder)
        target.set_ylim(-1.3 * strip - 0.25 * strip, top * 1.04)
        target.set_yticks([tick for tick in target.get_yticks() if 0.0 <= tick <= top])

    top = float(np.nanmax(np.concatenate([observed[has], predicted[has]])))
    axis.plot(minutes[::3], observed[::3], color=TEXT_COLOR, linewidth=0.4, zorder=2)
    axis.plot(minutes[::3], predicted[::3], color="#C2793A", linewidth=0.7, zorder=3)
    axis.set_xlim(0.0, minutes[-1])
    event_strips(axis, slice(None), 1.0 / fps / 60.0, top)
    axis.set_ylabel("firing rate (sp/s)", fontsize=6)
    axis.set_xlabel("time in session (min)", fontsize=6, labelpad=1)
    axis.tick_params(labelsize=5, length=2, pad=1)
    axis.set_title(f"{entry['session_id']} | fold model: "
                   f"{' + '.join(feature_label(name) for name in entry['features'])}", loc="left", fontsize=6, pad=10)
    windows = best_windows(observed, predicted, has, fps, len(zoom_axes))
    width = int(np.rint(RATE_INSET_SECONDS * fps))
    for letter, zoom, (start, _ccc) in zip("ABCD", zoom_axes, windows[:-1], strict=False):
        axis.axvspan(start / fps / 60.0, (start + width) / fps / 60.0, facecolor="none", edgecolor=REFERENCE_COLOR,
                     linewidth=0.6, zorder=4)
        axis.text((start + width / 2) / fps / 60.0, 1.0, letter, transform=axis.get_xaxis_transform(), ha="center",
                  va="bottom", fontsize=6, color=TEXT_COLOR)
        window = slice(start, start + width)
        seconds = np.arange(width) / fps
        zoom.plot(seconds, observed[window], color=TEXT_COLOR, linewidth=0.6)
        zoom.plot(seconds, predicted[window], color="#C2793A", linewidth=0.9)
        zoom.set_xlim(0.0, RATE_INSET_SECONDS)
        event_strips(zoom, window, 1.0 / fps,
                     float(np.nanmax(np.concatenate([observed[window], predicted[window]]))))
        zoom.set_title(f"{letter}: {start / fps / 60:.1f} min", loc="left", fontsize=5, pad=2)
        zoom.set_xticks([0.0, RATE_INSET_SECONDS / 2.0, RATE_INSET_SECONDS])
        zoom.set_xlabel("time (s)", fontsize=6, labelpad=1)
        zoom.tick_params(labelsize=5, length=2, pad=1)
    zoom_axes[0].set_ylabel("firing rate (sp/s)", fontsize=6)
    for zoom in zoom_axes[len(windows) - 1:]:
        zoom.set_visible(False)
    return windows


def session_traces_figure(unit_uid: str, paths: FigurePaths, identity: dict, settings: dict) -> tuple[list, list]:
    """
    Description
    -----------
    The kinematic model's prediction against the observed firing, every session predicted by the fold model
    that held it out, quiet and vocal frames together. Each session is a whole-session trace with its best
    periods drawn underneath; sessions are stacked ``SESSIONS_PER_PAGE`` to a portrait letter page. Sizes
    (user-ruled 2026-10-18): the whole-session trace ``SESSION_TRACE_SIZE`` inches, the ``SESSION_ZOOM_COUNT``
    period panels sharing its width at the same height; no numbers on the axes.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    pages (list)
        One matplotlib figure per page, each sized to the columns and session rows it holds.
    rows (list)
        One metrics dict per session (for the table), including the two windows' concordance and the median
        window's.
    """

    data = session_trace_data(unit_uid, paths, identity, settings)
    rng = np.random.default_rng(RATE_NULL_SEED)
    sessions = data["sessions"]
    margin = SESSION_PAGE_MARGIN
    trace_width, trace_height = SESSION_TRACE_SIZE
    left = margin + 0.45
    per_page = SESSIONS_PER_PAGE * SESSION_COLUMNS
    header = 0.0
    gap = 0.18
    block = 0.22 + trace_height + 0.28 + 0.18 + trace_height + 0.28 + 0.2
    pages, rows = [], []
    for page_index in range(0, len(sessions), per_page):
        on_page = sessions[page_index:page_index + per_page]
        n_columns = -(-len(on_page) // SESSIONS_PER_PAGE)
        # The page is sized to its content (user-ruled 2026-10-20): the columns and session rows actually
        # drawn plus a thin pad, so no empty letter-page border; panel sizes are unchanged.
        n_rows = min(len(on_page), SESSIONS_PER_PAGE)
        width = 2 * margin + 0.45 + n_columns * trace_width + (n_columns - 1) * SESSION_COLUMN_GAP
        height = 2 * margin + header + n_rows * block - 0.2
        figure = plt.figure(figsize=(width, height))
        for index, entry in enumerate(on_page):
            column, slot = divmod(index, SESSIONS_PER_PAGE)
            left = margin + 0.45 + column * (trace_width + SESSION_COLUMN_GAP)
            top = height - margin - header - slot * block
            trace_bottom = top - 0.22 - trace_height
            axis = figure.add_axes([left / width, trace_bottom / height, trace_width / width, trace_height / height])
            zoom_bottom = trace_bottom - 0.28 - 0.18 - trace_height
            zoom_width = (trace_width - (SESSION_ZOOM_COUNT - 1) * gap) / SESSION_ZOOM_COUNT
            zoom_axes = [figure.add_axes([(left + k * (zoom_width + gap)) / width, zoom_bottom / height,
                                          zoom_width / width, trace_height / height])
                         for k in range(SESSION_ZOOM_COUNT)]
            overlap = session_overlap(entry, settings, rng)
            windows = draw_session_trace(axis, zoom_axes, entry, overlap)
            row = dict(overlap["metrics"])
            for letter, (start, ccc) in zip("ABCD", windows[:-1], strict=False):
                row[f"window_{letter}_start_min"] = start / entry["fps"] / 60.0
                row[f"window_{letter}_ccc"] = ccc
            row["median_window_ccc"] = windows[-1][1]
            rows.append(row)
        pages.append(figure)
    return pages, rows


def tuning_curve(predictor: np.ndarray, outputs: dict, edges: np.ndarray, rng) -> dict:
    """
    Description
    -----------
    Spike rate in bins of a predictor relative to the session-mean expectation, with a block-bootstrap SE.

    Each frame's expectation is its session's mean spike rate in this regime, so a bin's value is observed
    spikes over expected spikes -- the binned analogue of the transfer's per-session intercepts. The bootstrap
    resamples contiguous history-length blocks within each session, since adjacent frames share almost all of
    their history and are not independent.

    Parameters
    ----------
    predictor (np.ndarray)
        Standardized predictor per frame.
    outputs (dict)
        ``labels``, ``session_index`` and ``block`` for the same frames.
    edges (np.ndarray)
        Bin edges.
    rng (np.random.Generator)
        Bootstrap generator.

    Returns
    -------
    curve (dict)
        ``x`` (median predictor per bin), ``gain``, ``se``.
    """

    labels, session_index, block = outputs["labels"], outputs["session_index"], outputs["block"]
    session_rate = {s: labels[session_index == s].mean() for s in np.unique(session_index)}
    expected = np.array([session_rate[s] for s in session_index])
    bins = np.clip(np.digitize(predictor, edges[1:-1]), 0, edges.size - 2)
    n_bins = edges.size - 1

    block_ids, block_index = np.unique(block, return_inverse=True)
    observed_by_block = np.zeros((block_ids.size, n_bins))
    expected_by_block = np.zeros((block_ids.size, n_bins))
    np.add.at(observed_by_block, (block_index, bins), labels)
    np.add.at(expected_by_block, (block_index, bins), expected)
    block_session = np.zeros(block_ids.size, dtype=np.int64)
    block_session[block_index] = session_index

    gain = observed_by_block.sum(axis=0) / np.maximum(expected_by_block.sum(axis=0), 1e-12)
    draws = np.empty((N_BOOTSTRAP, n_bins))
    session_blocks = [np.flatnonzero(block_session == s) for s in np.unique(block_session)]
    for draw in range(N_BOOTSTRAP):
        chosen = np.concatenate([rng.choice(members, size=members.size, replace=True) for members in session_blocks])
        draws[draw] = observed_by_block[chosen].sum(axis=0) / np.maximum(expected_by_block[chosen].sum(axis=0), 1e-12)
    x = np.array([np.median(predictor[bins == b]) if np.any(bins == b) else np.nan for b in range(n_bins)])
    return {"x": x, "gain": gain, "se": draws.std(axis=0, ddof=1)}


def draw_tuning(axis, quiet_predictor: np.ndarray, vocal_predictor: np.ndarray, data: dict, rng, xlabel: str,
                show_legend: bool, vocal_label: str = "vocal (all sessions)") -> None:
    """
    Description
    -----------
    Quiet and vocal tuning curves of one predictor on shared bins.

    Each regime's predictor is standardized within that regime, mirroring the calibration the transfer
    test fits, and bins are deciles of both regimes pooled so the two curves line up.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    quiet_predictor (np.ndarray)
        Raw predictor on the quiet frames.
    vocal_predictor (np.ndarray)
        Raw predictor on the vocal frames.
    data (dict)
        ``quiet`` and ``vocal`` outputs.
    rng (np.random.Generator)
        Bootstrap generator.
    xlabel (str)
        x-axis label.
    show_legend (bool)
        Whether to draw the legend.
    vocal_label (str)
        Legend entry for the vocal curve; the model panel carries the unit's rate elevation here.

    Returns
    -------
    None
    """

    quiet_z = (quiet_predictor - quiet_predictor.mean()) / quiet_predictor.std()
    vocal_z = (vocal_predictor - vocal_predictor.mean()) / vocal_predictor.std()
    edges = np.quantile(np.concatenate([quiet_z, vocal_z]), np.linspace(0.0, 1.0, N_TUNING_BINS + 1))
    for predictor, outputs, color, name in ((quiet_z, data["quiet"], QUIET_COLOR, "quiet (held-out session)"),
                                            (vocal_z, data["vocal"], VOCAL_COLOR, vocal_label)):
        curve = tuning_curve(predictor, outputs, edges, rng)
        axis.fill_between(curve["x"], curve["gain"] - curve["se"], curve["gain"] + curve["se"], color=color,
                          alpha=0.25, linewidth=0)
        axis.plot(curve["x"], curve["gain"], color=color, linewidth=1.2, marker="o", markersize=2.5, label=name)
    axis.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    axis.set_xlabel(xlabel)
    if show_legend:
        axis.legend(frameon=False, fontsize=5.5, loc="best")


def format_p(p_value: float, at_floor: bool, n_draws: int) -> str:
    """
    Description
    -----------
    An empirical p-value as text, saying so when it sits at the floor of its draw count.

    Parameters
    ----------
    p_value (float)
        Empirical p-value.
    at_floor (bool)
        Whether no draw reached the observed value.
    n_draws (int)
        Null draws.

    Returns
    -------
    text (str)
        e.g. ``p < 1.0e-05`` over ``(100,000 draws)``, on two lines.
    """

    relation = "<" if at_floor else "="
    return f"p {relation} {p_value:.1e}\n({n_draws:,} draws)"


def selection_fold_scores(path: list) -> dict:
    """
    Description
    -----------
    Per-inner-fold scores of every step on the representative path, keyed by step.

    The pipeline stores them on the path itself. A result file written before it did carries none, and
    the panel is drawn without them.

    Parameters
    ----------
    path (list)
        The representative fold's forward path.

    Returns
    -------
    scores (dict)
        Step to the list of inner-fold scores; empty when the path carries none.
    """

    return {int(row["step"]): [float(value) for value in row["fold_scores"]]
            for row in path if "fold_scores" in row}


def lighten(hex_color: str, factor: float) -> str:
    """
    Description
    -----------
    Blend a hex colour toward white, as the behavioural selection figures do for the carried-over base.

    Parameters
    ----------
    hex_color (str)
        Colour to lighten.
    factor (float)
        Fraction of the way to white, 0 to 1.

    Returns
    -------
    color (str)
        Lightened hex colour.
    """

    digits = hex_color.lstrip("#")
    channels = [int(digits[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{int(np.rint(c + (255 - c) * factor)):02x}" for c in channels)


def relative_luminance(hex_color: str) -> float:
    """
    Description
    -----------
    Perceived brightness of a hex colour, 0 (black) to 1 (white), to choose legible text on a bar.

    Parameters
    ----------
    hex_color (str)
        Colour.

    Returns
    -------
    luminance (float)
        Rec. 709 weighted brightness of the sRGB channels.
    """

    digits = hex_color.lstrip("#")
    red, green, blue = (int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def named_corner(name: str, width: float, height: float) -> list:
    """
    Description
    -----------
    An inset rectangle at a named corner, in the same geometry ``freest_corner`` chooses within.

    Parameters
    ----------
    name (str)
        ``upper left``, ``upper right``, ``lower left`` or ``lower right``.
    width (float)
        Inset width in axes fractions.
    height (float)
        Inset height in axes fractions.

    Returns
    -------
    rectangle (list)
        ``[x0, y0, width, height]`` for ``inset_axes``.
    """

    horizontal = {"left": INSET_MARGIN, "right": 1.0 - width - INSET_RIGHT_INSET}
    vertical = {"lower": INSET_MARGIN, "upper": 1.0 - height - INSET_TOP_INSET}
    where, side = name.split()
    return [horizontal[side], vertical[where], width, height]


def freest_corner(curves: list, width: float, height: float) -> list:
    """
    Description
    -----------
    One inset rectangle for a whole row of filter panels: the corner those curves occupy least.

    Chosen ONCE for every panel rather than per panel, so the insets line up across a row. Per-panel
    choice looks arbitrary when two filters leave different corners free, and worse when they tie and an
    incidental ordering decides -- which is exactly what put one of the calibration unit's insets on the
    opposite side from its neighbour.

    The upper right is the default and is kept unless the curves genuinely crowd it, rather than merely
    occupy it more than some other corner: a filter that only grazes it there still leaves it readable.

    Parameters
    ----------
    curves (list)
        One ``(curve, low, high)`` triple per panel: the plotted line and its band.
    width (float)
        Inset width in axes fractions.
    height (float)
        Inset height in axes fractions.

    Returns
    -------
    rectangle (list)
        ``[x0, y0, width, height]`` for ``inset_axes``, the same for every panel in the row.
    """

    left, right = INSET_MARGIN, 1.0 - width - INSET_RIGHT_INSET
    bottom, top = INSET_MARGIN, 1.0 - height - INSET_TOP_INSET
    corners = [(right, top), (right, bottom), (left, bottom), (left, top)]
    counts = dict.fromkeys(corners, 0)
    points = 0
    for curve, low, high in curves:
        span = max(float(np.max(high) - np.min(low)), 1e-12)
        x = np.linspace(0.0, 1.0, np.asarray(curve).size)
        scaled = [(np.asarray(values) - float(np.min(low))) / span for values in (curve, low, high)]
        points += 3 * np.asarray(curve).size
        for x0, y0 in corners:
            inside_x = (x >= x0 - 0.02) & (x <= x0 + width + 0.02)
            counts[(x0, y0)] += sum(int(np.sum(inside_x & (values >= y0 - 0.02)
                                               & (values <= y0 + height + 0.02)))
                                    for values in scaled)
    upper_right = corners[0]
    chosen = (upper_right if counts[upper_right] <= INSET_CROWDING_FRACTION * points
              else min(corners, key=lambda corner: counts[corner]))
    return [chosen[0], chosen[1], width, height]


def draw_fold_selection_grid(axis, encoding: dict, rejected_style: bool, max_rows: int = 8) -> None:
    """
    Description
    -----------
    Which features each fold of the rotation selected, as a feature-by-fold grid.

    Every fold screens and selects on its own training sessions, so correlated features trade places between
    folds; this panel is what shows that spread, while the panel beside it details one fold's path. A filled
    dot means the fold selected the feature, shaded by selection order so the anchor reads darkest; a hollow
    dot means the feature cleared that fold's screen but selection did not take it; an empty cell means it
    did not clear the screen. Rows are restricted to features some fold actually selected -- the features
    that only ever passed a screen would add rows without adding an outcome.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    rejected_style (bool)
        True when no fold selected anything, so each fold's entry is its best SCREENED feature: drawn in
        the same grey the rejected step carries, since it is a rejection, not a selection.
    max_rows (int)
        Most rows to draw; the least often selected are dropped and reported in the title.

    Returns
    -------
    None
    """

    selected_per_fold = [list(entry) for entry in encoding["selected_per_fold"]]
    screens = encoding["screen_per_fold"]
    sessions = [str(entry["test_session"]) for entry in encoding["filters_per_fold"]]
    representative = int(encoding["representative_fold"])

    counts = {}
    for selection in selected_per_fold:
        for feature in selection:
            counts[feature] = counts.get(feature, 0) + 1
    ordered = sorted(counts, key=lambda feature: (-counts[feature], feature))
    dropped = max(len(ordered) - max_rows, 0)
    ordered = ordered[:max_rows]

    for column, (selection, screen) in enumerate(zip(selected_per_fold, screens, strict=True)):
        survivors = {row["name"] for row in screen if bool(row["survived"])}
        for position, feature in enumerate(ordered):
            if feature in selection:
                order = selection.index(feature)
                axis.scatter(column, position, s=44, marker="o",
                             color=REJECTED_COLOR if rejected_style else feature_color(feature),
                             alpha=1.0 if rejected_style else 1.0 - 0.22 * order,
                             edgecolor=REFERENCE_COLOR if rejected_style else TEXT_COLOR, linewidth=0.4,
                             zorder=3)
            elif feature in survivors:
                axis.scatter(column, position, s=30, marker="o", facecolors="none",
                             edgecolors=feature_color(feature), linewidth=0.8, zorder=2)
    for position in range(len(ordered)):
        axis.axhline(position, color=REFERENCE_COLOR, linewidth=0.4, alpha=0.4, zorder=1)

    axis.set_xticks(range(len(sessions)))
    axis.set_xticklabels([session.split("_")[-1] for session in sessions], rotation=35, ha="right", fontsize=6)
    axis.get_xticklabels()[representative].set_fontweight("bold")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([feature_label(feature) for feature in ordered],
                         style="italic" if rejected_style else "normal")
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(-0.45, len(sessions) - 0.55)
    axis.set_ylim(len(ordered) - 0.6, -0.4)
    axis.set_xlabel("held-out session")
    title = ("best screened per fold (rejected) | screened (hollow)" if rejected_style
             else "selected per fold (filled) | screened (hollow)")
    axis.set_title(title if not dropped else f"{title} | {dropped} rarer feature(s) not shown", loc="left",
                   fontsize=PANEL_TITLE_SIZE)


def draw_screen_bars(axis, encoding: dict, fold_index: int) -> None:
    """
    Description
    -----------
    Every screened feature's held-out score for one fold, as bars ordered by that score.

    Drawn where a tuned unit shows its forward-selection path, because a unit that selected nothing has no
    path -- but it does have nineteen single-feature fits, and their spread is what says whether the screen
    rejected a near miss or found nothing at all. Also drawn for a unit whose fold DID pass a feature that
    then failed the held-out test, since that unit has no validated model either; the title says how many
    passed the fold's screen. All grey and italic.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    fold_index (int)
        Which fold's screen to draw.

    Returns
    -------
    None
    """

    rows = sorted(encoding["screen_per_fold"][fold_index], key=lambda row: row["score"])
    positions = np.arange(len(rows))
    best = max(range(len(rows)), key=lambda index: rows[index]["score"])
    colors = ["#9A9A9A" if index == best else REJECTED_COLOR for index in range(len(rows))]
    axis.barh(positions, [row["score"] for row in rows], height=0.7, color=colors, edgecolor="none")
    axis.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels([feature_label(row["name"]) for row in rows], fontsize=4, style="italic",
                         color=REFERENCE_COLOR)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=5)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xlabel("calibrated D\u00b2")
    passed = sum(1 for row in rows if row["survived"])
    axis.set_title("screened features (none survived)" if passed == 0 else
                   f"screened features ({passed} passed this fold's screen, failed held-out)", loc="left",
                   fontsize=PANEL_TITLE_SIZE)


def draw_selection(axis, forward_path: list, fold_scores: dict, title: str | None = None) -> None:
    """
    Description
    -----------
    Forward selection as horizontal bars, one per step, in selection order from the top, styled after
    ``plot_manifold_selection_trajectory``.

    Each bar is split into a light base (the model's score before this step) and a dark tip (this step's
    gain). Text is written at the start of each bar: the RANGE of the step's score over the inner folds (the
    worst and best validation session), and the step's gain. A range rather than a confidence interval,
    because there are only 2-5 inner folds: a t-based 99% interval over two folds spans several units of D2,
    and a normal-approximation one claims coverage it does not have. The number of inner folds is in the
    title. A rejected final step is drawn in grey below the accepted ones, its text italic and marked ``ns``.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    forward_path (list)
        The representative fold's forward path: ``step``, ``candidate``, ``decision``, ``mean``,
        ``improvement``, ``standard_error``.
    fold_scores (dict)
        Step to rebuilt inner-fold scores.
    title (str)
        Panel title; the forward-selection wording unless the unit has no selected model.

    Returns
    -------
    None
    """

    accepted = [row for row in forward_path if row["decision"] in ("ANCHOR", "ACCEPT")]
    rejected = [row for row in forward_path if row["decision"] == "REJECT"]
    bar_height = 0.88
    previous = 0.0
    n_inner = max((len(scores) for scores in fold_scores.values()), default=0)
    positions, labels, italics = [], [], []
    rows = [(position, row, "rejected" in row and bool(row["rejected"]))
            for position, row in enumerate(accepted)]
    rows += [(len(accepted) + offset + 0.4, row, True) for offset, row in enumerate(rejected)]
    for position, row, is_rejected in rows:
        base = "#D7D7D7" if is_rejected else lighten(feature_color(row["candidate"]), 0.65)
        tip = "#9A9A9A" if is_rejected else feature_color(row["candidate"])
        if previous > 0:
            axis.barh(position, previous, color=base, edgecolor="none", height=bar_height)
        axis.barh(position, row["mean"] - previous, left=previous, color=tip, edgecolor="none", height=bar_height)
        scores = fold_scores.get(int(row["step"]), [])
        text = f"range [{min(scores):.3f}, {max(scores):.3f}]" if scores else ""
        if row["decision"] != "ANCHOR":
            text += f"   \u0394 {row['improvement']:+.3f}{' (ns)' if is_rejected else ''}"
        underneath = tip if previous <= 0 else base
        axis.text(0.01 * max(r["mean"] for r in forward_path), position, text, ha="left", va="center", fontsize=6,
                  color="#FFFFFF" if relative_luminance(underneath) < 0.5 else TEXT_COLOR,
                  style="italic" if is_rejected else "normal")
        positions.append(position)
        labels.append(f"+{feature_label(row['candidate'])}")
        italics.append(is_rejected)
        if not is_rejected:
            previous = row["mean"]
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    for label, is_rejected in zip(axis.get_yticklabels(), italics, strict=True):
        label.set_style("italic" if is_rejected else "normal")
        label.set_color(REFERENCE_COLOR if is_rejected else TEXT_COLOR)
    axis.tick_params(axis="y", length=0)
    # Keep a bar the same thickness whether a unit selected one feature or four, rather than letting a
    # single step fill the panel -- while leaving room below the LAST row for its full height. A rejected
    # step sits at a fractional position (it is offset below the accepted ones), so counting rows rather
    # than reading the lowest position clipped it against the axis.
    axis.set_ylim(max(max(positions) + 0.5 * bar_height + 0.12, 2.5), -0.5)
    axis.set_xlim(0.0, 1.03 * max(row["mean"] for row in forward_path))
    axis.set_xlabel("calibrated D\u00b2")
    axis.set_title(title if title else f"forward selection ({n_inner} inner folds)", loc="left", fontsize=PANEL_TITLE_SIZE)


def stored_fold(paths: FigurePaths, identity: dict, fold_index: int) -> dict:
    """
    Description
    -----------
    One encoding fold's artifact, from its ``.npz`` file or, once the unit is finished, from the
    ``kinematic_encoding_folds`` section of its result file.

    Parameters
    ----------
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.
    identity (dict)
        The result file's ``identity`` block (``unit_uid`` names the fold).
    fold_index (int)
        Which fold.

    Returns
    -------
    fold (dict)
        The fold's artifact.
    """

    fold = read_fold_artifact(str(paths.results_directory), identity, fold_index)
    if fold is None:
        msg = f"{identity['unit_uid']}: fold {fold_index} is in neither a fold file nor the result file"
        raise FileNotFoundError(msg)
    return fold


def per_fold_transfer(paths: FigurePaths, identity: dict, settings: dict) -> list:
    """
    Description
    -----------
    Each fold's own model scored on every vocal frame, pooled with one shared slope, as the transfer is.

    Recomputed from the fold files by refitting the two-level calibration only; no model is refitted.

    Parameters
    ----------
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    transfers (list)
        One ``(fold_index, score, slope)`` per fold with a model.
    """

    steps = settings["kinematic_encoding"]["solver"]["calibration_steps"]
    transfers = []
    for fold_index in range(len(identity["courtship_sessions"])):
        fold = stored_fold(paths, identity, fold_index)
        # A fold that selected nothing still fitted its best screened feature, and that is what the panels
        # of an untuned unit describe; only a fold with no fitted filter at all is skipped.
        if bool(fold["no_model"]) and not ("best_screened" in fold and bool(fold["best_screened"])):
            continue
        everywhere = list(fold["vocal_everywhere"])
        if "bin_eta" in everywhere[0]:
            # the count grain: the same pooling the pipeline did, on the stored per-bin arrays
            eta = np.concatenate([np.asarray(entry["bin_eta"]) for entry in everywhere])
            counts = np.concatenate([np.asarray(entry["bin_counts"]) for entry in everywhere])
            session_index = np.concatenate([np.full(int(entry["n_bins"]), k) for k, entry in enumerate(everywhere)])
            score, slope = frozen_model_score(eta, counts, session_index, steps, "poisson",
                                              float(everywhere[0]["bin_width_frames"]),
                                              settings["kinematic_encoding"]["calibration"])
        else:
            eta = np.concatenate([np.asarray(entry["eta"]) for entry in everywhere])
            labels = np.concatenate([np.asarray(entry["labels"]) for entry in everywhere]).astype(np.float64)
            session_index = np.concatenate([np.full(np.asarray(entry["labels"]).size, k) for k, entry in enumerate(everywhere)])
            score, slope = frozen_model_score(eta, labels, session_index, steps, "bernoulli", 1.0,
                                              settings["kinematic_encoding"]["calibration"])
        transfers.append((fold_index, score, slope))
    return transfers


def draw_score(axis, fold_scores: list, tested_score: float, null: np.ndarray, p_text: str,
               slope_label: str, slopes: list, slope_caption: str, title: str, color: str) -> None:
    """
    Description
    -----------
    The TESTED score as the bar, its folds as dots over their range, beside its null.

    The bar used to be the folds' mean +/- SE with the tested value marked by a diamond on top, and that
    put two DIFFERENT quantities in one panel with the wrong one dominating. Three of the statistics
    drawn here are POOLED -- the transfer, the decode gain and the nested added score -- precisely
    because fold-averaging them is invalid: the concentration is fitted at scoring time, the macro
    average silently drops thin regions, and a small fold overfits its own calibration. On the decode
    gain the two differ by more than fivefold (an unweighted fold mean of 0.0019 against the
    event-weighted pooled 0.0108), so a reader took the bar for the result and the result for a marker.

    Now the bar IS the tested statistic, and the spread beside it is drawn as dots over a range line.

    THE SPREAD MUST BE THE SAME KIND OF QUANTITY AS THE BAR. A leave-one-out minimum used to be drawn
    over the dots as a tick, and it is not: the dots were scored on ONE fold each while the tick came
    from values scored on n-1 folds, so the tick looked like the lowest dot and was not. On cl0401's
    nested panel the dots ran 0.0117-0.0486 with a tick at 0.0219, which is the minimum of an entirely
    different set of numbers. The tick is gone; the minimum stays in the panel text, where it reads as
    the separate statistic it is, and callers pass leave-one-out values AS the dots wherever the
    pipeline stores them exactly.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    fold_scores (list)
        One score per fold with a model; a DESCRIPTOR of the spread, never the tested value.
    tested_score (float)
        The tested statistic -- pooled where the plan rules it pooled. This is the bar.
    null (np.ndarray)
        The tested score's null draws.
    p_text (str)
        Formatted p-value, with the AUROC on its own line: chance-referenced at 0.5 and, unlike the
        calibrated score, directly comparable between the two regimes.
    slope_label (str)
        Printed above the inset: what the per-fold values are and their tested value.
    slopes (list)
        Per-fold values for the inset: a vertical line over their range, a tick at their mean.
    slope_caption (str)
        What those values are, for the inset label.
    title (str)
        Panel title.
    color (str)
        Bar colour.

    Returns
    -------
    None
    """

    folds = [float(value) for value in fold_scores if np.isfinite(value)]
    axis.bar(0.0, float(tested_score), width=0.6, color=color, edgecolor="none")
    if folds:
        axis.plot([0.0, 0.0], [min(folds), max(folds)], color=TEXT_COLOR, linewidth=0.8, zorder=2)
        axis.scatter(np.zeros(len(folds)), folds, s=7, color="#FFFFFF", edgecolor=TEXT_COLOR,
                     linewidth=0.5, zorder=3)
    violin = axis.violinplot(null, positions=[1.0], widths=0.6, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(NULL_COLOR)
        body.set_edgecolor("none")
        body.set_alpha(0.8)
    axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.6)
    axis.set_xticks([0.0, 1.0])
    axis.set_xticklabels(["data", "null"])
    axis.set_xlim(-0.6, 1.6)
    marks = [0.0, float(tested_score), float(np.min(null)), float(np.max(null)), *folds]
    low, high = min(marks), max(marks)
    axis.set_ylim(low - 0.08 * (high - low), high + 0.55 * (high - low))
    axis.text(0.04, 0.99, p_text, transform=axis.transAxes, ha="left", va="top", fontsize=6, color=TEXT_COLOR)

    # The slope carries the DIRECTION the calibrated score is blind to, so its spread is what says the sign
    # does not rest on one session (vocal) or one selected model (quiet).
    inset = axis.inset_axes([0.60, 0.42, 0.32, 0.29])
    inset.plot([0.0, 0.0], [min(slopes), max(slopes)], color=TEXT_COLOR, linewidth=1.0, zorder=1)
    inset.scatter(np.zeros(len(slopes)), slopes, s=7, color=color, edgecolor=TEXT_COLOR, linewidth=0.3, zorder=3)
    inset.plot([-0.35, 0.35], [float(np.mean(slopes)), float(np.mean(slopes))], color=TEXT_COLOR, linewidth=1.4,
               zorder=2)
    inset.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    span = max(max(slopes) - min(slopes), 1e-6)
    inset.set_xlim(-1.0, 1.0)
    inset.set_ylim(min(0.0, min(slopes) - 0.35 * span), max(0.0, max(slopes) + 0.35 * span))
    inset.set_xticks([])
    inset.tick_params(axis="y", labelsize=5, length=2, pad=1)
    inset.set_title(slope_label, fontsize=5, color=TEXT_COLOR, pad=2)
    inset.set_xlabel(slope_caption, fontsize=4.5, color=TEXT_COLOR, labelpad=1)
    for side in ("top", "right", "bottom"):
        inset.spines[side].set_visible(False)
    axis.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE)


def gating_proximity_data(unit_uid: str, paths: FigurePaths, identity: dict, settings: dict) -> dict:
    """
    Description
    -----------
    Per-call and per-quiet-frame firing against inter-animal distance, for the gating panels.

    Model-free throughout: a call's rate is the spikes in ``[start, stop]`` over its own duration, counted
    from spike TIMES rather than binned frames, and a quiet frame's rate is its 0/1 label. Distance is read
    from the raw (unscaled) behavioural features, so the axis is in centimetres rather than SDs, and is
    taken at each call's onset frame.

    Distance comes in two rulers, both kept, because they are not interchangeable and each answers to
    something different. ``head`` is the distance between the animals' HEAD points (the ``space``
    coordinates, the Head node in cm); ``nose`` is the model's own ``nose-nose`` feature. Two mice side by
    side with their heads 2 cm apart can have noses pointing away from each other, so the nose-tip
    distance mixes proximity with head orientation: measured on cl0633 it ranks the calls at Spearman
    -0.41 against -0.54 for head points. The head ruler is the cleaner description of proximity; the nose
    ruler is what the gating statistic was computed on.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    data (dict)
        ``call_distance`` and ``quiet_distance``, each a dict of ``head`` and ``nose`` arrays; ``call_rate``
        and ``call_category`` per focal call; ``quiet_label`` per quiet anchor; and ``fps``.
    """

    cache_path = paths.cache_directory / f"{unit_uid}_gating_proximity.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)

    kinematic = settings["kinematic_encoding"]
    vocalization = settings["vocalization_settings"]
    per_session = assemble_unit_sessions(identity, str(paths.data_root), settings["kinematic_features"],
                                         kinematic["history_pre_seconds"], kinematic["clean_post_seconds"],
                                         vocalization["clean_against"],
                                         vocalization)
    sessions = list(identity["courtship_sessions"])
    raw, _camera_rates, track_names = load_behavioral_feature_data(
        behavior_file_paths=[str(paths.data_root / session) for session in sessions], csv_sep=",")

    call_rate, call_category, quiet_label = [], [], []
    call_distance = {"head": [], "nose": []}
    quiet_distance = {"head": [], "nose": []}
    fps = float(per_session[sessions[0]]["fps"])
    for session_id in sessions:
        session = per_session[session_id]
        names = track_names[session_id]
        partner = next(name for name in names if name != identity["mouse_id"])
        frame = raw[session_id]
        head = np.stack([frame[f"{identity['mouse_id']}.space{axis}"].to_numpy() for axis in "XYZ"], axis=1)
        partner_head = np.stack([frame[f"{partner}.space{axis}"].to_numpy() for axis in "XYZ"], axis=1)
        column = f"{identity['mouse_id']}-{partner}.nose-nose"
        if column not in frame.columns:
            column = f"{partner}-{identity['mouse_id']}.nose-nose"
        distances = {"head": np.linalg.norm(head - partner_head, axis=1).astype(np.float64),
                     "nose": frame[column].to_numpy().astype(np.float64)}
        finite_frame = np.all([np.isfinite(values) for values in distances.values()], axis=0)

        occupancy = np.zeros(session["n_frames"], dtype=np.int8)
        spikes = session["spike_frames"]
        occupancy[spikes[(spikes >= 0) & (spikes < session["n_frames"])]] = 1
        spike_seconds = np.sort(load_unit_spike_frames(str(paths.data_root), session_id,
                                                       identity["unit_id"])[0])

        usvs = load_session_usvs(str(paths.data_root), session_id)
        focal = usvs.filter(usvs["emitter"] == identity["mouse_id"])
        starts = focal["start"].to_numpy()
        stops = focal["stop"].to_numpy()
        categories = focal[settings["nested_vocal_decoding"]["region_label_column"]].to_numpy()
        for start, stop, category in zip(starts, stops, categories, strict=True):
            first = int(np.floor(start * fps))
            last = int(np.ceil(stop * fps))
            onset = min(max(int(np.rint(start * fps)), 0), session["n_frames"] - 1)
            if last <= first or last > session["n_frames"] or not finite_frame[onset]:
                continue
            for ruler, values in distances.items():
                call_distance[ruler].append(values[onset])
            # Counted from spike TIMES over the call's own span. Binning the frames instead rounds the
            # span outward to whole frames, catching spikes just outside the call -- 9.32 Hz against
            # 8.52 Hz on cl0633, which is a tenth of the effect being measured.
            call_rate.append((np.searchsorted(spike_seconds, stop) - np.searchsorted(spike_seconds, start))
                             / (stop - start))
            # A USV with no torus position has no category either (4 of 2,437 focal USVs on this day).
            # It keeps its rate and distance -- the gate curves need neither -- and is left out of the
            # by-category panel only.
            call_category.append(float(category) if np.isfinite(category) else np.nan)

        anchors = session["quiet"][finite_frame[session["quiet"]]]
        for ruler, values in distances.items():
            quiet_distance[ruler].append(values[anchors])
        quiet_label.append(occupancy[anchors].astype(np.float64))

    data = {"call_distance": {ruler: np.asarray(values) for ruler, values in call_distance.items()},
            "quiet_distance": {ruler: np.concatenate(values) for ruler, values in quiet_distance.items()},
            "call_rate": np.asarray(call_rate), "call_category": np.asarray(call_category),
            "quiet_label": np.concatenate(quiet_label), "fps": fps}
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(data, cache_file)
    return data


def quantile_distance_bins(data: dict, ruler: str) -> np.ndarray:
    """
    Description
    -----------
    Equal-count partner-distance bin edges, for the gate CURVE.

    A line plot draws each point at its bin's median distance, so unequal bin widths mislead nobody and
    equal counts put the resolution where the calls are: half of them fall within 6 cm, where the gate
    does all of its work. This is the wrong choice for the heatmap -- see :func:`distance_bins` -- and the
    right one here.

    Parameters
    ----------
    data (dict)
        From :func:`gating_proximity_data`.
    ruler (str)
        ``head`` or ``nose``.

    Returns
    -------
    edges (np.ndarray)
        ``N_GATE_BINS + 1`` edges.
    """

    return np.quantile(data["call_distance"][ruler], np.linspace(0.0, 1.0, N_GATE_BINS + 1))


def draw_gating_verdict(axis, gating: dict) -> None:
    """
    Description
    -----------
    Every feature's own effect on firing beside its interaction with vocalizing, sorted by the interaction.

    This is the comparison the sub-analysis exists to make: do a feature and vocalizing combine by ADDING
    or by MULTIPLYING? Both bars are INCREMENTS in one nested chain -- the grey is what vocalizing adds
    with the feature already in the model, the coloured is what their product adds with both main effects
    already in -- so they are the same kind of quantity and the picture is the comparison. A unit that
    simply fires to calls has a long grey bar and a stub beside it; a unit whose vocal response is gated
    by the feature has a product term that rivals the plain vocal response, and the coloured bar
    overtakes the grey one. Red marks exactly that, which is the sub-analysis's verdict, met by ONE
    feature of nineteen on the calibration unit.

    Drawn as the two whole MODELS instead -- additive against multiplicative, each over an intercept --
    the picture would be a different comparison from the colour: the multiplicative model contains the
    additive one, so it can never be shorter, and the gap between them is the coloured bar here while the
    grey one would not appear at all.

    Coloured by that comparison rather than by the interaction's p-value, because at this many scored
    rows the null is tight enough that twelve of nineteen features carry a significantly non-zero
    interaction, one of them worth a fortieth of its vocal response: significance says the product term
    is not nothing, not that the response is multiplicative.

    Both are LOSO held-out added deviance divided by the intercept-only deviance of the same rows, which
    is the fraction of what there was to explain -- the same quantity the kinematic encoding reports as
    D2, so the two panels of this figure can be read against each other. Nothing here is recomputed.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    gating (dict)
        The unit's ``vocal_gating`` section.

    Returns
    -------
    None
    """

    features = gating["per_feature"]
    ordered = sorted(features, key=lambda name: features[name]["interaction"] / features[name]["baseline"])
    positions = np.arange(len(ordered))
    vocal = [features[name]["vocal_main"] / features[name]["baseline"] for name in ordered]
    interaction = [features[name]["interaction"] / features[name]["baseline"] for name in ordered]
    colors = [GATE_SIGNIFICANT_COLOR if features[name]["interaction"] > features[name]["vocal_main"]
              else GATE_BELOW_COLOR for name in ordered]

    axis.barh(positions + 0.21, vocal, height=0.4, color=ADDITIVE_COLOR, edgecolor="none")
    axis.barh(positions - 0.21, interaction, height=0.4, color=colors, edgecolor="none")
    axis.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels([feature_label(name) for name in ordered], fontsize=4)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=5)
    axis.set_ylim(-0.6, len(ordered) - 0.4)
    axis.set_xlabel("held-out D\u00b2")
    # Below the axis rather than inside it: at this width a legend in the panel covers the bars it explains.
    handles = [Patch(color=ADDITIVE_COLOR, label="vocal"),
               Patch(color=GATE_BELOW_COLOR, label="interaction <"),
               Patch(color=GATE_SIGNIFICANT_COLOR, label="interaction >")]
    axis.legend(handles=handles, frameon=False, fontsize=5, loc="upper left",
                bbox_to_anchor=(-0.44, -0.03), borderaxespad=0.0, handlelength=1.0, handleheight=0.8,
                labelspacing=0.3)
    axis.set_title("vocal-feature interactions", loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_gate(axis, data: dict, ruler: str) -> None:
    """
    Description
    -----------
    Firing rate against inter-animal distance, during calls and on quiet frames.

    This is the gate itself, with no model in it: during calls the unit fires hard when the animals are
    close and falls silent as they separate, while on quiet frames it is near zero at every distance except
    at near contact.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    data (dict)
        From :func:`gating_proximity_data`.
    ruler (str)
        ``head`` or ``nose``.

    Returns
    -------
    None
    """

    distance = data["call_distance"][ruler]
    edges = quantile_distance_bins(data, ruler)
    call_bin = np.clip(np.digitize(distance, edges[1:-1]), 0, edges.size - 2)
    quiet_bin = np.clip(np.digitize(data["quiet_distance"][ruler], edges[1:-1]), 0, edges.size - 2)
    centres, during, quiet = [], [], []
    for index in range(edges.size - 1):
        calls = call_bin == index
        frames = quiet_bin == index
        centres.append(np.median(distance[calls]) if np.any(calls) else np.nan)
        during.append(data["call_rate"][calls].mean() if np.any(calls) else np.nan)
        quiet.append(data["quiet_label"][frames].mean() * data["fps"] if np.any(frames) else np.nan)
    correlation = spearmanr(distance, data["call_rate"])

    axis.plot(centres, during, color=VOCAL_COLOR, linewidth=1.2, marker="o", markersize=3,
              label="during USVs")
    axis.plot(centres, quiet, color=QUIET_COLOR, linewidth=1.2, marker="s", markersize=3, linestyle="--",
              label="quiet")
    axis.set_xlabel("partner distance (cm)")
    axis.set_ylabel("firing rate (sp/s)")
    axis.legend(frameon=False, fontsize=5.5, loc="upper right")
    axis.set_title(f"the gate (Spearman \u03c1 {correlation.statistic:+.2f})", loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_gate_by_category(axis, data: dict, ruler: str) -> None:
    """
    Description
    -----------
    During-call firing by acoustic category, split into close and far calls.

    A unit that coded call TYPE would show a category running high at both distances; a gate shows every
    category higher when close, which is what separates the two readings.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    data (dict)
        From :func:`gating_proximity_data`.
    ruler (str)
        ``head`` or ``nose``.

    Returns
    -------
    None
    """

    split = float(np.median(data["call_distance"][ruler]))
    close = data["call_distance"][ruler] <= split
    categories = sorted({int(value) for value in data["call_category"] if np.isfinite(value)})
    positions = np.arange(len(categories))
    for offset, (mask, color, name) in enumerate(((close, VOCAL_COLOR, f"close (<={split:.0f} cm)"),
                                                  (~close, lighten(VOCAL_COLOR, 0.55), "far"))):
        heights = [data["call_rate"][mask & (data["call_category"] == category)].mean()
                   if np.any(mask & (data["call_category"] == category)) else np.nan
                   for category in categories]
        axis.bar(positions + 0.42 * offset - 0.21, heights, width=0.4, color=color, edgecolor="none",
                 label=name)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(category) for category in categories])
    axis.set_xlabel("USV category")
    axis.set_ylabel("during-USV rate (sp/s)")
    axis.legend(frameon=False, fontsize=5.5, loc="upper right")
    axis.set_title("gate holds in every category", loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_gate_heatmap(axis, data: dict, ruler: str) -> None:
    """
    Description
    -----------
    During-call firing over distance and acoustic category, as a heatmap.

    Banding by column rather than by row is the statement: distance organises the response and category
    does not.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    data (dict)
        From :func:`gating_proximity_data`.
    ruler (str)
        ``head`` or ``nose``.

    Returns
    -------
    None
    """

    distance = data["call_distance"][ruler]
    edges = quantile_distance_bins(data, ruler)
    call_bin = np.clip(np.digitize(distance, edges[1:-1]), 0, edges.size - 2)
    categories = sorted({int(value) for value in data["call_category"] if np.isfinite(value)})
    grid = np.full((len(categories), edges.size - 1), np.nan)  # empty cells drawn as zero, see below
    labels = []
    for column in range(edges.size - 1):
        in_bin = call_bin == column
        labels.append(f"{np.median(distance[in_bin]):.0f}" if np.any(in_bin) else "")
        for row, category in enumerate(categories):
            cell = in_bin & (data["call_category"] == category)
            if np.any(cell):
                grid[row, column] = data["call_rate"][cell].mean()
    # A cell with no call of that category at that distance is drawn as zero rather than as missing. It
    # only ever happens far out, where every populated cell is zero anyway, so a gap in the colour there
    # says less than it costs in attention.
    #
    # Columns are EQUAL WIDTH and the axis is categorical, ticked with each bin's median distance. The bins
    # are the curve's equal-count ones, so all the gating panels describe the same partition -- but an
    # image draws every cell the same width whatever the axis claims, so placing equal-count bins on a
    # linear axis would stretch the narrow near bins across the panel and read as firing at a distance.
    #
    # Shaded smoothly ALONG each row and never across rows: distance is ordered, so interpolating between
    # neighbouring distance bins says something, while the acoustic categories are nominal and blending
    # category 3 into category 4 would invent a gradient between labels that have no order.
    filled = np.nan_to_num(grid, nan=0.0)
    columns = np.arange(filled.shape[1])
    fine = np.linspace(columns[0], columns[-1], HEATMAP_SHADING_POINTS)
    shaded = np.stack([np.interp(fine, columns, row) for row in filled])
    image = axis.imshow(shaded, aspect="auto", origin="lower", cmap=SEQUENTIAL_CMAP,
                        extent=(columns[0], columns[-1], -0.5, filled.shape[0] - 0.5),
                        interpolation="nearest")
    axis.set_xticks(columns)
    axis.set_xticklabels(labels, fontsize=5)
    axis.set_yticks(range(len(categories)))
    axis.set_yticklabels([str(category) for category in categories])
    axis.set_ylabel("USV category")
    axis.set_xlabel("partner distance (cm, bin median)")
    bar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.06)
    bar.ax.tick_params(labelsize=5)
    bar.set_label("during-USV rate (sp/s)", fontsize=5.5)
    axis.set_title("distance orders it, category does not", loc="left", fontsize=PANEL_TITLE_SIZE)


def vocal_event_data(unit_uid: str, paths: FigurePaths, identity: dict, settings: dict) -> dict:
    """
    Description
    -----------
    The per-call and per-window arrays the vocal panels draw, rebuilt from the design and cached.

    Only the event assembly is rebuilt. Every number the panels report -- scores, nulls, per-fold detail,
    the fitted tuning coefficients -- is read from the stored sections, and the assembly is what the
    storage rule leaves out because it costs seconds.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    events (dict)
        As ``assemble_unit_vocal_events``: per-call ``counts``, ``positions``, ``session_index`` and the
        ``occurrence_windows`` block of counts and vocal/quiet labels.
    """

    cache_path = paths.cache_directory / f"{unit_uid}_vocal_events.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)

    window = settings["vocalization_settings"]["spike_window"]
    events = assemble_unit_vocal_events(identity, str(paths.data_root), copy.deepcopy(settings),
                                        window["pre_offset_seconds"], window["width_seconds"])
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(events, cache_file)
    return events


def draw_occurrence_tuning(axis, occurrence: dict, windows: dict, session_ids: list, legend_at: str) -> None:
    """
    Description
    -----------
    How often a window is vocal at each spike count, one line per session.

    This is what the occurrence test measures, drawn rather than summarised: the statistic is a logistic of
    the vocal/quiet label on the window's spike count, so plotting the observed vocal fraction against that
    count shows the same relationship the fitted slope reports. Quiet windows are the claim-1 quiet tiles,
    laid at the analysed window's own width.

    ONE POINT PER COUNT, not per quantile bin. The count is a small integer -- zero holds 71% of the
    windows on the calibration unit -- so equal-count bins collapse the whole rise into three points, with
    everything from three spikes upward pooled into the last. Counts above ``OCCURRENCE_MAX_SPIKES`` are
    pooled, since those windows are few.

    One line per session because the test fits a shared slope but a SEPARATE INTERCEPT per session, so a
    pooled curve blends baselines the statistic keeps apart. On the calibration unit they nearly coincide
    (slopes 0.86 to 0.91), which is itself worth being able to see.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    occurrence (dict)
        The pickle's ``vocal_occurrence`` section.
    windows (dict)
        The ``occurrence_windows`` block: ``counts``, ``labels`` and ``session_index``.
    session_ids (list)
        Session names, indexed as ``session_index``.
    legend_at (str)
        Where the session legend goes. It has to move per unit: the curve's shape decides which corner is
        free, and a unit that fires rarely outside calls rises far earlier than one that does not.

    Returns
    -------
    None
    """

    counts, labels, sessions = windows["counts"], windows["labels"], windows["session_index"]
    capped = np.minimum(counts, OCCURRENCE_MAX_SPIKES)
    styles = ["-", "--", ":", "-."]
    for slot, session_id in enumerate(session_ids):
        rows = sessions == slot
        if not np.any(rows):
            continue
        x, fraction, spread = [], [], []
        for value in range(OCCURRENCE_MAX_SPIKES + 1):
            cell = rows & (capped == value)
            if not np.any(cell):
                continue
            share = float(labels[cell].mean())
            x.append(value)
            fraction.append(share)
            spread.append(float(np.sqrt(share * (1.0 - share) / cell.sum())))
        axis.errorbar(x, fraction, yerr=spread, color=VOCAL_COLOR, linewidth=1.0, marker="o",
                      markersize=2.5, capsize=0, linestyle=styles[slot % len(styles)],
                      label=session_id.split("_")[-1])
    # The share of windows that are vocal at all: where every point would sit if the count said nothing.
    # It moves with `max_quiet_tiles_per_session`, so it is labelled as the mix rather than left to read
    # as a property of the recording -- the slope and the AUROC, which the test reports, do not move.
    chance = float(labels.mean())
    axis.axhline(chance, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    axis.text(OCCURRENCE_MAX_SPIKES, chance, f"{chance:.2f} vocal overall", ha="right", va="top",
              fontsize=5, color=REFERENCE_COLOR)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xticks(range(0, OCCURRENCE_MAX_SPIKES + 1, 2))
    axis.set_xticklabels([str(value) for value in range(0, OCCURRENCE_MAX_SPIKES, 2)]
                         + [f"{OCCURRENCE_MAX_SPIKES}+"])
    axis.set_xlabel("spikes in the window")
    axis.set_ylabel("fraction vocal")
    # The fitted slope is not repeated here -- the score panel beside this one reports it.
    axis.legend(frameon=False, fontsize=5, loc=legend_at, title="session", title_fontsize=5)
    # Sensitivity, which the panel otherwise hides: it plots P(vocal | count), so a unit that fires on
    # half its calls and almost never otherwise looks as convincing as one that fires on nearly all of
    # them. The score already reflects the difference -- both average over every window -- but neither
    # number says which of the two a unit is.
    vocal_rows = labels > 0.5
    fires_on = (occurrence["fraction_vocal_windows_with_spike"]
                if "fraction_vocal_windows_with_spike" in occurrence
                else float(np.mean(counts[vocal_rows] > 0)))
    axis.text(0.97, 0.05, f"fires on {100.0 * fires_on:.0f}% of USVs", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=6, color=TEXT_COLOR)
    axis.set_title("Vocalizing: P (vocal | spike count)", loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_tuning_map(axis, identity_section: dict, events: dict, settings: dict) -> None:
    """
    Description
    -----------
    The neuron's fitted rate over the acoustic torus, averaged across folds.

    Drawn from the stored Fourier coefficients and each fold's amplitude, so no decoder is refitted: the
    surface is ``exp(alpha * basis(x) . beta)``, the same construction the decoder inverts. Cells where no
    training call ever fell are left blank, since the decoder masks them too.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    identity_section (dict)
        The pickle's ``vocalization_identity`` section.
    events (dict)
        Assembled events, for the calls that define occupancy.
    settings (dict)
        The run's settings.

    Returns
    -------
    None
    """

    surface_settings = settings["vocal_decoding"]["tuning_surface"]
    grid_n = surface_settings["grid_n"]
    grid, _cell = torus_grid(grid_n, settings["vocalization_settings"]["torus_period"])
    basis = fourier_basis(grid, surface_settings["fourier_k"], settings["vocalization_settings"]["torus_period"])
    surfaces = [np.exp(np.clip(fold["alpha"] * (basis @ np.asarray(fold["coefficients"])[:-1]), -10.0, 5.0))
                for fold in identity_section["per_fold"]]
    surface = grid_image(np.mean(surfaces, axis=0), grid_n)

    occupancy = torus_occupancy(events["positions"], grid, surface_settings["kde_bandwidth"], settings["vocalization_settings"]["torus_period"])
    surface = np.where(grid_image(occupancy, grid_n) >= surface_settings["decode_grid_min_occupancy"],
                       surface, np.nan)
    colormap = plt.get_cmap(SEQUENTIAL_CMAP).copy()
    colormap.set_bad("#FFFFFF")
    # SQUARE: the two axes are the same torus coordinate, so a stretched map would misrepresent distance
    # on it -- a bump that is round in the space would read as elongated.
    image = axis.imshow(np.ma.masked_invalid(surface), origin="lower", cmap=colormap, aspect="equal",
                        extent=(0.0, 1.0, 0.0, 1.0))
    bar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
    bar.ax.tick_params(labelsize=5, length=0)
    bar.outline.set_visible(False)
    bar.set_label("relative firing rate", fontsize=5.5)
    axis.set_xlabel("acoustic dimension 1")
    axis.set_ylabel("acoustic dimension 2")
    axis.set_title("USV manifold tuning", loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_category_rates(axis, identity_section: dict, settings: dict, session_ids: list) -> None:
    """
    Description
    -----------
    The neuron's fitted firing rate per acoustic category: the categorical counterpart of the torus
    tuning map.

    Drawn from the per-fold rates the decoder stored (one Poisson rate per category, shrunk toward the
    unit's pooled training rate), so nothing is refitted. The bar is the mean over folds and each dot one
    fold's rate, so a category whose rate moves between held-out sessions is visible. Each fold has its
    own marker, named in the legend by the session it held out: a fold's rates are fitted on the OTHER
    sessions, so a dot is "the rates without this session", not this session's own rate. Rates are
    stored per analysis window and are drawn per second.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    identity_section (dict)
        The pickle's ``vocalization_identity_multinomial`` section.
    settings (dict)
        The run's settings, for the window width.
    session_ids (list)
        Session names in the decoder's order, so a fold's ``held_out`` index names its session.

    Returns
    -------
    None
    """

    width = float(settings["vocalization_settings"]["spike_window"]["width_seconds"])
    categories = [float(value) for value in identity_section["categories"]]
    rates = np.asarray([fold["rates"] for fold in identity_section["per_fold"]], dtype=np.float64) / width
    positions = np.arange(len(categories))
    colors = [CATEGORY_COLORS[int(value) - 1] for value in categories]
    axis.bar(positions, rates.mean(axis=0), width=0.65, color=colors, edgecolor="none")
    for fold, fold_rates in zip(identity_section["per_fold"], rates, strict=True):
        held_out = int(fold["held_out"])
        axis.scatter(positions, fold_rates, s=9, marker=SESSION_MARKERS[held_out % len(SESSION_MARKERS)],
                     color="#FFFFFF", edgecolor=TEXT_COLOR, linewidth=0.5, zorder=3,
                     label=str(session_ids[held_out]).split("_")[-1])
    axis.legend(title="held out", fontsize=4.5, title_fontsize=4.5, loc="upper left", frameon=False,
                handletextpad=0.2, borderaxespad=0.2)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{int(value)}" for value in categories])
    axis.set_xlabel("USV category")
    axis.set_ylabel("firing rate (sp/s)")
    axis.set_title("USV category tuning", loc="left", fontsize=PANEL_TITLE_SIZE)


def category_label_points(positions: np.ndarray, labels: np.ndarray, categories: list,
                          settings: dict) -> list:
    """
    Description
    -----------
    Where to write each category's number on the torus scatter: the grid point inside the category's own
    local-majority region that maximises its local USV count times its margin over the next category, so
    the number sits where the category is both dense and clearly ahead.

    Each grid point carries every category's share of the wrapped-Gaussian-weighted count of this unit's
    nearby USVs (smoothed at ``ERROR_SIGMA``, on the gain maps' grid and occupancy mask). Kept from the
    first version of the borders because its number positions read best (user, 2026-10-29); the bundle's
    own ``centers`` put two of the four numbers on the plot edge. Share alone also put numbers on the
    edge, where a category is alone because it is barely there; count alone put category 1's number on
    its border with category 3 (cl0401), where both are dense.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n, 2)`` torus positions of the labelled USVs.
    labels (np.ndarray)
        ``(n,)`` category label per USV.
    categories (list)
        The categories, in order.
    settings (dict)
        The run's settings.

    Returns
    -------
    label_points (list)
        Per category, an ``(x, y)`` position, or None if the category wins nowhere.
    """

    surface_settings = settings["vocal_decoding"]["tuning_surface"]
    period = settings["vocalization_settings"]["torus_period"]
    grid, _cell = torus_grid(ERROR_DISPLAY_POINTS, period)
    delta = (grid[:, None, :] - positions[None, :, :] + 0.5 * period) % period - 0.5 * period
    weights = np.exp(-0.5 * np.sum(delta ** 2, axis=2) / ERROR_SIGMA ** 2)
    votes = np.stack([weights[:, labels == float(value)].sum(axis=1) for value in categories], axis=1)
    totals = votes.sum(axis=1)
    winner = np.argmax(votes, axis=1)
    occupancy = torus_occupancy(positions, grid, surface_settings["kde_bandwidth"], period)
    shown = (occupancy >= surface_settings["decode_grid_min_occupancy"]) & (totals > 0.0)
    share = np.divide(votes, totals[:, None], out=np.zeros_like(votes), where=totals[:, None] > 0.0)
    label_points = []
    for code in range(len(categories)):
        own = shown & (winner == code)
        if not own.any():
            label_points.append(None)
            continue
        others = np.delete(share, code, axis=1).max(axis=1)
        best = np.flatnonzero(own)[np.argmax(votes[own, code] * (share[own, code] - others[own]))]
        label_points.append((float(grid[best, 0]), float(grid[best, 1])))
    return label_points


def draw_category_scatter(axis, positions: np.ndarray, labels: np.ndarray, categories: list,
                          settings: dict) -> None:
    """
    Description
    -----------
    Every USV at its torus position, coloured by its USV category, with the category borders drawn in
    thin black and each category numbered: where the categories the decoder works with sit on the
    manifold the torus panels are drawn on.

    It sits in the category row so the category bars can be read against the torus panels: the
    categories are regions of this same manifold, and the scatter shows which region is which. USVs
    without a category label are left out, as the decoder leaves them out. Drawn in a shuffled order, so
    the last category plotted does not cover the others where they meet.

    The borders and the number positions come from the QLVM category bundle
    (``os_utils.load_qlvm_category_bundle``): the corpus-wide label grid that DEFINES ``qlvm_category``
    (200 x 200 over the whole torus), outlined with the repository's ``draw_category_outlines``, the
    helper every other QLVM figure uses. So the borders are complete, the same on every unit, and are the
    categories' actual definition. They were first drawn from a vote of the unit's own USVs, which left
    them broken wherever the unit had few USVs (user: unacceptable). The share of this unit's USVs whose
    label matches the grid at their position is printed as a check. The numbers replace a legend, which
    would collide with the confusion matrix beside it, and are placed by :func:`category_label_points`.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    positions (np.ndarray)
        ``(n, 2)`` torus positions of the unit's USVs.
    labels (np.ndarray)
        ``(n,)`` category label per USV, NaN where unlabelled.
    categories (list)
        The categories the decoder used, as stored in the section.
    settings (dict)
        The run's settings, for the number positions.

    Returns
    -------
    None
    """

    positions = np.asarray(positions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    keep = np.isin(labels, [float(value) for value in categories])
    order = np.random.default_rng(BOOTSTRAP_SEED).permutation(np.flatnonzero(keep))
    colors = np.asarray([CATEGORY_COLORS[int(value) - 1] for value in labels[order]])
    axis.scatter(positions[order, 0], positions[order, 1], s=1.2, c=colors, linewidths=0, alpha=0.8,
                 rasterized=True)

    bundle = load_qlvm_category_bundle()
    label_grid = np.asarray(bundle["label_grid"])
    resolution = label_grid.shape[0]
    cells = np.clip(np.floor(positions[keep] * resolution).astype(int), 0, resolution - 1)
    agreement = float(np.mean(label_grid[cells[:, 1], cells[:, 0]] == labels[keep]))
    print(f"  category map vs this unit's USV labels: {100.0 * agreement:.1f}% agree")
    draw_category_outlines(axis, np.asarray(bundle["axis"]), np.asarray(bundle["axis"]), label_grid,
                           colors=CATEGORY_BORDER_COLOR, linewidths=CATEGORY_BORDER_WIDTH, zorder=3)
    for value, point in zip(categories, category_label_points(positions[keep], labels[keep], categories,
                                                              settings), strict=True):
        if point is None:
            continue
        axis.text(point[0], point[1], f"{int(value)}", fontsize=6, fontweight="bold",
                  color=TEXT_COLOR, ha="center", va="center", zorder=4,
                  path_effects=[patheffects.withStroke(linewidth=1.5, foreground="#FFFFFF")])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xlabel("acoustic dimension 1")
    axis.set_ylabel("acoustic dimension 2")
    axis.set_title(f"USVs by category (n = {keep.sum():,})", loc="left", fontsize=PANEL_TITLE_SIZE)


def category_confusion(unit_uid: str, events: dict, identity_section: dict, settings: dict) -> dict:
    """
    Description
    -----------
    The held-out category posterior per USV, rebuilt from the stored per-fold rates and amplitudes, and
    averaged by true category: how far the neuron moves the decoder from the prior, toward or away from
    each category, for USVs of each true category.

    No refit. The events, codes and folds are the dispatcher's ``run_vocalization_identity`` calls in the
    same order; each fold's posterior is ``category_posterior`` with that fold's stored ``rates`` and
    ``alpha`` and the exposure the decoder computes. The rebuild must reproduce the stored per-USV gain
    (log posterior minus log prior at the true category) to ``CATEGORY_REBUILD_TOLERANCE``, or nothing is
    drawn.

    Drawn as the mean of log(posterior / prior), not the mean posterior probability. MEASURED on cl0401
    (20250919 imec1): the mean posterior rows read 0.31-0.38 / 0.10-0.11 / 0.22-0.23 / 0.29-0.36 against a
    prior of 0.34 / 0.11 / 0.23 / 0.32, so a probability matrix shows the category frequencies and hides
    the neuron. The log ratio removes the prior, and its diagonal, weighted by row counts, is exactly the
    tested gain.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem, for the message.
    events (dict)
        The assembled vocal events (``vocal_event_data``).
    identity_section (dict)
        The pickle's ``vocalization_identity_multinomial`` section.
    settings (dict)
        The run's settings.

    Returns
    -------
    confusion (dict)
        ``matrix`` (``(K, K)`` mean log posterior/prior, rows true, columns decoded), ``row_counts``,
        ``categories`` and the rebuilt ``gain``.
    """

    surface = settings["vocal_decoding"]["category_surface"]
    floor = surface["mean_count_floor"]
    codes, kept, values = category_codes(np.asarray(events["category_labels"]))
    counts = np.asarray(events["counts"], dtype=np.float64)[kept]
    session_index = np.asarray(events["session_index"])[kept]
    folds = category_context(codes, session_index, np.asarray(events["vocalization_start"])[kept],
                             int(settings["data_sufficiency"]["single_session_blocks"]))
    n_categories = values.size
    log_ratio = np.full((codes.size, n_categories), np.nan)
    stored = iter(identity_section["per_fold"])
    for fold in folds:
        train_rows, test_rows = fold["train_rows"], fold["test_rows"]
        # The same skip the decoder applies, so folds line up with the stored per-fold entries.
        if train_rows.size < surface["min_train_events"] or test_rows.size < surface["min_test_events"]:
            continue
        entry = next(stored)
        if entry["held_out"] != fold["held_out"]:
            msg = f"{unit_uid}: rebuilt fold {fold['held_out']} lines up with stored fold {entry['held_out']}"
            raise ValueError(msg)
        exposure = (max(float(counts[test_rows].mean()), floor) / max(float(counts[train_rows].mean()), floor))
        posterior = category_posterior(counts[test_rows], np.asarray(entry["rates"], dtype=np.float64),
                                       fold["prior"], float(entry["alpha"]), exposure, floor)
        log_ratio[test_rows] = posterior - np.log(fold["prior"])[None, :]

    at_truth = log_ratio[np.arange(codes.size), codes]
    difference = float(np.nanmax(np.abs(at_truth - np.asarray(identity_section["per_event_gain"]))))
    print(f"  {unit_uid}: category posterior rebuilt, |diff| vs stored per-USV gain {difference:.1e}")
    if not difference <= CATEGORY_REBUILD_TOLERANCE:
        msg = f"{unit_uid}: the rebuilt category posterior does not reproduce the stored per-USV gain"
        raise ValueError(msg)
    matrix = np.array([np.nanmean(log_ratio[codes == code], axis=0) for code in range(n_categories)])
    return {"matrix": matrix, "row_counts": np.bincount(codes, minlength=n_categories),
            "categories": values.tolist(), "gain": float(np.nanmean(at_truth))}


def draw_category_confusion(axis, confusion: dict) -> None:
    """
    Description
    -----------
    True category against decoded category: the mean held-out change from the prior, in nats, that the
    neuron's count gives each category, for USVs of each true category (:func:`category_confusion`).

    A positive diagonal is the neuron pointing at the right category; an off-diagonal cell says which
    categories the neuron confuses a USV with (positive) or rules out (negative). With one spike count per
    USV the decoder can only order categories by rate, so categories with similar rates share a sign
    pattern. The cell value log(posterior / prior) equals log p(count | category) - log p(count), a
    log-likelihood ratio, hence the colour-bar label. The colour range is this unit's own peak rounded up,
    as on the gain maps. Tick labels carry
    the category colours, the key for the scatter beside it.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    confusion (dict)
        From :func:`category_confusion`.

    Returns
    -------
    None
    """

    matrix = confusion["matrix"]
    n_categories = matrix.shape[0]
    peak = one_significant_figure_ceiling(float(np.nanmax(np.abs(matrix))))
    image = axis.imshow(matrix, cmap=GAIN_DIVERGING_CMAP, vmin=-peak, vmax=peak, aspect="equal")
    for row in range(n_categories):
        for column in range(n_categories):
            value = matrix[row, column]
            axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=4.5,
                      color="#FFFFFF" if abs(value) > 0.6 * peak else TEXT_COLOR)
    names = [f"{int(value)}" for value in confusion["categories"]]
    axis.set_xticks(range(n_categories), names)
    axis.set_yticks(range(n_categories), names)
    for ticks in (axis.get_xticklabels(), axis.get_yticklabels()):
        for tick, value in zip(ticks, confusion["categories"], strict=True):
            tick.set_color(CATEGORY_COLORS[int(value) - 1])
            tick.set_fontweight("bold")
    axis.tick_params(length=0)
    axis.set_xlabel("decoded USV category", fontsize=5, labelpad=1)
    axis.set_ylabel("true USV category", fontsize=5, labelpad=1)
    for spine in axis.spines.values():
        spine.set_visible(False)
    # The matrix is limited by its column's width, so it is grown into the gaps beside it, and the
    # colour bar sits underneath as an inset rather than taking space from the matrix.
    box = axis.get_position()
    grow = CONFUSION_GROWTH
    axis.set_position([box.x0 - 0.5 * (grow - 1.0) * box.width, box.y0 - 0.5 * (grow - 1.0) * box.height,
                       grow * box.width, grow * box.height])
    bar = axis.figure.colorbar(image, cax=axis.inset_axes([0.1, -0.36, 0.8, 0.05]), orientation="horizontal")
    bar.ax.tick_params(labelsize=5, length=0)
    bar.outline.set_visible(False)
    bar.set_label("log-likelihood ratio (nats)", fontsize=5)
    axis.set_title("Category confusion", loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_per_event_gain(axis, identity_section: dict, title: str = "Decoding vocal position") -> None:
    """
    Description
    -----------
    Distribution of the per-call decoding gain, so a result carried by a handful of calls is visible.

    The tested number is the mean of these, and a mean says nothing about whether the calls agree. The
    share above zero and the median against the mean do.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    identity_section (dict)
        The pickle's ``vocalization_identity`` section.

    Returns
    -------
    None
    """

    # The permuted distribution is NOT drawn beside this one. Permuting positions leaves nearly every call
    # at no gain, so at this resolution the null is a spike hidden under the observed peak -- it carries
    # no width to compare against. The tested comparison lives in the score panel to the right, where the
    # null is a distribution of means and has a scale worth seeing.
    gains = np.asarray(identity_section["per_event_gain"])
    gains = gains[np.isfinite(gains)]
    axis.hist(gains, bins=40, color=VOCAL_COLOR, edgecolor="none")
    axis.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    axis.axvline(float(np.mean(gains)), color=TEXT_COLOR, linewidth=1.0)
    axis.set_xlabel("gain per USV (nats)")
    axis.set_ylabel("vocalizations")
    axis.legend(handles=[Line2D([], [], color=TEXT_COLOR, linewidth=1.0,
                                label=f"mean {float(np.mean(gains)):+.3f} nats")],
                frameon=False, fontsize=5, loc="upper left")
    axis.text(0.03, 0.86, f"{100.0 * np.mean(gains > 0):.0f}% positive", transform=axis.transAxes,
              ha="left", va="top", fontsize=5, color=TEXT_COLOR)
    axis.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE)


def nested_event_data(unit_uid: str, paths: FigurePaths, identity: dict, settings: dict) -> dict:
    """
    Description
    -----------
    What the nested row needs per USV and per fold that the artifact does not store, rebuilt and cached:
    the USV positions the maps are drawn on, and the behaviour prior's own gain at every USV.

    Everything TESTED is read from the section. What is rebuilt here is the event assembly, the cohort
    control's prediction for each USV and the behaviour prior built from it -- the same calls the
    dispatcher makes, in the same order, so the per-USV arrays line up with the section's
    ``per_event_gain`` row for row. Two things are checked rather than assumed: the rebuilt
    concentrations must equal the stored ``kappas`` fold for fold, and the rebuilt behaviour rung is
    printed beside the stored one by the caller.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    data (dict)
        ``positions`` ``(n, 2)``; ``session_index`` ``(n,)``; ``session_ids``; ``behaviour_gain``
        ``(n,)`` -- the log behaviour prior at the truth minus the log occupancy prior there, NaN where
        the truth falls in a masked cell; ``per_fold_behaviour_over`` and ``per_fold_held_out_session``,
        one entry per held-out session in fold order; ``kappas`` per fold.
    """

    cache_path = paths.cache_directory / f"{unit_uid}_nested_events_manifold.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)

    nested = nested_settings(settings)
    kinematic = settings["kinematic_encoding"]
    vocalization = settings["vocalization_settings"]
    window = vocalization["spike_window"]
    events = assemble_unit_vocal_events(identity, str(paths.data_root), copy.deepcopy(settings),
                                        window["pre_offset_seconds"], window["width_seconds"])
    per_session = assemble_unit_sessions(identity, str(paths.data_root), settings["kinematic_features"],
                                         kinematic["history_pre_seconds"], kinematic["clean_post_seconds"],
                                         vocalization["clean_against"],
                                         vocalization)
    fps = float(per_session[identity["courtship_sessions"][0]]["fps"])
    design = nested_design(events, per_session, reduced_model_features(settings, "manifold"),
                           int(np.floor(kinematic["history_pre_seconds"] * fps)), nested["sigma_floor"])
    design, _n_lags = apply_behaviour_control(design, load_control_model(settings, "manifold"),
                                              str(identity["rec_date"]), settings)
    del per_session
    positions = np.asarray(design["positions"], dtype=np.float64)
    session_index = np.asarray(design["session_index"])
    folds = decoding_context(positions, session_index, settings["vocal_decoding"],
                             settings["vocalization_settings"]["torus_period"],
                             np.asarray(events["vocalization_start"])[design["kept"]],
                             int(settings["data_sufficiency"]["single_session_blocks"]))
    kappas, priors = behaviour_priors(folds, design["behaviour_control"]["predicted_positions"],
                                      positions, nested)
    session_ids = list(events["session_ids"])
    behaviour_gain = np.full(positions.shape[0], np.nan)
    per_fold_behaviour_over, per_fold_held_out = [], []
    for fold, prior in zip(folds, priors, strict=True):
        rows = np.asarray(fold["test_rows"])
        cells = np.asarray(fold["truth_cell"])[rows]
        value = np.asarray(prior)[rows, cells] - np.asarray(fold["log_prior"])[cells]
        value[~np.asarray(fold["mask"])[cells]] = np.nan
        behaviour_gain[rows] = value
        per_fold_behaviour_over.append(float(finite_mean(value)))
        per_fold_held_out.append(session_ids[int(session_index[rows[0]])])
    data = {"positions": positions, "session_index": session_index, "session_ids": session_ids,
            "behaviour_gain": behaviour_gain, "per_fold_behaviour_over": per_fold_behaviour_over,
            "per_fold_held_out_session": per_fold_held_out, "kappas": [float(k) for k in kappas]}
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(data, cache_file)
    return data


def category_nested_event_data(unit_uid: str, paths: FigurePaths, identity: dict, settings: dict) -> dict:
    """
    Description
    -----------
    The categorical counterpart of :func:`nested_event_data`: per scored USV, its torus position and the
    behaviour prior's own gain, rebuilt and cached.

    The calls are the dispatcher's ``run_category_nested_decoding`` in the same order -- event assembly,
    the design on the categorical control's features, the held-out day's base-rate-free category
    probabilities, the unit's category codes, the folds and the tuned tilts -- so the per-USV arrays line
    up row for row with the section's ``per_event_gain``. The behaviour gain at a USV is the log of the
    behaviour-tilted category prior at its true category minus the log of the fold's training category
    frequency there: the ladder's first step, per USV. Positions are carried only so the gains can be
    drawn on the same torus as the position target's; the category decoder never sees them.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    data (dict)
        As :func:`nested_event_data`: ``positions``, ``session_index``, ``session_ids``,
        ``behaviour_gain``, ``per_fold_behaviour_over``, ``per_fold_held_out_session`` and ``kappas``.
    """

    cache_path = paths.cache_directory / f"{unit_uid}_nested_events_multinomial.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)

    nested = nested_settings(settings)
    kinematic = settings["kinematic_encoding"]
    vocalization = settings["vocalization_settings"]
    window = vocalization["spike_window"]
    events = assemble_unit_vocal_events(identity, str(paths.data_root), copy.deepcopy(settings),
                                        window["pre_offset_seconds"], window["width_seconds"])
    per_session = assemble_unit_sessions(identity, str(paths.data_root), settings["kinematic_features"],
                                         kinematic["history_pre_seconds"], kinematic["clean_post_seconds"],
                                         vocalization["clean_against"], vocalization)
    fps = float(per_session[identity["courtship_sessions"][0]]["fps"])
    design = nested_design(events, per_session, reduced_model_features(settings, "multinomial"),
                           int(np.floor(kinematic["history_pre_seconds"] * fps)), nested["sigma_floor"])
    del per_session
    control = load_control_model(settings, "multinomial")
    evidence = category_control_evidence(control, str(identity["rec_date"]), design["behaviour"], settings)
    codes, labelled, values = category_codes(np.asarray(events["category_labels"])[design["kept"]])
    classes = [float(c) for c in np.asarray(control["classes"])]
    columns = [classes.index(float(v)) for v in values]
    log_evidence = np.log(np.maximum(evidence[labelled][:, columns], 1e-300))
    session_index = np.asarray(design["session_index"])[labelled]
    times = np.asarray(events["vocalization_start"])[design["kept"]][labelled]
    folds = category_context(codes, session_index, times, int(settings["data_sufficiency"]["single_session_blocks"]))
    kappas, tilts = category_behaviour_tilts(folds, codes, log_evidence, nested["behaviour_prior"]["kappa_grid"])
    surface = settings["vocal_decoding"]["category_surface"]
    session_ids = list(events["session_ids"])
    behaviour_gain = np.full(codes.size, np.nan)
    per_fold_behaviour_over, per_fold_held_out, kept_kappas = [], [], []
    for fold, tilt, kappa in zip(folds, tilts, kappas, strict=True):
        # The same skip the decoder applies, so the folds line up with the section's per-fold entries.
        if (fold["train_rows"].size < surface["min_train_events"]
                or fold["test_rows"].size < surface["min_test_events"]):
            continue
        rows = np.asarray(fold["test_rows"])
        truth = codes[rows]
        prior = tilted_category_prior(fold["prior"], np.asarray(tilt)[rows])
        value = np.log(prior[np.arange(truth.size), truth]) - np.log(fold["prior"][truth])
        behaviour_gain[rows] = value
        per_fold_behaviour_over.append(float(np.mean(value)))
        per_fold_held_out.append(session_ids[int(session_index[rows[0]])])
        kept_kappas.append(float(kappa))
    data = {"positions": np.asarray(design["positions"], dtype=np.float64)[labelled],
            "session_index": session_index, "session_ids": session_ids, "behaviour_gain": behaviour_gain,
            "per_fold_behaviour_over": per_fold_behaviour_over, "per_fold_held_out_session": per_fold_held_out,
            "kappas": kept_kappas}
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(data, cache_file)
    return data


def check_nested_rebuild(unit_uid: str, nested: dict, rebuilt: dict) -> None:
    """
    Description
    -----------
    Refuse to draw the nested row from a rebuild that does not reproduce the run.

    The maps and the ladder's per-fold dots come from arrays rebuilt here, not from the artifact, so
    the rebuild has to be shown to be the run's own: the tuned concentration per fold must match
    ``kappas`` exactly (it is a grid argmax, so any drift in the design or the control flips it to a
    different grid point rather than moving it slightly), and the rebuilt behaviour rung must match
    the stored one to rounding.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem, for the message.
    nested (dict)
        The pickle's ``nested_vocal_decoding_<target>`` section.
    rebuilt (dict)
        From :func:`nested_event_data`.

    Returns
    -------
    None
    """

    if list(rebuilt["kappas"]) != [float(k) for k in nested["kappas"]]:
        msg = (f"{unit_uid}: rebuilt concentrations {rebuilt['kappas']} differ from the stored "
               f"{nested['kappas']}; the rebuild is not the run's own design")
        raise ValueError(msg)
    if int(np.asarray(nested["per_event_gain"]).size) != int(rebuilt["positions"].shape[0]):
        msg = (f"{unit_uid}: rebuilt {rebuilt['positions'].shape[0]} events, the section stores "
               f"{np.asarray(nested['per_event_gain']).size}")
        raise ValueError(msg)
    rebuilt_rung = float(finite_mean(rebuilt["behaviour_gain"]))
    difference = abs(rebuilt_rung - float(nested["behaviour_over_no_behaviour"]))
    print(f"  {unit_uid}: behaviour rung rebuilt {rebuilt_rung:+.5f} vs stored "
          f"{nested['behaviour_over_no_behaviour']:+.5f} (|diff| {difference:.1e})")
    if difference > 1e-6:
        msg = f"{unit_uid}: the rebuilt behaviour rung does not reproduce the stored one"
        raise ValueError(msg)


def draw_nested_ladder(axis, nested: dict, rebuilt: dict, title: str = "Predicting vocal content") -> None:
    """
    Description
    -----------
    What each model knows about where a USV sits on the acoustic manifold: nothing, behaviour, behaviour
    and the neuron.

    The claim this row makes is that the neuron BEATS THE BEHAVIOUR MODEL, conditional on that model
    predicting better than chance, and the ladder shows both halves at once. The first step is the
    condition and the second is the test. It is not a claim that the neuron carries something beyond
    behaviour: a unit reading out a behaviour the model happens to lack would climb the same step, and
    separating the two needs a null that holds spike-to-behaviour and behaviour-to-USV while breaking
    only spike-to-USV, which cannot be had without another model.

    The currency is the POSTERIOR form's: the log density at the true position, in nats per USV, under
    the occupancy prior (first rung, the reference at zero), under the behaviour prior -- occupancy
    tilted by a von Mises at the cohort control's prediction, its concentration tuned by inner LOSO so
    it can never fall below the occupancy prior beyond tuning noise -- and under the neuron's posterior
    on top of that prior. The last step is the tested ``added``. Identity's gain is the same quantity
    under the first prior alone, so this ladder and the identity panel are one statistic under two
    priors.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    nested (dict)
        The pickle's ``nested_vocal_decoding_<target>`` section.
    rebuilt (dict)
        From :func:`nested_event_data`, for the per-fold behaviour rung.

    Returns
    -------
    None
    """

    # The reference is the occupancy prior, which is exactly zero by construction -- not a common mode
    # removed after the fact. Every rung is a paired difference at the same USVs, so the per-fold dots
    # are the spread of the thing the panel is about and nothing else.
    rungs = [("no behavior", 0.0, REJECTED_COLOR),
             ("behavior", float(nested["behaviour_over_no_behaviour"]), ADDITIVE_COLOR),
             ("+ neuron", float(nested["full_over_no_behaviour"]), QUIET_COLOR)]
    positions = np.arange(len(rungs))
    axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":", zorder=0)
    axis.plot(positions, [score for _name, score, _color in rungs], color=REFERENCE_COLOR,
              linewidth=0.8, zorder=1)

    # Per-FOLD values as the dots, the convention the identity panel already uses: each dot is one
    # held-out session's own mean, the filled marker is the pooled value over every USV. A fold's
    # "+ neuron" value is its behaviour rung plus its own added score, both means over the same USVs.
    behaviour_dots = np.asarray(rebuilt["per_fold_behaviour_over"], dtype=np.float64)
    added_dots = np.asarray([fold["added"] for fold in nested["per_fold"]], dtype=np.float64)
    spreads = [np.zeros(behaviour_dots.size), behaviour_dots, behaviour_dots + added_dots]
    for position, (_name, score, color) in zip(positions, rungs, strict=True):
        if position > 0:
            axis.plot([position, position], [spreads[position].min(), spreads[position].max()],
                      color=TEXT_COLOR, linewidth=0.8, zorder=2)
            axis.scatter(np.full(spreads[position].size, position), spreads[position], s=6,
                         color="#FFFFFF", edgecolor=TEXT_COLOR, linewidth=0.4, zorder=3)
        axis.scatter([position], [score], s=34, color=color, edgecolor=TEXT_COLOR, linewidth=0.5,
                     zorder=4)

    # The DELTA alone under the point it lands on: the tick beneath already names the point, and the
    # delta sign is what marks the number as a step rather than the rung's own value.
    lows = [min(0.0, float(spread.min())) for spread in spreads]
    highs = [max(0.0, float(spread.max())) for spread in spreads]
    baseline = min(lows)
    for start_index, value in enumerate([nested["behaviour_over_no_behaviour"], nested["added"]]):
        axis.annotate(f"Δ {value:+.3f}", xy=(start_index + 1, baseline), xytext=(0, -7),
                      textcoords="offset points", fontsize=5, color=TEXT_COLOR, ha="center", va="top")
    axis.text(0.03, 0.97, "κ " + " / ".join(f"{k:g}" for k in nested["kappas"]),
              transform=axis.transAxes, ha="left", va="top", fontsize=5, color=TEXT_COLOR)
    axis.set_xticks(positions)
    axis.set_xticklabels([name for name, _score, _color in rungs], fontsize=6)
    axis.set_xlim(-0.75, len(rungs) - 0.25)
    span = max(max(highs) - baseline, 1e-6)
    axis.set_ylim(baseline - 0.22 * span, max(highs) + 0.18 * span)
    axis.set_ylabel("Δ log-likelihood (nats / USV)")
    axis.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE)


# Set by SIGNAL, not by nominal resolution. Effective resolution is sqrt(n / N_eff), and sigma 0.03
# delivers the ruled 10x10 (11.0 cells per dimension) where 0.05 delivers about 6x6 -- but a permutation
# test over the per-call values, shuffled across calls and smoothed identically, says the finer setting
# is mostly noise: the share of the map's spatial variance that shuffling does NOT reproduce runs 43.5%
# at sigma 0.03 against 71.7% at 0.05 and 82.1% at 0.08 (cl0401). Resolution the data cannot support is
# not resolution, and it reads as patchiness a viewer will try to interpret.
ERROR_SIGMA = 0.05
ERROR_MIN_EFFECTIVE_CALLS = 10.0
# The gain panels follow the behavioural torus filter atlas: the shared diverging colormap of the
# visualization settings on main, and a symmetric limit set by EACH panel's own peak rounded up to one
# significant figure. A fixed +/-0.1 clipped 12-64% of every rendered field (peaks 0.19-0.66 nats per
# USV across the four example units), so panels are scaled to themselves and are not comparable across
# units by colour -- read the colour-bar ends.
GAIN_DIVERGING_CMAP = "BrBG_r"
# How many points the error fields are DRAWN at -- not their resolution, which is ERROR_SIGMA. Kept
# separate from the tuning surface's own grid so the two can be compared at different display grids;
# the occupancy that masks them is evaluated per POINT, so its threshold means the same at any grid.
ERROR_DISPLAY_POINTS = 60
ERROR_SHUFFLE_DRAWS = 1000
ERROR_SHUFFLE_SEED = 0


def grid_image(values: np.ndarray, grid_n: int) -> np.ndarray:
    """
    Description
    -----------
    Turn per-point values on the torus grid into an image whose columns run along acoustic dimension 1
    and whose rows run along acoustic dimension 2, which is what ``imshow(origin="lower")`` and
    ``contour`` expect for x = dimension 1, y = dimension 2.

    ``torus_grid`` lists its points with dimension 1 as the SLOW index, so a plain
    ``reshape(grid_n, grid_n)`` puts dimension 1 on the rows -- and rows are drawn along y. Every torus
    map was drawn that way until 2026-10-29, i.e. transposed against its own axis labels; the category
    scatter, which plots positions directly, is what exposed it (a USV at (0.2, 0.8) landed at x = 0.8,
    y = 0.2 on the maps). Every map now goes through this one function.

    Parameters
    ----------
    values (np.ndarray)
        ``(grid_n * grid_n,)`` values in ``torus_grid`` order.
    grid_n (int)
        Points per dimension.

    Returns
    -------
    image (np.ndarray)
        ``(grid_n, grid_n)`` with ``image[row, column]`` at dimension 2 = row, dimension 1 = column.
    """

    return np.asarray(values).reshape(grid_n, grid_n).T


def grid_centres(grid_n: int) -> np.ndarray:
    """
    Description
    -----------
    Cell-centre coordinates of the display grid along one dimension, matching ``torus_grid``, for
    ``contour``. ``np.linspace(0, 1, grid_n)`` would place each contour half a cell off the image.

    Parameters
    ----------
    grid_n (int)
        Points per dimension.

    Returns
    -------
    centres (np.ndarray)
        ``(grid_n,)`` centres in [0, 1).
    """

    return (np.arange(grid_n) + 0.5) / grid_n


def smoothed_manifold_field(positions: np.ndarray, values: np.ndarray, settings: dict) -> np.ndarray:
    """
    Description
    -----------
    A wrapped-Gaussian local average of a per-call value, on the SAME grid and mask as the tuning map.

    Sharing the grid is not tidiness. Built on its own grid this panel put its white stretches in
    different places from the tuning map beside it, which invites the reader to compare two pictures of
    one manifold that do not line up -- and a separately built grid could silently transpose the two
    acoustic dimensions, so that "dimension 1" meant one axis here and the other axis there.

    The VALUES are smoothed at ``ERROR_SIGMA``, which is set to deliver the ruled 10x10 resolution --
    see its note; it is measured, not derived from the cell width. The MASK is the tuning surface's own
    occupancy rule, which is the more permissive of the two candidates. MEASURED on the example units,
    that permissiveness costs little: 1% of shown points fall below ten effective calls and none below
    five, with a median of 68 and 117. ``ERROR_MIN_EFFECTIVE_CALLS`` records the threshold that was
    checked against.

    Parameters
    ----------
    positions (np.ndarray)
        ``(n, 2)`` true torus positions of the calls.
    values (np.ndarray)
        ``(n,)`` the per-call quantity to average.
    settings (dict)
        The run's settings, for the tuning surface's grid and occupancy rule.

    Returns
    -------
    field, signal_share (tuple)
        ``(grid_n, grid_n)`` smoothed values with NaN where the tuning map is also blank, and the
        fraction of the field's spatial variance that SHUFFLING does not reproduce.

    Notes
    -----
    The share is the panel's own guard, and it is needed. Shuffling the per-USV values across USVs
    breaks the link to position while leaving both distributions intact, so a field smoothed from
    shuffled values is what this map looks like when there is nothing there. MEASURED on the example
    units it runs 72% for one and 16% for another -- and the 16% panel still renders confident blue and
    red patches that a reader would interpret. A map that means nothing must say so on its face.

    WHERE THE GEOMETRY ENTERS, since the variance itself is computed without any: shuffling leaves the
    SET of per-USV values untouched, so their own spread is identical either way. What changes is what
    SMOOTHING does to them. Averaging values scattered at random shrinks their spread by about the root
    of the effective count; averaging values that cluster spatially shrinks it far less. MEASURED on
    cl0401: the per-USV spread is 0.0902 and survives smoothing at 0.0239 (a 3.8x shrink) where the
    shuffled fields land at 0.0127 (7.1x, close to the sqrt(68) = 8.2 that independent values predict).
    So the statistic is a test of spatial autocorrelation in the values, read through the same kernel
    the panel is drawn with.
    """

    surface_settings = settings["vocal_decoding"]["tuning_surface"]
    grid_n = ERROR_DISPLAY_POINTS
    period = settings["vocalization_settings"]["torus_period"]
    grid, _cell = torus_grid(grid_n, period)
    delta = (grid[:, None, :] - positions[None, :, :] + 0.5 * period) % period - 0.5 * period
    weights = np.exp(-0.5 * np.sum(delta ** 2, axis=2) / ERROR_SIGMA ** 2)
    totals = weights.sum(axis=1)
    field = np.divide(weights @ values, totals, out=np.full(grid.shape[0], np.nan),
                      where=totals > 0.0)
    occupancy = torus_occupancy(positions, grid, surface_settings["kde_bandwidth"], settings["vocalization_settings"]["torus_period"])
    shown = occupancy >= surface_settings["decode_grid_min_occupancy"]
    field[~shown] = np.nan

    generator = np.random.default_rng(ERROR_SHUFFLE_SEED)
    shuffled = np.column_stack([generator.permutation(values) for _ in range(ERROR_SHUFFLE_DRAWS)])
    null = np.divide(weights @ shuffled, totals[:, None], out=np.zeros((grid.shape[0], ERROR_SHUFFLE_DRAWS)),
                     where=totals[:, None] > 0.0)[shown]
    observed_spread = float(np.nanstd(field[shown]))
    share = (0.0 if observed_spread <= 0.0
             else max(0.0, 1.0 - (float(null.std(axis=0).mean()) / observed_spread) ** 2))
    return grid_image(field, grid_n), share


def one_significant_figure_ceiling(value: float) -> float:
    """
    Description
    -----------
    ``value`` rounded UP to one significant figure (0.0023 -> 0.003, 0.657 -> 0.7), the rule the torus
    filter atlas uses for its shared colour limit, so the peak always sits inside the scale.

    Parameters
    ----------
    value (float)
        A non-negative magnitude.

    Returns
    -------
    ceiling (float)
        The rounded-up value; 0.1 for a zero or non-finite input.
    """

    if not np.isfinite(value) or value <= 0.0:
        return 0.1
    decade = 10.0 ** np.floor(np.log10(value))
    return float(np.ceil(value / decade) * decade)


def draw_manifold_field(axis, field: np.ndarray, colormap: str, label: str, title: str,
                        symmetric: bool, signal_share: float | None = None,
                        show_y_label: bool = True) -> None:
    """
    Description
    -----------
    One smoothed torus field, drawn exactly as the tuning map is so the three panels can be compared.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    field (np.ndarray)
        The smoothed values, NaN where masked.
    colormap (str)
        Matplotlib colormap name.
    label (str)
        Colour-bar label.
    title (str)
        Panel title.
    symmetric (bool)
        Centre the scale on zero, so the sign reads off the colour.
    signal_share (float)
        The fraction of the field's spatial variance that shuffling does not reproduce, printed in the
        corner, or None for a panel that is not an estimate of anything (the call density is not).
    show_y_label (bool)
        Only the leftmost of the three manifold panels carries it; on the others it collided with the
        neighbour's colour bar, and all three share one acoustic dimension anyway.

    Returns
    -------
    None
    """

    # ROBUST limits, not the extremes. Scaling to max|value| let a handful of cells at the mask edge
    # set the whole range and squash the body of the map toward white -- MEASURED at 2.4x and 5.0x the
    # 95th percentile, which is why one field looked unrecognisable between two bandwidths whose
    # contents correlate at r = +0.90. The outermost 2% is clipped, without drawing caps for it.
    finite = field[np.isfinite(field)]
    limits = {}
    if symmetric:
        peak = one_significant_figure_ceiling(float(np.abs(finite).max()) if finite.size else 0.0)
        limits = {"vmin": -peak, "vmax": peak}
    elif finite.size:
        low, high = (float(np.percentile(finite, edge)) for edge in (2.0, 98.0))
        limits = {"vmin": low, "vmax": high}
    shown = plt.get_cmap(colormap).copy()
    shown.set_bad("#FFFFFF")
    image = axis.imshow(np.ma.masked_invalid(field), origin="lower", cmap=shown, aspect="equal",
                        extent=(0.0, 1.0, 0.0, 1.0), **limits)
    bar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
    bar.ax.tick_params(labelsize=5, length=0)
    bar.outline.set_visible(False)
    bar.set_label(label, fontsize=5.5)
    if symmetric:
        bar.set_ticks([limits["vmin"], 0.0, limits["vmax"]])
        bar.set_ticklabels([f"{limits['vmin']:.1g}", "0", f"{limits['vmax']:.1g}"])
    axis.set_xlabel("acoustic dimension 1")
    if show_y_label:
        axis.set_ylabel("acoustic dimension 2")
    if signal_share is not None:
        axis.text(0.03, 0.97, f"signal {signal_share:.0%}", transform=axis.transAxes, ha="left",
                  va="top", fontsize=5.5, color=TEXT_COLOR,
                  bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.75, "pad": 1.2})
    axis.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE)


def draw_call_density(axis, positions: np.ndarray, settings: dict) -> None:
    """
    Description
    -----------
    Where the USVs actually are, with the line at which the other manifold panels stop showing values.

    It goes FIRST of the three because it is the context the other two are read against. Without it a
    reader cannot tell that a gain landscape's red band sits in a thin part of the manifold while its
    blue body sits where the USVs pile up -- which is the difference between a model that fails and a
    model asked about USVs it has barely seen.

    The quantity is the same KDE-weighted occupancy that MASKS the other panels, so the contour drawn
    at the threshold is exactly their white boundary rather than an approximation of it.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    positions (np.ndarray)
        ``(n, 2)`` true torus positions of the USVs the nested statistic scored.
    settings (dict)
        The run's settings.

    Returns
    -------
    None
    """

    surface_settings = settings["vocal_decoding"]["tuning_surface"]
    grid, _cell = torus_grid(ERROR_DISPLAY_POINTS, settings["vocalization_settings"]["torus_period"])
    occupancy = torus_occupancy(np.asarray(positions), grid, surface_settings["kde_bandwidth"], settings["vocalization_settings"]["torus_period"])
    field = grid_image(occupancy, ERROR_DISPLAY_POINTS)
    # A short colour-bar label: the long one ran vertically almost into the next panel, and the title
    # plus this docstring already say the weighting is the tuning surface's own KDE occupancy.
    draw_manifold_field(axis, field, SEQUENTIAL_CMAP, "USVs", "USV density", symmetric=False)
    axis.contour(grid_centres(ERROR_DISPLAY_POINTS), grid_centres(ERROR_DISPLAY_POINTS), field,
                 levels=[surface_settings["decode_grid_min_occupancy"]], colors=["#FFFFFF"],
                 linewidths=0.7)


def draw_gain_map(axis, positions: np.ndarray, values: np.ndarray, title: str, settings: dict) -> None:
    """
    Description
    -----------
    Where on the acoustic manifold a model's per-USV gain lands, smoothed, in the tested currency.

    The maps used to be drawn in flat torus DISTANCE while the claim was scored in LIKELIHOOD, and on
    cl0401 the two disagreed in sign about whether behaviour helped at all -- the von Mises score fits a
    concentration and saturates, so nailing some USVs outweighs missing others, and mean distance does
    the reverse. The posterior form settles it: its statistic is already a per-USV log-density gain, so
    the map is the DECOMPOSITION of the tested number over the manifold rather than a companion to it.
    The field's mean over the scored USVs is the ladder's step; the colour says where it comes from.

    Two things the panel is NOT. It is not a map of the tuning surface -- a region where the neuron
    adds nothing can be one where it fires a lot, if behaviour already predicted those USVs -- and it
    is not an estimate of significance anywhere: the signal share in the corner says how much of its
    spatial variance shuffling the per-USV values fails to reproduce, which is a test of spatial
    autocorrelation read through the panel's own kernel, nothing more.

    Parameters
    ----------
    axis (matplotlib.axes.Axes)
        Target axis.
    positions (np.ndarray)
        ``(n, 2)`` true torus positions of the USVs.
    values (np.ndarray)
        ``(n,)`` the per-USV gain in nats; non-finite entries (USVs in masked cells) are left out.
    title (str)
        Panel title.
    settings (dict)
        The run's settings.

    Returns
    -------
    None
    """

    finite = np.isfinite(values)
    field, share = smoothed_manifold_field(np.asarray(positions)[finite], np.asarray(values)[finite],
                                           settings)
    draw_manifold_field(axis, field, GAIN_DIVERGING_CMAP, "worse  —  Δ log-likelihood (nats / USV)  —  "
                        "better", title, symmetric=True, signal_share=share, show_y_label=False)


def draw_nested_row(figure, cell, unit_uid: str, paths: FigurePaths, identity: dict, nested: dict, settings: dict,
                    target: str = "manifold") -> None:
    """
    Description
    -----------
    The behaviour-comparison row: what the behaviour model knows about a USV's position, and what the
    neuron adds to it.

    Its own row rather than two more panels beside the vocal ones, because it asks a different question of
    a different model -- the vocal row asks what the NEURON alone carries, this one what it carries that a
    behaviour model does not. Five panels: the ladder, the tested added score against its null, the USV
    density, and the two gain landscapes -- the behaviour model's over no behaviour (the ladder's first
    step over the manifold) and the neuron's over the behaviour model (its second, the tested one).

    Only a POSTERIOR section reaches this function (see ``nested_section``); the per-USV arrays the
    maps and the ladder's dots need are rebuilt by :func:`nested_event_data` and checked against the
    section before anything is drawn.

    Parameters
    ----------
    figure (matplotlib.figure.Figure)
        The figure the row is drawn into.
    cell (matplotlib.gridspec.SubplotSpec)
        The row's slot.
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    nested (dict)
        The pickle's ``nested_vocal_decoding_<target>`` section.
    settings (dict)
        The run's settings, for the manifold panels' shared grid.
    target (str)
        ``"manifold"`` or ``"multinomial"``. The categorical row has the same five panels: its ladder
        and score are in the same currency (nats per USV), and its gain landscapes are the per-USV
        category gains drawn at each USV's torus position, so the two rows can be read against each other.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    None
    """

    rebuilt = (nested_event_data(unit_uid, paths, identity, settings) if target == "manifold"
               else category_nested_event_data(unit_uid, paths, identity, settings))
    check_nested_rebuild(unit_uid, nested, rebuilt)
    widths, gap = gridspec_widths(REFERENCE_FIGURE_WIDTH * subplot_span(), [1.0, 0.7, 1.0, 1.0, 1.0], 0.5)
    columns = row_cells(cell, widths, [gap] * 4, figure.get_figwidth() * subplot_span())
    draw_nested_ladder(figure.add_subplot(columns[0]), nested, rebuilt,
                       "Predicting vocal content" if target == "manifold" else "Predicting vocal category")

    null = np.asarray(nested["null"])
    axis = figure.add_subplot(columns[1])
    per_fold_added = [fold["added"] for fold in nested["per_fold"]]
    draw_score(axis, per_fold_added, nested["added"], null,
               f"{format_p(nested['p'], nested['at_floor'], null.size)}", f"{nested['n_events']:,} USVs",
               per_fold_added, "per fold",
               "Vocal content model" if target == "manifold" else "Vocal category model", QUIET_COLOR)
    # The same label as the ladder beside it: this IS the ladder's right-hand step, in the same units,
    # so calling it something else read as a different measure.
    axis.set_ylabel("Δ log-likelihood (nats / USV)")

    draw_call_density(figure.add_subplot(columns[2]), rebuilt["positions"], settings)
    suffix = "" if target == "manifold" else " (category)"
    draw_gain_map(figure.add_subplot(columns[3]), rebuilt["positions"], rebuilt["behaviour_gain"],
                  f"Behavior model gain landscape{suffix}", settings)
    draw_gain_map(figure.add_subplot(columns[4]), rebuilt["positions"],
                  np.asarray(nested["per_event_gain"], dtype=np.float64),
                  f"Neuron-added gain landscape{suffix}", settings)


def draw_vocal_row(figure, cell, unit_uid: str, paths: FigurePaths, identity: dict, vocal: dict, settings: dict) -> None:
    """
    Description
    -----------
    The vocal row: does the unit predict that a call is coming, and which call it will be.

    Five panels, left to right -- the occurrence relationship and its score against the shift null, then
    the identity tuning map, its per-call spread, and its score against the permutation null. The two
    score panels keep the grammar of the kinematic row's: observed beside its null, with the per-fold
    spread as an inset.

    Both axes are measured on the same window, which runs from 50 ms before a call's start to 50 ms
    after, so roughly half of it lies INSIDE the call. Neither is purely predictive and neither should be
    described as such.

    Parameters
    ----------
    figure (matplotlib.figure.Figure)
        The figure the row is drawn into.
    cell (matplotlib.gridspec.SubplotSpec)
        The row's slot.
    unit_uid (str)
        Unit file stem.
    identity (dict)
        The pickle's ``identity`` block.
    vocal (dict)
        ``occurrence`` and ``identity`` sections.
    settings (dict)
        The run's settings.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    None
    """

    occurrence, identity_section = vocal["occurrence"], vocal["identity_manifold"]
    events = vocal_event_data(unit_uid, paths, identity, settings)
    widths, gap = gridspec_widths(REFERENCE_FIGURE_WIDTH * subplot_span(), [1.0, 0.7, 1.0, 1.0, 0.7], 0.6)
    columns = row_cells(cell, widths, [gap] * 4, figure.get_figwidth() * subplot_span())

    draw_occurrence_tuning(figure.add_subplot(columns[0]), occurrence, events["occurrence_windows"],
                           list(events["session_ids"]),
                           OCCURRENCE_LEGEND_AT[unit_uid] if unit_uid in OCCURRENCE_LEGEND_AT
                           else "upper left")

    occurrence_null = np.asarray(occurrence["null"])
    occurrence_axis = figure.add_subplot(columns[1])
    # Occurrence is a two-class decoder like identity: the bar is the pooled class-balanced gain (at most
    # log 2 under the equal prior), the dots the per-session gains, and the inset the amplitude each fold
    # tuned on its training sessions.
    draw_score(occurrence_axis, [fold["gain"] for fold in occurrence["per_fold"]],
               occurrence["gain"], occurrence_null,
               f"{format_p(occurrence['p'], occurrence['at_floor'], occurrence_null.size)}\n"
               f"AUROC {occurrence['auroc']:.2f}", f"amplitude {np.mean(occurrence['alphas']):.2f}",
               [float(alpha) for alpha in occurrence["alphas"]], "per fold", "Vocalizing model",
               VOCAL_COLOR)
    occurrence_axis.set_ylabel("gain (nats per window)")

    draw_tuning_map(figure.add_subplot(columns[2]), identity_section, events, settings)
    draw_per_event_gain(figure.add_subplot(columns[3]), identity_section)

    identity_null = np.asarray(identity_section["null"])
    identity_axis = figure.add_subplot(columns[4])
    draw_score(identity_axis, [fold["gain"] for fold in identity_section["per_fold"]],
               identity_section["gain"], identity_null,
               f"{format_p(identity_section['p'], identity_section['at_floor'], identity_null.size)}\n"
               f"{identity_section['nats_per_spike']:.3f} nats/spike",
               # The amplitude the decoder tuned on the training sessions, NOT a calibration slope -- the
               # identity axis has none. Its spread says whether the folds agreed on how much to trust
               # the map; a unit railed at the grid floor is one the decoder declined to use.
               f"amplitude {np.mean(identity_section['alphas']):.2f}",
               [float(alpha) for alpha in identity_section["alphas"]], "per fold",
               "Decoding performance", QUIET_COLOR)
    identity_axis.set_ylabel("gain (nats per USV)")


def draw_category_identity_row(figure, cell, unit_uid: str, identity_section: dict, settings: dict,
                               events: dict) -> None:
    """
    Description
    -----------
    Vocalization identity on the acoustic category: the categorical counterpart of the torus identity
    panels of the vocal row, drawn directly beneath them.

    Three panels in the slots the torus identity panels occupy above -- the firing rate per category in
    place of the torus tuning map, the per-USV gain, and the score against the permutation null -- so each
    panel sits under its torus counterpart. Occurrence does not depend on the target and is drawn once,
    in the row above; its two slots here carry the USVs on the torus coloured by category, with the
    category borders, and the held-out category confusion matrix.

    Parameters
    ----------
    figure (matplotlib.figure.Figure)
        The figure the row is drawn into.
    cell (matplotlib.gridspec.SubplotSpec)
        The row's slot.
    unit_uid (str)
        Unit file stem, for the rebuild check's messages.
    identity_section (dict)
        The pickle's ``vocalization_identity_multinomial`` section.
    settings (dict)
        The run's settings.
    events (dict)
        The assembled vocal events (``vocal_event_data``): positions, category labels, session names.

    Returns
    -------
    None
    """

    widths, gap = gridspec_widths(REFERENCE_FIGURE_WIDTH * subplot_span(), [1.0, 0.7, 1.0, 1.0, 0.7], 0.6)
    columns = row_cells(cell, widths, [gap] * 4, figure.get_figwidth() * subplot_span())
    draw_category_scatter(figure.add_subplot(columns[0]), events["positions"], events["category_labels"],
                          list(identity_section["categories"]), settings)
    draw_category_confusion(figure.add_subplot(columns[1]),
                            category_confusion(unit_uid, events, identity_section, settings))
    draw_category_rates(figure.add_subplot(columns[2]), identity_section, settings, list(events["session_ids"]))
    draw_per_event_gain(figure.add_subplot(columns[3]), identity_section, "Decoding vocal category")
    identity_null = np.asarray(identity_section["null"])
    identity_axis = figure.add_subplot(columns[4])
    draw_score(identity_axis, [fold["gain"] for fold in identity_section["per_fold"]],
               identity_section["gain"], identity_null,
               f"{format_p(identity_section['p'], identity_section['at_floor'], identity_null.size)}\n"
               f"{identity_section['nats_per_spike']:.3f} nats/spike",
               f"amplitude {np.mean(identity_section['alphas']):.2f}",
               [float(alpha) for alpha in identity_section["alphas"]], "per fold",
               "Decoding performance (category)", QUIET_COLOR)
    identity_axis.set_ylabel("gain (nats per USV)")


def model_figure(unit_uid: str, paths: FigurePaths, region: str, identity: dict, encoding: dict, settings: dict,
                 gating: dict | None = None, vocal: dict | None = None) -> plt.Figure:
    """
    Description
    -----------
    The figure for a unit whose representative fold found a model.

    Parameters
    ----------
    unit_uid (str)
        Unit file stem.
    region (str)
        Brain area, for the title.
    identity (dict)
        The pickle's ``identity`` block.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.
    gating (dict)
        The unit's ``vocal_gating`` section; its panels replace the model's tuning only for a unit that
        selected no model, which has no tuning to show.
    vocal (dict)
        ``occurrence``, ``identity_manifold`` and ``identity_multinomial`` sections, plus
        ``nested_manifold`` / ``nested_multinomial`` where drawable. Rows, in order: occurrence and torus
        identity; category identity; torus nested; category nested.
    paths (FigurePaths)
        Where the results, the session data and the figure cache live.

    Returns
    -------
    figure (matplotlib.figure.Figure)
        The figure.
    """

    # A unit whose kinematic encoding fails the held-out test has no tuning to show, whatever one fold's
    # screen let through: its panels describe what was screened, and the gating panels take the place of
    # the tuning. Every unit has a gating section; a unit whose encoding holds keeps its tuning panels.
    untuned = not kinematic_encoding_holds(encoding, settings)
    show_gating = untuned and gating is not None and bool(gating["ran"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = tuning_data(unit_uid, paths, identity, encoding, settings)
    features = list(encoding["selected"])
    n_features = len(features)
    n_lags = int(encoding["n_lags"])
    fps = float(encoding["fps"])
    seconds_before = -(n_lags - 1 - np.arange(n_lags)) / fps
    filters = np.asarray(encoding["filter"]).reshape(n_features, n_lags)
    band_sd = np.asarray(encoding["filter_band"]["sd"])
    representative = int(encoding["representative_fold"])

    path = encoding["forward_path_per_fold"][representative]
    fold_scores = selection_fold_scores(path)
    for row in path:
        rebuilt = fold_scores.get(int(row["step"]), [])
        print(f"  {unit_uid} step {row['step']} {row['candidate']}: stored mean {row['mean']:+.5f} | rebuilt "
              f"{np.mean(rebuilt) if rebuilt else float('nan'):+.5f} over {len(rebuilt)} inner folds")

    # Every panel keeps the dimensions it has in the two-feature figure. The rows that say nothing about
    # how many features were selected -- the top row and the vocal rows -- are laid out at the reference
    # width and left-aligned, and a filter or tuning panel is the same size whether its unit kept one
    # feature or four: the figure grows to fit the extra panels instead of stretching the other rows.
    show_vocal = vocal is not None
    show_category = show_vocal and "identity_multinomial" in vocal
    nested = vocal["nested_manifold"] if show_vocal and "nested_manifold" in vocal else None
    nested_category = vocal["nested_multinomial"] if show_vocal and "nested_multinomial" in vocal else None
    show_tuning = not untuned and not show_gating
    span = subplot_span()
    reference = REFERENCE_FIGURE_WIDTH * span
    # Row 2 (user-ruled 2026-10-20): with one or two features the filters, the model's tuning and each
    # feature's tuning run back to back in one row, with no empty slots; with three or more the filters get
    # row 2 to themselves and the tunings move to a row below, so no row runs long. Panel sizes are the
    # two-feature reference ones throughout.
    sizes = bottom_row_layout(reference, 2, n_features, False)
    filter_width, filter_gap = sizes["filters"][0], sizes["filter_gap"]
    model_width, tuning_width, tuning_gap = sizes["blocks"][1], sizes["tunings"][0], sizes["tuning_gap"]
    block_gap = sizes["gaps"][0]
    split_rows = show_tuning and n_features >= 3
    if show_gating:
        layout = bottom_row_layout(reference, n_features, n_features, True)
        kinematic_rows = [(layout["blocks"], layout["gaps"])]
    elif not show_tuning:
        kinematic_rows = [([filter_width] * n_features, [filter_gap] * (n_features - 1))]
    elif split_rows:
        kinematic_rows = [([filter_width] * n_features, [filter_gap] * (n_features - 1)),
                          ([model_width] + [tuning_width] * n_features, [block_gap] + [tuning_gap] * (n_features - 1))]
    else:
        kinematic_rows = [([filter_width] * n_features + [model_width] + [tuning_width] * n_features,
                           [filter_gap] * (n_features - 1) + [block_gap, block_gap] + [tuning_gap] * (n_features - 1))]
    total = max([reference] + [sum(widths) + sum(gaps) for widths, gaps in kinematic_rows])
    # Category rows sit under their torus counterparts (user-ruled 2026-10-28).
    heights = ([1.0] + [0.75] * len(kinematic_rows) + ([0.95] if show_vocal else [])
               + ([0.95] if show_category else []) + ([0.95] if nested else [])
               + ([0.95] if nested_category else []))
    vocal_row = 1 + len(kinematic_rows)
    height = 5.6 + 3.1 * (len(heights) - 2)
    figure = plt.figure(figsize=(total / span, height))
    # The first row starts a fixed distance below the title. The default top (0.88 of the height) left a
    # band that grew with every row added, ~2.5 in on a six-row page.
    outer = GridSpec(len(heights), 1, figure=figure, height_ratios=heights, hspace=0.45,
                     top=1.0 - TITLE_CLEARANCE_INCHES / height)
    top_widths, top_gap = gridspec_widths(reference, [0.8, 1.6, 0.7, 0.7], 0.5)
    top = row_cells(outer[0], top_widths, [top_gap] * 3, total)

    draw_fold_selection_grid(figure.add_subplot(top[0]), encoding, untuned)
    # A unit whose folds selected nothing has no forward path, so the panel carries the whole screen for
    # the representative fold instead -- every feature, ordered by its held-out score.
    if path and not untuned:
        draw_selection(figure.add_subplot(top[1]), path, fold_scores)
    else:
        draw_screen_bars(figure.add_subplot(top[1]), encoding, representative)

    quiet_axis = figure.add_subplot(top[2])
    quiet_folds = [(k, s) for k, s in enumerate(encoding["quiet_score_per_fold"])
                   if untuned or not encoding["no_model_per_fold"][k]]
    quiet_null = np.asarray(encoding["quiet_null"])
    quiet_text = (f"{format_p(encoding['quiet_p'], encoding['quiet_at_floor'], quiet_null.size)}\n"
                  f"AUROC {encoding['quiet_metrics']['auroc']:.2f}")
    quiet_slopes = [m["calibration_slope"] for k, m in enumerate(encoding["quiet_metrics_per_fold"])
                    if m is not None and (untuned or not encoding["no_model_per_fold"][k])]
    draw_score(quiet_axis, [s for _, s in quiet_folds], encoding["quiet_score"], quiet_null, quiet_text,
               f"slope {encoding['quiet_slope']:+.2f}", quiet_slopes, "per-fold", "quiet", QUIET_COLOR)
    quiet_axis.set_ylabel("calibrated D²")

    transfers = per_fold_transfer(paths, identity, settings)
    transfer_null = np.asarray(encoding["transfer_null"])
    vocal_axis = figure.add_subplot(top[3])
    vocal_slopes = [entry["slope"] for entry in encoding["transfer_leave_one_session_out"].values()]
    vocal_aurocs = [entry["auroc"] for entry in encoding["transfer_per_session"]]
    vocal_weights = [entry["n_frames"] for entry in encoding["transfer_per_session"]]
    vocal_text = (f"{format_p(encoding['transfer_p'], encoding['transfer_at_floor'], transfer_null.size)}\n"
                  f"AUROC {np.average(vocal_aurocs, weights=vocal_weights):.2f} "
                  f"[{min(vocal_aurocs):.2f}-{max(vocal_aurocs):.2f}]")
    # Drop-one scores, not per-fold: the bar is the POOLED transfer, and each of these is that same
    # pooled quantity with one session left out. Per-fold values are a different statistic.
    vocal_drops = [entry["score"] for entry in encoding["transfer_leave_one_session_out"].values()]
    draw_score(vocal_axis, vocal_drops, encoding["transfer_score"], transfer_null, vocal_text,
               f"slope {encoding['transfer_slope']:+.2f}", vocal_slopes, "LOSO", "vocal", VOCAL_COLOR)
    vocal_axis.set_ylabel("Poisson D² (100 ms bins)" if "transfer_grain" in encoding and encoding["transfer_grain"] == "count"
                          else "calibrated D²")
    rebuilt_representative = next(s for k, s, _ in transfers if k == representative)
    print(f"  {unit_uid} transfer: stored {encoding['transfer_score']:+.5f} | rebuilt representative "
          f"{rebuilt_representative:+.5f} | per fold {[round(s, 4) for _, s, _ in transfers]}")

    # Bottom row, left to right: the filters side by side, the whole model's tuning in the middle, then each
    # feature's own tuning -- so the model panel sits between what the filters are and how each one behaves.
    # With no selected model there is no tuning to show: the curves would describe a feature the screen
    # rejected, which is the one reading the panel must not invite. The filters stay; the space the tuning
    # panels would have taken is left for the gating figures.
    cells = [row_cells(outer[1 + index], widths, gaps, total) for index, (widths, gaps) in enumerate(kinematic_rows)]
    model_axis = None
    if show_gating:
        bottom = cells[0]
        filter_row = row_cells(bottom[0], layout["filters"], [layout["filter_gap"]] * (len(layout["filters"]) - 1),
                               layout["blocks"][0])
        tuning_row = filter_row
    elif not show_tuning:
        filter_row = cells[0]
        tuning_row = filter_row
    elif split_rows:
        filter_row = cells[0]
        model_axis = figure.add_subplot(cells[1][0])
        tuning_row = cells[1][1:]
    else:
        filter_row = cells[0][:n_features]
        model_axis = figure.add_subplot(cells[0][n_features])
        tuning_row = cells[0][n_features + 1:]
    if show_gating:
        proximity = gating_proximity_data(unit_uid, paths, identity, settings)
        draw_gating_verdict(figure.add_subplot(bottom[1]), gating)
        draw_gate(figure.add_subplot(bottom[2]), proximity, GATING_RULER)
        draw_gate_by_category(figure.add_subplot(bottom[3]), proximity, GATING_RULER)
        draw_gate_heatmap(figure.add_subplot(bottom[4]), proximity, GATING_RULER)
    # Both curves are referenced to their own regime's mean rate, which is exactly what hides how much
    # harder the unit fires during calls. That level is stored per session as the ratio of the vocal-frame
    # rate to the quiet rate the model was fitted on, and it rides in the legend rather than in a verdict:
    # a unit can fire far above its quiet rate while ordering frames no better than chance.
    ratios = [entry["level"]["rate_ratio"] for entry in encoding["transfer_per_session"]]
    weights = [entry["n_frames"] for entry in encoding["transfer_per_session"]]
    pooled_ratio = float(np.average(ratios, weights=weights))
    if show_tuning:
        draw_tuning(model_axis, data["quiet"]["eta"], data["vocal"]["eta"], data, rng, "model prediction (z)",
                    True, f"vocal {pooled_ratio:.1f}x quiet rate [{min(ratios):.1f}-{max(ratios):.1f}]")
        model_axis.set_ylabel("spike rate / session mean")
        model_axis.set_title("model tuning", loc="left", fontsize=PANEL_TITLE_SIZE)

    corner = freest_corner([(filters[index], filters[index] - band_sd[index], filters[index] + band_sd[index])
                            for index in range(n_features)], 0.36, 0.30)
    overrides = INSET_CORNERS[unit_uid] if unit_uid in INSET_CORNERS else {}
    for column, feature in enumerate(features):
        color = feature_color(feature)
        filter_axis = figure.add_subplot(filter_row[column])
        filter_axis.fill_between(seconds_before, filters[column] - band_sd[column], filters[column] + band_sd[column],
                                 color=color, alpha=0.35, linewidth=0)
        filter_axis.plot(seconds_before, filters[column], color=color, linewidth=1.2)
        filter_axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.6)
        # The spike-triggered average on its own axis: the model-free readout the fitted filter is a
        # deconvolution of, so a filter that merely tracks it is not an artifact of the penalty. Units
        # differ -- SD of the feature against log-odds per SD -- hence the second scale.
        sta_axis = filter_axis.inset_axes(named_corner(overrides[feature], 0.36, 0.30)
                                          if feature in overrides else corner)
        sta_mean, sta_sd = data["sta"]["difference"][column], data["sta"]["sd"][column]
        sta_axis.fill_between(seconds_before, sta_mean - sta_sd, sta_mean + sta_sd, color=STA_COLOR, alpha=0.5,
                              linewidth=0)
        sta_axis.plot(seconds_before, sta_mean, color=TEXT_COLOR, linewidth=0.8)
        sta_axis.set_xticks([-4, -2, 0])
        sta_axis.tick_params(axis="both", labelsize=3.5, length=1.5, pad=0.8)
        sta_axis.set_xlabel("time before spike (s)", fontsize=3.5, labelpad=0.8)
        sta_axis.set_ylabel("spike-triggered\naverage (z)", fontsize=3.5, labelpad=0.8)
        # The inset's y-axis always sits on the RIGHT, wherever the inset lands, so the two panels never
        # present the same quantity on different sides.
        sta_axis.yaxis.tick_right()
        sta_axis.yaxis.set_label_position("right")
        filter_axis.set_title(feature_label(feature), loc="left", fontsize=PANEL_TITLE_SIZE)
        filter_axis.set_xlabel("time before spike (s)")
        if column == 0:
            filter_axis.set_ylabel("quiet filter weight\n(log-odds per SD)")

        if not show_tuning:
            continue
        tuning_axis = figure.add_subplot(tuning_row[column])
        draw_tuning(tuning_axis, data["quiet"]["feature_output"][column], data["vocal"]["feature_output"][column],
                    data, rng, "filter output (z)", False)
        tuning_axis.set_title(feature_label(feature), loc="left", fontsize=PANEL_TITLE_SIZE)
        if column == 0:
            tuning_axis.set_ylabel("spike rate / session mean")

    next_row = vocal_row
    if show_vocal:
        draw_vocal_row(figure, outer[next_row], unit_uid, paths, identity, vocal, settings)
        next_row += 1
    if show_category:
        draw_category_identity_row(figure, outer[next_row], unit_uid, vocal["identity_multinomial"], settings,
                                   vocal_event_data(unit_uid, paths, identity, settings))
        next_row += 1
    if nested:
        draw_nested_row(figure, outer[next_row], unit_uid, paths, identity, nested, settings, "manifold")
        next_row += 1
    if nested_category:
        draw_nested_row(figure, outer[next_row], unit_uid, paths, identity, nested_category, settings, "multinomial")

    banner = (f"  |  NO KINEMATIC FEATURE HOLDS ON HELD-OUT DATA (quiet p {encoding['quiet_p']:.2f}): panels "
              f"describe screened features, not a validated model") if untuned else ""
    figure.suptitle(f"kinematic encoding | {region} | {identity['mouse_id']} {identity['rec_date']} "
                    f"{identity['unit_id']}{banner}", x=0.02, y=1.0 - 0.15 / height, ha="left", fontsize=10)
    return figure


def draw_screen(score_axis, p_axis, encoding: dict, settings: dict, identity: dict) -> None:
    """
    Description
    -----------
    The per-feature screen, every feature and every fold: held-out score against zero, p against the
    Bonferroni bar.

    A feature must clear the bar with a POSITIVE calibration slope; a negative slope means the filter
    learned on the training sessions points the wrong way on the held-out one. Those draws are hollow, so a
    point past the bar that still failed reads as what it is.

    Parameters
    ----------
    score_axis (matplotlib.axes.Axes)
        Axis for the single-feature scores.
    p_axis (matplotlib.axes.Axes)
        Axis for the p-values.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.
    identity (dict)
        The pickle's ``identity`` block.

    Returns
    -------
    None
    """

    screens = encoding["screen_per_fold"]
    names = [row["name"] for row in screens[0]]
    scores = np.array([[row["score"] for row in screen] for screen in screens])
    p_values = np.array([[row["p"] for row in screen] for screen in screens])
    slopes = np.array([[row["slope"] for row in screen] for screen in screens])
    order = np.argsort(-np.nanmean(scores, axis=0))
    bar = settings["kinematic_encoding"]["feature_selection"]["screen_alpha"] / len(names)
    session_ids = identity["courtship_sessions"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for fold, marker in zip(range(scores.shape[0]), markers, strict=False):
        faces = [feature_color(names[i]) if slopes[fold, i] > 0 else "none" for i in order]
        edges = [TEXT_COLOR if slopes[fold, i] > 0 else feature_color(names[i]) for i in order]
        score_axis.scatter(scores[fold, order], np.arange(len(names)), marker=marker, s=14, facecolors=faces,
                           edgecolors=edges, linewidth=0.7, label=f"held out {session_ids[fold]}")
        p_axis.scatter(-np.log10(p_values[fold, order]), np.arange(len(names)), marker=marker, s=14,
                       facecolors=faces, edgecolors=edges, linewidth=0.7)
    score_axis.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.6)
    p_axis.axvline(-np.log10(bar), color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
    p_axis.text(-np.log10(bar), -0.6, f" Bonferroni bar {bar:.1e}", fontsize=5, va="bottom", ha="right",
                color=TEXT_COLOR)
    score_axis.set_yticks(np.arange(len(names)))
    score_axis.set_yticklabels([feature_label(names[i]) for i in order], fontsize=5)
    score_axis.invert_yaxis()
    score_axis.set_xlabel("single-feature calibrated D\u00b2")
    p_axis.set_xlabel("-log10 p (shift null)")
    score_axis.legend(frameon=False, fontsize=5, loc="lower right")
    score_axis.set_title("screen (hollow = negative slope)", loc="left", fontsize=PANEL_TITLE_SIZE)


def no_model_figure(region: str, identity: dict, encoding: dict, settings: dict) -> plt.Figure:
    """
    Description
    -----------
    The figure for a unit where no fold selected anything AND no best screened feature was stored: the
    screen alone.

    Units run before the screen kept its best feature land here; everything since gets the standard
    figure, with its panels describing that rejected feature.

    Parameters
    ----------
    region (str)
        Brain area, for the title.
    identity (dict)
        The pickle's ``identity`` block.
    encoding (dict)
        The pickle's ``kinematic_encoding`` section.
    settings (dict)
        The run's settings.

    Returns
    -------
    figure (matplotlib.figure.Figure)
        The figure.
    """

    figure, (score_axis, p_axis) = plt.subplots(1, 2, figsize=(9.0, 6.0), sharey=True)
    draw_screen(score_axis, p_axis, encoding, settings, identity)
    figure.suptitle(f"kinematic encoding | {region} | {identity['mouse_id']} {identity['rec_date']} "
                    f"{identity['unit_id']}", x=0.02, ha="left", fontsize=10)
    figure.tight_layout()
    return figure


def nested_section(unit: dict, unit_uid: str, target: str) -> dict:
    """
    Description
    -----------
    The nested decoding's section when it is the POSTERIOR statistic, or nothing.

    The row draws only a section written by the posterior form -- identity's Bayes decoder with the
    cohort behaviour control entering as the prior -- which is recognisable by the ``point_regression``
    descriptor it carries beside the tested numbers. Two older shapes exist on disk and neither is
    drawn: a section without ``behaviour_control`` predates the 2026-09-22 cohort control and its added
    score is void; a section with the control but WITHOUT ``point_regression`` is the point-regression
    form that scored cl0401 at -0.012 on the new map, an instrument that cannot represent region-grain
    tuning. A unit whose section is in either shape has not been re-run yet, and the row is skipped
    rather than drawn from a superseded statistic. There is no sidecar fallback any more: the
    ``frozen_behaviour_model`` recomputations are old-map numbers and must not be reachable from here.

    Parameters
    ----------
    unit (dict)
        The unit's artifact.
    unit_uid (str)
        The unit's uid, for the message.
    target (str)
        Which target's section: ``'manifold'`` or ``'multinomial'``.

    Returns
    -------
    section (dict)
        The section to draw, or None to skip the row.
    """

    name = target_section("nested_vocal_decoding", target)
    section = unit[name] if name in unit else None
    # The categorical arm has no point-regression descriptor; its posterior form is recognised by the
    # control it carries.
    if (section is not None and "behaviour_control" in section
            and (target == "multinomial" or "point_regression" in section)):
        return section
    print(f"  the {target} nested row is SKIPPED for {unit_uid}: its section is not the posterior statistic "
          f"(no re-run under the shipped settings yet).")
    return None


def unit_brain_areas(catalog_path: str, unit_uids: list) -> dict:
    """
    Description
    -----------
    The catalog ``brain_area`` of each unit, keyed by ``unit_uid`` as the cohort builder forms it
    (``<mouse_id>_<rec_date>_<unit_id>``), for the figure titles.

    Parameters
    ----------
    catalog_path (str)
        The unit catalog (``data_roots.catalog_path``).
    unit_uids (list)
        Units to look up.

    Returns
    -------
    brain_areas (dict)
        ``unit_uid`` to its brain area.
    """

    catalog = pls.read_csv(configure_path(catalog_path), infer_schema_length=5000)
    by_uid = {f"{row['mouse_id']}_{row['rec_date']}_{row['unit_id']}": str(row["brain_area"])
              for row in catalog.iter_rows(named=True)}
    missing = [uid for uid in unit_uids if uid not in by_uid]
    if missing:
        msg = f"units not in the catalog {catalog_path}: {missing}"
        raise ValueError(msg)
    return {uid: by_uid[uid] for uid in unit_uids}


def plot_neural_modeling_units(results_directory: str, output_directory: str, data_root: str,
                               unit_uids: list | None = None, settings_path: str | None = None) -> list:
    """
    Description
    -----------
    Draw and save the result page(s) of every requested unit.

    Each unit gets ``<unit_uid>_modeling_results_p1.png`` (+ ``.svg``); a unit whose kinematic encoding
    holds on held-out data also gets its per-session trace pages (``_p2`` onward). A result file whose
    recorded settings hash differs from the settings in use is skipped and reported, because every
    panel rebuilds its context from those settings and the two would silently disagree. Rebuilt arrays
    are cached under ``<output_directory>/cache_<settings hash>``.

    Parameters
    ----------
    results_directory (str)
        Directory holding the unit result files (``<unit_uid>.pkl``) and the encoding fold artifacts.
    output_directory (str)
        Directory the figures (and the rebuild cache) are written to.
    data_root (str)
        Root of the session data tree.
    unit_uids (list | None)
        Units to draw; None draws every ``<unit_uid>.pkl`` in ``results_directory``.
    settings_path (str | None)
        Neural modeling settings file; None uses the shipped one.

    Returns
    -------
    written (list)
        Paths of the PNG pages written.
    """

    apply_plot_style()
    with pathlib.Path(settings_path if settings_path else NEURAL_SETTINGS_PATH).open() as settings_file:
        settings = json.load(settings_file)
    shipped_sha = compute_settings_sha256(settings)
    results = pathlib.Path(results_directory)
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = FigurePaths(results_directory=results, data_root=pathlib.Path(data_root),
                        cache_directory=output_directory / f"cache_{shipped_sha[:12]}")
    uids = list(unit_uids) if unit_uids else sorted(path.stem for path in results.glob("*.pkl"))
    brain_areas = unit_brain_areas(settings["data_roots"]["catalog_path"], uids)
    written = []
    for unit_uid in uids:
        region = brain_areas[unit_uid]
        artifact_path = paths.results_directory / f"{unit_uid}.pkl"
        if not artifact_path.exists():
            print(f"  SKIPPED {unit_uid}: no artifact in {paths.results_directory}")
            continue
        with artifact_path.open("rb") as unit_file:
            unit = pickle.load(unit_file)
        recorded_sha = unit["provenance"]["settings_sha256"]
        if recorded_sha != shipped_sha:
            print(f"  SKIPPED {unit_uid}: artifact settings hash {recorded_sha[:12]} is not the shipped "
                  f"file's {shipped_sha[:12]} -- not re-run under the current settings yet")
            continue
        encoding = unit["kinematic_encoding"]
        gating = unit["vocal_gating"] if "vocal_gating" in unit else None
        vocal = None
        if "vocal_occurrence" in unit and "vocalization_identity_manifold" in unit:
            vocal = {"occurrence": unit["vocal_occurrence"],
                     "identity_manifold": unit["vocalization_identity_manifold"]}
            if "vocalization_identity_multinomial" in unit:
                vocal["identity_multinomial"] = unit["vocalization_identity_multinomial"]
            for target in ("manifold", "multinomial"):
                section = nested_section(unit, unit_uid, target)
                if section is not None:
                    vocal[f"nested_{target}"] = section
        if not encoding["no_model"] or screened_only(encoding):
            figure = model_figure(unit_uid, paths, region, unit["identity"], encoding, settings, gating, vocal)
        else:
            figure = no_model_figure(region, unit["identity"], encoding, settings)
        destination = output_directory / f"{unit_uid}_modeling_results_p1.png"
        figure.savefig(destination, dpi=200, bbox_inches="tight")
        # The SVG keeps every line, marker and label as vectors; the manifold fields stay rasters inside
        # it, which is correct for a smoothed image.
        figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
        plt.close(figure)
        written.append(destination)
        print(f"wrote {destination} (+ .svg)")
        # The kinematic model against the observed firing, per session, in its own figure -- only for a unit
        # whose kinematic encoding holds on held-out data; otherwise there is no model whose traces mean anything.
        if not kinematic_encoding_holds(encoding, settings):
            print(f"  no session traces for {unit_uid}: kinematic encoding not significant on held-out data")
            continue
        pages, _metric_rows = session_traces_figure(unit_uid, paths, unit["identity"], settings)
        for page_number, page in enumerate(pages, 2):
            trace_path = output_directory / f"{unit_uid}_modeling_results_p{page_number}.png"
            # The plot style crops to the content; a page must keep its letter size, so cropping is off here.
            with plt.rc_context({"savefig.bbox": "standard"}):
                page.savefig(trace_path, dpi=200)
                page.savefig(trace_path.with_suffix(".svg"))
            plt.close(page)
            written.append(trace_path)
            print(f"wrote {trace_path} (+ .svg)")
    return written


@click.command(name="neural-modeling-figures")
@click.option("--results-directory", type=str, required=True,
              help="Directory holding the unit result files (<unit_uid>.pkl) and encoding fold artifacts.")
@click.option("--output-directory", type=str, required=True, help="Directory the figures are written to.")
@click.option("--data-root", type=str, required=True, help="Root of the session data tree.")
@click.option("--unit-uid", "unit_uids", type=str, multiple=True,
              help="Unit to draw (repeatable); default draws every result file in --results-directory.")
@click.option("--settings-path", type=str, default=None, help="Neural modeling settings; default is the shipped file.")
def neural_modeling_figures_cli(results_directory, output_directory, data_root, unit_uids, settings_path) -> None:
    """
    Description
    -----------
    Command-line entry for :func:`plot_neural_modeling_units`.

    Parameters
    ----------
    results_directory (str)
        Directory holding the unit result files.
    output_directory (str)
        Directory the figures are written to.
    data_root (str)
        Root of the session data tree.
    unit_uids (tuple)
        Units to draw; empty draws all.
    settings_path (str)
        Neural modeling settings file, or None for the shipped one.

    Returns
    -------
    None
    """

    plot_neural_modeling_units(results_directory, output_directory, data_root,
                               list(unit_uids) or None, settings_path)
