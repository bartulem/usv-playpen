"""
@author: bartulem
Unit tests for ``usv_playpen.visualizations.neural_modeling_plots``.

Coverage: the torus image orientation every map is drawn through (``grid_image``, ``grid_centres``),
the held-out category confusion rebuilt from stored per-fold rates and its refusal of a rebuild that does
not reproduce the stored per-USV gain, the confusion and category-scatter panels (on a synthetic
category map), the catalog lookup of brain areas, the settings-hash guard of the figure entry point, and
the registered command line.

Also covered, on synthetic sections shaped like the stored ones: the layout arithmetic (``subplot_span``,
``gridspec_widths``, ``row_cells``, ``bottom_row_layout``); the pure helpers (feature colour and label,
``screened_only``, ``kinematic_encoding_holds``, ``standard_error``, ``transfer_block``, ``format_p``,
``selection_fold_scores``, ``lighten``, ``relative_luminance``, ``named_corner``, ``freest_corner``,
``one_significant_figure_ceiling``, ``nested_section``); the per-frame arithmetic (``predictor_outputs``,
``block_resampled_sta``, ``crossfitted_probability``, ``masked_smooth``, ``overlap_metrics``,
``class_traces``, ``overlap_with_null``, ``best_windows``, ``tuning_curve``, ``smoothed_manifold_field``);
the per-fold transfer rebuilt from fold artifacts written to ``tmp_path`` in both grains; every
rebuild's cache read (a pre-written cache is returned without touching session data); the nested rebuild
check and its three refusals; every ``draw_*`` panel; the three page builders (``model_figure`` in its
tuned, split-row, gating and untuned layouts with every vocal row, ``no_model_figure`` and
``session_traces_figure`` across two pages); and the entry point end to end, from result files and
caches in ``tmp_path`` to PNG/SVG pages on disk. What stays uncovered is the part of each rebuild that
assembles real session data from the data tree.
"""

from __future__ import annotations

import copy
import json
import pickle

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import pytest
from click.testing import CliRunner
from matplotlib.colors import to_hex

from usv_playpen.modeling.modeling_metadata import compute_settings_sha256
from usv_playpen.neural_modeling.deviance_metrics import finite_mean
from usv_playpen.neural_modeling.main_neural_encoding_dispatcher import (
    fold_artifact_path,
)
from usv_playpen.neural_modeling.neural_artifacts import write_unit_section
from usv_playpen.neural_modeling.neural_category_decoding import (
    category_codes,
    category_context,
    category_gain,
)
from usv_playpen.neural_modeling.neural_vocal_decoding import fourier_basis, torus_grid
from usv_playpen.visualizations import neural_modeling_plots
from usv_playpen.visualizations.neural_modeling_plots import (
    DYADIC_COLOR,
    ERROR_DISPLAY_POINTS,
    GATE_BELOW_COLOR,
    GATE_SIGNIFICANT_COLOR,
    HEATMAP_SHADING_POINTS,
    INSET_MARGIN,
    INSET_RIGHT_INSET,
    INSET_TOP_INSET,
    N_GATE_BINS,
    NEURAL_SETTINGS_PATH,
    OTHER_COLOR,
    REJECTED_COLOR,
    SELF_COLOR,
    SESSION_ZOOM_COUNT,
    FigurePaths,
    best_windows,
    block_resampled_sta,
    bottom_row_layout,
    category_confusion,
    category_label_points,
    category_nested_event_data,
    check_nested_rebuild,
    class_traces,
    crossfitted_probability,
    draw_call_density,
    draw_category_confusion,
    draw_category_rates,
    draw_category_scatter,
    draw_fold_selection_grid,
    draw_gain_map,
    draw_gate,
    draw_gate_by_category,
    draw_gate_heatmap,
    draw_gating_verdict,
    draw_manifold_field,
    draw_nested_ladder,
    draw_occurrence_tuning,
    draw_per_event_gain,
    draw_rate_traces,
    draw_score,
    draw_screen_bars,
    draw_selection,
    draw_tuning,
    draw_tuning_map,
    feature_color,
    feature_label,
    format_p,
    freest_corner,
    gating_proximity_data,
    grid_centres,
    grid_image,
    gridspec_widths,
    kinematic_encoding_holds,
    lighten,
    masked_smooth,
    model_figure,
    named_corner,
    nested_event_data,
    nested_section,
    neural_modeling_figures_cli,
    no_model_figure,
    one_significant_figure_ceiling,
    overlap_metrics,
    overlap_with_null,
    per_fold_transfer,
    plot_neural_modeling_units,
    predictor_outputs,
    rate_trace_data,
    relative_luminance,
    row_cells,
    screened_only,
    selection_fold_scores,
    session_trace_data,
    session_traces_figure,
    smoothed_manifold_field,
    standard_error,
    stored_fold,
    subplot_span,
    transfer_block,
    tuning_curve,
    tuning_data,
    unit_brain_areas,
    vocal_event_data,
)

UNIT_UID = "181316_0_20250919_imec0_cl9999_ch000_good"
OVERRIDE_UID = "181316_0_20250919_imec1_cl0401_ch220_good"
SESSION_IDS = ["20250919_152924", "20250919_155842", "20250919_163351"]
SCREEN_NAMES = ["head-head", "self.speed", "other.speed", "allo_yaw-head", "nose-nose", "allo_pitch-head"]
N_LAGS = 20
FPS = 50.0
SESSION_SECONDS = 150.0
VOCAL_FRAMES = np.arange(1001, 1600, 2)


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Description
    -----------
    Close every figure a test left open, so the many-figures warning never trips warnings-as-errors.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    yield
    plt.close("all")


@pytest.fixture
def category_bundle(monkeypatch):
    """
    Description
    -----------
    Replace the QLVM category bundle with a synthetic four-quadrant map (category 1 lower left, 2 lower
    right, 3 upper left, 4 upper right), so the category scatter can be drawn without the shipped bundle.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        The pytest fixture.

    Returns
    -------
    bundle (dict)
        ``label_grid`` (40 x 40, rows along dimension 2) and ``axis`` (the cell centres).
    """

    resolution = 40
    label_grid = np.ones((resolution, resolution), dtype=np.int16)
    label_grid[:, resolution // 2:] += 1
    label_grid[resolution // 2:, :] += 2
    bundle = {"label_grid": label_grid, "axis": (np.arange(resolution) + 0.5) / resolution}
    monkeypatch.setattr(neural_modeling_plots, "load_qlvm_category_bundle", lambda: bundle)
    return bundle


def _shipped_settings() -> dict:
    """
    Description
    -----------
    A private copy of the shipped neural modeling settings.

    Parameters
    ----------
    None

    Returns
    -------
    settings (dict)
        The shipped settings, deep-copied.
    """

    with NEURAL_SETTINGS_PATH.open() as settings_file:
        return copy.deepcopy(json.load(settings_file))


def _category_unit(n_sessions: int = 3, per_session: int = 150, seed: int = 0) -> tuple[dict, dict, dict]:
    """
    Description
    -----------
    A synthetic unit whose spike count depends on the USV category, decoded by the real categorical
    decoder, in the shapes the figure reads: the assembled ``events``, the stored identity section, and
    the settings.

    Parameters
    ----------
    n_sessions (int)
        Number of sessions.
    per_session (int)
        USVs per session.
    seed (int)
        Random seed.

    Returns
    -------
    events (dict)
        ``category_labels``, ``counts``, ``session_index``, ``vocalization_start`` and ``positions``.
    section (dict)
        The identity section as the dispatcher stores it (``category_gain`` with detail, plus
        ``categories``).
    settings (dict)
        The shipped settings.
    """

    rng = np.random.default_rng(seed)
    n_events = n_sessions * per_session
    labels = rng.integers(1, 5, size=n_events).astype(np.float64)
    rates = np.array([1.0, 2.0, 4.0, 6.0])
    counts = rng.poisson(rates[(labels - 1).astype(int)]).astype(np.float64)
    session_index = np.repeat(np.arange(n_sessions), per_session)
    starts = np.tile(np.sort(rng.uniform(0.0, 1000.0, per_session)), n_sessions)
    positions = rng.uniform(0.0, 1.0, size=(n_events, 2))
    settings = _shipped_settings()
    codes, kept, values = category_codes(labels)
    folds = category_context(codes, session_index[kept], starts[kept],
                             int(settings["data_sufficiency"]["single_session_blocks"]))
    section = category_gain(counts[kept], codes, folds, settings["vocal_decoding"], with_detail=True)
    section["categories"] = values.tolist()
    events = {"category_labels": labels, "counts": counts, "session_index": session_index,
              "vocalization_start": starts, "positions": positions}
    return events, section, settings


def _identity(unit_uid: str = UNIT_UID) -> dict:
    """
    Description
    -----------
    The pickle's ``identity`` block for a synthetic unit recorded over the three sessions of ``SESSION_IDS``.

    Parameters
    ----------
    unit_uid (str)
        The unit's uid (``<mouse_id>_<rec_date>_<unit_id>``).

    Returns
    -------
    identity (dict)
        ``unit_uid``, ``mouse_id``, ``rec_date``, ``unit_id``, ``courtship_sessions`` and ``vocal_sessions``.
    """

    return {"unit_uid": unit_uid, "mouse_id": unit_uid[:8], "rec_date": int(unit_uid[9:17]),
            "unit_id": unit_uid[18:], "courtship_sessions": list(SESSION_IDS), "vocal_sessions": list(SESSION_IDS)}


def _paths(tmp_path) -> FigurePaths:
    """
    Description
    -----------
    A ``FigurePaths`` under ``tmp_path`` with its results and cache directories created; the data root is
    left absent, so any rebuild that reaches for session data fails loudly instead of reading real files.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        The pytest temporary directory.

    Returns
    -------
    paths (FigurePaths)
        ``results_directory``, ``data_root`` and ``cache_directory`` under ``tmp_path``.
    """

    paths = FigurePaths(results_directory=tmp_path / "results", data_root=tmp_path / "no_data",
                        cache_directory=tmp_path / "cache")
    paths.results_directory.mkdir(parents=True, exist_ok=True)
    paths.cache_directory.mkdir(parents=True, exist_ok=True)
    return paths


def _write_pickle(path, obj) -> None:
    """
    Description
    -----------
    Pickle an object to a path, creating its directory.

    Parameters
    ----------
    path (pathlib.Path)
        Destination file.
    obj (object)
        What to store.

    Returns
    -------
    None
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _encoding(features: list, holds: bool, seed: int = 0) -> dict:
    """
    Description
    -----------
    A synthetic ``kinematic_encoding`` section with every key the figure reads: the screen of each fold over
    ``SCREEN_NAMES`` (alternating slope signs, survivors = the selected features), a forward path on the
    representative fold that accepts every feature and rejects one more (the rejected step carries no inner
    fold scores), the quiet and transfer scores with their nulls, the fitted filter and its band.

    Parameters
    ----------
    features (list)
        The selected feature names, in selection order.
    holds (bool)
        Whether the held-out quiet test passes (``quiet_p`` 0.001 against 0.4).
    seed (int)
        Random seed.

    Returns
    -------
    encoding (dict)
        The section.
    """

    rng = np.random.default_rng(seed)
    n_features = len(features)
    screen = [[{"name": name, "score": 0.002 * (index + 1) + 0.001 * fold, "slope": 0.8 if index % 2 == 0 else -0.5,
                "p": 10.0 ** -(index + 1), "survived": name in features}
               for index, name in enumerate(SCREEN_NAMES)] for fold in range(len(SESSION_IDS))]
    path = [{"step": step, "candidate": feature, "mean": 0.05 * (step + 1),
             "improvement": 0.05 if step else float("nan"), "standard_error": 0.01,
             "decision": "ANCHOR" if step == 0 else "ACCEPT", "fold_scores": [0.04 * (step + 1), 0.06 * (step + 1)]}
            for step, feature in enumerate(features)]
    rejected_name = next(name for name in SCREEN_NAMES if name not in features)
    path.append({"step": n_features, "candidate": rejected_name, "mean": 0.05 * n_features + 0.001,
                 "improvement": 0.001, "standard_error": 0.01, "decision": "REJECT"})
    return {
        "no_model": False, "best_screened_only": False,
        "quiet_score": 0.1, "quiet_slope": 1.2, "quiet_p": 0.001 if holds else 0.4, "quiet_at_floor": holds,
        "quiet_null": rng.normal(0.0, 0.01, 200), "quiet_metrics": {"auroc": 0.8},
        "quiet_score_per_fold": [0.08, 0.12, 0.1], "no_model_per_fold": [False, True, False],
        "quiet_metrics_per_fold": [{"calibration_slope": 1.1}, None, {"calibration_slope": 1.3}],
        "transfer_score": 0.2, "transfer_slope": 0.8, "transfer_p": 0.001, "transfer_at_floor": True,
        "transfer_null": rng.normal(0.0, 0.01, 200), "transfer_grain": "count", "transfer_universe": "bout_window",
        "transfer_leave_one_session_out": {session: {"score": 0.15 + 0.02 * k, "slope": 0.7 + 0.05 * k}
                                           for k, session in enumerate(SESSION_IDS)},
        "transfer_per_session": [{"auroc": 0.65 + 0.05 * k, "n_frames": 1000 + 100 * k,
                                  "level": {"rate_ratio": 4.0 + k}} for k in range(len(SESSION_IDS))],
        "transfer_session_ids": list(SESSION_IDS),
        "n_lags": N_LAGS, "fps": FPS, "representative_fold": 2, "representative_session": SESSION_IDS[2],
        "selected": list(features), "selected_indices": list(range(n_features)),
        "filter": rng.normal(0.0, 0.01, n_features * N_LAGS), "filter_intercept": -3.0,
        "filter_band": {"sd": np.full((n_features, N_LAGS), 0.002)},
        "forward_path_per_fold": [path, path, path],
        "selected_per_fold": [list(features), list(features[:1]), list(features)],
        "screen_per_fold": screen,
        "filters_per_fold": [{"test_session": session} for session in SESSION_IDS],
    }


def _outputs(rng, n_features: int, sizes: list) -> dict:
    """
    Description
    -----------
    Synthetic :func:`predictor_outputs` for sessions of the given sizes: standard-normal feature outputs,
    a model output that is their sum minus two, and spikes drawn from its logistic.

    Parameters
    ----------
    rng (np.random.Generator)
        Random generator.
    n_features (int)
        Number of features.
    sizes (list)
        Frames per session.

    Returns
    -------
    outputs (dict)
        ``eta``, ``feature_output``, ``labels``, ``session_index``, ``block`` and ``session_ids``.
    """

    feature_output = rng.normal(size=(n_features, sum(sizes)))
    eta = feature_output.sum(axis=0) - 2.0
    labels = (rng.uniform(size=eta.size) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    session_index = np.concatenate([np.full(size, k) for k, size in enumerate(sizes)])
    block = np.concatenate([np.arange(size) // N_LAGS + 1000 * k for k, size in enumerate(sizes)])
    return {"eta": eta, "feature_output": feature_output, "labels": labels, "session_index": session_index,
            "block": block, "session_ids": SESSION_IDS[:len(sizes)]}


def _tuning_cache(n_features: int, seed: int = 0) -> dict:
    """
    Description
    -----------
    A synthetic :func:`tuning_data` result: quiet outputs on one held-out session, vocal outputs on three
    sessions, and a spike-triggered average with its band.

    Parameters
    ----------
    n_features (int)
        Number of features.
    seed (int)
        Random seed.

    Returns
    -------
    data (dict)
        ``quiet``, ``vocal`` and ``sta``.
    """

    rng = np.random.default_rng(seed)
    return {"quiet": _outputs(rng, n_features, [1500]), "vocal": _outputs(rng, n_features, [500, 500, 500]),
            "sta": {"difference": rng.normal(0.0, 0.2, (n_features, N_LAGS)),
                    "sd": np.full((n_features, N_LAGS), 0.05), "n_spikes": 100.0, "n_anchors": 1500.0}}


def _fold_artifacts(results_directory, identity: dict, seed: int = 0) -> None:
    """
    Description
    -----------
    Write the three encoding fold artifacts :func:`per_fold_transfer` reads: fold 0 a model at the count
    grain (per-bin predictor and counts), fold 1 no model at all (skipped), fold 2 the best SCREENED feature
    at the frame grain (per-frame predictor and 0/1 labels). Both fitted folds' predictors drive the data.

    Parameters
    ----------
    results_directory (pathlib.Path)
        Where the fold files go.
    identity (dict)
        The unit's ``identity`` block, which names the files.
    seed (int)
        Random seed.

    Returns
    -------
    None
    """

    rng = np.random.default_rng(seed)
    for fold_index in range(len(SESSION_IDS)):
        entries = []
        for _session in SESSION_IDS:
            eta = rng.normal(-1.0, 1.0, 400)
            if fold_index == 0:
                entries.append({"bin_eta": eta, "bin_counts": rng.poisson(np.exp(eta + 1.0)), "n_bins": eta.size,
                                "bin_width_frames": 15})
            else:
                entries.append({"eta": eta,
                                "labels": (rng.uniform(size=eta.size) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)})
        arrays = {"no_model": np.array(fold_index > 0), "vocal_everywhere": np.array(entries, dtype=object)}
        if fold_index == 2:
            arrays["best_screened"] = np.array(True)
        np.savez(fold_artifact_path(str(results_directory), identity, fold_index), **arrays)


def _gating_section() -> dict:
    """
    Description
    -----------
    A synthetic ``vocal_gating`` section of three features: one whose interaction overtakes its vocal
    response (the gated one) and two whose interaction stays below it.

    Parameters
    ----------
    None

    Returns
    -------
    gating (dict)
        ``ran`` and ``per_feature``.
    """

    return {"ran": True, "per_feature": {
        "head-head": {"interaction": 600.0, "vocal_main": 400.0, "baseline": 1000.0},
        "self.speed": {"interaction": 50.0, "vocal_main": 300.0, "baseline": 1000.0},
        "other.speed": {"interaction": -20.0, "vocal_main": 350.0, "baseline": 1000.0}}}


def _proximity(seed: int = 0) -> dict:
    """
    Description
    -----------
    A synthetic :func:`gating_proximity_data` result for a gated unit: the during-USV rate falls with partner
    distance in every category, quiet frames barely fire, and one USV has no category.

    Parameters
    ----------
    seed (int)
        Random seed.

    Returns
    -------
    data (dict)
        ``call_distance``, ``quiet_distance``, ``call_rate``, ``call_category``, ``quiet_label`` and ``fps``.
    """

    rng = np.random.default_rng(seed)
    head = rng.uniform(1.0, 30.0, 400)
    category = rng.integers(1, 5, 400).astype(np.float64)
    category[0] = np.nan
    quiet_head = rng.uniform(1.0, 30.0, 3000)
    return {"call_distance": {"head": head, "nose": 1.1 * head},
            "quiet_distance": {"head": quiet_head, "nose": 1.1 * quiet_head},
            "call_rate": 30.0 * np.exp(-head / 5.0) + rng.uniform(0.0, 0.5, head.size),
            "call_category": category, "quiet_label": (rng.uniform(size=3000) < 0.02).astype(np.float64),
            "fps": FPS}


def _nested(positions: np.ndarray, target: str, seed: int = 0) -> tuple[dict, dict]:
    """
    Description
    -----------
    A nested-decoding section in the posterior form and the rebuild that reproduces it: the same
    concentrations, one stored gain per rebuilt USV, and a stored behaviour rung equal to the rebuilt
    per-USV behaviour gains' finite mean (a few of them NaN, as for USVs in masked cells).

    Parameters
    ----------
    positions (np.ndarray)
        ``(n, 2)`` torus positions of the scored USVs.
    target (str)
        ``"manifold"`` or ``"multinomial"``; the manifold section carries ``point_regression``.
    seed (int)
        Random seed.

    Returns
    -------
    section (dict)
        The ``nested_vocal_decoding_<target>`` section.
    rebuilt (dict)
        The matching :func:`nested_event_data` result.
    """

    rng = np.random.default_rng(seed)
    n_events = positions.shape[0]
    behaviour_gain = rng.normal(0.01, 0.05, n_events)
    behaviour_gain[:5] = np.nan
    behaviour = finite_mean(behaviour_gain)
    kappas = [0.1, 0.0, 0.1] if target == "manifold" else [1.0, 1.0, 0.5]
    section = {"target": target, "behaviour_over_no_behaviour": behaviour, "added": 0.02,
               "full_over_no_behaviour": behaviour + 0.02,
               "per_fold": [{"added": 0.01}, {"added": 0.03}, {"added": 0.02}], "kappas": kappas,
               "null": rng.normal(0.0, 0.002, 200), "p": 0.005, "at_floor": True, "n_events": n_events,
               "per_event_gain": rng.normal(0.02, 0.1, n_events),
               "behaviour_control": {"features": ["head-head"], "held_out_day": "20250919"}}
    if target == "manifold":
        section["point_regression"] = {"added": -0.006}
    rebuilt = {"positions": positions, "session_index": np.repeat(np.arange(3), n_events // 3),
               "session_ids": list(SESSION_IDS), "behaviour_gain": behaviour_gain,
               "per_fold_behaviour_over": [behaviour - 0.005, behaviour, behaviour + 0.005],
               "per_fold_held_out_session": list(SESSION_IDS), "kappas": [float(k) for k in kappas]}
    return section, rebuilt


def _vocal_unit(seed: int = 0) -> tuple[dict, dict, dict, dict, dict]:
    """
    Description
    -----------
    Every vocal section of a synthetic unit, consistent with one set of assembled events: the occurrence
    and torus identity sections, the category identity section decoded by the real categorical decoder,
    and both nested sections with the rebuilds that reproduce them.

    Parameters
    ----------
    seed (int)
        Random seed.

    Returns
    -------
    events (dict)
        The :func:`vocal_event_data` result, with ``occurrence_windows`` and ``session_ids``.
    vocal (dict)
        ``occurrence``, ``identity_manifold``, ``identity_multinomial``, ``nested_manifold`` and
        ``nested_multinomial``.
    rebuilt_manifold (dict)
        The nested manifold rebuild.
    rebuilt_multinomial (dict)
        The nested category rebuild.
    settings (dict)
        The shipped settings.
    """

    events, category_section, settings = _category_unit(seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_events = events["counts"].size
    window_sessions = np.repeat(np.arange(3), 400)
    window_labels = (rng.uniform(size=window_sessions.size) < 0.3).astype(np.float64)
    window_counts = rng.poisson(np.where(window_labels > 0.5, 4.0, 0.5)).astype(np.float64)
    events["occurrence_windows"] = {"counts": window_counts, "labels": window_labels,
                                    "session_index": window_sessions}
    events["session_ids"] = list(SESSION_IDS)
    period = settings["vocalization_settings"]["torus_period"]
    n_coefficients = fourier_basis(torus_grid(2, period)[0], settings["vocal_decoding"]["tuning_surface"]["fourier_k"],
                                   period).shape[1] + 1
    identity_manifold = {"gain": 0.03, "alphas": [0.25, 0.5, 0.5], "null": rng.normal(0.0, 0.002, 200),
                         "p": 0.005, "at_floor": True, "nats_per_spike": 0.01,
                         "per_event_gain": rng.normal(0.03, 0.1, n_events),
                         "per_fold": [{"held_out": k, "alpha": 0.5, "coefficients": rng.normal(0.0, 0.3, n_coefficients),
                                       "gain": 0.02 + 0.01 * k} for k in range(3)]}
    identity_multinomial = {**category_section, "null": rng.normal(0.0, 0.002, 200), "p": 0.005,
                            "at_floor": True}
    occurrence = {"gain": 0.3, "per_fold": [{"gain": 0.28}, {"gain": 0.31}, {"gain": 0.33}],
                  "null": rng.normal(0.0, 0.005, 200), "p": 0.005, "at_floor": True, "auroc": 0.9,
                  "alphas": [0.5, 0.5, 0.75], "fraction_vocal_windows_with_spike": 0.93}
    nested_manifold, rebuilt_manifold = _nested(events["positions"], "manifold", seed)
    nested_multinomial, rebuilt_multinomial = _nested(events["positions"], "multinomial", seed + 2)
    vocal = {"occurrence": occurrence, "identity_manifold": identity_manifold,
             "identity_multinomial": identity_multinomial, "nested_manifold": nested_manifold,
             "nested_multinomial": nested_multinomial}
    return events, vocal, rebuilt_manifold, rebuilt_multinomial, settings


def _session_entry(session_id: str, rng, features: list, seconds: float = SESSION_SECONDS) -> dict:
    """
    Description
    -----------
    One session of a synthetic :func:`session_trace_data` result: a slowly modulated spike probability
    (NaN before the first full history), spikes drawn from it, bout windows and unassigned vocalizations.

    Parameters
    ----------
    session_id (str)
        The session's name.
    rng (np.random.Generator)
        Random generator.
    features (list)
        The fold model's features.
    seconds (float)
        The session's length.

    Returns
    -------
    entry (dict)
        ``session_id``, ``features``, ``n_frames``, ``spikes``, ``probability``, ``vocal``, ``unassigned``
        and ``fps``.
    """

    n_frames = int(seconds * FPS)
    frames = np.arange(n_frames)
    probability = 1.0 / (1.0 + np.exp(-(-2.5 + 1.5 * np.sin(frames / 300.0 + rng.uniform(0.0, 6.0)))))
    spikes = (rng.uniform(size=n_frames) < probability).astype(np.float64)
    probability[:N_LAGS - 1] = np.nan
    vocal = np.zeros(n_frames, dtype=bool)
    for start in range(1000, n_frames - 500, 1500):
        vocal[start:start + 300] = True
    unassigned = np.zeros(n_frames, dtype=bool)
    unassigned[1100:1120] = True
    return {"session_id": session_id, "features": list(features), "n_frames": n_frames, "spikes": spikes,
            "probability": probability, "vocal": vocal, "unassigned": unassigned, "fps": FPS}


def _session_traces(n_sessions: int, seed: int = 0) -> dict:
    """
    Description
    -----------
    A synthetic :func:`session_trace_data` result of ``n_sessions`` sessions.

    Parameters
    ----------
    n_sessions (int)
        Number of sessions.
    seed (int)
        Random seed.

    Returns
    -------
    data (dict)
        ``fps`` and ``sessions``.
    """

    rng = np.random.default_rng(seed)
    return {"fps": FPS, "sessions": [_session_entry(f"20250919_1{index:05d}", rng, ["self.speed", "head-head"])
                                     for index in range(n_sessions)]}


def _rate_trace(seed: int = 0) -> dict:
    """
    Description
    -----------
    A synthetic :func:`rate_trace_data` result: the held-out session's quiet frames (all frames outside
    the bout windows, minus one 20 s stretch, so one inset candidate has too few quiet frames) and two
    sessions' bout-window frames, each with a spike probability and spikes drawn
    from it.

    Parameters
    ----------
    seed (int)
        Random seed.

    Returns
    -------
    data (dict)
        ``fps``, ``held_out``, ``quiet`` and ``vocal``.
    """

    rng = np.random.default_rng(seed)
    entry = _session_entry(SESSION_IDS[2], rng, ["self.speed"])
    quiet_frames = np.flatnonzero(~entry["vocal"] & np.isfinite(entry["probability"]))
    quiet_frames = quiet_frames[(quiet_frames < 5000) | (quiet_frames >= 6000)]
    vocal = []
    for session_id in SESSION_IDS[:2]:
        other = _session_entry(session_id, rng, ["self.speed"])
        frames = np.flatnonzero(other["vocal"])
        vocal.append({"session_id": session_id, "frames": frames, "spikes": other["spikes"][frames],
                      "probability": other["probability"][frames], "n_frames": other["n_frames"]})
    return {"fps": FPS, "held_out": SESSION_IDS[2],
            "quiet": {"n_frames": entry["n_frames"], "spikes": entry["spikes"], "frames": quiet_frames,
                      "probability": entry["probability"][quiet_frames]},
            "vocal": vocal}


def _write_caches(cache_directory, unit_uid: str, n_features: int, vocal_unit: tuple | None,
                  n_trace_sessions: int = 2) -> None:
    """
    Description
    -----------
    Pre-write every rebuild cache a unit's pages read, under the file names the rebuild functions use, so
    the pages are drawn without session data.

    Parameters
    ----------
    cache_directory (pathlib.Path)
        The figure cache.
    unit_uid (str)
        The unit's uid.
    n_features (int)
        Features of the unit's model, for the tuning cache.
    vocal_unit (tuple | None)
        The output of :func:`_vocal_unit`, or None for a unit without vocal rows.
    n_trace_sessions (int)
        Sessions in the session-trace cache.

    Returns
    -------
    None
    """

    _write_pickle(cache_directory / f"{unit_uid}_tuning_bout_window.pkl", _tuning_cache(n_features))
    _write_pickle(cache_directory / f"{unit_uid}_gating_proximity.pkl", _proximity())
    _write_pickle(cache_directory / f"{unit_uid}_session_traces_v2_crossfit.pkl", _session_traces(n_trace_sessions))
    if vocal_unit is not None:
        events, _vocal, rebuilt_manifold, rebuilt_multinomial, _settings = vocal_unit
        _write_pickle(cache_directory / f"{unit_uid}_vocal_events.pkl", events)
        _write_pickle(cache_directory / f"{unit_uid}_nested_events_manifold.pkl", rebuilt_manifold)
        _write_pickle(cache_directory / f"{unit_uid}_nested_events_multinomial.pkl", rebuilt_multinomial)


def _per_session(seed: int = 0) -> dict:
    """
    Description
    -----------
    A synthetic ``assemble_unit_sessions`` result over ``SESSION_IDS``: three standard-normal features,
    spikes driven by the first, quiet anchors on the even frames and the frame rate.

    Parameters
    ----------
    seed (int)
        Random seed.

    Returns
    -------
    per_session (dict)
        Session id to ``feature_time_series``, ``spike_frames``, ``n_frames``, ``quiet`` and ``fps``.
    """

    rng = np.random.default_rng(seed)
    per_session = {}
    for session_id in SESSION_IDS:
        n_frames = 3000
        series = rng.normal(size=(n_frames, 3))
        spikes = np.flatnonzero(rng.uniform(size=n_frames) < 1.0 / (1.0 + np.exp(-(series[:, 0] - 2.0))))
        per_session[session_id] = {"feature_time_series": series, "spike_frames": spikes, "n_frames": n_frames,
                                   "quiet": np.arange(N_LAGS, n_frames, 2), "fps": FPS}
    return per_session


def _install_session_data(monkeypatch, per_session: dict) -> None:
    """
    Description
    -----------
    Patch the module's session-data seams: the session assembly returns ``per_session``, every session's
    focal vocal frames are ``VOCAL_FRAMES``, and the transfer keeps them as they are.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        The pytest fixture.
    per_session (dict)
        From :func:`_per_session`.

    Returns
    -------
    None
    """

    monkeypatch.setattr(neural_modeling_plots, "assemble_unit_sessions", lambda *_args: per_session)
    monkeypatch.setattr(neural_modeling_plots, "focal_vocal_frames", lambda *_args: (VOCAL_FRAMES, None, None))
    monkeypatch.setattr(neural_modeling_plots, "transfer_frames", lambda frames, *_args: frames)


def _uneven_category_unit(sizes: list, seed: int = 0) -> tuple[dict, dict, dict]:
    """
    Description
    -----------
    :func:`_category_unit` with sessions of unequal size, so a session too small to test is skipped by the
    decoder and must be skipped by the rebuild too.

    Parameters
    ----------
    sizes (list)
        USVs per session.
    seed (int)
        Random seed.

    Returns
    -------
    events (dict)
        ``category_labels``, ``counts``, ``session_index``, ``vocalization_start`` and ``positions``.
    section (dict)
        The identity section with ``categories``.
    settings (dict)
        The shipped settings.
    """

    rng = np.random.default_rng(seed)
    n_events = sum(sizes)
    labels = rng.integers(1, 5, size=n_events).astype(np.float64)
    counts = rng.poisson(np.array([1.0, 2.0, 4.0, 6.0])[(labels - 1).astype(int)]).astype(np.float64)
    session_index = np.concatenate([np.full(size, k) for k, size in enumerate(sizes)])
    starts = np.concatenate([np.sort(rng.uniform(0.0, 1000.0, size)) for size in sizes])
    settings = _shipped_settings()
    codes, kept, values = category_codes(labels)
    folds = category_context(codes, session_index[kept], starts[kept],
                             int(settings["data_sufficiency"]["single_session_blocks"]))
    section = category_gain(counts[kept], codes, folds, settings["vocal_decoding"], with_detail=True)
    section["categories"] = values.tolist()
    events = {"category_labels": labels, "counts": counts, "session_index": session_index,
              "vocalization_start": starts, "positions": rng.uniform(size=(n_events, 2))}
    return events, section, settings


def _titles(figure) -> list:
    """
    Description
    -----------
    Every left-aligned axis title on a figure, insets excluded.

    Parameters
    ----------
    figure (matplotlib.figure.Figure)
        The figure.

    Returns
    -------
    titles (list)
        One string per axis, empty where an axis has no title.
    """

    return [axis.get_title(loc="left") for axis in figure.axes]


class TestGridImage:
    """Every torus map is drawn with acoustic dimension 1 along x and dimension 2 along y."""

    def test_dimension_one_runs_along_columns(self):
        """A field equal to each point's dimension-1 coordinate must vary along columns, not rows."""
        grid, _cell = torus_grid(8, 1.0)
        image = grid_image(grid[:, 0], 8)
        assert np.allclose(image[0], grid_centres(8))
        assert np.allclose(image[:, 3], grid_centres(8)[3])

    def test_a_point_lands_where_its_coordinates_say(self):
        """A single grid point at (0.2, 0.8) must sit in image row of y = 0.8 and column of x = 0.2."""
        grid, _cell = torus_grid(10, 1.0)
        values = np.zeros(grid.shape[0])
        values[np.argmin(np.sum((grid - [0.25, 0.85]) ** 2, axis=1))] = 1.0
        row, column = np.unravel_index(np.argmax(grid_image(values, 10)), (10, 10))
        assert grid_centres(10)[column] == pytest.approx(0.25)
        assert grid_centres(10)[row] == pytest.approx(0.85)

    def test_centres_match_the_torus_grid(self):
        """The contour coordinates must be the torus grid's own cell centres."""
        grid, _cell = torus_grid(12, 1.0)
        assert np.allclose(np.unique(grid[:, 0]), grid_centres(12))


class TestCategoryConfusion:
    """The held-out confusion matrix rebuilt from the stored per-fold rates and amplitudes."""

    def test_rebuild_reproduces_the_tested_gain(self):
        """The count-weighted diagonal must equal the stored gain the permutation test scored."""
        events, section, settings = _category_unit()
        confusion = category_confusion("synthetic", events, section, settings)
        weights = confusion["row_counts"] / confusion["row_counts"].sum()
        assert float(np.sum(np.diag(confusion["matrix"]) * weights)) == pytest.approx(section["gain"], abs=1e-9)
        assert confusion["gain"] == pytest.approx(section["gain"], abs=1e-12)
        assert confusion["matrix"].shape == (4, 4)

    def test_a_tuned_unit_points_at_the_right_category(self):
        """For a unit whose rate rises with category, the extreme categories favour themselves."""
        events, section, settings = _category_unit()
        matrix = category_confusion("synthetic", events, section, settings)["matrix"]
        assert matrix[0, 0] > matrix[0, 3]
        assert matrix[3, 3] > matrix[3, 0]

    def test_a_rebuild_that_does_not_match_the_stored_gain_is_refused(self):
        """Stored rates that are not the run's own must stop the panel rather than draw it."""
        events, section, settings = _category_unit()
        tampered = copy.deepcopy(section)
        tampered["per_fold"][0]["rates"] = [rate * 1.5 for rate in tampered["per_fold"][0]["rates"]]
        with pytest.raises(ValueError, match="does not reproduce"):
            category_confusion("synthetic", events, tampered, settings)

    def test_panel_draws_every_cell(self):
        """The confusion panel must write one value per cell and label its colour bar."""
        events, section, settings = _category_unit()
        confusion = category_confusion("synthetic", events, section, settings)
        figure, axis = plt.subplots()
        draw_category_confusion(axis, confusion)
        assert len(axis.texts) == 16
        assert any("log-likelihood ratio" in inset.get_xlabel() for inset in axis.child_axes)
        plt.close(figure)


class TestCategoryScatter:
    """The USVs on the torus coloured by category, with the category map's borders."""

    def test_draws_borders_and_one_number_per_category(self, monkeypatch):
        """With a synthetic four-quadrant category map, the panel draws the borders and numbers 1-4."""
        resolution = 40
        axis_coordinates = (np.arange(resolution) + 0.5) / resolution
        label_grid = np.ones((resolution, resolution), dtype=np.int16)
        label_grid[:, resolution // 2:] += 1
        label_grid[resolution // 2:, :] += 2
        monkeypatch.setattr(neural_modeling_plots, "load_qlvm_category_bundle",
                            lambda: {"label_grid": label_grid, "axis": axis_coordinates})
        rng = np.random.default_rng(1)
        positions = rng.uniform(0.0, 1.0, size=(2000, 2))
        cells = np.floor(positions * resolution).astype(int)
        labels = label_grid[cells[:, 1], cells[:, 0]].astype(np.float64)
        figure, axis = plt.subplots()
        draw_category_scatter(axis, positions, labels, [1.0, 2.0, 3.0, 4.0], _shipped_settings())
        assert sorted(text.get_text() for text in axis.texts) == ["1", "2", "3", "4"]
        assert len(axis.collections) > 1
        plt.close(figure)


class TestEntryPoint:
    """The catalog lookup, the settings-hash guard and the command line."""

    def test_brain_areas_come_from_the_catalog(self, tmp_path):
        """A unit's area is read from the catalog row its uid is built from; an absent unit is an error."""
        catalog = tmp_path / "unit_catalog.csv"
        catalog.write_text("mouse_id,rec_date,unit_id,brain_area\n"
                           "181316_0,20250919,imec1_cl0401_ch220_good,PAG\n"
                           "181316_0,20250923,imec1_cl0499_ch322_good,VTA\n")
        areas = unit_brain_areas(str(catalog), ["181316_0_20250923_imec1_cl0499_ch322_good"])
        assert areas == {"181316_0_20250923_imec1_cl0499_ch322_good": "VTA"}
        with pytest.raises(ValueError, match="not in the catalog"):
            unit_brain_areas(str(catalog), ["181316_0_20250919_imec0_cl9999_ch000_good"])

    def test_a_result_written_under_other_settings_is_skipped(self, tmp_path, capsys):
        """A result file whose settings hash is not the run's must not be drawn."""
        uid = "181316_0_20250919_imec1_cl0401_ch220_good"
        catalog = tmp_path / "unit_catalog.csv"
        catalog.write_text(f"mouse_id,rec_date,unit_id,brain_area\n181316_0,20250919,{uid[18:]},PAG\n")
        settings = _shipped_settings()
        settings["data_roots"]["catalog_path"] = str(catalog)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        results = tmp_path / "results"
        results.mkdir()
        with (results / f"{uid}.pkl").open("wb") as result_file:
            pickle.dump({"provenance": {"settings_sha256": "0" * 64}}, result_file)
        written = plot_neural_modeling_units(str(results), str(tmp_path / "figures"), str(tmp_path),
                                             None, str(settings_path))
        assert written == []
        assert "SKIPPED" in capsys.readouterr().out

    def test_command_line_is_registered(self):
        """The command must answer --help under its registered name."""
        result = CliRunner().invoke(neural_modeling_figures_cli, ["--help"])
        assert result.exit_code == 0
        assert "neural-modeling-figures" in result.output


class TestLayoutArithmetic:
    """Panel sizes are stated in inches, so a unit with more features grows the page instead of shrinking panels."""

    def test_subplot_span_reads_the_rc_margins(self):
        """The span must be the right margin minus the left one under the current rcParams."""
        with plt.rc_context({"figure.subplot.left": 0.1, "figure.subplot.right": 0.85}):
            assert subplot_span() == pytest.approx(0.75)

    def test_gridspec_widths_fill_the_row(self):
        """Panels plus gaps must add up to the row, and the panels must keep their ratios."""
        widths, gap = gridspec_widths(10.0, [1.0, 2.0, 1.0], 0.5)
        assert sum(widths) + 2 * gap == pytest.approx(10.0)
        assert widths[1] == pytest.approx(2.0 * widths[0])

    def test_row_cells_returns_one_slot_per_panel_with_or_without_slack(self):
        """A row narrower than its slot is padded with an empty slot; an exact row is not."""
        figure = plt.figure(figsize=(8.0, 2.0))
        outer = figure.add_gridspec(1, 1)[0]
        padded = row_cells(outer, [1.0, 2.0], [0.5], 6.0)
        exact = row_cells(outer, [1.0, 2.0], [0.5], 3.5)
        assert len(padded) == 2
        assert len(exact) == 2
        assert padded[0].get_gridspec().ncols == 4
        assert exact[0].get_gridspec().ncols == 3

    def test_bottom_row_layout_keeps_the_reference_panel_size(self):
        """A four-feature row must be wider than the two-feature one with every panel the same width."""
        two = bottom_row_layout(10.0, 2, 2, False)
        four = bottom_row_layout(10.0, 4, 4, False)
        assert two["total"] == pytest.approx(10.0)
        assert four["total"] > two["total"]
        assert four["filters"][0] == pytest.approx(two["filters"][0])
        assert len(four["tunings"]) == 4

    def test_gating_layout_has_five_blocks_and_no_tunings(self):
        """The gating variant replaces the tuning panels with four gating blocks after the filters."""
        layout = bottom_row_layout(10.0, 3, 3, True)
        assert len(layout["blocks"]) == 5
        assert layout["tunings"] == []
        assert len(layout["filters"]) == 3
        assert layout["total"] == pytest.approx(sum(layout["blocks"]) + sum(layout["gaps"]))


class TestHelpers:
    """The small helpers every panel leans on."""

    def test_feature_color_follows_the_animal(self):
        """Self, partner and dyadic features carry the selection figures' three colours."""
        assert feature_color("self.speed") == SELF_COLOR
        assert feature_color("other.speed") == OTHER_COLOR
        assert feature_color("nose-nose") == DYADIC_COLOR

    def test_feature_label_is_the_repository_label(self):
        """A feature is named for the reader, from the male's point of view."""
        assert feature_label("self.speed") == "male speed"
        assert feature_label("other.speed") == "female speed"

    def test_screened_only_needs_the_flag_set(self):
        """Only a section that says its folds fell back to the best screened feature counts as screened-only."""
        assert screened_only({"best_screened_only": True})
        assert not screened_only({"best_screened_only": False})
        assert not screened_only({})

    def test_kinematic_encoding_holds_only_below_the_level(self):
        """The held-out quiet p must be finite and below fdr_q for the encoding to hold."""
        settings = {"significance": {"fdr_q": 0.05}}
        assert kinematic_encoding_holds({"quiet_p": 0.01}, settings)
        assert not kinematic_encoding_holds({"quiet_p": 0.05}, settings)
        assert not kinematic_encoding_holds({"quiet_p": float("nan")}, settings)

    def test_standard_error_skips_non_finite_values(self):
        """The SE is over the finite values only, and undefined below two of them."""
        assert standard_error([1.0, 3.0, np.nan]) == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
        assert np.isnan(standard_error([1.0, np.nan]))

    def test_transfer_block_keeps_the_artifact_universe(self):
        """An artifact without a universe was scored on USV spans and must be rebuilt on them."""
        settings = {"kinematic_encoding": {"transfer": {"universe": "bout_window", "pad": 1}}}
        assert transfer_block({}, settings) == {"universe": "usv_span", "pad": 1}
        assert transfer_block({"transfer_universe": "bout_window"}, settings)["universe"] == "bout_window"

    def test_format_p_marks_the_floor(self):
        """A p at the floor of its draws reads as an upper bound, otherwise as a value."""
        assert format_p(1e-3, True, 1000) == "p < 1.0e-03\n(1,000 draws)"
        assert format_p(0.25, False, 200).startswith("p = 2.5e-01")

    def test_selection_fold_scores_skips_rows_without_scores(self):
        """Only steps that carry inner-fold scores are keyed."""
        path = [{"step": 0, "fold_scores": [0.1, 0.2]}, {"step": 1}]
        assert selection_fold_scores(path) == {0: [0.1, 0.2]}

    def test_lighten_blends_toward_white(self):
        """No blend keeps the colour, a full blend is white, half of black is mid grey."""
        assert lighten("#123456", 0.0) == "#123456"
        assert lighten("#123456", 1.0) == "#ffffff"
        assert lighten("#000000", 0.5) == "#808080"

    def test_relative_luminance_spans_black_to_white(self):
        """Black is 0, white is 1 and green weighs more than blue."""
        assert relative_luminance("#000000") == pytest.approx(0.0)
        assert relative_luminance("#FFFFFF") == pytest.approx(1.0)
        assert relative_luminance("#00FF00") > relative_luminance("#0000FF")

    def test_named_corner_places_each_corner(self):
        """Every named corner is the rectangle freest_corner would choose there."""
        assert named_corner("lower left", 0.3, 0.2) == [INSET_MARGIN, INSET_MARGIN, 0.3, 0.2]
        assert named_corner("upper right", 0.3, 0.2) == pytest.approx(
            [1.0 - 0.3 - INSET_RIGHT_INSET, 1.0 - 0.2 - INSET_TOP_INSET, 0.3, 0.2])

    def test_freest_corner_keeps_upper_right_unless_crowded(self):
        """A curve along the bottom leaves the default; one that fills the upper right moves the inset."""
        flat = np.r_[np.zeros(50), np.ones(1)]
        assert freest_corner([(flat, flat, flat)], 0.36, 0.3) == pytest.approx(
            named_corner("upper right", 0.36, 0.3))
        rising = np.r_[np.zeros(50), np.full(50, 0.8)]
        chosen = freest_corner([(rising, rising, rising + 0.2)], 0.36, 0.3)
        assert chosen == pytest.approx(named_corner("upper left", 0.36, 0.3))

    def test_one_significant_figure_ceiling_rounds_up(self):
        """The peak is rounded up to one significant figure; zero and NaN fall back to 0.1."""
        assert one_significant_figure_ceiling(0.0023) == pytest.approx(0.003)
        assert one_significant_figure_ceiling(0.657) == pytest.approx(0.7)
        assert one_significant_figure_ceiling(5.0) == pytest.approx(5.0)
        assert one_significant_figure_ceiling(0.0) == pytest.approx(0.1)
        assert one_significant_figure_ceiling(float("nan")) == pytest.approx(0.1)


class TestNestedSection:
    """Only a nested section in the posterior form reaches the figure."""

    def test_posterior_sections_are_returned(self):
        """A manifold section needs the control and the point-regression descriptor; a category one the control."""
        manifold = {"behaviour_control": {}, "point_regression": {}}
        multinomial = {"behaviour_control": {}}
        unit = {"nested_vocal_decoding_manifold": manifold, "nested_vocal_decoding_multinomial": multinomial}
        assert nested_section(unit, UNIT_UID, "manifold") is manifold
        assert nested_section(unit, UNIT_UID, "multinomial") is multinomial

    def test_older_shapes_are_skipped(self, capsys):
        """A point-regression-era section, a pre-control section and a missing one are all skipped."""
        unit = {"nested_vocal_decoding_manifold": {"behaviour_control": {}},
                "nested_vocal_decoding_multinomial": {"added": 0.1}}
        assert nested_section(unit, UNIT_UID, "manifold") is None
        assert nested_section(unit, UNIT_UID, "multinomial") is None
        assert nested_section({}, UNIT_UID, "manifold") is None
        assert capsys.readouterr().out.count("SKIPPED") == 3


class TestFrameArithmetic:
    """The per-frame predictor, spike-triggered average and calibration the panels are drawn from."""

    def test_predictor_outputs_sum_to_the_model(self):
        """The model output is the feature outputs plus the intercept, each a lagged filter at the frame."""
        rng = np.random.default_rng(3)
        series = rng.normal(size=(500, 3))
        session = {"feature_time_series": series, "spike_frames": np.array([30, 100, 499, 600]), "n_frames": 500}
        filters = rng.normal(size=(2, 10))
        frames = np.arange(30, 200)
        outputs = predictor_outputs({"a": session, "b": session}, {"a": frames, "b": np.array([], dtype=np.int64)},
                                    [0, 2], filters, -1.0, 10)
        assert np.allclose(outputs["eta"], outputs["feature_output"].sum(axis=0) - 1.0)
        assert outputs["feature_output"][1, 0] == pytest.approx(series[21:31, 2] @ filters[1])
        assert outputs["labels"][frames == 30][0] == 1.0
        assert outputs["labels"].sum() == 2.0
        assert np.array_equal(outputs["block"], frames // 10)
        assert outputs["session_ids"] == ["a", "b"]

    def test_block_resampled_sta_finds_the_triggering_feature(self):
        """Spikes fired when feature 0 is high give a large STA at lag zero on feature 0 and none on feature 1."""
        rng = np.random.default_rng(4)
        series = rng.normal(size=(3000, 2))
        session = {"feature_time_series": series, "n_frames": 3000, "quiet": np.arange(N_LAGS, 3000)[::-1],
                   "spike_frames": np.flatnonzero(series[:, 0] > 1.5)}
        sta = block_resampled_sta({"s": session}, ["s"], [0, 1], N_LAGS, np.random.default_rng(0))
        assert sta["difference"].shape == (2, N_LAGS)
        assert sta["difference"][0, -1] > 1.5
        assert abs(sta["difference"][1, -1]) < 0.3
        assert np.all(sta["sd"] > 0.0)
        assert sta["n_anchors"] == 3000 - N_LAGS

    def test_crossfitted_probability_follows_the_predictor(self):
        """The calibrated probability lies in (0, 1) and rises with a predictor that drives the spikes."""
        rng = np.random.default_rng(5)
        eta = rng.normal(size=4000)
        labels = (rng.uniform(size=eta.size) < 1.0 / (1.0 + np.exp(-(eta - 1.0)))).astype(np.float64)
        group = np.repeat([0, 1], 2000)
        probability = crossfitted_probability(eta, labels, group, 5, 30)
        assert np.all((probability > 0.0) & (probability < 1.0))
        assert np.corrcoef(probability, eta)[0, 1] > 0.9

    def test_masked_smooth_stays_inside_its_class(self):
        """A constant inside the mask stays that constant, and frames outside it are NaN."""
        mask = np.zeros(200, dtype=bool)
        mask[50:150] = True
        values = np.where(mask, 3.0, 100.0)
        smoothed = masked_smooth(values, mask, 5.0)
        assert np.allclose(smoothed[mask], 3.0)
        assert np.all(np.isnan(smoothed[~mask]))

    def test_overlap_metrics_of_identical_traces(self):
        """Identical traces coincide exactly; a flat prediction has no slope or correlation."""
        trace = np.linspace(0.0, 5.0, 50)
        metrics = overlap_metrics(trace, trace)
        assert metrics == pytest.approx({"bias": 0.0, "slope": 1.0, "r": 1.0, "ccc": 1.0})
        flat = overlap_metrics(np.ones(50), trace)
        assert np.isnan(flat["slope"])
        assert np.isnan(flat["r"])
        assert overlap_metrics(trace + 1.0, trace)["ccc"] < 1.0

    def test_class_traces_accepts_spikes_at_frames_or_over_the_session(self):
        """Spikes given per class frame or over the whole session give the same traces."""
        rng = np.random.default_rng(6)
        spikes = (rng.uniform(size=1000) < 0.1).astype(np.float64)
        frames = np.arange(200, 800)
        probability = np.full(frames.size, 0.1)
        mask, observed, predicted = class_traces(spikes, frames, probability, 1000, FPS, 50.0)
        _mask, observed_at_frames, _predicted = class_traces(spikes[frames], frames, probability, 1000, FPS, 50.0)
        assert mask.sum() == frames.size
        assert np.allclose(observed[mask], observed_at_frames[mask])
        assert np.allclose(predicted[mask], 0.1 * FPS)

    def test_overlap_with_null_reports_a_shift_p(self):
        """A prediction that tracks the spikes beats its circular shifts."""
        rng = np.random.default_rng(7)
        frames = np.arange(4000)
        probability = 1.0 / (1.0 + np.exp(-(-2.0 + 2.0 * np.sin(frames / 200.0))))
        spikes = (rng.uniform(size=frames.size) < probability).astype(np.float64)
        result = overlap_with_null(spikes, frames, probability, frames.size, FPS, 250.0, [500, 1300, 2100, 2900])
        assert result["null_ccc"].size == 4
        assert result["ccc"] > result["null_ccc"].max()
        assert result["p_ccc"] == pytest.approx(0.2)

    def test_best_windows_without_candidates(self):
        """A session with no fully predicted window returns only the NaN median entry."""
        windows = best_windows(np.zeros(3000), np.zeros(3000), np.zeros(3000, dtype=bool), FPS, 4)
        assert len(windows) == 1
        assert windows[0][0] is None
        assert np.isnan(windows[0][1])

    def test_best_windows_do_not_overlap(self):
        """Chosen windows are at least one window apart and are followed by the median entry."""
        rng = np.random.default_rng(8)
        observed = np.cumsum(rng.normal(size=8000))
        predicted = observed + rng.normal(0.0, 5.0, observed.size)
        windows = best_windows(observed, predicted, np.ones(observed.size, dtype=bool), FPS, 3)
        starts = [start for start, _ccc in windows[:-1]]
        width = int(20.0 * FPS)
        assert 1 <= len(starts) <= 3
        assert all(abs(a - b) >= width for a in starts for b in starts if a != b)
        assert windows[-1][0] is None
        assert len(best_windows(observed, predicted, np.ones(observed.size, dtype=bool), FPS, 50)) < 51

    def test_tuning_curve_is_relative_to_the_session_mean(self):
        """With spikes independent of the predictor every bin sits near one; a driving predictor rises."""
        rng = np.random.default_rng(9)
        predictor = rng.normal(size=6000)
        outputs = {"labels": (rng.uniform(size=6000) < 1.0 / (1.0 + np.exp(-(2.0 * predictor - 2.0)))).astype(float),
                   "session_index": np.repeat([0, 1], 3000), "block": np.arange(6000) // 20}
        edges = np.quantile(predictor, np.linspace(0.0, 1.0, 6))
        curve = tuning_curve(predictor, outputs, edges, np.random.default_rng(0))
        assert curve["gain"][-1] > 2.0
        assert curve["gain"][0] < 0.5
        assert np.all(np.diff(curve["x"]) > 0.0)
        assert np.all(curve["se"] > 0.0)


class TestPerFoldTransfer:
    """Each fold's own model rescored on the vocal frames, from the fold artifacts."""

    def test_both_grains_are_scored_and_modelless_folds_skipped(self, tmp_path):
        """The count-grain and frame-grain folds are scored, positively, and the fold with no model is skipped."""
        paths = _paths(tmp_path)
        identity = _identity()
        _fold_artifacts(paths.results_directory, identity)
        transfers = per_fold_transfer(paths, identity, _shipped_settings())
        assert [fold for fold, _score, _slope in transfers] == [0, 2]
        assert all(score > 0.0 and slope > 0.0 for _fold, score, slope in transfers)

    def test_folds_archived_into_the_result_file_give_the_same_transfers(self, tmp_path):
        """A finished unit keeps its folds in its result file; the figures read them from there unchanged,
        and a fold found in neither place is an error."""
        paths = _paths(tmp_path)
        identity = _identity()
        _fold_artifacts(paths.results_directory, identity)
        settings = _shipped_settings()
        from_files = per_fold_transfer(paths, identity, settings)
        fold_files = sorted(paths.results_directory.glob(f"{identity['unit_uid']}_fold*.npz"))
        folds = [dict(np.load(path, allow_pickle=True)) for path in fold_files]
        write_unit_section(str(paths.results_directory), identity, "kinematic_encoding_folds", {"folds": folds},
                           settings)
        for path in fold_files:
            path.unlink()
        assert per_fold_transfer(paths, identity, settings) == from_files
        with pytest.raises(FileNotFoundError, match="neither"):
            stored_fold(paths, identity, len(folds))


class TestRebuildCaches:
    """A rebuild that finds its cache returns it without touching the session data."""

    @pytest.mark.parametrize(("rebuild", "file_name", "needs_encoding"), [
        (tuning_data, "{uid}_tuning_usv_span.pkl", True),
        (rate_trace_data, "{uid}_rate_trace_v2_crossfit.pkl", True),
        (session_trace_data, "{uid}_session_traces_v2_crossfit.pkl", False),
        (gating_proximity_data, "{uid}_gating_proximity.pkl", False),
        (vocal_event_data, "{uid}_vocal_events.pkl", False),
        (nested_event_data, "{uid}_nested_events_manifold.pkl", False),
        (category_nested_event_data, "{uid}_nested_events_multinomial.pkl", False),
    ])
    def test_cache_is_returned(self, tmp_path, rebuild, file_name, needs_encoding):
        """The cached object comes back as stored; the absent data root proves nothing was rebuilt."""
        paths = _paths(tmp_path)
        sentinel = {"cached": np.arange(3)}
        _write_pickle(paths.cache_directory / file_name.format(uid=UNIT_UID), sentinel)
        arguments = (UNIT_UID, paths, _identity(), {}, _shipped_settings()) if needs_encoding else (
            UNIT_UID, paths, _identity(), _shipped_settings())
        assert np.array_equal(rebuild(*arguments)["cached"], sentinel["cached"])


class TestNestedRebuildCheck:
    """The nested row is drawn only from a rebuild that reproduces the run."""

    def test_a_matching_rebuild_passes(self, capsys):
        """Same concentrations, same event count and the same behaviour rung pass and are reported."""
        section, rebuilt = _nested(np.random.default_rng(0).uniform(size=(90, 2)), "manifold")
        check_nested_rebuild(UNIT_UID, section, rebuilt)
        assert "behaviour rung rebuilt" in capsys.readouterr().out

    def test_other_concentrations_are_refused(self):
        """A different tuned concentration means a different design."""
        section, rebuilt = _nested(np.random.default_rng(0).uniform(size=(90, 2)), "manifold")
        rebuilt["kappas"] = [0.5, 0.0, 0.1]
        with pytest.raises(ValueError, match="concentrations"):
            check_nested_rebuild(UNIT_UID, section, rebuilt)

    def test_another_event_count_is_refused(self):
        """The rebuilt USVs must line up with the stored per-USV gains."""
        section, rebuilt = _nested(np.random.default_rng(0).uniform(size=(90, 2)), "manifold")
        section["per_event_gain"] = section["per_event_gain"][:-1]
        with pytest.raises(ValueError, match="rebuilt 90 events"):
            check_nested_rebuild(UNIT_UID, section, rebuilt)

    def test_another_behaviour_rung_is_refused(self):
        """A rebuilt behaviour rung off by more than rounding is not the run's own."""
        section, rebuilt = _nested(np.random.default_rng(0).uniform(size=(90, 2)), "manifold")
        section["behaviour_over_no_behaviour"] += 1e-3
        with pytest.raises(ValueError, match="behaviour rung"):
            check_nested_rebuild(UNIT_UID, section, rebuilt)


class TestKinematicPanels:
    """The selection, screen and score panels of the top row."""

    def test_fold_selection_grid_marks_selected_and_screened(self):
        """Selected features get filled dots, screened-only ones hollow, unscreened ones nothing; the representative is bold."""
        encoding = _encoding(["head-head", "self.speed"], True)
        for row in encoding["screen_per_fold"][0]:
            row["survived"] = False
        encoding["selected_per_fold"][0] = ["head-head"]
        figure, axis = plt.subplots()
        draw_fold_selection_grid(axis, encoding, False)
        assert [label.get_text() for label in axis.get_yticklabels()] == ["inter-partner distance", "male speed"]
        assert axis.get_xticklabels()[2].get_fontweight() == "bold"
        assert axis.get_title(loc="left").startswith("selected per fold")
        assert len(axis.collections) == 5
        plt.close(figure)

    def test_fold_selection_grid_rejected_style_and_dropped_rows(self):
        """Best-screened folds are drawn grey and italic, and rows beyond the cap are counted in the title."""
        encoding = _encoding(["head-head", "self.speed"], False)
        figure, axis = plt.subplots()
        draw_fold_selection_grid(axis, encoding, True, max_rows=1)
        assert len(axis.get_yticklabels()) == 1
        assert axis.get_yticklabels()[0].get_style() == "italic"
        assert "1 rarer feature(s) not shown" in axis.get_title(loc="left")
        assert to_hex(axis.collections[0].get_facecolor()[0]) == REJECTED_COLOR.lower()
        plt.close(figure)

    def test_screen_bars_are_ordered_and_count_survivors(self):
        """Bars run from the lowest score up and the title counts the features that passed this fold's screen."""
        encoding = _encoding(["head-head", "self.speed"], False)
        figure, axis = plt.subplots()
        draw_screen_bars(axis, encoding, 0)
        widths = [patch.get_width() for patch in axis.patches]
        assert widths == sorted(widths)
        assert "2 passed" in axis.get_title(loc="left")
        for row in encoding["screen_per_fold"][1]:
            row["survived"] = False
        draw_screen_bars(axis, encoding, 1)
        assert axis.get_title(loc="left") == "screened features (none survived)"
        plt.close(figure)

    def test_selection_bars_mark_the_rejected_step(self):
        """Each accepted step is labelled with its feature, the rejected one is italic grey and marked ns."""
        encoding = _encoding(["head-head", "self.speed"], True)
        path = encoding["forward_path_per_fold"][2]
        figure, axis = plt.subplots()
        draw_selection(axis, path, selection_fold_scores(path))
        labels = axis.get_yticklabels()
        assert [label.get_text() for label in labels] == ["+inter-partner distance", "+male speed", "+female speed"]
        assert labels[-1].get_style() == "italic"
        assert any("(ns)" in text.get_text() for text in axis.texts)
        assert axis.get_title(loc="left") == "forward selection (2 inner folds)"
        draw_selection(axis, path, {}, "custom")
        assert axis.get_title(loc="left") == "custom"
        plt.close(figure)

    def test_score_bar_is_the_tested_value(self):
        """The bar is the tested statistic, non-finite fold values are dropped, and the slope inset is drawn."""
        _figure, axis = plt.subplots()
        draw_score(axis, [0.1, np.nan, 0.3], 0.25, np.random.default_rng(0).normal(0.0, 0.01, 100), "p < 1e-3",
                   "slope +0.80", [0.7, 0.9], "per fold", "quiet", "#1D3557")
        assert axis.patches[0].get_height() == pytest.approx(0.25)
        assert len(axis.collections[0].get_offsets()) == 2
        assert len(axis.child_axes) == 1
        assert axis.get_title(loc="left") == "quiet"
        _figure, axis = plt.subplots()
        draw_score(axis, [], 0.1, np.zeros(10) + 0.01, "p", "s", [1.0], "c", "t", "#1D3557")
        assert axis.get_ylim()[1] > 0.1
        plt.close("all")

    def test_tuning_panel_draws_both_regimes(self):
        """Quiet and vocal curves are drawn with their bands, and the legend names both."""
        data = _tuning_cache(1)
        figure, axis = plt.subplots()
        draw_tuning(axis, data["quiet"]["eta"], data["vocal"]["eta"], data, np.random.default_rng(0), "z", True,
                    "vocal 4x")
        assert [line.get_label() for line in axis.lines[:2]] == ["quiet (held-out session)", "vocal 4x"]
        assert len(axis.collections) == 2
        assert axis.get_legend() is not None
        plt.close(figure)


class TestGatingPanels:
    """The gating verdict and the model-free gate panels."""

    def test_verdict_sorts_by_interaction_and_flags_the_gate(self):
        """Features run by interaction share, and only the one whose product term overtakes the vocal response is red."""
        figure, axis = plt.subplots()
        draw_gating_verdict(axis, _gating_section())
        assert [label.get_text() for label in axis.get_yticklabels()] == [
            "female speed", "male speed", "inter-partner distance"]
        interaction_bars = axis.patches[3:]
        assert [to_hex(bar.get_facecolor()) for bar in interaction_bars] == [
            GATE_BELOW_COLOR.lower(), GATE_BELOW_COLOR.lower(), GATE_SIGNIFICANT_COLOR.lower()]
        plt.close(figure)

    def test_gate_curve_falls_with_distance(self):
        """The during-USV curve falls with distance, its Spearman is negative, and quiet frames sit below it."""
        figure, axis = plt.subplots()
        draw_gate(axis, _proximity(), "head")
        during, quiet = axis.lines[0].get_ydata(), axis.lines[1].get_ydata()
        assert len(during) == N_GATE_BINS
        assert during[0] > during[-1]
        assert np.all(np.asarray(quiet) < np.asarray(during)[0])
        assert "ρ -" in axis.get_title(loc="left")
        plt.close(figure)

    def test_gate_by_category_is_higher_close(self):
        """Every category fires more on close USVs than on far ones, and the unlabelled USV is left out."""
        figure, axis = plt.subplots()
        draw_gate_by_category(axis, _proximity(), "head")
        heights = [bar.get_height() for bar in axis.patches]
        assert len(heights) == 8
        assert all(close > far for close, far in zip(heights[:4], heights[4:], strict=True))
        assert [label.get_text() for label in axis.get_xticklabels()] == ["1", "2", "3", "4"]
        plt.close(figure)

    def test_gate_heatmap_shades_along_distance(self):
        """One row per category, shaded along distance, with a colour bar; an empty cell is drawn as zero."""
        data = _proximity()
        far = data["call_distance"]["nose"] > np.quantile(data["call_distance"]["nose"], 0.8)
        data["call_category"][far & (data["call_category"] == 4.0)] = 3.0
        figure, axis = plt.subplots()
        draw_gate_heatmap(axis, data, "nose")
        image = axis.get_images()[0]
        assert image.get_array().shape == (4, HEATMAP_SHADING_POINTS)
        assert image.get_array()[3, -1] == 0.0
        assert len(axis.get_xticklabels()) == N_GATE_BINS
        assert len(figure.axes) == 2
        plt.close(figure)


class TestVocalPanels:
    """The occurrence, tuning-map, category-rate and per-USV gain panels."""

    def test_occurrence_tuning_one_line_per_session(self):
        """Each session with windows gets its own line; the stored sensitivity is printed when present."""
        events, vocal, _manifold, _multinomial, _settings = _vocal_unit()
        windows = events["occurrence_windows"]
        figure, axis = plt.subplots()
        draw_occurrence_tuning(axis, vocal["occurrence"], windows, SESSION_IDS, "upper left")
        assert [text.get_text() for text in axis.get_legend().get_texts()] == ["152924", "155842", "163351"]
        assert any(text.get_text() == "fires on 93% of USVs" for text in axis.texts)
        assert axis.get_xticklabels()[-1].get_text() == "10+"
        plt.close(figure)

    def test_occurrence_tuning_computes_sensitivity_without_the_stored_value(self):
        """An older section without the stored share is read from the windows, and an empty session is skipped."""
        events, vocal, _manifold, _multinomial, _settings = _vocal_unit()
        windows = {key: value[events["occurrence_windows"]["session_index"] < 2]
                   for key, value in events["occurrence_windows"].items()}
        occurrence = {key: value for key, value in vocal["occurrence"].items()
                      if key != "fraction_vocal_windows_with_spike"}
        expected = float(np.mean(windows["counts"][windows["labels"] > 0.5] > 0))
        figure, axis = plt.subplots()
        draw_occurrence_tuning(axis, occurrence, windows, SESSION_IDS, "center right")
        assert len(axis.get_legend().get_texts()) == 2
        assert any(text.get_text() == f"fires on {100.0 * expected:.0f}% of USVs" for text in axis.texts)
        plt.close(figure)

    def test_tuning_map_blanks_unoccupied_cells(self):
        """Cells no USV comes near are masked, the rest carry the fold-averaged rate surface."""
        _events, vocal, _manifold, _multinomial, settings = _vocal_unit()
        positions = np.random.default_rng(0).uniform(0.0, 0.4, size=(300, 2))
        figure, axis = plt.subplots()
        draw_tuning_map(axis, vocal["identity_manifold"], {"positions": positions}, settings)
        surface = axis.get_images()[0].get_array()
        grid_n = settings["vocal_decoding"]["tuning_surface"]["grid_n"]
        assert surface.shape == (grid_n, grid_n)
        assert surface.mask[int(0.7 * grid_n), int(0.7 * grid_n)]
        assert not surface.mask[int(0.2 * grid_n), int(0.2 * grid_n)]
        assert figure.axes[1].get_ylabel() == "relative firing rate"
        plt.close(figure)

    def test_category_rates_are_per_second_means(self):
        """Bars are the fold-mean rates divided by the window width; each fold is named by its held-out session."""
        _events, vocal, _manifold, _multinomial, settings = _vocal_unit()
        section = vocal["identity_multinomial"]
        width = settings["vocalization_settings"]["spike_window"]["width_seconds"]
        figure, axis = plt.subplots()
        draw_category_rates(axis, section, settings, SESSION_IDS)
        expected = np.mean([fold["rates"] for fold in section["per_fold"]], axis=0) / width
        assert np.allclose([bar.get_height() for bar in axis.patches], expected)
        assert len(axis.get_legend().get_texts()) == len(section["per_fold"])
        plt.close(figure)

    def test_per_event_gain_reports_mean_and_share(self):
        """The legend carries the mean of the finite gains and the text their positive share."""
        gains = np.array([0.1, -0.1, 0.3, np.nan])
        figure, axis = plt.subplots()
        draw_per_event_gain(axis, {"per_event_gain": gains}, "gain")
        assert axis.get_legend().get_texts()[0].get_text() == "mean +0.100 nats"
        assert any(text.get_text() == "67% positive" for text in axis.texts)
        assert axis.get_title(loc="left") == "gain"
        plt.close(figure)


class TestManifoldPanels:
    """The smoothed torus fields and the nested ladder."""

    def test_signal_share_separates_structure_from_noise(self):
        """A value tied to position keeps most of its spatial variance under shuffling; random values do not."""
        settings = _shipped_settings()
        rng = np.random.default_rng(0)
        positions = rng.uniform(size=(300, 2))
        field, structured = smoothed_manifold_field(positions, np.sin(2.0 * np.pi * positions[:, 0]), settings)
        _field, random = smoothed_manifold_field(positions, rng.normal(size=300), settings)
        assert field.shape == (ERROR_DISPLAY_POINTS, ERROR_DISPLAY_POINTS)
        assert structured > 0.8
        assert random < structured

    def test_constant_values_have_no_signal(self):
        """A field with no spread has a signal share of zero."""
        positions = np.random.default_rng(1).uniform(size=(200, 2))
        _field, share = smoothed_manifold_field(positions, np.ones(200), _shipped_settings())
        assert share == 0.0

    def test_manifold_field_limits(self):
        """A symmetric field is centred on zero at its rounded peak; a sequential one uses robust percentiles."""
        field = np.linspace(-0.23, 0.1, 100).reshape(10, 10)
        _figure, axis = plt.subplots()
        draw_manifold_field(axis, field, "BrBG_r", "gain", "title", symmetric=True, signal_share=0.5)
        assert axis.get_images()[0].get_clim() == pytest.approx((-0.3, 0.3))
        assert any(text.get_text() == "signal 50%" for text in axis.texts)
        _figure, axis = plt.subplots()
        draw_manifold_field(axis, field, "inferno", "n", "title", symmetric=False, show_y_label=False)
        assert axis.get_images()[0].get_clim() == pytest.approx(
            (np.percentile(field, 2.0), np.percentile(field, 98.0)))
        assert axis.get_ylabel() == ""
        _figure, axis = plt.subplots()
        draw_manifold_field(axis, np.full((4, 4), np.nan), "inferno", "n", "empty", symmetric=False)
        assert axis.get_title(loc="left") == "empty"
        plt.close("all")

    def test_call_density_and_gain_map(self):
        """The density panel draws its occupancy contour; the gain map drops non-finite USVs and is symmetric."""
        settings = _shipped_settings()
        rng = np.random.default_rng(2)
        positions = rng.uniform(0.0, 0.5, size=(200, 2))
        figure, (density_axis, gain_axis) = plt.subplots(1, 2)
        draw_call_density(density_axis, positions, settings)
        values = rng.normal(0.0, 0.1, 200)
        values[:3] = np.nan
        draw_gain_map(gain_axis, positions, values, "gain landscape", settings)
        assert density_axis.get_title(loc="left") == "USV density"
        assert len(density_axis.collections) >= 1
        low, high = gain_axis.get_images()[0].get_clim()
        assert low == pytest.approx(-high)
        assert gain_axis.get_title(loc="left") == "gain landscape"
        plt.close(figure)

    def test_nested_ladder_has_three_rungs(self):
        """The ladder names its three rungs, prints both steps and the concentrations."""
        section, rebuilt = _nested(np.random.default_rng(0).uniform(size=(90, 2)), "manifold")
        figure, axis = plt.subplots()
        draw_nested_ladder(axis, section, rebuilt)
        assert [label.get_text() for label in axis.get_xticklabels()] == ["no behavior", "behavior", "+ neuron"]
        steps = [text.get_text() for text in axis.texts if text.get_text().startswith("Δ")]
        assert steps == [f"Δ {section['behaviour_over_no_behaviour']:+.3f}", "Δ +0.020"]
        assert any(text.get_text() == "κ 0.1 / 0 / 0.1" for text in axis.texts)
        plt.close(figure)


class TestTracePanels:
    """The predicted-against-observed rate rows and pages."""

    def test_rate_traces_draw_both_regimes(self):
        """The quiet row gets its median-concordance inset and summary; the vocal row one line per session."""
        data = _rate_trace()
        figure, (quiet_axis, vocal_axis) = plt.subplots(2, 1)
        draw_rate_traces(quiet_axis, vocal_axis, data, _shipped_settings())
        assert len(quiet_axis.child_axes) == 1
        assert any(text.get_text().startswith("quiet: CCC") for text in quiet_axis.texts)
        summary = next(text.get_text() for text in vocal_axis.texts if "CCC" in text.get_text())
        assert summary.count("\n") == 1
        n_vocal = sum(entry["frames"].size for entry in data["vocal"])
        assert vocal_axis.get_xlim()[1] == pytest.approx(n_vocal / FPS)
        plt.close(figure)

    def test_rate_traces_skip_the_inset_without_a_quiet_window(self):
        """With no window at least 70% quiet there is no typical window to show, so no inset is drawn."""
        data = _rate_trace()
        keep = np.arange(data["quiet"]["frames"].size) % 2 == 0
        data["quiet"]["frames"] = data["quiet"]["frames"][keep]
        data["quiet"]["probability"] = data["quiet"]["probability"][keep]
        figure, (quiet_axis, vocal_axis) = plt.subplots(2, 1)
        draw_rate_traces(quiet_axis, vocal_axis, data, _shipped_settings())
        assert quiet_axis.child_axes == []
        plt.close(figure)

    def test_session_traces_figure_pages_and_rows(self, tmp_path):
        """Seven sessions fill one two-column page and spill onto a second; a short session hides unused zooms."""
        paths = _paths(tmp_path)
        data = _session_traces(7)
        data["sessions"][6] = _session_entry("20250919_199999", np.random.default_rng(1), ["self.speed"], 50.0)
        _write_pickle(paths.cache_directory / f"{UNIT_UID}_session_traces_v2_crossfit.pkl", data)
        pages, rows = session_traces_figure(UNIT_UID, paths, _identity(), _shipped_settings())
        assert len(pages) == 2
        assert len(rows) == 7
        assert pages[0].get_figwidth() > pages[1].get_figwidth()
        assert len(pages[1].axes) == 1 + SESSION_ZOOM_COUNT
        assert not all(axis.get_visible() for axis in pages[1].axes)
        assert "window_D_ccc" not in rows[6]
        assert np.isfinite(rows[0]["ccc_250ms"])
        assert "window_A_start_min" in rows[0]
        assert rows[0]["features"] == "self.speed + head-head"
        assert "male speed + inter-partner distance" in pages[0].axes[0].get_title(loc="left")


class TestPages:
    """The page builders, drawn from synthetic sections and pre-written caches."""

    def test_tuned_unit_with_every_vocal_row(self, tmp_path, category_bundle, capsys):
        """A unit whose encoding holds gets its tuning panels and all four vocal rows; inset overrides apply."""
        del category_bundle
        paths = _paths(tmp_path)
        identity = _identity(OVERRIDE_UID)
        vocal_unit = _vocal_unit()
        _write_caches(paths.cache_directory, OVERRIDE_UID, 2, vocal_unit)
        _fold_artifacts(paths.results_directory, identity)
        encoding = _encoding(["head-head", "allo_yaw-head"], True)
        figure = model_figure(OVERRIDE_UID, paths, "PAG", identity, encoding, vocal_unit[4], _gating_section(),
                              vocal_unit[1])
        titles = _titles(figure)
        for expected in ("forward selection (2 inner folds)", "model tuning", "Vocalizing: P (vocal | spike count)",
                         "USV manifold tuning", "Category confusion", "USV category tuning",
                         "Predicting vocal content", "Predicting vocal category",
                         "Neuron-added gain landscape (category)"):
            assert expected in titles
        assert "vocal-feature interactions" not in titles
        assert "NO KINEMATIC FEATURE" not in figure._suptitle.get_text()
        assert "transfer: stored +0.20000" in capsys.readouterr().out
        assert figure.get_figheight() == pytest.approx(5.6 + 3.1 * 4)

    def test_three_features_split_the_kinematic_row(self, tmp_path):
        """With three features the filters get a row of their own and the tunings move below them."""
        paths = _paths(tmp_path)
        identity = _identity()
        _write_caches(paths.cache_directory, UNIT_UID, 3, None)
        _fold_artifacts(paths.results_directory, identity)
        encoding = _encoding(["head-head", "self.speed", "other.speed"], True)
        encoding["transfer_grain"] = "frame"
        figure = model_figure(UNIT_UID, paths, "VTA", identity, encoding, _shipped_settings())
        titles = _titles(figure)
        assert titles.count("model tuning") == 1
        assert titles.count("male speed") == 2
        assert figure.get_figheight() == pytest.approx(5.6 + 3.1)
        vocal_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "vocal")
        assert vocal_axis.get_ylabel() == "calibrated D²"

    def test_untuned_unit_with_gating_shows_the_gate(self, tmp_path):
        """A unit that fails the held-out test gets the screen, the gating panels and the warning banner."""
        paths = _paths(tmp_path)
        identity = _identity()
        _write_caches(paths.cache_directory, UNIT_UID, 1, None)
        _fold_artifacts(paths.results_directory, identity)
        encoding = _encoding(["head-head"], False)
        encoding.update({"no_model": True, "best_screened_only": True})
        figure = model_figure(UNIT_UID, paths, "PAG", identity, encoding, _shipped_settings(), _gating_section())
        titles = _titles(figure)
        assert "vocal-feature interactions" in titles
        assert any(title.startswith("the gate") for title in titles)
        assert "gate holds in every category" in titles
        assert any(title.startswith("screened features") for title in titles)
        assert "model tuning" not in titles
        assert "NO KINEMATIC FEATURE HOLDS" in figure._suptitle.get_text()

    def test_untuned_unit_without_gating_keeps_only_filters(self, tmp_path):
        """With no gating section an untuned unit draws its filters and nothing in place of the tunings."""
        paths = _paths(tmp_path)
        identity = _identity()
        _write_caches(paths.cache_directory, UNIT_UID, 2, None)
        _fold_artifacts(paths.results_directory, identity)
        encoding = _encoding(["head-head", "self.speed"], False)
        figure = model_figure(UNIT_UID, paths, "PAG", identity, encoding, _shipped_settings(), None)
        titles = _titles(figure)
        assert titles.count("male speed") == 1
        assert "model tuning" not in titles
        assert not any(title.startswith("the gate") for title in titles)

    def test_no_model_figure_draws_the_screen(self):
        """A unit with nothing to draw but its screen gets one marker set per fold and the Bonferroni bar."""
        encoding = _encoding(["head-head"], False)
        figure = no_model_figure("PAG", _identity(), encoding, _shipped_settings())
        score_axis, p_axis = figure.axes[:2]
        assert len(score_axis.collections) == len(SESSION_IDS)
        assert score_axis.get_yticklabels()[0].get_text() == feature_label(SCREEN_NAMES[-1])
        assert any("Bonferroni bar" in text.get_text() for text in p_axis.texts)
        faces = score_axis.collections[0].get_facecolors()
        assert sum(face[3] == 0.0 for face in faces) == 3
        assert "PAG" in figure._suptitle.get_text()


class TestEndToEnd:
    """The entry point from result files and caches to pages on disk."""

    def test_pages_are_written_for_every_kind_of_unit(self, tmp_path, category_bundle, capsys):
        """A tuned unit gets its result page and trace page, a no-model unit its screen, a missing one is skipped."""
        del category_bundle
        settings = _shipped_settings()
        missing_uid = "181316_0_20250919_imec0_cl0002_ch002_good"
        uids = [UNIT_UID, OVERRIDE_UID, missing_uid]
        catalog = tmp_path / "unit_catalog.csv"
        catalog.write_text("mouse_id,rec_date,unit_id,brain_area\n"
                           + "".join(f"181316_0,20250919,{uid[18:]},PAG\n" for uid in uids))
        settings["data_roots"]["catalog_path"] = str(catalog)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        sha = compute_settings_sha256(settings)
        output = tmp_path / "figures"
        results = tmp_path / "results"
        results.mkdir()
        vocal_unit = _vocal_unit()
        _write_caches(output / f"cache_{sha[:12]}", UNIT_UID, 1, vocal_unit, n_trace_sessions=1)
        _fold_artifacts(results, _identity(UNIT_UID))
        events, vocal, _manifold, _multinomial, _settings = vocal_unit
        del events
        tuned = {"provenance": {"settings_sha256": sha}, "identity": _identity(UNIT_UID),
                 "kinematic_encoding": _encoding(["self.speed"], True), "vocal_gating": _gating_section(),
                 "vocal_occurrence": vocal["occurrence"], "vocalization_identity_manifold": vocal["identity_manifold"],
                 "vocalization_identity_multinomial": vocal["identity_multinomial"],
                 "nested_vocal_decoding_manifold": {"behaviour_control": {}},
                 "nested_vocal_decoding_multinomial": vocal["nested_multinomial"]}
        screen_only = {"provenance": {"settings_sha256": sha}, "identity": _identity(OVERRIDE_UID),
                       "kinematic_encoding": {**_encoding(["self.speed"], False), "no_model": True}}
        _write_pickle(results / f"{UNIT_UID}.pkl", tuned)
        _write_pickle(results / f"{OVERRIDE_UID}.pkl", screen_only)
        written = plot_neural_modeling_units(str(results), str(output), str(tmp_path / "no_data"), uids,
                                             str(settings_path))
        assert [path.name for path in written] == [f"{UNIT_UID}_modeling_results_p1.png",
                                                   f"{UNIT_UID}_modeling_results_p2.png",
                                                   f"{OVERRIDE_UID}_modeling_results_p1.png"]
        assert all(path.exists() and path.with_suffix(".svg").exists() for path in written)
        out = capsys.readouterr().out
        assert f"SKIPPED {missing_uid}: no artifact" in out
        assert f"no session traces for {OVERRIDE_UID}" in out
        assert "the manifold nested row is SKIPPED" in out


class TestCategoryRebuildEdges:
    """The category rebuild's fold alignment and the number placement on the category scatter."""

    def test_a_session_too_small_to_test_is_skipped_like_the_decoder(self):
        """A session below the decoder's minimum is skipped by both, so the rebuild still reproduces the gain."""
        events, section, settings = _uneven_category_unit([150, 150, 3])
        confusion = category_confusion("synthetic", events, section, settings)
        assert len(section["per_fold"]) == 2
        assert confusion["gain"] == pytest.approx(section["gain"], abs=1e-9)

    def test_a_misaligned_fold_is_refused(self):
        """A stored fold that held out another session than the rebuilt one stops the panel."""
        events, section, settings = _category_unit()
        section["per_fold"][0]["held_out"] = 99
        with pytest.raises(ValueError, match="lines up with stored fold 99"):
            category_confusion("synthetic", events, section, settings)

    def test_a_category_that_wins_nowhere_gets_no_number(self, category_bundle):
        """A category with a single USV inside another category's dense region is never the local winner."""
        del category_bundle
        rng = np.random.default_rng(3)
        positions = rng.uniform(0.0, 0.5, size=(1200, 2)) + rng.integers(0, 2, size=(1200, 1)) * [0.5, 0.0]
        positions[:400, 1] += 0.5
        labels = np.where(positions[:, 0] < 0.5, 1.0, 2.0)
        labels[:400] = 3.0
        positions = np.vstack([positions, [[0.2, 0.2]]])
        labels = np.r_[labels, 4.0]
        settings = _shipped_settings()
        points = category_label_points(positions, labels, [1.0, 2.0, 3.0, 4.0], settings)
        assert points[3] is None
        assert all(point is not None for point in points[:3])
        figure, axis = plt.subplots()
        draw_category_scatter(axis, positions, labels, [1.0, 2.0, 3.0, 4.0], settings)
        assert sorted(text.get_text() for text in axis.texts) == ["1", "2", "3"]
        plt.close(figure)


class TestRebuildBodies:
    """The rebuilds with their session-data loaders patched: what they compute, check and cache."""

    def test_tuning_data_rebuilds_checks_and_caches(self, tmp_path, monkeypatch, capsys):
        """The rebuilt vocal output matches the stored transfer, quiet frames come from the held-out session only."""
        per_session = _per_session()
        _install_session_data(monkeypatch, per_session)
        paths = _paths(tmp_path)
        encoding = _encoding(["self.speed", "head-head"], True)
        filters = np.asarray(encoding["filter"]).reshape(2, N_LAGS)
        expected = predictor_outputs(per_session, dict.fromkeys(SESSION_IDS, VOCAL_FRAMES), [0, 1], filters,
                                     encoding["filter_intercept"], N_LAGS)
        for k, entry in enumerate(encoding["transfer_per_session"]):
            entry["eta"] = expected["eta"][k * VOCAL_FRAMES.size:(k + 1) * VOCAL_FRAMES.size]
        encoding["spike_triggered_average"] = {"difference": np.zeros((2, N_LAGS))}
        data = tuning_data(UNIT_UID, paths, _identity(), encoding, _shipped_settings())
        assert np.allclose(data["vocal"]["eta"], expected["eta"])
        assert data["quiet"]["session_ids"] == [SESSION_IDS[2]]
        assert data["quiet"]["eta"].size == per_session[SESSION_IDS[2]]["quiet"].size
        assert data["sta"]["difference"].shape == (2, N_LAGS)
        assert "max |diff| 0.00e+00" in capsys.readouterr().out
        assert (paths.cache_directory / f"{UNIT_UID}_tuning_bout_window.pkl").exists()

    def test_tuning_data_refuses_a_different_vocal_frame_count(self, tmp_path, monkeypatch):
        """A rebuild that finds more vocal frames than the stored transfer scored is not the run's own."""
        _install_session_data(monkeypatch, _per_session())
        encoding = _encoding(["self.speed"], True)
        for entry in encoding["transfer_per_session"]:
            entry["eta"] = np.zeros(3)
        with pytest.raises(ValueError, match="stored transfer has 9"):
            tuning_data(UNIT_UID, _paths(tmp_path), _identity(), encoding, _shipped_settings())

    def test_rate_trace_data_calibrates_each_regime(self, tmp_path, monkeypatch):
        """Quiet probabilities cover the held-out session's sorted quiet frames, vocal ones each session's frames."""
        per_session = _per_session()
        _install_session_data(monkeypatch, per_session)
        paths = _paths(tmp_path)
        data = rate_trace_data(UNIT_UID, paths, _identity(), _encoding(["self.speed"], True), _shipped_settings())
        held_out = per_session[SESSION_IDS[2]]
        assert data["held_out"] == SESSION_IDS[2]
        assert np.array_equal(data["quiet"]["frames"], held_out["quiet"])
        assert data["quiet"]["spikes"].sum() == held_out["spike_frames"].size
        assert np.all((data["quiet"]["probability"] > 0.0) & (data["quiet"]["probability"] < 1.0))
        assert [entry["session_id"] for entry in data["vocal"]] == SESSION_IDS
        assert all(entry["probability"].size == VOCAL_FRAMES.size for entry in data["vocal"])
        assert (paths.cache_directory / f"{UNIT_UID}_rate_trace_v2_crossfit.pkl").exists()

    def test_session_trace_data_uses_each_fold_model(self, tmp_path, monkeypatch):
        """Each fitted fold predicts its own held-out session; a fold with no model at all is skipped, and a frame
        class without spikes is left unpredicted."""
        per_session = _per_session()
        held_out = per_session[SESSION_IDS[2]]
        held_out["spike_frames"] = held_out["spike_frames"][held_out["spike_frames"] % 2 == 0]
        _install_session_data(monkeypatch, per_session)
        identity = _identity()
        usvs = pls.DataFrame({"emitter": [identity["mouse_id"], None], "start": [10.0, 30.0], "stop": [10.5, 30.4]})
        monkeypatch.setattr(neural_modeling_plots, "load_session_usvs",
                            lambda _root, session_id: usvs if session_id == SESSION_IDS[0] else usvs.head(1))
        monkeypatch.setattr(neural_modeling_plots, "filter_vocalizations", lambda table, *_args: table)
        paths = _paths(tmp_path)
        rng = np.random.default_rng(0)
        folds = [{"no_model": np.array(False), "selected_indices": np.array([0]), "selected": np.array(["self.speed"]),
                  "coefficients": rng.normal(0.0, 0.1, N_LAGS)},
                 {"no_model": np.array(True)},
                 {"no_model": np.array(True), "best_screened": np.array(True), "selected_indices": np.array([0, 1]),
                  "selected": np.array(["self.speed", "head-head"]), "coefficients": rng.normal(0.0, 0.1, 2 * N_LAGS)}]
        for fold_index, arrays in enumerate(folds):
            np.savez(fold_artifact_path(str(paths.results_directory), identity, fold_index),
                     test_id=np.array(SESSION_IDS[fold_index]), **arrays)
        data = session_trace_data(UNIT_UID, paths, identity, _shipped_settings())
        assert [entry["session_id"] for entry in data["sessions"]] == [SESSION_IDS[0], SESSION_IDS[2]]
        assert data["sessions"][1]["features"] == ["self.speed", "head-head"]
        entry = data["sessions"][0]
        assert np.all(np.isnan(entry["probability"][:N_LAGS - 1]))
        assert np.isfinite(entry["probability"][per_session[SESSION_IDS[0]]["quiet"]]).all()
        assert np.array_equal(np.flatnonzero(entry["vocal"]), VOCAL_FRAMES)
        assert entry["unassigned"][int(30.2 * FPS)]
        assert not entry["unassigned"][int(10.2 * FPS)]
        assert not data["sessions"][1]["unassigned"].any()
        assert np.all(np.isnan(data["sessions"][1]["probability"][VOCAL_FRAMES]))
        assert (paths.cache_directory / f"{UNIT_UID}_session_traces_v2_crossfit.pkl").exists()

    def test_gating_proximity_data_measures_rates_and_distances(self, tmp_path, monkeypatch):
        """Rates come from spike times over each USV, distance at its onset; unusable USVs are dropped."""
        per_session = _per_session()
        identity = _identity()
        mouse, partner = identity["mouse_id"], "181316_9"
        n_frames = 3000
        partner_x = 5.0 + np.arange(n_frames) / 100.0
        partner_x[1500] = np.nan
        raw, track_names = {}, {}
        for index, session_id in enumerate(SESSION_IDS):
            nose = f"{mouse}-{partner}.nose-nose" if index == 0 else f"{partner}-{mouse}.nose-nose"
            raw[session_id] = pls.DataFrame({f"{mouse}.spaceX": np.zeros(n_frames), f"{mouse}.spaceY": np.zeros(n_frames),
                                             f"{mouse}.spaceZ": np.zeros(n_frames), f"{partner}.spaceX": partner_x,
                                             f"{partner}.spaceY": np.zeros(n_frames),
                                             f"{partner}.spaceZ": np.zeros(n_frames), nose: 2.0 * partner_x})
            track_names[session_id] = [mouse, partner]
        usvs = pls.DataFrame({"emitter": [mouse, mouse, mouse, mouse, partner],
                              "start": [10.0, 20.0, 59.9, 30.0, 40.0], "stop": [10.5, 20.2, 60.5, 30.2, 40.5],
                              "qlvm_category": [2.0, np.nan, 1.0, 3.0, 4.0]})

        def load_features(behavior_file_paths, csv_sep):
            """The behavioural features of every requested session, in the loader's return shape."""
            assert len(behavior_file_paths) == len(SESSION_IDS)
            assert csv_sep == ","
            return raw, {}, track_names

        monkeypatch.setattr(neural_modeling_plots, "assemble_unit_sessions", lambda *_args: per_session)
        monkeypatch.setattr(neural_modeling_plots, "load_behavioral_feature_data", load_features)
        monkeypatch.setattr(neural_modeling_plots, "load_unit_spike_frames",
                            lambda *_args: (np.array([20.1, 10.2, 10.1, 33.0]), None))
        monkeypatch.setattr(neural_modeling_plots, "load_session_usvs", lambda *_args: usvs)
        paths = _paths(tmp_path)
        data = gating_proximity_data(UNIT_UID, paths, identity, _shipped_settings())
        assert np.allclose(data["call_rate"], [4.0, 5.0] * 3)
        assert np.array_equal(data["call_category"], [2.0, np.nan] * 3, equal_nan=True)
        assert np.allclose(data["call_distance"]["head"], [partner_x[500], partner_x[1000]] * 3)
        assert np.allclose(data["call_distance"]["nose"], 2.0 * data["call_distance"]["head"])
        n_quiet = sum(int(np.sum(np.isfinite(partner_x[session["quiet"]]))) for session in per_session.values())
        assert data["quiet_distance"]["head"].size == n_quiet == data["quiet_label"].size
        assert data["fps"] == FPS
        assert (paths.cache_directory / f"{UNIT_UID}_gating_proximity.pkl").exists()

    def test_vocal_event_data_assembles_once(self, tmp_path, monkeypatch):
        """The events are assembled and cached on the first call and read back on the second."""
        calls = []

        def assemble(_identity_block, _data_root, _settings, pre_offset, width):
            """Record the window the assembly was asked for."""
            calls.append((pre_offset, width))
            return {"counts": np.arange(4.0)}

        monkeypatch.setattr(neural_modeling_plots, "assemble_unit_vocal_events", assemble)
        paths = _paths(tmp_path)
        first = vocal_event_data(UNIT_UID, paths, _identity(), _shipped_settings())
        second = vocal_event_data(UNIT_UID, paths, _identity(), _shipped_settings())
        assert calls == [(0.05, 0.1)]
        assert np.array_equal(first["counts"], second["counts"])


class TestCommandLine:
    """The command line runs the entry point."""

    def test_command_line_draws_through_the_entry_point(self, tmp_path):
        """Invoked with a result written under other settings, the command reports the skip and exits cleanly."""
        uid = "181316_0_20250919_imec1_cl0401_ch220_good"
        catalog = tmp_path / "unit_catalog.csv"
        catalog.write_text(f"mouse_id,rec_date,unit_id,brain_area\n181316_0,20250919,{uid[18:]},PAG\n")
        settings = _shipped_settings()
        settings["data_roots"]["catalog_path"] = str(catalog)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        _write_pickle(tmp_path / "results" / f"{uid}.pkl", {"provenance": {"settings_sha256": "0" * 64}})
        result = CliRunner().invoke(neural_modeling_figures_cli, [
            "--results-directory", str(tmp_path / "results"), "--output-directory", str(tmp_path / "figures"),
            "--data-root", str(tmp_path), "--unit-uid", uid, "--settings-path", str(settings_path)])
        assert result.exit_code == 0
        assert "SKIPPED" in result.output
