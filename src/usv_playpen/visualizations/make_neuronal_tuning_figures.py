"""
@author: bartulem
Makes per-cluster neuronal tuning figures: a single multi-page output
combining the behavioral feature tuning grid (one page per temporal
offset) and the vocal pages (Page 1: bout raster + pooled `usv_peth`
on top, `usv_property_tuning` 4x4 grid below; Page 2:
`usv_category_tuning` watersheds + `usv_category_peth` grid). Output
format is configurable via `figures.fig_format` in
visualizations_settings.json (`svg` default project-wide; this module
falls back to `pdf` if the key is missing). PDF is multi-page in one
file, other formats produce one file per page.

Output:
  ephys/tuning_curves/{cluster_id}_neuronal_tuning.{fig_format}
  (or, for non-PDF formats, ..._neuronal_tuning_p{N}_{label}.{fig_format})
"""

from __future__ import annotations

import contextlib
import csv
import math
import pathlib
import pickle
import warnings
from collections import Counter
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import gridspec
from matplotlib import patheffects as mpe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Patch
from scipy.stats import gaussian_kde, spearmanr
from tqdm import tqdm

from ..analyses.compute_behavioral_features import FeatureZoo
from ..analyses.compute_neuronal_tuning_curves import (
    CONTINUOUS_PROPERTIES,
    CATEGORICAL_FEATURES,
)
from ..analyses.decode_experiment_label import extract_information
from ..os_utils import first_match_or_raise
from .plot_style import apply_plot_style
from ..time_utils import is_gui_context, smart_wait
from .auxiliary_plot_functions import choose_animal_colors, create_colormap
from .figure_io import save_figure

apply_plot_style()


# Display-unit conversions: vocal_boundaries store CSV-native units; the
# plotter shows ms / kHz / etc. for readability. Axis-label strings live
# on FeatureZoo.vocal_labels (e.g. "low -- (kHz) -- high"), inherited via
# the class hierarchy and accessed as self.vocal_labels[prop] at draw time.
DISPLAY_FACTOR = {
    "duration": 1000.0,
    "mean_freq_hz": 1.0 / 1000.0,
    "peak_freq_hz": 1.0 / 1000.0,
    "freq_bandwidth_hz": 1.0 / 1000.0,
    "mean_amplitude": 1.0,
    "max_amplitude": 1.0,
    "spectral_entropy": 1.0,
    "mask_number": 1.0,
}

# Section-(b) feature ordering: 2 features per row × 4 rows
PROPERTY_ROW_ORDER = (
    ("duration", "mean_freq_hz"),
    ("peak_freq_hz", "freq_bandwidth_hz"),
    ("mean_amplitude", "max_amplitude"),
    ("spectral_entropy", "mask_number"),
)

# Page-2 section (c) layout: each row is one segmentation method (VAE
# in the first row, QLVM in the second), and each row holds the
# `_category` AND `_supercategory` variants side-by-side, contributing
# 3 cells (rate / occupancy / strip) each — so 6 cells per row in
# total, matching the behavioral feature grid's density.
SECTION_C_ROWS = (
    ("vae_category", "vae_supercategory"),
    ("qlvm_category", "qlvm_supercategory"),
)

# Hex palette
COLOR_BLACK = "#1F1F1F"
COLOR_GRAY_BAND = "#D3D3D3"
COLOR_GRAY_DASH = "#888888"
COLOR_LIGHT = "#CCCCCC"
COLOR_HATCH = "#BBBBBB"

# Population-level region grouping for the VMI summary figures. Maps
# canonical group labels (matched against
# `visualizations_parameter_dict['brain_area_colors']`) onto the raw
# `anatomy_region` strings in the unit triage pickle. SC pools every
# dorsal/intermediate SC subdivision; CENT pools cerebellar lobules;
# `Other` is the catch-all for everything not in the six explicit
# groups (mainly fiber tracts and small nuclei).
VMI_REGION_GROUPS: dict[str, set[str]] = {
    "PAG":  {"PAG"},
    "MRN":  {"MRN"},
    "VTA":  {"VTA"},
    "MB":   {"MB"},
    "CENT": {"CENT2", "CENT3"},
    "SC":   {"SCdw", "SCdg", "SCig", "SCiw", "SCop", "SCsg"},
}
VMI_REGION_ORDER: tuple[str, ...] = (
    "PAG", "MRN", "VTA", "MB", "CENT", "SC", "Other",
)

# Per-USV-property metadata for the population property-tuning
# distribution figures. Keys mirror the modality suffixes used by
# the triage pickle (`usv_property_self_<property>_excit`). Tolerance
# is the full-width window (so ±tol/2 around the cluster centre)
# applied to per-session `peak_bin_value` values during the
# consistency check; it's set to two upstream bin widths so the
# rule mirrors PETH's "±2 bins" convention. `unit_scale` and
# `unit_label` are display-only conversions for the x-axis (e.g. Hz
# → kHz).
USV_PROPERTY_META: dict[str, dict] = {
    "duration":          {"tol": 0.10, "unit_scale": 1.0,  "unit_label": "s",   "display_name": "USV duration"},
    "mean_freq_hz":      {"tol": 5000.0, "unit_scale": 1e-3, "unit_label": "kHz", "display_name": "USV mean freq"},
    "peak_freq_hz":      {"tol": 5000.0, "unit_scale": 1e-3, "unit_label": "kHz", "display_name": "USV peak freq"},
    "freq_bandwidth_hz": {"tol": 5000.0, "unit_scale": 1e-3, "unit_label": "kHz", "display_name": "USV freq bandwidth"},
    "mean_amplitude":    {"tol": 0.25, "unit_scale": 1.0,  "unit_label": "a.u.","display_name": "USV mean amplitude"},
    "max_amplitude":     {"tol": 0.80, "unit_scale": 1.0,  "unit_label": "a.u.","display_name": "USV max amplitude"},
    "spectral_entropy":  {"tol": 0.30, "unit_scale": 1.0,  "unit_label": "",    "display_name": "spectral entropy"},
    "mask_number":       {"tol": 2.0,  "unit_scale": 1.0,  "unit_label": "",    "display_name": "mask number"},
}
USV_PROPERTY_ORDER: tuple[str, ...] = tuple(USV_PROPERTY_META.keys())

# Categorical USV-tuning segmentations (the four `cat_feat` axes
# stored in the triage pickle's `usv_category_self_<segmentation>`
# modality keys). `n_classes` is the upstream class count for that
# segmentation, used to bound the per-region bar charts even when
# no units happen to land in some categories.
USV_CATEGORY_SEGMENTATIONS: tuple[str, ...] = (
    "vae_supercategory",
    "vae_category",
    "qlvm_supercategory",
    "qlvm_category",
)
USV_CATEGORY_N_CLASSES: dict[str, int] = {
    "vae_supercategory": 5,
    "vae_category":      10,
    "qlvm_supercategory": 7,
    "qlvm_category":     12,
}

# Behavioral tuning-summary bucket definitions.
#
# The behavioral modalities in the unit-triage pickle are keyed as
# `behavioral_beh_offset=0s_<prefix>.<feat>_<direction>`, where
# `<prefix>` is either a single mouse id (recorded-mouse self pose),
# another single mouse id (partner-self pose; ignored here), or a
# hyphenated pair (`<A>-<B>`, i.e. a dyadic feature that depends on
# BOTH animals — distances, allocentric angles toward partner, social-
# behavior tags such as orofacial-sei). The tuning-summary figure
# folds every per-feature consistency outcome into three orthogonal
# buckets, then assigns each unit to one of `2**3 = 8` disjoint tiers
# (`BEHAVIORAL_TIER_ORDER`).
#
# Per-feature consistency rule (matches the rest of the suite): a
# feature is "tuned" for a unit if either direction (excit / suppress)
# of its modality block has `n_significant >= k_min` significant
# sessions AND (when `require_majority` is True) those significant
# sessions form a majority of `n_tested`. The bucket flag is set if
# >= 1 feature in the bucket is tuned by that criterion.
BEHAVIORAL_POSE_FEATURES: frozenset[str] = frozenset({
    "allo_pitch", "allo_roll", "allo_yaw",
    "back_pitch", "back_yaw",
    "body_dir", "ego_yaw", "neck_elevation", "tail_curvature",
})

BEHAVIORAL_MOVEMENT_FEATURES: frozenset[str] = frozenset({
    "speed", "acceleration",
    "allo_pitch_1st_der", "allo_pitch_2nd_der",
    "allo_roll_1st_der",  "allo_roll_2nd_der",
    "allo_yaw_1st_der",   "allo_yaw_2nd_der",
    "back_pitch_1st_der", "back_pitch_2nd_der",
    "back_yaw_1st_der",   "back_yaw_2nd_der",
    "body_dir_1st_der",   "body_dir_2nd_der",
    "ego_yaw_1st_der",    "ego_yaw_2nd_der",
    "neck_elevation_1st_der", "neck_elevation_2nd_der",
    "tail_curvature_1st_der", "tail_curvature_2nd_der",
})

# The eight disjoint tiers produced by intersecting the three bucket
# flags. Display order is "most multimodal first": the triple
# intersection at the leftmost column, then the three two-bucket
# overlaps, then the three single-bucket tiers, with the trivial
# `none` tier at the rightmost column. A left-to-right scan therefore
# corresponds to "decreasingly multimodal tuning", which the figure
# uses to lead the reader's eye into the dominant tiers first.
BEHAVIORAL_TIER_ORDER: tuple[str, ...] = (
    "all_three",
    "pose+movement",
    "pose+social",
    "movement+social",
    "pose_only",
    "movement_only",
    "social_only",
    "none",
)

# Display labels for the heatmap column ticks. Kept short so the
# n_regions x 8 grid stays readable.
BEHAVIORAL_TIER_LABELS: dict[str, str] = {
    "none":            "none",
    "pose_only":       "P only",
    "movement_only":   "M only",
    "social_only":     "S only",
    "pose+movement":   "P+M",
    "pose+social":     "P+S",
    "movement+social": "M+S",
    "all_three":       "all 3",
}


# Page sizes are fixed by the layout invariants of each page (Page 1 has
# the section-(a) raster + usv_peth on top of the 4×4 usv_property_tuning grid; Page 2 has
# section-(c) 2×6 above section-(d) flowing 6 cols/row). They scale with
# the page dimensions, not with anything user-tunable, so are constants
# rather than settings.
VOCAL_PAGE1_FIGSIZE_INCHES = (16, 22)
VOCAL_PAGE2_FIGSIZE_INCHES = (16, 22)

# Section-(c) per-category strip plot scaling: switch to symlog when
# (max / min) firing-rate dynamic range exceeds the threshold; the
# linthresh defines the linear region around 0 in symlog units. Both are
# numerical knobs of the rendering algorithm rather than artist-tunable
# preferences.
VOCAL_STRIP_LOG_RATIO_THRESHOLD = 10.0
VOCAL_STRIP_SYMLOG_LINTHRESH = 0.5


def _parse_behavioral_modality_key(modality_key: str) -> tuple[str, str, str] | None:
    """
    Description
    -----------
    Parse a unit-triage `behavioral_beh_offset=0s_*` modality key into
    `(prefix, feature_name, direction)`. Returns `None` for any key
    that doesn't match the expected shape (so the caller can iterate
    over a full `modalities` dict without first filtering by prefix).

    The key shape, as produced by `unit_triage_aggregator.py`, is::

        behavioral_beh_offset=0s_<prefix>.<feat>_<direction>

    where `<prefix>` is either a single mouse id (`<a-zA-Z0-9>_<digit>`),
    a hyphenated pair (`<id_a>-<id_b>` for dyadic features), and
    `<feat>` is the kinematic-feature suffix (may itself contain
    hyphens, e.g. `nose-nose` or `allo_yaw-nose_1st_der`). The split
    uses string operations rather than regex because the hyphenated
    case would otherwise force a complex pattern that's harder to
    follow than the explicit rsplit / partition chain below.

    Parameters
    ----------
    modality_key (str)
        Modality key as stored under
        `unit["conditions"][cond]["modalities"]`.

    Returns
    -------
    parsed (tuple[str, str, str] | None)
        `(prefix, feature_name, direction)` on a match, `None`
        otherwise. `direction` is exactly `"excit"` or `"suppress"`.
    """

    tag = "behavioral_beh_offset=0s_"
    if not modality_key.startswith(tag):
        return None
    body = modality_key[len(tag):]
    if body.endswith("_excit"):
        direction = "excit"
        body = body[:-len("_excit")]
    elif body.endswith("_suppress"):
        direction = "suppress"
        body = body[:-len("_suppress")]
    else:
        return None
    if "." not in body:
        return None
    prefix, feat = body.split(".", 1)
    return prefix, feat, direction


def _decide_strip_xscale(
    observed: np.ndarray,
    null_p0_5: np.ndarray,
    null_p99_5: np.ndarray,
    log_ratio_threshold: float,
) -> str:
    """
    Description
    -----------
    Decide whether the per-category strip x-axis should be linear or
    symlog based on dynamic range. Min is computed only over strictly
    positive values to avoid division by zero; the presence of zeros
    does not by itself force linear scaling.

    Parameters
    ----------
    observed (np.ndarray)
        Per-category observed firing rates.
    null_p0_5, null_p99_5 (np.ndarray)
        Per-category shuffle 0.5 / 99.5 percentile bounds.
    log_ratio_threshold (float)
        If max/min(positive) > this, use symlog.

    Returns
    -------
    scale (str)
        "linear" or "symlog".
    """

    all_values = np.concatenate(
        [
            np.atleast_1d(observed).astype(float),
            np.atleast_1d(null_p0_5).astype(float),
            np.atleast_1d(null_p99_5).astype(float),
        ]
    )
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return "linear"
    positive = finite[finite > 0]
    if positive.size == 0:
        return "linear"
    if (finite.max() / positive.min()) > log_ratio_threshold:
        return "symlog"
    return "linear"


class NeuronalTuningFigureMaker(FeatureZoo):
    """
    Description
    -----------
    Per-cluster combined neuronal-tuning figure renderer. Reads each
    cluster's `*_tuning_curves_data.pkl` (carrying the unified behavioral
    + vocal payload) and emits one multi-page output per cluster: the
    behavioral feature pages (one per temporal offset, per plot-feature
    group) followed by the vocal pages (Page 1: bout raster + pooled
    `usv_peth` on top, `usv_property_tuning` 4x4 grid below; Page 2:
    `usv_category_tuning` watersheds + `usv_category_peth` grid). Pkls
    with neither behavioral nor vocal payload are skipped silently.
    Output filename is `{cluster_id}_neuronal_tuning.{fig_format}` (or,
    for non-PDF formats, `..._p{N}_{label}.{fig_format}`).

    Parameters
    ----------
    root_directory (str)
        Session root directory.
    visualizations_parameter_dict (dict)
        Settings dictionary; reads `male_colors`, `female_colors`, and
        the `neuronal_tuning_figures` block.
    message_output (Callable)
        Logger; defaults to print.
    """

    def __init__(self, **kwargs):
        """
        Description
        -----------
        Initialize the per-cluster figure maker. Loads `FeatureZoo`
        feature definitions (vocal boundaries / vocal labels / display
        units), validates and stashes the keyword arguments as attributes,
        records GUI-vs-CLI context, pins the path of the bundled
        latent-embedding segmentation file used by section (c), and primes
        a lazy segmentation cache.

        Parameters
        ----------
        **kwargs
            The expected keys are `root_directory`,
            `visualizations_parameter_dict` and `message_output`; each is set
            as an attribute and any other key raises ``TypeError``.

        Returns
        -------
        None
        """

        FeatureZoo.__init__(self)
        expected_kwargs = {'root_directory', 'visualizations_parameter_dict', 'message_output'}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword argument(s) "
                            f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                            f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val
        self.app_context_bool = is_gui_context()
        self._segmentation_path = (
            pathlib.Path(__file__).parent.parent
            / "_config"
            / "usv_latent_embedding_segmentation.npz"
        )
        self._segmentation_cache: dict | None = None

    # segmentation loading (lazy)

    def _load_segmentation(self) -> dict:
        """
        Description
        -----------
        Lazy-load the bundled latent-embedding segmentation file
        (`_config/usv_latent_embedding_segmentation.npz`) used to render the
        section-(c) categorical watersheds. Returns an empty dict if
        the file is absent. Cached on first call.

        Parameters
        ----------
        None

        Returns
        -------
        seg (dict)
            Mapping `cat_feat -> {label_grid, xx, yy, bounds,
            unique_labels}` for each categorical feature present in
            the segmentation file.
        """

        if self._segmentation_cache is not None:
            return self._segmentation_cache
        if not self._segmentation_path.exists():
            self._segmentation_cache = {}
            return self._segmentation_cache
        data = np.load(self._segmentation_path, allow_pickle=True)
        out: dict[str, dict] = {}
        for cat_feat in CATEGORICAL_FEATURES:
            if f"{cat_feat}__label_grid" not in data.files:
                continue
            out[cat_feat] = {
                "label_grid": data[f"{cat_feat}__label_grid"],
                "xx": data[f"{cat_feat}__xx"],
                "yy": data[f"{cat_feat}__yy"],
                "bounds": data[f"{cat_feat}__bounds"],
                "unique_labels": data[f"{cat_feat}__unique_labels"].tolist(),
            }
        self._segmentation_cache = out
        return out

    # color resolution

    def _sex_color(self, sex: str) -> str:
        """
        Description
        -----------
        Resolve the per-sex hex color used as the "primary" color for
        an emitter side. Reads `male_colors[0]` / `female_colors[0]`
        from the visualizations settings dict.

        Parameters
        ----------
        sex (str)
            Either "male" or "female"; anything other than "male"
            returns the female color.

        Returns
        -------
        hex_color (str)
            Six-digit hex color string with leading "#".
        """

        if sex == "male":
            return self.visualizations_parameter_dict["male_colors"][0]
        return self.visualizations_parameter_dict["female_colors"][0]

    # main entry point

    def make_neuronal_tuning_figures(self) -> None:
        """
        Description
        -----------
        Iterate over per-cluster pkls under `<root>/ephys/tuning_curves/`,
        and for each cluster render one combined output containing both
        the behavioral feature tuning pages and the vocal pages. Output
        format and per-cluster file naming are controlled by
        `figures.fig_format` (`pdf` produces a single multi-page file;
        other formats produce one file per page). Pkls
        with neither behavioral nor vocal payload are skipped silently.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        message_output = self.message_output if hasattr(self, "message_output") else print
        message_output(
            f"Making neuronal tuning figures started at: "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        root = pathlib.Path(self.root_directory)
        tuning_dir = root / "ephys" / "tuning_curves"
        if not tuning_dir.exists():
            message_output(f"  neuronal-figs: {tuning_dir} not found; skipping.")
            return

        pkl_files = sorted(tuning_dir.glob("*_tuning_curves_data.pkl"))
        if not pkl_files:
            message_output(f"  neuronal-figs: no tuning pkls in {tuning_dir}; skipping.")
            return

        viz_params = self.visualizations_parameter_dict.get(
            "neuronal_tuning_figures", {}
        )
        fig_format = str(
            self.visualizations_parameter_dict.get("figures", {}).get("fig_format", "pdf")
        ).lower()

        # try to load USV summary for the bout raster (left subplot of section a).
        usv_summary_df = None
        try:
            csv_path = first_match_or_raise(
                root=root,
                pattern="*_usv_summary.csv",
                recursive=True,
                label="USV summary CSV",
            )
            usv_summary_df = pls.read_csv(str(csv_path)).filter(pls.col("vae_supercategory") != 0)
        except (StopIteration, FileNotFoundError):
            pass

        # try to load tracking H5 (for behavioral mouse colors / animal IDs).
        # Search the whole session tree — mirrors `_load_behavioral_inputs`
        # and `_load_vocal_inputs` in compute_neuronal_tuning_curves.py.
        # Earlier revisions restricted to `<root>/video/`, which silently
        # produced an empty mouse_id_list (and therefore zero rendered
        # pages) on session layouts where the tracking H5 sits at the
        # session root instead of inside a `video/` subdirectory.
        mouse_id_list: list[str] = []
        mouse_colors: list = []
        try:
            tracked_file_loc = first_match_or_raise(
                root=root,
                pattern="[!speaker]*_points3d_translated_rotated_metric.h5",
                recursive=True,
                label="translated/rotated mouse points3d .h5",
            )
            with h5py.File(tracked_file_loc, mode="r") as tracking_data_3d:
                mouse_id_list = [
                    elem.decode("utf-8") for elem in tracking_data_3d["track_names"]
                ]
                session_exp_code = (
                    tracking_data_3d["experimental_code"][()].decode("utf-8")
                )
            experiment_info_dict = extract_information(experiment_code=session_exp_code)
            mouse_colors = choose_animal_colors(
                exp_info_dict=experiment_info_dict,
                visualizations_parameter_dict=self.visualizations_parameter_dict,
            )
        except (StopIteration, FileNotFoundError):
            pass

        segmentation = self._load_segmentation()

        # One-time warning when the tracking H5 was not found at all:
        # without it `mouse_id_list` is empty and behavioral pages cannot
        # render for any cluster, which used to fail silently per cluster.
        if not mouse_id_list:
            message_output(
                f"  neuronal-figs: no tracking H5 found under {root}; "
                "behavioral pages will be skipped for every cluster."
            )

        # double the spine thickness for all axes (mplstyle uses 0.75 default).
        spine_lw = 2.0 * float(plt.rcParams["axes.linewidth"])
        for pkl_file in tqdm(pkl_files, desc="neuronal tuning figures"):
            try:
                with pkl_file.open("rb") as fh:
                    cluster_data = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                message_output(f"  neuronal-figs: failed to load {pkl_file.name}: {exc}")
                continue
            cluster_id = pkl_file.stem.replace("_tuning_curves_data", "")
            has_behavioral = any(k.startswith("beh_offset=") for k in cluster_data)
            # Vocal payload is written by `_compute_one_cluster_vocal` under
            # top-level keys `usv_peth`, `usv_property_tuning`,
            # `usv_category_tuning`, `usv_category_peth`, `usv_metadata`.
            # An earlier revision of this predicate looked for `vocal_q*`,
            # which matches nothing the compute side ever writes — every
            # cluster would silently skip the vocal half.
            has_vocal = any(k.startswith("usv_") for k in cluster_data)
            # Skip whenever nothing can actually render: behavioral needs
            # both behavioral payload AND a populated mouse_id_list, vocal
            # needs vocal payload. If neither is renderable, do not open a
            # PdfPages — closing one with zero pages raises and the broad
            # except below would then mask the real cause.
            behavioral_renderable = has_behavioral and bool(mouse_id_list)
            if not (behavioral_renderable or has_vocal):
                continue
            out_path = tuning_dir / f"{cluster_id}_neuronal_tuning.{fig_format}"
            try:
                with plt.rc_context({"axes.linewidth": spine_lw}), self._open_save_target(
                    out_path=out_path,
                    cluster_id=cluster_id,
                    fig_format=fig_format,
                ) as save_fig:
                    if has_behavioral and mouse_id_list:
                        self._render_behavioral_pages(
                            cluster_data=cluster_data,
                            mouse_id_list=mouse_id_list,
                            mouse_colors=mouse_colors,
                            save_fig=save_fig,
                        )
                    if has_vocal:
                        self._render_vocal_pages(
                            cluster_data=cluster_data,
                            usv_summary_df=usv_summary_df,
                            segmentation=segmentation,
                            viz_params=viz_params,
                            save_fig=save_fig,
                        )
            except Exception as exc:  # noqa: BLE001
                message_output(
                    f"  neuronal-figs: failed to render {cluster_id}: {exc}"
                )
                continue

    # population VMI summary figures

    def _collect_vmi_best_session(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
    ) -> dict[str, list[dict]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit,
        find the "best" per-session VMI test (the session with the
        largest |VMI| among sessions that passed the `vmi_min_bouts`
        floor recorded in the pickle). Returns one entry per unit,
        grouped by canonical brain-area label (see `VMI_REGION_GROUPS`).

        The triage pickle's VMI tests are one-sided and split across
        `vmi_self_excit` / `vmi_self_suppress` modality keys with
        disjoint per-session lists; this method iterates over both
        keys, applies the bout-count floor + non-null p filter, and
        keeps the entry whose absolute VMI is the largest for that
        unit.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact produced
            by `unit_triage_aggregator.py`. The pickle must carry the
            usual `thresholds_used`, `catalog_path`, and `units` blocks.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV that pairs each
            `(mouse_id, rec_date, unit_id)` tuple with `cluster_group`
            and `somatic` flags. Defaults to the `catalog_path` field
            embedded in the triage pickle when omitted.

        Returns
        -------
        per_group (dict[str, list[dict]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a list of per-unit dicts with keys
            `vmi`, `p`, `significant`, `fr_baseline`, `fr_usv`,
            `n_bouts`. Empty groups are present (value = `[]`).
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        min_bouts = int(triage["thresholds_used"]["vmi_min_bouts"])

        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, list[dict]] = {g: [] for g in VMI_REGION_ORDER}

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            best_entry = None
            for cond in u["conditions"].values():
                modalities = cond["modalities"]
                for direction in ("vmi_self_excit", "vmi_self_suppress"):
                    if direction not in modalities:
                        continue
                    for entry in modalities[direction]["per_session"]:
                        if entry["n_bouts"] < min_bouts:
                            continue
                        if entry["p"] is None or entry["vmi"] is None:
                            continue
                        if (best_entry is None) or (abs(entry["vmi"]) > abs(best_entry["vmi"])):
                            best_entry = {
                                "vmi":         float(entry["vmi"]),
                                "p":           float(entry["p"]),
                                "significant": bool(entry["significant"]) and float(entry["p"]) < alpha,
                                "fr_baseline": float(entry["fr_baseline"]),
                                "fr_usv":      float(entry["fr_usv"]),
                                "n_bouts":     int(entry["n_bouts"]),
                            }
            if best_entry is not None:
                per_group[group].append(best_entry)

        return per_group

    def _resolve_region_colors(self) -> dict[str, str]:
        """
        Description
        -----------
        Pull the brain-area hex palette from
        `self.visualizations_parameter_dict['brain_area_colors']` and
        map the canonical `VMI_REGION_ORDER` labels to their colors.
        The `Other` group is intentionally matched against the
        lowercase `other` settings key so the JSON stays
        biologically-readable.

        Returns
        -------
        colors (dict[str, str])
            Mapping from canonical region label to hex color string.
        """

        palette = self.visualizations_parameter_dict["brain_area_colors"]
        return {
            "PAG":   palette["PAG"],
            "MRN":   palette["MRN"],
            "VTA":   palette["VTA"],
            "MB":    palette["MB"],
            "CENT":  palette["CENT"],
            "SC":    palette["SC"],
            "Other": palette["other"],
        }

    def make_vmi_fr_confound_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the FR-baseline × |VMI| confound diagnostic — a 2×4
        panel grid where the first seven cells scatter every
        good + somatic unit's best-session signed VMI against its
        baseline firing rate (one cell per brain-area group), and the
        eighth cell overlays the per-region |VMI| ECDFs for a direct
        across-region comparison.

        Coloring follows the brain-area palette in
        `visualizations_settings.json`: positive-significant points
        are drawn in the region's full color, negative-significant
        points in the same color at 50 % opacity, and non-significant
        points in the global unassigned-gray. Each scatter panel
        carries a small annotation reporting `N+`, `N-`, and the
        Spearman correlation between `log10(FR_baseline)` and `|VMI|`
        (the headline confound statistic).

        Output goes through `figure_io.save_figure` so the directory,
        format, and dpi all defer to `visualizations_settings.json`
        unless explicitly overridden.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact produced
            by `unit_triage_aggregator.py`.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` field embedded in the triage pickle when
            omitted.
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory. `None`
            defers to `visualizations_settings.json`.
        fig_format (str | None)
            Override the configured figure format. `None` defers to
            `visualizations_settings.json`.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_group = self._collect_vmi_best_session(triage_pkl_path, catalog_csv_path)
        region_colors = self._resolve_region_colors()
        non_sig_color = self.visualizations_parameter_dict["unassigned_colors"][0]

        with open(pathlib.Path(triage_pkl_path), "rb") as fh:
            triage = pickle.load(fh)
        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        min_bouts = int(triage["thresholds_used"]["vmi_min_bouts"])

        xlim_lo, xlim_hi = 1e-3, 1e2
        fig = plt.figure(figsize=(14, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.22,
            wspace=0.18,
            left=0.06, right=0.985,
            top=0.965, bottom=0.135,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            units = per_group[region]
            region_color = region_colors[region]

            if not units:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            fr_base = np.array([max(u["fr_baseline"], xlim_lo) for u in units])
            vmi = np.array([u["vmi"] for u in units])
            sig = np.array([u["significant"] for u in units])
            pos_sig = sig & (vmi > 0)
            neg_sig = sig & (vmi < 0)
            ns_mask = ~sig

            ax.scatter(
                fr_base[ns_mask], vmi[ns_mask],
                s=7, c=non_sig_color, alpha=0.45,
                edgecolors="none", rasterized=True,
            )
            ax.scatter(
                fr_base[neg_sig], vmi[neg_sig],
                s=14, c=region_color, alpha=0.5,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                rasterized=True,
            )
            ax.scatter(
                fr_base[pos_sig], vmi[pos_sig],
                s=14, c=region_color, alpha=0.95,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                rasterized=True,
            )

            rho, _ = spearmanr(np.log10(fr_base), np.abs(vmi))
            n_pos = int(pos_sig.sum())
            n_neg = int(neg_sig.sum())

            ax.set_xscale("log")
            ax.set_xlim(xlim_lo, xlim_hi)
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0.0, color=COLOR_BLACK, linewidth=0.5, linestyle=":")
            ax.set_title(f"{region}  (N={len(units)})", fontsize=10)
            if idx[0] == 1:
                ax.set_xlabel(r"FR$_\mathrm{baseline}$ (sp/s)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if idx[1] == 0:
                ax.set_ylabel("VMI (best session)", fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
            ax.text(
                0.03, 0.97,
                f"N+={n_pos}\nN-={n_neg}\n" + r"$\rho$=" + f"{rho:+.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=COLOR_HATCH, alpha=0.85),
            )

        # Aggregate |VMI| ECDF panel in the 8th cell.
        ax_agg = fig.add_subplot(gs[1, 3])
        for region in VMI_REGION_ORDER:
            units = per_group[region]
            if not units:
                continue
            abs_vmi = np.sort(np.array([abs(u["vmi"]) for u in units]))
            ecdf_y = np.arange(1, abs_vmi.size + 1) / abs_vmi.size
            ax_agg.plot(
                abs_vmi, ecdf_y,
                color=region_colors[region], linewidth=1.4,
                label=f"{region} (N={abs_vmi.size})",
            )
        ax_agg.set_xlim(0, 1)
        ax_agg.set_ylim(0, 1.02)
        ax_agg.set_xlabel("|VMI|  (best session per unit)", fontsize=9)
        ax_agg.set_ylabel("ECDF", fontsize=9)
        ax_agg.set_title("|VMI| distribution across regions", fontsize=10)
        ax_agg.tick_params(labelsize=8)
        ax_agg.legend(fontsize=7, frameon=False, loc="lower right")

        # Horizontal legend strip across the figure bottom — encodes the
        # significance / sign convention used in every scatter panel.
        leg_handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8,
                label=f"significant positive VMI (p<{alpha})",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8, alpha=0.5,
                label=f"significant negative VMI (p<{alpha})",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=non_sig_color, markeredgecolor="none",
                markersize=8, alpha=0.7,
                label="non-significant",
            ),
        ]
        fig.legend(
            handles=leg_handles, loc="lower center", ncol=3,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 0.03),
        )
        fig.text(
            0.5, 0.005,
            f"one dot per unit, best-session by |VMI|  ·  min n_bouts={min_bouts}"
            f"  ·  good + somatic only",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem="vmi_fr_confound",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_vmi_per_condition_medians(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            cond_a: str = "intact_female",
            cond_b: str = "mute_female",
    ) -> dict[str, list[dict]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit
        that has at least one valid per-session VMI test in BOTH
        `cond_a` AND `cond_b`, summarise the unit's signed VMI in each
        condition as the median of its per-session VMI values (across
        the `vmi_self_excit` + `vmi_self_suppress` modality keys).
        Also flags whether the unit was significant in either or both
        conditions, using the `vmi_alpha` threshold from the pickle.

        Valid per-session entries are those with `n_bouts >=
        vmi_min_bouts` and non-null `p` / `vmi`.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` stored in the triage pickle.
        cond_a, cond_b (str)
            The two condition keys to compare (must both exist in the
            pickle's `conditions_included` block).

        Returns
        -------
        per_group (dict[str, list[dict]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a list of per-unit dicts with keys
            `vmi_a`, `vmi_b`, `sig_a`, `sig_b`. Units missing either
            condition are excluded entirely.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        min_bouts = int(triage["thresholds_used"]["vmi_min_bouts"])
        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, list[dict]] = {g: [] for g in VMI_REGION_ORDER}

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue
            if cond_a not in u["conditions"] or cond_b not in u["conditions"]:
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            # Per-condition aggregation: median of signed VMI across
            # all valid per-session entries (both directions pooled —
            # excit/suppress entries are mutually exclusive per
            # session by construction of the one-sided tests).
            summaries: dict[str, dict] = {}
            all_fr_baseline: list[float] = []
            for cond_name in (cond_a, cond_b):
                vmi_values: list[float] = []
                sig_flag = False
                modalities = u["conditions"][cond_name]["modalities"]
                for direction in ("vmi_self_excit", "vmi_self_suppress"):
                    if direction not in modalities:
                        continue
                    for entry in modalities[direction]["per_session"]:
                        if entry["n_bouts"] < min_bouts:
                            continue
                        if entry["p"] is None or entry["vmi"] is None:
                            continue
                        vmi_values.append(float(entry["vmi"]))
                        all_fr_baseline.append(float(entry["fr_baseline"]))
                        if bool(entry["significant"]) and float(entry["p"]) < alpha:
                            sig_flag = True
                if not vmi_values:
                    summaries = {}
                    break
                summaries[cond_name] = {
                    "vmi_median": float(np.median(vmi_values)),
                    "sig":        sig_flag,
                }
            if not summaries:
                continue

            per_group[group].append({
                "vmi_a":              summaries[cond_a]["vmi_median"],
                "vmi_b":              summaries[cond_b]["vmi_median"],
                "sig_a":              summaries[cond_a]["sig"],
                "sig_b":              summaries[cond_b]["sig"],
                "fr_baseline_median": float(np.median(all_fr_baseline)),
            })

        return per_group

    def make_vmi_cross_condition_stability_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            cond_a: str = "intact_female",
            cond_b: str = "mute_female",
            n_bootstrap: int = 2000,
            random_seed: int = 0,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the cross-condition VMI stability diagnostic — a 2×4
        panel grid where the first seven cells scatter each unit's
        median signed VMI in `cond_a` against its median signed VMI
        in `cond_b` (one cell per brain-area group), and the eighth
        cell summarises the per-region Pearson correlation between
        the two conditions with 95 % bootstrap CIs.

        Only units that have valid sessions in BOTH conditions are
        included. Per-condition VMI is the median of signed
        per-session VMI values (across the excit / suppress modality
        keys); significance is "any valid session in that condition
        was significant at the pickle's `vmi_alpha`".

        Coloring follows the brain-area palette in
        `visualizations_settings.json`:

          * tuned in BOTH conditions, same sign: full region color
          * tuned in BOTH conditions, opposite sign: region color at
            50 % opacity (the sign-flippers — likely artifact or
            unstable cells)
          * tuned in ONLY ONE condition: region color at 30 % opacity
          * tuned in NEITHER: small unassigned-gray dot

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` field embedded in the triage pickle.
        cond_a, cond_b (str)
            Names of the two conditions to compare. Default to the
            intact-female vs mute-female contrast.
        n_bootstrap (int)
            Number of resampling iterations for the per-region
            Pearson-r confidence intervals (8th panel).
        random_seed (int)
            Seed for the bootstrap RNG, so the CIs are reproducible
            across reruns.
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_group = self._collect_vmi_per_condition_medians(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            cond_a=cond_a,
            cond_b=cond_b,
        )
        region_colors = self._resolve_region_colors()
        non_sig_color = self.visualizations_parameter_dict["unassigned_colors"][0]

        # Marker-size scale: linear in log10(FR_baseline) so each
        # decade of firing rate maps to the same number of points.
        # FR is clipped to [0.1, 30] sp/s before the log transform —
        # this is the band the bulk of good + somatic units fall in,
        # so concentrating the size range there gives a visible
        # gradient instead of compressing everything to similar dots.
        FR_SIZE_MIN, FR_SIZE_MAX = 4.0, 70.0
        FR_LOG_MIN, FR_LOG_MAX = -1.0, 1.5  # 0.1 Hz, ~31.6 Hz

        def _fr_to_size(fr_values: np.ndarray) -> np.ndarray:
            """
            Description
            -----------
            Map an array of `fr_baseline` values (sp/s) to matplotlib
            scatter marker sizes (points²) so dot area increases
            linearly with `log10(FR)`. FR is clipped to
            [`10**FR_LOG_MIN`, `10**FR_LOG_MAX`] to keep extreme
            outliers from compressing the rest of the range.

            Parameters
            ----------
            fr_values (np.ndarray)
                Per-unit firing rates in sp/s.

            Returns
            -------
            sizes (np.ndarray)
                Per-unit `s=` values for `ax.scatter`.
            """
            log_fr = np.log10(np.clip(fr_values, 10.0**FR_LOG_MIN, 10.0**FR_LOG_MAX))
            frac = (log_fr - FR_LOG_MIN) / (FR_LOG_MAX - FR_LOG_MIN)
            return FR_SIZE_MIN + (FR_SIZE_MAX - FR_SIZE_MIN) * frac

        rng = np.random.default_rng(random_seed)
        # Pearson r + bootstrap CIs per region for the 8th panel.
        r_summary: dict[str, dict] = {}
        for region in VMI_REGION_ORDER:
            units = per_group[region]
            if len(units) < 5:
                r_summary[region] = {"r": float("nan"), "lo": float("nan"), "hi": float("nan"),
                                     "n": len(units)}
                continue
            x = np.array([u["vmi_a"] for u in units])
            y = np.array([u["vmi_b"] for u in units])
            r_pt = float(np.corrcoef(x, y)[0, 1])
            n = x.size
            idx_pool = np.arange(n)
            boot_rs = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                sample = rng.choice(idx_pool, size=n, replace=True)
                xs, ys = x[sample], y[sample]
                if xs.std() == 0 or ys.std() == 0:
                    boot_rs[b] = float("nan")
                else:
                    boot_rs[b] = float(np.corrcoef(xs, ys)[0, 1])
            r_lo, r_hi = np.nanquantile(boot_rs, [0.025, 0.975])
            r_summary[region] = {"r": r_pt, "lo": float(r_lo), "hi": float(r_hi), "n": n}

        fig = plt.figure(figsize=(14, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.22,
            wspace=0.22,
            left=0.06, right=0.985,
            top=0.965, bottom=0.135,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            units = per_group[region]
            region_color = region_colors[region]

            if not units:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            vmi_a = np.array([u["vmi_a"] for u in units])
            vmi_b = np.array([u["vmi_b"] for u in units])
            sig_a = np.array([u["sig_a"] for u in units])
            sig_b = np.array([u["sig_b"] for u in units])
            fr_b = np.array([u["fr_baseline_median"] for u in units])
            sizes = _fr_to_size(fr_b)
            both_sig = sig_a & sig_b
            same_sign = (np.sign(vmi_a) == np.sign(vmi_b))
            both_same = both_sig & same_sign
            both_flip = both_sig & (~same_sign)
            one_only = (sig_a ^ sig_b)
            neither = ~(sig_a | sig_b)

            # Layer dots: neither to one-only to flip to same-sign.
            # Anything not "sig in both conditions" is rendered in the
            # unassigned-gray so the region color is reserved for
            # robustly tuned units; one-only gets a darker shade than
            # neither so the two groups stay visually distinct.
            # Dot SIZE encodes log10(FR_baseline) — sparse cells small,
            # high-FR cells large — so the (well-known) inverse
            # relationship between FR_baseline and |VMI| is readable
            # straight off the scatter.
            ax.scatter(
                vmi_a[neither], vmi_b[neither],
                s=sizes[neither], c=non_sig_color, alpha=0.30,
                edgecolors="none", rasterized=True,
            )
            ax.scatter(
                vmi_a[one_only], vmi_b[one_only],
                s=sizes[one_only], c=non_sig_color, alpha=0.65,
                edgecolors="none", rasterized=True,
            )
            ax.scatter(
                vmi_a[both_flip], vmi_b[both_flip],
                s=sizes[both_flip], c=region_color, alpha=0.55,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                rasterized=True,
            )
            ax.scatter(
                vmi_a[both_same], vmi_b[both_same],
                s=sizes[both_same], c=region_color, alpha=0.95,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                rasterized=True,
            )

            # Reference lines.
            ax.axhline(0.0, color=COLOR_BLACK, linewidth=0.5, linestyle=":")
            ax.axvline(0.0, color=COLOR_BLACK, linewidth=0.5, linestyle=":")
            ax.plot([-1, 1], [-1, 1], color=COLOR_GRAY_DASH, linewidth=0.6, linestyle="--")

            r_pt = r_summary[region]["r"]
            n_total = len(units)
            n_same = int(both_same.sum())
            n_flip = int(both_flip.sum())

            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{region}  (N={n_total})", fontsize=10)
            cond_a_label = cond_a.replace("_", " ")
            cond_b_label = cond_b.replace("_", " ")
            if idx[0] == 1:
                ax.set_xlabel(f"VMI ({cond_a_label})", fontsize=9)
            else:
                ax.set_xticklabels([])
            if idx[1] == 0:
                ax.set_ylabel(f"VMI ({cond_b_label})", fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
            ax.text(
                0.03, 0.97,
                f"r={r_pt:+.2f}\n"
                f"same={n_same}\n"
                f"flip={n_flip}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=COLOR_HATCH, alpha=0.85),
            )

            # Size legend inside the PAG panel (largest sample,
            # most populated scatter, so the FR-size mapping reads
            # off cleanly here).
            if region == "PAG":
                fr_ref_values = np.array([0.1, 1.0, 10.0])
                fr_ref_sizes = _fr_to_size(fr_ref_values)
                x_dot = 0.78
                y_top = 0.22
                dy = 0.07
                for i, (fr_val, sz) in enumerate(zip(fr_ref_values, fr_ref_sizes)):
                    ax.scatter(
                        x_dot, y_top - dy * i,
                        s=sz, c=COLOR_BLACK, edgecolors="none",
                        transform=ax.transAxes, clip_on=False, zorder=5,
                    )
                    ax.text(
                        x_dot + 0.045, y_top - dy * i,
                        f"{fr_val:g} sp/s", fontsize=7, color=COLOR_BLACK,
                        ha="left", va="center", transform=ax.transAxes,
                    )
                ax.text(
                    x_dot + 0.045, y_top + dy * 0.85,
                    "FRbase",
                    fontsize=7, color=COLOR_BLACK,
                    ha="left", va="center", transform=ax.transAxes,
                )

        # 8th panel: per-region Pearson r with bootstrap CIs.
        ax_r = fig.add_subplot(gs[1, 3])
        bar_x = np.arange(len(VMI_REGION_ORDER))
        bar_vals = np.array([r_summary[g]["r"] for g in VMI_REGION_ORDER])
        bar_lo = np.array([r_summary[g]["lo"] for g in VMI_REGION_ORDER])
        bar_hi = np.array([r_summary[g]["hi"] for g in VMI_REGION_ORDER])
        bar_n = [r_summary[g]["n"] for g in VMI_REGION_ORDER]
        bar_colors = [region_colors[g] for g in VMI_REGION_ORDER]

        err_lower = np.where(np.isnan(bar_vals - bar_lo), 0, bar_vals - bar_lo)
        err_upper = np.where(np.isnan(bar_hi - bar_vals), 0, bar_hi - bar_vals)

        ax_r.bar(
            bar_x, bar_vals,
            color=bar_colors,
            edgecolor=COLOR_BLACK, linewidth=0.6,
            width=0.78,
        )
        ax_r.errorbar(
            bar_x, bar_vals,
            yerr=np.vstack([err_lower, err_upper]),
            fmt="none", ecolor=COLOR_BLACK,
            elinewidth=0.8, capsize=3, capthick=0.8,
        )
        ax_r.axhline(0.0, color=COLOR_BLACK, linewidth=0.5)
        ax_r.set_xticks(bar_x)
        ax_r.set_xticklabels(
            [f"{g}\nN={n}" for g, n in zip(VMI_REGION_ORDER, bar_n)],
            fontsize=8,
        )
        ax_r.set_ylim(-0.2, 1.0)
        ax_r.set_ylabel(
            f"Pearson r  ({cond_a.replace('_', ' ')} vs {cond_b.replace('_', ' ')})",
            fontsize=9,
        )
        ax_r.set_title("cross-condition VMI correlation", fontsize=10)
        ax_r.tick_params(axis="y", labelsize=8)

        # Horizontal legend strip across the figure bottom.
        leg_handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8,
                label="sig in both (same sign)",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8, alpha=0.55,
                label="sig in both (sign flip)",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=non_sig_color, markeredgecolor="none",
                markersize=8, alpha=0.65,
                label="sig in one condition only",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=non_sig_color, markeredgecolor="none",
                markersize=8, alpha=0.30,
                label="sig in neither",
            ),
        ]
        fig.legend(
            handles=leg_handles, loc="lower center", ncol=4,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 0.03),
        )
        fig.text(
            0.5, 0.005,
            f"one dot per unit, median signed VMI across sessions  ·  "
            f"good + somatic, present in both {cond_a.replace('_', ' ')} "
            f"and {cond_b.replace('_', ' ')}  ·  "
            r"dot area $\propto$ $\log_{10}$ FRbase",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )


        out_path = save_figure(
            fig=fig,
            stem="vmi_cross_condition_stability",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_vmi_consistency(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            n_tested_min: int = 2,
    ) -> dict[str, list[dict]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit
        with at least `n_tested_min` valid sessions, summarise its
        cross-session VMI evidence as `(max |VMI|, consistency,
        n_tested)`. Per-session entries from both modality directions
        (`vmi_self_excit` / `vmi_self_suppress`) and both conditions
        are pooled into one (unit-level) view: a session counts as
        "tested" if any valid entry survived the `vmi_min_bouts`
        floor, and "significant" if any entry was flagged significant
        at the pickle's `vmi_alpha`. `consistency` is then
        `n_sig_sessions / n_tested_sessions`.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` stored in the triage pickle.
        n_tested_min (int)
            Minimum number of distinct sessions with a valid entry
            required to include the unit. Defaults to 2 — units with
            a single valid session have degenerate consistency
            (either 0 or 1) and pollute the distribution.

        Returns
        -------
        per_group (dict[str, list[dict]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a list of per-unit dicts with keys
            `max_abs_vmi`, `consistency`, `n_tested`, `n_sig`.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        min_bouts = int(triage["thresholds_used"]["vmi_min_bouts"])
        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, list[dict]] = {g: [] for g in VMI_REGION_ORDER}

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            # Pool per-session evidence across conditions + directions.
            # A session counts as tested if ANY direction had a valid
            # entry, and significant if ANY entry was sig.
            per_session_max_abs: dict[str, float] = {}
            per_session_sig:     dict[str, bool] = {}
            for cond in u["conditions"].values():
                modalities = cond["modalities"]
                for direction in ("vmi_self_excit", "vmi_self_suppress"):
                    if direction not in modalities:
                        continue
                    for entry in modalities[direction]["per_session"]:
                        if entry["n_bouts"] < min_bouts:
                            continue
                        if entry["p"] is None or entry["vmi"] is None:
                            continue
                        session = entry["session"]
                        abs_v = abs(float(entry["vmi"]))
                        prev = per_session_max_abs[session] if session in per_session_max_abs else -1.0
                        if abs_v > prev:
                            per_session_max_abs[session] = abs_v
                        is_sig = bool(entry["significant"]) and float(entry["p"]) < alpha
                        if is_sig:
                            per_session_sig[session] = True
                        elif session not in per_session_sig:
                            per_session_sig[session] = False

            n_tested = len(per_session_max_abs)
            if n_tested < n_tested_min:
                continue
            n_sig = sum(1 for v in per_session_sig.values() if v)
            max_abs_vmi = max(per_session_max_abs.values())
            per_group[group].append({
                "max_abs_vmi": float(max_abs_vmi),
                "consistency": float(n_sig) / float(n_tested),
                "n_tested":    int(n_tested),
                "n_sig":       int(n_sig),
            })

        return per_group

    def make_vmi_magnitude_consistency_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            n_tested_min: int = 2,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the magnitude × consistency diagnostic — a 2×4 panel
        grid where the first seven cells scatter each good + somatic
        unit's `max |VMI|` (across all valid per-session entries)
        against its per-session consistency
        (`n_significant / n_tested`), one cell per brain-area group.
        The eighth cell stacks each region's units into three tiers:
        never-significant (gray), sig but flickery (`consistency
        < 0.5`, region color at 50 % opacity), and sig with high
        consistency (`consistency >= 0.5`, full region color).

        Reference lines mark the strong-effect threshold at `|VMI| =
        0.5` and the majority threshold at `consistency = 0.5`. The
        top-right quadrant ("strong AND stable") is the
        robust-tuning population.

        Coloring follows the brain-area palette in
        `visualizations_settings.json`. Dot size is linear in
        `n_tested`, so units backed by more session-evidence are
        visually larger.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` embedded in the triage pickle.
        n_tested_min (int)
            Minimum number of valid sessions required to include the
            unit (default 2; single-session units have degenerate
            consistency).
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_group = self._collect_vmi_consistency(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            n_tested_min=n_tested_min,
        )
        region_colors = self._resolve_region_colors()
        non_sig_color = self.visualizations_parameter_dict["unassigned_colors"][0]

        # Dot size scaling — linear in n_tested. n_tested in this
        # dataset spans 2..6, so a linear map keeps every value
        # visually distinct.
        NT_SIZE_MIN, NT_SIZE_MAX = 6.0, 60.0
        NT_RANGE_MIN, NT_RANGE_MAX = 2, 6

        def _nt_to_size(nt: np.ndarray) -> np.ndarray:
            """
            Description
            -----------
            Map an array of `n_tested` integers to matplotlib scatter
            marker sizes (points²). Linear in `n_tested`, clipped to
            `[NT_RANGE_MIN, NT_RANGE_MAX]` so out-of-band values fall
            on the endpoints.

            Parameters
            ----------
            nt (np.ndarray)
                Per-unit `n_tested` values.

            Returns
            -------
            sizes (np.ndarray)
                Per-unit `s=` values.
            """
            clipped = np.clip(nt, NT_RANGE_MIN, NT_RANGE_MAX).astype(float)
            frac = (clipped - NT_RANGE_MIN) / (NT_RANGE_MAX - NT_RANGE_MIN)
            return NT_SIZE_MIN + (NT_SIZE_MAX - NT_SIZE_MIN) * frac

        fig = plt.figure(figsize=(14, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.22,
            wspace=0.22,
            left=0.06, right=0.985,
            top=0.965, bottom=0.135,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            units = per_group[region]
            region_color = region_colors[region]

            if not units:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x = np.array([u["max_abs_vmi"] for u in units])
            y = np.array([u["consistency"] for u in units])
            nt = np.array([u["n_tested"] for u in units])
            ns = np.array([u["n_sig"] for u in units])
            sizes = _nt_to_size(nt)
            sig_any = ns >= 1
            never_sig = ~sig_any
            stable = y >= 0.5
            flickery = sig_any & (~stable)
            n_total = len(units)
            n_stable = int(stable.sum())
            n_flicker = int(flickery.sum())
            n_never = int(never_sig.sum())

            # Background dots first (never-sig pile-up at y=0),
            # then flickery (mid-opacity), then stable (full).
            ax.scatter(
                x[never_sig], y[never_sig],
                s=sizes[never_sig], c=non_sig_color, alpha=0.45,
                edgecolors="none", rasterized=True,
            )
            ax.scatter(
                x[flickery], y[flickery],
                s=sizes[flickery], c=region_color, alpha=0.55,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                rasterized=True,
            )
            ax.scatter(
                x[stable], y[stable],
                s=sizes[stable], c=region_color, alpha=0.95,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                rasterized=True,
            )

            ax.axhline(0.5, color=COLOR_GRAY_DASH, linewidth=0.6, linestyle="--")
            ax.axvline(0.5, color=COLOR_GRAY_DASH, linewidth=0.6, linestyle="--")

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_title(f"{region}  (N={n_total})", fontsize=10)
            if idx[0] == 1:
                ax.set_xlabel("max |VMI| across sessions", fontsize=9)
            else:
                ax.set_xticklabels([])
            if idx[1] == 0:
                ax.set_ylabel(
                    r"consistency  ($n_\mathrm{sig}\,/\,n_\mathrm{tested}$)",
                    fontsize=9,
                )
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
            ax.text(
                0.03, 0.97,
                f"stable={n_stable}\n"
                f"flicker={n_flicker}\n"
                f"never={n_never}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=COLOR_HATCH, alpha=0.85),
            )

            # Size legend inside the CENT panel — three reference dots
            # at n_tested = 2, 4, 6 so the marker-area mapping reads
            # off cleanly. CENT's upper-right is reliably uncluttered
            # (CENT cells cluster near the bottom edge / left side of
            # the consistency axis) so the legend doesn't collide
            # with data.
            if region == "CENT":
                nt_ref_values = np.array([2, 4, 6])
                nt_ref_sizes = _nt_to_size(nt_ref_values)
                x_dot = 0.86
                y_top = 0.90
                dy = 0.075
                ax.text(
                    x_dot, y_top + dy * 0.95,
                    r"$n_\mathrm{tested}$", fontsize=7, color=COLOR_BLACK,
                    ha="center", va="center", transform=ax.transAxes,
                )
                for i, (nt_val, sz) in enumerate(zip(nt_ref_values, nt_ref_sizes)):
                    ax.scatter(
                        x_dot, y_top - dy * i,
                        s=sz, c=COLOR_BLACK, edgecolors="none",
                        transform=ax.transAxes, clip_on=False, zorder=5,
                    )
                    ax.text(
                        x_dot - 0.045, y_top - dy * i,
                        f"{int(nt_val)}", fontsize=7, color=COLOR_BLACK,
                        ha="right", va="center", transform=ax.transAxes,
                    )

        # 8th panel: stacked-bar tier breakdown per region.
        ax_stack = fig.add_subplot(gs[1, 3])
        bar_x = np.arange(len(VMI_REGION_ORDER))
        frac_never = np.zeros(len(VMI_REGION_ORDER))
        frac_flicker = np.zeros(len(VMI_REGION_ORDER))
        frac_stable = np.zeros(len(VMI_REGION_ORDER))
        n_per_region: list[int] = []
        for i, region in enumerate(VMI_REGION_ORDER):
            units = per_group[region]
            n = len(units)
            n_per_region.append(n)
            if n == 0:
                continue
            cons = np.array([u["consistency"] for u in units])
            ns = np.array([u["n_sig"] for u in units])
            mask_never = ns == 0
            mask_stable = cons >= 0.5
            mask_flicker = (~mask_never) & (~mask_stable)
            frac_never[i] = mask_never.sum() / n
            frac_flicker[i] = mask_flicker.sum() / n
            frac_stable[i] = mask_stable.sum() / n

        bar_colors = [region_colors[r] for r in VMI_REGION_ORDER]
        flicker_face = [mcolors.to_rgba(c, 0.5) for c in bar_colors]
        stable_face = [mcolors.to_rgba(c, 1.0) for c in bar_colors]
        never_face = mcolors.to_rgba(non_sig_color, 0.7)

        ax_stack.bar(
            bar_x, frac_never,
            color=[never_face] * len(VMI_REGION_ORDER),
            edgecolor=COLOR_BLACK, linewidth=0.4, width=0.78,
        )
        ax_stack.bar(
            bar_x, frac_flicker,
            bottom=frac_never,
            color=flicker_face,
            edgecolor=COLOR_BLACK, linewidth=0.4, width=0.78,
        )
        ax_stack.bar(
            bar_x, frac_stable,
            bottom=frac_never + frac_flicker,
            color=stable_face,
            edgecolor=COLOR_BLACK, linewidth=0.4, width=0.78,
        )
        ax_stack.set_xticks(bar_x)
        ax_stack.set_xticklabels(
            [f"{r}\nN={n}" for r, n in zip(VMI_REGION_ORDER, n_per_region)],
            fontsize=8,
        )
        ax_stack.set_ylim(0.0, 1.0)
        ax_stack.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_stack.set_ylabel("fraction of units", fontsize=9)
        ax_stack.set_title("consistency tier breakdown", fontsize=10)
        ax_stack.tick_params(axis="y", labelsize=8)

        # Horizontal legend strip across the figure bottom.
        leg_handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8,
                label=r"sig + stable  (consistency $\geq$ 0.5)",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8, alpha=0.55,
                label="sig + flickery  (consistency < 0.5)",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=non_sig_color, markeredgecolor="none",
                markersize=8, alpha=0.45,
                label="never significant",
            ),
        ]
        fig.legend(
            handles=leg_handles, loc="lower center", ncol=3,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 0.03),
        )
        fig.text(
            0.5, 0.005,
            f"one dot per unit  ·  good + somatic, "
            f"$n_{{\\mathrm{{tested}}}}\\geq{n_tested_min}$  ·  "
            r"dot area $\propto$ $n_\mathrm{tested}$",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem="vmi_magnitude_consistency",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_vmi_sign_flip_tally(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            n_tested_min: int = 2,
    ) -> dict[str, dict[str, int]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit
        with at least `n_tested_min` valid per-session VMI tests,
        count which categorical tier it falls into based on the SIGN
        of its significant sessions across both modality keys
        (`vmi_self_excit` + `vmi_self_suppress`) and both conditions:

          * `sig_pos_only`   — at least one significant +VMI session,
            no significant −VMI session.
          * `sig_neg_only`   — at least one significant −VMI session,
            no significant +VMI session.
          * `sig_both`       — at least one significant +VMI AND at
            least one significant −VMI session (the unit's tuning
            direction flipped across sessions and was statistically
            supported in both directions).
          * `never_sig`      — no significant session in either
            direction.

        Per-region returns a four-key counts dict so the downstream
        bar plot can stack the tiers as fractions of `N_units`.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` field stored in the triage pickle.
        n_tested_min (int)
            Minimum number of distinct sessions with a valid entry
            required to include the unit. Defaults to 2.

        Returns
        -------
        tally (dict[str, dict[str, int]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a dict with keys
            `sig_pos_only`, `sig_neg_only`, `sig_both`, `never_sig`,
            and `n_total`.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        min_bouts = int(triage["thresholds_used"]["vmi_min_bouts"])
        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        tally: dict[str, dict[str, int]] = {
            g: {"sig_pos_only": 0, "sig_neg_only": 0, "sig_both": 0,
                "never_sig":    0, "n_total":      0}
            for g in VMI_REGION_ORDER
        }

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            sessions_seen: set[str] = set()
            sig_pos = False
            sig_neg = False
            for cond in u["conditions"].values():
                modalities = cond["modalities"]
                for direction in ("vmi_self_excit", "vmi_self_suppress"):
                    if direction not in modalities:
                        continue
                    for entry in modalities[direction]["per_session"]:
                        if entry["n_bouts"] < min_bouts:
                            continue
                        if entry["p"] is None or entry["vmi"] is None:
                            continue
                        sessions_seen.add(entry["session"])
                        is_sig = bool(entry["significant"]) and float(entry["p"]) < alpha
                        if not is_sig:
                            continue
                        if float(entry["vmi"]) > 0:
                            sig_pos = True
                        elif float(entry["vmi"]) < 0:
                            sig_neg = True

            if len(sessions_seen) < n_tested_min:
                continue
            tally[group]["n_total"] += 1
            if sig_pos and sig_neg:
                tally[group]["sig_both"] += 1
            elif sig_pos:
                tally[group]["sig_pos_only"] += 1
            elif sig_neg:
                tally[group]["sig_neg_only"] += 1
            else:
                tally[group]["never_sig"] += 1

        return tally

    def make_vmi_sign_flip_summary_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            n_tested_min: int = 2,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the per-region significance-tier breakdown — one
        stacked vertical bar per brain-area group (PAG, MRN, VTA, MB,
        CENT, SC, Other), with the bar height fixed at 1.0 (fraction
        of units) and split into four tiers:

          * never-sig (bottom, unassigned-gray): no significant
            session in either direction.
          * sig+ only (region color, full opacity): committed to
            positive VMI across all significant sessions.
          * sig− only (region color, 50% opacity): committed to
            negative VMI.
          * sig-both (region color with diagonal hatch + black edge):
            cross-session sign-flippers whose flip was significant
            in both directions. The hatch + edge make this rare tier
            visually prominent — it's the headline of the figure.

        The `sig-both` fraction is annotated as text directly above
        each bar so it can be read off without measuring stack
        heights. Inputs come from
        `_collect_vmi_sign_flip_tally` with `n_tested_min` (default
        2 — same as figures 3 and 4 elsewhere).

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` embedded in the triage pickle.
        n_tested_min (int)
            Minimum valid sessions for inclusion.
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        tally = self._collect_vmi_sign_flip_tally(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            n_tested_min=n_tested_min,
        )
        region_colors = self._resolve_region_colors()
        non_sig_color = self.visualizations_parameter_dict["unassigned_colors"][0]

        # Read the per-session significance threshold from the same
        # pickle the tally was derived from, for the caption.
        with open(pathlib.Path(triage_pkl_path), "rb") as fh:
            alpha_disp = float(pickle.load(fh)["thresholds_used"]["vmi_alpha"])

        bar_x = np.arange(len(VMI_REGION_ORDER))
        n_per_region = [tally[g]["n_total"] for g in VMI_REGION_ORDER]
        frac_never = np.zeros(len(VMI_REGION_ORDER))
        frac_pos = np.zeros(len(VMI_REGION_ORDER))
        frac_neg = np.zeros(len(VMI_REGION_ORDER))
        frac_both = np.zeros(len(VMI_REGION_ORDER))
        for i, region in enumerate(VMI_REGION_ORDER):
            n = tally[region]["n_total"]
            if n == 0:
                continue
            frac_never[i] = tally[region]["never_sig"] / n
            frac_pos[i] = tally[region]["sig_pos_only"] / n
            frac_neg[i] = tally[region]["sig_neg_only"] / n
            frac_both[i] = tally[region]["sig_both"] / n

        bar_colors = [region_colors[g] for g in VMI_REGION_ORDER]
        never_face = mcolors.to_rgba(non_sig_color, 0.7)
        pos_face = [mcolors.to_rgba(c, 0.95) for c in bar_colors]
        neg_face = [mcolors.to_rgba(c, 0.50) for c in bar_colors]
        # `sig-both` is white-faced + hatched so it reads as a
        # distinct visual category rather than yet-another shade of
        # the region color. The hatch is drawn in black by the
        # default matplotlib hatch color, making the segment stand
        # out against the solid-fill tiers below it.
        both_face_color = "#FFFFFF"

        fig, ax = plt.subplots(figsize=(10.0, 5.4), dpi=150)
        bar_kwargs = dict(width=0.78, edgecolor=COLOR_BLACK, linewidth=0.4)

        # Stack order bottom to top: never-sig, sig+ only, sig− only, sig-both.
        ax.bar(
            bar_x, frac_never,
            color=[never_face] * len(VMI_REGION_ORDER),
            **bar_kwargs,
        )
        ax.bar(
            bar_x, frac_pos,
            bottom=frac_never,
            color=pos_face,
            **bar_kwargs,
        )
        ax.bar(
            bar_x, frac_neg,
            bottom=frac_never + frac_pos,
            color=neg_face,
            **bar_kwargs,
        )
        # `sig-both` is the rare tier — white face + diagonal hatch
        # so it reads as a distinct visual category, not a different
        # shade of the region color.
        ax.bar(
            bar_x, frac_both,
            bottom=frac_never + frac_pos + frac_neg,
            color=both_face_color,
            hatch="///",
            width=0.78, edgecolor=COLOR_BLACK, linewidth=0.9,
        )

        # Annotate sig-both fraction above each bar so it's directly readable.
        for i, region in enumerate(VMI_REGION_ORDER):
            if n_per_region[i] == 0:
                continue
            ax.text(
                bar_x[i], 1.015,
                f"{100 * frac_both[i]:.1f}%",
                ha="center", va="bottom", fontsize=8, color=COLOR_BLACK,
            )

        ax.set_xticks(bar_x)
        ax.set_xticklabels(
            [f"{r}\nN={n}" for r, n in zip(VMI_REGION_ORDER, n_per_region)],
            fontsize=9,
        )
        ax.set_ylim(0.0, 1.08)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("fraction of units", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)

        # Legend strip across the bottom. Squares use a neutral black
        # fill so the convention reads independent of region; the
        # sig-both swatch is a hatched white rectangle (matplotlib
        # `Patch`) so it matches the bar style exactly.

        leg_handles = [
            plt.Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=10,
                label="sig +VMI only",
            ),
            plt.Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=10, alpha=0.50,
                label="sig −VMI only",
            ),
            Patch(
                facecolor="#FFFFFF", edgecolor=COLOR_BLACK,
                linewidth=0.9, hatch="///",
                label="sig in both directions (cross-session sign flip)",
            ),
            plt.Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor=non_sig_color, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=10, alpha=0.7,
                label="never significant",
            ),
        ]
        fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
        fig.legend(
            handles=leg_handles, loc="lower center", ncol=4,
            fontsize=9, frameon=False,
            bbox_to_anchor=(0.5, 0.075),
        )
        fig.text(
            0.5, 0.02,
            f"good + somatic, $n_{{\\mathrm{{tested}}}}\\geq{n_tested_min}$  ·  "
            f"per-session $\\alpha={alpha_disp:.2f}$  ·  "
            "annotations above bars = sig-both fraction",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem="vmi_sign_flip_summary",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_pag_unit_positions(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
    ) -> list[dict]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic PAG
        unit, pull its best-session signed VMI (the per-session entry
        with the largest `|VMI|` across both modality directions,
        gated by the pickle's `vmi_min_bouts` floor and `vmi_alpha`)
        together with its Allen-CCF anatomical position (`loc_ap`,
        `loc_ml`, `loc_dv`, in µm) from the catalog. Units missing
        any of the three coordinates are dropped silently — the
        anatomical figure has no use for them.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` field embedded in the triage pickle.

        Returns
        -------
        per_unit (list[dict])
            One entry per PAG unit; keys: `loc_ap`, `loc_ml`,
            `loc_dv`, `vmi`, `significant`.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        min_bouts = int(triage["thresholds_used"]["vmi_min_bouts"])
        per_unit: list[dict] = []

        for u in triage["units"].values():
            if u["anatomy_region"] != "PAG":
                continue
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            # Anatomical coords. Drop units missing any of the three —
            # plotting them as zeros would put a spurious dense
            # cluster at the origin.
            try:
                loc_ap = float(cat_row["loc_ap"])
                loc_ml = float(cat_row["loc_ml"])
                loc_dv = float(cat_row["loc_dv"])
            except (KeyError, ValueError, TypeError):
                continue

            best_entry = None
            for cond in u["conditions"].values():
                modalities = cond["modalities"]
                for direction in ("vmi_self_excit", "vmi_self_suppress"):
                    if direction not in modalities:
                        continue
                    for entry in modalities[direction]["per_session"]:
                        if entry["n_bouts"] < min_bouts:
                            continue
                        if entry["p"] is None or entry["vmi"] is None:
                            continue
                        vmi = float(entry["vmi"])
                        if (best_entry is None) or (abs(vmi) > abs(best_entry["vmi"])):
                            best_entry = {
                                "vmi":         vmi,
                                "significant": bool(entry["significant"]) and float(entry["p"]) < alpha,
                            }
            if best_entry is None:
                continue

            per_unit.append({
                "loc_ap":      loc_ap,
                "loc_ml":      loc_ml,
                "loc_dv":      loc_dv,
                "vmi":         best_entry["vmi"],
                "significant": best_entry["significant"],
            })

        return per_unit

    def make_pag_anatomical_gradient_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            kde_grid_resolution: int = 220,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the PAG anatomical-gradient diagnostic on the sagittal
        projection (`loc_ap` x `loc_dv`) across three panels:

          1. Per-unit scatter — every good + somatic PAG unit colored
             by best-session signed VMI (sig +VMI = full opacity PAG
             palette color, sig -VMI = same color at 50 % opacity
             with a black edge, non-sig = unassigned-gray).
          2. 2-D KDE density of the sig +VMI subpopulation, rendered
             with the `figures.cmap` colormap from settings
             (defaults to `inferno`).
          3. 2-D KDE density of the sig -VMI subpopulation, same
             colormap, same axes extent so the two heatmaps and the
             scatter share a coordinate frame.

        All three panels span the same `(loc_ap, loc_dv)` range
        (computed once from the full unit set), so spatial clustering
        of either polarity can be compared directly against the
        whole-population scatter.

        Output goes through `figure_io.save_figure` so the directory,
        format, and dpi default to `visualizations_settings.json`.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` embedded in the triage pickle.
        kde_grid_resolution (int)
            Number of evaluation steps along each axis when
            rasterising the 2-D KDE. Higher gives a smoother heatmap
            at more compute cost. Defaults to 220.
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_unit = self._collect_pag_unit_positions(triage_pkl_path, catalog_csv_path)
        region_colors = self._resolve_region_colors()
        pag_color = region_colors["PAG"]
        non_sig_color = self.visualizations_parameter_dict["unassigned_colors"][0]
        # Build a sequential white -> PAG colormap for the two
        # fraction panels: at zero data the pixel renders as white,
        # which matches the figure background, so the alpha-fade at
        # low occupancy blends in seamlessly. PAG-coloured pixels
        # mark high local fractions.
        pag_rgb = tuple(int(pag_color[1 + 2 * k : 3 + 2 * k], 16) for k in range(3))
        density_cmap = create_colormap(input_parameter_dict={
            "cm_length":           255,
            "cm_name":             "pag_fraction_cm",
            "cm_type":             "sequential",
            "cm_start":            pag_rgb,
            "cm_end":              (255, 255, 255),
            "equalize_luminance":  True,
            "match_luminance_by":  "max",
            "change_saturation":   1.0,
            "cm_opacity":          1,
        })
        diff_cmap = plt.get_cmap("RdBu_r").copy()
        diff_cmap.set_bad(color=COLOR_BLACK)

        loc_ap = np.array([u["loc_ap"] for u in per_unit])
        loc_dv = np.array([u["loc_dv"] for u in per_unit])
        vmi = np.array([u["vmi"] for u in per_unit])
        sig = np.array([u["significant"] for u in per_unit])
        pos_sig = sig & (vmi > 0)
        neg_sig = sig & (vmi < 0)
        ns_mask = ~sig
        n_total = len(per_unit)
        n_pos = int(pos_sig.sum())
        n_neg = int(neg_sig.sum())

        # Shared sagittal extent (with a small padding) so the three
        # panels' coordinate frames line up exactly. The pad also
        # gives the KDE bandwidth a bit of room near the edges.
        ap_lo, ap_hi = float(loc_ap.min()), float(loc_ap.max())
        dv_lo, dv_hi = float(loc_dv.min()), float(loc_dv.max())
        ap_pad = 0.04 * (ap_hi - ap_lo)
        dv_pad = 0.04 * (dv_hi - dv_lo)
        ap_lo -= ap_pad; ap_hi += ap_pad
        dv_lo -= dv_pad; dv_hi += dv_pad

        # KDE grid shared between the sig+ and sig- panels so the two
        # heatmaps render at identical pixel positions.
        ap_grid = np.linspace(ap_lo, ap_hi, kde_grid_resolution)
        dv_grid = np.linspace(dv_lo, dv_hi, kde_grid_resolution)
        ap_mesh, dv_mesh = np.meshgrid(ap_grid, dv_grid)
        eval_points = np.vstack([ap_mesh.ravel(), dv_mesh.ravel()])

        def _kde_density(mask: np.ndarray) -> np.ndarray:
            """
            Description
            -----------
            Build a Gaussian KDE from the masked subset of the
            sagittal positions and evaluate it on the shared
            `(ap_mesh, dv_mesh)` grid. Returns an all-zero grid when
            the masked subset is too small to fit a KDE (< 3 points).

            Parameters
            ----------
            mask (np.ndarray)
                Boolean mask selecting which units to feed into the
                KDE.

            Returns
            -------
            density (np.ndarray)
                The evaluated density on the shared grid.
            """
            if int(mask.sum()) < 3:
                return np.zeros_like(ap_mesh)
            kde = gaussian_kde(np.vstack([loc_ap[mask], loc_dv[mask]]))
            return kde(eval_points).reshape(ap_mesh.shape)

        density_pos = _kde_density(pos_sig)
        density_neg = _kde_density(neg_sig)

        # Occupancy: KDE of every good + somatic PAG unit (i.e. where
        # we recorded). The sig +VMI and sig -VMI density panels are
        # divided by this to express the local fraction of cells at
        # each anatomical location that fall into each polarity,
        # removing the sampling-bias confound where regions that were
        # recorded more heavily look "denser" for every cell type.
        density_all = _kde_density(np.ones(n_total, dtype=bool))

        # KDE units integrate to 1; multiply by N to recover an
        # expected-count density. Then fraction = expected_count_subset
        # / expected_count_total per pixel. We do NOT hard-mask
        # low-occupancy pixels — instead, the per-pixel alpha used in
        # imshow scales smoothly with `density_all`, so the gradient
        # stays continuous across the cell and gradually fades out
        # where the population was poorly sampled.
        all_max = float(density_all.max()) if density_all.size else 0.0
        # A tiny epsilon avoids division by exact zero at corners
        # where the KDE evaluates to numerical zero; values there
        # will be rendered with alpha≈0 anyway.
        eps = 1e-12 * max(all_max, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_pos = (n_pos * density_pos) / (n_total * (density_all + eps))
            frac_neg = (n_neg * density_neg) / (n_total * (density_all + eps))
        frac_pos = np.nan_to_num(frac_pos, nan=0.0, posinf=0.0, neginf=0.0)
        frac_neg = np.nan_to_num(frac_neg, nan=0.0, posinf=0.0, neginf=0.0)
        frac_diff = frac_pos - frac_neg

        # Smooth occupancy-driven alpha: full opacity in the densest
        # 80 % of the cell, fades linearly to 0 below `alpha_floor`.
        alpha_floor = 0.20 * all_max if all_max > 0.0 else 0.0
        alpha_mask = (
            np.clip(density_all / alpha_floor, 0.0, 1.0)
            if alpha_floor > 0.0 else np.ones_like(density_all)
        )

        # Shared color scale across the two fraction panels, derived
        # from the well-sampled core (where `alpha_mask >= 0.5`) so a
        # handful of peripheral pixels with tiny denominators don't
        # blow up `vmax`. `vmin=0` because fractions are non-negative.
        core = alpha_mask >= 0.5
        frac_vmax = float(
            max(
                frac_pos[core].max() if core.any() else frac_pos.max(),
                frac_neg[core].max() if core.any() else frac_neg.max(),
            )
        )
        if frac_vmax <= 0.0:
            frac_vmax = 1.0
        # Symmetric, divergent scale for the difference panel, also
        # taken from the well-sampled core.
        diff_abs_max = float(
            np.abs(frac_diff[core]).max() if core.any() else np.abs(frac_diff).max()
        )
        if diff_abs_max <= 0.0:
            diff_abs_max = frac_vmax

        # Figure layout: four equal-aspect sagittal panels sharing
        # axis extent (scatter + sig+ occupancy-normalized density +
        # sig- occupancy-normalized density + sig+ minus sig- diff).
        fig = plt.figure(figsize=(18.0, 4.8), dpi=150)
        gs = gridspec.GridSpec(
            1, 4,
            figure=fig,
            wspace=0.34,
            left=0.045, right=0.99,
            top=0.93, bottom=0.20,
        )

        # Panel 1 — per-unit scatter.
        ax_sc = fig.add_subplot(gs[0, 0])
        ax_sc.scatter(
            loc_ap[ns_mask], loc_dv[ns_mask],
            s=8, c=non_sig_color, alpha=0.45,
            edgecolors="none", rasterized=True,
        )
        ax_sc.scatter(
            loc_ap[neg_sig], loc_dv[neg_sig],
            s=14, c=pag_color, alpha=0.50,
            edgecolors=COLOR_BLACK, linewidths=0.4,
            rasterized=True,
        )
        ax_sc.scatter(
            loc_ap[pos_sig], loc_dv[pos_sig],
            s=14, c=pag_color, alpha=0.95,
            edgecolors=COLOR_BLACK, linewidths=0.4,
            rasterized=True,
        )
        ax_sc.set_xlim(ap_lo, ap_hi)
        ax_sc.set_ylim(dv_lo, dv_hi)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.set_xlabel(r"loc_ap (µm, caudal to rostral)", fontsize=9)
        ax_sc.set_ylabel(r"loc_dv (µm, ventral to dorsal)", fontsize=9)
        ax_sc.set_title("sagittal scatter", fontsize=11)
        ax_sc.tick_params(labelsize=8)

        # Panel 2 — occupancy-normalized sig +VMI fraction.
        ax_pos = fig.add_subplot(gs[0, 1])
        im_pos = ax_pos.imshow(
            frac_pos,
            extent=(ap_lo, ap_hi, dv_lo, dv_hi),
            origin="lower",
            aspect="auto",
            cmap=density_cmap,
            vmin=0.0, vmax=frac_vmax,
            alpha=alpha_mask,
        )
        ax_pos.set_xlim(ap_lo, ap_hi)
        ax_pos.set_ylim(dv_lo, dv_hi)
        ax_pos.set_aspect("equal", adjustable="box")
        ax_pos.set_xlabel(r"loc_ap (µm, caudal to rostral)", fontsize=9)
        ax_pos.set_ylabel(r"loc_dv (µm, ventral to dorsal)", fontsize=9)
        ax_pos.set_title(f"sig +VMI fraction  (N+={n_pos})", fontsize=11)
        ax_pos.tick_params(labelsize=8)
        cbar_pos = fig.colorbar(im_pos, ax=ax_pos, fraction=0.046, pad=0.04)
        cbar_pos.ax.tick_params(labelsize=7)
        cbar_pos.set_label("fraction of recorded cells", fontsize=8)

        # Panel 3 — occupancy-normalized sig -VMI fraction.
        ax_neg = fig.add_subplot(gs[0, 2])
        im_neg = ax_neg.imshow(
            frac_neg,
            extent=(ap_lo, ap_hi, dv_lo, dv_hi),
            origin="lower",
            aspect="auto",
            cmap=density_cmap,
            vmin=0.0, vmax=frac_vmax,
            alpha=alpha_mask,
        )
        ax_neg.set_xlim(ap_lo, ap_hi)
        ax_neg.set_ylim(dv_lo, dv_hi)
        ax_neg.set_aspect("equal", adjustable="box")
        ax_neg.set_xlabel(r"loc_ap (µm, caudal to rostral)", fontsize=9)
        ax_neg.set_ylabel(r"loc_dv (µm, ventral to dorsal)", fontsize=9)
        ax_neg.set_title(f"sig -VMI fraction  (N-={n_neg})", fontsize=11)
        ax_neg.tick_params(labelsize=8)
        cbar_neg = fig.colorbar(im_neg, ax=ax_neg, fraction=0.046, pad=0.04)
        cbar_neg.ax.tick_params(labelsize=7)
        cbar_neg.set_label("fraction of recorded cells", fontsize=8)

        # Panel 4 — sig +VMI - sig -VMI fraction difference, divergent.
        ax_diff = fig.add_subplot(gs[0, 3])
        im_diff = ax_diff.imshow(
            frac_diff,
            extent=(ap_lo, ap_hi, dv_lo, dv_hi),
            origin="lower",
            aspect="auto",
            cmap=diff_cmap,
            vmin=-diff_abs_max, vmax=diff_abs_max,
            alpha=alpha_mask,
        )
        ax_diff.set_xlim(ap_lo, ap_hi)
        ax_diff.set_ylim(dv_lo, dv_hi)
        ax_diff.set_aspect("equal", adjustable="box")
        ax_diff.set_xlabel(r"loc_ap (µm, caudal to rostral)", fontsize=9)
        ax_diff.set_ylabel(r"loc_dv (µm, ventral to dorsal)", fontsize=9)
        ax_diff.set_title("sig +VMI − sig −VMI fraction", fontsize=11)
        ax_diff.tick_params(labelsize=8)
        cbar_diff = fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar_diff.ax.tick_params(labelsize=7)
        cbar_diff.set_label("Δ fraction (+VMI − −VMI)", fontsize=8)

        # Single legend strip + caption at the bottom.
        leg_handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=pag_color, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8,
                label=f"sig +VMI  (N+={n_pos})",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=pag_color, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=8, alpha=0.50,
                label=f"sig -VMI  (N-={n_neg})",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=non_sig_color, markeredgecolor="none",
                markersize=8, alpha=0.7,
                label="non-significant",
            ),
        ]
        fig.legend(
            handles=leg_handles, loc="lower center", ncol=3,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 0.05),
        )
        fig.text(
            0.5, 0.01,
            f"PAG good + somatic  ·  N={n_total}  ·  "
            "best-session signed VMI per unit",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem="vmi_pag_anatomical_gradient",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_vmi_distribution_per_unit(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and assign ONE signed VMI value
        per good + somatic unit via a hybrid metric that lets the
        downstream histograms satisfy all three intuitive constraints
        at once:

          * Non-significant units use the median of their per-session
            signed VMIs (so the bulk of un-tuned units pile up near
            zero, giving the gray background its natural peak at 0).
          * Sig +VMI-only units use the maximum of their significant
            +VMI sessions (always positive, so the sig+ overlay lives
            strictly in positive bins).
          * Sig −VMI-only units use the minimum of their significant
            −VMI sessions (always negative).
          * Sig-both units (the cross-session sign-flippers) collapse
            onto whichever direction had the larger absolute effect;
            they live in only one overlay so the per-unit counts add
            up.

        Each region's entry holds three parallel lists of signed
        per-unit VMIs:

          * `all` — every good + somatic unit.
          * `sig_pos` — units whose assigned value is positive (sig+
            only, plus the sig-both units whose +VMI max dominated).
          * `sig_neg` — units whose assigned value is negative.

        `sig_pos` and `sig_neg` are strict subsets of `all` (same
        per-unit value) so overlay bins always sit at the same x as
        the background's bin for that unit.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` field embedded in the triage pickle.

        Returns
        -------
        per_group (dict[str, dict[str, list[float]]])
            Mapping from canonical region label to `{all, sig_pos,
            sig_neg}` value lists.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        alpha = float(triage["thresholds_used"]["vmi_alpha"])
        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, dict[str, list[float]]] = {
            g: {"all": [], "sig_pos": [], "sig_neg": []}
            for g in VMI_REGION_ORDER
        }

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"
            bucket = per_group[group]

            vmi_values: list[float] = []
            sig_pos_vmis: list[float] = []
            sig_neg_vmis: list[float] = []
            for cond in u["conditions"].values():
                modalities = cond["modalities"]
                for direction in ("vmi_self_excit", "vmi_self_suppress"):
                    if direction not in modalities:
                        continue
                    for entry in modalities[direction]["per_session"]:
                        if entry["vmi"] is None:
                            continue
                        vmi = float(entry["vmi"])
                        vmi_values.append(vmi)
                        p_val = entry["p"]
                        if p_val is None:
                            continue
                        if not (bool(entry["significant"]) and float(p_val) < alpha):
                            continue
                        if vmi > 0:
                            sig_pos_vmis.append(vmi)
                        elif vmi < 0:
                            sig_neg_vmis.append(vmi)
            if not vmi_values:
                continue

            if sig_pos_vmis and sig_neg_vmis:
                # Sig-both: assign to whichever direction had the
                # larger absolute effect, so the unit contributes to
                # exactly one overlay and the per-unit counts stay
                # additive.
                pos_max = max(sig_pos_vmis)
                neg_min = min(sig_neg_vmis)
                if pos_max >= abs(neg_min):
                    per_unit_value = pos_max
                    overlay = "sig_pos"
                else:
                    per_unit_value = neg_min
                    overlay = "sig_neg"
            elif sig_pos_vmis:
                per_unit_value = max(sig_pos_vmis)
                overlay = "sig_pos"
            elif sig_neg_vmis:
                per_unit_value = min(sig_neg_vmis)
                overlay = "sig_neg"
            else:
                per_unit_value = float(np.median(vmi_values))
                overlay = None

            bucket["all"].append(per_unit_value)
            if overlay == "sig_pos":
                bucket["sig_pos"].append(per_unit_value)
            elif overlay == "sig_neg":
                bucket["sig_neg"].append(per_unit_value)

        return per_group

    def make_vmi_distribution_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            n_bins: int = 20,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render per-region histograms of best-session signed VMI for
        every good + somatic unit, with significant +VMI and −VMI
        sub-distributions overlaid on top of the full-population
        background. Layout matches the 2×4 grid convention from
        figs 1–5:

          * Seven per-region panels — one per brain-area group. Each
            panel shows: (a) a light-gray step-filled histogram of
            every unit in that region (background); (b) the sig +VMI
            subset overlaid in the region's palette color at full
            opacity; (c) the sig −VMI subset overlaid in the same
            color at 50 % opacity.
          * 8th panel — overlaid ECDFs of signed VMI per region, one
            line per region in its palette color, for direct
            comparison of skew across regions.

        Bins span [−1, 1] uniformly. One signed VMI value per unit (good
        + somatic) via `_collect_vmi_distribution_per_unit`: non-sig units
        use their per-session median, sig +VMI-only units the maximum of
        their sig +VMI sessions, sig −VMI-only units the minimum of their
        sig −VMI sessions, and sig-both units collapse onto the direction
        with the larger absolute effect. No `vmi_min_bouts` gating is
        applied here.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` field embedded in the triage pickle.
        n_bins (int)
            Number of histogram bins between VMI=-1 and VMI=+1. The
            default (20) matches the bin count used in the project's
            legacy single-panel sketch.
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_group = self._collect_vmi_distribution_per_unit(
            triage_pkl_path, catalog_csv_path,
        )
        region_colors = self._resolve_region_colors()
        bg_color = COLOR_LIGHT

        fig = plt.figure(figsize=(14.0, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.32,
            wspace=0.25,
            left=0.06, right=0.985,
            top=0.965, bottom=0.13,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
        bins = np.linspace(-1.0, 1.0, n_bins + 1)

        # Track the global histogram y-max so the seven scatter
        # panels share a y-axis — directly comparable across regions.
        y_max_per_panel: list[int] = []

        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            buckets = per_group[region]
            region_color = region_colors[region]

            if not buckets["all"]:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            vmi_all = np.array(buckets["all"])
            vmi_pos = np.array(buckets["sig_pos"])
            vmi_neg = np.array(buckets["sig_neg"])

            # Background — entire region population.
            counts_bg, _, _ = ax.hist(
                vmi_all, bins=bins,
                color=bg_color, alpha=0.35,
                histtype="stepfilled", edgecolor=COLOR_BLACK, linewidth=0.5,
            )
            # Sig −VMI overlay (50 % opacity of region color).
            if vmi_neg.size:
                ax.hist(
                    vmi_neg, bins=bins,
                    color=region_color, alpha=0.50,
                    histtype="stepfilled",
                    edgecolor=COLOR_BLACK, linewidth=0.4,
                )
            # Sig +VMI overlay (full region color).
            if vmi_pos.size:
                ax.hist(
                    vmi_pos, bins=bins,
                    color=region_color, alpha=0.95,
                    histtype="stepfilled",
                    edgecolor=COLOR_BLACK, linewidth=0.4,
                )
            y_max_per_panel.append(int(np.max(counts_bg)) if counts_bg.size else 0)

            ax.axvline(0.0, color=COLOR_BLACK, linewidth=0.5, linestyle=":")
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_title(f"{region}  (N={vmi_all.size})", fontsize=10)
            if idx[0] == 1:
                ax.set_xlabel("VMI (per-unit)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if idx[1] == 0:
                ax.set_ylabel("unit count", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.text(
                0.03, 0.97,
                f"N+={int(vmi_pos.size)}\nN-={int(vmi_neg.size)}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=COLOR_HATCH, alpha=0.85),
            )

        # Aggregate ECDF panel (8th cell): one line per region.
        ax_agg = fig.add_subplot(gs[1, 3])
        for region in VMI_REGION_ORDER:
            all_vmis = per_group[region]["all"]
            if not all_vmis:
                continue
            vmi_sorted = np.sort(np.array(all_vmis))
            ecdf_y = np.arange(1, vmi_sorted.size + 1) / vmi_sorted.size
            ax_agg.plot(
                vmi_sorted, ecdf_y,
                color=region_colors[region], linewidth=1.4,
                label=f"{region} (n={vmi_sorted.size})",
            )
        ax_agg.axvline(0.0, color=COLOR_BLACK, linewidth=0.5, linestyle=":")
        ax_agg.set_xlim(-1.05, 1.05)
        ax_agg.set_ylim(0.0, 1.02)
        ax_agg.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_agg.set_xlabel("VMI (per-unit)", fontsize=9)
        ax_agg.set_ylabel("ECDF", fontsize=9)
        ax_agg.set_title("signed VMI distribution across regions", fontsize=10)
        ax_agg.tick_params(labelsize=8)
        ax_agg.legend(fontsize=7, frameon=False, loc="upper left")

        # Horizontal legend strip + caption at the bottom.
        leg_handles = [
            plt.Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=10,
                label="sig +VMI",
            ),
            plt.Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor=COLOR_BLACK, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.4, markersize=10, alpha=0.50,
                label="sig -VMI",
            ),
            plt.Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor=bg_color, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.5, markersize=10, alpha=0.35,
                label="all units (background)",
            ),
        ]
        fig.legend(
            handles=leg_handles, loc="lower center", ncol=3,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 0.03),
        )
        fig.text(
            0.5, 0.005,
            "good + somatic  ·  one data point per unit  ·  "
            "non-sig: per-unit median; sig+: max sig+ session; sig-: min sig- session  "
            "(sig-both assigned to dominant direction)",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem="vmi_distribution",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_consistent_peth(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            direction: str = "excit",
            tol_s: float = 0.100,
            k_min: int = 2,
            require_majority: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit
        with at least two significant `usv_peth_self_{direction}` sessions
        whose `peak_t` values cluster within `tol_s` (largest
        in-tolerance subset of size `k_min` or more, optionally
        accounting for at least 50 % of the unit's sig sessions),
        return one summary per consistent unit:

          * `median_peak_t` — median of `peak_t` across the unit's
            sig sessions (the unit's representative response time).
          * `median_peak_z` — median of `peak_z` across the unit's
            sig sessions (the unit's representative response
            magnitude).
          * `n_sig` — number of significant sessions.
          * `k` — size of the largest in-tolerance peak_t cluster.

        The `direction` argument selects the modality scanned
        (`usv_peth_self_excit` or `usv_peth_self_suppress`). The
        consistency rule is the package agreed on for PETH:
        `±tol_s/2` window AND `k >= k_min` AND (optional) majority.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` field embedded in the triage pickle.
        direction (str)
            Which PETH modality to scan: `'excit'` or `'suppress'`
            (validated; selects `usv_peth_self_{direction}`). Defaults
            to `'excit'`.
        tol_s (float)
            Full-width tolerance for the largest-in-tolerance subset
            (so a 100 ms window = ±50 ms either side of the cluster
            centre). Defaults to 0.100 s.
        k_min (int)
            Minimum number of sig sessions that must agree within
            `tol_s` for the unit to count as consistent.
        require_majority (bool)
            When True, additionally requires `k / n_sig >= 0.5` —
            protects against units with many sig sessions where only
            `k_min` happen to align.

        Returns
        -------
        per_group (dict[str, list[dict]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a list of per-unit summary dicts
            for units that pass the consistency rule.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, list[dict]] = {g: [] for g in VMI_REGION_ORDER}

        if direction not in ("excit", "suppress"):
            raise ValueError(
                f"direction must be 'excit' or 'suppress'; got {direction!r}"
            )
        modality_key = f"usv_peth_self_{direction}"

        def _largest_in_tol(values: list[float]) -> int:
            """
            Description
            -----------
            Return the size of the largest subset of `values` whose
            max-min spread does not exceed `tol_s`. Sliding-window
            on the sorted values.

            Parameters
            ----------
            values (list[float])
                Per-session peak_t values for the unit.

            Returns
            -------
            k (int)
                Largest in-tolerance subset size.
            """
            if not values:
                return 0
            vs = sorted(values)
            best_k = 1
            lo = 0
            for hi in range(len(vs)):
                while vs[hi] - vs[lo] > tol_s:
                    lo += 1
                if hi - lo + 1 > best_k:
                    best_k = hi - lo + 1
            return best_k

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            pks: list[float] = []
            pzs: list[float] = []
            for cond in u["conditions"].values():
                m = cond["modalities"].get(modality_key)
                if m is None:
                    continue
                for e in m["per_session"]:
                    if not e["significant"]:
                        continue
                    if e["peak_t"] is None or e["peak_z"] is None:
                        continue
                    pks.append(float(e["peak_t"]))
                    pzs.append(float(e["peak_z"]))
            n_sig = len(pks)
            if n_sig < 2:
                continue
            k = _largest_in_tol(pks)
            if k < k_min:
                continue
            if require_majority and (k / n_sig) < 0.5:
                continue

            per_group[group].append({
                "n_sig":         n_sig,
                "k":             k,
                "median_peak_t": float(np.median(pks)),
                "median_peak_z": float(np.median(pzs)),
            })

        return per_group

    def make_peth_timing_distribution_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            direction: str = "excit",
            tol_s: float = 0.100,
            k_min: int = 2,
            require_majority: bool = True,
            n_bins: int = 40,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the per-region distribution of consistent excit-PETH
        anticipatory response timing. Layout is a 2×7 grid with one
        column per brain-area group:

          * Top row — histogram of each region's consistent units'
            median `peak_t` across [−2, 0] s.
          * Bottom row — scatter of median `peak_t` (x) against
            median `peak_z` (y) per consistent unit.

        Consistency filter is the one agreed during the PETH design
        session: per-unit `peak_t` values across significant excit
        sessions must contain a `k_min`-or-larger subset whose
        max-min spread is no greater than `tol_s` (defaults to 100
        ms full width, k_min=2), with an optional majority gate
        (`require_majority=True` by default).

        Per-unit anchor times use the median of `peak_t` across all
        significant sessions (not just the in-tolerance cluster), so
        the figure mirrors the median-aggregation convention used
        elsewhere in the population-VMI suite.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` embedded in the triage pickle.
        tol_s (float)
            Full-width tolerance for the consistency check (default
            0.100 s).
        k_min (int)
            Minimum in-tolerance subset size (default 2).
        require_majority (bool)
            Whether to also require the in-tolerance subset to
            account for at least half of the unit's sig sessions
            (default True).
        n_bins (int)
            Number of histogram bins between -2 and 0 s. Default 40
            (matches the per-session PETH bin width).
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        if direction not in ("excit", "suppress"):
            raise ValueError(
                f"direction must be 'excit' or 'suppress'; got {direction!r}"
            )
        per_group = self._collect_consistent_peth(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            direction=direction,
            tol_s=tol_s,
            k_min=k_min,
            require_majority=require_majority,
        )
        region_colors = self._resolve_region_colors()

        fig = plt.figure(figsize=(18.0, 6.6), dpi=150)
        gs = gridspec.GridSpec(
            2, len(VMI_REGION_ORDER),
            figure=fig,
            hspace=0.32,
            wspace=0.32,
            left=0.045, right=0.99,
            top=0.94, bottom=0.13,
        )
        # Histograms keep a linear x-axis (signed peak_t, USV-onset on
        # the right). Only the scatter row uses a log axis on
        # |peak_t| to expand the dense near-onset region — the long
        # pre-onset tail collapses into few units that are easier to
        # read on a log scale.
        LINEAR_XLIM = (-2.0, 0.05)
        linear_bins = np.linspace(-2.0, 0.0, n_bins + 1)
        LOG_XLIM_LOW_S = 0.020   # rightmost edge on |peak_t|
        LOG_XLIM_HIGH_S = 2.10   # leftmost edge on |peak_t|
        log_ticks = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
        log_tick_labels = ["25 ms", "50 ms", "100 ms", "250 ms",
                            "500 ms", "1 s", "2 s"]
        peak_z_lo = float("inf")
        peak_z_hi = float("-inf")
        for region in VMI_REGION_ORDER:
            for u in per_group[region]:
                peak_z_lo = min(peak_z_lo, u["median_peak_z"])
                peak_z_hi = max(peak_z_hi, u["median_peak_z"])
        if not np.isfinite(peak_z_lo):
            peak_z_lo, peak_z_hi = 0.0, 1.0
        peak_z_pad = 0.05 * (peak_z_hi - peak_z_lo)
        peak_z_lo -= peak_z_pad
        peak_z_hi += peak_z_pad

        for col, region in enumerate(VMI_REGION_ORDER):
            region_color = region_colors[region]
            units = per_group[region]
            n_units = len(units)

            # Top row — histogram of signed median peak_t (linear).
            ax_hist = fig.add_subplot(gs[0, col])
            if n_units:
                pks = np.array([u["median_peak_t"] for u in units])
                ax_hist.hist(
                    pks, bins=linear_bins,
                    color=region_color, alpha=0.95,
                    histtype="stepfilled",
                    edgecolor=COLOR_BLACK, linewidth=0.4,
                )
            ax_hist.axvline(0.0, color=COLOR_BLACK, linewidth=0.5, linestyle=":")
            ax_hist.set_xlim(*LINEAR_XLIM)
            ax_hist.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0])
            ax_hist.set_xlabel("peak_t (s, pre-USV)", fontsize=9)
            ax_hist.set_title(f"{region}  (N={n_units})", fontsize=10)
            ax_hist.tick_params(labelsize=8)
            if col == 0:
                ax_hist.set_ylabel("unit count", fontsize=9)

            # Bottom row — scatter of |median peak_t| × median peak_z.
            ax_sc = fig.add_subplot(gs[1, col])
            if n_units:
                abs_pks = np.array([abs(u["median_peak_t"]) for u in units])
                pzs = np.array([u["median_peak_z"] for u in units])
                ax_sc.scatter(
                    abs_pks, pzs,
                    s=18, c=region_color, alpha=0.90,
                    edgecolors=COLOR_BLACK, linewidths=0.4,
                    rasterized=True,
                )
            ax_sc.set_xscale("log")
            ax_sc.set_xlim(LOG_XLIM_HIGH_S, LOG_XLIM_LOW_S)
            ax_sc.set_xticks(log_ticks)
            ax_sc.set_xticklabels(log_tick_labels, fontsize=7, rotation=35, ha="right")
            ax_sc.set_ylim(peak_z_lo, peak_z_hi)
            ax_sc.set_xlabel("time before USV onset (log)", fontsize=9)
            ax_sc.tick_params(labelsize=8)
            if col == 0:
                ax_sc.set_ylabel("median peak_z", fontsize=9)

        # Bottom caption.
        fig.text(
            0.5, 0.015,
            f"good + somatic, consistent {direction} only  ·  "
            f"consistency = $\\geq${k_min} sig {direction} sessions within "
            f"±{int(1000*tol_s/2)} ms"
            f"{' AND >=50% majority' if require_majority else ''}  ·  "
            f"per-unit anchors = medians across all sig {direction} sessions  ·  "
            "histogram x = signed peak_t (linear); scatter x = |peak_t| (log, USV on the right)",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem=f"peth_{direction}_timing_distribution",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_consistent_property(
            self,
            triage_pkl_path: str | pathlib.Path,
            property_name: str,
            catalog_csv_path: str | pathlib.Path | None = None,
            direction: str = "excit",
            tol: float | None = None,
            k_min: int = 2,
            require_majority: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit
        with at least two significant `usv_property_self_<property>_excit`
        sessions whose `peak_bin_value` values cluster within `tol`
        (largest in-tolerance subset of size `k_min` or more, optionally
        accounting for at least 50 % of the unit's sig sessions),
        return one summary per consistent unit:

          * `median_peak_value` — median of `peak_bin_value` across
            the unit's sig sessions (the unit's representative
            property-space peak).
          * `median_peak_z` — median of `peak_z` across the unit's
            sig sessions (the unit's representative response
            magnitude).
          * `n_sig` — number of significant sessions.
          * `k` — size of the largest in-tolerance peak-value cluster.

        Same rule shape as `_collect_consistent_peth`, with the
        anchor swapped from `peak_t` (time) to `peak_bin_value`
        (property value). `tol` defaults to `USV_PROPERTY_META[property_name]["tol"]`
        when unset.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        property_name (str)
            One of `USV_PROPERTY_ORDER` (e.g. `'duration'`,
            `'mean_freq_hz'`).
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` field embedded in the triage pickle.
        tol (float | None)
            Full-width tolerance applied to the per-session
            `peak_bin_value` values during the consistency check.
            Defaults to `USV_PROPERTY_META[property_name]["tol"]`.
        k_min (int)
            Minimum number of sig sessions that must agree within
            `tol` for the unit to count as consistent.
        require_majority (bool)
            When True, additionally requires `k / n_sig >= 0.5`.

        Returns
        -------
        per_group (dict[str, list[dict]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a list of per-unit summary dicts
            for units that pass the consistency rule.
        """

        if property_name not in USV_PROPERTY_META:
            raise ValueError(
                f"unknown property_name {property_name!r}; "
                f"expected one of {tuple(USV_PROPERTY_META)}"
            )
        if tol is None:
            tol = float(USV_PROPERTY_META[property_name]["tol"])

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, list[dict]] = {g: [] for g in VMI_REGION_ORDER}

        if direction not in ("excit", "suppress"):
            raise ValueError(
                f"direction must be 'excit' or 'suppress'; got {direction!r}"
            )
        modality_key = f"usv_property_self_{property_name}_{direction}"

        def _largest_in_tol(values: list[float]) -> int:
            """
            Description
            -----------
            Sliding-window over sorted values returning the largest
            window of size k such that max(window) - min(window) <= tol.

            Parameters
            ----------
            values (list[float])
                Per-session peak_bin_value entries for the unit.

            Returns
            -------
            k (int)
                Largest in-tolerance subset size.
            """
            if not values:
                return 0
            vs = sorted(values)
            best_k = 1
            lo = 0
            for hi in range(len(vs)):
                while vs[hi] - vs[lo] > tol:
                    lo += 1
                if hi - lo + 1 > best_k:
                    best_k = hi - lo + 1
            return best_k

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            pvs: list[float] = []
            pzs: list[float] = []
            for cond in u["conditions"].values():
                m = cond["modalities"].get(modality_key)
                if m is None:
                    continue
                for e in m["per_session"]:
                    if not e["significant"]:
                        continue
                    if e["peak_bin_value"] is None or e["peak_z"] is None:
                        continue
                    pvs.append(float(e["peak_bin_value"]))
                    pzs.append(float(e["peak_z"]))
            n_sig = len(pvs)
            if n_sig < 2:
                continue
            k = _largest_in_tol(pvs)
            if k < k_min:
                continue
            if require_majority and (k / n_sig) < 0.5:
                continue

            per_group[group].append({
                "n_sig":            n_sig,
                "k":                k,
                "median_peak_value": float(np.median(pvs)),
                "median_peak_z":    float(np.median(pzs)),
            })

        return per_group

    def make_property_tuning_distribution_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            property_name: str,
            catalog_csv_path: str | pathlib.Path | None = None,
            direction: str = "excit",
            tol: float | None = None,
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the per-region distribution of consistent excit-tuned
        units' peak property-value for a single USV acoustic
        property. Layout is the same 2×4 grid used by the VMI
        distribution and PETH timing figures:

          * Seven per-region histograms of `median peak_bin_value`
            (consistent excit units only, region color, stepfilled).
          * 8th panel — overlaid ECDFs of the same per-region pools
            for direct across-region comparison.

        The x-axis uses the property's natural units, with display
        rescaling for Hz-valued properties (e.g. `peak_freq_hz` is
        rendered in kHz). Tolerance defaults to two upstream bin
        widths of that property (`USV_PROPERTY_META[property_name]["tol"]`).

        Histogram bins are built directly from the property's native
        grid (bin_width = `tol / 2`, the upstream property bin width).
        This avoids the empty-gap artefact you get when histogram bin
        edges don't line up with the discrete `peak_bin_value`
        positions — every histogram bar covers exactly one of the
        property's tuning-curve bins.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        property_name (str)
            One of `USV_PROPERTY_ORDER` (e.g. `'duration'`,
            `'mean_freq_hz'`).
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV. Defaults to the
            `catalog_path` embedded in the triage pickle.
        tol (float | None)
            Full-width tolerance for the consistency check (defaults
            to the per-property entry in `USV_PROPERTY_META`).
        k_min (int)
            Minimum in-tolerance subset size (default 2).
        require_majority (bool)
            Whether to also require the in-tolerance subset to
            account for at least half of the unit's sig sessions
            (default True).
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        meta = USV_PROPERTY_META[property_name]
        unit_scale = float(meta["unit_scale"])
        unit_label = str(meta["unit_label"])
        display_name = str(meta["display_name"])

        if direction not in ("excit", "suppress"):
            raise ValueError(
                f"direction must be 'excit' or 'suppress'; got {direction!r}"
            )
        per_group = self._collect_consistent_property(
            triage_pkl_path=triage_pkl_path,
            property_name=property_name,
            catalog_csv_path=catalog_csv_path,
            direction=direction,
            tol=tol,
            k_min=k_min,
            require_majority=require_majority,
        )
        region_colors = self._resolve_region_colors()

        # Shared x-axis range derived from the union of all consistent
        # units' median peak values, padded slightly so the extreme
        # bins aren't visually clipped.
        all_vals: list[float] = []
        for region in VMI_REGION_ORDER:
            for u in per_group[region]:
                all_vals.append(u["median_peak_value"])
        if not all_vals:
            x_lo, x_hi = 0.0, 1.0
        else:
            arr = np.array(all_vals)
            x_lo = float(arr.min())
            x_hi = float(arr.max())
            pad = 0.05 * max(x_hi - x_lo, 1e-12)
            x_lo -= pad
            x_hi += pad
        # Bin width = upstream property bin width = tol/2 (the rule
        # baked into USV_PROPERTY_META). Build histogram edges that
        # straddle every discrete property-bin centre, so each bar
        # covers exactly one tuning-curve bin and the histogram has
        # no spurious empty gaps.
        property_bin_width = (
            tol if tol is not None else float(meta["tol"])
        ) / 2.0
        # Snap the lo / hi to property-bin-centred edges.
        x_lo_snap = (np.floor(x_lo / property_bin_width) - 0.5) * property_bin_width
        x_hi_snap = (np.ceil(x_hi / property_bin_width) + 0.5) * property_bin_width
        n_bin_edges = int(np.round((x_hi_snap - x_lo_snap) / property_bin_width)) + 1
        bins = np.linspace(x_lo_snap, x_hi_snap, n_bin_edges)
        bins_disp = bins * unit_scale
        x_lo_disp = x_lo_snap * unit_scale
        x_hi_disp = x_hi_snap * unit_scale

        fig = plt.figure(figsize=(14.0, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.40,
            wspace=0.30,
            left=0.06, right=0.985,
            top=0.93, bottom=0.13,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

        x_label = (
            f"{display_name} ({unit_label})"
            if unit_label else display_name
        )

        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            units = per_group[region]
            region_color = region_colors[region]

            if not units:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            pks = np.array([u["median_peak_value"] for u in units]) * unit_scale
            ax.hist(
                pks, bins=bins_disp,
                color=region_color, alpha=0.95,
                histtype="stepfilled",
                edgecolor=COLOR_BLACK, linewidth=0.4,
            )
            ax.set_xlim(x_lo_disp, x_hi_disp)
            ax.set_title(f"{region}  (N={len(units)})", fontsize=10)
            ax.tick_params(labelsize=8)
            if idx[0] == 1:
                ax.set_xlabel(x_label, fontsize=9)
            if idx[1] == 0:
                ax.set_ylabel("unit count", fontsize=9)

        # 8th panel — ECDF overlay per region.
        ax_agg = fig.add_subplot(gs[1, 3])
        for region in VMI_REGION_ORDER:
            units = per_group[region]
            if not units:
                continue
            arr_sorted = np.sort(
                np.array([u["median_peak_value"] for u in units]) * unit_scale
            )
            ecdf_y = np.arange(1, arr_sorted.size + 1) / arr_sorted.size
            ax_agg.plot(
                arr_sorted, ecdf_y,
                color=region_colors[region], linewidth=1.4,
                label=f"{region} (N={arr_sorted.size})",
            )
        ax_agg.set_xlim(x_lo_disp, x_hi_disp)
        ax_agg.set_ylim(0.0, 1.02)
        ax_agg.set_xlabel(x_label, fontsize=9)
        ax_agg.set_ylabel("ECDF", fontsize=9)
        ax_agg.set_title(f"{display_name} — across regions", fontsize=10)
        ax_agg.tick_params(labelsize=8)
        ax_agg.legend(fontsize=7, frameon=False, loc="lower right")

        # Bottom caption.
        tol_disp = (tol if tol is not None else float(meta["tol"])) * unit_scale
        unit_tail = f" {unit_label}" if unit_label else ""
        fig.text(
            0.5, 0.01,
            f"good + somatic, consistent {direction} only  ·  "
            f"consistency = $\\geq${k_min} sig {direction} sessions within "
            f"±{tol_disp/2:g}{unit_tail}"
            f"{' AND >=50% majority' if require_majority else ''}  ·  "
            f"per-unit anchor = median peak_bin_value across all sig {direction} sessions",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem=f"property_tuning_{direction}_distribution_{property_name}",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def make_all_property_tuning_distribution_figures(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            direction: str = "excit",
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> list[pathlib.Path]:
        """
        Description
        -----------
        Convenience wrapper that renders the per-region tuning
        distribution figure for every USV acoustic property in
        `USV_PROPERTY_ORDER` in one call. Returns the full list of
        output paths in the same order so the caller can log /
        cross-reference them.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV.
        k_min (int)
            Minimum in-tolerance subset size (default 2).
        require_majority (bool)
            Whether to also require the in-tolerance subset to
            account for at least half of the unit's sig sessions
            (default True).
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_paths (list[pathlib.Path])
            One path per property in `USV_PROPERTY_ORDER`.
        """

        out_paths: list[pathlib.Path] = []
        for property_name in USV_PROPERTY_ORDER:
            out_paths.append(self.make_property_tuning_distribution_figure(
                triage_pkl_path=triage_pkl_path,
                property_name=property_name,
                catalog_csv_path=catalog_csv_path,
                direction=direction,
                k_min=k_min,
                require_majority=require_majority,
                out_dir=out_dir,
                fig_format=fig_format,
            ))
        return out_paths

    def _collect_consistent_category_self(
            self,
            triage_pkl_path: str | pathlib.Path,
            segmentation: str,
            catalog_csv_path: str | pathlib.Path | None = None,
            k_min: int = 2,
            require_majority: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Description
        -----------
        Walk the unit-triage pickle and, for every good + somatic unit
        with at least two significantly-UP sessions of
        `usv_category_self_<segmentation>` (i.e. sessions with
        `peak_abs_z >= z_threshold` AND `peak_signed_z > 0`) whose
        `best_cat` values agree on a single category (the mode count
        is `>= k_min` and meets the optional majority gate), return
        one summary per consistent unit:

          * `best_cat` — the mode of `best_cat` across the unit's
            up-sig sessions (the unit's preferred category).
          * `median_peak_signed_z` — median signed z across up-sig
            sessions (firing strength at that preferred category).
          * `median_n_sig_categories` — median number of categories
            that passed |z| >= z_threshold within a session
            (breadth-of-tuning proxy).
          * `median_selectivity` — median selectivity index (the
            normalised peakedness measure persisted by the upstream
            compute step).
          * `n_sig_up` — total number of up-sig sessions the unit
            contributed.
          * `k` — size of the mode (= how many sig sessions agreed
            on `best_cat`).

        Significance rule for `usv_category_self_*` (upstream code
        `unit_triage_aggregator.py:646-647`) is purely z-based —
        `peak_abs_z >= z_threshold`. No `max_run` requirement
        because category axes are discrete labels, not 1D-binned
        tuning curves.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        segmentation (str)
            One of `USV_CATEGORY_SEGMENTATIONS` (e.g.
            `'vae_supercategory'`).
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to `unit_catalog.csv`. Defaults to the
            `catalog_path` field embedded in the triage pickle.
        k_min (int)
            Minimum mode count (number of sig sessions agreeing on
            `best_cat`) for the unit to count as consistent.
        require_majority (bool)
            When True, also requires `mode_count / n_sig_up >= 0.5`.

        Returns
        -------
        per_group (dict[str, list[dict]])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a list of per-unit summary dicts.
        """

        if segmentation not in USV_CATEGORY_SEGMENTATIONS:
            raise ValueError(
                f"unknown segmentation {segmentation!r}; "
                f"expected one of {USV_CATEGORY_SEGMENTATIONS}"
            )

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_group: dict[str, list[dict]] = {g: [] for g in VMI_REGION_ORDER}
        modality_key = f"usv_category_self_{segmentation}"

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            group = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            best_cats: list[int] = []
            peak_zs:   list[float] = []
            n_sigs:    list[int] = []
            selects:   list[float] = []
            for cond in u["conditions"].values():
                m = cond["modalities"].get(modality_key)
                if m is None:
                    continue
                for e in m["per_session"]:
                    if not e["significant"]:
                        continue
                    psz = e.get("peak_signed_z")
                    bc = e.get("best_cat")
                    if psz is None or bc is None or float(psz) <= 0:
                        continue
                    best_cats.append(int(bc))
                    peak_zs.append(float(psz))
                    n_sigs.append(int(e.get("n_sig_categories", 0)))
                    sel = e.get("selectivity")
                    selects.append(float(sel) if sel is not None else float("nan"))
            n_sig_up = len(best_cats)
            if n_sig_up < 2:
                continue
            counter = Counter(best_cats)
            mode_cat, mode_count = counter.most_common(1)[0]
            if mode_count < k_min:
                continue
            if require_majority and (mode_count / n_sig_up) < 0.5:
                continue

            sel_arr = np.array([s for s in selects if not np.isnan(s)])
            per_group[group].append({
                "best_cat":                  int(mode_cat),
                "k":                         int(mode_count),
                "n_sig_up":                  n_sig_up,
                "median_peak_signed_z":      float(np.median(peak_zs)),
                "median_n_sig_categories":   float(np.median(n_sigs)),
                "median_selectivity":        float(np.median(sel_arr)) if sel_arr.size else float("nan"),
            })

        return per_group

    def make_category_peak_distribution_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            segmentation: str,
            catalog_csv_path: str | pathlib.Path | None = None,
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the per-region distribution of consistent units'
        preferred category (`best_cat`) for one USV-category
        segmentation. Layout is the standard 2×4 grid:

          * Seven per-region bar charts of consistent-unit counts
            per `best_cat` (1..N_classes), region palette colour.
          * 8th panel — region × category heatmap (rows = regions,
            cols = categories, colour = fraction of region's
            consistent units assigned to that category) rendered with
            the configured cmap from `visualizations_settings.json`.

        "Consistent" = good + somatic AND at least `k_min`
        sig-up-only sessions agreeing on `best_cat` (mode), with the
        optional majority gate.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        segmentation (str)
            One of `USV_CATEGORY_SEGMENTATIONS`.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit catalog CSV.
        k_min (int)
            Minimum mode count for consistency.
        require_majority (bool)
            Whether to also require `mode_count / n_sig_up >= 0.5`.
        out_dir (str | pathlib.Path | None)
            Override the configured visualizations directory.
        fig_format (str | None)
            Override the configured figure format.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_group = self._collect_consistent_category_self(
            triage_pkl_path=triage_pkl_path,
            segmentation=segmentation,
            catalog_csv_path=catalog_csv_path,
            k_min=k_min,
            require_majority=require_majority,
        )
        region_colors = self._resolve_region_colors()
        density_cmap = self.visualizations_parameter_dict["figures"]["cmap"]

        n_classes = USV_CATEGORY_N_CLASSES[segmentation]
        class_ids = np.arange(1, n_classes + 1)

        fig = plt.figure(figsize=(14.0, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.40,
            wspace=0.30,
            left=0.06, right=0.985,
            top=0.93, bottom=0.13,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

        # Per-region bar charts.
        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            units = per_group[region]
            region_color = region_colors[region]

            if not units:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            counts = np.zeros(n_classes, dtype=int)
            for u in units:
                if 1 <= u["best_cat"] <= n_classes:
                    counts[u["best_cat"] - 1] += 1

            ax.bar(
                class_ids, counts,
                color=region_color, alpha=0.95,
                edgecolor=COLOR_BLACK, linewidth=0.4,
                width=1.0,
            )
            ax.set_xticks(class_ids)
            ax.set_xlim(0.5, n_classes + 0.5)
            ax.set_title(f"{region}  (N={len(units)})", fontsize=10)
            ax.tick_params(labelsize=8)
            if idx[0] == 1:
                ax.set_xlabel("best_cat", fontsize=9)
            if idx[1] == 0:
                ax.set_ylabel("unit count", fontsize=9)

        # 8th cell — region × category heatmap, fraction-normalised.
        ax_hm = fig.add_subplot(gs[1, 3])
        frac_grid = np.zeros((len(VMI_REGION_ORDER), n_classes), dtype=float)
        n_per_region: list[int] = []
        for i, region in enumerate(VMI_REGION_ORDER):
            units = per_group[region]
            n = len(units)
            n_per_region.append(n)
            if not n:
                continue
            for u in units:
                if 1 <= u["best_cat"] <= n_classes:
                    frac_grid[i, u["best_cat"] - 1] += 1
            frac_grid[i] /= n
        im = ax_hm.imshow(
            frac_grid, aspect="auto", origin="lower",
            cmap=density_cmap, vmin=0.0, vmax=float(frac_grid.max() or 1.0),
        )
        ax_hm.set_xticks(np.arange(n_classes))
        ax_hm.set_xticklabels([str(i + 1) for i in range(n_classes)], fontsize=7)
        ax_hm.set_yticks(np.arange(len(VMI_REGION_ORDER)))
        ax_hm.set_yticklabels(
            [f"{r} (N={n})" for r, n in zip(VMI_REGION_ORDER, n_per_region)],
            fontsize=7,
        )
        ax_hm.set_xlabel("best_cat", fontsize=9)
        ax_hm.set_title("region × category", fontsize=10)
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("frac of region", fontsize=8)

        fig.text(
            0.5, 0.005,
            f"{segmentation}  ·  good + somatic, consistent (sig + up) only  ·  "
            f"consistency = $\\geq${k_min} sig-up sessions agree on best_cat"
            f"{' AND >=50% majority' if require_majority else ''}",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem=f"category_peak_distribution_{segmentation}",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def make_category_selectivity_breadth_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            segmentation: str,
            catalog_csv_path: str | pathlib.Path | None = None,
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the per-region joint distribution of consistent
        units' tuning **breadth** (`n_sig_categories`, x) and
        **selectivity** (`selectivity`, y) for one segmentation.
        Layout is the standard 2×4 grid:

          * Seven per-region scatters; dot per unit, x = median
            `n_sig_categories` across sig-up sessions, y = median
            `selectivity`, dot size scaled by median
            `peak_signed_z` (firing strength).
          * 8th panel — overlaid ECDFs of `selectivity` across
            regions for direct cross-region comparison.

        Lets you see at a glance whether a region's consistently
        tuned units are narrowly specialised (`n_sig_categories=1`,
        `selectivity` high) or broadly responsive
        (`n_sig_categories` many, `selectivity` low).

        Parameters
        ----------
        See `make_category_peak_distribution_figure` for the shared
        consistency-rule parameters. Otherwise identical.

        Returns
        -------
        out_path (pathlib.Path)
            Absolute path to the written figure.
        """

        per_group = self._collect_consistent_category_self(
            triage_pkl_path=triage_pkl_path,
            segmentation=segmentation,
            catalog_csv_path=catalog_csv_path,
            k_min=k_min,
            require_majority=require_majority,
        )
        region_colors = self._resolve_region_colors()
        n_classes = USV_CATEGORY_N_CLASSES[segmentation]

        # Marker-size mapping: linear in median peak_signed_z, clipped
        # to a plotting-friendly range so very strong outliers don't
        # dominate.
        SIZE_MIN, SIZE_MAX = 8.0, 60.0
        Z_LOW, Z_HIGH = 3.0, 20.0

        def _z_to_size(zs: np.ndarray) -> np.ndarray:
            """
            Description
            -----------
            Map per-unit `median_peak_signed_z` values to scatter
            marker sizes (points²). Linear in z, clipped to
            `[Z_LOW, Z_HIGH]` so extreme outliers don't blow up the
            dot range.

            Parameters
            ----------
            zs (np.ndarray)
                Per-unit median signed z values.

            Returns
            -------
            sizes (np.ndarray)
                Per-unit `s=` values.
            """
            clipped = np.clip(zs, Z_LOW, Z_HIGH)
            frac = (clipped - Z_LOW) / (Z_HIGH - Z_LOW)
            return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * frac

        fig = plt.figure(figsize=(14.0, 6.4), dpi=150)
        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.40,
            wspace=0.30,
            left=0.06, right=0.985,
            top=0.93, bottom=0.13,
        )
        panel_xy = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

        for region, idx in zip(VMI_REGION_ORDER, panel_xy):
            ax = fig.add_subplot(gs[idx])
            units = per_group[region]
            region_color = region_colors[region]
            if not units:
                ax.set_title(f"{region}  (N=0)", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            x_all = np.array([u["median_n_sig_categories"] for u in units])
            y_all = np.array([u["median_selectivity"] for u in units])
            z_all = np.array([u["median_peak_signed_z"] for u in units])

            # Drop units whose median n_sig_categories is 0. These pass
            # the sig-up gate (peak_z >= threshold) but no individual
            # category clears the per-bin (p0.5, p99.5) shuffle band
            # on median, which is confusing to interpret as a "breadth"
            # value of 0 — exclude them from this figure to keep the
            # axis honest.
            mask = x_all > 0
            x = x_all[mask]
            y = y_all[mask]
            z = z_all[mask]
            sizes = _z_to_size(z)
            ax.scatter(
                x, y, s=sizes, c=region_color, alpha=0.85,
                edgecolors=COLOR_BLACK, linewidths=0.4, rasterized=True,
            )

            # Per-region regression overlay (seaborn linear fit + 95%
            # CI band) plus Spearman rho annotated in the panel title.
            # Spearman is the right statistic given x is a discrete
            # category count and y is bounded in [0, 1]; the seaborn
            # line is purely a visual aid for the negative trend.
            rho_str = ""
            if x.size >= 5 and np.unique(x).size >= 2:
                rho, p_val = spearmanr(x, y)
                if np.isfinite(rho):
                    rho_str = f"  ρ={rho:+.2f}"
                sns.regplot(
                    x=x, y=y, ax=ax, scatter=False,
                    color=region_color,
                    line_kws={"linewidth": 1.2, "alpha": 0.85},
                    ci=95,
                )

            ax.set_xlim(0.0, n_classes + 0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(np.arange(0, n_classes + 1, max(1, n_classes // 5)))
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_title(f"{region}  (N={int(mask.sum())}){rho_str}", fontsize=10)
            ax.tick_params(labelsize=8)
            if idx[0] == 1:
                ax.set_xlabel(r"$n_\mathrm{sig\,categories}$ (median)", fontsize=9)
            if idx[1] == 0:
                ax.set_ylabel("selectivity (median)", fontsize=9)
            if region == "PAG":
                # In-axes size legend: 3 reference dots at z = 3, 10, 20.
                z_ref = np.array([3.0, 10.0, 20.0])
                s_ref = _z_to_size(z_ref)
                x_dot = 0.86
                y_top = 0.92
                dy = 0.075
                ax.text(
                    x_dot, y_top + dy * 0.95,
                    "med. $z$", fontsize=7, color=COLOR_BLACK,
                    ha="center", va="center", transform=ax.transAxes,
                )
                for i_ref, (z_v, sz) in enumerate(zip(z_ref, s_ref)):
                    ax.scatter(
                        x_dot, y_top - dy * i_ref,
                        s=sz, c=COLOR_BLACK, edgecolors="none",
                        transform=ax.transAxes, clip_on=False, zorder=5,
                    )
                    ax.text(
                        x_dot - 0.045, y_top - dy * i_ref,
                        f"{z_v:g}", fontsize=7, color=COLOR_BLACK,
                        ha="right", va="center", transform=ax.transAxes,
                    )

        ax_agg = fig.add_subplot(gs[1, 3])
        for region in VMI_REGION_ORDER:
            units = per_group[region]
            if not units:
                continue
            sel = np.sort(np.array([u["median_selectivity"] for u in units]))
            sel = sel[~np.isnan(sel)]
            if sel.size < 2:
                continue
            ecdf_y = np.arange(1, sel.size + 1) / sel.size
            ax_agg.plot(
                sel, ecdf_y,
                color=region_colors[region], linewidth=1.4,
                label=f"{region} (N={sel.size})",
            )
        ax_agg.set_xlim(0.0, 1.05)
        ax_agg.set_ylim(0.0, 1.02)
        ax_agg.set_xlabel("selectivity (median)", fontsize=9)
        ax_agg.set_ylabel("ECDF", fontsize=9)
        ax_agg.set_title("selectivity across regions", fontsize=10)
        ax_agg.tick_params(labelsize=8)
        ax_agg.legend(fontsize=7, frameon=False, loc="lower right")

        fig.text(
            0.5, 0.005,
            f"{segmentation}  ·  good + somatic, consistent (sig + up) only  ·  "
            f"consistency = $\\geq${k_min} sig-up sessions agree on best_cat"
            f"{' AND >=50% majority' if require_majority else ''}  ·  "
            r"dot area $\propto$ median peak_signed_z",
            ha="center", fontsize=9, color=COLOR_GRAY_DASH,
        )

        out_path = save_figure(
            fig=fig,
            stem=f"category_selectivity_breadth_{segmentation}",
            viz_settings=self.visualizations_parameter_dict,
            override_dir=out_dir,
            override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def make_all_category_figures(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> list[pathlib.Path]:
        """
        Description
        -----------
        Convenience wrapper that renders both category figures (peak
        distribution + selectivity-breadth scatter) for every
        segmentation in `USV_CATEGORY_SEGMENTATIONS`. Returns the
        list of output paths in render order so the caller can log
        and cross-reference them.

        Parameters
        ----------
        See per-segmentation method docstrings; identical knobs.

        Returns
        -------
        out_paths (list[pathlib.Path])
            8 paths: 4 segmentations × 2 figures each.
        """

        out_paths: list[pathlib.Path] = []
        for segmentation in USV_CATEGORY_SEGMENTATIONS:
            out_paths.append(self.make_category_peak_distribution_figure(
                triage_pkl_path=triage_pkl_path,
                segmentation=segmentation,
                catalog_csv_path=catalog_csv_path,
                k_min=k_min,
                require_majority=require_majority,
                out_dir=out_dir,
                fig_format=fig_format,
            ))
            out_paths.append(self.make_category_selectivity_breadth_figure(
                triage_pkl_path=triage_pkl_path,
                segmentation=segmentation,
                catalog_csv_path=catalog_csv_path,
                k_min=k_min,
                require_majority=require_majority,
                out_dir=out_dir,
                fig_format=fig_format,
            ))
        return out_paths

    # behavioral tuning summary (pose / movement / social tier matrix)

    def _compute_behavioral_bucket_flags(
            self,
            unit: dict,
            recorded_mouse_id: str,
            condition: str,
            k_min: int,
            require_majority: bool,
    ) -> dict[str, bool]:
        """
        Description
        -----------
        Compute the per-bucket `(pose, movement, social)` boolean
        flags for one unit. Encapsulates the modality-iteration +
        consistency-rule + dyadic-pooling logic that both
        `_classify_unit_behavioral_tier` (figure 1) and
        `make_three_set_overlap_venn_figure` (figure 2) need, so the
        same rule applies to both figures by construction.

        Bucket classification rules:
          * self pose / movement: `prefix == recorded_mouse_id` AND
            `feat` belongs to `BEHAVIORAL_POSE_FEATURES` or
            `BEHAVIORAL_MOVEMENT_FEATURES` respectively.
          * social: `prefix` contains a hyphen (dyadic feature key,
            e.g. `158112_0-156693_3.nose-nose`). The dyadic-prefix
            check is direction-agnostic — both `<self>-<partner>` and
            `<partner>-<self>` are counted as social because both
            require the partner's presence to be defined.
          * partner-self (single non-matching mouse id): explicitly
            skipped per the figure spec.

        Per-feature consistency rule (mirrors the rest of the suite):
          A feature is "tuned" if either direction (excit OR suppress)
          of its modality block has `n_significant >= k_min`
          significant sessions; when `require_majority` is True the
          significant sessions must also form a strict majority of
          `n_tested`. The bucket flag goes True as soon as ONE feature
          in the bucket meets the criterion (early-exit).

        Dyadic modality keys embed the partner-mouse identity
        (`<self>-<partner_id>.<feat>_<dir>`), so a unit recorded
        across sessions with different partners produces multiple
        dyadic keys for the same `<feat>`, each with `n_tested = 1`.
        The cross-session consistency rule then trivially fails on
        every dyadic key in isolation. To preserve the same
        "consistent in >= k_min sessions" semantics as for self
        features, we pool dyadic per-key counts by `(feat, direction)`
        across all partner identities BEFORE applying the
        consistency rule. The pooled counts are equivalent to "is
        this cell consistently tuned to feature F, ignoring which
        partner happened to be present" — exactly the question the
        social bucket is asking.

        Parameters
        ----------
        unit (dict)
            One entry from `triage["units"]`.
        recorded_mouse_id (str)
            The `mouse_id` field of the unit — used to disambiguate
            self-pose modalities from partner-self pose modalities.
        condition (str)
            Which `unit["conditions"][...]` block to walk
            (e.g. `"intact_female"`).
        k_min (int)
            Minimum sig-session count per feature.
        require_majority (bool)
            Apply the strict-majority gate on top of `k_min`.

        Returns
        -------
        flags (dict[str, bool])
            Mapping `{"pose": bool, "movement": bool, "social":
            bool}`. Units with no `condition` entry get all-False.
        """

        cond = unit["conditions"].get(condition)
        if not cond:
            return {"pose": False, "movement": False, "social": False}

        bucket_tuned = {"pose": False, "movement": False, "social": False}

        # Dyadic modality keys embed the partner-mouse identity
        # (`<self>-<partner_id>.<feat>_<dir>`), so a unit recorded
        # across sessions with different partners produces multiple
        # dyadic keys for the same `<feat>`, each with `n_tested = 1`.
        # The cross-session consistency rule then trivially fails on
        # every dyadic key in isolation. To preserve the same
        # "consistent in >= k_min sessions" semantics as for self
        # features, we pool dyadic per-key counts by `(feat, direction)`
        # across all partner identities BEFORE applying the
        # consistency rule. The pooled counts are equivalent to "is
        # this cell consistently tuned to feature F, ignoring which
        # partner happened to be present" — exactly the question the
        # social bucket is asking.
        dyadic_pool: dict[tuple[str, str], dict[str, int]] = {}

        for mkey, payload in cond.get("modalities", {}).items():
            parsed = _parse_behavioral_modality_key(mkey)
            if parsed is None:
                continue
            prefix, feat, direction = parsed
            if not isinstance(payload, dict):
                continue
            n_sig = int(payload.get("n_significant", 0))
            n_test = int(payload.get("n_tested", 0))

            if "-" in prefix:
                # Pool dyadic across partner identities.
                agg = dyadic_pool.setdefault((feat, direction), {"n_sig": 0, "n_test": 0})
                agg["n_sig"] += n_sig
                agg["n_test"] += n_test
                continue

            if prefix == recorded_mouse_id:
                if feat in BEHAVIORAL_POSE_FEATURES:
                    bucket = "pose"
                elif feat in BEHAVIORAL_MOVEMENT_FEATURES:
                    bucket = "movement"
                else:
                    continue
            else:
                # partner-self single-mouse-id prefix — explicitly
                # ignored per the figure spec.
                continue

            if bucket_tuned[bucket]:
                # Bucket already flagged by an earlier feature; skip
                # the consistency check.
                continue

            if n_sig < k_min:
                continue
            if require_majority and n_test > 0 and (n_sig / n_test) <= 0.5:
                continue
            bucket_tuned[bucket] = True

        # Apply the same consistency rule to the pooled dyadic
        # counts; flip the social bucket on the first `(feat,
        # direction)` pair that passes.
        for (feat, direction), agg in dyadic_pool.items():
            if bucket_tuned["social"]:
                break
            n_sig = agg["n_sig"]
            n_test = agg["n_test"]
            if n_sig < k_min:
                continue
            if require_majority and n_test > 0 and (n_sig / n_test) <= 0.5:
                continue
            bucket_tuned["social"] = True

        return bucket_tuned

    def _classify_unit_behavioral_tier(
            self,
            unit: dict,
            recorded_mouse_id: str,
            condition: str,
            k_min: int,
            require_majority: bool,
    ) -> str:
        """
        Description
        -----------
        Map the per-bucket flags produced by
        `_compute_behavioral_bucket_flags` to a string tier name in
        `BEHAVIORAL_TIER_ORDER`. Thin wrapper kept as a public-ish
        interface so figure (1)'s collector can iterate one unit at
        a time without knowing the bool-tuple → tier mapping.

        Parameters
        ----------
        See `_compute_behavioral_bucket_flags`; identical signature.

        Returns
        -------
        tier (str)
            One of the eight entries in `BEHAVIORAL_TIER_ORDER`.
        """

        flags = self._compute_behavioral_bucket_flags(
            unit=unit, recorded_mouse_id=recorded_mouse_id,
            condition=condition, k_min=k_min,
            require_majority=require_majority,
        )
        p, m, s = flags["pose"], flags["movement"], flags["social"]
        triple_to_tier = {
            (False, False, False): "none",
            (True,  False, False): "pose_only",
            (False, True,  False): "movement_only",
            (False, False, True ): "social_only",
            (True,  True,  False): "pose+movement",
            (True,  False, True ): "pose+social",
            (False, True,  True ): "movement+social",
            (True,  True,  True ): "all_three",
        }
        return triple_to_tier[(p, m, s)]

    # Modality-key prefixes that count as "vocal" for the
    # behavioral / social / vocal overlap. Each modality block under
    # one of these prefixes is checked with the same
    # `n_significant >= k_min` + optional majority-gate rule used by
    # the behavioral classifier — and the unit is flagged "vocal" as
    # soon as ANY modality passes. Direction is folded in automatically
    # (excit and suppress modalities each get their own key).
    _VOCAL_MODALITY_PREFIXES: tuple[str, ...] = (
        "vmi_self_",
        "usv_peth_self_",
        "usv_property_self_",
        "usv_category_self_",
        "usv_category_peth_self_",
    )

    def _compute_vocal_flag(
            self,
            unit: dict,
            condition: str,
            k_min: int,
            require_majority: bool,
    ) -> bool:
        """
        Description
        -----------
        Return True iff the unit is "vocally tuned" — i.e.
        consistently significant in at least one USV-related modality
        block under `condition`. Iterates every modality key whose
        prefix matches one of `_VOCAL_MODALITY_PREFIXES` (VMI,
        USV-PETH, USV-property tuning, USV-category tuning,
        USV-category-PETH) and applies the same per-feature
        consistency rule the behavioral classifier uses.

        Direction handling: each direction-split modality (e.g.
        `usv_property_self_duration_excit` and
        `usv_property_self_duration_suppress`) is its own key, so
        passing on EITHER direction independently flips the flag —
        same union semantics as the behavioral side.

        Parameters
        ----------
        unit (dict)
            One entry from `triage["units"]`.
        condition (str)
            Which `unit["conditions"][...]` block to walk.
        k_min (int)
            Minimum sig-session count per modality.
        require_majority (bool)
            Apply the strict-majority gate on top of `k_min`.

        Returns
        -------
        vocal (bool)
            True on first modality whose pooled (`n_sig`, `n_test`)
            satisfies the consistency rule; False otherwise.
        """

        cond = unit["conditions"].get(condition)
        if not cond:
            return False
        for mkey, payload in cond.get("modalities", {}).items():
            if not isinstance(payload, dict):
                continue
            if not any(mkey.startswith(px) for px in self._VOCAL_MODALITY_PREFIXES):
                continue
            n_sig = int(payload.get("n_significant", 0))
            n_test = int(payload.get("n_tested", 0))
            if n_sig < k_min:
                continue
            if require_majority and n_test > 0 and (n_sig / n_test) <= 0.5:
                continue
            return True
        return False

    def _collect_behavioral_tiers_per_region(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None,
            condition: str,
            k_min: int,
            require_majority: bool,
    ) -> tuple[dict[str, Counter], int]:
        """
        Description
        -----------
        Walk the unit-triage pickle and tally, per canonical brain-area
        bucket, how many good + somatic units fall into each
        `BEHAVIORAL_TIER_ORDER` tier. Returns the raw counts; the
        rendering layer normalises them to per-region fractions.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Absolute path to the unit-catalog CSV that pairs each
            `(mouse_id, rec_date, unit_id)` with `cluster_group` and
            `somatic` flags. Defaults to the `catalog_path` field
            embedded in the triage pickle when omitted.
        condition (str)
            Which `unit["conditions"][...]` block to walk
            (e.g. `"intact_female"`).
        k_min (int)
            Minimum sig-session count per feature for the consistency
            rule.
        require_majority (bool)
            Apply the strict-majority gate on top of `k_min`.

        Returns
        -------
        per_region (dict[str, Counter])
            Mapping from canonical region label (one of
            `VMI_REGION_ORDER`) to a `Counter` of tier counts. Tiers
            absent from a region simply do not appear in its
            `Counter` (the caller pads with 0).
        n_eligible (int)
            Total number of units that passed the good + somatic
            filter across all regions; used by the caption.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_region: dict[str, Counter] = {g: Counter() for g in VMI_REGION_ORDER}
        n_eligible = 0

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            region = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            tier = self._classify_unit_behavioral_tier(
                unit=u,
                recorded_mouse_id=u["mouse_id"],
                condition=condition,
                k_min=k_min,
                require_majority=require_majority,
            )
            per_region[region][tier] += 1
            n_eligible += 1

        return per_region, n_eligible

    def make_behavioral_tuning_summary_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            condition: str = "intact_female",
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render the behavioral-tuning tier matrix: a
        `len(VMI_REGION_ORDER) x len(BEHAVIORAL_TIER_ORDER)` heatmap
        whose `(r, c)` cell is the fraction of region `r`'s good +
        somatic units that fall into the `(Pose, Movement, Social)`
        disjoint tier `c`. Rows therefore sum to 1 by construction.

        Tier columns are arranged left-to-right by increasing
        multimodality:

            none, P only, M only, S only, P+M, P+S, M+S, all 3

        Bucket definitions:
          * **Pose** = 9 self raw postural features (allo/back/body/ego
            /neck/tail without derivatives).
          * **Movement** = 20 self features (`*_1st_der` / `*_2nd_der`
            of pose + `speed` + `acceleration`).
          * **Social** = 42 dyadic features (modality keys with a
            hyphen in the prefix, e.g. `158112_0-156693_3.nose-nose`).
            Partner-self pose is excluded per the figure spec.

        Per-feature consistency rule: see
        `_classify_unit_behavioral_tier`.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Optional override of the catalog CSV path; defaults to the
            `catalog_path` embedded in the triage pickle.
        condition (str)
            Which `unit["conditions"][...]` block to walk. Defaults to
            `"intact_female"` (the headline condition).
        k_min (int)
            Minimum sig-session count per feature.
        require_majority (bool)
            Apply the strict-majority gate on top of `k_min`.
        out_dir (str | pathlib.Path | None)
            Directory override; `None` falls back to
            `figures.save_directory`.
        fig_format (str | None)
            Output format override; `None` falls back to
            `figures.fig_format`.

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written figure file.
        """

        per_region, n_eligible = self._collect_behavioral_tiers_per_region(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            condition=condition,
            k_min=k_min,
            require_majority=require_majority,
        )

        n_rows = len(VMI_REGION_ORDER)
        n_cols = len(BEHAVIORAL_TIER_ORDER)
        fractions = np.zeros((n_rows, n_cols), dtype=float)
        counts = np.zeros((n_rows, n_cols), dtype=int)
        region_totals = np.zeros(n_rows, dtype=int)
        for r, region in enumerate(VMI_REGION_ORDER):
            ctr = per_region[region]
            total = int(sum(ctr.values()))
            region_totals[r] = total
            if total == 0:
                continue
            for c, tier in enumerate(BEHAVIORAL_TIER_ORDER):
                k = int(ctr.get(tier, 0))
                counts[r, c] = k
                fractions[r, c] = k / total

        # render
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        cmap_name = self.visualizations_parameter_dict.get(
            "figures", {}
        ).get("cmap", "inferno")
        im = ax.imshow(
            fractions, aspect="auto", cmap=cmap_name, vmin=0.0, vmax=1.0,
        )

        # Tick labels. The region row labels are individually
        # color-coded using the project's brain-area palette so a
        # glance at the y-axis still maps each row to the same color
        # as the per-region scatter / bar figures elsewhere in the
        # suite.
        region_colors = self._resolve_region_colors()
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(
            [f"{r}\n(n={int(t)})" for r, t in zip(VMI_REGION_ORDER, region_totals)],
            fontsize=7,
        )
        for label, region in zip(ax.get_yticklabels(), VMI_REGION_ORDER):
            label.set_color(region_colors[region])
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(
            [BEHAVIORAL_TIER_LABELS[t] for t in BEHAVIORAL_TIER_ORDER],
            rotation=30, ha="right", fontsize=7,
        )

        # Numeric annotations: light text on dark cells, dark text on
        # light cells (threshold at 0.5 of the [0, 1] colormap).
        for r in range(n_rows):
            for c in range(n_cols):
                frac = float(fractions[r, c])
                text_color = "#FFFFFF" if frac < 0.5 else COLOR_BLACK
                ax.text(
                    c, r, f"{frac:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=7,
                )

        # Tight box / cleaner spines for a heatmap look
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("fraction of region's good+somatic units", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        majority_tag = "majority" if require_majority else "no-majority"
        # Explicit `fontweight='light'` overrides the project mplstyle's
        # `axes.titleweight: bold` / `figure.titleweight: bold` so the
        # whole figure renders in Helvetica Light. Without this the
        # title text emits `font-weight: 700` and Inkscape resolves to
        # the system Helvetica Bold instead of the bundled Helvetica
        # Light, breaking visual consistency with the rest of the suite.
        fig.suptitle(
            f"behavioral tuning tier matrix · {condition.replace('_', ' ')} · "
            f"N={n_eligible} good+somatic units",
            fontsize=10, y=0.99, fontweight="light",
        )
        ax.set_title(
            f"pose={len(BEHAVIORAL_POSE_FEATURES)}f · "
            f"movement={len(BEHAVIORAL_MOVEMENT_FEATURES)}f · "
            f"social=dyadic only (42f) · "
            f"k_min={k_min}, {majority_tag}",
            fontsize=8, fontweight="light",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        out_path = save_figure(
            fig, "behavioral_tuning_tier_matrix",
            self.visualizations_parameter_dict,
            override_dir=out_dir, override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    # behavioral / social / vocal overlap (3-set Venn)

    def _collect_three_set_overlap_counts(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None,
            condition: str,
            k_min: int,
            require_majority: bool,
    ) -> tuple[Counter, int]:
        """
        Description
        -----------
        Walk the unit-triage pickle and tally how many good + somatic
        units fall into each of the eight `(behavioral, social,
        vocal)` boolean triples. "Behavioral" = `pose OR movement`
        flag from `_compute_behavioral_bucket_flags`; "social" = the
        same helper's social flag; "vocal" = `_compute_vocal_flag`.
        Counts are pooled across all brain regions — this figure is
        a population-level overlap summary, not a per-region split.

        Parameters
        ----------
        triage_pkl_path (str | pathlib.Path)
            Absolute path to the `unit_triage_*.pkl` artifact.
        catalog_csv_path (str | pathlib.Path | None)
            Catalog CSV override; defaults to the path embedded in
            the triage pickle.
        condition (str)
            Which `unit["conditions"][...]` block to walk.
        k_min (int)
            Minimum sig-session count per modality.
        require_majority (bool)
            Apply the strict-majority gate on top of `k_min`.

        Returns
        -------
        counts (Counter)
            Maps `(beh: bool, soc: bool, voc: bool)` tuples to the
            number of units falling in that disjoint tier. All eight
            tuples appear as keys (zero-padded by caller).
        n_eligible (int)
            Total good + somatic units that passed the catalog filter.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        counts: Counter = Counter()
        n_eligible = 0
        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            flags = self._compute_behavioral_bucket_flags(
                unit=u, recorded_mouse_id=u["mouse_id"],
                condition=condition,
                k_min=k_min, require_majority=require_majority,
            )
            beh = bool(flags["pose"] or flags["movement"])
            soc = bool(flags["social"])
            voc = self._compute_vocal_flag(
                unit=u, condition=condition,
                k_min=k_min, require_majority=require_majority,
            )
            counts[(beh, soc, voc)] += 1
            n_eligible += 1

        return counts, n_eligible

    def _draw_overlap_venn_on_ax(
            self,
            ax,
            counts: Counter,
            n_panel: int,
            vocal_color: str,
            panel_title: str | None = None,
            font_scale: float = 1.0,
    ) -> None:
        """
        Description
        -----------
        Render a single 3-set Venn (Kinematics / Social Features /
        Vocal) onto a pre-existing matplotlib axes. Shared by the
        population-level (one global panel) and per-region (2x4
        small-multiples) figures so the visual idiom is identical
        across both.

        Geometry: three equal-radius circles centred on the vertices
        of an equilateral triangle whose centroid is the origin.
        Each disjoint region is annotated with `n  (%)` text; the
        "None" tier sits in project gray beneath the circles.

        Set colours:
          * Kinematics: `male_colors[0]` (`#9AC0CD`) — the "self
            mouse" palette entry.
          * Social Features: `social_colors[0]` (`#5A6470`).
          * Vocal: the caller-supplied `vocal_color`. For the
            per-region figure that's the brain-area hex for that
            panel; for the aggregate panel it's the project's
            unassigned-grey.

        Parameters
        ----------
        ax (matplotlib.axes.Axes)
            Target axes. Aspect ratio + limits are set here.
        counts (Counter)
            Maps `(beh, soc, voc)` tuples to unit counts.
        n_panel (int)
            Denominator for the percentages; usually the total of
            `counts.values()`.
        vocal_color (str)
            Hex string for the Vocal circle.
        panel_title (str | None)
            Optional title (region name) rendered above the Venn.
        font_scale (float)
            Multiplier on font sizes; per-region panels use ~0.8 so
            text fits inside the smaller circles.

        Returns
        -------
        None
        """

        # Equilateral triangle of unit-radius circles, centroid at
        # the origin.
        r = 1.0
        cx = (-0.5, +0.5, 0.0)
        cy = (-math.sqrt(3.0) / 6.0, -math.sqrt(3.0) / 6.0, +math.sqrt(3.0) / 3.0)

        set_colors = (
            self.visualizations_parameter_dict["male_colors"][0],
            self.visualizations_parameter_dict["social_colors"][0],
            vocal_color,
        )
        set_labels = ("Kinematics", "Social Features", "Vocal")

        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.2)
        ax.axis("off")

        for i, color in enumerate(set_colors):
            ax.add_patch(Circle(
                (cx[i], cy[i]), r,
                facecolor=color, alpha=0.35,
                edgecolor=color, linewidth=1.5,
            ))

        # Set labels just outside each circle's far edge. Anchors are
        # tuned so labels don't collide with the count text.
        label_positions = ((-1.65, -1.10), (+1.65, -1.10), (0.0, +1.85))
        label_anchors = (("right", "top"), ("left", "top"), ("center", "bottom"))
        for label, pos, anchor, color in zip(
            set_labels, label_positions, label_anchors, set_colors
        ):
            ax.text(
                pos[0], pos[1], label,
                ha=anchor[0], va=anchor[1], color=color,
                fontsize=10 * font_scale, fontweight="light",
            )

        total = max(1, n_panel)

        def _annotate(triple, pos):
            n = counts.get(triple, 0)
            p = 100.0 * n / total
            ax.text(
                pos[0], pos[1], f"{n}\n({p:.1f}%)",
                ha="center", va="center",
                fontsize=9 * font_scale, fontweight="light",
                color=COLOR_BLACK,
            )

        _annotate((True,  False, False), (-1.05, -0.40))
        _annotate((False, True,  False), (+1.05, -0.40))
        _annotate((False, False, True ), (0.00,  +1.20))
        _annotate((True,  True,  False), (0.00,  -0.70))
        _annotate((True,  False, True ), (-0.60, +0.30))
        _annotate((False, True,  True ), (+0.60, +0.30))
        _annotate((True,  True,  True ), (0.00,  -0.10))

        n_none = counts.get((False, False, False), 0)
        p_none = 100.0 * n_none / total
        ax.text(
            0.0, -1.85, f"None: n={n_none}  ({p_none:.1f}%)",
            ha="center", va="bottom",
            fontsize=8 * font_scale, fontweight="light",
            color=COLOR_GRAY_DASH,
        )

        if panel_title is not None:
            ax.set_title(panel_title, fontsize=11 * font_scale, fontweight="light")

    def make_three_set_overlap_venn_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            condition: str = "intact_female",
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Render a 3-set Venn diagram (Behavioral / Social / Vocal)
        summarising the population-level overlap between cells with
        behavioral self-feature tuning (`pose OR movement` from
        figure 1's rule), social/dyadic tuning, and vocal tuning
        (any USV modality). Each of the seven inside regions is
        annotated with the count and the percentage of good + somatic
        units it contains; the eighth tier ("neither") is annotated
        outside the circles.

        Circles are drawn hand-rolled (no `matplotlib_venn`
        dependency): three equal-radius circles centred on the
        vertices of an equilateral triangle, semitransparent fills
        in three on-brand palette colours. Layout is fixed (not
        area-proportional) — exact area-proportional 3-set Venns
        don't exist for arbitrary tier cardinalities, and the
        numeric annotations carry the actual counts anyway.

        Set definitions:
          * **Behavioral** = `pose OR movement` from
            `_compute_behavioral_bucket_flags`. 29 self features (9
            pose + 20 movement) on the recorded mouse.
          * **Social** = the same helper's social flag — partner-
            pooled dyadic consistency over 42 dyadic features.
          * **Vocal** = `_compute_vocal_flag` — any of VMI / USV-PETH
            / USV-property / USV-category / USV-category-PETH passes
            the same per-modality consistency rule.

        Parameters
        ----------
        triage_pkl_path, catalog_csv_path, condition, k_min,
        require_majority, out_dir, fig_format
            Same semantics as
            `make_behavioral_tuning_summary_figure`.

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written figure file.
        """

        counts, n_eligible = self._collect_three_set_overlap_counts(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            condition=condition,
            k_min=k_min,
            require_majority=require_majority,
        )

        # Single global panel: vocal circle uses the project's
        # unassigned-grey because no specific brain area applies to
        # the all-pooled population.
        unassigned_grey = self.visualizations_parameter_dict["unassigned_colors"][0]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        self._draw_overlap_venn_on_ax(
            ax=ax,
            counts=counts,
            n_panel=n_eligible,
            vocal_color=unassigned_grey,
            panel_title=None,
            font_scale=1.0,
        )

        majority_tag = "majority" if require_majority else "no-majority"
        fig.suptitle(
            f"Kinematics / Social Features / Vocal overlap · "
            f"{condition.replace('_', ' ')} · "
            f"N={n_eligible} good+somatic units · "
            f"k_min={k_min}, {majority_tag}",
            fontsize=10, y=0.99, fontweight="light",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        out_path = save_figure(
            fig, "tuning_overlap_venn",
            self.visualizations_parameter_dict,
            override_dir=out_dir, override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    def _collect_three_set_overlap_counts_per_region(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None,
            condition: str,
            k_min: int,
            require_majority: bool,
    ) -> tuple[dict[str, Counter], int]:
        """
        Description
        -----------
        Per-region variant of `_collect_three_set_overlap_counts`.
        Walks every good + somatic unit, classifies into the
        `(behavioral, social, vocal)` triple using the same helpers,
        and bins the count under the unit's canonical brain-area
        bucket from `VMI_REGION_GROUPS`. Returns one `Counter` per
        region plus the total `n_eligible` across all regions (used
        by the aggregate panel of the per-region figure).

        Parameters
        ----------
        Same as `_collect_three_set_overlap_counts`.

        Returns
        -------
        per_region (dict[str, Counter])
            `{region_label: Counter[(beh, soc, voc) -> count]}` for
            every region in `VMI_REGION_ORDER` (empty Counters
            included so the caller doesn't need to check for missing
            keys).
        n_total (int)
            Sum of all per-region totals; equivalent to the global
            `n_eligible` from the population-level collector.
        """

        triage_pkl_path = pathlib.Path(triage_pkl_path)
        with open(triage_pkl_path, "rb") as fh:
            triage = pickle.load(fh)

        if catalog_csv_path is None:
            catalog_csv_path = triage["catalog_path"]
        catalog_csv_path = pathlib.Path(catalog_csv_path)
        cat_lookup: dict[tuple[str, int, str], dict] = {}
        with open(catalog_csv_path) as fh:
            for row in csv.DictReader(fh):
                cat_lookup[(row["mouse_id"], int(row["rec_date"]), row["unit_id"])] = row

        region_to_group = {
            region: group
            for group, regions in VMI_REGION_GROUPS.items()
            for region in regions
        }
        per_region: dict[str, Counter] = {g: Counter() for g in VMI_REGION_ORDER}
        n_total = 0

        for u in triage["units"].values():
            key = (u["mouse_id"], int(u["rec_date"]), u["unit_id"])
            if key not in cat_lookup:
                continue
            cat_row = cat_lookup[key]
            if u["kslabel"] != "good":
                continue
            if str(cat_row["somatic"]).strip().lower() != "true":
                continue

            anatomy = u["anatomy_region"]
            region = region_to_group[anatomy] if anatomy in region_to_group else "Other"

            flags = self._compute_behavioral_bucket_flags(
                unit=u, recorded_mouse_id=u["mouse_id"],
                condition=condition,
                k_min=k_min, require_majority=require_majority,
            )
            beh = bool(flags["pose"] or flags["movement"])
            soc = bool(flags["social"])
            voc = self._compute_vocal_flag(
                unit=u, condition=condition,
                k_min=k_min, require_majority=require_majority,
            )
            per_region[region][(beh, soc, voc)] += 1
            n_total += 1

        return per_region, n_total

    def make_per_region_overlap_venn_figure(
            self,
            triage_pkl_path: str | pathlib.Path,
            catalog_csv_path: str | pathlib.Path | None = None,
            condition: str = "intact_female",
            k_min: int = 2,
            require_majority: bool = True,
            out_dir: str | pathlib.Path | None = None,
            fig_format: str | None = None,
    ) -> pathlib.Path:
        """
        Description
        -----------
        Per-region variant of the Kinematics / Social Features /
        Vocal overlap Venn: a 2x4 grid where panels 1-7 are
        `VMI_REGION_ORDER` (PAG, MRN, VTA, MB, CENT, SC, Other) and
        the 8th panel is the all-regions aggregate. Each panel
        shares the same 3-circle layout from
        `_draw_overlap_venn_on_ax`, and the **Vocal circle takes the
        brain-area colour of that panel's region** — so a glance at
        any panel's Vocal hue identifies the region without reading
        the title. The aggregate panel uses the project's
        unassigned-grey for its Vocal circle.

        Parameters
        ----------
        triage_pkl_path, catalog_csv_path, condition, k_min,
        require_majority, out_dir, fig_format
            Identical semantics to
            `make_three_set_overlap_venn_figure`.

        Returns
        -------
        out_path (pathlib.Path)
            Path to the written figure file.
        """

        per_region, n_total = self._collect_three_set_overlap_counts_per_region(
            triage_pkl_path=triage_pkl_path,
            catalog_csv_path=catalog_csv_path,
            condition=condition,
            k_min=k_min,
            require_majority=require_majority,
        )

        region_colors = self._resolve_region_colors()
        unassigned_grey = self.visualizations_parameter_dict["unassigned_colors"][0]

        fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.0))
        axes_flat = axes.flat

        # Panels 1-7: one per region.
        for idx, region in enumerate(VMI_REGION_ORDER):
            ax = axes_flat[idx]
            counts = per_region[region]
            n_panel = int(sum(counts.values()))
            self._draw_overlap_venn_on_ax(
                ax=ax,
                counts=counts,
                n_panel=n_panel,
                vocal_color=region_colors[region],
                panel_title=f"{region}  (n={n_panel})",
                font_scale=0.82,
            )

        # Panel 8: all-regions aggregate. Pool every per-region
        # Counter into one and use the unassigned-grey for Vocal.
        aggregate: Counter = Counter()
        for region in VMI_REGION_ORDER:
            aggregate.update(per_region[region])
        self._draw_overlap_venn_on_ax(
            ax=axes_flat[7],
            counts=aggregate,
            n_panel=n_total,
            vocal_color=unassigned_grey,
            panel_title=f"All regions  (n={n_total})",
            font_scale=0.82,
        )

        majority_tag = "majority" if require_majority else "no-majority"
        fig.suptitle(
            f"Kinematics / Social Features / Vocal overlap · "
            f"{condition.replace('_', ' ')} · "
            f"per brain area  ·  N={n_total} good+somatic units · "
            f"k_min={k_min}, {majority_tag}",
            fontsize=11, y=0.995, fontweight="light",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        out_path = save_figure(
            fig, "tuning_overlap_venn_per_region",
            self.visualizations_parameter_dict,
            override_dir=out_dir, override_format=fig_format,
        )
        plt.close(fig)
        return out_path

    @contextlib.contextmanager
    def _open_save_target(
        self,
        out_path: pathlib.Path,
        cluster_id: str,
        fig_format: str,
    ):
        """
        Description
        -----------
        Yield a `save_fig(fig, label)` callable that persists each
        rendered figure. For PDF, all pages go into one file via
        `PdfPages`. For other formats (PNG / SVG / EPS / etc.), each
        page is saved as its own file with `_p{N}_{label}` suffix.

        Parameters
        ----------
        out_path (pathlib.Path)
            Target output path; the suffix is the chosen `fig_format`.
        cluster_id (str)
            Cluster identifier (used in per-page file names for non-PDF
            formats).
        fig_format (str)
            Extension to save under (lowercase, no leading dot).

        Yields
        ------
        save_fig (Callable[[matplotlib.figure.Figure, str], None])
            Callable that accepts an open figure and a short label and
            commits the figure to the appropriate output target.

        Returns
        -------
        None
        """

        dpi = int(
            self.visualizations_parameter_dict.get("figures", {}).get("dpi", 300)
        )

        if fig_format == "pdf":
            with PdfPages(out_path) as pdf:
                page_idx = {"n": 0}

                def save_fig(fig, label: str) -> None:  # noqa: ARG001
                    """
                    Description
                    -----------
                    PDF backend: append the rendered figure to the
                    open `PdfPages` document at the configured DPI,
                    then close it. The `label` argument is unused (PDF
                    pages have no per-page filename) but kept for
                    signature parity with the non-PDF branch.

                    Parameters
                    ----------
                    fig (matplotlib.figure.Figure)
                        Figure to commit and close.
                    label (str)
                        Short page label (ignored in this branch).

                    Returns
                    -------
                    None
                    """
                    page_idx["n"] += 1
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)

                yield save_fig
        else:
            page_idx = {"n": 0}
            out_dir = out_path.parent

            def save_fig(fig, label: str) -> None:
                """
                Description
                -----------
                Non-PDF backend: write the rendered figure to its own
                file at `<cluster_id>_neuronal_tuning_p{N}_{label}.{fig_format}`
                at the configured DPI, then close the figure. `N` is
                the running page index across all pages of this
                cluster.

                Parameters
                ----------
                fig (matplotlib.figure.Figure)
                    Figure to commit and close.
                label (str)
                    Short page label inserted into the filename.

                Returns
                -------
                None
                """
                page_idx["n"] += 1
                per_page_path = (
                    out_dir
                    / f"{cluster_id}_neuronal_tuning_p{page_idx['n']}_{label}.{fig_format}"
                )
                fig.savefig(per_page_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)

            yield save_fig

    # per-cluster page rendering

    # behavioral rendering

    def _render_behavioral_pages(
        self,
        cluster_data: dict,
        mouse_id_list: list[str],
        mouse_colors: list,
        save_fig,
    ) -> None:
        """
        Description
        -----------
        For each behavioral temporal offset key (`beh_offset=*s`) in the
        per-cluster pkl, render one page per plot-feature group
        (`individual.<mouse_id>` and `social`). Each page is a small
        gridspec of (line + occupancy) pairs for 1D features plus
        per-animal 2D spatial ratemap pairs. Smoothing, occupancy
        thresholding, and colorbar placement are handled exactly as the
        legacy behavioral plotter did.

        Parameters
        ----------
        cluster_data (dict)
            Loaded pkl payload (top-level keys include `beh_offset=*s`
            entries plus optional vocal keys).
        mouse_id_list (list[str])
            Animal IDs from the tracking H5; drives per-mouse coloring.
        mouse_colors (list)
            Hex colors per mouse, in the same order as `mouse_id_list`.
        save_fig (Callable[[Figure, str], None])
            Persists each rendered page; closes the figure.

        Returns
        -------
        None
        """

        beh_offset_keys = [k for k in cluster_data if k.startswith("beh_offset=")]
        if not beh_offset_keys:
            return

        social_color = self.visualizations_parameter_dict.get(
            "social_colors", ["#5A6470"]
        )[0]
        mouse_color_dict = {"social": social_color}
        mouse_colormap_dict: dict = {}
        for mouse_idx, mouse in enumerate(mouse_id_list):
            mouse_color_dict[mouse] = mouse_colors[mouse_idx]
            mouse_colormap_dict[mouse] = create_colormap(
                input_parameter_dict={
                    "cm_length": 255,
                    "cm_name": f"{mouse}",
                    "cm_type": "sequential",
                    "cm_start": (
                        int(mouse_colors[mouse_idx][1:3], 16),
                        int(mouse_colors[mouse_idx][3:5], 16),
                        int(mouse_colors[mouse_idx][5:7], 16),
                    ),
                    "cm_end": (255, 255, 255),
                    "equalize_luminance": True,
                    "match_luminance_by": "max",
                    "change_saturation": 0.5,
                    "cm_opacity": 1,
                }
            )

        plot_features: dict[str, list[str]] = {}
        for feature_key in cluster_data[beh_offset_keys[0]]:
            mouse_id = feature_key.split(".")[0]
            if (
                f"individual.{mouse_id}" not in plot_features
                and "-" not in mouse_id
            ):
                plot_features[f"individual.{mouse_id}"] = []
            if "-" not in mouse_id:
                plot_features[f"individual.{mouse_id}"].append(feature_key)
            else:
                plot_features.setdefault("social", []).append(feature_key)

        viz_params = self.visualizations_parameter_dict.get(
            "neuronal_tuning_figures", {}
        )
        ratemap_cmap = self.visualizations_parameter_dict.get(
            "figures", {}
        ).get("cmap", "inferno")
        # behavioral occupancy threshold lives on the analyses side (it's
        # a tuning-curve config alongside usv_property_min_occupancy_seconds);
        # compute writes it into behavioral_metadata, plotter reads from
        # there with a sane fallback for older pkls.
        occ_threshold_setting = float(
            cluster_data.get("behavioral_metadata", {}).get(
                "behavioral_min_occupancy_seconds", 1.0
            )
        )

        for offset in beh_offset_keys:
            for plot_feature_key, features in plot_features.items():
                if not features:
                    continue
                if "social" in plot_feature_key:
                    rm_color = mouse_color_dict["social"]
                    mouse_colormap = 0
                else:
                    rm_color = mouse_color_dict[plot_feature_key.split(".")[-1]]
                    mouse_colormap = mouse_colormap_dict[
                        plot_feature_key.split(".")[-1]
                    ]

                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore")
                    row_num = int(np.ceil((len(features) * 2) / 6))
                    # the legacy plotter used (6.4, row_num) with ~4 pt
                    # tick / 5 pt title fonts; bumping fonts to 8/9 pt
                    # without enlarging the figure crushes the layout.
                    # scale both axes ~2.5× to give the new fonts room
                    # and tighten wspace/hspace so cells stay roughly
                    # square.
                    fig = plt.figure(
                        figsize=(16.0, max(2.6, 2.6 * row_num)), tight_layout=False
                    )
                    gs = gridspec.GridSpec(
                        nrows=row_num,
                        ncols=6,
                        wspace=0.40,
                        hspace=0.45,
                        left=0.06,
                        right=0.96,
                        bottom=0.07,
                        top=0.94,
                    )
                    gs_x, gs_y = 0, 0

                    for feature in features:
                        feat_payload = cluster_data[offset][feature]
                        if "space" in feature:
                            cbar_width = 0.005
                            cbar_height = 0.04
                            ratemap_2d = feat_payload.get(
                                "rate_smoothed", feat_payload["rate"]
                            )
                            ax1 = fig.add_subplot(gs[gs_x, gs_y])
                            rm = ax1.imshow(
                                X=ratemap_2d, cmap=ratemap_cmap, vmin=0,
                                interpolation="gaussian", aspect="equal",
                            )
                            ax1.set_title(
                                "Spatial tuning", fontsize=13, pad=8.0,
                                fontweight="bold",
                            )
                            ax1.set_xticks([])
                            ax1.set_yticks([])
                            ax1.set_xlabel("X (cm)", fontsize=10, labelpad=1)
                            ax1.set_ylabel("Y (cm)", fontsize=10, labelpad=1)

                            ax1_pos = gs[gs_x, gs_y].get_position(fig)
                            cb_ax = fig.add_axes(
                                (ax1_pos.x1, ax1_pos.y1 - cbar_height,
                                 cbar_width, cbar_height)
                            )
                            cbar = fig.colorbar(rm, cax=cb_ax, orientation="vertical")
                            cbar_vmin, cbar_vmax = cbar.mappable.get_clim()
                            cbar.set_ticks([cbar_vmin, cbar_vmax])
                            cbar.set_ticklabels(
                                [f"{int(cbar_vmin)}", f"{round(cbar_vmax, 1)}"],
                                fontsize=7,
                            )
                            cbar.ax.tick_params(length=0, pad=0.5)
                            cbar.outline.set_visible(True)

                            ax2 = fig.add_subplot(gs[gs_x, gs_y + 1])
                            occ = ax2.imshow(
                                X=feat_payload["occupancy_seconds"],
                                cmap=mouse_colormap, vmin=0,
                                interpolation="gaussian", aspect="equal",
                            )
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                            ax2.set_xlabel("X (cm)", fontsize=10, labelpad=1)
                            ax2.set_ylabel("Y (cm)", fontsize=10, labelpad=1)

                            ax2_pos = gs[gs_x, gs_y + 1].get_position(fig)
                            cb_ax2 = fig.add_axes(
                                (ax2_pos.x1, ax2_pos.y1 - cbar_height,
                                 cbar_width, cbar_height)
                            )
                            cbar2 = fig.colorbar(occ, cax=cb_ax2, orientation="vertical")
                            cbar2_vmin, cbar2_vmax = cbar2.mappable.get_clim()
                            cbar2.set_ticks([cbar2_vmin, cbar2_vmax])
                            cbar2.set_ticklabels(
                                [f"{int(cbar2_vmin)}", f"{int(np.ceil(cbar2_vmax))}"],
                                fontsize=7,
                            )
                            cbar2.ax.tick_params(length=0, pad=0.5)
                            cbar2.outline.set_visible(True)
                            gs_y += 2
                            if gs_y > 5:
                                gs_y = 0; gs_x += 1
                        else:
                            occ_mask = (
                                feat_payload["occupancy_seconds"] > occ_threshold_setting
                            )
                            low_end_sh = feat_payload.get(
                                "null_p0_5_smoothed", feat_payload["null_p0_5"]
                            )
                            high_end_sh = feat_payload.get(
                                "null_p99_5_smoothed", feat_payload["null_p99_5"]
                            )
                            ratemap = feat_payload.get(
                                "rate_smoothed", feat_payload["rate"]
                            )
                            bin_edges = feat_payload["bin_edges"]
                            bin_centers = feat_payload["bin_centers"]

                            # raw orofacial/anogenital SEI are the only
                            # social features that exist in both directions
                            # (AtoB and BtoA); disambiguate those two with a
                            # directional mouse-index pair in the title.
                            # Their 1st / 2nd derivatives are also stored
                            # twice but should NOT receive the suffix per
                            # convention; all other social features are
                            # bidirectional (one entry only) and stay as-is.
                            prefix_str, feat_name = feature.split(".", 1)
                            directional_sei = {"orofacial-sei", "anogenital-sei"}
                            if "-" in prefix_str and feat_name in directional_sei:
                                m_a, m_b = prefix_str.split("-", 1)
                                try:
                                    idx_a = mouse_id_list.index(m_a) + 1
                                    idx_b = mouse_id_list.index(m_b) + 1
                                    # the bundled Helvetica.ttf has no
                                    # arrow glyphs (U+2192 etc.), so render
                                    # the arrow via matplotlib mathtext.
                                    title_text = (
                                        rf"{feat_name} ({idx_a}$\rightarrow${idx_b})"
                                    )
                                except ValueError:
                                    title_text = feat_name
                            else:
                                title_text = feat_name

                            ax1 = fig.add_subplot(gs[gs_x, gs_y])
                            # split x/y tick params: push xticklabels down
                            # so the leftmost xtick label no longer kisses
                            # the bottom yticklabel; keep yticks tight.
                            ax1.tick_params(axis="x", length=1.5, pad=4.0, labelsize=10)
                            ax1.tick_params(axis="y", length=1.5, pad=0.5, labelsize=10)
                            ax1.fill_between(
                                bin_centers[occ_mask],
                                low_end_sh[occ_mask],
                                high_end_sh[occ_mask],
                                where=high_end_sh[occ_mask] >= low_end_sh[occ_mask],
                                facecolor=COLOR_GRAY_BAND,
                                interpolate=True,
                            )
                            ax1.plot(
                                bin_centers[occ_mask],
                                ratemap[occ_mask],
                                lw=4.0, ls="-", c=rm_color, alpha=1.0,
                            )
                            ax1.set_title(
                                title_text, fontsize=13, pad=8.0,
                                fontweight="bold",
                            )
                            tx_min, tx_max = ax1.get_xlim()
                            x1_lo = max(
                                self.feature_boundaries[feat_name][0],
                                tx_min,
                            )
                            x1_hi = min(
                                self.feature_boundaries[feat_name][1],
                                tx_max,
                            )
                            ax1.set_xticks(
                                ticks=[x1_lo, x1_hi],
                                labels=[f"{x1_lo:.1f}", f"{x1_hi:.1f}"],
                                fontsize=10,
                            )
                            ax1.set_xlabel(
                                self.feature_labels[plot_feature_key.split(".")[0]][
                                    feat_name
                                ],
                                fontsize=10, labelpad=2,
                            )
                            ty_min, ty_max = ax1.get_ylim()
                            ax1.set_yticks(
                                ticks=[max(np.floor(ty_min), 0), np.ceil(ty_max)],
                                labels=[
                                    f"{max(np.floor(ty_min), 0):.1f}",
                                    f"{ty_max:.1f}",
                                ],
                                fontsize=10,
                            )
                            ax1.set_ylabel(
                                "Firing rate (sp/s)", fontsize=10, labelpad=-2,
                            )
                            ax1.set_box_aspect(1)

                            ax2 = fig.add_subplot(gs[gs_x, gs_y + 1])
                            ax2.tick_params(axis="x", length=1.5, pad=4.0, labelsize=10)
                            ax2.tick_params(axis="y", length=1.5, pad=0.5, labelsize=10)
                            ax2.stairs(
                                values=feat_payload["occupancy_seconds"],
                                edges=bin_edges,
                                fill=True,
                                color=rm_color,
                                edgecolor=COLOR_BLACK,
                                linewidth=0.3,
                            )
                            ax2.set_xticks(
                                ticks=[
                                    self.feature_boundaries[feat_name][0],
                                    self.feature_boundaries[feat_name][1],
                                ],
                                labels=[
                                    f"{self.feature_boundaries[feat_name][0]:.1f}",
                                    f"{self.feature_boundaries[feat_name][1]:.1f}",
                                ],
                                fontsize=10,
                            )
                            ax2.set_xlabel(
                                self.feature_labels[plot_feature_key.split(".")[0]][
                                    feat_name
                                ],
                                fontsize=10, labelpad=2,
                            )
                            _, ty2_max = ax2.get_ylim()
                            ax2.set_yticks(
                                ticks=[0, int(np.ceil(ty2_max))],
                                labels=["0", f"{int(np.ceil(ty2_max))}"],
                                fontsize=10,
                            )
                            ax2.set_ylabel(
                                "Occupancy (s)", fontsize=10, labelpad=-2,
                            )
                            ax2.set_box_aspect(1)
                            gs_y += 2
                            if gs_y > 5:
                                gs_y = 0; gs_x += 1

                save_fig(fig, f"behavioral_{offset}_{plot_feature_key}")

    def _render_vocal_pages(
        self,
        cluster_data: dict,
        usv_summary_df,
        segmentation: dict,
        viz_params: dict,
        save_fig,
    ) -> None:
        """
        Description
        -----------
        For each emitter present in the pkl, render Page 1 (sections a +
        b) and Page 2 (sections c + d) via the supplied `save_fig`
        callable.

        Parameters
        ----------
        cluster_data (dict)
            Loaded pkl payload with keys usv_peth, usv_property_tuning,
            usv_category_tuning, usv_category_peth, usv_metadata.
        usv_summary_df (pls.DataFrame | None)
            Filtered (post `vae_supercategory != 0`) USV summary if
            available; required for the bout raster.
        segmentation (dict)
            Loaded latent-embedding segmentation per categorical feature.
        viz_params (dict)
            `neuronal_tuning_figures` settings block.
        save_fig (Callable[[Figure, str], None])
            Persists each rendered page; closes the figure.

        Returns
        -------
        None
        """

        for emitter in cluster_data["usv_peth"]:
            sex_label = cluster_data["usv_peth"][emitter].get("sex", "x")
            self._render_page1(
                emitter=emitter,
                cluster_data=cluster_data,
                usv_summary_df=usv_summary_df,
                viz_params=viz_params,
                save_fig=save_fig,
                page_label=f"vocal_a_{sex_label}",
            )
            self._render_page2(
                emitter=emitter,
                cluster_data=cluster_data,
                segmentation=segmentation,
                viz_params=viz_params,
                save_fig=save_fig,
                page_label=f"vocal_b_{sex_label}",
            )

    # Page 1 — sections (a) and (b)

    def _render_page1(
        self,
        emitter: str,
        cluster_data: dict,
        usv_summary_df,
        viz_params: dict,
        save_fig,
        page_label: str,
    ) -> None:
        """
        Description
        -----------
        Render Page 1 of the vocal output for one emitter side: the
        bout raster + pooled `usv_peth` (section a) on top of the
        4×4 `usv_property_tuning` grid (section b), then commit the
        page via `save_fig`.

        Parameters
        ----------
        emitter (str)
            Emitter ID (e.g. mouse track name) keying into the vocal
            payloads.
        cluster_data (dict)
            Loaded per-cluster pkl payload.
        usv_summary_df (pls.DataFrame | None)
            Filtered USV summary (post `vae_supercategory != 0`); used
            by the bout raster. Pass `None` to skip the raster.
        viz_params (dict)
            `neuronal_tuning_figures` settings block.
        save_fig (Callable[[Figure, str], None])
            Persists the rendered figure and closes it.
        page_label (str)
            Short label embedded in non-PDF per-page filenames.

        Returns
        -------
        None
        """

        fig = plt.figure(figsize=VOCAL_PAGE1_FIGSIZE_INCHES, tight_layout=False)
        outer = gridspec.GridSpec(
            nrows=2, ncols=1, height_ratios=[5, 14], hspace=0.30,
            left=0.06, right=0.97, top=0.96, bottom=0.04,
        )

        # section (a)
        gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.18)
        ax_raster = fig.add_subplot(gs_a[0, 0])
        ax_peth = fig.add_subplot(gs_a[0, 1])
        self._draw_section_a(
            ax_raster=ax_raster,
            ax_peth=ax_peth,
            emitter=emitter,
            cluster_data=cluster_data,
            usv_summary_df=usv_summary_df,
        )

        # section (b) — 4x4 grid of square usv_property_tuning cells
        gs_b = gridspec.GridSpecFromSubplotSpec(
            4, 4, subplot_spec=outer[1, 0], wspace=0.40, hspace=0.45
        )
        self._draw_section_b(
            fig=fig,
            gs=gs_b,
            emitter=emitter,
            cluster_data=cluster_data,
        )

        save_fig(fig, page_label)

    # section (a) — raster + PETH

    def _draw_section_a(
        self,
        ax_raster,
        ax_peth,
        emitter: str,
        cluster_data: dict,
        usv_summary_df,
    ) -> None:
        """
        Description
        -----------
        Draw section (a) of vocal Page 1: bout raster on the left axes
        (per-bout spike raster + per-USV color bars), pooled `usv_peth`
        on the right axes (peri-USV PETH with shuffle band and
        observed line).

        Parameters
        ----------
        ax_raster (matplotlib.axes.Axes)
            Target axes for the bout raster (left).
        ax_peth (matplotlib.axes.Axes)
            Target axes for the PETH (right).
        emitter (str)
            Emitter ID keying into the vocal payloads.
        cluster_data (dict)
            Per-cluster pkl payload.
        usv_summary_df (pls.DataFrame | None)
            Filtered USV summary; if `None` the raster is skipped.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_peth"][emitter].get("sex", "male")
        line_color = self._sex_color(sex)

        # left: bout raster
        if usv_summary_df is None:
            ax_raster.text(
                0.5, 0.5, "USV summary CSV not available\n(raster omitted)",
                transform=ax_raster.transAxes, ha="center", va="center", fontsize=13, color=COLOR_GRAY_DASH,
            )
            ax_raster.set_xticks([])
            ax_raster.set_yticks([])
        else:
            self._draw_bout_raster(
                ax=ax_raster,
                emitter=emitter,
                usv_summary_df=usv_summary_df,
                cluster_data=cluster_data,
            )

        # right: PETH
        peth = cluster_data["usv_peth"][emitter]
        bin_centers = peth["bin_centers_s"]
        # smoothed versions are pre-computed in compute (smooth-then-percentile);
        # fall back to raw if smoothing was disabled (smoothing_sd == 0).
        rate = peth.get("rate_smoothed", peth["rate"])
        p0_5 = peth.get("null_p0_5_smoothed", peth["null_p0_5"])
        p99_5 = peth.get("null_p99_5_smoothed", peth["null_p99_5"])

        ax_peth.fill_between(
            bin_centers,
            p0_5,
            p99_5,
            where=np.isfinite(p0_5) & np.isfinite(p99_5),
            facecolor=COLOR_GRAY_BAND,
            interpolate=True,
            zorder=1,
        )
        ax_peth.plot(
            bin_centers,
            rate,
            lw=5.6,
            color=line_color,
            zorder=2,
        )
        ax_peth.axvline(x=0.0, color=COLOR_BLACK, ls="--", lw=0.5)
        ax_peth.set_xlim(bin_centers.min(), bin_centers.max())
        # explicit X ticks at 0.25 s spacing so 0 s is always visible
        peth_xticks = np.arange(-2.0, 0.0001, 0.25)
        ax_peth.set_xticks(peth_xticks)
        ax_peth.set_xticklabels([f"{v:g}" for v in peth_xticks])
        ax_peth.set_xlabel("time relative to USV onset (s)", fontsize=12)
        ax_peth.set_ylabel("firing rate (sp/s)", fontsize=12)
        n_anchors = peth.get("n_anchors", 0)
        ax_peth.set_title(f"PETH (n_USVs = {n_anchors})", fontsize=13, pad=3)
        ax_peth.tick_params(labelsize=11)

    def _draw_bout_raster(
        self,
        ax,
        emitter: str,
        usv_summary_df,
        cluster_data: dict,
    ) -> None:
        """
        Description
        -----------
        Draw the bout-onset-aligned raster on `ax`. Each bout (defined as
        a same-emitter USV preceded by ≥ `bout_quiet_seconds` of silence)
        becomes one row showing every USV (any emitter) within
        [-2, +2] s of the bout onset, plus this cluster's spike events.

        Parameters
        ----------
        ax (matplotlib axis)
            Target axis.
        emitter (str)
            Anchor-emitter string for bout definition.
        usv_summary_df (pls.DataFrame)
            Filtered USV summary.
        cluster_data (dict)
            Loaded vocal tuning pkl — needed for spike times and metadata.
            The bout-quiet-seconds setting is read from
            `cluster_data["usv_metadata"]["bout_quiet_seconds"]` so
            that the raster's bout definition matches what compute used.

        Returns
        -------
        None
        """

        bout_quiet_s = cluster_data.get("usv_metadata", {}).get("bout_quiet_seconds", 2.0)

        # find bouts of the anchor emitter
        emitters = [
            (e.strip() if e is not None else None)
            for e in usv_summary_df["emitter"].to_list()
        ]
        starts = usv_summary_df["start"].to_numpy()
        stops = usv_summary_df["stop"].to_numpy()
        anchor_mask = np.array([e == emitter for e in emitters])
        anchor_starts = starts[anchor_mask]
        anchor_stops = stops[anchor_mask]
        if anchor_starts.size == 0:
            ax.text(
                0.5, 0.5, "no anchor USVs",
                transform=ax.transAxes, ha="center", va="center", fontsize=13, color=COLOR_GRAY_DASH,
            )
            return

        order = np.argsort(anchor_starts)
        anchor_starts = anchor_starts[order]
        anchor_stops = anchor_stops[order]
        is_new_bout = np.empty(anchor_starts.size, dtype=bool)
        is_new_bout[0] = True
        is_new_bout[1:] = (anchor_starts[1:] - anchor_stops[:-1]) >= bout_quiet_s
        bout_onsets = anchor_starts[is_new_bout]

        if bout_onsets.size == 0:
            ax.text(
                0.5, 0.5, "no bouts", transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color=COLOR_GRAY_DASH,
            )
            return

        # spike data: the spike file path is not stored in the pkl;
        # locate the cluster's spike .npy under <session_root>/ephys
        # via session_root + cluster_id stored in metadata.
        spike_path_hint = cluster_data.get("usv_metadata", {}).get("session_root", None)
        if spike_path_hint is not None:
            cluster_id = cluster_data.get("usv_metadata", {}).get("cluster_id", None)
            if cluster_id:
                spike_npy_candidates = list(
                    pathlib.Path(spike_path_hint).rglob(f"cluster_data/{cluster_id}.npy")
                )
                if spike_npy_candidates:
                    arr = np.load(spike_npy_candidates[0])
                    spike_times = np.sort(np.asarray(arr[0, :], dtype=float))
                else:
                    spike_times = np.empty(0, dtype=float)
            else:
                spike_times = np.empty(0, dtype=float)
        else:
            spike_times = np.empty(0, dtype=float)

        x_lo, x_hi = -2.0, 2.0

        # USV-bar color: anchor's own sex for anchor-emitter USVs, the
        # opposite sex for any other emitter's USVs falling in the window.
        anchor_sex = cluster_data["usv_peth"][emitter].get("sex", "male")
        anchor_partner_sex = "female" if anchor_sex == "male" else "male"

        # for each bout, draw spikes (black) and USV-bars (sex-colored)
        # within [-2, +2] of the bout onset
        all_starts = starts
        all_stops = stops
        spike_y = []
        spike_x = []
        for row_idx, t0 in enumerate(bout_onsets):
            # USVs in window:
            mask = (all_stops > t0 + x_lo) & (all_starts < t0 + x_hi)
            for j in np.where(mask)[0]:
                s = max(all_starts[j] - t0, x_lo)
                e = min(all_stops[j] - t0, x_hi)
                em = emitters[j]
                if em == emitter:
                    color = self._sex_color(anchor_sex)
                else:
                    color = self._sex_color(anchor_partner_sex) if em is not None else COLOR_LIGHT
                # USV bars at the TOP of each row (just inside the upper edge);
                # spikes occupy the body of the row centered at row_idx + 0.5.
                # plot only anchor-emitter USVs (sex-colored) and
                # unassigned USVs (gray); skip partner-emitter USVs
                # entirely so the raster reflects this side's vocal
                # activity uncluttered by the other animal's calls.
                if em is not None and em != emitter:
                    continue
                ax.hlines(y=row_idx + 0.95, xmin=s, xmax=e, colors=color, linewidth=1.5)

            # spikes in window:
            if spike_times.size > 0:
                lo = np.searchsorted(spike_times, t0 + x_lo, "left")
                hi = np.searchsorted(spike_times, t0 + x_hi, "left")
                rel = spike_times[lo:hi] - t0
                spike_x.extend(rel.tolist())
                spike_y.extend([row_idx + 0.5] * rel.size)

        if spike_x:
            ax.vlines(
                x=spike_x, ymin=np.array(spike_y) - 0.4, ymax=np.array(spike_y) + 0.4,
                colors=COLOR_BLACK, linewidth=0.15,
            )

        ax.axvline(x=0.0, color=COLOR_BLACK, ls="--", lw=0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.25, bout_onsets.size + 0.25)
        ax.set_xlabel("time relative to bout onset (s)", fontsize=12)
        ax.set_ylabel("bout #", fontsize=12)
        ax.set_title(f"raster (n_bouts = {bout_onsets.size})", fontsize=13, pad=3)
        ax.tick_params(labelsize=11)

    # section (b) — within-USV continuous tuning grid

    def _draw_section_b(
        self,
        fig,
        gs,
        emitter: str,
        cluster_data: dict,
    ) -> None:
        """
        Description
        -----------
        Draw section (b) of vocal Page 1: a 4×4 grid of
        (line + occupancy) cell pairs, one row per pair of continuous
        USV properties (per `PROPERTY_ROW_ORDER`). Dispatches to
        `_draw_property_pair` for each cell pair.

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Parent figure (used to add subplots).
        gs (matplotlib.gridspec.GridSpecFromSubplotSpec)
            4×4 gridspec slot for section (b).
        emitter (str)
            Emitter ID keying into `usv_property_tuning`.
        cluster_data (dict)
            Per-cluster pkl payload.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_property_tuning"][emitter][CONTINUOUS_PROPERTIES[0]].get("sex", "male")
        line_color = self._sex_color(sex)

        for row_idx, prop_pair in enumerate(PROPERTY_ROW_ORDER):
            for pair_idx, prop in enumerate(prop_pair):
                ax_line = fig.add_subplot(gs[row_idx, 2 * pair_idx])
                ax_occ = fig.add_subplot(gs[row_idx, 2 * pair_idx + 1])
                self._draw_property_pair(
                    ax_line=ax_line,
                    ax_occ=ax_occ,
                    prop=prop,
                    cluster_payload=cluster_data["usv_property_tuning"][emitter][prop],
                    line_color=line_color,
                )

    def _draw_property_pair(
        self,
        ax_line,
        ax_occ,
        prop: str,
        cluster_payload: dict,
        line_color: str,
    ) -> None:
        """
        Description
        -----------
        Draw one (line, occupancy) cell pair for the section-(b) grid.
        The line axes shows observed firing rate (smoothed if
        available) plus the smoothed shuffle band; the occupancy axes
        shows a step-filled per-bin occupancy histogram. The x-axis is
        trimmed to the non-zero-occupancy range (with a 5% margin on
        each side) and given two ticks (min, max) of the data extent;
        the y-axis is given two ticks (floor, ceil) over the joint
        observed + shuffle range, also with a small auto-margin.

        Parameters
        ----------
        ax_line (matplotlib.axes.Axes)
            Target axes for the rate-vs-property line plot.
        ax_occ (matplotlib.axes.Axes)
            Target axes for the per-property occupancy histogram.
        prop (str)
            One of `CONTINUOUS_PROPERTIES`.
        cluster_payload (dict)
            The `usv_property_tuning[emitter][prop]` block.
        line_color (str)
            Hex color for the observed-rate line and occupancy fill;
            typically the per-emitter sex color.

        Returns
        -------
        None
        """

        f = DISPLAY_FACTOR[prop]
        centers = cluster_payload["bin_centers"] * f
        edges = cluster_payload["bin_edges"] * f
        # smoothed versions are pre-computed in compute; fall back to raw
        # if smoothing was disabled (smoothing_sd == 0).
        rate = cluster_payload.get("rate_smoothed", cluster_payload["rate"])
        occ = cluster_payload["occupancy_seconds"]
        p0_5 = cluster_payload.get("null_p0_5_smoothed", cluster_payload["null_p0_5"])
        p99_5 = cluster_payload.get("null_p99_5_smoothed", cluster_payload["null_p99_5"])

        # Firing rates are non-negative by construction (counts /
        # occupancy, both >= 0) and Gaussian smoothing preserves that;
        # clamp the shuffle floor at zero anyway to insulate the
        # rendered band from any negative values that future numerics
        # / smoothing changes could introduce.
        p0_5 = np.maximum(p0_5, 0.0)

        # Natural bounds: trim x-axis to bins where there is actually
        # something to render. The compute side sets `rate=NaN` (and the
        # shuffle stats follow) whenever a bin's occupancy is below
        # `usv_property_min_occupancy_seconds`. An earlier revision used
        # `occ > 0` to pick the visible bins, which accepted outlier
        # bins (tiny non-zero occupancy, NaN rate / NaN shuffle band) at
        # the far end of the property range and produced an x-axis
        # extending well past the rightmost rendered point. Filter by
        # `isfinite(rate)` AND the shuffle band so the displayed extent
        # matches the visible data extent exactly.
        #
        # CRITICAL — use bin CENTERS (not bin edges) for both xlim and
        # tick positions. Different properties have different numbers of
        # occupied bins, so edge-based extents pick up a half-binwidth
        # offset that varies per plot, which breaks visual buffer
        # uniformity. Center-based extents with a percent margin give a
        # constant 5%-on-each-side buffer regardless of bin width
        # (matches the matplotlib auto-margin that behavioral plots use).
        visible = (
            np.isfinite(rate) & np.isfinite(p0_5) & np.isfinite(p99_5)
        )
        nz = np.where(visible)[0]
        if nz.size > 0:
            sl = slice(int(nz[0]), int(nz[-1]) + 1)
            data_x_lo = float(centers[nz[0]])
            data_x_hi = float(centers[nz[-1]])
        else:
            sl = slice(None)
            data_x_lo, data_x_hi = float(centers[0]), float(centers[-1])
        x_margin = 0.05 * (data_x_hi - data_x_lo) if data_x_hi > data_x_lo else 0.0
        x_lo_lim = data_x_lo - x_margin
        x_hi_lim = data_x_hi + x_margin

        ax_line.fill_between(
            centers[sl],
            p0_5[sl],
            p99_5[sl],
            where=np.isfinite(p0_5[sl]) & np.isfinite(p99_5[sl]),
            facecolor=COLOR_GRAY_BAND,
            interpolate=True,
        )
        ax_line.plot(centers[sl], rate[sl], lw=4.0, color=line_color)
        ax_line.set_xlim(x_lo_lim, x_hi_lim)
        # two ticks per axis, at the visible data extents
        ax_line.set_xticks(
            ticks=[data_x_lo, data_x_hi],
            labels=[f"{data_x_lo:.1f}", f"{data_x_hi:.1f}"],
            fontsize=11,
        )
        # y-buffer: read matplotlib's auto-fit ylim (already includes
        # a small auto-margin from fill_between + plot), then floor /
        # ceil to nearest integer (clamped at 0) so the line never sits
        # exactly on the top / bottom spine.
        ty_min, ty_max = ax_line.get_ylim()
        y_lo = max(float(np.floor(ty_min)), 0.0)
        y_hi = float(np.ceil(ty_max))
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
        ax_line.set_ylim(y_lo, y_hi)
        ax_line.set_yticks(
            ticks=[y_lo, y_hi],
            labels=[f"{y_lo:.1f}", f"{y_hi:.1f}"],
            fontsize=11,
        )
        ax_line.set_xlabel(self.vocal_labels[prop], fontsize=12, labelpad=2)
        ax_line.set_ylabel("Firing rate (sp/s)", fontsize=12, labelpad=-2)
        ax_line.tick_params(axis="x", labelsize=11, pad=4.0)
        ax_line.tick_params(axis="y", labelsize=11, pad=0.5)
        ax_line.set_title(
            prop.split(".")[-1], fontsize=15, pad=8.0,
            fontweight="bold",
        )
        ax_line.set_box_aspect(1)

        # stepfilled histogram-style for occupancy (matplotlib's stairs
        # primitive consumes pre-binned values + edges and renders the
        # equivalent of histtype='stepfilled' without needing the raw
        # samples).
        ax_occ.stairs(
            values=occ,
            edges=edges,
            fill=True,
            color=line_color,
            edgecolor=COLOR_BLACK,
            linewidth=0.4,
        )
        ax_occ.set_xlim(x_lo_lim, x_hi_lim)
        ax_occ.set_xticks(
            ticks=[data_x_lo, data_x_hi],
            labels=[f"{data_x_lo:.1f}", f"{data_x_hi:.1f}"],
            fontsize=11,
        )
        _, ty2_max = ax_occ.get_ylim()
        occ_max = max(int(np.ceil(float(ty2_max))), 1)
        ax_occ.set_yticks(
            ticks=[0, occ_max],
            labels=["0", f"{occ_max}"],
            fontsize=11,
        )
        ax_occ.set_xlabel(self.vocal_labels[prop], fontsize=12, labelpad=2)
        ax_occ.set_ylabel("Occupancy (s)", fontsize=12, labelpad=-2)
        ax_occ.tick_params(axis="x", labelsize=11, pad=4.0)
        ax_occ.tick_params(axis="y", labelsize=11, pad=0.5)
        ax_occ.set_box_aspect(1)

    # Page 2 — sections (c) and (d)

    def _render_page2(
        self,
        emitter: str,
        cluster_data: dict,
        segmentation: dict,
        viz_params: dict,
        save_fig,
        page_label: str,
    ) -> None:
        """
        Description
        -----------
        Render Page 2 of the vocal output for one emitter side: section
        (c) categorical watersheds (`usv_category_tuning` rate / occupancy
        / strip — 2 rows × 6 cols, paired by VAE / QLVM method) on top
        of section (d) `usv_category_peth` per-category PETH grid (flat
        sequential 6 cols / row, all-NaN entries skipped). Outer
        `height_ratios` adapts to the actual `usv_category_peth` row
        count so cells stay square in both sections.

        Parameters
        ----------
        emitter (str)
            Emitter ID keying into the vocal payloads.
        cluster_data (dict)
            Per-cluster pkl payload.
        segmentation (dict)
            Loaded latent-embedding segmentation per categorical feature.
        viz_params (dict)
            `neuronal_tuning_figures` settings block.
        save_fig (Callable[[Figure, str], None])
            Persists the rendered figure and closes it.
        page_label (str)
            Short label embedded in non-PDF per-page filenames.

        Returns
        -------
        None
        """

        fig = plt.figure(figsize=VOCAL_PAGE2_FIGSIZE_INCHES, tight_layout=False)

        # pre-count section (d) non-empty cells across all four
        # categorical features so we can size the gridspec to a flat
        # 6-col flow (same density / spacing as the behavioral grid).
        d_cols = 6
        total_d_cells = 0
        for cf in CATEGORICAL_FEATURES:
            payload_d = cluster_data["usv_category_peth"][emitter][cf]
            rate_arr_d = payload_d.get("rate_smoothed", payload_d["rate"])
            for i in range(rate_arr_d.shape[0]):
                if np.isfinite(rate_arr_d[i]).any():
                    total_d_cells += 1
        d_rows = max(1, (total_d_cells + d_cols - 1) // d_cols)

        # outer ratio reflects the row count of each section so the
        # cell height in (c) and (d) matches (both use 6 cols).
        outer = gridspec.GridSpec(
            nrows=2, ncols=1,
            height_ratios=[len(SECTION_C_ROWS), d_rows],
            hspace=0.12,
            left=0.04, right=0.98, top=0.94, bottom=0.05,
        )

        # section (c): 2 rows × 6 cols. Each row pairs the `_category`
        # (3 cols) and `_supercategory` (3 cols) variants of one
        # segmentation method. wspace/hspace match the behavioral grid.
        gs_c = gridspec.GridSpecFromSubplotSpec(
            len(SECTION_C_ROWS), 6, subplot_spec=outer[0, 0],
            wspace=0.40, hspace=0.45,
        )
        self._draw_section_c(
            fig=fig,
            gs=gs_c,
            emitter=emitter,
            cluster_data=cluster_data,
            segmentation=segmentation,
            viz_params=viz_params,
        )

        # section (d): flat sequential PETH flow at 6 cols / row,
        # ordered VAE cat to VAE supercat to QLVM cat to QLVM supercat;
        # all-NaN entries are skipped so the row stays packed.
        gs_d = gridspec.GridSpecFromSubplotSpec(
            d_rows, d_cols, subplot_spec=outer[1, 0],
            wspace=0.40, hspace=0.45,
        )
        self._draw_section_d(
            fig=fig,
            gs=gs_d,
            emitter=emitter,
            cluster_data=cluster_data,
            n_cols=d_cols,
        )

        save_fig(fig, page_label)

    # section (c) — categorical 2 rows × 6 cols

    def _draw_section_c(
        self,
        fig,
        gs,
        emitter: str,
        cluster_data: dict,
        segmentation: dict,
        viz_params: dict,
    ) -> None:
        """
        Description
        -----------
        Draw section (c): 2 rows × 6 cols. Each row pairs the
        `_category` (3 cols: rate watershed | occupancy watershed |
        strip) and `_supercategory` (3 cols, same structure) variants
        of one segmentation method (`SECTION_C_ROWS`). The rate
        watershed uses the configurable `ratemap_cmap`; the occupancy
        watershed uses a per-emitter sequential colormap built from
        the emitter's sex color; the strip plot ranks per-category
        observed firing rate against the shuffle band.

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Parent figure.
        gs (matplotlib.gridspec.GridSpecFromSubplotSpec)
            2×6 gridspec slot for section (c).
        emitter (str)
            Emitter ID keying into `usv_category_tuning`.
        cluster_data (dict)
            Per-cluster pkl payload.
        segmentation (dict)
            Per-categorical-feature watershed segmentation.
        viz_params (dict)
            `neuronal_tuning_figures` settings block; reads
            `ratemap_cmap`, `vocal_strip_log_ratio_threshold`,
            `vocal_strip_symlog_linthresh`.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_category_tuning"][emitter][SECTION_C_ROWS[0][0]].get("sex", "male")
        dot_color = self._sex_color(sex)

        log_threshold = VOCAL_STRIP_LOG_RATIO_THRESHOLD
        symlog_linthresh = VOCAL_STRIP_SYMLOG_LINTHRESH
        ratemap_cmap = self.visualizations_parameter_dict.get(
            "figures", {}
        ).get("cmap", "inferno")

        # build a sequential colormap for the occupancy watershed from
        # the emitter's sex color (mirrors the per-mouse colormap that
        # behavioral occupancy plots use).
        emitter_color = dot_color
        emitter_cmap = create_colormap(
            input_parameter_dict={
                "cm_length": 255,
                "cm_name": f"emitter_{emitter}",
                "cm_type": "sequential",
                "cm_start": (
                    int(emitter_color[1:3], 16),
                    int(emitter_color[3:5], 16),
                    int(emitter_color[5:7], 16),
                ),
                "cm_end": (255, 255, 255),
                "equalize_luminance": True,
                "match_luminance_by": "max",
                "change_saturation": 0.5,
                "cm_opacity": 1,
            }
        )

        for row_idx, row_feats in enumerate(SECTION_C_ROWS):
            for sub_idx, cat_feat in enumerate(row_feats):
                col_offset = sub_idx * 3
                payload = cluster_data["usv_category_tuning"][emitter][cat_feat]
                seg = segmentation.get(cat_feat)

                ax_rate = fig.add_subplot(gs[row_idx, col_offset + 0])
                ax_occ = fig.add_subplot(gs[row_idx, col_offset + 1])
                ax_strip = fig.add_subplot(gs[row_idx, col_offset + 2])

                # build a human-readable method label, e.g. vae_category
                # to "VAE category", qlvm_supercategory to "QLVM supercategory".
                method_token, granularity = cat_feat.split("_", 1)
                method_label = method_token.upper()
                tuning_title = f"{method_label} {granularity} tuning"

                # column N+0: firing rate watershed (configurable cmap)
                self._draw_categorical_watershed(
                    ax=ax_rate,
                    seg=seg,
                    categories=payload["categories"],
                    values=payload["rate"],
                    cbar_label="firing rate (sp/s)",
                    fig=fig,
                    cmap=ratemap_cmap,
                )
                ax_rate.set_title(tuning_title, fontsize=12, pad=3)

                # column N+1: occupancy watershed (per-emitter cmap), with
                # category IDs overlaid at each region's pole of
                # inaccessibility (distance-transform peak)
                self._draw_categorical_watershed(
                    ax=ax_occ,
                    seg=seg,
                    categories=payload["categories"],
                    values=payload["occupancy_count"].astype(float),
                    cbar_label="USV count",
                    fig=fig,
                    cmap=emitter_cmap,
                    annotate_categories=True,
                )
                ax_occ.set_title("Occupancy", fontsize=12, pad=3)

                # column N+2: per-category strip
                self._draw_categorical_strip(
                    ax=ax_strip,
                    payload=payload,
                    dot_color=dot_color,
                    log_threshold=log_threshold,
                    symlog_linthresh=symlog_linthresh,
                )
                ax_strip.set_title("Tuning vs Shuffled", fontsize=12, pad=3)

    def _draw_categorical_watershed(
        self,
        ax,
        seg: dict | None,
        categories,
        values: np.ndarray,
        cbar_label: str,
        fig,
        cmap,
        annotate_categories: bool = False,
    ) -> None:
        """
        Description
        -----------
        Render a watershed-style imshow of per-category values over the
        bundled latent-embedding segmentation grid. Pixels with no cluster-side
        category get a hatched fill; per-category 1-NN boundaries are
        contour-drawn in `COLOR_BLACK`. When `annotate_categories=True`,
        each region's category ID is overlaid at its centroid (or a
        circular-mean fallback for QLVM regions that wrap the embedding
        boundary). When `seg` is `None`, draws a placeholder text and
        returns.

        Parameters
        ----------
        ax (matplotlib.axes.Axes)
            Target axes.
        seg (dict | None)
            Segmentation block for this categorical feature
            (`label_grid`, `xx`, `yy`, `bounds`, `unique_labels`).
        categories (array-like)
            Category IDs present in the cluster's data.
        values (np.ndarray)
            Per-category value to color (firing rate or USV count).
        cbar_label (str)
            Colorbar label string.
        fig (matplotlib.figure.Figure)
            Parent figure (needed for the colorbar).
        cmap (str | matplotlib.colors.Colormap)
            Colormap name or instance.
        annotate_categories (bool)
            If True, overlay each category's ID at its region centroid.

        Returns
        -------
        None
        """

        if seg is None:
            ax.text(
                0.5, 0.5, "segmentation\nunavailable",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color=COLOR_GRAY_DASH,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            return

        label_grid = seg["label_grid"]
        x_min, x_max, y_min, y_max = seg["bounds"]
        xx, yy = seg["xx"], seg["yy"]

        # build a "value per pixel" image: NaN where the pixel's label is
        # missing from the cluster's category list (e.g., this cluster
        # never saw that category)
        cat_to_value = dict(
            zip(
                np.asarray(categories).tolist(),
                np.asarray(values).tolist(),
                strict=False,
            )
        )
        flat = label_grid.ravel()
        value_grid = np.array(
            [cat_to_value.get(int(l), np.nan) for l in flat],
            dtype=float,
        ).reshape(label_grid.shape)

        # shade: hatched gray where NaN; sequential colormap elsewhere
        finite = np.isfinite(value_grid)
        if finite.any():
            vmin = 0.0
            vmax = float(np.nanmax(value_grid))
            if vmax <= 0:
                vmax = 1.0
            im = ax.imshow(
                np.ma.masked_invalid(value_grid),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.tick_params(labelsize=7)
        # hatched fill for invalid pixels (under-supported categories)
        invalid_mask = ~finite
        if invalid_mask.any():
            ax.contourf(
                xx, yy, invalid_mask.astype(float),
                levels=[0.5, 1.5], colors=[COLOR_HATCH], alpha=0.6, hatches=["//"],
            )

        # 1-NN boundary contours
        # remap label_grid to dense indices for clean half-integer contours
        unique_labels_in_grid = np.unique(label_grid)
        label_to_dense = {int(l): i for i, l in enumerate(unique_labels_in_grid)}
        dense = np.vectorize(label_to_dense.get)(label_grid).astype(np.int32)
        ax.contour(
            xx, yy, dense,
            levels=np.arange(unique_labels_in_grid.size + 1) - 0.5,
            colors=COLOR_BLACK, linewidths=0.8,
        )

        # optionally annotate each category region with its category ID
        # at the centroid of the category mask. For non-toroidal
        # embeddings (VAE) the arithmetic mean of (xx, yy) over the mask
        # lands inside the blob and we use it. For toroidal embeddings
        # (QLVM) a category can wrap across the embedding boundary, in
        # which case the arithmetic centroid falls in the empty gap
        # between halves; we detect that case (closest pixel to the
        # arithmetic centroid is NOT inside the mask) and fall back to
        # the circular mean on each axis using period = bounds range.
        if annotate_categories:
            period_x = x_max - x_min
            period_y = y_max - y_min
            cluster_cats = set(int(c) for c in np.asarray(categories).tolist())
            for cat_label in unique_labels_in_grid:
                cat_id = int(cat_label)
                if cat_id not in cluster_cats:
                    continue
                mask = label_grid == cat_id
                if not mask.any():
                    continue
                masked_x = xx[mask]
                masked_y = yy[mask]
                cx = float(masked_x.mean())
                cy = float(masked_y.mean())
                # check whether the arithmetic centroid lands inside the
                # category mask (find nearest pixel and inspect its label)
                d_a = (xx - cx) ** 2 + (yy - cy) ** 2
                nearest = np.unravel_index(np.argmin(d_a), d_a.shape)
                if not mask[nearest] and period_x > 0 and period_y > 0:
                    # circular mean per axis (atan2 of mean sin / mean cos),
                    # then map back into [x_min, x_max] / [y_min, y_max]
                    theta_x = 2.0 * np.pi * (masked_x - x_min) / period_x
                    theta_y = 2.0 * np.pi * (masked_y - y_min) / period_y
                    cx_c = (
                        x_min
                        + np.arctan2(
                            np.sin(theta_x).mean(),
                            np.cos(theta_x).mean(),
                        )
                        * period_x / (2.0 * np.pi)
                    )
                    cy_c = (
                        y_min
                        + np.arctan2(
                            np.sin(theta_y).mean(),
                            np.cos(theta_y).mean(),
                        )
                        * period_y / (2.0 * np.pi)
                    )
                    cx_c = x_min + ((cx_c - x_min) % period_x)
                    cy_c = y_min + ((cy_c - y_min) % period_y)
                    d_c = (xx - cx_c) ** 2 + (yy - cy_c) ** 2
                    nearest_c = np.unravel_index(np.argmin(d_c), d_c.shape)
                    if mask[nearest_c]:
                        cx, cy = float(cx_c), float(cy_c)
                ax.text(
                    cx, cy, str(cat_id),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=COLOR_BLACK,
                    path_effects=[
                        mpe.withStroke(linewidth=2.0, foreground="white"),
                    ],
                )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    def _draw_categorical_strip(
        self,
        ax,
        payload: dict,
        dot_color: str,
        log_threshold: float,
        symlog_linthresh: float,
    ) -> None:
        """
        Description
        -----------
        Per-category strip plot — one row per category showing the
        observed firing rate (dot, colored by emitter sex) on top of
        the [p0.5, p99.5] shuffle band (gray rectangle). All-NaN
        categories are dropped (the row is removed); the x-axis spans
        `[min(data, shuffle) − buffer, max(data, shuffle) + buffer]`
        and switches to symlog when the dynamic range exceeds
        `log_threshold`.

        Parameters
        ----------
        ax (matplotlib.axes.Axes)
            Target axes.
        payload (dict)
            `usv_category_tuning[emitter][cat_feat]` block.
        dot_color (str)
            Hex color for observed-rate dots (per-emitter sex color).
        log_threshold (float)
            If `max / min > log_threshold` over positive values, use
            symlog x-scale.
        symlog_linthresh (float)
            Width of the linear region around 0 in symlog units.

        Returns
        -------
        None
        """

        cats_all = np.asarray(payload["categories"])
        rate_all = np.asarray(payload["rate"], dtype=float)
        p0_5_all = np.asarray(payload["null_p0_5"], dtype=float)
        p99_5_all = np.asarray(payload["null_p99_5"], dtype=float)

        # Defensive non-negative clamp on the shuffle floor — firing
        # rates are non-negative by construction but a hard clamp at
        # zero guarantees the rendered shuffle rectangle never crosses
        # below the firing-rate floor regardless of upstream changes.
        p0_5_all = np.maximum(p0_5_all, 0.0)

        # drop categories with no observed firing rate (insufficient data
        # in the cluster); we don't want their numeric label on the y-axis
        # if there is nothing to show next to it.
        keep = np.isfinite(rate_all)
        cats = cats_all[keep]
        rate = rate_all[keep]
        p0_5 = p0_5_all[keep]
        p99_5 = p99_5_all[keep]

        # decide x-axis scale on the kept (finite) values
        scale = _decide_strip_xscale(rate, p0_5, p99_5, log_threshold)

        n = cats.size
        ys = np.arange(n)
        for i in range(n):
            lo = p0_5[i] if np.isfinite(p0_5[i]) else 0.0
            hi = p99_5[i] if np.isfinite(p99_5[i]) else 0.0
            ax.add_patch(
                plt.Rectangle(
                    (min(lo, hi), i - 0.3),
                    abs(hi - lo),
                    0.6,
                    facecolor=COLOR_GRAY_BAND,
                    edgecolor="none",
                )
            )
        ax.scatter(
            rate, ys, s=18, color=dot_color, edgecolor=COLOR_BLACK,
            linewidths=0.4, zorder=3,
        )

        ax.set_yticks(ys)
        ax.set_yticklabels([str(int(c)) for c in cats], fontsize=10)
        if n > 0:
            ax.set_ylim(n - 0.5, -0.5)  # category 0 at top
        ax.set_xlabel(f"firing rate (sp/s) [{scale}]", fontsize=10)
        ax.tick_params(axis="x", labelsize=10)

        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=symlog_linthresh)
        # x-axis from joint min(observed, shuffle) − buffer to max + buffer
        # (no longer pinned to 0; firing-rate floor doesn't have to be
        # visible on every plot). For symlog, a LINEAR buffer near zero
        # is enormous in log space and visually separates the leftmost
        # data point from the spine by an order of magnitude or more —
        # use a multiplicative (geometric) buffer when both endpoints
        # are positive so the proportional log-space margin is 5% on
        # each side, matching the visual proportion of a linear axis.
        all_vals = np.concatenate(
            [
                rate[np.isfinite(rate)],
                p0_5[np.isfinite(p0_5)],
                p99_5[np.isfinite(p99_5)],
            ]
        )
        if all_vals.size > 0:
            x_lo_data = float(all_vals.min())
            x_hi_data = float(all_vals.max())
            # 10% buffer on each side (linear or log-space) so the
            # leftmost / rightmost observed-rate dots never sit on the
            # spine. Earlier 5% buffers occasionally placed the dot
            # essentially on the axis edge, which looked clipped.
            if (
                scale == "symlog"
                and x_lo_data > 0
                and x_hi_data > x_lo_data
            ):
                pad = (x_hi_data / x_lo_data) ** 0.10
                x_lo_lim = x_lo_data / pad
                x_hi_lim = x_hi_data * pad
            else:
                x_range = x_hi_data - x_lo_data
                buffer_x = (
                    0.10 * x_range if x_range > 0
                    else max(0.10 * abs(x_hi_data), 0.10)
                )
                x_lo_lim = x_lo_data - buffer_x
                x_hi_lim = x_hi_data + buffer_x
            ax.set_xlim(x_lo_lim, x_hi_lim)
            # Two explicit ticks at the data extrema: matplotlib's auto-
            # ticker (especially with symlog) places several ticks that
            # overlap one another given the small panel size. Two ticks
            # at min / max guarantee legibility regardless of scale.
            # `.3g` keeps the labels concise across magnitudes (0.5 stays
            # "0.5", 120 stays "120", 0.005 stays "0.005").
            ax.set_xticks(
                ticks=[x_lo_data, x_hi_data],
                labels=[f"{x_lo_data:.3g}", f"{x_hi_data:.3g}"],
                fontsize=10,
            )
            # Suppress any minor-tick labels symlog might add — keep just
            # the two majors we just set.
            ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_box_aspect(1)

    # section (d) — per-category PETH grid (tiny)

    def _draw_section_d(
        self,
        fig,
        gs,
        emitter: str,
        cluster_data: dict,
        n_cols: int,
    ) -> None:
        """
        Description
        -----------
        Draw section (d) of vocal Page 2: a flat sequential
        `usv_category_peth` PETH grid at `n_cols` cols/row. Items flow
        VAE category to VAE supercategory to QLVM category to
        QLVM supercategory; each non-NaN per-category PETH gets one
        cell. Each cell has its x-axis tightened to its own finite
        data extent (left), with the right edge anchored at t = 0.
        Two ticks per axis; lines doubled in width for visibility;
        per-cell title formatted "{METHOD} {cat|supercat} #{N}".

        Parameters
        ----------
        fig (matplotlib.figure.Figure)
            Parent figure.
        gs (matplotlib.gridspec.GridSpecFromSubplotSpec)
            (`ceil(total_items/n_cols)`, `n_cols`) gridspec slot.
        emitter (str)
            Emitter ID keying into `usv_category_peth`.
        cluster_data (dict)
            Per-cluster pkl payload.
        n_cols (int)
            Number of columns per row in the flat flow.

        Returns
        -------
        None
        """

        sex = cluster_data["usv_category_peth"][emitter][CATEGORICAL_FEATURES[0]].get("sex", "male")
        line_color = self._sex_color(sex)

        # build a flat sequential list of (title, rate, p0_5, p99_5,
        # bin_centers) tuples in the order VAE cat to VAE supercat to
        # QLVM cat to QLVM supercat, dropping all-NaN entries so the
        # display row stays packed.
        items: list[dict] = []
        for cat_feat in CATEGORICAL_FEATURES:
            method_token, granularity = cat_feat.split("_", 1)
            method_label = method_token.upper()
            short_gran = "supercat" if "super" in granularity else "cat"
            title_prefix = f"{method_label} {short_gran}"

            payload = cluster_data["usv_category_peth"][emitter][cat_feat]
            cats = np.asarray(payload["categories"])
            bin_centers = payload["bin_centers_s"]
            # smoothed versions are pre-computed in compute; fall back
            # to raw if smoothing was disabled (smoothing_sd == 0).
            rate_arr = payload.get("rate_smoothed", payload["rate"])
            p0_5_arr = payload.get("null_p0_5_smoothed", payload["null_p0_5"])
            p99_5_arr = payload.get("null_p99_5_smoothed", payload["null_p99_5"])
            for i in range(cats.size):
                rate = rate_arr[i, :]
                if not np.isfinite(rate).any():
                    continue
                items.append(
                    {
                        "title": f"{title_prefix} #{int(cats[i])}",
                        "rate": rate,
                        "p0_5": p0_5_arr[i, :],
                        "p99_5": p99_5_arr[i, :],
                        "bin_centers": bin_centers,
                    }
                )

        for display_idx, item in enumerate(items):
            row = display_idx // n_cols
            col = display_idx % n_cols
            rate = item["rate"]
            p0_5 = item["p0_5"]
            p99_5 = item["p99_5"]
            bin_centers = item["bin_centers"]

            # Defensive non-negative clamp on the shuffle floor (see
            # property-tuning cell for the same rationale).
            p0_5 = np.maximum(p0_5, 0.0)

            # tighten x-axis to the finite-rate extent of THIS cell;
            # 5% margin on the left, with the right side anchored at 0
            # so the anchor reference always shows up as a tick.
            finite = np.isfinite(rate)
            nz = np.where(finite)[0]
            x_lo_cell = float(bin_centers[nz[0]])
            x_hi_cell = 0.0
            cell_range = x_hi_cell - x_lo_cell
            cell_margin = (
                0.05 * cell_range if cell_range > 0
                else max(0.05 * abs(x_lo_cell), 0.05)
            )

            # Extend the rendered curve / shuffle band by a single
            # synthetic point at the anchor (t=0) repeating the last
            # finite value, so the visible data reaches the right axis
            # bound. Without this the line stops at `bin_centers[-1]`
            # (half-a-binwidth left of the anchor) and leaves a visual
            # air gap between the data and the t=0 tick.
            last_idx = int(nz[-1])
            bin_centers_ext = np.append(bin_centers, 0.0)
            rate_ext = np.append(rate, rate[last_idx])
            p0_5_last = p0_5[last_idx] if np.isfinite(p0_5[last_idx]) else 0.0
            p99_5_last = p99_5[last_idx] if np.isfinite(p99_5[last_idx]) else 0.0
            p0_5_ext = np.append(p0_5, p0_5_last)
            p99_5_ext = np.append(p99_5, p99_5_last)

            ax = fig.add_subplot(gs[row, col])
            ax.fill_between(
                bin_centers_ext, p0_5_ext, p99_5_ext,
                where=np.isfinite(p0_5_ext) & np.isfinite(p99_5_ext),
                facecolor=COLOR_GRAY_BAND, interpolate=True,
            )
            ax.plot(bin_centers_ext, rate_ext, lw=3.2, color=line_color)
            ax.set_xlim(x_lo_cell - cell_margin, x_hi_cell + cell_margin)
            ax.set_xticks(
                ticks=[x_lo_cell, 0.0],
                labels=[f"{x_lo_cell:.1f}", "0"],
                fontsize=8,
            )
            # y-axis: matplotlib's auto-fit ylim already adds a small
            # margin from fill_between + plot; floor / ceil to int (with
            # min clamped at 0) gives the same buffer treatment as the
            # continuous-feature plots.
            ty_min, ty_max = ax.get_ylim()
            y_lo = max(float(np.floor(ty_min)), 0.0)
            y_hi = float(np.ceil(ty_max))
            if y_hi <= y_lo:
                y_hi = y_lo + 1.0
            ax.set_ylim(y_lo, y_hi)
            ax.set_yticks(
                ticks=[y_lo, y_hi],
                labels=[f"{y_lo:.1f}", f"{y_hi:.1f}"],
                fontsize=8,
            )
            ax.tick_params(axis="x", labelsize=8, length=1.5, pad=2.0)
            ax.tick_params(axis="y", labelsize=8, length=1.5, pad=0.5)
            ax.set_xlabel("Pre-USV time (s)", fontsize=10, labelpad=2)
            ax.set_ylabel("Firing rate (sp/s)", fontsize=10, labelpad=-2)
            ax.set_title(item["title"], fontsize=11, pad=2)
            ax.set_box_aspect(1)
