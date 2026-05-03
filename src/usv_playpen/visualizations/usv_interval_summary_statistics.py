"""
@author: bartulem
Cross-session inter-vocalization-interval (inter-USV interval) aggregation and
plotting helpers.

These helpers are thin wrappers around :mod:`usv_playpen.analyses.gmm_utils`
intended to keep the inter-USV interval notebook declarative. Plot functions follow
the convention used in :mod:`usv_playpen.visualizations.usv_summary_statistics`:
they return ``(fig, ax, stats_dict)``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from matplotlib.transforms import offset_copy

from ..analyses.compute_inter_usv_interval_distributions import (
    _read_session_lists,
    _session_source_map,
    compute_session_usv_intervals,
    fit_gmm_sweep,
)
from ..analyses.gmm_utils import (
    TMixture,
    bootstrap_lrt,
    gmm_quantile_logspace,
    plot_gmm_fit,
    select_n_components_step_up_lrt,
    summarize_best_gmm,
    summarize_best_t_mixture,
    t_mixture_quantile_logspace,
)
from ..analyses.usv_interval_archive import (
    git_sha_for_provenance,
    read_usv_interval_h5,
    reconstruct_best_model,
    write_ivi_h5,
)
from ..os_utils import configure_path


def build_master_usv_interval_dataframe(
    session_lists: list[str],
    noise_col_id: str,
    noise_categories: list[int],
    message_output=print,
) -> tuple[pls.DataFrame, dict]:
    """
    Description
    Reads one or more session-list text files and computes per-session
    same-emitter inter-USV intervals in **both** definitions (``s2s`` and ``e2s``) for
    every session. Returns a single tidy Polars DataFrame with one row
    per inter-USV interval, tagged with an ``interval_type`` column so downstream code
    can filter / facet by mode without re-running the expensive USV CSV
    pass.

    Both modes are always computed because they are derived from the
    same per-session iteration over the noise-filtered USV table; there
    is no compute saving from omitting one, and downstream comparisons
    (e.g. how much overlap the ``e2s`` filter introduces) require both
    to be present.

    The DataFrame has columns ``session_id``, ``source_list``,
    ``interval_type`` (``'s2s'`` / ``'e2s'``), ``sex`` (``'male'`` /
    ``'female'``), ``interval_s``, ``log_interval``, ``male_id``,
    ``female_id``.

    Pure compute helper -- no disk side-effects. To persist results,
    pass the returned frame and any GMM / LRT outputs to
    :func:`save_notebook_archive_to_h5`.

    Parameters
    session_lists (list[str])
        List of text file paths, each containing session roots (one
        per line).
    noise_col_id (str)
        Name of the noise classification column in the USV summary
        CSV.
    noise_categories (list[int])
        Integer labels in ``noise_col_id`` that identify a row as
        noise to be excluded.
    message_output (callable)
        Logging callable; defaults to :func:`print`.

    Returns
    usv_interval_df (pls.DataFrame)
        Tidy DataFrame, one row per inter-USV interval per ``interval_type``.
    summary (dict)
        Keys: ``'n_sessions_loaded'`` (number of session roots that
        produced data in either mode), ``'n_dropped'`` (mapping
        ``interval_type -> {'male': int, 'female': int}`` of
        non-positive intervals dropped per mode).
    """

    interval_types = ("s2s", "e2s")

    sessions = _read_session_lists(session_lists, message_output)
    source_map = _session_source_map(session_lists)

    rows: list[dict] = []
    n_dropped = {it: {"male": 0, "female": 0} for it in interval_types}
    sessions_with_data: set[str] = set()

    for session_root in sessions:
        session_id = Path(session_root).name
        source_list = source_map.get(session_root, "")

        for interval_type in interval_types:
            usv_interval = compute_session_usv_intervals(
                session_root=session_root,
                interval_type=interval_type,
                noise_col_id=noise_col_id,
                noise_categories=noise_categories,
            )
            if not usv_interval:
                continue
            sessions_with_data.add(session_root)
            n_dropped[interval_type]["male"] += usv_interval["n_dropped_male"]
            n_dropped[interval_type]["female"] += usv_interval["n_dropped_female"]

            for v in usv_interval["male"]:
                rows.append({
                    "session_id": session_id,
                    "source_list": source_list,
                    "interval_type": interval_type,
                    "sex": "male",
                    "interval_s": float(v),
                    "log_interval": float(np.log(v)),
                    "male_id": usv_interval["male_id"],
                    "female_id": usv_interval["female_id"],
                })
            for v in usv_interval["female"]:
                rows.append({
                    "session_id": session_id,
                    "source_list": source_list,
                    "interval_type": interval_type,
                    "sex": "female",
                    "interval_s": float(v),
                    "log_interval": float(np.log(v)),
                    "male_id": usv_interval["male_id"],
                    "female_id": usv_interval["female_id"],
                })

    # Build the frame with an explicit schema so downstream
    # `filter(pls.col('interval_type') == ...)` works even when zero
    # rows were collected (e.g. session list resolved to no readable
    # sessions on this host).
    usv_interval_schema = {
        "session_id": pls.Utf8,
        "source_list": pls.Utf8,
        "interval_type": pls.Utf8,
        "sex": pls.Utf8,
        "interval_s": pls.Float64,
        "log_interval": pls.Float64,
        "male_id": pls.Utf8,
        "female_id": pls.Utf8,
    }
    if rows:
        usv_interval_df = pls.from_dicts(rows, schema=usv_interval_schema)
    else:
        usv_interval_df = pls.DataFrame(schema=usv_interval_schema)

    summary = {
        "n_sessions_loaded": len(sessions_with_data),
        "n_dropped": n_dropped,
    }

    return usv_interval_df, summary


def plot_log_usv_interval_histograms(
    usv_interval_df: pls.DataFrame,
    bins: int,
    male_color: str,
    female_color: str,
    figsize: tuple = (5, 5),
    xlims: tuple = (-5.0, 5.0),
    edge_color: str = "#000000",
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Description
    Histograms of ``log_interval`` for males and females overlaid on a
    single axis, normalised to integrate to 1 within each sex. Useful
    as a sanity check before any GMM fitting.

    The caller is responsible for pre-filtering ``usv_interval_df`` to a single
    ``interval_type`` (the master DataFrame produced by
    :func:`build_master_usv_interval_dataframe` contains both ``s2s`` and
    ``e2s`` rows; passing it unfiltered would conflate the two modes
    in the histograms).

    Parameters
    usv_interval_df (pls.DataFrame)
        Tidy DataFrame from :func:`build_master_usv_interval_dataframe`.
    bins (int)
        Number of histogram bins.
    male_color (str)
        Colour for the male histogram.
    female_color (str)
        Colour for the female histogram.
    figsize (tuple)
        Figure size; defaults to (5, 5) (square).
    xlims (tuple)
        (low, high) bounds on the x-axis (in log-seconds); defaults to
        (-5.0, 5.0).
    edge_color (str)
        Colour for the step-filled histogram edges; defaults to
        '#000000'.

    Returns
    f (plt.Figure)
        The created figure.
    ax (plt.Axes)
        The axes containing the histograms.
    stats (dict)
        Keys ``'n_M'``, ``'n_F'``, ``'median_M_sec'``, ``'median_F_sec'``.
    """

    male_log = usv_interval_df.filter(pls.col("sex") == "male")["log_interval"].to_numpy()
    female_log = usv_interval_df.filter(pls.col("sex") == "female")["log_interval"].to_numpy()

    f, ax = plt.subplots(figsize=figsize)
    if male_log.size:
        ax.hist(male_log, bins=bins, density=True, alpha=0.5,
                histtype="stepfilled", color=male_color, edgecolor=edge_color,
                label=f"male (n={male_log.size})")
    if female_log.size:
        ax.hist(female_log, bins=bins, density=True, alpha=0.5,
                histtype="stepfilled", color=female_color, edgecolor=edge_color,
                label=f"female (n={female_log.size})")
    ax.set_xlabel(r"$\mathrm{log}_{\mathrm{interval}}$ (s)")
    ax.set_ylabel("Density")
    ax.set_xlim(xlims)
    ax.legend()

    stats = {
        "n_M": int(male_log.size),
        "n_F": int(female_log.size),
        "median_M_sec": float(np.exp(np.median(male_log))) if male_log.size else float("nan"),
        "median_F_sec": float(np.exp(np.median(female_log))) if female_log.size else float("nan"),
    }
    return f, ax, stats


def run_bic_sweep(
    usv_interval_df: pls.DataFrame,
    n_components_min: int,
    n_components_max: int,
    n_repeats: int,
    max_modes_reported: int,
    random_seed_base: int,
    model_class: str = "gauss",
) -> pls.DataFrame:
    """
    Description
    Convenience wrapper around :func:`compute_inter_usv_interval_distributions.fit_gmm_sweep`
    that takes a tidy inter-USV interval DataFrame and returns the same tidy results
    table.

    Pure compute helper -- no disk side-effects. Persist via
    :func:`save_notebook_archive_to_h5` once both modes' sweeps are in
    hand.

    Parameters
    usv_interval_df (pls.DataFrame)
        Tidy inter-USV interval DataFrame from :func:`build_master_usv_interval_dataframe`.
    n_components_min (int)
        Minimum number of GMM components.
    n_components_max (int)
        Maximum number of GMM components.
    n_repeats (int)
        Number of EM-init repeats per fit.
    max_modes_reported (int)
        Up to this many mixture modes are recorded per fit.
    random_seed_base (int)
        Base seed; rep ``r`` uses ``random_seed_base + r``.
    model_class (str)
        ``'gauss'`` or ``'t'``; passed straight through to
        :func:`fit_gmm_sweep`.

    Returns
    df_results (pls.DataFrame)
        Tidy GMM / t-mixture sweep results.
    """

    male_arr = usv_interval_df.filter(pls.col("sex") == "male")["interval_s"].to_numpy()
    female_arr = usv_interval_df.filter(pls.col("sex") == "female")["interval_s"].to_numpy()

    return fit_gmm_sweep(
        intervals_by_key={"male": male_arr, "female": female_arr},
        n_components_min=n_components_min,
        n_components_max=n_components_max,
        n_repeats=n_repeats,
        max_modes_reported=max_modes_reported,
        random_seed_base=random_seed_base,
        model_class=model_class,
    )


def plot_ic_curves(
    df_results: pls.DataFrame,
    male_color: str,
    female_color: str,
    figsize: tuple = (6, 4),
    ic_col: str = "bic",
    selected_n_components: dict | None = None,
) -> tuple[plt.Figure, tuple, dict]:
    """
    Description
    Plots the *minimum* information criterion across repeats vs.
    ``n_components`` for each sex on **twin y-axes**: males on the
    left axis, females on the right. The two sexes typically have
    very different sample sizes and therefore very different IC
    magnitudes (BIC / ICL both scale with log-likelihood, which sums
    over N), making a shared axis illegible.

    The min reduction across reps is intentional: averaging blurs
    "this n_components is genuinely worse" with "EM got unlucky on
    these inits".

    The ``ic_col`` argument selects which criterion to plot
    (``'bic'``, ``'aic'``, ``'icl'``, or ``'cv_neg_loglik'``); the IC
    family is shown as a diagnostic only -- model selection is the
    bootstrap-LRT step-up rule. When ``selected_n_components`` is
    supplied, the K it nominates is highlighted as a larger filled
    circle outlined in black on top of the line plot. Each axis is
    colour-tinted to its sex so the reader cannot misread which curve
    goes with which scale.

    Parameters
    df_results (pls.DataFrame)
        Tidy results from :func:`run_bic_sweep`. Must contain a
        column named ``ic_col``.
    male_color (str)
        Colour for the male curve (left y-axis).
    female_color (str)
        Colour for the female curve (right y-axis).
    figsize (tuple)
        Figure size; defaults to (6, 4).
    ic_col (str)
        IC column to plot. Defaults to ``'bic'``.
    selected_n_components (dict | None)
        Mapping ``sex -> int K_selected`` from the bootstrap-LRT
        step-up rule. The matching K on each curve is highlighted
        with a larger filled circle. Defaults to ``None`` (no
        highlight).

    Returns
    f (plt.Figure)
        The created figure.
    axes (tuple)
        ``(ax_left, ax_right)`` — the male and female axes.
    stats (dict)
        Mapping ``key -> {'best_n_comp', 'best_ic', 'parsimonious_n_comp',
        'parsimonious_ic', 'delta_vs_best'}``.
    """

    f, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    axis_for = {"male": ax_left, "female": ax_right}
    color_for = {"male": male_color, "female": female_color}
    stats: dict = {}
    edge_color = "#000000"

    for sex in ("male", "female"):
        if sex not in df_results['sex'].unique().to_list():
            continue
        sub = df_results.filter(pls.col('sex') == sex)
        min_ic = (
            sub.group_by('n_comp')
            .agg(pls.col(ic_col).min().alias(ic_col))
            .sort('n_comp')
        )
        n_arr = min_ic['n_comp'].to_numpy()
        b_arr = min_ic[ic_col].to_numpy()

        ax = axis_for[sex]
        col = color_for[sex]
        # uniform-size data points; no argmin / parsimony markers
        ax.plot(n_arr, b_arr, "-o", color=col, markersize=8)

        # outlined (not filled) black square at the K selected by the
        # bootstrap-LRT step-up procedure (passed in by the caller)
        if selected_n_components is not None and sex in selected_n_components:
            sel_n = int(selected_n_components[sex])
            n_list = list(int(x) for x in n_arr)
            if sel_n in n_list:
                sel_idx = n_list.index(sel_n)
                sel_val = float(b_arr[sel_idx])
                # Selected K marker: a larger filled circle (in the sex
                # colour) outlined in black, drawn on top of the line
                # plot's small circle. The black edge alone identifies
                # the selected K -- no nested square.
                ax.scatter(
                    [sel_n], [sel_val],
                    s=300, marker="o",
                    color=col, edgecolors=edge_color,
                    linewidths=1.5, zorder=6,
                )

        ic_display = {
            "bic": "BIC",
            "aic": "AIC",
            "icl": "ICL",
            "cv_neg_loglik": "CV-LL (deviance)",
        }.get(ic_col, ic_col.upper())
        ax.set_ylabel(f"{sex}: min {ic_display} across reps", color=col)
        ax.tick_params(axis="y", labelcolor=col)
        for spine_pos in ("left",) if sex == "male" else ("right",):
            ax.spines[spine_pos].set_color(col)

        stats[sex] = {
            "min_ic_per_K": dict(zip([int(n) for n in n_arr], [float(b) for b in b_arr], strict=False)),
            "selected_n_components": (
                int(selected_n_components[sex])
                if selected_n_components is not None and sex in selected_n_components
                else None
            ),
        }

    ax_left.set_xlabel("n_components")
    return f, (ax_left, ax_right), stats


def plot_best_fit_with_annotations(
    intervals_sec: np.ndarray,
    gmm,
    gmm_order: np.ndarray,
    color: str,
    figsize: tuple = (5, 5),
    bins: int = 80,
    xlims: tuple = (-5.0, 5.0),
    tau: float = 0.5,
    edge_color: str = "#000000",
    qq_inset_bbox: tuple | None = (0.62, 0.40, 0.34, 0.34),
    qq_n_q: int = 200,
    legend_corner: str = "upper right",
    auto_inset_below_legend: bool = False,
    auto_inset_size: tuple = (0.34, 0.30),
    show_components: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Description
    Wraps :func:`gmm_utils.plot_gmm_fit` and overlays a downward
    triangle on the mixture density curve at the location of each
    fitted component mean ``mu_k`` (each component's own peak in
    log-space, not the mixture-level local maxima). The bottom apex
    of every triangle sits exactly on the mixture curve; each
    triangle is labelled with a bold ``(letter)`` (``(a)``, ``(b)``,
    ...) above it. A text-only legend block (no marker glyphs)
    lists the same letters mapped to component medians in seconds
    (``exp(mu_k)``); the legend can be placed in any corner via
    ``legend_corner``.

    When ``qq_inset_bbox`` is supplied a Q-Q inset is drawn at that
    axes-fraction box, with axis labels and log-log scaling. The
    caller is responsible for picking a non-overlapping bbox for the
    chosen ``legend_corner``.

    Parameters
    intervals_sec (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval
        values (in seconds).
    gmm (GaussianMixture | TMixture)
        The fitted mixture in log-space.
    gmm_order (np.ndarray)
        Sort indices for the components (ascending by log-mean).
    color (str)
        Histogram fill colour and Q-Q-inset dot colour.
    figsize (tuple)
        Figure size; defaults to (5, 5) (square).
    bins (int)
        Number of histogram bins; defaults to 80.
    xlims (tuple)
        (low, high) bounds in log-space; defaults to (-5.0, 5.0).
    tau (float)
        Posterior threshold passed to
        :func:`gmm_utils.summarize_best_gmm` (used by the returned
        summary dict; no longer drawn on the plot). Defaults to 0.5.
    edge_color (str)
        Histogram edge colour and triangle outline; defaults to
        '#000000'.
    qq_inset_bbox (tuple | None)
        Four-tuple ``(x, y, w, h)`` in axes fractions of the host
        axes specifying the Q-Q inset position. ``None`` disables
        the inset entirely. Defaults to ``(0.62, 0.40, 0.34, 0.34)``
        -- the upper-right area below a top-right text legend.
    qq_n_q (int)
        Number of quantile probabilities used by the Q-Q inset;
        defaults to 200.
    legend_corner (str)
        Where to place the lettered text legend block. One of
        ``'upper right'``, ``'upper left'``, ``'lower right'``,
        ``'lower left'``. Defaults to ``'upper right'``.
    auto_inset_below_legend (bool)
        Auto-place the Q-Q inset directly under the rendered legend
        block; ignores ``qq_inset_bbox`` when True. Useful for
        variable-K runs where the legend height changes. Defaults
        to False.
    auto_inset_size (tuple)
        ``(width, height)`` in axes fractions used when
        ``auto_inset_below_legend`` is True. Defaults to
        ``(0.34, 0.30)``.
    show_components (bool)
        If True, overlay each individual mixture component's pdf
        (weighted by its mixing weight) on the histogram. Each
        component is drawn with a distinct (colour, linestyle) pair
        from a fixed palette so overlapping components remain
        distinguishable; the original sex-coloured mixture-sum curve
        is preserved underneath. Defaults to False.

    Returns
    f (plt.Figure)
        The created figure.
    ax (plt.Axes)
        The main (distribution) axes.
    summary (dict)
        Output of :func:`gmm_utils.summarize_best_gmm` (or
        :func:`gmm_utils.summarize_best_t_mixture` for t-mixtures),
        plus a ``'qq_pearson_r'`` key (NaN when ``qq_inset_bbox`` is
        None) recording the log-log Pearson correlation between
        empirical and model quantiles.
    """

    log_x = np.log(intervals_sec)
    f, ax = plot_gmm_fit(
        model=gmm,
        x=log_x,
        figsize=figsize,
        bins=bins,
        xlims=xlims,
        color=color,
        edge_color=edge_color,
        show_components=False,
        legend=False,
    )

    # Reserve ~18% headroom above the densest histogram bar / curve
    # peak so the bold component labels and (when placed up top) the
    # text legend / Q-Q inset never collide with the figure title.
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi * 1.18)

    # dispatch on model class -- Gaussian path uses analytic boundaries
    # via summarize_best_gmm; Student-t path uses summarize_best_t_mixture
    # which leaves boundaries empty (decision boundaries between t-components
    # have no closed form and are not currently rendered).
    if isinstance(gmm, TMixture):
        summary = summarize_best_t_mixture(gmm, gmm_order)
    else:
        summary = summarize_best_gmm(gmm, gmm_order, tau=tau)

    if show_components:
        _draw_mixture_components(
            ax, gmm, gmm_order, xlims=xlims,
        )

    logmeans = summary["logmeans"]
    qq_pearson_r = float("nan")

    if logmeans.size:
        peaks_log = logmeans.reshape(-1, 1)
        mixture_pdf_at_peaks = np.exp(gmm.score_samples(peaks_log))

        # Triangles: marker='v' is centred on the data point so the
        # bottom apex sits below the centre. Offsetting the marker
        # upward by half its height (in display points, via
        # offset_copy) lands the apex exactly on the curve.
        marker_size_pts2 = 80.0
        marker_height_pts = float(np.sqrt(marker_size_pts2)) * np.sqrt(3) / 2.0
        triangle_trans = offset_copy(
            ax.transData, fig=f, y=marker_height_pts / 2.0, units="points",
        )
        ax.scatter(
            logmeans, mixture_pdf_at_peaks,
            marker="v", s=marker_size_pts2, color=color, edgecolors=edge_color,
            linewidths=1.0, zorder=6, clip_on=False,
            transform=triangle_trans,
        )

        letters = [f"({chr(ord('a') + k)})" for k in range(logmeans.size)]
        # Bold letter sits just above the triangle's top edge; offset
        # accounts for the marker height we just applied so the label
        # clears the triangle.
        label_offset_pts = marker_height_pts + 4.0
        for letter, mu_k, y_k in zip(letters, logmeans, mixture_pdf_at_peaks, strict=False):
            ax.annotate(
                letter,
                xy=(mu_k, y_k),
                xytext=(0, label_offset_pts),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=edge_color,
                clip_on=False,
            )

        # Text-only legend block: one bold line per (letter, median),
        # rendered as a single multi-line Text so the lines are
        # left-aligned within the block (each '(a) =', '(b) =' starts
        # at the same x). The block is anchored to the requested corner
        # via ha/va; matplotlib's `multialignment='left'` does the
        # internal alignment.
        legend_str = "\n".join(
            f"{letter} = {np.exp(mu_k):.3g} s"
            for letter, mu_k in zip(letters, logmeans, strict=False)
        )
        legend_text = _draw_text_legend(
            ax, legend_str,
            corner=legend_corner,
            color=edge_color,
            fontsize=12,
        )
    else:
        legend_text = None

    if auto_inset_below_legend and legend_text is not None:
        # Compute the legend's actual rendered bottom edge in axes-frac
        # so the inset can sit directly below it regardless of how many
        # components the legend lists. Requires drawing the canvas
        # first so the renderer can measure the text bbox.
        f.canvas.draw()
        renderer = f.canvas.get_renderer()
        legend_bbox_disp = legend_text.get_window_extent(renderer=renderer)
        legend_bbox_axes = legend_bbox_disp.transformed(ax.transAxes.inverted())
        legend_bottom = float(legend_bbox_axes.y0)
        legend_left = float(legend_bbox_axes.x0)

        w, h = float(auto_inset_size[0]), float(auto_inset_size[1])
        # The inset's "Model (s)" y-label and tick labels extend to the
        # *left* of the data box, so we need ~0.08 axes-frac of padding
        # between the legend's left edge and the inset's data box left
        # edge. Without this, the y-label crowds the (c)/(d) triangle
        # markers that sit just to the left.
        y_label_clearance = 0.08
        x = max(0.02, min(1.0 - w - 0.02, legend_left + y_label_clearance))
        # Sit the inset *just* below the legend; the user-provided
        # height is taken at face value, so if peaks crowd the inset
        # bottom the caller should shrink ``auto_inset_size``.
        y = max(0.05, legend_bottom - h - 0.02)
        ax_qq = ax.inset_axes([x, y, w, h])
        qq_pearson_r = _draw_qq_into_axes(
            ax_qq,
            intervals_sec=intervals_sec,
            gmm=gmm,
            dot_color=color,
            line_color=edge_color,
            n_q=qq_n_q,
            inset=True,
        )
    elif qq_inset_bbox is not None:
        ax_qq = ax.inset_axes(list(qq_inset_bbox))
        qq_pearson_r = _draw_qq_into_axes(
            ax_qq,
            intervals_sec=intervals_sec,
            gmm=gmm,
            dot_color=color,
            line_color=edge_color,
            n_q=qq_n_q,
            inset=True,
        )

    summary = dict(summary)
    summary["qq_pearson_r"] = qq_pearson_r
    return f, ax, summary


def _draw_text_legend(
    ax: plt.Axes,
    legend_str: str,
    *,
    corner: str = "upper right",
    color: str = "#000000",
    fontsize: int = 12,
    pad_axes_frac: float = 0.02,
):
    """
    Description
    Anchors a single multi-line ``Text`` object to one of the four
    corners of ``ax`` with internal left-alignment, so each line's
    leading character (``(a) =``, ``(b) =``, ...) sits at the same
    x position regardless of how long the trailing seconds-value
    string ends up. A subtle white background panel separates the
    legend from histogram bars behind it. Returns the Text handle so
    the caller can measure its rendered bbox (used by the
    auto-inset-below-legend layout path).

    Parameters
    ax (plt.Axes)
        Destination axes.
    legend_str (str)
        Multi-line string; one line per legend entry.
    corner (str)
        ``'upper right'`` / ``'upper left'`` / ``'lower right'`` /
        ``'lower left'``. Defaults to ``'upper right'``.
    color (str)
        Text colour.
    fontsize (int)
        Font size in points.
    pad_axes_frac (float)
        Padding from the axes edge in axes fractions.

    Returns
    text (matplotlib.text.Text)
        The rendered text handle.
    """

    if corner == "upper right":
        x_frac, y_frac, ha, va = 1.0 - pad_axes_frac, 1.0 - pad_axes_frac, "right", "top"
    elif corner == "upper left":
        x_frac, y_frac, ha, va = pad_axes_frac, 1.0 - pad_axes_frac, "left", "top"
    elif corner == "lower right":
        x_frac, y_frac, ha, va = 1.0 - pad_axes_frac, pad_axes_frac, "right", "bottom"
    elif corner == "lower left":
        x_frac, y_frac, ha, va = pad_axes_frac, pad_axes_frac, "left", "bottom"
    else:
        msg = f"_draw_text_legend: unknown corner={corner!r}."
        raise ValueError(msg)

    return ax.text(
        x_frac, y_frac, legend_str,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize, fontweight="bold", color=color,
        multialignment="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
        clip_on=False,
        zorder=8,
    )


_COMPONENT_PALETTE: tuple[tuple[str, object], ...] = (
    ("#1f77b4", "--"),
    ("#ff7f0e", "-."),
    ("#2ca02c", ":"),
    ("#d62728", (0, (3, 1, 1, 1))),
    ("#9467bd", (0, (5, 1))),
    ("#8c564b", (0, (1, 1))),
    ("#e377c2", (0, (3, 1, 1, 1, 1, 1))),
    ("#7f7f7f", (0, (5, 2, 1, 2))),
)


def _draw_mixture_components(
    ax: plt.Axes,
    gmm,
    gmm_order: np.ndarray,
    *,
    xlims: tuple,
    n_grid: int = 500,
    lw: float = 1.5,
) -> None:
    """
    Description
    Overlays the **shape** of each fitted mixture component on the
    host axes, scaled so its peak lies exactly on the black
    mixture-sum curve drawn by :func:`gmm_utils.plot_gmm_fit`. Each
    component is drawn with a distinct (colour, linestyle) pair from
    :data:`_COMPONENT_PALETTE`, cycling if the mixture has more
    components than palette entries.

    The standard mixture decomposition would plot
    ``posterior(k|x) * mixture(x)`` for each component, which sums
    exactly to the mixture but leaves a small visible gap between
    each component's peak and the black mixture curve (because at
    component k's mode, the mixture also receives contributions from
    every other component). For a "shape" view that sits flush
    against the mixture curve, each component is rescaled by
    ``1 / posterior(k | mu_k)`` so its value at ``x = mu_k`` equals
    ``mixture(mu_k)``. The component shape (ratio of values across
    x) is preserved; the components no longer sum to the mixture.

    Components are rendered in the order given by ``gmm_order``
    (ascending log-mean), so component ``(a)`` is the leftmost peak.

    Parameters
    ax (plt.Axes)
        Destination axes (the main best-fit panel).
    gmm (GaussianMixture | TMixture)
        Fitted mixture in log-space.
    gmm_order (np.ndarray)
        Sort indices that map ascending-log-mean position -> original
        component index. ``gmm_order[0]`` is the index of component
        ``(a)`` in the model's internal arrays.
    xlims (tuple)
        ``(low, high)`` bounds in log-seconds within which to
        evaluate the per-component pdfs.
    n_grid (int)
        Number of evaluation points along the x-axis. Defaults to
        500, matching :func:`gmm_utils.plot_gmm_fit`.
    lw (float)
        Line width for the component curves. Defaults to 1.5.

    Returns
    """

    xx = np.linspace(xlims[0], xlims[1], n_grid).reshape(-1, 1)
    mixture_pdf = np.exp(gmm.score_samples(xx))           # shape (N,)
    posteriors = gmm.predict_proba(xx)                    # shape (N, K_orig)
    K = int(np.asarray(gmm_order).size)

    # Posterior of component k at its own mean (Gaussian / Student-t
    # modes both coincide with mu_k). 1 / this value is the scale that
    # lifts the component's peak onto the mixture curve.
    means_orig = np.asarray(gmm.means_).reshape(-1, 1)    # (K_orig, 1)
    posteriors_at_mu = gmm.predict_proba(means_orig)      # (K_orig, K_orig)

    for k_sorted in range(K):
        k_orig = int(gmm_order[k_sorted])
        post_k_at_mu_k = float(posteriors_at_mu[k_orig, k_orig])
        # Guard against pathological cases (collapsed component or
        # numerical underflow) where the posterior at the mean rounds
        # to zero; fall back to the un-scaled weighted-component view.
        scale = 1.0 / post_k_at_mu_k if post_k_at_mu_k > 0.0 else 1.0
        comp = posteriors[:, k_orig] * mixture_pdf * scale
        col, ls = _COMPONENT_PALETTE[k_sorted % len(_COMPONENT_PALETTE)]
        ax.plot(
            xx.ravel(), comp,
            color=col, linestyle=ls, lw=lw, zorder=4,
        )


def _draw_qq_into_axes(
    ax: plt.Axes,
    *,
    intervals_sec: np.ndarray,
    gmm,
    dot_color: str,
    line_color: str = "#000000",
    n_q: int = 200,
    inset: bool = False,
) -> float:
    """
    Description
    Renders a log-log Q-Q plot of empirical quantiles (in seconds)
    against model quantiles into the supplied axes; returns the
    log-space Pearson correlation as a goodness-of-fit summary.
    Internal helper shared by :func:`plot_qq` and the Q-Q inset of
    :func:`plot_best_fit_with_annotations`.

    Parameters
    ax (plt.Axes)
        Destination axes (a fresh figure axes for the standalone
        plot, or the inset axes for the embedded variant).
    intervals_sec (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval
        values.
    gmm (GaussianMixture | TMixture)
        The fitted mixture in log-space.
    dot_color (str)
        Colour for the empirical-vs-model quantile dots.
    line_color (str)
        Colour for the reference y=x diagonal.
    n_q (int)
        Number of quantile probabilities; defaults to 200.
    inset (bool)
        If True, render in compact mode (smaller dots, smaller tick
        font, no axis labels). Defaults to False.

    Returns
    pearson_r (float)
        Log-space Pearson correlation between empirical and model
        quantiles.
    """

    qs = np.linspace(0.01, 0.99, n_q)
    obs_q = np.quantile(intervals_sec, qs)
    if isinstance(gmm, TMixture):
        model_q = np.exp(t_mixture_quantile_logspace(qs, gmm))
    else:
        model_q = np.exp(gmm_quantile_logspace(qs, gmm))

    # Larger, more opaque dots so the cloud reads at small inset sizes;
    # the standalone variant uses an even larger marker.
    dot_size = 14 if inset else 36
    dot_alpha = 0.9 if inset else 0.7
    ax.scatter(obs_q, model_q, s=dot_size, alpha=dot_alpha, color=dot_color, edgecolors="none")
    lo = float(min(obs_q.min(), model_q.min()))
    hi = float(max(obs_q.max(), model_q.max()))
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=line_color)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if inset:
        ax.tick_params(axis="both", which="both", labelsize=7, length=2)
        # Inset retains compact axis labels so the diagnostic stays
        # self-describing without dominating the panel.
        ax.set_xlabel("Observed (s)", fontsize=8, labelpad=2)
        ax.set_ylabel("Model (s)", fontsize=8, labelpad=2)
        # Opaque white panel so the inset reads as a distinct object
        # against the histogram bars behind it.
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color(line_color)
            spine.set_linewidth(0.8)
    else:
        ax.set_xlabel("Observed quantiles (s)")
        ax.set_ylabel("Model quantiles (s)")
        ax.set_title("Q-Q (mixture model vs. data)")

    log_obs = np.log(obs_q)
    log_mod = np.log(model_q)
    return float(np.corrcoef(log_obs, log_mod)[0, 1])


def plot_qq(
    intervals_sec: np.ndarray,
    gmm,
    dot_color: str,
    figsize: tuple = (5, 5),
    n_q: int = 200,
    line_color: str = "#000000",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Description
    Q-Q plot of empirical quantiles (in seconds) against model
    quantiles obtained by inverting the analytic GMM CDF (no
    Monte-Carlo noise). The reference y=x line is drawn in
    ``line_color`` (black by default); the dots are drawn in
    ``dot_color`` (typically the male / female palette colour
    depending on the panel).

    When ``ax`` is supplied the plot is drawn into that axes (useful
    for embedding as an inset); otherwise a fresh figure of size
    ``figsize`` is created.

    Parameters
    intervals_sec (np.ndarray)
        A (n_samples,) shape ndarray of strictly positive interval
        values.
    gmm (GaussianMixture | TMixture)
        The fitted mixture in log-space.
    dot_color (str)
        Colour for the empirical-vs-model quantile dots.
    figsize (tuple)
        Figure size when ``ax`` is None; defaults to (5, 5).
    n_q (int)
        Number of quantile probabilities; defaults to 200.
    line_color (str)
        Colour for the reference y=x diagonal; defaults to '#000000'.
    ax (plt.Axes | None)
        Pre-existing axes to draw into; defaults to None (new
        figure).

    Returns
    f (plt.Figure)
        The figure containing the Q-Q axes.
    ax (plt.Axes)
        The axes.
    stats (dict)
        Keys ``'pearson_r'`` (in log-space) for goodness-of-fit at a
        glance.
    """

    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.figure

    pearson_r = _draw_qq_into_axes(
        ax,
        intervals_sec=intervals_sec,
        gmm=gmm,
        dot_color=dot_color,
        line_color=line_color,
        n_q=n_q,
        inset=False,
    )
    return f, ax, {"pearson_r": pearson_r}


def run_bootstrap_lrt_sweep(
    intervals_by_key: dict[str, np.ndarray],
    n_components_min: int,
    n_components_max: int,
    B: int = 50,
    n_subsample: int = 15000,
    model_class: str = "t",
    n_init_obs: int = 10,
    n_init_boot: int = 3,
    reg_covar: float = 1e-4,
    seed: int = 0,
    message_output=print,
) -> dict:
    """
    Description
    Runs the parametric bootstrap likelihood-ratio test for every
    consecutive pair ``(K, K+1)`` in the range
    ``[n_components_min, n_components_max]``, separately for each
    key in ``intervals_by_key``.

    Pure compute helper -- no disk side-effects, no step-up selection.
    Apply the step-up rule via :func:`select_n_components_from_lrt_sweep`
    on the returned sweep dict; persist via
    :func:`save_notebook_archive_to_h5`.

    Parameters
    intervals_by_key (dict)
        Mapping ``sex -> np.ndarray`` of strictly positive intervals
        (``sex`` typically ``'male'`` or ``'female'``).
    n_components_min (int)
        Smallest K in the sweep.
    n_components_max (int)
        Largest K in the sweep. Pairs tested are
        ``(K_min, K_min+1), ..., (K_max-1, K_max)``.
    B (int)
        Bootstrap replicates per pair.
    n_subsample (int)
        Subsample size for both observed and bootstrap fits.
    model_class (str)
        ``'gauss'`` or ``'t'``.
    n_init_obs (int)
        EM restarts for the observed fits.
    n_init_boot (int)
        EM restarts for each bootstrap fit.
    reg_covar (float)
        Component variance floor.
    seed (int)
        RNG seed.
    message_output (callable)
        Logging callable.

    Returns
    sweep (dict)
        Mapping ``key -> {(K_null, K_alt) -> result_dict}``, where
        each ``result_dict`` is the return value of
        :func:`gmm_utils.bootstrap_lrt`.
    """

    pairs = [(k, k + 1) for k in range(n_components_min, n_components_max)]
    sweep: dict = {}

    for sex, intervals_sec in intervals_by_key.items():
        if intervals_sec.size < 2:
            continue
        sweep[sex] = {}
        for (K_n, K_a) in pairs:
            message_output(f"  [{sex}] bootstrap LRT K={K_n} vs K={K_a}...")
            res = bootstrap_lrt(
                intervals_sec=intervals_sec,
                K_null=K_n,
                K_alt=K_a,
                B=B,
                n_subsample=n_subsample,
                model_class=model_class,
                n_init_obs=n_init_obs,
                n_init_boot=n_init_boot,
                reg_covar=reg_covar,
                seed=seed,
            )
            sweep[sex][(K_n, K_a)] = res
            message_output(
                f"    LR_obs={res['lr_obs']:.2f}, null_mean={res['null_mean']:.2f}, "
                f"null_95%={res['null_p95']:.2f}, p={res['p_value']:.3f}"
            )

    return sweep


def select_n_components_from_lrt_sweep(
    sweep: dict,
    alpha: float = 0.05,
    bonferroni: bool = False,
) -> dict:
    """
    Description
    Applies the step-up rule per key to a bootstrap-LRT sweep
    produced by :func:`run_bootstrap_lrt_sweep`. Optionally applies
    a Bonferroni correction across the number of tests per key.

    Parameters
    sweep (dict)
        Output of :func:`run_bootstrap_lrt_sweep`.
    alpha (float)
        Significance threshold; defaults to 0.05.
    bonferroni (bool)
        If True, divide alpha by the number of pairs per key.

    Returns
    selected (dict)
        Mapping ``key -> int K_selected``.
    """

    selected: dict = {}
    for key, pair_results in sweep.items():
        n_tests = len(pair_results)
        alpha_eff = alpha / n_tests if (bonferroni and n_tests > 0) else alpha
        selected[key] = select_n_components_step_up_lrt(pair_results, alpha=alpha_eff)
    return selected


def plot_bootstrap_lrt_panel(
    sweep: dict,
    figsize_per_panel: tuple = (4, 3),
    break_gap_factor: float = 1.5,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Description
    Renders one row per sex (male / female), one column per consecutive
    K-pair, showing the bootstrap null distribution of the LR
    statistic with the observed LR marked in red and the 99th-
    percentile of the null marked with a dotted line. The p-value is
    in each subplot's title.

    When the observed LR falls far above the bulk of the null
    distribution -- specifically when the gap between the rightmost
    null bin edge and ``LR_obs`` exceeds
    ``break_gap_factor * null_max``, where ``null_max`` is the largest
    null draw -- the cell is rendered as a horizontally split
    broken-axis pair: the left axes shows the null histogram on its
    own scale, the right axes shows ``LR_obs`` (and the 99% reference
    line if it falls on that side), and matching ``//`` slashes mark
    the discontinuity. This collapses the empty space that would
    otherwise dominate the cell when ``LR_obs`` is far from the null
    bulk.

    Parameters
    sweep (dict)
        Output of :func:`run_bootstrap_lrt_sweep`.
    figsize_per_panel (tuple)
        Width and height of each individual subplot in inches;
        defaults to (4, 3).
    break_gap_factor (float)
        Threshold for triggering the broken-axis split. The break is
        applied when ``LR_obs - null_max > break_gap_factor *
        null_max``. Defaults to 1.5 (i.e. the gap must be at least
        150% of the null's full width).

    Returns
    f (plt.Figure)
        The created figure.
    axes (np.ndarray)
        Array of axes, shape ``(n_keys, n_pairs)``. Cells that were
        broken into two axes hold a 2-tuple ``(ax_left, ax_right)``
        instead of a single Axes; cells without a break hold a single
        Axes as before.
    """

    keys = list(sweep.keys())
    if not keys:
        f, ax = plt.subplots(figsize=figsize_per_panel)
        return f, np.array([[ax]], dtype=object)

    pairs_per_key = {k: sorted(sweep[k].keys()) for k in keys}
    n_pairs = max(len(p) for p in pairs_per_key.values())

    figsize = (figsize_per_panel[0] * n_pairs, figsize_per_panel[1] * len(keys))
    f = plt.figure(figsize=figsize)
    outer = f.add_gridspec(len(keys), n_pairs, hspace=0.45, wspace=0.35)

    axes = np.empty((len(keys), n_pairs), dtype=object)

    for row, key in enumerate(keys):
        for col, (K_n, K_a) in enumerate(pairs_per_key[key]):
            res = sweep[key][(K_n, K_a)]
            lr_null = np.asarray(res["lr_null"])
            null_p99 = float(np.quantile(lr_null, 0.99)) if lr_null.size else float("nan")
            null_max = float(res.get("null_max", np.nan)) if res.get("null_max") is not None else (
                float(lr_null.max()) if lr_null.size else float("nan")
            )
            lr_obs = float(res["lr_obs"])
            gap = lr_obs - null_max
            need_break = (
                np.isfinite(null_max)
                and null_max > 0
                and gap > break_gap_factor * null_max
            )

            if need_break:
                # Two side-by-side axes inside this cell, sharing the
                # y-axis. Left axes covers the null bulk; right axes
                # covers LR_obs (plus any 95%/99% lines that landed
                # past the break).
                inner = outer[row, col].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
                ax_l = f.add_subplot(inner[0, 0])
                ax_r = f.add_subplot(inner[0, 1], sharey=ax_l)

                _draw_lrt_cell_pair(
                    ax_l, ax_r, lr_null, lr_obs,
                    null_p99=null_p99,
                )
                # Centre the title above the *whole* cell (both
                # sub-axes), not just over ``ax_l``. We derive the
                # cell's horizontal centre from the parent
                # SubplotSpec's figure-coord bbox so the title stays
                # centred regardless of the 3:1 width ratio between
                # ``ax_l`` and ``ax_r``.
                cell_bbox = outer[row, col].get_position(f)
                f.text(
                    (cell_bbox.x0 + cell_bbox.x1) / 2.0,
                    cell_bbox.y1 + 0.005,
                    f"{key}: K={K_n} vs K={K_a}\np = {res['p_value']:.3f}",
                    ha="center", va="bottom",
                    fontsize=plt.rcParams["axes.titlesize"],
                )
                ax_l.set_xlabel("LR statistic")
                ax_l.set_ylabel("count")
                # Combine handles from both axes so LR_obs (drawn on the
                # right axes) appears in the same legend as the null
                # reference lines (drawn on the left axes).
                h_l, lbl_l = ax_l.get_legend_handles_labels()
                h_r, lbl_r = ax_r.get_legend_handles_labels()
                ax_l.legend(h_l + h_r, lbl_l + lbl_r, fontsize=8, loc="upper right")
                axes[row, col] = (ax_l, ax_r)
            else:
                ax = f.add_subplot(outer[row, col])
                ax.hist(
                    lr_null,
                    bins=50,
                    color="#a0a0a0", edgecolor="#000000",
                    histtype="stepfilled",
                )
                ax.axvline(lr_obs, color="#cc0000", lw=2,
                           label=f"LR_obs = {lr_obs:.1f}")
                ax.axvline(null_p99, color="#000000", linestyle=":", lw=1,
                           label=f"null 99% = {null_p99:.1f}")
                ax.set_title(f"{key}: K={K_n} vs K={K_a}\np = {res['p_value']:.3f}")
                ax.set_xlabel("LR statistic")
                ax.set_ylabel("count")
                ax.legend(fontsize=8)
                axes[row, col] = ax

        for col in range(len(pairs_per_key[key]), n_pairs):
            ax = f.add_subplot(outer[row, col])
            ax.axis("off")
            axes[row, col] = ax

    return f, axes


def _draw_lrt_cell_pair(
    ax_l: plt.Axes,
    ax_r: plt.Axes,
    lr_null: np.ndarray,
    lr_obs: float,
    *,
    null_p99: float,
) -> None:
    """
    Description
    Renders a single broken-axis LRT cell across the supplied left /
    right axes. The left axes shows the null histogram and the 99%
    reference line if it falls within the null's range; the right
    axes shows the observed LR (and the 99% line if it falls past
    the break). Spines are hidden on the inside edges and matching
    ``//`` slashes are drawn at the break.

    Internal helper for :func:`plot_bootstrap_lrt_panel`.

    Parameters
    ax_l (plt.Axes)
        Left axes (covers null bulk).
    ax_r (plt.Axes)
        Right axes (covers LR_obs).
    lr_null (np.ndarray)
        Bootstrap null draws.
    lr_obs (float)
        Observed LR statistic.
    null_p99 (float)
        99th percentile of the null distribution.

    Returns
    """

    null_min = float(lr_null.min()) if lr_null.size else 0.0
    null_max = float(lr_null.max()) if lr_null.size else 0.0
    pad_left = 0.08 * max(null_max - null_min, 1.0)
    left_lo = null_min - pad_left
    left_hi = null_max + pad_left

    # right pane focuses on lr_obs with a tight, symmetric pad
    right_pad = 0.05 * max(lr_obs - null_max, 1.0)
    right_lo = lr_obs - right_pad
    right_hi = lr_obs + right_pad

    # null histogram on the left axes
    ax_l.hist(
        lr_null,
        bins=50,
        color="#a0a0a0", edgecolor="#000000",
        histtype="stepfilled",
    )
    # 99% reference line: drawn on whichever side it falls on
    if null_p99 <= left_hi:
        ax_l.axvline(null_p99, color="#000000", linestyle=":", lw=1,
                     label=f"null 99% = {null_p99:.1f}")
    elif right_lo <= null_p99 <= right_hi:
        ax_r.axvline(null_p99, color="#000000", linestyle=":", lw=1,
                     label=f"null 99% = {null_p99:.1f}")
    # observed LR on the right axes
    ax_r.axvline(lr_obs, color="#cc0000", lw=2,
                 label=f"LR_obs = {lr_obs:.1f}")

    ax_l.set_xlim(left_lo, left_hi)
    ax_r.set_xlim(right_lo, right_hi)

    # Hide the inner spines plus only the *right* axes' tick marks
    # and labels. We deliberately do NOT call ``ax_r.set_yticks([])``
    # because ``ax_r`` shares its y-axis with ``ax_l`` (via
    # ``sharey=ax_l``); clearing the tick list on either side wipes
    # the shared y-axis on both, which used to leave ``ax_l`` with no
    # tick marks or labels at all. ``tick_params`` only toggles
    # *display*, not the underlying tick locator, so it is safe under
    # sharey.
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Draw matching diagonal break markers ("//") via a marker shape rather
    # than line segments. Line segments drawn in axes coordinates render with
    # a slope of h/w in screen pixels, so the left and right halves of the
    # broken axis (which have different widths under the GridSpec width_ratios)
    # would otherwise show slashes at different visual angles. Markers are
    # sized in screen pixels, so the visual angle is identical on both sides.
    d = 0.5  # half-extent of the slash shape in marker units
    slash_marker = [(-1, -d), (1, d)]
    marker_kwargs = {
        "marker": slash_marker,
        "markersize": 10,
        "linestyle": "none",
        "color": "#000000",
        "mec": "#000000",
        "mew": 1,
        "clip_on": False,
    }
    ax_l.plot([1], [0], transform=ax_l.transAxes, **marker_kwargs)
    ax_r.plot([0], [0], transform=ax_r.transAxes, **marker_kwargs)


# HDF5 archive helpers for the notebook flow: persist the in-memory
# results of build_master_usv_interval_dataframe / run_bic_sweep /
# run_bootstrap_lrt_sweep into the same archive layout the CLI writes,
# and locate the most-recent archive in an output directory so plot
# cells can survive a kernel restart.


def save_notebook_archive_to_h5(
    output_directory: str,
    usv_interval_df: pls.DataFrame,
    usv_interval_summary: dict,
    usv_interval_cfg: dict,
    gmm_fits_by_mode: dict[str, pls.DataFrame] | None = None,
    lrt_sweep_by_mode: dict[str, dict] | None = None,
    message_output=print,
) -> Path:
    """
    Description
    Consolidates the notebook's in-memory inter-USV interval compute outputs into a
    single ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5`` archive in the same
    layout the CLI (:class:`InterUSVIntervalCalculator`) writes. Lets the notebook
    iterate on plot cells across kernel restarts via the HDF5 loader
    family (:func:`load_intervals_from_h5` etc.).

    Empty / missing modes are handled gracefully: a mode with no
    intervals is not written; a mode with intervals but no GMM /
    bootstrap-LRT outputs is written with the corresponding tables
    omitted.

    Parameters
    output_directory (str)
        Directory in which to write the archive. Run through
        :func:`configure_path`; created if missing.
    usv_interval_df (pls.DataFrame)
        Tidy inter-USV interval DataFrame returned by
        :func:`build_master_usv_interval_dataframe`. Must contain an
        ``interval_type`` column.
    usv_interval_summary (dict)
        ``summary`` dict returned alongside ``usv_interval_df`` by
        :func:`build_master_usv_interval_dataframe`. Used to populate the
        archive's ``n_sessions_loaded`` attribute and the per-mode
        ``drop_counts`` table.
    usv_interval_cfg (dict)
        The ``compute_inter_usv_interval_distributions`` block from
        ``analyses_settings.json``. Every parameter that drove the
        run is stored as a root-level attribute for provenance.
    gmm_fits_by_mode (dict | None)
        Optional ``{mode: pls.DataFrame}`` mapping of GMM-sweep
        results (one frame per mode) returned by
        :func:`run_bic_sweep`. ``None`` is equivalent to "no sweeps
        ran" -- the ``gmm_fits`` dataset is omitted from each mode.
    lrt_sweep_by_mode (dict | None)
        Optional ``{mode: sweep_dict}`` mapping returned by
        :func:`run_bootstrap_lrt_sweep`. The step-up rule is applied
        here (using ``usv_interval_cfg['bootstrap_lrt_alpha']`` and
        ``usv_interval_cfg['bootstrap_lrt_bonferroni']``) so the
        ``bootstrap_lrt`` table contains the same
        ``K_selected_step_up`` column as the CLI archive, and
        per-mode ``alpha_effective`` / ``K_selected_*`` attrs are
        recorded.
    message_output (callable)
        Logging callable; receives one summary line.

    Returns
    h5_path (pathlib.Path)
        Resolved path of the written archive.
    """

    out_dir = Path(configure_path(output_directory))
    out_dir.mkdir(parents=True, exist_ok=True)

    interval_types = ("s2s", "e2s")
    per_mode: dict[str, dict] = {}

    for it in interval_types:
        sub = usv_interval_df.filter(pls.col("interval_type") == it)
        # No intervals at all for this mode -- skip writing the group.
        if sub.height == 0 and (
            gmm_fits_by_mode is None or gmm_fits_by_mode.get(it) is None
        ) and (
            lrt_sweep_by_mode is None or lrt_sweep_by_mode.get(it) is None
        ):
            continue

        drops = usv_interval_summary.get("n_dropped", {}).get(it, {"male": 0, "female": 0})
        drop_df = pls.DataFrame([
            {"sex": "male",   "n_dropped": int(drops.get("male", 0))},
            {"sex": "female", "n_dropped": int(drops.get("female", 0))},
        ])

        mode_payload: dict = {
            "attrs": {},
            "intervals": sub,
            "drop_counts": drop_df,
            "gmm_fits": None,
            "bootstrap_lrt": None,
            "bootstrap_lrt_null": None,
        }

        gmm_fits = (gmm_fits_by_mode or {}).get(it)
        if gmm_fits is not None:
            mode_payload["gmm_fits"] = gmm_fits

        sweep = (lrt_sweep_by_mode or {}).get(it)
        if sweep:
            alpha = float(usv_interval_cfg["bootstrap_lrt_alpha"])
            bonferroni = bool(usv_interval_cfg["bootstrap_lrt_bonferroni"])
            lrt_rows: list[dict] = []
            null_rows: list[dict] = []
            selected_per_sex: dict[str, int] = {}
            alpha_eff_for_attr: float = alpha
            for sex, pair_results in sweep.items():
                n_tests = len(pair_results)
                alpha_eff = (alpha / n_tests) if (bonferroni and n_tests > 0) else alpha
                alpha_eff_for_attr = float(alpha_eff)
                K_sel = select_n_components_step_up_lrt(pair_results, alpha=alpha_eff)
                selected_per_sex[sex] = int(K_sel)
                for (K_n, K_a), res in pair_results.items():
                    lrt_rows.append({
                        "sex": sex,
                        "K_null": int(res["K_null"]),
                        "K_alt": int(res["K_alt"]),
                        "lr_obs": float(res["lr_obs"]),
                        "null_mean": float(res["null_mean"]),
                        "null_p95": float(res["null_p95"]),
                        "null_max": float(res["null_max"]),
                        "p_value": float(res["p_value"]),
                        "B": int(res["B"]),
                        "n_subsample": int(res["n_subsample"]),
                        "model_class": str(res["model_class"]),
                        "alpha_used": float(alpha_eff),
                        "K_selected_step_up": int(K_sel),
                    })
                    for b_idx, lr_b in enumerate(res["lr_null"]):
                        null_rows.append({
                            "sex": sex,
                            "K_null": int(K_n),
                            "K_alt": int(K_a),
                            "b": int(b_idx),
                            "lr_b": float(lr_b),
                        })
            if lrt_rows:
                mode_payload["bootstrap_lrt"] = pls.DataFrame(lrt_rows)
            if null_rows:
                mode_payload["bootstrap_lrt_null"] = pls.DataFrame(null_rows)
            mode_payload["attrs"]["alpha_effective"] = alpha_eff_for_attr
            mode_payload["attrs"]["K_selected_male"] = selected_per_sex.get("male", -1)
            mode_payload["attrs"]["K_selected_female"] = selected_per_sex.get("female", -1)

        per_mode[it] = mode_payload

    run_started_at = datetime.now()
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")

    analysis_attrs: dict = {
        "created_at_iso": run_started_at.isoformat(timespec="seconds"),
        # Use this module's location to seed the repo-root walk, not
        # ``out_dir`` (which is typically a user-chosen results
        # directory outside the repo and would always resolve to
        # "unknown").
        "git_sha": git_sha_for_provenance(Path(__file__).resolve().parent),
        "source_lists": [str(p) for p in usv_interval_cfg.get("session_lists", [])],
        "n_sessions_loaded": int(usv_interval_summary.get("n_sessions_loaded", 0)),
        "noise_col_id": usv_interval_cfg["noise_col_id"],
        "noise_categories": list(usv_interval_cfg["noise_categories"]),
        "fit_gmm": bool(usv_interval_cfg["fit_gmm"]),
        "n_components_min": int(usv_interval_cfg["n_components_min"]),
        "n_components_max": int(usv_interval_cfg["n_components_max"]),
        "n_repeats": int(usv_interval_cfg["n_repeats"]),
        "max_modes_reported": int(usv_interval_cfg["max_modes_reported"]),
        "random_seed_base": int(usv_interval_cfg["random_seed_base"]),
        "cv_n_folds": int(usv_interval_cfg["cv_n_folds"]),
        "cv_n_init": int(usv_interval_cfg["cv_n_init"]),
        "gmm_n_init": int(usv_interval_cfg["gmm_n_init"]),
        "gmm_reg_covar": float(usv_interval_cfg["gmm_reg_covar"]),
        "tau": float(usv_interval_cfg["tau"]),
        "model_class": str(usv_interval_cfg["model_class"]),
        "bootstrap_lrt_B": int(usv_interval_cfg["bootstrap_lrt_B"]),
        "bootstrap_lrt_n_subsample": int(usv_interval_cfg["bootstrap_lrt_n_subsample"]),
        "bootstrap_lrt_alpha": float(usv_interval_cfg["bootstrap_lrt_alpha"]),
        "bootstrap_lrt_bonferroni": bool(usv_interval_cfg["bootstrap_lrt_bonferroni"]),
    }

    h5_path = out_dir / f"usv_interval_analysis_{run_ts}.h5"
    write_ivi_h5(h5_path, analysis_attrs=analysis_attrs, per_mode=per_mode)
    message_output(f"  archive -> {h5_path}")
    return h5_path


def find_latest_archive(output_directory: str) -> Path:
    """
    Description
    Locates the most-recent ``usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5`` in
    ``output_directory``. Filenames are sorted lexicographically;
    because the timestamp is a fixed-width zero-padded string, this is
    equivalent to chronological order.

    Parameters
    output_directory (str)
        Directory to scan; passed through :func:`configure_path`.

    Returns
    h5_path (pathlib.Path)
        Path of the newest archive.
    """

    out_dir = Path(configure_path(output_directory))
    candidates = sorted(out_dir.glob("usv_interval_analysis_*.h5"))
    if not candidates:
        msg = (
            f"find_latest_archive: no usv_interval_analysis_*.h5 in {out_dir}. "
            "Run the compute cells (or the generate-usv-interval-distributions CLI) first."
        )
        raise FileNotFoundError(
            msg
        )
    return candidates[-1]


# HDF5-driven plotting entry points: re-render figures from the single
# self-describing archive written by InterUSVIntervalCalculator. The compute path
# writes one ``usv_interval_analysis_<ts>.h5``; the plot cells call these helpers
# to load and render. This is the seam separating computation from
# visualisation in the notebook.


def load_lrt_sweep_from_h5(
    h5_path: str,
    interval_type: str,
) -> dict:
    """
    Description
    Re-hydrates the in-memory ``sweep`` dict (the same shape produced
    by :func:`run_bootstrap_lrt_sweep`) from the two tables stored
    inside the inter-USV interval HDF5 archive: ``/<mode>/bootstrap_lrt`` for the
    per-pair summary stats and ``/<mode>/bootstrap_lrt_null`` for the
    long-form null distributions used in :func:`plot_bootstrap_lrt_panel`.

    Parameters
    h5_path (str)
        Path to the ``usv_interval_analysis_<ts>.h5`` archive (passed through
        :func:`configure_path`).
    interval_type (str)
        Mode label, ``'s2s'`` or ``'e2s'``.

    Returns
    sweep (dict)
        Mapping ``sex -> {(K_null, K_alt) -> result_dict}`` with
        every key the compute path wrote, including ``'lr_null'``
        as a numpy array.
    """

    archive = read_usv_interval_h5(h5_path)
    mode = archive["modes"].get(interval_type)
    if mode is None:
        msg = (
            f"load_lrt_sweep_from_h5: interval_type={interval_type!r} "
            f"not found in archive {h5_path}."
        )
        raise ValueError(
            msg
        )
    summary_df = mode.get("bootstrap_lrt")
    null_df = mode.get("bootstrap_lrt_null")
    if summary_df is None or null_df is None:
        msg = (
            f"load_lrt_sweep_from_h5: archive {h5_path} is missing the "
            f"bootstrap-LRT tables for interval_type={interval_type!r} "
            "(was fit_gmm=true when the archive was written?)."
        )
        raise ValueError(
            msg
        )

    sweep: dict = {}
    for row in summary_df.iter_rows(named=True):
        sex = row["sex"]
        K_n = int(row["K_null"])
        K_a = int(row["K_alt"])
        sweep.setdefault(sex, {})
        sub = null_df.filter(
            (pls.col("sex") == sex)
            & (pls.col("K_null") == K_n)
            & (pls.col("K_alt") == K_a)
        ).sort("b")
        lr_null = sub["lr_b"].to_numpy()
        sweep[sex][(K_n, K_a)] = {
            "K_null": K_n,
            "K_alt": K_a,
            "B": int(row["B"]),
            "n_subsample": int(row["n_subsample"]),
            "model_class": str(row["model_class"]),
            "lr_obs": float(row["lr_obs"]),
            "lr_null": lr_null,
            "p_value": float(row["p_value"]),
            "null_mean": float(row["null_mean"]),
            "null_p95": float(row["null_p95"]),
            "null_max": float(row["null_max"]),
        }
    return sweep


def selected_K_from_h5(
    h5_path: str,
    interval_type: str,
) -> dict:
    """
    Description
    Reads the per-sex step-up selected K from the mode group's
    ``/<mode>/attrs`` (``K_selected_male`` / ``K_selected_female``)
    in an inter-USV interval HDF5 archive and returns it as a ``{sex: K}`` dict ready
    to pass into :func:`plot_ic_curves` as the
    ``selected_n_components`` argument.

    Falls back to scanning the ``bootstrap_lrt`` table's
    ``K_selected_step_up`` column when the attributes are absent
    (older archives).

    Parameters
    h5_path (str)
        Path to the inter-USV interval archive.
    interval_type (str)
        Mode label.

    Returns
    selected (dict)
        Mapping ``sex -> int K_selected``.
    """

    archive = read_usv_interval_h5(h5_path)
    mode = archive["modes"].get(interval_type)
    if mode is None:
        msg = (
            f"selected_K_from_h5: interval_type={interval_type!r} "
            f"not found in archive {h5_path}."
        )
        raise ValueError(
            msg
        )

    attrs = mode.get("attrs", {})
    selected: dict = {}
    for sex_key, attr_key in (("male", "K_selected_male"), ("female", "K_selected_female")):
        val = attrs.get(attr_key)
        if val is not None and int(val) > 0:
            selected[sex_key] = int(val)

    if selected:
        return selected

    # Fallback: derive from the bootstrap_lrt table itself.
    df = mode.get("bootstrap_lrt")
    if df is None or df.height == 0:
        return {}
    for sex in df["sex"].unique().to_list():
        vals = df.filter(pls.col("sex") == sex)["K_selected_step_up"].unique().to_list()
        if vals:
            selected[sex] = int(vals[0])
    return selected


def load_intervals_from_h5(
    h5_path: str,
    interval_type: str,
) -> pls.DataFrame:
    """
    Description
    Convenience loader for the tidy per-interval table archived inside
    an inter-USV interval HDF5 file. Returns a polars DataFrame with the same schema
    as the in-memory frame from :func:`build_master_usv_interval_dataframe`,
    pre-filtered to a single ``interval_type``.

    Parameters
    h5_path (str)
        Path to the inter-USV interval archive.
    interval_type (str)
        Mode label.

    Returns
    df (pls.DataFrame)
        One row per inter-USV interval for the given interval_type.
    """

    archive = read_usv_interval_h5(h5_path)
    mode = archive["modes"].get(interval_type)
    if mode is None:
        msg = (
            f"load_intervals_from_h5: interval_type={interval_type!r} "
            f"not found in archive {h5_path}."
        )
        raise ValueError(
            msg
        )
    return mode["intervals"]


def load_gmm_fits_from_h5(
    h5_path: str,
    interval_type: str,
) -> pls.DataFrame:
    """
    Description
    Convenience loader for the GMM / t-mixture sweep table archived
    inside an inter-USV interval HDF5 file. Returns a polars DataFrame ready to pass
    to :func:`plot_ic_curves`.

    Parameters
    h5_path (str)
        Path to the inter-USV interval archive.
    interval_type (str)
        Mode label.

    Returns
    df (pls.DataFrame)
        Tidy GMM/t-mixture sweep results, including the per-component
        parameter columns (``logmean_k`` / ``logsd_k`` / ``weight_k``
        / ``nu_k``) needed to reconstruct fitted models without
        refitting.
    """

    archive = read_usv_interval_h5(h5_path)
    mode = archive["modes"].get(interval_type)
    if mode is None:
        msg = (
            f"load_gmm_fits_from_h5: interval_type={interval_type!r} "
            f"not found in archive {h5_path}."
        )
        raise ValueError(
            msg
        )
    df = mode.get("gmm_fits")
    if df is None:
        msg = (
            f"load_gmm_fits_from_h5: archive {h5_path} contains no "
            f"GMM sweep for interval_type={interval_type!r} "
            "(was fit_gmm=true when the archive was written?)."
        )
        raise ValueError(
            msg
        )
    return df


def load_best_fit_from_h5(
    h5_path: str,
    interval_type: str,
    sex: str,
    K: int,
    ic_col: str = "cv_neg_loglik",
):
    """
    Description
    Reconstructs the best-rep fitted mixture for ``(sex, K)`` from the
    archived sweep, without re-running EM. Returns a
    ``(model, gmm_order)`` pair shaped exactly like the live compute
    path returns, suitable for direct use by
    :func:`plot_best_fit_with_annotations` and :func:`plot_qq`.

    The ``model_class`` (``'gauss'`` / ``'t'``) is inferred from the
    archived row, so callers do not have to know which family the run
    used. Components are returned in ascending log-mean order; the
    returned ``gmm_order`` is therefore ``np.arange(K)``.

    Parameters
    h5_path (str)
        Path to the inter-USV interval archive.
    interval_type (str)
        Mode label.
    sex (str)
        ``'male'`` / ``'female'``.
    K (int)
        Number of components for the row to extract.
    ic_col (str)
        Information criterion used to pick the best rep within
        ``(sex, K)``; defaults to ``'cv_neg_loglik'``.

    Returns
    model (GaussianMixture | TMixture)
        Reconstructed mixture; ``score_samples`` etc. work directly.
    gmm_order (np.ndarray)
        ``np.arange(K)`` (model is pre-sorted by ascending log-mean).
    """

    df = load_gmm_fits_from_h5(h5_path, interval_type)
    return reconstruct_best_model(df, sex=sex, K=int(K), ic_col=ic_col)
