"""
@author: bartulem
Figures and text summaries for the USV-neuronal-coactivity analyses.

The compute lives in :mod:`usv_playpen.analyses.neuronal_coactivity_engine`;
this module turns the engine's result objects into the notebook's figures and
printed tables. Function-style plotters (each returns a
``matplotlib.figure.Figure``):

- ``plot_acoustic_confound`` — overlaid per-feature density histograms
  checking whether the two groups are matched in amplitude / frequency.
- ``plot_null_distributions`` — 3 metrics x 2 groups grid of chained-null
  histograms with each group's observed pooled-bootstrap mean overlaid.
- ``plot_per_session_pop_corr`` — one panel per session: the within-session
  circular-shuffle null of ``pop_corr`` with both groups' observed values.
- ``plot_cross_animal_slope`` — per-animal slope plot connecting
  ``pop_corr(group A)`` to ``pop_corr(group B)``.
- ``plot_amplitude_stratified`` — ``pop_corr`` per RMS-amplitude bin for both
  groups, with the unstratified means as reference lines.

Plus text-summary builders (each returns a ``str``): ``summarize_acoustic_confound``,
``summarize_group_comparison`` and ``summarize_amplitude_stratified``.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from usv_playpen.analyses.neuronal_coactivity_engine import (
    bootstrap_vs_null_stats,
    cohens_d,
)

# Default group / null / threshold colours (hex), read from the shared
# `coactivity_colors` block of `visualizations_settings.json` (matching the
# notebook) rather than hard-coded here. A missing file (partial install) falls
# back to the shipped literals so the module can still be imported.
_VIZ_SETTINGS_PATH = (
    pathlib.Path(__file__).parent.parent
    / "_parameter_settings" / "visualizations_settings.json"
)
try:
    with _VIZ_SETTINGS_PATH.open() as _vf:
        _COACTIVITY_COLORS = json.load(_vf)["coactivity_colors"]
except FileNotFoundError:
    _COACTIVITY_COLORS = {
        "group_a": "#DC143C", "group_b": "#1E90FF",
        "null": "#808080", "threshold": "#000000",
    }
COACTIVITY_GROUP_A_COLOR = _COACTIVITY_COLORS["group_a"]      # crimson
COACTIVITY_GROUP_B_COLOR = _COACTIVITY_COLORS["group_b"]      # dodgerblue
COACTIVITY_NULL_COLOR = _COACTIVITY_COLORS["null"]            # gray
COACTIVITY_THRESHOLD_COLOR = _COACTIVITY_COLORS["threshold"]  # black

# The three coactivity metrics and their human-readable panel titles.
_METRICS = ("r_sc", "similarity", "pop_corr")
_METRIC_TITLES = {
    "r_sc": "Pairwise Correlation ($r_{sc}$)",
    "similarity": "Cosine Similarity",
    "pop_corr": "Pop Vector Corr (Pearson)",
}


def _null_panel(
    ax,
    null_values,
    observed,
    *,
    null_color=COACTIVITY_NULL_COLOR,
    threshold_color=COACTIVITY_THRESHOLD_COLOR,
    null_label="Chained Null",
    label_threshold=True,
    percentile=99,
    bins=40,
    null_alpha=0.5,
    obs_linewidth=3.0,
    threshold_linewidth=1.0,
    legend_fontsize=8,
    legend_loc="best",
):
    """
    Description
    -----------
    Draw the recurring null-distribution panel used by
    :func:`plot_null_distributions` and :func:`plot_per_session_pop_corr`: a
    histogram of ``null_values``, a dashed vertical line at their ``percentile``,
    and one solid vertical line per observed value. Top / right spines are removed
    and a legend is added.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    null_values : numpy.ndarray
        Null-distribution samples to histogram.
    observed : list[tuple[float, str, str]]
        ``(value, hex_color, legend_label)`` triples, one solid line each.
    null_color, threshold_color : str
        Hex colours for the histogram and the percentile line.
    null_label : str | None
        Legend label for the histogram; ``None`` omits it.
    label_threshold : bool
        Whether the percentile line carries a legend label.
    percentile : float
        Percentile for the dashed threshold line (default 99).
    bins, null_alpha, obs_linewidth, threshold_linewidth, legend_fontsize, legend_loc
        Style knobs for the histogram, lines and legend.

    Returns
    -------
    float
        The computed ``percentile`` threshold of ``null_values``.
    """

    hist_kwargs = {"label": null_label} if null_label else {}
    ax.hist(null_values, bins=bins, color=null_color, alpha=null_alpha, **hist_kwargs)
    threshold = float(np.percentile(null_values, percentile))
    thr_kwargs = {"label": f"{percentile}%"} if label_threshold else {}
    ax.axvline(threshold, color=threshold_color, linestyle="--", linewidth=threshold_linewidth, **thr_kwargs)
    for value, color, label in observed:
        ax.axvline(value, color=color, linewidth=obs_linewidth, label=label)
    ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return threshold


def plot_acoustic_confound(
    group_a_acoustics,
    group_b_acoustics,
    *,
    features,
    feature_labels,
    chosen_animal,
    label_a="complex",
    label_b="simple",
    group_a_color=COACTIVITY_GROUP_A_COLOR,
    group_b_color=COACTIVITY_GROUP_B_COLOR,
) -> plt.Figure:
    """
    Description
    -----------
    Overlaid density histograms of each acoustic feature for the two groups, one
    subplot per feature, to eyeball whether the groups are matched in amplitude /
    frequency (the statistics live in :func:`summarize_acoustic_confound`).

    Parameters
    ----------
    group_a_acoustics, group_b_acoustics : dict[str, numpy.ndarray]
        Per-feature pooled per-call values for each group.
    features : sequence[str]
        Feature keys to plot, in order.
    feature_labels : dict[str, str]
        Axis label per feature key.
    chosen_animal : str
        Animal id for the figure title.
    label_a, label_b : str
        Group labels.
    group_a_color, group_b_color : str
        Hex colours per group.

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, axes = plt.subplots(1, len(features), figsize=(4.0 * len(features), 3.2))
    for ax, feature in zip(axes, features, strict=True):
        a = group_a_acoustics[feature][np.isfinite(group_a_acoustics[feature])]
        b = group_b_acoustics[feature][np.isfinite(group_b_acoustics[feature])]
        pooled = np.concatenate([a, b]) if (a.size or b.size) else np.array([0.0, 1.0])
        lo, hi = float(pooled.min()), float(pooled.max())
        bins = np.linspace(lo, hi, 40) if hi > lo else 40
        ax.hist(a, bins=bins, density=True, alpha=0.5, color=group_a_color, label=label_a)
        ax.hist(b, bins=bins, density=True, alpha=0.5, color=group_b_color, label=label_b)
        ax.set_xlabel(feature_labels[feature], fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"30 ms acoustic features — {label_a} vs {label_b} ({chosen_animal})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_null_distributions(
    results,
    *,
    category_column,
    group_a_ids,
    group_b_ids,
    label_a="complex",
    label_b="simple",
    group_a_color=COACTIVITY_GROUP_A_COLOR,
    group_b_color=COACTIVITY_GROUP_B_COLOR,
    null_color=COACTIVITY_NULL_COLOR,
    threshold_color=COACTIVITY_THRESHOLD_COLOR,
) -> plt.Figure:
    """
    Description
    -----------
    A 3 (metric) x 2 (group) grid of chained-null histograms, each overlaid with
    the group's observed pooled-bootstrap mean and the null's 99th percentile.

    Parameters
    ----------
    results : dict
        A :func:`run_group_comparison` result (uses ``boot_a`` / ``boot_b`` /
        ``chained_null_a`` / ``chained_null_b``).
    category_column : str
        Segmentation column name, for the figure title.
    group_a_ids, group_b_ids : sequence
        Category ids per group, for the figure title.
    label_a, label_b : str
        Group labels.
    group_a_color, group_b_color, null_color, threshold_color : str
        Hex colours.

    Returns
    -------
    matplotlib.figure.Figure
    """

    boot_a, boot_b = results["boot_a"], results["boot_b"]
    null_a, null_b = results["chained_null_a"], results["chained_null_b"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharey="row")
    for row, metric in enumerate(_METRICS):
        title = _METRIC_TITLES[metric]
        a_val = float(np.mean(boot_a[metric]))
        b_val = float(np.mean(boot_b[metric]))
        _null_panel(
            axes[row, 0], null_a[metric],
            [(a_val, group_a_color, f"Pooled Boot ({a_val:.4f})")],
            null_color=null_color, threshold_color=threshold_color,
        )
        axes[row, 0].set_title(f"{label_a.upper()}: {title}", fontsize=14)
        _null_panel(
            axes[row, 1], null_b[metric],
            [(b_val, group_b_color, f"Pooled Boot ({b_val:.4f})")],
            null_color=null_color, threshold_color=threshold_color,
        )
        axes[row, 1].set_title(f"{label_b.upper()}: {title}", fontsize=14)
        if row == 2:
            for col in (0, 1):
                axes[row, col].set_xlabel("Mean correlation / similarity", fontsize=12)
    fig.suptitle(
        f"Coactivity by `{category_column}`  ·  "
        f"{label_a} (IDs={group_a_ids}) vs {label_b} (IDs={group_b_ids})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_per_session_pop_corr(
    per_session_rows,
    *,
    chosen_animal,
    category_column,
    label_a="complex",
    label_b="simple",
    group_a_color=COACTIVITY_GROUP_A_COLOR,
    group_b_color=COACTIVITY_GROUP_B_COLOR,
    null_color=COACTIVITY_NULL_COLOR,
    threshold_color=COACTIVITY_THRESHOLD_COLOR,
) -> plt.Figure | None:
    """
    Description
    -----------
    One panel per session showing the within-session circular-shuffle null of
    ``pop_corr`` (both groups' onsets pooled) with the observed group-A and group-B
    ``pop_corr`` overlaid as coloured lines and the null's 99th percentile dashed.
    Rows must carry a ``null`` block (i.e. produced with ``n_shuffles`` set).

    Parameters
    ----------
    per_session_rows : list[dict]
        :func:`per_session_group_metrics` rows (need ``null`` / ``metrics_a`` /
        ``metrics_b`` / ``session_id`` / ``n_a`` / ``n_b``).
    chosen_animal, category_column : str
        For the figure title.
    label_a, label_b : str
        Group labels.
    group_a_color, group_b_color, null_color, threshold_color : str
        Hex colours.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when no session carries a null block (nothing to draw).
    """

    rows = [row for row in per_session_rows if "null" in row]
    if not rows:
        return None
    n_panels = len(rows)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows),
        sharex=False, sharey=False, squeeze=False,
    )
    for idx, row in enumerate(rows):
        ax = axes[idx // n_cols, idx % n_cols]
        pop_a = row["metrics_a"]["pop_corr"]
        pop_b = row["metrics_b"]["pop_corr"]
        _null_panel(
            ax, row["null"]["pop_corr"],
            [
                (pop_a, group_a_color, f"{label_a} (n={row['n_a']})  {pop_a:.3f}"),
                (pop_b, group_b_color, f"{label_b} (n={row['n_b']})  {pop_b:.3f}"),
            ],
            null_color=null_color, threshold_color=threshold_color,
            null_label=None, label_threshold=False, bins=30, null_alpha=0.55,
            obs_linewidth=2.2, threshold_linewidth=0.8, legend_fontsize=7, legend_loc="upper right",
        )
        ax.set_title(row["session_id"], fontsize=9)
        ax.set_xlabel("pop_corr", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(
        f"Per-session pop_corr — {chosen_animal}  ·  {label_a} vs {label_b} ({category_column})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_cross_animal_slope(
    cross_animal_results,
    *,
    category_column,
    group_a_ids,
    group_b_ids,
    label_a="complex",
    label_b="simple",
    group_a_color=COACTIVITY_GROUP_A_COLOR,
    group_b_color=COACTIVITY_GROUP_B_COLOR,
    null_color=COACTIVITY_NULL_COLOR,
    threshold_color=COACTIVITY_THRESHOLD_COLOR,
    sig_alpha=0.05,
) -> plt.Figure:
    """
    Description
    -----------
    Per-animal slope plot: one line per animal connecting its bootstrap-mean
    ``pop_corr(group A)`` (left) to ``pop_corr(group B)`` (right). Significant
    animals (``p_two < sig_alpha``) are coloured by direction (A>B vs B>A), n.s.
    animals gray, and the aggregate mean is overlaid as a dashed line.

    Parameters
    ----------
    cross_animal_results : dict[str, dict]
        Per-animal records with ``pop_a`` / ``pop_b`` / ``p_two``.
    category_column : str
        Segmentation column name, for the title.
    group_a_ids, group_b_ids : sequence
        Category ids per group, for the x tick labels.
    label_a, label_b : str
        Group labels.
    group_a_color, group_b_color, null_color, threshold_color : str
        Hex colours.
    sig_alpha : float
        Two-tailed significance threshold (default 0.05).

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for animal_id, record in cross_animal_results.items():
        delta = record["pop_a"] - record["pop_b"]
        if record["p_two"] < sig_alpha:
            line_color = group_a_color if delta > 0 else group_b_color
            line_alpha = 0.95
        else:
            line_color = null_color
            line_alpha = 0.55
        ax.plot([0, 1], [record["pop_a"], record["pop_b"]], color=line_color, alpha=line_alpha, linewidth=2.0, marker="o")
        ax.text(
            1.03, record["pop_b"],
            f"{animal_id}  (p={record['p_two']:.3f}{', *' if record['p_two'] < sig_alpha else ''})",
            fontsize=8, va="center", color=line_color,
        )
    mean_a = float(np.mean([r["pop_a"] for r in cross_animal_results.values()]))
    mean_b = float(np.mean([r["pop_b"] for r in cross_animal_results.values()]))
    ax.plot([0, 1], [mean_a, mean_b], color=threshold_color, linewidth=3.0, linestyle="--", label="mean of animals")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{label_a}\n(IDs={group_a_ids})", f"{label_b}\n(IDs={group_b_ids})"])
    ax.set_xlim(-0.15, 1.6)
    ax.set_ylabel("pop_corr (bootstrap mean)", fontsize=10)
    ax.set_title(f"Cross-animal pop_corr  ·  {category_column}  ·  N={len(cross_animal_results)} mice", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_amplitude_stratified(
    stratified_rows,
    pop_a_overall,
    pop_b_overall,
    *,
    chosen_animal,
    label_a="complex",
    label_b="simple",
    group_a_color=COACTIVITY_GROUP_A_COLOR,
    group_b_color=COACTIVITY_GROUP_B_COLOR,
    threshold_color=COACTIVITY_THRESHOLD_COLOR,
) -> plt.Figure:
    """
    Description
    -----------
    ``pop_corr`` per RMS-amplitude bin for both groups (log x, bin geometric
    centres), with the unstratified means drawn as dashed reference lines and a
    ``*`` over bins where the within-bin A-vs-B two-tailed p < 0.05.

    Parameters
    ----------
    stratified_rows : list[dict]
        Per-bin records with ``lo`` / ``hi`` / ``pop_a`` / ``pop_b`` / ``p_two``
        (NaN ``pop_a`` marks a bin with too few trials, skipped here).
    pop_a_overall, pop_b_overall : float
        Unstratified bootstrap-mean ``pop_corr`` per group.
    chosen_animal : str
        For the title.
    label_a, label_b : str
        Group labels.
    group_a_color, group_b_color, threshold_color : str
        Hex colours.

    Returns
    -------
    matplotlib.figure.Figure
    """

    valid = [row for row in stratified_rows if not np.isnan(row["pop_a"])]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if valid:
        centers = [float(np.sqrt(row["lo"] * row["hi"])) for row in valid]
        ax.plot(centers, [row["pop_a"] for row in valid], "-o", color=group_a_color, label=label_a)
        ax.plot(centers, [row["pop_b"] for row in valid], "-o", color=group_b_color, label=label_b)
        for row, center in zip(valid, centers, strict=True):
            if row["p_two"] < 0.05:
                ax.annotate("*", (center, max(row["pop_a"], row["pop_b"])), ha="center", va="bottom", fontsize=14, color=threshold_color)
        ax.set_xscale("log")
        ax.axhline(pop_a_overall, color=group_a_color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(pop_b_overall, color=group_b_color, linestyle="--", linewidth=1, alpha=0.6)
        ax.legend(frameon=False, fontsize=9)   # only labelled artists exist in the valid-bin branch
    else:
        ax.text(0.5, 0.5, "No amplitude bin held enough trials in both groups", ha="center", transform=ax.transAxes)
    ax.set_xlabel("30 ms RMS amplitude (bin geometric centre, log)", fontsize=10)
    ax.set_ylabel("pop_corr (bootstrap mean, matched N per bin)", fontsize=10)
    ax.set_title(
        f"Amplitude-stratified pop_corr — {chosen_animal}\n"
        f"dashed = unstratified means; * = within-bin {label_a}-vs-{label_b} p<0.05",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def summarize_acoustic_confound(
    group_a_acoustics,
    group_b_acoustics,
    *,
    features,
    chosen_animal,
    label_a="complex",
    label_b="simple",
) -> str:
    """
    Description
    -----------
    Per-feature Mann-Whitney U (distribution shift) + Cohen's d (effect size)
    comparison of the two groups' acoustics, returned as a printable table.

    Parameters
    ----------
    group_a_acoustics, group_b_acoustics : dict[str, numpy.ndarray]
        Per-feature pooled per-call values for each group.
    features : sequence[str]
        Feature keys to report.
    chosen_animal : str
        Animal id for the header.
    label_a, label_b : str
        Group labels.

    Returns
    -------
    str
        Multi-line summary table.
    """

    n_a = int(np.isfinite(group_a_acoustics[features[0]]).sum())
    n_b = int(np.isfinite(group_b_acoustics[features[0]]).sum())
    lines = [f"Acoustic confound check — {chosen_animal}: {label_a} (n={n_a}) vs {label_b} (n={n_b})", ""]
    for feature in features:
        a = group_a_acoustics[feature][np.isfinite(group_a_acoustics[feature])]
        b = group_b_acoustics[feature][np.isfinite(group_b_acoustics[feature])]
        if a.size < 2 or b.size < 2:
            lines.append(f"  {feature:18s}: insufficient data ({a.size} / {b.size})")
            continue
        _, p_value = st.mannwhitneyu(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        lines.append(
            f"  {feature:18s}: {label_a} median={np.median(a):>10.4g}  "
            f"{label_b} median={np.median(b):>10.4g}  Cohen's d={d:+.3f}  Mann-Whitney p={p_value:.2e}"
        )
    return "\n".join(lines)


def summarize_group_comparison(results, *, label_a="complex", label_b="simple") -> str:
    """
    Description
    -----------
    Printable summary of a :func:`run_group_comparison` result: the per-session
    observed A-minus-B deltas, the count of sessions with A>B per metric, each
    group's pooled-bootstrap-vs-chained-null p / Z (via
    :func:`bootstrap_vs_null_stats`), and the direct A-vs-B permutation test.

    Parameters
    ----------
    results : dict
        A :func:`run_group_comparison` result.
    label_a, label_b : str
        Group labels.

    Returns
    -------
    str
        Multi-line summary tables.
    """

    n_target = results["bootstrap_n"]
    lines = ["Per-session observed metrics (no bootstrap / no shuffle):"]
    lines.append(f"  {'session':<22} {'n_a':>4} {'n_b':>4}   {'r_sc Δ':>10}   {'sim Δ':>10}   {'pop Δ':>10}")
    tally = dict.fromkeys(_METRICS, 0)
    counted = 0
    for row in results["per_session"]:
        deltas = row["deltas"]
        lines.append(
            f"  {row['session_id']:<22} {row['n_a']:>4} {row['n_b']:>4}"
            f"   {deltas['r_sc']:>+10.4f}   {deltas['similarity']:>+10.4f}   {deltas['pop_corr']:>+10.4f}"
        )
        counted += 1
        for metric in _METRICS:
            if deltas[metric] > 0:
                tally[metric] += 1
    lines.append("")
    lines.append(f"Sessions where {label_a} > {label_b}, per metric:")
    for metric in _METRICS:
        lines.append(f"  {metric:<12}: {tally[metric]}/{counted} sessions")

    for label, boot, null in (
        (label_a, results["boot_a"], results["chained_null_a"]),
        (label_b, results["boot_b"], results["chained_null_b"]),
    ):
        lines.append("")
        lines.append("=" * 75)
        lines.append(f"{label.upper()} vs CHAINED NULL  (N={n_target})")
        lines.append("=" * 75)
        for metric in _METRICS:
            obs, null_mean, p, z = bootstrap_vs_null_stats(boot[metric], null[metric])
            lines.append(f"  {metric:<12}: boot={obs:+.4f} | null={null_mean:+.4f} | p={p:.4f}  (Z={z:+.2f})")

    perm = results["perm"]
    lines.append("")
    lines.append("=" * 75)
    lines.append(f"DIRECT PERMUTATION: {label_a.upper()} vs {label_b.upper()}")
    lines.append("=" * 75)
    for metric in _METRICS:
        r = perm[metric]
        lines.append(
            f"  Δ {metric:<12}: obs={r['observed_delta']:+.4f} | null mean={r['null_mean']:+.4f} | "
            f"p({label_a}>{label_b})={r['p_a_gt_b']:.4f} | p_two_tail={r['p_two_tailed']:.4f}  (Z={r['z_score']:+.2f})"
        )
    return "\n".join(lines)


def summarize_amplitude_stratified(
    stratified_rows,
    pop_a_overall,
    pop_b_overall,
    *,
    chosen_animal,
    n_bins,
    label_a="complex",
    label_b="simple",
) -> str:
    """
    Description
    -----------
    Printable table of ``pop_corr`` per RMS-amplitude bin for both groups, with
    the unstratified reference means, matching :func:`plot_amplitude_stratified`.

    Parameters
    ----------
    stratified_rows : list[dict]
        Per-bin records (``lo`` / ``hi`` / ``n_a`` / ``n_b`` / ``pop_a`` /
        ``pop_b`` / ``p_two``).
    pop_a_overall, pop_b_overall : float
        Unstratified bootstrap-mean ``pop_corr`` per group.
    chosen_animal : str
        Animal id for the header.
    n_bins : int
        Number of amplitude quantile bins.
    label_a, label_b : str
        Group labels.

    Returns
    -------
    str
        Multi-line summary table.
    """

    lines = [f"Amplitude-stratified pop_corr — {chosen_animal}  ({n_bins} RMS quantile bins)"]
    lines.append(
        f"  unstratified: {label_a} pop_corr={pop_a_overall:+.4f}  "
        f"{label_b}={pop_b_overall:+.4f}  Δ={pop_a_overall - pop_b_overall:+.4f}"
    )
    lines.append("")
    lines.append(f"  {'RMS bin':>24} {'n_a':>5} {'n_b':>5}  {'pop_a':>8} {'pop_b':>8} {'Δ':>8} {'p_two':>7}")
    for row in stratified_rows:
        bin_label = f"[{row['lo']:.2g}, {row['hi']:.2g})"
        if np.isnan(row["pop_a"]):
            lines.append(f"  {bin_label:>24} {row['n_a']:>5} {row['n_b']:>5}   (too few trials in a group)")
        else:
            lines.append(
                f"  {bin_label:>24} {row['n_a']:>5} {row['n_b']:>5}  "
                f"{row['pop_a']:>+8.4f} {row['pop_b']:>+8.4f} {row['pop_a'] - row['pop_b']:>+8.4f} {row['p_two']:>7.4f}"
            )
    return "\n".join(lines)
