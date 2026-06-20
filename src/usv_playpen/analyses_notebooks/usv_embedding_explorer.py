"""
Marimo notebook for interactively exploring USV embeddings (VAE / QLVM).

Usage
-----
Install the ``dev`` dependency group (it carries ``marimo`` and ``altair``,
both used by this notebook):

    pip install -e ".[dev]"

From the repo root, edit (reactive code view) or run (clean app view):

    uv run marimo edit src/usv_playpen/analyses_notebooks/usv_embedding_explorer.py
    uv run marimo run  src/usv_playpen/analyses_notebooks/usv_embedding_explorer.py

Settings (the ``usv_embedding`` block of ``visualizations_settings.json``)
supply the directory of session-list ``*.txt`` files and the consolidated
spectrogram/SAM2 store.

Architecture
------------
- Cell layout (the practical marimo minimum, 6 cells): imports; settings;
  widgets; pooled-data load; scatter (prepare + chart); explorer (controls +
  spectrogram grid). The scatter, the brushable ``mo.ui.altair_chart`` and the
  cell that reads its selection must stay separate -- marimo forbids reading a
  UI element's value in its defining cell -- and the load is isolated so a
  color/sample tweak never rebuilds the pooled DataFrame.
- Pick one or more session lists; their per-session ``usv_summary.csv`` rows are
  pooled (and cached to a per-selection parquet) by ``build_pooled_embeddings_df``.
- An altair scatter of the chosen embedding map (VAE UMAP or QLVM torus),
  colored by a categorical label (category / supercategory / session type /
  session id / emitter sex) OR a continuous metric through the colormap
  (density, duration, frequencies, amplitudes, spectral entropy), with an
  ``alt.selection_interval`` brush. Optional category-boundary contours overlay
  the scatter. The spec only embeds x / y / color, so hundreds of thousands of
  points fit under marimo's ``output_max_bytes``.
- Brushing samples spectrograms from the selection along an Archimedean spiral
  (centre -> edge) and renders them as a square grid to the RIGHT of the scatter
  -- read from the consolidated h5 for only the sampled rows, padded to the
  fixed window so each call's width reflects its true duration, embedded inline
  as a base64 PNG. The brushed rows are recovered with
  ``chart_widget.apply_selection`` (``.value`` fails on the layered chart).

All paths are run through ``os_utils.configure_path`` so ``/mnt/...`` strings
from cross-OS session lists resolve correctly on macOS / Linux.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _imports():
    import base64
    import hashlib
    import json
    from io import BytesIO
    from pathlib import Path

    import altair as alt
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pls

    alt.data_transformers.disable_max_rows()

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.make_usv_spectrograms import (
        _knn_boundary_grid as knn_boundary_grid,
        build_pooled_embeddings_df,
    )

    return (
        BytesIO,
        Path,
        alt,
        base64,
        build_pooled_embeddings_df,
        configure_path,
        h5py,
        hashlib,
        json,
        knn_boundary_grid,
        mo,
        np,
        pd,
        pls,
        plt,
    )


@app.cell
def _settings(Path, configure_path, json):
    # Cell 2 = ALL settings/config (imports are all in cell 1). Reads
    # visualizations_settings.json once and exposes everything downstream cells
    # need: the colormap, sex colors, the consolidated store path, the available
    # session lists (globbed from the input-files directory), and the scatter's
    # pixel dimensions. A missing file / block falls back gracefully.
    _settings_path = (
        Path(__file__).parent.parent
        / "_parameter_settings" / "visualizations_settings.json"
    )
    try:
        with _settings_path.open() as _vf:
            _viz = json.load(_vf)
    except (FileNotFoundError, json.JSONDecodeError):
        _viz = {}

    try:
        global_cmap = _viz["figures"]["cmap"]
    except KeyError:
        global_cmap = "inferno"

    # First male/female/unassigned colors drive the "emitter" coloring.
    try:
        sex_colors = {
            "male": _viz["male_colors"][0],
            "female": _viz["female_colors"][0],
            "unassigned": _viz["unassigned_colors"][0],
        }
    except (KeyError, IndexError):
        sex_colors = {"male": "#9AC0CD", "female": "#FF6347", "unassigned": "#9E9E9E"}

    # Session-list directory + consolidated store from the `usv_embedding` block
    # (canonical /mnt/falkner paths, resolved to the host mount via
    # configure_path). Glob every *.txt list into {label -> absolute path}.
    try:
        _emb = _viz["usv_embedding"]
        _input_dir = configure_path(_emb["input_files_directory"])
        consolidated_h5_path = configure_path(_emb["consolidated_h5_path"])
    except KeyError:
        _input_dir, consolidated_h5_path = None, None
    if _input_dir is not None and Path(_input_dir).is_dir():
        available_lists = {p.name: str(p) for p in sorted(Path(_input_dir).glob("*.txt"))}
    else:
        available_lists = {}

    # Scatter is kept SQUARE so the two embedding dimensions share one scale.
    CHART_DATA_WIDTH_PX = 520
    CHART_HEIGHT_PX = 520
    return (
        CHART_DATA_WIDTH_PX,
        CHART_HEIGHT_PX,
        available_lists,
        consolidated_h5_path,
        global_cmap,
        sex_colors,
    )


@app.cell
def _widgets(available_lists, mo):
    # Nothing selected by default: pooling reads one usv_summary.csv per session
    # over the share, so auto-building on launch (especially "all") can take
    # minutes. The user picks the list(s) to pool, and the loader only builds
    # once a selection exists. Each distinct selection is cached to local
    # parquet, so re-selecting it later reloads in seconds.
    lists_multiselect = mo.ui.multiselect(
        options=available_lists,
        label="Session lists",
    )
    map_dropdown = mo.ui.dropdown(
        options=["QLVM", "VAE"],
        value="QLVM",
        label="Map",
    )
    # Color by a CATEGORICAL label (category / supercategory) OR a CONTINUOUS
    # quantity rendered through the colormap (point density, or any per-USV
    # acoustic feature). Boundaries (below) are an independent, optional overlay.
    # {display label (no underscores, with units) -> internal value}. .value
    # returns the internal value, so the downstream logic is unchanged.
    color_dropdown = mo.ui.dropdown(
        options={
            "none": "none",
            "category (fine)": "category",
            "supercategory (coarse)": "supercategory",
            "session type": "session_type",
            "session (id)": "session",
            "emitter (sex)": "emitter",
            "density (counts)": "density",
            "duration (ms)": "duration",
            "mean frequency (kHz)": "mean_freq_hz",
            "peak frequency (kHz)": "peak_freq_hz",
            "frequency bandwidth (kHz)": "freq_bandwidth_hz",
            "mean amplitude (a.u.)": "mean_amplitude",
            "max amplitude (a.u.)": "max_amplitude",
            "spectral entropy (nats)": "spectral_entropy",
        },
        value="supercategory (coarse)",
        label="Color by",
    )
    # Boundaries draw the cluster outlines for the chosen categorical label
    # (category = fine, supercategory = coarse) as contour lines over the
    # scatter -- the discrete structure, without recoloring every point.
    boundary_dropdown = mo.ui.dropdown(
        options=["none", "category", "supercategory"],
        value="none",
        label="Boundaries",
    )
    # Sampled spectrograms shown in the grid beside the scatter (5 columns). The
    # default 50 makes a 5x10 grid whose block is square -- matching the
    # embedding's height exactly.
    n_samples_slider = mo.ui.slider(
        start=5, stop=50, step=5, value=50,
        label="Examples (spectrograms) plotted",
    )
    # Scatter is thumbnail-free: each point is ~50 bytes in the chart spec, so
    # hundreds of thousands of points fit inside the 30 MB cap. The slider tops
    # out at 500 K to leave headroom for altair rendering performance; above
    # ~100 K interaction starts to feel sluggish in some browsers.
    max_points_slider = mo.ui.slider(
        start=5_000, stop=500_000, step=5_000, value=50_000,
        label="Max points",
    )
    apply_mask_checkbox = mo.ui.checkbox(
        value=True,
        label="Apply mask",
    )
    # One control per row (label left, input right): full row width means the
    # marimo labels never wrap, and the (growing) multiselect only pushes the
    # rows below it down rather than shoving neighbours sideways.
    controls = mo.vstack(
        [
            lists_multiselect,
            map_dropdown,
            color_dropdown,
            boundary_dropdown,
            n_samples_slider,
            max_points_slider,
            apply_mask_checkbox,
        ],
        align="start",
        gap=0.4,
    )
    # NOTE: controls are displayed in the dedicated `_controls_display` cell
    # placed immediately above the plot (so they sit directly over it), not
    # here. That cell depends only on `controls`, so it still renders before any
    # selection -- independent of the data build below.
    return (
        apply_mask_checkbox,
        boundary_dropdown,
        color_dropdown,
        controls,
        lists_multiselect,
        map_dropdown,
        max_points_slider,
        n_samples_slider,
    )


@app.cell
def _load_pooled_df(
    Path,
    build_pooled_embeddings_df,
    hashlib,
    lists_multiselect,
    mo,
    pls,
):
    # Wrapped in an inner function so the no-selection case can early-return
    # None (a marimo cell body can't `return`, but a nested function can). No
    # selection -> pooled_df is None, which downstream cells short-circuit on
    # quietly (no mo.stop -> no "ancestor stopped" placeholder). The "pick a
    # list" hint is shown under the controls in `_controls_display`.
    def _():
        selected_paths = list(lists_multiselect.value)
        if not selected_paths:
            return None

        # Union the session roots across every selected list (dedup, skip blank
        # / # lines), recording each session's source list (its "type") for the
        # optional session_type coloring. Both the combined list file and the
        # parquet cache are keyed by a hash of the union, so each distinct
        # subset has its own cache.
        seen, lines = set(), []
        session_to_list = {}
        session_to_type = {}
        for _list_path in selected_paths:
            _list_label = Path(_list_path).stem
            # Classify the list by filename so the emitter -> sex mapping can be
            # corrected per session type. Most-specific first:
            # "courtship_male_male" must match male_male, not courtship. Unknown
            # lists (e.g. playback) keep the raw track-index convention.
            _ln = _list_label.lower()
            if "female_female" in _ln:
                _ltype = "female_female"
            elif "male_male" in _ln:
                _ltype = "male_male"
            elif "lone_male" in _ln:
                _ltype = "lone_male"
            elif "courtship" in _ln:
                _ltype = "male_female"
            else:
                _ltype = "other"
            try:
                _raw_lines = Path(_list_path).read_text().splitlines()
            except OSError as exc:
                mo.stop(
                    True,
                    mo.md(f"**Could not read** `{_list_path}`:\n\n```\n{exc}\n```"),
                )
            for _line in _raw_lines:
                _session = _line.strip()
                if _session and not _session.startswith("#") and _session not in seen:
                    seen.add(_session)
                    lines.append(_session)
                    # session_id is the session root basename; first list wins.
                    session_to_list[Path(_session).name] = _list_label
                    session_to_type[Path(_session).name] = _ltype

        selection_token = hashlib.md5("\n".join(sorted(seen)).encode()).hexdigest()[:12]
        cache_dir = Path.home() / ".usv_playpen_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        combined_list_path = cache_dir / f"sessions_combined_{selection_token}.txt"
        combined_list_path.write_text("\n".join(lines) + "\n")
        cache_path = str(cache_dir / f"pooled_embeddings_{selection_token}.parquet")

        try:
            pooled = build_pooled_embeddings_df(
                sessions_txt_path=str(combined_list_path),
                cache_path=cache_path,
                rebuild_cache=False,
                noise_col_id="vae_supercategory",
                noise_categories=(0,),
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            mo.stop(
                True,
                mo.md(f"**Could not build the pooled embeddings**:\n\n```\n{exc}\n```"),
            )
        # Tag every row with its source list (a marimo-side enrichment, not
        # cached); every pooled session came from a selected list.
        pooled = pooled.with_columns(
            pls.Series(
                "session_type",
                [session_to_list[_s] for _s in pooled["session_id"].to_list()],
            )
        )
        # Correct emitter sex by session type. build_pooled_embeddings_df
        # assigns sex purely by track index (0 = male, 1 = female), right only
        # for male-female sessions: female-female mislabels the track-0 animal
        # "male" (remap -> female) and male-male mislabels track-1 "female"
        # (remap -> male). Other types already come out correct.
        _types = [session_to_type[_s] for _s in pooled["session_id"].to_list()]
        _sexes = pooled["sex"].to_list()
        _corrected = []
        for _type, _sex in zip(_types, _sexes):
            if _type == "female_female" and _sex == "male":
                _corrected.append("female")
            elif _type == "male_male" and _sex == "female":
                _corrected.append("male")
            else:
                _corrected.append(_sex)
        return pooled.with_columns(pls.Series("sex", _corrected))

    pooled_df = _()
    return (pooled_df,)


@app.cell
def _scatter_chart(
    CHART_DATA_WIDTH_PX,
    CHART_HEIGHT_PX,
    alt,
    boundary_dropdown,
    color_dropdown,
    global_cmap,
    knn_boundary_grid,
    map_dropdown,
    max_points_slider,
    mo,
    np,
    pd,
    plt,
    pooled_df,
    sex_colors,
):
    # Inner function so the no-data case can early-return None (a marimo cell
    # body can't `return`). pooled_df is None until a list is picked. This cell
    # both prepares the display frame (map/color/boundary columns + sampling)
    # AND builds the chart -- they share the same reactive inputs.
    def _():
        if pooled_df is None:
            return None, None, None, None

        map_prefix = "vae" if map_dropdown.value == "VAE" else "qlvm"
        # QLVM torus coords are qlvm_dim1/qlvm_dim2 (not a UMAP); only VAE uses
        # the _umap1/_umap2 suffix.
        if map_prefix == "qlvm":
            x_col, y_col = "qlvm_dim1", "qlvm_dim2"
        else:
            x_col, y_col = "vae_umap1", "vae_umap2"

        # Color source: category/supercategory/session_type categorical; emitter
        # colors the derived sex column; density is computed below from the 2D
        # positions; the rest are continuous acoustic-feature columns.
        color_metric = color_dropdown.value
        if color_metric in ("category", "supercategory"):
            color_col, color_kind = f"{map_prefix}_{color_metric}", "categorical"
        elif color_metric == "session_type":
            color_col, color_kind = "session_type", "categorical"
        elif color_metric == "session":
            color_col, color_kind = "session_id", "categorical"
        elif color_metric == "emitter":
            color_col, color_kind = "sex", "emitter"
        elif color_metric == "density":
            color_col, color_kind = None, "density"
        elif color_metric == "none":
            color_col, color_kind = None, "none"
        else:
            color_col, color_kind = color_metric, "continuous"

        # Boundaries use the map-specific categorical label column (overlay).
        boundary_choice = boundary_dropdown.value
        boundary_col = (
            None if boundary_choice == "none" else f"{map_prefix}_{boundary_choice}"
        )

        keep = ["session_id", "row_index", x_col, y_col]
        # Dedupe: color-by "session" sets color_col == "session_id", which is
        # already in keep (likewise a color/boundary column could repeat).
        if color_col is not None and color_col not in keep:
            keep.append(color_col)
        if boundary_col is not None and boundary_col not in keep:
            keep.append(boundary_col)

        # Guard against a pooled frame whose CSVs predate the current schema.
        missing_cols = [c for c in keep if c not in pooled_df.columns]
        mo.stop(
            bool(missing_cols),
            mo.md(
                f"**Missing column(s)** {missing_cols} for the "
                f"**{map_prefix.upper()}** map. The pooled `usv_summary.csv`s do "
                "not carry these — re-run `infer-qlvm-latents` / "
                "`generate-usv-acoustic-features`, or rebuild the cache."
            ),
        )

        display_df = pooled_df.select(keep).drop_nulls(subset=[x_col, y_col])
        max_points = int(max_points_slider.value)
        if display_df.height > max_points:
            display_df = display_df.sample(n=max_points, seed=42)

        chart_pd = display_df.to_pandas()

        # No axes -- points alone. Keep the scales (data domain) but drop
        # ticks/labels/titles/spines via axis=None. Shared scales so the scatter
        # and the boundary overlay align exactly.
        if map_prefix == "qlvm":
            x_scale = alt.Scale(domain=[0.0, 1.0], nice=False)
            y_scale = alt.Scale(domain=[0.0, 1.0], nice=False)
        else:
            x_scale = alt.Scale(domain=[5, 18], nice=False)
            y_scale = alt.Scale(nice=False)
        x_enc = alt.X(x_col, type="quantitative", axis=None, scale=x_scale)
        y_enc = alt.Y(y_col, type="quantitative", axis=None, scale=y_scale)

        # Color setup: categorical -> fixed palette, emitter -> settings sex
        # colors, density / acoustic feature -> project colormap (quantitative).
        PALETTE = (
            "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B",
            "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#FF7F0E",
        )
        color_field = None
        cat_domain, cat_range = None, None
        if color_kind == "density":
            xa = chart_pd[x_col].to_numpy()
            ya = chart_pd[y_col].to_numpy()
            n_bins = 60
            counts, x_edges, y_edges = np.histogram2d(xa, ya, bins=n_bins)
            ix = np.clip(np.digitize(xa, x_edges) - 1, 0, n_bins - 1)
            iy = np.clip(np.digitize(ya, y_edges) - 1, 0, n_bins - 1)
            chart_pd["density"] = counts[ix, iy]
            color_field = "density"
        elif color_kind == "continuous":
            color_field = color_col
            # Rescale to the units shown in the dropdown label: frequencies are
            # stored in Hz -> kHz; duration is stored in seconds -> ms. Amplitude
            # (a.u.) and spectral entropy are shown as-is.
            if "freq" in color_col:
                chart_pd[color_col] = chart_pd[color_col] / 1000.0
            elif color_col == "duration":
                chart_pd[color_col] = chart_pd[color_col] * 1000.0
        elif color_kind == "emitter":
            color_field = color_col
            present = set(chart_pd[color_col].tolist())
            cat_domain = [_s for _s in ("male", "female", "unassigned") if _s in present]
            cat_range = [sex_colors[_s] for _s in cat_domain]
        elif color_kind == "categorical":
            color_field = color_col
            cat_domain = sorted(set(chart_pd[color_col].tolist()))
            cat_range = [PALETTE[i % len(PALETTE)] for i in range(len(cat_domain))]

        # Built-in Altair legend on the LEFT (vertical), TITLE OMITTED -- the
        # "Color by" dropdown already names the field, dropping the title also
        # sidesteps Vega-Lite's inability to rotate a legend title, and a left
        # legend keeps the scatter top-aligned with the spectrogram grid.
        mark_kwargs = dict(size=8, opacity=0.5)
        if color_field is None:
            mark_kwargs["color"] = "#9E9E9E"
        scatter = alt.Chart(chart_pd).mark_circle(**mark_kwargs).encode(x=x_enc, y=y_enc)
        if color_field is not None:
            if color_kind in ("density", "continuous"):
                color_enc = alt.Color(
                    f"{color_field}:Q",
                    scale=alt.Scale(scheme=global_cmap),
                    legend=alt.Legend(
                        title=None, orient="left", direction="vertical",
                        gradientLength=CHART_HEIGHT_PX, gradientThickness=30,
                        labelFontSize=14, labelColor="#000000",
                    ),
                )
            else:
                # Hide the legend when there are too many categories (e.g. many
                # sessions) -- a list of hundreds of ids is unreadable and the
                # 12-color palette cycles anyway. Points stay colored.
                _cat_legend = (
                    None if len(cat_domain) > 24 else alt.Legend(
                        title=None, orient="left", direction="vertical",
                        symbolSize=450, labelFontSize=15, rowPadding=10,
                        labelColor="#000000",
                    )
                )
                color_enc = alt.Color(
                    f"{color_field}:N",
                    scale=alt.Scale(domain=cat_domain, range=cat_range),
                    legend=_cat_legend,
                )
            scatter = scatter.encode(color=color_enc)

        # Black-outlined brush rectangle (the default near-white stroke is
        # barely visible over the scatter).
        brush = alt.selection_interval(
            name="brush",
            mark=alt.BrushConfig(
                stroke="#000000", strokeWidth=1.5,
                fill="#000000", fillOpacity=0.05,
            ),
        )
        scatter = scatter.add_params(brush)

        # Boundary overlay: KNN-predicted category grid (density-masked) ->
        # contour lines at half-integer class transitions -> one line per seg.
        layers = [scatter]
        if boundary_col is not None and chart_pd.shape[0] >= 5:
            bx_pts = chart_pd[x_col].to_numpy()
            by_pts = chart_pd[y_col].to_numpy()
            labels = chart_pd[boundary_col].to_numpy()
            if map_prefix == "qlvm":
                x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
            else:
                x_lo, x_hi = float(np.min(bx_pts)), float(np.max(bx_pts))
                y_lo, y_hi = float(np.min(by_pts)), float(np.max(by_pts))
            # Adapt grid resolution to point count, and keep the density mask
            # LOOSE (low min-count, strong smoothing) so the predicted-label
            # field stays connected -- a tight mask NaNs out lean cells and
            # fragments the contours into broken arcs. More neighbours also
            # smooths the k-NN boundary.
            grid_res = int(np.clip(np.sqrt(chart_pd.shape[0]) * 1.5, 80, 240))
            grid_xx, grid_yy, grid_labels = knn_boundary_grid(
                bx_pts, by_pts, labels, x_lo, x_hi, y_lo, y_hi,
                n_neighbors=25, grid_resolution=grid_res,
                density_smoothing_sigma=3.5, density_min_count=0.04,
            )
            if not np.all(np.isnan(grid_labels)):
                # Outline EACH category's region as the 0.5 contour of its own
                # binary mask, rather than contouring the integer label grid at
                # half-integer levels. The latter bunches several lines together
                # wherever non-consecutive category numbers sit adjacent (every
                # in-between level crosses there), making the boundary look
                # thicker in spots. Per-category 0.5 contours put each shared
                # border at exactly one position, so the line is uniform width.
                present_labels = [v for v in np.unique(grid_labels) if not np.isnan(v)]
                tmp_fig, tmp_ax = plt.subplots()
                seg_rows = []
                seg_id = 0
                for _lab in present_labels:
                    _mask = np.where(
                        np.isnan(grid_labels), 0.0, (grid_labels == _lab).astype(float)
                    )
                    _cs = tmp_ax.contour(grid_xx, grid_yy, _mask, levels=[0.5])
                    for level_segs in _cs.allsegs:
                        for seg in level_segs:
                            for order, (sx, sy) in enumerate(seg):
                                seg_rows.append(
                                    {"bx": float(sx), "by": float(sy),
                                     "seg": seg_id, "order": order}
                                )
                            seg_id += 1
                plt.close(tmp_fig)
                if seg_rows:
                        boundary_df = pd.DataFrame(seg_rows)

                        # Haloed contour: a thick BLACK outline under a bright
                        # core. Over the warm colormap (inferno; used when
                        # coloring by density / a continuous feature) a CYAN core
                        # pops; over the categorical palette a white core reads
                        # cleanly.
                        def _bline(_color, _width):
                            return (
                                alt.Chart(boundary_df)
                                .mark_line(color=_color, strokeWidth=_width, opacity=1.0)
                                .encode(
                                    x=alt.X("bx:Q", scale=x_scale, axis=None),
                                    y=alt.Y("by:Q", scale=y_scale, axis=None),
                                    detail="seg:N",
                                    order="order:Q",
                                )
                            )

                        _core = "#00E5E5" if color_kind in ("density", "continuous") else "#FFFFFF"
                        layers.append(_bline("#000000", 5.0))   # black outline
                        layers.append(_bline(_core, 2.4))       # bright core

        chart = (alt.layer(*layers) if len(layers) > 1 else scatter).properties(
            width=CHART_DATA_WIDTH_PX,
            height=CHART_HEIGHT_PX,
            background="#FFFFFF",
            padding=0,
        ).configure_legend(
            labelFontSize=15, symbolSize=450, rowPadding=10,
            gradientThickness=30, gradientLength=CHART_HEIGHT_PX,
            labelColor="#000000",
        )
        # Return the chart AND its underlying data + coord column names: marimo
        # can't read brushed rows from a LAYERED chart via .value, so _explorer
        # recovers them with chart_widget.apply_selection(chart_data) and uses
        # coord_x/coord_y for the spiral sampling.
        return mo.ui.altair_chart(chart), chart_pd, x_col, y_col

    chart_widget, chart_data, coord_x, coord_y = _()
    return chart_data, chart_widget, coord_x, coord_y


@app.cell
def _explorer(
    BytesIO,
    CHART_HEIGHT_PX,
    apply_mask_checkbox,
    available_lists,
    base64,
    chart_data,
    chart_widget,
    consolidated_h5_path,
    coord_x,
    coord_y,
    controls,
    global_cmap,
    h5py,
    lists_multiselect,
    mo,
    n_samples_slider,
    np,
    plt,
):
    def _():
        # The control panel is rendered at the TOP of THIS cell's output, with
        # the plot directly beneath it -- same cell, so nothing (no code block)
        # separates the widgets from the embedding. When there's no chart yet
        # (no list picked), the plot area is just a "pick a list" hint.
        if chart_widget is None:
            if not available_lists:
                _hint = mo.md(
                    "**No session lists found.** Set "
                    "`usv_embedding.input_files_directory` in "
                    "`_parameter_settings/visualizations_settings.json`."
                )
            elif not lists_multiselect.value:
                _hint = mo.md(
                    "_Pick one or more session lists above to load the embedding._"
                )
            else:
                _hint = mo.md("")
            return mo.vstack([controls, _hint], align="start", gap=0.5)

        apply_mask = bool(apply_mask_checkbox.value)

        n_pick_target = int(n_samples_slider.value)
        # Recover the brushed rows via apply_selection(chart_data): chart_widget
        # .value can't return data for a LAYERED chart (boundaries on), but
        # apply_selection works for both layered and single-layer charts.
        # (chart_data carries session_id / row_index for the h5 lookup.)
        try:
            selected_pd = chart_widget.apply_selection(chart_data)
        except (ValueError, TypeError):
            selected_pd = None

        # With NO active brush, apply_selection returns the WHOLE dataset (it has
        # no empty state). Treat "selection == all rows" as no brush, so we show
        # the hint until the user actually drags a rectangle (the only false
        # positive is a brush covering literally every point).
        no_brush = selected_pd is not None and len(selected_pd) == len(chart_data)
        if selected_pd is None or len(selected_pd) == 0 or no_brush:
            spectrograms_out = mo.md(
                "_Drag a rectangle on the scatter plot to sample "
                "spectrograms from inside that region._"
            )
        else:
            n_pick = min(n_pick_target, len(selected_pd))
            # Spiral sampling (ported from plot_umap_with_category_thumbnails):
            # build an Archimedean spiral from the brushed region's centroid out
            # to its farthest point, subsample n_pick positions along it, and
            # snap each to the nearest not-yet-used point. The picks come out
            # ordered centre -> edge, so the grid reads from the middle outward
            # instead of being random.
            _pts = selected_pd[[coord_x, coord_y]].to_numpy(dtype=float)
            _cx, _cy = float(_pts[:, 0].mean()), float(_pts[:, 1].mean())
            _r_max = float(np.hypot(_pts[:, 0] - _cx, _pts[:, 1] - _cy).max())
            if _r_max <= 0.0:
                _pick_idx = list(range(n_pick))
            else:
                _t = np.linspace(0.0, 1.0, 4000)
                _theta = 2.0 * np.pi * 4 * _t           # 4 revolutions
                _r = _r_max * _t
                _sx = _cx + _r * np.cos(_theta)
                _sy = _cy + _r * np.sin(_theta)
                _sel = np.linspace(0, _sx.size - 1, n_pick).astype(int)
                _used, _pick_idx = set(), []
                for _tx, _ty in zip(_sx[_sel], _sy[_sel]):
                    _d2 = (_pts[:, 0] - _tx) ** 2 + (_pts[:, 1] - _ty) ** 2
                    for _kk in np.argsort(_d2):
                        _k = int(_kk)
                        if _k not in _used:
                            _used.add(_k)
                            _pick_idx.append(_k)
                            break
                    if len(_pick_idx) >= n_pick:
                        break
            picks = selected_pd.iloc[_pick_idx]
            tiles = []
            # The fixed spectrogram storage window (time-bin count, e.g. 128). Every
            # spec is padded to this common width so a call's horizontal extent ==
            # its true duration RELATIVE TO THE WINDOW (a stable, absolute scale --
            # not the per-brush max, which would rescale every selection).
            time_window = None
            # Per-session mask-index cache so we read the small
            # ``spectrogram_index`` array only once per session even when
            # several samples come from the same session.
            mask_index_cache: dict = {}
            h5_open_error = None
            try:
                with h5py.File(consolidated_h5_path, "r") as h5:
                    for _, row in picks.iterrows():
                        sess = str(row["session_id"])
                        idx = int(row["row_index"])
                        spec_group_key = f"spectrogram/{sess}"
                        if spec_group_key not in h5:
                            continue
                        grp = h5[spec_group_key]
                        spec = grp["spectrograms"][idx, :, :].astype(np.float32)
                        time_window = spec.shape[1]
                        dur = int(grp["durations"][idx])
                        dur = max(1, min(dur, spec.shape[1]))

                        if apply_mask:
                            mask_group_key = f"mask/{sess}"
                            if mask_group_key in h5:
                                mask_grp = h5[mask_group_key]
                                if sess not in mask_index_cache:
                                    mask_index_cache[sess] = mask_grp["spectrogram_index"][:]
                                spec_indices = mask_index_cache[sess]
                                matching = np.where(spec_indices == idx)[0]
                                if matching.size > 0:
                                    masks_for_spec = mask_grp["segmentations"][
                                        matching, :, :dur
                                    ]
                                    combined_mask = np.any(masks_for_spec, axis=0)
                                    spec_to_show = spec[:, :dur] * combined_mask.astype(np.float32)
                                else:
                                    spec_to_show = spec[:, :dur]
                            else:
                                spec_to_show = spec[:, :dur]
                        else:
                            spec_to_show = spec[:, :dur]
                        tiles.append((sess, idx, spec_to_show))
            except (OSError, FileNotFoundError) as exc:
                tiles = []
                h5_open_error = mo.md(
                    f"**Could not open the consolidated store** "
                    f"`{consolidated_h5_path}`:\n\n```\n{exc}\n```"
                )

            if h5_open_error is not None:
                spectrograms_out = h5_open_error
            elif not tiles:
                spectrograms_out = mo.md(
                    "_None of the sampled rows had a matching spectrogram in the h5._"
                )
            else:
                # Per-USV spectrograms are min-max normalized to [0, 1] by
                # generate_spectrograms, so [0, 1] is the correct display window.
                # Pad each tile (left-aligned) to the FIXED storage window so its
                # horizontal extent == its true duration relative to that window: a
                # short call stays short, a window-length call fills the tile. Lay
                # the tiles out in a GRID (N_COLS columns) of LANDSCAPE tiles. The
                # block is forced to the scatter height (legend on the left, so the
                # scatter starts at the widget top and the grid aligns to it).
                N_COLS = 5
                w_target = time_window
                n_grid_rows = int(np.ceil(len(tiles) / N_COLS))
                tile_w_in = 1.15         # per-tile width (narrower -> less stretch)
                tile_aspect = 1.7        # width : height (landscape)
                tile_h_in = tile_w_in / tile_aspect
                fig, axes = plt.subplots(
                    n_grid_rows, N_COLS,
                    figsize=(N_COLS * tile_w_in, n_grid_rows * tile_h_in),
                    dpi=120, facecolor="#FFFFFF",
                )
                flat_axes = np.atleast_1d(axes).ravel()
                for slot, ax in enumerate(flat_axes):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_facecolor("#FFFFFF")
                    for _sp in ax.spines.values():
                        _sp.set_visible(False)
                    if slot >= len(tiles):
                        continue  # empty trailing slot
                    _, _, tile = tiles[slot]
                    n_freq, dur = tile.shape
                    # Center the call in the fixed window: width == dur / window
                    # (duration preserved), with equal zero-padding on both sides.
                    dur = min(dur, w_target)
                    pad_left = (w_target - dur) // 2
                    tile_padded = np.zeros((n_freq, w_target), dtype=tile.dtype)
                    tile_padded[:, pad_left:pad_left + dur] = tile[:, :dur]
                    ax.imshow(
                        tile_padded,
                        origin="lower",
                        aspect="auto",
                        cmap=global_cmap,
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                    )
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.08, hspace=0.12)

                buf = BytesIO()
                fig.savefig(
                    buf, format="png", dpi=120, facecolor="#FFFFFF",
                )
                plt.close(fig)
                # Embed the grid INLINE as a base64 data-URI (not mo.image,
                # which registers a per-render virtual file the browser fetches
                # over HTTP -- those 404 on re-render as marimo GCs the previous
                # one). Inline = self-contained, no fetch, no 404. Forced to the
                # scatter's data height; width scales proportionally.
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                spectrograms_out = mo.Html(
                    f'<img src="data:image/png;base64,{img_b64}" '
                    f'style="display:block;height:{CHART_HEIGHT_PX}px;width:auto;'
                    'margin:0;padding:0;" />'
                )
        plot_row = mo.hstack(
            [chart_widget, spectrograms_out],
            justify="start", gap=1, align="start",
        )
        # Controls on top, plot directly beneath -- one cell, nothing between.
        return mo.vstack([controls, plot_row], align="start", gap=0.5)

    _()
    return


if __name__ == "__main__":
    app.run()
