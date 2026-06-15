"""
Marimo notebook for interactively exploring USV embeddings (VAE / QLVM).

Usage
-----
Install the ``dev`` dependency group (it carries ``marimo`` and
``altair``, both used by this notebook):

    pip install -e ".[dev]"

From the ``src/usv_playpen/analyses_notebooks`` directory:

    marimo edit usv_embedding_explorer.py

Architecture
------------
- An altair scatter of the chosen UMAP map (VAE or QLVM), optionally
  color-coded by category / supercategory, carrying an
  ``alt.selection_interval`` brush. The scatter spec only embeds the
  ``x``, ``y`` and optional ``color`` columns -- NOT per-USV
  spectrogram thumbnails -- so the chart-spec payload scales as
  ``n_points * ~50 bytes``. That means hundreds of thousands of points
  can fit comfortably inside marimo's ``output_max_bytes``.
- Beneath the scatter, a matplotlib row of sample spectrograms is
  rebuilt every time the brush changes: the explorer cell reads the
  brushed rows from the chart_widget, samples ``n_samples`` of them,
  pulls the matching 128x128 spectrograms from the consolidated h5
  for ONLY those rows, encodes them at full quality, and renders them
  as a base64 PNG. Because we encode only ~8 thumbnails per brush we
  can afford high-quality, sharp output.
- Both panels are pinned to the same total CSS pixel width via a
  ``mo.Html`` wrapper with a hard ``max-width``, defeating the VS Code
  marimo plugin's tendency to stretch embedded images to fill the cell
  column. The matplotlib strip is rendered with a left padding equal
  to altair's y-axis-label margin so the spectrogram area sits flush
  under the scatter's data area.

All paths are run through ``os_utils.configure_path`` so ``/mnt/...``
strings from cross-OS session lists resolve correctly on macOS / Linux.
"""

import json
import pathlib

import marimo

# Load `visualizations_settings.json` at module import so the cmap
# default below is the project-wide `figures.cmap` (instead of a
# literal hard-coded in the spectrogram-strip cell). Resolves to
# `'inferno'` when the figures block is absent.
_VIZ_SETTINGS_PATH = (
    pathlib.Path(__file__).parent.parent
    / "_parameter_settings" / "visualizations_settings.json"
)
try:
    with _VIZ_SETTINGS_PATH.open() as _vf:
        _GLOBAL_CMAP = json.load(_vf).get("figures", {}).get("cmap", "inferno")
except FileNotFoundError:
    _GLOBAL_CMAP = "inferno"

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _imports():
    import base64
    from io import BytesIO
    from pathlib import Path

    import altair as alt
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    alt.data_transformers.disable_max_rows()

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.make_usv_spectrograms import (
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
        mo,
        np,
        plt,
    )


@app.cell
def _intro(mo):
    mo.md(r"""
    # USV embedding explorer

    Pick a map (**VAE** or **QLVM**), optionally color-code by
    category or supercategory, brush a rectangle on the scatter
    plot. A row of sharp spectrogram thumbnails sampled from the
    brushed region appears directly beneath the scatter --
    encoded on the fly only for those samples, so the chart-spec
    payload stays small and you can drop hundreds of thousands of
    points into the scatter.
    """)
    return


@app.cell
def _paths(Path, configure_path):
    sessions_txt_path = configure_path(
        "/mnt/falkner/Bartul/modeling/input_files/"
        "behavioral_courtship_intact_partners_sessions_list.txt"
    )
    consolidated_h5_path = configure_path(
        "/Volumes/falkner/Bartul/spectrograms/"
        "spectrograms_sam2masks_553sessions_602724vocalizations_20260514_153009.h5"
    )
    cache_path = str(
        Path.home() / ".usv_playpen_cache" / "pooled_embeddings.parquet"
    )
    rebuild_cache = False
    return cache_path, consolidated_h5_path, rebuild_cache, sessions_txt_path


@app.cell
def _load_pooled_df(
    build_pooled_embeddings_df,
    cache_path,
    mo,
    rebuild_cache,
    sessions_txt_path,
):
    pooled_df = build_pooled_embeddings_df(
        sessions_txt_path=sessions_txt_path,
        cache_path=cache_path,
        rebuild_cache=rebuild_cache,
        noise_col_id="vae_supercategory",
        noise_categories=(0,),
    )
    summary = mo.md(
        f"**{pooled_df.height:,}** non-noise vocalizations pooled across "
        f"**{pooled_df.select('session_id').unique().height}** sessions."
    )
    summary
    return (pooled_df,)


@app.cell
def _widgets(mo):
    map_dropdown = mo.ui.dropdown(
        options=["VAE", "QLVM"],
        value="VAE",
        label="Embedding map",
    )
    color_dropdown = mo.ui.dropdown(
        options=["none", "category", "supercategory"],
        value="supercategory",
        label="Color by",
    )
    n_samples_slider = mo.ui.slider(
        start=1, stop=16, step=1, value=8,
        label="Sampled spectrograms",
    )
    # Scatter is now thumbnail-free: each point is ~50 bytes in the
    # chart spec, so hundreds of thousands of points fit inside the
    # 30 MB cap. The slider tops out at 500 K to leave headroom for
    # altair rendering performance in the browser; above ~100 K
    # interaction starts to feel sluggish in some browsers.
    max_points_slider = mo.ui.slider(
        start=5_000, stop=500_000, step=5_000, value=50_000,
        label="Max points in scatter",
    )
    apply_mask_checkbox = mo.ui.checkbox(
        value=True,
        label="Apply SAM2 mask (zero out non-USV pixels)",
    )
    controls = mo.hstack(
        [
            map_dropdown,
            color_dropdown,
            n_samples_slider,
            max_points_slider,
            apply_mask_checkbox,
        ],
        justify="start",
    )
    return (
        apply_mask_checkbox,
        color_dropdown,
        controls,
        map_dropdown,
        max_points_slider,
        n_samples_slider,
    )


@app.cell
def _prepare_display(
    color_dropdown,
    map_dropdown,
    max_points_slider,
    mo,
    pooled_df,
):
    map_prefix = "vae" if map_dropdown.value == "VAE" else "qlvm"
    x_col = f"{map_prefix}_umap1"
    y_col = f"{map_prefix}_umap2"
    color_col = (
        None
        if color_dropdown.value == "none"
        else f"{map_prefix}_{color_dropdown.value}"
    )

    keep = ["session_id", "row_index", x_col, y_col]
    if color_col is not None:
        keep.append(color_col)

    display_df = pooled_df.select(keep).drop_nulls(subset=[x_col, y_col])
    max_points = int(max_points_slider.value)
    if display_df.height > max_points:
        display_df = display_df.sample(n=max_points, seed=42)

    debug = mo.md(
        f"`_prepare_display` -> capped at **{max_points:,}** points; "
        f"display_df has **{display_df.height:,}** rows "
        f"(map: {map_prefix.upper()})."
    )
    debug
    return color_col, display_df, map_prefix, x_col, y_col


@app.cell
def _layout_constants():
    CHART_DATA_WIDTH_PX = 800
    CHART_HEIGHT_PX = 640
    CHART_LEFT_AXIS_PX = 10
    # White space after the last spectrogram on the strip. Bigger -> more
    # right-side breathing room.
    CHART_RIGHT_PAD_PX = 45
    CHART_TOTAL_WIDTH_PX = CHART_DATA_WIDTH_PX + CHART_LEFT_AXIS_PX
    return (
        CHART_DATA_WIDTH_PX,
        CHART_HEIGHT_PX,
        CHART_LEFT_AXIS_PX,
        CHART_RIGHT_PAD_PX,
        CHART_TOTAL_WIDTH_PX,
    )


@app.cell
def _scatter_chart(
    CHART_DATA_WIDTH_PX,
    CHART_HEIGHT_PX,
    alt,
    color_col,
    display_df,
    map_prefix,
    mo,
    x_col,
    y_col,
):
    chart_pd = display_df.to_pandas()

    # Pretty axis titles -- maps the raw column name to its display
    # form ("VAE UMAP 1", "QLVM UMAP 2", ...).
    label_map = {
        "vae_umap1": "VAE UMAP 1",
        "vae_umap2": "VAE UMAP 2",
        "qlvm_umap1": "QLVM UMAP 1",
        "qlvm_umap2": "QLVM UMAP 2",
    }
    x_title = label_map.get(x_col, x_col)
    y_title = label_map.get(y_col, y_col)

    # Hide tick marks and tick labels on both axes; keep only the
    # domain (spine) and axis title. Title sits close to the spine via
    # ``titlePadding=2``; tweak the font size below to make it bigger.
    axis_kwargs = dict(
        grid=False, domain=True,
        domainColor="#000000",
        ticks=False, labels=False,
        titleColor="#000000",
        titleFontSize=14,
        titlePadding=2,
    )
    x_axis = alt.Axis(**axis_kwargs)
    y_axis = alt.Axis(**axis_kwargs)
    if map_prefix == "qlvm":
        x_enc = alt.X(
            x_col, type="quantitative", title=x_title,
            axis=x_axis, scale=alt.Scale(domain=[0.0, 1.0], nice=False),
        )
        y_enc = alt.Y(
            y_col, type="quantitative", title=y_title,
            axis=y_axis, scale=alt.Scale(domain=[0.0, 1.0], nice=False),
        )
    else:
        x_enc = alt.X(
            x_col, type="quantitative", title=x_title,
            axis=x_axis, scale=alt.Scale(domain=[5, 18], nice=False),
        )
        y_enc = alt.Y(y_col, type="quantitative", title=y_title, axis=y_axis)

    # When the user picks "Color by = none", paint the dots a soft grey
    # instead of altair's default blue.
    NONE_COLOR = "#9E9E9E"
    mark_kwargs = dict(size=8, opacity=0.5)
    if color_col is None:
        mark_kwargs["color"] = NONE_COLOR
    scatter_base = (
        alt.Chart(chart_pd)
        .mark_circle(**mark_kwargs)
        .encode(x=x_enc, y=y_enc)
    )
    if color_col is not None:
        # Legend rendered above the chart so it doesn't add horizontal
        # width on the right -- keeping the total altair width equal to
        # ``CHART_TOTAL_WIDTH_PX``.
        scatter_base = scatter_base.encode(
            color=alt.Color(
                f"{color_col}:N",
                legend=alt.Legend(
                    title=color_col,
                    labelColor="#000000",
                    titleColor="#000000",
                    orient="top",
                ),
            )
        )

    brush = alt.selection_interval(name="brush")
    chart = (
        scatter_base.add_params(brush)
        .properties(
            width=CHART_DATA_WIDTH_PX,
            height=CHART_HEIGHT_PX,
            background="#FFFFFF",
            padding=0,
        )
    )
    chart_widget = mo.ui.altair_chart(chart)
    return (chart_widget,)


@app.cell
def _explorer(
    BytesIO,
    CHART_LEFT_AXIS_PX,
    CHART_RIGHT_PAD_PX,
    CHART_TOTAL_WIDTH_PX,
    apply_mask_checkbox,
    base64,
    chart_widget,
    consolidated_h5_path,
    controls,
    h5py,
    mo,
    n_samples_slider,
    np,
    plt,
):
    strip_dpi = 100
    strip_total_w_in = CHART_TOTAL_WIDTH_PX / strip_dpi
    strip_data_w_in = (
        CHART_TOTAL_WIDTH_PX - CHART_LEFT_AXIS_PX - CHART_RIGHT_PAD_PX
    ) / strip_dpi
    jpeg_quality = 95  # high quality for the 8 sharp thumbnails

    apply_mask = bool(apply_mask_checkbox.value)

    n_pick_target = int(n_samples_slider.value)
    selected = chart_widget.value
    selected_pd = None
    if selected is not None:
        if hasattr(selected, "to_pandas"):
            selected_pd = selected.to_pandas()
        else:
            selected_pd = selected

    if selected_pd is None or len(selected_pd) == 0:
        spectrograms_out = mo.md(
            "_Drag a rectangle on the scatter plot to sample "
            "spectrograms from inside that region._"
        )
    else:
        n_pick = min(n_pick_target, len(selected_pd))
        picks = selected_pd.sample(n=n_pick)
        tiles = []
        # Per-session mask-index cache so we read the small
        # ``spectrogram_index`` array only once per session even when
        # several samples come from the same session.
        mask_index_cache: dict = {}
        with h5py.File(consolidated_h5_path, "r") as h5:
            for _, row in picks.iterrows():
                sess = str(row["session_id"])
                idx = int(row["row_index"])
                spec_group_key = f"spectrogram/{sess}"
                if spec_group_key not in h5:
                    continue
                grp = h5[spec_group_key]
                spec = grp["spectrograms"][idx, :, :].astype(np.float32)
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

        if not tiles:
            spectrograms_out = mo.md(
                "_None of the sampled rows had a matching spectrogram in the h5._"
            )
        else:
            ncols = n_pick_target
            usable_frac = 1.0 - (CHART_LEFT_AXIS_PX + CHART_RIGHT_PAD_PX) / CHART_TOTAL_WIDTH_PX
            tile_w_frac = usable_frac / ncols
            tile_w_in = strip_data_w_in / ncols
            wspace_frac = 0.05  # fraction of a tile's width
            gap_w_frac = tile_w_frac * wspace_frac

            fig = plt.figure(
                figsize=(strip_total_w_in, tile_w_in),
                dpi=strip_dpi,
                facecolor="#FFFFFF",
            )
            left_offset_frac = CHART_LEFT_AXIS_PX / CHART_TOTAL_WIDTH_PX
            for i in range(ncols):
                ax_left = left_offset_frac + i * tile_w_frac + gap_w_frac / 2
                ax_width = tile_w_frac - gap_w_frac
                ax = fig.add_axes([ax_left, 0.0, ax_width, 1.0])
                if i < len(tiles):
                    _, _, tile = tiles[i]
                    ax.imshow(
                        tile,
                        origin="lower",
                        aspect="auto",
                        cmap=_GLOBAL_CMAP,
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#FFFFFF")
                for sp in ax.spines.values():
                    sp.set_visible(False)

            buf = BytesIO()
            fig.savefig(
                buf,
                format="jpeg",
                dpi=strip_dpi,
                facecolor=fig.get_facecolor(),
                pil_kwargs={"quality": jpeg_quality},
            )
            plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            # Hard pixel width via inline CSS with ``!important`` to
            # defeat the VS Code marimo plugin's default ``max-width: 100%``
            # on embedded images, which would otherwise stretch the strip
            # to fill the cell column and overshoot the altair scatter.
            spectrograms_out = mo.Html(
                f"<div style=\"background-color:#FFFFFF;"
                f"width:{CHART_TOTAL_WIDTH_PX}px !important;"
                f"max-width:{CHART_TOTAL_WIDTH_PX}px !important;"
                "margin:0;padding:0;line-height:0;\">"
                f"<img src=\"data:image/jpeg;base64,{img_b64}\" "
                f"style=\"display:block;"
                f"width:{CHART_TOTAL_WIDTH_PX}px !important;"
                f"max-width:{CHART_TOTAL_WIDTH_PX}px !important;"
                "height:auto;margin:0;padding:0;\" />"
                "</div>"
            )
    mo.vstack([controls, chart_widget, spectrograms_out], gap=0, align="start")
    return


if __name__ == "__main__":
    app.run()
