"""
@author: bartulem
Extensive unit + integration tests for
``usv_playpen.visualizations.make_usv_spectrograms``.

The module under test is a 3k-line visualization file that, prior to
this suite, had zero coverage (ledger finding A3). Every public entry
point depends on heavy on-disk artifacts (multi-channel int16 audio
memmaps, consolidated SAM2 + spectrogram HDF5 stores, per-session USV
summary CSVs, 3D tracking HDF5s). To exercise the real code paths
without any of that real data, the tests below synthesize tiny stand-in
artifacts under ``tmp_path``:

  * ``_write_audio_memmap`` writes a correctly-named int16 memmap so the
    ``*_<sr>_<n>_<ch>_int16.mmap`` regex parse (finding A1) is exercised.
  * ``_write_consolidated_h5`` builds a miniature spectrogram/mask store.
  * ``_write_usv_summary_csv`` / ``_write_tracking_h5`` stand in for the
    per-session CSV / tracking HDF5.

``matplotlib.use("Agg")`` is set BEFORE the module is imported so the
plotting paths never need a display. Per-test ``filterwarnings`` markers
are narrow (message substring + category) and only cover legitimate
numpy / matplotlib / librosa noise; the project-wide
``filterwarnings = ["error"]`` otherwise turns any stray warning into a
failure.
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import re
import types

import altair as alt
import h5py
import marimo as mo
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.contour
import numpy as np
import pandas as pd
import polars as pls
import pytest

from usv_playpen import os_utils
from usv_playpen.notebooks import usv_embedding_explorer
from usv_playpen.os_utils import SQUEAK_CLASS_SELECTIONS, call_class_mask

from usv_playpen.visualizations.make_usv_spectrograms import (
    BANDWIDTH_BIMODAL_SPLIT_KHZ,
    EMBEDDING_COORD_COLS,
    SESSION_TYPE_FEMALE_COLOR,
    SESSION_TYPE_MALE_COLOR,
    USV_TIMELINE_FEMALE_COLOR,
    USV_TIMELINE_MALE_COLOR,
    USV_TIMELINE_UNASSIGNED_COLOR,
    USVSpectrogramPlotter,
    _count_usvs_per_session,
    _medoid_xy,
    _pick_category_samples,
    _pick_spiral_with_grid,
    _resolve_session_emitter_sexes,
    build_pooled_embeddings_df,
    plot_embedding_with_category_thumbnails,
    plot_session_type_usv_counts,
    plot_session_usv_timeline,
    plot_usv_property_histograms,
    render_embedding_thumbnails_for_cohort,
)


def test_session_type_and_timeline_colors_derive_from_settings():
    """The session-type and USV-timeline sex / unassigned colors are read from
    the `male_colors` / `female_colors` / `unassigned_colors` palette blocks of
    `visualizations_settings.json` rather than hard-coded, so the module-level
    constants must equal the shipped palette entries (guards against the plot
    colors drifting out of sync with the project palette)."""

    settings_path = (
        pathlib.Path(__import__("usv_playpen").__file__).parent
        / "_parameter_settings" / "visualizations_settings.json"
    )
    with settings_path.open() as settings_file:
        viz_settings = json.load(settings_file)

    assert SESSION_TYPE_MALE_COLOR == viz_settings["male_colors"][0]
    assert SESSION_TYPE_FEMALE_COLOR == viz_settings["female_colors"][0]
    assert USV_TIMELINE_MALE_COLOR == viz_settings["male_colors"][0]
    assert USV_TIMELINE_FEMALE_COLOR == viz_settings["female_colors"][0]
    assert USV_TIMELINE_UNASSIGNED_COLOR == viz_settings["unassigned_colors"][0]


# ---- synthetic-artifact builders ------------------------------------------


def _base_settings(
    *,
    mode: str = "single",
    save_dir: str = "",
    save_fig: bool = False,
    plot_raw_audio: bool = False,
    plot_cbar: bool = True,
    freq_limits: tuple[float, float] = (10.0, 40.0),
    time_window: tuple[float, float] = (0.0, 0.0),
    spectrograms_dir: str = "",
    apply_mask: bool = True,
    channel_of_interest: int = 0,
    auto_open_figure: bool = False,
) -> dict:
    """
    Description
    -----------
    Build a fresh ``visualizations_parameter_dict`` carrying the
    ``figures`` block (for cmap resolution) and a ``make_usv_spectrograms``
    block populated with every key the plotter reads. Returned fresh on
    each call so a test mutating it cannot leak into another.

    Parameters
    ----------
    mode (str)
        Dispatch mode ('single' / 'all' / 'stitched').
    save_dir (str)
        Output directory; empty string routes saves to
        ``<root>/data_animation_examples``.
    save_fig (bool)
        Whether ``_save_figure`` writes to disk.
    plot_raw_audio (bool)
        Whether to stack the raw waveform above each spectrogram.
    plot_cbar (bool)
        Whether to draw the right-side colorbar.
    freq_limits (tuple of float)
        Lower / upper frequency limits in kHz.
    time_window (tuple of float)
        Analysis window in seconds ([start, end]; end 0 -> full file).
    spectrograms_dir (str)
        Base dir holding spectrograms_*.h5
        (stitched / sequence modes resolve their inputs from it).
    apply_mask (bool)
        Master SAM2 mask toggle (stitched mode).
    channel_of_interest (int)
        Default channel for single-channel mode.

    Returns
    -------
    settings (dict)
        A two-key dict (``figures`` + ``make_usv_spectrograms``).
    """

    return {
        "figures": {"sequential_cmap": "inferno", "timestamp_in_name": False},
        "shared_resources": {
            "spectrograms_dir": spectrograms_dir,
        },
        "make_usv_spectrograms": {
            "save_dir": save_dir,
            "save_fig": save_fig,
            "fig_format": "png",
            "fig_dpi": 50,
            "fig_size": [4, 2],
            "transparent_fig_bg": False,
            "mode": mode,
            "channel_of_interest": channel_of_interest,
            "plot_raw_audio": plot_raw_audio,
            "time_window": list(time_window),
            "freq_limits": list(freq_limits),
            "usv_amplitude_color": "#808080",
            "nfft": 256,
            "plot_cbar": plot_cbar,
            "cbar_limits": [-70, 0],
            "apply_mask": apply_mask,
            "auto_open_figure": auto_open_figure,
        },
    }


def _write_audio_memmap(
    root: pathlib.Path,
    sampling_rate: int = 250_000,
    sample_num: int = 2_000,
    channel_num: int = 3,
) -> pathlib.Path:
    """
    Description
    -----------
    Write a synthetic concatenated int16 audio memmap at the canonical
    'ultrasonic' band location ``<root>/audio/hpss_filtered/`` with the canonical
    name ``<id>_concatenated_audio_hpss_filtered_<sr>_<n_samples>_<n_ch>_int16.mmap``
    (so ``os_utils.find_audio_mmap`` locates it and the parse in
    ``_load_audio_memmap`` resolves it) and fill it with a low-amplitude
    multi-channel sine so spectrograms are non-degenerate.

    Parameters
    ----------
    root (pathlib.Path)
        Session root directory; the memmap is written under
        ``<root>/audio/hpss_filtered`` (created if absent).
    sampling_rate, sample_num, channel_num (int)
        Encoded in the filename and used to shape the (sample, channel)
        int16 array.

    Returns
    -------
    path (pathlib.Path)
        Path to the written memmap file.
    """

    mmap_dir = root / "audio" / "hpss_filtered"
    mmap_dir.mkdir(parents=True, exist_ok=True)
    name = f"230101120000_concatenated_audio_hpss_filtered_{sampling_rate}_{sample_num}_{channel_num}_int16.mmap"
    path = mmap_dir / name
    t = np.arange(sample_num, dtype=np.float64) / sampling_rate
    data = np.empty((sample_num, channel_num), dtype=np.int16)
    for ch in range(channel_num):
        freq = 50_000.0 + 5_000.0 * ch
        data[:, ch] = (2_000.0 * np.sin(2.0 * np.pi * freq * t)).astype(np.int16)
    mm = np.memmap(path, dtype=np.int16, mode="w+", shape=(sample_num, channel_num))
    mm[:] = data
    mm.flush()
    del mm
    return path


def _write_consolidated_h5(
    path: pathlib.Path,
    session_key: str,
    *,
    n_usvs: int = 4,
    n_freq: int = 16,
    n_time: int = 32,
    with_mask: bool = True,
) -> pathlib.Path:
    """
    Description
    -----------
    Build a miniature consolidated spectrogram store mimicking the real
    SAM2 + spectrogram HDF5: a shared linear ``frequency_bins`` axis plus
    a ``spectrogram/<session_key>`` group with ``spectrograms``
    ([0, 1]-normalized) and ``durations`` datasets, and optionally a
    ``mask/<session_key>`` group with boolean ``segmentations`` and an
    integer ``spectrogram_index``.

    Parameters
    ----------
    path (pathlib.Path)
        Output HDF5 path.
    session_key (str)
        Session group name (must equal the session root's basename for
        stitched mode, or the ``session_id`` in the pooled DataFrame for
        the UMAP thumbnail figure).
    n_usvs, n_freq, n_time (int)
        Store dimensions. ``frequency_bins`` length equals ``n_freq``.
    with_mask (bool)
        Whether to also write the ``mask/<session_key>`` group.

    Returns
    -------
    path (pathlib.Path)
        The written HDF5 path.
    """

    rng = np.random.default_rng(0)
    specs = rng.random((n_usvs, n_freq, n_time)).astype(np.float32)
    durations = np.full(n_usvs, n_time, dtype=np.int64)
    freq_bins = np.linspace(30_000.0, 120_000.0, n_freq).astype(np.float64)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency_bins", data=freq_bins)
        grp = h5.create_group(f"spectrogram/{session_key}")
        grp.create_dataset("spectrograms", data=specs)
        grp.create_dataset("durations", data=durations)
        if with_mask:
            mask_grp = h5.create_group(f"mask/{session_key}")
            segs = rng.random((n_usvs, n_freq, n_time)) > 0.5
            mask_grp.create_dataset("segmentations", data=segs)
            mask_grp.create_dataset(
                "spectrogram_index", data=np.arange(n_usvs, dtype=np.int64)
            )
    return path


def _write_usv_summary_csv(
    audio_dir: pathlib.Path,
    rows: dict,
    name: str = "session_usv_summary.csv",
) -> pathlib.Path:
    """
    Description
    -----------
    Write a ``*_usv_summary.csv`` from a column->list mapping under the
    given ``audio_dir`` (created if absent).

    Parameters
    ----------
    audio_dir (pathlib.Path)
        Directory to write the CSV into.
    rows (dict)
        Column name -> list of values.
    name (str)
        File name (must end in ``_usv_summary.csv``).

    Returns
    -------
    path (pathlib.Path)
        The written CSV path.
    """

    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / name
    if "noise" not in rows:
        rows = {**rows, "noise": [False] * len(next(iter(rows.values())))}
    pls.DataFrame(rows).write_csv(path)
    return path


def _write_tracking_h5(
    video_dir: pathlib.Path,
    track_names: tuple[str, ...] = ("male_x", "female_y"),
    name: str = "session_points3d_translated_rotated_metric.h5",
    sexes: tuple[str, ...] = ("male", "female"),
) -> pathlib.Path:
    """
    Description
    -----------
    Write a stand-in 3D tracking HDF5 carrying only the ``track_names``
    dataset (the single field ``_resolve_session_emitter_sexes`` and the
    pooled-embeddings loader read from it), plus the session's
    ``<session>_metadata.yaml`` in the session root (``video_dir.parent``)
    whose ``Subjects`` record each track's sex -- the source every emitter
    -> sex mapping reads.

    Parameters
    ----------
    video_dir (pathlib.Path)
        Directory to write the HDF5 into (the session root's ``video``).
    track_names (tuple of str)
        Animal id strings, in track order.
    name (str)
        File name (must end in ``_points3d_translated_rotated_metric.h5``).
    sexes (tuple of str)
        The metadata sex of each track, paired with ``track_names`` in order
        (extra entries are ignored); defaults to a male-female pair.

    Returns
    -------
    path (pathlib.Path)
        The written HDF5 path.
    """

    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / name
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "track_names",
            data=np.array([n.encode("utf-8") for n in track_names]),
        )
    _write_session_metadata(video_dir.parent, dict(zip(track_names, sexes)))
    return path


def _write_session_metadata(session_root: pathlib.Path, subject_sexes: dict[str, str]) -> pathlib.Path:
    """
    Description
    -----------
    Write a minimal ``<session>_metadata.yaml`` whose ``Subjects`` block
    lists each subject id with its sex, as the recording GUI does.

    Parameters
    ----------
    session_root (pathlib.Path)
        Session root directory the metadata file goes into.
    subject_sexes (dict)
        ``{subject_id: sex}``.

    Returns
    -------
    path (pathlib.Path)
        The written metadata path.
    """

    session_root.mkdir(parents=True, exist_ok=True)
    lines = ["Subjects:"]
    for subject_id, sex in subject_sexes.items():
        lines += [f"- subject_id: '{subject_id}'", f"  sex: {sex}"]
    path = session_root / f"{session_root.name}_metadata.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(autouse=True)
def _close_figs():
    """
    Description
    -----------
    Close every open matplotlib figure after each test so the Agg
    backend does not accumulate state across the suite.

    Parameters
    ----------

    Returns
    -------
    None
    """

    yield
    plt.close("all")


# ---- USVSpectrogramPlotter.__init__ (A2) ----------------------------------


def test_init_stashes_kwargs_and_defaults(tmp_path):
    """Init stashes kwargs verbatim and applies message_output /
    cmap_override defaults."""
    settings = _base_settings()
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=settings,
    )
    assert plotter.visualizations_parameter_dict is settings
    assert plotter.message_output is print
    assert plotter.cmap_override is None
    # ``app_context_bool`` reflects whether a QApplication is alive, which
    # depends on whether a GUI test ran earlier in the same session -- so
    # only its type is asserted, not its value.
    assert isinstance(plotter.app_context_bool, bool)


def test_init_requires_root_directory():
    """Missing root_directory raises ValueError (A2 validation)."""
    with pytest.raises(ValueError, match="root_directory"):
        USVSpectrogramPlotter(visualizations_parameter_dict=_base_settings())


def test_init_requires_settings_dict(tmp_path):
    """Missing visualizations_parameter_dict raises ValueError (A2)."""
    with pytest.raises(ValueError, match="visualizations_parameter_dict"):
        USVSpectrogramPlotter(root_directory=str(tmp_path))


def test_init_preserves_explicit_message_output(tmp_path):
    """An explicit message_output is preserved (not overwritten by print)."""

    def _logger(_msg: str) -> None:
        return None

    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
        message_output=_logger,
    )
    assert plotter.message_output is _logger


# ---- _resolve_cmap --------------------------------------------------------


def test_resolve_cmap_override_wins(tmp_path):
    """A cmap_override takes precedence over the settings cmap string."""
    sentinel = object()
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
        cmap_override=sentinel,
    )
    assert plotter._resolve_cmap() is sentinel


def test_resolve_cmap_from_settings(tmp_path):
    """With no override, the cmap is read from the figures block."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    assert plotter._resolve_cmap() == "inferno"


# ---- _load_audio_memmap (A1) ----------------------------------------------


def test_load_audio_memmap_parses_filename(tmp_path):
    """The sr / sample / channel triple is parsed out of the encoded
    basename and the memmap is correctly shaped."""
    _write_audio_memmap(tmp_path, sampling_rate=192_000, sample_num=1_500, channel_num=2)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    audio, sr, n, ch, basename = plotter._load_audio_memmap()
    assert sr == 192_000
    assert n == 1_500
    assert ch == 2
    assert audio.shape == (1_500, 2)
    assert basename.endswith("_int16.mmap")


def test_load_audio_memmap_rejects_malformed_name(tmp_path):
    """A memmap whose name lacks the encoded triple is never picked up: the
    exact-name 'ultrasonic' band lookup finds no match and raises FileNotFoundError
    instead of an opaque parse failure (A1)."""
    bad_dir = tmp_path / "audio" / "hpss_filtered"
    bad_dir.mkdir(parents=True)
    (bad_dir / "totally_wrong_int16.mmap").write_bytes(np.zeros(8, dtype=np.int16).tobytes())
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    with pytest.raises(FileNotFoundError, match="ultrasonic audio memmap"):
        plotter._load_audio_memmap()


def test_load_audio_memmap_missing_file_raises(tmp_path):
    """No memmap under the root surfaces a FileNotFoundError."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    with pytest.raises(FileNotFoundError):
        plotter._load_audio_memmap()


# ---- _resolve_window ------------------------------------------------------


def test_resolve_window_end_zero_means_full_file(tmp_path):
    """An end of 0 is resolved to sample_num / sampling_rate."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(time_window=(0.0, 0.0)),
    )
    start_sig, end_sig, start_s, end_s = plotter._resolve_window(
        sample_num=2_000, sampling_rate=250_000
    )
    assert start_sig == 0
    assert end_sig == 2_000
    assert start_s == 0.0
    assert end_s == pytest.approx(2_000 / 250_000)


def test_resolve_window_explicit_bounds(tmp_path):
    """An explicit window maps to rounded sample indices."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(time_window=(0.1, 0.5)),
    )
    start_sig, end_sig, start_s, end_s = plotter._resolve_window(
        sample_num=500_000, sampling_rate=250_000
    )
    assert start_sig == 25_000
    assert end_sig == 125_000
    assert (start_s, end_s) == (0.1, 0.5)


def test_resolve_window_clamps_out_of_range_bounds(tmp_path):
    """A window extending past the recording (or before 0) is clamped to
    [0, sample_num], with the seconds re-derived from the clamped indices, so a
    caller's time vector (num = end_signal - start_signal) matches its numpy-
    clamped data slice instead of raising a length mismatch in ax.plot."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(time_window=(-1.0, 10.0)),
    )
    # -1 s start -> clamped to 0; 10 s end @ 250 kHz = 2_500_000 > 500_000 samples.
    start_sig, end_sig, start_s, end_s = plotter._resolve_window(
        sample_num=500_000, sampling_rate=250_000
    )
    assert start_sig == 0
    assert end_sig == 500_000
    assert (start_s, end_s) == pytest.approx((0.0, 2.0))


# ---- _compute_magnitude_spectrogram ---------------------------------------


def test_compute_magnitude_spectrogram_shape(tmp_path):
    """The magnitude spectrogram has 1 + nfft/2 frequency bins and is
    non-negative."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    seg = np.sin(np.linspace(0, 40, 2_000)).astype(np.float32)
    mag = plotter._compute_magnitude_spectrogram(seg, nfft=256)
    assert mag.shape[0] == 256 // 2 + 1
    assert np.all(mag >= 0.0)


# ---- _render_raw_audio / _render_spectrogram ------------------------------


def test_render_raw_audio_zero_signal(tmp_path):
    """A flat (all-zero) segment does not crash the amplitude auto-scale
    (the zero-peak guard kicks in)."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    fig, ax = plt.subplots()
    time_vec = np.linspace(0, 1, 100)
    plotter._render_raw_audio(
        ax, time_vec, np.zeros(100), color="#808080", title="flat"
    )
    lo, hi = ax.get_ylim()
    assert lo < hi


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_render_spectrogram_with_and_without_cbar(tmp_path):
    """Rendering a spectrogram panel works with and without a colorbar."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    mag = np.abs(np.random.default_rng(1).random((129, 40))) + 0.1
    for plot_cbar in (True, False):
        fig, ax = plt.subplots()
        plotter._render_spectrogram(
            ax=ax,
            fig=fig,
            magnitude=mag,
            sampling_rate=250_000,
            nfft=256,
            start_time_sec=0.0,
            freq_limits_hz=(10_000.0, 40_000.0),
            cmap="inferno",
            vmin=-70,
            vmax=0,
            title="spec",
            plot_cbar=plot_cbar,
        )
        plt.close(fig)


# ---- _save_figure ---------------------------------------------------------


def test_save_figure_noop_when_disabled(tmp_path):
    """With save_fig False, nothing is written."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(save_fig=False),
    )
    fig, _ = plt.subplots()
    plotter._save_figure(fig, "ch00", "audio_250000_2000_3_int16.mmap")
    assert not list(tmp_path.rglob("*.png"))


def test_save_figure_writes_to_data_animation_examples_when_save_dir_empty(tmp_path):
    """An empty save_dir routes the figure to <root>/data_animation_examples and
    the file name encodes the mode suffix and time window."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(
            save_fig=True, save_dir="", time_window=(0.0, 0.0)
        ),
    )
    fig, _ = plt.subplots()
    plotter._save_figure(fig, "ch00", "audio_250000_2000_3_int16.mmap")
    written = list((tmp_path / "data_animation_examples").glob("*.png"))
    assert len(written) == 1
    assert "ch00" in written[0].name


def test_save_figure_auto_opens_only_when_enabled_and_gui(tmp_path, mocker):
    """The saved figure is opened in the OS viewer only when auto_open_figure is
    on AND there is a GUI context; never otherwise."""
    run_mock = mocker.patch("usv_playpen.visualizations.make_usv_spectrograms.subprocess.run")
    startfile_mock = mocker.patch(
        "usv_playpen.visualizations.make_usv_spectrograms.os.startfile", create=True
    )

    # enabled + GUI context -> opened
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(
            save_fig=True, save_dir="", time_window=(0.0, 0.0), auto_open_figure=True
        ),
    )
    plotter.app_context_bool = True
    fig, _ = plt.subplots()
    plotter._save_figure(fig, "ch00", "audio_250000_2000_3_int16.mmap")
    assert run_mock.called or startfile_mock.called

    # GUI context but auto_open_figure off -> NOT opened
    run_mock.reset_mock()
    startfile_mock.reset_mock()
    plotter_off = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(
            save_fig=True, save_dir="", time_window=(0.0, 0.0), auto_open_figure=False
        ),
    )
    plotter_off.app_context_bool = True
    fig_off, _ = plt.subplots()
    plotter_off._save_figure(fig_off, "ch01", "audio_250000_2000_3_int16.mmap")
    assert not run_mock.called
    assert not startfile_mock.called

    # auto_open_figure on but no GUI context -> NOT opened
    plotter_headless = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(
            save_fig=True, save_dir="", time_window=(0.0, 0.0), auto_open_figure=True
        ),
    )
    plotter_headless.app_context_bool = False
    fig_h, _ = plt.subplots()
    plotter_headless._save_figure(fig_h, "ch02", "audio_250000_2000_3_int16.mmap")
    assert not run_mock.called
    assert not startfile_mock.called


def test_save_figure_explicit_save_dir(tmp_path):
    """An explicit save_dir is honoured."""
    out = tmp_path / "out"
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(
            save_fig=True, save_dir=str(out)
        ),
    )
    fig, _ = plt.subplots()
    plotter._save_figure(fig, "all_channels", "audio_250000_2000_3_int16.mmap")
    assert len(list(out.glob("*.png"))) == 1


# ---- plot_single_channel / plot_all_channels ------------------------------


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_single_channel_returns_figure(tmp_path):
    """plot_single_channel renders and (when enabled) saves a figure."""
    _write_audio_memmap(tmp_path)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(save_fig=True),
    )
    fig = plotter.plot_single_channel()
    assert isinstance(fig, plt.Figure)
    assert list((tmp_path / "data_animation_examples").glob("*.png"))


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_single_channel_with_raw_audio(tmp_path):
    """plot_raw_audio True adds the waveform row (2 axes)."""
    _write_audio_memmap(tmp_path)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(plot_raw_audio=True),
    )
    fig = plotter.plot_single_channel(channel=1)
    assert len(fig.axes) >= 2


def test_plot_single_channel_out_of_range(tmp_path):
    """An out-of-range channel raises ValueError."""
    _write_audio_memmap(tmp_path, channel_num=2)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    with pytest.raises(ValueError, match="out of range"):
        plotter.plot_single_channel(channel=9)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_all_channels(tmp_path):
    """plot_all_channels stacks every channel; with raw audio there are
    two rows per channel."""
    _write_audio_memmap(tmp_path, channel_num=2)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(plot_raw_audio=True),
    )
    fig = plotter.plot_all_channels()
    assert len(fig.axes) >= 4


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_all_channels_single_channel(tmp_path):
    """A one-channel recording with no raw-audio row exercises the
    ``total_rows == 1`` axes-wrapping branch."""
    _write_audio_memmap(tmp_path, channel_num=1)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(plot_raw_audio=False),
    )
    fig = plotter.plot_all_channels()
    assert isinstance(fig, plt.Figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_all_channels_no_cbar(tmp_path):
    """The no-colorbar branch renders without error."""
    _write_audio_memmap(tmp_path, channel_num=2)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(plot_cbar=False),
    )
    fig = plotter.plot_all_channels()
    assert isinstance(fig, plt.Figure)


# ---- plot_stitched --------------------------------------------------------


def _setup_stitched_session(tmp_path: pathlib.Path, *, with_mask: bool = True):
    """
    Description
    -----------
    Lay out a session directory for the stitched mode: an int16 memmap,
    a consolidated store keyed by the session basename, and a USV summary
    CSV whose rows align with the store's spectrogram order.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Pytest temporary directory used as the session root.
    with_mask (bool)
        Whether the consolidated store carries a mask group.

    Returns
    -------
    settings (dict)
        A stitched-mode settings dict pointing at the built store.
    """

    _write_audio_memmap(tmp_path)
    session_key = tmp_path.name
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_key, n_usvs=4, n_freq=16, n_time=32, with_mask=with_mask
    )
    _write_usv_summary_csv(
        tmp_path / "audio",
        {
            "start": [0.10, 0.30, 0.55, 0.80],
            "stop": [0.18, 0.38, 0.63, 0.88],
            "noise": [False, True, False, False],
            "qlvm_category": [1, 1, 2, 2],
        },
    )
    return _base_settings(
        mode="stitched",
        freq_limits=(30.0, 120.0),
        time_window=(0.0, 1.0),
        spectrograms_dir=spec_dir,
    )


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_stitched_with_mask(tmp_path):
    """The stitched timeline renders from the consolidated store with the
    SAM2 mask applied."""
    settings = _setup_stitched_session(tmp_path, with_mask=True)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=settings,
    )
    fig = plotter.plot_stitched()
    assert isinstance(fig, plt.Figure)


def test_plot_stitched_drops_noise_and_keeps_store_row_indices(tmp_path, mocker):
    """The stitched timeline leaves out the noise segment (row 1) and hands the canvas
    builder the remaining rows under their ORIGINAL summary row indices, which index the
    spectrogram store."""
    settings = _setup_stitched_session(tmp_path, with_mask=True)
    plotter = USVSpectrogramPlotter(root_directory=str(tmp_path), visualizations_parameter_dict=settings)
    # The synthetic memmap is shorter than the calls; widen the window over all four.
    mocker.patch.object(plotter, "_resolve_window", return_value=(0, 250_000, 0.0, 1.0))
    canvas_builder = mocker.patch.object(plotter, "_build_stitched_canvas", side_effect=RuntimeError("stop"))
    with pytest.raises(RuntimeError, match="stop"):
        plotter.plot_stitched()
    in_window_df = canvas_builder.call_args.args[-1]
    assert in_window_df["row_index"].to_list() == [0, 2, 3]


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_stitched_without_mask(tmp_path):
    """apply_mask False skips the mask branch but still renders."""
    settings = _setup_stitched_session(tmp_path, with_mask=False)
    settings["make_usv_spectrograms"]["apply_mask"] = False
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=settings,
    )
    fig = plotter.plot_stitched()
    assert isinstance(fig, plt.Figure)


def test_plot_stitched_missing_session_group(tmp_path):
    """A consolidated store without the session's group raises KeyError."""
    _write_audio_memmap(tmp_path)
    spec_dir = _write_spectrograms_dir(tmp_path / "spectrograms", "some_other_session", n_usvs=4)
    _write_usv_summary_csv(
        tmp_path / "audio",
        {"start": [0.1], "stop": [0.2], "qlvm_category": [1]},
    )
    settings = _base_settings(
        mode="stitched",
        freq_limits=(30.0, 120.0),
        time_window=(0.0, 1.0),
        spectrograms_dir=spec_dir,
    )
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=settings,
    )
    with pytest.raises(KeyError):
        plotter.plot_stitched()


def test_plot_stitched_freq_limits_out_of_range(tmp_path):
    """A freq_limits window selecting no store bins raises ValueError."""
    settings = _setup_stitched_session(tmp_path, with_mask=False)
    settings["make_usv_spectrograms"]["freq_limits"] = [200.0, 300.0]
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=settings,
    )
    with pytest.raises(ValueError, match="selects no"):
        plotter.plot_stitched()


# ---- make_usv_spectrograms dispatch ---------------------------------------


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_dispatch_single(tmp_path):
    """mode='single' dispatches to plot_single_channel."""
    _write_audio_memmap(tmp_path)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(mode="single"),
    )
    assert isinstance(plotter.make_usv_spectrograms(), plt.Figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_dispatch_all(tmp_path):
    """mode='all' dispatches to plot_all_channels."""
    _write_audio_memmap(tmp_path, channel_num=2)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(mode="all"),
    )
    assert isinstance(plotter.make_usv_spectrograms(), plt.Figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_dispatch_stitched(tmp_path):
    """mode='stitched' dispatches to plot_stitched."""
    settings = _setup_stitched_session(tmp_path)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=settings,
    )
    assert isinstance(plotter.make_usv_spectrograms(), plt.Figure)


def test_dispatch_unknown_mode(tmp_path):
    """An unknown mode raises ValueError."""
    _write_audio_memmap(tmp_path)
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(mode="banana"),
    )
    with pytest.raises(ValueError, match="Unknown make_usv_spectrograms.mode"):
        plotter.make_usv_spectrograms()


# ---- plot_usv_property_histograms -----------------------------------------


def _write_sessions_txt(tmp_path: pathlib.Path, roots: list[pathlib.Path]) -> pathlib.Path:
    """
    Description
    -----------
    Write a sessions-list text file (one root per line) plus a comment
    and a blank line to exercise the skip logic.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Directory to write the txt file into.
    roots (list of pathlib.Path)
        Session root directories to list.

    Returns
    -------
    path (pathlib.Path)
        The written txt path.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "sessions.txt"
    lines = ["# a comment", ""] + [str(r) for r in roots]
    path.write_text("\n".join(lines))
    return path


def test_plot_usv_property_histograms(tmp_path):
    """Histograms pool per-USV properties across listed sessions and
    drop noise rows."""
    sess = tmp_path / "sess1"
    _write_usv_summary_csv(
        sess / "audio",
        {
            "duration": [0.05, 0.10, 0.20, 0.30],
            "mean_amplitude": [0.5, 1.0, 1.5, 2.0],
            "mean_freq_hz": [40_000, 60_000, 80_000, 100_000],
            "freq_bandwidth_hz": [10_000, 20_000, 50_000, 70_000],
            "spectral_entropy": [1.0, 2.0, 3.0, 4.0],
            "qlvm_category": [1, 1, 1, 2],
            "noise": [True, False, False, False],
        },
    )
    txt = _write_sessions_txt(tmp_path, [sess])
    out = tmp_path / "hist.svg"
    fig = plot_usv_property_histograms(
        sessions_txt_path=str(txt),
        output_path=str(out),
        fig_format="svg",
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)
    assert out.exists()


def test_plot_usv_property_histograms_skips_missing_session(tmp_path):
    """A listed session with no CSV is logged and skipped (no raise)."""
    missing = tmp_path / "ghost"
    missing.mkdir()
    txt = _write_sessions_txt(tmp_path, [missing])
    logs: list[str] = []
    fig = plot_usv_property_histograms(
        sessions_txt_path=str(txt),
        message_output=logs.append,
    )
    assert isinstance(fig, plt.Figure)
    assert any("[skip]" in m for m in logs)


# ---- _count_usvs_per_session / plot_session_type_usv_counts ---------------


def test_count_usvs_per_session(tmp_path):
    """Per-session non-noise counts are returned in order."""
    sess = tmp_path / "s"
    _write_usv_summary_csv(
        sess / "audio",
        {"noise": [True, True, False, False, False]},
    )
    txt = _write_sessions_txt(tmp_path, [sess])
    counts = _count_usvs_per_session(str(txt), True, lambda *_: None)
    assert counts.tolist() == [3.0]


def test_plot_session_type_usv_counts(tmp_path):
    """The three-type bar chart renders with SEM error bars."""
    txts = {}
    for kind in ("mf", "ff", "lm"):
        s1 = tmp_path / f"{kind}_1"
        s2 = tmp_path / f"{kind}_2"
        _write_usv_summary_csv(s1 / "audio", {"qlvm_category": [1, 1, 2]})
        _write_usv_summary_csv(s2 / "audio", {"qlvm_category": [1, 2, 2, 2]})
        txts[kind] = _write_sessions_txt(tmp_path / f"list_{kind}", [s1, s2])
    out = tmp_path / "counts.pdf"
    fig = plot_session_type_usv_counts(
        male_female_txt_path=str(txts["mf"]),
        female_female_txt_path=str(txts["ff"]),
        lone_male_txt_path=str(txts["lm"]),
        output_path=str(out),
        fig_format="pdf",
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)
    assert out.exists()


# ---- _resolve_session_emitter_ids / plot_session_usv_timeline -------------


def test_resolve_session_emitter_sexes(tmp_path):
    """A male-female session maps each track to its metadata sex."""
    _write_tracking_h5(tmp_path / "video", ("M", "F"))
    assert _resolve_session_emitter_sexes(str(tmp_path)) == {"M": "male", "F": "female"}


def test_resolve_session_emitter_sexes_ignores_the_track_slot(tmp_path):
    """Sex comes from the metadata, not the slot: a female-female session maps
    both tracks female, and a female listed first stays female."""
    _write_tracking_h5(tmp_path / "video", ("A", "B"), sexes=("female", "female"))
    assert _resolve_session_emitter_sexes(str(tmp_path)) == {"A": "female", "B": "female"}
    other = tmp_path / "swapped"
    _write_tracking_h5(other / "video", ("F", "M"), sexes=("female", "male"))
    assert _resolve_session_emitter_sexes(str(other)) == {"F": "female", "M": "male"}


def test_resolve_session_emitter_sexes_unmatched_track_raises(tmp_path):
    """A track with no metadata subject raises instead of defaulting to its slot."""
    _write_tracking_h5(tmp_path / "video", ("M", "F"))
    _write_session_metadata(tmp_path, {"M": "male", "somebody_else": "female"})
    with pytest.raises(ValueError, match="no subject with a recorded sex"):
        _resolve_session_emitter_sexes(str(tmp_path))


def test_resolve_session_emitter_sexes_too_few(tmp_path):
    """Fewer than two tracked animals raises ValueError."""
    _write_tracking_h5(tmp_path / "video", ("only_one",))
    with pytest.raises(ValueError, match="need at least two"):
        _resolve_session_emitter_sexes(str(tmp_path))


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_session_usv_timeline(tmp_path):
    """Every non-noise USV is drawn as a colored interval; the optional
    time window clips the strip."""
    _write_tracking_h5(tmp_path / "video", ("M", "F"))
    _write_usv_summary_csv(
        tmp_path / "audio",
        {
            "start": [0.1, 0.5, 1.0, 2.0],
            "stop": [0.2, 0.6, 1.1, 2.1],
            "emitter": ["M", "F", "ghost", "M"],
            "qlvm_category": [1, 1, 1, 2],
            "noise": [False, False, True, False],
        },
    )
    out = tmp_path / "timeline.svg"
    fig = plot_session_usv_timeline(
        session_root=str(tmp_path),
        time_window=(0.0, 1.5),
        output_path=str(out),
        fig_format="svg",
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)
    assert out.exists()


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_plot_session_usv_timeline_full_session(tmp_path):
    """With no time_window the x-axis spans the whole session."""
    _write_tracking_h5(tmp_path / "video", ("M", "F"))
    _write_usv_summary_csv(
        tmp_path / "audio",
        {
            "start": [0.1, 0.5],
            "stop": [0.2, 0.6],
            "emitter": ["M", "F"],
            "qlvm_category": [1, 2],
        },
    )
    fig = plot_session_usv_timeline(
        session_root=str(tmp_path),
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)


# ---- build_pooled_embeddings_df -------------------------------------------


def _write_embedding_session(root: pathlib.Path, session_id: str):
    """
    Description
    -----------
    Write a per-session USV summary CSV carrying the embedding coordinate
    / label / extra columns plus a tracking HDF5, under ``root``.

    Parameters
    ----------
    root (pathlib.Path)
        Session root directory.
    session_id (str)
        Used only for documentation symmetry; the session id is derived
        from ``root.name`` by the loader.

    Returns
    -------
    None
    """

    _write_tracking_h5(root / "video", ("M", "F"))
    _write_usv_summary_csv(
        root / "audio",
        {
            "qlvm1": [1.1, 1.2, 1.3, 1.4],
            "qlvm2": [1.5, 1.6, 1.7, 1.8],
            "qlvm_duration1": [0.1, 0.2, 0.3, 0.4],
            "qlvm_duration2": [0.5, 0.6, 0.7, 0.8],
            "qlvm_duration_category": [1, 2, 1, 2],
            "noise": [True, False, False, False],
            "usv": [None, False, True, True],
            "squeak": [None, True, False, True],
            "qlvm_squeak1": [None, 0.25, None, None],
            "qlvm_squeak2": [None, 0.75, None, None],
            "qlvm_category": [1, 1, 2, 2],
            "qlvm_supercategory": [1, 1, 2, 2],
            "emitter": ["M", "F", "M", "ghost"],
            "duration": [0.05, 0.06, 0.07, 0.08],
            "mean_freq_hz": [40_000, 60_000, 80_000, 100_000],
            "peak_freq_hz": [45_000, 65_000, 85_000, 105_000],
            "freq_bandwidth_hz": [5_000, 6_000, 7_000, 8_000],
            "mean_amplitude": [0.1, 0.2, 0.3, 0.4],
            "max_amplitude": [0.5, 0.6, 0.7, 0.8],
            "spectral_entropy": [1.0, 1.1, 1.2, 1.3],
        },
    )


def test_build_pooled_embeddings_df_and_cache(tmp_path):
    """Pooled embeddings are built from CSVs, noise-filtered, sex-mapped,
    and round-trip through the parquet cache."""
    sess = tmp_path / "20230101_000000"
    _write_embedding_session(sess, "20230101_000000")
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "cache.parquet"
    pooled = build_pooled_embeddings_df(
        sessions_txt_path=str(txt),
        cache_path=str(cache),
        message_output=lambda *_: None,
    )
    assert cache.exists()
    assert pooled.height == 3  # the row the noise classifier flagged is dropped
    # every map's coordinates (null-filled where a map is missing) and the one category
    # column (optional, never null-filled); retired label columns a summary still carries
    # (qlvm_supercategory, qlvm_duration_category) are not pooled
    assert set(EMBEDDING_COORD_COLS).issubset(pooled.columns)
    assert "qlvm_category" in pooled.columns
    assert not {"qlvm_supercategory", "qlvm_duration_category"} & set(pooled.columns)
    assert pooled["qlvm_duration1"].to_list() == [0.2, 0.3, 0.4]
    assert pooled["qlvm_entropy1"].null_count() == pooled.height
    # the usv / squeak booleans are carried as Boolean (the thumbnails' default filter and
    # the explorer derive the call class from them)
    assert pooled["usv"].to_list() == [False, True, True]
    assert pooled["squeak"].to_list() == [True, False, True]
    assert pooled.schema["usv"] == pls.Boolean and pooled.schema["squeak"] == pls.Boolean
    assert "call_class" not in pooled.columns
    # the squeak map's coordinates are carried, null off the squeak rows
    assert pooled["qlvm_squeak1"].to_list() == [0.25, None, None]
    assert pooled["qlvm_squeak2"].to_list() == [0.75, None, None]
    assert "sex" in pooled.columns
    assert "emitter" in pooled.columns  # raw animal id retained for the explorer tooltip
    assert set(pooled["sex"].to_list()) <= {"male", "female", "unassigned"}
    # The per-USV acoustic features are pulled into the pool (continuous
    # color-by metrics for the embedding explorer).
    assert {"mean_amplitude", "peak_freq_hz", "spectral_entropy"}.issubset(pooled.columns)

    # Second call hits the cache (every required column present).
    logs: list[str] = []
    cached = build_pooled_embeddings_df(
        sessions_txt_path=str(txt),
        cache_path=str(cache),
        message_output=logs.append,
    )
    assert cached.height == pooled.height
    assert any("from cache" in m for m in logs)


def test_build_pooled_embeddings_df_sex_comes_from_metadata(tmp_path):
    """The pooled 'sex' column is the emitter's metadata sex, never its track
    slot: in a female-female session the track-0 animal is female; a call with
    no attributed emitter stays unassigned; editing the metadata invalidates
    the cache."""
    sess = tmp_path / "20230101_000000"
    _write_embedding_session(sess, "20230101_000000")
    _write_session_metadata(sess, {"M": "female", "F": "female"})
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "cache.parquet"
    pooled = build_pooled_embeddings_df(
        sessions_txt_path=str(txt), cache_path=str(cache), message_output=lambda *_: None,
    )
    assert dict(zip(pooled["emitter"].to_list(), pooled["sex"].to_list())) == {
        "F": "female", "M": "female", "ghost": "unassigned"}

    meta_path = sess / f"{sess.name}_metadata.yaml"
    meta_path.write_text("Subjects:\n- subject_id: 'M'\n  sex: male\n- subject_id: 'F'\n  sex: female\n")
    os.utime(meta_path, ns=(meta_path.stat().st_atime_ns, meta_path.stat().st_mtime_ns + 10**9))
    logs: list[str] = []
    rebuilt = build_pooled_embeddings_df(
        sessions_txt_path=str(txt), cache_path=str(cache), message_output=logs.append,
    )
    assert any("stale" in m for m in logs)
    assert dict(zip(rebuilt["emitter"].to_list(), rebuilt["sex"].to_list()))["M"] == "male"


def test_build_pooled_embeddings_df_unmatched_track_raises(tmp_path):
    """A tracked animal the metadata cannot resolve raises instead of being
    assigned a sex by its slot."""
    sess = tmp_path / "20230101_000000"
    _write_embedding_session(sess, "20230101_000000")
    _write_session_metadata(sess, {"M": "male"})
    txt = _write_sessions_txt(tmp_path, [sess])
    with pytest.raises(ValueError, match="'F' has no subject"):
        build_pooled_embeddings_df(sessions_txt_path=str(txt), message_output=lambda *_: None)


def test_build_pooled_embeddings_df_skips_empty_session(tmp_path):
    """A session whose usv_summary has zero rows (empty columns infer as String,
    which would break the integer noise filter / vertical concat) is skipped, not
    crashed on."""
    good = tmp_path / "20230101_000000"
    _write_embedding_session(good, "20230101_000000")
    empty = tmp_path / "20230102_000000"
    _write_usv_summary_csv(
        empty / "audio",
        {c: [] for c in (
            "qlvm1", "qlvm2", "qlvm_duration1", "qlvm_duration2",
            "qlvm_category", "qlvm_supercategory", "qlvm_duration_category",
            "emitter", "duration", "mean_freq_hz", "peak_freq_hz",
            "freq_bandwidth_hz", "mean_amplitude", "max_amplitude", "spectral_entropy")},
    )
    txt = _write_sessions_txt(tmp_path, [good, empty])
    pooled = build_pooled_embeddings_df(sessions_txt_path=str(txt), message_output=lambda *_: None)
    assert pooled.height == 3  # only the good session's non-noise rows; empty skipped


def test_build_pooled_embeddings_df_coerces_string_numeric_columns(tmp_path):
    """A non-empty session whose numeric columns are CSV-inferred as String (e.g.
    all-null coordinates) is coerced to the common dtype, so the diagonal concat
    across sessions does not raise 'String is incompatible with Float64'."""
    good = tmp_path / "20230101_000000"
    _write_embedding_session(good, "20230101_000000")
    weird = tmp_path / "20230103_000000"
    _write_tracking_h5(weird / "video", ("M", "F"))
    _write_usv_summary_csv(
        weird / "audio",
        {
            "qlvm_duration1": [None, None],  # all-null -> CSV-inferred as String/Null
            "qlvm_duration2": [None, None],
            "qlvm1": [1.0, 2.0],
            "qlvm2": [1.0, 2.0],
            "qlvm_duration_category": [None, None],
            "qlvm_category": [1, 2],
            "qlvm_supercategory": [1, 2],
            "emitter": ["M", "F"],
            "duration": [0.05, 0.06],
            "mean_freq_hz": [40_000, 60_000],
            "peak_freq_hz": [45_000, 65_000],
            "freq_bandwidth_hz": [5_000, 6_000],
            "mean_amplitude": [0.1, 0.2],
            "max_amplitude": [0.5, 0.6],
            "spectral_entropy": [1.0, 1.1],
        },
    )
    txt = _write_sessions_txt(tmp_path, [good, weird])
    pooled = build_pooled_embeddings_df(sessions_txt_path=str(txt), message_output=lambda *_: None)
    # good: 3 non-noise rows; weird: 2 non-noise rows -> 5 total, concat succeeded
    assert pooled.height == 5


def test_build_pooled_embeddings_df_rebuild_on_schema_miss(tmp_path):
    """An old cache missing a required column triggers a transparent
    rebuild."""
    sess = tmp_path / "20230102_000000"
    _write_embedding_session(sess, "20230102_000000")
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "stale.parquet"
    pls.DataFrame({"session_id": ["x"], "row_index": [0]}).write_parquet(cache)
    logs: list[str] = []
    pooled = build_pooled_embeddings_df(
        sessions_txt_path=str(txt),
        cache_path=str(cache),
        message_output=logs.append,
    )
    assert pooled.height == 3
    assert any("missing columns" in m for m in logs)


@pytest.mark.parametrize("old_column", ["call_class", "binary_squeak"])
def test_build_pooled_embeddings_df_rebuilds_an_old_layout_cache(tmp_path, old_column):
    """A cache written in an older layout -- with a ``call_class`` string column, or with
    the retired binary ``squeak`` flag alone -- lacks the ``usv`` boolean and is rebuilt
    even when its summaries fingerprint still matches; the rebuilt table carries the two
    booleans and no ``call_class``."""
    sess = tmp_path / "20230103_000000"
    _write_embedding_session(sess, "20230103_000000")
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "cache.parquet"
    build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache), message_output=lambda *_: None)
    fingerprint_key = "usv_playpen_summaries_fingerprint"
    metadata = {fingerprint_key: pls.read_parquet_metadata(str(cache))[fingerprint_key]}
    old_layout = pls.read_parquet(str(cache)).drop("usv")
    if old_column == "call_class":
        old_layout = old_layout.drop("squeak").with_columns(pls.lit("usv").alias("call_class"))
    old_layout.write_parquet(str(cache), metadata=metadata)
    logs: list[str] = []
    pooled = build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache), message_output=logs.append)
    assert any("missing columns" in m and "usv" in m for m in logs)
    assert {"usv", "squeak"} <= set(pooled.columns)
    assert "call_class" not in pooled.columns


def test_build_pooled_embeddings_df_rebuilds_when_a_summary_changes(tmp_path):
    """A cache built from older summaries is detected as stale by the summaries
    fingerprint (path + size + mtime) and rebuilt, so re-embedded coordinates are
    never served from an old cache; the old cache file is overwritten, not deleted
    beforehand."""
    sess = tmp_path / "20230104_000000"
    _write_embedding_session(sess, "20230104_000000")
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "cache.parquet"
    first = build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache),
                                       message_output=lambda *_: None)
    assert first["qlvm1"].to_list() == [1.2, 1.3, 1.4]

    # Re-embed: same session, new coordinates (and a later modification time).
    summary_path = next((sess / "audio").glob("*_usv_summary.csv"))
    table = pls.read_csv(summary_path).with_columns((pls.col("qlvm1") * 0.0 + 0.25).alias("qlvm1"))
    table.write_csv(summary_path)
    stat = summary_path.stat()
    os.utime(summary_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))

    logs: list[str] = []
    second = build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache),
                                        message_output=logs.append)
    assert any("is stale" in m and "other or older usv_summary.csv" in m for m in logs)
    assert not any("from cache" in m for m in logs)
    assert second["qlvm1"].to_list() == [0.25, 0.25, 0.25]

    # The rebuilt cache now matches and is served.
    logs.clear()
    third = build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache),
                                       message_output=logs.append)
    assert any("from cache" in m for m in logs)
    assert third["qlvm1"].to_list() == [0.25, 0.25, 0.25]


def test_build_pooled_embeddings_df_rebuilds_a_cache_without_fingerprint(tmp_path):
    """A cache that has every required column but no summaries fingerprint (written
    before fingerprints existed, e.g. the shared pooled_embeddings.parquet) is
    treated as stale rather than trusted."""
    sess = tmp_path / "20230105_000000"
    _write_embedding_session(sess, "20230105_000000")
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "legacy.parquet"
    fresh = build_pooled_embeddings_df(sessions_txt_path=str(txt), message_output=lambda *_: None)
    fresh.with_columns(pls.lit(9.0).alias("qlvm1")).write_parquet(cache)  # no metadata
    logs: list[str] = []
    pooled = build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache),
                                        message_output=logs.append)
    assert any("no summaries fingerprint" in m for m in logs)
    assert pooled["qlvm1"].to_list() == [1.2, 1.3, 1.4]


def test_build_pooled_embeddings_df_without_qlvm_labels(tmp_path):
    """Summaries without any QLVM label columns (embedded before
    infer-qlvm-latents wrote the labels) pool fine: the label columns are optional, not
    null-filled, and the written cache passes its own check on the next call."""
    sess = tmp_path / "20230106_000000"
    _write_tracking_h5(sess / "video", ("M", "F"))
    _write_usv_summary_csv(
        sess / "audio",
        {
            "qlvm1": [0.11, 0.12], "qlvm2": [0.15, 0.16],
            "noise": [False, False], "emitter": ["M", "F"], "duration": [0.05, 0.06],
            "mean_freq_hz": [40_000, 60_000], "peak_freq_hz": [45_000, 65_000],
            "freq_bandwidth_hz": [5_000, 6_000], "mean_amplitude": [0.1, 0.2],
            "max_amplitude": [0.5, 0.6], "spectral_entropy": [1.0, 1.1],
        },
    )
    txt = _write_sessions_txt(tmp_path, [sess])
    cache = tmp_path / "nolabels.parquet"
    pooled = build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache),
                                        message_output=lambda *_: None)
    assert pooled.height == 2
    assert "qlvm_category" not in pooled.columns and "qlvm_supercategory" not in pooled.columns
    logs: list[str] = []
    build_pooled_embeddings_df(sessions_txt_path=str(txt), cache_path=str(cache), message_output=logs.append)
    assert any("from cache" in m for m in logs)


def test_build_pooled_embeddings_df_no_sessions(tmp_path):
    """When no session loads, an empty but correctly-typed frame is
    returned."""
    ghost = tmp_path / "ghost"
    ghost.mkdir()
    txt = _write_sessions_txt(tmp_path, [ghost])
    logs: list[str] = []
    pooled = build_pooled_embeddings_df(
        sessions_txt_path=str(txt),
        message_output=logs.append,
    )
    assert pooled.height == 0
    assert "session_id" in pooled.columns
    assert any("No sessions" in m for m in logs)


# ---- pure sampling / geometry helpers -------------------------------------


def test_medoid_xy_edge_cases():
    """Empty -> origin; single point -> itself; the medoid always
    coincides with a data row."""
    assert _medoid_xy(np.zeros((0, 2))) == (0.0, 0.0)
    assert _medoid_xy(np.array([[3.0, 4.0]])) == (3.0, 4.0)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    cx, cy = _medoid_xy(pts)
    assert any(np.allclose([cx, cy], row) for row in pts)


@pytest.mark.parametrize("method", ["random", "nearest", "spread", "grid", "spiral"])
def test_pick_category_samples_methods(method):
    """Each sampling strategy returns up to n_per unique in-range
    indices."""
    rng = np.random.default_rng(7)
    pts = rng.random((40, 2))
    out = _pick_category_samples(pts, n_per=8, method=method, rng=rng)
    assert out.size == 8
    assert len(set(out.tolist())) == out.size
    assert out.min() >= 0 and out.max() < 40


def test_pick_category_samples_seeded_reproducible():
    """The seeded default_rng path makes random sampling reproducible."""
    pts = np.random.default_rng(0).random((30, 2))
    a = _pick_category_samples(pts, 5, "random", np.random.default_rng(42))
    b = _pick_category_samples(pts, 5, "random", np.random.default_rng(42))
    assert a.tolist() == b.tolist()


def test_pick_category_samples_empty():
    """Zero points returns an empty index array."""
    out = _pick_category_samples(
        np.zeros((0, 2)), 5, "random", np.random.default_rng(0)
    )
    assert out.size == 0


def test_pick_category_samples_unknown_method():
    """An unknown sampling method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sampling_method"):
        _pick_category_samples(
            np.random.default_rng(0).random((5, 2)),
            2,
            "nope",
            np.random.default_rng(0),
        )


def test_pick_spiral_with_grid_no_grid():
    """Without a label grid the spiral picker returns n_per snapped
    indices and the dense path is unfiltered."""
    pts = np.random.default_rng(3).random((50, 2))
    picks, xs, ys = _pick_spiral_with_grid(
        pts, n_per=6, cx0=0.5, cy0=0.5, r_max=0.5,
        labels_grid=None, xx=None, yy=None, cluster_label=1, rng=None,
    )
    assert picks.size == 6
    assert xs.size == ys.size and xs.size > 0


def test_pick_spiral_with_grid_zero_radius():
    """A non-positive r_max short-circuits to the first n_take indices."""
    pts = np.random.default_rng(3).random((10, 2))
    picks, xs, ys = _pick_spiral_with_grid(
        pts, n_per=4, cx0=0.0, cy0=0.0, r_max=0.0,
        labels_grid=None, xx=None, yy=None, cluster_label=1,
    )
    assert picks.tolist() == [0, 1, 2, 3]
    assert xs.size == 0 and ys.size == 0


def test_pick_spiral_with_grid_empty_pts():
    """Empty points return three empty arrays."""
    picks, xs, ys = _pick_spiral_with_grid(
        np.zeros((0, 2)), n_per=4, cx0=0.0, cy0=0.0, r_max=1.0,
        labels_grid=None, xx=None, yy=None, cluster_label=1,
    )
    assert picks.size == 0 and xs.size == 0 and ys.size == 0


def test_pick_spiral_with_grid_with_label_grid():
    """A label grid filters the dense spiral to in-cluster cells; with a
    random phase the rotation is applied."""
    rng = np.random.default_rng(11)
    pts = rng.random((60, 2))
    xx = np.linspace(0, 1, 20)
    yy = np.linspace(0, 1, 20)
    labels = np.ones((20, 20))
    labels[:, :10] = 2.0  # half the grid belongs to a different cluster
    picks, xs, ys = _pick_spiral_with_grid(
        pts, n_per=5, cx0=0.5, cy0=0.5, r_max=0.4,
        labels_grid=labels, xx=xx, yy=yy, cluster_label=1, rng=rng,
    )
    assert picks.size == 5


def test_pick_spiral_with_grid_filter_kills_all():
    """When the label grid excludes every spiral cell, the picker falls
    back to the unfiltered dense path instead of returning nothing."""
    rng = np.random.default_rng(13)
    pts = rng.random((40, 2))
    xx = np.linspace(0, 1, 16)
    yy = np.linspace(0, 1, 16)
    labels = np.full((16, 16), 2.0)  # nothing matches cluster_label == 1
    picks, xs, ys = _pick_spiral_with_grid(
        pts, n_per=5, cx0=0.5, cy0=0.5, r_max=0.3,
        labels_grid=labels, xx=xx, yy=yy, cluster_label=1, rng=None,
    )
    assert picks.size == 5
    assert xs.size > 0


def test_pick_category_samples_spiral_degenerate():
    """Spiral sampling on coincident points (r_max == 0) falls back to a
    random draw rather than dividing by zero."""
    pts = np.full((10, 2), 0.5)
    out = _pick_category_samples(pts, n_per=5, method="spiral", rng=np.random.default_rng(0))
    assert out.size == 5
    assert out.max() < 10


# ---- plot_embedding_with_category_thumbnails -----------------------------------


@pytest.fixture(autouse=True)
def _synthetic_category_bundle(qlvm_category_bundle):
    """
    Description
    -----------
    Every test of this module reads the synthetic QLVM category bundle (four
    quadrant categories, label positions at the quadrant centres) instead of the
    production bundle, so the category boundaries / centres / density they draw are
    known and the tests run without the lab share.

    Parameters
    ----------
    qlvm_category_bundle (pathlib.Path)
        The conftest fixture's bundle directory.

    Returns
    -------
    directory (pathlib.Path)
        The bundle directory.
    """

    return qlvm_category_bundle


def _make_pooled_df(session_id: str = "sessA", n_per_cat: int = 6) -> pls.DataFrame:
    """
    Description
    -----------
    Build a synthetic pooled-embeddings DataFrame with two non-noise
    categories (1, 2) plus extra sex / duration / mean_freq_hz columns.
    The row_index values index directly into the matching consolidated
    store group.

    Parameters
    ----------
    session_id (str)
        Session id shared by every row (keys into the store).
    n_per_cat (int)
        Rows per category.

    Returns
    -------
    df (pls.DataFrame)
        The synthetic pooled DataFrame.
    """

    rng = np.random.default_rng(9)
    n = n_per_cat * 2
    cats = [1] * n_per_cat + [2] * n_per_cat
    return pls.DataFrame(
        {
            "session_id": [session_id] * n,
            "row_index": list(range(n)),
            "qlvm1": 0.4 * rng.random(n) + 0.5 * (np.array(cats, dtype=float) - 1.0),
            "qlvm2": 0.4 * rng.random(n) + 0.5 * (np.array(cats, dtype=float) - 1.0),
            "qlvm_category": cats,
            "qlvm_duration1": rng.random(n),
            "qlvm_duration2": rng.random(n),
            "usv": [True] * n,
            "squeak": [False] * n,
            "sex": (["male", "female"] * n)[:n],
            "duration": rng.random(n) * 0.1,
            "mean_freq_hz": rng.random(n) * 50_000 + 40_000,
        }
    )


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_random(tmp_path):
    """The two-panel UMAP + thumbnail figure renders from a pre-built
    pooled DataFrame and the synthetic store (random sampling)."""
    pooled = _make_pooled_df("sessA", n_per_cat=6)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessA", n_usvs=12, n_freq=16, n_time=24)
    out = tmp_path / "umap.png"
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        n_samples_per_category=4,
        pooled_df=pooled,
        output_path=str(out),
        message_output=lambda *_: None,
        seed=42,
    )
    assert isinstance(fig, plt.Figure)
    assert out.exists()


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_conditional_map_colours_by_qlvm_category(tmp_path):
    """qlvm_map='qlvm_duration' groups and colours its calls by the regular map's
    qlvm_category (the only category column) and draws no category boundaries (the
    bundle partitions the regular map's torus only), saying so; the category-ID labels
    sit at the calls' means on that map, not at the bundle's label positions."""
    pooled = _make_pooled_df("sessQ", n_per_cat=6)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessQ", n_usvs=12, n_freq=16, n_time=24)
    messages = []
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        qlvm_map="qlvm_duration",
        n_samples_per_category=3,
        sampling_method="spiral",
        annotate_cluster_ids=True,
        draw_cluster_boundaries=True,
        pooled_df=pooled,
        message_output=messages.append,
    )
    assert any("defined on the qlvm map" in message for message in messages)
    scatter_ax = fig.axes[0]
    assert scatter_ax.get_xlabel() == "QLVM DURATION DIM 1"
    assert not any(isinstance(child, matplotlib.contour.ContourSet) for child in scatter_ax.get_children())
    placed = {text.get_text(): tuple(text.get_position()) for text in scatter_ax.texts if text.get_text() in ("1", "2")}
    for label in (1, 2):
        rows = pooled.filter(pls.col("qlvm_category") == label)
        assert placed[str(label)] == pytest.approx((rows["qlvm_duration1"].mean(), rows["qlvm_duration2"].mean()))


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_spiral_unstretched(tmp_path):
    """Spiral sampling + boundary overlay + unstretched specs + vertical
    tiling exercises the heavier rendering branches."""
    pooled = _make_pooled_df("sessB", n_per_cat=8)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessB", n_usvs=16, n_freq=16, n_time=20)
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        n_samples_per_category=4,
        sampling_method="spiral",
        draw_spiral_overlay=True,
        draw_cluster_boundaries=True,
        annotate_cluster_ids=True,
        unstretched_specs=True,
        tile_orientation="vertical",
        apply_mask=True,
        pooled_df=pooled,
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_explicit_centers(tmp_path):
    """A caller-supplied cluster-center dict drives the spiral origin and
    explicit category colors are honoured."""
    pooled = _make_pooled_df("sessC", n_per_cat=5)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessC", n_usvs=10, n_freq=16, n_time=18)
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        n_samples_per_category=3,
        sampling_method="spiral",
        cluster_centers_xy={1: (0.2, 0.2), 2: (0.7, 0.7)},
        category_colors={1: "#FF0000", 2: "#00FF00"},
        spiral_radius_abs=0.5,
        pooled_df=pooled,
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_json_provenance_centers(tmp_path):
    """A QLVM provenance JSON supplies the spiral cluster centers via
    direct key access on its ``cluster_centers`` list."""
    pooled = _make_pooled_df("sessD", n_per_cat=5)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessD", n_usvs=10, n_freq=16, n_time=18)
    prov = tmp_path / "qlvm_provenance.json"
    prov.write_text(json.dumps({"cluster_centers": [[0.2, 0.2], [0.7, 0.7]]}))
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        n_samples_per_category=3,
        sampling_method="spiral",
        cluster_centers_json_path=str(prov),
        pooled_df=pooled,
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_bundle_centers_and_boundaries(tmp_path):
    """On the regular map the QLVM category-ID labels sit at the category bundle's label
    positions (row i = category i + 1) and its label grid is drawn as the boundaries;
    the labels drawn are the pooled table's qlvm_category values."""
    pooled = _make_pooled_df("sessE", n_per_cat=5)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessE", n_usvs=10, n_freq=16, n_time=18)
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        qlvm_map="qlvm",
        n_samples_per_category=3,
        sampling_method="spiral",
        annotate_cluster_ids=True,
        draw_cluster_boundaries=True,
        pooled_df=pooled,
        message_output=lambda *_: None,
    )
    placed = {
        (text.get_text(), tuple(round(float(v), 4) for v in text.get_position()))
        for axis in fig.axes for text in axis.texts if text.get_text() in ("1", "2")
    }
    assert {("1", (0.25, 0.25)), ("2", (0.75, 0.25))} <= placed
    assert any(isinstance(child, matplotlib.contour.ContourSet) for child in fig.axes[0].get_children())


def test_plot_umap_thumbnails_bundle_label_mismatch_raises(tmp_path):
    """A pooled qlvm_category outside the bundle's 1..k means the summaries were labelled
    with another partition than the bundle drawn: raise."""
    pooled = _make_pooled_df().with_columns(pls.lit(7).alias("qlvm_category"))
    with pytest.raises(ValueError, match="different partitions"):
        plot_embedding_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path="unused",
            qlvm_map="qlvm",
                pooled_df=pooled,
            message_output=lambda *_: None,
        )


def test_plot_umap_thumbnails_bad_qlvm_map(tmp_path):
    """A qlvm_map outside os_utils.QLVM_MAPS (e.g. the retired 'vae') raises
    ValueError before any rendering."""
    with pytest.raises(ValueError, match="qlvm_map must be"):
        plot_embedding_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path="unused",
            qlvm_map="vae",
            pooled_df=_make_pooled_df(),
            message_output=lambda *_: None,
        )


def test_plot_umap_thumbnails_has_no_category_suffix_parameter():
    """qlvm_category is the one category column, so the figure takes no
    category_col_suffix parameter any more."""
    assert "category_col_suffix" not in inspect.signature(plot_embedding_with_category_thumbnails).parameters


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_excludes_squeaks_by_default(tmp_path):
    """Only pure USVs (usv true, squeak false) become thumbnail picks by default: pure
    squeaks and both rows (whose usv is true too) are left out alike, the count is logged,
    and exclude_squeaks=False keeps them."""
    pooled = _make_pooled_df("sessS", n_per_cat=6)
    # the category-1 rows alternate pure squeak (false, true) / both (true, true): with the
    # default filter only category 2 is left
    category_one = pls.col("qlvm_category") == 1
    pooled = pooled.with_columns(
        pls.when(category_one).then(pls.col("row_index") % 2 == 1).otherwise(pls.lit(True)).alias("usv"),
        category_one.alias("squeak"),
    )
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessS", n_usvs=12, n_freq=16, n_time=24)
    logs: list[str] = []
    fig = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused", consolidated_h5_path=str(h5_path),
        n_samples_per_category=3, pooled_df=pooled, message_output=logs.append, seed=1,
    )
    assert isinstance(fig, plt.Figure)
    assert any("Kept the 6 pure USV(s) of 12 placed calls (6 squeak, both or unclassed left out)" in message
               for message in logs)
    fig_all = plot_embedding_with_category_thumbnails(
        sessions_txt_path="unused", consolidated_h5_path=str(h5_path), exclude_squeaks=False,
        n_samples_per_category=3, pooled_df=pooled, message_output=lambda *_: None, seed=1,
    )
    assert isinstance(fig_all, plt.Figure)


def test_plot_umap_thumbnails_squeak_filter_needs_the_column(tmp_path):
    """Excluding squeaks from a pooled table without the usv / squeak booleans raises."""
    with pytest.raises(KeyError, match=r"no \['usv', 'squeak'\] columns"):
        plot_embedding_with_category_thumbnails(
            sessions_txt_path="unused", consolidated_h5_path="unused",
            pooled_df=_make_pooled_df().drop("usv"), message_output=lambda *_: None,
        )


def test_plot_umap_thumbnails_no_categories(tmp_path):
    """A pooled frame with no usable category leaves nothing to draw."""
    pooled = pls.DataFrame(
        {
            "session_id": ["s", "s"],
            "row_index": [0, 1],
            "qlvm1": [0.1, 0.2],
            "qlvm2": [0.3, 0.4],
            "qlvm_category": [None, None],
            "usv": [True, True],
            "squeak": [False, False],
        },
        schema_overrides={"qlvm_category": pls.Int64},
    )
    with pytest.raises(RuntimeError, match="No categories found"):
        plot_embedding_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path="unused",
            pooled_df=pooled,
            message_output=lambda *_: None,
        )


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_bad_tile_orientation(tmp_path):
    """An invalid tile_orientation raises ValueError."""
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessA", n_usvs=12)
    with pytest.raises(ValueError, match="tile_orientation must be"):
        plot_embedding_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path=str(h5_path),
            tile_orientation="diagonal",
            pooled_df=_make_pooled_df("sessA"),
            message_output=lambda *_: None,
        )


def test_bandwidth_split_constant_is_khz():
    """Guard against an accidental unit regression on the bimodal split
    constant (it is in display kHz, not Hz)."""
    assert 0.0 < BANDWIDTH_BIMODAL_SPLIT_KHZ < 200.0


# ---- plot_sequence (per-session embedding + continuous spectrogram) -------


def _write_spectrograms_dir(
    base: pathlib.Path,
    session_key: str,
    *,
    n_usvs: int = 4,
    n_freq: int = 16,
    n_time: int = 32,
    with_mask: bool = True,
) -> str:
    """Lay out a ``shared_resources.spectrograms_dir`` the way the readers resolve
    it: the consolidated store ``<base>/spectrograms_<key>.h5``. The QLVM category
    grid is not under it (it is the category bundle). Returns ``str(base)``."""
    base = pathlib.Path(base)
    base.mkdir(parents=True, exist_ok=True)
    _write_consolidated_h5(
        base / f"spectrograms_{session_key}.h5", session_key,
        n_usvs=n_usvs, n_freq=n_freq, n_time=n_time, with_mask=with_mask,
    )
    return str(base)


def _write_sequence_session(
    tmp_path: pathlib.Path, session_id: str = "20230101_120000", *, with_dur: bool = True
) -> pathlib.Path:
    """Lay out a synthetic session (audio memmap + usv_summary CSV with
    embedding columns + tracking h5) for the sequence figure. Four USVs in
    [0, 0.006] s: male, female, male, unassigned (track_names male_x/female_y)."""
    root = tmp_path / session_id
    audio_dir = root / "audio"
    _write_audio_memmap(root, channel_num=3)
    rows = {
        "start": [0.0005, 0.0015, 0.0030, 0.0045],
        "stop": [0.0010, 0.0020, 0.0035, 0.0050],
        "emitter": ["male_x", "female_y", "male_x", "other_z"],
        # per-USV loudest channel (0-indexed, < channel_num=3); ch 1 is the mode
        "peak_amp_ch": [1.0, 1.0, 0.0, 2.0],
        "qlvm1": [0.2, 0.4, 0.6, 0.8],
        "qlvm2": [0.3, 0.5, 0.7, 0.2],
        # every call a pure USV, so exclude_squeaks keeps all four
        "usv": [True, True, True, True],
        "squeak": [False, False, False, False],
    }
    if with_dur:
        rows["qlvm_duration1"] = [0.1, 0.3, 0.5, 0.7]
        rows["qlvm_duration2"] = [0.6, 0.4, 0.2, 0.9]
    _write_usv_summary_csv(audio_dir, rows, name=f"{session_id}_usv_summary.csv")
    _write_tracking_h5(
        root / "video",
        track_names=("male_x", "female_y"),
        name=f"{session_id}_points3d_translated_rotated_metric.h5",
    )
    return root


def _seq_settings(
    spectrograms_dir: pathlib.Path,
    save_dir: pathlib.Path,
    *,
    qlvm_map: str = "qlvm",
    plot_raw_audio: bool = False,
    apply_mask: bool = True,
    exclude_squeaks: bool = True,
) -> dict:
    """Build a sequence-mode settings dict: a make_usv_spectrograms block (with a
    `sequence` sub-dict, whose ``exclude_squeaks`` is the argument of that name),
    ``shared_resources.qlvm_map`` and the emitter color palettes. The consolidated
    store is resolved from ``spectrograms_dir`` (build it with
    ``_write_spectrograms_dir``); the regular map's landscape is the category
    bundle."""
    settings = _base_settings(
        mode="sequence",
        save_dir=str(save_dir),
        save_fig=True,
        plot_raw_audio=plot_raw_audio,
        plot_cbar=True,
        freq_limits=(30.0, 120.0),
        time_window=(0.0, 0.006),
        spectrograms_dir=str(spectrograms_dir),
        apply_mask=apply_mask,
        channel_of_interest=0,
    )
    settings["shared_resources"]["qlvm_map"] = qlvm_map
    settings["make_usv_spectrograms"]["sequence"] = {
        "draw_boundaries": True,
        "annotate_right": True,
        "mark_usv_segments": True,
        "exclude_squeaks": exclude_squeaks,
    }
    settings["male_colors"] = ["#9AC0CD", "#8CA252"]
    settings["female_colors"] = ["#FF6347", "#B851B4"]
    settings["unassigned_colors"] = ["#C0C0C0"]
    return settings


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
@pytest.mark.parametrize("qlvm_map", ["qlvm", "qlvm_duration"])
def test_plot_sequence_writes_figure(tmp_path, qlvm_map):
    """A sequence figure is rendered and written for the regular and a conditional
    map, sourcing that map's coords from the CSV and specs/audio from the store/memmap."""
    session_id = "20230101_120000"
    root = _write_sequence_session(tmp_path, session_id)
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=4, n_freq=16, n_time=32,
    )
    save_dir = tmp_path / "out"
    settings = _seq_settings(spec_dir, save_dir, qlvm_map=qlvm_map)
    fig = USVSpectrogramPlotter(
        root_directory=str(root), visualizations_parameter_dict=settings
    ).make_usv_spectrograms()
    assert isinstance(fig, plt.Figure)
    assert list(save_dir.glob(f"usv_spectrogram_*sequence_{qlvm_map}_*")) or list(
        save_dir.glob(f"usv_spectrogram_*sequence_{qlvm_map}.*"))


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_sequence_maps_draw_their_cohort_density(tmp_path):
    """The regular map draws the category bundle's density heatmap and boundary mask
    (two images) and a conditional map a bare square with a note (the bundle
    partitions the regular map only), both on the torus [0,1] square with no ticks."""
    session_id = "20230101_120000"
    root = _write_sequence_session(tmp_path, session_id)
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=4, n_freq=16, n_time=32,
    )
    save_dir = tmp_path / "out"
    for qlvm_map in ("qlvm", "qlvm_duration"):
        fig = USVSpectrogramPlotter(
            root_directory=str(root),
            visualizations_parameter_dict=_seq_settings(spec_dir, save_dir, qlvm_map=qlvm_map),
        ).plot_sequence()
        ax = fig.axes[0]
        if qlvm_map == "qlvm":
            assert len(ax.images) == 2
            np.testing.assert_allclose(ax.images[0].get_array(), os_utils.load_qlvm_category_bundle()["density"])
        else:
            assert len(ax.images) == 0
            assert "defined on the qlvm map" in ax.get_title()
        assert ax.get_xlim() == (0.0, 1.0)
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_sequence_raw_audio_uses_loudest_channel(tmp_path):
    """The raw-audio strip auto-picks the window's most-frequent peak_amp_ch
    (channel 1 in the fixture), not the configured channel_of_interest (0)."""
    session_id = "20230101_120000"
    root = _write_sequence_session(tmp_path, session_id)  # peak_amp_ch mode = 1
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=4, n_freq=16, n_time=32,
    )
    settings = _seq_settings(spec_dir, tmp_path / "out", qlvm_map="qlvm", plot_raw_audio=True)
    settings["make_usv_spectrograms"]["channel_of_interest"] = 0  # differs from loudest (1)
    settings["make_usv_spectrograms"]["time_window"] = [0.0, 0.006]
    fig = USVSpectrogramPlotter(
        root_directory=str(root), visualizations_parameter_dict=settings
    ).plot_sequence()

    mm_path = next((root / "audio" / "hpss_filtered").glob("*_int16.mmap"))
    mm = np.memmap(mm_path, dtype=np.int16, mode="r", shape=(2000, 3), order="C")
    raw_ydata = fig.axes[1].lines[0].get_ydata()  # axes: [left, raw, spec, cbar]
    n = len(raw_ydata)
    assert np.array_equal(raw_ydata, np.asarray(mm[:n, 1]))       # loudest channel
    assert not np.array_equal(raw_ydata, np.asarray(mm[:n, 0]))   # NOT channel_of_interest


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_sequence_draws_connecting_line(tmp_path):
    """The window USVs are joined by a single LineCollection with one segment per
    consecutive pair (n - 1 for n window USVs) and per-segment widths that vary
    with the inter-USV interval."""
    from matplotlib.collections import LineCollection

    session_id = "20230101_120000"
    root = _write_sequence_session(tmp_path, session_id)  # 4 window USVs, all with coords
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=4, n_freq=16, n_time=32,
    )
    settings = _seq_settings(spec_dir, tmp_path / "out", qlvm_map="qlvm")
    settings["make_usv_spectrograms"]["time_window"] = [0.0, 0.006]
    fig = USVSpectrogramPlotter(
        root_directory=str(root), visualizations_parameter_dict=settings
    ).plot_sequence()
    ax_left = fig.axes[0]
    line_cols = [c for c in ax_left.collections if isinstance(c, LineCollection)]
    assert len(line_cols) == 1
    # fixture coords do not wrap, so each of the 3 pairs is a single sub-segment
    assert len(line_cols[0].get_segments()) == 3  # 4 USVs -> 3 segments
    # the fixture's silent gaps are not all equal -> widths must vary
    assert len(set(np.round(line_cols[0].get_linewidths(), 6))) > 1


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_sequence_qlvm_path_wraps_on_torus(tmp_path):
    """Every QLVM map is a periodic unit torus: two USVs near opposite edges connect
    via the short toroidal route, so that pair's segment is split into two
    edge-clipped sub-segments; a pair that does not cross a seam is one segment."""
    from matplotlib.collections import LineCollection

    session_id = "20230101_120000"
    root = tmp_path / session_id
    _write_audio_memmap(root)
    rows = {
        "start": [0.001, 0.003], "stop": [0.002, 0.004],
        "emitter": ["male_x", "male_x"],
        "qlvm1": [0.95, 0.05], "qlvm2": [0.5, 0.5],  # opposite x-edges -> wraps
        "qlvm_duration1": [0.45, 0.55], "qlvm_duration2": [0.5, 0.5],  # no seam crossing
        "usv": [True, True], "squeak": [False, False],
    }
    _write_usv_summary_csv(root / "audio", rows, name=f"{session_id}_usv_summary.csv")
    _write_tracking_h5(
        root / "video", track_names=("male_x", "female_y"),
        name=f"{session_id}_points3d_translated_rotated_metric.h5",
    )
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=2, n_freq=16, n_time=32,
    )

    def _n_subsegments(qlvm_map: str) -> int:
        settings = _seq_settings(
            spec_dir, tmp_path / f"out_{qlvm_map}", qlvm_map=qlvm_map
        )
        settings["make_usv_spectrograms"]["time_window"] = [0.0, 0.006]
        fig = USVSpectrogramPlotter(
            root_directory=str(root), visualizations_parameter_dict=settings
        ).plot_sequence()
        lc = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)][0]
        return len(lc.get_segments())

    assert _n_subsegments("qlvm") == 2  # short route wraps the seam -> two pieces
    assert _n_subsegments("qlvm_duration") == 1   # no seam between them -> one piece


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_sequence_map_missing_coords_raises(tmp_path):
    """Choosing a map whose coordinates the session's CSV lacks (qlvm_duration1/qlvm_duration2)
    raises a clear, session-named ValueError naming the map."""
    session_id = "20230101_120000"
    root = _write_sequence_session(tmp_path, session_id, with_dur=False)
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=4, n_freq=16, n_time=32,
    )
    settings = _seq_settings(spec_dir, tmp_path / "out", qlvm_map="qlvm_duration")
    with pytest.raises(ValueError, match="qlvm_duration"):
        USVSpectrogramPlotter(
            root_directory=str(root), visualizations_parameter_dict=settings
        ).plot_sequence()


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
@pytest.mark.parametrize(("exclude_squeaks", "expected_numbers"), [(True, ["1", "2"]), (False, ["1", "2", "3"])])
def test_plot_sequence_drops_noise_and_optionally_squeaks(tmp_path, exclude_squeaks, expected_numbers):
    """The sequence always drops the noise segments and, with ``exclude_squeaks``,
    the squeak-bearing ones too: of four window calls (pure USV, noise, pure squeak,
    pure USV) the map numbers two calls with the flag on and three with it off, and
    the stitched spectrogram still renders (the store rows are looked up by the
    pre-filter row index)."""
    from matplotlib.collections import PathCollection

    session_id = "20230101_120000"
    root = tmp_path / session_id
    _write_audio_memmap(root, channel_num=3)
    rows = {
        "start": [0.0005, 0.0015, 0.0030, 0.0045],
        "stop": [0.0010, 0.0020, 0.0035, 0.0050],
        "emitter": ["male_x", "female_y", "male_x", "female_y"],
        "qlvm1": [0.2, 0.4, 0.6, 0.8],
        "qlvm2": [0.3, 0.5, 0.7, 0.2],
        "noise": [False, True, False, False],
        "usv": [True, None, False, True],
        "squeak": [False, None, True, False],
    }
    _write_usv_summary_csv(root / "audio", rows, name=f"{session_id}_usv_summary.csv")
    _write_tracking_h5(
        root / "video", track_names=("male_x", "female_y"),
        name=f"{session_id}_points3d_translated_rotated_metric.h5",
    )
    spec_dir = _write_spectrograms_dir(
        tmp_path / "spectrograms", session_id, n_usvs=4, n_freq=16, n_time=32,
    )
    settings = _seq_settings(spec_dir, tmp_path / "out", qlvm_map="qlvm", exclude_squeaks=exclude_squeaks)
    fig = USVSpectrogramPlotter(
        root_directory=str(root), visualizations_parameter_dict=settings
    ).plot_sequence()
    ax_left = fig.axes[0]
    assert [t.get_text() for t in ax_left.texts] == expected_numbers
    assert sum(isinstance(c, PathCollection) for c in ax_left.collections) == len(expected_numbers)
    # the noise call (qlvm1 = 0.4) is never placed on the map
    placed_x = [float(c.get_offsets()[0][0]) for c in ax_left.collections if isinstance(c, PathCollection)]
    assert 0.4 not in placed_x
    assert (0.6 in placed_x) is (not exclude_squeaks)


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_sequence_exclude_squeaks_requires_vocal_flags(tmp_path):
    """With ``exclude_squeaks`` on, a summary without the usv / squeak booleans raises
    instead of silently keeping every call."""
    session_id = "20230101_120000"
    root = tmp_path / session_id
    _write_audio_memmap(root, channel_num=3)
    _write_usv_summary_csv(
        root / "audio",
        {"start": [0.001], "stop": [0.002], "emitter": ["male_x"], "qlvm1": [0.2], "qlvm2": [0.3]},
        name=f"{session_id}_usv_summary.csv",
    )
    _write_tracking_h5(
        root / "video", track_names=("male_x", "female_y"),
        name=f"{session_id}_points3d_translated_rotated_metric.h5",
    )
    spec_dir = _write_spectrograms_dir(tmp_path / "spectrograms", session_id, n_usvs=1, n_freq=16, n_time=32)
    settings = _seq_settings(spec_dir, tmp_path / "out", qlvm_map="qlvm", exclude_squeaks=True)
    with pytest.raises(KeyError, match="usv"):
        USVSpectrogramPlotter(root_directory=str(root), visualizations_parameter_dict=settings).plot_sequence()


# ---- render_embedding_thumbnails_for_cohort (cohort driver) ---------------

def test_render_embedding_thumbnails_for_cohort_pools_and_dispatches(tmp_path, monkeypatch):
    """The cohort driver pools (deduplicated, first-seen order) roots from every
    *sessions_list.txt under shared_resources.input_files_directory, resolves the
    store from spectrograms_dir, and dispatches the renderer with the
    embedding_thumbnails block knobs."""
    input_dir = tmp_path / "input_files"
    input_dir.mkdir()
    (input_dir / "a_sessions_list.txt").write_text("/root/sessA\n/root/sessB\n# skip\n")
    (input_dir / "b_sessions_list.txt").write_text("/root/sessB\n/root/sessC\n")  # sessB duplicate
    (input_dir / "ephys_playback_sessions_list.txt").write_text("/root/sessP\n")  # playback -> dropped
    spec_dir = _write_spectrograms_dir(tmp_path / "spectrograms", "sessZ", n_usvs=2)

    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        # Read the combined session list WHILE it still exists (the driver unlinks
        # it in a finally after this call returns).
        captured["pooled_roots"] = pathlib.Path(kwargs["sessions_txt_path"]).read_text().splitlines()
        return plt.figure()

    monkeypatch.setattr(
        "usv_playpen.visualizations.make_usv_spectrograms.plot_embedding_with_category_thumbnails",
        _stub,
    )
    # Force a GUI context + capture the viewer-open (mocked so no real viewer
    # spawns under test) to verify the figure is opened at the end of the render.
    monkeypatch.setattr(
        "usv_playpen.visualizations.make_usv_spectrograms.is_gui_context", lambda: True
    )
    opened = []
    monkeypatch.setattr(
        "usv_playpen.visualizations.make_usv_spectrograms._open_path_in_default_viewer",
        lambda path, message_output=None: opened.append(str(path)),
    )

    viz = {
        "figures": {"save_directory": str(tmp_path / "figs"), "fig_format": "png", "dpi": 150, "seed": 7, "timestamp_in_name": True},
        "shared_resources": {"spectrograms_dir": spec_dir, "input_files_directory": str(input_dir),
                             "qlvm_map": "qlvm_duration"},
        "embedding_thumbnails": {
            "exclude_squeaks": True,
            "n_samples_per_category": 6, "tile_orientation": "vertical",
            "apply_mask": False, "mask_excluded_categories": [], "category_colors": None,
            "sampling_method": "random",
            "draw_cluster_boundaries": True,
            "draw_spiral_overlay": False, "spiral_show_only_for": None,
            "spiral_color": "#000000", "spiral_linewidth": 1.0,
            "spiral_radius_scale": 0.1, "spiral_radius_abs": 0.1,
            "spiral_n_turns": 3, "spiral_random_phase": True,
            "annotate_picks_on_scatter": False, "pick_number_fontsize": 9,
            "annotate_cluster_ids": True, "cluster_id_fontsize": 20,
            "thumbnail_hspace": 0.03,
            "thumbnail_wspace": 0.04, "unstretched_specs": True,
            "scatter_max_points": 1000,
            "fig_size": [10, 8],
        },
    }
    fig = render_embedding_thumbnails_for_cohort(viz, message_output=lambda *_a, **_kw: None)
    assert isinstance(fig, plt.Figure)
    # store resolved to the consolidated spectrograms_*.h5 under spec_dir
    assert pathlib.Path(captured["consolidated_h5_path"]).name.startswith("spectrograms_")
    # combined session list = deduped roots in first-seen order ('# skip' and the playback list dropped)
    assert captured["pooled_roots"] == ["/root/sessA", "/root/sessB", "/root/sessC"]
    # the throwaway combined list is unlinked after the render
    assert not pathlib.Path(captured["sessions_txt_path"]).exists()
    # block knobs forwarded verbatim
    assert captured["qlvm_map"] == "qlvm_duration"
    assert "category_col_suffix" not in captured
    assert captured["exclude_squeaks"] is True
    assert captured["n_samples_per_category"] == 6
    assert captured["tile_orientation"] == "vertical"
    assert tuple(captured["fig_size"]) == (10, 8)
    # fig_dpi + seed are sourced from the general figures block, not the block
    assert captured["fig_dpi"] == 150
    assert captured["seed"] == 7
    # the full set of ported knobs is forwarded
    assert captured["mask_excluded_categories"] == ()        # JSON [] -> tuple
    assert captured["category_colors"] is None
    assert captured["draw_spiral_overlay"] is False
    assert captured["unstretched_specs"] is True
    assert captured["annotate_cluster_ids"] is True
    # boundaries and centres come from the category bundle inside the figure, so
    # neither a k-NN knob nor a centres path is forwarded
    assert not any(key.startswith("knn_") for key in captured)
    assert "cluster_centers_npz_path" not in captured
    # the pooled-embeddings cache is resolved by convention under spectrograms_dir
    assert captured["embeddings_cache_path"] == str(
        pathlib.Path(spec_dir) / "embeddings" / "pooled_embeddings_production.parquet"
    )
    # in a GUI context the saved figure is opened at the end
    assert opened == [captured["output_path"]]
    # timestamp_in_name -> the filename ends with a _YYYYMMDD_HHMMSS stamp
    out_name = pathlib.Path(captured["output_path"]).name
    # named by the map alone (no label-column part)
    assert re.fullmatch(r"embedding_thumbnails_qlvm_duration_\d{8}_\d{6}\.png", out_name)

    # explicit category colours come from JSON, whose object keys are strings; the
    # driver keys them by the int qlvm_category so the figure's lookups match
    viz["embedding_thumbnails"]["category_colors"] = {"1": "#112233", "2": "#445566"}
    render_embedding_thumbnails_for_cohort(viz, message_output=lambda *_a, **_kw: None)
    assert captured["category_colors"] == {1: "#112233", 2: "#445566"}



@pytest.mark.parametrize("default_qlvm_map", ["qlvm", "qlvm_entropy"])
def test_explorer_map_dropdown_uses_the_gui_display_names(default_qlvm_map):
    """The explorer's Map dropdown lists every map of os_utils.QLVM_MAPS under its GUI
    name (os_utils.QLVM_MAP_DISPLAY_NAMES), in that order, plus the squeak map, returns
    the map prefix as its value, and starts on the shared map."""
    _output, definitions = usv_embedding_explorer._widgets.run(
        QLVM_MAPS=os_utils.QLVM_MAPS, QLVM_MAP_DISPLAY_NAMES=os_utils.QLVM_MAP_DISPLAY_NAMES,
        SQUEAK_CLASS_SELECTIONS=SQUEAK_CLASS_SELECTIONS, available_lists={},
        default_qlvm_map=default_qlvm_map, mo=mo,
    )
    map_dropdown = definitions["map_dropdown"]
    expected = [os_utils.QLVM_MAP_DISPLAY_NAMES[qlvm_map] for qlvm_map in os_utils.QLVM_MAPS] + ["Squeaks"]
    assert list(map_dropdown.options) == expected
    assert map_dropdown.options[os_utils.QLVM_MAP_DISPLAY_NAMES["qlvm_duration"]] == "qlvm_duration"
    assert map_dropdown.value == default_qlvm_map
    assert not any("|" in label for label in map_dropdown.options)


def _explorer_scatter_rows(qlvm_map: str, squeak_class: str) -> list[int]:
    """
    Description
    -----------
    Runs the embedding explorer's scatter cell on a five-row pooled table (one
    row per call class: usv, squeak, both, unclassed, usv; squeak-map
    coordinates only on the squeak and both rows) with the given map and
    squeak-class selection, and returns the ``row_index`` of the points it
    plots.

    Parameters
    ----------
    qlvm_map (str)
        The Map dropdown value (``"qlvm"`` or ``"qlvm_squeak"``).
    squeak_class (str)
        The Squeak class dropdown value (a ``SQUEAK_CLASS_SELECTIONS`` key).

    Returns
    -------
    rows (list[int])
        Sorted row indices drawn on the scatter.
    """

    pooled = pls.DataFrame({
        "session_id": ["s"] * 5,
        "row_index": list(range(5)),
        "qlvm1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "qlvm2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "qlvm_squeak1": [None, 0.2, 0.3, None, None],
        "qlvm_squeak2": [None, 0.2, 0.3, None, None],
        "usv": [True, False, True, None, True],
        "squeak": [False, True, True, None, False],
    })

    def control(value: object) -> types.SimpleNamespace:
        """A stand-in for a marimo UI element: only ``.value`` is read."""
        return types.SimpleNamespace(value=value)

    _output, definitions = usv_embedding_explorer._scatter_chart.run(
        CHART_DATA_WIDTH_PX=100, CHART_HEIGHT_PX=100, SQUEAK_CLASS_SELECTIONS=SQUEAK_CLASS_SELECTIONS,
        alt=alt, boundary_dropdown=control("none"), call_class_mask=call_class_mask,
        category_grid=None, category_grid_note=None, QLVM_CATEGORY_COLUMN="qlvm_category", QLVM_CATEGORY_MAP="qlvm",
        color_dropdown=control("none"), global_cmap="viridis",
        map_dropdown=control(qlvm_map), max_points_slider=control(1000), mo=mo, np=np, pd=pd, plt=plt,
        pooled_df=pooled, sessions_select=control([]), sex_colors={},
        squeak_class_dropdown=control(squeak_class),
    )
    return sorted(definitions["chart_data"]["row_index"].tolist())


@pytest.mark.parametrize("squeak_class", list(SQUEAK_CLASS_SELECTIONS))
def test_explorer_usv_maps_show_pure_usvs_only(squeak_class):
    """A USV map draws only pure USVs (usv & ~squeak) -- never a pure squeak, a both segment
    (although its usv is true) or an unclassed row -- whatever the squeak-class control says."""
    assert _explorer_scatter_rows("qlvm", squeak_class) == [0, 4]


@pytest.mark.parametrize(
    ("squeak_class", "expected_rows"),
    [("squeak+both", [1, 2]), ("squeak", [1]), ("both", [2])],
)
def test_explorer_squeak_map_class_filter(squeak_class, expected_rows):
    """The squeak map draws the squeak-bearing classes the squeak-class control selects:
    squeak + both (the default), squeak only, or both only."""
    assert _explorer_scatter_rows("qlvm_squeak", squeak_class) == expected_rows


@pytest.mark.parametrize("qlvm_map", ["qlvm", "qlvm_duration"])
def test_explorer_colours_by_qlvm_category_and_draws_bundle_boundaries_on_the_regular_map(qlvm_map):
    """Every USV map colours by the calls' qlvm_category (conditional maps included);
    the boundaries are the category bundle's grid on the regular map only: there they
    add the haloed boundary layers, on a conditional map none are drawn and the chart
    title says the categories are defined on the regular map."""
    pooled = pls.DataFrame({
        "session_id": ["s"] * 4,
        "row_index": list(range(4)),
        "qlvm1": [0.1, 0.6, 0.2, 0.7],
        "qlvm2": [0.1, 0.2, 0.6, 0.7],
        "qlvm_duration1": [0.3, 0.4, 0.5, 0.6],
        "qlvm_duration2": [0.3, 0.4, 0.5, 0.6],
        "qlvm_category": [1, 2, 3, 4],
        "usv": [True] * 4,
        "squeak": [False] * 4,
    })

    def control(value: object) -> types.SimpleNamespace:
        """A stand-in for a marimo UI element: only ``.value`` is read."""
        return types.SimpleNamespace(value=value)

    grid = os_utils.load_qlvm_category_bundle()["label_grid"]
    _output, definitions = usv_embedding_explorer._scatter_chart.run(
        CHART_DATA_WIDTH_PX=100, CHART_HEIGHT_PX=100, SQUEAK_CLASS_SELECTIONS=SQUEAK_CLASS_SELECTIONS,
        alt=alt, boundary_dropdown=control("category"), call_class_mask=call_class_mask,
        category_grid=grid, category_grid_note=None, QLVM_CATEGORY_COLUMN="qlvm_category", QLVM_CATEGORY_MAP="qlvm",
        color_dropdown=control("category"), global_cmap="viridis",
        map_dropdown=control(qlvm_map), max_points_slider=control(1000), mo=mo, np=np, pd=pd, plt=plt,
        pooled_df=pooled, sessions_select=control([]), sex_colors={},
        squeak_class_dropdown=control("squeak+both"),
    )
    assert sorted(definitions["chart_data"]["qlvm_category"].tolist()) == [1, 2, 3, 4]
    chart = definitions["chart_widget"]._chart
    if qlvm_map == "qlvm":
        assert isinstance(chart, alt.LayerChart) and len(chart.layer) == 3
        assert chart.title is alt.Undefined
    else:
        assert not isinstance(chart, alt.LayerChart)
        assert "defined on the qlvm map" in chart.title
