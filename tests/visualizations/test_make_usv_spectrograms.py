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

import json
import pathlib

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import pytest

from usv_playpen.visualizations.make_usv_spectrograms import (
    BANDWIDTH_BIMODAL_SPLIT_KHZ,
    EMBEDDING_ALL_COLS,
    USVSpectrogramPlotter,
    _count_usvs_per_session,
    _knn_boundary_grid,
    _medoid_xy,
    _pick_category_samples,
    _pick_spiral_with_grid,
    _resolve_session_emitter_ids,
    build_pooled_embeddings_df,
    plot_session_type_usv_counts,
    plot_session_usv_timeline,
    plot_umap_with_category_thumbnails,
    plot_usv_property_histograms,
)


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
    consolidated_h5: str = "",
    apply_mask: bool = True,
    channel_of_interest: int = 0,
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
        Output directory; empty string routes saves to ``<root>/audio``.
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
    consolidated_h5 (str)
        Path to the consolidated spectrogram store (stitched mode).
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
        "figures": {"cmap": "inferno"},
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
            "consolidated_spectrograms_h5": consolidated_h5,
            "apply_mask": apply_mask,
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
    Write a synthetic concatenated int16 audio memmap whose basename
    encodes ``_<sr>_<n_samples>_<n_ch>_int16.mmap`` (so the parse in
    ``_load_audio_memmap`` resolves it) and fill it with a low-amplitude
    multi-channel sine so spectrograms are non-degenerate.

    Parameters
    ----------
    root (pathlib.Path)
        Directory to write the file into (created if absent).
    sampling_rate, sample_num, channel_num (int)
        Encoded in the filename and used to shape the (sample, channel)
        int16 array.

    Returns
    -------
    path (pathlib.Path)
        Path to the written memmap file.
    """

    root.mkdir(parents=True, exist_ok=True)
    name = f"audio_{sampling_rate}_{sample_num}_{channel_num}_int16.mmap"
    path = root / name
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
    pls.DataFrame(rows).write_csv(path)
    return path


def _write_tracking_h5(
    video_dir: pathlib.Path,
    track_names: tuple[str, ...] = ("male_x", "female_y"),
    name: str = "session_points3d_translated_rotated_metric.h5",
) -> pathlib.Path:
    """
    Description
    -----------
    Write a stand-in 3D tracking HDF5 carrying only the ``track_names``
    dataset (the single field ``_resolve_session_emitter_ids`` and the
    pooled-embeddings loader read).

    Parameters
    ----------
    video_dir (pathlib.Path)
        Directory to write the HDF5 into.
    track_names (tuple of str)
        Animal id strings; index 0 is male, index 1 is female.
    name (str)
        File name (must end in ``_points3d_translated_rotated_metric.h5``).

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
    """A memmap whose name lacks the encoded triple raises a clear
    ValueError instead of an opaque parse failure (A1)."""
    bad = tmp_path / "totally_wrong_int16.mmap"
    bad.write_bytes(np.zeros(8, dtype=np.int16).tobytes())
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(),
    )
    with pytest.raises(ValueError, match="Cannot parse sampling rate"):
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


def test_save_figure_writes_to_audio_dir_when_save_dir_empty(tmp_path):
    """An empty save_dir routes the figure to <root>/audio and the file
    name encodes the mode suffix and time window."""
    plotter = USVSpectrogramPlotter(
        root_directory=str(tmp_path),
        visualizations_parameter_dict=_base_settings(
            save_fig=True, save_dir="", time_window=(0.0, 0.0)
        ),
    )
    fig, _ = plt.subplots()
    plotter._save_figure(fig, "ch00", "audio_250000_2000_3_int16.mmap")
    written = list((tmp_path / "audio").glob("*.png"))
    assert len(written) == 1
    assert "ch00" in written[0].name


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
    assert list((tmp_path / "audio").glob("*.png"))


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
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(
        h5_path, session_key, n_usvs=4, n_freq=16, n_time=32, with_mask=with_mask
    )
    _write_usv_summary_csv(
        tmp_path / "audio",
        {
            "start": [0.10, 0.30, 0.55, 0.80],
            "stop": [0.18, 0.38, 0.63, 0.88],
            "vae_supercategory": [1, 1, 2, 2],
        },
    )
    return _base_settings(
        mode="stitched",
        freq_limits=(30.0, 120.0),
        time_window=(0.0, 1.0),
        consolidated_h5=str(h5_path),
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
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "some_other_session")
    _write_usv_summary_csv(
        tmp_path / "audio",
        {"start": [0.1], "stop": [0.2], "vae_supercategory": [1]},
    )
    settings = _base_settings(
        mode="stitched",
        freq_limits=(30.0, 120.0),
        time_window=(0.0, 1.0),
        consolidated_h5=str(h5_path),
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
            "vae_supercategory": [0, 1, 1, 2],
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
        {"vae_supercategory": [0, 0, 1, 2, 2]},
    )
    txt = _write_sessions_txt(tmp_path, [sess])
    counts = _count_usvs_per_session(
        str(txt), "vae_supercategory", (0,), lambda *_: None
    )
    assert counts.tolist() == [3.0]


def test_plot_session_type_usv_counts(tmp_path):
    """The three-type bar chart renders with SEM error bars."""
    txts = {}
    for kind in ("mf", "ff", "lm"):
        s1 = tmp_path / f"{kind}_1"
        s2 = tmp_path / f"{kind}_2"
        _write_usv_summary_csv(s1 / "audio", {"vae_supercategory": [1, 1, 2]})
        _write_usv_summary_csv(s2 / "audio", {"vae_supercategory": [1, 2, 2, 2]})
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


def test_resolve_session_emitter_ids(tmp_path):
    """Track names 0/1 map to (male, female)."""
    _write_tracking_h5(tmp_path / "video", ("M", "F"))
    male, female = _resolve_session_emitter_ids(str(tmp_path))
    assert (male, female) == ("M", "F")


def test_resolve_session_emitter_ids_too_few(tmp_path):
    """Fewer than two tracked animals raises ValueError."""
    _write_tracking_h5(tmp_path / "video", ("only_one",))
    with pytest.raises(ValueError, match="need at least two"):
        _resolve_session_emitter_ids(str(tmp_path))


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
            "vae_supercategory": [1, 1, 0, 2],
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
            "vae_supercategory": [1, 2],
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
            "vae_umap1": [0.1, 0.2, 0.3, 0.4],
            "vae_umap2": [0.5, 0.6, 0.7, 0.8],
            "qlvm_dim1": [1.1, 1.2, 1.3, 1.4],
            "qlvm_dim2": [1.5, 1.6, 1.7, 1.8],
            "vae_category": [1, 2, 1, 2],
            "vae_supercategory": [0, 1, 1, 2],
            "qlvm_category": [1, 1, 2, 2],
            "qlvm_supercategory": [1, 1, 2, 2],
            "emitter": ["M", "F", "M", "ghost"],
            "duration": [0.05, 0.06, 0.07, 0.08],
            "mean_freq_hz": [40_000, 60_000, 80_000, 100_000],
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
    assert pooled.height == 3  # noise row (vae_supercategory == 0) dropped
    assert set(EMBEDDING_ALL_COLS).issubset(pooled.columns)
    assert "sex" in pooled.columns
    assert set(pooled["sex"].to_list()) <= {"male", "female", "unassigned"}

    # Second call hits the cache (every required column present).
    logs: list[str] = []
    cached = build_pooled_embeddings_df(
        sessions_txt_path=str(txt),
        cache_path=str(cache),
        message_output=logs.append,
    )
    assert cached.height == pooled.height
    assert any("from cache" in m for m in logs)


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


@pytest.mark.parametrize("method", ["random", "nearest", "farthest_point", "grid", "spiral"])
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


def test_knn_boundary_grid_shapes():
    """The k-NN boundary grid returns axis ticks and a float label grid
    with NaNs in low-density cells."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 0.2, 50), rng.normal(2, 0.2, 50)])
    y = np.concatenate([rng.normal(0, 0.2, 50), rng.normal(2, 0.2, 50)])
    labels = np.array([1] * 50 + [2] * 50)
    xx, yy, grid = _knn_boundary_grid(
        x, y, labels, x_lo=-1, x_hi=3, y_lo=-1, y_hi=3,
        n_neighbors=5, grid_resolution=40,
    )
    assert xx.shape == (40,) and yy.shape == (40,)
    assert grid.shape == (40, 40)
    assert np.isnan(grid).any()  # empty corners masked out


# ---- plot_umap_with_category_thumbnails -----------------------------------


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
            "vae_umap1": rng.random(n) + np.array(cats, dtype=float),
            "vae_umap2": rng.random(n) - np.array(cats, dtype=float),
            "vae_category": cats,
            "vae_supercategory": cats,
            "qlvm_dim1": rng.random(n),
            "qlvm_dim2": rng.random(n),
            "qlvm_category": cats,
            "qlvm_supercategory": cats,
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
    fig = plot_umap_with_category_thumbnails(
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
def test_plot_umap_thumbnails_spiral_unstretched(tmp_path):
    """Spiral sampling + boundary overlay + unstretched specs + vertical
    tiling exercises the heavier rendering branches."""
    pooled = _make_pooled_df("sessB", n_per_cat=8)
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessB", n_usvs=16, n_freq=16, n_time=20)
    fig = plot_umap_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        map_type="vae",
        category_col_suffix="supercategory",
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
    fig = plot_umap_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        n_samples_per_category=3,
        sampling_method="spiral",
        cluster_centers_xy={1: (1.0, -1.0), 2: (2.0, -2.0)},
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
    prov.write_text(json.dumps({"cluster_centers": [[1.0, -1.0], [2.0, -2.0]]}))
    fig = plot_umap_with_category_thumbnails(
        sessions_txt_path="unused",
        consolidated_h5_path=str(h5_path),
        n_samples_per_category=3,
        sampling_method="spiral",
        cluster_centers_json_path=str(prov),
        pooled_df=pooled,
        message_output=lambda *_: None,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_umap_thumbnails_bad_map_type(tmp_path):
    """An invalid map_type raises ValueError before any rendering."""
    with pytest.raises(ValueError, match="map_type must be"):
        plot_umap_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path="unused",
            map_type="bogus",
            pooled_df=_make_pooled_df(),
            message_output=lambda *_: None,
        )


def test_plot_umap_thumbnails_bad_category_suffix(tmp_path):
    """An invalid category_col_suffix raises ValueError."""
    with pytest.raises(ValueError, match="category_col_suffix must be"):
        plot_umap_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path="unused",
            category_col_suffix="bogus",
            pooled_df=_make_pooled_df(),
            message_output=lambda *_: None,
        )


def test_plot_umap_thumbnails_no_categories(tmp_path):
    """A pooled frame with only noise rows raises RuntimeError."""
    pooled = pls.DataFrame(
        {
            "session_id": ["s", "s"],
            "row_index": [0, 1],
            "vae_umap1": [0.1, 0.2],
            "vae_umap2": [0.3, 0.4],
            "vae_supercategory": [0, 0],
        }
    )
    with pytest.raises(RuntimeError, match="No non-noise categories"):
        plot_umap_with_category_thumbnails(
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
        plot_umap_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path=str(h5_path),
            tile_orientation="diagonal",
            pooled_df=_make_pooled_df("sessA"),
            message_output=lambda *_: None,
        )


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.filterwarnings("ignore:Glyph .* missing from font:UserWarning")
def test_plot_umap_thumbnails_bad_size_fraction(tmp_path):
    """A thumbnail_size_fraction outside (0, 1] raises ValueError."""
    h5_path = tmp_path / "store.h5"
    _write_consolidated_h5(h5_path, "sessA", n_usvs=12)
    with pytest.raises(ValueError, match="thumbnail_size_fraction"):
        plot_umap_with_category_thumbnails(
            sessions_txt_path="unused",
            consolidated_h5_path=str(h5_path),
            thumbnail_size_fraction=2.0,
            pooled_df=_make_pooled_df("sessA"),
            message_output=lambda *_: None,
        )


def test_bandwidth_split_constant_is_khz():
    """Guard against an accidental unit regression on the bimodal split
    constant (it is in display kHz, not Hz)."""
    assert 0.0 < BANDWIDTH_BIMODAL_SPLIT_KHZ < 200.0
