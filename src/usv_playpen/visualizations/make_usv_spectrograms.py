"""
@author: bartulem
Renders spectrograms for multi-channel USV recordings.

Three rendering modes are exposed by ``USVSpectrogramPlotter``:
(1) ``plot_single_channel`` — one channel's spectrogram, optionally
    stacked above the raw waveform of that channel.
(2) ``plot_all_channels`` — every channel's spectrogram in a single
    vertically stacked figure; the raw waveform of each channel can be
    interleaved above its spectrogram on request.
(3) ``plot_stitched`` — a session-timeline spectrogram built by
    placing the pre-computed `[0, 1]`-normalized per-USV spectrograms
    (from the consolidated HDF5 store) at their on-session start
    times. Gaps between USVs are zero. Resolves the store as the newest
    ``spectrograms_*.h5`` under ``shared_resources['spectrograms_dir']``.

Audio is read from the session's ``*_int16.mmap*`` file (the canonical
concatenated multi-channel int16 memmap), and spectrograms are computed
with ``librosa.stft`` over the user-specified time window. All rendering
parameters live in the ``make_usv_spectrograms`` block of
``visualizations_settings.json``.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

from collections.abc import Callable
from datetime import datetime

import click
import h5py
import librosa
import librosa.display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom
from scipy.signal.windows import tukey
from sklearn.neighbors import KNeighborsClassifier

from ..os_utils import (
    configure_path,
    first_match_or_raise,
    resolve_consolidated_h5_path,
    resolve_embedding_arrays_path,
)
from ..time_utils import is_gui_context, smart_wait
from .plot_style import apply_plot_style

# Register the bundled Helvetica weights + activate the project mplstyle so every
# spectrogram / sequence figure renders in the same font as the torus video.
apply_plot_style()


# Load the project-wide default cmap from `visualizations_settings.json`
# at module import. Used as the default for the module-level
# `plot_embedding_with_category_thumbnails` helper's `cmap=` arguments (was
# two hard-coded `'inferno'` literals). Class instances continue to
# resolve their cmap via `_resolve_cmap` (also reads `figures.cmap`).
# The `figures.cmap` entry is a required field of the packaged settings
# file, so it is read by direct key access; only the file-not-found
# case (e.g. a partial install) falls back to a literal so the module
# can still be imported.
_VIZ_SETTINGS_PATH = (
    pathlib.Path(__file__).parent.parent
    / "_parameter_settings" / "visualizations_settings.json"
)
try:
    with _VIZ_SETTINGS_PATH.open() as _vf:
        _GLOBAL_CMAP = json.load(_vf)["figures"]["cmap"]
except FileNotFoundError:
    _GLOBAL_CMAP = "inferno"

# Fraction of the figure width reserved on the right for the colorbar
# column when ``plot_cbar`` is True; the colorbar itself is anchored to
# the spectrogram axes via ``inset_axes`` so the spectrogram and raw
# audio panels keep identical widths.
CBAR_RIGHT_RESERVE = 0.92
CBAR_INSET_WIDTH = "1.2%"
CBAR_INSET_PAD_AXES_FRACTION = 0.02

# Raised-cosine taper applied along each per-USV spec's time axis
# before stitching, so each USV fades smoothly into the surrounding
# zero canvas instead of abruptly stepping from a non-trivially bright
# edge column. ``alpha`` is the Tukey shape parameter — the fraction
# of the window tapered at the start and end combined. 0.3 means the
# first 15% and last 15% of each spec are raised-cosine-faded.
STITCHED_EDGE_TAPER_ALPHA = 0.3

# Sequence-figure left-panel styling. Marker areas (pt^2) scale modestly with USV
# duration (so longer calls read bigger without dominating); the connecting path's
# per-segment linewidth scales with the inter-USV interval (shorter gap -> thinner)
# within this range; its per-segment color is a white -> emitter-color time
# gradient (white = start of the bout, emitter color towards the end).
_SEQ_MARKER_S_MIN = 35.0
_SEQ_MARKER_S_MAX = 70.0
_SEQ_LW_MIN = 0.5
_SEQ_LW_MAX = 3.0
_SEQ_EMBEDDING_CMAP = "gray_r"


class USVSpectrogramPlotter:
    """
    Description
    -----------
    Renders single-channel, all-channel and variance-weighted average
    USV spectrograms for a session. Audio is loaded from the session's
    concatenated ``*_int16.mmap*`` file as a numpy memmap. Rendering
    parameters (window, NFFT, frequency/amplitude limits, colormap,
    output paths, etc.) are read from
    ``visualizations_parameter_dict['make_usv_spectrograms']``.

    Parameters
    ----------
    root_directory (str)
        Session root directory containing the int16 audio memmap.
    visualizations_parameter_dict (dict)
        Loaded contents of ``visualizations_settings.json``. Must
        include a top-level ``make_usv_spectrograms`` section with all
        of the keys consumed below (the dict is accessed via direct
        bracket lookups — no defaults are applied).
    message_output (Callable)
        Logger callable for status messages; defaults to ``print``.

    Returns
    -------
    None
    """

    def __init__(self, **kwargs) -> None:
        """
        Description
        -----------
        Initialize the USV spectrogram plotter. Stashes every keyword
        argument verbatim on ``self`` (notably ``root_directory``,
        ``visualizations_parameter_dict`` and ``message_output``) and
        records the GUI-vs-CLI context flag so callers can integrate
        the plotter into either the GUI pipeline or a headless script.

        Parameters
        ----------
        **kwargs
            Forwarded as-is to ``self.__dict__``. ``root_directory`` and
            ``visualizations_parameter_dict`` are REQUIRED (a missing one
            raises ``ValueError``); ``message_output`` and
            ``cmap_override`` are optional and default to ``print`` /
            ``None`` respectively.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If either ``root_directory`` or
            ``visualizations_parameter_dict`` was not supplied.
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        # The pass-through above deliberately keeps the constructor open to
        # the optional keys (``message_output``, ``cmap_override``) so callers
        # need not pass them, but the two keys every rendering method reads
        # unconditionally MUST be present. Validate them here so a forgotten
        # keyword fails immediately with a clear message instead of surfacing
        # as an ``AttributeError`` deep inside ``plot_*``.
        for required_kw in ("root_directory", "visualizations_parameter_dict"):
            if not hasattr(self, required_kw):
                msg = (
                    f"USVSpectrogramPlotter requires the {required_kw!r} keyword "
                    f"argument."
                )
                raise ValueError(msg)

        if hasattr(self, "root_directory") and isinstance(self.root_directory, str):
            self.root_directory = configure_path(self.root_directory)

        if not hasattr(self, "message_output") or self.message_output is None:
            self.message_output = print

        if not hasattr(self, "cmap_override"):
            self.cmap_override = None

        self.app_context_bool = is_gui_context()

    def _resolve_cmap(self):
        """
        Description
        -----------
        Resolve the colormap to use for spectrogram rendering. If a
        ``cmap_override`` was passed to the constructor (typically a
        ``matplotlib.colors.Colormap`` instance produced by
        ``visualizations.auxiliary_plot_functions.create_colormap``)
        that takes precedence over the string name in the project-wide
        ``figures.cmap`` settings entry (read by direct key access — it
        is a required field of the settings dict). Either form is
        accepted by matplotlib's ``imshow`` / ``specshow`` ``cmap=``
        argument.

        Returns
        -------
        cmap (str | matplotlib.colors.Colormap)
            The colormap object or name to pass to ``imshow``.
        """

        if getattr(self, "cmap_override", None) is not None:
            return self.cmap_override
        return self.visualizations_parameter_dict["figures"]["cmap"]

    def _load_audio_memmap(self) -> tuple[np.memmap, int, int, int, str]:
        """
        Description
        -----------
        Locate and memory-map the session's concatenated multi-channel
        int16 audio file. The filename encodes the sampling rate,
        sample count and channel count (``*_<sr>_<n_samples>_<n_ch>_int16.mmap*``);
        these are parsed out and used to reshape the memmap.

        Parameters
        ----------

        Returns
        -------
        audio_data (np.memmap)
            ``(sample_num, channel_num)`` int16 memmap with C order.
        sampling_rate (int)
            Audio sampling rate in Hz.
        sample_num (int)
            Total number of audio samples per channel.
        channel_num (int)
            Number of audio channels.
        file_basename (str)
            The basename of the discovered memmap file (used for
            output-file naming).
        """

        audio_loc = first_match_or_raise(
            root=pathlib.Path(self.root_directory),
            pattern="*_int16.mmap*",
            recursive=True,
            label="concatenated int16 audio memmap",
        )
        file_basename = audio_loc.name
        # Parse the sampling-rate / sample-count / channel-count triple out
        # of the trailing ``_<sr>_<n_samples>_<n_ch>_int16.mmap`` segment with
        # a single anchored, keyed regex rather than three positional
        # ``split("_")[-2/-3/-4]`` lookups. The positional form silently
        # mis-parses (or raises an opaque ``ValueError: invalid literal``)
        # the moment any earlier ``_``-delimited token count changes; the
        # anchored regex instead fails loudly with the offending basename.
        meta_match = re.search(
            r"_(?P<sr>\d+)_(?P<n_samples>\d+)_(?P<n_ch>\d+)_int16\.mmap",
            file_basename,
        )
        if meta_match is None:
            msg = (
                f"Cannot parse sampling rate / sample count / channel count "
                f"from audio memmap basename {file_basename!r}; expected a "
                f"trailing '_<sr>_<n_samples>_<n_ch>_int16.mmap' segment."
            )
            raise ValueError(msg)
        sampling_rate = int(meta_match["sr"])
        sample_num = int(meta_match["n_samples"])
        channel_num = int(meta_match["n_ch"])

        audio_data = np.memmap(
            filename=audio_loc,
            dtype=np.int16,
            mode="r",
            shape=(sample_num, channel_num),
            order="C",
        )

        return audio_data, sampling_rate, sample_num, channel_num, file_basename

    def _resolve_window(self, sample_num: int, sampling_rate: int) -> tuple[int, int, float, float]:
        """
        Description
        -----------
        Convert the configured ``time_window`` (``[start, end]`` in
        seconds) into sample indices. An ``end`` of 0 is interpreted as
        "end of recording" and is replaced by ``sample_num /
        sampling_rate``.

        Parameters
        ----------
        sample_num (int)
            Total number of audio samples per channel.
        sampling_rate (int)
            Audio sampling rate in Hz.

        Returns
        -------
        start_signal (int)
            Inclusive start sample index of the analysis window.
        end_signal (int)
            Exclusive end sample index of the analysis window.
        start_time_sec (float)
            Window start in seconds.
        end_time_sec (float)
            Window end in seconds (resolved from 0 if necessary).
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        start_time_sec = float(cfg["time_window"][0])
        end_time_sec = float(cfg["time_window"][1])
        if end_time_sec == 0:
            end_time_sec = sample_num / sampling_rate

        start_signal = round(start_time_sec * sampling_rate)
        end_signal = round(end_time_sec * sampling_rate)

        return start_signal, end_signal, start_time_sec, end_time_sec

    def _compute_magnitude_spectrogram(
        self, audio_segment: np.ndarray, nfft: int
    ) -> np.ndarray:
        """
        Description
        -----------
        Compute the magnitude spectrogram ``|STFT(x)|`` of a 1-D audio
        segment using ``librosa.stft`` with the configured ``nfft`` and
        the librosa default hop ``nfft // 4``. The input is cast to
        float32 for the FFT.

        Parameters
        ----------
        audio_segment (np.ndarray)
            1-D audio samples for the channel/window of interest.
        nfft (int)
            STFT window length (number of FFT points).

        Returns
        -------
        magnitude (np.ndarray)
            ``(n_freq_bins, n_time_frames)`` linear magnitude
            spectrogram.
        """

        stft = librosa.stft(y=audio_segment.astype(np.float32), n_fft=nfft)
        return np.abs(stft)

    def _render_raw_audio(
        self,
        ax: plt.Axes,
        time_vec: np.ndarray,
        audio_segment: np.ndarray,
        color: str,
        title: str,
    ) -> None:
        """
        Description
        -----------
        Render a raw-waveform panel for a single channel/window. The
        x-axis is hidden because the panel is meant to sit immediately
        above the corresponding spectrogram, which carries the shared
        time axis. Y-limits are auto-scaled to the actual min/max of
        ``audio_segment`` and only two y-tick labels are shown (the
        floor and ceiling of the window's amplitude range).

        Parameters
        ----------
        ax (plt.Axes)
            Axes to draw on.
        time_vec (np.ndarray)
            1-D time vector (seconds) aligned to ``audio_segment``.
        audio_segment (np.ndarray)
            1-D audio samples to plot.
        color (str)
            Hex color used for the waveform.
        title (str)
            Panel title.

        Returns
        -------
        None
        """

        ax.minorticks_off()
        ax.plot(time_vec, audio_segment, color=color)
        ax.set_title(title)
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_xticks([])
        ax.set_xticklabels([])

        abs_peak = float(np.max(np.abs(audio_segment)))
        if abs_peak == 0:
            abs_peak = 1.0
        symmetric_limit = abs_peak * 1.05
        ax.set_ylim(-symmetric_limit, symmetric_limit)
        ax.set_yticks([-symmetric_limit, symmetric_limit])
        ax.set_yticklabels([f"{-symmetric_limit:.0f}", f"{symmetric_limit:.0f}"])
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0)

    def _render_spectrogram(
        self,
        ax: plt.Axes,
        fig: plt.Figure,
        magnitude: np.ndarray,
        sampling_rate: int,
        nfft: int,
        start_time_sec: float,
        freq_limits_hz: tuple[float, float],
        cmap: str,
        vmin: float | None,
        vmax: float | None,
        title: str,
        plot_cbar: bool,
    ) -> None:
        """
        Description
        -----------
        Render a single spectrogram panel. The linear magnitude is
        converted to dB (``20·log10`` relative to the panel's own
        maximum) for display. The y-axis shows frequency in kHz with
        labels only at the lower and upper limits. When a colorbar is
        requested it is placed to the right of the spectrogram axes
        (outside, not inset) with tick labels at the colormap limits
        only and no intermediate tick marks.

        Parameters
        ----------
        ax (plt.Axes)
            Axes to draw on.
        fig (plt.Figure)
            Parent figure (needed for colorbar placement).
        magnitude (np.ndarray)
            ``(n_freq, n_time)`` magnitude spectrogram (linear).
        sampling_rate (int)
            Audio sampling rate in Hz.
        nfft (int)
            STFT window length (number of FFT points). ``hop_length``
            is taken as ``nfft // 4``.
        start_time_sec (float)
            Window start in seconds (used to label the x-axis).
        freq_limits_hz (tuple of float)
            Lower / upper frequency limits in Hz.
        cmap (str)
            Matplotlib colormap name.
        vmin (float | None)
            Minimum dB value for color normalization. ``None`` lets
            matplotlib auto-scale.
        vmax (float | None)
            Maximum dB value for color normalization. ``None`` lets
            matplotlib auto-scale.
        title (str)
            Panel title.
        plot_cbar (bool)
            Whether to draw a right-side colorbar.

        Returns
        -------
        None
        """

        display_data = librosa.amplitude_to_db(magnitude, ref=np.max)

        hop_length = nfft // 4
        img = librosa.display.specshow(
            data=display_data,
            sr=sampling_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_ylim(freq_limits_hz[0], freq_limits_hz[1])
        ax.set_yticks([freq_limits_hz[0], freq_limits_hz[1]])
        ax.set_yticklabels(
            [f"{freq_limits_hz[0] / 1000:.0f}", f"{freq_limits_hz[1] / 1000:.0f}"]
        )
        ax.tick_params(axis="y", length=0)

        # Pin the auto-chosen tick positions with a FixedLocator (via
        # ``set_xticks``) BEFORE relabeling them, otherwise matplotlib
        # warns that ``set_xticklabels`` is being used against a non-fixed
        # locator (and, under the test suite's ``filterwarnings=error``,
        # that warning becomes a hard failure). The positions are
        # unchanged; only the labels gain the window-start offset.
        xtick_locs = ax.get_xticks()
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels([f"{xt + start_time_sec:.2f}" for xt in xtick_locs])

        if plot_cbar:
            cbar_vmin = img.get_clim()[0] if vmin is None else vmin
            cbar_vmax = img.get_clim()[1] if vmax is None else vmax
            cax = inset_axes(
                ax,
                width=CBAR_INSET_WIDTH,
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.0 + CBAR_INSET_PAD_AXES_FRACTION, 0.0, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cbar = fig.colorbar(img, cax=cax)
            cbar.set_ticks([cbar_vmin, cbar_vmax])
            cbar.set_ticklabels([f"{cbar_vmin:.0f}", f"{cbar_vmax:.0f}"])
            cbar.ax.tick_params(length=0)
            cbar.ax.minorticks_off()
            cbar.set_label("Amplitude (dB)")

    def _save_figure(self, fig: plt.Figure, suffix: str, file_basename: str) -> None:
        """
        Description
        -----------
        Persist a rendered figure to disk if ``save_fig`` is True.
        The filename includes the audio memmap basename prefix, the
        caller-supplied ``suffix`` (mode / channel info), and the
        time window, ensuring uniqueness across modes and windows.
        An empty ``save_dir`` routes the figure to
        ``<session>/data_animation_examples`` (the per-session
        visualization-output folder also used by the behavioral videos).

        Parameters
        ----------
        fig (plt.Figure)
            Figure to save.
        suffix (str)
            Mode/channel descriptor inserted into the filename
            (e.g. ``"ch03"``, ``"all_channels"``, ``"avg"``).
        file_basename (str)
            Memmap basename, used to derive a prefix for the output
            filename.

        Returns
        -------
        None
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        if not cfg["save_fig"]:
            return
        save_dir = configure_path(cfg["save_dir"]) if cfg["save_dir"] else ""
        if save_dir == "":
            save_dir = str(pathlib.Path(self.root_directory) / "data_animation_examples")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        start_time_sec = float(cfg["time_window"][0])
        end_time_sec = float(cfg["time_window"][1])
        prefix = file_basename.split("_int16.mmap", maxsplit=1)[0]
        stem = f"usv_spectrogram_{prefix}_{suffix}_from_{start_time_sec}s_to_{end_time_sec}s"
        # Honor the global figures.timestamp_in_name (matches figure_io.save_figure:
        # _<YYYYMMDD>_<HHMMSS>), so repeated renders of the same window do not
        # overwrite each other.
        if self.visualizations_parameter_dict["figures"]["timestamp_in_name"]:
            stem = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{stem}.{cfg['fig_format']}"
        out_path = pathlib.Path(save_dir) / filename
        fig.savefig(
            out_path,
            dpi=cfg["fig_dpi"],
            transparent=cfg["transparent_fig_bg"],
        )
        self.message_output(f"Saved spectrogram figure: {out_path}")

        # Open the saved figure in the OS default viewer, but only in an
        # interactive GUI context (``app_context_bool``) so headless / batch
        # runs over many sessions do not spawn viewer windows.
        if cfg["auto_open_figure"] and self.app_context_bool:
            self._open_in_default_viewer(out_path)

    def _open_in_default_viewer(self, path: pathlib.Path) -> None:
        """
        Description
        -----------
        Open a saved file in the operating system's default viewer
        (``open`` on macOS, ``os.startfile`` on Windows, ``xdg-open``
        elsewhere). Best-effort: any failure is reported via
        ``message_output`` rather than raised, so a missing opener never
        aborts a render.

        Parameters
        ----------
        path (pathlib.Path)
            File to open.

        Returns
        -------
        None
        """

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif os.name == "nt":
                os.startfile(str(path))  # noqa: S606 -- Windows-only default opener
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except OSError as exc:
            self.message_output(f"Could not open {path} in a viewer: {exc}")

    def plot_single_channel(self, channel: int | None = None) -> plt.Figure:
        """
        Description
        -----------
        Render a single-channel spectrogram (optionally with the raw
        waveform stacked above it) for the configured time window.

        Parameters
        ----------
        channel (int | None)
            Zero-based channel index to plot. If ``None``, the value of
            ``make_usv_spectrograms.channel_of_interest`` from the
            settings dict is used instead.

        Returns
        -------
        fig (plt.Figure)
            The rendered figure (also written to disk if ``save_fig``
            is True).
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        audio_data, sampling_rate, sample_num, channel_num, file_basename = (
            self._load_audio_memmap()
        )
        start_signal, end_signal, start_time_sec, end_time_sec = self._resolve_window(
            sample_num, sampling_rate
        )

        ch = cfg["channel_of_interest"] if channel is None else channel
        if not 0 <= ch < channel_num:
            msg = f"channel {ch} out of range for {channel_num}-channel recording."
            raise ValueError(msg)

        data_slice = np.asarray(audio_data[start_signal:end_signal, ch])
        time_vec = np.linspace(
            start_time_sec, end_time_sec, num=data_slice.shape[0], endpoint=False
        )

        row_num = 2 if cfg["plot_raw_audio"] else 1
        fig, axes = plt.subplots(
            nrows=row_num, ncols=1, figsize=tuple(cfg["fig_size"]), dpi=cfg["fig_dpi"]
        )
        if row_num == 1:
            axes = [axes]

        if cfg["plot_raw_audio"]:
            self._render_raw_audio(
                ax=axes[0],
                time_vec=time_vec,
                audio_segment=data_slice,
                color=cfg["usv_amplitude_color"],
                title=f"Raw signal (ch{ch:02d})",
            )

        magnitude = self._compute_magnitude_spectrogram(data_slice, cfg["nfft"])
        self._render_spectrogram(
            ax=axes[-1],
            fig=fig,
            magnitude=magnitude,
            sampling_rate=sampling_rate,
            nfft=cfg["nfft"],
            start_time_sec=start_time_sec,
            freq_limits_hz=(
                cfg["freq_limits"][0] * 1000,
                cfg["freq_limits"][1] * 1000,
            ),
            cmap=self._resolve_cmap(),
            vmin=cfg["cbar_limits"][0],
            vmax=cfg["cbar_limits"][1],
            title=f"Spectrogram (ch{ch:02d})",
            plot_cbar=cfg["plot_cbar"],
        )

        layout_rect = (0.0, 0.0, CBAR_RIGHT_RESERVE, 1.0) if cfg["plot_cbar"] else None
        fig.tight_layout(rect=layout_rect)
        self._save_figure(fig, f"ch{ch:02d}", file_basename)
        return fig

    def plot_all_channels(self) -> plt.Figure:
        """
        Description
        -----------
        Render every channel's spectrogram in a single vertically
        stacked figure. When ``plot_raw_audio`` is True, each channel
        contributes two rows (raw waveform above its spectrogram); the
        figure height scales with channel count so that per-channel
        panel height remains roughly constant.

        Parameters
        ----------

        Returns
        -------
        fig (plt.Figure)
            The rendered figure (also written to disk if ``save_fig``
            is True).
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        audio_data, sampling_rate, sample_num, channel_num, file_basename = (
            self._load_audio_memmap()
        )
        start_signal, end_signal, start_time_sec, end_time_sec = self._resolve_window(
            sample_num, sampling_rate
        )

        rows_per_channel = 2 if cfg["plot_raw_audio"] else 1
        total_rows = rows_per_channel * channel_num
        per_row_height = float(cfg["fig_size"][1])
        figsize = (cfg["fig_size"][0], per_row_height * total_rows / max(1, rows_per_channel))

        fig, axes = plt.subplots(
            nrows=total_rows, ncols=1, figsize=figsize, dpi=cfg["fig_dpi"]
        )
        if total_rows == 1:
            axes = [axes]
        axes = list(np.atleast_1d(axes))

        time_vec = np.linspace(
            start_time_sec,
            end_time_sec,
            num=end_signal - start_signal,
            endpoint=False,
        )

        for ch in range(channel_num):
            data_slice = np.asarray(audio_data[start_signal:end_signal, ch])
            row_idx_start = ch * rows_per_channel

            if cfg["plot_raw_audio"]:
                self._render_raw_audio(
                    ax=axes[row_idx_start],
                    time_vec=time_vec,
                    audio_segment=data_slice,
                    color=cfg["usv_amplitude_color"],
                    title=f"Raw signal (ch{ch:02d})",
                )

            magnitude = self._compute_magnitude_spectrogram(data_slice, cfg["nfft"])
            self._render_spectrogram(
                ax=axes[row_idx_start + (1 if cfg["plot_raw_audio"] else 0)],
                fig=fig,
                magnitude=magnitude,
                sampling_rate=sampling_rate,
                nfft=cfg["nfft"],
                start_time_sec=start_time_sec,
                freq_limits_hz=(
                    cfg["freq_limits"][0] * 1000,
                    cfg["freq_limits"][1] * 1000,
                ),
                cmap=self._resolve_cmap(),
                vmin=cfg["cbar_limits"][0],
                vmax=cfg["cbar_limits"][1],
                title=f"Spectrogram (ch{ch:02d})",
                plot_cbar=cfg["plot_cbar"],
            )

        layout_rect = (0.0, 0.0, CBAR_RIGHT_RESERVE, 1.0) if cfg["plot_cbar"] else None
        fig.tight_layout(rect=layout_rect)
        self._save_figure(fig, "all_channels", file_basename)
        return fig

    def _build_stitched_canvas(
        self,
        sampling_rate: int,
        start_time_sec: float,
        end_time_sec: float,
        session_key: str,
        in_window_df: pls.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Description
        -----------
        Build the continuous stitched-spectrogram canvas for one session over
        ``[start_time_sec, end_time_sec]``. A zero (black) canvas is filled by
        stamping each in-window USV's `[0, 1]`-normalized averaged spectrogram
        (from the consolidated store) at its true on-session time: the spec is
        SAM2-masked (when ``apply_mask``), edge-tapered, linearly resampled to its
        on-session duration in canvas bins, and blended into the canvas with
        ``np.maximum``. Inter-call gaps and masked-out regions stay zero (render
        black). The freq axis is cropped to ``cfg['freq_limits']`` (kHz).

        Parameters
        ----------
        sampling_rate (int)
            Audio sampling rate in Hz (sets the canvas frame rate via the hop).
        start_time_sec (float)
            Window start in seconds.
        end_time_sec (float)
            Window end in seconds.
        session_key (str)
            Consolidated-store session group name (root basename).
        in_window_df (pls.DataFrame)
            Rows for the in-window USVs; must carry ``row_index`` / ``start`` /
            ``stop`` columns (``row_index`` = spectrogram index in the store).

        Returns
        -------
        canvas_cropped (np.ndarray)
            ``(n_freq_cropped, canvas_n_bins)`` float32 stitched canvas.
        freq_bins_cropped (np.ndarray)
            The cropped frequency axis (Hz).

        Raises
        ------
        KeyError
            If the session group is absent from the store.
        ValueError
            If ``freq_limits`` selects no frequency bins from the store.
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        hop_length = cfg["nfft"] // 4
        canvas_fps = float(sampling_rate) / float(hop_length)
        canvas_n_bins = max(1, int(round((end_time_sec - start_time_sec) * canvas_fps)))

        h5_path = resolve_consolidated_h5_path(
            self.visualizations_parameter_dict["shared_resources"]["spectrograms_dir"]
        )
        with h5py.File(h5_path, "r") as h5:
            freq_bins = h5["frequency_bins"][:]
            sess_group_key = f"spectrogram/{session_key}"
            if sess_group_key not in h5:
                msg = (
                    f"Session {session_key!r} not present in "
                    f"{h5_path!r}; expected group '{sess_group_key}'."
                )
                raise KeyError(msg)
            sess_specs = h5[sess_group_key]["spectrograms"]
            sess_durations = h5[sess_group_key]["durations"][:]
            n_freq = sess_specs.shape[1]
            n_time_in_spec = sess_specs.shape[2]
            canvas = np.zeros((n_freq, canvas_n_bins), dtype=np.float32)

            apply_mask = bool(cfg["apply_mask"])
            mask_group_key = f"mask/{session_key}"
            if apply_mask and mask_group_key in h5:
                all_segmentations = h5[mask_group_key]["segmentations"][:]
                all_mask_spec_indices = h5[mask_group_key]["spectrogram_index"][:]
            else:
                all_segmentations = None
                all_mask_spec_indices = None

            for row in in_window_df.iter_rows(named=True):
                spec_idx = int(row["row_index"])
                usv_start = float(row["start"])
                usv_stop = float(row["stop"])
                n_target = max(
                    1, int(round((usv_stop - usv_start) * canvas_fps))
                )
                spec = sess_specs[spec_idx, :, :].astype(np.float32)
                valid_cols = max(1, min(int(sess_durations[spec_idx]), n_time_in_spec))
                spec_valid = spec[:, :valid_cols]

                if all_segmentations is not None:
                    mask_rows = np.where(all_mask_spec_indices == spec_idx)[0]
                    if mask_rows.size > 0:
                        combined_mask = np.any(
                            all_segmentations[mask_rows, :, :valid_cols], axis=0
                        )
                        spec_valid = spec_valid * combined_mask.astype(np.float32)

                if valid_cols >= 2:
                    edge_taper = tukey(valid_cols, alpha=STITCHED_EDGE_TAPER_ALPHA).astype(np.float32)
                    spec_valid = spec_valid * edge_taper[None, :]
                zoom_factor_time = n_target / float(spec_valid.shape[1])
                spec_resampled = zoom(spec_valid, (1.0, zoom_factor_time), order=1)
                tile_peak = float(spec_resampled.max())
                if tile_peak > 0.0:
                    spec_resampled = spec_resampled / tile_peak
                x_start = int(round((usv_start - start_time_sec) * canvas_fps))
                x_end = x_start + spec_resampled.shape[1]
                x_clip_start = max(0, x_start)
                x_clip_end = min(canvas_n_bins, x_end)
                if x_clip_start >= x_clip_end:
                    continue
                src_start = x_clip_start - x_start
                src_end = src_start + (x_clip_end - x_clip_start)
                canvas[:, x_clip_start:x_clip_end] = np.maximum(
                    canvas[:, x_clip_start:x_clip_end],
                    spec_resampled[:, src_start:src_end],
                )

        freq_lo_hz = cfg["freq_limits"][0] * 1000.0
        freq_hi_hz = cfg["freq_limits"][1] * 1000.0
        freq_mask = (freq_bins >= freq_lo_hz) & (freq_bins <= freq_hi_hz)
        canvas_cropped = canvas[freq_mask, :]
        freq_bins_cropped = freq_bins[freq_mask]
        # The store's freq axis is fixed (~30-120 kHz). A user-supplied
        # ``freq_limits`` entirely outside that range would silently
        # produce an empty crop and the imshow / set_yticks calls
        # below would then raise an unhelpful IndexError. Surface the
        # real cause here.
        if freq_bins_cropped.size == 0:
            msg = (
                f"freq_limits {tuple(cfg['freq_limits'])!r} kHz selects no "
                f"frequency bins from the consolidated store; the store's "
                f"freq axis spans {freq_bins[0] / 1000:.1f}-{freq_bins[-1] / 1000:.1f} kHz"
            )
            raise ValueError(msg)
        return canvas_cropped, freq_bins_cropped

    def _render_stitched_canvas(
        self,
        ax: plt.Axes,
        fig: plt.Figure,
        canvas_cropped: np.ndarray,
        freq_bins_cropped: np.ndarray,
        start_time_sec: float,
        end_time_sec: float,
        title: str,
        plot_cbar: bool,
    ) -> None:
        """
        Description
        -----------
        Render a prebuilt stitched canvas onto ``ax`` as a linear `[0, 1]`
        normalized-amplitude image spanning ``[start_time_sec, end_time_sec]`` by
        the cropped frequency range, with kHz y-labels at the limits and
        (optionally) a right-side `[0, 1]` colorbar labelled "Normalized
        amplitude". The axes facecolor is set black so inter-call gaps read as
        black (the image covers the data area, so this only affects any uncovered
        margin).

        Parameters
        ----------
        ax (plt.Axes)
            Axes to draw on.
        fig (plt.Figure)
            Parent figure (colorbar placement).
        canvas_cropped (np.ndarray)
            The stitched canvas (freq by time).
        freq_bins_cropped (np.ndarray)
            Cropped frequency axis (Hz).
        start_time_sec (float)
            Window start in seconds (x extent / label).
        end_time_sec (float)
            Window end in seconds (x extent).
        title (str)
            Panel title.
        plot_cbar (bool)
            Whether to draw the right-side colorbar.

        Returns
        -------
        None
        """

        stitched_vmin = 0.0
        stitched_vmax = 1.0
        ax.set_facecolor("#000000")
        img = ax.imshow(
            canvas_cropped,
            origin="lower",
            aspect="auto",
            cmap=self._resolve_cmap(),
            vmin=stitched_vmin,
            vmax=stitched_vmax,
            extent=(start_time_sec, end_time_sec, freq_bins_cropped[0], freq_bins_cropped[-1]),
        )
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("Time (s)", fontsize=6)
        # Compact y-axis: small tick labels at the band limits and the
        # "Frequency (kHz)" label pulled in close to the panel. NOTE: labelpad is
        # IGNORED under constrained_layout (the engine repositions the label), so
        # the label distance is set explicitly with set_label_coords instead.
        ax.set_ylabel("Frequency (kHz)", fontsize=6)
        ax.set_yticks([freq_bins_cropped[0], freq_bins_cropped[-1]])
        ax.set_yticklabels(
            [
                f"{freq_bins_cropped[0] / 1000:.0f}",
                f"{freq_bins_cropped[-1] / 1000:.0f}",
            ],
            fontsize=7,
        )
        ax.tick_params(axis="y", length=0)
        ax.yaxis.set_label_coords(-0.05, 0.5)

        if plot_cbar:
            # A layout-managed colorbar (constrained_layout sizes/places it next
            # to ``ax``). The previous inset_axes approach floated free and, under
            # tight_layout in the nested-gridspec sequence figure, drew its color
            # block at the wrong figure location.
            cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_ticks([stitched_vmin, stitched_vmax])
            cbar.set_ticklabels([f"{stitched_vmin:.0f}", f"{stitched_vmax:.0f}"])
            cbar.ax.tick_params(length=0)
            cbar.ax.minorticks_off()
            # labelpad is a no-op under constrained_layout (the engine pins the
            # colorbar label at the figure edge); it already sits right next to
            # the "0"/"1" tick labels, so there is nothing useful to tune here.
            cbar.set_label("Norm. amplitude", fontsize=6)

    def plot_stitched(self) -> plt.Figure:
        """
        Description
        -----------
        Render a session-timeline spectrogram by stitching the
        pre-computed per-USV averaged spectrograms from the
        consolidated HDF5 store (the newest ``spectrograms_*.h5`` under
        ``shared_resources['spectrograms_dir']``) into a zero canvas at
        each USV's true on-session time. The store contains, for every
        session, an ``(n_usvs, 128, 128)`` array of `[0, 1]`-normalized
        spectrograms in `/spectrogram/<session>/spectrograms` and a
        shared ``frequency_bins (128,)`` linear axis (~30–120 kHz).
        Each (128, 128) entry is a fixed-size resampled cutout of the
        whole USV; the actual on-session start and stop come from the
        per-session ``*_usv_summary.csv`` (row index in the CSV is the
        spec index in the store). For each USV whose
        ``[start, stop]`` falls inside the configured ``time_window``,
        the spec's 128-bin time axis is linearly resampled to the
        USV's on-session duration in canvas bins and stamped at the
        corresponding column of an `(n_freq, canvas_n_bins)` zero
        canvas. The freq axis is cropped to ``cfg['freq_limits']`` (kHz)
        before display. Because the saved specs are already normalized
        to `[0, 1]` (each spec is internally peak-normalized), the
        canvas is rendered as a *linear* normalized-amplitude image
        with its own fixed `[0, 1]` colorbar (``cfg['cbar_limits']`` is
        ignored for this mode); the colorbar label reads "Normalized
        amplitude" instead of "Amplitude (dB)". Gaps between USVs are
        exactly zero and render as the colormap's lowest color.

        Parameters
        ----------

        Returns
        -------
        fig (plt.Figure)
            The rendered figure (also written to disk if ``save_fig``
            is True).
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        audio_data, sampling_rate, sample_num, channel_num, file_basename = (
            self._load_audio_memmap()
        )
        del audio_data, channel_num
        _, _, start_time_sec, end_time_sec = self._resolve_window(
            sample_num, sampling_rate
        )

        session_key = pathlib.Path(self.root_directory).name

        usv_summary_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory),
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(str(usv_summary_path))
        in_window_df = (
            usv_df.with_row_index(name="row_index")
            .filter(
                (pls.col("start") < end_time_sec) & (pls.col("stop") > start_time_sec)
            )
            .select(["row_index", "start", "stop"])
        )

        canvas_cropped, freq_bins_cropped = self._build_stitched_canvas(
            sampling_rate, start_time_sec, end_time_sec, session_key, in_window_df
        )

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=tuple(cfg["fig_size"]), dpi=cfg["fig_dpi"],
            layout="constrained",
        )
        self._render_stitched_canvas(
            ax, fig, canvas_cropped, freq_bins_cropped, start_time_sec, end_time_sec,
            title=(
                f"Stitched spectrogram ({in_window_df.height} USVs in window, "
                f"session {session_key})"
            ),
            plot_cbar=cfg["plot_cbar"],
        )

        self._save_figure(fig, "stitched", file_basename)
        return fig

    def _draw_embedding_left_map(self, ax: plt.Axes, arrays_npz_path: str, draw_boundaries: bool = True) -> None:
        """
        Description
        -----------
        Draw a precomputed cohort embedding landscape on ``ax``: the density
        ``heatmap`` (gray_r) over its coordinate ``extent`` and — when
        ``draw_boundaries`` is True — the black category lines. Serves BOTH the
        QLVM torus (arrays ``.npz`` with no ``extent`` key → the unit square, also
        used by the torus-traversal video) and the VAE umap (a ``vae_density``
        ``.npz`` carrying its own ``extent``). The boundaries are a
        uniform-thickness neighbour-difference mask, NOT ``ax.contour`` on the
        label field: contour stacks several iso-lines wherever neighbouring region
        labels differ by more than one, which renders as uneven line thickness.

        Parameters
        ----------
        ax (plt.Axes)
            Axes to draw the embedding / boundaries on.
        arrays_npz_path (str)
            Path to the cohort arrays ``.npz`` — keys ``heatmap`` and
            ``ws_labels_periodic`` (the QLVM watershed field or the VAE
            category/supercategory field), plus an optional ``extent``
            ``[x0, x1, y0, y1]`` (absent for QLVM → the unit square).
        draw_boundaries (bool)
            Whether to overlay the black category lines; defaults to True.

        Returns
        -------
        None
        """

        arrays = np.load(configure_path(arrays_npz_path))
        heatmap = arrays["heatmap"]
        extent = tuple(float(v) for v in arrays["extent"]) if "extent" in arrays else (0.0, 1.0, 0.0, 1.0)
        nonzero = heatmap[heatmap > 0]
        vmax = float(np.percentile(nonzero, 95)) if nonzero.size else None
        ax.imshow(
            heatmap, origin="lower", extent=extent,
            cmap=_SEQ_EMBEDDING_CMAP, vmin=0, vmax=vmax, aspect="equal",
        )
        if draw_boundaries:
            labels = arrays["ws_labels_periodic"]
            # A pixel is a boundary iff it differs from any 4-neighbour -> a single
            # uniform-thickness line everywhere, regardless of label difference.
            boundary = np.zeros(labels.shape, dtype=bool)
            boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
            boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
            boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
            boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
            ax.imshow(
                np.where(boundary, 1.0, np.nan), origin="lower", extent=extent,
                cmap=mcolors.ListedColormap(["#000000"]), vmin=0, vmax=1,
                interpolation="nearest", aspect="equal", zorder=2,
            )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_sequence(self) -> plt.Figure:
        """
        Description
        -----------
        Render a per-session "USV sequence" figure for the shared analysis window
        ``time_window`` (``[start, end]`` seconds; the GUI sets it from a start +
        duration pair). LEFT:
        an embedding -- QLVM torus (a periodic [0, 1] density heatmap with black
        watershed category boundaries) or VAE (plain) -- where the window's USVs
        are colored by emitter (male / female / unassigned), sized by call
        duration, numbered ``1..n`` in time order, and joined by a connecting line
        whose color is a white -> male time gradient and whose per-segment width
        tracks the inter-USV silent gap (on the QLVM torus the line takes the short
        wrap-around route across an edge when that is closer). RIGHT: ONE
        continuous spectrogram over the same window -- the per-USV averaged
        spectrograms, SAM2-masked when ``apply_mask``, stitched at their true times
        onto a black background, with an optional raw-audio trace on top
        (``plot_raw_audio``). The left numbering matches the time order of calls
        along the right time axis.

        Parameters
        ----------

        Returns
        -------
        fig (plt.Figure)
            The rendered figure (also written to disk if ``save_fig`` is True).

        Raises
        ------
        ValueError
            If the chosen embedding's coordinate columns are absent from the
            session's USV summary CSV.
        """

        cfg = self.visualizations_parameter_dict["make_usv_spectrograms"]
        seq_cfg = cfg["sequence"]
        shared = self.visualizations_parameter_dict["shared_resources"]

        audio_data, sampling_rate, sample_num, channel_num, file_basename = (
            self._load_audio_memmap()
        )
        # The sequence window is the shared ``time_window`` (start, end); the GUI
        # presents it as start + duration but writes it back here.
        start_signal, end_signal, start_time_sec, end_time_sec = self._resolve_window(
            sample_num, sampling_rate
        )

        session_key = pathlib.Path(self.root_directory).name

        usv_summary_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory),
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(str(usv_summary_path)).with_row_index(name="row_index")

        embedding = seq_cfg["embedding"]
        x_col, y_col = (
            ("vae_umap1", "vae_umap2") if embedding == "vae" else ("qlvm_dim1", "qlvm_dim2")
        )
        if x_col not in usv_df.columns or y_col not in usv_df.columns:
            msg = (
                f"Session {session_key!r} USV summary has no {embedding!r} embedding "
                f"columns ({x_col!r}, {y_col!r}); choose a different 'embedding'."
            )
            raise ValueError(msg)

        male_id, female_id = _resolve_session_emitter_ids(str(self.root_directory))
        usv_df = usv_df.with_columns(
            pls.when(pls.col("emitter") == male_id).then(pls.lit("male"))
            .when(pls.col("emitter") == female_id).then(pls.lit("female"))
            .otherwise(pls.lit("unassigned"))
            .alias("sex")
        )

        seq_df = usv_df.filter(
            (pls.col("start") < end_time_sec) & (pls.col("stop") > start_time_sec)
        ).sort("start")

        sex_colors = {
            "male": self.visualizations_parameter_dict["male_colors"][0],
            "female": self.visualizations_parameter_dict["female_colors"][0],
            "unassigned": self.visualizations_parameter_dict["unassigned_colors"][0],
        }
        plot_raw_audio = bool(cfg["plot_raw_audio"])

        # constrained_layout manages spacing + the colorbar robustly (the old
        # tight_layout + inset_axes combination misplaced the colorbar).
        fig = plt.figure(figsize=tuple(cfg["fig_size"]), dpi=cfg["fig_dpi"], layout="constrained")
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.4], figure=fig)
        ax_left = fig.add_subplot(outer[0, 0])

        # # # # LEFT: precomputed cohort embedding landscape (density + category
        # boundaries), resolved by convention from the shared spectrograms dir --
        # QLVM -> <dir>/qlvm/arrays_{coarse,fine}.npz, VAE ->
        # <dir>/vae/vae_density_{coarse,fine}.npz; boundary_clustering picks
        # coarse/fine. If the npz is not present (e.g. the VAE density was never
        # precomputed) fall back to bare axes, but still strip ticks so the panel
        # never shows ticks/ticklabels.
        arrays_path = resolve_embedding_arrays_path(
            shared["spectrograms_dir"], embedding, seq_cfg["boundary_clustering"]
        )
        if pathlib.Path(arrays_path).exists():
            self._draw_embedding_left_map(ax_left, arrays_path, draw_boundaries=bool(seq_cfg["draw_boundaries"]))
        else:
            ax_left.set_xticks([])
            ax_left.set_yticks([])

        # Window USVs: emitter-colored points sized by duration, numbered 1..n in
        # time order (number INSIDE the point, black), connected by a time-gradient
        # line whose per-segment width tracks the inter-USV interval.
        seq_rows = list(enumerate(seq_df.iter_rows(named=True), start=1))
        durations = [float(row["stop"]) - float(row["start"]) for _, row in seq_rows]
        dur_lo, dur_hi = (min(durations), max(durations)) if durations else (0.0, 0.0)

        def _marker_size(duration: float) -> float:
            """Map a USV duration to a modest marker area (pt^2); equal/zero spread
            falls back to the midpoint so markers never blow up."""
            if dur_hi <= dur_lo:
                return 0.5 * (_SEQ_MARKER_S_MIN + _SEQ_MARKER_S_MAX)
            frac = (duration - dur_lo) / (dur_hi - dur_lo)
            return _SEQ_MARKER_S_MIN + frac * (_SEQ_MARKER_S_MAX - _SEQ_MARKER_S_MIN)

        path_pts: list[tuple[float, float]] = []
        path_starts: list[float] = []
        path_stops: list[float] = []
        for (seq_number, row), duration in zip(seq_rows, durations, strict=True):
            x_val = row[x_col]
            y_val = row[y_col]
            if x_val is None or y_val is None:
                continue
            path_pts.append((float(x_val), float(y_val)))
            path_starts.append(float(row["start"]))
            path_stops.append(float(row["stop"]))
            ax_left.scatter(
                [x_val], [y_val], s=_marker_size(duration), c=sex_colors[row["sex"]],
                edgecolors="#000000", linewidths=0.6, zorder=3,
            )
            ax_left.annotate(
                str(seq_number), (x_val, y_val),
                ha="center", va="center", fontsize=4.5, color="#000000", zorder=4,
            )

        # Connecting line through the sequence: per-segment COLOR is a white ->
        # male-color time gradient (white = start of the bout, emitter color toward
        # the end), per-segment WIDTH tracks the inter-USV silent gap -- the time
        # between the end of one call and the start of the next
        # (start(next) - stop(prev)); a longer silence -> thicker line.
        if len(path_pts) >= 2:
            n_seg = len(path_pts) - 1
            male_color = self.visualizations_parameter_dict["male_colors"][0]
            path_cmap = mcolors.LinearSegmentedColormap.from_list(
                "white_to_male", ["#FFFFFF", male_color]
            )
            # Overlapping calls (stop > next start) clamp to a zero gap.
            gaps = [max(0.0, path_starts[i + 1] - path_stops[i]) for i in range(n_seg)]
            # Robust width scale: cap the top of the range at the Tukey upper
            # outlier fence (Q3 + 1.5*IQR), clamped to the real max, so one very
            # long silence saturates at the max width instead of stretching the
            # scale and squashing every other segment to the minimum width.
            gap_lo = min(gaps)
            q1, q3 = float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))
            gap_hi = max(gap_lo, min(max(gaps), q3 + 1.5 * (q3 - q1)))

            def _segment_lw(gap: float) -> float:
                if gap_hi <= gap_lo:
                    return 0.5 * (_SEQ_LW_MIN + _SEQ_LW_MAX)
                frac = min(1.0, max(0.0, (gap - gap_lo) / (gap_hi - gap_lo)))
                return _SEQ_LW_MIN + frac * (_SEQ_LW_MAX - _SEQ_LW_MIN)

            # QLVM is a periodic unit torus ([0,1] in both dims), so the shortest
            # route between two points may wrap across an edge rather than run
            # straight over the map. For each pair pick the nearest periodic image
            # of the second point (wrap an axis whose gap exceeds 0.5) and emit two
            # collinear sub-segments -- p1 -> image, and the mirror (p1's image) ->
            # p2 -- both clipped to the [0,1] axes so the geodesic exits one edge
            # and re-enters the opposite. VAE is a plain plane: no wrapping.
            is_torus = embedding == "qlvm"

            def _wrap_offset(delta: float) -> float:
                if delta > 0.5:
                    return -1.0
                if delta < -0.5:
                    return 1.0
                return 0.0

            def _sub_segments(i: int) -> list[list[tuple[float, float]]]:
                (x1, y1), (x2, y2) = path_pts[i], path_pts[i + 1]
                mx = _wrap_offset(x2 - x1) if is_torus else 0.0
                my = _wrap_offset(y2 - y1) if is_torus else 0.0
                subs = [[(x1, y1), (x2 + mx, y2 + my)]]
                if mx or my:
                    subs.append([(x1 - mx, y1 - my), (x2, y2)])
                return subs

            segments: list[list[tuple[float, float]]] = []
            seg_colors = []
            seg_lws = []
            for i in range(n_seg):
                color = path_cmap(i / (n_seg - 1) if n_seg > 1 else 1.0)
                lw = _segment_lw(gaps[i])
                for sub in _sub_segments(i):
                    segments.append(sub)
                    seg_colors.append(color)
                    seg_lws.append(lw)
            ax_left.add_collection(
                LineCollection(segments, colors=seg_colors, linewidths=seg_lws, zorder=2)
            )

        # # # # RIGHT: ONE continuous averaged spectrogram over the window. The
        # per-USV averaged specs are stitched at their true times onto a black
        # background; the spectrogram keeps its full width but is made ~1/3 shorter
        # vertically via a bottom spacer row (raw audio, when shown, sits above it).
        in_window_df = seq_df.select(["row_index", "start", "stop"])
        canvas_cropped, freq_bins_cropped = self._build_stitched_canvas(
            sampling_rate, start_time_sec, end_time_sec, session_key, in_window_df
        )
        if plot_raw_audio:
            right_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer[0, 1], height_ratios=[3, 8, 4],
            )
            ax_raw = fig.add_subplot(right_gs[0, 0])
            ax_right = fig.add_subplot(right_gs[1, 0])
            # Auto-pick the channel that is loudest for the most USVs in the
            # window (per-USV ``peak_amp_ch``, the 0-indexed argmax channel from
            # das_inference). Averaging raw waveforms across mics is avoided on
            # purpose: the per-mic phase delays make the sum interfere
            # destructively. Falls back to channel_of_interest if the column is
            # absent or the window has no USVs.
            raw_ch = cfg["channel_of_interest"]
            if "peak_amp_ch" in seq_df.columns:
                peak_chs = seq_df["peak_amp_ch"].drop_nulls()
                if peak_chs.len() > 0:
                    raw_ch = int(round(float(peak_chs.mode().to_list()[0])))
            raw_ch = raw_ch if 0 <= raw_ch < channel_num else 0
            data_slice = np.asarray(audio_data[start_signal:end_signal, raw_ch])
            time_vec = np.linspace(
                start_time_sec, end_time_sec, num=data_slice.shape[0], endpoint=False
            )
            self._render_raw_audio(
                ax=ax_raw, time_vec=time_vec, audio_segment=data_slice,
                color=cfg["usv_amplitude_color"], title="",
            )
            # No y-axis decorations on the raw strip; the symmetric ylim set
            # by _render_raw_audio (+/- peak) keeps the peak from being clipped.
            ax_raw.set_ylabel("")
            ax_raw.set_yticks([])
        else:
            right_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[0, 1], height_ratios=[2, 1],
            )
            ax_right = fig.add_subplot(right_gs[0, 0])
        self._render_stitched_canvas(
            ax_right, fig, canvas_cropped, freq_bins_cropped,
            start_time_sec, end_time_sec,
            title="",
            plot_cbar=cfg["plot_cbar"],
        )
        if seq_cfg["annotate_right"]:
            for seq_number, row in seq_rows:
                ax_right.annotate(
                    str(seq_number), (float(row["start"]), freq_bins_cropped[-1]),
                    textcoords="offset points", xytext=(0, 2),
                    fontsize=7, color="#FFFFFF", ha="center", va="bottom", zorder=5,
                )
        # Optionally mark each USV with a horizontal emitter-colored bar along
        # the top of the spectrogram, spanning its [start, stop].
        if seq_cfg["mark_usv_segments"]:
            y_lo = float(freq_bins_cropped[0])
            y_hi = float(freq_bins_cropped[-1])
            y_bar = y_hi - 0.03 * (y_hi - y_lo)
            for _seq_number, row in seq_rows:
                ax_right.plot(
                    [float(row["start"]), float(row["stop"])], [y_bar, y_bar],
                    color=sex_colors[row["sex"]], linewidth=3.0,
                    solid_capstyle="butt", zorder=5,
                )

        self._save_figure(fig, f"sequence_{embedding}", file_basename)
        return fig

    def make_usv_spectrograms(self) -> plt.Figure:
        """
        Description
        -----------
        Dispatch to the rendering method named by
        ``make_usv_spectrograms.mode``: ``"single"`` →
        ``plot_single_channel``; ``"all"`` → ``plot_all_channels``;
        ``"stitched"`` → ``plot_stitched``; ``"sequence"`` →
        ``plot_sequence``. This is the entry point that the
        visualization pipeline (``visualize_data.py``) is expected to
        call once a ``make_usv_spectrograms_bool`` toggle has been
        wired through.

        Parameters
        ----------

        Returns
        -------
        fig (plt.Figure)
            The rendered figure (also written to disk if ``save_fig``
            is True).
        """

        self.message_output(
            f"USV spectrogram plotting started at: "
            f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        # Brief pause so the GUI flushes the "started" message before the
        # (blocking) render takes over the thread.
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        mode = self.visualizations_parameter_dict["make_usv_spectrograms"]["mode"]
        if mode == "single":
            fig = self.plot_single_channel()
        elif mode == "all":
            fig = self.plot_all_channels()
        elif mode == "stitched":
            fig = self.plot_stitched()
        elif mode == "sequence":
            fig = self.plot_sequence()
        else:
            msg = (
                f"Unknown make_usv_spectrograms.mode={mode!r}; "
                f"expected one of 'single', 'all', 'stitched', 'sequence'."
            )
            raise ValueError(msg)

        # Close the figure so a per-session pipeline run does not accumulate open
        # figures (matplotlib warns past ~20 and memory grows); it has already
        # been saved to disk (and optionally opened) by the plot_* method.
        plt.close(fig)
        return fig


# Display-unit conversions for the pooled-histogram function below:
# CSV-native units are seconds / Hz / raw a.u.; histograms are shown in
# ms / kHz / a.u. for readability. Mirrors the convention used elsewhere
# in the repo (e.g. ``DISPLAY_FACTOR`` in ``make_neuronal_tuning_figures``).
USV_PROPERTY_DISPLAY = {
    "duration": {
        "bounds_native": (0.0, 0.6),
        "display_factor": 1000.0,
        "label": "Duration (ms)",
        "annotation_fmt": "{:.0f}",
    },
    "mean_amplitude": {
        "bounds_native": (0.0, 2.5),
        "display_factor": 1.0,
        "label": "Mean amplitude (a.u.)",
        "annotation_fmt": "{:.2f}",
    },
    "mean_freq_hz": {
        "bounds_native": (30000.0, 120000.0),
        "display_factor": 1.0 / 1000.0,
        "label": "Mean frequency (kHz)",
        "annotation_fmt": "{:.1f}",
    },
    "freq_bandwidth_hz": {
        "bounds_native": (0.0, 90000.0),
        "display_factor": 1.0 / 1000.0,
        "label": "Frequency bandwidth (kHz)",
        "annotation_fmt": "{:.1f}",
    },
    "spectral_entropy": {
        "bounds_native": (0.0, 5.4),
        "display_factor": 1.0,
        "label": "Spectral entropy (nats)",
        "annotation_fmt": "{:.2f}",
    },
}

# x-axis split point (in display units, kHz) used to find the two
# bimodal peaks of the ``freq_bandwidth_hz`` distribution: argmax of
# the (Gaussian-smoothed) histogram on each side of this split is
# annotated. Smoothing sigma is in *bins*, not kHz, and is just large
# enough to suppress sample-noise peaks near the split.
BANDWIDTH_BIMODAL_SPLIT_KHZ = 30.0
BANDWIDTH_PEAK_SMOOTH_SIGMA_BINS = 3.0

HISTOGRAM_FACE_COLOR = "#808080"
HISTOGRAM_EDGE_COLOR = "#000000"
HISTOGRAM_N_BINS = 36

# Session-type bar-chart colors. Lone-male uses the project's male
# color, female-female the female color, and male-female a desaturated
# midpoint between them (50/50 male+female RGB blended 35% toward white).
SESSION_TYPE_MALE_COLOR = "#9AC0CD"
SESSION_TYPE_FEMALE_COLOR = "#FF6347"
SESSION_TYPE_MALE_FEMALE_COLOR = "#DFB8B3"


def plot_usv_property_histograms(
    sessions_txt_path: str,
    output_path: str | None = None,
    fig_format: str | None = None,
    noise_col_id: str = "vae_supercategory",
    noise_categories: tuple[int, ...] = (0,),
    fig_size: tuple[float, float] = (15.0, 3.0),
    fig_dpi: int = 300,
    message_output: Callable | None = None,
) -> plt.Figure:
    """
    Description
    -----------
    Render a row of five histograms summarising per-USV properties
    pooled across every session listed in ``sessions_txt_path``. The
    properties shown, in panel order, are: ``duration``,
    ``mean_amplitude``, ``mean_freq_hz``, ``freq_bandwidth_hz``,
    ``spectral_entropy``. Each session's ``*_usv_summary.csv`` is
    discovered automatically by recursive glob under the session root
    path (each line of the txt file is run through
    ``os_utils.configure_path`` first so paths that were written for a
    different OS / mountpoint still resolve here). Rows whose
    ``noise_col_id`` value is in ``noise_categories`` are dropped
    before pooling so the histograms reflect the non-noise
    distribution.

    Each panel uses 36 linearly-spaced bins over the FeatureZoo
    theoretical range for that property (converted to display units —
    ms / kHz / a.u.) and is rendered with ``histtype='stepfilled'``,
    ``color="#808080"`` (fill), ``edgecolor="#000000"`` (outline).
    Top / right spines are removed for visual cleanliness.

    Parameters
    ----------
    sessions_txt_path (str)
        Path to a text file with one session root path per line. Blank
        lines and lines starting with ``"#"`` are skipped. Path itself
        is run through ``configure_path``, as is every session path
        read from inside the file.
    output_path (str | None)
        Optional path at which to write the figure. Run through
        ``configure_path`` before save. If ``None``, the figure is
        only returned.
    fig_format (str | None)
        Optional output extension (e.g. ``"svg"``, ``"pdf"``,
        ``"png"``). When given, the file is saved with this extension
        regardless of any extension on ``output_path``. When ``None``,
        the extension is taken from ``output_path`` (matplotlib's
        default inference).
    noise_col_id (str)
        CSV column used to identify noise rows; default
        ``"vae_supercategory"``.
    noise_categories (tuple of int)
        Values of ``noise_col_id`` to drop as noise; default ``(0,)``.
    fig_size (tuple of float)
        Figure size in inches; default ``(15, 3)`` for a 5-wide row.
    fig_dpi (int)
        Figure DPI; default ``300``.
    message_output (Callable | None)
        Optional logger callable; defaults to ``print``. Used to
        report per-session discovery / load issues without raising.

    Returns
    -------
    fig (plt.Figure)
        The rendered figure (also written to disk if ``output_path``
        is given).
    """

    if message_output is None:
        message_output = print

    sessions_txt_path = configure_path(sessions_txt_path)
    with open(sessions_txt_path, "r") as txt_file:
        session_roots = [
            configure_path(stripped)
            for stripped in (line.strip() for line in txt_file)
            if stripped and not stripped.startswith("#")
        ]

    properties_in_csv = list(USV_PROPERTY_DISPLAY.keys())
    pooled: dict[str, list[np.ndarray]] = {p: [] for p in properties_in_csv}
    n_sessions_loaded = 0
    n_usvs_total = 0

    columns_to_read = list(set(properties_in_csv + [noise_col_id]))
    for session_root in session_roots:
        try:
            csv_path = first_match_or_raise(
                root=pathlib.Path(session_root) / "audio",
                pattern="*_usv_summary.csv",
                recursive=False,
                label="USV summary CSV",
            )
        except (FileNotFoundError, OSError) as exc:
            message_output(f"[skip] {session_root}: {exc}")
            continue
        try:
            df = pls.read_csv(str(csv_path), columns=columns_to_read)
        except pls.exceptions.ColumnNotFoundError:
            df = pls.read_csv(str(csv_path))
        except (OSError, IOError) as exc:
            message_output(f"[skip] {csv_path}: {exc}")
            continue
        if noise_col_id in df.columns and noise_categories:
            df = df.filter(~pls.col(noise_col_id).is_in(list(noise_categories)))
        if df.height == 0:
            continue
        n_sessions_loaded += 1
        n_usvs_total += df.height
        for p in properties_in_csv:
            if p in df.columns:
                pooled[p].append(df[p].drop_nulls().to_numpy())

    pooled_arrays: dict[str, np.ndarray] = {
        p: (np.concatenate(arrs) if arrs else np.array([], dtype=float))
        for p, arrs in pooled.items()
    }

    fig, axes = plt.subplots(
        nrows=1, ncols=len(properties_in_csv), figsize=fig_size, dpi=fig_dpi
    )
    for ax_idx, (ax, p) in enumerate(zip(axes, properties_in_csv)):
        spec = USV_PROPERTY_DISPLAY[p]
        lo_native, hi_native = spec["bounds_native"]
        factor = spec["display_factor"]
        lo_disp = lo_native * factor
        hi_disp = hi_native * factor
        fmt = spec["annotation_fmt"]

        values_native = pooled_arrays[p].astype(float, copy=True)
        out_of_range = (values_native < lo_native) | (values_native > hi_native)
        values_native[out_of_range] = np.nan
        values_disp = values_native[np.isfinite(values_native)] * factor

        bins = np.linspace(lo_disp, hi_disp, HISTOGRAM_N_BINS + 1)
        hist_counts, hist_edges = np.histogram(values_disp, bins=bins)
        ax.hist(
            values_disp,
            bins=bins,
            histtype="stepfilled",
            color=HISTOGRAM_FACE_COLOR,
            edgecolor=HISTOGRAM_EDGE_COLOR,
            linewidth=0.6,
        )
        ax.set_xlabel(spec["label"], fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("USV count", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_xlim(lo_disp, hi_disp)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.minorticks_off()
        ax.set_xticks(np.linspace(lo_disp, hi_disp, 4))
        hist_top = float(ax.get_ylim()[1])
        ax.set_yticks(np.linspace(0.0, hist_top, 4))
        ax.tick_params(axis="both", length=0, labelsize=8)

        if hist_top > 0 and values_disp.size > 0:
            marker_y = hist_top * 1.05
            label_y = hist_top * 1.12
            ax.set_ylim(0, hist_top * 1.22)
            if p == "freq_bandwidth_hz":
                bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
                smoothed_counts = gaussian_filter1d(
                    hist_counts.astype(float), sigma=BANDWIDTH_PEAK_SMOOTH_SIGMA_BINS
                )
                low_side = bin_centers < BANDWIDTH_BIMODAL_SPLIT_KHZ
                high_side = bin_centers >= BANDWIDTH_BIMODAL_SPLIT_KHZ
                peak_xs = []
                if low_side.any() and smoothed_counts[low_side].max() > 0:
                    peak_xs.append(
                        float(bin_centers[low_side][np.argmax(smoothed_counts[low_side])])
                    )
                if high_side.any() and smoothed_counts[high_side].max() > 0:
                    peak_xs.append(
                        float(bin_centers[high_side][np.argmax(smoothed_counts[high_side])])
                    )
                for peak_x in peak_xs:
                    ax.scatter(
                        peak_x, marker_y, marker="v", color="#000000",
                        s=22, clip_on=False, zorder=5,
                    )
                    ax.text(
                        peak_x, label_y, fmt.format(peak_x),
                        ha="center", va="bottom", fontsize=8, color="#000000",
                    )
            else:
                mean_x = float(np.mean(values_disp))
                ax.scatter(
                    mean_x, marker_y, marker="v", color="#000000",
                    s=22, clip_on=False, zorder=5,
                )
                ax.text(
                    mean_x, label_y, fmt.format(mean_x),
                    ha="center", va="bottom", fontsize=8, color="#000000",
                )

    fig.suptitle(
        f"Pooled USV property histograms "
        f"(n_sessions={n_sessions_loaded}, n_vocalizations={n_usvs_total})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if output_path is not None:
        out_path = pathlib.Path(configure_path(output_path))
        if fig_format is not None:
            out_path = out_path.with_suffix(f".{fig_format.lstrip('.')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        message_output(f"Saved histograms figure: {out_path}")

    return fig


def _count_usvs_per_session(
    sessions_txt_path: str,
    noise_col_id: str,
    noise_categories: tuple[int, ...],
    message_output: Callable,
) -> np.ndarray:
    """
    Description
    -----------
    Helper for ``plot_session_type_usv_counts``. Reads the session list
    at ``sessions_txt_path`` (one path per line, blank/``#`` lines
    skipped, ``configure_path`` applied per line), and for each session
    finds the ``*_usv_summary.csv``, drops noise rows, and returns the
    number of remaining vocalizations. Sessions whose CSV is missing or
    fails to read (e.g. a flaky SMB timeout) are logged via
    ``message_output`` and silently dropped from the returned array.

    Parameters
    ----------
    sessions_txt_path (str)
        Path to a text file listing one session root per line.
    noise_col_id (str)
        Column in the per-session CSV used to flag noise rows.
    noise_categories (tuple of int)
        Values of ``noise_col_id`` that mark noise rows to drop.
    message_output (Callable)
        Logger for per-session load failures.

    Returns
    -------
    counts (np.ndarray)
        1-D array of per-session non-noise USV counts, one entry per
        successfully loaded session.
    """

    sessions_txt_path = configure_path(sessions_txt_path)
    with open(sessions_txt_path, "r") as txt_file:
        session_roots = [
            configure_path(stripped)
            for stripped in (line.strip() for line in txt_file)
            if stripped and not stripped.startswith("#")
        ]

    counts: list[int] = []
    columns_to_read = [noise_col_id] if noise_col_id and noise_categories else None
    for session_root in session_roots:
        try:
            csv_path = first_match_or_raise(
                root=pathlib.Path(session_root) / "audio",
                pattern="*_usv_summary.csv",
                recursive=False,
                label="USV summary CSV",
            )
        except (FileNotFoundError, OSError) as exc:
            message_output(f"[skip] {session_root}: {exc}")
            continue
        try:
            if columns_to_read is not None:
                df = pls.read_csv(str(csv_path), columns=columns_to_read)
            else:
                df = pls.read_csv(str(csv_path))
        except pls.exceptions.ColumnNotFoundError:
            df = pls.read_csv(str(csv_path))
        except (OSError, IOError) as exc:
            message_output(f"[skip] {csv_path}: {exc}")
            continue
        if noise_col_id in df.columns and noise_categories:
            df = df.filter(~pls.col(noise_col_id).is_in(list(noise_categories)))
        counts.append(df.height)

    return np.asarray(counts, dtype=float)


def plot_session_type_usv_counts(
    male_female_txt_path: str,
    female_female_txt_path: str,
    lone_male_txt_path: str,
    output_path: str | None = None,
    fig_format: str | None = None,
    noise_col_id: str = "vae_supercategory",
    noise_categories: tuple[int, ...] = (0,),
    fig_size: tuple[float, float] = (6.0, 3.0),
    fig_dpi: int = 300,
    male_color: str = SESSION_TYPE_MALE_COLOR,
    female_color: str = SESSION_TYPE_FEMALE_COLOR,
    male_female_color: str = SESSION_TYPE_MALE_FEMALE_COLOR,
    message_output: Callable | None = None,
) -> plt.Figure:
    """
    Description
    -----------
    Render a horizontal bar chart comparing the mean number of (non-noise)
    USVs per session across three session types: male-female,
    female-female, and lone-male. Each bar carries a SEM error bar
    (``std / sqrt(n_sessions)``) computed from the per-session counts in
    that type's session list. Bars are color-coded:

    - Lone male       → ``male_color`` (default project male color,
                         ``#9AC0CD``).
    - Female-female   → ``female_color`` (default ``#FF6347``).
    - Male-female     → ``male_female_color`` (default ``#DFB8B3``, a
                         pastel midpoint between male and female RGB
                         blended toward white).

    Each session-list text file is parsed the same way as
    ``plot_usv_property_histograms``: one session root per line,
    ``#`` / blank lines skipped, every path run through
    ``configure_path``. For each session the function discovers the
    ``*_usv_summary.csv``, drops rows whose ``noise_col_id`` value is
    in ``noise_categories``, and counts the remaining rows. Sessions
    whose CSV is unreadable (missing or SMB timeout) are logged and
    excluded from that type's mean / SEM.

    Parameters
    ----------
    male_female_txt_path (str)
        Path to txt file listing male-female session roots.
    female_female_txt_path (str)
        Path to txt file listing female-female session roots.
    lone_male_txt_path (str)
        Path to txt file listing lone-male session roots.
    output_path (str | None)
        Optional path at which to write the figure. Run through
        ``configure_path`` before save.
    fig_format (str | None)
        Optional output extension override (``"svg"``, ``"pdf"``, etc).
        Replaces any extension on ``output_path`` when given.
    noise_col_id (str)
        CSV column used to flag noise rows; default
        ``"vae_supercategory"``.
    noise_categories (tuple of int)
        Values of ``noise_col_id`` to drop as noise; default ``(0,)``.
    fig_size (tuple of float)
        Figure size in inches; default ``(6, 3)``.
    fig_dpi (int)
        Figure DPI; default ``300``.
    male_color (str)
        Hex color for the lone-male bar.
    female_color (str)
        Hex color for the female-female bar.
    male_female_color (str)
        Hex color for the male-female bar.
    message_output (Callable | None)
        Logger callable; defaults to ``print``.

    Returns
    -------
    fig (plt.Figure)
        The rendered figure (also written to disk if ``output_path``
        is given).
    """

    if message_output is None:
        message_output = print

    mf_counts = _count_usvs_per_session(
        male_female_txt_path, noise_col_id, noise_categories, message_output
    )
    ff_counts = _count_usvs_per_session(
        female_female_txt_path, noise_col_id, noise_categories, message_output
    )
    lm_counts = _count_usvs_per_session(
        lone_male_txt_path, noise_col_id, noise_categories, message_output
    )

    def _mean_sem(arr: np.ndarray) -> tuple[float, float, int]:
        n = arr.size
        if n == 0:
            return 0.0, 0.0, 0
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return mean, sem, n

    mf_mean, mf_sem, mf_n = _mean_sem(mf_counts)
    ff_mean, ff_sem, ff_n = _mean_sem(ff_counts)
    lm_mean, lm_sem, lm_n = _mean_sem(lm_counts)

    # Bars stacked top-to-bottom in the order the user listed them:
    # male-female on top, female-female middle, lone-male bottom. With
    # ``barh`` the largest y value is plotted at the top.
    labels = [
        f"Male-female (n={mf_n})",
        f"Female-female (n={ff_n})",
        f"Lone male (n={lm_n})",
    ]
    means = [mf_mean, ff_mean, lm_mean]
    sems = [mf_sem, ff_sem, lm_sem]
    colors = [male_female_color, female_color, male_color]
    y_positions = [2, 1, 0]

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
    ax.barh(
        y_positions,
        means,
        xerr=sems,
        color=colors,
        edgecolor="#000000",
        linewidth=0.6,
        height=1.0,
        error_kw={"ecolor": "#000000", "elinewidth": 1.0, "capsize": 0},
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean USVs per session", fontsize=10)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.tick_params(axis="both", length=0, labelsize=8)

    max_with_error = max(m + s for m, s in zip(means, sems)) if means else 1.0
    ax.set_xlim(0, max_with_error * 1.15)
    ax.set_xticks(np.linspace(0, max_with_error * 1.15, 4))

    fig.tight_layout()

    if output_path is not None:
        out_path = pathlib.Path(configure_path(output_path))
        if fig_format is not None:
            out_path = out_path.with_suffix(f".{fig_format.lstrip('.')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        message_output(f"Saved session-type USV counts figure: {out_path}")

    return fig


# Default colors for the per-session USV timeline. Reuse the project
# male/female palette; "unassigned" uses a neutral grey to read as
# "no emitter identified".
USV_TIMELINE_MALE_COLOR = "#9AC0CD"
USV_TIMELINE_FEMALE_COLOR = "#FF6347"
USV_TIMELINE_UNASSIGNED_COLOR = "#C0C0C0"


def _resolve_session_emitter_ids(session_root: str) -> tuple[str, str]:
    """
    Description
    -----------
    Resolve the male / female emitter id strings from a session's
    ``*_points3d_translated_rotated_metric.h5`` tracking file. The
    h5's ``track_names`` array is the same string that appears in the
    USV summary CSV's ``emitter`` column, so this mapping is what
    drives the male / female / unassigned assignment downstream.
    Convention in this dataset: ``track_names[0]`` is the male and
    ``track_names[1]`` is the female (matches
    ``usv_summary_statistics.extract_session_metadata``).

    Parameters
    ----------
    session_root (str)
        Session root directory path (already ``configure_path``'d by
        the caller).

    Returns
    -------
    male_id, female_id (tuple of str)
        Track-name strings for the male and female animal.
    """

    tracking_file = first_match_or_raise(
        root=pathlib.Path(session_root) / "video",
        pattern="*_points3d_translated_rotated_metric.h5",
        recursive=True,
        label="3D tracking h5",
    )
    with h5py.File(str(tracking_file), "r") as h5_file:
        track_names = [item.decode("utf-8") for item in list(h5_file["track_names"])]
    if len(track_names) < 2:
        msg = (
            f"Session {session_root!r} tracking file lists "
            f"{len(track_names)} animals; need at least two."
        )
        raise ValueError(msg)
    return track_names[0], track_names[1]


def plot_session_usv_timeline(
    session_root: str,
    time_window: tuple[float, float] | None = None,
    output_path: str | None = None,
    fig_format: str | None = None,
    noise_col_id: str = "vae_supercategory",
    noise_categories: tuple[int, ...] = (0,),
    fig_size: tuple[float, float] = (7.5, 1.6),
    fig_dpi: int = 300,
    male_color: str = USV_TIMELINE_MALE_COLOR,
    female_color: str = USV_TIMELINE_FEMALE_COLOR,
    unassigned_color: str = USV_TIMELINE_UNASSIGNED_COLOR,
    rectangle_height: float = 2.5,
    rectangle_linewidth: float = 1.0,
    message_output: Callable | None = None,
) -> plt.Figure:
    """
    Description
    -----------
    Render a single horizontal timeline of every (non-noise) USV in
    ``session_root``, with each call drawn as a rectangle that spans
    its ``[start, stop]`` interval and is colored by its assigned
    emitter: ``male_color`` if the CSV's ``emitter`` matches the
    session's male track id, ``female_color`` if it matches the
    female, otherwise ``unassigned_color``. Three calls to
    ``ax.broken_barh`` (one per group) handle the drawing efficiently
    even for thousands of events.

    Emitter ids are looked up from the session's
    ``*_points3d_translated_rotated_metric.h5`` tracking file (same
    convention as ``usv_summary_statistics.extract_session_metadata``:
    ``track_names[0]`` is male, ``track_names[1]`` is female). The
    USV summary CSV is read non-recursively from ``<session>/audio``
    and rows whose ``noise_col_id`` value is in ``noise_categories``
    are dropped before rendering.

    Parameters
    ----------
    session_root (str)
        Session root path. Run through ``configure_path``.
    time_window (tuple of float | None)
        Optional ``(start_s, end_s)``. When given, only USVs
        overlapping this window are rendered and the x-axis is
        clipped to it. ``None`` shows the entire session.
    output_path (str | None)
        Optional save path. Run through ``configure_path``.
    fig_format (str | None)
        Optional extension override (``"svg"``, ``"pdf"``, ...).
    noise_col_id (str)
        CSV column used to identify noise; default
        ``"vae_supercategory"``.
    noise_categories (tuple of int)
        Values of ``noise_col_id`` to drop; default ``(0,)``.
    fig_size (tuple of float)
        Figure size in inches; default ``(7.5, 1.6)`` for a wide,
        short timeline strip.
    fig_dpi (int)
        Figure DPI; default ``300``.
    male_color / female_color / unassigned_color (str)
        Per-emitter rectangle colors.
    rectangle_height (float)
        Vertical extent of each USV rectangle (axes-data units; the
        y-axis is hidden so the absolute value mainly controls the
        strip's visual thickness).
    rectangle_linewidth (float)
        Outline width of each USV rectangle in points.
    message_output (Callable | None)
        Logger; defaults to ``print``.

    Returns
    -------
    fig (plt.Figure)
        The rendered figure (also written to disk if ``output_path``
        is given).
    """

    if message_output is None:
        message_output = print

    session_root = configure_path(session_root)
    session_path = pathlib.Path(session_root)
    session_id = session_path.name

    male_id, female_id = _resolve_session_emitter_ids(session_root)

    csv_path = first_match_or_raise(
        root=session_path / "audio",
        pattern="*_usv_summary.csv",
        recursive=False,
        label="USV summary CSV",
    )
    df = pls.read_csv(str(csv_path))
    if noise_col_id in df.columns and noise_categories:
        df = df.filter(~pls.col(noise_col_id).is_in(list(noise_categories)))

    df = df.with_columns(
        pls.when(pls.col("emitter") == male_id).then(pls.lit("male"))
        .when(pls.col("emitter") == female_id).then(pls.lit("female"))
        .otherwise(pls.lit("unassigned"))
        .alias("sex")
    )

    if time_window is not None:
        win_lo, win_hi = float(time_window[0]), float(time_window[1])
        df = df.filter((pls.col("stop") > win_lo) & (pls.col("start") < win_hi))
        x_lo, x_hi = win_lo, win_hi
    else:
        if df.height > 0:
            x_lo = 0.0
            x_hi = float(df["stop"].max())
        else:
            x_lo, x_hi = 0.0, 1.0

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
    y_center = 0.0
    y_bottom = y_center - rectangle_height / 2.0

    group_colors = {
        "male": male_color,
        "female": female_color,
        "unassigned": unassigned_color,
    }
    group_counts: dict[str, int] = {
        sex_label: int(df.filter(pls.col("sex") == sex_label).height)
        for sex_label in group_colors
    }

    # Draw smallest group first so the most populous group ends up on
    # top — otherwise sparse groups overwrite dense ones pixel-by-pixel
    # at this horizontal scale and visually exaggerate their prevalence.
    draw_order = sorted(group_colors.keys(), key=lambda lab: group_counts[lab])
    for sex_label in draw_order:
        sub = df.filter(pls.col("sex") == sex_label)
        if sub.height == 0:
            continue
        starts = sub["start"].to_numpy()
        durations = (sub["stop"] - sub["start"]).to_numpy()
        x_ranges = list(zip(starts.tolist(), durations.tolist()))
        ax.broken_barh(
            x_ranges,
            (y_bottom, rectangle_height),
            facecolors=group_colors[sex_label],
            edgecolors=group_colors[sex_label],
            linewidth=rectangle_linewidth,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_bottom - rectangle_height * 0.1, y_bottom + rectangle_height * 1.1)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.minorticks_off()
    ax.tick_params(axis="x", length=0, labelsize=8)

    fig.suptitle(
        f"USV timeline — session {session_id} "
        f"(total non-noise n={df.height})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    if output_path is not None:
        out_path = pathlib.Path(configure_path(output_path))
        if fig_format is not None:
            out_path = out_path.with_suffix(f".{fig_format.lstrip('.')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        message_output(f"Saved session USV timeline figure: {out_path}")

    return fig


# Embedding columns expected in each session's USV summary CSV. Two
# maps live side-by-side: VAE-based UMAP and QLVM-based UMAP. Both
# carry a category and a supercategory integer label.
EMBEDDING_COORD_COLS = (
    "vae_umap1",
    "vae_umap2",
    "qlvm_dim1",
    "qlvm_dim2",
)
EMBEDDING_LABEL_COLS = (
    "vae_category",
    "vae_supercategory",
    "qlvm_category",
    "qlvm_supercategory",
)
EMBEDDING_ALL_COLS = EMBEDDING_COORD_COLS + EMBEDDING_LABEL_COLS

# Per-USV acoustic features (written by compute_usv_acoustic_features into
# usv_summary.csv) -- pulled into the pooled embeddings DataFrame as continuous
# color-by metrics in the embedding explorer.
EMBEDDING_FEATURE_COLS = (
    "mean_freq_hz",
    "peak_freq_hz",
    "freq_bandwidth_hz",
    "mean_amplitude",
    "max_amplitude",
    "spectral_entropy",
)
# Extra per-USV columns pulled into the pooled embeddings DataFrame.
# They power the auxiliary scatters (sex, duration) and the acoustic-feature
# color-by metrics in ``plot_embedding_with_category_thumbnails`` / the embedding
# explorer. The cache file is invalidated automatically when these columns are
# missing -- see schema-check logic in ``build_pooled_embeddings_df``.
EMBEDDING_EXTRA_COLS = ("emitter", "duration") + EMBEDDING_FEATURE_COLS


def build_pooled_embeddings_df(
    sessions_txt_path: str,
    cache_path: str | None = None,
    rebuild_cache: bool = False,
    noise_col_id: str = "vae_supercategory",
    noise_categories: tuple[int, ...] = (0,),
    message_output: Callable | None = None,
) -> pls.DataFrame:
    """
    Description
    -----------
    Build (or load from a parquet cache) a single Polars DataFrame
    that pools per-USV embedding coordinates and category labels
    across every session listed in ``sessions_txt_path``. The returned
    DataFrame is the master table consumed by the marimo embedding
    explorer notebook: it carries the four UMAP coordinate columns
    (``vae_umap1/2``, ``qlvm_dim1/2``), the four label columns
    (``vae_category``, ``vae_supercategory``, ``qlvm_category``,
    ``qlvm_supercategory``), and — critically — a ``(session_id,
    row_index)`` pair per row that keys directly back into the
    consolidated spectrogram h5 at
    ``/spectrogram/<session_id>/spectrograms[row_index]`` (and the
    matching ``durations[row_index]``). The ``row_index`` is assigned
    BEFORE noise filtering, so it stays aligned with the original
    on-disk spectrogram order.

    When ``cache_path`` is supplied the function writes the pooled
    DataFrame to parquet so subsequent launches load in seconds
    instead of re-reading 100s of CSVs. Set ``rebuild_cache=True`` to
    force a fresh rebuild.

    Parameters
    ----------
    sessions_txt_path (str)
        Path to a text file listing one session root per line
        (``#`` / blank lines skipped). Each path is run through
        ``configure_path``.
    cache_path (str | None)
        Optional path at which to read / write a parquet cache. Run
        through ``configure_path``. When ``None``, the DataFrame is
        rebuilt from CSVs every time and never saved.
    rebuild_cache (bool)
        If True, ignore any existing cache file and rebuild from
        CSVs (then overwrite the cache).
    noise_col_id (str)
        CSV column used to flag noise rows; default
        ``"vae_supercategory"``.
    noise_categories (tuple of int)
        Values of ``noise_col_id`` to drop as noise; default
        ``(0,)``.
    message_output (Callable | None)
        Logger; defaults to ``print``. Per-session load failures
        (missing CSV, SMB timeouts, missing columns) are logged and
        the offending session is skipped.

    Returns
    -------
    pooled (pls.DataFrame)
        Schema:
            session_id (Utf8)
            row_index (UInt32)
            vae_umap1, vae_umap2 (Float64)
            vae_category, vae_supercategory (Int64)
            qlvm_dim1, qlvm_dim2 (Float64)
            qlvm_category, qlvm_supercategory (Int64)
        Columns missing from individual sessions become nulls in the
        pooled output (diagonal concat).
    """

    if message_output is None:
        message_output = print

    # Columns we now require in the cached parquet. If the file on
    # disk is missing any of these (older cache), trigger a rebuild
    # transparently so the caller doesn't have to flip ``rebuild_cache``
    # every time the schema is extended.
    required_extra_cols = {"sex", "duration"} | set(EMBEDDING_FEATURE_COLS)
    required_cols = (
        {"session_id", "row_index"}
        | set(EMBEDDING_ALL_COLS)
        | required_extra_cols
    )

    cache_p: pathlib.Path | None = None
    if cache_path is not None:
        cache_p = pathlib.Path(configure_path(cache_path))
        if cache_p.exists() and not rebuild_cache:
            cached = pls.read_parquet(str(cache_p))
            missing = required_cols - set(cached.columns)
            if not missing:
                message_output(f"Loading pooled embeddings DF from cache: {cache_p}")
                return cached
            message_output(
                f"Cache at {cache_p} is missing columns {sorted(missing)}; "
                f"rebuilding."
            )

    sessions_txt_path = configure_path(sessions_txt_path)
    with open(sessions_txt_path, "r") as txt_file:
        session_roots = [
            configure_path(stripped)
            for stripped in (line.strip() for line in txt_file)
            if stripped and not stripped.startswith("#")
        ]

    select_cols = list(
        set(EMBEDDING_ALL_COLS) | set(EMBEDDING_EXTRA_COLS) | {noise_col_id}
    )

    frames: list[pls.DataFrame] = []
    total_sessions = len(session_roots)
    for session_idx, session_root in enumerate(session_roots, start=1):
        # Periodic progress (the per-session CSV reads dominate the wall-clock,
        # especially over a network mount, and are otherwise silent).
        if session_idx == 1 or session_idx % 25 == 0 or session_idx == total_sessions:
            message_output(f"[pool] reading session {session_idx}/{total_sessions} ...")
        try:
            csv_path = first_match_or_raise(
                root=pathlib.Path(session_root) / "audio",
                pattern="*_usv_summary.csv",
                recursive=False,
                label="USV summary CSV",
            )
        except (FileNotFoundError, OSError) as exc:
            message_output(f"[skip] {session_root}: {exc}")
            continue
        try:
            df = pls.read_csv(str(csv_path), columns=select_cols)
        except pls.exceptions.ColumnNotFoundError:
            df = pls.read_csv(str(csv_path))
        except (OSError, IOError) as exc:
            message_output(f"[skip] {csv_path}: {exc}")
            continue

        df = df.with_row_index(name="row_index")
        if df.height == 0:
            # An empty usv_summary (a session with no vocalizations) infers every
            # column as String / Null, which both breaks the noise is_in filter and
            # mismatches the later vertical concat -- and it contributes no rows
            # anyway, so skip it.
            continue
        # CSV dtype inference disagrees across sessions (e.g. an all-null coordinate
        # or label column infers as String), which breaks the integer noise filter
        # and the diagonal concat ("String is incompatible with Float64"). Coerce
        # each numeric column to a consistent dtype (unparseable -> null) -- floats
        # for coordinates / acoustic features / duration, Int64 for the category
        # labels -- so every session frame lines up. String columns (emitter) and
        # the row index are left untouched.
        float_cols = set(EMBEDDING_COORD_COLS) | set(EMBEDDING_FEATURE_COLS) | {"duration"}
        label_cols = set(EMBEDDING_LABEL_COLS)
        casts = [
            pls.col(col_name).cast(pls.Float64, strict=False) if col_name in float_cols
            else pls.col(col_name).cast(pls.Int64, strict=False)
            for col_name in df.columns if col_name in float_cols or col_name in label_cols
        ]
        if casts:
            df = df.with_columns(casts)
        if noise_col_id in df.columns and noise_categories:
            df = df.filter(~pls.col(noise_col_id).is_in(list(noise_categories)))

        # Look up the session's male / female track ids from the
        # tracking h5 so we can map ``emitter`` -> ``sex``. Failure
        # here just yields empty ids; all rows end up as "unassigned".
        male_id: str = ""
        female_id: str = ""
        try:
            tracking_h5 = first_match_or_raise(
                root=pathlib.Path(session_root) / "video",
                pattern="*_points3d_translated_rotated_metric.h5",
                recursive=True,
                label="3D tracking h5",
            )
            with h5py.File(str(tracking_h5), "r") as h5_track:
                track_names = [
                    item.decode("utf-8") for item in list(h5_track["track_names"])
                ]
            if len(track_names) > 0:
                male_id = track_names[0]
            if len(track_names) > 1:
                female_id = track_names[1]
        except (FileNotFoundError, OSError, KeyError) as exc:
            message_output(f"[skip-tracks] {session_root}: {exc}")

        if "emitter" in df.columns:
            df = df.with_columns(
                pls.when(pls.col("emitter") == male_id)
                .then(pls.lit("male"))
                .when(pls.col("emitter") == female_id)
                .then(pls.lit("female"))
                .otherwise(pls.lit("unassigned"))
                .alias("sex")
            )
        else:
            df = df.with_columns(pls.lit("unassigned").alias("sex"))

        session_id = pathlib.Path(session_root).name
        df = df.with_columns(pls.lit(session_id).alias("session_id"))
        keep_cols = ["session_id", "row_index"] + [
            c for c in EMBEDDING_ALL_COLS if c in df.columns
        ]
        for c in ("sex", "duration") + EMBEDDING_FEATURE_COLS:
            if c in df.columns:
                keep_cols.append(c)
        frames.append(df.select(keep_cols))

    if not frames:
        message_output("No sessions could be loaded; returning empty DataFrame.")
        empty_schema: dict[str, pls.DataType] = {
            "session_id": pls.Utf8,
            "row_index": pls.UInt32,
        }
        for c in EMBEDDING_COORD_COLS:
            empty_schema[c] = pls.Float64
        for c in EMBEDDING_LABEL_COLS:
            empty_schema[c] = pls.Int64
        return pls.DataFrame(schema=empty_schema)

    pooled = pls.concat(frames, how="diagonal")

    # Guarantee every expected optional column exists (null-filled when the
    # whole cohort lacks it), so the parquet cache always carries the full
    # schema and required_cols stays satisfied -- otherwise a cohort missing a
    # feature would force a rebuild on every load. Extending the schema still
    # invalidates an older cache once (it lacks the new column), then this fill
    # keeps it stable.
    fill_dtypes = {c: pls.Float64 for c in ("duration",) + EMBEDDING_FEATURE_COLS}
    fill_dtypes["sex"] = pls.Utf8
    missing_fills = [
        pls.lit(None, dtype=dtype).alias(c)
        for c, dtype in fill_dtypes.items()
        if c not in pooled.columns
    ]
    if missing_fills:
        pooled = pooled.with_columns(missing_fills)

    if cache_p is not None:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        pooled.write_parquet(str(cache_p))
        message_output(
            f"Cached pooled embeddings DF to {cache_p} "
            f"({pooled.height:,} rows, {pooled.select('session_id').unique().height} sessions)"
        )

    return pooled


def build_vae_density_npz(
    sessions_txt_path: str,
    out_npz_path: str,
    *,
    label_col: str = "vae_supercategory",
    cache_path: str | None = None,
    grid: int = 300,
    smooth_sigma: float = 1.5,
    knn: int = 15,
    message_output: Callable | None = None,
) -> str:
    """
    Description
    -----------
    Precompute the cohort VAE embedding landscape and save it to an ``.npz`` in the
    SAME schema the sequence figure's left panel reads for QLVM, so the VAE panel can
    show a precomputed cohort gray_r density + category boundaries (rather than
    re-pooling ~600k coordinates on every per-session render). Unlike QLVM — whose
    coordinates live on the periodic unit torus and whose watershed arrays ship from
    the modeling pipeline — VAE coordinates live only in per-session
    ``usv_summary.csv`` files, so this builds the analogue once from the pooled
    cohort table.

    The output carries three arrays:
    ``heatmap`` — a 2-D histogram density of ``(vae_umap1, vae_umap2)`` over the
    cohort's coordinate range (optionally Gaussian-smoothed into a KDE-like density
    estimate), oriented ``[row=y, col=x]`` for ``origin="lower"``;
    ``ws_labels_periodic`` — the category field on the same grid, assigned by a
    nearest-neighbour classifier fit on the cohort points (the VAE analogue of the
    QLVM watershed label field, so the figure's neighbour-difference boundary mask
    works unchanged);
    ``extent`` — ``[x0, x1, y0, y1]`` (the umap coordinate range; QLVM omits this and
    defaults to the unit square).

    Run once per clustering granularity: ``label_col="vae_supercategory"`` for the
    COARSE map and ``label_col="vae_category"`` for the FINE map. Write them to
    ``<spectrograms_dir>/vae/vae_density_{coarse,fine}.npz`` so the figure resolves
    them from ``shared_resources.spectrograms_dir`` by convention.

    Parameters
    ----------
    sessions_txt_path (str)
        Path to a text file listing one session root per line (passed straight to
        ``build_pooled_embeddings_df``).
    out_npz_path (str)
        Destination ``.npz`` path (run through ``configure_path``).
    label_col (str)
        Cohort label column to rasterize into ``ws_labels_periodic`` —
        ``"vae_supercategory"`` (coarse) or ``"vae_category"`` (fine).
    cache_path (str | None)
        Optional parquet cache for the pooled DataFrame (forwarded to
        ``build_pooled_embeddings_df``).
    grid (int)
        Side length of the square density / label grid (``grid x grid`` cells).
    smooth_sigma (float)
        Gaussian-filter sigma (in grid cells) applied to the histogram density;
        ``0`` leaves the raw histogram.
    knn (int)
        Number of neighbours for the grid label classifier.
    message_output (Callable | None)
        Optional logger; ``None`` is silent.

    Returns
    -------
    out_path (str)
        The path the ``.npz`` was written to.
    """

    emit = message_output if message_output is not None else (lambda *_a, **_kw: None)
    pooled = build_pooled_embeddings_df(
        sessions_txt_path, cache_path=cache_path, message_output=message_output
    )
    sub = pooled.select(["vae_umap1", "vae_umap2", label_col]).drop_nulls()
    coords_x = sub["vae_umap1"].to_numpy().astype(np.float64)
    coords_y = sub["vae_umap2"].to_numpy().astype(np.float64)
    point_labels = sub[label_col].to_numpy()

    x0, x1 = float(coords_x.min()), float(coords_x.max())
    y0, y1 = float(coords_y.min()), float(coords_y.max())
    extent = np.array([x0, x1, y0, y1], dtype=np.float64)

    # Density: 2-D histogram over the coordinate range; counts come out [x, y] so
    # transpose to [y, x] for origin="lower". An optional Gaussian filter turns the
    # histogram into a smooth density estimate (rendered without interpolation).
    counts, _, _ = np.histogram2d(coords_x, coords_y, bins=grid, range=[[x0, x1], [y0, y1]])
    heatmap = counts.T
    if smooth_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)

    # Label field: a nearest-neighbour classifier fit on the cohort points, predicted
    # on the grid, yields a Voronoi-like category map (the VAE analogue of the QLVM
    # watershed labels) on which the figure's neighbour-difference mask draws lines.
    classifier = KNeighborsClassifier(n_neighbors=knn, weights="uniform")
    classifier.fit(np.column_stack([coords_x, coords_y]), point_labels)
    grid_x = np.linspace(x0, x1, grid)
    grid_y = np.linspace(y0, y1, grid)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    grid_labels = classifier.predict(
        np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    ).reshape(grid, grid)

    out_path = str(configure_path(out_npz_path))
    np.savez(
        out_path,
        heatmap=heatmap.astype(np.float32),
        ws_labels_periodic=grid_labels.astype(np.int16),
        extent=extent,
    )
    emit(f"Saved VAE cohort density ({label_col}, {coords_x.size} USVs) to {out_path}.")
    return out_path


@click.command(name="build-vae-density")
@click.option('--sessions-txt', type=str, required=True, help='Text file listing one session root per line.')
@click.option('--out-coarse', type=str, required=True, help='Output .npz path for the COARSE map (vae_supercategory).')
@click.option('--out-fine', type=str, required=True, help='Output .npz path for the FINE map (vae_category).')
@click.option('--cache-path', type=str, default=None, required=False, help='Optional parquet cache for the pooled DataFrame.')
@click.option('--grid', type=int, default=300, required=False, help='Density / label grid side length.')
@click.option('--smooth-sigma', type=float, default=1.5, required=False, help='Gaussian sigma (grid cells); 0 = raw histogram.')
@click.option('--knn', type=int, default=15, required=False, help='Neighbours for the grid label classifier.')
def build_vae_density_cli(sessions_txt, out_coarse, out_fine, cache_path, grid, smooth_sigma, knn) -> None:
    """
    Description
    -----------
    One-off CLI that builds BOTH cohort VAE landscape files consumed by the USV
    sequence figure: the COARSE map (``vae_supercategory`` boundaries) and the FINE
    map (``vae_category`` boundaries). Write them to
    ``<shared_resources.spectrograms_dir>/vae/vae_density_{coarse,fine}.npz`` so the
    figure resolves them automatically.

    Parameters
    ----------
    See the command options.

    Returns
    -------
    None
    """

    for label_col, out_path in (("vae_supercategory", out_coarse), ("vae_category", out_fine)):
        build_vae_density_npz(
            sessions_txt, out_path, label_col=label_col, cache_path=cache_path,
            grid=grid, smooth_sigma=smooth_sigma, knn=knn, message_output=print,
        )


def _pick_spiral_with_grid(
    pts: np.ndarray,
    n_per: int,
    cx0: float,
    cy0: float,
    r_max: float,
    labels_grid: np.ndarray | None,
    xx: np.ndarray | None,
    yy: np.ndarray | None,
    cluster_label: int,
    n_turns: int = 4,
    n_dense: int = 4000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Centroid-rooted Archimedean spiral with watershed-style filtering
    via a labels grid. Direct port of the original
    ``_pick_cluster_samples`` spiral branch: dense spiral path of
    ``n_dense`` points expanding from ``(cx0, cy0)`` out to ``r_max``
    over ``n_turns`` revolutions, filtered to grid cells whose
    predicted label matches ``cluster_label`` (and inside the data
    range), evenly subsampled to ``n_per`` positions, snapped to the
    nearest unused in-category point.

    Parameters
    ----------
    pts (np.ndarray)
        (M, 2) in-category coordinates.
    n_per (int)
        Number of picks to return.
    cx0, cy0 (float)
        Spiral center (use the cluster centroid for an exact match to
        the original code).
    r_max (float)
        Maximum spiral radius. Original uses ``dists_c.max()``.
    labels_grid (np.ndarray | None)
        ``(res_y, res_x)`` array of predicted category labels from a
        k-NN fit on (x, y) → category. ``None`` disables the filter.
    xx, yy (np.ndarray | None)
        ``(res_x,)`` and ``(res_y,)`` axis ticks corresponding to
        ``labels_grid`` columns and rows respectively.
    cluster_label (int)
        Integer label of THIS category in ``labels_grid``.
    n_turns (int)
        Number of revolutions the spiral makes from center to
        ``r_max``.
    n_dense (int)
        Number of points along the dense spiral path before grid
        filtering.
    rng (np.random.Generator | None)
        Random generator used to draw a single angular offset
        ``theta_offset`` in ``[0, 2 pi)`` that rotates the whole
        spiral. ``None`` disables the rotation (deterministic spiral
        starting from theta=0), which matches the original behavior.

    Returns
    -------
    local_picks (np.ndarray)
        Indices into ``pts`` for the selected rows.
    xs_in_dense, ys_in_dense (np.ndarray)
        Coordinates of the dense spiral path AFTER the label-grid
        filter (i.e., only the segments that fall inside this
        cluster's footprint). The caller plots these as the visible
        trajectory so the spiral never extends past the cluster's
        actual region — matching the picks' actual positions.
    """

    if pts.shape[0] == 0 or n_per <= 0:
        return (
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )
    n_take = min(n_per, pts.shape[0])
    if r_max <= 0.0:
        return (
            np.arange(n_take, dtype=int),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    t_dense = np.linspace(0.0, 1.0, n_dense)
    theta_offset = 0.0 if rng is None else float(rng.uniform(0.0, 2.0 * np.pi))
    theta_d = 2.0 * np.pi * n_turns * t_dense + theta_offset
    r_d = r_max * t_dense
    xs_d = cx0 + r_d * np.cos(theta_d)
    ys_d = cy0 + r_d * np.sin(theta_d)

    if labels_grid is not None and xx is not None and yy is not None:
        # Map (xs_d, ys_d) -> grid cells, filter to (a) inside data
        # range, (b) grid cell's predicted label == cluster_label.
        gx_lo, gx_hi = float(xx[0]), float(xx[-1])
        gy_lo, gy_hi = float(yy[0]), float(yy[-1])
        res_x = labels_grid.shape[1]
        res_y = labels_grid.shape[0]
        px = ((xs_d - gx_lo) / (gx_hi - gx_lo) * (res_x - 1)).astype(int)
        py = ((ys_d - gy_lo) / (gy_hi - gy_lo) * (res_y - 1)).astype(int)
        inside_box = (
            (xs_d >= gx_lo) & (xs_d <= gx_hi)
            & (ys_d >= gy_lo) & (ys_d <= gy_hi)
        )
        px_safe = np.clip(px, 0, res_x - 1)
        py_safe = np.clip(py, 0, res_y - 1)
        # NaN cells (low-density / masked) compare False under == so
        # they are correctly excluded.
        cell_label = labels_grid[py_safe, px_safe]
        inside = inside_box & (cell_label == cluster_label)
        xs_in_dense = xs_d[inside]
        ys_in_dense = ys_d[inside]
    else:
        xs_in_dense = xs_d
        ys_in_dense = ys_d

    if xs_in_dense.size == 0:
        # Filter killed everything (low-density cluster); fall back
        # to the unfiltered spiral.
        xs_in_dense, ys_in_dense = xs_d, ys_d

    # The dense in-cluster path is what gets plotted as the visible
    # spiral trajectory; the subsampled positions below are what feed
    # the snap-to-nearest data-point pick.
    if xs_in_dense.size >= n_take:
        sel = np.linspace(0, xs_in_dense.size - 1, n_take).astype(int)
        xs_in_pick, ys_in_pick = xs_in_dense[sel], ys_in_dense[sel]
    else:
        xs_in_pick, ys_in_pick = xs_in_dense, ys_in_dense

    used: set[int] = set()
    picks: list[int] = []
    for tx, ty in zip(xs_in_pick, ys_in_pick):
        d2 = (pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2
        for kk in np.argsort(d2):
            kk_int = int(kk)
            if kk_int not in used:
                used.add(kk_int)
                picks.append(kk_int)
                break
        if len(picks) >= n_take:
            break
    return np.asarray(picks, dtype=int), xs_in_dense, ys_in_dense


def _medoid_xy(
    pts: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """
    Description
    -----------
    Robust cluster center: Weiszfeld's geometric median of ``pts``
    snapped to the nearest actual data point. The geometric median is
    the L1 analogue of the mean and is robust to elongated /
    asymmetric clusters where the arithmetic centroid would fall in
    low-density space. ``pts`` is a ``(M, 2)`` array of in-category
    embedding coordinates; the returned ``(x, y)`` always coincides
    with one of the rows of ``pts``.

    Parameters
    ----------
    pts (np.ndarray)
        2D coordinates of the in-category points.
    max_iter (int)
        Maximum number of Weiszfeld iterations.
    tol (float)
        Convergence tolerance (Euclidean step size between iterations).

    Returns
    -------
    (cx, cy) (tuple of float)
        Coordinates of the in-category point closest to the geometric
        median, or ``(0.0, 0.0)`` if ``pts`` is empty.
    """

    n = pts.shape[0]
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(pts[0, 0]), float(pts[0, 1])

    c = pts.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - c, axis=1)
        if not (d > tol).any():
            break
        w = 1.0 / np.maximum(d, tol)
        c_new = (pts * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(c_new - c) < tol:
            c = c_new
            break
        c = c_new

    d_final = np.linalg.norm(pts - c, axis=1)
    nearest = int(np.argmin(d_final))
    return float(pts[nearest, 0]), float(pts[nearest, 1])


def _pick_category_samples(
    pts: np.ndarray,
    n_per: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Description
    -----------
    Pick up to ``n_per`` indices into ``pts`` according to one of five
    sampling strategies. ``pts`` is the (M, 2) array of `(x, y)`
    embedding coordinates for the rows belonging to one category.

    Strategies
    ----------
    - ``"random"``: uniform random draw.
    - ``"nearest"``: the ``n_per`` points closest to the category
      centroid (most "typical" / central samples).
    - ``"spread"``: iterative farthest-point sampling starting
      from the centroid — gives maximally spread coverage.
    - ``"grid"``: overlay a dense grid over the category's bounding
      box, keep up to ``n_per`` evenly spaced grid points, and for
      each pick the nearest data point that hasn't been picked yet.
    - ``"spiral"``: Archimedean spiral expanding from the cluster's
      **medoid** (geometric-median snapped to the nearest data point
      via Weiszfeld) out to the farthest in-category point; at each of
      ``n_per`` evenly spaced spiral positions take the nearest
      unused data point. Gives an ordered "central → peripheral"
      coverage. Using the medoid instead of the arithmetic centroid
      keeps the spiral's origin inside the cluster's dense region
      even for elongated / asymmetric clusters (e.g. VAE UMAP).

    Returns
    -------
    local_indices (np.ndarray)
        1-D int array of indices into ``pts`` of the selected rows,
        in sampling order (so the first index is the "first" pick).
    """

    n_take = min(n_per, pts.shape[0])
    if n_take == 0:
        return np.zeros(0, dtype=int)

    if method == "random":
        idxs = rng.choice(pts.shape[0], n_take, replace=False)
        return np.sort(idxs)

    cx0 = float(np.mean(pts[:, 0]))
    cy0 = float(np.mean(pts[:, 1]))

    if method == "nearest":
        dists = np.sqrt((pts[:, 0] - cx0) ** 2 + (pts[:, 1] - cy0) ** 2)
        return np.argsort(dists)[:n_take]

    if method == "spread":
        selected: list[int] = []
        sel_xy = np.array([[cx0, cy0]])
        remaining = np.arange(pts.shape[0])
        for _ in range(n_take):
            cands = pts[remaining]
            diffs = cands[:, None, :] - sel_xy[None, :, :]
            d2_min = (diffs ** 2).sum(axis=-1).min(axis=-1)
            best_r = int(np.argmax(d2_min))
            selected.append(int(remaining[best_r]))
            sel_xy = np.vstack([sel_xy, pts[remaining[best_r]]])
            remaining = np.delete(remaining, best_r)
        return np.asarray(selected, dtype=int)

    if method == "grid":
        x_lo, y_lo = pts.min(axis=0)
        x_hi, y_hi = pts.max(axis=0)
        n_side = max(int(np.ceil(np.sqrt(n_take) * 3)), 6)
        gx_vals = np.linspace(x_lo, x_hi, n_side + 2)[1:-1]
        gy_vals = np.linspace(y_lo, y_hi, n_side + 2)[1:-1]
        grid_cells = [(gx, gy) for gx in gx_vals for gy in gy_vals]
        if len(grid_cells) > n_take:
            sel_idx = np.linspace(0, len(grid_cells) - 1, n_take).astype(int)
            grid_cells = [grid_cells[i] for i in sel_idx]
        used: set[int] = set()
        picks: list[int] = []
        for gx, gy in grid_cells:
            d2 = (pts[:, 0] - gx) ** 2 + (pts[:, 1] - gy) ** 2
            for kk in np.argsort(d2):
                kk_int = int(kk)
                if kk_int not in used:
                    used.add(kk_int)
                    picks.append(kk_int)
                    break
            if len(picks) >= n_take:
                break
        return np.asarray(picks, dtype=int)

    if method == "spiral":
        # ``spiral`` is special-cased in the plotting function so it
        # can use the k-NN label grid as a watershed analog. If we
        # got here, the caller didn't route through that path -- fall
        # back to a centroid-rooted spiral with no grid filter (less
        # geometrically clean but still functional).
        cx0_c = float(np.mean(pts[:, 0]))
        cy0_c = float(np.mean(pts[:, 1]))
        dists_c = np.sqrt((pts[:, 0] - cx0_c) ** 2 + (pts[:, 1] - cy0_c) ** 2)
        r_max = float(dists_c.max()) if dists_c.size else 0.0
        if r_max <= 0.0:
            idxs = rng.choice(pts.shape[0], n_take, replace=False)
            return np.sort(idxs)
        n_turns = 4
        n_dense = 4000
        t_dense = np.linspace(0.0, 1.0, n_dense)
        theta = 2.0 * np.pi * n_turns * t_dense
        r = r_max * t_dense
        xs_d = cx0_c + r * np.cos(theta)
        ys_d = cy0_c + r * np.sin(theta)
        sel = np.linspace(0, n_dense - 1, n_take).astype(int)
        xs_s, ys_s = xs_d[sel], ys_d[sel]
        used_s: set[int] = set()
        picks_s: list[int] = []
        for tx, ty in zip(xs_s, ys_s):
            d2 = (pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2
            for kk in np.argsort(d2):
                kk_int = int(kk)
                if kk_int not in used_s:
                    used_s.add(kk_int)
                    picks_s.append(kk_int)
                    break
            if len(picks_s) >= n_take:
                break
        return np.asarray(picks_s, dtype=int)

    msg = (
        f"Unknown sampling_method {method!r}. "
        "Choose from 'random', 'nearest', 'spread', 'grid', 'spiral'."
    )
    raise ValueError(msg)


def _knn_boundary_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    n_neighbors: int = 15,
    grid_resolution: int = 200,
    density_smoothing_sigma: float = 2.5,
    density_min_count: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Fit a k-nearest-neighbor classifier on the scatter points and
    predict integer category labels on a regular 2D grid covering
    ``[x_lo, x_hi] x [y_lo, y_hi]``. The grid is then **masked by
    point density** (a Gaussian-smoothed 2D histogram of the same
    points): cells whose smoothed count is below
    ``density_min_count`` are set to NaN, so when the result is fed
    to ``ax.contour`` the contour algorithm skips empty UMAP regions
    and the boundaries appear only where data actually lives. The
    boundaries themselves still follow point density via k-NN, not
    centroid Voronoi geometry.

    Returns
    -------
    xx, yy (np.ndarray)
        ``(grid_resolution,)`` axis ticks (suitable for the ``X`` and
        ``Y`` arguments of ``ax.contour``).
    grid_labels (np.ndarray)
        ``(grid_resolution, grid_resolution)`` FLOAT array of
        predicted category labels (NaN where data density is too
        low), rows indexed by ``y``, columns by ``x``.
    """

    k = max(1, min(int(n_neighbors), x.size))
    clf = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    clf.fit(np.stack([x, y], axis=1), labels)
    xx = np.linspace(x_lo, x_hi, grid_resolution)
    yy = np.linspace(y_lo, y_hi, grid_resolution)
    xg, yg = np.meshgrid(xx, yy)
    grid_pred = clf.predict(np.stack([xg.ravel(), yg.ravel()], axis=1))
    grid_labels = grid_pred.reshape(grid_resolution, grid_resolution).astype(float)

    # Density mask: 2D histogram of points, smoothed slightly so
    # marginally-dense regions don't fragment. Cells below
    # ``density_min_count`` become NaN -> contour skips them.
    x_edges = np.linspace(x_lo, x_hi, grid_resolution + 1)
    y_edges = np.linspace(y_lo, y_hi, grid_resolution + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    counts = counts.T  # match (row=y, col=x) orientation of grid_labels
    if density_smoothing_sigma > 0:
        counts = gaussian_filter1d(counts, sigma=density_smoothing_sigma, axis=0)
        counts = gaussian_filter1d(counts, sigma=density_smoothing_sigma, axis=1)
    grid_labels[counts < density_min_count] = np.nan
    return xx, yy, grid_labels


def plot_embedding_with_category_thumbnails(
    sessions_txt_path: str,
    consolidated_h5_path: str,
    map_type: str = "vae",
    category_col_suffix: str = "supercategory",
    n_samples_per_category: int = 8,
    apply_mask: bool = True,
    mask_excluded_categories: tuple[int, ...] | int | None = (),
    category_colors: dict | None = None,
    sampling_method: str = "random",
    cluster_centers_xy: dict | None = None,
    cluster_centers_h5_path: str | None = None,
    cluster_centers_json_path: str | None = None,
    draw_spiral_overlay: bool = False,
    spiral_show_only_for: int | None = None,
    spiral_color: str = "#000000",
    spiral_linewidth: float = 2.0,
    spiral_radius_scale: float = 0.35,
    spiral_radius_abs: float | None = None,
    spiral_n_turns: int = 4,
    spiral_random_phase: bool = True,
    draw_cluster_boundaries: bool = True,
    knn_boundary_neighbors: int = 15,
    knn_boundary_resolution: int = 200,
    knn_boundary_density_min_count: float = 0.2,
    knn_boundary_density_smoothing_sigma: float = 2.5,
    annotate_picks_on_scatter: bool = True,
    pick_number_fontsize: float = 11.0,
    annotate_cluster_ids: bool = False,
    cluster_id_fontsize: float = 12.0,
    thumbnail_size_fraction: float = 0.5,
    thumbnail_hspace: float = 0.02,
    thumbnail_wspace: float = 0.05,
    tile_orientation: str = "horizontal",
    unstretched_specs: bool = False,
    fig_size: tuple[float, float] = (16.0, 12.0),
    fig_dpi: int = 300,
    output_path: str | None = None,
    fig_format: str | None = None,
    noise_col_id: str = "vae_supercategory",
    noise_categories: tuple[int, ...] = (0,),
    scatter_max_points: int = 50_000,
    scatter_point_size: float = 4.0,
    scatter_point_alpha: float = 0.5,
    pooled_df: pls.DataFrame | None = None,
    embeddings_cache_path: str | None = None,
    rebuild_embeddings_cache: bool = False,
    message_output: Callable | None = None,
    seed: int | None = 42,
) -> plt.Figure:
    """
    Description
    -----------
    Render a static two-panel figure summarising one of the per-USV
    embedding maps:

    - **Left panel**: embedding scatter of every (non-noise) USV across the
      sessions listed in ``sessions_txt_path``, colored by the chosen
      categorical label.
    - **Right panel**: a thumbnail grid of spectrograms. Each ROW
      corresponds to one category — ``n_samples_per_category`` USVs are
      randomly sampled from that category and rendered side by side.
      A thin colored rectangle is drawn around each row in the
      category's color, matching the scatter point coloring on the
      left.

    Spectrograms are pulled from the consolidated SAM2 + spectrogram
    HDF5 (``consolidated_h5_path``). SAM2 segmentation masks can be
    applied to zero out non-USV pixels; per-category control is
    available via ``mask_excluded_categories``, which lists categories
    where the mask should NOT be applied (useful when SAM2 produces
    poor masks for some call types).

    The pooled embeddings DataFrame is built by
    ``build_pooled_embeddings_df`` (or supplied directly via
    ``pooled_df`` to skip the loader). The function flips each mask
    along the frequency axis before applying it because masks are
    stored with image (top-down) row convention while spectrograms are
    drawn with audio (``origin='lower'``) convention.

    Parameters
    ----------
    sessions_txt_path (str)
        Path to a text file listing one session root per line.
    consolidated_h5_path (str)
        Path to the consolidated SAM2 + spectrogram HDF5 store.
    map_type (str)
        ``"vae"`` or ``"qlvm"`` - selects which embedding map to plot
        (the VAE umap or the QLVM torus).
    category_col_suffix (str)
        ``"category"`` or ``"supercategory"`` - selects which
        categorical label to color and group by.
    n_samples_per_category (int)
        How many spectrograms to display per category row.
    apply_mask (bool)
        Master toggle for SAM2 mask application.
    mask_excluded_categories (tuple of int)
        Categories whose spectrograms should be shown WITHOUT a mask
        even when ``apply_mask=True``.
    category_colors (dict | None)
        Optional mapping ``{category_int: hex_color}``. If ``None``,
        uses ``tab10`` / ``tab20`` automatically.
    annotate_picks_on_scatter (bool)
        If ``True``, overlay the integer pick index (1..N) on each
        sampled point in the main scatter so the row of spectrograms
        can be cross-referenced with the dot it came from.
    pick_number_fontsize (float)
        Font size for the pick-index annotations on the scatter.
    annotate_cluster_ids (bool)
        If ``True``, draw the integer cluster ID at the (resolved)
        center of each category on the main scatter. Centers are
        taken from ``cluster_centers_xy`` /
        ``cluster_centers_h5_path`` / ``cluster_centers_json_path``
        when supplied (same priority chain as the spiral overlay);
        otherwise the per-category mean of the displayed scatter
        points is used as a fallback so the label still lands inside
        its cluster.
    cluster_id_fontsize (float)
        Font size used for the cluster-ID labels when
        ``annotate_cluster_ids`` is ``True``.
    thumbnail_size_fraction (float)
        Final shrink multiplier applied to the thumbnail cell side.
        The cell side is first derived to make the whole thumbnail
        block match the LEFT main scatter's on-page footprint:
        ``target_side_in = min(left_region_w_in, top_scatter_row_h_in)``
        and ``cell_side_in = target_side_in / max(n_rows, n_cols)``.
        ``1.0`` keeps the scatter-matched size; ``<1.0`` shrinks
        further. Must satisfy ``0 < thumbnail_size_fraction <= 1.0``.
        Combined with ``ax.set_box_aspect(1)`` on every thumbnail
        axes (always on), each cell is square; auto-added spacer
        row/column at the bottom/right of the inner gridspec consume
        any remaining space in the right region.
    thumbnail_hspace (float)
        ``hspace`` for the right-side per-category x per-sample
        gridspec (fraction of average cell height). Default is
        ``0.02`` so adjacent cells touch; raise it if you want gaps.
    thumbnail_wspace (float)
        ``wspace`` for the right-side gridspec (fraction of average
        cell width). Default ``0.05``.
    tile_orientation (str)
        ``"horizontal"`` (default) tiles each category as a ROW of
        ``n_samples_per_category`` thumbnails (n_categories rows
        stacked vertically). ``"vertical"`` tiles each category as a
        COLUMN of ``n_samples_per_category`` thumbnails
        (n_categories columns side by side).
    unstretched_specs (bool)
        If ``True``, every spectrogram is zero-padded to a common
        ``(n_freq, max_dur)`` shape -- where ``max_dur`` is the
        largest stored ``durations`` value across all picks in
        this figure (computed in a pre-pass) -- with the valid
        slice centered horizontally and equal zero borders on
        both sides. The padded array is then shown with
        ``aspect="auto"`` so it fills the uniform slot. Net
        effect: every thumbnail has literally the same on-page
        shape and the same time scale, the longest USV fills its
        slot, and shorter calls sit centered with symmetric
        zero borders proportional to ``1 - dur / max_dur``.
        Default ``False`` keeps the existing ``aspect="auto"``
        behaviour, which stretches every spectrogram horizontally
        to fill its slot. Underlying spec data is identical in
        both cases -- the array is already cropped to the per-USV
        original time-bin count via the ``durations`` field;
        this toggle only changes display padding / aspect.
    fig_size, fig_dpi
        Matplotlib figure size and DPI.
    output_path, fig_format (str | None)
        Optional save path; ``fig_format`` overrides any extension on
        ``output_path``. Both run through ``configure_path``.
    noise_col_id, noise_categories
        Noise filtering passed through to
        ``build_pooled_embeddings_df`` (only used when ``pooled_df`` is
        ``None``).
    scatter_max_points (int)
        Optional cap on points rendered in the scatter (random sample
        with ``seed=42``); the per-category sampling still draws from
        the full pool so all categories are represented in the grid.
    scatter_point_size, scatter_point_alpha
        Marker size and alpha for the scatter.
    pooled_df (pls.DataFrame | None)
        Optionally pass a pre-built pooled-embeddings DataFrame to
        skip the loader. Must contain ``session_id``, ``row_index``,
        the embedding coordinate columns (``vae_umap1/2`` or
        ``qlvm_dim1/2``) and the category column.
    embeddings_cache_path (str | None)
        Parquet cache path passed to ``build_pooled_embeddings_df``.
    rebuild_embeddings_cache (bool)
        Force-rebuild the embeddings cache.
    message_output (Callable | None)
        Logger; defaults to ``print``.
    seed (int)
        Seed for the per-category sampling and the scatter downsample.

    Returns
    -------
    fig (plt.Figure)
        The rendered figure (also written to disk if ``output_path``
        is given).
    """

    if message_output is None:
        message_output = print

    # Defensive: callers often forget the trailing comma when passing
    # a single excluded category (``(6)`` is the int ``6``, not a
    # one-tuple). Accept int or any iterable and normalise to a tuple.
    if mask_excluded_categories is None:
        mask_excluded_categories = ()
    elif isinstance(mask_excluded_categories, int):
        mask_excluded_categories = (mask_excluded_categories,)
    else:
        mask_excluded_categories = tuple(mask_excluded_categories)

    sessions_txt_path = configure_path(sessions_txt_path)
    consolidated_h5_path = configure_path(consolidated_h5_path)

    map_prefix = map_type.lower()
    if map_prefix not in ("vae", "qlvm"):
        msg = f"map_type must be 'vae' or 'qlvm', got {map_type!r}."
        raise ValueError(msg)
    if category_col_suffix not in ("category", "supercategory"):
        msg = (
            f"category_col_suffix must be 'category' or 'supercategory', "
            f"got {category_col_suffix!r}."
        )
        raise ValueError(msg)

    # The QLVM torus coordinates are named qlvm_dim1/qlvm_dim2 (they are not a
    # UMAP); only the VAE embedding uses the _umap1/_umap2 suffix.
    if map_prefix == "qlvm":
        x_col, y_col = "qlvm_dim1", "qlvm_dim2"
    else:
        x_col, y_col = "vae_umap1", "vae_umap2"
    cat_col = f"{map_prefix}_{category_col_suffix}"

    if pooled_df is None:
        pooled_df = build_pooled_embeddings_df(
            sessions_txt_path=sessions_txt_path,
            cache_path=embeddings_cache_path,
            rebuild_cache=rebuild_embeddings_cache,
            noise_col_id=noise_col_id,
            noise_categories=noise_categories,
            message_output=message_output,
        )

    df_clean = pooled_df.drop_nulls(subset=[x_col, y_col, cat_col])
    categories = sorted(
        c for c in set(df_clean[cat_col].to_list()) if c not in noise_categories
    )
    if not categories:
        msg = "No non-noise categories found in pooled_df."
        raise RuntimeError(msg)
    n_categories = len(categories)

    if category_colors is None:
        cmap_name = "tab10" if n_categories <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        category_colors = {
            cat: mcolors.to_hex(cmap(i % cmap.N))
            for i, cat in enumerate(categories)
        }

    rng = np.random.default_rng(seed)

    # Resolve explicit cluster centers (used by the spiral sampler).
    # Priority order:
    #   1. ``cluster_centers_xy`` dict (caller-supplied, highest).
    #   2. ``cluster_centers_h5_path`` -> a small h5 produced by
    #      ``build_qlvm_clusters_h5.py`` with ``/coarse`` and ``/fine``
    #      groups, each carrying ``cluster_centers (N, 2)``. The
    #      group is picked from ``category_col_suffix``
    #      (``supercategory`` -> ``/coarse``; ``category`` -> ``/fine``).
    #   3. ``cluster_centers_json_path`` -> a single QLVM provenance
    #      JSON's ``cluster_centers`` list.
    # Center index ``i`` always maps to label ``i + 1`` (verified
    # against ``dataset_stats.cluster_sizes_standard`` key naming).
    cluster_centers_resolved: dict[int, tuple[float, float]] = {}
    if cluster_centers_xy is not None:
        cluster_centers_resolved = {
            int(k): (float(v[0]), float(v[1])) for k, v in cluster_centers_xy.items()
        }
    elif cluster_centers_h5_path is not None:
        cc_h5_group = "coarse" if category_col_suffix == "supercategory" else "fine"
        with h5py.File(configure_path(cluster_centers_h5_path), "r") as _cc_h5:
            if cc_h5_group not in _cc_h5:
                msg = (
                    f"Group '/{cc_h5_group}' not found in "
                    f"{cluster_centers_h5_path!r}; expected for "
                    f"category_col_suffix={category_col_suffix!r}."
                )
                raise KeyError(msg)
            centers_arr = _cc_h5[f"{cc_h5_group}/cluster_centers"][:]
        for i, c in enumerate(centers_arr):
            cluster_centers_resolved[i + 1] = (float(c[0]), float(c[1]))
    elif cluster_centers_json_path is not None:
        with open(configure_path(cluster_centers_json_path)) as _f:
            _prov = json.load(_f)
        # Direct key access: a provenance JSON explicitly handed in via
        # ``cluster_centers_json_path`` is asserted to carry the
        # ``cluster_centers`` list, so a missing key should surface
        # loudly rather than silently resolving zero centers.
        for i, c in enumerate(_prov["cluster_centers"]):
            cluster_centers_resolved[i + 1] = (float(c[0]), float(c[1]))

    # Downsample for the scatter visualisation BEFORE picks so the
    # boundary grid (computed on the displayed point set) is ready
    # for the spiral filter.
    scatter_df = df_clean
    if scatter_df.height > scatter_max_points:
        scatter_df = scatter_df.sample(n=scatter_max_points, seed=seed)

    # Pre-extract the displayed scatter arrays for reuse below.
    x_all = scatter_df[x_col].to_numpy()
    y_all = scatter_df[y_col].to_numpy()
    cat_all = scatter_df[cat_col].to_numpy()

    # Data-extent bounds used by ALL five UMAP panels so their axes
    # stay pinned to the actual data range and overlays (boundaries,
    # spirals, picks) get clipped to that range instead of pushing
    # the axes outward.
    x_lo, x_hi = float(np.min(x_all)), float(np.max(x_all))
    y_lo, y_hi = float(np.min(y_all)), float(np.max(y_all))
    x_pad = 0.03 * (x_hi - x_lo or 1.0)
    y_pad = 0.03 * (y_hi - y_lo or 1.0)

    # Cluster boundaries via a k-NN classifier on (x, y) -> category,
    # predicted on a 200x200 grid AND masked by point density so the
    # contour algorithm skips empty UMAP regions and the boundary lines
    # appear only where data actually lives.
    boundary_xx = boundary_yy = boundary_labels = None
    if (
        (draw_cluster_boundaries or sampling_method == "spiral")
        and np.unique(cat_all).size >= 2
    ):
        boundary_xx, boundary_yy, boundary_labels = _knn_boundary_grid(
            x_all, y_all, cat_all,
            x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
            n_neighbors=knn_boundary_neighbors,
            grid_resolution=knn_boundary_resolution,
            density_smoothing_sigma=knn_boundary_density_smoothing_sigma,
            density_min_count=knn_boundary_density_min_count,
        )
        if np.all(np.isnan(boundary_labels)):
            boundary_labels = None
        else:
            label_lo = float(np.nanmin(boundary_labels))
            label_hi = float(np.nanmax(boundary_labels))
            contour_levels = np.arange(label_lo + 0.5, label_hi + 0.5, 1.0)

    picks_per_category: dict[int, pls.DataFrame] = {}
    # When ``sampling_method == 'spiral'`` we also record the spiral
    # path each category traces (xs, ys) so the plotter can overlay
    # the trajectory onto the main scatter.
    spiral_path_per_category: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cat in categories:
        cat_df = df_clean.filter(pls.col(cat_col) == cat)
        n = min(n_samples_per_category, cat_df.height)
        if n == 0:
            continue
        cat_pts = np.stack(
            [cat_df[x_col].to_numpy(), cat_df[y_col].to_numpy()], axis=1
        )
        if sampling_method == "random":
            picks_per_category[cat] = cat_df.sample(
                n=n, seed=int(rng.integers(0, 1_000_000))
            )
        elif sampling_method == "spiral":
            # Direct port of the original spiral sampler:
            # explicit cluster center (from QLVM provenance JSON or
            # caller-supplied dict) when available; otherwise the
            # arithmetic centroid of in-category data points. r_max
            # is the largest distance from THAT center to an
            # in-category point times ``spiral_radius_scale``, which
            # lets the caller shrink the visible spiral to a tighter
            # neighborhood of the cluster center instead of covering
            # the full cluster footprint.
            if int(cat) in cluster_centers_resolved:
                sp_cx, sp_cy = cluster_centers_resolved[int(cat)]
            else:
                sp_cx = float(np.mean(cat_pts[:, 0]))
                sp_cy = float(np.mean(cat_pts[:, 1]))
            dists_c = np.sqrt(
                (cat_pts[:, 0] - sp_cx) ** 2 + (cat_pts[:, 1] - sp_cy) ** 2
            )
            # Two ways to size the spiral:
            #   - ``spiral_radius_abs`` (when not None): a fixed radius
            #     in embedding-space units applied identically to every
            #     cluster -> uniform spirals.
            #   - otherwise: ``spiral_radius_scale × dists_c.max()``,
            #     which is per-cluster proportional (larger / more
            #     spread clusters get larger spirals).
            if spiral_radius_abs is not None:
                sp_r_max = float(spiral_radius_abs)
            else:
                sp_r_max = (
                    float(dists_c.max()) * float(spiral_radius_scale)
                    if dists_c.size else 0.1
                )
            local_idx, xs_path, ys_path = _pick_spiral_with_grid(
                cat_pts, n_per=n,
                cx0=sp_cx, cy0=sp_cy, r_max=sp_r_max,
                labels_grid=boundary_labels,
                xx=boundary_xx, yy=boundary_yy,
                cluster_label=int(cat),
                n_turns=int(spiral_n_turns),
                rng=rng if spiral_random_phase else None,
            )
            picks_per_category[cat] = cat_df[local_idx.tolist()]
            if xs_path.size:
                spiral_path_per_category[cat] = (xs_path, ys_path)
        else:
            local_idx = _pick_category_samples(
                cat_pts, n_per=n, method=sampling_method, rng=rng,
            )
            picks_per_category[cat] = cat_df[local_idx.tolist()]

    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    outer_width_ratios = (1.0, 1.5)
    outer = fig.add_gridspec(
        1, 2, width_ratios=list(outer_width_ratios), wspace=0.08,
    )
    right_region_w_fraction = outer_width_ratios[1] / sum(outer_width_ratios)
    # Left column: a vertical stack of 5 UMAPs. The top scatter is
    # the large category-colored map; below it sit two pairs of
    # smaller auxiliary maps. ``height_ratios=[2, 1, 1]`` gives the
    # top scatter twice the row-height of the two auxiliary rows.
    left_inner = outer[0, 0].subgridspec(
        3, 1, height_ratios=[2, 1, 1], hspace=0.18,
    )

    def _overlay_boundaries(ax, alpha=1.0, linewidth=2.5):
        if boundary_labels is None:
            return
        ax.contour(
            boundary_xx, boundary_yy, boundary_labels,
            levels=contour_levels,
            colors="#000000",
            linewidths=linewidth,
            alpha=alpha,
            zorder=4,
        )

    # Top: large category-colored scatter.
    ax_scatter = fig.add_subplot(left_inner[0, 0])
    for cat in categories:
        sub = scatter_df.filter(pls.col(cat_col) == cat)
        if sub.height == 0:
            continue
        ax_scatter.scatter(
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            c=category_colors[cat],
            s=scatter_point_size,
            alpha=scatter_point_alpha,
            edgecolors="none",
            rasterized=True,
        )
    _overlay_boundaries(ax_scatter, alpha=1.0, linewidth=2.5)

    # When using ``spiral`` sampling, draw the parametric spiral(s) on
    # the main scatter so the user can see the trajectory the picks
    # follow. The picker already returned the dense spiral coords
    # (xs_path, ys_path); plot them in ``spiral_color`` at
    # ``spiral_linewidth``, clipped to the data range. When
    # ``spiral_show_only_for`` is not None, restrict drawing to that
    # single category label so a chosen cluster's spiral stands out
    # instead of overlaying every cluster at once.
    if draw_spiral_overlay and sampling_method == "spiral" and spiral_path_per_category:
        if spiral_show_only_for is not None:
            cats_to_draw = [c for c in spiral_path_per_category if c == spiral_show_only_for]
        else:
            cats_to_draw = list(spiral_path_per_category.keys())
        for cat in cats_to_draw:
            xs_path, ys_path = spiral_path_per_category[cat]
            ax_scatter.plot(
                xs_path, ys_path,
                color=spiral_color,
                linewidth=spiral_linewidth, alpha=0.95, zorder=5,
            )

    # Number picks on the main scatter so the user can match a
    # spectrogram's upper-left "N" to the dot it came from.
    if annotate_picks_on_scatter:
        for picks in picks_per_category.values():
            if picks is None or picks.height == 0:
                continue
            xs = picks[x_col].to_numpy()
            ys = picks[y_col].to_numpy()
            for k, (px, py) in enumerate(zip(xs, ys), start=1):
                ax_scatter.text(
                    px, py, str(k),
                    fontsize=pick_number_fontsize, fontweight="bold",
                    color="#000000",
                    ha="center", va="center",
                    zorder=10,
                )

    # Overlay the integer cluster ID at the centre of each category
    # so the scatter doubles as a legend. Centres come from the
    # already-resolved ``cluster_centers_resolved`` map (caller-
    # supplied xy / h5 / json, in that priority); when none was
    # provided we fall back to the per-category mean of the points
    # actually plotted on the scatter so the label still lands
    # inside its cluster.
    if annotate_cluster_ids:
        if cluster_centers_resolved:
            cluster_id_centres = {
                int(cat_id): (float(cx), float(cy))
                for cat_id, (cx, cy) in cluster_centers_resolved.items()
                if int(cat_id) in set(int(c) for c in categories)
            }
        else:
            cluster_id_centres = {}
            for cat in categories:
                cat_mask = cat_all == cat
                if not np.any(cat_mask):
                    continue
                cluster_id_centres[int(cat)] = (
                    float(np.mean(x_all[cat_mask])),
                    float(np.mean(y_all[cat_mask])),
                )
        for cat_id, (cx, cy) in cluster_id_centres.items():
            ax_scatter.text(
                cx, cy, str(int(cat_id)),
                fontsize=cluster_id_fontsize, fontweight="bold",
                color="#000000",
                ha="center", va="center",
                zorder=20,
            )

    # QLVM is a quantized latent variable model, not UMAP -- label
    # accordingly.
    map_axis_token = "DIM" if map_prefix == "qlvm" else "UMAP"
    ax_scatter.set_xlabel(
        f"{map_prefix.upper()} {map_axis_token} 1", fontsize=12,
    )
    ax_scatter.set_ylabel(
        f"{map_prefix.upper()} {map_axis_token} 2", fontsize=12,
    )
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    # Match the four small panels: full box (all 4 spines visible) at
    # the same slightly-thicker line width.
    main_panel_spine_lw = 1.4
    for sp in ax_scatter.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(main_panel_spine_lw)
    # Pin the axes range to the data extent BEFORE the box-aspect
    # call so the spiral overlay (clip_on=True by default) gets
    # clipped to the data area instead of pushing the axes outward.
    ax_scatter.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax_scatter.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax_scatter.set_box_aspect(1)

    sex_male_color = "#9AC0CD"
    sex_female_color = "#FF6347"

    # The four small panels are roughly half the linear size of the
    # big top scatter (column split 1x2). With the SAME ``scatter_df``
    # of points packed into ~1/4 the axes area, marker density would
    # look 4x higher. Compensate by scaling marker AREA by ~1/4.
    small_panel_marker_size = max(0.5, scatter_point_size * 0.25)

    # Middle row: male / female emitter scatters.
    mid_grid = left_inner[1, 0].subgridspec(1, 2, wspace=0.08)
    ax_male = fig.add_subplot(mid_grid[0, 0])
    ax_female = fig.add_subplot(mid_grid[0, 1])

    if "sex" in scatter_df.columns:
        sex_arr = scatter_df["sex"].to_numpy()
        for ax, target, target_color in (
            (ax_male, "male", sex_male_color),
            (ax_female, "female", sex_female_color),
        ):
            target_mask = sex_arr == target
            ax.scatter(
                x_all[target_mask], y_all[target_mask],
                c=target_color,
                s=small_panel_marker_size,
                alpha=scatter_point_alpha,
                edgecolors="none",
                rasterized=True,
            )
    ax_male.set_xlabel("male emitted", fontsize=10)
    ax_female.set_xlabel("female emitted", fontsize=10)

    # Bottom row: duration / mean-frequency scatters colored by value
    # (project default cmap from `figures.cmap`). Points are sorted
    # ascending by value so high (bright) values are drawn last and sit
    # on top of low (dark) ones, giving a legible gradient even with
    # overplotting.
    bot_grid = left_inner[2, 0].subgridspec(1, 2, wspace=0.08)
    ax_dur = fig.add_subplot(bot_grid[0, 0])
    ax_freq = fig.add_subplot(bot_grid[0, 1])

    def _render_continuous(ax, values, label_template):
        finite = np.isfinite(values)
        if finite.sum() == 0:
            return
        lo, hi = np.percentile(values[finite], [2, 98])
        if lo == hi:
            hi = lo + 1.0
        v_finite = values[finite]
        order = np.argsort(v_finite)
        ax.scatter(
            x_all[finite][order],
            y_all[finite][order],
            c=v_finite[order],
            cmap=_GLOBAL_CMAP,
            vmin=lo,
            vmax=hi,
            s=small_panel_marker_size,
            alpha=scatter_point_alpha,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel(label_template.format(lo=lo, hi=hi), fontsize=10)

    if "duration" in scatter_df.columns:
        durs = scatter_df["duration"].to_numpy().astype(float) * 1000.0  # s -> ms
        _render_continuous(ax_dur, durs, "duration ({lo:.0f}-{hi:.0f} ms)")
    else:
        ax_dur.set_xlabel("duration (n/a)", fontsize=10)

    if "mean_freq_hz" in scatter_df.columns:
        freqs = scatter_df["mean_freq_hz"].to_numpy().astype(float) / 1000.0  # Hz -> kHz
        _render_continuous(ax_freq, freqs, "mean freq ({lo:.0f}-{hi:.0f} kHz)")
    else:
        ax_freq.set_xlabel("mean freq (n/a)", fontsize=10)

    # All four small panels get a full box (all 4 spines visible) at
    # slightly-thicker line width than the matplotlib default.
    small_panel_spine_lw = 1.4
    for ax in (ax_male, ax_female, ax_dur, ax_freq):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(small_panel_spine_lw)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_box_aspect(1)
        # Overlay cluster boundaries at lower alpha so each panel reads
        # as living inside the same UMAP geometry.
        _overlay_boundaries(ax, alpha=1.0, linewidth=1.5)

    # Right-side per-category x per-sample grid. ``tile_orientation``
    # picks whether each category is a horizontal ROW of thumbnails
    # (default) or a vertical COLUMN. The WHOLE thumbnail block is
    # sized to match the upper-left main scatter's footprint: block
    # width = block height = min(left_region_w_in, top_scatter_row_h_in).
    # Cells fill their slots (no set_box_aspect) so the specs end up
    # naturally spec-shaped -- wider than tall in vertical mode (10
    # rows / 7 cols), taller than wide in horizontal mode. Spacer
    # row/column appended on the bottom/right so the block sits at
    # the top-left of the right region.
    if tile_orientation not in ("horizontal", "vertical"):
        msg = (
            f"tile_orientation must be 'horizontal' or 'vertical'; "
            f"got {tile_orientation!r}."
        )
        raise ValueError(msg)
    if not (0.0 < thumbnail_size_fraction <= 1.0):
        msg = (
            f"thumbnail_size_fraction must be in (0, 1]; "
            f"got {thumbnail_size_fraction!r}."
        )
        raise ValueError(msg)
    if tile_orientation == "horizontal":
        grid_n_rows = n_categories
        grid_n_cols = n_samples_per_category
    else:
        grid_n_rows = n_samples_per_category
        grid_n_cols = n_categories

    left_region_w_in = fig_size[0] * (
        outer_width_ratios[0] / sum(outer_width_ratios)
    )
    right_region_w_in = fig_size[0] * right_region_w_fraction
    right_region_h_in = fig_size[1]
    top_row_h_in = fig_size[1] * (2.0 / 4.0)  # height_ratios [2, 1, 1]
    block_side_in = thumbnail_size_fraction * min(
        left_region_w_in, top_row_h_in,
    )
    block_w_in = block_side_in
    block_h_in = block_side_in
    cell_w_in = block_w_in / grid_n_cols
    cell_h_in = block_h_in / grid_n_rows

    width_ratios = [1.0] * grid_n_cols
    n_extra_cols = 0
    width_excess = right_region_w_in - block_w_in
    if width_excess > 0:
        width_ratios = [1.0] * grid_n_cols + [width_excess / cell_w_in]
        n_extra_cols = 1

    height_ratios = [1.0] * grid_n_rows
    n_extra_rows = 0
    height_excess = right_region_h_in - block_h_in
    if height_excess > 0:
        height_ratios = [1.0] * grid_n_rows + [height_excess / cell_h_in]
        n_extra_rows = 1

    inner = outer[0, 1].subgridspec(
        grid_n_rows + n_extra_rows, grid_n_cols + n_extra_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=thumbnail_wspace, hspace=thumbnail_hspace,
    )

    # ``(cat_idx, sample_idx)`` -> ``(grid_row, grid_col)`` dispatch.
    # Horizontal: each category is a row, sample index walks columns.
    # Vertical:   each category is a column, sample index walks rows.
    def _grid_pos(cat_idx: int, sample_idx: int) -> tuple[int, int]:
        if tile_orientation == "horizontal":
            return cat_idx, sample_idx
        return sample_idx, cat_idx

    with h5py.File(consolidated_h5_path, "r") as h5:
        mask_index_cache: dict[str, np.ndarray] = {}

        def _strip_ax_chrome(ax: plt.Axes) -> None:
            """Hide ticks + every spine + the axes frame so no
            rectangular outline is ever drawn around a thumbnail.
            ``set_frame_on(False)`` is defensive on top of the per-
            spine ``set_visible(False)`` calls in case any draw path
            leaves the frame visible."""
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_frame_on(False)

        # Pre-pass for ``unstretched_specs``: walk every pick across
        # every category, read its stored ``durations`` value, and use
        # the maximum as the shared x-axis upper bound. Anchoring to
        # the actual max-displayed dur (rather than the storage cap
        # of 128) means the longest USV in the figure fills its slot
        # horizontally and every other USV scales proportionally,
        # which keeps the right-side whitespace bounded instead of
        # crushing 95% of calls into ~25% of slot width.
        unstretched_x_max = None
        if unstretched_specs:
            max_dur = 1
            for cat in categories:
                picks = picks_per_category.get(cat)
                if picks is None:
                    continue
                for row in picks.iter_rows(named=True):
                    sess = str(row["session_id"])
                    spec_idx = int(row["row_index"])
                    spec_group_key = f"spectrogram/{sess}"
                    if spec_group_key not in h5:
                        continue
                    grp = h5[spec_group_key]
                    n_time_max = int(grp["spectrograms"].shape[2])
                    d = int(grp["durations"][spec_idx])
                    d = max(1, min(d, n_time_max))
                    if d > max_dur:
                        max_dur = d
            unstretched_x_max = float(max_dur)

        for cat_idx, cat in enumerate(categories):
            picks = picks_per_category.get(cat)
            if picks is None:
                for sample_idx in range(n_samples_per_category):
                    g_row, g_col = _grid_pos(cat_idx, sample_idx)
                    ax = fig.add_subplot(inner[g_row, g_col])
                    _strip_ax_chrome(ax)
                continue

            this_mask = apply_mask and (cat not in mask_excluded_categories)

            picks_rows = list(picks.iter_rows(named=True))
            for sample_idx in range(n_samples_per_category):
                g_row, g_col = _grid_pos(cat_idx, sample_idx)
                ax = fig.add_subplot(inner[g_row, g_col])
                if sample_idx >= len(picks_rows):
                    _strip_ax_chrome(ax)
                    continue
                row = picks_rows[sample_idx]
                sess = str(row["session_id"])
                spec_idx = int(row["row_index"])
                spec_group_key = f"spectrogram/{sess}"
                if spec_group_key not in h5:
                    _strip_ax_chrome(ax)
                    continue
                grp = h5[spec_group_key]
                spec = grp["spectrograms"][spec_idx, :, :].astype(np.float32)
                dur = int(grp["durations"][spec_idx])
                dur = max(1, min(dur, spec.shape[1]))
                spec_valid = spec[:, :dur]

                if this_mask:
                    mask_group_key = f"mask/{sess}"
                    if mask_group_key in h5:
                        mask_grp = h5[mask_group_key]
                        if sess not in mask_index_cache:
                            mask_index_cache[sess] = mask_grp["spectrogram_index"][:]
                        si = mask_index_cache[sess]
                        matching = np.where(si == spec_idx)[0]
                        if matching.size > 0:
                            masks = mask_grp["segmentations"][matching, :, :dur]
                            combined = np.any(masks, axis=0)
                            spec_valid = spec_valid * combined.astype(np.float32)

                if unstretched_specs:
                    # Pad every spec to a common ``(n_freq, max_dur)``
                    # shape by surrounding the valid time slice with
                    # zeros split evenly left/right, so the call sits
                    # centered horizontally and the array fed to
                    # ``imshow`` has identical dimensions for every
                    # thumbnail. Combined with ``aspect="auto"`` and
                    # the uniform gridspec slots, this gives the
                    # truly identical-shape thumbnails the caller
                    # asked for: longest USV fills the slot, shorter
                    # ones get progressively wider zero borders.
                    n_freq = int(spec_valid.shape[0])
                    w_target = int(unstretched_x_max)
                    pad_total = max(0, w_target - dur)
                    pad_left = pad_total // 2
                    spec_padded = np.zeros(
                        (n_freq, w_target), dtype=spec_valid.dtype,
                    )
                    spec_padded[:, pad_left:pad_left + dur] = spec_valid
                    ax.imshow(
                        spec_padded, origin="lower", aspect="auto",
                        cmap=_GLOBAL_CMAP, vmin=0.0, vmax=1.0,
                        interpolation="nearest",
                    )
                else:
                    ax.imshow(
                        spec_valid, origin="lower", aspect="auto",
                        cmap=_GLOBAL_CMAP, vmin=0.0, vmax=1.0,
                        interpolation="nearest",
                    )
                # White N in the upper-left corner of each thumbnail
                # so the user can match the same N drawn at the
                # picked dot's position in the main scatter.
                ax.text(
                    0.04, 0.95, str(sample_idx + 1),
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=7, fontweight="bold",
                    color="#FFFFFF",
                )
                _strip_ax_chrome(ax)

    fig.tight_layout()

    if output_path is not None:
        out_path = pathlib.Path(configure_path(output_path))
        if fig_format is not None:
            out_path = out_path.with_suffix(f".{fig_format.lstrip('.')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=fig_dpi)
        message_output(f"Saved embedding+thumbnails figure: {out_path}")

    return fig


def render_embedding_thumbnails_for_cohort(
    visualizations_parameter_dict: dict,
    message_output: Callable | None = None,
) -> plt.Figure:
    """
    Description
    -----------
    Cohort-level driver for the embedding + per-category spectrogram thumbnails
    figure. Pools every ``*sessions_list.txt`` under
    ``usv_embedding['input_files_directory']`` into one combined session list,
    resolves the consolidated store as the newest ``spectrograms_*.h5`` under
    ``shared_resources['spectrograms_dir']``, and renders
    ``plot_embedding_with_category_thumbnails`` with the knobs from the
    ``embedding_thumbnails`` settings block. The figure is written to
    ``figures['save_directory']``.

    Like the QLVM torus video this reads its inputs from settings rather than from
    a session directory, so it runs ONCE outside the per-session visualization
    loop (dispatched from ``visualize_data`` when
    ``make_embedding_thumbnails_bool`` is set).

    Parameters
    ----------
    visualizations_parameter_dict (dict)
        The full visualizations settings dict, carrying the ``figures``,
        ``shared_resources``, ``usv_embedding`` and ``embedding_thumbnails``
        blocks.
    message_output (Callable | None)
        Progress / diagnostic sink. Defaults to the built-in ``print``.

    Returns
    -------
    fig (plt.Figure)
        The rendered two-panel figure (also saved to ``figures['save_directory']``).
    """

    log = message_output or print
    cfg = visualizations_parameter_dict["embedding_thumbnails"]
    figures = visualizations_parameter_dict["figures"]

    input_files_dir = pathlib.Path(
        configure_path(visualizations_parameter_dict["usv_embedding"]["input_files_directory"])
    )
    store_path = resolve_consolidated_h5_path(
        visualizations_parameter_dict["shared_resources"]["spectrograms_dir"]
    )

    # Pool every cohort session list into one deduplicated combined list (same
    # cohort definition as the embedding explorer / VAE-density precompute).
    list_files = sorted(input_files_dir.glob("*sessions_list.txt"))
    if not list_files:
        raise FileNotFoundError(
            f"embedding thumbnails: no '*sessions_list.txt' under '{input_files_dir}'."
        )
    roots, seen = [], set()
    for list_file in list_files:
        for line in list_file.read_text().splitlines():
            root = line.strip()
            if root and not root.startswith("#") and root not in seen:
                seen.add(root)
                roots.append(root)
    log(f"[embedding-thumbnails] pooled {len(roots)} session roots from {len(list_files)} list(s).")

    with tempfile.NamedTemporaryFile(
        "w", suffix="_embedding_thumbnails_sessions.txt", delete=False
    ) as combined_file:
        combined_file.write("\n".join(roots))
        combined_sessions_txt = combined_file.name

    # Cluster-center provenance for the QLVM cluster-ID labels / spiral centers:
    # the newest qlvm_clusters_*.h5 under the spectrograms dir (it carries the
    # /coarse + /fine cluster_centers). Only meaningful for the QLVM map; the VAE
    # umap has no equivalent, so leave it unset there (centers fall back to the
    # data-derived medoids/centroids).
    cluster_centers_h5_path = None
    if cfg["map_type"] == "qlvm":
        spec_base = pathlib.Path(
            configure_path(visualizations_parameter_dict["shared_resources"]["spectrograms_dir"])
        )
        cc_matches = sorted(spec_base.glob("qlvm_clusters_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cc_matches:
            cluster_centers_h5_path = str(cc_matches[0])

    cache_path = configure_path(cfg["embeddings_cache_path"]) if cfg["embeddings_cache_path"] else None

    out_dir = pathlib.Path(configure_path(figures["save_directory"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_format = figures["fig_format"]
    output_path = str(
        out_dir / f"embedding_thumbnails_{cfg['map_type']}_{cfg['category_col_suffix']}.{fig_format}"
    )

    return plot_embedding_with_category_thumbnails(
        sessions_txt_path=combined_sessions_txt,
        consolidated_h5_path=store_path,
        map_type=cfg["map_type"],
        category_col_suffix=cfg["category_col_suffix"],
        n_samples_per_category=cfg["n_samples_per_category"],
        apply_mask=cfg["apply_mask"],
        mask_excluded_categories=tuple(cfg["mask_excluded_categories"]),
        category_colors=cfg["category_colors"],
        sampling_method=cfg["sampling_method"],
        cluster_centers_h5_path=cluster_centers_h5_path,
        draw_spiral_overlay=cfg["draw_spiral_overlay"],
        spiral_show_only_for=cfg["spiral_show_only_for"],
        spiral_color=cfg["spiral_color"],
        spiral_linewidth=cfg["spiral_linewidth"],
        spiral_radius_scale=cfg["spiral_radius_scale"],
        spiral_radius_abs=cfg["spiral_radius_abs"],
        spiral_n_turns=cfg["spiral_n_turns"],
        spiral_random_phase=cfg["spiral_random_phase"],
        draw_cluster_boundaries=cfg["draw_cluster_boundaries"],
        knn_boundary_neighbors=cfg["knn_boundary_neighbors"],
        knn_boundary_resolution=cfg["knn_boundary_resolution"],
        knn_boundary_density_min_count=cfg["knn_boundary_density_min_count"],
        knn_boundary_density_smoothing_sigma=cfg["knn_boundary_density_smoothing_sigma"],
        annotate_picks_on_scatter=cfg["annotate_picks_on_scatter"],
        pick_number_fontsize=cfg["pick_number_fontsize"],
        annotate_cluster_ids=cfg["annotate_cluster_ids"],
        cluster_id_fontsize=cfg["cluster_id_fontsize"],
        tile_orientation=cfg["tile_orientation"],
        thumbnail_size_fraction=cfg["thumbnail_size_fraction"],
        thumbnail_hspace=cfg["thumbnail_hspace"],
        thumbnail_wspace=cfg["thumbnail_wspace"],
        unstretched_specs=cfg["unstretched_specs"],
        scatter_max_points=cfg["scatter_max_points"],
        fig_size=tuple(cfg["fig_size"]),
        fig_dpi=figures["dpi"],
        output_path=output_path,
        fig_format=fig_format,
        embeddings_cache_path=cache_path,
        seed=figures["seed"],
        message_output=message_output,
    )
