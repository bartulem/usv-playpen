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
    times. Gaps between USVs are zero. Requires
    ``cfg['consolidated_spectrograms_h5']`` to point at the store.

Audio is read from the session's ``*_int16.mmap*`` file (the canonical
concatenated multi-channel int16 memmap), and spectrograms are computed
with ``librosa.stft`` over the user-specified time window. All rendering
parameters live in the ``make_usv_spectrograms`` block of
``visualizations_settings.json``.
"""

from __future__ import annotations

import pathlib

from collections.abc import Callable

import h5py
import librosa
import librosa.display
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d, zoom
from scipy.signal.windows import tukey
from sklearn.neighbors import KNeighborsClassifier

from ..os_utils import configure_path, first_match_or_raise
from ..time_utils import is_gui_context

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
            Forwarded as-is to ``self.__dict__``. Expected keys:
            ``root_directory``, ``visualizations_parameter_dict``,
            ``message_output``.

        Returns
        -------
        None
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

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
        that takes precedence over the string name in the
        ``make_usv_spectrograms.spectrogram_cmap`` settings entry.
        Either form is accepted by matplotlib's ``imshow`` /
        ``specshow`` ``cmap=`` argument.

        Returns
        -------
        cmap (str | matplotlib.colors.Colormap)
            The colormap object or name to pass to ``imshow``.
        """

        if getattr(self, "cmap_override", None) is not None:
            return self.cmap_override
        return self.visualizations_parameter_dict["make_usv_spectrograms"][
            "spectrogram_cmap"
        ]

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
        channel_num = int(file_basename.split("_")[-2])
        sample_num = int(file_basename.split("_")[-3])
        sampling_rate = int(file_basename.split("_")[-4])

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

        xtick_locs = ax.get_xticks()
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
            save_dir = str(pathlib.Path(self.root_directory) / "audio")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        start_time_sec = float(cfg["time_window"][0])
        end_time_sec = float(cfg["time_window"][1])
        prefix = file_basename.split("_int16.mmap", maxsplit=1)[0]
        filename = (
            f"usv_spectrogram_{prefix}_{suffix}_"
            f"from_{start_time_sec}s_to_{end_time_sec}s.{cfg['fig_format']}"
        )
        out_path = pathlib.Path(save_dir) / filename
        fig.savefig(
            out_path,
            dpi=cfg["fig_dpi"],
            transparent=cfg["transparent_fig_bg"],
        )
        self.message_output(f"Saved spectrogram figure: {out_path}")

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

    def plot_stitched(self) -> plt.Figure:
        """
        Description
        -----------
        Render a session-timeline spectrogram by stitching the
        pre-computed per-USV averaged spectrograms from the
        consolidated HDF5 store (path in
        ``cfg['consolidated_spectrograms_h5']``) into a zero canvas at
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

        hop_length = cfg["nfft"] // 4
        canvas_fps = float(sampling_rate) / float(hop_length)
        canvas_n_bins = max(1, int(round((end_time_sec - start_time_sec) * canvas_fps)))

        h5_path = configure_path(cfg["consolidated_spectrograms_h5"])
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

        stitched_vmin = 0.0
        stitched_vmax = 1.0

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=tuple(cfg["fig_size"]), dpi=cfg["fig_dpi"]
        )
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
        ax.set_title(
            f"Stitched spectrogram ({in_window_df.height} USVs in window, "
            f"session {session_key})"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_yticks([freq_bins_cropped[0], freq_bins_cropped[-1]])
        ax.set_yticklabels(
            [
                f"{freq_bins_cropped[0] / 1000:.0f}",
                f"{freq_bins_cropped[-1] / 1000:.0f}",
            ]
        )
        ax.tick_params(axis="y", length=0)

        if cfg["plot_cbar"]:
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
            cbar.set_ticks([stitched_vmin, stitched_vmax])
            cbar.set_ticklabels(
                [f"{stitched_vmin:.0f}", f"{stitched_vmax:.0f}"]
            )
            cbar.ax.tick_params(length=0)
            cbar.ax.minorticks_off()
            cbar.set_label("Normalized amplitude")

        layout_rect = (0.0, 0.0, CBAR_RIGHT_RESERVE, 1.0) if cfg["plot_cbar"] else None
        fig.tight_layout(rect=layout_rect)
        self._save_figure(fig, "stitched", file_basename)
        return fig

    def make_usv_spectrograms(self) -> plt.Figure:
        """
        Description
        -----------
        Dispatch to the rendering method named by
        ``make_usv_spectrograms.mode``: ``"single"`` →
        ``plot_single_channel``; ``"all"`` → ``plot_all_channels``;
        ``"stitched"`` → ``plot_stitched``. This is the entry point
        that the visualization pipeline (``visualize_data.py``) is
        expected to call once a ``make_usv_spectrograms_bool`` toggle
        has been wired through.

        Parameters
        ----------

        Returns
        -------
        fig (plt.Figure)
            The rendered figure (also written to disk if ``save_fig``
            is True).
        """

        mode = self.visualizations_parameter_dict["make_usv_spectrograms"]["mode"]
        if mode == "single":
            return self.plot_single_channel()
        if mode == "all":
            return self.plot_all_channels()
        if mode == "stitched":
            return self.plot_stitched()
        msg = (
            f"Unknown make_usv_spectrograms.mode={mode!r}; "
            f"expected one of 'single', 'all', 'stitched'."
        )
        raise ValueError(msg)


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
    "qlvm_umap1",
    "qlvm_umap2",
)
EMBEDDING_LABEL_COLS = (
    "vae_category",
    "vae_supercategory",
    "qlvm_category",
    "qlvm_supercategory",
)
EMBEDDING_ALL_COLS = EMBEDDING_COORD_COLS + EMBEDDING_LABEL_COLS

# Extra per-USV columns pulled into the pooled embeddings DataFrame.
# They power the auxiliary scatters (sex, duration, mean frequency) in
# ``plot_umap_with_category_thumbnails``. The cache file is invalidated
# automatically when these columns are missing -- see schema-check
# logic in ``build_pooled_embeddings_df``.
EMBEDDING_EXTRA_COLS = ("emitter", "duration", "mean_freq_hz")


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
    (``vae_umap1/2``, ``qlvm_umap1/2``), the four label columns
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
            qlvm_umap1, qlvm_umap2 (Float64)
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
    required_extra_cols = {"sex", "duration", "mean_freq_hz"}
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
            df = pls.read_csv(str(csv_path), columns=select_cols)
        except pls.exceptions.ColumnNotFoundError:
            df = pls.read_csv(str(csv_path))
        except (OSError, IOError) as exc:
            message_output(f"[skip] {csv_path}: {exc}")
            continue

        df = df.with_row_index(name="row_index")
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
        for c in ("sex", "duration", "mean_freq_hz"):
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

    if cache_p is not None:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        pooled.write_parquet(str(cache_p))
        message_output(
            f"Cached pooled embeddings DF to {cache_p} "
            f"({pooled.height:,} rows, {pooled.select('session_id').unique().height} sessions)"
        )

    return pooled


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
    - ``"farthest_point"``: iterative farthest-point sampling starting
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

    if method == "farthest_point":
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
        "Choose from 'random', 'nearest', 'farthest_point', 'grid', 'spiral'."
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


def plot_umap_with_category_thumbnails(
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
    annotate_picks_on_scatter: bool = True,
    pick_number_fontsize: float = 11.0,
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

    - **Left panel**: UMAP scatter of every (non-noise) USV across the
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
        ``"vae"`` or ``"qlvm"`` - selects which UMAP map to plot.
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
        ``<map>_umap1/2`` and the category column.
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

    x_col = f"{map_prefix}_umap1"
    y_col = f"{map_prefix}_umap2"
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
        import json as _json
        with open(configure_path(cluster_centers_json_path)) as _f:
            _prov = _json.load(_f)
        for i, c in enumerate(_prov.get("cluster_centers", [])):
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
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.5], wspace=0.08,
    )
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
            )
    ax_male.set_xlabel("male emitted", fontsize=10)
    ax_female.set_xlabel("female emitted", fontsize=10)

    # Bottom row: duration / mean-frequency scatters colored by value
    # (inferno colormap). Points are sorted ascending by value so high
    # (bright) values are drawn last and sit on top of low (dark) ones,
    # giving a legible gradient even with overplotting.
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
            cmap="inferno",
            vmin=lo,
            vmax=hi,
            s=small_panel_marker_size,
            alpha=scatter_point_alpha,
            edgecolors="none",
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

    inner = outer[0, 1].subgridspec(
        n_categories, n_samples_per_category, wspace=0.05, hspace=0.18,
    )

    row_axes_per_category: dict[int, list[plt.Axes]] = {cat: [] for cat in categories}

    with h5py.File(consolidated_h5_path, "r") as h5:
        mask_index_cache: dict[str, np.ndarray] = {}

        for row_idx, cat in enumerate(categories):
            picks = picks_per_category.get(cat)
            if picks is None:
                for col_idx in range(n_samples_per_category):
                    ax = fig.add_subplot(inner[row_idx, col_idx])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    row_axes_per_category[cat].append(ax)
                continue

            this_mask = apply_mask and (cat not in mask_excluded_categories)

            picks_rows = list(picks.iter_rows(named=True))
            for col_idx in range(n_samples_per_category):
                ax = fig.add_subplot(inner[row_idx, col_idx])
                row_axes_per_category[cat].append(ax)
                if col_idx >= len(picks_rows):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    continue
                row = picks_rows[col_idx]
                sess = str(row["session_id"])
                spec_idx = int(row["row_index"])
                spec_group_key = f"spectrogram/{sess}"
                if spec_group_key not in h5:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
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

                ax.imshow(
                    spec_valid, origin="lower", aspect="auto",
                    cmap="inferno", vmin=0.0, vmax=1.0,
                    interpolation="nearest",
                )
                # White N in the upper-left corner of each thumbnail
                # so the user can match the same N drawn at the
                # picked dot's position in the main scatter.
                ax.text(
                    0.04, 0.95, str(col_idx + 1),
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=7, fontweight="bold",
                    color="#FFFFFF",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

    fig.canvas.draw()
    rect_pad = 0.004
    for cat, axes_list in row_axes_per_category.items():
        if not axes_list:
            continue
        poses = [ax.get_position() for ax in axes_list]
        x0 = min(p.x0 for p in poses) - rect_pad
        y0 = min(p.y0 for p in poses) - rect_pad
        x1 = max(p.x1 for p in poses) + rect_pad
        y1 = max(p.y1 for p in poses) + rect_pad
        rect = mpatches.Rectangle(
            (x0, y0), (x1 - x0), (y1 - y0),
            transform=fig.transFigure,
            fill=False,
            edgecolor=category_colors[cat],
            linewidth=2.5,
            clip_on=False,
        )
        fig.patches.append(rect)

    fig.tight_layout()

    if output_path is not None:
        out_path = pathlib.Path(configure_path(output_path))
        if fig_format is not None:
            out_path = out_path.with_suffix(f".{fig_format.lstrip('.')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        message_output(f"Saved UMAP+thumbnails figure: {out_path}")

    return fig
