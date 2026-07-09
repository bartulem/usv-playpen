"""
@author: bartulem
Run USV inference on WAV files and create annotations.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from tqdm import tqdm

from ..os_utils import configure_path, first_match_or_raise, wait_for_subprocesses
from ..time_utils import is_gui_context, smart_wait
from ..visualizations.figure_io import save_figure
from ..visualizations.plot_style import apply_plot_style
from ..yaml_utils import load_session_metadata, save_session_metadata

apply_plot_style()


# DAS annotation filenames are produced by `das predict` from the input WAV's
# basename, so they reach us as `<device>_<...>_<chXX>_<...>annotations.csv`,
# e.g. `m_260421185826_ch01_cropped_to_video_hpss_filtered_annotations.csv`,
# where <device> is 'm' (master) or 's' (slave) and <chXX> is the two-digit
# channel index. Both the timestamp segment before <chXX> and the
# pipeline-suffix segment after it ('cropped_to_video_hpss_filtered') contain
# underscores, so the regex anchors only on the device prefix and the channel
# token and tolerates any intervening/trailing content (the trailing `.*` is
# what lets the channel sit anywhere before `annotations.csv`, not just
# immediately before it).
_DAS_ANNOTATION_FILE_RE = re.compile(r"^([ms])_.*_(ch\d{2})_.*annotations\.csv$")


def _write_usv_summary_csv(merged: list, out_path: pathlib.Path) -> None:
    """Write the per-session USV summary CSV from a list of merged interval dicts.

    Single source of truth for the summary schema, shared by all three
    ``summarize_das_findings`` branches (noise-filtered with >1 USV, the single-USV
    case, and filtering-disabled) so the column set / formatting can never drift
    between them.

    Parameters
    ----------
    merged : list
        List of merged USV interval dicts, each carrying ``start``/``stop``/
        ``peak_amp_ch``/``mean_amp_ch``/``chs_count``/``chs_detected`` keys.
    out_path : pathlib.Path
        Destination path for the ``*_usv_summary.csv`` file.

    Returns
    -------
    (None)
    """
    pls.DataFrame({
        "usv_id": [f"{_num:04d}" for _num in range(len(merged))],
        "start": [u['start'] for u in merged],
        "stop": [u['stop'] for u in merged],
        "duration": [u['stop'] - u['start'] for u in merged],
        "peak_amp_ch": [float(u['peak_amp_ch']) for u in merged],
        "mean_amp_ch": [float(u['mean_amp_ch']) for u in merged],
        "chs_count": [float(u['chs_count']) for u in merged],
        "chs_detected": [str(u['chs_detected']) for u in merged],
        "emitter": [None] * len(merged),
    }).write_csv(file=out_path)


class FindMouseVocalizations:
    def __init__(
        self,
        root_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the FindMouseVocalizations class.

        Parameters
        ----------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Function to output messages; defaults to None.

        Returns
        -------
        None
        """

        if input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent.parent / "_parameter_settings/processing_settings.json"
            ) as json_file:
                _defaults = json.load(json_file)
            self.input_parameter_dict = _defaults["usv_inference"]["FindMouseVocalizations"]
        else:
            self.input_parameter_dict = input_parameter_dict["usv_inference"]["FindMouseVocalizations"]

        self.root_directory = root_directory
        self.message_output = message_output or print

        self.app_context_bool = is_gui_context()

    def das_command_line_inference(self) -> None:
        """
        Description
        -----------
        This method takes WAV files as input and runs DAS inference on them to generate
        tentative USV segments in the recording.

        Parameters
        ----------

        Returns
        -------
        .csv annotation files
            CSV files w/ onsets and offsets of all detected USV segments,
            shape: (N_USV, VOC_TYPE, START_SEC, END_SEC).
        """

        self.message_output(
            f"DAS inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}. Please be patient, this can take >5 min/file."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        das_conda_name = self.input_parameter_dict["das_command_line_inference"][
            "das_conda_env_name"
        ]
        model_base = str(pathlib.Path(configure_path(self.input_parameter_dict['das_command_line_inference']['das_model_directory'])) / self.input_parameter_dict['das_command_line_inference']['model_name_base'])
        thresh = self.input_parameter_dict["das_command_line_inference"][
            "segment_confidence_threshold"
        ]
        min_len = self.input_parameter_dict["das_command_line_inference"][
            "segment_minlen"
        ]
        fill_gap = self.input_parameter_dict["das_command_line_inference"][
            "segment_fillgap"
        ]
        save_format = self.input_parameter_dict["das_command_line_inference"][
            "output_file_type"
        ]

        hpss_dir = pathlib.Path(self.root_directory) / "audio" / "hpss_filtered"

        # run inference
        for one_file in sorted(hpss_dir.glob("*.wav*")):
            self.message_output(
                f"Running DAS inference on: {one_file.name}"
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            conda_exe = os.environ.get('CONDA_EXE', 'conda')
            clean_env = os.environ.copy()
            clean_env.pop('PYTHONHOME', None)
            inference_subp = subprocess.Popen(
                args=[conda_exe, 'run', '--no-capture-output', '-n', das_conda_name, 'das', 'predict', one_file, model_base,
                      '--segment-thres', str(thresh), '--segment-minlen', str(min_len),
                      '--segment-fillgap', str(fill_gap), '--save-format', str(save_format)],
                cwd=hpss_dir,
                env=clean_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                shell=False,
            )

            # DAS inference on a long recording can take hours on CPU, so the
            # per-file budget is generous. A 12 h ceiling still catches a
            # genuinely hung process (GPU lost, file-descriptor deadlock)
            # rather than letting it sit indefinitely.
            wait_for_subprocesses(
                subps=[inference_subp],
                max_seconds=12 * 60 * 60,
                label=f"DAS inference on {pathlib.Path(one_file).name}",
                poll_interval_s=5,
                message_output=self.message_output,
                raise_on_nonzero=False,
                raise_on_timeout=False,
            )

        # create save directory if it doesn't exist
        das_dir = pathlib.Path(self.root_directory) / "audio" / "das_annotations"
        das_dir.mkdir(parents=True, exist_ok=True)

        # move annotation files to save directory
        # NB: materialize the directory listing before moving — moving entries
        # out of `hpss_dir` while its iterator is live can skip files on some
        # filesystems. The suffix is matched with `endswith` (not a substring)
        # so only true `.{save_format}` outputs are moved, never a name that
        # merely contains that token.
        for one_file in sorted(hpss_dir.iterdir()):
            if one_file.name.endswith(f".{save_format}"):
                shutil.move(src=one_file, dst=das_dir / one_file.name)

    def summarize_das_findings(self) -> None:
        """
        Description
        -----------
        This method takes CSV files generated by DAS inference and creates a summary file,
        containing information about individual USV segment start and stop times, duration,
        peak amplitude channel, mean amplitude channel, total number of channels it was
        detected on, list of channels it was detected on, and emitter ID.

        Parameters
        ----------

        Returns
        -------
        .csv summary file
            CSV file w/ information about all detected USV segments,
            shape: (N_USV, USV_ID, START, STOP, DURATION, PEAK_AMP_CH,
            MEAN_AMP_CH, CHS_COUNT, CHS_DETECTED, EMITTER).
        """

        self.message_output(
            f"DAS summary started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        ch_conversion_dict = {
            "m_ch01": 0,
            "m_ch02": 1,
            "m_ch03": 2,
            "m_ch04": 3,
            "m_ch05": 4,
            "m_ch06": 5,
            "m_ch07": 6,
            "m_ch08": 7,
            "m_ch09": 8,
            "m_ch10": 9,
            "m_ch11": 10,
            "m_ch12": 11,
            "s_ch01": 12,
            "s_ch02": 13,
            "s_ch03": 14,
            "s_ch04": 15,
            "s_ch05": 16,
            "s_ch06": 17,
            "s_ch07": 18,
            "s_ch08": 19,
            "s_ch09": 20,
            "s_ch10": 21,
            "s_ch11": 22,
            "s_ch12": 23,
        }

        session_id = pathlib.Path(self.root_directory).name

        annot_dir = pathlib.Path(self.root_directory) / "audio" / "das_annotations"
        das_annotation_files = sorted(annot_dir.glob("*.csv")) if annot_dir.is_dir() else []
        if not das_annotation_files:
            self.message_output(
                f"No DAS annotations found in directory: {self.root_directory}. Skipping summary generation."
            )
            return

        try:
            # Phase 1: load all channel CSVs, filter noise, collect flat segment list
            # Each entry is (start_seconds, stop_seconds, channel_numeric_index).
            all_segments = []
            for one_file in das_annotation_files:
                m = _DAS_ANNOTATION_FILE_RE.match(one_file.name)
                if m is None:
                    self.message_output(
                        f"Skipping {one_file.name}: filename does not match expected "
                        f"DAS annotation pattern '<device>_..._<chXX>_...annotations.csv'."
                    )
                    continue
                file_id = f"{m.group(1)}_{m.group(2)}"
                if file_id not in ch_conversion_dict:
                    self.message_output(
                        f"Skipping {one_file.name}: unrecognized device/channel '{file_id}'."
                    )
                    continue
                channel_df = pls.read_csv(source=str(one_file))
                channel_df = channel_df.filter(pls.col("name") != "noise")
                ch_num = ch_conversion_dict[file_id]
                for seg_row in channel_df.iter_rows(named=True):
                    all_segments.append(
                        (seg_row["start_seconds"], seg_row["stop_seconds"], ch_num)
                    )

            # Phase 2: sort all segments by start time
            all_segments.sort(key=lambda seg: seg[0])

            # Phase 3: greedy interval merge across all channels
            # Two intervals (a_start, a_stop) and (b_start, b_stop) overlap (open-ended) when:
            #   a_start < b_stop  and  b_start < a_stop
            # Since segments are sorted by start, only the running stop needs comparing.
            # Each merged entry is a dict with start, stop, chs_detected (set), and placeholder fields.
            merged = []
            for start, stop, ch_idx in all_segments:
                if merged and start < merged[-1]['stop']:
                    # overlaps with current merged interval: extend and record channel
                    merged[-1]['stop'] = max(merged[-1]['stop'], stop)
                    merged[-1]['chs_detected'].add(ch_idx)
                else:
                    # no overlap: start a new merged interval
                    merged.append({
                        'start': start,
                        'stop': stop,
                        'chs_detected': {ch_idx},
                        'peak_amp_ch': 0.0,
                        'mean_amp_ch': 0.0,
                    })

            # Convert channel sets to sorted lists and compute counts
            for usv in merged:
                usv['chs_detected'] = sorted(usv['chs_detected'])
                usv['chs_count'] = len(usv['chs_detected'])

            n_usv = len(merged)
            self.message_output(
                f"Merged {n_usv} USV intervals from {len(all_segments)} raw detections across {len(das_annotation_files)} channels."
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            # Whether to run the Phase-4 putative-noise rejection (amplitude +
            # spectrogram correlation/variance checks). When False, every merged
            # interval is kept and written to the summary CSV as-is.
            filter_putative_noise_bool = self.input_parameter_dict[
                "summarize_das_findings"
            ]["filter_putative_noise_bool"]

            # Phase 4: amplitude + spectrogram quality checks
            # (skipped entirely when filter_putative_noise_bool is False)
            if filter_putative_noise_bool and n_usv > 1:
                audio_file_loc = first_match_or_raise(
                    root=pathlib.Path(self.root_directory) / "audio" / "hpss_filtered",
                    pattern="*.mmap",
                    label="concatenated audio mmap",
                )
                audio_file_name = audio_file_loc.name
                # The mmap filename encodes its array metadata as the last four
                # underscore-separated tokens, in the trailing layout
                # '..._<sampling_rate>_<sample_num>_<channel_num>_<dtype>.mmap'.
                # Parsing right-to-left: [-1][:-5] is the dtype with the trailing
                # '.mmap' (5 chars) stripped, [-2] the channel count, [-3] the
                # sample count, [-4] the sampling rate.
                data_type, channel_num, sample_num, audio_sampling_rate = (
                    audio_file_name.split("_")[-1][:-5],
                    int(audio_file_name.split("_")[-2]),
                    int(audio_file_name.split("_")[-3]),
                    int(audio_file_name.split("_")[-4]),
                )
                audio_file_data = np.memmap(
                    filename=audio_file_loc,
                    mode="r",
                    dtype=data_type,
                    shape=(sample_num, channel_num),
                )

                len_win_signal = self.input_parameter_dict["summarize_das_findings"][
                    "len_win_signal"
                ]
                low_freq_cutoff = self.input_parameter_dict["summarize_das_findings"][
                    "low_freq_cutoff"
                ]
                frequency_resolution = audio_sampling_rate / len_win_signal
                lower_bin = int(np.floor(low_freq_cutoff / frequency_resolution))

                condition_0_list = np.full(shape=n_usv, fill_value=False)
                mean_signal_correlations = np.full(n_usv, np.nan)
                signal_variance = np.full(n_usv, np.nan)

                for i, usv in tqdm(
                    enumerate(merged),
                    desc="Computing spectrogram correlations/variance in progress...",
                    total=n_usv,
                    position=0,
                    leave=True,
                ):
                    start_usv = int(np.floor(usv['start'] * audio_sampling_rate))
                    stop_usv = int(np.ceil(usv['stop'] * audio_sampling_rate))
                    # Materialize the USV sample window once: each fresh memmap index
                    # re-reads the same byte range from disk, and this window is used
                    # three times (peak/mean amplitude channel + the STFT input below).
                    window = np.asarray(audio_file_data[start_usv:stop_usv, :])
                    peak_amp_ch = np.unravel_index(
                        np.argmax(window),
                        window.shape,
                    )[1]
                    mean_amp_ch = np.argmax(
                        np.abs(window).mean(axis=0)
                    )
                    usv['peak_amp_ch'] = int(peak_amp_ch)
                    usv['mean_amp_ch'] = int(mean_amp_ch)
                    usv_detected_chs = usv['chs_detected']

                    # remove USV segments if they don't appear on both peak and mean amplitude channels; this is clearly noise
                    condition_0_list[i] = (
                        peak_amp_ch not in usv_detected_chs
                        or mean_amp_ch not in usv_detected_chs
                    )

                    # the following section computes channel-wise signal correlations in the frequency domain
                    if len(usv_detected_chs) > 1:
                        spectrogram_data_selected_ch = np.abs(
                            librosa.stft(
                                window[:, usv_detected_chs]
                                .astype("float32")
                                .T,
                                n_fft=len_win_signal,
                            )
                        )
                        # Defensive: if lower_bin sits past the STFT's
                        # freq axis the slice is empty and corrcoef
                        # would later raise an obscure broadcasting
                        # error; surface the real problem here.
                        if lower_bin >= spectrogram_data_selected_ch.shape[1]:
                            msg = (
                                f"lower_bin ({lower_bin}) exceeds STFT freq-axis "
                                f"length ({spectrogram_data_selected_ch.shape[1]}); "
                                "check `low_freq_cutoff` vs `len_win_signal` / sampling rate"
                            )
                            raise ValueError(msg)
                        reshaped_spectrogram = spectrogram_data_selected_ch[
                            :, lower_bin:, :
                        ].reshape(len(usv_detected_chs), -1)
                        correlation_matrix = np.corrcoef(reshaped_spectrogram)
                        unique_correlations = correlation_matrix[
                            np.triu_indices(n=len(usv_detected_chs), k=1)
                        ]
                        mean_signal_correlations[i] = np.mean(unique_correlations)
                    else:
                        spectrogram_data_selected_ch = (
                            np.abs(
                                librosa.stft(
                                    window[:, usv_detected_chs[0]]
                                    .astype("float32")
                                    .T,
                                    n_fft=len_win_signal,
                                )
                            )
                            ** 2
                        )
                        signal_variance[i] = np.var(
                            spectrogram_data_selected_ch
                            / np.max(spectrogram_data_selected_ch)
                        )

                # mean_signal_correlations is filled only on the multi-channel branch
                # and signal_variance only on the single-channel branch, so either array
                # can be entirely NaN (e.g. every USV detected on a single channel leaves
                # mean_signal_correlations all-NaN). np.nanpercentile over an all-NaN
                # array raises a RuntimeWarning and returns NaN, which would propagate
                # into max()/min() order-dependently and defeat the configured floor/
                # ceiling. Guard each percentile and fall back to the configured cutoff
                # so the threshold is deterministic when no descriptor values exist.
                noise_corr_cutoff_min = self.input_parameter_dict["summarize_das_findings"][
                    "noise_corr_cutoff_min"
                ]
                noise_var_cutoff_max = self.input_parameter_dict["summarize_das_findings"][
                    "noise_var_cutoff_max"
                ]
                if np.any(~np.isnan(mean_signal_correlations)):
                    noise_corr_cutoff = max(
                        float(np.nanpercentile(mean_signal_correlations, q=6)),
                        noise_corr_cutoff_min,
                    )
                else:
                    noise_corr_cutoff = noise_corr_cutoff_min
                if np.any(~np.isnan(signal_variance)):
                    noise_var_cutoff = min(
                        float(np.nanpercentile(signal_variance, q=94)),
                        noise_var_cutoff_max,
                    )
                else:
                    noise_var_cutoff = noise_var_cutoff_max
                self.message_output(
                    f"Spectrogram correlation cutoff (6th percentile of distribution): {noise_corr_cutoff:.2f}"
                )
                self.message_output(
                    f"Single channel variance cutoff (94th percentile of distribution): {noise_var_cutoff:.4f}"
                )

                # filter noise: drop USVs failing amplitude-channel, correlation, or variance checks
                drop_counter = 0
                kept_merged = []
                for i, usv in enumerate(merged):
                    # DAS precision is 94%, therefore remove 6% of USVs with the lowest signal correlations
                    condition_1 = (
                        not np.isnan(mean_signal_correlations[i])
                        and mean_signal_correlations[i] < noise_corr_cutoff
                    )
                    # for signals detected only on one channel, filter based on variance
                    condition_2 = (
                        not np.isnan(signal_variance[i])
                        and signal_variance[i] < noise_var_cutoff
                    )
                    if condition_0_list[i] or condition_1 or condition_2:
                        drop_counter += 1
                    else:
                        kept_merged.append(usv)
                merged = kept_merged

                self.message_output(
                    f"Number of detections dropped due to low signal correlation/variance across channels: {drop_counter}"
                )

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 4), dpi=300)
                ax[0].hist(
                    x=mean_signal_correlations[~np.isnan(mean_signal_correlations)],
                    bins=20,
                    histtype="stepfilled",
                    color="#BBD5E8",
                    edgecolor="#202020",
                    alpha=0.5,
                )
                ax[0].set_xlabel("Mean signal/spectral correlation")
                ax[0].set_ylabel("Number of putative USVs")
                ax[0].axvline(x=noise_corr_cutoff, ls="-.", lw=1.2, c="#202020")
                ax[1].hist(
                    x=signal_variance[~np.isnan(signal_variance)],
                    bins=20,
                    histtype="stepfilled",
                    color="#BBD5E8",
                    edgecolor="#202020",
                    alpha=0.5,
                )
                ax[1].set_xlabel("Signal/spectral variance")
                ax[1].set_ylabel("Number of putative USVs")
                ax[1].axvline(x=noise_var_cutoff, ls="-.", lw=1.2, c="#202020")
                save_figure(
                    fig,
                    stem=f"{session_id}_usv_signal_correlation_histogram",
                    viz_settings=getattr(self, "visualizations_parameter_dict", None),
                    override_dir=pathlib.Path(self.root_directory) / "audio",
                    timestamp_in_name=False,
                )
                plt.close()

                self.message_output(
                    f"In this session, {len(merged)} USVs were detected."
                )

                # save the summary file
                _write_usv_summary_csv(
                    merged,
                    pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_summary.csv",
                )

            elif filter_putative_noise_bool and n_usv == 1:
                # A lone USV has no descriptor distribution to filter against, so
                # the statistical noise rejection above is skipped. The detection
                # is nonetheless real: compute its peak/mean amplitude channels
                # directly and still emit the summary CSV, rather than silently
                # dropping it and zeroing the session USV count (which is what
                # the bare `n_usv > 1` gate used to do).
                audio_file_loc = first_match_or_raise(
                    root=pathlib.Path(self.root_directory) / "audio" / "hpss_filtered",
                    pattern="*.mmap",
                    label="concatenated audio mmap",
                )
                audio_file_name = audio_file_loc.name
                # The mmap filename encodes its array metadata as the last four
                # underscore-separated tokens, in the trailing layout
                # '..._<sampling_rate>_<sample_num>_<channel_num>_<dtype>.mmap'.
                # Parsing right-to-left: [-1][:-5] is the dtype with the trailing
                # '.mmap' (5 chars) stripped, [-2] the channel count, [-3] the
                # sample count, [-4] the sampling rate.
                data_type, channel_num, sample_num, audio_sampling_rate = (
                    audio_file_name.split("_")[-1][:-5],
                    int(audio_file_name.split("_")[-2]),
                    int(audio_file_name.split("_")[-3]),
                    int(audio_file_name.split("_")[-4]),
                )
                audio_file_data = np.memmap(
                    filename=audio_file_loc,
                    mode="r",
                    dtype=data_type,
                    shape=(sample_num, channel_num),
                )
                lone_usv = merged[0]
                start_usv = int(np.floor(lone_usv['start'] * audio_sampling_rate))
                stop_usv = int(np.ceil(lone_usv['stop'] * audio_sampling_rate))
                # Single merged USV: read the window once and reuse for the peak +
                # mean amplitude channels (same pattern as the per-USV loop above).
                window = np.asarray(audio_file_data[start_usv:stop_usv, :])
                peak_amp_ch = np.unravel_index(
                    np.argmax(window),
                    window.shape,
                )[1]
                mean_amp_ch = np.argmax(
                    np.abs(window).mean(axis=0)
                )
                lone_usv['peak_amp_ch'] = int(peak_amp_ch)
                lone_usv['mean_amp_ch'] = int(mean_amp_ch)

                self.message_output(
                    f"In this session, {len(merged)} USVs were detected."
                )

                # save the summary file
                _write_usv_summary_csv(
                    merged,
                    pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_summary.csv",
                )

            elif not filter_putative_noise_bool and n_usv >= 1:
                # Putative-noise filtering disabled: keep every merged interval
                # without amplitude/spectrogram rejection. The intervals are
                # already start-sorted (Phase 2 sorts the raw detections and the
                # greedy merge preserves that order), so the peak/mean amplitude
                # channels stay at their 0.0 placeholders and the summary CSV is
                # written directly from the merged list.
                self.message_output(
                    f"Putative-noise filtering disabled; {len(merged)} USVs kept without amplitude/spectrogram checks."
                )

                # save the summary file
                _write_usv_summary_csv(
                    merged,
                    pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_summary.csv",
                )

            # load metadata
            metadata, metadata_path = load_session_metadata(
                root_directory=self.root_directory,
                logger=self.message_output
            )
            if metadata is not None:
                metadata['Session']['session_usv_count'] = len(merged)
                save_session_metadata(data=metadata, filepath=metadata_path, logger=self.message_output)

        except (IndexError, FileNotFoundError) as exc:
            self.message_output(
                f"DAS summary skipped for '{self.root_directory}': {type(exc).__name__}: {exc}"
            )
