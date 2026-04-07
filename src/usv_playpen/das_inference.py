"""
@author: bartulem
Run USV inference on WAV files and create annotations.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime

import librosa
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from tqdm import tqdm

from .time_utils import is_gui_context, smart_wait
from .yaml_utils import load_session_metadata, save_session_metadata

fm.fontManager.addfont(pathlib.Path(__file__).parent / "fonts/Helvetica.ttf")
plt.style.use(pathlib.Path(__file__).parent / "_config/usv_playpen.mplstyle")


class FindMouseVocalizations:
    def __init__(
        self,
        root_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        exp_settings_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Initializes the FindMouseVocalizations class.

        Parameter
        ---------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        ecp_settings_dict (dict)
            Experimental settings; defaults to None.
        message_output (function)
            Function to output messages; defaults to None.

        Returns
        -------
        -------
        """

        if input_parameter_dict is None or root_directory is None:
            with open(
                pathlib.Path(__file__).parent / "_parameter_settings/processing_settings.json"
            ) as json_file:
                _defaults = json.load(json_file)

            if input_parameter_dict is None:
                self.input_parameter_dict = _defaults["usv_inference"]["FindMouseVocalizations"]
            else:
                self.input_parameter_dict = input_parameter_dict["usv_inference"]["FindMouseVocalizations"]

            if root_directory is None:
                self.root_directory = _defaults["usv_inference"]["root_directory"]
            else:
                self.root_directory = root_directory
        else:
            self.input_parameter_dict = input_parameter_dict["usv_inference"]["FindMouseVocalizations"]
            self.root_directory = root_directory

        self.exp_settings_dict = exp_settings_dict
        self.message_output = message_output or print

        self.app_context_bool = is_gui_context()

    def das_command_line_inference(self) -> None:
        """
        Description
        ----------
        This method takes WAV files as input and runs DAS inference on them to generate
        tentative USV segments in the recording.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        .csv annotation files
            CSV files w/ onsets and offsets of all detected USV segments,
            shape: (N_USV, VOC_TYPE, START_SEC, END_SEC).
        ----------
        """

        self.message_output(
            f"DAS inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}. Please be patient, this can take >5 min/file."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        das_conda_name = self.input_parameter_dict["das_command_line_inference"][
            "das_conda_env_name"
        ]
        model_base = str(pathlib.Path(self.input_parameter_dict['das_command_line_inference']['das_model_directory']) / self.input_parameter_dict['das_command_line_inference']['model_name_base'])
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

            while True:
                status_poll = inference_subp.poll()
                if status_poll is None:
                    smart_wait(app_context_bool=self.app_context_bool, seconds=5)
                else:
                    break

        # create save directory if it doesn't exist
        das_dir = pathlib.Path(self.root_directory) / "audio" / "das_annotations"
        das_dir.mkdir(parents=True, exist_ok=True)

        # move annotation files to save directory
        for one_file in hpss_dir.iterdir():
            if f".{save_format}" in one_file.name:
                shutil.move(src=one_file, dst=das_dir / one_file.name)

    def summarize_das_findings(self) -> None:
        """
        Description
        ----------
        This method takes CSV files generated by DAS inference and creates a summary file,
        containing information about individual USV segment start and stop times, duration,
        peak amplitude channel, mean amplitude channel, total number of channels it was
        detected on, list of channels it was detected on, and emitter ID.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        .csv summary file
            CSV file w/ information about all detected USV segments,
            shape: (N_USV, START, STOP, DURATION, PEAK_AMP_CH,
            MEAN_AMP_CH, CHs_COUNT, CHS_DETECTED, EMMITER).
        ----------
        """

        self.message_output(
            f"DAS summary started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}."
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

        try:
            das_annotation_files = sorted(
                (pathlib.Path(self.root_directory) / "audio" / "das_annotations").glob("*.csv")
            )

            # Phase 1: load all channel CSVs, filter noise, collect flat segment list
            # Each entry is (start_seconds, stop_seconds, channel_numeric_index).
            all_segments = []
            for one_file in das_annotation_files:
                file_id = (
                    one_file.name.split("_")[0]
                    + "_"
                    + one_file.name.split("_")[2]
                )
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

            # Phase 4: amplitude + spectrogram quality checks
            if n_usv > 1:
                audio_file_loc = sorted(
                    (pathlib.Path(self.root_directory) / "audio" / "hpss_filtered").glob("*.mmap")
                )[0]
                audio_file_name = audio_file_loc.name
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
                    peak_amp_ch = np.unravel_index(
                        np.argmax(audio_file_data[start_usv:stop_usv, :]),
                        audio_file_data.shape,
                    )[1]
                    mean_amp_ch = np.argmax(
                        np.abs(audio_file_data[start_usv:stop_usv, :]).mean(axis=0)
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
                                audio_file_data[start_usv:stop_usv, usv_detected_chs]
                                .astype("float32")
                                .T,
                                n_fft=len_win_signal,
                            )
                        )
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
                                    audio_file_data[start_usv:stop_usv, usv_detected_chs[0]]
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

                noise_corr_cutoff = max(
                    float(np.nanpercentile(mean_signal_correlations, q=6)),
                    self.input_parameter_dict["summarize_das_findings"][
                        "noise_corr_cutoff_min"
                    ],
                )
                noise_var_cutoff = min(
                    float(np.nanpercentile(signal_variance, q=94)),
                    self.input_parameter_dict["summarize_das_findings"][
                        "noise_var_cutoff_max"
                    ],
                )
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
                    edgecolor="#000000",
                    alpha=0.5,
                )
                ax[0].set_xlabel("Mean signal/spectral correlation")
                ax[0].set_ylabel("Number of putative USVs")
                ax[0].axvline(x=noise_corr_cutoff, ls="-.", lw=1.2, c="#000000")
                ax[1].hist(
                    x=signal_variance[~np.isnan(signal_variance)],
                    bins=20,
                    histtype="stepfilled",
                    color="#BBD5E8",
                    edgecolor="#000000",
                    alpha=0.5,
                )
                ax[1].set_xlabel("Signal/spectral variance")
                ax[1].set_ylabel("Number of putative USVs")
                ax[1].axvline(x=noise_var_cutoff, ls="-.", lw=1.2, c="#000000")
                fig.savefig(
                    fname=pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_signal_correlation_histogram.svg",
                    dpi=300,
                )
                plt.close()

                self.message_output(
                    f"In this session, {len(merged)} USVs were detected."
                )

                # save the summary file
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
                }).write_csv(
                    file=pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_summary.csv",
                )

            # load metadata
            metadata, metadata_path = load_session_metadata(
                root_directory=self.root_directory,
                logger=self.message_output
            )
            if metadata is not None:
                metadata['Session']['session_usv_count'] = len(merged) if len(merged) > 1 else 0
                save_session_metadata(data=metadata, filepath=metadata_path, logger=self.message_output)

        except (IndexError, FileNotFoundError):
            self.message_output(
                f"No DAS annotations found in directory: {self.root_directory}. Skipping summary generation."
            )
