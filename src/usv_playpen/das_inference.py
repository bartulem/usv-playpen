"""
@author: bartulem
Run USV inference on WAV files and create annotations.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime

import librosa
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .time_utils import *

fm.fontManager.addfont(pathlib.Path(__file__).parent / "fonts/Helvetica.ttf")
plt.style.use(pathlib.Path(__file__).parent / "_config/usv_playpen.mplstyle")


class FindMouseVocalizations:
    def __init__(
        self,
        root_directory: str = None,
        input_parameter_dict: dict = None,
        exp_settings_dict: dict = None,
        message_output: callable = None,
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

        if input_parameter_dict is None:
            with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "_parameter_settings/processing_settings.json",
                )
            ) as json_file:
                self.input_parameter_dict = json.load(json_file)["usv_inference"][
                    "FindMouseVocalizations"
                ]
        else:
            self.input_parameter_dict = input_parameter_dict["usv_inference"][
                "FindMouseVocalizations"
            ]

        if root_directory is None:
            with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "_parameter_settings/processing_settings.json",
                )
            ) as json_file:
                self.root_directory = json.load(json_file)["usv_inference"][
                    "root_directory"
                ]
        else:
            self.root_directory = root_directory

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

        if exp_settings_dict is None:
            self.exp_settings_dict = None
        else:
            self.exp_settings_dict = exp_settings_dict

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
        model_base = f"{self.input_parameter_dict['das_command_line_inference']['das_model_directory']}{os.sep}{self.input_parameter_dict['das_command_line_inference']['model_name_base']}"
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

        if os.name == "nt":
            command_addition = "cmd /c "
            shell_usage_bool = False
        else:
            command_addition = 'eval "$(conda shell.bash hook)" && '
            shell_usage_bool = True

        # run inference
        for one_file in sorted(
            glob.glob(
                pathname=os.path.join(
                    f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered", "*.wav*"
                )
            )
        ):
            self.message_output(
                f"Running DAS inference on: {os.path.basename(one_file)}"
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            inference_subp = subprocess.Popen(
                args=f"""{command_addition}conda activate {das_conda_name} && das predict {one_file} {model_base} --segment-thres {thresh} --segment-minlen {min_len} --segment-fillgap {fill_gap} --save-format {save_format}""",
                cwd=f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                shell=shell_usage_bool,
            )

            while True:
                status_poll = inference_subp.poll()
                if status_poll is None:
                    smart_wait(app_context_bool=self.app_context_bool, seconds=5)
                else:
                    break

        # create save directory if it doesn't exist
        pathlib.Path(
            f"{self.root_directory}{os.sep}audio{os.sep}das_annotations"
        ).mkdir(parents=True, exist_ok=True)

        # move CSV files to save directory and remove them from WAV directory
        for one_file in os.listdir(
            f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered"
        ):
            if f".{save_format}" in one_file:
                shutil.move(
                    src=f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered{os.sep}{one_file}",
                    dst=f"{self.root_directory}{os.sep}audio{os.sep}das_annotations{os.sep}{one_file}",
                )

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

        session_id = self.root_directory.split(os.sep)[-1]

        try:
            das_annotation_files = sorted(
                glob.glob(
                    f"{self.root_directory}{os.sep}audio{os.sep}das_annotations{os.sep}*.csv"
                )
            )

            # extract data from all CSV files
            usv_data = {}
            for one_file in das_annotation_files:
                file_id = (
                    one_file.split(os.sep)[-1].split("_")[0]
                    + "_"
                    + one_file.split(os.sep)[-1].split("_")[2]
                )
                usv_data[file_id] = pd.read_csv(
                    filepath_or_buffer=one_file, sep=",", index_col=0
                )

            # filter noise (usually the last row of every DAS file)
            for channel_id, channel_data in usv_data.items():
                usv_data[channel_id] = channel_data.loc[channel_data["name"] != "noise"]

            usv_summary = pd.DataFrame.from_dict(
                data={
                    "usv_id": [0.0],
                    "start": [0.0],
                    "stop": [0.0],
                    "duration": [0.0],
                    "peak_amp_ch": [0.0],
                    "mean_amp_ch": [0.0],
                    "chs_count": [0.0],
                    "chs_detected": [[0.0]],
                    "emitter": [np.nan],
                },
                orient="columns",
            )

            usv_summary_null = usv_summary.copy()

            # extract USV onsets and offsets, and channels they are detected on
            progress_bar = tqdm(
                list(usv_data.keys()),
                desc="USV match search on {:s}".format(""),
                position=0,
                leave=True,
            )
            usv_num = 0
            for channel_id in progress_bar:
                progress_bar.set_description(
                    f"USV match search on {channel_id:s}", refresh=True
                )
                channel_data = usv_data[channel_id]
                if not channel_data.empty:
                    for index, row in channel_data.iterrows():
                        try:
                            next_idx_overlap_with_previous = pd.Interval(
                                left=usv_summary.iloc[-1, 1],
                                right=usv_summary.iloc[-1, 2],
                                closed="neither",
                            ).overlaps(
                                pd.Interval(
                                    left=channel_data.loc[index, "start_seconds"],
                                    right=channel_data.loc[index, "stop_seconds"],
                                    closed="neither",
                                )
                            )
                        except (IndexError, KeyError):
                            next_idx_overlap_with_previous = False

                        if not next_idx_overlap_with_previous:
                            if usv_num > 0:
                                usv_summary = usv_summary._append(
                                    usv_summary_null, ignore_index=True
                                )
                            chs_count_temp = 1
                            chs_detected_temp = [ch_conversion_dict[channel_id]]
                            usv_summary.iloc[-1, 1] = row["start_seconds"]
                            usv_summary.iloc[-1, 2] = row["stop_seconds"]
                            usv_num += 1
                            for (
                                channel_id_other,
                                channel_data_other,
                            ) in usv_data.items():
                                if (
                                    channel_id != channel_id_other
                                    and not channel_data_other.empty
                                ):
                                    for (
                                        index_other,
                                        row_other,
                                    ) in channel_data_other.iterrows():
                                        if pd.Interval(
                                            left=row["start_seconds"],
                                            right=row["stop_seconds"],
                                            closed="neither",
                                        ).overlaps(
                                            pd.Interval(
                                                left=row_other["start_seconds"],
                                                right=row_other["stop_seconds"],
                                                closed="neither",
                                            )
                                        ):
                                            usv_summary.iloc[-1, 1] = min(
                                                usv_summary.iloc[-1, 1],
                                                row_other["start_seconds"],
                                            )
                                            usv_summary.iloc[-1, 2] = max(
                                                usv_summary.iloc[-1, 2],
                                                row_other["stop_seconds"],
                                            )
                                            chs_count_temp += 1
                                            chs_detected_temp.append(
                                                ch_conversion_dict[channel_id_other]
                                            )
                                            # the following segment checks for USV segments that are complete in some channels but broken in pieces in others
                                            try:
                                                if (
                                                    index_other + 1
                                                    <= channel_data_other.shape[0]
                                                    and pd.Interval(
                                                        left=row["start_seconds"],
                                                        right=row["stop_seconds"],
                                                        closed="neither",
                                                    ).overlaps(
                                                        pd.Interval(
                                                            left=channel_data_other.loc[
                                                                index_other + 1,
                                                                "start_seconds",
                                                            ],
                                                            right=channel_data_other.loc[
                                                                index_other + 1,
                                                                "stop_seconds",
                                                            ],
                                                            closed="neither",
                                                        )
                                                    )
                                                ):
                                                    channel_data_other.drop(
                                                        labels=[
                                                            index_other,
                                                            index_other + 1,
                                                        ],
                                                        inplace=True,
                                                    )
                                                else:
                                                    channel_data_other.drop(
                                                        labels=index_other, inplace=True
                                                    )
                                            except (IndexError, KeyError):
                                                channel_data_other.drop(
                                                    labels=index_other, inplace=True
                                                )

                                            break

                            channel_data.drop(labels=index, inplace=True)

                            usv_summary.iloc[-1, 6] = chs_count_temp
                            usv_summary.iat[-1, 7] = chs_detected_temp

            # compute USV durations and order them by start time
            usv_summary["duration"] = usv_summary["stop"] - usv_summary["start"]
            usv_summary.sort_values(by="start", ascending=True, inplace=True)

            # find peak and mean amplitude channels and filter out noise
            mean_signal_correlations = np.zeros(usv_summary.shape[0])
            mean_signal_correlations[:] = np.nan

            signal_variance = np.zeros(usv_summary.shape[0])
            signal_variance[:] = np.nan

            audio_file_loc = sorted(
                glob.glob(
                    f"{self.root_directory}{os.sep}audio{os.sep}hpss_filtered{os.sep}*.mmap"
                )
            )[0]
            audio_file_name = os.path.basename(audio_file_loc)
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

            condition_0_list = np.full(shape=usv_summary.shape[0], fill_value=False)
            for index, row in tqdm(
                usv_summary.iterrows(),
                desc="Computing spectrogram correlations/variance in progress...",
                total=usv_summary.shape[0],
                position=0,
                leave=True,
            ):
                start_usv = int(np.floor(row["start"] * audio_sampling_rate))
                stop_usv = int(np.ceil(row["stop"] * audio_sampling_rate))
                peak_amp_ch = np.unravel_index(
                    np.argmax(audio_file_data[start_usv:stop_usv, :]),
                    audio_file_data.shape,
                )[1]
                mean_amp_ch = np.argmax(
                    np.abs(audio_file_data[start_usv:stop_usv, :]).mean(axis=0)
                )
                usv_summary.at[index, "peak_amp_ch"] = peak_amp_ch
                usv_summary.at[index, "mean_amp_ch"] = mean_amp_ch
                usv_detected_chs = row["chs_detected"]

                # remove USV segments if they don't appear on both peak and mean amplitude channels; this is clearly noise
                condition_0_list[index] = (
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
                    mean_signal_correlations[index] = np.mean(unique_correlations)
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
                    signal_variance[index] = np.var(
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
            drop_counter = 0
            for index, row in usv_summary.iterrows():
                # DAS precision is 94%, therefore remove 6% of USVs with the lowest signal correlations
                condition_1 = False
                if ~np.isnan(mean_signal_correlations[index]):
                    if mean_signal_correlations[index] < noise_corr_cutoff:
                        condition_1 = True

                # For signals detected only on one channel, filter based on variance
                condition_2 = False
                if ~np.isnan(signal_variance[index]):
                    if signal_variance[index] < noise_var_cutoff:
                        condition_2 = True

                if condition_0_list[index] or condition_1 or condition_2:
                    usv_summary.drop(labels=index, inplace=True)
                    drop_counter += 1

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
                fname=f"{self.root_directory}{os.sep}audio{os.sep}{session_id}_usv_signal_correlation_histogram.svg",
                dpi=300,
            )
            plt.close()

            # give ID number to each USV
            usv_summary["usv_id"] = [
                f"{_num:04d}" for _num in range(usv_summary.shape[0])
            ]

            self.message_output(
                f"In this session, {usv_summary.shape[0]} USVs were detected."
            )

            # save the summary file
            usv_summary.to_csv(
                path_or_buf=f"{self.root_directory}{os.sep}audio{os.sep}{session_id}_usv_summary.csv",
                sep=",",
                index=False,
            )

        except (IndexError, FileNotFoundError):
            self.message_output(
                f"No DAS annotations found in directory: {self.root_directory}. Skipping summary generation."
            )
