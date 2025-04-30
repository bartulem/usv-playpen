"""
@author: bartulem
Makes dataset to run and runs vocalocator inference.
"""

from PyQt6.QtTest import QTest
from datetime import datetime
import h5py
import json
import numpy as np
import os
import pathlib
import polars as pls
import subprocess
from tqdm import tqdm
from src.assign_vocalizations_utils import (get_arena_dimensions, load_usv_segments, load_tracks_from_h5, to_float, write_to_h5,
                                            get_conf_sets_6d, are_points_in_conf_set)

class Vocalocator:

    def __init__(self, **kwargs) -> None:

        """
        Description
        ----------
        Initializes the Vocalocator class.
        ----------

        Parameters
        ----------
        root_directory (str)
            Root directory containing mouse tracking data.
        input_parameter_dict (dict)
           Processing parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.
        ----------

        Returns
        ----------
        ----------
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val


    def prepare_for_vocalocator(self) -> None:
        """
        Description
        ----------
        Prepares the root directory for vocalocator inference.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        self.message_output(f"Preparing data for vocal assignment started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        audio_file_path = next(pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}").glob(f"**{os.sep}*_concatenated_audio_*.mmap"), None)
        usv_segments_path = next(pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}").glob("*_usv_summary.csv"), None)
        track_file_path = next(pathlib.Path(f"{self.root_directory}{os.sep}video{os.sep}").glob(f"**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5"), None)
        arena_info_path = next(pathlib.Path(f"{self.input_parameter_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc']}{os.sep}video{os.sep}").glob(f"**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5"), None)

        video_frame_count_file_path = next(pathlib.Path(f"{self.root_directory}{os.sep}video{os.sep}").glob(f"**{os.sep}*_camera_frame_count_dict.json"), None)
        with open(video_frame_count_file_path, 'r') as frame_count_infile:
            video_frame_rate = json.load(frame_count_infile)['median_empirical_camera_sr']

        output_path = pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}sound_localization")
        output_path.mkdir(exist_ok=True, parents=True)
        output_path_file = output_path / 'dset.h5'

        # get arena dimensions
        arena_dimensions = get_arena_dimensions(arena_dims_path=arena_info_path)

        # get USV segments
        usv_segments = load_usv_segments(usv_segments_path)

        # load audio data
        audio_file_name_components = audio_file_path.stem.split('_')
        audio_file_dtype = audio_file_name_components[-1]
        audio_file_channel_num = int(audio_file_name_components[-2])
        audio_file_sample_num = int(audio_file_name_components[-3])
        audio_file_sample_rate = int(audio_file_name_components[-4])

        handle = np.memmap(filename=audio_file_path,
                           dtype=audio_file_dtype,
                           mode='r',
                           shape=(audio_file_sample_num, audio_file_channel_num))

        # extract relevant data from each file
        usv_onsets_in_samples = (usv_segments[:, 0] * audio_file_sample_rate).astype(int)
        usv_offsets_in_samples = (usv_segments[:, 1] * audio_file_sample_rate).astype(int)

        tracks, node_names = load_tracks_from_h5(track_file_path)
        onsets_in_seconds = usv_segments[:, 0]
        onsets_in_video_frames = (onsets_in_seconds * video_frame_rate).astype(int)
        track_locations_at_usv_onsets = tracks[onsets_in_video_frames] * 1000

        audio = [to_float(np.array(handle[onset:offset, :])) for onset, offset in tqdm(zip(usv_onsets_in_samples, usv_offsets_in_samples),
                                                                                       total=usv_segments.shape[0])]
        audio_lengths = usv_offsets_in_samples - usv_onsets_in_samples
        length_idx = np.cumsum(np.insert(audio_lengths, obj=0, values=0))

        # write data to file
        extra_metadata = {
            "arena_dims_units": "mm",
            "audio_sr": audio_file_sample_rate,
            "video_fps": video_frame_rate,
            "arena_dims": arena_dimensions}

        write_to_h5(output_path=output_path_file,
                    audio=audio,
                    node_names=node_names,
                    locations=track_locations_at_usv_onsets,
                    length_idx=length_idx,
                    extra_metadata=extra_metadata)

    def run_vocalocator(self) -> None:
        """
        Description
        ----------
        Run vocalocator inference.

        NB: The assessment.h5 file contains:
        point_predictions (shape: (n_vocalizations, n_nodes, n_dimensions)):
            This is the mean of the gaussian distribution output by the model for each vocalization.
            The unit is mm and the reference frame has its origin at the center of the arena floor.

        raw_model_output (shape: (n_vocalizations, 27)):
            This is the unnormalized vector produced by the model to parametrize the gaussian distribution.

        scaled_locations (shape: (n_vocalizations, n_mice, n_nodes, n_dimensions)):
            Animal poses for each vocalization. They are copied directly from the 'locations' array in the
            dataset used by vocalocator.assess, but only contain the nodes listed in config.json.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        ----------
        """

        self.message_output(f"Vocalization assignment started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        vcl_conda_name = self.input_parameter_dict['vocalocator']['vcl_conda_env_name']
        model_directory = self.input_parameter_dict['vocalocator']['model_directory']
        model_config_path = f"{model_directory}{os.sep}config.json"
        data_file_path = f"{self.root_directory}{os.sep}audio{os.sep}sound_localization{os.sep}dset.h5"
        output_file_path = f"{self.root_directory}{os.sep}audio{os.sep}sound_localization{os.sep}assessment.h5"
        track_file_path = next(pathlib.Path(f"{self.root_directory}{os.sep}video{os.sep}").glob(f"**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5"), None)
        usv_summary_file_path = next(pathlib.Path(f"{self.root_directory}{os.sep}audio{os.sep}").glob(f"**{os.sep}*_usv_summary.csv"), None)

        if os.name == 'nt':
            command_addition = 'cmd /c '
            shell_usage_bool = False
        else:
            command_addition = 'eval "$(conda shell.bash hook)" && '
            shell_usage_bool = True

        subprocess.run(args=f'''{command_addition}conda activate {vcl_conda_name} && python -m vocalocator.assess --config {model_config_path} --data {data_file_path} --inference -o {output_file_path}''',
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT,
                       cwd=model_directory,
                       shell=shell_usage_bool)

        QTest.qWait(1000)

        # conduct 6D inference
        with h5py.File(output_file_path, mode='r') as ctx:
            raw_output = ctx['raw_model_output'][:]
            model_config = json.loads(ctx.attrs['model_config'])
            arena_dims = np.array(model_config['DATA']['ARENA_DIMS'])
            true_locs = ctx['scaled_locations'][:]

        conf_sets, conf_sets_noangle, pdfs = get_conf_sets_6d(raw_output, arena_dims, 1.0, True)

        pts_in_set = np.stack([are_points_in_conf_set(conf_sets, true_locs[:, mouse_idx, ...], arena_dims,) for mouse_idx in range(2)],axis=1,)

        none_in_set = pts_in_set.sum(axis=1) == 0
        one_in_set = pts_in_set.sum(axis=1) == 1
        two_in_set = pts_in_set.sum(axis=1) == 2

        self.message_output(f"Vocalization percentage attributed to NO mouse: {none_in_set.sum()}")
        self.message_output(f"Vocalization percentage attributed to ONE mouse: {one_in_set.sum()}")
        self.message_output(f"Vocalization percentage attributed to BOTH mice: {two_in_set.sum()}")

        # Make assignment vector and save to disk:
        assignments = np.full((len(pts_in_set),), -1, dtype=int)
        assignments[one_in_set & pts_in_set[:, 0]] = 0  # Mouse 1
        assignments[one_in_set & pts_in_set[:, 1]] = 1  # Mouse 2
        assignments[two_in_set] = 2  # Both (ambiguous)

        np.save(f"{self.root_directory}{os.sep}audio{os.sep}sound_localization{os.sep}assessment_assn.npy", assignments)

        QTest.qWait(1000)

        # get assignment results into the usv_summary file
        with h5py.File(name=track_file_path, mode='r') as f:
            track_names = [item.decode('utf-8') for item in list(f['track_names'])]

        usv_summary_df = pls.read_csv(usv_summary_file_path)

        sound_loc_assignment_arr = np.load(f"{self.root_directory}{os.sep}audio{os.sep}sound_localization{os.sep}assessment_assn.npy")

        for mouse_idx, mouse in enumerate(track_names):
            usv_summary_df = usv_summary_df.with_columns(
                pls.when(pls.lit(sound_loc_assignment_arr == mouse_idx))
                .then(pls.lit(mouse))
                .otherwise(pls.col('emitter'))
                .alias('emitter')
            )

        usv_summary_df.write_csv(file=usv_summary_file_path, separator=',', include_header=True)
