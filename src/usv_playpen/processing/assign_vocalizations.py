"""
@author: bartulem
Makes dataset to run and runs vocalocator inference.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from datetime import datetime

import h5py
import numpy as np
import polars as pls
from tqdm import tqdm

from ..os_utils import AUDIO_MMAP_BAND_FOLDERS, atomic_output_path, configure_path, find_audio_mmap, first_match_or_raise, order_usv_summary_columns
from ..time_utils import is_gui_context, smart_wait
from ..yaml_utils import extract_animal_sexes, load_session_metadata, save_session_metadata
from .assign_vocalizations_utils import (
    are_points_in_conf_set,
    get_arena_dimensions,
    get_conf_sets_6d,
    load_tracks_from_h5,
    load_usv_segments,
    to_float,
    write_to_h5,
)


class Vocalocator:

    def __init__(self, **kwargs) -> None:

        """
        Description
        -----------
        Initializes the Vocalocator class.

        Parameters
        ----------
        root_directory (str)
            Root directory containing mouse tracking data.
        input_parameter_dict (dict)
           Processing parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        expected_kwargs = {'root_directory', 'input_parameter_dict', 'message_output'}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword argument(s) "
                            f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                            f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        self.app_context_bool = is_gui_context()

    def prepare_for_vocalocator(self) -> None:
        """
        Description
        -----------
        Prepares the root directory for vocalocator inference. The ``dset.h5``
        bundle includes a per-call ``animal_id`` field of per-animal SEX codes
        (0 = male, 1 = female), derived from the session metadata's
        ``Subjects`` matched to the track names -- an input feature of the
        Vocalocator identity model.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(f"Preparing data for vocal assignment started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # The audio band the vocalocator model was trained on, the setting
        # vocalocator.vcl_audio_band: 'ultrasonic' (default) is the 30 kHz high-passed memmap
        # (audio/hpss_filtered) the current models were trained on, 'broadband' the
        # 2 kHz high-passed one (audio/broadband_filtered), for a model trained on
        # 2-125 kHz audio. The lookup takes the exact folder and name and needs exactly
        # one match (a recursive glob here used to pick the alphabetically first memmap
        # anywhere under 'audio', e.g. a stray unfiltered one in 'cropped_to_video').
        vcl_audio_band = self.input_parameter_dict['vocalocator']['vcl_audio_band']
        if vcl_audio_band not in AUDIO_MMAP_BAND_FOLDERS:
            error_message = f"vocalocator.vcl_audio_band must be one of {sorted(AUDIO_MMAP_BAND_FOLDERS)}, got {vcl_audio_band!r}."
            raise ValueError(error_message)
        audio_file_path = find_audio_mmap(root_directory=self.root_directory, band=vcl_audio_band)
        self.message_output(f"Vocalocator audio: the {vcl_audio_band!r} band memmap {audio_file_path.name}.")
        usv_segments_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'audio',
            pattern='*_usv_summary.csv',
            label="USV summary CSV",
        )
        track_file_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'video',
            pattern='[0-9]*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="3D translated/rotated/metric track H5",
        )
        arena_info_path = first_match_or_raise(
            root=pathlib.Path(configure_path(self.input_parameter_dict['anipose_operations']['ConvertTo3D']['conduct_anipose_triangulation']['calibration_file_loc'])) / 'video',
            pattern='[0-9]*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="arena calibration translated/rotated/metric H5",
        )

        video_frame_count_file_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'video',
            pattern='*_camera_frame_count_dict.json',
            recursive=True,
            label="camera frame count JSON",
        )
        with open(video_frame_count_file_path, 'r') as frame_count_infile:
            video_frame_rate = json.load(frame_count_infile)['median_empirical_camera_sr']

        output_path = pathlib.Path(self.root_directory) / 'audio' / 'sound_localization'
        output_path.mkdir(exist_ok=True, parents=True)
        output_path_file = output_path / 'dset.h5'

        if not output_path_file.exists():

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
            # A USV onset that lands at or beyond the last tracked video frame
            # (rounding, a recording that ends mid-vocalization, or a track that
            # is a few frames shorter than the audio) would index past the end of
            # `tracks` and raise IndexError. Clamp the frame index to the last
            # valid frame so those edge-of-recording USVs reuse the final pose
            # rather than aborting the whole assignment run.
            onsets_in_video_frames = np.clip(onsets_in_video_frames, 0, tracks.shape[0] - 1)
            track_locations_at_usv_onsets = tracks[onsets_in_video_frames] * 1000

            audio = [to_float(handle[onset:offset, :]) for onset, offset in tqdm(zip(usv_onsets_in_samples, usv_offsets_in_samples),
                                                                                           total=usv_segments.shape[0])]
            # Derive each segment length from the audio array actually sliced out
            # of the memmap, not from the nominal (offset - onset). A USV whose
            # offset sample exceeds the recording length is silently truncated by
            # the slice, so the nominal length would overstate it and desync the
            # `length_idx` offsets from the concatenated `audio` written to disk.
            audio_lengths = np.array([segment.shape[0] for segment in audio], dtype=np.int64)
            length_idx = np.cumsum(np.insert(audio_lengths, obj=0, values=0))

            # write data to file
            extra_metadata = {
                "arena_dims_units": "mm",
                "audio_sr": audio_file_sample_rate,
                "video_fps": video_frame_rate,
                "arena_dims": arena_dimensions}

            # Vocalocator's `animal_id` field is a SEX code per candidate animal
            # (0 = male, 1 = female), in the tracks-array animal order -- an
            # input feature of the identity model, not an animal index.
            # np.arange only coincidentally produced the right codes for
            # male/female (courtship) sessions; female-female sessions need
            # [1, 1] and male-male [0, 0], so the codes are derived from the
            # session metadata's Subjects, matched to the track names
            # (yaml_utils.extract_animal_sexes, the package's one sex lookup).
            with h5py.File(track_file_path, mode='r') as track_file:
                # some track h5 files carry stray whitespace in track names
                # (e.g. ' 158800_0'); strip before matching against Subjects
                track_names = [item.decode('utf-8').strip('\x00').strip() for item in list(track_file['track_names'])]
            if len(track_names) != tracks.shape[1]:
                err_msg = (
                    f"Track h5 '{track_file_path}' is inconsistent: {len(track_names)} track_names "
                    f"but {tracks.shape[1]} animals on the tracks array."
                )
                raise ValueError(err_msg)
            # Raises (FileNotFoundError / ValueError) on missing metadata, a track with no
            # matching subject_id, or a sex other than male / female: never a guess.
            animal_sex = extract_animal_sexes(self.root_directory, track_names, logger=self.message_output)
            sex_to_code = {'male': 0, 'female': 1}
            animal_id_codes = [sex_to_code[animal_sex[track_name]] for track_name in track_names]
            animal_ids = np.array(animal_id_codes, dtype=np.int32)
            self.message_output(
                "Vocalocator animal_id sex codes (0=male, 1=female): "
                + ", ".join(f"{name}={code}" for name, code in zip(track_names, animal_id_codes, strict=True))
            )

            write_to_h5(output_path=output_path_file,
                        audio=audio,
                        node_names=node_names,
                        locations=track_locations_at_usv_onsets,
                        length_idx=length_idx,
                        animal_ids=animal_ids,
                        extra_metadata=extra_metadata)

    def run_vocalocator(self) -> None:
        """
        Description
        -----------
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

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(f"Vocalization assignment started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        vcl_conda_name = self.input_parameter_dict['vocalocator']['vcl_conda_env_name']
        model_directory = configure_path(self.input_parameter_dict['vocalocator']['vcl_model_directory'])
        model_config_path = str(pathlib.Path(model_directory) / 'config.json')
        data_file_path = str(pathlib.Path(self.root_directory) / 'audio' / 'sound_localization' / 'dset.h5')
        output_file_path = str(pathlib.Path(self.root_directory) / 'audio' / 'sound_localization' / 'assessment.h5')
        track_file_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'video',
            pattern='[0-9]*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="3D translated/rotated/metric track H5",
        )
        usv_summary_file_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'audio',
            pattern='*_usv_summary.csv',
            recursive=True,
            label="USV summary CSV",
        )

        conda_exe = os.environ.get('CONDA_EXE', 'conda')
        clean_env = os.environ.copy()
        clean_env.pop('PYTHONHOME', None)
        subprocess.run(
            args=[conda_exe, 'run', '--no-capture-output', '-n', vcl_conda_name, 'python', '-m', 'vocalocator.assess',
                  '--config', model_config_path, '--data', data_file_path, '--inference', '-o', output_file_path],
            cwd=model_directory,
            env=clean_env,
            shell=False,
            check=True
        )

        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # conduct 6D inference
        with h5py.File(output_file_path, mode='r') as ctx:
            raw_output = ctx['raw_model_output'][:]
            model_config = json.loads(ctx.attrs['model_config'])
            arena_dims = np.array(model_config['DATA']['ARENA_DIMS'])
            true_locs = ctx['scaled_locations'][:]

        # Confidence-set hyperparameters (temperature, spatial grid resolution,
        # angular bin count, angle-PDF sample count / seed, confidence level) come
        # from the `assign_vocalizations` settings block rather than being
        # hard-coded here. The same grid resolution / angle-bin count is passed to
        # both get_conf_sets_6d and are_points_in_conf_set so the confidence-set
        # lookup samples the identical grid the sets were built on.
        av_settings = self.input_parameter_dict['assign_vocalizations']
        conf_grid_resolution = tuple(int(v) for v in av_settings['grid_resolution'])
        conf_n_angle_bins = av_settings['n_angle_bins']
        conf_sets, _, _ = get_conf_sets_6d(
            raw_output, arena_dims, av_settings['temperature'], True,
            grid_resolution=conf_grid_resolution,
            n_angle_bins=conf_n_angle_bins,
            n_samples=av_settings['n_samples'],
            confidence_level=av_settings['confidence_level'],
            angle_pdf_seed=av_settings['angle_pdf_seed'],
        )

        pts_in_set = np.stack([are_points_in_conf_set(conf_sets, true_locs[:, mouse_idx, ...], arena_dims, grid_resolution=conf_grid_resolution, n_angle_bins=conf_n_angle_bins,) for mouse_idx in range(true_locs.shape[1])],axis=1,)

        none_in_set = pts_in_set.sum(axis=1) == 0
        one_in_set = pts_in_set.sum(axis=1) == 1
        two_in_set = pts_in_set.sum(axis=1) == 2

        # Make assignment vector and save to disk:
        mouse_one_vocalizations = one_in_set & pts_in_set[:, 0]
        mouse_two_vocalizations = one_in_set & pts_in_set[:, 1]
        assignments = np.full((len(pts_in_set),), -1, dtype=int)
        assignments[mouse_one_vocalizations] = 0  # Mouse 1
        assignments[mouse_two_vocalizations] = 1  # Mouse 2
        assignments[two_in_set] = 2  # Both (ambiguous)

        self.message_output(f"Vocalization count attributed to NO mouse: {none_in_set.sum()}")
        self.message_output(f"Vocalization count attributed to ONE mouse: {one_in_set.sum()}")
        self.message_output(f"Vocalization count attributed to mouse #1: {mouse_one_vocalizations.sum()}")
        self.message_output(f"Vocalization count attributed to mouse #2: {mouse_two_vocalizations.sum()}")
        self.message_output(f"Vocalization count attributed to BOTH mice: {two_in_set.sum()}")

        np.save(pathlib.Path(self.root_directory) / 'audio' / 'sound_localization' / 'assessment_assn.npy', assignments)

        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # get assignment results into the usv_summary file
        with h5py.File(name=track_file_path, mode='r') as f:
            track_names = [item.decode('utf-8').strip() for item in list(f['track_names'])]

        usv_summary_df = pls.read_csv(str(usv_summary_file_path), schema_overrides={"usv_id": pls.String})

        sound_loc_assignment_arr = np.load(pathlib.Path(self.root_directory) / 'audio' / 'sound_localization' / 'assessment_assn.npy')

        # Initialise every emitter to None, then set only the assigned rows.
        # Building the column off `pls.col('emitter')` would let an unassigned
        # USV (sound_loc_assignment_arr == -1) keep whatever stale emitter the
        # CSV already held from a prior assignment pass; starting from None
        # guarantees the emitter reflects only this run's assignments.
        emitter_expression = pls.lit(value=None, dtype=pls.String)

        for mouse_idx, mouse in enumerate(track_names):
            emitter_expression = (
                pls.when(pls.lit(sound_loc_assignment_arr == mouse_idx))
                .then(pls.lit(mouse))
                .otherwise(emitter_expression)
            )

        usv_summary_df = usv_summary_df.with_columns(
            emitter_expression.alias('emitter')
        )

        # usv_summary.csv holds every other per-USV column too: publish atomically, in
        # the canonical column order (os_utils.USV_SUMMARY_COLUMN_ORDER).
        with atomic_output_path(usv_summary_file_path) as tmp_summary_path:
            order_usv_summary_columns(usv_summary_df).write_csv(file=str(tmp_summary_path), separator=',', include_header=True)

        # load metadata
        metadata, metadata_path = load_session_metadata(
            root_directory=self.root_directory,
            logger=self.message_output
        )
        if metadata is not None:
            metadata['Session']['session_usv_assigned'] = True
            for subject in metadata['Subjects']:
                for mouse_idx, track_name in enumerate(track_names):
                    if str(subject['subject_id']) == track_name:
                        subject['num_assigned_vocalizations'] = int((assignments == mouse_idx).sum())
                        break
            save_session_metadata(data=metadata, filepath=metadata_path, logger=self.message_output)

    def run_vocalocator_ssl(self) -> None:
        """
        Description
        -----------
        Run vocalocator-ssl inference.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(f"Vocalization assignment started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        vcl_conda_name = self.input_parameter_dict['vocalocator']['vcl_conda_env_name']
        model_directory = configure_path(self.input_parameter_dict['vocalocator']['vcl_model_directory'])
        data_file_path = pathlib.Path(self.root_directory) / 'audio' / 'sound_localization'
        track_file_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'video',
            pattern='[0-9]*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="3D translated/rotated/metric track H5",
        )
        usv_summary_file_path = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'audio',
            pattern='*_usv_summary.csv',
            recursive=True,
            label="USV summary CSV",
        )

        try:
            # Locate the calibration NPZ file; if none is found the resulting StopIteration is caught below and the method logs a warning and returns (no error propagates to the caller)
            cal_file = next(f.name for f in sorted(pathlib.Path(model_directory).glob("*cal*.npz")))

            conda_exe = os.environ.get('CONDA_EXE', 'conda')
            clean_env = os.environ.copy()
            clean_env.pop('PYTHONHOME', None)
            subprocess.run(
                args=[conda_exe, 'run', '--no-capture-output', '-n', vcl_conda_name, 'python', '-m', 'vocalocatorssl',
                      '--data', str(data_file_path), '--save-path', model_directory, '--predict',
                      '-o', str(data_file_path / 'model_predictions.npz')],
                cwd=model_directory,
                env=clean_env,
                shell=False,
                check=True
            )
            subprocess.run(
                args=[conda_exe, 'run', '--no-capture-output', '-n', vcl_conda_name, 'python', '-m', 'vocalocatorssl.assign',
                      str(data_file_path / 'model_predictions.npz'),
                      '--calibration-results', str(pathlib.Path(model_directory) / cal_file)],
                cwd=model_directory,
                env=clean_env,
                shell=False,
                check=True
            )
        except (StopIteration, subprocess.CalledProcessError):
            # This catches both the missing file (StopIteration) and shell command failures
            self.message_output("No calibration NPZ file found in model directory or the subprocess failed.")
            return

        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # load track IDs and the usv_summary file
        with h5py.File(name=track_file_path, mode='r') as f:
            track_names = [item.decode('utf-8').strip() for item in list(f['track_names'])]

        usv_summary_df = pls.read_csv(str(usv_summary_file_path), schema_overrides={"usv_id": pls.String})

        # get assignments
        model_predictions_archive = np.load(file=pathlib.Path(self.root_directory) / 'audio' / 'sound_localization' / 'model_predictions.npz', allow_pickle=True)
        assignment_array_candidates = [k for k in model_predictions_archive.files if k.endswith('assignments')]
        if not assignment_array_candidates:
            msg = (
                f"No '*assignments' array found in model_predictions.npz "
                f"(archive keys: {list(model_predictions_archive.files)})."
            )
            raise KeyError(msg)
        assignment_array_id = assignment_array_candidates[0]
        assignments = model_predictions_archive[assignment_array_id]

        # get assignment statistics
        unique_values, counts = np.unique(assignments, return_counts=True)
        value_to_count = dict(zip(unique_values, counts))
        total = len(assignments)
        unassigned = value_to_count.get(-1, 0)
        assigned = total - unassigned

        # Guard the percentage reporting against an empty predictions array
        # (total == 0 would otherwise raise ZeroDivisionError here, crashing the
        # run after the expensive SSL subprocesses already finished). The emitter
        # assignment below still runs, so an empty session keeps its 'emitter'
        # column (all null) for downstream consumers.
        if total == 0:
            self.message_output("No vocalizations to assign (empty predictions array).")
        else:
            self.message_output(f"Out of {total} vocalizations, {assigned} have been assigned, or {round(assigned*100/total, 2)}%.")
            self.message_output(f"{unassigned} vocalizations have not been assigned, or {round(unassigned*100/total, 2)}%.")
            for animal_id, track_id in enumerate(track_names):
                count = value_to_count.get(animal_id, 0)
                self.message_output(f"Mouse {track_id} has been assigned {count} vocalizations, or {round(count*100/total, 2)}%.")

        # assign None to all unassigned vocalizations
        emitter_expression = pls.lit(value=None, dtype=pls.String)

        for mouse_idx, mouse_name in enumerate(track_names):
            emitter_expression = (
                pls.when(pls.lit(assignments) == mouse_idx)
                .then(pls.lit(mouse_name))
                .otherwise(emitter_expression)
            )

        usv_summary_df = usv_summary_df.with_columns(
            emitter_expression.alias('emitter')
        )

        # usv_summary.csv holds every other per-USV column too: publish atomically, in
        # the canonical column order (os_utils.USV_SUMMARY_COLUMN_ORDER).
        with atomic_output_path(usv_summary_file_path) as tmp_summary_path:
            order_usv_summary_columns(usv_summary_df).write_csv(file=str(tmp_summary_path), separator=',', include_header=True)

        # load metadata
        metadata, metadata_path = load_session_metadata(
            root_directory=self.root_directory,
            logger=self.message_output
        )
        if metadata is not None:
            metadata['Session']['session_usv_assigned'] = True
            for subject in metadata['Subjects']:
                for mouse_idx, track_name in enumerate(track_names):
                    if str(subject['subject_id']) == track_name:
                        subject['num_assigned_vocalizations'] = int((assignments == mouse_idx).sum())
                        break
            save_session_metadata(data=metadata, filepath=metadata_path, logger=self.message_output)
