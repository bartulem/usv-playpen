"""
@author: bartulem
Loads input files (behavior and vocalization) for running GLMs.
"""

import h5py
import numpy as np
import os
import pathlib
import pickle
import polars as pls
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel


def load_behavioral_feature_data(behavior_file_paths: list = None,
                                 csv_sep: str = ',') -> tuple:
    """
    Loads behavior data from a .csv file.

    Parameters
    ----------
    behavior_file_paths : list
        Paths to the sessions containing behavioral feature data.
    csv_sep : str, optional
        Separator used in the .csv file.

    Returns
    -------
    behavior_data : tuple (dict. dict, dict)
        Behavior, camera frame rate and track name data (keys are file names and values pd.DataFrames, float, list).
    """

    beh_feature_data_dict = {}
    camera_fr_dict = {}
    mouse_track_names_dict = {}
    for behavior_file_path in behavior_file_paths:
        features_csv_file_path = next(pathlib.Path(f"{behavior_file_path}{os.sep}video{os.sep}").glob(f"**{os.sep}*_points3d_translated_rotated_metric_behavioral_features.csv"), None)
        track_file_path = next(pathlib.Path(f"{behavior_file_path}{os.sep}video{os.sep}").glob(f"**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5"), None)
        with h5py.File(name=track_file_path, mode='r') as h5_file_mouse_obj:
            camera_fr_dict[behavior_file_path.split(os.sep)[-1]] = float(h5_file_mouse_obj['recording_frame_rate'][()])
            mouse_track_names_dict[behavior_file_path.split(os.sep)[-1]] = [item.decode('utf-8') for item in list(h5_file_mouse_obj['track_names'])]
        beh_feature_data_dict[behavior_file_path.split(os.sep)[-1]] = pls.read_csv(source=features_csv_file_path, separator=csv_sep)

    return beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict


def load_usv_info_data(root_directories: list = None,
                       mouse_ids_dict: dict = None,
                       camera_fps_dict: dict = None,
                       features_dict: dict = None,
                       csv_sep: str = ',',
                       rate_smoothing_sd: int | float = None,
                       filter_history: int | float = None,
                       clean_filter_history: bool = None,
                       consider_bouts_bool: bool = None,
                       usv_bout_time: int | float = None,
                       min_usv_per_bout: int = None) -> dict:
    """
    Loads USV information data from a .csv file.

    Parameters
    ----------
    root_directories : list
        Root directories of the input sessions.
    mouse_ids_dict : dict
        Sessions with mouse ID lists.
    camera_fps_dict : dict
        Sessions with camera frame rates.
    features_dict : dict
        Sessions with behavioral feature data.
    csv_sep : str, optional
        Separator used in the .csv file.
    rate_smoothing_sd : int / float
        Smoothing sigma (in bins) for USV rate.
    filter_history : int / float
        Amount of time preceding each USV onset.
    clean_filter_history : bool
        Whether to use only USVs which are NOT preceded by other USVs in filter history time.
    consider_bouts_bool : bool
        Take into account USV bouts instead of single USVs.
    usv_bout_time : int / float
        Duration of a USV bout (in s).
    min_usv_per_bout : int
        Minimal number of USVs within a bout.

    Returns
    -------
    usv_data_dict : dict
        Nested dictionary: session - mouseID - USV start/stop.
    """

    usv_data_dict = {}
    for one_root_directory in root_directories:
        session_id = one_root_directory.split(os.sep)[-1]
        usv_data_dict[session_id] = {}
        usv_summary_data = pls.read_csv(source=next(pathlib.Path(f"{one_root_directory}{os.sep}audio{os.sep}").glob(f"**{os.sep}*_usv_summary.csv"), None),
                                        separator=csv_sep)
        mouse_track_names = mouse_ids_dict[session_id]

        for mouse_name in mouse_track_names:
            usv_data_dict[session_id][mouse_name] = {}

            # finds start and stop times of USVs for this particular mouse
            usv_data_dict[session_id][mouse_name]['start'] = np.array(usv_summary_data.filter(pls.col('emitter') == mouse_name)['start'])
            usv_data_dict[session_id][mouse_name]['stop'] = np.array(usv_summary_data.filter(pls.col('emitter') == mouse_name)['stop'])

            # these are the theoretical upper limit of USVs for this mouse (so categorized to that mouse + all the uncategorized ones)
            usv_start_uncategorized_joined = np.array(usv_summary_data.filter((pls.col('emitter').is_null()) | (pls.col('emitter') == mouse_name))['start'])
            usv_stop_uncategorized_joined = np.array(usv_summary_data.filter((pls.col('emitter').is_null()) | (pls.col('emitter') == mouse_name))['stop'])

            # here, we binarize the USV data: 1 signifies USV is happening in this tracking frame, 0 means no USV
            session_duration_frames = features_dict[session_id].shape[0]
            usv_frame_events = np.zeros(session_duration_frames, dtype=float)
            for usv_idx in range(usv_data_dict[session_id][mouse_name]['start'].size):
                event_start_frame = max(0, np.floor(usv_data_dict[session_id][mouse_name]['start'][usv_idx] * camera_fps_dict[session_id]).astype(int))
                event_stop_frame = min(session_duration_frames, np.ceil(usv_data_dict[session_id][mouse_name]['stop'][usv_idx] * camera_fps_dict[session_id]).astype(int))

                if event_start_frame < event_stop_frame:
                    usv_frame_events[event_start_frame:event_stop_frame] = 1

            # here we compute an instantaneous USV rate, which is in number of USVs per second
            astropy_kernel_1d = Gaussian1DKernel(stddev=rate_smoothing_sd)
            usv_frame_rate = convolve(data=usv_frame_events / (1 / camera_fps_dict[session_id]),
                                      kernel=astropy_kernel_1d,
                                      boundary='extend',
                                      nan_treatment='interpolate',
                                      preserve_nan=True)

            usv_data_dict[session_id][mouse_name]['usv_count'] = usv_frame_events
            usv_data_dict[session_id][mouse_name]['usv_rate'] = usv_frame_rate

            """
            clean history here means that there is NO USV event in the time preceding the current USV event for the duration of filter_history
            we use USV starts as a reference for the USV, but we could use USV stops as well

            there are 3 possible research questions here:
            (1) USV and no-USV difference (NO event in the filter history period for BOTH)
            (2) USV and no-USV difference (USV can be present only in the pre-USV period)
            (3) USV and no-USV difference (USV Can be present in BOTH) - here you are basically trying to see whether you can decode if a vocalization is happening or not

            we need a more robust method to filter:
            finding USVs (starts/ends, should do both) can be: (1) find all USVs, (2) find USVs which are NOT preceded by another USV in the time preceding the current USV event for the duration of filter_history
            finding NO-USV epochs: (1) NO USVs in the filter history period (2) USV can be present in the NO-USV period (flag whether NO-USV epochs can be duplicated)
            filter additionally by behavior: e.g., exclude epochs where the animals are too distant, etc.

            Pseudo-code for NO-USV epochs:
            - store start/stop of each USV event in tracking frames (easily done above)
            """

            if clean_filter_history:
                usv_starts_filter_bool = np.concatenate((np.array([True]), usv_data_dict[session_id][mouse_name]['start'][1:] - usv_data_dict[session_id][mouse_name]['stop'][:-1] > filter_history), axis=0) & (usv_data_dict[session_id][mouse_name]['start'] > filter_history)
            else:
                usv_starts_filter_bool = (usv_data_dict[session_id][mouse_name]['start'] > filter_history)
            usv_starts_filtered = usv_data_dict[session_id][mouse_name]['start'][usv_starts_filter_bool]

            # the following segment eliminates USV events which are not followed by a bout of specified duration
            if consider_bouts_bool:
                usv_bouts = []
                for idx, usv_start in enumerate(usv_starts_filtered):
                    if np.sum((usv_data_dict[session_id][mouse_name]['start'] >= usv_start) & (usv_data_dict[session_id][mouse_name]['start'] < usv_start + usv_bout_time)) > min_usv_per_bout:
                        usv_bouts.append(usv_start)
                usv_starts_filtered = np.array(usv_bouts)

            usv_data_dict[session_id][mouse_name]['glm_usv'] = usv_starts_filtered

            # find an equal number, or fewer if former is impossible, time points where no USV bout occurs
            no_usv_epochs = np.array(np.zeros(usv_start_uncategorized_joined.size))
            no_usv_epochs[:] = np.nan

            # filter out no-USV epochs
            for noe_idx in range(usv_starts_filtered.size):
                valid_value_not_found = True
                counter = 0
                while valid_value_not_found:
                    if counter < 1000:
                        potential_no_usv = np.random.uniform(low=filter_history, high=features_dict[session_id].shape[0] / camera_fps_dict[session_id])
                        outside_wider_usv_range = (np.abs(potential_no_usv - usv_stop_uncategorized_joined) > filter_history).all()
                        # no_overlap_existing_values = (np.abs(potential_no_usv - no_usv_epochs[~np.isnan(no_usv_epochs)]) > filter_history).all()
                        if outside_wider_usv_range and potential_no_usv not in no_usv_epochs:
                            no_usv_epochs[noe_idx] = potential_no_usv
                            valid_value_not_found = False
                        else:
                            counter += 1
                    else:
                        break

            no_usv_epochs = np.sort(no_usv_epochs[~np.isnan(no_usv_epochs)])
            usv_data_dict[session_id][mouse_name]['glm_none'] = no_usv_epochs

    return usv_data_dict

def load_pickle_modeling_data(pickle_file_path):
    """
    Loads data from a .pickle file.

    Parameters
    ----------
    pickle_file_path : str
        Path to the .pickle file.

    Returns
    -------
    modeling_data : dict
        Modeling data.
    """

    with open(pickle_file_path, 'rb') as pickle_file:
        modeling_data = pickle.load(pickle_file)

    return modeling_data
