"""
@author: bartulem
Make neuronal tuning curves for 3D behavioral features.
"""

from PyQt6.QtTest import QTest
from datetime import datetime
import glob
import h5py
import numpy as np
import os
import pathlib
import pickle
import polars as pls
from tqdm import tqdm
from typing import Tuple
from .compute_behavioral_features import FeatureZoo

def generate_ratemaps(feature_arr: np.ndarray = None,
                      spike_arr: np.ndarray = None,
                      shuffled_spike_arr: np.ndarray = None,
                      min_val: int = None,
                      max_val: int = None,
                      num_bins: int = None,
                      camera_fr: int|float = None,
                      space_bool: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    ----------
    This function computes occupancy stats
    and spike counts for a given feature array.
    ----------

    Parameters
    ----------
    feature_arr (np.ndarray)
        A (n_frames) shape ndarray containing behavioral feature data.
    spike_arr (np.ndarray)
        A (n_spikes) shape ndarray containing spike event frames.
    shuffled_spike_arr (np.ndarray)
        A (n_shuffles, n_frames) shape ndarray containing shuffled spike event frames.
    min_val (int)
        Minimum possible value feature could attain.
    max_val (int)
        Maximum possible value feature could attain.
    num_bins (int)
        Number of bins to divide features in.
    camera_fr (int / float)
        Camera frame rate.
    space_bool (bool)
        Boolean indicating if feature is spatial.
    ----------

    Returns
    ----------
    ratemap: np.ndarray
        A (n_bins, 2) shape ndarray containing spike counts and occupancy (in seconds)
        for each feature bin (first column spike counts, second column occ).
    sh_counts: np.ndarray
        A (n_bins) shape ndarray containing shuffled spike counts
    bin_centers: np.ndarray
        A (n_bins) shape ndarray containing bin centers for given feature.
    bin_edges: np.ndarray
        A (n_bins) shape ndarray containing bin edges for given feature.
    ----------
    """

    if space_bool:
        bins_in_one_dir = int(np.ceil(np.sqrt(num_bins)))
        bin_edges = np.linspace(min_val, max_val, bins_in_one_dir + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        spike_events_x = np.take(a=feature_arr[:, 0],
                                 indices=spike_arr)
        spike_events_y = np.take(a=feature_arr[:, 1],
                                 indices=spike_arr)
        ratemap = np.zeros((bins_in_one_dir, bins_in_one_dir, 2))
        for i in range(1, np.shape(bin_edges)[0], 1):
            for j in range(1, np.shape(bin_edges)[0], 1):
                ratemap[i - 1, j - 1, 0] = np.sum(((spike_events_x > bin_edges[i - 1]) * (spike_events_x <= bin_edges[i]))
                                                  * ((spike_events_y > bin_edges[j - 1]) * (spike_events_y <= bin_edges[j])))
                ratemap[i - 1, j - 1, 1] = np.sum(((feature_arr[:, 0] > bin_edges[i - 1]) * (feature_arr[:, 0] <= bin_edges[i]))
                                                  * ((feature_arr[:, 1] > bin_edges[j - 1]) * (feature_arr[:, 1] <= bin_edges[j]))) / camera_fr

        return ratemap, bin_centers, bin_edges

    else:
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        spike_event_features = np.take(a=feature_arr,
                                       indices=spike_arr)
        ratemap = np.zeros((num_bins, 2))
        for i in range(1, np.shape(bin_edges)[0], 1):
            ratemap[i - 1, 0] = np.sum((spike_event_features > bin_edges[i - 1]) * (spike_event_features <= bin_edges[i]))
            ratemap[i - 1, 1] = np.sum((feature_arr > bin_edges[i - 1]) * (feature_arr <= bin_edges[i])) / camera_fr

        # shuffled
        sh_counts = np.zeros((shuffled_spike_arr.shape[0], num_bins))
        for sh in range(shuffled_spike_arr.shape[0]):
            spike_features_sh = np.take(a=feature_arr,
                                        indices=shuffled_spike_arr[sh, :])
            for i in range(1, np.shape(bin_edges)[0], 1):
                sh_counts[sh, i - 1] = np.sum((spike_features_sh > bin_edges[i - 1]) * (spike_features_sh <= bin_edges[i]))

        return ratemap, sh_counts, bin_centers, bin_edges


def shuffle_spikes(spike_array: np.ndarray = None,
                   total_fr_num: int = None,
                   shuffle_min_fr: int = None,
                   shuffle_max_fr: int = None,
                   n_shuffles: int = None) -> np.ndarray:
    """
    Description
    ----------
    This function takes in an array containing spike times (converted
    to frames) and shuffles these timepoints by a random amount in
    each shuffle.

    Nb: This shuffling method takes spike times that exceed tracking
    end time and wraps them around to the beginning of the recording.
    ----------

    Parameters
    ----------
    spike_array (np.ndarray)
         A (n_spikes) shape ndarray containing spike event frames.
    total_fr_num (int)
        Total number of frames in the recording.
    shuffle_min_fr (int)
        Minimum number of frames to shuffle spikes by.
    shuffle_max_fr (int)
        Maximum number of frames to shuffle spikes by.
    n_shuffles (int)
        Number of shuffles to perform.
    ----------

    Returns
    ----------
    shuffled_spike_arr (np.ndarray)
        A (n_shuffles, n_frames) shape ndarray containing shuffled spike event frames.
    ----------
    """

    shuffled_spike_arr = np.tile(spike_array, reps=(n_shuffles, 1))
    shuffled_amounts = np.random.randint(low=shuffle_min_fr,
                                         high=shuffle_max_fr,
                                         size=n_shuffles,
                                         dtype=np.int32)
    shuffled_spike_arr = shuffled_spike_arr + shuffled_amounts[:, np.newaxis]
    shuffled_spike_arr[shuffled_spike_arr >= total_fr_num] = shuffled_spike_arr[shuffled_spike_arr >= total_fr_num] - total_fr_num
    shuffled_spike_arr = np.sort(shuffled_spike_arr, axis=1)

    return shuffled_spike_arr


class NeuronalTuning(FeatureZoo):

    def __init__(self, **kwargs):
        """
        Initializes the NeuronalTuning class.

        Parameter
        ---------
        root_directory : str
            Root directory for data; defaults to None.
        tuning_parameters_dict : dict
            Dictionary of behavioral parameters; defaults to None.
        message_output : function
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def calculate_neuronal_tuning_curves(self) -> None:
        """
        Description
        ----------
        This method calculates neuronal tuning curves for 3D behavioral features.
        ----------

        Parameter
        ---------
        Uses the following set of parameters:
            root_directory : str
                Directory of recording files of interest.
            temporal_offsets : list
                Offsets of interest between spikes and behavior.
            n_shuffles : list
                Number of shuffles; defaults to 1000.
            total_bin_num : int
                Number of bins to divide non-spatial features in.
            n_spatial_bins : int
                Total number of spatial bins
            spatial_scale_cm
                Length from center to edge in cm.

        Returns
        -------
        neuronal_tuning_curves : .pkl
            Pickle file containing all ratemaps and shuffled data.
        """

        self.message_output(f"Computing behavioral tuning curves started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        # load behavioral feature data
        behavioral_data_file = glob.glob(pathname=f"{self.root_directory}{os.sep}**{os.sep}*_behavioral_features.csv*",
                                         recursive=True)[0]
        behavioral_data = pls.read_csv(behavioral_data_file)

        # load mouse and camera frame rate info
        mouse_data_h5 = glob.glob(pathname=f"{self.root_directory}{os.sep}**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5*",
                                  recursive=True)[0]
        with h5py.File(mouse_data_h5, mode='r') as tracking_data_3d:
            animal_ids = [elem.decode('utf-8') for elem in list(tracking_data_3d['track_names'])]
            empirical_camera_sr = float(tracking_data_3d['recording_frame_rate'][()])

        # load spike data
        cluster_file_lst = glob.glob(pathname=f"{self.root_directory}{os.sep}ephys{os.sep}**{os.sep}cluster_data{os.sep}*.npy",
                                     recursive=True)

        for cluster_file in tqdm(cluster_file_lst):
            cluster_data_frames = np.load(file=cluster_file)[1, :]
            cluster_id = os.path.basename(cluster_file)[:-4]

            neuronal_tuning_curves_data = {}
            for one_offset in self.tuning_parameters_dict['temporal_offsets']:
                cluster_data_frames = cluster_data_frames.astype(np.int32) + int(np.floor(one_offset * empirical_camera_sr))
                cluster_data_frames = cluster_data_frames[(cluster_data_frames >= 0) & (cluster_data_frames <= behavioral_data.shape[0])]
                file_name_addendum_offset = f'beh_offset={one_offset}s'
                neuronal_tuning_curves_data[file_name_addendum_offset] = {}

                # shuffle spike data
                cluster_data_shuffled = shuffle_spikes(spike_array=cluster_data_frames,
                                                       total_fr_num=behavioral_data.shape[0],
                                                       shuffle_min_fr=int(np.floor(20 * empirical_camera_sr)),
                                                       shuffle_max_fr=int(np.floor(60 * empirical_camera_sr)),
                                                       n_shuffles=self.tuning_parameters_dict['n_shuffles'])

                # compute tuning curves
                space_computed_rm = {one_mouse: False for one_mouse in animal_ids}
                for column in behavioral_data.columns:
                    if 'space' not in column:
                        neuronal_tuning_curves_data[file_name_addendum_offset][column] = {}
                        (neuronal_tuning_curves_data[file_name_addendum_offset][column]['ratemaps'],
                         neuronal_tuning_curves_data[file_name_addendum_offset][column]['sh_counts'],
                         neuronal_tuning_curves_data[file_name_addendum_offset][column]['bin_centers'],
                         neuronal_tuning_curves_data[file_name_addendum_offset][column]['bin_edges']) = generate_ratemaps(feature_arr=np.array(behavioral_data[column]),
                                                                                                                          spike_arr=cluster_data_frames,
                                                                                                                          shuffled_spike_arr=cluster_data_shuffled,
                                                                                                                          min_val=self.feature_boundaries[column.split('.')[-1]][0],
                                                                                                                          max_val=self.feature_boundaries[column.split('.')[-1]][1],
                                                                                                                          num_bins=self.tuning_parameters_dict['total_bin_num'],
                                                                                                                          camera_fr=empirical_camera_sr,
                                                                                                                          space_bool=False)

                    else:
                        if space_computed_rm[f"{column.split('.')[0]}"] is False:
                            neuronal_tuning_curves_data[file_name_addendum_offset][f"{column.split('.')[0]}.space"] = {}
                            (neuronal_tuning_curves_data[file_name_addendum_offset][f"{column.split('.')[0]}.space"]['ratemaps'],
                             neuronal_tuning_curves_data[file_name_addendum_offset][f"{column.split('.')[0]}.space"]['bin_centers'],
                             neuronal_tuning_curves_data[file_name_addendum_offset][f"{column.split('.')[0]}.space"]['bin_edges']) = generate_ratemaps(feature_arr=np.stack(arrays=(np.array(behavioral_data[f"{column.split('.')[0]}.spaceX"]),
                                                                                                                                                                                    np.array(behavioral_data[f"{column.split('.')[0]}.spaceY"])),
                                                                                                                                                                            axis=1),
                                                                                                                                                       spike_arr=cluster_data_frames,
                                                                                                                                                       shuffled_spike_arr=cluster_data_shuffled,
                                                                                                                                                       min_val=-self.tuning_parameters_dict['spatial_scale_cm'],
                                                                                                                                                       max_val=self.tuning_parameters_dict['spatial_scale_cm'],
                                                                                                                                                       num_bins=self.tuning_parameters_dict['n_spatial_bins'],
                                                                                                                                                       camera_fr=empirical_camera_sr,
                                                                                                                                                       space_bool=True)

                            space_computed_rm[f"{column.split('.')[0]}"] = True

            # save tuning curves to file
            pathlib.Path(f"{self.root_directory}{os.sep}ephys{os.sep}tuning_curves").mkdir(parents=True, exist_ok=True)
            with open(f"{self.root_directory}{os.sep}ephys{os.sep}tuning_curves{os.sep}{cluster_id}_tuning_curves_data.pkl", 'wb') as neuronal_tuning_curves_pkl:
                pickle.dump(neuronal_tuning_curves_data, neuronal_tuning_curves_pkl)
