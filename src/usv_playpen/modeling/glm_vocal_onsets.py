"""
@author: bartulem
Run GLM to predict vocal output w/ behavioral features.
"""

from datetime import datetime
import json
import numpy as np
import os
import pathlib
import polars as pls
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from .glm_cross_session_normalization import zscore_different_sessions_together
from .glm_bases_functions import raised_cosine
from .glm_load_input_files import load_behavioral_feature_data, load_usv_info_data
from ..analyses.compute_behavioral_features import FeatureZoo


class GeneralizedLinearModelPipeline(FeatureZoo):

    def __init__(self, glm_settings_dict, **kwargs):
        """
        Initializes the GeneralizedLinearModelOnsets class.

        Parameter
        ---------
        glm_settings_dict (dict, optional):
            Dictionary containing GLM settings.
        ---------

        Returns
        -------
        -------
        """

        if glm_settings_dict is None:
            with open(pathlib.Path(__file__).parent.parent / '_parameter_settings/glm_settings.json', 'r') as settings_json_file:
                self.glm_settings_dict = json.load(settings_json_file)['extract_and_save_glm_input_data_']
        else:
            self.glm_settings_dict = glm_settings_dict

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def extract_and_save_glm_input_data_(self) -> None:
        """
        Description
        ----------
        This method extracts and saves data necessary to run GLM analyses
        on vocal onsets in the appropriate format.
        ----------

        Parameters
        ----------
        You can set parameters in the 'glm_settings.json' file located in the
        '_parameter_settings' folder. Parameters include:
            'session_paths' : list
                Paths to the root directories of relevant sessions.
            'csv_sheet_delimiter' : str
                Delimiter used in the csv files with behavioral features AND vocalizations.
            'camera_sampling_rate' : float
                Camera sampling rate (in frames/s).
            'filter_history' : float
                Time window (in seconds) prior to vocal onset to consider.
            'random_seed' : int or None
                Random seed for reproducibility. If None, no seed is set.
            'cosine_bases_params' : dict
                Parameters for the raised cosine basis functions, including:
                    'neye' : int
                        Number of identity basis vectors at front.
                    'ncos' : int
                        Number of vectors that are raised cosines. Cannot be 0 or negative.
                    'kpeaks_proportion' : float
                        Proportion of the filter history to set the peak of the last
                        cosine basis (the first peak is always at 0).
                    'b' : int
                        Offset for nonlinear scaling.  larger values -> more linear
                        scaling of vectors.
                    'plot_bool' : bool
                        Whether to plot the basis functions or not.
            'vocal_rate_smoothing_sd' : float
                Standard deviation (in seconds) of the Gaussian kernel used to
                smooth vocalization rates.
            'clean_filter_history_bool' : bool
                Whether to use ONLY USVs which are NOT preceded by other USVs in filter history time.
            'consider_bouts_bool' : bool
                Whether to consider USV bouts (True) or individual USVs (False).
            'usv_bout_time' : float
                Maximum silent interval (in seconds) between two consecutive USVs
                to consider them part of the same bout (only relevant if
                'consider_bouts_bool' is True).
            'min_usv_per_bout' : int
                Minimum number of USVs to consider a sequence of USVs a bout
                (only relevant if 'consider_bouts_bool' is True).
            'predictor_mouse' : int
                Index of the mouse whose behavioral features will be used as predictors;
                automatically, the other mouse will be the one whose vocal onsets are predicted.
            'behavioral_predictors' : list
                Behavioral features to consider as predictors.
            'include_1st_der_features_bool' : bool
                Whether to include the 1st derivative of each behavioral feature.
            'include_2nd_der_features_bool' : bool
                Whether to include the 2nd derivative of each behavioral feature./
            'include_predictor_mouse_vocalizations' : str
                Whether to include the vocalizations of the predictor mouse as a feature,
                and if so, whether to include them as 'rate' or 'event'. If None, vocalizations are not included.
        ----------

        Returns
        ----------
        glm_results_dict : .pkl file
            Results of the GLM for each feature separately saved in pickle file.
        ----------
        """

        # set random seed
        if self.glm_settings_dict['random_seed'] is not None:
            np.random.seed(self.glm_settings_dict['random_seed'])
        else:
            np.random.seed(None)

        # set and plot bases parameters
        cosine_bases = raised_cosine(neye=self.glm_settings_dict['cosine_bases_params']['neye'],
                                     ncos=self.glm_settings_dict['cosine_bases_params']['ncos'],
                                     kpeaks=[0, int(np.floor(self.glm_settings_dict['filter_history'] * self.glm_settings_dict['camera_sampling_rate'] * self.glm_settings_dict['cosine_bases_params']['kpeaks_proportion']))],
                                     b=self.glm_settings_dict['cosine_bases_params']['b'],
                                     w=self.glm_settings_dict['filter_history'] * self.glm_settings_dict['camera_sampling_rate'])

        if self.glm_settings_dict['cosine_bases_params']['plot_bool']:
            plt.figure(figsize=(5, 3))
            plt.plot(np.arange(-(self.glm_settings_dict['filter_history'] * self.glm_settings_dict['camera_sampling_rate']), 0), cosine_bases, lw=1)
            plt.title('Raised cosine bases')
            plt.xlabel('Delay (fr)')
            plt.show()

        # load behavioral feature data files
        beh_feature_data_dict, camera_fr_dict, mouse_track_names_dict = load_behavioral_feature_data(behavior_file_paths=self.glm_settings_dict['session_paths'],
                                                                                                     csv_sep=self.glm_settings_dict['csv_sheet_delimiter'])

        # load USV summary data
        usv_data_dict = load_usv_info_data(root_directories=self.glm_settings_dict['session_paths'],
                                           mouse_ids_dict=mouse_track_names_dict,
                                           camera_fps_dict=camera_fr_dict,
                                           features_dict=beh_feature_data_dict,
                                           csv_sep=self.glm_settings_dict['csv_sheet_delimiter'],
                                           rate_smoothing_sd=self.glm_settings_dict['vocal_rate_smoothing_sd'],
                                           filter_history=self.glm_settings_dict['filter_history'],
                                           clean_filter_history=self.glm_settings_dict['clean_filter_history_bool'],
                                           consider_bouts_bool=self.glm_settings_dict['consider_bouts_bool'],
                                           usv_bout_time=self.glm_settings_dict['usv_bout_time'],
                                           min_usv_per_bout=self.glm_settings_dict['min_usv_per_bout'])

        # in each dataframe, keep only behavioral features of interest and drop others
        revised_behavioral_predictors = []
        for behavioral_session in beh_feature_data_dict.keys():
            columns_to_keep = []
            predictor_mouse_name = mouse_track_names_dict[behavioral_session][self.glm_settings_dict['predictor_mouse']]
            test_mouse_name = mouse_track_names_dict[behavioral_session][abs(self.glm_settings_dict['predictor_mouse'] - 1)]
            for feature in beh_feature_data_dict[behavioral_session].columns:
                if feature.split('.')[1] in self.glm_settings_dict['behavioral_predictors']:
                    if feature.split('.')[0] == predictor_mouse_name:
                        columns_to_keep.append(feature)
                        if feature.split('.')[-1] not in revised_behavioral_predictors:
                            revised_behavioral_predictors.append(feature.split('.')[-1])
                        if feature.split('.')[1] != 'speed' and feature.split('.')[1] != 'acceleration':
                            if self.glm_settings_dict['include_1st_der_features_bool']:
                                columns_to_keep.append(f'{feature}_1st_der')
                                if f"{feature.split('.')[-1]}_1st_der" not in revised_behavioral_predictors:
                                    revised_behavioral_predictors.append(f"{feature.split('.')[-1]}_1st_der")
                            if self.glm_settings_dict['include_2nd_der_features_bool']:
                                columns_to_keep.append(f'{feature}_2nd_der')
                                if f"{feature.split('.')[-1]}_2nd_der" not in revised_behavioral_predictors:
                                    revised_behavioral_predictors.append(f"{feature.split('.')[-1]}_2nd_der")
                    elif 'diff' in feature:
                        columns_to_keep.append(feature)
                        if feature.split('.')[-1] not in revised_behavioral_predictors:
                            revised_behavioral_predictors.append(feature.split('.')[-1])
                        if self.glm_settings_dict['include_1st_der_features_bool']:
                            columns_to_keep.append(f'{feature}_1st_der')
                            if f"{feature.split('.')[-1]}_1st_der" not in revised_behavioral_predictors:
                                revised_behavioral_predictors.append(f"{feature.split('.')[-1]}_1st_der")
                        if self.glm_settings_dict['include_2nd_der_features_bool']:
                            columns_to_keep.append(f'{feature}_2nd_der')
                            if f"{feature.split('.')[-1]}_2nd_der" not in revised_behavioral_predictors:
                                revised_behavioral_predictors.append(f"{feature.split('.')[-1]}_2nd_der")
                    elif ('-' in feature.split('.')[0] and
                          feature.split('.')[1].split('-')[self.glm_settings_dict['predictor_mouse']] != 'allo_yaw' and
                          feature.split('.')[1].split('-')[1 - self.glm_settings_dict['predictor_mouse']] != 'TTI'):
                        columns_to_keep.append(feature)
                        if feature.split('.')[-1] not in revised_behavioral_predictors:
                            revised_behavioral_predictors.append(feature.split('.')[-1])
                        if self.glm_settings_dict['include_1st_der_features_bool']:
                            columns_to_keep.append(f'{feature}_1st_der')
                            if f"{feature.split('.')[-1]}_1st_der" not in revised_behavioral_predictors:
                                revised_behavioral_predictors.append(f"{feature.split('.')[-1]}_1st_der")
                        if self.glm_settings_dict['include_2nd_der_features_bool']:
                            columns_to_keep.append(f'{feature}_2nd_der')
                            if f"{feature.split('.')[-1]}_2nd_der" not in revised_behavioral_predictors:
                                revised_behavioral_predictors.append(f"{feature.split('.')[-1]}_2nd_der")

            if self.glm_settings_dict['include_predictor_mouse_vocalizations'] == 'rate':
                beh_feature_data_dict[behavioral_session] = beh_feature_data_dict[behavioral_session].select(columns_to_keep).with_columns(pls.Series(f"{predictor_mouse_name}.usv_other",
                                                                                                                                                      usv_data_dict[behavioral_session][predictor_mouse_name]['usv_rate']),
                                                                                                                                           pls.Series(f"{test_mouse_name}.usv_self",
                                                                                                                                                      usv_data_dict[behavioral_session][test_mouse_name]['usv_rate']))
                if 'usv_other' not in revised_behavioral_predictors:
                    revised_behavioral_predictors.append('usv_other')
                if 'usv_self' not in revised_behavioral_predictors:
                    revised_behavioral_predictors.append('usv_self')
            elif self.glm_settings_dict['include_predictor_mouse_vocalizations'] == 'event':
                beh_feature_data_dict[behavioral_session] = beh_feature_data_dict[behavioral_session].select(columns_to_keep).with_columns(pls.Series(f"{predictor_mouse_name}.usv_other",
                                                                                                                                                      usv_data_dict[behavioral_session][predictor_mouse_name]['usv_count']),
                                                                                                                                           pls.Series(f"{test_mouse_name}.usv_self",
                                                                                                                                                      usv_data_dict[behavioral_session][test_mouse_name]['usv_count']))
                if 'usv_other' not in revised_behavioral_predictors:
                    revised_behavioral_predictors.append('usv_other')
                if 'usv_self' not in revised_behavioral_predictors:
                    revised_behavioral_predictors.append('usv_self')
            else:
                beh_feature_data_dict[behavioral_session] = beh_feature_data_dict[behavioral_session].select(columns_to_keep)

        # z-score behavioral data
        beh_feature_data_dict = zscore_different_sessions_together(data_dict=beh_feature_data_dict,
                                                                   feature_lst=revised_behavioral_predictors,
                                                                   feature_bounds=self.feature_boundaries)

        # get all behavioral feature data in arrays around USV and no-USV times
        glm_feature_arr_dict = {}
        for beh_feature in tqdm(revised_behavioral_predictors, desc=f'GLM data extraction progress (per feature)'):
            glm_feature_arr_dict[beh_feature] = {}
            for session_idx, beh_session_id in enumerate(beh_feature_data_dict.keys()):
                glm_feature_arr_dict[beh_feature][beh_session_id] = {}
                session_feature_match = [match for match in beh_feature_data_dict[beh_session_id].columns if match.endswith(beh_feature)][0]

                usv_starts_filtered = usv_data_dict[beh_session_id][mouse_track_names_dict[beh_session_id][abs(self.glm_settings_dict['predictor_mouse']-1)]]['glm_usv']
                no_usv_epochs = usv_data_dict[beh_session_id][mouse_track_names_dict[beh_session_id][abs(self.glm_settings_dict['predictor_mouse']-1)]]['glm_none']

                # select a behavioral feature of interest in those time sequences
                history_frames = int(np.floor(camera_fr_dict[beh_session_id] * self.glm_settings_dict['filter_history']))

                usv_feature_arr = np.zeros((usv_starts_filtered.size, history_frames))
                usv_feature_arr[:] = np.nan
                no_usv_feature_arr = np.zeros((no_usv_epochs.size, history_frames))
                no_usv_feature_arr[:] = np.nan

                for usv_event_idx in range(usv_starts_filtered.size):
                    usv_event_start = int(np.floor(camera_fr_dict[beh_session_id] * usv_starts_filtered[usv_event_idx]))
                    usv_feature_arr_temp = beh_feature_data_dict[beh_session_id].slice(usv_event_start - history_frames, history_frames).select(f"{session_feature_match}").to_series().to_numpy().copy()

                    nan_indices = np.where(np.isnan(usv_feature_arr_temp))[0]
                    usv_feature_arr_temp[nan_indices] = np.nanmean(usv_feature_arr_temp)

                    usv_feature_arr[usv_event_idx, :] = usv_feature_arr_temp

                for no_usv_event_idx in range(no_usv_epochs.size):
                    no_usv_event_start = int(np.floor(camera_fr_dict[beh_session_id] * no_usv_epochs[no_usv_event_idx]))
                    no_usv_feature_arr_temp = beh_feature_data_dict[beh_session_id].slice(no_usv_event_start - history_frames, history_frames).select(f"{session_feature_match}").to_series().to_numpy().copy()

                    np_usv_nan_indices = np.where(np.isnan(no_usv_feature_arr_temp))[0]
                    no_usv_feature_arr_temp[np_usv_nan_indices] = np.nanmean(no_usv_feature_arr_temp)

                    no_usv_feature_arr[no_usv_event_idx, :] = no_usv_feature_arr_temp

                glm_feature_arr_dict[beh_feature][beh_session_id]['usv_feature_arr'] = usv_feature_arr
                glm_feature_arr_dict[beh_feature][beh_session_id]['no_usv_feature_arr'] = no_usv_feature_arr

        # save data as pkl file
        if self.glm_settings_dict['predictor_mouse'] == 0:
            target_mouse_ = 'female'
        else:
            target_mouse_ = 'male'

        file_name_ = f"glm_usv_onsets_{target_mouse_}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.glm_settings_dict['session_paths'])}_sessions.pkl"
        with open(f"{self.glm_settings_dict['save_dir']}{os.sep}{file_name_}", 'wb') as glm_feature_arr_pickle:
            pickle.dump(glm_feature_arr_dict, glm_feature_arr_pickle)

        #     # concatenate data from different sessions
        #     usv_total_arr, no_usv_total_arr = 0, 0
        #     for session_num, session in enumerate(glm_feature_arr_dict.keys()):
        #         if session_num == 0:
        #             usv_total_arr = glm_feature_arr_dict[session]['usv_feature_arr'].copy()
        #             no_usv_total_arr = glm_feature_arr_dict[session]['no_usv_feature_arr'].copy()
        #         else:
        #             usv_total_arr = np.concatenate((usv_total_arr, glm_feature_arr_dict[session]['usv_feature_arr']), axis=0)
        #             no_usv_total_arr = np.concatenate((no_usv_total_arr, glm_feature_arr_dict[session]['no_usv_feature_arr']), axis=0)
        #
        #     # select equal number of USV and no-USV events
        #     num_of_events = min(usv_total_arr.shape[0], no_usv_total_arr.shape[0])
        #
        #     rand_indices_usv = np.sort(np.random.choice(np.arange(0, usv_total_arr.shape[0], 1), size=num_of_events, replace=False))
        #     usv_arr_subsampled = np.take(usv_total_arr, rand_indices_usv, axis=0)
        #
        #     rand_indices_no_usv = np.sort(np.random.choice(np.arange(0, no_usv_total_arr.shape[0], 1), size=num_of_events, replace=False))
        #     no_usv_arr_subsampled = np.take(no_usv_total_arr, rand_indices_no_usv, axis=0)
        #
        #     # if beh_feature == 'nose-nose':
        #     #     print(usv_arr_subsampled.shape)
        #     #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2), dpi=300, tight_layout=True)
        #     #     ax.plot(usv_arr_subsampled.mean(axis=0), color='#31B096', lw=1.0)
        #     #     ax.plot(no_usv_arr_subsampled.mean(axis=0), color='#D3D3D3', lw=1.0)
        #     #     # for iter_ in range(no_usv_arr_subsampled.shape[0]):
        #     #     #     ax.plot(usv_arr_subsampled[iter_, :], color='#D3D3D3', lw=.001)
        #     #     ax.set_title(f"USV vs NO-USV split for feature {beh_feature}", fontsize=10)
        #     #     # ax.set_xticks(np.arange(0, 350, 75), np.round(np.arange(-2, 0.5, 0.5), decimals=2))
        #     #     ax.set_xlabel(xlabel='Time prior to event (s)', fontsize=8)
        #     #     ax.set_ylabel(ylabel='z-score', fontsize=8)
        #     #     plt.show()
        #
        #     # set uo and conduct logistic regression
        #     glm_results_dict[beh_feature]['filter_shapes'] = np.zeros((self.num_splits, self.filter_history * 150))
        #     glm_results_dict[beh_feature]['score'] = np.zeros(self.num_splits)
        #     glm_results_dict[beh_feature]['precision'] = np.zeros(self.num_splits)
        #     glm_results_dict[beh_feature]['recall'] = np.zeros(self.num_splits)
        #     glm_results_dict[beh_feature]['f1'] = np.zeros(self.num_splits)
        #     glm_results_dict[beh_feature]['auc'] = np.zeros(self.num_splits)
        #     glm_results_dict[beh_feature]['ll'] = np.zeros(self.num_splits)
        #     for split_idx in tqdm(range(self.num_splits), desc=f'GLM estimate progress for {beh_feature}'):
        #         test_size = int(np.floor(num_of_events * self.test_proportion))
        #         train_size = num_of_events - test_size
        #
        #         all_indices = np.arange(num_of_events).astype(np.int32)
        #         np.random.shuffle(all_indices)
        #         usv_train_indices = all_indices[:train_size]
        #         usv_test_indices = all_indices[train_size:]
        #
        #         usv_train_data = usv_arr_subsampled[usv_train_indices, :]
        #         usv_train_labels = np.ones(train_size)
        #         usv_test_data = usv_arr_subsampled[usv_test_indices, :]
        #         usv_test_labels = np.ones(test_size)
        #
        #         np.random.shuffle(all_indices)
        #         no_usv_train_indices = all_indices[:train_size]
        #         no_usv_test_indices = all_indices[train_size:]
        #
        #         no_usv_train_data = no_usv_arr_subsampled[no_usv_train_indices, :]
        #         no_usv_train_labels = np.zeros(train_size)
        #         no_usv_test_data = no_usv_arr_subsampled[no_usv_test_indices, :]
        #         no_usv_test_labels = np.zeros(test_size)
        #
        #         train_data_concatenated = np.concatenate((usv_train_data, no_usv_train_data), axis=0)
        #         train_labels_concatenated = np.concatenate((usv_train_labels, no_usv_train_labels), axis=0)
        #
        #         train_indices_joint = np.arange(train_labels_concatenated.shape[0]).astype(np.int32)
        #         np.random.shuffle(train_indices_joint)
        #
        #         test_data_concatenated = np.concatenate((usv_test_data, no_usv_test_data), axis=0)
        #         test_labels_concatenated = np.concatenate((usv_test_labels, no_usv_test_labels), axis=0)
        #
        #         test_indices_joint = np.arange(test_labels_concatenated.shape[0]).astype(np.int32)
        #         np.random.shuffle(test_indices_joint)
        #
        #         x_train = train_data_concatenated[train_indices_joint, :]
        #         x_test = test_data_concatenated[test_indices_joint, :]
        #         y_train = train_labels_concatenated[train_indices_joint]
        #         y_test = test_labels_concatenated[test_indices_joint]
        #
        #         x_train = np.dot(x_train, cosine_bases)
        #         x_test = np.dot(x_test, cosine_bases)
        #
        #         lr = LogisticRegressionCV(penalty=self.logistic_regression_params['penalty'],
        #                                   Cs=self.logistic_regression_params['cs'],
        #                                   cv=self.logistic_regression_params['cv'],
        #                                   class_weight='balanced',
        #                                   solver=self.logistic_regression_params['solver'],
        #                                   max_iter=self.logistic_regression_params['max_iter'],
        #                                   multi_class=self.logistic_regression_params['multi_class']).fit(x_train, y_train)
        #
        #         glm_results_dict[beh_feature]['filter_shapes'][split_idx] = np.ravel(np.dot(lr.coef_, cosine_bases.T))
        #         glm_results_dict[beh_feature]['score'][split_idx] = lr.score(x_test, y_test)
        #         glm_results_dict[beh_feature]['precision'][split_idx] = precision_score(y_test, lr.predict(x_test))
        #         glm_results_dict[beh_feature]['recall'][split_idx] = recall_score(y_test, lr.predict(x_test))
        #         glm_results_dict[beh_feature]['f1'][split_idx] = f1_score(y_test, lr.predict(x_test), average='micro')
        #         glm_results_dict[beh_feature]['auc'][split_idx] = roc_auc_score(y_test, lr.predict_proba(x_test)[:, 1])
        #         glm_results_dict[beh_feature]['ll'][split_idx] = log_loss(y_test, lr.predict_proba(x_test))
        #
        # with open(f"{self.save_dir}{os.sep}{self.save_file_name}_{num_of_events}events.pkl", 'wb') as glm_pickle:
        #     pickle.dump(glm_results_dict, glm_pickle)
