"""
@author: bartulem
Code to generate summary figure for data preprocessing.
"""

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from imgstore import new_for_filename

plt.style.use('./project.mplstyle')


class SummaryPlotter:

    def __init__(self, input_parameter_dict=None, root_directory=None):
        if root_directory is None:
            with open('input_parameters.json', 'r') as json_file:
                self.root_directory = json.load(json_file)['preprocessing_plot']['root_directory']
        else:
            self.root_directory = root_directory

        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['preprocessing_plot']['SummaryPlotter']
        else:
            self.input_parameter_dict = input_parameter_dict

    def preprocessing_summary(self, prediction_error_dict, phidget_data_dictionary):
        """
        Description
        ----------
        This method generates a plot summarizing the first data preprocessing stage,
        with session details (e.g., name, mice used, duration, etc.), variables
        measure with the phidget device (humidity, illumination, temperature) and
        the error estimates from predicting LED on start times with the Avisoft
        recorder data vs. the actual video frames these events appeared at.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            prediction_error_dict (dict)
                Dict containing arrays w/ prediction errors.
            phidget_data_dictionary (dict)
                Dictionary containing lux, humidity and temperature data.
            duration_min (int / float)
                Actual duration of the experiment (camera w/ the least frames).
            root_directory (str)
                Root directory for a recording session.
            hist_c (str)
                Color for the histogram bars.
            hist_ec (str)
                Edge color for the histogram bars.
            median_err_c (str)
                Median error color.
            mean_err_c (str)
                Mean error color.
        ----------

        Returns
        ----------
        preprocessing_plot (fig)
            Figure summarizing the preprocessing of experimental data.
        ----------
        """

        # get the total number of frames in the video
        json_loc = glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json")[0]
        with open(json_loc, 'r') as camera_count_json_file:
            duration_min = json.load(camera_count_json_file)['total_video_time_least']

        # calculate error statistics
        plot_statistics_dict = {}
        for device_id in prediction_error_dict.keys():
            plot_statistics_dict[device_id] = {}
            plot_statistics_dict[device_id]['error_median'] = np.round(np.nanmedian(prediction_error_dict[device_id]), 2)
            plot_statistics_dict[device_id]['error_mean'] = np.round(np.nanmean(prediction_error_dict[device_id]), 2)
            plot_statistics_dict[device_id]['error_sem'] = np.nanstd(prediction_error_dict[device_id]) / prediction_error_dict[device_id].shape[0]
            plot_statistics_dict[device_id]['low_ci'] = plot_statistics_dict[device_id]['error_mean'] - (2.58 * plot_statistics_dict[device_id]['error_sem'])
            plot_statistics_dict[device_id]['high_ci'] = plot_statistics_dict[device_id]['error_mean'] + (2.58 * plot_statistics_dict[device_id]['error_sem'])

        # calculate phidget-data statistics
        min_hum = np.round(np.nanmin(phidget_data_dictionary['humidity']), 2)
        max_hum = np.round(np.nanmax(phidget_data_dictionary['humidity']), 2)
        med_hum = np.round(np.nanmedian(phidget_data_dictionary['humidity']), 2)
        min_lux = np.round(np.nanmin(phidget_data_dictionary['lux']), 3)
        max_lux = np.round(np.nanmax(phidget_data_dictionary['lux']), 3)
        med_lux = np.round(np.nanmedian(phidget_data_dictionary['lux']), 3)
        min_temp = np.round(np.nanmin(phidget_data_dictionary['temperature']), 2)
        max_temp = np.round(np.nanmax(phidget_data_dictionary['temperature']), 2)
        med_temp = np.round(np.nanmedian(phidget_data_dictionary['temperature']), 2)

        # get audio information
        memmap_audio_file = glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}*.mmap")
        audio_sampling_rate = int(int(memmap_audio_file[0].split(os.sep)[-1].split('_')[3]) / 1000)
        audio_sample_number = memmap_audio_file[0].split(os.sep)[-1].split('_')[4]
        audio_ch_number = memmap_audio_file[0].split(os.sep)[-1].split('_')[5]

        # get relevant video metadata
        used_cameras = []
        total_frames = []
        motif_version = "Ø"
        camera_gain = 0
        camera_frame_rate = 0
        camera_exposure = 0
        experimenter = "Ø"
        animal_1 = "Ø"
        animal_2 = "Ø"
        cage_1 = "Ø"
        cage_2 = "Ø"
        dob_1 = "Ø"
        dob_2 = "Ø"
        sex_1 = "Ø"
        sex_2 = "Ø"
        gen_1 = "Ø"
        gen_2 = "Ø"
        hou_1 = "Ø"
        hou_2 = "Ø"

        counter = 0
        for sub_directory in os.listdir(f"{self.root_directory}{os.sep}video"):
            if any([cam in sub_directory for cam in ['21241563', '21369048', '21372315', '21372316', '22085397']]) and 'calibration' not in sub_directory:
                used_cameras.append(sub_directory.split('.')[-1])
                img_store = new_for_filename(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}{os.sep}metadata.yaml")
                total_frames.append(img_store.frame_count)

                if counter == 0:
                    user_meta_data = img_store.user_metadata
                    motif_version = f"v{user_meta_data['motif_version']}"
                    camera_gain = user_meta_data['gain']
                    camera_exposure = user_meta_data['exposuretime']
                    camera_frame_rate = user_meta_data['hwframerate']
                    for exp_key in user_meta_data.keys():
                        if 'experimenter' in exp_key and user_meta_data[exp_key] != '':
                            experimenter = user_meta_data[exp_key]
                        if 'mouse_ID_m1' in exp_key and user_meta_data[exp_key] != '':
                            animal_1 = user_meta_data[exp_key]
                        if 'mouse_ID_m2' in exp_key and user_meta_data[exp_key] != '':
                            animal_2 = user_meta_data[exp_key]
                        if 'cage_ID_m1' in exp_key and user_meta_data[exp_key] != '':
                            cage_1 = user_meta_data[exp_key]
                        if 'cage_ID_m2' in exp_key and user_meta_data[exp_key] != '':
                            cage_2 = user_meta_data[exp_key]
                        if 'DOB_m1' in exp_key and user_meta_data[exp_key] != '':
                            dob_1 = user_meta_data[exp_key]
                        if 'DOB_m2' in exp_key and user_meta_data[exp_key] != '':
                            dob_2 = user_meta_data[exp_key]
                        if 'sex_m1' in exp_key and user_meta_data[exp_key] != '':
                            sex_1 = user_meta_data[exp_key]
                        if 'sex_m2' in exp_key and user_meta_data[exp_key] != '':
                            sex_2 = user_meta_data[exp_key]
                        if 'genotype_m1' in exp_key and user_meta_data[exp_key] != '':
                            gen_1 = user_meta_data[exp_key]
                        if 'genotype_m2' in exp_key and user_meta_data[exp_key] != '':
                            gen_2 = user_meta_data[exp_key]
                        if 'housing_m1' in exp_key and user_meta_data[exp_key] != '':
                            hou_1 = user_meta_data[exp_key]
                        if 'housing_m2' in exp_key and user_meta_data[exp_key] != '':
                            hou_2 = user_meta_data[exp_key]
                counter += 1

        # optimize histogram
        for device_id in plot_statistics_dict.keys():
            plot_statistics_dict[device_id]['most_extreme_value'] = int(np.round(np.nanmax(np.abs(prediction_error_dict[device_id]))))
            if plot_statistics_dict[device_id]['most_extreme_value'] > 10.5:
                plot_statistics_dict[device_id]['plot_x_min'] = -plot_statistics_dict[device_id]['most_extreme_value'] - .5
                plot_statistics_dict[device_id]['plot_x_max'] = plot_statistics_dict[device_id]['most_extreme_value'] + .5
            else:
                plot_statistics_dict[device_id]['plot_x_min'] = -10.5
                plot_statistics_dict[device_id]['plot_x_max'] = 10.5

        fig, ax = plt.subplots(nrows=1, ncols=len(plot_statistics_dict.keys()))
        axr = ax.ravel()
        for device_num, device_id in enumerate(plot_statistics_dict.keys()):
            axr[device_num].hist(prediction_error_dict[device_id],
                                 bins=np.arange(-plot_statistics_dict[device_id]['most_extreme_value'] - .5,
                                                plot_statistics_dict[device_id]['most_extreme_value'] + 1, 1),
                                 color='#838B8B')

            axr[device_num].set_xlim(plot_statistics_dict[device_id]['plot_x_min'], plot_statistics_dict[device_id]['plot_x_max'])

            axr[device_num].axvline(x=0, ls='-.', lw=1.2, c='#000000')
            axr[device_num].axvline(x=plot_statistics_dict[device_id]['error_median'],
                                    ls='-', lw=1.4, c='#FF6347')
            axr[device_num].axvline(x=plot_statistics_dict[device_id]['error_mean'],
                                    ls='-', lw=1.4, c='#00C78C')

            axr[device_num].set_xlabel('undershoot <--- Prediction error (video frames) ---> overshoot')
            axr[device_num].set_ylabel('Total number of instances (#)')
            axr[device_num].set_title(device_id, pad=20)

            axr[device_num].text(x=0.775, y=0.18, s=r"median$_{error}$: " + f"{plot_statistics_dict[device_id]['error_median']} fr", verticalalignment='top',
                                 transform=axr[device_num].transAxes, fontsize=6, color='#FF6347')
            axr[device_num].text(x=0.775, y=0.15, s=r"mean$_{error}$: " + f"{plot_statistics_dict[device_id]['error_mean']} fr", verticalalignment='top',
                                 transform=axr[device_num].transAxes, fontsize=6, color='#00C78C')
            axr[device_num].text(x=0.775, y=0.12, s=f"99% CI [{plot_statistics_dict[device_id]['low_ci']:.3f}, {plot_statistics_dict[device_id]['high_ci']:.3f}]",
                                 verticalalignment='top', transform=axr[device_num].transAxes, fontsize=6)
            axr[device_num].text(x=0.775, y=0.09, s=r"N$_{test}$=" + f"{prediction_error_dict[device_id].shape[0]}",
                                 verticalalignment='top', transform=axr[device_num].transAxes, fontsize=6)

            if 'm_' in device_id:
                axr[device_num].text(x=0.005, y=1.04, s=r"$\bf{experiment|}$" + f" {self.root_directory.split(os.sep)[-1]}        " + r"$\bf{by|}$" +
                                                        f" {experimenter}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.985, s=r"$\bf{aID}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12)
                axr[device_num].text(x=0.115, y=0.9775, s=f"{animal_1}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.245, y=0.9775, s=f"{animal_2}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.945, s=r"$\bf{cID}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12)
                axr[device_num].text(x=0.115, y=0.9375, s=f"{cage_1}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.245, y=0.9375, s=f"{cage_2}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.905, s=r"$\bf{dob}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12)
                axr[device_num].text(x=0.115, y=0.8975, s=f"{dob_1}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.245, y=0.8975, s=f"{dob_2}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.865, s=r"$\bf{sex}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12)
                axr[device_num].text(x=0.115, y=0.8575, s=f"{sex_1}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.245, y=0.8575, s=f"{sex_2}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.825, s=r"$\bf{gen}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12)
                axr[device_num].text(x=0.115, y=0.8175, s=f"{gen_1}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.245, y=0.8175, s=f"{gen_2}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.785, s=r"$\bf{hou}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12)
                axr[device_num].text(x=0.115, y=0.7775, s=f"{hou_1}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.245, y=0.7775, s=f"{hou_2}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.715, s=r"$\bf{duration}$: " + f"{int(round(duration_min))} s", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.005, y=0.685, s=r"$\bf{audio}$: " + f"{audio_sample_number} ({audio_ch_number}ch, {audio_sampling_rate} kHz)",
                                     verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.005, y=0.655, s=r"$\bf{video parameters}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

                axr[device_num].text(x=0.005, y=0.625, s=r"$\bf{motif}$: " + f"{motif_version}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.005, y=0.595, s=r"$\bf{gain}$: " + f"{camera_gain} dB", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.005, y=0.565, s=r"$\bf{exposure}$: " + f"{camera_exposure} μs", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.005, y=0.535, s=r"$\bf{framerate}$: " + f"{camera_frame_rate} fps", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.005, y=0.505, s=r"$\bf{cameras}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                cam_y_txt = 0.475
                for cam_idx in range(len(used_cameras)):
                    axr[device_num].text(x=0.005, y=cam_y_txt, s=f"{used_cameras[cam_idx]}: {total_frames[cam_idx]} fr", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                    cam_y_txt -= .03

                axr[device_num].text(x=0.625, y=0.985, s=r"$\bf{humidity (\%)}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12, color='#8EE5EE')
                axr[device_num].text(x=0.625, y=0.945, s=r"$\bf{min}$   $\bf{med}$   $\bf{max}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=10)
                axr[device_num].text(x=0.625, y=0.91, s=f"{min_hum}    {med_hum}      {max_hum}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.625, y=0.855, s=r"$\bf{illumination (lux)}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12, color='#EEB422')
                axr[device_num].text(x=0.625, y=0.815, s=r"$\bf{min}$    $\bf{med}$   $\bf{max}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=10)
                axr[device_num].text(x=0.625, y=0.78, s=f"{min_lux}   {med_lux}     {max_lux}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)
                axr[device_num].text(x=0.625, y=0.725, s=r"$\bf{temperature (°C)}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=12, color='#FF7F50')
                axr[device_num].text(x=0.625, y=0.685, s=r"$\bf{min}$   $\bf{med}$   $\bf{max}$", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=10)
                axr[device_num].text(x=0.625, y=0.65, s=f"{min_temp}    {med_temp}      {max_temp}", verticalalignment='top', transform=axr[device_num].transAxes, fontsize=8)

            if 's_' in device_id:
                phidget_data_dictionary['humidity'] = phidget_data_dictionary['humidity'][~np.isnan(phidget_data_dictionary['humidity'])]
                axin1 = axr[device_num].inset_axes([0.05, 0.8, 0.15, 0.15], transform=axr[device_num].transAxes)
                axin1.hist(phidget_data_dictionary['humidity'], color='#8EE5EE')
                axin1.set_yticks([])
                try:
                    axin1.set_xticks([round(phidget_data_dictionary['humidity'].min(), 2),
                                      round(phidget_data_dictionary['humidity'].mean(), 2),
                                      round(phidget_data_dictionary['humidity'].max(), 2)])
                except ValueError:
                    pass
                axin1.set_xlabel('humidity (%)')

                axin2 = axr[1].inset_axes([0.225, 0.8, 0.15, 0.15], transform=axr[1].transAxes)
                axin2.plot(phidget_data_dictionary['humidity'], ls='-', lw=.3, color='#8EE5EE')
                axin2.set_xticks([])
                axin2.set_yticks([])
                axin2.set_xlabel('time (s)')

                phidget_data_dictionary['lux'] = phidget_data_dictionary['lux'][~np.isnan(phidget_data_dictionary['lux'])]
                axin3 = axr[device_num].inset_axes([0.05, 0.5, 0.15, 0.15], transform=axr[device_num].transAxes)
                axin3.hist(phidget_data_dictionary['lux'], color='#EEB422')
                axin3.set_yticks([])
                try:
                    axin3.set_xticks([round(phidget_data_dictionary['lux'].min(), 2),
                                      round(phidget_data_dictionary['lux'].mean(), 2),
                                      round(phidget_data_dictionary['lux'].max(), 2)])
                except ValueError:
                    pass
                axin3.set_xlabel('illumination (lux)')

                axin4 = axr[1].inset_axes([0.225, 0.5, 0.15, 0.15], transform=axr[1].transAxes)
                axin4.plot(phidget_data_dictionary['lux'], ls='-', lw=.3, color='#EEB422')
                axin4.set_xticks([])
                axin4.set_yticks([])
                axin4.set_xlabel('time (s)')

                phidget_data_dictionary['temperature'] = phidget_data_dictionary['temperature'][~np.isnan(phidget_data_dictionary['temperature'])]
                axin5 = axr[device_num].inset_axes([0.05, 0.2, 0.15, 0.15], transform=axr[device_num].transAxes)
                axin5.hist(phidget_data_dictionary['temperature'], color='#FF7F50')
                axin5.set_yticks([])
                try:
                    axin5.set_xticks([round(phidget_data_dictionary['temperature'].min(), 2),
                                      round(phidget_data_dictionary['temperature'].mean(), 2),
                                      round(phidget_data_dictionary['temperature'].max(), 2)])
                except ValueError:
                    pass
                axin5.set_xlabel('temperature (°C)')

                axin6 = axr[1].inset_axes([0.225, 0.2, 0.15, 0.15], transform=axr[1].transAxes)
                axin6.plot(phidget_data_dictionary['temperature'], ls='-', lw=.3, color='#FF7F50')
                axin6.set_xticks([])
                axin6.set_yticks([])
                axin6.set_xlabel('time (s)')

        if os.path.exists(f"{self.root_directory}{os.sep}sync"):
            fig.savefig(f"{self.root_directory}{os.sep}sync{os.sep}{self.root_directory.split(os.sep)[-1]}_summary.png",
                        dpi=300)
        else:
            print("The specified figure saving directory doesn't exist. Try again.")
