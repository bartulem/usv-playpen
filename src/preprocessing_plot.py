"""
@author: bartulem
Generates a summary figure for data preprocessing.
"""

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import wave
from imgstore import new_for_filename

plt.style.use(pathlib.Path(__file__).parent / '_config/usv_playpen.mplstyle')


class SummaryPlotter:

    def __init__(self, input_parameter_dict: dict = None,
                 root_directory: str = None) -> None:
        """
        Initializes the SummaryPlotter class.

        Parameter
        ---------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
           Processing parameters; defaults to None.

        Returns
        -------
        -------
        """

        if root_directory is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.root_directory = json.load(json_file)['preprocessing_plot']['root_directory']
        else:
            self.root_directory = root_directory

        if input_parameter_dict is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['preprocessing_plot']['SummaryPlotter']
        else:
            self.input_parameter_dict = input_parameter_dict

    def preprocessing_summary(self,
                              ipi_discrepancy_dict: dict = None,
                              phidget_data_dictionary: dict = None) -> None:
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
        ipi_discrepancy_dict (dict)
           Contains arrays of A/V IPI discrepancies (in ms) and video ipi start frames.
        phidget_data_dictionary (dict)
            Contains lux, humidity and temperature data.
        ----------

        Returns
        ----------
        preprocessing_plot (fig)
            Figure summarizing the preprocessing of experimental data.
        ----------
        """

        # get the total number of frames in the video
        json_loc = sorted(glob.glob(f"{self.root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json"))[0]
        with open(json_loc, 'r') as camera_count_json_file:
            duration_min = json.load(camera_count_json_file)['total_video_time_least']

        # calculate error statistics
        plot_statistics_dict = {}
        for device_id in ipi_discrepancy_dict.keys():
            plot_statistics_dict[device_id] = {}
            plot_statistics_dict[device_id]['error_median'] = np.round(np.nanmedian(ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms']), decimals=2)
            plot_statistics_dict[device_id]['error_mean'] = np.round(np.nanmean(ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms']), decimals=2)
            plot_statistics_dict[device_id]['error_sem'] = np.nanstd(ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms']) / ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms'].size
            plot_statistics_dict[device_id]['low_ci'] = plot_statistics_dict[device_id]['error_mean'] - (2.58 * plot_statistics_dict[device_id]['error_sem'])
            plot_statistics_dict[device_id]['high_ci'] = plot_statistics_dict[device_id]['error_mean'] + (2.58 * plot_statistics_dict[device_id]['error_sem'])

        # calculate phidget-data statistics
        min_hum = np.round(np.nanmin(phidget_data_dictionary['humidity']), decimals=2)
        max_hum = np.round(np.nanmax(phidget_data_dictionary['humidity']), decimals=2)
        med_hum = np.round(np.nanmedian(phidget_data_dictionary['humidity']), decimals=2)
        min_lux = np.round(np.nanmin(phidget_data_dictionary['lux']), decimals=3)
        max_lux = np.round(np.nanmax(phidget_data_dictionary['lux']), decimals=3)
        med_lux = np.round(np.nanmedian(phidget_data_dictionary['lux']), decimals=3)
        min_temp = np.round(np.nanmin(phidget_data_dictionary['temperature']), decimals=2)
        max_temp = np.round(np.nanmax(phidget_data_dictionary['temperature']), decimals=2)
        med_temp = np.round(np.nanmedian(phidget_data_dictionary['temperature']), decimals=2)

        # get audio information
        wav_audio_files = sorted(glob.glob(f"{self.root_directory}{os.sep}audio{os.sep}cropped_to_video{os.sep}*.wav"))
        with wave.open(wav_audio_files[0], mode='rb') as example_audio_file:
            audio_sampling_rate = example_audio_file.getframerate()
            audio_sample_number = example_audio_file.getnframes()
        audio_ch_number = len(wav_audio_files)

        # get relevant video metadata
        used_cameras = []
        total_frames = []
        cam_esr = []
        cam_durations = []
        motif_version = "Ø"
        video_encoding = "Ø"
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
            if os.path.isdir(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}") and '.' in sub_directory and '_' in sub_directory and 'calibration' not in sub_directory:
                used_cameras.append(sub_directory.split('.')[-1])
                img_store = new_for_filename(f"{self.root_directory}{os.sep}video{os.sep}{sub_directory}{os.sep}metadata.yaml")
                total_frames.append(img_store.frame_count)
                frame_times = img_store.get_frame_metadata()['frame_time']
                video_duration = frame_times[-1] - frame_times[0]
                cam_durations.append(video_duration)
                cam_esr.append(round(number=img_store.frame_count / video_duration, ndigits=3))

                if counter == 0:
                    video_encoding = img_store._format
                    user_meta_data = img_store.user_metadata
                    motif_version = f"v{user_meta_data['motif_version']}"
                    camera_gain = user_meta_data['gain']
                    camera_exposure = user_meta_data['exposuretime']
                    camera_frame_rate = user_meta_data['hwframerate']
                    for exp_key in user_meta_data.keys():
                        if 'experimenter' == exp_key and user_meta_data[exp_key] != '':
                            experimenter = user_meta_data[exp_key]
                        if 'subject' == exp_key and user_meta_data[exp_key] != '':
                            subject_entry = user_meta_data[exp_key].split(',')
                            animal_1 = subject_entry[0]
                            if len(subject_entry) > 1:
                                animal_2 = subject_entry[1]
                        if 'cage' == exp_key and user_meta_data[exp_key] != '':
                            cage_entry = user_meta_data[exp_key].split(',')
                            cage_1 = cage_entry[0]
                            if len(cage_entry) > 1:
                                cage_2 = cage_entry[1]
                        if 'dob' == exp_key and user_meta_data[exp_key] != '':
                            dob_entry = user_meta_data[exp_key].split(',')
                            dob_1 = dob_entry[0]
                            if len(dob_entry) > 1:
                                dob_2 = dob_entry[1]
                        if 'sex' == exp_key and user_meta_data[exp_key] != '':
                            sex_entry = user_meta_data[exp_key].split(',')
                            sex_1 = sex_entry[0]
                            if len(sex_entry) > 1:
                                sex_2 = sex_entry[1]
                        if 'strain' == exp_key and user_meta_data[exp_key] != '':
                            gen_entry = user_meta_data[exp_key].split(',')
                            gen_1 = gen_entry[0]
                            if len(gen_entry) > 1:
                                gen_2 = gen_entry[1]
                        if 'housing' == exp_key and user_meta_data[exp_key] != '':
                            hou_entry = user_meta_data[exp_key].split(',')
                            hou_1 = hou_entry[0]
                            if len(hou_entry) > 1:
                                hou_2 = hou_entry[1]
                counter += 1

        # optimize histogram
        for device_id in plot_statistics_dict.keys():
            plot_statistics_dict[device_id]['most_extreme_value'] = int(np.round(np.nanmax(np.abs(ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms']))))
            if plot_statistics_dict[device_id]['most_extreme_value'] >= 10:
                plot_statistics_dict[device_id]['plot_x_min'] = -plot_statistics_dict[device_id]['most_extreme_value'] - 1.5
                plot_statistics_dict[device_id]['plot_x_max'] = plot_statistics_dict[device_id]['most_extreme_value'] + 1.5
            else:
                plot_statistics_dict[device_id]['plot_x_min'] = -10.5
                plot_statistics_dict[device_id]['plot_x_max'] = 10.5

        fig, ax = plt.subplots(nrows=2, ncols=len(plot_statistics_dict.keys()), figsize=(12.8, 9.6))
        for device_num, device_id in enumerate(plot_statistics_dict.keys()):
            ax[0, device_num].hist(ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms'],
                                 bins=np.arange(-plot_statistics_dict[device_id]['most_extreme_value'] - .5,
                                                plot_statistics_dict[device_id]['most_extreme_value'] + 1, 1),
                                 histtype='stepfilled',
                                 color='#BBD5E826',
                                 edgecolor='#00000040')

            ax[0, device_num].set_xlim(plot_statistics_dict[device_id]['plot_x_min'], plot_statistics_dict[device_id]['plot_x_max'])

            ax[0, device_num].axvline(x=0, ls='-.', lw=1.2, c='#000000')
            ax[0, device_num].axvline(x=plot_statistics_dict[device_id]['error_median'],
                                    ls='-', lw=1.4, c='#FF6347')
            ax[0, device_num].axvline(x=plot_statistics_dict[device_id]['error_mean'],
                                    ls='-', lw=1.4, c='#00C78C')

            ax[0, device_num].set_xlabel('undershoot <--- A-V IPI start discrepancy (ms) ---> overshoot')
            ax[0, device_num].set_ylabel('Inter-pulse interval count (#)')
            ax[0, device_num].set_title(device_id, fontsize=12, pad=25)

            ax[0, device_num].text(x=0.775, y=0.18, s=r"median$_{error}$: " + f"{plot_statistics_dict[device_id]['error_median']} ms", verticalalignment='top',
                                 transform=ax[0, device_num].transAxes, fontsize=6, color='#FF6347')
            ax[0, device_num].text(x=0.775, y=0.15, s=r"mean$_{error}$: " + f"{plot_statistics_dict[device_id]['error_mean']} ms", verticalalignment='top',
                                 transform=ax[0, device_num].transAxes, fontsize=6, color='#00C78C')
            ax[0, device_num].text(x=0.775, y=0.12, s=f"99% CI [{plot_statistics_dict[device_id]['low_ci']:.3f}, {plot_statistics_dict[device_id]['high_ci']:.3f}]",
                                 verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=6)
            ax[0, device_num].text(x=0.775, y=0.09, s=r"N$_{IPI}$=" + f"{ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms'].size}",
                                 verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=6)

            ax[1, device_num].scatter(ipi_discrepancy_dict[device_id]['video_ipi_start_frames'], ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms'], color='#BBD5E826', edgecolor='#00000040', s=12)
            ax[1, device_num].set_xlabel('Camera frame number')
            ax[1, device_num].set_ylabel('undershoot <--- A-V IPI start discrepancy (ms) ---> overshoot')
            if device_num == 0 and 'nidq_ipi_discrepancy_ms' in ipi_discrepancy_dict[device_id].keys():
                axin_nidq_ = ax[1, device_num].inset_axes([0.6, 0.7, 0.39, 0.29], transform=ax[1, device_num].transAxes)
                axin_nidq_.scatter(ipi_discrepancy_dict[device_id]['nidq_ipi_start_samples'], ipi_discrepancy_dict[device_id]['nidq_ipi_discrepancy_ms'], color='#BBD5E826', edgecolor='#00000040', s=4)
                axin_nidq_.tick_params(axis='both', labelsize=3)
                axin_nidq_.set_xlabel('NIDQ sample number', fontsize=5)
                axin_nidq_.set_ylabel('NIDQ-V (ms)', fontsize=5)

            if 'm_' in device_id:
                ax[0, device_num].text(x=0.005, y=1.04, s=r"$\bf{experiment|}$" + f" {self.root_directory.split(os.sep)[-1]}        " + r"$\bf{by|}$" +
                                                        f" {experimenter}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.985, s=r"$\bf{aID}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12)
                ax[0, device_num].text(x=0.115, y=0.9775, s=f"{animal_1}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.245, y=0.9775, s=f"{animal_2}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.945, s=r"$\bf{cID}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12)
                ax[0, device_num].text(x=0.115, y=0.9375, s=f"{cage_1}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.245, y=0.9375, s=f"{cage_2}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.905, s=r"$\bf{dob}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12)
                ax[0, device_num].text(x=0.115, y=0.8975, s=f"{dob_1}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.245, y=0.8975, s=f"{dob_2}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.865, s=r"$\bf{sex}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12)
                ax[0, device_num].text(x=0.115, y=0.8575, s=f"{sex_1}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.245, y=0.8575, s=f"{sex_2}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.825, s=r"$\bf{gen}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12)
                ax[0, device_num].text(x=0.115, y=0.8175, s=f"{gen_1}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.245, y=0.8175, s=f"{gen_2}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.785, s=r"$\bf{hou}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12)
                ax[0, device_num].text(x=0.115, y=0.7775, s=f"{hou_1}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.245, y=0.7775, s=f"{hou_2}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.715, s=r"$\bf{duration}$: " + f"{int(round(duration_min))} s", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.005, y=0.685, s=r"$\bf{audio}$: " + f"{audio_sample_number} ({audio_ch_number}ch, {audio_sampling_rate} kHz)",
                                     verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

                ax[0, device_num].text(x=0.005, y=0.655, s=r"$\bf{motif}$: " + f"{motif_version}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.005, y=0.625, s=r"$\bf{codec}$: " + f"{video_encoding}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.005, y=0.595, s=r"$\bf{user set gain}$: " + f"{camera_gain} dB", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.005, y=0.565, s=r"$\bf{user set exposure}$: " + f"{camera_exposure} μs", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.005, y=0.535, s=r"$\bf{user set framerate}$: " + f"{camera_frame_rate} fps", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.005, y=0.505, s=r"$\bf{cameras}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                cam_y_txt = 0.475
                for cam_idx in range(len(used_cameras)):
                    ax[0, device_num].text(x=0.005, y=cam_y_txt, s=f"{used_cameras[cam_idx]}: {total_frames[cam_idx]} fr, {cam_esr[cam_idx]} fps", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                    cam_y_txt -= .03

                ax[0, device_num].text(x=0.625, y=0.985, s=r"$\bf{humidity (\%)}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12, color='#8EE5EE')
                ax[0, device_num].text(x=0.625, y=0.945, s=r"$\bf{min}$   $\bf{med}$   $\bf{max}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=10)
                ax[0, device_num].text(x=0.625, y=0.91, s=f"{min_hum}    {med_hum}      {max_hum}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.625, y=0.855, s=r"$\bf{illumination (lux)}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12, color='#EEB422')
                ax[0, device_num].text(x=0.625, y=0.815, s=r"$\bf{min}$    $\bf{med}$   $\bf{max}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=10)
                ax[0, device_num].text(x=0.625, y=0.78, s=f"{min_lux}   {med_lux}     {max_lux}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)
                ax[0, device_num].text(x=0.625, y=0.725, s=r"$\bf{temperature (°C)}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=12, color='#FF7F50')
                ax[0, device_num].text(x=0.625, y=0.685, s=r"$\bf{min}$   $\bf{med}$   $\bf{max}$", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=10)
                ax[0, device_num].text(x=0.625, y=0.65, s=f"{min_temp}    {med_temp}      {max_temp}", verticalalignment='top', transform=ax[0, device_num].transAxes, fontsize=8)

            if 's_' in device_id:
                phidget_data_dictionary['humidity'] = phidget_data_dictionary['humidity'][~np.isnan(phidget_data_dictionary['humidity'])]
                axin1 = ax[0, device_num].inset_axes([0.05, 0.8, 0.15, 0.15], transform=ax[0, device_num].transAxes)
                axin1.hist(phidget_data_dictionary['humidity'], color='#8EE5EE')
                axin1.set_yticks([])
                try:
                    axin1.set_xticks([round(phidget_data_dictionary['humidity'].min(), 2),
                                      round(phidget_data_dictionary['humidity'].mean(), 2),
                                      round(phidget_data_dictionary['humidity'].max(), 2)])
                except ValueError:
                    pass
                axin1.set_xlabel('humidity (%)')

                axin2 = ax[0, device_num].inset_axes([0.225, 0.8, 0.15, 0.15], transform=ax[0, device_num].transAxes)
                axin2.plot(phidget_data_dictionary['humidity'], ls='-', lw=.3, color='#8EE5EE')
                axin2.set_xticks([])
                axin2.set_yticks([])
                axin2.set_xlabel('time (s)')

                phidget_data_dictionary['lux'] = phidget_data_dictionary['lux'][~np.isnan(phidget_data_dictionary['lux'])]
                axin3 = ax[0, device_num].inset_axes([0.05, 0.5, 0.15, 0.15], transform=ax[0, device_num].transAxes)
                axin3.hist(phidget_data_dictionary['lux'], color='#EEB422')
                axin3.set_yticks([])
                try:
                    axin3.set_xticks([round(phidget_data_dictionary['lux'].min(), 2),
                                      round(phidget_data_dictionary['lux'].mean(), 2),
                                      round(phidget_data_dictionary['lux'].max(), 2)])
                except ValueError:
                    pass
                axin3.set_xlabel('illumination (lux)')

                axin4 = ax[0, device_num].inset_axes([0.225, 0.5, 0.15, 0.15], transform=ax[0, device_num].transAxes)
                axin4.plot(phidget_data_dictionary['lux'], ls='-', lw=.3, color='#EEB422')
                axin4.set_xticks([])
                axin4.set_yticks([])
                axin4.set_xlabel('time (s)')

                phidget_data_dictionary['temperature'] = phidget_data_dictionary['temperature'][~np.isnan(phidget_data_dictionary['temperature'])]
                axin5 = ax[0, device_num].inset_axes([0.05, 0.2, 0.15, 0.15], transform=ax[0, device_num].transAxes)
                axin5.hist(phidget_data_dictionary['temperature'], color='#FF7F50')
                axin5.set_yticks([])
                try:
                    axin5.set_xticks([round(phidget_data_dictionary['temperature'].min(), 2),
                                      round(phidget_data_dictionary['temperature'].mean(), 2),
                                      round(phidget_data_dictionary['temperature'].max(), 2)])
                except ValueError:
                    pass
                axin5.set_xlabel('temperature (°C)')

                axin6 = ax[0, device_num].inset_axes([0.225, 0.2, 0.15, 0.15], transform=ax[0, device_num].transAxes)
                axin6.plot(phidget_data_dictionary['temperature'], ls='-', lw=.3, color='#FF7F50')
                axin6.set_xticks([])
                axin6.set_yticks([])
                axin6.set_xlabel('time (s)')

        fig.savefig(fname=f"{self.root_directory}{os.sep}sync{os.sep}{self.root_directory.split(os.sep)[-1]}_summary.svg",
                    dpi=300)
        plt.close()
