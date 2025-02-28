"""
@author: bartulem
Make tuning curve figures for 3D behavioral features.
"""

from PyQt6.QtTest import QTest
from datetime import datetime
import glob
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pathlib
import pickle
from tqdm import tqdm
import warnings
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from src.visualizations.auxiliary_plot_functions import create_colormap, choose_animal_colors
from src.analyses.decode_experiment_label import extract_information
from src.analyses.compute_behavioral_features import FeatureZoo

plt.style.use(pathlib.Path(__file__).parent.parent / '_config/usv_playpen.mplstyle')

class RatemapFigureMaker(FeatureZoo):

    def __init__(self, **kwargs):
        """
        Initializes the RatemapFigureMaker class.

        Parameter
        ---------
        root_directory : str
            Root directory for data; defaults to None.
        visualizations_parameter_dict : dict
            Dictionary of all visualization params; defaults to None.
        message_output : function
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        FeatureZoo.__init__(self)

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

    def neuronal_tuning_figures(self) -> None:
        """
        Description
        ----------
        This method plots tuning curves for 3D behavioral features.
        ----------

        Parameter
        ---------
        Uses the following set of parameters:
            smoothing_sd : int
                Standard deviation for Gaussian smoothing kernel (in numbers of bins).
            occ_threshold : int / float
                Threshold for minimum occupancy (in seconds).

        Returns
        -------
        neuronal_tuning_curves : .pdf
            Figure w/ tuning curves for individual behavioral features.
        """

        self.message_output(f"Making behavioral tuning curves started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        astropy_kernel_1d = Gaussian1DKernel(stddev=self.visualizations_parameter_dict['neuronal_tuning_figures']['smoothing_sd'])
        astropy_kernel_2d = Gaussian2DKernel(x_stddev=self.visualizations_parameter_dict['neuronal_tuning_figures']['smoothing_sd'], y_stddev=self.visualizations_parameter_dict['neuronal_tuning_figures']['smoothing_sd'])

        # load experimental code and get mouse colors
        tracked_file_loc = glob.glob(f"{self.root_directory}{os.sep}video{os.sep}**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5")[0]
        with h5py.File(tracked_file_loc, mode='r') as tracking_data_3d:
            mouse_id_list = [elem.decode('utf-8') for elem in list(tracking_data_3d['track_names'])]
            session_exp_code = tracking_data_3d['experimental_code'][()].decode('utf-8')
        experiment_info_dict = extract_information(experiment_code=session_exp_code)
        mouse_colors = choose_animal_colors(exp_info_dict=experiment_info_dict, visualizations_parameter_dict=self.visualizations_parameter_dict)

        # load tuning curve data
        cluster_tuning_curve_files = sorted(glob.glob(f"{self.root_directory}{os.sep}ephys{os.sep}tuning_curves{os.sep}*_tuning_curves_data.pkl"))

        for one_cluster_file in tqdm(cluster_tuning_curve_files):

            # load data for particular cluster
            with open(one_cluster_file, 'rb') as neuronal_tuning_curves_pkl:
                cluster_data = pickle.load(neuronal_tuning_curves_pkl)

            # find total animal number and establish color-code
            mouse_color_dict = {'social': '#000000'}
            mouse_colormap_dict = {}
            for mouse_idx, mouse in enumerate(mouse_id_list):
                mouse_color_dict[mouse] = mouse_colors[mouse_idx]
                mouse_colormap_dict[mouse] = create_colormap(input_parameter_dict={'cm_length': 255,
                                                                                   'cm_name': f'{mouse}',
                                                                                   'cm_type': 'sequential',
                                                                                   'cm_start': (int(mouse_colors[mouse_idx][1:3], 16),
                                                                                                int(mouse_colors[mouse_idx][3:5], 16),
                                                                                                int(mouse_colors[mouse_idx][5:7], 16)),
                                                                                   'cm_end': (255, 255, 255),
                                                                                   'equalize_luminance': True,
                                                                                   'match_luminance_by': 'max',
                                                                                   'change_saturation': .5,
                                                                                   'cm_opacity': 1})

            plot_features = {}
            for feature_key in cluster_data[next(iter(cluster_data))].keys():
                mouse_id = feature_key.split('.')[0]
                if f'individual.{mouse_id}' not in plot_features.keys() and '-' not in mouse_id:
                    plot_features[f'individual.{mouse_id}'] = []

                if '-' not in mouse_id:
                    plot_features[f'individual.{mouse_id}'].append(feature_key)
                else:
                    if 'social' not in plot_features.keys():
                        plot_features['social'] = []
                    plot_features['social'].append(feature_key)

            for offset in cluster_data.keys():
                with PdfPages(f"{one_cluster_file[:-8]}plots_{offset}.pdf") as pdf_fig:
                    for plot_feature_key in plot_features.keys():
                        if 'social' in plot_feature_key:
                            rm_color = mouse_color_dict['social']
                            mouse_colormap = 0
                        else:
                            rm_color = mouse_color_dict[plot_feature_key.split('.')[-1]]
                            mouse_colormap = mouse_colormap_dict[plot_feature_key.split('.')[-1]]

                        with warnings.catch_warnings():
                            warnings.simplefilter(action="ignore")

                            if len(plot_features[plot_feature_key]) > 0:
                                row_num = int(np.ceil((len(plot_features[plot_feature_key]) * 2) / 6))
                                fig = plt.figure(figsize=(6.4, float(row_num)), tight_layout=False)
                                gs = gridspec.GridSpec(nrows=row_num, ncols=6, wspace=0.5, hspace=0.5,
                                                       left=0.05, right=0.95, bottom=0.05, top=0.95)

                                gs_x = 0
                                gs_y = 0
                                for feature_idx, feature in enumerate(plot_features[plot_feature_key]):

                                    if 'space' in feature:
                                        cbar_width = .005
                                        cbar_height = .04

                                        ratemap = cluster_data[offset][feature]['ratemaps'][:, :, 0] / cluster_data[offset][feature]['ratemaps'][:, :, 1]
                                        ratemap = convolve(data=ratemap,
                                                           kernel=astropy_kernel_2d,
                                                           boundary='extend',
                                                           nan_treatment='interpolate',
                                                           preserve_nan=False)

                                        ax1 = fig.add_subplot(gs[gs_x, gs_y])
                                        rm = ax1.imshow(X=ratemap,
                                                        cmap='cividis',
                                                        vmin=0,
                                                        interpolation='gaussian',
                                                        aspect='equal')
                                        ax1.set_title(label="Spatial tuning",
                                                      fontsize=5,
                                                      pad=1.1)
                                        ax1.set_xticks([])
                                        ax1.set_xlabel('X (cm)',
                                                       fontsize=4,
                                                       labelpad=1)
                                        ax1.set_yticks([])
                                        ax1.set_ylabel('Y (cm)',
                                                       fontsize=4,
                                                       labelpad=1)

                                        ax_position_x1 = gs[gs_x, gs_y].get_position(fig).x1
                                        cb_ax = fig.add_axes((ax_position_x1, 0.91, cbar_width, cbar_height))
                                        cbar = fig.colorbar(mappable=rm,
                                                            orientation='vertical',
                                                            cax=cb_ax)
                                        cbar_vmin, cbar_vmax = cbar.mappable.get_clim()
                                        cbar.set_ticks([cbar_vmin, cbar_vmax])
                                        cbar.set_ticklabels(ticklabels=[f"{int(cbar_vmin)}", f"{round(cbar_vmax, 1)}"], fontsize=4)
                                        cbar.ax.tick_params(axis='both', which='both', length=0, pad=.5)
                                        cbar.outline.set_visible(True)

                                        ax2 = fig.add_subplot(gs[gs_x, gs_y + 1])
                                        occ = ax2.imshow(X=cluster_data[offset][feature]['ratemaps'][:, :, 1],
                                                         cmap=mouse_colormap,
                                                         vmin=0,
                                                         interpolation='gaussian',
                                                         aspect='equal')
                                        ax2.set_title(label="Spatial tuning (occ)",
                                                      fontsize=5,
                                                      pad=1.1)
                                        ax2.set_xticks([])
                                        ax2.set_xlabel('X (cm)',
                                                       fontsize=4,
                                                       labelpad=1)
                                        ax2.set_yticks([])
                                        ax2.set_ylabel('Y (cm)',
                                                       fontsize=4,
                                                       labelpad=1)

                                        ax2_position_x1 = gs[gs_x, gs_y + 1].get_position(fig).x1
                                        cb_ax2 = fig.add_axes((ax2_position_x1, 0.91, cbar_width, cbar_height))
                                        cbar2 = fig.colorbar(mappable=occ,
                                                             orientation='vertical',
                                                             cax=cb_ax2)
                                        cbar2_vmin, cbar2_vmax = cbar2.mappable.get_clim()
                                        cbar2.set_ticks([cbar2_vmin, cbar2_vmax])
                                        cbar2.set_ticklabels(ticklabels=[f"{int(cbar2_vmin)}", f"{int(np.ceil(cbar2_vmax))}"], fontsize=4)
                                        cbar2.ax.tick_params(axis='both', which='both', length=0, pad=.5)
                                        cbar2.outline.set_visible(True)

                                        gs_y += 2
                                        if gs_y > 5:
                                            gs_y = 0
                                            gs_x += 1
                                    else:
                                        occ_threshold = cluster_data[offset][feature]['ratemaps'][:, 1] > self.visualizations_parameter_dict['neuronal_tuning_figures']['occ_threshold']
                                        low_end_sh = convolve(data=np.percentile(cluster_data[offset][feature]['sh_counts'], q=.5, axis=0) / cluster_data[offset][feature]['ratemaps'][:, 1],
                                                              kernel=astropy_kernel_1d,
                                                              boundary='extend',
                                                              nan_treatment='interpolate',
                                                              preserve_nan=True)

                                        high_end_sh = convolve(data=np.percentile(cluster_data[offset][feature]['sh_counts'], q=99.5, axis=0) / cluster_data[offset][feature]['ratemaps'][:, 1],
                                                               kernel=astropy_kernel_1d,
                                                               boundary='extend',
                                                               nan_treatment='interpolate',
                                                               preserve_nan=True)

                                        ratemap = convolve(data=cluster_data[offset][feature]['ratemaps'][:, 0] / cluster_data[offset][feature]['ratemaps'][:, 1],
                                                           kernel=astropy_kernel_1d,
                                                           boundary='extend',
                                                           nan_treatment='interpolate',
                                                           preserve_nan=True)

                                        width = cluster_data[offset][feature]['bin_edges'][1] - cluster_data[offset][feature]['bin_edges'][0]

                                        ax1 = fig.add_subplot(gs[gs_x, gs_y])
                                        ax1.tick_params(axis='both', which='both', length=1.5, pad=.25)
                                        ax1.fill_between(cluster_data[offset][feature]['bin_centers'][occ_threshold],
                                                         low_end_sh[occ_threshold],
                                                         high_end_sh[occ_threshold],
                                                         where=high_end_sh[occ_threshold] >= low_end_sh[occ_threshold],
                                                         facecolor='#D3D3D3',
                                                         interpolate=True)
                                        ax1.plot(cluster_data[offset][feature]['bin_centers'][occ_threshold],
                                                 ratemap[occ_threshold],
                                                 lw=1,
                                                 ls='-',
                                                 c=rm_color,
                                                 alpha=1.)
                                        ax1.set_title(feature.split('.')[-1],
                                                      fontsize=5,
                                                      pad=1.1)
                                        temp_xmin, temp_xmax = ax1.get_xlim()
                                        ax1_xmin_temp = max(self.feature_boundaries[feature.split('.')[-1]][0], temp_xmin)
                                        ax1_xmax_temp = min(self.feature_boundaries[feature.split('.')[-1]][1], temp_xmax)
                                        ax1.set_xticks(ticks=[ax1_xmin_temp, ax1_xmax_temp],
                                                       labels=[f"{ax1_xmin_temp:.1f}", f"{ax1_xmax_temp:.1f}"],
                                                       rotation=0,
                                                       fontsize=4)
                                        ax1.set_xlabel(f"{self.feature_labels[plot_feature_key.split('.')[0]][feature.split('.')[-1]]}",
                                                       fontsize=4,
                                                       labelpad=1)
                                        temp_ymin, temp_ymax = ax1.get_ylim()
                                        ax1.set_yticks(ticks=[max(np.floor(temp_ymin), 0), np.ceil(temp_ymax)],
                                                       labels=[f"{max(np.floor(temp_ymin), 0):.1f}", f"{temp_ymax:.1f}"],
                                                       rotation=0,
                                                       fontsize=4)
                                        ax1.set_ylabel('Firing rate (sp/s)',
                                                       fontsize=4,
                                                       labelpad=1)
                                        ax1.set_box_aspect(1)

                                        ax2 = fig.add_subplot(gs[gs_x, gs_y + 1])
                                        ax2.tick_params(axis='both', which='both', length=1.5, pad=.25)
                                        ax2.bar(x=cluster_data[offset][feature]['bin_centers'],
                                                height=cluster_data[offset][feature]['ratemaps'][:, 1],
                                                width=width,
                                                align='center',
                                                color=rm_color,
                                                ec='#000000',
                                                lw=.1)
                                        ax2.set_title(label=f"{feature.split('.')[-1]} (occ)",
                                                      fontsize=5,
                                                      pad=1.1)
                                        ax2.set_xticks(ticks=[self.feature_boundaries[feature.split('.')[-1]][0],
                                                              self.feature_boundaries[feature.split('.')[-1]][1]],
                                                       labels=[f"{self.feature_boundaries[feature.split('.')[-1]][0]:.1f}",
                                                               f"{self.feature_boundaries[feature.split('.')[-1]][1]:.1f}"],
                                                       rotation=0,
                                                       fontsize=4)
                                        ax2.set_xlabel(f"{self.feature_labels[plot_feature_key.split('.')[0]][feature.split('.')[-1]]}",
                                                       fontsize=4,
                                                       labelpad=1)
                                        temp_ymin2, temp_ymax2 = ax2.get_ylim()
                                        ax2.set_yticks(ticks=[0, int(np.ceil(temp_ymax2)) - 10], labels=['0', f'{int(np.ceil(temp_ymax2)) - 10}'], rotation=0, fontsize=4)
                                        ax2.set_ylabel('Occupancy (s)',
                                                       fontsize=4,
                                                       labelpad=1)
                                        ax2.set_box_aspect(1)

                                        gs_y += 2
                                        if gs_y > 5:
                                            gs_y = 0
                                            gs_x += 1

                                pdf_fig.savefig(dpi=600)
                                plt.clf()
                                plt.close('all')
