"""
@author: bartulem
Computes behavioral features for files containing 3D tracked mouse body points.

[A] INDIVIDUAL FEATURES
(0) Head position (X,Y,Z) (1) Speed (2) Acceleration (3) Neck elevation (4) Neck elevation der (5) Neck elevation 2der
(6) Head roll (7) Head roll der (8) Head roll 2der (9) Head pitch (10) Head pitch der (11) Head pitch 2der
(12) Head yaw (13) Head yaw der (14) Head yaw 2der (15) Ego head yaw (16) Ego head yaw der (17) Ego head yaw 2der
(18) Back pitch (19) Back pitch der (20) Back pitch 2der (21) Back yaw (22) Back yaw der (23) Back Yaw 2der
(24) Body yaw (25) Body yaw der (26) Body yaw 2der (27) Tail curvature (28) Tail curvature der (29) Tail curvature 2der


[B] SOCIAL FEATURES (DISTANCES & ANGLES)
(0) Nose distance (1) Nose distance der (2) Nose distance 2der (3) TTI distance (4) TTI distance der (5) TTI distance 2der
(6) Nose-TTI distance  (7) Nose-TTI distance der  (8) Nose-TTI distance 2der  (9) TTI-Nose distance  (10) TTI-Nose distance der  (11) TTI-Nose distance 2der
(12) Neck elevation distance (13) Neck elevation distance der (14) Neck elevation distance 2der (15) Speed difference (16) Speed difference der (17) Speed difference 2der
(18) Yaw-Nose (19) Yaw-Nose der (20) Yaw-Nose 2der (21) Nose-Yaw (22) Nose-Yaw der (23) Nose-Yaw 2der
(24) Yaw-TTI (25) Yaw-TTI der (26) Yaw-TTI 2der (27) TTI-Yaw (28) TTI-Yaw angle der (29) TTI-Yaw 2der
"""

from PyQt6.QtTest import QTest
import glob
import h5py
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pathlib
import polars as pls
import warnings
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel
from typing import Tuple
from src.visualizations.auxiliary_plot_functions import create_colormap, choose_animal_colors
from .decode_experiment_label import extract_information

plt.style.use(pathlib.Path(__file__).parent.parent / '_config/usv_playpen.mplstyle')


def generate_feature_distributions(feature_arr: np.ndarray = None,
                                   min_val: int|float = None,
                                   max_val: int|float = None,
                                   num_bins: int|float = None,
                                   camera_fr: int|float = None,
                                   space_bool: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    ----------
    This function computes occupancy
    histograms for a given feature array.
    ----------

    Parameters
    ----------
    feature_arr (np.ndarray)
        A (n_frames) shape ndarray containing behavioral feature data.
    min_val (int / float)
        Minimum acceptable value feature could attain.
    max_val (int / float)
        Maximum acceptable value feature could attain.
    num_bins (int / float)
        Number of bins to divide features in.
    camera_fr (int / float)
        Camera frame rate.
    space_bool (bool)
        Boolean indicating if feature is spatial.
    ----------

    Returns
    ----------
    occ_array: np.ndarray
        A (num_bins) or (num_bins, num_bins) shape ndarray
        of occupancy (in seconds) for each feature.
    bin_centers: np.ndarray
        A (num_bins) shape ndarray of bin centers for given feature.
    bin_edges: np.ndarray
        A (num_bins) shape ndarray of bin edges for given feature.
    ----------
    """

    if space_bool:
        bins_in_one_dir = int(np.ceil(np.sqrt(num_bins)))
        bin_edges = np.linspace(min_val, max_val, bins_in_one_dir + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        occ_array = np.zeros((bins_in_one_dir, bins_in_one_dir))
        for i in range(1, np.shape(bin_edges)[0], 1):
            for j in range(1, np.shape(bin_edges)[0], 1):
                occ_array[i - 1, j - 1] = np.sum(((feature_arr[:, 0] > bin_edges[i - 1]) * (feature_arr[:, 0] <= bin_edges[i]))
                                                 * ((feature_arr[:, 1] > bin_edges[j - 1]) * (feature_arr[:, 1] <= bin_edges[j]))) / camera_fr
    else:
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        occ_array = np.zeros(num_bins)
        for i in range(1, np.shape(bin_edges)[0], 1):
            occ_array[i - 1] = np.sum((feature_arr > bin_edges[i - 1]) * (feature_arr <= bin_edges[i])) / camera_fr

    return occ_array, bin_centers, bin_edges


def calculate_derivatives(input_arr: np.ndarray = None,
                          diff_bins: int = None,
                          is_angle: bool = False,
                          capture_fr: int|float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays w/ first and second derivatives.
    NB: Computed according to the central difference derivative!

    Parameter
    ---------
    input_arr : np.ndarray
         A (n_frames, n_features) shape ndarray containing feature data to compute derivatives on.
    diff_bins : int
        Number of bins for the central difference derivative; defaults to None.
    is_angle : bool
        Is the feature data in angles or not; defaults to False.
    capture_fr : int / float
        Capture frame rate of the cameras; defaults to None (fps).

    Returns
    -------
    first_der, second_der : tuple (np.ndarray, np.ndarray)
        A tuple of 2 (n_frames, n_features) shape np.ndarray
        w/ first and second derivatives for selected features.
    """

    nan_positions = np.isnan(input_arr)

    first_der = np.zeros(shape=input_arr.shape)
    first_der[:] = np.nan

    second_der = np.zeros(shape=input_arr.shape)
    second_der[:] = np.nan

    # calculate first derivative
    first_der[diff_bins:-diff_bins, :] = input_arr[(diff_bins * 2):, :] - input_arr[:-(diff_bins * 2), :]
    if is_angle:
        first_der[first_der > 180] -= 360
        first_der[first_der < -180] += 360
    first_der = first_der / (2. * diff_bins / capture_fr)
    first_der[nan_positions] = np.nan

    # calculate second derivative
    second_der[diff_bins:-diff_bins, :] = first_der[(diff_bins * 2):, :] - first_der[:-(diff_bins * 2), :]
    second_der = second_der / (2. * diff_bins / capture_fr)
    second_der[nan_positions] = np.nan

    return first_der, second_der


def calculate_tail_curvature(input_arr: np.ndarray = None) -> np.ndarray:
    """
    Returns arrays w/ the tail curvature metric.

    NB: Calculate the tangent vectors: For each point,
    calculate the tangent vector by subtracting the previous
    point from the next point. Normalize the resulting vector
    to obtain a unit tangent vector. The tangent vector gives
    the direction of the tail at each point.

    Estimate the second derivative: To estimate the second derivative,
    consider the points between the tangent vectors. For each point,
    take the difference between the two neighboring tangent vectors and
    divide it by the distance between those points. This gives an
    approximation of the second derivative, which represents the curvature.

    Finally, calculate the average curvature by taking the mean over
    all the individual curvatures.

    Parameter
    ---------
    input_arr : np.ndarray
         A (n_nodes, n_frames, 3) shape ndarray to compute tail curvature on.

    Returns
    -------
    average_tail_curvature : np.ndarray
         A (n_frames, 1) shape ndarray containing the average tail curvature.
    """

    # calculate tangent vectors
    tangent_vectors = np.diff(input_arr, axis=1)
    tangent_vectors = tangent_vectors / np.linalg.norm(tangent_vectors, axis=2)[..., np.newaxis]

    # estimate the second derivative (curvature)
    curvature = np.diff(tangent_vectors, axis=1) / np.linalg.norm(tangent_vectors[:, :-1], axis=2)[..., np.newaxis]

    # calculate the average curvature for each time point
    avg_curvature = np.mean(np.linalg.norm(curvature, axis=2), axis=1)

    return np.reshape(avg_curvature, newshape=(avg_curvature.shape[0], 1))


def calculate_planar_social_angle(point1_arr: np.ndarray = None,
                                  point2_arr: np.ndarray = None,
                                  point3_arr: np.ndarray = None) -> np.ndarray:
    """
    Return arrays w/ planar social angle
    (e.g., points 1 and 2 can be the "head" and "nose"
    tracks of mouse 1, and point 3 can be the "head"
    of mouse 2, so where is the head of the second mouse
    relative to the viewing direction of the first mouse)

    Parameter
    ---------
    point1_arr : np.ndarray
         A (n_frames, 2) shape ndarray of first point in XY dimensions.
    point2_arr : np.ndarray
         A (n_frames, 2) shape ndarray of second point in XY dimensions.
    point3_arr : np.ndarray
         A (n_frames, 2) shape ndarray of third point in XY dimensions.

    Returns
    -------
    angles_arr : np.ndarray
         A (n_frames) shape ndarray of planar social angle of interest.
    """

    # shape points into vectors
    vector1 = np.stack((point1_arr, point2_arr), axis=1)
    vector2 = np.stack((point1_arr, point3_arr), axis=1)

    # calculate the distance between the two points
    diff_vector1 = vector1[:, 1, :] - vector1[:, 0, :]
    diff_vector2 = vector2[:, 1, :] - vector2[:, 0, :]

    # calculate the angle between the vectors in radians
    angles_radians = np.arctan2(diff_vector2[:, 1], diff_vector2[:, 0]) - \
                     np.arctan2(diff_vector1[:, 1], diff_vector1[:, 0])

    # convert to degrees
    angles_arr = angles_radians * 180. / np.pi
    angles_arr[angles_arr < -180] += 360
    angles_arr[angles_arr > 180] -= 360

    return angles_arr


def calculate_speed(tracked_points_array: np.ndarray = None,
                    capture_framerate: int|float = None,
                    smoothing_time_window: int|float = None) -> np.ndarray:
    """
    Returns arrays w/ centroid (body minus tail) speed data.

    Parameter
    ---------
    tracked_points_array : np.ndarray
         A (n_frames, n_nodes, n_dimensions)
         shape ndarray of tracked points.
    capture_framerate : int / float
        Recording camera framerate; defaults to None (fps).
    smoothing_time_window : int / float
        Time window to perform smoothing over; defaults to None (s).

    Returns
    -------
    speeds : np.ndarray
        A (n_frames) shape ndarray of centroid speed data.
    """

    speed_smoothing_kernel = Gaussian1DKernel(stddev=int(np.floor(smoothing_time_window * capture_framerate)))

    speed = np.zeros((tracked_points_array.shape[0], 1))

    mouse_centroid = np.nanmean(tracked_points_array, axis=1)
    frame_differential_centroid = mouse_centroid[1:, :] - mouse_centroid[:-1, :]
    euclidean_distance_centroid = 100 * np.linalg.norm(frame_differential_centroid, axis=1) / (1 / capture_framerate)
    speed_centroid = convolve(data=euclidean_distance_centroid,
                              kernel=speed_smoothing_kernel,
                              boundary='extend',
                              nan_treatment='interpolate',
                              preserve_nan=True)
    speed_centroid = np.concatenate((np.array([np.nan]), speed_centroid), dtype=np.float64)
    speed[:, 0] = speed_centroid

    return speed


def get_average_point(data_arr: np.ndarray) -> np.ndarray:
    """
    Finds the closest point to head average.

    Parameter
    ---------
    data_arr : np.ndarray
         A (4, n_frames, 3) shape ndarray of head point data.

    Returns
    -------
    closest_point_to_average : np.ndarray
        A (4, 3) shape ndarray of closest points to average.
    """

    point_combinations = list(itertools.combinations(np.arange(data_arr.shape[1]), r=2))
    point_combinations = sorted(point_combinations, key=lambda x: x[1])
    point_deviations = np.zeros(shape=(data_arr.shape[0], len(point_combinations)), dtype=np.float64)
    for i, one_combination in enumerate(point_combinations):
        point_deviations[:, i] = ((data_arr[:, one_combination[0], :] - data_arr[:, one_combination[1], :]) ** 2).sum(axis=1)

    average_point_deviations = np.nanmean(point_deviations, axis=0)

    distances_from_average = np.sqrt(((point_deviations - average_point_deviations) ** 2).sum(axis=1))

    min_distance = 100000000.
    points_temp = np.zeros(shape=(data_arr.shape[1], data_arr.shape[2]), dtype=np.float64)
    points_temp[:] = np.nan
    closest_point_to_average = None
    for dist_idx, distance in enumerate(distances_from_average):
        if distance < min_distance:
            for ii in range(data_arr.shape[1]):
                points_temp[ii, :] = data_arr[dist_idx, ii, :]
            if ~np.isnan(points_temp).any():
                min_distance = distance
                closest_point_to_average = points_temp

    return closest_point_to_average


def get_head_root(data_arr: np.ndarray,
                  closest_point_to_average: np.ndarray,
                  rotation_type: str = None) -> np.ndarray:
    """
    Computes the head-root rotation matrices.

    Parameter
    ---------
    data_arr : np.ndarray
         A (4, n_frames, 3) shape ndarray of head point data.
    closest_point_to_average : np.ndarray
         A (4, 3) shape ndarray of the closest points to average.
    rotation_type : str
        Type of rotation to perform; defaults to None.

    Returns
    -------
    global_heads_rot : np.ndarray
        A (n_frames, 3, 3) shape ndarray of head rotation matrices.
    """

    num_of_points = data_arr.shape[1]
    fractions_arr = np.zeros(shape=(data_arr.shape[0], 1, num_of_points), dtype=np.float64)
    fractions_arr[:] = 1 / num_of_points

    mu_average = fractions_arr @ closest_point_to_average
    closest_average_diff = closest_point_to_average - mu_average

    u_arr_ca, sv_vector_ca, vh_arr_ca = np.linalg.svd(closest_average_diff)

    axis_unit = np.zeros(shape=(data_arr.shape[0], num_of_points, 3), dtype=np.float64)
    axis_unit[:, 0, :] = mu_average[:, 0, :]
    axis_unit[:, 1:, :] = mu_average + vh_arr_ca[:]
    axis_unit_transpose = np.swapaxes(axis_unit, axis1=1, axis2=2)

    mu_data = fractions_arr @ data_arr
    data_diff = data_arr - mu_data
    closest_average_diff = np.swapaxes(closest_average_diff, axis1=1, axis2=2)
    svd_input_matrix = np.einsum('ijk,ikl->ijl', closest_average_diff, data_diff)

    u_arr_data, sv_vector_data, vh_arr_data = np.linalg.svd(svd_input_matrix)
    u_arr_data_transpose = np.swapaxes(u_arr_data, axis1=1, axis2=2)
    vh_arr_data_transpose = np.swapaxes(vh_arr_data, axis1=1, axis2=2)

    det_multi = np.linalg.det(u_arr_data_transpose) * np.linalg.det(vh_arr_data_transpose)
    relevant_indices = np.where(det_multi < 0)[0]
    vh_arr_data_transpose[relevant_indices, :, 2] = -vh_arr_data_transpose[relevant_indices, :, 2]

    rotation_matrix = np.einsum('ijk,ikl->ijl', vh_arr_data_transpose, u_arr_data_transpose)

    mu_average_transpose = np.swapaxes(mu_average, axis1=1, axis2=2)
    mu_data_transpose = np.swapaxes(mu_data, axis1=1, axis2=2)
    temp = np.einsum('ijk,ikl->ijl', rotation_matrix, mu_average_transpose)
    temp_diff = mu_data_transpose - temp
    rotation_root_ax = np.einsum('ijk,ikl->ijl', rotation_matrix, axis_unit_transpose)

    orig = rotation_root_ax[:, :, 0] + temp_diff[:, :].squeeze()
    head_x = rotation_root_ax[:, :, 1] + temp_diff[:, :].squeeze() - orig
    head_y = rotation_root_ax[:, :, 2] + temp_diff[:, :].squeeze() - orig
    head_z = rotation_root_ax[:, :, 3] + temp_diff[:, :].squeeze() - orig

    if rotation_type == 'regular':
        h_x = head_x / np.linalg.norm(head_x, axis=1)[:, np.newaxis]
        h_y = head_y / np.linalg.norm(head_y, axis=1)[:, np.newaxis]
        h_z = np.cross(h_x, h_y)
    else:
        if rotation_type == 'roll_issue':
            h_x = head_x / np.linalg.norm(head_x, axis=1)[:, np.newaxis]
            h_z = head_z / np.linalg.norm(head_z, axis=1)[:, np.newaxis]
            h_y = np.cross(h_z, h_x)
        else:
            h_y = head_y / np.linalg.norm(head_y, axis=1)[:, np.newaxis]
            h_z = head_z / np.linalg.norm(head_z, axis=1)[:, np.newaxis]
            h_x = np.cross(h_z, h_y)

    global_heads_rot = np.array([h_x, h_y, h_z])
    global_heads_rot = np.swapaxes(global_heads_rot, axis1=0, axis2=1)

    return global_heads_rot


def get_back_root(point_data_3d: np.ndarray,
                  mouse_id: int = None,
                  neck_point_pos: int = None,
                  tti_point_pos: int = None,
                  root_method: str = None,
                  spatial_resolution_tolerance: int|float = .001) -> np.ndarray:
    """
    Computes the back-root rotation matrices.

    Parameter
    ---------
    point_data_3d : np.ndarray
         A (n_frames, n_mice, n_nodes, 3) shape ndarray of tracked points.
    mouse_id : int
         Index of mouse.
    neck_point_pos : int
         Index of neck point in array.
    tti_point_pos : int
         Index of TTI point in array.
    root_method :str
         Rotation method to use.
    spatial_resolution_tolerance : int / float
         The least necessary amount of
         distance between points (in meters).

    Returns
    -------
    back_roots : np.ndarray
        A (n_frames, 3, 3) shape ndarray of rotation matrices.
    """

    back_roots = np.zeros((point_data_3d.shape[0], 3, 3), dtype=np.float64)
    back_roots[:] = np.nan

    if root_method == 'default':
        x_dir = point_data_3d[:, mouse_id, neck_point_pos, :] - point_data_3d[:, mouse_id, tti_point_pos, :]
        x_dir[:, 2] = 0.

        root_len = np.linalg.norm(x=x_dir, axis=1).astype(np.float64)
        root_len[root_len < spatial_resolution_tolerance] = np.nan
        x_dir = (x_dir / root_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(point_data_3d.shape[0], 3)
        z_dir = np.tile(A=np.array([0, 0, 1]), reps=(point_data_3d.shape[0], 1))
        y_dir = np.cross(z_dir, x_dir)

        back_roots = np.transpose(np.stack(arrays=[x_dir, y_dir, z_dir], axis=1), axes=(0, 2, 1))

    elif root_method == 'root_inv':
        x_dir = point_data_3d[:, mouse_id, neck_point_pos, :] - point_data_3d[:, mouse_id, tti_point_pos, :]
        root_len = np.linalg.norm(x=x_dir, axis=1).astype(np.float64)
        root_len[root_len < spatial_resolution_tolerance] = np.nan
        x_dir = (x_dir / root_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(point_data_3d.shape[0], 3)

        y_dir = np.zeros((point_data_3d.shape[0], 3))
        y_dir[:, 0] = -x_dir[:, 1]
        y_dir[:, 1] = x_dir[:, 0]
        y_dir[:, 2] = 0.
        y_len = np.linalg.norm(x=y_dir, axis=1).astype(np.float64)
        y_dir = (y_dir / y_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(point_data_3d.shape[0], 3)

        z_dir = np.cross(x_dir, y_dir)
        z_len = np.linalg.norm(x=z_dir, axis=1).astype(np.float64)
        z_dir = (z_dir / z_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(point_data_3d.shape[0], 3)

        back_roots = np.stack(arrays=[x_dir, y_dir, z_dir], axis=1)

    else:
        x_dir = point_data_3d[:, mouse_id, neck_point_pos, :] - point_data_3d[:, mouse_id, tti_point_pos, :]
        x_dir[:, 2] = 0.

        root_len = np.linalg.norm(x=x_dir, axis=1).astype(np.float64)
        root_len[root_len < spatial_resolution_tolerance] = np.nan
        x_dir = (x_dir / root_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(point_data_3d.shape[0], 3)

        y_dir = np.zeros((point_data_3d.shape[0], 3))
        y_dir[:, 0] = -x_dir[:, 1]
        y_dir[:, 1] = x_dir[:, 0]
        y_dir[:, 2] = 0.
        y_len = np.linalg.norm(x=y_dir, axis=1).astype(np.float64)
        y_dir = (y_dir / y_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(point_data_3d.shape[0], 3)

        z_dir = np.tile(A=np.array([0, 0, 1]), reps=(point_data_3d.shape[0], 1))

        back_roots = np.stack([x_dir, y_dir, z_dir], axis=1)

    return back_roots


def get_euler_ang(rot_matrix: np.ndarray) -> np.ndarray:
    """
    Computes Euler angles.
    NB: always in the order roll, pitch, yaw!

    Parameter
    ---------
    rot_matrix : np.ndarray
        A (n_frames, 3, 3) shape ndarray of rotation matrices.

    Returns
    -------
    desired_euler_angles : np.ndarray
        A (n_frames, 3) shape ndarray of Euler angles
        (in units of degrees); column are in order:
        roll, pitch, yaw.
    """

    rot_matrix_reshaped = np.reshape(a=rot_matrix, newshape=(rot_matrix.shape[0], 9)).copy()

    temp = np.sqrt((rot_matrix_reshaped[:, 8] * rot_matrix_reshaped[:, 8]) + (rot_matrix_reshaped[:, 5] * rot_matrix_reshaped[:, 5]))
    problematic_indices = np.where(temp < 0.0001)[0]
    good_indices = np.setdiff1d(np.arange(0, rot_matrix.shape[0], 1), problematic_indices)

    desired_euler_angles = np.zeros((rot_matrix.shape[0], 3))

    desired_euler_angles[good_indices, 0] = -np.arctan2(-rot_matrix_reshaped[good_indices, 5], rot_matrix_reshaped[good_indices, 8]) * 180. / np.pi
    desired_euler_angles[good_indices, 1] = np.arctan2(rot_matrix_reshaped[good_indices, 2], temp[good_indices]) * 180. / np.pi
    desired_euler_angles[good_indices, 2] = np.arctan2(-rot_matrix_reshaped[good_indices, 1], rot_matrix_reshaped[good_indices, 0]) * 180. / np.pi

    desired_euler_angles[problematic_indices, 0] = 0.0
    desired_euler_angles[problematic_indices, 1] = np.arctan2(rot_matrix_reshaped[problematic_indices, 2], temp[problematic_indices]) * 180. / np.pi
    desired_euler_angles[problematic_indices, 2] = np.arctan2(rot_matrix_reshaped[problematic_indices, 3], rot_matrix_reshaped[problematic_indices, 4]) * 180. / np.pi

    return desired_euler_angles


def get_back_angles(back_directions: np.ndarray) -> np.ndarray:
    """
    Computes Euler angles for the back:
        pitch: angle between the back and the horizontal plane
        yaw: angle between the back and the z-axis

    Parameter
    ---------
    back_directions : np.ndarray
        A (n_frames, 3) shape ndarray of back direction data
        Array of back direction data.

    Returns
    -------
    back_euler_angles : np.ndarray
        A (n_frames, 2) shape ndarray of
        back pitch and back yaw angles (in that order).
    """

    back_euler_angles = np.zeros((back_directions.shape[0], 2))

    def get_rotation(argument_ang):
        # rotations about z (yaw) are ok, but bounds on the other two to avoid flipping
        if abs(argument_ang[0]) > np.pi * 0.5 or abs(argument_ang[1]) > np.pi * 0.5:
            return [-1], [-1]
        rotate_z = np.array([[np.cos(argument_ang[2]), -np.sin(argument_ang[2]), 0], [np.sin(argument_ang[2]), np.cos(argument_ang[2]), 0], [0, 0, 1]])
        rotate_y = np.array([[np.cos(argument_ang[1]), 0, np.sin(argument_ang[1])], [0, 1, 0], [-np.sin(argument_ang[1]), 0, np.cos(argument_ang[1])]])
        rotate_x = np.array([[1, 0, 0], [0, np.cos(argument_ang[0]), -np.sin(argument_ang[0])], [0, np.sin(argument_ang[0]), np.cos(argument_ang[0])]])
        rotator = np.dot(rotate_x, np.dot(rotate_y, rotate_z))
        new_vector = np.einsum('ij,kj->ki', rotator, back_directions)
        return new_vector, rotator

    def distance_to_x_axis(argument_ang):
        # rotate around yaw and roll to center around zero
        rot_check, rot_m = get_rotation([0., argument_ang[0], argument_ang[1]])
        check_vec = np.array([np.nanmean(rot_check[:, 0]), np.nanmean(rot_check[:, 1]), np.nanmean(rot_check[:, 2])])
        check_vec = check_vec / np.sqrt(sum(check_vec ** 2))
        angle = np.arctan2(np.linalg.norm(np.cross(check_vec, np.array([1, 0, 0]))), np.dot(check_vec, np.array([1, 0, 0])))
        return abs(angle)

    res = minimize(distance_to_x_axis,
                   x0=np.array([-0.25 * np.pi + np.random.rand() * 0.5 * np.pi, -0.25 * np.pi + np.random.rand() * 0.5 * np.pi]),
                   method='nelder-mead',
                   options={'xatol': 1e-6, 'disp': False})

    temp_angles_back = res.x
    rotated_back_directions, back_rotator = get_rotation([0., temp_angles_back[0], temp_angles_back[1]])

    rotated_back_directions = rotated_back_directions / np.linalg.norm(rotated_back_directions)
    back_euler_angles[:, 0] = np.arctan2(rotated_back_directions[:, 2], np.sqrt(rotated_back_directions[:, 0] ** 2 + rotated_back_directions[:, 1] ** 2)) * 180. / np.pi
    back_euler_angles[:, 1] = -np.arctan2(rotated_back_directions[:, 1], rotated_back_directions[:, 0]) * 180. / np.pi

    return back_euler_angles


class FeatureZoo:
    feature_boundaries = {'speed': [0, 36], 'acceleration': [-120, 120],
                          'neck_elevation': [0, 12], 'neck_elevation_1st_der': [-18, 18], 'neck_elevation_2nd_der': [-90, 90],
                          'allo_roll': [-180, 180], 'allo_roll_1st_der': [-480, 480], 'allo_roll_2nd_der': [-2880, 2880],
                          'allo_pitch': [-90, 90], 'allo_pitch_1st_der': [-480, 480], 'allo_pitch_2nd_der': [-2880, 2880],
                          'allo_yaw': [-180, 180], 'allo_yaw_1st_der': [-480, 480], 'allo_yaw_2nd_der': [-2880, 2880],
                          'ego_yaw': [-180, 180], 'ego_yaw_1st_der': [-480, 480], 'ego_yaw_2nd_der': [-2880, 2880],
                          'back_pitch': [-54, 54], 'back_pitch_1st_der': [-90, 90], 'back_pitch_2nd_der': [-720, 720],
                          'back_yaw': [-36, 36], 'back_yaw_1st_der': [-90, 90], 'back_yaw_2nd_der': [-720, 720],
                          'body_dir': [-180, 180], 'body_dir_1st_der': [-480, 480], 'body_dir_2nd_der': [-2880, 2880],
                          'tail_curvature': [0, 1], 'tail_curvature_1st_der': [-1.8, 1.8], 'tail_curvature_2nd_der': [-10.8, 10.8],

                          'nose-nose': [0, 90], 'nose-nose_1st_der': [-36, 36], 'nose-nose_2nd_der': [-240, 240],
                          'TTI-TTI': [0, 90], 'TTI-TTI_1st_der': [-36, 36], 'TTI-TTI_2nd_der': [-240, 240],
                          'nose-TTI': [0, 90], 'nose-TTI_1st_der': [-36, 36], 'nose-TTI_2nd_der': [-240, 240],
                          'TTI-nose': [0, 90], 'TTI-nose_1st_der': [-36, 36], 'TTI-nose_2nd_der': [-240, 240],
                          'neck_elevation_diff': [-12, 12], 'neck_elevation_diff_1st_der': [-18, 18], 'neck_elevation_diff_2nd_der': [-90, 90],
                          'speed_diff': [-36, 36], 'speed_diff_1st_der': [-180, 180], 'speed_diff_2nd_der': [-1800, 1800],

                          'allo_yaw-nose': [-180, 180], 'allo_yaw-nose_1st_der': [-480, 480], 'allo_yaw-nose_2nd_der': [-2880, 2880],
                          'nose-allo_yaw': [-180, 180], 'nose-allo_yaw_1st_der': [-480, 480], 'nose-allo_yaw_2nd_der': [-2880, 2880],
                          'allo_yaw-TTI': [-180, 180], 'allo_yaw-TTI_1st_der': [-480, 480], 'allo_yaw-TTI_2nd_der': [-2880, 2880],
                          'TTI-allo_yaw': [-180, 180], 'TTI-allo_yaw_1st_der': [-480, 480], 'TTI-allo_yaw_2nd_der': [-2880, 2880]
                        }

    feature_labels = {'individual': {'speed': 'slow -- (cm/s) -- fast',
                                     'acceleration': 'slow -- (cm/s²) -- fast',
                                     'neck_elevation': 'down -- (cm) -- up',
                                     'neck_elevation_1st_der': 'down -- (cm/s) -- up',
                                     'neck_elevation_2nd_der': 'down -- (cm/s²) -- up',
                                     'allo_roll': 'ccw -- (°) -- cw',
                                     'allo_roll_1st_der': 'ccw -- (°/s) -- cw',
                                     'allo_roll_2nd_der': 'ccw -- (°/s²) -- cw',
                                     'allo_pitch': 'down -- (°) -- up',
                                     'allo_pitch_1st_der': 'down -- (°/s) -- up',
                                     'allo_pitch_2nd_der': 'down -- (°/s²) -- up',
                                     'allo_yaw': 'ccw -- (°) -- cw',
                                     'allo_yaw_1st_der': 'ccw -- (°/s) -- cw',
                                     'allo_yaw_2nd_der': 'ccw -- (°/s²) -- cw',
                                     'ego_yaw': 'ccw -- (°) -- cw',
                                     'ego_yaw_1st_der': 'ccw -- (°/s) -- cw',
                                     'ego_yaw_2nd_der': 'ccw -- (°/s²) -- cw',
                                     'back_pitch': 'down -- (°) -- up',
                                     'back_pitch_1st_der': 'down -- (°/s) -- up',
                                     'back_pitch_2nd_der': 'down -- (°/s²) -- up',
                                     'back_yaw': 'ccw -- (°) -- cw',
                                     'back_yaw_1st_der': 'ccw -- (°/s) -- cw',
                                     'back_yaw_2nd_der': 'ccw -- (°/s²) -- cw',
                                     'body_dir': 'ccw -- (°) -- cw',
                                     'body_dir_1st_der': 'ccw -- (°/s) -- cw',
                                     'body_dir_2nd_der': 'ccw -- (°/s²) -- cw',
                                     'tail_curvature': 'straight -- (a.u.) -- curved',
                                     'tail_curvature_1st_der': 'straight -- (a.u.) -- curved',
                                     'tail_curvature_2nd_der': 'straight -- (a.u.) -- curved'},
                      'social': {'nose-nose': 'near -- (cm) -- far',
                                 'nose-nose_1st_der': 'near -- (cm/s) -- far',
                                 'nose-nose_2nd_der': 'near -- (cm/s²) -- far',
                                 'TTI-TTI': 'near -- (cm) -- far',
                                 'TTI-TTI_1st_der': 'near -- (cm/s) -- far',
                                 'TTI-TTI_2nd_der': 'near -- (cm/s²) -- far',
                                 'nose-TTI': 'near -- (cm) -- far',
                                 'nose-TTI_1st_der': 'near -- (cm/s) -- far',
                                 'nose-TTI_2nd_der': 'near -- (cm/s²) -- far',
                                 'TTI-nose': 'near -- (cm) -- far',
                                 'TTI-nose_1st_der': 'near -- (cm/s) -- far',
                                 'TTI-nose_2nd_der': 'near -- (cm/s²) -- far',
                                 'neck_elevation_diff': 'm2 higher -- (cm) -- m1 higher',
                                 'neck_elevation_diff_1st_der': 'm2 higher -- (cm) -- m1 higher',
                                 'neck_elevation_diff_2nd_der': 'm2 higher -- (cm) -- m1 higher',
                                 'speed_diff': 'm2 faster -- m1 faster',
                                 'speed_diff_1st_der': 'm2 faster -- m1 faster',
                                 'speed_diff_2nd_der': 'm2 faster -- m1 faster',
                                 'allo_yaw-nose': 'ccw -- (°) -- cw',
                                 'allo_yaw-nose_1st_der': 'ccw -- (°/s) -- cw',
                                 'allo_yaw-nose_2nd_der': 'ccw -- (°/s²) -- cw',
                                 'nose-allo_yaw': 'ccw -- (°) -- cw',
                                 'nose-allo_yaw_1st_der': 'ccw -- (°/s) -- cw',
                                 'nose-allo_yaw_2nd_der': 'ccw -- (°/s²) -- cw',
                                 'allo_yaw-TTI': 'ccw -- (°) -- cw',
                                 'allo_yaw-TTI_1st_der': 'ccw -- (°/s) -- cw',
                                 'allo_yaw-TTI_2nd_der': 'ccw -- (°/s²) -- cw',
                                 'TTI-allo_yaw': 'ccw -- (°) -- cw',
                                 'TTI-allo_yaw_1st_der': 'ccw -- (°/s) -- cw',
                                 'TTI-allo_yaw_2nd_der': 'ccw -- (°/s²) -- cw'}
                      }


    def __init__(self, **kwargs):
        """
        Initializes the FeatureZoo class.

        Parameter
        ---------
        root_directory : str
            Root directory for data; defaults to None.
        neuronal_tuning_figures_dict : dict
            Dictionary of analyses parameters; defaults to None.
        message_output : function
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        with open((pathlib.Path(__file__).parent.parent / '_parameter_settings/visualizations_settings.json'), 'r') as json_file:
            self.visualizations_parameter_dict = json.load(json_file)


    def plot_feature_distributions(self, **kwargs):
        """
        Plots histograms of all available behavioral features.

        Parameter
        ---------
        feature_dict : dict
            Binned feature data; defaults to None.
        mouse_id_list : list
            Mouse IDs to plot; defaults to None.
        session_exp_code : str
            Session experiment code; defaults to None.
        plot_file_name : str
            Name of the plot file; defaults to None.

        Returns
        -------
        behavioral_feature_distributions : .pdf
           Plot of all feature histograms.
        """

        feature_dict = kwargs['feature_dict'] if 'feature_dict' in kwargs.keys() and isinstance(kwargs['feature_dict'], dict) else None
        mouse_id_list = kwargs['mouse_id_list'] if 'mouse_id_list' in kwargs.keys() and isinstance(kwargs['mouse_id_list'], list) else None
        session_exp_code = kwargs['session_exp_code'] if 'session_exp_code' in kwargs.keys() and isinstance(kwargs['session_exp_code'], str) else None
        plot_file_name = kwargs['plot_file_name'] if 'plot_file_name' in kwargs.keys() and isinstance(kwargs['plot_file_name'], str) else None

        # get colors
        experiment_info_dict = extract_information(experiment_code=session_exp_code)
        mouse_colors = choose_animal_colors(exp_info_dict=experiment_info_dict, visualizations_parameter_dict=self.visualizations_parameter_dict)

        if feature_dict is not None and mouse_id_list is not None:

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
            for feature_key in feature_dict.keys():
                mouse_id = feature_key.split('.')[0]
                if f'individual.{mouse_id}' not in plot_features.keys() and '-' not in mouse_id:
                    plot_features[f'individual.{mouse_id}'] = []

                if '-' not in mouse_id:
                    plot_features[f'individual.{mouse_id}'].append(feature_key)
                else:
                    if 'social' not in plot_features.keys():
                        plot_features['social'] = []
                    plot_features['social'].append(feature_key)

            with PdfPages(plot_file_name) as pdf_fig:
                for plot_feature_key in plot_features.keys():
                    if 'social' in plot_feature_key:
                        histogram_color = mouse_color_dict['social']
                        mouse_colormap = 0
                    else:
                        histogram_color = mouse_color_dict[plot_feature_key.split('.')[-1]]
                        mouse_colormap = mouse_colormap_dict[plot_feature_key.split('.')[-1]]

                    with warnings.catch_warnings():
                        warnings.simplefilter(action="ignore")

                        row_num = int(np.ceil((len(plot_features[plot_feature_key])) / 6))
                        fig = plt.figure(figsize=(6.4, float(row_num)), dpi=600, tight_layout=True)
                        fig.suptitle(t=f'{plot_feature_key}', fontsize=8)
                        gs = gridspec.GridSpec(nrows=row_num, ncols=6)

                        gs_x = 0
                        gs_y = 0
                        for feature_idx, feature in enumerate(plot_features[plot_feature_key]):
                            if 'space' in feature:
                                cbar_width = .005
                                cbar_height = .04
                                cbar_ypos_extra = .11

                                ax = fig.add_subplot(gs[gs_x, gs_y])
                                occ = ax.imshow(X=feature_dict[feature]['occ_array'][:, :],
                                                cmap=mouse_colormap,
                                                vmin=0,
                                                interpolation='gaussian',
                                                aspect='equal')
                                ax.set_title(label='Spatial occupancy (smoothed)',
                                             fontsize=4,
                                             pad=4)
                                ax.set_xticks([])
                                ax.set_xlabel(xlabel='X (cm)',
                                              fontsize=4,
                                              labelpad=1)
                                ax.set_yticks([])
                                ax.set_ylabel(ylabel='Y (cm)',
                                              fontsize=4,
                                              labelpad=1)

                                ax_position = ax.get_position()
                                cb_ax = fig.add_axes((ax_position.x0 + 0.03,
                                                      ax_position.y0 + cbar_ypos_extra,
                                                      cbar_width,
                                                      cbar_height))
                                cbar = fig.colorbar(mappable=occ,
                                                    orientation='vertical',
                                                    cax=cb_ax)
                                cbar_vmin, cbar_vmax = cbar.mappable.get_clim()
                                cbar.set_ticks([cbar_vmin, cbar_vmax])
                                cbar.set_ticklabels(ticklabels=[f"{int(cbar_vmin)}", f"{int(np.ceil(cbar_vmax))}"], fontsize=2)
                                cbar.ax.tick_params(axis='both', which='both', length=0, pad=.5)
                                cbar.outline.set_visible(True)

                                gs_y += 1
                                if gs_y > 5:
                                    gs_y = 0
                                    gs_x += 1
                            else:
                                ax = fig.add_subplot(gs[gs_x, gs_y])
                                ax.tick_params(axis='both', which='both', length=1.5, pad=.25)
                                ax.bar(x=feature_dict[feature]['bin_centers'],
                                       height=feature_dict[feature]['occ_array'],
                                       width=feature_dict[feature]['bin_edges'][1] - feature_dict[feature]['bin_edges'][0],
                                       align='center',
                                       color=histogram_color,
                                       ec='#000000',
                                       alpha=.75,
                                       lw=.1)
                                ax.set_title(label=f"{feature.split('.')[-1]}",
                                             fontsize=4,
                                             pad=1)
                                ax.set_xticks(ticks=[self.feature_boundaries[feature.split('.')[-1]][0],
                                                     self.feature_boundaries[feature.split('.')[-1]][1]],
                                              labels=[f"{self.feature_boundaries[feature.split('.')[-1]][0]:.1f}",
                                                      f"{self.feature_boundaries[feature.split('.')[-1]][1]:.1f}"],
                                              rotation=0,
                                              fontsize=2)
                                ax.set_xlabel(xlabel=f"{self.feature_labels[plot_feature_key.split('.')[0]][feature.split('.')[-1]]}",
                                              fontsize=3,
                                              labelpad=1)
                                temp_ymin, temp_ymax = ax.get_ylim()
                                ax.set_yticks(ticks=[0, int(np.ceil(temp_ymax))-10], labels=['0', f'{int(np.ceil(temp_ymax))-10}'], rotation=0, fontsize=2)
                                ax.set_ylabel(ylabel='Occupancy (s)',
                                              fontsize=3,
                                              labelpad=1)
                                ax.set_box_aspect(1)

                                gs_y += 1
                                if gs_y > 5:
                                    gs_y = 0
                                    gs_x += 1

                        pdf_fig.savefig(dpi=600)
                        plt.clf()
                        plt.close('all')



    def save_behavioral_features_to_file(self):
        """
        Computes and saves behavioral features to file.

        Parameter
        ---------
        Uses the following set of parameters:
            head_points : list
                Head points to use for computing Euler angles; defaults to ["Head", "Ear_R", "Ear_L", "Nose"].
                NB: order of points is important!
            tail_points : list
                Tail points to use for computing tail curvature; defaults to ["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"].
                NB: order of points is important!
            back_root_points : list
                Back root points to use for computing Euler angles; defaults to ["Neck", "Trunk", "TTI"].
                NB: order of points is important!
            capture_fr : int
                Capture framerate; defaults to 150 (fps).
            derivative_bins : int
                Number of bins to use for computing derivatives; defaults to +/- 10.

        Returns
        -------
        behavioral_features_csv : .csv
           Data sheet w/ behavioral features.
        """

        self.message_output(f"Computing behavioral features started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}")
        QTest.qWait(1000)

        tracked_file_loc = glob.glob(f"{self.root_directory}{os.sep}video{os.sep}**{os.sep}[!speaker]*_points3d_translated_rotated_metric.h5")[0]

        # load tracking data
        with h5py.File(tracked_file_loc, mode='r') as tracking_data_3d:
            mouse_data = np.array(tracking_data_3d['tracks']).astype(np.float64)
            mouse_nodes = [elem.decode('utf-8') for elem in list(tracking_data_3d['node_names'])]
            track_names = [elem.decode('utf-8') for elem in list(tracking_data_3d['track_names'])]
            experimental_code = tracking_data_3d['experimental_code'][()].decode('utf-8')
            empirical_camera_sr = float(tracking_data_3d['recording_frame_rate'][()])

        self.message_output(f"Working on tracking data of shape {mouse_data.shape} with experiment code '{experimental_code}' ({empirical_camera_sr} fps), track names {track_names} \n"
                            f"and nodes {mouse_nodes}")
        QTest.qWait(1000)

        # # # compute individual features
        head_position = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        speed = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        speed_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        speed_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        neck_elevation = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        neck_elevation_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        neck_elevation_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        global_head_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_head_angles_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_head_angles_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        ego_head_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        ego_head_angles_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        ego_head_angles_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_back_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 2))
        global_back_angles_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 2))
        global_back_angles_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 2))
        global_root_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_root_angles_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_root_angles_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        tail_curvature = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        tail_curvature_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        tail_curvature_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))

        for mouse_num in range(mouse_data.shape[1]):

            # # head position (in cm)
            head_position[mouse_num, :, :] = mouse_data[:, mouse_num, mouse_nodes.index('Head'), :] * 100

            # # speed (cm/s) and acceleration (cm/s^2)
            exclude_tail_points = [mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][1]), mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][2]),
                                   mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][3]), mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][4])]
            exclude_tail_mask = np.ones(mouse_data.shape[2], dtype=bool)
            exclude_tail_mask[exclude_tail_points] = False

            speed[mouse_num, :, :] = calculate_speed(tracked_points_array=mouse_data[:, mouse_num, exclude_tail_mask, :],
                                                     capture_framerate=empirical_camera_sr,
                                                     smoothing_time_window=.015)

            speed_1st_der[mouse_num, :, :], speed_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=speed[mouse_num, :, :],
                                                                                                   diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                   is_angle=False,
                                                                                                   capture_fr=empirical_camera_sr)

            # # neck elevation (cm)
            neck_elevation_temp = mouse_data[:, mouse_num, mouse_nodes.index('Neck'), 2] * 100
            neck_elevation_temp[neck_elevation_temp < 0] = np.nan
            neck_elevation[mouse_num, :, 0] = neck_elevation_temp
            neck_elevation_1st_der[mouse_num, :, :], neck_elevation_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=neck_elevation[mouse_num, :, :],
                                                                                                                     diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                                     is_angle=False,
                                                                                                                     capture_fr=empirical_camera_sr)

            # # head Euler angles (degrees)
            head_input_arr = np.array([mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['head_points'][0]), :],
                                       mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['head_points'][1]), :],
                                       mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['head_points'][2]), :],
                                       mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['head_points'][3]), :]])

            head_input_arr = np.swapaxes(head_input_arr, axis1=0, axis2=1)

            average_head_point = get_average_point(head_input_arr)
            global_head_root = get_head_root(head_input_arr, average_head_point, rotation_type='regular')
            global_head_angles[mouse_num, :, :] = get_euler_ang(global_head_root)

            # in some sessions, the average ear points end up. e.g., being tracked more posterior to the head point, so a rotation matrix modification is necessary
            roll_extreme_proportion = np.count_nonzero((global_head_angles[mouse_num, :, 0] < -120) | (global_head_angles[mouse_num, :, 0] > 120)) / global_head_angles[mouse_num, :, 0].shape[0]
            if roll_extreme_proportion > 0.5:
                global_head_root = get_head_root(head_input_arr, average_head_point, rotation_type='roll_issue')
                global_head_angles[mouse_num, :, :] = get_euler_ang(global_head_root)

            pitch_positive_proportion = np.count_nonzero(global_head_angles[mouse_num, :, 1] > 0) / global_head_angles[mouse_num, :, 1].shape[0]
            if pitch_positive_proportion > 0.5:
                global_head_root = get_head_root(head_input_arr, average_head_point, rotation_type='pitch_issue')
                global_head_angles[mouse_num, :, :] = get_euler_ang(global_head_root)
                global_head_angles[mouse_num, :, 0] = -global_head_angles[mouse_num, :, 0]


            global_head_angles_1st_der[mouse_num, :, :], global_head_angles_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=global_head_angles[mouse_num, :, :],
                                                                                                                             diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                                             is_angle=True,
                                                                                                                             capture_fr=empirical_camera_sr)

            # # egocentric head angles (degrees)
            back_root_inv_oriented = get_back_root(point_data_3d=mouse_data, mouse_id=mouse_num,
                                                   neck_point_pos=mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][0]),
                                                   tti_point_pos=mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][2]),
                                                   root_method='other')

            root_oriented_x = np.einsum('ijk,ik->ij', back_root_inv_oriented, global_head_root[:, 0, :])
            root_oriented_y = np.einsum('ijk,ik->ij', back_root_inv_oriented, global_head_root[:, 1, :])
            root_oriented_z = np.einsum('ijk,ik->ij', back_root_inv_oriented, global_head_root[:, 2, :])
            head_relative_to_body_root = np.array([root_oriented_x, root_oriented_y, root_oriented_z])
            head_relative_to_body_root = np.swapaxes(head_relative_to_body_root, axis1=0, axis2=1)

            ego_head_angles[mouse_num, :, :] = get_euler_ang(head_relative_to_body_root)
            ego_head_angles_1st_der[mouse_num, :, :], ego_head_angles_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=ego_head_angles[mouse_num, :, :],
                                                                                                                       diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                                       is_angle=True,
                                                                                                                       capture_fr=empirical_camera_sr)

            # # back Euler angles (pitch and yaw, degrees)
            back_root_inv = get_back_root(point_data_3d=mouse_data, mouse_id=mouse_num,
                                          neck_point_pos=mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][0]),
                                          tti_point_pos=mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][2]),
                                          root_method='root_inv')

            inv_back_vector = mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][0]), :] - mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][1]), :]
            inv_normalized_back_vector = inv_back_vector / np.linalg.norm(inv_back_vector)
            back_directions = np.einsum('bij,bj->bi', back_root_inv, inv_normalized_back_vector)

            global_back_angles[mouse_num, :, :] = get_back_angles(back_directions)

            global_back_angles_1st_der[mouse_num, :, :], global_back_angles_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=global_back_angles[mouse_num, :, :],
                                                                                                                             diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                                             is_angle=True,
                                                                                                                             capture_fr=empirical_camera_sr)
            # # global body direction (degrees)
            back_root = get_back_root(point_data_3d=mouse_data, mouse_id=mouse_num,
                                      neck_point_pos=mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][0]),
                                      tti_point_pos=mouse_nodes.index(self.behavioral_parameters_dict['back_root_points'][2]),
                                      root_method='default')

            global_root_angles[mouse_num, :, :] = -get_euler_ang(back_root)
            global_root_angles_1st_der[mouse_num, :, :], global_root_angles_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=global_root_angles[mouse_num, :, :],
                                                                                                                             diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                                             is_angle=True,
                                                                                                                             capture_fr=empirical_camera_sr)

            # # tail curvature (arbitrary units
            tail_curvature_arr = np.array([mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][0]), :],
                                           mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][1]), :],
                                           mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][2]), :],
                                           mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][3]), :],
                                           mouse_data[:, mouse_num, mouse_nodes.index(self.behavioral_parameters_dict['tail_points'][4]), :]])

            tail_curvature_arr = np.swapaxes(tail_curvature_arr, axis1=0, axis2=1)

            tail_curvature[mouse_num, :, :] = calculate_tail_curvature(input_arr=tail_curvature_arr)
            tail_curvature_1st_der[mouse_num, :, :], tail_curvature_2nd_der[mouse_num, :, :] = calculate_derivatives(input_arr=tail_curvature[mouse_num, :, :],
                                                                                                                     diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                                                     is_angle=False,
                                                                                                                     capture_fr=empirical_camera_sr)

        # # # compute social features
        if mouse_data.shape[1] == 2:

            # # social distances (cm or cm/s)
            social_distances = np.zeros((mouse_data.shape[0], 6))

            # nose to nose distance between mice
            social_distances[:, 0] = np.linalg.norm(mouse_data[:, 0, mouse_nodes.index('Nose'), :] - mouse_data[:, 1, mouse_nodes.index('Nose'), :], axis=1) * 100

            # TTI to TTI distance between mice
            social_distances[:, 1] = np.linalg.norm(mouse_data[:, 0, mouse_nodes.index('TTI'), :] - mouse_data[:, 1, mouse_nodes.index('TTI'), :], axis=1) * 100

            # nose (first mouse) to TTI (second mouse) distance between mice
            social_distances[:, 2] = np.linalg.norm(mouse_data[:, 0, mouse_nodes.index('Nose'), :] - mouse_data[:, 1, mouse_nodes.index('TTI'), :], axis=1) * 100

            # TTI (first mouse) to nose (second mouse) distance between mice
            social_distances[:, 3] = np.linalg.norm(mouse_data[:, 0, mouse_nodes.index('TTI'), :] - mouse_data[:, 1, mouse_nodes.index('Nose'), :], axis=1) * 100

            # neck elevation distance
            social_distances[:, 4] = neck_elevation[0, :, 0] - neck_elevation[1, :, 0]

            # speed difference
            social_distances[:, 5] = speed[0, :, 0] - speed[1, :, 0]

            # compute derivatives
            social_distances_1st_der, social_distances_2nd_der = calculate_derivatives(input_arr=social_distances,
                                                                                       diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                       is_angle=False,
                                                                                       capture_fr=empirical_camera_sr)

            # # social angles (in degrees)
            social_angles = np.zeros((mouse_data.shape[0], 4))

            # planar head direction(mouse1)-Nose(mouse2) angle
            social_angles[:, 0] = calculate_planar_social_angle(point1_arr=mouse_data[:, 0, mouse_nodes.index('Head'), :2],
                                                                point2_arr=mouse_data[:, 0, mouse_nodes.index('Nose'), :2],
                                                                point3_arr=mouse_data[:, 1, mouse_nodes.index('Nose'), :2])

            # planar head direction(mouse2)-Nose(mouse1) angle
            social_angles[:, 1] = calculate_planar_social_angle(point1_arr=mouse_data[:, 1, mouse_nodes.index('Head'), :2],
                                                                point2_arr=mouse_data[:, 1, mouse_nodes.index('Nose'), :2],
                                                                point3_arr=mouse_data[:, 0, mouse_nodes.index('Nose'), :2])

            # planar head direction(mouse1)-TTI(mouse2) angle
            social_angles[:, 2] = calculate_planar_social_angle(point1_arr=mouse_data[:, 0, mouse_nodes.index('Head'), :2],
                                                                point2_arr=mouse_data[:, 0, mouse_nodes.index('Nose'), :2],
                                                                point3_arr=mouse_data[:, 1, mouse_nodes.index('TTI'), :2])

            # planar head direction(mouse2)-TTI(mouse1) angle
            social_angles[:, 3] = calculate_planar_social_angle(point1_arr=mouse_data[:, 1, mouse_nodes.index('Head'), :2],
                                                                point2_arr=mouse_data[:, 1, mouse_nodes.index('Nose'), :2],
                                                                point3_arr=mouse_data[:, 0, mouse_nodes.index('TTI'), :2])


            social_angles_1st_der, social_angles_2nd_der = calculate_derivatives(input_arr=social_angles,
                                                                                 diff_bins=self.behavioral_parameters_dict['derivative_bins'],
                                                                                 is_angle=True,
                                                                                 capture_fr=empirical_camera_sr)

        # # # save data to .csv file
        behavioral_features_df = pls.DataFrame()

        for mouse_num in range(mouse_data.shape[1]):
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.spaceX", head_position[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.spaceY", head_position[mouse_num, :, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.spaceZ", head_position[mouse_num, :, 2]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.speed", speed[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.acceleration", speed_1st_der[mouse_num, :, 0]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.neck_elevation", neck_elevation[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.neck_elevation_1st_der", neck_elevation_1st_der[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.neck_elevation_2nd_der", neck_elevation_2nd_der[mouse_num, :, 0]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_roll", global_head_angles[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_roll_1st_der", global_head_angles_1st_der[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_roll_2nd_der", global_head_angles_2nd_der[mouse_num, :, 0]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_pitch", global_head_angles[mouse_num, :, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_pitch_1st_der", global_head_angles_1st_der[mouse_num, :, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_pitch_2nd_der", global_head_angles_2nd_der[mouse_num, :, 1]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_yaw", global_head_angles[mouse_num, :, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_yaw_1st_der", global_head_angles_1st_der[mouse_num, :, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.allo_yaw_2nd_der", global_head_angles_2nd_der[mouse_num, :, 2]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.ego_yaw", ego_head_angles[mouse_num, :, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.ego_yaw_1st_der", ego_head_angles_1st_der[mouse_num, :, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.ego_yaw_2nd_der", ego_head_angles_2nd_der[mouse_num, :, 2]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.back_pitch", global_back_angles[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.back_pitch_1st_der", global_back_angles_1st_der[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.back_pitch_2nd_der", global_back_angles_2nd_der[mouse_num, :, 0]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.back_yaw", global_back_angles[mouse_num, :, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.back_yaw_1st_der", global_back_angles_1st_der[mouse_num, :, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.back_yaw_2nd_der", global_back_angles_2nd_der[mouse_num, :, 1]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.body_dir", global_root_angles[mouse_num, :, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.body_dir_1st_der", global_root_angles_1st_der[mouse_num, :, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.body_dir_2nd_der", global_root_angles_2nd_der[mouse_num, :, 2]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.tail_curvature", tail_curvature[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.tail_curvature_1st_der", tail_curvature_1st_der[mouse_num, :, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[mouse_num]}.tail_curvature_2nd_der", tail_curvature_2nd_der[mouse_num, :, 0]))


        if mouse_data.shape[1] == 2:

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-nose", social_distances[:, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-nose_1st_der", social_distances_1st_der[:, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-nose_2nd_der", social_distances_2nd_der[:, 0]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-TTI", social_distances[:, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-TTI_1st_der", social_distances_1st_der[:, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-TTI_2nd_der", social_distances_2nd_der[:, 1]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-TTI", social_distances[:, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-TTI_1st_der", social_distances_1st_der[:, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-TTI_2nd_der", social_distances_2nd_der[:, 2]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-nose", social_distances[:, 3]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-nose_1st_der", social_distances_1st_der[:, 3]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-nose_2nd_der", social_distances_2nd_der[:, 3]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.neck_elevation_diff", social_distances[:, 4]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.neck_elevation_diff_1st_der", social_distances_1st_der[:, 4]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.neck_elevation_diff_2nd_der", social_distances_2nd_der[:, 4]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.speed_diff", social_distances[:, 5]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.speed_diff_1st_der", social_distances_1st_der[:, 5]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.speed_diff_2nd_der", social_distances_2nd_der[:, 5]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.allo_yaw-nose", social_angles[:, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.allo_yaw-nose_1st_der", social_angles_1st_der[:, 0]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.allo_yaw-nose_2nd_der", social_angles_2nd_der[:, 0]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-allo_yaw", social_angles[:, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-allo_yaw_1st_der", social_angles_1st_der[:, 1]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.nose-allo_yaw_2nd_der", social_angles_2nd_der[:, 1]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.allo_yaw-TTI", social_angles[:, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.allo_yaw-TTI_1st_der", social_angles_1st_der[:, 2]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.allo_yaw-TTI_2nd_der", social_angles_2nd_der[:, 2]))

            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-allo_yaw", social_angles[:, 3]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-allo_yaw_1st_der", social_angles_1st_der[:, 3]))
            behavioral_features_df = behavioral_features_df.with_columns(pls.Series(f"{track_names[0]}-{track_names[1]}.TTI-allo_yaw_2nd_der", social_angles_2nd_der[:, 3]))

        # # # # compute feature distributions
        feature_distribution_dict = {}
        space_computed_occ = {one_mouse: False for one_mouse in track_names}
        for column in behavioral_features_df.columns:
            if 'space' not in column:
                feature_distribution_dict[column] = {}
                (feature_distribution_dict[column]['occ_array'],
                 feature_distribution_dict[column]['bin_centers'],
                 feature_distribution_dict[column]['bin_edges']) = generate_feature_distributions(feature_arr=behavioral_features_df.select(column).to_numpy(),
                                                                                                  min_val=self.feature_boundaries[column.split('.')[1]][0],
                                                                                                  max_val=self.feature_boundaries[column.split('.')[1]][1],
                                                                                                  num_bins=36,
                                                                                                  camera_fr=empirical_camera_sr,
                                                                                                  space_bool=False)
            else:
                if space_computed_occ[f"{column.split('.')[0]}"] is False:
                    feature_distribution_dict[column] = {}
                    (feature_distribution_dict[column]['occ_array'],
                     feature_distribution_dict[column]['bin_centers'],
                     feature_distribution_dict[column]['bin_edges']) = generate_feature_distributions(feature_arr=np.stack(arrays=(np.array(behavioral_features_df.select(f"{column.split('.')[0]}.spaceX").to_numpy()),
                                                                                                                                   np.array(behavioral_features_df.select(f"{column.split('.')[0]}.spaceY").to_numpy())),
                                                                                                                           axis=1),
                                                                                                      min_val=-32,
                                                                                                      max_val=32,
                                                                                                      num_bins=196,
                                                                                                      camera_fr=empirical_camera_sr,
                                                                                                      space_bool=True)
                    space_computed_occ[f"{column.split('.')[0]}"] = True

        # # # # plot feature distributions
        self.plot_feature_distributions(feature_dict=feature_distribution_dict,
                                        mouse_id_list=track_names,
                                        session_exp_code=experimental_code,
                                        plot_file_name=f"{tracked_file_loc[:-3]}_behavioral_features_histograms.pdf")

        # # # # save data to .csv file
        behavioral_features_df.write_csv(file=f"{tracked_file_loc[:-3]}_behavioral_features.csv", separator=',', include_header=True)
