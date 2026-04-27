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
(6) Nose-TTI distance (7) Nose-TTI distance der (8) Nose-TTI distance 2der (9) TTI-Nose distance (10) TTI-Nose distance der (11) TTI-Nose distance 2der
(12) Yaw-Nose (13) Yaw-Nose der (14) Yaw-Nose 2der (15) Nose-Yaw (16) Nose-Yaw der (17) Nose-Yaw 2der
(18) Yaw-TTI (19) Yaw-TTI der (20) Yaw-TTI 2der (21) TTI-Yaw (22) TTI-Yaw der (23) TTI-Yaw 2der
(24) Pitch-Nose (25) Pitch-Nose der (26) Pitch-Nose 2der (27) Nose-Pitch (28) Nose-Pitch der (29) Nose-Pitch 2der
(30) Pitch-TTI (31) Pitch-TTI der (32) Pitch-TTI 2der (33) TTI-Pitch (34) TTI-Pitch der (35) TTI-Pitch 2der

NB: Yaw-* and Pitch-* are egocentric — the yaw/pitch components of the
target body point expressed in the observer's anatomical head frame
(via get_egocentric_direction). yaw=0,pitch=0 means the target sits on
the observer's gaze axis.

[C] SOCIAL ENGAGEMENT INDICES
(0) mouse1-mouse2 orofacial SEI (1) mouse1-mouse2 orofacial SEI der (2) mouse1-mouse2 orofacial SEI 2der
(3) mouse1-mouse2 anogenital SEI (4) mouse1-mouse2 anogenital SEI der (5) mouse1-mouse2 anogenital SEI 2der
(6) mouse2-mouse1 orofacial SEI (7) mouse2-mouse1 orofacial SEI der (8) mouse2-mouse1 orofacial SEI 2der
(9) mouse2-mouse1 anogenital SEI (10) mouse2-mouse1 anogenital SEI der (11) mouse2-mouse1 anogenital SEI 2der
"""

from __future__ import annotations

import itertools
import json
import pathlib
import warnings
from datetime import datetime

import h5py
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
from astropy.convolution import Gaussian1DKernel, convolve
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize

from ..os_utils import first_match_or_raise
from ..time_utils import is_gui_context, smart_wait
from ..visualizations.auxiliary_plot_functions import (
    choose_animal_colors,
    create_colormap,
)
from .decode_experiment_label import extract_information

fm.fontManager.addfont(pathlib.Path(__file__).parent.parent / "fonts/Helvetica.ttf")
plt.style.use(pathlib.Path(__file__).parent.parent / "_config/usv_playpen.mplstyle")


def generate_feature_distributions(
    feature_arr: np.ndarray,
    min_val: int | float,
    max_val: int | float,
    num_bins: int | float,
    camera_fr: int | float,
    space_bool: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    occ_array (np.ndarray)
        A (num_bins) or (num_bins, num_bins) shape ndarray
        of occupancy (in seconds) for each feature.
    bin_centers (np.ndarray)
        A (num_bins) shape ndarray of bin centers for given feature.
    bin_edges (np.ndarray)
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
                occ_array[i - 1, j - 1] = (
                    np.sum(
                        (
                            (feature_arr[:, 0] > bin_edges[i - 1])
                            * (feature_arr[:, 0] <= bin_edges[i])
                        )
                        * (
                            (feature_arr[:, 1] > bin_edges[j - 1])
                            * (feature_arr[:, 1] <= bin_edges[j])
                        )
                    )
                    / camera_fr
                )
    else:
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        occ_array = np.zeros(num_bins)
        for i in range(1, np.shape(bin_edges)[0], 1):
            occ_array[i - 1] = (
                np.sum((feature_arr > bin_edges[i - 1]) * (feature_arr <= bin_edges[i]))
                / camera_fr
            )

    return occ_array, bin_centers, bin_edges


def calculate_derivatives(
    input_arr: np.ndarray,
    diff_bins: int,
    capture_fr: int | float,
    is_angle: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays w/ first and second derivatives.
    NB: Computed according to the central difference derivative!

    The first derivative at frame t is approximated as
    (x[t + diff_bins] - x[t - diff_bins]) / (2 * diff_bins / capture_fr),
    i.e. a symmetric finite difference over a window of (2 * diff_bins)
    frames centered on t. The second derivative is computed identically
    on the first-derivative array, so the effective stencil has total
    width (4 * diff_bins) frames. The first and last `diff_bins` samples
    of each derivative are returned as NaN because the central-difference
    window cannot be evaluated there. Locations of NaN in `input_arr` are
    propagated to both output arrays so that downstream consumers can
    distinguish "boundary NaN" from "missing data NaN".

    When `is_angle` is True the first derivative is wrapped into
    (-180, 180] before division so that crossings of the +/-180 deg
    discontinuity (e.g. yaw wrap-around) do not produce spurious large
    angular velocities; the second derivative is left unwrapped because
    physical angular accelerations are not periodic.

    Parameters
    ----------
    input_arr (np.ndarray)
         A (n_frames, n_features) shape ndarray containing feature data to compute derivatives on.
    diff_bins : int
        Number of bins for the central difference derivative; defaults to None.
    is_angle (bool)
        Is the feature data in angles or not; defaults to False.
    capture_fr (int / float)
        Capture frame rate of the cameras; defaults to None (fps).

    Returns
    -------
    first_der, second_der (tuple (np.ndarray, np.ndarray))
        A tuple of 2 (n_frames, n_features) shape np.ndarray
        w/ first and second derivatives for selected features.
    """

    nan_positions = np.isnan(input_arr)

    first_der = np.zeros(shape=input_arr.shape)
    first_der[:] = np.nan

    second_der = np.zeros(shape=input_arr.shape)
    second_der[:] = np.nan

    # calculate first derivative
    first_der[diff_bins:-diff_bins, :] = (
        input_arr[(diff_bins * 2) :, :] - input_arr[: -(diff_bins * 2), :]
    )
    if is_angle:
        first_der[first_der > 180] -= 360
        first_der[first_der < -180] += 360
    first_der = first_der / (2.0 * diff_bins / capture_fr)
    first_der[nan_positions] = np.nan

    # calculate second derivative
    second_der[diff_bins:-diff_bins, :] = (
        first_der[(diff_bins * 2) :, :] - first_der[: -(diff_bins * 2), :]
    )
    second_der = second_der / (2.0 * diff_bins / capture_fr)
    second_der[nan_positions] = np.nan

    return first_der, second_der


def calculate_sei(
        tracks: np.ndarray,
        speed_arr: np.ndarray,
        observer_idx: int,
        observed_idx: int,
        observed_node_idx: int,
        observer_head_root: np.ndarray,
        idx_nose: int = 0,
        idx_tti: int = 3,
        idx_head: int = 5,
        v_max: float = None,
        sigma_yaw_deg: int | float = 45.0,
        sigma_pitch_deg: int | float = 45.0,
) -> np.ndarray:
    """
    Computes the Social Engagement Index (SEI) using a pursuit-proximity weight transition.

    The SEI quantifies a subject's engagement with a partner by integrating postural
    orientation, movement vigor, and distance. It transitions between 'action-based
    pursuit' at a distance and 'attention-based focus' up close.

    Mathematical Logic:
    -------------------
    1. Orientation: A separable, axis-aligned Gaussian gate over the
       observer's egocentric (yaw, pitch) error toward the target. The
       inter-point vector (target - observer_head) is rotated into the
       observer's anatomical head frame via `observer_head_root`, and
       its yaw and pitch components in that frame are read off via
       `get_egocentric_direction`. The gate is
           gaze = exp(-yaw^2 / (2 * sigma_yaw^2))
                * exp(-pitch^2 / (2 * sigma_pitch^2))
       so that yaw=0,pitch=0 (target on the observer's gaze axis) gives
       1, and the score decays smoothly as the target moves off-axis in
       either channel. sigma_yaw_deg and sigma_pitch_deg control the
       acceptance cone width per channel and are tunable.
    2. Sharpening: As distance (d) decreases, a dynamic *bounded*
       exponent `gamma = 1 + tanh(L / d)` mildly sharpens the gate,
       requiring slightly higher angular precision for high scores at
       close range. The `tanh` saturates at 1 as `d -> 0`, so `gamma`
       lives in `[1, 2]` and the gate cannot collapse: at infinite
       distance gamma = 1 (no sharpening), at touching distance
       gamma = 2 (gate squared). This replaces the legacy
       `gamma = 1 + L/d` form, which grew unboundedly (gamma > 20 at
       common close-engagement distances) and crushed `gaze_score`
       to near-zero exactly when the SEI should have been highest —
       the near-touching, sniffing frames. With the bounded form the
       Gaussian gate keeps doing the angular-acceptance work and the
       sharpening only contributes a soft second-order tightening.
    3. Social Weight (W): An exponential interpolator that balances
       speed and proximity:
       - At distance (d > L): Speed (V/V_max) is required to identify
         active pursuit.
       - Up close (d < L): Proximity (e^-d/L) ensures engagement remains
         high during stationary investigation (sniffing), even if
         locomotor speed is zero.

    Note: this version of the SEI replaces the legacy 3D cosine
    similarity with the explicit (yaw, pitch) Gaussian gate. Output is
    in [0, 1] (unsigned); the legacy SEI's negative values for
    "facing-away" frames are absorbed into the smooth Gaussian decay
    (gate becomes near-zero rather than negative when the target sits
    far off the observer's gaze axis).

    Parameters:
    -----------
    tracks : np.ndarray
        3D tracking data of shape (n_frames, n_animals, n_bodypoints, 3).
    speed_arr : np.ndarray
        (n_frames,) array of instantaneous speed for the observer mouse.
    observer_idx : int
        Index of the animal whose engagement is being measured.
    observed_idx : int
        Index of the social partner animal.
    observed_node_idx : int
        The index of the target body point on the observed partner (e.g., Nose or TTI).
    observer_head_root : np.ndarray
        A (n_frames, 3, 3) rotation tensor for the observer mouse, with
        rows (h_x, h_y, h_z) expressing the observer's anatomical body
        axes in world coordinates (typically `global_head_roots[observer_idx]`).
    idx_nose : int, optional
        Index for the observer's Nose (default 0).
    idx_tti : int, optional
        Index for the observer's Tail-Thorax Interface/Tail-base (default 3).
    idx_head : int, optional
        Index for the observer's Head/Pivot (default 5).
    v_max : float, optional
        Normalization factor for speed. Defaults to the 99th percentile of speed_arr.
    sigma_yaw_deg : int / float, optional
        Standard deviation (in degrees) of the yaw-channel Gaussian
        gate. Defaults to 45.0.
    sigma_pitch_deg : int / float, optional
        Standard deviation (in degrees) of the pitch-channel Gaussian
        gate. Defaults to 45.0.

    Returns:
    --------
    sei : np.ndarray
        A (n_frames,) array of SEI values in [0, 1].
    """

    obs_head = tracks[:, observer_idx, idx_head, :]
    obs_nose = tracks[:, observer_idx, idx_nose, :]
    obs_tti = tracks[:, observer_idx, idx_tti, :]
    target_point = tracks[:, observed_idx, observed_node_idx, :]

    # compute body length (Nose to TTI)
    body_length = np.linalg.norm(obs_nose - obs_tti, axis=1)
    body_length[body_length == 0] = np.nan

    # compute 3D distance between observer's nose and target point
    d_raw = np.linalg.norm(target_point - obs_nose, axis=1)
    d_norm = d_raw / (body_length + 1e-6)

    # egocentric (yaw, pitch) of the target as seen from the observer's head
    yaw_deg, pitch_deg = get_egocentric_direction(
        head_root=observer_head_root,
        head_pivot=obs_head,
        target_point=target_point,
    )
    yaw_rad = yaw_deg * np.pi / 180.0
    pitch_rad = pitch_deg * np.pi / 180.0
    sigma_yaw_rad = sigma_yaw_deg * np.pi / 180.0
    sigma_pitch_rad = sigma_pitch_deg * np.pi / 180.0

    with np.errstate(divide='ignore', invalid='ignore'):
        # separable Gaussian gate over the (yaw, pitch) error; in [0, 1],
        # peaks at 1 when the target is exactly on the observer's gaze axis
        gaze_score = np.exp(-(yaw_rad ** 2) / (2 * sigma_yaw_rad ** 2)) * np.exp(
            -(pitch_rad ** 2) / (2 * sigma_pitch_rad ** 2)
        )

        if v_max is None:
            v_max = np.nanpercentile(speed_arr, 99)
        v_norm = np.clip(speed_arr / (v_max + 1e-6), 0, 1)

        # Bounded exponent: `tanh(L/d_norm)` saturates at 1 as d_norm
        # -> 0, so `gamma` lies in [1, 2] for all distances. The legacy
        # form `gamma = 1 + 1/d_norm` was unbounded and produced
        # `gamma > 20` at common close-engagement distances (e.g.,
        # `d_norm = 0.05`, ~5% of a body length), which crushed the
        # Gaussian gate to ~0 unless `(yaw, pitch) = (0, 0)` exactly —
        # collapsing the SEI to zero precisely on the sniffing /
        # nose-to-nose frames it was meant to score highest.
        gamma = 1 + np.tanh(1 / (d_norm + 1e-6))

        # W_social: interpolator between speed-based pursuit and distance-based attention
        w_pursuit = (1 - np.exp(-d_norm)) * v_norm
        w_proximity = np.exp(-d_norm)
        w_social = w_pursuit + w_proximity

        sei = (gaze_score ** gamma) * w_social

    return sei

def calculate_tail_curvature(input_arr: np.ndarray) -> np.ndarray:
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

    The function returns a dimensionless "tail bendiness index" rather
    than a curvature in physical units. The inner step computes a true
    discrete curvature `kappa_i = ||t_{i+1} - t_i|| / s_i` (in 1/length
    units, where `t_i` is the unit tangent of segment i and `s_i` is the
    arc-length step between consecutive tangent midpoints, taken here as
    the segment length); this is the mathematically correct
    arc-length-normalized curvature. The per-frame mean of those
    `||kappa_i||` is then **rescaled by the per-frame mean tail segment
    length** so that the returned quantity is unitless. This rescale is
    appropriate because the tracked tail segments are not equal in
    length, so the absolute 1/length value depends on which intervals
    happen to be short in a given frame; multiplying back by the mean
    segment length cancels that unit-of-length dependence and produces a
    quantity that is comparable across frames and animals (and lands in
    roughly the [0, 1] range expected by the histogram boundaries in
    `FeatureZoo.feature_boundaries`). It can be interpreted as the
    average per-segment turning angle (in radians for small bends)
    weighted by how anisotropic the segment lengths are within a frame.

    Parameters
    ----------
    input_arr (np.ndarray)
         A (n_frames, n_nodes, 3) shape ndarray to compute tail curvature on.

    Returns
    -------
    average_tail_curvature (np.ndarray)
         A (n_frames, 1) shape ndarray containing the average tail curvature.
    """

    # raw inter-node segment vectors and their lengths (= arc-length step)
    segment_vectors = np.diff(input_arr, axis=1)
    segment_lengths = np.linalg.norm(segment_vectors, axis=2)

    # unit tangent vectors per segment
    tangent_vectors = segment_vectors / segment_lengths[..., np.newaxis]

    # estimate curvature as the change in unit tangent divided by the
    # arc-length step between consecutive segments (1/length units)
    curvature = (
        np.diff(tangent_vectors, axis=1)
        / segment_lengths[:, :-1, np.newaxis]
    )

    # per-frame mean curvature magnitude (1/length units)
    avg_curvature = np.mean(np.linalg.norm(curvature, axis=2), axis=1)

    # rescale by the per-frame mean segment length to produce a unitless
    # "tail bendiness index" (cancels the inner 1/length units and lands
    # the output in roughly [0, 1] for typical mouse postures)
    mean_segment_length = np.nanmean(segment_lengths, axis=1)
    avg_curvature = avg_curvature * mean_segment_length

    return np.reshape(avg_curvature, newshape=(avg_curvature.shape[0], 1))


def get_egocentric_direction(
    head_root: np.ndarray,
    head_pivot: np.ndarray,
    target_point: np.ndarray,
    spatial_resolution_tolerance: int | float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the per-frame egocentric direction (yaw, pitch) of a target
    point as seen from an observer's head pivot, expressed in the
    observer's anatomical head frame.

    The method consists of three steps. First, the world-frame vector
    `v = target_point - head_pivot` is built. Second, `v` is rotated
    into the observer's anatomical head frame by left-multiplying with
    `head_root` (the (n_frames, 3, 3) tensor produced by
    `get_head_root`, whose rows are the body-axes h_x, h_y, h_z written
    in world coordinates), giving `v_local = head_root @ v`. Third, the
    spherical coordinates of `v_local` in the head frame are extracted:

        yaw   = atan2(v_local.y, v_local.x)            in (-180, 180] deg
        pitch = atan2(v_local.z, sqrt(v_local.x^2 + v_local.y^2))
                                                       in (-90, 90] deg

    Geometric interpretation:
        - `yaw` is the signed left/right offset of the target from the
          observer's head-forward axis (h_x). yaw = 0 deg means the
          target sits in the observer's sagittal plane in the
          forward direction; positive yaw means the target is on the
          observer's left (toward h_y); +/-180 deg means the target is
          directly behind.
        - `pitch` is the signed elevation of the target above (positive)
          or below (negative) the observer's gaze axis. pitch = 0 deg
          means the target lies in the equatorial plane through the
          observer's gaze (the plane spanned by h_x and h_y); +90 deg
          means the target sits along the observer's dorsal axis (h_z).

    Together (yaw, pitch) parametrize every direction on the unit sphere
    centered at `head_pivot`, with the gimbal-lock degeneracy at
    pitch = +/-90 deg (target directly along h_z; yaw becomes
    coordinate-undefined). Because the underlying transformation is a
    rigid rotation (head_root is orthonormal by construction in
    get_head_root), no length or angular information is lost in the
    change of basis - this is the spherical decomposition of v in the
    observer's head frame, not a projection.

    Frames where `‖v‖ < spatial_resolution_tolerance` (target essentially
    coincident with the head pivot) are returned as NaN in both yaw and
    pitch, since the direction is geometrically undefined. Frames where
    any of `head_root`, `head_pivot`, or `target_point` contain NaN
    propagate NaN to the outputs.

    Parameters
    ----------
    head_root (np.ndarray)
        A (n_frames, 3, 3) shape ndarray of observer head rotation
        matrices, with rows (h_x, h_y, h_z) expressing the observer's
        anatomical body axes in world coordinates. Typically produced by
        `get_head_root`.
    head_pivot (np.ndarray)
        A (n_frames, 3) shape ndarray of observer head pivot point
        positions in world coordinates (usually the `Head` tracked
        point).
    target_point (np.ndarray)
        A (n_frames, 3) shape ndarray of target point positions in
        world coordinates.
    spatial_resolution_tolerance (int / float)
        Minimum acceptable norm of the inter-point vector
        (target_point - head_pivot), in the same units as the inputs
        (typically meters); frames below this length yield NaN outputs.
        Defaults to 0.001.

    Returns
    -------
    yaw_deg (np.ndarray)
        A (n_frames,) shape ndarray of signed yaw angles in
        (-180, 180] degrees.
    pitch_deg (np.ndarray)
        A (n_frames,) shape ndarray of signed pitch angles in
        [-90, 90] degrees.
    """

    v = target_point - head_pivot

    v_len = np.linalg.norm(v, axis=1).astype(np.float64)
    invalid = v_len < spatial_resolution_tolerance

    v_local = np.einsum("ijk,ik->ij", head_root, v)

    yaw_deg = np.arctan2(v_local[:, 1], v_local[:, 0]) * 180.0 / np.pi
    pitch_deg = (
        np.arctan2(
            v_local[:, 2],
            np.sqrt(v_local[:, 0] ** 2 + v_local[:, 1] ** 2),
        )
        * 180.0
        / np.pi
    )

    yaw_deg[invalid] = np.nan
    pitch_deg[invalid] = np.nan

    return yaw_deg, pitch_deg


def calculate_speed(
    tracked_points_array: np.ndarray,
    capture_framerate: int | float,
    smoothing_time_window: int | float,
) -> np.ndarray:
    """
    Returns arrays w/ centroid (body minus tail) speed data.

    On every frame the body centroid is computed as the per-frame
    `np.nanmean` of the supplied tracked points, so NaN-tracked nodes
    are silently excluded from the centroid (rather than poisoning it).
    The instantaneous speed is then the Euclidean norm of the
    frame-to-frame centroid displacement, divided by 1/capture_framerate
    and multiplied by 100 to convert units from m/frame into cm/s
    (assuming the input points are in meters).

    The raw 1D speed trace is smoothed with an `astropy` Gaussian1DKernel
    whose stddev is `floor(smoothing_time_window * capture_framerate)`
    samples, using `boundary='extend'`, `nan_treatment='interpolate'`,
    and `preserve_nan=True`; this means short stretches of NaN are
    interpolated through during convolution but the original NaN
    locations remain NaN in the output. The very first frame is
    returned as NaN (since the displacement at t=0 is undefined), so
    the output has the same length as the input along the time axis.

    Parameters
    ----------
    tracked_points_array (np.ndarray)
         A (n_frames, n_nodes, n_dimensions)
         shape ndarray of tracked points.
    capture_framerate (int / float)
        Recording camera framerate; defaults to None (fps).
    smoothing_time_window (int / float)
        Time window to perform smoothing over; defaults to None (s).

    Returns
    -------
    speeds (np.ndarray)
        A (n_frames) shape ndarray of centroid speed data.
    """

    speed_smoothing_kernel = Gaussian1DKernel(
        stddev=int(np.floor(smoothing_time_window * capture_framerate))
    )

    speed = np.zeros((tracked_points_array.shape[0], 1))

    mouse_centroid = np.nanmean(tracked_points_array, axis=1)
    frame_differential_centroid = mouse_centroid[1:, :] - mouse_centroid[:-1, :]
    euclidean_distance_centroid = (
        100
        * np.linalg.norm(frame_differential_centroid, axis=1)
        / (1 / capture_framerate)
    )
    speed_centroid = convolve(
        euclidean_distance_centroid,
        kernel=speed_smoothing_kernel,
        boundary="extend",
        nan_treatment="interpolate",
        preserve_nan=True,
    )
    speed_centroid = np.concatenate(
        (np.array([np.nan]), speed_centroid), dtype=np.float64
    )
    speed[:, 0] = speed_centroid

    return speed


def get_head_root(
    data_arr: np.ndarray,
    head_idx: int = 0,
    ear_r_idx: int = 1,
    ear_l_idx: int = 2,
    nose_idx: int = 3,
    spatial_resolution_tolerance: int | float = 0.001,
) -> np.ndarray:
    """
    Computes the per-frame head-root rotation matrices from anatomical landmarks.

    The body frame is built directly from the four head tracked points
    (Head, Ear_R, Ear_L, Nose) on every frame, without any cross-session
    reference template, SVD-of-reference, or Kabsch fit. Because the axes
    are anchored to anatomy, their chirality and labelling are stable
    across sessions and animals. This removes the need for the legacy
    "rotation_type" dispatch (regularXY / regularYX / roll_issue /
    pitch_issueYZ / pitch_issueZY) and the manual sign-correction logic
    that compensated for sign ambiguity of SVD right-singular vectors.

    Construction (per frame):
        h_x = (Nose - Head) / ||Nose - Head||
            anteroposterior (rostral-caudal) body axis, pointing forward
            (toward the nose).
        v_y_raw = (Ear_L - Ear_R)
            tentative inter-aural axis, pointing toward the left ear.
        h_z = (h_x x v_y_raw) / ||h_x x v_y_raw||
            dorsoventral body axis, pointing dorsally (upward when the
            head is upright). Computed as a cross product so its
            handedness is fixed by anatomy and is independent of how
            non-coplanar the four points are.
        h_y = h_z x h_x
            mediolateral body axis, exactly orthogonal to both h_x and
            h_z (Gram-Schmidt re-orthogonalization). Pointing toward the
            left ear.

    The returned (n_frames, 3, 3) tensor has rows (h_x, h_y, h_z). This
    is consistent with the convention expected by `get_euler_ang` (which
    decomposes the matrix as roll-pitch-yaw with rows being body axes
    expressed in world coordinates).

    Frames where any of the four head points is NaN, or where any of the
    three norms (forward axis, dorsal axis) falls below
    `spatial_resolution_tolerance`, are returned as NaN-filled rotation
    matrices so that downstream Euler-angle computation propagates NaN
    rather than silently producing degenerate frames.

    Parameters
    ----------
    data_arr (np.ndarray)
        A (n_frames, n_head_points, 3) shape ndarray of head point data.
        The point order along axis 1 must be [Head, Ear_R, Ear_L, Nose]
        unless the index parameters below are overridden.
    head_idx (int)
        Position of the Head point along axis 1; defaults to 0.
    ear_r_idx (int)
        Position of the right ear point along axis 1; defaults to 1.
    ear_l_idx (int)
        Position of the left ear point along axis 1; defaults to 2.
    nose_idx (int)
        Position of the Nose point along axis 1; defaults to 3.
    spatial_resolution_tolerance (int / float)
        Minimum acceptable norm of the rostral-caudal and dorsoventral
        axes (in the same units as `data_arr`, typically meters); axes
        below this length on a given frame yield a NaN rotation matrix.
        Defaults to 0.001.

    Returns
    -------
    global_heads_rot (np.ndarray)
        A (n_frames, 3, 3) shape ndarray of head rotation matrices.
        For each valid frame, the rows are (h_x, h_y, h_z) with h_x
        pointing forward (Nose direction), h_y pointing left, and h_z
        pointing dorsally.
    """

    n_frames = data_arr.shape[0]

    head_pt = data_arr[:, head_idx, :]
    ear_r_pt = data_arr[:, ear_r_idx, :]
    ear_l_pt = data_arr[:, ear_l_idx, :]
    nose_pt = data_arr[:, nose_idx, :]

    h_x_raw = nose_pt - head_pt
    h_x_len = np.linalg.norm(h_x_raw, axis=1).astype(np.float64)
    h_x_len[h_x_len < spatial_resolution_tolerance] = np.nan
    h_x = h_x_raw / h_x_len[:, np.newaxis]

    v_y_raw = ear_l_pt - ear_r_pt

    h_z_raw = np.cross(h_x, v_y_raw)
    h_z_len = np.linalg.norm(h_z_raw, axis=1).astype(np.float64)
    h_z_len[h_z_len < spatial_resolution_tolerance] = np.nan
    h_z = h_z_raw / h_z_len[:, np.newaxis]

    h_y = np.cross(h_z, h_x)

    global_heads_rot = np.zeros(shape=(n_frames, 3, 3), dtype=np.float64)
    global_heads_rot[:, 0, :] = h_x
    global_heads_rot[:, 1, :] = h_y
    global_heads_rot[:, 2, :] = h_z

    return global_heads_rot


def get_back_root(
    point_data_3d: np.ndarray,
    mouse_id: int = None,
    neck_point_pos: int = None,
    tti_point_pos: int = None,
    root_method: str = None,
    spatial_resolution_tolerance: int | float = 0.001,
) -> np.ndarray:
    """
    Computes the back-root rotation matrices.

    For every frame, the back x-axis is taken to be the (Neck - TTI)
    vector, optionally projected to the horizontal plane (by zeroing
    its z-component) before unit-normalization. Frames whose unit-
    normalized x-axis would have a length below
    `spatial_resolution_tolerance` (in the same units as the input,
    typically meters) are returned as NaN-filled rotation matrices so
    that ill-defined (e.g. neck and TTI coincident) frames cannot
    produce silently wrong rotations downstream.

    Three different conventions are supported via `root_method`, which
    differ in how the y- and z-axes are populated and whether the back
    is treated as a strictly planar (XY) frame or as a tilted frame
    that follows the back's pitch/roll:

    - "default": `x_dir = (Neck - TTI)` projected to XY and normalized;
      `z_dir` is the world up vector (0, 0, 1) tiled across frames; and
      `y_dir = z_dir x x_dir` (right-handed completion). The returned
      matrices have x_dir, y_dir, z_dir as **columns** (the result is
      transposed before return), so each frame's matrix maps body-frame
      vectors expressed as columns into world coordinates. Use this for
      yaw-only "ground-plane" rotations.

    - "root_inv": `x_dir = (Neck - TTI)` is **not** projected to XY (so
      it carries the back's pitch); `y_dir` is its 2D perpendicular in
      the XY plane (`(-x_dir.y, x_dir.x, 0)`) and `z_dir = x_dir x y_dir`,
      both unit-normalized. The returned matrices have x_dir, y_dir,
      z_dir as **rows**; this is the body-from-world rotation suitable
      for inverting back rotation off head data to obtain egocentric
      head angles.

    - any other value (the "else" branch): same as "default" for x_dir
      (projected to XY and normalized) but `y_dir` is built as the 2D
      perpendicular of `x_dir` in the XY plane and `z_dir` is forced to
      be the world up vector. The returned matrices are stacked with
      x_dir, y_dir, z_dir as **rows**. This is the "planar back" frame
      used to compute world-frame back pitch/yaw via `get_back_angles`.

    Parameters
    ----------
    point_data_3d (np.ndarray)
         A (n_frames, n_mice, n_nodes, 3) shape ndarray of tracked points.
    mouse_id (int)
         Index of mouse.
    neck_point_pos (int)
         Index of neck point in array.
    tti_point_pos (int)
         Index of TTI point in array.
    root_method (str)
         Rotation method to use.
    spatial_resolution_tolerance (int / float)
         The least necessary amount of
         distance between points (in meters).

    Returns
    -------
    back_roots (np.ndarray)
        A (n_frames, 3, 3) shape ndarray of rotation matrices.
    """

    back_roots = np.zeros((point_data_3d.shape[0], 3, 3), dtype=np.float64)
    back_roots[:] = np.nan

    if root_method == "default":
        x_dir = (
            point_data_3d[:, mouse_id, neck_point_pos, :]
            - point_data_3d[:, mouse_id, tti_point_pos, :]
        )
        x_dir[:, 2] = 0.0

        root_len = np.linalg.norm(x=x_dir, axis=1).astype(np.float64)
        root_len[root_len < spatial_resolution_tolerance] = np.nan
        x_dir = (x_dir / root_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(
            point_data_3d.shape[0], 3
        )
        z_dir = np.tile(A=np.array([0, 0, 1]), reps=(point_data_3d.shape[0], 1))
        y_dir = np.cross(z_dir, x_dir)

        back_roots = np.transpose(
            np.stack(arrays=[x_dir, y_dir, z_dir], axis=1), axes=(0, 2, 1)
        )

    elif root_method == "root_inv":
        x_dir = (
            point_data_3d[:, mouse_id, neck_point_pos, :]
            - point_data_3d[:, mouse_id, tti_point_pos, :]
        )
        root_len = np.linalg.norm(x=x_dir, axis=1).astype(np.float64)
        root_len[root_len < spatial_resolution_tolerance] = np.nan
        x_dir = (x_dir / root_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(
            point_data_3d.shape[0], 3
        )

        y_dir = np.zeros((point_data_3d.shape[0], 3))
        y_dir[:, 0] = -x_dir[:, 1]
        y_dir[:, 1] = x_dir[:, 0]
        y_dir[:, 2] = 0.0
        y_len = np.linalg.norm(x=y_dir, axis=1).astype(np.float64)
        y_dir = (y_dir / y_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(
            point_data_3d.shape[0], 3
        )

        z_dir = np.cross(x_dir, y_dir)
        z_len = np.linalg.norm(x=z_dir, axis=1).astype(np.float64)
        z_dir = (z_dir / z_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(
            point_data_3d.shape[0], 3
        )

        back_roots = np.stack(arrays=[x_dir, y_dir, z_dir], axis=1)

    else:
        x_dir = (
            point_data_3d[:, mouse_id, neck_point_pos, :]
            - point_data_3d[:, mouse_id, tti_point_pos, :]
        )
        x_dir[:, 2] = 0.0

        root_len = np.linalg.norm(x=x_dir, axis=1).astype(np.float64)
        root_len[root_len < spatial_resolution_tolerance] = np.nan
        x_dir = (x_dir / root_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(
            point_data_3d.shape[0], 3
        )

        y_dir = np.zeros((point_data_3d.shape[0], 3))
        y_dir[:, 0] = -x_dir[:, 1]
        y_dir[:, 1] = x_dir[:, 0]
        y_dir[:, 2] = 0.0
        y_len = np.linalg.norm(x=y_dir, axis=1).astype(np.float64)
        y_dir = (y_dir / y_len.reshape(point_data_3d.shape[0], 1)[None, :]).reshape(
            point_data_3d.shape[0], 3
        )

        z_dir = np.tile(A=np.array([0, 0, 1]), reps=(point_data_3d.shape[0], 1))

        back_roots = np.stack([x_dir, y_dir, z_dir], axis=1)

    return back_roots


def get_euler_ang(rot_matrix: np.ndarray) -> np.ndarray:
    """
    Computes Euler angles.
    NB: always in the order roll, pitch, yaw!

    Decomposes each per-frame 3x3 rotation matrix into intrinsic
    Tait-Bryan angles using the convention where the rows of the input
    matrix are the body axes (h_x, h_y, h_z) expressed in world
    coordinates (i.e. the matrix is `R_body_from_world`). For the
    well-conditioned (non-gimbal-locked) case the angles are recovered as

        roll  =  atan2( R[1, 2],  R[2, 2] )
        pitch =  atan2( R[0, 2],  sqrt(R[1, 2]**2 + R[2, 2]**2) )
        yaw   =  atan2(-R[0, 1],  R[0, 0] )

    and converted from radians to degrees. Frames where the gimbal-lock
    indicator `sqrt(R[1, 2]**2 + R[2, 2]**2)` falls below 1e-4
    (i.e. pitch is within ~0.006 deg of +/-90 deg) are flagged as
    problematic; for those frames roll is set to 0 deg (the residual
    rotation is fully described by yaw at the singularity) and pitch
    and yaw are recovered from `R[0, 2]`, `R[1, 0]`, and `R[1, 1]`.
    NaN entries in the input are propagated to the output.

    Parameters
    ----------
    rot_matrix (np.ndarray)
        A (n_frames, 3, 3) shape ndarray of rotation matrices.

    Returns
    -------
    desired_euler_angles (np.ndarray)
        A (n_frames, 3) shape ndarray of Euler angles
        (in units of degrees); column are in order:
        roll, pitch, yaw.
    """

    rot_matrix_reshaped = np.reshape(rot_matrix, newshape=(rot_matrix.shape[0], 9)).copy()

    temp = np.sqrt(
        (rot_matrix_reshaped[:, 8] * rot_matrix_reshaped[:, 8])
        + (rot_matrix_reshaped[:, 5] * rot_matrix_reshaped[:, 5])
    )
    problematic_indices = np.where(temp < 0.0001)[0]
    good_indices = np.setdiff1d(
        np.arange(0, rot_matrix.shape[0], 1), problematic_indices
    )

    desired_euler_angles = np.zeros((rot_matrix.shape[0], 3))

    desired_euler_angles[good_indices, 0] = (
        -np.arctan2(
            -rot_matrix_reshaped[good_indices, 5], rot_matrix_reshaped[good_indices, 8]
        )
        * 180.0
        / np.pi
    )
    desired_euler_angles[good_indices, 1] = (
        np.arctan2(rot_matrix_reshaped[good_indices, 2], temp[good_indices])
        * 180.0
        / np.pi
    )
    desired_euler_angles[good_indices, 2] = (
        np.arctan2(
            -rot_matrix_reshaped[good_indices, 1], rot_matrix_reshaped[good_indices, 0]
        )
        * 180.0
        / np.pi
    )

    desired_euler_angles[problematic_indices, 0] = 0.0
    desired_euler_angles[problematic_indices, 1] = (
        np.arctan2(
            rot_matrix_reshaped[problematic_indices, 2], temp[problematic_indices]
        )
        * 180.0
        / np.pi
    )
    desired_euler_angles[problematic_indices, 2] = (
        np.arctan2(
            rot_matrix_reshaped[problematic_indices, 3],
            rot_matrix_reshaped[problematic_indices, 4],
        )
        * 180.0
        / np.pi
    )

    return desired_euler_angles


def get_back_angles(back_directions: np.ndarray) -> np.ndarray:
    """
    Computes Euler angles for the back:
        pitch: angle between the back and the horizontal plane
        yaw: angle between the back and the z-axis

    Parameters
    ----------
    back_directions (np.ndarray)
        A (n_frames, 3) shape ndarray of back direction data
        Array of back direction data.

    Returns
    -------
    back_euler_angles (np.ndarray)
        A (n_frames, 2) shape ndarray of
        back pitch and back yaw angles (in that order).
    """

    back_euler_angles = np.zeros((back_directions.shape[0], 2))

    def get_rotation(argument_ang):
        """
        Applies an intrinsic Rx -> Ry -> Rz rotation to back_directions.

        Bounds pitch (argument_ang[0]) and roll (argument_ang[1]) to ±pi/2
        to avoid gimbal-style flipping; yaw (argument_ang[2]) is unconstrained.

        Parameters
        ----------
        argument_ang (array-like of float)
            A length-3 vector (pitch, roll, yaw) in radians.

        Returns
        -------
        (tuple)
            (new_vector, rotator) — rotated back_directions as a
            (n_frames, 3) np.ndarray and the corresponding (3, 3)
            rotation matrix. If the pitch or roll bound is exceeded,
            returns ([-1], [-1]) as a sentinel.
        """

        # rotations about z (yaw) are ok, but bounds on the other two to avoid flipping
        if abs(argument_ang[0]) > np.pi * 0.5 or abs(argument_ang[1]) > np.pi * 0.5:
            return [-1], [-1]
        rotate_z = np.array(
            [
                [np.cos(argument_ang[2]), -np.sin(argument_ang[2]), 0],
                [np.sin(argument_ang[2]), np.cos(argument_ang[2]), 0],
                [0, 0, 1],
            ]
        )
        rotate_y = np.array(
            [
                [np.cos(argument_ang[1]), 0, np.sin(argument_ang[1])],
                [0, 1, 0],
                [-np.sin(argument_ang[1]), 0, np.cos(argument_ang[1])],
            ]
        )
        rotate_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(argument_ang[0]), -np.sin(argument_ang[0])],
                [0, np.sin(argument_ang[0]), np.cos(argument_ang[0])],
            ]
        )
        rotator = np.dot(rotate_x, np.dot(rotate_y, rotate_z))
        new_vector = np.einsum("ij,kj->ki", rotator, back_directions)
        return new_vector, rotator

    def distance_to_x_axis(argument_ang):
        """
        Objective used by the Nelder-Mead optimizer. Rotates the back
        directions by (0, roll, yaw), takes the mean direction, and returns
        the unsigned angle between that mean direction and the x-axis.

        Parameters
        ----------
        argument_ang (array-like of float)
            A length-2 vector (roll, yaw) in radians.

        Returns
        -------
        (float)
            Absolute angle (radians) between the mean rotated back direction
            and the x-axis. Minimized to find the canonical back frame.
        """

        # rotate around yaw and roll to center around zero
        rot_check, rot_m = get_rotation([0.0, argument_ang[0], argument_ang[1]])
        check_vec = np.array(
            [
                np.nanmean(rot_check[:, 0]),
                np.nanmean(rot_check[:, 1]),
                np.nanmean(rot_check[:, 2]),
            ]
        )
        check_vec = check_vec / np.sqrt(sum(check_vec**2))
        angle = np.arctan2(
            np.linalg.norm(np.cross(check_vec, np.array([1, 0, 0]))),
            np.dot(check_vec, np.array([1, 0, 0])),
        )
        return abs(angle)

    # use a locally seeded RNG so the Nelder-Mead initial guess is bit-exactly
    # reproducible across runs while keeping a non-trivial start in (-pi/4, pi/4)
    _rng = np.random.default_rng(seed=0)
    res = minimize(
        distance_to_x_axis,
        x0=np.array(
            [
                -0.25 * np.pi + _rng.random() * 0.5 * np.pi,
                -0.25 * np.pi + _rng.random() * 0.5 * np.pi,
            ]
        ),
        method="nelder-mead",
        options={"xatol": 1e-6, "disp": False},
    )

    temp_angles_back = res.x
    rotated_back_directions, back_rotator = get_rotation(
        [0.0, temp_angles_back[0], temp_angles_back[1]]
    )

    # both arctan2 calls below are invariant under scaling of all components
    # by the same positive scalar, so a Frobenius-norm rescale would be a
    # no-op; the per-frame components are used directly.
    back_euler_angles[:, 0] = (
        np.arctan2(
            rotated_back_directions[:, 2],
            np.sqrt(
                rotated_back_directions[:, 0] ** 2 + rotated_back_directions[:, 1] ** 2
            ),
        )
        * 180.0
        / np.pi
    )
    back_euler_angles[:, 1] = (
        -np.arctan2(rotated_back_directions[:, 1], rotated_back_directions[:, 0])
        * 180.0
        / np.pi
    )

    return back_euler_angles


class FeatureZoo:
    feature_boundaries = {
        "speed": [0, 54],
        "acceleration": [-180, 180],
        "neck_elevation": [0, 12],
        "neck_elevation_1st_der": [-18, 18],
        "neck_elevation_2nd_der": [-90, 90],
        "allo_roll": [-180, 180],
        "allo_roll_1st_der": [-480, 480],
        "allo_roll_2nd_der": [-4500, 4500],
        "allo_pitch": [-90, 90],
        "allo_pitch_1st_der": [-480, 480],
        "allo_pitch_2nd_der": [-4500, 4500],
        "allo_yaw": [-180, 180],
        "allo_yaw_1st_der": [-480, 480],
        "allo_yaw_2nd_der": [-4500, 4500],
        "ego_yaw": [-180, 180],
        "ego_yaw_1st_der": [-480, 480],
        "ego_yaw_2nd_der": [-4500, 4500],
        "back_pitch": [-54, 54],
        "back_pitch_1st_der": [-90, 90],
        "back_pitch_2nd_der": [-720, 720],
        "back_yaw": [-36, 36],
        "back_yaw_1st_der": [-90, 90],
        "back_yaw_2nd_der": [-720, 720],
        "body_dir": [-180, 180],
        "body_dir_1st_der": [-480, 480],
        "body_dir_2nd_der": [-4500, 4500],
        "tail_curvature": [0, 1],
        "tail_curvature_1st_der": [-1.8, 1.8],
        "tail_curvature_2nd_der": [-18, 18],
        "nose-nose": [0, 90],
        "nose-nose_1st_der": [-54, 54],
        "nose-nose_2nd_der": [-240, 240],
        "TTI-TTI": [0, 90],
        "TTI-TTI_1st_der": [-54, 54],
        "TTI-TTI_2nd_der": [-240, 240],
        "nose-TTI": [0, 90],
        "nose-TTI_1st_der": [-54, 54],
        "nose-TTI_2nd_der": [-240, 240],
        "TTI-nose": [0, 90],
        "TTI-nose_1st_der": [-54, 54],
        "TTI-nose_2nd_der": [-240, 240],
        "allo_yaw-nose": [-180, 180],
        "allo_yaw-nose_1st_der": [-480, 480],
        "allo_yaw-nose_2nd_der": [-4500, 4500],
        "nose-allo_yaw": [-180, 180],
        "nose-allo_yaw_1st_der": [-480, 480],
        "nose-allo_yaw_2nd_der": [-4500, 4500],
        "allo_yaw-TTI": [-180, 180],
        "allo_yaw-TTI_1st_der": [-480, 480],
        "allo_yaw-TTI_2nd_der": [-4500, 4500],
        "TTI-allo_yaw": [-180, 180],
        "TTI-allo_yaw_1st_der": [-480, 480],
        "TTI-allo_yaw_2nd_der": [-4500, 4500],
        "allo_pitch-nose": [-90, 90],
        "allo_pitch-nose_1st_der": [-480, 480],
        "allo_pitch-nose_2nd_der": [-4500, 4500],
        "nose-allo_pitch": [-90, 90],
        "nose-allo_pitch_1st_der": [-480, 480],
        "nose-allo_pitch_2nd_der": [-4500, 4500],
        "allo_pitch-TTI": [-90, 90],
        "allo_pitch-TTI_1st_der": [-480, 480],
        "allo_pitch-TTI_2nd_der": [-4500, 4500],
        "TTI-allo_pitch": [-90, 90],
        "TTI-allo_pitch_1st_der": [-480, 480],
        "TTI-allo_pitch_2nd_der": [-4500, 4500],
        "orofacial-sei": [0, 1],
        "orofacial-sei_1st_der": [-6, 6],
        "orofacial-sei_2nd_der": [-36, 36],
        "anogenital-sei": [0, 1],
        "anogenital-sei_1st_der": [-6, 6],
        "anogenital-sei_2nd_der": [-36, 36]
    }

    feature_labels = {
        "individual": {
            "speed": "slow -- (cm/s) -- fast",
            "acceleration": "slow -- (cm/s²) -- fast",
            "neck_elevation": "down -- (cm) -- up",
            "neck_elevation_1st_der": "down -- (cm/s) -- up",
            "neck_elevation_2nd_der": "down -- (cm/s²) -- up",
            "allo_roll": "ccw -- (°) -- cw",
            "allo_roll_1st_der": "ccw -- (°/s) -- cw",
            "allo_roll_2nd_der": "ccw -- (°/s²) -- cw",
            "allo_pitch": "down -- (°) -- up",
            "allo_pitch_1st_der": "down -- (°/s) -- up",
            "allo_pitch_2nd_der": "down -- (°/s²) -- up",
            "allo_yaw": "ccw -- (°) -- cw",
            "allo_yaw_1st_der": "ccw -- (°/s) -- cw",
            "allo_yaw_2nd_der": "ccw -- (°/s²) -- cw",
            "ego_yaw": "ccw -- (°) -- cw",
            "ego_yaw_1st_der": "ccw -- (°/s) -- cw",
            "ego_yaw_2nd_der": "ccw -- (°/s²) -- cw",
            "back_pitch": "down -- (°) -- up",
            "back_pitch_1st_der": "down -- (°/s) -- up",
            "back_pitch_2nd_der": "down -- (°/s²) -- up",
            "back_yaw": "ccw -- (°) -- cw",
            "back_yaw_1st_der": "ccw -- (°/s) -- cw",
            "back_yaw_2nd_der": "ccw -- (°/s²) -- cw",
            "body_dir": "ccw -- (°) -- cw",
            "body_dir_1st_der": "ccw -- (°/s) -- cw",
            "body_dir_2nd_der": "ccw -- (°/s²) -- cw",
            "tail_curvature": "straight -- (a.u.) -- curved",
            "tail_curvature_1st_der": "straight -- (a.u.) -- curved",
            "tail_curvature_2nd_der": "straight -- (a.u.) -- curved",
        },
        "social": {
            "nose-nose": "near -- (cm) -- far",
            "nose-nose_1st_der": "near -- (cm/s) -- far",
            "nose-nose_2nd_der": "near -- (cm/s²) -- far",
            "TTI-TTI": "near -- (cm) -- far",
            "TTI-TTI_1st_der": "near -- (cm/s) -- far",
            "TTI-TTI_2nd_der": "near -- (cm/s²) -- far",
            "nose-TTI": "near -- (cm) -- far",
            "nose-TTI_1st_der": "near -- (cm/s) -- far",
            "nose-TTI_2nd_der": "near -- (cm/s²) -- far",
            "TTI-nose": "near -- (cm) -- far",
            "TTI-nose_1st_der": "near -- (cm/s) -- far",
            "TTI-nose_2nd_der": "near -- (cm/s²) -- far",
            "allo_yaw-nose": "ccw -- (°) -- cw",
            "allo_yaw-nose_1st_der": "ccw -- (°/s) -- cw",
            "allo_yaw-nose_2nd_der": "ccw -- (°/s²) -- cw",
            "nose-allo_yaw": "ccw -- (°) -- cw",
            "nose-allo_yaw_1st_der": "ccw -- (°/s) -- cw",
            "nose-allo_yaw_2nd_der": "ccw -- (°/s²) -- cw",
            "allo_yaw-TTI": "ccw -- (°) -- cw",
            "allo_yaw-TTI_1st_der": "ccw -- (°/s) -- cw",
            "allo_yaw-TTI_2nd_der": "ccw -- (°/s²) -- cw",
            "TTI-allo_yaw": "ccw -- (°) -- cw",
            "TTI-allo_yaw_1st_der": "ccw -- (°/s) -- cw",
            "TTI-allo_yaw_2nd_der": "ccw -- (°/s²) -- cw",
            "allo_pitch-nose": "down -- (°) -- up",
            "allo_pitch-nose_1st_der": "down -- (°/s) -- up",
            "allo_pitch-nose_2nd_der": "down -- (°/s²) -- up",
            "nose-allo_pitch": "down -- (°) -- up",
            "nose-allo_pitch_1st_der": "down -- (°/s) -- up",
            "nose-allo_pitch_2nd_der": "down -- (°/s²) -- up",
            "allo_pitch-TTI": "down -- (°) -- up",
            "allo_pitch-TTI_1st_der": "down -- (°/s) -- up",
            "allo_pitch-TTI_2nd_der": "down -- (°/s²) -- up",
            "TTI-allo_pitch": "down -- (°) -- up",
            "TTI-allo_pitch_1st_der": "down -- (°/s) -- up",
            "TTI-allo_pitch_2nd_der": "down -- (°/s²) -- up",
            "orofacial-sei": "asocial -- (a.u.) -- engaged",
            "orofacial-sei_1st_der": "asocial -- (a.u./s) -- engaged",
            "orofacial-sei_2nd_der": "asocial -- (a.u./s²) -- engaged",
            "anogenital-sei": "asocial -- (a.u.) -- engaged",
            "anogenital-sei_1st_der": "asocial -- (a.u./s) -- engaged",
            "anogenital-sei_2nd_der": "asocial -- (a.u./s²) -- engaged"
        },
    }

    def __init__(self, **kwargs):
        """
        Initializes the FeatureZoo class.

        All keyword arguments are captured into `self.__dict__` verbatim,
        so the instance exposes every supplied kwarg as an attribute (no
        whitelisting). The visualizations parameter dictionary is loaded
        from `_parameter_settings/visualizations_settings.json` (relative
        to the package root) and stored in
        `self.visualizations_parameter_dict`. The boolean
        `self.app_context_bool` records whether the current process is
        running inside the GUI context; downstream methods use it to
        decide whether to call `smart_wait` interactively or in
        headless mode.

        The kwargs documented below are the ones the rest of the class
        actually consumes; other kwargs are accepted for forward
        compatibility but are not used here.

        Parameters
        ----------
        root_directory (str)
            Root directory for data; defaults to None.
        neuronal_tuning_figures_dict (dict)
            Analyzes parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        with open(
            pathlib.Path(__file__).parent.parent
            / "_parameter_settings/visualizations_settings.json"
        ) as json_file:
            self.visualizations_parameter_dict = json.load(json_file)

        self.app_context_bool = is_gui_context()

    def plot_feature_distributions(self, **kwargs):
        """
        Plots histograms of all available behavioral features.

        Iterates over the feature distribution dictionary produced by
        `save_behavioral_features_to_file` and renders each feature's
        occupancy histogram (or, for spatial features, its 2D occupancy
        map) on a multi-panel grid laid out by `gridspec`. Per-mouse
        traces are colored according to the experiment-specific palette
        chosen by `choose_animal_colors`, and 2D spatial maps are
        rendered with the colormap returned by `create_colormap` so
        that maps from different mice are visually distinguishable
        within a session. Per-feature x-axis tick labels are taken from
        the class-level `feature_boundaries` dict, and per-feature x-axis
        titles from the class-level `feature_labels` dict; both are
        keyed by the suffix that follows the mouse-id prefix in the
        feature column name. The full grid is written as a multi-page
        PDF at the path supplied via `plot_file_name`. This method is
        called by `save_behavioral_features_to_file` after the
        distributions are computed; calling it directly is also
        supported for ad-hoc replotting from a precomputed
        `feature_dict`.

        Parameters
        ----------
        feature_dict (dict)
            Mapping from feature column name to a dict with keys
            "occ_array", "bin_centers", and "bin_edges" (as produced by
            `generate_feature_distributions`); must be supplied for any
            output to be produced.
        mouse_id_list (list of str)
            Track names of the mice in the recording, used to colorize
            traces and label panels.
        session_exp_code (str)
            Experimental code string (used by `extract_information` to
            pick the per-experiment animal palette).
        plot_file_name (str)
            Path for the output PDF.

        Returns
        -------
        behavioral_feature_distributions (.pdf file)
           Plot of all feature histograms.
        """

        feature_dict = (
            kwargs["feature_dict"]
            if "feature_dict" in kwargs and isinstance(kwargs["feature_dict"], dict)
            else None
        )
        mouse_id_list = (
            kwargs["mouse_id_list"]
            if "mouse_id_list" in kwargs and isinstance(kwargs["mouse_id_list"], list)
            else None
        )
        session_exp_code = (
            kwargs["session_exp_code"]
            if "session_exp_code" in kwargs
            and isinstance(kwargs["session_exp_code"], str)
            else None
        )
        plot_file_name = (
            kwargs["plot_file_name"]
            if "plot_file_name" in kwargs and isinstance(kwargs["plot_file_name"], str)
            else None
        )

        # get colors
        experiment_info_dict = extract_information(experiment_code=session_exp_code)
        mouse_colors = choose_animal_colors(
            exp_info_dict=experiment_info_dict,
            visualizations_parameter_dict=self.visualizations_parameter_dict,
        )

        if feature_dict is not None and mouse_id_list is not None:
            mouse_color_dict = {"social": "#000000"}
            mouse_colormap_dict = {}
            for mouse_idx, mouse in enumerate(mouse_id_list):
                mouse_color_dict[mouse] = mouse_colors[mouse_idx]
                mouse_colormap_dict[mouse] = create_colormap(
                    input_parameter_dict={
                        "cm_length": 255,
                        "cm_name": f"{mouse}",
                        "cm_type": "sequential",
                        "cm_start": (
                            int(mouse_colors[mouse_idx][1:3], 16),
                            int(mouse_colors[mouse_idx][3:5], 16),
                            int(mouse_colors[mouse_idx][5:7], 16),
                        ),
                        "cm_end": (255, 255, 255),
                        "equalize_luminance": True,
                        "match_luminance_by": "max",
                        "change_saturation": 0.5,
                        "cm_opacity": 1,
                    }
                )

            plot_features = {}
            for feature_key in feature_dict.keys():
                mouse_id = feature_key.split(".")[0]
                if (
                    f"individual.{mouse_id}" not in plot_features
                    and "-" not in mouse_id
                ):
                    plot_features[f"individual.{mouse_id}"] = []

                if "-" not in mouse_id:
                    plot_features[f"individual.{mouse_id}"].append(feature_key)
                else:
                    if "social" not in plot_features:
                        plot_features["social"] = []
                    plot_features["social"].append(feature_key)

            with PdfPages(plot_file_name) as pdf_fig:
                for plot_feature_key in plot_features:
                    if "social" in plot_feature_key:
                        histogram_color = mouse_color_dict["social"]
                        mouse_colormap = 0
                    else:
                        histogram_color = mouse_color_dict[
                            plot_feature_key.split(".")[-1]
                        ]
                        mouse_colormap = mouse_colormap_dict[
                            plot_feature_key.split(".")[-1]
                        ]

                    with warnings.catch_warnings():
                        warnings.simplefilter(action="ignore")

                        row_num = int(
                            np.ceil((len(plot_features[plot_feature_key])) / 6)
                        )
                        fig = plt.figure(
                            figsize=(6.4, float(row_num)), dpi=600, tight_layout=True
                        )
                        fig.suptitle(t=f"{plot_feature_key}", fontsize=8)
                        gs = gridspec.GridSpec(nrows=row_num, ncols=6)

                        gs_x = 0
                        gs_y = 0
                        for feature_idx, feature in enumerate(
                            plot_features[plot_feature_key]
                        ):
                            if "space" in feature:
                                cbar_width = 0.005
                                cbar_height = 0.04
                                cbar_ypos_extra = 0.11

                                ax = fig.add_subplot(gs[gs_x, gs_y])
                                occ = ax.imshow(
                                    X=feature_dict[feature]["occ_array"][:, :],
                                    cmap=mouse_colormap,
                                    vmin=0,
                                    interpolation="gaussian",
                                    aspect="equal",
                                )
                                ax.set_title(
                                    label="Spatial occupancy (smoothed)",
                                    fontsize=4,
                                    pad=4,
                                )
                                ax.set_xticks([])
                                ax.set_xlabel(xlabel="X (cm)", fontsize=4, labelpad=1)
                                ax.set_yticks([])
                                ax.set_ylabel(ylabel="Y (cm)", fontsize=4, labelpad=1)

                                ax_position = ax.get_position()
                                cb_ax = fig.add_axes(
                                    (
                                        ax_position.x0 + 0.03,
                                        ax_position.y0 + cbar_ypos_extra,
                                        cbar_width,
                                        cbar_height,
                                    )
                                )
                                cbar = fig.colorbar(
                                    mappable=occ, orientation="vertical", cax=cb_ax
                                )
                                cbar_vmin, cbar_vmax = cbar.mappable.get_clim()
                                cbar.set_ticks([cbar_vmin, cbar_vmax])
                                cbar.set_ticklabels(
                                    ticklabels=[
                                        f"{int(cbar_vmin)}",
                                        f"{int(np.ceil(cbar_vmax))}",
                                    ],
                                    fontsize=2,
                                )
                                cbar.ax.tick_params(
                                    axis="both", which="both", length=0, pad=0.5
                                )
                                cbar.outline.set_visible(True)

                                gs_y += 1
                                if gs_y > 5:
                                    gs_y = 0
                                    gs_x += 1
                            else:
                                ax = fig.add_subplot(gs[gs_x, gs_y])
                                ax.tick_params(
                                    axis="both", which="both", length=1.5, pad=0.25
                                )
                                ax.bar(
                                    x=feature_dict[feature]["bin_centers"],
                                    height=feature_dict[feature]["occ_array"],
                                    width=feature_dict[feature]["bin_edges"][1]
                                    - feature_dict[feature]["bin_edges"][0],
                                    align="center",
                                    color=histogram_color,
                                    ec="#000000",
                                    alpha=0.75,
                                    lw=0.1,
                                )
                                if '-sei' not in feature.split('.')[-1]:
                                    ax.set_title(
                                        label=f"{feature.split('.')[-1]}", fontsize=4, pad=1
                                    )
                                else:
                                    ax.set_title(
                                        label=f"{feature}", fontsize=3, pad=1
                                    )

                                ax.set_xticks(
                                    ticks=[
                                        self.feature_boundaries[feature.split(".")[-1]][
                                            0
                                        ],
                                        self.feature_boundaries[feature.split(".")[-1]][
                                            1
                                        ],
                                    ],
                                    labels=[
                                        f"{self.feature_boundaries[feature.split('.')[-1]][0]:.1f}",
                                        f"{self.feature_boundaries[feature.split('.')[-1]][1]:.1f}",
                                    ],
                                    rotation=0,
                                    fontsize=2,
                                )
                                ax.set_xlabel(
                                    xlabel=f"{self.feature_labels[plot_feature_key.split('.')[0]][feature.split('.')[-1]]}",
                                    fontsize=3,
                                    labelpad=1,
                                )
                                temp_ymin, temp_ymax = ax.get_ylim()
                                ax.set_yticks(
                                    ticks=[0, int(np.ceil(temp_ymax)) - 10],
                                    labels=["0", f"{int(np.ceil(temp_ymax)) - 10}"],
                                    rotation=0,
                                    fontsize=2,
                                )
                                ax.set_ylabel(
                                    ylabel="Occupancy (s)", fontsize=3, labelpad=1
                                )
                                ax.set_box_aspect(1)

                                gs_y += 1
                                if gs_y > 5:
                                    gs_y = 0
                                    gs_x += 1

                        pdf_fig.savefig(dpi=600)
                        plt.clf()
                        plt.close("all")

    def save_behavioral_features_to_file(self):
        """
        Computes and saves behavioral features to file.

        Top-level driver for the per-session feature pipeline. Locates
        the translated/rotated 3D points file
        (`*_points3d_translated_rotated_metric.h5`) under
        `self.root_directory / 'video'`, loads the (n_frames, n_mice,
        n_nodes, 3) tracks together with the node names, track names,
        experimental code, and the empirical camera frame rate, and then
        for each mouse computes:

            - head position (cm) and centroid speed/acceleration (cm/s,
              cm/s^2) using `calculate_speed` and `calculate_derivatives`,
              both excluding the four distal tail points from the
              centroid;
            - neck elevation in cm and its derivatives (frames with
              negative neck-z values are masked to NaN);
            - allocentric head Euler angles (roll, pitch, yaw, in
              degrees) via `get_head_root` + `get_euler_ang`, and their
              first/second derivatives;
            - egocentric head yaw (head yaw expressed in the back's
              ground-plane frame) via `get_back_root(root_method="other")`
              and matrix composition;
            - back pitch and yaw (and derivatives) via
              `get_back_root(root_method="root_inv")` followed by
              `get_back_angles`;
            - body direction (ground-plane back yaw) via
              `get_back_root(root_method="default")` plus an `arctan2`,
              with derivatives;
            - tail curvature via `calculate_tail_curvature`, with
              derivatives.

        For every ordered pair of mice the function then computes the
        social distances (nose-nose, TTI-TTI, nose-TTI, TTI-nose) and
        the egocentric social angles via `get_egocentric_direction`,
        which expresses each target body point (Nose / TTI of the
        partner) in the observer's anatomical head frame and returns
        signed yaw (left/right of gaze axis) and pitch (above/below
        gaze axis) per frame. Both directions of every (observer,
        target_node) pair are stored, yielding eight angle quartets:
        allo_yaw-nose, nose-allo_yaw, allo_yaw-TTI, TTI-allo_yaw plus
        the matching allo_pitch-nose, nose-allo_pitch, allo_pitch-TTI,
        TTI-allo_pitch. The orofacial and anogenital social engagement
        indices are computed via `calculate_sei`, which now uses the
        same egocentric (yaw, pitch) decomposition for its gaze gate.

        The full table is materialized as a `polars.DataFrame`,
        feature-by-feature feature distributions are computed via
        `generate_feature_distributions` (using the per-feature ranges
        from `self.feature_boundaries`), the distribution PDF is
        rendered by `plot_feature_distributions`, and the assembled
        table is written next to the input file as
        `*_behavioral_features.csv`.

        The class instance must be configured with `root_directory`,
        `behavioral_parameters_dict` (with keys "head_points",
        "tail_points", "back_root_points", "derivative_bins"),
        `message_output` (a callable used for status messages), and
        `app_context_bool` (set automatically by `__init__`).

        Parameters
        ----------
        ----------

        Returns
        -------
        behavioral_features_csv (.csv file)
           Data sheet w/ behavioral features.
        """

        self.message_output(
            f"Computing behavioral features started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        tracked_file_loc = first_match_or_raise(
            root=pathlib.Path(self.root_directory) / 'video',
            pattern='[!speaker]*_points3d_translated_rotated_metric.h5',
            recursive=True,
            label="translated/rotated mouse points3d .h5",
        )

        # load tracking data
        with h5py.File(tracked_file_loc, mode="r") as tracking_data_3d:
            mouse_data = np.array(tracking_data_3d["tracks"]).astype(np.float64)
            mouse_nodes = [
                elem.decode("utf-8") for elem in tracking_data_3d["node_names"]
            ]
            track_names = [
                elem.decode("utf-8") for elem in tracking_data_3d["track_names"]
            ]
            experimental_code = tracking_data_3d["experimental_code"][()].decode(
                "utf-8"
            )
            empirical_camera_sr = float(tracking_data_3d["recording_frame_rate"][()])

        self.message_output(
            f"Working on tracking data of shape {mouse_data.shape} with experiment code '{experimental_code}' ({empirical_camera_sr} fps), track names {track_names} \n"
            f"and nodes {mouse_nodes}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        # # # compute individual features
        head_position = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        speed = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        speed_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        speed_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        neck_elevation = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        neck_elevation_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        neck_elevation_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        global_head_roots = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3, 3)
        )
        global_head_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_head_angles_1st_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3)
        )
        global_head_angles_2nd_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3)
        )
        ego_head_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        ego_head_angles_1st_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3)
        )
        ego_head_angles_2nd_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3)
        )
        global_back_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 2))
        global_back_angles_1st_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 2)
        )
        global_back_angles_2nd_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 2)
        )
        global_root_angles = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 3))
        global_root_angles_1st_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3)
        )
        global_root_angles_2nd_der = np.zeros(
            (mouse_data.shape[1], mouse_data.shape[0], 3)
        )
        tail_curvature = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        tail_curvature_1st_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))
        tail_curvature_2nd_der = np.zeros((mouse_data.shape[1], mouse_data.shape[0], 1))

        for mouse_num in range(mouse_data.shape[1]):
            # # head position (in cm)
            head_position[mouse_num, :, :] = (
                    mouse_data[:, mouse_num, mouse_nodes.index("Head"), :] * 100
            )

            # # speed (cm/s) and acceleration (cm/s^2)
            exclude_tail_points = [
                mouse_nodes.index(self.behavioral_parameters_dict["tail_points"][1]),
                mouse_nodes.index(self.behavioral_parameters_dict["tail_points"][2]),
                mouse_nodes.index(self.behavioral_parameters_dict["tail_points"][3]),
                mouse_nodes.index(self.behavioral_parameters_dict["tail_points"][4]),
            ]
            exclude_tail_mask = np.ones(mouse_data.shape[2], dtype=bool)
            exclude_tail_mask[exclude_tail_points] = False

            speed[mouse_num, :, :] = calculate_speed(
                tracked_points_array=mouse_data[:, mouse_num, exclude_tail_mask, :],
                capture_framerate=empirical_camera_sr,
                smoothing_time_window=0.015,
            )

            speed_1st_der[mouse_num, :, :], speed_2nd_der[mouse_num, :, :] = (
                calculate_derivatives(
                    input_arr=speed[mouse_num, :, :],
                    diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                    is_angle=False,
                    capture_fr=empirical_camera_sr,
                )
            )

            # # neck elevation (cm)
            neck_elevation_temp = (
                    mouse_data[:, mouse_num, mouse_nodes.index("Neck"), 2] * 100
            )
            neck_elevation_temp[neck_elevation_temp < 0] = np.nan
            neck_elevation[mouse_num, :, 0] = neck_elevation_temp
            (
                neck_elevation_1st_der[mouse_num, :, :],
                neck_elevation_2nd_der[mouse_num, :, :],
            ) = calculate_derivatives(
                input_arr=neck_elevation[mouse_num, :, :],
                diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                is_angle=False,
                capture_fr=empirical_camera_sr,
            )

            # # head Euler angles (degrees)
            head_input_arr = np.array(
                [
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["head_points"][0]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["head_points"][1]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["head_points"][2]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["head_points"][3]
                        ),
                        :,
                    ],
                ]
            )

            head_input_arr = np.swapaxes(head_input_arr, axis1=0, axis2=1)

            global_head_root = get_head_root(data_arr=head_input_arr)
            global_head_roots[mouse_num, :, :, :] = global_head_root
            global_head_angles[mouse_num, :, :] = get_euler_ang(global_head_root)

            (
                global_head_angles_1st_der[mouse_num, :, :],
                global_head_angles_2nd_der[mouse_num, :, :],
            ) = calculate_derivatives(
                input_arr=global_head_angles[mouse_num, :, :],
                diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                is_angle=True,
                capture_fr=empirical_camera_sr,
            )

            # # egocentric head angles (degrees)
            back_root_inv_oriented = get_back_root(
                point_data_3d=mouse_data,
                mouse_id=mouse_num,
                neck_point_pos=mouse_nodes.index(
                    self.behavioral_parameters_dict["back_root_points"][0]
                ),
                tti_point_pos=mouse_nodes.index(
                    self.behavioral_parameters_dict["back_root_points"][2]
                ),
                root_method="other",
            )

            root_oriented_x = np.einsum(
                "ijk,ik->ij", back_root_inv_oriented, global_head_root[:, 0, :]
            )
            root_oriented_y = np.einsum(
                "ijk,ik->ij", back_root_inv_oriented, global_head_root[:, 1, :]
            )
            root_oriented_z = np.einsum(
                "ijk,ik->ij", back_root_inv_oriented, global_head_root[:, 2, :]
            )
            head_relative_to_body_root = np.array(
                [root_oriented_x, root_oriented_y, root_oriented_z]
            )
            head_relative_to_body_root = np.swapaxes(
                head_relative_to_body_root, axis1=0, axis2=1
            )

            ego_head_angles[mouse_num, :, :] = get_euler_ang(head_relative_to_body_root)
            (
                ego_head_angles_1st_der[mouse_num, :, :],
                ego_head_angles_2nd_der[mouse_num, :, :],
            ) = calculate_derivatives(
                input_arr=ego_head_angles[mouse_num, :, :],
                diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                is_angle=True,
                capture_fr=empirical_camera_sr,
            )

            # # back Euler angles (pitch and yaw, degrees)
            back_root_inv = get_back_root(
                point_data_3d=mouse_data,
                mouse_id=mouse_num,
                neck_point_pos=mouse_nodes.index(
                    self.behavioral_parameters_dict["back_root_points"][0]
                ),
                tti_point_pos=mouse_nodes.index(
                    self.behavioral_parameters_dict["back_root_points"][2]
                ),
                root_method="root_inv",
            )

            inv_back_vector = (
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["back_root_points"][0]
                        ),
                        :,
                    ]
                    - mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["back_root_points"][1]
                        ),
                        :,
                    ]
            )
            # per-frame unit-normalize so that np.nanmean inside get_back_angles
            # weights every frame equally (the previous unkeyed np.linalg.norm
            # returned the Frobenius norm of the whole matrix, biasing the mean
            # toward frames with longer Neck-Trunk segments)
            inv_normalized_back_vector = inv_back_vector / np.linalg.norm(
                inv_back_vector, axis=1, keepdims=True
            )
            back_directions = np.einsum(
                "bij,bj->bi", back_root_inv, inv_normalized_back_vector
            )

            global_back_angles[mouse_num, :, :] = get_back_angles(back_directions)

            (
                global_back_angles_1st_der[mouse_num, :, :],
                global_back_angles_2nd_der[mouse_num, :, :],
            ) = calculate_derivatives(
                input_arr=global_back_angles[mouse_num, :, :],
                diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                is_angle=True,
                capture_fr=empirical_camera_sr,
            )
            # # global body direction (degrees)
            back_root = get_back_root(
                point_data_3d=mouse_data,
                mouse_id=mouse_num,
                neck_point_pos=mouse_nodes.index(
                    self.behavioral_parameters_dict["back_root_points"][0]
                ),
                tti_point_pos=mouse_nodes.index(
                    self.behavioral_parameters_dict["back_root_points"][2]
                ),
                root_method="default",
            )

            global_root_angles[mouse_num, :, :] = -get_euler_ang(back_root)
            (
                global_root_angles_1st_der[mouse_num, :, :],
                global_root_angles_2nd_der[mouse_num, :, :],
            ) = calculate_derivatives(
                input_arr=global_root_angles[mouse_num, :, :],
                diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                is_angle=True,
                capture_fr=empirical_camera_sr,
            )

            # # tail curvature (arbitrary units
            tail_curvature_arr = np.array(
                [
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["tail_points"][0]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["tail_points"][1]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["tail_points"][2]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["tail_points"][3]
                        ),
                        :,
                    ],
                    mouse_data[
                        :,
                        mouse_num,
                        mouse_nodes.index(
                            self.behavioral_parameters_dict["tail_points"][4]
                        ),
                        :,
                    ],
                ]
            )

            tail_curvature_arr = np.swapaxes(tail_curvature_arr, axis1=0, axis2=1)

            tail_curvature[mouse_num, :, :] = calculate_tail_curvature(
                input_arr=tail_curvature_arr
            )
            (
                tail_curvature_1st_der[mouse_num, :, :],
                tail_curvature_2nd_der[mouse_num, :, :],
            ) = calculate_derivatives(
                input_arr=tail_curvature[mouse_num, :, :],
                diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                is_angle=False,
                capture_fr=empirical_camera_sr,
            )

        # # # save individual data to .csv file
        behavioral_features_df = pls.DataFrame()

        for mouse_num in range(mouse_data.shape[1]):
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.spaceX", head_position[mouse_num, :, 0]
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.spaceY", head_position[mouse_num, :, 1]
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.spaceZ", head_position[mouse_num, :, 2]
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(f"{track_names[mouse_num]}.speed", speed[mouse_num, :, 0])
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.acceleration",
                    speed_1st_der[mouse_num, :, 0],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.neck_elevation",
                    neck_elevation[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.neck_elevation_1st_der",
                    neck_elevation_1st_der[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.neck_elevation_2nd_der",
                    neck_elevation_2nd_der[mouse_num, :, 0],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_roll",
                    global_head_angles[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_roll_1st_der",
                    global_head_angles_1st_der[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_roll_2nd_der",
                    global_head_angles_2nd_der[mouse_num, :, 0],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_pitch",
                    global_head_angles[mouse_num, :, 1],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_pitch_1st_der",
                    global_head_angles_1st_der[mouse_num, :, 1],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_pitch_2nd_der",
                    global_head_angles_2nd_der[mouse_num, :, 1],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_yaw",
                    global_head_angles[mouse_num, :, 2],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_yaw_1st_der",
                    global_head_angles_1st_der[mouse_num, :, 2],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.allo_yaw_2nd_der",
                    global_head_angles_2nd_der[mouse_num, :, 2],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.ego_yaw",
                    ego_head_angles[mouse_num, :, 2],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.ego_yaw_1st_der",
                    ego_head_angles_1st_der[mouse_num, :, 2],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.ego_yaw_2nd_der",
                    ego_head_angles_2nd_der[mouse_num, :, 2],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.back_pitch",
                    global_back_angles[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.back_pitch_1st_der",
                    global_back_angles_1st_der[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.back_pitch_2nd_der",
                    global_back_angles_2nd_der[mouse_num, :, 0],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.back_yaw",
                    global_back_angles[mouse_num, :, 1],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.back_yaw_1st_der",
                    global_back_angles_1st_der[mouse_num, :, 1],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.back_yaw_2nd_der",
                    global_back_angles_2nd_der[mouse_num, :, 1],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.body_dir",
                    global_root_angles[mouse_num, :, 2],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.body_dir_1st_der",
                    global_root_angles_1st_der[mouse_num, :, 2],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.body_dir_2nd_der",
                    global_root_angles_2nd_der[mouse_num, :, 2],
                )
            )

            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.tail_curvature",
                    tail_curvature[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.tail_curvature_1st_der",
                    tail_curvature_1st_der[mouse_num, :, 0],
                )
            )
            behavioral_features_df = behavioral_features_df.with_columns(
                pls.Series(
                    f"{track_names[mouse_num]}.tail_curvature_2nd_der",
                    tail_curvature_2nd_der[mouse_num, :, 0],
                )
            )

        # # # compute and save social features for all unique pairs
        if mouse_data.shape[1] >= 2:
            # generate all unique pairs of mouse indices
            mouse_pairs = list(itertools.combinations(range(mouse_data.shape[1]), 2))

            for mouse1_idx, mouse2_idx in mouse_pairs:
                # # social distances (cm) for the current pair
                social_distances = np.zeros((mouse_data.shape[0], 4))

                # nose to nose distance
                social_distances[:, 0] = (
                        np.linalg.norm(
                            mouse_data[:, mouse1_idx, mouse_nodes.index("Nose"), :]
                            - mouse_data[:, mouse2_idx, mouse_nodes.index("Nose"), :],
                            axis=1,
                        )
                        * 100
                )

                # TTI to TTI distance
                social_distances[:, 1] = (
                        np.linalg.norm(
                            mouse_data[:, mouse1_idx, mouse_nodes.index("TTI"), :]
                            - mouse_data[:, mouse2_idx, mouse_nodes.index("TTI"), :],
                            axis=1,
                        )
                        * 100
                )

                # nose (mouse1) to TTI (mouse2) distance
                social_distances[:, 2] = (
                        np.linalg.norm(
                            mouse_data[:, mouse1_idx, mouse_nodes.index("Nose"), :]
                            - mouse_data[:, mouse2_idx, mouse_nodes.index("TTI"), :],
                            axis=1,
                        )
                        * 100
                )

                # TTI (mouse1) to nose (mouse2) distance
                social_distances[:, 3] = (
                        np.linalg.norm(
                            mouse_data[:, mouse1_idx, mouse_nodes.index("TTI"), :]
                            - mouse_data[:, mouse2_idx, mouse_nodes.index("Nose"), :],
                            axis=1,
                        )
                        * 100
                )

                # compute derivatives for distances
                social_distances_1st_der, social_distances_2nd_der = calculate_derivatives(
                    input_arr=social_distances,
                    diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                    is_angle=False,
                    capture_fr=empirical_camera_sr,
                )

                # # egocentric social angles (yaw and pitch, in degrees) for the
                # current pair, expressed in the observer's anatomical head
                # frame: yaw = signed left/right of observer's gaze axis,
                # pitch = signed elevation above/below observer's gaze axis.
                # Layout: columns 0..3 = yaw to {m2.Nose, m1.Nose seen from m2,
                # m2.TTI, m1.TTI seen from m2}; columns 4..7 = matching pitch.
                social_angles = np.zeros((mouse_data.shape[0], 8))

                head1 = mouse_data[:, mouse1_idx, mouse_nodes.index("Head"), :]
                head2 = mouse_data[:, mouse2_idx, mouse_nodes.index("Head"), :]
                nose1 = mouse_data[:, mouse1_idx, mouse_nodes.index("Nose"), :]
                nose2 = mouse_data[:, mouse2_idx, mouse_nodes.index("Nose"), :]
                tti1 = mouse_data[:, mouse1_idx, mouse_nodes.index("TTI"), :]
                tti2 = mouse_data[:, mouse2_idx, mouse_nodes.index("TTI"), :]
                root1 = global_head_roots[mouse1_idx, :, :, :]
                root2 = global_head_roots[mouse2_idx, :, :, :]

                # mouse1 observes mouse2's nose
                social_angles[:, 0], social_angles[:, 4] = get_egocentric_direction(
                    head_root=root1, head_pivot=head1, target_point=nose2
                )

                # mouse2 observes mouse1's nose (reverse direction; column
                # naming follows the legacy asymmetric pattern nose-allo_yaw)
                social_angles[:, 1], social_angles[:, 5] = get_egocentric_direction(
                    head_root=root2, head_pivot=head2, target_point=nose1
                )

                # mouse1 observes mouse2's TTI
                social_angles[:, 2], social_angles[:, 6] = get_egocentric_direction(
                    head_root=root1, head_pivot=head1, target_point=tti2
                )

                # mouse2 observes mouse1's TTI
                social_angles[:, 3], social_angles[:, 7] = get_egocentric_direction(
                    head_root=root2, head_pivot=head2, target_point=tti1
                )

                # compute derivatives for angles
                social_angles_1st_der, social_angles_2nd_der = calculate_derivatives(
                    input_arr=social_angles,
                    diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                    is_angle=True,
                    capture_fr=empirical_camera_sr,
                )

                # add this pair's social features to the DataFrame
                pair_name = f"{track_names[mouse1_idx]}-{track_names[mouse2_idx]}"

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-nose", social_distances[:, 0])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-nose_1st_der", social_distances_1st_der[:, 0])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-nose_2nd_der", social_distances_2nd_der[:, 0])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-TTI", social_distances[:, 1])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-TTI_1st_der", social_distances_1st_der[:, 1])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-TTI_2nd_der", social_distances_2nd_der[:, 1])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-TTI", social_distances[:, 2])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-TTI_1st_der", social_distances_1st_der[:, 2])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-TTI_2nd_der", social_distances_2nd_der[:, 2])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-nose", social_distances[:, 3])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-nose_1st_der", social_distances_1st_der[:, 3])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-nose_2nd_der", social_distances_2nd_der[:, 3])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_yaw-nose", social_angles[:, 0])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_yaw-nose_1st_der", social_angles_1st_der[:, 0])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_yaw-nose_2nd_der", social_angles_2nd_der[:, 0])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-allo_yaw", social_angles[:, 1])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-allo_yaw_1st_der", social_angles_1st_der[:, 1])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-allo_yaw_2nd_der", social_angles_2nd_der[:, 1])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_yaw-TTI", social_angles[:, 2])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_yaw-TTI_1st_der", social_angles_1st_der[:, 2])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_yaw-TTI_2nd_der", social_angles_2nd_der[:, 2])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-allo_yaw", social_angles[:, 3])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-allo_yaw_1st_der", social_angles_1st_der[:, 3])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-allo_yaw_2nd_der", social_angles_2nd_der[:, 3])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_pitch-nose", social_angles[:, 4])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_pitch-nose_1st_der", social_angles_1st_der[:, 4])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_pitch-nose_2nd_der", social_angles_2nd_der[:, 4])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-allo_pitch", social_angles[:, 5])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-allo_pitch_1st_der", social_angles_1st_der[:, 5])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.nose-allo_pitch_2nd_der", social_angles_2nd_der[:, 5])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_pitch-TTI", social_angles[:, 6])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_pitch-TTI_1st_der", social_angles_1st_der[:, 6])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.allo_pitch-TTI_2nd_der", social_angles_2nd_der[:, 6])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-allo_pitch", social_angles[:, 7])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-allo_pitch_1st_der", social_angles_1st_der[:, 7])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.TTI-allo_pitch_2nd_der", social_angles_2nd_der[:, 7])
                )

                # # social engagement index (SEI)
                social_engagement_indices = np.zeros((mouse_data.shape[0], 4))

                # mouse1-mouse2 orofacial SEI
                social_engagement_indices[:, 0] = calculate_sei(tracks=mouse_data,
                                                                speed_arr=speed[mouse1_idx, :, 0],
                                                                observer_idx=mouse1_idx,
                                                                observed_idx=mouse2_idx,
                                                                observed_node_idx=mouse_nodes.index("Nose"),
                                                                observer_head_root=global_head_roots[mouse1_idx, :, :, :],
                                                                idx_nose=mouse_nodes.index("Nose"),
                                                                idx_tti=mouse_nodes.index("TTI"),
                                                                idx_head=mouse_nodes.index("Head"))

                # mouse1-mouse2 anogenital SEI
                social_engagement_indices[:, 1] = calculate_sei(tracks=mouse_data,
                                                                speed_arr=speed[mouse1_idx, :, 0],
                                                                observer_idx=mouse1_idx,
                                                                observed_idx=mouse2_idx,
                                                                observed_node_idx=mouse_nodes.index("TTI"),
                                                                observer_head_root=global_head_roots[mouse1_idx, :, :, :],
                                                                idx_nose=mouse_nodes.index("Nose"),
                                                                idx_tti=mouse_nodes.index("TTI"),
                                                                idx_head=mouse_nodes.index("Head"))

                # mouse2-mouse1 orofacial SEI
                social_engagement_indices[:, 2] = calculate_sei(tracks=mouse_data,
                                                                speed_arr=speed[mouse2_idx, :, 0],
                                                                observer_idx=mouse2_idx,
                                                                observed_idx=mouse1_idx,
                                                                observed_node_idx=mouse_nodes.index("Nose"),
                                                                observer_head_root=global_head_roots[mouse2_idx, :, :, :],
                                                                idx_nose=mouse_nodes.index("Nose"),
                                                                idx_tti=mouse_nodes.index("TTI"),
                                                                idx_head=mouse_nodes.index("Head"))

                # mouse2-mouse1 anogenital SEI
                social_engagement_indices[:, 3] = calculate_sei(tracks=mouse_data,
                                                                speed_arr=speed[mouse2_idx, :, 0],
                                                                observer_idx=mouse2_idx,
                                                                observed_idx=mouse1_idx,
                                                                observed_node_idx=mouse_nodes.index("TTI"),
                                                                observer_head_root=global_head_roots[mouse2_idx, :, :, :],
                                                                idx_nose=mouse_nodes.index("Nose"),
                                                                idx_tti=mouse_nodes.index("TTI"),
                                                                idx_head=mouse_nodes.index("Head"))

                # compute derivatives for SEI
                social_engagement_indices_1st_der, social_engagement_indices_2nd_der = calculate_derivatives(
                    input_arr=social_engagement_indices,
                    diff_bins=self.behavioral_parameters_dict["derivative_bins"],
                    is_angle=False,
                    capture_fr=empirical_camera_sr,
                )

                # add this pair's SEIs to the DataFrame
                pair_name_reverse = f"{track_names[mouse2_idx]}-{track_names[mouse1_idx]}"

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.orofacial-sei", social_engagement_indices[:, 0])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.orofacial-sei_1st_der", social_engagement_indices_1st_der[:, 0])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.orofacial-sei_2nd_der", social_engagement_indices_2nd_der[:, 0])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.anogenital-sei", social_engagement_indices[:, 1])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.anogenital-sei_1st_der", social_engagement_indices_1st_der[:, 1])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name}.anogenital-sei_2nd_der", social_engagement_indices_2nd_der[:, 1])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name_reverse}.orofacial-sei", social_engagement_indices[:, 2])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name_reverse}.orofacial-sei_1st_der", social_engagement_indices_1st_der[:, 2])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name_reverse}.orofacial-sei_2nd_der", social_engagement_indices_2nd_der[:, 2])
                )

                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name_reverse}.anogenital-sei", social_engagement_indices[:, 3])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name_reverse}.anogenital-sei_1st_der", social_engagement_indices_1st_der[:, 3])
                )
                behavioral_features_df = behavioral_features_df.with_columns(
                    pls.Series(f"{pair_name_reverse}.anogenital-sei_2nd_der", social_engagement_indices_2nd_der[:, 3])
                )

        # # # # compute feature distributions
        feature_distribution_dict = {}
        space_computed_occ = dict.fromkeys(track_names, False)
        for column in behavioral_features_df.columns:
            if "space" not in column:
                feature_distribution_dict[column] = {}
                (
                    feature_distribution_dict[column]["occ_array"],
                    feature_distribution_dict[column]["bin_centers"],
                    feature_distribution_dict[column]["bin_edges"],
                ) = generate_feature_distributions(
                    feature_arr=behavioral_features_df.select(column).to_numpy(),
                    min_val=self.feature_boundaries[column.split(".")[1]][0],
                    max_val=self.feature_boundaries[column.split(".")[1]][1],
                    num_bins=36,
                    camera_fr=empirical_camera_sr,
                    space_bool=False,
                )
            elif space_computed_occ[f"{column.split('.')[0]}"] is False:
                feature_distribution_dict[column] = {}
                (
                    feature_distribution_dict[column]["occ_array"],
                    feature_distribution_dict[column]["bin_centers"],
                    feature_distribution_dict[column]["bin_edges"],
                ) = generate_feature_distributions(
                    feature_arr=np.stack(
                        arrays=(
                            np.array(
                                behavioral_features_df.select(
                                    f"{column.split('.')[0]}.spaceX"
                                ).to_numpy()
                            ),
                            np.array(
                                behavioral_features_df.select(
                                    f"{column.split('.')[0]}.spaceY"
                                ).to_numpy()
                            ),
                        ),
                        axis=1,
                    ),
                    min_val=-32,
                    max_val=32,
                    num_bins=196,
                    camera_fr=empirical_camera_sr,
                    space_bool=True,
                )
                space_computed_occ[f"{column.split('.')[0]}"] = True

        # # # # plot feature distributions
        self.plot_feature_distributions(
            feature_dict=feature_distribution_dict,
            mouse_id_list=track_names,
            session_exp_code=experimental_code,
            plot_file_name=f"{tracked_file_loc.with_suffix('')}_behavioral_features_histograms.pdf",
        )

        # # # # save data to .csv file
        behavioral_features_df.write_csv(
            file=f"{tracked_file_loc.with_suffix('')}_behavioral_features.csv",
            separator=",",
            include_header=True,
        )
