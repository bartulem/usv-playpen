"""
@author: bartulem
Helper functions for making vocalocator datasets.
"""

import h5py
from joblib import Parallel, delayed
import numpy as np
import pathlib
import polars as pls
from scipy.stats import vonmises as sp_vonmises
from typing import Optional, Any
from tqdm import tqdm

softplus = lambda x: np.log1p(np.exp(x))

def get_arena_dimensions(arena_dims_path: pathlib.Path = None) -> np.ndarray:
    """
    Description
    ----------
    Prepares the root directory for vocalocator inference.
    ----------

    Parameters
    ----------
    arena_dims_path (pathlib.Path)
        Path to the 3D arena tracking file.
    ----------

    Returns
    ----------
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    ----------
    """

    with h5py.File(arena_dims_path, mode='r') as f:
        arena_nodes = [node_item.decode('utf-8') for node_item in list(f['node_names'])]
        node_list_indices = [arena_nodes.index('North'),
                             arena_nodes.index('West'),
                             arena_nodes.index('South'),
                             arena_nodes.index('East')]
        four_corners = f['tracks'][0, 0, node_list_indices, :]
        x_dim = np.max(four_corners[:, 0]) - np.min(four_corners[:, 0])
        y_dim = np.max(four_corners[:, 1]) - np.min(four_corners[:, 1])

    return np.array([x_dim, y_dim])

def load_usv_segments(segment_file: pathlib.Path = None) -> np.ndarray:
    """
    Description
    ----------
    Loads start/stop of each USV from CSV file
    and transfers it into a Numpy array.
    ----------

    Parameters
    ----------
    segment_file (pathlib.Path)
        Path to the USV summary CSV file.
    ----------

    Returns
    ----------
    usv_segments (np.ndarray)
        A (USV_NUM, 2) shape ndarray containing start and stop of each USV.
    ----------
    """

    usv_summary_df = pls.read_csv(segment_file)

    return np.stack(arrays=[usv_summary_df['start'].to_numpy(), usv_summary_df['stop'].to_numpy()],
                    axis=1)

def load_tracks_from_h5(h5_file_path: pathlib.Path = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    ----------
    Loads tracks and node names from 3D tracking file
    ----------

    Parameters
    ----------
    h5_file_path (pathlib.Path)
        Path to the 3D tracking file.
    ----------

    Returns
    ----------
    (tuple[np.ndarray, np.ndarray])
        Mouse tracks and node names.
    ----------
    """

    with h5py.File(name=h5_file_path, mode='r') as f:
        tracks = f['tracks'][:]
        names = f['node_names'][:]
        return tracks, names

def to_float(input_array: np.ndarray = None) -> np.ndarray:
    """
    Description
    ----------
    Converts the input array to float16.
    ----------

    Parameters
    ----------
    input_array (np.ndarray)
        Input array to be converted to float16.
    ----------

    Returns
    ----------
    (np.ndarray)
        Array converted to float16.
    ----------
    """

    return (input_array.astype(np.float32) / np.iinfo(input_array.dtype).max).astype(np.float16)

def write_to_h5(output_path: pathlib.Path = None,
                audio: list[tuple[np.ndarray, Any]] = None,
                node_names: np.ndarray = None,
                locations: np.ndarray = None,
                length_idx: np.ndarray = None,
                extra_metadata: Optional[dict] = None) -> None:
    """
    Description
    ----------
    Writes audio data, node names, locations, and length indices to an HDF5 file.
    ----------

    Parameters
    ----------
    output_path (pathlib.Path)
        Path to the output HDF5 file.
    audio (list[tuple[np.ndarray]])
        List of audio segments to be written to the file.
    node_names (np.ndarray)
        Array of node names.
    locations (np.ndarray)
        Track locations at USV onsets.
    length_idx (np.ndarray)
        Array of USV durations.
    extra_metadata (dict)
        Additional metadata to be stored in the HDF5 file.
    ----------

    Returns
    ----------
    ----------
    """

    with h5py.File(output_path, mode='w') as f:
        if extra_metadata is not None:
            for k, v in extra_metadata.items():
                f.attrs[k] = v
        f.create_dataset(name='audio', data=np.concatenate(audio, axis=0), dtype=np.float16)
        f.create_dataset(name='node_names', data=node_names)
        f.create_dataset(name='locations', data=locations)
        f.create_dataset(name='length_idx', data=length_idx)

def eval_pdf_with_angle(points_spatial: np.ndarray = None,
                        points_angular: np.ndarray = None,
                        mean_2d: np.ndarray = None,
                        cov_2d: np.ndarray = None,
                        center_rad: Optional[float] = None,
                        concentration: Optional[float] = None,
                        histogram: Optional[np.ndarray] = None,) -> np.ndarray:
    """
    Description
    ----------
    Evaluate the multivariate normal PDF at points. Assumes the points
    and the mean are in the same coordinate system.
    ----------

    Parameters
    ----------
    points_spatial (np.ndarray)
        Points in spatial coordinates. Shape: (*n_points, 2)
    points_angular (np.ndarray)
        Points in angular coordinates. Shape: (*n_points, 1)
    mean_2d (np.ndarray)
        Mean of the multivariate normal distribution. Shape: (2,)
    cov_2d (np.ndarray)
        Covariance matrix of the multivariate normal distribution. Shape: (2, 2)
    center_rad (float)
        Center of the von Mises distribution in radians.
    concentration (float)
        Concentration parameter of the von Mises distribution.
    histogram (np.ndarray)
        Histogram of the angular distribution. Shape: (n_bins,)
    ----------

    Returns
    ----------
    (np.ndarray)
        Evaluated PDF at the given points.
    ----------
    """

    points_orig_shape = points_spatial.shape[:-1]
    points_spatial = points_spatial.reshape(-1, 2)
    diff = points_spatial - mean_2d

    if histogram is None:
        angular_pdf = sp_vonmises.pdf(points_angular, loc=center_rad, kappa=concentration)
    else:
        angular_pdf = histogram / histogram.sum()
    try:
        precision = np.linalg.inv(cov_2d)
        log_prob = -0.5 * np.einsum("ij,jk,ik->i", diff, precision, diff)
        log_prob = log_prob[:, None] + np.log(angular_pdf + 1e-12)[None, :]
        log_prob -= log_prob.max()
        probs = np.exp(log_prob)
        probs /= probs.sum()

        return probs.reshape(*points_orig_shape, len(points_angular))

    except np.linalg.LinAlgError:
        closest_point_idx = np.argmin(np.linalg.norm(points_spatial - mean_2d, axis=-1))
        probs = np.zeros(points_spatial.shape[0], len(points_angular))
        probs[closest_point_idx, :] = 1

        return probs.reshape(*points_orig_shape, len(points_angular))

def compute_covs_6d(raw_outputs: np.ndarray,
                    arena_dims: np.ndarray) -> np.ndarray:
    """
    Description
    ----------
    Computes the covariance matrix from the raw output of the model.
    ----------

    Parameters
    ----------
    raw_outputs (np.ndarray)
        Raw output from the model. Shape: (B, num_outputs)
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    ----------

    Returns
    ----------
    (np.ndarray)
        Covariance matrix in mm^2. Shape: (B, 6, 6)
    ----------
    """

    raw_outputs = raw_outputs
    n_dims = 6
    L = np.zeros((raw_outputs.shape[0], n_dims, n_dims))
    idxs = np.tril_indices(n_dims)
    L[:, idxs[0], idxs[1]] = raw_outputs[:, n_dims:]
    new_diagonals = softplus(np.diagonal(L, axis1=-2, axis2=-1))  # (batch, 2)
    L[:, np.arange(n_dims), np.arange(n_dims)] = new_diagonals
    scale = 0.5 * arena_dims.max()
    L = L * scale
    covs = np.einsum("bjk,bmk->bjm", L, L)

    return covs

def estimate_angle_pdf(pred_6d_mean: np.ndarray,
                       pred_6d_cov: np.ndarray,
                       n_samples: int = 1000,
                       theta_bins: Optional[np.ndarray] = None,) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    ----------
    Estimate the angle PDF from the 6d covariance matrix.
    ----------

    Parameters
    ----------
    pred_6d_mean (np.ndarray)
        Mean of the 6d Gaussian distribution. Shape: (6,)
    pred_6d_cov (np.ndarray)
        Covariance matrix of the 6d Gaussian distribution. Shape: (6, 6)
    n_samples (int)
        Number of samples to draw from the Gaussian distribution.
    theta_bins (np.ndarray)
        Bins for the angle histogram. If None, defaults to 46 bins from -pi to pi.
    ----------

    Returns
    ----------
    (tuple[np.ndarray, np.ndarray])
        Tuple containing the angle bins and the estimated PDF.
    ----------
    """

    if theta_bins is None:
        theta_bins = np.linspace(-np.pi, np.pi, 46, endpoint=True)

    gaussian_rv = np.random.multivariate_normal(mean=pred_6d_mean, cov=pred_6d_cov, size=n_samples)
    angles = np.arctan2(gaussian_rv[:, 1] - gaussian_rv[:, 4], gaussian_rv[:, 0] - gaussian_rv[:, 3])
    angle_pdf, _ = np.histogram(angles, bins=theta_bins, density=False)
    angle_pdf = angle_pdf / angle_pdf.sum()

    return theta_bins, angle_pdf

def get_confidence_set(pdf: np.ndarray = None,
                       confidence_level: float = None) -> np.ndarray:
    """
    Description
    ----------
    Get the confidence set for the given PDF. Makes no assumptions about
    the location of the origin in the arena.
    ----------

    Parameters
    ----------
    pdf (np.ndarray)
        Probability density function. Shape: (y_res, x_res, angle_res)
    confidence_level (float)
        Confidence level for the confidence set. Should be between 0 and 1.
    ----------

    Returns
    ----------
    (np.ndarray)
        Boolean array of the same shape as pdf, indicating the confidence set.
    ----------
    """

    orig_shape = pdf.shape
    flat_pdf = pdf.flatten()
    sorted_indices = np.argsort(flat_pdf)[::-1]
    cumsum = np.cumsum(flat_pdf[sorted_indices])
    idx_in_confidence_set = sorted_indices[cumsum < confidence_level]
    confidence_set = np.zeros_like(flat_pdf, dtype=bool)
    confidence_set[idx_in_confidence_set] = True

    return confidence_set.reshape(orig_shape)


def make_xy_grid(arena_dims: np.ndarray = None,
                 render_dims: np.ndarray = None,) -> np.ndarray:
    """
    Description
    ----------
    Generates a grid of points for evaluating a PMF.
    Places origin in the bottom left corner of the arena
    ----------

    Parameters
    ----------
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    render_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the render.
    ----------

    Returns
    ----------
    (np.ndarray)
        A grid of points for evaluating a PMF. Shape: (y_res, x_res, 2)
    ----------
    """

    test_points = np.stack(np.meshgrid(np.linspace(-arena_dims[0] / 2, arena_dims[0] / 2, render_dims[0]), np.linspace(-arena_dims[1] / 2, arena_dims[1] / 2, render_dims[1]),), axis=-1,)

    return test_points

def convert_from_arb(output: np.ndarray = None,
                     arena_dims: np.ndarray = None) -> np.ndarray:
    """
    Description
    ----------
    Converts the output from the model to a more interpretable format.
    ----------

    Parameters
    ----------
    output (np.ndarray)
        Output from the model. Shape: (B, 6)
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    ----------

    Returns
    ----------
    (np.ndarray)
        Converted output. Shape: (B, 6)
    ----------
    """

    scale_factor = arena_dims.max() / 2
    output = output * scale_factor

    return output

def get_conf_sets_6d(raw_output: np.ndarray = None,
                     arena_dims_mm: np.ndarray = None,
                     temperature: float = 1.0,
                     return_pdf: bool = False,) -> tuple:
    """
    Description
    ----------
    Computes confidence sets and optionally PDFs from raw output.
    ----------

    Parameters
    ----------
    raw_output (np.ndarray)
        Raw output from the model.
    arena_dims_mm (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    temperature (float)
        Temperature parameter for scaling.
    return_pdf (bool)
        Whether to return PDFs or not.
    ----------

    Returns
    ----------
    (tuple)
        Contains confidence sets and optionally PDFs.
    ----------
    """

    pred_means_mm = convert_from_arb(raw_output[:, :6], arena_dims=arena_dims_mm)
    pred_cov_6d_mm = compute_covs_6d(raw_output, arena_dims_mm) * temperature
    points_spatial = make_xy_grid(arena_dims_mm, (100, 100))
    bins_angular = np.linspace(-np.pi, np.pi, 46, endpoint=True)
    points_angular = 0.5 * (bins_angular[1:] + bins_angular[:-1])  # Bin centers

    def routine(mean_6d, cov_6d):
        _, est_angle_pdf = estimate_angle_pdf(mean_6d, cov_6d, n_samples=500, theta_bins=bins_angular)
        total_pdf = eval_pdf_with_angle(points_spatial=points_spatial,
                                        points_angular=points_angular,
                                        mean_2d=mean_6d[:2],
                                        cov_2d=cov_6d[:2, :2],
                                        histogram=est_angle_pdf,)
        conf_set = get_confidence_set(total_pdf, 0.95)
        conf_set_no_angle = get_confidence_set(total_pdf.sum(axis=-1), 0.95)
        return conf_set, conf_set_no_angle, total_pdf

    results = Parallel(n_jobs=-1)(delayed(routine)(mean_6d, cov_6d) for mean_6d, cov_6d in tqdm(zip(pred_means_mm, pred_cov_6d_mm), total=len(pred_means_mm)))

    conf_sets = np.array([result[0] for result in results])
    conf_sets_noangle = np.array([result[1] for result in results])
    pdfs = np.array([result[2] for result in results])

    if return_pdf:
        return conf_sets, conf_sets_noangle, pdfs

    return conf_sets, conf_sets_noangle

def are_points_in_conf_set(confidence_sets: np.ndarray = None,
                           points: np.ndarray = None,
                           arena_dims: np.ndarray = None) -> np.ndarray:
    """
    Description
    ----------
    Given an array of confidence sets and points, computes whether each point
    is inside the respective confidence set.
    ----------

    Parameters
    ----------
    confidence_sets (np.ndarray)
        Confidence sets. Shape: (n, y_res, x_res)
    points (np.ndarray)
        Points to test. Shape: (n, n_node, 3)
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    ----------

    Returns
    ----------
    (np.ndarray)
        Boolean array of shape (n)
    ----------
    """

    head_to_nose_vecs = points[:, 0, :] - points[:, 1, :]
    head_to_nose_yaw = np.arctan2(head_to_nose_vecs[:, 1], head_to_nose_vecs[:, 0])
    nose_points = np.clip(points[:, 0, :2], -arena_dims[:2] / 2, arena_dims[:2] / 2)
    y_bins = np.linspace(-arena_dims[1] / 2, arena_dims[1] / 2, 100)
    x_bins = np.linspace(-arena_dims[0] / 2, arena_dims[0] / 2, 100)
    angle_bins = np.linspace(-np.pi, np.pi, 45, endpoint=False)
    y_bin_indices = np.digitize(nose_points[:, 1], y_bins) - 1
    x_bin_indices = np.digitize(nose_points[:, 0], x_bins) - 1
    angle_bin_indices = np.digitize(head_to_nose_yaw, angle_bins) - 1

    if len(confidence_sets.shape) == 3:  # no angles in the confidence set
        in_set = confidence_sets[np.arange(confidence_sets.shape[0]), y_bin_indices, x_bin_indices]
    elif len(confidence_sets.shape) == 4:  # has angles
        in_set = confidence_sets[np.arange(len(confidence_sets)), y_bin_indices, x_bin_indices, angle_bin_indices,]
    else:
        raise ValueError("Invalid confidence set shape")

    return in_set
