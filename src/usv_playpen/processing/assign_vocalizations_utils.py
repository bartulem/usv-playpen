"""
@author: bartulem
Helper functions for making vocalocator datasets.
"""

from __future__ import annotations

import pathlib
from typing import Any

import h5py
import numpy as np
import polars as pls
from joblib import Parallel, delayed
from tqdm import tqdm

from ..os_utils import atomic_output_path

# Fixed seed for the Monte-Carlo angle-PDF estimation so that the sound-
# localization angle marginals -- and therefore the vocalization-to-mouse
# assignment that consumes them -- are reproducible across runs. A single fixed
# seed is used by design (every vocalization shares the same MC draw).
_ANGLE_PDF_SEED = 0


def softplus(x):
    """
    Description
    -----------
    Numerically stable softplus activation: f(x) = log(1 + exp(x)).
    Used to convert unbounded real-valued outputs into strictly positive values
    (e.g., variances) without introducing NaNs for large negative inputs.

    Parameters
    ----------
    x (np.ndarray or float)
        Input value(s) of arbitrary shape.

    Returns
    -------
    (np.ndarray or float)
        Element-wise softplus of the input, same shape as x.
    """

    return np.log1p(np.exp(x))


def get_arena_dimensions(arena_dims_path: pathlib.Path) -> np.ndarray:
    """
    Description
    -----------
    Reads the arena's four corner nodes (North/West/South/East) from a 3D arena
    tracking H5 file and returns the arena's X and Y extents (the span between
    the min and max corner coordinates along each axis).

    Parameters
    ----------
    arena_dims_path (pathlib.Path)
        Path to the 3D arena tracking file.

    Returns
    -------
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    """

    with h5py.File(arena_dims_path, mode="r") as f:
        arena_nodes = [node_item.decode("utf-8") for node_item in list(f["node_names"])]
        node_list_indices = [
            arena_nodes.index("North"),
            arena_nodes.index("West"),
            arena_nodes.index("South"),
            arena_nodes.index("East"),
        ]
        four_corners = f["tracks"][0, 0, node_list_indices, :]
        x_dim = np.max(four_corners[:, 0]) - np.min(four_corners[:, 0])
        y_dim = np.max(four_corners[:, 1]) - np.min(four_corners[:, 1])

    return np.array([x_dim, y_dim])


def load_usv_segments(segment_file: pathlib.Path) -> np.ndarray:
    """
    Description
    -----------
    Loads start/stop of each USV from CSV file
    and transfers it into a Numpy array.

    Parameters
    ----------
    segment_file (pathlib.Path)
        Path to the USV summary CSV file.

    Returns
    -------
    usv_segments (np.ndarray)
        A (USV_NUM, 2) shape ndarray containing start and stop of each USV.
    """

    usv_summary_df = pls.read_csv(str(segment_file))

    return np.stack(
        arrays=[usv_summary_df["start"].to_numpy(), usv_summary_df["stop"].to_numpy()],
        axis=1,
    )


def load_tracks_from_h5(
    h5_file_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Loads tracks and node names from 3D tracking file

    Parameters
    ----------
    h5_file_path (pathlib.Path)
        Path to the 3D tracking file.

    Returns
    -------
    (tuple[np.ndarray, np.ndarray])
        Mouse tracks and node names.
    """

    with h5py.File(name=h5_file_path, mode="r") as f:
        tracks = f["tracks"][:]
        names = f["node_names"][:]
        return tracks, names


def to_float(input_array: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Normalizes an integer-typed array to the [-1, 1] range (or [0, 1] for an
    unsigned dtype) by dividing every element by the maximum value representable
    by the input dtype, then casts the result to float16. Intended for rescaling
    raw integer PCM audio into a normalized floating-point representation.

    Note: the input array must have an integer dtype, because the normalization
    divisor is `np.iinfo(input_array.dtype).max`; passing a floating-point array
    raises ValueError ("Invalid integer data type").

    Parameters
    ----------
    input_array (np.ndarray)
        Integer-typed input array (e.g., raw PCM audio samples) to be normalized
        and converted to float16.

    Returns
    -------
    (np.ndarray)
        Input array normalized by its integer dtype's maximum value and cast to
        float16.
    """

    return (input_array.astype(np.float32) / np.iinfo(input_array.dtype).max).astype(
        np.float16
    )


def write_to_h5(
    output_path: pathlib.Path,
    audio: list[tuple[np.ndarray, Any]],
    node_names: np.ndarray,
    locations: np.ndarray,
    length_idx: np.ndarray,
    animal_ids: np.ndarray | None = None,
    extra_metadata: dict | None = None,
) -> None:
    """
    Description
    -----------
    Writes audio data, node names, locations, and length indices to an HDF5 file.
    The per-USV audio segments are concatenated along axis 0 into a single
    'audio' dataset.

    Parameters
    ----------
    output_path (pathlib.Path)
        Path to the output HDF5 file.
    audio (list[np.ndarray])
        List of per-USV audio segments; concatenated along axis 0 into one
        'audio' dataset.
    node_names (np.ndarray)
        Array of node names.
    locations (np.ndarray)
        Track locations at USV onsets.
    length_idx (np.ndarray)
        Array of USV durations.
    animal_ids (np.ndarray)
        1-D array of integer animal IDs (e.g. [0, 1]). When provided, a
        dataset of shape (num_calls, num_animals) is written under the key
        'animal_id', tiling the IDs across all calls.
    extra_metadata (dict)
        Additional metadata to be stored in the HDF5 file.

    Returns
    -------
    None
    """

    with atomic_output_path(output_path) as tmp_path, h5py.File(tmp_path, mode="w") as f:
        if extra_metadata is not None:
            for k, v in extra_metadata.items():
                f.attrs[k] = v
        # np.concatenate raises on an empty sequence, so a session with zero USV
        # segments would crash here. Fall back to an empty (0, n_channels) float16
        # array (n_channels taken from the first segment when available, else 0)
        # so the 'audio' dataset is still written with a consistent dtype/rank.
        if len(audio) > 0:
            audio_data = np.concatenate(audio, axis=0)
        else:
            audio_data = np.empty((0, 0), dtype=np.float16)
        f.create_dataset(
            name="audio", data=audio_data, dtype=np.float16
        )
        f.create_dataset(name="node_names", data=node_names)
        f.create_dataset(name="locations", data=locations)
        f.create_dataset(name="length_idx", data=length_idx)
        if animal_ids is not None:
            num_calls = locations.shape[0]
            f.create_dataset(name="animal_id", data=np.tile(animal_ids, (num_calls, 1)))


def eval_pdf_with_angle(
    points_spatial: np.ndarray,
    points_angular: np.ndarray,
    mean_2d: np.ndarray,
    cov_2d: np.ndarray,
    histogram: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
    Evaluates a joint spatial-by-angular probability mass function over a grid
    of points, returning a discrete normalized PMF (not a continuous MVN value).
    The spatial component is a 2D Gaussian with the supplied mean and covariance
    (evaluated as the einsum log-probability against the precision matrix); the
    angular component is the supplied `histogram` renormalized to sum to 1. The
    two components are combined in log space, the per-grid maximum is subtracted
    for numerical stability, exponentiated, and the whole array is normalized to
    sum to 1 over the (spatial x angular) grid. Assumes the points and the mean
    are in the same coordinate system.

    If the covariance matrix is singular (np.linalg.LinAlgError on inversion),
    the function falls back to placing all mass on the spatial grid point closest
    to the mean, spread uniformly over the angular bins and normalized to sum to 1
    (a proper distribution, like the main path) so the downstream cumsum-based
    confidence set is non-empty.

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
    histogram (np.ndarray)
        Histogram of the angular distribution, renormalized internally to sum
        to 1. Shape: (n_bins,)

    Returns
    -------
    (np.ndarray)
        Evaluated PDF at the given points.
    """

    points_orig_shape = points_spatial.shape[:-1]
    points_spatial = points_spatial.reshape(-1, 2)
    diff = points_spatial - mean_2d

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
        probs = np.zeros((points_spatial.shape[0], len(points_angular)))
        # Normalize the degenerate fallback to a proper distribution (sums to 1),
        # matching the main path's `probs /= probs.sum()`. An unnormalized fallback
        # (summing to len(points_angular)) makes the downstream cumsum-based
        # get_confidence_set hit 1.0 on the first point, so the strict `< alpha`
        # mask is all-False and an EMPTY confidence set is returned for any
        # vocalization with a singular covariance.
        probs[closest_point_idx, :] = 1.0 / len(points_angular)

        return probs.reshape(*points_orig_shape, len(points_angular))


def compute_covs_6d(raw_outputs: np.ndarray, arena_dims: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Computes a batch of 6x6 covariance matrices from the raw model output by
    reconstructing a lower-triangular Cholesky factor L. The 21 lower-triangular
    entries are read from raw_outputs[:, 6:] (the first 6 columns hold the 6D
    means and are not used here), softplus is applied to L's diagonal to
    guarantee positive entries, L is scaled by half the larger arena dimension
    (0.5 * arena_dims.max()) to convert from arbitrary units into mm, and the
    covariance is formed as covs = L @ L^T. raw_outputs is therefore expected to
    have at least 27 columns (6 means + 21 lower-triangular factor entries).

    Parameters
    ----------
    raw_outputs (np.ndarray)
        Raw output from the model. Shape: (B, num_outputs), where num_outputs is
        at least 27: columns [:6] are the 6D means and columns [6:27] are the 21
        lower-triangular Cholesky-factor entries.
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.

    Returns
    -------
    (np.ndarray)
        Covariance matrix in mm^2. Shape: (B, 6, 6)
    """

    n_dims = 6
    L = np.zeros((raw_outputs.shape[0], n_dims, n_dims))
    idxs = np.tril_indices(n_dims)
    L[:, idxs[0], idxs[1]] = raw_outputs[:, n_dims:]
    new_diagonals = softplus(np.diagonal(L, axis1=-2, axis2=-1))  # (batch, 6)
    L[:, np.arange(n_dims), np.arange(n_dims)] = new_diagonals
    scale = 0.5 * arena_dims.max()
    L = L * scale
    covs = np.einsum("bjk,bmk->bjm", L, L)

    return covs


def estimate_angle_pdf(
    pred_6d_mean: np.ndarray,
    pred_6d_cov: np.ndarray,
    n_samples: int = 1000,
    theta_bins: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Estimate the angle PDF from the 6d covariance matrix.

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
    seed (int | None)
        Optional seed for the Monte-Carlo draw; `None` uses a fresh default-RNG
        state (non-reproducible). Passing a fixed seed makes the estimated angle
        PDF -- and the downstream assignment -- reproducible across runs.

    Returns
    -------
    (tuple[np.ndarray, np.ndarray])
        Tuple containing the angle bins and the estimated PDF.
    """

    if theta_bins is None:
        theta_bins = np.linspace(-np.pi, np.pi, 46, endpoint=True)

    rng = np.random.default_rng(seed)
    gaussian_rv = rng.multivariate_normal(
        mean=pred_6d_mean, cov=pred_6d_cov, size=n_samples
    )
    angles = np.arctan2(
        gaussian_rv[:, 1] - gaussian_rv[:, 4], gaussian_rv[:, 0] - gaussian_rv[:, 3]
    )
    angle_pdf, _ = np.histogram(angles, bins=theta_bins, density=False)
    angle_pdf = angle_pdf / angle_pdf.sum()

    return theta_bins, angle_pdf


def get_confidence_set(
    pdf: np.ndarray, confidence_level: float
) -> np.ndarray:
    """
    Description
    -----------
    Get the confidence set for the given PDF. Makes no assumptions about
    the location of the origin in the arena.

    Parameters
    ----------
    pdf (np.ndarray)
        Probability density function. Shape: (y_res, x_res, angle_res)
    confidence_level (float)
        Confidence level for the confidence set. Should be between 0 and 1.

    Returns
    -------
    (np.ndarray)
        Boolean array of the same shape as pdf, indicating the confidence set.
    """

    orig_shape = pdf.shape
    flat_pdf = pdf.flatten()
    sorted_indices = np.argsort(flat_pdf)[::-1]
    sorted_pdf = flat_pdf[sorted_indices]
    cumsum = np.cumsum(sorted_pdf)
    # Include the cell that crosses the confidence threshold, not just the
    # cells strictly below it. A cell is in the set iff the cumulative mass
    # *before* it (cumsum minus the cell's own mass) is still below the level;
    # this is the minimal high-density region whose total mass is >= the
    # requested confidence_level. The old `cumsum < confidence_level` mask
    # dropped the boundary-crossing cell, so the realised coverage was always
    # one cell short of the nominal level (under-coverage). This convention
    # still yields the empty set at confidence_level == 0 (the first cell's
    # pre-mass of 0 is not < 0) and the full set at confidence_level == 1.
    idx_in_confidence_set = sorted_indices[(cumsum - sorted_pdf) < confidence_level]
    confidence_set = np.zeros_like(flat_pdf, dtype=bool)
    confidence_set[idx_in_confidence_set] = True

    return confidence_set.reshape(orig_shape)


def make_xy_grid(
    arena_dims: np.ndarray,
    render_dims: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
    Generates a grid of points for evaluating a PMF.
    Places origin in the bottom left corner of the arena

    Parameters
    ----------
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.
    render_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the render.

    Returns
    -------
    (np.ndarray)
        A grid of points for evaluating a PMF. Shape: (y_res, x_res, 2)
    """

    test_points = np.stack(
        np.meshgrid(
            np.linspace(-arena_dims[0] / 2, arena_dims[0] / 2, render_dims[0]),
            np.linspace(-arena_dims[1] / 2, arena_dims[1] / 2, render_dims[1]),
        ),
        axis=-1,
    )

    return test_points


def convert_from_arb(
    output: np.ndarray, arena_dims: np.ndarray
) -> np.ndarray:
    """
    Description
    -----------
    Converts the output from the model to a more interpretable format.

    Parameters
    ----------
    output (np.ndarray)
        Output from the model. Shape: (B, 6)
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.

    Returns
    -------
    (np.ndarray)
        Converted output. Shape: (B, 6)
    """

    scale_factor = arena_dims.max() / 2
    output = output * scale_factor

    return output


def get_conf_sets_6d(
    raw_output: np.ndarray,
    arena_dims_mm: np.ndarray,
    temperature: float = 1.0,
    return_pdf: bool = False,
) -> tuple:
    """
    Description
    -----------
    Computes confidence sets and optionally PDFs from raw output.

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

    Returns
    -------
    (tuple)
        Contains confidence sets and optionally PDFs.
    """

    pred_means_mm = convert_from_arb(raw_output[:, :6], arena_dims=arena_dims_mm)
    pred_cov_6d_mm = compute_covs_6d(raw_output, arena_dims_mm) * temperature
    points_spatial = make_xy_grid(arena_dims_mm, (100, 100))
    bins_angular = np.linspace(-np.pi, np.pi, 46, endpoint=True)
    points_angular = 0.5 * (bins_angular[1:] + bins_angular[:-1])  # Bin centers

    def routine(mean_6d, cov_6d):
        """
        Description
        -----------
        Per-vocalization worker that builds the joint spatial/angular PDF
        from a 6D mean and covariance, then extracts 95% confidence sets
        with and without the angular marginal.

        Parameters
        ----------
        mean_6d (np.ndarray)
            A (6,) shape vector: (x, y, cos_yaw, sin_yaw, cos_roll, sin_roll).
        cov_6d (np.ndarray)
            A (6, 6) covariance matrix for mean_6d.

        Returns
        -------
        (tuple)
            (conf_set, conf_set_no_angle, total_pdf) where
            conf_set / conf_set_no_angle are boolean masks over the spatial grid
            and total_pdf is the full 3D PDF over (x, y, angle).
        """

        _, est_angle_pdf = estimate_angle_pdf(
            mean_6d, cov_6d, n_samples=500, theta_bins=bins_angular, seed=_ANGLE_PDF_SEED
        )
        total_pdf = eval_pdf_with_angle(
            points_spatial=points_spatial,
            points_angular=points_angular,
            mean_2d=mean_6d[:2],
            cov_2d=cov_6d[:2, :2],
            histogram=est_angle_pdf,
        )
        conf_set = get_confidence_set(total_pdf, 0.95)
        conf_set_no_angle = get_confidence_set(total_pdf.sum(axis=-1), 0.95)
        return conf_set, conf_set_no_angle, total_pdf

    results = Parallel(n_jobs=-1)(
        delayed(routine)(mean_6d, cov_6d)
        for mean_6d, cov_6d in tqdm(
            zip(pred_means_mm, pred_cov_6d_mm, strict=False), total=len(pred_means_mm)
        )
    )

    conf_sets = np.array([result[0] for result in results])
    conf_sets_noangle = np.array([result[1] for result in results])

    if return_pdf:
        pdfs = np.array([result[2] for result in results])
        return conf_sets, conf_sets_noangle, pdfs

    return conf_sets, conf_sets_noangle


def are_points_in_conf_set(
    confidence_sets: np.ndarray,
    points: np.ndarray,
    arena_dims: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
    Given an array of confidence sets and points, computes whether each point
    is inside the respective confidence set.

    Parameters
    ----------
    confidence_sets (np.ndarray)
        Confidence sets. Shape: (n, y_res, x_res)
    points (np.ndarray)
        Points to test. Shape: (n, n_node, 3)
    arena_dims (np.ndarray)
        A (2,) shape ndarray containing the X and Y dimensions of the arena.

    Returns
    -------
    (np.ndarray)
        Boolean array of shape (n)
    """

    # Point index 0 is the nose and index 1 is the head, so head_to_nose_vecs
    # points from the head toward the nose; its arctan2 gives the heading (yaw).
    head_to_nose_vecs = points[:, 0, :] - points[:, 1, :]
    head_to_nose_yaw = np.arctan2(head_to_nose_vecs[:, 1], head_to_nose_vecs[:, 0])
    # Clip the nose's (x, y) into the arena bounds so digitize cannot produce an
    # out-of-range index; the angle is intentionally left unclipped because its
    # bins already span the full [-pi, pi) circle.
    nose_points = np.clip(points[:, 0, :2], -arena_dims[:2] / 2, arena_dims[:2] / 2)
    # The PDF (and hence the confidence set) is *sampled at* the 100 grid
    # coordinates `linspace(-dim/2, dim/2, 100)` along each spatial axis (see
    # make_xy_grid in get_conf_sets_6d), so confidence_set[..., i, ...] is the
    # PDF value at grid coordinate `grid[i]`. To look a query point up against
    # that grid it must be mapped to its NEAREST grid coordinate, which means
    # digitizing against the MIDPOINTS between adjacent grid coordinates -- not
    # against the grid coordinates themselves. Digitizing against the grid
    # coordinates (the previous behaviour) shifted every point by half a cell,
    # so a point sitting just past a grid coordinate was assigned the lower cell
    # even though the upper grid coordinate was closer. The angular axis is
    # genuinely binned (a 45-bin histogram with edges `linspace(-pi, pi, 46)`),
    # so it is digitized against those 46 bin edges, matching estimate_angle_pdf.
    y_grid = np.linspace(-arena_dims[1] / 2, arena_dims[1] / 2, 100)
    x_grid = np.linspace(-arena_dims[0] / 2, arena_dims[0] / 2, 100)
    y_bins = 0.5 * (y_grid[1:] + y_grid[:-1])
    x_bins = 0.5 * (x_grid[1:] + x_grid[:-1])
    angle_bins = np.linspace(-np.pi, np.pi, 46, endpoint=True)
    # Digitizing a (clipped) point against the 99 midpoints returns its nearest
    # grid index directly in [0, 99] -- NO -1 offset (the -1 only applied to the
    # old grid-coordinate-as-edge scheme). A point below the first midpoint maps
    # to grid cell 0; one above the last maps to grid cell 99.
    y_bin_indices = np.digitize(nose_points[:, 1], y_bins)
    x_bin_indices = np.digitize(nose_points[:, 0], x_bins)
    # The angular axis carries 45 bins; digitizing against the 46 histogram
    # edges yields indices 0..45, where index 45 only occurs for yaw == +pi
    # (the closed right edge of the last bin). Fold that single boundary value
    # back into the last bin so the index never runs past the 45-wide axis.
    angle_bin_indices = np.digitize(head_to_nose_yaw, angle_bins) - 1
    angle_bin_indices = np.clip(angle_bin_indices, 0, angle_bins.shape[0] - 2)

    if len(confidence_sets.shape) == 3:  # no angles in the confidence set
        in_set = confidence_sets[
            np.arange(confidence_sets.shape[0]), y_bin_indices, x_bin_indices
        ]
    elif len(confidence_sets.shape) == 4:  # has angles
        in_set = confidence_sets[
            np.arange(len(confidence_sets)),
            y_bin_indices,
            x_bin_indices,
            angle_bin_indices,
        ]
    else:
        raise ValueError("Invalid confidence set shape")

    return in_set
