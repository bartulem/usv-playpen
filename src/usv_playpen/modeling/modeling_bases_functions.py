"""
@author: bartulem
Defines basis functions for temporal filtering in GLM analyses.
Code adapted from Jan Clemens' lab:
https://github.com/janclemenslab/glm_utils/blob/master/src/glm_utils/bases.py
"""

import numpy as np
import scipy.interpolate as si
import scipy.linalg


def laplacian_pyramid(width: int, levels: int, step: float, fwhm: float, normalize: bool = True) -> np.ndarray:
    """
    Generates a 1D Laplacian pyramid basis matrix.

    The Laplacian pyramid provides a multi-resolution representation of the temporal filter.
    It consists of Gaussians at different scales (levels), where each level doubles the
    width of the Gaussian. Spacing between levels can be adjusted for denser sampling.

    Parameters
    ----------
    width : int
        The temporal span (number of frames) of the basis functions.
    levels : int
        The number of scales/levels in the pyramid.
    step : float
        Spacing between levels (e.g., 1.0 for regular, 0.5 for half-levels).
    fwhm : float
        Full width at half-max for the Gaussians at the finest level (Level 1).
    normalize : bool, optional
        If True, normalizes each basis vector to unit L2 norm. Defaults to True.

    Returns
    -------
    basis_matrix : np.ndarray
        The basis matrix of shape [time, bases].
    """

    B = list()
    rg = np.arange(0, width)
    for ii in np.arange(0, levels, step, dtype=float):
        lvl_minus_2 = float(2 ** (float(ii) - 2.0))
        lvl_minus_1 = float(2 ** (float(ii) - 1.0))

        cens = lvl_minus_2 + np.arange(int(width / lvl_minus_1 - 1)) * lvl_minus_1

        if len(cens):
            cens = np.floor((width - (np.max(cens) - np.min(cens) + 1)) / 2 + cens) + 1
            gwidth = lvl_minus_1 / 2.35 * fwhm
            for jj in range(1, len(cens)):
                v = np.exp(-(rg - cens[jj]) ** 2 / (2 * gwidth ** 2))
                if normalize:
                    v = v / np.linalg.norm(v)
                B.append(v)
    return np.stack(B).T


def _nlin(x: np.ndarray) -> np.ndarray:
    """Logarithmic non-linear transform."""
    return np.log(x + 1e-20)


def _invnl(x: np.ndarray) -> np.ndarray:
    """Inverse of the logarithmic non-linear transform."""
    return np.exp(x) - 1e-20


def _ff(x: np.ndarray, c: np.ndarray, db: float) -> np.ndarray:
    """
    Calculates raised cosine values for given centers and spacing.

    Parameters
    ----------
    x : np.ndarray
        Temporal data points.
    c : np.ndarray
        Centers of the cosine peaks.
    db : float
        Spacing between cosine peaks.

    Returns
    -------
    kbasis : np.ndarray
        The raised cosine values.
    """
    kbasis = (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / db / 2))) + 1) / 2
    return kbasis


def _normalizecols(A: np.ndarray) -> np.ndarray:
    """
    Normalizes the columns of a 2D array to unit L2 norm.

    Parameters
    ----------
    A : np.ndarray
        The input 2D array.

    Returns
    -------
    normalized_A : np.ndarray
        The array with columns normalized to unit length.
    """
    B = A / np.tile(np.sqrt(sum(A ** 2, 0)), (np.size(A, 0), 1))
    B = np.nan_to_num(B)  # To get rid of nans out of zero divisions
    return B


def raised_cosine(neye: int, ncos: int, kpeaks: list, b: int, w: int = None, nbasis: int = None) -> np.ndarray:
    """
    Creates a basis of raised cosines with an optional identity buffer.

    Raised cosines are frequently used in GLMs to represent temporal kernels because
    they allow for non-linear tiling of time (e.g., higher resolution near the event
    and lower resolution further in the past).

    Parameters
    ----------
    neye : int
        Number of identity basis vectors to place at the start (dense sampling).
    ncos : int
        Number of raised cosine vectors.
    kpeaks : list
        Positions of the first and last cosine peaks [start, end].
    b : int
        Offset for non-linear scaling (larger = more linear).
    w : int, optional
        Desired number of time points (window length). Padds or discards as needed.
    nbasis : int, optional
        Desired total number of basis vectors.

    Returns
    -------
    basis_matrix : np.ndarray
        The raised cosine basis matrix of shape [time, bases].
    """

    kpeaks = np.array(kpeaks)

    yrnge = _nlin(kpeaks + b)  # nonlinear transform, b is nonlinearity of scaling

    db = (yrnge[1] - yrnge[0]) / (ncos - 1)  # spacing between cosine peaks
    ctrs = np.linspace(yrnge[0], yrnge[1], ncos)  # centers for cosine peaks

    # mxt is for the kernel, without the nonlinear transform
    mxt = _invnl(yrnge[1] + 2 * db) - b  # max time bin
    kt = np.arange(0, mxt)  # kernel time points/ no nonlinear transform yet
    nt = len(kt)  # number of kernel time points

    # Now we transform kernel time points through nonlinearity and tile them
    e1 = np.tile(_nlin(kt + b), (ncos, 1))
    # Tiling the center points for matrix multiplication
    e2 = np.tile(ctrs, (nt, 1)).T

    # Creating the raised cosines
    kbasis0 = _ff(e1, e2, db)

    # Concatenate identity vectors and create basis kernel (kbasis)
    a1 = np.concatenate((np.eye(neye), np.zeros((nt, neye))), axis=0)
    a2 = np.concatenate((np.zeros((neye, ncos)), kbasis0.T), axis=0)
    kbasis = np.concatenate((a1, a2), axis=1)
    kbasis = np.flipud(kbasis)
    nb = np.size(kbasis, 1)  # number of current bases

    # Modifying number of output bases if nbasis is given
    if nbasis is None:
        pass
    elif nb < nbasis:  # if desired number of bases greater, add more zero bases
        kbasis = np.concatenate((kbasis, np.zeros((kbasis.shape[0],
                                                   nbasis - nb))), axis=1)
    elif nb > nbasis:  # if desired number of bases less, get the front bases
        kbasis = kbasis[:, :nbasis]

    # Modifying number of time points (e.g. window) in the basis kernel. If the w value is
    # greater than basis time points, padding zeros to back in time.
    # If w value is lower than basis points back in time are discarded.
    if w is None:
        pass
    elif w > kbasis.shape[0]:
        kbasis = np.concatenate((np.zeros((w - kbasis.shape[0],
                                           kbasis.shape[1])), kbasis), axis=0)
    elif w < kbasis.shape[0]:
        kbasis = kbasis[-w:, :]

    kbasis = _normalizecols(kbasis)
    return kbasis


def bsplines(width: int, positions: list, degree: int = 3, periodic: bool = False) -> np.ndarray:
    """
    Generates a basis matrix using B-splines.

    B-splines provide a smooth, flexible basis for representing temporal filters.
    The basis functions are defined by their polynomial degree and the positions
    of knots (centers).

    Parameters
    ----------
    width : int
        The temporal span over which the splines are evaluated.
    positions : list
        Positions of the individual basis functions (knots).
    degree : int, optional
        Polynomial degree of the splines. Defaults to 3.
    periodic : bool, optional
        If True, creates periodic splines. Defaults to False.

    Returns
    -------
    basis_matrix : np.ndarray
        The B-spline basis matrix of shape [time, bases].
    """
    t = np.arange(width)
    n_positions = len(positions)
    y_dummy = np.zeros(n_positions)

    positions, coe_ffs, degree = si.splrep(positions,
                                           y_dummy,
                                           k=degree,
                                           per=periodic)
    ncoe_ffs = len(coe_ffs)
    bsplines_list = []
    for i_spline in range(n_positions):
        coe_ffs_val = [1.0 if ispl == i_spline else 0.0 for ispl in range(ncoe_ffs)]
        bsplines_list.append((positions, coe_ffs_val, degree))

    B = np.array([si.splev(t, spline) for spline in bsplines_list])
    B = B[:, ::-1].T  # invert so bases "begin" at the right and transpose to [time x bases]
    return B


def multifeature_basis(B: np.ndarray, nb_features: int = 1) -> np.ndarray:
    """
    Creates a block diagonal basis matrix for multiple features.

    Repeats the input basis matrix `B` along the diagonal, once for each feature,
    to allow independent filtering of multiple behavioral predictors.

    Parameters
    ----------
    B : np.ndarray
        The 2D basis matrix for a single feature.
    nb_features : int, optional
        The number of features to expand the basis for. Defaults to 1.

    Returns
    -------
    block_diagonal_basis : np.ndarray
        A sparse-like block diagonal matrix.
    """

    return scipy.linalg.block_diag(*[B for _ in range(nb_features)])


def identity(width: int) -> np.ndarray:
    """
    Returns an identity matrix as a basis.

    This represents the 'raw' temporal filter where each frame is its own basis
    function. Flipped so that index 0 corresponds to the time point nearest the event.

    Parameters
    ----------
    width : int
        The temporal span (number of frames).

    Returns
    -------
    basis_matrix : np.ndarray
        Identity matrix of shape [width, width].
    """
    return np.identity(width)[::-1, :]


def comb(width: int, spacing: int) -> np.ndarray:
    """
    Alias for trivial_spacing. Samples equally spaced discrete time points.

    Parameters
    ----------
    width : int
        Total time span.
    spacing : int
        Interval between sampled points.

    Returns
    -------
    basis_matrix : np.ndarray
        A sparse basis matrix representing temporal subsampling.
    """
    return trivial_spacing(width, spacing)


def trivial_spacing(width: int, spacing: int) -> np.ndarray:
    """
    Generates a basis for sampling equally spaced temporal points.

    This basis matrix acts as a subsampling operator, selecting specific
    discrete frames at regular intervals relative to the event.

    Parameters
    ----------
    width : int
        Total temporal span of the basis.
    spacing : int
        Frames between each sampled time point.

    Returns
    -------
    basis_matrix : np.ndarray
        Matrix of shape [time, bases] with 1s at sampled indices.
    """

    spaced_base = np.zeros((width, width // spacing))
    for ii in range(width // spacing):
        spaced_base[-ii * spacing - 1, ii] = 1

    return spaced_base
