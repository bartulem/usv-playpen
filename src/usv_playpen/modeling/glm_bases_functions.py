"""
Defines basis functions.
Code adapted from Jan Clemens lab:
https://github.com/janclemenslab/glm_utils/blob/master/src/glm_utils/bases.py
"""

import numpy as np
import scipy.interpolate as si
import scipy.linalg


def laplacian_pyramid(width, levels, step, fwhm, normalize=True):
    """
    Get a 1d Laplacian pyramid basis matrix.
        Args:
            width (int): Time span of the basis functions.
            levels (int): Number of levels.
            step (float): Spacing of levels (1= regular Laplacian pyramid, 0.5 = Laplacian pyramid with half-levels).
                          At each full-step, the width of the Gaussian is doubled
            fwhm (float): Full width at half-max for the Gaussians at level 1 (the finest level).
            normalize (boolean, optional): Normalize each basis to unit L2 norm. Defaults to True.
        Returns:
            [time, bases] - np matrix with basis functions.
        Adapted from:
            Mineault, P. J., Barthelmé, S. & Pack, C. C.
            Improved classification images with sparse priors in a smooth basis.
            Journal of Vision 9, 1–24 (2009).
    """

    B = list()
    rg = np.arange(0, width)
    for ii in np.arange(0, levels, step, dtype=float):
        cens = 2 ** (ii - 2) + np.arange(int(width / (2 ** (ii - 1)) - 1)) * (2 ** (ii - 1))
        if len(cens):  # check if there are any basis functions for that level
            cens = np.floor((width - (np.max(cens) - np.min(cens) + 1)) / 2 + cens) + 1
            gwidth = 2 ** (ii - 1) / 2.35 * fwhm
            for jj in range(1, len(cens)):
                v = np.exp(-(rg - cens[jj]) ** 2 / 2 / gwidth ** 2)
                if normalize:
                    v = v / np.linalg.norm(v)
                B.append(v)
    return np.stack(B).T


def _nlin(x): return np.log(x + 1e-20)


def _invnl(x): return np.exp(x) - 1e-20


def _ff(x, c, db):
    """
    Gives the raised cosine of data points `x` with classified centers
    `c` with spacing between cosine peaks `db`.
    """
    kbasis = (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / db / 2))) + 1) / 2
    return kbasis


def _normalizecols(A):
    """ Normalize the columns of a 2D array."""
    B = A / np.tile(np.sqrt(sum(A ** 2, 0)), (np.size(A, 0), 1))
    B = np.nan_to_num(B)  # To get rid of nans out of zero divisions
    return B


def raised_cosine(neye, ncos, kpeaks, b, w=None, nbasis=None):
    """
    Creates and plots basis of raised cosines. Adapted from Weber & Pillow 2017.
    link:
        https://github.com/aiweber/GLM_and_Izhikevich/blob/master/makeBasis_StimKernel.m
    Parameters
    ----------
    neye : int
            Number of identity basis vectors at front. It defines the number of
            first columns which has identity matrix for dense sampling of data
            just preceding the event.
    ncos : int
            Number of vectors that are raised cosines. Cannot be 0 or negative.
    kpeaks : list
            Peak positions of 1st and last cosines relative to the start
            of cosine basis vectors (e.g. [0 10])
    b : int
            Offset for nonlinear scaling.  larger values -> more linear
            scaling of vectors.
    w : int, optional
            Desired number of time points (e.g. window length) of the bases.
            It must be same as the window of time delay embedded data.
            When a value is not given it assigns w as the full time length of the
            basis kernel.
    nbasis : int, optional
            Desired number of basis vectors
    Returns
    -------
    kbasis : ndarray [time, bases]
        A basis of raised cosines as columns.
    """

    kpeaks = np.array(kpeaks)

    yrnge = _nlin(kpeaks + b)  # no_nlinear transform, b is no_nlinearity of scaling

    db = (yrnge[1] - yrnge[0]) / (ncos - 1)  # spacing between cosine peaks
    ctrs = np.linspace(yrnge[0], yrnge[1], ncos)  # centers for cosine peaks

    # mxt is for the kernel, without the no_nlinear transform
    mxt = _invnl(yrnge[1] + 2 * db) - b  # max time bin
    kt = np.arange(0, mxt)  # kernel time points/ no no_nlinear transform yet
    nt = len(kt)  # number of kernel time points

    # Now we transform kernel time points through no_nlinearity and tile them
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


def bsplines(width, positions, degree: int = 3, periodic: bool = False):
    """
    Get basis matrix for b-splines.
    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep
    Usage: B = get_bspline_basis(width=100, positions=np.arange(5, 100, 10))
    Args:
        width (array-like): Time span over which the splines are evaluated.
        positions (list-like): Positions of individual basis functions.
        degree (int): Polynomial degree of the splines. Defaults to 3.
        periodic (bool): . Defaults to False.
    Returns:
        [time, bases] matrix with basis functions
    """
    t = np.arange(width)
    n_positions = len(positions)
    y_dummy = np.zeros(n_positions)

    positions, coe_ffs, degree = si.splrep(positions,
                                           y_dummy,
                                           k=degree,
                                           per=periodic)
    ncoe_ffs = len(coe_ffs)
    bsplines = []
    for i_spline in range(n_positions):
        coe_ffs = [1.0 if ispl == i_spline else 0.0 for ispl in range(ncoe_ffs)]
        bsplines.append((positions, coe_ffs, degree))

    B = np.array([si.splev(t, spline) for spline in bsplines])
    B = B[:, ::-1].T  # invert so bases "begin" at the right and transpose to [time x bases]
    return B


def multifeature_basis(B, nb_features: int = 1):
    """
    Get block diagonal matrix from a 2-D matrix (B) repeated once per feature.
    Args:
        B ([type]): 2-D matrix with basis functions.
        nb_features (int, optional): number of features. Defaults to 1.
    Returns:
        [type]: block diagonal matrix
    """
    return scipy.linalg.block_diag(*[B for ii in range(nb_features)])


def identity(width):
    return np.identity(width)[::-1, :]


def comb(width, spacing):
    return trivial_spacing(width, spacing)


def trivial_spacing(width, spacing):
    """
    Trivial base for sampling equally spaced time points.
    Args:
        width (int): Time span of the basis functions.
        spacing (int): space between sampled time points.
    Returns:
        [time, bases] - np matrix with basis functions.
    """

    spaced_base = np.zeros((width, width // spacing))
    for ii in range(width // spacing):
        spaced_base[-ii * spacing - 1, ii] = 1

    return spaced_base
