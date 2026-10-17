"""
@author: bartulem
Defines basis functions for temporal filtering in modeling analyses.
Code adapted from Jan Clemens' lab:
https://github.com/janclemenslab/glm_utils/blob/master/src/glm_utils/bases.py
"""

from __future__ import annotations

import copy

import numpy as np
import scipy.interpolate as si
from pygam.utils import b_spline_basis


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
    if not B:
        raise ValueError(f'laplacian_pyramid produced no basis vectors for width={width}, levels={levels}, step={step}')
    return np.stack(B).T


def _nlin(x: np.ndarray) -> np.ndarray:
    """
    Logarithmic non-linear transform.

    Computes ``log(x + 1e-20)``; the tiny offset keeps ``x = 0`` finite (about
    ``-46.05``) instead of ``-inf``. Used to place the raised-cosine centres on a
    log-stretched time axis.

    Parameters
    ----------
    x : np.ndarray
        Non-negative input values (e.g. time bins shifted by the offset ``b``).

    Returns
    -------
    y : np.ndarray
        ``log(x + 1e-20)``, the same shape as ``x``.
    """
    return np.log(x + 1e-20)


def _invnl(x: np.ndarray) -> np.ndarray:
    """
    Inverse of the logarithmic non-linear transform.

    Computes ``exp(x) - 1e-20``, which undoes :func:`_nlin` exactly.

    Parameters
    ----------
    x : np.ndarray
        Values on the log-stretched axis.

    Returns
    -------
    y : np.ndarray
        ``exp(x) - 1e-20``, the same shape as ``x``.
    """
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
    B = A / np.sqrt(np.sum(A ** 2, axis=0))
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
        Desired number of time points (window length). Pads or discards as needed.
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

    # si.splrep returns the full knot vector (longer than the input positions) and the
    # spline order k; these intentionally overwrite the input positions/degree names with
    # splrep's own (semantically different) outputs used for evaluation below.
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


def gam_bspline_basis(width: int, n_splines: int, spline_order: int) -> np.ndarray:
    """
    Description
    -----------
    The B-spline basis pyGAM builds for a spline term over a time axis of
    ``width`` frames: ``n_splines`` B-splines of degree ``spline_order`` on
    knots spaced evenly between the first and the last frame (pyGAM's
    ``basis='ps'`` with the edge knots at the data range), evaluated at every
    frame with pyGAM's own ``b_spline_basis``. It is exactly the lag-axis basis
    of the bout-onset GAM's ``te(value, lag)`` term, so a filter fitted on it has
    the GAM's temporal resolution. Every row sums to 1.

    Parameters
    ----------
    width : int
        Number of frames (lags) of the time axis.
    n_splines : int
        Number of B-splines (at least ``spline_order + 1``).
    spline_order : int
        Polynomial degree of the B-splines (3 = cubic, pyGAM's default).

    Returns
    -------
    basis_matrix : np.ndarray
        ``(width, n_splines)`` basis, one column per B-spline.

    Raises
    ------
    ValueError
        ``n_splines`` is smaller than ``spline_order + 1`` or ``width`` is below 2.
    """

    if width < 2:
        error_message = f"gam_bspline_basis needs a time axis of at least 2 frames; got width={width}."
        raise ValueError(error_message)
    if n_splines < spline_order + 1:
        error_message = f"gam_bspline_basis needs n_splines >= spline_order + 1; got n_splines={n_splines}, spline_order={spline_order}."
        raise ValueError(error_message)
    frames = np.arange(width, dtype=np.float64)
    return np.asarray(b_spline_basis(frames, edge_knots=np.array([0.0, width - 1.0]), n_splines=n_splines,
                                     spline_order=spline_order, sparse=False, periodic=False, verbose=False), dtype=np.float64)


TEMPORAL_BASIS_TYPES = ("none", "bspline")


def resolve_temporal_basis(model_block: dict, history_frames: int) -> tuple[np.ndarray | None, dict]:
    """
    Description
    -----------
    Resolves the temporal representation of a JAX linear model's filters from its
    settings block (``hyperparameters.linear_models.multinomial_logistic`` or
    ``.manifold_regression``) and returns the hyperparameters the fit actually uses.

    * ``temporal_basis.type == "none"``: the filter has one free weight per frame,
      penalised by the block's own L2 and finite-difference smoothness settings
      (``lambda_smooth_fixed``, ``l2_reg_fixed``, ``smoothness_derivative_order``,
      reflective edge rows for order 2). The block is returned unchanged apart from
      ``smoothness_reflective_edges = True``.
    * ``temporal_basis.type == "bspline"`` (both blocks' shipped default; for the multinomial model the same selection and score as the per-frame basis on the same inputs, a fiftieth of the fitting time, and no fold lost to float32 softmax saturation of the raw-lag fit; for the torus model the per-frame fit cannot recover a planted filter shape, see below): every feature's history of
      ``history_frames`` frames is projected onto ``n_splines`` B-splines of degree
      ``spline_order`` (:func:`gam_bspline_basis`, pyGAM's lag basis, divided by
      ``history_frames`` so each projected column is a weighted average of the
      history, as the GAM averages its per-lag contributions per event, which keeps
      the inputs on the features' scale), so the model fits ``n_splines``
      coefficients per feature, and the penalty mirrors pyGAM's:
      the second-order difference penalty on neighbouring spline coefficients with
      open boundaries, at strength ``temporal_basis.lambda_smooth_fixed`` (tuned over
      ``temporal_basis.lambda_smooth_decades_each_side`` decades when the block's
      ``tune_regularization_bool`` is set), plus an L2 penalty on the spline
      coefficients at ``temporal_basis.l2_reg_fixed`` (fixed; its tuning grid is
      pinned to that value). With ``temporal_basis.l2_lag_alpha`` > 0 that L2 grows
      with the spline's distance from onset, ``l2 * (1 + alpha * lag**2)`` (``lag``
      the spline centre's distance from onset as a fraction of the window), the
      lag-varying ridge of penalised distributed-lag models: it is implemented by
      scaling each basis column by ``1 / sqrt(1 + alpha * lag**2)``, so a uniform L2
      on the coefficients of the scaled basis is exactly the lag-varying L2 on the
      unscaled ones, and the same scaled basis maps coefficients back to frame
      weights (the stored ``basis_coefficients`` are those of the scaled basis; the
      second-difference penalty acts on them too).
      Frame binning cannot be combined with it.

      The multinomial default strength, ``lambda_smooth_fixed = 1e-5`` tuned over
      +/- 2 decades (1e-7 to 1e-3), comes from a penalty pilot on the intact-partner
      male cohort (top screened feature ``allo_yaw-head``, 10 session folds): the
      cross-validated macro-AUC is flat from no penalty up to 1e-4 (paired change
      +0.0003 +/- 0.0004 at 1e-5) and falls beyond it (-0.0022 at 1e-3, -0.0106 at
      1e-1, where the second-order penalty has straightened every filter into a
      line); 1e-5 is the largest strength with no measurable loss and it damps the
      unpenalised fit's wiggle at the oldest lags. The averaging projection makes
      the coefficients ~``history_frames`` times larger than raw per-frame sums, so
      useful strengths sit far below the per-frame model's ``lambda_smooth_fixed``.
      The torus (manifold) default, ``lambda_smooth_fixed = 1e-8`` tuned over
      +/- 2 decades (1e-10 to 1e-6), comes from the same pilot design (top screened
      feature ``self.neck_elevation``): every strength costs some macro von Mises
      log-score, monotonically (paired change -0.00003 +/- 0.00001 at 1e-8, -0.00027
      at 1e-7, -0.0022 at 1e-5, -0.0084 from 1 upward), because the signal is a sharp
      rise over the last ~0.3 s that the second-difference penalty flattens, and the
      unpenalised fit has no edge wiggle to remove; 1e-8 is the largest strength
      whose loss is negligible (~0.4 % of the feature's margin over the null).
      The shipped ``n_splines`` (20 for both models) and the torus model's
      ``l2_reg_fixed`` (1e-4; the multinomial model's is set below) come from a basis
      experiment on the same cohort (development sessions, 5 session-grouped folds),
      scoring each configuration by (i) the cross-validated score of the top features,
      (ii) the filter weight a within-session-PERMUTED (no-signal) fit puts at the
      window ends, relative to the mid-window median, and (iii) the recovery of
      PLANTED filters of known shape (bumps at -3 s and -1.5 s, ramps over the last
      0.3 s and 1 s) generated through the real neck-elevation histories. On the
      torus, 8 splines with no L2 put 7.3x the mid-window weight at -4 s and 11x at
      0 s in a no-signal fit (the near-unpenalised end coefficients absorb noise, and
      the rise follows the window edge when the window is shortened); an L2 of 1e-4
      cuts that to ~2-3x, recovers every planted shape at r >= 0.90 with 20 splines,
      and costs ~9 % of the cross-validated gain; the per-frame fit recovers none of
      them (r -0.18 to 0.36). For the multinomial model 20 splines gave the highest
      cross-validated macro-AUC and recovered every planted shape at r >= 0.96; a
      uniform L2 only cost score there, and its L2 is set with the lag-varying ridge
      below. A residual 2-4x end weight remains in every configuration, so fitted filters are
      read against their permuted null.
      The torus model's ``l2_lag_alpha = 10`` comes from
      a follow-up on the window's far edge: with a uniform L2 the torus filters still
      rose at -4 s (inter-partner distance 1.04 vs a permuted 95th percentile of 0.67,
      x1e-3), and neither a clamped basis nor a 1000x stronger smoothness changed it,
      while the lag-varying ridge at alpha 10 brought the far-edge weight of head pitch
      and neck elevation to their permuted level (0.22 / 0.26 and 0.15 / 0.17), kept
      the planted-shape recovery (r >= 0.87, a bump at -3 s included) and the onset
      rise, and cost ~0 cross-validated gain except for inter-partner distance (-7 %),
      whose oldest frames carry real information (history cut off by the window).
      The multinomial model's ``l2_reg_fixed = 1e-5`` with ``l2_lag_alpha = 100`` comes
      from the same test: its unpenalised spline put a no-signal spike at -4 s
      (inter-partner distance 15.1 vs an onset peak of 22.1, x1e-3); this setting cuts it
      to 0.76 vs 8.8, costs at most 4 % of the cross-validated macro-AUC and keeps the
      planted-shape recovery at r >= 0.89 (a bump at -3 s 0.91). The history window
      stays at 4 s: with an 8 s re-extraction inter-partner distance and head pitch
      predict ~40 % better (the residual far-edge weight is history before the window,
      with no edge rise in the longer window) while neck elevation, which acts in the
      last second, loses 14 %.

    Parameters
    ----------
    model_block (dict)
        The model's settings block, holding a ``temporal_basis`` sub-block
        (``type``, ``n_splines``, ``spline_order``, ``lambda_smooth_fixed``,
        ``lambda_smooth_decades_each_side``).
    history_frames (int)
        Frames of the history window (the length of every feature's filter).

    Returns
    -------
    basis (np.ndarray | None)
        ``(history_frames, n_splines)`` projection basis (pyGAM's B-splines divided
        by ``history_frames``; the same matrix turns fitted coefficients back into
        frame weights), or None for ``"none"``.
    effective_block (dict)
        A copy of ``model_block`` with the penalty settings the fit uses
        (``lambda_smooth_fixed``, ``l2_reg_fixed``, ``smoothness_derivative_order``,
        the tuning decades) and ``smoothness_reflective_edges``.

    Raises
    ------
    ValueError
        An unknown ``temporal_basis.type``, or ``"bspline"`` with a
        ``bin_resizing_factor`` other than 1.
    """

    temporal_basis = model_block["temporal_basis"]
    basis_type = temporal_basis["type"]
    if basis_type not in TEMPORAL_BASIS_TYPES:
        error_message = f"temporal_basis.type must be one of {TEMPORAL_BASIS_TYPES}; got {basis_type!r}."
        raise ValueError(error_message)
    effective_block = copy.deepcopy(model_block)
    if basis_type == "none":
        effective_block["smoothness_reflective_edges"] = True
        return None, effective_block
    if int(model_block["bin_resizing_factor"]) != 1:
        error_message = ("temporal_basis.type 'bspline' projects the full-resolution history onto B-splines and cannot be "
                         f"combined with bin_resizing_factor={model_block['bin_resizing_factor']}; set it to 1.")
        raise ValueError(error_message)
    # Divided by the window length so each projected column is a weighted AVERAGE of the
    # history (the GAM likewise averages its per-lag contributions per event), keeping the
    # inputs on the features' z-score scale: a plain sum over the ~history/n_splines frames
    # under each bump inflates them ~that many times, saturating the softmax.
    basis = gam_bspline_basis(int(history_frames), int(temporal_basis["n_splines"]), int(temporal_basis["spline_order"])) / float(history_frames)
    lag_alpha = float(temporal_basis["l2_lag_alpha"])
    if lag_alpha > 0.0:
        frames_axis = np.arange(int(history_frames), dtype=np.float64)
        centres = (frames_axis[:, None] * basis).sum(axis=0) / basis.sum(axis=0)
        lag_fraction = np.clip((float(history_frames) - 1.0 - centres) / (float(history_frames) - 1.0), 0.0, 1.0)
        basis = basis / np.sqrt(1.0 + lag_alpha * lag_fraction ** 2)[None, :]
    effective_block["lambda_smooth_fixed"] = float(temporal_basis["lambda_smooth_fixed"])
    effective_block["l2_reg_fixed"] = float(temporal_basis["l2_reg_fixed"])
    effective_block["smoothness_derivative_order"] = 2
    effective_block["smoothness_reflective_edges"] = False
    effective_block["tune_regularization_params"]["lambda_smooth_decades_each_side"] = int(temporal_basis["lambda_smooth_decades_each_side"])
    effective_block["tune_regularization_params"]["l2_reg_decades_each_side"] = 0
    return basis, effective_block


def project_history_onto_basis(history: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Projects one feature's history matrix onto a temporal basis: each event's
    ``history_frames`` values become one weight per basis function
    (``history @ basis``), so a filter fitted on the projected columns is the
    basis combination ``basis @ coefficients`` on the frame axis.

    Parameters
    ----------
    history (np.ndarray)
        ``(n_events, history_frames)`` feature history.
    basis (np.ndarray)
        ``(history_frames, n_basis)`` basis.

    Returns
    -------
    projected (np.ndarray)
        ``(n_events, n_basis)`` projected history.

    Raises
    ------
    ValueError
        The history length differs from the basis length.
    """

    if history.shape[1] != basis.shape[0]:
        error_message = f"history has {history.shape[1]} frames but the basis covers {basis.shape[0]}."
        raise ValueError(error_message)
    return history @ basis


def basis_coefficients_to_frames(coefficients: np.ndarray, basis: np.ndarray, n_features: int, axis: int) -> np.ndarray:
    """
    Description
    -----------
    Turns fitted basis coefficients back into filters on the frame axis. Along
    ``axis`` the coefficients are laid out feature by feature (``n_features``
    blocks of ``n_basis`` values, the order of the projected design matrix);
    each block becomes ``basis @ block``, so that axis grows from
    ``n_features * n_basis`` to ``n_features * history_frames`` and every
    downstream reader of the weights sees the same layout as a full-resolution fit.

    Parameters
    ----------
    coefficients (np.ndarray)
        Fitted weights, e.g. the multinomial ``coef_`` ``(n_classes,
        n_features * n_basis)`` (``axis=1``) or the torus ``coef_``
        ``(n_features * n_basis, n_outputs)`` (``axis=0``).
    basis (np.ndarray)
        ``(history_frames, n_basis)`` basis the design matrix was projected on.
    n_features (int)
        Number of features.
    axis (int)
        Axis holding the ``n_features * n_basis`` coefficients.

    Returns
    -------
    frames (np.ndarray)
        The weights with that axis of length ``n_features * history_frames``.

    Raises
    ------
    ValueError
        The axis length is not ``n_features * n_basis``.
    """

    coefficients = np.asarray(coefficients, dtype=np.float64)
    history_frames, n_basis = basis.shape
    if coefficients.shape[axis] != n_features * n_basis:
        error_message = (f"axis {axis} holds {coefficients.shape[axis]} coefficients, expected "
                         f"n_features * n_basis = {n_features} * {n_basis}.")
        raise ValueError(error_message)
    moved = np.moveaxis(coefficients, axis, -1)
    blocks = moved.reshape(*moved.shape[:-1], n_features, n_basis)
    frames = (blocks @ basis.T).reshape(*moved.shape[:-1], n_features * history_frames)
    return np.moveaxis(frames, -1, axis)
