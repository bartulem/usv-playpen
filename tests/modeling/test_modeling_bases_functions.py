"""
@author: bartulem
Unit tests for the temporal basis-function generators in
``usv_playpen.modeling.modeling_bases_functions`` (adapted from the
Clemens-lab GLM basis utilities).

These bases tile a temporal-filter window for GLM/GAM fits, so the
properties that matter downstream are structural: column-normalised
raised cosines, the flipped identity time ordering, and the
multi-resolution Gaussian levels of the Laplacian pyramid. Each test
asserts such an analytic invariant on a small synthetic window rather
than pinning a frozen matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.modeling.modeling_bases_functions import (
    _ff,
    _invnl,
    _nlin,
    _normalizecols,
    bsplines,
    identity,
    laplacian_pyramid,
    raised_cosine,
)


# Non-linear log transform pair


class TestNlinInvnl:

    def test_round_trip_recovers_input(self):
        """``_invnl(_nlin(x))`` returns ``x`` for positive inputs (the
        ``1e-20`` epsilon is far below any biological magnitude)."""

        x = np.array([0.5, 1.0, 5.0, 100.0])
        np.testing.assert_allclose(_invnl(_nlin(x)), x, rtol=1e-9)

    def test_nlin_is_monotone_increasing(self):
        """The log transform preserves ordering."""

        x = np.linspace(0.1, 10.0, 50)
        assert np.all(np.diff(_nlin(x)) > 0)


# _ff (raised cosine values)


class TestRaisedCosineValues:

    def test_values_bounded_in_unit_interval(self):
        """Raised-cosine bumps are bounded in ``[0, 1]`` everywhere."""

        x = np.linspace(-10.0, 10.0, 201)
        vals = _ff(x, c=np.array(0.0), db=2.0)
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)

    def test_peak_at_centre(self):
        """The bump attains its maximum of 1 at the centre ``c``."""

        x = np.linspace(-5.0, 5.0, 1001)
        c = 1.0
        vals = _ff(x, c=np.array(c), db=2.0)
        assert vals[np.argmax(vals)] == pytest.approx(1.0, abs=1e-6)
        assert x[np.argmax(vals)] == pytest.approx(c, abs=0.02)


# _normalizecols


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
class TestNormalizeCols:

    def test_columns_have_unit_l2_norm(self):
        """Every non-zero column is rescaled to unit L2 norm."""

        A = np.array([[1.0, 0.0, 2.0],
                      [2.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0]])
        B = _normalizecols(A)
        col_norms = np.linalg.norm(B, axis=0)
        # Column 1 is all-zero -> stays zero (nan_to_num); others unit.
        np.testing.assert_allclose(col_norms, np.array([1.0, 0.0, 1.0]), atol=1e-12)

    def test_zero_column_is_zeroed_not_nan(self):
        """An all-zero column divides by zero internally but is mapped to
        zeros, never NaN/inf, by the ``nan_to_num`` guard."""

        A = np.zeros((4, 2))
        B = _normalizecols(A)
        assert np.all(np.isfinite(B))
        assert np.all(B == 0.0)


# raised_cosine basis


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
class TestRaisedCosine:

    def test_shape_respects_w_and_nbasis(self):
        """Requesting ``w`` time points and ``nbasis`` bases yields a
        matrix of exactly that shape."""

        B = raised_cosine(neye=2, ncos=5, kpeaks=[0, 10], b=2, w=30, nbasis=8)
        assert B.shape == (30, 8)

    def test_columns_normalised_to_unit_norm(self):
        """The final step column-normalises, so every populated basis
        column has unit L2 norm (zero-padded columns stay zero)."""

        B = raised_cosine(neye=2, ncos=5, kpeaks=[0, 10], b=2, w=30, nbasis=8)
        col_norms = np.linalg.norm(B, axis=0)
        for n in col_norms:
            assert n == pytest.approx(1.0, abs=1e-9) or n == pytest.approx(0.0, abs=1e-12)

    def test_values_are_finite(self):
        """No NaN/inf survive the nonlinear-time construction."""

        B = raised_cosine(neye=1, ncos=4, kpeaks=[0, 8], b=1)
        assert np.all(np.isfinite(B))


# bsplines basis


class TestBSplines:

    def test_shape_time_by_bases(self):
        """A B-spline basis spans ``width`` time points with one column
        per requested knot position."""

        positions = [0, 5, 10, 15, 20]
        B = bsplines(width=25, positions=positions, degree=3)
        assert B.shape == (25, len(positions))

    def test_partition_of_unity_in_interior(self):
        """A clamped cubic B-spline basis forms a partition of unity:
        the columns sum to ~1 across the interior of the span."""

        positions = [0, 5, 10, 15, 20]
        B = bsplines(width=25, positions=positions, degree=3)
        row_sums = B.sum(axis=1)
        interior = row_sums[5:20]
        np.testing.assert_allclose(interior, np.ones_like(interior), atol=1e-6)


# identity


class TestIdentityBasis:

    def test_is_flipped_identity(self):
        """``identity`` returns the row-flipped identity so column 0
        corresponds to the frame nearest the event."""

        out = identity(4)
        np.testing.assert_allclose(out, np.identity(4)[::-1, :])
        assert out.shape == (4, 4)


# laplacian_pyramid


class TestLaplacianPyramid:

    def test_shape_is_time_by_bases(self):
        """The pyramid returns a 2-D ``[time, bases]`` matrix spanning
        the requested width with at least one basis."""

        B = laplacian_pyramid(width=64, levels=4, step=1.0, fwhm=1.0)
        assert B.ndim == 2
        assert B.shape[0] == 64
        assert B.shape[1] >= 1

    def test_normalized_columns_have_unit_norm(self):
        """With ``normalize=True`` each Gaussian basis column is unit
        L2 norm."""

        B = laplacian_pyramid(width=64, levels=4, step=1.0, fwhm=1.0, normalize=True)
        col_norms = np.linalg.norm(B, axis=0)
        np.testing.assert_allclose(col_norms, np.ones_like(col_norms), atol=1e-9)
