"""Sparse B-spline basis construction used by MonotoneLSPIA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix


def open_uniform_knots(num_control_points: int, degree: int) -> NDArray[np.float64]:
    """Return an open-uniform knot vector on ``[0, 1]``."""
    if num_control_points <= degree:
        raise ValueError("num_control_points must be greater than degree")
    knots = np.zeros(num_control_points + degree + 1, dtype=float)
    knots[-(degree + 1) :] = 1.0
    num_internal = num_control_points - degree - 1
    if num_internal:
        knots[degree + 1 : degree + 1 + num_internal] = np.arange(
            1, num_internal + 1, dtype=float
        ) / (num_internal + 1)
    return knots


def validate_knots(knots: ArrayLike, num_control_points: int, degree: int) -> NDArray[np.float64]:
    """Validate and return a one-dimensional knot vector."""
    result = np.asarray(knots, dtype=float).reshape(-1)
    expected = num_control_points + degree + 1
    if result.size != expected:
        raise ValueError("knots must contain num_control_points + degree + 1 values")
    if not np.all(np.isfinite(result)):
        raise ValueError("knots must contain only finite values")
    if np.any(np.diff(result) < 0):
        raise ValueError("knots must be nondecreasing")
    if result[0] != 0 or result[-1] != 1 or np.any((result < 0) | (result > 1)):
        raise ValueError("knots must lie on [0, 1] and include both endpoints")
    return result


def basis_matrix(
    parameters: ArrayLike,
    num_control_points: int,
    *,
    degree: int = 3,
    knots: ArrayLike | None = None,
) -> csr_matrix:
    """Construct the sparse open B-spline collocation matrix.

    Parameters are normalized coordinates and must lie on ``[0, 1]``.
    Each row contains at most ``degree + 1`` nonzero values.
    """
    t = np.asarray(parameters, dtype=float)
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("parameters must be a nonempty finite array")
    t = t.reshape(-1)
    if np.any((t < 0) | (t > 1)):
        raise ValueError("parameters must lie on [0, 1]")
    if not isinstance(num_control_points, (int, np.integer)) or num_control_points <= 0:
        raise ValueError("num_control_points must be a positive integer")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if num_control_points <= degree:
        raise ValueError("num_control_points must be greater than degree")
    actual_knots = (
        open_uniform_knots(num_control_points, degree)
        if knots is None
        else validate_knots(knots, num_control_points, degree)
    )

    matrix = csr_matrix(
        BSpline.design_matrix(t, actual_knots, degree, extrapolate=False)
    )
    matrix.eliminate_zeros()
    return matrix
