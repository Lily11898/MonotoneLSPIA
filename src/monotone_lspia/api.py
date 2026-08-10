"""Public fitting and prediction API."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._algorithm import projected_lspia
from ._basis import basis_matrix, open_uniform_knots, validate_knots
from .model import Diagnostics, MonotoneLSPIAModel

VERSION = "1.0.0"


def _finite_vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite array")
    return array.reshape(-1)


def _collocation_diagnostics(collocation: Any) -> tuple[int, float]:
    """Return a conservative numerical rank and condition estimate.

    Only the small Gram matrix is densified, so diagnostic computation does
    not materialize the full observation-by-coefficient matrix.
    """
    gram = (collocation.T @ collocation).toarray()
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = float(max(0.0, eigenvalues[-1]))
    if largest == 0.0:
        return 0, float("inf")
    eigenvalue_tolerance = max(collocation.shape) * np.finfo(float).eps * largest
    positive = eigenvalues[eigenvalues > eigenvalue_tolerance]
    rank = int(positive.size)
    condition_number = (
        float("inf")
        if rank < collocation.shape[1]
        else float(np.sqrt(largest / positive[0]))
    )
    return rank, condition_number


def fit(
    x: ArrayLike,
    y: ArrayLike,
    *,
    direction: str = "increasing",
    num_control_points: int | None = None,
    degree: int = 3,
    knots: ArrayLike | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    weight: str = "practical",
    step_size: float | None = None,
    safety_factor: float = 0.99,
    fix_endpoints: bool = False,
    store_history: bool = False,
) -> MonotoneLSPIAModel:
    """Fit a monotone B-spline with Projected LSPIA.

    Observations are sorted by ``x`` and the predictor interval is mapped to
    ``[0, 1]``. Repeated predictor values are allowed.
    """
    x_array, y_array = _finite_vector(x, "x"), _finite_vector(y, "y")
    if x_array.size != y_array.size:
        raise ValueError("x and y must contain the same number of observations")
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if weight not in {"practical", "theoretical"}:
        raise ValueError("weight must be 'practical' or 'theoretical'")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if x_array.size < degree + 1:
        raise ValueError("at least degree + 1 observations are required")
    if np.max(x_array) == np.min(x_array):
        raise ValueError("x must contain at least two distinct values")
    if num_control_points is None:
        num_control_points = min(12, x_array.size)
    if not isinstance(num_control_points, (int, np.integer)) or num_control_points <= degree:
        raise ValueError("num_control_points must be an integer greater than degree")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if step_size is not None and (not np.isfinite(step_size) or step_size <= 0):
        raise ValueError("step_size must be positive")
    if not np.isfinite(safety_factor) or not 0 < safety_factor < 1:
        raise ValueError("safety_factor must lie strictly between 0 and 1")

    order = np.argsort(x_array, kind="stable")
    x_sorted, y_sorted = x_array[order], y_array[order]
    x_minimum, x_maximum = float(x_sorted[0]), float(x_sorted[-1])
    t = (x_sorted - x_minimum) / (x_maximum - x_minimum)
    actual_knots = (
        open_uniform_knots(num_control_points, degree)
        if knots is None
        else validate_knots(knots, num_control_points, degree)
    )
    collocation = basis_matrix(
        t,
        num_control_points,
        degree=degree,
        knots=actual_knots,
    )
    coefficients, sequence, iteration = projected_lspia(
        y_sorted,
        t,
        num_control_points,
        degree,
        actual_knots,
        direction=direction,
        max_iterations=max_iterations,
        tolerance=tolerance,
        weight=weight,
        step_size=step_size,
        safety_factor=safety_factor,
        fix_endpoints=bool(fix_endpoints),
        store_history=bool(store_history),
        matrix=collocation,
    )
    fitted_values = np.asarray(collocation @ coefficients).reshape(-1)
    matrix_rank, condition_number = _collocation_diagnostics(collocation)
    diagnostic_t = np.linspace(0.0, 1.0, max(1000, 20 * num_control_points))
    diagnostic_curve = np.asarray(
        basis_matrix(
            diagnostic_t,
            num_control_points,
            degree=degree,
            knots=actual_knots,
        )
        @ coefficients
    ).reshape(-1)
    diagnostic_x = x_minimum + diagnostic_t * (x_maximum - x_minimum)
    signed_slope = np.diff(diagnostic_curve) / np.diff(diagnostic_x)
    signed_difference = np.diff(coefficients)
    if direction == "decreasing":
        signed_slope = -signed_slope
        signed_difference = -signed_difference
    diagnostics = Diagnostics(
        converged=iteration.converged,
        iterations=iteration.iterations,
        step_size=iteration.step_size,
        error_history=iteration.error_history,
        update_history=iteration.update_history,
        constraint_violation_history=iteration.violation_history,
        coefficient_violation=float(np.sum(np.maximum(0.0, -signed_difference) ** 2)),
        sampled_curve_violation=float(np.mean(np.maximum(0.0, -signed_slope) ** 2)),
        minimum_signed_slope=float(np.min(signed_slope)),
        collocation_rank=matrix_rank,
        full_column_rank=matrix_rank == num_control_points,
        condition_number=condition_number,
    )
    return MonotoneLSPIAModel(
        software="MonotoneLSPIA",
        version=VERSION,
        direction=direction,
        degree=int(degree),
        num_control_points=int(num_control_points),
        knots=actual_knots.copy(),
        coefficients=coefficients,
        x_minimum=x_minimum,
        x_maximum=x_maximum,
        training_x=x_sorted,
        training_y=y_sorted,
        fitted_values=fitted_values,
        rmse=float(np.sqrt(np.mean((y_sorted - fitted_values) ** 2))),
        diagnostics=diagnostics,
        coefficient_sequence=sequence,
    )


def predict(
    model: MonotoneLSPIAModel,
    x_query: ArrayLike,
    *,
    extrapolation: str = "error",
) -> Any:
    """Evaluate a fitted model, rejecting extrapolation by default."""
    if not isinstance(model, MonotoneLSPIAModel) or model.software != "MonotoneLSPIA":
        raise TypeError("model must be returned by monotone_lspia.fit")
    values = np.asarray(x_query, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("x_query must be nonempty and finite")
    if extrapolation not in {"error", "clip"}:
        raise ValueError("extrapolation must be 'error' or 'clip'")
    original_shape = values.shape
    column = values.reshape(-1)
    tolerance = 100 * np.finfo(float).eps * max(1.0, abs(model.x_minimum), abs(model.x_maximum))
    if extrapolation == "error" and (
        np.any(column < model.x_minimum - tolerance) or np.any(column > model.x_maximum + tolerance)
    ):
        raise ValueError("x_query lies outside the fitted range; use extrapolation='clip'")
    t = np.clip(
        (column - model.x_minimum) / (model.x_maximum - model.x_minimum),
        0.0,
        1.0,
    )
    result = np.asarray(
        basis_matrix(
            t,
            model.num_control_points,
            degree=model.degree,
            knots=model.knots,
        )
        @ model.coefficients
    ).reshape(original_shape)
    return float(result) if result.ndim == 0 else result


def version() -> str:
    """Return the software version."""
    return VERSION


# MATLAB-style aliases ease line-by-line comparison with the reference release.
monotone_lspia_fit = fit
monotone_lspia_predict = predict
monotone_lspia_basis_matrix = basis_matrix
monotone_lspia_version = version
