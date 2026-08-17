"""Numerical kernel for Projected LSPIA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from ._basis import basis_matrix


@dataclass(slots=True)
class IterationInfo:
    step_size: float
    iterations: int
    converged: bool
    error_history: NDArray[np.float64]
    update_history: NDArray[np.float64]
    violation_history: NDArray[np.float64]


def pava(values: NDArray[np.float64], direction: str = "increasing") -> NDArray[np.float64]:
    """Euclidean projection of a vector onto a monotone cone."""
    x = np.asarray(values, dtype=float)
    original_shape = x.shape
    x = x.reshape(-1)
    if direction == "decreasing":
        return (-pava(-x, "increasing")).reshape(original_shape)
    if direction != "increasing":
        raise ValueError("direction must be 'increasing' or 'decreasing'")

    n = x.size
    levels = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)
    starts = np.zeros(n, dtype=int)
    ends = np.zeros(n, dtype=int)
    num_blocks = 0
    for index, value in enumerate(x):
        levels[num_blocks] = value
        weights[num_blocks] = 1.0
        starts[num_blocks] = index
        ends[num_blocks] = index
        num_blocks += 1
        while num_blocks > 1 and levels[num_blocks - 2] > levels[num_blocks - 1]:
            left = num_blocks - 2
            right = num_blocks - 1
            weight = weights[left] + weights[right]
            levels[left] = (weights[left] * levels[left] + weights[right] * levels[right]) / weight
            weights[left] = weight
            ends[left] = ends[right]
            num_blocks -= 1
    projected = np.zeros(n, dtype=float)
    for block in range(num_blocks):
        projected[starts[block] : ends[block] + 1] = levels[block]
    return projected.reshape(original_shape)


def fixed_endpoint_projection(
    values: NDArray[np.float64],
    left_endpoint: float,
    right_endpoint: float,
    direction: str = "increasing",
) -> NDArray[np.float64]:
    """Project onto a monotone cone while keeping both endpoints fixed.

    For an increasing fit, the free interior coefficients are projected onto
    the isotonic cone and then clipped to the feasible interval bounded by the
    two fixed endpoints.  Decreasing fits are handled by a sign reversal.
    """
    candidate = np.asarray(values, dtype=float)
    original_shape = candidate.shape
    candidate = candidate.reshape(-1)
    if candidate.size < 2:
        raise ValueError("fix_endpoints requires at least two control coefficients")
    if direction == "decreasing":
        if left_endpoint < right_endpoint:
            raise ValueError(
                "fixed endpoint observations are incompatible with a decreasing fit"
            )
        return (
            -fixed_endpoint_projection(
                -candidate,
                -left_endpoint,
                -right_endpoint,
                "increasing",
            )
        ).reshape(original_shape)
    if direction != "increasing":
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if left_endpoint > right_endpoint:
        raise ValueError(
            "fixed endpoint observations are incompatible with an increasing fit"
        )

    projected = np.empty_like(candidate)
    projected[0] = left_endpoint
    projected[-1] = right_endpoint
    if candidate.size > 2:
        interior = pava(candidate[1:-1], "increasing")
        projected[1:-1] = np.clip(interior, left_endpoint, right_endpoint)
    return projected.reshape(original_shape)


def initialize_coefficients(y: NDArray[np.float64], num_control_points: int) -> NDArray[np.float64]:
    """Initialize coefficients by sampling the ordered response data."""
    n = num_control_points - 1
    coefficients = np.zeros(num_control_points, dtype=float)
    coefficients[0], coefficients[-1] = y[0], y[-1]
    for index in range(1, n):
        sample_index = int(np.floor(y.size * index / n))
        coefficients[index] = y[sample_index]
    return coefficients


def practical_step(matrix: csr_matrix, safety_factor: float = 0.99) -> float:
    gram = matrix.T @ matrix
    bound = float(np.max(np.asarray(gram.sum(axis=1)).reshape(-1)))
    if bound <= np.finfo(float).eps:
        raise ValueError("collocation matrix is degenerate")
    return safety_factor * 2.0 / bound


def theoretical_step(matrix: csr_matrix) -> float:
    try:
        eigenvalues = np.linalg.eigvalsh((matrix.T @ matrix).toarray())
        eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
        positive = eigenvalues[eigenvalues > 1e-12]
        if positive.size == 0 or eigenvalues[-1] <= 0:
            raise ValueError("degenerate spectrum")
        return float(2.0 / (eigenvalues[-1] + positive[0]))
    except (ValueError, np.linalg.LinAlgError):
        return practical_step(matrix)


def coefficient_violation(coefficients: NDArray[np.float64], direction: str) -> float:
    differences = np.diff(coefficients)
    if direction == "decreasing":
        differences = -differences
    return float(np.sum(np.maximum(0.0, -differences) ** 2))


def projected_lspia(
    y: NDArray[np.float64],
    t: NDArray[np.float64],
    num_control_points: int,
    degree: int,
    knots: NDArray[np.float64],
    *,
    direction: str,
    max_iterations: int,
    tolerance: float,
    weight: str,
    step_size: float | None,
    safety_factor: float,
    fix_endpoints: bool,
    store_history: bool,
    matrix: csr_matrix | None = None,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]] | None, IterationInfo]:
    """Run the Projected LSPIA iteration."""
    if matrix is None:
        matrix = basis_matrix(t, num_control_points, degree=degree, knots=knots)
    if fix_endpoints:
        left_endpoint = float(np.mean(y[t == np.min(t)]))
        right_endpoint = float(np.mean(y[t == np.max(t)]))
    else:
        left_endpoint = right_endpoint = 0.0
    initial = initialize_coefficients(y, num_control_points)
    coefficients = (
        fixed_endpoint_projection(initial, left_endpoint, right_endpoint, direction)
        if fix_endpoints
        else pava(initial, direction)
    )
    if step_size is None:
        step_size = (
            practical_step(matrix, safety_factor)
            if weight == "practical"
            else theoretical_step(matrix)
        )
    history = [coefficients.copy()] if store_history else None
    error_history = [float(np.mean((y - matrix @ coefficients) ** 2))]
    update_history: list[float] = []
    violation_history = [coefficient_violation(coefficients, direction)]
    converged = False

    for _ in range(max_iterations):
        residual = y - matrix @ coefficients
        candidate = coefficients + step_size * np.asarray(matrix.T @ residual).reshape(-1)
        next_coefficients = (
            fixed_endpoint_projection(candidate, left_endpoint, right_endpoint, direction)
            if fix_endpoints
            else pava(candidate, direction)
        )
        relative_update = float(
            np.linalg.norm(next_coefficients - coefficients)
            / max(1.0, np.linalg.norm(coefficients))
        )
        coefficients = next_coefficients
        update_history.append(relative_update)
        error_history.append(float(np.mean((y - matrix @ coefficients) ** 2)))
        violation_history.append(coefficient_violation(coefficients, direction))
        if history is not None:
            history.append(coefficients.copy())
        if relative_update < tolerance:
            converged = True
            break

    info = IterationInfo(
        step_size=float(step_size),
        iterations=len(update_history),
        converged=converged,
        error_history=np.asarray(error_history),
        update_history=np.asarray(update_history),
        violation_history=np.asarray(violation_history),
    )
    return coefficients, history, info
