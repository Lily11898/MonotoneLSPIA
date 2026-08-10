"""Research-only comparison methods used by the validation scripts."""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from monotone_lspia import basis_matrix
from monotone_lspia._algorithm import initialize_coefficients, pava, practical_step
from monotone_lspia._basis import open_uniform_knots


def unconstrained_lspia(
    y: np.ndarray,
    t: np.ndarray,
    num_control_points: int,
    *,
    degree: int = 3,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, int]:
    knots = open_uniform_knots(num_control_points, degree)
    matrix = basis_matrix(t, num_control_points, degree=degree, knots=knots)
    coefficients = initialize_coefficients(np.asarray(y), num_control_points)
    step = practical_step(matrix)
    for iteration in range(1, max_iterations + 1):
        next_coefficients = coefficients + step * (matrix.T @ (y - matrix @ coefficients))
        update = np.linalg.norm(next_coefficients - coefficients) / max(
            1.0, np.linalg.norm(coefficients)
        )
        coefficients = np.asarray(next_coefficients).reshape(-1)
        if update < tolerance:
            break
    return coefficients, iteration


def moving_average(y: np.ndarray, window: int = 7) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    half = window // 2
    return np.asarray(
        [np.mean(y[max(0, i - half) : min(y.size, i + half + 1)]) for i in range(y.size)]
    )


def isotonic(y: np.ndarray, direction: str = "increasing") -> np.ndarray:
    return pava(np.asarray(y), direction)


def monotone_smoothing_spline(
    y: np.ndarray,
    t: np.ndarray,
    num_control_points: int,
    *,
    direction: str = "increasing",
    degree: int = 3,
    alpha_grid: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0),
    cv_folds: int = 4,
) -> tuple[np.ndarray, float]:
    """Constrained penalized B-spline with deterministic K-fold CV."""
    knots = open_uniform_knots(num_control_points, degree)
    matrix = basis_matrix(t, num_control_points, degree=degree, knots=knots).toarray()
    second_difference = np.diff(np.eye(num_control_points), n=2, axis=0)
    sign = 1.0 if direction == "increasing" else -1.0
    constraint = LinearConstraint(sign * np.diff(np.eye(num_control_points), axis=0), 0, np.inf)

    def solve(a: np.ndarray, response: np.ndarray, alpha: float) -> np.ndarray:
        gram = a.T @ a + alpha * second_difference.T @ second_difference
        rhs = a.T @ response
        initial = pava(np.linalg.lstsq(gram, rhs, rcond=None)[0], direction)
        result = minimize(
            lambda p: (
                0.5 * np.dot(a @ p - response, a @ p - response)
                + 0.5 * alpha * np.dot(second_difference @ p, second_difference @ p)
            ),
            initial,
            jac=lambda p: gram @ p - rhs,
            constraints=[constraint],
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 2000},
        )
        if not result.success:
            return pava(result.x, direction)
        return result.x

    fold_ids = np.arange(y.size) % max(2, min(cv_folds, y.size))
    errors = []
    for alpha in alpha_grid:
        fold_errors = []
        for fold in np.unique(fold_ids):
            train = fold_ids != fold
            coefficients = solve(matrix[train], y[train], alpha)
            fold_errors.append(np.mean((matrix[~train] @ coefficients - y[~train]) ** 2))
        errors.append(np.mean(fold_errors))
    best_alpha = float(alpha_grid[int(np.argmin(errors))])
    return solve(matrix, y, best_alpha), best_alpha


def evaluate(coefficients: np.ndarray, t: np.ndarray, degree: int = 3) -> np.ndarray:
    knots = open_uniform_knots(coefficients.size, degree)
    return np.asarray(
        basis_matrix(t, coefficients.size, degree=degree, knots=knots) @ coefficients
    ).reshape(-1)
