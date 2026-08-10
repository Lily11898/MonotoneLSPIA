"""Shared decreasing-fit methods used by the real-data validation workflows."""

from __future__ import annotations

import contextlib
import csv
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from baselines import unconstrained_lspia
from pygam import LinearGAM, s
from run_value_proposition_benchmark import (
    _penalized_components,
    _pygam_converged,
    _solve_penalized,
)
from sklearn.isotonic import IsotonicRegression

from monotone_lspia import basis_matrix, fit

METHODS = (
    "LSPIA",
    "IsotonicRegression",
    "Penalized monotone B-spline",
    "pyGAM",
    "MonotoneLSPIA",
)
SMOOTH_METHODS = (
    "LSPIA",
    "Penalized monotone B-spline",
    "pyGAM",
    "MonotoneLSPIA",
)
LABELS = {
    "LSPIA": "LSPIA",
    "IsotonicRegression": "Isotonic",
    "Penalized monotone B-spline": "Penalized B-spline",
    "pyGAM": "pyGAM",
    "MonotoneLSPIA": "MonotoneLSPIA",
}
COLORS = {
    "LSPIA": "#d62728",
    "IsotonicRegression": "#9467bd",
    "Penalized monotone B-spline": "#2ca02c",
    "pyGAM": "#ff7f0e",
    "MonotoneLSPIA": "#1f77b4",
}
LINESTYLES = {
    "LSPIA": "--",
    "IsotonicRegression": "-.",
    "Penalized monotone B-spline": "-",
    "pyGAM": ":",
    "MonotoneLSPIA": "-",
}

NUM_CONTROL_POINTS = 12
DEGREE = 3
CV_FOLDS = 4
MAX_ITERATIONS = 5000
TOLERANCE = 1e-8
ALPHA_GRID = np.logspace(-8, 2, 6)
LAMBDA_GRID = np.logspace(-3, 3, 7)


@dataclass(frozen=True)
class MethodOutput:
    """Predictions and diagnostics for one fitted method."""

    test_prediction: np.ndarray
    grid_prediction: np.ndarray
    selected_parameter: float | None
    iterations: int
    converged: bool


def normalized(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Map predictor values to the fitted interval."""
    return np.clip((np.asarray(values) - minimum) / (maximum - minimum), 0.0, 1.0)


def lspia_output(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    t_grid: np.ndarray,
) -> MethodOutput:
    coefficients, iterations = unconstrained_lspia(
        y_train,
        t_train,
        NUM_CONTROL_POINTS,
        degree=DEGREE,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
    )
    test_matrix = basis_matrix(t_test, NUM_CONTROL_POINTS, degree=DEGREE)
    grid_matrix = basis_matrix(t_grid, NUM_CONTROL_POINTS, degree=DEGREE)
    return MethodOutput(
        test_prediction=np.asarray(test_matrix @ coefficients).reshape(-1),
        grid_prediction=np.asarray(grid_matrix @ coefficients).reshape(-1),
        selected_parameter=None,
        iterations=iterations,
        converged=iterations < MAX_ITERATIONS,
    )


def isotonic_output(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    x_grid: np.ndarray,
) -> MethodOutput:
    estimator = IsotonicRegression(increasing=False, out_of_bounds="clip")
    estimator.fit(x_train, y_train)
    return MethodOutput(
        test_prediction=np.asarray(estimator.predict(x_test)).reshape(-1),
        grid_prediction=np.asarray(estimator.predict(x_grid)).reshape(-1),
        selected_parameter=None,
        iterations=0,
        converged=True,
    )


def penalized_output(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    t_grid: np.ndarray,
) -> MethodOutput:
    """Fit a decreasing spline by constraining an increasing fit to ``-y``."""
    matrix, difference, constraint, knots = _penalized_components(
        t_train,
        NUM_CONTROL_POINTS,
        DEGREE,
    )
    fold_ids = np.arange(y_train.size) % CV_FOLDS
    errors: list[float] = []
    successful = True
    for alpha in ALPHA_GRID:
        fold_errors: list[float] = []
        for fold_id in range(CV_FOLDS):
            training = fold_ids != fold_id
            result = _solve_penalized(
                matrix[training],
                -y_train[training],
                float(alpha),
                difference,
                constraint,
            )
            successful = successful and bool(result.success)
            residual = -(matrix[~training] @ result.x) - y_train[~training]
            fold_errors.append(float(np.mean(np.square(residual))))
        errors.append(float(np.mean(fold_errors)))
    selected = float(ALPHA_GRID[int(np.argmin(errors))])
    final = _solve_penalized(matrix, -y_train, selected, difference, constraint)
    test_matrix = basis_matrix(
        t_test,
        NUM_CONTROL_POINTS,
        degree=DEGREE,
        knots=knots,
    )
    grid_matrix = basis_matrix(
        t_grid,
        NUM_CONTROL_POINTS,
        degree=DEGREE,
        knots=knots,
    )
    return MethodOutput(
        test_prediction=-np.asarray(test_matrix @ final.x).reshape(-1),
        grid_prediction=-np.asarray(grid_matrix @ final.x).reshape(-1),
        selected_parameter=selected,
        iterations=int(getattr(final, "nit", 0)),
        converged=successful and bool(final.success),
    )


def pygam_output(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    x_grid: np.ndarray,
) -> MethodOutput:
    term = s(
        0,
        n_splines=NUM_CONTROL_POINTS,
        spline_order=DEGREE,
        constraints="monotonic_dec",
    )
    gam = LinearGAM(
        term,
        max_iter=MAX_ITERATIONS,
        tol=max(TOLERANCE, 1e-4),
        verbose=False,
    )
    captured = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        gam.gridsearch(x_train[:, None], y_train, lam=LAMBDA_GRID, progress=False)
    selected = float(np.asarray(gam.terms[0].lam).reshape(-1)[0])
    converged, iterations = _pygam_converged(gam)
    return MethodOutput(
        test_prediction=np.asarray(gam.predict(x_test[:, None])).reshape(-1),
        grid_prediction=np.asarray(gam.predict(x_grid[:, None])).reshape(-1),
        selected_parameter=selected,
        iterations=iterations,
        converged=converged,
    )


def monotone_lspia_output(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    x_grid: np.ndarray,
) -> MethodOutput:
    model = fit(
        x_train,
        y_train,
        direction="decreasing",
        num_control_points=NUM_CONTROL_POINTS,
        degree=DEGREE,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
    )
    return MethodOutput(
        test_prediction=np.asarray(model.predict(x_test)).reshape(-1),
        grid_prediction=np.asarray(model.predict(x_grid)).reshape(-1),
        selected_parameter=None,
        iterations=model.diagnostics.iterations,
        converged=model.diagnostics.converged,
    )


def fit_methods(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    x_grid: np.ndarray,
) -> dict[str, MethodOutput]:
    """Fit all five methods using identical observations and query points."""
    x_minimum = float(np.min(x_train))
    x_maximum = float(np.max(x_train))
    t_train = normalized(x_train, x_minimum, x_maximum)
    t_test = normalized(x_test, x_minimum, x_maximum)
    t_grid = normalized(x_grid, x_minimum, x_maximum)
    return {
        "LSPIA": lspia_output(t_train, y_train, t_test, t_grid),
        "IsotonicRegression": isotonic_output(x_train, y_train, x_test, x_grid),
        "Penalized monotone B-spline": penalized_output(
            t_train,
            y_train,
            t_test,
            t_grid,
        ),
        "pyGAM": pygam_output(x_train, y_train, x_test, x_grid),
        "MonotoneLSPIA": monotone_lspia_output(
            x_train,
            y_train,
            x_test,
            x_grid,
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of consistently keyed dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
