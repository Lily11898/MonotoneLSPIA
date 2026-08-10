"""Reproduce the decreasing UCI power-plant case study."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygam
import scipy
import sklearn
from baselines import unconstrained_lspia
from pygam import LinearGAM, s
from run_value_proposition_benchmark import (
    _penalized_components,
    _pygam_converged,
    _solve_penalized,
)
from sklearn.isotonic import IsotonicRegression

from monotone_lspia import basis_matrix, fit

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "ccpp"
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

SPLIT_SEED = 294
TRAIN_FRACTION = 0.8
NUM_CONTROL_POINTS = 16
DEGREE = 3
CV_FOLDS = 4
MAX_ITERATIONS = 5000
TOLERANCE = 1e-8
ALPHA_GRID = np.logspace(-8, 2, 6)
LAMBDA_GRID = np.logspace(-3, 3, 7)


@dataclass(frozen=True)
class MethodOutput:
    """Predictions and diagnostics for one case-study method."""

    test_prediction: np.ndarray
    grid_prediction: np.ndarray
    selected_parameter: float | None
    iterations: int
    converged: bool


def normalized(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Map predictor values to the training interval and clip held-out extremes."""
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
    """Fit a decreasing spline by fitting an increasing spline to -y."""
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
    gam = LinearGAM(term, max_iter=MAX_ITERATIONS, tol=max(TOLERANCE, 1e-4), verbose=False)
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
        test_prediction=np.asarray(model.predict(x_test, extrapolation="clip")).reshape(-1),
        grid_prediction=np.asarray(model.predict(x_grid)).reshape(-1),
        selected_parameter=None,
        iterations=model.diagnostics.iterations,
        converged=model.diagnostics.converged,
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run all methods, save the held-out table, and create the figure."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(ROOT / "data" / "CCPP" / "Folds5x2_pp.xlsx", sheet_name=0)
    x = data["AT"].to_numpy(float)
    y = data["PE"].to_numpy(float)
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(x.size)
    cut = int(TRAIN_FRACTION * x.size)
    train, test = order[:cut], order[cut:]
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]
    x_minimum = float(np.min(x_train))
    x_maximum = float(np.max(x_train))
    x_grid = np.linspace(x_minimum, x_maximum, 1000)
    t_train = normalized(x_train, x_minimum, x_maximum)
    t_test = normalized(x_test, x_minimum, x_maximum)
    t_grid = normalized(x_grid, x_minimum, x_maximum)

    outputs = {
        "LSPIA": lspia_output(t_train, y_train, t_test, t_grid),
        "IsotonicRegression": isotonic_output(x_train, y_train, x_test, x_grid),
        "Penalized monotone B-spline": penalized_output(
            t_train,
            y_train,
            t_test,
            t_grid,
        ),
        "pyGAM": pygam_output(x_train, y_train, x_test, x_grid),
        "MonotoneLSPIA": monotone_lspia_output(x_train, y_train, x_test, x_grid),
    }

    result_rows: list[dict[str, object]] = []
    for method in METHODS:
        output = outputs[method]
        slope = np.diff(output.grid_prediction) / np.diff(x_grid)
        violation = float(max(0.0, np.max(slope)))
        rmse = float(np.sqrt(np.mean(np.square(output.test_prediction - y_test))))
        mae = float(np.mean(np.abs(output.test_prediction - y_test)))
        result_rows.append(
            {
                "method": method,
                "test_rmse_mw": rmse,
                "test_mae_mw": mae,
                "maximum_positive_slope_mw_per_celsius": violation,
                "selected_parameter": output.selected_parameter,
                "iterations": output.iterations,
                "converged": output.converged,
            }
        )
        print(
            f"{method}: RMSE={rmse:.6f}, MAE={mae:.6f}, "
            f"maximum positive slope={violation:.6g}, "
            f"selected parameter={output.selected_parameter}"
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    axes[0].scatter(x, y, s=8, color="0.65", alpha=0.12, label="Observations")
    for method in METHODS:
        axes[0].plot(
            x_grid,
            outputs[method].grid_prediction,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.2 if method == "MonotoneLSPIA" else 1.5,
            label=LABELS[method],
        )
    axes[0].set_ylabel("Net electrical output (MW)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, frameon=True)

    # Isotonic regression is shown in the fitted-trend panel but omitted here:
    # its piecewise-linear prediction is not differentiable at its breakpoints.
    for method in SMOOTH_METHODS:
        slope = np.diff(outputs[method].grid_prediction) / np.diff(x_grid)
        axes[1].plot(
            x_grid[:-1],
            slope,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.2 if method == "MonotoneLSPIA" else 1.4,
            label=LABELS[method],
        )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel(r"Ambient temperature ($^\circ$C)")
    axes[1].set_ylabel(r"Sampled slope (MW/$^\circ$C)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8, frameon=True, loc="lower left")
    figure.tight_layout()
    png_path = RESULTS / "ccpp_case_study.png"
    pdf_path = RESULTS / "ccpp_case_study.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    write_rows(RESULTS / "ccpp_results.csv", result_rows)
    configuration = {
        "observations": int(x.size),
        "training_observations": int(train.size),
        "test_observations": int(test.size),
        "split_seed": SPLIT_SEED,
        "training_fraction": TRAIN_FRACTION,
        "num_control_points": NUM_CONTROL_POINTS,
        "degree": DEGREE,
        "cv_folds": CV_FOLDS,
        "alpha_grid": ALPHA_GRID.tolist(),
        "lambda_grid": LAMBDA_GRID.tolist(),
        "python_dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pygam": pygam.__version__,
        },
    }
    (RESULTS / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    print(f"observations={x.size}, training={train.size}, test={test.size}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {RESULTS / 'ccpp_results.csv'}")


if __name__ == "__main__":
    main()
