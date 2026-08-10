"""Benchmark the practical value proposition of MonotoneLSPIA.

The comparison is deliberately limited to the two closest smooth monotone
competitors:

* a constrained penalized B-spline solved with SciPy/SLSQP; and
* pyGAM's monotone P-spline using the documented ``gridsearch`` workflow.

The script reports recovery accuracy, end-to-end tuning cost, fixed-
hyperparameter refit cost, scaling, convergence, and an auditable trace of
every MonotoneLSPIA projection.  Results are written to a separate directory
so that the publication validation outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import platform
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pygam
import scipy
from pygam import LinearGAM, s
from scipy.optimize import LinearConstraint, OptimizeResult, minimize

from monotone_lspia import basis_matrix, fit
from monotone_lspia._algorithm import pava
from monotone_lspia._basis import open_uniform_knots

ROOT = Path(__file__).parents[1]
METHODS = ("MonotoneLSPIA", "Penalized monotone B-spline", "pyGAM")
COLORS = {
    "MonotoneLSPIA": "#1f77b4",
    "Penalized monotone B-spline": "#2ca02c",
    "pyGAM": "#9467bd",
}


def truth_functions(x: np.ndarray) -> dict[str, np.ndarray]:
    """Representative smooth monotone targets used in the earlier study."""
    return {
        "sensor_log": np.log1p(5 * x) / np.log(6),
        "dose_response": 1 / (1 + np.exp(-14 * (x - 0.45))),
        "cdf_plateau": 0.08 + 0.84 / (1 + np.exp(-22 * (x - 0.55))),
    }


def recovery_metrics(
    prediction: np.ndarray, truth: np.ndarray, x: np.ndarray
) -> tuple[float, float]:
    slope = np.diff(prediction) / np.diff(x)
    return (
        float(np.sqrt(np.mean((prediction - truth) ** 2))),
        float(max(0.0, -np.min(slope))),
    )


@dataclass
class MethodResult:
    prediction: np.ndarray
    selected_parameter: float | None
    end_to_end_seconds: float
    refit_seconds: float
    iterations: int
    converged: bool
    message: str


def _penalized_components(
    x: np.ndarray, num_control_points: int, degree: int
) -> tuple[np.ndarray, np.ndarray, LinearConstraint, np.ndarray]:
    knots = open_uniform_knots(num_control_points, degree)
    matrix = basis_matrix(
        x, num_control_points, degree=degree, knots=knots
    ).toarray()
    second_difference = np.diff(np.eye(num_control_points), n=2, axis=0)
    first_difference = np.diff(np.eye(num_control_points), axis=0)
    constraint = LinearConstraint(first_difference, 0.0, np.inf)
    return matrix, second_difference, constraint, knots


def _solve_penalized(
    matrix: np.ndarray,
    y: np.ndarray,
    alpha: float,
    second_difference: np.ndarray,
    constraint: LinearConstraint,
) -> OptimizeResult:
    gram = matrix.T @ matrix + alpha * (second_difference.T @ second_difference)
    rhs = matrix.T @ y
    initial = pava(np.linalg.lstsq(gram, rhs, rcond=None)[0], "increasing")
    result = minimize(
        lambda coefficients: (
            0.5 * np.dot(matrix @ coefficients - y, matrix @ coefficients - y)
            + 0.5
            * alpha
            * np.dot(
                second_difference @ coefficients,
                second_difference @ coefficients,
            )
        ),
        initial,
        jac=lambda coefficients: gram @ coefficients - rhs,
        constraints=[constraint],
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 2000},
    )
    if not result.success:
        result.x = pava(np.asarray(result.x), "increasing")
    return result


def penalized_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    alpha_grid: np.ndarray,
    cv_folds: int,
) -> MethodResult:
    matrix, difference, constraint, knots = _penalized_components(
        x, num_control_points, degree
    )
    grid_matrix = basis_matrix(
        x_grid, num_control_points, degree=degree, knots=knots
    ).toarray()
    fold_ids = np.arange(y.size) % max(2, min(cv_folds, y.size))
    start = time.perf_counter()
    errors: list[float] = []
    for alpha in alpha_grid:
        fold_errors: list[float] = []
        for fold_id in np.unique(fold_ids):
            train = fold_ids != fold_id
            result = _solve_penalized(
                matrix[train], y[train], float(alpha), difference, constraint
            )
            fold_errors.append(
                float(np.mean((matrix[~train] @ result.x - y[~train]) ** 2))
            )
        errors.append(float(np.mean(fold_errors)))
    selected = float(alpha_grid[int(np.argmin(errors))])
    refit_start = time.perf_counter()
    final = _solve_penalized(matrix, y, selected, difference, constraint)
    refit_seconds = time.perf_counter() - refit_start
    end_to_end_seconds = time.perf_counter() - start
    return MethodResult(
        prediction=np.asarray(grid_matrix @ final.x).reshape(-1),
        selected_parameter=selected,
        end_to_end_seconds=end_to_end_seconds,
        refit_seconds=refit_seconds,
        iterations=int(getattr(final, "nit", 0)),
        converged=bool(final.success),
        message=str(final.message),
    )


def _new_pygam(
    num_control_points: int,
    degree: int,
    lam: float | None,
    max_iterations: int,
    tolerance: float,
) -> LinearGAM:
    term = s(
        0,
        n_splines=num_control_points,
        spline_order=degree,
        lam=0.6 if lam is None else lam,
        constraints="monotonic_inc",
    )
    return LinearGAM(
        term,
        max_iter=max_iterations,
        tol=tolerance,
        verbose=False,
    )


def _pygam_converged(gam: LinearGAM) -> tuple[bool, int]:
    differences = np.asarray(gam.logs_.get("diffs", []), dtype=float)
    iterations = int(differences.size)
    if not iterations:
        return False, 0
    return bool(differences[-1] < gam.tol), iterations


def pygam_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    lam_grid: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> MethodResult:
    x_column = x[:, None]
    grid_column = x_grid[:, None]
    gam = _new_pygam(
        num_control_points, degree, None, max_iterations, tolerance
    )
    captured = io.StringIO()
    start = time.perf_counter()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        gam.gridsearch(x_column, y, lam=lam_grid, progress=False)
    end_to_end_seconds = time.perf_counter() - start
    selected = float(np.asarray(gam.terms[0].lam).reshape(-1)[0])

    refit = _new_pygam(
        num_control_points, degree, selected, max_iterations, tolerance
    )
    refit_start = time.perf_counter()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        refit.fit(x_column, y)
    refit_seconds = time.perf_counter() - refit_start
    converged, iterations = _pygam_converged(gam)
    message = captured.getvalue().strip().replace("\n", "; ")
    return MethodResult(
        prediction=np.asarray(gam.predict(grid_column)).reshape(-1),
        selected_parameter=selected,
        end_to_end_seconds=end_to_end_seconds,
        refit_seconds=refit_seconds,
        iterations=iterations,
        converged=converged,
        message=message,
    )


def monotone_lspia_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    max_iterations: int,
    tolerance: float,
) -> MethodResult:
    start = time.perf_counter()
    model = fit(
        x,
        y,
        num_control_points=num_control_points,
        degree=degree,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    elapsed = time.perf_counter() - start
    return MethodResult(
        prediction=np.asarray(model.predict(x_grid)).reshape(-1),
        selected_parameter=None,
        end_to_end_seconds=elapsed,
        refit_seconds=elapsed,
        iterations=model.diagnostics.iterations,
        converged=model.diagnostics.converged,
        message="",
    )


def method_functions() -> dict[str, Callable[..., MethodResult]]:
    return {
        "MonotoneLSPIA": monotone_lspia_fit,
        "Penalized monotone B-spline": penalized_fit,
        "pyGAM": pygam_fit,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_accuracy(
    *,
    num_trials: int,
    output: Path,
    sigmas: tuple[float, ...],
    num_control_points: int,
    degree: int,
    alpha_grid: np.ndarray,
    lam_grid: np.ndarray,
    cv_folds: int,
    max_iterations: int,
    tolerance: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x = np.linspace(0.0, 1.0, 120)
    x_grid = np.linspace(0.0, 1.0, 1000)
    functions = method_functions()
    rows: list[dict[str, object]] = []
    for dataset_index, name in enumerate(truth_functions(x), start=1):
        truth = truth_functions(x)[name]
        truth_grid = truth_functions(x_grid)[name]
        for sigma_index, sigma in enumerate(sigmas, start=1):
            for trial in range(1, num_trials + 1):
                rng = np.random.default_rng(
                    100000 * dataset_index + 1000 * sigma_index + trial
                )
                noisy = truth + sigma * rng.standard_normal(x.size)
                common = {
                    "num_control_points": num_control_points,
                    "degree": degree,
                    "max_iterations": max_iterations,
                    "tolerance": tolerance,
                }
                for method in METHODS:
                    if method == "Penalized monotone B-spline":
                        result = functions[method](
                            x,
                            noisy,
                            x_grid,
                            num_control_points=num_control_points,
                            degree=degree,
                            alpha_grid=alpha_grid,
                            cv_folds=cv_folds,
                        )
                    elif method == "pyGAM":
                        result = functions[method](
                            x,
                            noisy,
                            x_grid,
                            num_control_points=num_control_points,
                            degree=degree,
                            lam_grid=lam_grid,
                            max_iterations=max_iterations,
                            tolerance=max(tolerance, 1e-4),
                        )
                    else:
                        result = functions[method](x, noisy, x_grid, **common)
                    rmse, violation = recovery_metrics(
                        result.prediction, truth_grid, x_grid
                    )
                    rows.append(
                        {
                            "dataset": name,
                            "sigma": sigma,
                            "trial": trial,
                            "method": method,
                            "recovery_rmse": rmse,
                            "maximum_negative_slope": violation,
                            "end_to_end_seconds": result.end_to_end_seconds,
                            "refit_seconds": result.refit_seconds,
                            "selected_parameter": result.selected_parameter,
                            "iterations": result.iterations,
                            "converged": result.converged,
                            "message": result.message,
                        }
                    )
            print(f"accuracy: {name} sigma={sigma:g} complete")

    write_rows(output / "accuracy_trials.csv", rows)
    summary: list[dict[str, object]] = []
    for name in truth_functions(x):
        for sigma in sigmas:
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == name
                    and row["sigma"] == sigma
                    and row["method"] == method
                ]
                rmse = np.asarray([row["recovery_rmse"] for row in selected], float)
                end_time = np.asarray(
                    [row["end_to_end_seconds"] for row in selected], float
                )
                refit_time = np.asarray([row["refit_seconds"] for row in selected], float)
                summary.append(
                    {
                        "dataset": name,
                        "sigma": sigma,
                        "method": method,
                        "recovery_rmse_mean": float(np.mean(rmse)),
                        "recovery_rmse_sd": float(np.std(rmse, ddof=1))
                        if rmse.size > 1
                        else 0.0,
                        "end_to_end_seconds_median": float(np.median(end_time)),
                        "refit_seconds_median": float(np.median(refit_time)),
                        "maximum_negative_slope_max": float(
                            max(row["maximum_negative_slope"] for row in selected)
                        ),
                        "convergence_rate": float(
                            np.mean([bool(row["converged"]) for row in selected])
                        ),
                    }
                )
    write_rows(output / "accuracy_summary.csv", summary)
    return rows, summary


def run_scaling(
    *,
    output: Path,
    sizes: tuple[int, ...],
    repeats: int,
    num_control_points: int,
    degree: int,
    alpha_grid: np.ndarray,
    lam_grid: np.ndarray,
    cv_folds: int,
    max_iterations: int,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for size in sizes:
        x = np.linspace(0.0, 1.0, size)
        truth = truth_functions(x)["dose_response"]
        for repeat in range(1, repeats + 1):
            rng = np.random.default_rng(900000 + 100 * size + repeat)
            noisy = truth + 0.05 * rng.standard_normal(size)
            for method in METHODS:
                if method == "MonotoneLSPIA":
                    result = monotone_lspia_fit(
                        x,
                        noisy,
                        x,
                        num_control_points=num_control_points,
                        degree=degree,
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                    )
                elif method == "Penalized monotone B-spline":
                    result = penalized_fit(
                        x,
                        noisy,
                        x,
                        num_control_points=num_control_points,
                        degree=degree,
                        alpha_grid=alpha_grid,
                        cv_folds=cv_folds,
                    )
                else:
                    result = pygam_fit(
                        x,
                        noisy,
                        x,
                        num_control_points=num_control_points,
                        degree=degree,
                        lam_grid=lam_grid,
                        max_iterations=max_iterations,
                        tolerance=max(tolerance, 1e-4),
                    )
                rows.append(
                    {
                        "observations": size,
                        "repeat": repeat,
                        "method": method,
                        "end_to_end_seconds": result.end_to_end_seconds,
                        "refit_seconds": result.refit_seconds,
                        "iterations": result.iterations,
                        "converged": result.converged,
                    }
                )
        print(f"scaling: n={size} complete")
    write_rows(output / "scaling_trials.csv", rows)
    return rows


def run_projection_audit(
    *,
    output: Path,
    num_control_points: int,
    degree: int,
    max_iterations: int,
    tolerance: float,
) -> list[dict[str, object]]:
    x = np.linspace(0.0, 1.0, 120)
    truth = truth_functions(x)["dose_response"]
    rng = np.random.default_rng(271828)
    noisy = truth + 0.10 * rng.standard_normal(x.size)
    model = fit(
        x,
        noisy,
        num_control_points=num_control_points,
        degree=degree,
        max_iterations=max_iterations,
        tolerance=tolerance,
        store_history=True,
    )
    assert model.coefficient_sequence is not None
    t = (x - x.min()) / (x.max() - x.min())
    matrix = basis_matrix(
        t,
        num_control_points,
        degree=degree,
        knots=model.knots,
    )
    rows: list[dict[str, object]] = []
    initial = model.coefficient_sequence[0]
    rows.append(
        {
            "iteration": 0,
            "pre_projection_violation": 0.0,
            "projection_correction_norm": 0.0,
            "post_projection_violation": float(
                np.sum(np.maximum(0.0, -np.diff(initial)) ** 2)
            ),
            "mean_squared_residual": model.diagnostics.error_history[0],
            "relative_update": 0.0,
            "projection_reproduced": True,
        }
    )
    for iteration, previous in enumerate(model.coefficient_sequence[:-1], start=1):
        residual = noisy - matrix @ previous
        candidate = previous + model.diagnostics.step_size * np.asarray(
            matrix.T @ residual
        ).reshape(-1)
        projected = pava(candidate, "increasing")
        recorded = model.coefficient_sequence[iteration]
        rows.append(
            {
                "iteration": iteration,
                "pre_projection_violation": float(
                    np.sum(np.maximum(0.0, -np.diff(candidate)) ** 2)
                ),
                "projection_correction_norm": float(
                    np.linalg.norm(projected - candidate)
                ),
                "post_projection_violation": float(
                    np.sum(np.maximum(0.0, -np.diff(recorded)) ** 2)
                ),
                "mean_squared_residual": model.diagnostics.error_history[iteration],
                "relative_update": model.diagnostics.update_history[iteration - 1],
                "projection_reproduced": bool(
                    np.allclose(projected, recorded, rtol=1e-12, atol=1e-14)
                ),
            }
        )
    write_rows(output / "projection_audit.csv", rows)
    return rows


def build_decision_report(
    summary: list[dict[str, object]],
    projection_rows: list[dict[str, object]],
    output: Path,
    acceptable_rmse_ratio: float,
    clear_speedup: float,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    condition_rows: list[dict[str, object]] = []
    conditions = sorted({(row["dataset"], row["sigma"]) for row in summary})
    for competitor in METHODS[1:]:
        ratios: list[float] = []
        end_speedups: list[float] = []
        refit_speedups: list[float] = []
        joint = 0
        for dataset, sigma in conditions:
            ours = next(
                row
                for row in summary
                if row["dataset"] == dataset
                and row["sigma"] == sigma
                and row["method"] == "MonotoneLSPIA"
            )
            theirs = next(
                row
                for row in summary
                if row["dataset"] == dataset
                and row["sigma"] == sigma
                and row["method"] == competitor
            )
            rmse_ratio = float(ours["recovery_rmse_mean"]) / float(
                theirs["recovery_rmse_mean"]
            )
            speedup = float(theirs["end_to_end_seconds_median"]) / float(
                ours["end_to_end_seconds_median"]
            )
            ratios.append(rmse_ratio)
            end_speedups.append(speedup)
            refit_speedups.append(
                float(theirs["refit_seconds_median"])
                / float(ours["refit_seconds_median"])
            )
            if rmse_ratio <= acceptable_rmse_ratio and speedup >= clear_speedup:
                joint += 1
            condition_rows.append(
                {
                    "dataset": dataset,
                    "sigma": sigma,
                    "competitor": competitor,
                    "rmse_ratio_ours_over_competitor": rmse_ratio,
                    "end_to_end_speedup": speedup,
                    "fixed_parameter_refit_speedup": refit_speedups[-1],
                    "acceptable_accuracy": rmse_ratio <= acceptable_rmse_ratio,
                    "clear_end_to_end_speedup": speedup >= clear_speedup,
                    "joint_success": rmse_ratio <= acceptable_rmse_ratio
                    and speedup >= clear_speedup,
                }
            )
        comparisons[competitor] = {
            "rmse_ratio_ours_over_competitor_median": float(np.median(ratios)),
            "rmse_ratio_range": [float(np.min(ratios)), float(np.max(ratios))],
            "end_to_end_speedup_median": float(np.median(end_speedups)),
            "end_to_end_speedup_range": [
                float(np.min(end_speedups)),
                float(np.max(end_speedups)),
            ],
            "fixed_parameter_refit_speedup_median": float(
                np.median(refit_speedups)
            ),
            "acceptable_accuracy_conditions": int(
                np.sum(np.asarray(ratios) <= acceptable_rmse_ratio)
            ),
            "clear_speedup_conditions": int(
                np.sum(np.asarray(end_speedups) >= clear_speedup)
            ),
            "joint_success_conditions": joint,
            "total_conditions": len(conditions),
        }
    report: dict[str, object] = {
        "predeclared_criteria": {
            "acceptable_accuracy": (
                f"MonotoneLSPIA recovery RMSE <= {acceptable_rmse_ratio:.2f} times competitor RMSE"
            ),
            "clear_end_to_end_speedup": (
                f"competitor end-to-end time / MonotoneLSPIA time >= {clear_speedup:.1f}"
            ),
        },
        "comparisons": comparisons,
        "projection_audit": {
            "stored_iterates": len(projection_rows),
            "iterations_with_active_projection": int(
                np.sum(
                    [
                        float(row["projection_correction_norm"]) > 1e-14
                        for row in projection_rows
                    ]
                )
            ),
            "maximum_pre_projection_violation": float(
                max(float(row["pre_projection_violation"]) for row in projection_rows)
            ),
            "maximum_post_projection_violation": float(
                max(float(row["post_projection_violation"]) for row in projection_rows)
            ),
            "all_projection_steps_reproduced": bool(
                all(bool(row["projection_reproduced"]) for row in projection_rows)
            ),
        },
    }
    (output / "decision_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_rows(output / "comparison_by_condition.csv", condition_rows)
    return report


def plot_results(
    summary: list[dict[str, object]],
    scaling_rows: list[dict[str, object]],
    projection_rows: list[dict[str, object]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in METHODS:
        selected = [row for row in summary if row["method"] == method]
        axes[0].scatter(
            np.mean([float(row["recovery_rmse_mean"]) for row in selected]),
            np.median([float(row["end_to_end_seconds_median"]) for row in selected]),
            s=80,
            label=method,
            color=COLORS[method],
        )
    axes[0].set_xlabel("Mean recovery RMSE across conditions")
    axes[0].set_ylabel("Median end-to-end time (s)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    sizes = sorted({int(row["observations"]) for row in scaling_rows})
    for method in METHODS:
        medians = []
        for size in sizes:
            medians.append(
                np.median(
                    [
                        float(row["end_to_end_seconds"])
                        for row in scaling_rows
                        if row["method"] == method and row["observations"] == size
                    ]
                )
            )
        axes[1].plot(
            sizes,
            medians,
            marker="o",
            label=method,
            color=COLORS[method],
        )
    axes[1].set_xlabel("Number of observations")
    axes[1].set_ylabel("Median end-to-end time (s)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "accuracy_cost_and_scaling.png", dpi=220)
    figure.savefig(output / "accuracy_cost_and_scaling.pdf")
    plt.close(figure)

    iterations = np.asarray([row["iteration"] for row in projection_rows], int)
    pre = np.asarray(
        [row["pre_projection_violation"] for row in projection_rows], float
    )
    post = np.asarray(
        [row["post_projection_violation"] for row in projection_rows], float
    )
    correction = np.asarray(
        [row["projection_correction_norm"] for row in projection_rows], float
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(iterations, np.maximum(pre, 1e-18), label="Before projection")
    axes[0].semilogy(iterations, np.maximum(post, 1e-18), label="After projection")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Coefficient-cone violation")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(
        iterations, np.maximum(correction, 1e-18), color="#d62728"
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Projection correction norm")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "projection_audit.png", dpi=220)
    figure.savefig(output / "projection_audit.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--scale-sizes", type=int, nargs="+", default=[120, 1000, 10000])
    parser.add_argument("--scale-repeats", type=int, default=3)
    parser.add_argument("--num-control-points", type=int, default=12)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--acceptable-rmse-ratio", type=float, default=1.10)
    parser.add_argument("--clear-speedup", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "value_proposition",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    alpha_grid = np.logspace(-8, 2, 6)
    lam_grid = np.logspace(-3, 3, 7)

    configuration = {
        "num_trials": arguments.num_trials,
        "sigmas": arguments.sigmas,
        "scale_sizes": arguments.scale_sizes,
        "scale_repeats": arguments.scale_repeats,
        "num_control_points": arguments.num_control_points,
        "degree": arguments.degree,
        "cv_folds": arguments.cv_folds,
        "max_iterations": arguments.max_iterations,
        "monotone_lspia_tolerance": arguments.tolerance,
        "pygam_tolerance": max(arguments.tolerance, 1e-4),
        "penalized_alpha_grid": alpha_grid.tolist(),
        "pygam_lambda_grid": lam_grid.tolist(),
        "acceptable_rmse_ratio": arguments.acceptable_rmse_ratio,
        "clear_speedup": arguments.clear_speedup,
        "design_notes": [
            f"All methods use {arguments.num_control_points} cubic spline basis functions; pyGAM retains its documented default intercept.",
            "The number of basis functions is fixed a priori and is not tuned for any method.",
            "The penalized B-spline uses deterministic four-fold CV; pyGAM uses its documented gridsearch/GCV workflow.",
            "End-to-end time includes smoothing-parameter selection; refit time assumes the selected parameter is already known.",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "pygam": pygam.__version__,
        },
    }
    (arguments.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    _, summary = run_accuracy(
        num_trials=arguments.num_trials,
        output=arguments.output,
        sigmas=tuple(arguments.sigmas),
        num_control_points=arguments.num_control_points,
        degree=arguments.degree,
        alpha_grid=alpha_grid,
        lam_grid=lam_grid,
        cv_folds=arguments.cv_folds,
        max_iterations=arguments.max_iterations,
        tolerance=arguments.tolerance,
    )
    scaling_rows = run_scaling(
        output=arguments.output,
        sizes=tuple(arguments.scale_sizes),
        repeats=arguments.scale_repeats,
        num_control_points=arguments.num_control_points,
        degree=arguments.degree,
        alpha_grid=alpha_grid,
        lam_grid=lam_grid,
        cv_folds=arguments.cv_folds,
        max_iterations=arguments.max_iterations,
        tolerance=arguments.tolerance,
    )
    projection_rows = run_projection_audit(
        output=arguments.output,
        num_control_points=arguments.num_control_points,
        degree=arguments.degree,
        max_iterations=arguments.max_iterations,
        tolerance=arguments.tolerance,
    )
    report = build_decision_report(
        summary,
        projection_rows,
        arguments.output,
        arguments.acceptable_rmse_ratio,
        arguments.clear_speedup,
    )
    plot_results(summary, scaling_rows, projection_rows, arguments.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
