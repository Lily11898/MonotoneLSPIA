"""Run a timing-only benchmark with identical public workflow boundaries.

Accuracy is intentionally not recomputed here.  Every timed call starts with
in-memory ``x`` and ``y`` arrays and ends after model construction, fitting,
prediction on the supplied grid, and a common monotonicity check.  The total
workflow includes roughness-parameter selection where applicable; the fixed
workflow assumes that the selected roughness parameter is already known.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import os
import platform
import subprocess
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
from run_value_proposition_benchmark import (
    COLORS,
    METHODS,
    _new_pygam,
    _penalized_components,
    _pygam_converged,
    _solve_penalized,
    truth_functions,
)

from monotone_lspia import basis_matrix, fit

ROOT = Path(__file__).parents[1]
WORKFLOWS = ("total", "fixed")


@dataclass(frozen=True)
class WorkflowResult:
    selected_parameter: float | None
    maximum_negative_slope: float
    prediction_checksum: float
    converged: bool
    iterations: int


def _monotonicity_check(prediction: np.ndarray, x_grid: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    slopes = np.diff(prediction) / np.diff(x_grid)
    violation = float(max(0.0, -np.min(slopes)))
    checksum = float(np.sum(prediction))
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("workflow produced non-finite predictions")
    return violation, checksum


def monotone_lspia_workflow(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    max_iterations: int,
    tolerance: float,
    selected_parameter: float | None = None,
) -> WorkflowResult:
    del selected_parameter
    model = fit(
        x,
        y,
        num_control_points=num_control_points,
        degree=degree,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    prediction = model.predict(x_grid)
    violation, checksum = _monotonicity_check(prediction, x_grid)
    return WorkflowResult(
        None,
        violation,
        checksum,
        model.diagnostics.converged,
        model.diagnostics.iterations,
    )


def penalized_total_workflow(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    alpha_grid: np.ndarray,
    cv_folds: int,
) -> WorkflowResult:
    matrix, difference, constraint, knots = _penalized_components(x, num_control_points, degree)
    grid_matrix = basis_matrix(x_grid, num_control_points, degree=degree, knots=knots).toarray()
    fold_ids = np.arange(y.size) % max(2, min(cv_folds, y.size))
    errors: list[float] = []
    all_successful = True
    for alpha in alpha_grid:
        fold_errors: list[float] = []
        for fold_id in np.unique(fold_ids):
            train = fold_ids != fold_id
            result = _solve_penalized(matrix[train], y[train], float(alpha), difference, constraint)
            all_successful = all_successful and bool(result.success)
            fold_errors.append(float(np.mean((matrix[~train] @ result.x - y[~train]) ** 2)))
        errors.append(float(np.mean(fold_errors)))
    selected = float(alpha_grid[int(np.argmin(errors))])
    final = _solve_penalized(matrix, y, selected, difference, constraint)
    prediction = grid_matrix @ final.x
    violation, checksum = _monotonicity_check(prediction, x_grid)
    return WorkflowResult(
        selected,
        violation,
        checksum,
        all_successful and bool(final.success),
        int(getattr(final, "nit", 0)),
    )


def penalized_fixed_workflow(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    selected_parameter: float,
) -> WorkflowResult:
    matrix, difference, constraint, knots = _penalized_components(x, num_control_points, degree)
    grid_matrix = basis_matrix(x_grid, num_control_points, degree=degree, knots=knots).toarray()
    final = _solve_penalized(matrix, y, float(selected_parameter), difference, constraint)
    prediction = grid_matrix @ final.x
    violation, checksum = _monotonicity_check(prediction, x_grid)
    return WorkflowResult(
        float(selected_parameter),
        violation,
        checksum,
        bool(final.success),
        int(getattr(final, "nit", 0)),
    )


def pygam_total_workflow(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    lam_grid: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> WorkflowResult:
    gam = _new_pygam(num_control_points, degree, None, max_iterations, tolerance)
    captured = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        gam.gridsearch(x[:, None], y, lam=lam_grid, progress=False)
    selected = float(np.asarray(gam.terms[0].lam).reshape(-1)[0])
    prediction = np.asarray(gam.predict(x_grid[:, None])).reshape(-1)
    violation, checksum = _monotonicity_check(prediction, x_grid)
    converged, iterations = _pygam_converged(gam)
    return WorkflowResult(selected, violation, checksum, converged, iterations)


def pygam_fixed_workflow(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    selected_parameter: float,
    max_iterations: int,
    tolerance: float,
) -> WorkflowResult:
    gam = _new_pygam(
        num_control_points,
        degree,
        float(selected_parameter),
        max_iterations,
        tolerance,
    )
    captured = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        gam.fit(x[:, None], y)
    prediction = np.asarray(gam.predict(x_grid[:, None])).reshape(-1)
    violation, checksum = _monotonicity_check(prediction, x_grid)
    converged, iterations = _pygam_converged(gam)
    return WorkflowResult(float(selected_parameter), violation, checksum, converged, iterations)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _timed(function: Callable[[], WorkflowResult]) -> tuple[float, WorkflowResult]:
    gc.collect()
    start = time.perf_counter_ns()
    result = function()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return elapsed, result


def _method_call(
    method: str,
    workflow: str,
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    selected: dict[str, float | None],
    options: dict[str, object],
) -> Callable[[], WorkflowResult]:
    common = {
        "num_control_points": int(options["num_control_points"]),
        "degree": int(options["degree"]),
    }
    if method == "MonotoneLSPIA":
        return lambda: monotone_lspia_workflow(
            x,
            y,
            x_grid,
            **common,
            max_iterations=int(options["max_iterations"]),
            tolerance=float(options["tolerance"]),
            selected_parameter=selected[method],
        )
    if method == "Penalized monotone B-spline" and workflow == "total":
        return lambda: penalized_total_workflow(
            x,
            y,
            x_grid,
            **common,
            alpha_grid=np.asarray(options["alpha_grid"], dtype=float),
            cv_folds=int(options["cv_folds"]),
        )
    if method == "Penalized monotone B-spline":
        return lambda: penalized_fixed_workflow(
            x,
            y,
            x_grid,
            **common,
            selected_parameter=float(selected[method]),
        )
    if workflow == "total":
        return lambda: pygam_total_workflow(
            x,
            y,
            x_grid,
            **common,
            lam_grid=np.asarray(options["lam_grid"], dtype=float),
            max_iterations=int(options["max_iterations"]),
            tolerance=max(float(options["tolerance"]), 1e-4),
        )
    return lambda: pygam_fixed_workflow(
        x,
        y,
        x_grid,
        **common,
        selected_parameter=float(selected[method]),
        max_iterations=int(options["max_iterations"]),
        tolerance=max(float(options["tolerance"]), 1e-4),
    )


def benchmark_condition(
    *,
    scenario: str,
    dataset: str,
    sigma: float,
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    warmups: int,
    repeats: int,
    options: dict[str, object],
    seed: int,
) -> list[dict[str, object]]:
    selected: dict[str, float | None] = {
        "MonotoneLSPIA": None,
        "Penalized monotone B-spline": None,
        "pyGAM": None,
    }
    for _ in range(warmups):
        for method in METHODS:
            total_call = _method_call(method, "total", x, y, x_grid, selected, options)
            total_result = total_call()
            selected[method] = total_result.selected_parameter
        for method in METHODS:
            _method_call(method, "fixed", x, y, x_grid, selected, options)()

    rng = np.random.default_rng(seed)
    tasks = [(workflow, method) for workflow in WORKFLOWS for method in METHODS]
    rows: list[dict[str, object]] = []
    for repeat in range(1, repeats + 1):
        order = rng.permutation(len(tasks))
        for position, order_index in enumerate(order, start=1):
            workflow, method = tasks[int(order_index)]
            elapsed, result = _timed(
                _method_call(method, workflow, x, y, x_grid, selected, options)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "dataset": dataset,
                    "sigma": sigma,
                    "observations": x.size,
                    "prediction_points": x_grid.size,
                    "repeat": repeat,
                    "execution_order": position,
                    "workflow": workflow,
                    "method": method,
                    "seconds": elapsed,
                    "selected_parameter": result.selected_parameter,
                    "maximum_negative_slope": result.maximum_negative_slope,
                    "prediction_checksum": result.prediction_checksum,
                    "converged": result.converged,
                    "iterations": result.iterations,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["scenario"]),
                str(row["dataset"]),
                float(row["sigma"]),
                int(row["observations"]),
                int(row["prediction_points"]),
                str(row["workflow"]),
                str(row["method"]),
            )
            for row in rows
        }
    )
    summary: list[dict[str, object]] = []
    for key in keys:
        selected = [
            row
            for row in rows
            if (
                str(row["scenario"]),
                str(row["dataset"]),
                float(row["sigma"]),
                int(row["observations"]),
                int(row["prediction_points"]),
                str(row["workflow"]),
                str(row["method"]),
            )
            == key
        ]
        seconds = np.asarray([row["seconds"] for row in selected], dtype=float)
        violations = np.asarray([row["maximum_negative_slope"] for row in selected], dtype=float)
        converged = np.asarray([row["converged"] for row in selected], dtype=bool)
        summary.append(
            {
                "scenario": key[0],
                "dataset": key[1],
                "sigma": key[2],
                "observations": key[3],
                "prediction_points": key[4],
                "workflow": key[5],
                "method": key[6],
                "repeats": seconds.size,
                "seconds_q25": float(np.quantile(seconds, 0.25)),
                "seconds_median": float(np.median(seconds)),
                "seconds_q75": float(np.quantile(seconds, 0.75)),
                "maximum_negative_slope": float(np.max(violations)),
                "convergence_rate": float(np.mean(converged)),
            }
        )
    return summary


def comparison_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    conditions = sorted(
        {
            (
                str(row["scenario"]),
                str(row["dataset"]),
                float(row["sigma"]),
                int(row["observations"]),
                int(row["prediction_points"]),
                str(row["workflow"]),
            )
            for row in summary
        }
    )
    comparisons: list[dict[str, object]] = []
    for condition in conditions:
        ours = next(
            row
            for row in summary
            if (
                str(row["scenario"]),
                str(row["dataset"]),
                float(row["sigma"]),
                int(row["observations"]),
                int(row["prediction_points"]),
                str(row["workflow"]),
            )
            == condition
            and row["method"] == "MonotoneLSPIA"
        )
        for competitor in METHODS[1:]:
            theirs = next(
                row
                for row in summary
                if (
                    str(row["scenario"]),
                    str(row["dataset"]),
                    float(row["sigma"]),
                    int(row["observations"]),
                    int(row["prediction_points"]),
                    str(row["workflow"]),
                )
                == condition
                and row["method"] == competitor
            )
            comparisons.append(
                {
                    "scenario": condition[0],
                    "dataset": condition[1],
                    "sigma": condition[2],
                    "observations": condition[3],
                    "prediction_points": condition[4],
                    "workflow": condition[5],
                    "competitor": competitor,
                    "monotone_lspia_convergence_rate": ours["convergence_rate"],
                    "competitor_convergence_rate": theirs["convergence_rate"],
                    "competitor_over_monotone_lspia_speedup": (
                        float(theirs["seconds_median"]) / float(ours["seconds_median"])
                    ),
                }
            )
    return comparisons


def _system_value(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, object]:
    numpy_configuration = io.StringIO()
    with contextlib.redirect_stdout(numpy_configuration):
        np.show_config()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": _system_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": _system_value(["sysctl", "-n", "hw.memsize"]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "pygam": pygam.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "numpy_configuration": numpy_configuration.getvalue(),
    }


def plot_scaling(summary: list[dict[str, object]], output: Path) -> None:
    scaling = [row for row in summary if row["scenario"] == "scaling"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for axis, workflow in zip(axes, WORKFLOWS):
        for method in METHODS:
            selected = sorted(
                [row for row in scaling if row["workflow"] == workflow and row["method"] == method],
                key=lambda row: int(row["observations"]),
            )
            sizes = np.asarray([row["observations"] for row in selected], dtype=int)
            median = np.asarray([row["seconds_median"] for row in selected], dtype=float)
            lower = median - np.asarray([row["seconds_q25"] for row in selected], float)
            upper = np.asarray([row["seconds_q75"] for row in selected], float) - median
            axis.errorbar(
                sizes,
                median,
                yerr=np.vstack([lower, upper]),
                marker="o",
                capsize=3,
                label=method,
                color=COLORS[method],
            )
        axis.set_title(f"{workflow.capitalize()} workflow")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of observations and prediction points")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Elapsed time (s), median and IQR")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "fair_timing_scaling.png", dpi=220)
    figure.savefig(output / "fair_timing_scaling.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--scale-sizes", type=int, nargs="+", default=[120, 1000, 10000, 100000])
    parser.add_argument("--num-control-points", type=int, default=12)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-accuracy-conditions", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "fair_timing",
    )
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "num_control_points": arguments.num_control_points,
        "degree": arguments.degree,
        "cv_folds": arguments.cv_folds,
        "max_iterations": arguments.max_iterations,
        "tolerance": arguments.tolerance,
        "alpha_grid": np.logspace(-8, 2, 6),
        "lam_grid": np.logspace(-3, 3, 7),
    }
    configuration = {
        "warmups": arguments.warmups,
        "repeats": arguments.repeats,
        "scale_sizes": arguments.scale_sizes,
        "skip_accuracy_conditions": arguments.skip_accuracy_conditions,
        "options": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in options.items()
        },
        "timing_boundaries": {
            "total": (
                "in-memory x,y to model construction, hyperparameter selection where "
                "applicable, final fit, prediction, and common monotonicity check"
            ),
            "fixed": (
                "in-memory x,y to model construction with a known selected parameter, "
                "one fit, prediction, and common monotonicity check"
            ),
        },
        "method_order": "randomized independently within each repeat",
        "environment": environment_metadata(),
    }
    (arguments.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    if not arguments.skip_accuracy_conditions:
        x = np.linspace(0.0, 1.0, 120)
        x_grid = np.linspace(0.0, 1.0, 1000)
        for dataset_index, (dataset, truth) in enumerate(truth_functions(x).items(), start=1):
            for sigma_index, sigma in enumerate((0.02, 0.05, 0.10), start=1):
                rng = np.random.default_rng(100000 * dataset_index + 1000 * sigma_index + 1)
                y = truth + sigma * rng.standard_normal(x.size)
                rows.extend(
                    benchmark_condition(
                        scenario="accuracy_condition",
                        dataset=dataset,
                        sigma=sigma,
                        x=x,
                        y=y,
                        x_grid=x_grid,
                        warmups=arguments.warmups,
                        repeats=arguments.repeats,
                        options=options,
                        seed=710000 + 100 * dataset_index + sigma_index,
                    )
                )
                print(f"timing: {dataset} sigma={sigma:g} complete")

    for size in arguments.scale_sizes:
        x = np.linspace(0.0, 1.0, size)
        truth = truth_functions(x)["dose_response"]
        rng = np.random.default_rng(800000 + size)
        y = truth + 0.05 * rng.standard_normal(size)
        rows.extend(
            benchmark_condition(
                scenario="scaling",
                dataset="dose_response",
                sigma=0.05,
                x=x,
                y=y,
                x_grid=x,
                warmups=arguments.warmups,
                repeats=arguments.repeats,
                options=options,
                seed=810000 + size,
            )
        )
        print(f"timing: scaling n={size} complete")

    _write_rows(arguments.output / "timing_trials.csv", rows)
    summary = summarize(rows)
    _write_rows(arguments.output / "timing_summary.csv", summary)
    comparisons = comparison_rows(summary)
    _write_rows(arguments.output / "timing_comparisons.csv", comparisons)
    plot_scaling(summary, arguments.output)
    print(f"saved timing results to {arguments.output}")


if __name__ == "__main__":
    main()
