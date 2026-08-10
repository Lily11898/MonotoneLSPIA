"""Run the final five-method recovery-accuracy experiment for the paper."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygam
import scipy
import sklearn
from baselines import evaluate, unconstrained_lspia
from run_value_proposition_benchmark import (
    monotone_lspia_fit,
    penalized_fit,
    pygam_fit,
)
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).parents[1]

METHODS = (
    "LSPIA",
    "IsotonicRegression",
    "Penalized monotone B-spline",
    "pyGAM",
    "MonotoneLSPIA",
)

METHOD_LABELS = {
    "LSPIA": "LSPIA",
    "IsotonicRegression": "Isotonic",
    "Penalized monotone B-spline": "Penalized B-spline",
    "pyGAM": "pyGAM",
    "MonotoneLSPIA": "MonotoneLSPIA",
}

STYLES = {
    "LSPIA": {"color": "#d62728", "marker": "o", "linestyle": "--"},
    "IsotonicRegression": {
        "color": "#9467bd",
        "marker": "^",
        "linestyle": "-.",
    },
    "Penalized monotone B-spline": {
        "color": "#2ca02c",
        "marker": "D",
        "linestyle": "-",
    },
    "pyGAM": {"color": "#ff7f0e", "marker": "s", "linestyle": ":"},
    "MonotoneLSPIA": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
}

DATASET_TITLES = {
    "sensor_log": "Sensor log",
    "dose_response": "Dose response",
    "cdf_plateau": "CDF plateau",
}


def truth_functions(x: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three deterministic monotone target functions."""
    return {
        "sensor_log": np.log1p(5.0 * x) / np.log(6.0),
        "dose_response": 1.0 / (1.0 + np.exp(-14.0 * (x - 0.45))),
        "cdf_plateau": 0.08 + 0.84 / (1.0 + np.exp(-22.0 * (x - 0.55))),
    }


def recovery_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    x_grid: np.ndarray,
) -> tuple[float, float]:
    """Return recovery RMSE and the maximum sampled monotonicity violation."""
    slope = np.diff(prediction) / np.diff(x_grid)
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    violation = float(max(0.0, -np.min(slope)))
    return rmse, violation


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a CSV file with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reusable_focused_rows(
    path: Path,
    *,
    num_trials: int,
    sigmas: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Load compatible completed rows for the three expensive competitors."""
    reusable_methods = {
        "MonotoneLSPIA",
        "Penalized monotone B-spline",
        "pyGAM",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            sigma = float(source["sigma"])
            trial = int(source["trial"])
            method = source["method"]
            if sigma not in sigmas or trial > num_trials or method not in reusable_methods:
                continue
            dataset_index = list(DATASET_TITLES).index(source["dataset"]) + 1
            sigma_index = sigmas.index(sigma) + 1
            selected = source.get("selected_parameter", "")
            rows.append(
                {
                    "dataset": source["dataset"],
                    "sigma": sigma,
                    "trial": trial,
                    "seed": 100000 * dataset_index + 1000 * sigma_index + trial,
                    "method": method,
                    "recovery_rmse": float(source["recovery_rmse"]),
                    "maximum_negative_slope": float(
                        source["maximum_negative_slope"]
                    ),
                    "selected_parameter": None
                    if selected in {"", "None"}
                    else float(selected),
                    "iterations": int(source["iterations"]),
                    "converged": source["converged"].strip().lower() == "true",
                }
            )
    expected = len(DATASET_TITLES) * len(sigmas) * num_trials * len(reusable_methods)
    if len(rows) != expected:
        raise ValueError(
            f"reuse file contains {len(rows)} compatible rows; expected {expected}"
        )
    return rows


def run_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
    alpha_grid: np.ndarray,
    lambda_grid: np.ndarray,
    cv_folds: int,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    """Fit one method using its documented final-experiment workflow."""
    if method == "LSPIA":
        coefficients, iterations = unconstrained_lspia(
            y,
            x,
            num_control_points,
            degree=degree,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return {
            "prediction": evaluate(coefficients, x_grid, degree=degree),
            "selected_parameter": None,
            "iterations": iterations,
            "converged": iterations < max_iterations,
        }

    if method == "IsotonicRegression":
        estimator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        estimator.fit(x, y)
        return {
            "prediction": np.asarray(estimator.predict(x_grid)),
            "selected_parameter": None,
            "iterations": 1,
            "converged": True,
        }

    if method == "Penalized monotone B-spline":
        result = penalized_fit(
            x,
            y,
            x_grid,
            num_control_points=num_control_points,
            degree=degree,
            alpha_grid=alpha_grid,
            cv_folds=cv_folds,
        )
        return {
            "prediction": result.prediction,
            "selected_parameter": result.selected_parameter,
            "iterations": result.iterations,
            "converged": result.converged,
        }

    if method == "pyGAM":
        result = pygam_fit(
            x,
            y,
            x_grid,
            num_control_points=num_control_points,
            degree=degree,
            lam_grid=lambda_grid,
            max_iterations=max_iterations,
            tolerance=max(tolerance, 1e-4),
        )
        return {
            "prediction": result.prediction,
            "selected_parameter": result.selected_parameter,
            "iterations": result.iterations,
            "converged": result.converged,
        }

    if method == "MonotoneLSPIA":
        result = monotone_lspia_fit(
            x,
            y,
            x_grid,
            num_control_points=num_control_points,
            degree=degree,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return {
            "prediction": result.prediction,
            "selected_parameter": None,
            "iterations": result.iterations,
            "converged": result.converged,
        }

    raise ValueError(f"unknown method: {method}")


def summarize(
    rows: list[dict[str, Any]],
    sigmas: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Aggregate paired Monte Carlo trials for plotting and reporting."""
    summary: list[dict[str, Any]] = []
    for dataset in DATASET_TITLES:
        for sigma in sigmas:
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["sigma"] == sigma
                    and row["method"] == method
                ]
                rmse = np.asarray([row["recovery_rmse"] for row in selected], dtype=float)
                violations = np.asarray(
                    [row["maximum_negative_slope"] for row in selected],
                    dtype=float,
                )
                summary.append(
                    {
                        "dataset": dataset,
                        "sigma": sigma,
                        "method": method,
                        "recovery_rmse_mean": float(np.mean(rmse)),
                        "recovery_rmse_sd": float(np.std(rmse, ddof=1))
                        if rmse.size > 1
                        else 0.0,
                        "recovery_rmse_se": float(np.std(rmse, ddof=1) / np.sqrt(rmse.size))
                        if rmse.size > 1
                        else 0.0,
                        "maximum_negative_slope_max": float(np.max(violations)),
                        "convergence_rate": float(
                            np.mean([bool(row["converged"]) for row in selected])
                        ),
                    }
                )
    return summary


def plot_summary(summary: list[dict[str, Any]], output: Path) -> None:
    """Create the final recovery-RMSE figure in raster and vector formats."""
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)

    for axis, (dataset, title) in zip(axes, DATASET_TITLES.items(), strict=True):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in summary
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: float(row["sigma"]),
            )
            sigma = np.asarray([row["sigma"] for row in selected], dtype=float)
            mean = np.asarray(
                [row["recovery_rmse_mean"] for row in selected], dtype=float
            )
            standard_error = np.asarray(
                [row["recovery_rmse_se"] for row in selected], dtype=float
            )
            axis.errorbar(
                sigma,
                mean,
                yerr=standard_error,
                label=METHOD_LABELS[method],
                linewidth=2.0 if method == "MonotoneLSPIA" else 1.35,
                markersize=5.0,
                capsize=2.4,
                **STYLES[method],
            )
        axis.set_title(title)
        axis.set_xlabel("Noise standard deviation")
        axis.set_xticks([0.02, 0.05, 0.10])
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Recovery RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    png_path = output / "figure3_recovery_rmse.png"
    pdf_path = output / "figure3_recovery_rmse.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def run(
    *,
    num_trials: int,
    output: Path,
    sigmas: tuple[float, ...],
    num_control_points: int,
    degree: int,
    cv_folds: int,
    max_iterations: int,
    tolerance: float,
    reuse_focused_results: Path | None,
) -> None:
    """Execute all paired trials and create publication artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    alpha_grid = np.logspace(-8, 2, 6)
    lambda_grid = np.logspace(-3, 3, 7)
    x = np.linspace(0.0, 1.0, 120)
    x_grid = np.linspace(0.0, 1.0, 1000)

    reused_path: str | None = None
    if reuse_focused_results is not None:
        resolved_reuse = reuse_focused_results.resolve()
        try:
            reused_path = str(resolved_reuse.relative_to(ROOT.resolve()))
        except ValueError:
            reused_path = str(resolved_reuse)

    configuration = {
        "num_trials": num_trials,
        "num_observations": int(x.size),
        "evaluation_grid_size": int(x_grid.size),
        "sigmas": list(sigmas),
        "num_control_points": num_control_points,
        "degree": degree,
        "cv_folds": cv_folds,
        "max_iterations": max_iterations,
        "monotone_lspia_tolerance": tolerance,
        "pygam_tolerance": max(tolerance, 1e-4),
        "penalized_alpha_grid": alpha_grid.tolist(),
        "pygam_lambda_grid": lambda_grid.tolist(),
        "methods": list(METHODS),
        "reused_focused_results": reused_path,
        "design_notes": [
            "All spline methods use 12 cubic spline basis functions; pyGAM retains its documented default intercept.",
            "The penalized B-spline selects alpha by deterministic four-fold cross-validation.",
            "pyGAM selects lambda through its documented gridsearch/GCV workflow.",
            "Isotonic regression calls sklearn.isotonic.IsotonicRegression directly.",
            "Timing claims are not taken from this accuracy experiment.",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "pygam": pygam.__version__,
        },
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    rows = (
        []
        if reuse_focused_results is None
        else reusable_focused_rows(
            reuse_focused_results,
            num_trials=num_trials,
            sigmas=sigmas,
        )
    )
    methods_to_run = (
        METHODS
        if reuse_focused_results is None
        else ("LSPIA", "IsotonicRegression")
    )
    if reuse_focused_results is not None:
        print(
            "Reused completed MonotoneLSPIA, penalized B-spline, and pyGAM "
            f"rows from: {reuse_focused_results}"
        )
    for dataset_index, dataset in enumerate(DATASET_TITLES, start=1):
        truth = truth_functions(x)[dataset]
        truth_grid = truth_functions(x_grid)[dataset]
        for sigma_index, sigma in enumerate(sigmas, start=1):
            for trial in range(1, num_trials + 1):
                seed = 100000 * dataset_index + 1000 * sigma_index + trial
                noisy = truth + sigma * np.random.default_rng(seed).standard_normal(x.size)
                for method in methods_to_run:
                    result = run_method(
                        method,
                        x,
                        noisy,
                        x_grid,
                        num_control_points=num_control_points,
                        degree=degree,
                        alpha_grid=alpha_grid,
                        lambda_grid=lambda_grid,
                        cv_folds=cv_folds,
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                    )
                    prediction = np.asarray(result.pop("prediction"), dtype=float)
                    rmse, violation = recovery_metrics(
                        prediction, truth_grid, x_grid
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "sigma": sigma,
                            "trial": trial,
                            "seed": seed,
                            "method": method,
                            "recovery_rmse": rmse,
                            "maximum_negative_slope": violation,
                            **result,
                        }
                    )
            write_rows(output / "final_accuracy_checkpoint.csv", rows)
            print(f"{dataset} sigma={sigma:g}: {num_trials} paired trials complete")

    trial_path = output / "final_accuracy_trials.csv"
    summary_path = output / "final_accuracy_summary.csv"
    write_rows(trial_path, rows)
    summary = summarize(rows, sigmas)
    write_rows(summary_path, summary)
    plot_summary(summary, output)
    print(f"Saved: {trial_path}")
    print(f"Saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--num-control-points", type=int, default=12)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--reuse-focused-results",
        type=Path,
        default=None,
        help=(
            "Reuse the compatible three-method accuracy_trials.csv and only "
            "compute LSPIA and IsotonicRegression"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "final_accuracy",
    )
    arguments = parser.parse_args()
    run(
        num_trials=arguments.num_trials,
        output=arguments.output,
        sigmas=tuple(arguments.sigmas),
        num_control_points=arguments.num_control_points,
        degree=arguments.degree,
        cv_folds=arguments.cv_folds,
        max_iterations=arguments.max_iterations,
        tolerance=arguments.tolerance,
        reuse_focused_results=arguments.reuse_focused_results,
    )


if __name__ == "__main__":
    main()
