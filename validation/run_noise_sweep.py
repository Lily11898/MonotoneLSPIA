"""Legacy synthetic recovery experiment retained for historical comparison.

This script contains the superseded Denoise+LSPIA experiment and must not be
used to generate the final SoftwareX figures. Use
``run_final_accuracy_comparison.py`` instead.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from baselines import (
    evaluate,
    isotonic,
    monotone_smoothing_spline,
    moving_average,
    unconstrained_lspia,
)

from monotone_lspia import fit

ROOT = Path(__file__).parents[1]


def datasets(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sensor_log": np.log1p(5 * x) / np.log(6),
        "dose_response": 1 / (1 + np.exp(-14 * (x - 0.45))),
        "cdf_plateau": 0.08 + 0.84 / (1 + np.exp(-22 * (x - 0.55))),
    }


def metrics(prediction: np.ndarray, truth: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    slope = np.diff(prediction) / np.diff(x)
    return float(np.sqrt(np.mean((prediction - truth) ** 2))), float(max(0, -np.min(slope)))


def run(num_trials: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, 120)
    sigmas = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
    rows: list[dict[str, object]] = []
    for dataset_index, (name, truth) in enumerate(datasets(x).items(), start=1):
        for sigma_index, sigma in enumerate(sigmas, start=1):
            collected: dict[str, list[tuple[float, float, float, int]]] = {
                name: []
                for name in (
                    "LSPIA",
                    "Denoise+LSPIA",
                    "Isotonic",
                    "Monotone Smooth",
                    "MonotoneLSPIA",
                )
            }
            for trial in range(1, num_trials + 1):
                rng = np.random.default_rng(100000 * dataset_index + 1000 * sigma_index + trial)
                noisy = truth + sigma * rng.standard_normal(x.size)

                start = time.perf_counter()
                coefficients, iterations = unconstrained_lspia(noisy, x, 16)
                prediction = evaluate(coefficients, x)
                rmse, violation = metrics(prediction, truth, x)
                collected["LSPIA"].append(
                    (rmse, violation, time.perf_counter() - start, iterations)
                )

                start = time.perf_counter()
                coefficients, iterations = unconstrained_lspia(moving_average(noisy), x, 16)
                prediction = evaluate(coefficients, x)
                rmse, violation = metrics(prediction, truth, x)
                collected["Denoise+LSPIA"].append(
                    (rmse, violation, time.perf_counter() - start, iterations)
                )

                start = time.perf_counter()
                prediction = isotonic(noisy)
                rmse, violation = metrics(prediction, truth, x)
                collected["Isotonic"].append((rmse, violation, time.perf_counter() - start, 1))

                start = time.perf_counter()
                coefficients, _ = monotone_smoothing_spline(noisy, x, 16)
                prediction = evaluate(coefficients, x)
                rmse, violation = metrics(prediction, truth, x)
                collected["Monotone Smooth"].append(
                    (rmse, violation, time.perf_counter() - start, 1)
                )

                start = time.perf_counter()
                model = fit(x, noisy, num_control_points=16, max_iterations=300)
                prediction = model.predict(x)
                rmse, violation = metrics(prediction, truth, x)
                collected["MonotoneLSPIA"].append(
                    (rmse, violation, time.perf_counter() - start, model.diagnostics.iterations)
                )

            for method, values in collected.items():
                array = np.asarray(values)
                rows.append(
                    {
                        "dataset": name,
                        "sigma": sigma,
                        "method": method,
                        "recovery_rmse_mean": np.mean(array[:, 0]),
                        "recovery_rmse_sd": np.std(array[:, 0], ddof=1) if num_trials > 1 else 0,
                        "maximum_negative_slope_mean": np.mean(array[:, 1]),
                        "runtime_median_seconds": np.median(array[:, 2]),
                        "iterations_median": np.median(array[:, 3]),
                    }
                )
            print(name, sigma, "complete")
    with (output / "noise_sweep_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    arguments = parser.parse_args()
    run(arguments.num_trials, arguments.output)
