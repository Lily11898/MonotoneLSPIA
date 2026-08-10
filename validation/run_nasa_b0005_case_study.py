"""Reproduce the NASA B0005 battery-capacity degradation case study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import comparison_methods as comparison
import scipy
from scipy.io import loadmat

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "nasa_b0005"
DATA_PATH = ROOT / "data" / "NASA_B0005" / "B0005.mat"
NUM_FOLDS = 5
GRID_SIZE = 1000


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the input is inside the repository."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def load_capacity(data_path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Extract discharge-cycle index and capacity from the official MAT file."""
    battery = loadmat(data_path, squeeze_me=True, struct_as_record=False)["B0005"]
    capacity = np.asarray(
        [float(cycle.data.Capacity) for cycle in battery.cycle if cycle.type == "discharge"],
        dtype=float,
    )
    if capacity.size != 168:
        raise ValueError(f"Unexpected number of B0005 discharge cycles: {capacity.size}")
    cycle_index = np.arange(1, capacity.size + 1, dtype=float)
    return cycle_index, capacity


def main(output: Path = RESULTS, data_path: Path = DATA_PATH) -> None:
    """Run interleaved validation, fit the full series, and save artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    x, y = load_capacity(data_path)
    x_grid = np.linspace(float(x[0]), float(x[-1]), GRID_SIZE)
    errors = {method: [] for method in comparison.METHODS}
    fold_rows: list[dict[str, object]] = []

    positions = np.arange(x.size)
    for fold_id in range(NUM_FOLDS):
        test = positions % NUM_FOLDS == fold_id
        test[0] = False
        test[-1] = False
        train = ~test
        outputs = comparison.fit_methods(x[train], y[train], x[test], x_grid)
        for method in comparison.METHODS:
            residual = outputs[method].test_prediction - y[test]
            errors[method].append(residual)
            fold_rows.append(
                {
                    "fold": fold_id + 1,
                    "method": method,
                    "test_observations": int(np.sum(test)),
                    "rmse_ah": float(np.sqrt(np.mean(np.square(residual)))),
                    "mae_ah": float(np.mean(np.abs(residual))),
                }
            )
        print(f"interleaved fold {fold_id + 1} complete")

    full_outputs = comparison.fit_methods(x, y, x, x_grid)
    result_rows: list[dict[str, object]] = []
    for method in comparison.METHODS:
        residual = np.concatenate(errors[method])
        slope = np.diff(full_outputs[method].grid_prediction) / np.diff(x_grid)
        violation = float(max(0.0, np.max(slope)))
        method_output = full_outputs[method]
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        result_rows.append(
            {
                "method": method,
                "interleaved_cv_rmse_ah": rmse,
                "interleaved_cv_mae_ah": mae,
                "maximum_positive_slope_ah_per_cycle": violation,
                "selected_parameter_full_fit": method_output.selected_parameter,
                "iterations_full_fit": method_output.iterations,
                "converged_full_fit": method_output.converged,
            }
        )
        print(
            f"{method}: CV RMSE={rmse:.8f} Ah, MAE={mae:.8f} Ah, "
            f"maximum positive slope={violation:.8g} Ah/cycle, "
            f"selected parameter={method_output.selected_parameter}"
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    axes[0].scatter(
        x,
        y,
        s=18,
        color="0.60",
        alpha=0.45,
        label="Measured capacity",
    )
    for method in comparison.METHODS:
        axes[0].plot(
            x_grid,
            full_outputs[method].grid_prediction,
            color=comparison.COLORS[method],
            linestyle=comparison.LINESTYLES[method],
            linewidth=2.2 if method == "MonotoneLSPIA" else 1.5,
            label=comparison.LABELS[method],
        )
    axes[0].set_ylabel("Capacity (Ah)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, frameon=True)

    slopes: dict[str, np.ndarray] = {}
    for method in comparison.SMOOTH_METHODS:
        slope = np.diff(full_outputs[method].grid_prediction) / np.diff(x_grid)
        slopes[method] = slope
        axes[1].plot(
            x_grid[:-1],
            slope,
            color=comparison.COLORS[method],
            linestyle=comparison.LINESTYLES[method],
            linewidth=2.2 if method == "MonotoneLSPIA" else 1.4,
            label=comparison.LABELS[method],
        )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    y_maximum = axes[1].get_ylim()[1]
    axes[1].axhspan(0.0, y_maximum, color="#d62728", alpha=0.06, zorder=0)
    lspia_slope = slopes["LSPIA"]
    violation_index = int(np.argmax(lspia_slope))
    axes[1].annotate(
        "LSPIA monotonicity violation",
        xy=(x_grid[violation_index], lspia_slope[violation_index]),
        xytext=(38, y_maximum * 0.78),
        arrowprops={"arrowstyle": "->", "color": comparison.COLORS["LSPIA"]},
        color=comparison.COLORS["LSPIA"],
        fontsize=8,
    )
    axes[1].set_xlabel("Discharge cycle")
    axes[1].set_ylabel("Sampled slope (Ah/cycle)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8, frameon=True, loc="lower right")
    figure.tight_layout()
    png_path = output / "nasa_b0005_case_study.png"
    pdf_path = output / "nasa_b0005_case_study.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    comparison.write_rows(output / "nasa_b0005_results.csv", result_rows)
    comparison.write_rows(output / "nasa_b0005_fold_results.csv", fold_rows)
    configuration = {
        "source": "NASA PCoE B0005 lithium-ion battery aging data",
        "source_data": portable_path(data_path),
        "discharge_cycles": int(x.size),
        "observed_local_capacity_increases": int(np.sum(np.diff(y) > 0.0)),
        "validation_observations": int(x.size - 2),
        "validation": "five deterministic interleaved folds; endpoints always trained",
        "num_control_points": comparison.NUM_CONTROL_POINTS,
        "degree": comparison.DEGREE,
        "alpha_grid": comparison.ALPHA_GRID.tolist(),
        "lambda_grid": comparison.LAMBDA_GRID.tolist(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    print(f"discharge cycles={x.size}, local increases={np.sum(np.diff(y) > 0.0)}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {output / 'nasa_b0005_results.csv'}")
    print(f"Saved: {output / 'nasa_b0005_fold_results.csv'}")
    print(f"Saved: {output / 'configuration.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    arguments = parser.parse_args()
    main(arguments.output, arguments.data)
