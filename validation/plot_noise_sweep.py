"""Plot the legacy, superseded Denoise+LSPIA noise sweep.

Use ``run_final_accuracy_comparison.py`` for the final SoftwareX recovery
figure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[1]

DATASET_TITLES = {
    "sensor_log": "Sensor log",
    "dose_response": "Dose response",
    "cdf_plateau": "CDF plateau",
}

METHODS = (
    "LSPIA",
    "Denoise+LSPIA",
    "Isotonic",
    "Monotone Smooth",
    "MonotoneLSPIA",
)

STYLES = {
    "LSPIA": {"color": "#d62728", "marker": "o", "linestyle": "--"},
    "Denoise+LSPIA": {"color": "#ff7f0e", "marker": "s", "linestyle": ":"},
    "Isotonic": {"color": "#9467bd", "marker": "^", "linestyle": "-."},
    "Monotone Smooth": {"color": "#2ca02c", "marker": "D", "linestyle": "-"},
    "MonotoneLSPIA": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise ValueError(
            f"Expected 90 summary rows from the 100-trial design; found {len(rows)}"
        )
    return rows


def plot(input_path: Path, output_stem: Path, num_trials: int) -> None:
    rows = read_rows(input_path)
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)

    for axis, (dataset, title) in zip(axes, DATASET_TITLES.items(), strict=True):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for method in METHODS:
            method_rows = sorted(
                (row for row in dataset_rows if row["method"] == method),
                key=lambda row: float(row["sigma"]),
            )
            sigma = np.array([float(row["sigma"]) for row in method_rows])
            mean = np.array(
                [float(row["recovery_rmse_mean"]) for row in method_rows]
            )
            standard_error = np.array(
                [float(row["recovery_rmse_sd"]) for row in method_rows]
            ) / np.sqrt(num_trials)
            style = STYLES[method]
            axis.errorbar(
                sigma,
                mean,
                yerr=standard_error,
                label=method,
                linewidth=1.7 if method == "MonotoneLSPIA" else 1.25,
                markersize=4.5,
                capsize=2.2,
                **style,
            )
        axis.set_title(title)
        axis.set_xlabel("Noise standard deviation")
        axis.grid(True, alpha=0.25)
        axis.set_xlim(0.005, 0.105)

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
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Saved {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "noise_sweep_summary.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "results" / "noise_sweep_recovery",
    )
    parser.add_argument("--num-trials", type=int, default=100)
    arguments = parser.parse_args()
    plot(arguments.input, arguments.output_stem, arguments.num_trials)
