"""Create the publication representative monotonicity figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from baselines import evaluate, unconstrained_lspia

from monotone_lspia import fit

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"

NUM_OBSERVATIONS = 120
NUM_CONTROL_POINTS = 12
DEGREE = 3
NOISE_STANDARD_DEVIATION = 0.10
RANDOM_SEED = 31
MAX_ITERATIONS = 500
TOLERANCE = 1e-8


def main(output: Path = RESULTS) -> None:
    """Generate Figure 2 and print every value cited in its caption."""
    output.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, NUM_OBSERVATIONS)
    truth = 0.08 + 0.84 / (1.0 + np.exp(-22.0 * (x - 0.55)))
    rng = np.random.default_rng(RANDOM_SEED)
    noisy = truth + NOISE_STANDARD_DEVIATION * rng.standard_normal(x.size)

    lspia_coefficients, lspia_iterations = unconstrained_lspia(
        noisy,
        x,
        NUM_CONTROL_POINTS,
        degree=DEGREE,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
    )
    monotone_model = fit(
        x,
        noisy,
        direction="increasing",
        num_control_points=NUM_CONTROL_POINTS,
        degree=DEGREE,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
    )

    x_grid = np.linspace(0.0, 1.0, 1000)
    truth_grid = 0.08 + 0.84 / (1.0 + np.exp(-22.0 * (x_grid - 0.55)))
    lspia_prediction = evaluate(lspia_coefficients, x_grid, degree=DEGREE)
    monotone_prediction = np.asarray(monotone_model.predict(x_grid))
    lspia_slopes = np.diff(lspia_prediction) / np.diff(x_grid)
    monotone_slopes = np.diff(monotone_prediction) / np.diff(x_grid)

    minimum_lspia_slope = float(np.min(lspia_slopes))
    minimum_monotone_slope = float(np.min(monotone_slopes))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    axes[0].scatter(
        x,
        noisy,
        s=18,
        color="0.65",
        alpha=0.55,
        label="Noisy observations",
    )
    axes[0].plot(
        x_grid,
        truth_grid,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="True curve",
    )
    axes[0].plot(
        x_grid,
        lspia_prediction,
        color="#d62728",
        linewidth=1.8,
        label="LSPIA",
    )
    axes[0].plot(
        x_grid,
        monotone_prediction,
        color="#1f77b4",
        linewidth=2.0,
        label="MonotoneLSPIA",
    )
    axes[0].set_ylabel("Response")
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        x_grid[:-1],
        lspia_slopes,
        color="#d62728",
        linewidth=1.8,
        label="LSPIA",
    )
    axes[1].plot(
        x_grid[:-1],
        monotone_slopes,
        color="#1f77b4",
        linewidth=2.0,
        label="MonotoneLSPIA",
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Predictor")
    axes[1].set_ylabel("Sampled slope")
    axes[1].legend(frameon=True)
    axes[1].grid(True, alpha=0.25)

    figure.tight_layout()
    png_path = output / "figure2_representative_fit.png"
    pdf_path = output / "figure2_representative_fit.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Observations: {NUM_OBSERVATIONS}")
    print(f"Control coefficients: {NUM_CONTROL_POINTS}")
    print(f"LSPIA iterations: {lspia_iterations}")
    print(f"MonotoneLSPIA iterations: {monotone_model.diagnostics.iterations}")
    print(f"MonotoneLSPIA converged: {monotone_model.diagnostics.converged}")
    print(f"Minimum LSPIA sampled slope: {minimum_lspia_slope:.12g}")
    print(f"Minimum MonotoneLSPIA sampled slope: {minimum_monotone_slope:.12g}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    arguments = parser.parse_args()
    main(arguments.output)
