"""Validate repeated-predictor fitting on the Puromycin dataset.

The predictive assessment leaves out one complete interior concentration
level at a time, so replicate observations at the same predictor value never
appear in both the training and validation sets.  The penalized monotone
B-spline selects its roughness penalty by a nested concentration-level
cross-validation performed inside each outer training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from run_value_proposition_benchmark import _penalized_components, _solve_penalized

from monotone_lspia import basis_matrix, fit

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "puromycin"
NUM_CONTROL_POINTS = 6
DEGREE = 3
MAX_ITERATIONS = 2000
TOLERANCE = 1e-10
ALPHA_GRID = np.logspace(-8, 2, 6)


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the input is inside the repository."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class PenalizedModel:
    """Small fitted-model container used only by this validation script."""

    coefficients: np.ndarray
    knots: np.ndarray
    x_minimum: float
    x_maximum: float
    alpha: float

    def predict(self, x_query: np.ndarray) -> np.ndarray:
        t = (np.asarray(x_query, dtype=float) - self.x_minimum) / (
            self.x_maximum - self.x_minimum
        )
        matrix = basis_matrix(
            t,
            NUM_CONTROL_POINTS,
            degree=DEGREE,
            knots=self.knots,
        )
        return np.asarray(matrix @ self.coefficients).reshape(-1)


def fit_penalized(x: np.ndarray, y: np.ndarray, alpha: float) -> PenalizedModel:
    """Fit the controlled penalized monotone B-spline at one penalty."""
    x_minimum = float(np.min(x))
    x_maximum = float(np.max(x))
    t = (np.asarray(x, dtype=float) - x_minimum) / (x_maximum - x_minimum)
    matrix, difference, constraint, knots = _penalized_components(
        t,
        NUM_CONTROL_POINTS,
        DEGREE,
    )
    result = _solve_penalized(matrix, np.asarray(y, dtype=float), alpha, difference, constraint)
    return PenalizedModel(
        coefficients=np.asarray(result.x, dtype=float),
        knots=np.asarray(knots, dtype=float),
        x_minimum=x_minimum,
        x_maximum=x_maximum,
        alpha=float(alpha),
    )


def select_penalty_by_level(x: np.ndarray, y: np.ndarray) -> float:
    """Select alpha without splitting replicate predictor levels."""
    levels = np.unique(x)
    validation_levels = levels[1:-1]
    if validation_levels.size == 0:
        raise ValueError("at least three concentration levels are required")
    errors: list[float] = []
    for alpha in ALPHA_GRID:
        squared_errors: list[float] = []
        for held_level in validation_levels:
            training = x != held_level
            model = fit_penalized(x[training], y[training], float(alpha))
            residual = model.predict(x[~training]) - y[~training]
            squared_errors.extend(np.square(residual).tolist())
        errors.append(float(np.mean(squared_errors)))
    return float(ALPHA_GRID[int(np.argmin(errors))])


def leave_one_level_out(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, list[dict[str, object]]]:
    """Return grouped LOCO RMSEs and auditable fold-level records."""
    monotone_predictions: list[float] = []
    penalized_predictions: list[float] = []
    observations: list[float] = []
    fold_rows: list[dict[str, object]] = []
    for held_level in np.unique(x)[1:-1]:
        training = x != held_level
        monotone_model = fit(
            x[training],
            y[training],
            num_control_points=NUM_CONTROL_POINTS,
            degree=DEGREE,
            max_iterations=MAX_ITERATIONS,
            tolerance=TOLERANCE,
        )
        selected_alpha = select_penalty_by_level(x[training], y[training])
        penalized_model = fit_penalized(x[training], y[training], selected_alpha)
        monotone_fold = np.asarray(monotone_model.predict(x[~training])).reshape(-1)
        penalized_fold = penalized_model.predict(x[~training])
        observed_fold = np.asarray(y[~training], dtype=float)
        monotone_predictions.extend(monotone_fold.tolist())
        penalized_predictions.extend(penalized_fold.tolist())
        observations.extend(observed_fold.tolist())
        fold_rows.append(
            {
                "held_concentration": float(held_level),
                "held_observations": int(observed_fold.size),
                "selected_penalty": selected_alpha,
                "monotone_lspia_fold_rmse": float(
                    np.sqrt(np.mean(np.square(monotone_fold - observed_fold)))
                ),
                "penalized_bspline_fold_rmse": float(
                    np.sqrt(np.mean(np.square(penalized_fold - observed_fold)))
                ),
            }
        )
    observed = np.asarray(observations)
    monotone_rmse = float(
        np.sqrt(np.mean(np.square(np.asarray(monotone_predictions) - observed)))
    )
    penalized_rmse = float(
        np.sqrt(np.mean(np.square(np.asarray(penalized_predictions) - observed)))
    )
    return monotone_rmse, penalized_rmse, fold_rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    output: Path = RESULTS,
    data_path: Path = ROOT / "data" / "Puromycin.csv",
) -> None:
    """Run the validation, save metrics, and create the publication figure."""
    output.mkdir(parents=True, exist_ok=True)
    data = np.genfromtxt(
        data_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    summary_rows: list[dict[str, object]] = []
    all_fold_rows: list[dict[str, object]] = []

    for axis, state in zip(axes, ("treated", "untreated"), strict=True):
        selected = data["state"] == state
        x = np.asarray(data["conc"][selected], dtype=float)
        y = np.asarray(data["rate"][selected], dtype=float)
        model = fit(
            x,
            y,
            num_control_points=NUM_CONTROL_POINTS,
            degree=DEGREE,
            max_iterations=MAX_ITERATIONS,
            tolerance=TOLERANCE,
        )
        monotone_loco, penalized_loco, fold_rows = leave_one_level_out(x, y)
        for row in fold_rows:
            all_fold_rows.append({"group": state, **row})
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 500)
        prediction = np.asarray(model.predict(x_grid)).reshape(-1)
        axis.scatter(x, y, alpha=0.65, label="Observations")
        axis.plot(x_grid, prediction, linewidth=2.2, label="MonotoneLSPIA")
        axis.set(title=f"{state.title()} cells", xlabel="Concentration")
        axis.grid(True, alpha=0.3)
        summary_rows.append(
            {
                "group": state,
                "observations": int(x.size),
                "concentration_levels": int(np.unique(x).size),
                "training_rmse": model.rmse,
                "minimum_signed_slope": model.diagnostics.minimum_signed_slope,
                "iterations": model.diagnostics.iterations,
                "converged": model.diagnostics.converged,
                "monotone_lspia_loco_rmse": monotone_loco,
                "penalized_bspline_loco_rmse": penalized_loco,
            }
        )
        print(
            f"{state}: training RMSE={model.rmse:.6f}, "
            f"minimum slope={model.diagnostics.minimum_signed_slope:.6g}, "
            f"MonotoneLSPIA LOCO RMSE={monotone_loco:.6f}, "
            f"penalized B-spline LOCO RMSE={penalized_loco:.6f}"
        )

    axes[0].set_ylabel("Reaction rate")
    axes[0].legend(frameon=True)
    figure.tight_layout()
    png_path = output / "puromycin_fit.png"
    pdf_path = output / "puromycin_fit.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    write_rows(output / "puromycin_summary.csv", summary_rows)
    write_rows(output / "puromycin_loco_folds.csv", all_fold_rows)
    configuration = {
        "source_data": portable_path(data_path),
        "validation": "leave one complete interior concentration level out",
        "replicates_kept_together": True,
        "num_control_points": NUM_CONTROL_POINTS,
        "degree": DEGREE,
        "max_iterations": MAX_ITERATIONS,
        "tolerance": TOLERANCE,
        "penalized_alpha_grid": ALPHA_GRID.tolist(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {output / 'puromycin_summary.csv'}")
    print(f"Saved: {output / 'puromycin_loco_folds.csv'}")
    print(f"Saved: {output / 'configuration.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "Puromycin.csv",
    )
    arguments = parser.parse_args()
    main(arguments.output, arguments.data)
