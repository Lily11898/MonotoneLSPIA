"""Fit treated and untreated Puromycin reaction-rate observations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monotone_lspia import fit

root = Path(__file__).parents[1]
data = np.genfromtxt(
    root / "data" / "Puromycin.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
figure, axes = plt.subplots(1, 2, figsize=(10, 4))

for axis, state in zip(axes, ("treated", "untreated"), strict=True):
    selected = data["state"] == state
    x, y = data["conc"][selected], data["rate"][selected]
    model = fit(
        x,
        y,
        num_control_points=6,
        max_iterations=2000,
        tolerance=1e-10,
    )
    x_grid = np.linspace(x.min(), x.max(), 500)
    axis.scatter(x, y, alpha=0.65)
    axis.plot(x_grid, model.predict(x_grid), linewidth=2)
    axis.set(title=f"{state.title()} cells", xlabel="Concentration", ylabel="Rate")
    axis.grid(True, alpha=0.3)
    print(state, "RMSE=", model.rmse, "iterations=", model.diagnostics.iterations)

figure.tight_layout()
figure.savefig(root / "results" / "puromycin_fit.png", dpi=300)
