"""Minimal MonotoneLSPIA example."""

import numpy as np

from monotone_lspia import fit, plot

rng = np.random.default_rng(7)
x = np.linspace(10, 80, 100)
y = np.log(x) + 0.08 * rng.standard_normal(x.size)

model = fit(x, y, direction="increasing", num_control_points=12)
x_query = np.linspace(x.min(), x.max(), 500)
y_query = model.predict(x_query)

print(f"Converged: {model.diagnostics.converged}")
print(f"Iterations: {model.diagnostics.iterations}")
print(f"Training RMSE: {model.rmse:.6f}")
print(f"Minimum fitted difference: {np.min(np.diff(y_query)):.3e}")

figure, _ = plot(model)
figure.savefig("getting_started.png", dpi=200)
