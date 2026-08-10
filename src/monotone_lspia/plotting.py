"""Plotting helpers for fitted models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .api import predict
from .model import MonotoneLSPIAModel


def plot(model: MonotoneLSPIAModel) -> tuple[Any, tuple[Any, Any]]:
    """Plot the fitted curve and convergence histories."""
    import matplotlib.pyplot as plt

    x_grid = np.linspace(model.x_minimum, model.x_maximum, 600)
    y_grid = predict(model, x_grid)
    figure, (fit_axes, history_axes) = plt.subplots(1, 2, figsize=(11, 4.2))
    fit_axes.scatter(model.training_x, model.training_y, s=24, alpha=0.55)
    fit_axes.plot(x_grid, y_grid, linewidth=2)
    fit_axes.grid(True, alpha=0.3)
    fit_axes.set(xlabel="x", ylabel="y", title=f"{model.direction} fit (RMSE={model.rmse:.4g})")
    fit_axes.legend(["Observations", "Monotone B-spline"])
    error = np.maximum(model.diagnostics.error_history, np.finfo(float).tiny)
    history_axes.semilogy(np.arange(error.size), error, linewidth=1.7)
    update = model.diagnostics.update_history
    if update.size:
        history_axes.semilogy(
            np.arange(1, update.size + 1),
            np.maximum(update, np.finfo(float).tiny),
            linewidth=1.3,
        )
    history_axes.grid(True, alpha=0.3)
    history_axes.set(
        xlabel="Iteration",
        ylabel="Value",
        title=f"Convergence ({model.diagnostics.iterations} iterations)",
    )
    history_axes.legend(["Mean squared residual", "Relative update"])
    figure.tight_layout()
    return figure, (fit_axes, history_axes)


monotone_lspia_plot = plot
