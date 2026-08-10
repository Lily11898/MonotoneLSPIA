"""Model containers returned by :func:`monotone_lspia.fit`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Diagnostics:
    converged: bool
    iterations: int
    step_size: float
    error_history: NDArray[np.float64]
    update_history: NDArray[np.float64]
    constraint_violation_history: NDArray[np.float64]
    coefficient_violation: float
    sampled_curve_violation: float
    minimum_signed_slope: float
    collocation_rank: int
    full_column_rank: bool
    condition_number: float


@dataclass(slots=True)
class MonotoneLSPIAModel:
    software: str
    version: str
    direction: str
    degree: int
    num_control_points: int
    knots: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    x_minimum: float
    x_maximum: float
    training_x: NDArray[np.float64]
    training_y: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    rmse: float
    diagnostics: Diagnostics
    coefficient_sequence: list[NDArray[np.float64]] | None = None

    def predict(self, x: Any, *, extrapolation: str = "error") -> Any:
        """Evaluate this model at new predictor values."""
        from .api import predict

        return predict(self, x, extrapolation=extrapolation)

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively structured dictionary."""
        return asdict(self)

    def save(self, path: str | PathLike[str]) -> Path:
        """Save this model in the versioned MonotoneLSPIA NPZ format."""
        from .persistence import save

        return save(self, path)
