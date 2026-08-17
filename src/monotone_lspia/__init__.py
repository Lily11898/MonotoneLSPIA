"""Monotone B-spline fitting with Projected LSPIA."""

from ._basis import basis_matrix
from .api import (
    fit,
    monotone_lspia_basis_matrix,
    monotone_lspia_fit,
    monotone_lspia_predict,
    monotone_lspia_version,
    predict,
    version,
)
from .model import Diagnostics, MonotoneLSPIAModel
from .persistence import load, save
from .plotting import monotone_lspia_plot, plot

__version__ = "1.0.2"

__all__ = [
    "Diagnostics",
    "MonotoneLSPIAModel",
    "basis_matrix",
    "fit",
    "load",
    "monotone_lspia_basis_matrix",
    "monotone_lspia_fit",
    "monotone_lspia_plot",
    "monotone_lspia_predict",
    "monotone_lspia_version",
    "plot",
    "predict",
    "save",
    "version",
]
