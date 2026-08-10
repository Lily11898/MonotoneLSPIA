"""Versioned, pickle-free persistence for fitted MonotoneLSPIA models."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .model import Diagnostics, MonotoneLSPIAModel

_FORMAT_NAME = "monotone_lspia_model"
_FORMAT_VERSION = 1


def save(model: MonotoneLSPIAModel, path: str | PathLike[str]) -> Path:
    """Save a fitted model to a compressed, pickle-free NPZ archive."""
    if not isinstance(model, MonotoneLSPIAModel) or model.software != "MonotoneLSPIA":
        raise TypeError("model must be returned by monotone_lspia.fit")
    destination = Path(path)
    metadata = {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "software": model.software,
        "software_version": model.version,
        "direction": model.direction,
        "degree": model.degree,
        "num_control_points": model.num_control_points,
        "x_minimum": model.x_minimum,
        "x_maximum": model.x_maximum,
        "rmse": model.rmse,
        "diagnostics": {
            "converged": model.diagnostics.converged,
            "iterations": model.diagnostics.iterations,
            "step_size": model.diagnostics.step_size,
            "coefficient_violation": model.diagnostics.coefficient_violation,
            "sampled_curve_violation": model.diagnostics.sampled_curve_violation,
            "minimum_signed_slope": model.diagnostics.minimum_signed_slope,
            "collocation_rank": model.diagnostics.collocation_rank,
            "full_column_rank": model.diagnostics.full_column_rank,
            "condition_number": (
                model.diagnostics.condition_number
                if np.isfinite(model.diagnostics.condition_number)
                else None
            ),
        },
        "has_coefficient_sequence": model.coefficient_sequence is not None,
    }
    sequence = (
        np.vstack(model.coefficient_sequence)
        if model.coefficient_sequence is not None
        else np.empty((0, model.num_control_points), dtype=float)
    )
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata)),
            knots=model.knots,
            coefficients=model.coefficients,
            training_x=model.training_x,
            training_y=model.training_y,
            fitted_values=model.fitted_values,
            error_history=model.diagnostics.error_history,
            update_history=model.diagnostics.update_history,
            constraint_violation_history=model.diagnostics.constraint_violation_history,
            coefficient_sequence=sequence,
        )
    return destination


def _metadata(archive: Any) -> dict[str, Any]:
    metadata = json.loads(str(archive["metadata"].item()))
    if metadata.get("format") != _FORMAT_NAME:
        raise ValueError("archive is not a MonotoneLSPIA model")
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported MonotoneLSPIA model format version")
    return metadata


def load(path: str | PathLike[str]) -> MonotoneLSPIAModel:
    """Load a model saved by :func:`save` without enabling pickle."""
    source = Path(path)
    required = {
        "metadata",
        "knots",
        "coefficients",
        "training_x",
        "training_y",
        "fitted_values",
        "error_history",
        "update_history",
        "constraint_violation_history",
        "coefficient_sequence",
    }
    try:
        with np.load(source, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise ValueError("model archive is missing required arrays")
            metadata = _metadata(archive)
            diagnostic_values = metadata["diagnostics"]
            coefficients = np.asarray(archive["coefficients"], dtype=float).copy()
            num_control_points = int(metadata["num_control_points"])
            if coefficients.shape != (num_control_points,):
                raise ValueError("model archive has an invalid coefficient shape")
            stored_sequence = np.asarray(archive["coefficient_sequence"], dtype=float)
            if stored_sequence.ndim != 2 or stored_sequence.shape[1] != num_control_points:
                raise ValueError("model archive has an invalid coefficient history")
            sequence = (
                [row.copy() for row in stored_sequence]
                if metadata["has_coefficient_sequence"]
                else None
            )
            diagnostics = Diagnostics(
                converged=bool(diagnostic_values["converged"]),
                iterations=int(diagnostic_values["iterations"]),
                step_size=float(diagnostic_values["step_size"]),
                error_history=np.asarray(archive["error_history"], dtype=float).copy(),
                update_history=np.asarray(archive["update_history"], dtype=float).copy(),
                constraint_violation_history=np.asarray(
                    archive["constraint_violation_history"], dtype=float
                ).copy(),
                coefficient_violation=float(diagnostic_values["coefficient_violation"]),
                sampled_curve_violation=float(diagnostic_values["sampled_curve_violation"]),
                minimum_signed_slope=float(diagnostic_values["minimum_signed_slope"]),
                collocation_rank=int(diagnostic_values["collocation_rank"]),
                full_column_rank=bool(diagnostic_values["full_column_rank"]),
                condition_number=(
                    float("inf")
                    if diagnostic_values["condition_number"] is None
                    else float(diagnostic_values["condition_number"])
                ),
            )
            return MonotoneLSPIAModel(
                software=str(metadata["software"]),
                version=str(metadata["software_version"]),
                direction=str(metadata["direction"]),
                degree=int(metadata["degree"]),
                num_control_points=num_control_points,
                knots=np.asarray(archive["knots"], dtype=float).copy(),
                coefficients=coefficients,
                x_minimum=float(metadata["x_minimum"]),
                x_maximum=float(metadata["x_maximum"]),
                training_x=np.asarray(archive["training_x"], dtype=float).copy(),
                training_y=np.asarray(archive["training_y"], dtype=float).copy(),
                fitted_values=np.asarray(archive["fitted_values"], dtype=float).copy(),
                rmse=float(metadata["rmse"]),
                diagnostics=diagnostics,
                coefficient_sequence=sequence,
            )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("could not load a valid MonotoneLSPIA model archive") from error
