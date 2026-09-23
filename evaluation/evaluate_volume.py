"""Tree-level stem-volume error metrics.

Author: Shahab Alaedin Baloochi

Predicted and reference stem volumes are expected to come from the same
TreeQSM-based reconstruction procedure. For each tree, signed relative volume
error is

    (V_pred - V_ref) / V_ref * 100.

The summary reports mean absolute relative error, maximum absolute relative
error and mean signed relative error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TreeStemVolume:
    """Volume comparison for one tree."""

    tree_id: str
    predicted_volume: float
    reference_volume: float
    signed_relative_error_percent: float
    absolute_relative_error_percent: float


@dataclass(frozen=True)
class StemVolumeSummary:
    """Aggregate tree-level volume error statistics."""

    mean_absolute_relative_error_percent: float
    maximum_absolute_relative_error_percent: float
    mean_signed_relative_error_percent: float
    per_tree: tuple[TreeStemVolume, ...]


def evaluate_tree_stem_volume(
    tree_id: str,
    predicted_volume: float,
    reference_volume: float,
) -> TreeStemVolume:
    """Compute signed and absolute relative volume error for one tree."""
    pred = float(predicted_volume)
    ref = float(reference_volume)

    if not np.isfinite(pred) or not np.isfinite(ref):
        raise ValueError("predicted_volume and reference_volume must be finite.")
    if pred < 0.0:
        raise ValueError("predicted_volume must be non-negative.")
    if ref <= 0.0:
        raise ValueError("reference_volume must be positive.")

    signed = (pred - ref) / ref * 100.0
    return TreeStemVolume(
        tree_id=str(tree_id),
        predicted_volume=pred,
        reference_volume=ref,
        signed_relative_error_percent=float(signed),
        absolute_relative_error_percent=float(abs(signed)),
    )


def summarize_stem_volumes(
    per_tree: Sequence[TreeStemVolume],
) -> StemVolumeSummary:
    """Summarise individual-tree relative volume errors."""
    items = tuple(per_tree)
    if not items:
        raise ValueError("At least one tree is required.")

    ids = [item.tree_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("tree_id values must be unique.")

    pred = np.asarray([item.predicted_volume for item in items], dtype=np.float64)
    ref = np.asarray([item.reference_volume for item in items], dtype=np.float64)
    signed = np.asarray(
        [item.signed_relative_error_percent for item in items],
        dtype=np.float64,
    )

    if not np.isfinite(pred).all() or not np.isfinite(ref).all() or not np.isfinite(signed).all():
        raise ValueError("Volume inputs and errors must be finite.")
    if np.any(pred < 0.0) or np.any(ref <= 0.0):
        raise ValueError("Predicted volumes must be non-negative and reference volumes positive.")

    absolute = np.abs(signed)
    return StemVolumeSummary(
        mean_absolute_relative_error_percent=float(absolute.mean()),
        maximum_absolute_relative_error_percent=float(absolute.max()),
        mean_signed_relative_error_percent=float(signed.mean()),
        per_tree=items,
    )


__all__ = [
    "TreeStemVolume",
    "StemVolumeSummary",
    "evaluate_tree_stem_volume",
    "summarize_stem_volumes",
]
