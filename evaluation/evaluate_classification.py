"""Full-tree classification evaluation and threshold selection.

Author: Shahab Alaedin Baloochi

The module evaluates complete-tree probability outputs or binary masks. The
validation threshold grid is 0.00--1.00 in steps of 0.01. The primary reported
threshold is 0.50 and the sensitivity threshold is 0.63.

Threshold search supports pooled F2 and mean-per-tree F2. The aggregation must
be supplied explicitly. If several thresholds have the same maximum F2, the
smallest threshold is returned.

Binary masks can come from any post-processing stage. Reproducing final
post-processing results requires a mask from the complete post-processing
pipeline used for those results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .classification_metrics import (
    AggregatedClassification,
    PerTreeClassification,
    aggregate_per_tree_classification,
    branch_index_to_binary,
    evaluate_binary_classification,
)


PAPER_THRESHOLD_GRID_STEP = 0.01
PAPER_PRIMARY_THRESHOLD = 0.50
PAPER_SENSITIVITY_THRESHOLD = 0.63


@dataclass(frozen=True)
class ProbabilityTree:
    """One full-tree probability prediction and TreeQSM branch reference."""

    tree_id: str
    probabilities: NDArray[np.floating]
    branch_index: NDArray[np.integer]


@dataclass(frozen=True)
class MaskTree:
    """One full-tree binary prediction and TreeQSM branch reference."""

    tree_id: str
    prediction_mask: NDArray[np.bool_] | NDArray[np.integer]
    branch_index: NDArray[np.integer]


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Validation-set F2 threshold-search result."""

    selected_threshold: float
    selected_f2: float
    aggregation: Literal["pooled", "mean_per_tree"]
    thresholds: NDArray[np.float64]
    f2_scores: NDArray[np.float64]


def paper_threshold_grid() -> NDArray[np.float64]:
    """Return 101 thresholds: 0.00, 0.01, ..., 1.00."""
    return np.arange(101, dtype=np.float64) / 100.0


def _validate_probabilities(probabilities: NDArray[np.floating], n: int) -> NDArray[np.float64]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (n,):
        raise ValueError(f"probabilities must have shape ({n},), got {probs.shape}.")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contains NaN or infinite values.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0,1].")
    return probs


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be finite and lie in [0,1].")
    return value


def _prediction_from_probability(
    probabilities: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.uint8]:
    # Keep thresholding consistent with inference.
    return (probabilities >= threshold).astype(np.uint8, copy=False)


def evaluate_probability_trees(
    trees: Sequence[ProbabilityTree],
    *,
    threshold: float,
    unsegmented_id: int = 0,
    zero_division: float = 0.0,
) -> AggregatedClassification:
    """Evaluate raw full-tree probabilities at one fixed threshold."""
    if not trees:
        raise ValueError("At least one tree is required.")
    tau = _validate_threshold(threshold)

    results: list[PerTreeClassification] = []
    for tree in trees:
        if not isinstance(tree, ProbabilityTree):
            raise TypeError("trees must contain ProbabilityTree objects.")
        labels, valid = branch_index_to_binary(
            tree.branch_index,
            unsegmented_id=unsegmented_id,
        )
        probs = _validate_probabilities(tree.probabilities, labels.shape[0])
        pred = _prediction_from_probability(probs, tau)
        metrics = evaluate_binary_classification(
            labels,
            pred,
            valid_mask=valid,
            zero_division=zero_division,
        )
        results.append(PerTreeClassification(tree_id=str(tree.tree_id), metrics=metrics))

    return aggregate_per_tree_classification(results, zero_division=zero_division)


def evaluate_mask_trees(
    trees: Sequence[MaskTree],
    *,
    unsegmented_id: int = 0,
    zero_division: float = 0.0,
) -> AggregatedClassification:
    """Evaluate final full-tree binary masks after post-processing."""
    if not trees:
        raise ValueError("At least one tree is required.")

    results: list[PerTreeClassification] = []
    for tree in trees:
        if not isinstance(tree, MaskTree):
            raise TypeError("trees must contain MaskTree objects.")
        labels, valid = branch_index_to_binary(
            tree.branch_index,
            unsegmented_id=unsegmented_id,
        )

        pred = np.asarray(tree.prediction_mask)
        if pred.shape != labels.shape:
            raise ValueError(
                f"prediction_mask for tree {tree.tree_id!r} has shape {pred.shape}; "
                f"expected {labels.shape}."
            )
        if pred.dtype == np.bool_:
            pred_binary = pred.astype(np.uint8, copy=False)
        else:
            if not np.all((pred == 0) | (pred == 1)):
                raise ValueError("prediction_mask must contain only bool/0/1 values.")
            pred_binary = pred.astype(np.uint8, copy=False)

        metrics = evaluate_binary_classification(
            labels,
            pred_binary,
            valid_mask=valid,
            zero_division=zero_division,
        )
        results.append(PerTreeClassification(tree_id=str(tree.tree_id), metrics=metrics))

    return aggregate_per_tree_classification(results, zero_division=zero_division)


def select_validation_f2_threshold(
    trees: Sequence[ProbabilityTree],
    *,
    aggregation: Literal["pooled", "mean_per_tree"],
    unsegmented_id: int = 0,
    zero_division: float = 0.0,
) -> ThresholdSearchResult:
    """Select the validation threshold that maximises F2 on the 0.01 grid."""
    if aggregation not in ("pooled", "mean_per_tree"):
        raise ValueError("aggregation must be 'pooled' or 'mean_per_tree'.")
    thresholds = paper_threshold_grid()
    scores = np.empty_like(thresholds)

    for i, threshold in enumerate(thresholds):
        evaluated = evaluate_probability_trees(
            trees,
            threshold=float(threshold),
            unsegmented_id=unsegmented_id,
            zero_division=zero_division,
        )
        if aggregation == "pooled":
            scores[i] = evaluated.pooled.f2
        else:
            scores[i] = evaluated.mean_per_tree.f2

    if not np.isfinite(scores).all():
        raise RuntimeError("Threshold search produced a non-finite F2 score.")

    best_score = float(scores.max())
    # The first maximum on the ascending grid is the smallest threshold.
    best_indices = np.flatnonzero(scores == best_score)
    selected = float(thresholds[int(best_indices[0])])

    return ThresholdSearchResult(
        selected_threshold=selected,
        selected_f2=best_score,
        aggregation=aggregation,
        thresholds=thresholds,
        f2_scores=scores,
    )


__all__ = [
    "PAPER_THRESHOLD_GRID_STEP",
    "PAPER_PRIMARY_THRESHOLD",
    "PAPER_SENSITIVITY_THRESHOLD",
    "ProbabilityTree",
    "MaskTree",
    "ThresholdSearchResult",
    "paper_threshold_grid",
    "evaluate_probability_trees",
    "evaluate_mask_trees",
    "select_validation_f2_threshold",
]
