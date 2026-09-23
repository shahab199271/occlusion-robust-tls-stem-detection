"""Binary stem-classification metrics.

Author: Shahab Alaedin Baloochi

Stem is the positive class. The module computes accuracy, precision, recall,
F1 and F2, together with pooled and mean-per-tree aggregation.

TreeQSM branch-index convention:
* branch index 1: stem
* branch index >1: non-stem
* unsegmented sentinel: excluded from evaluation

The default unsegmented sentinel is 0. Converted datasets can supply another
sentinel through ``unsegmented_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


PAPER_F_BETA = 2.0


@dataclass(frozen=True)
class ConfusionCounts:
    """Binary confusion counts with stem as the positive class."""

    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def total(self) -> int:
        return int(self.tp + self.tn + self.fp + self.fn)

    @property
    def positives(self) -> int:
        return int(self.tp + self.fn)

    @property
    def predicted_positives(self) -> int:
        return int(self.tp + self.fp)

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        return ConfusionCounts(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy, precision, recall, F1 and F2 for one evaluated set."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    counts: ConfusionCounts | None


@dataclass(frozen=True)
class PerTreeClassification:
    """Classification metrics for one tree."""

    tree_id: str
    metrics: ClassificationMetrics


@dataclass(frozen=True)
class AggregatedClassification:
    """Pooled and mean-per-tree classification results."""

    pooled: ClassificationMetrics
    mean_per_tree: ClassificationMetrics
    per_tree: tuple[PerTreeClassification, ...]
    num_trees: int
    num_evaluated_points: int


def _validate_zero_division(zero_division: float) -> float:
    value = float(zero_division)
    if not np.isfinite(value):
        raise ValueError("zero_division must be finite.")
    return value


def _safe_ratio(numerator: float, denominator: float, zero_division: float) -> float:
    if denominator == 0:
        return float(zero_division)
    return float(numerator / denominator)


def _as_binary_vector(values: NDArray[np.integer] | Sequence[int], name: str) -> NDArray[np.uint8]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values 0/1.")
    return array.astype(np.uint8, copy=False)


def branch_index_to_binary(
    branch_index: NDArray[np.integer] | Sequence[int],
    *,
    unsegmented_id: int = 0,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Convert branch indices to stem/non-stem labels.

    Labels at excluded positions are placeholders and must be ignored through
    the returned ``valid_mask``.
    """
    try:
        excluded_id = int(unsegmented_id)
    except (TypeError, ValueError) as exc:
        raise TypeError("unsegmented_id must be integer-like.") from exc
    if excluded_id == 1:
        raise ValueError("unsegmented_id cannot be 1 because branch index 1 is stem.")

    branch = np.asarray(branch_index)
    if branch.ndim != 1 or branch.size == 0:
        raise ValueError("branch_index must be a non-empty 1D array.")
    if not np.issubdtype(branch.dtype, np.integer):
        # Accept floating arrays only when every value is exactly integral.
        if not np.issubdtype(branch.dtype, np.floating) or not np.isfinite(branch).all():
            raise TypeError("branch_index must contain finite integer-valued data.")
        rounded = np.rint(branch)
        if not np.array_equal(branch, rounded):
            raise ValueError("branch_index contains non-integer values.")
        branch = rounded.astype(np.int64)
    else:
        branch = branch.astype(np.int64, copy=False)

    allowed = (branch == excluded_id) | (branch >= 1)
    if not np.all(allowed):
        unexpected = np.unique(branch[~allowed])
        raise ValueError(
            "branch_index contains values incompatible with the requested "
            f"unsegmented_id={excluded_id}: {unexpected[:10].tolist()}"
        )

    valid = branch != excluded_id
    labels = np.zeros(branch.shape[0], dtype=np.uint8)
    labels[branch == 1] = 1
    # branch > 1 remains non-stem (0).
    return labels, valid


def compute_confusion_counts(
    y_true: NDArray[np.integer] | Sequence[int],
    y_pred: NDArray[np.integer] | Sequence[int],
    *,
    valid_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
) -> ConfusionCounts:
    """Compute TP/TN/FP/FN for the stem-positive binary task."""
    true = _as_binary_vector(y_true, "y_true")
    pred = _as_binary_vector(y_pred, "y_pred")
    if pred.shape != true.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shape, got {true.shape} and {pred.shape}."
        )

    if valid_mask is None:
        mask = np.ones(true.shape[0], dtype=bool)
    else:
        mask = np.asarray(valid_mask)
        if mask.shape != true.shape:
            raise ValueError(
                f"valid_mask must have shape {true.shape}, got {mask.shape}."
            )
        if mask.dtype != np.bool_:
            raise TypeError("valid_mask must be boolean.")
    if not np.any(mask):
        raise ValueError("No evaluated points remain after applying valid_mask.")

    t = true[mask]
    p = pred[mask]

    tp = int(np.count_nonzero((t == 1) & (p == 1)))
    tn = int(np.count_nonzero((t == 0) & (p == 0)))
    fp = int(np.count_nonzero((t == 0) & (p == 1)))
    fn = int(np.count_nonzero((t == 1) & (p == 0)))
    counts = ConfusionCounts(tp=tp, tn=tn, fp=fp, fn=fn)

    if counts.total != int(mask.sum()):
        raise RuntimeError("Confusion counts do not sum to evaluated point count.")
    return counts


def metrics_from_counts(
    counts: ConfusionCounts,
    *,
    zero_division: float = 0.0,
) -> ClassificationMetrics:
    """Compute accuracy, precision, recall, F1 and F2 from confusion counts."""
    if not isinstance(counts, ConfusionCounts):
        raise TypeError("counts must be ConfusionCounts.")
    if min(counts.tp, counts.tn, counts.fp, counts.fn) < 0:
        raise ValueError("Confusion counts must be non-negative.")
    if counts.total <= 0:
        raise ValueError("At least one evaluated point is required.")

    zd = _validate_zero_division(zero_division)

    accuracy = _safe_ratio(counts.tp + counts.tn, counts.total, zd)
    precision = _safe_ratio(counts.tp, counts.tp + counts.fp, zd)
    recall = _safe_ratio(counts.tp, counts.tp + counts.fn, zd)

    f1_den = precision + recall
    f1 = _safe_ratio(2.0 * precision * recall, f1_den, zd)

    f2_den = 4.0 * precision + recall
    f2 = _safe_ratio(5.0 * precision * recall, f2_den, zd)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        counts=counts,
    )


def evaluate_binary_classification(
    y_true: NDArray[np.integer] | Sequence[int],
    y_pred: NDArray[np.integer] | Sequence[int],
    *,
    valid_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
    zero_division: float = 0.0,
) -> ClassificationMetrics:
    """Evaluate one binary prediction set."""
    counts = compute_confusion_counts(y_true, y_pred, valid_mask=valid_mask)
    return metrics_from_counts(counts, zero_division=zero_division)


def aggregate_per_tree_classification(
    tree_metrics: Iterable[PerTreeClassification],
    *,
    zero_division: float = 0.0,
) -> AggregatedClassification:
    """Compute pooled counts and direct mean-per-tree metric averages."""
    items = tuple(tree_metrics)
    if not items:
        raise ValueError("At least one tree is required.")

    ids = [item.tree_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("tree_id values must be unique.")

    pooled_counts = ConfusionCounts(tp=0, tn=0, fp=0, fn=0)
    for item in items:
        if not isinstance(item, PerTreeClassification):
            raise TypeError("tree_metrics must contain PerTreeClassification objects.")
        pooled_counts = pooled_counts + item.metrics.counts

    pooled = metrics_from_counts(pooled_counts, zero_division=zero_division)

    # Average each metric directly over trees.
    mean_metrics = ClassificationMetrics(
        accuracy=float(np.mean([x.metrics.accuracy for x in items])),
        precision=float(np.mean([x.metrics.precision for x in items])),
        recall=float(np.mean([x.metrics.recall for x in items])),
        f1=float(np.mean([x.metrics.f1 for x in items])),
        f2=float(np.mean([x.metrics.f2 for x in items])),
        counts=None,
    )

    return AggregatedClassification(
        pooled=pooled,
        mean_per_tree=mean_metrics,
        per_tree=items,
        num_trees=len(items),
        num_evaluated_points=pooled_counts.total,
    )


__all__ = [
    "PAPER_F_BETA",
    "ConfusionCounts",
    "ClassificationMetrics",
    "PerTreeClassification",
    "AggregatedClassification",
    "branch_index_to_binary",
    "compute_confusion_counts",
    "metrics_from_counts",
    "evaluate_binary_classification",
    "aggregate_per_tree_classification",
]
