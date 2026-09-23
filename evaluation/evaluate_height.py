"""Relative-height recall and tree-level stem-height evaluation.

Author: Shahab Alaedin Baloochi

Stem recall is computed independently in lower, middle and upper thirds of
relative tree height for each tree and then averaged across trees.

Predicted and reference stem heights are vertical z ranges. Per-tree signed
relative height error is ``(H_pred - H_ref) / H_ref * 100``. The summary
contains mean absolute relative error, maximum absolute relative error and the
aggregate relative height bias based on summed predicted and reference heights.

Relative-height bins use [0, 1/3), [1/3, 2/3), and [2/3, 1]. A bin with no
reference stem points has undefined recall (NaN) and is excluded from that
bin's cross-tree mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .classification_metrics import branch_index_to_binary


HEIGHT_BIN_NAMES = ("lower", "middle", "upper")


@dataclass(frozen=True)
class TreeHeightBinRecall:
    tree_id: str
    lower: float
    middle: float
    upper: float
    stem_support_lower: int
    stem_support_middle: int
    stem_support_upper: int


@dataclass(frozen=True)
class MeanHeightBinRecall:
    lower: float
    middle: float
    upper: float
    contributing_trees_lower: int
    contributing_trees_middle: int
    contributing_trees_upper: int
    per_tree: tuple[TreeHeightBinRecall, ...]


@dataclass(frozen=True)
class TreeStemHeight:
    tree_id: str
    predicted_height: float
    reference_height: float
    signed_relative_error_percent: float
    absolute_relative_error_percent: float


@dataclass(frozen=True)
class StemHeightSummary:
    mean_absolute_relative_error_percent: float
    maximum_absolute_relative_error_percent: float
    relative_height_bias_percent: float
    per_tree: tuple[TreeStemHeight, ...]


def _validate_points(points: NDArray[np.floating]) -> NDArray[np.float64]:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"points must have shape (N,3) with N>0, got {xyz.shape}.")
    if not np.isfinite(xyz).all():
        raise ValueError("points contains NaN or infinite coordinates.")
    if float(xyz[:, 2].max() - xyz[:, 2].min()) <= 0.0:
        raise ValueError("Tree vertical range must be positive.")
    return xyz


def relative_tree_height(points: NDArray[np.floating]) -> NDArray[np.float64]:
    """Compute (z-z_min)/(z_max-z_min) for the complete tree."""
    xyz = _validate_points(points)
    z = xyz[:, 2]
    return (z - z.min()) / (z.max() - z.min())


def relative_height_bin_ids(points: NDArray[np.floating]) -> NDArray[np.int8]:
    """Assign lower/middle/upper equal relative-height thirds."""
    h = relative_tree_height(points)
    ids = np.empty(h.shape[0], dtype=np.int8)
    ids[h < (1.0 / 3.0)] = 0
    ids[(h >= (1.0 / 3.0)) & (h < (2.0 / 3.0))] = 1
    ids[h >= (2.0 / 3.0)] = 2
    return ids


def evaluate_tree_height_bin_recall(
    tree_id: str,
    points: NDArray[np.floating],
    branch_index: NDArray[np.integer],
    prediction_mask: NDArray[np.bool_] | NDArray[np.integer],
    *,
    unsegmented_id: int = 0,
) -> TreeHeightBinRecall:
    """Compute one tree's stem recall independently within each height third."""
    xyz = _validate_points(points)
    labels, valid = branch_index_to_binary(
        branch_index,
        unsegmented_id=unsegmented_id,
    )
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("branch_index and points must align one-to-one.")

    pred = np.asarray(prediction_mask)
    if pred.shape != labels.shape:
        raise ValueError("prediction_mask and points must align one-to-one.")
    if pred.dtype != np.bool_:
        if not np.all((pred == 0) | (pred == 1)):
            raise ValueError("prediction_mask must contain only bool/0/1 values.")
        pred = pred.astype(bool)
    else:
        pred = pred.astype(bool, copy=False)

    bins = relative_height_bin_ids(xyz)

    recalls: list[float] = []
    supports: list[int] = []
    for bin_id in range(3):
        eval_mask = valid & (bins == bin_id)
        true_stem = eval_mask & (labels == 1)
        support = int(true_stem.sum())
        supports.append(support)
        if support == 0:
            recalls.append(float("nan"))
        else:
            tp = int(np.count_nonzero(true_stem & pred))
            recalls.append(float(tp / support))

    return TreeHeightBinRecall(
        tree_id=str(tree_id),
        lower=recalls[0],
        middle=recalls[1],
        upper=recalls[2],
        stem_support_lower=supports[0],
        stem_support_middle=supports[1],
        stem_support_upper=supports[2],
    )


def mean_height_bin_recall(
    per_tree: Sequence[TreeHeightBinRecall],
) -> MeanHeightBinRecall:
    """Average defined per-tree recalls independently for each height bin."""
    items = tuple(per_tree)
    if not items:
        raise ValueError("At least one tree is required.")
    ids = [x.tree_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("tree_id values must be unique.")

    matrix = np.asarray(
        [[x.lower, x.middle, x.upper] for x in items],
        dtype=np.float64,
    )
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("At least one height bin has no tree with reference stem support.")

    means = np.asarray(
        [
            np.mean(matrix[finite[:, j], j])
            for j in range(3)
        ],
        dtype=np.float64,
    )

    return MeanHeightBinRecall(
        lower=float(means[0]),
        middle=float(means[1]),
        upper=float(means[2]),
        contributing_trees_lower=int(counts[0]),
        contributing_trees_middle=int(counts[1]),
        contributing_trees_upper=int(counts[2]),
        per_tree=items,
    )


def _vertical_range(points: NDArray[np.float64], mask: NDArray[np.bool_], name: str) -> float:
    if mask.shape != (points.shape[0],):
        raise ValueError(f"{name} must have shape ({points.shape[0]},).")
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        raise ValueError(f"{name} is empty; stem height is undefined.")
    z = points[ids, 2]
    height = float(z.max() - z.min())
    if height <= 0.0:
        raise ValueError(f"{name} has zero vertical range.")
    return height


def evaluate_tree_stem_height(
    tree_id: str,
    points: NDArray[np.floating],
    branch_index: NDArray[np.integer],
    final_prediction_mask: NDArray[np.bool_] | NDArray[np.integer],
    *,
    unsegmented_id: int = 0,
) -> TreeStemHeight:
    """Compute predicted/reference vertical stem ranges and relative error."""
    xyz = _validate_points(points)
    branch = np.asarray(branch_index)
    labels, _ = branch_index_to_binary(
        branch,
        unsegmented_id=unsegmented_id,
    )
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("branch_index and points must align one-to-one.")

    pred = np.asarray(final_prediction_mask)
    if pred.shape != labels.shape:
        raise ValueError("final_prediction_mask and points must align one-to-one.")
    if pred.dtype != np.bool_:
        if not np.all((pred == 0) | (pred == 1)):
            raise ValueError("final_prediction_mask must contain only bool/0/1 values.")
        pred = pred.astype(bool)
    else:
        pred = pred.astype(bool, copy=False)

    reference = labels == 1

    h_pred = _vertical_range(xyz, pred, "final_prediction_mask")
    h_ref = _vertical_range(xyz, reference, "reference stem mask")

    signed = (h_pred - h_ref) / h_ref * 100.0
    return TreeStemHeight(
        tree_id=str(tree_id),
        predicted_height=h_pred,
        reference_height=h_ref,
        signed_relative_error_percent=float(signed),
        absolute_relative_error_percent=float(abs(signed)),
    )


def summarize_stem_heights(
    per_tree: Sequence[TreeStemHeight],
) -> StemHeightSummary:
    """Compute MARE, maximum absolute error and aggregate relative height bias."""
    items = tuple(per_tree)
    if not items:
        raise ValueError("At least one tree is required.")
    ids = [x.tree_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("tree_id values must be unique.")

    pred = np.asarray([x.predicted_height for x in items], dtype=np.float64)
    ref = np.asarray([x.reference_height for x in items], dtype=np.float64)
    signed = np.asarray(
        [x.signed_relative_error_percent for x in items],
        dtype=np.float64,
    )
    absolute = np.abs(signed)

    if not (
        np.isfinite(pred).all()
        and np.isfinite(ref).all()
        and np.isfinite(signed).all()
        and np.all(pred > 0.0)
        and np.all(ref > 0.0)
    ):
        raise ValueError("Stem-height inputs must be finite and positive.")

    bias = (pred.sum() - ref.sum()) / ref.sum() * 100.0

    return StemHeightSummary(
        mean_absolute_relative_error_percent=float(absolute.mean()),
        maximum_absolute_relative_error_percent=float(absolute.max()),
        relative_height_bias_percent=float(bias),
        per_tree=items,
    )


__all__ = [
    "HEIGHT_BIN_NAMES",
    "TreeHeightBinRecall",
    "MeanHeightBinRecall",
    "TreeStemHeight",
    "StemHeightSummary",
    "relative_tree_height",
    "relative_height_bin_ids",
    "evaluate_tree_height_bin_recall",
    "mean_height_bin_recall",
    "evaluate_tree_stem_height",
    "summarize_stem_heights",
]
