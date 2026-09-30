"""Loss functions for TLS stem-detector training.

Author: Shahab Alaedin Baloochi
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from preprocessing import BuiltTreeDataset


@dataclass(frozen=True)
class StemDetectionLoss:
    total: Tensor
    classification: Tensor
    graph_smoothness: Tensor


def compute_positive_class_weight(trees: Sequence[BuiltTreeDataset]) -> float:
    """Compute one fixed positive-class weight from the labelled training set.

    The weight is N_non_stem / N_stem, matching the standard binary
    ``pos_weight`` convention used by BCEWithLogitsLoss.
    """
    if not trees:
        raise ValueError("At least one training tree is required.")

    positive = 0
    negative = 0
    for tree in trees:
        if tree.labels is None:
            raise ValueError(f"Training tree {tree.tree_id!r} has no labels.")
        labels = np.asarray(tree.labels.labels)
        valid = np.asarray(tree.labels.label_mask, dtype=bool)
        if labels.shape != valid.shape:
            raise ValueError(f"Label/mask shape mismatch for tree {tree.tree_id!r}.")
        positive += int(np.count_nonzero((labels == 1) & valid))
        negative += int(np.count_nonzero((labels == 0) & valid))

    if positive <= 0:
        raise ValueError("The training set contains no labelled stem points.")
    if negative <= 0:
        raise ValueError("The training set contains no labelled non-stem points.")
    return float(negative / positive)


def graph_smoothness_loss(
    probabilities: Tensor,
    edge_index: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Mean squared probability difference across fixed graph edges."""
    if probabilities.ndim != 1:
        raise ValueError("probabilities must have shape (N,).")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2,E).")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must use an integer dtype.")
    if edge_index.device != probabilities.device:
        raise ValueError("edge_index and probabilities must be on the same device.")

    if edge_index.shape[1] == 0:
        return probabilities.new_zeros(())

    source, target = edge_index[0], edge_index[1]
    if torch.any(source < 0) or torch.any(target < 0):
        raise ValueError("edge_index contains negative indices.")
    if torch.any(source >= probabilities.numel()) or torch.any(target >= probabilities.numel()):
        raise ValueError("edge_index contains an out-of-range index.")

    edge_mask: Tensor | None = None
    if valid_mask is not None:
        if valid_mask.shape != probabilities.shape:
            raise ValueError("valid_mask must have shape (N,).")
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.to(dtype=torch.bool)
        if valid_mask.device != probabilities.device:
            raise ValueError("valid_mask and probabilities must be on the same device.")
        edge_mask = valid_mask[source] & valid_mask[target]
        if not torch.any(edge_mask):
            return probabilities.new_zeros(())
        source = source[edge_mask]
        target = target[edge_mask]

    differences = probabilities[source] - probabilities[target]
    return differences.square().mean()


def stem_detection_loss(
    logits: Tensor,
    targets: Tensor,
    edge_index: Tensor,
    *,
    valid_mask: Tensor | None,
    pos_weight: float | Tensor,
    graph_smoothness_weight: float,
) -> StemDetectionLoss:
    """Class-weighted BCE on logits plus graph-smoothness regularisation."""
    if logits.ndim != 1:
        raise ValueError("logits must have shape (N,).")
    if targets.shape != logits.shape:
        raise ValueError("targets must have the same shape as logits.")
    if graph_smoothness_weight < 0.0:
        raise ValueError("graph_smoothness_weight must be non-negative.")

    if valid_mask is None:
        valid = torch.ones_like(logits, dtype=torch.bool)
    else:
        if valid_mask.shape != logits.shape:
            raise ValueError("valid_mask must have the same shape as logits.")
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)

    if not torch.any(valid):
        raise ValueError("The sampled subgraph contains no labelled points.")

    valid_targets = targets[valid].to(device=logits.device, dtype=logits.dtype)
    if torch.any((valid_targets != 0) & (valid_targets != 1)):
        raise ValueError("Labelled targets must contain only 0/1 values.")

    if isinstance(pos_weight, Tensor):
        weight = pos_weight.to(device=logits.device, dtype=logits.dtype).reshape(())
    else:
        weight = logits.new_tensor(float(pos_weight))
    if not torch.isfinite(weight) or weight <= 0:
        raise ValueError("pos_weight must be finite and positive.")

    classification = F.binary_cross_entropy_with_logits(
        logits[valid],
        valid_targets,
        pos_weight=weight,
        reduction="mean",
    )
    probabilities = torch.sigmoid(logits)
    smoothness = graph_smoothness_loss(
        probabilities,
        edge_index,
        valid_mask=valid,
    )
    total = classification + float(graph_smoothness_weight) * smoothness
    return StemDetectionLoss(
        total=total,
        classification=classification,
        graph_smoothness=smoothness,
    )
