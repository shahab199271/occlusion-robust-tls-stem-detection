"""Exhaustive full-tree inference for the TLS stem detector.

Author: Shahab Alaedin Baloochi

Trees are processed with non-overlapping connected subgraphs until every point
is covered exactly once. Crossing edges are omitted within each forward pass,
and subgraph outputs are scattered back to original point order. Logits are
converted to probabilities with sigmoid and a fixed decision threshold is
applied.

Whole-tree partitioning is handled by preprocessing.subgraph_sampling.
Occluded trees must be preprocessed again after point removal so features and
the Euclidean k-NN graph reflect only surviving points. Threshold selection is
handled by evaluation code; this module only applies a supplied threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from preprocessing import BuiltTreeDataset, InferencePartition
from preprocessing.subgraph_sampling import validate_inference_partition


PAPER_PRIMARY_THRESHOLD = 0.50
PAPER_SENSITIVITY_THRESHOLD = 0.63


@dataclass(frozen=True)
class FullTreeInferenceResult:
    """Raw full-tree inference outputs aligned to original point order."""

    tree_id: str
    logits: NDArray[np.float32]
    probabilities: NDArray[np.float32]
    predictions: NDArray[np.uint8]
    threshold: float
    subgraph_index: NDArray[np.int32]
    subgraph_sizes: NDArray[np.int64]

    @property
    def num_points(self) -> int:
        return int(self.logits.shape[0])

    @property
    def num_subgraphs(self) -> int:
        return int(self.subgraph_sizes.shape[0])

    @property
    def num_predicted_stem(self) -> int:
        return int(self.predictions.sum())


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be finite and lie in [0, 1].")
    return value


def _validate_feature_arrays(
    node_features: NDArray[np.floating],
    positions: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features = np.asarray(node_features)
    xyz = np.asarray(positions)

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"node_features must have shape (N,C) with N>0, got {features.shape}."
        )
    if xyz.ndim != 2 or xyz.shape != (features.shape[0], 3):
        raise ValueError(
            "positions must have shape (N,3) and align one-to-one with "
            f"node_features; got {xyz.shape}."
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError("node_features must have a floating-point dtype.")
    if not np.issubdtype(xyz.dtype, np.floating):
        raise TypeError("positions must have a floating-point dtype.")
    if not np.isfinite(features).all():
        raise ValueError("node_features contains NaN or infinite values.")
    if not np.isfinite(xyz).all():
        raise ValueError("positions contains NaN or infinite values.")

    # Preserve the stored preprocessing dtype here. Conversion to the model's
    # dtype happens one subgraph at a time, avoiding a second full-tree copy.
    return features, xyz


def _validate_partition_structure(
    partition: InferencePartition,
    num_points: int,
) -> NDArray[np.int32]:
    """Validate exact-once coverage and local edge indexing without a global graph."""
    if not isinstance(partition, InferencePartition):
        raise TypeError("partition must be preprocessing.InferencePartition.")
    if int(partition.num_points) != int(num_points):
        raise ValueError(
            f"partition.num_points={partition.num_points} does not match N={num_points}."
        )
    if int(partition.target_size) <= 0:
        raise ValueError("partition.target_size must be positive.")
    if not partition.subgraphs:
        raise ValueError("partition contains no inference subgraphs.")

    assignment = np.full(num_points, -1, dtype=np.int32)
    counts = np.zeros(num_points, dtype=np.int32)

    for block_id, subgraph in enumerate(partition.subgraphs):
        ids = np.asarray(subgraph.point_ids, dtype=np.int64)
        edge_index = np.asarray(subgraph.edge_index, dtype=np.int64)

        if ids.ndim != 1 or ids.size == 0:
            raise ValueError(f"Subgraph {block_id} has invalid/empty point_ids.")
        if ids.size > partition.target_size:
            raise ValueError(
                f"Subgraph {block_id} has {ids.size} nodes, exceeding "
                f"target_size={partition.target_size}."
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(f"Subgraph {block_id} contains duplicate point IDs.")
        if np.any(ids < 0) or np.any(ids >= num_points):
            raise ValueError(f"Subgraph {block_id} contains out-of-range point IDs.")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Subgraph {block_id} edge_index must have shape (2,E), "
                f"got {edge_index.shape}."
            )
        if edge_index.size:
            if np.any(edge_index < 0) or np.any(edge_index >= ids.size):
                raise ValueError(
                    f"Subgraph {block_id} contains an invalid local edge index."
                )
            if np.any(edge_index[0] == edge_index[1]):
                raise ValueError(f"Subgraph {block_id} contains a self-edge.")

        counts[ids] += 1
        assignment[ids] = np.int32(block_id)

    missing = int(np.sum(counts == 0))
    repeated = int(np.sum(counts > 1))
    if missing or repeated:
        raise ValueError(
            "Inference partition must cover every point exactly once; "
            f"missing={missing}, repeated={repeated}."
        )
    if np.any(assignment < 0):
        raise RuntimeError("Internal error: not every point received a subgraph index.")

    return assignment


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer the device and floating dtype from the model without moving it."""
    tensors = list(model.parameters()) + list(model.buffers())
    if not tensors:
        return torch.device("cpu"), torch.float32

    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(
            "Model parameters/buffers span multiple devices. Move the model to "
            "one device before inference."
        )
    device = next(iter(devices))

    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if not floating:
        return device, torch.float32
    dtypes = {tensor.dtype for tensor in floating}
    if len(dtypes) != 1:
        raise ValueError(
            "Model floating parameters/buffers use multiple dtypes. Normalize "
            "the model dtype before inference."
        )
    return device, next(iter(dtypes))


def _snapshot_training_flags(model: nn.Module) -> list[tuple[nn.Module, bool]]:
    return [(module, bool(module.training)) for module in model.modules()]


def _restore_training_flags(states: list[tuple[nn.Module, bool]]) -> None:
    # Assign flags directly so mixed train/eval states are restored exactly;
    # calling root.train(...) would recursively overwrite child-specific states.
    for module, state in states:
        module.training = state


def _forward_one_subgraph(
    model: nn.Module,
    features: NDArray[np.floating],
    positions: NDArray[np.floating],
    edge_index: NDArray[np.int64],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    x = torch.as_tensor(features, dtype=dtype, device=device)
    pos = torch.as_tensor(positions, dtype=dtype, device=device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    output = model(x, pos, edges)
    if not isinstance(output, Tensor):
        raise TypeError(
            "The inference model must return a torch.Tensor of per-point logits "
            "when called as model(node_features, positions, edge_index)."
        )
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or output.shape[0] != x.shape[0]:
        raise RuntimeError(
            "Model must return exactly one logit per subgraph point; "
            f"got shape {tuple(output.shape)} for {x.shape[0]} points."
        )
    if not output.is_floating_point():
        raise RuntimeError("Model logits must have a floating-point dtype.")
    if not torch.isfinite(output).all():
        raise RuntimeError("Model produced NaN or infinite logits.")
    return output


def infer_partition(
    model: nn.Module,
    node_features: NDArray[np.floating],
    positions: NDArray[np.floating],
    partition: InferencePartition,
    *,
    threshold: float = PAPER_PRIMARY_THRESHOLD,
    tree_id: str = "",
) -> FullTreeInferenceResult:
    """Run sequential, non-overlapping inference over an existing partition.

    This low-level function validates exact-once point coverage and local edge
    index ranges. It cannot verify that local edges are the exact induced edges
    of a particular whole-tree graph because that graph is not supplied here.
    ``infer_tree_dataset`` performs the stronger repository-level validation.
    """
    threshold_value = _validate_threshold(threshold)
    features, xyz = _validate_feature_arrays(node_features, positions)
    num_points = int(features.shape[0])

    model_input_channels = getattr(model, "input_channels", None)
    if model_input_channels is not None:
        try:
            expected_channels = int(model_input_channels)
        except (TypeError, ValueError) as exc:
            raise TypeError("model.input_channels must be integer-like.") from exc
        if features.shape[1] != expected_channels:
            raise ValueError(
                f"Model expects {expected_channels} input channels, but "
                f"node_features has {features.shape[1]}."
            )

    subgraph_index = _validate_partition_structure(partition, num_points)
    device, dtype = _model_device_and_dtype(model)

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"Unsupported model floating dtype for inference: {dtype}.")

    logits = np.empty(num_points, dtype=np.float32)
    probabilities = np.empty(num_points, dtype=np.float32)
    written = np.zeros(num_points, dtype=bool)

    training_states = _snapshot_training_flags(model)
    model.eval()

    try:
        with torch.inference_mode():
            for block_id, subgraph in enumerate(partition.subgraphs):
                ids = np.asarray(subgraph.point_ids, dtype=np.int64)
                local_edges = np.asarray(subgraph.edge_index, dtype=np.int64)

                block_logits = _forward_one_subgraph(
                    model,
                    features[ids],
                    xyz[ids],
                    local_edges,
                    device=device,
                    dtype=dtype,
                )
                # Store full-tree outputs in float32. Compute sigmoid *after*
                # converting logits to float32 so logits and probabilities stay
                # numerically self-consistent even when the model itself runs in
                # float16/bfloat16.
                block_logits_f32 = block_logits.detach().to(dtype=torch.float32)
                block_probs_f32 = torch.sigmoid(block_logits_f32)
                block_logits_np = block_logits_f32.to(device="cpu").numpy()
                block_probs_np = block_probs_f32.to(device="cpu").numpy()

                if written[ids].any():
                    raise RuntimeError(
                        f"Subgraph {block_id} attempted to overwrite an already "
                        "predicted point."
                    )
                logits[ids] = block_logits_np
                probabilities[ids] = block_probs_np
                written[ids] = True
    finally:
        _restore_training_flags(training_states)

    if not written.all():
        raise RuntimeError(
            f"Inference ended with {int((~written).sum())} unpredicted points."
        )
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Full-tree inference contains NaN or infinite outputs.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("Sigmoid probabilities fall outside [0,1].")

    predictions = (probabilities >= threshold_value).astype(np.uint8, copy=False)
    sizes = np.asarray(
        [subgraph.num_nodes for subgraph in partition.subgraphs],
        dtype=np.int64,
    )

    result = FullTreeInferenceResult(
        tree_id=str(tree_id),
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        threshold=threshold_value,
        subgraph_index=subgraph_index,
        subgraph_sizes=sizes,
    )
    validate_inference_result(result)
    return result


def infer_tree_dataset(
    model: nn.Module,
    dataset: BuiltTreeDataset,
    *,
    threshold: float = PAPER_PRIMARY_THRESHOLD,
    partition: Optional[InferencePartition] = None,
    validate_partition: bool = True,
) -> FullTreeInferenceResult:
    """Run exhaustive inference on a preprocessed whole tree.

    Uses dataset.node_features as model input and dataset.features.local_xyz
    as positional coordinates. Labels are not used during inference.
    """
    if not isinstance(dataset, BuiltTreeDataset):
        raise TypeError("dataset must be preprocessing.BuiltTreeDataset.")

    node_features = np.asarray(dataset.node_features)
    local_xyz = np.asarray(dataset.features.local_xyz)

    if node_features.ndim != 2 or node_features.shape[0] != dataset.num_points:
        raise ValueError("dataset.node_features is not aligned with dataset points.")
    if local_xyz.shape != (dataset.num_points, 3):
        raise ValueError("dataset.features.local_xyz must have shape (N,3).")

    # The repository's 10D representation begins with the same local metric XYZ
    # coordinates used for positional attention. Checking this here prevents a
    # silent point-order or preprocessing mismatch.
    if node_features.shape[1] >= 3 and not np.allclose(
        node_features[:, :3],
        local_xyz,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The first three node-feature channels do not match local_xyz."
        )

    sampler = dataset.make_sampler()
    current_partition = (
        sampler.partition_inference() if partition is None else partition
    )

    if validate_partition:
        checks = validate_inference_partition(sampler, current_partition)
        if not bool(checks.get("all_checks_pass", False)):
            raise RuntimeError(
                "Repository inference-partition validation failed: "
                f"{checks}"
            )

    return infer_partition(
        model,
        node_features,
        local_xyz,
        current_partition,
        threshold=threshold,
        tree_id=dataset.tree_id,
    )


def validate_inference_result(result: FullTreeInferenceResult) -> dict[str, bool | int]:
    """Validate internal consistency of a full-tree inference result."""
    if not isinstance(result, FullTreeInferenceResult):
        raise TypeError("result must be FullTreeInferenceResult.")

    n = result.logits.shape[0]
    threshold_valid = bool(
        np.isfinite(result.threshold) and 0.0 <= float(result.threshold) <= 1.0
    )
    assigned_ids = (
        np.unique(result.subgraph_index)
        if result.subgraph_index.shape == (n,) and n > 0
        else np.empty(0, dtype=np.int32)
    )
    expected_ids = np.arange(result.subgraph_sizes.size, dtype=np.int32)

    checks: dict[str, bool | int] = {
        "threshold_valid": threshold_valid,
        "nonempty": bool(n > 0),
        "probability_shape": bool(result.probabilities.shape == (n,)),
        "prediction_shape": bool(result.predictions.shape == (n,)),
        "subgraph_index_shape": bool(result.subgraph_index.shape == (n,)),
        "finite_logits": bool(np.isfinite(result.logits).all()),
        "finite_probabilities": bool(np.isfinite(result.probabilities).all()),
        "probabilities_in_unit_interval": bool(
            np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0))
        ),
        "binary_predictions": bool(
            np.all((result.predictions == 0) | (result.predictions == 1))
        ),
        "threshold_consistent": bool(
            np.array_equal(
                result.predictions,
                (result.probabilities >= result.threshold).astype(np.uint8),
            )
        ),
        "all_points_assigned_to_subgraph": bool(
            np.all(result.subgraph_index >= 0)
        ),
        "subgraph_count_consistent": bool(
            result.subgraph_sizes.ndim == 1
            and result.subgraph_sizes.size > 0
            and np.all(result.subgraph_sizes > 0)
            and int(result.subgraph_sizes.sum()) == n
            and np.array_equal(assigned_ids, expected_ids)
        ),
    }

    # Sigmoid/logit consistency is checked in float32 because stored inference
    # Stored inference arrays use float32.
    expected = torch.sigmoid(torch.from_numpy(result.logits)).numpy()
    checks["sigmoid_consistent"] = bool(
        np.allclose(
            result.probabilities,
            expected,
            rtol=2e-6,
            atol=2e-7,
        )
    )

    checks["all_checks_pass"] = bool(
        all(bool(value) for key, value in checks.items() if key != "all_checks_pass")
    )
    return checks


def save_inference_npz(
    result: FullTreeInferenceResult,
    output_path: str | Path,
) -> Path:
    """Save raw full-tree outputs without changing original point order."""
    checks = validate_inference_result(result)
    if not bool(checks["all_checks_pass"]):
        raise ValueError(f"Invalid inference result: {checks}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tree_id=np.asarray(result.tree_id),
        logits=result.logits,
        probabilities=result.probabilities,
        predictions=result.predictions,
        threshold=np.asarray(result.threshold, dtype=np.float64),
        subgraph_index=result.subgraph_index,
        subgraph_sizes=result.subgraph_sizes,
    )
    return path


__all__ = [
    "PAPER_PRIMARY_THRESHOLD",
    "PAPER_SENSITIVITY_THRESHOLD",
    "FullTreeInferenceResult",
    "infer_partition",
    "infer_tree_dataset",
    "validate_inference_result",
    "save_inference_npz",
]