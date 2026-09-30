"""Training loop for the graph-based TLS stem detector.

Author: Shahab Alaedin Baloochi
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from evaluation import ProbabilityTree, evaluate_probability_trees
from inference import infer_tree_dataset
from preprocessing import BuiltTreeDataset

from .config import TrainingConfig
from .losses import compute_positive_class_weight, stem_detection_loss


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_classification_loss: float
    train_graph_smoothness_loss: float
    mean_gradient_norm: float
    validation_f2: float


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_f2: float
    epochs_completed: int
    stopped_early: bool
    positive_class_weight: float
    history: tuple[EpochRecord, ...]
    best_checkpoint: Path
    last_checkpoint: Path


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _validate_tree_collection(trees: Sequence[BuiltTreeDataset], name: str) -> None:
    if not trees:
        raise ValueError(f"{name} must contain at least one tree.")
    ids = [str(tree.tree_id) for tree in trees]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate tree IDs.")
    for tree in trees:
        if tree.labels is None:
            raise ValueError(f"{name} tree {tree.tree_id!r} has no labels.")


def _choose_training_tree(
    clean_tree: BuiltTreeDataset,
    occluded_variants: Mapping[str, Sequence[BuiltTreeDataset]] | None,
    *,
    clean_probability: float,
    rng: np.random.Generator,
) -> BuiltTreeDataset:
    if occluded_variants is None or rng.random() < clean_probability:
        return clean_tree

    variants = tuple(occluded_variants.get(str(clean_tree.tree_id), ()))
    if not variants:
        return clean_tree
    selected = int(rng.integers(0, len(variants)))
    return variants[selected]


def _sample_labelled_subgraph(
    tree: BuiltTreeDataset,
    rng: np.random.Generator,
    *,
    requested_stratum: int | None = None,
    max_attempts: int = 12,
):
    if tree.labels is None:
        raise ValueError(f"Tree {tree.tree_id!r} has no labels.")
    sampler = tree.make_sampler()
    last_sampling_error: RuntimeError | None = None
    for _ in range(max_attempts):
        try:
            sample = sampler.sample_training(
                rng=rng,
                requested_stratum=requested_stratum,
                require_full_size=True,
            )
        except RuntimeError as exc:
            last_sampling_error = exc
            continue
        if np.any(tree.labels.label_mask[sample.point_ids]):
            return sample

    message = (
        f"Could not sample a labelled {tree.subgraph_size}-point subgraph "
        f"from tree {tree.tree_id!r} after {max_attempts} attempts."
    )
    if last_sampling_error is not None:
        raise RuntimeError(message) from last_sampling_error
    raise RuntimeError(message)


def _subgraph_tensors(
    tree: BuiltTreeDataset,
    point_ids: np.ndarray,
    edge_index: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if tree.labels is None:
        raise ValueError(f"Tree {tree.tree_id!r} has no labels.")

    features = torch.as_tensor(
        np.asarray(tree.node_features)[point_ids],
        dtype=torch.float32,
        device=device,
    )
    positions = torch.as_tensor(
        np.asarray(tree.features.local_xyz)[point_ids],
        dtype=torch.float32,
        device=device,
    )
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    targets = torch.as_tensor(
        np.asarray(tree.labels.labels)[point_ids],
        dtype=torch.float32,
        device=device,
    )
    valid_mask = torch.as_tensor(
        np.asarray(tree.labels.label_mask)[point_ids],
        dtype=torch.bool,
        device=device,
    )
    return features, positions, edges, targets, valid_mask


def train_one_epoch(
    model: nn.Module,
    train_trees: Sequence[BuiltTreeDataset],
    optimizer: torch.optim.Optimizer,
    *,
    config: TrainingConfig,
    pos_weight: float,
    device: torch.device,
    rng: np.random.Generator,
    occluded_variants: Mapping[str, Sequence[BuiltTreeDataset]] | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, float, float]:
    """Train for one epoch using height-stratified connected subgraphs."""
    model.train()
    order = rng.permutation(len(train_trees))

    total_loss = 0.0
    total_bce = 0.0
    total_smooth = 0.0
    total_grad_norm = 0.0
    steps = 0

    amp_enabled = bool(config.use_amp and device.type == "cuda")
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for tree_index in order:
        clean_tree = train_trees[int(tree_index)]
        for sample_index in range(config.subgraphs_per_tree_per_epoch):
            tree = _choose_training_tree(
                clean_tree,
                occluded_variants,
                clean_probability=config.clean_probability,
                rng=rng,
            )
            # The sampler selects lower, middle, or crown strata uniformly
            # before choosing the BFS seed within that stratum.
            sample = _sample_labelled_subgraph(
                tree,
                rng,
                requested_stratum=None,
            )
            features, positions, edges, targets, valid_mask = _subgraph_tensors(
                tree,
                sample.point_ids,
                sample.edge_index,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(features, positions, edges)
                if not isinstance(logits, Tensor):
                    raise RuntimeError("The model must return one logit tensor per point.")
                losses = stem_detection_loss(
                    logits,
                    targets,
                    edges,
                    valid_mask=valid_mask,
                    pos_weight=pos_weight,
                    graph_smoothness_weight=config.graph_smoothness_weight,
                )

            scaler.scale(losses.total).backward()
            if amp_enabled:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.grad_clip_max_norm,
                norm_type=2.0,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(losses.total.detach().cpu())
            total_bce += float(losses.classification.detach().cpu())
            total_smooth += float(losses.graph_smoothness.detach().cpu())
            total_grad_norm += float(grad_norm.detach().cpu())
            steps += 1

    if steps == 0:
        raise RuntimeError("No optimisation steps were performed.")

    return (
        total_loss / steps,
        total_bce / steps,
        total_smooth / steps,
        total_grad_norm / steps,
    )


def validation_f2(
    model: nn.Module,
    validation_trees: Sequence[BuiltTreeDataset],
    *,
    threshold: float,
    aggregation: str,
) -> float:
    """Evaluate F2 from exhaustive full-tree validation predictions."""
    probability_trees: list[ProbabilityTree] = []
    sentinel_ids = {
        int(tree.labels.unsegmented_id)
        for tree in validation_trees
        if tree.labels is not None
    }
    if len(sentinel_ids) != 1:
        raise ValueError("Validation trees must use one consistent unsegmented_id.")
    unsegmented_id = next(iter(sentinel_ids))

    for tree in validation_trees:
        if tree.labels is None:
            raise ValueError(f"Validation tree {tree.tree_id!r} has no labels.")
        result = infer_tree_dataset(model, tree, threshold=threshold)
        probability_trees.append(
            ProbabilityTree(
                tree_id=str(tree.tree_id),
                probabilities=result.probabilities,
                branch_index=tree.labels.branch_ids,
            )
        )

    evaluated = evaluate_probability_trees(
        probability_trees,
        threshold=threshold,
        unsegmented_id=unsegmented_id,
    )
    if aggregation == "pooled":
        return float(evaluated.pooled.f2)
    if aggregation == "mean_per_tree":
        return float(evaluated.mean_per_tree.f2)
    raise ValueError("aggregation must be 'pooled' or 'mean_per_tree'.")


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_f2_value: float,
    pos_weight: float,
    config: TrainingConfig,
) -> dict[str, object]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_f2": float(validation_f2_value),
        "positive_class_weight": float(pos_weight),
        "config": config.to_dict(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def fit(
    model: nn.Module,
    train_trees: Sequence[BuiltTreeDataset],
    validation_trees: Sequence[BuiltTreeDataset],
    *,
    config: TrainingConfig = TrainingConfig(),
    device: str | torch.device | None = None,
    occluded_variants: Mapping[str, Sequence[BuiltTreeDataset]] | None = None,
    test_tree_ids: Sequence[str] | None = None,
) -> TrainingResult:
    """Train with AdamW and retain the checkpoint with the best validation F2."""
    _validate_tree_collection(train_trees, "train_trees")
    _validate_tree_collection(validation_trees, "validation_trees")

    train_ids = {str(tree.tree_id) for tree in train_trees}
    validation_ids = {str(tree.tree_id) for tree in validation_trees}
    overlap = train_ids & validation_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Training/validation tree leakage detected: {preview}")

    test_ids = tuple(str(tree_id) for tree_id in (test_tree_ids or ()))
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("test_tree_ids contains duplicates.")
    test_id_set = set(test_ids)
    if test_id_set & train_ids or test_id_set & validation_ids:
        raise ValueError("Test tree IDs must be disjoint from training and validation IDs.")

    for tree in (*train_trees, *validation_trees):
        if int(tree.subgraph_size) != int(config.subgraph_size):
            raise ValueError(
                f"Tree {tree.tree_id!r} uses subgraph_size={tree.subgraph_size}; "
                f"config requires {config.subgraph_size}."
            )

    if occluded_variants is not None:
        if set(occluded_variants) != train_ids:
            missing = sorted(train_ids - set(occluded_variants))
            extra = sorted(set(occluded_variants) - train_ids)
            raise ValueError(
                "Occlusion-aware training requires variants for every training tree; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
        for tree_id, variants in occluded_variants.items():
            variants = tuple(variants)
            if len(variants) != 2:
                raise ValueError(
                    f"Training tree {tree_id!r} must provide exactly two occluded variants."
                )
            for variant in variants:
                if variant.labels is None:
                    raise ValueError(f"Occluded variant {variant.tree_id!r} has no labels.")
                if int(variant.subgraph_size) != int(config.subgraph_size):
                    raise ValueError(
                        f"Occluded variant {variant.tree_id!r} has incompatible subgraph size."
                    )

    set_random_seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(target_device)

    pos_weight = compute_positive_class_weight(train_trees)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    history_path = output_dir / "history.json"
    split_path = output_dir / "split_manifest.json"
    summary_path = output_dir / "summary.json"

    _write_json(
        split_path,
        {
            "train": [str(tree.tree_id) for tree in train_trees],
            "validation": [str(tree.tree_id) for tree in validation_trees],
            "test": list(test_ids),
        },
    )

    amp_enabled = bool(config.use_amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[EpochRecord] = []
    best_f2 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        train_loss, bce_loss, smooth_loss, grad_norm = train_one_epoch(
            model,
            train_trees,
            optimizer,
            config=config,
            pos_weight=pos_weight,
            device=target_device,
            rng=rng,
            occluded_variants=occluded_variants,
            scaler=scaler,
        )
        val_f2 = validation_f2(
            model,
            validation_trees,
            threshold=config.validation_threshold,
            aggregation=config.validation_aggregation,
        )

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_classification_loss=bce_loss,
            train_graph_smoothness_loss=smooth_loss,
            mean_gradient_norm=grad_norm,
            validation_f2=val_f2,
        )
        history.append(record)

        payload = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            validation_f2_value=val_f2,
            pos_weight=pos_weight,
            config=config,
        )
        torch.save(payload, last_path)

        if val_f2 > best_f2:
            best_f2 = val_f2
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(payload, best_path)
        else:
            epochs_without_improvement += 1

        _write_json(history_path, [asdict(item) for item in history])

        if epochs_without_improvement >= config.early_stopping_patience:
            stopped_early = True
            break

    if best_epoch <= 0 or not best_path.exists():
        raise RuntimeError("Training finished without a valid best checkpoint.")

    best_state = torch.load(best_path, map_location=target_device)
    model.load_state_dict(best_state["model_state_dict"], strict=True)

    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_f2=best_f2,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        positive_class_weight=pos_weight,
        history=tuple(history),
        best_checkpoint=best_path,
        last_checkpoint=last_path,
    )
    _write_json(
        summary_path,
        {
            "best_epoch": result.best_epoch,
            "best_validation_f2": result.best_validation_f2,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "positive_class_weight": result.positive_class_weight,
            "best_checkpoint": str(result.best_checkpoint),
            "last_checkpoint": str(result.last_checkpoint),
            "config": config.to_dict(),
        },
    )
    return result
