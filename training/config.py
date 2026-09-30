"""Training configuration for TLS stem detection.

Author: Shahab Alaedin Baloochi
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


# Values reported in the manuscript.
PAPER_LEARNING_RATE = 1e-3
PAPER_WEIGHT_DECAY = 1e-4
PAPER_MAX_EPOCHS = 40
PAPER_EARLY_STOPPING_PATIENCE = 10
PAPER_SUBGRAPH_SIZE = 8192
PAPER_PRIMARY_THRESHOLD = 0.50
PAPER_CLEAN_PROBABILITY = 0.50

# Repository defaults for quantities whose numerical values are not reported.
# Both remain configurable from TrainingConfig.
DEFAULT_GRAPH_SMOOTHNESS_WEIGHT = 1e-2
DEFAULT_GRAD_CLIP_MAX_NORM = 1.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_SUBGRAPHS_PER_TREE_PER_EPOCH = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Settings used by the training loop.

    The optimizer, epoch count, early-stopping patience, subgraph size and
    primary threshold follow the manuscript. Graph-smoothness weight and the
    clipping norm are explicit repository defaults because the manuscript does
    not give their numerical values.
    """

    learning_rate: float = PAPER_LEARNING_RATE
    weight_decay: float = PAPER_WEIGHT_DECAY
    max_epochs: int = PAPER_MAX_EPOCHS
    early_stopping_patience: int = PAPER_EARLY_STOPPING_PATIENCE
    subgraph_size: int = PAPER_SUBGRAPH_SIZE
    graph_smoothness_weight: float = DEFAULT_GRAPH_SMOOTHNESS_WEIGHT
    grad_clip_max_norm: float = DEFAULT_GRAD_CLIP_MAX_NORM
    subgraphs_per_tree_per_epoch: int = DEFAULT_SUBGRAPHS_PER_TREE_PER_EPOCH
    validation_threshold: float = PAPER_PRIMARY_THRESHOLD
    validation_aggregation: Literal["pooled", "mean_per_tree"] = "pooled"
    clean_probability: float = PAPER_CLEAN_PROBABILITY
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    random_seed: int = DEFAULT_RANDOM_SEED
    use_amp: bool = False
    output_dir: Path = Path("training_output")

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.subgraph_size <= 0:
            raise ValueError("subgraph_size must be positive.")
        if self.graph_smoothness_weight < 0.0:
            raise ValueError("graph_smoothness_weight must be non-negative.")
        if self.grad_clip_max_norm <= 0.0:
            raise ValueError("grad_clip_max_norm must be positive.")
        if self.subgraphs_per_tree_per_epoch <= 0:
            raise ValueError("subgraphs_per_tree_per_epoch must be positive.")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("AdamW beta values must lie in [0,1).")
        if self.adam_eps <= 0.0:
            raise ValueError("adam_eps must be positive.")
        if not 0.0 <= self.validation_threshold <= 1.0:
            raise ValueError("validation_threshold must lie in [0,1].")
        if self.validation_aggregation not in ("pooled", "mean_per_tree"):
            raise ValueError("validation_aggregation must be 'pooled' or 'mean_per_tree'.")
        if not 0.0 <= self.clean_probability <= 1.0:
            raise ValueError("clean_probability must lie in [0,1].")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values
