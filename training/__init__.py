"""Training utilities for occlusion-robust TLS stem detection.

Author: Shahab Alaedin Baloochi
"""

from .config import (
    DEFAULT_GRAD_CLIP_MAX_NORM,
    DEFAULT_GRAPH_SMOOTHNESS_WEIGHT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SUBGRAPHS_PER_TREE_PER_EPOCH,
    PAPER_CLEAN_PROBABILITY,
    PAPER_EARLY_STOPPING_PATIENCE,
    PAPER_LEARNING_RATE,
    PAPER_MAX_EPOCHS,
    PAPER_PRIMARY_THRESHOLD,
    PAPER_SUBGRAPH_SIZE,
    PAPER_WEIGHT_DECAY,
    TrainingConfig,
)
from .losses import (
    StemDetectionLoss,
    compute_positive_class_weight,
    graph_smoothness_loss,
    stem_detection_loss,
)
from .trainer import (
    EpochRecord,
    TrainingResult,
    fit,
    set_random_seed,
    train_one_epoch,
    validation_f2,
)

__all__ = [
    "PAPER_LEARNING_RATE",
    "PAPER_WEIGHT_DECAY",
    "PAPER_MAX_EPOCHS",
    "PAPER_EARLY_STOPPING_PATIENCE",
    "PAPER_SUBGRAPH_SIZE",
    "PAPER_PRIMARY_THRESHOLD",
    "PAPER_CLEAN_PROBABILITY",
    "DEFAULT_GRAPH_SMOOTHNESS_WEIGHT",
    "DEFAULT_GRAD_CLIP_MAX_NORM",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_SUBGRAPHS_PER_TREE_PER_EPOCH",
    "TrainingConfig",
    "StemDetectionLoss",
    "compute_positive_class_weight",
    "graph_smoothness_loss",
    "stem_detection_loss",
    "EpochRecord",
    "TrainingResult",
    "set_random_seed",
    "train_one_epoch",
    "validation_f2",
    "fit",
]
