"""Evaluation utilities for TLS stem detection.

Author: Shahab Alaedin Baloochi

Exports full-tree classification metrics, validation threshold search,
height-stratified stem recall, tree-level stem-height metrics and tree-level
stem-volume error metrics.
"""

from .classification_metrics import (
    PAPER_F_BETA,
    AggregatedClassification,
    ClassificationMetrics,
    ConfusionCounts,
    PerTreeClassification,
    aggregate_per_tree_classification,
    branch_index_to_binary,
    compute_confusion_counts,
    evaluate_binary_classification,
    metrics_from_counts,
)
from .evaluate_classification import (
    PAPER_PRIMARY_THRESHOLD,
    PAPER_SENSITIVITY_THRESHOLD,
    PAPER_THRESHOLD_GRID_STEP,
    MaskTree,
    ProbabilityTree,
    ThresholdSearchResult,
    evaluate_mask_trees,
    evaluate_probability_trees,
    paper_threshold_grid,
    select_validation_f2_threshold,
)
from .evaluate_volume import (
    StemVolumeSummary,
    TreeStemVolume,
    evaluate_tree_stem_volume,
    summarize_stem_volumes,
)
from .evaluate_height import (
    HEIGHT_BIN_NAMES,
    MeanHeightBinRecall,
    StemHeightSummary,
    TreeHeightBinRecall,
    TreeStemHeight,
    evaluate_tree_height_bin_recall,
    evaluate_tree_stem_height,
    mean_height_bin_recall,
    relative_height_bin_ids,
    relative_tree_height,
    summarize_stem_heights,
)

__all__ = [
    "PAPER_F_BETA",
    "PAPER_THRESHOLD_GRID_STEP",
    "PAPER_PRIMARY_THRESHOLD",
    "PAPER_SENSITIVITY_THRESHOLD",
    "ConfusionCounts",
    "ClassificationMetrics",
    "PerTreeClassification",
    "AggregatedClassification",
    "branch_index_to_binary",
    "compute_confusion_counts",
    "metrics_from_counts",
    "evaluate_binary_classification",
    "aggregate_per_tree_classification",
    "ProbabilityTree",
    "MaskTree",
    "ThresholdSearchResult",
    "paper_threshold_grid",
    "evaluate_probability_trees",
    "evaluate_mask_trees",
    "select_validation_f2_threshold",
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
    "TreeStemVolume",
    "StemVolumeSummary",
    "evaluate_tree_stem_volume",
    "summarize_stem_volumes",
]
