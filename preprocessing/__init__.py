"""Preprocessing utilities for occlusion-robust TLS stem detection.

Author: Shahab Alaedin Baloochi

This package exposes the repository's public preprocessing interface: point-cloud
loading and feature extraction, fixed 3D Euclidean k-NN graph construction,
connected subgraph sampling, and per-tree dataset assembly.
"""

from .feature_extraction import (
    FEATURE_NAMES,
    FeatureExtractionResult,
    TreeStatistics,
    extract_all_features,
    load_xyz,
)
from .graph_construction import (
    DEFAULT_K,
    GraphConstructionResult,
    build_fixed_knn_graph,
)
from .subgraph_sampling import (
    DEFAULT_SUBGRAPH_SIZE,
    ConnectedSubgraphSampler,
    InferencePartition,
    SampledSubgraph,
)
from .build_dataset import (
    BuiltTreeDataset,
    LabelData,
    build_tree_dataset,
    make_binary_stem_labels,
    save_built_dataset,
    validate_built_dataset,
)

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractionResult",
    "TreeStatistics",
    "extract_all_features",
    "load_xyz",
    "DEFAULT_K",
    "GraphConstructionResult",
    "build_fixed_knn_graph",
    "DEFAULT_SUBGRAPH_SIZE",
    "ConnectedSubgraphSampler",
    "InferencePartition",
    "SampledSubgraph",
    "BuiltTreeDataset",
    "LabelData",
    "build_tree_dataset",
    "make_binary_stem_labels",
    "save_built_dataset",
    "validate_built_dataset",
]
