"""Build per-tree preprocessing artifacts for occlusion-robust TLS stem detection.

Author: Shahab Alaedin Baloochi

This module orchestrates the three preprocessing stages used by the repository:

1. load metric XYZ coordinates,
2. build the fixed symmetric 3D Euclidean k-NN graph,
3. compute the 10D point representation (local XYZ + 7 engineered features),
4. optionally attach TreeQSM-derived binary stem labels,
5. validate compatibility with connected 8,192-point subgraph sampling.

The fixed graph is the single source of truth for point k-NN neighbourhoods:
``graph.knn_indices`` is passed directly to feature extraction, so the PCA
features and the network graph use the same precomputed Euclidean neighbours.

Label convention
----------------
The manuscript defines branch index 1 as stem, branch index >1 as non-stem,
and branch index 0 as unsegmented/excluded. Therefore ``unsegmented_id=0`` is
the default. Some converted CSV files may use a different sentinel (for
example -1); this must be supplied explicitly rather than guessed.

Unsegmented points are retained in the geometric arrays and fixed graph so that
original point IDs remain stable. Their training/evaluation label is -1 and
``label_mask`` is False. Downstream training/evaluation code must use this mask.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .feature_extraction import FEATURE_NAMES, FeatureExtractionResult, extract_all_features, load_xyz
    from .graph_construction import (
        DEFAULT_K,
        GraphConstructionResult,
        build_fixed_knn_graph,
        validate_graph,
    )
    from .subgraph_sampling import (
        DEFAULT_SUBGRAPH_SIZE,
        STRATUM_NAMES,
        ConnectedSubgraphSampler,
        InferencePartition,
        validate_inference_partition,
        validate_sampled_subgraph,
    )
except ImportError:  # allow direct execution from preprocessing/
    from feature_extraction import FEATURE_NAMES, FeatureExtractionResult, extract_all_features, load_xyz
    from graph_construction import (
        DEFAULT_K,
        GraphConstructionResult,
        build_fixed_knn_graph,
        validate_graph,
    )
    from subgraph_sampling import (
        DEFAULT_SUBGRAPH_SIZE,
        STRATUM_NAMES,
        ConnectedSubgraphSampler,
        InferencePartition,
        validate_inference_partition,
        validate_sampled_subgraph,
    )


LABEL_EXCLUDED = np.int8(-1)
LABEL_NON_STEM = np.int8(0)
LABEL_STEM = np.int8(1)
SCHEMA_VERSION = 1


@dataclass
class LabelData:
    """TreeQSM-derived labels aligned one-to-one with input points."""

    branch_ids: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    unsegmented_id: int

    @property
    def num_labelled(self) -> int:
        return int(self.label_mask.sum())

    @property
    def num_excluded(self) -> int:
        return int((~self.label_mask).sum())

    @property
    def num_stem(self) -> int:
        return int(np.sum(self.labels == LABEL_STEM))

    @property
    def num_non_stem(self) -> int:
        return int(np.sum(self.labels == LABEL_NON_STEM))


@dataclass
class BuiltTreeDataset:
    """Complete preprocessed representation of one individual TLS tree."""

    tree_id: str
    source_path: Path
    points: np.ndarray
    graph: GraphConstructionResult
    features: FeatureExtractionResult
    labels: Optional[LabelData]
    subgraph_size: int

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def node_features(self) -> np.ndarray:
        return self.features.node_features

    def make_sampler(self) -> ConnectedSubgraphSampler:
        return ConnectedSubgraphSampler(
            self.points,
            self.graph.edge_index,
            target_size=self.subgraph_size,
        )


def _read_csv_branch_ids(path: Path) -> Optional[np.ndarray]:
    """Read a headered CSV ``branch_id`` column without changing row order.

    Returns None for non-CSV inputs or CSV files without a branch_id column.
    """
    if path.suffix.lower() != ".csv":
        return None

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        name_lookup = {name.strip().lower(): name for name in reader.fieldnames if name is not None}
        key = name_lookup.get("branch_id")
        if key is None:
            return None

        values: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            raw = row.get(key)
            if raw is None or not raw.strip():
                raise ValueError(f"Missing branch_id at CSV row {row_number} in '{path}'.")
            try:
                numeric = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric branch_id '{raw}' at CSV row {row_number} in '{path}'."
                ) from exc
            integer = int(numeric)
            if numeric != integer:
                raise ValueError(
                    f"branch_id must be integer-valued; got {raw} at row {row_number}."
                )
            values.append(integer)

    return np.asarray(values, dtype=np.int64)


def make_binary_stem_labels(
    branch_ids: np.ndarray,
    *,
    unsegmented_id: int = 0,
) -> LabelData:
    """Convert TreeQSM branch indices to stem/non-stem labels.

    Stem: branch_id == 1
    Non-stem: branch_id > 1
    Excluded/unsegmented: branch_id == unsegmented_id

    Any other value is rejected. This prevents an undocumented sentinel such as
    -1 from being silently interpreted as a valid class.
    """
    branch_ids = np.asarray(branch_ids)
    if branch_ids.ndim != 1:
        raise ValueError("branch_ids must be one-dimensional.")
    if not np.issubdtype(branch_ids.dtype, np.integer):
        if not np.all(np.isfinite(branch_ids)) or not np.all(branch_ids == np.floor(branch_ids)):
            raise ValueError("branch_ids must contain finite integer values.")
        branch_ids = branch_ids.astype(np.int64)
    else:
        branch_ids = branch_ids.astype(np.int64, copy=False)

    stem = branch_ids == 1
    non_stem = branch_ids > 1
    excluded = branch_ids == int(unsegmented_id)
    valid = stem | non_stem | excluded
    if not np.all(valid):
        unexpected = np.unique(branch_ids[~valid])
        preview = ", ".join(str(int(v)) for v in unexpected[:10])
        suffix = "..." if unexpected.size > 10 else ""
        raise ValueError(
            "Unexpected branch_id value(s): " + preview + suffix + ". "
            f"The current unsegmented_id is {unsegmented_id}. "
            "Pass the correct sentinel explicitly if this file uses a converted convention."
        )

    labels = np.full(branch_ids.shape, LABEL_EXCLUDED, dtype=np.int8)
    labels[non_stem] = LABEL_NON_STEM
    labels[stem] = LABEL_STEM
    label_mask = ~excluded

    return LabelData(
        branch_ids=branch_ids,
        labels=labels,
        label_mask=label_mask,
        unsegmented_id=int(unsegmented_id),
    )


def build_tree_dataset(
    input_path: str | Path,
    *,
    k: int = DEFAULT_K,
    subgraph_size: int = DEFAULT_SUBGRAPH_SIZE,
    unsegmented_id: int = 0,
    require_labels: bool = False,
    query_batch_size: Optional[int] = 200_000,
    verbose: bool = False,
) -> BuiltTreeDataset:
    """Build all preprocessing products for one individual tree.

    No neighbourhood is recomputed inside feature extraction: the exact k-NN
    index matrix returned by ``build_fixed_knn_graph`` is passed to
    ``extract_all_features``.
    """
    path = Path(input_path)
    points = load_xyz(path)

    graph = build_fixed_knn_graph(
        points,
        k=k,
        query_batch_size=query_batch_size,
        verbose=False,
    )
    features = extract_all_features(
        points,
        k=k,
        point_knn_indices=graph.knn_indices,
        verbose=False,
    )

    if not np.array_equal(features.knn_indices, graph.knn_indices):
        raise RuntimeError("Feature extraction did not preserve the fixed graph k-NN indices.")

    branch_ids = _read_csv_branch_ids(path)
    label_data: Optional[LabelData]
    if branch_ids is None:
        if require_labels:
            raise ValueError(
                f"'{path}' does not provide a headered CSV branch_id column, but labels are required."
            )
        label_data = None
    else:
        if branch_ids.shape[0] != points.shape[0]:
            raise RuntimeError(
                f"XYZ/branch_id row mismatch: {points.shape[0]} XYZ rows vs "
                f"{branch_ids.shape[0]} branch IDs."
            )
        label_data = make_binary_stem_labels(
            branch_ids,
            unsegmented_id=unsegmented_id,
        )

    dataset = BuiltTreeDataset(
        tree_id=path.stem,
        source_path=path,
        points=points,
        graph=graph,
        features=features,
        labels=label_data,
        subgraph_size=int(subgraph_size),
    )

    if verbose:
        print_dataset_summary(dataset, k=k)
    return dataset


def validate_built_dataset(
    dataset: BuiltTreeDataset,
    *,
    k: int = DEFAULT_K,
    test_subgraphs: bool = True,
    brute_force_graph_samples: int = 16,
    random_seed: int = 0,
) -> dict[str, int | bool]:
    """Run end-to-end consistency checks across graph, features, labels and sampler."""
    n = dataset.num_points
    f = dataset.features
    g = dataset.graph

    checks: dict[str, int | bool] = {
        "num_points": n,
        "node_feature_shape_ok": bool(f.node_features.shape == (n, 10)),
        "engineered_shape_ok": bool(f.engineered.shape == (n, 7)),
        "local_xyz_shape_ok": bool(f.local_xyz.shape == (n, 3)),
        "knn_shape_ok": bool(g.knn_indices.shape == (n, k)),
        "feature_knn_equals_graph_knn": bool(np.array_equal(f.knn_indices, g.knn_indices)),
        "all_features_finite": bool(np.isfinite(f.node_features).all()),
        "engineered_in_unit_interval": bool(
            np.all((f.engineered >= 0.0) & (f.engineered <= 1.0))
        ),
        "graph_node_alignment_ok": bool(g.num_nodes == n),
    }

    graph_checks = validate_graph(
        g,
        points=dataset.points,
        k=k,
        brute_force_samples=brute_force_graph_samples,
        random_seed=random_seed,
    )
    checks["graph_validation_pass"] = bool(graph_checks["all_checks_pass"])
    checks["graph_num_components"] = int(g.num_components)

    if dataset.labels is not None:
        labels = dataset.labels
        checks.update(
            {
                "label_rows_match_points": bool(labels.labels.shape == (n,)),
                "label_mask_rows_match_points": bool(labels.label_mask.shape == (n,)),
                "branch_id_rows_match_points": bool(labels.branch_ids.shape == (n,)),
                "label_values_valid": bool(np.all(np.isin(labels.labels, [-1, 0, 1]))),
                "excluded_labels_match_mask": bool(
                    np.array_equal(labels.labels == LABEL_EXCLUDED, ~labels.label_mask)
                ),
                "num_labelled": labels.num_labelled,
                "num_excluded": labels.num_excluded,
                "num_stem": labels.num_stem,
                "num_non_stem": labels.num_non_stem,
            }
        )
        checks["label_accounting_ok"] = bool(
            labels.num_labelled + labels.num_excluded == n
            and labels.num_stem + labels.num_non_stem == labels.num_labelled
        )

    if test_subgraphs:
        sampler = dataset.make_sampler()
        training_ok = True
        rng = np.random.default_rng(random_seed)
        for stratum in STRATUM_NAMES:
            sample = sampler.sample_training(
                rng=rng,
                requested_stratum=stratum,
                require_full_size=True,
            )
            sample_checks = validate_sampled_subgraph(
                sampler,
                sample,
                require_full_size=True,
            )
            training_ok = training_ok and bool(sample_checks["all_checks_pass"])
        checks["training_subgraphs_pass"] = bool(training_ok)

        partition = sampler.partition_inference()
        partition_checks = validate_inference_partition(sampler, partition)
        checks["inference_partition_pass"] = bool(partition_checks["all_checks_pass"])
        checks["inference_num_subgraphs"] = int(partition.num_subgraphs)
        checks["inference_points_missing"] = int(partition_checks["points_missing"])
        checks["inference_points_repeated"] = int(partition_checks["points_repeated"])

        # Strong point-ID test: reconstruct the complete feature matrix through
        # inference subgraphs and require bit-for-bit equality.
        reconstructed = np.empty_like(f.node_features)
        for subgraph in partition.subgraphs:
            reconstructed[subgraph.point_ids] = f.node_features[subgraph.point_ids]
        checks["feature_reconstruction_exact"] = bool(
            np.array_equal(reconstructed, f.node_features)
        )

        if dataset.labels is not None:
            reconstructed_labels = np.empty_like(dataset.labels.labels)
            for subgraph in partition.subgraphs:
                reconstructed_labels[subgraph.point_ids] = dataset.labels.labels[subgraph.point_ids]
            checks["label_reconstruction_exact"] = bool(
                np.array_equal(reconstructed_labels, dataset.labels.labels)
            )

    ignored_keys = {
        "num_points",
        "graph_num_components",
        "num_labelled",
        "num_excluded",
        "num_stem",
        "num_non_stem",
        "inference_num_subgraphs",
        "inference_points_missing",
        "inference_points_repeated",
    }
    boolean_checks = [
        bool(value)
        for key, value in checks.items()
        if key not in ignored_keys and isinstance(value, (bool, np.bool_))
    ]
    checks["all_checks_pass"] = bool(all(boolean_checks))
    return checks


def save_built_dataset(dataset: BuiltTreeDataset, output_path: str | Path, *, k: int = DEFAULT_K) -> Path:
    """Save the reusable full-tree preprocessing products as a compressed NPZ."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "tree_id": np.asarray(dataset.tree_id),
        "xyz": dataset.points,
        "local_xyz": dataset.features.local_xyz,
        "engineered_features": dataset.features.engineered,
        "node_features": dataset.features.node_features,
        "feature_names": np.asarray(FEATURE_NAMES),
        "knn_indices": dataset.graph.knn_indices,
        "edge_index": dataset.graph.edge_index,
        "undirected_edges": dataset.graph.undirected_edges,
        "component_labels": dataset.graph.component_labels,
        "num_components": np.asarray(dataset.graph.num_components, dtype=np.int64),
        "k": np.asarray(k, dtype=np.int64),
        "subgraph_size": np.asarray(dataset.subgraph_size, dtype=np.int64),
        "stem_axis_point": dataset.features.stem_axis_point,
        "stem_axis_direction": dataset.features.stem_axis_direction,
        "root_voxel_index": np.asarray(dataset.features.root_voxel_index, dtype=np.int64),
        "root_voxel_center": dataset.features.root_voxel_center,
        "z_min": np.asarray(dataset.features.statistics.z_min, dtype=np.float64),
        "z_max": np.asarray(dataset.features.statistics.z_max, dtype=np.float64),
        "tree_height": np.asarray(dataset.features.statistics.height, dtype=np.float64),
        "x_center": np.asarray(dataset.features.statistics.x_center, dtype=np.float64),
        "y_center": np.asarray(dataset.features.statistics.y_center, dtype=np.float64),
    }
    if dataset.labels is not None:
        arrays.update(
            {
                "branch_ids": dataset.labels.branch_ids,
                "labels": dataset.labels.labels,
                "label_mask": dataset.labels.label_mask,
                "unsegmented_id": np.asarray(dataset.labels.unsegmented_id, dtype=np.int64),
            }
        )

    np.savez_compressed(output, **arrays)
    return output


def validate_saved_dataset(output_path: str | Path, expected: BuiltTreeDataset) -> dict[str, bool]:
    """Reload a saved NPZ and verify the principal arrays exactly."""
    path = Path(output_path)
    with np.load(path, allow_pickle=False) as saved:
        checks = {
            "xyz_exact": bool(np.array_equal(saved["xyz"], expected.points)),
            "node_features_exact": bool(
                np.array_equal(saved["node_features"], expected.features.node_features)
            ),
            "knn_indices_exact": bool(
                np.array_equal(saved["knn_indices"], expected.graph.knn_indices)
            ),
            "edge_index_exact": bool(
                np.array_equal(saved["edge_index"], expected.graph.edge_index)
            ),
        }
        if expected.labels is not None:
            checks["labels_exact"] = bool(
                np.array_equal(saved["labels"], expected.labels.labels)
            )
            checks["label_mask_exact"] = bool(
                np.array_equal(saved["label_mask"], expected.labels.label_mask)
            )
    checks["all_checks_pass"] = bool(all(checks.values()))
    return checks


def print_dataset_summary(dataset: BuiltTreeDataset, *, k: int = DEFAULT_K) -> None:
    print("\n[Built tree dataset]")
    print(f"Tree ID:                 {dataset.tree_id}")
    print(f"Points:                  {dataset.num_points:,}")
    print(f"Node features:           {dataset.features.node_features.shape}")
    print(f"Fixed graph k:           {k}")
    print(f"Graph components:        {dataset.graph.num_components:,}")
    print(f"Directed edge entries:   {dataset.graph.edge_index.shape[1]:,}")
    if dataset.labels is None:
        print("Labels:                  not available")
    else:
        labels = dataset.labels
        print(f"Labelled points:         {labels.num_labelled:,}")
        print(f"Excluded/unsegmented:    {labels.num_excluded:,}")
        print(f"Stem points:             {labels.num_stem:,}")
        print(f"Non-stem points:         {labels.num_non_stem:,}")
        print(f"Unsegmented ID:          {labels.unsegmented_id}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build one preprocessed TLS tree dataset.")
    parser.add_argument("input", type=Path, help="Input .csv or .txt tree point cloud.")
    parser.add_argument("--output", type=Path, default=None, help="Optional compressed .npz output.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Fixed Euclidean k-NN size.")
    parser.add_argument(
        "--subgraph-size",
        type=int,
        default=DEFAULT_SUBGRAPH_SIZE,
        help="Connected subgraph target size (default: 8192).",
    )
    parser.add_argument(
        "--unsegmented-id",
        type=int,
        default=0,
        help="Branch ID used for excluded/unsegmented points (paper default: 0).",
    )
    parser.add_argument("--require-labels", action="store_true", help="Fail if branch_id is unavailable.")
    parser.add_argument("--validate", action="store_true", help="Run full integration validation.")
    parser.add_argument("--seed", type=int, default=0, help="Validation random seed.")
    args = parser.parse_args()

    dataset = build_tree_dataset(
        args.input,
        k=args.k,
        subgraph_size=args.subgraph_size,
        unsegmented_id=args.unsegmented_id,
        require_labels=args.require_labels,
        verbose=True,
    )

    if args.validate:
        checks = validate_built_dataset(
            dataset,
            k=args.k,
            test_subgraphs=True,
            random_seed=args.seed,
        )
        print("\n[Integration validation]")
        for key, value in checks.items():
            print(f"{key:<38} {value}")
        if not checks["all_checks_pass"]:
            raise RuntimeError("Dataset integration validation failed.")
        print("Status:                                OK")

    if args.output is not None:
        saved = save_built_dataset(dataset, args.output, k=args.k)
        print(f"Saved:                    {saved}")
        if args.validate:
            save_checks = validate_saved_dataset(saved, dataset)
            for key, value in save_checks.items():
                print(f"save_{key:<33} {value}")
            if not save_checks["all_checks_pass"]:
                raise RuntimeError("Saved dataset round-trip validation failed.")


if __name__ == "__main__":
    _main()
