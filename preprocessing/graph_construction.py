"""Fixed 3D Euclidean k-NN graph construction for TLS tree point clouds.

Author: Shahab Alaedin Baloochi

Implements Section 2.2.2 of the manuscript. Each TLS point is a graph node.
For every point, the k=16 nearest neighbours are selected in metric 3D XYZ
space. Because k-NN relations are not necessarily reciprocal, the union of
all selected pairs is symmetrised and represented as an undirected graph.
For PyTorch Geometric compatibility, ``edge_index`` stores both directions of
each undirected edge.

The graph is geometric and fixed: it is computed once from XYZ coordinates
and must not be recomputed in learned feature space between network layers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DEFAULT_K = 16


@dataclass
class GraphConstructionResult:
    """Outputs of fixed Euclidean k-NN graph construction."""

    knn_indices: np.ndarray
    edge_index: np.ndarray
    undirected_edges: np.ndarray
    component_labels: np.ndarray
    num_components: int

    @property
    def num_nodes(self) -> int:
        return int(self.knn_indices.shape[0])

    @property
    def num_directed_edges(self) -> int:
        """Number of directed entries stored in PyG-style edge_index."""
        return int(self.edge_index.shape[1])

    @property
    def num_undirected_edges(self) -> int:
        return int(self.undirected_edges.shape[0])


def _validate_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError("At least two points are required to construct a graph.")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite coordinates.")
    return points


def compute_knn_indices(
    points: np.ndarray,
    k: int = DEFAULT_K,
    query_batch_size: Optional[int] = 200_000,
) -> np.ndarray:
    """Return the k nearest *other* points for every point in 3D XYZ space.

    Parameters
    ----------
    points:
        Array of shape (N, 3), in metric XYZ coordinates.
    k:
        Number of Euclidean neighbours. The manuscript uses k=16.
    query_batch_size:
        Optional number of query points processed per cKDTree call. Batching
        reduces temporary memory for very large trees without changing the
        graph. Set to ``None`` to query all points at once.

    Returns
    -------
    np.ndarray
        Integer array with shape (N, k). Self-neighbours are excluded. The
        order follows increasing Euclidean distance as returned by cKDTree.

    Notes
    -----
    Exact coordinate duplicates may legitimately appear as zero-distance
    neighbours. Only the point's own row index is excluded.
    """
    points = _validate_xyz(points)
    n = points.shape[0]

    if k <= 0:
        raise ValueError("k must be positive.")
    if n <= k:
        raise ValueError(f"Need N > k, got N={n}, k={k}.")
    if query_batch_size is not None and query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive or None.")

    tree = cKDTree(points)
    knn = np.empty((n, k), dtype=np.int64)
    batch = n if query_batch_size is None else min(int(query_batch_size), n)

    for start in range(0, n, batch):
        stop = min(start + batch, n)
        _, candidates = tree.query(points[start:stop], k=k + 1)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim == 1:
            candidates = candidates[:, None]

        # Remove the query point itself by index rather than assuming it is the
        # first result. This remains correct when exact duplicate coordinates
        # produce several zero-distance neighbours.
        for local_row, global_row in enumerate(range(start, stop)):
            row = candidates[local_row]
            row_without_self = row[row != global_row]
            if row_without_self.size < k:
                # This is exceptionally unlikely with cKDTree, but a larger
                # query protects against pathological tie ordering among many
                # exact duplicate coordinates.
                extra_k = min(n, max(k + 2, 2 * k + 1))
                _, expanded = tree.query(points[global_row], k=extra_k)
                expanded = np.atleast_1d(expanded).astype(np.int64, copy=False)
                row_without_self = expanded[expanded != global_row]
            if row_without_self.size < k:
                raise RuntimeError(
                    f"Could not obtain {k} non-self neighbours for point {global_row}."
                )
            knn[global_row] = row_without_self[:k]

    return knn


def symmetrise_knn(knn_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrise directed k-NN selections.

    Returns
    -------
    edge_index:
        PyTorch-Geometric-style integer array of shape (2, E), containing both
        directions for every undirected edge.
    undirected_edges:
        Unique unordered node pairs with shape (E_undirected, 2), stored as
        ``(min(i, j), max(i, j))``.
    """
    knn_indices = np.asarray(knn_indices, dtype=np.int64)
    if knn_indices.ndim != 2:
        raise ValueError("knn_indices must have shape (N, k).")

    n, k = knn_indices.shape
    if n == 0 or k == 0:
        raise ValueError("knn_indices must be non-empty.")
    if np.any(knn_indices < 0) or np.any(knn_indices >= n):
        raise ValueError("knn_indices contains an out-of-range node index.")

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = knn_indices.reshape(-1)
    if np.any(src == dst):
        raise ValueError("knn_indices contains a self-neighbour.")

    # Canonical unordered representation of each selected pair. Encode each
    # unordered pair as one int64 value before np.unique; this uses much less
    # temporary memory than uniquing an (N*k, 2) array for million-point trees.
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    pair_codes = lo * np.int64(n) + hi
    unique_codes = np.unique(pair_codes)
    undirected_edges = np.column_stack(
        (unique_codes // np.int64(n), unique_codes % np.int64(n))
    ).astype(np.int64, copy=False)

    # PyG convention: an undirected graph is represented by both orientations.
    forward = undirected_edges
    reverse = undirected_edges[:, ::-1]
    directed = np.vstack((forward, reverse))

    # Deterministic lexicographic ordering of edge_index entries.
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]
    edge_index = directed.T.copy()

    return edge_index, undirected_edges


def _connected_components_from_edges(
    num_nodes: int,
    undirected_edges: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return connected-component count and per-node labels."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    undirected_edges = np.asarray(undirected_edges, dtype=np.int64)
    if undirected_edges.size == 0:
        return num_nodes, np.arange(num_nodes, dtype=np.int64)

    rows = np.concatenate((undirected_edges[:, 0], undirected_edges[:, 1]))
    cols = np.concatenate((undirected_edges[:, 1], undirected_edges[:, 0]))
    data = np.ones(rows.shape[0], dtype=np.uint8)
    adjacency = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    n_components, labels = connected_components(adjacency, directed=False)
    return int(n_components), labels.astype(np.int64, copy=False)


def build_fixed_knn_graph(
    points: np.ndarray,
    k: int = DEFAULT_K,
    query_batch_size: Optional[int] = 200_000,
    verbose: bool = False,
) -> GraphConstructionResult:
    """Construct the manuscript's fixed, symmetric 3D Euclidean k-NN graph."""
    points = _validate_xyz(points)
    knn_indices = compute_knn_indices(points, k=k, query_batch_size=query_batch_size)
    edge_index, undirected_edges = symmetrise_knn(knn_indices)
    n_components, component_labels = _connected_components_from_edges(
        points.shape[0], undirected_edges
    )

    result = GraphConstructionResult(
        knn_indices=knn_indices,
        edge_index=edge_index,
        undirected_edges=undirected_edges,
        component_labels=component_labels,
        num_components=n_components,
    )

    if verbose:
        print_graph_summary(result, k=k)
    return result


def validate_graph(
    result: GraphConstructionResult,
    points: Optional[np.ndarray] = None,
    k: int = DEFAULT_K,
    brute_force_samples: int = 32,
    random_seed: int = 0,
) -> dict[str, int | bool]:
    """Run structural checks and optional brute-force nearest-neighbour checks.

    The brute-force test compares the *distances* of selected neighbours with
    the true k smallest non-self Euclidean distances. Comparing distances rather
    than only indices correctly handles exact duplicate coordinates and distance
    ties, for which several neighbour-index sets can be equally valid.
    """
    n = result.num_nodes
    edge_index = np.asarray(result.edge_index, dtype=np.int64)
    knn = np.asarray(result.knn_indices, dtype=np.int64)

    checks: dict[str, int | bool] = {}
    checks["knn_shape_ok"] = knn.shape == (n, k)
    checks["edge_index_shape_ok"] = edge_index.ndim == 2 and edge_index.shape[0] == 2
    checks["self_neighbours"] = int(np.sum(knn == np.arange(n)[:, None]))
    checks["invalid_knn_indices"] = int(np.sum((knn < 0) | (knn >= n)))
    checks["self_edges"] = int(np.sum(edge_index[0] == edge_index[1]))
    checks["invalid_edge_indices"] = int(
        np.sum((edge_index < 0) | (edge_index >= n))
    )

    codes = edge_index[0].astype(np.int64) * np.int64(n) + edge_index[1]
    unique_codes = np.unique(codes)
    checks["duplicate_directed_edges"] = int(codes.size - unique_codes.size)

    reverse_codes = edge_index[1].astype(np.int64) * np.int64(n) + edge_index[0]
    checks["missing_reverse_edges"] = int(
        np.sum(~np.isin(reverse_codes, unique_codes, assume_unique=False))
    )
    checks["num_components"] = int(result.num_components)

    if points is not None and brute_force_samples > 0:
        points = _validate_xyz(points)
        if points.shape[0] != n:
            raise ValueError("points and graph have different numbers of nodes.")
        rng = np.random.default_rng(random_seed)
        sample_count = min(int(brute_force_samples), n)
        sample_ids = rng.choice(n, size=sample_count, replace=False)
        brute_force_ok = True

        for i in sample_ids:
            diff = points - points[i]
            all_dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            all_dist[i] = np.inf
            true_k = np.partition(all_dist, k - 1)[:k]
            selected = all_dist[knn[i]]
            if not np.allclose(
                np.sort(selected),
                np.sort(true_k),
                rtol=1e-10,
                atol=1e-12,
            ):
                brute_force_ok = False
                break
        checks["brute_force_neighbour_distances_ok"] = brute_force_ok

    structural_ok = (
        bool(checks["knn_shape_ok"])
        and bool(checks["edge_index_shape_ok"])
        and checks["self_neighbours"] == 0
        and checks["invalid_knn_indices"] == 0
        and checks["self_edges"] == 0
        and checks["invalid_edge_indices"] == 0
        and checks["duplicate_directed_edges"] == 0
        and checks["missing_reverse_edges"] == 0
    )
    if "brute_force_neighbour_distances_ok" in checks:
        structural_ok = structural_ok and bool(checks["brute_force_neighbour_distances_ok"])
    checks["all_checks_pass"] = structural_ok
    return checks


def print_graph_summary(result: GraphConstructionResult, k: int = DEFAULT_K) -> None:
    counts = np.bincount(result.component_labels, minlength=result.num_components)
    largest = int(counts.max()) if counts.size else 0
    print("\n[Graph construction]")
    print(f"Points:                     {result.num_nodes:,}")
    print(f"k:                          {k}")
    print(f"Raw k-NN selections:        {result.num_nodes * k:,}")
    print(f"Unique undirected edges:    {result.num_undirected_edges:,}")
    print(f"PyG directed edge entries:  {result.num_directed_edges:,}")
    print(f"edge_index shape:           {result.edge_index.shape}")
    print(f"Connected components:       {result.num_components:,}")
    print(f"Largest component:          {largest:,}")


def _load_xyz_for_cli(path: Path) -> np.ndarray:
    """Reuse the repository's common CSV/TXT XYZ loader when available."""
    try:
        from .feature_extraction import load_xyz  # type: ignore
    except ImportError:
        from feature_extraction import load_xyz  # type: ignore
    return load_xyz(path)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct a fixed symmetric 3D Euclidean k-NN graph."
    )
    parser.add_argument("input", type=Path, help="Input .csv or .txt point cloud.")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz output.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k-NN size (default: 16).")
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=200_000,
        help="cKDTree query batch size (default: 200000).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run structural and sampled brute-force graph checks.",
    )
    args = parser.parse_args()

    points = _load_xyz_for_cli(args.input)
    result = build_fixed_knn_graph(
        points,
        k=args.k,
        query_batch_size=args.query_batch_size,
        verbose=True,
    )

    if args.validate:
        checks = validate_graph(result, points=points, k=args.k)
        print("\n[Validation]")
        for name, value in checks.items():
            print(f"{name:<38} {value}")
        if not checks["all_checks_pass"]:
            raise RuntimeError("Graph validation failed.")
        print("Status:                                OK")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            knn_indices=result.knn_indices,
            edge_index=result.edge_index,
            undirected_edges=result.undirected_edges,
            component_labels=result.component_labels,
            num_components=np.asarray(result.num_components, dtype=np.int64),
            k=np.asarray(args.k, dtype=np.int64),
        )
        print(f"Saved:                       {args.output}")


if __name__ == "__main__":
    _main()
