"""Connected subgraph sampling for TLS stem detection.

Author: Shahab Alaedin Baloochi

Training follows Section 2.4 of the manuscript: a height-stratified seed is
expanded by BFS on the precomputed fixed Euclidean k-NN graph to obtain an
8,192-point connected subgraph. Inference uses the same fixed graph and covers
every point exactly once with connected, non-overlapping blocks. The manuscript
does not specify the exact inference partition rule, so a deterministic
connectivity-preserving tree partition is used here to avoid tiny residual
fragments. No graph edge is added or recomputed.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, depth_first_order

DEFAULT_SUBGRAPH_SIZE = 8192
DEFAULT_STRATUM_EDGES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
STRATUM_NAMES = ("lower", "middle", "crown")


@dataclass
class SampledSubgraph:
    point_ids: np.ndarray
    edge_index: np.ndarray
    seed_id: int
    stratum: Optional[str] = None

    @property
    def num_nodes(self) -> int:
        return int(self.point_ids.size)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class InferencePartition:
    subgraphs: list[SampledSubgraph]
    num_points: int
    target_size: int

    @property
    def num_subgraphs(self) -> int:
        return len(self.subgraphs)

    @property
    def sizes(self) -> np.ndarray:
        return np.asarray([s.num_nodes for s in self.subgraphs], dtype=np.int64)


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"points must have shape (N, 3) with N>0, got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("points contains NaN or infinite coordinates.")
    return points


def _validate_edge_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}.")
    if edge_index.size and (
        np.any(edge_index < 0)
        or np.any(edge_index >= num_nodes)
        or np.any(edge_index[0] == edge_index[1])
    ):
        raise ValueError("edge_index contains an invalid node index or self-edge.")
    return edge_index


def build_csr_adjacency(edge_index: np.ndarray, num_nodes: int) -> csr_matrix:
    edge_index = _validate_edge_index(edge_index, num_nodes)
    if edge_index.shape[1] == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=np.uint8)
    adjacency = csr_matrix(
        (np.ones(edge_index.shape[1], dtype=np.uint8), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
    )
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def compute_relative_height(points: np.ndarray) -> np.ndarray:
    points = _validate_points(points)
    z = points[:, 2]
    h = float(z.max() - z.min())
    return np.zeros(len(points)) if h <= 1e-12 else (z - z.min()) / (h + 1e-12)


def assign_height_strata(
    relative_height: np.ndarray,
    stratum_edges: Sequence[float] = DEFAULT_STRATUM_EDGES,
) -> np.ndarray:
    relative_height = np.asarray(relative_height, dtype=np.float64)
    edges = np.asarray(stratum_edges, dtype=np.float64)
    if relative_height.ndim != 1 or not np.isfinite(relative_height).all():
        raise ValueError("relative_height must be a finite 1D array.")
    if edges.shape != (4,) or not np.all(np.diff(edges) > 0):
        raise ValueError("stratum_edges must contain four strictly increasing values.")
    if edges[0] > 0.0 or edges[-1] < 1.0:
        raise ValueError("stratum_edges must cover [0,1].")
    return np.clip(np.digitize(relative_height, edges[1:-1]), 0, 2).astype(np.int8)


def _bfs_collect(
    adjacency: csr_matrix,
    seed_id: int,
    max_nodes: int,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = adjacency.shape[0]
    if not (0 <= seed_id < n) or max_nodes <= 0:
        raise ValueError("Invalid BFS seed or max_nodes.")
    allowed = np.ones(n, dtype=bool) if allowed_mask is None else np.asarray(allowed_mask, dtype=bool)
    if allowed.shape != (n,) or not allowed[seed_id]:
        raise ValueError("Invalid allowed_mask for BFS.")

    seen = np.zeros(n, dtype=bool)
    seen[seed_id] = True
    queue: deque[int] = deque([int(seed_id)])
    selected: list[int] = []
    while queue and len(selected) < max_nodes:
        node = queue.popleft()
        selected.append(node)
        for nb in adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]:
            nb = int(nb)
            if allowed[nb] and not seen[nb]:
                seen[nb] = True
                queue.append(nb)
    return np.asarray(selected, dtype=np.int64)


def induce_local_edge_index(adjacency: csr_matrix, point_ids: np.ndarray) -> np.ndarray:
    point_ids = np.asarray(point_ids, dtype=np.int64)
    n = adjacency.shape[0]
    if point_ids.ndim != 1 or point_ids.size == 0:
        raise ValueError("point_ids must be a non-empty 1D array.")
    if np.any(point_ids < 0) or np.any(point_ids >= n) or np.unique(point_ids).size != point_ids.size:
        raise ValueError("point_ids contains invalid or duplicate indices.")
    local = adjacency[point_ids][:, point_ids].tocsr()
    local.sort_indices()
    rows, cols = local.nonzero()
    return np.vstack((rows, cols)).astype(np.int64, copy=False)


class ConnectedSubgraphSampler:
    def __init__(
        self,
        points: np.ndarray,
        global_edge_index: np.ndarray,
        target_size: int = DEFAULT_SUBGRAPH_SIZE,
        stratum_edges: Sequence[float] = DEFAULT_STRATUM_EDGES,
    ) -> None:
        self.points = _validate_points(points)
        self.num_nodes = len(self.points)
        if target_size <= 0:
            raise ValueError("target_size must be positive.")
        self.target_size = int(target_size)
        self.global_edge_index = _validate_edge_index(global_edge_index, self.num_nodes)
        self.adjacency = build_csr_adjacency(self.global_edge_index, self.num_nodes)
        if (self.adjacency != self.adjacency.T).nnz:
            raise ValueError("global_edge_index must be the symmetrised fixed graph.")

        n_components, labels = connected_components(self.adjacency, directed=False)
        self.num_components = int(n_components)
        self.component_labels = labels.astype(np.int64, copy=False)
        self.component_sizes = np.bincount(labels, minlength=n_components).astype(np.int64)
        self.stratum_edges = tuple(float(x) for x in stratum_edges)
        self.relative_height = compute_relative_height(self.points)
        self.height_strata = assign_height_strata(self.relative_height, self.stratum_edges)

    def sample_training(
        self,
        rng: Optional[np.random.Generator] = None,
        requested_stratum: Optional[int | str] = None,
        require_full_size: bool = True,
    ) -> SampledSubgraph:
        rng = np.random.default_rng() if rng is None else rng
        if requested_stratum is None:
            sid = int(rng.integers(0, 3))
        elif isinstance(requested_stratum, str):
            if requested_stratum not in STRATUM_NAMES:
                raise ValueError(f"Unknown stratum: {requested_stratum}")
            sid = STRATUM_NAMES.index(requested_stratum)
        else:
            sid = int(requested_stratum)
            if sid not in (0, 1, 2):
                raise ValueError("requested_stratum must be 0, 1, 2, or a stratum name.")

        viable = (
            self.component_sizes[self.component_labels] >= self.target_size
            if require_full_size
            else np.ones(self.num_nodes, dtype=bool)
        )
        candidates = np.flatnonzero((self.height_strata == sid) & viable)
        if candidates.size == 0:
            raise RuntimeError(f"No viable seed in stratum '{STRATUM_NAMES[sid]}'.")
        seed = int(rng.choice(candidates))
        ids = _bfs_collect(self.adjacency, seed, self.target_size)
        if require_full_size and ids.size != self.target_size:
            raise RuntimeError("BFS did not reach the requested training size.")
        return SampledSubgraph(ids, induce_local_edge_index(self.adjacency, ids), seed, STRATUM_NAMES[sid])

    def _select_inference_root(self, component_nodes: np.ndarray) -> int:
        xyz = self.points[component_nodes]
        centre = np.median(xyz, axis=0)
        d2 = np.einsum("ij,ij->i", xyz - centre, xyz - centre)
        candidates = component_nodes[np.flatnonzero(d2 == d2.min())]
        return int(candidates.min())

    def _partition_component(self, component_nodes: np.ndarray) -> list[SampledSubgraph]:
        component_nodes = np.sort(np.asarray(component_nodes, dtype=np.int64))
        n = component_nodes.size
        local_adj = self.adjacency[component_nodes][:, component_nodes].tocsr()
        local_adj.sort_indices()
        root_global = self._select_inference_root(component_nodes)
        root = int(np.searchsorted(component_nodes, root_global))

        if n <= self.target_size:
            local_ids = _bfs_collect(local_adj, root, n)
            global_ids = component_nodes[local_ids]
            return [SampledSubgraph(global_ids, induce_local_edge_index(self.adjacency, global_ids), root_global)]

        order, pred = depth_first_order(local_adj, i_start=root, directed=False, return_predecessors=True)
        order = np.asarray(order, dtype=np.int64)
        parent = np.asarray(pred, dtype=np.int64)
        if order.size != n:
            raise RuntimeError("Spanning tree did not cover the component.")
        parent[root] = -1

        child_ids = np.flatnonzero(parent >= 0).astype(np.int64)
        tree_children = csr_matrix(
            (np.ones(child_ids.size, dtype=np.uint8), (parent[child_ids], child_ids)),
            shape=(n, n),
        )
        tree_children.sort_indices()
        subtree_size = np.ones(n, dtype=np.int64)
        for node in order[::-1]:
            if parent[node] >= 0:
                subtree_size[parent[node]] += subtree_size[node]

        active = np.ones(n, dtype=bool)
        remaining = int(n)
        chunks: list[tuple[np.ndarray, int]] = []
        while remaining > self.target_size:
            candidates = np.flatnonzero(active & (subtree_size > 0) & (subtree_size <= self.target_size))
            if candidates.size == 0:
                raise RuntimeError("No valid connectivity-preserving cut found.")
            sizes = subtree_size[candidates]
            best = candidates[sizes == sizes.max()]
            cut_root = int(best[np.argmin(component_nodes[best])])

            stack = [cut_root]
            selected: list[int] = []
            while stack:
                node = int(stack.pop())
                if not active[node]:
                    continue
                selected.append(node)
                a, b = tree_children.indptr[node], tree_children.indptr[node + 1]
                stack.extend(int(c) for c in tree_children.indices[a:b] if active[c])
            selected_arr = np.asarray(selected, dtype=np.int64)
            if selected_arr.size != int(subtree_size[cut_root]):
                raise RuntimeError("Subtree-size bookkeeping mismatch.")

            removed = int(selected_arr.size)
            active[selected_arr] = False
            subtree_size[selected_arr] = 0
            remaining -= removed
            chunks.append((selected_arr, cut_root))
            ancestor = int(parent[cut_root])
            while ancestor >= 0:
                subtree_size[ancestor] -= removed
                ancestor = int(parent[ancestor])

        residual = np.flatnonzero(active).astype(np.int64)
        if residual.size == 0 or residual.size > self.target_size:
            raise RuntimeError("Invalid final inference residual.")
        chunks.append((residual, root))

        output: list[SampledSubgraph] = []
        for selected, seed_local in chunks:
            allowed = np.zeros(n, dtype=bool)
            allowed[selected] = True
            bfs_ids = _bfs_collect(local_adj, seed_local, selected.size, allowed)
            if bfs_ids.size != selected.size or not np.array_equal(np.sort(bfs_ids), np.sort(selected)):
                raise RuntimeError("Inference block is not connected.")
            global_ids = component_nodes[bfs_ids]
            output.append(
                SampledSubgraph(
                    global_ids,
                    induce_local_edge_index(self.adjacency, global_ids),
                    int(component_nodes[seed_local]),
                )
            )
        return output

    def partition_inference(self) -> InferencePartition:
        subgraphs: list[SampledSubgraph] = []
        for cid in range(self.num_components):
            nodes = np.flatnonzero(self.component_labels == cid).astype(np.int64)
            if nodes.size:
                subgraphs.extend(self._partition_component(nodes))
        subgraphs.sort(key=lambda s: int(s.point_ids.min()))
        return InferencePartition(subgraphs, self.num_nodes, self.target_size)


def _is_connected_local(subgraph: SampledSubgraph) -> bool:
    if subgraph.num_nodes <= 1:
        return True
    if subgraph.edge_index.shape[1] == 0:
        return False
    adj = csr_matrix(
        (np.ones(subgraph.edge_index.shape[1], dtype=np.uint8), (subgraph.edge_index[0], subgraph.edge_index[1])),
        shape=(subgraph.num_nodes, subgraph.num_nodes),
    )
    n_components, _ = connected_components(adj, directed=False)
    return int(n_components) == 1


def validate_sampled_subgraph(
    sampler: ConnectedSubgraphSampler,
    subgraph: SampledSubgraph,
    require_full_size: bool = False,
) -> dict[str, int | bool]:
    ids = np.asarray(subgraph.point_ids, dtype=np.int64)
    local = np.asarray(subgraph.edge_index, dtype=np.int64)
    expected = induce_local_edge_index(sampler.adjacency, ids)
    checks: dict[str, int | bool] = {
        "nonempty": bool(ids.size),
        "size_within_limit": bool(ids.size <= sampler.target_size),
        "full_size": bool(ids.size == sampler.target_size),
        "unique_point_ids": bool(np.unique(ids).size == ids.size),
        "valid_global_ids": bool(np.all((ids >= 0) & (ids < sampler.num_nodes))),
        "valid_local_edges": bool(local.size == 0 or np.all((local >= 0) & (local < ids.size))),
        "self_edges": int(np.sum(local[0] == local[1])) if local.size else 0,
        "connected": bool(_is_connected_local(subgraph)),
        "induced_edge_mapping_exact": bool(np.array_equal(local, expected)),
        "seed_inside_subgraph": bool(np.any(ids == subgraph.seed_id)),
    }
    ok = all(
        bool(checks[k])
        for k in (
            "nonempty", "size_within_limit", "unique_point_ids", "valid_global_ids",
            "valid_local_edges", "connected", "induced_edge_mapping_exact", "seed_inside_subgraph",
        )
    ) and checks["self_edges"] == 0
    if require_full_size:
        ok = bool(ok and checks["full_size"])
    checks["all_checks_pass"] = bool(ok)
    return checks


def validate_inference_partition(
    sampler: ConnectedSubgraphSampler,
    partition: InferencePartition,
) -> dict[str, int | bool]:
    sizes = partition.sizes
    all_ids = np.concatenate([s.point_ids for s in partition.subgraphs]) if partition.subgraphs else np.empty(0, dtype=np.int64)
    counts = np.bincount(all_ids, minlength=sampler.num_nodes) if all_ids.size else np.zeros(sampler.num_nodes, dtype=np.int64)
    checks: dict[str, int | bool] = {
        "num_subgraphs": partition.num_subgraphs,
        "total_selected_points": int(all_ids.size),
        "max_subgraph_size": int(sizes.max()) if sizes.size else 0,
        "min_subgraph_size": int(sizes.min()) if sizes.size else 0,
        "points_missing": int(np.sum(counts == 0)),
        "points_repeated": int(np.sum(counts > 1)),
        "exact_once_coverage": bool(np.all(counts == 1)),
        "all_sizes_within_limit": bool(np.all(sizes <= sampler.target_size)),
        "all_subgraphs_connected": bool(all(_is_connected_local(s) for s in partition.subgraphs)),
        "all_edge_mappings_exact": bool(all(
            np.array_equal(s.edge_index, induce_local_edge_index(sampler.adjacency, s.point_ids))
            for s in partition.subgraphs
        )),
        "all_seeds_inside_subgraphs": bool(all(np.any(s.point_ids == s.seed_id) for s in partition.subgraphs)),
    }
    checks["all_checks_pass"] = bool(
        checks["exact_once_coverage"]
        and checks["all_sizes_within_limit"]
        and checks["all_subgraphs_connected"]
        and checks["all_edge_mappings_exact"]
        and checks["all_seeds_inside_subgraphs"]
    )
    return checks


def print_partition_summary(partition: InferencePartition) -> None:
    sizes = partition.sizes
    print("\n[Inference subgraph partition]")
    print(f"Tree points:               {partition.num_points:,}")
    print(f"Target subgraph size:      {partition.target_size:,}")
    print(f"Number of subgraphs:       {partition.num_subgraphs:,}")
    if sizes.size:
        print(f"Largest subgraph:          {sizes.max():,}")
        print(f"Smallest subgraph:         {sizes.min():,}")
        print(f"Full-size subgraphs:       {np.sum(sizes == partition.target_size):,}")
        print(f"Total covered points:      {sizes.sum():,}")


def _load_repository_modules():
    try:
        from .feature_extraction import load_xyz
        from .graph_construction import build_fixed_knn_graph
    except ImportError:
        from feature_extraction import load_xyz
        from graph_construction import build_fixed_knn_graph
    return load_xyz, build_fixed_knn_graph


def _main() -> None:
    parser = argparse.ArgumentParser(description="Sample connected subgraphs from a fixed TLS k-NN graph.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("inference", "training"), default="inference")
    parser.add_argument("--size", type=int, default=DEFAULT_SUBGRAPH_SIZE)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stratum", choices=STRATUM_NAMES, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    load_xyz, build_fixed_knn_graph = _load_repository_modules()
    points = load_xyz(args.input)
    graph = build_fixed_knn_graph(points, k=args.k)
    sampler = ConnectedSubgraphSampler(points, graph.edge_index, target_size=args.size)

    if args.mode == "training":
        sample = sampler.sample_training(np.random.default_rng(args.seed), args.stratum, True)
        print(f"Training subgraph: {sample.num_nodes:,} points, stratum={sample.stratum}, seed={sample.seed_id}")
        if args.validate:
            checks = validate_sampled_subgraph(sampler, sample, True)
            print(checks)
            if not checks["all_checks_pass"]:
                raise RuntimeError("Training subgraph validation failed.")
    else:
        partition = sampler.partition_inference()
        print_partition_summary(partition)
        if args.validate:
            checks = validate_inference_partition(sampler, partition)
            print(checks)
            if not checks["all_checks_pass"]:
                raise RuntimeError("Inference partition validation failed.")


if __name__ == "__main__":
    _main()
