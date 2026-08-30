"""Connected subgraph sampling for TLS stem detection.

Author: Shahab Alaedin Baloochi

Implements the connected subgraph processing described in Section 2.4 of the
manuscript. Subgraphs are expanded with breadth-first search (BFS) on the
precomputed fixed Euclidean k-NN graph; the graph is never recomputed in
feature space.

Two modes are provided:

1. Training sampling
   - sample a seed from a height stratum (lower / middle / crown),
   - expand by BFS on the fixed graph,
   - return a connected subgraph of 8,192 points whenever the seed's connected
     component contains at least 8,192 points.

2. Exhaustive inference partitioning
   - partition each fixed-graph connected component through a deterministic
     breadth-first spanning tree,
   - cut connectivity-preserving tree subgraphs with at most 8,192 points,
   - subgraphs do not overlap and every tree point is covered exactly once,
   - graph edges crossing subgraph boundaries are omitted locally, as stated
     in the manuscript.

Reproducibility notes
---------------------
The manuscript specifies height-stratified training seeds but does not provide
numerical cut-points for lower/middle/crown. This implementation therefore
makes the cut-points explicit and configurable through ``stratum_edges``. The
default is three equal relative-height intervals (0, 1/3, 2/3, 1), consistent
with the paper's separate evaluation protocol that reports lower/middle/upper
recall in three equal relative-height bins. If the original experiment used
different training cut-points, pass them explicitly.

The manuscript also states that inference proceeds sequentially through
non-overlapping connected subgraphs but does not prescribe an exact inference
seed/partition rule. A naive repeated BFS carve can fragment the unassigned
remainder into many tiny pieces. To avoid that artefact while preserving every
explicit manuscript constraint, inference here first builds a breadth-first
spanning tree inside each connected component and removes the largest active
spanning-tree subtree that fits within the 8,192-point budget. Removing a tree
subtree leaves the remainder connected in the spanning tree, so artificial
one-point residuals are avoided. The root is chosen deterministically as the
highest-Z point of each original connected component (ties: smallest point ID).
No graph edge is added or recomputed; only the batching partition is defined.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

DEFAULT_SUBGRAPH_SIZE = 8192
DEFAULT_STRATUM_EDGES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
STRATUM_NAMES = ("lower", "middle", "crown")


@dataclass
class SampledSubgraph:
    """A connected subgraph with original global point IDs and local edges."""

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
    """Non-overlapping exhaustive inference subgraphs for one complete tree."""

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
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("points must be non-empty.")
    if not np.isfinite(points).all():
        raise ValueError("points contains NaN or infinite coordinates.")
    return points


def _validate_edge_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}.")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if edge_index.size and (np.any(edge_index < 0) or np.any(edge_index >= num_nodes)):
        raise ValueError("edge_index contains an out-of-range node index.")
    if edge_index.size and np.any(edge_index[0] == edge_index[1]):
        raise ValueError("edge_index contains a self-edge.")
    return edge_index


def build_csr_adjacency(edge_index: np.ndarray, num_nodes: int) -> csr_matrix:
    """Build a binary CSR adjacency from the precomputed fixed graph."""
    edge_index = _validate_edge_index(edge_index, num_nodes)
    if edge_index.shape[1] == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=np.uint8)

    adjacency = csr_matrix(
        (
            np.ones(edge_index.shape[1], dtype=np.uint8),
            (edge_index[0], edge_index[1]),
        ),
        shape=(num_nodes, num_nodes),
        dtype=np.uint8,
    )
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def compute_relative_height(points: np.ndarray) -> np.ndarray:
    """Return per-point relative tree height in [0, 1]."""
    points = _validate_points(points)
    z = points[:, 2]
    z_min = float(z.min())
    height = float(z.max() - z_min)
    if height <= 1e-12:
        return np.zeros(points.shape[0], dtype=np.float64)
    return (z - z_min) / (height + 1e-12)


def assign_height_strata(
    relative_height: np.ndarray,
    stratum_edges: Sequence[float] = DEFAULT_STRATUM_EDGES,
) -> np.ndarray:
    """Assign lower/middle/crown integer labels 0, 1, 2."""
    relative_height = np.asarray(relative_height, dtype=np.float64)
    if relative_height.ndim != 1:
        raise ValueError("relative_height must be one-dimensional.")
    if not np.isfinite(relative_height).all():
        raise ValueError("relative_height contains NaN or infinite values.")

    edges = np.asarray(stratum_edges, dtype=np.float64)
    if edges.shape != (4,):
        raise ValueError("stratum_edges must contain exactly four boundaries.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("stratum_edges must be strictly increasing.")
    if edges[0] > 0.0 or edges[-1] < 1.0:
        raise ValueError("stratum_edges must cover the full [0,1] interval.")

    labels = np.digitize(relative_height, edges[1:-1], right=False)
    return np.clip(labels, 0, 2).astype(np.int8, copy=False)


def _bfs_collect(
    adjacency: csr_matrix,
    seed_id: int,
    max_nodes: int,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Collect up to ``max_nodes`` by BFS while preserving connectivity."""
    n = adjacency.shape[0]
    if not (0 <= seed_id < n):
        raise ValueError(f"seed_id={seed_id} is outside [0,{n}).")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")

    if allowed_mask is None:
        allowed = np.ones(n, dtype=bool)
    else:
        allowed = np.asarray(allowed_mask, dtype=bool)
        if allowed.shape != (n,):
            raise ValueError("allowed_mask must have shape (N,).")
    if not allowed[seed_id]:
        raise ValueError("The BFS seed is not available in allowed_mask.")

    seen = np.zeros(n, dtype=bool)
    seen[seed_id] = True
    queue: deque[int] = deque([int(seed_id)])
    selected: list[int] = []
    indptr = adjacency.indptr
    indices = adjacency.indices

    while queue and len(selected) < max_nodes:
        node = queue.popleft()
        selected.append(node)
        if len(selected) == max_nodes:
            break

        start, stop = indptr[node], indptr[node + 1]
        for neighbour in indices[start:stop]:
            neighbour = int(neighbour)
            if allowed[neighbour] and not seen[neighbour]:
                seen[neighbour] = True
                queue.append(neighbour)

    return np.asarray(selected, dtype=np.int64)


def induce_local_edge_index(adjacency: csr_matrix, point_ids: np.ndarray) -> np.ndarray:
    """Extract the fixed global edges internal to one subgraph and remap locally.

    This is an induced subgraph of the precomputed global graph. Neighbourhoods
    are not recomputed; boundary-crossing edges are omitted.
    """
    point_ids = np.asarray(point_ids, dtype=np.int64)
    n = adjacency.shape[0]
    if point_ids.ndim != 1 or point_ids.size == 0:
        raise ValueError("point_ids must be a non-empty one-dimensional array.")
    if np.any(point_ids < 0) or np.any(point_ids >= n):
        raise ValueError("point_ids contains an out-of-range global index.")
    if np.unique(point_ids).size != point_ids.size:
        raise ValueError("point_ids contains duplicate nodes.")

    # Sparse slicing preserves only edges whose endpoints are both in point_ids
    # and automatically maps them to local row/column indices 0..m-1.
    local_csr = adjacency[point_ids][:, point_ids].tocsr()
    local_csr.sort_indices()
    rows, cols = local_csr.nonzero()
    return np.vstack((rows, cols)).astype(np.int64, copy=False)


class ConnectedSubgraphSampler:
    """Reusable sampler that preprocesses graph metadata once per tree.

    The adjacency, connected components, relative heights and height strata are
    computed once and reused across many training samples. This avoids rebuilding
    the full graph structures for every mini-batch.
    """

    def __init__(
        self,
        points: np.ndarray,
        global_edge_index: np.ndarray,
        target_size: int = DEFAULT_SUBGRAPH_SIZE,
        stratum_edges: Sequence[float] = DEFAULT_STRATUM_EDGES,
    ) -> None:
        self.points = _validate_points(points)
        self.num_nodes = int(self.points.shape[0])
        self.global_edge_index = _validate_edge_index(global_edge_index, self.num_nodes)
        if target_size <= 0:
            raise ValueError("target_size must be positive.")
        self.target_size = int(target_size)
        self.stratum_edges = tuple(float(x) for x in stratum_edges)

        self.adjacency = build_csr_adjacency(self.global_edge_index, self.num_nodes)
        # Section 2.2.2 treats the symmetrised k-NN graph as undirected.
        # Refuse an asymmetric input rather than silently changing graph semantics.
        if (self.adjacency != self.adjacency.T).nnz != 0:
            raise ValueError(
                "global_edge_index is not symmetric; pass the symmetrised fixed k-NN graph."
            )

        n_components, component_labels = connected_components(
            self.adjacency, directed=False
        )
        self.num_components = int(n_components)
        self.component_labels = component_labels.astype(np.int64, copy=False)
        self.component_sizes = np.bincount(
            self.component_labels, minlength=self.num_components
        ).astype(np.int64, copy=False)

        self.relative_height = compute_relative_height(self.points)
        self.height_strata = assign_height_strata(
            self.relative_height, stratum_edges=self.stratum_edges
        )

    def sample_training(
        self,
        rng: Optional[np.random.Generator] = None,
        requested_stratum: Optional[int | str] = None,
        require_full_size: bool = True,
    ) -> SampledSubgraph:
        """Sample one height-stratified connected training subgraph."""
        if rng is None:
            rng = np.random.default_rng()

        if requested_stratum is None:
            stratum_id = int(rng.integers(0, 3))
        elif isinstance(requested_stratum, str):
            if requested_stratum not in STRATUM_NAMES:
                raise ValueError(f"Unknown stratum '{requested_stratum}'.")
            stratum_id = STRATUM_NAMES.index(requested_stratum)
        else:
            stratum_id = int(requested_stratum)
            if stratum_id not in (0, 1, 2):
                raise ValueError("requested_stratum must be 0, 1, 2 or a stratum name.")

        if require_full_size:
            viable = self.component_sizes[self.component_labels] >= self.target_size
        else:
            viable = np.ones(self.num_nodes, dtype=bool)

        candidates = np.flatnonzero((self.height_strata == stratum_id) & viable)
        if candidates.size == 0:
            raise RuntimeError(
                f"No viable seed exists in stratum '{STRATUM_NAMES[stratum_id]}' "
                f"for target_size={self.target_size}."
            )

        seed_id = int(rng.choice(candidates))
        point_ids = _bfs_collect(
            self.adjacency,
            seed_id=seed_id,
            max_nodes=self.target_size,
        )
        if require_full_size and point_ids.size != self.target_size:
            raise RuntimeError(
                f"BFS returned {point_ids.size} points; expected {self.target_size}."
            )

        return SampledSubgraph(
            point_ids=point_ids,
            edge_index=induce_local_edge_index(self.adjacency, point_ids),
            seed_id=seed_id,
            stratum=STRATUM_NAMES[stratum_id],
        )

    def _select_inference_root(self, component_nodes: np.ndarray) -> int:
        """Choose a deterministic root for one original graph component.

        The manuscript does not specify the inference seed rule. We use the
        highest-Z point and break exact-height ties by the smallest original
        point ID. This choice affects batching boundaries only.
        """
        component_nodes = np.asarray(component_nodes, dtype=np.int64)
        z = self.points[component_nodes, 2]
        z_max = float(z.max())
        tied = component_nodes[np.flatnonzero(z == z_max)]
        return int(tied.min())

    def _partition_component_connectivity_preserving(
        self,
        component_nodes: np.ndarray,
    ) -> list[SampledSubgraph]:
        """Partition one original component without fragmenting its remainder.

        A breadth-first spanning tree is constructed from the fixed graph. At
        each step, the largest currently active spanning-tree subtree that fits
        within ``target_size`` is cut off. A tree subtree is connected, and its
        removal leaves the remaining tree connected. Therefore the procedure
        avoids the large number of tiny residual components produced by naive
        repeated BFS carving.
        """
        component_nodes = np.asarray(component_nodes, dtype=np.int64)
        if component_nodes.ndim != 1 or component_nodes.size == 0:
            raise ValueError("component_nodes must be a non-empty 1D array.")

        component_nodes = np.sort(component_nodes)
        n_component = int(component_nodes.size)
        component_adjacency = self.adjacency[component_nodes][:, component_nodes].tocsr()
        component_adjacency.sort_indices()

        root_global = self._select_inference_root(component_nodes)
        root_local = int(np.searchsorted(component_nodes, root_global))
        if n_component <= self.target_size:
            bfs_local = _bfs_collect(
                component_adjacency,
                seed_id=root_local,
                max_nodes=n_component,
            )
            if bfs_local.size != n_component:
                raise RuntimeError("Small inference component is not BFS-connected.")
            bfs_global = component_nodes[bfs_local]
            return [
                SampledSubgraph(
                    point_ids=bfs_global,
                    edge_index=induce_local_edge_index(self.adjacency, bfs_global),
                    seed_id=root_global,
                    stratum=None,
                )
            ]

        if component_nodes[root_local] != root_global:
            raise RuntimeError("Could not map inference root to component-local index.")

        bfs_order, predecessors = breadth_first_order(
            component_adjacency,
            i_start=root_local,
            directed=False,
            return_predecessors=True,
        )
        bfs_order = np.asarray(bfs_order, dtype=np.int64)
        predecessors = np.asarray(predecessors, dtype=np.int64)
        if bfs_order.size != n_component:
            raise RuntimeError("BFS spanning tree did not cover the complete component.")

        parent = predecessors.copy()
        parent[root_local] = -1

        child_ids = np.flatnonzero(parent >= 0).astype(np.int64, copy=False)
        parent_ids = parent[child_ids]
        tree_children = csr_matrix(
            (
                np.ones(child_ids.size, dtype=np.uint8),
                (parent_ids, child_ids),
            ),
            shape=(n_component, n_component),
            dtype=np.uint8,
        )
        tree_children.sort_indices()

        # Initial BFS-tree subtree sizes.
        subtree_size = np.ones(n_component, dtype=np.int64)
        for node in bfs_order[::-1]:
            p = parent[node]
            if p >= 0:
                subtree_size[p] += subtree_size[node]

        active = np.ones(n_component, dtype=bool)
        remaining = n_component
        chunks_local: list[tuple[np.ndarray, int]] = []

        child_indptr = tree_children.indptr
        children = tree_children.indices

        while remaining > self.target_size:
            candidates = np.flatnonzero(
                active
                & (subtree_size > 0)
                & (subtree_size <= self.target_size)
            )
            if candidates.size == 0:
                raise RuntimeError(
                    "No connectivity-preserving inference cut fits the target size."
                )

            candidate_sizes = subtree_size[candidates]
            best_size = int(candidate_sizes.max())
            best_candidates = candidates[candidate_sizes == best_size]
            # Deterministic tie break in original point-ID space.
            cut_root = int(
                best_candidates[
                    np.argmin(component_nodes[best_candidates])
                ]
            )

            stack = [cut_root]
            selected_list: list[int] = []
            while stack:
                node = int(stack.pop())
                if not active[node]:
                    continue
                selected_list.append(node)
                start, stop = child_indptr[node], child_indptr[node + 1]
                for child in children[start:stop]:
                    child = int(child)
                    if active[child]:
                        stack.append(child)

            selected = np.asarray(selected_list, dtype=np.int64)
            expected_size = int(subtree_size[cut_root])
            if selected.size != expected_size:
                raise RuntimeError(
                    "Internal spanning-tree subtree-size bookkeeping is inconsistent."
                )

            active[selected] = False
            subtree_size[selected] = 0
            remaining -= int(selected.size)
            chunks_local.append((selected, cut_root))

            # Only ancestors of the cut root change their active subtree size.
            ancestor = int(parent[cut_root])
            while ancestor >= 0:
                subtree_size[ancestor] -= int(selected.size)
                ancestor = int(parent[ancestor])

        residual = np.flatnonzero(active).astype(np.int64, copy=False)
        if residual.size == 0 or residual.size > self.target_size:
            raise RuntimeError("Invalid residual size after inference partitioning.")
        chunks_local.append((residual, root_local))

        subgraphs: list[SampledSubgraph] = []
        for selected_local, seed_local in chunks_local:
            selected_global = component_nodes[selected_local]
            seed_global = int(component_nodes[seed_local])

            # Reorder each chosen connected block by an actual BFS expansion on
            # the fixed graph restricted to that block. Membership is unchanged.
            allowed_local = np.zeros(n_component, dtype=bool)
            allowed_local[selected_local] = True
            bfs_selected_local = _bfs_collect(
                component_adjacency,
                seed_id=seed_local,
                max_nodes=int(selected_local.size),
                allowed_mask=allowed_local,
            )
            if bfs_selected_local.size != selected_local.size:
                raise RuntimeError("Final inference block is not BFS-connected.")
            bfs_selected_global = component_nodes[bfs_selected_local]

            if not np.array_equal(
                np.sort(bfs_selected_global),
                np.sort(selected_global),
            ):
                raise RuntimeError("BFS reordering changed inference block membership.")

            subgraphs.append(
                SampledSubgraph(
                    point_ids=bfs_selected_global,
                    edge_index=induce_local_edge_index(
                        self.adjacency,
                        bfs_selected_global,
                    ),
                    seed_id=seed_global,
                    stratum=None,
                )
            )

        return subgraphs

    def partition_inference(self) -> InferencePartition:
        """Cover every point exactly once with connected, non-overlapping blocks.

        Each original fixed-graph connected component is handled independently.
        Within a component, a BFS spanning tree defines connectivity-preserving
        cuts, preventing the severe remainder fragmentation caused by naive BFS
        carving. Every final block is then explicitly BFS-ordered on the fixed
        graph, and no boundary-crossing edge is included in its local edge set.
        """
        subgraphs: list[SampledSubgraph] = []

        for component_id in range(self.num_components):
            component_nodes = np.flatnonzero(
                self.component_labels == component_id
            ).astype(np.int64, copy=False)
            if component_nodes.size == 0:
                continue
            subgraphs.extend(
                self._partition_component_connectivity_preserving(component_nodes)
            )

        # Deterministic full-tree ordering: process blocks by minimum original
        # point ID. This ordering does not alter memberships or predictions.
        subgraphs.sort(key=lambda sg: int(sg.point_ids.min()))

        return InferencePartition(
            subgraphs=subgraphs,
            num_points=self.num_nodes,
            target_size=self.target_size,
        )


def _is_connected_local(subgraph: SampledSubgraph) -> bool:
    if subgraph.num_nodes <= 1:
        return True
    if subgraph.edge_index.shape[1] == 0:
        return False
    adjacency = csr_matrix(
        (
            np.ones(subgraph.edge_index.shape[1], dtype=np.uint8),
            (subgraph.edge_index[0], subgraph.edge_index[1]),
        ),
        shape=(subgraph.num_nodes, subgraph.num_nodes),
    )
    n_components, _ = connected_components(adjacency, directed=False)
    return int(n_components) == 1


def validate_sampled_subgraph(
    sampler: ConnectedSubgraphSampler,
    subgraph: SampledSubgraph,
    require_full_size: bool = False,
) -> dict[str, int | bool]:
    """Validate connectivity, IDs, size and exact induced-edge mapping."""
    ids = np.asarray(subgraph.point_ids, dtype=np.int64)
    local = np.asarray(subgraph.edge_index, dtype=np.int64)

    expected = induce_local_edge_index(sampler.adjacency, ids)
    checks: dict[str, int | bool] = {
        "nonempty": bool(ids.size > 0),
        "size_within_limit": bool(ids.size <= sampler.target_size),
        "full_size": bool(ids.size == sampler.target_size),
        "unique_point_ids": bool(np.unique(ids).size == ids.size),
        "valid_global_ids": bool(np.all((ids >= 0) & (ids < sampler.num_nodes))),
        "local_edge_shape_ok": bool(local.ndim == 2 and local.shape[0] == 2),
        "valid_local_edges": bool(
            local.size == 0 or np.all((local >= 0) & (local < ids.size))
        ),
        "self_edges": int(np.sum(local[0] == local[1])) if local.size else 0,
        "connected": bool(_is_connected_local(subgraph)),
        "induced_edge_mapping_exact": bool(np.array_equal(local, expected)),
    }
    ok = (
        checks["nonempty"]
        and checks["size_within_limit"]
        and checks["unique_point_ids"]
        and checks["valid_global_ids"]
        and checks["local_edge_shape_ok"]
        and checks["valid_local_edges"]
        and checks["self_edges"] == 0
        and checks["connected"]
        and checks["induced_edge_mapping_exact"]
    )
    if require_full_size:
        ok = bool(ok and checks["full_size"])
    checks["all_checks_pass"] = bool(ok)
    return checks


def validate_inference_partition(
    sampler: ConnectedSubgraphSampler,
    partition: InferencePartition,
) -> dict[str, int | bool]:
    """Validate exact-once coverage, non-overlap, connectivity and edge mapping."""
    sizes = partition.sizes
    all_ids = (
        np.concatenate([s.point_ids for s in partition.subgraphs])
        if partition.subgraphs
        else np.empty(0, dtype=np.int64)
    )
    counts = (
        np.bincount(all_ids, minlength=sampler.num_nodes)
        if all_ids.size
        else np.zeros(sampler.num_nodes, dtype=np.int64)
    )

    connected_flags = [_is_connected_local(s) for s in partition.subgraphs]
    mapping_flags = [
        np.array_equal(s.edge_index, induce_local_edge_index(sampler.adjacency, s.point_ids))
        for s in partition.subgraphs
    ]

    checks: dict[str, int | bool] = {
        "num_subgraphs": int(partition.num_subgraphs),
        "total_selected_points": int(all_ids.size),
        "max_subgraph_size": int(sizes.max()) if sizes.size else 0,
        "min_subgraph_size": int(sizes.min()) if sizes.size else 0,
        "points_missing": int(np.sum(counts == 0)),
        "points_repeated": int(np.sum(counts > 1)),
        "exact_once_coverage": bool(np.all(counts == 1)),
        "all_sizes_within_limit": bool(np.all(sizes <= sampler.target_size)),
        "all_subgraphs_connected": bool(all(connected_flags)),
        "all_edge_mappings_exact": bool(all(mapping_flags)),
    }
    checks["all_checks_pass"] = bool(
        checks["exact_once_coverage"]
        and checks["all_sizes_within_limit"]
        and checks["all_subgraphs_connected"]
        and checks["all_edge_mappings_exact"]
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
        print(f"Residual subgraphs:        {np.sum(sizes < partition.target_size):,}")
        print(f"Total covered points:      {sizes.sum():,}")


def _load_repository_modules():
    try:
        from .feature_extraction import load_xyz  # type: ignore
        from .graph_construction import build_fixed_knn_graph  # type: ignore
    except ImportError:
        from feature_extraction import load_xyz  # type: ignore
        from graph_construction import build_fixed_knn_graph  # type: ignore
    return load_xyz, build_fixed_knn_graph


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample connected 8,192-point BFS subgraphs from a fixed TLS k-NN graph."
    )
    parser.add_argument("input", type=Path, help="Input .csv or .txt point cloud.")
    parser.add_argument(
        "--mode",
        choices=("inference", "training"),
        default="inference",
        help="Subgraph mode (default: inference).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SUBGRAPH_SIZE,
        help="Maximum/target subgraph size (default: 8192).",
    )
    parser.add_argument("--k", type=int, default=16, help="Global Euclidean k-NN size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training sampling.")
    parser.add_argument(
        "--stratum",
        choices=STRATUM_NAMES,
        default=None,
        help="Optional fixed training seed stratum.",
    )
    parser.add_argument("--validate", action="store_true", help="Run structural validation.")
    args = parser.parse_args()

    load_xyz, build_fixed_knn_graph = _load_repository_modules()
    points = load_xyz(args.input)
    graph = build_fixed_knn_graph(points, k=args.k, verbose=False)
    sampler = ConnectedSubgraphSampler(
        points,
        graph.edge_index,
        target_size=args.size,
    )

    print(f"[Input] {args.input.name}: {points.shape[0]:,} points")
    print(f"[Fixed graph] k={args.k}, directed edges={graph.edge_index.shape[1]:,}")
    print(f"[Graph components] {sampler.num_components}")

    if args.mode == "training":
        sample = sampler.sample_training(
            rng=np.random.default_rng(args.seed),
            requested_stratum=args.stratum,
            require_full_size=True,
        )
        print("\n[Training subgraph]")
        print(f"Seed ID:                  {sample.seed_id:,}")
        print(f"Seed stratum:             {sample.stratum}")
        print(f"Points:                   {sample.num_nodes:,}")
        print(f"Local directed edges:     {sample.num_edges:,}")
        if args.validate:
            checks = validate_sampled_subgraph(sampler, sample, require_full_size=True)
            print("\n[Validation]")
            for key, value in checks.items():
                print(f"{key:<32} {value}")
            if not checks["all_checks_pass"]:
                raise RuntimeError("Training subgraph validation failed.")
            print("Status:                          OK")
    else:
        partition = sampler.partition_inference()
        print_partition_summary(partition)
        if args.validate:
            checks = validate_inference_partition(sampler, partition)
            print("\n[Validation]")
            for key, value in checks.items():
                print(f"{key:<32} {value}")
            if not checks["all_checks_pass"]:
                raise RuntimeError("Inference partition validation failed.")
            print("Status:                          OK")


if __name__ == "__main__":
    _main()
