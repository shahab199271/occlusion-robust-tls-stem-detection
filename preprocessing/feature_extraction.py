"""Feature extraction for occlusion-robust TLS stem detection.

Author: Shahab Alaedin Baloochi

Implements Sections 2.2.1--2.2.4 of the manuscript:
    1. relative height
    2. radial distance to tree centre
    3. distance to a coarse stem axis
    4. angle-weighted depth from a basal root
    5. vertical continuity from voxel occupancy
    6. local curvature from k-NN PCA
    7. local verticality from k-NN PCA

The final point representation is 10D: local XYZ coordinates + 7 engineered
features. CSV and whitespace-delimited TXT inputs are supported; only the
first three columns are interpreted as X, Y, Z and any additional columns are
ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

EPS = 1e-12

FEATURE_NAMES = (
    "relative_height",
    "radial_distance",
    "axis_distance",
    "angle_weighted_depth",
    "vertical_continuity",
    "curvature",
    "verticality",
)


@dataclass(frozen=True)
class TreeStatistics:
    z_min: float
    z_max: float
    height: float
    x_center: float
    y_center: float


@dataclass
class FeatureExtractionResult:
    local_xyz: np.ndarray
    engineered: np.ndarray
    node_features: np.ndarray
    knn_indices: np.ndarray
    stem_axis_point: np.ndarray
    stem_axis_direction: np.ndarray
    root_voxel_index: int
    root_voxel_center: np.ndarray
    statistics: TreeStatistics


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    if points.shape[0] < 3:
        raise ValueError("At least three points are required.")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite values.")
    return points


def _first_three_tokens(line: str) -> tuple[list[str], str]:
    """Return the first tokens and an inferred delimiter mode."""
    stripped = line.strip()
    if not stripped:
        return [], "whitespace"
    if "," in stripped:
        return [x.strip() for x in stripped.split(",")], "comma"
    return stripped.split(), "whitespace"


def load_xyz(file_path: str | Path) -> np.ndarray:
    """Load XYZ from a .csv or .txt file, ignoring columns after the first 3.

    Headered and headerless files are both accepted. CSV is interpreted as
    comma-delimited; TXT accepts arbitrary whitespace. A TXT file that happens
    to use commas is also accepted because delimiter detection is content-based.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(
            f"Unsupported extension '{path.suffix}'. Expected .csv or .txt."
        )

    first_nonempty = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                first_nonempty = line
                break
    if first_nonempty is None:
        raise ValueError(f"Input file is empty: {path}")

    tokens, delimiter_mode = _first_three_tokens(first_nonempty)
    if len(tokens) < 3:
        raise ValueError("Input file must contain at least three columns (X, Y, Z).")

    try:
        [float(tokens[i]) for i in range(3)]
        skiprows = 0
    except ValueError:
        skiprows = 1

    delimiter = "," if delimiter_mode == "comma" else None
    try:
        xyz = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            usecols=(0, 1, 2),
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not parse the first three columns of '{path}' as numeric XYZ."
        ) from exc

    if xyz.ndim == 1:
        xyz = xyz.reshape(1, -1)
    return _validate_points(xyz)


# -----------------------------------------------------------------------------
# Section 2.2.1: tree-level reference statistics and local coordinates
# -----------------------------------------------------------------------------

def compute_tree_statistics(points: np.ndarray) -> TreeStatistics:
    points = _validate_points(points)
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    return TreeStatistics(
        z_min=z_min,
        z_max=z_max,
        height=z_max - z_min,
        x_center=float(np.median(points[:, 0])),
        y_center=float(np.median(points[:, 1])),
    )


def to_tree_local_coordinates(points: np.ndarray, stats: TreeStatistics) -> np.ndarray:
    points = _validate_points(points)
    local = points.copy()
    local[:, 0] -= stats.x_center
    local[:, 1] -= stats.y_center
    local[:, 2] -= stats.z_min
    return local


# -----------------------------------------------------------------------------
# Feature 1: relative height, Eq. (2)
# -----------------------------------------------------------------------------

def compute_relative_height(points: np.ndarray, stats: TreeStatistics) -> np.ndarray:
    if stats.height <= EPS:
        raise ValueError("Tree height is zero or numerically negligible.")
    return (points[:, 2] - stats.z_min) / (stats.height + EPS)


# -----------------------------------------------------------------------------
# Feature 2: radial distance to centre, Eqs. (3)--(4)
# -----------------------------------------------------------------------------

def compute_radial_distance(points: np.ndarray, stats: TreeStatistics) -> np.ndarray:
    dx = points[:, 0] - stats.x_center
    dy = points[:, 1] - stats.y_center
    radius = np.hypot(dx, dy)
    r_max = float(radius.max())
    if r_max <= EPS:
        return np.zeros(points.shape[0], dtype=np.float64)
    return radius / (r_max + EPS)


# -----------------------------------------------------------------------------
# Shared voxel helpers
# -----------------------------------------------------------------------------

def _voxel_keys(
    points: np.ndarray,
    voxel_size_xy: float,
    voxel_size_z: float,
) -> np.ndarray:
    if voxel_size_xy <= 0 or voxel_size_z <= 0:
        raise ValueError("Voxel sizes must be positive.")
    origin = points.min(axis=0)
    scaled = np.column_stack(
        (
            (points[:, 0] - origin[0]) / voxel_size_xy,
            (points[:, 1] - origin[1]) / voxel_size_xy,
            (points[:, 2] - origin[2]) / voxel_size_z,
        )
    )
    return np.floor(scaled + EPS).astype(np.int64)


# -----------------------------------------------------------------------------
# Feature 5: vertical continuity, Eq. (12)
# Computed before Feature 3 because it weights the coarse axis centroid.
# -----------------------------------------------------------------------------

def compute_vertical_continuity(
    points: np.ndarray,
    voxel_size_xy: float = 0.10,
    voxel_size_z: float = 0.10,
    num_vertical_layers: int = 10,
) -> np.ndarray:
    points = _validate_points(points)
    if num_vertical_layers <= 0:
        raise ValueError("num_vertical_layers must be positive.")

    keys = _voxel_keys(points, voxel_size_xy, voxel_size_z)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    occupied = {tuple(v) for v in unique_keys.tolist()}

    # The paper specifies L=10 layers "around" the current layer but does not
    # prescribe the endpoint convention for an even L. We use five layers
    # below/current-side and four above: offsets -5,...,+4 (10 layers total).
    lower = num_vertical_layers // 2
    offsets = range(-lower, num_vertical_layers - lower)

    voxel_scores = np.zeros(len(unique_keys), dtype=np.float64)
    for idx, (u, v, w) in enumerate(unique_keys):
        occupied_layers = 0
        for dz in offsets:
            z_layer = int(w + dz)
            layer_hit = False
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    if (int(u + du), int(v + dv), z_layer) in occupied:
                        layer_hit = True
                        break
                if layer_hit:
                    break
            occupied_layers += int(layer_hit)
        voxel_scores[idx] = occupied_layers / float(num_vertical_layers)

    return voxel_scores[inverse]


# -----------------------------------------------------------------------------
# Feature 3: distance to estimated stem axis, Eqs. (5)--(8)
# -----------------------------------------------------------------------------

def estimate_stem_axis(
    points: np.ndarray,
    relative_height: np.ndarray,
    vertical_continuity: np.ndarray,
    mid_stem_range: tuple[float, float] = (0.05, 0.35),
    support_radius_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    points = _validate_points(points)
    relative_height = np.asarray(relative_height, dtype=np.float64)
    vertical_continuity = np.asarray(vertical_continuity, dtype=np.float64)
    if relative_height.shape != (points.shape[0],):
        raise ValueError("relative_height must have shape (N,).")
    if vertical_continuity.shape != (points.shape[0],):
        raise ValueError("vertical_continuity must have shape (N,).")

    low, high = mid_stem_range
    if not (0 <= low < high <= 1):
        raise ValueError("mid_stem_range must satisfy 0 <= low < high <= 1.")
    mask = (relative_height >= low) & (relative_height <= high)
    if not np.any(mask):
        raise ValueError("No points lie in the requested mid-stem height range.")

    mid_points = points[mask]
    weights = vertical_continuity[mask]
    weight_sum = float(weights.sum())
    p_stem = (
        np.sum(mid_points * weights[:, None], axis=0) / weight_sum
        if weight_sum > EPS
        else mid_points.mean(axis=0)
    )

    displacement = points - p_stem
    support_mask = np.linalg.norm(displacement, axis=1) <= support_radius_m
    support = displacement[support_mask]
    if support.shape[0] < 3:
        raise ValueError("Fewer than three points are available for axis estimation.")

    # Unit vector that best aligns, in least-squares sense, with p_i - p_stem:
    # principal eigenvector of the uncentred second-moment matrix through p_stem.
    second_moment = support.T @ support
    eigvals, eigvecs = np.linalg.eigh(second_moment)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    direction /= np.linalg.norm(direction) + EPS
    if direction[2] < 0:
        direction = -direction
    return p_stem, direction


def compute_stem_axis_distance(
    points: np.ndarray,
    p_stem: np.ndarray,
    v_stem: np.ndarray,
) -> np.ndarray:
    points = _validate_points(points)
    p_stem = np.asarray(p_stem, dtype=np.float64).reshape(3)
    v_stem = np.asarray(v_stem, dtype=np.float64).reshape(3)
    v_stem /= np.linalg.norm(v_stem) + EPS

    u = points - p_stem
    projection = (u @ v_stem)[:, None] * v_stem[None, :]
    distances = np.linalg.norm(u - projection, axis=1)
    d_max = float(distances.max())
    if d_max <= EPS:
        return np.zeros(points.shape[0], dtype=np.float64)
    return distances / (d_max + EPS)


# -----------------------------------------------------------------------------
# Feature 4: angle-weighted depth from a basal root, Eqs. (9)--(11)
# -----------------------------------------------------------------------------

def _voxelize_mean_centres(
    points: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = _voxel_keys(points, voxel_size, voxel_size)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    centres = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, d], minlength=len(unique_keys))
            / counts
            for d in range(3)
        ]
    )
    return unique_keys, inverse, centres


def _knn_without_self(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    if k <= 0:
        raise ValueError("k must be positive.")
    if n <= k:
        raise ValueError(f"Need N > k, got N={n}, k={k}.")

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    distances = np.asarray(distances)
    indices = np.asarray(indices)

    row_ids = np.arange(n)[:, None]
    not_self = indices != row_ids
    indices = indices[not_self].reshape(n, k)
    distances = distances[not_self].reshape(n, k)
    return distances, indices


def _build_undirected_weighted_voxel_graph(
    centres: np.ndarray,
    k: int = 16,
) -> csr_matrix:
    n = centres.shape[0]
    if n == 1:
        return csr_matrix((1, 1), dtype=np.float64)
    k_eff = min(k, n - 1)
    _, neighbours = _knn_without_self(centres, k_eff)

    rows = np.repeat(np.arange(n), k_eff)
    cols = neighbours.reshape(-1)
    delta = centres[cols] - centres[rows]
    d = np.linalg.norm(delta, axis=1)
    dz = np.abs(delta[:, 2])

    valid = d > EPS
    rows, cols, d, dz = rows[valid], cols[valid], d[valid], dz[valid]
    cos_alpha = np.clip(dz / (d + EPS), 0.0, 1.0)
    sin_alpha = np.sqrt(np.maximum(0.0, 1.0 - cos_alpha**2))
    costs = (1.0 + 2.0 * sin_alpha) * d

    directed_graph = csr_matrix((costs, (rows, cols)), shape=(n, n))
    return directed_graph.maximum(directed_graph.T).tocsr()


def _select_basal_root(
    points: np.ndarray,
    voxel_keys: np.ndarray,
    point_to_voxel: np.ndarray,
    voxel_centres: np.ndarray,
) -> int:
    """Deterministic implementation of the manuscript's basal root description.

    The manuscript specifies the voxel with the lowest vertical coordinate among
    those nearest to the median horizontal position of the tree base. Because it
    does not define a separate numerical thickness for "tree base", we use the
    lowest occupied 0.10 m voxel layer (the same scale used by this descriptor),
    compute the median XY of points in that layer, and choose the nearest voxel
    centre within that same lowest layer.
    """
    min_layer = int(voxel_keys[:, 2].min())
    base_voxel_mask = voxel_keys[:, 2] == min_layer
    base_point_mask = base_voxel_mask[point_to_voxel]
    base_xy = np.median(points[base_point_mask, :2], axis=0)
    base_ids = np.flatnonzero(base_voxel_mask)
    nearest = np.argmin(np.linalg.norm(voxel_centres[base_ids, :2] - base_xy, axis=1))
    return int(base_ids[nearest])


def compute_angle_weighted_depth(
    points: np.ndarray,
    voxel_size: float = 0.10,
    voxel_knn_k: int = 16,
    return_debug: bool = False,
):
    points = _validate_points(points)
    keys, inverse, centres = _voxelize_mean_centres(points, voxel_size)
    n_voxels = centres.shape[0]
    if n_voxels == 1:
        depth = np.zeros(points.shape[0], dtype=np.float64)
        if return_debug:
            return depth, 0, centres[0], n_voxels
        return depth

    root = _select_basal_root(points, keys, inverse, centres)
    graph = _build_undirected_weighted_voxel_graph(centres, k=voxel_knn_k)
    distances = np.asarray(dijkstra(graph, directed=False, indices=root), dtype=np.float64)

    finite = np.isfinite(distances)
    if not np.any(finite):
        raise RuntimeError("No finite shortest-path distances were found from the root.")
    max_reachable = float(distances[finite].max())
    distances[~finite] = max_reachable
    normalized = (
        np.zeros_like(distances)
        if max_reachable <= EPS
        else distances / (max_reachable + EPS)
    )
    depth = normalized[inverse]
    if return_debug:
        return depth, root, centres[root].copy(), n_voxels
    return depth


# -----------------------------------------------------------------------------
# Features 6 and 7: k-NN PCA curvature and verticality, Eqs. (13)--(16)
# -----------------------------------------------------------------------------

def compute_knn_indices(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = _validate_points(points)
    _, indices = _knn_without_self(points, k)
    return indices.astype(np.int64, copy=False)


def compute_knn_pca_features(
    points: np.ndarray,
    knn_indices: np.ndarray,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    points = _validate_points(points)
    knn_indices = np.asarray(knn_indices, dtype=np.int64)
    if knn_indices.ndim != 2 or knn_indices.shape[0] != points.shape[0]:
        raise ValueError("knn_indices must have shape (N, k).")
    if knn_indices.shape[1] < 2:
        raise ValueError("At least two neighbours are required for PCA.")
    if np.any(knn_indices < 0) or np.any(knn_indices >= points.shape[0]):
        raise ValueError("knn_indices contains an out-of-range point index.")

    n, k = knn_indices.shape
    curvature = np.empty(n, dtype=np.float64)
    verticality = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        neighbourhoods = points[knn_indices[start:stop]]
        mu = neighbourhoods.mean(axis=1, keepdims=True)
        centred = neighbourhoods - mu
        covariance = np.einsum("bki,bkj->bij", centred, centred, optimize=True) / float(k - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        lambda3 = eigenvalues[:, 0]
        lambda_sum = eigenvalues.sum(axis=1)
        curvature[start:stop] = lambda3 / (lambda_sum + EPS)

        normal = eigenvectors[:, :, 0]
        verticality[start:stop] = 1.0 - np.abs(normal[:, 2])

    return curvature, verticality


# -----------------------------------------------------------------------------
# Complete 7-feature + XYZ extraction, Eq. (17)
# -----------------------------------------------------------------------------

def extract_all_features(
    points: np.ndarray,
    k: int = 16,
    point_knn_indices: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> FeatureExtractionResult:
    points = _validate_points(points)
    stats = compute_tree_statistics(points)
    if stats.height <= EPS:
        raise ValueError("Tree height is zero or numerically negligible.")

    local_xyz = to_tree_local_coordinates(points, stats)
    relative_height = compute_relative_height(points, stats)
    radial_distance = compute_radial_distance(points, stats)

    vertical_continuity = compute_vertical_continuity(
        points,
        voxel_size_xy=0.10,
        voxel_size_z=0.10,
        num_vertical_layers=10,
    )

    p_stem, v_stem = estimate_stem_axis(
        points,
        relative_height,
        vertical_continuity,
        mid_stem_range=(0.05, 0.35),
        support_radius_m=1.0,
    )
    axis_distance = compute_stem_axis_distance(points, p_stem, v_stem)

    angle_weighted_depth, root_idx, root_center, n_path_voxels = compute_angle_weighted_depth(
        points,
        voxel_size=0.10,
        voxel_knn_k=16,
        return_debug=True,
    )

    if point_knn_indices is None:
        point_knn_indices = compute_knn_indices(points, k=k)
    else:
        point_knn_indices = np.asarray(point_knn_indices, dtype=np.int64)

    curvature, verticality = compute_knn_pca_features(points, point_knn_indices)

    engineered = np.column_stack(
        (
            relative_height,
            radial_distance,
            axis_distance,
            angle_weighted_depth,
            vertical_continuity,
            curvature,
            verticality,
        )
    )

    tiny_roundoff = 1e-10
    if np.any(engineered < -tiny_roundoff) or np.any(engineered > 1.0 + tiny_roundoff):
        raise RuntimeError("An engineered feature fell outside the expected [0,1] interval.")
    engineered = np.clip(engineered, 0.0, 1.0)

    node_features = np.column_stack((local_xyz, engineered))
    if not np.isfinite(node_features).all():
        raise RuntimeError("Feature extraction produced NaN or infinite values.")

    result = FeatureExtractionResult(
        local_xyz=local_xyz,
        engineered=engineered,
        node_features=node_features,
        knn_indices=point_knn_indices,
        stem_axis_point=p_stem,
        stem_axis_direction=v_stem,
        root_voxel_index=root_idx,
        root_voxel_center=root_center,
        statistics=stats,
    )

    if verbose:
        print_feature_summary(result, n_path_voxels=n_path_voxels, k=k)
    return result


def print_feature_summary(
    result: FeatureExtractionResult,
    n_path_voxels: Optional[int] = None,
    k: int = 16,
) -> None:
    stats = result.statistics
    print("\n[Feature extraction]")
    print(f"Points:                 {result.node_features.shape[0]:,}")
    print(f"Tree height:            {stats.height:.6f} m")
    print(f"Horizontal centre XY:   ({stats.x_center:.6f}, {stats.y_center:.6f})")
    print(f"Point k-NN:             k={k}")
    if n_path_voxels is not None:
        print(f"Path-depth voxels:      {n_path_voxels:,}")
    print(f"Stem-axis point:        {np.array2string(result.stem_axis_point, precision=6)}")
    print(f"Stem-axis direction:    {np.array2string(result.stem_axis_direction, precision=6)}")
    print(f"Basal root voxel:       {np.array2string(result.root_voxel_center, precision=6)}")
    print("\nFeature statistics:")
    for j, name in enumerate(FEATURE_NAMES):
        values = result.engineered[:, j]
        print(
            f"  {name:<22} min={values.min():.6f} "
            f"max={values.max():.6f} mean={values.mean():.6f}"
        )
    print(f"\nEngineered shape:      {result.engineered.shape}")
    print(f"Final input shape:      {result.node_features.shape}")
    print(f"NaN count:              {np.isnan(result.node_features).sum()}")
    print(f"Inf count:              {np.isinf(result.node_features).sum()}")
    print("Status:                 OK")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Extract TLS stem-detection features.")
    parser.add_argument("input", type=Path, help="Input .csv or .txt point cloud.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz output path containing XYZ, features and k-NN indices.",
    )
    parser.add_argument("--k", type=int, default=16, help="Point k-NN size (default: 16).")
    args = parser.parse_args()

    points = load_xyz(args.input)
    print(f"[Input] {args.input.name}: {points.shape[0]:,} points, using columns 1--3 as XYZ")
    result = extract_all_features(points, k=args.k, verbose=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            xyz=points,
            local_xyz=result.local_xyz,
            engineered=result.engineered,
            node_features=result.node_features,
            knn_indices=result.knn_indices,
            stem_axis_point=result.stem_axis_point,
            stem_axis_direction=result.stem_axis_direction,
            root_voxel_center=result.root_voxel_center,
            feature_names=np.asarray(FEATURE_NAMES),
        )
        print(f"Saved:                  {args.output}")


if __name__ == "__main__":
    _main()
