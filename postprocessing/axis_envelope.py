"""Axis-envelope-guided stem expansion for TLS post-processing.

Author: Shahab Alaedin Baloochi

The tree height is divided into adaptive vertical bins. In bins containing stem
reference points, the local centre is the median XY coordinate and the radius
is the 95th percentile horizontal distance, clipped to 0.05--0.55 m. Missing
bins use nearest valid-bin propagation, followed by a 7-bin moving average.
Candidate points are added by BFS on the fixed k-NN graph when
probability > tau_lo and the point lies inside the local envelope.

Envelope estimation and expansion are run twice, with the second envelope
estimated from the first-pass candidate. tau_lo is supplied by the caller.
Moving-average edges use nearest-value padding and interpolation uses constant
endpoint extension.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


EPS = 1e-12

PAPER_BIN_HEIGHT_M = 0.20
PAPER_MIN_BINS = 15
PAPER_MAX_BINS = 100
PAPER_RADIUS_QUANTILE = 0.95
PAPER_MIN_RADIUS_M = 0.05
PAPER_MAX_RADIUS_M = 0.55
PAPER_SMOOTH_WINDOW = 7
PAPER_NUM_EXPANSION_PASSES = 2


@dataclass(frozen=True)
class AxisEnvelope:
    """Estimated and smoothed stem-axis envelope for one pass."""

    z_min: float
    z_max: float
    tree_height: float
    z_edges: NDArray[np.float64]
    z_centres: NDArray[np.float64]
    valid_bins: NDArray[np.bool_]
    raw_centres_xy: NDArray[np.float64]
    raw_radii: NDArray[np.float64]
    filled_centres_xy: NDArray[np.float64]
    filled_radii: NDArray[np.float64]
    smooth_centres_xy: NDArray[np.float64]
    smooth_radii: NDArray[np.float64]

    @property
    def num_bins(self) -> int:
        return int(self.z_centres.size)


@dataclass(frozen=True)
class EnvelopeMembership:
    """Per-point centreline/envelope quantities."""

    centre_xy: NDArray[np.float64]
    radius: NDArray[np.float64]
    horizontal_distance: NDArray[np.float64]
    inside: NDArray[np.bool_]


@dataclass(frozen=True)
class ExpansionPassResult:
    """Result of one envelope-estimation + connected-expansion pass."""

    envelope: AxisEnvelope
    membership: EnvelopeMembership
    eligible_mask: NDArray[np.bool_]
    candidate_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class TwoPassExpansionResult:
    """Two-pass axis-envelope expansion result."""

    initial_core_mask: NDArray[np.bool_]
    first_pass: ExpansionPassResult
    second_pass: ExpansionPassResult

    @property
    def final_candidate_mask(self) -> NDArray[np.bool_]:
        return self.second_pass.candidate_mask


def _validate_points(points: NDArray[np.floating]) -> NDArray[np.float64]:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"points must have shape (N,3) with N>0, got {xyz.shape}.")
    if not np.isfinite(xyz).all():
        raise ValueError("points contains NaN or infinite coordinates.")
    return xyz


def _validate_mask(mask: NDArray[np.bool_], num_points: int, name: str) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.shape != (num_points,):
        raise ValueError(f"{name} must have shape ({num_points},), got {array.shape}.")
    if array.dtype != np.bool_:
        # Accept 0/1 arrays only; reject arbitrary numeric masks silently becoming bool.
        if not (
            np.issubdtype(array.dtype, np.integer)
            and np.all((array == 0) | (array == 1))
        ):
            raise TypeError(f"{name} must be boolean (or integer 0/1).")
        array = array.astype(bool)
    if not np.any(array):
        raise ValueError(f"{name} must contain at least one selected point.")
    return array.astype(bool, copy=False)


def _validate_probabilities(
    probabilities: NDArray[np.floating],
    num_points: int,
) -> NDArray[np.float64]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (num_points,):
        raise ValueError(
            f"probabilities must have shape ({num_points},), got {probs.shape}."
        )
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contains NaN or infinite values.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0,1].")
    return probs


def _validate_tau_lo(tau_lo: float) -> float:
    value = float(tau_lo)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("tau_lo must be finite and lie in [0,1].")
    return value


def _validate_edge_index(
    edge_index: NDArray[np.integer],
    num_points: int,
    *,
    require_symmetric: bool = True,
) -> NDArray[np.int64]:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2,E), got {edges.shape}.")
    if edges.size:
        if np.any(edges < 0) or np.any(edges >= num_points):
            raise ValueError("edge_index contains an out-of-range node index.")
        if np.any(edges[0] == edges[1]):
            raise ValueError("edge_index contains a self-edge.")

    if require_symmetric and edges.shape[1]:
        adjacency = csr_matrix(
            (
                np.ones(edges.shape[1], dtype=np.uint8),
                (edges[0], edges[1]),
            ),
            shape=(num_points, num_points),
        )
        adjacency.data[:] = 1
        adjacency.eliminate_zeros()
        if (adjacency != adjacency.T).nnz:
            raise ValueError(
                "edge_index must be the symmetrised precomputed k-NN graph."
            )
    return edges


def adaptive_num_bins(tree_height_m: float) -> int:
    """Paper equation: min(100, max(15, round(H / 0.20)))."""
    height = float(tree_height_m)
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("tree_height_m must be finite and positive.")

    # Python's built-in round uses bankers' rounding. The manuscript only says
    # 'round'; to make conventional nearest-integer behaviour explicit for
    # positive H/0.20 values, use floor(x + 0.5).
    rounded = int(np.floor(height / PAPER_BIN_HEIGHT_M + 0.5))
    return int(min(PAPER_MAX_BINS, max(PAPER_MIN_BINS, rounded)))


def _assign_bins(
    z: NDArray[np.float64],
    z_edges: NDArray[np.float64],
) -> NDArray[np.int64]:
    # Left-closed/right-open bins, with the final bin including z_max.
    ids = np.searchsorted(z_edges, z, side="right") - 1
    return np.clip(ids, 0, z_edges.size - 2).astype(np.int64, copy=False)


def _nearest_fill_1d(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        raise ValueError("At least one valid bin is required for propagation.")

    missing_ids = np.flatnonzero(~valid)
    for idx in missing_ids:
        distances = np.abs(valid_ids - idx)
        nearest = int(valid_ids[int(np.argmin(distances))])
        out[idx] = out[nearest]
    return out


def _nearest_fill_2d(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must be two-dimensional.")
    out = array.copy()
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        raise ValueError("At least one valid bin is required for propagation.")

    missing_ids = np.flatnonzero(~valid)
    for idx in missing_ids:
        distances = np.abs(valid_ids - idx)
        nearest = int(valid_ids[int(np.argmin(distances))])
        out[idx] = out[nearest]
    return out


def _moving_average_1d(
    values: NDArray[np.floating],
    window: int = PAPER_SMOOTH_WINDOW,
) -> NDArray[np.float64]:
    """Simple arithmetic moving average with deterministic edge replication."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    if not np.isfinite(array).all():
        raise ValueError("values contains NaN or infinite values.")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer.")
    if window > array.size:
        # Adaptive binning normally provides at least 15 bins.
        raise ValueError("window cannot exceed the number of values.")

    half = window // 2
    padded = np.pad(array, (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def estimate_axis_envelope(
    points: NDArray[np.floating],
    reference_mask: NDArray[np.bool_],
) -> AxisEnvelope:
    """Estimate the adaptive, smoothed axis envelope.

    ``reference_mask`` is the high-confidence core in pass 1 and the first-pass
    candidate in pass 2.
    """
    xyz = _validate_points(points)
    reference = _validate_mask(reference_mask, xyz.shape[0], "reference_mask")

    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())
    height = z_max - z_min
    if height <= EPS:
        raise ValueError("Tree height must be positive.")

    n_bins = adaptive_num_bins(height)
    z_edges = np.linspace(z_min, z_max, n_bins + 1, dtype=np.float64)
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    bin_ids = _assign_bins(xyz[:, 2], z_edges)

    raw_centres = np.full((n_bins, 2), np.nan, dtype=np.float64)
    raw_radii = np.full(n_bins, np.nan, dtype=np.float64)
    valid = np.zeros(n_bins, dtype=bool)

    reference_ids = np.flatnonzero(reference)
    reference_bins = bin_ids[reference_ids]

    for bin_id in np.unique(reference_bins):
        ids = reference_ids[reference_bins == bin_id]
        xy = xyz[ids, :2]

        centre = np.median(xy, axis=0)
        horizontal_distance = np.linalg.norm(xy - centre[None, :], axis=1)
        radius = float(np.quantile(horizontal_distance, PAPER_RADIUS_QUANTILE))
        radius = float(np.clip(radius, PAPER_MIN_RADIUS_M, PAPER_MAX_RADIUS_M))

        raw_centres[bin_id] = centre
        raw_radii[bin_id] = radius
        valid[bin_id] = True

    if not np.any(valid):
        raise RuntimeError("No vertical bin contains reference stem points.")

    # Manuscript: nearest-neighbour propagation from closest valid bin.
    filled_centres = _nearest_fill_2d(raw_centres, valid)
    filled_radii = _nearest_fill_1d(raw_radii, valid)

    # Manuscript: arithmetic 7-bin moving average (±3 bins).
    smooth_centres = np.column_stack(
        (
            _moving_average_1d(filled_centres[:, 0]),
            _moving_average_1d(filled_centres[:, 1]),
        )
    )
    smooth_radii = _moving_average_1d(filled_radii)

    # The mean of values already inside [rmin, rmax] must remain inside the
    # same interval; the explicit check catches numerical/programming errors.
    if np.any(smooth_radii < PAPER_MIN_RADIUS_M - 1e-12) or np.any(
        smooth_radii > PAPER_MAX_RADIUS_M + 1e-12
    ):
        raise RuntimeError("Smoothed radii escaped the manuscript radius bounds.")

    return AxisEnvelope(
        z_min=z_min,
        z_max=z_max,
        tree_height=height,
        z_edges=z_edges,
        z_centres=z_centres,
        valid_bins=valid,
        raw_centres_xy=raw_centres,
        raw_radii=raw_radii,
        filled_centres_xy=filled_centres,
        filled_radii=filled_radii,
        smooth_centres_xy=smooth_centres,
        smooth_radii=smooth_radii,
    )


def envelope_membership(
    points: NDArray[np.floating],
    envelope: AxisEnvelope,
) -> EnvelopeMembership:
    """Evaluate horizontal distance to the piecewise-linear centreline."""
    xyz = _validate_points(points)
    if not isinstance(envelope, AxisEnvelope):
        raise TypeError("envelope must be AxisEnvelope.")

    # np.interp gives piecewise-linear interpolation between smoothed bin
    # centres and constant endpoint extension outside the first/last bin centre.
    centre_x = np.interp(
        xyz[:, 2],
        envelope.z_centres,
        envelope.smooth_centres_xy[:, 0],
    )
    centre_y = np.interp(
        xyz[:, 2],
        envelope.z_centres,
        envelope.smooth_centres_xy[:, 1],
    )
    radius = np.interp(
        xyz[:, 2],
        envelope.z_centres,
        envelope.smooth_radii,
    )

    centre_xy = np.column_stack((centre_x, centre_y))
    distance = np.linalg.norm(xyz[:, :2] - centre_xy, axis=1)
    inside = distance <= radius

    return EnvelopeMembership(
        centre_xy=centre_xy,
        radius=radius,
        horizontal_distance=distance,
        inside=inside,
    )



def _selected_mask_is_connected(
    adjacency: csr_matrix,
    selected_mask: NDArray[np.bool_],
) -> bool:
    selected = np.flatnonzero(selected_mask)
    if selected.size <= 1:
        return True

    allowed = np.zeros(adjacency.shape[0], dtype=bool)
    allowed[selected] = True
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    seed = int(selected[0])
    seen[seed] = True
    queue: deque[int] = deque([seed])
    count = 0

    while queue:
        node = queue.popleft()
        count += 1
        start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
        for neighbour in adjacency.indices[start:end]:
            nb = int(neighbour)
            if allowed[nb] and not seen[nb]:
                seen[nb] = True
                queue.append(nb)

    return count == int(selected.size)


def _build_adjacency(
    edge_index: NDArray[np.integer],
    num_points: int,
) -> csr_matrix:
    edges = _validate_edge_index(edge_index, num_points, require_symmetric=True)
    adjacency = csr_matrix(
        (
            np.ones(edges.shape[1], dtype=np.uint8),
            (edges[0], edges[1]),
        ),
        shape=(num_points, num_points),
    )
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def expand_connected_candidate(
    points: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    edge_index: NDArray[np.integer],
    seed_mask: NDArray[np.bool_],
    envelope: AxisEnvelope,
    tau_lo: float,
) -> ExpansionPassResult:
    """Breadth-first expansion constrained by probability and axis envelope."""
    xyz = _validate_points(points)
    probs = _validate_probabilities(probabilities, xyz.shape[0])
    seeds = _validate_mask(seed_mask, xyz.shape[0], "seed_mask")
    threshold = _validate_tau_lo(tau_lo)
    adjacency = _build_adjacency(edge_index, xyz.shape[0])

    if not _selected_mask_is_connected(adjacency, seeds):
        raise ValueError(
            "seed_mask must be connected in the precomputed k-NN graph. "
            "Step 2 assumes the single connected core retained by Step 1."
        )

    membership = envelope_membership(xyz, envelope)
    if not np.all(probs[seeds] > threshold):
        raise ValueError(
            "Every seed/core point must have probability > tau_lo. "
            "This follows from Step 1 using tau_hi with tau_lo < tau_hi."
        )

    eligible = (probs > threshold) & membership.inside

    # The already accepted seed/core is always retained. In the intended paper
    # pipeline these points satisfy tau_hi > tau_lo anyway.
    selected = seeds.copy()
    queued = seeds.copy()
    queue: deque[int] = deque(int(i) for i in np.flatnonzero(seeds))

    while queue:
        node = queue.popleft()
        start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
        for neighbour in adjacency.indices[start:end]:
            nb = int(neighbour)
            if queued[nb]:
                continue
            queued[nb] = True
            if eligible[nb]:
                selected[nb] = True
                queue.append(nb)

    if not _selected_mask_is_connected(adjacency, selected):
        raise RuntimeError("BFS expansion produced a disconnected candidate.")

    return ExpansionPassResult(
        envelope=envelope,
        membership=membership,
        eligible_mask=eligible,
        candidate_mask=selected,
    )


def two_pass_axis_envelope_expansion(
    points: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    edge_index: NDArray[np.integer],
    high_confidence_core_mask: NDArray[np.bool_],
    tau_lo: float,
) -> TwoPassExpansionResult:
    """Apply envelope estimation and BFS expansion twice."""
    xyz = _validate_points(points)
    core = _validate_mask(
        high_confidence_core_mask,
        xyz.shape[0],
        "high_confidence_core_mask",
    )
    _validate_probabilities(probabilities, xyz.shape[0])
    _validate_tau_lo(tau_lo)
    _validate_edge_index(edge_index, xyz.shape[0], require_symmetric=True)

    envelope1 = estimate_axis_envelope(xyz, core)
    pass1 = expand_connected_candidate(
        xyz,
        probabilities,
        edge_index,
        core,
        envelope1,
        tau_lo,
    )

    # Re-estimate from the first-pass candidate, then repeat the expansion.
    envelope2 = estimate_axis_envelope(xyz, pass1.candidate_mask)
    pass2 = expand_connected_candidate(
        xyz,
        probabilities,
        edge_index,
        pass1.candidate_mask,
        envelope2,
        tau_lo,
    )

    if not np.all(pass1.candidate_mask <= pass2.candidate_mask):
        raise RuntimeError("Second-pass expansion unexpectedly removed first-pass points.")

    return TwoPassExpansionResult(
        initial_core_mask=core.copy(),
        first_pass=pass1,
        second_pass=pass2,
    )


__all__ = [
    "PAPER_BIN_HEIGHT_M",
    "PAPER_MIN_BINS",
    "PAPER_MAX_BINS",
    "PAPER_RADIUS_QUANTILE",
    "PAPER_MIN_RADIUS_M",
    "PAPER_MAX_RADIUS_M",
    "PAPER_SMOOTH_WINDOW",
    "PAPER_NUM_EXPANSION_PASSES",
    "AxisEnvelope",
    "EnvelopeMembership",
    "ExpansionPassResult",
    "TwoPassExpansionResult",
    "adaptive_num_bins",
    "estimate_axis_envelope",
    "envelope_membership",
    "expand_connected_candidate",
    "two_pass_axis_envelope_expansion",
]
