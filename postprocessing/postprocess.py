"""High-confidence stem-core identification for TLS post-processing.

Author: Shahab Alaedin Baloochi

This module implements Step 1 of Section 2.8 in the manuscript
"Occlusion-Robust Stem Detection in Individual-Tree Terrestrial Laser Scanning
Point Clouds Using Graph-Based Deep Learning".

Published procedure implemented here
------------------------------------
1. Apply a strict global high-confidence threshold ``tau_hi`` to the predicted
   stem probabilities.
2. Group points with probability >= ``tau_hi`` into connected components using
   the *precomputed* symmetric Euclidean k-NN graph.
3. Retain the component whose lowest point is closest to the tree base.
4. Perform no reconstruction, interpolation, or gap filling.

The manuscript does not publish a numerical value for ``tau_hi``. Therefore it
is a required caller input rather than a guessed default.

Reproducibility boundary
------------------------
The paper describes selecting the component "whose lowest point is closest to
the tree base" but does not give a separate numerical tree-base point or a
tie-breaking rule for components with identical lowest height. Because the
criterion is expressed through the *lowest point*, this implementation uses the
tree's minimum z as the base height and selects the component with the smallest
vertical distance between its minimum z and that base height. This is equivalent
to selecting the component with the smallest minimum z.

If two or more components have exactly the same base distance, the paper does
not specify what to do. For deterministic behaviour, ties are resolved by:
    1. larger component size;
    2. smaller minimum original point index.

No ground-truth labels are used by this module, and the graph is never rebuilt
or modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    from .axis_envelope import (
        TwoPassExpansionResult,
        two_pass_axis_envelope_expansion,
    )
except ImportError:  # allow direct execution from postprocessing/
    from axis_envelope import (
        TwoPassExpansionResult,
        two_pass_axis_envelope_expansion,
    )


EPS = 1e-12


@dataclass(frozen=True)
class HighConfidenceComponent:
    """Summary of one high-confidence connected component."""

    component_id: int
    point_ids: NDArray[np.int64]
    size: int
    lowest_point_id: int
    lowest_z: float
    base_distance_z: float


@dataclass(frozen=True)
class HighConfidenceCoreResult:
    """Complete Step-1 output aligned to the original tree point order."""

    tau_hi: float
    tree_base_z: float
    high_confidence_mask: NDArray[np.bool_]
    component_labels: NDArray[np.int64]
    components: tuple[HighConfidenceComponent, ...]
    selected_component_id: int
    core_mask: NDArray[np.bool_]

    @property
    def num_high_confidence(self) -> int:
        return int(self.high_confidence_mask.sum())

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def num_core_points(self) -> int:
        return int(self.core_mask.sum())

    @property
    def core_point_ids(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.core_mask).astype(np.int64, copy=False)


@dataclass(frozen=True)
class CoreEnvelopePostprocessingResult:
    """Combined output of manuscript post-processing Steps 1 and 2 only.

    Step 3 (TreeQSM-style local cylinder filtering) is intentionally not
    included in this repository wrapper.
    """

    tau_hi: float
    tau_lo: float
    core: HighConfidenceCoreResult
    expansion: TwoPassExpansionResult

    @property
    def final_candidate_mask(self) -> NDArray[np.bool_]:
        """Final candidate after the second axis-envelope expansion pass."""
        return self.expansion.final_candidate_mask

    @property
    def num_final_points(self) -> int:
        return int(self.final_candidate_mask.sum())


def _validate_points(points: NDArray[np.floating]) -> NDArray[np.float64]:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"points must have shape (N,3) with N>0, got {xyz.shape}.")
    if not np.isfinite(xyz).all():
        raise ValueError("points contains NaN or infinite coordinates.")
    return xyz


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


def _validate_tau_hi(tau_hi: float) -> float:
    value = float(tau_hi)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("tau_hi must be finite and lie in [0,1].")
    return value


def _validate_threshold_pair(tau_hi: float, tau_lo: float) -> tuple[float, float]:
    """Validate the manuscript requirement tau_lo < tau_hi."""
    hi = _validate_tau_hi(tau_hi)
    lo = float(tau_lo)
    if not np.isfinite(lo) or not 0.0 <= lo <= 1.0:
        raise ValueError("tau_lo must be finite and lie in [0,1].")
    if not lo < hi:
        raise ValueError(
            "Post-processing requires tau_lo < tau_hi, as stated in the manuscript."
        )
    return hi, lo


def _build_symmetric_adjacency(
    edge_index: NDArray[np.integer],
    num_points: int,
) -> csr_matrix:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2,E), got {edges.shape}.")
    if edges.size:
        if np.any(edges < 0) or np.any(edges >= num_points):
            raise ValueError("edge_index contains an out-of-range node index.")
        if np.any(edges[0] == edges[1]):
            raise ValueError("edge_index contains a self-edge.")

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

    if (adjacency != adjacency.T).nnz:
        raise ValueError(
            "edge_index must be the symmetrised precomputed Euclidean k-NN graph."
        )
    return adjacency


def _component_summaries(
    xyz: NDArray[np.float64],
    high_ids: NDArray[np.int64],
    labels_high: NDArray[np.int64],
    tree_base_z: float,
) -> tuple[HighConfidenceComponent, ...]:
    summaries: list[HighConfidenceComponent] = []
    num_components = int(labels_high.max()) + 1

    for component_id in range(num_components):
        member_local = np.flatnonzero(labels_high == component_id)
        point_ids = high_ids[member_local]

        z_values = xyz[point_ids, 2]
        lowest_z = float(z_values.min())

        # Deterministic lowest point if several points share the same minimum z.
        lowest_candidates = point_ids[
            np.isclose(z_values, lowest_z, rtol=0.0, atol=EPS)
        ]
        lowest_point_id = int(lowest_candidates.min())

        summaries.append(
            HighConfidenceComponent(
                component_id=component_id,
                point_ids=point_ids.astype(np.int64, copy=False),
                size=int(point_ids.size),
                lowest_point_id=lowest_point_id,
                lowest_z=lowest_z,
                base_distance_z=float(abs(lowest_z - tree_base_z)),
            )
        )

    return tuple(summaries)


def _select_base_anchored_component(
    components: tuple[HighConfidenceComponent, ...],
) -> HighConfidenceComponent:
    if not components:
        raise ValueError("No high-confidence connected components are available.")

    # Paper criterion first: lowest point closest to tree base.
    # Deterministic tie-breaks are explicitly repository conventions.
    return min(
        components,
        key=lambda component: (
            component.base_distance_z,
            -component.size,
            int(component.point_ids.min()),
        ),
    )


def identify_high_confidence_core(
    points: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    edge_index: NDArray[np.integer],
    tau_hi: float,
) -> HighConfidenceCoreResult:
    """Identify the paper's single base-anchored high-confidence stem core.

    Parameters
    ----------
    points:
        Whole-tree XYZ coordinates in metric units. Global or tree-local XY may
        be used; only z and point order enter Step 1.
    probabilities:
        One predicted stem probability per point.
    edge_index:
        Precomputed symmetric whole-tree Euclidean k-NN graph. Connectivity is
        used exactly as supplied; the graph is not rebuilt in feature space.
    tau_hi:
        Strict high-confidence threshold from the post-processing configuration.
        The manuscript requires probability >= tau_hi for Step 1 but does not
        publish the numerical value.

    Returns
    -------
    HighConfidenceCoreResult
        Masks and component metadata aligned to the original point order.

    Raises
    ------
    RuntimeError
        If no point reaches tau_hi. The manuscript does not specify a fallback,
        so this implementation does not silently lower the threshold or create
        a core by another method.
    """
    xyz = _validate_points(points)
    probs = _validate_probabilities(probabilities, xyz.shape[0])
    threshold = _validate_tau_hi(tau_hi)
    adjacency = _build_symmetric_adjacency(edge_index, xyz.shape[0])

    high_mask = probs >= threshold
    high_ids = np.flatnonzero(high_mask).astype(np.int64, copy=False)
    if high_ids.size == 0:
        raise RuntimeError(
            "No point satisfies probability >= tau_hi. "
            "The manuscript does not define a fallback core."
        )

    induced = adjacency[high_ids][:, high_ids].tocsr()
    induced.sort_indices()
    num_components, labels_high = connected_components(
        induced,
        directed=False,
        return_labels=True,
    )
    if int(num_components) <= 0:
        raise RuntimeError("Failed to identify high-confidence components.")

    tree_base_z = float(xyz[:, 2].min())
    components = _component_summaries(
        xyz,
        high_ids,
        labels_high.astype(np.int64, copy=False),
        tree_base_z,
    )
    selected = _select_base_anchored_component(components)

    component_labels = np.full(xyz.shape[0], -1, dtype=np.int64)
    component_labels[high_ids] = labels_high.astype(np.int64, copy=False)

    core_mask = np.zeros(xyz.shape[0], dtype=bool)
    core_mask[selected.point_ids] = True

    # Internal invariants: the selected core is exactly one connected component,
    # is a subset of the high-confidence mask, and uses no added/gap-filled point.
    if not np.all(high_mask[core_mask]):
        raise RuntimeError("Selected core contains a non-high-confidence point.")
    if int(core_mask.sum()) != selected.size:
        raise RuntimeError("Selected core size is inconsistent with component metadata.")
    core_induced = adjacency[selected.point_ids][:, selected.point_ids].tocsr()
    n_core_components, _ = connected_components(core_induced, directed=False)
    if int(n_core_components) != 1:
        raise RuntimeError("Selected high-confidence core is not connected.")

    result = HighConfidenceCoreResult(
        tau_hi=threshold,
        tree_base_z=tree_base_z,
        high_confidence_mask=high_mask,
        component_labels=component_labels,
        components=components,
        selected_component_id=selected.component_id,
        core_mask=core_mask,
    )
    validate_high_confidence_core_result(result, xyz, probs)
    return result


def postprocess_core_envelope(
    points: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    edge_index: NDArray[np.integer],
    *,
    tau_hi: float,
    tau_lo: float,
) -> CoreEnvelopePostprocessingResult:
    """Run manuscript post-processing Steps 1 and 2 as one pipeline.

    This function intentionally stops after the second axis-envelope expansion
    pass. It does *not* implement Step 3 (TreeQSM-style patch/cylinder
    filtering).

    Parameters
    ----------
    points:
        Whole-tree metric XYZ coordinates aligned with ``probabilities`` and
        ``edge_index``.
    probabilities:
        One predicted stem probability per point, normally the full-tree output
        of ``inference/inference.py``.
    edge_index:
        The same precomputed symmetric Euclidean k-NN graph used by the
        repository preprocessing/inference pipeline.
    tau_hi:
        Required high-confidence threshold for Step 1. The manuscript does not
        publish its numerical value.
    tau_lo:
        Required lower expansion threshold for Step 2. The manuscript requires
        ``tau_lo < tau_hi`` and does not publish its numerical value.

    Returns
    -------
    CoreEnvelopePostprocessingResult
        Step-1 core metadata plus both Step-2 expansion passes, all aligned to
        original point order.
    """
    hi, lo = _validate_threshold_pair(tau_hi, tau_lo)

    core = identify_high_confidence_core(
        points,
        probabilities,
        edge_index,
        tau_hi=hi,
    )

    expansion = two_pass_axis_envelope_expansion(
        points,
        probabilities,
        edge_index,
        core.core_mask,
        tau_lo=lo,
    )

    if not np.array_equal(expansion.initial_core_mask, core.core_mask):
        raise RuntimeError(
            "Step-2 initial core does not exactly match the Step-1 selected core."
        )
    if not np.all(core.core_mask <= expansion.first_pass.candidate_mask):
        raise RuntimeError("Step 2 unexpectedly removed Step-1 core points.")
    if not np.all(
        expansion.first_pass.candidate_mask <= expansion.second_pass.candidate_mask
    ):
        raise RuntimeError("Second expansion pass unexpectedly removed first-pass points.")

    result = CoreEnvelopePostprocessingResult(
        tau_hi=hi,
        tau_lo=lo,
        core=core,
        expansion=expansion,
    )
    checks = validate_core_envelope_postprocessing_result(
        result,
        points,
        probabilities,
    )
    if not bool(checks["all_checks_pass"]):
        raise RuntimeError(
            "Combined Step-1/Step-2 post-processing validation failed: "
            f"{checks}"
        )
    return result


def validate_core_envelope_postprocessing_result(
    result: CoreEnvelopePostprocessingResult,
    points: NDArray[np.floating],
    probabilities: NDArray[np.floating],
) -> dict[str, bool | int]:
    """Validate the combined Steps 1-2 result without recomputing the pipeline."""
    if not isinstance(result, CoreEnvelopePostprocessingResult):
        raise TypeError("result must be CoreEnvelopePostprocessingResult.")

    xyz = _validate_points(points)
    probs = _validate_probabilities(probabilities, xyz.shape[0])

    threshold_pair_valid = True
    try:
        hi, lo = _validate_threshold_pair(result.tau_hi, result.tau_lo)
    except (TypeError, ValueError):
        threshold_pair_valid = False
        hi = float("nan")
        lo = float("nan")

    core_checks = validate_high_confidence_core_result(result.core, xyz, probs)
    expansion = result.expansion
    final_mask = np.asarray(expansion.final_candidate_mask)

    shape_ok = bool(final_mask.shape == (xyz.shape[0],))
    final_nonempty = bool(shape_ok and np.any(final_mask))
    core_matches = bool(
        expansion.initial_core_mask.shape == result.core.core_mask.shape
        and np.array_equal(expansion.initial_core_mask, result.core.core_mask)
    )
    monotonic = bool(
        expansion.first_pass.candidate_mask.shape == result.core.core_mask.shape
        and expansion.second_pass.candidate_mask.shape == result.core.core_mask.shape
        and np.all(result.core.core_mask <= expansion.first_pass.candidate_mask)
        and np.all(
            expansion.first_pass.candidate_mask
            <= expansion.second_pass.candidate_mask
        )
    )

    # Because Step 1 has probability >= tau_hi and tau_hi > tau_lo, every core
    # point also exceeds tau_lo. Every subsequently added Step-2 point must
    # strictly exceed tau_lo.
    final_probability_consistent = bool(
        shape_ok
        and threshold_pair_valid
        and np.all(probs[final_mask] > lo)
    )

    checks: dict[str, bool | int] = {
        "threshold_pair_valid": threshold_pair_valid,
        "stored_thresholds_match_core": bool(
            threshold_pair_valid
            and np.isclose(result.core.tau_hi, hi, rtol=0.0, atol=0.0)
        ),
        "core_valid": bool(core_checks.get("all_checks_pass", False)),
        "final_mask_shape": shape_ok,
        "final_nonempty": final_nonempty,
        "step2_starts_from_step1_core": core_matches,
        "candidate_masks_monotonic": monotonic,
        "final_probability_consistent": final_probability_consistent,
    }
    checks["all_checks_pass"] = bool(
        all(bool(value) for key, value in checks.items() if key != "all_checks_pass")
    )
    return checks


def validate_high_confidence_core_result(
    result: HighConfidenceCoreResult,
    points: NDArray[np.floating],
    probabilities: NDArray[np.floating],
) -> dict[str, bool | int]:
    """Validate point alignment and Step-1 threshold/base-selection invariants."""
    if not isinstance(result, HighConfidenceCoreResult):
        raise TypeError("result must be HighConfidenceCoreResult.")

    xyz = _validate_points(points)
    probs = _validate_probabilities(probabilities, xyz.shape[0])
    n = xyz.shape[0]

    selected = next(
        (
            component
            for component in result.components
            if component.component_id == result.selected_component_id
        ),
        None,
    )

    expected_high = probs >= result.tau_hi
    expected_base_z = float(xyz[:, 2].min())

    checks: dict[str, bool | int] = {
        "high_mask_shape": bool(result.high_confidence_mask.shape == (n,)),
        "component_labels_shape": bool(result.component_labels.shape == (n,)),
        "core_mask_shape": bool(result.core_mask.shape == (n,)),
        "high_threshold_exact": bool(
            np.array_equal(result.high_confidence_mask, expected_high)
        ),
        "base_height_exact": bool(
            np.isclose(result.tree_base_z, expected_base_z, rtol=0.0, atol=EPS)
        ),
        "has_components": bool(len(result.components) > 0),
        "selected_component_exists": bool(selected is not None),
        "core_nonempty": bool(np.any(result.core_mask)),
        "core_subset_of_high": bool(
            np.all(~result.core_mask | result.high_confidence_mask)
        ),
        "non_high_labels_are_minus_one": bool(
            np.all(result.component_labels[~result.high_confidence_mask] == -1)
        ),
    }

    if selected is None:
        checks["core_matches_selected_component"] = False
        checks["selected_is_base_anchored"] = False
    else:
        expected_core = np.zeros(n, dtype=bool)
        expected_core[selected.point_ids] = True
        checks["core_matches_selected_component"] = bool(
            np.array_equal(result.core_mask, expected_core)
        )
        best = _select_base_anchored_component(result.components)
        checks["selected_is_base_anchored"] = bool(
            best.component_id == result.selected_component_id
        )

    checks["all_checks_pass"] = bool(
        all(bool(value) for key, value in checks.items() if key != "all_checks_pass")
    )
    return checks


__all__ = [
    "HighConfidenceComponent",
    "HighConfidenceCoreResult",
    "CoreEnvelopePostprocessingResult",
    "identify_high_confidence_core",
    "validate_high_confidence_core_result",
    "postprocess_core_envelope",
    "validate_core_envelope_postprocessing_result",
]
