"""Paper-faithful synthetic occlusion protocol for TLS tree point clouds.

Author: Shahab Alaedin Baloochi

This module implements the *published* protocol in Sections 2.6--2.7 of the
manuscript "Occlusion-Robust Stem Detection in Individual-Tree Terrestrial
Laser Scanning Point Clouds Using Graph-Based Deep Learning".

Published protocol encoded here
-------------------------------
* Tree-local placement uses relative height ``z_norm in [0, 1]`` and planar
  offsets.
* Each controlled-evaluation scenario contains exactly two horizontal bands
  plus ``K`` additional spatially coherent 3-D regions.
* Valid additional-region severity is height dependent:
    - tree height < 2 m: K = 0
    - 2 m <= height <= 15 m: K in {1, 2, 3, 4}
    - height > 15 m: K in {5, 6, 7}
* Additional-region placement classes are sampled with probabilities
  0.40 axis-proximal, 0.40 off-axis crown, 0.20 unconstrained.
* Axis-proximal region centres lie within 25% of the maximum radial extent
  from the same coarse stem axis used by the stem-axis-distance feature.
* Off-axis crown centres lie beyond 50% of the maximum radial extent and have
  z_norm > 0.35.
* For training augmentation, two independent tree-level variants are produced.
  Each of the two horizontal bands is independently active with p = 0.8.
* For controlled evaluation, both horizontal bands are always active.
* Scenario parameters can be generated once, serialized, and reused across
  methods so every method sees the same occlusion mask.

Reproducibility boundary
------------------------
The manuscript does NOT publish numerical distributions for horizontal-band
thickness, primitive dimensions, primitive-type probabilities, centre sampling
within an allowed placement class, or orientation distributions. Those values
are therefore *not guessed* in this module. They are supplied explicitly via
``OcclusionGeometryConfig``. The paper-level constants above are fixed and
validated by the code.

The phrase "vertically elongated regions" is also not given an exact analytic
shape in the manuscript. Here it is represented explicitly as a vertical
ellipsoid; its dimensions remain caller supplied. This operationalisation is
identified in metadata as ``vertical_ellipsoid`` rather than silently claimed
as a paper-specified formula.

Occlusion masks are applied to the complete tree before any occluded-tree input
representation is built. The caller must then rerun the repository preprocessing
pipeline on the surviving points, as required by Section 2.7, so geometric
features and the fixed Euclidean k-NN graph are recomputed from visible points
only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import cos, pi, sin
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray


EPS = 1e-12

EVALUATION_BAND_COUNT = 2
TRAINING_BAND_PROBABILITY = 0.8
AXIS_PROXIMAL_PROBABILITY = 0.40
OFF_AXIS_CROWN_PROBABILITY = 0.40
UNCONSTRAINED_PROBABILITY = 0.20
AXIS_PROXIMAL_MAX_RADIAL_FRACTION = 0.25
OFF_AXIS_CROWN_MIN_RADIAL_FRACTION = 0.50
OFF_AXIS_CROWN_MIN_Z_NORM = 0.35

PlacementClass = Literal["axis_proximal", "off_axis_crown", "unconstrained"]
PrimitiveKind = Literal["ellipsoid", "box", "cylinder", "vertical_ellipsoid"]
ScenarioMode = Literal["evaluation", "training"]


@dataclass(frozen=True)
class Range:
    """Closed numeric range used only for manuscript-unspecified geometry."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("Range endpoints must be finite.")
        if self.low <= 0.0 or self.high < self.low:
            raise ValueError("Range must satisfy 0 < low <= high.")

    def sample(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return float(self.low)
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class AngleRange:
    """Finite angular range in radians for manuscript-unspecified orientation."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("AngleRange endpoints must be finite.")
        if self.high < self.low:
            raise ValueError("AngleRange must satisfy low <= high.")

    def sample(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return float(self.low)
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class OcclusionGeometryConfig:
    """Explicit geometry choices omitted from the manuscript.

    All horizontal sizes are fractions of the tree's maximum radial extent.
    All vertical sizes are fractions of tree height. No defaults are provided,
    because the paper does not report these distributions.
    """

    band_thickness_z: Range
    ellipsoid_radius_x: Range
    ellipsoid_radius_y: Range
    ellipsoid_radius_z: Range
    box_half_x: Range
    box_half_y: Range
    box_half_z: Range
    cylinder_radius: Range
    cylinder_half_height_z: Range
    vertical_radius_x: Range
    vertical_radius_y: Range
    vertical_radius_z: Range
    yaw_radians: AngleRange
    pitch_radians: AngleRange
    roll_radians: AngleRange
    primitive_probabilities: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        probs = np.asarray(self.primitive_probabilities, dtype=np.float64)
        if probs.shape != (4,) or not np.isfinite(probs).all() or np.any(probs < 0.0):
            raise ValueError("primitive_probabilities must contain four finite non-negative values.")
        if not np.isclose(float(probs.sum()), 1.0, atol=1e-12):
            raise ValueError("primitive_probabilities must sum to 1.")
        if self.vertical_radius_z.low <= max(
            self.vertical_radius_x.high, self.vertical_radius_y.high
        ):
            raise ValueError(
                "vertical_ellipsoid must be vertically elongated for every sampled size: "
                "vertical_radius_z.low must exceed both horizontal maxima."
            )


@dataclass(frozen=True)
class TreeOcclusionFrame:
    """Tree-level frame and coarse stem-axis information used for placement."""

    z_min: float
    z_max: float
    height: float
    axis_point: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    max_radial_extent: float


@dataclass(frozen=True)
class OcclusionRegion:
    """One serializable occlusion region in normalized tree-local coordinates."""

    kind: str
    placement: str
    center_x: float
    center_y: float
    center_z_norm: float
    size_x: float
    size_y: float
    size_z: float
    yaw_radians: float
    pitch_radians: float
    roll_radians: float
    active: bool = True


@dataclass(frozen=True)
class OcclusionScenario:
    """A reusable tree-level occlusion scenario."""

    mode: ScenarioMode
    tree_height_m: float
    k_additional: int
    regions: tuple[OcclusionRegion, ...]
    seed: int | None = None

    @property
    def active_regions(self) -> tuple[OcclusionRegion, ...]:
        return tuple(region for region in self.regions if region.active)

    def to_json_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> "OcclusionScenario":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        regions = tuple(OcclusionRegion(**item) for item in payload["regions"])
        return cls(
            mode=payload["mode"],
            tree_height_m=float(payload["tree_height_m"]),
            k_additional=int(payload["k_additional"]),
            regions=regions,
            seed=payload.get("seed"),
        )


@dataclass(frozen=True)
class OcclusionApplication:
    """Boolean masks resulting from one reusable scenario."""

    keep_mask: NDArray[np.bool_]
    removed_mask: NDArray[np.bool_]
    removed_by_region: tuple[NDArray[np.bool_], ...]

    @property
    def num_removed(self) -> int:
        return int(self.removed_mask.sum())

    @property
    def num_kept(self) -> int:
        return int(self.keep_mask.sum())


def _validate_points(points: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("points must contain at least one point.")
    if not np.isfinite(array).all():
        raise ValueError("points contain NaN or infinite values.")
    return array


def _normalise_direction(direction: Sequence[float]) -> NDArray[np.float64]:
    vector = np.asarray(direction, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("axis_direction must be finite.")
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("axis_direction must be non-zero.")
    vector = vector / norm
    if vector[2] < 0.0:
        vector = -vector
    return vector


def build_tree_frame(
    points: NDArray[np.floating],
    axis_point: Sequence[float],
    axis_direction: Sequence[float],
) -> TreeOcclusionFrame:
    """Build the paper's tree-local height/radial frame from a complete tree."""
    xyz = _validate_points(points)
    p0 = np.asarray(axis_point, dtype=np.float64).reshape(3)
    if not np.isfinite(p0).all():
        raise ValueError("axis_point must be finite.")
    direction = _normalise_direction(axis_direction)

    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())
    height = z_max - z_min
    if height <= EPS:
        raise ValueError("Tree height must be positive.")

    displacement = xyz - p0
    projection = (displacement @ direction)[:, None] * direction[None, :]
    radial = np.linalg.norm(displacement - projection, axis=1)
    max_radial = float(radial.max())
    if max_radial <= EPS:
        raise ValueError("Maximum radial extent must be positive.")

    return TreeOcclusionFrame(
        z_min=z_min,
        z_max=z_max,
        height=height,
        axis_point=tuple(float(v) for v in p0),
        axis_direction=tuple(float(v) for v in direction),
        max_radial_extent=max_radial,
    )


def valid_k_values(tree_height_m: float) -> tuple[int, ...]:
    """Return the manuscript-allowed additional-region severities for a tree."""
    height = float(tree_height_m)
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("tree_height_m must be finite and positive.")
    if height < 2.0:
        return (0,)
    if height <= 15.0:
        return (1, 2, 3, 4)
    return (5, 6, 7)


def validate_k_for_height(tree_height_m: float, k_additional: int) -> int:
    k = int(k_additional)
    allowed = valid_k_values(tree_height_m)
    if k not in allowed:
        raise ValueError(
            f"K={k} is not allowed for tree height {tree_height_m:.6g} m; "
            f"paper-allowed values are {allowed}."
        )
    return k


def sample_placement_classes(
    k_additional: int,
    rng: np.random.Generator,
) -> tuple[PlacementClass, ...]:
    """Sample the manuscript's 40% / 40% / 20% placement distribution."""
    k = int(k_additional)
    if not 0 <= k <= 7:
        raise ValueError("k_additional must lie in [0, 7].")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator.")
    values = np.asarray(
        ["axis_proximal", "off_axis_crown", "unconstrained"],
        dtype=object,
    )
    probs = np.asarray(
        [
            AXIS_PROXIMAL_PROBABILITY,
            OFF_AXIS_CROWN_PROBABILITY,
            UNCONSTRAINED_PROBABILITY,
        ],
        dtype=np.float64,
    )
    sampled = rng.choice(values, size=k, replace=True, p=probs)
    return tuple(str(v) for v in sampled)  # type: ignore[return-value]


def _sample_center_normalized(
    placement: PlacementClass,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Sample a centre inside the *published* admissible placement zone.

    The paper specifies the admissible zones but not a within-zone density.
    This helper uses explicit uniform draws in polar radius, azimuth and
    relative height. It is an implementation choice, not a paper-reported
    distribution; keeping it isolated makes replacement straightforward.
    """
    theta = float(rng.uniform(0.0, 2.0 * pi))
    if placement == "axis_proximal":
        radius = float(rng.uniform(0.0, AXIS_PROXIMAL_MAX_RADIAL_FRACTION))
        z_norm = float(rng.uniform(0.0, 1.0))
    elif placement == "off_axis_crown":
        # Strictly greater than 0.50 and 0.35; np.nextafter prevents exact boundary.
        r0 = np.nextafter(OFF_AXIS_CROWN_MIN_RADIAL_FRACTION, 1.0)
        z0 = np.nextafter(OFF_AXIS_CROWN_MIN_Z_NORM, 1.0)
        radius = float(rng.uniform(r0, 1.0))
        z_norm = float(rng.uniform(z0, 1.0))
    elif placement == "unconstrained":
        radius = float(rng.uniform(0.0, 1.0))
        z_norm = float(rng.uniform(0.0, 1.0))
    else:
        raise ValueError(f"Unknown placement class: {placement}")

    return radius * cos(theta), radius * sin(theta), z_norm


def _sample_primitive_kind(
    geometry: OcclusionGeometryConfig,
    rng: np.random.Generator,
) -> PrimitiveKind:
    kinds = np.asarray(
        ["ellipsoid", "box", "cylinder", "vertical_ellipsoid"],
        dtype=object,
    )
    kind = rng.choice(kinds, p=np.asarray(geometry.primitive_probabilities))
    return str(kind)  # type: ignore[return-value]


def _sample_additional_region(
    placement: PlacementClass,
    geometry: OcclusionGeometryConfig,
    rng: np.random.Generator,
) -> OcclusionRegion:
    cx, cy, cz = _sample_center_normalized(placement, rng)
    kind = _sample_primitive_kind(geometry, rng)
    yaw = geometry.yaw_radians.sample(rng)
    pitch = geometry.pitch_radians.sample(rng)
    roll = geometry.roll_radians.sample(rng)

    if kind == "ellipsoid":
        sx = geometry.ellipsoid_radius_x.sample(rng)
        sy = geometry.ellipsoid_radius_y.sample(rng)
        sz = geometry.ellipsoid_radius_z.sample(rng)
    elif kind == "box":
        sx = geometry.box_half_x.sample(rng)
        sy = geometry.box_half_y.sample(rng)
        sz = geometry.box_half_z.sample(rng)
    elif kind == "cylinder":
        radius = geometry.cylinder_radius.sample(rng)
        sx = radius
        sy = radius
        sz = geometry.cylinder_half_height_z.sample(rng)
    elif kind == "vertical_ellipsoid":
        sx = geometry.vertical_radius_x.sample(rng)
        sy = geometry.vertical_radius_y.sample(rng)
        sz = geometry.vertical_radius_z.sample(rng)
    else:  # defensive
        raise RuntimeError(f"Unhandled primitive kind: {kind}")

    return OcclusionRegion(
        kind=kind,
        placement=placement,
        center_x=cx,
        center_y=cy,
        center_z_norm=cz,
        size_x=sx,
        size_y=sy,
        size_z=sz,
        yaw_radians=yaw,
        pitch_radians=pitch,
        roll_radians=roll,
        active=True,
    )


def generate_scenario(
    frame: TreeOcclusionFrame,
    k_additional: int,
    geometry: OcclusionGeometryConfig,
    *,
    mode: ScenarioMode,
    seed: int | None = None,
) -> OcclusionScenario:
    """Generate one reusable scenario consistent with Sections 2.6--2.7."""
    k = validate_k_for_height(frame.height, k_additional)
    if mode not in ("evaluation", "training"):
        raise ValueError("mode must be 'evaluation' or 'training'.")
    rng = np.random.default_rng(seed)

    regions: list[OcclusionRegion] = []
    for _ in range(EVALUATION_BAND_COUNT):
        active = True if mode == "evaluation" else bool(
            rng.random() < TRAINING_BAND_PROBABILITY
        )
        regions.append(
            OcclusionRegion(
                kind="horizontal_band",
                placement="stem_intersecting",
                center_x=0.0,
                center_y=0.0,
                center_z_norm=float(rng.uniform(0.0, 1.0)),
                size_x=1.0,
                size_y=1.0,
                size_z=geometry.band_thickness_z.sample(rng) / 2.0,
                yaw_radians=0.0,
                pitch_radians=0.0,
                roll_radians=0.0,
                active=active,
            )
        )

    placements = sample_placement_classes(k, rng)
    regions.extend(
        _sample_additional_region(placement, geometry, rng)
        for placement in placements
    )

    scenario = OcclusionScenario(
        mode=mode,
        tree_height_m=float(frame.height),
        k_additional=k,
        regions=tuple(regions),
        seed=seed,
    )
    validate_scenario(scenario)
    return scenario


def generate_two_training_variants(
    frame: TreeOcclusionFrame,
    k_additional: int,
    geometry: OcclusionGeometryConfig,
    *,
    seed: int | None = None,
) -> tuple[OcclusionScenario, OcclusionScenario]:
    """Create the two distinct tree-level variants stated in Section 2.7."""
    master = np.random.default_rng(seed)
    child_seeds = master.integers(0, np.iinfo(np.int64).max, size=2, dtype=np.int64)
    return (
        generate_scenario(
            frame,
            k_additional,
            geometry,
            mode="training",
            seed=int(child_seeds[0]),
        ),
        generate_scenario(
            frame,
            k_additional,
            geometry,
            mode="training",
            seed=int(child_seeds[1]),
        ),
    )


def validate_scenario(scenario: OcclusionScenario) -> None:
    """Validate every manuscript-explicit scenario constraint."""
    validate_k_for_height(scenario.tree_height_m, scenario.k_additional)
    if scenario.mode not in ("evaluation", "training"):
        raise ValueError("Invalid scenario mode.")
    if len(scenario.regions) != EVALUATION_BAND_COUNT + scenario.k_additional:
        raise ValueError("Scenario must contain exactly two bands plus K additional regions.")

    bands = scenario.regions[:EVALUATION_BAND_COUNT]
    if any(region.kind != "horizontal_band" for region in bands):
        raise ValueError("The first two scenario regions must be horizontal bands.")
    if scenario.mode == "evaluation" and not all(region.active for region in bands):
        raise ValueError("Both horizontal bands must be active during evaluation.")

    for region in scenario.regions:
        if not 0.0 <= region.center_z_norm <= 1.0:
            raise ValueError("Region center_z_norm must lie in [0, 1].")
        if region.size_z <= 0.0:
            raise ValueError("Region vertical size must be positive.")

    for region in scenario.regions[EVALUATION_BAND_COUNT:]:
        radial = float(np.hypot(region.center_x, region.center_y))
        if region.placement == "axis_proximal":
            if radial > AXIS_PROXIMAL_MAX_RADIAL_FRACTION + 1e-12:
                raise ValueError("Axis-proximal centre exceeds 25% maximum radial extent.")
        elif region.placement == "off_axis_crown":
            if radial <= OFF_AXIS_CROWN_MIN_RADIAL_FRACTION:
                raise ValueError("Off-axis crown centre must exceed 50% radial extent.")
            if region.center_z_norm <= OFF_AXIS_CROWN_MIN_Z_NORM:
                raise ValueError("Off-axis crown centre must have z_norm > 0.35.")
        elif region.placement == "unconstrained":
            pass
        else:
            raise ValueError(f"Unknown additional-region placement: {region.placement}")


def _points_in_frame(
    points: NDArray[np.float64],
    frame: TreeOcclusionFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return axis-centred planar coordinates normalized by max radial extent."""
    p0 = np.asarray(frame.axis_point, dtype=np.float64)
    direction = _normalise_direction(frame.axis_direction)

    # Construct a deterministic orthonormal plane basis perpendicular to axis.
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(direction @ reference)) > 0.95:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = np.cross(direction, reference)
    e1 /= np.linalg.norm(e1) + EPS
    e2 = np.cross(direction, e1)
    e2 /= np.linalg.norm(e2) + EPS

    displacement = points - p0
    x_local = (displacement @ e1) / frame.max_radial_extent
    y_local = (displacement @ e2) / frame.max_radial_extent
    z_norm = (points[:, 2] - frame.z_min) / frame.height
    return x_local, y_local, z_norm


def _mask_region_normalized(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    region: OcclusionRegion,
) -> NDArray[np.bool_]:
    if not region.active:
        return np.zeros(x.shape[0], dtype=bool)

    if region.kind == "horizontal_band":
        return np.abs(z - region.center_z_norm) <= region.size_z

    dx = x - region.center_x
    dy = y - region.center_y
    dz = z - region.center_z_norm

    # Inverse Z-Y-X Euler rotation: transform tree-frame offsets into the
    # primitive's local coordinate system. Orientation distributions are
    # explicitly caller supplied because the manuscript does not publish them.
    cy, sy = cos(region.yaw_radians), sin(region.yaw_radians)
    cp, sp = cos(region.pitch_radians), sin(region.pitch_radians)
    cr, sr = cos(region.roll_radians), sin(region.roll_radians)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    offsets = np.column_stack((dx, dy, dz))
    local = offsets @ rotation
    xr, yr, zr = local[:, 0], local[:, 1], local[:, 2]

    if region.kind in ("ellipsoid", "vertical_ellipsoid"):
        value = (
            (xr / region.size_x) ** 2
            + (yr / region.size_y) ** 2
            + (zr / region.size_z) ** 2
        )
        return value <= 1.0

    if region.kind == "box":
        return (
            (np.abs(xr) <= region.size_x)
            & (np.abs(yr) <= region.size_y)
            & (np.abs(zr) <= region.size_z)
        )

    if region.kind == "cylinder":
        return (
            (xr * xr + yr * yr <= region.size_x * region.size_x)
            & (np.abs(zr) <= region.size_z)
        )

    raise ValueError(f"Unknown occlusion kind: {region.kind}")


def apply_scenario(
    points: NDArray[np.floating],
    frame: TreeOcclusionFrame,
    scenario: OcclusionScenario,
) -> OcclusionApplication:
    """Apply a stored scenario and return exact keep/remove masks."""
    xyz = _validate_points(points)
    validate_scenario(scenario)
    if not np.isclose(scenario.tree_height_m, frame.height, rtol=1e-9, atol=1e-9):
        raise ValueError("Scenario tree height does not match the provided tree frame.")

    x, y, z = _points_in_frame(xyz, frame)
    per_region: list[NDArray[np.bool_]] = []
    removed = np.zeros(xyz.shape[0], dtype=bool)
    for region in scenario.regions:
        mask = _mask_region_normalized(x, y, z, region)
        per_region.append(mask)
        removed |= mask

    keep = ~removed
    return OcclusionApplication(
        keep_mask=keep,
        removed_mask=removed,
        removed_by_region=tuple(per_region),
    )


def remove_occluded_points(
    array: NDArray,
    application: OcclusionApplication,
) -> NDArray:
    """Filter any point-aligned array while preserving original row order."""
    values = np.asarray(array)
    if values.ndim == 0 or values.shape[0] != application.keep_mask.shape[0]:
        raise ValueError("array first dimension must match the occlusion mask length.")
    return values[application.keep_mask]


def paper_protocol_summary() -> dict[str, object]:
    """Machine-readable summary of manuscript-explicit constants."""
    return {
        "evaluation_horizontal_bands": EVALUATION_BAND_COUNT,
        "training_horizontal_band_probability": TRAINING_BAND_PROBABILITY,
        "k_range": [0, 7],
        "placement_probabilities": {
            "axis_proximal": AXIS_PROXIMAL_PROBABILITY,
            "off_axis_crown": OFF_AXIS_CROWN_PROBABILITY,
            "unconstrained": UNCONSTRAINED_PROBABILITY,
        },
        "axis_proximal_max_radial_fraction": AXIS_PROXIMAL_MAX_RADIAL_FRACTION,
        "off_axis_crown_min_radial_fraction": OFF_AXIS_CROWN_MIN_RADIAL_FRACTION,
        "off_axis_crown_min_z_norm": OFF_AXIS_CROWN_MIN_Z_NORM,
        "training_variants_per_tree": 2,
        "clean_occluded_subgraph_mixture": [0.5, 0.5],
        "recompute_features_and_knn_after_point_removal": True,
    }


__all__ = [
    "EPS",
    "EVALUATION_BAND_COUNT",
    "TRAINING_BAND_PROBABILITY",
    "AXIS_PROXIMAL_PROBABILITY",
    "OFF_AXIS_CROWN_PROBABILITY",
    "UNCONSTRAINED_PROBABILITY",
    "AXIS_PROXIMAL_MAX_RADIAL_FRACTION",
    "OFF_AXIS_CROWN_MIN_RADIAL_FRACTION",
    "OFF_AXIS_CROWN_MIN_Z_NORM",
    "Range",
    "AngleRange",
    "OcclusionGeometryConfig",
    "TreeOcclusionFrame",
    "OcclusionRegion",
    "OcclusionScenario",
    "OcclusionApplication",
    "build_tree_frame",
    "valid_k_values",
    "validate_k_for_height",
    "sample_placement_classes",
    "generate_scenario",
    "generate_two_training_variants",
    "validate_scenario",
    "apply_scenario",
    "remove_occluded_points",
    "paper_protocol_summary",
]
