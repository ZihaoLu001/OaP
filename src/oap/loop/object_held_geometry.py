"""Task-independent geometry for attributing a gripper obstruction.

``ObjectHeld`` is a measured real-robot predicate.  Gripper obstruction says
that *something* prevented an empty close; this module supplies the independent
spatial attribution that the currently tracked reconstructed subject occupies
the physical volume between the two GN01 pads.  It contains no task names,
stage names, grasp poses, or tuned metric/angle thresholds.

The helpers are pure NumPy geometry.  Runtime adapters may construct the three
oriented boxes from measured robot/gripper FK and the integrity-bound twin, but
no simulator contact state or contact history enters this calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ClosingRegionIntersection",
    "OrientedBox",
    "closing_region_intersection",
    "oriented_boxes_intersect",
]


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be one finite 3-vector")
    return vector.copy()


def _proper_rotation(value: object, *, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be one finite 3x3 rotation")
    # This tolerance only distinguishes a floating-point rotation matrix from
    # malformed geometry.  It is derived from machine precision and is not a
    # physical distance/angle acceptance threshold.
    roundoff = float(np.sqrt(np.finfo(float).eps))
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=roundoff,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        rtol=0.0,
        atol=roundoff,
    ):
        raise ValueError(f"{name} is not a proper orthonormal rotation")
    return rotation.copy()


@dataclass(frozen=True)
class OrientedBox:
    """A closed OBB: world centre, local axes as columns, and half extents."""

    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray

    @classmethod
    def validated(
        cls,
        *,
        center: object,
        axes: object,
        half_extents: object,
        name: str,
    ) -> "OrientedBox":
        half = _finite_vector(half_extents, name=f"{name} half extents")
        if np.any(half <= 0.0):
            raise ValueError(f"{name} half extents must be strictly positive")
        return cls(
            center=_finite_vector(center, name=f"{name} center"),
            axes=_proper_rotation(axes, name=f"{name} axes"),
            half_extents=half,
        )

    def projected_interval(self, axis: np.ndarray) -> tuple[float, float]:
        """Exact support interval of this OBB along one unit world axis."""
        direction = _finite_vector(axis, name="projection axis")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("projection axis must be nonzero")
        direction /= norm
        midpoint = float(direction @ self.center)
        radius = float(np.abs(direction @ self.axes) @ self.half_extents)
        return midpoint - radius, midpoint + radius


def oriented_boxes_intersect(first: OrientedBox, second: OrientedBox) -> bool:
    """Closed-set OBB intersection via the complete 15-axis SAT.

    Touching is intersection.  ``nextafter`` only admits one floating-point
    representation step at a computed boundary; it is not a sensor tolerance.
    """
    candidate_axes = [
        *(first.axes[:, index] for index in range(3)),
        *(second.axes[:, index] for index in range(3)),
    ]
    candidate_axes.extend(np.cross(first.axes[:, i], second.axes[:, j]) for i in range(3) for j in range(3))
    roundoff = float(np.sqrt(np.finfo(float).eps))
    delta = second.center - first.center
    for raw_axis in candidate_axes:
        norm = float(np.linalg.norm(raw_axis))
        if norm <= roundoff:
            # Parallel box axes yield a zero cross-product and therefore no
            # additional separating direction.
            continue
        axis = raw_axis / norm
        center_distance = abs(float(axis @ delta))
        first_radius = float(np.abs(axis @ first.axes) @ first.half_extents)
        second_radius = float(np.abs(axis @ second.axes) @ second.half_extents)
        touching_limit = np.nextafter(
            first_radius + second_radius,
            np.inf,
        )
        if center_distance > touching_limit:
            return False
    return True


@dataclass(frozen=True)
class ClosingRegionIntersection:
    """The reconstructed GN01 closing OBB and its subject intersection."""

    intersects: bool
    region: OrientedBox

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-safe geometric evidence."""
        return {
            "intersects": bool(self.intersects),
            "closing_region_center": self.region.center.tolist(),
            "closing_region_axes": self.region.axes.tolist(),
            "closing_region_half_extents_m": (self.region.half_extents.tolist()),
        }


def closing_region_intersection(
    *,
    subject: OrientedBox,
    left_pad: OrientedBox,
    right_pad: OrientedBox,
    gripper_axes: object,
) -> ClosingRegionIntersection:
    """Test whether ``subject`` intersects the current physical pad gap.

    The closing region is reconstructed in the measured gripper FK frame:

    * along gripper Y (the GN01 closing axis), it spans exactly between the two
      pad inner support faces;
    * along gripper X/Z, it is the overlap of the two real pad collision-box
      projections, i.e. the physical face area shared by both fingers.

    A missing, inverted, zero-volume, or otherwise ambiguous pad gap raises
    ``ValueError`` so the runtime can report UNKNOWN rather than inventing a
    region.  No object dimensions, pose, or task text influence region size.
    """
    basis = _proper_rotation(gripper_axes, name="gripper axes")
    pad_intervals = [[pad.projected_interval(basis[:, axis]) for axis in range(3)] for pad in (left_pad, right_pad)]
    closing_axis = 1
    left_coordinate = float(basis[:, closing_axis] @ left_pad.center)
    right_coordinate = float(basis[:, closing_axis] @ right_pad.center)
    if left_coordinate == right_coordinate:
        raise ValueError("GN01 pad ordering is ambiguous on the closing axis")
    lower_index, upper_index = (0, 1) if left_coordinate < right_coordinate else (1, 0)

    lower: list[float] = []
    upper: list[float] = []
    for axis in range(3):
        if axis == closing_axis:
            # Highest point of the lower pad and lowest point of the upper pad
            # are the two physical inner faces.
            lo = float(pad_intervals[lower_index][axis][1])
            hi = float(pad_intervals[upper_index][axis][0])
        else:
            # Only the shared pad-face footprint can be obstructed by a body
            # lying between both fingers.
            lo = max(
                float(pad_intervals[0][axis][0]),
                float(pad_intervals[1][axis][0]),
            )
            hi = min(
                float(pad_intervals[0][axis][1]),
                float(pad_intervals[1][axis][1]),
            )
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(
                f"GN01 pad geometry has no unambiguous positive-volume closing region on gripper axis {axis}"
            )
        lower.append(lo)
        upper.append(hi)

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    local_center = 0.5 * (lower_array + upper_array)
    region = OrientedBox.validated(
        center=basis @ local_center,
        axes=basis,
        half_extents=0.5 * (upper_array - lower_array),
        name="GN01 closing region",
    )
    return ClosingRegionIntersection(
        intersects=oriented_boxes_intersect(subject, region),
        region=region,
    )
