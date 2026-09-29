"""Define the fixed, task-agnostic predicate basis of a TaskProgram.

The VLM composes these predicates into ``running`` and ``terminal`` costs.
Production GPU sampling evaluates the residuals directly; measured stage
termination reuses the terminal predicates on re-observed state.

Every predicate exposes:
  * raw_residual(anchors) -> float >= 0 (the unthresholded Table I residual)
  * residual(anchors) -> float >= 0   (0 == satisfied; the tolerance is baked in)
  * satisfied(anchors)                (residual <= eps)
  * is_geometric: bool                (False only for the NonGeometric sentinel,
    which routes an unsupported goal directly to CANNOT_VERIFY)
  * referenced_anchors()              (for the groundability gate)
  * to_dict()/from_dict()             (the VLM emits/consumes these as JSON)

This basis is written ONCE. A new task is a new *composition* of these
predicates produced by the VLM -- never a new predicate class authored per task.
Adding a predicate (e.g. a fluid-state sensor) WOULD be per-class code, which is
precisely why goals needing one are REFUSED (CANNOT_VERIFY), not silently hacked.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Any

import numpy as np

from .anchors import AnchorSet
from .geometry import (
    angle_between, point_line_distance, signed_plane_distance, unit,
)

_EPS = 1e-6

def _require_direction3(value: Any, label: str) -> np.ndarray:
    """Return a finite, non-degenerate literal direction vector.

    A zero direction does not define a projection, plane, or half-space.
    Reject it at the typed-program boundary instead of letting CPU and JAX
    silently normalize it to zero and assign meaningless costs.
    """
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite numeric length-3 vector"
        ) from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite numeric length-3 vector")
    if float(np.linalg.norm(vector)) <= _EPS:
        raise ValueError(f"{label} norm must be > {_EPS:g}")
    return vector


def relative_orientation_frame_failures(
    body: str,
    anchors: AnchorSet,
    *,
    frame_prefix: str = "",
) -> list[str]:
    """Validate the owned current triad and frozen episode-start triad.

    Six finite unit vectors are not enough: a signed material pose requires
    two orthonormal, right-handed frames, and the current axes must ride the
    same physical owner as ``body``.  Initial axes are immutable world data and
    therefore must not be attached to a moving owner.
    """
    errors: list[str] = []
    body_anchor = anchors.get(body)
    expected_owner = None
    if body_anchor is not None:
        expected_owner = body_anchor.attached_to
        if expected_owner is None and body_anchor.kind == "subject":
            # The canonical subject point is the owner itself; its attached
            # material axes correctly carry ``attached_to='subject'``.
            expected_owner = body
    if body_anchor is None:
        errors.append(f"body_anchor_missing:{body}")

    groups = {
        "current": [f"{body}_{frame_prefix}frame_{axis}" for axis in "xyz"],
        "initial": [f"{body}_initial_{frame_prefix}frame_{axis}" for axis in "xyz"],
    }
    for label, names in groups.items():
        columns: list[np.ndarray] = []
        for name in names:
            anchor = anchors.get(name)
            if anchor is None:
                errors.append(f"{label}_frame_anchor_missing:{name}")
                continue
            if anchor.axis is None:
                errors.append(f"{label}_frame_axis_missing:{name}")
                continue
            axis = np.asarray(anchor.axis, dtype=float)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                errors.append(f"{label}_frame_axis_nonfinite:{name}")
                continue
            columns.append(axis)
            if label == "current" and anchor.attached_to != expected_owner:
                errors.append(
                    f"current_frame_owner_mismatch:{name}:"
                    f"{anchor.attached_to!r}!={expected_owner!r}"
                )
            if label == "initial" and anchor.attached_to is not None:
                errors.append(
                    f"initial_frame_must_be_frozen:{name}:"
                    f"owner={anchor.attached_to!r}"
                )
            if label == "initial" and anchor.kind != "initial":
                errors.append(
                    f"initial_frame_kind_invalid:{name}:kind={anchor.kind!r}"
                )
            if label == "initial" and bool(anchor.dynamic):
                errors.append(f"initial_frame_must_be_static:{name}")
        if len(columns) != 3:
            continue
        rotation = np.column_stack(columns)
        gram_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        )
        determinant = float(np.linalg.det(rotation))
        if gram_error > 1e-4:
            errors.append(
                f"{label}_frame_not_orthonormal:max_error={gram_error:.6g}"
            )
        if determinant < 1.0 - 1e-4:
            errors.append(
                f"{label}_frame_not_right_handed:det={determinant:.6g}"
            )
    return errors


class Predicate:
    """Base class of the closed predicate basis."""

    type: str = "predicate"
    is_geometric: bool = True

    def residual(self, anchors: AnchorSet) -> float:
        """Return the non-negative constraint residual (0 == satisfied)."""
        raise NotImplementedError

    def satisfied(self, anchors: AnchorSet, eps: float = _EPS) -> bool:
        """Return True when the residual is within eps."""
        return self.residual(anchors) <= eps

    def raw_residual(self, anchors: AnchorSet) -> float:
        """Return the residual before a terminal tolerance is applied.

        ``residual`` retains the thresholded verifier API. Running costs must
        use this method, including when a declared tolerance is zero. Exact
        contact terms are evaluated by the contact-aware rollout adapter, not
        inferred from sparse pose anchors.
        """
        for name in ("tol", "tol_deg"):
            if hasattr(self, name):
                return replace(self, **{name: 0.0}).residual(anchors)
        return self.residual(anchors)

    def referenced_anchors(self) -> list[str]:
        """Return the anchor ids this predicate dereferences."""
        return []

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-object form the VLM emits/consumes."""
        raise NotImplementedError

    def validate(self) -> None:
        """Validate constructor-level invariants before CPU/GPU evaluation."""

    @staticmethod
    def allowed_fields(predicate_type: str) -> frozenset[str] | None:
        """Return the exact JSON fields for a registered predicate type."""
        predicate_class = _REGISTRY.get(predicate_type)
        if predicate_class is None:
            return None
        return frozenset(
            {"type", *(field.name for field in fields(predicate_class))}
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Predicate":
        """Deserialize any registered predicate from its JSON-object form."""
        if not isinstance(d, dict):
            raise ValueError("predicate JSON must be an object")
        t = d.get("type")
        if not isinstance(t, str) or t not in _REGISTRY:
            raise ValueError(f"unknown predicate type {t!r}")
        allowed = Predicate.allowed_fields(t)
        assert allowed is not None
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(
                f"{t} has unknown fields {unknown}; "
                f"allowed fields are {sorted(allowed)}"
            )
        return _REGISTRY[t]._from(d)

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "Predicate":
        raise NotImplementedError


@dataclass
class PointPointDistance(Predicate):
    """||p(a) - p(b)|| within `dist` +/- `tol`. dist=0 => coincidence (on/in/near)."""
    type = "point_point_distance"
    a: str = ""
    b: str = ""
    dist: float = 0.0
    tol: float = 0.0

    def residual(self, anchors: AnchorSet) -> float:
        d = float(np.linalg.norm(anchors.point(self.a) - anchors.point(self.b)))
        return max(0.0, abs(d - self.dist) - self.tol)

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "dist": self.dist, "tol": self.tol}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "PointPointDistance":
        return cls(a=d["a"], b=d["b"], dist=float(d.get("dist", 0.0)), tol=float(d.get("tol", 0.0)))


@dataclass
class CaptureDistance(Predicate):
    """Generic dense point-distance grasp progress to a subject anchor.

    The implemented residual is the Euclidean distance from the measured
    ``gripper_center`` site to ``subject`` (normally an object-center anchor).
    It is not a closing-region-to-reconstructed-mesh distance. It encodes no
    object-specific grasp point, pose, orientation, approach direction, jaw
    command, or contact schedule. Exact measured :class:`ObjectHeld` evidence
    remains the grasp terminal condition.
    """

    type = "capture_distance"
    subject: str = ""

    def residual(self, anchors: AnchorSet) -> float:
        return float(
            np.linalg.norm(
                anchors.point("gripper_center")
                - anchors.point(self.subject)
            )
        )

    def referenced_anchors(self) -> list[str]:
        return ["gripper_center", self.subject]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "subject": self.subject}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "CaptureDistance":
        subject = d.get("subject")
        if not isinstance(subject, str) or not subject:
            raise ValueError(
                "capture_distance.subject must be a non-empty string"
            )
        return cls(subject=subject)


@dataclass
class RelativePosition(Predicate):
    """p(a) - p(b) matches a world-frame offset within Euclidean tolerance.

    This is the vector form of a displacement/relative-pose translation goal.
    Unlike combining a scalar distance with one projected-axis constraint, it
    also penalizes drift in the two orthogonal directions.  It is generic
    task vocabulary: the program supplies the anchors and desired vector.
    """
    type = "relative_position"
    a: str = ""
    b: str = ""
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tol: float = 0.01

    def residual(self, anchors: AnchorSet) -> float:
        error = (
            anchors.point(self.a)
            - anchors.point(self.b)
            - np.asarray(self.offset, dtype=float)
        )
        return max(0.0, float(np.linalg.norm(error)) - self.tol)

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "a": self.a,
            "b": self.b,
            "offset": list(self.offset),
            "tol": self.tol,
        }

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "RelativePosition":
        return cls(
            a=d["a"],
            b=d["b"],
            offset=tuple(d.get("offset", (0.0, 0.0, 0.0))),
            tol=float(d.get("tol", 0.01)),
        )


@dataclass
class MinDistance(Predicate):
    """||p(a) - p(b)|| >= clearance (obstacle avoidance / 'avoid')."""
    type = "min_distance"
    a: str = ""
    b: str = ""
    clearance: float = 0.0

    def residual(self, anchors: AnchorSet) -> float:
        d = float(np.linalg.norm(anchors.point(self.a) - anchors.point(self.b)))
        return max(0.0, self.clearance - d)

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "clearance": self.clearance}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "MinDistance":
        return cls(a=d["a"], b=d["b"], clearance=float(d.get("clearance", 0.0)))


@dataclass
class AboveBy(Predicate):
    """Signed displacement of p(a) over p(b) along `axis` equals `dist` +/- `tol`.
    Default axis = world +Z (a above b by dist)."""
    type = "above_by"
    a: str = ""
    b: str = ""
    dist: float = 0.0
    tol: float = 0.01
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_direction3(self.axis, f"{self.type}.axis")

    def residual(self, anchors: AnchorSet) -> float:
        n = unit(_require_direction3(self.axis, f"{self.type}.axis"))
        signed = float(np.dot(anchors.point(self.a) - anchors.point(self.b), n))
        return max(0.0, abs(signed - self.dist) - self.tol)

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "dist": self.dist, "tol": self.tol, "axis": list(self.axis)}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "AboveBy":
        return cls(a=d["a"], b=d["b"], dist=float(d.get("dist", 0.0)), tol=float(d.get("tol", 0.01)),
                   axis=tuple(d.get("axis", (0.0, 0.0, 1.0))))


@dataclass
class OnPlane(Predicate):
    """p(a) lies on the plane through p(b) with normal `axis` (world frame)."""
    type = "on_plane"
    a: str = ""
    b: str = ""
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tol: float = 0.01

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_direction3(self.axis, f"{self.type}.axis")

    def residual(self, anchors: AnchorSet) -> float:
        axis = _require_direction3(self.axis, f"{self.type}.axis")
        return max(
            0.0,
            abs(signed_plane_distance(
                anchors.point(self.a),
                anchors.point(self.b),
                axis,
            )) - self.tol,
        )

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "axis": list(self.axis), "tol": self.tol}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "OnPlane":
        return cls(a=d["a"], b=d["b"], axis=tuple(d.get("axis", (0.0, 0.0, 1.0))), tol=float(d.get("tol", 0.01)))


@dataclass
class PointOnLine(Predicate):
    """p(a) lies on the line through p(b) with direction = axis-anchor `dir_anchor`."""
    type = "point_on_line"
    a: str = ""
    b: str = ""
    dir_anchor: str = ""
    tol: float = 0.01

    def residual(self, anchors: AnchorSet) -> float:
        return max(0.0, point_line_distance(anchors.point(self.a), anchors.point(self.b),
                                            anchors.axis(self.dir_anchor)) - self.tol)

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b, self.dir_anchor]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "dir_anchor": self.dir_anchor, "tol": self.tol}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "PointOnLine":
        return cls(a=d["a"], b=d["b"], dir_anchor=d["dir_anchor"], tol=float(d.get("tol", 0.01)))


@dataclass
class AxisParallel(Predicate):
    """axis(a) parallel (or antiparallel) to axis(b) within tol_deg."""
    type = "axis_parallel"
    a: str = ""
    b: str = ""
    antiparallel: bool = False
    tol_deg: float = 10.0

    def residual(self, anchors: AnchorSet) -> float:
        va, vb = anchors.axis(self.a), anchors.axis(self.b)
        ang = angle_between(va, -vb if self.antiparallel else vb)
        return max(0.0, ang - math.radians(self.tol_deg))

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "antiparallel": self.antiparallel, "tol_deg": self.tol_deg}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "AxisParallel":
        return cls(a=d["a"], b=d["b"], antiparallel=bool(d.get("antiparallel", False)), tol_deg=float(d.get("tol_deg", 10.0)))


@dataclass
class AxisAngle(Predicate):
    """Angle between axis(a) and axis(b) equals theta_deg +/- tol_deg.
    Pour/tilt: a = subject (e.g. can spout) axis, b = world-up axis."""
    type = "axis_angle"
    a: str = ""
    b: str = ""
    theta_deg: float = 90.0
    tol_deg: float = 10.0

    def residual(self, anchors: AnchorSet) -> float:
        ang = math.degrees(angle_between(anchors.axis(self.a), anchors.axis(self.b)))
        return max(0.0, abs(ang - self.theta_deg) - self.tol_deg) * math.pi / 180.0

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "theta_deg": self.theta_deg, "tol_deg": self.tol_deg}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "AxisAngle":
        return cls(a=d["a"], b=d["b"], theta_deg=float(d.get("theta_deg", 90.0)), tol_deg=float(d.get("tol_deg", 10.0)))


@dataclass
class RelativeOrientation(Predicate):
    """Full signed SO(3) goal relative to one body's episode-start frame.

    ``target_axis`` is expressed in the body's *initial material frame* and
    ``target_angle_deg`` is signed by the right-hand rule.  The desired frame
    is therefore ``R_initial @ Exp(target_axis * target_angle)``.  The
    residual is the geodesic distance on SO(3), tolerance-banded in radians.

    A single principal-axis angle cannot distinguish many different 3-D
    orientations (and the conventional up-face normal is deliberately
    hemisphere-pinned).  This predicate instead reads the raw, directed
    ``{body}_frame_[xyz]`` material triad and its immutable
    ``{body}_initial_frame_[xyz]`` copy.  Those anchors are emitted only from
    a complete quaternion after ``observe_scene`` verifies a separately
    provisioned authority and mints a process-local token bound to the body,
    observation/capture/frame ids, schema, quaternion, calibration, and
    independent-validation digests. A packet assertion, raw quaternion, or
    legacy Boolean is insufficient; normal groundability then keeps measured
    stage status UNKNOWN rather than inventing orientation from yaw or an
    unsigned axis.
    """

    type = "relative_orientation"
    body: str = ""
    target_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    target_angle_deg: float = 0.0
    tol_deg: float = 10.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.body, str) or not self.body:
            raise ValueError(
                "relative_orientation.body must be a non-empty string"
            )
        _require_direction3(self.target_axis, f"{self.type}.target_axis")
        angle = float(self.target_angle_deg)
        tolerance = float(self.tol_deg)
        if not math.isfinite(angle) or not -180.0 <= angle <= 180.0:
            raise ValueError(
                f"{self.type}.target_angle_deg must be in [-180, 180]"
            )
        if (
            not math.isfinite(tolerance)
            or tolerance < 0.0
            or tolerance >= 180.0
        ):
            raise ValueError(f"{self.type}.tol_deg must be in [0, 180)")

    def current_frame_names(self) -> list[str]:
        return [f"{self.body}_frame_{axis}" for axis in "xyz"]

    def initial_frame_names(self) -> list[str]:
        return [f"{self.body}_initial_frame_{axis}" for axis in "xyz"]

    def frame_contract_failures(self, anchors: AnchorSet) -> list[str]:
        return relative_orientation_frame_failures(self.body, anchors)

    @staticmethod
    def _frame(anchors: AnchorSet, names: list[str]) -> np.ndarray | None:
        rotation = np.column_stack([anchors.axis(name) for name in names])
        if not np.all(np.isfinite(rotation)):
            return None
        gram_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if gram_error > 1e-4 or determinant < 1.0 - 1e-4:
            return None
        return rotation

    def residual(self, anchors: AnchorSet) -> float:
        self.validate()
        if self.frame_contract_failures(anchors):
            # Planning receives a maximal finite residual. Measured grading
            # detects the same failure before scalar evaluation and reports
            # UNKNOWN instead of treating malformed evidence as task failure.
            return math.pi
        initial = self._frame(anchors, self.initial_frame_names())
        current = self._frame(anchors, self.current_frame_names())
        if initial is None or current is None:
            # A malformed finite triad is not a pose measurement.  It must not
            # certify success; missing triads are routed to UNKNOWN before this
            # method by the measured groundability gate.
            return math.pi
        axis = unit(
            _require_direction3(
                self.target_axis,
                f"{self.type}.target_axis",
            )
        )
        angle = math.radians(float(self.target_angle_deg))
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=float,
        )
        delta = (
            np.eye(3)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        )
        desired = initial @ delta
        error = desired.T @ current
        cosine = float(np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0))
        geodesic = math.acos(cosine)
        return max(0.0, geodesic - math.radians(float(self.tol_deg)))

    def referenced_anchors(self) -> list[str]:
        return self.current_frame_names() + self.initial_frame_names()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "body": self.body,
            "target_axis": list(self.target_axis),
            "target_angle_deg": self.target_angle_deg,
            "tol_deg": self.tol_deg,
        }

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "RelativeOrientation":
        return cls(
            body=d["body"],
            target_axis=tuple(d.get("target_axis", (1.0, 0.0, 0.0))),
            target_angle_deg=float(d.get("target_angle_deg", 0.0)),
            tol_deg=float(d.get("tol_deg", 10.0)),
        )


@dataclass
class RotationProgress(Predicate):
    """At least min_angle_deg of signed relative rotation about initial axis.

    ``axis`` is expressed in the initial RAW body frame, matching recorded
    free-joint quaternions. It is not the mesh-to-canonical geometry frame.
    There is no symmetric target tolerance. Numerical slack is 1e-6 degree,
    not the legacy Predicate terminal epsilon of 1e-6 radians.
    """

    type = "rotation_progress"
    body: str = ""
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    min_angle_deg: float = 20.0
    scale_deg: float = 10.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.body, str) or not self.body:
            raise ValueError("rotation_progress.body must be a non-empty string")
        _require_direction3(self.axis, "rotation_progress.axis")
        if not math.isfinite(float(self.min_angle_deg)) or not 0.0 < self.min_angle_deg < 180.0:
            raise ValueError("rotation_progress.min_angle_deg must be in (0, 180)")
        if not math.isfinite(float(self.scale_deg)) or self.scale_deg <= 0.0:
            raise ValueError("rotation_progress.scale_deg must be finite and positive")

    def current_frame_names(self) -> list[str]:
        return [f"{self.body}_raw_frame_{a}" for a in "xyz"]

    def initial_frame_names(self) -> list[str]:
        return [f"{self.body}_initial_raw_frame_{a}" for a in "xyz"]

    def frame_contract_failures(self, anchors: AnchorSet) -> list[str]:
        return relative_orientation_frame_failures(self.body, anchors, frame_prefix="raw_")

    def angle_deg(self, anchors: AnchorSet) -> float:
        from .rotation_progress import rotation_progress_radians
        failures = self.frame_contract_failures(anchors)
        if failures:
            raise ValueError("rotation_progress frame contract: " + ";".join(failures))
        initial = np.column_stack([anchors.axis(n) for n in self.initial_frame_names()])
        current = np.column_stack([anchors.axis(n) for n in self.current_frame_names()])
        return math.degrees(float(rotation_progress_radians(initial, current, self.axis)))

    def residual(self, anchors: AnchorSet) -> float:
        if self.frame_contract_failures(anchors):
            return math.pi
        return math.radians(max(0.0, self.min_angle_deg - self.angle_deg(anchors)))

    def satisfied(self, anchors: AnchorSet, eps: float = _EPS) -> bool:
        del eps
        from .rotation_progress import ANGLE_NUMERICAL_EPS_DEG
        return not self.frame_contract_failures(anchors) and self.angle_deg(anchors) >= self.min_angle_deg - ANGLE_NUMERICAL_EPS_DEG

    def referenced_anchors(self) -> list[str]:
        return self.current_frame_names() + self.initial_frame_names()

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "body": self.body,
                "axis": list(self.axis), "min_angle_deg": self.min_angle_deg,
                "scale_deg": self.scale_deg}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "RotationProgress":
        return cls(body=d["body"], axis=tuple(d["axis"]),
                   min_angle_deg=float(d["min_angle_deg"]),
                   scale_deg=float(d.get("scale_deg", 10.0)))


@dataclass
class AngleAboutAxis(Predicate):
    """SIGNED rotation of axis(a) about hinge axis(b), measured from axis(ref).

    Articulation (lid/door/valve open-close): a = the moving part's axis NOW,
    ref = the same part's axis in the CLOSED reference pose, b = the hinge
    axis. Both a and ref are projected onto the plane perpendicular to b; the
    residual is the signed dihedral's distance from theta_deg (right-hand
    rule about b), tolerance-banded, in radians (commensurate with the meter
    residuals in the squared stage cost). AxisAngle cannot express this: the
    UNSIGNED angle between two axes is symmetric in the opening direction and
    cannot pin WHICH way past the reference the part has swung.
    """

    type = "angle_about_axis"
    a: str = ""
    b: str = ""
    ref: str = ""
    theta_deg: float = 90.0
    tol_deg: float = 10.0

    def residual(self, anchors: AnchorSet) -> float:
        b = unit(anchors.axis(self.b))
        va = np.asarray(anchors.axis(self.a), dtype=float)
        vr = np.asarray(anchors.axis(self.ref), dtype=float)
        pa = va - b * float(va @ b)
        pr = vr - b * float(vr @ b)
        # A part axis parallel to the hinge axis has no measurable opening
        # angle: the predicate is unsatisfiable (max residual), never a
        # silent zero -- the verifier must not certify what it cannot measure.
        if float(np.linalg.norm(pa)) < 1e-9 or float(np.linalg.norm(pr)) < 1e-9:
            return math.pi
        pa, pr = unit(pa), unit(pr)
        angle = math.atan2(float(np.cross(pr, pa) @ b), float(pr @ pa))
        target = math.radians(self.theta_deg)
        # Signed angles live on a circle.  Compare the shortest wrapped delta
        # so +179 and -179 degrees differ by two degrees, not 358.
        delta = math.atan2(
            math.sin(angle - target),
            math.cos(angle - target),
        )
        return max(0.0, abs(delta) - math.radians(self.tol_deg))

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b, self.ref]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "ref": self.ref,
                "theta_deg": self.theta_deg, "tol_deg": self.tol_deg}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "AngleAboutAxis":
        return cls(a=d["a"], b=d["b"], ref=d["ref"],
                   theta_deg=float(d.get("theta_deg", 90.0)),
                   tol_deg=float(d.get("tol_deg", 10.0)))


@dataclass
class InRegion(Predicate):
    """Normalized AABB boundary violation, as defined in Table I."""
    type = "in_region"
    a: str = ""
    region: str = ""

    def residual(self, anchors: AnchorSet) -> float:
        reg = anchors[self.region]
        if reg.region_half is None:
            raise ValueError(f"anchor {self.region!r} is not a region (no region_half)")
        half = np.asarray(reg.region_half, dtype=float)
        if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
            raise ValueError("in_region requires finite positive half-sizes")
        outside = np.maximum(np.abs(anchors.point(self.a) - reg.point) - half, 0.0)
        return float(np.max(outside / half))

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.region]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "region": self.region}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "InRegion":
        return cls(a=d["a"], region=d["region"])


@dataclass
class SignedAxisGap(Predicate):
    """sign * ((p(b) - p(a)) . axis) >= margin. left_of: axis=+x, sign=+1;
    right_of: axis=+x, sign=-1. Generalizes a side-of relation to any axis."""
    type = "signed_axis_gap"
    a: str = ""
    b: str = ""
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    margin: float = 0.0
    sign: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_direction3(self.axis, f"{self.type}.axis")
        if not math.isfinite(float(self.sign)) or float(self.sign) not in {
            -1.0,
            1.0,
        }:
            raise ValueError(f"{self.type}.sign must be -1 or +1")

    def residual(self, anchors: AnchorSet) -> float:
        n = unit(_require_direction3(self.axis, f"{self.type}.axis"))
        gap = self.sign * float(np.dot(anchors.point(self.b) - anchors.point(self.a), n))
        return max(0.0, self.margin - gap)

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "axis": list(self.axis), "margin": self.margin, "sign": self.sign}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "SignedAxisGap":
        return cls(a=d["a"], b=d["b"], axis=tuple(d.get("axis", (1.0, 0.0, 0.0))),
                   margin=float(d.get("margin", 0.0)), sign=float(d.get("sign", 1.0)))


@dataclass
class GripperState(Predicate):
    """Observable physical jaw width in metres.

    This predicate reads the measured/predicted joint state exposed as the
    runtime ``gripper_width`` anchor.  It is a terminal/verdict fact, not an
    actuator command.  ``GripperCommand`` is the distinct running-cost
    primitive for steering the sampled control target.
    """
    type = "gripper_state"
    state: str = "closed"          # 'open' | 'closed'; width_m overrides it
    width_m: float | None = None
    tol: float = 0.01

    def _target(self) -> float:
        if self.width_m is not None:
            return float(self.width_m)
        return 0.085 if self.state == "open" else 0.0

    def residual(self, anchors: AnchorSet) -> float:
        a = anchors.get("gripper_width")
        if a is None:
            return 0.0
        width = float(a.point[0])
        target = self._target()
        if self.width_m is not None:
            return max(0.0, abs(width - target) - self.tol)
        # ``open`` and ``closed`` are states, not aperture set-points. A jaw
        # that opens beyond the calibrated nominal remains open, and a jaw
        # that closes farther remains closed. An explicit ``width_m`` is still
        # an ordinary two-sided numeric target above.
        if self.state == "open":
            return max(0.0, target - self.tol - width)
        return max(0.0, width - (target + self.tol))

    def referenced_anchors(self) -> list[str]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "state": self.state, "width_m": self.width_m, "tol": self.tol}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "GripperState":
        w = d.get("width_m")
        return cls(state=d.get("state", "closed"), width_m=None if w is None else float(w), tol=float(d.get("tol", 0.01)))


@dataclass
class GripperCommand(Predicate):
    """The signed jaw-effort intent, not the measured aperture.

    In ``running`` the GPU scorer reads the candidate's control trajectory
    through the runtime ``gripper_command`` signal. In ``terminal`` it grades
    the last ACKNOWLEDGED command at the Stage boundary, so a Stage that must
    still be holding cannot exit on a transient measured hold whose command
    has already been released (measured 2026-08-07: one such exit sent an
    open command into the next Stage and dropped the object immediately).

    Kept distinct from ``GripperState`` throughout: an actuator target must
    never masquerade as measured physical closure when an object obstructs
    the fingers, and an obstructed aperture must never masquerade as a
    command.
    """

    type = "gripper_command"
    state: str = "closed"          # closed -> +1 effort, open -> -1 effort
    width_m: float | None = None
    tol: float = 0.0

    def _target(self) -> float:
        """Equivalent target opening width in metres, not measured aperture."""
        if self.width_m is not None:
            width = float(self.width_m)
            if not math.isfinite(width) or not 0.0 <= width <= 0.085:
                raise ValueError("gripper_command.width_m must be in [0, 0.085] metres")
            return width
        if self.state not in {"open", "closed"}:
            raise ValueError("gripper_command.state must be open or closed")
        return 0.085 if self.state == "open" else 0.0

    def residual(self, anchors: AnchorSet) -> float:
        a = anchors.get("gripper_command")
        if a is None:
            return 0.0
        width = 0.085 * 0.5 * (1.0 - float(np.clip(a.point[0], -1.0, 1.0)))
        return max(0.0, abs(width - self._target()) - self.tol)

    def raw_residual(self, anchors: AnchorSet) -> float:
        if anchors.missing(["gripper_command"]):
            raise ValueError("gripper_command requires a finite control-command signal")
        return replace(self, tol=0.0).residual(anchors)

    def referenced_anchors(self) -> list[str]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "state": self.state,
            "width_m": self.width_m,
            "tol": self.tol,
        }

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "GripperCommand":
        w = d.get("width_m")
        return cls(
            state=d.get("state", "closed"),
            width_m=None if w is None else float(w),
            tol=float(d.get("tol", 0.0)),
        )


@dataclass
class ObjectHeld(Predicate):
    """Require the object owning ``object_anchor`` to be held.

    ``ObjectHeld`` is the backend-independent program fact.  Exact rollout
    evidence is simultaneous left/right-pad contact with the canonical subject;
    real execution may establish the same fact from calibrated gripper
    obstruction and object--TCP rigidity.  Anchor-only interpreters cannot
    observe either evidence source, so the sampling/device and measured-verdict
    adapters supply exact Boolean evidence. In ``running`` its finite
    full-horizon not-held fraction ranks complete sampled trajectories. In
    ``terminal`` it contributes a finite planning residual at the full-horizon
    endpoint and remains a measured condition for stage advance. It creates no
    executable-prefix barrier. No contact-duration threshold, grasp pose,
    waypoint, or object-specific geometry is encoded here.
    """

    type = "object_held"
    object_anchor: str = ""

    def residual(self, anchors: AnchorSet) -> float:
        # Anchor-only interpreters cannot observe contact; the GPU cost adapter
        # supplies the exact Boolean terminal residual. The explicit anchor
        # reference still participates in groundability.
        return 0.0

    def raw_residual(self, anchors: AnchorSet, *, held: bool | None = None) -> float:
        """Table I: 1 minus simultaneous two-finger contact at this step."""
        del anchors
        if not isinstance(held, (bool, np.bool_)):
            raise ValueError("object_held requires exact Boolean held evidence")
        return float(not held)

    def referenced_anchors(self) -> list[str]:
        return [self.object_anchor]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "object_anchor": self.object_anchor}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "ObjectHeld":
        allowed = {"type", "object_anchor"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(
                f"object_held has unknown fields {unknown}; "
                f"allowed fields are {sorted(allowed)}"
            )
        anchor = d.get("object_anchor")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(
                "object_held.object_anchor must be a non-empty string"
            )
        return cls(object_anchor=anchor)


@dataclass
class TemporalHold(Predicate):
    """`inner` must hold for >= `frames` consecutive frames (enforced by the
    interpreter over a trajectory). Single-frame residual == inner residual."""
    type = "temporal_hold"
    inner: Predicate = None  # type: ignore[assignment]
    frames: int = 3

    @property
    def is_geometric(self) -> bool:  # type: ignore[override]
        return self.inner.is_geometric

    def residual(self, anchors: AnchorSet) -> float:
        return self.inner.residual(anchors)

    def referenced_anchors(self) -> list[str]:
        return self.inner.referenced_anchors()

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "inner": self.inner.to_dict(), "frames": self.frames}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "TemporalHold":
        return cls(inner=Predicate.from_dict(d["inner"]), frames=int(d.get("frames", 3)))


@dataclass
class InteractionAlign(Predicate):
    """COUPLED point-distance + axis-alignment between an ACTIVE and a PASSIVE
    anchor (OmniManip C={O_active,O_passive,d,theta}; subsumes CoPa collinear-and-
    opposite and MOKA function-point->target-point).

    residual = w_d * max(0, |‖p(a)-p(b)‖ - dist| - tol)
             + w_theta * max(0, |angle(axis_a, (±)axis_b) - theta_deg| - tol_deg)[rad]

    The coupled residual gives sampling one dense scalar for simultaneous
    translation and rotation without supplying a pose initializer. ``axis_a``
    and ``axis_b`` are anchor ids, or ``self`` to use ``a``/``b`` directly."""
    type = "interaction_align"
    a: str = ""
    b: str = ""
    dist: float = 0.0
    tol: float = 0.01
    axis_a: str = "self"
    axis_b: str = "self"
    theta_deg: float = 0.0
    tol_deg: float = 10.0
    antiparallel: bool = False
    w_d: float = 1.0
    w_theta: float = 1.0

    def _axis(self, which: str, ref: str, anchors: AnchorSet) -> np.ndarray:
        return anchors.axis(ref if which == "self" else which)

    def residual(self, anchors: AnchorSet) -> float:
        d = float(np.linalg.norm(anchors.point(self.a) - anchors.point(self.b)))
        d_res = max(0.0, abs(d - self.dist) - self.tol)
        va = self._axis(self.axis_a, self.a, anchors)
        vb = self._axis(self.axis_b, self.b, anchors)
        ang = math.degrees(angle_between(va, -vb if self.antiparallel else vb))
        ang_res = max(0.0, abs(ang - self.theta_deg) - self.tol_deg) * math.pi / 180.0
        return float(self.w_d * d_res + self.w_theta * ang_res)

    def referenced_anchors(self) -> list[str]:
        names = [self.a, self.b]
        if self.axis_a != "self":
            names.append(self.axis_a)
        if self.axis_b != "self":
            names.append(self.axis_b)
        return names

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b, "dist": self.dist, "tol": self.tol,
                "axis_a": self.axis_a, "axis_b": self.axis_b, "theta_deg": self.theta_deg,
                "tol_deg": self.tol_deg, "antiparallel": self.antiparallel,
                "w_d": self.w_d, "w_theta": self.w_theta}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "InteractionAlign":
        return cls(a=d["a"], b=d["b"], dist=float(d.get("dist", 0.0)), tol=float(d.get("tol", 0.01)),
                   axis_a=d.get("axis_a", "self"), axis_b=d.get("axis_b", "self"),
                   theta_deg=float(d.get("theta_deg", 0.0)), tol_deg=float(d.get("tol_deg", 10.0)),
                   antiparallel=bool(d.get("antiparallel", False)),
                   w_d=float(d.get("w_d", 1.0)), w_theta=float(d.get("w_theta", 1.0)))


@dataclass
class AlongAxisGap(Predicate):
    """Signed distance of p(a) ALONG a referenced PART's own axis from p(c), with a
    collinearity term keeping a on that axis line (CoPa Table-III 'point A a signed
    distance x ALONG vector B from C'). Unlike AboveBy/SignedAxisGap (which measure
    along a caller-supplied WORLD axis), b provides the axis -- screw-advance /
    tool-tip-reach / insert-along-the-vase-axis / spoon-into-cup-depth.

    residual = max(0, |((p(a)-p(c))·axis(b)) - dist| - tol) + w_perp*‖(p(a)-p(c)) × axis(b)‖"""
    type = "along_axis_gap"
    a: str = ""
    c: str = ""
    b: str = ""              # axis-provider anchor
    dist: float = 0.0
    tol: float = 0.01
    w_perp: float = 1.0

    def residual(self, anchors: AnchorSet) -> float:
        rel = anchors.point(self.a) - anchors.point(self.c)
        n = anchors.axis(self.b)
        along = float(np.dot(rel, n))
        perp = float(np.linalg.norm(np.cross(rel, n)))
        return max(0.0, abs(along - self.dist) - self.tol) + self.w_perp * perp

    def referenced_anchors(self) -> list[str]:
        return [self.a, self.c, self.b]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "c": self.c, "b": self.b,
                "dist": self.dist, "tol": self.tol, "w_perp": self.w_perp}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "AlongAxisGap":
        return cls(a=d["a"], c=d["c"], b=d["b"], dist=float(d.get("dist", 0.0)),
                   tol=float(d.get("tol", 0.01)), w_perp=float(d.get("w_perp", 1.0)))


@dataclass
class Contact(Predicate):
    """Finite running residual requiring actor-to-dynamic-target contact.

    The current rollout adapter accepts the controlled subject or ``robot`` as
    the actor and one dynamic body as target; it is not an arbitrary pairwise
    contact engine.
    """

    type = "contact"
    a: str = ""
    b: str = ""

    def residual(self, anchors: AnchorSet) -> float:
        # Sparse pose anchors cannot establish contact. The full-physics
        # planner supplies the exact existence residual.
        return 0.0

    def referenced_anchors(self) -> list[str]:
        return [name for name in (self.a, self.b) if name != "robot"]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "Contact":
        return cls(a=d["a"], b=d["b"])


@dataclass
class NoContact(Predicate):
    """Finite running residual discouraging actor-to-target contact.

    It has the same current actor/target and one-target-per-Stage restrictions
    as :class:`Contact`.
    """

    type = "no_contact"
    a: str = ""
    b: str = ""

    def residual(self, anchors: AnchorSet) -> float:
        return 0.0

    def referenced_anchors(self) -> list[str]:
        return [name for name in (self.a, self.b) if name != "robot"]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "a": self.a, "b": self.b}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "NoContact":
        return cls(a=d["a"], b=d["b"])


@dataclass
class RobotSubjectContact(Predicate):
    """Exact contact between the robot and the controlled subject.

    The two actor roles are structural and therefore have no user-provided
    anchor fields.  Planning reads an exact full-physics contact reduction;
    measured stage termination reads the dedicated current-contact channel.
    Sparse pose anchors deliberately cannot infer contact from proximity.
    """

    type = "robot_subject_contact"

    def residual(self, anchors: AnchorSet) -> float:
        measurement = anchors.get("robot_subject_contact")
        if measurement is None:
            # Planning uses the exact rollout reduction below the AnchorSet
            # interpreter.  Evidence grading separately fails closed when the
            # dedicated current-contact measurement is absent.
            return 0.0
        return 0.0 if float(measurement.point[0]) > 0.5 else 1.0

    def referenced_anchors(self) -> list[str]:
        # Make the relation structurally controllable/observable without
        # exposing configurable actor fields to JSON or the VLM.
        return ["subject"]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "RobotSubjectContact":
        del d
        return cls()


@dataclass
class ToolMediation(Predicate):
    """Running tool-use constraint for indirect action through ``tool``.

    The exact rollout backend evaluates two contact indicators at every
    predicted state. Robot-to-target contact contributes one finite Boolean
    residual, equivalent to ``NoContact(robot,target)``.
    ``require_tool_target_contact`` controls the positive mechanism term: when
    true, the candidate also pays one Boolean residual until tool-to-target
    contact exists at least once, equivalent to ``Contact(tool,target)``.
    More contact steps do not improve either term. Both use MuJoCo's exact pair with
    ``dist <= 0``; there is no contact-duration, penetration-depth,
    normal-angle, or task-specific distance threshold.

    Anchor-only evaluation deliberately contributes no proxy distance.  The
    production GPU scorer supplies the contact-aware finite cost directly, and
    refuses if that evidence is unavailable. Program validation permits this
    predicate only in ``Stage.running``: measured task completion remains an
    observable ``terminal`` condition, while executed causal evidence is
    logged separately.
    """

    type = "tool_mediation"
    tool: str = ""
    target: str = ""
    require_tool_target_contact: bool = True

    def residual(self, anchors: AnchorSet) -> float:
        # Exact contact is not recoverable from sparse endpoint anchors.  Never
        # invent a distance proxy here; GPU/explicit-contact evaluators apply
        # the finite contact/no-contact compatibility residuals and fail closed when
        # evidence is absent.
        return 0.0

    def raw_residual(
        self, anchors: AnchorSet, *, robot_target_contact: bool | None = None,
        tool_target_contact: bool | None = None,
    ) -> np.ndarray:
        """Table I's two-component residual, from whole-horizon contact events.

        Apply Eq. (3) componentwise and sum the two penalties. A missing
        contact channel is not evidence that no contact occurred.
        """
        del anchors
        if not isinstance(robot_target_contact, (bool, np.bool_)):
            raise ValueError("tool_mediation requires robot-target contact evidence")
        if self.require_tool_target_contact and not isinstance(tool_target_contact, (bool, np.bool_)):
            raise ValueError("tool_mediation requires tool-target contact evidence")
        return np.array([
            float(robot_target_contact),
            float(self.require_tool_target_contact and not tool_target_contact),
        ])

    def referenced_anchors(self) -> list[str]:
        return [self.tool, self.target]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "tool": self.tool,
            "target": self.target,
        }
        # Preserve the canonical bytes/hashes of historical programs, whose
        # original two-field form means the default contact-required contract.
        if not self.require_tool_target_contact:
            data["require_tool_target_contact"] = False
        return data

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "ToolMediation":
        required = d.get("require_tool_target_contact", True)
        if not isinstance(required, bool):
            raise ValueError(
                "tool_mediation.require_tool_target_contact must be Boolean"
            )
        return cls(
            tool=d["tool"],
            target=d["target"],
            require_tool_target_contact=required,
        )


@dataclass
class NonGeometric(Predicate):
    """Sentinel the VLM emits when a goal's success is NOT a grounded geometric/
    kinematic relation (fluid level, cloth-folded, cleanliness, force/'tight').
    is_geometric=False -> the single-tier router returns CANNOT_VERIFY.
    This is how the honest boundary stays UNIFORM instead of becoming per-task code."""
    type = "non_geometric"
    is_geometric = False
    reason: str = "non_geometric_success"

    def residual(self, anchors: AnchorSet) -> float:
        return 0.0  # cannot be analytically checked; never asserts success

    def satisfied(self, anchors: AnchorSet, eps: float = _EPS) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "reason": self.reason}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "NonGeometric":
        return cls(reason=d.get("reason", "non_geometric_success"))


@dataclass
class SupportedOn(Predicate):
    """The subject's whole support face rests on the support body's top face.

    ``subject``/``support`` are body vocabulary prefixes (a manifest id or the
    canonical ``subject``/``target``), not point anchors. The residual reads
    the subject's four material bottom-face corners (``{subject}
    _bottom_corner_i``, body-attached, so they tilt with the body) against the
    support's top-face frame (``{support}_top`` point, ``{support}_axis``/
    ``{support}_normal`` basis, in-plane half extents from the support's
    region anchor):

      max-corner footprint overhang  +  max-corner |top-plane gap| hinge,

    with ``tol`` bounding BOTH components symmetrically. Zero means every
    corner is inside the support footprint (to within ``tol``) and within
    ``tol`` of the top plane -- full-footprint containment plus face-on-face
    support, tilt-sensitive by construction (a tilted body lifts a corner off
    the plane). The tolerance is the measurement floor, not slack: corner
    positions come through reconstruction dims and tracking whose combined
    noise is millimetres, so a zero-tolerance overhang asks for certainty the
    sensor chain cannot provide (measured: a physically seated, released,
    at-rest placement held a frozen 0.1 mm overhang for 60 gates). Task
    general: no object-specific logic or hand tuning.
    """

    type = "supported_on"
    subject: str = ""
    support: str = ""
    tol: float = 0.005

    def _corner_names(self) -> list[str]:
        return [
            f"{self.subject}_bottom_corner_{index}" for index in range(4)
        ]

    def contact_target_ref(self, anchors: AnchorSet) -> str:
        """Return the support anchor only for the controlled subject."""
        subject = anchors.get(self.subject)
        if subject is None:
            raise ValueError(
                f"supported_on references ungrounded subject {self.subject!r}"
            )
        if (
            self.subject != "subject"
            and getattr(subject, "attached_to", None) != "subject"
        ):
            raise ValueError(
                "supported_on.subject must be the controlled subject or an "
                f"anchor attached to it, got {self.subject!r}"
            )

        support = anchors.get(self.support)
        top_ref = f"{self.support}_top"
        support_top = anchors.get(top_ref)
        missing = [
            ref
            for ref, anchor in (
                (self.support, support),
                (top_ref, support_top),
            )
            if anchor is None
        ]
        if missing:
            raise ValueError(
                "supported_on references ungrounded support anchors: "
                f"{sorted(missing)}"
            )
        assert support is not None and support_top is not None
        support_owner = getattr(support, "attached_to", None)
        top_owner = getattr(support_top, "attached_to", None)
        if (
            self.support == "subject"
            or support_owner == "subject"
            or top_owner == "subject"
            or self.subject == self.support
        ):
            raise ValueError(
                "supported_on.support must be distinct from the controlled "
                f"subject, got {self.support!r}"
            )
        if (
            not bool(getattr(support, "dynamic", False))
            or not support_owner
            or support_owner != top_owner
        ):
            raise ValueError(
                "supported_on.support must uniquely identify one dynamic "
                f"physics body, got {self.support!r}"
            )
        return top_ref

    def validate_grounded_contract(self, anchors: AnchorSet) -> str:
        """Validate every derived anchor consumed by :meth:`residual`."""
        top_ref = self.contact_target_ref(anchors)
        support = anchors[self.support]
        if support.region_half is None:
            raise ValueError(
                f"supported_on support {self.support!r} is not a region "
                "(missing region_half)"
            )
        planar_half = np.asarray(support.region_half, dtype=float)[:2]
        if not bool(np.all(np.isfinite(planar_half))):
            raise ValueError(
                f"supported_on support {self.support!r} has non-finite "
                "planar region_half"
            )

        axis_ref = f"{self.support}_axis"
        normal_ref = f"{self.support}_normal"
        corner_refs = self._corner_names()
        derived_refs = (top_ref, axis_ref, normal_ref, *corner_refs)
        missing = sorted(ref for ref in derived_refs if anchors.get(ref) is None)
        if missing:
            raise ValueError(
                "supported_on references ungrounded derived anchors: "
                f"{missing}"
            )

        support_owner = getattr(support, "attached_to", None)
        wrong_support_owner = sorted(
            ref
            for ref in (top_ref, axis_ref, normal_ref)
            if getattr(anchors[ref], "attached_to", None) != support_owner
        )
        if wrong_support_owner:
            raise ValueError(
                "supported_on support-derived anchors must share one physics "
                f"owner: {wrong_support_owner}"
            )

        wrong_corner_owner = sorted(
            ref
            for ref in corner_refs
            if getattr(anchors[ref], "attached_to", None) != "subject"
        )
        if wrong_corner_owner:
            raise ValueError(
                "supported_on bottom corners must be attached to the "
                f"controlled subject: {wrong_corner_owner}"
            )

        nonfinite = sorted(
            ref for ref in derived_refs if not anchors[ref].is_finite()
        )
        if nonfinite:
            raise ValueError(
                "supported_on derived anchors must be finite: "
                f"{nonfinite}"
            )
        missing_axes = sorted(
            ref
            for ref in (axis_ref, normal_ref)
            if anchors[ref].axis is None
        )
        if missing_axes:
            raise ValueError(
                "supported_on support frame anchors need axes: "
                f"{missing_axes}"
            )
        assert anchors[axis_ref].axis is not None
        assert anchors[normal_ref].axis is not None
        if float(
            np.linalg.norm(
                np.cross(anchors[axis_ref].axis, anchors[normal_ref].axis)
            )
        ) <= 1e-6:
            raise ValueError(
                "supported_on support axis and normal must be independent"
            )
        return top_ref

    def residual(self, anchors: AnchorSet) -> float:
        support_region = anchors[self.support]
        if support_region.region_half is None:
            raise ValueError(
                f"anchor {self.support!r} is not a region (no region_half)"
            )
        half = np.asarray(support_region.region_half, dtype=float)[:2]
        top = anchors.point(f"{self.support}_top")
        x_axis = unit(anchors.axis(f"{self.support}_axis"))
        z_axis = unit(anchors.axis(f"{self.support}_normal"))
        y_axis = unit(np.cross(z_axis, x_axis))
        overhang = 0.0
        plane_gap = 0.0
        for name in self._corner_names():
            rel = anchors.point(name) - top
            u = float(np.dot(rel, x_axis))
            v = float(np.dot(rel, y_axis))
            w = float(np.dot(rel, z_axis))
            overhang = max(
                overhang,
                abs(u) - float(half[0]) - float(self.tol),
                abs(v) - float(half[1]) - float(self.tol),
            )
            plane_gap = max(plane_gap, abs(w) - float(self.tol))
        return max(0.0, overhang) + max(0.0, plane_gap)

    def referenced_anchors(self) -> list[str]:
        return [
            *self._corner_names(),
            self.support,
            f"{self.support}_top",
            f"{self.support}_axis",
            f"{self.support}_normal",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "subject": self.subject,
            "support": self.support,
            "tol": self.tol,
        }

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "SupportedOn":
        return cls(
            subject=d["subject"],
            support=d["support"],
            tol=float(d.get("tol", 0.005)),
        )


@dataclass
class AtRest(Predicate):
    """The subject's measured linear and angular speed are within tolerance.

    Velocity is a dedicated measured-state channel, like the jaw width behind
    ``GripperState``: the residual reads the ``{subject}_velocity`` /
    ``{subject}_angular_velocity`` vector anchors the measurement layer
    grounds (offline: exact twin qvel; device: pose-grid tail differences).
    When the channel is absent the stage stays UNKNOWN -- absence never
    grades as rest. The residual is dimensionless:

      max(0, |v|/lin_tol - 1) + max(0, |w|/ang_tol - 1).
    """

    type = "at_rest"
    subject: str = ""
    lin_tol: float = 0.01
    ang_tol: float = 0.1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for label, value in (
            ("lin_tol", self.lin_tol),
            ("ang_tol", self.ang_tol),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{self.type}.{label} must be > 0")

    def velocity_anchor_names(self) -> list[str]:
        return [
            f"{self.subject}_velocity",
            f"{self.subject}_angular_velocity",
        ]

    def residual(self, anchors: AnchorSet) -> float:
        linear_name, angular_name = self.velocity_anchor_names()
        linear = anchors.get(linear_name)
        angular = anchors.get(angular_name)
        if linear is None or angular is None:
            raise ValueError(
                f"at_rest needs measured velocity anchors {linear_name!r}/"
                f"{angular_name!r}; absence must stay UNKNOWN, never rest"
            )
        linear_speed = float(np.linalg.norm(linear.point))
        angular_speed = float(np.linalg.norm(angular.point))
        return (
            max(0.0, linear_speed / float(self.lin_tol) - 1.0)
            + max(0.0, angular_speed / float(self.ang_tol) - 1.0)
        )

    def referenced_anchors(self) -> list[str]:
        # The scene anchor keeps groundability honest; the velocity channel is
        # measured state handled by _missing_terminal_measurements, exactly
        # like gripper_width.
        return [f"{self.subject}_center"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "subject": self.subject,
            "lin_tol": self.lin_tol,
            "ang_tol": self.ang_tol,
        }

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "AtRest":
        return cls(
            subject=d["subject"],
            lin_tol=float(d.get("lin_tol", 0.01)),
            ang_tol=float(d.get("ang_tol", 0.1)),
        )


@dataclass
class NotHeld(Predicate):
    """Require the object owning ``object_anchor`` to NOT be held.

    The exact evidence channel is ``ObjectHeld``'s, inverted: rollout
    left/right-pad contact on device, calibrated measured grasp evidence at
    the terminal gate. Unknown held evidence stays UNKNOWN -- it never grades
    as released. This is the object-side release fact; ``GripperState(open)``
    remains the independent jaw-side fact.
    """

    type = "not_held"
    object_anchor: str = ""

    def residual(self, anchors: AnchorSet) -> float:
        # Anchor-only interpreters cannot observe contact; the GPU cost and
        # measured-verdict adapters supply the exact Boolean evidence.
        return 0.0

    def raw_residual(self, anchors: AnchorSet, *, held: bool | None = None) -> float:
        """Table I: simultaneous two-finger contact at this step."""
        del anchors
        if not isinstance(held, (bool, np.bool_)):
            raise ValueError("not_held requires exact Boolean held evidence")
        return float(held)

    def referenced_anchors(self) -> list[str]:
        return [self.object_anchor]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "object_anchor": self.object_anchor}

    @classmethod
    def _from(cls, d: dict[str, Any]) -> "NotHeld":
        anchor = d.get("object_anchor")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(
                "not_held requires a non-empty object_anchor string"
            )
        return cls(object_anchor=anchor)


_PREDICATE_CLASSES: list[type[Predicate]] = [
    PointPointDistance, CaptureDistance, RelativePosition, MinDistance,
    AboveBy, OnPlane, PointOnLine, AxisParallel,
    AxisAngle, RelativeOrientation, RotationProgress, AngleAboutAxis, InRegion, SignedAxisGap,
    GripperState,
    GripperCommand,
    ObjectHeld, TemporalHold, InteractionAlign, AlongAxisGap,
    Contact, NoContact, RobotSubjectContact, ToolMediation,
    SupportedOn, AtRest, NotHeld,
    NonGeometric,
]
_REGISTRY: dict[str, type[Predicate]] = {c.type: c for c in _PREDICATE_CLASSES}
