"""Typed, canonical wire contract for the remote H100 planner.

Only planning crosses this boundary.  The lab process keeps perception,
tracking, stage termination, safety checks, and the sole robot executor.
Requests and responses are deliberately plain JSON so the transport can be an
SSH-forwarded loopback HTTP connection without importing a second RPC stack.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from oap.loop.execution_prefix import resolve_execution_prefix
from oap.loop.planner_profile import PlannerProfile
from oap.program import Anchor, AnchorSet, TaskProgram, program_sha256

PROTOCOL_VERSION = "oap.remote_planner.v3"

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class PlanningProtocolError(ValueError):
    """A malformed, stale, replayed, or identity-mismatched planning message."""


class ReplayRejected(PlanningProtocolError):
    """A request or response id was already consumed."""


class DeadlineExceeded(PlanningProtocolError):
    """A request or response arrived outside its execution deadline."""


class IdentityMismatch(PlanningProtocolError):
    """A program, state, model, asset, planner, or request hash mismatched."""


def _strict_keys(
    data: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    if not isinstance(data, Mapping):
        raise PlanningProtocolError(f"{label} must be a JSON object")
    optional = optional or set()
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required - optional)
    if missing:
        raise PlanningProtocolError(f"{label} missing fields: {missing}")
    if unknown:
        raise PlanningProtocolError(f"{label} has unknown fields: {unknown}")


def _identifier(value: Any, label: str) -> str:
    text = str(value)
    if not _ID_RE.fullmatch(text):
        raise PlanningProtocolError(
            f"{label} must match {_ID_RE.pattern!r}, got {text!r}"
        )
    return text


def _hash(value: Any, label: str) -> str:
    text = str(value)
    if not _HASH_RE.fullmatch(text):
        raise PlanningProtocolError(
            f"{label} must be a lowercase sha256 hex digest"
        )
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanningProtocolError(f"{label} must be an integer")
    out = int(value)
    if out < minimum:
        raise PlanningProtocolError(f"{label} must be >= {minimum}")
    return out


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PlanningProtocolError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanningProtocolError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise PlanningProtocolError(f"{label} must be finite")
    return out


def _vector(value: Any, label: str, *, size: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanningProtocolError(f"{label} must be a JSON array")
    out = tuple(_finite(item, f"{label}[{i}]") for i, item in enumerate(value))
    if size is not None and len(out) != size:
        raise PlanningProtocolError(
            f"{label} must contain {size} values, got {len(out)}"
        )
    return out


def _matrix(
    value: Any,
    label: str,
    *,
    columns: int,
    nonempty: bool = True,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanningProtocolError(f"{label} must be a JSON array of rows")
    rows = tuple(
        _vector(row, f"{label}[{i}]", size=columns)
        for i, row in enumerate(value)
    )
    if nonempty and not rows:
        raise PlanningProtocolError(f"{label} must not be empty")
    return rows


def _json_safe(value: Any, label: str = "value") -> Any:
    """Convert NumPy containers to finite, canonical JSON data."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite(value, label)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            _json_safe(item, f"{label}[{i}]")
            for i, item in enumerate(value)
        ]
    raise PlanningProtocolError(
        f"{label} contains non-JSON type {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Return the byte-stable JSON representation used by all wire hashes."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnchorPayload:
    """JSON representation of one grounded anchor."""

    point: tuple[float, ...]
    axis: tuple[float, ...] | None = None
    region_half: tuple[float, ...] | None = None
    kind: str = "keypoint"
    attached_to: str | None = None
    confidence: float = 1.0
    visibility: float = 1.0
    age_since_seen: int = 0
    last_pose: tuple[float, ...] | None = None
    dynamic: bool = False

    @staticmethod
    def from_anchor(anchor: Anchor) -> "AnchorPayload":
        return AnchorPayload(
            point=tuple(float(v) for v in anchor.point),
            axis=(
                None
                if anchor.axis is None
                else tuple(float(v) for v in anchor.axis)
            ),
            region_half=(
                None
                if anchor.region_half is None
                else tuple(float(v) for v in anchor.region_half)
            ),
            kind=str(anchor.kind),
            attached_to=anchor.attached_to,
            confidence=float(anchor.confidence),
            visibility=float(anchor.visibility),
            age_since_seen=int(anchor.age_since_seen),
            last_pose=(
                None
                if anchor.last_pose is None
                else tuple(float(v) for v in anchor.last_pose)
            ),
            dynamic=bool(anchor.dynamic),
        )

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AnchorPayload":
        _strict_keys(
            data,
            required={
                "point",
                "axis",
                "region_half",
                "kind",
                "attached_to",
                "confidence",
                "visibility",
                "age_since_seen",
                "last_pose",
                "dynamic",
            },
            label="anchor",
        )
        attached_to = data["attached_to"]
        if attached_to is not None:
            attached_to = str(attached_to)
        dynamic = data["dynamic"]
        if not isinstance(dynamic, bool):
            raise PlanningProtocolError("anchor.dynamic must be Boolean")
        payload = AnchorPayload(
            point=_vector(data["point"], "anchor.point", size=3),
            axis=(
                None
                if data["axis"] is None
                else _vector(data["axis"], "anchor.axis", size=3)
            ),
            region_half=(
                None
                if data["region_half"] is None
                else _vector(
                    data["region_half"], "anchor.region_half", size=3
                )
            ),
            kind=str(data["kind"]),
            attached_to=attached_to,
            confidence=_finite(data["confidence"], "anchor.confidence"),
            visibility=_finite(data["visibility"], "anchor.visibility"),
            age_since_seen=_integer(
                data["age_since_seen"], "anchor.age_since_seen"
            ),
            last_pose=(
                None
                if data["last_pose"] is None
                else _vector(data["last_pose"], "anchor.last_pose", size=3)
            ),
            dynamic=dynamic,
        )
        if not 0.0 <= payload.confidence <= 1.0:
            raise PlanningProtocolError(
                "anchor.confidence must be in [0, 1]"
            )
        if not 0.0 <= payload.visibility <= 1.0:
            raise PlanningProtocolError(
                "anchor.visibility must be in [0, 1]"
            )
        if payload.axis is not None and float(
            np.linalg.norm(payload.axis)
        ) <= 1e-12:
            raise PlanningProtocolError("anchor.axis must be nonzero")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": list(self.point),
            "axis": None if self.axis is None else list(self.axis),
            "region_half": (
                None if self.region_half is None else list(self.region_half)
            ),
            "kind": self.kind,
            "attached_to": self.attached_to,
            "confidence": self.confidence,
            "visibility": self.visibility,
            "age_since_seen": self.age_since_seen,
            "last_pose": (
                None if self.last_pose is None else list(self.last_pose)
            ),
            "dynamic": self.dynamic,
        }

    def to_anchor(self) -> Anchor:
        return Anchor(
            point=np.asarray(self.point, dtype=float),
            axis=None if self.axis is None else np.asarray(self.axis, dtype=float),
            region_half=(
                None
                if self.region_half is None
                else np.asarray(self.region_half, dtype=float)
            ),
            kind=self.kind,
            attached_to=self.attached_to,
            confidence=self.confidence,
            visibility=self.visibility,
            age_since_seen=self.age_since_seen,
            last_pose=(
                None
                if self.last_pose is None
                else np.asarray(self.last_pose, dtype=float)
            ),
            dynamic=self.dynamic,
        )


def anchors_to_payload(anchors: AnchorSet | None) -> dict[str, AnchorPayload]:
    if anchors is None:
        return {}
    return {
        name: AnchorPayload.from_anchor(anchors[name])
        for name in anchors.names()
    }


def anchors_from_payload(
    payload: Mapping[str, AnchorPayload],
) -> AnchorSet:
    return AnchorSet(
        {str(name): anchor.to_anchor() for name, anchor in payload.items()}
    )


@dataclass(frozen=True)
class RigidBodyGroundingPayload:
    """Body-local anchor template generated from one measured rigid pose.

    ``local_anchors`` contains no task geometry or hand-authored transform.
    The lab derives every local point/axis by applying the inverse of
    ``reference_pose7`` to the already-grounded scene anchors.  The complete
    payload is included in :class:`PlanningState` and therefore in
    ``state_sha256`` and the signed request identity.
    """

    reference_pose7: tuple[float, ...]
    local_anchors: dict[str, AnchorPayload]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RigidBodyGroundingPayload":
        _strict_keys(
            data,
            required={"reference_pose7", "local_anchors"},
            label="rigid body grounding",
        )
        raw_anchors = data["local_anchors"]
        if not isinstance(raw_anchors, Mapping) or not raw_anchors:
            raise PlanningProtocolError(
                "rigid body grounding local_anchors must be a nonempty object"
            )
        payload = RigidBodyGroundingPayload(
            reference_pose7=_vector(
                data["reference_pose7"],
                "rigid body grounding reference_pose7",
                size=7,
            ),
            local_anchors={
                str(name): AnchorPayload.from_dict(anchor)
                for name, anchor in raw_anchors.items()
            },
        )
        quat_norm = float(np.linalg.norm(payload.reference_pose7[3:7]))
        if not 0.999 <= quat_norm <= 1.001:
            raise PlanningProtocolError(
                "rigid body grounding reference quaternion must be unit length"
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_pose7": list(self.reference_pose7),
            "local_anchors": {
                name: anchor.to_dict()
                for name, anchor in sorted(self.local_anchors.items())
            },
        }


@dataclass(frozen=True)
class PlanningState:
    """All live, per-cycle inputs needed by the GPU optimizer."""

    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    applied_ctrl: tuple[float, ...]
    object_pose7: tuple[float, ...]
    nominal_candidate_id: int
    nominal_knots: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any]
    anchors: dict[str, AnchorPayload]
    cost_scale_anchors: dict[str, AnchorPayload]
    rigid_body_grounding: dict[str, RigidBodyGroundingPayload]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PlanningState":
        _strict_keys(
            data,
            required={
                "qpos",
                "qvel",
                "applied_ctrl",
                "object_pose7",
                "nominal_candidate_id",
                "nominal_knots",
                "metadata",
                "anchors",
                "cost_scale_anchors",
                "rigid_body_grounding",
            },
            label="planning state",
        )
        anchors = data["anchors"]
        scale_anchors = data["cost_scale_anchors"]
        raw_grounding = data["rigid_body_grounding"]
        if not isinstance(anchors, Mapping) or not isinstance(
            scale_anchors, Mapping
        ):
            raise PlanningProtocolError(
                "anchors and cost_scale_anchors must be JSON objects"
            )
        if not isinstance(raw_grounding, Mapping):
            raise PlanningProtocolError(
                "rigid_body_grounding must be a JSON object"
            )
        metadata = data["metadata"]
        if not isinstance(metadata, Mapping):
            raise PlanningProtocolError("state.metadata must be a JSON object")
        state = PlanningState(
            qpos=_vector(data["qpos"], "state.qpos"),
            qvel=_vector(data["qvel"], "state.qvel"),
            applied_ctrl=_vector(
                data["applied_ctrl"], "state.applied_ctrl", size=8
            ),
            object_pose7=_vector(
                data["object_pose7"], "state.object_pose7", size=7
            ),
            nominal_candidate_id=_integer(
                data["nominal_candidate_id"],
                "state.nominal_candidate_id",
            ),
            nominal_knots=_matrix(
                data["nominal_knots"],
                "state.nominal_knots",
                columns=8,
            ),
            metadata=_json_safe(dict(metadata), "state.metadata"),
            anchors={
                str(name): AnchorPayload.from_dict(value)
                for name, value in anchors.items()
            },
            cost_scale_anchors={
                str(name): AnchorPayload.from_dict(value)
                for name, value in scale_anchors.items()
            },
            rigid_body_grounding={
                _identifier(body, "rigid body name"):
                RigidBodyGroundingPayload.from_dict(value)
                for body, value in raw_grounding.items()
            },
        )
        if not state.qpos or not state.qvel:
            raise PlanningProtocolError("state qpos/qvel must not be empty")
        quat_norm = float(np.linalg.norm(state.object_pose7[3:7]))
        if not (0.5 <= quat_norm <= 1.5):
            raise PlanningProtocolError(
                "state.object_pose7 quaternion has implausible norm"
            )
        for body, grounding in state.rigid_body_grounding.items():
            for anchor_name, anchor in grounding.local_anchors.items():
                if not anchor_name:
                    raise PlanningProtocolError(
                        "rigid body local anchor name must not be empty"
                    )
                if not anchor.dynamic or anchor.attached_to != body:
                    raise PlanningProtocolError(
                        "rigid body local anchors must be dynamic and attached "
                        "to their owning body"
                    )
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "qpos": list(self.qpos),
            "qvel": list(self.qvel),
            "applied_ctrl": list(self.applied_ctrl),
            "object_pose7": list(self.object_pose7),
            "nominal_candidate_id": self.nominal_candidate_id,
            "nominal_knots": [list(row) for row in self.nominal_knots],
            "metadata": _json_safe(self.metadata, "state.metadata"),
            "anchors": {
                name: anchor.to_dict()
                for name, anchor in sorted(self.anchors.items())
            },
            "cost_scale_anchors": {
                name: anchor.to_dict()
                for name, anchor in sorted(
                    self.cost_scale_anchors.items()
                )
            },
            "rigid_body_grounding": {
                body: grounding.to_dict()
                for body, grounding in sorted(
                    self.rigid_body_grounding.items()
                )
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def rigid_body_grounding_sha256(self) -> str:
        return canonical_sha256({
            body: grounding.to_dict()
            for body, grounding in sorted(
                self.rigid_body_grounding.items()
            )
        })


@dataclass(frozen=True)
class ExecutionFeasibility:
    """Task-independent real actuator limits bound into a remote solve."""

    control_hz: float
    execution_dt_s: float
    controller_timing_compatible: bool
    controller_prefix_knot_times: tuple[float, ...]
    controller_segment_durations_s: tuple[float, ...]
    joint_position_min_rad: tuple[float, ...]
    joint_position_max_rad: tuple[float, ...]
    joint_velocity_max_rad_s: tuple[float, ...]
    joint_torque_max_nm: tuple[float, ...]
    effective_joint_limits_sha256: str
    gripper_width_min_m: float
    gripper_width_max_m: float
    gripper_velocity_min_m_s: float
    gripper_velocity_max_m_s: float
    gripper_limits_sha256: str

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ExecutionFeasibility":
        _strict_keys(
            data,
            required={
                "control_hz",
                "execution_dt_s",
                "controller_timing_compatible",
                "controller_prefix_knot_times",
                "controller_segment_durations_s",
                "joint_position_min_rad",
                "joint_position_max_rad",
                "joint_velocity_max_rad_s",
                "joint_torque_max_nm",
                "effective_joint_limits_sha256",
                "gripper_width_min_m",
                "gripper_width_max_m",
                "gripper_velocity_min_m_s",
                "gripper_velocity_max_m_s",
                "gripper_limits_sha256",
            },
            label="execution feasibility",
        )
        if not isinstance(data["controller_timing_compatible"], bool):
            raise PlanningProtocolError(
                "execution feasibility controller_timing_compatible must be boolean"
            )
        prefix_times = _vector(
            data["controller_prefix_knot_times"],
            "execution feasibility controller_prefix_knot_times",
        )
        segment_durations = _vector(
            data["controller_segment_durations_s"],
            "execution feasibility controller_segment_durations_s",
        )
        if (
            len(prefix_times) < 2
            or len(segment_durations) != len(prefix_times) - 1
            or abs(prefix_times[0]) > 1e-12
            or prefix_times[-1] > 1.0 + 1e-12
            or any(
                right <= left
                for left, right in zip(prefix_times, prefix_times[1:])
            )
            or any(value <= 0.0 for value in segment_durations)
        ):
            raise PlanningProtocolError(
                "execution feasibility prefix schedule is inconsistent"
            )
        lower = _vector(
            data["joint_position_min_rad"],
            "execution feasibility joint_position_min_rad",
            size=7,
        )
        upper = _vector(
            data["joint_position_max_rad"],
            "execution feasibility joint_position_max_rad",
            size=7,
        )
        velocity = _vector(
            data["joint_velocity_max_rad_s"],
            "execution feasibility joint_velocity_max_rad_s",
            size=7,
        )
        torque = _vector(
            data["joint_torque_max_nm"],
            "execution feasibility joint_torque_max_nm",
            size=7,
        )
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise PlanningProtocolError(
                "execution feasibility joint position intervals are invalid"
            )
        if any(value <= 0.0 for value in velocity):
            raise PlanningProtocolError(
                "execution feasibility joint velocity limits must be positive"
            )
        if any(value <= 0.0 for value in torque):
            raise PlanningProtocolError(
                "execution feasibility joint torque limits must be positive"
            )
        width_min = _finite(
            data["gripper_width_min_m"],
            "execution feasibility gripper_width_min_m",
        )
        width_max = _finite(
            data["gripper_width_max_m"],
            "execution feasibility gripper_width_max_m",
        )
        gripper_velocity_min = _finite(
            data["gripper_velocity_min_m_s"],
            "execution feasibility gripper_velocity_min_m_s",
        )
        gripper_velocity = _finite(
            data["gripper_velocity_max_m_s"],
            "execution feasibility gripper_velocity_max_m_s",
        )
        control_hz = _finite(
            data["control_hz"], "execution feasibility control_hz"
        )
        execution_dt_s = _finite(
            data["execution_dt_s"], "execution feasibility execution_dt_s"
        )
        if (
            width_min >= width_max
            or gripper_velocity_min < 0.0
            or gripper_velocity_min >= gripper_velocity
            or control_hz <= 0.0
            or execution_dt_s <= 0.0
        ):
            raise PlanningProtocolError(
                "execution feasibility rates and gripper interval must be positive"
            )
        return ExecutionFeasibility(
            control_hz=control_hz,
            execution_dt_s=execution_dt_s,
            controller_timing_compatible=data["controller_timing_compatible"],
            controller_prefix_knot_times=prefix_times,
            controller_segment_durations_s=segment_durations,
            joint_position_min_rad=lower,
            joint_position_max_rad=upper,
            joint_velocity_max_rad_s=velocity,
            joint_torque_max_nm=torque,
            effective_joint_limits_sha256=_hash(
                data["effective_joint_limits_sha256"],
                "execution feasibility effective_joint_limits_sha256",
            ),
            gripper_width_min_m=width_min,
            gripper_width_max_m=width_max,
            gripper_velocity_min_m_s=gripper_velocity_min,
            gripper_velocity_max_m_s=gripper_velocity,
            gripper_limits_sha256=_hash(
                data["gripper_limits_sha256"],
                "execution feasibility gripper_limits_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_hz": self.control_hz,
            "execution_dt_s": self.execution_dt_s,
            "controller_timing_compatible": self.controller_timing_compatible,
            "controller_prefix_knot_times": list(
                self.controller_prefix_knot_times
            ),
            "controller_segment_durations_s": list(
                self.controller_segment_durations_s
            ),
            "joint_position_min_rad": list(self.joint_position_min_rad),
            "joint_position_max_rad": list(self.joint_position_max_rad),
            "joint_velocity_max_rad_s": list(self.joint_velocity_max_rad_s),
            "joint_torque_max_nm": list(self.joint_torque_max_nm),
            "effective_joint_limits_sha256": self.effective_joint_limits_sha256,
            "gripper_width_min_m": self.gripper_width_min_m,
            "gripper_width_max_m": self.gripper_width_max_m,
            "gripper_velocity_min_m_s": self.gripper_velocity_min_m_s,
            "gripper_velocity_max_m_s": self.gripper_velocity_max_m_s,
            "gripper_limits_sha256": self.gripper_limits_sha256,
        }

    def to_device_validity(self) -> dict[str, Any]:
        return {
            "action_feasibility_enabled": True,
            **self.to_dict(),
        }


@dataclass(frozen=True)
class PlannerConfig:
    """Task-independent optimizer settings bound into the request."""

    pool_size: int
    horizon_steps: int
    rng_seed: int
    execution_prefix_fraction: float
    subject_height_m: float
    execution_prefix_steps: int
    execution_feasibility: ExecutionFeasibility | None = None

    def resolved_execution_prefix(self) -> tuple[int, float]:
        """Return the authoritative discrete steps and effective fraction."""
        try:
            resolved = resolve_execution_prefix(
                horizon_steps=self.horizon_steps,
                execution_prefix_fraction=self.execution_prefix_fraction,
                execution_prefix_steps=self.execution_prefix_steps,
            )
        except ValueError as exc:
            raise PlanningProtocolError(str(exc)) from exc
        return resolved.resolved_steps, resolved.effective_fraction

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PlannerConfig":
        _strict_keys(
            data,
            required={
                "pool_size",
                "horizon_steps",
                "rng_seed",
                "execution_prefix_fraction",
                "execution_prefix_steps",
                "subject_height_m",
                "execution_feasibility",
            },
            optional=set(),
            label="planner config",
        )
        config = PlannerConfig(
            pool_size=_integer(data["pool_size"], "config.pool_size", minimum=2),
            horizon_steps=_integer(
                data["horizon_steps"], "config.horizon_steps", minimum=1
            ),
            rng_seed=_integer(data["rng_seed"], "config.rng_seed"),
            execution_prefix_fraction=_finite(
                data["execution_prefix_fraction"],
                "config.execution_prefix_fraction",
            ),
            subject_height_m=_finite(
                data["subject_height_m"], "config.subject_height_m"
            ),
            execution_feasibility=(
                None
                if data["execution_feasibility"] is None
                else ExecutionFeasibility.from_dict(
                    data["execution_feasibility"]
                )
            ),
            execution_prefix_steps=_integer(
                data["execution_prefix_steps"],
                "config.execution_prefix_steps",
                minimum=1,
            ),
        )
        if not 0.0 < config.execution_prefix_fraction <= 1.0:
            raise PlanningProtocolError(
                "execution_prefix_fraction must be in (0, 1]"
            )
        if config.subject_height_m < 0.0:
            raise PlanningProtocolError("subject_height_m must be >= 0")
        prefix_steps, _effective_fraction = (
            config.resolved_execution_prefix()
        )
        feasibility = config.execution_feasibility
        if feasibility is not None:
            if config.horizon_steps < 2:
                raise PlanningProtocolError(
                    "execution feasibility requires horizon_steps >= 2"
                )
            expected_endpoint = float(prefix_steps - 1) / float(
                config.horizon_steps - 1
            )
            if abs(
                feasibility.controller_prefix_knot_times[-1]
                - expected_endpoint
            ) > 1e-12:
                raise PlanningProtocolError(
                    "execution feasibility prefix endpoint does not match "
                    "the rollout control sample"
                )
            step_boundaries = [
                int(math.ceil(config.horizon_steps * value - 1e-12))
                for value in feasibility.controller_prefix_knot_times
            ]
            cumulative_ticks = [
                int(math.floor(
                    step * feasibility.execution_dt_s
                    * feasibility.control_hz
                    + 0.5
                ))
                for step in step_boundaries
            ]
            expected_durations = [
                (right - left) / feasibility.control_hz
                for left, right in zip(
                    cumulative_ticks,
                    cumulative_ticks[1:],
                )
            ]
            if any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(
                    feasibility.controller_segment_durations_s,
                    expected_durations,
                )
            ):
                raise PlanningProtocolError(
                    "execution feasibility controller durations do not match "
                    "cumulative nearest-tick quantization"
                )
        return config

    def to_dict(self) -> dict[str, Any]:
        result = {
            "pool_size": self.pool_size,
            "horizon_steps": self.horizon_steps,
            "rng_seed": self.rng_seed,
            "execution_prefix_fraction": self.execution_prefix_fraction,
            "subject_height_m": self.subject_height_m,
            "execution_feasibility": (
                None
                if self.execution_feasibility is None
                else self.execution_feasibility.to_dict()
            ),
        }
        result["execution_prefix_steps"] = self.execution_prefix_steps
        return result


@dataclass(frozen=True)
class PlanningRequest:
    """One immutable remote planning request."""

    request_id: str
    episode_id: str
    stage_index: int
    cycle_index: int
    issued_at_unix_s: float
    deadline_unix_s: float
    program_json: str
    program_sha256: str
    state: PlanningState
    state_sha256: str
    model_sha256: str
    assets_sha256: str
    planner_sha256: str
    service_instance_id: str
    planner_profile: PlannerProfile
    planner_profile_sha256: str
    config: PlannerConfig

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PlanningRequest":
        _strict_keys(
            data,
            required={
                "protocol_version",
                "request_id",
                "episode_id",
                "stage_index",
                "cycle_index",
                "issued_at_unix_s",
                "deadline_unix_s",
                "program_json",
                "program_sha256",
                "state",
                "state_sha256",
                "model_sha256",
                "assets_sha256",
                "planner_sha256",
                "service_instance_id",
                "planner_profile",
                "planner_profile_sha256",
                "config",
            },
            label="planning request",
        )
        if data["protocol_version"] != PROTOCOL_VERSION:
            raise PlanningProtocolError(
                f"unsupported protocol_version {data['protocol_version']!r}"
            )
        state = PlanningState.from_dict(data["state"])
        if not isinstance(data["planner_profile"], Mapping):
            raise PlanningProtocolError(
                "planner_profile must be a JSON object"
            )
        try:
            planner_profile = PlannerProfile.from_dict(
                data["planner_profile"]
            )
        except (TypeError, ValueError) as exc:
            raise PlanningProtocolError(str(exc)) from exc
        request = PlanningRequest(
            request_id=_identifier(data["request_id"], "request_id"),
            episode_id=_identifier(data["episode_id"], "episode_id"),
            stage_index=_integer(data["stage_index"], "stage_index"),
            cycle_index=_integer(data["cycle_index"], "cycle_index"),
            issued_at_unix_s=_finite(
                data["issued_at_unix_s"], "issued_at_unix_s"
            ),
            deadline_unix_s=_finite(
                data["deadline_unix_s"], "deadline_unix_s"
            ),
            program_json=str(data["program_json"]),
            program_sha256=_hash(
                data["program_sha256"], "program_sha256"
            ),
            state=state,
            state_sha256=_hash(data["state_sha256"], "state_sha256"),
            model_sha256=_hash(data["model_sha256"], "model_sha256"),
            assets_sha256=_hash(data["assets_sha256"], "assets_sha256"),
            planner_sha256=_hash(
                data["planner_sha256"], "planner_sha256"
            ),
            service_instance_id=_identifier(
                data["service_instance_id"], "service_instance_id"
            ),
            planner_profile=planner_profile,
            planner_profile_sha256=_hash(
                data["planner_profile_sha256"],
                "planner_profile_sha256",
            ),
            config=PlannerConfig.from_dict(data["config"]),
        )
        request.validate_content_hashes()
        profile = request.planner_profile
        config = request.config
        if request.planner_profile_sha256 != profile.sha256:
            raise IdentityMismatch(
                "planner_profile_sha256 does not match planner_profile"
            )
        if (
            config.pool_size != profile.pool_size
            or config.horizon_steps != profile.horizon_steps
            or config.execution_prefix_steps
            != profile.execution_prefix_steps
        ):
            raise IdentityMismatch(
                "planner config does not match planner_profile"
            )
        if len(request.state.nominal_knots) != profile.num_knots:
            raise PlanningProtocolError(
                "nominal knot count does not match planner_profile.num_knots"
            )
        if request.cycle_index >= profile.max_stage_cycles:
            raise PlanningProtocolError(
                "cycle_index must be smaller than "
                "planner_profile.max_stage_cycles"
            )
        _prefix_steps, effective_fraction = config.resolved_execution_prefix()
        expected_fraction = (
            profile.execution_prefix_steps / profile.horizon_steps
        )
        if (
            abs(effective_fraction - expected_fraction) > 1e-12
            or abs(
                config.execution_prefix_fraction - expected_fraction
            ) > 1e-12
        ):
            raise IdentityMismatch(
                "execution prefix does not match planner_profile"
            )
        feasibility = request.config.execution_feasibility
        if feasibility is not None:
            knot_count = len(request.state.nominal_knots)
            if knot_count < 2:
                raise PlanningProtocolError(
                    "execution feasibility requires at least two action knots"
                )
            endpoint = feasibility.controller_prefix_knot_times[-1]
            expected_prefix_times = [0.0]
            expected_prefix_times.extend(
                index / float(knot_count - 1)
                for index in range(1, knot_count - 1)
                if index / float(knot_count - 1) < endpoint
            )
            expected_prefix_times.append(endpoint)
            if (
                len(expected_prefix_times)
                != len(feasibility.controller_prefix_knot_times)
                or any(
                    abs(actual - expected) > 1e-12
                    for actual, expected in zip(
                        feasibility.controller_prefix_knot_times,
                        expected_prefix_times,
                    )
                )
            ):
                raise PlanningProtocolError(
                    "execution feasibility must preserve every action-spline "
                    "break before the rollout endpoint"
                )
        if request.deadline_unix_s <= request.issued_at_unix_s:
            raise PlanningProtocolError(
                "deadline_unix_s must be after issued_at_unix_s"
            )
        return request

    def validate_content_hashes(self) -> None:
        try:
            program = TaskProgram.from_json(self.program_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PlanningProtocolError("program_json is invalid") from exc
        actual_program = program_sha256(program)
        if self.program_json != program.to_json():
            raise PlanningProtocolError(
                "program_json must use canonical TaskProgram JSON"
            )
        if actual_program != self.program_sha256:
            raise IdentityMismatch(
                "program_sha256 does not match canonical program_json"
            )
        if self.state.sha256 != self.state_sha256:
            raise IdentityMismatch(
                "state_sha256 does not match canonical planning state"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "stage_index": self.stage_index,
            "cycle_index": self.cycle_index,
            "issued_at_unix_s": self.issued_at_unix_s,
            "deadline_unix_s": self.deadline_unix_s,
            "program_json": self.program_json,
            "program_sha256": self.program_sha256,
            "state": self.state.to_dict(),
            "state_sha256": self.state_sha256,
            "model_sha256": self.model_sha256,
            "assets_sha256": self.assets_sha256,
            "planner_sha256": self.planner_sha256,
            "service_instance_id": self.service_instance_id,
            "planner_profile": self.planner_profile.to_dict(),
            "planner_profile_sha256": self.planner_profile_sha256,
            "config": self.config.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PlanningResponse:
    """A plan or an explicit no-plan result, with complete provenance."""

    request_id: str
    episode_id: str
    stage_index: int
    cycle_index: int
    request_sha256: str
    state_sha256: str
    program_sha256: str
    model_sha256: str
    assets_sha256: str
    planner_sha256: str
    planner_profile_sha256: str
    accepted_at_unix_s: float
    completed_at_unix_s: float
    deadline_unix_s: float
    service_instance_id: str
    algorithm: str
    pool_size: int
    horizon_steps: int
    rng_seed: int
    plan_valid: bool
    candidate_id: int | None
    knots: tuple[tuple[float, ...], ...] | None
    failure_reason: str | None
    final_n_valid: int | None
    flat_objective: bool
    initial_robot_scene_contact: bool | None
    sampler_elapsed_s: float
    selected_metadata: dict[str, Any]
    selected_rollout_diagnostics: dict[str, Any] | None
    call_site_records: tuple[dict[str, Any], ...]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PlanningResponse":
        _strict_keys(
            data,
            required={
                "protocol_version",
                "request_id",
                "episode_id",
                "stage_index",
                "cycle_index",
                "request_sha256",
                "state_sha256",
                "program_sha256",
                "model_sha256",
                "assets_sha256",
                "planner_sha256",
                "planner_profile_sha256",
                "accepted_at_unix_s",
                "completed_at_unix_s",
                "deadline_unix_s",
                "service_instance_id",
                "algorithm",
                "pool_size",
                "horizon_steps",
                "rng_seed",
                "plan_valid",
                "candidate_id",
                "knots",
                "failure_reason",
                "final_n_valid",
                "flat_objective",
                "initial_robot_scene_contact",
                "sampler_elapsed_s",
                "selected_metadata",
                "selected_rollout_diagnostics",
                "call_site_records",
            },
            label="planning response",
        )
        if data["protocol_version"] != PROTOCOL_VERSION:
            raise PlanningProtocolError(
                f"unsupported protocol_version {data['protocol_version']!r}"
            )
        plan_valid = data["plan_valid"]
        flat = data["flat_objective"]
        initial_contact = data["initial_robot_scene_contact"]
        if not isinstance(plan_valid, bool) or not isinstance(flat, bool):
            raise PlanningProtocolError(
                "plan_valid and flat_objective must be Boolean"
            )
        if initial_contact is not None and not isinstance(
            initial_contact, bool
        ):
            raise PlanningProtocolError(
                "initial_robot_scene_contact must be Boolean or null"
            )
        raw_candidate = data["candidate_id"]
        candidate_id = (
            None
            if raw_candidate is None
            else _integer(raw_candidate, "candidate_id")
        )
        raw_valid = data["final_n_valid"]
        final_n_valid = (
            None
            if raw_valid is None
            else _integer(raw_valid, "final_n_valid")
        )
        knots = (
            None
            if data["knots"] is None
            else _matrix(data["knots"], "response.knots", columns=8)
        )
        selected_metadata = data["selected_metadata"]
        diagnostics = data["selected_rollout_diagnostics"]
        records = data["call_site_records"]
        if not isinstance(selected_metadata, Mapping):
            raise PlanningProtocolError(
                "selected_metadata must be a JSON object"
            )
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise PlanningProtocolError(
                "selected_rollout_diagnostics must be an object or null"
            )
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise PlanningProtocolError(
                "call_site_records must be an array of JSON objects"
            )
        response = PlanningResponse(
            request_id=_identifier(data["request_id"], "request_id"),
            episode_id=_identifier(data["episode_id"], "episode_id"),
            stage_index=_integer(data["stage_index"], "stage_index"),
            cycle_index=_integer(data["cycle_index"], "cycle_index"),
            request_sha256=_hash(
                data["request_sha256"], "request_sha256"
            ),
            state_sha256=_hash(data["state_sha256"], "state_sha256"),
            program_sha256=_hash(
                data["program_sha256"], "program_sha256"
            ),
            model_sha256=_hash(data["model_sha256"], "model_sha256"),
            assets_sha256=_hash(data["assets_sha256"], "assets_sha256"),
            planner_sha256=_hash(
                data["planner_sha256"], "planner_sha256"
            ),
            planner_profile_sha256=_hash(
                data["planner_profile_sha256"],
                "planner_profile_sha256",
            ),
            accepted_at_unix_s=_finite(
                data["accepted_at_unix_s"], "accepted_at_unix_s"
            ),
            completed_at_unix_s=_finite(
                data["completed_at_unix_s"], "completed_at_unix_s"
            ),
            deadline_unix_s=_finite(
                data["deadline_unix_s"], "deadline_unix_s"
            ),
            service_instance_id=_identifier(
                data["service_instance_id"], "service_instance_id"
            ),
            algorithm=str(data["algorithm"]),
            pool_size=_integer(data["pool_size"], "pool_size", minimum=2),
            horizon_steps=_integer(
                data["horizon_steps"], "horizon_steps", minimum=1
            ),
            rng_seed=_integer(data["rng_seed"], "rng_seed"),
            plan_valid=plan_valid,
            candidate_id=candidate_id,
            knots=knots,
            failure_reason=(
                None
                if data["failure_reason"] is None
                else str(data["failure_reason"])
            ),
            final_n_valid=final_n_valid,
            flat_objective=flat,
            initial_robot_scene_contact=initial_contact,
            sampler_elapsed_s=_finite(
                data["sampler_elapsed_s"], "sampler_elapsed_s"
            ),
            selected_metadata=_json_safe(
                dict(selected_metadata), "selected_metadata"
            ),
            selected_rollout_diagnostics=(
                None
                if diagnostics is None
                else _json_safe(
                    dict(diagnostics),
                    "selected_rollout_diagnostics",
                )
            ),
            call_site_records=tuple(
                _json_safe(dict(record), "call_site_record")
                for record in records
            ),
        )
        if response.completed_at_unix_s < response.accepted_at_unix_s:
            raise PlanningProtocolError(
                "completed_at_unix_s precedes accepted_at_unix_s"
            )
        if response.plan_valid:
            if response.knots is None or response.candidate_id is None:
                raise PlanningProtocolError(
                    "valid response requires candidate_id and knots"
                )
            if response.final_n_valid is None or response.final_n_valid < 1:
                raise PlanningProtocolError(
                    "valid response requires final_n_valid >= 1"
                )
            if response.failure_reason is not None:
                raise PlanningProtocolError(
                    "valid response cannot carry failure_reason"
                )
        else:
            if response.knots is not None:
                raise PlanningProtocolError(
                    "invalid response must not carry executable knots"
                )
            if not response.failure_reason:
                raise PlanningProtocolError(
                    "invalid response requires a failure_reason"
                )
        return response

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "stage_index": self.stage_index,
            "cycle_index": self.cycle_index,
            "request_sha256": self.request_sha256,
            "state_sha256": self.state_sha256,
            "program_sha256": self.program_sha256,
            "model_sha256": self.model_sha256,
            "assets_sha256": self.assets_sha256,
            "planner_sha256": self.planner_sha256,
            "planner_profile_sha256": self.planner_profile_sha256,
            "accepted_at_unix_s": self.accepted_at_unix_s,
            "completed_at_unix_s": self.completed_at_unix_s,
            "deadline_unix_s": self.deadline_unix_s,
            "service_instance_id": self.service_instance_id,
            "algorithm": self.algorithm,
            "pool_size": self.pool_size,
            "horizon_steps": self.horizon_steps,
            "rng_seed": self.rng_seed,
            "plan_valid": self.plan_valid,
            "candidate_id": self.candidate_id,
            "knots": (
                None
                if self.knots is None
                else [list(row) for row in self.knots]
            ),
            "failure_reason": self.failure_reason,
            "final_n_valid": self.final_n_valid,
            "flat_objective": self.flat_objective,
            "initial_robot_scene_contact": (
                self.initial_robot_scene_contact
            ),
            "sampler_elapsed_s": self.sampler_elapsed_s,
            "selected_metadata": _json_safe(
                self.selected_metadata, "selected_metadata"
            ),
            "selected_rollout_diagnostics": (
                None
                if self.selected_rollout_diagnostics is None
                else _json_safe(
                    self.selected_rollout_diagnostics,
                    "selected_rollout_diagnostics",
                )
            ),
            "call_site_records": [
                _json_safe(record, "call_site_record")
                for record in self.call_site_records
            ],
        }


class ReplayGuard:
    """In-memory one-use guard for request or response ids."""

    def __init__(self, max_entries: int | None = None):
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = (
            None if max_entries is None else int(max_entries)
        )
        self._seen: dict[str, str] = {}
        self._lock = threading.Lock()

    def mark(self, identifier: str, digest: str) -> None:
        identifier = _identifier(identifier, "replay identifier")
        digest = _hash(digest, "replay digest")
        with self._lock:
            previous = self._seen.get(identifier)
            if previous is not None:
                detail = (
                    "same payload"
                    if previous == digest
                    else "different payload"
                )
                raise ReplayRejected(
                    f"replayed identifier {identifier!r} ({detail})"
                )
            self._seen[identifier] = digest
            while (
                self._max_entries is not None
                and len(self._seen) > self._max_entries
            ):
                self._seen.pop(next(iter(self._seen)))


def validate_request_at_service(
    request: PlanningRequest,
    *,
    now_unix_s: float,
    planner_sha256: str,
    service_instance_id: str,
    max_future_skew_s: float = 1.0,
    max_request_ttl_s: float = 600.0,
) -> None:
    """Check freshness and the exact planner identity before a GPU solve."""
    now = _finite(now_unix_s, "now_unix_s")
    expected_planner = _hash(planner_sha256, "service planner_sha256")
    if request.planner_sha256 != expected_planner:
        raise IdentityMismatch("request planner_sha256 is not deployed here")
    if request.service_instance_id != _identifier(
        service_instance_id, "service service_instance_id"
    ):
        raise IdentityMismatch(
            "request service_instance_id is not deployed here"
        )
    if request.issued_at_unix_s > now + float(max_future_skew_s):
        raise DeadlineExceeded("request issued_at is in the future")
    if now > request.deadline_unix_s:
        raise DeadlineExceeded("request deadline has already expired")
    if (
        request.deadline_unix_s - request.issued_at_unix_s
        > float(max_request_ttl_s)
    ):
        raise DeadlineExceeded("request TTL exceeds service policy")


def validate_response_for_request(
    response: PlanningResponse,
    request: PlanningRequest,
    *,
    now_unix_s: float,
) -> None:
    """Bind a response to exactly one fresh request; mismatch means no motion."""
    expected = {
        "request_id": request.request_id,
        "episode_id": request.episode_id,
        "stage_index": request.stage_index,
        "cycle_index": request.cycle_index,
        "request_sha256": request.sha256,
        "state_sha256": request.state_sha256,
        "program_sha256": request.program_sha256,
        "model_sha256": request.model_sha256,
        "assets_sha256": request.assets_sha256,
        "planner_sha256": request.planner_sha256,
        "planner_profile_sha256": request.planner_profile_sha256,
        "service_instance_id": request.service_instance_id,
        "deadline_unix_s": request.deadline_unix_s,
        "pool_size": request.config.pool_size,
        "horizon_steps": request.config.horizon_steps,
        "rng_seed": request.config.rng_seed,
    }
    for field, wanted in expected.items():
        if getattr(response, field) != wanted:
            raise IdentityMismatch(
                f"response {field} does not match its request"
            )
    now = _finite(now_unix_s, "now_unix_s")
    if (
        response.completed_at_unix_s > request.deadline_unix_s
        or now > request.deadline_unix_s
    ):
        raise DeadlineExceeded("planning response arrived after its deadline")
    if response.plan_valid:
        if response.knots is None or (
            len(response.knots) != request.planner_profile.num_knots
        ):
            raise PlanningProtocolError(
                "response knot count does not match planner_profile.num_knots"
            )
        sites = {str(record.get("site")) for record in response.call_site_records}
        required = {"trajectory_cost", "physics_gate"}
        if not required.issubset(sites):
            raise PlanningProtocolError(
                "valid response lacks remote trajectory_cost/physics_gate evidence"
            )
        for record in response.call_site_records:
            if record.get("program_sha256") != request.program_sha256:
                raise IdentityMismatch(
                    "remote call-site program hash mismatches request"
                )
