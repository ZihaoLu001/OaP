"""Task-agnostic rigid-body grounding for remote movable-object costs."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from oap.program import AnchorSet, TaskProgram
from oap.utils.se3 import quat_to_mat

from .protocol import (
    AnchorPayload,
    IdentityMismatch,
    PlanningProtocolError,
    PlanningState,
    RigidBodyGroundingPayload,
)


def _unit_pose7(value: Any, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise PlanningProtocolError(f"{label} must be a finite pose7")
    norm = float(np.linalg.norm(pose[3:7]))
    if not 0.999 <= norm <= 1.001:
        raise PlanningProtocolError(f"{label} quaternion must be unit length")
    pose = pose.copy()
    pose[3:7] /= norm
    return pose


def _to_local(anchor: AnchorPayload, pose7: np.ndarray) -> AnchorPayload:
    rotation = quat_to_mat(pose7[3:7])
    point = rotation.T @ (
        np.asarray(anchor.point, dtype=float) - pose7[:3]
    )
    axis = (
        None
        if anchor.axis is None
        else rotation.T @ np.asarray(anchor.axis, dtype=float)
    )
    last_pose = (
        None
        if anchor.last_pose is None
        else rotation.T @ (
            np.asarray(anchor.last_pose, dtype=float) - pose7[:3]
        )
    )
    return replace(
        anchor,
        point=tuple(float(value) for value in point),
        axis=(
            None
            if axis is None
            else tuple(float(value) for value in axis)
        ),
        last_pose=(
            None
            if last_pose is None
            else tuple(float(value) for value in last_pose)
        ),
    )


def _to_world(anchor: AnchorPayload, pose7: np.ndarray) -> AnchorPayload:
    rotation = quat_to_mat(pose7[3:7])
    point = pose7[:3] + rotation @ np.asarray(anchor.point, dtype=float)
    axis = (
        None
        if anchor.axis is None
        else rotation @ np.asarray(anchor.axis, dtype=float)
    )
    last_pose = (
        None
        if anchor.last_pose is None
        else pose7[:3]
        + rotation @ np.asarray(anchor.last_pose, dtype=float)
    )
    return replace(
        anchor,
        point=tuple(float(value) for value in point),
        axis=(
            None
            if axis is None
            else tuple(float(value) for value in axis)
        ),
        last_pose=(
            None
            if last_pose is None
            else tuple(float(value) for value in last_pose)
        ),
    )


def build_rigid_body_grounding(
    *,
    world: Any,
    qpos: Any,
    anchors: AnchorSet | None,
) -> dict[str, RigidBodyGroundingPayload]:
    """Generate body-local templates from the measured state and anchors.

    The mapping of body names to free-joint addresses comes from the compiled,
    release-bound model.  Every transform is inferred from the current pose;
    there are no object-specific constants or task-dependent branches.
    """
    if anchors is None:
        return {}
    state_qpos = np.asarray(qpos, dtype=float)
    if state_qpos.ndim != 1 or not np.all(np.isfinite(state_qpos)):
        raise PlanningProtocolError("grounding qpos must be a finite vector")
    addresses = dict(getattr(world, "movable_qpos_addr", {}) or {})
    if not addresses:
        return {}
    subject_address = getattr(world, "object_qpos_addr", None)
    if subject_address is not None:
        subject_address = int(subject_address)
    payload: dict[str, RigidBodyGroundingPayload] = {}
    for body, raw_address in sorted(addresses.items()):
        address = int(raw_address)
        if subject_address is not None and address == subject_address:
            continue
        if address < 0 or address + 7 > state_qpos.size:
            raise PlanningProtocolError(
                f"movable body {body!r} qpos address is outside the state"
            )
        local_anchors: dict[str, AnchorPayload] = {}
        pose7 = _unit_pose7(
            state_qpos[address:address + 7],
            f"movable body {body!r} reference pose",
        )
        for name in anchors.names():
            anchor = anchors[name]
            if anchor.attached_to != body or not anchor.dynamic:
                continue
            local_anchors[name] = _to_local(
                AnchorPayload.from_anchor(anchor),
                pose7,
            )
        if local_anchors:
            payload[str(body)] = RigidBodyGroundingPayload(
                reference_pose7=tuple(float(value) for value in pose7),
                local_anchors=local_anchors,
            )
    return payload


def _anchors_equivalent(
    actual: AnchorPayload,
    expected: AnchorPayload,
) -> bool:
    exact_fields = (
        "kind",
        "attached_to",
        "age_since_seen",
        "dynamic",
    )
    if any(
        getattr(actual, field) != getattr(expected, field)
        for field in exact_fields
    ):
        return False
    if not np.isclose(actual.confidence, expected.confidence):
        return False
    if not np.isclose(actual.visibility, expected.visibility):
        return False
    for field in ("point", "axis", "region_half", "last_pose"):
        left = getattr(actual, field)
        right = getattr(expected, field)
        if (left is None) != (right is None):
            return False
        if left is not None and not np.allclose(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            rtol=0.0,
            atol=1e-9,
        ):
            return False
    return True


def validated_rigid_body_grounder(
    *,
    world: Any,
    state: PlanningState,
    program: TaskProgram,
    stage_index: int,
) -> Any:
    """Validate request templates against the released model and current state.

    A stage that references a non-subject movable anchor cannot be evaluated
    without a matching body-local template.  Such a request is rejected before
    GPU work instead of silently turning the objective into a constant.
    """
    subject_address = getattr(world, "object_qpos_addr", None)
    if subject_address is not None:
        subject_address = int(subject_address)
    available = {
        str(name): int(address)
        for name, address in (
            getattr(world, "movable_qpos_addr", {}) or {}
        ).items()
        if subject_address is None or int(address) != subject_address
    }
    for body, grounding in state.rigid_body_grounding.items():
        address = available.get(body)
        if address is None:
            raise IdentityMismatch(
                f"rigid grounding body {body!r} is absent from released model"
            )
        if address < 0 or address + 7 > len(state.qpos):
            raise IdentityMismatch(
                f"rigid grounding body {body!r} has an invalid qpos address"
            )
        current_pose = _unit_pose7(
            state.qpos[address:address + 7],
            f"request body {body!r} pose",
        )
        reference_pose = _unit_pose7(
            grounding.reference_pose7,
            f"request body {body!r} grounding reference",
        )
        if not np.allclose(
            current_pose,
            reference_pose,
            rtol=0.0,
            atol=1e-9,
        ):
            raise IdentityMismatch(
                f"rigid grounding body {body!r} does not match request qpos"
            )
        for anchor_name, local_anchor in grounding.local_anchors.items():
            current_anchor = state.anchors.get(anchor_name)
            if current_anchor is None or not _anchors_equivalent(
                _to_world(local_anchor, reference_pose),
                current_anchor,
            ):
                raise IdentityMismatch(
                    f"rigid grounding anchor {anchor_name!r} does not match "
                    "the measured request anchor"
                )

    if stage_index < 0 or stage_index >= len(program.stages):
        raise PlanningProtocolError("stage index is outside the program")
    stage = program.stages[stage_index]
    required: dict[str, set[str]] = {}
    for predicate in (*stage.running, *stage.terminal):
        for anchor_name in predicate.referenced_anchors():
            anchor = state.anchors.get(anchor_name)
            if anchor is None or anchor.attached_to not in available:
                continue
            required.setdefault(str(anchor.attached_to), set()).add(anchor_name)
    for body, anchor_names in required.items():
        required_grounding = state.rigid_body_grounding.get(body)
        if required_grounding is None:
            raise IdentityMismatch(
                f"stage requires ungrounded non-subject movable body {body!r}"
            )
        missing = sorted(
            anchor_names - set(required_grounding.local_anchors)
        )
        if missing:
            raise IdentityMismatch(
                f"stage movable body {body!r} lacks local anchors {missing}"
            )

    templates = dict(state.rigid_body_grounding)

    def ground(aset: AnchorSet, poses: Mapping[str, Any]) -> None:
        for body in required:
            if body not in poses:
                raise PlanningProtocolError(
                    f"rollout omitted required movable body {body!r}"
                )
        for body, raw_pose in poses.items():
            grounding = templates.get(str(body))
            if grounding is None:
                continue
            pose7 = _unit_pose7(
                raw_pose,
                f"rollout movable body {body!r} pose",
            )
            for anchor_name, local_anchor in (
                grounding.local_anchors.items()
            ):
                aset.add(anchor_name, _to_world(local_anchor, pose7).to_anchor())

    return ground
