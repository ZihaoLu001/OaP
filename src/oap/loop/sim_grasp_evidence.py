"""Opt-in measured grasp evidence for simulation stage transitions.

The frozen executor treats endpoint two-sided contact as ``ObjectHeld``.  The
``measured_v1`` canary keeps contact as one necessary observation and
corroborates it with the measured jaw aperture and the current subject's
occupancy of the physical volume between the two fingertip pads.  This module
also defines ``contact_width_v1``, which uses bilateral contact and a measured
aperture of at least 10 mm consistently in rollout costs and stage transitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from oap.loop.object_held_geometry import (
    OrientedBox,
    closing_region_intersection,
)

SIM_HELD_EVIDENCE_ENV = "OAP_SIM_HELD_EVIDENCE"
SIM_HELD_EVIDENCE_FROZEN = "frozen"
SIM_HELD_EVIDENCE_MEASURED_V1 = "measured_v1"
SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1 = "contact_width_v1"
SIM_HELD_EVIDENCE_MODES = (
    SIM_HELD_EVIDENCE_FROZEN,
    SIM_HELD_EVIDENCE_MEASURED_V1,
    SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1,
)
GRASP_BRIDGE_TELEMETRY_ENV = "OAP_GRASP_BRIDGE_TELEMETRY"
GRASP_BRIDGE_TELEMETRY_OFF = "off"
GRASP_BRIDGE_TELEMETRY_A191_V1 = "a191_v1"
GRASP_BRIDGE_TELEMETRY_MODES = (
    GRASP_BRIDGE_TELEMETRY_OFF,
    GRASP_BRIDGE_TELEMETRY_A191_V1,
)
SIM_HELD_MIN_WIDTH_M = 0.010
SIM_HELD_DWELL_SAMPLES = 2


def resolve_sim_held_evidence_mode() -> str:
    """Resolve the opt-in simulator evidence mode; a typo is an error."""
    raw = (os.environ.get(SIM_HELD_EVIDENCE_ENV) or "").strip()
    if not raw:
        return SIM_HELD_EVIDENCE_FROZEN
    if raw not in SIM_HELD_EVIDENCE_MODES:
        raise ValueError(f"{SIM_HELD_EVIDENCE_ENV}={raw!r}: expected one of {', '.join(SIM_HELD_EVIDENCE_MODES)}")
    return raw


def resolve_grasp_bridge_telemetry_mode() -> str:
    """Resolve the opt-in A191 evidence-only recording mode.

    This switch is deliberately separate from ``SIM_HELD_EVIDENCE_ENV``:
    enabling it records the same post-prefix measurements for every bridge
    configuration but never changes a stage decision, rollout cost, candidate
    validity, or offline grade.
    """
    raw = (os.environ.get(GRASP_BRIDGE_TELEMETRY_ENV) or "").strip()
    if not raw:
        return GRASP_BRIDGE_TELEMETRY_OFF
    if raw not in GRASP_BRIDGE_TELEMETRY_MODES:
        raise ValueError(
            f"{GRASP_BRIDGE_TELEMETRY_ENV}={raw!r}: expected one of "
            f"{', '.join(GRASP_BRIDGE_TELEMETRY_MODES)}"
        )
    return raw


def should_record_canonical_grasp_evidence(
    *,
    held_evidence_mode: str,
    telemetry_mode: str,
) -> bool:
    """Return whether canonical evidence should be persisted this cycle."""
    if held_evidence_mode not in SIM_HELD_EVIDENCE_MODES:
        raise ValueError(
            f"unsupported simulator held evidence mode {held_evidence_mode!r}"
        )
    if telemetry_mode not in GRASP_BRIDGE_TELEMETRY_MODES:
        raise ValueError(
            f"unsupported grasp bridge telemetry mode {telemetry_mode!r}"
        )
    return bool(
        held_evidence_mode in (
            SIM_HELD_EVIDENCE_MEASURED_V1,
            SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1,
        )
        or telemetry_mode == GRASP_BRIDGE_TELEMETRY_A191_V1
    )


def _finite_width(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        width = float(value)
    except (TypeError, ValueError):
        return None
    return width if np.isfinite(width) else None


def contact_width_held(bilateral_contact: Any, width_m: Any) -> Any:
    """Basic held predicate for scalar, NumPy and JAX physics-step evidence."""
    # Float32 rounds 0.010 downward. Compare against the first representable
    # float32 at or above the actual threshold so host float64 and GPU float32
    # agree on the same measured value, without changing the 10 mm criterion.
    threshold = SIM_HELD_MIN_WIDTH_M
    if getattr(width_m, "dtype", None) == np.dtype("float32"):
        threshold = float(np.nextafter(np.float32(threshold), np.float32(np.inf)))
    return (
        bilateral_contact
        & (width_m >= threshold)
        & (width_m < float("inf"))
    )


def fuse_contact_width_evidence(
    *,
    bilateral_contact: bool | None,
    measured_width_m: object,
) -> dict[str, Any]:
    """Record contact/aperture evidence without spatial or dwell conditions."""
    width = _finite_width(measured_width_m)
    contact = (
        bool(bilateral_contact)
        if isinstance(bilateral_contact, (bool, np.bool_))
        else None
    )
    width_ok = None if width is None else bool(width >= SIM_HELD_MIN_WIDTH_M)
    held = None
    if contact is not None and width is not None:
        held = bool(contact_width_held(contact, width))
    return {
        "schema": "oap_sim_contact_width_evidence_v1",
        "subject_held": held,
        "observable": held is not None,
        "bilateral_contact": contact,
        "gripper_width_m": width,
        "min_gripper_width_m": SIM_HELD_MIN_WIDTH_M,
        "width_corroborates": width_ok,
    }


def fuse_sim_held_evidence(
    *,
    bilateral_contact: bool | None,
    measured_width_m: object,
    closing_region_observable: bool,
    subject_intersects_closing_region: bool | None,
    min_width_m: float = SIM_HELD_MIN_WIDTH_M,
) -> dict[str, Any]:
    """Fuse three necessary grasp observations using three-valued logic.

    Any observed false condition proves ``not held``.  All three conditions
    must be observed true to prove ``held``; otherwise the result is unknown.
    The spatial condition is intersection with the pad closing region, not
    full containment of the entire object.
    """
    width = _finite_width(measured_width_m)
    width_ok = None if width is None else bool(width >= float(min_width_m))
    contact_ok = bool(bilateral_contact) if isinstance(bilateral_contact, (bool, np.bool_)) else None
    spatial_ok = (
        bool(subject_intersects_closing_region)
        if closing_region_observable and isinstance(subject_intersects_closing_region, (bool, np.bool_))
        else None
    )
    conditions = (contact_ok, width_ok, spatial_ok)
    if any(value is False for value in conditions):
        held: bool | None = False
    elif all(value is True for value in conditions):
        held = True
    else:
        held = None
    return {
        "schema": "oap_sim_held_evidence_v1",
        "subject_held": held,
        "observable": held is not None,
        "bilateral_contact": contact_ok,
        "gripper_width_m": width,
        "min_gripper_width_m": float(min_width_m),
        "width_corroborates": width_ok,
        "closing_region_observable": bool(closing_region_observable),
        "subject_intersects_closing_region": spatial_ok,
    }


def current_sim_closing_region_evidence(world: Any) -> dict[str, Any]:
    """Attribute the current simulated subject to the GN01 pad gap.

    The caller must first refresh host kinematics for the selected executed
    prefix.  This helper reads those current transforms only; it never runs
    dynamics or uses host contact state.
    """
    evidence: dict[str, Any] = {
        "schema": "oap_sim_closing_region_evidence_v1",
        "observable": False,
        "intersects_closing_region": None,
        "reason_code": None,
    }

    def fail(code: str, detail: object | None = None) -> dict[str, Any]:
        evidence["reason_code"] = code
        evidence["reason"] = code if detail is None else f"{code}:{detail}"
        return evidence

    try:
        import mujoco

        model = world.model
        data = world.data
        subject_qpos_addr = int(world.object_qpos_addr)
        subject_joints = [
            joint_id
            for joint_id in range(int(model.njnt))
            if (
                int(model.jnt_qposadr[joint_id]) == subject_qpos_addr
                and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
            )
        ]
        if len(subject_joints) != 1:
            return fail("subject_freejoint_role_ambiguous", len(subject_joints))
        subject_body_id = int(model.jnt_bodyid[subject_joints[0]])

        def belongs_to_subject(body_id: int) -> bool:
            current = int(body_id)
            visited: set[int] = set()
            while current >= 0 and current not in visited:
                if current == subject_body_id:
                    return True
                visited.add(current)
                parent = int(model.body_parentid[current])
                if parent == current:
                    break
                current = parent
            return False

        subject_geom_ids = [
            geom_id
            for geom_id in range(int(model.ngeom))
            if (
                belongs_to_subject(int(model.geom_bodyid[geom_id]))
                and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
            )
        ]
        if not subject_geom_ids:
            return fail("subject_collision_geometry_missing")
        supported_subject_geom_types = {
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        }
        unsupported = [
            int(geom_id)
            for geom_id in subject_geom_ids
            if int(model.geom_type[geom_id]) not in supported_subject_geom_types
        ]
        if unsupported:
            return fail(
                "subject_collision_geometry_unsupported_non_box_or_cylinder",
                unsupported,
            )

        pad_names = (
            "gn01_left_finger_tip_collision",
            "gn01_right_finger_tip_collision",
        )
        pad_ids = [
            int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
            )
            for name in pad_names
        ]
        missing_pads = [name for name, geom_id in zip(pad_names, pad_ids, strict=True) if geom_id < 0]
        if missing_pads:
            return fail("gn01_pad_geometry_missing", missing_pads)
        non_box_pads = [
            name
            for name, geom_id in zip(pad_names, pad_ids, strict=True)
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX)
        ]
        if non_box_pads:
            return fail("gn01_pad_geometry_unsupported_non_box", non_box_pads)

        def world_box(
            geom_id: int,
            *,
            name: str,
        ) -> tuple[OrientedBox, str]:
            geom_type = int(model.geom_type[int(geom_id)])
            geom_size = np.asarray(model.geom_size[int(geom_id)], dtype=float)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                half_extents = geom_size[:3]
                extent_source = "collision_box"
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                # MuJoCo cylinder size is (radius, half-length).  Its local
                # axis is Z, so this is the conservative oriented bounding box.
                half_extents = np.array(
                    [geom_size[0], geom_size[0], geom_size[1]],
                    dtype=float,
                )
                extent_source = "collision_cylinder_bounding_box"
            else:
                raise ValueError(f"unsupported collision geom type {geom_type}")
            return OrientedBox.validated(
                center=data.geom_xpos[int(geom_id)],
                axes=np.asarray(data.geom_xmat[int(geom_id)], dtype=float).reshape(3, 3),
                half_extents=half_extents,
                name=name,
            ), extent_source

        left_pad, _ = world_box(pad_ids[0], name="left pad")
        right_pad, _ = world_box(pad_ids[1], name="right pad")
        gripper_axes = np.asarray(data.site_xmat[int(world.site_id)], dtype=float).reshape(3, 3)
        subject_boxes_and_sources = [
            world_box(
                geom_id,
                name=f"subject collision geom {geom_id}",
            )
            for geom_id in subject_geom_ids
        ]
        results = [
            closing_region_intersection(
                subject=subject_box,
                left_pad=left_pad,
                right_pad=right_pad,
                gripper_axes=gripper_axes,
            )
            for subject_box, _ in subject_boxes_and_sources
        ]
        intersects = any(result.intersects for result in results)
        region = results[0].region
        subject_extent_sources = [source for _, source in subject_boxes_and_sources]
    except Exception as exc:  # noqa: BLE001 - evidence fails closed
        return fail("spatial_geometry_evaluation_failed", repr(exc))

    evidence.update(
        {
            "observable": True,
            "intersects_closing_region": bool(intersects),
            "reason_code": (
                "subject_intersects_physical_closing_region"
                if intersects
                else "subject_outside_physical_closing_region"
            ),
            "reason": (
                "subject_intersects_physical_closing_region"
                if intersects
                else "subject_outside_physical_closing_region"
            ),
            "closing_region_center": region.center.tolist(),
            "closing_region_axes": region.axes.tolist(),
            "closing_region_half_extents_m": region.half_extents.tolist(),
            "subject_collision_geom_ids": [int(value) for value in subject_geom_ids],
            "subject_extent_source": (
                subject_extent_sources[0]
                if len(set(subject_extent_sources)) == 1
                else "mixed_collision_geometry_bounding_boxes"
            ),
            "subject_extent_sources": list(subject_extent_sources),
            "pad_geometry_names": list(pad_names),
            "fk_source": "current_post_prefix_mujoco_kinematics",
        }
    )
    return evidence


@dataclass
class ExecutedSampleDwell:
    """Require consecutive distinct executed samples before returning true."""

    required_true_samples: int = SIM_HELD_DWELL_SAMPLES
    true_streak: int = 0
    last_sample_id: int | str | None = None

    def __post_init__(self) -> None:
        if int(self.required_true_samples) < 1:
            raise ValueError("required_true_samples must be positive")

    def observe(
        self,
        value: bool | None,
        *,
        sample_id: int | str | None,
    ) -> bool | None:
        """Observe one sample; duplicates do not advance the dwell counter."""
        if value is not True:
            # A loss or an unobservable reading invalidates the acquisition
            # dwell immediately, even if a caller repeats the same sample id.
            # Remembering a non-null id also prevents a contradictory re-read
            # of that same executed sample from counting as a new true sample.
            self.true_streak = 0
            if sample_id is not None:
                self.last_sample_id = sample_id
            return False if value is False else None
        if sample_id is None:
            return None
        if sample_id != self.last_sample_id:
            self.last_sample_id = sample_id
            self.true_streak += 1
        return True if self.true_streak >= self.required_true_samples else None
