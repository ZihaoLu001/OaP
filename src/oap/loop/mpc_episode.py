"""Joint-space receding-horizon execution over explicit program stages.

For every stage the controller starts from a content-free hold, optimizes its
declared running/terminal objective, applies one prefix, obtains an updated
physical state, and checks terminal on that measured state. Later cycles of
the same stage shift the previous winner; a new stage starts from the latest
acknowledged actuator hold because its objective changed.

The decision variables are full arm-and-gripper joint-target knots. Dry-run
execution consumes the selected MJWarp prefix state. Real execution uses the
numeric joint/gripper hook and then the whole-scene reobservation hook supplied
by :func:`oap.loop.runner.run_episode`. Both paths re-ground the same
program before terminal evaluation; neither uses execution-time IK.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from oap.program.cost_shaping import validate_cost_shaping_profile
from oap.loop.mpc_loop import (
    run_receding_horizon,
)
from oap.loop.sampling import (
    CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
    validate_candidate_selection_mode,
)
from oap.loop.execution_prefix import (
    DEFAULT_EXECUTION_PREFIX_FRACTION,
    resolve_execution_prefix,
    validate_execution_prefix_fraction,
    validate_mppi_stage_execution_prefix_steps,
)
from oap.loop.sim_grasp_evidence import (
    SIM_HELD_DWELL_SAMPLES,
    SIM_HELD_EVIDENCE_FROZEN,
    SIM_HELD_EVIDENCE_MEASURED_V1,
    SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1,
    ExecutedSampleDwell,
    resolve_sim_held_evidence_mode,
)
from oap.twin.batched_rollout import (
    DEFAULT_CEM_ELITE_FRACTION,
    DEFAULT_CEM_MIN_STD_FRACTION,
    DEFAULT_CEM_ROUNDS,
    DEFAULT_EXECUTION_STEPS,
    DEFAULT_ARM_VELOCITY_WEIGHT,
    DEFAULT_MPPI_TEMPERATURE,
    MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    MTP_JAW_MODE_PRESERVE_NOMINAL,
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_SIGMA_FRACTION,
    PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
    current_robot_subject_contact,
    validate_horizon_steps,
    validate_cem_elite_fraction,
    validate_cem_rounds,
    validate_local_sigma_fraction,
    validate_mtp_jaw_mode,
    validate_predictive_sampler_mode,
    validate_mppi_temperature,
    validate_mppi_execution_mode,
    validate_arm_velocity_weight,
)
from oap.loop.plan import DEFAULT_N_KNOTS, validate_num_knots
from oap.twin.control_profile import (
    CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
    validate_control_profile,
)
logger = logging.getLogger("oap.loop.mpc_episode")


def _acknowledged_gripper_command_latent(
    acknowledged: dict[str, Any],
    *,
    control_profile: str | None,
    pick_v8_causal_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
) -> float | None:
    """Decode only the command unit authorized by the active profile."""
    if (
        pick_v8_causal_profile is not None
        or unified_mppi_effort_profile is not None
    ) and control_profile is not None:
        from oap.twin.control_knots import (
            jaw_command_state_from_native_control,
        )
        from oap.twin.pick_v8_causal import pick_v8_causal_uses_width_jaw

        field = (
            "commanded_width_m"
            if pick_v8_causal_profile is not None
            and pick_v8_causal_uses_width_jaw(pick_v8_causal_profile)
            else "commanded_effort_n"
        )
        native = acknowledged.get(field)
        if native is not None and np.isfinite(float(native)):
            return float(jaw_command_state_from_native_control(
                float(native),
                control_profile=control_profile,
                pick_v8_causal_profile=pick_v8_causal_profile,
                unified_mppi_effort_profile=unified_mppi_effort_profile,
                source="terminal_acknowledged_jaw_command",
            ).latent)
    effort = acknowledged.get("commanded_effort_n")
    if effort is not None and np.isfinite(float(effort)):
        return float(np.clip(float(effort) / 80.0, -1.0, 1.0))
    width = acknowledged.get("commanded_width_m")
    if (
        control_profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
        and width is not None
        and np.isfinite(float(width))
    ):
        return float(np.clip(1.0 - 2.0 * float(width) / 0.085, -1.0, 1.0))
    return None


def _record_measured_program_state(
    stage: Any, anchors: Any, evidence: dict[str, Any],
) -> None:
    """Record post-execution FK and scalar predicates without changing control.

    These are predicate residuals on the current anchor snapshot, not the
    device running-cost norm or a predicted trajectory contribution. Command
    residuals read the acknowledged command and do not establish jaw closure.
    Contact and held terms need their separate execution-evidence channels.
    """
    from oap.program.verdict import measurement_channels

    center = anchors.get("gripper_center")
    closing = anchors.get("gripper_closing_axis")
    if (
        center is not None and closing is not None
        and center.kind == "robot" and closing.kind == "robot"
        and center.is_finite() and closing.is_finite()
        and center.axis is not None and closing.axis is not None
    ):
        evidence["measured_gripper_pose"] = {
            "position_m": center.point.tolist(),
            "tool_axis": center.axis.tolist(),
            "closing_axis": closing.axis.tolist(),
            "frame": "plan_world",
            "source": "current_joint_state_fk",
        }

    scalar_units = {
        "point_point_distance": "m", "min_distance": "m",
        "relative_position": "m", "signed_axis_gap": "m",
        "axis_parallel": "rad", "axis_angle": "rad",
        "gripper_state": "m", "gripper_command": "m",
        "point_on_line": "m", "in_region": "dimensionless",
        "rotation_progress": "rad",
    }
    event_channels = {
        "object_held": "object_held_endpoint_for_declared_body",
        "not_held": "object_held_endpoint_for_declared_body",
        "tool_mediation": "executed_tool_contact_trace",
    }
    measurements = []
    for index, predicate in enumerate(stage.running):
        term_type = predicate.type
        references = list(predicate.referenced_anchors())
        missing = list(dict.fromkeys(anchors.missing(
            references + list(measurement_channels(predicate))
        )))
        residual = None
        satisfied = None
        source = "unavailable"
        if term_type not in scalar_units:
            missing.append(event_channels.get(
                term_type, "unsupported_measured_predicate:" + term_type,
            ))
        elif not missing:
            try:
                value = float(predicate.raw_residual(anchors))
                if np.isfinite(value):
                    satisfied_now = bool(predicate.satisfied(anchors))
                    residual = value
                    satisfied = satisfied_now
                    source = (
                        "acknowledged_command"
                        if term_type == "gripper_command"
                        else "measured_anchor_state"
                    )
                else:
                    missing.append("nonfinite_measured_residual")
            except Exception:  # noqa: BLE001 - telemetry must not change control
                missing.append("unevaluable_measured_predicate:" + term_type)
        measurements.append({
            "term_index": index,
            "type": term_type,
            "anchors": references,
            "residual": residual,
            "satisfied": satisfied,
            "measured": residual is not None,
            "unmeasured_channels": missing,
            "source": source,
            "unit": scalar_units.get(
                term_type,
                "dimensionless" if term_type in event_channels else "unknown",
            ),
            "residual_kind": "raw_table_i_residual",
        })
    evidence["running_measurements"] = measurements

__all__ = [
    "EpisodeResult",
    "apply_profile_sealed_episode_success",
    "run_mpc_episode",
]


def _cycle_budget_for_stage(
    *,
    max_cycles: int,
    budget_scope: str,
    cycles_already_executed: int,
    optimizer: str,
) -> int:
    """Return this stage's allowance under a stage or episode-wide budget."""
    if optimizer == "mppi" and budget_scope == "episode":
        return max(0, int(max_cycles) - int(cycles_already_executed))
    return int(max_cycles)

def _held_from_trace_or_world(
    execution_contact: dict[str, Any],
    world: Any,
) -> bool:
    """Read held state only from the executed MJWarp contact trace.

    ``world`` remains in the private helper signature for call-site stability,
    but production never runs host contact physics as a fallback.
    """
    del world
    traced_now = execution_contact.get("subject_two_sided_now")
    if traced_now is None:
        raise RuntimeError(
            "MJWarp execution trace omitted subject_two_sided_now; "
            "refusing CPU contact fallback"
        )
    return bool(traced_now)


def _held_from_sim_execution_evidence(
    execution_contact: dict[str, Any],
    execution_evidence: dict[str, Any],
    world: Any,
    *,
    mode: str,
) -> bool | None:
    """Read the selected simulator held channel for a stage transition."""
    if mode == SIM_HELD_EVIDENCE_FROZEN:
        return _held_from_trace_or_world(execution_contact, world)
    if mode not in (
        SIM_HELD_EVIDENCE_MEASURED_V1,
        SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1,
    ):
        raise ValueError(f"unsupported simulator held evidence mode {mode!r}")
    measured = execution_evidence.get("sim_held_evidence")
    if not isinstance(measured, dict):
        return None
    if not bool(measured.get("observable", False)):
        return None
    held = measured.get("subject_held")
    if not isinstance(held, (bool, np.bool_)):
        return None
    return bool(held)


def _terminal_gate_tol_floored_stage(stage: Any, floor: float) -> Any:
    """Addendum 103: stage-advance gate copy with metric terminal tolerances
    lifted to ``floor`` (metres). Only positive sub-floor ``tol`` fields are
    lifted -- ``tol == 0`` terms keep their exact semantics, angular
    ``tol_deg`` terms are out of scope, and TemporalHold lifts its inner
    predicate. The original stage keeps pricing the cost and grading the
    verdict; the copy exists solely for "may the stage exit"."""
    import dataclasses as _dc

    def lift(p: Any) -> Any:
        inner = getattr(p, "inner", None)
        if inner is not None:
            lifted = lift(inner)
            return _dc.replace(p, inner=lifted) if lifted is not inner else p
        tol = getattr(p, "tol", None)
        if isinstance(tol, (int, float)) and 0.0 < float(tol) < floor:
            return _dc.replace(p, tol=float(floor))
        return p

    lifted_terminal = [lift(p) for p in stage.terminal]
    if all(a is b for a, b in zip(lifted_terminal, stage.terminal)):
        return stage
    return _dc.replace(stage, terminal=lifted_terminal)


def _is_hold_acquiring_stage(program: Any, stage: Any) -> bool:
    """Addendum 105: the FIRST stage whose terminal asserts object_held is
    the acquisition stage. The selective gate floor exempts it -- an early
    exit there means a marginal grip (toolpush floor-era rolls 0/3 vs 2/2
    without the floor). Later hold-asserting stages (carry/push while held)
    keep the floor."""
    def asserts_held(st: Any) -> bool:
        for p in getattr(st, "terminal", ()):
            if getattr(p, "type", None) == "object_held":
                return True
            inner = getattr(p, "inner", None)
            if inner is not None and getattr(inner, "type", None) == "object_held":
                return True
        return False

    for st in getattr(program, "stages", ()):
        if asserts_held(st):
            return st is stage
    return False


def _current_terminal_contact_satisfied(
    stage: Any,
    anchors: Any,
    current_execution_contact: Any,
) -> tuple[bool, bool | None]:
    """Resolve terminal Contact only from this executed prefix's E-1 map.

    The Boolean in ``subject_target_now_by_body`` is produced at the selected
    prefix endpoint.  Cumulative frame counts, penetration maxima, and the
    rollout's H-1 terminal map are intentionally not accepted here.
    """
    from oap.program.verdict import (
        stage_contact_target_bodies,
        terminal_contact_requirement,
    )

    requirement = terminal_contact_requirement(stage, anchors)
    if requirement is None:
        return False, None
    target = anchors.get(requirement.b)
    target_body = (
        getattr(target, "attached_to", None)
        if target is not None
        else None
    )
    if not target_body or target_body == "subject":
        return True, None
    target_body = str(target_body)
    try:
        bound = stage_contact_target_bodies(
            stage,
            anchors,
            (target_body,),
        )
    except ValueError:
        return True, None
    if bound != (target_body,):
        return True, None
    if not isinstance(current_execution_contact, dict):
        return True, None
    endpoint_by_body = current_execution_contact.get(
        "subject_target_now_by_body"
    )
    if not isinstance(endpoint_by_body, dict):
        return True, None
    endpoint_names = tuple(str(name) for name in endpoint_by_body)
    if endpoint_names != (target_body,):
        return True, None
    value = endpoint_by_body.get(target_body)
    if not isinstance(value, (bool, np.bool_)):
        return True, None
    return True, bool(value)


class EpisodeResult:
    """The outcome of a joint-space episode in sim."""

    def __init__(self, success: bool, obj_pose7: np.ndarray,
                 stage_results: list[Any], outcome: str, *,
                 held_close_width_m: float | None = None,
                 held_status: bool | None = False,
                 close_was_commanded: bool = False,
                 tool_mediation_valid: bool | None = None) -> None:
        self.terminal_success = bool(success)
        self.success = success
        self.obj_pose7 = obj_pose7
        self.stage_results = stage_results          # per-stage receding-horizon logs
        self.outcome = outcome
        self.held_close_width_m = held_close_width_m
        self.held_status = held_status
        self.close_was_commanded = bool(close_was_commanded)
        self.tool_mediation_valid = tool_mediation_valid

    @property
    def chunks(self) -> int:
        """Number of actual MPC cycles, including refused no-valid cycles."""
        return sum(len(stage_cycles) for stage_cycles in self.stage_results)

    @property
    def tool_use_success(self) -> bool | None:
        """Backward-compatible tool-use diagnostic, never episode success."""
        if self.tool_mediation_valid is None:
            return None
        return bool(self.success and self.tool_mediation_valid)


def apply_profile_sealed_episode_success(
    result: EpisodeResult,
    unified_mppi_effort_profile: str | None,
) -> EpisodeResult:
    """Apply the canonical toolpush mechanism gate without rewriting geometry.

    ``terminal_success`` remains the measured TaskProgram fact.  Only the new
    sealed toolpush profile promotes that fact to episode success when exact
    accumulated execution evidence also proves eraser contact and zero direct
    robot-target contact.  Existing profiles retain their historical result
    semantics byte-for-byte.
    """
    from oap.twin.unified_mppi_effort import (
        unified_mppi_effort_requires_tool_mediation_success,
    )

    if (
        unified_mppi_effort_requires_tool_mediation_success(
            unified_mppi_effort_profile
        )
        and result.terminal_success
        and result.tool_mediation_valid is not True
    ):
        result.success = False
        result.outcome = "CANNOT_VERIFY_TOOL_MEDIATION"
    return result


def _displacement_initial_centers(program: Any, subject: Any,
                                  reference: Any) -> tuple[str, ...]:
    """Initial centers used by a displacement term, including role aliases."""
    from oap.program.predicates import SignedAxisGap, TemporalHold

    names: set[str] = set()
    for stage in program.stages:
        for predicate in (*stage.running, *stage.terminal):
            inner = predicate.inner if isinstance(predicate, TemporalHold) else predicate
            if isinstance(inner, SignedAxisGap):
                names.update(name for name in inner.referenced_anchors()
                             if name.endswith("_initial_center"))
    for role, obj in (("subject", subject), ("target", reference)):
        if obj is not None:
            aliases = {f"{role}_initial_center", f"{obj.name}_initial_center"}
            if names & aliases:
                names.update(aliases)
    return tuple(sorted(names))


def _bind_initial_centers(initial_anchors: Any, current_anchors: Any,
                          names: tuple[str, ...], *consumers: Any) -> None:
    """Bind the displacement datum without changing current or angular anchors.

    ``consumers`` includes the stage cost-scale set passed by reference into
    the current MPC loop. Subsequent device costs are rebuilt from these same
    updated values; neither a compiled evaluator nor a stage retains old data.
    """
    for target in (initial_anchors, *consumers):
        fresh = current_anchors.copy()
        for name in names:
            if name in fresh:
                target.add(name, fresh[name])


def _anchors_at(subject: Any, objects: Any, obj_pose7: np.ndarray, *,
                band_floor: float, reference: Any = None,
                world: Any = None,
                initial_anchors: Any = None,
                exact_plan_world_grounding: bool = False) -> Any:
    """Ground program anchors at the current measured/planning-world pose.

    The one grounding used by every measurement-side evaluation -- the episode
    verdict AND the per-cycle stage-terminal check -- so they can never
    diverge from each other (they may only diverge from the SCORER, whose
    predicted_end_anchors is a different mechanism by design).

    Only an explicit offline exact-plan permission enables signed material
    frames.  Then the world's free-joint qpos is authoritative: current frames
    are regenerated from that quaternion on every MPC cycle while episode-start
    ``*_initial_*`` anchors remain frozen.  A world object by itself, or a
    serialized/raw quaternion, never grants that permission.
    """
    from oap.loop.observe import (
        anchors_from_observation,
        ground_offline_plan_world,
    )

    obj = np.asarray(obj_pose7, dtype=float)
    if exact_plan_world_grounding:
        if world is None:
            raise ValueError(
                "exact plan-world grounding requires the current world"
            )
        address = int(world.object_qpos_addr)
        exact_obj = np.asarray(
            world.data.qpos[address:address + 7],
            dtype=float,
        )
        if exact_obj.shape != (7,) or not np.all(np.isfinite(exact_obj)):
            raise RuntimeError("current plan-world subject pose is invalid")
        if not np.array_equal(exact_obj, obj):
            raise RuntimeError(
                "MPC subject pose diverged from current plan-world qpos"
            )
        _, aset = ground_offline_plan_world(
            SimpleNamespace(offline=True),
            objects,
            subject,
            world,
            tag="mpc_current_plan_world",
            table_top_z=band_floor,
            reference=reference,
            initial_anchors=initial_anchors,
        )
    else:
        w, x, y, z = obj[3], obj[4], obj[5], obj[6]
        yaw = float(
            np.arctan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        )
        obs = {
            "schema": "sim_plan_world_observation",
            "pos_base": [float(v) for v in obj[:3]],
            "yaw_base_rad": yaw,
            "quat_wxyz": [float(v) for v in obj[3:7]],
        }
        aset = anchors_from_observation(
            obs,
            subject,
            objects,
            table_top_z=band_floor,
            reference=reference,
        )
        # Preserve the episode-start datum when re-grounding a current pose.
        # Recreating ``*_initial_center`` from x(t) would make every
        # displacement predicate compare x(t) against itself and never end.
        if initial_anchors is not None:
            frozen = initial_anchors.copy()
            for name in frozen.names():
                if (
                    getattr(frozen[name], "kind", None) == "initial"
                    or "_initial_" in name
                ):
                    aset.add(name, frozen[name])
    if world is not None:
        if not exact_plan_world_grounding:
            _reground_movables(aset, world, subject, objects)
        from oap.program.anchors import Anchor

        point = np.asarray(
            world.data.site_xpos[int(world.site_id)],
            dtype=float,
        ).copy()
        rotation = np.asarray(
            world.data.site_xmat[int(world.site_id)],
            dtype=float,
        ).reshape(3, 3)
        axis = rotation[:, 2].copy()
        closing_axis = rotation[:, 1].copy()
        if (
            point.shape == (3,)
            and np.all(np.isfinite(point))
            and axis.shape == (3,)
            and np.all(np.isfinite(axis))
            and np.linalg.norm(axis) > 1e-9
            and closing_axis.shape == (3,)
            and np.all(np.isfinite(closing_axis))
            and np.linalg.norm(closing_axis) > 1e-9
        ):
            aset.add(
                "gripper_center",
                Anchor(point=point, axis=axis, kind="robot"),
            )
            aset.add(
                "gripper_closing_axis",
                Anchor(point=point, axis=closing_axis, kind="robot"),
            )
        width = _measured_gripper_width(world)
        if width is not None and np.isfinite(width):
            aset.add(
                "gripper_width",
                Anchor(
                    point=np.array([float(width), 0.0, 0.0]),
                    kind="robot",
                ),
            )
        _ground_measured_velocities(aset, world, subject, objects)
    return aset


def _free_body_qvel(world: Any, qpos_addr: int) -> np.ndarray | None:
    """Return the 6-dof [linear(3), angular(3)] qvel of one free joint."""
    model = world.model
    matches = np.flatnonzero(
        np.asarray(model.jnt_qposadr, dtype=int) == int(qpos_addr)
    )
    if matches.size != 1:
        return None
    dof = int(model.jnt_dofadr[int(matches[0])])
    qvel = np.asarray(world.data.qvel[dof:dof + 6], dtype=float)
    if qvel.shape != (6,) or not np.all(np.isfinite(qvel)):
        return None
    return qvel


def _ground_measured_velocities(aset: Any, world: Any, subject: Any,
                                objects: Any) -> None:
    """Ground the ``{body}_velocity`` measured channels from the live twin.

    Offline this is the exact simulator state -- the honest measured velocity
    for ``AtRest``. On hardware no equivalent tracked-velocity channel is
    grounded yet, so the anchors stay absent there and an AtRest terminal
    remains UNKNOWN (fail closed) rather than fabricated.
    """
    from oap.program.anchors import Anchor

    def add(prefix: str, qvel: np.ndarray) -> None:
        aset.add(f"{prefix}_velocity",
                 Anchor(point=qvel[:3].copy(), kind="measurement"))
        aset.add(f"{prefix}_angular_velocity",
                 Anchor(point=qvel[3:6].copy(), kind="measurement"))

    subject_qvel = _free_body_qvel(world, int(world.object_qpos_addr))
    if subject_qvel is not None:
        add("subject", subject_qvel)
        subject_name = getattr(subject, "name", None)
        if subject_name and subject_name != "subject":
            add(str(subject_name), subject_qvel)
    for name, qpos_addr in (
        getattr(world, "movable_qpos_addr", None) or {}
    ).items():
        if name == getattr(subject, "name", None):
            continue
        body_qvel = _free_body_qvel(world, int(qpos_addr))
        if body_qvel is not None:
            add(str(name), body_qvel)


def _ground_robot_subject_contact_measurement(
    anchors: Any,
    world: Any,
) -> None:
    """Add the exact current MuJoCo robot-subject contact measurement.

    This helper is used only by the offline physical execution lane. A live
    planning twin is not a contact sensor and therefore never supplies this
    measured terminal channel for hardware execution.
    """
    from oap.program.anchors import Anchor

    touching = current_robot_subject_contact(
        world.model,
        world.data,
        refresh=True,
    )
    anchors.add(
        "robot_subject_contact",
        Anchor(
            point=np.array([1.0 if touching else 0.0, 0.0, 0.0]),
            kind="measurement",
        ),
    )


def _stage_needs_robot_subject_contact_measurement(stage: Any) -> bool:
    """Whether this Stage terminal reads the exact robot-subject channel."""
    from oap.program.verdict import measurement_channels

    return any(
        "robot_subject_contact" in measurement_channels(predicate)
        for predicate in stage.terminal
    )


def _apply_observed_anchor_reliability(
    anchors: Any,
    stage: Any,
    evidence: dict[str, Any],
    *,
    tau_reliable: float,
) -> None:
    """Attach measured confidence/visibility before a real terminal gate.

    Structural world anchors and measured robot-FK anchors do not come from the
    vision tracker. Every other terminal anchor must have an explicit
    observation entry; a missing entry is set unreliable rather than inheriting
    ``Anchor``'s simulation-friendly confidence default of one.
    """
    if float(tau_reliable) <= 0.0:
        return
    mapping = evidence.get("anchor_reliability")
    mapping = mapping if isinstance(mapping, dict) else {}
    required: list[str] = []
    for predicate in stage.terminal:
        required.extend(predicate.referenced_anchors())
    for name in dict.fromkeys(required):
        anchor = anchors.get(name)
        if anchor is None or getattr(anchor, "kind", None) in {"world", "robot"}:
            continue
        row = mapping.get(name)
        if not isinstance(row, dict):
            anchor.confidence = 0.0
            anchor.visibility = 0.0
            continue
        try:
            confidence = float(row["confidence"])
            visibility = float(row["visibility"])
        except (KeyError, TypeError, ValueError):
            confidence = visibility = 0.0
        if (
            not np.isfinite(confidence)
            or not np.isfinite(visibility)
            or not 0.0 <= confidence <= 1.0
            or not 0.0 <= visibility <= 1.0
        ):
            confidence = visibility = 0.0
        anchor.confidence = confidence
        anchor.visibility = visibility


def _terminal_reliability_threshold(
    observation_readiness: Any,
    evidence: dict[str, Any],
) -> float:
    """Use a current, identity-backed maskless track without inventing a score.

    FoundationPose ``track_one`` returns a pose but no calibrated scalar
    confidence.  Once the independent registration, freshness, synchronized
    capture, and tracker-validity gates have all passed, terminal predicates
    evaluate that measured pose directly.  This returns zero only to avoid
    fabricating a confidence score; it does not claim a statistical pose-error
    bound.  Missing or invalid tracking evidence retains the configured
    reliability threshold and therefore fails closed.
    """
    configured = float(
        getattr(
            observation_readiness,
            "min_terminal_anchor_reliability",
            0.0,
        )
    )
    tracking_state = evidence.get("tracking_state")
    registration = evidence.get("mask_scored_registration")
    if (
        evidence.get("tracking_state_valid") is True
        and isinstance(tracking_state, dict)
        and tracking_state.get("state_valid") is True
        and isinstance(registration, dict)
        and isinstance(registration.get("capture_id"), str)
    ):
        return 0.0
    return configured


def ground_body_at_pose(aset: Any, name: str, pose7: np.ndarray, obj: Any) -> None:
    """Write one body's full anchor vocabulary at a given world pose.

    THE single implementation, shared by the measurement side (which reads the
    pose out of the live sim) and the SCORER (which reads it out of a candidate
    rollout's end state). Two copies of this is precisely how the wall-thickness
    reader and writer drifted apart into a 4x mass disagreement, so there is
    one function and two callers.
    """
    from oap.loop.observe import _body_anchors, _canonical_frame
    from oap.program.anchors import Anchor

    pose = np.asarray(pose7, dtype=float)
    centre = pose[:3].copy()
    R = _canonical_frame(pose[3:7], getattr(obj, "q_om", None))
    L, W, H = obj.size_lwh
    aset.add(name, Anchor(point=centre.copy(), kind="region", dynamic=True,
                          attached_to=name,
                          region_half=np.array([L / 2.0, W / 2.0, max(H / 2.0, 0.5)])))
    aset.add(f"{name}_center", Anchor(
        point=centre.copy(), kind="object", dynamic=True, attached_to=name))
    for k, v in _body_anchors(name, centre, R, obj.size_lwh).items():
        v.dynamic = True          # this body moves; the gate must know
        v.attached_to = name      # owner -> live qvel for the unified rest gate
        aset.add(k, v)


def movable_grounder(subject: Any, objects: Any) -> Any:
    """A closure the SCORER can call to ground non-subject movers.

    The optimizer must not learn about SceneObject -- it scores rows, not
    scenes -- so the caller that already owns the object list hands it this
    hook instead. Without it a push goal is INERT to the optimizer: every
    candidate scores the same because the only body the scorer moves is the
    subject, so there is no gradient to follow and the pre-motion gate refuses
    the program. (The gate refusing is correct behaviour, not a workaround: it
    is what stops a silent no-op from being reported as a plan.)
    """
    subject_name = getattr(subject, "name", None)
    # An explicit sequence check, not `objects or []`: callers that have no
    # scene (tests, the single-mover path) pass a sentinel, and grounding
    # nothing is the correct answer there -- iterating it is a crash.
    seq = objects if isinstance(objects, (list, tuple)) else ()
    by_name = {getattr(o, "name", None): o for o in seq}

    def ground(aset: Any, poses: dict[str, np.ndarray]) -> None:
        for name, pose7 in (poses or {}).items():
            obj = by_name.get(name)
            if obj is None or name == subject_name:
                continue
            ground_body_at_pose(aset, name, pose7, obj)

    return ground


def _reground_movables(aset: Any, world: Any, subject: Any, objects: Any) -> None:
    """Re-ground every NON-SUBJECT movable body from the live sim state.

    A pushed box's pose is not predicted, it is SIMULATED -- so the anchors
    that grade it must be read back from the physics, exactly the way the
    subject's are. Without this the goal is inert: the outcome model never
    moves the box, so the terminal residual cannot respond to anything the
    robot does, and the pre-motion gate refuses the program (correctly, until
    the body can actually move).
    """
    addrs = getattr(world, "movable_qpos_addr", None) or {}
    if not addrs:
        return
    subject_name = getattr(subject, "name", None)
    for obj in (objects or []):
        name = getattr(obj, "name", None)
        if not name or name == subject_name or name not in addrs:
            continue
        adr = int(addrs[name])
        ground_body_at_pose(aset, name, world.data.qpos[adr:adr + 7], obj)


def _measured_gripper_width(world: Any) -> float | None:
    """Read the live virtual finger-width joint, if the model exposes it."""
    robot = getattr(world, "robot", None) or {}
    joint_ids = robot.get("gripper_joint_ids", {})
    jid = joint_ids.get("finger_width")
    if jid is None:
        return None
    qpos_addr = int(world.model.jnt_qposadr[int(jid)])
    return float(world.data.qpos[qpos_addr])


def _ground_verdict(program: Any, subject: Any, objects: Any, obj_pose7: np.ndarray,
                    *, band_floor: float, reference: Any = None,
                    episode: Any = None, world: Any = None,
                    exact_plan_world_grounding: bool = False,
                    execution_contact_valid: bool | None = None,
                    execution_contact: dict[str, Any] | None = None,
                    stage_evidence: Any = None,
                    initial_anchors: Any = None,
                    anchor_reliability: dict[str, Any] | None = None,
                    tau_reliable: float = 0.0,
                    ground_offline_robot_subject_contact: bool = False,
                    ) -> Any:
    """The typed program outcome on the already synchronized twin state.

    This is a pure computation and never calls a camera, tracker, or observation
    callback. Ground the anchors from the ACTUAL object pose (position + yaw,
    matching the EE path's
    :func:`~oap.loop.observe.anchors_from_observation`), fed the LIVE sim
    pose already updated from the post-prefix snapshot. The sim freejoint was
    reset to ``subject_pose0`` =
    ``to_plan_pose(obs0)`` (the PLAN frame: ``obs0.pos_base.z + z_real_to_plan`` for
    a live observation), and BOTH the subject pose and the ``table`` band ground at
    that same plan-frame ``band_floor`` -- so the ``above_by`` residual is
    self-consistent and physically correct (a RESTING object correctly FAILS a lift
    check; it rises as the object is lifted). It equals stage-0 ``anchors0`` exactly
    only when ``z_real_to_plan == 0`` (offline / sim dry-run); otherwise this is the
    pure-plan value while ``anchors0`` is grounded in a mixed frame.

    This deliberately does NOT use ``predicted_end_anchors`` (the candidate SCORER's
    mechanism): that snaps the subject to ``table + 0.5*height + max(0, lift)``, so
    a tall resting object lands at a center height that clears a loose ``above_by``
    band -- certifying a never-grasped object as "lifted". The verdict must reflect
    where the object IS, not where a successful plan would put it.
    """
    from oap.loop import certify as certify_mod

    anchors = _anchors_at(subject, objects, obj_pose7, band_floor=band_floor,
                          reference=reference, world=world,
                          initial_anchors=initial_anchors,
                          exact_plan_world_grounding=(
                              exact_plan_world_grounding
                          ))
    if (
        ground_offline_robot_subject_contact
        and _stage_needs_robot_subject_contact_measurement(program.stages[-1])
    ):
        if world is None:
            raise ValueError(
                "offline robot-subject contact measurement needs a world"
            )
        _ground_robot_subject_contact_measurement(anchors, world)
    _apply_observed_anchor_reliability(
        anchors,
        program.stages[-1],
        {"anchor_reliability": anchor_reliability or {}},
        tau_reliable=tau_reliable,
    )
    return certify_mod.certify(
        program,
        anchors,
        episode=episode,
        execution_contact_valid=execution_contact_valid,
        execution_contact=execution_contact,
        stage_evidence=stage_evidence,
        table_top_z=band_floor,
        tau_reliable=tau_reliable,
        displacement_tolerance_m=None,
    )


def _bind_rotation_progress_reference(anchors0: Any, measured: Any,
                                      bodies: tuple[str, ...], *copies: Any) -> None:
    """Freeze measured episode-start raw frames in each planning reference."""
    from oap.loop.observe import _freeze_initial_anchor
    for body in bodies:
        for letter in "xyz":
            source = f"{body}_raw_frame_{letter}"
            destination = f"{body}_initial_raw_frame_{letter}"
            if measured.missing([source]) or measured[source].axis is None:
                raise ValueError(f"rotation_progress missing exact raw frame {source}")
            for target in (anchors0, *copies):
                if target is not None:
                    target.add(destination, _freeze_initial_anchor(measured[source]))


def _bind_episode_start_datums(
    anchors0: Any, measured_start: Any, displacement_names: tuple[str, ...],
    rotation_bodies: tuple[str, ...],
) -> dict[str, Any]:
    """Freeze the observed initial state before any control prefix executes."""
    missing = measured_start.missing(displacement_names)
    if missing:
        raise ValueError(f"episode-start displacement anchors missing: {missing}")
    _bind_initial_centers(anchors0, measured_start, displacement_names)
    _bind_rotation_progress_reference(anchors0, measured_start, rotation_bodies)
    evidence: dict[str, Any] = {}
    if displacement_names:
        evidence["initial_displacement_datum"] = {
            "source": "episode_start_before_first_control",
            "centers": {name: anchors0.point(name).tolist() for name in displacement_names},
        }
    if rotation_bodies:
        evidence["rotation_progress_reference"] = {
            "source": "episode_start_before_first_control",
            "raw_frames_by_body": {
                body: [anchors0.axis(f"{body}_initial_raw_frame_{a}").tolist() for a in "xyz"]
                for body in rotation_bodies
            },
        }
    return evidence


def run_mpc_episode(
    *, world: Any, plan_xml: Any, program: Any, anchors0: Any,
    obj_pose7: np.ndarray, sub_h: float,
    band_floor: float, rng: np.random.Generator,
    max_stage_cycles: int = 9,
    mppi_cycle_budget_scope: str = "episode",
    horizon_steps: int = DEFAULT_EXECUTION_STEPS,
    num_knots: int = DEFAULT_N_KNOTS,
    sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
    candidate_selection_mode: str = (
        CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
    ),
    mtp_jaw_mode: str = MTP_JAW_MODE_PRESERVE_NOMINAL,
    local_sigma_fraction: float = (
        PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
    ),
    cem_rounds: int = DEFAULT_CEM_ROUNDS,
    cem_elite_fraction: float = DEFAULT_CEM_ELITE_FRACTION,
    cem_min_std_fraction: float = DEFAULT_CEM_MIN_STD_FRACTION,
    optimizer: str = "cem",
    mppi_temperature: float = DEFAULT_MPPI_TEMPERATURE,
    mppi_execution_mode: str = MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    control_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
    pick_v8_causal_profile: str | None = None,
    arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
    record_candidate_cost_telemetry: bool = False,
    table_contact_force_weight: float | None = None,
    control_regularizer_calibration_profile: str | None = None,
    joint_velocity_force_dial_arm: str | None = None,
    joint_velocity_force_phase3_finalist_arm: str | None = None,
    joint_velocity_force_phase3_seed: int | None = None,
    joint_velocity_force_trial_budget: str | None = None,
    cost_shaping_profile: str | None = None,
    executed_action_state: Any = None,
    execution_prefix_fraction: float = DEFAULT_EXECUTION_PREFIX_FRACTION,
    mppi_stage_execution_prefix_steps: tuple[int, ...] | None = None,
    pool_size: int = 2048, renderer: Any = None,
    out_dir: Any = None,
    subject: Any = None, objects: Any = None, reference: Any = None,
    episode: Any = None,
    execute_prefix_fn: Any = None,
    reobserve_fn: Any = None,
    solve_step_fn: Any = None,
    execution_feasibility: dict[str, Any] | None = None,
    observation_readiness: Any = None,
    initial_observation_timestamp_s: float | None = None,
    initial_robot_timestamp_s: float | None = None,
    initial_terminal_evidence: dict[str, Any] | None = None,
    exact_plan_world_grounding: bool = False,
) -> EpisodeResult:
    """Run the joint-space MPC pipeline over the program's stages.

    Uses :func:`~oap.loop.plan.sample_candidates` at each explicit stage
    entry for the measured-boundary/future-controller-hold nominal, then shifts the
    previous winner across later MPC cycles within that stage. A stage change
    is an objective discontinuity, so carrying the old stage's unfinished
    control tape is not a warm start for the new objective. Grounds the program verdict from the
    LIVE sim object pose
    via :func:`_ground_verdict` (the EE path's ``anchors_from_observation`` on the
    sim pose -- faithful, not the scorer's predicted-end). ``program`` may be None
    (then it runs every stage's cycles and reports the physical lift, for a grasp
    smoke test); when ``program`` is set, ``subject`` + ``objects`` MUST be too
    (they ground the verdict). ``episode`` (optional) receives the four call-site
    records -- trajectory_cost + physics_gate (inside the chunk), termination_gate +
    success_verification (here) -- so the joint packet carries the same audit
    trail the EE path does.
    """
    execution_prefix_fraction = validate_execution_prefix_fraction(
        execution_prefix_fraction
    )
    from oap.loop.plan import sample_candidates, sample_mppi_candidates

    if int(max_stage_cycles) < 1:
        raise ValueError("max_stage_cycles must be >= 1")
    mppi_cycle_budget_scope = str(mppi_cycle_budget_scope).strip().lower()
    if mppi_cycle_budget_scope not in {"episode", "stage"}:
        raise ValueError(
            "mppi_cycle_budget_scope must be 'episode' or 'stage'"
        )
    horizon_steps = validate_horizon_steps(horizon_steps)
    num_knots = validate_num_knots(num_knots)
    sigma_fraction = validate_local_sigma_fraction(
        sigma_fraction
    )
    sampler_mode = validate_predictive_sampler_mode(sampler_mode)
    candidate_selection_mode = validate_candidate_selection_mode(
        candidate_selection_mode
    )
    candidate_selection_kwargs = (
        {}
        if candidate_selection_mode
        == CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
        else {"candidate_selection_mode": candidate_selection_mode}
    )
    mtp_jaw_mode = validate_mtp_jaw_mode(mtp_jaw_mode)
    local_sigma_fraction = validate_local_sigma_fraction(
        local_sigma_fraction
    )
    cem_rounds = validate_cem_rounds(cem_rounds)
    cem_elite_fraction = validate_cem_elite_fraction(cem_elite_fraction)
    cem_min_std_fraction = validate_local_sigma_fraction(
        cem_min_std_fraction
    )
    optimizer = str(optimizer).strip().lower()
    if optimizer not in {"cem", "mppi"}:
        raise ValueError("optimizer must be 'cem' or 'mppi'")
    if mppi_stage_execution_prefix_steps is not None:
        mppi_stage_execution_prefix_steps = (
            validate_mppi_stage_execution_prefix_steps(
                mppi_stage_execution_prefix_steps
            )
        )
        if optimizer != "mppi":
            raise ValueError(
                "stage execution-prefix steps are supported only by MPPI"
            )
        if execute_prefix_fn is not None or solve_step_fn is not None:
            raise ValueError(
                "stage execution-prefix steps are offline local-MPPI-only"
            )
    mppi_temperature = validate_mppi_temperature(mppi_temperature)
    mppi_execution_mode = validate_mppi_execution_mode(mppi_execution_mode)
    control_profile = validate_control_profile(control_profile, allow_none=True)
    sim_held_evidence_mode = resolve_sim_held_evidence_mode()
    from oap.twin.unified_mppi_effort import (
        UNIFIED_MPPI_EFFORT_V1,
        validate_unified_mppi_effort_profile_arg,
    )
    unified_mppi_effort_profile = validate_unified_mppi_effort_profile_arg(
        unified_mppi_effort_profile,
        allow_none=True,
    )
    from oap.twin.pick_v8_causal import (
        PICK_V8_SUCCESS_STAGE_PREFIXES,
        pick_v8_causal_uses_original_prefix_schedule,
        pick_v8_causal_uses_width_jaw,
        validate_pick_v8_causal_profile,
    )
    pick_v8_causal_profile = validate_pick_v8_causal_profile(
        pick_v8_causal_profile,
        allow_none=True,
    )
    if (
        pick_v8_causal_profile is not None
        and unified_mppi_effort_profile is not None
    ):
        raise ValueError("Pick-v8 causal and unified profiles conflict")
    if pick_v8_causal_profile is not None:
        expected_control = (
            "legacy_velocity_width"
            if pick_v8_causal_uses_width_jaw(pick_v8_causal_profile)
            else "legacy_velocity_effort"
        )
        expected_schedule = (
            PICK_V8_SUCCESS_STAGE_PREFIXES
            if pick_v8_causal_uses_original_prefix_schedule(
                pick_v8_causal_profile
            )
            else None
        )
        if (
            control_profile != expected_control
            or mppi_stage_execution_prefix_steps != expected_schedule
        ):
            raise ValueError("Pick-v8 causal runtime contract mismatch")
    if unified_mppi_effort_profile == UNIFIED_MPPI_EFFORT_V1 and (
        control_profile != "joint_velocity_force"
        or mppi_stage_execution_prefix_steps is not None
    ):
        raise ValueError(
            "unified_mppi_effort_v1 requires JVF effort actuation and one "
            "uniform execution prefix"
        )
    cost_shaping_profile = validate_cost_shaping_profile(
        cost_shaping_profile,
        allow_none=True,
    )
    from oap.twin.control_regularizer import (
        ExecutedActionState,
        validate_control_regularizer_calibration_profile,
    )

    regularizer_profile = validate_control_regularizer_calibration_profile(
        control_regularizer_calibration_profile,
        allow_none=True,
    )
    if regularizer_profile is not None and not isinstance(
        executed_action_state, ExecutedActionState
    ):
        raise ValueError(
            "active control regularizer requires episode ExecutedActionState"
        )
    arm_velocity_weight = validate_arm_velocity_weight(arm_velocity_weight)
    if (
        optimizer != "mppi"
        and mppi_execution_mode != MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
    ):
        raise ValueError("best_valid_sample requires optimizer='mppi'")
    mppi_execution_kwargs = (
        {}
        if mppi_execution_mode == MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
        else {"mppi_execution_mode": mppi_execution_mode}
    )
    if program is not None and (subject is None or objects is None):
        raise ValueError("run_mpc_episode: a program needs subject + objects "
                         "to ground its verdict (anchors_from_observation)")
    if exact_plan_world_grounding and execute_prefix_fn is not None:
        raise ValueError(
            "exact plan-world grounding is simulation-only and cannot be "
            "combined with an execution prefix callback"
        )
    # Initial scene integrity is established by the global first MJWarp batch.
    # There is deliberately no host-contact preflight or CPU physics fallback.
    oadr = int(world.object_qpos_addr)
    obj = np.asarray(obj_pose7, dtype=float).copy()
    held_close_width_m: float | None = None
    held_status: bool | None = False
    close_was_commanded = False
    stage_results: list[Any] = []
    stages = list(program.stages) if program is not None else [None]
    from oap.program.predicates import RotationProgress, TemporalHold
    rotation_predicates = tuple(
        inner for s in stages if s is not None
        for predicate in (*s.running, *s.terminal)
        for inner in (predicate.inner if isinstance(predicate, TemporalHold) else predicate,)
        if isinstance(inner, RotationProgress)
    )
    rotation_bodies = tuple(dict.fromkeys(p.body for p in rotation_predicates))
    if rotation_bodies and not exact_plan_world_grounding:
        raise ValueError("rotation_progress experiment requires exact offline pose grounding")
    if (
        mppi_stage_execution_prefix_steps is not None
        and len(mppi_stage_execution_prefix_steps) != len(stages)
    ):
        raise ValueError(
            "mppi_stage_execution_prefix_steps must provide exactly one "
            f"entry per stage: got {len(mppi_stage_execution_prefix_steps)} "
            f"for {len(stages)} stages"
        )
    stage_prefix_fractions = tuple(
        resolve_execution_prefix(
            horizon_steps=horizon_steps,
            execution_prefix_fraction=execution_prefix_fraction,
            execution_prefix_steps=steps,
        ).effective_fraction
        for steps in (mppi_stage_execution_prefix_steps or ())
    )
    outcome = "MAX_STAGES"
    success = False
    terminal_success = False
    success_verification_attempted = False
    execution_contact_total: dict[str, Any] | None = (
        {
            "subject_movable_frames": 0,
            "robot_movable_frames": 0,
            "max_subject_movable_pen_m": 0.0,
            "max_robot_movable_pen_m": 0.0,
            "subject_two_sided_frames": 0,
            "subject_two_sided_recent": 0,
            "subject_two_sided_now": False,
        }
        if execute_prefix_fn is None
        else None
    )
    # The physical state is authoritative from the first rollout onward.
    # Passing ``None`` made both GPU rollout and sim replay initialize the arm
    # at each sampled knot[0], i.e. teleport away from the measured robot state.
    # Start every candidate from current qpos/qvel; controls may then drive away
    # from that state through ordinary servo dynamics. Carry the resulting state
    # across cycles and stages so an already-held object is never reset.
    world.data.qpos[oadr:oadr + 7] = obj
    import mujoco

    # The object pose was externally observed; refresh body/geom/site pose
    # caches only.  Dry-run dynamics, constraints, and contacts are computed by
    # the selected MJWarp rollout, never by a host ``mj_forward``/``mj_step``.
    mujoco.mj_kinematics(world.model, world.data)
    phys_state: tuple[np.ndarray, np.ndarray] = (
        np.asarray(world.data.qpos, dtype=float).copy(),
        np.asarray(world.data.qvel, dtype=float).copy(),
    )

    next_chunk_idx = 0
    planning_observation_timestamp_s = initial_observation_timestamp_s
    planning_robot_timestamp_s = initial_robot_timestamp_s
    planning_terminal_evidence = dict(initial_terminal_evidence or {})
    displacement_initial_names = (
        _displacement_initial_centers(program, subject, reference)
        if exact_plan_world_grounding and program is not None else ()
    )
    episode_reference_evidence: dict[str, Any] = {}
    if displacement_initial_names or rotation_bodies:
        measured_start = _anchors_at(
            subject, objects, obj, band_floor=band_floor, reference=reference,
            world=world, initial_anchors=None, exact_plan_world_grounding=True,
        )
        episode_reference_evidence = _bind_episode_start_datums(
            anchors0, measured_start, displacement_initial_names, rotation_bodies,
        )
    for stage_idx, stage in enumerate(stages):
        stage_execution_prefix_fraction = (
            stage_prefix_fractions[stage_idx]
            if stage_prefix_fractions
            else execution_prefix_fraction
        )
        if execute_prefix_fn is None:
            held_status = _held_from_trace_or_world(
                execution_contact_total or {},
                world,
            )

        from oap.loop.sampling import _tool_was_used

        def seed_pool_fn(o: np.ndarray) -> Any:
            """Build the one content-free nominal; later cycles only shift it."""
            del o
            candidate_builder = (
                sample_mppi_candidates
                if optimizer == "mppi"
                else sample_candidates
            )
            return candidate_builder(
                world=world,
                num_knots=num_knots,
                **(
                    {
                        "control_profile": control_profile,
                        "pick_v8_causal_profile": pick_v8_causal_profile,
                        "unified_mppi_effort_profile": (
                            unified_mppi_effort_profile
                        ),
                    }
                    if optimizer == "mppi"
                    else {}
                ),
            )

        anchors_stage = None
        reground_anchors_fn = None
        if program is not None:
            # Use the SAME current-world grounding as the measured stage
            # verifier below. ``predicted_end_anchors`` was the legacy
            # single-mover shortcut: it updated the subject but did not mark
            # the pushed box dynamic, so the optimizer saw a flat ordinary
            # terminal cost while the post-execution verifier correctly
            # required tool mediation. One grounding path keeps scorer and
            # verifier structurally identical.
            def reground_anchors_fn(o: np.ndarray) -> Any:
                return _anchors_at(
                    subject,
                    objects,
                    o,
                    band_floor=band_floor,
                    reference=reference,
                    world=world,
                    initial_anchors=anchors0,
                    exact_plan_world_grounding=(
                        exact_plan_world_grounding
                    ),
                )

            anchors_stage = reground_anchors_fn(obj)

        # The history belongs to the entire stage visit, not to a fixed-size
        # attempt.  TemporalHold can therefore become true on any consecutive
        # measured cycles within max_stage_cycles.
        measurement_history: list[Any] = []
        terminal_gate_calls = 0
        terminal_gate_fn = None
        held_stage_dwell = ExecutedSampleDwell(
            required_true_samples=SIM_HELD_DWELL_SAMPLES
        )
        if program is not None:
            from oap.program.verdict import (
                evaluate_stage,
                stage_requires_object_held,
            )

            # The gate and terminal cost use the identical authored thresholds.
            gate_stage = stage
            _gate_floor_raw = os.environ.get(
                "OAP_TERMINAL_GATE_TOL_FLOOR"
            )
            if _gate_floor_raw:
                if float(_gate_floor_raw) != 0.0:
                    raise ValueError("public paper contract forbids terminal tolerance floors")

            def terminal_gate_fn(
                o: np.ndarray,
                execution_contact: dict[str, Any],
                evidence: dict[str, Any] | None = None,
                _stage: Any = gate_stage,
            ) -> Any:
                nonlocal terminal_gate_calls
                evidence = evidence or {}
                pre_solve_gate = terminal_gate_calls == 0
                terminal_gate_calls += 1
                if terminal_gate_calls <= 2:
                    evidence.update(episode_reference_evidence)
                anchors_now = _anchors_at(
                    subject,
                    objects,
                    o,
                    band_floor=band_floor,
                    reference=reference,
                    world=world,
                    initial_anchors=anchors0,
                    exact_plan_world_grounding=(
                        exact_plan_world_grounding
                    ),
                )
                if (
                    execute_prefix_fn is None
                    and _stage_needs_robot_subject_contact_measurement(_stage)
                ):
                    _ground_robot_subject_contact_measurement(
                        anchors_now,
                        world,
                    )
                tau_reliable = _terminal_reliability_threshold(
                    observation_readiness, evidence
                )
                _apply_observed_anchor_reliability(
                    anchors_now,
                    _stage,
                    evidence,
                    tau_reliable=tau_reliable,
                )
                # Ground the acknowledged COMMAND channel so a terminal
                # GripperCommand can be graded. This is the controller's own
                # acknowledgement, not the measured aperture: a Stage must not
                # exit on a transient measured hold whose command has already
                # been released. Absent evidence leaves the anchor missing, so
                # the stage stays UNKNOWN rather than advancing.
                acknowledged = evidence.get("acknowledged_jaw_command")
                if isinstance(acknowledged, dict):
                    from oap.program.anchors import Anchor

                    command_latent = _acknowledged_gripper_command_latent(
                        acknowledged,
                        control_profile=control_profile,
                        pick_v8_causal_profile=pick_v8_causal_profile,
                        unified_mppi_effort_profile=(
                            unified_mppi_effort_profile
                        ),
                    )
                    if command_latent is not None:
                        anchors_now.add(
                            "gripper_command",
                            Anchor(
                                point=np.array([
                                    command_latent,
                                    0.0,
                                    0.0,
                                ]),
                                kind="measurement",
                            ),
                        )
                if not pre_solve_gate:
                    _record_measured_program_state(stage, anchors_now, evidence)
                # One post-prefix tracking snapshot is one measured Stage
                # sample.  The same snapshot updates the twin and grades the
                # terminal; there is no second verdict observation.
                measurement_history.append(anchors_now)
                history_now = measurement_history
                held_now: bool | None = None
                if stage_requires_object_held(_stage):
                    if execute_prefix_fn is not None:
                        held_now = (
                            evidence.get("subject_held")
                            if bool(evidence.get(
                                "grasp_evidence_observable", False
                            ))
                            else None
                        )
                    else:
                        raw_held = _held_from_sim_execution_evidence(
                            execution_contact,
                            evidence,
                            world,
                            mode=sim_held_evidence_mode,
                        )
                        held_now = raw_held
                        if (
                            sim_held_evidence_mode
                            == SIM_HELD_EVIDENCE_MEASURED_V1
                        ):
                            held_now = held_stage_dwell.observe(
                                raw_held,
                                sample_id=evidence.get(
                                    "executed_prefix_sample_id"
                                ),
                            )
                            evidence["sim_held_stage_gate"] = {
                                "raw_subject_held": raw_held,
                                "qualified_subject_held": held_now,
                                "true_streak": int(
                                    held_stage_dwell.true_streak
                                ),
                                "required_true_samples": int(
                                    held_stage_dwell.required_true_samples
                                ),
                                "executed_prefix_sample_id": evidence.get(
                                    "executed_prefix_sample_id"
                                ),
                            }
                tool_ok = _tool_was_used(
                    execution_contact,
                    program=program,
                    stage_idx=stage_idx,
                    anchors=anchors_now,
                )
                terminal_contact_required, terminal_contact_now = (
                    _current_terminal_contact_satisfied(
                        _stage,
                        anchors_now,
                        (
                            None
                            if pre_solve_gate
                            else evidence.get("contact_measurement")
                        ),
                    )
                )
                evaluation_kwargs: dict[str, Any] = {}
                if terminal_contact_required:
                    evaluation_kwargs["terminal_contact_satisfied"] = (
                        terminal_contact_now
                    )
                evaluation = evaluate_stage(
                    _stage,
                    history_now,
                    tau_reliable=tau_reliable,
                    rest_satisfied=None,
                    tool_mediation_satisfied=tool_ok,
                    object_held_satisfied=held_now,
                    table_top_z=band_floor,
                    displacement_tolerance_m=None,
                    **evaluation_kwargs,
                )
                return evaluation

        logger.info(
            "[mpc-episode] stage %d held_at_entry=%s -> receding-horizon "
            "(max %d cycles, execution prefix=%.0f%% of %d-step horizon)",
            stage_idx,
            held_status,
            int(max_stage_cycles),
            100.0 * stage_execution_prefix_fraction,
            horizon_steps,
        )
        stage_cycle_budget = _cycle_budget_for_stage(
            max_cycles=int(max_stage_cycles),
            budget_scope=mppi_cycle_budget_scope,
            cycles_already_executed=int(next_chunk_idx),
            optimizer=optimizer,
        )
        if optimizer == "mppi" and mppi_cycle_budget_scope == "episode":
            # The reference demo has one global 10,000-step environment loop,
            # not a fresh 10,000-step allowance for every semantic stage.
            if stage_cycle_budget <= 0:
                success = False
                outcome = "EPISODE_STEP_LIMIT"
                break
        regularizer_stage_kwargs = (
            {}
            if regularizer_profile is None
            else {
                "control_regularizer_calibration_profile": regularizer_profile,
                "executed_action_state": executed_action_state,
            }
        )
        dial_stage_kwargs = (
            {}
            if joint_velocity_force_dial_arm is None
            else {
                "joint_velocity_force_dial_arm": (
                    joint_velocity_force_dial_arm
                )
            }
        )
        phase3_stage_kwargs = (
            {}
            if joint_velocity_force_phase3_finalist_arm is None
            else {
                "joint_velocity_force_phase3_finalist_arm": (
                    joint_velocity_force_phase3_finalist_arm
                ),
                "joint_velocity_force_phase3_seed": (
                    joint_velocity_force_phase3_seed
                ),
            }
        )
        unified_stage_kwargs = (
            {}
            if unified_mppi_effort_profile is None
            else {
                "unified_mppi_effort_profile": unified_mppi_effort_profile
            }
        )
        if pick_v8_causal_profile is not None:
            unified_stage_kwargs["pick_v8_causal_profile"] = (
                pick_v8_causal_profile
            )
        cycles = run_receding_horizon(
            world=world,
            plan_xml=plan_xml,
            seed_pool_fn=seed_pool_fn,
            obj_pose7=obj,
            rng=rng,
            n_cycles=stage_cycle_budget,
            horizon_steps=horizon_steps,
            sigma_fraction=sigma_fraction,
            sampler_mode=sampler_mode,
            mtp_jaw_mode=mtp_jaw_mode,
            local_sigma_fraction=local_sigma_fraction,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            optimizer=optimizer,
            mppi_temperature=mppi_temperature,
            control_profile=control_profile,
            **unified_stage_kwargs,
            arm_velocity_weight=arm_velocity_weight,
            record_candidate_cost_telemetry=(
                record_candidate_cost_telemetry
            ),
            table_contact_force_weight=table_contact_force_weight,
            cost_shaping_profile=cost_shaping_profile,
            joint_velocity_force_trial_budget=(
                joint_velocity_force_trial_budget
            ),
            **regularizer_stage_kwargs,
            **dial_stage_kwargs,
            **phase3_stage_kwargs,
            execution_prefix_frac=stage_execution_prefix_fraction,
            pool_size=pool_size,
            program=program,
            stage_idx=stage_idx,
            anchors=anchors_stage,
            cost_scale_anchors=anchors_stage,
            sub_h=sub_h,
            renderer=renderer,
            out_dir=(
                Path(out_dir) / f"stage_{stage_idx:02d}"
                if out_dir
                else None
            ),
            chunk_idx0=next_chunk_idx,
            start_state=phys_state,
            # Each explicit VLM stage is a new objective. Start it from the
            # current controller target; only receding cycles of that same
            # objective shift the previous winner.
            episode=episode,
            terminal_gate_fn=terminal_gate_fn,
            reground_anchors_fn=reground_anchors_fn,
            movable_grounder=movable_grounder(subject, objects),
            execution_contact_prior=execution_contact_total,
            execute_prefix_fn=execute_prefix_fn,
            reobserve_fn=reobserve_fn,
            solve_step_fn=solve_step_fn,
            execution_feasibility=execution_feasibility,
            observation_readiness=observation_readiness,
            initial_observation_timestamp_s=(
                planning_observation_timestamp_s
            ),
            initial_robot_timestamp_s=planning_robot_timestamp_s,
            initial_terminal_evidence=planning_terminal_evidence,
            controlled_subject_contact_evidence_name=(
                str(subject.name)
                if subject is not None and getattr(subject, "name", None)
                else None
            ),
            **candidate_selection_kwargs,
            **mppi_execution_kwargs,
        )
        stage_results.append(cycles)
        next_chunk_idx += len(cycles)

        last_cycle = cycles[-1]
        if program is not None and last_cycle.step is None:
            from oap.program import CallSite, log_call_site

            for site in (
                CallSite.TRAJECTORY_COST,
                CallSite.PHYSICS_GATE,
            ):
                log_call_site(
                    episode,
                    site,
                    program,
                    payload={
                        "stage_index": int(stage_idx),
                        "stage_name": stage.name,
                        "evaluated": False,
                        "reason": "measured_terminal_true_before_solve",
                    },
                )
        planning_terminal_evidence = dict(
            getattr(last_cycle, "execution_evidence", None) or {}
        )
        if (
            observation_readiness is not None
            and execute_prefix_fn is not None
            and last_cycle.scene_observation is not None
        ):
            planning_observation_timestamp_s = (
                last_cycle.scene_observation.get("capture_timestamp_s")
            )
            last_evidence = dict(
                getattr(last_cycle, "execution_evidence", None) or {}
            )
            planning_robot_timestamp_s = last_evidence.get(
                "robot_state_timestamp_s"
            )
        if last_cycle.execution_contact_cumulative is not None:
            execution_contact_total = dict(
                last_cycle.execution_contact_cumulative
            )
        terminal_gate_status = last_cycle.terminal_gate_status
        stage_evidence = getattr(last_cycle, "terminal_gate_evidence", None)
        stop_reason = getattr(last_cycle, "stop_reason", None)

        # The driver leaves world.data at the executed (or explicitly restored
        # refused) physical state, which is the sole state carried onward.
        phys_state = (world.data.qpos.copy(), world.data.qvel.copy())
        obj = np.asarray(
            world.data.qpos[oadr:oadr + 7], dtype=float
        ).copy()

        if execute_prefix_fn is None:
            held_status = _held_from_trace_or_world(
                execution_contact_total or {},
                world,
            )
            measured_width = _measured_gripper_width(world)
            held_close_width_m = (
                measured_width if held_status is True else None
            )
        else:
            evidence = dict(
                getattr(last_cycle, "execution_evidence", None) or {}
            )
            events = list(evidence.get("gripper_events") or [])
            if events:
                close_was_commanded = bool(
                    close_was_commanded
                    or any(
                        event.get("force_n") is not None
                        and float(event["force_n"]) > 0.0
                        for event in events
                    )
                )
            elif evidence.get("gripper_commanded_closed") is True:
                close_was_commanded = True
            else:
                last_effort = evidence.get(
                    "last_commanded_gripper_effort_n"
                )
                if (
                    last_effort is not None
                    and float(last_effort) > 0.0
                ):
                    close_was_commanded = True

            held_observable = bool(
                evidence.get("grasp_evidence_observable", False)
            )
            observed_held = (
                evidence.get("subject_held")
                if held_observable
                else None
            )
            measured_width_raw = evidence.get("gripper_width_m")
            measured_width = (
                None
                if measured_width_raw is None
                else float(measured_width_raw)
            )
            held_status = (
                None
                if observed_held is None
                else bool(observed_held)
            )
            if held_status is True:
                if measured_width is not None:
                    held_close_width_m = measured_width
            elif held_status is False:
                held_close_width_m = None

        if program is not None:
            from oap.program import CallSite, log_call_site

            log_call_site(
                episode,
                CallSite.TERMINATION_GATE,
                program,
                payload={
                    "stage_index": int(stage_idx),
                    "stage_name": stage.name if stage is not None else "",
                    "held": held_close_width_m is not None,
                    "obj_z": float(obj[2]),
                    "terminal_gate_status": terminal_gate_status,
                    "cycles": len(cycles),
                    "last_chunk_index": int(last_cycle.cycle),
                    "stop_reason": stop_reason,
                },
            )

        if stop_reason is not None:
            if stop_reason == "no_valid_plan":
                outcome = f"STAGE_NO_VALID_PLAN:{stage_idx}"
            else:
                outcome = (
                    f"STAGE_HARD_FAILURE:{stage_idx}:{stop_reason}"
                )
            success = False
            break

        # Final verification may corroborate only a positively completed final
        # stage; it cannot bypass a false/unknown prerequisite.
        if (
            program is not None
            and terminal_gate_status is True
            and stage_idx == len(stages) - 1
        ):
            success_verification_attempted = True
            task_execution_valid = _tool_was_used(
                execution_contact_total or {},
                program=program,
                stage_idx=stage_idx,
                anchors=anchors0,
                terminal=True,
            )
            final_execution_evidence = dict(
                getattr(last_cycle, "execution_evidence", None) or {}
            )
            v = _ground_verdict(
                program,
                subject,
                objects,
                obj,
                band_floor=band_floor,
                reference=reference,
                episode=episode,
                world=world,
                exact_plan_world_grounding=exact_plan_world_grounding,
                execution_contact_valid=task_execution_valid,
                execution_contact=execution_contact_total,
                stage_evidence=stage_evidence,
                initial_anchors=anchors0,
                anchor_reliability=dict(
                    final_execution_evidence.get("anchor_reliability") or {}
                ),
                tau_reliable=_terminal_reliability_threshold(
                    observation_readiness, final_execution_evidence
                ),
                ground_offline_robot_subject_contact=(
                    execute_prefix_fn is None
                ),
            )
            logger.info(
                "[mpc-episode] stage %d verdict=%s obj_z=%.3f",
                stage_idx,
                getattr(v.outcome, "value", v.outcome),
                float(obj[2]),
            )
            if episode is not None:
                try:
                    measured_rows = dict(
                        (
                            getattr(
                                last_cycle,
                                "execution_evidence",
                                None,
                            )
                            or {}
                        ).get("measured_movable_state_after", {})
                    )
                    movable_pose7 = (
                        {
                            str(name): [
                                float(x) for x in row["pose7"]
                            ]
                            for name, row in measured_rows.items()
                            if row.get("pose7") is not None
                        }
                        if measured_rows
                        else {
                            name: [
                                float(x)
                                for x in world.data.qpos[
                                    int(addr):int(addr) + 7
                                ]
                            ]
                            for name, addr in (
                                getattr(
                                    world, "movable_qpos_addr", None
                                )
                                or {}
                            ).items()
                        }
                    )
                    episode.record_verdict(
                        v,
                        chunk_idx=int(last_cycle.cycle),
                        context={
                            "stage_index": int(stage_idx),
                            "stage_name": (
                                program.stages[stage_idx].name
                                if stage_idx < len(program.stages)
                                else ""
                            ),
                            "subject_pose7": [
                                float(x)
                                for x in np.asarray(obj).ravel()[:7]
                            ],
                            "movable_pose7": movable_pose7,
                            "execution_contact": (
                                None
                                if execution_contact_total is None
                                else dict(execution_contact_total)
                            ),
                            "execution_tool_mediation_valid":
                                task_execution_valid,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[mpc-episode] stage verdict not recorded; "
                        "the run continues",
                        exc_info=True,
                    )
            measured_outcome = str(
                getattr(v.outcome, "value", v.outcome)
            )
            if measured_outcome == "VERIFIED_SUCCESS":
                terminal_success = True
                success, outcome = True, "VERIFIED_SUCCESS"
                break
            terminal_success = False
            success, outcome = False, measured_outcome

        if program is not None and terminal_gate_status is not True:
            label = (
                "UNSATISFIED"
                if terminal_gate_status is False
                else "UNKNOWN"
            )
            logger.warning(
                "[mpc-episode] stage %d evaluation %s after %d continuous "
                "cycle(s) -- stopping, no cold re-entry",
                stage_idx,
                label,
                len(cycles),
            )
            success, outcome = False, f"STAGE_{label}:{stage_idx}"
            break

    if program is not None and not success_verification_attempted:
        # The four-call-site identity is an episode-level audit invariant, even
        # when an earlier physical or stage gate correctly makes success
        # verification ineligible.  Record that explicit non-evaluation instead
        # of leaving a misleading "missing call site" violation.
        from oap.program import CallSite, log_call_site

        log_call_site(
            episode,
            CallSite.SUCCESS_VERIFICATION,
            program,
            payload={
                "eligible": False,
                "reason": outcome,
            },
        )

    tool_mediation_valid: bool | None = None
    if program is not None and program.stages:
        from oap.program.verdict import stage_requires_tool_mediation

        final_stage = program.stages[-1]
        if stage_requires_tool_mediation(final_stage, anchors0):
            from oap.loop.sampling import _tool_was_used

            tool_mediation_valid = _tool_was_used(
                execution_contact_total or {},
                program=program,
                stage_idx=len(program.stages) - 1,
                anchors=anchors0,
                terminal=True,
            )

    result = EpisodeResult(
        success=success,
        obj_pose7=obj,
        stage_results=stage_results,
        outcome=outcome,
        held_close_width_m=held_close_width_m,
        held_status=held_status,
        close_was_commanded=close_was_commanded,
        tool_mediation_valid=tool_mediation_valid,
    )
    result.terminal_success = terminal_success
    return apply_profile_sealed_episode_success(
        result,
        unified_mppi_effort_profile,
    )
