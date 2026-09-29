"""Sim-in-the-loop sampling MPC: physics rollouts during optimization.

Production uses host-orchestrated iterative CEM around a content-free measured
hold or the shifted previous solution. Each round jointly samples all future
arm and binary-gripper latent knots, evaluates full-physics rollouts and
TaskProgram costs on device, and refits a diagonal mean/std from valid elites.
That compact ``K x 8`` refit is copied to the host before the next GPU round.
The final round's lowest-cost valid rollout is selected on device. A one-round
predictive-sampling mode remains only as an equal-budget baseline.

Production has one rollout backend for every stage: MJWarp on GPU, using the
plan XML's exact contact-active convex arm meshes and the real gn01 linkage.
There is no task-family routing and no CPU/no-GPU fallback.  If the GPU bridge,
model build, compilation, or rollout fails, physics validity is not established
and the receding-horizon executor refuses motion. This module has no CPU
rollout route or CPU planner API.

Failure philosophy: any GPU failure leaves ``stats.ran`` false.  The incoming
nominal is retained for diagnostics only; it is not a valid plan and the
execution layer moves zero joints.

The evaluation batch is logged through
:func:`oap.program.hashing.log_call_site` at ``trajectory_cost``. The
call-site label is part of the logged episode schema; with sampling active,
call site 1 records a live optimizer objective, not a post-hoc batch score.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from oap.loop.plan import predicted_end_anchors
from oap.loop.sim_grasp_evidence import (
    GRASP_BRIDGE_TELEMETRY_A191_V1,
    resolve_grasp_bridge_telemetry_mode,
)
from oap.program import Anchor, CallSite, log_call_site, trajectory_cost
from oap.program.cost_shaping import validate_cost_shaping_profile, stage_cost_shaping_plan
from oap.program.interpreter import planning_cost_scales
from oap.program.predicates import (
    AlongAxisGap,
    AngleAboutAxis,
    AxisAngle,
    AxisParallel,
    Contact,
    GripperCommand,
    GripperState,
    InteractionAlign,
    NoContact,
    NotHeld,
    ObjectHeld,
    PointOnLine,
    RobotSubjectContact,
    TemporalHold,
    ToolMediation,
)
from oap.twin import mjx_screen
from oap.twin.batched_rollout import (
    DEFAULT_ACTION_RATE_WEIGHT,
    DEFAULT_ARM_VELOCITY_WEIGHT,
    DEFAULT_CEM_ELITE_FRACTION,
    DEFAULT_CEM_MIN_STD_FRACTION,
    DEFAULT_CEM_ROUNDS,
    DEFAULT_EXECUTION_STEPS,
    DEFAULT_MPPI_TEMPERATURE,
    MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    MTP_JAW_MODE_PRESERVE_NOMINAL,
    PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
    PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
    PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON,
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_SIGMA_FRACTION,
    PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
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
from oap.twin.control_knots import (
    ControlBounds,
    ControlKnots,
)
from oap.twin.control_profile import (
    CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
    control_profile_uses_joint_velocity_arm,
    control_profile_uses_width_jaw,
    joint_velocity_force_device_proposal_seed,
    evaluation_device_seed,
    validate_control_profile,
    validate_joint_velocity_force_phase3_finalist_arm,
    validate_joint_velocity_force_phase3_seed,
)
from oap.twin.types import TABLE_TOP_Z

logger = logging.getLogger("oap.loop.sampling")

__all__ = [
    "SamplerStats",
    "CANDIDATE_SELECTION_MODES",
    "CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST",
    "CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST",
    "POOL_SIZE_ENV",
    "PREFIX_HOLD_BARRIER_ENV",
    "PREFIX_HOLD_BARRIER_FROZEN",
    "PREFIX_HOLD_BARRIER_SOFT_ONLY",
    "perturb_controls",
    "resolve_prefix_hold_barrier_mode",
    "sample_once",
    "pool_size_from_env",
    "validate_candidate_selection_mode",
    "validate_pool_size",
]

POOL_SIZE_ENV = "OAP_POOL_SIZE"
PREFIX_HOLD_BARRIER_ENV = "OAP_PREFIX_HOLD_BARRIER"
PREFIX_HOLD_BARRIER_FROZEN = "frozen"
PREFIX_HOLD_BARRIER_SOFT_ONLY = "soft_only"
PREFIX_HOLD_BARRIER_MODES = (
    PREFIX_HOLD_BARRIER_FROZEN,
    PREFIX_HOLD_BARRIER_SOFT_ONLY,
)
CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST = "valid_then_total_cost"
CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST = "result_terminal_earliest"


def resolve_prefix_hold_barrier_mode() -> str:
    """Resolve the opt-in prefix barrier mode; a typo is an error."""
    raw = (os.environ.get(PREFIX_HOLD_BARRIER_ENV) or "").strip()
    if not raw:
        return PREFIX_HOLD_BARRIER_FROZEN
    if raw not in PREFIX_HOLD_BARRIER_MODES:
        raise ValueError(
            f"{PREFIX_HOLD_BARRIER_ENV}={raw!r}: expected one of "
            f"{', '.join(PREFIX_HOLD_BARRIER_MODES)}"
        )
    return raw


def _prefix_hold_barrier_record(
    structural_required: bool,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Return the structural and applied device prefix-held requirements."""
    resolved = resolve_prefix_hold_barrier_mode() if mode is None else mode
    if resolved not in PREFIX_HOLD_BARRIER_MODES:
        raise ValueError(f"unsupported prefix-hold barrier mode {resolved!r}")
    structural = bool(structural_required)
    applied = structural and resolved != PREFIX_HOLD_BARRIER_SOFT_ONLY
    return {
        "mode": resolved,
        "structural_require_prefix_held": structural,
        "applied_require_prefix_held": applied,
    }


def _apply_prefix_hold_barrier(
    device_validity: dict[str, Any],
    structural_required: bool,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Apply the resolved hard-mask bit and return its identity record."""
    record = _prefix_hold_barrier_record(
        structural_required,
        mode=mode,
    )
    device_validity["require_prefix_held"] = bool(
        record["applied_require_prefix_held"]
    )
    return record


def _prefix_hold_barrier_should_record(
    record: Mapping[str, Any] | None,
    *,
    telemetry_mode: str,
) -> bool:
    """Keep default packets frozen while making every A191 arm auditable."""
    if record is None:
        return False
    return bool(
        record.get("mode") != PREFIX_HOLD_BARRIER_FROZEN
        or telemetry_mode == GRASP_BRIDGE_TELEMETRY_A191_V1
    )


def _device_proposal_seed_policy(trial_budget: str | None) -> str:
    """Preserve sampling packet vocabulary outside the new cost-focus arm."""
    return (
        "explicit_evaluation_device_seed"
        if evaluation_device_seed() is not None
        else
        "registered_cost_focus_seed_active_only"
        if trial_budget in (
            JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
            JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
        )
        else "fixed_zero"
    )


CANDIDATE_SELECTION_MODES = (
    CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
    CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST,
)
# Retained solely for the injected host runner used by unit tests and
# equal-budget benchmark tooling. Production proposals are generated on device
# by BatchedRollout and never use this path.
_HOST_REFERENCE_NOISE_FRACTIONS = (0.02, 0.20)

# Rollout validity thresholds: a fake penetration deeper than 4 mm is invalid.
_MAX_FAKE_PEN_M = 0.004
# Exact convex hulls have no proxy-geometry phantom band. Contact is measured
# from zero penetration and the generic 2 mm soft-contact allowance is shared
# by the exact GPU production lane.
_MAX_ARM_PEN_MESH_M = 0.002
_INVALID_RANK = 1  # score-tuple head: valid candidates sort strictly first


def validate_pool_size(value: Any) -> int:
    """Return one exact integer sampling-batch size."""
    if isinstance(value, bool):
        raise ValueError("pool_size must be an integer >= 2")
    try:
        n = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pool_size must be an integer >= 2") from exc
    if (
        n < 2
        or not np.isfinite(numeric)
        or numeric != float(n)
    ):
        raise ValueError(
            f"pool_size must be an integer >= 2, got {value!r}"
        )
    return n


def validate_candidate_selection_mode(value: Any) -> str:
    """Return one typed candidate-result selection policy."""
    if not isinstance(value, str):
        raise ValueError(
            "candidate_selection_mode must be one of "
            f"{CANDIDATE_SELECTION_MODES}, got {value!r}"
        )
    mode = value.strip()
    if mode not in CANDIDATE_SELECTION_MODES:
        raise ValueError(
            "candidate_selection_mode must be one of "
            f"{CANDIDATE_SELECTION_MODES}, got {value!r}"
        )
    return mode


def _stage_uses_result_terminal_earliest(stage: Any) -> bool:
    """Whether every terminal has a truthful sampled-trajectory channel.

    This is a vocabulary capability check, not a task/stage name allow-list.
    RobotSubjectContact currently has exact native endpoint evidence but no
    16-sample trajectory, so that one predicate conservatively keeps ordinary
    valid-then-cost selection. All geometric, jaw-state, held, support, contact,
    and rest predicates use the same terminal-joint/earliest policy.
    """
    terminal = tuple(getattr(stage, "terminal", ()))
    return bool(terminal) and not any(
        isinstance(predicate, RobotSubjectContact)
        for predicate in terminal
    )


def _stage_uses_flip_v7_full_pose_terminal_earliest(
    stage: Any,
    unified_mppi_effort_profile: Any,
) -> bool:
    """Match only the sealed Flip-v7 full-pose alignment stage."""
    return (
        unified_mppi_effort_profile
        == "unified_mppi_effort_uniform20_n2001_flip_full_pose_earliest_v7"
        and getattr(stage, "name", None)
        == "align_robot_proximal_eraser_end_open_full_pose"
        and [predicate.to_dict() for predicate in getattr(stage, "running", ())]
        == [
            {
                "type": "relative_position",
                "a": "gripper_center",
                "b": "eraser_material_minus_x_grasp",
                "offset": [0.0, 0.0, 0.0],
                "tol": 0.012,
            },
            {
                "type": "axis_parallel",
                "a": "gripper_center",
                "b": "world_up",
                "antiparallel": True,
                "tol_deg": 12.0,
            },
            {
                "type": "axis_angle",
                "a": "gripper_closing_axis",
                "b": "eraser_axis",
                "theta_deg": 90.0,
                "tol_deg": 12.0,
            },
            {
                "type": "gripper_command",
                "state": "open",
                "width_m": None,
                "tol": 0.0,
            },
        ]
        and [predicate.to_dict() for predicate in getattr(stage, "terminal", ())]
        == [
            {
                "type": "relative_position",
                "a": "gripper_center",
                "b": "eraser_material_minus_x_grasp",
                "offset": [0.0, 0.0, 0.0],
                "tol": 0.012,
            },
            {
                "type": "axis_parallel",
                "a": "gripper_center",
                "b": "world_up",
                "antiparallel": True,
                "tol_deg": 12.0,
            },
            {
                "type": "axis_angle",
                "a": "gripper_closing_axis",
                "b": "eraser_axis",
                "theta_deg": 90.0,
                "tol_deg": 12.0,
            },
        ]
    )


def _stage_uses_cup_v8_fine_terminal_earliest(
    stage: Any,
    unified_mppi_effort_profile: Any,
) -> bool:
    """Match only the sealed Cup-v8 final fine-recenter stage."""
    return (
        unified_mppi_effort_profile
        == "unified_mppi_effort_uniform20_n2001_cup_fine_earliest_v8"
        and getattr(stage, "name", None)
        == "fine_recenter_side_preapproach_cup_open"
        and [predicate.to_dict() for predicate in getattr(stage, "running", ())]
        == [
            {
                "type": "relative_position",
                "a": "gripper_center",
                "b": "cup_center",
                "offset": [-0.1, 0.0, 0.005],
                "tol": 0.025,
            },
            {
                "type": "axis_parallel",
                "a": "gripper_center",
                "b": "world_x",
                "antiparallel": False,
                "tol_deg": 12.0,
            },
            {
                "type": "axis_parallel",
                "a": "gripper_closing_axis",
                "b": "world_y",
                "antiparallel": False,
                "tol_deg": 12.0,
            },
            {
                "type": "gripper_command",
                "state": "open",
                "width_m": None,
                "tol": 0.0,
            },
        ]
        and [predicate.to_dict() for predicate in getattr(stage, "terminal", ())]
        == [
            {
                "type": "relative_position",
                "a": "gripper_center",
                "b": "cup_center",
                "offset": [-0.1, 0.0, 0.005],
                "tol": 0.035,
            },
            {
                "type": "axis_parallel",
                "a": "gripper_center",
                "b": "world_x",
                "antiparallel": False,
                "tol_deg": 15.0,
            },
            {
                "type": "axis_parallel",
                "a": "gripper_closing_axis",
                "b": "world_y",
                "antiparallel": False,
                "tol_deg": 15.0,
            },
            {
                "type": "gripper_state",
                "state": "open",
                "width_m": None,
                "tol": 0.01,
            },
        ]
    )


def _stage_uses_result_terminal_earliest_for_profile(
    stage: Any,
    unified_mppi_effort_profile: Any,
) -> bool:
    """Apply exact staged-profile scopes; preserve generic legacy behavior."""
    if (
        unified_mppi_effort_profile
        == "unified_mppi_effort_uniform20_n2001_flip_full_pose_earliest_v7"
    ):
        return _stage_uses_flip_v7_full_pose_terminal_earliest(
            stage,
            unified_mppi_effort_profile,
        )
    if (
        unified_mppi_effort_profile
        == "unified_mppi_effort_uniform20_n2001_cup_fine_earliest_v8"
    ):
        return _stage_uses_cup_v8_fine_terminal_earliest(
            stage,
            unified_mppi_effort_profile,
        )
    return _stage_uses_result_terminal_earliest(stage)




def pool_size_from_env() -> int:
    """Resolve the fixed total rollout budget.

    ``2048`` is the production real-tool-use batch.  The authorization token
    binds this resolved value for hardware runs. ``OAP_POOL_SIZE`` remains
    available for simulation ablations; CEM divides it evenly across rounds.
    """
    _default = "2048"
    raw = (os.environ.get(POOL_SIZE_ENV) or _default).strip() or _default
    try:
        return validate_pool_size(raw)
    except ValueError as exc:
        raise ValueError(
            f"{POOL_SIZE_ENV}={raw!r}: the sampling batch needs an integer "
            "with at least 2 candidates (nominal + a sample)"
        ) from exc


@dataclass
class SamplerStats:
    """Record of one sim-in-the-loop optimizer call."""

    batch: dict[str, Any] = field(default_factory=dict)
    pool_size: int = 0
    ran: bool = False
    # Stable, machine-readable reason why the sampling backend could not
    # establish physics validity.  This is deliberately separate from
    # ``initial_robot_scene_contact``: a GPU failure provides no contact
    # evidence, but it is not itself evidence of a bad initial scene.
    failure_reason: str | None = None
    elapsed_s: float = 0.0
    flat_objective: bool = False         # cost identical across VALID candidates
    final_n_valid: int | None = None      # None means physics never established validity
    # Formal terminal-selected exact replay has an execution-protocol gate
    # separate from device physics validity. ``None`` preserves every other
    # selector/profile path; False is a typed no-valid-plan refusal.
    execution_plan_accepted: bool | None = None
    # In a dry-run, the selected candidate's exact MJWarp state at the executed
    # knot interval becomes the next physical simulation state.  These fields
    # remain None for injected unit-test runners and are never used by the real
    # robot executor.
    selected_prefix_state: tuple[np.ndarray, np.ndarray] | None = None
    selected_prefix_control: np.ndarray | None = None
    selected_prefix_contact: dict[str, Any] | None = None
    # Experiment-only every-post-step qpos tape from the exact selected GPU
    # rollout.  The default path leaves this None and does not materialize an
    # H-long state array on device or host.
    selected_prefix_qpos_trace: np.ndarray | None = None
    selected_prefix_step_dt_s: float | None = None
    # Compact, JSON-safe diagnostics for the full selected rollout. These are
    # evidence only: selection has already happened on the device cost, and
    # the host never rescales or vetoes a candidate from these values.
    selected_rollout_diagnostics: dict[str, Any] | None = None
    # Candidate-invariant scene-integrity evidence from the first physical
    # MJWarp step.  ``None`` means the device rollout did not attest it.
    initial_robot_scene_contact: bool | None = None
    # MPPI optimizes torque/effort actions, while ``best.knots`` is the
    # realized joint-position trajectory used by the execution/evidence path.
    # Keep the optimizer mean separate so receding-horizon warm start never
    # feeds realized positions back as velocities.
    mppi_updated_action_plan: np.ndarray | None = None
    # Exact action endpoints selected for physical execution. This is distinct
    # from both the cost-weighted warm mean and the realized position carrier.
    mppi_selected_action_plan: np.ndarray | None = None


def _sampling_failure_reason(
    exc: BaseException,
    *,
    default: str,
) -> str:
    """Classify backend failures without leaking backend exception strings."""
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    if any(
        marker in message
        for marker in (
            "cuda_error_out_of_memory",
            "cuda out of memory",
            "out of memory",
        )
    ):
        return "gpu_out_of_memory"
    return default


def _initial_robot_scene_contact_from_rows(
    rows: dict[int, dict[str, Any]],
    candidate_ids: list[int],
) -> bool:
    """Require one candidate-invariant Boolean from the GPU rollout.

    Every candidate begins from the same measured state and knot-zero control,
    so disagreement is an invalid device result rather than optimizer
    information.  This function deliberately has no geometric fallback on the
    host: missing, non-Boolean, or inconsistent evidence fails closed.
    """
    values: list[bool] = []
    for candidate_id in candidate_ids:
        row = rows.get(int(candidate_id))
        if row is None or "initial_robot_scene_contact" not in row:
            raise RuntimeError(
                "GPU rollout omitted initial_robot_scene_contact for "
                f"candidate {candidate_id}"
            )
        value = row["initial_robot_scene_contact"]
        if not isinstance(value, (bool, np.bool_)):
            raise RuntimeError(
                "GPU rollout initial_robot_scene_contact must be Boolean for "
                f"candidate {candidate_id}, got {type(value).__name__}"
            )
        values.append(bool(value))
    if not values:
        raise RuntimeError("GPU rollout returned no initial-contact evidence")
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError(
            "GPU rollout candidates disagree on initial_robot_scene_contact"
        )
    return values[0]


def _anchors_at(anchors: Any, pose7: np.ndarray, *, table_top_z: float,
                sub_h: float, start_quat_wxyz: np.ndarray | None = None,
                movables: dict | None = None,
                movable_grounder: Any = None,
                gripper_center: Any = None,
                gripper_axis: Any = None,
                gripper_closing_axis: Any = None,
                gripper_width: float | None = None,
                gripper_command: float | None = None) -> Any:
    """The AnchorSet implied by one simulated state.

    When the caller supplies the chunk-start orientation, the simulated final
    orientation rotates the subject's axis anchors too -- a pour candidate's
    achieved TILT is visible to the program cost, not just its position.

    ``movables`` are the OTHER free bodies' poses from the same rollout state,
    grounded through ``movable_grounder`` (a hook, because this module scores
    rows and must not learn about scene objects). Without them a goal about a
    body the robot does not hold is INERT to the optimizer: predicted_end_anchors
    moves only the subject, so every candidate yields the same residual and
    there is no gradient for a push to descend.
    """
    pose7 = np.asarray(pose7, dtype=float)
    aset = predicted_end_anchors(anchors, {"metrics": {
        "final_object_xy": pose7[:2].tolist(),
        "final_object_lift_m": max(0.0, float(pose7[2]) - float(table_top_z)
                                   - 0.5 * float(sub_h)),
        "final_object_quat_wxyz": pose7[3:7].tolist(),
    }}, table_top_z=table_top_z, sub_h=sub_h, start_quat_wxyz=start_quat_wxyz)
    if movables and movable_grounder is not None:
        movable_grounder(aset, movables)
    if gripper_center is not None:
        point = np.asarray(gripper_center, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("rollout gripper_center must be a finite xyz")
        axis = None
        if gripper_axis is not None:
            axis = np.asarray(gripper_axis, dtype=float)
            if (
                axis.shape != (3,)
                or not np.all(np.isfinite(axis))
                or np.linalg.norm(axis) <= 1e-9
            ):
                raise ValueError(
                    "rollout gripper_axis must be one finite nonzero xyz"
                )
        aset.add(
            "gripper_center",
            Anchor(point=point, axis=axis, kind="robot"),
        )
        if gripper_closing_axis is not None:
            closing_axis = np.asarray(gripper_closing_axis, dtype=float)
            if (
                closing_axis.shape != (3,)
                or not np.all(np.isfinite(closing_axis))
                or np.linalg.norm(closing_axis) <= 1e-9
            ):
                raise ValueError(
                    "rollout gripper_closing_axis must be one finite "
                    "nonzero xyz"
                )
            aset.add(
                "gripper_closing_axis",
                Anchor(point=point, axis=closing_axis, kind="robot"),
            )
    if gripper_width is not None:
        width = float(gripper_width)
        if not np.isfinite(width):
            raise ValueError("rollout gripper_width must be finite")
        aset.add(
            "gripper_width",
            Anchor(point=np.array([width, 0.0, 0.0]), kind="robot"),
        )
    if gripper_command is not None:
        command = float(gripper_command)
        if not np.isfinite(command):
            raise ValueError("rollout gripper_command must be finite")
        aset.add(
            "gripper_command",
            Anchor(point=np.array([command, 0.0, 0.0]), kind="robot"),
        )
    return aset


def _predicate_uses_gripper_axis(
    predicate: Any,
    anchor_name: str = "gripper_center",
) -> bool:
    """Whether a predicate reads one reserved measured gripper FK axis."""
    while isinstance(predicate, TemporalHold):
        predicate = predicate.inner
    if isinstance(predicate, (AxisParallel, AxisAngle)):
        return anchor_name in (predicate.a, predicate.b)
    if isinstance(predicate, AngleAboutAxis):
        return anchor_name in (
            predicate.a,
            predicate.b,
            predicate.ref,
        )
    if isinstance(predicate, PointOnLine):
        return predicate.dir_anchor == anchor_name
    if isinstance(predicate, AlongAxisGap):
        return predicate.b == anchor_name
    if isinstance(predicate, InteractionAlign):
        axis_a = (
            predicate.a
            if predicate.axis_a == "self"
            else predicate.axis_a
        )
        axis_b = (
            predicate.b
            if predicate.axis_b == "self"
            else predicate.axis_b
        )
        return anchor_name in (axis_a, axis_b)
    return False


def _tool_was_used(row: dict, *, program: Any, stage_idx: int, anchors: Any,
                   terminal: bool = False) -> bool | None:
    """Audit executed ``tool_mediation(tool, target)`` evidence.

    The planner never infers indirect tool use from a movable terminal anchor.
    Every declared stage audits the exact no-direct-contact relation. Only a
    stage whose explicit ``require_tool_target_contact`` flag is true also
    requires one exact tool-target contact. During task-level reporting
    (``terminal=True``), the whole program contract requires no direct contact
    anywhere and at least one tool contact if any stage requested it. Evidence
    is logged separately and never redefines the measured terminal. The
    contact existence diagnostic has no duration, penetration-depth, or angle
    threshold; target progress and explicit geometry provide the dense signal.
    """
    if program is None or anchors is None or not program.stages:
        return True
    from oap.program.verdict import (
        stage_requires_tool_mediation,
        stage_requires_tool_target_contact,
        tool_mediation_target_bodies,
    )

    if terminal:
        mediation_stages = [
            stage
            for stage in program.stages
            if stage_requires_tool_mediation(stage, anchors)
        ]
        if not mediation_stages:
            return True
        # Program validation requires the same pair in every stage. Selecting
        # one declaration here only grounds that explicit pair; the positive
        # evidence flag is aggregated across all declarations below.
        stage = mediation_stages[-1]
        contact_requirements = tuple(
            stage_requires_tool_target_contact(candidate, anchors)
            for candidate in mediation_stages
        )
        require_tool_contact = any(contact_requirements)
        all_mediation_stages_require_contact = all(contact_requirements)
    else:
        idx = max(0, min(int(stage_idx), len(program.stages) - 1))
        stage = program.stages[idx]
        if not stage_requires_tool_mediation(stage, anchors):
            return True
        require_tool_contact = stage_requires_tool_target_contact(
            stage,
            anchors,
        )
        all_mediation_stages_require_contact = require_tool_contact
    subject_frames: float | None = None
    robot_frames: float | None = None
    owners = {
        str(getattr(anchors[name], "attached_to"))
        for name in anchors.names()
        if (
            getattr(anchors[name], "dynamic", False)
            and getattr(anchors[name], "attached_to", None)
            not in (None, "subject")
        )
    }

    # When grounding proves that the scene has exactly one non-subject movable
    # body, the episode-wide generic counters are already exact counters for
    # that body.  Prefer them over a later stage-scoped map: the latter may
    # cover only cycles in which ToolMediation was active and must never erase
    # an earlier direct robot contact.  This is structural (free-body
    # ownership), not a task/name special case.
    required = ("robot_movable_frames", "subject_movable_frames")
    if len(owners) == 1 and all(name in row for name in required):
        try:
            expected = tool_mediation_target_bodies(
                stage,
                anchors,
                tuple(owners),
            )
        except ValueError:
            return None
        if set(expected) == owners:
            robot_frames = float(row["robot_movable_frames"])
            subject_frames = float(row["subject_movable_frames"])

    # GPU rows are already reduced against this Stage's explicit target geom
    # mask. Keep the body ids beside the scalars so a row from another stage
    # cannot be misinterpreted.
    row_targets = tuple(
        str(name) for name in row.get("mediation_target_bodies", ()) if name
    )
    if (
        subject_frames is None
        and
        row_targets
        and "robot_target_frames" in row
        and "subject_target_frames" in row
    ):
        try:
            expected = tool_mediation_target_bodies(
                stage,
                anchors,
                row_targets,
            )
        except ValueError:
            return None
        if set(expected) != set(row_targets):
            return None
        robot_frames = float(row["robot_target_frames"])
        subject_frames = float(row["subject_target_frames"])

    # Executed-prefix evidence is retained per body across cycles/stages. Select
    # only the current/final terminal's target; a distractor's contact remains
    # auditable without satisfying or vetoing this goal.
    subject_by_body = row.get("subject_target_frames_by_body")
    robot_by_body = row.get("robot_target_frames_by_body")
    if (
        subject_frames is None
        and isinstance(subject_by_body, dict)
        and isinstance(robot_by_body, dict)
    ):
        known = tuple(dict.fromkeys(
            [str(name) for name in subject_by_body]
            + [str(name) for name in robot_by_body]
        ))
        try:
            expected = tool_mediation_target_bodies(stage, anchors, known)
        except ValueError:
            return None
        robot_frames = sum(
            float(robot_by_body.get(name, 0.0)) for name in expected
        )
        subject_frames = sum(
            float(subject_by_body.get(name, 0.0)) for name in expected
        )

    # Backward-compatible only when the grounding proves there is exactly one
    # non-subject dynamic body owner. Old test/parity rows then remain
    # unambiguous; a multi-movable row without body-scoped evidence is UNKNOWN,
    # never silently aggregated.
    if subject_frames is None:
        if len(owners) == 1 and all(name in row for name in required):
            robot_frames = float(row["robot_movable_frames"])
            subject_frames = float(row["subject_movable_frames"])
        else:
            return None

    # At task-level verification, positive evidence belongs only to stages
    # whose typed ToolMediation declaration explicitly requested contact.
    # Direct robot-target contact remains the whole-episode counter selected
    # above.  A mixed barrier/use program without the scoped map is UNKNOWN;
    # only legacy programs that required contact in every mediation stage may
    # safely fall back to the global subject counter.
    if terminal and require_tool_contact:
        required_subject_by_body = row.get(
            "required_subject_target_frames_by_body"
        )
        if isinstance(required_subject_by_body, dict):
            known = tuple(dict.fromkeys(
                [str(name) for name in required_subject_by_body]
                + [str(name) for name in row.get(
                    "robot_target_frames_by_body", {}
                )]
                + [str(name) for name in row.get(
                    "mediation_target_bodies", ()
                ) if name]
            ))
            try:
                expected = tool_mediation_target_bodies(
                    stage,
                    anchors,
                    known,
                )
            except ValueError:
                return None
            subject_frames = sum(
                float(required_subject_by_body.get(name, 0.0))
                for name in expected
            )
        elif not all_mediation_stages_require_contact:
            return None

    assert robot_frames is not None
    no_direct_contact = (
        robot_frames == 0.0
    )
    subject_contact = (
        subject_frames > 0.0
    )
    return no_direct_contact and (
        subject_contact if require_tool_contact else True
    )


def _row_physics_valid(
    row: dict[str, float],
    *,
    arm_pen_thresh: float,
    table_top_z: float = TABLE_TOP_Z,
) -> bool:
    """Task-independent collision and world-validity checks for one rollout.

    The rollout has already removed only typed subject-contact targets from
    ``max_obj_world_pen``: explicit ToolMediation targets and dynamic planes in
    ``OnPlane(subject_anchor, target_plane)``. Every remaining support contact
    keeps generic obstacle semantics, whether or not that support has a
    freejoint. Holding and tool-use semantics live only in explicit program
    terms: ordinary relations contribute finite cost, while an explicitly
    requested legacy ``ToolMediation`` relation contributes the same finite
    direct-contact residual as ``NoContact(robot,target)``.
    """
    from oap.program.verdict import ejection_floor_z

    floor = ejection_floor_z(table_top_z)
    pose_groups = [
        np.asarray(row.get("final_pose7", []), dtype=float),
        *[
            np.asarray(pose, dtype=float)
            for pose in (row.get("final_movable_pose7") or {}).values()
        ],
        *[
            np.asarray(path, dtype=float)
            for path in (row.get("movable_traj7") or {}).values()
        ],
    ]
    stays_in_world = all(
        poses.size == 0
        or (
            poses.shape[-1] >= 3
            and np.all(np.isfinite(poses))
            and bool(np.all(poses[..., 2] >= floor))
        )
        for poses in pose_groups
    )
    return bool(
        row["max_obj_world_pen"] < _MAX_FAKE_PEN_M
        and row["max_pad_table_pen"] < _MAX_FAKE_PEN_M
        and row.get("max_arm_pen", 0.0) < arm_pen_thresh
        and stays_in_world
    )


def _score_row(row: dict[str, float] | None, *,
               program: Any, stage_idx: int, anchors: Any,
               table_top_z: float, sub_h: float,
               start_quat_wxyz: np.ndarray | None = None,
               arm_pen_thresh: float = _MAX_ARM_PEN_MESH_M,
               movable_grounder: Any = None,
               execution_prefix_frac: float = 1.0,
               cost_shaping_profile: str | None = None,
               ) -> tuple[float, float, float]:
    """Score one rollout row: (invalid, cost, -two_sided_frames).

    Holding is never an implicit validity condition, and it is never a
    feasibility barrier. ``ObjectHeld`` contributes only when it appears
    explicitly in the active stage, and only as a finite cost: running adds the
    exact full-horizon not-held fraction, terminal adds a Boolean endpoint
    residual. There is no acquisition-vs-retention split; losing the object is
    ranked, heavily, rather than declared infeasible, so a batch that has not
    yet found a retaining candidate still has a feasible set to improve. Stage
    advance uses freshly measured terminal evidence, ``ObjectHeld`` included.

    Both the host reference and production device lane evaluate terminal
    predicates only at the predicted full-horizon endpoint. The executable
    prefix is a state-transfer and safety boundary, never a second objective.

    The cost is the standard optimal-control objective
    ``J = sum_t l(x_t) + phi(x_T)`` with the program supplying both terms:
    ``running`` predicates are evaluated over the rollout's sub-sampled state
    trajectory, and ``terminal`` predicates at the final predicted state
    (:func:`oap.program.trajectory_cost`). Only terminal predicates are
    evaluated on the re-observed real state to advance a stage. Generic
    physical safety remains a task-independent feasibility gate.

    There is deliberately no fallback objective.  A missing/un-groundable
    program is not a different planner: it is a refusal to rank candidates.
    """
    if row is None:
        return (float(_INVALID_RANK), float("inf"), 0.0)
    if "device_program_cost" in row:
        # Production exact-GPU path: the fixed program cost and every generic
        # feasibility condition were evaluated while the rollout
        # trajectories were still device arrays. Never rebuild AnchorSets or
        # reinterpret those candidates on the host.
        cost = float(row["device_program_cost"])
        valid = bool(row.get("device_program_valid", False))
        if not np.isfinite(cost):
            return (float(_INVALID_RANK), float("inf"), 0.0)
        return (
            0.0 if valid else float(_INVALID_RANK),
            cost,
            0.0,
        )
    active_stage = None
    idx = 0
    if program is not None and getattr(program, "stages", None):
        idx = max(0, min(int(stage_idx), len(program.stages) - 1))
        active_stage = program.stages[idx]
    from oap.program.verdict import object_held_anchors

    running_terms: list[Any] = []
    if active_stage is not None:
        for predicate in active_stage.running:
            while isinstance(predicate, TemporalHold):
                predicate = predicate.inner
            running_terms.append(predicate)
    terminal_object_held = (
        object_held_anchors(active_stage)
        if active_stage is not None
        else ()
    )
    require_terminal_object_held = bool(terminal_object_held)
    running_object_held = (
        [
            predicate
            for predicate in running_terms
            if isinstance(predicate, ObjectHeld)
        ]
        if active_stage is not None
        else []
    )
    running_not_held = (
        [
            predicate
            for predicate in running_terms
            if isinstance(predicate, NotHeld)
        ]
        if active_stage is not None
        else []
    )
    terminal_not_held_present = bool(active_stage is not None and any(
        isinstance(
            (
                predicate.inner
                if isinstance(predicate, TemporalHold)
                else predicate
            ),
            NotHeld,
        )
        for predicate in active_stage.terminal
    ))
    running_mediation = (
        [
            predicate
            for predicate in running_terms
            if isinstance(predicate, ToolMediation)
        ]
        if active_stage is not None
        else []
    )
    running_contact_relations = (
        [
            predicate
            for predicate in running_terms
            if isinstance(predicate, (Contact, NoContact))
        ]
        if active_stage is not None
        else []
    )
    running_robot_subject_contact = (
        [
            predicate
            for predicate in running_terms
            if isinstance(predicate, RobotSubjectContact)
        ]
        if active_stage is not None
        else []
    )
    terminal_robot_subject_contact = bool(
        active_stage is not None
        and any(
            isinstance(
                (
                    predicate.inner
                    if isinstance(predicate, TemporalHold)
                    else predicate
                ),
                RobotSubjectContact,
            )
            for predicate in active_stage.terminal
        )
    )
    if len(running_robot_subject_contact) > 1:
        return (float(_INVALID_RANK), float("inf"), 0.0)
    valid = _row_physics_valid(
        row,
        arm_pen_thresh=arm_pen_thresh,
        table_top_z=table_top_z,
    )
    final = np.asarray(row["final_pose7"], dtype=float)
    if program is None or anchors is None or not program.stages:
        # Pure-function callers may still inspect the physics gate, but an
        # infinite cost cannot rank anything. sample_once refuses this
        # configuration before invoking the scorer, so production cannot
        # execute a fallback plan.
        return (
            0.0 if valid else float(_INVALID_RANK),
            float("inf"),
            0.0,
        )
    stage = program.stages[idx]
    host_running_scales, host_terminal_scales = planning_cost_scales(stage, anchors)
    host_shaping = (stage_cost_shaping_plan(stage, cost_shaping_profile)
                    if cost_shaping_profile is not None else None)

    def binary_running_cost(predicate: Any, violation_fraction: float) -> float:
        """Mean of Eq. (3) for an exact binary stream, not norm(mean(r))."""
        term_index = next(i for i, term in enumerate(running_terms) if term is predicate)
        scale = host_running_scales[term_index]
        weight = host_shaping.running_multipliers[term_index] if host_shaping else 1.0
        return float(weight) * float(violation_fraction) * (math.sqrt(1.0 + (1.0 / scale) ** 2) - 1.0)

    def binary_terminal_cost(predicate_type: type, violated: bool) -> float:
        total = 0.0
        for term_index, term in enumerate(stage.terminal):
            while isinstance(term, TemporalHold):
                term = term.inner
            if isinstance(term, predicate_type):
                weight = host_shaping.terminal_multipliers[term_index] if host_shaping else 1.0
                total += float(weight) * float(violated) / host_terminal_scales[term_index] ** 2
        return total
    held_anchor_names = [
        *(predicate.object_anchor for predicate in running_object_held),
        *terminal_object_held,
    ]
    if any(name not in anchors for name in held_anchor_names):
        return (float(_INVALID_RANK), float("inf"), 0.0)
    # The acquisition-vs-retention split existed only to decide which stages
    # got an exact-held feasibility barrier at the executable prefix. That
    # barrier is gone on both lanes; ObjectHeld is a finite cost for every
    # stage that names it.
    needs_gripper_center = "gripper_center" in stage.referenced_anchors()
    needs_gripper_axis = any(
        _predicate_uses_gripper_axis(predicate)
        for predicate in stage.running + stage.terminal
    )
    needs_gripper_closing_axis = any(
        _predicate_uses_gripper_axis(
            predicate,
            "gripper_closing_axis",
        )
        for predicate in stage.running + stage.terminal
    )
    gripper_traj = row.get("gripper_traj3")
    final_gripper = row.get("final_gripper_center")
    if needs_gripper_center and gripper_traj is None and final_gripper is None:
        # Never substitute the seed, target, or commanded TCP for measured
        # rollout kinematics.  The objective is ungroundable and therefore
        # cannot authorize motion.
        return (float(_INVALID_RANK), float("inf"), 0.0)
    gripper_axis_traj = row.get("gripper_axis_traj3")
    final_gripper_axis = row.get("final_gripper_axis")
    if (
        needs_gripper_axis
        and gripper_axis_traj is None
        and final_gripper_axis is None
    ):
        # Orientation cost must see the measured rollout FK. A static current
        # anchor or commanded orientation would make every candidate's axis
        # identical and silently remove grasp-pose optimization.
        return (float(_INVALID_RANK), float("inf"), 0.0)
    gripper_closing_axis_traj = row.get(
        "gripper_closing_axis_traj3"
    )
    final_gripper_closing_axis = row.get(
        "final_gripper_closing_axis"
    )
    if (
        needs_gripper_closing_axis
        and gripper_closing_axis_traj is None
        and final_gripper_closing_axis is None
    ):
        return (float(_INVALID_RANK), float("inf"), 0.0)
    running_gripper_command = any(
        isinstance(
            predicate.inner if isinstance(predicate, TemporalHold) else predicate,
            GripperCommand,
        )
        for predicate in stage.running
    )
    terminal_gripper_state = any(
        isinstance(
            predicate.inner if isinstance(predicate, TemporalHold) else predicate,
            GripperState,
        )
        for predicate in stage.terminal
    )
    width_traj = row.get("gripper_width_traj")
    final_width = row.get("final_gripper_width")
    command_traj = row.get("gripper_command_traj")
    if (
        (running_gripper_command and command_traj is None)
        or (terminal_gripper_state and final_width is None)
    ):
        # A missing command or measured joint state must not let either
        # predicate's anchor-only fallback authorize a rollout.
        return (float(_INVALID_RANK), float("inf"), 0.0)
    try:
        traj7 = row.get("pose_traj7")
        mov_traj = row.get("movable_traj7") or {}
        mov_final = row.get("final_movable_pose7") or {}
        if traj7 is not None:
            traj = np.asarray(traj7, dtype=float)
            grip = (
                np.asarray(gripper_traj, dtype=float)
                if gripper_traj is not None
                else None
            )
            widths = (
                np.asarray(width_traj, dtype=float)
                if width_traj is not None
                else None
            )
            commands = (
                np.asarray(command_traj, dtype=float)
                if command_traj is not None
                else None
            )
            grip_axes = (
                np.asarray(gripper_axis_traj, dtype=float)
                if gripper_axis_traj is not None
                else None
            )
            grip_closing_axes = (
                np.asarray(gripper_closing_axis_traj, dtype=float)
                if gripper_closing_axis_traj is not None
                else None
            )
            # Same index into every body's trajectory: the running cost must see
            # the subject and the pushed body at the SAME instant, or a
            # running predicate relating them compares different times.
            states = [_anchors_at(anchors, p, table_top_z=table_top_z, sub_h=sub_h,
                                  start_quat_wxyz=start_quat_wxyz,
                                  movables={n: v[k] for n, v in mov_traj.items()
                                            if k < len(v)} or mov_final,
                                  movable_grounder=movable_grounder,
                                  gripper_center=(
                                      grip[k] if grip is not None and k < len(grip)
                                      else final_gripper
                                  ),
                                  gripper_axis=(
                                      grip_axes[k]
                                      if (
                                          grip_axes is not None
                                          and k < len(grip_axes)
                                      )
                                      else final_gripper_axis
                                  ),
                                  gripper_closing_axis=(
                                      grip_closing_axes[k]
                                      if (
                                          grip_closing_axes is not None
                                          and k < len(grip_closing_axes)
                                      )
                                      else final_gripper_closing_axis
                                  ),
                                  gripper_width=(
                                      widths[k]
                                      if widths is not None and k < len(widths)
                                      else final_width
                                  ),
                                  gripper_command=(
                                      commands[k]
                                      if (
                                          commands is not None
                                          and k < len(commands)
                                      )
                                      else None
                                  ))
                      for k, p in enumerate(traj)]
        else:
            states = [_anchors_at(anchors, final, table_top_z=table_top_z,
                                  sub_h=sub_h, start_quat_wxyz=start_quat_wxyz,
                                  movables=mov_final,
                                  movable_grounder=movable_grounder,
                                  gripper_center=final_gripper,
                                  gripper_axis=final_gripper_axis,
                                  gripper_closing_axis=(
                                      final_gripper_closing_axis
                                  ),
                                  gripper_width=final_width,
                                  gripper_command=(
                                      command_traj[-1]
                                      if command_traj is not None
                                      else None
                                  ))]
        cost = float(trajectory_cost(
            stage,
            states,
            contact_terms_in_rollout=True,
            reference_anchors=anchors,
            cost_shaping_profile=cost_shaping_profile,
        ))
        if running_mediation or running_contact_relations:
            def exact_contact_any(actor: str) -> bool | None:
                if actor == "robot":
                    scalar = row.get("robot_target_contact_any_step")
                    frames = row.get("robot_target_frames")
                    trajectory = row.get("robot_target_contact_traj")
                else:
                    scalar = row.get("tool_target_contact_any_step")
                    frames = row.get("subject_target_frames")
                    trajectory = row.get("tool_target_contact_traj")
                if scalar is not None:
                    return bool(scalar)
                if frames is not None:
                    return float(frames) > 0.0
                if trajectory is None:
                    return None
                contact_array = np.asarray(trajectory, dtype=bool)
                if contact_array.shape != (len(states),):
                    return None
                return bool(np.any(contact_array))

            subject_contact = exact_contact_any("subject")
            robot_contact = exact_contact_any("robot")
            if subject_contact is None or robot_contact is None:
                return (float(_INVALID_RANK), float("inf"), 0.0)

            if len(running_mediation) > 1:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            if running_mediation:
                # Legacy ToolMediation is the finite compatibility spelling of
                # Contact(tool,target) + NoContact(robot,target).
                cost += binary_running_cost(running_mediation[0], float(robot_contact))
                if (
                    running_mediation[0].require_tool_target_contact
                    and not subject_contact
                ):
                    cost += binary_running_cost(running_mediation[0], 1.0)

            for relation in running_contact_relations:
                present = (
                    robot_contact
                    if "robot" in (relation.a, relation.b)
                    else subject_contact
                )
                cost += binary_running_cost(relation, float(
                    (not present)
                    if isinstance(relation, Contact)
                    else present
                ))
        if running_object_held:
            held = row.get("object_held_traj")
            if held is None:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            held_arr = np.asarray(held, dtype=bool)
            if held_arr.shape != (len(states),):
                return (float(_INVALID_RANK), float("inf"), 0.0)
            not_held_fraction = row.get("object_not_held_fraction")
            if not_held_fraction is None:
                # Host-reference rows may lack the exact-step fraction. Keep
                # their bounded-grid fallback finite and deterministic;
                # production always supplies the exact device reduction.
                not_held_fraction = float(np.mean(~held_arr))
            not_held_fraction = float(not_held_fraction)
            if (
                not np.isfinite(not_held_fraction)
                or not 0.0 <= not_held_fraction <= 1.0
            ):
                return (float(_INVALID_RANK), float("inf"), 0.0)
            cost += sum(binary_running_cost(term, not_held_fraction) for term in running_object_held)
            # No exact-held rejection at the executable-prefix endpoint; the
            # host reference must agree with the device, which now expresses
            # ObjectHeld only as finite running and terminal residuals. See the
            # end of `device_cost.cost` for the measurement that removed it.

        if running_robot_subject_contact:
            missing_fraction = row.get(
                "robot_subject_contact_missing_fraction"
            )
            if missing_fraction is None:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            missing_fraction = float(missing_fraction)
            if (
                not np.isfinite(missing_fraction)
                or not 0.0 <= missing_fraction <= 1.0
            ):
                return (float(_INVALID_RANK), float("inf"), 0.0)
            running_scales, _terminal_scales = planning_cost_scales(
                stage,
                anchors,
            )
            contact_index = next(
                index
                for index, predicate in enumerate(stage.running)
                if isinstance(
                    (
                        predicate.inner
                        if isinstance(predicate, TemporalHold)
                        else predicate
                    ),
                    RobotSubjectContact,
                )
            )
            cost += binary_running_cost(running_robot_subject_contact[0], missing_fraction)

        if running_not_held:
            held = row.get("object_held_traj")
            if held is None:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            held_arr = np.asarray(held, dtype=bool)
            if held_arr.shape != (len(states),):
                return (float(_INVALID_RANK), float("inf"), 0.0)
            not_held_fraction = row.get("object_not_held_fraction")
            if not_held_fraction is None:
                not_held_fraction = float(np.mean(~held_arr))
            not_held_fraction = float(not_held_fraction)
            if (
                not np.isfinite(not_held_fraction)
                or not 0.0 <= not_held_fraction <= 1.0
            ):
                return (float(_INVALID_RANK), float("inf"), 0.0)
            # The exact complement of running ObjectHeld: the fraction of
            # steps that still pinch the object.
            cost += sum(binary_running_cost(term, 1.0 - not_held_fraction) for term in running_not_held)

        if terminal_not_held_present:
            if "final_two_sided_now" not in row:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            if bool(row["final_two_sided_now"]):
                _running_scales, terminal_scales = planning_cost_scales(
                    stage,
                    anchors,
                )
                not_held_index = next(
                    index
                    for index, predicate in enumerate(stage.terminal)
                    if isinstance(
                        (
                            predicate.inner
                            if isinstance(predicate, TemporalHold)
                            else predicate
                        ),
                        NotHeld,
                    )
                )
                cost += binary_terminal_cost(NotHeld, True)

        if require_terminal_object_held:
            # Exact endpoint evidence is not an AnchorSet quantity. Missing
            # evidence remains ungroundable, but a grounded false endpoint is
            # a finite Boolean terminal residual. This lets the MPC rank and
            # execute partial acquisition progress; only the fresh measured
            # terminal verdict advances the stage.
            if "final_two_sided_now" not in row:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            if not bool(row["final_two_sided_now"]):
                _running_scales, terminal_scales = planning_cost_scales(
                    stage,
                    anchors,
                )
                held_index = next(
                    index
                    for index, predicate in enumerate(stage.terminal)
                    if isinstance(
                        (
                            predicate.inner
                            if isinstance(predicate, TemporalHold)
                            else predicate
                        ),
                        ObjectHeld,
                    )
                )
                cost += binary_terminal_cost(ObjectHeld, True)

        if terminal_robot_subject_contact:
            if "robot_subject_contact_endpoint" not in row:
                return (float(_INVALID_RANK), float("inf"), 0.0)
            if not bool(row["robot_subject_contact_endpoint"]):
                _running_scales, terminal_scales = planning_cost_scales(
                    stage,
                    anchors,
                )
                contact_index = next(
                    index
                    for index, predicate in enumerate(stage.terminal)
                    if isinstance(
                        (
                            predicate.inner
                            if isinstance(predicate, TemporalHold)
                            else predicate
                        ),
                        RobotSubjectContact,
                    )
                )
                cost += binary_terminal_cost(RobotSubjectContact, True)
    except (KeyError, ValueError, TypeError):
        return (float(_INVALID_RANK), float("inf"), 0.0)
    if not np.isfinite(cost):
        return (float(_INVALID_RANK), float("inf"), 0.0)
    return (
        0.0 if valid else float(_INVALID_RANK),
        cost,
        0.0,
    )


# --- Joint-space predictive shooting --------------------------------------
# Host-reference proposal retained for injected tests and offline benchmarks.
# Production uses the device-resident single-Gaussian proposal generator in
# batched_rollout.py.
def perturb_controls(knots: np.ndarray, *, rng: np.random.Generator,
                     bounds: ControlBounds, frac: float) -> np.ndarray:
    """One joint-space shooting sample: Gaussian noise on the (K, 8) knots.

    ``sigma = bounds.noise_sigma(frac)`` is the length-8 per-actuator std (mjpc
    convention). A shared bias plus per-knot jitter explores the arm. Knot zero
    is kept at the measured physical boundary supplied by the receding loop,
    so the spline starts from measured state without an instantaneous jump.

    The gripper width is sampled by the identical rule as every arm control.
    Only knot zero is fixed to the measured boundary for spline continuity.
    """
    knots = np.asarray(knots, dtype=float)
    sigma = bounds.noise_sigma(frac)                       # (8,) per-actuator
    rigid = rng.normal(0.0, sigma)                         # (8,) shared across knots
    jitter = rng.normal(0.0, 0.5 * sigma, size=knots.shape)  # (K,8) per-knot, incl width
    perturbed = knots + rigid[None, :] + jitter
    perturbed[0] = knots[0]
    return bounds.clip(perturbed)


# Backward-compatible private spelling for unit-test injection and experiment
# code written before the proposal became a shared public contract.
_perturb_controls = perturb_controls


def sample_once(
    *, world: Any, plan_xml: Any, candidates: list[ControlKnots],
    metadata: dict[int, dict[str, Any]], obj_pose7: np.ndarray,
    rng: np.random.Generator,
    program: Any = None, stage_idx: int = 0, anchors: Any = None,
    cost_scale_anchors: Any = None,
    episode: Any = None, chunk_idx: int = 0,
    sub_h: float = 0.0,
    bounds: ControlBounds | None = None,
    start_state: tuple[np.ndarray, np.ndarray] | None = None,
    rollout_runner: Callable[..., dict[int, dict[str, float]]] | None = None,
    pool_size: int = 2048,
    horizon_steps: int = DEFAULT_EXECUTION_STEPS,
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
    pick_v8_causal_temperature_state: Any | None = None,
    mppi_halton_seed: int | None = None,
    mppi_nominal_actions: np.ndarray | None = None,
    arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
    record_candidate_cost_telemetry: bool = False,
    table_contact_force_weight: float | None = None,
    control_regularizer_calibration_profile: str | None = None,
    joint_velocity_force_dial_arm: str | None = None,
    joint_velocity_force_phase3_finalist_arm: str | None = None,
    joint_velocity_force_phase3_seed: int | None = None,
    joint_velocity_force_trial_budget: str | None = None,
    cost_shaping_profile: str | None = None,
    previous_executed_action: np.ndarray | None = None,
    control_regularizer_boundary_valid: bool | None = None,
    movable_grounder: Any = None,
    execution_prefix_frac: float,
    execution_feasibility: Mapping[str, Any] | None = None,
    controlled_subject_contact_evidence_name: str | None = None,
) -> tuple[list[ControlKnots], dict[int, dict[str, Any]], SamplerStats]:
    """Run one joint-space predictive-sampling or iterative-CEM solve.

    The decision variable is the K x 8 joint knot, not the end-effector pose.
    The stage-entry current-control hold (or the previous winner shifted with a
    held tail) initializes the mean. Production iteratively samples a
    task-agnostic diagonal Gaussian, rolls every candidate through the same
    full-physics GPU primitive, and refits mean/std from valid elites. All
    eight future latent dimensions are sampled; knot zero remains the measured
    boundary. ``cem_rounds=1`` retains the equal-budget predictive-sampling
    baseline. There is no analytic proposal, waypoint, grasp pose, task router,
    or task-specific proposal family.

    The production backend is the GPU Warp rollout
    (:class:`~oap.twin.batched_rollout.BatchedRollout`, bare (K,8) arrays,
    exact convex arm meshes + the real gn01 linkage) for every stage. A missing
    or failed GPU leaves physics validity unknown, so the executor refuses
    motion. ``rollout_runner`` is an explicitly nonproduction host reference
    path for unit tests and equal-budget benchmarks; :func:`solve_mpc_step`,
    the production caller, never supplies it.

    ``plan_xml`` is required because the :class:`BatchedRollout` screen builds
    its reduced model from the plan XML. The GPU screen always uses the real
    gn01 linkage.
    ``start_state`` (a physical (qpos, qvel)) is threaded to Warp, so a held
    step continues from the live state instead of a fresh table reset.
    """
    from oap.loop.gpu_step_trace import gpu_step_trace_enabled

    stats = SamplerStats(pool_size=len(candidates))
    grasp_bridge_telemetry_mode = resolve_grasp_bridge_telemetry_mode()
    record_step_trace = gpu_step_trace_enabled()
    failure_started_s = time.perf_counter()
    horizon_steps = validate_horizon_steps(horizon_steps)
    sigma_fraction = validate_local_sigma_fraction(
        sigma_fraction
    )
    sampler_mode = validate_predictive_sampler_mode(sampler_mode)
    candidate_selection_mode = validate_candidate_selection_mode(
        candidate_selection_mode
    )
    mtp_jaw_mode = validate_mtp_jaw_mode(mtp_jaw_mode)
    if (
        sampler_mode != PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
        and mtp_jaw_mode != MTP_JAW_MODE_PRESERVE_NOMINAL
    ):
        raise ValueError(
            "sampled MTP jaw mode requires sampler_mode='mtp_global_local'"
        )
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
    if (
        candidate_selection_mode
        == CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
        and optimizer != "mppi"
    ):
        raise ValueError(
            "result_terminal_earliest requires optimizer='mppi'"
        )
    if (
        candidate_selection_mode
        == CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
        and joint_velocity_force_dial_arm is None
        and sampler_mode not in (
            *PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
            PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
        )
    ):
        raise ValueError(
            "result_terminal_earliest requires an MPPI sampler"
        )
    if (
        sampler_mode == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
        and optimizer != "mppi"
    ):
        raise ValueError("mtp_global_local requires optimizer='mppi'")
    if (
        sampler_mode == PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON
        and optimizer != "mppi"
    ):
        raise ValueError("reference_effort_halton requires optimizer='mppi'")
    mppi_temperature = validate_mppi_temperature(mppi_temperature)
    mppi_execution_mode = validate_mppi_execution_mode(mppi_execution_mode)
    control_profile = validate_control_profile(control_profile, allow_none=True)
    cost_shaping_profile = validate_cost_shaping_profile(
        cost_shaping_profile,
        allow_none=True,
    )
    phase3_arm = validate_joint_velocity_force_phase3_finalist_arm(
        joint_velocity_force_phase3_finalist_arm,
        allow_none=True,
    )
    phase3_seed = validate_joint_velocity_force_phase3_seed(
        joint_velocity_force_phase3_seed,
        allow_none=True,
    )
    if (phase3_arm is None) != (phase3_seed is None):
        raise ValueError("Phase-3 finalist arm and seed must be supplied together")
    if phase3_arm is not None and joint_velocity_force_dial_arm is not None:
        raise ValueError("Phase-3 finalist conflicts with DIAL arm")
    expected_device_seed = joint_velocity_force_device_proposal_seed(
        trial_budget=joint_velocity_force_trial_budget,
        dial_arm=joint_velocity_force_dial_arm,
        phase3_arm=phase3_arm,
        phase3_seed=phase3_seed,
    )
    device_seed_policy = _device_proposal_seed_policy(
        joint_velocity_force_trial_budget
    )
    if control_profile is not None and mppi_halton_seed != expected_device_seed:
        raise ValueError(
            "control-profile diagnostic requires its registered fixed device "
            f"proposal seed {expected_device_seed}"
        )
    arm_velocity_weight = validate_arm_velocity_weight(arm_velocity_weight)
    if (
        sampler_mode not in PREDICTIVE_SAMPLER_MPPI_HALTON_MODES
        and local_sigma_fraction >= sigma_fraction
    ):
        raise ValueError(
            "two_scale requires local_sigma_fraction < sigma_fraction "
            f"(broad), got {local_sigma_fraction} >= {sigma_fraction}"
        )
    if not candidates:
        return candidates, metadata, stats
    if (
        program is None
        or anchors is None
        or not getattr(program, "stages", None)
    ):
        logger.error(
            "[sampler] step %d: no grounded TaskProgram objective; "
            "refusing to rank or execute candidates",
            chunk_idx,
        )
        stats.final_n_valid = 0
        return candidates, metadata, stats
    from oap.program.verdict import (
        contact_relation_target_bodies,
        stage_contact_target_bodies,
        stage_requires_contact_relations,
    )

    active_stage = program.stages[
        max(0, min(int(stage_idx), len(program.stages) - 1))
    ]
    result_terminal_earliest_active = (
        candidate_selection_mode
        == CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
        and _stage_uses_result_terminal_earliest_for_profile(
            active_stage,
            unified_mppi_effort_profile,
        )
    )
    flip_v7_full_pose_terminal_earliest_authorized = (
        result_terminal_earliest_active
        and _stage_uses_flip_v7_full_pose_terminal_earliest(
            active_stage,
            unified_mppi_effort_profile,
        )
    )
    cup_v8_fine_terminal_earliest_authorized = (
        result_terminal_earliest_active
        and _stage_uses_cup_v8_fine_terminal_earliest(
            active_stage,
            unified_mppi_effort_profile,
        )
    )
    mediation_cost_active = stage_requires_contact_relations(
        active_stage,
        anchors,
    )
    screen = None
    prefix_hold_barrier_record: dict[str, Any] | None = None
    if rollout_runner is None:
        # One production backend for every stage.  A missing GPU is an
        # unavailable safety-critical sensor: do not substitute another solver
        # or reinterpret the unrolled nominal as permission to move.
        if not mjx_screen.is_available():
            logger.error(
                "[sampler] step %d: GPU/Warp backend unavailable; "
                "physics validity unknown and execution will be refused",
                chunk_idx,
            )
            stats.failure_reason = "gpu_backend_unavailable"
            stats.elapsed_s = time.perf_counter() - failure_started_s
            return candidates, metadata, stats

        from oap.twin.batched_rollout import BatchedRollout, plan_dt_from_env

        # Eager validation remains outside the GPU failure catch. A bad
        # planning-grid value is a configuration error, not an unavailable
        # rollout.
        plan_dt_from_env()
        regularizer_build_kwargs = (
            {}
            if control_regularizer_calibration_profile is None
            else {
                "control_regularizer_calibration_profile": (
                    control_regularizer_calibration_profile
                )
            }
        )
        try:
            screen = BatchedRollout.build(
                world,
                plan_xml,
                gripper="linkage",
                paper_mppi=(optimizer == "mppi"),
                control_profile=control_profile,
                table_contact_force_weight=(
                    table_contact_force_weight
                    if table_contact_force_weight is not None
                    else getattr(program, "table_contact_force_weight", None)
                ),
                **regularizer_build_kwargs,
            )
        except Exception as e:  # pragma: no cover - GPU/env dependent
            stats.failure_reason = _sampling_failure_reason(
                e,
                default="gpu_rollout_build_failed",
            )
            stats.elapsed_s = time.perf_counter() - failure_started_s
            logger.error(
                "[sampler] step %d: exact GPU rollout build failed (%s); "
                "physics validity unknown and execution will be refused",
                chunk_idx,
                e,
            )
            return candidates, metadata, stats
        mediation_targets: tuple[str, ...] = ()
        contact_targets: tuple[str, ...] = ()
        dynamic_body_names = [
            name for name, _ in screen.movable_addrs
        ]
        if mediation_cost_active:
            try:
                mediation_targets = contact_relation_target_bodies(
                    active_stage,
                    anchors,
                    dynamic_body_names,
                )
            except ValueError as exc:
                stats.failure_reason = "tool_mediation_binding_failed"
                stats.elapsed_s = time.perf_counter() - failure_started_s
                logger.error(
                    "[sampler] step %d: tool-mediation target binding failed "
                    "(%s); physics validity unknown and execution will be "
                    "refused",
                    chunk_idx,
                    exc,
                )
                return candidates, metadata, stats
        try:
            contact_targets = stage_contact_target_bodies(
                active_stage,
                anchors,
                dynamic_body_names,
            )
        except ValueError as exc:
            stats.failure_reason = "contact_target_binding_failed"
            stats.elapsed_s = time.perf_counter() - failure_started_s
            logger.error(
                "[sampler] step %d: contact-target binding failed "
                "(%s); physics validity unknown and execution will be "
                "refused",
                chunk_idx,
                exc,
            )
            return candidates, metadata, stats

        from oap.program.device_cost import make_device_trajectory_cost
        from oap.program.verdict import ejection_floor_z

        state_qpos = np.asarray(
            start_state[0] if start_state is not None else world.data.qpos,
            dtype=float,
        )
        movable_start_pose7 = {
            str(name): state_qpos[int(address):int(address) + 7].copy()
            for name, address in screen.world_movable_addrs
        }
        device_cost_fn = make_device_trajectory_cost(
            active_stage,
            anchors,
            cost_scale_anchors=cost_scale_anchors,
            subject_start_pose7=np.asarray(obj_pose7, dtype=float),
            movable_start_pose7=movable_start_pose7,
            table_top_z=TABLE_TOP_Z,
            sub_h=float(sub_h),
            # Eq. (5) averages every predicted physics step. AtRest is a
            # compatibility term and uses the actual planning step duration.
            sample_dt_s=float(getattr(screen, "plan_dt", 0.002)),
            cost_shaping_profile=cost_shaping_profile,
            pick_v8_causal_profile=pick_v8_causal_profile,
        )
        device_validity = {
            "max_fake_pen_m": float(_MAX_FAKE_PEN_M),
            "max_arm_pen_m": float(_MAX_ARM_PEN_MESH_M),
            "ejection_floor_z": float(ejection_floor_z(TABLE_TOP_Z)),
            **dict(execution_feasibility or {}),
            "action_feasibility_enabled": execution_feasibility is not None,
        }
        if active_stage is not None:
            from oap.program.predicates import ObjectHeld, TemporalHold

            # The frozen device path makes running ObjectHeld a hard endpoint
            # validity requirement. ``soft_only`` removes only that mask; the
            # explicit ObjectHeld residual still ranks every rollout.
            require_prefix_held = False
            for predicate in active_stage.running:
                while isinstance(predicate, TemporalHold):
                    predicate = predicate.inner
                if isinstance(predicate, ObjectHeld):
                    require_prefix_held = True
                    break
            prefix_hold_barrier_record = _apply_prefix_hold_barrier(
                device_validity,
                require_prefix_held,
            )
            if (
                require_prefix_held
                and prefix_hold_barrier_record["mode"]
                == PREFIX_HOLD_BARRIER_SOFT_ONLY
            ):
                logger.info(
                    "[sampler] step %d: prefix-held device barrier disabled "
                    "(%s=%s); explicit ObjectHeld remains a finite cost",
                    chunk_idx,
                    PREFIX_HOLD_BARRIER_ENV,
                    PREFIX_HOLD_BARRIER_SOFT_ONLY,
                )

    t0 = time.perf_counter()
    pool = list(candidates)
    meta = dict(metadata)
    next_id = max((int(c.candidate_id) for c in pool), default=0) + 1
    start_quat = np.asarray(obj_pose7, dtype=float)[3:7]
    arm_pen_thresh = _MAX_ARM_PEN_MESH_M
    best_seq: ControlKnots | None = None
    best: tuple[float, float, float] = (
        float(_INVALID_RANK),
        float("inf"),
        0.0,
    )
    rows: dict[int, dict[str, Any]] = {}

    try:
        if int(pool_size) < 2:
            raise ValueError("pool_size must be >= 2")
        if bounds is None:
            bounds = ControlBounds.from_world(world)

        # Production supplies one incumbent: current hold at stage entry, or
        # the shifted previous winner with its tail held on every later cycle.
        nominal = pool[0]
        nominal_knots = np.asarray(nominal.knots, dtype=float)
        nominal_meta = dict(meta.get(int(nominal.candidate_id), {}))
        if screen is not None:
            # The only host-side random operation is drawing one scalar seed.
            # Proposal tensors, spline controls, rollouts, scoring, validity,
            # and lexicographic argmin all remain on the device.
            device_seed = (
                int(mppi_halton_seed)
                if optimizer == "mppi" and mppi_halton_seed is not None
                else int(
                    rng.integers(
                        0, np.iinfo(np.int32).max, dtype=np.int64
                    )
                )
            )
            ctrl_low, ctrl_high = bounds.as_array()
            if optimizer == "mppi":
                result_selection_kwargs = (
                    {"prefer_earliest_joint_terminal_sample": True}
                    if result_terminal_earliest_active
                    else {}
                )
                if flip_v7_full_pose_terminal_earliest_authorized:
                    result_selection_kwargs[
                        "flip_v7_full_pose_terminal_earliest_authorized"
                    ] = True
                if cup_v8_fine_terminal_earliest_authorized:
                    result_selection_kwargs[
                        "cup_v8_fine_terminal_earliest_authorized"
                    ] = True
                regularizer_run_kwargs = (
                    {}
                    if control_regularizer_calibration_profile is None
                    else {
                        "previous_executed_action": previous_executed_action,
                        "control_regularizer_boundary_valid": (
                            control_regularizer_boundary_valid
                        ),
                    }
                )
                dial_run_kwargs = (
                    {}
                    if joint_velocity_force_dial_arm is None
                    else {
                        "joint_velocity_force_dial_arm": (
                            joint_velocity_force_dial_arm
                        )
                    }
                )
                unified_run_kwargs = (
                    {}
                    if unified_mppi_effort_profile is None
                    else {
                        "unified_mppi_effort_profile": (
                            unified_mppi_effort_profile
                        )
                    }
                )
                if pick_v8_causal_profile is not None:
                    unified_run_kwargs["pick_v8_causal_profile"] = (
                        pick_v8_causal_profile
                    )
                    unified_run_kwargs["pick_v8_causal_temperature_state"] = (
                        pick_v8_causal_temperature_state
                    )
                elif pick_v8_causal_temperature_state is not None:
                    unified_run_kwargs["pick_v8_causal_temperature_state"] = (
                        pick_v8_causal_temperature_state
                    )
                result = screen.run_mppi_sampling(
                    start_state=start_state,
                    nominal_knots=nominal_knots,
                    ctrl_low=ctrl_low,
                    ctrl_high=ctrl_high,
                    total_rollouts=int(pool_size),
                    temperature=mppi_temperature,
                    execution_mode=mppi_execution_mode,
                    seed=device_seed,
                    nominal_actions=mppi_nominal_actions,
                    sigma_fraction=sigma_fraction,
                    local_sigma_fraction=local_sigma_fraction,
                    sampler_mode=sampler_mode,
                    mtp_jaw_mode=mtp_jaw_mode,
                    obj_pose7=obj_pose7,
                    n_steps=horizon_steps,
                    execution_prefix_frac=execution_prefix_frac,
                    device_cost_fn=device_cost_fn,
                    device_validity=device_validity,
                    contact_target_bodies=contact_targets,
                    mediation_target_bodies=mediation_targets,
                    controlled_subject_contact_evidence_name=(
                        controlled_subject_contact_evidence_name
                    ),
                    record_step_trace=record_step_trace,
                    arm_velocity_weight=arm_velocity_weight,
                    record_candidate_cost_telemetry=(
                        record_candidate_cost_telemetry
                    ),
                    **unified_run_kwargs,
                    **regularizer_run_kwargs,
                    **dial_run_kwargs,
                    **result_selection_kwargs,
                )
            elif cem_rounds == 1:
                result = screen.run_predictive_sampling(
                    start_state=start_state,
                    nominal_knots=nominal_knots,
                    ctrl_low=ctrl_low,
                    ctrl_high=ctrl_high,
                    pool_size=int(pool_size),
                    seed=device_seed,
                    sigma_fraction=sigma_fraction,
                    sampler_mode=sampler_mode,
                    local_sigma_fraction=local_sigma_fraction,
                    obj_pose7=obj_pose7,
                    n_steps=horizon_steps,
                    execution_prefix_frac=execution_prefix_frac,
                    device_cost_fn=device_cost_fn,
                    device_validity=device_validity,
                    contact_target_bodies=contact_targets,
                    mediation_target_bodies=mediation_targets,
                    controlled_subject_contact_evidence_name=(
                        controlled_subject_contact_evidence_name
                    ),
                    record_step_trace=record_step_trace,
                    arm_velocity_weight=arm_velocity_weight,
                    record_candidate_cost_telemetry=(
                        record_candidate_cost_telemetry
                    ),
                )
            else:
                result = screen.run_cem_sampling(
                    start_state=start_state,
                    nominal_knots=nominal_knots,
                    ctrl_low=ctrl_low,
                    ctrl_high=ctrl_high,
                    total_rollouts=int(pool_size),
                    rounds=cem_rounds,
                    elite_fraction=cem_elite_fraction,
                    min_std_fraction=cem_min_std_fraction,
                    seed=device_seed,
                    sigma_fraction=sigma_fraction,
                    obj_pose7=obj_pose7,
                    n_steps=horizon_steps,
                    execution_prefix_frac=execution_prefix_frac,
                    device_cost_fn=device_cost_fn,
                    device_validity=device_validity,
                    contact_target_bodies=contact_targets,
                    mediation_target_bodies=mediation_targets,
                    controlled_subject_contact_evidence_name=(
                        controlled_subject_contact_evidence_name
                    ),
                    record_step_trace=record_step_trace,
                    arm_velocity_weight=arm_velocity_weight,
                    record_candidate_cost_telemetry=(
                        record_candidate_cost_telemetry
                    ),
                )
            selected_index = int(result.selected_index)
            selected_id = (
                int(nominal.candidate_id)
                if selected_index == 0
                else next_id + selected_index - 1
            )
            sampling_name = result.selected_family
            selected_meta = dict(nominal_meta)
            selected_meta.update({
                "sampling": sampling_name,
                "parent": int(nominal.candidate_id),
                "device_candidate_index": selected_index,
            })
            meta[selected_id] = selected_meta
            selected_row = dict(result.selected_row)
            rows = {selected_id: selected_row}
            selected_valid = bool(
                selected_row.get("device_program_valid", False)
            )
            best = (
                0.0 if selected_valid else float(_INVALID_RANK),
                float(selected_row.get("device_program_cost", float("inf"))),
                0.0,
            )
            n_valid = int(result.n_valid)
            stats.pool_size = int(result.total_rollouts or result.pool_size)
            stats.final_n_valid = n_valid
            replay_packet = (
                result.diagnostics.get("exact_replay")
                if isinstance(result.diagnostics, dict)
                else None
            )
            if (
                isinstance(replay_packet, dict)
                and "accepted_for_execution" in replay_packet
            ):
                accepted = replay_packet["accepted_for_execution"]
                if not isinstance(accepted, (bool, np.bool_)):
                    raise RuntimeError(
                        "exact replay accepted_for_execution must be Boolean"
                    )
                stats.execution_plan_accepted = bool(accepted)
            stats.flat_objective = bool(result.flat_objective)
            initial_contact = selected_row.get(
                "initial_robot_scene_contact"
            )
            if not isinstance(initial_contact, (bool, np.bool_)):
                raise RuntimeError(
                    "GPU selected row omitted Boolean "
                    "initial_robot_scene_contact"
                )
            stats.initial_robot_scene_contact = bool(initial_contact)
            if stats.flat_objective:
                logger.warning(
                    "[sampler] step %d: FLAT device objective over %d "
                    "candidates; refusing motion",
                    chunk_idx,
                    stats.pool_size,
                )
                stats.final_n_valid = 0
            elif (
                n_valid > 0
                and selected_valid
                and stats.execution_plan_accepted is not False
            ):
                best_seq = ControlKnots(
                    candidate_id=selected_id,
                    knots=np.clip(
                        np.asarray(result.selected_knots, dtype=float),
                        ctrl_low,
                        ctrl_high,
                    ),
                )
            stats.batch = {
                "algorithm": (
                    "mppi" if optimizer == "mppi" else
                    "predictive_sampling" if cem_rounds == 1 else "cem"
                ),
                "best_cost": float(best[1]),
                "best_valid": float(selected_valid),
                "batch_mean_cost": float(result.mean_cost),
                "n_valid": float(n_valid),
                "pool_size": float(result.total_rollouts or result.pool_size),
                "samples_per_round": float(
                    result.round_summaries[0].get(
                        "samples", result.pool_size
                    )
                    if result.round_summaries else result.pool_size
                ),
                "num_knots": float(nominal_knots.shape[0]),
                "rounds": float(result.rounds),
                "elite_fraction": (
                    None if optimizer == "mppi" else float(cem_elite_fraction)
                ),
                "elite_count": (
                    None if optimizer == "mppi" else float(
                        max(
                            2,
                            int(math.ceil(
                                cem_elite_fraction * result.pool_size
                            )),
                        )
                    )
                ),
                "mppi_temperature": (
                    float(mppi_temperature) if optimizer == "mppi" else None
                ),
                "arm_velocity_weight": (
                    float(arm_velocity_weight)
                    if optimizer == "mppi" else None
                ),
                **(
                    {"control_profile": control_profile}
                    if control_profile is not None else {}
                ),
                "candidate_cost_telemetry_recorded": bool(
                    record_candidate_cost_telemetry
                ),
                "mppi_execution_mode": (
                    mppi_execution_mode if optimizer == "mppi" else None
                ),
                "mppi_effective_samples": (
                    result.mppi_effective_samples
                    if optimizer == "mppi" else None
                ),
                "mppi_next_temperature": (
                    result.mppi_next_temperature
                    if optimizer == "mppi" else None
                ),
                "mppi_proposal": (
                    (
                        "dial_annealed_normalized_8d_gaussian"
                        if joint_velocity_force_dial_arm is not None
                        else (
                            "mtp_global_local_linear_m3_n50_arm_velocity"
                            if control_profile_uses_joint_velocity_arm(
                                control_profile
                            )
                            else "mtp_global_local_linear_m3_n50_arm_torque"
                        )
                        if result.sampler_mode
                        == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                        else
                        (
                            "gaussian_halton_degree1_arm_velocity_jaw_width_actions"
                            if control_profile
                            == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
                            else "gaussian_halton_degree1_arm_velocity_jaw_effort_actions"
                            if control_profile_uses_joint_velocity_arm(
                                control_profile
                            )
                            else "gaussian_halton_degree1_arm_torque_jaw_effort_actions"
                        )
                    )
                    if optimizer == "mppi" else None
                ),
                "mppi_action_space": (
                    (
                        "joint_velocity_rad_s_plus_continuous_gripper_width_latent"
                        if control_profile
                        == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
                        else "joint_velocity_rad_s_plus_continuous_signed_gripper_effort"
                        if control_profile_uses_joint_velocity_arm(
                            control_profile
                        )
                        else "normalized_joint_torque_plus_continuous_signed_gripper_effort"
                    )
                    if optimizer == "mppi" else None
                ),
                "mppi_prediction_action_dt_s": (
                    float(result.round_summaries[0]["action_dt_s"])
                    if optimizer == "mppi" and result.round_summaries
                    else None
                ),
                "mppi_external_control_dt_s": (
                    float(horizon_steps)
                    * 0.002
                    * float(execution_prefix_frac)
                    if optimizer == "mppi" else None
                ),
                "mppi_arm_control": (
                    (
                        "native_velocity_servo_kv600_ctrlrange_pm0p2_zoh"
                        if control_profile_uses_joint_velocity_arm(
                            control_profile
                        )
                        else "gravity_compensated_direct_joint_torque"
                    )
                    if optimizer == "mppi" else None
                ),
                **(
                    {
                        "mppi_gripper_cost_command": (
                            "signed_latent_open_minus1_closed_plus1"
                        ),
                        "mppi_gripper_actuator_control": (
                            "physical_width_m"
                            if control_profile_uses_width_jaw(control_profile)
                            else "signed_effort_n"
                        ),
                    }
                    if control_profile is not None and optimizer == "mppi"
                    else {}
                ),
                "optimization_rollouts": (
                    float(pool_size - 1) if optimizer == "mppi" else None
                ),
                "validation_rollouts": (
                    1.0 if optimizer == "mppi" else None
                ),
                "min_std_fraction": (
                    None if optimizer == "mppi" else
                    float(cem_min_std_fraction)
                ),
                "action_rate_weight": float(DEFAULT_ACTION_RATE_WEIGHT),
                "round_summaries": list(result.round_summaries),
                "n_nominal": (
                    2.0
                    if joint_velocity_force_dial_arm is not None else 1.0
                ),
                "n_gaussian": float(result.n_gaussian),
                "n_local_gaussian": float(result.n_local_gaussian),
                "n_broad_gaussian": float(result.n_broad_gaussian),
                "sampler_mode": result.sampler_mode,
                "sigma_half_range": sigma_fraction,
                "local_sigma_half_range": local_sigma_fraction,
                "proposal_families": (
                    [
                        "nominal",
                        "mtp_global_tensor_linear",
                        "local_diagonal_gaussian",
                        "best_evaluated_candidate",
                    ]
                    if result.sampler_mode
                    == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    else
                    [
                        "nominal",
                        "diagonal_gaussian",
                        "softmax_weighted_mean",
                    ]
                    if result.sampler_mode == "mppi"
                    else
                    ["mean", "diagonal_gaussian"]
                    if result.sampler_mode == "cem"
                    else ["nominal", "diagonal_gaussian"]
                    if result.sampler_mode == PREDICTIVE_SAMPLER_SINGLE_SCALE
                    else [
                        "nominal",
                        "local_diagonal_gaussian",
                        "broad_diagonal_gaussian",
                    ]
                ),
                "selected_family": result.selected_family,
                "spline": (
                    "zero_order_hold" if optimizer == "mppi" else "linear"
                ),
                "device_seed": float(device_seed),
                **(
                    {"proposal_seed_policy": device_seed_policy}
                    if control_profile is not None else {}
                ),
                "selected_index": float(selected_index),
                "selected_round": float(result.selected_round),
                "selected_table_contact_force_cost": float(
                    selected_row.get("table_contact_force_cost", 0.0)
                ),
                "selected_max_table_contact_force": float(
                    selected_row.get("max_table_contact_force", 0.0)
                ),
            }
            if _prefix_hold_barrier_should_record(
                prefix_hold_barrier_record,
                telemetry_mode=grasp_bridge_telemetry_mode,
            ):
                recorded_barrier = dict(prefix_hold_barrier_record)
                recorded_barrier["telemetry_mode"] = (
                    grasp_bridge_telemetry_mode
                )
                stats.batch["prefix_hold_barrier"] = recorded_barrier
            if result.diagnostics is not None:
                stats.batch["solve_diagnostics"] = result.diagnostics
            if optimizer == "mppi":
                if result.refit_mean is None:
                    raise RuntimeError("MPPI omitted its updated action plan")
                stats.mppi_updated_action_plan = np.asarray(
                    result.refit_mean, dtype=float
                ).copy()
                if result.selected_proposal is not None:
                    stats.mppi_selected_action_plan = np.asarray(
                        result.selected_proposal, dtype=float
                    ).copy()
                    stats.batch["selected_action_plan"] = (
                        stats.mppi_selected_action_plan.tolist()
                    )
            stats.batch["selected_knots"] = np.asarray(
                best_seq.knots if best_seq is not None
                else result.selected_knots,
                dtype=float,
            ).tolist()
            try:
                from oap.program import program_sha256

                stats.batch["program_sha256"] = program_sha256(program)
            except Exception:  # noqa: BLE001 - identity is best-effort here
                stats.batch["program_sha256"] = None
            logger.info(
                "[sampler] optimizer population: mode=%s incumbent=1 "
                "gaussian=%d local=%d broad=%d selected=%s",
                result.sampler_mode,
                result.n_gaussian,
                result.n_local_gaussian,
                result.n_broad_gaussian,
                result.selected_family,
            )
            stats.batch["validity_counts"] = {
                str(name): int(value)
                for name, value in result.validity_counts.items()
            }
            validity_log = (
                logger.warning if n_valid == 0 else logger.info
            )
            validity_log(
                "[sampler] step %d validity rejection counts: %s",
                chunk_idx,
                " ".join(
                    f"{name}={value}"
                    for name, value in sorted(
                        result.validity_counts.items()
                    )
                ),
            )
        else:
            # Explicitly nonproduction: deterministic host proposals and host
            # ranking are retained only for unit-test dependency injection and
            # equal-budget benchmark reference lanes.
            assert rollout_runner is not None
            while len(pool) < int(pool_size):
                noise_fraction = _HOST_REFERENCE_NOISE_FRACTIONS[
                    (len(pool) - 1)
                    % len(_HOST_REFERENCE_NOISE_FRACTIONS)
                ]
                child_meta = dict(nominal_meta)
                child_meta["sampling"] = "independent_perturbation"
                child_meta["parent"] = int(nominal.candidate_id)
                child_meta["noise_fraction"] = float(noise_fraction)
                meta[next_id] = child_meta
                pool.append(ControlKnots(
                    candidate_id=next_id,
                    knots=_perturb_controls(
                        nominal_knots,
                        rng=rng,
                        bounds=bounds,
                        frac=noise_fraction,
                    ),
                ))
                next_id += 1
            stats.pool_size = len(pool)
            rows = rollout_runner(sequences=pool, obj_pose7=obj_pose7)
            score_by_id = {
                int(sequence.candidate_id): _score_row(
                    rows.get(int(sequence.candidate_id)),
                    program=program,
                    stage_idx=stage_idx,
                    anchors=anchors,
                    table_top_z=TABLE_TOP_Z,
                    movable_grounder=movable_grounder,
                    sub_h=sub_h,
                    start_quat_wxyz=start_quat,
                    arm_pen_thresh=arm_pen_thresh,
                    execution_prefix_frac=float(execution_prefix_frac),
                    cost_shaping_profile=cost_shaping_profile,
                )
                for sequence in pool
            }
            scored = sorted(
                pool,
                key=lambda sequence: score_by_id[int(sequence.candidate_id)],
            )
            best_seq = scored[0]
            best = score_by_id[int(best_seq.candidate_id)]
            n_valid = sum(
                score[0] == 0.0 for score in score_by_id.values()
            )
            stats.final_n_valid = int(n_valid)
            finite_costs = [
                score[1] for score in score_by_id.values()
                if np.isfinite(score[1])
            ]
            valid_costs = [
                score[1] for score in score_by_id.values()
                if score[0] == 0.0 and np.isfinite(score[1])
            ]
            objective_costs = (
                valid_costs if len(valid_costs) >= 2 else finite_costs
            )
            if len(objective_costs) > 1:
                spread = max(objective_costs) - min(objective_costs)
                scale = max(
                    1.0,
                    max(abs(cost) for cost in objective_costs),
                )
                if spread <= 1e-9 * scale:
                    stats.flat_objective = True
                    stats.final_n_valid = 0
                    best_seq = None
            stats.batch = {
                "algorithm": "host_reference_only",
                "best_cost": float(best[1]),
                "best_valid": float(1.0 - best[0]),
                "batch_mean_cost": (
                    float(np.mean(finite_costs))
                    if finite_costs
                    else float("inf")
                ),
                "n_valid": float(n_valid),
                "pool_size": float(len(pool)),
                "rounds": 1.0,
                "n_tool_target_contact": float(sum(
                    bool(row.get("tool_target_contact_any_step", False))
                    for row in rows.values()
                    if isinstance(row, dict)
                )),
                "n_robot_target_contact": float(sum(
                    bool(row.get("robot_target_contact_any_step", False))
                    for row in rows.values()
                    if isinstance(row, dict)
                )),
                "n_full_horizon_held": float(sum(
                    int(row.get("object_not_held_steps", -1)) == 0
                    for row in rows.values()
                    if isinstance(row, dict)
                )),
                "n_execution_prefix_held": float(sum(
                    int(row.get("prefix_object_not_held_steps", -1)) == 0
                    for row in rows.values()
                    if isinstance(row, dict)
                )),
                "noise_fraction": float(
                    _HOST_REFERENCE_NOISE_FRACTIONS[0]
                ),
                "noise_fraction_max": float(
                    _HOST_REFERENCE_NOISE_FRACTIONS[-1]
                ),
            }
        logger.info(
            "[sampler] step %d optimizer (%s): best_cost=%.5f "
            "valid=%d/%d mean_cost=%.5f",
            chunk_idx,
            stats.batch["algorithm"],
            best[1],
            n_valid,
            stats.pool_size,
            stats.batch["batch_mean_cost"],
        )
    except Exception as exc:
        stats.failure_reason = _sampling_failure_reason(
            exc,
            default=(
                "gpu_sampling_failed"
                if screen is not None
                else "sampling_failed"
            ),
        )
        stats.elapsed_s = time.perf_counter() - failure_started_s
        logger.warning(
            "[sampler] step %d: GPU sampling failed (%s); "
            "nominal is not authorized for execution",
            chunk_idx,
            exc,
        )
        return candidates, metadata, stats

    stats.ran = True
    stats.elapsed_s = time.perf_counter() - t0
    if best_seq is not None:
        selected = [best_seq]
        selected_row = rows.get(int(best_seq.candidate_id))
        if selected_row is not None:
            selected_meta = dict(
                meta.get(int(best_seq.candidate_id), {})
            )

            def _json_pose(value: Any) -> list[float] | None:
                if value is None:
                    return None
                array = np.asarray(value, dtype=float)
                if array.shape != (7,) or not np.all(np.isfinite(array)):
                    return None
                return array.tolist()

            def _json_pose_map(value: Any) -> dict[str, list[float]]:
                if not isinstance(value, dict):
                    return {}
                return {
                    str(name): pose
                    for name, raw in value.items()
                    if (pose := _json_pose(raw)) is not None
                }

            def _json_finite_scalar(value: Any) -> float | None:
                if value is None:
                    return None
                array = np.asarray(value, dtype=float)
                if array.shape != () or not np.isfinite(array):
                    return None
                return float(array)

            def _json_vector(value):
                if value is None:
                    return None
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    return None
                return [float(x) for x in arr]

            stats.selected_rollout_diagnostics = {
                "candidate_id": int(best_seq.candidate_id),
                **(
                    {
                        "control_profile": control_profile,
                        "prefix_gripper_command_semantics": (
                            "signed_latent_open_minus1_closed_plus1"
                        ),
                        "prefix_ctrl_gripper_semantics": (
                            "physical_width_m"
                            if control_profile_uses_width_jaw(control_profile)
                            else "signed_effort_n"
                        ),
                        "device_gripper_cost_command_semantics": (
                            "signed_latent_open_minus1_closed_plus1"
                        ),
                        "prefix_gripper_command_latent": _json_finite_scalar(
                            selected_row.get("prefix_gripper_command_latent")
                        ),
                        "prefix_gripper_actuator_command": _json_finite_scalar(
                            selected_row.get("prefix_gripper_actuator_command")
                        ),
                        "prefix_gripper_actuator_command_semantics": (
                            "physical_width_m"
                            if control_profile_uses_width_jaw(control_profile)
                            else "signed_effort_n"
                        ),
                    }
                    if control_profile is not None else {}
                ),
                # Full simulator state at the end of the executed prefix.
                # Evidence only; the controller never reads it back. It permits
                # independent replay, rendering, and fixed-settle evaluation.
                "prefix_qpos": _json_vector(selected_row.get("prefix_qpos")),
                "prefix_qvel": _json_vector(selected_row.get("prefix_qvel")),
                "prefix_ctrl": _json_vector(selected_row.get("prefix_ctrl")),
                "prefix_gripper_command": _json_finite_scalar(
                    selected_row.get("prefix_gripper_command")
                ),
                "sampling": selected_meta.get("sampling", "nominal"),
                "parent": selected_meta.get("parent"),
                "device_candidate_index": selected_meta.get(
                    "device_candidate_index"
                ),
                "noise_fraction": selected_meta.get("noise_fraction"),
                "device_program_cost": float(
                    selected_row.get(
                        "device_program_cost",
                        best[1],
                    )
                ),
                "device_program_valid": bool(
                    selected_row.get(
                        "device_program_valid",
                        best[0] == 0.0,
                    )
                ),
                "object_not_held_steps": int(
                    selected_row.get("object_not_held_steps", -1)
                ),
                "object_not_held_fraction": float(
                    selected_row.get("object_not_held_fraction", -1.0)
                ),
                "execution_prefix_steps": int(
                    selected_row.get("execution_prefix_steps", -1)
                ),
                "prefix_object_not_held_steps": int(
                    selected_row.get(
                        "prefix_object_not_held_steps",
                        -1,
                    )
                ),
                "tool_target_contact_any_step": bool(
                    selected_row.get(
                        "tool_target_contact_any_step",
                        False,
                    )
                ),
                "robot_target_contact_any_step": bool(
                    selected_row.get(
                        "robot_target_contact_any_step",
                        False,
                    )
                ),
                "robot_subject_contact_frames": int(
                    selected_row.get("robot_subject_contact_frames", 0)
                ),
                "robot_subject_contact_missing_fraction": (
                    _json_finite_scalar(
                        selected_row.get(
                            "robot_subject_contact_missing_fraction"
                        )
                    )
                ),
                "robot_subject_contact_endpoint": bool(
                    selected_row.get(
                        "robot_subject_contact_endpoint",
                        False,
                    )
                ),
                "subject_target_frames": int(
                    selected_row.get("subject_target_frames", 0)
                ),
                "robot_target_frames": int(
                    selected_row.get("robot_target_frames", 0)
                ),
                "max_obj_table_pen": _json_finite_scalar(
                    selected_row.get("max_obj_table_pen")
                ),
                "max_obj_other_support_pen": _json_finite_scalar(
                    selected_row.get("max_obj_other_support_pen")
                ),
                "max_subject_contact_target_pen": _json_finite_scalar(
                    selected_row.get("max_subject_contact_target_pen")
                ),
                "table_contact_force_cost": _json_finite_scalar(
                    selected_row.get("table_contact_force_cost")
                ),
                "max_table_contact_force": _json_finite_scalar(
                    selected_row.get("max_table_contact_force")
                ),
                "final_two_sided_now": bool(
                    selected_row.get("final_two_sided_now", False)
                ),
                "prefix_pose7": _json_pose(
                    selected_row.get("prefix_pose7")
                ),
                "final_pose7": _json_pose(
                    selected_row.get("final_pose7")
                ),
                "prefix_movable_pose7": _json_pose_map(
                    selected_row.get("prefix_movable_pose7")
                ),
                "final_movable_pose7": _json_pose_map(
                    selected_row.get("final_movable_pose7")
                ),
            }
            if screen is not None:
                prefix_qpos_trace = selected_row.pop(
                    "prefix_qpos_trace",
                    None,
                )
                if record_step_trace and prefix_qpos_trace is None:
                    raise RuntimeError(
                        "GPU step tracing was enabled but the selected device "
                        "winner omitted its post-step qpos trace"
                    )
                prefix_qpos = selected_row.get("prefix_qpos")
                prefix_qvel = selected_row.get("prefix_qvel")
                prefix_ctrl = selected_row.get("prefix_ctrl")
                prefix_carrier_ctrl = selected_row.get("prefix_carrier_ctrl")
                if (
                    prefix_qpos is not None
                    and prefix_qvel is not None
                    and start_state is not None
                ):
                    stats.selected_prefix_state = (
                        screen.selected_prefix_world_state(
                            prefix_qpos=np.asarray(prefix_qpos, dtype=float),
                            prefix_qvel=np.asarray(prefix_qvel, dtype=float),
                            start_state=start_state,
                        )
                    )
                    world_qpos, world_qvel = stats.selected_prefix_state
                    stats.selected_rollout_diagnostics["prefix_qpos"] = (
                        _json_vector(world_qpos)
                    )
                    stats.selected_rollout_diagnostics["prefix_qvel"] = (
                        _json_vector(world_qvel)
                    )
                    if prefix_qpos_trace is not None:
                        from oap.loop.gpu_step_trace import (
                            map_rollout_qpos_trace_to_world,
                        )

                        prefix_qpos_trace_array = np.asarray(prefix_qpos_trace)
                        expected_prefix_steps = int(
                            selected_row.get("execution_prefix_steps", -1)
                        )
                        expected_trace_shape = (
                            expected_prefix_steps,
                            int(screen.model.nq),
                        )
                        if prefix_qpos_trace_array.shape != expected_trace_shape:
                            raise RuntimeError(
                                "selected GPU qpos trace shape does not match "
                                "the adopted execution prefix: "
                                f"{prefix_qpos_trace_array.shape} != "
                                f"{expected_trace_shape}"
                            )
                        stats.selected_prefix_qpos_trace = (
                            map_rollout_qpos_trace_to_world(
                                prefix_qpos_trace_array,
                                start_qpos=np.asarray(start_state[0]),
                                qpos_src=screen.qpos_src,
                                qpos_dst=screen.qpos_dst,
                                rollout_nq=int(screen.model.nq),
                            )
                        )
                        stats.selected_prefix_step_dt_s = float(
                            screen.plan_dt
                        )
                        if not np.array_equal(
                            stats.selected_prefix_qpos_trace[-1],
                            world_qpos,
                        ):
                            raise RuntimeError(
                                "selected GPU qpos trace endpoint does not "
                                "equal the adopted prefix endpoint"
                            )
                if prefix_carrier_ctrl is not None:
                    stats.selected_prefix_control = (
                        screen.selected_prefix_world_control(
                            prefix_ctrl=np.asarray(
                                prefix_carrier_ctrl, dtype=float
                            ),
                            start_control=np.asarray(
                                world.data.ctrl,
                                dtype=float,
                            ),
                        )
                    )
                    stats.selected_rollout_diagnostics[
                        "prefix_carrier_ctrl"
                    ] = _json_vector(stats.selected_prefix_control)
                elif prefix_ctrl is not None:
                    stats.selected_prefix_control = (
                        screen.selected_prefix_world_control(
                            prefix_ctrl=np.asarray(prefix_ctrl, dtype=float),
                            start_control=np.asarray(
                                world.data.ctrl,
                                dtype=float,
                            ),
                        )
                    )
                    stats.selected_rollout_diagnostics["prefix_ctrl"] = (
                        _json_vector(stats.selected_prefix_control)
                    )
                prefix_contact = selected_row.get(
                    "prefix_execution_contact"
                )
                if isinstance(prefix_contact, dict):
                    stats.selected_prefix_contact = dict(prefix_contact)
        meta[int(best_seq.candidate_id)] = dict(
            meta.get(int(best_seq.candidate_id), {}),
            sampling_output="best_valid_optimizer_output",
        )
    else:
        selected = list(candidates)

    idx = max(0, min(int(stage_idx), len(program.stages) - 1))
    log_call_site(episode, CallSite.TRAJECTORY_COST, program, payload={
        "chunk_index": int(chunk_idx),
        "stage_index": idx,
        "stage_name": program.stages[idx].name,
        "planner": (
            "cem_gpu_warp_exact_joint"
            if stats.batch.get("algorithm") == "cem"
            else "predictive_sampling_gpu_warp_exact_joint"
        ),
        "pool": stats.pool_size,
        "batch": stats.batch,
        "selected_rollout": stats.selected_rollout_diagnostics,
        "flat_objective": bool(stats.flat_objective),
        "elapsed_s": stats.elapsed_s,
    })
    logger.info(
        "[sampler] step %d: %d rollouts (%s) in %.1fs "
        "(joint-space; exact GPU simulator inside the optimizer)",
        chunk_idx,
        stats.pool_size,
        stats.batch.get("algorithm", "predictive_sampling"),
        stats.elapsed_s,
    )
    return selected, meta, stats
