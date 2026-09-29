"""JAX-native TaskProgram cost for batched physics rollouts.

The production rollout already owns batched subject, movable-body, and robot
trajectories on the accelerator.  This module evaluates a :class:`Stage`
directly on those arrays, before any candidate is copied to the host.

``make_device_trajectory_cost`` separates the fixed predicate/anchor topology
from each cycle's grounded anchor values and start poses.  One static JAX
function is cached per topology while the grounding is passed as a fixed-shape
dynamic pytree, so cache reuse can never retain stale observed poses.  The
returned evaluator contains only JAX array operations, returns one task cost
per candidate, and has no candidate loop or NumPy/CPU fallback.

The grounding convention intentionally matches
``oap.loop.sampling._anchors_at``:

* the subject and every anchor attached to ``"subject"`` follow the subject
  pose trajectory;
* anchors attached to a named movable body follow that body's pose trajectory;
* all other anchors remain fixed;
* ``gripper_center`` point and axis come from rollout FK (the axis is the
  planning site's measured local +Z direction in world coordinates);
  ``gripper_closing_axis`` is the same site's measured local +Y / finger
  separation direction. ``gripper_width`` comes from the simulated joint
  state, while ``gripper_command`` comes from the candidate control target;
  neither is inferred from the other.
* every typed ``terminal`` residual is evaluated once, at the predicted
  full-horizon endpoint. The executable-prefix endpoint is not a planning
  objective. Measured stage advancement uses only fresh tracking state;
* ``object_held`` is a finite cost everywhere, for every stage that names it,
  with no acquisition-vs-retention split. Running ``object_held`` adds the
  exact fraction of full-horizon physics steps without the declared two-sided
  hold; terminal ``object_held`` adds one exact Boolean residual at the
  predicted horizon endpoint. Losing the object is therefore ranked rather
  than declared infeasible, so
  MPC can make progress from an empty grasp and can recover a slipped grasp
  instead of finding an empty feasible set. Stage advancement is unaffected: it
  still requires freshly measured terminal evidence, ``object_held`` included.
* running ``contact`` and ``no_contact`` receive exact full-physics-horizon
  reductions and remain finite typed objective terms. Terminal ``contact``
  reads only the exact sampled H-1 full-physics endpoint Boolean; an earlier
  contact or a positive all-step fraction cannot satisfy it. Legacy
  ``tool_mediation`` is compiled to the same finite mechanism residuals for
  compatibility. Contact events are never approximated from the bounded pose
  sample grid.

Unsupported predicates or incomplete grounding raise before a rollout can be
authorized.  ``TemporalHold`` is unwrapped for planning cost, exactly as in
``program.interpreter.trajectory_cost``; its frame-count semantics remain a
measured terminal-verification concern.  ``NonGeometric`` contributes zero,
also matching the CPU planning cost.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from typing import Any, NamedTuple

import numpy as np

from .anchors import Anchor, AnchorSet
from .cost_shaping import (
    TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
    continuous_geometric_predicate_identity,
    stage_cost_shaping_plan,
    validate_cost_shaping_profile,
)
from .interpreter import planning_cost_scales
from .predicates import (
    AboveBy,
    AlongAxisGap,
    AngleAboutAxis,
    AtRest,
    AxisAngle,
    AxisParallel,
    CaptureDistance,
    Contact,
    GripperCommand,
    GripperState,
    InRegion,
    InteractionAlign,
    MinDistance,
    NoContact,
    NonGeometric,
    NotHeld,
    ObjectHeld,
    OnPlane,
    PointOnLine,
    PointPointDistance,
    Predicate,
    RelativeOrientation,
    RotationProgress,
    RelativePosition,
    RobotSubjectContact,
    SignedAxisGap,
    SupportedOn,
    TemporalHold,
    ToolMediation,
)
from .program import Stage


class DeviceCostGroundingError(ValueError):
    """The stage cannot be grounded completely in the device rollout."""


def _zero_band(predicate: Any) -> Any:
    """Return a copy of ``predicate`` with its tolerance band set to zero.

    Evaluating the same residual at zero band recovers the raw geometric
    quantity that the hinge discards inside the band. Band-free predicates
    are returned unchanged and receive exactly the same Eq. (3) norm.
    """
    import dataclasses as _dc

    for field in ("tol", "tol_deg"):
        if hasattr(predicate, field):
            return _dc.replace(predicate, **{field: 0.0})
    return predicate


def _tool_mediation_positive_contact_residual(
    contact_fraction: Any,
    *,
    shaping_profile: str | None,
) -> Any:
    """Return the registered finite positive tool-contact residual."""
    if shaping_profile == TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2:
        return 1.0 - contact_fraction
    return (contact_fraction <= 0.0).astype(contact_fraction.dtype)


DeviceCostFn = Callable[
    [
        Any,
        Mapping[str, Any] | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any,
    ],
    Any,
]


@dataclass(frozen=True)
class DeviceCostSpec:
    """Hashable static program topology used by the scorer caches."""

    stage_json: str
    anchor_layout: tuple[tuple[str, str | None, bool, bool, bool], ...]
    movable_owners: tuple[str, ...]


class DeviceCostParams(NamedTuple):
    """Fixed-shape per-cycle grounding passed to JAX as dynamic arguments."""

    anchor_points: Any
    anchor_axes: Any
    anchor_regions: Any
    running_scales: Any
    terminal_scales: Any
    subject_start_pose7: Any
    movable_start_pose7: Any
    table_top_z: Any
    sub_h: Any
    dt: Any
    # Physical seconds between pose samples; required only by AtRest (None
    # elsewhere -- an empty pytree leaf, never traced).
    sample_dt_s: Any = None
    # V11 disturbance penalty (env OAP_DISTURB_PENALTY_W): per
    # UNDECLARED movable name -> (weight, cycle-start pose7). None = off;
    # keyed by name so consumer lanes cannot mis-order.
    disturb_movables: Any = None


@dataclass(eq=False, frozen=True)
class DeviceCostEvaluator:
    """One static evaluator plus the current cycle's dynamic grounding.

    ``function`` returns the ``(N,)`` total used for ranking/selection.
    ``breakdown_function`` returns ``(total, per_term)`` where ``per_term`` has
    shape ``(n_terms, N)`` in ``term_descriptors`` order -- a small diagnostic
    copy, never a second objective. The total keeps its original accumulation
    order, so term sums may differ from it by float rounding only.
    ``trajectory_breakdown_function`` returns the raw terminal residuals with
    shape ``(n_terminal, N, T)`` on the caller's pose-sample grid. It is also
    diagnostic-only and shares the objective's residual interpreter.
    """

    spec: DeviceCostSpec
    function: DeviceCostFn
    params: DeviceCostParams
    breakdown_function: DeviceCostFn | None = None
    trajectory_breakdown_function: DeviceCostFn | None = None
    term_descriptors: tuple[dict[str, Any], ...] = ()

    def __call__(
        self,
        subject_pose_traj7: Any,
        movable_pose_traj7: Mapping[str, Any] | None = None,
        gripper_center_traj3: Any | None = None,
        gripper_width_traj: Any | None = None,
        object_held_traj: Any | None = None,
        tool_target_contact_traj: Any | None = None,
        robot_target_contact_traj: Any | None = None,
        gripper_axis_traj3: Any | None = None,
        gripper_closing_axis_traj3: Any | None = None,
        object_not_held_fraction: Any | None = None,
        tool_target_contact_fraction: Any | None = None,
        robot_target_contact_any_step: Any | None = None,
        gripper_command_traj: Any | None = None,
        robot_subject_contact_missing_fraction: Any | None = None,
        robot_subject_contact_endpoint: Any | None = None,
    ) -> Any:
        return self.function(
            subject_pose_traj7,
            movable_pose_traj7,
            gripper_center_traj3,
            gripper_width_traj,
            object_held_traj,
            tool_target_contact_traj,
            robot_target_contact_traj,
            gripper_axis_traj3,
            gripper_closing_axis_traj3,
            object_not_held_fraction,
            tool_target_contact_fraction,
            robot_target_contact_any_step,
            gripper_command_traj,
            robot_subject_contact_missing_fraction,
            robot_subject_contact_endpoint,
            self.params,
        )

    def breakdown(
        self,
        subject_pose_traj7: Any,
        movable_pose_traj7: Mapping[str, Any] | None = None,
        gripper_center_traj3: Any | None = None,
        gripper_width_traj: Any | None = None,
        object_held_traj: Any | None = None,
        tool_target_contact_traj: Any | None = None,
        robot_target_contact_traj: Any | None = None,
        gripper_axis_traj3: Any | None = None,
        gripper_closing_axis_traj3: Any | None = None,
        object_not_held_fraction: Any | None = None,
        tool_target_contact_fraction: Any | None = None,
        robot_target_contact_any_step: Any | None = None,
        gripper_command_traj: Any | None = None,
        robot_subject_contact_missing_fraction: Any | None = None,
        robot_subject_contact_endpoint: Any | None = None,
    ) -> Any:
        if self.breakdown_function is None:
            raise TypeError(
                "this evaluator predates per-term breakdown; rebuild it with "
                "make_device_trajectory_cost"
            )
        return self.breakdown_function(
            subject_pose_traj7,
            movable_pose_traj7,
            gripper_center_traj3,
            gripper_width_traj,
            object_held_traj,
            tool_target_contact_traj,
            robot_target_contact_traj,
            gripper_axis_traj3,
            gripper_closing_axis_traj3,
            object_not_held_fraction,
            tool_target_contact_fraction,
            robot_target_contact_any_step,
            gripper_command_traj,
            robot_subject_contact_missing_fraction,
            robot_subject_contact_endpoint,
            self.params,
        )

    def trajectory_breakdown(
        self,
        subject_pose_traj7: Any,
        movable_pose_traj7: Mapping[str, Any] | None = None,
        gripper_center_traj3: Any | None = None,
        gripper_width_traj: Any | None = None,
        object_held_traj: Any | None = None,
        tool_target_contact_traj: Any | None = None,
        robot_target_contact_traj: Any | None = None,
        gripper_axis_traj3: Any | None = None,
        gripper_closing_axis_traj3: Any | None = None,
        object_not_held_fraction: Any | None = None,
        tool_target_contact_fraction: Any | None = None,
        robot_target_contact_any_step: Any | None = None,
        gripper_command_traj: Any | None = None,
        robot_subject_contact_missing_fraction: Any | None = None,
        robot_subject_contact_endpoint: Any | None = None,
    ) -> Any:
        """Return raw terminal residuals on every supplied pose sample."""
        if self.trajectory_breakdown_function is None:
            raise TypeError(
                "this evaluator predates terminal trajectory breakdown; rebuild "
                "it with make_device_trajectory_cost"
            )
        return self.trajectory_breakdown_function(
            subject_pose_traj7,
            movable_pose_traj7,
            gripper_center_traj3,
            gripper_width_traj,
            object_held_traj,
            tool_target_contact_traj,
            robot_target_contact_traj,
            gripper_axis_traj3,
            gripper_closing_axis_traj3,
            object_not_held_fraction,
            tool_target_contact_fraction,
            robot_target_contact_any_step,
            gripper_command_traj,
            robot_subject_contact_missing_fraction,
            robot_subject_contact_endpoint,
            self.params,
        )


_EVALUATOR_CACHE: dict[
    DeviceCostSpec,
    tuple[DeviceCostFn, DeviceCostFn, DeviceCostFn],
] = {}


_SUPPORTED = (
    PointPointDistance,
    CaptureDistance,
    RelativePosition,
    MinDistance,
    AboveBy,
    OnPlane,
    PointOnLine,
    AxisParallel,
    AxisAngle,
    RelativeOrientation,
    RotationProgress,
    AngleAboutAxis,
    InRegion,
    SignedAxisGap,
    GripperCommand,
    GripperState,
    InteractionAlign,
    AlongAxisGap,
    Contact,
    NoContact,
    RobotSubjectContact,
    ToolMediation,
    NonGeometric,
    ObjectHeld,
    SupportedOn,
    AtRest,
    NotHeld,
)


def _planning_predicate(predicate: Predicate) -> Predicate:
    """Unwrap TemporalHold for running/terminal planning cost."""
    while isinstance(predicate, TemporalHold):
        predicate = predicate.inner
    if type(predicate) not in _SUPPORTED:
        raise TypeError(
            "device cost does not support predicate "
            f"{type(predicate).__name__!r}; refusing CPU fallback"
        )
    return predicate


def _requirements(
    predicate: Predicate,
) -> tuple[set[str], set[str], set[str], bool, bool]:
    """Return anchor requirements and physical-width/command flags."""
    points: set[str] = set()
    axes: set[str] = set()
    regions: set[str] = set()
    needs_width = False
    needs_command = False

    if isinstance(
        predicate,
        (
            PointPointDistance,
            RelativePosition,
            MinDistance,
            AboveBy,
        ),
    ):
        points.update((predicate.a, predicate.b))
    elif isinstance(predicate, CaptureDistance):
        points.update(("gripper_center", predicate.subject))
    elif isinstance(predicate, OnPlane):
        points.update((predicate.a, predicate.b))
    elif isinstance(predicate, PointOnLine):
        points.update((predicate.a, predicate.b))
        axes.add(predicate.dir_anchor)
    elif isinstance(predicate, (AxisParallel, AxisAngle)):
        axes.update((predicate.a, predicate.b))
    elif isinstance(predicate, (RelativeOrientation, RotationProgress)):
        axes.update(predicate.referenced_anchors())
    elif isinstance(predicate, AngleAboutAxis):
        axes.update((predicate.a, predicate.b, predicate.ref))
    elif isinstance(predicate, InRegion):
        points.update((predicate.a, predicate.region))
        regions.add(predicate.region)
    elif isinstance(predicate, SignedAxisGap):
        points.update((predicate.a, predicate.b))
    elif isinstance(predicate, ObjectHeld):
        points.add(predicate.object_anchor)
    elif isinstance(predicate, GripperState):
        needs_width = True
    elif isinstance(predicate, GripperCommand):
        needs_command = True
    elif isinstance(predicate, InteractionAlign):
        points.update((predicate.a, predicate.b))
        axes.add(predicate.a if predicate.axis_a == "self" else predicate.axis_a)
        axes.add(predicate.b if predicate.axis_b == "self" else predicate.axis_b)
    elif isinstance(predicate, AlongAxisGap):
        points.update((predicate.a, predicate.c))
        axes.add(predicate.b)
    elif isinstance(predicate, (Contact, NoContact)):
        points.update(
            name for name in (predicate.a, predicate.b) if name != "robot"
        )
    elif isinstance(predicate, RobotSubjectContact):
        points.add("subject")
    elif isinstance(predicate, ToolMediation):
        # Ground both declared roles, but add no distance proxy. Exact contact
        # feasibility is evaluated by the rollout gate.
        points.update((predicate.tool, predicate.target))
    elif isinstance(predicate, SupportedOn):
        points.update(
            f"{predicate.subject}_bottom_corner_{index}" for index in range(4)
        )
        points.update((predicate.support, f"{predicate.support}_top"))
        axes.update(
            (f"{predicate.support}_axis", f"{predicate.support}_normal")
        )
        regions.add(predicate.support)
    elif isinstance(predicate, AtRest):
        points.add(f"{predicate.subject}_center")
    elif isinstance(predicate, NotHeld):
        points.add(predicate.object_anchor)
    elif isinstance(predicate, NonGeometric):
        pass
    else:  # Guard against a future class being added without device semantics.
        raise TypeError(
            "device cost does not support predicate "
            f"{type(predicate).__name__!r}; refusing CPU fallback"
        )
    return points, axes, regions, needs_width, needs_command


def _finite_pose7(value: Any, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise DeviceCostGroundingError(f"{label} must be one finite pose7")
    return pose


def _validate_anchor(anchor: Anchor, name: str, *, need_axis: bool, need_region: bool) -> None:
    if not np.all(np.isfinite(anchor.point)):
        raise DeviceCostGroundingError(f"anchor {name!r} has a non-finite point")
    if need_axis:
        if anchor.axis is None:
            raise DeviceCostGroundingError(f"anchor {name!r} has no axis")
        if not np.all(np.isfinite(anchor.axis)) or np.linalg.norm(anchor.axis) <= 1e-9:
            raise DeviceCostGroundingError(f"anchor {name!r} has an invalid axis")
    if need_region:
        if anchor.region_half is None:
            raise DeviceCostGroundingError(
                f"anchor {name!r} is not a grounded region"
            )
        if not np.all(np.isfinite(anchor.region_half)):
            raise DeviceCostGroundingError(
                f"anchor {name!r} has non-finite region extents"
            )
        if np.any(np.asarray(anchor.region_half, dtype=float) <= 0.0):
            raise DeviceCostGroundingError(
                f"anchor {name!r} must have positive region extents"
            )


def _v11_disturb_movables(stage, movable_start_pose7):
    """V11: {undeclared movable name: (weight, cycle-start pose7)} or None.
    Active only when OAP_DISTURB_PENALTY_W > 0. The declared list
    comes from the stage's movable_objects field (empty on pre-v11
    programs, penalizing every non-subject movable -- by design)."""
    import os as _os
    try:
        w = float(_os.environ.get("OAP_DISTURB_PENALTY_W", "0") or 0)
    except ValueError:
        w = 0.0
    if w <= 0.0 or not movable_start_pose7:
        return None
    declared = {str(n) for n in (getattr(stage, "movable_objects", ()) or ())}
    out = {
        str(name): (np.asarray(w, dtype=float),
                    np.asarray(pose7, dtype=float).reshape(7))
        for name, pose7 in dict(movable_start_pose7).items()
        if str(name) not in declared
    }
    return out or None


def make_device_trajectory_cost(
    stage: Stage,
    anchors: AnchorSet,
    *,
    cost_scale_anchors: AnchorSet | None = None,
    subject_start_pose7: Any,
    movable_start_pose7: Mapping[str, Any] | None = None,
    table_top_z: float,
    sub_h: float,
    dt: float | None = None,
    sample_dt_s: float | None = None,
    cost_shaping_profile: str | None = None,
    pick_v8_causal_profile: str | None = None,
) -> DeviceCostEvaluator:
    """Compile a stage into a JAX-traceable, batched task-cost function.

    Args:
        stage: Fixed program stage. Continuous ``running`` predicates are
            integrated over every sampled state, Boolean running predicates use
            exact full-physics-step counts, and ``terminal`` predicates are
            evaluated at the last state.
        anchors: Grounded chunk-start anchors.
        cost_scale_anchors: Optional stage-entry grounding used only to freeze
            dimensionless residual scales across receding-horizon cycles.
            Dynamic anchor values still come from ``anchors`` and the latest
            observed start poses.
        subject_start_pose7: Chunk-start subject ``[xyz, qwxyz]`` pose.
        movable_start_pose7: Chunk-start poses for every non-subject body whose
            attached anchors the stage references.
        table_top_z, sub_h: Subject-z convention used by CPU ``_anchors_at``.
        dt: Deprecated compatibility argument; if supplied must equal ``1/T``.
            The running objective is intrinsically a mean, as in Eq. (5).

    Returns:
        A :class:`DeviceCostEvaluator` callable as
        ``cost(subject_pose_traj7, movable_pose_traj7,
        gripper_center_traj3, gripper_width_traj, object_held_traj,
        ..., gripper_axis_traj3, gripper_closing_axis_traj3,
        object_not_held_fraction,
        tool_target_contact_fraction, robot_target_contact_any_step,
        gripper_command_traj) -> (N,)``.
        Point and axis trajectory shapes are both ``(N,T,3)``. Exact terminal
        held evidence remains ``(N,T)`` on that sample grid; each full-step
        reduction has shape ``(N,)``. The executable-prefix state stays in the
        rollout/execution layer and is never an input to this objective.
        ``function`` and ``spec`` are static and reusable; ``params`` holds the
        current dynamic grounding.
        With an explicit cost-shaping profile, the total is the post-shaping
        score used for selection while ``breakdown_function`` retains each raw
        pre-shaping predicate contribution. Static multipliers and identities
        in ``term_descriptors`` close the two representations exactly. The
        default keeps the historical descriptor and cache schemas unchanged.

    Raises:
        TypeError: an unsupported predicate would otherwise require a fallback.
        DeviceCostGroundingError: a referenced anchor/body/FK signal is absent
            or invalid.
    """
    shaping_profile = validate_cost_shaping_profile(
        cost_shaping_profile,
        allow_none=True,
    )
    shaping_plan = (
        None
        if shaping_profile is None
        else stage_cost_shaping_plan(stage, shaping_profile)
    )
    running_multipliers = (
        (1.0,) * len(stage.running)
        if shaping_plan is None
        else shaping_plan.running_multipliers
    )
    terminal_multipliers = (
        (1.0,) * len(stage.terminal)
        if shaping_plan is None
        else shaping_plan.terminal_multipliers
    )
    # Terminal multipliers are applied in the accumulation below, mirroring
    # the running lane.  The registered value must still be finite and
    # non-negative; which values may be registered at all is gated by the
    # cost-shaping whitelist, not here.
    for multiplier in terminal_multipliers:
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise DeviceCostGroundingError(
                "terminal cost multipliers must be finite and non-negative"
            )
    start_subject = _finite_pose7(subject_start_pose7, "subject_start_pose7")
    start_movables = {
        str(name): _finite_pose7(pose, f"movable_start_pose7[{name!r}]")
        for name, pose in (movable_start_pose7 or {}).items()
    }
    if not math.isfinite(float(table_top_z)):
        raise DeviceCostGroundingError("table_top_z must be finite")
    if not math.isfinite(float(sub_h)) or float(sub_h) < 0.0:
        raise DeviceCostGroundingError("sub_h must be finite and non-negative")
    if dt is not None and (not math.isfinite(float(dt)) or float(dt) <= 0.0):
        raise DeviceCostGroundingError("dt must be finite and positive, or omitted")

    original_running = tuple(stage.running)
    original_terminal = tuple(stage.terminal)
    for predicate in original_running + original_terminal:
        planned = _planning_predicate(predicate)
        try:
            planned.validate()
        except ValueError as exc:
            raise DeviceCostGroundingError(
                f"invalid {planned.type} predicate: {exc}"
            ) from exc
        if isinstance(planned, (RelativeOrientation, RotationProgress)):
            frame_failures = planned.frame_contract_failures(anchors)
            if frame_failures:
                raise DeviceCostGroundingError(
                    f"{planned.type} frame contract invalid: "
                    + ";".join(frame_failures)
                )
        if isinstance(predicate, TemporalHold) and isinstance(planned, ObjectHeld):
            raise DeviceCostGroundingError(
                "ObjectHeld cannot be wrapped in TemporalHold; held evidence "
                "uses exact full-step running evidence or exact endpoint "
                "evidence"
            )
    for predicate in original_terminal:
        if isinstance(predicate, TemporalHold) and isinstance(
            _planning_predicate(predicate),
            (Contact, NoContact),
        ):
            raise DeviceCostGroundingError(
                "terminal Contact/NoContact cannot be wrapped in TemporalHold; "
                "contact is exact evidence at one full-horizon endpoint"
            )
    running = tuple(_planning_predicate(p) for p in original_running)
    terminal = tuple(_planning_predicate(p) for p in original_terminal)
    if any(isinstance(predicate, GripperState) for predicate in running):
        raise DeviceCostGroundingError(
            "GripperState is terminal-only measured state; running cost uses "
            "GripperCommand"
        )
    running_held = tuple(
        predicate for predicate in running
        if isinstance(predicate, ObjectHeld)
    )
    terminal_held = tuple(
        predicate for predicate in terminal
        if isinstance(predicate, ObjectHeld)
    )
    running_not_held = tuple(
        predicate for predicate in running
        if isinstance(predicate, NotHeld)
    )
    terminal_not_held = tuple(
        predicate for predicate in terminal
        if isinstance(predicate, NotHeld)
    )
    if len(running_held) > 1:
        raise DeviceCostGroundingError(
            "a stage may contain at most one running ObjectHeld predicate"
        )
    if len(terminal_held) > 1:
        raise DeviceCostGroundingError(
            "a stage may contain at most one terminal ObjectHeld predicate"
        )
    if len(running_not_held) > 1 or len(terminal_not_held) > 1:
        raise DeviceCostGroundingError(
            "a stage may contain at most one NotHeld predicate per scope"
        )
    if (running_held and running_not_held) or (
        terminal_held and terminal_not_held
    ):
        raise DeviceCostGroundingError(
            "ObjectHeld and NotHeld contradict each other within one scope"
        )
    at_rest_predicates = tuple(
        predicate for predicate in running + terminal
        if isinstance(predicate, AtRest)
    )
    if at_rest_predicates and (
        sample_dt_s is None
        or not math.isfinite(float(sample_dt_s))
        or float(sample_dt_s) <= 0.0
    ):
        raise DeviceCostGroundingError(
            "at_rest needs the physical pose-sample spacing sample_dt_s"
        )
    running_mediation = tuple(
        predicate for predicate in running
        if isinstance(predicate, ToolMediation)
    )
    terminal_mediation = tuple(
        predicate for predicate in terminal
        if isinstance(predicate, ToolMediation)
    )
    if terminal_mediation:
        raise DeviceCostGroundingError(
            "ToolMediation is a running cost and cannot be terminal"
        )
    if len(running_mediation) > 1:
        raise DeviceCostGroundingError(
            "a stage may contain at most one running ToolMediation predicate"
        )
    needs_tool_mediation = bool(running_mediation)
    requires_tool_target_contact = any(
        predicate.require_tool_target_contact
        for predicate in running_mediation
    )
    running_contact_relations = tuple(
        predicate
        for predicate in running
        if isinstance(predicate, (Contact, NoContact))
    )
    terminal_contact_relations = tuple(
        predicate
        for predicate in terminal
        if isinstance(predicate, (Contact, NoContact))
    )
    terminal_no_contact = tuple(
        predicate
        for predicate in terminal_contact_relations
        if isinstance(predicate, NoContact)
    )
    terminal_subject_contact = tuple(
        predicate
        for predicate in terminal_contact_relations
        if isinstance(predicate, Contact)
    )
    if terminal_no_contact:
        raise DeviceCostGroundingError(
            "terminal NoContact lacks the reviewed exact endpoint contract"
        )
    if len(terminal_subject_contact) > 1:
        raise DeviceCostGroundingError(
            "a stage may contain at most one terminal Contact"
        )
    if terminal_subject_contact and "robot" in (
        terminal_subject_contact[0].a,
        terminal_subject_contact[0].b,
    ):
        raise DeviceCostGroundingError(
            "terminal Contact must relate the controlled subject to one target"
        )
    needs_subject_contact_fraction = (
        requires_tool_target_contact
        or any(
            "robot" not in (predicate.a, predicate.b)
            for predicate in running_contact_relations
        )
    )
    needs_subject_contact_endpoint = bool(terminal_subject_contact)
    needs_subject_contact = (
        needs_subject_contact_fraction or needs_subject_contact_endpoint
    )
    needs_robot_contact = (
        needs_tool_mediation
        or any(
            "robot" in (predicate.a, predicate.b)
            for predicate in running_contact_relations
        )
    )
    running_robot_subject_contact = tuple(
        predicate
        for predicate in running
        if isinstance(predicate, RobotSubjectContact)
    )
    terminal_robot_subject_contact = tuple(
        predicate
        for predicate in terminal
        if isinstance(predicate, RobotSubjectContact)
    )
    if (
        len(running_robot_subject_contact) > 1
        or len(terminal_robot_subject_contact) > 1
    ):
        raise DeviceCostGroundingError(
            "a stage may contain at most one RobotSubjectContact per scope"
        )
    needs_robot_subject_contact = bool(
        running_robot_subject_contact or terminal_robot_subject_contact
    )
    point_names: set[str] = set()
    axis_names: set[str] = set()
    region_names: set[str] = set()
    needs_width = False
    needs_command = False
    for predicate in running + terminal:
        req_p, req_a, req_r, req_w, req_c = _requirements(predicate)
        point_names.update(req_p)
        axis_names.update(req_a)
        region_names.update(req_r)
        needs_width = needs_width or req_w
        needs_command = needs_command or req_c

    required_names = point_names | axis_names | region_names
    runtime_fk_names = {"gripper_center", "gripper_closing_axis"}
    needs_gripper_center = "gripper_center" in required_names
    needs_gripper_closing_axis = "gripper_closing_axis" in axis_names
    ordinary_names = required_names - runtime_fk_names
    anchor_defs: dict[str, Anchor] = {}
    required_movable_owners: set[str] = set()
    for name in ordinary_names:
        anchor = anchors.get(name)
        if anchor is None:
            raise DeviceCostGroundingError(f"missing referenced anchor {name!r}")
        _validate_anchor(
            anchor,
            name,
            need_axis=name in axis_names,
            need_region=name in region_names,
        )
        owner = anchor.attached_to
        if anchor.dynamic and owner is None:
            raise DeviceCostGroundingError(
                f"dynamic anchor {name!r} has no attached_to body"
            )
        if owner and owner != "subject":
            if owner not in start_movables:
                raise DeviceCostGroundingError(
                    f"anchor {name!r} follows movable {owner!r}, but its "
                    "chunk-start pose is absent"
                )
            required_movable_owners.add(owner)
        anchor_defs[name] = anchor

    if terminal_subject_contact:
        terminal_relation = terminal_subject_contact[0]
        actor = anchor_defs[terminal_relation.a]
        target = anchor_defs[terminal_relation.b]
        if (
            terminal_relation.a != "subject"
            and actor.attached_to != "subject"
        ):
            raise DeviceCostGroundingError(
                "terminal Contact actor must resolve to the canonical subject"
            )
        terminal_target_owner = target.attached_to
        if (
            terminal_relation.b == "subject"
            or terminal_target_owner in (None, "subject")
        ):
            raise DeviceCostGroundingError(
                "terminal Contact target must resolve to one distinct exact "
                "rollout body"
            )

        # One sampled subject-target channel can represent exactly one body.
        # When the stage also has running subject contact guidance, lock both
        # scopes to that same target instead of silently OR-ing other bodies.
        exact_target_owners = {str(terminal_target_owner)}
        for relation in running_contact_relations:
            if "robot" in (relation.a, relation.b):
                continue
            role_anchors = (
                anchor_defs[relation.a],
                anchor_defs[relation.b],
            )
            subject_indices = [
                index
                for index, (name, anchor) in enumerate(
                    zip((relation.a, relation.b), role_anchors, strict=True)
                )
                if name == "subject" or anchor.attached_to == "subject"
            ]
            if len(subject_indices) != 1:
                raise DeviceCostGroundingError(
                    "subject contact relation must contain exactly one "
                    "canonical subject actor"
                )
            relation_target = role_anchors[1 - subject_indices[0]]
            if relation_target.attached_to in (None, "subject"):
                raise DeviceCostGroundingError(
                    "subject contact target must resolve to one distinct exact "
                    "rollout body"
                )
            exact_target_owners.add(str(relation_target.attached_to))
        for mediation in running_mediation:
            mediation_target = anchor_defs[mediation.target]
            if mediation_target.attached_to in (None, "subject"):
                raise DeviceCostGroundingError(
                    "tool mediation target must resolve to one distinct exact "
                    "rollout body"
                )
            exact_target_owners.add(str(mediation_target.attached_to))
        if len(exact_target_owners) != 1:
            raise DeviceCostGroundingError(
                "terminal and running subject contact relations must share one "
                "exact rollout target body"
            )

    subject_template = anchors.get("subject")
    subject_is_needed = any(
        name == "subject" or anchor_defs[name].attached_to == "subject"
        for name in ordinary_names
    )
    if subject_is_needed:
        if subject_template is None:
            raise DeviceCostGroundingError(
                "a subject-attached anchor is referenced but 'subject' is absent"
            )
        _validate_anchor(subject_template, "subject", need_axis=False, need_region=False)

    for owner in required_movable_owners:
        owner_anchor = anchors.get(owner)
        if owner_anchor is None:
            raise DeviceCostGroundingError(
                f"movable owner anchor {owner!r} is absent"
            )
        _validate_anchor(owner_anchor, owner, need_axis=False, need_region=False)

    for predicate in (
        running_held + terminal_held + running_not_held + terminal_not_held
    ):
        anchor = anchor_defs[predicate.object_anchor]
        if (
            predicate.object_anchor != "subject"
            and anchor.attached_to != "subject"
        ):
            raise DeviceCostGroundingError(
                f"{predicate.type}.object_anchor must resolve to the "
                f"canonical subject, got {predicate.object_anchor!r} "
                f"attached_to {anchor.attached_to!r}"
            )
    # AtRest angular speed reads the OWNER's quaternion trajectory; resolve
    # each subject prefix to its transported pose source at compile time.
    at_rest_owners: dict[str, str] = {}
    for predicate in at_rest_predicates:
        center_name = f"{predicate.subject}_center"
        center_anchor = anchor_defs.get(center_name)
        if center_anchor is None:
            raise DeviceCostGroundingError(
                f"at_rest needs anchor {center_name!r}"
            )
        owner = (
            "subject"
            if predicate.subject == "subject"
            or center_anchor.attached_to == "subject"
            else center_anchor.attached_to
        )
        if not owner:
            raise DeviceCostGroundingError(
                "at_rest needs a tracked dynamic body, got static "
                f"{predicate.subject!r}"
            )
        at_rest_owners[predicate.subject] = str(owner)
    # `acquisition_only_terminal` / `requires_prefix_hold` used to select the
    # stages that got an exact-held feasibility barrier at the executable
    # prefix. That barrier is gone (see the end of `cost`), and with it the
    # only reason to distinguish an acquisition stage from a retention stage
    # here: ObjectHeld is now a finite cost for both, and stage advance is
    # decided by freshly measured terminals either way.

    movable_owners = tuple(sorted(required_movable_owners))
    parameter_names = tuple(sorted(
        set(anchor_defs)
        | set(movable_owners)
        | ({"subject"} if subject_is_needed else set())
    ))
    parameter_anchors = {name: anchors[name] for name in parameter_names}
    anchor_indices = {
        name: index for index, name in enumerate(parameter_names)
    }
    movable_indices = {
        name: index for index, name in enumerate(movable_owners)
    }
    grounded_names = tuple(sorted(anchor_defs))
    anchor_metadata = {
        name: (
            anchor_defs[name].attached_to,
            name in axis_names,
            name in region_names,
        )
        for name in grounded_names
    }
    stage_spec = {
        "running": [predicate.to_dict() for predicate in running],
        "terminal": [predicate.to_dict() for predicate in terminal],
    }
    if pick_v8_causal_profile is not None:
        # The v2 flag used to select a width-space jaw residual here.  Every
        # path now uses that residual, so the profile only has to validate and
        # be recorded; there is no longer a v1/v2 fork in the cost itself.
        from oap.twin.pick_v8_causal import validate_pick_v8_causal_profile

        validated_causal_profile = validate_pick_v8_causal_profile(
            pick_v8_causal_profile
        )
        stage_spec["pick_v8_causal_profile"] = validated_causal_profile
    if shaping_profile is not None:
        stage_spec["cost_shaping_profile"] = shaping_profile
        stage_spec["cost_shaping_running_multipliers"] = list(
            running_multipliers
        )
    spec = DeviceCostSpec(
        stage_json=json.dumps(
            stage_spec,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        anchor_layout=tuple(
            (
                name,
                parameter_anchors[name].attached_to,
                bool(parameter_anchors[name].dynamic),
                name in axis_names,
                name in region_names,
            )
            for name in parameter_names
        ),
        movable_owners=movable_owners,
    )

    def vectors(values: list[Any]) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape((-1, 3))

    running_scales, terminal_scales = planning_cost_scales(
        stage,
        anchors if cost_scale_anchors is None else cost_scale_anchors,
    )
    params = DeviceCostParams(
        anchor_points=vectors(
            [parameter_anchors[name].point for name in parameter_names]
        ),
        anchor_axes=vectors(
            [
                (
                    parameter_anchors[name].axis
                    if name in axis_names
                    else np.zeros(3, dtype=float)
                )
                for name in parameter_names
            ]
        ),
        anchor_regions=vectors(
            [
                (
                    parameter_anchors[name].region_half
                    if name in region_names
                    else np.zeros(3, dtype=float)
                )
                for name in parameter_names
            ]
        ),
        running_scales=np.asarray(
            running_scales,
            dtype=float,
        ),
        terminal_scales=np.asarray(
            terminal_scales,
            dtype=float,
        ),
        subject_start_pose7=start_subject,
        movable_start_pose7=np.stack(
            [start_movables[name] for name in movable_owners],
            axis=0,
        ) if movable_owners else np.empty((0, 7), dtype=float),
        disturb_movables=_v11_disturb_movables(stage, start_movables),
        table_top_z=np.asarray(table_top_z, dtype=float),
        sub_h=np.asarray(sub_h, dtype=float),
        dt=np.asarray(0.0 if dt is None else dt, dtype=float),
        sample_dt_s=(
            None
            if sample_dt_s is None
            else np.asarray(float(sample_dt_s), dtype=float)
        ),
    )

    term_descriptors_list: list[dict[str, Any]] = []
    zeroed_running_by_index = {
        int(item["running_term_index"]): item
        for item in (
            () if shaping_plan is None else shaping_plan.zeroed_running_terms
        )
    }
    for scope, predicates, original_predicates, multipliers in (
        ("running", running, original_running, running_multipliers),
        ("terminal", terminal, original_terminal, terminal_multipliers),
    ):
        for index, (predicate, original_predicate, multiplier) in enumerate(
            zip(predicates, original_predicates, multipliers, strict=True)
        ):
            descriptor = {
                "scope": scope,
                "term_index": index,
                **predicate.to_dict(),
            }
            if shaping_profile is not None:
                zeroed_detail = (
                    zeroed_running_by_index.get(index)
                    if scope == "running" and multiplier == 0.0
                    else None
                )
                identity = continuous_geometric_predicate_identity(
                    original_predicate
                )
                if identity is None and zeroed_detail is not None:
                    identity = zeroed_detail["identity"]
                descriptor["cost_shaping"] = {
                    "profile": shaping_profile,
                    "multiplier": float(multiplier),
                    "identity": identity,
                    "zeroed_running_twin": bool(
                        scope == "running" and multiplier == 0.0
                    ),
                }
            term_descriptors_list.append(descriptor)
    term_descriptors = tuple(term_descriptors_list)
    cached = _EVALUATOR_CACHE.get(spec)
    if cached is not None:
        cached_cost, cached_breakdown, cached_trajectory_breakdown = cached
        return DeviceCostEvaluator(
            spec=spec,
            function=cached_cost,
            params=params,
            breakdown_function=cached_breakdown,
            trajectory_breakdown_function=cached_trajectory_breakdown,
            term_descriptors=term_descriptors,
        )

    def cost_with_terms(
        subject_pose_traj7: Any,
        movable_pose_traj7: Mapping[str, Any] | None = None,
        gripper_center_traj3: Any | None = None,
        gripper_width_traj: Any | None = None,
        object_held_traj: Any | None = None,
        tool_target_contact_traj: Any | None = None,
        robot_target_contact_traj: Any | None = None,
        gripper_axis_traj3: Any | None = None,
        gripper_closing_axis_traj3: Any | None = None,
        object_not_held_fraction: Any | None = None,
        tool_target_contact_fraction: Any | None = None,
        robot_target_contact_any_step: Any | None = None,
        gripper_command_traj: Any | None = None,
        robot_subject_contact_missing_fraction: Any | None = None,
        robot_subject_contact_endpoint: Any | None = None,
        grounding: DeviceCostParams | None = None,
        *,
        _terminal_trajectory_only: bool = False,
    ) -> Any:
        """Evaluate all candidates using JAX operations only."""
        import jax
        import jax.numpy as jnp

        if grounding is None:
            raise DeviceCostGroundingError(
                "device cost grounding parameters are absent"
            )
        subject_pose_samples = jnp.asarray(subject_pose_traj7)
        if (
            subject_pose_samples.ndim != 3
            or subject_pose_samples.shape[-1] != 7
        ):
            raise DeviceCostGroundingError(
                "subject_pose_traj7 must have shape (N,T,7)"
            )
        if subject_pose_samples.shape[1] < 1:
            raise DeviceCostGroundingError(
                "subject_pose_traj7 must contain at least one state"
            )
        n_candidates, sample_horizon = subject_pose_samples.shape[:2]
        if dt is not None and not math.isclose(float(dt), 1.0 / sample_horizon, rel_tol=1e-7):
            raise DeviceCostGroundingError("Eq. (5) requires dt=1/T or an omitted dt")
        dtype = subject_pose_samples.dtype
        movable_poses = movable_pose_traj7 or {}

        def checked_trajectory(value: Any, label: str, width: int) -> Any:
            arr = jnp.asarray(value, dtype=dtype)
            if (
                arr.ndim != 3
                or arr.shape != (n_candidates, sample_horizon, width)
            ):
                raise DeviceCostGroundingError(
                    f"{label} must have shape (N,T,{width}) matching the subject"
                )
            return arr

        def checked_scalar_trajectory(value: Any, label: str) -> Any:
            arr = jnp.asarray(value, dtype=dtype)
            if (
                arr.ndim != 2
                or arr.shape != (n_candidates, sample_horizon)
            ):
                raise DeviceCostGroundingError(
                    f"{label} must have shape (N,T) matching the subject"
                )
            return arr

        subject_pose = subject_pose_samples
        evaluation_horizon = sample_horizon

        movable_arrays: dict[str, Any] = {}
        for owner in movable_owners:
            if owner not in movable_poses:
                raise DeviceCostGroundingError(
                    f"rollout trajectory for movable {owner!r} is absent"
                )
            movable_array = checked_trajectory(
                movable_poses[owner],
                f"movable_pose_traj7[{owner!r}]",
                7,
            )
            movable_arrays[owner] = movable_array

        gripper_center = None
        if needs_gripper_center:
            if gripper_center_traj3 is None:
                raise DeviceCostGroundingError(
                    "stage references gripper_center but rollout FK is absent"
                )
            gripper_center = checked_trajectory(
                gripper_center_traj3,
                "gripper_center_traj3",
                3,
            )

        gripper_axis = None
        if "gripper_center" in axis_names:
            if gripper_axis_traj3 is None:
                raise DeviceCostGroundingError(
                    "stage references axis(gripper_center) but rollout FK "
                    "site axis is absent"
                )
            gripper_axis = checked_trajectory(
                gripper_axis_traj3,
                "gripper_axis_traj3",
                3,
            )

        gripper_closing_axis = None
        if needs_gripper_closing_axis:
            if gripper_closing_axis_traj3 is None:
                raise DeviceCostGroundingError(
                    "stage references axis(gripper_closing_axis) but rollout "
                    "FK finger-separation axis is absent"
                )
            gripper_closing_axis = checked_trajectory(
                gripper_closing_axis_traj3,
                "gripper_closing_axis_traj3",
                3,
            )

        gripper_width = None
        if needs_width:
            if gripper_width_traj is None:
                raise DeviceCostGroundingError(
                    "stage contains GripperState but rollout width is absent"
                )
            gripper_width = checked_scalar_trajectory(
                gripper_width_traj,
                "gripper_width_traj",
            )

        gripper_command = None
        if needs_command:
            if gripper_command_traj is None:
                raise DeviceCostGroundingError(
                    "stage contains GripperCommand but rollout control target "
                    "is absent"
                )
            gripper_command = checked_scalar_trajectory(
                gripper_command_traj,
                "gripper_command_traj",
            )

        object_held = None
        if terminal_held or running_held or terminal_not_held or running_not_held:
            if object_held_traj is None:
                raise DeviceCostGroundingError(
                    "stage contains ObjectHeld/NotHeld but the sampled exact "
                    "held trajectory is absent"
                )
            object_held = jnp.asarray(
                object_held_traj,
                dtype=bool,
            )
            if (
                object_held.ndim != 2
                or object_held.shape != (n_candidates, sample_horizon)
            ):
                raise DeviceCostGroundingError(
                    "object_held_traj must have shape (N,T) matching the subject"
                )

        object_not_held = None
        if running_held or running_not_held:
            if object_not_held_fraction is None:
                raise DeviceCostGroundingError(
                    "stage contains running ObjectHeld/NotHeld but full-step "
                    "not-held fraction is absent"
                )
            object_not_held = jnp.asarray(
                object_not_held_fraction, dtype=dtype
            )
            if object_not_held.shape != (n_candidates,):
                raise DeviceCostGroundingError(
                    "object_not_held_fraction must have shape (N,)"
                )

        tool_target_contact = None
        sampled_subject_target_contact = None
        subject_target_contact_endpoint = None
        robot_target_contact = None
        if needs_subject_contact or needs_robot_contact:
            if (
                needs_subject_contact_fraction
                and tool_target_contact_fraction is None
            ):
                raise DeviceCostGroundingError(
                    "stage requires subject-target contact but the full-step "
                    "contact fraction is absent"
                )
            if (
                needs_subject_contact_endpoint
                and tool_target_contact_traj is None
            ):
                raise DeviceCostGroundingError(
                    "terminal Contact requires the exact sampled subject-target "
                    "contact trajectory at native H-1"
                )
            if needs_robot_contact and robot_target_contact_any_step is None:
                raise DeviceCostGroundingError(
                    "stage requires all-step robot-target "
                    "contact evidence is absent"
                )
            if needs_subject_contact_fraction:
                tool_target_contact = jnp.asarray(
                    tool_target_contact_fraction, dtype=dtype
                )
            if needs_subject_contact_endpoint:
                sampled_subject_target_contact = jnp.asarray(
                    tool_target_contact_traj,
                    dtype=bool,
                )
                if sampled_subject_target_contact.shape != (
                    n_candidates,
                    sample_horizon,
                ):
                    raise DeviceCostGroundingError(
                        "tool_target_contact_traj must have exact sampled shape "
                        "(N,T) matching the subject; T[-1] is native H-1"
                    )
                subject_target_contact_endpoint = (
                    sampled_subject_target_contact[:, sample_horizon - 1]
                )
            if needs_robot_contact:
                robot_target_contact = jnp.asarray(
                    robot_target_contact_any_step, dtype=bool
                )
            if (
                tool_target_contact is not None
                and tool_target_contact.shape != (n_candidates,)
            ):
                raise DeviceCostGroundingError(
                    "tool_target_contact_fraction must have shape (N,)"
                )
            if (
                robot_target_contact is not None
                and robot_target_contact.shape != (n_candidates,)
            ):
                raise DeviceCostGroundingError(
                    "robot_target_contact_any_step must have shape (N,)"
                )

        robot_subject_missing = None
        robot_subject_endpoint = None
        if needs_robot_subject_contact:
            if robot_subject_contact_missing_fraction is None:
                raise DeviceCostGroundingError(
                    "RobotSubjectContact requires the exact full-physics "
                    "missing-contact fraction"
                )
            if robot_subject_contact_endpoint is None:
                raise DeviceCostGroundingError(
                    "RobotSubjectContact requires exact full-horizon endpoint "
                    "contact evidence"
                )
            robot_subject_missing = jnp.asarray(
                robot_subject_contact_missing_fraction,
                dtype=dtype,
            )
            robot_subject_endpoint = jnp.asarray(
                robot_subject_contact_endpoint,
                dtype=bool,
            )
            if robot_subject_missing.shape != (n_candidates,):
                raise DeviceCostGroundingError(
                    "robot_subject_contact_missing_fraction must have shape "
                    "(N,)"
                )
            if robot_subject_endpoint.shape != (n_candidates,):
                raise DeviceCostGroundingError(
                    "robot_subject_contact_endpoint must have shape (N,)"
                )

        def constant(value: Any) -> Any:
            return jnp.asarray(value, dtype=dtype)

        def running_norm(value: Any, scale: Any) -> Any:
            """Eq. (3), including zero-band, binary, and vector components."""
            u = value / scale
            return u * (u / (jnp.hypot(1.0, u) + 1.0))

        anchor_points = constant(grounding.anchor_points)
        anchor_axes = constant(grounding.anchor_axes)
        anchor_regions = constant(grounding.anchor_regions)
        dynamic_subject_start = constant(grounding.subject_start_pose7)
        dynamic_movable_starts = constant(grounding.movable_start_pose7)

        def unit(value: Any, eps: float = 1e-12) -> Any:
            norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
            return value / jnp.where(norm < eps, jnp.ones_like(norm), norm)

        def quat_matrix(quat: Any) -> Any:
            norm = jnp.linalg.norm(quat, axis=-1, keepdims=True)
            identity_q = jnp.zeros_like(quat).at[..., 0].set(1.0)
            q = jnp.where(
                norm < 1e-12,
                identity_q,
                quat / jnp.where(norm < 1e-12, jnp.ones_like(norm), norm),
            )
            w, x, y, z = (q[..., i] for i in range(4))
            return jnp.stack(
                (
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ),
                axis=-1,
            ).reshape(quat.shape[:-1] + (3, 3))

        def rotation_delta(pose: Any, start_pose: np.ndarray) -> Any:
            current = quat_matrix(pose[..., 3:7])
            start = quat_matrix(constant(start_pose[3:7]))
            return jnp.einsum(
                "...ij,kj->...ik",
                current,
                start,
                precision=jax.lax.Precision.HIGHEST,
            )

        baseline = (
            constant(grounding.table_top_z)
            + 0.5 * constant(grounding.sub_h)
        )
        subject_center = subject_pose[..., :3]
        subject_center = subject_center.at[..., 2].set(
            jnp.maximum(subject_center[..., 2], baseline)
        )
        subject_rot = rotation_delta(subject_pose, dynamic_subject_start)

        points: dict[str, Any] = {}
        axes: dict[str, Any] = {}
        regions: dict[str, Any] = {}

        for name in grounded_names:
            owner, has_axis, has_region = anchor_metadata[name]
            parameter_index = anchor_indices[name]
            anchor_point = anchor_points[parameter_index]
            if name == "subject":
                transported_point = subject_center
                rot = subject_rot
            elif owner == "subject":
                rel = (
                    anchor_point
                    - anchor_points[anchor_indices["subject"]]
                )
                transported_point = subject_center + jnp.einsum(
                    "...ij,j->...i", subject_rot, rel
                )
                rot = subject_rot
            elif owner:
                pose = movable_arrays[owner]
                rel = anchor_point - anchor_points[anchor_indices[owner]]
                rot = rotation_delta(
                    pose,
                    dynamic_movable_starts[movable_indices[owner]],
                )
                transported_point = pose[..., :3] + jnp.einsum(
                    "...ij,j->...i", rot, rel
                )
            else:
                transported_point = jnp.broadcast_to(
                    anchor_point,
                    (n_candidates, evaluation_horizon, 3),
                )
                rot = None
            points[name] = transported_point
            if has_axis:
                transported_axis = anchor_axes[parameter_index]
                if rot is not None:
                    transported_axis = unit(
                        jnp.einsum("...ij,j->...i", rot, transported_axis)
                    )
                else:
                    transported_axis = jnp.broadcast_to(
                        transported_axis,
                        (n_candidates, evaluation_horizon, 3),
                    )
                axes[name] = transported_axis
            if has_region:
                regions[name] = anchor_regions[parameter_index]

        if gripper_center is not None:
            points["gripper_center"] = gripper_center
        if gripper_axis is not None:
            axes["gripper_center"] = unit(gripper_axis)
        if gripper_closing_axis is not None:
            axes["gripper_closing_axis"] = unit(gripper_closing_axis)

        def point(name: str) -> Any:
            try:
                return points[name]
            except KeyError as exc:  # Static trace-time error, never fallback.
                raise DeviceCostGroundingError(
                    f"point anchor {name!r} is not grounded"
                ) from exc

        def axis(name: str) -> Any:
            try:
                return axes[name]
            except KeyError as exc:
                raise DeviceCostGroundingError(
                    f"axis anchor {name!r} is not grounded"
                ) from exc

        def angle_between(a: Any, b: Any) -> Any:
            cosine = jnp.sum(unit(a) * unit(b), axis=-1)
            return jnp.arccos(jnp.clip(cosine, -1.0, 1.0))

        def residual(predicate: Predicate) -> Any:
            if isinstance(predicate, PointPointDistance):
                distance = jnp.linalg.norm(point(predicate.a) - point(predicate.b), axis=-1)
                return jnp.maximum(
                    0.0, jnp.abs(distance - predicate.dist) - predicate.tol
                )
            if isinstance(predicate, CaptureDistance):
                return jnp.linalg.norm(
                    point("gripper_center") - point(predicate.subject),
                    axis=-1,
                )
            if isinstance(predicate, RelativePosition):
                error = (
                    point(predicate.a)
                    - point(predicate.b)
                    - constant(predicate.offset)
                )
                return jnp.maximum(
                    0.0, jnp.linalg.norm(error, axis=-1) - predicate.tol
                )
            if isinstance(predicate, MinDistance):
                distance = jnp.linalg.norm(point(predicate.a) - point(predicate.b), axis=-1)
                return jnp.maximum(0.0, predicate.clearance - distance)
            if isinstance(predicate, AboveBy):
                direction = unit(constant(predicate.axis))
                signed = jnp.sum(
                    (point(predicate.a) - point(predicate.b)) * direction,
                    axis=-1,
                )
                return jnp.maximum(
                    0.0, jnp.abs(signed - predicate.dist) - predicate.tol
                )
            if isinstance(predicate, OnPlane):
                normal = unit(constant(predicate.axis))
                signed = jnp.sum(
                    (point(predicate.a) - point(predicate.b)) * normal,
                    axis=-1,
                )
                return jnp.maximum(0.0, jnp.abs(signed) - predicate.tol)
            if isinstance(predicate, PointOnLine):
                delta = point(predicate.a) - point(predicate.b)
                direction = unit(axis(predicate.dir_anchor))
                perpendicular = delta - jnp.sum(
                    delta * direction, axis=-1, keepdims=True
                ) * direction
                return jnp.maximum(
                    0.0,
                    jnp.linalg.norm(perpendicular, axis=-1) - predicate.tol,
                )
            if isinstance(predicate, AxisParallel):
                second = -axis(predicate.b) if predicate.antiparallel else axis(predicate.b)
                angle = angle_between(axis(predicate.a), second)
                return jnp.maximum(
                    0.0, angle - math.radians(predicate.tol_deg)
                )
            if isinstance(predicate, AxisAngle):
                angle = angle_between(axis(predicate.a), axis(predicate.b))
                return jnp.maximum(
                    0.0,
                    jnp.abs(angle - math.radians(predicate.theta_deg))
                    - math.radians(predicate.tol_deg),
                )
            if isinstance(predicate, RotationProgress):
                from .rotation_progress import rotation_progress_radians
                current = jnp.stack([axis(n) for n in predicate.current_frame_names()], axis=-1)
                initial = jnp.stack([axis(n) for n in predicate.initial_frame_names()], axis=-1)
                progress = rotation_progress_radians(initial, current, constant(predicate.axis), xp=jnp)
                return jnp.maximum(0.0, math.radians(predicate.min_angle_deg) - progress)
            if isinstance(predicate, RelativeOrientation):
                current = jnp.stack(
                    [axis(name) for name in predicate.current_frame_names()],
                    axis=-1,
                )
                initial = jnp.stack(
                    [axis(name) for name in predicate.initial_frame_names()],
                    axis=-1,
                )
                target_axis = unit(constant(predicate.target_axis))
                ax, ay, az = (
                    target_axis[0],
                    target_axis[1],
                    target_axis[2],
                )
                skew = jnp.stack(
                    [
                        jnp.stack([0.0 * ax, -az, ay]),
                        jnp.stack([az, 0.0 * ax, -ax]),
                        jnp.stack([-ay, ax, 0.0 * ax]),
                    ],
                    axis=0,
                )
                target_angle = math.radians(predicate.target_angle_deg)
                delta = (
                    jnp.eye(3, dtype=dtype)
                    + math.sin(target_angle) * skew
                    + (1.0 - math.cos(target_angle)) * (skew @ skew)
                )
                desired = jnp.einsum("...ij,jk->...ik", initial, delta)
                error = jnp.einsum("...ji,...jk->...ik", desired, current)
                trace = jnp.trace(error, axis1=-2, axis2=-1)
                geodesic = jnp.arccos(
                    jnp.clip((trace - 1.0) * 0.5, -1.0, 1.0)
                )

                def proper(rotation: Any) -> Any:
                    gram = jnp.einsum(
                        "...ji,...jk->...ik", rotation, rotation
                    )
                    gram_error = jnp.max(
                        jnp.abs(gram - jnp.eye(3, dtype=dtype)),
                        axis=(-2, -1),
                    )
                    determinant = jnp.linalg.det(rotation)
                    return (gram_error <= 1e-4) & (
                        determinant >= 1.0 - 1e-4
                    )

                value = jnp.maximum(
                    0.0,
                    geodesic - math.radians(predicate.tol_deg),
                )
                return jnp.where(
                    proper(initial) & proper(current),
                    value,
                    math.pi,
                )
            if isinstance(predicate, AngleAboutAxis):
                hinge = unit(axis(predicate.b))
                current = axis(predicate.a)
                reference = axis(predicate.ref)
                current_projected = current - hinge * jnp.sum(
                    current * hinge, axis=-1, keepdims=True
                )
                reference_projected = reference - hinge * jnp.sum(
                    reference * hinge, axis=-1, keepdims=True
                )
                current_norm = jnp.linalg.norm(current_projected, axis=-1)
                reference_norm = jnp.linalg.norm(reference_projected, axis=-1)
                current_unit = unit(current_projected)
                reference_unit = unit(reference_projected)
                sine = jnp.sum(
                    jnp.cross(reference_unit, current_unit) * hinge,
                    axis=-1,
                )
                cosine = jnp.sum(reference_unit * current_unit, axis=-1)
                angle = jnp.arctan2(sine, cosine)
                target = math.radians(predicate.theta_deg)
                delta = jnp.arctan2(
                    jnp.sin(angle - target),
                    jnp.cos(angle - target),
                )
                value = jnp.maximum(
                    0.0,
                    jnp.abs(delta)
                    - math.radians(predicate.tol_deg),
                )
                invalid = (current_norm < 1e-9) | (reference_norm < 1e-9)
                return jnp.where(invalid, math.pi, value)
            if isinstance(predicate, InRegion):
                half = regions[predicate.region]
                outside = jnp.maximum(
                    jnp.abs(point(predicate.a) - point(predicate.region)) - half,
                    0.0,
                )
                return jnp.max(outside / half, axis=-1)
            if isinstance(predicate, SignedAxisGap):
                direction = unit(constant(predicate.axis))
                gap = predicate.sign * jnp.sum(
                    (point(predicate.b) - point(predicate.a)) * direction,
                    axis=-1,
                )
                return jnp.maximum(0.0, predicate.margin - gap)
            if isinstance(predicate, ObjectHeld):
                assert object_held is not None
                return 1.0 - object_held.astype(dtype)
            if isinstance(predicate, GripperState):
                assert gripper_width is not None
                target = (
                    float(predicate.width_m)
                    if predicate.width_m is not None
                    else (0.085 if predicate.state == "open" else 0.0)
                )
                if predicate.width_m is not None:
                    return jnp.maximum(
                        0.0,
                        jnp.abs(gripper_width - target) - predicate.tol,
                    )
                if predicate.state == "open":
                    return jnp.maximum(
                        0.0, target - predicate.tol - gripper_width
                    )
                return jnp.maximum(
                    0.0, gripper_width - (target + predicate.tol)
                )
            if isinstance(predicate, GripperCommand):
                assert gripper_command is not None
                # Map the latent to the virtual width before differencing, on
                # every path.  planning_cost_scales gives this family
                # _GRIPPER_DOMAIN_M = 0.085 m, so the residual has to be a
                # length.  The signed-effort latent is dimensionless and spans
                # 2.0; dividing that span by 0.085 m inflated the term 11.76x,
                # and 138x once squared.  Easing an 80 N squeeze to 72 N then
                # cost 1.384, while the entire price of dropping the object is
                # running ObjectHeld 1.0 + terminal ObjectHeld 1.0 = 2.0 -- so
                # the jaw term dominated retention in the ranked pool.
                #
                # This is also the operation order that reproduces successful
                # v8 in float32, which the causal profiles require exactly;
                # keeping a single branch is what makes that exactness
                # unconditional.  It is the same function as a latent-space
                # scale of 2.0 -- test_v2_gripper_scale_repairs_legacy_width_
                # algebra_without_default_drift asserts that equality to
                # 1e-15 -- so the divisor stays 0.085 m and no scale moves.
                command = (
                    0.5
                    * (1.0 - jnp.clip(gripper_command, -1.0, 1.0))
                    * 0.085
                )
                target = predicate._target()
                return jnp.maximum(
                    0.0,
                    jnp.abs(command - target)
                    - predicate.tol,
                )
            if isinstance(predicate, InteractionAlign):
                distance = jnp.linalg.norm(
                    point(predicate.a) - point(predicate.b), axis=-1
                )
                distance_residual = jnp.maximum(
                    0.0, jnp.abs(distance - predicate.dist) - predicate.tol
                )
                axis_a = axis(
                    predicate.a if predicate.axis_a == "self" else predicate.axis_a
                )
                axis_b = axis(
                    predicate.b if predicate.axis_b == "self" else predicate.axis_b
                )
                if predicate.antiparallel:
                    axis_b = -axis_b
                angle = angle_between(axis_a, axis_b)
                angle_residual = jnp.maximum(
                    0.0,
                    jnp.abs(angle - math.radians(predicate.theta_deg))
                    - math.radians(predicate.tol_deg),
                )
                return (
                    predicate.w_d * distance_residual
                    + predicate.w_theta * angle_residual
                )
            if isinstance(predicate, AlongAxisGap):
                delta = point(predicate.a) - point(predicate.c)
                direction = axis(predicate.b)
                along = jnp.sum(delta * direction, axis=-1)
                perpendicular = jnp.linalg.norm(
                    jnp.cross(delta, direction), axis=-1
                )
                return (
                    jnp.maximum(
                        0.0,
                        jnp.abs(along - predicate.dist) - predicate.tol,
                    )
                    + predicate.w_perp * perpendicular
                )
            if isinstance(predicate, SupportedOn):
                # Same math as the host residual, on transported anchors: the
                # subject's four MATERIAL bottom corners (they ride the
                # subject pose, so tilt lifts one) against the support's
                # top-face frame. Metres: max-corner footprint overhang plus
                # max-corner |top-plane gap| hinge.
                top = point(f"{predicate.support}_top")
                x_axis = unit(axis(f"{predicate.support}_axis"))
                z_axis = unit(axis(f"{predicate.support}_normal"))
                y_axis = unit(jnp.cross(z_axis, x_axis))
                half = regions[predicate.support]
                corner_overhangs = []
                corner_gaps = []
                for corner_index in range(4):
                    rel = point(
                        f"{predicate.subject}_bottom_corner_{corner_index}"
                    ) - top
                    u = (
                        jnp.abs(jnp.sum(rel * x_axis, axis=-1))
                        - half[0]
                        - predicate.tol
                    )
                    v = (
                        jnp.abs(jnp.sum(rel * y_axis, axis=-1))
                        - half[1]
                        - predicate.tol
                    )
                    w = (
                        jnp.abs(jnp.sum(rel * z_axis, axis=-1))
                        - predicate.tol
                    )
                    corner_overhangs.append(jnp.maximum(u, v))
                    corner_gaps.append(w)
                overhang = jnp.max(jnp.stack(corner_overhangs), axis=0)
                plane_gap = jnp.max(jnp.stack(corner_gaps), axis=0)
                return (
                    jnp.maximum(0.0, overhang)
                    + jnp.maximum(0.0, plane_gap)
                )
            if isinstance(predicate, AtRest):
                # Speeds from the pose sample grid: backward differences of
                # the transported centre point and of the owner's quaternion,
                # divided by the physical sample spacing. The first sample
                # repeats the first difference so running integration keeps
                # the (N, T) shape.
                if evaluation_horizon < 2:
                    raise DeviceCostGroundingError(
                        "at_rest needs at least two pose samples"
                    )
                sample_dt = constant(grounding.sample_dt_s)
                center = point(f"{predicate.subject}_center")
                linear_delta = center[:, 1:] - center[:, :-1]
                linear_speed = (
                    jnp.linalg.norm(linear_delta, axis=-1) / sample_dt
                )
                owner = at_rest_owners[predicate.subject]
                quat = (
                    subject_pose[..., 3:7]
                    if owner == "subject"
                    else movable_arrays[owner][..., 3:7]
                )
                quat = unit(quat)
                # Chord form: 2*asin(||q1 -/+ q2||/2) is well-conditioned for
                # the small per-sample rotations rest-checking cares about;
                # arccos(|<q1,q2>|) loses ~5e-4 rad to float32 near identity,
                # the same order as ang_tol over one sample.
                chord = jnp.minimum(
                    jnp.linalg.norm(quat[:, 1:] - quat[:, :-1], axis=-1),
                    jnp.linalg.norm(quat[:, 1:] + quat[:, :-1], axis=-1),
                )
                # ||q1 - q2|| = 2 sin(delta_theta / 4) for unit quaternions.
                angular_delta = 4.0 * jnp.arcsin(
                    jnp.clip(0.5 * chord, 0.0, 1.0)
                )
                angular_speed = angular_delta / sample_dt
                per_step = (
                    jnp.maximum(
                        0.0, linear_speed / predicate.lin_tol - 1.0
                    )
                    + jnp.maximum(
                        0.0, angular_speed / predicate.ang_tol - 1.0
                    )
                )
                return jnp.concatenate(
                    [per_step[:, :1], per_step], axis=1
                )
            if isinstance(predicate, (Contact, NoContact)):
                # Accumulated explicitly below from exact all-step contact
                # reductions; sparse pose anchors cannot reconstruct contact.
                return jnp.zeros(
                    (n_candidates, evaluation_horizon),
                    dtype=dtype,
                )
            if isinstance(predicate, RobotSubjectContact):
                # Exact native-step and endpoint evidence is accumulated
                # outside the bounded pose-sample interpreter below.
                return jnp.zeros(
                    (n_candidates, evaluation_horizon),
                    dtype=dtype,
                )
            if isinstance(predicate, ToolMediation):
                # Accumulated explicitly below from exact all-step contact
                # reductions. Sparse endpoint anchors cannot reconstruct it.
                return jnp.zeros(
                    (n_candidates, evaluation_horizon),
                    dtype=dtype,
                )
            if isinstance(predicate, NonGeometric):
                return jnp.zeros(
                    (n_candidates, evaluation_horizon),
                    dtype=dtype,
                )
            raise TypeError(  # pragma: no cover - compile pass rejects first.
                "unsupported device predicate; refusing CPU fallback"
            )

        total = jnp.zeros((n_candidates,), dtype=dtype)
        # Per-term contributions are stacked for diagnostics only; ``total``
        # keeps its original piecewise accumulation order so selection stays
        # bit-identical to the pre-breakdown scorer.
        term_contributions: list[Any] = []
        terminal_residual_trajectories: list[Any] = []
        running_scales = constant(grounding.running_scales)
        terminal_scales = constant(grounding.terminal_scales)
        full_endpoint_index = sample_horizon - 1
        for index, predicate in enumerate(running):
            if isinstance(predicate, ObjectHeld):
                assert object_not_held is not None
                # Exact contact loss is a finite running residual over the full
                # horizon. It ranks retention behavior but is not generic
                # physical infeasibility; dense gripper-to-object geometry is
                # expressed separately by the typed program.
                scale = running_scales[index]
                contribution = object_not_held * running_norm(1.0, scale)
                term_contributions.append(contribution)
                total = total + running_multipliers[index] * contribution
                continue
            if isinstance(predicate, NotHeld):
                assert object_not_held is not None
                # The exact complement: the fraction of full-physics steps
                # WITH a two-sided hold ranks how much of the horizon still
                # pinches the object.
                scale = running_scales[index]
                contribution = (1.0 - object_not_held) * running_norm(1.0, scale)
                term_contributions.append(contribution)
                total = total + running_multipliers[index] * contribution
                continue
            if isinstance(predicate, RobotSubjectContact):
                assert robot_subject_missing is not None
                scale = running_scales[index]
                contribution = robot_subject_missing * running_norm(1.0, scale)
                term_contributions.append(contribution)
                total = total + running_multipliers[index] * contribution
                continue
            scale = running_scales[index]
            if isinstance(predicate, (Contact, NoContact)):
                if "robot" in (predicate.a, predicate.b):
                    assert robot_target_contact is not None
                    present = robot_target_contact.astype(dtype)
                else:
                    assert tool_target_contact is not None
                    present = (tool_target_contact > 0.0).astype(dtype)
                relation_residual = (
                    1.0 - present
                    if isinstance(predicate, Contact)
                    else present
                )
                contribution = running_norm(relation_residual, scale)
                term_contributions.append(contribution)
                total = total + running_multipliers[index] * contribution
                continue
            if isinstance(predicate, ToolMediation):
                assert robot_target_contact is not None
                mediation_pieces = []
                if predicate.require_tool_target_contact:
                    assert tool_target_contact is not None
                    if (
                        shaping_profile
                        == TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2
                    ):
                        # Conventional running contact cost: reward the exact
                        # fraction of full-physics horizon steps that maintain
                        # tool-target contact.  This remains a finite ranking
                        # term; terminal success still requires fresh endpoint
                        # contact and direct robot-target contact is penalized
                        # independently below.
                        contact_missing = _tool_mediation_positive_contact_residual(
                            tool_target_contact,
                            shaping_profile=shaping_profile,
                        )
                    else:
                        # Legacy compatibility: one exact tool-target contact
                        # step is sufficient for the existence residual.
                        contact_missing = _tool_mediation_positive_contact_residual(
                            tool_target_contact,
                            shaping_profile=shaping_profile,
                        )
                    piece = running_norm(contact_missing, scale)
                    mediation_pieces.append(piece)
                piece = running_norm(robot_target_contact.astype(dtype), scale)
                mediation_pieces.append(piece)
                contribution = mediation_pieces[0]
                for piece in mediation_pieces[1:]:
                    contribution = contribution + piece
                term_contributions.append(contribution)
                total = total + running_multipliers[index] * contribution
                continue
            value = residual(_zero_band(predicate))
            contribution = jnp.mean(running_norm(value, scale), axis=1)
            term_contributions.append(contribution)
            total = total + running_multipliers[index] * contribution
        for index, predicate in enumerate(terminal):
            if isinstance(predicate, ObjectHeld):
                assert object_held is not None
                # Terminal predicates are evaluated only at the predicted
                # full-horizon endpoint x_H.  The executable prefix x_E is a
                # state-transfer boundary, never a second planning objective.
                not_held_trajectory = (~object_held).astype(dtype)
                terminal_residual_trajectories.append(not_held_trajectory)
                endpoint_not_held = not_held_trajectory[:, full_endpoint_index]
                contribution = (
                    endpoint_not_held
                    / terminal_scales[index]
                ) ** 2
                term_contributions.append(contribution)
                total = total + terminal_multipliers[index] * contribution
                continue
            if isinstance(predicate, NotHeld):
                assert object_held is not None
                held_trajectory = object_held.astype(dtype)
                terminal_residual_trajectories.append(held_trajectory)
                endpoint_held = held_trajectory[:, full_endpoint_index]
                contribution = (
                    endpoint_held
                    / terminal_scales[index]
                ) ** 2
                term_contributions.append(contribution)
                total = total + terminal_multipliers[index] * contribution
                continue
            if isinstance(predicate, RobotSubjectContact):
                assert robot_subject_endpoint is not None
                if _terminal_trajectory_only:
                    raise DeviceCostGroundingError(
                        "terminal RobotSubjectContact trajectory diagnostics "
                        "require an exact sampled trajectory channel"
                    )
                endpoint_missing = (~robot_subject_endpoint).astype(dtype)
                contribution = (
                    endpoint_missing / terminal_scales[index]
                ) ** 2
                term_contributions.append(contribution)
                total = total + terminal_multipliers[index] * contribution
                continue
            if isinstance(predicate, Contact):
                assert subject_target_contact_endpoint is not None
                assert sampled_subject_target_contact is not None
                contact_missing_trajectory = (
                    ~sampled_subject_target_contact
                ).astype(dtype)
                terminal_residual_trajectories.append(
                    contact_missing_trajectory
                )
                endpoint_missing = contact_missing_trajectory[
                    :, full_endpoint_index
                ]
                contribution = (
                    endpoint_missing / terminal_scales[index]
                ) ** 2
                term_contributions.append(contribution)
                total = total + terminal_multipliers[index] * contribution
                continue
            value = residual(predicate)
            terminal_residual_trajectories.append(value)
            endpoint_value = value[:, full_endpoint_index]
            contribution = (endpoint_value / terminal_scales[index]) ** 2
            term_contributions.append(contribution)
            total = total + terminal_multipliers[index] * contribution
        per_term = (
            jnp.stack(term_contributions, axis=0)
            if term_contributions
            else jnp.zeros((0, n_candidates), dtype=dtype)
        )
        if _terminal_trajectory_only:
            return (
                jnp.stack(terminal_residual_trajectories, axis=0)
                if terminal_residual_trajectories
                else jnp.zeros(
                    (0, n_candidates, sample_horizon),
                    dtype=dtype,
                )
            )
        return total, per_term

    def cost(*args: Any, **kwargs: Any) -> Any:
        return cost_with_terms(*args, **kwargs)[0]

    def trajectory_breakdown(*args: Any, **kwargs: Any) -> Any:
        return cost_with_terms(
            *args,
            **kwargs,
            _terminal_trajectory_only=True,
        )

    _EVALUATOR_CACHE[spec] = (
        cost,
        cost_with_terms,
        trajectory_breakdown,
    )
    return DeviceCostEvaluator(
        spec=spec,
        function=cost,
        params=params,
        breakdown_function=cost_with_terms,
        trajectory_breakdown_function=trajectory_breakdown,
        term_descriptors=term_descriptors,
    )


def device_trajectory_cost(
    stage: Stage,
    anchors: AnchorSet,
    *,
    subject_start_pose7: Any,
    subject_pose_traj7: Any,
    movable_start_pose7: Mapping[str, Any] | None = None,
    movable_pose_traj7: Mapping[str, Any] | None = None,
    gripper_center_traj3: Any | None = None,
    gripper_width_traj: Any | None = None,
    object_held_traj: Any | None = None,
    tool_target_contact_traj: Any | None = None,
    robot_target_contact_traj: Any | None = None,
    gripper_axis_traj3: Any | None = None,
    gripper_closing_axis_traj3: Any | None = None,
    object_not_held_fraction: Any | None = None,
    tool_target_contact_fraction: Any | None = None,
    robot_target_contact_any_step: Any | None = None,
    gripper_command_traj: Any | None = None,
    robot_subject_contact_missing_fraction: Any | None = None,
    robot_subject_contact_endpoint: Any | None = None,
    table_top_z: float,
    sub_h: float,
    dt: float | None = None,
    cost_shaping_profile: str | None = None,
) -> Any:
    """One-shot convenience wrapper around :func:`make_device_trajectory_cost`.

    Repeated production calls should build once and fuse the returned function
    into the rollout ``jit``. This wrapper still performs no per-candidate host
    work and never falls back to the CPU interpreter.
    """
    evaluator = make_device_trajectory_cost(
        stage,
        anchors,
        subject_start_pose7=subject_start_pose7,
        movable_start_pose7=movable_start_pose7,
        table_top_z=table_top_z,
        sub_h=sub_h,
        dt=dt,
        cost_shaping_profile=cost_shaping_profile,
    )
    return evaluator(
        subject_pose_traj7,
        movable_pose_traj7,
        gripper_center_traj3,
        gripper_width_traj,
        object_held_traj,
        tool_target_contact_traj,
        robot_target_contact_traj,
        gripper_axis_traj3,
        gripper_closing_axis_traj3,
        object_not_held_fraction,
        tool_target_contact_fraction,
        robot_target_contact_any_step,
        gripper_command_traj,
        robot_subject_contact_missing_fraction,
        robot_subject_contact_endpoint,
    )
