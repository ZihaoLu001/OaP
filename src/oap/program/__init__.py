"""OaP TaskProgram core (the crown jewel).

ONE mechanism for arbitrary / unknown tasks, ZERO per-task code:
  ground (anchors) -> a VLM SYNTHESIZES a TaskProgram over a fixed
  predicate basis -> ONE GPU sampling MPC ranks joint-control rollouts with its
  running and terminal costs -> physics gate -> the SAME program verifies the
  terminal predicates closed-loop on the real re-observation
  -> a UNIFORM CANNOT_VERIFY router refuses what is not a grounded
  geometric/kinematic relation.

Role in the two-stage pipeline: this package IS the planning stage's task
semantics; ``oap-reconstruct`` builds the scene it is grounded on.
Predicate interpretation is unit-testable independently, while production
trajectory search is the GPU rollout in :mod:`oap.loop.sampling`.
Frontier-VLM calls plug in behind the synthesis-backend interface. Every
program evaluation at the four call sites
(trajectory_cost, physics_gate, termination_gate, success_verification) routes
through :func:`oap.program.hashing.log_call_site`.
"""
from __future__ import annotations

from .anchors import Anchor, AnchorSet
from .bindings import (
    ProgramBindingError, ProgramBindings, bind_program_to_scene,
)
from .hashing import (
    REQUIRED_CALL_SITES, CallSite, CallSiteRecord, call_site_violations,
    log_call_site, program_sha256,
)
from .interpreter import (
    running_cost,
    terminal_cost,
    terminal_satisfied,
    trajectory_cost,
)
from .predicates import (
    AboveBy, AlongAxisGap, AngleAboutAxis, AtRest, AxisAngle, AxisParallel,
    CaptureDistance, Contact, GripperCommand, GripperState, InRegion, InteractionAlign,
    MinDistance, NoContact, NotHeld, ObjectHeld, NonGeometric, OnPlane,
    PointOnLine,
    PointPointDistance,
    Predicate, RelativeOrientation, RotationProgress, RelativePosition, RobotSubjectContact,
    SignedAxisGap,
    SupportedOn, TemporalHold,
    ToolMediation,
)
from .program import ConstraintProgram, Stage, TaskProgram
from .synthesis import (
    available_backends, build_prompt, parse_program,
    register_backend, synthesize, validate_program,
)
from .verdict import (
    EpisodeVerdict,
    StageEvaluation,
    VerificationOutcome,
    evaluate_stage,
    groundability_failures,
    stage_terminal_now,
    verify,
)

__all__ = [
    "Anchor", "AnchorSet",
    "ProgramBindingError", "ProgramBindings", "bind_program_to_scene",
    "Predicate", "PointPointDistance", "CaptureDistance", "RelativePosition", "MinDistance", "AboveBy", "OnPlane", "PointOnLine",
    "AxisParallel", "AxisAngle", "RelativeOrientation", "RotationProgress", "AngleAboutAxis",
    "InRegion", "SignedAxisGap",
    "GripperCommand", "GripperState", "ObjectHeld", "TemporalHold",
    "Contact", "NoContact", "RobotSubjectContact",
    "InteractionAlign",
    "AlongAxisGap", "ToolMediation", "NonGeometric",
    "SupportedOn", "AtRest", "NotHeld",
    "TaskProgram", "ConstraintProgram", "Stage",
    "running_cost", "terminal_cost", "trajectory_cost",
    "terminal_satisfied",
    "synthesize", "available_backends", "register_backend", "build_prompt",
    "parse_program", "validate_program",
    "VerificationOutcome", "EpisodeVerdict", "StageEvaluation", "evaluate_stage",
    "stage_terminal_now", "verify", "groundability_failures",
    "program_sha256", "CallSite", "CallSiteRecord", "log_call_site",
    "call_site_violations", "REQUIRED_CALL_SITES",
]
