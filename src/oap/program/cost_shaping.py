"""Categorical, task-agnostic TaskProgram cost-shaping profiles.

Cost shaping changes only how otherwise valid candidates are ranked.  It does
not edit a :class:`TaskProgram`, change a terminal tolerance, or participate in
physics/action validity.  The general non-baseline profile removes a duplicated
continuous geometric relation from the running integral when the same typed
relation and target operands already occur in that stage's terminal objective.
A separate, exact-contract-bound Pick-v9 profile suppresses only the duplicated
S0 relative-position path integral while retaining continuous orientation and
open-jaw guidance, and keeps only direct S4 release guidance (open jaw, not
held, and supported on).  Every terminal contribution remains score-active and
remains a measured stage success condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .predicates import Predicate, TemporalHold
from .program import Stage


BASELINE_COST_SHAPING = "baseline"
TYPED_RESIDUAL_LIBRARY_V1 = "typed_residual_library_v1"
TYPED_RESIDUAL_LIBRARY_V2 = "typed_residual_library_v2"
TYPED_RESIDUAL_LIBRARY_V3 = "typed_residual_library_v3"
TYPED_RESIDUAL_LIBRARY_V4 = "typed_residual_library_v4"
TWIN_TERMINAL_FOCUS_V1 = "twin_terminal_focus_v1"
PICK_V9_S4_DIRECT_RELEASE_V1 = "pick_v9_s4_direct_release_v1"
PICK_V8_COMBINED_DIRECT_RELEASE_V3 = "pick_v8_combined_direct_release_v3"
PICK_V8_COMBINED_RELEASE_LATERAL_V5 = "pick_v8_combined_release_lateral_v5"
PICK_V8_AT_REST_TERMINAL_V6 = "pick_v8_at_rest_terminal_v6"
CUP_V7_S1_RUNNING_POSITION_FOCUS_V1 = (
    "cup_v7_s1_running_position_focus_v1"
)
CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1 = (
    "cup_simple_tilt_terminal_held_60_v1"
)
FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1 = (
    "flip_v1_s0_running_position_focus_v1"
)
TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1 = (
    "toolpush_8cm_mediation_weight_4_v1"
)
TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2 = (
    "toolpush_8cm_mediation_duration_weight_4_v2"
)
COST_SHAPING_PROFILES = (
    BASELINE_COST_SHAPING,
    TYPED_RESIDUAL_LIBRARY_V1,
    TYPED_RESIDUAL_LIBRARY_V2,
    TYPED_RESIDUAL_LIBRARY_V3,
    TYPED_RESIDUAL_LIBRARY_V4,
    TWIN_TERMINAL_FOCUS_V1,
    PICK_V9_S4_DIRECT_RELEASE_V1,
    PICK_V8_COMBINED_DIRECT_RELEASE_V3,
    PICK_V8_COMBINED_RELEASE_LATERAL_V5,
    PICK_V8_AT_REST_TERMINAL_V6,
    CUP_V7_S1_RUNNING_POSITION_FOCUS_V1,
    CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1,
    FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1,
    TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1,
    TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
)
_COST_SHAPING_PROFILE_ALIASES = {
    "pick_v8_combined_release_lateral_uniform30_v6": (
        PICK_V8_COMBINED_RELEASE_LATERAL_V5
    ),
    "pick_v8_combined_release_lateral_uniform20_v6": (
        PICK_V8_COMBINED_RELEASE_LATERAL_V5
    ),
}

# These predicates expose a continuous geometric residual.  Event/state and
# dynamics semantics are intentionally absent even when a program repeats them
# in both scopes: contact, gripper command/state, held/not-held, ToolMediation,
# AtRest, TemporalHold frame counts, and NonGeometric are never suppressed.
_CONTINUOUS_GEOMETRIC_TYPES = frozenset({
    "point_point_distance",
    "capture_distance",
    "relative_position",
    "min_distance",
    "above_by",
    "on_plane",
    "point_on_line",
    "axis_parallel",
    "axis_angle",
    "relative_orientation",
    "angle_about_axis",
    "in_region",
    "signed_axis_gap",
    "interaction_align",
    "along_axis_gap",
    "supported_on",
})

# A tolerance defines the zero band around a relation; it does not change the
# relation's target.  Running and terminal tolerances are commonly different
# on purpose.  All target quantities (offset, dist, theta, target axis/angle,
# clearance, margin, etc.) remain identity-bearing.
_TOLERANCE_FIELDS = frozenset({"tol", "tol_deg", "lin_tol", "ang_tol"})

# Bind the exceptional recovery profile to the complete reviewed Pick-v9 S0
# and S4 contracts, not merely to stage names or predicate occurrences.  A task edit,
# reordered predicate, changed tolerance, or lookalike stage therefore makes
# the profile an auditable no-op instead of silently broadening its scope.
_PICK_V9_S0_NAME = "approach_eraser_open"
_PICK_V9_S0_RUNNING = (
    {
        "a": "gripper_center",
        "b": "eraser_center",
        "offset": [-0.024, 0.001, 0.004],
        "tol": 0.012,
        "type": "relative_position",
    },
    {
        "a": "gripper_center",
        "antiparallel": True,
        "b": "world_up",
        "tol_deg": 10.0,
        "type": "axis_parallel",
    },
    {
        "a": "gripper_closing_axis",
        "b": "eraser_axis",
        "theta_deg": 90.0,
        "tol_deg": 10.0,
        "type": "axis_angle",
    },
    {"state": "open", "tol": 0.0, "type": "gripper_command", "width_m": None},
)
_PICK_V9_S0_TERMINAL = _PICK_V9_S0_RUNNING[:3]
_PICK_V9_S4_NAME = "release_eraser"
_PICK_V9_S4_RUNNING = (
    {"state": "open", "tol": 0.0, "type": "gripper_command", "width_m": None},
    {"object_anchor": "eraser", "type": "not_held"},
    {
        "a": "gripper_center",
        "b": "eraser_center",
        "clearance": 0.06,
        "type": "min_distance",
    },
    {
        "subject": "eraser",
        "support": "spacemouse_box",
        "tol": 0.005,
        "type": "supported_on",
    },
    {
        "a": "eraser_center",
        "b": "spacemouse_box_center",
        "dir_anchor": "spacemouse_box_normal",
        "tol": 0.01,
        "type": "point_on_line",
    },
    {
        "ang_tol": 0.1,
        "lin_tol": 0.01,
        "subject": "eraser",
        "type": "at_rest",
    },
)
_PICK_V9_S4_TERMINAL = (
    {
        "subject": "eraser",
        "support": "spacemouse_box",
        "tol": 0.005,
        "type": "supported_on",
    },
    {"object_anchor": "eraser", "type": "not_held"},
)

# Same S4, plus the at_rest terminal.  A separate tuple rather than an edit:
# the existing one seals the v10/v11 programs and a VERIFIED Pick run.
_PICK_AT_REST_S4_TERMINAL = _PICK_V9_S4_TERMINAL + (
    {
        "ang_tol": 0.1,
        "lin_tol": 0.01,
        "subject": "eraser",
        "type": "at_rest",
    },
)

_CUP_V7_S1_NAME = "align_side_preapproach_cup_open"
_CUP_V7_S1_RUNNING = (
    {
        "a": "gripper_center",
        "b": "cup_center",
        "offset": [-0.1, 0.0, 0.005],
        "tol": 0.025,
        "type": "relative_position",
    },
    {
        "a": "gripper_center",
        "antiparallel": False,
        "b": "world_x",
        "tol_deg": 12.0,
        "type": "axis_parallel",
    },
    {
        "a": "gripper_closing_axis",
        "antiparallel": False,
        "b": "world_y",
        "tol_deg": 12.0,
        "type": "axis_parallel",
    },
    {
        "state": "open",
        "tol": 0.0,
        "type": "gripper_command",
        "width_m": None,
    },
)
_CUP_V7_S1_TERMINAL = (
    {
        "a": "gripper_center",
        "b": "cup_center",
        "offset": [-0.1, 0.0, 0.005],
        "tol": 0.035,
        "type": "relative_position",
    },
    {
        "a": "gripper_center",
        "antiparallel": False,
        "b": "world_x",
        "tol_deg": 15.0,
        "type": "axis_parallel",
    },
    {
        "a": "gripper_closing_axis",
        "antiparallel": False,
        "b": "world_y",
        "tol_deg": 15.0,
        "type": "axis_parallel",
    },
    {
        "state": "open",
        "tol": 0.01,
        "type": "gripper_state",
        "width_m": None,
    },
)

_FLIP_V1_S0_NAME = "approach_robot_proximal_eraser_end_open"
_FLIP_V1_S0_RUNNING = (
    {
        "a": "gripper_center",
        "b": "eraser_material_minus_x_grasp",
        "offset": [0.0, 0.0, 0.0],
        "tol": 0.012,
        "type": "relative_position",
    },
    {
        "a": "gripper_center",
        "antiparallel": True,
        "b": "world_up",
        "tol_deg": 12.0,
        "type": "axis_parallel",
    },
    {
        "a": "gripper_closing_axis",
        "b": "eraser_axis",
        "theta_deg": 90.0,
        "tol_deg": 12.0,
        "type": "axis_angle",
    },
    {
        "state": "open",
        "tol": 0.0,
        "type": "gripper_command",
        "width_m": None,
    },
)
_FLIP_V1_S0_TERMINAL = _FLIP_V1_S0_RUNNING[:3]

_TOOLPUSH_8CM_STAGES = {
    "acquire_eraser_box_contact_held": (
        (
            {
                "state": "closed",
                "tol": 0.0,
                "type": "gripper_command",
                "width_m": None,
            },
            {"object_anchor": "eraser", "type": "object_held"},
            {
                "target": "spacemouse_box_center",
                "tool": "eraser_center",
                "type": "tool_mediation",
            },
            {
                "a": "eraser_center",
                "b": "spacemouse_box_center",
                "dist": 0.0,
                "tol": 0.0,
                "type": "point_point_distance",
            },
        ),
        (
            {"a": "eraser_center", "b": "spacemouse_box_center", "type": "contact"},
            {"object_anchor": "eraser", "type": "object_held"},
        ),
    ),
    "push_box_positive_x_8cm_with_eraser_held": (
        (
            {
                "state": "closed",
                "tol": 0.0,
                "type": "gripper_command",
                "width_m": None,
            },
            {"object_anchor": "eraser", "type": "object_held"},
            {
                "target": "spacemouse_box_center",
                "tool": "eraser_center",
                "type": "tool_mediation",
            },
            {
                "a": "eraser_center",
                "b": "spacemouse_box_center",
                "dist": 0.0,
                "tol": 0.0,
                "type": "point_point_distance",
            },
            {
                "a": "spacemouse_box_center",
                "b": "spacemouse_box_initial_center",
                "dir_anchor": "world_x",
                "tol": 0.01,
                "type": "point_on_line",
            },
            {
                "a": "spacemouse_box_initial_center",
                "axis": [1.0, 0.0, 0.0],
                "b": "spacemouse_box_center",
                "margin": 0.08,
                "sign": 1.0,
                "type": "signed_axis_gap",
            },
        ),
        (
            {
                "a": "spacemouse_box_initial_center",
                "axis": [1.0, 0.0, 0.0],
                "b": "spacemouse_box_center",
                "margin": 0.08,
                "sign": 1.0,
                "type": "signed_axis_gap",
            },
            {
                "a": "spacemouse_box_center",
                "b": "spacemouse_box_initial_center",
                "dir_anchor": "world_x",
                "tol": 0.01,
                "type": "point_on_line",
            },
            {"object_anchor": "eraser", "type": "object_held"},
        ),
    ),
}

def validate_cost_shaping_profile(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return one registered profile, or fail closed."""
    if value is None and allow_none:
        return None
    if isinstance(value, str):
        # Prefix-only causal categories share the exact C5 objective. Resolve
        # their public experiment names before any CPU/device cost compiler
        # sees the profile so they cannot acquire a second cost identity.
        value = _COST_SHAPING_PROFILE_ALIASES.get(value, value)
    if not isinstance(value, str) or value not in COST_SHAPING_PROFILES:
        raise ValueError(
            "cost_shaping_profile must be one of "
            f"{COST_SHAPING_PROFILES}, got {value!r}"
        )
    return value


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("predicate identity contains a non-finite float")
        return 0.0 if value == 0.0 else float(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items())
        }
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    raise TypeError(
        "predicate identity contains unsupported value "
        f"{type(value).__name__}"
    )


def continuous_geometric_predicate_identity(
    predicate: Predicate,
) -> dict[str, Any] | None:
    """Return the typed relation/target identity eligible for suppression.

    ``None`` is the explicit non-eligible result for semantic, Boolean, jaw,
    held, and dynamic predicates.  The returned dictionary is JSON-safe and
    stable; callers can put it directly in evidence packets.
    """
    # TemporalHold carries trajectory/history semantics (including ``frames``)
    # that are not equivalent to its inner predicate's pointwise residual.
    # Neither scope may alias it to an ordinary continuous-geometric term.
    if isinstance(predicate, TemporalHold):
        return None
    planned = predicate
    if planned.type not in _CONTINUOUS_GEOMETRIC_TYPES:
        return None
    data = dict(planned.to_dict())
    predicate_type = str(data.pop("type"))
    for field in _TOLERANCE_FIELDS:
        data.pop(field, None)
    return {
        "type": predicate_type,
        "operands_and_target": _json_value(data),
    }


def _hashable_identity(identity: Mapping[str, Any]) -> Any:
    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(
                (str(key), freeze(item))
                for key, item in sorted(value.items())
            )
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        return value

    return freeze(identity)


def _is_exact_pick_v9_s4_release(stage: Stage) -> bool:
    return (
        stage.name == _PICK_V9_S4_NAME
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.running)
        == _PICK_V9_S4_RUNNING
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.terminal)
        == _PICK_V9_S4_TERMINAL
    )


def _is_exact_pick_at_rest_s4_release(stage: Stage) -> bool:
    """Same S4, with the at_rest terminal appended."""
    return (
        stage.name == _PICK_V9_S4_NAME
        and tuple(_json_value(pr.to_dict()) for pr in stage.running)
        == _PICK_V9_S4_RUNNING
        and tuple(_json_value(pr.to_dict()) for pr in stage.terminal)
        == _PICK_AT_REST_S4_TERMINAL
    )


def _is_exact_pick_v9_s0_approach(stage: Stage) -> bool:
    return (
        stage.name == _PICK_V9_S0_NAME
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.running)
        == _PICK_V9_S0_RUNNING
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.terminal)
        == _PICK_V9_S0_TERMINAL
    )


def _is_exact_cup_v7_s1(stage: Stage) -> bool:
    return (
        stage.name == _CUP_V7_S1_NAME
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.running)
        == _CUP_V7_S1_RUNNING
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.terminal)
        == _CUP_V7_S1_TERMINAL
    )


def _is_exact_flip_v1_s0(stage: Stage) -> bool:
    return (
        stage.name == _FLIP_V1_S0_NAME
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.running)
        == _FLIP_V1_S0_RUNNING
        and tuple(_json_value(predicate.to_dict()) for predicate in stage.terminal)
        == _FLIP_V1_S0_TERMINAL
    )


_CUP_SIMPLE_TILT_STAGES = {
    "tilt_held_cup_positive_x_60deg": (
        (
            {
                "state": "closed",
                "tol": 0.0,
                "type": "gripper_command",
                "width_m": None,
            },
            {
                "body": "cup",
                "target_angle_deg": 60.0,
                "target_axis": [0.0, 1.0, 0.0],
                "tol_deg": 10.0,
                "type": "relative_orientation",
            },
        ),
        (
            {"object_anchor": "cup", "type": "object_held"},
            {
                "body": "cup",
                "target_angle_deg": 60.0,
                "target_axis": [0.0, 1.0, 0.0],
                "tol_deg": 10.0,
                "type": "relative_orientation",
            },
        ),
    ),
}


def _is_exact_cup_simple_tilt_stage(stage: Stage) -> bool:
    expected = _CUP_SIMPLE_TILT_STAGES.get(stage.name)
    return expected is not None and (
        tuple(_json_value(predicate.to_dict()) for predicate in stage.running)
        == expected[0]
        and tuple(
            _json_value(predicate.to_dict()) for predicate in stage.terminal
        )
        == expected[1]
    )


def _is_exact_toolpush_8cm_stage(stage: Stage) -> bool:
    expected = _TOOLPUSH_8CM_STAGES.get(stage.name)
    return expected is not None and (
        tuple(_json_value(predicate.to_dict()) for predicate in stage.running)
        == expected[0]
        and tuple(
            _json_value(predicate.to_dict()) for predicate in stage.terminal
        )
        == expected[1]
    )


@dataclass(frozen=True)
class StageCostShapingPlan:
    """Static per-stage multipliers and auditable zeroed identities."""

    profile: str
    running_multipliers: tuple[float, ...]
    terminal_multipliers: tuple[float, ...]
    zeroed_running_terms: tuple[dict[str, Any], ...]


def cost_shaping_profile_binds(program: Any, profile: Any) -> bool:
    """Whether ``profile`` actually shapes any stage of ``program``.

    Each profile seals the stage identities it applies to, and
    ``stage_cost_shaping_plan`` returns all-1.0 multipliers for a stage it does
    not seal.  Per stage that is correct -- a program has several stages and a
    profile targets one -- but across the WHOLE program it means a profile can
    be declared, pass every identity assertion, and change nothing.

    ``cup_simple_tilt_terminal_held_60_v1`` seals
    ``tilt_held_cup_positive_x_60deg``; Cup v20, v23 and v24 name their tilt
    stage ``tilt_held_cup_positive_x_30deg``, so its advertised
    ``terminal[0] = 60.0`` boost on ``ObjectHeld`` was inert on all three.  A
    stage whose name matches but whose contents were edited already raises;
    only the name mismatch was silent.

    The criterion is a moved multiplier, not a matched name: a future profile
    that seals a stage and then leaves every multiplier at 1.0 is still a no-op.
    ``baseline`` (and ``None``, which the runner accepts for the same meaning)
    declare "no shaping" honestly and always bind.
    """
    registered = validate_cost_shaping_profile(profile, allow_none=True)
    if registered is None or registered in (
        BASELINE_COST_SHAPING,
        TYPED_RESIDUAL_LIBRARY_V1,
        TYPED_RESIDUAL_LIBRARY_V2,
        TYPED_RESIDUAL_LIBRARY_V3,
        TYPED_RESIDUAL_LIBRARY_V4,
    ):
        # Both declare a TOTAL pricing rule for any program: baseline "no
        # shaping", the library "by-type weights". Neither can silently no-op.
        return True
    for stage in getattr(program, "stages", ()):
        plan = stage_cost_shaping_plan(stage, registered)
        if (
            any(multiplier != 1.0 for multiplier in plan.running_multipliers)
            or any(multiplier != 1.0 for multiplier in plan.terminal_multipliers)
            or plan.zeroed_running_terms
        ):
            return True
    return False


# Per-type running-term weights for the supported predicate vocabulary.
# Terminal terms remain at 1.0; later library revisions adjust these running
# weights and goal parity without changing measured acceptance.
_TYPED_RESIDUAL_LIBRARY_WEIGHTS = {
    "robot_subject_contact": 10.0,
    "tool_mediation": 4.0,
}


def stage_cost_shaping_plan(
    stage: Stage,
    profile: Any,
) -> StageCostShapingPlan:
    """Compile one stage's categorical, task-independent shaping plan.

    A running identity may match at most one terminal identity.  Multiple
    terminal copies would make the claimed twin relationship ambiguous, so the
    focus profile refuses before a rollout can be authorized.
    """
    registered = validate_cost_shaping_profile(profile)
    terminal_multipliers = (1.0,) * len(stage.terminal)
    if registered == BASELINE_COST_SHAPING:
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=(1.0,) * len(stage.running),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=(),
        )

    if registered == TYPED_RESIDUAL_LIBRARY_V1:
        # The library prices residuals by type over any generated program:
        # robot contact at 10, ToolMediation at 4, and other types at 1.
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(
                _TYPED_RESIDUAL_LIBRARY_WEIGHTS.get(pred.type, 1.0)
                for pred in stage.running
            ),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=(),
        )

    if registered == TYPED_RESIDUAL_LIBRARY_V2:
        # GOAL-PARITY revision of the typed library (Addendum 87/88). The v1
        # per-type boosts stay -- they are what the reference tasks measured
        # as necessary support (contact 10, ToolMediation 4) -- but
        # within each stage every DISPLACEMENT-FAMILY term (the class a
        # generated program uses as its goal: signed_axis_gap,
        # relative_position, above_by, along_axis_gap, point_on_line,
        # capture_distance) is raised to the LARGEST boosted weight present in
        # that stage, so a generated goal can never be outbid by the library's
        # own support terms. Measured basis: byte-identical programs flip
        # outcome in BOTH directions on these weights (push: A72/A78/E3;
        # toolpush: A87). Stages with no boosted term reduce to v1 exactly.
        displacement = {
            "signed_axis_gap", "relative_position", "above_by",
            "along_axis_gap", "point_on_line", "capture_distance",
        }
        base = tuple(
            _TYPED_RESIDUAL_LIBRARY_WEIGHTS.get(pred.type, 1.0)
            for pred in stage.running
        )
        ceiling = max(base, default=1.0)
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(
                max(w, ceiling) if pred.type in displacement else w
                for w, pred in zip(base, stage.running)
            ),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=(),
        )

    if registered == TYPED_RESIDUAL_LIBRARY_V3:
        # v1 minus the contact boost (Addendum 91). Three regimes on one
        # byte-identical toolpush program read v1 PASS / baseline FAIL /
        # v2 FAIL: the v1 RATIO (mediation 4 : displacement 1) is what the
        # tool approach needs, while push's pathology was only ever the
        # contact boost against unboosted goals (A72/A78/E3). Dropping that
        # single boost makes v3 coincide with the PROVEN regime per program
        # class: baseline for contact-carrying push programs, v1 for
        # mediation-carrying toolpush programs.
        v3 = dict(_TYPED_RESIDUAL_LIBRARY_WEIGHTS)
        v3["robot_subject_contact"] = 1.0
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(
                v3.get(pred.type, 1.0) for pred in stage.running
            ),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=(),
        )

    if registered == TYPED_RESIDUAL_LIBRARY_V4:
        # v3 + stage-conditional goal parity (Addendum 104d). Discriminated on
        # byte-identical push bytes: v3(=baseline on push) crawls +2.3 mm;
        # v2 goal-parity self-terminates at +118.7 mm (single task outcome). v2 was
        # rejected only because its unconditional ceiling broke ToolPush's
        # proven mediation 4 : displacement 1 ratio (A91). v4 therefore
        # boosts the displacement family to 10 ONLY in stages whose running
        # list carries robot_subject_contact (the nonprehensile
        # contact-carrying signature). ToolPush/pick drafts audited: zero
        # such stages, so v4 is plan-identical to v3 there by construction.
        displacement = {
            "signed_axis_gap", "relative_position", "above_by",
            "along_axis_gap", "point_on_line", "capture_distance",
        }
        v4 = dict(_TYPED_RESIDUAL_LIBRARY_WEIGHTS)
        v4["robot_subject_contact"] = 1.0
        contact_carrying = any(
            pred.type == "robot_subject_contact" for pred in stage.running
        )
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(
                10.0
                if contact_carrying and pred.type in displacement
                else v4.get(pred.type, 1.0)
                for pred in stage.running
            ),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=(),
        )

    if registered == CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1:
        if stage.name not in _CUP_SIMPLE_TILT_STAGES:
            return StageCostShapingPlan(
                profile=registered,
                running_multipliers=(1.0,) * len(stage.running),
                terminal_multipliers=terminal_multipliers,
                zeroed_running_terms=(),
            )
        if not _is_exact_cup_simple_tilt_stage(stage):
            raise ValueError(
                "cost shaping refuses a modified sealed Cup tilt stage contract"
            )
        # terminal[0] is object_held (binary {0,1}); terminal[1] is
        # relative_orientation (squared normalised residual, range [0, 289]).
        # Measured on the sealed v10 packet, the multiplier needed for a held
        # candidate to win is 12.58 at S3 c0 and rises past 53 by c22, so 10 is
        # insufficient before the first replan and 60 clears the whole window
        # over which the cup was still physically held.
        terminal = [1.0] * len(stage.terminal)
        terminal[0] = 60.0
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=(1.0,) * len(stage.running),
            terminal_multipliers=tuple(terminal),
            zeroed_running_terms=(),
        )

    if registered in {
        TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1,
        TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
    }:
        if stage.name not in _TOOLPUSH_8CM_STAGES:
            return StageCostShapingPlan(
                profile=registered,
                running_multipliers=(1.0,) * len(stage.running),
                terminal_multipliers=terminal_multipliers,
                zeroed_running_terms=(),
            )
        if not _is_exact_toolpush_8cm_stage(stage):
            raise ValueError(
                "cost shaping refuses a modified sealed ToolPush-8cm stage contract"
            )
        running = [1.0] * len(stage.running)
        running[2] = 4.0
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(running),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=(),
        )

    if registered == CUP_V7_S1_RUNNING_POSITION_FOCUS_V1:
        if stage.name != _CUP_V7_S1_NAME:
            return StageCostShapingPlan(
                profile=registered,
                running_multipliers=(1.0,) * len(stage.running),
                terminal_multipliers=terminal_multipliers,
                zeroed_running_terms=(),
            )
        if not _is_exact_cup_v7_s1(stage):
            raise ValueError(
                "cost shaping refuses a modified sealed Cup-v7 S1 contract"
            )
        running = [1.0] * len(stage.running)
        running[0] = 0.0
        data = dict(stage.running[0].to_dict())
        predicate_type = str(data.pop("type"))
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(running),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=({
                "running_term_index": 0,
                "terminal_term_index": 0,
                "reason": (
                    "cup_v7_s1_duplicate_position_path_suppression"
                ),
                "identity": {
                    "type": predicate_type,
                    "operands_and_target": _json_value(data),
                },
            },),
        )

    if registered == FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1:
        if stage.name != _FLIP_V1_S0_NAME:
            return StageCostShapingPlan(
                profile=registered,
                running_multipliers=(1.0,) * len(stage.running),
                terminal_multipliers=terminal_multipliers,
                zeroed_running_terms=(),
            )
        if not _is_exact_flip_v1_s0(stage):
            raise ValueError(
                "cost shaping refuses a modified sealed Flip-v1 S0 contract"
            )
        running = [1.0] * len(stage.running)
        running[0] = 0.0
        data = dict(stage.running[0].to_dict())
        predicate_type = str(data.pop("type"))
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(running),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=({
                "running_term_index": 0,
                "terminal_term_index": 0,
                "reason": (
                    "flip_v1_s0_duplicate_position_path_suppression"
                ),
                "identity": {
                    "type": predicate_type,
                    "operands_and_target": _json_value(data),
                },
            },),
        )

    if registered in (
        PICK_V9_S4_DIRECT_RELEASE_V1,
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
        PICK_V8_AT_REST_TERMINAL_V6,
    ):
        multipliers = [1.0] * len(stage.running)
        zeroed: tuple[dict[str, Any], ...] = ()
        if (
            registered == PICK_V9_S4_DIRECT_RELEASE_V1
            and _is_exact_pick_v9_s0_approach(stage)
        ):
            zeroed_indices = (0,)
            reason = "pick_v9_s0_terminal_duplicate_running_suppression"
        elif (
            registered == PICK_V8_AT_REST_TERMINAL_V6
            and _is_exact_pick_at_rest_s4_release(stage)
        ):
            # Identical running suppression to the v3 profile; the only change
            # is that the program now carries at_rest as a TERMINAL, so the
            # gate requires rest.  Keeping the running suppression identical is
            # what makes this a single variable.
            zeroed_indices = (2, 4, 5)
            reason = "pick_v8_at_rest_terminal_running_suppression"
        elif _is_exact_pick_v9_s4_release(stage):
            zeroed_indices = (
                (2, 5)
                if registered == PICK_V8_COMBINED_RELEASE_LATERAL_V5
                else (2, 4, 5)
            )
            if registered == PICK_V8_COMBINED_RELEASE_LATERAL_V5:
                reason = "pick_v8_c5_release_lateral_running_suppression"
            elif registered == PICK_V8_COMBINED_DIRECT_RELEASE_V3:
                reason = "pick_v8_c3_direct_release_running_suppression"
            else:
                reason = "pick_v9_s4_direct_release_running_suppression"
        else:
            zeroed_indices = ()
            reason = ""
        if zeroed_indices:
            zeroed_items: list[dict[str, Any]] = []
            for running_index in zeroed_indices:
                predicate = stage.running[running_index]
                data = dict(predicate.to_dict())
                predicate_type = str(data.pop("type"))
                multipliers[running_index] = 0.0
                zeroed_items.append({
                    "running_term_index": running_index,
                    "terminal_term_index": None,
                    "reason": reason,
                    "identity": {
                        "type": predicate_type,
                        "operands_and_target": _json_value(data),
                    },
                })
            zeroed = tuple(zeroed_items)
        return StageCostShapingPlan(
            profile=registered,
            running_multipliers=tuple(multipliers),
            terminal_multipliers=terminal_multipliers,
            zeroed_running_terms=zeroed,
        )

    terminal_by_identity: dict[Any, list[int]] = {}
    terminal_identity_values: dict[Any, dict[str, Any]] = {}
    for index, predicate in enumerate(stage.terminal):
        identity = continuous_geometric_predicate_identity(predicate)
        if identity is None:
            continue
        key = _hashable_identity(identity)
        terminal_by_identity.setdefault(key, []).append(index)
        terminal_identity_values[key] = identity

    multipliers: list[float] = []
    zeroed: list[dict[str, Any]] = []
    for running_index, predicate in enumerate(stage.running):
        identity = continuous_geometric_predicate_identity(predicate)
        if identity is None:
            multipliers.append(1.0)
            continue
        key = _hashable_identity(identity)
        matches = terminal_by_identity.get(key, [])
        if len(matches) > 1:
            raise ValueError(
                "ambiguous terminal predicate identity for cost shaping: "
                f"stage={stage.name!r}, running_index={running_index}, "
                f"terminal_indices={matches}, identity={identity!r}"
            )
        if len(matches) == 1:
            multipliers.append(0.0)
            zeroed.append({
                "running_term_index": running_index,
                "terminal_term_index": matches[0],
                "identity": terminal_identity_values[key],
            })
        else:
            multipliers.append(1.0)
    return StageCostShapingPlan(
        profile=registered,
        running_multipliers=tuple(multipliers),
        terminal_multipliers=terminal_multipliers,
        zeroed_running_terms=tuple(zeroed),
    )


def cost_shaping_identity(profile: Any) -> dict[str, Any]:
    """Return the complete immutable formula/safety identity."""
    registered = validate_cost_shaping_profile(profile)
    common = {
        "schema": "oap_cost_shaping_identity_v1",
        "profile": registered,
        "terminal_multiplier": 1.0,
        "terminal_predicates_thresholds_selectors": "unchanged",
        "hard_validity_and_safety": "unchanged",
        "task_program_json": "unchanged",
        "formal_control_regularizer": "unchanged",
    }
    if registered == BASELINE_COST_SHAPING:
        return {
            **common,
            "mode": "registered_baseline",
            "running_multiplier": 1.0,
            "zeroed_predicate_classes": [],
        }
    if registered == TYPED_RESIDUAL_LIBRARY_V1:
        return {
            **common,
            "mode": "typed_residual_library_by_predicate_type",
            "running_multiplier_by_type": dict(
                _TYPED_RESIDUAL_LIBRARY_WEIGHTS
            ),
            "running_multiplier_default": 1.0,
            "zeroed_predicate_classes": [],
        }
    if registered == PICK_V9_S4_DIRECT_RELEASE_V1:
        return {
            **common,
            "mode": "pick_v9_s0_s4_direct_running_only",
            "matching": "exact_reviewed_pick_v9_s0_and_s4_stage_contracts",
            "matched_running_multiplier": 0.0,
            "unmatched_running_multiplier": 1.0,
            "preserved_running_predicate_classes_by_stage": {
                "approach_eraser_open": [
                    "axis_parallel",
                    "axis_angle",
                    "gripper_command",
                ],
                "release_eraser": [
                    "gripper_command",
                    "not_held",
                    "supported_on",
                ],
            },
            "zeroed_running_predicate_classes_by_stage": {
                "approach_eraser_open": [
                    "relative_position",
                ],
                "release_eraser": [
                    "min_distance",
                    "point_on_line",
                    "at_rest",
                ],
            },
            "zeroed_scope": "running_only",
            "zeroed_stages": ["approach_eraser_open", "release_eraser"],
            "terminal_contract": (
                "s0_relative_position_axis_parallel_axis_angle_and_"
                "s4_supported_on_not_held_unchanged"
            ),
        }
    if registered == PICK_V8_COMBINED_DIRECT_RELEASE_V3:
        return {
            **common,
            "mode": "pick_v8_c3_release_direct_running_only",
            "matching": "exact_reviewed_pick_v8_release_stage_contract",
            "matched_running_multiplier": 0.0,
            "unmatched_running_multiplier": 1.0,
            "preserved_running_predicate_classes_by_stage": {
                "release_eraser": [
                    "gripper_command",
                    "not_held",
                    "supported_on",
                ],
            },
            "zeroed_running_predicate_classes_by_stage": {
                "release_eraser": [
                    "min_distance",
                    "point_on_line",
                    "at_rest",
                ],
            },
            "zeroed_scope": "running_only",
            "zeroed_stages": ["release_eraser"],
            "terminal_contract": "supported_on_not_held_unchanged",
        }
    if registered == PICK_V8_COMBINED_RELEASE_LATERAL_V5:
        return {
            **common,
            "mode": "pick_v8_c5_release_lateral_running_only",
            "matching": "exact_reviewed_pick_v8_release_stage_contract",
            "matched_running_multiplier": 0.0,
            "unmatched_running_multiplier": 1.0,
            "preserved_running_predicate_classes_by_stage": {
                "release_eraser": [
                    "gripper_command",
                    "not_held",
                    "supported_on",
                    "point_on_line",
                ],
            },
            "zeroed_running_predicate_classes_by_stage": {
                "release_eraser": ["min_distance", "at_rest"],
            },
            "zeroed_scope": "running_only",
            "zeroed_stages": ["release_eraser"],
            "terminal_contract": "supported_on_not_held_unchanged",
        }
    if registered == TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1:
        return {
            **common,
            "mode": "toolpush_8cm_mediation_weight_4",
            "matching": "exact_sealed_toolpush_8cm_contact_and_push_stages",
            "running_term_type": "tool_mediation",
            "running_term_index": 2,
            "running_multiplier": 4.0,
            "unmatched_running_multiplier": 1.0,
            "weighted_stages": list(_TOOLPUSH_8CM_STAGES),
            "terminal_contract": "all_terminal_terms_unchanged",
        }
    if registered == TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2:
        return {
            **common,
            "mode": "toolpush_8cm_mediation_duration_weight_4",
            "matching": "exact_sealed_toolpush_8cm_contact_and_push_stages",
            "running_term_type": "tool_mediation",
            "running_term_index": 2,
            "running_multiplier": 4.0,
            "positive_contact_residual": "one_minus_full_horizon_contact_fraction",
            "robot_direct_contact_residual": "any_step_boolean",
            "unmatched_running_multiplier": 1.0,
            "weighted_stages": list(_TOOLPUSH_8CM_STAGES),
            "terminal_contract": "all_terminal_terms_unchanged",
        }
    if registered == CUP_V7_S1_RUNNING_POSITION_FOCUS_V1:
        return {
            **common,
            "mode": "cup_v7_s1_running_position_suppression",
            "matching": "exact_sealed_cup_v7_s1_stage_contract",
            "matched_running_multiplier": 0.0,
            "unmatched_running_multiplier": 1.0,
            "zeroed_scope": "running_only",
            "zeroed_stages": [_CUP_V7_S1_NAME],
            "zeroed_running_term_indices": [0],
            "preserved_running_term_indices": [1, 2, 3],
            "terminal_contract": "all_four_s1_terminal_terms_unchanged",
        }
    if registered == FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1:
        return {
            **common,
            "mode": "flip_v1_s0_running_position_suppression",
            "matching": "exact_sealed_flip_v1_s0_stage_contract",
            "matched_running_multiplier": 0.0,
            "unmatched_running_multiplier": 1.0,
            "zeroed_scope": "running_only",
            "zeroed_stages": [_FLIP_V1_S0_NAME],
            "zeroed_running_term_indices": [0],
            "preserved_running_term_indices": [1, 2, 3],
            "terminal_contract": "all_three_s0_terminal_terms_unchanged",
        }
    return {
        **common,
        "mode": "terminal_twin_running_suppression",
        "matching": (
            "same_stage_continuous_geometric_type_operands_and_target_"
            "excluding_tolerance"
        ),
        "matched_running_multiplier": 0.0,
        "unmatched_running_multiplier": 1.0,
        "always_preserved_running_classes": [
            "contact_and_no_contact",
            "robot_subject_contact",
            "tool_mediation",
            "gripper_command_and_state",
            "object_held_and_not_held",
            "at_rest",
            "temporal_hold",
            "non_geometric",
        ],
    }


def cost_shaping_term_multipliers(
    term_descriptors: tuple[dict[str, Any], ...],
) -> tuple[float, ...]:
    """Recover validated static multipliers from evaluator descriptors."""
    multipliers: list[float] = []
    profiles: set[str] = set()
    annotated = 0
    for descriptor in term_descriptors:
        shaping = descriptor.get("cost_shaping")
        if shaping is None:
            multipliers.append(1.0)
            continue
        annotated += 1
        if not isinstance(shaping, Mapping):
            raise ValueError("cost_shaping term descriptor must be an object")
        profile = validate_cost_shaping_profile(shaping.get("profile"))
        profiles.add(profile)
        multiplier = shaping.get("multiplier")
        four_x_profiles = {
            TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1,
            TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
        }
        if profile in (TYPED_RESIDUAL_LIBRARY_V1, TYPED_RESIDUAL_LIBRARY_V2,
                       TYPED_RESIDUAL_LIBRARY_V3, TYPED_RESIDUAL_LIBRARY_V4):
            # The library's full multiplier range: the by-type weights plus
            # the default (v2 raises displacement terms to the stage ceiling,
            # so its range is the same set). Any other value is a
            # construction error. The v3 omission here killed two episodes at
            # the first mediation-priced solve (Addendum 97).
            allowed_multipliers = (1.0, 4.0, 10.0)
        elif profile == CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1:
            allowed_multipliers = (1.0, 60.0)
        else:
            allowed_multipliers = (
                (1.0, 4.0) if profile in four_x_profiles else (0.0, 1.0)
            )
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or float(multiplier) not in allowed_multipliers
        ):
            raise ValueError(
                "cost_shaping multiplier is not registered for its profile"
            )
        # A terminal multiplier may differ from 1.0 only for a profile that
        # registered it above.  The blanket refusal predated any such profile;
        # keeping it would make a registered terminal weight unusable, which is
        # how Cup v12/v13 became dead code that raises at the first S3 solve.
        if (
            descriptor.get("scope") == "terminal"
            and float(multiplier) != 1.0
            and profile not in {
                CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1,
            }
        ):
            raise ValueError("cost shaping may not change terminal terms")
        multipliers.append(float(multiplier))
    if len(profiles) > 1:
        raise ValueError("term descriptors mix cost_shaping profiles")
    if annotated not in (0, len(term_descriptors)):
        raise ValueError(
            "cost_shaping metadata must cover every term descriptor"
        )
    return tuple(multipliers)
