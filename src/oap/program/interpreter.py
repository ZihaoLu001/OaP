"""NumPy reference evaluation of the pose-grounded part of a TaskProgram.

The same VLM-synthesized program is consumed by planning, measured stage
termination, and final verification. Its ``running`` predicates provide dense
planning signal; its ``terminal`` predicates are both the predicted endpoint
cost and the measured real-state success condition.

A "trajectory" is a list[AnchorSet] (predicted by the outcome model, or the
single re-observed real AnchorSet wrapped in a length-1 list).
Exact contact costs are added by the rollout adapter: sparse pose anchors
cannot recover two-finger contact or whole-horizon tool-contact events.
"""
from __future__ import annotations

import math

import numpy as np

from .anchors import AnchorSet
from .cost_shaping import stage_cost_shaping_plan
from .predicates import (
    AboveBy,
    AlongAxisGap,
    AngleAboutAxis,
    AtRest,
    AxisAngle,
    AxisParallel,
    CaptureDistance,
    GripperCommand,
    GripperState,
    InRegion,
    InteractionAlign,
    MinDistance,
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

_EPS = 1e-6
_TRANSLATION_DOMAIN_M = 1.0
_ANGLE_DOMAIN_RAD = math.pi
_GRIPPER_DOMAIN_M = 0.085
_BOOLEAN_DOMAIN = 1.0
# A declared tolerance states where a residual is zero, not how steeply it is
# weighted. Divide by an arbitrarily tight tolerance and that one predicate can
# mint an arbitrarily large gradient inside a sum it shares with fixed-domain
# families, which no longer compare to anything.
#
# Measured 2026-08-03 on the VLM placement program, stage `place_eraser_on_box`:
# terminal `on_plane` declares tol=0.01 m while the stage's own grounded
# translation domain is 0.5142 m, so a 9.2 cm height error scored 84.8 per
# endpoint while losing the object entirely -- running plus both terminal
# endpoints -- scored 3.0. Retention was 1.5% of the residual budget, the
# optimizer spent the grasp to buy millimetres, and 89 of 96 winning candidates
# predicted not-held for the full horizon while remaining physically valid.
#
# Flooring the divisor at a fraction of the stage's own translation domain keeps
# a declared tolerance meaningful where it is loose and stops it from dominating
# where it is tight. Satisfaction is unaffected: stage advance still tests the
# raw predicate against the program's own tol, never this scale.
_DECLARED_SCALE_FLOOR_FRACTION = 0.1


def _planning_predicate(predicate: Predicate) -> Predicate:
    """Return the scalar predicate evaluated by the planning objective."""
    while isinstance(predicate, TemporalHold):
        predicate = predicate.inner
    return predicate


def _translation_anchor_names(predicate: Predicate) -> tuple[str, ...]:
    """Point anchors whose grounded spread defines the metric fallback."""
    if isinstance(
        predicate,
        (
            PointPointDistance,
            CaptureDistance,
            RelativePosition,
            MinDistance,
            AboveBy,
            OnPlane,
        ),
    ):
        if isinstance(predicate, CaptureDistance):
            return ("gripper_center", predicate.subject)
        return (predicate.a, predicate.b)
    if isinstance(predicate, PointOnLine):
        return (predicate.a, predicate.b)
    if isinstance(predicate, InRegion):
        return (predicate.a, predicate.region)
    if isinstance(predicate, SignedAxisGap):
        return (predicate.a, predicate.b)
    if isinstance(predicate, AlongAxisGap):
        return (predicate.a, predicate.c)
    if isinstance(predicate, InteractionAlign):
        return (predicate.a, predicate.b)
    return ()


def _declared_translation_scale(predicate: Predicate) -> float:
    """Length already declared by a metric predicate, in metres."""
    if isinstance(predicate, PointPointDistance):
        return max(abs(float(predicate.dist)), float(predicate.tol))
    if isinstance(predicate, CaptureDistance):
        return 0.0
    if isinstance(predicate, RelativePosition):
        return max(
            float(np.linalg.norm(np.asarray(predicate.offset, dtype=float))),
            float(predicate.tol),
        )
    if isinstance(predicate, MinDistance):
        return abs(float(predicate.clearance))
    if isinstance(predicate, AboveBy):
        return max(abs(float(predicate.dist)), float(predicate.tol))
    if isinstance(predicate, (OnPlane, PointOnLine)):
        return float(predicate.tol)
    if isinstance(predicate, SignedAxisGap):
        return abs(float(predicate.margin))
    if isinstance(predicate, AlongAxisGap):
        return max(abs(float(predicate.dist)), float(predicate.tol))
    if isinstance(predicate, InteractionAlign):
        return max(abs(float(predicate.dist)), float(predicate.tol))
    return 0.0


def _translation_family_scale(
    predicates: tuple[Predicate, ...],
    anchors: AnchorSet,
) -> float:
    """Task-independent metric domain for zero-error, zero-tolerance terms.

    The scale is grounded only from the current stage: the AABB diagonal of
    its referenced point anchors, declared metric magnitudes, and referenced
    region half-extents.  A completely degenerate stage falls back to the
    canonical one-metre SI domain.  No candidate-batch statistic is involved.
    """
    names: set[str] = set()
    declared = 0.0
    region_extent = 0.0
    for predicate in predicates:
        names.update(_translation_anchor_names(predicate))
        declared = max(declared, _declared_translation_scale(predicate))
    points: list[np.ndarray] = []
    for name in sorted(names):
        anchor = anchors.get(name)
        if anchor is None:
            continue
        point = np.asarray(anchor.point, dtype=float)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            points.append(point)
        if anchor.region_half is not None:
            half = np.asarray(anchor.region_half, dtype=float)
            if half.shape == (3,) and np.all(np.isfinite(half)):
                region_extent = max(
                    region_extent,
                    float(np.linalg.norm(half)),
                )
    diagonal = 0.0
    if points:
        stacked = np.stack(points, axis=0)
        diagonal = float(
            np.linalg.norm(np.max(stacked, axis=0) - np.min(stacked, axis=0))
        )
    scale = max(diagonal, declared, region_extent)
    return scale if scale > _EPS else _TRANSLATION_DOMAIN_M


def _anchor_body_owner(name: str, anchors: AnchorSet) -> str | None:
    """Return the physical body carrying ``name``, when it is grounded."""
    anchor = anchors.get(name)
    if anchor is None:
        return None
    return str(anchor.attached_to or name)


def _body_radius_scale(owner: str, anchors: AnchorSet) -> float | None:
    """Return a reconstruction-grounded characteristic radius for one body.

    The grounder contract is ``region_half=[L/2, W/2, max(H/2, 0.5)]``:
    its first two components are the reconstructed footprint, while the third
    is deliberately widened for planar ``InRegion`` tests.  The uniform
    ``*_top`` anchor is exactly ``H/2`` from the owner centre even when tilted.
    Their Euclidean combination is the body's half-diagonal: a physical unit,
    not a grasp threshold or task heuristic.
    """
    footprint_radii: list[float] = []
    height_halves: list[float] = []
    owner_anchor = anchors.get(owner)
    owner_point = (
        np.asarray(owner_anchor.point, dtype=float)
        if owner_anchor is not None
        else None
    )
    for name in anchors.names():
        anchor = anchors[name]
        if str(anchor.attached_to or name) != owner:
            continue
        if (
            name.endswith("_top")
            and anchor.kind == "part"
            and owner_point is not None
            and owner_point.shape == (3,)
            and np.all(np.isfinite(owner_point))
        ):
            point = np.asarray(anchor.point, dtype=float)
            if point.shape == (3,) and np.all(np.isfinite(point)):
                height_halves.append(
                    float(np.linalg.norm(point - owner_point))
                )
        if anchor.region_half is None:
            continue
        half = np.asarray(anchor.region_half, dtype=float)
        if (
            half.shape != (3,)
            or not np.all(np.isfinite(half))
            or np.any(half <= _EPS)
        ):
            raise ValueError(
                f"body region {name!r} must have finite positive "
                "region_half for held-geometry normalization"
            )
        footprint_radii.append(float(np.linalg.norm(half[:2])))
    if not footprint_radii:
        return None
    radius = math.hypot(
        max(footprint_radii),
        max(height_halves, default=0.0),
    )
    return radius if radius > _EPS else None


def _held_gripper_body_scale(
    predicate: Predicate,
    predicates: tuple[Predicate, ...],
    anchors: AnchorSet,
) -> float | None:
    """Ground the dense held-body relation independently of goal geometry."""
    if isinstance(predicate, CaptureDistance):
        other = predicate.subject
    elif (
        isinstance(predicate, PointPointDistance)
        and predicate.dist == 0.0
        and predicate.tol == 0.0
        and "gripper_center" in {predicate.a, predicate.b}
    ):
        other = (
            predicate.b
            if predicate.a == "gripper_center"
            else predicate.a
        )
    else:
        return None
    other_owner = _anchor_body_owner(other, anchors)
    if other_owner is None:
        return None
    held_owners = {
        owner
        for term in predicates
        if isinstance(term, ObjectHeld)
        if (owner := _anchor_body_owner(term.object_anchor, anchors)) is not None
    }
    if other_owner not in held_owners:
        return None
    return _body_radius_scale(other_owner, anchors)


def _predicate_family(predicate: Predicate) -> str:
    if isinstance(
        predicate,
        (ObjectHeld, NotHeld, RobotSubjectContact, ToolMediation),
    ):
        return "boolean"
    if isinstance(predicate, (GripperCommand, GripperState)):
        return "gripper"
    if isinstance(
        predicate,
        (AxisParallel, AxisAngle, RelativeOrientation, RotationProgress, AngleAboutAxis),
    ):
        return "angle"
    if isinstance(predicate, InRegion):
        # Its planning residual is normalized by the declared region geometry
        # per axis, so an intentionally loose axis cannot dilute progress on
        # the axes that are currently outside the set.
        return "unit"
    if isinstance(predicate, AtRest):
        # Its residual normalizes each speed by its own declared tolerance.
        return "unit"
    if _translation_anchor_names(predicate):
        return "translation"
    if isinstance(predicate, NonGeometric):
        return "boolean"
    return "unit"


def planning_cost_scales(
    stage: Stage,
    reference_anchors: AnchorSet,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return fixed, dimensionless least-squares scales for one stage.

    Metric predicates use a length explicitly declared by the program when
    they have one (distance, offset, margin, clearance, or tolerance).
    Zero-length metric objectives share the stage's grounded translation
    domain, except the explicit gripper-to-``ObjectHeld``-body relation: its
    units come from that reconstructed body's half-diagonal, so unrelated goal
    geometry cannot silently dilute retention. ``InRegion`` instead uses its
    native per-axis region geometry. Angular, gripper, Boolean, and unitless
    predicates use one fixed physical family domain.

    Crucially, no scale depends on the predicate's residual at stage entry.
    A nearly satisfied relation therefore cannot create an arbitrarily small
    divisor that traps the optimizer at the current state.  The scales depend
    only on the fixed program and the stage geometry used to establish the
    zero-length translation domain. Angular predicates use their declared
    tolerance when it is positive and the canonical pi-radian domain
    otherwise.  No scale depends on sampled candidates, pool size, batch
    extrema, or receding-horizon progress.
    """
    running = tuple(_planning_predicate(p) for p in stage.running)
    terminal = tuple(_planning_predicate(p) for p in stage.terminal)
    all_predicates = running + terminal
    translation_scale = _translation_family_scale(
        all_predicates,
        reference_anchors,
    )
    family_fallback = {
        "translation": translation_scale,
        "angle": _ANGLE_DOMAIN_RAD,
        "gripper": _GRIPPER_DOMAIN_M,
        "boolean": _BOOLEAN_DOMAIN,
        "unit": 1.0,
    }
    declared_floor = _DECLARED_SCALE_FLOOR_FRACTION * float(translation_scale)

    def scale(predicate: Predicate) -> float:
        if isinstance(predicate, RotationProgress):
            # An explicit angular cost unit is independent of the one-sided
            # acceptance threshold. It is not an equality tolerance.
            return math.radians(float(predicate.scale_deg))
        if isinstance(predicate, InRegion):
            return 1.0
        if isinstance(predicate, SupportedOn):
            # Predicate-local, reconstruction-derived: the support's smaller
            # in-plane half extent is the natural footprint length. The
            # declared plane tolerance is deliberately NOT the divisor -- a
            # 5 mm tol would inflate a centimetre overhang into a dominating
            # squared term and recreate the trapped-scale failure mode.
            support = reference_anchors.get(predicate.support)
            half = getattr(support, "region_half", None)
            if half is not None:
                in_plane = np.asarray(half, dtype=float)[:2]
                if np.all(np.isfinite(in_plane)) and np.all(in_plane > _EPS):
                    return float(np.min(in_plane))
            return float(translation_scale)
        held_body_scale = _held_gripper_body_scale(
            predicate,
            all_predicates,
            reference_anchors,
        )
        if held_body_scale is not None:
            return held_body_scale
        family = _predicate_family(predicate)
        if family == "translation":
            # V11 (env OAP_DECLARED_SCALE_EXACT=1): an explicitly
            # authored positive tol IS the ruler -- skip the offset-max and
            # the 0.1x floor, so a 1 mm completion tol is a 1 mm scale.
            # Fixed authored tols are not the residual-dependent trapped
            # scale the docstring above forbids.
            import os as _os
            if _os.environ.get("OAP_DECLARED_SCALE_EXACT") == "1":
                _tol = getattr(predicate, "tol", None)
                if isinstance(_tol, (int, float)) and float(_tol) > _EPS:
                    return float(_tol)
            declared = _declared_translation_scale(predicate)
            if declared > _EPS:
                return float(max(declared, declared_floor))
        if family == "angle":
            tolerance = math.radians(
                max(0.0, float(predicate.tol_deg))
            )
            if tolerance > _EPS:
                return float(tolerance)
        return float(family_fallback[family])

    n_running = len(running)
    scales = tuple(scale(predicate) for predicate in all_predicates)
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("all planning cost scales must be finite and positive")
    return scales[:n_running], scales[n_running:]


def running_penalty(residual, scale=1.0):
    """Eq. (3), elementwise; vector residual components are summed by callers."""
    scale = np.asarray(scale, dtype=float)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("cost scales must be finite and positive")
    value = np.asarray(residual, dtype=float) / scale
    # Algebraically sqrt(1 + value**2) - 1, without cancellation near zero.
    return value * (value / (np.hypot(1.0, value) + 1.0))


def terminal_penalty(residual, threshold=0.0, scale=1.0):
    """Eq. (4), elementwise, with the same residual units for tau and scale."""
    scale = np.asarray(scale, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("cost scales must be finite and positive")
    if not np.all(np.isfinite(threshold)) or np.any(threshold < 0.0):
        raise ValueError("terminal thresholds must be finite and non-negative")
    return (np.maximum(np.asarray(residual, dtype=float) - threshold, 0.0) / scale) ** 2


def _scaled_cost(
    predicates: tuple[Predicate, ...],
    scales: tuple[float, ...],
    anchors: AnchorSet,
    multipliers: tuple[float, ...] | None = None,
    *,
    running: bool = False,
    contact_terms_in_rollout: bool = False,
) -> float:
    total = 0.0
    weights = (
        (1.0,) * len(predicates)
        if multipliers is None
        else multipliers
    )
    for predicate, scale, multiplier in zip(
        predicates,
        scales,
        weights,
        strict=True,
    ):
        if isinstance(predicate, (ObjectHeld, NotHeld, ToolMediation)):
            # Exact contact evidence is not carried by AnchorSet. The host
            # sampler adds these terms using the same Eq. (3)/(4) mappings;
            # the device evaluator evaluates the full objective directly.
            if not contact_terms_in_rollout:
                raise ValueError(
                    f"{predicate.type} requires exact contact evidence; "
                    "use the contact-aware device evaluator"
                )
            continue
        if running:
            residual = predicate.raw_residual(anchors)
            contribution = running_penalty(residual, scale)
        else:
            # The compatibility predicate API already subtracts its terminal
            # threshold. Apply no second subtraction here.
            contribution = terminal_penalty(predicate.residual(anchors), scale=scale)
        total += float(multiplier) * float(np.sum(contribution))
    return float(total)


def running_cost(
    stage: Stage,
    anchors: AnchorSet,
    *,
    reference_anchors: AnchorSet | None = None,
    cost_shaping_profile: str | None = None,
) -> float:
    """The running cost l(x_t): normalized task residuals at one state.

    This is the standard sampling-MPC running term (cf. hydrax
    ``Task.running_cost``, MuJoCo MPC's per-timestep residual sum). Running
    terms shape search but never decide measured stage success.
    """
    reference = anchors if reference_anchors is None else reference_anchors
    scales, _ = planning_cost_scales(stage, reference)
    predicates = tuple(_planning_predicate(p) for p in stage.running)
    multipliers = None
    if cost_shaping_profile is not None:
        multipliers = stage_cost_shaping_plan(
            stage,
            cost_shaping_profile,
        ).running_multipliers
    return _scaled_cost(predicates, scales, anchors, multipliers, running=True)


def terminal_cost(
    stage: Stage,
    anchors: AnchorSet,
    *,
    reference_anchors: AnchorSet | None = None,
    cost_shaping_profile: object | None = None,
) -> float:
    """The terminal cost phi(x_T): normalized endpoint residuals.

    The standard terminal term (cf. hydrax ``Task.terminal_cost``): the
    stage's sub-goal constraints, evaluated only at the trajectory's end.

    Registered terminal multipliers are applied here exactly as running
    multipliers are in ``running_cost``.  They used to be dropped on both
    lanes, so a profile could register a terminal weight that never reached
    the ranked cost at all.
    """
    reference = anchors if reference_anchors is None else reference_anchors
    _, scales = planning_cost_scales(stage, reference)
    predicates = tuple(_planning_predicate(p) for p in stage.terminal)
    multipliers = None
    if cost_shaping_profile is not None:
        multipliers = stage_cost_shaping_plan(
            stage,
            cost_shaping_profile,
        ).terminal_multipliers
    return _scaled_cost(predicates, scales, anchors, multipliers)


def trajectory_cost(stage: Stage, trajectory: list[AnchorSet], *,
                    dt: float | None = None,
                    reference_anchors: AnchorSet | None = None,
                    cost_shaping_profile: str | None = None,
                    contact_terms_in_rollout: bool = False) -> float:
    """Eq. (5) task contribution: mean running cost plus endpoint cost.

    Supply H predicted states to evaluate the H-state mean. Physics penalties
    are added by the rollout/controller layer. ``dt`` is a deprecated
    compatibility argument: if supplied it must equal 1/H; it cannot change
    the relative weighting of running and terminal costs.
    ``contact_terms_in_rollout=True`` is reserved for a contact-aware caller
    that separately adds exact contact costs; otherwise such terms raise.
    """
    if not trajectory:
        return 0.0
    if dt is not None and not math.isclose(float(dt), 1.0 / len(trajectory), rel_tol=1e-7):
        raise ValueError("Eq. (5) uses mean running cost; dt must equal 1/H or be omitted")
    reference = trajectory[0] if reference_anchors is None else reference_anchors
    running_scales, terminal_scales = planning_cost_scales(stage, reference)
    running = tuple(_planning_predicate(p) for p in stage.running)
    terminal = tuple(_planning_predicate(p) for p in stage.terminal)
    running_multipliers = None
    if cost_shaping_profile is not None:
        running_multipliers = stage_cost_shaping_plan(
            stage,
            cost_shaping_profile,
        ).running_multipliers
    run = sum(
        _scaled_cost(
            running,
            running_scales,
            state,
            running_multipliers,
            running=True,
            contact_terms_in_rollout=contact_terms_in_rollout,
        )
        for state in trajectory
    )
    terminal_multipliers = None
    if cost_shaping_profile is not None:
        terminal_multipliers = stage_cost_shaping_plan(
            stage,
            cost_shaping_profile,
        ).terminal_multipliers
    endpoint = _scaled_cost(
        terminal,
        terminal_scales,
        trajectory[-1],
        terminal_multipliers,
        contact_terms_in_rollout=contact_terms_in_rollout,
    )
    return float(run) / len(trajectory) + endpoint


def terminal_satisfied(
    stage: Stage,
    trajectory: list[AnchorSet],
    eps: float = _EPS,
) -> bool:
    """All terminal constraints hold at the stage END, with temporal-hold
    constraints requiring `frames` consecutive satisfied frames at the tail."""
    if not trajectory:
        return False
    final = trajectory[-1]
    for p in stage.terminal:
        from .verdict import measurement_channels
        inner = _planning_predicate(p)
        if isinstance(inner, (ObjectHeld, NotHeld, ToolMediation)):
            # No contact evidence is available in this AnchorSet-only helper.
            # The production stage verifier accepts explicit measured facts.
            return False
        if final.missing(measurement_channels(inner)):
            return False
        if isinstance(p, TemporalHold):
            need = max(1, int(p.frames))
            tail = trajectory[-need:]
            if len(tail) < need or not all(p.inner.satisfied(s, eps) for s in tail):
                return False
        else:
            if not p.satisfied(final, eps):
                return False
    return True
