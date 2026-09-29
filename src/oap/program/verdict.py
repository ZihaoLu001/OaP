"""Evaluate measured terminal conditions or refuse unverifiable success.

Role in the two-stage pipeline: this is call site 4 of the planning stage
(success verification on the re-observed real scene) and the honest boundary of
the whole system.

The router is AUTOMATIC and UNIFORM -- one computable test applied identically
to every task, never a per-task threshold or predicate:

  1. GROUNDABILITY: every anchor the program references must be finitely grounded
     in the (re-)observed scene. Else -> CANNOT_VERIFY(ungroundable:<anchor>).
  2. FALSIFIABILITY: any NonGeometric terminal -> CANNOT_VERIFY.
  3. HARD VERIFICATION: evaluate the SAME final-stage ``terminal`` predicates on
     measured anchors; all satisfied -> VERIFIED_SUCCESS, else VERIFIED_FAIL.

There is no second VLM judge. ``running`` predicates shape predicted
trajectories and never participate in measured success.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from .anchors import AnchorSet
from .geometry import unit
from .predicates import (
    Contact,
    NoContact,
    NotHeld,
    ObjectHeld,
    OnPlane,
    Predicate,
    RelativeOrientation,
    RobotSubjectContact,
    SignedAxisGap,
    SupportedOn,
    TemporalHold,
    ToolMediation,
)
from .program import TaskProgram, Stage


@dataclass(frozen=True)
class StageEvaluation:
    """One composed answer for both MPC stage termination and final verification.

    ``status`` is deliberately three-valued: ``None`` means that the available
    evidence cannot establish either success or failure.  Every hard piece of
    terminal evidence is carried alongside it so callers cannot accidentally
    grade geometry in one place and bolt undeclared host checks on elsewhere.
    ``rest_satisfied`` remains packet-compatible diagnostic telemetry only.
    """

    status: Optional[bool]
    reason: str
    terminal_satisfied: Optional[bool]
    temporal_hold_satisfied: Optional[bool]
    rest_satisfied: Optional[bool]
    tool_mediation_satisfied: Optional[bool]
    object_held_satisfied: Optional[bool]
    ejected: tuple[str, ...] = ()
    # Diagnostic telemetry only: the raw measured residual of each evaluable
    # terminal predicate on the final measured frame. Success is decided by
    # the Boolean evidence above, never re-derived from these numbers.
    measured_residuals: tuple[dict[str, Any], ...] = ()
    # Exact full-physics contact at the executed endpoint for a declared
    # terminal Contact(subject, target). ``None`` means the endpoint channel
    # was not measured, which must never be confused with no contact. Kept
    # last so existing positional construction retains its field order.
    terminal_contact_satisfied: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence for an episode packet."""
        packet = {
            "status": self.status,
            "reason": self.reason,
            "terminal_satisfied": self.terminal_satisfied,
            "temporal_hold_satisfied": self.temporal_hold_satisfied,
            "rest_satisfied": self.rest_satisfied,
            "tool_mediation_satisfied": self.tool_mediation_satisfied,
            "object_held_satisfied": self.object_held_satisfied,
            "ejected": list(self.ejected),
            "measured_residuals": [
                dict(entry) for entry in self.measured_residuals
            ],
        }
        # Preserve legacy/default packet bytes when no terminal Contact was
        # declared or measured.
        contact_declared = any(
            entry.get("type") == "contact"
            for entry in self.measured_residuals
        )
        if self.terminal_contact_satisfied is not None or contact_declared:
            packet["terminal_contact_satisfied"] = (
                None
                if self.terminal_contact_satisfied is None
                else bool(self.terminal_contact_satisfied)
            )
        return packet


def evaluable_terminal(stage: Stage, anchors: AnchorSet) -> list[Predicate]:
    """The stage's terminal constraints that can actually be MEASURED here.

    One uniform notion of "evaluable", shared by the per-cycle terminal gate
    and the pre-motion objective check, so the two can never disagree about
    what counts. A constraint drops out when it is non-geometric (a
    NonGeometric sentinel) or when its evidence is absent from this grounding
    -- today that means a GripperState with no ``gripper_width`` anchor.
    :func:`evaluate_stage` and :func:`verify` check that dedicated measurement
    channel first and return UNKNOWN/CANNOT_VERIFY when it is absent, so
    filtering it from the scalar residual list can never authorize success.
    A non-geometric terminal is handled separately by :func:`evaluate_stage`
    as unmeasurable evidence, so it likewise cannot disappear from a mixed
    terminal while another predicate authorizes a Stage transition.
    """
    from .predicates import AtRest, GripperCommand, GripperState

    out: list[Predicate] = []
    for p in stage.terminal:
        inner = p.inner if isinstance(p, TemporalHold) else p
        if not getattr(inner, "is_geometric", False):
            continue
        if (
            isinstance(inner, GripperCommand)
            and anchors.get("gripper_command") is None
        ):
            # The acknowledged command channel is absent from this snapshot;
            # _missing_terminal_measurements keeps the stage UNKNOWN.
            continue
        if isinstance(inner, (ObjectHeld, NotHeld)):
            # Holding/release is evaluated from exact physics/measured
            # evidence below, never from the deliberate zero scalar residual.
            continue
        if isinstance(inner, (Contact, NoContact)):
            # Contact is graded from the exact executed endpoint channel in
            # evaluate_stage, never from its neutral planning residual.
            continue
        if isinstance(inner, GripperState) and anchors.get("gripper_width") is None:
            continue
        if isinstance(inner, AtRest) and anchors.missing(
            inner.velocity_anchor_names()
        ):
            # The measured velocity channel is absent from this snapshot;
            # _missing_terminal_measurements keeps the stage UNKNOWN.
            continue
        out.append(inner)
    return out


def measurement_channels(predicate: Predicate) -> tuple[str, ...]:
    """Dedicated measured-state channels a predicate reads, if any.

    These are the channels whose absence must never be reported as a
    satisfied residual.  ``Predicate.residual`` deliberately returns ``0.0``
    for a missing anchor because it is also the planning cost primitive,
    where a channel the scene does not expose must contribute nothing.  That
    neutrality is wrong for evidence: a ``0.0`` from an absent
    ``gripper_command`` is indistinguishable from a measured closed command
    (misread that way on 2026-08-07 while auditing Stage exits).  Callers
    grading or recording evidence use this to say ``unmeasured`` instead.
    """
    from .predicates import AtRest, GripperCommand, GripperState

    while isinstance(predicate, TemporalHold):
        predicate = predicate.inner
    if isinstance(predicate, GripperState):
        return ("gripper_width",)
    if isinstance(predicate, GripperCommand):
        return ("gripper_command",)
    if isinstance(predicate, AtRest):
        return tuple(predicate.velocity_anchor_names())
    if isinstance(predicate, RobotSubjectContact):
        return ("robot_subject_contact",)
    return ()


def _missing_terminal_measurements(
    stage: Stage,
    anchors: AnchorSet,
) -> tuple[str, ...]:
    """Return dedicated measured-state channels absent from this snapshot.

    Geometric anchor residuals and robot-state measurements have different
    grounding routes.  In particular, ``GripperState`` reads the measured jaw
    width rather than a scene anchor.  It may be omitted from
    :func:`evaluable_terminal` when that channel is unavailable, but the stage
    must then remain unknown -- the missing term must never disappear while
    another geometric terminal term authorizes success.
    """
    missing: list[str] = []
    for predicate in stage.terminal:
        # A Stage that declares an exit jaw command may not advance on a
        # snapshot that cannot say what the controller was told to do, and
        # the same holds for measured aperture and measured velocity.
        for name in measurement_channels(predicate):
            if anchors.missing([name]) and name not in missing:
                missing.append(name)
    return tuple(missing)


def object_held_anchors(stage: Stage) -> tuple[str, ...]:
    """Return explicit terminal held-object anchors."""
    object_anchors: list[str] = []
    for predicate in stage.terminal:
        wrapped = isinstance(predicate, TemporalHold)
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if isinstance(predicate, ObjectHeld):
            if wrapped:
                raise ValueError(
                    "ObjectHeld cannot be wrapped in TemporalHold"
                )
            object_anchors.append(predicate.object_anchor)
    if len(object_anchors) > 1:
        raise ValueError(
            "a stage may contain at most one terminal ObjectHeld"
        )
    return tuple(object_anchors)


def stage_requires_object_held(stage: Stage) -> bool:
    """Whether this stage's grading needs measured held/not-held evidence.

    True for an explicit ``ObjectHeld`` (preserve a hold) and equally for an
    explicit ``NotHeld`` (certify a release): both are graded from the same
    exact evidence channel, with opposite polarity.
    """
    for predicate in (*stage.running, *stage.terminal):
        wrapped = isinstance(predicate, TemporalHold)
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if isinstance(predicate, (ObjectHeld, NotHeld)):
            if wrapped:
                raise ValueError(
                    f"{predicate.type} cannot be wrapped in TemporalHold"
                )
            return True
    return False


def held_evidence_requirement(stage: Stage) -> tuple[str, bool] | None:
    """Return ``(object_anchor, must_be_held)`` demanded by this terminal.

    ``ObjectHeld`` demands held evidence True; ``NotHeld`` demands it False.
    A stage may declare at most one held-polarity terminal; declaring both is
    a contradiction rejected here and by program validation.
    """
    requirements: list[tuple[str, bool]] = []
    for predicate in stage.terminal:
        wrapped = isinstance(predicate, TemporalHold)
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if isinstance(predicate, (ObjectHeld, NotHeld)):
            if wrapped:
                raise ValueError(
                    f"{predicate.type} cannot be wrapped in TemporalHold"
                )
            requirements.append(
                (predicate.object_anchor, isinstance(predicate, ObjectHeld))
            )
    if not requirements:
        return None
    if len(requirements) > 1:
        raise ValueError(
            "a stage may contain at most one terminal held-polarity "
            "predicate (object_held or not_held)"
        )
    return requirements[0]


def terminal_contact_requirement(
    stage: Stage,
    anchors: AnchorSet,
) -> Contact | None:
    """Return the one native terminal subject Contact, or fail closed.

    Terminal ``NoContact`` and robot-contact variants do not yet have the
    symmetric fresh endpoint contract and therefore remain unsupported.
    """
    terms: list[Contact] = []
    for predicate in stage.terminal:
        if isinstance(predicate, TemporalHold):
            inner = predicate.inner
            if isinstance(inner, (Contact, NoContact)):
                raise ValueError(
                    f"terminal {inner.type} cannot be wrapped in TemporalHold"
                )
            continue
        if isinstance(predicate, NoContact):
            raise ValueError(
                "terminal no_contact lacks a native fresh endpoint contract"
            )
        if not isinstance(predicate, Contact):
            continue
        if "robot" in (predicate.a, predicate.b):
            raise ValueError(
                "terminal contact must relate the controlled subject to a target"
            )
        actor = anchors.get(predicate.a)
        target = anchors.get(predicate.b)
        target_owner = (
            getattr(target, "attached_to", None)
            if target is not None
            else None
        )
        if (
            anchors.get("subject") is None
            or actor is None
            or target is None
        ):
            # Missing grounding is UNKNOWN, not a guessed role violation.
            # Keep the declaration so evaluate_stage can report the absent
            # anchor/binding without treating the neutral residual as success.
            terms.append(predicate)
            continue
        actor_is_subject = (
            predicate.a == "subject"
            or getattr(actor, "attached_to", None) == "subject"
        )
        target_is_distinct = (
            predicate.b != "subject"
            and target_owner is not None
            and target_owner != "subject"
        )
        if (
            not actor_is_subject
            or not target_is_distinct
            or predicate.a == predicate.b
        ):
            raise ValueError(
                "terminal contact must be ordered as one controlled subject "
                "anchor followed by one distinct target"
            )
        terms.append(predicate)
    if len(terms) > 1:
        raise ValueError("a stage may contain at most one terminal Contact")
    return terms[0] if terms else None


def _terminal_contact_binding_grounded(
    requirement: Contact | None,
    anchors: AnchorSet,
) -> bool:
    """Whether endpoint contact can be bound to the canonical body roles."""
    if requirement is None:
        return True
    subject = anchors.get("subject")
    actor = anchors.get(requirement.a)
    target = anchors.get(requirement.b)
    if subject is None or actor is None or target is None:
        return False
    actor_is_subject = (
        requirement.a == "subject"
        or getattr(actor, "attached_to", None) == "subject"
    )
    target_owner = getattr(target, "attached_to", None)
    return bool(
        actor_is_subject
        and requirement.b != "subject"
        and target_owner is not None
        and target_owner != "subject"
        and requirement.a != requirement.b
    )


def observable_terminal(stage: Stage, anchors: AnchorSet) -> list[Predicate]:
    """The stage's terminal constraints the ROBOT CAN MOVE, exactly.

    A goal the controlled state cannot influence is unpursuable and
    ungradable: every candidate scores identically, the ranking falls through
    to the physics tiebreak, and the run optimizes something the program never
    asked for (measured on 9 of 16 audited tasks, 2026-07-24).

    The test is STRUCTURAL, not numerical: a residual reads the subject iff it
    references an anchor the outcome model actually transforms -- ``subject``
    or anything ``attached_to`` it (the same set
    :func:`~oap.loop.plan.predicted_end_anchors` moves). Sampling the
    residual's VALUE instead cannot answer this: two shipped predicates have
    UNBOUNDED zero sets (``signed_axis_gap`` is a half-space, ``min_distance``
    the outside of a ball), so an already-satisfied goal reads flat for any
    fixed probe radius -- which would refuse the very tasks that are already
    correct. Structure has no threshold, no lattice and no false positives.
    """
    moved = ({"subject"}
             | {n for n in anchors.names()
                if getattr(anchors[n], "attached_to", None) == "subject"}
             # ...and every anchor riding a NON-SUBJECT body the physics can
             # move. Restricting this to the subject was right while the twin
             # had one free body; once a tool can push a box, it refused the
             # push as inert even though the rollout moves the box and the
             # scorer now grades it.
             | {n for n in anchors.names()
                if getattr(anchors[n], "dynamic", False)})
    candidates = list(evaluable_terminal(stage, anchors))
    terminal_contact = terminal_contact_requirement(stage, anchors)
    if terminal_contact is not None:
        # It is structurally controllable through the subject even though its
        # neutral AnchorSet residual is intentionally not host-gradable.
        candidates.append(terminal_contact)
    return [
        predicate
        for predicate in candidates
        if set(predicate.referenced_anchors()) & moved
    ]


def stage_requires_tool_mediation(stage: Stage, anchors: AnchorSet) -> bool:
    """Whether this stage carries a legacy finite tool-mediation objective.

    Nothing is inferred from terminal object-motion predicates.  The declared
    tool must be the controlled subject (or an anchor rigidly attached to it);
    otherwise applying subject-contact evidence to this declaration would be
    dishonest, so planning fails closed.
    """
    return bool(_validated_tool_mediation_terms(stage, anchors))


def _contact_relation_target_refs(
    stage: Stage,
    anchors: AnchorSet,
    *,
    subject_contact_only: bool = False,
) -> tuple[str, ...]:
    """Validate composable contact relations and return their target refs."""
    refs: list[str] = []
    for predicate in stage.running:
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if not isinstance(predicate, (Contact, NoContact)):
            continue
        roles: list[tuple[str, str]] = []
        for ref in (predicate.a, predicate.b):
            if ref == "robot":
                roles.append(("robot", ref))
                continue
            anchor = anchors.get(ref)
            if anchor is None:
                raise ValueError(
                    f"{predicate.type} references ungrounded anchor {ref!r}"
                )
            owner = getattr(anchor, "attached_to", None)
            role = "subject" if ref == "subject" or owner == "subject" else "target"
            roles.append((role, ref))
        actor_indices = [
            index
            for index, (role, _ref) in enumerate(roles)
            if role in {"subject", "robot"}
        ]
        if len(actor_indices) != 1:
            raise ValueError(
                f"{predicate.type} must relate the controlled subject or robot "
                "to one distinct target"
            )
        actor_role = roles[actor_indices[0]][0]
        target_ref = roles[1 - actor_indices[0]][1]
        if roles[1 - actor_indices[0]][0] != "target":
            raise ValueError(f"{predicate.type} has no distinct target")
        if subject_contact_only and (
            actor_role != "subject" or not isinstance(predicate, Contact)
        ):
            continue
        if target_ref not in refs:
            refs.append(target_ref)
    return tuple(refs)


def stage_requires_contact_relations(stage: Stage, anchors: AnchorSet) -> bool:
    """Whether exact full-physics contact evidence enters the running cost."""
    return bool(
        _contact_relation_target_refs(stage, anchors)
        or _validated_tool_mediation_terms(stage, anchors)
    )


def contact_relation_target_bodies(
    stage: Stage,
    anchors: AnchorSet,
    dynamic_body_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve Contact/NoContact targets for exact rollout evidence."""
    refs = list(_tool_mediation_target_refs(stage, anchors))
    for ref in _contact_relation_target_refs(stage, anchors):
        if ref not in refs:
            refs.append(ref)
    resolved = _resolve_dynamic_body_refs(
        refs,
        anchors,
        dynamic_body_names,
        relation="contact relation",
    )
    if len(resolved) > 1:
        raise ValueError(
            "one stage currently requires contact relations to share one "
            f"dynamic target body, got {list(resolved)}"
        )
    return resolved


def stage_requires_tool_target_contact(
    stage: Stage,
    anchors: AnchorSet,
) -> bool:
    """Whether this stage also requests one exact tool-target contact.

    Every :class:`ToolMediation` term contributes the finite
    ``NoContact(robot,target)`` compatibility residual. This flag separates
    the optional positive ``Contact(tool,target)`` residual.
    """
    return any(
        term.require_tool_target_contact
        for term in _validated_tool_mediation_terms(stage, anchors)
    )


def _tool_mediation_terms(stage: Stage) -> tuple[ToolMediation, ...]:
    """Return explicit running declarations, unwrapping defensively.

    Synthesis validation rejects a temporal wrapper because contact duration is
    not part of this zero-threshold contract.  Unwrapping here nevertheless
    prevents a manually constructed Stage from silently bypassing the gate.
    """
    terms: list[ToolMediation] = []
    for predicate in stage.running:
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if isinstance(predicate, ToolMediation):
            terms.append(predicate)
    return tuple(terms)


def _validated_tool_mediation_terms(
    stage: Stage,
    anchors: AnchorSet,
) -> tuple[ToolMediation, ...]:
    terms = _tool_mediation_terms(stage)
    for term in terms:
        tool = anchors.get(term.tool)
        target = anchors.get(term.target)
        if tool is None:
            raise ValueError(
                f"tool_mediation references ungrounded tool {term.tool!r}"
            )
        if target is None:
            raise ValueError(
                f"tool_mediation references ungrounded target {term.target!r}"
            )
        tool_owner = getattr(tool, "attached_to", None)
        if term.tool != "subject" and tool_owner != "subject":
            raise ValueError(
                "tool_mediation.tool must be the controlled subject or an "
                f"anchor attached to it, got {term.tool!r}"
            )
        target_owner = getattr(target, "attached_to", None)
        if term.target == "subject" or target_owner == "subject":
            raise ValueError(
                "tool_mediation.target must be distinct from the controlled "
                f"subject, got {term.target!r}"
            )
    return terms


def _tool_mediation_target_refs(
    stage: Stage,
    anchors: AnchorSet,
) -> tuple[str, ...]:
    """Target-anchor refs from explicit ``tool_mediation`` running terms."""
    refs: list[str] = []
    for predicate in _validated_tool_mediation_terms(stage, anchors):
        if predicate.target not in refs:
            refs.append(predicate.target)
    return tuple(refs)


def _resolve_dynamic_body_refs(
    refs: Sequence[str],
    anchors: AnchorSet,
    dynamic_body_names: Sequence[str],
    *,
    relation: str,
) -> tuple[str, ...]:
    """Resolve anchor references to unique free-body names, or fail closed."""
    if not refs:
        return ()
    known = tuple(dict.fromkeys(str(name) for name in dynamic_body_names if name))
    resolved: set[str] = set()
    unresolved: list[str] = []
    for ref in refs:
        anchor = anchors.get(ref)
        if anchor is None:
            unresolved.append(ref)
            continue
        owner = getattr(anchor, "attached_to", None)
        # An explicit owner is authoritative. Falling back to the anchor name
        # after that owner fails would silently bind stale/malformed grounding
        # to a similarly prefixed body.
        probes = (
            (str(owner),)
            if owner and owner != "subject"
            else (str(ref),)
        )
        matches = [
            body
            for body in known
            if any(
                probe == body or probe.startswith(f"{body}_")
                for probe in probes
            )
        ]
        if matches:
            longest = max(len(body) for body in matches)
            best = [body for body in matches if len(body) == longest]
            if len(best) == 1:
                resolved.add(best[0])
                continue
        unresolved.append(ref)
    if unresolved:
        raise ValueError(
            f"{relation} target anchors do not uniquely map to dynamic "
            f"physics bodies: {sorted(unresolved)}; available={list(known)}"
        )
    return tuple(name for name in known if name in resolved)


def tool_mediation_target_bodies(
    stage: Stage,
    anchors: AnchorSet,
    dynamic_body_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve explicitly declared mediation targets to physics body names.

    ``dynamic_body_names`` comes from free joints in the compiled model.  An
    An anchor with ``attached_to`` resolves only through that authoritative
    owner; an ownerless anchor falls back to the longest exact ``body`` /
    ``body_*`` vocabulary prefix. This is the same structural convention used
    by scene/program binding: no task name, instruction keyword, geometric
    threshold, or manifest order participates.

    A declared target that cannot be mapped is unsafe to grade.  Failing
    loudly is preferable to silently aggregating every movable body, which lets
    contact with an unrelated distractor satisfy (or veto) tool mediation.
    """
    return _resolve_dynamic_body_refs(
        _tool_mediation_target_refs(stage, anchors),
        anchors,
        dynamic_body_names,
        relation="tool_mediation",
    )


def stage_contact_target_bodies(
    stage: Stage,
    anchors: AnchorSet,
    dynamic_body_names: Sequence[str],
) -> tuple[str, ...]:
    """Return bodies the controlled subject may physically contact.

    The exemption is derived only from typed relations in the current
    VLM-generated stage:

    * an explicit legacy :class:`ToolMediation` target,
    * the dynamic plane owner in ``OnPlane(subject_anchor, target_plane)``,
    * the support body in ``SupportedOn(subject, support)``.

    ``SupportedOn`` states that the subject RESTS ON the support, so that
    contact is required by the goal in exactly the way ``OnPlane``'s is.
    Omitting it (measured 2026-08-08) left every placement Stage with an
    empty contact-target set, so the eraser-on-box contact the task demands
    was classified as a forbidden other-support penetration and held to the
    4 mm hard gate. It also left ``max_subject_contact_target_pen`` reading
    zero on every candidate -- not because the channel is stable but because
    nothing ever entered it.

    ``InRegion`` alone does not imply contact, and a goal body mentioned by an
    arbitrary geometric predicate is not made collision-free. Static planes
    such as the table also remain world obstacles. A dynamic target with a
    missing or ambiguous physics owner fails closed instead of silently
    weakening collision validity.

    This set is intentionally broader than
    :func:`tool_mediation_target_bodies`: it controls only subject/world
    collision classification. Tool/robot contact diagnostics remain scoped to
    explicit ``ToolMediation`` targets.
    """
    refs = list(_tool_mediation_target_refs(stage, anchors))
    for ref in _contact_relation_target_refs(
        stage,
        anchors,
        subject_contact_only=True,
    ):
        if ref not in refs:
            refs.append(ref)
    for predicate in (*stage.running, *stage.terminal):
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if isinstance(predicate, SupportedOn):
            # Resting ON a body is a required contact, the same relation
            # OnPlane expresses against a plane. `support` is a body
            # vocabulary PREFIX rather than a point anchor, so the reference
            # handed to the resolver is the support's own top-face anchor --
            # the one the residual already reads -- and resolution stays the
            # same structural rule every other relation uses.
            ref = f"{predicate.support}_top"
            if predicate.support != "subject" and ref not in refs:
                refs.append(ref)
            continue
        if not isinstance(predicate, OnPlane):
            continue
        subject_anchor = anchors.get(predicate.a)
        plane_anchor = anchors.get(predicate.b)
        if subject_anchor is None or plane_anchor is None:
            missing = [
                ref
                for ref, anchor in (
                    (predicate.a, subject_anchor),
                    (predicate.b, plane_anchor),
                )
                if anchor is None
            ]
            raise ValueError(
                "on_plane contact relation references ungrounded anchors: "
                f"{sorted(missing)}"
            )
        subject_owned = (
            predicate.a == "subject"
            or getattr(subject_anchor, "attached_to", None) == "subject"
        )
        if not subject_owned:
            continue
        if not bool(getattr(plane_anchor, "dynamic", False)):
            continue
        owner = getattr(plane_anchor, "attached_to", None)
        if not owner or owner == "subject":
            raise ValueError(
                "on_plane dynamic contact target must have a distinct "
                f"attached_to physics body, got {predicate.b!r}"
            )
        if predicate.b not in refs:
            refs.append(predicate.b)
    return _resolve_dynamic_body_refs(
        refs,
        anchors,
        dynamic_body_names,
        relation="stage contact",
    )


def stage_terminal_now(stage: Stage, anchors: AnchorSet,
                          eps: float = 1e-6) -> Optional[bool]:
    """Grade ONE stage's terminal (sub-goal) constraints at a measured pose.

    The per-cycle termination criterion of the receding-horizon driver: the
    same hinge family the scorer's TERMINAL term optimizes, fed the MEASURED
    anchors instead of a rollout's predicted end-state. Three-valued so the
    caller can distinguish "checked and failed" from "nothing checkable":

      True  -- every geometric terminal constraint is within eps: the stage
               is DONE now (this moment is its terminal time).
      False -- at least one geometric terminal constraint is unsatisfied.
      None  -- no evaluable condition: empty terminal, only NonGeometric
               sentinels, or a referenced anchor is not finitely grounded.
               (The caller keeps its budget behavior; None is NOT a failure.)
    """
    return evaluate_stage(
        stage,
        [anchors],
        eps=eps,
        settled_snapshot=True,
        rest_satisfied=None,
        check_ejection=False,
    ).status


class VerificationOutcome(str, Enum):
    """The three honest outcomes an episode can have."""

    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    VERIFIED_FAIL = "VERIFIED_FAIL"
    CANNOT_VERIFY = "CANNOT_VERIFY"


@dataclass
class EpisodeVerdict:
    """A versioned verdict with machine-readable provenance and reason.

    ``tier`` remains serialized for the v1 log schema; current success
    verification emits only ``hard`` or a non-success refusal route.
    """

    outcome: VerificationOutcome
    tier: str                         # 'hard' | 'refused' (v1 compatibility field)
    provenance: str                   # e.g. 'measured_terminal' | 'groundability'
    reason: str = "ok"
    # For a mixed geometric/non-geometric task, report whether the measurable
    # geometric subset held while the overall result remains CANNOT_VERIFY.
    partial_geometric_success: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verdict for the episode evidence log."""
        return {"outcome": self.outcome.value, "tier": self.tier,
                "provenance": self.provenance, "reason": self.reason,
                "partial_geometric_success": self.partial_geometric_success}


def groundability_failures(program: TaskProgram, anchors: AnchorSet) -> list[str]:
    """Return scene-anchor ids that are absent or not finitely grounded.

    ``gripper_center`` and ``gripper_closing_axis`` are not perception
    anchors: planning and real execution ground them from robot FK. Their
    availability is checked by the robot/model preflight, so a scene-only
    groundability check must not report them missing.
    """
    refs = [
        name
        for name in program.referenced_anchors()
        if name not in {"gripper_center", "gripper_closing_axis"}
    ]
    return anchors.missing(refs)


# Absolute sanity floor, used when a caller cannot supply the table height.
# TABLE_TOP_Z in every scene in this project is ~0.02 m, so a body a quarter of
# a metre BELOW the origin has left the world, not been placed badly.
_ABSOLUTE_FLOOR_Z = -0.25
# How far under the table counts as ejected rather than resting. A body sitting
# on the table reads z >= table_top_z; contact softness and OBB centring cost a
# few millimetres, never a hundred.
_EJECT_MARGIN_M = 0.10


def ejection_floor_z(table_top_z: float | None = None) -> float:
    """Return the single task-independent lower world bound used by all gates."""
    return (
        _ABSOLUTE_FLOOR_Z
        if table_top_z is None
        else float(table_top_z) - _EJECT_MARGIN_M
    )


def _ejected_anchors(anchors: AnchorSet, *,
                     table_top_z: float | None = None) -> list[str]:
    """Anchors that have left the physical world, worst first.

    Not a goal check -- a check that the goal was reached in a world worth
    grading. Only finite points are considered; a non-finite anchor is already
    routed to CANNOT_VERIFY by the groundability gate above.
    """
    floor = ejection_floor_z(table_top_z)
    out: list[tuple[float, str]] = []
    for name in anchors.names():
        a = anchors.get(name)
        pt = getattr(a, "point", None)
        if pt is None:
            continue
        if getattr(a, "kind", None) == "measurement":
            # Measurement-channel anchors (e.g. the {body}_velocity vectors)
            # store non-positional quantities in .point; a 0.2 m/s descent is
            # not a body 0.2 m under the table. Only world POINTS can eject.
            continue
        z = float(pt[2])
        if z == z and abs(z) != float("inf") and z < floor:   # finite and under
            out.append((z, name))
    out.sort()
    return [f"{n}@z={z:.3f}" for z, n in out]


def evaluate_stage(
    stage: Stage,
    trajectory: Sequence[AnchorSet],
    eps: float = 1e-6,
    *,
    tau_reliable: float = 0.0,
    settled_snapshot: bool = False,
    rest_satisfied: bool | None = None,
    tool_mediation_satisfied: bool | None = None,
    object_held_satisfied: bool | None = None,
    terminal_contact_satisfied: bool | None = None,
    check_ejection: bool = True,
    table_top_z: float | None = None,
    displacement_tolerance_m: float | None = None,
) -> StageEvaluation:
    """Compose all falsifiable evidence for one stage in one deterministic order.

    A live MPC gate passes its accumulated re-observation history, so
    :class:`TemporalHold` really means N consecutive measured cycles.  Final
    verification passes one explicitly settled snapshot; there the hold reduces
    to its inner predicate, matching what that measurement can establish.

    ``rest_satisfied`` is retained as diagnostic packet metadata only. It is
    not an implicit completion condition: if a task needs a velocity or dwell
    condition, that condition must be an explicit typed terminal predicate and
    a matching GPU objective. Ejection is a hard failure. Tool-mediation
    evidence is reported alongside
    the result but does not redefine task completion: the explicit measured
    ``terminal`` predicates decide the task. Legacy ``ToolMediation`` compiles
    to finite ``NoContact(robot,target)`` and optional
    ``Contact(tool,target)`` residuals; it is not a hidden hard barrier.
    Missing predicted tool-target contact contributes one Boolean
    horizon-existence residual only when that running relation has
    ``require_tool_target_contact=true``, without a duration, penetration, or
    normal-angle threshold. This separation is necessary on hardware, where
    pose is measured but causal contact may be unobservable.

    ``terminal_contact_satisfied`` is deliberately three-valued and may only
    describe the fresh exact full-physics endpoint of the executed prefix.
    ``None`` is missing evidence, not separation; callers must never fill it
    from a horizon-existence or accumulated-contact signal.
    """
    frames = list(trajectory)
    if not frames:
        return StageEvaluation(
            status=None,
            reason="no_measurement",
            terminal_satisfied=None,
            temporal_hold_satisfied=None,
            rest_satisfied=rest_satisfied,
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=object_held_satisfied,
        )
    final = frames[-1]

    reliability_threshold = float(tau_reliable)
    if (
        not math.isfinite(reliability_threshold)
        or not 0.0 <= reliability_threshold <= 1.0
    ):
        raise ValueError("tau_reliable must be finite and in [0, 1]")

    unmeasurable: list[str] = []
    for predicate in stage.terminal:
        while isinstance(predicate, TemporalHold):
            predicate = predicate.inner
        if not getattr(predicate, "is_geometric", False):
            unmeasurable.append(
                str(
                    getattr(
                        predicate,
                        "reason",
                        getattr(predicate, "type", type(predicate).__name__),
                    )
                )
            )
    if unmeasurable:
        return StageEvaluation(
            status=None,
            reason="terminal_unmeasurable:" + ",".join(unmeasurable),
            terminal_satisfied=None,
            temporal_hold_satisfied=None,
            rest_satisfied=rest_satisfied,
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=object_held_satisfied,
        )

    missing_measurements = _missing_terminal_measurements(stage, final)
    if missing_measurements:
        # Fail closed, but say which terms could not be read. Returning with
        # an empty evidence list made an unreadable channel look like a
        # terminal that never declared the term.
        unreadable: list[dict[str, Any]] = []
        for predicate in stage.terminal:
            inner = (
                predicate.inner
                if isinstance(predicate, TemporalHold)
                else predicate
            )
            absent = [
                c for c in measurement_channels(inner) if final.missing([c])
            ]
            unreadable.append({
                "type": getattr(inner, "type", type(inner).__name__),
                "anchors": list(inner.referenced_anchors()),
                "residual": None,
                "satisfied": None,
                "measured": not absent,
                "unmeasured_channels": absent,
            })
        return StageEvaluation(
            status=None,
            reason=(
                "terminal_measurement_missing:"
                + ",".join(missing_measurements)
            ),
            terminal_satisfied=None,
            temporal_hold_satisfied=None,
            rest_satisfied=rest_satisfied,
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=object_held_satisfied,
            measured_residuals=unreadable,
        )

    evaluable_ids = {id(predicate) for predicate in evaluable_terminal(stage, final)}
    evaluable_original: list[Predicate] = []
    for predicate in stage.terminal:
        inner = predicate.inner if isinstance(predicate, TemporalHold) else predicate
        if id(inner) in evaluable_ids:
            evaluable_original.append(predicate)
    held_requirement = held_evidence_requirement(stage)
    held_anchors = (
        (held_requirement[0],) if held_requirement is not None else ()
    )
    requires_object_held = held_requirement is not None
    contact_requirement = terminal_contact_requirement(stage, final)
    requires_terminal_contact = contact_requirement is not None
    contact_relation_grounded = _terminal_contact_binding_grounded(
        contact_requirement,
        final,
    )
    effective_terminal_contact = (
        terminal_contact_satisfied if contact_relation_grounded else None
    )
    if (
        not evaluable_original
        and not requires_object_held
        and not requires_terminal_contact
    ):
        return StageEvaluation(
            status=None,
            reason="terminal_unknown",
            terminal_satisfied=None,
            temporal_hold_satisfied=None,
            rest_satisfied=rest_satisfied,
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=object_held_satisfied,
        )

    required: list[str] = []
    for predicate in evaluable_original:
        required.extend(predicate.referenced_anchors())
    required.extend(held_anchors)
    if contact_requirement is not None:
        required.extend(contact_requirement.referenced_anchors())
    if final.missing(required):
        return StageEvaluation(
            status=None,
            reason="terminal_ungroundable",
            terminal_satisfied=None,
            temporal_hold_satisfied=None,
            rest_satisfied=rest_satisfied,
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=object_held_satisfied,
        )

    invalid_pose_measurements: list[str] = []
    invalid_pose_types: list[str] = []
    for predicate in evaluable_original:
        inner = (
            predicate.inner
            if isinstance(predicate, TemporalHold)
            else predicate
        )
        from .predicates import RotationProgress
        if not isinstance(inner, (RelativeOrientation, RotationProgress)):
            continue
        check_frames = (
            frames[-max(1, int(predicate.frames)):]
            if isinstance(predicate, TemporalHold) and not settled_snapshot
            else [final]
        )
        for frame_index, anchors in enumerate(check_frames):
            for failure in inner.frame_contract_failures(anchors):
                if inner.type not in invalid_pose_types:
                    invalid_pose_types.append(inner.type)
                invalid_pose_measurements.append(
                    f"{inner.body}[{frame_index}]:{failure}"
                )
    if invalid_pose_measurements:
        return StageEvaluation(
            status=None,
            reason=(
                "terminal_measurement_invalid:" + ",".join(invalid_pose_types) + ":"
                + ";".join(invalid_pose_measurements)
            ),
            terminal_satisfied=None,
            temporal_hold_satisfied=None,
            rest_satisfied=rest_satisfied,
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=object_held_satisfied,
        )

    if reliability_threshold > 0.0:
        unreliable: list[str] = []
        for predicate in evaluable_original:
            inner = (
                predicate.inner
                if isinstance(predicate, TemporalHold)
                else predicate
            )
            names = inner.referenced_anchors()
            check_frames = (
                frames[-max(1, int(predicate.frames)):]
                if isinstance(predicate, TemporalHold)
                and not settled_snapshot
                else [final]
            )
            for anchors in check_frames:
                unreliable.extend(
                    anchors.unreliable(names, reliability_threshold)
                )
        unreliable.extend(
            final.unreliable(held_anchors, reliability_threshold)
        )
        unreliable = list(dict.fromkeys(unreliable))
        if unreliable:
            return StageEvaluation(
                status=None,
                reason="terminal_unreliable:" + ",".join(unreliable),
                terminal_satisfied=None,
                temporal_hold_satisfied=None,
                rest_satisfied=rest_satisfied,
                tool_mediation_satisfied=tool_mediation_satisfied,
                object_held_satisfied=object_held_satisfied,
            )

    temporal_results: list[bool] = []
    terminal_results: list[bool] = []
    measured_residuals: list[dict[str, Any]] = []

    def satisfied(predicate: Predicate, anchors: AnchorSet) -> bool:
        if displacement_tolerance_m not in (None, 0.0):
            raise ValueError("public paper contract uses the authored terminal threshold without displacement slack")
        return predicate.satisfied(anchors, eps)

    def record_measured(predicate: Predicate, satisfied: bool | None) -> None:
        inner = (
            predicate.inner
            if isinstance(predicate, TemporalHold)
            else predicate
        )
        channels = measurement_channels(inner)
        unmeasured = [c for c in channels if final.missing([c])]
        if unmeasured:
            # residual() returns a neutral 0.0 for an absent channel because
            # it doubles as the planning cost; recording that 0.0 here would
            # read as a satisfied measurement.
            value = None
        else:
            try:
                value = float(inner.residual(final))
            except Exception:  # noqa: BLE001 - telemetry must not veto grading
                value = None
        measured_residuals.append({
            "type": getattr(inner, "type", type(inner).__name__),
            "anchors": list(inner.referenced_anchors()),
            "residual": value,
            "satisfied": None if satisfied is None else bool(satisfied),
            "measured": not unmeasured,
            "unmeasured_channels": unmeasured,
        })
        from .predicates import RotationProgress
        if isinstance(inner, RotationProgress) and not inner.frame_contract_failures(final):
            measured_residuals[-1].update({
                "angle_deg": inner.angle_deg(final),
                "min_angle_deg": inner.min_angle_deg,
                "axis_initial_raw_body": list(inner.axis),
                "reference_raw_frame": [final.axis(n).tolist() for n in inner.initial_frame_names()],
            })

    for predicate in evaluable_original:
        if isinstance(predicate, TemporalHold):
            need = max(1, int(predicate.frames))
            if settled_snapshot:
                held = satisfied(predicate.inner, final)
            else:
                tail = frames[-need:]
                held = (
                    len(tail) >= need
                    and all(
                        not anchors.missing(predicate.inner.referenced_anchors())
                        and satisfied(predicate.inner, anchors)
                        for anchors in tail
                    )
                )
            temporal_results.append(bool(held))
            terminal_results.append(bool(held))
            record_measured(predicate, bool(held))
        else:
            satisfied_now = bool(satisfied(predicate, final))
            terminal_results.append(satisfied_now)
            record_measured(predicate, satisfied_now)

    # Terms the scalar path deliberately skips still belong in the evidence.
    # Dropping them made an unmeasured channel indistinguishable from a term
    # the Stage never declared: a Stage-2 record showed three residuals while
    # its terminal named four (2026-08-07).
    graded = {id(p) for p in evaluable_original}
    for predicate in stage.terminal:
        if id(predicate) in graded:
            continue
        inner = (
            predicate.inner if isinstance(predicate, TemporalHold) else predicate
        )
        if isinstance(inner, (ObjectHeld, NotHeld)):
            known = object_held_satisfied
            if known is not None and isinstance(inner, NotHeld):
                known = not known
            record_measured(predicate, known)
        elif isinstance(inner, Contact):
            measured_residuals.append({
                "type": inner.type,
                "anchors": list(inner.referenced_anchors()),
                "residual": (
                    None
                    if effective_terminal_contact is None
                    else float(not effective_terminal_contact)
                ),
                "satisfied": (
                    None
                    if effective_terminal_contact is None
                    else bool(effective_terminal_contact)
                ),
                "measured": effective_terminal_contact is not None,
                "unmeasured_channels": (
                    []
                    if effective_terminal_contact is not None
                    else [
                        (
                            "subject_target_contact_endpoint"
                            if contact_relation_grounded
                            else "terminal_contact_relation_binding"
                        )
                    ]
                ),
            })
        else:
            record_measured(predicate, None)

    scalar_terminal_ok = (
        all(terminal_results) if terminal_results else None
    )
    terminal_ok = scalar_terminal_ok
    if requires_terminal_contact:
        if terminal_ok is False or effective_terminal_contact is False:
            terminal_ok = False
        elif effective_terminal_contact is None:
            terminal_ok = None
        else:
            # Exact endpoint contact is sufficient for the contact-only case
            # and conjunctive with every scalar terminal predicate otherwise.
            terminal_ok = True
    temporal_ok = (
        all(temporal_results) if temporal_results else None)
    # Preserve ejection evidence even while the goal is still unsatisfied.
    # The stage controller can stop immediately on this irreversible physical
    # failure instead of executing more prefixes.
    ejected = tuple(
        _ejected_anchors(final, table_top_z=table_top_z)
        if check_ejection else ())
    if scalar_terminal_ok is False:
        status = False
        reason = (
            "temporal_hold_pending"
            if temporal_results and not all(temporal_results)
            else "terminal_unsatisfied")
    elif (
        held_requirement is not None
        and object_held_satisfied is not None
        and bool(object_held_satisfied) != held_requirement[1]
    ):
        status, reason = False, (
            "object_not_held"
            if held_requirement[1]
            else "object_still_held"
        )
    elif requires_terminal_contact and not contact_relation_grounded:
        status, reason = None, "terminal_contact_relation_unknown"
    elif requires_terminal_contact and effective_terminal_contact is False:
        status, reason = False, "terminal_contact_unsatisfied"
    elif requires_object_held and object_held_satisfied is None:
        status, reason = None, "object_held_evidence_unknown"
    elif requires_terminal_contact and effective_terminal_contact is None:
        status, reason = None, "terminal_contact_evidence_unknown"
    elif ejected:
        status, reason = False, f"ejected:{','.join(ejected)}"
    else:
        status, reason = True, "terminal_satisfied"

    return StageEvaluation(
        status=status,
        reason=reason,
        terminal_satisfied=terminal_ok,
        temporal_hold_satisfied=temporal_ok,
        rest_satisfied=rest_satisfied,
        tool_mediation_satisfied=tool_mediation_satisfied,
        object_held_satisfied=object_held_satisfied,
        terminal_contact_satisfied=(
            effective_terminal_contact if requires_terminal_contact else None
        ),
        ejected=ejected,
        measured_residuals=tuple(measured_residuals),
    )


def verify(program: TaskProgram, real_anchors: AnchorSet,
           eps: float = 1e-6, *,
           tau_reliable: float = 0.0,
           observability: dict[str, float] | None = None,
           min_observability: float = 0.0,
           table_top_z: float | None = None,
           tool_mediation_satisfied: bool | None = None,
           displacement_tolerance_m: float | None = None,
           stage_evidence: StageEvaluation | None = None) -> EpisodeVerdict:
    """The single, task-agnostic verification router.

    Analytic observation-integrity gates precede the geometric check and route
    unobservable conditions to CANNOT_VERIFY with a distinct reason.
    Defaults (tau_reliable=0,
    observability=None) reproduce the legacy finite-only groundability behavior
    exactly.

    ``tool_mediation_satisfied`` is retained for packet compatibility and is
    carried into :class:`StageEvaluation` as a running-constraint diagnostic.
    It never redefines or vetoes the explicit measured terminal.
    """
    final_stage = program.stages[-1]
    required: list[str] = []
    for predicate in final_stage.terminal:
        required.extend(predicate.referenced_anchors())
    required = list(dict.fromkeys(required))
    # 0. OBSERVABILITY (analytic visible-fraction) -- a heavily-occluded anchor
    #    with a finite-but-hallucinated point must refuse with reason=occluded.
    if observability is not None and min_observability > 0.0:
        occluded = [n for n in required
                    if observability.get(n, 1.0) < min_observability]
        if occluded:
            return EpisodeVerdict(VerificationOutcome.CANNOT_VERIFY, "refused", "observability",
                                  reason=f"occluded:{','.join(occluded)}")

    # 1. Groundability: referenced but absent OR non-finite -> never localized.
    missing = real_anchors.missing(required)
    if missing:
        return EpisodeVerdict(VerificationOutcome.CANNOT_VERIFY, "refused", "groundability",
                              reason=f"ungroundable:{','.join(missing)}")

    missing_measurements = _missing_terminal_measurements(
        final_stage,
        real_anchors,
    )
    if missing_measurements:
        return EpisodeVerdict(
            VerificationOutcome.CANNOT_VERIFY,
            "refused",
            "terminal_evidence",
            reason=(
                "terminal_measurement_missing:"
                + ",".join(missing_measurements)
            ),
        )

    # 2. Reliability: localized but tracked below the confidence floor -> lost
    #    track (distinct from ungroundable: it WAS seen, then drifted/occluded).
    if tau_reliable > 0.0:
        lost = [n for n in real_anchors.unreliable(required, tau_reliable) if n not in missing]
        if lost:
            return EpisodeVerdict(VerificationOutcome.CANNOT_VERIFY, "refused", "lost_track",
                                  reason=f"lost_track:{','.join(lost)}")

    # 2/3. Hard, falsifiable verification when success is fully geometric.
    if program.is_fully_geometric():
        # TERMINAL semantics: the task's goal condition is the FINAL stage's
        # terminal. Earlier stages' terminal conditions are sequencing milestones --
        # transient by design ("gripper closed", "lifted 15 cm") and verified
        # when their stage advanced. Conjoining them here made a finished
        # place unverifiable live: a subject placed INSIDE the box can never
        # again satisfy the lift stage's height band, so the terminal verdict
        # stayed VERIFIED_FAIL even though separate visual evidence appeared
        # successful. Only the final stage defines the current task terminal.
        evaluation = evaluate_stage(
            program.stages[-1],
            [real_anchors],
            eps=eps,
            settled_snapshot=True,
            displacement_tolerance_m=displacement_tolerance_m,
            rest_satisfied=(
                stage_evidence.rest_satisfied if stage_evidence else None),
            tool_mediation_satisfied=tool_mediation_satisfied,
            object_held_satisfied=(
                stage_evidence.object_held_satisfied
                if stage_evidence else None),
            terminal_contact_satisfied=(
                stage_evidence.terminal_contact_satisfied
                if stage_evidence else None),
            table_top_z=table_top_z,
        )
        if evaluation.status is None:
            return EpisodeVerdict(
                VerificationOutcome.CANNOT_VERIFY,
                "refused",
                "terminal_evidence",
                reason=evaluation.reason,
            )
        ok = evaluation.status
        if not ok:
            return EpisodeVerdict(
                VerificationOutcome.VERIFIED_FAIL,
                "hard",
                "measured_terminal",
                reason=evaluation.reason,
            )
        return EpisodeVerdict(
            VerificationOutcome.VERIFIED_SUCCESS,
            "hard", "measured_terminal",
            reason="final_stage_terminal_satisfied")

    # 4. Success is not geometrically expressible -- REFUSE, and report whether
    # the geometric SUBSET (the falsifiable proxy) was achieved.
    #
    # There is no second verifier: unsupported terminal state is refused here
    # rather than delegated to a soft visual opinion.
    geom_proxy_ok = _geometric_proxy_satisfied(program, real_anchors, eps)
    reasons = program.non_geometric_reasons() or ["non_geometric_success"]
    return EpisodeVerdict(VerificationOutcome.CANNOT_VERIFY, "refused", "unsupported_terminal",
                          reason=f"non_geometric:{','.join(reasons)}",
                          partial_geometric_success=geom_proxy_ok)


def _geometric_proxy_satisfied(program: TaskProgram, anchors: AnchorSet, eps: float) -> bool:
    """Are the FINAL stage's GEOMETRIC terminal constraints satisfied
    (ignoring the non-geometric ones)? This is the falsifiable pose/geometry
    proxy reported alongside a CANNOT_VERIFY on the true (non-geometric)
    outcome -- terminal semantics as in :func:`verify` (earlier stages are
    transient sequencing milestones, not terminal conditions)."""
    for p in program.stages[-1].terminal:
        if p.is_geometric and not p.satisfied(anchors, eps):
            return False
    return True
